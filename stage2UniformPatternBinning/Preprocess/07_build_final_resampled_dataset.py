import os
import json
import orjson
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# =====================================================
# 1) 路径映射：支持可配置 split_token
#    旧版本可能硬编码某个中间输出目录名：
#    parts = base.split("/<CODEBOOK_OUTPUT_ROOT_NAME>/")
# =====================================================

def remap_codebook_path_to_npy(codebook_path: str, new_dataset_root: str, *, split_token: str) -> str:
    base = codebook_path.replace("_codebook", "")
    parts = base.split(split_token)
    if len(parts) < 2:
        raise ValueError(f"split_token not found in path: token={split_token}, path={base}")
    rel = parts[1].lstrip("/")

    npy_rel = rel.replace(".npz", ".npy")
    return os.path.join(new_dataset_root, npy_rel)


# =====================================================
# 2) 读取 valid_indices（沿用你旧逻辑） :contentReference[oaicite:5]{index=5}
#    这里保持结构兼容：返回 valid_indices[n][c] = [start,end]
# =====================================================

def load_valid_indices_from_metadata(data_path: str):
    def load_indices(desc_path: str, mode):
        with open(desc_path, "r") as f:
            desc = json.load(f)
        if mode is None:
            return desc["valid_indices"]
        elif mode == "_past_feat_dynamic_real":
            return desc["past_feat_dynamic_real_valid_indices"]
        elif mode == "_feat_dynamic_real":
            return desc["feat_dynamic_real_valid_indices"]
        else:
            raise NotImplementedError(mode)

    dataset_part_name = os.path.basename(data_path).split(".npy")[0]

    if os.path.exists(data_path.replace(".npy", ".json")):
        return load_indices(data_path.replace(".npy", ".json"), None)

    if "_past_feat_dynamic_real" in dataset_part_name:
        desc_name = dataset_part_name.replace("_past_feat_dynamic_real", "") + ".json"
        return load_indices(os.path.join(os.path.dirname(data_path), desc_name), "_past_feat_dynamic_real")

    if "_feat_dynamic_real" in dataset_part_name:
        desc_name = dataset_part_name.replace("_feat_dynamic_real", "") + ".json"
        return load_indices(os.path.join(os.path.dirname(data_path), desc_name), "_feat_dynamic_real")

    if "_values" in dataset_part_name:
        desc_name = dataset_part_name.split("_values")[0] + ".json"
        return load_indices(os.path.join(os.path.dirname(data_path), desc_name), None)

    if "_data" in dataset_part_name:
        desc_name = dataset_part_name.split("_data")[0] + ".json"
        return load_indices(os.path.join(os.path.dirname(data_path), desc_name), None)

    raise RuntimeError(f"cannot infer desc json for: {data_path}")


# =====================================================
# 3) 可选：归一化（沿用你旧 normalize 的思路） :contentReference[oaicite:6]{index=6}
#    你现在要 zero padding，所以我们：
#    - 先对真实片段做 z-score（忽略 NaN）
#    - 再把 NaN 转 0（便于后续 merge）
# =====================================================




# =====================================================
# 4) 交错合成（你描述的核心逻辑）
#    group_seqs: list[np.ndarray], 每个是 1D (len<=4096, float32)
#    out: zeros(4096), 填充规则：
#      out[0]=s0[0], out[1]=s1[0], (out[2]=s2[0], out[3]=s3[0]), out[4]=s0[1], ...
# =====================================================

def interleave_group_sequences(group_seqs, out_len=4096):
    k = len(group_seqs)
    per_len = out_len // k

    out = np.zeros(out_len, dtype=np.float32)
    for i, s in enumerate(group_seqs):
        L = min(len(s), per_len)
        out[i : i + k * L : k] = s[:L]

    return out



# =====================================================
# 5) NDJSON + offset：随机读取某个 index 的 results
#    ndjson line format: {"index":"123","results":[...]}
# =====================================================

def load_sample_results_for_cell(ndjson_path: str, offset_map: dict, index_id: int):
    off = offset_map[str(index_id)]
    with open(ndjson_path, "rb") as f:
        f.seek(off)
        line = f.readline()
    obj = orjson.loads(line)
    return obj["results"]



def zscore_sequence_ignore_nan(seq: np.ndarray) -> np.ndarray:
    seq = seq.astype(np.float32, copy=True)

    mask = ~np.isnan(seq)
    if mask.any():
        vals = seq[mask]
        mean = vals.mean()
        std = vals.std()
        if std < 1e-6:
            std = 1.0
        seq[mask] = (vals - mean) / std

    return seq


# =====================================================
# 6) worker：处理单个 index -> 写临时 dat + meta
#    meta 记录每条输出序列来源，确保 dat 可追溯
# =====================================================
def build_sequences_for_single_cell(args):
    (
        index_id,
        ndjson_path,
        offset_map,
        registered_index,
        new_dataset_root,
        split_token,
        out_len,
        tmp_dir,
        do_normalize,
    ) = args

    results = load_sample_results_for_cell(ndjson_path, offset_map, index_id)

    do_progress = (index_id % 100 == 0)
    pbar = None
    if do_progress:
        pbar = tqdm(
            total=len(results),
            desc=f"[PID {os.getpid()}] index {index_id}",
            position=index_id % 8,
            leave=False,
            ncols=80,
        )

    npy_cache = {}
    valid_cache = {}

    tmp_dat = os.path.join(tmp_dir, f"index_{index_id:05d}.dat")
    tmp_meta = os.path.join(tmp_dir, f"index_{index_id:05d}.meta.ndjson")

    CHUNK_SIZE = 2048
    buf_data = []
    buf_meta = []

    rng = np.random.default_rng(index_id)

    with open(tmp_dat, "wb") as fdat, open(tmp_meta, "wb") as fmeta:
        for j, group in enumerate(results):
            group_size = len(group)
            per_seq_len = out_len // group_size

            group_seqs = []
            group_meta_items = []

            for (file_id, n, c, level, valid_len) in group:
                file_id = int(file_id)
                n = int(n)
                c = int(c)
                level = int(level)

                codebook_path = registered_index[file_id]
                npy_path = remap_codebook_path_to_npy(
                    codebook_path,
                    new_dataset_root,
                    split_token=split_token,
                )

                if npy_path not in npy_cache:
                    npy_cache[npy_path] = np.load(npy_path, mmap_mode="r")
                if npy_path not in valid_cache:
                    valid_cache[npy_path] = load_valid_indices_from_metadata(npy_path)

                arr = npy_cache[npy_path]
                valid_indices = valid_cache[npy_path]

                start, end = valid_indices[n][c]
                start = int(start)
                end = int(end)

                raw_len = end - start

                # -------------------------------
                # 随机裁剪窗口（新增 crop_start）
                # -------------------------------
                if raw_len > per_seq_len:
                    max_offset = raw_len - per_seq_len
                    offset = rng.integers(0, max_offset + 1)
                    crop_start = start + offset
                    seq = arr[n, c, crop_start : crop_start + per_seq_len]
                    crop_len = per_seq_len
                else:
                    crop_start = start
                    crop_len = raw_len
                    seq = arr[n, c, start:end]

                seq = np.asarray(seq, dtype=np.float32)

                if do_normalize:
                    seq = zscore_sequence_ignore_nan(seq)
                else:
                    seq = np.nan_to_num(seq, nan=0.0).astype(np.float32, copy=False)

                group_seqs.append(seq)
                group_meta_items.append({
                    "file_id": file_id,
                    "n": n,
                    "c": c,
                    "level": level,
                    "start": start,           # 原始 valid start
                    "end": end,               # 原始 valid end
                    "raw_len": int(raw_len),  # 原始长度
                    "crop_start": int(crop_start),  # ✅ 实际截取起点
                    "crop_len": int(crop_len),      # ✅ 实际使用长度
                    "npy_path": npy_path,
                })

            if pbar is not None:
                pbar.update(1)

            if group_size == 1:
                # out = np.zeros(out_len, dtype=np.float32) old 2025.12.31的时候是填充zeros,但是现在这个东西不满足4096长度。
                out = np.full(out_len, np.nan, dtype=np.float32)
                L = min(out_len, len(group_seqs[0]))
                out[:L] = group_seqs[0][:L]
            else:
                out = interleave_group_sequences(group_seqs, out_len=out_len)
                out = zscore_sequence_ignore_nan(out)

            buf_data.append(out)
            buf_meta.append({
                "index_id": int(index_id),
                "pos_in_index": int(j),
                "group_size": int(group_size),
                "per_seq_len": int(per_seq_len),
                "items": group_meta_items,
            })

            if len(buf_data) >= CHUNK_SIZE:
                np.stack(buf_data, axis=0).tofile(fdat)
                buf_data.clear()

                for rec in buf_meta:
                    fmeta.write(orjson.dumps(rec))
                    fmeta.write(b"\n")
                buf_meta.clear()

        if buf_data:
            np.stack(buf_data, axis=0).tofile(fdat)
            for rec in buf_meta:
                fmeta.write(orjson.dumps(rec))
                fmeta.write(b"\n")

        if pbar is not None:
            pbar.close()

    return index_id



# =====================================================
# 7) 主控：并行处理 indices -> 生成临时文件 -> 按 index 顺序拼接为最终 dat
# =====================================================

import os
import json
import orjson
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def build_final_dataset_from_indices(
    *,
    ndjson_path: str,
    offset_path: str,
    registered_index_json: str,
    new_dataset_root: str,
    split_token: str,
    out_dat_path: str,
    out_shape_path: str,
    out_meta_path: str,
    tmp_dir: str,
    index_ids: list,
    workers: int = 16,
    out_len: int = 4096,
    do_normalize: bool = True,
    current_number: int = 8000,
):
    os.makedirs(tmp_dir, exist_ok=True)

    # load offset map
    with open(offset_path, "rb") as f:
        offset_map = orjson.loads(f.read())

    # load registered_index: file_id -> codebook_path
    with open(registered_index_json, "r", encoding="utf-8") as f:
        registered_index = {int(k): v for k, v in json.load(f).items()}

    # sanity: all index ids exist
    for idx in index_ids:
        if str(idx) not in offset_map:
            raise KeyError(f"index_id not found in offset_map: {idx}")

    tasks = [
        (
            int(idx),
            ndjson_path,
            offset_map,
            registered_index,
            new_dataset_root,
            split_token,
            out_len,
            tmp_dir,
            do_normalize,
        )
        for idx in index_ids
    ]

    with Pool(processes=workers) as pool:
        list(tqdm(
            pool.imap_unordered(build_sequences_for_single_cell, tasks),
            total=len(tasks),
            desc="Processing indices"
        ))

    # 拼接：严格按 index 顺序写入最终 dat + meta
    if os.path.exists(out_dat_path):
        os.remove(out_dat_path)
    if os.path.exists(out_meta_path):
        os.remove(out_meta_path)

    total_rows = 0
    with open(out_dat_path, "ab") as fdat, open(out_meta_path, "ab") as fmeta:
        for idx in tqdm(sorted(index_ids), desc="Concatenating", ncols=100):
            tmp_dat = os.path.join(tmp_dir, f"index_{int(idx):05d}.dat")
            tmp_meta = os.path.join(tmp_dir, f"index_{int(idx):05d}.meta.ndjson")

            # dat
            with open(tmp_dat, "rb") as fr:
                buf = fr.read()
            fdat.write(buf)

            # meta
            with open(tmp_meta, "rb") as fr:
                fmeta.write(fr.read())

            # rows: 每个 index 固定 5000 条（你说的）
            total_rows += current_number

    # shape.npy: [num_samples, out_len]
    np.save(out_shape_path, np.array([total_rows, out_len], dtype=np.int64))
    print(f"✔ Done: dat={out_dat_path}, meta={out_meta_path}, shape={out_shape_path}")


if __name__ == "__main__":
    # ============ 你需要改的配置 ============

    NDJSON_PATH = "cell_usage_sampled2re.ndjson"
    OFFSET_PATH = "cell_usage_sampled2re.offset.json"
    REGISTERED_INDEX_JSON = "registered_codebook_index.json"

    NEW_DATASET_ROOT = "<RAW_DATA_ROOT>"
    SPLIT_TOKEN = "/<CODEBOOK_OUTPUT_ROOT_NAME>/"   # 你要可调的 split 符号

    OUT_DAT = "sequences_by_indexre.dat"
    OUT_SHAPE = "sequences_by_indexre_shape.npy"
    OUT_META = "sequences_by_indexre.meta.ndjson"
    TMP_DIR = "./tmp_index_datre"

    WORKERS = 98
    group_size= 8
    OUT_LEN = 4096
    DO_NORMALIZE = True
    current_number = 8000

    # 你可以按需要只跑部分 index
    INDEX_IDS = list(range(0, 2499))  # or [123,124,...]

    build_final_dataset_from_indices(
        ndjson_path=NDJSON_PATH,
        offset_path=OFFSET_PATH,
        registered_index_json=REGISTERED_INDEX_JSON,
        new_dataset_root=NEW_DATASET_ROOT,
        split_token=SPLIT_TOKEN,
        out_dat_path=OUT_DAT,
        out_shape_path=OUT_SHAPE,
        out_meta_path=OUT_META,
        tmp_dir=TMP_DIR,
        index_ids=INDEX_IDS,
        workers=WORKERS,
        out_len=OUT_LEN,
        do_normalize=DO_NORMALIZE,
        current_number = current_number
    )
