import numpy as np
import os
import json
import torch
import sys
sys.path.append(os.environ.get("SWIN_REPO_ROOT", "<SWIN_REPO_ROOT>"))
from swin.arch import Swin
from swin.config.transformervqvae_config import transformer_vqvae_mini_config3
from setproctitle import setproctitle

def load_valid_indices_from_metadata(data_path):
    def load_indices(desc_path, mode):
        with open(desc_path, "r") as f:
            desc = json.load(f)
        if mode is None:
            return desc["valid_indices"]
        elif '_past_feat_dynamic_real' == mode:
            return desc['past_feat_dynamic_real_valid_indices']
        elif '_feat_dynamic_real' == mode:
            return desc['feat_dynamic_real_valid_indices']
        else:
            raise NotImplementedError

    dataset_part_name = data_path.split('/')[-1].split(".npy")[0] # linux
    if os.path.exists(data_path.replace(".npy", ".json")):
        # 一般来说，json的name等于npy的name
        desc_path = data_path.replace(".npy", ".json")
        return load_indices(desc_path, None)
    elif '_past_feat_dynamic_real' in dataset_part_name: # LOTSA的past_feat_dynamic_real读取
        desc_name = dataset_part_name.split('_past_feat_dynamic_real')[0] + dataset_part_name.split('_past_feat_dynamic_real')[1] + ".json"
        # 获取data_path上一级目录
        desc_path = os.path.join(os.path.dirname(data_path), desc_name)
        return load_indices(desc_path, '_past_feat_dynamic_real')
    elif '_feat_dynamic_real' in dataset_part_name: # LOTSA的'_feat_dynamic_real'读取
        desc_name = dataset_part_name.split('_feat_dynamic_real')[0] + dataset_part_name.split('_past_feat_dynamic_real')[1] + ".json"
        # 获取data_path上一级目录
        desc_path = os.path.join(os.path.dirname(data_path), desc_name)
        return load_indices(desc_path, '_feat_dynamic_real')
    elif '_values' in dataset_part_name:
        desc_name = dataset_part_name.split('_values')[0] + ".json"
        desc_path = os.path.join(os.path.dirname(data_path), desc_name)
        return load_indices(desc_path, None)
    elif '_data' in dataset_part_name: # UCR和UAD
        desc_name = dataset_part_name.split('_data')[0] + ".json"
        desc_path = os.path.join(os.path.dirname(data_path), desc_name)
        return load_indices(desc_path, None)
    else:
        assert False, "dataset_part_name: {}".format(dataset_part_name)


def slice_valid_segments_and_normalize(data, valid_indices):
    """
    输入:
        data: N, C, L numpy array
        valid_indices: list[list[[start, end]]] 结构，与 data 对应

    输出:
        processed: N, C, L2，L2 = 所有有效序列长度的最大值
    """

    N, C, L = data.shape

    # -----------------------------
    # 1. 找到每个 (n, c) 的有效长度
    # -----------------------------
    valid_lengths = np.zeros((N, C), dtype=int)

    for n in range(N):
        for c in range(C):
            start, end = valid_indices[n][c]
            valid_lengths[n, c] = end - start

    L2 = valid_lengths.max()   # 所有 (n,c) 中最长的有效长度
    # print("Computed L2 =", L2)

    # -----------------------------
    # 2. 初始化输出数组为 NaN
    # -----------------------------
    processed = np.full((N, C, L2), np.nan, dtype=np.float32)

    # -----------------------------
    # 3. 填充有效序列并归一化
    # -----------------------------
    for n in range(N):
        for c in range(C):

            start, end = valid_indices[n][c]
            seq = data[n, c, start:end]     # 截取有效序列

            # -------- 去掉 seq 内部的 NaN（理论上不应有，但加稳健性）--------
            valid = seq[~np.isnan(seq)]

            if len(valid) == 0:
                # 若无有效值，保持全 NaN
                continue

            # -------- mean / std 归一化 --------
            mean = valid.mean()
            std = valid.std()
            if std < 1e-4:
                std = 1.0

            normed = (seq - mean) / std

            # -------- 写入 processed --------
            length = len(normed)
            processed[n, c, :length] = normed
    
    return processed


# ============================================================
# ★ resizing function：统一处理成 4096 长度
# ============================================================

def resize_single_channel_to_fixed_length(seq, valid_length, target_len=4096, mode="truncate"):

    # 有效部分
    seq_valid = seq[:valid_length]
    L = valid_length

    # ------------------------------- pad -------------------------------
    if L < target_len:
        out = np.full((target_len,), np.nan, dtype=np.float32)
        out[:L] = seq_valid
        return out

    # ------------------------------- match -------------------------------
    if L == target_len:
        return seq_valid.astype(np.float32)

    # ------------------------------- truncate -------------------------------
    if mode == "truncate":
        return seq_valid[:target_len].astype(np.float32)

    # ------------------------------- interval -------------------------------
    if mode == "interval":
        intervals = [1, 2, 4, 8, 16, 32]

        candidate = L // target_len  # floor(L / 4096)

        possible = [i for i in intervals if i <= candidate]

        if not possible:
            return seq_valid[:target_len].astype(np.float32)

        interval = max(possible)

        idx = np.arange(0, interval * target_len, interval)
        return seq_valid[idx].astype(np.float32)

    raise ValueError(f"Invalid mode: {mode}")


# ============================================================
# ★ 生成(N,C,4096) resized 完整输入
# ============================================================

def build_resized_model_inputs(processed, valid_indices, resize_mode="interval"):
    N, C, L2 = processed.shape
    out = np.zeros((N, C, 4096), dtype=np.float32)

    for n in range(N):
        for c in range(C):
            start, end = valid_indices[n][c]
            valid_length = end - start

            seq = processed[n, c]  # 含 NaN padding

            out[n, c] = resize_single_channel_to_fixed_length(
                seq,
                valid_length,
                target_len=4096,
                mode=resize_mode
            )

    return out  # (N,C,4096)


# ============================================================
# ★ 主函数：最终整合版 export_patch_features_with_vqvae
# ============================================================

def export_patch_features_with_vqvae(
        processed,           # (N,C,L2)
        valid_indices,       # 原始 json 读取的 start/end
        model, batch_size,
        device,
        save_dir, name,
        resize_mode="interval"
    ):
    """
    最终整合版：
    - 按每个 (N,C) 自适应采样 → 4096
    - 批推理 VQVAE
    - mask=False 的 codebook id → nan
    - reshape 回 (N,C,T)
    - 保存为压缩 npz
    """

    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # 1) 生成 (N,C,4096) 输入
    # -----------------------------
    # print("🔧 Generating per-channel resized inputs...")
    resized = build_resized_model_inputs(processed, valid_indices, resize_mode)
    N, C, _ = resized.shape

    # flatten for batch inference
    flat = resized.reshape(N*C, 4096)
    num_series = flat.shape[0]

    ids_list = []
    mask_list = []
    mse_list = []
    mae_list = []

    model.to(device)
    model.eval()

    # -----------------------------
    # 2) Batch inference（无梯度）
    # -----------------------------
    # print("🚀 Running VQ-VAE inference...")

    for start in range(0, num_series, batch_size):
        end = min(start + batch_size, num_series)

        batch_np = flat[start:end]                      # shape = (B, 4096)
        batch_t = torch.tensor(batch_np, dtype=torch.float32).to(device)

        original_B = batch_t.shape[0]                  # 记录原始 batch size

        # ----------★ 如果 B=1，则 repeat 成 B=3 ----------
        need_slice_back = False
        if original_B == 1:
            batch_t = batch_t.repeat(4, 1)             # (1,4096) → (3,4096)
            need_slice_back = True

        # print(batch_t.shape, 'xxxx')

        with torch.inference_mode():
            codebook_ids, codebook_mask, loss, _, _, _ = model.inference(
                batch_t,
                return_reconstruction=True
            )
            

        # ----------★ 如果重复过，则只取第 0 条样本 ----------
        if need_slice_back:
            codebook_ids  = codebook_ids[0:1]          # (1, T)
            codebook_mask = codebook_mask[0:1]
            # loss['recons_batch'] 是 (3,) → 只取第0个
            loss_mse, loss_mae = loss["recons_batch"]
            
            loss_mse = loss_mse[0:1]
            loss_mae = loss_mae[0:1]
            loss["recons_batch"] = (loss_mse, loss_mae)

        ids_np = codebook_ids.cpu().numpy()               # (B,T)
        mask_np = codebook_mask.cpu().numpy().astype(bool)

        ids_list.append(ids_np)
        mask_list.append(mask_np)

        batch_mse, batch_mae = loss["recons_batch"]
        # print(batch_mse)
        mse_list.append(batch_mse.cpu().numpy())
        mae_list.append(batch_mae.cpu().numpy())

        if start > 0 and start % (batch_size * 200) == 0:
            print(f"  → Processed {start}/{num_series}")

    # -----------------------------
    # 3) 拼接
    # -----------------------------
    ids_all = np.concatenate(ids_list, axis=0)      # (N*C,T)
    mask_all = np.concatenate(mask_list, axis=0)    # (N*C,T)
    mse_all = np.concatenate(mse_list, axis=0)      # (N*C,)
    mae_all = np.concatenate(mae_list, axis=0)

    # -----------------------------
    # 4) 施加 mask → 无效位置置为 nan
    # -----------------------------
    masked_ids = np.where(mask_all, ids_all, np.nan).astype(np.float32)

    # -----------------------------
    # 5) reshape 回 (N,C,T)
    # -----------------------------
    T = masked_ids.shape[1]
    masked_ids_final = masked_ids.reshape(N, C, T)
    mse_final = mse_all.reshape(N, C)
    mae_final = mae_all.reshape(N, C)

    # -----------------------------
    # 6) 保存
    # -----------------------------
    # print("💾 Saving outputs...")
    # print(name)

    np.savez_compressed(
        os.path.join(save_dir, f"{name}_codebook.npz"),
        codebook_ids=masked_ids_final
    )

    np.savez_compressed(
        os.path.join(save_dir, f"{name}_maemse.npz"),
        mse=mse_final,
        mae=mae_final
    )

    # print(f"✔ Saved codebook → {name}_codebook.npz")
    # print(f"✔ Saved loss     → {name}_maemse.npz")

    return masked_ids_final, mse_final, mae_final


from tqdm import tqdm
import copy


def iter_dataset_npy_files(root):
    npy_files = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".npy"):
                npy_files.append(os.path.join(dirpath, fn))
    return npy_files

def group_dataset_files_by_parent(path, root):
    """
    例如: root/Monash/M3_Other_Dataset/yahoo.npy
    rel = Monash/M3_Other_Dataset/yahoo.npy
    parts = ['Monash', 'M3_Other_Dataset', 'yahoo.npy']
    key = 父目录名 = 'M3_Other_Dataset'
    """
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    else:
        return parts[0]

from tqdm.auto import tqdm



def load_base_model_once(ckpt_path, config):
    model = Swin(config=config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 如果你想 compile，只能在主进程地编译
    # 在 worker 内部不能再 compile，否则冻结
    # model = torch.compile(model)

    model.eval()
    return model

import copy

def replicate_model_to_devices(base_model, device_list):
    models = {}
    for dev in device_list:
        m = copy.deepcopy(base_model)      # 深拷贝模型结构+参数
        m.to(torch.device(dev))
        models[dev] = m
    return models


def run_dataset_worker(
        gpu_idx,
        device,
        model,                 # ★ 直接传入模型对象
        config,
        task_queue,
        data_root,
        save_root,
        batch_size,
        resize_mode
    ):
    setproctitle(f"BLAST_GPU{gpu_idx}")

    import torch
    from tqdm.auto import tqdm

    torch.cuda.set_device(device)
    model.to(device)
    model.eval()

    print(f"[GPU {device}] Worker started.")

    while True:
        try:
            key, file_list = task_queue.get(timeout=5)
        except:
            print(f"[GPU {device}] Done.")
            break

        pbar = tqdm(
            file_list,
            desc=f"[GPU {device}] {key}",
            position=gpu_idx,
            leave=True
        )

        for fpath in pbar:
            try:
                data = np.load(fpath)
            except:
                continue

            if data.ndim != 3:
                continue

            try:
                valid_indices = load_valid_indices_from_metadata(fpath)
            except:
                continue

            try:
                processed = slice_valid_segments_and_normalize(data, valid_indices)
            except:
                continue

            rel_dir = os.path.relpath(os.path.dirname(fpath), data_root)
            save_dir = os.path.join(save_root, rel_dir)
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(fpath))[0]

            try:
                export_patch_features_with_vqvae(
                    processed=processed,
                    valid_indices=valid_indices,
                    model=model,
                    batch_size=batch_size,
                    device=device,
                    save_dir=save_dir,
                    name=base_name,
                    resize_mode=resize_mode
                )
            except Exception as e:
                print(f"[GPU {device}] Error: {e}")


import multiprocessing as mp

def process_all_datasets_in_parallel(
        data_root="<RAW_DATA_ROOT>",
        save_root="<CODEBOOK_OUTPUT_ROOT>",
        base_model=None,             # ★ 已加载好权重的模型
        config=None,
        batch_size=32,
        device_list=("cuda:0","cuda:1","cuda:2","cuda:3", "cuda:4","cuda:5","cuda:6","cuda:7"),
        resize_mode="interval"
    ):

    assert base_model is not None

    # 收集所有 npy
    all_npy = iter_dataset_npy_files(data_root)
    groups = {}
    for path in all_npy:
        key = group_dataset_files_by_parent(path, data_root)
        groups.setdefault(key, []).append(path)

    manager = mp.Manager()
    task_queue = manager.Queue()

    for item in groups.items():
        task_queue.put(item)

    # ★ 为每张 GPU 创建单独模型副本
    gpu_models = replicate_model_to_devices(base_model, device_list)

    processes = []
    for gpu_idx, dev in enumerate(device_list):
        model_for_dev = gpu_models[dev]     # ★ 传入该 GPU 的模型对象

        p = mp.Process(
            target=run_dataset_worker,
            args=(
                gpu_idx,
                dev,
                model_for_dev,        # ★ 模型对象进入 worker
                config,
                task_queue,
                data_root,
                save_root,
                batch_size,
                resize_mode
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n🎉 All GPUs finished all tasks.")




if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    ckpt = "<SWIN_CHECKPOINT_PATH>"
    base_model = load_base_model_once(
            ckpt_path=ckpt,
            config=transformer_vqvae_mini_config3
        )

    process_all_datasets_in_parallel(
            data_root="<RAW_DATA_ROOT>/Chronos/Training_Corpus_Kernel_Synth_1m/",
            save_root="<CODEBOOK_OUTPUT_ROOT>/Chronos/Training_Corpus_Kernel_Synth_1m/",
            base_model=base_model,              # ★ 主进程创建好的模型
            config=transformer_vqvae_mini_config3,
            batch_size=2048,
            device_list=("cuda:0","cuda:1","cuda:2","cuda:3"),
            # device_list=("cuda:0","cuda:2"),
            resize_mode="truncate"
        )
