import h5py
import numpy as np
from tqdm import tqdm
import multiprocessing as mp



def sample_sequences_for_single_cell(
    worker_id: int,
    h5_path: str,
    registered_index_rev: dict,
    task_queue,
    result_queue,
    *,
    sample_size: int = 8000,
    min_valid_len: int = 512,
    use_mae_mse_filter: bool = False,
    mae_threshold: float = 0.2,
    mse_threshold: float = 0.2,
    base_seed: int = 42,
):
    """
    每个 worker 的行为（新逻辑）：
      1. 只保留 valid_len >= min_valid_len 的单条序列
      2. （可选）MAE / MSE 过滤
      3. 不做任何组合，每次只加入一条序列
      4. 优先使用过滤后的 items（不重复）
      5. 若 items 数量不足 sample_size，则允许从历史 items 中随机重复补齐
      6. 保证 results 长度 == sample_size
    """

    import numpy as np
    import h5py

    # ---------- 工具函数 ----------
    def load_error_maps(path, cache):
        if path not in cache:
            data = np.load(path)
            cache[path] = (data["mae"], data["mse"])
        return cache[path]

    # ---------- 主循环 ----------
    f = h5py.File(h5_path, "r")

    while True:
        index_id = task_queue.get()
        if index_id is None:
            break

        # index 不存在
        if str(index_id) not in f:
            result_queue.put((index_id, [], "empty"))
            continue

        rows = f[str(index_id)][:]   # [M, 5]
        error_cache = {}
        items = []

        # ---------- Step 0: 严格过滤 ----------
        for row in rows:
            file_id, n, c, level, valid_len = map(int, row)

            if valid_len < min_valid_len:
                continue

            if use_mae_mse_filter:
                maemse_path = (
                    registered_index_rev[str(file_id)]
                    .replace("_codebook", "_maemse")
                )
                mae, mse = load_error_maps(maemse_path, error_cache)
                if mae[n, c] > mae_threshold or mse[n, c] > mse_threshold:
                    continue

            items.append((file_id, n, c, level, valid_len))

        if not items:
            result_queue.put((index_id, [], "empty"))
            continue

        # ---------- Step 1: 随机逐条采样（无组合） ----------
        rng = np.random.default_rng(base_seed + index_id)
        results = []

        # 1️⃣ 优先使用过滤后的 items（不重复）
        order = rng.permutation(len(items))
        for idx in order:
            # results.append(items[idx])
            results.append([items[idx]])
            if len(results) >= sample_size:
                break

        # 2️⃣ 不足 sample_size，则允许历史 items 随机重复补齐
        if len(results) < sample_size:
            need = sample_size - len(results)
            extra_idx = rng.integers(0, len(items), size=need)
            for i in extra_idx:
                # results.append(items[i])
                results.append([items[i]])

        # 此时保证：
        # - len(results) == sample_size
        # - 每个元素都是单条序列
        # - 无组合、无 buffer、无 while 死循环
        result_queue.put((index_id, results, "ok"))

    f.close()



if __name__ == "__main__":
    import json
    import multiprocessing as mp
    from tqdm import tqdm
    import orjson

    # =========================
    # 1. 基本配置（你只需要改这里）
    # =========================
    H5_PATH = "cell_usage_uint32_2500.h5"
    REGISTERED_INDEX_JSON = "registered_codebook_index.json"

    NUM_INDEX = 2500         # index 数量
    NUM_WORKERS = 96         # 并行进程数

    SAMPLE_SIZE = 8000
    MIN_VALID_LEN = 511

    USE_MAE_MSE_FILTER = False
    MAE_THRESHOLD = 0.2
    MSE_THRESHOLD = 0.2

    BASE_SEED = 42


    # =========================
    # 2. 读取 registered_index
    # =========================
    if USE_MAE_MSE_FILTER:
        with open(REGISTERED_INDEX_JSON, "r") as f:
            registered_index = json.load(f)
        registered_index_rev = {
            str(fid): path for fid, path in registered_index.items()
        }
    else:
        registered_index_rev = None

    # =========================
    # 3. multiprocessing 队列
    # =========================
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # 所有 index 入队
    for index_id in range(NUM_INDEX):
        task_queue.put(index_id)

    # 结束信号
    for _ in range(NUM_WORKERS):
        task_queue.put(None)

    # =========================
    # 4. 启动 workers
    # =========================
    workers = []
    for wid in range(NUM_WORKERS):
        p = mp.Process(
            target=sample_sequences_for_single_cell,
            args=(
                wid,
                H5_PATH,
                registered_index_rev,
                task_queue,
                result_queue,
            ),
            kwargs=dict(
                sample_size=SAMPLE_SIZE,
                min_valid_len=MIN_VALID_LEN,
                use_mae_mse_filter=USE_MAE_MSE_FILTER,
                mae_threshold=MAE_THRESHOLD,
                mse_threshold=MSE_THRESHOLD,
                base_seed=BASE_SEED,
            )
        )
        p.start()
        workers.append(p)

    # =========================
    # 5. 收集结果
    # =========================
    sampled_map = {}
    pbar = tqdm(total=NUM_INDEX, desc="Processing indices", ncols=100)

    for _ in range(NUM_INDEX):
        index_id, samples, status = result_queue.get()
        if status == "ok":
            sampled_map[str(index_id)] = samples
        else:
            print(f"[WARN] index {index_id} failed: {status}")
        pbar.update(1)

    pbar.close()

    
    print('✅result finish!')


    # =========================
    # 6. 等待 worker 结束
    # =========================
    for p in workers:
        p.join()

    print('✅worker finish!')

    # =========================
    # 7. 保存结果
    # =========================

    import orjson

    NDJSON_PATH = "cell_usage_sampled2re.ndjson"
    OFFSET_PATH = "cell_usage_sampled2re.offset.json"

    def to_py(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, list):
            return [to_py(x) for x in obj]
        if isinstance(obj, tuple):
            return [to_py(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        return obj

    offset_map = {}

    with open(NDJSON_PATH, "wb") as f:
        for index_id, results in sampled_map.items():
            offset = f.tell()
            rec = {
                "index": str(index_id),
                "results": to_py(results),   # ⭐ 关键修复
            }
            line = orjson.dumps(rec)
            f.write(line)
            f.write(b"\n")
            offset_map[str(index_id)] = offset

    with open(OFFSET_PATH, "wb") as f:
        f.write(orjson.dumps(offset_map))

    print(f"✅ Done. Result saved to: {NDJSON_PATH}")
