import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Optional


# ============================================================
# 你已有的函数（假设已在环境中）
# ============================================================
# load_partition_parameters(...)
# assign_vectors_to_cells(vectors_Nx32, params)


# ============================================================
# 单文件处理（子进程）
# ============================================================
def convert_embedding_file_to_cell_index(args):
    """
    args:
        (npz_path, params)

    读取:
        embedding: (N, C, 5, 32)

    写出:
        cell: (N, C, 5)
        文件名 *_cellindex.npz
    """
    npz_path, params = args

    data = np.load(npz_path)
    if "embedding" not in data.files:
        raise KeyError(f"{npz_path} missing key 'embedding'")

    E = data["embedding"]     # (N, C, 5, 32)
    if E.ndim != 4 or E.shape[-1] != 32:
        raise ValueError(f"{npz_path} invalid shape {E.shape}")

    N, C, S, D = E.shape
    assert S == 5 and D == 32

    # -------- flatten to (N*C*S, 32) --------
    E_flat = E.reshape(N * C * S, D)

    # -------- assign cell ids --------
    cell_flat = assign_vectors_to_cells(E_flat, params)   # (N*C*S,)

    # -------- reshape back --------
    cell = cell_flat.reshape(N, C, S).astype(np.int32)

    out_path = npz_path.replace("_embedding.npz", "_cellindex.npz")
    np.savez_compressed(out_path, cell=cell)

    return npz_path


def encode_partition_raw_ids(z, edges, b):
    # z: (N,32) already in PCA space
    N = z.shape[0]
    t = np.zeros((N,32), dtype=np.int32)
    for i in range(32):
        ei = edges[i]
        ti = np.searchsorted(ei, z[:, i], side="right") - 1
        ti = np.clip(ti, 0, len(ei) - 2)
        t[:, i] = ti

    # mixed radix encoding
    raw = np.zeros(N, dtype=np.int64)
    stride = 1
    for i in range(32):
        raw += t[:, i].astype(np.int64) * stride
        stride *= int(b[i])
    return raw  # (N,)

def assign_vectors_to_cells(vectors_Nx32, params):
    # vectors_Nx32: your 734-mean outputs, shape (N,32)
    mean = params["mean"]
    comp = params["components"]
    edges = params["edges"]
    b = params["b"]
    table = params["map_table"]

    z = (vectors_Nx32.astype(np.float32) - mean) @ comp.T  # (N,32)
    raw = encode_partition_raw_ids(z, edges, b)
    final = table[raw]
    return final  # (N,) in [0,999]

def load_partition_parameters(path):
    mp = np.load(path)

    mean = mp["mean"]
    components = mp["components"]
    lambdas = mp["lambdas"]
    b = mp["b"].astype(int)

    edges = [mp[f"edges_{i}"] for i in range(32)]
    map_table = mp["map_table"]

    return {
        "mean": mean,
        "components": components,
        "lambdas": lambdas,
        "b": b,
        "edges": edges,
        "map_table": map_table,
    }

# ============================================================
# 主流程：文件级多进程
# ============================================================
def build_cell_index_for_directory(
    root_dir: str,
    partition_param_npz: str,
    num_workers: Optional[int] = None,
):
    # -------- load partition params (一次) --------
    params = load_partition_parameters(partition_param_npz)

    # -------- collect files --------
    codebook_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith("_embedding.npz"):
                codebook_files.append(os.path.join(root, f))

    codebook_files.sort()
    total = len(codebook_files)
    print(f"[INFO] Found {total} *_codebook.npz files")

    if total == 0:
        return

    # -------- worker count --------
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)

    print(f"[INFO] Using {num_workers} worker processes")

    tasks = [(p, params) for p in codebook_files]

    # -------- multiprocessing --------
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(convert_embedding_file_to_cell_index, tasks),
            total=total,
            desc="Building cell index",
            unit="file",
        ):
            pass


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ROOT_DIR = "<CODEBOOK_OUTPUT_ROOT>"
    PARTITION_PARAM = "codebook_32d_2500cells.npz"

    build_cell_index_for_directory(
        root_dir=ROOT_DIR,
        partition_param_npz=PARTITION_PARAM,
        num_workers=None,   # 自动：CPU 核数 / 2
    )
