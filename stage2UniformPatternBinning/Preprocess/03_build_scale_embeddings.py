import os
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# ============================================================
# 全局配置
# ============================================================
SCALES = [32, 64, 128, 255, 255]     # sum = 734
CODEBOOK_SIZE = 2048
EMBED_DIM = 32


def compute_weighted_scale_embedding(
    emb: np.ndarray,
):
    """
    emb: [K, D]
         - D 维度无 NaN
         - K 维度可能存在整行 NaN（任意位置）

    return: [D]
    """
    K, D = emb.shape

    if K == 0:
        return np.zeros((D,), dtype=emb.dtype)

    # 有效行：只要这一行不是 NaN
    # （任选一个维度检查即可）
    valid_mask = ~np.isnan(emb[:, 0])   # [K]

    if not np.any(valid_mask):
        # 没有任何有效 code
        return np.zeros((D,), dtype=emb.dtype)

    emb_valid = emb[valid_mask]          # [K_valid, D]

    K_valid = emb_valid.shape[0]
    if K_valid == 1:
        return emb_valid[0]

    # 线性递增权重（按原 K 位置，强调“后段 code”）
    base_w = np.linspace(0.0, 1.0, K, dtype=emb.dtype)  # [K]
    w = base_w[valid_mask]                              # [K_valid]

    # 严格归一化（只在有效 K 上）
    s = w.sum()
    if s == 0:
        w[:] = 1.0 / K_valid
    else:
        w /= s

    # 加权聚合
    return (emb_valid * w[:, None]).sum(axis=0)



# ============================================================
# 单文件处理函数（子进程执行）
# ============================================================
def convert_codebook_file_to_scale_embeddings(args):
    """
    输入:
        args = (npz_path, codebook_l)

    输出:
        None（直接写文件）
    """
    npz_path, codebook_l = args

    data = np.load(npz_path)
    arr = data["codebook_ids"] if "codebook_ids" in data.files else data

    if arr.ndim != 3:
        raise ValueError(f"{npz_path} 的维度不是 (N, C, 734)")

    N, C, L = arr.shape
    if L != sum(SCALES):
        raise ValueError(f"{npz_path}: L={L} != {sum(SCALES)}")

    # 输出 (N, C, 5, 32)
    out = np.zeros((N, C, len(SCALES), EMBED_DIM), dtype=np.float32)

    start = 0
    for si, scale_len in enumerate(SCALES):
        end = start + scale_len
        slice_s = arr[:, :, start:end]   # (N, C, scale_len)

        for n in range(N):
            for c in range(C):
                codes = slice_s[n, c]
                codes = codes[~np.isnan(codes)]

                if codes.size == 0:
                    continue

                codes = codes.astype(np.int64)
                emb = codebook_l[codes]          # (K, 32)
                out[n, c, si] = compute_weighted_scale_embedding(emb)

                # out[n, c, si] = emb.mean(axis=0)

        start = end

    out_path = npz_path.replace("_codebook.npz", "_embedding.npz")
    np.savez_compressed(out_path, embedding=out)

    return npz_path

from typing import Optional

# ============================================================
# 主流程：多进程按文件并行
# ============================================================
def build_scale_embeddings_for_directory(
    root_dir: str,
    checkpoint_path: str,
    num_workers: Optional[int] = None
):
    # ---------- load codebook embedding ----------
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    codebook_l = checkpoint["model_state_dict"]["vqmodel._codebook.embed"]
    codebook_l = codebook_l.squeeze().detach().cpu().numpy()

    if codebook_l.shape != (CODEBOOK_SIZE, EMBED_DIM):
        raise ValueError(f"codebook shape mismatch: {codebook_l.shape}")

    # ---------- collect files ----------
    codebook_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith("_codebook.npz"):
                codebook_files.append(os.path.join(root, f))

    codebook_files.sort()
    total = len(codebook_files)
    print(f"[INFO] Found {total} *_codebook.npz files")

    if total == 0:
        return

    # ---------- worker config ----------
    if num_workers is None:
        # I/O + CPU 混合任务，不建议用满
        num_workers = max(1, cpu_count() // 2)

    print(f"[INFO] Using {num_workers} worker processes")

    # ---------- multiprocessing ----------
    tasks = [(p, codebook_l) for p in codebook_files]

    with Pool(processes=num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(convert_codebook_file_to_scale_embeddings, tasks),
            total=total,
            desc="Building embeddings",
            unit="file",
        ):
            pass


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ROOT_DIR = "<CODEBOOK_OUTPUT_ROOT>"
    CKPT_PATH = "<SWIN_CHECKPOINT_PATH>"

    build_scale_embeddings_for_directory(
        root_dir=ROOT_DIR,
        checkpoint_path=CKPT_PATH,
        num_workers=None,   # 自动：CPU 核数 / 2
    )
