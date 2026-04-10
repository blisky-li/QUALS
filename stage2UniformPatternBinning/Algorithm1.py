import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Optional: set your repo root or working directory
os.chdir(r"<SWIN_REPO_ROOT>")
ckpt_path = r"<SWIN_CHECKPOINT_PATH>"
codebook_key = "vqmodel._codebook.embed"  # change if your key differs
if ckpt_path.startswith("<"):
    raise ValueError("Please set ckpt_path to your Swin checkpoint .pt file.")

checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint)
if codebook_key not in state_dict:
    raise KeyError(f"codebook_key not found: {codebook_key}")

codebook = state_dict[codebook_key].squeeze()
print("Codebook shape:", tuple(codebook.shape))

cb = codebook.detach().cpu().numpy()
if cb.ndim != 2 or cb.shape[1] != 32:
    raise ValueError(f"Expected (2048, 32), got {cb.shape}")
pca = PCA(svd_solver="full")
pca.fit(cb)
print(pca.n_components_)
print(pca.explained_variance_)
plt.plot(pca.explained_variance_)
plt.yscale("log")
lam = pca.explained_variance_
ratio = lam[1:] / (lam[:-1] + 1e-12)

print("ratio mean:", ratio.mean())
print("ratio std :", ratio.std())
print("cv        :", ratio.std() / (ratio.mean() + 1e-12))
i = np.arange(1, len(lam)+1)
slope, intercept = np.polyfit(i, np.log(lam + 1e-12), 1)

b = -slope
r = np.exp(-b)
print("b:", b, "ratio r:", r)

import numpy as np
from sklearn.decomposition import PCA

def build_bins_by_spectrum(lambdas, target_cells=1000, min_bins=1, max_bins=32):
    # lambdas: (32,) descending
    w = np.sqrt(lambdas / lambdas[0])  # spectrum weight
    b = np.ones_like(w, dtype=np.int32) * min_bins

    def prod(x): 
        p = 1
        for v in x: p *= int(v)
        return p

    # greedy: increase bins where it helps most
    while prod(b) < target_cells:
        # score: higher weight and fewer bins => higher priority
        score = w / b
        j = int(np.argmax(score))
        if b[j] >= max_bins:
            score[j] = -1
            j = int(np.argmax(score))
            if score[j] < 0:
                break
        b[j] += 1
    return b  # (32,)

def fit_codebook_partition(cb_2048x32, target_cells=1000):
    cb = cb_2048x32.astype(np.float32)

    # full PCA (no truncation)
    pca = PCA(n_components=32, svd_solver="full", whiten=False)
    z_cb = pca.fit_transform(cb)  # (2048,32)

    lambdas = pca.explained_variance_.astype(np.float32)  # (32,)
    b = build_bins_by_spectrum(lambdas, target_cells=target_cells)

    # per-dim quantile edges
    edges = []
    for i in range(32):
        bi = int(b[i])
        qs = np.linspace(0.0, 1.0, bi + 1, dtype=np.float32)
        edges_i = np.quantile(z_cb[:, i], qs).astype(np.float32)
        edges.append(edges_i)

    # raw_id for codebook occupancy
    raw_ids = encode_raw_ids(z_cb, edges, b)

    # merge to exactly target_cells by equal-mass (codebook-based)
    map_table = build_merge_table_by_occupancy(raw_ids, int(np.prod(b)), target_cells)

    params = {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),    # (32,32)
        "lambdas": lambdas,
        "b": b.astype(np.int32),
        "edges": edges,                                      # list of 32 arrays
        "map_table": map_table.astype(np.int32),             # (prod_b,)
    }
    return params

def encode_raw_ids(z, edges, b):
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

def build_merge_table_by_occupancy(raw_ids, raw_space_size, target_cells):
    # histogram over raw ids using codebook points
    hist = np.bincount(raw_ids.astype(np.int64), minlength=raw_space_size).astype(np.int64)
    total = hist.sum()
    # target mass per bucket
    per = total / target_cells

    table = np.zeros(raw_space_size, dtype=np.int32)
    acc = 0.0
    cur = 0
    for rid in range(raw_space_size):
        table[rid] = cur
        acc += hist[rid]
        if cur < target_cells - 1 and acc >= (cur + 1) * per:
            cur += 1
    return table

def assign_to_1000(vectors_Nx32, params):
    # vectors_Nx32: your 734-mean outputs, shape (N,32)
    mean = params["mean"]
    comp = params["components"]
    edges = params["edges"]
    b = params["b"]
    table = params["map_table"]

    z = (vectors_Nx32.astype(np.float32) - mean) @ comp.T  # (N,32)
    raw = encode_raw_ids(z, edges, b)
    final = table[raw]
    return final  # (N,) in [0,999]
import numpy as np

def save_partition_path(params, save_path):
    """
    params: dict from fit_codebook_partition
    """
    np.savez(
        save_path,

        mean=params["mean"],                 # (32,)
        components=params["components"],     # (32,32)
        lambdas=params["lambdas"],           # (32,)
        b=params["b"],                       # (32,)

        # edges 是 ragged list，需单独保存
        **{f"edges_{i}": params["edges"][i] for i in range(32)},

        map_table=params["map_table"],       # (prod(b),)
        raw_space_size=len(params["map_table"]),
    )

    print(f"Saved partition path to {save_path}")
import numpy as np
from sklearn.decomposition import PCA
import torch

# =========================
# 0. 读取 codebook
# =========================

# Output paths


checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint)
if codebook_key not in state_dict:
    raise KeyError(f"codebook_key not found: {codebook_key}")

cb = state_dict[codebook_key].squeeze().detach().cpu().numpy()
assert cb.shape == (2048, 32)


# =========================
# 1. 基于 codebook 拟合划分路径（一次性）
# =========================
TARGET_CELLS = 2500

params = fit_codebook_partition(
    cb_2048x32=cb,
    target_cells=TARGET_CELLS,
)

print("Partition fitted.")
print("Per-dimension bins:", params["b"])
print("Raw grid size:", np.prod(params["b"]))

# =========================
# 2. 保存“冻结路径”
# =========================
SAVE_PATH = "codebook_32d_2500cells.npz"

save_partition_path(
    params=params,
    save_path=SAVE_PATH,
)

print(f"Partition path saved to: {SAVE_PATH}")
