import os
import numpy as np
from tqdm import tqdm


def export_sorted_cell_distribution(
    root_dir: str,
    out_txt: str,
    num_cells: int = 10000,
):
    """
    从 root_dir 扫描所有 *_cellindex.npz
    统计 10000 个 cell 的出现次数
    按 count 升序保存为 txt
    """

    # --------------------------------------------------
    # Step 1: 初始化计数器
    # --------------------------------------------------
    counts = np.zeros(num_cells, dtype=np.int64)

    # --------------------------------------------------
    # Step 2: 收集所有 *_cellindex.npz
    # --------------------------------------------------
    cellindex_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith("_cellindex.npz"):
                cellindex_files.append(os.path.join(root, f))

    cellindex_files.sort()
    print(f"[INFO] Found {len(cellindex_files)} *_cellindex.npz files")

    # --------------------------------------------------
    # Step 3: 逐文件统计
    # --------------------------------------------------
    for path in tqdm(cellindex_files, desc="Counting cell usage"):
        data = np.load(path)
        if "cell" not in data.files:
            raise KeyError(f"{path} missing key 'cell'")

        cell = data["cell"]              # (N, C, 5)
        flat = cell.reshape(-1)          # (N*C*5,)

        # 防御性过滤（如果你确信没有非法值，可去掉）
        flat = flat[(flat >= 0) & (flat < num_cells)]

        uniq, cnt = np.unique(flat, return_counts=True)
        counts[uniq] += cnt

    # --------------------------------------------------
    # Step 4: 按 count 排序并写 txt
    # --------------------------------------------------
    order = np.argsort(counts)   # 升序：稀疏 -> 密集

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# cell_id\tcount\n")
        for idx in order:
            f.write(f"{idx}\t{int(counts[idx])}\n")

    print(f"[DONE] Saved sorted cell distribution to {out_txt}")


if __name__ == "__main__":
    # ROOT_DIR = "<CODEBOOK_OUTPUT_ROOT>/Series/Mixed/"
    ROOT_DIR = "<CODEBOOK_OUTPUT_ROOT>"
    OUT_TXT = "mixed_cell_distribution_sorted_all.txt"
    NUM_CELLS = 2500

    export_sorted_cell_distribution(
        root_dir=ROOT_DIR,
        out_txt=OUT_TXT,
        num_cells=NUM_CELLS,
    )
