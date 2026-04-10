import h5py
import numpy as np


def count_records_per_cell(
    h5_path: str,
    num_cells: int = 10000,
):
    """
    返回:
        counts: np.ndarray, shape (num_cells,), dtype=int64
                counts[i] = cell i 的记录数
    """
    counts = np.zeros(num_cells, dtype=np.int64)

    with h5py.File(h5_path, "r") as f:
        for cell in range(num_cells):
            dset = f[str(cell)]
            counts[cell] = dset.shape[0]

    return counts

def find_first_nonempty_cell(h5_path, num_cells=10000):
    """
    返回:
        cell_id: 最小的、包含数据的 cell index
        count:   该 cell 的记录数
    """
    with h5py.File(h5_path, "r") as f:
        for cell in range(num_cells):
            dset = f[str(cell)]
            if dset.shape[0] > 0:
                return cell, dset.shape[0]

    return None, 0


def preview_cell_records(h5_path, cell_id, max_rows=10):
    """
    打印 cell 内前 max_rows 条记录
    """
    with h5py.File(h5_path, "r") as f:
        dset = f[str(cell_id)]

        print(f"[INFO] Cell {cell_id}")
        print(f"  Total records: {dset.shape[0]}")
        print("  Columns: [file_id, n, c, s]")

        rows = dset[:max_rows]
        print("  First records:")
        print(rows)

        return rows


def export_cell_shape_summary(
    h5_path: str,
    out_txt: str,
    num_cells: int = 10000,
):
    """
    将每个 cell 的 dataset shape 写成 txt
    每行:
        cell_id \t num_records \t dim
    """
    with h5py.File(h5_path, "r") as f, open(out_txt, "w", encoding="utf-8") as fw:
        fw.write("# cell_id\tcount\tdim\n")

        for cell in range(num_cells):
            dset = f[str(cell)]
            count, dim = dset.shape  # (K, 4)
            fw.write(f"{cell}\t{count}\t{dim}\n")

    print(f"[DONE] Cell shapes written to {out_txt}")

if __name__ == "__main__":
    H5_PATH = "cell_usage_uint32_2500.h5"
    NUM_CELLS = 2500

    counts = count_records_per_cell(H5_PATH, NUM_CELLS)

    print("Total records:", counts.sum())
    print("Non-empty cells:", np.sum(counts > 0))
    print("Top-10 largest cells:", np.sort(counts)[-10:])
    print("Top-10 smallest cells:", np.sort(counts)[:10])

    # 1️⃣ 找最小非空 cell
    cell_id, count = find_first_nonempty_cell(H5_PATH, NUM_CELLS)

    if cell_id is None:
        print("[ERROR] No non-empty cells found")
    else:
        print(f"[FOUND] Min non-empty cell = {cell_id}, count = {count}")

        # 2️⃣ 查看该 cell 的内容
        preview_cell_records(H5_PATH, cell_id, max_rows=10)

    OUT_TXT = "cell_shapes.txt"
    NUM_CELLS = 2500

    export_cell_shape_summary(
        h5_path=H5_PATH,
        out_txt=OUT_TXT,
        num_cells=NUM_CELLS,
    )
