import h5py
import numpy as np

def initialize_cell_usage_hdf5(
    filename="cell_usage_uint32.h5",
    num_cells=10000,
):
    """
    每个 cell 一个 dataset:
        shape = (0, 4), dtype=uint32
        [:, 0] = file_id
        [:, 1] = n
        [:, 2] = c
        [:, 3] = s (scale 0-4)
    """
    with h5py.File(filename, "w") as f:
        for i in range(num_cells):
            f.create_dataset(
                str(i),
                shape=(0, 5),
                maxshape=(None, 5),
                dtype=np.uint32,
                chunks=(65536, 5),
            )
    print(f"[INIT] cell HDF5 initialized: {filename}")

def append_cell_records_to_hdf5(filename, batch_map):
    """
    batch_map:
        { cell_id(int): np.ndarray(shape=(K,4), dtype=uint32) }
    """
    with h5py.File(filename, "a") as f:
        for cell, records in batch_map.items():
            if records is None or len(records) == 0:
                continue

            dset = f[str(cell)]
            old_len = dset.shape[0]
            new_len = old_len + records.shape[0]

            dset.resize((new_len, 5))
            dset[old_len:new_len, :] = records

import numpy as np

def collect_cell_records_from_cellindex_file(args):
    """
    args:
        (file_id, abs_path)

    读取:
        cell: shape (N, C, 5)
        codebook_ids: shape (N, C, D)

    返回:
        local_map:
            { cell_id: np.ndarray(shape=(K,5), dtype=uint32) }
    """
    file_id, abs_path = args
    file_id = np.uint32(file_id)

    # ---------- load cellindex ----------
    data = np.load(abs_path)
    if "cell" not in data.files:
        raise KeyError(f"{abs_path} missing key 'cell'")
    arr = data["cell"]   # (N, C, 5)

    if arr.ndim != 3 or arr.shape[2] != 5:
        raise ValueError(f"{abs_path} invalid shape {arr.shape}")

    # ---------- load codebook ----------
    codebook_path = abs_path.replace("_cellindex.npz", "_codebook.npz")
    if not os.path.exists(codebook_path):
        raise FileNotFoundError(codebook_path)

    cb = np.load(codebook_path)
    codebook_ids = cb["codebook_ids"] if "codebook_ids" in cb.files else cb
    # shape = (N, C, D)

    N, C, S = arr.shape
    _, _, D = codebook_ids.shape
    assert D >= 255, f"{codebook_path}: D={D} < 255"

    local_map = {}

    for s in range(S):
        slice_s = arr[:, :, s]   # (N, C)

        n_idx, c_idx = np.where(slice_s >= 0)
        cells = slice_s[n_idx, c_idx].astype(np.int32)

        for cell in np.unique(cells):
            sel = (cells == cell)

            nn = n_idx[sel]
            cc = c_idx[sel]

            K = sel.sum()
            rows = np.empty((K, 5), dtype=np.uint32)

            rows[:, 0] = file_id
            rows[:, 1] = nn
            rows[:, 2] = cc
            rows[:, 3] = s

            # ---------- compute valid_len ----------
            for i in range(K):
                codes = codebook_ids[nn[i], cc[i], -255:]
                valid_len = np.count_nonzero(~np.isnan(codes)) * 16
                rows[i, 4] = valid_len

            if cell in local_map:
                local_map[cell].append(rows)
            else:
                local_map[cell] = [rows]

    for cell in local_map:
        local_map[cell] = np.concatenate(local_map[cell], axis=0)

    return local_map


from multiprocessing import Pool
from tqdm import tqdm
import json
import os


def build_cell_usage_hdf5_from_cellindex(
    cellindex_index_json,
    h5_filename,
    num_workers=32,
    flush_ratio=0.05,
    num_cells=10000,
):
    with open(cellindex_index_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    file_index = {}

    for k, v in raw.items():
        file_id = int(k)

        if not v.endswith("_codebook.npz"):
            raise ValueError(f"unexpected path in index json: {v}")

        cellindex_path = v.replace("_codebook.npz", "_cellindex.npz")

        if not os.path.exists(cellindex_path):
            raise FileNotFoundError(cellindex_path)

        file_index[file_id] = cellindex_path

    tasks = list(file_index.items())
    total = len(tasks)
    flush_every = max(1, int(total * flush_ratio))

    print(f"[INFO] cellindex files: {total}")
    print(f"[INFO] flush_every: {flush_every}")

    cell_map = {i: [] for i in range(num_cells)}
    processed = 0

    with Pool(processes=num_workers) as pool:
        for local_map in tqdm(
            pool.imap_unordered(collect_cell_records_from_cellindex_file, tasks),
            total=total,
            desc="Processing cellindex",
        ):
            processed += 1

            for cell, arr in local_map.items():
                cell_map[cell].append(arr)

            if processed % flush_every == 0:
                flush_map = {
                    cell: np.concatenate(blocks, axis=0)
                    for cell, blocks in cell_map.items()
                    if blocks
                }
                append_cell_records_to_hdf5(h5_filename, flush_map)
                cell_map = {i: [] for i in range(num_cells)}

    # final flush
    flush_map = {
        cell: np.concatenate(blocks, axis=0)
        for cell, blocks in cell_map.items()
        if blocks
    }
    append_cell_records_to_hdf5(h5_filename, flush_map)

    print("[DONE] cell usage mapping completed")

if __name__ == "__main__":
    CELLINDEX_INDEX_JSON = "registered_codebook_index.json"
    H5_FILENAME = "cell_usage_uint32_2500.h5"

    NUM_CELLS = 2500
    NUM_WORKERS = 48
    FLUSH_RATIO = 0.05

    initialize_cell_usage_hdf5(
        filename=H5_FILENAME,
        num_cells=NUM_CELLS,
    )

    build_cell_usage_hdf5_from_cellindex(
        cellindex_index_json=CELLINDEX_INDEX_JSON,
        h5_filename=H5_FILENAME,
        num_workers=NUM_WORKERS,
        flush_ratio=FLUSH_RATIO,
        num_cells=NUM_CELLS,
    )
