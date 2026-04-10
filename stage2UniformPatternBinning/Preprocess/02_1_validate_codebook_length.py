import os
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# ============================================================
# 单文件检查函数（子进程执行）
# ============================================================
def validate_single_codebook_file(args):
    npz_path, expected_last_dim = args

    try:
        data = np.load(npz_path, mmap_mode="r")
    except Exception as e:
        return {
            "ok": False,
            "path": npz_path,
            "error": f"np.load failed: {repr(e)}",
        }

    if "codebook_ids" not in data.files:
        return {
            "ok": False,
            "path": npz_path,
            "error": "missing key: codebook_ids",
            "keys": list(data.files),
        }

    cb = data["codebook_ids"]

    if cb.ndim == 0 or cb.shape[-1] != expected_last_dim:
        return {
            "ok": False,
            "path": npz_path,
            "error": "invalid last dimension",
            "shape": list(cb.shape),
            "expected_last_dim": expected_last_dim,
        }

    return {"ok": True}


# ============================================================
# 主检查函数（多进程）
# ============================================================
from typing import Optional

def validate_codebook_length_for_directory(
    path_B: str,
    expected_last_dim: int = 734,
    report_json: str = "codebook_dim_check_report.json",
    num_workers: Optional[int] = None,
):

    """
    多进程检查 path_B 下所有 *_codebook.npz
    """

    # ----------------------------
    # 收集所有 codebook 文件
    # ----------------------------
    codebook_paths = []

    level1_dirs = sorted([
        d for d in os.listdir(path_B)
        if os.path.isdir(os.path.join(path_B, d))
    ])

    for d1 in level1_dirs:
        abs1 = os.path.join(path_B, d1)
        for root, dirs, files in os.walk(abs1):
            dirs.sort()
            files.sort()
            for fname in files:
                if fname.endswith("_codebook.npz"):
                    codebook_paths.append(os.path.join(root, fname))

    total_files = len(codebook_paths)
    print(f"[INFO] Found {total_files} *_codebook.npz files")

    if total_files == 0:
        print("[WARN] No codebook files found")
        return None, None

    # ----------------------------
    # 多进程参数
    # ----------------------------
    if num_workers is None:
        # I/O + CPU 混合，建议不要用满
        num_workers = max(1, cpu_count() // 2)

    print(f"[INFO] Using {num_workers} worker processes")

    # ----------------------------
    # 多进程检查
    # ----------------------------
    bad_reports = []
    valid = 0

    with Pool(processes=num_workers) as pool:
        tasks = (
            (path, expected_last_dim)
            for path in codebook_paths
        )

        for result in tqdm(
            pool.imap_unordered(validate_single_codebook_file, tasks),
            total=total_files,
            desc="Checking codebooks",
            unit="file",
        ):
            if result["ok"]:
                valid += 1
            else:
                bad_reports.append(result)

    # ----------------------------
    # 汇总 & 保存报告
    # ----------------------------
    summary = {
        "root": path_B,
        "expected_last_dim": expected_last_dim,
        "total_codebooks": total_files,
        "valid_codebooks": valid,
        "invalid_codebooks": len(bad_reports),
        "num_workers": num_workers,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "bad_reports": bad_reports,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[CHECK DONE]")
    print(json.dumps(summary, indent=2))
    print(f"详细异常已保存到: {report_json}")

    return summary, bad_reports


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    path_B = "<CODEBOOK_OUTPUT_ROOT>"
    validate_codebook_length_for_directory(
        path_B=path_B,
        expected_last_dim=734,
        report_json="codebook_dim_check_report.json",
        num_workers=24,   # 自动
    )
