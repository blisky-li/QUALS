import numpy as np
import os
import json
import torch
import sys
sys.path.append(os.environ.get("TSVQ_REPO_ROOT", "<TSVQ_REPO_ROOT>"))
from patchddp.arch import TransformerModel
from patchddp.config.transformervqvae_config import transformer_vqvae_general_config2
from setproctitle import setproctitle

import os

def find_missing_codebook_outputs(path_A, path_B, output_txt):
    missing = []

    # 遍历路径 A
    for root, dirs, files in os.walk(path_A):
        for f in files:
            if not f.endswith(".npy"):
                continue

            # 如果文件名包含 label（不区分大小写），跳过
            if "label" in f.lower():
                continue

            npy_abs = os.path.join(root, f)

            # 构造相对路径
            rel_path = os.path.relpath(npy_abs, path_A)

            # 对应 B 路径中的 npz
            npz_rel = rel_path.replace(".npy", "_codebook.npz")
            npz_abs = os.path.join(path_B, npz_rel)

            # 检查是否存在
            if not os.path.exists(npz_abs):
                missing.append((f, npy_abs))

    # 写入 txt
    with open(output_txt, "w", encoding="utf-8") as fw:
        for fname, abs_path in missing:
            fw.write(f"{fname}, {abs_path}\n")

    print(f"检查完成，缺失文件数量: {len(missing)}")
    print(f"结果已写入: {output_txt}")

    return missing

# ======================
# 使用示例
# ======================
path_A = "<RAW_DATA_ROOT>"
path_B = "<CODEBOOK_OUTPUT_ROOT>"
output_txt = "missing_codebook_files_without_label.txt"

find_missing_codebook_outputs(path_A, path_B, output_txt)
