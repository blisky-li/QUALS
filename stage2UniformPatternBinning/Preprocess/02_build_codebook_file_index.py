import os
import json



def build_codebook_file_index(path_B, output_json):
    """
    为每个 *_codebook.npz 分配一个连续的 uint32 file_id
    保存：
      1) {file_id (int): abs_path}
      2) {abs_path: file_id (int)}
    """

    id_to_path = {}
    path_to_id = {}

    cur_id = 0  # 连续 uint32

    # 一级目录，保证顺序稳定
    level1_dirs = sorted([
        d for d in os.listdir(path_B)
        if os.path.isdir(os.path.join(path_B, d))
    ])

    for d1 in level1_dirs:
        abs1 = os.path.join(path_B, d1)

        # 深度优先遍历（os.walk 本身就是确定性的）
        for root, dirs, files in os.walk(abs1):
            dirs.sort()   # 明确排序，保证可复现
            files.sort()

            # 仅处理 *_codebook.npz
            codebook_files = [
                f for f in files
                if f.endswith("_codebook.npz")
            ]

            for f in codebook_files:
                npz_abs = os.path.join(root, f)

                file_id = cur_id
                cur_id += 1

                # 存为 int（后续可直接 cast np.uint32）
                id_to_path[file_id] = npz_abs
                path_to_id[npz_abs] = file_id

    # -------------------------------
    # 保存 1：file_id -> path
    # -------------------------------
    with open(output_json, "w", encoding="utf-8") as fw:
        json.dump(
            id_to_path,
            fw,
            indent=2,
            ensure_ascii=False
        )

    # -------------------------------
    # 保存 2：path -> file_id
    # -------------------------------
    reverse_json = output_json.replace(".json", "_reverse.json")
    with open(reverse_json, "w", encoding="utf-8") as fw:
        json.dump(
            path_to_id,
            fw,
            indent=2,
            ensure_ascii=False
        )

    print(f"[DONE] 已完成编号，共 {cur_id} 个 *_codebook.npz 文件")
    print(f"正向 JSON : {output_json}")
    print(f"反向 JSON : {reverse_json}")

    return id_to_path, path_to_id


# ---------------- 示例 ----------------
if __name__ == "__main__":
    path_B = "<CODEBOOK_OUTPUT_ROOT>"
    output_json = "registered_codebook_index.json"
    build_codebook_file_index(path_B, output_json)
