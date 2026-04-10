# Preprocess Tutorial

这个目录整理为开源版预处理流水线，按原来的 `step1 -> step7` 顺序重新命名，方便作为一条完整的预处理流程阅读和使用。

运行前需要先统一修改每个脚本顶部或 `__main__` 里的路径常量。
路径常量统一使用占位符，例如 `<RAW_DATA_ROOT>`、`<CODEBOOK_OUTPUT_ROOT>`、`<SWIN_CHECKPOINT_PATH>`，请替换为你自己的数据根目录、checkpoint 路径和输出目录。

## Pipeline Order

1. [01_extract_patch_features.py](01_extract_patch_features.py)
   作用：读取原始 `.npy` 时序数据，按 `valid_indices` 截取有效片段、归一化、缩放到固定长度，然后调用 Swin VQ 模型导出 patch code。
   主要输出：`*_codebook.npz` 和 `*_maemse.npz`。
   关键函数：
   `load_valid_indices_from_metadata`
   `slice_valid_segments_and_normalize`
   `build_resized_model_inputs`
   `export_patch_features_with_vqvae`
   `process_all_datasets_in_parallel`

2. [01_1_check_missing_codebook_files.py](01_1_check_missing_codebook_files.py)
   作用：检查原始 `.npy` 数据是否都已经生成了对应的 `*_codebook.npz`。
   主要输出：缺失文件清单 txt。
   关键函数：
   `find_missing_codebook_outputs`

3. [02_build_codebook_file_index.py](02_build_codebook_file_index.py)
   作用：给全部 `*_codebook.npz` 分配稳定的连续 `file_id`，生成正向和反向索引。
   主要输出：`registered_codebook_index.json` 和对应 reverse json。
   关键函数：
   `build_codebook_file_index`

4. [02_1_validate_codebook_length.py](02_1_validate_codebook_length.py)
   作用：批量检查 `*_codebook.npz` 的最后一维是否符合预期，例如这里默认检查 734。
   主要输出：`codebook_dim_check_report.json`。
   关键函数：
   `validate_single_codebook_file`
   `validate_codebook_length_for_directory`

5. [03_build_scale_embeddings.py](03_build_scale_embeddings.py)
   作用：把 `*_codebook.npz` 中的 codebook id 映射回 32 维 embedding，并按 5 个尺度聚合成 `embedding` 表示。
   主要输出：`*_embedding.npz`。
   关键函数：
   `compute_weighted_scale_embedding`
   `convert_codebook_file_to_scale_embeddings`
   `build_scale_embeddings_for_directory`

6. [04_assign_cells_from_embeddings.py](04_assign_cells_from_embeddings.py)
   作用：读取 `*_embedding.npz`，结合预先拟合好的 cell 划分参数（例如 `codebook_32d_2500cells.npz`），把每个 embedding 映射到 cell id。
   主要输出：`*_cellindex.npz`。
   关键函数：
   `load_partition_parameters`
   `assign_vectors_to_cells`
   `convert_embedding_file_to_cell_index`
   `build_cell_index_for_directory`

7. [05_build_cell_usage_hdf5.py](05_build_cell_usage_hdf5.py)
   作用：汇总全部 `*_cellindex.npz` 和 `*_codebook.npz`，构建按 cell 分桶的 HDF5 索引表。
   主要输出：例如 `cell_usage_uint32_2500.h5`。
   每条记录通常包含：
   `file_id, n, c, scale_id, valid_len`
   关键函数：
   `initialize_cell_usage_hdf5`
   `collect_cell_records_from_cellindex_file`
   `append_cell_records_to_hdf5`
   `build_cell_usage_hdf5_from_cellindex`

8. [05_2_export_global_cell_distribution.py](05_2_export_global_cell_distribution.py)
   作用：扫描所有 `*_cellindex.npz`，统计全局 cell 使用频次，并按稀疏到密集排序导出。
   主要输出：`mixed_cell_distribution_sorted_all.txt`。
   关键函数：
   `export_sorted_cell_distribution`

9. [05_1_inspect_cell_usage_hdf5.py](05_1_inspect_cell_usage_hdf5.py)
   作用：检查 step 5 生成的 HDF5 内容，统计每个 cell 的记录数、查看首个非空 cell、导出 shape 摘要。
   主要输出：`cell_shapes.txt` 等统计信息。
   关键函数：
   `count_records_per_cell`
   `find_first_nonempty_cell`
   `preview_cell_records`
   `export_cell_shape_summary`

10. [05_1_1_plot_cell_samples.ipynb](05_1_1_plot_cell_samples.ipynb)
    作用：交互式查看某些 cell 的样本或可视化结果。
    说明：这是 notebook，适合人工检查，不是批处理主脚本。

11. [06_sample_sequences_per_cell.py](06_sample_sequences_per_cell.py)
    作用：从 step 5 的 HDF5 索引中，按 cell 抽样候选序列，支持按 `valid_len` 和可选的 `mae/mse` 阈值过滤。
    主要输出：`cell_usage_sampled2re.ndjson` 和对应 offset 文件。
    关键函数：
    `sample_sequences_for_single_cell`

12. [07_build_final_resampled_dataset.py](07_build_final_resampled_dataset.py)
    作用：读取 step 6 的抽样结果，把原始序列重新映射回来，随机裁剪、交错合成、归一化，最后写成 `.dat` 数据集和配套 meta。
    主要输出：
    `sequences_by_indexre.dat`
    `sequences_by_indexre_shape.npy`
    `sequences_by_indexre.meta.ndjson`
    关键函数：
    `remap_codebook_path_to_npy`
    `load_valid_indices_from_metadata`
    `interleave_group_sequences`
    `load_sample_results_for_cell`
    `build_sequences_for_single_cell`
    `build_final_dataset_from_indices`

## Data Flow

整体数据流可以概括为：

`raw npy/json`
-> `*_codebook.npz` / `*_maemse.npz`
-> `registered_codebook_index.json`
-> `*_embedding.npz`
-> `*_cellindex.npz`
-> `cell_usage_uint32_2500.h5`
-> `cell_usage_sampled2re.ndjson`
-> final `.dat/.npy/.ndjson`

## Notes

- `05_1`、`05_1_1`、`05_2` 更偏分析和检查，不一定是训练前必须执行的硬步骤。
- `codebook_32d_2500cells.npz` 不是这个目录里的脚本生成的原始文件，它通常是另外拟合好的 cell 划分参数，然后被 step 4 使用。
- 如果准备公开这套流程，建议后续再把各脚本里的绝对路径改成命令行参数或统一配置文件。
