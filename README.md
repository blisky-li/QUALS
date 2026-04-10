# QUALS: Corpus Equilibrium for Universal Forecasting via Pattern Quantization and Learnability Synchronization

This repository contains the reference implementation of **QUALS**, a scalable corpus construction pipeline for universal time-series forecasting.
QUALS improves pre-training data efficiency by reorganizing large heterogeneous time-series corpora into a **pattern-balanced** and **learnability-synchronized** training corpus.

The core idea is simple: not all time series patterns are equally represented, and not all patterns are equally easy to learn.
QUALS first discovers reusable temporal motifs through vector quantization, then builds uniform semantic bins over the corpus, and finally learns bin-level sampling weights that synchronize convergence across easy and difficult patterns.

> Paper: `QUALS: Corpus Equilibrium for Universal Forecasting via Pattern Quantization and Learnability Synchronization`

> Source code: `https://github.com/blisky-li/QUALS`

> Corpus link: `https://huggingface.co/datasets/Blisky-li/QUALS` (The data of 300G+ is currently being slowly uploaded to HuggingFace)

## Overview

QUALS is organized as a three-stage pipeline:

1. **Stage 1: VQ-SwinTTS Pattern Learning**
   Train a VQ-SwinTTS backbone to encode time-series patches into discrete multi-scale codebook representations.

2. **Stage 2: Uniform Pattern Binning**
   Apply the frozen VQ-SwinTTS model to a large raw corpus, aggregate codebook representations into series-level embeddings, and assign each sequence to density-balanced semantic bins.

3. **Stage 3: GRPO Learnability Synchronization**
   Estimate bin-wise convergence speeds from proxy training logs and optimize bin sampling weights with offline GRPO, producing a learnability-balanced sampling distribution.

The resulting corpus can be used to pre-train universal forecasting models with a substantially reduced token budget while preserving diverse and difficult temporal patterns.

## Repository Structure

```text
quals/
├── environment.txt                # conda environment specification
├── stage1VQSwinTTS/
│   ├── swin/                       # VQ-SwinTTS architecture, config, runner, dataset
│   └── basicts/                    # training utilities used by Stage 1
├── stage2UniformPatternBinning/
│   ├── Algorithm1.py               # spectral allocation / pattern-uniform bin construction
│   └── Preprocess/                 # corpus preprocessing and bin-assignment pipeline
└── stage3GRPO/
    ├── Algorithm2grpo.py           # offline GRPO optimizer for bin sampling weights
    ├── weight_mlp.py               # policy network for bin-level weights
    ├── run_stage3_grpo.sh          # launch script
    └── README.md                   # Stage 3 details
```

## Installation

Create the conda environment from the provided environment file:

```bash
conda env create -f environment.txt
conda activate quals
```

The reference environment was tested with Python 3.9 and PyTorch 2.7.1 + CUDA 12.8.
If your CUDA driver is different, install the matching PyTorch build first, then install the remaining packages listed in `environment.txt`.

## Data Preparation

QUALS expects time-series datasets to be stored as `.npy` arrays with companion metadata `.json` files containing valid-index information.
For open-source release, all local paths are represented by placeholders:

```text
<PROJECT_ROOT>              # root directory of this repository
<RAW_DATA_ROOT>             # raw or processed input time-series corpus
<CODEBOOK_OUTPUT_ROOT>      # output directory for Stage 2 codebook/binning artifacts
<SWIN_REPO_ROOT>            # Stage 1 VQ-SwinTTS code root
<SWIN_CHECKPOINT_PATH>      # trained VQ-SwinTTS checkpoint
<GRPO_METRICS_LOG_DIR>      # proxy training logs used for learnability profiling
<BASELINE_WEIGHT_PATH>      # optional baseline weights for comparison/reporting
<STAGE3_OUTPUT_DIR>         # output directory for GRPO weights
```

Replace these placeholders with your own paths before running the scripts.

## Stage 1: Train VQ-SwinTTS

Stage 1 learns patch-level temporal patterns through a Swin-style hierarchical encoder and vector quantization.
The trained model provides a shared codebook used by later corpus binning steps.

Main code:

```text
stage1VQSwinTTS/swin/arch/
stage1VQSwinTTS/swin/config/
stage1VQSwinTTS/swin/runner/
```

Typical usage:

```bash
cd <PROJECT_ROOT>/quals/stage1VQSwinTTS
python basicts/launcher.py \
  --cfg swin/config/swintts.py \
  --gpus 0
```

Important outputs:

```text
<SWIN_CHECKPOINT_PATH>      # VQ-SwinTTS checkpoint
```

## Stage 2: Build Uniform Pattern Bins

Stage 2 converts the raw corpus into discrete pattern bins.
It first extracts VQ-SwinTTS codebook IDs, then builds series-level embeddings, assigns them to spectral bins, and finally samples sequences from the bin-balanced corpus.

Detailed tutorial:

```text
stage2UniformPatternBinning/Preprocess/TUTORIAL.md
```

Recommended order:

```bash
cd <PROJECT_ROOT>/quals/stage2UniformPatternBinning/Preprocess

# 1. Extract VQ-SwinTTS codebook features from raw series.
python 01_extract_patch_features.py

# 2. Check whether every source file has a codebook output.
python 01_1_check_missing_codebook_files.py

# 3. Register all codebook files with stable file IDs.
python 02_build_codebook_file_index.py

# 4. Validate codebook tensor lengths.
python 02_1_validate_codebook_length.py

# 5. Convert codebook IDs to multi-scale embeddings.
python 03_build_scale_embeddings.py

# 6. Assign embeddings to pattern cells.
python 04_assign_cells_from_embeddings.py

# 7. Build cell-wise HDF5 indices.
python 05_build_cell_usage_hdf5.py

# 8. Inspect or visualize the cell distribution.
python 05_1_inspect_cell_usage_hdf5.py
python 05_2_export_global_cell_distribution.py

# 9. Sample sequences from each cell and build the final corpus.
python 06_sample_sequences_per_cell.py
python 07_build_final_resampled_dataset.py
```

Important outputs:

```text
*_codebook.npz                  # codebook IDs extracted by VQ-SwinTTS
*_embedding.npz                 # series-level multi-scale embeddings
*_cellindex.npz                 # pattern-bin assignments
registered_codebook_index.json  # stable file-id mapping
cell_usage_uint32_2500.h5       # cell-wise index table
sequences_by_indexre.dat        # final sampled corpus
sequences_by_indexre_shape.npy  # memmap shape
```

## Stage 3: Learn GRPO Sampling Weights

Stage 3 uses offline learnability profiles to learn bin sampling weights.
Given validation curves from a uniformly sampled proxy training run, it computes bin-wise convergence speeds and optimizes a policy network with GRPO.

Main code:

```text
stage3GRPO/Algorithm2grpo.py
stage3GRPO/weight_mlp.py
stage3GRPO/run_stage3_grpo.sh
```

Run:

```bash
cd <PROJECT_ROOT>/quals/stage3GRPO

CUDA_VISIBLE_DEVICES=0 bash run_stage3_grpo.sh \
  --metrics_dir <GRPO_METRICS_LOG_DIR> \
  --baseline_weights <BASELINE_WEIGHT_PATH> \
  --out_dir <STAGE3_OUTPUT_DIR>
```

Important outputs:

```text
stage3_grpo_weight.npy            # learned bin sampling weights
stage3_grpo_rate.npz              # computed learnability profile
stage3_grpo_model.pt              # learned MLP policy and embeddings
stage3_grpo_report.json           # similarity metrics against baseline weights
```

## Using the QUALS Corpus

After Stage 2 and Stage 3, use the sampled corpus and learned bin weights to construct a pre-training sampler for your forecasting model.
A typical sampler should:

1. Load the final memmap corpus produced by Stage 2.
2. Load bin assignments or cell indices.
3. Load `stage3_grpo_weight.npy`.
4. Sample bins according to the learned weights.
5. Sample sequences within each selected bin for model pre-training.

## Citation

Comming Soon
