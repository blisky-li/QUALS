# Stage 3 GRPO Weight Optimization

这个文件夹保存 GRPO 生成 bin weight 的开源版训练代码。默认参数已经固定为当前希望使用的配置：

```text
start_iter = 5000
end_iter = 120000
rate_mode = progress_ratio

lr = 2e-05
epochs = 480
steps_per_epoch = 50

hidden_dim = 96
emb_init_std = 0.02
emb_lr_mult = 2.0
fixed_std_ratio = 0.02
learn_log_std = 1
log_std_init = -5
explore_ratio = 1.0

batch_size = 1000
batch_groups = 64
stage_batch = 64
```

## Files

- `Algorithm2grpo.py`: 主要优化脚本，读取每个 iteration 的 bin 指标，计算 `progress_ratio` rate，然后用 GRPO 学习每个 bin 的 weight。
- `weight_mlp.py`: 小型 MLP，将可学习的 bin embedding 映射为每个 bin 的 log-weight。
- `run_stage3_grpo.sh`: 一键启动脚本，默认使用 `CUDA_VISIBLE_DEVICES=1`，也可以在命令前覆盖。

## Run

默认运行：

```bash
cd <PROJECT_ROOT>/quals/stage3GRPO
bash run_stage3_grpo.sh
```

指定 GPU 和输出目录：

```bash
CUDA_VISIBLE_DEVICES=1 bash run_stage3_grpo.sh \
  --out_dir <STAGE3_OUTPUT_DIR>
```

训练结束后会保存：

- `stage3_grpo_weight.npy`: GRPO 学到的 bin sampling weight。
- `stage3_grpo_model.pt`: MLP 与 embedding 参数。
- `stage3_grpo_rate.npz`: 根据 `start_iter/end_iter/rate_mode` 计算出的 rate。
- `stage3_grpo_report.json`: weight 和 baseline 的相似性指标。

## Default Inputs

默认读取：

```text
metrics_dir = <GRPO_METRICS_LOG_DIR>
baseline_weights = <BASELINE_WEIGHT_PATH>
```

如果要替换输入，可以使用：

```bash
bash run_stage3_grpo.sh \
  --metrics_dir /path/to/metrics/logs \
  --baseline_weights /path/to/baseline_weight.npy
```
