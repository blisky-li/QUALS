import argparse
import json
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from weight_mlp import WeightMLP


DEFAULT_METRICS_DIR = "<GRPO_METRICS_LOG_DIR>"
DEFAULT_BASELINE = "<BASELINE_WEIGHT_PATH>"
DEFAULT_OUT_DIR = "<STAGE3_OUTPUT_DIR>"


def _load_array(path: str, key: Optional[str] = None) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"array path not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        data = np.load(path)
        if key:
            if key not in data:
                raise KeyError(f"key '{key}' not found in {path}, keys={list(data.keys())}")
            return data[key]
        if "arr_0" in data:
            return data["arr_0"]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError(f"npz has multiple arrays; specify --key. keys={list(data.keys())}")
    return np.loadtxt(path, delimiter=",")


def _flatten_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    return x.reshape(-1)


def _load_metrics_dir(
    metrics_dir: str,
    metrics_regex: str,
    metric_key: str,
    start_iter: int,
    end_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isdir(metrics_dir):
        raise NotADirectoryError(f"metrics_dir not found: {metrics_dir}")

    pattern = re.compile(metrics_regex)
    items = []
    for name in os.listdir(metrics_dir):
        match = pattern.match(name)
        if not match:
            continue
        try:
            iter_id = int(match.group(1))
        except Exception:
            continue
        if int(start_iter) <= iter_id <= int(end_iter):
            items.append((iter_id, os.path.join(metrics_dir, name)))

    if not items:
        raise ValueError(f"no metrics files matched regex in {metrics_dir}")

    items.sort(key=lambda x: x[0])
    iters = []
    metrics = []
    for iter_id, path in items:
        arr = _load_array(path, key=metric_key)
        metrics.append(_flatten_1d(arr).astype(np.float32))
        iters.append(iter_id)
    return np.asarray(iters, dtype=np.int64), np.stack(metrics, axis=0)


def _compute_rate(
    init_perf: np.ndarray,
    cur_perf: np.ndarray,
    last_perf: np.ndarray,
    init_iter: int,
    cur_iter: int,
    last_iter: int,
    mode: str,
    eps: float,
) -> np.ndarray:
    init_perf = _flatten_1d(init_perf).astype(np.float32)
    cur_perf = _flatten_1d(cur_perf).astype(np.float32)
    last_perf = _flatten_1d(last_perf).astype(np.float32)

    init_iter = int(init_iter)
    cur_iter = int(cur_iter)
    last_iter = int(last_iter)
    if not (init_iter < cur_iter < last_iter):
        raise ValueError(f"expect init_iter < cur_iter < last_iter, got {init_iter}, {cur_iter}, {last_iter}")

    dt1 = float(cur_iter - init_iter)
    dt2 = float(last_iter - cur_iter)
    dt_all = float(last_iter - init_iter)

    if mode == "ratio":
        s1 = (init_perf - cur_perf) / max(dt1, eps)
        s2 = (cur_perf - last_perf) / max(dt2, eps)
        rate = s1 / (s2 + eps)
    elif mode == "progress_ratio":
        prog_perf = (init_perf - cur_perf) / (init_perf - last_perf + eps)
        prog_iter = dt1 / max(dt_all, eps)
        rate = prog_perf / (prog_iter + eps)
    elif mode == "slope":
        rate = (init_perf - cur_perf) / max(dt1, eps)
    else:
        raise ValueError(f"unknown rate mode: {mode}")

    return np.maximum(rate, eps)


def _normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(weights.sum())
    if total > 0 and np.isfinite(total):
        weights = weights / total
    return weights


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _overlap(pred: np.ndarray, baseline: np.ndarray, frac: float = 0.1, bottom: bool = False) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    k = max(1, int(np.ceil(pred.size * frac)))
    if bottom:
        base_idx = set(np.argsort(baseline)[:k].tolist())
        pred_idx = set(np.argsort(pred)[:k].tolist())
    else:
        base_idx = set(np.argsort(-baseline)[:k].tolist())
        pred_idx = set(np.argsort(-pred)[:k].tolist())
    return len(base_idx & pred_idx) / float(k)


def _topk_mass_recall(pred: np.ndarray, baseline: np.ndarray, frac: float = 0.1) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    k = max(1, int(np.ceil(pred.size * frac)))
    base_top = np.argsort(-baseline)[:k]
    pred_top = set(np.argsort(-pred)[:k].tolist())
    inter = [idx for idx in base_top.tolist() if idx in pred_top]
    return float(baseline[inter].sum()) / max(float(baseline[base_top].sum()), 1e-12)


def _metrics(weights: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    weights = _normalize(weights)
    baseline = _normalize(baseline)
    return {
        "std": float(weights.std()),
        "baseline_std": float(baseline.std()),
        "std_abs_diff": float(abs(weights.std() - baseline.std())),
        "pearson": _pearson(weights, baseline),
        "top10_overlap": _overlap(weights, baseline, bottom=False),
        "bottom10_overlap": _overlap(weights, baseline, bottom=True),
        "topk_mass_recall": _topk_mass_recall(weights, baseline),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-3 GRPO bin-weight optimizer with tuned default hyperparameters."
    )

    parser.add_argument("--metrics_dir", default=DEFAULT_METRICS_DIR)
    parser.add_argument("--metrics_regex", default=r".*_(\d+)\.npz$")
    parser.add_argument("--metric_key", default="mean_bin")
    parser.add_argument("--metrics_start_iter", type=int, default=5000)
    parser.add_argument("--metrics_end_iter", type=int, default=120000)
    parser.add_argument("--baseline_weights", default=DEFAULT_BASELINE)

    parser.add_argument("--num_bins", type=int, default=2499)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--emb_init_std", type=float, default=0.02)
    parser.add_argument("--emb_lr_mult", type=float, default=2.0)
    parser.add_argument("--freeze_emb", type=int, choices=[0, 1], default=0)

    parser.add_argument("--rate_mode", choices=["ratio", "progress_ratio", "slope"], default="progress_ratio")
    parser.add_argument("--rate_eps", type=float, default=1e-6)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--min_weight", type=float, default=1e-6)

    parser.add_argument("--batch_size", type=int, default=1000, help="Number of bins sampled per group.")
    parser.add_argument("--batch_groups", type=int, default=64)
    parser.add_argument("--stage_batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=480)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--target_mode", choices=["mean_rate", "median_rate", "const"], default="mean_rate")
    parser.add_argument("--target_const", type=float, default=1.0)
    parser.add_argument("--loss_mode", choices=["grpo", "mse"], default="grpo")
    parser.add_argument("--learn_log_std", type=int, choices=[0, 1], default=1)
    parser.add_argument("--log_std_init", type=float, default=-5.0)
    parser.add_argument("--fixed_std_ratio", type=float, default=0.02)
    parser.add_argument("--log_std_min", type=float, default=-10.0)
    parser.add_argument("--log_std_max", type=float, default=10.0)
    parser.add_argument("--grpo_adv_eps", type=float, default=1e-6)
    parser.add_argument("--grpo_clip_eps", type=float, default=0.2)
    parser.add_argument("--explore_ratio", type=float, default=1.0)

    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out_weights", default=None)
    parser.add_argument("--save_model", default=None)
    parser.add_argument("--save_rate", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_weights is None:
        args.out_weights = os.path.join(args.out_dir, "stage3_grpo_weight.npy")
    if args.save_model is None:
        args.save_model = os.path.join(args.out_dir, "stage3_grpo_model.pt")
    if args.save_rate is None:
        args.save_rate = os.path.join(args.out_dir, "stage3_grpo_rate.npz")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    iters, metrics = _load_metrics_dir(
        args.metrics_dir,
        metrics_regex=args.metrics_regex,
        metric_key=args.metric_key,
        start_iter=args.metrics_start_iter,
        end_iter=args.metrics_end_iter,
    )
    init_iter = int(iters[0])
    last_iter = int(iters[-1])
    init_perf = metrics[0]
    last_perf = metrics[-1]
    mid_mask = (iters > init_iter) & (iters < last_iter)
    stage_iters = iters[mid_mask]
    stage_metrics = metrics[mid_mask]
    if stage_iters.size == 0:
        raise ValueError("no middle stages found between init and last.")

    metrics_bins = int(stage_metrics.shape[1])
    num_bins = min(int(args.num_bins), metrics_bins)
    if int(args.num_bins) != metrics_bins:
        print(f"[warn] args.num_bins={args.num_bins}, metrics_bins={metrics_bins}. Using {num_bins}.")
    stage_metrics = stage_metrics[:, :num_bins]
    init_perf = init_perf[:num_bins]
    last_perf = last_perf[:num_bins]

    rates = []
    for cur_iter, cur_perf in zip(stage_iters.tolist(), stage_metrics):
        rates.append(
            _compute_rate(
                init_perf,
                cur_perf,
                last_perf,
                init_iter=init_iter,
                cur_iter=int(cur_iter),
                last_iter=last_iter,
                mode=args.rate_mode,
                eps=args.rate_eps,
            ).astype(np.float32)
        )
    rates = np.stack(rates, axis=0)
    np.savez(args.save_rate, iters=stage_iters, rates=rates)
    print(f"[info] init_iter={init_iter} last_iter={last_iter} stages={rates.shape[0]} bins={rates.shape[1]}")

    device = args.device
    rate_t = torch.tensor(rates, dtype=torch.float32, device=device)
    num_stages = int(rate_t.shape[0])

    emb_param = torch.nn.Parameter(
        torch.randn((num_bins, int(args.emb_dim)), device=device, dtype=torch.float32) * float(args.emb_init_std)
    )
    if int(args.freeze_emb) == 1:
        emb_param.requires_grad_(False)

    if args.loss_mode == "grpo":
        out_activation = "none"
        model_min_weight = 0.0
    else:
        out_activation = "softplus"
        model_min_weight = args.min_weight

    model = WeightMLP(
        in_dim=int(args.emb_dim),
        hidden_dim=int(args.hidden_dim),
        out_activation=out_activation,
        min_weight=model_min_weight,
    ).to(device)

    params = [{"params": model.parameters(), "lr": float(args.lr)}]
    if emb_param.requires_grad:
        params.append({"params": [emb_param], "lr": float(args.lr) * float(args.emb_lr_mult)})

    if args.loss_mode == "grpo":
        if int(args.learn_log_std) == 1:
            log_std = torch.nn.Parameter(torch.tensor(float(args.log_std_init), device=device, dtype=torch.float32))
            params.append({"params": [log_std], "lr": float(args.lr)})
        else:
            if args.fixed_std_ratio <= 0:
                raise ValueError("--fixed_std_ratio must be > 0 when --learn_log_std=0.")
            log_std = torch.tensor(float(np.log(1.0 + float(args.fixed_std_ratio))), device=device, dtype=torch.float32)
    else:
        log_std = None

    optimizer = torch.optim.AdamW(params, weight_decay=float(args.weight_decay))

    if args.target_mode == "mean_rate":
        target_t = rate_t.mean(dim=1)
    elif args.target_mode == "median_rate":
        target_t = rate_t.median(dim=1).values
    else:
        target_t = torch.full((num_stages,), float(args.target_const), device=device)

    batch_size = min(int(args.batch_size), int(num_bins))
    batch_groups = max(int(args.batch_groups), 1)
    stage_batch = min(max(int(args.stage_batch), 1), int(num_stages))
    explore_ratio = max(0.0, min(1.0, float(args.explore_ratio)))

    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_prod_var = 0.0

        for _ in range(int(args.steps_per_epoch)):
            stage_idx = torch.randint(0, num_stages, (stage_batch,), device=device)
            idx = torch.randint(0, num_bins, (stage_batch, batch_groups, batch_size), device=device)
            flat_idx = idx.reshape(-1)

            batch_emb = emb_param[flat_idx]
            pred_out = model(batch_emb).reshape(stage_batch, batch_groups, batch_size)

            rate_stage = rate_t[stage_idx]
            rate_sel = rate_stage.gather(1, idx.reshape(stage_batch, -1)).reshape(stage_batch, batch_groups, batch_size)
            target_stage = target_t[stage_idx].reshape(stage_batch, 1, 1)

            if args.loss_mode == "grpo":
                log_mu = pred_out
                log_std_clamped = torch.clamp(log_std, args.log_std_min, args.log_std_max)
                std = torch.exp(log_std_clamped)

                explore_mask = (torch.rand_like(log_mu) < explore_ratio).to(log_mu.dtype)
                eps = torch.randn_like(log_mu)
                logw = log_mu + std * eps * explore_mask
                w = torch.exp(logw) + float(args.min_weight)

                prod = w * rate_sel
                mean_prod = prod.mean(dim=2, keepdim=True)
                reward = -((prod - mean_prod) ** 2)
                adv = (reward - reward.mean(dim=2, keepdim=True)) / (
                    reward.std(dim=2, keepdim=True) + float(args.grpo_adv_eps)
                )

                logw_det = logw.detach()
                log_mu_old = log_mu.detach()
                log_std_old = log_std_clamped.detach()
                std_old = std.detach()
                logp_old = (
                    -logw_det
                    - log_std_old
                    - 0.5 * np.log(2 * np.pi)
                    - 0.5 * ((logw_det - log_mu_old) / std_old) ** 2
                )
                logp = (
                    -logw_det
                    - log_std_clamped
                    - 0.5 * np.log(2 * np.pi)
                    - 0.5 * ((logw_det - log_mu) / std) ** 2
                )
                ratio = torch.exp(logp - logp_old)
                adv_masked = adv * explore_mask
                denom = torch.clamp(explore_mask.sum(), min=1.0)
                surr1 = ratio * adv_masked
                surr2 = torch.clamp(ratio, 1.0 - args.grpo_clip_eps, 1.0 + args.grpo_clip_eps) * adv_masked
                loss = -torch.min(surr1, surr2).sum() / denom

                w_det = torch.exp(log_mu) + float(args.min_weight)
                prod_det = w_det * rate_sel
                mse_metric = torch.mean(torch.mean((prod_det - target_stage) ** 2, dim=2))
                prod_for_stats = prod_det
            else:
                pred_w = pred_out
                prod_det = pred_w * rate_sel
                loss = torch.mean(torch.mean((prod_det - target_stage) ** 2, dim=2))
                mse_metric = loss
                prod_for_stats = prod_det

            prod_flat = prod_for_stats.reshape(-1)
            prod_var = torch.mean((prod_flat - prod_flat.mean()) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_mse += float(mse_metric.item())
            running_prod_var += float(prod_var.item())

        loss_avg = running_loss / max(int(args.steps_per_epoch), 1)
        mse_avg = running_mse / max(int(args.steps_per_epoch), 1)
        prod_var_avg = running_prod_var / max(int(args.steps_per_epoch), 1)
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"loss={loss_avg:.6f} mse_metric={mse_avg:.6f} prod_var={prod_var_avg:.6f}"
        )

    model.eval()
    with torch.no_grad():
        if args.loss_mode == "grpo":
            raw_t = torch.exp(model(emb_param)) + float(args.min_weight)
        else:
            raw_t = model(emb_param)
        learned_weights = raw_t.detach().cpu().numpy().astype(np.float32).reshape(-1)

    np.save(args.out_weights, learned_weights)
    print(f"[saved] weights: {args.out_weights}")

    ckpt = {
        "state_dict": model.state_dict(),
        "emb_param": emb_param.detach().cpu(),
        "config": vars(args),
        "init_iter": int(init_iter),
        "last_iter": int(last_iter),
        "stage_iters": stage_iters.tolist(),
        "log_std": float(log_std.detach().cpu().item()) if isinstance(log_std, torch.Tensor) else None,
    }
    torch.save(ckpt, args.save_model)
    print(f"[saved] model: {args.save_model}")

    baseline = _load_array(args.baseline_weights).reshape(-1)[: learned_weights.size]
    report = {
        "weights": args.out_weights,
        "metrics": _metrics(learned_weights, baseline),
        "config": vars(args),
    }
    report_path = os.path.join(args.out_dir, "stage3_grpo_report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[saved] report: {report_path}")


if __name__ == "__main__":
    main()
