"""Training and evaluation loops.

This module intentionally keeps optimization and metric computation logic in one
place so train/infer behavior remains consistent across refactors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from emulator.common.distributed import ddp_all_reduce_sum
from emulator.data.loaders import build_loader
from emulator.data.normalization import normalize_inputs_inplace, normalize_targets
from emulator.training.losses import slope_matching_loss_softmask, weighted_mse_loss


def _is_perceiver_family(model_name: str) -> bool:
    return str(model_name) in ("perceiver3", "perceiver3_cnn")


def train_one_epoch_ddp(
    model,
    loader,
    optimizer,
    device,
    use_amp=False,
    amp_dtype=torch.bfloat16,
    scaler=None,
    x_center=None,
    x_scale=None,
    x_clip: float = 5.0,
    pmean_center: torch.Tensor | None = None,
    pmean_scale: torch.Tensor | None = None,
    pmean_clip: float | None = None,
    do_x_aug: bool = False,
    x_aug_prob: float = 1.0,
    x_aug_scale: float = 0.05,
    x_aug_bias: float = 0.02,
    y_mean=None,
    y_std=None,
    model_name: str = "baseline",
    station_feat: torch.Tensor | None = None,
    loss_mode: str = "mse",
    wmse_q_t: torch.Tensor | None = None,
    tail_peak_thr_t: torch.Tensor | None = None,
    wmse_alpha: float = 4.0,
    wmse_s: float = 0.10,
    wmse_use_abs: bool = True,
    tail_lambda: float = 0.10,
    slope_lambda: float = 0.01,
    slope_mask_s: float = 0.10,
    slope_robust: str = "charb",
    slope_charb_eps: float = 1e-3,
    slope_huber_delta: float = 0.05,
):
    """Run one training epoch and return normalized + physical metrics."""
    model.train()
    criterion = nn.MSELoss()

    total_mse_norm_sum = 0.0
    total_mse_phys_sum = 0.0
    total_mae_phys_sum = 0.0
    total_samples = 0

    gate_sum = 0.0
    gate_count = 0

    use_slope = str(loss_mode).endswith("_slope")
    core_loss_mode = str(loss_mode)[:-6] if use_slope else str(loss_mode)

    base_use_wmse = core_loss_mode in ("wmse", "wmse_tail")
    tail_use_wmse = core_loss_mode in ("wmse_tail", "mse_wtail")
    use_tail = core_loss_mode in ("mse_tail", "wmse_tail", "mse_wtail")

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        normalize_inputs_inplace(
            batch,
            x_center=x_center,
            x_scale=x_scale,
            x_clip=float(x_clip),
            do_aug=bool(do_x_aug),
            aug_prob=float(x_aug_prob),
            aug_scale=float(x_aug_scale),
            aug_bias=float(x_aug_bias),
            pmean_center=pmean_center,
            pmean_scale=pmean_scale,
            pmean_clip=pmean_clip,
        )

        y_raw = batch.y.float()
        y_norm = normalize_targets(y_raw, y_mean, y_std)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                if _is_perceiver_family(model_name):
                    pred_norm = model(batch, station_feat=station_feat, return_aux=False)
                else:
                    pred_norm = model(batch)

            pred_phys = pred_norm.float() * y_std + y_mean

            if base_use_wmse:
                base_loss = weighted_mse_loss(
                    pred=pred_phys,
                    target=y_raw,
                    q_value=wmse_q_t,
                    alpha=wmse_alpha,
                    s=wmse_s,
                    use_abs=wmse_use_abs,
                )
            else:
                base_loss = criterion(pred_phys, y_raw)

            tail_loss = pred_phys.new_zeros(())
            if use_tail:
                peak_score = y_raw.max(dim=1).values
                m = peak_score >= tail_peak_thr_t
                if m.any():
                    if tail_use_wmse:
                        tail_loss = weighted_mse_loss(
                            pred=pred_phys[m],
                            target=y_raw[m],
                            q_value=wmse_q_t,
                            alpha=wmse_alpha,
                            s=wmse_s,
                            use_abs=wmse_use_abs,
                        )
                    else:
                        tail_loss = criterion(pred_phys[m], y_raw[m])

            slope_loss = pred_phys.new_zeros(())
            if use_slope:
                if wmse_q_t is None:
                    raise ValueError("wmse_q_t is required for *_slope loss modes (used as tau for the soft mask).")
                slope_loss = slope_matching_loss_softmask(
                    pred=pred_phys,
                    target=y_raw,
                    tau=wmse_q_t,
                    mask_s=float(slope_mask_s),
                    robust=str(slope_robust),
                    charb_eps=float(slope_charb_eps),
                    huber_delta=float(slope_huber_delta),
                )

            loss = base_loss + float(tail_lambda) * tail_loss + float(slope_lambda) * slope_loss

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            if _is_perceiver_family(model_name):
                pred_norm = model(batch, station_feat=station_feat, return_aux=False)
            else:
                pred_norm = model(batch)

            pred_phys = pred_norm.float() * y_std + y_mean

            if base_use_wmse:
                base_loss = weighted_mse_loss(
                    pred=pred_phys,
                    target=y_raw,
                    q_value=wmse_q_t,
                    alpha=wmse_alpha,
                    s=wmse_s,
                    use_abs=wmse_use_abs,
                )
            else:
                base_loss = criterion(pred_phys, y_raw)

            tail_loss = pred_phys.new_zeros(())
            if use_tail:
                peak_score = y_raw.max(dim=1).values
                m = peak_score >= tail_peak_thr_t
                if m.any():
                    if tail_use_wmse:
                        tail_loss = weighted_mse_loss(
                            pred=pred_phys[m],
                            target=y_raw[m],
                            q_value=wmse_q_t,
                            alpha=wmse_alpha,
                            s=wmse_s,
                            use_abs=wmse_use_abs,
                        )
                    else:
                        tail_loss = criterion(pred_phys[m], y_raw[m])

            slope_loss = pred_phys.new_zeros(())
            if use_slope:
                if wmse_q_t is None:
                    raise ValueError("wmse_q_t is required for *_slope loss modes (used as tau for the soft mask).")
                slope_loss = slope_matching_loss_softmask(
                    pred=pred_phys,
                    target=y_raw,
                    tau=wmse_q_t,
                    mask_s=float(slope_mask_s),
                    robust=str(slope_robust),
                    charb_eps=float(slope_charb_eps),
                    huber_delta=float(slope_huber_delta),
                )

            loss = base_loss + float(tail_lambda) * tail_loss + float(slope_lambda) * slope_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            err_norm = pred_norm.detach().float() - y_norm.detach().float()
            mse_norm = (err_norm**2).mean()

            err_phys = pred_phys.detach() - y_raw.detach()
            mse_phys = (err_phys**2).mean()
            mae_phys = err_phys.abs().mean()

        bs = batch.num_graphs
        total_mse_norm_sum += float(mse_norm.item()) * bs
        total_mse_phys_sum += float(mse_phys.item()) * bs
        total_mae_phys_sum += float(mae_phys.item()) * bs
        total_samples += bs

        if _is_perceiver_family(model_name):
            with torch.no_grad():
                _, aux = model(batch, station_feat=station_feat, return_aux=True)
                g = aux.get("gate_mean_per_sample", None)
                if g is not None:
                    gate_sum += float(g.mean().item()) * bs
                    gate_count += bs

    mse_norm_sum_t = torch.tensor(total_mse_norm_sum, device=device, dtype=torch.float64)
    mse_phys_sum_t = torch.tensor(total_mse_phys_sum, device=device, dtype=torch.float64)
    mae_phys_sum_t = torch.tensor(total_mae_phys_sum, device=device, dtype=torch.float64)
    samples_t = torch.tensor(total_samples, device=device, dtype=torch.float64)

    ddp_all_reduce_sum(mse_norm_sum_t)
    ddp_all_reduce_sum(mse_phys_sum_t)
    ddp_all_reduce_sum(mae_phys_sum_t)
    ddp_all_reduce_sum(samples_t)

    mse_norm = (mse_norm_sum_t / samples_t).item()
    mse_phys = (mse_phys_sum_t / samples_t).item()
    mae_phys = (mae_phys_sum_t / samples_t).item()
    rmse_phys = float(np.sqrt(mse_phys))

    gate_mean_train = None
    if _is_perceiver_family(model_name) and gate_count > 0:
        gate_mean_train = gate_sum / gate_count

    return mse_norm, mse_phys, rmse_phys, mae_phys, gate_mean_train


@torch.no_grad()
def evaluate_ddp(
    model,
    loader,
    device,
    use_amp=False,
    amp_dtype=torch.bfloat16,
    x_center=None,
    x_scale=None,
    x_clip: float = 5.0,
    pmean_center: torch.Tensor | None = None,
    pmean_scale: torch.Tensor | None = None,
    pmean_clip: float | None = None,
    y_mean=None,
    y_std=None,
    model_name: str = "baseline",
    station_feat: torch.Tensor | None = None,
):
    """Evaluate a model on one loader and return normalized + physical metrics."""
    model.eval()

    total_mse_norm_sum = 0.0
    total_mse_phys_sum = 0.0
    total_mae_phys_sum = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        normalize_inputs_inplace(
            batch,
            x_center=x_center,
            x_scale=x_scale,
            x_clip=float(x_clip),
            do_aug=False,
            pmean_center=pmean_center,
            pmean_scale=pmean_scale,
            pmean_clip=pmean_clip,
        )

        y_raw = batch.y.float()
        y_norm = normalize_targets(y_raw, y_mean, y_std)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                if _is_perceiver_family(model_name):
                    pred_norm = model(batch, station_feat=station_feat, return_aux=False)
                else:
                    pred_norm = model(batch)
            pred_phys = pred_norm.float() * y_std + y_mean
        else:
            if _is_perceiver_family(model_name):
                pred_norm = model(batch, station_feat=station_feat, return_aux=False)
            else:
                pred_norm = model(batch)
            pred_phys = pred_norm.float() * y_std + y_mean

        err_norm = pred_norm.detach().float() - y_norm.detach().float()
        mse_norm = (err_norm**2).mean()

        err_phys = pred_phys.detach() - y_raw.detach()
        mse_phys = (err_phys**2).mean()
        mae_phys = err_phys.abs().mean()

        bs = batch.num_graphs
        total_mse_norm_sum += float(mse_norm.item()) * bs
        total_mse_phys_sum += float(mse_phys.item()) * bs
        total_mae_phys_sum += float(mae_phys.item()) * bs
        total_samples += bs

    mse_norm_sum_t = torch.tensor(total_mse_norm_sum, device=device, dtype=torch.float64)
    mse_phys_sum_t = torch.tensor(total_mse_phys_sum, device=device, dtype=torch.float64)
    mae_phys_sum_t = torch.tensor(total_mae_phys_sum, device=device, dtype=torch.float64)
    samples_t = torch.tensor(total_samples, device=device, dtype=torch.float64)

    ddp_all_reduce_sum(mse_norm_sum_t)
    ddp_all_reduce_sum(mse_phys_sum_t)
    ddp_all_reduce_sum(mae_phys_sum_t)
    ddp_all_reduce_sum(samples_t)

    mse_norm = (mse_norm_sum_t / samples_t).item()
    mse_phys = (mse_phys_sum_t / samples_t).item()
    mae_phys = (mae_phys_sum_t / samples_t).item()
    rmse_phys = float(np.sqrt(mse_phys))
    return mse_norm, mse_phys, rmse_phys, mae_phys


@torch.no_grad()
def collect_test_preds_unnorm(
    model_raw,
    loader,
    device,
    use_amp=False,
    amp_dtype=torch.bfloat16,
    x_center=None,
    x_scale=None,
    x_clip: float = 5.0,
    pmean_center: torch.Tensor | None = None,
    pmean_scale: torch.Tensor | None = None,
    pmean_clip: float | None = None,
    y_mean=None,
    y_std=None,
    model_name: str = "baseline",
    station_feat: torch.Tensor | None = None,
):
    """Collect denormalized predictions/targets plus sample tags."""
    model_raw.eval()
    y_true_all = []
    y_pred_all = []
    tags_all = []

    for batch in loader:
        tags = getattr(batch, "tag", None)
        if tags is None:
            tags = [""] * batch.num_graphs
        tags_all.extend(list(tags))

        batch = batch.to(device, non_blocking=True)
        normalize_inputs_inplace(
            batch,
            x_center=x_center,
            x_scale=x_scale,
            x_clip=float(x_clip),
            do_aug=False,
            pmean_center=pmean_center,
            pmean_scale=pmean_scale,
            pmean_clip=pmean_clip,
        )

        y_true_raw = batch.y.float()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                if _is_perceiver_family(model_name):
                    pred_norm = model_raw(batch, station_feat=station_feat, return_aux=False)
                else:
                    pred_norm = model_raw(batch)
        else:
            if _is_perceiver_family(model_name):
                pred_norm = model_raw(batch, station_feat=station_feat, return_aux=False)
            else:
                pred_norm = model_raw(batch)

        pred_phys = pred_norm.detach().float() * y_std + y_mean

        y_true_all.append(y_true_raw.detach().cpu())
        y_pred_all.append(pred_phys.detach().cpu())

    y_true = torch.cat(y_true_all, dim=0).numpy() if len(y_true_all) else np.zeros((0, y_mean.numel()), dtype=np.float32)
    y_pred = torch.cat(y_pred_all, dim=0).numpy() if len(y_pred_all) else np.zeros((0, y_mean.numel()), dtype=np.float32)
    tags = np.array(tags_all, dtype=object)
    return y_true, y_pred, tags


@torch.no_grad()
def eval_full_metrics_and_logs(
    model_raw,
    dataset,
    device,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    mp_context,
    use_amp,
    amp_dtype,
    x_center,
    x_scale,
    x_clip: float,
    pmean_center: torch.Tensor | None,
    pmean_scale: torch.Tensor | None,
    pmean_clip: float | None,
    y_mean,
    y_std,
    model_name: str,
    station_feat: torch.Tensor | None,
    peak_frac: float = 0.05,
):
    """Compute full-pass validation metrics and interpretable aux logs."""
    loader_full = build_loader(
        dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        mp_context=mp_context,
    )

    model_raw.eval()

    per_sample_peak = []
    per_sample_mse = []
    per_sample_mae = []

    aux_gate = []
    aux_node_max = []
    aux_node_ent = []
    aux_time_recent = []

    for batch in loader_full:
        batch = batch.to(device, non_blocking=True)
        normalize_inputs_inplace(
            batch,
            x_center=x_center,
            x_scale=x_scale,
            x_clip=float(x_clip),
            do_aug=False,
            pmean_center=pmean_center,
            pmean_scale=pmean_scale,
            pmean_clip=pmean_clip,
        )

        y_true = batch.y.float()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                if _is_perceiver_family(model_name):
                    pred_norm, aux = model_raw(batch, station_feat=station_feat, return_aux=True)
                else:
                    pred_norm = model_raw(batch)
                    aux = {}
        else:
            if _is_perceiver_family(model_name):
                pred_norm, aux = model_raw(batch, station_feat=station_feat, return_aux=True)
            else:
                pred_norm = model_raw(batch)
                aux = {}

        y_pred = pred_norm.detach().float() * y_std + y_mean

        peak_score = y_true.max(dim=1).values
        err = y_pred - y_true
        mse_i = (err**2).mean(dim=1)
        mae_i = err.abs().mean(dim=1)

        per_sample_peak.append(peak_score.detach().cpu())
        per_sample_mse.append(mse_i.detach().cpu())
        per_sample_mae.append(mae_i.detach().cpu())

        if "gate_mean_per_sample" in aux:
            aux_gate.append(aux["gate_mean_per_sample"].detach().float().cpu())

        if _is_perceiver_family(model_name):
            aux_node_max.append(aux["node_attn_max_per_sample"].detach().float().cpu())
            aux_node_ent.append(aux["node_attn_entropy_per_sample"].detach().float().cpu())
            aux_time_recent.append(aux["time_attn_recent_per_sample"].detach().float().cpu())

    peak_score = torch.cat(per_sample_peak, dim=0).numpy()
    mse_i = torch.cat(per_sample_mse, dim=0).numpy()
    mae_i = torch.cat(per_sample_mae, dim=0).numpy()

    rmse_all = float(np.sqrt(mse_i.mean())) if mse_i.size else float("nan")
    mae_all = float(mae_i.mean()) if mae_i.size else float("nan")

    n = peak_score.size
    k = max(1, int(np.ceil(float(peak_frac) * n)))
    idx_sorted = np.argsort(peak_score)
    peak_idx = idx_sorted[-k:]

    rmse_peak = float(np.sqrt(mse_i[peak_idx].mean())) if n else float("nan")
    mae_peak = float(mae_i[peak_idx].mean()) if n else float("nan")

    def _mean_over(mask_idx, xs_list):
        if not xs_list:
            return None
        t = torch.cat(xs_list, dim=0)
        arr = t.detach().to(dtype=torch.float32).cpu().numpy()
        return float(arr[mask_idx].mean()) if arr.size else None

    all_idx = np.arange(n, dtype=np.int64)

    logs = dict(
        rmse_all=rmse_all,
        mae_all=mae_all,
        rmse_peak=rmse_peak,
        mae_peak=mae_peak,
    )

    if _is_perceiver_family(model_name):
        logs["gate_mean_all"] = _mean_over(all_idx, aux_gate)
        logs["gate_mean_peak"] = _mean_over(peak_idx, aux_gate)
        logs["node_attn_max_all"] = _mean_over(all_idx, aux_node_max)
        logs["node_attn_max_peak"] = _mean_over(peak_idx, aux_node_max)
        logs["node_attn_entropy_all"] = _mean_over(all_idx, aux_node_ent)
        logs["node_attn_entropy_peak"] = _mean_over(peak_idx, aux_node_ent)
        logs["time_attn_recent_all"] = _mean_over(all_idx, aux_time_recent)
        logs["time_attn_recent_peak"] = _mean_over(peak_idx, aux_time_recent)

    return logs


__all__ = [
    "train_one_epoch_ddp",
    "evaluate_ddp",
    "collect_test_preds_unnorm",
    "eval_full_metrics_and_logs",
]
