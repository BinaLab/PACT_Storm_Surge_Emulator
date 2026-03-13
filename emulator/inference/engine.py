"""Inference execution helpers."""

from __future__ import annotations

import time

import numpy as np
import torch

from emulator.data.normalization import normalize_inputs_inplace


@torch.no_grad()
def infer_one_loader(
    model,
    loader,
    device,
    x_center,
    x_scale,
    x_clip,
    pmean_center,
    pmean_scale,
    pmean_clip,
    y_mean,
    y_std,
    use_amp: bool,
    amp_dtype,
    model_name: str,
    station_feat: torch.Tensor | None,
    gpu_sync_timing: bool,
):
    """Run inference on one loader and return metrics, runtime, and predictions."""
    model.eval()

    sum_sq = 0.0
    sum_abs = 0.0
    n_items = 0

    y_true_all = []
    y_pred_all = []
    tags_all = []

    if gpu_sync_timing and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

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
            pmean_center=pmean_center,
            pmean_scale=pmean_scale,
            pmean_clip=pmean_clip,
        )

        y_true = batch.y.float()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                if model_name == "perceiver3":
                    pred_norm = model(batch, station_feat=station_feat)
                else:
                    pred_norm = model(batch)
        else:
            if model_name == "perceiver3":
                pred_norm = model(batch, station_feat=station_feat)
            else:
                pred_norm = model(batch)

        y_pred = pred_norm.float() * y_std + y_mean

        err = y_pred - y_true
        sum_sq += float((err**2).sum().item())
        sum_abs += float(err.abs().sum().item())
        n_items += int(err.numel())

        y_true_all.append(y_true.detach().cpu())
        y_pred_all.append(y_pred.detach().cpu())

    if gpu_sync_timing and device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    rmse = float(np.sqrt(sum_sq / max(1, n_items)))
    mae = float(sum_abs / max(1, n_items))

    y_true_cat = torch.cat(y_true_all, dim=0).numpy() if y_true_all else np.zeros((0, y_mean.numel()), np.float32)
    y_pred_cat = torch.cat(y_pred_all, dim=0).numpy() if y_pred_all else np.zeros((0, y_mean.numel()), np.float32)
    tags = np.array(tags_all, dtype=object)

    return rmse, mae, dt, y_true_cat, y_pred_cat, tags


__all__ = ["infer_one_loader"]
