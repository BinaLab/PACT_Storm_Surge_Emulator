#!/usr/bin/env python3
"""Storm Surge Emulator inference endpoint.

This file keeps only endpoint orchestration (`main`) and relies on modular
implementations under `emulator/` for data handling, model definitions, and
inference-time execution utilities.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from emulator.common import infer_dataset_tag
from emulator.data import (
    ForcingGraphStore,
    ForcingGraphView,
    _as_feature_vector,
    _try_load_station_json,
    build_loader,
    make_all_years_test_indices,
    make_year_split_indices,
    station_features_from_json,
)
from emulator.inference import classify_past_future, infer_one_loader, parse_year_tag
from emulator.models import (
    PACT,
    SpatialOnlyGraphSAGEBatch,
    SpatioTemporalGraphSAGEBatch,
    canonical_head_type,
    canonical_temporal_block,
)


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*",
)

def main():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--ckpt", required=True, help="Path to .pth checkpoint from train.py")
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Train/val root used for year-split test if --test_root_dir is empty",
    )

    # optional test root
    parser.add_argument("--test_root_dir", type=str, default="", help="If set: use ALL years from this root (e.g., CMIP6)")

    # station
    parser.add_argument("--station", type=str, default=None)
    parser.add_argument("--station_json_dir", type=str, default="./station_json")
    parser.add_argument(
        "--strict_station_test",
        action="store_true",
        help="If station not found in external test store, raise instead of using ALL",
    )

    # overrides
    parser.add_argument("--model", type=str, default="", choices=["", "baseline", "perceiver3"])
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="",
        choices=["", "GraphSAGE", "CNN"],
        help=(
            "Expected checkpoint encoder. Empty reads checkpoint args; legacy checkpoints "
            "default to GraphSAGE."
        ),
    )
    parser.add_argument(
        "--temporal_block",
        type=canonical_temporal_block,
        default=None,
        choices=["MLP", "LSTM", "GRU", "Transformer"],
        help=(
            "Expected PACT temporal block. Empty reads checkpoint args; legacy checkpoints "
            "default to Transformer. 'attn' is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--head_type",
        type=canonical_head_type,
        default=None,
        choices=["single", "dual"],
        help=(
            "Expected PACT prediction head. Empty reads checkpoint args; legacy checkpoints "
            "default to dual."
        ),
    )
    parser.add_argument("--history_hours", type=int, default=-1, help="Override history_hours; -1 uses ckpt args")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--torch_threads", type=int, default=1)

    # dataloader
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--mp_context", type=str, default="fork", choices=["fork", "spawn"])

    # amp
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # timing / outputs
    parser.add_argument("--gpu_sync_timing", action="store_true", help="Synchronize CUDA for accurate timing")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument(
        "--model_label",
        type=str,
        default="",
        help=(
            "Human-readable model/checkpoint label used in automatic run folder names "
            "(for example, P3_Best). Empty infers it from the checkpoint filename."
        ),
    )
    parser.add_argument(
        "--inference_results_root",
        type=str,
        default="./All_Inference_Results",
        help="Parent directory used when --out_dir is omitted.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help=(
            "Exact directory for metrics and predictions. When omitted, infer.py creates "
            "All_Inference_Results/<Station>_<ModelLabel>_<Source>_To_<Target>_<timestamp>/outputs/."
        ),
    )

    # optional subset of years (comma-separated), avoids hardcoding
    # If not set, we evaluate all available years in the selected test set.
    parser.add_argument(
        "--years",
        type=str,
        default="",
        help="Optional comma-separated list of year tags to evaluate (e.g., '1979_1980,1980_1981'). Default: all years.",
    )

    args = parser.parse_args()

    # threads
    os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
    torch.set_num_threads(args.torch_threads)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tf32
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_amp = bool(args.amp and device.type == "cuda")

    # load checkpoint (CPU)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}

    # resolve model + history from ckpt unless overridden
    model_name = args.model if args.model else str(ckpt_args.get("model", "baseline"))
    checkpoint_encoder_type = str(ckpt_args.get("encoder_type") or "GraphSAGE")
    encoder_type = args.encoder_type if args.encoder_type else checkpoint_encoder_type
    num_layers = int(ckpt_args.get("num_layers", 2))
    if encoder_type == "GraphSAGE" and num_layers < 2:
        raise ValueError(f"GraphSAGE encoder requires num_layers >= 2, got {num_layers} in checkpoint args.")
    if args.encoder_type and args.encoder_type != checkpoint_encoder_type:
        raise ValueError(
            "Encoder override does not match the checkpoint: "
            f"checkpoint={checkpoint_encoder_type!r}, requested={args.encoder_type!r}. "
            "GraphSAGE and CNN have different parameter layouts; use a checkpoint trained "
            "with the requested encoder."
        )
    checkpoint_temporal_block = canonical_temporal_block(
        ckpt_args.get("temporal_block") or "Transformer"
    )
    temporal_block = args.temporal_block or checkpoint_temporal_block
    if args.temporal_block and args.temporal_block != checkpoint_temporal_block:
        raise ValueError(
            "Temporal block override does not match the checkpoint: "
            f"checkpoint={checkpoint_temporal_block!r}, requested={args.temporal_block!r}. "
            "Use a checkpoint trained with the requested temporal block."
        )
    checkpoint_head_type = canonical_head_type(ckpt_args.get("head_type") or "dual")
    head_type = args.head_type or checkpoint_head_type
    if model_name == "perceiver3" and args.head_type and args.head_type != checkpoint_head_type:
        raise ValueError(
            "Head type override does not match the checkpoint: "
            f"checkpoint={checkpoint_head_type!r}, requested={args.head_type!r}. "
            "Use a checkpoint trained with the requested prediction head."
        )
    history_hours = args.history_hours if args.history_hours >= 0 else int(ckpt_args.get("history_hours", 0))
    if history_hours % 6 != 0:
        raise ValueError(f"history_hours must be multiple of 6, got {history_hours}")
    history_steps = history_hours // 6

    # station: prefer CLI; else fallback to ckpt args if present
    station_key = args.station
    if not station_key:
        station_key = ckpt_args.get("station", None) or ckpt_args.get("filter", None)
    if station_key == "":
        station_key = None

    # Unified inference artifact layout. NCEP remains `NCEP`; CMIP6 dataset
    # directory names retain their `CMIP6_` prefix.
    source_tag = infer_dataset_tag(args.root_dir) or "data"
    target_root = args.test_root_dir.strip() or args.root_dir
    target_tag = infer_dataset_tag(target_root) or "data"
    station_tag = station_key or "ALL"

    model_label = args.model_label.strip()
    if not model_label:
        checkpoint_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        expected_prefix = f"{source_tag}_{station_tag}_"
        if checkpoint_stem.startswith(expected_prefix):
            model_label = checkpoint_stem[len(expected_prefix):]
        else:
            station_marker = f"_{station_tag}_"
            model_label = (
                checkpoint_stem.split(station_marker, 1)[1]
                if station_marker in checkpoint_stem
                else model_name
            )
    model_label = model_label or model_name
    args.model_label = model_label

    if args.out_dir:
        args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    else:
        results_root = os.path.abspath(os.path.expanduser(args.inference_results_root))
        run_base = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            f"{station_tag}_{model_label}_{source_tag}_To_{target_tag}",
        ).strip("_")
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(
            results_root,
            f"{run_base}_{run_timestamp}",
            "outputs",
        )
    os.makedirs(args.out_dir, exist_ok=True)

    print("=========================================", flush=True)
    print(f"[infer] device={device}", flush=True)
    print(f"[infer] ckpt={args.ckpt}", flush=True)
    print(f"[infer] model={model_name}", flush=True)
    print(f"[infer] encoder_type={encoder_type}", flush=True)
    print(f"[infer] temporal_block={temporal_block}", flush=True)
    print(f"[infer] head_type={head_type}", flush=True)
    print(f"[infer] history_hours={history_hours} (steps={history_steps})", flush=True)
    print(f"[infer] station={station_key or 'ALL'}", flush=True)
    print(f"[infer] model_label={model_label}", flush=True)
    print(f"[infer] source={source_tag}", flush=True)
    print(f"[infer] target={target_tag}", flush=True)
    print(f"[infer] root_dir={args.root_dir}", flush=True)
    print(f"[infer] test_root_dir={args.test_root_dir or '<empty => ROOT_DIR year-split test>'}", flush=True)
    print(f"[infer] out_dir={args.out_dir}", flush=True)
    print("=========================================", flush=True)

    # If you want: make CUDA report earlier if something goes wrong
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # stats from checkpoint (PROPER normalization)
    # current train.py stores x_center/x_scale (robust/zscore/mag)
    # Backward-compatible load: fall back to x_mean/x_std if needed.
    x_center_cpu = _as_feature_vector(ckpt.get("x_center", ckpt.get("x_mean")), "x_center")
    x_scale_cpu = _as_feature_vector(ckpt.get("x_scale", ckpt.get("x_std")), "x_scale").clamp_min(1e-6)

    x_clip = float(ckpt.get("x_clip", ckpt_args.get("x_clip", 5.0)))

    # NEW: p_mean normalization stats (scalar) if present
    pmean_center_cpu = ckpt.get("pmean_center", None)
    pmean_scale_cpu = ckpt.get("pmean_scale", None)

    # y stats are per-horizon vectors; keep as (1, H)
    y_mean_cpu = ckpt["y_mean"].float().view(1, -1)
    y_std_cpu = ckpt["y_std"].float().view(1, -1).clamp_min(1e-6)

    # move stats to device AFTER cache clear
    x_center = x_center_cpu.to(device)
    x_scale = x_scale_cpu.to(device)
    y_mean = y_mean_cpu.to(device)
    y_std = y_std_cpu.to(device)

    pmean_center = (pmean_center_cpu.to(device=device, dtype=torch.float32) if pmean_center_cpu is not None else None)
    pmean_scale = (
        pmean_scale_cpu.to(device=device, dtype=torch.float32).clamp_min(1e-6) if pmean_scale_cpu is not None else None
    )

    # station feature (for perceiver3)
    station_feat = None
    if model_name == "perceiver3" and station_key is not None:
        st_json = _try_load_station_json(args.station_json_dir, station_key)
        if st_json is None:
            print(f"[WARN] station JSON not found for '{station_key}'. Using learned station token only.", flush=True)
        else:
            station_feat = station_features_from_json(st_json).to(device=device, dtype=torch.float32)
            print(f"[Station JSON] loaded '{station_key}' -> feat_dim={station_feat.numel()}", flush=True)

    # Build only the store used by this inference mode. Filtering filenames
    # before torch.load avoids materializing unrelated stations in CPU RAM.
    external = bool(args.test_root_dir.strip() != "")
    split_parameters = dict(
        train_frac=float(ckpt_args.get("train_ratio", 0.6)),
        val_frac=float(ckpt_args.get("val_ratio", 0.2)),
        shuffle_years=bool(ckpt_args.get("shuffle_years", False)),
        future_only=bool(ckpt_args.get("future_only", False)),
        future_year_threshold=int(ckpt_args.get("future_year_threshold", 2030)),
        split_seed=int(ckpt_args.get("seed", 42)),
    )
    evaluation_scope = "external_all_years" if external else "checkpoint_year_split"
    if external:
        if os.path.realpath(args.test_root_dir) == os.path.realpath(args.root_dir):
            print(
                "[WARN] TEST_ROOT_DIR resolves to ROOT_DIR. This selects ALL source years "
                "before --years filtering, including training/validation years; "
                "it is not a held-out test by construction. "
                "Omit --test_root_dir to use the checkpoint's test split.",
                flush=True,
            )
        store_test = ForcingGraphStore(
            args.test_root_dir,
            pattern="*graphs.pt",
            station_filter=station_key,
            strict_station_filter=args.strict_station_test,
        )
        test_indices_all = make_all_years_test_indices(
            store_test,
            station_filter=station_key,
            strict=args.strict_station_test,
        )
        store_for_test = store_test
    else:
        store_source = ForcingGraphStore(
            args.root_dir,
            pattern="*graphs.pt",
            station_filter=station_key,
            # Preserve the existing non-strict year-split fallback behavior.
            strict_station_filter=False,
        )
        test_indices_all = make_year_split_indices(
            store_source,
            part="test",
            station_filter=station_key,
            **split_parameters,
        )
        store_for_test = store_source
    test_tag = target_tag

    if len(test_indices_all) == 0:
        raise RuntimeError("No test samples found (check station filter / data paths).")

    # model config: pull the real hyperparams from ckpt args
    in_channels = store_for_test.graphs[test_indices_all[0]].x.size(-1)
    out_channels = store_for_test.graphs[test_indices_all[0]].y.numel()

    hidden_channels = int(ckpt_args.get("hidden_channels", 64))
    dropout = float(ckpt_args.get("dropout", 0.0))
    head_dropout = float(ckpt_args.get("head_dropout", 0.0))

    # model-specific defaults from ckpt
    node_read_heads = int(ckpt_args.get("node_read_heads", 8))
    time_read_heads = int(ckpt_args.get("time_read_heads", 8))
    transformer_layers = int(ckpt_args.get("transformer_layers", 2))
    transformer_ff_mult = float(ckpt_args.get("transformer_ff_mult", 4.0))
    transformer_dropout = float(ckpt_args.get("transformer_dropout", 0.05))
    max_time_steps = int(ckpt_args.get("max_time_steps", 32))
    gate_mode = str(ckpt_args.get("gate_mode", "window"))
    gate_bias_init = float(ckpt_args.get("gate_bias_init", -2.0))
    tail_tanh_clip = float(ckpt_args.get("tail_tanh_clip", 2.5))
    alpha_init_logit = float(ckpt_args.get("alpha_init_logit", -2.0))

    # p_mean ablation knobs must match train.py to load weights strictly
    use_pmean = bool(ckpt_args.get("use_pmean", False))
    pmean_dim = int(ckpt_args.get("pmean_dim", 32))
    perceiver_pmean_mode = str(ckpt_args.get("perceiver_pmean_mode", "tokens"))

    if model_name == "baseline":
        if history_steps == 0:
            model = SpatialOnlyGraphSAGEBatch(
                in_channels,
                hidden_channels,
                out_channels,
                num_layers=num_layers,
                dropout=dropout,
                use_pmean=use_pmean,
                pmean_dim=pmean_dim,
                encoder_type=encoder_type,
            )
        else:
            W = history_steps + 1
            model = SpatioTemporalGraphSAGEBatch(
                in_channels,
                hidden_channels,
                out_channels,
                num_layers=num_layers,
                dropout=dropout,
                use_pmean=use_pmean,
                pmean_T=W,
                pmean_dim=pmean_dim,
                encoder_type=encoder_type,
            )
    elif model_name == "perceiver3":
        if history_steps == 0:
            raise ValueError("perceiver3 requires history>0")
        station_feat_dim = int(station_feat.numel()) if station_feat is not None else 0

        use_tokens = bool(use_pmean) and (perceiver_pmean_mode in ("tokens", "both"))
        use_global = bool(use_pmean) and (perceiver_pmean_mode in ("global", "both"))

        model = PACT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            n_node_read_heads=node_read_heads,
            n_time_read_heads=time_read_heads,
            n_transformer_layers=transformer_layers,
            transformer_ff_mult=transformer_ff_mult,
            transformer_dropout=transformer_dropout,
            head_dropout=head_dropout,
            gate_mode=gate_mode,
            gate_bias_init=gate_bias_init,
            tail_tanh_clip=tail_tanh_clip,
            alpha_init_logit=alpha_init_logit,
            max_time_steps=max_time_steps,
            station_feat_dim=station_feat_dim,
            use_station_meta=True,
            use_pmean_tokens=use_tokens,
            use_pmean_global=use_global,
            pmean_dim=pmean_dim,
            encoder_type=encoder_type,
            temporal_block=temporal_block,
            head_type=head_type,
        )
    else:
        raise ValueError(
            "infer.py currently supports model="
            "baseline/perceiver3, "
            f"got {model_name}"
        )

    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # group indices by year and run year-by-year (gives correct per-year timing)
    year_to_indices = defaultdict(list)
    for idx in test_indices_all:
        tag = store_for_test.graph_tags[idx]
        parts = tag.split("_")
        if len(parts) >= 2:
            year_tag = f"{parts[0]}_{parts[1]}"
        else:
            year_tag = "UNKNOWN"
        year_to_indices[year_tag].append(idx)

    years = sorted(year_to_indices.keys())

    # optional year subset
    if args.years.strip():
        want = [y.strip() for y in args.years.split(",") if y.strip()]
        years = [y for y in years if y in set(want)]
        if not years:
            raise RuntimeError(f"--years was set but none matched available years. want={want} available={sorted(year_to_indices.keys())}")

    print(f"[infer] test_tag={test_tag} years={years}", flush=True)

    results = {}
    y_true_all = []
    y_pred_all = []
    tags_all = []

    pin_memory = bool(args.pin_memory)
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)

    # Timing summary (exclude the partial boundary year 2014-2015)
    timing_year_seconds = []
    timing_year_labels = []

    for y in years:
        idxs = year_to_indices[y]
        ds = ForcingGraphView(store_for_test, idxs, history_steps=history_steps)
        loader = build_loader(
            ds,
            sampler=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=int(args.prefetch_factor),
            mp_context=args.mp_context,
        )

        rmse, mae, dt, y_t, y_p, tags = infer_one_loader(
            model=model,
            loader=loader,
            device=device,
            x_center=x_center,
            x_scale=x_scale,
            x_clip=float(x_clip),
            pmean_center=pmean_center,
            pmean_scale=pmean_scale,
            pmean_clip=float(x_clip),
            y_mean=y_mean,
            y_std=y_std,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            model_name=model_name,
            station_feat=station_feat,
            gpu_sync_timing=bool(args.gpu_sync_timing),
        )

        results[y] = dict(samples=int(len(idxs)), rmse=float(rmse), mae=float(mae), seconds=float(dt), unit="physical")
        print(f"[Year {y} | physical] samples={len(idxs)} rmse={rmse:.6e} mae={mae:.6e} time={dt:.2f}s", flush=True)

        # For avg-per-year timing, exclude the partial boundary year tag 2014_2015
        # (this file often contains an incomplete year due to dataset boundary).
        rep_tag = store_for_test.graph_tags[idxs[0]] if len(idxs) > 0 else ""
        y0_rep, y1_rep = parse_year_tag(rep_tag)
        if not (y0_rep == 2014 and y1_rep == 2015):
            timing_year_seconds.append(float(dt))
            timing_year_labels.append(f"{y0_rep}_{y1_rep}" if (y0_rep is not None and y1_rep is not None) else str(y))

        # Summaries use exactly the evaluated samples; --save_npz only controls disk I/O.
        y_true_all.append(y_t)
        y_pred_all.append(y_p)
        tags_all.append(tags)

    # Avg time per year (exclude 2014-2015)
    if timing_year_seconds:
        avg_year_time = float(np.mean(timing_year_seconds))
        print(f"[Timing | excl 2014-2015] avg_time_per_year={avg_year_time:.2f}s over {len(timing_year_seconds)} years", flush=True)
        results["_avg_time_per_year_excl_2014_2015"] = dict(seconds=float(avg_year_time), n_years=int(len(timing_year_seconds)), excluded="2014-2015")
    else:
        print("[Timing] avg_time_per_year_excl_2014_2015: n/a (no eligible years)", flush=True)
        results["_avg_time_per_year_excl_2014_2015"] = dict(seconds=float('nan'), n_years=0, excluded="2014-2015")

    # overall metrics (ALL / past / future)
    # NOTE: rmse/mae are computed in PHYSICAL space (original y units),
    # because y_true is taken directly from batch.y and predictions are denormalized:
    #   y_pred = pred_norm * y_std + y_mean.
    #
    # Past/future split rule (fixed):
    #   past  : 1979–2014
    #   future: 2070–2099
    def _metrics_from_arrays(YT: np.ndarray, YP: np.ndarray) -> tuple[float, float]:
        if YT.size == 0:
            return (float("nan"), float("nan"))
        err = (YP.astype(np.float64) - YT.astype(np.float64)).reshape(-1)
        return (float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err))))

    YT_all = np.concatenate(y_true_all, axis=0)
    YP_all = np.concatenate(y_pred_all, axis=0)
    TAGS_all = np.concatenate(tags_all, axis=0)
    overall_rmse, overall_mae = _metrics_from_arrays(YT_all, YP_all)

    groups = np.array(
        [classify_past_future(parse_year_tag(t)[0]) for t in TAGS_all],
        dtype=object,
    )
    m_past = groups == "past"
    m_future = groups == "future"
    past_rmse, past_mae = _metrics_from_arrays(YT_all[m_past], YP_all[m_past])
    fut_rmse, fut_mae = _metrics_from_arrays(YT_all[m_future], YP_all[m_future])

    results["_overall"] = dict(rmse=float(overall_rmse), mae=float(overall_mae), unit="physical")
    results["_overall_past"] = dict(rmse=float(past_rmse), mae=float(past_mae), unit="physical", years="1979-2014")
    results["_overall_future"] = dict(rmse=float(fut_rmse), mae=float(fut_mae), unit="physical", years="2070-2099")

    print(f"[Overall | physical] rmse={overall_rmse:.6e} mae={overall_mae:.6e}", flush=True)
    print(f"[Overall Past 1979-2014 | physical] rmse={past_rmse:.6e} mae={past_mae:.6e} (n_graphs={int(m_past.sum())})", flush=True)
    print(f"[Overall Future 2070-2099 | physical] rmse={fut_rmse:.6e} mae={fut_mae:.6e} (n_graphs={int(m_future.sum())})", flush=True)
    # write json
    meta = dict(
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        test_tag=test_tag,
        source_tag=source_tag,
        target_tag=target_tag,
        station=station_tag,
        model=model_name,
        model_label=model_label,
        encoder_type=encoder_type,
        temporal_block=temporal_block,
        head_type=head_type,
        history_hours=int(history_hours),
        ckpt=args.ckpt,
        root_dir=args.root_dir,
        test_root_dir=args.test_root_dir,
        evaluation_scope=evaluation_scope,
        split_parameters=split_parameters if not external else None,
        years_evaluated=years,
        inference_args=vars(args).copy(),
        checkpoint_args=ckpt_args,
        results=results,
        metric_space="physical",
        metric_note="RMSE/MAE computed on denormalized predictions in original y units (e.g., meters).",
        # record p_mean usage for reproducibility
        use_pmean=bool(use_pmean),
        perceiver_pmean_mode=str(perceiver_pmean_mode),
        x_clip=float(x_clip),
    )
    encoder_file_tag = "" if encoder_type == "GraphSAGE" else f"_{encoder_type}"
    temporal_file_tag = (
        ""
        if model_name != "perceiver3" or temporal_block == "Transformer"
        else f"_{temporal_block}"
    )
    head_file_tag = "" if model_name != "perceiver3" or head_type == "dual" else f"_{head_type}"
    model_file_tag = f"{encoder_file_tag}{temporal_file_tag}{head_file_tag}"
    json_path = os.path.join(
        args.out_dir,
        f"metrics_per_year_{test_tag}_{station_tag}_{model_name}{model_file_tag}.json",
    )
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved per-year metrics -> {json_path}", flush=True)

    # save npz
    if args.save_npz and y_true_all:
        npz_path = os.path.join(
            args.out_dir,
            f"preds_{test_tag}_{station_tag}_{model_name}{model_file_tag}_ALLYEARS.npz",
        )
        np.savez(npz_path, y_true=YT_all, y_pred=YP_all, tags=TAGS_all)
        print(f"Saved predictions -> {npz_path}", flush=True)


if __name__ == "__main__":
    main()
