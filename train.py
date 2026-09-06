#!/usr/bin/env python3
"""Storm Surge Emulator training endpoint.

This file intentionally keeps only the CLI endpoint (`main`) and orchestrates
components implemented under `emulator/`.

High-level module layout:
- `emulator/common`: DDP/runtime/atomic-IO helpers.
- `emulator/data`: graph store, split policy, normalization, station metadata,
  and train-derived statistics.
- `emulator/models`: baseline and perceiver model architectures.
- `emulator/training`: losses and train/eval engines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from emulator.common import (
    ddp_is_initialized,
    infer_dataset_tag,
    infer_stats_threads,
    is_main_process,
    print0,
    set_seed,
    temp_numpy_threads,
    write_json_atomic,
)
from emulator.data import (
    ForcingGraphStore,
    ForcingGraphView,
    _graph_has_pmean,
    _try_load_station_json,
    build_loader,
    compute_pmean_stats_from_store_rank0,
    compute_train_loss_thresholds_from_store,
    compute_x_mag_stats_from_store_rank0,
    compute_x_robust_stats_from_store_rank0,
    compute_x_stats_distributed_from_store,
    compute_y_stats_distributed_from_store,
    make_all_years_test_indices,
    make_year_split_indices,
    station_features_from_json,
    years_from_indices,
)
from emulator.models import (
    PACT,
    SpatialOnlyGraphSAGEBatch,
    SpatioTemporalGraphSAGEBatch,
    canonical_head_type,
    canonical_temporal_block,
)
from emulator.training import (
    collect_test_preds_unnorm,
    eval_full_metrics_and_logs,
    evaluate_ddp,
    train_one_epoch_ddp,
)


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*",
)

def main():
    parser = argparse.ArgumentParser()

    def _bool_int(value):
        v = str(value).strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return 1
        if v in ("0", "false", "no", "n", "off"):
            return 0
        raise argparse.ArgumentTypeError("expected one of: 0/1, true/false, yes/no, on/off")

    # -------------------------
    # Data roots
    # -------------------------
    parser.add_argument("--root_dir", type=str, default="./Data/NCEP/graphs")
    parser.add_argument("--test_root_dir", type=str, default="", help="If set: train/val on root_dir, test on ALL years in test_root_dir")

    # Station filter (single station workflow for now)
    parser.add_argument("--filter", type=str, default=None, help="Station key, e.g., Battery")
    parser.add_argument("--station", type=str, default=None, help="Alias for --filter (preferred)")

    # Station JSON directory
    parser.add_argument("--station_json_dir", type=str, default="./station_json")

    # Split args
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--shuffle_years",
        "--shuffle_split_years",
        dest="shuffle_years",
        type=_bool_int,
        default=0,
        choices=[0, 1],
        help="If 1, shuffle year groups with --seed before train/val/test ratio split.",
    )
    parser.add_argument(
        "--future_only",
        "--future_only_years",
        dest="future_only",
        type=_bool_int,
        default=0,
        choices=[0, 1],
        help="If 1, keep only year tags with any year component > --future_year_threshold before ratio split.",
    )
    parser.add_argument(
        "--future_year_threshold",
        type=int,
        default=2030,
        help="Year threshold used by --future_only; default keeps tags containing years after 2030.",
    )

    # -------------------------
    # Training knobs
    # -------------------------
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Single-process gradient accumulation steps; values >1 are rejected when world_size != 1.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05, help="Light regularization; set 0.0 if it hurts.")
    parser.add_argument("--history_hours", type=int, default=24)

    # -------------------------
    # OOD robustness knobs (NEW)
    # -------------------------
    parser.add_argument("--x_norm", type=str, default="robust", choices=["zscore", "robust", "mag"],
                        help=(
                            "Input normalization. "
                            "- zscore: (x-mean)/std computed on TRAIN (distributed)\n"
                            "- robust: (x-center)/scale using per-feature percentiles on TRAIN (rank0)\n"
                            "          center=(p_lo+p_hi)/2, scale=(p_hi-p_lo)/2 (OOD-stable)\n"
                            "- mag   : x / mag using per-feature magnitude on TRAIN (rank0), mag=p_hi percentile of |x|\n"
                            "          (no centering; can help if scale shift dominates, but won’t remove mean offset)."
                        ))
    parser.add_argument("--x_p_lo", type=float, default=1.0,
                        help="Lower percentile for robust normalization (e.g., 1).")
    parser.add_argument("--x_p_hi", type=float, default=99.0,
                        help="Upper percentile for robust normalization (e.g., 99).")
    parser.add_argument("--x_nodes_per_graph", type=int, default=256,
                        help="How many nodes to sample per graph for robust percentile stats.")
    parser.add_argument("--x_clip", type=float, default=5.0,
                        help="Clamp normalized X to [-x_clip, x_clip]. Helps OOD stability.")

    # Train-only feature jitter (in normalized space)
    parser.add_argument("--x_aug", type=int, default=1, choices=[0, 1],
                        help="Enable train-time feature scale+bias jitter (in normalized space).")
    parser.add_argument("--x_aug_prob", type=float, default=1.0,
                        help="Probability to apply augmentation per batch (0..1).")
    parser.add_argument("--x_aug_scale", type=float, default=0.05,
                        help="Scale jitter range. a ~ U(1-x_aug_scale, 1+x_aug_scale).")
    parser.add_argument("--x_aug_bias", type=float, default=0.02,
                        help="Bias jitter std in normalized space. b ~ N(0, x_aug_bias).")

    # -------------------------
    # Loss knobs (NEW)
    # -------------------------
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="mse",
        choices=["mse", "wmse", "mse_tail", "wmse_tail", "mse_wtail", "mse_slope", "wmse_slope", "mse_tail_slope", "wmse_tail_slope", "mse_wtail_slope"],
        help="Base modes: mse, wmse, mse_tail, wmse_tail, mse_wtail. Suffix *_slope adds slope-matching smoothness with a soft mask (see --slope_* args).",
    )
    parser.add_argument("--wmse_q", type=float, default=95.0,
                        help="Percentile for q threshold computed on TRAIN (over |y| across all horizons).")
    parser.add_argument("--wmse_alpha", type=float, default=4.0,
                        help="alpha in w(y)=1+alpha*sigmoid((|y|-q)/s).")
    parser.add_argument("--wmse_s", type=float, default=0.10,
                        help="s in meters (softness) for weighted MSE. Typical: 0.05~0.2.")
    parser.add_argument("--wmse_use_abs", type=int, default=1, choices=[0, 1],
                        help="1: use |y| in weight; 0: use y directly.")
    parser.add_argument("--tail_frac", type=float, default=0.05,
                        help="Top fraction by GT peak (max over horizons) for tail auxiliary loss.")
    parser.add_argument("--tail_lambda", type=float, default=0.10,
                        help="Weight for tail auxiliary loss. Start small: 0.05~0.2.")

    # -------------------------
    # Smoothness (non-autoregressive) — slope-matching loss (NEW)
    # -------------------------
    # Enabled by choosing a *_slope loss_mode.
    # Adds: L_total = L_base + tail_lambda*L_tail + slope_lambda*L_slope
    # L_slope matches first differences across the horizon axis:
    #   (ŷ[:,1:]-ŷ[:,:-1]) ≈ (y[:,1:]-y[:,:-1])
    # Soft mask to avoid smoothing extremes:
    #   w = sigmoid((tau - peak_abs)/mask_s), tau defaults to TRAIN q(|y|) percentile (wmse_q)
    parser.add_argument("--slope_lambda", type=float, default=0.01,
                        help="Weight for slope-matching smoothness loss. Typical: 0.001~0.05. Used only for *_slope modes.")
    parser.add_argument("--slope_mask_s", type=float, default=0.10,
                        help="Soft-mask sharpness in meters: w=sigmoid((tau-peak)/s). Smaller => harder mask. Typical: 0.05~0.2.")
    parser.add_argument("--slope_robust", type=str, default="charb", choices=["charb", "huber"],
                        help="Robust penalty for slope error. charb=Charbonnier; huber=Huber.")
    parser.add_argument("--slope_charb_eps", type=float, default=1e-3,
                        help="Charbonnier eps in meters for slope robust penalty.")
    parser.add_argument("--slope_huber_delta", type=float, default=0.05,
                        help="Huber delta in meters for slope robust penalty.")

    # -------------------------
    # Scheduler (default cosine+warmup; keep ROP optional)
    # -------------------------
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "rop"])
    parser.add_argument("--min_lr", type=float, default=1e-6, help="eta_min for cosine")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1,
                        help="Linear warmup start lr factor (e.g., 0.1 means start at 10% of lr)")

    parser.add_argument("--rop_factor", type=float, default=0.5)
    parser.add_argument("--rop_patience", type=int, default=20)
    parser.add_argument("--rop_threshold", type=float, default=1e-4)
    parser.add_argument("--rop_cooldown", type=int, default=0)
    parser.add_argument("--rop_min_lr", type=float, default=1e-6)
    parser.add_argument("--rop_metric", type=str, default="val_rmse_phys",
                        choices=["val_rmse_phys", "val_rmse_peak"],
                        help="Metric used for ROP stepping. val_rmse_peak is top5% GT peak RMSE from rank0 full-pass (broadcasted to all ranks).")

    # -------------------------
    # DataLoader knobs
    # -------------------------
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--mp_context", type=str, default="fork", choices=["fork", "spawn"])

    # -------------------------
    # Speed knobs
    # -------------------------
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--tf32", action="store_true")

    # AMP
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # -------------------------
    # Model selection
    # -------------------------
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "perceiver3"])
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="GraphSAGE",
        choices=["GraphSAGE", "CNN"],
        help=(
            "Spatial encoder. GraphSAGE uses edge_index; CNN reshapes each graph "
            "from (H*W,F) to (F,H,W) using grid_H/grid_W stored in the input data."
        ),
    )

    # -------------------------
    # OPTIONAL: p_mean injection (ablation knob)
    # -------------------------
    parser.add_argument(
        "--use_pmean",
        action="store_true",
        help=(
            "If set, expose global mean pressure metadata (p_mean_hist/p_mean_curr) to the model. "
            "Baseline: Option 1 (concat global encoding). Perceiver3: controlled by --perceiver_pmean_mode (tokens/global/both). "
            "Safe to enable even if the dataset lacks p_mean fields (it becomes a no-op)."
        ),
    )
    parser.add_argument(
        "--pmean_dim",
        type=int,
        default=32,
        help=(
            "Embedding dimension used by the baseline's p_mean encoder when --use_pmean is enabled. "
            "(For perceiver3, p_mean tokens are projected directly to hidden_channels.)"
        ),
    )



    # p_mean injection (Perceiver3 ablation): how to use p_mean_hist inside perceiver3
    # When --use_pmean is enabled:
    #   - Baseline always uses Option 1 (concat a learned global encoding).
    #   - Perceiver3 can use p_mean_hist in multiple ways for controlled ablations:
    #       * tokens : Option 3 — create time-indexed context tokens (strong story, lag-aware)
    #       * global : Option 1-style — encode p_mean_hist into ONE global vector and concat to head inputs
    #       * both   : enable both pathways (often strongest, slightly more parameters)
    # If --use_pmean is NOT set, this flag is ignored.
    parser.add_argument(
        "--perceiver_pmean_mode",
        type=str,
        default="tokens",
        choices=["tokens", "global", "both"],
        help=(
            "When --model perceiver3 and --use_pmean is set: choose how p_mean_hist is injected. "
            "tokens=append time-aligned p_mean tokens (Option 3); "
            "global=encode p_mean_hist into a global vector and concatenate to the forecasting head (Option 1-style); "
            "both=enable both tokens and global."
        ),
    )
    # Common head knobs
    parser.add_argument(
        "--head_type",
        type=canonical_head_type,
        default="dual",
        choices=["single", "dual"],
        help="PACT prediction head: single base MLP or the existing gated dual head.",
    )
    parser.add_argument("--head_dropout", type=float, default=0.0)

    # Peak-aware gate/tail knobs (used by PACT's dual head only)
    parser.add_argument("--gate_mode", type=str, default="window", choices=["window", "horizon"])
    parser.add_argument("--gate_bias_init", type=float, default=-2.0)
    parser.add_argument("--tail_tanh_clip", type=float, default=2.5)
    parser.add_argument("--alpha_init_logit", type=float, default=-2.0, help="Used by perceiver3")


    # Model3 knobs
    parser.add_argument(
        "--temporal_block",
        type=canonical_temporal_block,
        default="Transformer",
        choices=["MLP", "LSTM", "GRU", "Transformer"],
        help="PACT middle temporal block; 'attn' is accepted as an alias for Transformer.",
    )
    parser.add_argument("--node_read_heads", type=int, default=8)
    parser.add_argument("--time_read_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_ff_mult", type=float, default=4.0)
    parser.add_argument("--transformer_dropout", type=float, default=0.05)
    parser.add_argument("--max_time_steps", type=int, default=32)

    # Tagging
    parser.add_argument("--run_tag", type=str, default=None)

    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        parser.error(f"--grad_accum_steps must be >= 1, got {args.grad_accum_steps}.")

    # -------------------------
    # Resolve station filter alias
    # -------------------------
    station_key = args.station if (args.station is not None and args.station != "") else args.filter
    if station_key is None or station_key == "":
        station_key = None

    # -------------------------
    # CPU thread control
    # -------------------------
    os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
    torch.set_num_threads(args.torch_threads)

    # -------------------------
    # TF32 (H100 friendly)
    # -------------------------
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # -------------------------
    # DDP init (SAFE on Slurm + torchrun + single-process runs)
    # -------------------------
    have_rank = ("RANK" in os.environ) or ("SLURM_PROCID" in os.environ)
    if have_rank:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    if args.grad_accum_steps > 1 and world_size != 1:
        raise RuntimeError(
            "Gradient accumulation is intentionally supported only for world_size == 1: "
            f"got grad_accum_steps={args.grad_accum_steps}, world_size={world_size}. "
            "Set --grad_accum_steps 1 for DDP or launch a single training process."
        )

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        n_vis = torch.cuda.device_count()
        if n_vis < 1:
            raise RuntimeError("CUDA is available but torch.cuda.device_count() == 0. Check environment.")
        cuda_id = 0 if n_vis == 1 else (local_rank % n_vis)

        torch.cuda.set_device(cuda_id)
        device = torch.device(f"cuda:{cuda_id}")
    else:
        cuda_id = -1
        device = torch.device("cpu")

    if world_size > 1:
        import socket
        import subprocess

        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))

        def _infer_master_addr() -> str:
            ma = os.environ.get("MASTER_ADDR", "").strip()
            if ma:
                return ma

            nodelist = os.environ.get("SLURM_NODELIST", "").strip()
            if nodelist:
                try:
                    out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True)
                    host = out.splitlines()[0].strip()
                    if host:
                        return host
                except Exception:
                    pass

            ip = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "").strip()
            if ip:
                return ip.split(",")[0].split()[0]

            try:
                return socket.gethostname()
            except Exception:
                return "127.0.0.1"

        def _infer_master_port() -> str:
            mp = os.environ.get("MASTER_PORT", "").strip()
            if mp:
                return mp

            jid = os.environ.get("SLURM_JOB_ID", "").strip()
            if jid.isdigit():
                return str(29500 + (int(jid) % 1000))

            return str(29500 + (os.getpid() % 1000))

        os.environ.setdefault("MASTER_ADDR", _infer_master_addr())
        os.environ.setdefault("MASTER_PORT", _infer_master_port())

        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    set_seed(args.seed + rank)

    if is_main_process():
        print(
            f"[DDP] world_size={world_size} rank={rank} local_rank={local_rank} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"visible_cuda={torch.cuda.device_count() if use_cuda else 0} chosen_cuda_id={cuda_id} device={device}",
            flush=True
        )
        if args.grad_accum_steps > 1:
            print(
                f"[GradAccum] micro_batch_size={args.batch_size} "
                f"steps={args.grad_accum_steps} "
                f"nominal_effective_batch_size={args.batch_size * args.grad_accum_steps}",
                flush=True,
            )

    # AMP setup
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_amp = bool(args.amp and device.type == "cuda")
    if use_amp and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=False)

    assert args.history_hours % 6 == 0, "--history_hours must be multiple of 6."
    history_steps = args.history_hours // 6
    print0(f"[History] history_hours={args.history_hours}, steps={history_steps}, window_len={history_steps + 1}")

    # -------------------------
    # Load train/val store
    # -------------------------
    store = ForcingGraphStore(
        args.root_dir,
        pattern="*graphs.pt",
        log_fn=print0,
        station_filter=station_key,
    )

    train_idx = make_year_split_indices(
        store,
        part="train",
        station_filter=station_key,
        train_frac=args.train_ratio,
        val_frac=args.val_ratio,
        shuffle_years=bool(args.shuffle_years),
        future_only=bool(args.future_only),
        future_year_threshold=args.future_year_threshold,
        split_seed=args.seed,
        log_fn=print0,
    )
    val_idx = make_year_split_indices(
        store,
        part="val",
        station_filter=station_key,
        train_frac=args.train_ratio,
        val_frac=args.val_ratio,
        shuffle_years=bool(args.shuffle_years),
        future_only=bool(args.future_only),
        future_year_threshold=args.future_year_threshold,
        split_seed=args.seed,
        log_fn=print0,
    )
    test_idx = make_year_split_indices(
        store,
        part="test",
        station_filter=station_key,
        train_frac=args.train_ratio,
        val_frac=args.val_ratio,
        shuffle_years=bool(args.shuffle_years),
        future_only=bool(args.future_only),
        future_year_threshold=args.future_year_threshold,
        split_seed=args.seed,
        log_fn=print0,
    )

    # External test store (e.g., CMIP6)
    store_test = None
    if args.test_root_dir is not None and args.test_root_dir.strip() != "":
        store_test = ForcingGraphStore(
            args.test_root_dir,
            pattern="*graphs.pt",
            log_fn=print0,
            station_filter=station_key,
        )
        test_idx_ext = make_all_years_test_indices(store_test, station_filter=station_key, log_fn=print0)
        test_tag = infer_dataset_tag(args.test_root_dir) or "external"
        print0(f"[Test mode] External test enabled. Using ALL years from: {args.test_root_dir}")
    else:
        test_idx_ext = None
        test_tag = "year_split_test"
        # Note: do not assume a specific dataset; infer tags from root_dir/test_root_dir.
        print0(f"[Test mode] Using year-split test from train/val root ({infer_dataset_tag(args.root_dir)} -> {infer_dataset_tag(args.root_dir)}).")

    if is_main_process():
        print(f"Samples: train={len(train_idx)} val={len(val_idx)} test(split)={len(test_idx)}")
        print(f"[Split years] train ({len(years_from_indices(store, train_idx))}): {years_from_indices(store, train_idx)}")
        print(f"[Split years]   val ({len(years_from_indices(store, val_idx))}): {years_from_indices(store, val_idx)}")
        print(f"[Split years]  test ({len(years_from_indices(store, test_idx))}): {years_from_indices(store, test_idx)}")

    # -------------------------
    # Compute global X stats (robust / zscore / magnitude)
    # -------------------------
    p_lo_cpu = None
    p_hi_cpu = None

    if args.x_norm == "zscore":
        print0("Computing global x_mean/x_std from TRAIN split (distributed)...")
        x_center, x_scale = compute_x_stats_distributed_from_store(store, train_idx, device)
        x_center_cpu, x_scale_cpu = x_center.cpu(), x_scale.cpu()
        print0(f"[x_norm=zscore] x_mean: {x_center_cpu}")
        print0(f"[x_norm=zscore] x_std : {x_scale_cpu}")

    elif args.x_norm == "robust":
        # NEW: allow x_nodes_per_graph <= 0 to mean "auto default"
        nodes_per_graph_eff = int(args.x_nodes_per_graph) if int(args.x_nodes_per_graph) > 0 else 256

        if is_main_process():
            print0(
                f"Computing robust x_center/x_scale from TRAIN (rank0) "
                f"percentiles p{args.x_p_lo:g}..p{args.x_p_hi:g}, nodes_per_graph={nodes_per_graph_eff} ..."
            )
            # Performance note: temporarily use more CPU threads for one-time NumPy percentiles (restored afterwards).
            stats_threads = infer_stats_threads(args.torch_threads)
            with temp_numpy_threads(stats_threads):
                xc, xs, p_lo_vec, p_hi_vec = compute_x_robust_stats_from_store_rank0(
                    store=store,
                    train_indices=train_idx,
                    p_lo=args.x_p_lo,
                    p_hi=args.x_p_hi,
                    nodes_per_graph=nodes_per_graph_eff,
                    seed=args.seed,
                )
            t = torch.stack([xc, xs, p_lo_vec, p_hi_vec], dim=0).to(device=device, dtype=torch.float32)  # (4,F)
        else:
            Fdim = int(store.graphs[train_idx[0]].x.size(-1))
            t = torch.zeros((4, Fdim), device=device, dtype=torch.float32)

        if ddp_is_initialized():
            dist.broadcast(t, src=0)

        x_center = t[0]
        x_scale  = t[1].clamp_min(1e-6)
        p_lo_cpu = t[2].detach().cpu()
        p_hi_cpu = t[3].detach().cpu()

        x_center_cpu, x_scale_cpu = x_center.detach().cpu(), x_scale.detach().cpu()
        print0(f"[x_norm=robust] x_center: {x_center_cpu}")
        print0(f"[x_norm=robust] x_scale : {x_scale_cpu}")
        print0(f"[x_norm=robust] p_lo    : {p_lo_cpu}")
        print0(f"[x_norm=robust] p_hi    : {p_hi_cpu}")

    else:
        # args.x_norm == "mag"
        # Magnitude-only scaling (rank0 compute, broadcast): x_norm = x / mag
        # This is *not* offset-invariant; use robust if mean shifts matter (often true for pressure).
        nodes_per_graph_eff = int(args.x_nodes_per_graph) if int(args.x_nodes_per_graph) > 0 else 256

        if is_main_process():
            print0(
                f"Computing magnitude X scale from TRAIN (rank0) "
                f"mag=p{args.x_p_hi:g} percentile of |x|, nodes_per_graph={nodes_per_graph_eff} ..."
            )
            stats_threads = infer_stats_threads(args.torch_threads)
            with temp_numpy_threads(stats_threads):
                x_mag = compute_x_mag_stats_from_store_rank0(
                    store=store,
                    train_indices=train_idx,
                    p_hi=args.x_p_hi,
                    nodes_per_graph=nodes_per_graph_eff,
                    seed=args.seed,
                )
            # For mag-mode we set center=0 and scale=mag
            xc = torch.zeros_like(x_mag)
            xs = x_mag
            # Keep p_hi_cpu as a debug vector (stores the magnitude used), p_lo_cpu unused.
            p_lo_vec = torch.zeros_like(x_mag)
            p_hi_vec = x_mag
            t = torch.stack([xc, xs, p_lo_vec, p_hi_vec], dim=0).to(device=device, dtype=torch.float32)
        else:
            Fdim = int(store.graphs[train_idx[0]].x.size(-1))
            t = torch.zeros((4, Fdim), device=device, dtype=torch.float32)

        if ddp_is_initialized():
            dist.broadcast(t, src=0)

        x_center = t[0]
        x_scale  = t[1].clamp_min(1e-6)
        p_lo_cpu = None
        p_hi_cpu = t[3].detach().cpu()  # magnitude per feature

        x_center_cpu, x_scale_cpu = x_center.detach().cpu(), x_scale.detach().cpu()
        print0(f"[x_norm=mag] x_center (zeros): {x_center_cpu}")
        print0(f"[x_norm=mag] x_mag_scale     : {x_scale_cpu}")
        print0(f"[x_norm=mag] mag_p_hi        : {args.x_p_hi:g}")

    
# -------------------------
    # Compute global p_mean stats (optional, for later model ablations)
    # -------------------------
    # If your preprocessed graphs include p_mean_hist / p_mean_curr, we compute TRAIN-only
    # normalization parameters here and broadcast to all ranks.
    #
    # This makes p_mean safe to add later as:
    #   - baseline option: concat global encoding
    #   - perceiver3 option: context tokens
    #
    # If p_mean is not present in your dataset yet, these stay as None and the
    # normalization path becomes a no-op.
    pmean_center_cpu = None
    pmean_scale_cpu = None
    pmean_extra = None

    if is_main_process():
        stats_threads = infer_stats_threads(args.torch_threads)
        with temp_numpy_threads(stats_threads):
            pc, ps, extra = compute_pmean_stats_from_store_rank0(
                store=store,
                train_indices=train_idx,
                history_steps=history_steps,
                mode=args.x_norm,
                p_lo=args.x_p_lo,
                p_hi=args.x_p_hi,
            )
        if pc is not None and ps is not None:
            pmean_center_cpu = pc
            pmean_scale_cpu = ps
            pmean_extra = extra
            tpm = torch.tensor([float(pc.item()), float(ps.item())], device=device, dtype=torch.float32)
        else:
            tpm = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)
    else:
        tpm = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)

    if ddp_is_initialized():
        dist.broadcast(tpm, src=0)

    # Non-rank0 processes reconstruct tensors from broadcasted scalars.
    if pmean_center_cpu is None and pmean_scale_cpu is None:
        if float(tpm[1].item()) > 0:
            pmean_center_cpu = torch.tensor(float(tpm[0].item()), dtype=torch.float32)
            pmean_scale_cpu  = torch.tensor(float(tpm[1].item()), dtype=torch.float32)

    if pmean_center_cpu is not None and pmean_scale_cpu is not None:
        print0(f"[p_mean norm] mode={args.x_norm} center={float(pmean_center_cpu.item()):.6e} scale={float(pmean_scale_cpu.item()):.6e}")
        if is_main_process() and isinstance(pmean_extra, dict) and len(pmean_extra) > 0:
            print0(f"[p_mean norm] extra: {pmean_extra}")


# -------------------------
    # Compute global y stats (same)
    # -------------------------
    print0("Computing global y_mean/y_std from TRAIN split (distributed)...")
    y_mean, y_std = compute_y_stats_distributed_from_store(store, train_idx, device)
    y_mean_cpu, y_std_cpu = y_mean.cpu(), y_std.cpu()
    print0(f"y_mean: {y_mean_cpu}")
    print0(f"y_std : {y_std_cpu}")

    # -------------------------
    # Loss thresholds from TRAIN (rank0 compute, broadcast)
    # -------------------------
    use_slope = str(args.loss_mode).endswith("_slope")
    core_loss_mode = str(args.loss_mode)[:-6] if use_slope else str(args.loss_mode)
    use_wmse = core_loss_mode in ("wmse", "wmse_tail", "mse_wtail")
    use_tail = core_loss_mode in ("mse_tail", "wmse_tail", "mse_wtail")
    need_thr = use_wmse or use_tail or use_slope

    wmse_q_value = 0.0
    tail_peak_thr = 0.0

    if need_thr:
        if is_main_process():
            # Performance note: temporarily use more CPU threads for one-time NumPy percentiles (restored afterwards).
            stats_threads = infer_stats_threads(args.torch_threads)
            with temp_numpy_threads(stats_threads):
                wmse_q_value, tail_peak_thr = compute_train_loss_thresholds_from_store(
                    store=store,
                    train_indices=train_idx,
                    wmse_q_percentile=args.wmse_q,
                    tail_frac=args.tail_frac,
                    wmse_use_abs=bool(args.wmse_use_abs),
                )
            t2 = torch.tensor([wmse_q_value, tail_peak_thr], device=device, dtype=torch.float32)
        else:
            t2 = torch.zeros(2, device=device, dtype=torch.float32)

        if ddp_is_initialized():
            dist.broadcast(t2, src=0)

        wmse_q_value = float(t2[0].item())
        tail_peak_thr = float(t2[1].item())

    wmse_q_t = torch.tensor(wmse_q_value, device=device, dtype=torch.float32)
    tail_peak_thr_t = torch.tensor(tail_peak_thr, device=device, dtype=torch.float32)

    if is_main_process():
        print("\n================ METRICS NOTE ================")
        print("Loss                : computed in PHYSICAL y units (meters)")
        print("                      (model predicts y_norm; loss uses y_phys = y_norm*y_std + y_mean)")
        print("rmse_phys / mae_phys: computed in ORIGINAL y units")
        print("Checkpoint selection: based on val_rmse_phys (physical)")
        print("Peak metrics         : computed on rank0 full val set, top 5% by GT peak")
        print("Scheduler            : default cosine+warmup; optional ROP stepped by args.rop_metric")
        print("----------------------------------------------")
        print(f"loss_mode            : {args.loss_mode}")
        if use_wmse:
            print(f"wmse_q_percentile    : {args.wmse_q:.1f} -> q_value={wmse_q_value:.6f} m")
            print(f"wmse_alpha           : {args.wmse_alpha}")
            print(f"wmse_s               : {args.wmse_s} m")
            print(f"wmse_use_abs         : {bool(args.wmse_use_abs)}")
        if use_tail:
            print(f"tail_frac            : {args.tail_frac} -> peak_thr={tail_peak_thr:.6f} m")
            print(f"tail_lambda          : {args.tail_lambda}")
        print("----------------------------------------------")
        print(f"x_norm               : {args.x_norm}")
        if args.x_norm == "robust":
            print(f"x_p_lo/x_p_hi        : {args.x_p_lo:g}/{args.x_p_hi:g}")
            print(f"x_nodes_per_graph    : {args.x_nodes_per_graph}")
        print(f"x_clip               : {args.x_clip}")
        print(f"x_aug                : {bool(args.x_aug)}")
        if args.x_aug:
            print(f"x_aug_prob/scale/bias: {args.x_aug_prob}/{args.x_aug_scale}/{args.x_aug_bias}")
        print("==============================================\n")

    # move stats to device
    x_center_dev = x_center_cpu.to(device=device, dtype=torch.float32)
    x_scale_dev  = x_scale_cpu.to(device=device, dtype=torch.float32)

    # NEW: move p_mean stats to device (scalar tensors). If not available, keep None.
    pmean_center_dev = (pmean_center_cpu.to(device=device, dtype=torch.float32) if pmean_center_cpu is not None else None)
    pmean_scale_dev  = (pmean_scale_cpu.to(device=device, dtype=torch.float32).clamp_min(1e-6) if pmean_scale_cpu is not None else None)

    y_mean_dev = y_mean_cpu.to(device=device, dtype=torch.float32).view(1, -1)
    y_std_dev  = y_std_cpu.to(device=device, dtype=torch.float32).view(1, -1)

    # -------------------------
    # Station features (optional; used by perceiver3; safe if missing)
    # -------------------------
    station_feat = None
    station_json = _try_load_station_json(args.station_json_dir, station_key) if station_key is not None else None
    if station_key is not None:
        if station_json is None:
            print0(f"Warning: station JSON not found for '{station_key}' under {args.station_json_dir}. "
                   f"Model3 will rely on learned station token only.")
        else:
            sf = station_features_from_json(station_json)
            station_feat = sf.to(device=device, dtype=torch.float32)
            print0(f"[Station JSON] loaded '{station_key}' -> station_feat_dim={station_feat.numel()}")
            if station_feat is not None:
                print0(f"station_feat={station_feat.detach().cpu().numpy().round(6).tolist()}")


    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = ForcingGraphView(store, train_idx, history_steps=history_steps)
    val_dataset   = ForcingGraphView(store, val_idx,   history_steps=history_steps)

    if store_test is None:
        test_dataset = ForcingGraphView(store, test_idx, history_steps=history_steps)
    else:
        test_dataset = ForcingGraphView(store_test, test_idx_ext, history_steps=history_steps)

    # -------------------------
    # Samplers + loaders
    # -------------------------
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,  drop_last=False) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if world_size > 1 else None
    test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if world_size > 1 else None

    pin_memory = bool(args.pin_memory)
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)
    prefetch_factor = int(args.prefetch_factor)
    mp_context = args.mp_context

    train_loader = build_loader(train_dataset, train_sampler, args.batch_size, args.num_workers,
                                pin_memory, persistent_workers, prefetch_factor, mp_context)
    val_loader = build_loader(val_dataset, val_sampler, args.batch_size, args.num_workers,
                              pin_memory, persistent_workers, prefetch_factor, mp_context)
    test_loader = build_loader(test_dataset, test_sampler, args.batch_size, args.num_workers,
                               pin_memory, persistent_workers, prefetch_factor, mp_context)

    # -------------------------
    # Model creation
    # -------------------------
    in_channels = store.graphs[train_idx[0]].x.size(-1)
    out_channels = store.graphs[train_idx[0]].y.numel()

    model_name = args.model

    # ---------------------------------------------------------
    # OPTIONAL: p_mean metadata availability check (for clarity)
    #
    # If you enabled --use_pmean but your current graphs do not
    # contain p_mean fields yet, the model-side injection becomes
    # a no-op (thanks to hasattr checks). We still print a single
    # rank0 warning so ablation runs are not silently misleading.
    # ---------------------------------------------------------
    if is_main_process() and bool(args.use_pmean):
        has_pmean_any = False
        try:
            # Only a light check on a few train samples to keep startup fast.
            for _idx in train_idx[: min(50, len(train_idx))]:
                if _graph_has_pmean(store.graphs[_idx]):
                    has_pmean_any = True
                    break
        except Exception:
            # If anything goes wrong, stay conservative: do not claim availability.
            has_pmean_any = False
        if not has_pmean_any:
            print0("[Warning] --use_pmean was set, but p_mean fields were not found in the loaded graphs (train split). "
                   "Model injection will be skipped. Did you run preprocessing/time_align with p_mean saving enabled?")

    if history_steps == 0:
        if model_name == "baseline":
            print0(f"[Model] SpatialOnlyGraphSAGEBatch (history=0, encoder={args.encoder_type})")
            model = SpatialOnlyGraphSAGEBatch(
                in_channels=in_channels,
                hidden_channels=args.hidden_channels,
                out_channels=out_channels,
                num_layers=args.num_layers,
                dropout=args.dropout,
                # p_mean injection: optional p_mean usage (ablation)
                use_pmean=bool(args.use_pmean),
                pmean_dim=int(args.pmean_dim),
                encoder_type=args.encoder_type,
            ).to(device)
        else:
            raise ValueError(
                f"--model {model_name} requires history>0. Use --model baseline for history=0."
            )
    else:
        if model_name == "baseline":
            print0(f"[Model] SpatioTemporalGraphSAGEBatch + LSTM (encoder={args.encoder_type})")
            model = SpatioTemporalGraphSAGEBatch(
                in_channels=in_channels,
                hidden_channels=args.hidden_channels,
                out_channels=out_channels,
                num_layers=args.num_layers,
                dropout=args.dropout,
                # p_mean injection: optional p_mean usage (ablation)
                # Window length W = history_steps + 1 (includes current time).
                use_pmean=bool(args.use_pmean),
                pmean_T=int(history_steps + 1),
                pmean_dim=int(args.pmean_dim),
                encoder_type=args.encoder_type,
            ).to(device)
        elif model_name == "perceiver3":
            station_feat_dim = int(station_feat.numel()) if (station_feat is not None) else 0
            print0(
                f"[Model] PACT (encoder={args.encoder_type}, "
                f"temporal_block={args.temporal_block}, head_type={args.head_type})"
            )
            model = PACT(
                in_channels=in_channels,
                hidden_channels=args.hidden_channels,
                out_channels=out_channels,
                num_layers=args.num_layers,
                dropout=args.dropout,
                n_node_read_heads=args.node_read_heads,
                n_time_read_heads=args.time_read_heads,
                n_transformer_layers=args.transformer_layers,
                transformer_ff_mult=args.transformer_ff_mult,
                transformer_dropout=args.transformer_dropout,
                head_dropout=args.head_dropout,
                gate_mode=args.gate_mode,
                gate_bias_init=args.gate_bias_init,
                tail_tanh_clip=args.tail_tanh_clip,
                alpha_init_logit=args.alpha_init_logit,
                max_time_steps=args.max_time_steps,
                station_feat_dim=station_feat_dim,
                use_station_meta=True,
                # p_mean injection: optional p_mean usage (ablation)
                # Perceiver3 supports multiple injection modes controlled by --perceiver_pmean_mode:
                #   tokens : Option 3 (time-indexed context tokens)
                #   global : Option 1-style (global encoding concatenated to head inputs)
                #   both   : enable both pathways
                use_pmean_tokens=bool(args.use_pmean) and (args.perceiver_pmean_mode in ("tokens", "both")),
                use_pmean_global=bool(args.use_pmean) and (args.perceiver_pmean_mode in ("global", "both")),
                pmean_dim=int(args.pmean_dim),
                encoder_type=args.encoder_type,
                temporal_block=args.temporal_block,
                head_type=args.head_type,
            ).to(device)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    if world_size > 1:
        if device.type == "cuda":
            model = DDP(model, device_ids=[cuda_id], output_device=cuda_id, find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # -------------------------
    # Scheduler
    # -------------------------
    if args.scheduler == "cosine":
        warm = int(max(0, args.warmup_epochs))
        if warm > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=float(args.warmup_start_factor),
                total_iters=warm
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - warm),
                eta_min=float(args.min_lr)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warm]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs), eta_min=float(args.min_lr)
            )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.rop_factor,
            patience=args.rop_patience,
            threshold=args.rop_threshold,
            threshold_mode="rel",
            cooldown=args.rop_cooldown,
            min_lr=args.rop_min_lr
        )

    # -------------------------
    # Output naming
    # -------------------------
    dataset_tag = infer_dataset_tag(args.root_dir) or "data"
    station_tag = (station_key if station_key is not None else "ALL")
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_parts = [args.loss_mode]
    if args.grad_accum_steps > 1:
        loss_parts += [f"ga{args.grad_accum_steps}"]

    tag_use_slope = str(args.loss_mode).endswith("_slope")
    tag_core_loss = str(args.loss_mode)[:-6] if tag_use_slope else str(args.loss_mode)

    if tag_core_loss in ("wmse", "wmse_tail", "mse_wtail"):
        loss_parts += [
            f"q{int(round(float(args.wmse_q)))}",
            f"a{float(args.wmse_alpha):g}",
            f"s{float(args.wmse_s):g}",
            f"abs{int(args.wmse_use_abs)}",
        ]

    if tag_core_loss in ("mse_tail", "wmse_tail", "mse_wtail"):
        loss_parts += [
            f"tf{float(args.tail_frac):g}",
            f"lam{float(args.tail_lambda):g}",
        ]


    if tag_use_slope:
        loss_parts += [
            f"sl{float(args.slope_lambda):g}",
            f"ms{float(args.slope_mask_s):g}",
            f"rb{args.slope_robust}",
        ]
        if args.slope_robust == "charb":
            loss_parts += [f"eps{float(args.slope_charb_eps):g}"]
        else:
            loss_parts += [f"del{float(args.slope_huber_delta):g}"]

    if args.model == "perceiver3":
        if args.head_type == "dual":
            loss_parts += [f"gm{args.gate_mode}"]
        else:
            loss_parts += ["hsingle"]

    # p_mean injection: include p_mean injection mode in run tag (for clean ablations)
    if bool(args.use_pmean):
        if args.model == "perceiver3":
            loss_parts += [f"pmean{args.perceiver_pmean_mode}"]
        else:
            loss_parts += ["pmean"]

    # include xnorm in tag (short, safe)
    loss_parts += [f"x{args.x_norm}", f"xc{float(args.x_clip):g}"]
    if int(args.x_aug) == 1:
        loss_parts += [f"aug{args.x_aug_scale:g}_{args.x_aug_bias:g}"]

    loss_tag = "_".join(loss_parts)

    os.makedirs(f"results_{station_tag}", exist_ok=True)
    os.makedirs(f"checkpoints_{station_tag}", exist_ok=True)

    best_val_rmse_phys = float("inf")

    def _safe_slug(s: str, max_len: int = 40) -> str:
        s = str(s) if s is not None else "run"
        s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")
        if not s:
            s = "run"
        if len(s) <= max_len:
            return s
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
        return s[: max_len - 9] + "_" + h

    cfg_id = hashlib.md5(json.dumps(vars(args), sort_keys=True).encode("utf-8")).hexdigest()[:10]
    run_slug = _safe_slug(run_tag, max_len=40)

    ckpt_dir = f"checkpoints_{station_tag}"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_ckpt_path = os.path.join(
        ckpt_dir,
        f"best_{model_name}_{loss_tag}_{cfg_id}_{run_slug}.pth",
    )

    # -------------------------
    # Meta JSON (atomic write for reproducibility)
    # -------------------------
    if is_main_process():
        meta_path = os.path.join(ckpt_dir, f"meta_{model_name}_{loss_tag}_{cfg_id}_{run_slug}.json")
        try:
            meta = {
                "cfg_id": cfg_id,
                "run_tag": run_tag,
                "run_slug": run_slug,
                "dataset_tag": dataset_tag,
                "test_tag": test_tag,
                "station_tag": station_tag,
                "model": model_name,
                "encoder_type": args.encoder_type,
                "temporal_block": args.temporal_block,
                "head_type": args.head_type,
                "grad_accum_steps": int(args.grad_accum_steps),
                "loss_tag": loss_tag,
                "root_dir": args.root_dir,
                "test_root_dir": args.test_root_dir,
                "world_size": world_size,
                "rank0_host": os.environ.get("HOSTNAME", ""),
                "x_norm": args.x_norm,
                "x_clip": float(args.x_clip),
                "x_p_lo": float(args.x_p_lo),
                "x_p_hi": float(args.x_p_hi),
                "x_nodes_per_graph": int(args.x_nodes_per_graph),
                "x_aug": bool(args.x_aug),
                "x_aug_prob": float(args.x_aug_prob),
                "x_aug_scale": float(args.x_aug_scale),
                "x_aug_bias": float(args.x_aug_bias),
                "args": vars(args),
                "created_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            write_json_atomic(meta_path, meta, indent=2)
            try:
                sz = os.path.getsize(meta_path)
            except Exception:
                sz = -1
            print0(f"[meta] wrote meta json -> {meta_path} (bytes={sz})")
        except Exception as e:
            print0(f"[WARN] could not write meta json: {e}")
            print0(traceback.format_exc())

    train_log = []
    val_log = []

    train_t0 = time.time()

    # -------------------------
    # Train loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_mse_norm, tr_mse_phys, tr_rmse_phys, tr_mae_phys, gate_mean_train = train_one_epoch_ddp(
            model, train_loader, optimizer, device,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
            x_center=x_center_dev, x_scale=x_scale_dev, x_clip=args.x_clip,
            pmean_center=pmean_center_dev, pmean_scale=pmean_scale_dev, pmean_clip=args.x_clip,
            do_x_aug=bool(args.x_aug),
            x_aug_prob=args.x_aug_prob,
            x_aug_scale=args.x_aug_scale,
            x_aug_bias=args.x_aug_bias,
            y_mean=y_mean_dev, y_std=y_std_dev,
            model_name=model_name,
            station_feat=station_feat,
            loss_mode=args.loss_mode,
            wmse_q_t=wmse_q_t,
            tail_peak_thr_t=tail_peak_thr_t,
            wmse_alpha=args.wmse_alpha,
            wmse_s=args.wmse_s,
            wmse_use_abs=bool(args.wmse_use_abs),
            tail_lambda=args.tail_lambda,
            slope_lambda=args.slope_lambda,
            slope_mask_s=args.slope_mask_s,
            slope_robust=args.slope_robust,
            slope_charb_eps=args.slope_charb_eps,
            slope_huber_delta=args.slope_huber_delta,
            grad_accum_steps=args.grad_accum_steps,
        )

        va_mse_norm, va_mse_phys, va_rmse_phys, va_mae_phys = evaluate_ddp(
            model, val_loader, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
            x_center=x_center_dev, x_scale=x_scale_dev, x_clip=args.x_clip,
            pmean_center=pmean_center_dev, pmean_scale=pmean_scale_dev, pmean_clip=args.x_clip,
            y_mean=y_mean_dev, y_std=y_std_dev,
            model_name=model_name,
            station_feat=station_feat,
        )

        extra = None
        if is_main_process():
            raw_model = model.module if isinstance(model, DDP) else model
            extra = eval_full_metrics_and_logs(
                model_raw=raw_model,
                dataset=val_dataset,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                mp_context=mp_context,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                x_center=x_center_dev,
                x_scale=x_scale_dev,
                x_clip=args.x_clip,
                pmean_center=pmean_center_dev,
                pmean_scale=pmean_scale_dev,
                pmean_clip=args.x_clip,
                y_mean=y_mean_dev,
                y_std=y_std_dev,
                model_name=model_name,
                station_feat=station_feat,
                peak_frac=0.05,
            )

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            metric_for_rop = float(va_rmse_phys)
            if args.rop_metric == "val_rmse_peak":
                if is_main_process():
                    metric_peak = float(extra["rmse_peak"]) if (extra is not None and "rmse_peak" in extra) else float(va_rmse_phys)
                    m = torch.tensor(metric_peak, device=device, dtype=torch.float32)
                else:
                    m = torch.tensor(0.0, device=device, dtype=torch.float32)
                if ddp_is_initialized():
                    dist.broadcast(m, src=0)
                metric_for_rop = float(m.item())
            scheduler.step(metric_for_rop)

        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_dt = time.time() - epoch_t0

        if is_main_process():
            now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            msg = (
                f"[{now_utc}] Epoch {epoch:03d} | "
                f"train_rmse_phys={tr_rmse_phys:.6e} | train_mae_phys={tr_mae_phys:.6e} | "
                f"val_rmse_phys={va_rmse_phys:.6e} | val_mae_phys={va_mae_phys:.6e} | "
            )

            if extra is not None:
                msg += (
                    f"val_rmse_peak_top5%GT={extra['rmse_peak']:.6e} | "
                    f"val_mae_peak_top5%GT={extra['mae_peak']:.6e} | "
                )

            if model_name == "perceiver3":
                nm_all = extra.get("node_attn_max_all", None) if extra else None
                nm_pk = extra.get("node_attn_max_peak", None) if extra else None
                ne_all = extra.get("node_attn_entropy_all", None) if extra else None
                ne_pk = extra.get("node_attn_entropy_peak", None) if extra else None
                tr_all = extra.get("time_attn_recent_all", None) if extra else None
                tr_pk = extra.get("time_attn_recent_peak", None) if extra else None

                msg += (
                    f"node_attn_max(val_all)={(nm_all if nm_all is not None else float('nan')):.3f} | "
                    f"node_attn_max(val_peak)={(nm_pk if nm_pk is not None else float('nan')):.3f} | "
                    f"node_attn_entropy(val_all)={(ne_all if ne_all is not None else float('nan')):.3f} | "
                    f"node_attn_entropy(val_peak)={(ne_pk if ne_pk is not None else float('nan')):.3f} | "
                    f"time_attn_recent(val_all)={(tr_all if tr_all is not None else float('nan')):.3f} | "
                    f"time_attn_recent(val_peak)={(tr_pk if tr_pk is not None else float('nan')):.3f} | "
                )
                if args.head_type == "dual":
                    gtr = gate_mean_train if gate_mean_train is not None else float("nan")
                    gall = extra.get("gate_mean_all", None) if extra else None
                    gpk = extra.get("gate_mean_peak", None) if extra else None
                    raw_model = model.module if isinstance(model, DDP) else model
                    alpha = float(torch.sigmoid(raw_model.alpha_logit).detach().cpu().item())
                    msg += (
                        f"gate_mean(train)={gtr:.3f} | "
                        f"gate_mean(val_all)={(gall if gall is not None else float('nan')):.3f} | "
                        f"gate_mean(val_peak)={(gpk if gpk is not None else float('nan')):.3f} | "
                        f"alpha={alpha:.3f} | "
                        f"tail_clip={args.tail_tanh_clip:.2f} | "
                    )

            msg += f"lr={cur_lr:.3e} | epoch_time={epoch_dt:.1f}s"
            print(msg)

            train_log.append((tr_mse_norm, tr_mse_phys, tr_rmse_phys, tr_mae_phys))
            val_log.append((va_mse_norm, va_mse_phys, va_rmse_phys, va_mae_phys))

            if va_rmse_phys < best_val_rmse_phys:
                best_val_rmse_phys = va_rmse_phys
                raw_model = model.module if isinstance(model, DDP) else model
                torch.save(
                    {
                        "model_state_dict": raw_model.state_dict(),
                        "args": vars(args),

                        # NEW: x_center/x_scale (robust or zscore, depending on args.x_norm)
                        "x_center": x_center_cpu,
                        "x_scale": x_scale_cpu,

                        # Backward compatibility keys (old scripts might expect these)
                        "x_mean": x_center_cpu,
                        "x_std": x_scale_cpu,

                        "x_norm": args.x_norm,
                        "x_clip": float(args.x_clip),
                        "x_p_lo": float(args.x_p_lo),
                        "x_p_hi": float(args.x_p_hi),
                        "x_nodes_per_graph": int(args.x_nodes_per_graph),
                        "x_robust_p_lo_vec": (p_lo_cpu if p_lo_cpu is not None else None),
                        "x_robust_p_hi_vec": (p_hi_cpu if p_hi_cpu is not None else None),

                        # NEW: p_mean normalization stats (scalar). Stored for reproducible ablations.
                        "pmean_center": (pmean_center_cpu if pmean_center_cpu is not None else None),
                        "pmean_scale" : (pmean_scale_cpu if pmean_scale_cpu is not None else None),

                        "y_mean": y_mean_cpu,
                        "y_std": y_std_cpu,
                        "train_log": train_log,
                        "val_log": val_log,
                        "best_val_rmse_phys": best_val_rmse_phys,
                    },
                    best_ckpt_path,
                )
                print(f"  [best] val_rmse_phys={best_val_rmse_phys:.6e} -> {best_ckpt_path}")

    if ddp_is_initialized():
        dist.barrier()

    total_train_dt = time.time() - train_t0
    if is_main_process():
        h = int(total_train_dt // 3600)
        m = int((total_train_dt % 3600) // 60)
        s = int(total_train_dt % 60)
        print(f"[Total training time] {h:02d}:{m:02d}:{s:02d} (hh:mm:ss)")

    # -------------------------
    # Load best checkpoint and test
    # -------------------------
    ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])

    y_mean_cpu = ckpt["y_mean"]
    y_std_cpu  = ckpt["y_std"]

    # Backward compatible load
    x_center_cpu = ckpt.get("x_center", ckpt.get("x_mean"))
    x_scale_cpu  = ckpt.get("x_scale",  ckpt.get("x_std"))

    x_center_dev = x_center_cpu.to(device=device, dtype=torch.float32)
    x_scale_dev  = x_scale_cpu.to(device=device, dtype=torch.float32).clamp_min(1e-6)
    # NEW: load p_mean stats (if present in checkpoint); keep None otherwise
    pmean_center_cpu = ckpt.get("pmean_center", None)
    pmean_scale_cpu  = ckpt.get("pmean_scale", None)
    pmean_center_dev = (pmean_center_cpu.to(device=device, dtype=torch.float32) if pmean_center_cpu is not None else None)
    pmean_scale_dev  = (pmean_scale_cpu.to(device=device, dtype=torch.float32).clamp_min(1e-6) if pmean_scale_cpu is not None else None)

    y_mean_dev = y_mean_cpu.to(device=device, dtype=torch.float32).view(1, -1)
    y_std_dev  = y_std_cpu.to(device=device, dtype=torch.float32).view(1, -1)

    te_mse_norm, te_mse_phys, te_rmse_phys, te_mae_phys = evaluate_ddp(
        model, test_loader, device,
        use_amp=use_amp, amp_dtype=amp_dtype,
        x_center=x_center_dev, x_scale=x_scale_dev, x_clip=float(ckpt.get("x_clip", args.x_clip)),
        pmean_center=pmean_center_dev, pmean_scale=pmean_scale_dev, pmean_clip=float(ckpt.get("x_clip", args.x_clip)),
        y_mean=y_mean_dev, y_std=y_std_dev,
        model_name=model_name,
        station_feat=station_feat,
    )

    if ddp_is_initialized():
        dist.barrier()

    # -------------------------
    # Save UNNORMALIZED test preds/GT (rank0 full set)
    # -------------------------
    if is_main_process():
        test_loader_full = build_loader(
            test_dataset, sampler=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            mp_context=mp_context,
        )

        y_true, y_pred, tags = collect_test_preds_unnorm(
            raw_model, test_loader_full, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
            x_center=x_center_dev, x_scale=x_scale_dev, x_clip=float(ckpt.get("x_clip", args.x_clip)),
            pmean_center=pmean_center_dev, pmean_scale=pmean_scale_dev, pmean_clip=float(ckpt.get("x_clip", args.x_clip)),
            y_mean=y_mean_dev, y_std=y_std_dev,
            model_name=model_name,
            station_feat=station_feat,
        )

        pred_dir = f"results_{station_tag}"
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(
            pred_dir,
            f"test_preds_{model_name}_{loss_tag}_{cfg_id}_{run_slug}.npz",
        )
        np.savez(pred_path, y_true=y_true, y_pred=y_pred, tags=tags)
        print(f"Saved test predictions (UNNORMALIZED) -> {pred_path}")

        print(f"[Test] rmse_phys={te_rmse_phys:.6e} | mae_phys={te_mae_phys:.6e}")

        summary_dir = f"results_{station_tag}"
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(
            summary_dir,
            f"summary_{model_name}_{loss_tag}_{cfg_id}_{run_slug}.npz",
        )
        np.savez(
            summary_path,
            best_val_rmse_phys=ckpt.get("best_val_rmse_phys", float("nan")),
            test_mse_norm=te_mse_norm,
            test_mse_phys=te_mse_phys,
            test_rmse_phys=te_rmse_phys,
            test_mae_phys=te_mae_phys,
        )
        print(f"Saved summary -> {summary_path}")

    if ddp_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Finished!")
        # try:
        #     if dist.is_available() and dist.is_initialized():
        #         dist.destroy_process_group()
        # except Exception:
        #     pass
