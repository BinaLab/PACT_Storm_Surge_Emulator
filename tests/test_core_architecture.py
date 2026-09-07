"""Core-study controls, lag semantics, and current checkpoint loading."""

import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

import infer
import train
from emulator.models import GridCNNEncoder, PACT
from emulator.models.architectures import _make_graphsage_layers
from emulator.data import station_features_from_json


class CoreArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def make_model(**kwargs):
        return PACT(5, 8, 6, num_layers=2, n_node_read_heads=2,
                    n_time_read_heads=2, n_transformer_layers=2,
                    max_time_steps=9, transformer_dropout=0,
                    station_feat_dim=7, **kwargs)

    @staticmethod
    def make_batch(steps, pressure=False):
        generator = torch.Generator().manual_seed(91)
        graphs = []
        for _ in range(2):
            full_history = torch.randn(4, 9, 5, generator=generator)
            graph = Data(
                x=full_history[:, -1], x_hist=full_history[:, -steps:],
                edge_index=torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]]),
                y=torch.randn(1, 6, generator=generator), grid_H=2, grid_W=2,
            )
            if pressure:
                graph.p_mean_hist = torch.arange(9, dtype=torch.float32)[-steps:].view(1, -1)
            graphs.append(graph)
        return Batch.from_data_list(graphs)

    def test_cnn_matches_budget_without_changing_output_shape_or_spatial_depth(self):
        cnn = GridCNNEncoder(5, 128, 2, 0)
        sage = _make_graphsage_layers(5, 128, 2)
        self.assertEqual(sum(p.numel() for p in cnn.parameters()), 34870)
        self.assertEqual(sum(p.numel() for p in sage.parameters()), 34304)
        self.assertEqual([(c.in_channels, c.out_channels) for c in cnn.layers], [(5, 29), (29, 128)])
        result = cnn(torch.randn(2 * 3 * 4, 5), (2, 3, 4))
        self.assertEqual(result.shape, (24, 128))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in cnn.parameters()))
        for depth in (1, 3):
            other = GridCNNEncoder(5, 128, depth, 0)
            self.assertEqual(len(other.layers), depth)
            self.assertEqual(other.layers[-1].out_channels, 128)
        with self.assertRaisesRegex(ValueError, "intermediate_channel"):
            GridCNNEncoder(5, 128, 2, 0, intermediate_channel=0)

    def test_overlapping_lags_keep_the_same_memory_before_each_temporal_family(self):
        station = torch.arange(7, dtype=torch.float32) / 7
        for encoder, temporal, pressure in itertools.product(
            ("GraphSAGE", "CNN"), ("MLP", "LSTM", "GRU", "Transformer"), (False, True)
        ):
            with self.subTest(encoder=encoder, temporal=temporal, pressure=pressure):
                model = self.make_model(encoder_type=encoder, temporal_block=temporal,
                                        head_type="single", use_pmean_tokens=pressure).eval()
                middle = (model.temporal_mlp[0] if temporal == "MLP" else
                          model.transformer[0] if temporal == "Transformer" else model.temporal_rnn)
                memories, indices = [], []
                h1 = middle.register_forward_pre_hook(lambda module, args: memories.append(args[0].detach().clone()))
                h2 = model.lag_embed.register_forward_pre_hook(lambda module, args: indices.append(args[0].tolist()))
                with torch.no_grad():
                    model(self.make_batch(3, pressure), station_feat=station)
                    model(self.make_batch(9, pressure), station_feat=station)
                h1.remove()
                h2.remove()
                self.assertEqual(indices, [[2, 1, 0], list(range(8, -1, -1))])
                short, long = memories
                torch.testing.assert_close(short[:, :3], long[:, 6:9], rtol=0, atol=0)
                if pressure:
                    torch.testing.assert_close(short[:, 3:], long[:, 15:18], rtol=0, atol=0)

    def test_all_histories_keep_parameter_count_and_residual_is_only_for_zero_hours(self):
        station = torch.arange(7, dtype=torch.float32) / 7
        for encoder, temporal, head in itertools.product(
            ("GraphSAGE", "CNN"), ("MLP", "LSTM", "GRU", "Transformer"), ("single", "dual")
        ):
            with self.subTest(encoder=encoder, temporal=temporal, head=head), torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model = self.make_model(encoder_type=encoder, temporal_block=temporal, head_type=head).eval()
                parameter_count = sum(p.numel() for p in model.parameters())
                attention, queries, head_inputs = [], [], []
                def capture_attention(module, args, output):
                    queries.append(args[0].detach().clone())
                    attention.append(output[0].detach().clone())
                h1 = model.forecast_attn.register_forward_hook(capture_attention)
                h2 = model.mlp_base.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0].detach().clone()))
                for steps in range(1, 10):
                    model.zero_grad(set_to_none=True)
                    batch = self.make_batch(steps)
                    prediction = model(batch, station_feat=station)
                    self.assertEqual(prediction.shape, (2, 6))
                    self.assertTrue(torch.isfinite(prediction).all())
                    expected = attention[-1] + queries[-1] if steps == 1 else attention[-1]
                    torch.testing.assert_close(head_inputs[-1].view(2, 6, 8), expected, rtol=0, atol=0)
                    (prediction - batch.y).square().mean().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                    if steps == 1:
                        self.assertGreater(prediction.detach().std(dim=1).min().item(), 1e-5)
                        self.assertGreater(model.horizon_embed.weight.grad.abs().max().item(), 1e-6)
                    if steps in (1, 3, 9):
                        with torch.no_grad():
                            auxiliary_prediction, _ = model(batch, station_feat=station, return_aux=True)
                        torch.testing.assert_close(prediction, auxiliary_prediction, rtol=1e-5, atol=1e-6)
                    self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
                h1.remove()
                h2.remove()

    def test_inference_restores_current_architecture_and_rejects_old_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graphs = root / "graphs"
            graphs.mkdir()
            batch = self.make_batch(3)
            raw_graphs = batch.to_data_list()
            for graph in raw_graphs:
                graph.x_hist = graph.x_hist.permute(1, 0, 2).contiguous()
            torch.save(raw_graphs, graphs / "2000_2001_Battery_fixture_graphs.pt")
            for encoder in ("GraphSAGE", "CNN"):
                for temporal in ("MLP", "LSTM", "GRU", "Transformer"):
                    model = self.make_model(encoder_type=encoder, temporal_block=temporal,
                                            head_type="single", cnn_intermediate_channel=32).eval()
                    ckpt = dict(args=dict(model="perceiver3", hidden_channels=8, history_hours=12,
                                          num_layers=2, node_read_heads=2, time_read_heads=2,
                                          transformer_layers=2, max_time_steps=9, encoder_type=encoder,
                                          temporal_block=temporal, head_type="single",
                                          cnn_intermediate_channel=32),
                                model_state_dict=model.state_dict(), x_center=torch.zeros(5),
                                x_scale=torch.ones(5), x_clip=0,
                                y_mean=torch.zeros(6), y_std=torch.ones(6), time_encoding="relative_lag")
                    path = root / f"{encoder}_{temporal}.pth"
                    torch.save(ckpt, path)
                    # Provide the same feature dimension while keeping JSON values explicit.
                    stations = root / "stations"
                    stations.mkdir(exist_ok=True)
                    (stations / "Battery.json").write_text(json.dumps(dict(lat=0, lon=0)))
                    with torch.no_grad():
                        expected = model(batch, station_feat=station_features_from_json(dict(lat=0, lon=0)))
                    out = root / f"out_{encoder}_{temporal}"
                    argv = ["infer.py", "--ckpt", str(path), "--root_dir", str(graphs),
                            "--test_root_dir", str(graphs), "--station", "Battery", "--station_json_dir", str(stations),
                            "--out_dir", str(out), "--save_npz", "--num_workers", "0", "--batch_size", "2"]
                    with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                        infer.main()
                    meta = json.loads(next(out.glob("metrics*.json")).read_text())
                    self.assertEqual(meta["time_encoding"], "relative_lag")
                    self.assertEqual(meta["cnn_intermediate_channel"], 32 if encoder == "CNN" else None)
                    with np.load(next(out.glob("preds*.npz")), allow_pickle=True) as arrays:
                        np.testing.assert_allclose(arrays["y_pred"], expected.numpy(), rtol=1e-5, atol=1e-6)
                    if encoder == "CNN" and temporal == "MLP":
                        with patch.object(sys, "argv", [*argv, "--cnn_intermediate_channel", "29"]), self.assertRaisesRegex(ValueError, "does not match the checkpoint"):
                            infer.main()
                    if temporal == "MLP":
                        # Old embedding keys must fail strict loading, rather than
                        # silently interpreting sequence positions as physical lags.
                        state = ckpt["model_state_dict"]
                        state["time_embed.weight"] = state.pop("lag_embed.weight")
                        torch.save(ckpt, path)
                        with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "lag_embed.weight"):
                            infer.main()

    def test_inference_requires_explicit_cnn_checkpoint_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing_width.pth"
            for family in ("baseline", "perceiver3"):
                torch.save(dict(args=dict(model=family, encoder_type="CNN", hidden_channels=8)), path)
                argv = ["infer.py", "--ckpt", str(path), "--root_dir", tmp]
                with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), self.assertRaisesRegex(ValueError, "missing cnn_intermediate_channel"):
                    infer.main()

    def test_train_checkpoint_and_inference_support_zero_and_positive_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graphs, stations = root / "graphs", root / "stations"
            graphs.mkdir()
            stations.mkdir()
            (stations / "Battery.json").write_text(json.dumps(dict(lat=40, lon=-74)))
            for year in range(2000, 2005):
                items = self.make_batch(9).to_data_list()
                for graph in items:
                    graph.x_hist = graph.x_hist.permute(1, 0, 2).contiguous()
                    graph.y += (year - 2000) * 0.2
                torch.save(items, graphs / f"{year}_{year+1}_Battery_fixture_graphs.pt")
            for history in (0, 12):
                width = 29 if history == 0 else 32
                out = root / f"training_{history}"
                argv = ["train.py", "--root_dir", str(graphs), "--station", "Battery",
                        "--station_json_dir", str(stations), "--output_dir", str(out),
                        "--model", "perceiver3", "--encoder_type", "CNN", "--temporal_block", "MLP",
                        "--head_type", "single", "--history_hours", str(history), "--epochs", "1",
                        "--hidden_channels", "8", "--node_read_heads", "2", "--time_read_heads", "2",
                        "--transformer_layers", "1", "--batch_size", "2", "--num_workers", "0",
                        "--loss_mode", "mse", "--x_norm", "zscore", "--x_aug", "0", "--x_clip", "0"]
                if history:
                    argv += ["--cnn_intermediate_channel", str(width)]
                with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                checkpoint = next(out.glob("best*.pth"))
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                self.assertEqual(saved["args"]["cnn_intermediate_channel"], width)
                self.assertEqual(saved["time_encoding"], "relative_lag")
                self.assertIn("lag_embed.weight", saved["model_state_dict"])
                self.assertEqual(saved["model_state_dict"]["cnn_encoder.layers.0.weight"].shape, (width, 5, 3, 3))
                snapshot = (out / "config_used.sh").read_text()
                self.assertIn(f"CNN_INTERMEDIATE_CHANNEL={width}", snapshot)
                self.assertNotIn("TIME_ENCODING=", snapshot)
                output = root / f"inference_{history}"
                argv = ["infer.py", "--ckpt", str(checkpoint), "--root_dir", str(graphs),
                        "--station_json_dir", str(stations), "--out_dir", str(output), "--save_npz", "--num_workers", "0"]
                with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    infer.main()
                metadata = json.loads(next(output.glob("metrics*.json")).read_text())
                self.assertEqual(metadata["time_encoding"], "relative_lag")
                self.assertEqual(metadata["cnn_intermediate_channel"], width)
                self.assertEqual(metadata["zero_history_query_residual"], history == 0)
                self.assertTrue(np.isfinite(metadata["results"]["_overall"]["rmse"]))
                with np.load(next(out.glob("test_preds*.npz")), allow_pickle=True) as trained, np.load(next(output.glob("preds*.npz")), allow_pickle=True) as inferred:
                    np.testing.assert_array_equal(trained["tags"], inferred["tags"])
                    np.testing.assert_allclose(trained["y_pred"], inferred["y_pred"], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
