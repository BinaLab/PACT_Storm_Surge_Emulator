"""End-to-end regression tests for evaluation population and output ordering."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data import Data

import infer
from emulator.data import build_loader
from emulator.models import SpatialOnlyGraphSAGEBatch


class InferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graphs = self.root / "graphs"
        self.graphs.mkdir()
        self.years = [1979, 1980, 1981, 1982, 1983, 1984, 2070, 2071, 2072, 2073, 2074, 2075]
        for i, year in enumerate(self.years):
            # Unequal year sizes exercise sample-weighted overall metrics.
            graphs = [Data(
                x=torch.ones(4, 5),
                edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
                y=torch.tensor([float(i + 1), float(i + 2)]),
            ) for _ in range(1 + i % 3)]
            torch.save(graphs, self.graphs / f"{year}_{year + 1}_Battery_fixture_graphs.pt")
        model = SpatialOnlyGraphSAGEBatch(5, 8, 2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        self.ckpt = dict(
            model_state_dict=model.state_dict(),
            args=dict(model="baseline", history_hours=0, hidden_channels=8,
                      station="Battery", train_ratio=0.5, val_ratio=0.25, seed=7),
            x_center=torch.zeros(5), x_scale=torch.ones(5),
            y_mean=torch.zeros(2), y_std=torch.ones(2),
        )

    def run_inference(self, name, options=(), checkpoint_args=None):
        self.ckpt["args"].update(checkpoint_args or {})
        ckpt_path = self.root / f"{name}.pth"
        torch.save(self.ckpt, ckpt_path)
        out_dir = self.root / name
        argv = ["infer.py", "--ckpt", str(ckpt_path), "--root_dir", str(self.graphs),
                "--out_dir", str(out_dir), "--num_workers", "0", "--batch_size", "2", *options]
        stdout = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
            infer.main()
        metadata = json.loads(next(out_dir.glob("metrics*.json")).read_text())
        return metadata, out_dir, stdout.getvalue()

    def test_checkpoint_ratios_and_shuffle_define_test_set(self):
        regular, _, _ = self.run_inference("ratio")
        self.assertEqual(regular["inference_args"]["batch_size"], 2)
        self.assertEqual(regular["checkpoint_args"]["train_ratio"], 0.5)
        self.assertEqual(regular["years_evaluated"], ["2073_2074", "2074_2075", "2075_2076"])
        shuffled, _, _ = self.run_inference("shuffle", checkpoint_args={"shuffle_years": 1})
        self.assertEqual(shuffled["years_evaluated"], ["1981_1982", "1984_1985", "2070_2071"])
        self.assertEqual(shuffled["split_parameters"]["split_seed"], 7)

    def test_future_filter_and_threshold_come_from_checkpoint(self):
        metadata, _, _ = self.run_inference("future", checkpoint_args={
            "future_only": 1, "future_year_threshold": 2073,
            "train_ratio": 0.6, "val_ratio": 0.2,
        })
        self.assertEqual(metadata["years_evaluated"], ["2075_2076"])

    def test_selected_year_metrics_do_not_depend_on_saving_npz(self):
        options = ["--test_root_dir", str(self.graphs), "--years", "1979_1980,2070_2071"]
        plain, plain_dir, log = self.run_inference("plain", options)
        saved, saved_dir, _ = self.run_inference("saved", [*options, "--save_npz"])
        self.assertIn("not a held-out test", log)
        self.assertEqual(plain["evaluation_scope"], "external_all_years")
        self.assertFalse(list(plain_dir.glob("*.npz")))
        with np.load(next(saved_dir.glob("*.npz")), allow_pickle=True) as arrays:
            error = arrays["y_pred"] - arrays["y_true"]
            self.assertAlmostEqual(plain["results"]["_overall"]["rmse"], float(np.sqrt(np.mean(error ** 2))), places=6)
            tags = arrays["tags"].tolist()
            self.assertEqual(tags, sorted(tags))
        for key in ("_overall", "_overall_past", "_overall_future"):
            self.assertEqual(plain["results"][key], saved["results"][key])
        self.assertIn("n_graphs=1", log)

    def test_loader_only_shuffles_when_requested(self):
        dataset = [Data(x=torch.ones(1, 1), tag=str(i)) for i in range(8)]
        kwargs = dict(dataset=dataset, sampler=None, batch_size=2, num_workers=0,
                      pin_memory=False, persistent_workers=False, prefetch_factor=0, mp_context="fork")
        evaluation = build_loader(**kwargs)
        self.assertIsInstance(evaluation.sampler, SequentialSampler)
        self.assertEqual([tag for batch in evaluation for tag in batch.tag], [str(i) for i in range(8)])
        self.assertIsInstance(build_loader(**kwargs, shuffle=True).sampler, RandomSampler)

    def test_pressure_checkpoint_can_evaluate_graphs_without_pressure(self):
        model = SpatialOnlyGraphSAGEBatch(5, 8, 2, use_pmean=True, pmean_dim=4)
        self.ckpt["model_state_dict"] = model.state_dict()
        with self.assertWarnsRegex(RuntimeWarning, "p_mean_curr is missing"):
            metadata, _, _ = self.run_inference("missing_pressure", checkpoint_args={
                "use_pmean": True, "pmean_dim": 4,
            })
        self.assertTrue(np.isfinite(metadata["results"]["_overall"]["rmse"]))


if __name__ == "__main__":
    unittest.main()
