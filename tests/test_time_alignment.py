"""Regression checks for timestamp corruption in station preprocessing."""

from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from preprocessing.time_align_unified import (
    build_forcing_time_index,
    compute_last_full_day_for_year,
    process_one_pair,
)


class StationTimeAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pact-time-alignment-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "dicts").mkdir()
        (self.root / "graphs").mkdir()
        self.csv_path = self.root / "2000_2001_Battery.csv"
        times = pd.date_range("2000-10-25 01:00", "2000-11-01 23:00", freq="h")
        self.frame = pd.DataFrame({
            "time": times,
            "nc": np.arange(len(times), dtype=float),
            "nc_tide": np.zeros(len(times)),
        })

    def build_graphs(self):
        process_one_pair(
            year=2000,
            station="Battery",
            forcing=np.zeros((32, 2, 2, 5), dtype=np.float32),
            p_mean_t=None,
            t_forcing=build_forcing_time_index(2000, 32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            H=2,
            W=2,
            version="peryear",
            out_root_fixed=self.root,
            out_root_peryear=self.root,
            csv_dir=self.root,
            last_full_day=date(2000, 11, 1),
        )

    def test_valid_timestamps_preserve_label_alignment(self):
        self.frame.to_csv(self.csv_path, index=False)
        with redirect_stdout(StringIO()):
            cutoff = compute_last_full_day_for_year(2000, ["Battery"], self.root)
            self.build_graphs()
        self.assertEqual(cutoff, date(2000, 11, 1))
        graphs = torch.load(
            next((self.root / "graphs").glob("*.pt")),
            map_location="cpu", weights_only=False,
        )
        self.assertEqual(len(graphs), 4)
        self.assertEqual(graphs[0].center_time, "2000-11-01 00:00:00")
        expected = self.frame.loc[
            self.frame["time"].between("2000-11-01 00:00", "2000-11-01 05:00"), "nc"
        ].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(graphs[0].y.numpy(), expected)

    def test_shifted_missing_and_duplicate_hours_fail_in_both_entrypoints(self):
        shifted = self.frame.copy()
        shifted["time"] += pd.Timedelta(hours=1)
        missing = self.frame.drop(index=20)
        duplicate = self.frame.copy()
        duplicate.loc[20, "time"] = duplicate.loc[19, "time"]
        invalid = self.frame.copy()
        invalid.loc[20, "time"] = pd.NaT
        for name, frame in [("shifted", shifted), ("missing", missing), ("duplicate", duplicate), ("invalid", invalid)]:
            with self.subTest(name=name):
                frame.to_csv(self.csv_path, index=False)
                with self.assertRaisesRegex(ValueError, "Station timestamp mismatch"):
                    compute_last_full_day_for_year(2000, ["Battery"], self.root)
                with self.assertRaisesRegex(ValueError, "Station timestamp mismatch"):
                    self.build_graphs()
        self.assertEqual(list((self.root / "graphs").iterdir()), [])

    def test_legacy_csv_without_times_keeps_explicit_fallback(self):
        self.frame.drop(columns="time").to_csv(self.csv_path, index=False)
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                compute_last_full_day_for_year(2000, ["Battery"], self.root),
                date(2000, 11, 1),
            )
            self.build_graphs()
        self.assertIn("legacy assumption", output.getvalue())
        self.assertEqual(len(list((self.root / "graphs").glob("*.pt"))), 1)


if __name__ == "__main__":
    unittest.main()
