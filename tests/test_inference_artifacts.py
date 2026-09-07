"""Launchers save resolved configs and replay commands without mutable config dependencies."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class InferenceArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pact artifacts ' $ ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.python_env = self.root / "python env"
        (self.python_env / "bin").mkdir(parents=True)
        (self.python_env / "bin" / "python").symlink_to(sys.executable)
        self.env = os.environ.copy()
        for name in ("BASH_ENV", "ENV", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SESSION_NAME"):
            self.env.pop(name, None)
        self.env.update(PATH=str(self.bin_dir) + os.pathsep + self.env["PATH"],
                        CONDA_PREFIX=str(self.python_env),
                        TORCH_THREADS="3",
                        TMUX_TEST_CAPTURE=str(self.root / "tmux_command.txt"))
        tmux = self.bin_dir / "tmux"
        tmux.write_text(f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\n"
                        "Path(os.environ['TMUX_TEST_CAPTURE']).write_text(sys.argv[-1])\n")
        tmux.chmod(0o755)
        # Avoid querying real GPU hardware in this launcher-only test.
        probe = self.bin_dir / "nvidia-smi"
        probe.write_text("#!/usr/bin/env bash\nexit 0\n")
        probe.chmod(0o755)
        self.fake_infer = self.root / "record args.py"
        self.fake_infer.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "args=sys.argv[1:]\n"
            "out=Path(args[args.index('--out_dir')+1])\n"
            "out.mkdir(parents=True,exist_ok=True)\n"
            "(out/'invocation.json').write_text(json.dumps(args))\n"
        )
        checkpoint_dir = self.root / "check points"
        checkpoint_dir.mkdir()
        old = checkpoint_dir / "old.pth"
        self.selected = checkpoint_dir / "selected.pth"
        old.touch()
        self.selected.touch()
        os.utime(old, (100, 100))
        os.utime(self.selected, (200, 200))
        self.common = self.root / "common config.sh"
        self.label = "P3 label ' $literal;()"
        values = dict(INFER_PY=str(self.fake_infer), ROOT_DIR="./source data/graphs",
                      TEST_ROOT_DIR="./target data/graphs", STATION="Battery", MODEL="baseline",
                      MODEL_LABEL=self.label, BATCH_SIZE="7", DO_CONDA="0", TORCH_GPU_PROBE="0",
                      USE_AMP="0", USE_TF32="0", CKPT_PATH="./check points/*.pth",
                      CNN_INTERMEDIATE_CHANNEL="29", TIME_ENCODING="relative_lag",
                      INFERENCE_RESULTS_ROOT="./results with spaces")
        self.common.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n")
        self.config = self.root / "leaf config.sh"
        self.config.write_text(f"source {shlex.quote(str(self.common))}\nBATCH_SIZE=11\n"
                               "RUNS=('first|./target data/graphs' 'second|')\n")

    def run_bash(self, *args, cwd=None):
        result = subprocess.run(["bash", *map(str, args)], cwd=cwd or self.root,
                                env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def read_snapshot(self, snapshot):
        script = ('source "$1"; printf "%s\\0" "$BATCH_SIZE" "$CKPT_PATH" '
                  '"$MODEL_LABEL" "$ROOT_DIR" "$TEST_ROOT_DIR" "${RUNS[@]}" "$NUM_WORKERS"')
        output = self.run_bash("-c", script, "snapshot", snapshot, cwd="/tmp").stdout
        return output.rstrip("\0").split("\0")

    def assert_command_replays(self, run_dir):
        record = run_dir / "outputs" / "invocation.json"
        original = json.loads(record.read_text())
        command = run_dir / "command_used.sh"
        self.run_bash("-n", command)
        self.run_bash(command, cwd="/tmp")
        self.assertEqual(json.loads(record.read_text()), original)
        self.assertEqual(original[original.index("--batch_size") + 1], "11")
        self.assertEqual(original[original.index("--ckpt") + 1], str(self.selected.resolve()))
        self.assertEqual(original[original.index("--model_label") + 1], self.label)
        self.assertEqual(original[original.index("--cnn_intermediate_channel") + 1], "29")
        self.assertEqual(original[original.index("--time_encoding") + 1], "relative_lag")

    def test_tmux_uses_snapshot_even_after_sources_and_glob_change(self):
        self.run_bash(PROJECT / "infer.sh", self.config)
        snapshot = next((self.root / "results with spaces").glob("*/infer_config_used.sh"))
        self.config.unlink()
        self.common.unlink()
        newer = self.selected.parent / "newer.pth"
        newer.touch()
        os.utime(newer, (300, 300))
        values = self.read_snapshot(snapshot)
        self.assertEqual(values[:3], ["11", str(self.selected.resolve()), self.label])
        self.assertEqual(values[3], str(self.root / "source data/graphs"))
        self.assertEqual(values[4], str(self.root / "target data/graphs"))
        self.assertEqual(values[-1], "0")  # Launcher default absent from the input configs.
        threads = self.run_bash("-c", 'source "$1"; printf "%s" "$TORCH_THREADS"',
                                "snapshot", snapshot).stdout
        self.assertEqual(threads, "3")  # Resolved value inherited from the launch environment.
        runner = snapshot.parent / "run_infer.sh"
        self.run_bash("-n", runner)
        self.run_bash("-c", (self.root / "tmux_command.txt").read_text(), cwd="/tmp")
        self.assert_command_replays(snapshot.parent)

    def test_multi_saves_independent_config_and_command_for_each_target(self):
        self.run_bash(PROJECT / "infer_multi.sh", self.config)
        snapshots = list((self.root / "results with spaces").glob("*/infer_config_used.sh"))
        self.assertEqual(len(snapshots), 2)
        self.config.unlink()
        self.common.unlink()
        targets = set()
        for snapshot in snapshots:
            values = self.read_snapshot(snapshot)
            self.assertEqual(values[:3], ["11", str(self.selected.resolve()), self.label])
            self.assertEqual(len(values), 7)  # RUNS has only this target, not the whole sweep.
            targets.add(values[4])
            self.assertEqual(values[5].split("|", 1)[1], values[4])
            self.assert_command_replays(snapshot.parent)
        self.assertEqual(targets, {"", str(self.root / "target data/graphs")})

    def test_empty_architecture_expectations_are_not_passed_to_inference(self):
        with self.config.open("a") as config:
            config.write("CNN_INTERMEDIATE_CHANNEL=''\nTIME_ENCODING=''\n")
        self.run_bash(PROJECT / "infer_multi.sh", self.config)
        for record in (self.root / "results with spaces").glob("*/outputs/invocation.json"):
            args = json.loads(record.read_text())
            self.assertNotIn("--cnn_intermediate_channel", args)
            self.assertNotIn("--time_encoding", args)

    def test_training_launcher_saves_and_passes_architecture_settings(self):
        fake_train = self.root / "record train.py"
        fake_train.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "args=sys.argv[1:]\n"
            "out=Path(args[args.index('--output_dir')+1])\n"
            "(out/'train_invocation.json').write_text(json.dumps(args))\n"
        )
        with self.config.open("a") as config:
            config.write(f"TRAIN_PY={shlex.quote(str(fake_train))}\n"
                         "MODEL=perceiver3\nENCODER_TYPE=CNN\nnum_gpus=1\nUSE_TMUX=0\n"
                         "HISTORY_HOURS_LIST=(0)\nALL_RESULTS_ROOT='./train results'\n")
        env_before = self.env
        self.env = self.env.copy()
        for key in ("SLURM_NTASKS_PER_NODE", "SLURM_JOB_ID", "PACT_RUNSTAMP", "PACT_RUN_NAME"):
            self.env.pop(key, None)
        try:
            self.run_bash(PROJECT / "train.sh", self.config)
        finally:
            self.env = env_before
        record = next((self.root / "train results").glob("*/train_invocation.json"))
        args = json.loads(record.read_text())
        self.assertEqual(args[args.index("--cnn_intermediate_channel") + 1], "29")
        self.assertEqual(args[args.index("--time_encoding") + 1], "relative_lag")
        self.assertEqual(args[args.index("--history_hours") + 1], "0")
        values = self.run_bash("-c", 'source "$1"; printf "%s\\n" "$CNN_INTERMEDIATE_CHANNEL" "$TIME_ENCODING"',
                               "snapshot", record.parent / "config_used.sh").stdout.splitlines()
        self.assertEqual(values, ["29", "relative_lag"])


if __name__ == "__main__":
    unittest.main()
