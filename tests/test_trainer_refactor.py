"""
Tests for the trainer split and the shared-ownership rules of Phase 1.

The point of these is structural, not numerical: the refactor is only meaningful
if `train_xgboost.py` genuinely cannot reach into the LSTM trainer, and if no
stale `train.py` command survives anywhere a user might copy it from.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def module_imports(path: Path) -> set[str]:
    """Top-level module names imported by a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
            names.add(node.module)
    return names


class TrainerLayoutTest(unittest.TestCase):
    def test_train_lstm_exists_and_train_py_is_gone(self):
        self.assertTrue((ROOT / "train_lstm.py").exists())
        self.assertFalse((ROOT / "train.py").exists())

    def test_shared_module_exists(self):
        self.assertTrue((ROOT / "src" / "training_common.py").exists())

    def test_xgboost_trainer_does_not_import_the_lstm_trainer(self):
        """The dependency direction has to run through the shared module."""
        imports = module_imports(ROOT / "train_xgboost.py")
        self.assertNotIn("train_lstm", imports)
        self.assertNotIn("train", imports)
        self.assertIn("src.training_common", imports)

    def test_lstm_trainer_does_not_import_the_xgboost_trainer(self):
        imports = module_imports(ROOT / "train_lstm.py")
        self.assertNotIn("train_xgboost", imports)
        self.assertIn("src.training_common", imports)

    def test_shared_helpers_are_not_owned_by_a_trainer(self):
        """
        Helpers used by both families must live in the shared module.

        A trainer re-defining one of these is how the two families silently drift
        apart into two different evaluation methodologies.
        """
        shared = {
            "load_config",
            "get_horizons",
            "parse_horizon_list",
            "load_horizon_dataset",
            "scale_features",
            "strip_arrays",
            "artifact_path",
            "print_metrics",
            "json_safe",
        }
        import src.training_common as common

        for name in shared:
            self.assertTrue(hasattr(common, name), f"{name} missing from src.training_common")

        for trainer in ("train_lstm.py", "train_xgboost.py"):
            tree = ast.parse((ROOT / trainer).read_text(encoding="utf-8"))
            defined = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            overlap = defined & shared
            self.assertFalse(overlap, f"{trainer} re-defines shared helpers: {overlap}")


class ObsoleteReferenceTest(unittest.TestCase):
    """
    Nothing runnable may still point at `train.py`.

    Two different standards apply, because the risk differs. In *code* any
    mention is suspect: a bare `"train.py"` string is almost certainly a
    subprocess argument or an import path. In *prose* only a copyable command is
    suspect — documentation has to be able to say "train.py was renamed to
    train_lstm.py" without that being flagged as a stale reference, otherwise the
    migration cannot be described at all.
    """

    # Any mention at all, for source files.
    CODE_PATTERN = re.compile(r"(?<![\w_])train\.py")
    # Only something a reader could paste into a shell, for documentation.
    COMMAND_PATTERN = re.compile(r"(?:python[0-9.]*\s+|\./|\$\s*)train\.py")

    CODE_SUFFIXES = {".py", ".sh", ".bat", ".ipynb"}
    PROSE_SUFFIXES = {".md", ".yaml", ".yml"}

    def candidate_files(self):
        for pattern in ("*.py", "*.md", "*.yaml", "*.yml", "*.sh", "*.bat", "*.ipynb"):
            for path in ROOT.rglob(pattern):
                parts = set(path.parts)
                if parts & {".git", "logs", "data", ".venv", "__pycache__"}:
                    continue
                # This file necessarily contains the string it is searching for.
                if path.resolve() == Path(__file__).resolve():
                    continue
                yield path

    def test_no_obsolete_train_py_references(self):
        offenders: list[str] = []
        for path in self.candidate_files():
            if path.suffix in self.CODE_SUFFIXES:
                pattern = self.CODE_PATTERN
            elif path.suffix in self.PROSE_SUFFIXES:
                pattern = self.COMMAND_PATTERN
            else:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:100]}")
        self.assertFalse(offenders, "obsolete train.py references:\n" + "\n".join(offenders))

    def test_the_scanner_still_catches_a_real_stale_command(self):
        """Guard against the tightened pattern being too permissive to be useful."""
        for line in (
            "python3 train.py --horizon 21",
            "python train.py",
            "$ train.py --horizon 5",
            "./train.py",
        ):
            self.assertTrue(self.COMMAND_PATTERN.search(line), line)
        for line in ("`train.py` was renamed to `train_lstm.py`", "train.py -> train_lstm.py"):
            self.assertIsNone(self.COMMAND_PATTERN.search(line), line)

    def test_notebook_uses_the_new_command(self):
        notebook = ROOT / "notebooks" / "final_project_colab.ipynb"
        if not notebook.exists():
            self.skipTest("notebook not present")
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        text = json.dumps(payload)
        self.assertNotIn("'train.py'", text)
        self.assertIn("train_lstm.py", text)


class ConfigurationCoherenceTest(unittest.TestCase):
    """
    The shipped config must describe what the code actually does.

    Dead settings are worse than missing ones: a reader tuning
    `early_stopping_rounds` on a trainer that no longer early-stops would be
    changing nothing and concluding it made no difference.
    """

    def config(self) -> dict:
        import yaml

        with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def test_every_key_the_code_reads_is_present(self):
        config = self.config()
        required = [
            ("regression_target", "mode"),
            ("regression_loss", "mse_weight"),
            ("regression_loss", "huber_weight"),
            ("regression_loss", "cross_sectional_ic_weight"),
            ("regression_loss", "huber_beta"),
            ("market_model", "maximum_shrinkage"),
            ("portfolio", "top_k"),
            ("walk_forward", "folds"),
            ("uncertainty", "confidence_level"),
            ("xgboost", "n_estimators"),
        ]
        for path in required:
            node = config
            for key in path:
                self.assertIn(key, node, f"missing config key: {'.'.join(path)}")
                node = node[key]

        for key in (
            "early_stopping_metric",
            "selection_score_magnitude_weight",
            "lstm_batching",
            "dates_per_batch",
            "exclude_market_wide_features",
            "minimum_sector_members",
            "feature_blocklist",
            "ticker_sectors",
            "acceptance_thresholds",
        ):
            self.assertIn(key, config, f"missing config key: {key}")

    def test_no_setting_survives_for_a_feature_that_was_removed(self):
        config = self.config()
        # Early stopping was replaced by round-ladder selection on folds.
        self.assertNotIn("early_stopping_rounds", config["xgboost"])
        # The old Huber+correlation loss was replaced by the composite objective.
        self.assertNotIn("correlation_weight", config["regression_loss"])

    def test_loss_weights_are_valid(self):
        from src.losses import resolve_loss_config

        resolved = resolve_loss_config(self.config()["regression_loss"])
        weights = [
            resolved["mse_weight"],
            resolved["huber_weight"],
            resolved["cross_sectional_ic_weight"],
        ]
        self.assertTrue(all(weight >= 0 for weight in weights))
        self.assertGreater(sum(weights), 0)

    def test_target_mode_is_one_the_code_supports(self):
        from src.regression import TARGET_MODES, resolve_target_config

        mode = resolve_target_config(self.config()["regression_target"])["mode"]
        self.assertIn(mode, TARGET_MODES)

    def test_every_ticker_has_a_sector(self):
        config = self.config()
        unmapped = [t for t in config["tickers"] if t not in config["ticker_sectors"]]
        self.assertFalse(unmapped, f"tickers with no sector: {unmapped}")

    def test_checkpoint_metric_is_the_combined_criterion(self):
        """Guard the fix: selecting on MSE or IC alone each has a failure mode."""
        from src.regression import SELECTION_SCORE_KEY

        self.assertEqual(self.config()["early_stopping_metric"], SELECTION_SCORE_KEY)


class CommandLineInterfaceTest(unittest.TestCase):
    """Both trainers must expose a working --help without side effects."""

    def run_help(self, script: str) -> str:
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        return completed.stdout

    def test_train_lstm_cli(self):
        output = self.run_help("train_lstm.py")
        for flag in ("--horizon", "--horizons", "--walk-forward", "--ensemble-size"):
            self.assertIn(flag, output)

    def test_train_xgboost_cli(self):
        output = self.run_help("train_xgboost.py")
        for flag in ("--horizon", "--horizons", "--walk-forward"):
            self.assertIn(flag, output)

    def test_train_all_models_cli(self):
        output = self.run_help("train_all_models.py")
        self.assertIn("--compare", output)


if __name__ == "__main__":
    unittest.main()
