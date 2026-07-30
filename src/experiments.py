"""
Reproducible purged walk-forward experiment framework.

Why this exists
---------------
The test period of this project has been inspected many times across many
iterations. Whatever guarantee an untouched hold-out once provided is gone: every
look leaks a little information into the next design decision, and after enough
looks "test performance" measures the analyst as much as the model. It is
therefore treated here as a **development holdout** — reported, but never used to
choose anything.

Every decision that used to be made on that period is instead made on purged
walk-forward out-of-fold predictions: feature selection, hyperparameters, the
loss function, return calibration, model blending, uncertainty calibration and
the decision thresholds.

What a run records
------------------
Enough to reproduce it and to judge it:

* exact fold boundaries, so "fold 2" is unambiguous;
* the full candidate configuration and every random seed;
* per-fold metrics, plus mean / std / min / max across folds;
* identical baselines for every candidate, evaluated on identical rows;
* machine-readable JSON and CSV, and a Markdown summary for the report.

Selection rule
--------------
Candidates are ranked by ``mean - std`` across folds rather than by mean alone.
A configuration that wins on average while swinging wildly between folds has not
demonstrated an edge; it has demonstrated sensitivity to the period. Requiring
consistency is what makes a walk-forward result worth more than a single split.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.regression import summarise_folds
from src.training_common import json_safe
from src.validation import TemporalSplit, purged_walk_forward_splits


@dataclass
class ExperimentCandidate:
    """One configuration to be evaluated on every fold."""

    name: str
    overrides: dict = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FoldResult:
    """One candidate evaluated on one fold."""

    candidate: str
    fold: int
    label: str
    boundaries: dict
    metrics: dict
    seed: int
    fit_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def describe_environment() -> dict:
    """Record enough of the environment to explain a numerical difference later."""
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("numpy", "pandas", "torch", "xgboost", "sklearn", "scipy"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = "not installed"
    return versions


class WalkForwardExperiment:
    """
    Run a set of candidates over identical purged walk-forward folds.

    The fold set is built once and shared by every candidate. That is what makes
    the comparison fair: candidates differ only in their configuration, never in
    which rows they were trained or scored on.
    """

    def __init__(
        self,
        name: str,
        full_df: pd.DataFrame,
        horizon: int,
        folds: int = 3,
        initial_train_ratio: float = 0.50,
        validation_ratio: float = 0.15,
        purge_horizon: int | None = None,
        expanding: bool = True,
        base_seed: int = 42,
        output_dir: Path | None = None,
        restrict_to_dates_before: pd.Timestamp | None = None,
    ) -> None:
        self.name = str(name)
        self.horizon = int(horizon)
        self.base_seed = int(base_seed)
        self.output_dir = Path(output_dir) if output_dir else None

        # Folds are cut only from history the experiment is allowed to see. When a
        # cut-off is supplied the development holdout is excluded outright, so no
        # selection decision can be influenced by it even accidentally.
        self.full_df = (
            full_df[full_df.index <= pd.Timestamp(restrict_to_dates_before)]
            if restrict_to_dates_before is not None
            else full_df
        )
        self.restricted_to = (
            str(pd.Timestamp(restrict_to_dates_before).date())
            if restrict_to_dates_before is not None
            else None
        )

        self.splits: list[TemporalSplit] = purged_walk_forward_splits(
            self.full_df.index.unique(),
            folds=int(folds),
            initial_train_ratio=float(initial_train_ratio),
            validation_ratio=float(validation_ratio),
            purge_horizon=self.horizon if purge_horizon is None else int(purge_horizon),
            expanding=bool(expanding),
        )
        self.results: list[FoldResult] = []
        self.started = time.time()

    def fold_boundaries(self) -> list[dict]:
        return [split.describe() for split in self.splits]

    def run(
        self,
        candidates: Sequence[ExperimentCandidate],
        evaluate: Callable[[ExperimentCandidate, TemporalSplit, pd.DataFrame, int], dict],
        verbose: bool = True,
    ) -> "WalkForwardExperiment":
        """
        Fit and score every (candidate, fold) pair.

        ``evaluate`` receives the candidate, the split, the panel and a seed, and
        returns a metrics dictionary. It is expected to refit from scratch: a fold
        that reuses a model fitted on later data is not a walk-forward fold.

        A candidate that raises on one fold is recorded as failed and the run
        continues. One bad configuration should not destroy an experiment that
        takes hours.
        """
        for candidate in candidates:
            for split in self.splits:
                seed = self.base_seed + 1000 * (1 + self.splits.index(split))
                started = time.time()
                if verbose:
                    bounds = split.describe()
                    print(
                        f"  [{self.name}] {candidate.name} / {split.label}: "
                        f"test {bounds['test']['start']} .. {bounds['test']['end']}"
                    )
                try:
                    metrics = evaluate(candidate, split, self.full_df, seed)
                    extra = metrics.pop("_extra", {}) if isinstance(metrics, dict) else {}
                except Exception as exc:  # noqa: BLE001 - one candidate must not kill the run
                    print(f"    FAILED: {type(exc).__name__}: {exc}")
                    metrics, extra = {}, {"error": f"{type(exc).__name__}: {exc}"}

                self.results.append(
                    FoldResult(
                        candidate=candidate.name,
                        fold=split.index,
                        label=split.label,
                        boundaries=split.describe(),
                        metrics=metrics,
                        seed=seed,
                        fit_seconds=float(time.time() - started),
                        extra=extra,
                    )
                )
                if verbose and metrics:
                    print(
                        f"    IC {metrics.get('cross_sectional_ic', 0.0):+.4f} | "
                        f"MSE {metrics.get('mse', 0.0):.6f} | "
                        f"MSE skill {metrics.get('mse_skill_vs_historical_mean', 0.0):+.4f} | "
                        f"{time.time() - started:.0f}s"
                    )
        return self

    # -- Aggregation ------------------------------------------------------

    def results_frame(self) -> pd.DataFrame:
        rows = []
        for result in self.results:
            row = {
                "candidate": result.candidate,
                "fold": result.fold,
                "label": result.label,
                "test_start": result.boundaries["test"]["start"],
                "test_end": result.boundaries["test"]["end"],
                "train_dates": result.boundaries["train"]["dates"],
                "seed": result.seed,
                "fit_seconds": round(result.fit_seconds, 2),
                "error": result.extra.get("error", ""),
            }
            row.update({key: value for key, value in result.metrics.items() if np.isscalar(value)})
            rows.append(row)
        return pd.DataFrame(rows)

    def summarise(self, keys: Sequence[str]) -> dict[str, dict]:
        """Mean / std / min / max per candidate, over the folds it completed."""
        summary: dict[str, dict] = {}
        for candidate in dict.fromkeys(result.candidate for result in self.results):
            fold_metrics = [
                result.metrics
                for result in self.results
                if result.candidate == candidate and result.metrics
            ]
            summary[candidate] = {
                "folds_completed": len(fold_metrics),
                "metrics": summarise_folds(fold_metrics, keys),
            }
        return summary

    def rank(
        self,
        objective: str = "cross_sectional_ic",
        magnitude_key: str | None = "mse_skill_vs_historical_mean",
        magnitude_weight: float = 0.25,
        stability_penalty: bool = True,
    ) -> list[dict]:
        """
        Rank candidates by out-of-fold consistency.

        The score mirrors the checkpoint criterion used inside training, so the
        experiment selects on the same definition of "better" that the model was
        optimised against — ranking skill plus a weighted magnitude term.
        """
        ranking: list[dict] = []
        for candidate in dict.fromkeys(result.candidate for result in self.results):
            fold_metrics = [
                result.metrics
                for result in self.results
                if result.candidate == candidate and result.metrics
            ]
            if not fold_metrics:
                ranking.append(
                    {"candidate": candidate, "folds": 0, "selection_score": -np.inf, "failed": True}
                )
                continue

            scores = []
            for metrics in fold_metrics:
                score = float(metrics.get(objective, 0.0))
                if magnitude_key:
                    magnitude = float(metrics.get(magnitude_key, 0.0))
                    if np.isfinite(magnitude):
                        score += float(magnitude_weight) * float(np.clip(magnitude, -1.0, 1.0))
                scores.append(score)

            values = np.asarray(scores, dtype=float)
            dispersion = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ranking.append(
                {
                    "candidate": candidate,
                    "folds": len(fold_metrics),
                    "mean_score": float(values.mean()),
                    "score_std": dispersion,
                    "selection_score": float(
                        values.mean() - (dispersion if stability_penalty else 0.0)
                    ),
                    "worst_fold_score": float(values.min()),
                    "per_fold_score": [float(value) for value in scores],
                    "mean_objective": float(
                        np.mean([float(m.get(objective, 0.0)) for m in fold_metrics])
                    ),
                    "objective_positive_in_every_fold": bool(
                        all(float(m.get(objective, 0.0)) > 0 for m in fold_metrics)
                    ),
                    "failed": False,
                }
            )
        return sorted(ranking, key=lambda row: row["selection_score"], reverse=True)

    def best_candidate(self, **kwargs) -> str | None:
        ranking = self.rank(**kwargs)
        if not ranking or ranking[0].get("failed"):
            return None
        return str(ranking[0]["candidate"])

    # -- Persistence ------------------------------------------------------

    def payload(
        self,
        candidates: Sequence[ExperimentCandidate],
        summary_keys: Sequence[str],
        configuration: dict | None = None,
    ) -> dict:
        return {
            "experiment": self.name,
            "horizon": self.horizon,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.time() - self.started, 1),
            "environment": describe_environment(),
            "base_seed": self.base_seed,
            "development_holdout_excluded_after": self.restricted_to,
            "holdout_policy": (
                "the final test period has been inspected repeatedly and is treated "
                "as a development holdout; no selection decision uses it"
            ),
            "selection_rule": "mean(out-of-fold score) - std(out-of-fold score)",
            "fold_boundaries": self.fold_boundaries(),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "configuration": configuration or {},
            "per_fold": [result.to_dict() for result in self.results],
            "summary": self.summarise(summary_keys),
            "ranking": self.rank(),
        }

    def save(
        self,
        candidates: Sequence[ExperimentCandidate],
        summary_keys: Sequence[str],
        configuration: dict | None = None,
    ) -> dict[str, Path]:
        """Write JSON, CSV and a Markdown summary. Returns the paths written."""
        if self.output_dir is None:
            raise ValueError("WalkForwardExperiment needs an output_dir to save results.")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        payload = self.payload(candidates, summary_keys, configuration)
        json_path = self.output_dir / f"experiment_{self.name}_h{self.horizon}.json"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(json_safe(payload), file, indent=2)

        csv_path = self.output_dir / f"experiment_{self.name}_h{self.horizon}.csv"
        self.results_frame().to_csv(csv_path, index=False)

        markdown_path = self.output_dir / f"experiment_{self.name}_h{self.horizon}.md"
        markdown_path.write_text(self.markdown(payload), encoding="utf-8")

        return {"json": json_path, "csv": csv_path, "markdown": markdown_path}

    def markdown(self, payload: dict) -> str:
        """Human-readable summary for the written report."""
        lines = [
            f"# Experiment: {payload['experiment']} (horizon {payload['horizon']})",
            "",
            f"Created: {payload['created']}  ",
            f"Elapsed: {payload['elapsed_seconds']}s  ",
            f"Base seed: {payload['base_seed']}  ",
            f"Selection rule: `{payload['selection_rule']}`  ",
            "",
            f"> {payload['holdout_policy']}",
            "",
            "## Fold boundaries",
            "",
            "| fold | train | validation | test | test dates |",
            "| --- | --- | --- | --- | --- |",
        ]
        for bounds in payload["fold_boundaries"]:
            lines.append(
                f"| {bounds['fold']} "
                f"| {bounds['train']['start']} .. {bounds['train']['end']} "
                f"| {bounds['validation']['start']} .. {bounds['validation']['end']} "
                f"| {bounds['test']['start']} .. {bounds['test']['end']} "
                f"| {bounds['test']['dates']} |"
            )

        lines += [
            "",
            "## Ranking",
            "",
            "| rank | candidate | selection score | mean | std | worst fold | mean IC | IC>0 all folds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for position, row in enumerate(payload["ranking"], start=1):
            if row.get("failed"):
                lines.append(f"| {position} | {row['candidate']} | FAILED | | | | | |")
                continue
            lines.append(
                f"| {position} | {row['candidate']} | {row['selection_score']:+.4f} "
                f"| {row['mean_score']:+.4f} | {row['score_std']:.4f} "
                f"| {row['worst_fold_score']:+.4f} | {row['mean_objective']:+.4f} "
                f"| {'yes' if row['objective_positive_in_every_fold'] else 'no'} |"
            )

        lines += ["", "## Per-fold detail", ""]
        for candidate, block in payload["summary"].items():
            lines.append(f"### {candidate} ({block['folds_completed']} folds)")
            lines.append("")
            lines.append("| metric | mean | std | min | max |")
            lines.append("| --- | --- | --- | --- | --- |")
            for metric, stats in block["metrics"].items():
                lines.append(
                    f"| {metric} | {stats['mean']:+.6f} | {stats['std']:.6f} "
                    f"| {stats['min']:+.6f} | {stats['max']:+.6f} |"
                )
            lines.append("")

        environment = payload["environment"]
        lines += [
            "## Environment",
            "",
            "| component | version |",
            "| --- | --- |",
        ]
        for key, value in environment.items():
            lines.append(f"| {key} | {value} |")

        return "\n".join(lines) + "\n"
