#!/usr/bin/env python3
"""CLI entrypoint to evaluate a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `python scripts/evaluate.py` without installing the project.
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import (
    EvaluationMetrics,
    ScenarioEvaluationResult,
    default_evaluation_suite,
    evaluate_suite,
    make_evaluation_cases,
)
from policy import BriscolaFeatureExtractor, LinearSoftmaxPolicy


def parse_args() -> argparse.Namespace:
    """Read evaluation-suite parameters from the CLI."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained Briscola checkpoint against fixed scenarios.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-ambiente-start", type=int, default=100_000)
    parser.add_argument("--seed-policy-start", type=int, default=200_000)
    parser.add_argument("--learner-giocatore-id", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games deve essere positivo")
    return args


def main() -> None:
    args = parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    learner = learner_from_checkpoint(checkpoint)
    suite = default_evaluation_suite()
    cases = make_evaluation_cases(
        games=args.games,
        seed_ambiente_start=args.seed_ambiente_start,
        seed_policy_start=args.seed_policy_start,
    )

    results = evaluate_suite(
        learner_policy=learner,
        suite=suite,
        cases=cases,
        learner_giocatore_id=args.learner_giocatore_id,
        greedy=not args.stochastic,
    )

    output_path = args.output or default_output_path(args.checkpoint, args.games)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            report_to_dict(
                checkpoint_path=args.checkpoint,
                checkpoint=checkpoint,
                games=args.games,
                seed_ambiente_start=args.seed_ambiente_start,
                seed_policy_start=args.seed_policy_start,
                greedy=not args.stochastic,
                results=results,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print_results(results)
    print(f"saved_report={output_path}")


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load the JSON checkpoint produced by `scripts/train.py`."""

    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def learner_from_checkpoint(checkpoint: dict[str, Any]) -> LinearSoftmaxPolicy:
    """Rebuild the final learner saved in the checkpoint."""

    extractor = BriscolaFeatureExtractor()
    saved_features = checkpoint.get("feature_names")
    if saved_features != list(extractor.feature_names):
        raise ValueError("Le feature del checkpoint non corrispondono all'estrattore")

    learner = checkpoint.get("learner", {})
    return LinearSoftmaxPolicy(
        theta=learner["theta"],
        feature_extractor=extractor,
        name=learner.get("name", "learner"),
    )


def default_output_path(checkpoint_path: Path, games: int) -> Path:
    """Save the report next to the evaluated checkpoint."""

    return checkpoint_path.parent / f"evaluation_report_games_{games}.json"


def report_to_dict(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    games: int,
    seed_ambiente_start: int,
    seed_policy_start: int,
    greedy: bool,
    results: dict[str, ScenarioEvaluationResult],
) -> dict[str, Any]:
    """Convert results and configuration into readable JSON."""

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_update_index": checkpoint["update_index"],
        "checkpoint_training_config": checkpoint["config"],
        "games_per_scenario": games,
        "seed_ambiente_start": seed_ambiente_start,
        "seed_policy_start": seed_policy_start,
        "greedy": greedy,
        "scenarios": [
            {
                "name": scenario_name,
                "metrics": metrics_to_dict(result.metrics),
            }
            for scenario_name, result in results.items()
        ],
    }


def metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    """Serialize aggregate metrics without including individual games."""

    return {
        "games": metrics.games,
        "win_rate": metrics.win_rate,
        "draw_rate": metrics.draw_rate,
        "loss_rate": metrics.loss_rate,
        "mean_point_difference": metrics.mean_point_difference,
        "standard_error": metrics.standard_error,
        "confidence_interval_95": list(metrics.confidence_interval_95),
    }


def print_results(results: dict[str, ScenarioEvaluationResult]) -> None:
    """Print a compact table for quick comparison."""

    scenario_width = max(42, *(len(name) for name in results))
    header = (
        "scenario",
        "games",
        "win",
        "draw",
        "loss",
        "mean_margin",
        "stderr",
        "ci95",
    )
    print(
        f"{header[0]:{scenario_width}} {header[1]:>5} {header[2]:>7} {header[3]:>7} "
        f"{header[4]:>7} {header[5]:>12} {header[6]:>8} {header[7]:>23}"
    )
    for scenario_name, result in results.items():
        metrics = result.metrics
        ci_low, ci_high = metrics.confidence_interval_95
        print(
            f"{scenario_name:{scenario_width}} "
            f"{metrics.games:5d} "
            f"{metrics.win_rate:7.3f} "
            f"{metrics.draw_rate:7.3f} "
            f"{metrics.loss_rate:7.3f} "
            f"{metrics.mean_point_difference:12.2f} "
            f"{metrics.standard_error:8.2f} "
            f"[{ci_low:7.2f}, {ci_high:7.2f}]"
        )


if __name__ == "__main__":
    main()
