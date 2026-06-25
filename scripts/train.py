#!/usr/bin/env python3
"""CLI entrypoint to run self-play training and save a checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `python scripts/train.py` without installing the project as a package.
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import (
    EvaluationCase,
    EvaluationMetrics,
    EvaluationSuite,
    ScenarioEvaluationResult,
    default_evaluation_suite,
    evaluate_suite,
    make_evaluation_cases,
)
from policy import (
    FEATURE_SET_NAMES,
    BriscolaFeatureExtractor,
    LinearSoftmaxPolicy,
    NeuralSoftmaxPolicy,
    NewFeatureSetExtractor,
    build_feature_extractor,
)
from training import (
    BASELINE_MODES,
    BootstrapPolicySchedule,
    MATCHUP_SAMPLING_MODES,
    REWARD_MODES,
    ReinforceConfig,
    RewardConfig,
    SelfPlayConfig,
    SelfPlayStats,
    SelfPlayTrainer,
    SnapshotPool,
)


POLICY_TYPES = {"linear", "neural"}


def parse_args() -> argparse.Namespace:
    """Read CLI parameters that configure the training loop."""

    parser = argparse.ArgumentParser(
        description="Train a softmax Briscola policy with self-play.",
    )
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--snapshot-interval", type=int, default=5)
    parser.add_argument("--max-pool-size", type=int, default=20)
    parser.add_argument("--bootstrap-updates", type=int, default=0)
    parser.add_argument("--keep-initial-pool", action="store_true")
    parser.add_argument(
        "--drop-initial-pool",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learner-giocatore-id", type=int, default=0)
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SET_NAMES),
        default="base",
    )
    parser.add_argument(
        "--policy-type",
        choices=sorted(POLICY_TYPES),
        default="linear",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINE_MODES),
        default="time_dependent",
    )
    parser.add_argument("--max-update-norm", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    neural_baseline_group = parser.add_mutually_exclusive_group()
    neural_baseline_group.add_argument(
        "--neural-learned-baseline",
        dest="neural_learned_baseline",
        action="store_true",
        default=None,
    )
    neural_baseline_group.add_argument(
        "--no-neural-learned-baseline",
        dest="neural_learned_baseline",
        action="store_false",
    )
    parser.add_argument(
        "--reward-mode",
        choices=sorted(REWARD_MODES),
        default="combined_terminal",
    )
    parser.add_argument("--reward-alpha", type=float, default=1.0)
    parser.add_argument("--reward-lambda-margin", type=float, default=0.2)
    parser.add_argument("--greedy-non-learner", action="store_true")
    parser.add_argument(
        "--matchup-sampling",
        choices=sorted(MATCHUP_SAMPLING_MODES),
        default="per_episode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments/results/checkpoint.json",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=PROJECT_ROOT / "experiments/results/train_log.jsonl",
    )
    parser.add_argument("--best-checkpoint-interval", type=int, default=0)
    parser.add_argument("--best-checkpoint-games", type=int, default=100)
    parser.add_argument(
        "--best-checkpoint-seed-ambiente-start",
        type=int,
        default=300_000,
    )
    parser.add_argument(
        "--best-checkpoint-seed-policy-start",
        type=int,
        default=400_000,
    )
    parser.add_argument("--best-checkpoint-output", type=Path, default=None)
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates deve essere positivo")
    if args.bootstrap_updates < 0:
        parser.error("--bootstrap-updates deve essere non negativo")
    if args.hidden_size <= 0:
        parser.error("--hidden-size deve essere positivo")
    if args.entropy_coef < 0.0:
        parser.error("--entropy-coef deve essere non negativo")
    if args.policy_type == "neural":
        args.neural_learned_baseline = args.neural_learned_baseline is not False
    elif args.neural_learned_baseline is True:
        parser.error("--neural-learned-baseline richiede --policy-type neural")
    else:
        args.neural_learned_baseline = False
    if args.keep_initial_pool and args.drop_initial_pool:
        parser.error("--keep-initial-pool e --drop-initial-pool sono incompatibili")
    if args.best_checkpoint_interval < 0:
        parser.error("--best-checkpoint-interval deve essere non negativo")
    if args.best_checkpoint_games <= 0:
        parser.error("--best-checkpoint-games deve essere positivo")
    return args


def main() -> None:
    args = parse_args()

    # The extractor fixes the order and number of features used by theta.
    extractor = build_feature_extractor(args.feature_set)
    learner = initialize_learner(args, extractor)
    # The pool stores frozen learner copies used as opponents/partner.
    pool = SnapshotPool(
        feature_extractor=extractor,
        max_size=args.max_pool_size,
        keep_initial=args.keep_initial_pool,
    )
    # Reward and REINFORCE remain visible run parameters, not hidden choices.
    reward_config = RewardConfig(
        mode=args.reward_mode,
        alpha=args.reward_alpha,
        lambda_margin=args.reward_lambda_margin,
    )
    reinforce_config = ReinforceConfig(
        learning_rate=args.learning_rate,
        baseline=args.baseline,
        max_update_norm=args.max_update_norm,
        entropy_coef=args.entropy_coef,
    )
    self_play_config = SelfPlayConfig(
        batch_size=args.batch_size,
        snapshot_interval=args.snapshot_interval,
        learner_giocatore_id=args.learner_giocatore_id,
        reward_config=reward_config,
        reinforce_config=reinforce_config,
        bootstrap_schedule=BootstrapPolicySchedule(
            bootstrap_updates=args.bootstrap_updates,
        ),
        greedy_non_learner=args.greedy_non_learner,
        matchup_sampling=args.matchup_sampling,
        neural_learned_baseline=args.neural_learned_baseline,
    )
    trainer = SelfPlayTrainer(
        learner=learner,
        pool=pool,
        config=self_play_config,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    best_output = default_best_checkpoint_path(args)
    best_checkpoint_summary: dict[str, Any] | None = None
    best_checkpoint_score: float | None = None
    if args.best_checkpoint_interval > 0:
        if best_output.resolve() == args.output.resolve():
            raise ValueError(
                "best checkpoint e final checkpoint devono avere path diverse"
            )
        best_output.parent.mkdir(parents=True, exist_ok=True)
        best_suite = default_evaluation_suite()
        best_cases = make_evaluation_cases(
            games=args.best_checkpoint_games,
            seed_ambiente_start=args.best_checkpoint_seed_ambiente_start,
            seed_policy_start=args.best_checkpoint_seed_policy_start,
        )
    else:
        best_suite = None
        best_cases = None

    last_stats: SelfPlayStats | None = None
    with args.log.open("w", encoding="utf-8") as log_file:
        # Each update collects a batch, updates the learner, and writes one JSONL row.
        for _ in range(args.updates):
            last_stats = trainer.train_update()
            record = stats_to_dict(last_stats)
            if should_evaluate_best_checkpoint(
                update_index=last_stats.update_index,
                args=args,
            ):
                if best_suite is None or best_cases is None:
                    raise RuntimeError("Best-checkpoint evaluation non inizializzata")
                evaluation_summary = evaluate_for_best_checkpoint(
                    learner=learner,
                    args=args,
                    suite=best_suite,
                    cases=best_cases,
                )
                evaluation_summary["update_index"] = last_stats.update_index
                score = evaluation_summary["score"]
                improved = (
                    best_checkpoint_score is None
                    or score > best_checkpoint_score
                )
                record["best_checkpoint_score"] = score
                record["best_checkpoint_updated"] = improved
                record["best_checkpoint_path"] = str(best_output)
                record["best_checkpoint_scenario_scores"] = {
                    scenario["name"]: scenario["metrics"]["mean_point_difference"]
                    for scenario in evaluation_summary["scenarios"]
                }
                if improved:
                    best_checkpoint_score = score
                    best_checkpoint_summary = evaluation_summary
                    checkpoint = checkpoint_to_dict(
                        args=args,
                        extractor=extractor,
                        learner=learner,
                        pool=pool,
                        trainer=trainer,
                        last_stats=last_stats,
                        best_checkpoint_summary=best_checkpoint_summary,
                        best_checkpoint_path=best_output,
                    )
                    checkpoint["best_selection"] = best_checkpoint_summary
                    best_output.write_text(
                        json.dumps(checkpoint, indent=2),
                        encoding="utf-8",
                    )

            log_file.write(json.dumps(record) + "\n")
            message = (
                "update={update_index} episodes={episodes} "
                "mean_return={mean_return:.4f} margin={mean_score_margin:.2f} "
                "grad_norm={gradient_norm:.4f}"
            )
            if record.get("mean_entropy") is not None:
                message += " entropy={mean_entropy:.4f}"
            if record.get("mean_value_loss") is not None:
                message += " value_loss={mean_value_loss:.4f}"
            if record.get("best_checkpoint_score") is not None:
                message += " best_score={best_checkpoint_score:.2f}"
                if record.get("best_checkpoint_updated"):
                    message += " best_updated=1"
            message += " pool={pool_size}"
            print(message.format(**record))

    # The checkpoint saves state and configuration; evaluation stays a separate step.
    checkpoint = checkpoint_to_dict(
        args=args,
        extractor=extractor,
        learner=learner,
        pool=pool,
        trainer=trainer,
        last_stats=last_stats,
        best_checkpoint_summary=best_checkpoint_summary,
        best_checkpoint_path=(
            best_output if best_checkpoint_summary is not None else None
        ),
    )
    args.output.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(f"saved_checkpoint={args.output}")
    if best_checkpoint_summary is not None:
        print(f"saved_best_checkpoint={best_output}")
    print(f"saved_log={args.log}")


def stats_to_dict(stats: SelfPlayStats) -> dict[str, Any]:
    """Convert update metrics into a readable JSONL row."""

    train_stats = stats.train_stats
    record = {
        "update_index": stats.update_index,
        "episodes": train_stats.episodes,
        "learner_decisions": train_stats.learner_decisions,
        "mean_return": train_stats.mean_return,
        "mean_score_margin": train_stats.mean_score_margin,
        "gradient_norm": train_stats.gradient_norm,
        "baseline": train_stats.baseline,
        "baseline_values": list(train_stats.baseline_values),
        "pool_size": stats.pool_size,
        "snapshot_added": stats.snapshot_added,
    }
    if train_stats.mean_entropy is not None:
        record["mean_entropy"] = train_stats.mean_entropy
    if train_stats.mean_value_loss is not None:
        record["mean_value_loss"] = train_stats.mean_value_loss
    return record


def initialize_learner(
    args: argparse.Namespace,
    extractor: BriscolaFeatureExtractor | NewFeatureSetExtractor,
) -> LinearSoftmaxPolicy | NeuralSoftmaxPolicy:
    """Create the requested learner without changing the default linear path."""

    if args.policy_type == "linear":
        return LinearSoftmaxPolicy.initialize(
            feature_extractor=extractor,
            rng=random.Random(args.seed),
            scale=args.init_scale,
            name="learner",
        )

    if args.policy_type == "neural":
        return NeuralSoftmaxPolicy.initialize(
            feature_extractor=extractor,
            rng=random.Random(args.seed),
            hidden_size=args.hidden_size,
            scale=args.init_scale,
            name="learner",
        )

    raise ValueError(f"Policy type non supportata: {args.policy_type}")


def learner_to_dict(
    learner: LinearSoftmaxPolicy | NeuralSoftmaxPolicy,
) -> dict[str, Any]:
    """Serialize the learner in a checkpoint-friendly shape."""

    payload: dict[str, Any] = {
        "name": learner.name,
        "theta": learner.theta.tolist(),
    }
    if isinstance(learner, NeuralSoftmaxPolicy):
        payload["policy_type"] = "neural"
        payload["hidden_size"] = learner.hidden_size
    else:
        payload["policy_type"] = "linear"
    return payload


def snapshot_to_dict(snapshot: Any) -> dict[str, Any]:
    """Serialize one frozen pool snapshot."""

    payload = {
        "name": snapshot.name,
        "update_index": snapshot.update_index,
        "theta": snapshot.theta.tolist(),
        "policy_type": snapshot.policy_type,
    }
    if snapshot.hidden_size is not None:
        payload["hidden_size"] = snapshot.hidden_size
    return payload


def default_best_checkpoint_path(args: argparse.Namespace) -> Path:
    """Return the external best-checkpoint path used by periodic evaluation."""

    if args.best_checkpoint_output is not None:
        return args.best_checkpoint_output
    return args.output.parent / "best_checkpoint.json"


def should_evaluate_best_checkpoint(
    *,
    update_index: int,
    args: argparse.Namespace,
) -> bool:
    """Evaluate periodically and always include the final trained learner."""

    if args.best_checkpoint_interval <= 0:
        return False
    return (
        update_index % args.best_checkpoint_interval == 0
        or update_index == args.updates
    )


def evaluate_for_best_checkpoint(
    *,
    learner: LinearSoftmaxPolicy | NeuralSoftmaxPolicy,
    args: argparse.Namespace,
    suite: EvaluationSuite,
    cases: list[EvaluationCase],
) -> dict[str, Any]:
    """Evaluate the current learner and build the best-selection summary."""

    results = evaluate_suite(
        learner_policy=learner,
        suite=suite,
        cases=cases,
        learner_giocatore_id=args.learner_giocatore_id,
        greedy=True,
    )
    score = score_evaluation_results(results)
    return {
        "metric": "mean_scenario_margin",
        "score": score,
        "update_index": None,
        "games_per_scenario": args.best_checkpoint_games,
        "seed_ambiente_start": args.best_checkpoint_seed_ambiente_start,
        "seed_policy_start": args.best_checkpoint_seed_policy_start,
        "greedy": True,
        "scenarios": [
            {
                "name": scenario_name,
                "metrics": metrics_to_dict(result.metrics),
            }
            for scenario_name, result in results.items()
        ],
    }


def score_evaluation_results(
    results: dict[str, ScenarioEvaluationResult],
) -> float:
    """Score one evaluation suite by averaging scenario mean margins."""

    if not results:
        raise ValueError("Serve almeno uno scenario per selezionare il best checkpoint")
    margins = [
        result.metrics.mean_point_difference
        for result in results.values()
    ]
    return float(sum(margins) / len(margins))


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


def checkpoint_to_dict(
    *,
    args: argparse.Namespace,
    extractor: BriscolaFeatureExtractor | NewFeatureSetExtractor,
    learner: LinearSoftmaxPolicy | NeuralSoftmaxPolicy,
    pool: SnapshotPool,
    trainer: SelfPlayTrainer,
    last_stats: SelfPlayStats | None,
    best_checkpoint_summary: dict[str, Any] | None = None,
    best_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Build the final checkpoint with learner, pool, and configuration."""

    checkpoint = {
        "kind": "briscola_rl_4players_training_checkpoint",
        "update_index": trainer.update_index,
        "seed": args.seed,
        "feature_names": list(extractor.feature_names),
        "learner": learner_to_dict(learner),
        "pool": [snapshot_to_dict(snapshot) for snapshot in pool.snapshots],
        "config": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "snapshot_interval": args.snapshot_interval,
            "max_pool_size": args.max_pool_size,
            "bootstrap_updates": args.bootstrap_updates,
            "keep_initial_pool": args.keep_initial_pool,
            "learner_giocatore_id": args.learner_giocatore_id,
            "feature_set": args.feature_set,
            "policy_type": args.policy_type,
            "hidden_size": args.hidden_size,
            "init_scale": args.init_scale,
            "learning_rate": args.learning_rate,
            "baseline": args.baseline,
            "max_update_norm": args.max_update_norm,
            "entropy_coef": args.entropy_coef,
            "neural_learned_baseline": args.neural_learned_baseline,
            "reward_mode": args.reward_mode,
            "reward_alpha": args.reward_alpha,
            "reward_lambda_margin": args.reward_lambda_margin,
            "greedy_non_learner": args.greedy_non_learner,
            "matchup_sampling": args.matchup_sampling,
            "best_checkpoint_interval": args.best_checkpoint_interval,
            "best_checkpoint_games": args.best_checkpoint_games,
            "best_checkpoint_seed_ambiente_start": (
                args.best_checkpoint_seed_ambiente_start
            ),
            "best_checkpoint_seed_policy_start": (
                args.best_checkpoint_seed_policy_start
            ),
            "best_checkpoint_output": (
                str(default_best_checkpoint_path(args))
                if args.best_checkpoint_interval > 0
                else None
            ),
        },
        "last_stats": stats_to_dict(last_stats) if last_stats is not None else None,
    }
    if best_checkpoint_summary is not None:
        checkpoint["best_checkpoint"] = {
            "path": str(best_checkpoint_path),
            "metric": best_checkpoint_summary["metric"],
            "score": best_checkpoint_summary["score"],
            "update_index": best_checkpoint_summary["update_index"],
        }
    return checkpoint


if __name__ == "__main__":
    main()
