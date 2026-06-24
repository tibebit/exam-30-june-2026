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

from policy import BriscolaFeatureExtractor, LinearSoftmaxPolicy, NeuralSoftmaxPolicy
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
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates deve essere positivo")
    if args.bootstrap_updates < 0:
        parser.error("--bootstrap-updates deve essere non negativo")
    if args.hidden_size <= 0:
        parser.error("--hidden-size deve essere positivo")
    if args.entropy_coef < 0.0:
        parser.error("--entropy-coef deve essere non negativo")
    if args.keep_initial_pool and args.drop_initial_pool:
        parser.error("--keep-initial-pool e --drop-initial-pool sono incompatibili")
    return args


def main() -> None:
    args = parse_args()

    # The extractor fixes the order and number of features used by theta.
    extractor = BriscolaFeatureExtractor()
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
    )
    trainer = SelfPlayTrainer(
        learner=learner,
        pool=pool,
        config=self_play_config,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    last_stats: SelfPlayStats | None = None
    with args.log.open("w", encoding="utf-8") as log_file:
        # Each update collects a batch, updates the learner, and writes one JSONL row.
        for _ in range(args.updates):
            last_stats = trainer.train_update()
            record = stats_to_dict(last_stats)
            log_file.write(json.dumps(record) + "\n")
            message = (
                "update={update_index} episodes={episodes} "
                "mean_return={mean_return:.4f} margin={mean_score_margin:.2f} "
                "grad_norm={gradient_norm:.4f}"
            )
            if record.get("mean_entropy") is not None:
                message += " entropy={mean_entropy:.4f}"
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
    )
    args.output.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(f"saved_checkpoint={args.output}")
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
    return record


def initialize_learner(
    args: argparse.Namespace,
    extractor: BriscolaFeatureExtractor,
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


def checkpoint_to_dict(
    *,
    args: argparse.Namespace,
    extractor: BriscolaFeatureExtractor,
    learner: LinearSoftmaxPolicy | NeuralSoftmaxPolicy,
    pool: SnapshotPool,
    trainer: SelfPlayTrainer,
    last_stats: SelfPlayStats | None,
) -> dict[str, Any]:
    """Build the final checkpoint with learner, pool, and configuration."""

    return {
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
            "policy_type": args.policy_type,
            "hidden_size": args.hidden_size,
            "init_scale": args.init_scale,
            "learning_rate": args.learning_rate,
            "baseline": args.baseline,
            "max_update_norm": args.max_update_norm,
            "entropy_coef": args.entropy_coef,
            "reward_mode": args.reward_mode,
            "reward_alpha": args.reward_alpha,
            "reward_lambda_margin": args.reward_lambda_margin,
            "greedy_non_learner": args.greedy_non_learner,
            "matchup_sampling": args.matchup_sampling,
        },
        "last_stats": stats_to_dict(last_stats) if last_stats is not None else None,
    }


if __name__ == "__main__":
    main()
