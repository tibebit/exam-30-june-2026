#!/usr/bin/env python3
"""Runner for the agreed Briscola RL training protocol v1.

It codifies the experimental grid so runs are repeatable without committing
generated models or logs.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAIN_LEARNING_RATES = ("0.003", "0.01", "0.03", "0.1")
STRESS_LEARNING_RATES = ("0.1", "0.3", "0.9")
CLIPPING_PROBE_NORMS: tuple[str | None, ...] = (None, "1.0", "5.0")
BOOTSTRAP_UPDATES = (0, 30)
MATCHUP_SAMPLINGS = ("per_episode", "per_rotation_block")
FEATURE_SETS = ("base", "new_interactions")
POLICY_TYPES = ("linear", "neural")
DEFAULT_POLICY_TYPES = ("linear",)
DEFAULT_HIDDEN_SIZES = (64,)

REWARD_PRESETS = {
    "win_heavy": ("1.0", "0.1"),
    "current_baseline": ("1.0", "0.2"),
    "balanced": ("0.5", "0.5"),
    "margin_heavy": ("0.2", "1.0"),
}


@dataclass(frozen=True)
class RunConfig:
    phase: str
    policy_type: str
    seed: int
    batch_size: int
    updates: int
    evaluation_games: int | None
    learning_rate: str
    bootstrap_updates: int
    matchup_sampling: str
    reward_mode: str
    reward_alpha: str
    reward_lambda_margin: str
    feature_set: str = "base"
    baseline: str = "time_dependent"
    snapshot_interval: int = 5
    max_pool_size: int = 20
    init_scale: str = "0.01"
    max_update_norm: str | None = None
    hidden_size: int | None = None
    neural_learned_baseline: bool = True


def value_token(value: str | int) -> str:
    """Make path-safe tokens while keeping parameter names readable."""

    return str(value).replace("-", "minus_").replace(".", "_")


def run_directory(config: RunConfig) -> Path:
    """Build the canonical local folder for one run."""

    clipping = (
        "no_clipping"
        if config.max_update_norm is None
        else f"max_update_norm_{value_token(config.max_update_norm)}"
    )
    root = PROJECT_ROOT / "models"
    if config.feature_set != "base":
        root = root / f"feature_set_{config.feature_set}"

    directory = (
        root
        / f"learning_rate_{value_token(config.learning_rate)}"
    )
    if config.policy_type == "neural":
        if config.hidden_size is None:
            raise ValueError("Neural runs require hidden_size")
        baseline_token = (
            "learned_value_baseline"
            if config.neural_learned_baseline
            else "simple_reinforce_baseline"
        )
        directory = directory / (
            f"policy_neural_hidden_size_{config.hidden_size}_{baseline_token}"
        )
    return (
        directory
        / config.reward_mode
        / (
            f"reward_alpha_{value_token(config.reward_alpha)}"
            f"_lambda_margin_{value_token(config.reward_lambda_margin)}"
        )
        / f"batch_size_{config.batch_size}_updates_{config.updates}"
        / (
            f"baseline_{config.baseline}"
            f"_snapshot_interval_{config.snapshot_interval}"
            f"_pool_size_{config.max_pool_size}_{clipping}"
        )
        / f"matchup_sampling_{config.matchup_sampling}"
        / f"bootstrap_updates_{config.bootstrap_updates}"
        / f"seed_{config.seed}"
    )


def train_command(config: RunConfig, python_bin: str) -> list[str]:
    """Return the exact train command for one config."""

    directory = run_directory(config)
    command = [
        python_bin,
        "-B",
        "scripts/train.py",
        "--seed",
        str(config.seed),
        "--batch-size",
        str(config.batch_size),
        "--updates",
        str(config.updates),
        "--snapshot-interval",
        str(config.snapshot_interval),
        "--max-pool-size",
        str(config.max_pool_size),
        "--init-scale",
        config.init_scale,
        "--learning-rate",
        config.learning_rate,
        "--feature-set",
        config.feature_set,
        "--baseline",
        config.baseline,
        "--reward-mode",
        config.reward_mode,
        "--reward-alpha",
        config.reward_alpha,
        "--reward-lambda-margin",
        config.reward_lambda_margin,
        "--bootstrap-updates",
        str(config.bootstrap_updates),
        "--keep-initial-pool",
        "--matchup-sampling",
        config.matchup_sampling,
        "--output",
        str(directory / "checkpoint.json"),
        "--log",
        str(directory / "train_log.jsonl"),
    ]
    if config.policy_type != "linear":
        command.extend(["--policy-type", config.policy_type])
    if config.policy_type == "neural":
        if config.hidden_size is None:
            raise ValueError("Neural runs require hidden_size")
        command.extend(["--hidden-size", str(config.hidden_size)])
        if config.neural_learned_baseline:
            command.append("--neural-learned-baseline")
        else:
            command.append("--no-neural-learned-baseline")
    if config.max_update_norm is not None:
        command.extend(["--max-update-norm", config.max_update_norm])
    return command


def evaluate_command(config: RunConfig, python_bin: str) -> list[str] | None:
    """Return the exact greedy evaluation command for one config."""

    if config.evaluation_games is None:
        return None
    directory = run_directory(config)
    return [
        python_bin,
        "-B",
        "scripts/evaluate.py",
        "--checkpoint",
        str(directory / "checkpoint.json"),
        "--games",
        str(config.evaluation_games),
        "--output",
        str(directory / f"evaluation_report_games_{config.evaluation_games}.json"),
    ]


def cartesian(
    *,
    phase: str,
    seeds: Iterable[int],
    policy_types: Iterable[str],
    hidden_sizes: Iterable[int],
    batch_size: int,
    updates: int,
    evaluation_games: int | None,
    learning_rates: Iterable[str],
    bootstrap_updates: Iterable[int],
    matchup_samplings: Iterable[str],
    reward_mode: str,
    reward_presets: Iterable[str],
    feature_sets: Iterable[str] = ("base",),
    max_update_norms: Iterable[str | None] = (None,),
) -> list[RunConfig]:
    """Build a deterministic list of run configs."""

    configs: list[RunConfig] = []
    for seed in seeds:
        for policy_type in policy_types:
            selected_hidden_sizes: Iterable[int | None]
            if policy_type == "neural":
                selected_hidden_sizes = hidden_sizes
            else:
                selected_hidden_sizes = (None,)
            for hidden_size in selected_hidden_sizes:
                for learning_rate in learning_rates:
                    for feature_set in feature_sets:
                        for bootstrap in bootstrap_updates:
                            for matchup in matchup_samplings:
                                for preset in reward_presets:
                                    alpha, lambda_margin = REWARD_PRESETS[preset]
                                    for max_update_norm in max_update_norms:
                                        configs.append(
                                            RunConfig(
                                                phase=phase,
                                                policy_type=policy_type,
                                                seed=seed,
                                                batch_size=batch_size,
                                                updates=updates,
                                                evaluation_games=evaluation_games,
                                                learning_rate=learning_rate,
                                                feature_set=feature_set,
                                                bootstrap_updates=bootstrap,
                                                matchup_sampling=matchup,
                                                reward_mode=reward_mode,
                                                reward_alpha=alpha,
                                                reward_lambda_margin=lambda_margin,
                                                max_update_norm=max_update_norm,
                                                hidden_size=hidden_size,
                                            )
                                        )
    return configs


def parse_max_update_norm(value: str) -> str | None:
    """Parse clipping values while allowing an explicit no-clipping token."""

    if value.lower() in {"none", "null", "no_clipping"}:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid --max-update-norm: {value}") from exc
    if parsed < 0.0:
        raise SystemExit("--max-update-norm must be non-negative or none")
    return value


def selected_max_update_norms(
    values: list[str] | None,
    default: tuple[str | None, ...],
) -> tuple[str | None, ...]:
    """Select clipping thresholds for phases that compare optimizer stability."""

    if not values:
        return default
    return tuple(parse_max_update_norm(value) for value in values)


def selected_values(
    *,
    phase: str,
    values: list[str] | None,
    default: tuple[str, ...],
    name: str,
    require_explicit: bool,
) -> tuple[str, ...]:
    """Use defaults for fixed phases and force explicit selections later."""

    if values:
        return tuple(values)
    if require_explicit:
        raise SystemExit(
            f"{phase} requires --{name.replace('_', '-')} to avoid accidental large grids"
        )
    return default


def selected_int_values(
    *,
    phase: str,
    values: list[int] | None,
    default: tuple[int, ...],
    name: str,
    require_explicit: bool,
) -> tuple[int, ...]:
    """Integer variant of selected_values."""

    if values:
        return tuple(values)
    if require_explicit:
        raise SystemExit(
            f"{phase} requires --{name.replace('_', '-')} to avoid accidental large grids"
        )
    return default


def selected_policy_types(args: argparse.Namespace) -> tuple[str, ...]:
    """Preserve the linear default while allowing neural runs on the same grid."""

    if args.policy_type:
        return tuple(args.policy_type)
    if args.all:
        return POLICY_TYPES
    return DEFAULT_POLICY_TYPES


def selected_hidden_sizes(args: argparse.Namespace) -> tuple[int, ...]:
    """Select neural hidden sizes; linear runs ignore this axis."""

    return tuple(args.hidden_size or DEFAULT_HIDDEN_SIZES)


def build_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Build run configs for the requested protocol phase."""

    seeds = tuple(args.seed or [5000])
    feature_sets = tuple(args.feature_set or ("base",))
    policy_types = selected_policy_types(args)
    hidden_sizes = selected_hidden_sizes(args)

    if args.phase == "stress_lr":
        return cartesian(
            phase=args.phase,
            seeds=seeds,
            policy_types=policy_types,
            hidden_sizes=hidden_sizes,
            batch_size=200,
            updates=50,
            evaluation_games=(
                args.evaluation_games
                if args.evaluate_stress and not args.skip_evaluation
                else None
            ),
            learning_rates=tuple(args.learning_rate or STRESS_LEARNING_RATES),
            bootstrap_updates=(0,),
            matchup_samplings=("per_episode",),
            reward_mode="combined_terminal",
            reward_presets=("current_baseline",),
            feature_sets=feature_sets,
            max_update_norms=selected_max_update_norms(
                args.max_update_norm,
                (None,),
            ),
        )

    if args.phase == "clipping_probe":
        return cartesian(
            phase=args.phase,
            seeds=seeds,
            policy_types=policy_types,
            hidden_sizes=hidden_sizes,
            batch_size=180,
            updates=300,
            evaluation_games=None if args.skip_evaluation else 500,
            learning_rates=tuple(args.learning_rate or STRESS_LEARNING_RATES),
            bootstrap_updates=tuple(args.bootstrap_updates or (0,)),
            matchup_samplings=tuple(args.matchup_sampling or ("per_episode",)),
            reward_mode="combined_terminal",
            reward_presets=("current_baseline",),
            feature_sets=feature_sets,
            max_update_norms=selected_max_update_norms(
                args.max_update_norm,
                CLIPPING_PROBE_NORMS,
            ),
        )

    if args.phase == "dense_presa_probe":
        return cartesian(
            phase=args.phase,
            seeds=seeds,
            policy_types=policy_types,
            hidden_sizes=hidden_sizes,
            batch_size=180,
            updates=300,
            evaluation_games=None if args.skip_evaluation else 500,
            learning_rates=tuple(args.learning_rate or ("0.9",)),
            bootstrap_updates=tuple(args.bootstrap_updates or (0,)),
            matchup_samplings=tuple(args.matchup_sampling or ("per_episode",)),
            reward_mode="dense_presa",
            reward_presets=tuple(
                args.reward_preset or ("current_baseline", "balanced")
            ),
            feature_sets=feature_sets,
            max_update_norms=selected_max_update_norms(
                args.max_update_norm,
                (None,),
            ),
        )

    if args.phase == "pilot_combined":
        return cartesian(
            phase=args.phase,
            seeds=seeds,
            policy_types=policy_types,
            hidden_sizes=hidden_sizes,
            batch_size=180,
            updates=300,
            evaluation_games=None if args.skip_evaluation else 100,
            learning_rates=tuple(args.learning_rate or MAIN_LEARNING_RATES),
            bootstrap_updates=tuple(args.bootstrap_updates or BOOTSTRAP_UPDATES),
            matchup_samplings=tuple(args.matchup_sampling or MATCHUP_SAMPLINGS),
            reward_mode="combined_terminal",
            reward_presets=("current_baseline",),
            feature_sets=feature_sets,
            max_update_norms=selected_max_update_norms(
                args.max_update_norm,
                (None,),
            ),
        )

    require_selection = not args.all
    learning_rates = selected_values(
        phase=args.phase,
        values=args.learning_rate,
        default=MAIN_LEARNING_RATES,
        name="learning_rate",
        require_explicit=require_selection,
    )
    bootstrap_updates = selected_int_values(
        phase=args.phase,
        values=args.bootstrap_updates,
        default=BOOTSTRAP_UPDATES,
        name="bootstrap_updates",
        require_explicit=require_selection,
    )
    matchup_samplings = selected_values(
        phase=args.phase,
        values=args.matchup_sampling,
        default=MATCHUP_SAMPLINGS,
        name="matchup_sampling",
        require_explicit=require_selection,
    )

    if args.phase == "series_combined":
        return cartesian(
            phase=args.phase,
            seeds=seeds,
            policy_types=policy_types,
            hidden_sizes=hidden_sizes,
            batch_size=300,
            updates=500,
            evaluation_games=None if args.skip_evaluation else 1000,
            learning_rates=learning_rates,
            bootstrap_updates=bootstrap_updates,
            matchup_samplings=matchup_samplings,
            reward_mode="combined_terminal",
            reward_presets=("current_baseline",),
            feature_sets=feature_sets,
            max_update_norms=selected_max_update_norms(
                args.max_update_norm,
                (None,),
            ),
        )

    reward_presets = tuple(args.reward_preset or REWARD_PRESETS)

    if args.phase == "reward_combined":
        return cartesian(
            phase=args.phase,
            seeds=seeds,
            policy_types=policy_types,
            hidden_sizes=hidden_sizes,
            batch_size=300,
            updates=500,
            evaluation_games=None if args.skip_evaluation else 1000,
            learning_rates=learning_rates,
            bootstrap_updates=bootstrap_updates,
            matchup_samplings=matchup_samplings,
            reward_mode="combined_terminal",
            reward_presets=reward_presets,
            feature_sets=feature_sets,
            max_update_norms=selected_max_update_norms(
                args.max_update_norm,
                (None,),
            ),
        )

    if args.phase == "dense_presa":
        dense_presets = tuple(args.reward_preset or ("current_baseline", "balanced"))
        return cartesian(
            phase=args.phase,
            seeds=seeds,
            policy_types=policy_types,
            hidden_sizes=hidden_sizes,
            batch_size=300,
            updates=500,
            evaluation_games=None if args.skip_evaluation else 1000,
            learning_rates=learning_rates,
            bootstrap_updates=bootstrap_updates,
            matchup_samplings=matchup_samplings,
            reward_mode="dense_presa",
            reward_presets=dense_presets,
            feature_sets=feature_sets,
            max_update_norms=selected_max_update_norms(
                args.max_update_norm,
                (None,),
            ),
        )

    raise AssertionError(f"Unsupported phase: {args.phase}")


def maybe_run(
    *,
    command: list[str],
    execute: bool,
    force: bool,
    output_path: Path,
) -> None:
    """Print or execute one command, skipping completed outputs by default."""

    print(shlex.join(command))
    if not execute:
        return
    if output_path.exists() and not force:
        print(f"SKIP existing output: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local Briscola RL experimental protocol.",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "stress_lr",
            "clipping_probe",
            "dense_presa_probe",
            "pilot_combined",
            "series_combined",
            "reward_combined",
            "dense_presa",
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--evaluate-stress", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument(
        "--policy-type",
        action="append",
        choices=POLICY_TYPES,
        help="Policy family to run. Repeat to compare linear and neural.",
    )
    parser.add_argument(
        "--hidden-size",
        action="append",
        type=int,
        help="Neural hidden size. Ignored by linear runs.",
    )
    parser.add_argument("--learning-rate", action="append")
    parser.add_argument("--feature-set", action="append", choices=FEATURE_SETS)
    parser.add_argument("--bootstrap-updates", action="append", type=int)
    parser.add_argument(
        "--max-update-norm",
        action="append",
        help='Clipping threshold. Use "none" for the no-clipping condition.',
    )
    parser.add_argument(
        "--matchup-sampling",
        action="append",
        choices=MATCHUP_SAMPLINGS,
    )
    parser.add_argument(
        "--reward-preset",
        action="append",
        choices=tuple(REWARD_PRESETS),
    )
    parser.add_argument(
        "--evaluation-games",
        type=int,
        default=100,
        help="Used only by stress_lr unless the phase has a fixed protocol value.",
    )
    args = parser.parse_args()
    if args.hidden_size and any(value <= 0 for value in args.hidden_size):
        raise SystemExit("--hidden-size must be positive")

    configs = build_configs(args)
    print(f"phase={args.phase} runs={len(configs)} execute={args.execute}")
    for index, config in enumerate(configs, start=1):
        directory = run_directory(config)
        print(
            f"\n# run {index}/{len(configs)} "
            f"phase={config.phase} policy_type={config.policy_type}"
        )
        print(f"# output_dir={directory}")
        maybe_run(
            command=train_command(config, args.python_bin),
            execute=args.execute,
            force=args.force,
            output_path=directory / "checkpoint.json",
        )
        eval_cmd = evaluate_command(config, args.python_bin)
        if eval_cmd is not None:
            maybe_run(
                command=eval_cmd,
                execute=args.execute,
                force=args.force,
                output_path=directory
                / f"evaluation_report_games_{config.evaluation_games}.json",
            )


if __name__ == "__main__":
    main()
