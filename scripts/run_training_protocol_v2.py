#!/usr/bin/env python3
"""CLI orchestrator for the agreed Briscola RL training protocol v2.

Named phases define controlled experimental grids. Each configuration is
translated into the existing train and evaluation entrypoints, with a stable
local output path. The runner does not implement training or evaluation logic,
and it never commits generated models or logs.
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
if str(PROJECT_ROOT) not in sys.path:
    # Allow `python scripts/run_training_protocol_v2.py` without installation.
    sys.path.insert(0, str(PROJECT_ROOT))

from policy.feature_sets import FEATURE_SET_NAMES

# Historical exploratory axes are retained so earlier protocol phases remain
# reproducible. They are not the defaults of the consolidated v2 phases.
MAIN_LEARNING_RATES = ("0.003", "0.01", "0.03", "0.1")
# Stress values start above the main grid, which already includes 0.1.
STRESS_LEARNING_RATES = ("0.3", "0.5", "0.9")
WARM_START_UPDATES = (0, 30)
MATCHUP_SAMPLINGS = ("per_episode", "per_rotation_block")

# Feature-set names come from the production selector so the runner cannot
# accept a value that scripts/train.py would reject.
FEATURE_SETS = tuple(sorted(FEATURE_SET_NAMES))
DEFAULT_FEATURE_SETS = ("base",)
LINEAR_FEATURE_COMPARISON_SETS = ("base_aligned", "new_aligned")
POLICY_TYPES = ("linear", "neural")
DEFAULT_POLICY_TYPES = ("linear",)
DEFAULT_HIDDEN_SIZES = (64,)

NEURAL_CALIBRATION_LEARNING_RATES = ("0.0003", "0.001", "0.003")
NEURAL_CALIBRATION_HIDDEN_SIZES = (32, 64)
NEURAL_COMMON_FEATURE_SET = "common_atomic"
NEURAL_FEATURE_SETS = ("common_atomic", "base_aligned", "new_aligned")
NEURAL_WARM_START_UPDATES = (30,)
NEURAL_MATCHUP_SAMPLINGS = ("per_rotation_block",)
NEURAL_REWARD_MODE = "combined_terminal"
NEURAL_REWARD_PRESET = "current_baseline"
NEURAL_ENTROPY_COEF = "0.0"

LEGACY_PHASES = (
    "stress_lr",
    "dense_presa_probe",
    "pilot_combined",
    "series_combined",
    "reward_combined",
    "dense_presa",
)

# Current fixed condition for representation ablations. CLI options can
# override one axis deliberately, but the defaults remain explicit here.
CONSOLIDATED_LEARNING_RATES = ("0.9",)
CONSOLIDATED_WARM_START_UPDATES = (30,)
CONSOLIDATED_MATCHUP_SAMPLINGS = ("per_rotation_block",)

# Reward weights are named to keep reward-grid commands readable.
REWARD_PRESETS = {
    "win_heavy": ("1.0", "0.1"),
    "current_baseline": ("1.0", "0.2"),
    "balanced": ("0.5", "0.5"),
    "margin_heavy": ("0.2", "1.0"),
}


@dataclass(frozen=True)
class RunConfig:
    """One fully specified training and evaluation run."""

    phase: str
    policy_type: str
    seed: int
    batch_size: int
    updates: int
    evaluation_games: int | None
    learning_rate: str
    warm_start_updates: int
    matchup_sampling: str
    reward_mode: str
    reward_alpha: str
    reward_lambda_margin: str
    feature_set: str = "base"
    baseline: str = "time_dependent"
    snapshot_interval: int = 5
    max_pool_size: int = 20
    init_scale: str = "0.01"
    entropy_coef: str = "0.0"
    hidden_size: int | None = None
    neural_learned_baseline: bool = True


@dataclass(frozen=True)
class FeatureComparisonPhase:
    """Fixed budget for one controlled linear representation comparison."""

    feature_sets: tuple[str, ...]
    batch_size: int
    updates: int
    evaluation_games: int


@dataclass(frozen=True)
class NeuralProtocolPhase:
    """One ordered phase of the neural training protocol."""

    feature_sets: tuple[str, ...]
    learning_rates: tuple[str, ...]
    hidden_sizes: tuple[int, ...]
    learned_baselines: tuple[bool, ...]
    batch_size: int
    updates: int
    evaluation_games: int
    reward_mode: str = NEURAL_REWARD_MODE
    reward_presets: tuple[str, ...] = (NEURAL_REWARD_PRESET,)
    require_learning_rate: bool = False
    require_hidden_size: bool = False
    require_feature_set: bool = False
    require_learned_baseline: bool = False
    require_reward_preset: bool = False
    require_single_learning_rate: bool = False
    require_single_hidden_size: bool = False
    require_single_feature_set: bool = False
    require_single_reward_preset: bool = False
    compare_learned_baselines: bool = False


# These phases keep the common atomic representation fixed. They compare only
# the historical and new engineered interaction families.
FEATURE_COMPARISON_PHASES = {
    "feature_linear_light": FeatureComparisonPhase(
        feature_sets=LINEAR_FEATURE_COMPARISON_SETS,
        batch_size=180,
        updates=300,
        evaluation_games=500,
    ),
    "feature_linear_intensive": FeatureComparisonPhase(
        feature_sets=LINEAR_FEATURE_COMPARISON_SETS,
        batch_size=300,
        updates=500,
        evaluation_games=1000,
    ),
}


# Neural phases change one family of choices at a time. Adam learning rate and
# hidden size are calibrated together; every later phase requires that choice.
NEURAL_PROTOCOL_PHASES = {
    "neural_calibration_light": NeuralProtocolPhase(
        feature_sets=(NEURAL_COMMON_FEATURE_SET,),
        learning_rates=NEURAL_CALIBRATION_LEARNING_RATES,
        hidden_sizes=NEURAL_CALIBRATION_HIDDEN_SIZES,
        learned_baselines=(True,),
        batch_size=180,
        updates=300,
        evaluation_games=500,
    ),
    "neural_calibration_intensive": NeuralProtocolPhase(
        feature_sets=(NEURAL_COMMON_FEATURE_SET,),
        learning_rates=(),
        hidden_sizes=(),
        learned_baselines=(True,),
        batch_size=300,
        updates=500,
        evaluation_games=1000,
        require_learning_rate=True,
        require_hidden_size=True,
        require_single_learning_rate=True,
        require_single_hidden_size=True,
        compare_learned_baselines=True,
    ),
    "neural_value_baseline_ablation": NeuralProtocolPhase(
        feature_sets=(NEURAL_COMMON_FEATURE_SET,),
        learning_rates=(),
        hidden_sizes=(),
        learned_baselines=(True, False),
        batch_size=300,
        updates=500,
        evaluation_games=1000,
        require_learning_rate=True,
        require_hidden_size=True,
        require_single_learning_rate=True,
        require_single_hidden_size=True,
    ),
    "neural_features_light": NeuralProtocolPhase(
        feature_sets=NEURAL_FEATURE_SETS,
        learning_rates=(),
        hidden_sizes=(),
        learned_baselines=(True, False),
        batch_size=180,
        updates=300,
        evaluation_games=500,
        require_learning_rate=True,
        require_hidden_size=True,
        require_learned_baseline=True,
        require_single_learning_rate=True,
        require_single_hidden_size=True,
    ),
    "neural_features_intensive": NeuralProtocolPhase(
        feature_sets=NEURAL_FEATURE_SETS,
        learning_rates=(),
        hidden_sizes=(),
        learned_baselines=(True, False),
        batch_size=300,
        updates=500,
        evaluation_games=1000,
        require_learning_rate=True,
        require_hidden_size=True,
        require_feature_set=True,
        require_learned_baseline=True,
        require_single_learning_rate=True,
        require_single_hidden_size=True,
        require_single_feature_set=True,
    ),
    "neural_reward_combined_light": NeuralProtocolPhase(
        feature_sets=NEURAL_FEATURE_SETS,
        learning_rates=(),
        hidden_sizes=(),
        learned_baselines=(True, False),
        batch_size=180,
        updates=300,
        evaluation_games=500,
        reward_presets=tuple(REWARD_PRESETS),
        require_learning_rate=True,
        require_hidden_size=True,
        require_feature_set=True,
        require_learned_baseline=True,
        require_single_learning_rate=True,
        require_single_hidden_size=True,
        require_single_feature_set=True,
    ),
    "neural_reward_combined_intensive": NeuralProtocolPhase(
        feature_sets=NEURAL_FEATURE_SETS,
        learning_rates=(),
        hidden_sizes=(),
        learned_baselines=(True, False),
        batch_size=300,
        updates=500,
        evaluation_games=1000,
        reward_presets=tuple(REWARD_PRESETS),
        require_learning_rate=True,
        require_hidden_size=True,
        require_feature_set=True,
        require_learned_baseline=True,
        require_reward_preset=True,
        require_single_learning_rate=True,
        require_single_hidden_size=True,
        require_single_feature_set=True,
        require_single_reward_preset=True,
    ),
    "neural_dense_presa_light": NeuralProtocolPhase(
        feature_sets=NEURAL_FEATURE_SETS,
        learning_rates=(),
        hidden_sizes=(),
        learned_baselines=(True,),
        batch_size=180,
        updates=300,
        evaluation_games=500,
        reward_mode="dense_presa",
        reward_presets=("current_baseline", "balanced"),
        require_learning_rate=True,
        require_hidden_size=True,
        require_feature_set=True,
        require_learned_baseline=True,
        require_single_learning_rate=True,
        require_single_hidden_size=True,
    ),
    "neural_dense_presa_intensive": NeuralProtocolPhase(
        feature_sets=NEURAL_FEATURE_SETS,
        learning_rates=(),
        hidden_sizes=(),
        learned_baselines=(True,),
        batch_size=300,
        updates=500,
        evaluation_games=1000,
        reward_mode="dense_presa",
        reward_presets=("current_baseline", "balanced"),
        require_learning_rate=True,
        require_hidden_size=True,
        require_feature_set=True,
        require_learned_baseline=True,
        require_reward_preset=True,
        require_single_learning_rate=True,
        require_single_hidden_size=True,
        require_single_feature_set=True,
        require_single_reward_preset=True,
    ),
}


def value_token(value: str | int) -> str:
    """Make path-safe tokens while keeping parameter names readable."""

    return str(value).replace("-", "minus_").replace(".", "_")


def run_directory(config: RunConfig) -> Path:
    """Build the canonical local folder for one run."""

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
            f"_pool_size_{config.max_pool_size}"
        )
        / f"matchup_sampling_{config.matchup_sampling}"
        / f"warm_start_updates_{config.warm_start_updates}"
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
        "--warm-start-updates",
        str(config.warm_start_updates),
        # The v2 protocol preserves the initial learner snapshot. It stays
        # fixed rather than becoming another experimental axis in this runner.
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
        command.extend(
            [
                "--hidden-size",
                str(config.hidden_size),
                "--entropy-coef",
                config.entropy_coef,
            ]
        )
        if config.neural_learned_baseline:
            command.append("--neural-learned-baseline")
        else:
            command.append("--no-neural-learned-baseline")
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
    warm_start_updates: Iterable[int],
    matchup_samplings: Iterable[str],
    reward_mode: str,
    reward_presets: Iterable[str],
    feature_sets: Iterable[str] = ("base",),
    neural_learned_baselines: Iterable[bool] = (True,),
    entropy_coef: str = "0.0",
) -> list[RunConfig]:
    """Expand explicitly selected axes in a deterministic, inspectable order."""

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
                        for warm_start in warm_start_updates:
                            for matchup in matchup_samplings:
                                for preset in reward_presets:
                                    alpha, lambda_margin = REWARD_PRESETS[preset]
                                    for neural_learned_baseline in neural_learned_baselines:
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
                                                warm_start_updates=warm_start,
                                                matchup_sampling=matchup,
                                                reward_mode=reward_mode,
                                                reward_alpha=alpha,
                                                reward_lambda_margin=lambda_margin,
                                                entropy_coef=entropy_coef,
                                                hidden_size=hidden_size,
                                                neural_learned_baseline=(
                                                    neural_learned_baseline
                                                ),
                                            )
                                        )
    return configs


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
    """Select a policy family for legacy linear protocol phases."""

    if args.policy_type:
        return tuple(args.policy_type)
    if args.all:
        return POLICY_TYPES
    return DEFAULT_POLICY_TYPES


def selected_hidden_sizes(args: argparse.Namespace) -> tuple[int, ...]:
    """Select neural hidden sizes; linear runs ignore this axis."""

    return tuple(args.hidden_size or DEFAULT_HIDDEN_SIZES)


def phase_values(
    *,
    phase: str,
    values: list[str] | None,
    allowed: tuple[str, ...],
    name: str,
    require_explicit: bool,
    require_single: bool,
) -> tuple[str, ...]:
    """Select string values for one constrained neural phase."""

    if require_explicit and not values:
        raise SystemExit(
            f"{phase} richiede --{name.replace('_', '-')} scelto in una fase precedente",
        )
    selected = tuple(values or allowed)
    if not selected:
        raise SystemExit(f"{phase} richiede almeno un valore --{name.replace('_', '-')}")
    if allowed and not set(selected) <= set(allowed):
        raise SystemExit(
            f"{phase} supporta soltanto {name}={sorted(allowed)}",
        )
    if require_single and len(selected) != 1:
        raise SystemExit(f"{phase} richiede un solo --{name.replace('_', '-')}")
    return selected


def phase_int_values(
    *,
    phase: str,
    values: list[int] | None,
    allowed: tuple[int, ...],
    name: str,
    require_explicit: bool,
    require_single: bool,
) -> tuple[int, ...]:
    """Select integer values for one constrained neural phase."""

    if require_explicit and not values:
        raise SystemExit(
            f"{phase} richiede --{name.replace('_', '-')} scelto in una fase precedente",
        )
    selected = tuple(values or allowed)
    if not selected:
        raise SystemExit(f"{phase} richiede almeno un valore --{name.replace('_', '-')}")
    if allowed and not set(selected) <= set(allowed):
        raise SystemExit(
            f"{phase} supporta soltanto {name}={sorted(allowed)}",
        )
    if require_single and len(selected) != 1:
        raise SystemExit(f"{phase} richiede un solo --{name.replace('_', '-')}")
    return selected


def neural_learned_baselines(
    *,
    args: argparse.Namespace,
    phase: str,
    neural_phase: NeuralProtocolPhase,
) -> tuple[bool, ...]:
    """Select the learned-baseline condition without mixing ablation phases."""

    requested = args.neural_learned_baseline
    if neural_phase.compare_learned_baselines:
        if requested is not None:
            raise SystemExit(
                f"{phase} confronta entrambe le learned value baseline; non usare il flag",
            )
        return neural_phase.learned_baselines
    if neural_phase.require_learned_baseline and requested is None:
        raise SystemExit(
            f"{phase} richiede --neural-learned-baseline o --no-neural-learned-baseline",
        )
    selected = neural_phase.learned_baselines if requested is None else (requested,)
    if not set(selected) <= set(neural_phase.learned_baselines):
        raise SystemExit(
            f"{phase} supporta soltanto learned value baseline="
            f"{list(neural_phase.learned_baselines)}",
        )
    return selected


def build_neural_configs(
    *,
    args: argparse.Namespace,
    phase: str,
    neural_phase: NeuralProtocolPhase,
    seeds: tuple[int, ...],
) -> list[RunConfig]:
    """Build one neural phase while keeping unrelated axes fixed."""

    if args.policy_type and set(args.policy_type) != {"neural"}:
        raise SystemExit(f"{phase} confronta soltanto policy_type=neural")
    if args.all:
        raise SystemExit(f"{phase} non supporta --all")
    if args.warm_start_updates:
        raise SystemExit(f"{phase} mantiene warm_start_updates fisso")
    if args.matchup_sampling:
        raise SystemExit(f"{phase} mantiene matchup_sampling fisso")

    learning_rates = phase_values(
        phase=phase,
        values=args.learning_rate,
        allowed=neural_phase.learning_rates,
        name="learning_rate",
        require_explicit=neural_phase.require_learning_rate,
        require_single=neural_phase.require_single_learning_rate,
    )
    hidden_sizes = phase_int_values(
        phase=phase,
        values=args.hidden_size,
        allowed=neural_phase.hidden_sizes,
        name="hidden_size",
        require_explicit=neural_phase.require_hidden_size,
        require_single=neural_phase.require_single_hidden_size,
    )
    feature_sets = phase_values(
        phase=phase,
        values=args.feature_set,
        allowed=neural_phase.feature_sets,
        name="feature_set",
        require_explicit=neural_phase.require_feature_set,
        require_single=neural_phase.require_single_feature_set,
    )
    reward_presets = phase_values(
        phase=phase,
        values=args.reward_preset,
        allowed=neural_phase.reward_presets,
        name="reward_preset",
        require_explicit=neural_phase.require_reward_preset,
        require_single=neural_phase.require_single_reward_preset,
    )

    return cartesian(
        phase=phase,
        seeds=seeds,
        policy_types=("neural",),
        hidden_sizes=hidden_sizes,
        batch_size=neural_phase.batch_size,
        updates=neural_phase.updates,
        evaluation_games=(
            None if args.skip_evaluation else neural_phase.evaluation_games
        ),
        learning_rates=learning_rates,
        warm_start_updates=NEURAL_WARM_START_UPDATES,
        matchup_samplings=NEURAL_MATCHUP_SAMPLINGS,
        reward_mode=neural_phase.reward_mode,
        reward_presets=reward_presets,
        feature_sets=feature_sets,
        neural_learned_baselines=neural_learned_baselines(
            args=args,
            phase=phase,
            neural_phase=neural_phase,
        ),
        entropy_coef=NEURAL_ENTROPY_COEF,
    )


def build_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Build run configs for the requested protocol phase."""

    seeds = tuple(args.seed or [5000])

    neural_phase = NEURAL_PROTOCOL_PHASES.get(args.phase)
    if neural_phase is not None:
        return build_neural_configs(
            args=args,
            phase=args.phase,
            neural_phase=neural_phase,
            seeds=seeds,
        )

    selected_feature_sets = tuple(args.feature_set or DEFAULT_FEATURE_SETS)
    policy_types = selected_policy_types(args)
    hidden_sizes = selected_hidden_sizes(args)

    if "neural" in policy_types:
        raise SystemExit(
            "Usa una fase neural_*: le fasi storiche e lineari non "
            "condividono il protocollo di Adam.",
        )
    if args.neural_learned_baseline is not None:
        raise SystemExit(
            "--neural-learned-baseline e --no-neural-learned-baseline "
            "sono validi soltanto nelle fasi neural_*.",
        )

    feature_phase = FEATURE_COMPARISON_PHASES.get(args.phase)
    if feature_phase is not None:
        if args.policy_type and set(args.policy_type) != {"linear"}:
            raise SystemExit(
                f"{args.phase} confronta soltanto policy_type=linear",
            )
        requested_feature_sets = tuple(args.feature_set or feature_phase.feature_sets)
        unsupported_feature_sets = set(requested_feature_sets) - set(
            feature_phase.feature_sets,
        )
        if unsupported_feature_sets:
            raise SystemExit(
                f"{args.phase} supporta soltanto feature set "
                f"{sorted(feature_phase.feature_sets)}",
            )
        return cartesian(
            phase=args.phase,
            seeds=seeds,
            policy_types=("linear",),
            hidden_sizes=(),
            batch_size=feature_phase.batch_size,
            updates=feature_phase.updates,
            evaluation_games=(
                None if args.skip_evaluation else feature_phase.evaluation_games
            ),
            learning_rates=tuple(args.learning_rate or CONSOLIDATED_LEARNING_RATES),
            warm_start_updates=tuple(
                args.warm_start_updates or CONSOLIDATED_WARM_START_UPDATES
            ),
            matchup_samplings=tuple(
                args.matchup_sampling or CONSOLIDATED_MATCHUP_SAMPLINGS
            ),
            reward_mode="combined_terminal",
            reward_presets=("current_baseline",),
            feature_sets=requested_feature_sets,
        )

    # Historical phases below remain available to replay earlier experiments.
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
            warm_start_updates=(0,),
            matchup_samplings=("per_episode",),
            reward_mode="combined_terminal",
            reward_presets=("current_baseline",),
            feature_sets=selected_feature_sets,
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
            warm_start_updates=tuple(args.warm_start_updates or (0,)),
            matchup_samplings=tuple(args.matchup_sampling or ("per_episode",)),
            reward_mode="dense_presa",
            reward_presets=tuple(
                args.reward_preset or ("current_baseline", "balanced")
            ),
            feature_sets=selected_feature_sets,
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
            warm_start_updates=tuple(args.warm_start_updates or WARM_START_UPDATES),
            matchup_samplings=tuple(args.matchup_sampling or MATCHUP_SAMPLINGS),
            reward_mode="combined_terminal",
            reward_presets=("current_baseline",),
            feature_sets=selected_feature_sets,
        )

    require_selection = not args.all
    learning_rates = selected_values(
        phase=args.phase,
        values=args.learning_rate,
        default=MAIN_LEARNING_RATES,
        name="learning_rate",
        require_explicit=require_selection,
    )
    warm_start_updates = selected_int_values(
        phase=args.phase,
        values=args.warm_start_updates,
        default=WARM_START_UPDATES,
        name="warm_start_updates",
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
            warm_start_updates=warm_start_updates,
            matchup_samplings=matchup_samplings,
            reward_mode="combined_terminal",
            reward_presets=("current_baseline",),
            feature_sets=selected_feature_sets,
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
            warm_start_updates=warm_start_updates,
            matchup_samplings=matchup_samplings,
            reward_mode="combined_terminal",
            reward_presets=reward_presets,
            feature_sets=selected_feature_sets,
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
            warm_start_updates=warm_start_updates,
            matchup_samplings=matchup_samplings,
            reward_mode="dense_presa",
            reward_presets=dense_presets,
            feature_sets=selected_feature_sets,
        )

    raise AssertionError(f"Unsupported phase: {args.phase}")


def maybe_run(
    *,
    command: list[str],
    execute: bool,
    force: bool,
    output_path: Path,
) -> None:
    """Print or execute one command, preserving completed outputs by default."""

    print(shlex.join(command))
    if not execute:
        return
    if output_path.exists() and not force:
        print(f"SKIP existing output: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    # Dry runs and real runs share the same parsed configuration and grid.
    parser = argparse.ArgumentParser(
        description="Run the local Briscola RL experimental protocol.",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            *LEGACY_PHASES,
            *FEATURE_COMPARISON_PHASES,
            *NEURAL_PROTOCOL_PHASES,
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
        help="Policy family for legacy phases; neural_* phases always use neural.",
    )
    parser.add_argument(
        "--hidden-size",
        action="append",
        type=int,
        help="Neural hidden size. Ignored by linear runs.",
    )
    parser.add_argument("--learning-rate", action="append")
    parser.add_argument("--feature-set", action="append", choices=FEATURE_SETS)
    neural_baseline_group = parser.add_mutually_exclusive_group()
    neural_baseline_group.add_argument(
        "--neural-learned-baseline",
        dest="neural_learned_baseline",
        action="store_true",
        default=None,
        help="Use the learned value baseline in a neural_* phase.",
    )
    neural_baseline_group.add_argument(
        "--no-neural-learned-baseline",
        dest="neural_learned_baseline",
        action="store_false",
        help="Disable the learned value baseline in a neural_* phase.",
    )
    parser.add_argument("--warm-start-updates", action="append", type=int)
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
