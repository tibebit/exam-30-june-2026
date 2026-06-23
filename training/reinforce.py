"""REINFORCE update for collected Briscola episodes."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Literal

import numpy as np

from game.rules import squadra_avversaria_di
from policy import LinearSoftmaxPolicy
from policy.linear_softmax_policy import add_scaled_in_place, vector_norm

from .episode import EpisodeResult, TrajectoryStep


BaselineMode = Literal["none", "batch_mean", "time_dependent"]
BASELINE_MODES = {"none", "batch_mean", "time_dependent"}


@dataclass(frozen=True)
class ReinforceConfig:
    """Configuration for a single REINFORCE update."""

    learning_rate: float = 0.01
    baseline: BaselineMode = "time_dependent"
    # Default None: clipping is a hyperparameter, not part of the initial protocol.
    max_update_norm: float | None = None

    def __post_init__(self) -> None:
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate deve essere non negativo")
        if self.baseline not in BASELINE_MODES:
            raise ValueError(f"Baseline non supportata: {self.baseline}")
        if self.max_update_norm is not None and self.max_update_norm < 0.0:
            raise ValueError("max_update_norm deve essere non negativo o None")


@dataclass(frozen=True)
class TrainStats:
    """Synthetic metrics produced by an update."""

    episodes: int
    learner_decisions: int
    mean_return: float
    mean_score_margin: float
    gradient_norm: float
    baseline: BaselineMode
    baseline_values: tuple[float, ...]


def reinforce_update(
    policy: LinearSoftmaxPolicy,
    episodes: list[EpisodeResult],
    config: ReinforceConfig = ReinforceConfig(),
) -> TrainStats:
    """Apply one REINFORCE update from an already collected episode batch."""

    if not episodes:
        raise ValueError("Serve almeno un episodio per fare un update")

    steps = _all_steps(episodes)
    if not steps:
        raise ValueError("Serve almeno una decisione del learner per fare un update")

    baseline_values = _baseline_values(episodes, config.baseline)
    gradient = np.zeros_like(policy.theta, dtype=np.float32)

    for episode in episodes:
        for decision_index, step in enumerate(episode.steps):
            advantage = step.reward_to_go - _baseline_for_step(
                decision_index,
                baseline=config.baseline,
                baseline_values=baseline_values,
            )
            grad_log_probability = policy.grad_log_probability(
                step.osservazione,
                step.azione,
            )
            # Average over episodes: the learning rate stays tied to games.
            add_scaled_in_place(
                gradient,
                grad_log_probability,
                advantage / len(episodes),
            )

    gradient_norm = vector_norm(gradient)
    policy.apply_gradient(
        gradient,
        learning_rate=config.learning_rate,
        max_update_norm=config.max_update_norm,
    )

    return TrainStats(
        episodes=len(episodes),
        learner_decisions=len(steps),
        mean_return=float(mean(episode.episode_return for episode in episodes)),
        mean_score_margin=float(mean(_score_margin(episode) for episode in episodes)),
        gradient_norm=gradient_norm,
        baseline=config.baseline,
        baseline_values=baseline_values,
    )


def _all_steps(episodes: list[EpisodeResult]) -> list[TrajectoryStep]:
    return [step for episode in episodes for step in episode.steps]


def _baseline_values(
    episodes: list[EpisodeResult],
    baseline: BaselineMode,
) -> tuple[float, ...]:
    if baseline == "none":
        return ()

    if baseline == "batch_mean":
        return (float(mean(step.reward_to_go for step in _all_steps(episodes))),)

    if baseline == "time_dependent":
        returns_by_decision: list[list[float]] = []
        for episode in episodes:
            for decision_index, step in enumerate(episode.steps):
                while len(returns_by_decision) <= decision_index:
                    returns_by_decision.append([])
                returns_by_decision[decision_index].append(step.reward_to_go)
        return tuple(float(mean(values)) for values in returns_by_decision)

    raise ValueError(f"Baseline non supportata: {baseline}")


def _baseline_for_step(
    decision_index: int,
    *,
    baseline: BaselineMode,
    baseline_values: tuple[float, ...],
) -> float:
    if baseline == "none":
        return 0.0
    if baseline == "batch_mean":
        return baseline_values[0]
    if baseline == "time_dependent":
        return baseline_values[decision_index]
    raise ValueError(f"Baseline non supportata: {baseline}")


def _score_margin(episode: EpisodeResult) -> int:
    squadra_avversaria = squadra_avversaria_di(episode.learner_squadra)
    return (
        episode.punteggi_finali[episode.learner_squadra]
        - episode.punteggi_finali[squadra_avversaria]
    )
