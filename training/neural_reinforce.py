"""PyTorch REINFORCE update for neural Briscola policies."""

from __future__ import annotations

from collections.abc import Sequence
import random
from statistics import mean

import numpy as np
import torch
from torch import nn

from game.observation import Osservazione
from policy import BriscolaFeatureExtractor, NeuralSoftmaxPolicy

from .episode import EpisodeResult
from .reinforce import (
    ReinforceConfig,
    TrainStats,
    _all_steps,
    _baseline_for_step,
    _baseline_values,
    _score_margin,
)


class NeuralValueBaseline(nn.Module):
    """One-hidden-layer value baseline for neural REINFORCE."""

    def __init__(
        self,
        feature_extractor: BriscolaFeatureExtractor | None = None,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size deve essere positivo")

        self.feature_extractor = feature_extractor or BriscolaFeatureExtractor()
        self.hidden_size = int(hidden_size)
        input_size = self.feature_extractor.size()
        self.hidden_layer = nn.Linear(input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 1)
        self._zero_parameters()

    @classmethod
    def initialize(
        cls,
        feature_extractor: BriscolaFeatureExtractor | None = None,
        rng: random.Random | None = None,
        hidden_size: int = 64,
        scale: float = 0.01,
    ) -> NeuralValueBaseline:
        """Initialize small random value parameters matching the feature dimension."""

        feature_extractor = feature_extractor or BriscolaFeatureExtractor()
        rng = rng or random.Random()
        baseline = cls(
            feature_extractor=feature_extractor,
            hidden_size=hidden_size,
        )
        values = np.asarray(
            [
                rng.uniform(-scale, scale)
                for _ in range(
                    baseline.parameter_count(
                        feature_extractor.size(),
                        hidden_size,
                    )
                )
            ],
            dtype=np.float32,
        )
        baseline.load_flat_parameters(values)
        return baseline

    @staticmethod
    def parameter_count(input_size: int, hidden_size: int) -> int:
        """Return the number of flat parameters for the configured value MLP."""

        if input_size <= 0:
            raise ValueError("input_size deve essere positivo")
        if hidden_size <= 0:
            raise ValueError("hidden_size deve essere positivo")
        return hidden_size * input_size + hidden_size + hidden_size + 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate V(observation) from one pooled state feature vector."""

        hidden = torch.tanh(self.hidden_layer(features))
        return self.output_layer(hidden).squeeze(-1)

    def value_tensor(self, osservazione: Osservazione) -> torch.Tensor:
        """Return a differentiable value estimate for the observation state."""

        return self.forward(self._state_feature_tensor(osservazione))

    def _state_feature_tensor(self, osservazione: Osservazione) -> torch.Tensor:
        cards = list(osservazione.azioni_legali)
        if not cards:
            raise ValueError("No legal actions available")
        feature_batch = np.asarray(
            [
                self.feature_extractor.extract(osservazione, carta)
                for carta in cards
            ],
            dtype=np.float32,
        )
        # Pool legal-action features so the baseline depends on the state, not
        # on the sampled action. This keeps the policy-gradient estimator valid.
        return torch.from_numpy(feature_batch.mean(axis=0))

    def load_flat_parameters(self, values: Sequence[float] | np.ndarray) -> None:
        """Load parameters from a flat deterministic initialization vector."""

        values_array = np.asarray(values, dtype=np.float32)
        expected_shape = (
            self.parameter_count(self.feature_extractor.size(), self.hidden_size),
        )
        if values_array.shape != expected_shape:
            raise ValueError(
                f"Theta shape {values_array.shape} does not match "
                f"value parameter shape {expected_shape}"
            )

        offset = 0
        with torch.no_grad():
            for parameter in self._ordered_parameters():
                size = parameter.numel()
                chunk = values_array[offset : offset + size].reshape(parameter.shape)
                parameter.copy_(torch.from_numpy(chunk))
                offset += size

    def _ordered_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return (
            self.hidden_layer.weight,
            self.hidden_layer.bias,
            self.output_layer.weight,
            self.output_layer.bias,
        )

    def _zero_parameters(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()


def neural_reinforce_update(
    policy: NeuralSoftmaxPolicy,
    episodes: list[EpisodeResult],
    config: ReinforceConfig = ReinforceConfig(),
    optimizer: torch.optim.Optimizer | None = None,
    value_baseline: NeuralValueBaseline | None = None,
    value_optimizer: torch.optim.Optimizer | None = None,
) -> TrainStats:
    """Apply one PyTorch REINFORCE update from collected episode batches."""

    if not episodes:
        raise ValueError("Serve almeno un episodio per fare un update")

    steps = _all_steps(episodes)
    if not steps:
        raise ValueError("Serve almeno una decisione del learner per fare un update")

    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = config.learning_rate
    if value_baseline is not None:
        if value_optimizer is None:
            value_optimizer = torch.optim.Adam(
                value_baseline.parameters(),
                lr=config.learning_rate,
            )
        for parameter_group in value_optimizer.param_groups:
            parameter_group["lr"] = config.learning_rate

    baseline_values = (
        ()
        if value_baseline is not None
        else _baseline_values(episodes, config.baseline)
    )
    optimizer.zero_grad()
    if value_optimizer is not None:
        value_optimizer.zero_grad()

    losses: list[torch.Tensor] = []
    entropies: list[float] = []
    value_losses: list[torch.Tensor] = []
    value_loss_values: list[float] = []
    for episode in episodes:
        for decision_index, step in enumerate(episode.steps):
            reward_to_go = torch.tensor(step.reward_to_go, dtype=torch.float32)
            if value_baseline is None:
                advantage = reward_to_go - _baseline_for_step(
                    decision_index,
                    baseline=config.baseline,
                    baseline_values=baseline_values,
                )
            else:
                baseline_prediction = value_baseline.value_tensor(step.osservazione)
                advantage = reward_to_go - baseline_prediction.detach()
                raw_value_loss = 0.5 * (baseline_prediction - reward_to_go).pow(2)
                value_losses.append(raw_value_loss / len(steps))
                value_loss_values.append(float(raw_value_loss.detach().item()))
            cards, logits = policy.action_logits_tensor(step.osservazione)
            if step.azione not in cards:
                raise ValueError("Action is not legal")
            action_index = cards.index(step.azione)
            log_probabilities = torch.log_softmax(logits, dim=0)
            probabilities = torch.softmax(logits, dim=0)
            entropy = -(probabilities * log_probabilities).sum()
            entropies.append(float(entropy.detach().item()))
            # Average over episodes: the learning rate stays tied to games.
            losses.append(
                -log_probabilities[action_index] * (advantage / len(episodes))
                - config.entropy_coef * entropy / len(episodes)
            )

    loss = torch.stack(losses).sum()
    if value_losses:
        loss = loss + torch.stack(value_losses).sum()
    loss.backward()

    gradient_norm = _gradient_norm(policy)
    optimizer.step()
    if value_optimizer is not None:
        value_optimizer.step()

    return TrainStats(
        episodes=len(episodes),
        learner_decisions=len(steps),
        mean_return=float(mean(episode.episode_return for episode in episodes)),
        mean_score_margin=float(mean(_score_margin(episode) for episode in episodes)),
        gradient_norm=gradient_norm,
        baseline="learned_value" if value_baseline is not None else config.baseline,
        baseline_values=() if value_baseline is not None else baseline_values,
        mean_entropy=float(mean(entropies)),
        mean_value_loss=(
            float(mean(value_loss_values))
            if value_loss_values
            else None
        ),
    )


def _gradient_norm(policy: NeuralSoftmaxPolicy) -> float:
    gradients = [
        parameter.grad.detach().norm(2)
        for parameter in policy.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(gradients), ord=2).item())
