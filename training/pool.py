"""Snapshot pool for self-play training."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from policy import BriscolaFeatureExtractor, LinearSoftmaxPolicy, NeuralSoftmaxPolicy


POLICY_TYPE_LINEAR = "linear"
POLICY_TYPE_NEURAL = "neural"


@dataclass(frozen=True)
class Snapshot:
    """Frozen parameters of a historical policy."""

    name: str
    theta: np.ndarray
    update_index: int
    policy_type: str = POLICY_TYPE_LINEAR
    hidden_size: int | None = None


@dataclass
class SnapshotPool:
    """Mechanical pool of sampleable historical snapshots."""

    feature_extractor: BriscolaFeatureExtractor
    max_size: int = 20
    keep_initial: bool = False
    snapshots: list[Snapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise ValueError("max_size deve essere almeno 1")

    def __len__(self) -> int:
        return len(self.snapshots)

    def add_policy(
        self,
        policy: LinearSoftmaxPolicy | NeuralSoftmaxPolicy,
        name: str,
        update_index: int,
    ) -> None:
        """Save a frozen copy of the policy parameters."""

        theta = np.array(policy.theta, dtype=np.float32, copy=True)
        theta.setflags(write=False)
        policy_type = _policy_type(policy)
        self.snapshots.append(
            Snapshot(
                name=name,
                theta=theta,
                update_index=update_index,
                policy_type=policy_type,
                hidden_size=(
                    policy.hidden_size if policy_type == POLICY_TYPE_NEURAL else None
                ),
            )
        )
        self._trim()

    def sample_policy(
        self,
        rng: random.Random,
    ) -> LinearSoftmaxPolicy | NeuralSoftmaxPolicy:
        """Sample a snapshot and rebuild an independent policy."""

        if not self.snapshots:
            raise ValueError("Non si puo campionare da un pool vuoto")
        snapshot = rng.choice(self.snapshots)
        if snapshot.policy_type == POLICY_TYPE_NEURAL:
            if snapshot.hidden_size is None:
                raise ValueError("Neural snapshot senza hidden_size")
            return NeuralSoftmaxPolicy(
                theta=np.array(snapshot.theta, dtype=np.float32, copy=True),
                feature_extractor=self.feature_extractor,
                hidden_size=snapshot.hidden_size,
                name=snapshot.name,
            )

        return LinearSoftmaxPolicy(
            theta=np.array(snapshot.theta, dtype=np.float32, copy=True),
            feature_extractor=self.feature_extractor,
            name=snapshot.name,
        )

    def _trim(self) -> None:
        if len(self.snapshots) <= self.max_size:
            return

        if self.keep_initial and self.max_size > 1:
            initial = self.snapshots[0]
            recent = self.snapshots[-(self.max_size - 1) :]
            self.snapshots = [initial, *recent]
            return

        self.snapshots = self.snapshots[-self.max_size :]


def _policy_type(policy: LinearSoftmaxPolicy | NeuralSoftmaxPolicy) -> str:
    if isinstance(policy, NeuralSoftmaxPolicy):
        return POLICY_TYPE_NEURAL
    return POLICY_TYPE_LINEAR
