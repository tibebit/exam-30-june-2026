"""Snapshot pool for self-play training."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from policy import BriscolaFeatureExtractor, LinearSoftmaxPolicy


@dataclass(frozen=True)
class Snapshot:
    """Frozen parameters of a historical policy."""

    name: str
    theta: np.ndarray
    update_index: int


@dataclass
class SnapshotPool:
    """Mechanical pool of sampleable historical snapshots."""

    feature_extractor: BriscolaFeatureExtractor
    max_size: int = 20
    keep_initial: bool = True
    snapshots: list[Snapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise ValueError("max_size deve essere almeno 1")

    def __len__(self) -> int:
        return len(self.snapshots)

    def add_policy(
        self,
        policy: LinearSoftmaxPolicy,
        name: str,
        update_index: int,
    ) -> None:
        """Save a frozen copy of the policy parameters."""

        theta = np.array(policy.theta, dtype=np.float32, copy=True)
        theta.setflags(write=False)
        self.snapshots.append(
            Snapshot(
                name=name,
                theta=theta,
                update_index=update_index,
            )
        )
        self._trim()

    def sample_policy(self, rng: random.Random) -> LinearSoftmaxPolicy:
        """Sample a snapshot and rebuild an independent policy."""

        if not self.snapshots:
            raise ValueError("Non si puo campionare da un pool vuoto")
        snapshot = rng.choice(self.snapshots)
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
