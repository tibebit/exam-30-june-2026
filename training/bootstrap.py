"""Bootstrap policy schedule for early Briscola training."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from policy import GreedyPolicy, HeuristicPolicy, Policy, RandomPolicy


PolicyFactory = Callable[[], Policy]


@dataclass(frozen=True)
class BootstrapPolicySchedule:
    """Sample fixed baseline policies during the initial training updates."""

    bootstrap_updates: int = 30
    policy_factories: tuple[PolicyFactory, ...] = field(
        default_factory=lambda: (RandomPolicy, GreedyPolicy, HeuristicPolicy)
    )

    def __post_init__(self) -> None:
        if self.bootstrap_updates < 0:
            raise ValueError("bootstrap_updates must be non-negative")
        if not self.policy_factories:
            raise ValueError("At least one bootstrap policy factory is required")

    def active(self, update_index: int) -> bool:
        """Return whether bootstrap policies should be used for this update."""

        return update_index < self.bootstrap_updates

    def sample_policy(self, rng: random.Random) -> Policy:
        """Sample one fresh baseline policy instance."""

        return rng.choice(self.policy_factories)()
