"""Self-play orchestration for Briscola training."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from game.rules import NUMERO_GIOCATORI, valida_giocatore_id
from policy import LinearSoftmaxPolicy

from .episode import EpisodeResult, collect_episode
from .pool import SnapshotPool
from .reinforce import ReinforceConfig, TrainStats, reinforce_update
from .rewards import RewardConfig


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for the self-play orchestrator."""

    batch_size: int = 500
    snapshot_interval: int = 50
    learner_giocatore_id: int = 0
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    reinforce_config: ReinforceConfig = field(default_factory=ReinforceConfig)
    greedy_non_learner: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size deve essere positivo")
        if self.batch_size % NUMERO_GIOCATORI != 0:
            raise ValueError("batch_size deve essere multiplo di 4")
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval deve essere positivo")
        valida_giocatore_id(self.learner_giocatore_id)


@dataclass(frozen=True)
class SelfPlayStats:
    """Statistics produced by one self-play update."""

    update_index: int
    train_stats: TrainStats
    pool_size: int
    snapshot_added: bool


@dataclass
class SelfPlayTrainer:
    """Orchestrate self-play batches, REINFORCE updates, and learner snapshots."""

    learner: LinearSoftmaxPolicy
    pool: SnapshotPool
    config: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    seed: int = 0
    update_index: int = 0
    master_rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.master_rng = random.Random(self.seed)
        if len(self.pool) == 0:
            self.pool.add_policy(
                self.learner,
                name="initial",
                update_index=0,
            )

    def train_update(self) -> SelfPlayStats:
        """Collect a balanced batch and update only the learner policy."""

        episodes = [
            self._collect_training_episode(episode_index)
            for episode_index in range(self.config.batch_size)
        ]
        train_stats = reinforce_update(
            self.learner,
            episodes,
            self.config.reinforce_config,
        )

        self.update_index += 1
        snapshot_added = False
        if self.update_index % self.config.snapshot_interval == 0:
            self.pool.add_policy(
                self.learner,
                name=f"snapshot_{self.update_index}",
                update_index=self.update_index,
            )
            snapshot_added = True

        return SelfPlayStats(
            update_index=self.update_index,
            train_stats=train_stats,
            pool_size=len(self.pool),
            snapshot_added=snapshot_added,
        )

    def train(self, updates: int) -> list[SelfPlayStats]:
        """Run multiple consecutive updates."""

        if updates < 0:
            raise ValueError("updates deve essere non negativo")
        return [self.train_update() for _ in range(updates)]

    def _collect_training_episode(self, episode_index: int) -> EpisodeResult:
        primo_giocatore_id = episode_index % NUMERO_GIOCATORI
        seed_ambiente = self.master_rng.getrandbits(32)
        seed_policy = self.master_rng.getrandbits(32)

        return collect_episode(
            learner_policy=self.learner,
            compagno_policy=self.pool.sample_policy(self.master_rng),
            avversario_successivo_policy=self.pool.sample_policy(self.master_rng),
            avversario_precedente_policy=self.pool.sample_policy(self.master_rng),
            learner_giocatore_id=self.config.learner_giocatore_id,
            seed_ambiente=seed_ambiente,
            primo_giocatore_id=primo_giocatore_id,
            rng_policy=random.Random(seed_policy),
            reward_config=self.config.reward_config,
            greedy_non_learner=self.config.greedy_non_learner,
        )
