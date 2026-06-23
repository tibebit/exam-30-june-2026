"""Self-play orchestration for Briscola training."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from game.rules import NUMERO_GIOCATORI, valida_giocatore_id
from policy import LinearSoftmaxPolicy, Policy

from .bootstrap import BootstrapPolicySchedule
from .episode import EpisodeResult, collect_episode
from .pool import SnapshotPool
from .reinforce import ReinforceConfig, TrainStats, reinforce_update
from .rewards import RewardConfig


MatchupSamplingMode = Literal["per_episode", "per_rotation_block"]
MATCHUP_SAMPLING_MODES = {"per_episode", "per_rotation_block"}


@dataclass(frozen=True)
class PolicyMatchup:
    """Fixed non-learner policy trio used in one or more episodes."""

    compagno_policy: Policy
    avversario_successivo_policy: Policy
    avversario_precedente_policy: Policy


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configurazione dell'orchestratore self-play."""

    batch_size: int = 500
    snapshot_interval: int = 50
    learner_giocatore_id: int = 0
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    reinforce_config: ReinforceConfig = field(default_factory=ReinforceConfig)
    bootstrap_schedule: BootstrapPolicySchedule = field(
        default_factory=BootstrapPolicySchedule
    )
    greedy_non_learner: bool = False
    matchup_sampling: MatchupSamplingMode = "per_episode"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size deve essere positivo")
        if self.batch_size % NUMERO_GIOCATORI != 0:
            raise ValueError("batch_size deve essere multiplo di 4")
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval deve essere positivo")
        if self.matchup_sampling not in MATCHUP_SAMPLING_MODES:
            raise ValueError(f"Matchup sampling non supportato: {self.matchup_sampling}")
        valida_giocatore_id(self.learner_giocatore_id)


@dataclass(frozen=True)
class SelfPlayStats:
    """Statistiche prodotte da un update self-play."""

    update_index: int
    train_stats: TrainStats
    pool_size: int
    snapshot_added: bool


@dataclass
class SelfPlayTrainer:
    """Orchestra batch self-play, update REINFORCE e snapshot del learner."""

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
        """Raccoglie un batch bilanciato e aggiorna solo la policy learner."""

        episodes = self._collect_training_batch()
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
        """Esegue piu' update consecutivi."""

        if updates < 0:
            raise ValueError("updates deve essere non negativo")
        return [self.train_update() for _ in range(updates)]

    def _collect_training_batch(self) -> list[EpisodeResult]:
        if self.config.matchup_sampling == "per_episode":
            return [
                self._collect_training_episode(
                    primo_giocatore_id=episode_index % NUMERO_GIOCATORI,
                )
                for episode_index in range(self.config.batch_size)
            ]

        if self.config.matchup_sampling == "per_rotation_block":
            episodes: list[EpisodeResult] = []
            for _ in range(self.config.batch_size // NUMERO_GIOCATORI):
                matchup = self._sample_matchup()
                for primo_giocatore_id in range(NUMERO_GIOCATORI):
                    episodes.append(
                        self._collect_training_episode(
                            primo_giocatore_id=primo_giocatore_id,
                            matchup=matchup,
                        )
                    )
            return episodes

        raise ValueError(
            f"Matchup sampling non supportato: {self.config.matchup_sampling}"
        )

    def _sample_matchup(self) -> PolicyMatchup:
        return PolicyMatchup(
            compagno_policy=self._sample_non_learner_policy(),
            avversario_successivo_policy=self._sample_non_learner_policy(),
            avversario_precedente_policy=self._sample_non_learner_policy(),
        )

    def _collect_training_episode(
        self,
        *,
        primo_giocatore_id: int,
        matchup: PolicyMatchup | None = None,
    ) -> EpisodeResult:
        seed_ambiente = self.master_rng.getrandbits(32)
        seed_policy = self.master_rng.getrandbits(32)
        if matchup is None:
            matchup = self._sample_matchup()

        return collect_episode(
            learner_policy=self.learner,
            compagno_policy=matchup.compagno_policy,
            avversario_successivo_policy=matchup.avversario_successivo_policy,
            avversario_precedente_policy=matchup.avversario_precedente_policy,
            learner_giocatore_id=self.config.learner_giocatore_id,
            seed_ambiente=seed_ambiente,
            primo_giocatore_id=primo_giocatore_id,
            rng_policy=random.Random(seed_policy),
            reward_config=self.config.reward_config,
            greedy_non_learner=self.config.greedy_non_learner,
        )

    def _sample_non_learner_policy(self) -> Policy:
        if self.config.bootstrap_schedule.active(self.update_index):
            return self.config.bootstrap_schedule.sample_policy(self.master_rng)
        return self.pool.sample_policy(self.master_rng)
