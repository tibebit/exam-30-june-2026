"""Training utilities for Briscola reinforcement learning."""

from .episode import (
    MOSSE_PER_GIOCATORE,
    MOSSE_TOTALI_PARTITA,
    EpisodeResult,
    TrajectoryStep,
    collect_episode,
)
from .bootstrap import BootstrapPolicySchedule
from .pool import Snapshot, SnapshotPool
from .neural_reinforce import NeuralValueBaseline, neural_reinforce_update
from .reinforce import (
    BASELINE_MODES,
    ReinforceConfig,
    TrainablePolicy,
    TrainStats,
    reinforce_update,
)
from .rewards import (
    PUNTI_TOTALI_PARTITA,
    REWARD_MODES,
    RewardConfig,
    calcola_margine,
    calcola_segno,
    normalizza_margine,
    reward_finale,
    reward_presa,
)
from .self_play import (
    MATCHUP_SAMPLING_MODES,
    MatchupSamplingMode,
    SelfPlayConfig,
    SelfPlayStats,
    SelfPlayTrainer,
)

__all__ = [
    "MOSSE_PER_GIOCATORE",
    "MOSSE_TOTALI_PARTITA",
    "PUNTI_TOTALI_PARTITA",
    "BASELINE_MODES",
    "MATCHUP_SAMPLING_MODES",
    "REWARD_MODES",
    "BootstrapPolicySchedule",
    "EpisodeResult",
    "MatchupSamplingMode",
    "NeuralValueBaseline",
    "ReinforceConfig",
    "RewardConfig",
    "Snapshot",
    "SnapshotPool",
    "SelfPlayConfig",
    "SelfPlayStats",
    "SelfPlayTrainer",
    "TrajectoryStep",
    "TrainablePolicy",
    "TrainStats",
    "calcola_margine",
    "calcola_segno",
    "collect_episode",
    "normalizza_margine",
    "neural_reinforce_update",
    "reinforce_update",
    "reward_finale",
    "reward_presa",
]
