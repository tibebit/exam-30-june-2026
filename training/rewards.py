"""Reward helpers for Briscola training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PUNTI_TOTALI_PARTITA = 120
REWARD_MODES = {"combined_terminal", "dense_presa"}

RewardMode = Literal["combined_terminal", "dense_presa"]


@dataclass(frozen=True)
class RewardConfig:
    """Explicit training reward configuration."""

    mode: RewardMode = "combined_terminal"
    alpha: float = 1.0
    lambda_margin: float = 0.2

    def __post_init__(self) -> None:
        if self.mode not in REWARD_MODES:
            raise ValueError(f"Reward mode non supportata: {self.mode}")
        if self.alpha < 0.0:
            raise ValueError("alpha deve essere non negativo")
        if self.lambda_margin < 0.0:
            raise ValueError("lambda_margin deve essere non negativo")


def calcola_margine(punti_squadra: int, punti_avversari: int) -> int:
    """Compute the margin from the learner's team perspective."""

    return punti_squadra - punti_avversari


def calcola_segno(margine: int) -> float:
    """Encode win, loss, and draw as a numeric sign."""

    if margine > 0:
        return 1.0
    if margine < 0:
        return -1.0
    return 0.0


def normalizza_margine(margine: int) -> float:
    """Scale a point margin to the full-game range."""

    return float(margine) / PUNTI_TOTALI_PARTITA


def reward_finale(
    punti_squadra: int,
    punti_avversari: int,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Compute the reward assigned at the end of the game."""

    margine = calcola_margine(punti_squadra, punti_avversari)
    segno = calcola_segno(margine)

    if config.mode == "combined_terminal":
        return float(
            config.alpha * segno
            + config.lambda_margin * normalizza_margine(margine)
        )
    if config.mode == "dense_presa":
        return float(config.alpha * segno)

    raise ValueError(f"Reward mode non supportata: {config.mode}")


def reward_presa(
    punti_presa: int,
    presa_vinta_da_squadra: bool,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Compute the immediate reward when a presa is completed."""

    if config.mode == "combined_terminal":
        return 0.0

    if config.mode == "dense_presa":
        segno = 1.0 if presa_vinta_da_squadra else -1.0
        return float(
            config.lambda_margin * segno * (punti_presa / PUNTI_TOTALI_PARTITA)
        )

    raise ValueError(f"Reward mode non supportata: {config.mode}")
