"""Episode collection for Briscola training."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.cards import Carta
from game.environment import Ambiente
from game.observation import Osservazione
from game.rules import (
    NUMERO_GIOCATORI,
    squadra_avversaria_di,
    squadra_di,
    valida_giocatore_id,
)
from policy import Policy

from .rewards import PUNTI_TOTALI_PARTITA, RewardConfig, reward_finale, reward_presa


MOSSE_PER_GIOCATORE = 10
MOSSE_TOTALI_PARTITA = NUMERO_GIOCATORI * MOSSE_PER_GIOCATORE


@dataclass
class TrajectoryStep:
    """One learner decision to use in the policy gradient."""

    osservazione: Osservazione
    azione: Carta
    global_step_index: int
    reward_to_go: float = 0.0


@dataclass
class EpisodeResult:
    """Complete result of one training episode."""

    steps: list[TrajectoryStep]
    rewards: list[float]
    punteggi_finali: dict[str, int]
    learner_giocatore_id: int
    learner_squadra: str
    episode_return: float


def collect_episode(
    *,
    learner_policy: Policy,
    compagno_policy: Policy,
    avversario_successivo_policy: Policy,
    avversario_precedente_policy: Policy,
    learner_giocatore_id: int,
    seed_ambiente: int,
    primo_giocatore_id: int,
    rng_policy: random.Random,
    reward_config: RewardConfig = RewardConfig(),
    greedy_non_learner: bool = False,
) -> EpisodeResult:
    """Play a full game and collect only the learner decisions."""

    valida_giocatore_id(learner_giocatore_id)
    valida_giocatore_id(primo_giocatore_id)

    learner_squadra = squadra_di(learner_giocatore_id)
    learner_squadra_avversaria = squadra_avversaria_di(learner_squadra)
    ambiente = Ambiente(seed=seed_ambiente, primo_giocatore_id=primo_giocatore_id)
    policy_per_giocatore = _policy_per_giocatore(
        learner_policy=learner_policy,
        compagno_policy=compagno_policy,
        avversario_successivo_policy=avversario_successivo_policy,
        avversario_precedente_policy=avversario_precedente_policy,
        learner_giocatore_id=learner_giocatore_id,
    )

    steps: list[TrajectoryStep] = []
    rewards: list[float] = []
    global_step_index = 0

    while not ambiente.finita:
        giocatore_id = ambiente.giocatore_corrente
        osservazione = ambiente.osserva(giocatore_id)
        policy = policy_per_giocatore[giocatore_id]
        greedy = False if giocatore_id == learner_giocatore_id else greedy_non_learner
        azione = policy.select_action(osservazione, rng_policy, greedy=greedy)

        if giocatore_id == learner_giocatore_id:
            steps.append(
                TrajectoryStep(
                    osservazione=osservazione,
                    azione=azione,
                    global_step_index=global_step_index,
                )
            )

        esito = ambiente.gioca(azione)
        immediate_reward = 0.0

        if esito.presa_completata:
            if esito.vincitore_presa is None:
                raise RuntimeError("Una presa completata deve avere un vincitore")
            immediate_reward += reward_presa(
                punti_presa=esito.punti_presa,
                presa_vinta_da_squadra=(
                    squadra_di(esito.vincitore_presa) == learner_squadra
                ),
                config=reward_config,
            )

        if esito.partita_finita:
            immediate_reward += reward_finale(
                punti_squadra=esito.punteggi[learner_squadra],
                punti_avversari=esito.punteggi[learner_squadra_avversaria],
                config=reward_config,
            )

        rewards.append(float(immediate_reward))
        global_step_index += 1

    ambiente.verifica_integrita_stato()
    punteggi_finali = dict(ambiente.punteggi)
    _verifica_integrita_episode(steps, rewards, punteggi_finali)

    for step in steps:
        step.reward_to_go = float(sum(rewards[step.global_step_index :]))

    return EpisodeResult(
        steps=steps,
        rewards=rewards,
        punteggi_finali=punteggi_finali,
        learner_giocatore_id=learner_giocatore_id,
        learner_squadra=learner_squadra,
        episode_return=float(sum(rewards)),
    )


def _policy_per_giocatore(
    *,
    learner_policy: Policy,
    compagno_policy: Policy,
    avversario_successivo_policy: Policy,
    avversario_precedente_policy: Policy,
    learner_giocatore_id: int,
) -> dict[int, Policy]:
    """Map policies by turn distance from the learner."""

    return {
        learner_giocatore_id: learner_policy,
        (learner_giocatore_id + 1) % NUMERO_GIOCATORI: avversario_successivo_policy,
        (learner_giocatore_id + 2) % NUMERO_GIOCATORI: compagno_policy,
        (learner_giocatore_id + 3) % NUMERO_GIOCATORI: avversario_precedente_policy,
    }


def _verifica_integrita_episode(
    steps: list[TrajectoryStep],
    rewards: list[float],
    punteggi_finali: dict[str, int],
) -> None:
    if len(rewards) != MOSSE_TOTALI_PARTITA:
        raise RuntimeError(
            "Una partita di Briscola deve durare "
            f"{MOSSE_TOTALI_PARTITA} mosse, registrate {len(rewards)}"
        )
    if len(steps) != MOSSE_PER_GIOCATORE:
        raise RuntimeError(
            "Il learner deve giocare "
            f"{MOSSE_PER_GIOCATORE} carte, registrate {len(steps)}"
        )
    if sum(punteggi_finali.values()) != PUNTI_TOTALI_PARTITA:
        raise RuntimeError(
            "I punteggi finali devono sommare "
            f"{PUNTI_TOTALI_PARTITA}, ottenuto {punteggi_finali}"
        )
