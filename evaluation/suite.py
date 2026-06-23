"""Evaluation suites built from repeated frozen Briscola matches."""

from __future__ import annotations

from dataclasses import dataclass, field

from game.rules import (
    NUMERO_GIOCATORI,
    squadra_avversaria_di,
    squadra_di,
    valida_giocatore_id,
)
from policy import (
    AdvancedHeuristicPolicy,
    GreedyPolicy,
    HeuristicPolicy,
    PerfectHeuristicPolicy,
    Policy,
    RandomPolicy,
)

from .match import MatchResult, play_match
from .metrics import EvaluationMetrics, compute_metrics


@dataclass(frozen=True)
class EvaluationCase:
    """One independent match configuration for evaluation."""

    seed_ambiente: int
    seed_policy: int
    primo_giocatore_id: int


@dataclass(frozen=True)
class EvaluationScenario:
    """Frozen policy roles used to evaluate one learner."""

    name: str
    compagno_policy: Policy
    avversario_successivo_policy: Policy
    avversario_precedente_policy: Policy


@dataclass(frozen=True)
class ScenarioEvaluationResult:
    """Aggregated result for one learner evaluation scenario."""

    scenario_name: str
    metrics: EvaluationMetrics
    matches: tuple[MatchResult, ...]


@dataclass(frozen=True)
class EvaluationSuite:
    """Named collection of immutable learner evaluation scenarios."""

    scenarios: tuple[EvaluationScenario, ...] = field(default_factory=tuple)


def make_evaluation_cases(
    *,
    games: int,
    seed_ambiente_start: int,
    seed_policy_start: int,
) -> list[EvaluationCase]:
    """Build deterministic evaluation cases with rotating first player."""

    if games <= 0:
        raise ValueError("games deve essere positivo")

    return [
        EvaluationCase(
            seed_ambiente=seed_ambiente_start + index,
            seed_policy=seed_policy_start + index,
            primo_giocatore_id=index % NUMERO_GIOCATORI,
        )
        for index in range(games)
    ]


def default_evaluation_suite() -> EvaluationSuite:
    """Return a small fixed suite of baseline learner scenarios."""

    return EvaluationSuite(
        scenarios=(
            EvaluationScenario(
                name="random_eval",
                compagno_policy=RandomPolicy(),
                avversario_successivo_policy=RandomPolicy(),
                avversario_precedente_policy=RandomPolicy(),
            ),
            EvaluationScenario(
                name="random_partner_heuristic_opponents_eval",
                compagno_policy=RandomPolicy(),
                avversario_successivo_policy=HeuristicPolicy(),
                avversario_precedente_policy=HeuristicPolicy(),
            ),
            EvaluationScenario(
                name="heuristic_eval",
                compagno_policy=HeuristicPolicy(),
                avversario_successivo_policy=HeuristicPolicy(),
                avversario_precedente_policy=HeuristicPolicy(),
            ),
            EvaluationScenario(
                name="greedy_eval",
                compagno_policy=GreedyPolicy(),
                avversario_successivo_policy=GreedyPolicy(),
                avversario_precedente_policy=GreedyPolicy(),
            ),
            EvaluationScenario(
                name="advanced_heuristic_eval",
                compagno_policy=AdvancedHeuristicPolicy(),
                avversario_successivo_policy=AdvancedHeuristicPolicy(),
                avversario_precedente_policy=AdvancedHeuristicPolicy(),
            ),
            EvaluationScenario(
                name="advanced_partner_perfect_heuristic_opponents_eval",
                compagno_policy=AdvancedHeuristicPolicy(),
                avversario_successivo_policy=PerfectHeuristicPolicy(),
                avversario_precedente_policy=PerfectHeuristicPolicy(),
            ),
        )
    )


def evaluate_learner_scenario(
    *,
    learner_policy: Policy,
    scenario: EvaluationScenario,
    cases: list[EvaluationCase],
    learner_giocatore_id: int = 0,
    greedy: bool = True,
) -> ScenarioEvaluationResult:
    """Evaluate one learner in one frozen scenario."""

    if not cases:
        raise ValueError("Serve almeno un caso di evaluation")
    valida_giocatore_id(learner_giocatore_id)

    matches = tuple(
        play_match(
            policies_by_player=_policies_by_player(
                learner_policy=learner_policy,
                scenario=scenario,
                learner_giocatore_id=learner_giocatore_id,
            ),
            seed_ambiente=case.seed_ambiente,
            seed_policy=case.seed_policy,
            primo_giocatore_id=case.primo_giocatore_id,
            greedy=greedy,
        )
        for case in cases
    )
    point_differences = [
        _margine_per_learner(match, learner_giocatore_id)
        for match in matches
    ]

    return ScenarioEvaluationResult(
        scenario_name=scenario.name,
        metrics=compute_metrics(point_differences),
        matches=matches,
    )


def evaluate_suite(
    *,
    learner_policy: Policy,
    suite: EvaluationSuite,
    cases: list[EvaluationCase],
    learner_giocatore_id: int = 0,
    greedy: bool = True,
) -> dict[str, ScenarioEvaluationResult]:
    """Evaluate one learner across all scenarios in a suite."""

    return {
        scenario.name: evaluate_learner_scenario(
            learner_policy=learner_policy,
            scenario=scenario,
            cases=cases,
            learner_giocatore_id=learner_giocatore_id,
            greedy=greedy,
        )
        for scenario in suite.scenarios
    }


def _policies_by_player(
    *,
    learner_policy: Policy,
    scenario: EvaluationScenario,
    learner_giocatore_id: int,
) -> dict[int, Policy]:
    return {
        learner_giocatore_id: learner_policy,
        (learner_giocatore_id + 1) % NUMERO_GIOCATORI: (
            scenario.avversario_successivo_policy
        ),
        (learner_giocatore_id + 2) % NUMERO_GIOCATORI: scenario.compagno_policy,
        (learner_giocatore_id + 3) % NUMERO_GIOCATORI: (
            scenario.avversario_precedente_policy
        ),
    }


def _margine_per_learner(
    match: MatchResult,
    learner_giocatore_id: int,
) -> int:
    learner_squadra = squadra_di(learner_giocatore_id)
    avversari = squadra_avversaria_di(learner_squadra)
    return match.punteggi_finali[learner_squadra] - match.punteggi_finali[avversari]
