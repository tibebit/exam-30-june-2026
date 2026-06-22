"""Evaluation utilities for frozen Briscola policies."""

from .match import MatchResult, play_match
from .metrics import EvaluationMetrics, compute_metrics, standard_error
from .suite import (
    EvaluationCase,
    EvaluationScenario,
    EvaluationSuite,
    ScenarioEvaluationResult,
    default_evaluation_suite,
    evaluate_learner_scenario,
    evaluate_suite,
    make_evaluation_cases,
)

__all__ = [
    "EvaluationCase",
    "EvaluationMetrics",
    "EvaluationScenario",
    "EvaluationSuite",
    "MatchResult",
    "ScenarioEvaluationResult",
    "compute_metrics",
    "default_evaluation_suite",
    "evaluate_learner_scenario",
    "evaluate_suite",
    "make_evaluation_cases",
    "play_match",
    "standard_error",
]
