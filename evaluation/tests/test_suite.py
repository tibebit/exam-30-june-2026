from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from evaluation.match import MatchResult
from evaluation.suite import (
    EvaluationCase,
    EvaluationScenario,
    EvaluationSuite,
    default_evaluation_suite,
    evaluate_learner_scenario,
    evaluate_suite,
    make_evaluation_cases,
)


@dataclass
class FakePolicy:
    name: str


def match_result(pari: int, dispari: int) -> MatchResult:
    return MatchResult(
        punteggi_finali={"pari": pari, "dispari": dispari},
        squadra_vincitrice="pari" if pari > dispari else "dispari",
        margine_squadra_pari=pari - dispari,
        seed_ambiente=0,
        seed_policy=0,
        primo_giocatore_id=0,
    )


class TestEvaluationSuite(unittest.TestCase):
    def test_make_evaluation_cases_ruota_primo_giocatore(self):
        # Ogni caso e' una partita indipendente con apertura bilanciata.
        cases = make_evaluation_cases(
            games=6,
            seed_ambiente_start=100,
            seed_policy_start=200,
        )

        self.assertEqual(
            cases,
            [
                EvaluationCase(100, 200, 0),
                EvaluationCase(101, 201, 1),
                EvaluationCase(102, 202, 2),
                EvaluationCase(103, 203, 3),
                EvaluationCase(104, 204, 0),
                EvaluationCase(105, 205, 1),
            ],
        )

    def test_make_evaluation_cases_rifiuta_games_non_positivo(self):
        # Una suite senza partite non produce metriche interpretabili.
        with self.assertRaises(ValueError):
            make_evaluation_cases(
                games=0,
                seed_ambiente_start=100,
                seed_policy_start=200,
            )

    def test_default_evaluation_suite_ha_scenari_stabili(self):
        # I nomi dei benchmark default devono restare leggibili e prevedibili.
        suite = default_evaluation_suite()

        self.assertEqual(
            [scenario.name for scenario in suite.scenarios],
            [
                "random_eval",
                "random_partner_heuristic_opponents_eval",
                "heuristic_eval",
                "greedy_eval",
                "advanced_heuristic_eval",
            ],
        )

    def test_evaluate_learner_scenario_mappa_i_ruoli_dal_learner(self):
        # La suite valuta learner, avversario successivo, compagno e precedente.
        learner = FakePolicy("learner")
        scenario = EvaluationScenario(
            name="custom",
            compagno_policy=FakePolicy("compagno"),
            avversario_successivo_policy=FakePolicy("successivo"),
            avversario_precedente_policy=FakePolicy("precedente"),
        )
        cases = [EvaluationCase(100, 200, 2)]

        with patch(
            "evaluation.suite.play_match",
            return_value=match_result(70, 50),
        ) as play:
            evaluate_learner_scenario(
                learner_policy=learner,
                scenario=scenario,
                cases=cases,
                learner_giocatore_id=1,
            )

        policies_by_player = play.call_args.kwargs["policies_by_player"]
        self.assertIs(policies_by_player[1], learner)
        self.assertIs(policies_by_player[2], scenario.avversario_successivo_policy)
        self.assertIs(policies_by_player[3], scenario.compagno_policy)
        self.assertIs(policies_by_player[0], scenario.avversario_precedente_policy)
        self.assertEqual(play.call_args.kwargs["seed_ambiente"], 100)
        self.assertEqual(play.call_args.kwargs["seed_policy"], 200)
        self.assertEqual(play.call_args.kwargs["primo_giocatore_id"], 2)
        self.assertTrue(play.call_args.kwargs["greedy"])

    def test_margine_metriche_usa_prospettiva_del_learner(self):
        # Se il learner e' dispari, il margine deve essere dispari meno pari.
        learner = FakePolicy("learner")
        scenario = EvaluationScenario(
            name="custom",
            compagno_policy=FakePolicy("compagno"),
            avversario_successivo_policy=FakePolicy("successivo"),
            avversario_precedente_policy=FakePolicy("precedente"),
        )
        cases = [
            EvaluationCase(100, 200, 0),
            EvaluationCase(101, 201, 1),
        ]

        with patch(
            "evaluation.suite.play_match",
            side_effect=[
                match_result(70, 50),
                match_result(40, 80),
            ],
        ):
            result = evaluate_learner_scenario(
                learner_policy=learner,
                scenario=scenario,
                cases=cases,
                learner_giocatore_id=1,
            )

        self.assertEqual(result.scenario_name, "custom")
        self.assertEqual(result.metrics.games, 2)
        self.assertEqual(result.metrics.mean_point_difference, 10.0)
        self.assertEqual(result.metrics.win_rate, 0.5)
        self.assertEqual(result.metrics.loss_rate, 0.5)

    def test_evaluate_suite_esegue_tutti_gli_scenari(self):
        # La suite restituisce un risultato per ogni scenario configurato.
        learner = FakePolicy("learner")
        suite = EvaluationSuite(
            scenarios=(
                EvaluationScenario(
                    name="first",
                    compagno_policy=FakePolicy("c1"),
                    avversario_successivo_policy=FakePolicy("s1"),
                    avversario_precedente_policy=FakePolicy("p1"),
                ),
                EvaluationScenario(
                    name="second",
                    compagno_policy=FakePolicy("c2"),
                    avversario_successivo_policy=FakePolicy("s2"),
                    avversario_precedente_policy=FakePolicy("p2"),
                ),
            )
        )
        cases = [EvaluationCase(100, 200, 0)]

        with patch(
            "evaluation.suite.play_match",
            return_value=match_result(70, 50),
        ):
            results = evaluate_suite(
                learner_policy=learner,
                suite=suite,
                cases=cases,
            )

        self.assertEqual(set(results), {"first", "second"})
        self.assertEqual(results["first"].metrics.mean_point_difference, 20.0)
        self.assertEqual(results["second"].metrics.mean_point_difference, 20.0)

    def test_greedy_false_viene_propagato_al_match(self):
        # La suite default e' greedy, ma puo' orchestrare evaluation stocastiche.
        learner = FakePolicy("learner")
        scenario = EvaluationScenario(
            name="custom",
            compagno_policy=FakePolicy("compagno"),
            avversario_successivo_policy=FakePolicy("successivo"),
            avversario_precedente_policy=FakePolicy("precedente"),
        )
        cases = [EvaluationCase(100, 200, 0)]

        with patch(
            "evaluation.suite.play_match",
            return_value=match_result(70, 50),
        ) as play:
            evaluate_learner_scenario(
                learner_policy=learner,
                scenario=scenario,
                cases=cases,
                greedy=False,
            )

        self.assertFalse(play.call_args.kwargs["greedy"])

    def test_casi_invalidi_solleva_value_error(self):
        # Serve almeno un caso valido e un learner id nel range 0..3.
        learner = FakePolicy("learner")
        scenario = EvaluationScenario(
            name="custom",
            compagno_policy=FakePolicy("compagno"),
            avversario_successivo_policy=FakePolicy("successivo"),
            avversario_precedente_policy=FakePolicy("precedente"),
        )

        with self.assertRaises(ValueError):
            evaluate_learner_scenario(
                learner_policy=learner,
                scenario=scenario,
                cases=[],
            )

        with self.assertRaises(ValueError):
            evaluate_learner_scenario(
                learner_policy=learner,
                scenario=scenario,
                cases=[EvaluationCase(100, 200, 0)],
                learner_giocatore_id=4,
            )


if __name__ == "__main__":
    unittest.main()
