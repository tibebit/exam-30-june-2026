from __future__ import annotations

import random
import unittest

import numpy as np

from game.cards import Carta
from game.observation import Osservazione
from policy import BriscolaFeatureExtractor, NeuralSoftmaxPolicy
from training.episode import EpisodeResult, TrajectoryStep
from training.neural_reinforce import NeuralValueBaseline, neural_reinforce_update
from training.reinforce import ReinforceConfig


def osservazione_con_mano() -> Osservazione:
    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=(Carta("coppe", "asso"), Carta("bastoni", "due")),
        mano_compagno_visibile=False,
        mano_compagno=(),
        seme_briscola="denari",
        briscola_esposta=Carta("denari", "asso"),
        proprietario_briscola_esposta=None,
        carte_sul_campo=(),
        carte_giocate=(),
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=0,
        punteggio_avversari=0,
        primo_giocatore_presa=0,
        giocatore_corrente=0,
        carte_nel_mazzo=28,
        indice_presa=0,
        posizione_nella_presa=0,
    )


def episodio(reward_to_go: float) -> EpisodeResult:
    obs = osservazione_con_mano()
    return EpisodeResult(
        steps=[
            TrajectoryStep(
                osservazione=obs,
                azione=obs.azioni_legali[0],
                global_step_index=0,
                reward_to_go=reward_to_go,
            )
        ],
        rewards=[],
        punteggi_finali={"pari": 70, "dispari": 50},
        learner_giocatore_id=0,
        learner_squadra="pari",
        episode_return=reward_to_go,
    )


def flat_parameters(module) -> np.ndarray:
    return np.concatenate(
        [
            parameter.detach().cpu().numpy().reshape(-1)
            for parameter in module.parameters()
        ]
    ).astype(np.float32)


class TestNeuralReinforceUpdate(unittest.TestCase):
    def test_update_modifica_parametri_con_optimizer_pytorch(self):
        # The neural path uses autograd and an optimizer instead of manual gradients.
        policy = NeuralSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(0),
            hidden_size=4,
        )
        before = policy.theta.copy()

        stats = neural_reinforce_update(
            policy,
            [episodio(2.0)],
            ReinforceConfig(learning_rate=0.01, baseline="none"),
        )

        self.assertEqual(stats.episodes, 1)
        self.assertEqual(stats.learner_decisions, 1)
        self.assertGreater(stats.gradient_norm, 0.0)
        self.assertIsNotNone(stats.mean_entropy)
        self.assertGreater(stats.mean_entropy, 0.0)
        self.assertFalse(np.allclose(policy.theta, before))

    def test_learning_rate_zero_non_modifica_parametri(self):
        # A zero learning rate still computes statistics without moving weights.
        policy = NeuralSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(1),
            hidden_size=4,
        )
        before = policy.theta.copy()

        stats = neural_reinforce_update(
            policy,
            [episodio(2.0)],
            ReinforceConfig(learning_rate=0.0, baseline="none"),
        )

        self.assertGreater(stats.gradient_norm, 0.0)
        self.assertTrue(np.allclose(policy.theta, before))

    def test_entropy_coef_puo_muovere_parametri_senza_advantage(self):
        # The entropy bonus is part of the neural loss only when its coefficient is positive.
        extractor = BriscolaFeatureExtractor()
        without_entropy = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(2),
            hidden_size=4,
        )
        with_entropy = without_entropy.copy()
        before_without = without_entropy.theta.copy()
        before_with = with_entropy.theta.copy()

        neural_reinforce_update(
            without_entropy,
            [episodio(0.0)],
            ReinforceConfig(learning_rate=0.01, baseline="none", entropy_coef=0.0),
        )
        neural_reinforce_update(
            with_entropy,
            [episodio(0.0)],
            ReinforceConfig(learning_rate=0.01, baseline="none", entropy_coef=0.01),
        )

        self.assertTrue(np.allclose(without_entropy.theta, before_without))
        self.assertFalse(np.allclose(with_entropy.theta, before_with))

    def test_learned_value_baseline_viene_aggiornata(self):
        # The learned baseline is trained as a value estimator, separate from the policy.
        extractor = BriscolaFeatureExtractor()
        policy = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(3),
            hidden_size=4,
        )
        value_baseline = NeuralValueBaseline.initialize(
            extractor,
            rng=random.Random(4),
            hidden_size=4,
        )
        before_value = flat_parameters(value_baseline)

        stats = neural_reinforce_update(
            policy,
            [episodio(2.0)],
            ReinforceConfig(learning_rate=0.01, baseline="none"),
            value_baseline=value_baseline,
        )

        self.assertEqual(stats.baseline, "learned_value")
        self.assertEqual(stats.baseline_values, ())
        self.assertIsNotNone(stats.mean_value_loss)
        self.assertGreater(stats.mean_value_loss, 0.0)
        self.assertFalse(np.allclose(flat_parameters(value_baseline), before_value))

    def test_episodi_vuoti_solleva_value_error(self):
        # Neural updates need at least one collected episode.
        policy = NeuralSoftmaxPolicy.initialize(BriscolaFeatureExtractor())

        with self.assertRaises(ValueError):
            neural_reinforce_update(policy, [])


if __name__ == "__main__":
    unittest.main()
