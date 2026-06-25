from __future__ import annotations

import random
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

import numpy as np

from policy import BriscolaFeatureExtractor, NeuralSoftmaxPolicy
from training.bootstrap import BootstrapPolicySchedule
from training.episode import EpisodeResult
from training.reinforce import ReinforceConfig, TrainStats
from training.rewards import RewardConfig
from training.self_play import SelfPlayConfig, SelfPlayTrainer


@dataclass
class FakePolicy:
    name: str
    theta: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0], dtype=np.float32)
    )


@dataclass
class FakePool:
    snapshots: list[FakePolicy] = field(default_factory=list)
    shared_sample: FakePolicy | None = None
    sample_calls: int = 0
    added: list[tuple[str, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.snapshots)

    def add_policy(
        self,
        policy: FakePolicy,
        name: str,
        update_index: int,
    ) -> None:
        self.added.append((name, update_index))
        self.snapshots.append(FakePolicy(name=name, theta=policy.theta.copy()))

    def sample_policy(self, rng: random.Random) -> FakePolicy:
        self.sample_calls += 1
        if self.shared_sample is not None:
            return self.shared_sample
        return FakePolicy(name=f"sample_{self.sample_calls}")


@dataclass
class FakeBootstrapSchedule:
    bootstrap_updates: int = 30
    shared_sample: FakePolicy | None = field(
        default_factory=lambda: FakePolicy("bootstrap")
    )
    sample_calls: int = 0

    def active(self, update_index: int) -> bool:
        return update_index < self.bootstrap_updates

    def sample_policy(self, rng: random.Random) -> FakePolicy:
        self.sample_calls += 1
        if self.shared_sample is not None:
            return self.shared_sample
        return FakePolicy(name=f"bootstrap_{self.sample_calls}")


def episodio_finto() -> EpisodeResult:
    return EpisodeResult(
        steps=[],
        rewards=[],
        punteggi_finali={"pari": 60, "dispari": 60},
        learner_giocatore_id=0,
        learner_squadra="pari",
        episode_return=0.0,
    )


def stats_finte(episodes: int) -> TrainStats:
    return TrainStats(
        episodes=episodes,
        learner_decisions=episodes * 10,
        mean_return=0.0,
        mean_score_margin=0.0,
        gradient_norm=0.0,
        baseline="none",
        baseline_values=(),
    )


class TestSelfPlayConfig(unittest.TestCase):
    def test_config_default_e_valori_validi(self):
        # Defaults describe a configurable loop, but not a complete runner.
        config = SelfPlayConfig()

        self.assertEqual(config.batch_size, 500)
        self.assertEqual(config.snapshot_interval, 50)
        self.assertEqual(config.learner_giocatore_id, 0)
        self.assertIsInstance(config.reward_config, RewardConfig)
        self.assertIsInstance(config.reinforce_config, ReinforceConfig)
        self.assertIsInstance(config.bootstrap_schedule, BootstrapPolicySchedule)
        self.assertEqual(config.bootstrap_schedule.bootstrap_updates, 0)
        self.assertFalse(config.greedy_non_learner)
        self.assertEqual(config.matchup_sampling, "per_episode")
        self.assertIsNone(config.neural_learned_baseline)

    def test_config_rifiuta_valori_illegali(self):
        # Fail fast on batch, snapshot interval, and player id.
        with self.assertRaises(ValueError):
            SelfPlayConfig(batch_size=0)

        with self.assertRaises(ValueError):
            SelfPlayConfig(batch_size=6)

        with self.assertRaises(ValueError):
            SelfPlayConfig(snapshot_interval=0)

        with self.assertRaises(ValueError):
            SelfPlayConfig(learner_giocatore_id=4)

        with self.assertRaises(ValueError):
            SelfPlayConfig(matchup_sampling="unknown")  # type: ignore[arg-type]


class TestSelfPlayTrainer(unittest.TestCase):
    def test_pool_vuoto_riceve_snapshot_initial(self):
        # The trainer guarantees a sampleable pool before the first episode.
        learner = FakePolicy("learner")
        pool = FakePool()

        SelfPlayTrainer(learner=learner, pool=pool)  # type: ignore[arg-type]

        self.assertEqual(pool.added, [("initial", 0)])
        self.assertEqual([snapshot.name for snapshot in pool.snapshots], ["initial"])

    def test_primo_giocatore_ruota_in_modo_bilanciato_nel_batch(self):
        # Rotation balances who opens the game before the single update.
        learner = FakePolicy("learner")
        pool = FakePool(snapshots=[FakePolicy("initial")])
        config = SelfPlayConfig(batch_size=8, snapshot_interval=99)

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()) as collect,
            patch(
                "training.self_play.reinforce_update",
                return_value=stats_finte(8),
            ) as reinforce,
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
                seed=123,
            )
            trainer.train_update()

        self.assertEqual(
            [
                call.kwargs["primo_giocatore_id"]
                for call in collect.call_args_list
            ],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )
        reinforce.assert_called_once()
        self.assertEqual(len(reinforce.call_args.args[1]), 8)

    def test_config_reward_reinforce_e_greedy_passano_ai_blocchi_giusti(self):
        # self_play forwards configs without reinterpreting them.
        learner = FakePolicy("learner")
        pool = FakePool(snapshots=[FakePolicy("initial")])
        reward_config = RewardConfig(mode="dense_presa", lambda_margin=0.4)
        reinforce_config = ReinforceConfig(learning_rate=0.2, baseline="none")
        config = SelfPlayConfig(
            batch_size=4,
            snapshot_interval=99,
            reward_config=reward_config,
            reinforce_config=reinforce_config,
            greedy_non_learner=True,
        )

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()) as collect,
            patch(
                "training.self_play.reinforce_update",
                return_value=stats_finte(4),
            ) as reinforce,
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
            )
            trainer.train_update()

        for call in collect.call_args_list:
            self.assertIs(call.kwargs["reward_config"], reward_config)
            self.assertTrue(call.kwargs["greedy_non_learner"])
        self.assertIs(reinforce.call_args.args[2], reinforce_config)

    def test_value_baseline_neurale_passa_solo_al_update_neural(self):
        # Learned value baselines are an optional neural-only training component.
        extractor = BriscolaFeatureExtractor()
        learner = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(0),
            hidden_size=4,
        )
        pool = FakePool(snapshots=[FakePolicy("initial")])
        config = SelfPlayConfig(
            batch_size=4,
            snapshot_interval=99,
            neural_learned_baseline=True,
        )

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()),
            patch(
                "training.self_play.neural_reinforce_update",
                return_value=stats_finte(4),
            ) as neural_update,
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
                seed=123,
            )
            trainer.train_update()

        self.assertIsNotNone(trainer.neural_value_baseline)
        self.assertIsNotNone(trainer.neural_value_optimizer)
        self.assertIs(
            neural_update.call_args.kwargs["value_baseline"],
            trainer.neural_value_baseline,
        )
        self.assertIs(
            neural_update.call_args.kwargs["value_optimizer"],
            trainer.neural_value_optimizer,
        )

    def test_value_baseline_neurale_default_auto(self):
        # Neural learners use the learned value baseline by default.
        extractor = BriscolaFeatureExtractor()
        learner = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(1),
            hidden_size=4,
        )
        pool = FakePool(snapshots=[FakePolicy("initial")])
        config = SelfPlayConfig(batch_size=4, snapshot_interval=99)

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()),
            patch(
                "training.self_play.neural_reinforce_update",
                return_value=stats_finte(4),
            ) as neural_update,
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
                seed=123,
            )
            trainer.train_update()

        self.assertIsNotNone(trainer.neural_value_baseline)
        self.assertIs(
            neural_update.call_args.kwargs["value_baseline"],
            trainer.neural_value_baseline,
        )

    def test_value_baseline_neurale_puo_essere_disattivata(self):
        # Explicit opt-out keeps ablations comparable to the previous neural path.
        extractor = BriscolaFeatureExtractor()
        learner = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(2),
            hidden_size=4,
        )
        pool = FakePool(snapshots=[FakePolicy("initial")])
        config = SelfPlayConfig(
            batch_size=4,
            snapshot_interval=99,
            neural_learned_baseline=False,
        )

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()),
            patch(
                "training.self_play.neural_reinforce_update",
                return_value=stats_finte(4),
            ) as neural_update,
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
                seed=123,
            )
            trainer.train_update()

        self.assertIsNone(trainer.neural_value_baseline)
        self.assertIsNone(neural_update.call_args.kwargs["value_baseline"])

    def test_snapshot_viene_aggiunto_solo_all_intervallo_configurato(self):
        # The pool grows only when update_index reaches snapshot_interval.
        learner = FakePolicy("learner")
        pool = FakePool()
        config = SelfPlayConfig(batch_size=4, snapshot_interval=2)

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()),
            patch("training.self_play.reinforce_update", return_value=stats_finte(4)),
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
            )
            first = trainer.train_update()
            second = trainer.train_update()

        self.assertFalse(first.snapshot_added)
        self.assertTrue(second.snapshot_added)
        self.assertEqual(pool.added, [("initial", 0), ("snapshot_2", 2)])
        self.assertEqual(second.pool_size, 2)

    def test_master_seed_rende_riproducibili_seed_ambiente_e_policy(self):
        # A single trainer seed reproduces both environment and policy sampling.
        def seed_trace(seed: int) -> list[tuple[int, float]]:
            learner = FakePolicy("learner")
            pool = FakePool(snapshots=[FakePolicy("initial")])
            config = SelfPlayConfig(batch_size=4, snapshot_interval=99)
            trace: list[tuple[int, float]] = []

            def collect_side_effect(**kwargs):
                trace.append(
                    (
                        kwargs["seed_ambiente"],
                        kwargs["rng_policy"].random(),
                    )
                )
                return episodio_finto()

            with (
                patch("training.self_play.collect_episode", side_effect=collect_side_effect),
                patch("training.self_play.reinforce_update", return_value=stats_finte(4)),
            ):
                trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                    learner=learner,
                    pool=pool,
                    config=config,
                    seed=seed,
                )
                trainer.train_update()

            return trace

        self.assertEqual(seed_trace(123), seed_trace(123))
        self.assertNotEqual(seed_trace(123), seed_trace(456))

    def test_pool_viene_campionato_tre_volte_per_episodio(self):
        # The three draws are separate, even if they can return the same snapshot.
        learner = FakePolicy("learner")
        shared_policy = FakePolicy("shared_snapshot")
        pool = FakePool(
            snapshots=[FakePolicy("initial")],
            shared_sample=shared_policy,
        )
        config = SelfPlayConfig(batch_size=4, snapshot_interval=99)

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()) as collect,
            patch("training.self_play.reinforce_update", return_value=stats_finte(4)),
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
            )
            trainer.train_update()

        self.assertEqual(pool.sample_calls, 12)
        for call in collect.call_args_list:
            self.assertIs(call.kwargs["compagno_policy"], shared_policy)
            self.assertIs(call.kwargs["avversario_successivo_policy"], shared_policy)
            self.assertIs(call.kwargs["avversario_precedente_policy"], shared_policy)

    def test_bootstrap_viene_usato_nei_primi_update(self):
        # Early updates can use fixed baseline policies instead of self-play snapshots.
        learner = FakePolicy("learner")
        pool = FakePool(snapshots=[FakePolicy("initial")])
        bootstrap_schedule = FakeBootstrapSchedule()
        config = SelfPlayConfig(
            batch_size=4,
            snapshot_interval=99,
            bootstrap_schedule=bootstrap_schedule,  # type: ignore[arg-type]
        )

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()) as collect,
            patch("training.self_play.reinforce_update", return_value=stats_finte(4)),
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
                update_index=29,
            )
            trainer.train_update()

        self.assertEqual(bootstrap_schedule.sample_calls, 12)
        self.assertEqual(pool.sample_calls, 0)
        for call in collect.call_args_list:
            self.assertIs(call.kwargs["compagno_policy"], bootstrap_schedule.shared_sample)
            self.assertIs(
                call.kwargs["avversario_successivo_policy"],
                bootstrap_schedule.shared_sample,
            )
            self.assertIs(
                call.kwargs["avversario_precedente_policy"],
                bootstrap_schedule.shared_sample,
            )

    def test_pool_viene_campionato_tre_volte_per_blocco_di_rotazione(self):
        # A rotation block keeps the matchup fixed while primo_giocatore_id rotates.
        learner = FakePolicy("learner")
        pool = FakePool(snapshots=[FakePolicy("initial")])
        config = SelfPlayConfig(
            batch_size=8,
            snapshot_interval=99,
            matchup_sampling="per_rotation_block",
        )

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()) as collect,
            patch("training.self_play.reinforce_update", return_value=stats_finte(8)),
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
            )
            trainer.train_update()

        self.assertEqual(pool.sample_calls, 6)
        self.assertEqual(
            [
                call.kwargs["primo_giocatore_id"]
                for call in collect.call_args_list
            ],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )

        first_block = collect.call_args_list[:4]
        second_block = collect.call_args_list[4:]
        for call in first_block:
            self.assertEqual(call.kwargs["compagno_policy"].name, "sample_1")
            self.assertEqual(
                call.kwargs["avversario_successivo_policy"].name,
                "sample_2",
            )
            self.assertEqual(
                call.kwargs["avversario_precedente_policy"].name,
                "sample_3",
            )
        for call in second_block:
            self.assertEqual(call.kwargs["compagno_policy"].name, "sample_4")
            self.assertEqual(
                call.kwargs["avversario_successivo_policy"].name,
                "sample_5",
            )
            self.assertEqual(
                call.kwargs["avversario_precedente_policy"].name,
                "sample_6",
            )

    def test_bootstrap_viene_campionato_per_blocco_di_rotazione(self):
        # Bootstrap and rotation blocks compose: one baseline trio is reused for four games.
        learner = FakePolicy("learner")
        pool = FakePool(snapshots=[FakePolicy("initial")])
        bootstrap_schedule = FakeBootstrapSchedule(shared_sample=None)
        config = SelfPlayConfig(
            batch_size=8,
            snapshot_interval=99,
            bootstrap_schedule=bootstrap_schedule,  # type: ignore[arg-type]
            matchup_sampling="per_rotation_block",
        )

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()) as collect,
            patch("training.self_play.reinforce_update", return_value=stats_finte(8)),
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
                update_index=29,
            )
            trainer.train_update()

        self.assertEqual(bootstrap_schedule.sample_calls, 6)
        self.assertEqual(pool.sample_calls, 0)
        self.assertEqual(
            [
                call.kwargs["primo_giocatore_id"]
                for call in collect.call_args_list
            ],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )

        first_block = collect.call_args_list[:4]
        second_block = collect.call_args_list[4:]
        for call in first_block:
            self.assertEqual(call.kwargs["compagno_policy"].name, "bootstrap_1")
            self.assertEqual(
                call.kwargs["avversario_successivo_policy"].name,
                "bootstrap_2",
            )
            self.assertEqual(
                call.kwargs["avversario_precedente_policy"].name,
                "bootstrap_3",
            )
        for call in second_block:
            self.assertEqual(call.kwargs["compagno_policy"].name, "bootstrap_4")
            self.assertEqual(
                call.kwargs["avversario_successivo_policy"].name,
                "bootstrap_5",
            )
            self.assertEqual(
                call.kwargs["avversario_precedente_policy"].name,
                "bootstrap_6",
            )

    def test_train_esegue_piu_update_e_rifiuta_valore_negativo(self):
        # train(updates) is only an explicit repetition of train_update.
        learner = FakePolicy("learner")
        pool = FakePool()
        config = SelfPlayConfig(batch_size=4, snapshot_interval=99)

        with (
            patch("training.self_play.collect_episode", return_value=episodio_finto()),
            patch("training.self_play.reinforce_update", return_value=stats_finte(4)),
        ):
            trainer = SelfPlayTrainer(  # type: ignore[arg-type]
                learner=learner,
                pool=pool,
                config=config,
            )
            stats = trainer.train(3)

            with self.assertRaises(ValueError):
                trainer.train(-1)

        self.assertEqual([stat.update_index for stat in stats], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
