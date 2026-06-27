from __future__ import annotations

import random
import unittest
from dataclasses import dataclass

from training.warm_start import WarmStartPolicySchedule


@dataclass
class FakePolicy:
    name: str


def make_fake_policy() -> FakePolicy:
    return FakePolicy("fake")


class TestWarmStartPolicySchedule(unittest.TestCase):
    def test_default_non_attiva_warm_start(self):
        # The default keeps normal self-play unchanged unless the CLI enables it.
        schedule = WarmStartPolicySchedule()

        self.assertEqual(schedule.warm_start_updates, 0)
        self.assertFalse(schedule.active(0))

    def test_active_rispetta_numero_update_configurato(self):
        # Warm-start applies only before the configured cutoff update.
        schedule = WarmStartPolicySchedule(warm_start_updates=3)

        self.assertTrue(schedule.active(0))
        self.assertTrue(schedule.active(2))
        self.assertFalse(schedule.active(3))

    def test_config_rifiuta_valori_non_validi(self):
        # Invalid schedules fail before training starts.
        with self.assertRaises(ValueError):
            WarmStartPolicySchedule(warm_start_updates=-1)

        with self.assertRaises(ValueError):
            WarmStartPolicySchedule(policy_factories=())

    def test_sample_policy_crea_istanza_dalla_factory(self):
        # Sampling returns a fresh policy instance from the configured factories.
        schedule = WarmStartPolicySchedule(
            warm_start_updates=1,
            policy_factories=(make_fake_policy,),
        )

        policy = schedule.sample_policy(random.Random(0))

        self.assertIsInstance(policy, FakePolicy)
        self.assertEqual(policy.name, "fake")


if __name__ == "__main__":
    unittest.main()
