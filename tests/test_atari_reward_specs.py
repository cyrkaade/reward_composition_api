from __future__ import annotations

import unittest

from local_gym.classes.atari_reward_specs import get_atari_reward_spec


class AtariPartialRewardTrackerTest(unittest.TestCase):
    def test_life_loss_penalty_tracks_decreases(self):
        spec = get_atari_reward_spec("ALE/Breakout-v5")
        tracker = spec.new_tracker()

        initial = tracker.reset({"lives": 5})
        self.assertEqual(initial.partial, 0.0)

        self.assertEqual(tracker.step({"lives": 5}).partial, 0.0)
        lost_one = tracker.step({"lives": 4})
        self.assertEqual(lost_one.lost_lives, 1.0)
        self.assertEqual(lost_one.partial, -1.0)

        lost_two = tracker.step({"lives": 2})
        self.assertEqual(lost_two.lost_lives, 2.0)
        self.assertEqual(lost_two.partial, -2.0)

    def test_reset_clears_previous_episode_state(self):
        spec = get_atari_reward_spec("ALE/Seaquest-v5")
        tracker = spec.new_tracker()

        tracker.reset({"lives": 4})
        self.assertEqual(tracker.step({"lives": 3}).partial, -5.0)

        tracker.reset({"lives": 4})
        self.assertEqual(tracker.step({"lives": 4}).partial, 0.0)

    def test_bonus_life_or_nondecreasing_lives_is_not_penalized(self):
        spec = get_atari_reward_spec("ALE/Qbert-v5")
        tracker = spec.new_tracker()

        tracker.reset({"lives": 2})
        bonus_life = tracker.step({"lives": 3})
        self.assertEqual(bonus_life.lost_lives, 0.0)
        self.assertEqual(bonus_life.partial, 0.0)

        stable = tracker.step({"lives": 3})
        self.assertEqual(stable.lost_lives, 0.0)
        self.assertEqual(stable.partial, 0.0)

    def test_clipped_score_life_loss_partial_is_deterministic(self):
        spec = get_atari_reward_spec("ALE/Seaquest-v5")

        first = spec.new_tracker()
        first.reset({"lives": 4})
        first_step = first.step({"lives": 3}, true_reward=20.0, partial_source="clipped_score_life_loss")

        second = spec.new_tracker()
        second.reset({"lives": 4})
        second_step = second.step({"lives": 3}, true_reward=20.0, partial_source="clipped_score_life_loss")

        self.assertEqual(first_step.partial, -4.0)
        self.assertEqual(first_step.life_loss_penalty, -5.0)
        self.assertEqual(first_step.score_partial, 1.0)
        self.assertEqual(first_step, second_step)

    def test_score_partial_matches_true_reward_without_life_penalty(self):
        spec = get_atari_reward_spec("ALE/Seaquest-v5")
        tracker = spec.new_tracker()

        tracker.reset({"lives": 4})
        step = tracker.step({"lives": 3}, true_reward=20.0, partial_source="score")

        self.assertEqual(step.partial, 20.0)
        self.assertEqual(step.score_partial, 20.0)
        self.assertEqual(step.life_loss_penalty, 0.0)
        self.assertEqual(step.lost_lives, 1.0)

    def test_score_life_loss_partial_keeps_true_reward_and_life_penalty(self):
        spec = get_atari_reward_spec("ALE/Qbert-v5")
        tracker = spec.new_tracker()

        tracker.reset({"lives": 4})
        step = tracker.step({"lives": 3}, true_reward=25.0, partial_source="score_life_loss")

        self.assertEqual(step.partial, 20.0)
        self.assertEqual(step.score_partial, 25.0)
        self.assertEqual(step.life_loss_penalty, -5.0)
        self.assertEqual(step.lost_lives, 1.0)


if __name__ == "__main__":
    unittest.main()
