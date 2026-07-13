from __future__ import annotations

import unittest

from local_gym.classes.atari_reward_specs import get_atari_reward_spec


class AtariRewardSpecTest(unittest.TestCase):
    def test_known_atari_spec_is_environment_metadata_only(self):
        spec = get_atari_reward_spec("ALE/Breakout-v5")

        self.assertEqual(spec.env_id, "ALE/Breakout-v5")
        self.assertEqual(spec.slug, "breakout")
        self.assertFalse(hasattr(spec, "new_tracker"))

    def test_another_known_atari_spec_gets_slug(self):
        spec = get_atari_reward_spec("ALE/SpaceInvaders-v5")

        self.assertEqual(spec.env_id, "ALE/SpaceInvaders-v5")
        self.assertEqual(spec.slug, "spaceinvaders")


if __name__ == "__main__":
    unittest.main()
