"""Manual imperfect partial reward for Space Invaders.

This keeps sparse score events but clips their magnitude, and adds a small
penalty when the agent loses a life. It is intentionally weaker than the
full environment reward.
"""

from __future__ import annotations

import numpy as np

from reward_composition_api.registry import PartialRewardStep


class SpaceInvadersManualPartial:
    component_keys = ("score_reward", "score_event", "life_loss_penalty", "lost_lives")

    def __init__(self, life_loss_penalty: float = 2.0):
        self.life_loss_penalty = float(life_loss_penalty)
        self.previous_lives: int | None = None

    def reset(self, info: dict | None = None) -> None:
        self.previous_lives = self._lives(info)

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        lives = self._lives(info)
        lost_lives = 0
        if lives is not None and self.previous_lives is not None:
            lost_lives = max(self.previous_lives - lives, 0)
        self.previous_lives = lives

        score_reward = float(true_reward)
        score_event = float(np.clip(score_reward, 0.0, 1.0))
        life_penalty = -self.life_loss_penalty * float(lost_lives)
        return PartialRewardStep(
            partial=score_event + life_penalty,
            components={
                "score_reward": score_reward,
                "score_event": score_event,
                "life_loss_penalty": life_penalty,
                "lost_lives": float(lost_lives),
            },
        )

    def _lives(self, info: dict | None) -> int | None:
        if not info or "lives" not in info:
            return None
        return int(info["lives"])


def register(registry) -> None:
    registry.register(
        name="spaceinvaders_manual_partial",
        suite="atari",
        factory=lambda env_id: SpaceInvadersManualPartial(life_loss_penalty=2.0),
        description="SpaceInvaders partial: clipped positive score event plus -2 per lost life.",
        env_ids=("ALE/SpaceInvaders-v5", "ALE/SpaceInvaders-ram-v5"),
        component_keys=SpaceInvadersManualPartial.component_keys,
    )
