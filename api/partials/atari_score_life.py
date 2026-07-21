"""Generic imperfect partial reward for Atari games with lives.

Keeps sparse positive score events but clips their magnitude, and adds a
small penalty when the agent loses a life. Intentionally weaker than the
full environment reward. Works for any Atari game that reports a positive
score and a 'lives' count (Breakout, SpaceInvaders, Seaquest, Qbert, ...).
"""

from __future__ import annotations

import numpy as np


class AtariScoreLifePartial:
    component_keys = ("score_reward", "score_event", "life_loss_penalty", "lost_lives")

    def __init__(self, life_loss_penalty: float = 2.0):
        self.life_loss_penalty = float(life_loss_penalty)
        self.previous_lives: int | None = None

    def reset(self, info: dict | None = None) -> None:
        self.previous_lives = self._lives(info)

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        lives = self._lives(info)
        lost_lives = 0
        if lives is not None and self.previous_lives is not None:
            lost_lives = max(self.previous_lives - lives, 0)
        self.previous_lives = lives

        score_reward = float(true_reward)
        score_event = float(np.clip(score_reward, 0.0, 1.0))
        life_penalty = -self.life_loss_penalty * float(lost_lives)
        return {
            "partial": score_event + life_penalty,
            "components": {
                "score_reward": score_reward,
                "score_event": score_event,
                "life_loss_penalty": life_penalty,
                "lost_lives": float(lost_lives),
            },
        }

    def _lives(self, info: dict | None) -> int | None:
        if not info or "lives" not in info:
            return None
        return int(info["lives"])


def register(registry) -> None:
    registry.register(
        name="atari_score_life",
        suite="atari",
        factory=lambda env_id: AtariScoreLifePartial(life_loss_penalty=2.0),
        description="Generic Atari partial: clipped positive score event plus -2 per lost life.",
        env_ids=(
            "ALE/Breakout-v5",
            "ALE/SpaceInvaders-v5",
            "ALE/Seaquest-v5",
            "ALE/Qbert-v5",
            "ALE/MsPacman-v5",
        ),
        component_keys=AtariScoreLifePartial.component_keys,
    )
