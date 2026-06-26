"""A true-like shaped partial reward for Atari Pac-Man."""

from __future__ import annotations

from reward_composition_api.registry import PartialRewardStep


class PacmanScoreLifePartial:
    """Use the game score reward plus a penalty whenever Pac-Man loses a life."""

    def __init__(self, life_loss_penalty: float = 1.0):
        self.life_loss_penalty = float(life_loss_penalty)
        self.previous_lives: int | None = None

    def reset(self, info: dict | None = None) -> None:
        self.previous_lives = int((info or {}).get("lives", 0))

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        lives = int((info or {}).get("lives", 0))
        lost_lives = 0 if self.previous_lives is None else max(self.previous_lives - lives, 0)
        self.previous_lives = lives

        score_reward = float(true_reward)
        life_loss_penalty = -self.life_loss_penalty * float(lost_lives)
        return PartialRewardStep(
            partial=score_reward + life_loss_penalty,
            components={
                "score_reward": score_reward,
                "life_loss_penalty": life_loss_penalty,
                "lost_lives": float(lost_lives),
            },
        )


def register(registry) -> None:
    registry.register(
        name="pacman_score_life",
        suite="atari",
        factory=lambda env_id: PacmanScoreLifePartial(life_loss_penalty=1.0),
        description="Pac-Man score reward plus a -1 penalty for each life lost.",
        env_ids=("ALE/Pacman-v5",),
        component_keys=("score_reward", "life_loss_penalty", "lost_lives"),
    )
