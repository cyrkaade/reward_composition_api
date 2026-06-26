from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym


@dataclass(frozen=True)
class AtariRewardStep:
    partial: float
    life_loss_penalty: float
    score_partial: float
    lost_lives: float
    lives: float

    def as_info(self) -> dict[str, float]:
        return {
            "partial_reward": self.partial,
            "life_loss_penalty": self.life_loss_penalty,
            "score_partial": self.score_partial,
            "lost_lives": self.lost_lives,
            "lives": self.lives,
        }


@dataclass(frozen=True)
class AtariRewardSpec:
    env_id: str
    slug: str
    life_loss_penalty: float
    component_keys: tuple[str, ...] = ("life_loss_penalty", "score_partial", "lost_lives", "lives")

    def new_tracker(self) -> "AtariPartialRewardTracker":
        return AtariPartialRewardTracker(self)

    def zero_step(self, info: dict | None = None) -> AtariRewardStep:
        lives = float((info or {}).get("lives", 0.0))
        return AtariRewardStep(partial=0.0, life_loss_penalty=0.0, score_partial=0.0, lost_lives=0.0, lives=lives)


class AtariPartialRewardTracker:
    def __init__(self, spec: AtariRewardSpec):
        self.spec = spec
        self.prev_lives: int | None = None

    def reset(self, info: dict | None = None) -> AtariRewardStep:
        self.prev_lives = self._lives(info)
        return self.spec.zero_step(info)

    def step(self, info: dict, true_reward: float = 0.0, partial_source: str = "life_loss") -> AtariRewardStep:
        lives = self._lives(info)
        if self.prev_lives is None:
            lost_lives = 0
        else:
            lost_lives = max(self.prev_lives - lives, 0)
        self.prev_lives = lives

        life_loss_penalty = -self.spec.life_loss_penalty * float(lost_lives)
        if partial_source == "life_loss":
            score_partial = 0.0
        elif partial_source == "clipped_score_life_loss":
            score_partial = float(max(0.0, min(float(true_reward), 1.0)))
        elif partial_source in {"score", "score_life_loss"}:
            score_partial = float(true_reward)
        else:
            raise ValueError(f"Unsupported Atari partial source: {partial_source}")

        if partial_source == "score":
            life_loss_penalty = 0.0

        return AtariRewardStep(
            partial=life_loss_penalty + score_partial,
            life_loss_penalty=life_loss_penalty,
            score_partial=score_partial,
            lost_lives=float(lost_lives),
            lives=float(lives),
        )

    @staticmethod
    def _lives(info: dict | None) -> int:
        if not info:
            return 0
        return int(info.get("lives", 0))


ATARI_REWARD_SPECS: dict[str, AtariRewardSpec] = {
    "ALE/Breakout-v5": AtariRewardSpec(
        env_id="ALE/Breakout-v5",
        slug="breakout",
        life_loss_penalty=1.0,
    ),
    "ALE/Seaquest-v5": AtariRewardSpec(
        env_id="ALE/Seaquest-v5",
        slug="seaquest",
        life_loss_penalty=5.0,
    ),
    "ALE/Qbert-v5": AtariRewardSpec(
        env_id="ALE/Qbert-v5",
        slug="qbert",
        life_loss_penalty=5.0,
    ),
    "ALE/SpaceInvaders-v5": AtariRewardSpec(
        env_id="ALE/SpaceInvaders-v5",
        slug="spaceinvaders",
        life_loss_penalty=2.0,
    ),
}


def get_atari_reward_spec(env_id: str) -> AtariRewardSpec:
    if env_id in ATARI_REWARD_SPECS:
        return ATARI_REWARD_SPECS[env_id]
    if env_id in supported_atari_envs():
        return AtariRewardSpec(env_id=env_id, slug=_slugify_atari(env_id), life_loss_penalty=1.0)
    supported = ", ".join(sorted(supported_atari_envs()))
    raise ValueError(f"Unsupported Atari env '{env_id}'. Supported envs: {supported}")


def supported_atari_envs() -> Iterable[str]:
    envs = set(ATARI_REWARD_SPECS)
    _try_register_atari_envs()
    envs.update(env_id for env_id in gym.envs.registry.keys() if env_id.startswith("ALE/") and env_id.endswith("-v5"))
    return tuple(sorted(envs))


def _try_register_atari_envs() -> None:
    try:
        import ale_py
    except ImportError:
        return
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)


def _slugify_atari(env_id: str) -> str:
    name = env_id.split("/", 1)[-1].rsplit("-", 1)[0]
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
