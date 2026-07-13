from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym


@dataclass(frozen=True)
class AtariRewardSpec:
    env_id: str
    slug: str


ATARI_REWARD_SPECS: dict[str, AtariRewardSpec] = {
    "ALE/Breakout-v5": AtariRewardSpec(
        env_id="ALE/Breakout-v5",
        slug="breakout",
    ),
    "ALE/Seaquest-v5": AtariRewardSpec(
        env_id="ALE/Seaquest-v5",
        slug="seaquest",
    ),
    "ALE/Qbert-v5": AtariRewardSpec(
        env_id="ALE/Qbert-v5",
        slug="qbert",
    ),
    "ALE/SpaceInvaders-v5": AtariRewardSpec(
        env_id="ALE/SpaceInvaders-v5",
        slug="spaceinvaders",
    ),
}


def get_atari_reward_spec(env_id: str) -> AtariRewardSpec:
    if env_id in ATARI_REWARD_SPECS:
        return ATARI_REWARD_SPECS[env_id]
    if env_id in supported_atari_envs():
        return AtariRewardSpec(env_id=env_id, slug=_slugify_atari(env_id))
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
