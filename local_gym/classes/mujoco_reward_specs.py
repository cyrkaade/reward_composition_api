from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym


@dataclass(frozen=True)
class MuJoCoRewardSpec:
    env_id: str
    slug: str


MUJOCO_REWARD_SPECS: dict[str, MuJoCoRewardSpec] = {
    "Reacher-v5": MuJoCoRewardSpec(
        env_id="Reacher-v5",
        slug="reacher",
    ),
    "HalfCheetah-v5": MuJoCoRewardSpec(
        env_id="HalfCheetah-v5",
        slug="halfcheetah",
    ),
    "Hopper-v5": MuJoCoRewardSpec(
        env_id="Hopper-v5",
        slug="hopper",
    ),
    "Walker2d-v5": MuJoCoRewardSpec(
        env_id="Walker2d-v5",
        slug="walker2d",
    ),
}


def get_mujoco_reward_spec(env_id: str) -> MuJoCoRewardSpec:
    if env_id in MUJOCO_REWARD_SPECS:
        return MUJOCO_REWARD_SPECS[env_id]
    if env_id in supported_mujoco_envs():
        return MuJoCoRewardSpec(env_id=env_id, slug=_slugify(env_id))
    supported = ", ".join(sorted(supported_mujoco_envs()))
    raise ValueError(f"Unsupported MuJoCo env '{env_id}'. Supported envs: {supported}")


def supported_mujoco_envs() -> Iterable[str]:
    envs = set(MUJOCO_REWARD_SPECS)
    envs.update(env_id for env_id in gym.envs.registry.keys() if _looks_like_mujoco(env_id))
    return tuple(sorted(envs))


def _looks_like_mujoco(env_id: str) -> bool:
    names = (
        "Ant",
        "HalfCheetah",
        "Hopper",
        "Humanoid",
        "HumanoidStandup",
        "InvertedDoublePendulum",
        "InvertedPendulum",
        "Pusher",
        "Reacher",
        "Swimmer",
        "Walker2d",
    )
    return env_id.startswith(names)


def _slugify(env_id: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in env_id.rsplit("-", 1)[0]).strip("_")
