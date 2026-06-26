from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym


@dataclass(frozen=True)
class MuJoCoRewardSpec:
    env_id: str
    slug: str
    partial_keys: tuple[str, ...]
    component_keys: tuple[str, ...]
    partial_weights: tuple[float, ...] | None = None

    def partial_reward(self, info: dict) -> float:
        weights = self.partial_weights or tuple(1.0 for _ in self.partial_keys)
        return float(sum(float(info.get(key, 0.0)) * weight for key, weight in zip(self.partial_keys, weights)))

    def components(self, info: dict) -> dict[str, float]:
        values = {key: float(info.get(key, 0.0)) for key in self.component_keys}
        values["partial"] = self.partial_reward(info)
        return values

    def with_partial_profile(self, profile: str) -> "MuJoCoRewardSpec":
        if profile == "default":
            return self
        if profile == "ctrl_half":
            if "reward_ctrl" not in self.component_keys:
                raise ValueError(f"Partial profile '{profile}' requires reward_ctrl in {self.env_id}")
            keys = tuple(dict.fromkeys((*self.partial_keys, "reward_ctrl")))
            weights = tuple(0.5 if key == "reward_ctrl" else 1.0 for key in keys)
            return MuJoCoRewardSpec(
                env_id=self.env_id,
                slug=f"{self.slug}_ctrlhalf",
                partial_keys=keys,
                partial_weights=weights,
                component_keys=self.component_keys,
            )
        if profile == "true_like":
            return MuJoCoRewardSpec(
                env_id=self.env_id,
                slug=f"{self.slug}_truelike",
                partial_keys=self.component_keys,
                partial_weights=tuple(1.0 for _ in self.component_keys),
                component_keys=self.component_keys,
            )
        raise ValueError(f"Unsupported MuJoCo partial profile: {profile}")


MUJOCO_REWARD_SPECS: dict[str, MuJoCoRewardSpec] = {
    "Reacher-v5": MuJoCoRewardSpec(
        env_id="Reacher-v5",
        slug="reacher",
        partial_keys=("reward_dist",),
        component_keys=("reward_dist", "reward_ctrl"),
    ),
    "HalfCheetah-v5": MuJoCoRewardSpec(
        env_id="HalfCheetah-v5",
        slug="halfcheetah",
        partial_keys=("reward_forward",),
        component_keys=("reward_forward", "reward_ctrl"),
    ),
    "Hopper-v5": MuJoCoRewardSpec(
        env_id="Hopper-v5",
        slug="hopper",
        partial_keys=("reward_forward", "reward_survive"),
        component_keys=("reward_forward", "reward_survive", "reward_ctrl"),
    ),
    "Walker2d-v5": MuJoCoRewardSpec(
        env_id="Walker2d-v5",
        slug="walker2d",
        partial_keys=("reward_forward", "reward_survive"),
        component_keys=("reward_forward", "reward_survive", "reward_ctrl"),
    ),
}


def get_mujoco_reward_spec(env_id: str) -> MuJoCoRewardSpec:
    if env_id in MUJOCO_REWARD_SPECS:
        return MUJOCO_REWARD_SPECS[env_id]
    if env_id in supported_mujoco_envs():
        return MuJoCoRewardSpec(env_id=env_id, slug=_slugify(env_id), partial_keys=(), component_keys=())
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
