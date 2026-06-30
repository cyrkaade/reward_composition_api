from __future__ import annotations

from local_gym.classes.atari_reward_specs import get_atari_reward_spec, supported_atari_envs
from local_gym.classes.mujoco_reward_specs import get_mujoco_reward_spec, supported_mujoco_envs

from reward_composition_api.config import ATARI_SUITE, MUJOCO_SUITE
from reward_composition_api.registry import PartialRegistry, PartialRewardStep


class MuJoCoComponentPartial:
    def __init__(self, env_id: str, profile: str):
        self.spec = get_mujoco_reward_spec(env_id).with_partial_profile(profile)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        components = self.spec.components(info)
        partial = components.pop("partial")
        return PartialRewardStep(partial=partial, components=components)


class AtariSourcePartial:
    def __init__(self, env_id: str, source: str):
        self.spec = get_atari_reward_spec(env_id)
        self.source = source
        self.tracker = self.spec.new_tracker()

    def reset(self, info: dict | None = None) -> None:
        self.tracker.reset(info)

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        step = self.tracker.step(info, true_reward=float(true_reward), partial_source=self.source)
        return PartialRewardStep(
            partial=step.partial,
            components={
                "life_loss_penalty": step.life_loss_penalty,
                "score_partial": step.score_partial,
                "lost_lives": step.lost_lives,
                "lives": step.lives,
            },
        )


def build_builtin_registry() -> PartialRegistry:
    registry = PartialRegistry()
    mujoco_envs = tuple(supported_mujoco_envs())
    atari_envs = tuple(supported_atari_envs())

    registry.register(
        "default",
        MUJOCO_SUITE,
        lambda env_id: MuJoCoComponentPartial(env_id, "default"),
        description="Default MuJoCo component-key partial used by current experiments.",
        env_ids=mujoco_envs,
    )
    registry.register(
        "ctrl_half",
        MUJOCO_SUITE,
        lambda env_id: MuJoCoComponentPartial(env_id, "ctrl_half"),
        description="MuJoCo default partial plus half-weight control cost.",
        env_ids=mujoco_envs,
    )
    registry.register(
        "true_like",
        MUJOCO_SUITE,
        lambda env_id: MuJoCoComponentPartial(env_id, "true_like"),
        description="MuJoCo partial that sums all known true reward components.",
        env_ids=mujoco_envs,
    )

    for source in ("life_loss", "clipped_score_life_loss", "score", "score_life_loss"):
        registry.register(
            source,
            ATARI_SUITE,
            lambda env_id, source=source: AtariSourcePartial(env_id, source),
            description=f"Atari partial source '{source}'.",
            env_ids=atari_envs,
            component_keys=("life_loss_penalty", "score_partial", "lost_lives", "lives"),
        )

    return registry


def partials_for_display(suite: str | None = None) -> list[dict[str, str]]:
    registry = build_builtin_registry()
    return [
        {
            "suite": spec.suite,
            "name": spec.name,
            "envs": ", ".join(spec.env_ids),
            "description": spec.description,
        }
        for spec in registry.list(suite)
    ]
