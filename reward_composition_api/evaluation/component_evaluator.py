from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reward_composition_api.evaluation.components import summarize_component_rows
from reward_composition_api.registry import PartialSpec


ModelObservationFn = Callable[[Any, Any], Any]
ActionConverterFn = Callable[[Any, Any], Any]
PartialStepFn = Callable[[Any, Any, Any, float, bool, bool, dict], tuple[float, dict[str, float]]]
ResetRewardStateFn = Callable[[dict], None]


def zero_partial_step(previous_obs, action, observation, true_reward, terminated, truncated, info) -> tuple[float, dict[str, float]]:
    return 0.0, {}


def evaluate_policy_components(
    *,
    model,
    env_id: str,
    make_env,
    custom_partial: PartialSpec | None,
    stats_source,
    n_eval_episodes: int,
    seed: int,
    deterministic: bool,
    component_keys: tuple[str, ...],
    model_observation: ModelObservationFn,
    action_converter: ActionConverterFn,
    default_partial_step: PartialStepFn = zero_partial_step,
    reset_reward_state: ResetRewardStateFn | None = None,
) -> dict[str, float]:
    rows = []
    env = make_env(env_id)
    partial = custom_partial.create(env_id) if custom_partial else None
    summary_keys = ["total", "partial", "residual", *component_keys, "length"]

    try:
        for episode_index in range(n_eval_episodes):
            obs, info = env.reset(seed=seed + episode_index)
            if reset_reward_state is not None:
                reset_reward_state(info)
            if partial is not None:
                partial.reset(info)
            done = False
            total = 0.0
            partial_total = 0.0
            length = 0
            components = {key: 0.0 for key in component_keys}

            while not done:
                model_obs = model_observation(stats_source, obs)
                action, _ = model.predict(model_obs, deterministic=deterministic)
                env_action = action_converter(env, action)
                new_obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                total += float(reward)
                length += 1

                if partial is None:
                    step_partial, step_components = default_partial_step(
                        obs,
                        env_action,
                        new_obs,
                        float(reward),
                        terminated,
                        truncated,
                        info,
                    )
                else:
                    partial_step = partial.step(obs, env_action, new_obs, reward, terminated, truncated, info)
                    step_partial = partial_step.partial
                    step_components = partial_step.components
                partial_total += float(step_partial)

                for key, value in step_components.items():
                    if key in components:
                        components[key] += float(value)
                obs = new_obs

            residual = total - partial_total
            rows.append(
                {
                    "total": total,
                    "partial": partial_total,
                    "residual": residual,
                    "length": float(length),
                    **components,
                }
            )
    finally:
        env.close()

    return summarize_component_rows(rows, summary_keys)
