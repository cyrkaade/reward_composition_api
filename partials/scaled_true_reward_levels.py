"""Controlled 0.4/0.6/0.8 partial-reward levels for parity studies.

For a partial reward ``P = alpha * R``, the project's partiality estimator
``Cov(R, P) / Var(R)`` is exactly ``alpha`` (up to floating-point error).
These partials intentionally control reward magnitude; they do not remove
information, so their preference ranking correlation with true reward is 1.
"""

from __future__ import annotations


def partial_04(obs, action, next_obs, true_reward, terminated, truncated, info):
    reward = float(true_reward)
    partial = 0.4 * reward
    return {
        "partial": partial,
        "components": {
            "scaled_true_reward": partial,
            "omitted_true_reward": reward - partial,
        },
    }


def partial_06(obs, action, next_obs, true_reward, terminated, truncated, info):
    reward = float(true_reward)
    partial = 0.6 * reward
    return {
        "partial": partial,
        "components": {
            "scaled_true_reward": partial,
            "omitted_true_reward": reward - partial,
        },
    }


def partial_08(obs, action, next_obs, true_reward, terminated, truncated, info):
    reward = float(true_reward)
    partial = 0.8 * reward
    return {
        "partial": partial,
        "components": {
            "scaled_true_reward": partial,
            "omitted_true_reward": reward - partial,
        },
    }


def register(registry) -> None:
    levels = (
        ("scaled_true_04", partial_04, "0.4"),
        ("scaled_true_06", partial_06, "0.6"),
        ("scaled_true_08", partial_08, "0.8"),
    )
    suites = {
        "mujoco": ("Reacher-v5",),
        "atari": ("ALE/SpaceInvaders-v5",),
    }
    component_keys = ("scaled_true_reward", "omitted_true_reward")

    for suite, env_ids in suites.items():
        for name, reward_fn, level in levels:
            registry.register(
                name=name,
                suite=suite,
                factory=lambda env_id, fn=reward_fn: fn,
                description=f"Controlled {level} * true-reward partial.",
                env_ids=env_ids,
                component_keys=component_keys,
            )
