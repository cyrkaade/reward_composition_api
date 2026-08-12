"""Regression tests for the selection-grid priors in partials/sel_*.py.

These lock the three properties a later edit could silently break: that every
prior still resolves through the exact call the trainer makes (the `file:name`
form, which is what killed PRE4 block F), that the saturating priors actually
saturate, and that the over-priced-effort priors actually charge for effort.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rcomp.partials import PartialRegistry, PartialRegistryError, load_partial_reference

# (module, name, suite, env_id)
SELECTION_PARTIALS = [
    ("sel_lunarlander", name, "box2d", "LunarLander-v3")
    for name in ("sll_pad", "sll_pad_speed", "sll_pad_fuel", "sll_touchdown", "sll_upright")
] + [
    ("sel_reacher", name, "mujoco", "Reacher-v5")
    for name in ("srch_dist", "srch_dist_cap", "srch_x", "srch_quad", "srch_hit")
] + [
    ("sel_pusher", name, "mujoco", "Pusher-v5")
    for name in ("spsh_goal_reach", "spsh_reach", "spsh_goal_cap", "spsh_goal", "spsh_timid")
] + [
    ("sel_swimmer", name, "mujoco", "Swimmer-v5")
    for name in ("sswm_cap10", "sswm_cap18", "sswm_speed", "sswm_straight", "sswm_timid")
] + [
    ("sel_halfcheetah", name, "mujoco", "HalfCheetah-v5")
    for name in ("shc_cap10", "shc_cap20", "shc_forward", "shc_timid", "shc_level")
] + [
    ("sel_walker2d", name, "mujoco", "Walker2d-v5")
    for name in ("swk_cap05", "swk_cap15", "swk_stand", "swk_target", "swk_timid")
] + [
    ("sel_ant", name, "mujoco", "Ant-v5")
    for name in ("sant_cap03", "sant_cap10", "sant_radial", "sant_stand", "sant_timid")
] + [
    ("sel_humanoidstandup", name, "mujoco", "HumanoidStandup-v5")
    for name in ("shs_cap60", "shs_cap120", "shs_rise", "shs_crouch", "shs_timid")
]

OBS_DIM = {
    "LunarLander-v3": 8,
    "Reacher-v5": 10,
    "Pusher-v5": 23,
    "Swimmer-v5": 8,
    "HalfCheetah-v5": 17,
    "Walker2d-v5": 17,
    "Ant-v5": 105,
    "HumanoidStandup-v5": 348,
}

ACT_DIM = {
    "LunarLander-v3": None,  # Discrete(4)
    "Reacher-v5": 2,
    "Pusher-v5": 7,
    "Swimmer-v5": 2,
    "HalfCheetah-v5": 6,
    "Walker2d-v5": 6,
    "Ant-v5": 8,
    "HumanoidStandup-v5": 17,
}

# Every info key any selection prior reads, so a partial that expects a missing
# key would still be exercised rather than silently taking the 0.0 default.
FULL_INFO = {
    "reward_dist": -0.3,
    "reward_near": -0.2,
    "reward_ctrl": -0.15,
    "reward_survive": 1.0,
    "reward_forward": 0.7,
    "reward_linup": 40.0,
    "reward_quadctrl": -0.09,
    "reward_impact": -0.14,
    "x_velocity": 0.7,
    "y_velocity": 0.2,
    "x_position": 1.5,
    "y_position": 0.5,
}


def _dummy_step(env_id, *, info=None, action_scale=0.5, terminated=False):
    obs = np.full(OBS_DIM[env_id], 0.1, dtype=np.float64)
    action = 2 if ACT_DIM[env_id] is None else np.full(ACT_DIM[env_id], action_scale)
    payload = dict(FULL_INFO)
    payload.update(info or {})
    return dict(
        obs=obs,
        action=action,
        next_obs=obs,
        true_reward=0.0,
        terminated=terminated,
        truncated=False,
        info=payload,
    )


def _make(module, name, suite, env_id):
    spec = load_partial_reference(f"{module}:{name}", suite, PartialRegistry())
    return spec, spec.create(env_id)


@pytest.mark.parametrize("module,name,suite,env_id", SELECTION_PARTIALS)
def test_resolves_and_returns_finite(module, name, suite, env_id):
    """The `file:name` form is what --partial has to accept; a bare name resolves
    the FILE, so a registry-only check would pass for a reference the CLI rejects."""
    spec, partial = _make(module, name, suite, env_id)
    assert spec.env_ids == (env_id,)
    partial.reset({})
    for _ in range(3):
        step = partial.step(**_dummy_step(env_id))
        assert math.isfinite(step.partial)
        assert all(math.isfinite(value) for value in step.components.values())


@pytest.mark.parametrize("module,name,suite,env_id", SELECTION_PARTIALS)
def test_rejects_other_envs(module, name, suite, env_id):
    spec, _ = _make(module, name, suite, env_id)
    other = "Pusher-v5" if env_id != "Pusher-v5" else "Reacher-v5"
    with pytest.raises(PartialRegistryError):
        spec.create(other)


@pytest.mark.parametrize("module,name,suite,env_id", SELECTION_PARTIALS)
def test_deterministic(module, name, suite, env_id):
    _, a = _make(module, name, suite, env_id)
    _, b = _make(module, name, suite, env_id)
    a.reset({})
    b.reset({})
    for _ in range(3):
        assert a.step(**_dummy_step(env_id)).partial == b.step(**_dummy_step(env_id)).partial


# module, name, env_id, info key that carries the objective, cap value
SATURATING = [
    ("sel_swimmer", "sswm_cap10", "Swimmer-v5", "x_velocity", 0.10),
    ("sel_swimmer", "sswm_cap18", "Swimmer-v5", "x_velocity", 0.18),
    ("sel_halfcheetah", "shc_cap10", "HalfCheetah-v5", "x_velocity", 1.0),
    ("sel_halfcheetah", "shc_cap20", "HalfCheetah-v5", "x_velocity", 2.0),
    ("sel_walker2d", "swk_cap05", "Walker2d-v5", "x_velocity", 0.5),
    ("sel_walker2d", "swk_cap15", "Walker2d-v5", "x_velocity", 1.5),
    ("sel_ant", "sant_cap03", "Ant-v5", "x_velocity", 0.3),
    ("sel_ant", "sant_cap10", "Ant-v5", "x_velocity", 1.0),
    ("sel_humanoidstandup", "shs_cap60", "HumanoidStandup-v5", "reward_linup", 60.0),
    ("sel_humanoidstandup", "shs_cap120", "HumanoidStandup-v5", "reward_linup", 120.0),
]


@pytest.mark.parametrize("module,name,env_id,key,cap", SATURATING)
def test_saturates_above_cap(module, name, env_id, key, cap):
    """A capped prior must stop paying above the cap and must still track below it.
    Saturation is the whole reason these are not just --partial-alpha under
    another name: scaling a prior by c is exactly alpha=c, capping it is not."""
    _, partial = _make(module, name, "mujoco", env_id)

    def value(objective):
        partial.reset({})
        return partial.step(**_dummy_step(env_id, info={key: objective, "reward_survive": 0.0}, action_scale=0.0)).partial

    assert value(cap * 10) == pytest.approx(value(cap * 100))
    assert value(cap * 0.5) < value(cap)
    assert value(cap) == pytest.approx(value(cap * 10))


# module, name, env_id, weight the prior charges, weight the true reward charges
OVER_PRICED_EFFORT = [
    ("sel_pusher", "spsh_timid", "Pusher-v5", 0.3, 0.1),
    ("sel_swimmer", "sswm_timid", "Swimmer-v5", 0.05, 1e-4),
    ("sel_halfcheetah", "shc_timid", "HalfCheetah-v5", 1.0, 0.1),
    ("sel_walker2d", "swk_timid", "Walker2d-v5", 0.2, 1e-3),
    ("sel_ant", "sant_timid", "Ant-v5", 1.5, 0.5),
    ("sel_humanoidstandup", "shs_timid", "HumanoidStandup-v5", 50.0, 0.1),
]


@pytest.mark.parametrize("module,name,env_id,weight,true_weight", OVER_PRICED_EFFORT)
def test_charges_more_for_effort_than_the_true_reward(module, name, env_id, weight, true_weight):
    """These priors are supposed to over-price effort. An edit that drops the term
    or lowers it below the environment's own weight turns them back into the true
    reward, which is exactly the failure mode that made three of them useless in
    the first draft."""
    assert weight > true_weight
    _, partial = _make(module, name, "mujoco", env_id)

    def value(scale):
        partial.reset({})
        return partial.step(**_dummy_step(env_id, action_scale=scale)).partial

    charged = value(0.0) - value(1.0)
    expected = weight * ACT_DIM[env_id] * 1.0
    assert charged == pytest.approx(expected)


def test_lunarlander_fuel_term_matches_the_env():
    """sll_pad_fuel is the only selection prior that reconstructs a true-reward
    term from the action, so it has to match lunar_lander.py exactly: -0.30 for
    the main engine (action 2) and -0.03 for either side engine (actions 1, 3)."""
    _, partial = _make("sel_lunarlander", "sll_pad_fuel", "box2d", "LunarLander-v3")
    charged = {}
    for action in (0, 1, 2, 3):
        partial.reset({})
        step = _dummy_step("LunarLander-v3")
        step["action"] = action
        charged[action] = partial.step(**step).components["fuel"]
    assert charged == {0: 0.0, 1: -0.03, 2: -0.30, 3: -0.03}


def test_lunarlander_touchdown_is_lenient_about_speed():
    """sll_touchdown is the gameable candidate: it must pay its landing bonus on
    leg contact alone, at a descent speed the real env scores -100 for."""
    _, partial = _make("sel_lunarlander", "sll_touchdown", "box2d", "LunarLander-v3")
    crash = np.array([0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 1.0, 1.0])  # legs down, falling fast
    miss = np.array([0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0])   # no leg contact
    partial.reset({})
    landed = partial.step(**{**_dummy_step("LunarLander-v3", terminated=True), "next_obs": crash})
    partial.reset({})
    failed = partial.step(**{**_dummy_step("LunarLander-v3", terminated=True), "next_obs": miss})
    assert landed.components["terminal_judgement"] == 60.0
    assert failed.components["terminal_judgement"] == -60.0
