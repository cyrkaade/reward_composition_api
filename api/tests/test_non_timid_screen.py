"""Regression tests for the independent non-timid screen proxies."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rcomp.partials import PartialRegistry, load_partial_reference


CANDIDATES = {
    "Pusher-v5": (
        "npsh_progress", "npsh_progress_smooth", "npsh_success_smooth", "npsh_progress_calm",
        "npsh_progress_success",
    ),
    "Swimmer-v5": ("nswm_target12", "nswm_target18"),
    "Walker2d-v5": ("nwalk_speed_cap10", "nwalk_speed_cap20", "nwalk_posture_cap15"),
    "Ant-v5": ("nant_forward_smooth", "nant_radial_smooth", "nant_target_smooth", "nant_posture_smooth"),
    "BipedalWalker-v3": ("nbw_speed_smooth",),
}

OBS_DIM = {"Pusher-v5": 23, "Swimmer-v5": 8, "Walker2d-v5": 17, "Ant-v5": 105, "BipedalWalker-v3": 24}
ACT_DIM = {"Pusher-v5": 7, "Swimmer-v5": 2, "Walker2d-v5": 6, "Ant-v5": 8, "BipedalWalker-v3": 4}


def _make(name, env_id):
    suite = "box2d" if env_id == "BipedalWalker-v3" else "mujoco"
    spec = load_partial_reference(f"non_timid_screen:{name}", suite, PartialRegistry())
    return spec.create(env_id)


def _step(env_id, *, obs=None, next_obs=None, action=None, info=None):
    obs = np.zeros(OBS_DIM[env_id], dtype=np.float64) if obs is None else np.asarray(obs, dtype=np.float64)
    next_obs = obs.copy() if next_obs is None else np.asarray(next_obs, dtype=np.float64)
    action = np.zeros(ACT_DIM[env_id], dtype=np.float64) if action is None else np.asarray(action, dtype=np.float64)
    payload = {
        "x_velocity": 0.3,
        "y_velocity": 0.1,
        "x_position": 1.0,
        "y_position": 0.5,
        "reward_survive": 1.0,
    }
    payload.update(info or {})
    return dict(
        obs=obs,
        action=action,
        next_obs=next_obs,
        true_reward=0.0,
        terminated=False,
        truncated=False,
        info=payload,
    )


@pytest.mark.parametrize(
    "env_id,name",
    [(env_id, name) for env_id, names in CANDIDATES.items() for name in names],
)
def test_candidates_resolve_and_return_finite(env_id, name):
    partial = _make(name, env_id)
    partial.reset({})
    result = partial.step(**_step(env_id))
    assert math.isfinite(result.partial)
    assert all(math.isfinite(value) for value in result.components.values())


def _pusher_state(*, fingertip=(0.0, 0.0, 0.0), obj=(0.5, 0.0, 0.0), goal=(0.0, 0.0, 0.0)):
    state = np.zeros(23, dtype=np.float64)
    state[14:17] = fingertip
    state[17:20] = obj
    state[20:23] = goal
    return state


def test_pusher_progress_rewards_closing_the_object_goal_gap():
    partial = _make("npsh_progress", "Pusher-v5")
    before = _pusher_state(fingertip=(0.4, 0.0, 0.0), obj=(0.5, 0.0, 0.0))
    closer = _pusher_state(fingertip=(0.3, 0.0, 0.0), obj=(0.3, 0.0, 0.0))
    farther = _pusher_state(fingertip=(0.6, 0.0, 0.0), obj=(0.6, 0.0, 0.0))
    partial.reset({})
    good = partial.step(**_step("Pusher-v5", obs=before, next_obs=closer)).partial
    partial.reset({})
    bad = partial.step(**_step("Pusher-v5", obs=before, next_obs=farther)).partial
    assert good > bad


def test_pusher_smoothness_charges_action_change_not_sustained_action():
    partial = _make("npsh_progress_smooth", "Pusher-v5")
    state = _pusher_state()
    action = np.ones(7)
    partial.reset({})
    first = partial.step(**_step("Pusher-v5", obs=state, next_obs=state, action=action))
    same = partial.step(**_step("Pusher-v5", obs=state, next_obs=state, action=action))
    changed = partial.step(**_step("Pusher-v5", obs=state, next_obs=state, action=-action))
    assert first.components["smoothness"] == 0.0
    assert same.components["smoothness"] == 0.0
    assert changed.components["smoothness"] < 0.0


@pytest.mark.parametrize("name,target", [("nswm_target12", 0.12), ("nswm_target18", 0.18)])
def test_swimmer_target_band_penalizes_both_too_slow_and_too_fast(name, target):
    partial = _make(name, "Swimmer-v5")
    at_target = partial.step(**_step("Swimmer-v5", info={"x_velocity": target})).partial
    too_slow = partial.step(**_step("Swimmer-v5", info={"x_velocity": target - 0.10})).partial
    too_fast = partial.step(**_step("Swimmer-v5", info={"x_velocity": target + 0.10})).partial
    assert at_target > too_slow
    assert at_target > too_fast


def test_walker_speed_proxy_saturates_and_ignores_true_alive_and_action_terms():
    partial = _make("nwalk_speed_cap10", "Walker2d-v5")

    def value(vx, survive, action):
        return partial.step(**_step(
            "Walker2d-v5", action=np.full(6, action),
            info={"x_velocity": vx, "reward_survive": survive, "reward_ctrl": -999.0},
        )).partial

    assert value(1.0, 0.0, 0.0) == pytest.approx(value(10.0, 100.0, 1.0))
    assert value(0.5, 0.0, 0.0) < value(1.0, 0.0, 0.0)


def test_ant_smoothness_charges_changes_but_not_constant_torque():
    partial = _make("nant_forward_smooth", "Ant-v5")
    action = np.ones(8)
    partial.reset({})
    first = partial.step(**_step("Ant-v5", action=action))
    same = partial.step(**_step("Ant-v5", action=action))
    changed = partial.step(**_step("Ant-v5", action=-action))
    assert first.components["smoothness"] == 0.0
    assert same.components["smoothness"] == 0.0
    assert changed.components["smoothness"] < 0.0


def test_bipedal_smoothness_charges_action_change_not_action_size():
    partial = _make("nbw_speed_smooth", "BipedalWalker-v3")
    state = np.zeros(24)
    action = np.ones(4)
    partial.reset({})
    first = partial.step(**_step("BipedalWalker-v3", obs=state, next_obs=state, action=action))
    same = partial.step(**_step("BipedalWalker-v3", obs=state, next_obs=state, action=action))
    changed = partial.step(**_step("BipedalWalker-v3", obs=state, next_obs=state, action=-action))
    assert first.components["smoothness"] == 0.0
    assert same.components["smoothness"] == 0.0
    assert changed.components["smoothness"] < 0.0
