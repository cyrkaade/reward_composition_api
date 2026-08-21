"""Aligned, deliberately incomplete partials for the reasonable-prior screen.

Every rewarded term has the same direction as the task reward or is a modest,
human-plausible safety/effort term.  The grid intentionally excludes the old
gameable constructions: no crash bonus, no reward for sideways/backwards motion,
no target-speed penalty above an arbitrary optimum, and no oversized action cost.

Uniform reward rescaling is not represented because VecNormalize removes it in
partial-only training.  Candidate families instead vary omissions, monotone caps,
and relative weights.  Atari candidates live in ``reasonable_atari_partials`` so
their RAM-only boundary remains obvious and independently testable.
"""

from __future__ import annotations

import numpy as np


def _flat(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


class LunarLanderReasonablePartial:
    """Potential shaping using an aligned subset of LunarLander's terms.

    All candidates omit the environment's +/-100 terminal replacement, so none
    reconstructs the true reward.  Potential differences avoid paying forever
    merely for hovering in an attractive state.
    """

    component_keys = ("pad_progress", "speed_progress", "tilt_progress", "spin_progress", "leg_progress", "fuel")

    def __init__(
        self,
        *,
        distance_weight: float = 100.0,
        speed_weight: float = 100.0,
        tilt_weight: float = 0.0,
        spin_weight: float = 0.0,
        leg_weight: float = 0.0,
        fuel_weight: float = 0.0,
    ):
        self.weights = np.asarray(
            [distance_weight, speed_weight, tilt_weight, spin_weight, leg_weight],
            dtype=np.float64,
        )
        self.fuel_weight = float(fuel_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def _terms(self, state) -> np.ndarray:
        s = _flat(state)
        if s.size < 8:
            raise ValueError(f"LunarLander partial expected 8 observations, got {s.size}")
        return np.asarray(
            [
                -float(np.hypot(s[0], s[1])),
                -float(np.hypot(s[2], s[3])),
                -abs(float(s[4])),
                -abs(float(s[5])),
                float(s[6] + s[7]),
            ],
            dtype=np.float64,
        )

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        changes = self.weights * (self._terms(next_obs) - self._terms(obs))
        action_id = int(_flat(action)[0])
        raw_fuel = 0.30 if action_id == 2 else 0.03 if action_id in (1, 3) else 0.0
        fuel = -self.fuel_weight * raw_fuel
        values = (*changes.tolist(), fuel)
        return {
            "partial": float(sum(values)),
            "components": dict(zip(self.component_keys, map(float, values))),
        }


class ReacherReasonablePartial:
    """Distance proxies plus a nonnegative fraction of the true effort cost."""

    component_keys = ("reach", "effort")

    def __init__(self, *, shape: str, ctrl_weight: float, distance_floor: float | None = None):
        self.shape = str(shape)
        self.ctrl_weight = float(ctrl_weight)
        self.distance_floor = distance_floor

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        state = _flat(next_obs)
        if state.size < 10:
            raise ValueError(f"Reacher partial expected 10 observations, got {state.size}")
        dx, dy = float(state[8]), float(state[9])
        euclidean = float(np.hypot(dx, dy))
        if self.shape == "x":
            reach = -abs(dx)
        elif self.shape == "l1":
            reach = -(abs(dx) + abs(dy))
        elif self.shape == "distance":
            reach = float((info or {}).get("reward_dist", -euclidean))
        elif self.shape == "hit":
            reach = 1.0 if euclidean < 0.05 else 0.0
        else:
            raise ValueError(f"unknown Reacher shape: {self.shape}")
        if self.distance_floor is not None and self.shape != "hit":
            reach = max(reach, -float(self.distance_floor))
        effort = -self.ctrl_weight * float(np.square(_flat(action)).sum())
        return {"partial": float(reach + effort), "components": {"reach": float(reach), "effort": effort}}


class PusherReasonablePartial:
    """Goal/reach terms, or goal-potential progress, with modest regularizers."""

    component_keys = ("goal", "reach", "effort", "success", "smoothness")

    def __init__(
        self,
        *,
        goal_mode: str = "distance",
        goal_weight: float = 1.0,
        near_weight: float = 0.5,
        ctrl_weight: float = 0.0,
        goal_floor: float | None = None,
        success_bonus: float = 0.0,
        smooth_weight: float = 0.0,
    ):
        self.goal_mode = str(goal_mode)
        self.goal_weight = float(goal_weight)
        self.near_weight = float(near_weight)
        self.ctrl_weight = float(ctrl_weight)
        self.goal_floor = goal_floor
        self.success_bonus = float(success_bonus)
        self.smooth_weight = float(smooth_weight)
        self.previous_action: np.ndarray | None = None

    def reset(self, info: dict | None = None) -> None:
        self.previous_action = None

    @staticmethod
    def _distances(state) -> tuple[float, float]:
        s = _flat(state)
        if s.size < 23:
            raise ValueError(f"Pusher partial expected 23 observations, got {s.size}")
        fingertip, obj, goal = s[14:17], s[17:20], s[20:23]
        return float(np.linalg.norm(obj - goal)), float(np.linalg.norm(fingertip - obj))

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        old_goal_distance, _ = self._distances(obs)
        goal_distance, near_distance = self._distances(next_obs)
        if self.goal_mode == "progress":
            goal = self.goal_weight * (old_goal_distance - goal_distance)
        elif self.goal_mode == "distance":
            goal = -self.goal_weight * goal_distance
            if self.goal_floor is not None:
                goal = max(goal, -self.goal_weight * float(self.goal_floor))
        else:
            raise ValueError(f"unknown Pusher goal mode: {self.goal_mode}")
        reach = -self.near_weight * near_distance
        effort = -self.ctrl_weight * float(np.square(_flat(action)).sum())
        success = self.success_bonus if goal_distance < 0.12 else 0.0
        current_action = _flat(action)
        smoothness = 0.0
        if self.previous_action is not None:
            smoothness = -self.smooth_weight * float(np.square(current_action - self.previous_action).sum())
        self.previous_action = current_action.copy()
        values = (goal, reach, effort, success, smoothness)
        return {
            "partial": float(sum(values)),
            "components": dict(zip(self.component_keys, map(float, values))),
        }


class CappedLocomotionPartial:
    """Stay healthy and make forward progress, with monotone capped credit."""

    component_keys = ("survive", "capped_forward", "effort")

    def __init__(self, *, forward_cap: float, ctrl_weight: float = 0.0):
        self.forward_cap = float(forward_cap)
        self.ctrl_weight = float(ctrl_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        payload = info or {}
        survive = float(payload.get("reward_survive", 0.0))
        forward = float(payload.get("reward_forward", payload.get("x_velocity", 0.0)))
        capped_forward = min(forward, self.forward_cap)
        effort = -self.ctrl_weight * float(np.square(_flat(action)).sum())
        return {
            "partial": float(survive + capped_forward + effort),
            "components": {"survive": survive, "capped_forward": capped_forward, "effort": effort},
        }


class SwimmerReasonablePartial:
    """Forward velocity with a monotone cap; backwards motion remains negative."""

    component_keys = ("capped_forward", "effort")

    def __init__(self, *, forward_cap: float, ctrl_weight: float = 0.0):
        self.forward_cap = float(forward_cap)
        self.ctrl_weight = float(ctrl_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        forward = float((info or {}).get("x_velocity", 0.0))
        capped_forward = min(forward, self.forward_cap)
        effort = -self.ctrl_weight * float(np.square(_flat(action)).sum())
        return {
            "partial": float(capped_forward + effort),
            "components": {"capped_forward": capped_forward, "effort": effort},
        }


class BipedalReasonablePartial:
    """Capped forward velocity with an optional mild upright preference."""

    component_keys = ("capped_forward", "upright")

    def __init__(self, *, forward_cap: float, upright_weight: float = 0.0):
        self.forward_cap = float(forward_cap)
        self.upright_weight = float(upright_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        state = _flat(next_obs)
        if state.size < 3:
            raise ValueError(f"BipedalWalker partial expected 24 observations, got {state.size}")
        forward = float(state[2]) * 10.0
        capped_forward = min(forward, self.forward_cap)
        upright = -self.upright_weight * abs(float(state[0]))
        return {
            "partial": float(capped_forward + upright),
            "components": {"capped_forward": capped_forward, "upright": upright},
        }


_PARTIALS: dict[str, tuple[str, str, type, dict]] = {}


def _add_family(prefix: str, suite: str, env_id: str, cls: type, variants: dict[str, dict]) -> None:
    for suffix, kwargs in variants.items():
        _PARTIALS[f"{prefix}_{suffix}"] = (suite, env_id, cls, kwargs)


_add_family("rll", "box2d", "LunarLander-v3", LunarLanderReasonablePartial, {
    "pad_speed": {},
    "tilt25": {"tilt_weight": 25.0},
    "tilt50": {"tilt_weight": 50.0},
    "legs": {"leg_weight": 10.0},
    "tilt_legs": {"tilt_weight": 50.0, "leg_weight": 10.0},
    "spin": {"spin_weight": 25.0},
    "fuel": {"fuel_weight": 1.0},
    "tilt_fuel": {"tilt_weight": 50.0, "fuel_weight": 1.0},
})

_add_family("rrch", "mujoco", "Reacher-v5", ReacherReasonablePartial, {
    "x_ctrl100": {"shape": "x", "ctrl_weight": 1.0},
    "x_ctrl050": {"shape": "x", "ctrl_weight": 0.5},
    "l1_ctrl100": {"shape": "l1", "ctrl_weight": 1.0},
    "l1_ctrl050": {"shape": "l1", "ctrl_weight": 0.5},
    "dist_ctrl025": {"shape": "distance", "ctrl_weight": 0.25},
    "dist_ctrl050": {"shape": "distance", "ctrl_weight": 0.5},
    "cap05_ctrl050": {"shape": "distance", "ctrl_weight": 0.5, "distance_floor": 0.05},
    "hit05_ctrl025": {"shape": "hit", "ctrl_weight": 0.25},
})

_add_family("rpsh", "mujoco", "Pusher-v5", PusherReasonablePartial, {
    "task_ctrl15": {"ctrl_weight": 0.15},
    "task_ctrl20": {"ctrl_weight": 0.20},
    "cap15_ctrl10": {"ctrl_weight": 0.10, "goal_floor": 0.15},
    "cap20_ctrl10": {"ctrl_weight": 0.10, "goal_floor": 0.20},
    "progress_near": {"goal_mode": "progress", "goal_weight": 20.0},
    "progress_smooth": {"goal_mode": "progress", "goal_weight": 20.0, "smooth_weight": 0.05},
    "progress_success": {"goal_mode": "progress", "goal_weight": 20.0, "near_weight": 0.25, "success_bonus": 1.0},
    "progress_ctrl05": {"goal_mode": "progress", "goal_weight": 20.0, "ctrl_weight": 0.05},
})

_add_family("rswm", "mujoco", "Swimmer-v5", SwimmerReasonablePartial, {
    "cap12": {"forward_cap": 0.12},
    "cap16": {"forward_cap": 0.16},
    "cap20": {"forward_cap": 0.20},
    "cap24": {"forward_cap": 0.24},
    "cap28": {"forward_cap": 0.28},
    "cap32": {"forward_cap": 0.32},
    "cap40": {"forward_cap": 0.40},
    "cap28_ctrl001": {"forward_cap": 0.28, "ctrl_weight": 0.001},
})

_add_family("rhop", "mujoco", "Hopper-v5", CappedLocomotionPartial, {
    "cap00": {"forward_cap": 0.0},
    "cap10": {"forward_cap": 0.10},
    "cap15": {"forward_cap": 0.15},
    "cap20": {"forward_cap": 0.20},
    "cap30": {"forward_cap": 0.30},
    "cap40": {"forward_cap": 0.40},
    "cap60": {"forward_cap": 0.60},
    "cap80": {"forward_cap": 0.80},
})

_add_family("rbw", "box2d", "BipedalWalker-v3", BipedalReasonablePartial, {
    "cap10": {"forward_cap": 0.10},
    "cap20": {"forward_cap": 0.20},
    "cap30": {"forward_cap": 0.30},
    "cap40": {"forward_cap": 0.40},
    "cap50": {"forward_cap": 0.50},
    "cap75": {"forward_cap": 0.75},
    "cap30_upright": {"forward_cap": 0.30, "upright_weight": 0.25},
    "cap50_upright": {"forward_cap": 0.50, "upright_weight": 0.25},
})

_add_family("rwalk", "mujoco", "Walker2d-v5", CappedLocomotionPartial, {
    "cap050": {"forward_cap": 0.50},
    "cap100": {"forward_cap": 1.00},
    "cap150": {"forward_cap": 1.50},
    "cap175": {"forward_cap": 1.75},
    "cap200": {"forward_cap": 2.00},
    "cap225": {"forward_cap": 2.25},
    "cap250": {"forward_cap": 2.50},
    "cap300": {"forward_cap": 3.00},
})

_add_family("rant", "mujoco", "Ant-v5", CappedLocomotionPartial, {
    "cap05": {"forward_cap": 0.05},
    "cap10": {"forward_cap": 0.10},
    "cap20": {"forward_cap": 0.20},
    "cap30": {"forward_cap": 0.30},
    "cap50": {"forward_cap": 0.50},
    "cap30_ctrl05": {"forward_cap": 0.30, "ctrl_weight": 0.05},
    "cap50_ctrl10": {"forward_cap": 0.50, "ctrl_weight": 0.10},
    "cap75_ctrl10": {"forward_cap": 0.75, "ctrl_weight": 0.10},
})


def register(registry) -> None:
    for name, (suite, env_id, cls, kwargs) in _PARTIALS.items():
        registry.register(
            name=name,
            suite=suite,
            factory=lambda requested_env, cls=cls, kwargs=kwargs: cls(**kwargs),
            description=f"Aligned incomplete candidate for the reasonable-prior screen: {name}",
            env_ids=(env_id,),
            component_keys=cls.component_keys,
        )
