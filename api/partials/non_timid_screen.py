"""Non-timid proxy rewards for the independent prior screen.

These candidates deliberately avoid the ``*_timid`` construction (the true
reward with an enlarged copy of its own action-magnitude cost).  They use
structurally different signals instead: potential progress, sparse success,
action *change*, joint speed, target bands, posture, or direction-blind motion.

The screen, not this file, decides whether any candidate is good enough for the
confirmatory composition grid.  No candidate is promoted merely because its
formula sounds plausible.
"""

from __future__ import annotations

import numpy as np


def _flat(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


class PusherProxyPartial:
    """Progress/success proxies with non-true smoothness regularizers.

    Pusher-v5 observation layout is qpos[0:7], qvel[0:7], fingertip[14:17],
    object[17:20], goal[20:23].  The true reward repeatedly charges absolute
    object-goal distance, absolute fingertip-object distance, and action
    magnitude.  Here the goal term is instead a *change in potential*; optional
    costs charge action change or joint speed, neither of which is in the true
    reward.
    """

    component_keys = ("goal_progress", "near_proxy", "success", "touch", "smoothness", "joint_calm")

    def __init__(
        self,
        *,
        progress_scale: float,
        near_weight: float,
        success_bonus: float = 0.0,
        touch_bonus: float = 0.0,
        smooth_weight: float = 0.0,
        joint_speed_weight: float = 0.0,
        goal_radius: float = 0.12,
        touch_radius: float = 0.08,
    ):
        self.progress_scale = float(progress_scale)
        self.near_weight = float(near_weight)
        self.success_bonus = float(success_bonus)
        self.touch_bonus = float(touch_bonus)
        self.smooth_weight = float(smooth_weight)
        self.joint_speed_weight = float(joint_speed_weight)
        self.goal_radius = float(goal_radius)
        self.touch_radius = float(touch_radius)
        self.previous_goal_distance: float | None = None
        self.previous_action: np.ndarray | None = None

    def reset(self, info: dict | None = None) -> None:
        self.previous_goal_distance = None
        self.previous_action = None

    @staticmethod
    def _geometry(state) -> tuple[float, float]:
        s = _flat(state)
        if s.size < 23:
            raise ValueError(f"Pusher proxy expected 23 observations, got {s.size}")
        fingertip = s[14:17]
        obj = s[17:20]
        goal = s[20:23]
        return float(np.linalg.norm(obj - goal)), float(np.linalg.norm(obj - fingertip))

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        goal_distance, near_distance = self._geometry(next_obs)
        if self.previous_goal_distance is None:
            self.previous_goal_distance = self._geometry(obs)[0] if obs is not None else goal_distance
        progress = self.progress_scale * (self.previous_goal_distance - goal_distance)
        self.previous_goal_distance = goal_distance

        near_proxy = -self.near_weight * near_distance
        success = self.success_bonus if goal_distance < self.goal_radius else 0.0
        touch = self.touch_bonus if near_distance < self.touch_radius else 0.0

        current_action = _flat(action)
        smoothness = 0.0
        if self.previous_action is not None and self.smooth_weight:
            smoothness = -self.smooth_weight * float(np.square(current_action - self.previous_action).sum())
        self.previous_action = current_action.copy()

        s = _flat(next_obs)
        joint_calm = -self.joint_speed_weight * float(np.square(s[7:14]).sum())
        total = progress + near_proxy + success + touch + smoothness + joint_calm
        return {
            "partial": float(total),
            "components": {
                "goal_progress": float(progress),
                "near_proxy": float(near_proxy),
                "success": float(success),
                "touch": float(touch),
                "smoothness": float(smoothness),
                "joint_calm": float(joint_calm),
            },
        }


class SwimmerTargetPartial:
    """Reward a speed band rather than unbounded forward velocity.

    The true Swimmer reward is essentially ``x_velocity``.  A target band has a
    deliberately different optimum: going faster than the target is penalized
    just as going slower is.  No true control-cost term is copied.
    """

    component_keys = ("target_band", "x_velocity")

    def __init__(self, target_speed: float, tolerance: float = 0.03):
        self.target_speed = float(target_speed)
        self.tolerance = float(tolerance)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        vx = float((info or {}).get("x_velocity", 0.0))
        target_band = -max(abs(vx - self.target_speed) - self.tolerance, 0.0)
        return {
            "partial": float(target_band),
            "components": {"target_band": float(target_band), "x_velocity": vx},
        }


class WalkerProxyPartial:
    """Capped speed with optional posture proxies, no alive or true effort term."""

    component_keys = ("capped_speed", "angle_proxy", "height_proxy")

    def __init__(
        self,
        speed_cap: float,
        *,
        angle_weight: float = 0.0,
        height_weight: float = 0.0,
        target_height: float = 1.2,
    ):
        self.speed_cap = float(speed_cap)
        self.angle_weight = float(angle_weight)
        self.height_weight = float(height_weight)
        self.target_height = float(target_height)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        vx = float((info or {}).get("x_velocity", 0.0))
        s = _flat(next_obs)
        if s.size < 2:
            raise ValueError(f"Walker proxy expected at least 2 observations, got {s.size}")
        capped_speed = min(vx, self.speed_cap)
        angle_proxy = -self.angle_weight * abs(float(s[1]))
        height_proxy = -self.height_weight * abs(float(s[0]) - self.target_height)
        return {
            "partial": float(capped_speed + angle_proxy + height_proxy),
            "components": {
                "capped_speed": float(capped_speed),
                "angle_proxy": float(angle_proxy),
                "height_proxy": float(height_proxy),
            },
        }


class BipedalSmoothPartial:
    """Forward proxy regularized by action change, not the true torque bill."""

    component_keys = ("capped_speed", "smoothness", "upright")

    def __init__(self, speed_cap: float = 0.5, smooth_weight: float = 0.05, upright_weight: float = 0.0):
        self.speed_cap = float(speed_cap)
        self.smooth_weight = float(smooth_weight)
        self.upright_weight = float(upright_weight)
        self.previous_action: np.ndarray | None = None

    def reset(self, info: dict | None = None) -> None:
        self.previous_action = None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        state = _flat(next_obs)
        if state.size < 3:
            raise ValueError(f"Bipedal proxy expected at least 3 observations, got {state.size}")
        velocity = float(state[2]) * 10.0
        capped_speed = min(velocity, self.speed_cap)
        upright = -self.upright_weight * abs(float(state[0]))
        current_action = _flat(action)
        smoothness = 0.0
        if self.previous_action is not None:
            smoothness = -self.smooth_weight * float(np.square(current_action - self.previous_action).sum())
        self.previous_action = current_action.copy()
        return {
            "partial": float(capped_speed + smoothness + upright),
            "components": {
                "capped_speed": float(capped_speed),
                "smoothness": float(smoothness),
                "upright": float(upright),
            },
        }


class AntProxyPartial:
    """Ant locomotion proxies regularized by action change, not action size."""

    component_keys = ("progress", "survive", "smoothness", "height_proxy")

    def __init__(
        self,
        mode: str,
        *,
        speed_cap: float | None = None,
        target_speed: float = 0.5,
        smooth_weight: float = 0.25,
        height_weight: float = 0.0,
        target_height: float = 0.6,
    ):
        self.mode = str(mode)
        self.speed_cap = speed_cap
        self.target_speed = float(target_speed)
        self.smooth_weight = float(smooth_weight)
        self.height_weight = float(height_weight)
        self.target_height = float(target_height)
        self.previous_action: np.ndarray | None = None

    def reset(self, info: dict | None = None) -> None:
        self.previous_action = None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        payload = info or {}
        vx = float(payload.get("x_velocity", 0.0))
        vy = float(payload.get("y_velocity", 0.0))
        x = float(payload.get("x_position", 0.0))
        y = float(payload.get("y_position", 0.0))

        if self.mode == "forward":
            progress = vx
        elif self.mode == "radial":
            radius = float(np.hypot(x, y))
            progress = (x * vx + y * vy) / radius if radius > 1e-6 else 0.0
        elif self.mode == "target":
            progress = self.target_speed - abs(vx - self.target_speed)
        else:
            raise ValueError(f"unknown Ant proxy mode: {self.mode}")
        if self.speed_cap is not None:
            progress = min(progress, float(self.speed_cap))

        current_action = _flat(action)
        smoothness = 0.0
        if self.previous_action is not None:
            smoothness = -self.smooth_weight * float(np.square(current_action - self.previous_action).sum())
        self.previous_action = current_action.copy()

        s = _flat(next_obs)
        if s.size < 1:
            raise ValueError("Ant proxy expected a torso-height observation")
        height_proxy = -self.height_weight * abs(float(s[0]) - self.target_height)
        survive = float(payload.get("reward_survive", 0.0))
        return {
            "partial": float(progress + survive + smoothness + height_proxy),
            "components": {
                "progress": float(progress),
                "survive": survive,
                "smoothness": float(smoothness),
                "height_proxy": float(height_proxy),
            },
        }


_PARTIALS = {
    # Pusher: potential/success signals plus costs absent from the true reward.
    "npsh_progress": ("mujoco", "Pusher-v5", PusherProxyPartial, dict(
        progress_scale=20.0, near_weight=0.5,
    )),
    "npsh_progress_smooth": ("mujoco", "Pusher-v5", PusherProxyPartial, dict(
        progress_scale=20.0, near_weight=0.5, smooth_weight=0.10,
    )),
    "npsh_success_smooth": ("mujoco", "Pusher-v5", PusherProxyPartial, dict(
        progress_scale=10.0, near_weight=0.25, success_bonus=1.0,
        touch_bonus=0.10, smooth_weight=0.05,
    )),
    "npsh_progress_calm": ("mujoco", "Pusher-v5", PusherProxyPartial, dict(
        progress_scale=20.0, near_weight=0.5, joint_speed_weight=0.02,
    )),
    "npsh_progress_success": ("mujoco", "Pusher-v5", PusherProxyPartial, dict(
        progress_scale=20.0, near_weight=0.25, success_bonus=1.0, touch_bonus=0.10,
    )),
    # Swimmer: bounded target bands, not monotone copies of x velocity.
    "nswm_target12": ("mujoco", "Swimmer-v5", SwimmerTargetPartial, dict(target_speed=0.12)),
    "nswm_target18": ("mujoco", "Swimmer-v5", SwimmerTargetPartial, dict(target_speed=0.18)),
    # Walker: no healthy bonus, no true control cost, and nonlinear speed caps.
    "nwalk_speed_cap10": ("mujoco", "Walker2d-v5", WalkerProxyPartial, dict(speed_cap=1.0)),
    "nwalk_speed_cap20": ("mujoco", "Walker2d-v5", WalkerProxyPartial, dict(speed_cap=2.0)),
    "nwalk_posture_cap15": ("mujoco", "Walker2d-v5", WalkerProxyPartial, dict(
        speed_cap=1.5, angle_weight=0.5, height_weight=0.25,
    )),
    # BipedalWalker: action-to-action smoothness is absent from the true reward.
    "nbw_speed_smooth": ("box2d", "BipedalWalker-v3", BipedalSmoothPartial, dict(
        speed_cap=0.5, smooth_weight=0.05, upright_weight=0.25,
    )),
    # Ant: smoothness is action-to-action change, not the true |action|^2 bill.
    "nant_forward_smooth": ("mujoco", "Ant-v5", AntProxyPartial, dict(
        mode="forward", speed_cap=1.0,
    )),
    "nant_radial_smooth": ("mujoco", "Ant-v5", AntProxyPartial, dict(
        mode="radial", speed_cap=1.0,
    )),
    "nant_target_smooth": ("mujoco", "Ant-v5", AntProxyPartial, dict(
        mode="target", target_speed=0.5,
    )),
    "nant_posture_smooth": ("mujoco", "Ant-v5", AntProxyPartial, dict(
        mode="forward", speed_cap=1.0, height_weight=1.0,
    )),
}


def register(registry) -> None:
    for name, (suite, env_id, cls, kwargs) in _PARTIALS.items():
        registry.register(
            name=name,
            suite=suite,
            factory=lambda requested_env, cls=cls, kwargs=kwargs: cls(**kwargs),
            description=f"Independent non-timid screen proxy '{name}'",
            env_ids=(env_id,),
            component_keys=cls.component_keys,
        )
