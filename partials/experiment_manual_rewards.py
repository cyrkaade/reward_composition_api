"""Manual partial and full reward functions for the large benchmark runs."""

from __future__ import annotations

import numpy as np

from reward_composition_api.registry import PartialRewardStep


def _info_value(info: dict | None, key: str, default: float = 0.0) -> float:
    return float((info or {}).get(key, default))


def _action_array(action) -> np.ndarray:
    return np.asarray(action, dtype=np.float64).reshape(-1)


class MuJoCoComponentReward:
    def __init__(self, *, forward_cap: float | None = None, include_ctrl: bool = False):
        self.forward_cap = forward_cap
        self.include_ctrl = include_ctrl

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        forward = _info_value(info, "reward_forward")
        survive = _info_value(info, "reward_survive")
        ctrl = _info_value(info, "reward_ctrl")
        partial_forward = min(forward, self.forward_cap) if self.forward_cap is not None else forward
        partial = partial_forward + survive + (ctrl if self.include_ctrl else 0.0)
        return PartialRewardStep(
            partial=float(partial),
            components={
                "reward_forward": forward,
                "partial_forward": float(partial_forward),
                "reward_survive": survive,
                "reward_ctrl": ctrl,
                "ctrl_included": float(ctrl if self.include_ctrl else 0.0),
            },
        )


class AtariScoreLifePartial:
    def __init__(self, *, life_loss_penalty: float, include_full_score: bool = False):
        self.life_loss_penalty = float(life_loss_penalty)
        self.include_full_score = bool(include_full_score)
        self.previous_lives: int | None = None

    def reset(self, info: dict | None = None) -> None:
        self.previous_lives = int((info or {}).get("lives", 0))

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        lives = int((info or {}).get("lives", 0))
        lost_lives = 0 if self.previous_lives is None else max(self.previous_lives - lives, 0)
        self.previous_lives = lives

        score = float(true_reward)
        score_component = score if self.include_full_score else float(np.clip(score, 0.0, 1.0))
        life_component = -self.life_loss_penalty * float(lost_lives)
        partial = score_component + life_component
        return PartialRewardStep(
            partial=float(partial),
            components={
                "score_reward": score,
                "score_component": float(score_component),
                "life_loss_penalty": float(life_component),
                "lost_lives": float(lost_lives),
                "lives": float(lives),
            },
        )


class TrueRewardPassthrough:
    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        reward = float(true_reward)
        return PartialRewardStep(
            partial=reward,
            components={"true_env_reward": reward},
        )


class BipedalWalkerHeuristicPartial:
    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        state = np.asarray(next_obs, dtype=np.float64)
        act = _action_array(action)
        hull_angle = float(state[0]) if state.size > 0 else 0.0
        angular_velocity = float(state[1]) if state.size > 1 else 0.0
        horizontal_velocity = float(state[2]) if state.size > 2 else 0.0
        vertical_velocity = float(state[3]) if state.size > 3 else 0.0
        left_contact = float(state[8]) if state.size > 8 else 0.0
        right_contact = float(state[13]) if state.size > 13 else 0.0

        forward = 2.0 * horizontal_velocity
        upright = -0.8 * abs(hull_angle)
        stability = -0.15 * abs(angular_velocity) - 0.05 * abs(vertical_velocity)
        contact = 0.1 * (left_contact + right_contact)
        partial = forward + upright + stability + contact

        return PartialRewardStep(
            partial=float(partial),
            components={
                "forward_velocity": float(forward),
                "upright_penalty": float(upright),
                "stability_penalty": float(stability),
                "leg_contact_bonus": float(contact),
                "motor_cost_omitted": float(-0.00035 * 80.0 * np.sum(np.abs(act))),
            },
        )


class CarRacingHeuristicPartial:
    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        act = _action_array(action)
        steer = float(act[0]) if act.size > 0 else 0.0
        gas = float(act[1]) if act.size > 1 else 0.0
        brake = float(act[2]) if act.size > 2 else 0.0

        road_center_bonus = self._road_center_bonus(next_obs)
        gas_bonus = 0.05 * max(gas, 0.0)
        control_penalty = -0.02 * abs(steer) - 0.03 * max(brake, 0.0)
        partial = road_center_bonus + gas_bonus + control_penalty
        return PartialRewardStep(
            partial=float(partial),
            components={
                "road_center_bonus": float(road_center_bonus),
                "gas_bonus": float(gas_bonus),
                "control_penalty": float(control_penalty),
                "track_progress_omitted": 0.0,
            },
        )

    def _road_center_bonus(self, observation) -> float:
        image = np.asarray(observation)
        if image.ndim != 3 or image.shape[-1] < 3:
            return 0.0
        bottom = image[int(image.shape[0] * 0.55) :, :, :3].astype(np.float64)
        gray = np.abs(bottom[:, :, 0] - bottom[:, :, 1]) < 12
        gray &= np.abs(bottom[:, :, 1] - bottom[:, :, 2]) < 12
        gray &= bottom[:, :, 0] > 80
        if not np.any(gray):
            return -0.05
        cols = np.where(gray)[1]
        center = (bottom.shape[1] - 1) / 2.0
        offset = abs(float(np.mean(cols)) - center) / max(center, 1.0)
        return 0.1 * (1.0 - min(offset, 1.0))


def register(registry) -> None:
    registry.register(
        "hopper_manual_partial",
        "mujoco",
        lambda env_id: MuJoCoComponentReward(forward_cap=1.0, include_ctrl=False),
        env_ids=("Hopper-v5",),
        component_keys=("reward_forward", "partial_forward", "reward_survive", "reward_ctrl", "ctrl_included"),
    )
    registry.register(
        "hopper_manual_full",
        "mujoco",
        lambda env_id: MuJoCoComponentReward(forward_cap=None, include_ctrl=True),
        env_ids=("Hopper-v5",),
        component_keys=("reward_forward", "partial_forward", "reward_survive", "reward_ctrl", "ctrl_included"),
    )
    registry.register(
        "walker2d_manual_partial",
        "mujoco",
        lambda env_id: MuJoCoComponentReward(forward_cap=1.0, include_ctrl=False),
        env_ids=("Walker2d-v5",),
        component_keys=("reward_forward", "partial_forward", "reward_survive", "reward_ctrl", "ctrl_included"),
    )
    registry.register(
        "walker2d_manual_full",
        "mujoco",
        lambda env_id: MuJoCoComponentReward(forward_cap=None, include_ctrl=True),
        env_ids=("Walker2d-v5",),
        component_keys=("reward_forward", "partial_forward", "reward_survive", "reward_ctrl", "ctrl_included"),
    )
    registry.register(
        "breakout_manual_partial",
        "atari",
        lambda env_id: AtariScoreLifePartial(life_loss_penalty=1.0, include_full_score=False),
        env_ids=("ALE/Breakout-v5",),
        component_keys=("score_reward", "score_component", "life_loss_penalty", "lost_lives", "lives"),
    )
    registry.register(
        "breakout_manual_full",
        "atari",
        lambda env_id: AtariScoreLifePartial(life_loss_penalty=0.0, include_full_score=True),
        env_ids=("ALE/Breakout-v5",),
        component_keys=("score_reward", "score_component", "life_loss_penalty", "lost_lives", "lives"),
    )
    registry.register(
        "spaceinvaders_manual_partial",
        "atari",
        lambda env_id: AtariScoreLifePartial(life_loss_penalty=2.0, include_full_score=False),
        env_ids=("ALE/SpaceInvaders-v5",),
        component_keys=("score_reward", "score_component", "life_loss_penalty", "lost_lives", "lives"),
    )
    registry.register(
        "spaceinvaders_manual_full",
        "atari",
        lambda env_id: AtariScoreLifePartial(life_loss_penalty=0.0, include_full_score=True),
        env_ids=("ALE/SpaceInvaders-v5",),
        component_keys=("score_reward", "score_component", "life_loss_penalty", "lost_lives", "lives"),
    )
    registry.register(
        "bipedalwalker_manual_partial",
        "box2d",
        lambda env_id: BipedalWalkerHeuristicPartial(),
        env_ids=("BipedalWalker-v3",),
        component_keys=("forward_velocity", "upright_penalty", "stability_penalty", "leg_contact_bonus", "motor_cost_omitted"),
    )
    registry.register(
        "bipedalwalker_manual_full",
        "box2d",
        lambda env_id: TrueRewardPassthrough(),
        env_ids=("BipedalWalker-v3",),
        component_keys=("true_env_reward",),
    )
    registry.register(
        "carracing_manual_partial",
        "box2d",
        lambda env_id: CarRacingHeuristicPartial(),
        env_ids=("CarRacing-v3",),
        component_keys=("road_center_bonus", "gas_bonus", "control_penalty", "track_progress_omitted"),
    )
    registry.register(
        "carracing_manual_full",
        "box2d",
        lambda env_id: TrueRewardPassthrough(),
        env_ids=("CarRacing-v3",),
        component_keys=("true_env_reward",),
    )
