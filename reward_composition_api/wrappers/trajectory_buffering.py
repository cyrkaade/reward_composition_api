from __future__ import annotations

from copy import deepcopy
from typing import Any

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from reward_composition_api.data_structures import Trajectory


class BufferingWrapper(VecEnvWrapper):
    """Saves transitions of underlying VecEnv.

    Retrieve saved transitions using `pop_transitions()`.
    """

    def __init__(self, venv: VecEnv, closed_form_fn: Any, synthetic_expert_fn: Any | None = None):
        """Builds BufferingWrapper.

        Args:
            venv: The wrapped VecEnv.
        """
        super().__init__(venv)
        self.temp_trajectories = [Trajectory] * self.num_envs
        self.finished_trajectories = []
        self.vec_cffns = [deepcopy(closed_form_fn) for _ in range(self.num_envs)]
        self.vec_expfn = [deepcopy(synthetic_expert_fn) for _ in range(self.num_envs)] if synthetic_expert_fn else None

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        for env_idx, reward_fn in enumerate(self.vec_cffns):
            self.temp_trajectories[env_idx] = Trajectory()
            reward_fn.reward(obs[env_idx], 0, False, False)
        if self.vec_expfn:
            for env_idx, reward_fn in enumerate(self.vec_expfn):
                reward_fn.reward(obs[env_idx], 0, False, False)
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)
        self._saved_acts = actions

    def step_wait(self):
        new_v_obs, v_rews, v_dones, v_infos = self.venv.step_wait()

        # replace obs with terminal observation if environment is done
        v_obs = [v_infos[env_id].get("terminal_observation", new_v_obs[env_id]) for env_id in range(self.num_envs)]

        for env_idx in range(self.num_envs):
            partial_reward = (
                self.vec_cffns[env_idx].reward(
                    v_obs[env_idx],
                    self._saved_acts[env_idx],
                    v_infos[env_idx]["game_over"],
                    v_infos[env_idx]["awake"],
                )
                if self.vec_cffns[env_idx]
                else 0
            )
            expert_reward = (
                self.vec_expfn[env_idx].reward(
                    v_obs[env_idx],
                    self._saved_acts[env_idx],
                    v_infos[env_idx]["game_over"],
                    v_infos[env_idx]["awake"],
                )
                if self.vec_expfn
                else v_rews[env_idx]
            )
            self.temp_trajectories[env_idx].push_state(
                v_obs[env_idx],
                self._saved_acts[env_idx],
                v_dones[env_idx],
                v_infos[env_idx],
                expert_reward,
                partial_reward,
            )
            if v_dones[env_idx]:
                self.finished_trajectories.append(self.temp_trajectories[env_idx])
                self.temp_trajectories[env_idx] = Trajectory()
                self.vec_cffns[env_idx].reset_prev_shaping()
                self.vec_cffns[env_idx].reward(v_obs[env_idx], 0, False, False)
                if self.vec_expfn:
                    self.vec_expfn[env_idx].reset_prev_shaping()
                    self.vec_expfn[env_idx].reward(v_obs[env_idx], 0, False, False)

        return v_obs, v_rews, v_dones, v_infos

    def pop_trajectories(self):
        trajectories = self.finished_trajectories
        self.finished_trajectories = []
        return trajectories
