"""Declarative per-suite specifications.

Each :class:`Suite` bundles everything that differs between the four
environment families: environment discovery, default experiment knobs, PPO
hyperparameters, raw-env construction (including suite-specific wrappers),
the observation-normalization rule, and reward-model feature extraction.

Heavy dependencies (torch, stable-baselines3) are imported lazily inside the
methods that need them so that importing this module — and therefore
``rcomp.config`` and the CLI — stays fast.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LunarLanderSaveInfo(gym.Wrapper):
    """Expose LunarLander terminal flags in ``info``."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        unwrapped = self.env.unwrapped
        info["game_over"] = bool(getattr(unwrapped, "game_over", False))
        lander = getattr(unwrapped, "lander", None)
        info["awake"] = bool(getattr(lander, "awake", False)) if lander is not None else False
        return observation, reward, terminated, truncated, info


class AtariFireResetEnv(gym.Wrapper):
    """Press FIRE after reset and after life loss for games that need it."""

    def __init__(self, env):
        super().__init__(env)
        meanings = list(env.unwrapped.get_action_meanings())
        self.fire_action = meanings.index("FIRE") if "FIRE" in meanings else None
        self.second_action = 2 if self.fire_action is not None and len(meanings) > 2 else None
        self.prev_lives = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.fire_action is None:
            self.prev_lives = int(info.get("lives", 0))
            return observation, info

        observation, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)
            self.prev_lives = int(info.get("lives", 0))
            return observation, info

        if self.second_action is not None:
            observation, _, terminated, truncated, info = self.env.step(self.second_action)
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
                self.prev_lives = int(info.get("lives", 0))
                return observation, info

        self.prev_lives = int(info.get("lives", 0))
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        lives = int(info.get("lives", 0))
        lost_life = self.fire_action is not None and not (terminated or truncated) and 0 < lives < self.prev_lives

        if lost_life:
            observation, extra_reward, terminated, truncated, info = self.env.step(self.fire_action)
            reward += extra_reward
            if self.second_action is not None and not (terminated or truncated):
                observation, extra_reward, terminated, truncated, info = self.env.step(self.second_action)
                reward += extra_reward

        self.prev_lives = int(info.get("lives", 0))
        return observation, reward, terminated, truncated, info


def slugify(env_id: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in env_id.rsplit("-", 1)[0]).strip("_")


def _try_register_atari_envs() -> None:
    try:
        import ale_py
    except ImportError:
        return
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)


class Suite:
    """Generic Gymnasium suite; the other suites override what differs."""

    name = "gym"
    default_log_dir = "logs/gym_ablations"
    default_n_eval_episodes = 5
    default_final_eval_episodes = 10
    default_active_learning = True
    default_preset: str | None = None
    presets: tuple[str, ...] | None = None
    collection_label = "Gym steps"
    curve_x_scale = 1e6
    curve_x_label = "Timesteps (millions)"
    curve_y_floor: float | None = None
    summary_component_keys: tuple[str, ...] = ("total", "partial", "residual")
    wrapper_reset_info: dict[str, float] = {}
    cast_true_reward_info = True

    def supported_envs(self) -> tuple[str, ...]:
        _try_register_atari_envs()
        return tuple(sorted(gym.envs.registry.keys()))

    def default_envs(self) -> tuple[str, ...]:
        return ("CartPole-v1",)

    def default_env_id(self) -> str:
        return "CartPole-v1"

    def collection_defaults(self, env_ids: tuple[str, ...]) -> tuple[int, int]:
        """Default (collection_timesteps, fragment_length) for these envs."""
        return 2000, 1

    def setup(self, config) -> None:
        return None

    def make_raw_env(self, env_id: str) -> gym.Env:
        return gym.make(env_id)

    def should_normalize_observation(self, observation_space: spaces.Space) -> bool:
        return isinstance(observation_space, spaces.Box) and len(observation_space.shape or ()) == 1

    def ppo_hyperparams(self, config, probe_env: gym.Env) -> dict[str, Any]:
        is_image = isinstance(probe_env.observation_space, spaces.Box) and len(probe_env.observation_space.shape or ()) == 3
        hyperparams = {
            "policy": "CnnPolicy" if is_image else "MlpPolicy",
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
        }
        hyperparams.update(deepcopy(config.policy_learning_kwargs or {}))
        return hyperparams

    def observation_features(self, observation_space: spaces.Space, observation) -> np.ndarray:
        from .envs import observation_features

        return observation_features(observation_space, observation)

    def eval_model_observation(self, stats_source, observation):
        from .envs import policy_observation

        return policy_observation(stats_source, observation)

    def slug(self, env_id: str) -> str:
        return slugify(env_id)

    def metadata_component_keys(self, custom_partial) -> list[str]:
        return list(custom_partial.component_keys) if custom_partial is not None else []

    def extra_metadata(self, config) -> dict[str, Any]:
        return {}


class Box2DSuite(Suite):
    name = "box2d"
    default_log_dir = "logs/box2d_ablations"

    _env_prefixes = ("LunarLander", "BipedalWalker", "CarRacing")

    def supported_envs(self) -> tuple[str, ...]:
        return tuple(env_id for env_id in super().supported_envs() if env_id.startswith(self._env_prefixes))

    def default_envs(self) -> tuple[str, ...]:
        supported = self.supported_envs()
        return tuple(env for env in ("LunarLander-v3", "BipedalWalker-v3", "CarRacing-v3") if env in supported)

    def default_env_id(self) -> str:
        defaults = self.default_envs()
        return defaults[0] if defaults else "LunarLander-v3"

    def collection_defaults(self, env_ids: tuple[str, ...]) -> tuple[int, int]:
        if env_ids and all(env_id.startswith("LunarLander") for env_id in env_ids):
            return 10_000, 25
        return 2000, 1

    def make_raw_env(self, env_id: str) -> gym.Env:
        env = gym.make(env_id)
        if env_id.startswith("LunarLander"):
            return LunarLanderSaveInfo(env)
        return env


class MuJoCoSuite(Suite):
    name = "mujoco"
    default_log_dir = "logs/mujoco_ablations"
    default_n_eval_episodes = 10
    default_final_eval_episodes = 50
    default_preset = "auto"
    presets = ("auto", "generic", "reacher")
    collection_label = "steps"
    curve_x_scale = 1e7
    curve_x_label = "Timesteps (1e7)"
    curve_y_floor = -4
    cast_true_reward_info = False

    _env_prefixes = (
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

    def supported_envs(self) -> tuple[str, ...]:
        envs = set(self.default_envs())
        envs.update(env_id for env_id in gym.envs.registry.keys() if env_id.startswith(self._env_prefixes))
        return tuple(sorted(envs))

    def default_envs(self) -> tuple[str, ...]:
        return ("Reacher-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")

    def default_env_id(self) -> str:
        return "Reacher-v5"

    def collection_defaults(self, env_ids: tuple[str, ...]) -> tuple[int, int]:
        return 1500, 1

    def should_normalize_observation(self, observation_space: spaces.Space) -> bool:
        return True

    def ppo_hyperparams(self, config, probe_env: gym.Env) -> dict[str, Any]:
        from torch import nn

        if config.preset == "reacher" or (config.preset == "auto" and config.env_id == "Reacher-v5"):
            hyperparams = {
                "policy": "MlpPolicy",
                "n_steps": 512,
                "batch_size": 32,
                "gamma": 0.9,
                "learning_rate": 0.000104019,
                "ent_coef": 7.52585e-08,
                "clip_range": 0.3,
                "n_epochs": 5,
                "gae_lambda": 1.0,
                "max_grad_norm": 0.9,
                "vf_coef": 0.950368,
                "policy_kwargs": {
                    "log_std_init": -2,
                    "ortho_init": False,
                    "activation_fn": nn.ReLU,
                    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                },
            }
        else:
            hyperparams = {
                "policy": "MlpPolicy",
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
                "learning_rate": 3e-4,
                "ent_coef": 0.0,
                "clip_range": 0.2,
                "n_epochs": 10,
                "gae_lambda": 0.95,
                "max_grad_norm": 0.5,
                "vf_coef": 0.5,
                "policy_kwargs": {
                    "activation_fn": nn.Tanh,
                    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                },
            }
        hyperparams.update(deepcopy(config.policy_learning_kwargs or {}))
        return hyperparams

    def eval_model_observation(self, stats_source, observation):
        from .envs import normalize_obs

        return normalize_obs(stats_source, observation)

    def extra_metadata(self, config) -> dict[str, Any]:
        return {"preset": config.preset}


class AtariSuite(Suite):
    name = "atari"
    default_log_dir = "logs/atari_ablations"
    default_active_learning = False
    collection_label = "Atari steps"
    summary_component_keys = ("total", "partial", "residual", "lost_lives")
    wrapper_reset_info = {"model_reward": 0.0, "learned_reward": 0.0}

    def supported_envs(self) -> tuple[str, ...]:
        envs = set(self.default_envs())
        _try_register_atari_envs()
        envs.update(env_id for env_id in gym.envs.registry.keys() if env_id.startswith("ALE/") and env_id.endswith("-v5"))
        return tuple(sorted(envs))

    def default_envs(self) -> tuple[str, ...]:
        return ("ALE/Breakout-v5", "ALE/Seaquest-v5", "ALE/Qbert-v5", "ALE/SpaceInvaders-v5")

    def default_env_id(self) -> str:
        return "ALE/Breakout-v5"

    def collection_defaults(self, env_ids: tuple[str, ...]) -> tuple[int, int]:
        return 50_000, 64

    def setup(self, config) -> None:
        import random

        import torch as th

        register_atari_envs()
        random.seed(config.seed)
        np.random.seed(config.seed)
        th.manual_seed(config.seed)

    def make_raw_env(self, env_id: str) -> gym.Env:
        register_atari_envs()
        env = gym.make(env_id, obs_type="ram", frameskip=4, repeat_action_probability=0.25)
        return AtariFireResetEnv(env)

    def should_normalize_observation(self, observation_space: spaces.Space) -> bool:
        return True

    def ppo_hyperparams(self, config, probe_env: gym.Env) -> dict[str, Any]:
        from torch import nn

        hyperparams = {
            "policy": "MlpPolicy",
            "n_steps": 128,
            "batch_size": 256,
            "gamma": 0.99,
            "learning_rate": 2.5e-4,
            "ent_coef": 0.01,
            "clip_range": 0.1,
            "n_epochs": 4,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "policy_kwargs": {
                "activation_fn": nn.ReLU,
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        }
        hyperparams.update(deepcopy(config.policy_learning_kwargs or {}))
        return hyperparams

    def observation_features(self, observation_space: spaces.Space, observation) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32).reshape(-1) / 255.0

    def eval_model_observation(self, stats_source, observation):
        from .envs import normalize_obs

        return normalize_obs(stats_source, observation)

    def slug(self, env_id: str) -> str:
        name = env_id.split("/", 1)[-1].rsplit("-", 1)[0]
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")

    def metadata_component_keys(self, custom_partial) -> list[str]:
        partial_keys = custom_partial.component_keys if custom_partial is not None else ()
        return ["total", "partial", "residual", *partial_keys, "length"]

    def extra_metadata(self, config) -> dict[str, Any]:
        return {
            "obs_type": "ram",
            "frameskip": 4,
            "repeat_action_probability": 0.25,
            "fire_reset": True,
            "auto_fire_after_life_loss": True,
            "action_encoding": "one_hot",
        }


def register_atari_envs() -> None:
    try:
        import ale_py
    except ImportError as exc:
        raise RuntimeError("Atari experiments require ale-py. Install it with `pip install ale-py`.") from exc

    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)

    if "ALE/Breakout-v5" not in gym.envs.registry:
        from ale_py.registration import register_v5_envs

        register_v5_envs()


SUITES: dict[str, Suite] = {
    "mujoco": MuJoCoSuite(),
    "atari": AtariSuite(),
    "box2d": Box2DSuite(),
    "gym": Suite(),
}

SUITE_NAMES = tuple(SUITES)


def get_suite(name: str) -> Suite:
    return SUITES[name]
