from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from local_gym.classes.atari_reward_specs import AtariRewardSpec, get_atari_reward_spec
from local_gym.wrappers.buffering_wrapper import Trajectory
from reward_model.reward_model import RewardModel
from reward_composition_api.config import ExperimentConfig
from reward_composition_api.registry import PartialSpec
from reward_composition_api.results import RunResult

from .atari_evaluation import (
    AtariComponentEvalCallback,
    _component_keys,
    evaluate_atari_components,
    write_atari_component_summary,
)
from .common import (
    SaveVecNormalizeOnBest,
    include_partial_feature,
    learn_policy,
    load_vecnormalize_eval_env,
    make_raw_eval_env as make_common_raw_eval_env,
    normalize_obs,
    resolve_custom_partial,
)
from .rlhf import RlhfTrainer
from .reporting import (
    BackendRunPaths,
    report_eval_curve,
    select_final_policy,
)


GENERIC_ATARI_RAM_PPO_PRESET = {
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


@dataclass
class AtariLearnedRewardRuntime:
    spec: AtariRewardSpec
    composition: str
    action_n: int
    partial_source: str = "life_loss"
    custom_partial: PartialSpec | None = None
    reward_model: RewardModel | None = None
    output_mean: float | None = None
    output_std: float | None = None
    target_mean: float = 0.0
    target_std: float = 1.0
    reward_min: float | None = None
    reward_max: float | None = None
    reward_scale: float = 1.0
    normalize: bool = False
    include_partial_feature: bool = True


class AtariPreferenceRewardWrapper(gym.Wrapper):
    def __init__(self, env, runtime: AtariLearnedRewardRuntime):
        super().__init__(env)
        self.runtime = runtime
        self.tracker = runtime.spec.new_tracker()
        self.partial = runtime.custom_partial.create(runtime.spec.env_id) if runtime.custom_partial else None
        self._last_obs = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_obs = observation
        step = self.tracker.reset(info)
        if self.partial is not None:
            self.partial.reset(info)
        info.update(step.as_info())
        info["model_reward"] = 0.0
        info["learned_reward"] = 0.0
        return observation, info

    def _partial_reward(self, previous_obs, action, observation, true_reward, terminated, truncated, info):
        if self.partial is not None:
            step = self.partial.step(previous_obs, action, observation, true_reward, terminated, truncated, info)
            return step.partial, step.components
        step = self.tracker.step(info, true_reward=float(true_reward), partial_source=self.runtime.partial_source)
        info.update(step.as_info())
        return step.partial, {
            "life_loss_penalty": step.life_loss_penalty,
            "score_partial": step.score_partial,
            "lost_lives": step.lost_lives,
            "lives": step.lives,
        }

    def _model_reward(self, observation, action, partial_reward):
        if self.runtime.reward_model is None:
            return 0.0

        partial_feature = partial_reward if self.runtime.include_partial_feature else 0.0
        model_input = np.concatenate(
            [
                atari_observation_features(observation),
                one_hot_action(action, self.runtime.action_n),
                np.asarray([partial_feature], dtype=np.float32),
            ]
        )
        with th.no_grad():
            output = self.runtime.reward_model(th.as_tensor(model_input, dtype=th.float32).view(1, -1)).reshape(-1)[0]

        value = float(output.item())
        if self.runtime.normalize and self.runtime.output_mean is not None and self.runtime.output_std is not None:
            value = (
                (value - self.runtime.output_mean)
                / max(self.runtime.output_std, 1e-8)
                * self.runtime.target_std
                + self.runtime.target_mean
            )
        value *= self.runtime.reward_scale
        if self.runtime.reward_min is not None or self.runtime.reward_max is not None:
            value = float(np.clip(value, self.runtime.reward_min, self.runtime.reward_max))
        return value

    def step(self, action):
        previous_obs = self._last_obs
        observation, true_reward, terminated, truncated, info = self.env.step(action)
        partial_reward, partial_components = self._partial_reward(
            previous_obs,
            action,
            observation,
            true_reward,
            terminated,
            truncated,
            info,
        )
        model_reward = self._model_reward(observation, action, partial_reward)

        if self.runtime.composition == "partial":
            training_reward = partial_reward
        elif self.runtime.composition == "feedback":
            training_reward = model_reward
        elif self.runtime.composition in {"naive", "delta"}:
            training_reward = partial_reward + model_reward
        else:
            raise ValueError(f"Unsupported Atari reward composition: {self.runtime.composition}")

        info["true_reward"] = float(true_reward)
        info["partial_reward"] = partial_reward
        info["partial_components"] = partial_components
        info["model_reward"] = model_reward
        info["learned_reward"] = training_reward
        self._last_obs = observation
        return observation, training_reward, terminated, truncated, info


class AtariFireResetEnv(gym.Wrapper):
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


def run_atari_experiment(config: ExperimentConfig) -> RunResult:
    register_atari_envs()
    random.seed(config.seed)
    np.random.seed(config.seed)
    th.manual_seed(config.seed)

    spec = get_atari_reward_spec(config.env_id)
    custom_partial = _resolve_custom_partial(config)
    run_name = config.run_name or default_run_name(config, spec)
    variant_name = config.variant_name or config.mode
    config = replace(config, run_name=run_name, variant_name=variant_name)
    if config.mode in {"true", "partial"}:
        return train_true_or_partial(config, spec, custom_partial)
    return train_preference_mode(config, spec, custom_partial)


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


def atari_observation_features(observation) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1) / 255.0


def one_hot_action(action, action_n: int) -> np.ndarray:
    action_index = int(np.asarray(action).reshape(-1)[0])
    if action_index < 0 or action_index >= action_n:
        raise ValueError(f"Action index {action_index} is outside action space size {action_n}")
    features = np.zeros(action_n, dtype=np.float32)
    features[action_index] = 1.0
    return features


def make_raw_env(env_id: str):
    register_atari_envs()
    env = gym.make(env_id, obs_type="ram", frameskip=4, repeat_action_probability=0.25)
    return AtariFireResetEnv(env)


def ppo_hyperparams(config: ExperimentConfig):
    hyperparams = deepcopy(GENERIC_ATARI_RAM_PPO_PRESET)
    hyperparams.update(config.policy_learning_kwargs or {})
    return hyperparams


def make_raw_eval_env(env_id: str):
    return make_common_raw_eval_env(make_raw_env, env_id)


def make_vecnormalize_env(env_fn, n_envs: int, monitor_dir: Path) -> VecNormalize:
    env = make_vec_env(
        env_fn,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
        monitor_dir=str(monitor_dir),
    )
    return VecNormalize(env, norm_obs=True, norm_reward=True)


def make_eval_env(env_id: str, stats_source: VecNormalize | None = None) -> VecNormalize:
    env = VecNormalize(make_raw_eval_env(env_id), norm_obs=True, norm_reward=False, training=False)
    if stats_source is not None:
        env.obs_rms = stats_source.obs_rms
        env.ret_rms = stats_source.ret_rms
    return env


def load_eval_env(env_id: str, stats_path: Path) -> VecNormalize:
    return load_vecnormalize_eval_env(env_id, stats_path, make_raw_eval_env)


def make_trajectory_converter(action_n: int, include_partial_feature: bool):
    def convert(trajectory: Trajectory):
        rows = []
        for state in trajectory.states:
            partial_feature = state["partial_rew"] if include_partial_feature else 0.0
            rows.append(
                [
                    *atari_observation_features(state["obs"]).tolist(),
                    *one_hot_action(state["act"], action_n).tolist(),
                    float(partial_feature),
                ]
            )
        return rows

    return convert


def collect_policy_trajectories(
    model: PPO,
    stats_source,
    env_id: str,
    spec: AtariRewardSpec,
    partial_source: str,
    custom_partial: PartialSpec | None,
    total_timesteps: int,
    seed: int,
) -> list[Trajectory]:
    env = make_raw_env(env_id)
    partial = custom_partial.create(env_id) if custom_partial else None
    trajectories = []
    trajectory = Trajectory()
    obs, info = env.reset(seed=seed)
    tracker = spec.new_tracker()
    tracker.reset(info)
    if partial is not None:
        partial.reset(info)
    steps = 0

    try:
        while steps < total_timesteps:
            model_obs = normalize_obs(stats_source, obs)
            action, _ = model.predict(model_obs, deterministic=False)
            action = int(np.asarray(action).reshape(-1)[0])
            new_obs, true_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if partial is None:
                partial_reward = tracker.step(info, true_reward=float(true_reward), partial_source=partial_source).partial
            else:
                partial_reward = partial.step(obs, action, new_obs, true_reward, terminated, truncated, info).partial
            trajectory.push_state(new_obs, action, done, info, float(true_reward), partial_reward)
            steps += 1

            if done:
                trajectories.append(trajectory)
                trajectory = Trajectory()
                obs, info = env.reset()
                tracker.reset(info)
                if partial is not None:
                    partial.reset(info)
            else:
                obs = new_obs
    finally:
        env.close()

    if trajectory.states:
        trajectories.append(trajectory)
    return trajectories


def build_callbacks(
    config: ExperimentConfig,
    run_dir: Path,
    train_env: VecNormalize,
    eval_env: VecNormalize,
    spec: AtariRewardSpec,
    custom_partial: PartialSpec | None,
):
    eval_freq = max(config.eval_freq // config.n_envs, 1)
    best_stats_path = run_dir / "best_model" / "best_vecnormalize.pkl"
    best_callbacks: list[BaseCallback] = [SaveVecNormalizeOnBest(train_env, best_stats_path)]
    if config.stop_reward is not None:
        best_callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=config.stop_reward, verbose=1))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=CallbackList(best_callbacks),
    )
    component_callback = AtariComponentEvalCallback(
        run_dir / "eval" / "component_evaluations.csv",
        config.env_id,
        spec,
        make_env=make_raw_env,
        partial_source=config.partial_source,
        custom_partial=custom_partial,
        eval_freq=eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        verbose=1,
    )
    return CallbackList([eval_callback, component_callback])


def train_true_or_partial(config: ExperimentConfig, spec: AtariRewardSpec, custom_partial: PartialSpec | None) -> RunResult:
    run_dir = Path(config.log_dir) / config.run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    if config.mode == "true":
        env_fn = lambda: make_raw_env(config.env_id)
    elif config.mode == "partial":
        probe_env = make_raw_env(config.env_id)
        action_n = int(probe_env.action_space.n)
        probe_env.close()
        runtime = AtariLearnedRewardRuntime(
            spec=spec,
            composition="partial",
            action_n=action_n,
            partial_source=config.partial_source,
            custom_partial=custom_partial,
        )
        env_fn = lambda: AtariPreferenceRewardWrapper(make_raw_env(config.env_id), runtime)
    else:
        raise ValueError(f"Unsupported mode for this path: {config.mode}")

    train_env = make_vecnormalize_env(env_fn, config.n_envs, run_dir / "monitor")
    eval_env = make_eval_env(config.env_id, train_env)
    callbacks = build_callbacks(config, run_dir, train_env, eval_env, spec, custom_partial)

    model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **ppo_hyperparams(config))
    learn_policy(
        model,
        config.timesteps,
        callbacks,
        progress_bar=config.progress_bar,
        log_interval=config.policy_log_interval,
    )

    return save_and_report(config, model, train_env, eval_env, run_dir, spec, synthetic_queries=0, custom_partial=custom_partial)


def train_preference_mode(config: ExperimentConfig, spec: AtariRewardSpec, custom_partial: PartialSpec | None) -> RunResult:
    run_dir = Path(config.log_dir) / config.run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    probe_env = make_raw_env(config.env_id)
    obs_size = int(np.prod(probe_env.observation_space.shape))
    action_n = int(probe_env.action_space.n)
    probe_env.close()

    runtime = AtariLearnedRewardRuntime(
        spec=spec,
        composition=config.mode,
        action_n=action_n,
        partial_source=config.partial_source,
        custom_partial=custom_partial,
        target_mean=config.model_reward_target_mean,
        target_std=config.model_reward_target_std,
        reward_min=config.model_reward_min,
        reward_max=config.model_reward_max,
        reward_scale=config.model_reward_scale,
        normalize=config.normalize_model_reward,
        include_partial_feature=include_partial_feature(config),
    )
    train_env = make_vecnormalize_env(
        lambda: AtariPreferenceRewardWrapper(make_raw_env(config.env_id), runtime),
        config.n_envs,
        run_dir / "monitor",
    )
    eval_env = make_eval_env(config.env_id, train_env)
    callbacks = build_callbacks(config, run_dir, train_env, eval_env, spec, custom_partial)
    model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **ppo_hyperparams(config))

    reward_model = RewardModel(input_size=obs_size + action_n + 1, hidden_sizes=config.reward_hidden_sizes)
    convert_traj = make_trajectory_converter(action_n, runtime.include_partial_feature)
    total_queries = RlhfTrainer(
        config,
        model,
        runtime,
        callbacks,
        reward_model,
        convert_traj,
        lambda round_index, collection_steps: collect_policy_trajectories(
            model,
            train_env,
            env_id=config.env_id,
            spec=spec,
            partial_source=config.partial_source,
            custom_partial=custom_partial,
            total_timesteps=collection_steps,
            seed=config.seed * 1000 + round_index * 100,
        ),
        continuous=False,
        collection_label="Atari steps",
    ).run()

    return save_and_report(
        config,
        model,
        train_env,
        eval_env,
        run_dir,
        spec,
        synthetic_queries=total_queries,
        runtime=runtime,
        custom_partial=custom_partial,
    )


def save_and_report(
    config: ExperimentConfig,
    model: PPO,
    train_env: VecNormalize,
    eval_env: VecNormalize,
    run_dir: Path,
    spec: AtariRewardSpec,
    synthetic_queries: int,
    runtime: AtariLearnedRewardRuntime | None = None,
    custom_partial: PartialSpec | None = None,
) -> RunResult:
    paths = BackendRunPaths(run_dir)
    model.save(paths.final_model)
    train_env.save(paths.vecnormalize)

    actual_timesteps = int(model.num_timesteps)
    best_logged_reward, best_logged_timestep = report_eval_curve(
        paths.eval_log,
        paths.true_reward_curve,
        max(config.timesteps, actual_timesteps),
        config.plot_mode,
        config.smooth_window,
        x_scale=1e6,
        x_label="Timesteps (millions)",
        y_floor=None,
    )

    final_stats = evaluate_atari_components(
        model,
        config.env_id,
        spec,
        make_env=make_raw_env,
        partial_source=config.partial_source,
        custom_partial=custom_partial,
        stats_source=train_env,
        n_eval_episodes=config.final_eval_episodes,
        seed=config.seed + 50_000,
    )
    write_atari_component_summary(paths.final_component_evaluation, actual_timesteps, final_stats, custom_partial)

    final_policy, final_eval_env = select_final_policy(
        config,
        model,
        eval_env,
        run_dir,
        load_eval_env,
        PPO.load,
        load_best_stats=True,
    )

    mean_reward, std_reward = evaluate_policy(
        final_policy,
        final_eval_env,
        n_eval_episodes=config.final_eval_episodes,
        deterministic=True,
        return_episode_rewards=False,
    )
    selected_stats = evaluate_atari_components(
        final_policy,
        config.env_id,
        spec,
        make_env=make_raw_env,
        partial_source=config.partial_source,
        custom_partial=custom_partial,
        stats_source=final_eval_env,
        n_eval_episodes=config.final_eval_episodes,
        seed=config.seed + 60_000,
    )

    metadata = {
        "env_id": config.env_id,
        "env_slug": spec.slug,
        "mode": config.mode,
        "run_name": config.run_name,
        "variant": config.variant_name,
        "requested_timesteps": config.timesteps,
        "actual_timesteps": actual_timesteps,
        "seed": config.seed,
        "n_envs": config.n_envs,
        "initial_timesteps": config.initial_timesteps,
        "policy_timesteps_per_round": config.policy_timesteps_per_round,
        "final_policy_timesteps": config.final_policy_timesteps,
        "policy_learning_kwargs": config.policy_learning_kwargs or {},
        "obs_type": "ram",
        "frameskip": 4,
        "repeat_action_probability": 0.25,
        "fire_reset": True,
        "auto_fire_after_life_loss": True,
        "action_encoding": "one_hot",
        "partial_source": config.partial_source,
        "partial_reference": config.partial,
        "synthetic_queries": synthetic_queries,
        "query_budget": config.query_budget if config.mode in {"feedback", "naive", "delta"} else 0,
        "fragment_length": config.fragment_length if config.mode in {"feedback", "naive", "delta"} else None,
        "active_learning": config.active_learning if config.mode in {"feedback", "naive", "delta"} else None,
        "reward_hidden_sizes": list(config.reward_hidden_sizes),
        "reward_model_lr": config.reward_model_lr if config.mode in {"feedback", "naive", "delta"} else None,
        "pretrain_reward_model": config.pretrain_reward_model if config.mode in {"feedback", "naive", "delta"} else None,
        "pretrain_target": config.pretrain_target if config.pretrain_reward_model else None,
        "include_partial_feature": include_partial_feature(config) if config.mode in {"feedback", "naive", "delta"} else None,
        "partial_keys": _partial_keys(config, custom_partial),
        "component_keys": list(_component_keys(custom_partial)),
        "life_loss_penalty_weight": spec.life_loss_penalty,
        "best_logged_true_reward": best_logged_reward,
        "best_logged_timestep": best_logged_timestep,
        "selected_policy_true_reward_mean": float(mean_reward),
        "selected_policy_true_reward_std": float(std_reward),
        "selected_policy_components": selected_stats,
    }
    if runtime is not None:
        metadata.update(
            {
                "model_reward_min": runtime.reward_min,
                "model_reward_max": runtime.reward_max,
                "model_reward_scale": runtime.reward_scale,
                "normalize_model_reward": runtime.normalize,
                "model_reward_output_mean": runtime.output_mean,
                "model_reward_output_std": runtime.output_std,
                "model_reward_target_mean": runtime.target_mean,
                "model_reward_target_std": runtime.target_std,
                "reward_composition": runtime.composition,
            }
        )

    metadata_path = paths.metadata
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"{config.final_policy.title()} deterministic true reward: {mean_reward:.3f} +/- {std_reward:.3f}")
    print(
        "Component means: "
        f"total={selected_stats['mean_total']:.3f}, "
        f"partial={selected_stats['mean_partial']:.3f}, "
        f"residual={selected_stats['mean_residual']:.3f}, "
        f"lost_lives={selected_stats.get('mean_lost_lives', 0.0):.3f}"
    )
    print(f"Synthetic queries consumed: {synthetic_queries}")
    print(f"Saved model and logs to {run_dir}")

    train_env.close()
    final_eval_env.close()
    return RunResult(
        run_dir=run_dir,
        metadata_path=metadata_path,
        model_path=paths.final_model.with_suffix(".zip"),
        vecnormalize_path=paths.vecnormalize,
        synthetic_queries=synthetic_queries,
        metadata=metadata,
    )


def default_run_name(config: ExperimentConfig, spec: AtariRewardSpec) -> str:
    variant = config.variant_name or config.mode
    steps = f"{config.timesteps // 1_000_000}m" if config.timesteps >= 1_000_000 else f"{config.timesteps}"
    return f"{spec.slug}_{variant}_{steps}_seed{config.seed}"


def _partial_keys(config: ExperimentConfig, custom_partial: PartialSpec | None):
    if custom_partial is not None:
        return [custom_partial.name]
    if config.partial_source == "life_loss":
        return ["life_loss_penalty"]
    return ["life_loss_penalty", "score_partial"]


def _resolve_custom_partial(config: ExperimentConfig) -> PartialSpec | None:
    return resolve_custom_partial(config)
