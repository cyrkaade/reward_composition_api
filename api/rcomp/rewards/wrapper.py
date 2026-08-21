"""The single learned-reward runtime and env wrapper shared by all suites
and reward compositions. Suite differences are injected through the
runtime: the observation-feature function, the info dict emitted at reset,
and whether the true reward is cast to float in ``info``."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.func import functional_call, stack_module_state

from ..envs import action_features
from ..partials import PartialSpec
from .model import RewardModel


PARTIAL_OBSERVATION_KEY = "_partial_observation"


def resolve_torch_device(name: str) -> th.device:
    """Resolve a config device string to a concrete torch device.

    ``auto`` follows torch availability.  ``cuda`` is taken at face value so a
    misconfigured job fails at ``.to()`` rather than silently spending hours on
    the CPU.
    """

    if name == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(name)


def partial_observation_from_info(observation, info: dict, *, consume: bool = False):
    """Return the private observation intended for a hand-written partial.

    Non-Atari environments do not set the key and therefore retain their
    historical behavior.  Atari returns pixels as ``observation`` and puts a
    synchronized RAM snapshot under the private key.
    """

    if consume:
        return info.pop(PARTIAL_OBSERVATION_KEY, observation)
    return info.get(PARTIAL_OBSERVATION_KEY, observation)


def reward_model_features(
    observation_features: Callable[[spaces.Space, Any], np.ndarray],
    observation_space: spaces.Space,
    action_space: spaces.Space,
    observation,
    action,
    partial_reward: float,
    include_partial_feature: bool,
) -> np.ndarray:
    partial_feature = partial_reward if include_partial_feature else 0.0
    return np.concatenate(
        [
            observation_features(observation_space, observation),
            action_features(action_space, action),
            np.asarray([partial_feature], dtype=np.float32),
        ]
    )


@dataclass
class LearnedRewardRuntime:
    env_id: str
    composition: str
    observation_space: spaces.Space
    action_space: spaces.Space
    observation_features: Callable[[spaces.Space, Any], np.ndarray]
    custom_partial: PartialSpec | None = None
    reward_model: RewardModel | None = None
    reward_models: list[RewardModel] | None = None
    model_device: str = "cpu"
    model_generation: int = 0
    output_mean: float | None = None
    output_std: float | None = None
    gate_stats: dict | None = None
    gate_error_stats: dict | None = None
    rm_diagnostics: dict | None = None
    rm_diagnostics_before: dict | None = None
    reward_model_training: list = field(default_factory=list)
    holdout_diagnostics: list = field(default_factory=list)
    query_fisher: list = field(default_factory=list)
    pretrain_stats: dict | None = None
    target_mean: float = 0.0
    target_std: float = 1.0
    reward_min: float | None = None
    reward_max: float | None = None
    reward_scale: float = 1.0
    normalize: bool = False
    normalize_partial: bool = False
    partial_mean: float = 0.0
    partial_std: float = 1.0
    partial_alpha: float = 1.0
    gate_partial: bool = False
    include_partial_feature: bool = True
    reset_info: dict[str, float] = field(default_factory=dict)
    cast_true_reward: bool = True

    def transform_model_output(self, value: float) -> float:
        if self.normalize and self.output_mean is not None and self.output_std is not None:
            value = (value - self.output_mean) / max(self.output_std, 1e-8) * self.target_std + self.target_mean
        value *= self.reward_scale
        if self.reward_min is not None or self.reward_max is not None:
            value = float(np.clip(value, self.reward_min, self.reward_max))
        return value

    def transform_partial_reward(self, value: float) -> float:
        if self.normalize_partial:
            value = (value - self.partial_mean) / max(self.partial_std, 1e-8)
        return value

    def composed_partial_reward(self, value: float) -> float:
        return self.partial_alpha * self.transform_partial_reward(value)

    def model_reward_weight(self) -> float:
        """Coefficient applied to the learned reward in the composition."""
        return 1.0 - self.partial_alpha if self.composition == "weighted_sum" else 1.0

    def model_prediction_scale(self) -> float:
        """Linear scale of raw model returns used by active query scoring.

        All query fragments have equal length, so the affine normalizer's mean
        shift cancels between a pair; only this scale changes preferences.
        """
        scale = self.reward_scale * self.model_reward_weight()
        if self.normalize and self.output_std is not None:
            scale *= self.target_std / max(self.output_std, 1e-8)
        return scale

    def active_reward_models(self) -> list[RewardModel]:
        """Ensemble members currently backing the learned reward, or []."""
        if self.reward_models:
            return list(self.reward_models)
        return [self.reward_model] if self.reward_model is not None else []

    def notify_models_changed(self) -> None:
        """Invalidate any cached view of the ensemble's stacked parameters.

        The training routines mutate the members' weights in place, so a cache
        keyed on object identity would go stale without ever looking stale.
        Callers bump this counter instead; policy training bumps it at every
        entry point, which is sufficient because weights only ever change
        between policy rounds.
        """

        self.model_generation += 1

    def compose_reward(self, partial_reward: float, model_reward: float) -> float:
        if self.composition == "partial":
            return partial_reward
        if self.composition == "feedback":
            return model_reward
        if self.composition in {"naive", "delta"}:
            return self.composed_partial_reward(partial_reward) + model_reward
        if self.composition == "weighted_sum":
            return self.composed_partial_reward(partial_reward) + self.model_reward_weight() * model_reward
        raise ValueError(f"Unsupported reward composition: {self.composition}")

    def model_features(self, observation, action, partial_reward: float) -> np.ndarray:
        return reward_model_features(
            self.observation_features,
            self.observation_space,
            self.action_space,
            observation,
            action,
            self.transform_partial_reward(partial_reward),
            self.include_partial_feature,
        )


class PreferenceRewardWrapper(gym.Wrapper):
    def __init__(self, env, runtime: LearnedRewardRuntime, defer_model_reward: bool = False):
        super().__init__(env)
        self.runtime = runtime
        self.partial = runtime.custom_partial.create(runtime.env_id) if runtime.custom_partial else None
        self._last_partial_obs = None
        # When deferred, this wrapper builds the model input but does not run
        # the model; BatchedModelRewardWrapper scores every sub-env at once and
        # writes the composed reward back at the vec level.
        self.defer_model_reward = bool(defer_model_reward)
        self.pending_model_features: np.ndarray | None = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_partial_obs = partial_observation_from_info(observation, info, consume=True)
        if self.partial is not None:
            self.partial.reset(info)
        info.update(dict(self.runtime.reset_info))
        return observation, info

    def partial_reward(self, previous_obs, action, observation, true_reward, terminated, truncated, info) -> tuple[float, dict]:
        if self.partial is None:
            return 0.0, {}
        step = self.partial.step(previous_obs, action, observation, true_reward, terminated, truncated, info)
        return step.partial, step.components

    def model_reward(self, observation, action, partial_reward: float) -> float:
        reward_models = self.runtime.reward_models or ([self.runtime.reward_model] if self.runtime.reward_model is not None else [])
        if not reward_models:
            return 0.0

        model_input = self.runtime.model_features(observation, action, partial_reward)
        device = resolve_torch_device(self.runtime.model_device)
        with th.no_grad():
            model_tensor = th.as_tensor(model_input, dtype=th.float32, device=device).view(1, -1)
            outputs = th.stack([model(model_tensor).reshape(-1)[0] for model in reward_models])
            output = th.mean(outputs)
        return self.runtime.transform_model_output(float(output.item()))

    def model_output_and_gate(self, observation, action, partial_reward: float) -> tuple[float, float]:
        """Delta output h and per-state gate g(s,a) in [0,1], averaged over the ensemble."""
        reward_models = self.runtime.reward_models or ([self.runtime.reward_model] if self.runtime.reward_model is not None else [])
        if not reward_models:
            return 0.0, 1.0
        model_input = self.runtime.model_features(observation, action, partial_reward)
        device = resolve_torch_device(self.runtime.model_device)
        with th.no_grad():
            model_tensor = th.as_tensor(model_input, dtype=th.float32, device=device).view(1, -1)
            h = th.mean(th.stack([model(model_tensor).reshape(-1)[0] for model in reward_models]))
            g = th.mean(th.stack([model.gate(model_tensor).reshape(-1)[0] for model in reward_models]))
        return self.runtime.transform_model_output(float(h.item())), float(g.item())

    def compose_reward(self, partial_reward: float, model_reward: float) -> float:
        return self.runtime.compose_reward(partial_reward, model_reward)

    def step(self, action):
        previous_partial_obs = self._last_partial_obs
        observation, true_reward, terminated, truncated, info = self.env.step(action)
        partial_observation = partial_observation_from_info(observation, info, consume=True)
        partial_reward, partial_components = self.partial_reward(
            previous_partial_obs,
            action,
            partial_observation,
            true_reward,
            terminated,
            truncated,
            info,
        )
        if self.runtime.gate_partial and self.runtime.composition in {"delta", "naive"}:
            model_reward, gate = self.model_output_and_gate(observation, action, partial_reward)
            # composed_partial_reward applies alpha; the gate then scales it per state.
            # (This path used to call transform_partial_reward directly, silently
            # dropping partial_alpha whenever the gate was enabled.)
            training_reward = gate * self.runtime.composed_partial_reward(partial_reward) + model_reward
            info["gate"] = gate
        elif self.defer_model_reward:
            # Build the features HERE, from this env's genuine post-step
            # observation. DummyVecEnv auto-resets a finished env and reports
            # the reset observation, so a vec-level wrapper reading `new_obs`
            # would score reset frames on every episode boundary.
            self.pending_model_features = self.runtime.model_features(observation, action, partial_reward)
            model_reward = 0.0
            training_reward = 0.0
        else:
            model_reward = self.model_reward(observation, action, partial_reward)
            training_reward = self.compose_reward(partial_reward, model_reward)

        info["true_reward"] = float(true_reward) if self.runtime.cast_true_reward else true_reward
        info["partial_reward"] = partial_reward
        info["partial_components"] = partial_components
        info["model_reward"] = model_reward
        info["learned_reward"] = training_reward
        self._last_partial_obs = partial_observation
        return observation, training_reward, terminated, truncated, info


def preference_reward_wrappers(venv) -> list[PreferenceRewardWrapper]:
    """The per-env :class:`PreferenceRewardWrapper` behind each sub-env."""

    envs = getattr(venv, "envs", None)
    if envs is None:
        raise TypeError("batched reward inference needs a DummyVecEnv-style venv exposing .envs")
    wrappers = []
    for index, env in enumerate(envs):
        current = env
        while current is not None and not isinstance(current, PreferenceRewardWrapper):
            current = getattr(current, "env", None)
        if current is None:
            raise TypeError(f"sub-env {index} is not wrapped in PreferenceRewardWrapper")
        wrappers.append(current)
    return wrappers


class BatchedRewardDummyVecEnv(DummyVecEnv):
    """DummyVecEnv that scores every sub-env in ONE reward-model forward.

    Each :class:`PreferenceRewardWrapper` below runs in deferred mode: it builds
    its own model input from its own post-step observation and stashes it, but
    leaves the reward at zero.  This class stacks those inputs, evaluates the
    ensemble once, and writes the composed reward back.  Only the batch
    dimension changes, so the rewards PPO sees match the per-env path to float32
    rounding (a batched GEMM accumulates in a different order than a batch-1
    matvec; the arithmetic is the same).

    Two structural notes:

    * It **subclasses** DummyVecEnv rather than wrapping it.  A VecEnvWrapper
      would deepen the training env's stack, and SB3's
      ``sync_envs_normalization`` walks the training and eval stacks in lockstep
      and asserts they have the same shape.

    * Each sub-env's ``Monitor`` sits inside the vec env, so it records the
      deferred placeholder (0.0) rather than the composed training reward:
      ``rollout/ep_rew_mean`` and ``monitor/*.csv`` are not meaningful under this
      flag.  Nothing in this project reads them -- evaluation goes through
      ``make_raw_env`` on the true reward, and the analyzers read
      ``metadata.json`` and ``eval/evaluations.npz``.  The composed reward is
      still published per step in ``info["learned_reward"]``.
    """

    def __init__(self, env_fns, runtime: LearnedRewardRuntime, batch_ensemble: bool = False):
        super().__init__(env_fns)
        self.runtime = runtime
        self.batch_ensemble = bool(batch_ensemble)
        self.sub_wrappers = preference_reward_wrappers(self)
        self._stacked_generation = -1
        self._stacked: tuple | None = None

    def reset(self):
        for wrapper in self.sub_wrappers:
            wrapper.pending_model_features = None
        return super().reset()

    def step_wait(self):
        observations, rewards, dones, infos = super().step_wait()
        features = []
        for index, wrapper in enumerate(self.sub_wrappers):
            pending = wrapper.pending_model_features
            if pending is None:
                raise RuntimeError(
                    f"sub-env {index} produced no reward-model features; BatchedRewardDummyVecEnv "
                    "requires PreferenceRewardWrapper(..., defer_model_reward=True)"
                )
            features.append(pending)
            wrapper.pending_model_features = None

        model_rewards = self.model_rewards(np.stack(features))
        for index, info in enumerate(infos):
            model_reward = model_rewards[index]
            composed = self.runtime.compose_reward(float(info.get("partial_reward", 0.0)), model_reward)
            info["model_reward"] = model_reward
            info["learned_reward"] = composed
            rewards[index] = composed
        return observations, rewards, dones, infos

    def model_rewards(self, features: np.ndarray) -> list[float]:
        reward_models = self.runtime.active_reward_models()
        if not reward_models:
            return [0.0] * len(features)
        device = resolve_torch_device(self.runtime.model_device)
        with th.no_grad():
            tensor = th.as_tensor(features, dtype=th.float32, device=device)
            if self.batch_ensemble and len(reward_models) > 1:
                outputs = self.stacked_outputs(reward_models, tensor)
            else:
                outputs = th.stack([model(tensor).reshape(-1) for model in reward_models])
            means = th.mean(outputs, dim=0).cpu().numpy()
        return [self.runtime.transform_model_output(float(value)) for value in means]

    def stacked_outputs(self, reward_models: list[RewardModel], tensor: th.Tensor) -> th.Tensor:
        """One vmapped forward over stacked member parameters."""
        if self._stacked is None or self._stacked_generation != self.runtime.model_generation:
            params, buffers = stack_module_state(reward_models)
            base = deepcopy(reward_models[0]).to("meta")
            self._stacked = (base, params, buffers)
            self._stacked_generation = self.runtime.model_generation
        base, params, buffers = self._stacked

        def call(member_params, member_buffers, batch):
            return functional_call(base, (member_params, member_buffers), (batch,))

        stacked = th.vmap(call, in_dims=(0, 0, None))(params, buffers, tensor)
        return stacked.reshape(len(reward_models), -1)
