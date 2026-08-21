"""Batched reward-model inference must match the per-env path.

``--batch-env-reward-inference`` moves the reward-model forward from each
sub-env wrapper up to a vec-level wrapper that scores all sub-envs at once.
That is an optimization, not a method change, so it is regression-tested on two
levels, because only one of them can be exact:

* **The model inputs are bit-identical.**  This is the real correctness
  property, and it is what would break silently: ``DummyVecEnv`` auto-resets a
  finished env and reports the reset observation, so a vec-level wrapper that
  read ``new_obs`` would score reset frames on every episode boundary.  The
  features are therefore built by each sub-env wrapper, before the auto-reset.

* **The rewards agree to float32 rounding, not bitwise.**  A batched GEMM
  accumulates in a different order than a batch-1 matvec, which moves the last
  ULP (measured: 6e-8 absolute on a plain Linear, half of float32 eps).  The
  arithmetic is the same; the rounding is not.  So a batched run does not
  reproduce an unbatched run seed-for-seed over millions of chaotic RL steps,
  the same way :file:`CLAUDE.md` already records for different machines.

The vmapped ensemble additionally caches stacked member parameters, so it has
to notice when training mutates those weights in place.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rcomp.envs import make_train_env, observation_features
from rcomp.partials import PartialSpec
from rcomp.rewards.model import RewardModel
from rcomp.rewards.wrapper import (
    BatchedRewardDummyVecEnv,
    LearnedRewardRuntime,
    PreferenceRewardWrapper,
)

EPISODE_LENGTH = 4
STEPS = 13
# One float32 ULP of headroom: batching changes accumulation order, nothing else.
FLOAT32_ROUNDING = 1e-6


class _CountingEnv(gym.Env):
    """Terminates every ``EPISODE_LENGTH`` steps, with the step index in the
    observation.  Scoring an auto-reset frame instead of the real post-step
    frame therefore changes the model input, and the comparisons below fail."""

    observation_space = spaces.Box(-1e3, 1e3, shape=(3,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self, offset: int = 0):
        self.offset = float(offset)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0, self.offset, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        observation = np.array([float(self.t), self.offset, float(action)], dtype=np.float32)
        terminated = self.t >= EPISODE_LENGTH
        return observation, float(self.t) + self.offset, terminated, False, {}


def _partial_spec() -> PartialSpec:
    class _Partial:
        def reset(self, info=None):
            return None

        def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
            # Deliberately reads next_obs so a wrong observation would show up.
            return float(np.asarray(next_obs)[0]) * 0.5 - float(action)

    return PartialSpec(name="counting", suite="gym", factory=lambda env_id: _Partial())


def _models(count: int, seed: int = 0) -> list[RewardModel]:
    th.manual_seed(seed)
    models = [
        RewardModel(input_size=6, hidden_sizes=(8,), tanh_output=True, tanh_scale=5.0)
        for _ in range(count)
    ]
    for model in models:
        model.eval()
    return models


def _runtime(models: list[RewardModel], composition: str) -> LearnedRewardRuntime:
    return LearnedRewardRuntime(
        env_id="Counting-v0",
        composition=composition,
        observation_space=_CountingEnv.observation_space,
        action_space=_CountingEnv.action_space,
        observation_features=observation_features,
        custom_partial=_partial_spec(),
        reward_models=models,
        normalize=True,
        output_mean=0.05,
        output_std=0.4,
        normalize_partial=True,
        partial_mean=1.5,
        partial_std=0.8,
        partial_alpha=0.4,
    )


def _record_features(runtime: LearnedRewardRuntime) -> list[np.ndarray]:
    """Capture every model input the runtime builds, in call order."""
    records: list[np.ndarray] = []
    original = type(runtime).model_features

    def recorder(observation, action, partial_reward):
        features = original(runtime, observation, action, partial_reward)
        records.append(np.asarray(features).copy())
        return features

    runtime.model_features = recorder  # instance attribute shadows the method
    return records


def _sequential_env(runtime: LearnedRewardRuntime, n_envs: int) -> DummyVecEnv:
    return DummyVecEnv(
        [lambda i=i: PreferenceRewardWrapper(_CountingEnv(offset=i), runtime) for i in range(n_envs)]
    )


def _batched_env(runtime: LearnedRewardRuntime, n_envs: int, batch_ensemble: bool = False):
    return BatchedRewardDummyVecEnv(
        [
            lambda i=i: PreferenceRewardWrapper(_CountingEnv(offset=i), runtime, defer_model_reward=True)
            for i in range(n_envs)
        ],
        runtime,
        batch_ensemble=batch_ensemble,
    )


def _roll(env, n_envs: int) -> tuple[list[np.ndarray], int]:
    env.reset()
    rewards, dones_seen = [], 0
    rng = np.random.default_rng(1234)
    for _ in range(STEPS):
        actions = rng.integers(0, 2, size=n_envs)
        _, reward, dones, _ = env.step(actions)
        rewards.append(np.asarray(reward).copy())
        dones_seen += int(np.sum(dones))
    return rewards, dones_seen


@pytest.mark.parametrize("composition", ["feedback", "naive", "weighted_sum"])
def test_batched_path_scores_bit_identical_model_inputs(composition):
    """The property that would fail silently: which observation gets scored."""
    n_envs = 4
    models = _models(3)

    sequential_runtime = _runtime(models, composition)
    sequential_features = _record_features(sequential_runtime)
    _, seq_dones = _roll(_sequential_env(sequential_runtime, n_envs), n_envs)

    batched_runtime = _runtime(models, composition)
    batched_features = _record_features(batched_runtime)
    _, batch_dones = _roll(_batched_env(batched_runtime, n_envs), n_envs)

    # The auto-reset path must actually be exercised, or this proves nothing.
    assert seq_dones > 0 and seq_dones == batch_dones
    assert len(sequential_features) == len(batched_features) == STEPS * n_envs
    for index, (left, right) in enumerate(zip(sequential_features, batched_features)):
        np.testing.assert_array_equal(left, right, err_msg=f"model input {index} differs")


@pytest.mark.parametrize("composition", ["feedback", "naive", "weighted_sum"])
def test_batched_rewards_match_the_per_env_path_to_float32(composition):
    n_envs = 4
    models = _models(3)

    sequential, _ = _roll(_sequential_env(_runtime(models, composition), n_envs), n_envs)
    batched, _ = _roll(_batched_env(_runtime(models, composition), n_envs), n_envs)

    for step, (left, right) in enumerate(zip(sequential, batched)):
        np.testing.assert_allclose(
            left, right, rtol=FLOAT32_ROUNDING, atol=FLOAT32_ROUNDING, err_msg=f"step {step}"
        )


def test_vmapped_ensemble_matches_the_sequential_ensemble():
    n_envs = 4
    models = _models(3)

    looped, _ = _roll(_batched_env(_runtime(models, "naive"), n_envs, batch_ensemble=False), n_envs)
    vmapped, _ = _roll(_batched_env(_runtime(models, "naive"), n_envs, batch_ensemble=True), n_envs)

    for step, (left, right) in enumerate(zip(looped, vmapped)):
        np.testing.assert_allclose(
            left, right, rtol=FLOAT32_ROUNDING, atol=FLOAT32_ROUNDING, err_msg=f"step {step}"
        )


def test_vmapped_ensemble_cache_follows_in_place_weight_updates():
    n_envs = 2
    models = _models(3)
    runtime = _runtime(models, "feedback")
    env = _batched_env(runtime, n_envs, batch_ensemble=True)

    before, _ = _roll(env, n_envs)

    # Mutate the members in place, exactly as the training routines do.
    with th.no_grad():
        for model in models:
            model.head.bias.add_(1.0)
    runtime.notify_models_changed()

    after, _ = _roll(env, n_envs)
    assert not np.allclose(before[0], after[0]), "stacked-parameter cache went stale after training"

    # And the refreshed cache still agrees with the plain loop.
    reference, _ = _roll(_batched_env(runtime, n_envs, batch_ensemble=False), n_envs)
    np.testing.assert_allclose(after[0], reference[0], rtol=FLOAT32_ROUNDING, atol=FLOAT32_ROUNDING)


def test_batched_vec_env_rejects_a_sub_env_that_is_not_deferring():
    runtime = _runtime(_models(1), "feedback")
    env = BatchedRewardDummyVecEnv([lambda: PreferenceRewardWrapper(_CountingEnv(), runtime)], runtime)
    env.reset()
    with pytest.raises(RuntimeError, match="no reward-model features"):
        env.step(np.zeros(1, dtype=np.int64))


def test_training_stack_matches_the_eval_stack_for_normalization_sync(tmp_path):
    """SB3 walks the training and eval VecEnv stacks in lockstep.

    ``sync_envs_normalization`` asserts that whenever the training env is a
    VecEnvWrapper the eval env is one too, so batched inference has to be a
    DummyVecEnv *subclass*.  An earlier revision wrapped the vec env instead and
    every run died at the first evaluation.
    """
    from stable_baselines3.common.vec_env import sync_envs_normalization

    from rcomp.envs import make_eval_env

    n_envs = 2
    runtime = _runtime(_models(2), "naive")
    train_env = make_train_env(
        lambda: PreferenceRewardWrapper(_CountingEnv(), runtime, defer_model_reward=True),
        n_envs,
        tmp_path / "monitor",
        normalize=True,
        vec_env_cls=BatchedRewardDummyVecEnv,
        vec_env_kwargs={"runtime": runtime, "batch_ensemble": False},
    )
    eval_env = make_eval_env(lambda env_id: _CountingEnv(), "Counting-v0", train_env)

    train_env.reset()
    train_env.step(np.zeros(n_envs, dtype=np.int64))
    sync_envs_normalization(train_env, eval_env)  # must not raise


def test_batched_vec_env_composes_a_nonzero_training_reward(tmp_path):
    """The deferred placeholder must never survive into the reward PPO sees."""
    n_envs = 2
    runtime = _runtime(_models(2), "naive")
    env = make_train_env(
        lambda: PreferenceRewardWrapper(_CountingEnv(), runtime, defer_model_reward=True),
        n_envs,
        tmp_path / "monitor",
        normalize=False,
        vec_env_cls=BatchedRewardDummyVecEnv,
        vec_env_kwargs={"runtime": runtime, "batch_ensemble": False},
    )
    env.reset()
    for _ in range(EPISODE_LENGTH):
        _, reward, _, infos = env.step(np.zeros(n_envs, dtype=np.int64))
        assert np.all(np.asarray(reward) != 0.0)
        for index, info in enumerate(infos):
            assert info["learned_reward"] == pytest.approx(float(reward[index]), abs=1e-6)
