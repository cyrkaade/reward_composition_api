"""Regression tests for --unhealthy-penalty.

The properties a future change could silently break, in order of how much damage
breaking them would do:

  * default off leaves the environment and the metadata exactly as they are;
  * the flag never reaches an EVALUATION env, so a modified run is still scored
    on the standard environment every other arm is scored on;
  * forcing terminated=False is genuinely equivalent to the gymnasium
    constructor argument terminate_when_unhealthy=False, rather than merely
    similar to it;
  * info["reward_survive"] carries the replacement, because the hand-written
    priors read their survive term from that key and would otherwise get the
    termination removal without the penalty.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rcomp.config import ConfigError, ExperimentConfig, normalize_experiment_config
from rcomp.envs import UnhealthyPenaltyWrapper, apply_unhealthy_penalty

pytest.importorskip("mujoco")

HEALTHY_ENVS = ["Walker2d-v5", "Ant-v5"]


def _rollout(env, actions):
    out = []
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        out.append((np.asarray(observation, dtype=np.float64), float(reward), bool(terminated), bool(truncated), dict(info)))
        if truncated:
            break
    return out


def _actions(env, n, seed=0):
    rng = np.random.default_rng(seed)
    low, high = env.action_space.low, env.action_space.high
    return [rng.uniform(low, high).astype(np.float32) for _ in range(n)]


@pytest.mark.parametrize("env_id", HEALTHY_ENVS)
def test_forcing_terminated_matches_terminate_when_unhealthy_false(env_id):
    """The wrapper's termination change is the constructor argument, not an approximation."""
    wrapped = UnhealthyPenaltyWrapper(gym.make(env_id), penalty=1.0)
    constructor = gym.make(env_id, terminate_when_unhealthy=False)
    wrapped.reset(seed=11)
    constructor.reset(seed=11)

    actions = _actions(constructor, 300)
    a = _rollout(wrapped, actions)
    b = _rollout(constructor, actions)
    assert len(a) == len(b)

    for (obs_a, rew_a, term_a, _, info_a), (obs_b, rew_b, term_b, _, info_b) in zip(a, b):
        np.testing.assert_allclose(obs_a, obs_b, atol=0, rtol=0)
        assert term_a is False and term_b is False
        # The only reward difference is the healthy term being swapped for +-1.
        replacement = 1.0 if info_a["reward_survive"] > 0 else -1.0
        assert rew_a == pytest.approx(rew_b - info_b["reward_survive"] + replacement, abs=1e-9)

    # The rollout has to actually visit the unhealthy region or the test proves nothing.
    assert any(info["reward_survive"] < 0 for *_, info in a)
    wrapped.close()
    constructor.close()


@pytest.mark.parametrize("env_id", HEALTHY_ENVS)
def test_penalty_replaces_the_healthy_bonus_in_reward_and_info(env_id):
    standard = gym.make(env_id)
    wrapped = UnhealthyPenaltyWrapper(gym.make(env_id), penalty=2.5)
    standard.reset(seed=11)
    wrapped.reset(seed=11)

    seen_healthy = seen_unhealthy = False
    for action in _actions(standard, 500):
        _, base_reward, _, _, base_info = standard.step(action)
        _, reward, terminated, _, info = wrapped.step(action)
        assert terminated is False
        healthy = base_info["reward_survive"] > 0
        expected = 2.5 if healthy else -2.5
        assert info["reward_survive"] == pytest.approx(expected)
        assert info["unhealthy_penalty"] == pytest.approx(2.5)
        assert reward == pytest.approx(base_reward - base_info["reward_survive"] + expected, abs=1e-9)
        seen_healthy |= healthy
        seen_unhealthy |= not healthy
    assert seen_healthy and seen_unhealthy
    standard.close()
    wrapped.close()


@pytest.mark.parametrize("env_id", HEALTHY_ENVS)
def test_timelimit_truncation_survives_the_wrapper(env_id):
    """gym.make puts TimeLimit inside the wrapper, so episodes still end at 1000."""
    env = UnhealthyPenaltyWrapper(gym.make(env_id), penalty=1.0)
    env.reset(seed=5)
    truncated = False
    for step in range(1001):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert terminated is False
        if truncated:
            break
    assert truncated
    assert step == 999
    env.close()


def test_partial_sees_the_penalty_through_info():
    """sel_walker2d reads its survive term from info, so the partial arm gets the same treatment."""
    from rcomp.partials import PartialRegistry, load_partial_reference

    spec = load_partial_reference("sel_walker2d:swk_cap15", "mujoco", PartialRegistry())
    partial = spec.create("Walker2d-v5")

    env = UnhealthyPenaltyWrapper(gym.make("Walker2d-v5"), penalty=1.0)
    obs, _ = env.reset(seed=17)
    partial.reset()
    survive_terms = []
    for action in _actions(env, 300):
        next_obs, reward, terminated, truncated, info = env.step(action)
        step = partial.step(obs, action, next_obs, reward, terminated, truncated, info)
        survive_terms.append(step.components["survive"])
        obs = next_obs
        if truncated:
            break
    assert min(survive_terms) == pytest.approx(-1.0)
    assert max(survive_terms) == pytest.approx(1.0)
    env.close()


def test_default_off_is_a_no_op():
    env = gym.make("Walker2d-v5")
    assert apply_unhealthy_penalty(env, None) is env
    env.close()

    wrapped = apply_unhealthy_penalty(gym.make("Walker2d-v5"), 1.0)
    assert isinstance(wrapped, UnhealthyPenaltyWrapper)
    wrapped.close()


def test_rejects_an_env_without_a_health_rule():
    env = gym.make("HalfCheetah-v5")
    with pytest.raises(ValueError, match="is_healthy"):
        UnhealthyPenaltyWrapper(env, penalty=1.0)
    env.close()


def test_config_rejects_a_non_positive_penalty():
    for value in (0.0, -1.0):
        config = ExperimentConfig(suite="mujoco", env_id="Walker2d-v5", mode="true", unhealthy_penalty=value)
        with pytest.raises(ConfigError, match="unhealthy_penalty"):
            normalize_experiment_config(config)


def test_trainer_applies_the_penalty_to_training_but_not_to_evaluation():
    """The one property the whole design rests on: eval stays the standard env."""
    from rcomp.suites import get_suite
    from rcomp.trainer import ExperimentRunner

    config = normalize_experiment_config(
        ExperimentConfig(suite="mujoco", env_id="Walker2d-v5", mode="true", unhealthy_penalty=1.0)
    )
    runner = ExperimentRunner(config)

    train_env = runner.make_train_raw_env()
    assert isinstance(train_env, UnhealthyPenaltyWrapper)
    train_env.close()

    eval_env = get_suite(config.suite).make_raw_env(config.env_id)
    assert not isinstance(eval_env, UnhealthyPenaltyWrapper)
    eval_env.close()

    off = normalize_experiment_config(ExperimentConfig(suite="mujoco", env_id="Walker2d-v5", mode="true"))
    plain = ExperimentRunner(off).make_train_raw_env()
    assert not isinstance(plain, UnhealthyPenaltyWrapper)
    plain.close()
