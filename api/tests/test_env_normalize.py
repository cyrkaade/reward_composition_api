"""--env-normalize overrides the suite's VecNormalize decision.

Normalization was previously a fixed property of the suite (MuJoCo: always on),
so an arm could not be run without it. That made "SB3 out-of-the-box" not
actually out-of-the-box, and left no way to check whether a tuned preset's
advantage depends on normalization at all.

The default is 'auto' - the suite decides, exactly as every archived run did.
"""

from __future__ import annotations

import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rcomp.config import ConfigError, ExperimentConfig, normalize_experiment_config
from rcomp.envs import make_train_env
from rcomp.trainer import ExperimentRunner


def runner(**overrides):
    config = normalize_experiment_config(
        ExperimentConfig(suite="mujoco", env_id="Reacher-v5", mode="true", **overrides)
    )
    return ExperimentRunner(config)


def resolve(runner_obj) -> bool:
    _, _, normalize, _ = runner_obj.probe_spaces()
    return normalize


def test_default_is_auto_and_keeps_the_suite_decision():
    """MuJoCo normalizes; 'auto' must not change that, or ~6,000 archived runs
    stop being comparable with new ones."""
    run = runner()
    assert run.config.env_normalize == "auto"
    assert resolve(run) is True


def test_off_forces_normalization_away_on_a_suite_that_wants_it():
    run = runner(env_normalize="off")
    assert resolve(run) is False


def test_on_forces_normalization():
    run = runner(env_normalize="on")
    assert resolve(run) is True


def test_resolved_value_is_recorded_not_just_the_request():
    """metadata carries what actually happened. 'auto' alone is unreadable after
    the fact, because its meaning depends on the suite and the obs space."""
    run = runner(env_normalize="off")
    assert run.normalize_applied is None  # not resolved until probed
    resolve(run)
    assert run.normalize_applied is False


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ConfigError):
        normalize_experiment_config(
            ExperimentConfig(suite="mujoco", env_id="Reacher-v5", mode="true", env_normalize="sometimes")
        )


def test_make_train_env_actually_wraps_only_when_asked(tmp_path):
    """The flag is only meaningful if the wrapper follows it."""
    import gymnasium as gym

    env_fn = lambda: gym.make("Pendulum-v1")
    wrapped = make_train_env(env_fn, n_envs=1, monitor_dir=tmp_path, normalize=True)
    bare = make_train_env(env_fn, n_envs=1, monitor_dir=tmp_path, normalize=False)
    try:
        assert isinstance(wrapped, VecNormalize)
        assert not isinstance(bare, VecNormalize)
        assert isinstance(bare, DummyVecEnv)
    finally:
        wrapped.close()
        bare.close()
