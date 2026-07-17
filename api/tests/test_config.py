from __future__ import annotations

from pathlib import Path

import pytest

from rcomp.config import (
    ConfigError,
    ExperimentConfig,
    SweepConfig,
    normalize_experiment_config,
    normalize_sweep_config,
    suite_default_envs,
)


def test_gym_suite_defaults():
    config = normalize_experiment_config(ExperimentConfig(suite="gym", mode="true"))

    assert config.env_id == "CartPole-v1"
    assert config.log_dir == Path("logs/gym_ablations")
    assert config.n_eval_episodes == 5
    assert config.final_eval_episodes == 10
    assert config.collection_timesteps == 2000
    assert config.fragment_length == 1
    assert config.active_learning is True
    assert config.preset is None


def test_box2d_lunar_lander_defaults():
    config = normalize_experiment_config(ExperimentConfig(suite="box2d", env_id="LunarLander-v3", mode="true"))

    assert config.log_dir == Path("logs/box2d_ablations")
    assert config.collection_timesteps == 10_000
    assert config.fragment_length == 25
    assert config.n_eval_episodes == 5
    assert config.final_eval_episodes == 10


def test_mujoco_suite_defaults():
    config = normalize_experiment_config(ExperimentConfig(suite="mujoco", mode="true"))

    assert config.env_id == "Reacher-v5"
    assert config.log_dir == Path("logs/mujoco_ablations")
    assert config.n_eval_episodes == 10
    assert config.final_eval_episodes == 50
    assert config.collection_timesteps == 1500
    assert config.fragment_length == 1
    assert config.active_learning is True
    assert config.preset == "auto"


def test_atari_suite_defaults():
    pytest.importorskip("ale_py")
    config = normalize_experiment_config(ExperimentConfig(suite="atari", mode="true"))

    assert config.env_id == "ALE/Breakout-v5"
    assert config.log_dir == Path("logs/atari_ablations")
    assert config.n_eval_episodes == 5
    assert config.final_eval_episodes == 10
    assert config.collection_timesteps == 50_000
    assert config.fragment_length == 64
    assert config.active_learning is False


def test_explicit_values_are_not_overridden():
    config = normalize_experiment_config(
        ExperimentConfig(suite="gym", mode="true", n_eval_episodes=3, collection_timesteps=99, fragment_length=7, active_learning=False)
    )

    assert config.n_eval_episodes == 3
    assert config.collection_timesteps == 99
    assert config.fragment_length == 7
    assert config.active_learning is False


def test_partial_required_modes():
    for mode in ("partial", "naive", "delta"):
        with pytest.raises(ConfigError, match="requires --partial"):
            normalize_experiment_config(ExperimentConfig(suite="gym", mode=mode))

    config = normalize_experiment_config(ExperimentConfig(suite="gym", mode="delta", partial="example_cartpole"))
    assert config.partial == "example_cartpole"

    for mode in ("true", "feedback"):
        assert normalize_experiment_config(ExperimentConfig(suite="gym", mode=mode)).mode == mode


def test_validation_errors():
    with pytest.raises(ConfigError, match="Unsupported suite"):
        normalize_experiment_config(ExperimentConfig(suite="legacy"))
    with pytest.raises(ConfigError, match="Unsupported mode"):
        normalize_experiment_config(ExperimentConfig(suite="gym", mode="bogus"))
    with pytest.raises(ConfigError, match="env"):
        normalize_experiment_config(ExperimentConfig(suite="gym", env_id="NoSuchEnv-v0", mode="true"))
    with pytest.raises(ConfigError, match="rlhf_rounds"):
        normalize_experiment_config(ExperimentConfig(suite="gym", mode="true", rlhf_rounds=0))
    with pytest.raises(ConfigError, match="device"):
        normalize_experiment_config(ExperimentConfig(suite="gym", mode="true", device="tpu"))
    with pytest.raises(ConfigError, match="preset"):
        normalize_experiment_config(ExperimentConfig(suite="mujoco", mode="true", preset="bogus"))
    with pytest.raises(ConfigError, match="reward_hidden_sizes"):
        normalize_experiment_config(ExperimentConfig(suite="gym", mode="true", reward_hidden_sizes=(0,)))


def test_suite_default_envs():
    assert suite_default_envs("gym") == ("CartPole-v1",)
    assert suite_default_envs("mujoco") == ("Reacher-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")


def test_sweep_normalization_requires_partial():
    with pytest.raises(ConfigError, match="require --partial"):
        normalize_sweep_config(SweepConfig(suite="gym"))

    config = normalize_sweep_config(SweepConfig(suite="gym", partial="example_cartpole", log_dir="some/dir"))
    assert config.env_ids == ("CartPole-v1",)
    assert config.manifest == Path("some/dir") / "manifest.jsonl"
    assert config.collection_timesteps == 2000
    assert config.fragment_length == 1
