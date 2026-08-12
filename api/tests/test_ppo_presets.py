from __future__ import annotations

import gymnasium as gym
import pytest
from torch import nn

from rcomp.config import ExperimentConfig, normalize_experiment_config
from rcomp.ppo_presets import (
    PRESETS,
    LinearSchedule,
    PresetError,
    lookup_preset,
    preset_key,
    resolve_ppo_preset,
    tuned_ppo_hyperparams,
)
from rcomp.suites import get_suite


class _ProbeEnv:
    """Minimal stand-in for a probe env; only observation_space is read."""

    def __init__(self, shape=(11,)):
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=shape)


@pytest.mark.parametrize(
    "env_id, expected",
    [
        ("Hopper-v5", "Hopper"),
        ("Hopper-v4", "Hopper"),
        ("Walker2d-v5", "Walker2d"),
        ("LunarLander-v3", "LunarLander"),
        ("ALE/Breakout-v5", "atari"),
        ("ALE/SpaceInvaders-v5", "atari"),
    ],
)
def test_preset_key_is_version_agnostic(env_id, expected):
    assert preset_key(env_id) == expected


def test_lookup_returns_a_copy_so_callers_cannot_mutate_the_table():
    first = lookup_preset("Walker2d-v5")
    first["ppo"]["learning_rate"] = 999.0
    assert lookup_preset("Walker2d-v5")["ppo"]["learning_rate"] == pytest.approx(5.05041e-05)


def test_unknown_env_has_no_preset():
    assert lookup_preset("NotAnEnv-v1") is None
    with pytest.raises(PresetError, match="No PPO preset"):
        tuned_ppo_hyperparams("NotAnEnv-v1")


def test_hopper_preset_is_sb3_defaults_because_both_tuned_blocks_lost():
    """Hopper has no working tuned block, so the flag must be a no-op there.

    Two candidates were measured and both lost. The rl-zoo Hopper-v4 block pins
    Hopper-v5 at the survive-only optimum (~1000, episode length 1000). The
    gSDE-derived replacement won a 3-seed x 1M pilot but lost the 10-seed x 5M
    head-to-head against stock at matched n_envs: peak 2390 vs 3369, stock ahead
    in 9 of 10 seeds (exact p=0.004). Guard against silently reintroducing either.
    """
    hyperparams = tuned_ppo_hyperparams("Hopper-v5")

    assert lookup_preset("Hopper-v5")["tuned"] is False
    assert hyperparams["n_steps"] == 2048
    assert hyperparams["learning_rate"] == pytest.approx(3e-4)
    assert hyperparams["gamma"] == pytest.approx(0.99)
    assert "clip_range_vf" not in hyperparams
    assert "policy_kwargs" not in hyperparams


def test_tuned_flag_is_inert_where_no_tuned_block_exists():
    """Pusher and Hopper both fall through to SB3 defaults, so runs with and
    without --tuned-hyperparams stay comparable on those two environments."""
    for env_id in ("Hopper-v5", "Pusher-v5"):
        stock = get_suite("mujoco").ppo_hyperparams(
            normalize_experiment_config(ExperimentConfig(suite="mujoco", env_id=env_id, mode="true")),
            _ProbeEnv(),
        )
        tuned = get_suite("mujoco").ppo_hyperparams(
            normalize_experiment_config(
                ExperimentConfig(suite="mujoco", env_id=env_id, mode="true", tuned_hyperparams=True)
            ),
            _ProbeEnv(),
        )
        assert stock == tuned, env_id


def test_reacher_preset_agrees_with_the_pre_existing_suite_preset():
    """The suite already carried a hand-copied Reacher block; they must not drift."""
    config = normalize_experiment_config(ExperimentConfig(suite="mujoco", env_id="Reacher-v5", mode="true"))
    suite_defaults = get_suite("mujoco").default_ppo_hyperparams(config, _ProbeEnv())
    preset = tuned_ppo_hyperparams("Reacher-v5")

    for key in ("n_steps", "batch_size", "gamma", "learning_rate", "ent_coef", "clip_range", "n_epochs", "gae_lambda", "max_grad_norm", "vf_coef"):
        assert preset[key] == pytest.approx(suite_defaults[key]), key


def test_linear_schedule_anneals_to_zero():
    schedule = LinearSchedule(2.5e-4)
    assert schedule(1.0) == pytest.approx(2.5e-4)
    assert schedule(0.5) == pytest.approx(1.25e-4)
    assert schedule(0.0) == pytest.approx(0.0)


def test_schedules_resolve_to_callables_and_warn():
    with pytest.warns(RuntimeWarning, match="schedule"):
        hyperparams = resolve_ppo_preset(PRESETS["CartPole"])
    assert isinstance(hyperparams["learning_rate"], LinearSchedule)
    assert isinstance(hyperparams["clip_range"], LinearSchedule)


def test_atari_preset_does_not_override_the_ram_obs_policy():
    """AtariSuite uses obs_type='ram' + MlpPolicy; the zoo block assumes CnnPolicy."""
    assert "policy" not in PRESETS["atari"]["ppo"]


def test_every_preset_resolves():
    for key in PRESETS:
        resolved = resolve_ppo_preset(PRESETS[key], warn_on_schedule=False)
        assert resolved, key
        policy_kwargs = resolved.get("policy_kwargs")
        if policy_kwargs:
            assert not isinstance(policy_kwargs.get("activation_fn"), str), key


def test_tuned_hyperparams_is_off_by_default():
    """Archived runs must stay comparable, so nothing changes without the flag."""
    config = normalize_experiment_config(ExperimentConfig(suite="mujoco", env_id="Hopper-v5", mode="true"))
    assert config.tuned_hyperparams is False

    hyperparams = get_suite("mujoco").ppo_hyperparams(config, _ProbeEnv())
    assert hyperparams["n_steps"] == 2048
    assert hyperparams["learning_rate"] == pytest.approx(3e-4)


def test_tuned_hyperparams_flag_applies_the_preset():
    config = normalize_experiment_config(
        ExperimentConfig(suite="mujoco", env_id="Walker2d-v5", mode="true", tuned_hyperparams=True)
    )
    hyperparams = get_suite("mujoco").ppo_hyperparams(config, _ProbeEnv())

    assert hyperparams["n_steps"] == 512
    assert hyperparams["learning_rate"] == pytest.approx(5.05041e-05)


def test_policy_learning_kwargs_still_win_over_the_preset():
    config = normalize_experiment_config(
        ExperimentConfig(
            suite="mujoco",
            env_id="Walker2d-v5",
            mode="true",
            tuned_hyperparams=True,
            policy_learning_kwargs={"n_steps": 256},
        )
    )
    hyperparams = get_suite("mujoco").ppo_hyperparams(config, _ProbeEnv())

    assert hyperparams["n_steps"] == 256
    assert hyperparams["learning_rate"] == pytest.approx(5.05041e-05)


def test_every_preset_records_its_provenance():
    for key, preset in PRESETS.items():
        assert preset.get("source"), key
        assert "tuned" in preset, key
        assert preset.get("reference"), key
        assert preset.get("ppo"), key
