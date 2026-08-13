"""Regression tests for --round0-collection-timesteps and --tanh-scale.

Both flags are opt-in: unset/1.0 must reproduce the previous behaviour exactly,
because every archived run and every non-selection grid predates them.
"""

from __future__ import annotations

import pytest
import torch as th

from rcomp.config import ConfigError, ExperimentConfig, normalize_experiment_config
from rcomp.data import Trajectory
from rcomp.rewards.model import RewardModel
from rcomp.trainer import RlhfTrainer


class DummyRuntime:
    normalize_partial = False


def _trainer(config, calls):
    def collect(round_index, collection_steps, stream_index):
        calls.append((round_index, collection_steps, stream_index))
        trajectories = []
        for index in range(4):
            trajectory = Trajectory()
            trajectory.push_state(
                [float(stream_index), float(index)],
                [0.0],
                False,
                {},
                true_rew=0.0,
                partial_rew=0.0,
            )
            trajectories.append(trajectory)
        return trajectories

    return RlhfTrainer(
        config=config,
        model=None,
        runtime=DummyRuntime(),
        callbacks=None,
        reward_model=object(),
        convert_traj=lambda trajectory: [state["obs"] for state in trajectory.states],
        collect_trajectories=collect,
        collection_label="steps",
    )


# --------------------------------------------------------------------------
# round0_collection_timesteps
# --------------------------------------------------------------------------

def test_round0_collection_unset_reproduces_current_behaviour_exactly():
    # separate protocol: two streams of collection_timesteps at round 0
    calls = []
    trainer = _trainer(
        ExperimentConfig(collection_timesteps=100, fragment_length=1, round0_data_protocol="separate"),
        calls,
    )
    trainer.collect_round_data(0)
    trainer.collect_round_data(1)
    assert calls == [(0, 100, 0), (0, 100, 1), (1, 100, 0)]

    # legacy protocol: one doubled rollout at round 0
    calls = []
    trainer = _trainer(
        ExperimentConfig(collection_timesteps=100, fragment_length=1, round0_data_protocol="legacy"),
        calls,
    )
    trainer.collect_round_data(0)
    trainer.collect_round_data(1)
    assert calls == [(0, 200, 0), (1, 100, 0)]


def test_round0_collection_override_applies_to_round_zero_only():
    # separate protocol: both round-0 streams grow, rounds 1+ untouched
    calls = []
    trainer = _trainer(
        ExperimentConfig(
            collection_timesteps=100,
            round0_collection_timesteps=500,
            fragment_length=1,
            round0_data_protocol="separate",
        ),
        calls,
    )
    trainer.collect_round_data(0)
    trainer.collect_round_data(1)
    trainer.collect_round_data(2)
    assert calls == [(0, 500, 0), (0, 500, 1), (1, 100, 0), (2, 100, 0)]

    # legacy protocol keeps its historical doubling, applied to the override
    calls = []
    trainer = _trainer(
        ExperimentConfig(
            collection_timesteps=100,
            round0_collection_timesteps=500,
            fragment_length=1,
            round0_data_protocol="legacy",
        ),
        calls,
    )
    trainer.collect_round_data(0)
    trainer.collect_round_data(1)
    assert calls == [(0, 1000, 0), (1, 100, 0)]


def test_round0_collection_timesteps_must_be_positive():
    with pytest.raises(ConfigError, match="round0_collection_timesteps"):
        normalize_experiment_config(
            ExperimentConfig(suite="gym", env_id="CartPole-v1", mode="feedback", round0_collection_timesteps=0)
        )


def test_round0_collection_timesteps_defaults_to_none():
    assert ExperimentConfig().round0_collection_timesteps is None


# --------------------------------------------------------------------------
# tanh_scale
# --------------------------------------------------------------------------

def test_tanh_scale_one_is_bit_identical_to_plain_tanh():
    th.manual_seed(0)
    reference = RewardModel(input_size=4, hidden_sizes=(32,), tanh_output=True)
    scaled = RewardModel(input_size=4, hidden_sizes=(32,), tanh_output=True, tanh_scale=1.0)
    scaled.load_state_dict(reference.state_dict())
    x = th.randn(8, 5, 4) * 10.0
    with th.no_grad():
        assert th.equal(reference(x), scaled(x))


def test_tanh_scale_widens_the_near_linear_region():
    th.manual_seed(0)
    narrow = RewardModel(input_size=4, hidden_sizes=(32,), tanh_output=True, tanh_scale=1.0)
    wide = RewardModel(input_size=4, hidden_sizes=(32,), tanh_output=True, tanh_scale=5.0)
    wide.load_state_dict(narrow.state_dict())
    x = th.randn(64, 5, 4) * 20.0

    with th.no_grad():
        pre = narrow.head(narrow.trunk(x))
        out_narrow = narrow(x)
        out_wide = wide(x)

    # both keep the hard [-1, 1] bound
    assert float(out_narrow.abs().max()) <= 1.0
    assert float(out_wide.abs().max()) <= 1.0
    # the scaled output is exactly tanh(pre / 5), never more saturated
    assert th.allclose(out_wide, th.tanh(pre / 5.0))
    assert bool((out_wide.abs() <= out_narrow.abs() + 1e-7).all())

    # gradients survive where scale=1 has saturated: at pre-activation 3,
    # d/dx tanh(x) ~ 0.0099 but d/dx tanh(x/5) ~ 0.14
    point = th.tensor([3.0], requires_grad=True)
    th.tanh(point / 1.0).backward()
    grad_narrow = float(point.grad)
    point = th.tensor([3.0], requires_grad=True)
    th.tanh(point / 5.0).backward()
    grad_wide = float(point.grad)
    assert grad_wide > 10 * grad_narrow


def test_tanh_scale_default_and_validation():
    assert ExperimentConfig().tanh_scale == 1.0
    assert RewardModel(input_size=4).tanh_scale == 1.0

    base = dict(suite="gym", env_id="CartPole-v1", mode="feedback")
    with pytest.raises(ConfigError, match="tanh_scale"):
        normalize_experiment_config(ExperimentConfig(tanh_scale=0.0, **base))
    with pytest.raises(ConfigError, match="tanh_scale"):
        normalize_experiment_config(ExperimentConfig(tanh_scale=5.0, **base))
    normalize_experiment_config(ExperimentConfig(tanh_scale=5.0, tanh_model_reward=True, **base))
