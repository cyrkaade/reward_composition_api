"""End-to-end smoke tests: CartPole under each of the five modes with tiny
budgets, asserting the run directory artifacts and metadata keys."""

from __future__ import annotations

import json

import pytest

from rcomp.config import ExperimentConfig
from rcomp.trainer import run_experiment

EXPECTED_METADATA_KEYS = {
    "env_id",
    "mode",
    "run_name",
    "variant",
    "requested_timesteps",
    "actual_timesteps",
    "seed",
    "n_envs",
    "synthetic_queries",
    "query_budget",
    "fragment_length",
    "active_learning",
    "active_query_strategy",
    "reward_hidden_sizes",
    "partial_reference",
    "best_logged_true_reward",
    "best_logged_timestep",
    "env_slug",
    "partial_keys",
    "component_keys",
    "selected_policy_true_reward_mean",
    "selected_policy_true_reward_std",
    "selected_policy_components",
}


def smoke_config(mode: str, tmp_path, **overrides) -> ExperimentConfig:
    needs_partial = mode in ("partial", "naive", "delta")
    return ExperimentConfig(
        suite="gym",
        env_id="CartPole-v1",
        mode=mode,
        partial="example_cartpole" if needs_partial else None,
        timesteps=300,
        n_envs=1,
        seed=1,
        eval_freq=200,
        n_eval_episodes=1,
        final_eval_episodes=1,
        smooth_window=1,
        log_dir=tmp_path,
        rlhf_rounds=2,
        query_budget=10,
        collection_timesteps=40,
        fragment_length=5,
        reward_model_epochs=2,
        reward_model_patience=2,
        reward_hidden_sizes=(8,),
        active_learning_batches=4,
        policy_learning_kwargs={"n_steps": 64, "batch_size": 32},
        **overrides,
    )


@pytest.mark.parametrize("mode", ["true", "partial", "feedback", "naive", "delta"])
def test_smoke_all_modes(mode, tmp_path):
    result = run_experiment(smoke_config(mode, tmp_path))

    run_dir = result.run_dir
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "true_reward_curve.png").exists()
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "vecnormalize.pkl").exists()
    assert (run_dir / "eval" / "evaluations.npz").exists()
    assert (run_dir / "eval" / "component_evaluations.csv").exists()
    assert (run_dir / "eval" / "final_component_evaluation.csv").exists()
    assert (run_dir / "monitor").is_dir()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert EXPECTED_METADATA_KEYS <= set(metadata)
    assert metadata["env_id"] == "CartPole-v1"
    assert metadata["mode"] == mode
    assert metadata["env_slug"] == "cartpole"
    assert metadata["variant"] == mode
    assert metadata["actual_timesteps"] >= 300
    assert isinstance(metadata["selected_policy_true_reward_mean"], float)
    assert "mean_total" in metadata["selected_policy_components"]

    if mode in ("feedback", "naive", "delta"):
        assert metadata["synthetic_queries"] > 0
        assert metadata["reward_composition"] == mode
        assert metadata["query_budget"] == 10
    else:
        assert metadata["synthetic_queries"] == 0
        assert metadata["query_budget"] == 0

    if mode in ("partial", "naive", "delta"):
        assert metadata["partial_keys"] == ["example_cartpole"]
        # function-style partials declare no component_keys; named components
        # only appear when registered via register(..., component_keys=...)
        assert metadata["component_keys"] == []
        assert metadata["partial_reference"] == "example_cartpole"
    else:
        assert metadata["partial_keys"] == []


def test_run_name_and_default_naming(tmp_path):
    result = run_experiment(smoke_config("true", tmp_path))

    assert result.run_dir.name == "cartpole_true_300_seed1"
    assert result.metadata["run_name"] == "cartpole_true_300_seed1"


def test_ensemble_delta_mode(tmp_path):
    result = run_experiment(
        smoke_config(
            "delta",
            tmp_path,
            reward_model_ensemble_size=2,
            active_query_strategy="ensemble",
            active_learning=True,
        )
    )

    assert result.metadata["reward_model_ensemble_size"] == 2
    assert result.metadata["synthetic_queries"] > 0
