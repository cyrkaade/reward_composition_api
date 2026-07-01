from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from reward_composition_api.cli import main as cli_main
from reward_composition_api.config import ExperimentConfig, SummaryConfig, SweepConfig, normalize_experiment_config
from reward_composition_api.errors import ConfigError
from reward_composition_api.parsing import parse_int_tuple, parse_key_value_mapping
from reward_composition_api.partial_reward import build_builtin_registry
from reward_composition_api.registry import load_partial_reference
from reward_composition_api.summaries import summarize_runs
from reward_composition_api.sweeps import plan_sweep, run_sweep
from reward_composition_api.reward_models import split_preference_k_folds
from reward_composition_api.training import query_schedule
from reward_composition_api.data_structures.preference import Preference
from reward_composition_api.reward_models.reward_model import RewardModel


class RewardCompositionApiTest(unittest.TestCase):
    def test_builtin_mujoco_partial_matches_component_profile(self):
        registry = build_builtin_registry()
        partial = registry.resolve("default", "mujoco").create("Reacher-v5")
        partial.reset({"reward_dist": 0.0, "reward_ctrl": 0.0})

        step = partial.step(
            obs=None,
            action=None,
            next_obs=None,
            true_reward=0.0,
            terminated=False,
            truncated=False,
            info={"reward_dist": 1.25, "reward_ctrl": -0.2},
        )

        self.assertEqual(step.partial, 1.25)
        self.assertEqual(step.components["reward_dist"], 1.25)
        self.assertEqual(step.components["reward_ctrl"], -0.2)

    def test_builtin_atari_partial_tracks_life_loss(self):
        registry = build_builtin_registry()
        partial = registry.resolve("life_loss", "atari").create("ALE/Breakout-v5")
        partial.reset({"lives": 5})

        step = partial.step(
            obs=None,
            action=0,
            next_obs=None,
            true_reward=0.0,
            terminated=False,
            truncated=False,
            info={"lives": 4},
        )

        self.assertEqual(step.partial, -1.0)
        self.assertEqual(step.components["lost_lives"], 1.0)

    def test_user_partial_module_loading(self):
        registry = build_builtin_registry()
        with tempfile.TemporaryDirectory() as tmp:
            module_path = Path(tmp) / "my_partials.py"
            module_path.write_text(
                "\n".join(
                    [
                        "def constant_partial(**kwargs):",
                        "    return {'partial': 3.5, 'components': {'constant': 3.5}}",
                        "",
                        "def register(registry):",
                        "    registry.register(",
                        "        name='constant',",
                        "        suite='custom',",
                        "        factory=lambda env_id: constant_partial,",
                        "        component_keys=('constant',),",
                        "    )",
                    ]
                ),
                encoding="utf-8",
            )

            spec = load_partial_reference(f"{module_path}:constant", "mujoco", registry)
            step = spec.create("Anything-v0").step(None, None, None, 0.0, False, False, {})

        self.assertEqual(spec.name, "constant")
        self.assertEqual(step.partial, 3.5)
        self.assertEqual(step.components["constant"], 3.5)

    def test_partials_folder_shorthand_loading(self):
        registry = build_builtin_registry()

        spec = load_partial_reference("cartpole_alive", "gym", registry)
        step = spec.create("CartPole-v1").step(None, 0, None, 1.0, False, False, {})

        self.assertEqual(spec.name, "cartpole_alive")
        self.assertEqual(step.partial, 1.0)
        self.assertEqual(step.components["alive_bonus"], 1.0)

    def test_pacman_score_life_partial_is_true_like_and_stateful(self):
        registry = build_builtin_registry()
        spec = load_partial_reference("pacman_score_life", "atari", registry)
        partial = spec.create("ALE/Pacman-v5")
        partial.reset({"lives": 4})

        score_step = partial.step(None, 0, None, 2.0, False, False, {"lives": 4})
        life_loss_step = partial.step(None, 0, None, 0.0, False, False, {"lives": 3})

        self.assertEqual(score_step.partial, 2.0)
        self.assertEqual(life_loss_step.partial, -1.0)
        self.assertEqual(life_loss_step.components["lost_lives"], 1.0)

    def test_config_validation_rejects_bad_rounds(self):
        with self.assertRaises(ConfigError):
            normalize_experiment_config(ExperimentConfig(rlhf_rounds=0))

    def test_cli_friendly_mapping_parsers(self):
        self.assertEqual(parse_int_tuple("64,64"), (64, 64))
        self.assertEqual(
            parse_key_value_mapping("{n_steps:256,batch_size:64,learning_rate:0.0003}"),
            {"n_steps": 256, "batch_size": 64, "learning_rate": 0.0003},
        )

    def test_reward_model_accepts_custom_hidden_sizes(self):
        model = RewardModel(input_size=4, hidden_sizes=(8, 8))

        self.assertEqual(model.net[0].in_features, 4)
        self.assertEqual(model.net[-1].out_features, 1)

    def test_preference_k_folds_use_all_pairs_once(self):
        pairs = [Preference(None, None, float(index)) for index in range(11)]

        folds = split_preference_k_folds(pairs, 5)
        flattened = [pair for fold in folds for pair in fold]

        self.assertEqual(len(folds), 5)
        self.assertEqual(len(flattened), len(pairs))
        self.assertEqual({id(pair) for pair in flattened}, {id(pair) for pair in pairs})
        self.assertLessEqual(max(len(fold) for fold in folds) - min(len(fold) for fold in folds), 1)

    def test_atari_feedback_mode_is_supported(self):
        config = normalize_experiment_config(ExperimentConfig(suite="atari", mode="feedback"))

        self.assertEqual(config.mode, "feedback")
        self.assertFalse(config.active_learning)

    def test_sweep_dry_run_writes_manifest_and_preserves_atari_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_sweep(
                    SweepConfig(
                        suite="atari",
                        env_ids=("ALE/Breakout-v5",),
                        seeds=(2,),
                        timesteps=500_000,
                        log_dir=root / "runs",
                        manifest=root / "manifest.jsonl",
                        execute=False,
                    )
                )

            lines = result.manifest_path.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in lines]

        self.assertFalse(result.executed)
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [row["variant"] for row in rows],
            ["true_reference", "partial_only", "feedback_only_random", "naive_scratch_random", "delta_scratch_random"],
        )
        self.assertEqual(rows[0]["command"][1:4], ["-m", "reward_composition_api", "train"])

    def test_mujoco_planned_run_matches_old_variant_naming_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            planned = plan_sweep(
                SweepConfig(
                    suite="mujoco",
                    env_ids=("Reacher-v5",),
                    seeds=(2,),
                    timesteps=5_000_000,
                    log_dir=Path(tmp),
                )
            )

        self.assertEqual(len(planned), 11)
        self.assertEqual(planned[0].run_dir.name, "reacher_true_reference_5000k_seed2")
        self.assertIn("--query-budget", planned[0].command)
        self.assertIn("1400", planned[0].command)
        self.assertEqual(query_schedule(1400, 5), [280, 280, 280, 280, 280])

    def test_sweep_command_includes_reward_ensemble_knobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            planned = plan_sweep(
                SweepConfig(
                    suite="mujoco",
                    env_ids=("Reacher-v5",),
                    seeds=(2,),
                    log_dir=Path(tmp),
                    reward_model_ensemble_size=5,
                    active_query_strategy="ensemble",
                )
            )

        command = planned[0].command
        self.assertIn("--reward-model-ensemble-size", command)
        self.assertIn("5", command)
        self.assertIn("--active-query-strategy", command)
        self.assertIn("ensemble", command)

    def test_lunar_lander_defaults_match_legacy_fragment_settings(self):
        config = normalize_experiment_config(
            ExperimentConfig(
                suite="box2d",
                env_id="LunarLander-v3",
            )
        )

        self.assertEqual(config.fragment_length, 25)
        self.assertEqual(config.collection_timesteps, 10_000)

    def test_summary_aggregation_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "env_id": "Reacher-v5",
                        "variant": "partial_only",
                        "mode": "partial",
                        "seed": 1,
                        "run_name": "run",
                        "requested_timesteps": 10,
                        "actual_timesteps": 10,
                        "synthetic_queries": 0,
                        "active_query_strategy": "ensemble",
                        "reward_model_ensemble_size": 5,
                        "selected_policy_true_reward_mean": 1.5,
                        "selected_policy_true_reward_std": 0.1,
                        "selected_policy_components": {"mean_partial": 1.0, "mean_residual": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = summarize_runs(
                    SummaryConfig(
                        suite="mujoco",
                        root=root,
                        summary_csv=root / "summary.csv",
                        aggregate_csv=root / "aggregate.csv",
                    )
                )
            summary_exists = result.summary_csv.exists()
            aggregate_exists = result.aggregate_csv.exists()

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.aggregate_rows[0]["mean_selected_true_reward"], 1.5)
        self.assertEqual(result.rows[0]["active_query_strategy"], "ensemble")
        self.assertEqual(result.rows[0]["reward_model_ensemble_size"], 5)
        self.assertTrue(summary_exists)
        self.assertTrue(aggregate_exists)

    def test_cli_smoke_commands(self):
        for argv in (
            ["list-envs", "--suite", "mujoco"],
            ["list-envs", "--suite", "gym"],
            ["list-partials", "--suite", "atari"],
            ["validate-partial", "--suite", "mujoco", "--env-id", "Reacher-v5", "--partial", "default"],
            ["validate-partial", "--suite", "gym", "--env-id", "CartPole-v1", "--partial", "cartpole_alive"],
        ):
            with self.subTest(argv=argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli_main(argv), 0)

    def test_production_code_outside_backend_does_not_import_backend(self):
        root = Path(__file__).resolve().parents[1] / "reward_composition_api"
        offenders = []
        for path in root.rglob("*.py"):
            relative = path.relative_to(root)
            if relative.parts[0] == "backend":
                continue
            text = path.read_text(encoding="utf-8")
            if "reward_composition_api.backend" in text or "from .backend" in text:
                offenders.append(str(relative))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
