from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from local_gym.classes.atari_reward_specs import get_atari_reward_spec
from local_gym.classes.mujoco_reward_specs import get_mujoco_reward_spec

from .config import ATARI_SUITE, BOX2D_SUITE, GYM_SUITE, MUJOCO_SUITE, SweepConfig, normalize_sweep_config
from .results import PlannedRun, SweepResult


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    active_learning: bool | None = None
    pretrain: bool = False
    pretrain_target: str = "partial"
    include_partial_feature: bool | None = None


def ablation_variants(suite: str) -> list[Variant]:
    if suite == MUJOCO_SUITE:
        variants = [
            Variant(name="true_reference", mode="true"),
            Variant(name="partial_only", mode="partial"),
            Variant(name="feedback_only_al", mode="feedback", active_learning=True, include_partial_feature=False),
        ]
        for mode in ["naive", "delta"]:
            for pretrain in [False, True]:
                for active_learning in [False, True]:
                    name_parts = [mode]
                    name_parts.append("pretrain" if pretrain else "scratch")
                    name_parts.append("al" if active_learning else "random")
                    variants.append(
                        Variant(
                            name="_".join(name_parts),
                            mode=mode,
                            active_learning=active_learning,
                            pretrain=pretrain,
                            pretrain_target="partial",
                            include_partial_feature=True,
                        )
                    )
        return variants

    if suite == ATARI_SUITE:
        return [
            Variant(name="true_reference", mode="true"),
            Variant(name="partial_only", mode="partial"),
            Variant(name="feedback_only_random", mode="feedback", active_learning=False, include_partial_feature=False),
            Variant(name="naive_scratch_random", mode="naive", active_learning=False, include_partial_feature=True),
            Variant(name="delta_scratch_random", mode="delta", active_learning=False, include_partial_feature=True),
        ]

    if suite in (BOX2D_SUITE, GYM_SUITE):
        return [
            Variant(name="true_reference", mode="true"),
            Variant(name="partial_only", mode="partial"),
            Variant(name="naive_scratch_random", mode="naive", active_learning=False, include_partial_feature=True),
            Variant(name="delta_scratch_random", mode="delta", active_learning=False, include_partial_feature=True),
        ]

    raise ValueError(f"Unsupported sweep suite: {suite}")


def plan_sweep(config: SweepConfig) -> list[PlannedRun]:
    config = normalize_sweep_config(config)
    planned = []
    for env_id in config.env_ids or ():
        for seed in config.seeds:
            for variant in ablation_variants(config.suite):
                command, run_dir = build_command(config, env_id, seed, variant)
                planned.append(
                    PlannedRun(
                        env_id=env_id,
                        seed=seed,
                        variant=variant.name,
                        mode=variant.mode,
                        run_dir=run_dir,
                        completed=(run_dir / "metadata.json").exists(),
                        command=command,
                    )
                )
    return planned


def run_sweep(config: SweepConfig) -> SweepResult:
    config = normalize_sweep_config(config)
    planned_runs = plan_sweep(config)
    pending_runs = [run for run in planned_runs if not (run.completed and config.skip_completed)]
    write_manifest(Path(config.manifest), [run.as_manifest_row() for run in planned_runs])
    print(f"wrote manifest with {len(planned_runs)} planned runs to {config.manifest}")
    print(f"{len(pending_runs)} runs pending")

    if not config.execute:
        for run in pending_runs:
            print(shell_join(run.command))
        print("\nDry run only. Re-run with --execute to launch the study.")
        return SweepResult(Path(config.manifest), planned_runs, pending_runs, executed=False)

    for index, run in enumerate(pending_runs, start=1):
        print(f"\n[{index}/{len(pending_runs)}] running {run.run_dir}")
        subprocess.run(run.command, check=True)
    return SweepResult(Path(config.manifest), planned_runs, pending_runs, executed=True)


def build_command(config: SweepConfig, env_id: str, seed: int, variant: Variant) -> tuple[list[str], Path]:
    if config.suite == MUJOCO_SUITE:
        slug = get_mujoco_reward_spec(env_id).slug
    elif config.suite == ATARI_SUITE:
        slug = get_atari_reward_spec(env_id).slug
    else:
        slug = slugify(env_id)
    run_name = f"{slug}_{variant.name}_{config.timesteps // 1000}k_seed{seed}"
    run_dir = Path(config.log_dir) / run_name
    command = [
        sys.executable,
        "-m",
        "reward_composition_api",
        "train",
        "--suite",
        config.suite,
        "--env-id",
        env_id,
        "--mode",
        variant.mode,
        "--variant-name",
        variant.name,
        "--run-name",
        run_name,
        "--log-dir",
        str(config.log_dir),
        "--timesteps",
        str(config.timesteps),
        "--seed",
        str(seed),
        "--n-envs",
        str(config.n_envs),
        "--eval-freq",
        str(config.eval_freq),
        "--n-eval-episodes",
        str(config.n_eval_episodes),
        "--final-eval-episodes",
        str(config.final_eval_episodes),
        "--query-budget",
        str(config.query_budget),
        "--rlhf-rounds",
        str(config.rlhf_rounds),
        "--collection-timesteps",
        str(config.collection_timesteps),
        "--fragment-length",
        str(config.fragment_length),
        "--reward-model-epochs",
        str(config.reward_model_epochs),
        "--reward-model-patience",
        str(config.reward_model_patience),
        "--reward-model-batch-size",
        str(config.reward_model_batch_size),
        "--reward-model-ensemble-size",
        str(config.reward_model_ensemble_size),
        "--active-query-strategy",
        config.active_query_strategy,
        "--active-learning-batches",
        str(config.active_learning_batches),
        "--device",
        config.device,
    ]

    if config.suite == MUJOCO_SUITE:
        command.extend(["--preset", str(config.preset)])
    if config.suite == ATARI_SUITE:
        command.extend(["--partial-source", config.partial_source])
    if config.partial:
        command.extend(["--partial", config.partial])
    if config.progress_bar:
        command.append("--progress-bar")
    if config.normalize_model_reward:
        command.append("--normalize-model-reward")
    if config.model_reward_min is not None:
        command.extend(["--model-reward-min", str(config.model_reward_min)])
    if config.model_reward_max is not None:
        command.extend(["--model-reward-max", str(config.model_reward_max)])
    if config.model_reward_target_mean is not None:
        command.extend(["--model-reward-target-mean", str(config.model_reward_target_mean)])
    if config.model_reward_target_std is not None:
        command.extend(["--model-reward-target-std", str(config.model_reward_target_std)])

    if variant.mode in {"feedback", "naive", "delta"}:
        command.append("--active-learning" if variant.active_learning else "--no-active-learning")
        if variant.pretrain:
            command.extend(
                [
                    "--pretrain-reward-model",
                    "--pretrain-target",
                    variant.pretrain_target,
                    "--pretrain-epochs",
                    str(config.pretrain_epochs),
                    "--pretrain-batch-size",
                    str(config.pretrain_batch_size),
                    "--pretrain-lr",
                    str(config.pretrain_lr),
                ]
            )
        if variant.include_partial_feature is True:
            command.append("--include-partial-feature")
        elif variant.include_partial_feature is False:
            command.append("--no-partial-feature")

    return command, run_dir


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def slugify(env_id: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in env_id.rsplit("-", 1)[0]).strip("_")
