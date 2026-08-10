"""The single experiment runner for all suites and all five reward modes,
including the RLHF round loop for the preference modes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import torch as th
from gymnasium.spaces.utils import flatdim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from .config import PREFERENCE_MODES, ExperimentConfig, normalize_experiment_config
from .data import Trajectory
from .envs import TrajectoryCollector, load_eval_env, make_eval_env, make_train_env
from .evaluation import (
    ComponentEvalCallback,
    RunPaths,
    evaluate_components,
    component_fieldnames,
    report_eval_curve,
    select_final_policy,
    write_component_summary_csv,
)
from .partials import include_partial_feature, resolve_custom_partial
from .rewards.model import RewardModel
from .rewards.preferences import (
    choose_query_pairs,
    fragment_trajectories,
    gate_partial_error_stats,
    gate_statistics,
    query_fisher_information,
    reward_model_diagnostics,
    pretrain_reward_model,
    pretrain_reward_model_bt,
    rate_pairs_from_true_reward,
    reward_model_io_stats,
    train_preference_reward_ensemble,
    train_preference_reward_model,
)
from .rewards.wrapper import LearnedRewardRuntime, PreferenceRewardWrapper
from .suites import Suite, get_suite


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metadata_path: Path
    model_path: Path
    vecnormalize_path: Path | None = None
    synthetic_queries: int = 0
    metadata: dict = field(default_factory=dict)


class SaveVecNormalizeOnBest(BaseCallback):
    def __init__(self, env: VecNormalize, save_path: Path):
        super().__init__()
        self.env = env
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.env.save(self.save_path)
        return True


def query_schedule(query_budget: int, rounds: int) -> list[int]:
    unit = query_budget // rounds
    schedule = [unit] * rounds
    for i in range(query_budget - sum(schedule)):
        schedule[i % len(schedule)] += 1
    return schedule


def policy_training_schedule(total_timesteps: int, rounds: int, timesteps_per_round: int | None = None) -> list[int]:
    if timesteps_per_round is not None:
        return [timesteps_per_round] * rounds

    policy_steps_per_round = total_timesteps // rounds
    leftover_policy_steps = total_timesteps - policy_steps_per_round * rounds
    return [
        policy_steps_per_round + (leftover_policy_steps if round_index == rounds - 1 else 0)
        for round_index in range(rounds)
    ]


def trajectory_collection_seed(seed: int, round_index: int, stream_index: int) -> int:
    """Keep the historical stream-0 seed and put stream B far outside the
    consecutive per-vector-env seed range used by Stable-Baselines3."""
    return seed * 1000 + round_index * 100 + stream_index * 10_000_000


def query_selection_seed(seed: int, round_index: int) -> int:
    """Independent deterministic seed stream for query-pair construction."""
    return seed * 1_000_003 + round_index * 10_007 + 20_000_003


def learn_policy(
    model,
    total_timesteps: int,
    callback,
    progress_bar: bool,
    reset_num_timesteps: bool = True,
    log_interval: int | None = None,
) -> None:
    if total_timesteps <= 0:
        return

    learn_kwargs = {
        "total_timesteps": int(total_timesteps),
        "callback": callback,
        "progress_bar": progress_bar,
        "reset_num_timesteps": reset_num_timesteps,
    }
    if log_interval is not None:
        learn_kwargs["log_interval"] = log_interval
    model.learn(**learn_kwargs)


class RlhfTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        model,
        runtime,
        callbacks,
        reward_model: RewardModel | list[RewardModel],
        convert_traj: Callable[[Trajectory], list[list[float]]],
        collect_trajectories: Callable[[int, int, int], list[Trajectory]],
        collection_label: str,
    ):
        self.config = config
        self.model = model
        self.runtime = runtime
        self.callbacks = callbacks
        self.reward_models = reward_model if isinstance(reward_model, list) else [reward_model]
        self.reward_model = self.reward_models[0]
        self.convert_traj = convert_traj
        self.collect_trajectories = collect_trajectories
        self.collection_label = collection_label
        self.rated_train = []
        self.rated_val = []
        self.total_queries = 0
        self.pretraining_done = False
        self.partial_stat_count = 0
        self.partial_stat_mean = 0.0
        self.partial_stat_m2 = 0.0
        self.schedule = query_schedule(config.query_budget, config.rlhf_rounds)
        self.policy_steps_by_round = policy_training_schedule(
            config.timesteps,
            config.rlhf_rounds,
            config.policy_timesteps_per_round,
        )
        self.add_partial_to_predictions = config.mode in {"naive", "delta"}

    def run(self) -> int:
        self.train_initial_policy()
        for round_index, round_query_budget in enumerate(self.schedule):
            self.run_round(round_index, round_query_budget)
        self.train_final_policy()
        return self.total_queries

    def train_initial_policy(self) -> None:
        config = self.config
        if config.initial_timesteps:
            print(f"initial PPO training on {config.mode} reward for {config.initial_timesteps} timesteps")
            learn_policy(
                self.model,
                config.initial_timesteps,
                self.callbacks,
                progress_bar=config.progress_bar,
                reset_num_timesteps=False,
                log_interval=config.policy_log_interval,
            )

    def run_round(self, round_index: int, round_query_budget: int) -> None:
        config = self.config
        if round_index == 0 and config.round0_data_protocol != "legacy":
            collection_description = (
                f"two independently seeded sets of {config.collection_timesteps} {self.collection_label} "
                f"({config.round0_data_protocol} protocol)"
            )
        else:
            collection_steps = config.collection_timesteps * (2 if round_index == 0 else 1)
            collection_description = f"{collection_steps} {self.collection_label}"
        print(f"\nPreference round {round_index}: collecting {collection_description} for {round_query_budget} queries")
        if round_query_budget <= 0 and not self._needs_pretraining():
            print("skipping preference collection because no queries are scheduled")
            self.train_policy_round(round_index)
            return

        pretrain_trajectories, query_trajectories = self.collect_round_data(round_index)
        self.maybe_pretrain_reward_model(pretrain_trajectories)
        self.add_query_pairs(query_trajectories, round_query_budget, round_index)
        self.maybe_train_reward_model()
        self.train_policy_round(round_index)
        if self.total_queries >= config.query_budget:
            print("synthetic query budget exhausted")

    def _needs_pretraining(self) -> bool:
        return bool(self.config.pretrain_reward_model and not self.pretraining_done)

    def collect_round_data(self, round_index: int) -> tuple[list[Trajectory], list[Trajectory]]:
        """Collect the pretraining and query pools for one feedback round.

        ``legacy`` preserves the historical behavior exactly: round 0 is one
        doubled rollout and ``pretrain_holdout`` optionally takes its ordered
        first/second halves.  The opt-in protocols collect two equal-size sets
        with distinct deterministic seed streams before any pretraining occurs:

        - ``separate`` pretrains on set A and queries set B;
        - ``overlap`` pretrains and queries set A, while still collecting set B
          so the leak control uses the same environment-interaction budget.

        Scratch controls using ``separate`` query the same stream-B pool, with
        the same number of candidate trajectories, as pretrained arms.  Because
        A and B come from separate resets/seeds, neither protocol depends on the
        completion-order layout returned by ``BufferingWrapper``.
        """
        config = self.config
        if round_index == 0 and config.round0_data_protocol != "legacy":
            set_a = self.collect_trajectories(round_index, config.collection_timesteps, 0)
            set_b = self.collect_trajectories(round_index, config.collection_timesteps, 1)
            self.update_partial_stats([*set_a, *set_b])
            # Equal collection steps do not guarantee equal query capacity when
            # episode boundaries discard fragment remainders. Materializing full
            # fragments and trimming both streams to the smaller count gives the
            # scratch, disjoint, and overlap arms exactly equal candidate-pool
            # sizes. Feeding these fixed-length fragments back through
            # fragment_trajectories later is idempotent.
            fragments_a = fragment_trajectories(set_a, config.fragment_length)
            fragments_b = fragment_trajectories(set_b, config.fragment_length)
            matched_fragments = min(len(fragments_a), len(fragments_b))
            pretrain_trajectories = fragments_a[:matched_fragments]
            separate_query_trajectories = fragments_b[:matched_fragments]
            query_trajectories = (
                pretrain_trajectories
                if config.round0_data_protocol == "overlap"
                else separate_query_trajectories
            )
            print(
                f"round-0 data: matched {matched_fragments} full fragments per stream; "
                f"query set {'A (overlap)' if config.round0_data_protocol == 'overlap' else 'B (separate)'} "
                f"has {len(query_trajectories)} fragments"
            )
            return pretrain_trajectories, query_trajectories

        collection_steps = config.collection_timesteps * (2 if round_index == 0 else 1)
        trajectories = self.collect_trajectories(round_index, collection_steps, 0)
        self.update_partial_stats(trajectories)
        return self.split_for_pretraining(trajectories)

    def split_for_pretraining(self, trajectories: list[Trajectory]) -> tuple[list[Trajectory], list[Trajectory]]:
        """Keep pretraining and round-0 query collection on disjoint rollouts.

        Without this the reward model is fit on exactly the states it is then
        asked to express preferences about, which inflates the round-0 fit and
        makes the cold-start claim ("a pretrained model selects better queries")
        indistinguishable from "a pretrained model has memorised these states".
        Round 0 already collects 2x collection_timesteps, so halving it leaves the
        query supply at the same level every other round sees.
        """
        if not (self.config.pretrain_holdout and self._needs_pretraining()):
            return trajectories, trajectories
        cut = len(trajectories) // 2
        if cut == 0:
            return trajectories, trajectories
        print(f"pretrain holdout: fitting on {cut} trajectories, querying from the other {len(trajectories) - cut}")
        return trajectories[:cut], trajectories[cut:]

    def update_partial_stats(self, trajectories: list[Trajectory]) -> None:
        """Welford running mean/std over every partial-reward step seen so far,
        shared by the model input feature, the delta loss, and the composed reward."""
        if not self.runtime.normalize_partial:
            return
        for trajectory in trajectories:
            for state in trajectory.states:
                value = float(state["partial_rew"])
                self.partial_stat_count += 1
                delta = value - self.partial_stat_mean
                self.partial_stat_mean += delta / self.partial_stat_count
                self.partial_stat_m2 += delta * (value - self.partial_stat_mean)
        if self.partial_stat_count >= 2:
            self.runtime.partial_mean = self.partial_stat_mean
            self.runtime.partial_std = max((self.partial_stat_m2 / self.partial_stat_count) ** 0.5, 1e-8)
            print(f"partial reward stats: mean={self.runtime.partial_mean:.4f}, std={self.runtime.partial_std:.4f}")

    def maybe_pretrain_reward_model(self, trajectories: list[Trajectory]) -> None:
        config = self.config
        if self._needs_pretraining():
            print(f"pretraining reward model on {config.pretrain_target} target with {config.pretrain_loss} loss")
            for model_index, reward_model in enumerate(self.reward_models):
                if len(self.reward_models) > 1:
                    print(f"pretraining reward ensemble member {model_index + 1}/{len(self.reward_models)}")
                if config.pretrain_loss == "bt":
                    stats = pretrain_reward_model_bt(
                        reward_model,
                        trajectories,
                        self.convert_traj,
                        target=config.pretrain_target,
                        epochs=config.pretrain_epochs,
                        batch_size=config.pretrain_batch_size,
                        learning_rate=config.pretrain_lr,
                        fragment_length=config.fragment_length,
                        max_pairs=config.pretrain_pairs,
                        patience=config.pretrain_patience,
                        temperature=config.pretrain_bt_temperature,
                        tie_margin=config.pretrain_bt_tie_margin,
                        loss_reduction=(
                            config.reward_model_loss_reduction
                            if config.pretrain_bt_match_reward_training
                            else "sum"
                        ),
                        weight_l1=(config.reward_model_l1 if config.pretrain_bt_match_reward_training else 0.0),
                        output_l1=(config.reward_output_l1 if config.pretrain_bt_match_reward_training else 0.0),
                    )
                    if stats and self.runtime.pretrain_stats is None:
                        self.runtime.pretrain_stats = stats
                else:
                    pretrain_reward_model(
                        reward_model,
                        trajectories,
                        self.convert_traj,
                        target=config.pretrain_target,
                        epochs=config.pretrain_epochs,
                        batch_size=config.pretrain_batch_size,
                        learning_rate=config.pretrain_lr,
                    )
            self.pretraining_done = True

    def add_query_pairs(self, trajectories: list[Trajectory], round_query_budget: int, round_index: int) -> None:
        config = self.config
        query_model = self.reward_models if (self.total_queries > 0 or self.pretraining_done) else None
        query_rng = random.Random(query_selection_seed(config.seed, round_index)) if config.dedicated_query_rng else None
        pairs = choose_query_pairs(
            trajectories,
            query_model,
            query_count=min(round_query_budget, config.query_budget - self.total_queries),
            fragment_length=config.fragment_length,
            active_learning=config.active_learning,
            convert_traj=self.convert_traj,
            add_partial_to_predictions=self.add_partial_to_predictions,
            dropout_samples=config.dropout_samples,
            dropout_p=config.dropout_p,
            active_learning_batches=config.active_learning_batches,
            active_query_strategy=config.active_query_strategy,
            transform_partial=self.runtime.composed_partial_reward,
            rng=query_rng,
        )
        rated_pairs = rate_pairs_from_true_reward(pairs)

        # How informative were the queries this round's selector actually chose?
        # Scored with the model that did the choosing, i.e. before this round's
        # training. At round 0 that model is either partial-pretrained or random,
        # which is exactly the cold-start comparison.
        if config.query_fisher_diagnostic:
            fisher = query_fisher_information(
                self.reward_models,
                rated_pairs,
                self.convert_traj,
                partial_mean=self.runtime.partial_mean,
                partial_std=self.runtime.partial_std,
                partial_alpha=config.partial_alpha,
                add_partial=self.add_partial_to_predictions,
            )
            if fisher:
                fisher["round"] = len(self.runtime.query_fisher)
                fisher["selector_was_trained"] = query_model is not None
                self.runtime.query_fisher.append(fisher)
                print(
                    f"query Fisher information (round {fisher['round']}): "
                    f"mean={fisher['fisher_mean']:.4g} median={fisher['fisher_median']:.4g} "
                    f"(selector had a model: {fisher['selector_was_trained']})"
                )

        split = int(len(rated_pairs) * 0.8)
        self.rated_train.extend(rated_pairs[:split])
        self.rated_val.extend(rated_pairs[split:])
        self.total_queries += len(rated_pairs)
        print(f"rated {len(rated_pairs)} synthetic preference pairs; cumulative={self.total_queries}")

    def maybe_train_reward_model(self) -> None:
        config = self.config
        if self.rated_train:
            # M3: score the model on held-out preferences BEFORE any preference
            # training. For a partial-pretrained model this is the head start it
            # gets over a randomly initialised one; recorded once, on the first
            # round, while the model is still in its initial state.
            if config.reward_model_diagnostics and self.runtime.rm_diagnostics_before is None:
                self.runtime.rm_diagnostics_before = reward_model_diagnostics(
                    self.reward_models,
                    self.rated_val or self.rated_train,
                    self.convert_traj,
                    partial_mean=self.runtime.partial_mean,
                    partial_std=self.runtime.partial_std,
                    partial_alpha=config.partial_alpha,
                )
                if self.runtime.rm_diagnostics_before:
                    d = self.runtime.rm_diagnostics_before
                    print(
                        f"reward model BEFORE preference training: bt_loss={d['bt_loss']:.4f} "
                        f"accuracy={d['accuracy']:.3f} (pretrained={config.pretrain_reward_model})"
                    )
            if len(self.reward_models) > 1:
                member_training_stats = train_preference_reward_ensemble(
                    self.reward_models,
                    self.rated_train + self.rated_val,
                    convert_traj=self.convert_traj,
                    use_delta_loss=config.mode == "delta",
                    batch_size=config.reward_model_batch_size,
                    epochs=config.reward_model_epochs,
                    patience=config.reward_model_patience,
                    learning_rate=config.reward_model_lr,
                    loss_reduction=config.reward_model_loss_reduction,
                    weight_l1=config.reward_model_l1,
                    output_l1=config.reward_output_l1,
                    training_mode=config.ensemble_training,
                    partial_mean=self.runtime.partial_mean,
                    partial_std=self.runtime.partial_std,
                    partial_alpha=config.partial_alpha,
                    partial_alpha_penalty=config.partial_alpha_penalty,
                    partial_prediction_coef=config.partial_prediction_coef,
                    gate_holdout=config.gate_holdout,
                    gate_learning_rate=config.gate_lr,
                    gate_epochs=config.gate_epochs,
                    gate_patience=config.gate_patience,
                    gate_prior_penalty=config.gate_prior_penalty,
                    train_accuracy_stop=config.reward_model_train_accuracy_stop,
                )
                self.runtime.reward_model = None
                self.runtime.reward_models = self.reward_models
            else:
                training_stats = train_preference_reward_model(
                    self.reward_model,
                    self.rated_train,
                    self.rated_val,
                    convert_traj=self.convert_traj,
                    use_delta_loss=config.mode == "delta",
                    batch_size=config.reward_model_batch_size,
                    epochs=config.reward_model_epochs,
                    patience=config.reward_model_patience,
                    learning_rate=config.reward_model_lr,
                    loss_reduction=config.reward_model_loss_reduction,
                    weight_l1=config.reward_model_l1,
                    output_l1=config.reward_output_l1,
                    partial_mean=self.runtime.partial_mean,
                    partial_std=self.runtime.partial_std,
                    partial_alpha=config.partial_alpha,
                    partial_alpha_penalty=config.partial_alpha_penalty,
                    partial_prediction_coef=config.partial_prediction_coef,
                    gate_holdout=config.gate_holdout,
                    gate_learning_rate=config.gate_lr,
                    gate_epochs=config.gate_epochs,
                    gate_patience=config.gate_patience,
                    gate_prior_penalty=config.gate_prior_penalty,
                    train_accuracy_stop=config.reward_model_train_accuracy_stop,
                )
                member_training_stats = (
                    [{"member_index": 0, "training_mode": "single", **training_stats}]
                    if training_stats is not None
                    else []
                )
                self.runtime.reward_model = self.reward_model
                self.runtime.reward_models = None
            self.runtime.reward_model_training.append(
                {
                    "round": len(self.runtime.reward_model_training),
                    "cumulative_queries": self.total_queries,
                    "members": member_training_stats,
                }
            )
            if config.learn_partial_alpha:
                alphas = [float(model.alpha.item()) for model in self.reward_models if model.alpha is not None]
                self.runtime.partial_alpha = sum(alphas) / len(alphas)
                print(f"learned partial alpha: {self.runtime.partial_alpha:.4f}")
            if config.gate_partial:
                gate_trajectories = [pair.t1 for pair in self.rated_train + self.rated_val] + [
                    pair.t2 for pair in self.rated_train + self.rated_val
                ]
                self.runtime.gate_stats = gate_statistics(self.reward_models, gate_trajectories, self.convert_traj)
                if self.runtime.gate_stats:
                    s = self.runtime.gate_stats
                    print(f"learned gate g: mean={s['mean']:.3f} p10={s['p10']:.3f} p50={s['p50']:.3f} p90={s['p90']:.3f} "
                          f"(frac<0.1={s['frac_below_0.1']:.2f}, frac>0.9={s['frac_above_0.9']:.2f})")
                if config.gate_diagnostic:
                    self.runtime.gate_error_stats = gate_partial_error_stats(
                        self.reward_models, gate_trajectories, self.convert_traj
                    )
                    if self.runtime.gate_error_stats:
                        d = self.runtime.gate_error_stats
                        print(
                            f"gate diagnostic: corr(g, |partial-true|)={d['corr_gate_partial_error']:+.3f} "
                            f"(negative = the gate closes where the partial is wrong); "
                            f"mean g on high-error states={d['mean_gate_high_error']:.3f} vs "
                            f"low-error={d['mean_gate_low_error']:.3f}"
                        )
            stat_trajectories = [pair.t1 for pair in self.rated_train + self.rated_val] + [
                pair.t2 for pair in self.rated_train + self.rated_val
            ]
            self.runtime.output_mean, self.runtime.output_std = reward_model_io_stats(
                self.reward_models,
                stat_trajectories,
                self.convert_traj,
            )
            if self.runtime.reward_model_training:
                # Keep the scale trajectory, not only the final round. This is
                # essential for diagnosing whether an adaptive training stop
                # prevents a learned reward from growing until it overwhelms
                # the partial reward late in policy training.
                round_training = self.runtime.reward_model_training[-1]
                round_training["model_reward_output_mean"] = self.runtime.output_mean
                round_training["model_reward_output_std"] = self.runtime.output_std
            print(f"reward model output stats: mean={self.runtime.output_mean}, std={self.runtime.output_std}")

            # M2: how much does the trained model rely on the partial input feature?
            # Measured by ablating it, not by comparing weight magnitudes.
            if config.reward_model_diagnostics and self.rated_val and config.ensemble_training != "full":
                self.runtime.rm_diagnostics = reward_model_diagnostics(
                    self.reward_models,
                    self.rated_val,
                    self.convert_traj,
                    partial_mean=self.runtime.partial_mean,
                    partial_std=self.runtime.partial_std,
                    partial_alpha=config.partial_alpha,
                )
                if self.runtime.rm_diagnostics:
                    d = self.runtime.rm_diagnostics
                    print(
                        f"reward model on held-out prefs: bt_loss={d['bt_loss']:.4f} acc={d['accuracy']:.3f}; "
                        f"with the partial feature ablated acc={d['accuracy_partial_ablated']:.3f} "
                        f"(drop={d['accuracy_drop_when_ablated']:+.3f} = reliance on the partial feature)"
                    )
            elif config.reward_model_diagnostics and self.rated_val:
                print(
                    "skipping post-training reward-model diagnostics: ensemble_training=full "
                    "trained on every queried pair, so rated_val is not held out"
                )

    def train_policy_round(self, round_index: int) -> None:
        config = self.config
        policy_steps = self.policy_steps_by_round[round_index]
        print(f"training PPO on {config.mode} reward for {policy_steps} timesteps")
        learn_policy(
            self.model,
            policy_steps,
            self.callbacks,
            progress_bar=config.progress_bar,
            reset_num_timesteps=False,
            log_interval=config.policy_log_interval,
        )

    def train_final_policy(self) -> None:
        config = self.config
        if config.final_policy_timesteps:
            print(f"final PPO training on {config.mode} reward for {config.final_policy_timesteps} timesteps")
            learn_policy(
                self.model,
                config.final_policy_timesteps,
                self.callbacks,
                progress_bar=config.progress_bar,
                reset_num_timesteps=False,
                log_interval=config.policy_log_interval,
            )


def default_run_name(config: ExperimentConfig, suite: Suite) -> str:
    variant = config.variant_name or config.mode
    steps = f"{config.timesteps // 1_000_000}m" if config.timesteps >= 1_000_000 else f"{config.timesteps}"
    return f"{suite.slug(config.env_id)}_{variant}_{steps}_seed{config.seed}"


def make_reward_models(input_size: int, config: ExperimentConfig) -> RewardModel | list[RewardModel]:
    models = [
        RewardModel(
            input_size=input_size,
            hidden_sizes=config.reward_hidden_sizes,
            learn_alpha=config.learn_partial_alpha,
            alpha_init=config.partial_alpha,
            predict_partial=config.partial_prediction_coef > 0,
            batchnorm_output=config.batchnorm_model_reward,
            gate_partial=config.gate_partial,
            gate_init=config.gate_init,
            tanh_output=config.tanh_model_reward,
        )
        for _ in range(config.reward_model_ensemble_size)
    ]
    # Start in eval mode so single-step inference uses running stats before the
    # first reward-model training round (no-op when batchnorm is disabled).
    for model in models:
        model.eval()
    return models[0] if len(models) == 1 else models


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        config = normalize_experiment_config(config)
        self.suite = get_suite(config.suite)
        self.config = replace(
            config,
            run_name=config.run_name or default_run_name(config, self.suite),
            variant_name=config.variant_name or config.mode,
        )
        self.custom_partial = resolve_custom_partial(self.config)

    @property
    def run_dir(self) -> Path:
        return Path(self.config.log_dir) / self.config.run_name

    def run(self) -> RunResult:
        self.suite.setup(self.config)
        if self.config.mode in PREFERENCE_MODES:
            return self.train_preference_mode()
        return self.train_true_or_partial()

    def eval_freq(self) -> int:
        return max(self.config.eval_freq // self.config.n_envs, 1)

    def probe_spaces(self):
        probe_env = self.suite.make_raw_env(self.config.env_id)
        observation_space = probe_env.observation_space
        action_space = probe_env.action_space
        normalize = self.suite.should_normalize_observation(observation_space)
        hyperparams = self.suite.ppo_hyperparams(self.config, probe_env)
        probe_env.close()
        return observation_space, action_space, normalize, hyperparams

    def build_runtime(self, composition: str, observation_space, action_space, **kwargs) -> LearnedRewardRuntime:
        return LearnedRewardRuntime(
            env_id=self.config.env_id,
            composition=composition,
            observation_space=observation_space,
            action_space=action_space,
            observation_features=self.suite.observation_features,
            custom_partial=self.custom_partial,
            reset_info=dict(self.suite.wrapper_reset_info),
            cast_true_reward=self.suite.cast_true_reward_info,
            **kwargs,
        )

    def build_envs_and_callbacks(self, env_fn, run_dir: Path, normalize: bool):
        config = self.config
        train_env = make_train_env(env_fn, config.n_envs, run_dir / "monitor", normalize)
        eval_env = make_eval_env(self.suite.make_raw_env, config.env_id, train_env)

        best_callbacks: list[BaseCallback] = []
        if isinstance(train_env, VecNormalize):
            best_callbacks.append(SaveVecNormalizeOnBest(train_env, run_dir / "best_model" / "best_vecnormalize.pkl"))
        if config.stop_reward is not None:
            best_callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=config.stop_reward, verbose=1))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir / "eval"),
            eval_freq=self.eval_freq(),
            n_eval_episodes=config.n_eval_episodes,
            deterministic=True,
            render=False,
            callback_on_new_best=CallbackList(best_callbacks) if best_callbacks else None,
        )
        component_callback = ComponentEvalCallback(
            run_dir / "eval" / "component_evaluations.csv",
            self.suite,
            config.env_id,
            custom_partial=self.custom_partial,
            eval_freq=self.eval_freq(),
            n_eval_episodes=config.n_eval_episodes,
            verbose=1,
        )
        return train_env, eval_env, CallbackList([eval_callback, component_callback])

    def train_true_or_partial(self) -> RunResult:
        config = self.config
        run_dir = self.run_dir
        run_dir.mkdir(exist_ok=True, parents=True)
        observation_space, action_space, normalize, hyperparams = self.probe_spaces()

        if config.mode == "true":
            env_fn = lambda: self.suite.make_raw_env(config.env_id)
        else:
            runtime = self.build_runtime("partial", observation_space, action_space)
            env_fn = lambda: PreferenceRewardWrapper(self.suite.make_raw_env(config.env_id), runtime)

        train_env, eval_env, callbacks = self.build_envs_and_callbacks(env_fn, run_dir, normalize)
        model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **hyperparams)
        learn_policy(
            model,
            config.timesteps,
            callbacks,
            progress_bar=config.progress_bar,
            log_interval=config.policy_log_interval,
        )

        return self.save_and_report(model, train_env, eval_env, run_dir, synthetic_queries=0)

    def train_preference_mode(self) -> RunResult:
        config = self.config
        run_dir = self.run_dir
        run_dir.mkdir(exist_ok=True, parents=True)
        observation_space, action_space, normalize, hyperparams = self.probe_spaces()

        runtime = self.build_runtime(
            config.mode,
            observation_space,
            action_space,
            target_mean=config.model_reward_target_mean,
            target_std=config.model_reward_target_std,
            reward_min=config.model_reward_min,
            reward_max=config.model_reward_max,
            reward_scale=config.model_reward_scale,
            normalize=config.normalize_model_reward,
            normalize_partial=config.normalize_partial_reward,
            partial_alpha=config.partial_alpha,
            gate_partial=config.gate_partial,
            include_partial_feature=include_partial_feature(config),
        )
        train_env, eval_env, callbacks = self.build_envs_and_callbacks(
            lambda: PreferenceRewardWrapper(self.suite.make_raw_env(config.env_id), runtime),
            run_dir,
            normalize,
        )
        model = PPO(env=train_env, verbose=1, seed=config.seed, device=config.device, **hyperparams)

        input_size = flatdim(observation_space) + flatdim(action_space) + 1
        reward_model = make_reward_models(input_size, config)
        convert_traj = self.trajectory_converter(runtime)
        total_queries = RlhfTrainer(
            config,
            model,
            runtime,
            callbacks,
            reward_model,
            convert_traj,
            lambda round_index, collection_steps, stream_index: TrajectoryCollector(vec_env=train_env, agent=model).rollout_trajectories(
                total_timesteps=collection_steps,
                seed=trajectory_collection_seed(config.seed, round_index, stream_index),
            ),
            collection_label=self.suite.collection_label,
        ).run()

        return self.save_and_report(model, train_env, eval_env, run_dir, synthetic_queries=total_queries, runtime=runtime)

    def trajectory_converter(self, runtime: LearnedRewardRuntime):
        def convert(trajectory: Trajectory):
            return [
                runtime.model_features(state["obs"], state["act"], state["partial_rew"]).tolist()
                for state in trajectory.states
            ]

        return convert

    def save_and_report(
        self,
        model: PPO,
        train_env,
        eval_env,
        run_dir: Path,
        synthetic_queries: int,
        runtime: LearnedRewardRuntime | None = None,
    ) -> RunResult:
        config = self.config
        suite = self.suite
        paths = RunPaths(run_dir)
        model.save(paths.final_model)
        if config.save_reward_model and runtime is not None:
            reward_models = runtime.reward_models or ([runtime.reward_model] if runtime.reward_model else [])
            if reward_models:
                # small MLPs; keeping them makes any later offline analysis of the
                # learned reward possible without re-running training
                th.save([m.state_dict() for m in reward_models], run_dir / "reward_model.pt")
        vecnormalize_path = None
        if isinstance(train_env, VecNormalize):
            train_env.save(paths.vecnormalize)
            vecnormalize_path = paths.vecnormalize

        actual_timesteps = int(model.num_timesteps)
        best_logged_reward, best_logged_timestep = report_eval_curve(
            paths.eval_log,
            paths.true_reward_curve,
            max(config.timesteps, actual_timesteps),
            config.plot_mode,
            config.smooth_window,
            x_scale=suite.curve_x_scale,
            x_label=suite.curve_x_label,
            y_floor=suite.curve_y_floor,
        )

        final_stats = evaluate_components(
            model,
            suite,
            config.env_id,
            custom_partial=self.custom_partial,
            stats_source=train_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 50_000,
        )
        write_component_summary_csv(
            paths.final_component_evaluation,
            actual_timesteps,
            final_stats,
            component_fieldnames(self.custom_partial),
        )

        final_policy, final_eval_env = select_final_policy(
            config,
            model,
            eval_env,
            run_dir,
            lambda env_id, stats_path: load_eval_env(suite.make_raw_env, env_id, stats_path),
            PPO.load,
            load_best_stats=isinstance(train_env, VecNormalize),
        )

        mean_reward, std_reward = evaluate_policy(
            final_policy,
            final_eval_env,
            n_eval_episodes=config.final_eval_episodes,
            deterministic=True,
            return_episode_rewards=False,
        )
        selected_stats = evaluate_components(
            final_policy,
            suite,
            config.env_id,
            custom_partial=self.custom_partial,
            stats_source=final_eval_env,
            n_eval_episodes=config.final_eval_episodes,
            seed=config.seed + 60_000,
        )

        metadata = {
            **self.common_metadata(actual_timesteps, synthetic_queries, best_logged_reward, best_logged_timestep),
            "env_slug": suite.slug(config.env_id),
            **suite.extra_metadata(config),
            "partial_keys": [self.custom_partial.name] if self.custom_partial else [],
            "component_keys": suite.metadata_component_keys(self.custom_partial),
            "selected_policy_true_reward_mean": float(mean_reward),
            "selected_policy_true_reward_std": float(std_reward),
            "selected_policy_components": selected_stats,
            **self.runtime_metadata(runtime),
        }

        paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.print_summary(float(mean_reward), float(std_reward), selected_stats, synthetic_queries)

        train_env.close()
        final_eval_env.close()
        return RunResult(
            run_dir=run_dir,
            metadata_path=paths.metadata,
            model_path=paths.final_model.with_suffix(".zip"),
            vecnormalize_path=vecnormalize_path,
            synthetic_queries=synthetic_queries,
            metadata=metadata,
        )

    def common_metadata(
        self,
        actual_timesteps: int,
        synthetic_queries: int,
        best_logged_reward: float | None,
        best_logged_timestep: int | None,
    ) -> dict:
        config = self.config
        is_preference = config.mode in PREFERENCE_MODES
        return {
            "env_id": config.env_id,
            "mode": config.mode,
            "run_name": config.run_name,
            "variant": config.variant_name,
            "requested_timesteps": config.timesteps,
            "actual_timesteps": actual_timesteps,
            "seed": config.seed,
            "n_envs": config.n_envs,
            "initial_timesteps": config.initial_timesteps,
            "policy_timesteps_per_round": config.policy_timesteps_per_round,
            "final_policy_timesteps": config.final_policy_timesteps,
            "final_policy": config.final_policy,
            "collection_timesteps": config.collection_timesteps,
            "policy_learning_kwargs": config.policy_learning_kwargs or {},
            "tuned_hyperparams": config.tuned_hyperparams,
            "synthetic_queries": synthetic_queries,
            "query_budget": config.query_budget if is_preference else 0,
            "fragment_length": config.fragment_length if is_preference else None,
            "active_learning": config.active_learning if is_preference else None,
            "active_query_strategy": config.active_query_strategy if is_preference else None,
            "dedicated_query_rng": config.dedicated_query_rng if is_preference else None,
            "reward_hidden_sizes": list(config.reward_hidden_sizes),
            "reward_model_lr": config.reward_model_lr if is_preference else None,
            "reward_model_epochs": config.reward_model_epochs if is_preference else None,
            "reward_model_patience": config.reward_model_patience if is_preference else None,
            "reward_model_batch_size": config.reward_model_batch_size if is_preference else None,
            "reward_model_train_accuracy_stop": config.reward_model_train_accuracy_stop if is_preference else None,
            "reward_model_loss_reduction": config.reward_model_loss_reduction if is_preference else None,
            "reward_model_l1": config.reward_model_l1 if is_preference else None,
            "reward_output_l1": config.reward_output_l1 if is_preference else None,
            "ensemble_training": config.ensemble_training if is_preference else None,
            "reward_model_ensemble_size": config.reward_model_ensemble_size if is_preference else None,
            "pretrain_reward_model": config.pretrain_reward_model if is_preference else None,
            "pretrain_target": config.pretrain_target if config.pretrain_reward_model else None,
            "pretrain_loss": config.pretrain_loss if config.pretrain_reward_model else None,
            "pretrain_holdout": config.pretrain_holdout if config.pretrain_reward_model else None,
            "round0_data_protocol": config.round0_data_protocol if is_preference else None,
            "pretrain_pairs": config.pretrain_pairs if config.pretrain_reward_model and config.pretrain_loss == "bt" else None,
            "pretrain_bt_temperature": config.pretrain_bt_temperature if config.pretrain_reward_model and config.pretrain_loss == "bt" else None,
            "pretrain_bt_tie_margin": config.pretrain_bt_tie_margin if config.pretrain_reward_model and config.pretrain_loss == "bt" else None,
            "pretrain_bt_match_reward_training": (
                config.pretrain_bt_match_reward_training
                if config.pretrain_reward_model and config.pretrain_loss == "bt"
                else None
            ),
            "tanh_model_reward": config.tanh_model_reward if is_preference else None,
            "include_partial_feature": include_partial_feature(config) if is_preference else None,
            "normalize_partial_reward": config.normalize_partial_reward if is_preference else None,
            "partial_alpha": config.partial_alpha if is_preference else None,
            "learn_partial_alpha": config.learn_partial_alpha if is_preference else None,
            "partial_alpha_penalty": config.partial_alpha_penalty if config.learn_partial_alpha else None,
            "partial_prediction_coef": config.partial_prediction_coef if is_preference else None,
            "batchnorm_model_reward": config.batchnorm_model_reward if is_preference else None,
            "partial_reference": config.partial,
            "best_logged_true_reward": best_logged_reward,
            "best_logged_timestep": best_logged_timestep,
        }

    def runtime_metadata(self, runtime: LearnedRewardRuntime | None) -> dict:
        if runtime is None:
            return {}
        return {
            "model_reward_min": runtime.reward_min,
            "model_reward_max": runtime.reward_max,
            "model_reward_scale": runtime.reward_scale,
            "normalize_model_reward": runtime.normalize,
            "model_reward_output_mean": runtime.output_mean,
            "model_reward_output_std": runtime.output_std,
            "model_reward_target_mean": runtime.target_mean,
            "model_reward_target_std": runtime.target_std,
            "partial_reward_mean": runtime.partial_mean,
            "partial_reward_std": runtime.partial_std,
            "final_partial_alpha": runtime.partial_alpha,
            "gate_partial": runtime.gate_partial,
            "gate_stats": runtime.gate_stats,
            "gate_holdout": self.config.gate_holdout,
            "gate_lr": self.config.gate_lr,
            "gate_init": self.config.gate_init,
            "gate_prior_penalty": self.config.gate_prior_penalty,
            "gate_error_stats": runtime.gate_error_stats,
            "rm_diagnostics": runtime.rm_diagnostics,
            "rm_diagnostics_before_training": runtime.rm_diagnostics_before,
            "reward_model_training": runtime.reward_model_training,
            "query_fisher": runtime.query_fisher,
            "pretrain_stats": runtime.pretrain_stats,
            "reward_composition": runtime.composition,
        }

    def print_summary(self, mean_reward: float, std_reward: float, selected_stats: dict, synthetic_queries: int) -> None:
        print(f"{self.config.final_policy.title()} deterministic true reward: {mean_reward:.3f} +/- {std_reward:.3f}")
        values = ", ".join(
            f"{key}={selected_stats.get(f'mean_{key}', 0.0):.3f}" for key in self.suite.summary_component_keys
        )
        print(f"Component means: {values}")
        print(f"Synthetic queries consumed: {synthetic_queries}")
        print(f"Saved model and logs to {self.run_dir}")


def run_experiment(config: ExperimentConfig) -> RunResult:
    return ExperimentRunner(config).run()
