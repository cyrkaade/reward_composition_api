"""Approximate STARC diagnostics for LunarLander partial rewards.

This module is deliberately separate from training.  It estimates a VAL-STARC
distance from fresh rollouts and never changes the reward used by an experiment.

The approximation uses one fixed reference policy, fits the two policy-value
functions with the same random-feature ridge regressor, applies the VAL temporal
difference, normalises both canonical rewards, and measures their L2 distance on
held-out episodes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConfigError
from .partials import PartialRegistry, PartialSpec, load_partial_reference
from .suites import get_suite


MIX_HEURISTIC_PROBABILITY = 0.9


@dataclass(frozen=True)
class StarcConfig:
    suite: str
    env_id: str
    partial: str
    timesteps: int = 100_000
    gamma: float = 0.99
    seed: int = 0
    policy: str = "mix"  # random | mix (90% heuristic actions, 10% random)
    value_features: int = 512
    bootstrap_samples: int = 300
    canonicalisation: str = "minimal"  # minimal | val


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    next_state: np.ndarray
    true_reward: float
    partial_reward: float
    done: bool
    step_index: int
    max_episode_steps: int


@dataclass
class RandomFeatureValueModel:
    state_mean: np.ndarray
    state_std: np.ndarray
    random_weights: np.ndarray
    random_bias: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    coefficients: np.ndarray

    def predict(self, states: np.ndarray) -> np.ndarray:
        design = _design_matrix(
            states,
            self.state_mean,
            self.state_std,
            self.random_weights,
            self.random_bias,
            self.feature_mean,
            self.feature_std,
        )
        return design @ self.coefficients


def estimate_starc(config: StarcConfig) -> dict[str, Any]:
    """Estimate a held-out VAL/L2 STARC distance.

    The returned ``starc_distance`` is lower-is-better and lies in [0, 2].
    ``starc_alignment`` is the display-friendly ``1 - distance / 2``.
    """

    _validate_config(config)
    partial_spec = load_partial_reference(config.partial, config.suite, PartialRegistry())
    episodes = collect_starc_episodes(config, partial_spec)
    if len(episodes) < 8:
        raise ConfigError(
            f"STARC collected only {len(episodes)} complete episodes; increase --timesteps"
        )

    if config.canonicalisation == "minimal":
        train_episodes = episodes
        evaluation_episodes = episodes
        canonical_by_episode = minimal_canonical_rewards(episodes, config)
        value_diagnostics = _shaping_removed_diagnostics(episodes, canonical_by_episode)
    else:
        train_episodes, evaluation_episodes = _split_episodes(episodes, config.seed)
        train_states, train_targets = _value_training_data(train_episodes, config.gamma)
        model = fit_value_model(
            train_states,
            train_targets,
            feature_count=config.value_features,
            seed=config.seed,
        )
        canonical_by_episode = [
            _canonical_rewards(episode, model, config.gamma)
            for episode in evaluation_episodes
        ]
        evaluation_states, evaluation_targets = _value_training_data(
            evaluation_episodes, config.gamma
        )
        value_predictions = model.predict(evaluation_states)
        value_diagnostics = {
            "true": _regression_diagnostics(
                evaluation_targets[:, 0], value_predictions[:, 0]
            ),
            "partial": _regression_diagnostics(
                evaluation_targets[:, 1], value_predictions[:, 1]
            ),
        }

    canonical_true = np.concatenate([pair[0] for pair in canonical_by_episode])
    canonical_partial = np.concatenate([pair[1] for pair in canonical_by_episode])
    distance, alignment, true_norm, partial_norm = normalized_l2_distance(
        canonical_true, canonical_partial
    )
    ci_low, ci_high = _bootstrap_distance_interval(
        canonical_by_episode,
        config.bootstrap_samples,
        config.seed + 1,
    )

    return {
        "suite": config.suite,
        "env_id": config.env_id,
        "partial": partial_spec.name,
        "method": (
            "approximate_minimal_L2_STARC"
            if config.canonicalisation == "minimal"
            else "approximate_VALPotential_L2_STARC"
        ),
        "canonicalisation": config.canonicalisation,
        "starc_distance": distance,
        "starc_distance_scale": "0=equivalent, 2=opposite",
        "starc_alignment": alignment,
        "starc_alignment_scale": "1=equivalent, 0=opposite",
        "distance_bootstrap_95_ci": [ci_low, ci_high],
        "canonical_true_rms": true_norm,
        "canonical_partial_rms": partial_norm,
        "gamma": config.gamma,
        "reference_policy": config.policy,
        "heuristic_action_probability": (
            MIX_HEURISTIC_PROBABILITY if config.policy == "mix" else 0.0
        ),
        "timesteps_requested": config.timesteps,
        "complete_episodes": len(episodes),
        "training_episodes": len(train_episodes),
        "evaluation_episodes": len(evaluation_episodes),
        "training_transitions": sum(len(episode) for episode in train_episodes),
        "evaluation_transitions": sum(len(episode) for episode in evaluation_episodes),
        "value_features": config.value_features,
        "value_diagnostics": value_diagnostics,
        "bootstrap_samples": config.bootstrap_samples,
        "seed": config.seed,
        "warning": (
            "Approximation for ranking rewards within this environment; not an exact "
            "percentage alignment or a replacement for true-reward policy evaluation."
        ),
    }


def collect_starc_episodes(
    config: StarcConfig, partial_spec: PartialSpec
) -> list[list[Transition]]:
    """Collect complete episodes from the suite-faithful LunarLander env."""

    suite = get_suite(config.suite)
    env = suite.make_raw_env(config.env_id)
    partial = partial_spec.create(config.env_id)
    rng = np.random.default_rng(config.seed)
    episodes: list[list[Transition]] = []
    current: list[Transition] = []
    total_steps = 0
    episode_index = 0

    max_episode_steps = int(getattr(env.spec, "max_episode_steps", 1000) or 1000)

    try:
        obs, info = env.reset(seed=config.seed)
        partial.reset(info)
        while total_steps < config.timesteps:
            action = _reference_action(env, obs, rng, config.policy)
            next_obs, true_reward, terminated, truncated, info = env.step(action)
            partial_step = partial.step(
                obs,
                action,
                next_obs,
                float(true_reward),
                terminated,
                truncated,
                info,
            )
            done = bool(terminated or truncated)
            current.append(
                Transition(
                    state=np.asarray(obs, dtype=np.float64).reshape(-1).copy(),
                    next_state=np.asarray(next_obs, dtype=np.float64).reshape(-1).copy(),
                    true_reward=float(true_reward),
                    partial_reward=float(partial_step.partial),
                    done=done,
                    step_index=len(current),
                    max_episode_steps=max_episode_steps,
                )
            )
            total_steps += 1

            if done:
                episodes.append(current)
                current = []
                episode_index += 1
                obs, info = env.reset(seed=config.seed + episode_index)
                partial.reset(info)
            else:
                obs = next_obs
    finally:
        env.close()

    # An unfinished episode has a censored return and is intentionally excluded.
    return episodes


def fit_value_model(
    states: np.ndarray,
    targets: np.ndarray,
    *,
    feature_count: int,
    seed: int,
    ridge: float = 1e-3,
) -> RandomFeatureValueModel:
    """Fit both value functions with one shared linear approximation operator.

    Using the same feature map and solve for both rewards preserves scaling and
    sign relationships instead of introducing differences from two unrelated
    neural-network fits.
    """

    states = np.asarray(states, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if states.ndim != 2 or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("states must be 2D and targets must have two columns")

    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0)
    state_std[state_std < 1e-8] = 1.0
    normalised = (states - state_mean) / state_std

    rng = np.random.default_rng(seed)
    random_weights = rng.normal(
        0.0, 1.0 / math.sqrt(states.shape[1]), size=(states.shape[1], feature_count)
    )
    random_bias = rng.uniform(-math.pi, math.pi, size=feature_count)
    raw_features = _raw_features(normalised, random_weights, random_bias)
    feature_mean = raw_features.mean(axis=0)
    feature_std = raw_features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    design = np.column_stack(
        [np.ones(len(states)), (raw_features - feature_mean) / feature_std]
    )

    penalty = np.eye(design.shape[1], dtype=np.float64) * (ridge * len(states))
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return RandomFeatureValueModel(
        state_mean=state_mean,
        state_std=state_std,
        random_weights=random_weights,
        random_bias=random_bias,
        feature_mean=feature_mean,
        feature_std=feature_std,
        coefficients=coefficients,
    )


def minimal_canonical_rewards(
    episodes: list[list[Transition]], config: StarcConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Canonicalise by solving directly for the potential that best cancels R.

    The VAL path fits ``V`` to Monte-Carlo returns and then hopes that
    ``R + gamma*V(s') - V(s)`` has the shaping removed.  That is indirect: the
    regression optimises return prediction, not shaping removal, and a mediocre
    fit leaves shaping behind.  Here we minimise the thing we actually want,

        min_w || r + (gamma*Psi(s') - Psi(s)) w ||^2 ,

    which is an ordinary least-squares problem whose residual *is* the canonical
    reward.  This is the paper's *minimal* canonicalisation (Prop. 3: for any
    weighted L2 norm a minimal canonicalisation exists and is unique) restricted
    to ``span(Psi)``, and the paper notes minimal canonicalisations give tighter
    regret bounds than VAL.

    It keeps every property the metric needs.  ``w`` is linear in ``r``, so
    ``c`` is linear.  And if ``R' = R + gamma*Phi(s') - Phi(s)`` with
    ``Phi in span(Psi)``, then ``w(R') = w(R) - w_Phi`` and the residuals are
    *identical* -- representable shaping is removed exactly, not approximately.

    ``Phi(s') = 0`` is forced at terminal steps, which is what makes potential
    shaping policy-invariant in an episodic task (Ng et al. 1999).
    """

    states = np.asarray(
        [_state_with_time(t, next_state=False) for e in episodes for t in e]
    )
    next_states = np.asarray(
        [_state_with_time(t, next_state=True) for e in episodes for t in e]
    )
    not_done = np.asarray(
        [0.0 if t.done else 1.0 for e in episodes for t in e], dtype=np.float64
    )
    rewards = np.asarray(
        [(t.true_reward, t.partial_reward) for e in episodes for t in e],
        dtype=np.float64,
    )

    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std[std < 1e-8] = 1.0
    rng = np.random.default_rng(config.seed)
    random_weights = rng.normal(
        0.0, 1.0 / math.sqrt(states.shape[1]), size=(states.shape[1], config.value_features)
    )
    random_bias = rng.uniform(-math.pi, math.pi, size=config.value_features)

    def potential_basis(raw: np.ndarray) -> np.ndarray:
        z = (raw - mean) / std
        return np.column_stack(
            [np.ones(len(z)), z, np.square(z), np.tanh(z @ random_weights + random_bias)]
        )

    design = config.gamma * not_done[:, None] * potential_basis(next_states) - potential_basis(
        states
    )
    gram = design.T @ design + 1e-6 * len(design) * np.eye(design.shape[1])
    canonical = rewards - design @ np.linalg.solve(gram, design.T @ rewards)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    offset = 0
    for episode in episodes:
        block = canonical[offset : offset + len(episode)]
        out.append((block[:, 0].copy(), block[:, 1].copy()))
        offset += len(episode)
    return out


def _shaping_removed_diagnostics(
    episodes: list[list[Transition]],
    canonical_by_episode: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, float | None]]:
    """How much of each reward the fitted potential absorbed.

    ``residual_fraction`` near 0 means the reward was almost pure shaping and
    the canonical form is mostly estimation noise; near 1 means the potential
    explained little and the canonical reward carries the signal.
    """

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0

    raw_true = np.asarray([t.true_reward for e in episodes for t in e])
    raw_partial = np.asarray([t.partial_reward for e in episodes for t in e])
    canonical_true = np.concatenate([pair[0] for pair in canonical_by_episode])
    canonical_partial = np.concatenate([pair[1] for pair in canonical_by_episode])
    out = {}
    for key, raw, canonical in (
        ("true", raw_true, canonical_true),
        ("partial", raw_partial, canonical_partial),
    ):
        raw_rms = rms(raw)
        out[key] = {
            "raw_rms": raw_rms,
            "canonical_rms": rms(canonical),
            "residual_fraction": (
                None if raw_rms <= 1e-12 else rms(canonical) / raw_rms
            ),
        }
    return out


def normalized_l2_distance(
    canonical_true: np.ndarray, canonical_partial: np.ndarray
) -> tuple[float, float, float, float]:
    """Return STARC distance, display alignment, and the two RMS norms."""

    true_values = np.asarray(canonical_true, dtype=np.float64).reshape(-1)
    partial_values = np.asarray(canonical_partial, dtype=np.float64).reshape(-1)
    if true_values.shape != partial_values.shape or not len(true_values):
        raise ValueError("canonical rewards must be non-empty arrays of equal length")

    true_norm = float(np.sqrt(np.mean(np.square(true_values))))
    partial_norm = float(np.sqrt(np.mean(np.square(partial_values))))
    true_standard = true_values / true_norm if true_norm > 1e-12 else np.zeros_like(true_values)
    partial_standard = (
        partial_values / partial_norm
        if partial_norm > 1e-12
        else np.zeros_like(partial_values)
    )
    distance = float(np.sqrt(np.mean(np.square(true_standard - partial_standard))))
    distance = float(np.clip(distance, 0.0, 2.0))
    alignment = float(np.clip(1.0 - distance / 2.0, 0.0, 1.0))
    return distance, alignment, true_norm, partial_norm


def starc_json(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)


def default_starc_output_path(metrics: dict[str, Any]) -> Path:
    env_slug = _slugify(str(metrics["env_id"]))
    partial_slug = _slugify(str(metrics["partial"]))
    return (
        Path("logs")
        / "starc"
        / env_slug
        / f"{partial_slug}_seed{metrics['seed']}_steps{metrics['timesteps_requested']}.json"
    )


def save_starc_result(
    metrics: dict[str, Any], output: str | Path | None = None
) -> Path:
    path = Path(output) if output is not None else default_starc_output_path(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(starc_json(metrics) + "\n", encoding="utf-8")
    return path


def _validate_config(config: StarcConfig) -> None:
    if config.suite != "box2d" or config.env_id != "LunarLander-v3":
        raise ConfigError(
            "Approximate STARC currently supports only --suite box2d "
            "--env-id LunarLander-v3"
        )
    if config.timesteps < 500:
        raise ConfigError("STARC requires at least 500 timesteps")
    if not 0.0 < config.gamma < 1.0:
        raise ConfigError("STARC gamma must be between 0 and 1")
    if config.policy not in {"random", "mix"}:
        raise ConfigError("STARC policy must be random or mix")
    if config.value_features < 8:
        raise ConfigError("STARC value_features must be at least 8")
    if config.canonicalisation not in {"minimal", "val"}:
        raise ConfigError("STARC canonicalisation must be minimal or val")
    if config.bootstrap_samples < 0:
        raise ConfigError("STARC bootstrap_samples cannot be negative")


def _reference_action(env, observation, rng: np.random.Generator, policy: str):
    if policy == "mix" and rng.random() < MIX_HEURISTIC_PROBABILITY:
        from gymnasium.envs.box2d.lunar_lander import heuristic

        return heuristic(env.unwrapped, observation)
    return int(rng.integers(env.action_space.n))


def _split_episodes(
    episodes: list[list[Transition]], seed: int
) -> tuple[list[list[Transition]], list[list[Transition]]]:
    order = np.random.default_rng(seed + 17).permutation(len(episodes))
    split = int(round(0.75 * len(order)))
    split = min(max(split, 4), len(order) - 2)
    return [episodes[index] for index in order[:split]], [
        episodes[index] for index in order[split:]
    ]


def _state_with_time(transition: Transition, *, next_state: bool) -> np.ndarray:
    state = transition.next_state if next_state else transition.state
    step = transition.step_index + (1 if next_state else 0)
    time_fraction = min(step / max(transition.max_episode_steps, 1), 1.0)
    return np.concatenate([state, np.asarray([time_fraction], dtype=np.float64)])


def _value_training_data(
    episodes: list[list[Transition]], gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    targets: list[tuple[float, float]] = []
    for episode in episodes:
        true_return = 0.0
        partial_return = 0.0
        episode_targets: list[tuple[float, float]] = []
        for transition in reversed(episode):
            true_return = transition.true_reward + gamma * true_return
            partial_return = transition.partial_reward + gamma * partial_return
            episode_targets.append((true_return, partial_return))
        episode_targets.reverse()
        for transition, target in zip(episode, episode_targets):
            states.append(_state_with_time(transition, next_state=False))
            targets.append(target)
    return np.asarray(states, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _canonical_rewards(
    episode: list[Transition], model: RandomFeatureValueModel, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(
        [_state_with_time(transition, next_state=False) for transition in episode]
    )
    next_states = np.asarray(
        [_state_with_time(transition, next_state=True) for transition in episode]
    )
    values = model.predict(states)
    next_values = model.predict(next_states)
    not_done = np.asarray([not transition.done for transition in episode], dtype=np.float64)
    rewards = np.asarray(
        [
            (transition.true_reward, transition.partial_reward)
            for transition in episode
        ],
        dtype=np.float64,
    )
    canonical = rewards + gamma * not_done[:, None] * next_values - values
    return canonical[:, 0], canonical[:, 1]


def _raw_features(
    normalised_states: np.ndarray,
    random_weights: np.ndarray,
    random_bias: np.ndarray,
) -> np.ndarray:
    random_features = np.tanh(normalised_states @ random_weights + random_bias)
    return np.column_stack([normalised_states, np.square(normalised_states), random_features])


def _design_matrix(
    states: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    random_weights: np.ndarray,
    random_bias: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    normalised = (states - state_mean) / state_std
    raw = _raw_features(normalised, random_weights, random_bias)
    return np.column_stack([np.ones(len(states)), (raw - feature_mean) / feature_std])


def _regression_diagnostics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(np.square(target - prediction))))
    variance = float(np.mean(np.square(target - target.mean())))
    r2 = None if variance <= 1e-12 else float(1.0 - np.mean(np.square(target - prediction)) / variance)
    return {"rmse": rmse, "r2": r2}


def _bootstrap_distance_interval(
    canonical_by_episode: list[tuple[np.ndarray, np.ndarray]],
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if samples == 0 or len(canonical_by_episode) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(samples):
        selected = rng.integers(0, len(canonical_by_episode), size=len(canonical_by_episode))
        true_values = np.concatenate([canonical_by_episode[index][0] for index in selected])
        partial_values = np.concatenate([canonical_by_episode[index][1] for index in selected])
        distances.append(normalized_l2_distance(true_values, partial_values)[0])
    return float(np.percentile(distances, 2.5)), float(np.percentile(distances, 97.5))


def _slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
