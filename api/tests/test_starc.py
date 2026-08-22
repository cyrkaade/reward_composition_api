"""STARC estimator: the invariances that define the metric.

A STARC metric is a canonicalisation (which must delete potential shaping), a
normalisation (which must delete positive scaling), and a metric.  The tests
below exercise each of those properties directly rather than only the helper
arithmetic, because a canonicalisation that silently stops removing shaping
still returns plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from rcomp.cli import build_parser
from rcomp.config import ConfigError
from rcomp.starc import (
    StarcConfig,
    Transition,
    _validate_config,
    minimal_canonical_rewards,
    normalized_l2_distance,
)

GAMMA = 0.99


def _synthetic_episodes(seed=0, episodes=12, length=40):
    """Fake rollouts -- no env needed, so these stay fast and deterministic."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        states = rng.normal(size=(length + 1, 8))
        block = []
        for index in range(length):
            done = index == length - 1
            block.append(
                Transition(
                    state=states[index],
                    next_state=states[index + 1],
                    true_reward=float(rng.normal()),
                    partial_reward=0.0,
                    done=done,
                    step_index=index,
                    max_episode_steps=1000,
                )
            )
        out.append(block)
    return out


def _with_partial(episodes, fn):
    """Copy the episodes, setting partial_reward from fn(transition)."""
    return [
        [
            Transition(
                state=t.state,
                next_state=t.next_state,
                true_reward=t.true_reward,
                partial_reward=fn(t),
                done=t.done,
                step_index=t.step_index,
                max_episode_steps=t.max_episode_steps,
            )
            for t in episode
        ]
        for episode in episodes
    ]


def _alignment(episodes, features=64):
    config = StarcConfig(
        suite="box2d",
        env_id="LunarLander-v3",
        partial="unused",
        gamma=GAMMA,
        seed=0,
        value_features=features,
    )
    canonical = minimal_canonical_rewards(episodes, config)
    return normalized_l2_distance(
        np.concatenate([pair[0] for pair in canonical]),
        np.concatenate([pair[1] for pair in canonical]),
    )[1]


def test_normalized_l2_starc_invariances():
    true = np.asarray([-2.0, -0.5, 1.0, 3.0])

    distance, alignment, _, _ = normalized_l2_distance(true, 7.0 * true)
    assert distance == pytest.approx(0.0, abs=1e-12)
    assert alignment == pytest.approx(1.0)

    distance, alignment, _, _ = normalized_l2_distance(true, -true)
    assert distance == pytest.approx(2.0)
    assert alignment == pytest.approx(0.0)


def test_identical_rewards_are_perfectly_aligned():
    episodes = _synthetic_episodes()
    assert _alignment(_with_partial(episodes, lambda t: t.true_reward)) == pytest.approx(
        1.0, abs=1e-6
    )


def test_positive_scaling_is_removed_end_to_end():
    """s(R) = c(R)/n(c(R)) must make 5R indistinguishable from R."""
    episodes = _synthetic_episodes()
    assert _alignment(
        _with_partial(episodes, lambda t: 5.0 * t.true_reward)
    ) == pytest.approx(1.0, abs=1e-6)


def test_negating_a_reward_gives_zero_alignment():
    episodes = _synthetic_episodes()
    assert _alignment(_with_partial(episodes, lambda t: -t.true_reward)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_minimal_canonicalisation_removes_representable_potential_shaping():
    """THE defining property: R and R + gamma*Phi(s') - Phi(s) must be identical.

    The potential basis spans a constant and the raw state, so an affine Phi is
    represented exactly.  What is left is the ridge term in the least-squares
    solve, which biases w by O(lambda) and costs about 5e-4 of alignment -- so
    this is pinned at 1e-3, not at solver precision.  For comparison the VAL
    canonicalisation scores 0.72 on a potential it cannot represent.

    Phi(s') is forced to zero at a terminal step, which is what makes the
    shaping policy-invariant in an episodic task (Ng et al. 1999).
    """
    episodes = _synthetic_episodes()
    weights = np.asarray([0.7, -1.3, 2.0, 0.4, -0.9, 1.1, 0.2, -0.6])
    offset = 3.5

    def shaped(t):
        phi_next = 0.0 if t.done else float(t.next_state @ weights) + offset
        return t.true_reward + GAMMA * phi_next - (float(t.state @ weights) + offset)

    assert _alignment(_with_partial(episodes, shaped)) == pytest.approx(1.0, abs=1e-3)


def test_a_pure_potential_reward_canonicalises_to_nothing():
    """A reward that is only shaping carries no preference information at all.

    Measured: rms 4.21 raw collapses to 3.5e-5 canonical, a factor of ~1.2e5.
    Asserted relatively so the bound stays meaningful if the fixture changes.
    """
    episodes = _synthetic_episodes()
    weights = np.asarray([0.5, 1.0, -0.4, 0.3, 0.8, -1.2, 0.1, 0.6])

    def pure(t):
        phi_next = 0.0 if t.done else float(t.next_state @ weights)
        return GAMMA * phi_next - float(t.state @ weights)

    config = StarcConfig(
        suite="box2d", env_id="LunarLander-v3", partial="unused",
        gamma=GAMMA, seed=0, value_features=64,
    )
    shaped_episodes = _with_partial(episodes, pure)
    canonical = minimal_canonical_rewards(shaped_episodes, config)
    partial = np.concatenate([pair[1] for pair in canonical])
    raw = np.asarray([t.partial_reward for e in shaped_episodes for t in e])
    assert np.sqrt(np.mean(np.square(partial))) < 1e-4 * np.sqrt(np.mean(np.square(raw)))


def test_starc_distance_is_symmetric():
    episodes = _synthetic_episodes(seed=3)
    forward = _with_partial(episodes, lambda t: 0.4 * t.true_reward + 0.2)
    backward = [
        [
            Transition(
                state=t.state, next_state=t.next_state,
                true_reward=t.partial_reward, partial_reward=t.true_reward,
                done=t.done, step_index=t.step_index,
                max_episode_steps=t.max_episode_steps,
            )
            for t in episode
        ]
        for episode in forward
    ]
    assert _alignment(forward) == pytest.approx(_alignment(backward), abs=1e-9)


def test_val_canonicalisation_is_still_available_but_not_the_default():
    assert StarcConfig(suite="box2d", env_id="LunarLander-v3",
                       partial="x").canonicalisation == "minimal"
    with pytest.raises(ConfigError, match="canonicalisation must be"):
        _validate_config(
            StarcConfig(suite="box2d", env_id="LunarLander-v3", partial="x",
                        canonicalisation="epic")
        )


def test_starc_is_explicitly_lunar_lander_only():
    with pytest.raises(ConfigError, match="currently supports only"):
        _validate_config(
            StarcConfig(
                suite="mujoco",
                env_id="Reacher-v5",
                partial="reacher_distance_partial",
            )
        )


def test_starc_cli_defaults_are_safe_and_separate():
    args = build_parser().parse_args(
        ["starc", "--partial", "lunar_lander_alignment:lla_full", "--no-save"]
    )

    assert args.command == "starc"
    assert args.suite == "box2d"
    assert args.env_id == "LunarLander-v3"
    assert args.timesteps == 100_000
    assert args.policy == "mix"
    assert args.no_save is True
    assert args.canonicalisation == "minimal"
    assert args.value_features == 512


def test_starc_cli_help_renders_percent_sign(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["starc", "--help"])

    assert exc_info.value.code == 0
    assert "heuristic 90% of the time" in capsys.readouterr().out
