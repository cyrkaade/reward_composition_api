"""The alignment ladder must stay a faithful decomposition of the true reward.

The property that makes the ladder interpretable is that all eight weights at
1.0 reproduces the environment's reward exactly.  If a future edit changes a
constant, mis-signs a term, or drops the terminal replacement, that identity
breaks and this test catches it.
"""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("Box2D")

from rcomp.partials import PartialRegistry, load_partial_reference  # noqa: E402

ENV_ID = "LunarLander-v3"
RUNGS = ("lla_a00", "lla_a30", "lla_a50", "lla_a75", "lla_a95", "lla_full")

# Observations are float32 while the env computes shaping in float64, so a
# difference of a few 1e-5 on a ~100-magnitude potential is the floor, not a bug.
OBS_ROUNDING = 1e-3


def _make(name):
    ref = f"lunar_lander_alignment:{name}"
    return load_partial_reference(ref, "box2d", PartialRegistry()).create(ENV_ID)


def _rollout(partial, episodes, seed0, policy):
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed0)
    true, part, terminals = [], [], []
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed0 + ep)
            partial.reset(info)
            for _ in range(1000):
                action = policy(env, obs, rng)
                nobs, reward, term, trunc, inf = env.step(action)
                true.append(float(reward))
                part.append(float(partial.step(obs, action, nobs, float(reward),
                                               term, trunc, inf).partial))
                if term:
                    terminals.append((float(reward), float(part[-1])))
                obs = nobs
                if term or trunc:
                    break
    finally:
        env.close()
    return np.array(true), np.array(part), terminals


def _random(env, obs, rng):
    return int(rng.integers(0, 4))


def _heuristic(env, obs, rng):
    from gymnasium.envs.box2d.lunar_lander import heuristic

    return heuristic(env.unwrapped, obs)


@pytest.mark.parametrize("policy", [_random, _heuristic], ids=["random", "heuristic"])
def test_all_weights_one_reproduces_the_true_reward(policy):
    true, part, terminals = _rollout(_make("lla_full"), 12, 4242, policy)
    assert len(true) > 500
    assert np.abs(true - part).max() < OBS_ROUNDING
    # And specifically at the terminal step, which is a replacement not a sum.
    for env_reward, partial_reward in terminals:
        assert env_reward == pytest.approx(partial_reward, abs=OBS_ROUNDING)


def test_terminal_classification_matches_the_env_on_landings_and_crashes():
    """+100 iff Box2D put the lander to sleep; the ladder infers that from state."""
    _, _, terminals = _rollout(_make("lla_full"), 40, 1000, _heuristic)
    signs = {np.sign(env_r) for env_r, _ in terminals}
    assert signs == {1.0, -1.0}, f"need both landings and crashes, saw {signs}"
    for env_reward, partial_reward in terminals:
        assert np.sign(env_reward) == np.sign(partial_reward)
        assert abs(env_reward) == pytest.approx(100.0)


@pytest.mark.parametrize("name", RUNGS)
def test_every_rung_resolves_and_runs(name):
    partial = _make(name)
    true, part, _ = _rollout(partial, 3, 7, _random)
    assert len(part) == len(true)
    assert np.isfinite(part).all()


def test_a_zero_weight_removes_exactly_that_term():
    """Zeroing one weight must change the partial by that term alone."""
    from partials.lunar_lander_alignment import LunarLanderWeightedPartial

    # Both built raw so both return the plain dict; the registry wraps it.
    full = LunarLanderWeightedPartial()
    no_tilt = LunarLanderWeightedPartial(tilt=0.0)
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(3)
    obs, info = env.reset(seed=3)
    full.reset(info)
    no_tilt.reset(info)
    diffs = []
    try:
        for _ in range(300):
            action = int(rng.integers(0, 4))
            nobs, reward, term, trunc, inf = env.step(action)
            a = full.step(obs, action, nobs, float(reward), term, trunc, inf)["partial"]
            b = no_tilt.step(obs, action, nobs, float(reward), term, trunc, inf)["partial"]
            if not term:
                # The removed term is -100*(|ang'| - |ang|); reconstruct it.
                expected = -100.0 * (abs(float(nobs[4])) - abs(float(obs[4])))
                diffs.append(abs((a - b) - expected))
            obs = nobs
            if term or trunc:
                break
    finally:
        env.close()
    assert diffs
    assert max(diffs) < 1e-4


def test_fuel_term_is_reconstructed_from_the_discrete_action():
    from partials.lunar_lander_alignment import (
        MAIN_ENGINE_COST,
        SIDE_ENGINE_COST,
        LunarLanderWeightedPartial,
    )

    partial = LunarLanderWeightedPartial()
    assert partial._powers(0) == (0.0, 0.0)
    assert partial._powers(1) == (0.0, 1.0)
    assert partial._powers(2) == (1.0, 0.0)
    assert partial._powers(3) == (0.0, 1.0)
    assert MAIN_ENGINE_COST == 0.30
    assert SIDE_ENGINE_COST == 0.03
