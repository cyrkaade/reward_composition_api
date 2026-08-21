"""Safety and wiring checks for the broad reasonable-prior calibration grid."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rcomp.partials import PartialRegistry, load_partial_reference


ENV = {
    "ll": ("box2d", "LunarLander-v3", 8, None),
    "bipedal": ("box2d", "BipedalWalker-v3", 24, 4),
    "ant": ("mujoco", "Ant-v5", 105, 8),
    "reacher": ("mujoco", "Reacher-v5", 10, 2),
    "pusher": ("mujoco", "Pusher-v5", 23, 7),
    "hopper": ("mujoco", "Hopper-v5", 11, 3),
    "swimmer": ("mujoco", "Swimmer-v5", 8, 2),
    "walker": ("mujoco", "Walker2d-v5", 17, 6),
    "mspacman": ("atari", "ALE/MsPacman-v5", 128, None),
    "qbert": ("atari", "ALE/Qbert-v5", 128, None),
    "pong": ("atari", "ALE/Pong-v5", 128, None),
}


def _rows():
    return [line.split() for line in Path("jobs/params_reasonable.txt").read_text(encoding="utf-8").splitlines()]


def _step(cell: str, true_reward: float):
    _, _, obs_dim, action_dim = ENV[cell]
    if cell in {"mspacman", "qbert", "pong"}:
        obs = np.zeros(128, dtype=np.uint8)
        next_obs = obs.copy()
        if cell == "mspacman":
            obs[[6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 119]] = (20, 30, 40, 50, 40, 20, 30, 40, 50, 60, 3)
            next_obs[[6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 119]] = (20, 30, 40, 50, 44, 20, 30, 40, 50, 60, 4)
        elif cell == "qbert":
            obs[[21, 43, 67]] = (1, 40, 60)
            next_obs[[21, 43, 67]] = (2, 48, 68)
        else:
            obs[[13, 14, 51, 54]] = (1, 2, 40, 60)
            next_obs[[13, 14, 51, 54]] = (1, 3, 50, 54)
        action = 0
    else:
        obs = np.full(obs_dim, 0.1, dtype=np.float64)
        next_obs = obs.copy()
        action = 2 if action_dim is None else np.full(action_dim, 0.2, dtype=np.float64)
        if cell == "pusher":
            obs[14:17], obs[17:20], obs[20:23] = (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4, 0.0, 0.0)
            next_obs[:] = obs
            next_obs[17] = 0.18
    info = {
        "reward_dist": -0.2,
        "reward_near": -0.1,
        "reward_ctrl": -0.05,
        "reward_survive": 1.0,
        "reward_forward": 0.3,
        "x_velocity": 0.3,
    }
    return dict(
        obs=obs, action=action, next_obs=next_obs, true_reward=true_reward,
        terminated=False, truncated=False, info=info,
    )


def test_parameter_grid_has_three_requested_arm_types_and_paired_seeds():
    rows = _rows()
    assert len(rows) == 550
    counts = Counter(row[0] for row in rows)
    assert counts == {cell: 50 for cell in ENV}
    by_arm = defaultdict(list)
    for cell, variant, seed, reference in rows:
        by_arm[(cell, variant)].append(int(seed))
        if variant in ("true", "vanilla"):
            assert reference == "-"
        elif cell in {"mspacman", "qbert", "pong"}:
            assert reference.startswith("reasonable_atari_partials:")
        else:
            assert reference.startswith("reasonable_partials:")
    for seeds in by_arm.values():
        assert sorted(seeds) == list(range(5))
    for cell in ENV:
        variants = {variant for this_cell, variant in by_arm if this_cell == cell}
        assert {"true", "vanilla"} <= variants
        assert len(variants - {"true", "vanilla"}) == 8


def test_every_candidate_resolves_is_finite_and_ignores_true_reward():
    references = {(cell, reference) for cell, variant, seed, reference in _rows() if reference != "-"}
    assert len(references) == 88
    for cell, reference in references:
        suite, env_id, _, _ = ENV[cell]
        spec_a = load_partial_reference(reference, suite, PartialRegistry())
        spec_b = load_partial_reference(reference, suite, PartialRegistry())
        partial_a, partial_b = spec_a.create(env_id), spec_b.create(env_id)
        partial_a.reset({})
        partial_b.reset({})
        low = partial_a.step(**_step(cell, -999.0))
        high = partial_b.step(**_step(cell, 999.0))
        assert math.isfinite(low.partial), reference
        assert all(math.isfinite(value) for value in low.components.values()), reference
        assert low.partial == high.partial, reference
        assert low.components == high.components, reference


def test_locomotion_caps_are_monotone_not_target_penalties():
    prefixes = ("rhop_", "rwalk_", "rant_", "rswm_", "rbw_")
    for cell, reference in {(row[0], row[3]) for row in _rows() if row[3] != "-"}:
        name = reference.rsplit(":", 1)[-1]
        if not name.startswith(prefixes):
            continue
        suite, env_id, _, _ = ENV[cell]
        low_partial = load_partial_reference(reference, suite, PartialRegistry()).create(env_id)
        high_partial = load_partial_reference(reference, suite, PartialRegistry()).create(env_id)
        low_step, high_step = _step(cell, 0.0), _step(cell, 0.0)
        if cell == "bipedal":
            low_step["next_obs"][2] = 0.01
            high_step["next_obs"][2] = 0.20
        else:
            low_step["info"]["reward_forward"] = low_step["info"]["x_velocity"] = 0.01
            high_step["info"]["reward_forward"] = high_step["info"]["x_velocity"] = 5.0
        low = low_partial.step(**low_step).partial
        high = high_partial.step(**high_step).partial
        assert high >= low, reference
