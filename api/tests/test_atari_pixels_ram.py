"""Atari pixel-policy/RM and private-RAM partial regression tests."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from rcomp.partials import PartialRegistry, PartialSpec, load_partial_reference
from rcomp.rewards.model import PixelRewardModel
from rcomp.rewards.wrapper import LearnedRewardRuntime, PreferenceRewardWrapper


class _RamInfoPixelEnv(gym.Env):
    observation_space = spaces.Box(0, 255, shape=(4, 8, 8), dtype=np.uint8)
    action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {"_partial_observation": np.arange(128, dtype=np.uint8)}

    def step(self, action):
        pixels = np.full(self.observation_space.shape, 255, dtype=np.uint8)
        ram = np.arange(128, dtype=np.uint8)
        ram[10] = 99
        return pixels, 7.0, False, False, {"_partial_observation": ram}


class _RecorderPartial:
    def __init__(self):
        self.calls = []

    def reset(self, info=None):
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        self.calls.append((np.asarray(obs).copy(), np.asarray(next_obs).copy()))
        return 1.0


def test_preference_wrapper_routes_ram_only_to_partial_and_pixels_to_policy():
    recorder = _RecorderPartial()
    spec = PartialSpec(name="record", suite="atari", factory=lambda env_id: recorder)
    runtime = LearnedRewardRuntime(
        env_id="ALE/Fake-v5",
        composition="partial",
        observation_space=_RamInfoPixelEnv.observation_space,
        action_space=_RamInfoPixelEnv.action_space,
        observation_features=lambda space, obs: np.asarray(obs, dtype=np.float32).reshape(-1) / 255.0,
        custom_partial=spec,
    )
    env = PreferenceRewardWrapper(_RamInfoPixelEnv(), runtime)
    observation, info = env.reset()
    next_observation, reward, _, _, info = env.step(0)

    assert observation.shape == (4, 8, 8)
    assert next_observation.shape == (4, 8, 8)
    assert reward == 1.0
    assert recorder.calls[0][0].shape == (128,)
    assert recorder.calls[0][1].shape == (128,)
    assert recorder.calls[0][1][10] == 99
    assert "_partial_observation" not in info


def test_pixel_reward_model_preserves_fragment_dimensions():
    model = PixelRewardModel(
        observation_shape=(4, 84, 84),
        extra_feature_size=7,
        hidden_sizes=(32,),
        predict_partial=True,
        gate_partial=True,
        tanh_output=True,
    )
    features = th.rand(2, 3, 4 * 84 * 84 + 7)
    assert model(features).shape == (2, 3, 1)
    assert model.predict_partial(features).shape == (2, 3, 1)
    assert model.gate(features).shape == (2, 3, 1)


ATARI_CANDIDATES = {
    "ALE/MsPacman-v5": (
        "namsp_pellets", "namsp_pellets_visit", "namsp_visit_motion",
        "namsp_pellets_safe", "namsp_pellets_motion",
    ),
    "ALE/Qbert-v5": (
        "naqbert_tiles", "naqbert_tiles_visit", "naqbert_visit_motion",
        "naqbert_tiles_motion", "naqbert_tiles_visit_motion",
    ),
    "ALE/Pong-v5": (
        "napong_rally", "napong_motion", "napong_track", "napong_score", "napong_score_track",
    ),
}


def _run_ram_partial(env_id: str, name: str, true_reward: float):
    spec = load_partial_reference(f"atari_ram_screen:{name}", "atari", PartialRegistry())
    partial = spec.create(env_id)
    previous = np.zeros(128, dtype=np.uint8)
    current = previous.copy()
    if env_id.endswith("MsPacman-v5"):
        previous[[10, 16, 119]] = (40, 60, 3)
        current[[10, 16, 119]] = (44, 60, 4)
    elif env_id.endswith("Qbert-v5"):
        previous[[43, 67, 21]] = (40, 60, 1)
        current[[43, 67, 21]] = (48, 68, 2)
    else:
        previous[[13, 14, 49, 51, 54]] = (1, 2, 40, 40, 60)
        current[[13, 14, 49, 51, 54]] = (1, 3, 44, 50, 54)
    partial.reset({})
    return partial.step(previous, 0, current, true_reward, False, False, {})


def test_atari_screen_candidates_are_ram_only_and_ignore_true_reward():
    for env_id, names in ATARI_CANDIDATES.items():
        for name in names:
            low = _run_ram_partial(env_id, name, -999.0)
            high = _run_ram_partial(env_id, name, 999.0)
            assert np.isfinite(low.partial)
            assert low.partial == high.partial
            assert low.components == high.components


def test_weighted_sum_is_a_convex_normalized_mixture():
    runtime = LearnedRewardRuntime(
        env_id="ALE/Fake-v5",
        composition="weighted_sum",
        observation_space=_RamInfoPixelEnv.observation_space,
        action_space=_RamInfoPixelEnv.action_space,
        observation_features=lambda space, obs: np.asarray(obs, dtype=np.float32).reshape(-1),
        normalize_partial=True,
        partial_mean=2.0,
        partial_std=2.0,
        partial_alpha=0.25,
    )
    wrapper = PreferenceRewardWrapper(_RamInfoPixelEnv(), runtime)

    # norm(partial=6) = 2 and model_reward is already normalized by model_reward().
    assert wrapper.compose_reward(6.0, 4.0) == 0.25 * 2.0 + 0.75 * 4.0
    assert runtime.model_reward_weight() == 0.75
    runtime.normalize = True
    runtime.output_std = 2.0
    assert runtime.model_prediction_scale() == 0.375
