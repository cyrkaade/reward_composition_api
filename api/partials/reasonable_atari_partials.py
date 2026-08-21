"""RAM-only, aligned Atari candidates for the reasonable-prior screen.

The policy and learned reward model continue to receive only stacked 84x84
grayscale pixels.  These hand-written partials receive synchronized 128-byte ALE
RAM snapshots through AtariSuite's private partial-reward path and never inspect
``true_reward``.

RAM addresses follow the AtariARI annotations already used by
``atari_ram_screen.py``.  Every candidate contains direct task progress: pellets
for MsPacman, cube-color changes for Qbert, and the agent's score for Pong.  The
screen therefore excludes motion-only and rally-only proxies.
"""

from __future__ import annotations

import numpy as np


def _ram(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.uint8).reshape(-1)
    if result.size != 128:
        raise ValueError(f"Atari RAM partial expected 128 bytes, got {result.size}")
    return result


class MsPacmanReasonableRamPartial:
    component_keys = ("pellets", "coverage", "safety_progress")

    def __init__(self, *, visit_weight: float = 0.0, safety_weight: float = 0.0):
        self.visit_weight = float(visit_weight)
        self.safety_weight = float(safety_weight)
        self.visited: set[tuple[int, int]] = set()

    def reset(self, info: dict | None = None) -> None:
        self.visited.clear()

    @staticmethod
    def _position(ram: np.ndarray) -> tuple[int, int]:
        return int(ram[10]), int(ram[16])

    @staticmethod
    def _ghost_distance(ram: np.ndarray, player: tuple[int, int]) -> float:
        px, py = player
        return float(min(
            abs(px - int(ram[x_index])) + abs(py - int(ram[y_index]))
            for x_index, y_index in zip(range(6, 10), range(12, 16))
        ))

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        previous, current = _ram(obs), _ram(next_obs)
        old_position, position = self._position(previous), self._position(current)
        dot_delta = int(current[119]) - int(previous[119])
        pellets = float(dot_delta if 0 < dot_delta <= 4 else 0)
        cell = (position[0] // 4, position[1] // 4)
        coverage = self.visit_weight if position != (0, 0) and cell not in self.visited else 0.0
        self.visited.add(cell)
        safety_change = self._ghost_distance(current, position) - self._ghost_distance(previous, old_position)
        safety = self.safety_weight * float(np.clip(safety_change / 8.0, -1.0, 1.0))
        return {
            "partial": float(pellets + coverage + safety),
            "components": {"pellets": pellets, "coverage": coverage, "safety_progress": safety},
        }


QBERT_TILE_COLOR_INDICES = (21, 52, 54, 83, 85, 87, 98, 100, 102, 104, 1, 3, 5, 7, 9, 32, 34, 36, 38, 40, 42)


class QbertReasonableRamPartial:
    component_keys = ("tile_progress", "coverage")

    def __init__(self, *, visit_weight: float = 0.0):
        self.visit_weight = float(visit_weight)
        self.visited: set[tuple[int, int]] = set()

    def reset(self, info: dict | None = None) -> None:
        self.visited.clear()

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        previous, current = _ram(obs), _ram(next_obs)
        changes = int(np.count_nonzero(
            current[list(QBERT_TILE_COLOR_INDICES)] != previous[list(QBERT_TILE_COLOR_INDICES)]
        ))
        tile_progress = float(min(changes, 3))
        position = (int(current[43]), int(current[67]))
        cell = (position[0] // 4, position[1] // 4)
        coverage = self.visit_weight if position != (0, 0) and cell not in self.visited else 0.0
        self.visited.add(cell)
        return {
            "partial": float(tile_progress + coverage),
            "components": {"tile_progress": tile_progress, "coverage": coverage},
        }


class PongReasonableRamPartial:
    component_keys = ("score_progress", "tracking_progress", "rally_bonus")

    def __init__(self, *, tracking_weight: float = 0.0, rally_bonus: float = 0.0):
        self.tracking_weight = float(tracking_weight)
        self.rally_bonus = float(rally_bonus)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        previous, current = _ram(obs), _ram(next_obs)
        agent_score_delta = int(current[14]) - int(previous[14])
        score_progress = 1.0 if agent_score_delta == 1 else 0.0
        old_distance = abs(int(previous[51]) - int(previous[54]))
        distance = abs(int(current[51]) - int(current[54]))
        tracking = self.tracking_weight * float(np.clip((old_distance - distance) / 8.0, -1.0, 1.0))
        point_changed = bool(current[13] != previous[13] or current[14] != previous[14])
        rally = 0.0 if point_changed else self.rally_bonus
        return {
            "partial": float(score_progress + tracking + rally),
            "components": {
                "score_progress": score_progress,
                "tracking_progress": tracking,
                "rally_bonus": rally,
            },
        }


_PARTIALS: dict[str, tuple[str, type, dict]] = {}


def _add_family(prefix: str, env_id: str, cls: type, variants: dict[str, dict]) -> None:
    for suffix, kwargs in variants.items():
        _PARTIALS[f"{prefix}_{suffix}"] = (env_id, cls, kwargs)


_add_family("rmsp", "ALE/MsPacman-v5", MsPacmanReasonableRamPartial, {
    "pellets": {},
    "visit02": {"visit_weight": 0.02},
    "visit05": {"visit_weight": 0.05},
    "visit10": {"visit_weight": 0.10},
    "visit25": {"visit_weight": 0.25},
    "safe05": {"safety_weight": 0.05},
    "safe10": {"safety_weight": 0.10},
    "visit10_safe05": {"visit_weight": 0.10, "safety_weight": 0.05},
})

_add_family("rqb", "ALE/Qbert-v5", QbertReasonableRamPartial, {
    "tiles": {},
    "visit01": {"visit_weight": 0.01},
    "visit02": {"visit_weight": 0.02},
    "visit05": {"visit_weight": 0.05},
    "visit10": {"visit_weight": 0.10},
    "visit20": {"visit_weight": 0.20},
    "visit40": {"visit_weight": 0.40},
    "visit75": {"visit_weight": 0.75},
})

_add_family("rpong", "ALE/Pong-v5", PongReasonableRamPartial, {
    "score": {},
    "track05": {"tracking_weight": 0.05},
    "track10": {"tracking_weight": 0.10},
    "track25": {"tracking_weight": 0.25},
    "track50": {"tracking_weight": 0.50},
    "rally002": {"rally_bonus": 0.002},
    "rally005": {"rally_bonus": 0.005},
    "track10_rally005": {"tracking_weight": 0.10, "rally_bonus": 0.005},
})


def register(registry) -> None:
    for name, (env_id, cls, kwargs) in _PARTIALS.items():
        registry.register(
            name=name,
            suite="atari",
            factory=lambda requested_env, cls=cls, kwargs=kwargs: cls(**kwargs),
            description=f"RAM-only aligned incomplete candidate for the reasonable-prior screen: {name}",
            env_ids=(env_id,),
            component_keys=cls.component_keys,
        )
