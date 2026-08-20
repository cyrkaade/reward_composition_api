"""RAM-only Atari priors for the independent non-timid screen.

Policy observations and learned-reward inputs are stacked pixels.  These
partials receive synchronized 128-byte ALE RAM snapshots instead.  None reads
``true_reward`` or a score field: MsPacman uses pellet progress, position
coverage, motion, and ghost separation; Qbert uses cube-color changes,
position coverage, and motion.

RAM indices follow the AtariARI benchmark annotations
(``mila-iqia/atari-representation-learning/.../ram_annotations.py``):
MsPacman player=(10,16), ghosts=(6:10,12:16), dots=119;
Qbert player=(43,67), cube colors=(21 documented bytes);
Pong paddles=(50,51), ball=(49,54), score=(13,14).
"""

from __future__ import annotations

import numpy as np


def _ram(value) -> np.ndarray:
    ram = np.asarray(value, dtype=np.uint8).reshape(-1)
    if ram.size != 128:
        raise ValueError(f"Atari RAM partial expected 128 bytes, got {ram.size}")
    return ram


class MsPacmanRamPartial:
    component_keys = ("pellet_progress", "new_position", "motion", "safety_progress")

    def __init__(
        self,
        *,
        pellet_weight: float = 0.0,
        visit_bonus: float = 0.0,
        motion_weight: float = 0.0,
        safety_weight: float = 0.0,
    ):
        self.pellet_weight = float(pellet_weight)
        self.visit_bonus = float(visit_bonus)
        self.motion_weight = float(motion_weight)
        self.safety_weight = float(safety_weight)
        self.visited: set[tuple[int, int]] = set()

    def reset(self, info: dict | None = None) -> None:
        self.visited.clear()

    @staticmethod
    def _position(ram: np.ndarray) -> tuple[int, int]:
        return int(ram[10]), int(ram[16])

    @staticmethod
    def _ghost_distance(ram: np.ndarray, position: tuple[int, int]) -> float:
        px, py = position
        distances = [abs(px - int(ram[x_index])) + abs(py - int(ram[y_index])) for x_index, y_index in zip(range(6, 10), range(12, 16))]
        return float(min(distances))

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        previous = _ram(obs)
        current = _ram(next_obs)
        previous_position = self._position(previous)
        current_position = self._position(current)

        # A four-frame action can consume several pellets.  Large positive jumps
        # are resets/wraps, not progress, and are deliberately ignored.
        dot_delta = int(current[119]) - int(previous[119])
        pellet_progress = self.pellet_weight * float(dot_delta if 0 < dot_delta <= 4 else 0)

        cell = (current_position[0] // 4, current_position[1] // 4)
        new_position = self.visit_bonus if current_position != (0, 0) and cell not in self.visited else 0.0
        self.visited.add(cell)

        displacement = abs(current_position[0] - previous_position[0]) + abs(current_position[1] - previous_position[1])
        motion = self.motion_weight * min(float(displacement) / 8.0, 1.0)

        safety_change = self._ghost_distance(current, current_position) - self._ghost_distance(previous, previous_position)
        safety_progress = self.safety_weight * float(np.clip(safety_change / 8.0, -1.0, 1.0))
        total = pellet_progress + new_position + motion + safety_progress
        return {
            "partial": float(total),
            "components": {
                "pellet_progress": float(pellet_progress),
                "new_position": float(new_position),
                "motion": float(motion),
                "safety_progress": float(safety_progress),
            },
        }


QBERT_TILE_COLOR_INDICES = (21, 52, 54, 83, 85, 87, 98, 100, 102, 104, 1, 3, 5, 7, 9, 32, 34, 36, 38, 40, 42)


class QbertRamPartial:
    component_keys = ("tile_progress", "new_position", "motion")

    def __init__(self, *, tile_weight: float = 0.0, visit_bonus: float = 0.0, motion_weight: float = 0.0):
        self.tile_weight = float(tile_weight)
        self.visit_bonus = float(visit_bonus)
        self.motion_weight = float(motion_weight)
        self.visited: set[tuple[int, int]] = set()

    def reset(self, info: dict | None = None) -> None:
        self.visited.clear()

    @staticmethod
    def _position(ram: np.ndarray) -> tuple[int, int]:
        return int(ram[43]), int(ram[67])

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        previous = _ram(obs)
        current = _ram(next_obs)
        previous_position = self._position(previous)
        current_position = self._position(current)

        changes = int(np.count_nonzero(current[list(QBERT_TILE_COLOR_INDICES)] != previous[list(QBERT_TILE_COLOR_INDICES)]))
        tile_progress = self.tile_weight * float(min(changes, 3))

        cell = (current_position[0] // 4, current_position[1] // 4)
        new_position = self.visit_bonus if current_position != (0, 0) and cell not in self.visited else 0.0
        self.visited.add(cell)

        displacement = abs(current_position[0] - previous_position[0]) + abs(current_position[1] - previous_position[1])
        motion = self.motion_weight * min(float(displacement) / 16.0, 1.0)
        return {
            "partial": float(tile_progress + new_position + motion),
            "components": {
                "tile_progress": float(tile_progress),
                "new_position": float(new_position),
                "motion": float(motion),
            },
        }


class PongRamPartial:
    """RAM-only Pong proxies ranging from gameable rallies to score+tracking."""

    component_keys = ("score_progress", "tracking_progress", "rally_bonus", "ball_motion")

    def __init__(
        self,
        *,
        score_weight: float = 0.0,
        tracking_weight: float = 0.0,
        rally_bonus: float = 0.0,
        motion_weight: float = 0.0,
    ):
        self.score_weight = float(score_weight)
        self.tracking_weight = float(tracking_weight)
        self.rally_bonus = float(rally_bonus)
        self.motion_weight = float(motion_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        previous = _ram(obs)
        current = _ram(next_obs)

        score_delta = int(current[14]) - int(previous[14])
        score_progress = self.score_weight if score_delta == 1 else 0.0

        previous_distance = abs(int(previous[51]) - int(previous[54]))
        current_distance = abs(int(current[51]) - int(current[54]))
        tracking_change = float(np.clip((previous_distance - current_distance) / 8.0, -1.0, 1.0))
        tracking_progress = self.tracking_weight * tracking_change

        point_changed = current[13] != previous[13] or current[14] != previous[14]
        rally = 0.0 if point_changed else self.rally_bonus
        displacement = abs(int(current[49]) - int(previous[49])) + abs(int(current[54]) - int(previous[54]))
        ball_motion = self.motion_weight * min(float(displacement) / 12.0, 1.0)
        return {
            "partial": float(score_progress + tracking_progress + rally + ball_motion),
            "components": {
                "score_progress": float(score_progress),
                "tracking_progress": float(tracking_progress),
                "rally_bonus": float(rally),
                "ball_motion": float(ball_motion),
            },
        }


_PARTIALS = {
    "namsp_pellets": ("ALE/MsPacman-v5", MsPacmanRamPartial, dict(pellet_weight=1.0)),
    "namsp_pellets_visit": (
        "ALE/MsPacman-v5",
        MsPacmanRamPartial,
        dict(pellet_weight=0.75, visit_bonus=0.25),
    ),
    "namsp_visit_motion": (
        "ALE/MsPacman-v5",
        MsPacmanRamPartial,
        dict(visit_bonus=1.0, motion_weight=0.05),
    ),
    "namsp_pellets_safe": (
        "ALE/MsPacman-v5",
        MsPacmanRamPartial,
        dict(pellet_weight=1.0, safety_weight=0.10),
    ),
    "namsp_pellets_motion": (
        "ALE/MsPacman-v5",
        MsPacmanRamPartial,
        dict(pellet_weight=1.0, visit_bonus=0.10, motion_weight=0.05),
    ),
    "naqbert_tiles": ("ALE/Qbert-v5", QbertRamPartial, dict(tile_weight=1.0)),
    "naqbert_tiles_visit": (
        "ALE/Qbert-v5",
        QbertRamPartial,
        dict(tile_weight=0.75, visit_bonus=0.25),
    ),
    "naqbert_visit_motion": (
        "ALE/Qbert-v5",
        QbertRamPartial,
        dict(visit_bonus=1.0, motion_weight=0.05),
    ),
    "naqbert_tiles_motion": (
        "ALE/Qbert-v5",
        QbertRamPartial,
        dict(tile_weight=1.0, motion_weight=0.05),
    ),
    "naqbert_tiles_visit_motion": (
        "ALE/Qbert-v5",
        QbertRamPartial,
        dict(tile_weight=1.0, visit_bonus=0.10, motion_weight=0.05),
    ),
    "napong_rally": ("ALE/Pong-v5", PongRamPartial, dict(rally_bonus=0.01)),
    "napong_motion": ("ALE/Pong-v5", PongRamPartial, dict(motion_weight=0.05)),
    "napong_track": ("ALE/Pong-v5", PongRamPartial, dict(tracking_weight=1.0)),
    "napong_score": ("ALE/Pong-v5", PongRamPartial, dict(score_weight=1.0)),
    "napong_score_track": (
        "ALE/Pong-v5",
        PongRamPartial,
        dict(score_weight=1.0, tracking_weight=0.5, rally_bonus=0.005),
    ),
}


def register(registry) -> None:
    for name, (env_id, cls, kwargs) in _PARTIALS.items():
        registry.register(
            name=name,
            suite="atari",
            factory=lambda requested_env, cls=cls, kwargs=kwargs: cls(**kwargs),
            description=f"RAM-only independent Atari prior '{name}'",
            env_ids=(env_id,),
            component_keys=cls.component_keys,
        )
