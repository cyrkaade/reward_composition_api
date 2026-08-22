"""RAM-only, aligned Atari candidates for the reasonable-prior screen.

The policy and learned reward model continue to receive only stacked 84x84
grayscale pixels.  These hand-written partials receive synchronized 128-byte ALE
RAM snapshots through AtariSuite's private partial-reward path and never inspect
``true_reward``.

RAM addresses follow the AtariARI annotations already used by
``atari_ram_screen.py``.  Every candidate contains direct task progress: pellets
for MsPacman, cube-color changes for Qbert, the agent's score for Pong, and the
brick counter for Breakout.  The screen therefore excludes motion-only and
rally-only proxies.

The Breakout and Pong addresses were verified against 60k-120k random-policy
steps rather than taken on trust (2026-08-22):

  Breakout  ram[77] +1 on every scoring step, corr(delta, reward) = 1.000
            ram[72] paddle: +5.8 mean delta on RIGHT, -6.4 on LEFT
            ram[99] ball x: |paddle - ball| is 80 in the step before a life is
                    lost against 56 overall, which is the miss signature
  Pong      ram[14] +1 on all 75 agent points and never otherwise
            ram[13] +1 on all 2751 conceded points and never otherwise

PCC against the true reward, per fragment of 25, 40k steps, seeds 0-2, all
candidates stepped on the SAME trajectory so the comparison is paired.  Note
``rcomp partiality`` cannot produce these: it hands the partial the pixel
observation, and the RAM substitution lives in the training wrapper.

  Breakout  rbo_bricks 0.966   track05 0.808  track10 0.582  track25 0.276
                               track50 0.140
  Pong      rpong_score 0.226  rally005 0.252  track05 0.155  track10 0.114
                               track25 0.085  track50 0.074

These are RANDOM-POLICY numbers and both ends are distorted by that, in
opposite directions.  Breakout's 0.966 is an overestimate: a random paddle only
ever reaches the 1-point rows, so counting bricks and scoring coincide, while a
trained agent breaks the 4- and 7-point rows where they diverge.  Pong's 0.226
is an underestimate for the mirror reason: a random agent scored 75 points and
conceded 2751, so a prior that rewards scoring and ignores conceding looks
nearly constant next to a true reward that is almost always -1.  Treat them as
a ranking within a game, not as a cross-game alignment scale.
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


class BreakoutReasonableRamPartial:
    """Count bricks, optionally reward getting the paddle under the ball.

    The true reward pays 1, 4 or 7 per brick depending on how high the row is;
    ``ram[77]`` counts every brick as 1.  That is precisely what makes this
    partial partial -- it knows a brick fell, not what the brick was worth.
    Measured over 60k random steps the env paid rewards from {0, 1, 4} while the
    counter moved by exactly 1 each time.
    """

    component_keys = ("brick_progress", "tracking_progress")

    def __init__(self, *, tracking_weight: float = 0.0):
        self.tracking_weight = float(tracking_weight)

    def reset(self, info: dict | None = None) -> None:
        return None

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        previous, current = _ram(obs), _ram(next_obs)
        # ram[77] restarts at 0 on a new screen and on a lost life, so a large
        # or negative jump is bookkeeping rather than progress.  Only forward
        # steps of a plausible size count, the same guard MsPacman's dots use.
        brick_delta = int(current[77]) - int(previous[77])
        bricks = float(brick_delta if 0 < brick_delta <= 3 else 0)
        old_gap = abs(int(previous[72]) - int(previous[99]))
        gap = abs(int(current[72]) - int(current[99]))
        tracking = self.tracking_weight * float(np.clip((old_gap - gap) / 8.0, -1.0, 1.0))
        return {
            "partial": float(bricks + tracking),
            "components": {"brick_progress": bricks, "tracking_progress": tracking},
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

_add_family("rbo", "ALE/Breakout-v5", BreakoutReasonableRamPartial, {
    "bricks": {},
    "track05": {"tracking_weight": 0.05},
    "track10": {"tracking_weight": 0.10},
    "track25": {"tracking_weight": 0.25},
    "track50": {"tracking_weight": 0.50},
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
