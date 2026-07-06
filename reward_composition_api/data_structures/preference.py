from __future__ import annotations

from dataclasses import dataclass

from reward_composition_api.data_structures.trajectory import Trajectory


@dataclass
class Preference:
    t1: Trajectory
    t2: Trajectory
    rating: float
