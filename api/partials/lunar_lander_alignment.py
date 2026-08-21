"""LunarLander-v3 priors on an ALIGNMENT LADDER.

Every partial here is the true reward with each of its eight terms scaled by an
independent weight, so the family spans a continuum from "knows almost nothing"
to "knows almost everything" without ever inventing a term the environment does
not have.  With all eight weights at 1.0 the partial reproduces the environment's
reward exactly (asserted in tests/test_lunar_lander_alignment.py), which is what
makes the ladder interpretable: a weight of 0.5 removes half of one real term
rather than adding an unrelated one.

Ground truth, read from gymnasium 1.2.3 ``lunar_lander.py`` (verified 2026-08-21):

    shaping = -100*sqrt(x^2+y^2) -100*sqrt(vx^2+vy^2) -100*|angle|
              + 10*leg1 + 10*leg2
    reward  = shaping(s') - shaping(s) - 0.30*m_power - 0.03*s_power
    on termination the step reward is REPLACED (not added to) by
        -100  if game_over or |x| >= 1      (crash / out of bounds)
        +100  if the lander body is asleep  (came to rest)

The replacement, not addition, is why ``game over`` and ``landed`` are separate
weights from the shaping ones: at w=1 they must *override* the step, and they do.

Classifying that terminal step from the observation alone: Box2D zeroes a body's
linear and angular velocity when it puts it to sleep, so the +100 branch is
exactly ``vx == vy == spin == 0.0`` at a terminated step.  Measured over 581
terminations spanning heuristic, noisy-heuristic, random and idle-heavy policies:
0 mismatches.  No true-reward value is read; the partial stays honest.

Observation layout: [x, y, vx, vy, angle, angular_velocity, leg1, leg2].
Discrete action ids: 0 idle, 1 left, 2 main, 3 right, so m_power is 1.0 iff
action==2 and s_power is 1.0 iff action in {1,3}.

The five registered ladder rungs are the weight rows from the alignment grid
(PCC of the partial against the true reward); the ``align`` value quoted per
entry is the score that grid reported, and the name carries it rounded.
"""

from __future__ import annotations

import numpy as np

MAIN_ENGINE_COST = 0.30
SIDE_ENGINE_COST = 0.03
TERMINAL_MAGNITUDE = 100.0


class LunarLanderWeightedPartial:
    """The true LunarLander reward with per-term weights.

    All weights 1.0 == the environment's reward, term for term.
    """

    component_keys = ("shaping_delta", "fuel", "terminal")

    def __init__(
        self,
        *,
        distance: float = 1.0,
        speed: float = 1.0,
        tilt: float = 1.0,
        leg: float = 1.0,
        side_engine: float = 1.0,
        main_engine: float = 1.0,
        game_over: float = 1.0,
        landed: float = 1.0,
        continuous: bool = False,
    ):
        self.w_distance = float(distance)
        self.w_speed = float(speed)
        self.w_tilt = float(tilt)
        self.w_leg = float(leg)
        self.w_side = float(side_engine)
        self.w_main = float(main_engine)
        self.w_game_over = float(game_over)
        self.w_landed = float(landed)
        self.continuous = bool(continuous)
        self.prev_shaping: float | None = None

    def reset(self, info: dict | None = None) -> None:
        self.prev_shaping = None

    def _shaping(self, state) -> float:
        s = np.asarray(state, dtype=np.float64)
        return float(
            -100.0 * self.w_distance * np.sqrt(s[0] * s[0] + s[1] * s[1])
            - 100.0 * self.w_speed * np.sqrt(s[2] * s[2] + s[3] * s[3])
            - 100.0 * self.w_tilt * abs(s[4])
            + 10.0 * self.w_leg * s[6]
            + 10.0 * self.w_leg * s[7]
        )

    def _powers(self, action):
        """(m_power, s_power), reconstructed from the action exactly as the env does."""
        a = np.asarray(action).reshape(-1)
        if self.continuous:
            m = (float(np.clip(a[0], 0.0, 1.0)) + 1.0) * 0.5 if a[0] > 0.0 else 0.0
            s = float(np.clip(abs(a[1]), 0.5, 1.0)) if abs(a[1]) > 0.5 else 0.0
            return m, s
        code = int(a[0])
        return (1.0 if code == 2 else 0.0), (1.0 if code in (1, 3) else 0.0)

    @staticmethod
    def _came_to_rest(next_obs) -> bool:
        """True iff Box2D put the lander to sleep, i.e. the env paid +100."""
        s = np.asarray(next_obs, dtype=np.float64)
        return s[2] == 0.0 and s[3] == 0.0 and s[5] == 0.0

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        # Track the potential on every step, terminal included, so the partial's
        # internal state stays a faithful mirror of the env's `prev_shaping`.
        if self.prev_shaping is None and obs is not None:
            self.prev_shaping = self._shaping(obs)
        shaping = self._shaping(next_obs)
        shaping_delta = 0.0 if self.prev_shaping is None else shaping - self.prev_shaping
        self.prev_shaping = shaping

        m_power, s_power = self._powers(action)
        fuel = -(MAIN_ENGINE_COST * self.w_main * m_power
                 + SIDE_ENGINE_COST * self.w_side * s_power)

        if terminated:
            # The env REPLACES the step reward here; so do we.
            if self._came_to_rest(next_obs):
                terminal = TERMINAL_MAGNITUDE * self.w_landed
            else:
                terminal = -TERMINAL_MAGNITUDE * self.w_game_over
            total = terminal
            shaping_delta = 0.0
            fuel = 0.0
        else:
            terminal = 0.0
            total = shaping_delta + fuel

        return {
            "partial": float(total),
            "components": {
                "shaping_delta": float(shaping_delta),
                "fuel": float(fuel),
                "terminal": float(terminal),
            },
        }


# The alignment ladder.  Weight rows are taken verbatim from the alignment grid;
# `align` is the PCC-against-true that grid reported for the row.
_LADDER = {
    "lla_a00": dict(
        align=0.019,
        weights=dict(distance=0.0, speed=0.0, tilt=0.0, leg=1.0,
                     side_engine=0.5, main_engine=0.5, game_over=1.0, landed=1.0),
    ),
    "lla_a30": dict(
        align=0.308,
        weights=dict(distance=0.0, speed=0.0, tilt=0.5, leg=1.0,
                     side_engine=1.0, main_engine=1.0, game_over=1.0, landed=1.0),
    ),
    "lla_a50": dict(
        align=0.528,
        weights=dict(distance=0.5, speed=0.0, tilt=0.0, leg=1.0,
                     side_engine=0.5, main_engine=0.5, game_over=1.0, landed=1.0),
    ),
    "lla_a75": dict(
        align=0.748,
        weights=dict(distance=1.0, speed=0.0, tilt=1.0, leg=1.0,
                     side_engine=0.5, main_engine=0.5, game_over=1.0, landed=1.0),
    ),
    "lla_a95": dict(
        align=0.953,
        weights=dict(distance=0.5, speed=0.5, tilt=1.0, leg=1.0,
                     side_engine=1.0, main_engine=1.0, game_over=1.0, landed=1.0),
    ),
    # Not a rung: the identity.  Registered so the "all ones == true reward"
    # property is exercisable from the CLI as well as from the test suite.
    "lla_full": dict(align=1.0, weights=dict()),
}


def _describe(name, entry) -> str:
    w = entry["weights"]
    if not w:
        return "LunarLander true reward reconstructed exactly (all term weights 1.0)."
    shown = ", ".join(f"{k}={v}" for k, v in w.items())
    return f"LunarLander alignment rung (PCC~{entry['align']:.3f}): {shown}"


def register(registry) -> None:
    for name, entry in _LADDER.items():
        weights = dict(entry["weights"])
        registry.register(
            name=name,
            suite="box2d",
            factory=(lambda env_id, _w=weights: LunarLanderWeightedPartial(
                **_w, continuous="Continuous" in env_id)),
            description=_describe(name, entry),
            env_ids=("LunarLander-v3", "LunarLanderContinuous-v3"),
            component_keys=LunarLanderWeightedPartial.component_keys,
        )
