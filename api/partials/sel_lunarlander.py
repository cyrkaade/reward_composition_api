"""Candidate LunarLander-v3 priors for the environment/partial selection grid.

Five hand-written rewards, each a plausible thing a designer would write after
watching the lander for a minute, and each blind to something the true reward
cares about. None is a rescaling of any other: scaling every weight in a partial
by c is exactly `--partial-alpha c` (the partial is linear in its weights), so a
"weaker" prior built by rescaling would test reward SCALE, not how much of the
task the prior knows. These delete or replace whole terms instead.

Ground truth, read from gymnasium 1.2.3 lunar_lander.py:

    shaping = -100*dist -100*speed -100*|angle| + 10*leg1 + 10*leg2
    reward  = shaping(s') - shaping(s) - 0.30*m_power - 0.03*s_power
    on termination the step reward is REPLACED by -100 (crash / out of bounds)
    or +100 (lander asleep, i.e. came to rest)

Discrete action ids: 0 idle, 1 left engine, 2 main engine, 3 right engine, so
m_power is 1.0 iff action==2 and s_power is 1.0 iff action in {1,3}. That makes
the fuel term exactly reproducible from the action alone.

Observation layout: [x, y, vx, vy, angle, angular_velocity, leg1, leg2].
"""

from __future__ import annotations

import numpy as np

MAIN_ENGINE_COST = 0.30
SIDE_ENGINE_COST = 0.03


def _fuel_cost(action) -> float:
    """The true reward's fuel term, reconstructed from the discrete action."""
    a = int(np.asarray(action).reshape(-1)[0])
    if a == 2:
        return MAIN_ENGINE_COST
    if a in (1, 3):
        return SIDE_ENGINE_COST
    return 0.0


class ShapingPartial:
    """Potential-based shaping F = Phi(s') - Phi(s) over a chosen subset of the
    true reward's five shaping terms, with optional fuel accounting and an
    optional hand-written terminal judgement."""

    component_keys = ("shaping_delta", "fuel", "terminal_judgement")

    def __init__(
        self,
        w_dist: float = 0.0,
        w_speed: float = 0.0,
        w_tilt: float = 0.0,
        w_spin: float = 0.0,
        w_leg: float = 0.0,
        charge_fuel: bool = False,
        terminal_bonus: float = 0.0,
        terminal_rule: str = "none",
    ):
        self.w_dist = float(w_dist)
        self.w_speed = float(w_speed)
        self.w_tilt = float(w_tilt)
        self.w_spin = float(w_spin)
        self.w_leg = float(w_leg)
        self.charge_fuel = bool(charge_fuel)
        self.terminal_bonus = float(terminal_bonus)
        self.terminal_rule = str(terminal_rule)
        self.prev = None

    def reset(self, info: dict | None = None) -> None:
        self.prev = None

    def _potential(self, state) -> float:
        s = np.asarray(state, dtype=np.float64)
        x, y, vx, vy, ang, spin = s[0], s[1], s[2], s[3], s[4], s[5]
        leg1, leg2 = s[6], s[7]
        return float(
            -self.w_dist * np.sqrt(x * x + y * y)
            - self.w_speed * np.sqrt(vx * vx + vy * vy)
            - self.w_tilt * abs(ang)
            - self.w_spin * abs(spin)
            + self.w_leg * (leg1 + leg2)
        )

    def _terminal(self, next_obs) -> float:
        if self.terminal_rule == "none" or self.terminal_bonus == 0.0:
            return 0.0
        s = np.asarray(next_obs, dtype=np.float64)
        if self.terminal_rule == "legs":
            # Deliberately lenient: "both legs touched down" counts as a landing
            # even at a speed the real env scores -100 for. A designer who checks
            # contact and nothing else writes this.
            landed = s[6] > 0.5 and s[7] > 0.5
        else:
            raise ValueError(f"unknown terminal rule: {self.terminal_rule}")
        return self.terminal_bonus if landed else -self.terminal_bonus

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        if self.prev is None and obs is not None:
            self.prev = self._potential(obs)
        current = self._potential(next_obs)
        shaping_delta = 0.0 if self.prev is None else current - self.prev
        self.prev = current

        fuel = -_fuel_cost(action) if self.charge_fuel else 0.0
        terminal = self._terminal(next_obs) if terminated else 0.0

        return {
            "partial": float(shaping_delta + fuel + terminal),
            "components": {
                "shaping_delta": float(shaping_delta),
                "fuel": float(fuel),
                "terminal_judgement": float(terminal),
            },
        }


_PARTIALS = {
    # 1. AXIS: knows only where the pad is. No idea that arriving there slowly,
    #    upright, on both legs, or cheaply is worth anything.
    "sll_pad": dict(w_dist=100.0),
    # 2. AXIS: pad plus "arrive slowly". Still blind to attitude, legs, fuel and
    #    to the crash/land distinction.
    "sll_pad_speed": dict(w_dist=100.0, w_speed=100.0),
    # 3. OMISSION, other way round: knows the pad, the speed AND the fuel bill,
    #    which is the term that makes hovering unattractive, but has no notion of
    #    orientation, leg contact, or what happens at touchdown.
    "sll_pad_fuel": dict(w_dist=100.0, w_speed=100.0, charge_fuel=True),
    # 4. GAMEABLE: full attitude-aware shaping plus a terminal judgement that
    #    checks leg contact ONLY. Slamming into the pad legs-first scores +60 here
    #    and -100 in the true reward, so the prior actively endorses a crash.
    "sll_touchdown": dict(
        w_dist=60.0, w_speed=50.0, w_tilt=100.0, w_leg=10.0,
        terminal_bonus=60.0, terminal_rule="legs",
    ),
    # 5. PROXY SUBSTITUTION: a designer who believes the whole problem is keeping
    #    the craft level and getting the legs down. Attitude and spin dominate,
    #    horizontal precision is an afterthought, descent speed is ignored.
    "sll_upright": dict(w_dist=20.0, w_tilt=100.0, w_spin=30.0, w_leg=10.0),
}


def _factory(kwargs):
    return lambda env_id: ShapingPartial(**kwargs)


def register(registry) -> None:
    for name, kwargs in _PARTIALS.items():
        registry.register(
            name=name,
            suite="box2d",
            factory=_factory(kwargs),
            description=f"LunarLander selection-grid prior '{name}'",
            env_ids=("LunarLander-v3",),
            component_keys=ShapingPartial.component_keys,
        )
