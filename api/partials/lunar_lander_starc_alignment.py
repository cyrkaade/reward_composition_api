"""LunarLander-v3 priors on a STARC alignment ladder: 0.2 / 0.4 / 0.6 / 0.8 / 1.0.

Measured with ``rcomp starc`` at its defaults -- minimal-L2 canonicalisation,
512 potential features, gamma 0.99, the fixed 90%-heuristic reference policy,
100k steps.  Calibrated by bisection on seeds 0-4, then checked on seeds 5-9,
which were used to fit nothing:

    target   terminal w   seeds 0-4 (fit)    seeds 5-9 (held out)
      0.2      -0.5558    0.200 +/- 0.007    0.207 +/- 0.017
      0.4      -0.1977    0.400 +/- 0.008    0.414 +/- 0.018
      0.6      +0.0424    0.600 +/- 0.005    0.614 +/- 0.011
      0.8      +0.3385    0.800 +/- 0.003    0.807 +/- 0.004
      1.0      +1.0000    1.000              1.000

Systematic spread from the one free hyperparameter (the size of the potential
basis) is about +/-0.03 at the bottom of the ladder and +/-0.01 at the top; the
rungs are 0.2 apart, so they stay cleanly ordered and non-overlapping under
every setting tried (128 / 256 / 512 / 1024 features).  Quote the targets as
labels; quote the measurements with their spread.

WHY THE TERMINAL WEIGHT IS THE ONLY KNOB.  LunarLander's shaping term is
literally ``shaping(s') - shaping(s)`` -- textbook potential shaping -- and
canonicalisation exists precisely to delete potential shaping.  So the four
shaping weights are nearly invisible to STARC and the fuel term is far too
small to matter.  Measured on seeds 0-2 with everything else held at 1.0:

    zero all four shaping weights   0.793      drop the fuel term      0.982
    scale the shaping x5            0.717      scale the fuel x10      0.846
    negate the shaping              0.549      negate the fuel         0.964
    crash penalty only              0.818      landing bonus only      0.827
    scale the terminal x0.1         0.643      remove the terminal     0.562

Nothing there gets near 0.2.  The terminal +/-100 is the only term with the
range, which is why it is the single knob and why the ladder is exactly
monotone in it.

WHAT A LOW RUNG MEANS.  Alignment ``a`` means ``cos(theta) = 1 - 2*(1-a)**2``
between the two canonical rewards, so anything below **0.293** is *negatively*
correlated with the true reward.  A 0.2 rung is therefore forced to be
adversarial: there is no merely-uninformative reward that scores 0.2.  Here it
shows up as an inverted terminal, with shaping and fuel still exactly right:

    rung       terminal w    landing pays    crashing pays
    lls_a20      -0.5558        -55.6           +55.6
    lls_a40      -0.1977        -19.8           +19.8
    lls_a60      +0.0424         +4.2            -4.2
    lls_a80      +0.3385        +33.9           -33.9
    lls_a100     +1.0000       +100.0          -100.0

So the ladder runs "pays you to crash" -> "barely cares how it ends" -> "the
true reward".  Read a low rung as a *wrong* prior, not an *ignorant* one; the
PCC ladder in ``lunar_lander_alignment.py`` is the one that varies how much of
the reward structure the prior knows.

``lls_a100`` is the identity -- weight 1.0 makes the partial bit-identical to
the environment reward -- so its 1.0 is a tautology, not a measurement.
"""

from __future__ import annotations

from partials.lunar_lander_alignment import LunarLanderWeightedPartial


# name -> (target alignment, calibrated terminal weight, held-out mean)
STARC_LADDER = {
    "lls_a20": (0.2, -0.5558, 0.207),
    "lls_a40": (0.4, -0.1977, 0.414),
    "lls_a60": (0.6, 0.0424, 0.614),
    "lls_a80": (0.8, 0.3385, 0.807),
    "lls_a100": (1.0, 1.0000, 1.000),
}


def register(registry) -> None:
    for name, (target, terminal_weight, measured) in STARC_LADDER.items():
        registry.register(
            name=name,
            suite="box2d",
            factory=(
                lambda env_id, weight=terminal_weight: LunarLanderWeightedPartial(
                    game_over=weight,
                    landed=weight,
                    continuous="Continuous" in env_id,
                )
            ),
            description=(
                f"LunarLander STARC rung {target:.1f} "
                f"(held-out mean {measured:.3f}, terminal weight {terminal_weight:g}; "
                "shaping and fuel exact)"
            ),
            env_ids=("LunarLander-v3", "LunarLanderContinuous-v3"),
            component_keys=LunarLanderWeightedPartial.component_keys,
        )
