"""Parameter file for the fast pixel-Atari pilot.

One seed, 1M policy timesteps, three arms per game: the true reward, zero-label
RLHF (vanilla), and five hand-written RAM partials trained alone.  The five are
the endpoints and midpoints of each game's one-dimensional shaping ladder in
partials/reasonable_atari_partials.py, so the ladder is still readable with
three of the eight rungs dropped.
"""

from pathlib import Path

SEEDS = [0]

CANDIDATES = {
    "mspacman": ["rmsp_pellets", "rmsp_visit02", "rmsp_visit10", "rmsp_visit25", "rmsp_visit10_safe05"],
    "qbert": ["rqb_tiles", "rqb_visit02", "rqb_visit10", "rqb_visit40", "rqb_visit75"],
    "pong": ["rpong_score", "rpong_track10", "rpong_track50", "rpong_rally005", "rpong_track10_rally005"],
}


def main() -> None:
    rows = []
    for cell, partials in CANDIDATES.items():
        for seed in SEEDS:
            rows.append(f"{cell} true {seed} -")
            rows.append(f"{cell} vanilla {seed} -")
            for name in partials:
                rows.append(f"{cell} {name} {seed} reasonable_atari_partials:{name}")
    out = Path(__file__).with_name("params_fastatari.txt")
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
