"""Generate the true/vanilla/partial-only reasonable-prior calibration grid.

The candidate lists are intentionally ordinary Python sequences with no
five-candidate gate: extend any environment with more logically justified
partials and regenerate.  The initial design has eight candidates per
environment, five paired seeds, and eleven environments:

    11 * (true + vanilla + 8 partials) * 5 = 550 independent runs

CPU (Box2D/MuJoCo) and Atari rows are emitted separately so their Slurm arrays
can run independently on the appropriate hardware.
"""

from pathlib import Path


SEEDS = range(5)
CANDIDATES = {
    "ll": (
        "reasonable_partials:rll_pad_speed",
        "reasonable_partials:rll_tilt25",
        "reasonable_partials:rll_tilt50",
        "reasonable_partials:rll_legs",
        "reasonable_partials:rll_tilt_legs",
        "reasonable_partials:rll_spin",
        "reasonable_partials:rll_fuel",
        "reasonable_partials:rll_tilt_fuel",
    ),
    "bipedal": (
        "reasonable_partials:rbw_cap10",
        "reasonable_partials:rbw_cap20",
        "reasonable_partials:rbw_cap30",
        "reasonable_partials:rbw_cap40",
        "reasonable_partials:rbw_cap50",
        "reasonable_partials:rbw_cap75",
        "reasonable_partials:rbw_cap30_upright",
        "reasonable_partials:rbw_cap50_upright",
    ),
    "ant": (
        "reasonable_partials:rant_cap05",
        "reasonable_partials:rant_cap10",
        "reasonable_partials:rant_cap20",
        "reasonable_partials:rant_cap30",
        "reasonable_partials:rant_cap50",
        "reasonable_partials:rant_cap30_ctrl05",
        "reasonable_partials:rant_cap50_ctrl10",
        "reasonable_partials:rant_cap75_ctrl10",
    ),
    "reacher": (
        "reasonable_partials:rrch_x_ctrl100",
        "reasonable_partials:rrch_x_ctrl050",
        "reasonable_partials:rrch_l1_ctrl100",
        "reasonable_partials:rrch_l1_ctrl050",
        "reasonable_partials:rrch_dist_ctrl025",
        "reasonable_partials:rrch_dist_ctrl050",
        "reasonable_partials:rrch_cap05_ctrl050",
        "reasonable_partials:rrch_hit05_ctrl025",
    ),
    "pusher": (
        "reasonable_partials:rpsh_task_ctrl15",
        "reasonable_partials:rpsh_task_ctrl20",
        "reasonable_partials:rpsh_cap15_ctrl10",
        "reasonable_partials:rpsh_cap20_ctrl10",
        "reasonable_partials:rpsh_progress_near",
        "reasonable_partials:rpsh_progress_smooth",
        "reasonable_partials:rpsh_progress_success",
        "reasonable_partials:rpsh_progress_ctrl05",
    ),
    "hopper": (
        "reasonable_partials:rhop_cap00",
        "reasonable_partials:rhop_cap10",
        "reasonable_partials:rhop_cap15",
        "reasonable_partials:rhop_cap20",
        "reasonable_partials:rhop_cap30",
        "reasonable_partials:rhop_cap40",
        "reasonable_partials:rhop_cap60",
        "reasonable_partials:rhop_cap80",
    ),
    "swimmer": (
        "reasonable_partials:rswm_cap12",
        "reasonable_partials:rswm_cap16",
        "reasonable_partials:rswm_cap20",
        "reasonable_partials:rswm_cap24",
        "reasonable_partials:rswm_cap28",
        "reasonable_partials:rswm_cap32",
        "reasonable_partials:rswm_cap40",
        "reasonable_partials:rswm_cap28_ctrl001",
    ),
    "walker": (
        "reasonable_partials:rwalk_cap050",
        "reasonable_partials:rwalk_cap100",
        "reasonable_partials:rwalk_cap150",
        "reasonable_partials:rwalk_cap175",
        "reasonable_partials:rwalk_cap200",
        "reasonable_partials:rwalk_cap225",
        "reasonable_partials:rwalk_cap250",
        "reasonable_partials:rwalk_cap300",
    ),
    "mspacman": (
        "reasonable_atari_partials:rmsp_pellets",
        "reasonable_atari_partials:rmsp_visit02",
        "reasonable_atari_partials:rmsp_visit05",
        "reasonable_atari_partials:rmsp_visit10",
        "reasonable_atari_partials:rmsp_visit25",
        "reasonable_atari_partials:rmsp_safe05",
        "reasonable_atari_partials:rmsp_safe10",
        "reasonable_atari_partials:rmsp_visit10_safe05",
    ),
    "qbert": (
        "reasonable_atari_partials:rqb_tiles",
        "reasonable_atari_partials:rqb_visit01",
        "reasonable_atari_partials:rqb_visit02",
        "reasonable_atari_partials:rqb_visit05",
        "reasonable_atari_partials:rqb_visit10",
        "reasonable_atari_partials:rqb_visit20",
        "reasonable_atari_partials:rqb_visit40",
        "reasonable_atari_partials:rqb_visit75",
    ),
    "pong": (
        "reasonable_atari_partials:rpong_score",
        "reasonable_atari_partials:rpong_track05",
        "reasonable_atari_partials:rpong_track10",
        "reasonable_atari_partials:rpong_track25",
        "reasonable_atari_partials:rpong_track50",
        "reasonable_atari_partials:rpong_rally002",
        "reasonable_atari_partials:rpong_rally005",
        "reasonable_atari_partials:rpong_track10_rally005",
    ),
}

ATARI_CELLS = {"mspacman", "qbert", "pong"}
CELLS = tuple(CANDIDATES)


def variant(reference: str) -> str:
    return reference.rsplit(":", 1)[-1]


def rows(cells) -> list[str]:
    output = []
    for cell in cells:
        output.extend(f"{cell} true {seed} -" for seed in SEEDS)
        output.extend(f"{cell} vanilla {seed} -" for seed in SEEDS)
        for reference in CANDIDATES[cell]:
            output.extend(f"{cell} {variant(reference)} {seed} {reference}" for seed in SEEDS)
    return output


all_lines = rows(CELLS)
cpu_lines = rows(cell for cell in CELLS if cell not in ATARI_CELLS)
atari_lines = rows(cell for cell in CELLS if cell in ATARI_CELLS)

outputs = {
    "params_reasonable.txt": all_lines,
    "params_reasonable_cpu.txt": cpu_lines,
    "params_reasonable_atari.txt": atari_lines,
}
for filename, lines in outputs.items():
    path = Path(__file__).with_name(filename)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} with {len(lines)} lines")

if len(all_lines) != 550 or len(cpu_lines) != 400 or len(atari_lines) != 150:
    raise SystemExit(
        "candidate count changed: update the Slurm --array bounds and EXPECTED_ROWS "
        f"(all={len(all_lines)}, cpu={len(cpu_lines)}, atari={len(atari_lines)})"
    )
