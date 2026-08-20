"""Generate the independent true-vs-partial calibration screen.

Each of eleven environments has one true-reward reference and exactly five
hand-written partials, each run for seeds 0..4 and 2M policy timesteps. There
is no reward model, query budget, or RLHF arm in this first-stage experiment.
"""

from pathlib import Path


SEEDS = range(5)
CANDIDATES = {
    "ll": (
        "sel_lunarlander:sll_upright",
        "sel_lunarlander:sll_pad",
        "sel_lunarlander:sll_touchdown",
        "sel_lunarlander:sll_pad_speed",
        "sel_lunarlander:sll_pad_fuel",
    ),
    "reacher": (
        "sel_reacher:srch_x",
        "sel_reacher:srch_hit",
        "sel_reacher:srch_dist_cap",
        "sel_reacher:srch_quad",
        "sel_reacher:srch_dist",
    ),
    "pusher": (
        "non_timid_screen:npsh_progress",
        "non_timid_screen:npsh_progress_smooth",
        "non_timid_screen:npsh_success_smooth",
        "non_timid_screen:npsh_progress_calm",
        "non_timid_screen:npsh_progress_success",
    ),
    "swimmer": (
        "non_timid_screen:nswm_target12",
        "sel_swimmer:sswm_cap10",
        "non_timid_screen:nswm_target18",
        "sel_swimmer:sswm_cap18",
        "sel_swimmer:sswm_straight",
    ),
    "hopper": (
        "hopper_gameable_bounce",
        "mujoco_capped_low:hopper_capped_low",
        "hopper_levels:hopper_p25",
        "hopper_levels:hopper_p50",
        "hopper_levels:hopper_p75",
    ),
    "bipedal": (
        "sel_bipedal:sbw_upright",
        "sel_bipedal:sbw_target",
        "sel_bipedal:sbw_speed_cap",
        "non_timid_screen:nbw_speed_smooth",
        "sel_bipedal:sbw_speed",
    ),
    "walker": (
        "sel_walker2d:swk_stand",
        "sel_walker2d:swk_target",
        "non_timid_screen:nwalk_speed_cap10",
        "non_timid_screen:nwalk_posture_cap15",
        "non_timid_screen:nwalk_speed_cap20",
    ),
    "ant": (
        "sel_ant:sant_radial",
        "sel_ant:sant_cap03",
        "non_timid_screen:nant_target_smooth",
        "non_timid_screen:nant_posture_smooth",
        "non_timid_screen:nant_forward_smooth",
    ),
    "mspacman": (
        "atari_ram_screen:namsp_pellets",
        "atari_ram_screen:namsp_pellets_visit",
        "atari_ram_screen:namsp_visit_motion",
        "atari_ram_screen:namsp_pellets_safe",
        "atari_ram_screen:namsp_pellets_motion",
    ),
    "qbert": (
        "atari_ram_screen:naqbert_tiles",
        "atari_ram_screen:naqbert_tiles_visit",
        "atari_ram_screen:naqbert_visit_motion",
        "atari_ram_screen:naqbert_tiles_motion",
        "atari_ram_screen:naqbert_tiles_visit_motion",
    ),
    "pong": (
        "atari_ram_screen:napong_rally",
        "atari_ram_screen:napong_motion",
        "atari_ram_screen:napong_track",
        "atari_ram_screen:napong_score",
        "atari_ram_screen:napong_score_track",
    ),
}

ATARI_CELLS = {"mspacman", "qbert", "pong"}
CELLS = tuple(CANDIDATES)


def variant(reference: str) -> str:
    return reference.rsplit(":", 1)[-1]


def cell_lines(cell: str) -> list[str]:
    references = CANDIDATES[cell]
    if len(references) != 5:
        raise SystemExit(f"{cell} must have exactly five candidates; got {len(references)}")
    lines = [f"{cell} true {seed} -" for seed in SEEDS]
    for reference in references:
        lines.extend(f"{cell} {variant(reference)} {seed} {reference}" for seed in SEEDS)
    return lines


all_lines = [line for cell in CELLS for line in cell_lines(cell)]
cpu_lines = [line for cell in CELLS if cell not in ATARI_CELLS for line in cell_lines(cell)]
atari_lines = [line for cell in CELLS if cell in ATARI_CELLS for line in cell_lines(cell)]

outputs = {
    "params_ntscreen.txt": (all_lines, 330),
    "params_ntscreen_cpu.txt": (cpu_lines, 240),
    "params_ntscreen_atari.txt": (atari_lines, 90),
}
for filename, (lines, expected) in outputs.items():
    path = Path(__file__).with_name(filename)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if len(lines) != expected:
        raise SystemExit(f"internal error: expected {expected} rows for {filename}, generated {len(lines)}")
    print(f"wrote {path} with {len(lines)} lines")
