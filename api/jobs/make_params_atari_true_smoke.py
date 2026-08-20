"""Generate the 15-run Atari pixel-policy true-reward smoke grid."""

from pathlib import Path


CELLS = ("mspacman", "qbert", "pong")
SEEDS = range(5)

lines = [f"{cell} {seed}" for cell in CELLS for seed in SEEDS]
if len(lines) != 15:
    raise SystemExit(f"internal error: expected 15 rows, generated {len(lines)}")

path = Path(__file__).with_name("params_atari_true_smoke.txt")
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {path} with {len(lines)} lines")
