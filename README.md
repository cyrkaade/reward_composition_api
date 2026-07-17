# Reward Composition API

Research CLI for comparing true, partial, preference-learned, naive-composed, and delta-composed rewards in Gymnasium, MuJoCo, Box2D, and Atari environments.

## Reward modes

| Mode | Policy reward | Meaning |
|---|---:|---|
| `true` | `R` | Environment reward baseline |
| `partial` | `P` | Hand-written partial reward baseline |
| `feedback` | `r_hat` | Vanilla preference learning |
| `naive` | `P + r_hat` | Add a full learned reward to the partial reward |
| `delta` | `P + r_delta` | Learn the correction while accounting for `P` in the preference loss |

`partial`, `naive`, and `delta` require a manually written partial reward.

## Install

Python 3.10-3.13 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Run an experiment

```powershell
.\.venv\Scripts\python.exe -m reward_composition_api train `
  --suite mujoco `
  --env-id Reacher-v5 `
  --mode delta `
  --partial partials/reacher_distance_partial.py `
  --timesteps 800000 `
  --query-budget 150 `
  --seed 7 `
  --final-policy last
```

Use `--final-policy last` for research comparisons that must not use true reward to select a checkpoint. The default, `best`, is useful for debugging but chooses the checkpoint with the best true-reward evaluation.

## Useful commands

```powershell
python -m reward_composition_api list-envs --suite mujoco
python -m reward_composition_api list-partials --suite mujoco
python -m reward_composition_api validate-partial --suite mujoco --env-id Reacher-v5 --partial partials/reacher_distance_partial.py
python -m reward_composition_api partiality --suite mujoco --env-id Reacher-v5 --partial partials/reacher_distance_partial.py
python -m reward_composition_api summarize --suite mujoco --root logs/mujoco_ablations
```

See [USAGE_DOCUMENTATION.md](USAGE_DOCUMENTATION.md) for custom partials, sweeps, partiality metrics, and output files. Run `python -m reward_composition_api --help` or append `--help` to a subcommand for every option.
