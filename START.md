# Reward Composition API

Minimal API for comparing true reward, partial reward, vanilla RLHF, naive composition, and delta composition across Gymnasium, MuJoCo, Box2D, and Atari runs.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## One Run

```powershell
.\.venv\Scripts\python.exe -m reward_composition_api train `
  --suite mujoco `
  --env-id Reacher-v5 `
  --mode delta `
  --timesteps 5000000 `
  --seed 0
```

## Many Runs

```powershell
.\.venv\Scripts\python.exe -m reward_composition_api sweep `
  --suite mujoco `
  --env-ids Reacher-v5 `
  --seeds 0 1 2 `
  --timesteps 5000000
```

## Package CLI

```powershell
.\.venv\Scripts\python.exe -m reward_composition_api list-envs --suite mujoco
.\.venv\Scripts\python.exe -m reward_composition_api list-partials --suite atari
.\.venv\Scripts\python.exe -m reward_composition_api train --suite mujoco --env-id Reacher-v5 --mode delta --timesteps 5000000
.\.venv\Scripts\python.exe -m reward_composition_api sweep --suite atari --env-ids ALE/Breakout-v5 --seeds 0 1 --timesteps 1000000
.\.venv\Scripts\python.exe -m reward_composition_api summarize --suite mujoco --root logs/mujoco_ablations
```

For k-fold reward ensembles, opt in on RLHF-style modes with `--reward-model-ensemble-size 5`. With the default `--active-query-strategy auto`, active learning uses ensemble variance when the ensemble size is greater than one, and keeps the legacy MC-dropout querying when the size is one. Use `--active-query-strategy ensemble` to require ensemble querying explicitly.

Run outputs are written under `logs/<experiment-name>/...` and include `metadata.json`, monitor CSVs, eval curves, `true_reward_curve.png`, and saved policies.
