# Reward Composition API

Minimal API for comparing true reward, partial reward, vanilla RLHF, naive composition, and delta composition across Gymnasium, MuJoCo, Box2D, and Atari runs.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## One Generic Run

```powershell
.\.venv\Scripts\python.exe experiments\run_generic_experiment.py `
  --env-id Pendulum-v1 `
  --closed-form-file experiments\rewards\pendulum_angle.py `
  --experiment-name pendulum_partial_delta_100k_10k_eval `
  --method delta `
  --initial-timesteps 20000 `
  --policy-timesteps-per-round 20000 `
  --final-policy-timesteps 20000 `
  --rlhf-rounds 3 `
  --n-pairs 120 `
  --collection-timesteps 6000 `
  --fragment-length 20 `
  --reward-hidden-sizes 64,64 `
  --reward-epochs 15 `
  --reward-batch-size 32 `
  --normalize-model-reward `
  --model-reward-target-mean -0.5 `
  --model-reward-target-std 0.5 `
  --eval-episodes 20 `
  --eval-interval 10000 `
  --policy-learning-kwargs "{n_steps:256,batch_size:64,n_epochs:5,gamma:0.95,learning_rate:0.0003}" `
  --policy-log-interval 10000
```

## Many Methods Or Seeds

```powershell
.\.venv\Scripts\python.exe experiments\run_generic_experiment.py `
  --env-id Pendulum-v1 `
  --closed-form-file experiments\rewards\pendulum_angle.py `
  --experiment-name pendulum_compare `
  --methods true partial vanilla naive delta `
  --seeds 0 1 2 `
  --timesteps 200000 `
  --n-pairs 1000
```

Method aliases: `true`, `partial`, `vanilla`, `feedback`, `naive`, `delta`.

## Package CLI

```powershell
.\.venv\Scripts\python.exe -m reward_composition_api list-envs --suite mujoco
.\.venv\Scripts\python.exe -m reward_composition_api list-partials --suite atari
.\.venv\Scripts\python.exe -m reward_composition_api train --suite mujoco --env-id Reacher-v5 --mode delta --timesteps 5000000
.\.venv\Scripts\python.exe -m reward_composition_api sweep --suite atari --env-ids ALE/Breakout-v5 --seeds 0 1 --timesteps 1000000
.\.venv\Scripts\python.exe -m reward_composition_api summarize --suite mujoco --root logs/mujoco_ablations
```

Run outputs are written under `logs/<experiment-name>/...` and include `metadata.json`, monitor CSVs, eval curves, `true_reward_curve.png`, and saved policies.
