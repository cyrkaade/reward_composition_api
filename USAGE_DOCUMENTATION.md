# Reward Composition API Usage Guide

This folder is an independent project copy of the reward-composition API.

Use this folder as the project root:

```powershell
cd reward_composition_api
```

It includes the API package plus the local modules it needs: `reward_composition_api`, `local_gym`, `tests`, `partials`, and `requirements.txt`.

## Install

```powershell
cd reward_composition_api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Custom Partial Rewards

Put custom partial rewards here:

```powershell
C:\Users\PC\Documents\coding\research\reward_composition_api\partials
```

Simplest format:

```python
def partial_reward(obs, action, next_obs, true_reward, terminated, truncated, info):
    partial = float(true_reward)
    return {
        "partial": partial,
        "components": {"my_component": partial},
    }
```

Save it as:

```powershell
partials\my_reward.py
```

Validate it:

```powershell
python -m reward_composition_api validate-partial --suite gym --env-id CartPole-v1 --partial my_reward
```

Run with it:

```powershell
python -m reward_composition_api train --suite gym --env-id CartPole-v1 --mode delta --partial my_reward --timesteps 10000
```

The shorthand `--partial my_reward` means: load `partials\my_reward.py` and use its `partial_reward` function.

## Multiple Partials In One File

Create `partials\my_file.py`:

```python
def reward_a(**kwargs):
    return 1.0


def reward_b(obs, action, next_obs, true_reward, terminated, truncated, info):
    return {"partial": float(true_reward), "components": {"true_reward_copy": float(true_reward)}}


def register(registry):
    registry.register(name="reward_a", suite="gym", factory=lambda env_id: reward_a)
    registry.register(
        name="reward_b",
        suite="gym",
        factory=lambda env_id: reward_b,
        component_keys=("true_reward_copy",),
    )
```

Run one of them:

```powershell
python -m reward_composition_api train --suite gym --env-id CartPole-v1 --mode delta --partial my_file:reward_b
```

## Suites

Use these suites:

- `gym`: generic Gymnasium environments, including classic control and most installed env IDs.
- `box2d`: Box2D environments like `LunarLander-v3`, `BipedalWalker-v3`, and `CarRacing-v3`.
- `mujoco`: MuJoCo environments. Built-in component partials exist for Reacher, HalfCheetah, Hopper, and Walker2d; other MuJoCo envs should use a custom partial.
- `atari`: installed ALE Atari games, using RAM observations in the specialized Atari backend.

List available environments:

```powershell
python -m reward_composition_api list-envs --suite gym
python -m reward_composition_api list-envs --suite box2d
python -m reward_composition_api list-envs --suite mujoco
python -m reward_composition_api list-envs --suite atari
```

## Training Modes

- `true`: train on the environment reward.
- `partial`: train only on your partial reward.
- `feedback`: train only on the learned preference reward model.
- `naive`: train on partial plus learned model reward.
- `delta`: train on partial plus learned residual correction.

## Examples

CartPole with a custom partial:

```powershell
python -m reward_composition_api train --suite gym --env-id CartPole-v1 --mode delta --partial cartpole_alive --timesteps 10000 --n-envs 1
```

LunarLander with a custom partial:

```powershell
python -m reward_composition_api train --suite box2d --env-id LunarLander-v3 --mode delta --partial my_lander_reward --timesteps 100000 --n-envs 4
```

Any installed MuJoCo env with a custom partial:

```powershell
python -m reward_composition_api train --suite mujoco --env-id Ant-v5 --mode delta --partial my_ant_reward --timesteps 100000 --n-envs 4
```

Atari game:

```powershell
python -m reward_composition_api train --suite atari --env-id ALE/Pong-v5 --mode delta --partial-source life_loss --timesteps 1000000 --n-envs 8
```

Generic Gym dry-run sweep:

```powershell
python -m reward_composition_api sweep --suite gym --env-ids CartPole-v1 --seeds 0 1 2 --timesteps 100000 --partial cartpole_alive
```

Add `--execute` to actually run the sweep.

## Outputs

Runs save into:

- `logs/gym_ablations`
- `logs/box2d_ablations`
- `logs/mujoco_ablations`
- `logs/atari_ablations`

Important files:

- `metadata.json`
- `final_model.zip`
- `vecnormalize.pkl` when observation normalization is used
- `true_reward_curve.png`
- `eval/component_evaluations.csv`
- `eval/final_component_evaluation.csv`
