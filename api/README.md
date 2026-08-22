# rcomp

Compare RLHF reward-composition strategies by training PPO
(stable-baselines3) agents under five reward modes, across four environment
suites (`gym`, `box2d`, `mujoco`, `atari`).

## Install

```bash
cd api
pip install -e .
```

This installs the `rcomp` console script (equivalent to `python -m rcomp`).
Everything below can be run from inside `api/`; run outputs land in
`api/logs/<suite>_ablations/<run_name>/`.

## The five reward modes

| Mode | Training reward | Reward model |
|------|-----------------|--------------|
| `true` | The environment's own reward | — |
| `partial` | A human-written partial reward only | — |
| `feedback` | A learned preference reward model only | Bradley–Terry pairwise loss |
| `naive` | partial + learned model | Bradley–Terry pairwise loss |
| `delta` | partial + learned residual correction | Delta loss: the model learns only what the partial misses |

The preference modes (`feedback`, `naive`, `delta`) run an RLHF loop: each
round collects trajectories from the current policy on the live training
env, fragments them, picks query pairs (randomly, or by active learning via
MC-dropout or ensemble preference variance), rates the pairs synthetically
from summed true reward, trains the reward model (single or k-fold
ensemble), then trains PPO on that round's share of the timesteps. The
reward model can optionally be pretrained on partial/residual/true targets
(`--pretrain-reward-model`).

Modes `partial`, `naive`, and `delta` require `--partial`.

## Writing partial rewards

Partial rewards are always written manually by humans as Python files in the
`partials/` folder (the one next to your working directory, or the packaged
`api/partials/`). Reference them on the CLI as `--partial my_file` or
`--partial my_file:name`.

The simplest style is a single function:

```python
# partials/my_partial.py
def partial_reward(obs, action, next_obs, true_reward, terminated, truncated, info):
    alive = 1.0 if not terminated else 0.0
    return {"partial": alive, "components": {"alive_bonus": alive}}
```

The function is called once per step with keyword arguments and returns
either a plain float or `{"partial": float, "components": {name: float}}`.
Named components are tracked in the component-evaluation CSVs.

Stateful partials use a class (instantiated once per run) with optional
`reset(info)`:

```python
# partials/my_stateful_partial.py
class PartialReward:
    def reset(self, info):
        self.prev_lives = int(info.get("lives", 0))

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):
        lives = int(info.get("lives", 0))
        lost = max(self.prev_lives - lives, 0)
        self.prev_lives = lives
        return {"partial": true_reward - lost, "components": {"lost_lives": float(lost)}}
```

Several named partials can live in one file, referenced as
`--partial my_partials:half_true`:

```python
# partials/my_partials.py
def half_true(obs, action, next_obs, true_reward, terminated, truncated, info):
    return 0.5 * true_reward

def register(registry):
    registry.register(
        name="half_true",
        suite="gym",
        factory=lambda env_id: half_true,
        description="Half of the true reward",
        component_keys=(),
    )
```

A plain `PARTIALS = {"half_true": lambda env_id: half_true}` dict, or a
module-level `partial` callable/object, also works.

## Commands

One example per command:

```bash
# Train one experiment
rcomp train --suite gym --env-id CartPole-v1 --mode delta --partial example_cartpole \
    --timesteps 200000 --rlhf-rounds 5 --query-budget 200

# Plan an ablation sweep (dry run; add --execute to launch)
rcomp sweep --suite mujoco --env-ids Reacher-v5 --seeds 1 2 3 --partial my_partial

# Aggregate finished runs into summary.csv / aggregate.csv
rcomp summarize --suite mujoco

# List supported environments
rcomp list-envs --suite box2d

# List partial rewards found in the partials folders
rcomp list-partials

# Smoke-check a partial file
rcomp validate-partial --suite gym --env-id CartPole-v1 --partial example_cartpole

# Estimate how much a partial matches true reward (random-policy fragments)
rcomp partiality --suite gym --env-id CartPole-v1 --partial example_cartpole --timesteps 20000

# Estimate approximate STARC alignment for a LunarLander partial
rcomp starc --partial lunar_lander_alignment:lla_a75 --timesteps 100000

# Plot final reward by partiality x query budget across finished runs
rcomp plot-partiality --runs-root logs --partiality-root logs/partiality
```

The STARC command is an isolated diagnostic: it collects fresh LunarLander
rollouts and does not change experiment training or reward composition. It
reports a distance from 0 (equivalent) to 2 (opposite), plus a display-friendly
alignment from 1 to 0. The result is an approximation for *ranking* partials
within LunarLander, not an exact percentage or a replacement for true-reward
evaluation — the STARC paper reports the same caveat for its own sampled
estimates. The 100,000-step default is intentional: shorter runs make the
estimate noticeably noisier, especially for low-alignment partials.

`--canonicalisation minimal` (the default) solves directly for the potential
that best cancels the reward, which removes representable potential shaping
exactly. `--canonicalisation val` is the older path that regresses Monte-Carlo
returns; it removes shaping far less well (a policy-equivalent shaped reward
scored 0.72 instead of 1.0) and its numbers swing with `--value-features`, so
prefer `minimal` unless reproducing an earlier measurement.

The calibrated LunarLander STARC grid partials are
`lunar_lander_starc_alignment:lls_a20`, `lls_a40`, `lls_a60`, `lls_a80`, and
`lls_a100`, targeting alignment 0.2 / 0.4 / 0.6 / 0.8 / 1.0. Calibrated on
seeds 0-4 and measured on held-out seeds 5-9 at 0.207 / 0.414 / 0.614 / 0.807 /
1.000. Only the terminal weight varies between rungs, because LunarLander's
shaping term *is* potential shaping and canonicalisation deletes it; see the
module docstring for what that means for the low rungs.

`rcomp train --help` lists every knob; the flags are generated from
`rcomp.config.ExperimentConfig`, so the dataclass is the single source of
truth.

## Python API

```python
from rcomp import ExperimentConfig, run_experiment

result = run_experiment(ExperimentConfig(
    suite="gym", env_id="CartPole-v1", mode="delta",
    partial="example_cartpole", timesteps=200_000,
))
print(result.metadata["selected_policy_true_reward_mean"])
```

## Run outputs

Each run directory contains `metadata.json`, monitor CSVs (`monitor/`),
`best_model/`, `eval/` (`evaluations.npz`, `component_evaluations.csv`,
`final_component_evaluation.csv`), `true_reward_curve.png`,
`final_model.zip`, and `vecnormalize.pkl` when observation normalization is
used.
