# Usage reference

Run commands from the repository root. The examples use `python`; use the virtual-environment interpreter shown in the README when the environment is not activated.

## Custom partial rewards

Create `partials/my_reward.py`:

```python
def partial_reward(obs, action, next_obs, true_reward, terminated, truncated, info):
    value = float(info.get("my_component", 0.0))
    return {
        "partial": value,
        "components": {"my_component": value},
    }
```

Validate it before training:

```powershell
python -m reward_composition_api validate-partial `
  --suite mujoco --env-id Reacher-v5 --partial my_reward
```

`my_reward` resolves to `partials/my_reward.py`. A direct file path also works. For a file that registers several partials, use `file_or_module:registered_name`.

## Training

```powershell
python -m reward_composition_api train `
  --suite mujoco `
  --env-id Reacher-v5 `
  --mode delta `
  --partial my_reward `
  --timesteps 800000 `
  --query-budget 150 `
  --rlhf-rounds 5 `
  --seed 7 `
  --final-policy last
```

Suites are `gym`, `box2d`, `mujoco`, and `atari`. Use `list-envs` to see the environments installed for a suite.

Important preference-learning options:

- `--fragment-length`: transitions per compared fragment.
- `--query-budget`: total preference comparisons.
- `--rlhf-rounds`: collection/model-update/policy-update cycles.
- `--reward-model-ensemble-size 5`: train an uncertainty ensemble.
- `--active-learning --active-query-strategy auto`: use ensemble variance when an ensemble is present, otherwise MC dropout.
- `--final-policy last`: evaluate the final checkpoint without true-reward checkpoint selection.

For a fair comparison, keep the seed, PPO settings, query budget, collection budget, and evaluation protocol identical across `feedback`, `naive`, and `delta`.

## Partiality

```powershell
python -m reward_composition_api partiality `
  --suite mujoco `
  --env-id Reacher-v5 `
  --partial my_reward `
  --timesteps 60000 `
  --fragment-length 25 `
  --seed 7
```

The result reports two different quantities:

- `rho`: Pearson correlation between partial and true fragment returns. Use this when targeting correlation levels such as 0.4, 0.6, and 0.8.
- `partiality`: `cov(P, R) / var(R)`, which also changes when `P` is merely rescaled.

## Sweeps

The sweep command first prints a plan. Add `--execute` only after checking it.

```powershell
python -m reward_composition_api sweep `
  --suite mujoco `
  --env-ids Reacher-v5 `
  --seeds 0 1 2 `
  --timesteps 800000 `
  --partial my_reward
```

## Outputs

Each run directory contains the main reproducibility artifacts:

- `metadata.json`: configuration and final metrics.
- `final_model.zip`: final policy.
- `vecnormalize.pkl`: normalization statistics.
- `eval/evaluations.npz`: true-reward evaluation curve.
- `eval/component_evaluations.csv`: seeded component evaluations during training.
- `eval/final_component_evaluation.csv`: seeded final-policy component evaluation.
- `true_reward_curve.png`: plotted evaluation curve.

Summarize completed runs with:

```powershell
python -m reward_composition_api summarize --suite mujoco --root PATH_TO_RUNS
```
