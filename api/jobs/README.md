# Triton experiment scripts

Two array jobs. Everything is version-controlled, so on Triton it is
`git pull` then `sbatch` — no pasting long commands into a terminal.

## What changed since the previous batch

| Fix | How |
|---|---|
| Gate trained on data the model had already memorized | `--gate-holdout` fits and early-stops the gate on two halves of the held-out preferences |
| Gate double-counted the partial (it was also a model input) | `--no-include-partial-feature` on every arm of job 1 |
| Gate started at 0.5 and drifted to 0 | `--gate-init 0.95` + `--gate-prior-penalty 0.01`, so "use the partial fully" is the null hypothesis |
| Learning rate 0.01 saturated the sigmoid | `--gate-lr 0.001` |
| No way to tell whether the gate learned trust | `--gate-diagnostic` records `corr(g, \|partial - true\|)` |
| Scores came from an oracle best-of-20 checkpoint pick | `--final-policy last` |
| LunarLander stalled at ~1050 of 1400 queries | `--collection-timesteps 30000` (verified: 1400/1400) |
| HalfCheetah is bimodal, its medians are noise | replaced by Reacher-v5 |
| Coarse eval curves (20 points x 5 episodes) | `--eval-freq 50000 --n-eval-episodes 20` |
| Runs lost to wall-clock timeouts | `--time=24:00:00`, `--requeue`, and the scripts skip finished runs so a resubmit only fills gaps |

## Job 1 — `run_gatealpha.sh` (300 runs)

Does the per-state gate beat a plain constant alpha? The gate can only shrink the
partial, so a gate that helps may just be acting as a smaller alpha. Fixed-alpha
controls at 1.0 / 0.5 / 0.25 / 0.1 run under otherwise identical settings.

4 cells x 5 variants x 15 seeds, budget 700.

Cells span the safe/unsafe shaping axis:

| Cell | Env | Partial | Shaping |
|---|---|---|---|
| `ll_approach` | LunarLander-v3 | `lunar_lander_approach` | potential-based (safe) |
| `ll_p50` | LunarLander-v3 | `lunarlander_p50` | potential-based **+ non-potential terminal bonus** (unsafe) |
| `pusher` | Pusher-v5 | `pusher_honest` | aligned subset of the true reward |
| `reacher` | Reacher-v5 | `reacher_distance_partial` | aligned subset of the true reward |

## Job 2 — `run_main.sh` (270 runs)

The headline table, scored on the final policy instead of an oracle checkpoint.
3 cells x {feedback, naive} x {350, 700, 1400} x 15 seeds.

## Running them

```bash
ssh akishea1@triton.aalto.fi

cd /scratch/work/akishea1/reward_composition_api
git pull

cd api
module load mamba
source activate /scratch/work/akishea1/envs/rcomp
pip install -e .            # picks up the new gate flags

# sanity: the new flags exist and the params files are the expected length
python -m rcomp train --help | grep -oE '^  --gate-[a-z-]+' | wc -l   # expect 8 (--gate-partial plus the 7 new ones)
wc -l jobs/params_gatealpha.txt                                       # expect 300
wc -l jobs/params_main.txt                                            # expect 270

mkdir -p logs/slurm
sbatch jobs/run_gatealpha.sh
sbatch jobs/run_main.sh
slurm q
```

## Filling gaps after a failure

Both scripts exit immediately if `metadata.json` already exists, so resubmitting
the whole array only re-runs what is missing:

```bash
sbatch jobs/run_gatealpha.sh
sbatch jobs/run_main.sh
```

To find what is missing first:

```bash
cd /scratch/work/akishea1/reward_composition_api/api
for d in logs/ga_* logs/last_*; do
  total=$(ls -d $d/*/ 2>/dev/null | wc -l)
  done_=$(ls $d/*/metadata.json 2>/dev/null | wc -l)
  echo "$d: $done_/$total complete"
done
```

## Fetching results (run in a local PowerShell window)

```powershell
cd C:\Users\PC\Documents\coding\research\reconstructing\api\logs
scp -r akishea1@triton.aalto.fi:/scratch/work/akishea1/reward_composition_api/api/logs/ga_* .
scp -r akishea1@triton.aalto.fi:/scratch/work/akishea1/reward_composition_api/api/logs/last_* .
```

## What to read out of the results

**Job 1, the decisive question.** Compare the `gate` arm against the best fixed
alpha in the same cell. If some fixed alpha matches the gate, the gate is not a
contribution and a one-line setting replaces it. If the gate beats every fixed
alpha, the per-state part is doing real work.

**Job 1, the direct test.** `gate_error_stats.corr_gate_partial_error` in each
gate run's `metadata.json`. Negative means the gate closes where the partial
disagrees with the true reward, which is what the gate is supposed to do. Near
zero (or `null`, meaning the gate collapsed to a constant) means it is not
tracking reliability at all, whatever it did to the score.

**Job 2.** Medians and success rates per cell, plus peak-vs-final from
`eval/evaluations.npz` — with `--final-policy last` the collapse is no longer
hidden by the checkpoint picker.

Always report medians and success/failure rates, not means. Reacher fails
catastrophically on roughly 1 seed in 5; that is expected and is why the median
is the statistic to quote.
