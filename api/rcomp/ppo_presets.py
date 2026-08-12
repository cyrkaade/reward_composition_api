"""Per-environment PPO hyperparameters taken from the literature.

WHY THIS FILE EXISTS
--------------------
Only Reacher ever had a tuned PPO config in this project; every other
environment fell through to stable-baselines3's stock defaults (n_steps 2048,
lr 3e-4 constant, clip_range 0.2, gae_lambda 0.95). That is what produced the
Hopper "the partial beats the true reward" artifact: PPO on the true Hopper
reward peaks at ~3530 (above the tuned reference of 2410) and then loses 20% of
it by 5M steps, because lr 3e-4 is held constant with no value-function
clipping on an environment with early termination. Walker2d had the mirror
problem - at 1M steps it reached 728 against a tuned reference of 3479, i.e.
roughly 4x less sample-efficient, so its "true" arm had not converged when the
budget ran out.

Values are transcribed from ``hyperparams/ppo.yml`` in DLR-RM/rl-baselines3-zoo
(the numbers cross-check against the published sb3/ppo-<env> model cards), plus
the PrefPPO block of B-Pref (arXiv 2111.03026, Table 2) which is the reference
*preference-based* on-policy setup. Each entry records where it came from and
what score the source reports, so a claim like "our PPO is weak" can be checked
against a number instead of an impression.

HOW TO USE IT
-------------
Nothing here is applied unless a run passes ``--tuned-hyperparams``; the default
remains stock behaviour so the ~6,000 archived runs stay comparable. Precedence
inside :meth:`rcomp.suites.Suite.ppo_hyperparams` is::

    suite defaults  ->  this file (only with --tuned-hyperparams)  ->  --policy-learning-kwargs

Inspect what a run would use with ``python -m rcomp list-presets [--env-id X]``.

NO ZOO BLOCK IN THIS FILE WAS TUNED ON A v5 ENVIRONMENT
-------------------------------------------------------
Verified against ``hyperparams/ppo.yml`` on 2026-08-12: the zoo's MuJoCo entries
are keyed ``-v2``/``-v3``/``-v4``, and its benchmark table reports ``-v3`` scores.
There is no published PPO block tuned on any ``-v5`` environment. Every MuJoCo
preset here is therefore a v2/v4 block applied to v5, which matters because v5
changed reward code, not just plumbing:

- **Pusher-v5 fixed a reward bug.** ``reward_dist`` and ``reward_near`` were
  computed from the state *before* the physics step and are now computed after.
  v5 also fixed ``info["reward_ctrl"]`` not being multiplied by its weight and
  added ``info["reward_near"]``. ``partials/pusher_honest.py`` reads those two
  keys, so it is correct on v5 and would have been wrong on v4.
- **Reacher-v5** carries the same reward-timing fix, so the Reacher-v2 block in
  this file was tuned against a different reward function than the one it runs.

Neither is fatal - Pusher and Reacher both converge cleanly here with tight true
arms (Pusher -26.0..-21.9, Reacher -4.4..-3.3 over 10 seeds, 0 failures) - but a
paper should say "rl-zoo's v4 block applied to v5", not "tuned for v5".

FOUR THINGS TO KNOW BEFORE RELYING ON A PRESET
----------------------------------------------
1. ``reference`` is NOT applied, and ON HOPPER THAT BROKE THE RUN (measured
   2026-08-10, ``logs/e0t_hopper_*``). It records the source's ``n_envs``,
   ``n_timesteps`` and ``normalize`` so job scripts can match the recipe
   deliberately. Most sources use ``n_envs: 1``; this project uses 8, which
   makes each update 8x larger and 8x rarer than the source intended.

   Hopper at ``n_envs=8`` with this preset gets 488 updates over 2M steps
   instead of the source's ~3,900, at lr 9.8e-5 (a tenth of stock) with
   ``log_std_init=-2`` and ``gamma 0.999``. Every seed locked onto the
   survive-only local optimum within 0.2M steps and never left: final 1011 vs
   2073 for stock SB3 at the same budget (p=0.014), episode length 1000.0,
   total ``reward_forward`` 1.0. The agent stands still for the full episode.

   So: **pass ``--n-envs 1`` alongside ``--tuned-hyperparams`` for any preset
   whose ``reference`` says ``n_envs: 1``**, unless you have checked that the
   larger batch still works for that env. DummyVecEnv steps serially, so
   ``n_envs`` is close to wall-clock neutral. Walker2d survived the mismatch
   (4986 vs 3038 at 2M, +1949) but is the exception, not the rule.
2. Learning-rate / clip-range *schedules* (``lin_*`` in the zoo) are unsafe in
   preference modes. ``trainer.train_policy_round`` calls ``learn()`` once per
   RLHF round with ``reset_num_timesteps=False``, so SB3 recomputes
   ``_total_timesteps = round_steps + num_timesteps`` and the schedule
   sawtooths: it anneals to zero inside every round and jumps back up at the
   next one. :func:`resolve_ppo_preset` warns when it hands one out.
3. ``tuned: False`` entries are placeholders, not tuned configs. rl-zoo has no
   tuned PPO block for Ant (only the shared ``normalize/1e6/MlpPolicy`` anchor)
   and none at all for Pusher. PPO reaches 1327 +/- 452 on Ant against SAC's
   4616, so a missing entry there is a statement about PPO, not an oversight.
4. Observation/reward normalization is decided by
   ``Suite.should_normalize_observation``, not here. The zoo runs LunarLander
   *without* VecNormalize while this project wraps it; that difference is left
   alone deliberately because the LunarLander results are the paper's core.
"""

from __future__ import annotations

import re
import warnings
from copy import deepcopy
from typing import Any

__all__ = [
    "PRESETS",
    "PresetError",
    "LinearSchedule",
    "preset_key",
    "lookup_preset",
    "resolve_ppo_preset",
    "resolve_activation_fn",
    "tuned_ppo_hyperparams",
    "describe_presets",
]


class PresetError(Exception):
    """Raised when a tuned preset is requested for an environment that has none."""


class LinearSchedule:
    """SB3-style schedule: ``progress_remaining`` 1 -> 0 maps to ``initial`` -> 0.

    A named class rather than a lambda so it survives ``PPO.save`` and shows up
    legibly in logs and metadata.
    """

    def __init__(self, initial: float):
        self.initial = float(initial)

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial

    def __repr__(self) -> str:
        return f"LinearSchedule({self.initial!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LinearSchedule) and other.initial == self.initial


def _lin(initial: float) -> dict[str, Any]:
    return {"schedule": "linear", "initial": initial}


# Shared policy_kwargs block used by the Optuna-tuned MuJoCo entries in the zoo.
_MUJOCO_POLICY_KWARGS: dict[str, Any] = {
    "log_std_init": -2,
    "ortho_init": False,
    "activation_fn": "relu",
    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
}

# SB3 stock defaults, kept explicit so `tuned: False` entries are readable.
_SB3_DEFAULT_PPO: dict[str, Any] = {
    "policy": "MlpPolicy",
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "ent_coef": 0.0,
    "clip_range": 0.2,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
}


# Keys are version-stripped base names ("Hopper" matches Hopper-v3/v4/v5).
# An exact env id ("ALE/Breakout-v5") may also be used and wins over the base
# name; "atari" is the catch-all for the ALE suite.
PRESETS: dict[str, dict[str, Any]] = {
    # ---------------------------------------------------------------- MuJoCo
    "Hopper": {
        "suite": "mujoco",
        "tuned": False,
        "source": (
            "SB3 defaults. Neither the rl-zoo Hopper-v4 block nor a gSDE-derived "
            "replacement beat them on Hopper-v5; see note. Kept explicit so the flag is "
            "inert here instead of silently degrading the run."
        ),
        "benchmark": (
            "measured here on Hopper-v5, 10 seeds x 5M, n_envs=1, VecNormalize on, "
            "--final-policy last (logs/hop5m_stock): median peak 3369, median final 1410, "
            "median drawdown 56%. rl-zoo reports PPO 2410 +/- 10 @1M (Hopper-v3); "
            "SAC 2326, TQC 3754. Hopper is EXCLUDED from the paper - its true arm collapses "
            "under every configuration measured."
        ),
        "reference": {"n_envs": 1, "n_timesteps": 5_000_000, "normalize": True},
        "note": (
            "REVERTED TO SB3 DEFAULTS 2026-08-12. Two configs have now been tried and both lost. "
            "(a) The rl-zoo Hopper-v4 block (n_steps 512, lr 9.8e-5, gamma .999, gae .99, "
            "n_epochs 5, ReLU, log_std_init -2, ortho_init False) pins Hopper-v5 at the "
            "survive-only local optimum: ~1000 reward, episode length 1000, total "
            "reward_forward 1.0 - the agent stands still and banks the +1/step healthy bonus, "
            "at n_envs 1 or 8, out to 560k. Four single-variable rescues all failed. "
            "(b) The gSDE-derived replacement (arXiv 2005.05719 minus gSDE: n_steps 512, "
            "batch 32, lr 3e-5, clip .4, clip_range_vf .5, n_epochs 20, gae .9, SB3-default "
            "policy_kwargs) looked good on a 3-seed x 1M pilot (peak 2995, final 2127) but "
            "was measured head-to-head against stock at 10 seeds x 5M, matched n_envs=1 and "
            "normalization (logs/hop5m_*): it LOSES. Peak 2390 vs 3369 (stock wins 9/10 "
            "seeds, exact p=0.004); final 381 vs 1410 (p=0.037); drawdown 76% vs 56%; 2 of 10 "
            "seeds never left the survive-only optimum. The 1M pilot was measuring the wrong "
            "budget - stock's median peak is at 525k-3.45M depending on n_envs, so a 1M "
            "readout flattered the preset. Restoring SB3 defaults makes --tuned-hyperparams "
            "a no-op on Hopper rather than an active regression. "
            "SEPARATELY: Hopper's TRUE arm collapses under every config tried - 20% drawdown "
            "for stock at n_envs 8, 56% for stock at n_envs 1, 76% for the gSDE config, and "
            "the rl-zoo block never learns. PPO on the ground-truth reward is unstable here "
            "regardless of hyperparameters, which is an environment property, not a setup "
            "bug. Hopper therefore cannot carry any claim about reward-model "
            "overoptimization; always report peak next to final."
        ),
        "ppo": deepcopy(_SB3_DEFAULT_PPO),
    },
    "Walker2d": {
        "suite": "mujoco",
        "tuned": True,
        "source": "rl-zoo ppo.yml Walker2d-v4 (Optuna-tuned)",
        "benchmark": (
            "PPO 3479 +/- 822 @ 1M (Walker2d-v3); SAC 3863, TQC 4381. "
            "Measured here on Walker2d-v5 at n_envs=8, 2M: 4986 vs 3038 for stock SB3 (+1949, "
            "p=0.064), drawdown 1.6%"
        ),
        "reference": {"n_envs": 1, "n_timesteps": 1_000_000, "normalize": True},
        "note": (
            "Unlike Hopper, this rl-zoo block DOES work here, and it survives n_envs=8. Keep "
            "it. The two differences that matter: it has no log_std_init override (so SB3's "
            "default 0 applies, not -2) and gamma is 0.99 rather than 0.999 - exactly the two "
            "settings that sink the Hopper block. Walker was still climbing +15% in the last "
            "quarter at 2M, so give it 5M."
        ),
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 512,
            "batch_size": 32,
            "gamma": 0.99,
            "learning_rate": 5.05041e-05,
            "ent_coef": 0.000585045,
            "clip_range": 0.1,
            "n_epochs": 20,
            "gae_lambda": 0.95,
            "max_grad_norm": 1,
            "vf_coef": 0.871923,
        },
    },
    "HalfCheetah": {
        "suite": "mujoco",
        "tuned": True,
        "source": "rl-zoo ppo.yml HalfCheetah-v4 (Optuna-tuned)",
        "benchmark": "PPO 5819 +/- 664 @ 1M (HalfCheetah-v3)",
        "reference": {"n_envs": 1, "n_timesteps": 1_000_000, "normalize": True},
        "note": (
            "CANDIDATE, NOT YET QUALIFIED - pilot before building a grid on it. "
            "HalfCheetah-v5 never terminates (fixed 1000-step episodes, verified), which "
            "is the low-variance property Walker2d and Hopper lack, and its partial "
            "(partials/halfcheetah_forward.py = reward_forward, dropping a 0.1-weight "
            "control cost on 6 actuators) has the same shape as pusher_honest, the only "
            "prior in this project that is genuinely incomplete. Dynamic range is large "
            "(~-300 untrained to ~5800), unlike Reacher's 2.4 units. "
            "THE KNOWN RISK is bimodality: ~45% of seeds learn to run (~5000) and the rest "
            "do not (~1800), with nothing between, so a median just reports which mode the "
            "middle seed hit. But that was measured on STOCK SB3 - an audit on 2026-08-12 "
            "found all 399 archived HalfCheetah runs have tuned_hyperparams unset, and NOT "
            "ONE of them is a mode=true run. The block below has never been tried here and "
            "there is no ground-truth baseline to compare against. Run 5 seeds of mode=true "
            "and 5 of mode=partial first: if the true arm is unimodal it is the best "
            "remaining environment; if it splits, drop it the way Walker2d was dropped."
        ),
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 512,
            "batch_size": 64,
            "gamma": 0.98,
            "learning_rate": 2.0633e-05,
            "ent_coef": 0.000401762,
            "clip_range": 0.1,
            "n_epochs": 20,
            "gae_lambda": 0.92,
            "max_grad_norm": 0.8,
            "vf_coef": 0.58096,
            "policy_kwargs": deepcopy(_MUJOCO_POLICY_KWARGS),
        },
    },
    "Reacher": {
        "suite": "mujoco",
        "tuned": True,
        "source": "rl-zoo ppo.yml Reacher-v2 (Optuna-tuned)",
        "benchmark": "median ~ -3.9 @ 5M in this project (closer to 0 is better)",
        "reference": {"n_envs": 1, "n_timesteps": 1_000_000, "normalize": True},
        "note": "Identical to the pre-existing MuJoCoSuite 'reacher' preset.",
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 512,
            "batch_size": 32,
            "gamma": 0.9,
            "learning_rate": 0.000104019,
            "ent_coef": 7.52585e-08,
            "clip_range": 0.3,
            "n_epochs": 5,
            "gae_lambda": 1.0,
            "max_grad_norm": 0.9,
            "vf_coef": 0.950368,
            "policy_kwargs": {
                "log_std_init": -2,
                "ortho_init": False,
                "activation_fn": "relu",
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        },
    },
    "Humanoid": {
        "suite": "mujoco",
        "tuned": True,
        "source": "rl-zoo ppo.yml Humanoid-v4 (Optuna-tuned)",
        "benchmark": "no PPO entry in the zoo benchmark table; SAC 6232 and TQC 7239 @ 2M",
        "reference": {"n_envs": 1, "n_timesteps": 10_000_000, "normalize": True},
        "note": (
            "Wants 1e7 steps for a single run. Multiply by 5 RLHF rounds and 15 "
            "seeds before proposing it. The absence of a PPO benchmark score is "
            "the relevant signal."
        ),
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 512,
            "batch_size": 256,
            "gamma": 0.95,
            "learning_rate": 3.56987e-05,
            "ent_coef": 0.00238306,
            "clip_range": 0.3,
            "n_epochs": 5,
            "gae_lambda": 0.9,
            "max_grad_norm": 2,
            "vf_coef": 0.431892,
            "policy_kwargs": deepcopy(_MUJOCO_POLICY_KWARGS),
        },
    },
    "HumanoidStandup": {
        "suite": "mujoco",
        "tuned": True,
        "source": "rl-zoo ppo.yml HumanoidStandup-v2 (Optuna-tuned)",
        "benchmark": "not in the zoo benchmark table",
        "reference": {"n_envs": 1, "n_timesteps": 10_000_000, "normalize": True},
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 512,
            "batch_size": 32,
            "gamma": 0.99,
            "learning_rate": 2.55673e-05,
            "ent_coef": 3.62109e-06,
            "clip_range": 0.3,
            "n_epochs": 20,
            "gae_lambda": 0.9,
            "max_grad_norm": 0.7,
            "vf_coef": 0.430793,
            "policy_kwargs": deepcopy(_MUJOCO_POLICY_KWARGS),
        },
    },
    "Swimmer": {
        "suite": "mujoco",
        "tuned": True,
        "source": "rl-zoo ppo.yml Swimmer-v4",
        "benchmark": "PPO 282 +/- 10 @ 1M (Swimmer-v3)",
        "reference": {"n_envs": 4, "n_timesteps": 1_000_000, "normalize": True},
        "note": (
            "CANDIDATE, NOT YET QUALIFIED. Swimmer-v5 is the only environment available "
            "here that CANNOT terminate at all - it always runs the full 1000 steps "
            "(verified), so none of the Walker2d/Hopper failure modes (early termination, "
            "bimodal ceilings, episode-length farming, fragment starvation) are even "
            "expressible. That makes it the lowest-variance option in the suite. "
            "GAMMA 0.9999 IS LOAD-BEARING, not a tuning detail: Swimmer's forward reward "
            "per step is tiny and the credit for a stroke arrives many steps later, so at "
            "gamma 0.99 (effective horizon 100 of 1000 steps) PPO does not learn to swim. "
            "Do not override it, and do not copy this preset's gamma to any other entry. "
            "A 'delete the control cost' partial does NOT work here - ctrl_cost_weight is "
            "1e-4, so true-minus-cost is effectively the true reward. Use a direction-blind "
            "partial instead: sqrt(obs[3]^2 + obs[4]^2), the SPEED of the front tip rather "
            "than its x-velocity. Measured on 400 random steps, that correlates 0.23 with "
            "reward_forward while obs[3] alone correlates 0.97 - it rewards swimming fast "
            "in the wrong direction, which is a genuinely incomplete prior and the mistake "
            "a human designer actually makes."
        ),
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 1024,
            "batch_size": 256,
            "gamma": 0.9999,
            "learning_rate": 6e-4,
            "gae_lambda": 0.98,
        },
    },
    "InvertedPendulum": {
        "suite": "mujoco",
        "tuned": True,
        "source": "rl-zoo ppo.yml InvertedPendulum-v2 (Optuna-tuned)",
        "benchmark": "not in the zoo benchmark table",
        "reference": {"n_envs": 1, "n_timesteps": 1_000_000, "normalize": True},
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 32,
            "batch_size": 64,
            "gamma": 0.999,
            "learning_rate": 0.000222425,
            "ent_coef": 1.37976e-07,
            "clip_range": 0.4,
            "n_epochs": 5,
            "gae_lambda": 0.9,
            "max_grad_norm": 0.3,
            "vf_coef": 0.19816,
        },
    },
    "InvertedDoublePendulum": {
        "suite": "mujoco",
        "tuned": True,
        "source": "rl-zoo ppo.yml InvertedDoublePendulum-v2 (Optuna-tuned)",
        "benchmark": "not in the zoo benchmark table",
        "reference": {"n_envs": 1, "n_timesteps": 1_000_000, "normalize": True},
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 128,
            "batch_size": 512,
            "gamma": 0.98,
            "learning_rate": 0.000155454,
            "ent_coef": 1.05057e-06,
            "clip_range": 0.4,
            "n_epochs": 10,
            "gae_lambda": 0.8,
            "max_grad_norm": 0.5,
            "vf_coef": 0.695929,
        },
    },
    "Ant": {
        "suite": "mujoco",
        "tuned": False,
        "source": "rl-zoo ppo.yml has only the shared anchor (normalize/1e6/MlpPolicy) for Ant-v4",
        "benchmark": "PPO 1327 +/- 452 @ 1M (Ant-v3) vs SAC 4616, TQC 3339",
        "reference": {"n_envs": 1, "n_timesteps": 1_000_000, "normalize": True},
        "note": (
            "NOT a tuned config - SB3 defaults. PPO reaches under a third of SAC "
            "here, so Ant fails the premise that the true-reward arm is a usable "
            "ceiling. Do not add it to a PPO-only pipeline."
        ),
        "ppo": deepcopy(_SB3_DEFAULT_PPO),
    },
    "Pusher": {
        "suite": "mujoco",
        "tuned": False,
        "source": "no PPO block in rl-zoo ppo.yml; SB3 defaults",
        "benchmark": "median ~ -23 @ 3M in this project (closer to 0 is better)",
        "reference": {"n_envs": 8, "n_timesteps": 3_000_000, "normalize": True},
        "note": (
            "NOT a tuned config, but it converges by ~1.5M with no drawdown, so "
            "the defaults are adequate here. Kept explicit so the gap is visible."
        ),
        "ppo": deepcopy(_SB3_DEFAULT_PPO),
    },
    # ----------------------------------------------------------------- Box2D
    "LunarLander": {
        "suite": "box2d",
        "tuned": True,
        "source": "rl-zoo ppo.yml LunarLander-v3",
        "benchmark": "PPO 242 +/- 32 @ 1M (LunarLander-v2); solved >= 200",
        "reference": {"n_envs": 16, "n_timesteps": 1_000_000, "normalize": False},
        "note": (
            "The zoo runs this WITHOUT VecNormalize; Suite.should_normalize_observation "
            "still wraps it here. Left alone on purpose - the LunarLander results are "
            "the paper's core and the stock config already reaches 283 (true arm, 3% "
            "drawdown, 0/10 seeds collapsing), so there is nothing to fix."
        ),
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 1024,
            "batch_size": 64,
            "gae_lambda": 0.98,
            "gamma": 0.999,
            "n_epochs": 4,
            "ent_coef": 0.01,
        },
    },
    "LunarLanderContinuous": {
        "suite": "box2d",
        "tuned": True,
        "source": "rl-zoo ppo.yml LunarLanderContinuous-v3",
        "benchmark": "shares the discrete LunarLander block",
        "reference": {"n_envs": 16, "n_timesteps": 1_000_000, "normalize": False},
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 1024,
            "batch_size": 64,
            "gae_lambda": 0.98,
            "gamma": 0.999,
            "n_epochs": 4,
            "ent_coef": 0.01,
        },
    },
    "BipedalWalker": {
        "suite": "box2d",
        "tuned": True,
        "source": "rl-zoo ppo.yml BipedalWalker-v3",
        "benchmark": "solved >= 300",
        "reference": {"n_envs": 32, "n_timesteps": 5_000_000, "normalize": True},
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 2048,
            "batch_size": 64,
            "gae_lambda": 0.95,
            "gamma": 0.999,
            "n_epochs": 10,
            "ent_coef": 0.0,
            "learning_rate": 3e-4,
            "clip_range": 0.18,
        },
    },
    "BipedalWalkerHardcore": {
        "suite": "box2d",
        "tuned": True,
        "source": "rl-zoo ppo.yml BipedalWalkerHardcore-v3",
        "benchmark": "the zoo budgets 1e8 steps for this",
        "reference": {"n_envs": 16, "n_timesteps": 100_000_000, "normalize": True},
        "note": "Uses linear schedules - see the schedule warning in the module docstring.",
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 2048,
            "batch_size": 64,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "n_epochs": 10,
            "ent_coef": 0.001,
            "learning_rate": _lin(2.5e-4),
            "clip_range": _lin(0.2),
        },
    },
    "CarRacing": {
        "suite": "box2d",
        "tuned": True,
        "source": "rl-zoo ppo.yml CarRacing-v3",
        "benchmark": "not in the zoo benchmark table",
        "reference": {"n_envs": 8, "n_timesteps": 4_000_000, "normalize": "reward only"},
        "note": (
            "The zoo pairs this with FrameSkip(2), 64x64 grayscale resize and "
            "frame_stack 2, none of which this codebase applies. The PPO kwargs "
            "alone will not reproduce the source result. Uses a linear LR."
        ),
        "ppo": {
            "policy": "CnnPolicy",
            "n_steps": 512,
            "batch_size": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_epochs": 10,
            "ent_coef": 0.0,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "learning_rate": _lin(1e-4),
            "clip_range": 0.2,
            "use_sde": True,
            "sde_sample_freq": 4,
            "policy_kwargs": {
                "log_std_init": -2,
                "ortho_init": False,
                "activation_fn": "gelu",
                "net_arch": {"pi": [256], "vf": [256]},
            },
        },
    },
    # -------------------------------------------------------- classic control
    "CartPole": {
        "suite": "gym",
        "tuned": True,
        "source": "rl-zoo ppo.yml CartPole-v1",
        "benchmark": "solved = 500",
        "reference": {"n_envs": 8, "n_timesteps": 100_000, "normalize": False},
        "note": "Uses linear schedules - see the schedule warning in the module docstring.",
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 32,
            "batch_size": 256,
            "gae_lambda": 0.8,
            "gamma": 0.98,
            "n_epochs": 20,
            "ent_coef": 0.0,
            "learning_rate": _lin(0.001),
            "clip_range": _lin(0.2),
        },
    },
    "MountainCar": {
        "suite": "gym",
        "tuned": True,
        "source": "rl-zoo ppo.yml MountainCar-v0",
        "benchmark": "solved >= -110",
        "reference": {"n_envs": 16, "n_timesteps": 1_000_000, "normalize": True},
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 16,
            "gae_lambda": 0.98,
            "gamma": 0.99,
            "n_epochs": 4,
            "ent_coef": 0.0,
        },
    },
    "MountainCarContinuous": {
        "suite": "gym",
        "tuned": True,
        "source": "rl-zoo ppo.yml MountainCarContinuous-v0 (Optuna-tuned)",
        "benchmark": "solved >= 90",
        "reference": {"n_envs": 1, "n_timesteps": 20_000, "normalize": True},
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 8,
            "batch_size": 256,
            "gamma": 0.9999,
            "learning_rate": 7.77e-05,
            "ent_coef": 0.00429,
            "clip_range": 0.1,
            "n_epochs": 10,
            "gae_lambda": 0.9,
            "max_grad_norm": 5,
            "vf_coef": 0.19,
            "use_sde": True,
            "policy_kwargs": {"log_std_init": -3.29, "ortho_init": False},
        },
    },
    "Acrobot": {
        "suite": "gym",
        "tuned": True,
        "source": "rl-zoo ppo.yml Acrobot-v1",
        "benchmark": "not in the zoo benchmark table",
        "reference": {"n_envs": 16, "n_timesteps": 1_000_000, "normalize": True},
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 256,
            "gae_lambda": 0.94,
            "gamma": 0.99,
            "n_epochs": 4,
            "ent_coef": 0.0,
        },
    },
    "Pendulum": {
        "suite": "gym",
        "tuned": True,
        "source": "rl-zoo ppo.yml Pendulum-v1",
        "benchmark": "not in the zoo benchmark table",
        "reference": {"n_envs": 4, "n_timesteps": 100_000, "normalize": False},
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 1024,
            "gae_lambda": 0.95,
            "gamma": 0.9,
            "n_epochs": 10,
            "ent_coef": 0.0,
            "learning_rate": 1e-3,
            "clip_range": 0.2,
            "use_sde": True,
            "sde_sample_freq": 4,
        },
    },
    # ----------------------------------------------------------------- Atari
    "atari": {
        "suite": "atari",
        "tuned": True,
        "source": "rl-zoo ppo.yml shared `atari:` block",
        "benchmark": "the zoo budgets 1e7 steps per game",
        "reference": {"n_envs": 8, "n_timesteps": 10_000_000, "normalize": False},
        "note": (
            "Identical to the existing AtariSuite defaults except that lr and "
            "clip_range are linearly annealed here. `policy` is deliberately "
            "absent: the zoo block assumes image observations + CnnPolicy, while "
            "AtariSuite.make_raw_env uses obs_type='ram' and MlpPolicy, so the "
            "suite's own choice must stand. The zoo also applies AtariWrapper + "
            "frame_stack 4, which this codebase does not. Atari runs here have "
            "historically not learned - see CLAUDE.md."
        ),
        "ppo": {
            "n_steps": 128,
            "batch_size": 256,
            "n_epochs": 4,
            "learning_rate": _lin(2.5e-4),
            "clip_range": _lin(0.1),
            "vf_coef": 0.5,
            "ent_coef": 0.01,
        },
    },
}


_VERSION_SUFFIX = re.compile(r"-v\d+$")


def preset_key(env_id: str) -> str:
    """Version-stripped lookup key: ``Hopper-v5`` -> ``Hopper``.

    ALE ids (``ALE/Breakout-v5``) collapse to ``atari`` because the zoo tunes
    one shared block for every game rather than per-game configs.
    """
    if env_id.startswith("ALE/"):
        return "atari"
    return _VERSION_SUFFIX.sub("", env_id)


def lookup_preset(env_id: str) -> dict[str, Any] | None:
    """Return the preset for ``env_id``, or ``None`` when there is none.

    An exact env id wins over the version-stripped base name, so a future
    ``"Hopper-v5"`` entry could override the shared ``"Hopper"`` one.
    """
    for key in (env_id, preset_key(env_id)):
        if key in PRESETS:
            return deepcopy(PRESETS[key])
    return None


def resolve_activation_fn(name: str):
    from torch import nn

    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        "leaky_relu": nn.LeakyReLU,
    }
    key = name.lower().replace(" ", "")
    if key not in activations:
        raise PresetError(f"Unknown activation_fn '{name}'. Known: {', '.join(sorted(activations))}")
    return activations[key]


def _resolve_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("schedule") == "linear":
        return LinearSchedule(value["initial"])
    return value


def resolve_ppo_preset(preset: dict[str, Any], *, warn_on_schedule: bool = True) -> dict[str, Any]:
    """Turn a stored preset into kwargs ready for ``PPO(**kwargs)``.

    Resolves ``activation_fn`` name strings to torch classes (imported lazily,
    so this module stays cheap for the CLI) and ``{"schedule": "linear"}``
    markers to :class:`LinearSchedule` callables.
    """
    hyperparams = deepcopy(preset.get("ppo", {}))

    scheduled = [name for name, value in hyperparams.items() if isinstance(value, dict) and value.get("schedule")]
    if scheduled and warn_on_schedule:
        warnings.warn(
            f"Preset uses schedule(s) for {', '.join(sorted(scheduled))}. SB3 recomputes progress "
            "per learn() call, and preference modes call learn() once per RLHF round with "
            "reset_num_timesteps=False, so the schedule restarts every round. Override with "
            "--policy-learning-kwargs for multi-round runs.",
            RuntimeWarning,
            stacklevel=2,
        )

    resolved = {name: _resolve_value(value) for name, value in hyperparams.items()}

    policy_kwargs = resolved.get("policy_kwargs")
    if isinstance(policy_kwargs, dict) and isinstance(policy_kwargs.get("activation_fn"), str):
        policy_kwargs = dict(policy_kwargs)
        policy_kwargs["activation_fn"] = resolve_activation_fn(policy_kwargs["activation_fn"])
        resolved["policy_kwargs"] = policy_kwargs

    return resolved


def tuned_ppo_hyperparams(env_id: str, *, warn_on_schedule: bool = True) -> dict[str, Any]:
    """Resolved PPO kwargs for ``env_id``; raises when no preset exists.

    Failing loudly is deliberate: a run launched with ``--tuned-hyperparams``
    that silently fell back to stock defaults would be indistinguishable in the
    logs from one that did not.
    """
    preset = lookup_preset(env_id)
    if preset is None:
        known = ", ".join(sorted(PRESETS))
        raise PresetError(
            f"No PPO preset for '{env_id}'. Add one to rcomp/ppo_presets.py or drop "
            f"--tuned-hyperparams. Known keys: {known}"
        )
    return resolve_ppo_preset(preset, warn_on_schedule=warn_on_schedule)


def describe_presets(env_id: str | None = None, suite: str | None = None) -> list[dict[str, Any]]:
    """Rows for ``rcomp list-presets``: one dict per matching preset."""
    if env_id is not None:
        preset = lookup_preset(env_id)
        if preset is None:
            return []
        return [{"key": preset_key(env_id), **preset}]
    return [
        {"key": key, **deepcopy(preset)}
        for key, preset in sorted(PRESETS.items())
        if suite is None or preset.get("suite") == suite
    ]
