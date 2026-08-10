# Shared environment table, sourced by the job scripts so all five environments
# are defined in exactly one place.
#
# Each environment uses one aligned-but-incomplete partial: the true reward with
# some terms dropped or capped (usually the control cost). Same character of
# partial across all five, so partiality is comparable between environments.
#
# Exports per cell: SUITE ENV PARTIAL PARTIAL_LOWCAP STEPS EXTRA TUNED
#
# ---------------------------------------------------------------------------
# CHANGED 2026-08-10 - Hopper/Walker2d cut from 5M to 2M steps.
#
# The "partial beats the true reward" result on these two was an artifact of the
# PPO setup, not a property of the partials:
#
#   Hopper true   peak 3532 @3.45M -> final 1830 @5M   (drawdown 20%, 5/10 seeds >20%)
#   Hopper partial peak 2627       -> final 2440       (drawdown 3%,  0/10 seeds)
#   Walker true   peak 6285        -> final 5945       (true actually WINS)
#   Walker partial peak 5922       -> final 5793
#
# Neither inversion is significant (Hopper Mann-Whitney p=0.076 / paired p=0.131;
# Walker p=1.000 / p=0.695). Hopper's true arm reaches a BETTER policy than the
# partial and then loses it, because everything except Reacher ran on stock SB3
# defaults (lr 3e-4 constant, n_steps 2048, no clip_range_vf) for 5x the budget
# the tuned reference uses. Walker's true arm was simply not converged: at 1M it
# sat at 728 against the tuned reference's 3479, and was still gaining +15% in
# the last quarter of training.
#
# So: 2M steps (rl-zoo budgets 1M; we run 8 parallel envs, so updates are larger)
# and --tuned-hyperparams, which pulls the per-env PPO block from
# rcomp/ppo_presets.py. Inspect it with `python -m rcomp list-presets --env-id X`.
#
# WARNING: resubmitting an OLD hopper/walker job to fill gaps will now run at 2M
# and will not be comparable with the 5M runs already in that log directory.
# Pin the old value with --timesteps 5000000 if you are gap-filling.
# ---------------------------------------------------------------------------

set_env_vars () {
  # Default: no tuned preset, so any job that does not opt in behaves exactly as
  # it did before. Only the two environments that were actually broken opt in.
  TUNED=""
  PARTIAL_LOWCAP=""

  case "$1" in
    ll)
      SUITE=box2d;  ENV=LunarLander-v3; PARTIAL=lunar_lander_approach
      STEPS=2000000; EXTRA="--collection-timesteps 30000" ;;
    reacher)
      SUITE=mujoco; ENV=Reacher-v5;     PARTIAL=reacher_distance_partial
      STEPS=5000000; EXTRA="" ;;
    pusher)
      SUITE=mujoco; ENV=Pusher-v5;      PARTIAL=pusher_honest
      STEPS=3000000; EXTRA="" ;;
    # Hopper/Walker2d: PARTIAL is back to the ORIGINAL partial, because the
    # premise for replacing it (that it beat the true reward) did not survive
    # the check above. PARTIAL_LOWCAP still exposes the replacement from
    # partials/mujoco_capped_low.py for jobs that want to test it - re-run E0
    # with the tuned setup first and only switch if the inversion is still there.
    hopper)
      SUITE=mujoco; ENV=Hopper-v5;      PARTIAL=hopper_capped_forward_survive
      PARTIAL_LOWCAP=mujoco_capped_low:hopper_capped_low
      STEPS=2000000; EXTRA=""; TUNED="--tuned-hyperparams" ;;
    walker)
      SUITE=mujoco; ENV=Walker2d-v5;    PARTIAL=walker2d_survive_forward
      PARTIAL_LOWCAP=mujoco_capped_low:walker2d_capped_low
      STEPS=2000000; EXTRA=""; TUNED="--tuned-hyperparams" ;;
    *) echo "unknown env cell: $1" >&2; exit 1 ;;
  esac
}
