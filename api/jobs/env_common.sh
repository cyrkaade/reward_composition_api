# Shared environment table, sourced by the job scripts so all five environments
# are defined in exactly one place.
#
# Each environment uses one aligned-but-incomplete partial: the true reward with
# some terms dropped or capped (usually the control cost). Same character of
# partial across all five, so partiality is comparable between environments.

set_env_vars () {
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
    # Hopper/Walker2d use the LOW-cap partials. The previous ones
    # (hopper_capped_forward_survive, walker2d_survive_forward) produced better
    # policies than training on the ground-truth reward itself, which inverts the
    # premise: with no gap between partial and true there is nothing for
    # preference learning to fill. See partials/mujoco_capped_low.py.
    hopper)
      SUITE=mujoco; ENV=Hopper-v5;      PARTIAL=mujoco_capped_low:hopper_capped_low
      STEPS=5000000; EXTRA="" ;;
    walker)
      SUITE=mujoco; ENV=Walker2d-v5;    PARTIAL=mujoco_capped_low:walker2d_capped_low
      STEPS=5000000; EXTRA="" ;;
    *) echo "unknown env cell: $1" >&2; exit 1 ;;
  esac
}
