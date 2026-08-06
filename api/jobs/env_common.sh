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
    hopper)
      SUITE=mujoco; ENV=Hopper-v5;      PARTIAL=hopper_capped_forward_survive
      STEPS=5000000; EXTRA="" ;;
    walker)
      SUITE=mujoco; ENV=Walker2d-v5;    PARTIAL=walker2d_survive_forward
      STEPS=5000000; EXTRA="" ;;
    *) echo "unknown env cell: $1" >&2; exit 1 ;;
  esac
}
