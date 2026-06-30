from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class SaveVecNormalizeOnBest(BaseCallback):
    def __init__(self, env: VecNormalize, save_path: Path):
        super().__init__()
        self.env = env
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.env.save(self.save_path)
        return True


def query_schedule(query_budget: int, rounds: int) -> list[int]:
    unit = query_budget // rounds
    schedule = [unit] * rounds
    for i in range(query_budget - sum(schedule)):
        schedule[i % len(schedule)] += 1
    return schedule


def policy_training_schedule(total_timesteps: int, rounds: int, timesteps_per_round: int | None = None) -> list[int]:
    if timesteps_per_round is not None:
        return [timesteps_per_round] * rounds

    policy_steps_per_round = total_timesteps // rounds
    leftover_policy_steps = total_timesteps - policy_steps_per_round * rounds
    return [
        policy_steps_per_round + (leftover_policy_steps if round_index == rounds - 1 else 0)
        for round_index in range(rounds)
    ]


def learn_policy(
    model,
    total_timesteps: int,
    callback,
    progress_bar: bool,
    reset_num_timesteps: bool = True,
    log_interval: int | None = None,
) -> None:
    if total_timesteps <= 0:
        return

    learn_kwargs = {
        "total_timesteps": int(total_timesteps),
        "callback": callback,
        "progress_bar": progress_bar,
        "reset_num_timesteps": reset_num_timesteps,
    }
    if log_interval is not None:
        learn_kwargs["log_interval"] = log_interval
    model.learn(**learn_kwargs)
