"""Example partial reward for CartPole-v1.

Partial rewards are always written manually, as plain Python files in this
``partials/`` folder. This file is the simplest possible template: a single
``partial_reward`` function. Reference it on the CLI as

    rcomp train --suite gym --env-id CartPole-v1 --mode delta --partial example_cartpole

The function is called once per environment step with keyword arguments and
returns either a plain float, or a dict with the partial value plus named
components (components show up in eval CSVs and metadata):

    {"partial": <float>, "components": {<name>: <float>, ...}}

Other supported file styles (see the README for full examples):
  * ``partial`` — a callable or an object with ``step(...)`` and optional
    ``reset(info)`` for stateful partials;
  * ``PartialReward`` — a class, instantiated once per run;
  * ``PARTIALS`` — a dict of name -> factory for several partials in one file;
  * ``register(registry)`` — full control, including per-env restrictions,
    referenced as ``--partial <file>:<name>``.
"""


def partial_reward(obs, action, next_obs, true_reward, terminated, truncated, info):
    # CartPole's true reward is +1 per step until the pole falls. This partial
    # captures only the survival component.
    alive_bonus = 1.0 if not terminated else 0.0
    return {
        "partial": alive_bonus,
        "components": {"alive_bonus": alive_bonus},
    }
