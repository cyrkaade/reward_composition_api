"""Command-line interface.

The ``train``, ``sweep``, and ``summarize`` flags are derived programmatically
from the config dataclass fields (name, type, default, and a small metadata
dict for choices/help), so every knob is declared exactly once in
``rcomp.config``.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import sys
import types
import typing
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    MUJOCO_SUITE,
    TRAIN_SUITES,
    ExperimentConfig,
    RewardCompositionError,
    SummaryConfig,
    SweepConfig,
    suite_supported_envs,
)
from .partials import PartialRegistry, load_partial_reference, scan_partials
from .suites import SUITE_NAMES, get_suite


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "train": _handle_train,
        "sweep": _handle_sweep,
        "summarize": _handle_summarize,
        "list-envs": _handle_list_envs,
        "list-partials": _handle_list_partials,
        "validate-partial": _handle_validate_partial,
        "partiality": _handle_partiality,
        "plot-partiality": _handle_plot_partiality,
    }
    try:
        return handlers[args.command](args)
    except RewardCompositionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rcomp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run one experiment")
    add_config_arguments(train_parser, ExperimentConfig)

    sweep_parser = subparsers.add_parser("sweep", help="Plan or execute ablation-style sweeps")
    add_config_arguments(sweep_parser, SweepConfig)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize run metadata into CSV files")
    add_config_arguments(summarize_parser, SummaryConfig)

    list_envs_parser = subparsers.add_parser("list-envs", help="List supported environments")
    list_envs_parser.add_argument("--suite", choices=SUITE_NAMES, default=None)

    list_partials_parser = subparsers.add_parser("list-partials", help="List partial rewards found in the partials folders")
    list_partials_parser.add_argument("--suite", choices=SUITE_NAMES, default=None)

    validate_parser = subparsers.add_parser("validate-partial", help="Import and smoke-check a partial reward")
    validate_parser.add_argument("--suite", choices=SUITE_NAMES, default=MUJOCO_SUITE)
    validate_parser.add_argument("--env-id", default=None)
    validate_parser.add_argument("--partial", required=True)

    partiality_parser = subparsers.add_parser("partiality", help="Estimate how much a partial reward matches true reward")
    partiality_parser.add_argument("--suite", choices=SUITE_NAMES, default=MUJOCO_SUITE)
    partiality_parser.add_argument("--env-id", required=True)
    partiality_parser.add_argument("--partial", required=True)
    partiality_parser.add_argument("--timesteps", type=int, default=100_000)
    partiality_parser.add_argument("--fragment-length", type=int, default=25)
    partiality_parser.add_argument("--seed", type=int, default=0)
    partiality_parser.add_argument("--policy", choices=["random", "trained", "mix"], default="random",
                                   help="Rollout policy for the estimate: random (default), trained (PPO on true reward), or mix")
    partiality_parser.add_argument("--policy-timesteps", type=int, default=300_000,
                                   help="PPO timesteps when --policy is trained/mix")
    partiality_parser.add_argument("--output", default=None)
    partiality_parser.add_argument("--no-save", action="store_true")

    partiality_plot_parser = subparsers.add_parser("plot-partiality", help="Plot final reward by partiality and query budget")
    partiality_plot_parser.add_argument("--runs-root", default="logs")
    partiality_plot_parser.add_argument("--partiality-root", default=str(Path("logs") / "partiality"))
    partiality_plot_parser.add_argument("--output", default=str(Path("logs") / "partiality" / "partiality_grid.png"))
    partiality_plot_parser.add_argument("--env-id", default=None)
    partiality_plot_parser.add_argument("--title", default="Partiality vs RLHF queries")

    return parser


def add_config_arguments(parser: argparse.ArgumentParser, config_cls: type) -> None:
    hints = typing.get_type_hints(config_cls)
    for config_field in dataclasses.fields(config_cls):
        _add_field_argument(parser, config_field, hints[config_field.name])


def config_from_args(config_cls: type, args: argparse.Namespace):
    hints = typing.get_type_hints(config_cls)
    kwargs = {}
    for config_field in dataclasses.fields(config_cls):
        value = getattr(args, config_field.name)
        inner_types, _ = _unwrap_optional(hints[config_field.name])
        if isinstance(value, list) and any(typing.get_origin(inner) is tuple for inner in inner_types):
            value = tuple(value)
        kwargs[config_field.name] = value
    return config_cls(**kwargs)


def _add_field_argument(parser: argparse.ArgumentParser, config_field: dataclasses.Field, hint) -> None:
    meta = config_field.metadata
    name = config_field.name
    flag = "--" + name.replace("_", "-")
    inner_types, optional = _unwrap_optional(hint)
    default = config_field.default
    kwargs: dict[str, Any] = {"dest": name, "default": default, "help": meta.get("help", "")}
    if "choices" in meta:
        kwargs["choices"] = list(meta["choices"])

    if bool in inner_types:
        parser.add_argument(flag, action="store_true", dest=name, default=default, help=meta.get("help", ""))
        if optional or default is True:
            parser.add_argument(f"--no-{name.replace('_', '-')}", action="store_false", dest=name, help=f"Disable {flag}")
        return

    tuple_types = [inner for inner in inner_types if typing.get_origin(inner) is tuple]
    if tuple_types:
        item_type = typing.get_args(tuple_types[0])[0]
        if meta.get("nargs"):
            parser.add_argument(flag, nargs=meta["nargs"], type=item_type, **kwargs)
        else:
            parser.add_argument(flag, type=parse_int_tuple, **kwargs)
        return

    if any(typing.get_origin(inner) is dict for inner in inner_types):
        parser.add_argument(flag, type=parse_key_value_mapping, **kwargs)
        return

    if float in inner_types:
        parser.add_argument(flag, type=parse_optional_float if optional else float, **kwargs)
        return

    if int in inner_types:
        parser.add_argument(flag, type=int, **kwargs)
        return

    parser.add_argument(flag, type=str, **kwargs)


def _unwrap_optional(hint) -> tuple[list, bool]:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = list(typing.get_args(hint))
        inner = [arg for arg in args if arg is not type(None)]
        return inner, type(None) in args
    return [hint], False


def _handle_train(args) -> int:
    from .trainer import run_experiment

    run_experiment(config_from_args(ExperimentConfig, args))
    return 0


def _handle_sweep(args) -> int:
    from .sweeps import run_sweep

    run_sweep(config_from_args(SweepConfig, args))
    return 0


def _handle_summarize(args) -> int:
    from .sweeps import summarize_runs

    summarize_runs(config_from_args(SummaryConfig, args))
    return 0


def _handle_list_envs(args) -> int:
    suites = [args.suite] if args.suite else list(TRAIN_SUITES)
    for suite in suites:
        print(f"{suite}:")
        for env_id in suite_supported_envs(suite):
            print(f"  {env_id}")
    return 0


def _handle_list_partials(args) -> int:
    specs = scan_partials(args.suite)
    for spec in specs:
        envs = f" [{', '.join(spec.env_ids)}]" if spec.env_ids else ""
        print(f"{spec.suite}/{spec.name}{envs}: {spec.description}")
    return 0


def _handle_validate_partial(args) -> int:
    env_id = args.env_id or get_suite(args.suite).default_env_id()
    partial_spec = load_partial_reference(args.partial, args.suite, PartialRegistry())
    partial = partial_spec.create(env_id)
    info = {"reward_dist": 1.0, "reward_ctrl": -0.5, "lives": 3}
    partial.reset(info)
    step = partial.step(
        obs=np.zeros(4, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        next_obs=np.ones(4, dtype=np.float32),
        true_reward=1.0,
        terminated=False,
        truncated=False,
        info=info,
    )
    print(f"partial '{partial_spec.suite}/{partial_spec.name}' OK for {env_id}: partial={step.partial}")
    return 0


def _handle_partiality(args) -> int:
    from .partiality import PartialityConfig, estimate_partiality, partiality_json, save_partiality_result

    metrics = estimate_partiality(
        PartialityConfig(
            suite=args.suite,
            env_id=args.env_id,
            partial=args.partial,
            timesteps=args.timesteps,
            fragment_length=args.fragment_length,
            seed=args.seed,
            policy=args.policy,
            policy_timesteps=args.policy_timesteps,
        )
    )
    if not args.no_save:
        output_path = save_partiality_result(metrics, args.output)
        print(f"saved partiality result to {output_path}", file=sys.stderr)
    print(partiality_json(metrics))
    return 0


def _handle_plot_partiality(args) -> int:
    from .partiality import PartialityGridConfig, plot_partiality_grid

    output_path = plot_partiality_grid(
        PartialityGridConfig(
            runs_root=args.runs_root,
            partiality_root=args.partiality_root,
            output=args.output,
            env_id=args.env_id,
            title=args.title,
        )
    )
    print(f"saved partiality grid to {output_path}")
    return 0


def parse_int_tuple(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)

    text = str(value).strip()
    if not text:
        return ()
    text = text.replace("x", ",")
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if any(item <= 0 for item in values):
        raise ValueError("sizes must be positive integers")
    return values


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", "off"}:
        return None
    return float(value)


def parse_key_value_mapping(value: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None or isinstance(value, dict):
        return value

    text = value.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _parse_loose_mapping(text)

    if not isinstance(parsed, dict):
        raise ValueError("expected a mapping")
    return {str(key): item for key, item in parsed.items()}


def _parse_loose_mapping(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    if not text:
        return {}

    result: dict[str, Any] = {}
    for item in _split_top_level(text, ","):
        key, raw_value = _split_key_value(item)
        result[_clean_key(key)] = _parse_scalar(raw_value)
    return result


def _split_key_value(item: str) -> tuple[str, str]:
    for separator in (":", "="):
        parts = _split_top_level(item, separator, maxsplit=1)
        if len(parts) == 2:
            return parts[0], parts[1]
    raise ValueError(f"expected key:value in '{item}'")


def _split_top_level(text: str, separator: str, maxsplit: int | None = None) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    splits = 0

    for index, char in enumerate(text):
        if quote:
            if char == quote and text[index - 1 : index] != "\\":
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth -= 1
            continue
        if char == separator and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
            splits += 1
            if maxsplit is not None and splits >= maxsplit:
                break

    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _clean_key(value: str) -> str:
    return value.strip().strip("'\"")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value.strip("'\"")
