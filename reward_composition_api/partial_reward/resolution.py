from __future__ import annotations

from reward_composition_api.registry import PartialSpec, load_partial_reference

from .builtins import build_builtin_registry


def resolve_custom_partial(config) -> PartialSpec | None:
    if not config.partial:
        return None
    registry = build_builtin_registry()
    return load_partial_reference(config.partial, config.suite, registry)


def include_partial_feature(config) -> bool:
    if config.include_partial_feature is not None:
        return bool(config.include_partial_feature)
    return config.mode in {"naive", "delta"}
