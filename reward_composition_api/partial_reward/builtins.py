from __future__ import annotations

from reward_composition_api.registry import PartialRegistry


def build_builtin_registry() -> PartialRegistry:
    return PartialRegistry()


def partials_for_display(suite: str | None = None) -> list[dict[str, str]]:
    registry = build_builtin_registry()
    return [
        {
            "suite": spec.suite,
            "name": spec.name,
            "envs": ", ".join(spec.env_ids),
            "description": spec.description,
        }
        for spec in registry.list(suite)
    ]
