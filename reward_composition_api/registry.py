from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .errors import PartialRegistryError


@dataclass(frozen=True)
class PartialRewardStep:
    partial: float
    components: dict[str, float] = field(default_factory=dict)


class CallablePartialReward:
    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn

    def reset(self, info: dict | None = None) -> None:
        reset = getattr(self.fn, "reset", None)
        if callable(reset):
            reset(info)

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        value = self.fn(
            obs=obs,
            action=action,
            next_obs=next_obs,
            true_reward=true_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return coerce_partial_step(value)


@dataclass(frozen=True)
class PartialSpec:
    name: str
    suite: str
    factory: Callable[[str], Any]
    description: str = ""
    env_ids: tuple[str, ...] = ()
    component_keys: tuple[str, ...] = ()

    def create(self, env_id: str):
        if self.env_ids and env_id not in self.env_ids:
            supported = ", ".join(self.env_ids)
            raise PartialRegistryError(f"Partial '{self.name}' does not support env '{env_id}'. Supported envs: {supported}")
        instance = self.factory(env_id)
        if hasattr(instance, "step"):
            return instance
        if callable(instance):
            return CallablePartialReward(instance)
        raise PartialRegistryError(f"Partial '{self.name}' factory must return a callable or object with step()")


class PartialRegistry:
    def __init__(self):
        self._partials: dict[tuple[str, str], PartialSpec] = {}

    def register(
        self,
        name: str,
        suite: str,
        factory: Callable[[str], Any] | Callable[[], Any],
        description: str = "",
        env_ids: tuple[str, ...] | list[str] = (),
        component_keys: tuple[str, ...] | list[str] = (),
    ) -> PartialSpec:
        if not name or "/" in name:
            raise PartialRegistryError("partial names must be non-empty and must not contain '/'")
        if not callable(factory):
            raise PartialRegistryError(f"factory for partial '{name}' is not callable")

        normalized_factory = _normalize_factory(factory)
        spec = PartialSpec(
            name=name,
            suite=suite,
            factory=normalized_factory,
            description=description,
            env_ids=tuple(env_ids),
            component_keys=tuple(component_keys),
        )
        key = (suite, name)
        if key in self._partials:
            raise PartialRegistryError(f"Partial '{suite}/{name}' is already registered")
        self._partials[key] = spec
        return spec

    def resolve(self, reference: str, suite: str) -> PartialSpec:
        if "/" in reference:
            ref_suite, name = reference.split("/", 1)
        else:
            ref_suite, name = suite, reference
        key = (ref_suite, name)
        try:
            return self._partials[key]
        except KeyError as exc:
            choices = ", ".join(f"{item.suite}/{item.name}" for item in self.list())
            raise PartialRegistryError(f"Unknown partial '{reference}'. Available partials: {choices}") from exc

    def list(self, suite: str | None = None) -> list[PartialSpec]:
        values = list(self._partials.values())
        if suite is not None:
            values = [value for value in values if value.suite == suite]
        return sorted(values, key=lambda item: (item.suite, item.name))


def coerce_partial_step(value: Any) -> PartialRewardStep:
    if isinstance(value, PartialRewardStep):
        return value
    if isinstance(value, (int, float)):
        return PartialRewardStep(partial=float(value))
    if isinstance(value, dict):
        if "partial" not in value:
            raise PartialRegistryError("partial result dictionaries must contain a 'partial' key")
        components = value.get("components", {})
        if components is None:
            components = {}
        if not isinstance(components, dict):
            raise PartialRegistryError("'components' must be a dictionary when provided")
        return PartialRewardStep(
            partial=float(value["partial"]),
            components={str(key): float(component_value) for key, component_value in components.items()},
        )
    raise PartialRegistryError(f"Unsupported partial return type: {type(value).__name__}")


def load_user_partial_module(
    module_ref: str,
    registry: PartialRegistry,
    default_name: str | None = None,
    default_suite: str = "custom",
) -> ModuleType:
    module = _import_module(module_ref)
    if hasattr(module, "register"):
        module.register(registry)
    elif hasattr(module, "PARTIALS"):
        partials = getattr(module, "PARTIALS")
        if not isinstance(partials, dict):
            raise PartialRegistryError("PARTIALS must be a dictionary")
        for name, value in partials.items():
            if isinstance(value, dict):
                registry.register(
                    name=name,
                    suite=value.get("suite", "custom"),
                    factory=value["factory"],
                    description=value.get("description", ""),
                    env_ids=tuple(value.get("env_ids", ())),
                    component_keys=tuple(value.get("component_keys", ())),
                )
            else:
                registry.register(name=name, suite="custom", factory=value)
    elif hasattr(module, "PartialReward"):
        name = default_name or Path(module_ref).stem
        registry.register(name=name, suite=default_suite, factory=getattr(module, "PartialReward"))
    elif hasattr(module, "partial_reward"):
        name = default_name or Path(module_ref).stem
        registry.register(name=name, suite=default_suite, factory=lambda env_id: getattr(module, "partial_reward"))
    elif hasattr(module, "partial"):
        name = default_name or Path(module_ref).stem
        registry.register(name=name, suite=default_suite, factory=lambda env_id: getattr(module, "partial"))
    else:
        raise PartialRegistryError(
            "User partial modules must define register(registry), PARTIALS, "
            "PartialReward, partial_reward, or partial"
        )
    return module


def load_partial_reference(reference: str, suite: str, registry: PartialRegistry) -> PartialSpec:
    if ":" not in reference:
        try:
            return registry.resolve(reference, suite)
        except PartialRegistryError:
            module_ref = _resolve_partials_module(reference)
            load_user_partial_module(module_ref, registry, default_name=Path(reference).stem, default_suite=suite)
            try:
                return registry.resolve(Path(reference).stem, suite)
            except PartialRegistryError:
                return registry.resolve(Path(reference).stem, "custom")
    module_ref, partial_name = reference.rsplit(":", 1)
    if not module_ref or not partial_name:
        raise PartialRegistryError("Partial references must be '<module>:<partial_name>' or a built-in name")
    module_ref = _resolve_partials_module(module_ref)
    load_user_partial_module(module_ref, registry, default_name=partial_name, default_suite=suite)
    try:
        return registry.resolve(partial_name, suite)
    except PartialRegistryError:
        return registry.resolve(partial_name, "custom")


def _normalize_factory(factory: Callable[..., Any]) -> Callable[[str], Any]:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return lambda env_id: factory(env_id)

    required_positional = [
        param
        for param in signature.parameters.values()
        if param.default is inspect._empty
        and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_env_id_kw = "env_id" in signature.parameters
    if len(required_positional) == 0 and not has_env_id_kw:
        return lambda env_id: factory()
    return lambda env_id: factory(env_id)


def _import_module(module_ref: str) -> ModuleType:
    path = Path(module_ref)
    if path.suffix == ".py" or path.exists():
        if not path.exists():
            raise PartialRegistryError(f"Partial module file does not exist: {module_ref}")
        module_name = f"_reward_composition_user_partial_{abs(hash(path.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise PartialRegistryError(f"Could not import partial module file: {module_ref}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module(module_ref)
    except Exception as exc:
        raise PartialRegistryError(f"Could not import partial module '{module_ref}': {exc}") from exc


def _resolve_partials_module(module_ref: str) -> str:
    path = Path(module_ref)
    candidates = []
    if path.suffix == ".py" or path.exists():
        candidates.append(path)
    else:
        candidates.extend(
            [
                Path("partials") / f"{module_ref}.py",
                Path("partials") / module_ref,
                path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return module_ref
