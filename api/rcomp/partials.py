"""Loading and validation of human-written partial rewards.

Partial rewards are always plain Python files in a ``partials/`` folder. A
file may define, in order of precedence:

- ``register(registry)`` — register any number of named partials;
- ``PARTIALS`` — a dict of name -> factory (or name -> spec dict);
- ``PartialReward`` — a class instantiated per run;
- ``partial_reward`` — a function called once per step;
- ``partial`` — a callable or object with ``step(...)``.

Step callables receive ``(obs, action, next_obs, true_reward, terminated,
truncated, info)`` as keyword arguments and return either a float or
``{"partial": float, "components": {name: float}}``. Objects may also expose
``reset(info)``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .config import RewardCompositionError


class PartialRegistryError(RewardCompositionError):
    pass


@dataclass(frozen=True)
class PartialRewardStep:
    partial: float
    components: dict[str, float] = field(default_factory=dict)


class ObjectPartialReward:
    """Adapter for object-style partials: delegates reset()/step() and
    coerces the step result (PartialRewardStep, dict, or float)."""

    def __init__(self, obj: Any):
        self.obj = obj

    def reset(self, info: dict | None = None) -> None:
        reset = getattr(self.obj, "reset", None)
        if callable(reset):
            reset(info)

    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info) -> PartialRewardStep:
        value = self.obj.step(obs, action, next_obs, true_reward, terminated, truncated, info)
        return coerce_partial_step(value)


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
            return ObjectPartialReward(instance)
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

        spec = PartialSpec(
            name=name,
            suite=suite,
            factory=_normalize_factory(factory),
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
    if hasattr(value, "partial"):
        components = getattr(value, "components", {}) or {}
        return PartialRewardStep(
            partial=float(value.partial),
            components={str(key): float(component_value) for key, component_value in dict(components).items()},
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
    module_ref, partial_name = _split_reference(reference)
    if partial_name is None:
        try:
            return registry.resolve(reference, suite)
        except PartialRegistryError:
            module_ref = _resolve_partials_module(reference)
            load_user_partial_module(module_ref, registry, default_name=Path(reference).stem, default_suite=suite)
            try:
                return registry.resolve(Path(reference).stem, suite)
            except PartialRegistryError:
                return registry.resolve(Path(reference).stem, "custom")
    if not module_ref or not partial_name:
        raise PartialRegistryError("Partial references must be '<module>:<partial_name>' or a partial module name")
    module_ref = _resolve_partials_module(module_ref)
    load_user_partial_module(module_ref, registry, default_name=partial_name, default_suite=suite)
    try:
        return registry.resolve(partial_name, suite)
    except PartialRegistryError:
        return registry.resolve(partial_name, "custom")


def resolve_custom_partial(config) -> PartialSpec | None:
    if not config.partial:
        return None
    return load_partial_reference(config.partial, config.suite, PartialRegistry())


def include_partial_feature(config) -> bool:
    if config.include_partial_feature is not None:
        return bool(config.include_partial_feature)
    return config.mode in {"naive", "delta"}


def partials_search_roots() -> list[Path]:
    """Folders searched for partial files: ./partials, then the packaged one."""
    roots = [Path("partials"), Path(__file__).resolve().parent.parent / "partials"]
    unique = []
    for root in roots:
        if root.resolve() not in [item.resolve() for item in unique]:
            unique.append(root)
    return unique


def scan_partials(suite: str | None = None) -> list[PartialSpec]:
    """Load every partial file found in the partials folders and list its specs."""
    specs: dict[tuple[str, str], PartialSpec] = {}
    for root in partials_search_roots():
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.py")):
            if path.name.startswith("_"):
                continue
            registry = PartialRegistry()
            try:
                load_user_partial_module(str(path), registry, default_name=path.stem)
            except Exception as exc:
                print(f"warning: skipping {path}: {exc}")
                continue
            for spec in registry.list():
                specs.setdefault((spec.suite, spec.name), spec)
    values = list(specs.values())
    if suite is not None:
        values = [spec for spec in values if spec.suite in (suite, "custom")]
    return sorted(values, key=lambda item: (item.suite, item.name))


def _split_reference(reference: str) -> tuple[str, str | None]:
    """Split '<module>:<name>' references; a Windows drive-letter colon
    (e.g. 'C:/partials/foo.py') does not count as a name separator."""
    if ":" in reference:
        module_ref, partial_name = reference.rsplit(":", 1)
        if module_ref and partial_name and not any(sep in partial_name for sep in "/\\"):
            return module_ref, partial_name
    return reference, None


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
        module_name = f"_rcomp_user_partial_{abs(hash(path.resolve()))}"
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
        for root in partials_search_roots():
            candidates.append(root / f"{module_ref}.py")
            candidates.append(root / module_ref)
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return module_ref
