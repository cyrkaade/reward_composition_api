from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from rcomp.cli import main as cli_main
from rcomp.partials import (
    PartialRegistry,
    PartialRegistryError,
    coerce_partial_step,
    load_partial_reference,
    scan_partials,
)


def write_module(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_partial_reward_function_style(tmp_path):
    path = write_module(
        tmp_path,
        "fn_style",
        "def partial_reward(obs, action, next_obs, true_reward, terminated, truncated, info):\n"
        "    return {'partial': 2.0, 'components': {'bonus': 2.0}}\n",
    )

    spec = load_partial_reference(str(path), "gym", PartialRegistry())
    partial = spec.create("CartPole-v1")
    partial.reset({})
    step = partial.step(None, 0, None, 1.0, False, False, {})

    assert spec.name == "fn_style"
    assert spec.suite == "gym"
    assert step.partial == 2.0
    assert step.components == {"bonus": 2.0}


def test_partial_object_style(tmp_path):
    path = write_module(
        tmp_path,
        "obj_style",
        "def partial(**kwargs):\n"
        "    return 0.5 * kwargs['true_reward']\n",
    )

    spec = load_partial_reference(str(path), "gym", PartialRegistry())
    step = spec.create("CartPole-v1").step(None, 0, None, 4.0, False, False, {})

    assert step.partial == 2.0
    assert step.components == {}


def test_partial_reward_class_style(tmp_path):
    path = write_module(
        tmp_path,
        "class_style",
        "class PartialReward:\n"
        "    def reset(self, info):\n"
        "        self.count = 0\n"
        "    def step(self, obs, action, next_obs, true_reward, terminated, truncated, info):\n"
        "        self.count += 1\n"
        "        return {'partial': float(self.count), 'components': {'count': float(self.count)}}\n",
    )

    spec = load_partial_reference(str(path), "mujoco", PartialRegistry())
    partial = spec.create("Reacher-v5")
    partial.reset({})

    assert partial.step(None, None, None, 0.0, False, False, {}).partial == 1.0
    assert partial.step(None, None, None, 0.0, False, False, {}).partial == 2.0


def test_partials_dict_style(tmp_path):
    path = write_module(
        tmp_path,
        "dict_style",
        "def half(**kwargs):\n"
        "    return 0.5 * kwargs['true_reward']\n"
        "def full(**kwargs):\n"
        "    return kwargs['true_reward']\n"
        "PARTIALS = {\n"
        "    'half': lambda env_id: half,\n"
        "    'full': {'factory': lambda env_id: full, 'suite': 'gym', 'description': 'full reward', 'env_ids': ('CartPole-v1',)},\n"
        "}\n",
    )

    registry = PartialRegistry()
    spec = load_partial_reference(f"{path}:half", "gym", registry)
    assert spec.create("CartPole-v1").step(None, 0, None, 6.0, False, False, {}).partial == 3.0

    full_spec = registry.resolve("full", "gym")
    assert full_spec.description == "full reward"
    with pytest.raises(PartialRegistryError, match="does not support env"):
        full_spec.create("MountainCar-v0")


def test_register_style_with_named_reference(tmp_path):
    path = write_module(
        tmp_path,
        "reg_style",
        "def constant(**kwargs):\n"
        "    return {'partial': 3.5, 'components': {'constant': 3.5}}\n"
        "def register(registry):\n"
        "    registry.register(name='constant', suite='custom', factory=lambda env_id: constant, component_keys=('constant',))\n",
    )

    spec = load_partial_reference(f"{path}:constant", "mujoco", PartialRegistry())
    step = spec.create("Anything-v0").step(None, None, None, 0.0, False, False, {})

    assert spec.name == "constant"
    assert spec.component_keys == ("constant",)
    assert step.partial == 3.5


def test_packaged_partials_folder_shorthand():
    spec = load_partial_reference("example_cartpole", "gym", PartialRegistry())
    step = spec.create("CartPole-v1").step(None, 0, None, 1.0, False, False, {})

    assert spec.name == "example_cartpole"
    assert step.partial == 1.0
    assert step.components["alive_bonus"] == 1.0


def test_missing_module_and_invalid_module(tmp_path):
    with pytest.raises(PartialRegistryError):
        load_partial_reference(str(tmp_path / "nope.py"), "gym", PartialRegistry())

    path = write_module(tmp_path, "empty_style", "x = 1\n")
    with pytest.raises(PartialRegistryError, match="must define"):
        load_partial_reference(str(path), "gym", PartialRegistry())


def test_coercion_rules():
    assert coerce_partial_step(1.5).partial == 1.5
    assert coerce_partial_step({"partial": 2, "components": {"a": 1}}).components == {"a": 1.0}
    assert coerce_partial_step({"partial": 2, "components": None}).components == {}

    with pytest.raises(PartialRegistryError, match="'partial' key"):
        coerce_partial_step({"components": {}})
    with pytest.raises(PartialRegistryError, match="dictionary"):
        coerce_partial_step({"partial": 1.0, "components": [1, 2]})
    with pytest.raises(PartialRegistryError, match="Unsupported partial return type"):
        coerce_partial_step("nope")


def test_registry_rejects_duplicates_and_bad_names():
    registry = PartialRegistry()
    registry.register(name="a", suite="gym", factory=lambda env_id: lambda **kwargs: 0.0)

    with pytest.raises(PartialRegistryError, match="already registered"):
        registry.register(name="a", suite="gym", factory=lambda env_id: lambda **kwargs: 0.0)
    with pytest.raises(PartialRegistryError, match="must not contain"):
        registry.register(name="a/b", suite="gym", factory=lambda env_id: lambda **kwargs: 0.0)
    with pytest.raises(PartialRegistryError, match="Unknown partial"):
        registry.resolve("missing", "gym")


def test_scan_partials_finds_packaged_example():
    specs = scan_partials()
    assert any(spec.name == "example_cartpole" for spec in specs)


def test_validate_partial_cli_round_trip():
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = cli_main(["validate-partial", "--suite", "gym", "--env-id", "CartPole-v1", "--partial", "example_cartpole"])

    assert exit_code == 0
    assert "example_cartpole" in stdout.getvalue()
    assert "OK" in stdout.getvalue()
