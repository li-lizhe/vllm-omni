# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression tests for per-stage engine-args reconstruction (issue #8816).

A stage's ``engine_args`` reach ``OmniEngineArgs`` as a plain dict.  Two nested
configs cannot survive that round trip unchanged:

* a plain ``structured_outputs_config`` breaks ``create_engine_config()``;
* a plain ``fault_tolerance_config`` is read by ``EngineArgs.__post_init__`` as
  an explicit ``--fault-tolerance-config`` and silently enables fault tolerance
  for a stage that never asked for it.

The second one needs provenance, not a type check: upstream's auto-enable is
intended for a dict the user wrote, and such a dict may legitimately equal the
default.  ``_to_dict`` therefore records that it flattened a typed config, and
only that case is rebuilt.
"""

from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import pytest
from vllm.config import FaultToleranceConfig, StructuredOutputsConfig
from vllm.engine.arg_utils import EngineArgs

from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import restore_serialized_engine_args
from vllm_omni.entrypoints.stage_utils import FT_CONFIG_WAS_TYPED_KEY, _to_dict

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEFAULT_TIMEOUT = FaultToleranceConfig().engine_recovery_timeout_sec


def test_upstream_still_auto_enables_fault_tolerance_for_a_plain_dict():
    """Pin the upstream behaviour the stage-init path has to stay clear of."""
    args = EngineArgs(fault_tolerance_config=asdict(FaultToleranceConfig()))

    assert args.enable_fault_tolerance is True


def test_to_dict_marks_a_flattened_fault_tolerance_config():
    args = OmniEngineArgs(
        model="dummy-model",
        fault_tolerance_config=FaultToleranceConfig(engine_recovery_timeout_sec=180),
    )

    serialized = _to_dict(args)

    assert serialized[FT_CONFIG_WAS_TYPED_KEY] is True
    assert isinstance(serialized["fault_tolerance_config"], dict)
    assert serialized["enable_fault_tolerance"] is False


def test_to_dict_does_not_mark_a_plain_dict():
    """A caller-written dict is an explicit config, not provenance."""
    explicit = {"fault_tolerance_config": {"engine_recovery_timeout_sec": 180}}

    assert FT_CONFIG_WAS_TYPED_KEY not in _to_dict(explicit)


def test_restore_keeps_an_untouched_default_off_the_fault_tolerance_path():
    restored = restore_serialized_engine_args(
        {
            "fault_tolerance_config": asdict(FaultToleranceConfig()),
            "enable_fault_tolerance": False,
        },
        ft_config_was_typed=True,
    )

    assert isinstance(restored["fault_tolerance_config"], FaultToleranceConfig)
    assert restored["fault_tolerance_config"].engine_recovery_timeout_sec == DEFAULT_TIMEOUT
    assert EngineArgs(**restored).enable_fault_tolerance is False


def test_restore_keeps_a_custom_serialized_timeout():
    restored = restore_serialized_engine_args(
        {
            "fault_tolerance_config": {"engine_recovery_timeout_sec": 180},
            "enable_fault_tolerance": False,
        },
        ft_config_was_typed=True,
    )

    assert restored["fault_tolerance_config"].engine_recovery_timeout_sec == 180
    assert EngineArgs(**restored).enable_fault_tolerance is False


def test_restore_leaves_an_explicit_dict_alone():
    """No provenance: a dict keeps upstream's auto-enable semantics."""
    restored = restore_serialized_engine_args(
        {
            "fault_tolerance_config": {"engine_recovery_timeout_sec": 30},
            "enable_fault_tolerance": False,
        }
    )

    assert isinstance(restored["fault_tolerance_config"], dict)
    assert EngineArgs(**restored).enable_fault_tolerance is True


def test_restore_leaves_a_default_valued_explicit_dict_alone():
    """Equality with the default cannot establish provenance."""
    restored = restore_serialized_engine_args(
        {
            "fault_tolerance_config": {"engine_recovery_timeout_sec": DEFAULT_TIMEOUT},
            "enable_fault_tolerance": False,
        }
    )

    assert isinstance(restored["fault_tolerance_config"], dict)
    assert EngineArgs(**restored).enable_fault_tolerance is True


def test_restore_ignores_provenance_when_there_is_no_dict():
    restored = restore_serialized_engine_args(
        {"enable_fault_tolerance": False},
        ft_config_was_typed=True,
    )

    assert "fault_tolerance_config" not in restored
    assert EngineArgs(**restored).enable_fault_tolerance is False


def test_restore_rebuilds_a_serialized_structured_outputs_config():
    restored = restore_serialized_engine_args({"structured_outputs_config": {"reasoning_parser": "deepseek_r1"}})

    assert isinstance(restored["structured_outputs_config"], StructuredOutputsConfig)
    assert restored["structured_outputs_config"].reasoning_parser == "deepseek_r1"


def test_restore_ignores_values_that_are_already_dataclasses():
    config = FaultToleranceConfig(engine_recovery_timeout_sec=30)

    restored = restore_serialized_engine_args(
        {"fault_tolerance_config": config},
        ft_config_was_typed=True,
    )

    assert restored["fault_tolerance_config"] is config


def _captured_engine_args(engine_args_dict, model="dummy-model"):
    """Run ``build_vllm_config`` and report what reached ``create_engine_config``."""
    stage = SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args=engine_args_dict,
        runtime=SimpleNamespace(devices=None),
    )
    sentinel_executor = object()
    captured = {}

    def create_config(args, *unused_args, **unused_kwargs):
        captured["enabled"] = args.enable_fault_tolerance
        captured["config"] = args.fault_tolerance_config
        return SimpleNamespace(quant_config=None)

    with (
        mock.patch.object(
            stage_init_utils.OmniEngineArgs,
            "create_engine_config",
            autospec=True,
            side_effect=create_config,
        ),
        mock.patch.object(stage_init_utils.Executor, "get_class", return_value=sentinel_executor),
        mock.patch.object(stage_init_utils.OmniINCConfig, "maybe_upgrade", return_value=None),
    ):
        stage_init_utils.build_vllm_config(stage, model=model, engine_args_dict=engine_args_dict)

    return captured


def test_build_vllm_config_keeps_a_serialized_default_off_the_fault_tolerance_path():
    """The #8816 regression: a stage that never opted in must not be opted in."""
    serialized = _to_dict(OmniEngineArgs(model="dummy-model"))
    serialized = {
        key: serialized[key] for key in ("enable_fault_tolerance", "fault_tolerance_config") if key in serialized
    }
    serialized[FT_CONFIG_WAS_TYPED_KEY] = True

    captured = _captured_engine_args(serialized)

    assert captured["enabled"] is False
    assert isinstance(captured["config"], FaultToleranceConfig)
    assert captured["config"].engine_recovery_timeout_sec == DEFAULT_TIMEOUT


def test_build_vllm_config_leaves_a_plain_dict_on_the_auto_enable_path():
    """An explicit dict is still an explicit request for fault tolerance."""
    captured = _captured_engine_args({"fault_tolerance_config": {"engine_recovery_timeout_sec": 180}})

    assert captured["enabled"] is True
    assert captured["config"].engine_recovery_timeout_sec == 180


def test_build_vllm_config_does_not_leak_the_marker_into_the_engine_args():
    """The transport marker is consumed at the boundary, never on the dataclass."""
    seen = {}

    def create_config(args, *unused_args, **unused_kwargs):
        seen["fields"] = set(vars(args))
        return SimpleNamespace(quant_config=None)

    with (
        mock.patch.object(
            stage_init_utils.OmniEngineArgs,
            "create_engine_config",
            autospec=True,
            side_effect=create_config,
        ),
        mock.patch.object(stage_init_utils.Executor, "get_class", return_value=object()),
        mock.patch.object(stage_init_utils.OmniINCConfig, "maybe_upgrade", return_value=None),
    ):
        stage_init_utils.build_vllm_config(
            SimpleNamespace(
                stage_id=0,
                stage_type="llm",
                engine_args={FT_CONFIG_WAS_TYPED_KEY: True, "enable_fault_tolerance": False},
                runtime=SimpleNamespace(devices=None),
            ),
            model="dummy-model",
            engine_args_dict={FT_CONFIG_WAS_TYPED_KEY: True, "enable_fault_tolerance": False},
        )

    assert FT_CONFIG_WAS_TYPED_KEY not in seen["fields"]


def test_a_round_trip_through_to_dict_stays_off_the_fault_tolerance_path():
    """The end-to-end contract on the keys this boundary actually transports.

    A full ``asdict`` payload cannot be fed back as-is — upstream rejects a
    serialized ``CompilationConfig`` before the FT branch is ever reached — so
    only the fault-tolerance keys travel, exactly as the stage boundary does.
    """
    serialized = _to_dict(OmniEngineArgs(model="dummy-model"))
    marker = serialized.pop(FT_CONFIG_WAS_TYPED_KEY)
    transported = {
        key: serialized[key] for key in ("enable_fault_tolerance", "fault_tolerance_config") if key in serialized
    }

    restored = restore_serialized_engine_args(transported, ft_config_was_typed=marker)

    assert marker is True
    assert EngineArgs(**restored).enable_fault_tolerance is False


def test_legacy_stage_builder_keeps_provenance_for_a_typed_stage():
    """The #8816 reproduction: a typed engine_args object reaches the stage path."""
    stage = SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args=OmniEngineArgs(model="dummy-model"),
        runtime=SimpleNamespace(devices=None),
    )
    with mock.patch.object(
        stage_init_utils,
        "_finalize_engine_args_dict",
        side_effect=lambda engine_args, **unused: engine_args,
    ):
        projected = stage_init_utils.build_legacy_engine_args_dict(stage, "dummy-model")

    assert projected[FT_CONFIG_WAS_TYPED_KEY] is True
    assert projected["enable_fault_tolerance"] is False
    transport = {
        key: value
        for key, value in projected.items()
        if key in ("enable_fault_tolerance", "fault_tolerance_config", FT_CONFIG_WAS_TYPED_KEY)
    }
    assert _captured_engine_args(transport)["enabled"] is False


def test_legacy_stage_builder_does_not_mark_a_plain_stage_dict():
    """A stage config written as a dict keeps upstream's explicit-config reading."""
    explicit = {"enable_fault_tolerance": False, "fault_tolerance_config": asdict(FaultToleranceConfig())}
    stage = SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args=dict(explicit),
        runtime=SimpleNamespace(devices=None),
    )
    with mock.patch.object(
        stage_init_utils,
        "_finalize_engine_args_dict",
        side_effect=lambda engine_args, **unused: engine_args,
    ):
        projected = stage_init_utils.build_legacy_engine_args_dict(stage, "dummy-model")

    assert FT_CONFIG_WAS_TYPED_KEY not in projected
    assert projected["fault_tolerance_config"] == explicit["fault_tolerance_config"]
    assert stage.engine_args["enable_fault_tolerance"] is False
