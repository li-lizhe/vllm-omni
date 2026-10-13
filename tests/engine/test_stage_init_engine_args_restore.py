# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression tests for the per-stage engine-args reconstruction (issue #8816).

A stage's ``engine_args`` are serialized into a plain dict and handed back to
``OmniEngineArgs`` in :func:`build_vllm_config`.  Nested configs survive that
round trip as dicts, and ``EngineArgs.__post_init__`` reads a *dict*
``fault_tolerance_config`` as an explicit ``--fault-tolerance-config``: it logs
the "automatically enabled" warning and flips ``enable_fault_tolerance`` on for
a stage that never asked for it.
"""

from dataclasses import asdict

import pytest
from vllm.config import FaultToleranceConfig
from vllm.engine.arg_utils import EngineArgs

from vllm_omni.engine.stage_init_utils import restore_serialized_engine_args

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_a_serialized_fault_tolerance_config_enables_fault_tolerance():
    """Pin the upstream behaviour the stage-init path has to stay clear of."""
    serialized = asdict(FaultToleranceConfig())

    args = EngineArgs(fault_tolerance_config=dict(serialized))

    assert args.enable_fault_tolerance is True


def test_restore_keeps_a_stage_off_the_fault_tolerance_path():
    serialized = {
        "fault_tolerance_config": asdict(FaultToleranceConfig()),
        "enable_fault_tolerance": False,
    }

    restored = restore_serialized_engine_args(dict(serialized))

    assert isinstance(restored["fault_tolerance_config"], FaultToleranceConfig)
    assert (
        restored["fault_tolerance_config"].engine_recovery_timeout_sec
        == FaultToleranceConfig().engine_recovery_timeout_sec
    )
    assert EngineArgs(**restored).enable_fault_tolerance is False


def test_restore_leaves_an_explicit_opt_in_alone():
    restored = restore_serialized_engine_args(
        {
            "fault_tolerance_config": {"engine_recovery_timeout_sec": 30},
            "enable_fault_tolerance": True,
        }
    )

    assert isinstance(restored["fault_tolerance_config"], FaultToleranceConfig)
    assert restored["fault_tolerance_config"].engine_recovery_timeout_sec == 30
    assert EngineArgs(**restored).enable_fault_tolerance is True


def test_restore_ignores_values_that_are_already_dataclasses():
    config = FaultToleranceConfig(engine_recovery_timeout_sec=30)

    restored = restore_serialized_engine_args({"fault_tolerance_config": config})

    assert restored["fault_tolerance_config"] is config
