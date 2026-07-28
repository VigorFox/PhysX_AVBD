#!/usr/bin/env python3
"""Gate dynamic-pair native revolute free-spin motor semantics."""

from __future__ import annotations

import run_snippet_joint_revolute_motor_freespin_headless as free_spin_runner


free_spin_runner.CASE_NAME = "revolute-motor-dynamic-freespin"
free_spin_runner.TOPOLOGY = "dynamic-dynamic"
free_spin_runner.EXPECTED_DYNAMIC_ACTORS = "2"
free_spin_runner.REQUIRE_MOMENTUM = True
free_spin_runner.OUTPUT_TAG = "REVOLUTE_MOTOR_DYNAMIC_FREESPIN"


if __name__ == "__main__":
    raise SystemExit(free_spin_runner.main())
