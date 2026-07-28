#!/usr/bin/env python3
"""Gate dynamic-pair native revolute motor plus active-limit semantics."""

from __future__ import annotations

import run_snippet_joint_revolute_motor_limit_headless as limit_runner


limit_runner.CASE_NAME = "revolute-motor-dynamic-limit"
limit_runner.TOPOLOGY = "dynamic-dynamic"
limit_runner.EXPECTED_DYNAMIC_ACTORS = "2"
limit_runner.OUTPUT_TAG = "REVOLUTE_MOTOR_DYNAMIC_LIMIT"


if __name__ == "__main__":
    raise SystemExit(limit_runner.main())
