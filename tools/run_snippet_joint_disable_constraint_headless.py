#!/usr/bin/env python3
"""Gate external fixed-joint eDISABLE_CONSTRAINT consumption headlessly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetJoint_64.exe"
FREQUENCY = 60
ORDERS = ("normal", "swapped")
SEPARATION_ERROR_MINIMUM = 100.0
RELATIVE_SPEED_MINIMUM = 50.0
REACTION_MAXIMUM = 1.0e-4
CENTER_OF_MASS_ERROR_MAXIMUM = 1.0e-2
TOTAL_MOMENTUM_MAXIMUM = 5.0e-2


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    actor_order: str
    expected_status: str


def make_specs(mode: str) -> tuple[RunSpec, ...]:
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for solver, execution in lanes:
        for actor_order in ORDERS:
            expected = "PASS"
            if mode == "baseline" and solver == "avbd":
                expected = "FAIL"
            specs.append(
                RunSpec(
                    f"{solver}-{execution}-{actor_order}",
                    solver,
                    execution,
                    actor_order,
                    expected,
                )
            )
    return tuple(specs)


def parse_fields(line: str, prefix: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line[len(prefix) :].split():
        if "=" not in token:
            errors.append(f"malformed token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate key: {key}")
        fields[key] = value
    return fields, errors


def parse_float(
    fields: dict[str, str], key: str, errors: list[str]
) -> float | None:
    try:
        value = float(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key}={fields.get(key)!r}, expected float")
        return None
    if not math.isfinite(value):
        errors.append(f"{key}={value}, expected finite")
        return None
    return value


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, dict[str, str]]:
    frames = FREQUENCY * 10
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=force-pair-disabled",
        f"--execution={spec.execution}",
        f"--actor-order={spec.actor_order}",
        f"--frames={frames}",
        f"--dt={1.0 / FREQUENCY:.12g}",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    result = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout
    )
    combined = result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr

    gate_prefix = "[AVBD_GATE] "
    flag_prefix = "[PROBE] [SnippetJointConstraintFlag] "
    gate_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(gate_prefix)
    ]
    flag_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(flag_prefix)
    ]
    errors: list[str] = []
    fields: dict[str, str] = {}
    flag_fields: dict[str, str] = {}
    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append(
            "visible window detected: "
            + ", ".join(result.visible_window_titles)
        )
    if len(gate_lines) != 1:
        errors.append(f"gate count is {len(gate_lines)}, expected 1")
    else:
        fields, parse_errors = parse_fields(gate_lines[0], gate_prefix)
        errors.extend(parse_errors)
    if len(flag_lines) != 1:
        errors.append(f"flag witness count is {len(flag_lines)}, expected 1")
    else:
        flag_fields, parse_errors = parse_fields(flag_lines[0], flag_prefix)
        errors.extend(parse_errors)

    exact = {
        "schema": "1",
        "snippet": "SnippetJoint",
        "case": "force-pair-disabled",
        "joint": "fixed",
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(frames),
        "completedFrames": str(frames),
        "seed": "1",
        "dispatcherThreads": "2",
        "capability": "PARTIAL",
        "validation": "PROBE",
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "launchFailures": "0",
        "topologyDynamicActors": "2",
        "topologyStaticActors": "0",
        "topologyConstraints": "1",
        "finalDynamicActors": "2",
        "finalStaticActors": "0",
        "finalConstraints": "1",
        "cleanupDynamicActors": "0",
        "cleanupStaticActors": "0",
        "cleanupConstraints": "0",
        "cleanupComplete": "1",
        "forceFixture": "dynamic-pair-disabled",
        "forceActorOrder": spec.actor_order,
        "forcePairActorOrderValid": "1",
        "status": spec.expected_status,
    }
    for key, expected in exact.items():
        if fields.get(key) != expected:
            errors.append(f"{key}={fields.get(key)!r}, expected {expected!r}")
    flag_exact = {
        "requestedDisabled": "1",
        "readbackDisabled": "1",
        "separationErrorMin": "100",
        "relativeSpeedMin": "50",
        "reactionMax": "9.99999975e-05",
        "centerOfMassErrorMax": "0.00999999978",
        "totalMomentumMax": "0.0500000007",
    }
    for key, expected in flag_exact.items():
        if flag_fields.get(key) != expected:
            errors.append(
                f"flag {key}={flag_fields.get(key)!r}, expected {expected!r}"
            )

    expected_exit = 0 if spec.expected_status == "PASS" else 1
    if result.returncode != expected_exit:
        errors.append(
            f"exit code {result.returncode}, expected {expected_exit}"
        )
    reason = fields.get("reason")
    if spec.expected_status == "PASS" and reason != "none":
        errors.append(f"reason={reason!r}, expected 'none'")
    if spec.expected_status == "FAIL" and reason not in {
        "disabled_constraint_reaction",
        "disabled_constraint_motion",
    }:
        errors.append(f"unexpected baseline reason: {reason!r}")

    metrics = {
        key: parse_float(fields, key, errors)
        for key in (
            "meanSampleForceMagnitude",
            "forcePairMaxSeparationError",
            "forcePairMaxRelativeSpeed",
            "forcePairMaxCenterOfMassError",
            "forcePairMaxTotalMomentum",
        )
    }
    if all(value is not None for value in metrics.values()):
        semantics_pass = (
            metrics["meanSampleForceMagnitude"] <= REACTION_MAXIMUM
            and metrics["forcePairMaxSeparationError"]
            >= SEPARATION_ERROR_MINIMUM
            and metrics["forcePairMaxRelativeSpeed"]
            >= RELATIVE_SPEED_MINIMUM
            and metrics["forcePairMaxCenterOfMassError"]
            <= CENTER_OF_MASS_ERROR_MAXIMUM
            and metrics["forcePairMaxTotalMomentum"]
            <= TOTAL_MOMENTUM_MAXIMUM
        )
        if spec.expected_status == "PASS" and not semantics_pass:
            errors.append("PASS status lacks disabled-constraint semantics")
        if spec.expected_status == "FAIL" and semantics_pass:
            errors.append("FAIL status lacks an independent red metric")

    print(
        f"[DISABLE_CONSTRAINT_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"readback={flag_fields.get('readbackDisabled', 'MISSING')} "
        f"reaction={fields.get('meanSampleForceMagnitude', 'MISSING')} "
        f"maxSeparation={fields.get('forcePairMaxSeparationError', 'MISSING')} "
        f"maxRelativeSpeed={fields.get('forcePairMaxRelativeSpeed', 'MISSING')} "
        f"maxCOM={fields.get('forcePairMaxCenterOfMassError', 'MISSING')} "
        f"maxMomentum={fields.get('forcePairMaxTotalMomentum', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(f"[DISABLE_CONSTRAINT_ERROR] name={spec.name} error={error}")
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SnippetJoint external fixed-joint "
            "eDISABLE_CONSTRAINT matrix without a window."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "acceptance"),
        default="baseline",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        parser.error(f"executable not found: {executable}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    specs = make_specs(args.mode)
    passed_runs = 0
    status_counts = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    for spec in specs:
        passed, fields = run_one(spec, bin_dir, args.timeout)
        passed_runs += int(passed)
        status = fields.get("status", "ERROR")
        status_counts[status if status in status_counts else "ERROR"] += 1

    passed = passed_runs == len(specs)
    print(
        f"[DISABLE_CONSTRAINT_SUMMARY] mode={args.mode} "
        f"runs={passed_runs}/{len(specs)} "
        f"physicalPass={status_counts['PASS']} "
        f"physicalFail={status_counts['FAIL']} "
        f"error={status_counts['ERROR']} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
