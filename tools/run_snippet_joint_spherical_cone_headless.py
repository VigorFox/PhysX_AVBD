#!/usr/bin/env python3
"""Gate native PxSphericalJoint asymmetric cone semantics headlessly."""

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
FRAMES = 360
CASES = ("spherical-cone-inside", "spherical-cone-outside")
TOPOLOGIES = ("static-dynamic", "dynamic-dynamic")
LIMIT_Y = math.pi / 9.0
LIMIT_Z = 7.0 * math.pi / 36.0
INSIDE_Y = math.pi / 36.0
INSIDE_Z = math.pi / 6.0
OUTSIDE_Y = math.pi / 10.0
OUTSIDE_Z = math.pi / 6.0
FINAL_RADIUS_TOLERANCE = 0.01
LATE_RADIUS_TOLERANCE = 0.02
INSIDE_DEVIATION_TOLERANCE = 0.01
MINIMUM_RADIUS_CORRECTION = 0.10
ANGULAR_MOMENTUM_MAXIMUM = 1.0e-3
ANCHOR_SEPARATION_MAXIMUM = 1.0e-3


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    case: str
    topology: str
    expected_status: str


def make_specs(mode: str) -> tuple[RunSpec, ...]:
    if mode == "authority":
        lanes = (("tgs", "parallel"),)
    elif mode == "probe":
        lanes = (("avbd", "parallel"), ("avbd", "sequential"))
    else:
        lanes = (
            ("tgs", "parallel"),
            ("avbd", "parallel"),
            ("avbd", "sequential"),
        )

    specs: list[RunSpec] = []
    for solver, execution in lanes:
        for case in CASES:
            for topology in TOPOLOGIES:
                expected = "PASS"
                if mode == "baseline" and solver == "avbd":
                    expected = "FAIL"
                specs.append(
                    RunSpec(
                        f"{solver}-{execution}-{case}-{topology}",
                        solver,
                        execution,
                        case,
                        topology,
                        expected,
                    )
                )
    return tuple(specs)


def parse_fields(
    line: str, prefix: str
) -> tuple[dict[str, str], list[str]]:
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


def check_close(
    fields: dict[str, str],
    key: str,
    expected: float,
    tolerance: float,
    errors: list[str],
) -> float | None:
    value = parse_float(fields, key, errors)
    if value is not None and abs(value - expected) > tolerance:
        errors.append(
            f"{key}={value}, expected {expected} +/- {tolerance}"
        )
    return value


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case}",
        f"--topology={spec.topology}",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
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
    fixture_prefix = "[PROBE] [SnippetJointSphericalCone] "
    cleanup_prefix = "[PROBE] [SnippetJointSphericalConeCleanup] "
    gate_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(gate_prefix)
    ]
    fixture_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(fixture_prefix)
    ]
    cleanup_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(cleanup_prefix)
    ]
    errors: list[str] = []
    gate: dict[str, str] = {}
    fixture: dict[str, str] = {}
    cleanup: dict[str, str] = {}
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
        gate, parse_errors = parse_fields(gate_lines[0], gate_prefix)
        errors.extend(parse_errors)
    if len(fixture_lines) != 1:
        errors.append(
            f"fixture witness count is {len(fixture_lines)}, expected 1"
        )
    else:
        fixture, parse_errors = parse_fields(
            fixture_lines[0], fixture_prefix
        )
        errors.extend(parse_errors)
    if len(cleanup_lines) != 1:
        errors.append(
            f"cleanup witness count is {len(cleanup_lines)}, expected 1"
        )
    else:
        cleanup, parse_errors = parse_fields(
            cleanup_lines[0], cleanup_prefix
        )
        errors.extend(parse_errors)

    gate_exact = {
        "schema": "1",
        "snippet": "SnippetJoint",
        "case": spec.case,
        "joint": "spherical",
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "seed": "1",
        "dispatcherThreads": "2",
        "capability": "PARTIAL",
        "validation": "PROBE",
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "launchFailures": "0",
        "status": spec.expected_status,
    }
    for key, expected in gate_exact.items():
        if gate.get(key) != expected:
            errors.append(
                f"gate {key}={gate.get(key)!r}, expected {expected!r}"
            )
    fixture_exact = {
        "case": spec.case,
        "topology": spec.topology,
        "limitEnabled": "1",
        "actorOrderValid": "1",
        "stateSamples": str(FRAMES),
        "nonFiniteSamples": "0",
        "initialDynamicActors": (
            "2" if spec.topology == "dynamic-dynamic" else "1"
        ),
        "initialStaticActors": (
            "0" if spec.topology == "dynamic-dynamic" else "1"
        ),
        "initialConstraints": "1",
        "finalDynamicActors": (
            "2" if spec.topology == "dynamic-dynamic" else "1"
        ),
        "finalStaticActors": (
            "0" if spec.topology == "dynamic-dynamic" else "1"
        ),
        "finalConstraints": "1",
    }
    for key, expected in fixture_exact.items():
        if fixture.get(key) != expected:
            errors.append(
                f"fixture {key}={fixture.get(key)!r}, "
                f"expected {expected!r}"
            )
    cleanup_exact = {
        "dynamicActors": "0",
        "staticActors": "0",
        "constraints": "0",
        "cleanupComplete": "1",
    }
    for key, expected in cleanup_exact.items():
        if cleanup.get(key) != expected:
            errors.append(
                f"cleanup {key}={cleanup.get(key)!r}, "
                f"expected {expected!r}"
            )

    expected_exit = 0 if spec.expected_status == "PASS" else 1
    if result.returncode != expected_exit:
        errors.append(
            f"exit code {result.returncode}, expected {expected_exit}"
        )
    reason = gate.get("reason")
    if spec.expected_status == "PASS" and reason != "none":
        errors.append(f"reason={reason!r}, expected 'none'")
    if spec.expected_status == "FAIL" and reason not in {
        "spherical_cone_inside_state_disturbed",
        "spherical_cone_limit_not_enforced",
        "spherical_cone_conservation",
    }:
        errors.append(f"unexpected baseline reason: {reason!r}")

    check_close(fixture, "limitY", LIMIT_Y, 1.0e-6, errors)
    check_close(fixture, "limitZ", LIMIT_Z, 1.0e-6, errors)
    initial_y = INSIDE_Y if spec.case.endswith("inside") else OUTSIDE_Y
    initial_z = INSIDE_Z if spec.case.endswith("inside") else OUTSIDE_Z
    expected_initial_radius = math.sqrt(
        (initial_y / LIMIT_Y) ** 2 + (initial_z / LIMIT_Z) ** 2
    )
    initial_radius = check_close(
        fixture,
        "initialEllipseRadius",
        expected_initial_radius,
        1.0e-4,
        errors,
    )
    final_radius = parse_float(fixture, "finalEllipseRadius", errors)
    min_late_radius = parse_float(
        fixture, "minimumLateEllipseRadius", errors
    )
    max_late_radius = parse_float(
        fixture, "maximumLateEllipseRadius", errors
    )
    inside_deviation = parse_float(
        fixture, "maximumInsideDeviation", errors
    )
    radius_correction = parse_float(
        fixture, "radiusCorrection", errors
    )
    angular_momentum = parse_float(
        fixture, "maximumTotalAngularMomentum", errors
    )
    anchor_separation = parse_float(
        fixture, "maximumAnchorSeparation", errors
    )

    metrics = (
        initial_radius,
        final_radius,
        min_late_radius,
        max_late_radius,
        inside_deviation,
        radius_correction,
        angular_momentum,
        anchor_separation,
    )
    if all(value is not None for value in metrics):
        assert initial_radius is not None
        assert final_radius is not None
        assert min_late_radius is not None
        assert max_late_radius is not None
        assert inside_deviation is not None
        assert radius_correction is not None
        assert angular_momentum is not None
        assert anchor_separation is not None
        if spec.case.endswith("inside"):
            semantics_pass = (
                initial_radius < 1.0
                and inside_deviation <= INSIDE_DEVIATION_TOLERANCE
            )
        else:
            semantics_pass = (
                initial_radius > 1.0
                and 1.0 - FINAL_RADIUS_TOLERANCE
                <= final_radius
                <= 1.0 + FINAL_RADIUS_TOLERANCE
                and min_late_radius >= 1.0 - LATE_RADIUS_TOLERANCE
                and max_late_radius <= 1.0 + LATE_RADIUS_TOLERANCE
                and radius_correction >= MINIMUM_RADIUS_CORRECTION
            )
        semantics_pass = (
            semantics_pass
            and anchor_separation <= ANCHOR_SEPARATION_MAXIMUM
            and (
                spec.topology != "dynamic-dynamic"
                or angular_momentum <= ANGULAR_MOMENTUM_MAXIMUM
            )
        )
        if spec.expected_status == "PASS" and not semantics_pass:
            errors.append("PASS status lacks spherical-cone semantics")
        if spec.expected_status == "FAIL" and semantics_pass:
            errors.append("FAIL status lacks an independent red metric")

    print(
        f"[SPHERICAL_CONE_RUN] name={spec.name} "
        f"status={gate.get('status', 'MISSING')} "
        f"reason={gate.get('reason', 'MISSING')} "
        f"initialRadius={fixture.get('initialEllipseRadius', 'MISSING')} "
        f"finalRadius={fixture.get('finalEllipseRadius', 'MISSING')} "
        f"lateRadius=[{fixture.get('minimumLateEllipseRadius', 'MISSING')},"
        f"{fixture.get('maximumLateEllipseRadius', 'MISSING')}] "
        f"insideDeviation={fixture.get('maximumInsideDeviation', 'MISSING')} "
        f"angularMomentum={fixture.get('maximumTotalAngularMomentum', 'MISSING')} "
        f"anchor={fixture.get('maximumAnchorSeparation', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(f"[SPHERICAL_CONE_ERROR] name={spec.name} error={error}")
    return not errors, gate


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SnippetJoint native asymmetric spherical-cone matrix "
            "without a window."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "baseline", "acceptance"),
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
        passed, gate = run_one(spec, bin_dir, args.timeout)
        passed_runs += int(passed)
        status = gate.get("status", "ERROR")
        status_counts[status if status in status_counts else "ERROR"] += 1

    passed = passed_runs == len(specs)
    print(
        f"[SPHERICAL_CONE_SUMMARY] mode={args.mode} "
        f"runs={passed_runs}/{len(specs)} "
        f"physicalPass={status_counts['PASS']} "
        f"physicalFail={status_counts['FAIL']} "
        f"error={status_counts['ERROR']} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
