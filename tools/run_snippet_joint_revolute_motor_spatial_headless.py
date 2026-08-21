#!/usr/bin/env python3
"""Gate combined off-center/off-principal revolute motor response headlessly."""

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
FRAMES = 360
DT = 1.0 / 60.0


@dataclass(frozen=True)
class RunSpec:
    solver: str
    execution: str
    expected_status: str


def specs_for(mode: str) -> tuple[RunSpec, ...]:
    if mode == "authority":
        return (RunSpec("tgs", "parallel", "PASS"),)
    if mode == "probe":
        return (
            RunSpec("avbd", "parallel", "PASS"),
            RunSpec("avbd", "sequential", "PASS"),
        )
    if mode == "baseline":
        return (
            RunSpec("tgs", "parallel", "PASS"),
            RunSpec("avbd", "parallel", "PASS"),
            RunSpec("avbd", "sequential", "PASS"),
        )
    return (
        RunSpec("tgs", "parallel", "PASS"),
        RunSpec("avbd", "parallel", "PASS"),
        RunSpec("avbd", "sequential", "PASS"),
    )


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


def finite_float(
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


def run_one(spec: RunSpec, bin_dir: Path, timeout: float) -> bool:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=revolute-motor-spatial",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
        f"--dt={DT:.12g}",
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
    prefixes = {
        "gate": "[AVBD_GATE] ",
        "fixture": "[PROBE] [SnippetJointRevoluteMotorOffCenter] ",
        "cleanup": (
            "[PROBE] [SnippetJointRevoluteMotorOffCenterCleanup] "
        ),
    }
    parsed: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for name, prefix in prefixes.items():
        lines = [
            line.strip()
            for line in combined.splitlines()
            if line.startswith(prefix)
        ]
        if len(lines) != 1:
            errors.append(f"{name} count is {len(lines)}, expected 1")
            parsed[name] = {}
        else:
            parsed[name], parse_errors = parse_fields(lines[0], prefix)
            errors.extend(parse_errors)
    gate = parsed["gate"]
    fixture = parsed["fixture"]
    cleanup = parsed["cleanup"]

    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append(
            "visible window detected: "
            + ", ".join(result.visible_window_titles)
        )
    for key, expected in {
        "schema": "1",
        "snippet": "SnippetJoint",
        "case": "revolute-motor-spatial",
        "joint": "revolute",
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "capability": "PARTIAL",
        "validation": "PROBE",
        "status": spec.expected_status,
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
    }.items():
        if gate.get(key) != expected:
            errors.append(
                f"gate {key}={gate.get(key)!r}, expected {expected!r}"
            )
    for key, expected in {
        "case": "revolute-motor-spatial",
        "topology": "world-dynamic",
        "actorOrderValid": "1",
        "driveEnabled": "1",
        "targetVelocity": "2",
        "forceLimit": "1000",
        "inertiaX": "1",
        "inertiaY": "4",
        "inertiaZ": "7",
        "stateSamples": str(FRAMES),
        "nonFiniteSamples": "0",
        "initialDynamicActors": "1",
        "initialStaticActors": "0",
        "initialConstraints": "1",
        "finalDynamicActors": "1",
        "finalStaticActors": "0",
        "finalConstraints": "1",
    }.items():
        if fixture.get(key) != expected:
            errors.append(
                f"fixture {key}={fixture.get(key)!r}, expected {expected!r}"
            )
    for key, expected in {
        "dynamicActors": "0",
        "staticActors": "0",
        "constraints": "0",
        "cleanupComplete": "1",
    }.items():
        if cleanup.get(key) != expected:
            errors.append(
                f"cleanup {key}={cleanup.get(key)!r}, expected {expected!r}"
            )

    expected_exit = 0 if spec.expected_status == "PASS" else 1
    if result.returncode != expected_exit:
        errors.append(
            f"exit code {result.returncode}, expected {expected_exit}"
        )
    if spec.expected_status == "PASS" and gate.get("reason") != "none":
        errors.append(f"reason={gate.get('reason')!r}, expected 'none'")
    if spec.expected_status == "FAIL" and gate.get("reason") not in {
        "revolute_motor_spatial_velocity",
        "revolute_motor_spatial_swing",
        "revolute_motor_spatial_anchor_velocity",
        "revolute_motor_spatial_joint_error",
    }:
        errors.append(f"unexpected failure reason: {gate.get('reason')!r}")

    metric_keys = (
        "initialPerpendicularLeverArm",
        "initialOffPrincipalResponse",
        "bodyAngle",
        "finalHingeVelocity",
        "finalHingeVelocityError",
        "maximumLateHingeVelocityError",
        "maximumLateSwingVelocity",
        "finalAnchorPointSpeed",
        "maximumLateAnchorPointSpeed",
        "maximumLinearSpeed",
        "maximumLinearReaction",
        "maximumAnchorError",
        "maximumAxisMisalignment",
    )
    metrics = {
        key: finite_float(fixture, key, errors) for key in metric_keys
    }
    if all(value is not None for value in metrics.values()):
        if metrics["initialPerpendicularLeverArm"] < 0.5:
            errors.append("fixture lacks a perpendicular anchor lever arm")
        if metrics["initialOffPrincipalResponse"] < 0.05:
            errors.append("fixture lacks an off-principal inertia response")
        if abs(metrics["bodyAngle"]) < 0.25:
            errors.append("fixture lacks a material body rotation")
        physical_pass = (
            abs(metrics["finalHingeVelocity"] - 2.0) <= 0.05
            and metrics["finalHingeVelocityError"] <= 0.05
            and metrics["maximumLateHingeVelocityError"] <= 0.05
            and metrics["maximumLateSwingVelocity"] <= 0.05
            and metrics["finalAnchorPointSpeed"] <= 0.05
            and metrics["maximumLateAnchorPointSpeed"] <= 0.05
            and metrics["maximumAnchorError"] <= 2.0e-3
            and metrics["maximumAxisMisalignment"] <= 1.0e-3
        )
        if spec.expected_status == "PASS" and not physical_pass:
            errors.append("PASS status lacks complete spatial motor semantics")
        if spec.expected_status == "FAIL" and physical_pass:
            errors.append("FAIL status lacks an independent red metric")

    print(
        f"[REVOLUTE_MOTOR_SPATIAL_RUN] solver={spec.solver} "
        f"execution={spec.execution} "
        f"status={gate.get('status', 'MISSING')} "
        f"reason={gate.get('reason', 'MISSING')} "
        f"lever={fixture.get('initialPerpendicularLeverArm', 'MISSING')} "
        f"offPrincipal="
        f"{fixture.get('initialOffPrincipalResponse', 'MISSING')} "
        f"hinge={fixture.get('finalHingeVelocity', 'MISSING')} "
        f"lateError="
        f"{fixture.get('maximumLateHingeVelocityError', 'MISSING')} "
        f"swing={fixture.get('maximumLateSwingVelocity', 'MISSING')} "
        f"anchorSpeed="
        f"{fixture.get('maximumLateAnchorPointSpeed', 'MISSING')} "
        f"linearSpeed={fixture.get('maximumLinearSpeed', 'MISSING')} "
        f"reaction={fixture.get('maximumLinearReaction', 'MISSING')} "
        f"anchor={fixture.get('maximumAnchorError', 'MISSING')} "
        f"axis={fixture.get('maximumAxisMisalignment', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(
            f"[REVOLUTE_MOTOR_SPATIAL_ERROR] "
            f"solver={spec.solver} execution={spec.execution} error={error}"
        )
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run combined off-center/off-principal native revolute motor "
            "headlessly."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "baseline", "acceptance"),
        default="baseline",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        parser.error(f"executable not found: {bin_dir / EXECUTABLE}")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")

    specs = specs_for(args.mode)
    passes = sum(
        run_one(spec, bin_dir, args.timeout) for spec in specs
    )
    passed = passes == len(specs)
    print(
        f"[REVOLUTE_MOTOR_SPATIAL_SUMMARY] mode={args.mode} "
        f"runs={passes}/{len(specs)} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
