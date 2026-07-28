#!/usr/bin/env python3
"""Gate contact-coupled native revolute motor semantics headlessly."""

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
            RunSpec("avbd", "parallel", "FAIL"),
            RunSpec("avbd", "sequential", "FAIL"),
        )
    if mode == "baseline":
        return (
            RunSpec("tgs", "parallel", "PASS"),
            RunSpec("avbd", "parallel", "FAIL"),
            RunSpec("avbd", "sequential", "FAIL"),
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


def positive_int(
    fields: dict[str, str], key: str, errors: list[str]
) -> int | None:
    try:
        value = int(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key}={fields.get(key)!r}, expected integer")
        return None
    if value <= 0:
        errors.append(f"{key}={value}, expected positive")
        return None
    return value


def run_one(spec: RunSpec, bin_dir: Path, timeout: float) -> bool:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=revolute-motor-contact",
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
        "fixture": "[PROBE] [SnippetJointRevoluteMotorContact] ",
        "cleanup": "[PROBE] [SnippetJointRevoluteMotorContactCleanup] ",
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
        "case": "revolute-motor-contact",
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
        "case": "revolute-motor-contact",
        "topology": "dynamic-dynamic-ground",
        "actorOrderValid": "1",
        "driveEnabled": "1",
        "targetVelocity": "2",
        "forceLimit": "1000",
        "radius": "0.5",
        "halfHeight": "0.5",
        "centerHeight": "0.5",
        "stateSamples": str(FRAMES),
        "nonFiniteSamples": "0",
        "initialDynamicActors": "2",
        "initialStaticActors": "1",
        "initialConstraints": "1",
        "finalDynamicActors": "2",
        "finalStaticActors": "1",
        "finalConstraints": "1",
    }.items():
        if fixture.get(key) != expected:
            errors.append(
                f"fixture {key}={fixture.get(key)!r}, expected {expected!r}"
            )
    positive_int(fixture, "contactEvents", errors)
    positive_int(fixture, "contactPointCount", errors)
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
        "revolute_motor_contact_relative_velocity",
        "revolute_motor_contact_drive_reaction",
        "revolute_motor_contact_joint_error",
    }:
        errors.append(f"unexpected failure reason: {gate.get('reason')!r}")

    metric_keys = (
        "finalRelativeVelocity",
        "finalRelativeError",
        "maximumLateRelativeError",
        "meanLateDriveReaction",
        "maximumLateDriveReaction",
        "totalNormalImpulse",
        "totalTangentialImpulse",
        "maximumTangentialImpulse",
        "maximumAnchorError",
        "maximumAxisMisalignment",
        "maximumCenterHeightError",
    )
    metrics = {
        key: finite_float(fixture, key, errors) for key in metric_keys
    }
    positive_int(fixture, "lateDriveReactionSamples", errors)
    if all(value is not None for value in metrics.values()):
        physical_pass = (
            abs(metrics["finalRelativeVelocity"] - 2.0) <= 0.05
            and metrics["finalRelativeError"] <= 0.05
            and metrics["maximumLateRelativeError"] <= 0.05
            and metrics["totalNormalImpulse"] > 0.0
            and metrics["meanLateDriveReaction"] >= 1.0e-3
            and metrics["maximumLateDriveReaction"] > 0.0
            and metrics["maximumAnchorError"] <= 1.0e-3
            and metrics["maximumAxisMisalignment"] <= 1.0e-3
            and metrics["maximumCenterHeightError"] <= 0.02
        )
        if spec.expected_status == "PASS" and not physical_pass:
            errors.append("PASS status lacks contact-coupled motor semantics")
        if spec.expected_status == "FAIL" and physical_pass:
            errors.append("FAIL status lacks an independent red metric")

    print(
        f"[REVOLUTE_MOTOR_CONTACT_RUN] solver={spec.solver} "
        f"execution={spec.execution} "
        f"status={gate.get('status', 'MISSING')} "
        f"reason={gate.get('reason', 'MISSING')} "
        f"relative={fixture.get('finalRelativeVelocity', 'MISSING')} "
        f"lateError={fixture.get('maximumLateRelativeError', 'MISSING')} "
        f"contacts={fixture.get('contactEvents', 'MISSING')} "
        f"points={fixture.get('contactPointCount', 'MISSING')} "
        f"normalImpulse={fixture.get('totalNormalImpulse', 'MISSING')} "
        f"driveReaction="
        f"{fixture.get('meanLateDriveReaction', 'MISSING')} "
        f"anchor={fixture.get('maximumAnchorError', 'MISSING')} "
        f"height={fixture.get('maximumCenterHeightError', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(
            f"[REVOLUTE_MOTOR_CONTACT_ERROR] solver={spec.solver} "
            f"execution={spec.execution} error={error}"
        )
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run contact-coupled native revolute motor headlessly."
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
        f"[REVOLUTE_MOTOR_CONTACT_SUMMARY] mode={args.mode} "
        f"runs={passes}/{len(specs)} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
