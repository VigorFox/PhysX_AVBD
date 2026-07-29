#!/usr/bin/env python3
"""Gate contact-coupled native revolute motor semantics headlessly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path

from avbd_contact_objective_ir_gate import validate_contact_objective_ir
from avbd_joint_objective_ir_gate import validate_joint_objective_ir
from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetJoint_64.exe"
FRAMES = 360
DT = 1.0 / 60.0
DIAGNOSTIC_CADENCE = 4
OBJECTIVE_IR_PREFIX = "[avbd:objective-ir] "
OBJECTIVE_IR_PARTITION_FIELDS = (
    "objectivePositionRows",
    "objectivePointRows",
    "objectiveManifoldRows",
    "objectiveComponentRows",
    "objectiveJointRows",
    "objectiveUnsupportedRows",
    "objectiveLegacyRows",
    "objectiveInvalidRows",
)


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


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, int, int, int]:
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
    env["PHYSX_AVBD_ITER_DIAG"] = "1" if spec.solver == "avbd" else "0"
    # Frame 1 is always emitted by the engine, so the transient Unsupported
    # owner remains covered while a lower cadence keeps diagnostics from
    # consuming the fixed 60-second physics budget.
    env["PHYSX_AVBD_ITER_DIAG_EVERY"] = str(DIAGNOSTIC_CADENCE)
    env["PHYSX_AVBD_ITER_DIAG_SEQUENTIAL"] = (
        "1" if spec.execution == "sequential" else "0"
    )
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

    objective_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(OBJECTIVE_IR_PREFIX)
    ]
    objective_joint_rows = 0
    objective_unsupported_rows = 0
    objective_fingerprint = 0
    if spec.solver == "avbd":
        joint_objective_errors, _ = validate_joint_objective_ir(
            combined,
            expected_owner="JointFinalize",
            allow_unsupported=True,
            require_unsupported=True,
        )
        errors.extend(joint_objective_errors)
        contact_objective_errors, _ = validate_contact_objective_ir(
            combined,
            required_owners=(
                "PositionAL",
                "JointFinalize",
                "Unsupported",
            ),
            allow_unsupported=True,
        )
        errors.extend(contact_objective_errors)
        if not objective_lines:
            errors.append("missing compiled-objective diagnostics")
        for line in objective_lines:
            fields, parse_errors = parse_fields(
                line, OBJECTIVE_IR_PREFIX
            )
            errors.extend(parse_errors)
            try:
                rows = int(fields["rows"])
                objective_fingerprint += int(
                    fields["objectiveFingerprint"]
                )
                partition_values = {
                    key: int(fields[key])
                    for key in OBJECTIVE_IR_PARTITION_FIELDS
                }
            except (KeyError, ValueError):
                errors.append(
                    "compiled-objective line has missing/non-integer field"
                )
                continue
            partition = sum(partition_values.values())
            if rows != partition:
                errors.append(
                    f"compiled-objective rows={rows}, "
                    f"partition={partition}"
                )
            if partition_values["objectiveInvalidRows"] != 0:
                errors.append("compiled-objective Invalid row detected")
            explicit_rows = (
                partition_values["objectiveJointRows"] +
                partition_values["objectiveUnsupportedRows"]
            )
            if rows > 0 and explicit_rows != rows:
                errors.append(
                    f"frame={fields.get('frame', 'MISSING')} "
                    "contact-coupled objective is not explicitly owned: "
                    f"joint="
                    f"{partition_values['objectiveJointRows']} "
                    f"unsupported="
                    f"{partition_values['objectiveUnsupportedRows']} "
                    f"legacy="
                    f"{partition_values['objectiveLegacyRows']} "
                    f"invalid="
                    f"{partition_values['objectiveInvalidRows']} "
                    f"rows={rows}"
                )
            objective_joint_rows += partition_values[
                "objectiveJointRows"
            ]
            objective_unsupported_rows += partition_values[
                "objectiveUnsupportedRows"
            ]
        if objective_joint_rows <= 0:
            errors.append("compiled JointFinalize owner was not observed")
        if objective_unsupported_rows <= 0:
            errors.append(
                "explicit Unsupported transient-contact owner was not "
                "observed"
            )
    elif objective_lines:
        errors.append("unexpected compiled-objective diagnostics on TGS")
    elif "[avbd:joint-objective-ir] " in combined:
        errors.append("unexpected joint-objective diagnostics on TGS")

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
        f"jointOwnerRows={objective_joint_rows} "
        f"unsupportedOwnerRows={objective_unsupported_rows} "
        f"objectiveFingerprint={objective_fingerprint} "
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
    return (
        not errors,
        objective_joint_rows,
        objective_unsupported_rows,
        objective_fingerprint,
    )


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
    results = [
        run_one(spec, bin_dir, args.timeout) for spec in specs
    ]
    passes = sum(result[0] for result in results)
    avbd_ir = [
        result[1:]
        for spec, result in zip(specs, results)
        if spec.solver == "avbd"
    ]
    ir_consistent = len(avbd_ir) < 2 or all(
        result == avbd_ir[0] for result in avbd_ir[1:]
    )
    if not ir_consistent:
        print(
            "[REVOLUTE_MOTOR_CONTACT_ERROR] "
            "parallel/sequential compiled-objective fingerprint mismatch"
        )
    passed = passes == len(specs) and ir_consistent
    print(
        f"[REVOLUTE_MOTOR_CONTACT_SUMMARY] mode={args.mode} "
        f"runs={passes}/{len(specs)} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
