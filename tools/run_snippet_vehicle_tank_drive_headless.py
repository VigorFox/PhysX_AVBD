#!/usr/bin/env python3
"""Run the SnippetVehicleTankDrive official command-cycle headless gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
DEFAULT_VEHICLE_DATA_DIR = (
    REPO_ROOT / "physx" / "snippets" / "media" / "vehicledata"
)
EXECUTABLE = "SnippetVehicleTankDrive_64.exe"
REQUESTED_FRAMES = 1700


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str


def specs(mode: str) -> tuple[RunSpec, ...]:
    if mode == "authority":
        return (RunSpec("tgs-parallel-r1", "tgs", "parallel"),)
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    repeats = (1, 2) if mode == "acceptance" else (1,)
    return tuple(
        RunSpec(f"{solver}-{execution}-r{repeat}", solver, execution)
        for repeat in repeats
        for solver, execution in lanes
    )


def parse_gate(line: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append(f"malformed gate token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate gate key: {key}")
        fields[key] = value
    return fields, errors


def validate_numbers(fields: dict[str, str], errors: list[str]) -> None:
    float_keys = (
        "maxDifferentialRatioError",
        "maxTrackSpeedMismatch",
        "maxEngineSpeed",
        "maxClutchSlip",
        "initialBrakeMaxSpeed",
        "forwardDisplacement",
        "sharpTurnHeading",
        "spotTurnHeading",
        "spotTurnTranslation",
        "gentleTurnHeading",
        "reverseDisplacement",
        "minHeight",
        "maxHeight",
    )
    int_keys = (
        "completedFrames",
        "commands",
        "wheels",
        "constraints",
        "tracks",
        "trackWheels",
        "roadHitFrames",
        "roadHitSamples",
        "driveFrames",
        "brakeFrames",
        "activeConstraintFrames",
        "activeConstraintRows",
        "tankDifferentialFrames",
        "tankConstraintFrames",
        "clutchEngagedFrames",
        "command0Frames",
        "command1Frames",
        "command2Frames",
        "command3Frames",
        "command4Frames",
        "command5Frames",
    )
    for key in float_keys:
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected finite float")
    for key in int_keys:
        try:
            if int(fields[key]) < 0:
                errors.append(f"{key} is negative")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected integer")


def run_one(
    spec: RunSpec,
    bin_dir: Path,
    vehicle_data_dir: Path,
    timeout: float,
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        f"--vehicleDataPath={vehicle_data_dir}",
        "--headless",
        f"--solver={spec.solver}",
        "--case=command-cycle",
        f"--execution={spec.execution}",
        f"--frames={REQUESTED_FRAMES}",
        "--dt=0.0166666675",
        "--dispatcher-threads=4",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(REQUESTED_FRAMES)
    result = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout
    )
    combined = result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr
    lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]
    errors: list[str] = []
    fields: dict[str, str] = {}
    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append(
            "visible window detected: "
            + ", ".join(result.visible_window_titles)
        )
    if len(lines) != 1:
        errors.append(f"gate count is {len(lines)}, expected exactly 1")
    else:
        fields, parse_errors = parse_gate(lines[0])
        errors.extend(parse_errors)
    required = {
        "schema": "1",
        "snippet": "SnippetVehicleTankDrive",
        "solver": spec.solver,
        "case": "command-cycle",
        "execution": spec.execution,
        "frames": str(REQUESTED_FRAMES),
        "commands": "6",
        "wheels": "4",
        "constraints": "1",
        "tracks": "2",
        "trackWheels": "4",
        "nonFinite": "0",
        "drivetrainNonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
        "pvd": "0",
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )
    validate_numbers(fields, errors)
    try:
        completed = int(fields["completedFrames"])
        if not 1550 <= completed <= REQUESTED_FRAMES:
            errors.append(
                f"completedFrames={completed}, expected 1550..{REQUESTED_FRAMES}"
            )
        if int(fields["roadHitFrames"]) <= completed // 2:
            errors.append("road support did not cover a majority of frames")
        if int(fields["tankDifferentialFrames"]) != completed:
            errors.append("tank differential was not valid on every frame")
        if int(fields["tankConstraintFrames"]) != completed:
            errors.append("tank track groups were not valid on every frame")
        for key in (
            "driveFrames",
            "brakeFrames",
            "activeConstraintFrames",
            "activeConstraintRows",
            "clutchEngagedFrames",
            "command0Frames",
            "command1Frames",
            "command2Frames",
            "command3Frames",
            "command4Frames",
            "command5Frames",
        ):
            if int(fields[key]) <= 0:
                errors.append(f"{key} did not witness activity")
    except (KeyError, ValueError):
        pass
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")
    print(
        f"[VEHICLE_TANK_DRIVE_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            f"[VEHICLE_TANK_DRIVE_RUN_ERROR] name={spec.name} error={error}"
        )
    return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    exact_keys = (
        "completedFrames",
        "roadHitFrames",
        "roadHitSamples",
        "driveFrames",
        "brakeFrames",
        "activeConstraintFrames",
        "activeConstraintRows",
        "tankDifferentialFrames",
        "tankConstraintFrames",
        "clutchEngagedFrames",
        "command0Frames",
        "command1Frames",
        "command2Frames",
        "command3Frames",
        "command4Frames",
        "command5Frames",
    )
    numeric_keys = (
        "maxDifferentialRatioError",
        "maxTrackSpeedMismatch",
        "maxEngineSpeed",
        "maxClutchSlip",
        "initialBrakeMaxSpeed",
        "forwardDisplacement",
        "sharpTurnHeading",
        "spotTurnHeading",
        "spotTurnTranslation",
        "gentleTurnHeading",
        "reverseDisplacement",
        "minHeight",
        "maxHeight",
    )
    passed = True
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        first = results[f"{solver}-{execution}-r1"]
        second = results[f"{solver}-{execution}-r2"]
        mismatches = [
            key for key in exact_keys if first.get(key) != second.get(key)
        ]
        for key in numeric_keys:
            try:
                if abs(float(first[key]) - float(second[key])) > 1e-4:
                    mismatches.append(key)
            except (KeyError, ValueError):
                mismatches.append(key)
        pair_ok = not mismatches
        passed = passed and pair_ok
        print(
            "[VEHICLE_TANK_DRIVE_REPEAT] "
            f"pair={solver}-{execution} "
            f"status={'PASS' if pair_ok else 'FAIL'} "
            f"mismatches={','.join(mismatches) if mismatches else 'none'}"
        )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "acceptance"),
        default="probe",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument(
        "--vehicle-data-dir", type=Path, default=DEFAULT_VEHICLE_DATA_DIR
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    vehicle_data_dir = args.vehicle_data_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            f"[VEHICLE_TANK_DRIVE_RUNNER_ERROR] missing executable: "
            f"{bin_dir / EXECUTABLE}"
        )
        return 2
    for filename in ("Base.json", "EngineDrive.json"):
        if not (vehicle_data_dir / filename).is_file():
            print(
                f"[VEHICLE_TANK_DRIVE_RUNNER_ERROR] missing vehicle data: "
                f"{vehicle_data_dir / filename}"
            )
            return 2
    if args.timeout <= 0:
        print("[VEHICLE_TANK_DRIVE_RUNNER_ERROR] --timeout must be positive")
        return 2
    infrastructure_ok = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(
            spec, bin_dir, vehicle_data_dir, args.timeout
        )
        infrastructure_ok = infrastructure_ok and passed
        results[spec.name] = fields
    repeat_ok = (
        compare_repeats(results)
        if infrastructure_ok and args.mode == "acceptance"
        else True
    )
    accepted = infrastructure_ok and repeat_ok
    print(
        f"[VEHICLE_TANK_DRIVE_MATRIX] mode={args.mode} "
        f"infrastructure={'PASS' if infrastructure_ok else 'FAIL'} "
        f"repeatability={'PASS' if repeat_ok else 'FAIL'} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
