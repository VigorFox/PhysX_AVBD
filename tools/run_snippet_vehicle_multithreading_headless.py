#!/usr/bin/env python3
"""Run the official 1024-vehicle task graph as a hidden correctness gate."""

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
EXECUTABLE = "SnippetVehicleMultithreading_64.exe"
REQUESTED_FRAMES = 300
VEHICLES = 1024
WORKER_PARTITIONS = 4
UPDATES_PER_PARTITION_PER_FRAME = VEHICLES // WORKER_PARTITIONS


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    repeat: int


def specs(mode: str) -> tuple[RunSpec, ...]:
    if mode == "authority":
        return (
            RunSpec("tgs-parallel-r1", "tgs", "parallel", 1),
            RunSpec("tgs-parallel-r2", "tgs", "parallel", 2),
        )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    repeats = (1, 2) if mode == "acceptance" else (1,)
    return tuple(
        RunSpec(f"{solver}-{execution}-r{repeat}", solver, execution, repeat)
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


FLOAT_KEYS = (
    "maxTireLongForce",
    "maxTireLatForce",
    "minFinalForwardDisplacement",
    "maxFinalForwardDisplacement",
    "maxLateralDrift",
    "minHeight",
    "maxHeight",
    "maxLinearSpeed",
    "maxAngularSpeed",
)
INT_KEYS = (
    "completedFrames",
    "commands",
    "vehicles",
    "wheels",
    "constraints",
    "workerPartitions",
    "updateBatchSize",
    "substeps",
    "beginComponentCalls",
    "endComponentCalls",
    "allTaskPartitionsCompleteFrames",
    "continuationCompleteFrames",
    "taskRuns",
    "taskRuns0",
    "taskRuns1",
    "taskRuns2",
    "taskRuns3",
    "taskVehicleUpdates",
    "taskVehicleUpdates0",
    "taskVehicleUpdates1",
    "taskVehicleUpdates2",
    "taskVehicleUpdates3",
    "offMainTaskRuns",
    "waitTaskRuns",
    "waitTaskReleases",
    "roadHitVehicleFrames",
    "roadHitSamples",
    "driveVehicleFrames",
    "tireForceVehicleFrames",
    "nonZeroLongForceSamples",
    "nonZeroLatForceSamples",
    "tireForceNonFinite",
    "activeConstraintVehicleFrames",
    "activeConstraintRows",
)


def validate_numbers(fields: dict[str, str], errors: list[str]) -> None:
    for key in FLOAT_KEYS:
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected finite float")
    for key in INT_KEYS:
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
        "--case=task-graph",
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
        "snippet": "SnippetVehicleMultithreading",
        "solver": spec.solver,
        "case": "task-graph",
        "execution": spec.execution,
        "frames": str(REQUESTED_FRAMES),
        "commands": "1",
        "vehicles": str(VEHICLES),
        "wheels": str(VEHICLES * 4),
        "constraints": str(VEHICLES),
        "workerPartitions": str(WORKER_PARTITIONS),
        "updateBatchSize": "1",
        "substeps": "1",
        "tireForceNonFinite": "0",
        "nonFinite": "0",
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
        if not 250 <= completed <= REQUESTED_FRAMES:
            errors.append(
                f"completedFrames={completed}, expected 250..{REQUESTED_FRAMES}"
            )
        expected_component_calls = completed * VEHICLES
        for key in ("beginComponentCalls", "endComponentCalls"):
            if int(fields[key]) != expected_component_calls:
                errors.append(
                    f"{key}={fields[key]}, expected {expected_component_calls}"
                )
        for key in (
            "allTaskPartitionsCompleteFrames",
            "continuationCompleteFrames",
            "waitTaskRuns",
            "waitTaskReleases",
        ):
            if int(fields[key]) != completed:
                errors.append(f"{key}={fields[key]}, expected {completed}")
        if int(fields["taskRuns"]) != completed * WORKER_PARTITIONS:
            errors.append("aggregate task run count is incomplete")
        if int(fields["taskVehicleUpdates"]) != expected_component_calls:
            errors.append("aggregate vehicle update count is incomplete")
        for partition in range(WORKER_PARTITIONS):
            if int(fields[f"taskRuns{partition}"]) != completed:
                errors.append(f"partition {partition} task run count is incomplete")
            expected_updates = completed * UPDATES_PER_PARTITION_PER_FRAME
            if int(fields[f"taskVehicleUpdates{partition}"]) != expected_updates:
                errors.append(
                    f"partition {partition} vehicle updates are incomplete"
                )
        vehicle_frames = completed * VEHICLES
        for key in (
            "roadHitVehicleFrames",
            "driveVehicleFrames",
            "tireForceVehicleFrames",
        ):
            if int(fields[key]) <= vehicle_frames // 2:
                errors.append(f"{key} did not cover a majority of vehicle frames")
        if int(fields["offMainTaskRuns"]) <= 0:
            errors.append("no vehicle task ran off the main thread")
        if int(fields["nonZeroLongForceSamples"]) <= 0:
            errors.append("no longitudinal tire-force activity")
        if float(fields["maxTireLongForce"]) <= 1.0:
            errors.append("longitudinal tire force is missing")
        if float(fields["minFinalForwardDisplacement"]) <= 1.0:
            errors.append("not every vehicle produced forward rigid response")
        if float(fields["maxLinearSpeed"]) <= 1.0:
            errors.append("vehicle speed response is missing")
    except (KeyError, ValueError):
        pass
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")
    print(
        f"[VEHICLE_MULTITHREADING_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            f"[VEHICLE_MULTITHREADING_RUN_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, fields


def compare_repeats(
    results: dict[str, dict[str, str]], lanes: tuple[tuple[str, str], ...]
) -> bool:
    passed = True
    for solver, execution in lanes:
        first = results[f"{solver}-{execution}-r1"]
        second = results[f"{solver}-{execution}-r2"]
        mismatches = [
            key for key in INT_KEYS if first.get(key) != second.get(key)
        ]
        for key in FLOAT_KEYS:
            try:
                if abs(float(first[key]) - float(second[key])) > 1e-4:
                    mismatches.append(key)
            except (KeyError, ValueError):
                mismatches.append(key)
        pair_ok = not mismatches
        passed = passed and pair_ok
        print(
            "[VEHICLE_MULTITHREADING_REPEAT] "
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
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    vehicle_data_dir = args.vehicle_data_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            f"[VEHICLE_MULTITHREADING_RUNNER_ERROR] missing executable: "
            f"{bin_dir / EXECUTABLE}"
        )
        return 2
    for filename in ("Base.json", "DirectDrive.json"):
        if not (vehicle_data_dir / filename).is_file():
            print(
                f"[VEHICLE_MULTITHREADING_RUNNER_ERROR] "
                f"missing vehicle data: {vehicle_data_dir / filename}"
            )
            return 2

    results: dict[str, dict[str, str]] = {}
    passed = True
    for spec in specs(args.mode):
        run_passed, fields = run_one(
            spec, bin_dir, vehicle_data_dir, args.timeout
        )
        passed = passed and run_passed
        results[spec.name] = fields

    if args.mode == "authority":
        passed = compare_repeats(
            results, (("tgs", "parallel"),)
        ) and passed
    elif args.mode == "acceptance":
        passed = compare_repeats(
            results,
            (
                ("tgs", "parallel"),
                ("avbd", "parallel"),
                ("avbd", "sequential"),
            ),
        ) and passed
    print(
        "[VEHICLE_MULTITHREADING_SUMMARY] "
        f"mode={args.mode} runs={len(results)} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
