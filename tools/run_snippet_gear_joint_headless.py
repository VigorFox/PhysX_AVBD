#!/usr/bin/env python3
"""Run the SnippetGearJoint runtime/binary matrix without showing UI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetGearJoint_64.exe"


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    case_name: str
    serialization: str
    extra_args: tuple[str, ...] = ()


def lanes() -> tuple[tuple[str, str, str], ...]:
    return (
        ("tgs-parallel", "tgs", "parallel"),
        ("avbd-parallel", "avbd", "parallel"),
        ("avbd-sequential", "avbd", "sequential"),
    )


def specs() -> list[RunSpec]:
    runtime_variants = (
        ("steady-positive", "steady", ("--ratio=2.5",)),
        ("steady-negative", "steady", ("--ratio=-2.5",)),
        ("unit-ratio", "unit-ratio", ()),
        ("phase-offset", "phase-offset", ()),
        ("reverse", "reverse", ()),
        ("sinusoidal", "sinusoidal", ()),
        ("external", "external-impulse", ()),
    )
    binary_variants = (
        ("steady-positive", "steady", ("--ratio=2.5",)),
        ("steady-negative", "steady", ("--ratio=-2.5",)),
        ("external", "external-impulse", ()),
    )
    result: list[RunSpec] = []
    for serialization, variants in (
        ("runtime", runtime_variants),
        ("binary", binary_variants),
    ):
        for variant_name, case_name, extra_args in variants:
            for lane_name, solver, execution in lanes():
                result.append(
                    RunSpec(
                        f"gear-{variant_name}-{serialization}-{lane_name}",
                        solver,
                        execution,
                        case_name,
                        serialization,
                        extra_args,
                    )
                )
    return result


def parse_authority(line: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append(f"malformed authority token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate authority key: {key}")
        fields[key] = value
    return fields, errors


def require_fields(
    fields: dict[str, str],
    expected: dict[str, str],
    errors: list[str],
) -> None:
    for key, value in expected.items():
        actual = fields.get(key)
        if actual != value:
            errors.append(f"{key}={actual!r}, expected {value!r}")


def require_positive_integer(
    fields: dict[str, str], key: str, errors: list[str]
) -> None:
    value = fields.get(key)
    try:
        positive = value is not None and int(value) > 0
    except ValueError:
        positive = False
    if not positive:
        errors.append(f"{key}={value!r}, expected positive integer")


def run_one(
    spec: RunSpec, mode: str, bin_dir: Path, timeout_seconds: float
) -> tuple[bool, dict[str, str]]:
    executable = bin_dir / EXECUTABLE
    argv = [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        "--dt=0.0166666675",
        "--seed=1",
        "--dispatcher-threads=2",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        "--frames=1200",
        f"--serialization={spec.serialization}",
        *spec.extra_args,
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    result = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout_seconds
    )

    combined = result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr
    authority_lines = [
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
    if len(authority_lines) != 1:
        errors.append(
            f"authority count is {len(authority_lines)}, expected exactly 1"
        )
    else:
        fields, parse_errors = parse_authority(authority_lines[0])
        errors.extend(parse_errors)

    require_fields(
        fields,
        {
            "schema": "1",
            "snippet": "SnippetGearJoint",
            "case": spec.case_name,
            "solver": spec.solver,
            "execution": spec.execution,
            "requestedFrames": "1200",
            "serialization": spec.serialization,
            "capability": "PARTIAL",
            "validation": "PROBE",
            "cleanupComplete": "1",
            "pvd": "0",
        },
        errors,
    )

    status = fields.get("status")
    successful_physics = status == "PASS"
    probe_failure_allowed = (
        mode == "probe"
        and spec.serialization == "binary"
        and status == "FAIL"
    )
    if successful_physics:
        require_fields(
            fields,
            {
                "reason": "none",
                "completedFrames": "1200",
                "seed": "1",
                "dispatcherThreads": "2",
                "nonFinite": "0",
                "physicsErrors": "0",
                "physicsWarnings": "0",
                "fetchFailures": "0",
                "fetchErrorState": "0",
                "topologyOk": "1",
                "dynamicActors": "2",
                "staticActors": "0",
                "constraints": "3",
                "shapes0": "5",
                "shapes1": "2",
                "dependencyIdentity": "1",
                "actorIdentity": "1",
            },
            errors,
        )
        if spec.case_name == "external-impulse":
            require_fields(
                fields,
                {
                    "impulseEvents": "1",
                    "driveEnabledReadback": "0",
                    "impulseResponseSamples": "4",
                },
                errors,
            )
    elif not probe_failure_allowed:
        errors.append(
            f"status={status!r}, expected 'PASS'"
            + (" or classified binary FAIL" if mode == "probe" else "")
        )

    if spec.serialization == "binary" and successful_physics:
        require_fields(
            fields,
            {
                "serializationRequested": "1",
                "registryCreated": "1",
                "collectionCompleted": "1",
                "serializable": "1",
                "serializeSuccess": "1",
                "binaryBlockAllocated": "1",
                "binaryAligned": "1",
                "deserializeSuccess": "1",
                "loadedActors": "2",
                "loadedConstraints": "3",
                "loadedRevolute": "2",
                "loadedGear": "1",
                "authoringReleased": "1",
                "loadedCollectionReleased": "1",
                "binaryBlockFreed": "1",
            },
            errors,
        )
        require_positive_integer(fields, "serializedBytes", errors)
        require_positive_integer(fields, "loadedObjects", errors)
    elif spec.serialization == "runtime" and successful_physics:
        require_fields(
            fields,
            {
                "serializationRequested": "0",
                "registryCreated": "0",
                "deserializeSuccess": "0",
                "loadedCollectionReleased": "0",
                "binaryBlockFreed": "0",
            },
            errors,
        )

    expected_exit = 1 if probe_failure_allowed else 0
    if result.returncode != expected_exit:
        errors.append(
            f"exit code {result.returncode}, expected {expected_exit}"
        )
    if result.stderr:
        errors.append("stderr is not empty")

    print(
        f"[GEAR_JOINT_RUN] name={spec.name} "
        f"status={status or 'MISSING'} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[GEAR_JOINT_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("probe", "acceptance"), default="probe"
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[GEAR_JOINT_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0.0:
        print("[GEAR_JOINT_RUNNER_ERROR] --timeout must be positive")
        return 2

    all_runner_passed = True
    physics_passes = 0
    physics_failures = 0
    runtime_passes = 0
    binary_passes = 0
    for spec in specs():
        passed, fields = run_one(spec, args.mode, bin_dir, args.timeout)
        all_runner_passed = all_runner_passed and passed
        physics_passes += fields.get("status") == "PASS"
        physics_failures += fields.get("status") == "FAIL"
        if fields.get("status") == "PASS":
            runtime_passes += spec.serialization == "runtime"
            binary_passes += spec.serialization == "binary"

    print(
        f"[GEAR_JOINT_MATRIX] mode={args.mode} "
        f"physicsPasses={physics_passes} physicsFailures={physics_failures} "
        f"runtimePasses={runtime_passes} binaryPasses={binary_passes} "
        f"status={'PASS' if all_runner_passed else 'FAIL'}"
    )
    return 0 if all_runner_passed else 1


if __name__ == "__main__":
    sys.exit(main())
