#!/usr/bin/env python3
"""Run the SnippetRackJoint probe or acceptance matrix without showing UI."""

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
EXECUTABLE = "SnippetRackJoint_64.exe"


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    case_name: str
    ratio: str
    serialization: str


def specs() -> list[RunSpec]:
    result: list[RunSpec] = []
    for case_name in ("pinion-impulse", "rack-impulse"):
        for ratio in ("2", "-2"):
            result.append(
                RunSpec(
                    f"{case_name}-ratio-{ratio}-tgs",
                    "tgs",
                    "parallel",
                    case_name,
                    ratio,
                    "runtime",
                )
            )
            for execution in ("parallel", "sequential"):
                result.append(
                    RunSpec(
                        f"{case_name}-ratio-{ratio}-avbd-{execution}",
                        "avbd",
                        execution,
                        case_name,
                        ratio,
                        "runtime",
                    )
                )
    for case_name, ratio in (
        ("pinion-impulse", "2"),
        ("rack-impulse", "-2"),
    ):
        result.append(
            RunSpec(
                f"{case_name}-ratio-{ratio}-binary-tgs",
                "tgs",
                "parallel",
                case_name,
                ratio,
                "binary",
            )
        )
        for execution in ("parallel", "sequential"):
            result.append(
                RunSpec(
                    f"{case_name}-ratio-{ratio}-binary-avbd-{execution}",
                    "avbd",
                    execution,
                    case_name,
                    ratio,
                    "binary",
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


def run_one(
    spec: RunSpec, mode: str, bin_dir: Path, timeout_seconds: float
) -> tuple[bool, dict[str, str]]:
    executable = bin_dir / EXECUTABLE
    argv = [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        "--frames=240",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
        f"--ratio={spec.ratio}",
        "--impulse-frame=30",
        f"--serialization={spec.serialization}",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = "240"
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

    required = {
        "schema": "1",
        "snippet": "SnippetRackJoint",
        "solver": spec.solver,
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": "240",
        "serialization": spec.serialization,
        "ratio": spec.ratio,
        "impulseFrame": "30",
        "cleanupComplete": "1",
        "pvd": "0",
    }
    for key, expected in required.items():
        actual = fields.get(key)
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")

    successful_physics = fields.get("status") == "PASS"
    if successful_physics or mode == "acceptance":
        successful_required = {
            "completedFrames": "240",
            "impulseEvents": "1",
            "responseSamples": "60",
            "projectionSamples": "1",
            "nonFinite": "0",
            "fetchFailures": "0",
            "fatalErrors": "0",
            "dependencyIdentity": "1",
            "actorIdentity": "1",
            "sceneActors": "2",
            "sceneConstraints": "3",
        }
        for key, expected in successful_required.items():
            actual = fields.get(key)
            if actual != expected:
                errors.append(f"{key}={actual!r}, expected {expected!r}")

    if spec.serialization == "binary" and (
        successful_physics or mode == "acceptance"
    ):
        serialization_required = {
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
            "loadedRevolute": "1",
            "loadedPrismatic": "1",
            "loadedRack": "1",
            "authoringReleased": "1",
            "loadedCollectionReleased": "1",
            "binaryBlockFreed": "1",
        }
        for key, expected in serialization_required.items():
            actual = fields.get(key)
            if actual != expected:
                errors.append(f"{key}={actual!r}, expected {expected!r}")
        for key in ("serializedBytes", "loadedObjects"):
            value = fields.get(key)
            try:
                positive = value is not None and int(value) > 0
            except ValueError:
                positive = False
            if not positive:
                errors.append(f"{key}={value!r}, expected positive integer")
    elif spec.serialization == "runtime" and successful_physics:
        runtime_required = {
            "serializationRequested": "0",
            "registryCreated": "0",
            "deserializeSuccess": "0",
            "loadedCollectionReleased": "0",
            "binaryBlockFreed": "0",
        }
        for key, expected in runtime_required.items():
            actual = fields.get(key)
            if actual != expected:
                errors.append(f"{key}={actual!r}, expected {expected!r}")

    if mode == "acceptance":
        if result.returncode != 0:
            errors.append(f"exit code {result.returncode}, expected 0")
        if fields.get("status") != "PASS":
            errors.append(
                f"status={fields.get('status')!r}, expected 'PASS'"
            )
        if fields.get("reason") != "none":
            errors.append(
                f"reason={fields.get('reason')!r}, expected 'none'"
            )
    elif result.returncode not in (0, 1):
        errors.append(
            f"probe exit code {result.returncode}, expected physics result 0/1"
        )
    if fields.get("status") not in ("PASS", "FAIL"):
        errors.append(
            f"status={fields.get('status')!r}, expected PASS or FAIL"
        )

    print(
        f"[RACK_JOINT_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[RACK_JOINT_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("probe", "acceptance"), default="probe"
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[RACK_JOINT_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0.0:
        print("[RACK_JOINT_RUNNER_ERROR] --timeout must be positive")
        return 2

    all_infrastructure_passed = True
    physics_passes = 0
    physics_failures = 0
    for spec in specs():
        passed, fields = run_one(spec, args.mode, bin_dir, args.timeout)
        all_infrastructure_passed = all_infrastructure_passed and passed
        physics_passes += fields.get("status") == "PASS"
        physics_failures += fields.get("status") == "FAIL"

    print(
        f"[RACK_JOINT_MATRIX] mode={args.mode} "
        f"physicsPasses={physics_passes} physicsFailures={physics_failures} "
        f"status={'PASS' if all_infrastructure_passed else 'FAIL'}"
    )
    return 0 if all_infrastructure_passed else 1


if __name__ == "__main__":
    sys.exit(main())
