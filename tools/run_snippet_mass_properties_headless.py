#!/usr/bin/env python3
"""Run the SnippetMassProperties impulse-response headless matrix."""

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
EXECUTABLE = "SnippetMassProperties_64.exe"


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    repeat: int


def specs(mode: str) -> tuple[RunSpec, ...]:
    if mode == "authority":
        return (RunSpec("tgs-parallel-r1", "tgs", "parallel", 1),)
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


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=impulse-response",
        f"--execution={spec.execution}",
        "--frames=1",
        "--dt=0.0166666675",
        "--dispatcher-threads=4",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = "1"
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
        "snippet": "SnippetMassProperties",
        "solver": spec.solver,
        "case": "impulse-response",
        "execution": spec.execution,
        "frames": "1",
        "completedFrames": "1",
        "bodies": "6",
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
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")
    for key in (
        "massMin",
        "massMax",
        "linearScaleMin",
        "linearScaleMax",
        "angularScaleMin",
        "angularScaleMax",
        "maxLinearResidual",
        "maxAngularResidual",
        "rotatedOmegaX",
        "rotatedOmegaZ",
        "speed0",
        "speed1",
        "speed2",
        "speed3",
        "speed4",
        "speed5",
    ):
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"{key}={fields.get(key)!r}, expected finite float"
            )

    print(
        f"[MASS_PROPERTIES_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[MASS_PROPERTIES_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    keys = (
        "massMin",
        "massMax",
        "linearScaleMin",
        "linearScaleMax",
        "angularScaleMin",
        "angularScaleMax",
        "maxLinearResidual",
        "maxAngularResidual",
        "rotatedOmegaX",
        "rotatedOmegaZ",
        "speed0",
        "speed1",
        "speed2",
        "speed3",
        "speed4",
        "speed5",
    )
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        first = results[f"{solver}-{execution}-r1"]
        second = results[f"{solver}-{execution}-r2"]
        mismatches: list[str] = []
        for key in keys:
            try:
                if abs(float(first[key]) - float(second[key])) > 1e-5:
                    mismatches.append(key)
            except (KeyError, ValueError):
                mismatches.append(key)
        pair_ok = not mismatches
        passed = passed and pair_ok
        print(
            f"[MASS_PROPERTIES_REPEAT] pair={solver}-{execution} "
            f"status={'PASS' if pair_ok else 'FAIL'} "
            f"mismatches={','.join(mismatches) if mismatches else 'none'}"
        )
    return passed


def compare_execution(results: dict[str, dict[str, str]]) -> bool:
    try:
        parallel = results["avbd-parallel-r1"]
        sequential = results["avbd-sequential-r1"]
        keys = (
            "linearScaleMin",
            "linearScaleMax",
            "angularScaleMin",
            "angularScaleMax",
            "rotatedOmegaX",
            "rotatedOmegaZ",
        )
        close = all(
            abs(float(parallel[key]) - float(sequential[key])) < 1e-5
            for key in keys
        )
    except (KeyError, ValueError) as exc:
        print(f"[MASS_PROPERTIES_EXECUTION_ERROR] error={exc}")
        return False
    print(
        "[MASS_PROPERTIES_EXECUTION] "
        f"status={'PASS' if close else 'FAIL'}"
    )
    return close


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "acceptance"),
        default="probe",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[MASS_PROPERTIES_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0:
        print("[MASS_PROPERTIES_RUNNER_ERROR] --timeout must be positive")
        return 2

    infrastructure_ok = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(spec, bin_dir, args.timeout)
        infrastructure_ok = infrastructure_ok and passed
        results[spec.name] = fields
        if not passed:
            break
    execution_ok = (
        compare_execution(results)
        if infrastructure_ok and args.mode != "authority"
        else True
    )
    repeat_ok = (
        compare_repeats(results)
        if infrastructure_ok and args.mode == "acceptance"
        else True
    )
    accepted = infrastructure_ok and execution_ok and repeat_ok
    print(
        f"[MASS_PROPERTIES_MATRIX] mode={args.mode} "
        f"infrastructure={'PASS' if infrastructure_ok else 'FAIL'} "
        f"execution={'PASS' if execution_ok else 'FAIL'} "
        f"repeatability={'PASS' if repeat_ok else 'FAIL'} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
