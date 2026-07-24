#!/usr/bin/env python3
"""Run the SnippetSplitFetchResults callback-lifecycle headless matrix."""

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
EXECUTABLE = "SnippetSplitFetchResults_64.exe"
FRAMES = 180


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


def positive_integer(
    fields: dict[str, str], key: str, errors: list[str]
) -> None:
    try:
        if int(fields[key]) <= 0:
            errors.append(f"{key}={fields[key]!r}, expected positive")
    except (KeyError, ValueError):
        errors.append(
            f"{key}={fields.get(key)!r}, expected positive integer"
        )


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=split-callbacks",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
        "--dt=0.0166666675",
        "--dispatcher-threads=4",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(FRAMES)
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
        "snippet": "SnippetSplitFetchResults",
        "solver": spec.solver,
        "case": "split-callbacks",
        "execution": spec.execution,
        "frames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "fetchStartCalls": str(FRAMES),
        "processCallbackCalls": str(FRAMES),
        "fetchFinishCalls": str(FRAMES),
        "continuationReleases": str(FRAMES),
        "phaseViolations": "0",
        "overflowContactPoints": "0",
        "nonFiniteContacts": "0",
        "nonFiniteActors": "0",
        "fetchFailures": "0",
        "dynamicBodies": "750",
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
        "pairHeaders",
        "callbackInvocations",
        "callbackPairs",
        "contactPoints",
        "storedContactPoints",
        "nonzeroImpulses",
    ):
        positive_integer(fields, key, errors)
    for key in ("minBodyY", "maxBodyY", "maxBodySpeed"):
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"{key}={fields.get(key)!r}, expected finite float"
            )

    print(
        f"[SPLIT_FETCH_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[SPLIT_FETCH_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    physical_tolerances = {
        "minBodyY": 0.1,
        "maxBodyY": 0.1,
        "maxBodySpeed": 1.0,
    }
    count_keys = {
        "pairHeaders",
        "callbackInvocations",
        "callbackPairs",
        "contactPoints",
    }
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        first = results[f"{solver}-{execution}-r1"]
        second = results[f"{solver}-{execution}-r2"]
        mismatches: list[str] = []
        for key, tolerance in physical_tolerances.items():
            try:
                if abs(float(first[key]) - float(second[key])) > tolerance:
                    mismatches.append(key)
            except (KeyError, ValueError):
                mismatches.append(key)
        for key in count_keys:
            try:
                a = int(first[key])
                b = int(second[key])
                tolerance = max(8, int(max(a, b) * 0.05))
                if abs(a - b) > tolerance:
                    mismatches.append(key)
            except (KeyError, ValueError):
                mismatches.append(key)
        try:
            a = int(first["nonzeroImpulses"])
            b = int(second["nonzeroImpulses"])
            if abs(a - b) > max(8, int(max(a, b) * 0.10)):
                mismatches.append("nonzeroImpulses")
        except (KeyError, ValueError):
            mismatches.append("nonzeroImpulses")
        pair_ok = not mismatches
        passed = passed and pair_ok
        print(
            f"[SPLIT_FETCH_REPEAT] pair={solver}-{execution} "
            f"status={'PASS' if pair_ok else 'FAIL'} "
            f"mismatches={','.join(mismatches) if mismatches else 'none'}"
        )
    return passed


def compare_execution(results: dict[str, dict[str, str]]) -> bool:
    try:
        parallel = results["avbd-parallel-r1"]
        sequential = results["avbd-sequential-r1"]
        close = (
            abs(float(parallel["minBodyY"]) - float(sequential["minBodyY"]))
            < 0.2
            and abs(
                float(parallel["maxBodyY"])
                - float(sequential["maxBodyY"])
            )
            < 0.2
            and abs(
                float(parallel["maxBodySpeed"])
                - float(sequential["maxBodySpeed"])
            )
            < 2.0
        )
    except (KeyError, ValueError) as exc:
        print(f"[SPLIT_FETCH_EXECUTION_ERROR] error={exc}")
        return False
    print(
        "[SPLIT_FETCH_EXECUTION] "
        f"status={'PASS' if close else 'FAIL'} "
        f"parallelPoints={parallel.get('contactPoints', 'missing')} "
        f"sequentialPoints={sequential.get('contactPoints', 'missing')}"
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
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[SPLIT_FETCH_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0:
        print("[SPLIT_FETCH_RUNNER_ERROR] --timeout must be positive")
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
        f"[SPLIT_FETCH_MATRIX] mode={args.mode} "
        f"infrastructure={'PASS' if infrastructure_ok else 'FAIL'} "
        f"execution={'PASS' if execution_ok else 'FAIL'} "
        f"repeatability={'PASS' if repeat_ok else 'FAIL'} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
