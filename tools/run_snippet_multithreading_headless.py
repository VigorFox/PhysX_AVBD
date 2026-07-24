#!/usr/bin/env python3
"""Run the SnippetMultiThreading concurrent-query headless matrix."""

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
EXECUTABLE = "SnippetMultiThreading_64.exe"
FRAMES = 120
CYCLES = 2
QUERY_THREADS = 4
DISPATCHER_THREADS = 4
RAYS_PER_FRAME = 1024


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
        "--case=concurrent-query",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
        f"--cycles={CYCLES}",
        f"--query-threads={QUERY_THREADS}",
        "--dt=0.0166666675",
        f"--dispatcher-threads={DISPATCHER_THREADS}",
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

    total_frames = FRAMES * CYCLES
    total_rays = total_frames * RAYS_PER_FRAME
    required = {
        "schema": "1",
        "snippet": "SnippetMultiThreading",
        "solver": spec.solver,
        "case": "concurrent-query",
        "execution": spec.execution,
        "frames": str(FRAMES),
        "cycles": str(CYCLES),
        "queryThreads": str(QUERY_THREADS),
        "dispatcherThreads": str(DISPATCHER_THREADS),
        "completedFrames": str(total_frames),
        "completedCycles": str(CYCLES),
        "raycastBatches": str(total_frames),
        "raysExpected": str(total_rays),
        "raysCompleted": str(total_rays),
        "dynamicBodies": str(275 * CYCLES),
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": str(CYCLES),
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

    for key in ("minBodyY", "maxBodyY", "maxBodySpeed"):
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"{key}={fields.get(key)!r}, expected finite float"
            )
    try:
        if int(fields["rayHits"]) <= 0:
            errors.append(
                f"rayHits={fields['rayHits']!r}, expected positive"
            )
    except (KeyError, ValueError):
        errors.append(
            f"rayHits={fields.get('rayHits')!r}, expected positive integer"
        )

    print(
        f"[MULTITHREADING_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            f"[MULTITHREADING_RUN_ERROR] name={spec.name} error={error}"
        )
    return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    tolerances = {
        "minBodyY": 0.05,
        "maxBodyY": 0.05,
        "maxBodySpeed": 0.5,
    }
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        first = results[f"{solver}-{execution}-r1"]
        second = results[f"{solver}-{execution}-r2"]
        mismatches: list[str] = []
        for key, tolerance in tolerances.items():
            try:
                if abs(float(first[key]) - float(second[key])) > tolerance:
                    mismatches.append(key)
            except (KeyError, ValueError):
                mismatches.append(key)
        try:
            first_hits = int(first["rayHits"])
            second_hits = int(second["rayHits"])
            if abs(first_hits - second_hits) > total_hit_tolerance(
                first_hits, second_hits
            ):
                mismatches.append("rayHits")
        except (KeyError, ValueError):
            mismatches.append("rayHits")
        pair_ok = not mismatches
        passed = passed and pair_ok
        print(
            f"[MULTITHREADING_REPEAT] pair={solver}-{execution} "
            f"status={'PASS' if pair_ok else 'FAIL'} "
            f"mismatches={','.join(mismatches) if mismatches else 'none'}"
        )
    return passed


def total_hit_tolerance(first: int, second: int) -> int:
    return max(16, int(max(first, second) * 0.001))


def compare_execution(results: dict[str, dict[str, str]]) -> bool:
    try:
        parallel = results["avbd-parallel-r1"]
        sequential = results["avbd-sequential-r1"]
        close = (
            abs(float(parallel["minBodyY"]) - float(sequential["minBodyY"]))
            < 0.1
            and abs(
                float(parallel["maxBodyY"])
                - float(sequential["maxBodyY"])
            )
            < 0.1
            and abs(
                float(parallel["maxBodySpeed"])
                - float(sequential["maxBodySpeed"])
            )
            < 1.0
        )
    except (KeyError, ValueError) as exc:
        print(f"[MULTITHREADING_EXECUTION_ERROR] error={exc}")
        return False
    print(
        "[MULTITHREADING_EXECUTION] "
        f"status={'PASS' if close else 'FAIL'} "
        f"parallelHits={parallel.get('rayHits', 'missing')} "
        f"sequentialHits={sequential.get('rayHits', 'missing')}"
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
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[MULTITHREADING_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0:
        print("[MULTITHREADING_RUNNER_ERROR] --timeout must be positive")
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
        f"[MULTITHREADING_MATRIX] mode={args.mode} "
        f"infrastructure={'PASS' if infrastructure_ok else 'FAIL'} "
        f"execution={'PASS' if execution_ok else 'FAIL'} "
        f"repeatability={'PASS' if repeat_ok else 'FAIL'} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
