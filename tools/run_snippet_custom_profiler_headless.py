#!/usr/bin/env python3
"""Run SnippetCustomProfiler callback and scene-lifecycle coverage headlessly."""

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
EXECUTABLE = "SnippetCustomProfiler_64.exe"
FRAMES = 120


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    repeat: int


def specs(mode: str) -> tuple[RunSpec, ...]:
    lanes = (("tgs", "parallel"),)
    if mode != "authority":
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


def parse_int(
    fields: dict[str, str], key: str, errors: list[str]
) -> int | None:
    try:
        return int(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key}={fields.get(key)!r}, expected integer")
        return None


def parse_float(
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


def run_one(
    spec: RunSpec,
    bin_dir: Path,
    timeout: float,
    frames: int,
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=stack-workload",
        f"--execution={spec.execution}",
        f"--frames={frames}",
        "--dt=0.0166666675",
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

    exact = {
        "schema": "1",
        "snippet": "SnippetCustomProfiler",
        "solver": spec.solver,
        "case": "stack-workload",
        "execution": spec.execution,
        "frames": str(frames),
        "completedFrames": str(frames),
        "profiledFrames": str(frames),
        "callbackInstalled": "1",
        "callbackCleared": "1",
        "invalidNames": "0",
        "initialStaticActors": "1",
        "initialDynamicActors": "276",
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
        "pvd": "0",
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
    }
    for key, expected in exact.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )

    starts = parse_int(fields, "zoneStarts", errors)
    ends = parse_int(fields, "zoneEnds", errors)
    detached_starts = parse_int(fields, "detachedStarts", errors)
    detached_ends = parse_int(fields, "detachedEnds", errors)
    if starts is not None and starts <= frames:
        errors.append(f"zoneStarts={starts}, expected > {frames}")
    if starts is not None and ends is not None and starts != ends:
        errors.append(f"zone start/end mismatch: {starts}/{ends}")
    if (
        detached_starts is not None
        and detached_ends is not None
        and detached_starts != detached_ends
    ):
        errors.append(
            "detached zone start/end mismatch: "
            f"{detached_starts}/{detached_ends}"
        )

    for key in ("integerData", "floatData", "recordFrames"):
        value = parse_int(fields, key, errors)
        if value is not None and value < 0:
            errors.append(f"{key}={value}, expected nonnegative")

    min_y = parse_float(fields, "minBodyY", errors)
    max_speed = parse_float(fields, "maxBodySpeed", errors)
    displacement = parse_float(fields, "projectileDisplacement", errors)
    for key in (
        "finalProjectileX",
        "finalProjectileY",
        "finalProjectileZ",
    ):
        parse_float(fields, key, errors)
    if min_y is not None and abs(min_y) >= 100000.0:
        errors.append(f"minBodyY={min_y}, expected bounded scene")
    if max_speed is not None and not 1.0 < max_speed < 100000.0:
        errors.append(
            f"maxBodySpeed={max_speed}, expected 1 < value < 100000"
        )
    if displacement is not None and not 1.0 < displacement < 100000.0:
        errors.append(
            "projectileDisplacement="
            f"{displacement}, expected 1 < value < 100000"
        )
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")

    print(
        f"[CUSTOM_PROFILER_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            f"[CUSTOM_PROFILER_RUN_ERROR] name={spec.name} error={error}"
        )
    return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    keys = (
        "completedFrames",
        "profiledFrames",
        "callbackInstalled",
        "callbackCleared",
        "detachedStarts",
        "detachedEnds",
        "integerData",
        "floatData",
        "recordFrames",
        "invalidNames",
        "initialStaticActors",
        "initialDynamicActors",
        "nonFinite",
        "fetchFailures",
        "fatalErrors",
        "cleanupComplete",
    )
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        first = results[f"{solver}-{execution}-r1"]
        second = results[f"{solver}-{execution}-r2"]
        mismatches = [key for key in keys if first.get(key) != second.get(key)]
        pair_passed = not mismatches
        passed = passed and pair_passed
        print(
            f"[CUSTOM_PROFILER_REPEAT] pair={solver}-{execution} "
            f"status={'PASS' if pair_passed else 'FAIL'} "
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
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--frames", type=int, default=FRAMES)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[CUSTOM_PROFILER_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0:
        print(
            "[CUSTOM_PROFILER_RUNNER_ERROR] --timeout must be positive"
        )
        return 2
    if args.frames < 60:
        print(
            "[CUSTOM_PROFILER_RUNNER_ERROR] "
            "--frames must be at least 60"
        )
        return 2

    accepted = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(
            spec, bin_dir, args.timeout, args.frames
        )
        accepted = accepted and passed
        results[spec.name] = fields
    if accepted and args.mode == "acceptance":
        accepted = compare_repeats(results)
    print(
        f"[CUSTOM_PROFILER_MATRIX] mode={args.mode} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
