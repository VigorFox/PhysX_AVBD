#!/usr/bin/env python3
"""Run the SnippetTriggers 9-scenario headless semantics matrix."""

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
EXECUTABLE = "SnippetTriggers_64.exe"
FRAMES = 240


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
        "--case=trigger-matrix",
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
        "snippet": "SnippetTriggers",
        "solver": spec.solver,
        "case": "trigger-matrix",
        "execution": spec.execution,
        "frames": str(FRAMES),
        "scenarios": "9",
        "passedScenarios": "9",
        "negativeControlPasses": "2",
        "triggerTriggerScenarios": "2",
        "overlapActuationScenarios": "3",
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
    for key, minimum in (
        ("foundScenarios", 7),
        ("lostScenarios", 7),
        ("removalScenarios", 2),
        ("totalFound", 7),
        ("totalLost", 7),
        ("objectRemoved", 2),
    ):
        try:
            if int(fields[key]) < minimum:
                errors.append(f"{key}={fields[key]}, expected >= {minimum}")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected integer")
    try:
        distance = float(fields["maxPositiveTriggerMinDistance"])
        if not math.isfinite(distance) or distance >= 8.0:
            errors.append(
                "maxPositiveTriggerMinDistance="
                f"{distance}, expected finite < 8"
            )
    except (KeyError, ValueError):
        errors.append(
            "maxPositiveTriggerMinDistance="
            f"{fields.get('maxPositiveTriggerMinDistance')!r}, "
            "expected float"
        )
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")

    print(
        f"[TRIGGERS_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[TRIGGERS_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    keys = (
        "passedScenarios",
        "foundScenarios",
        "lostScenarios",
        "negativeControlPasses",
        "removalScenarios",
        "triggerTriggerScenarios",
        "overlapActuationScenarios",
        "totalFound",
        "totalLost",
        "objectRemoved",
        "maxPositiveTriggerMinDistance",
    )
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        first = results[f"{solver}-{execution}-r1"]
        second = results[f"{solver}-{execution}-r2"]
        mismatches = [key for key in keys if first.get(key) != second.get(key)]
        pair_ok = not mismatches
        passed = passed and pair_ok
        print(
            f"[TRIGGERS_REPEAT] pair={solver}-{execution} "
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
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(f"[TRIGGERS_RUNNER_ERROR] missing executable: {bin_dir / EXECUTABLE}")
        return 2
    if args.timeout <= 0:
        print("[TRIGGERS_RUNNER_ERROR] --timeout must be positive")
        return 2

    accepted = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(spec, bin_dir, args.timeout)
        accepted = accepted and passed
        results[spec.name] = fields
    if accepted and args.mode == "acceptance":
        accepted = compare_repeats(results)
    print(
        f"[TRIGGERS_MATRIX] mode={args.mode} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
