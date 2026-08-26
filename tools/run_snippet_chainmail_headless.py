#!/usr/bin/env python3
"""Run the large-island SnippetChainmail headless matrix."""

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
EXECUTABLE = "SnippetChainmail_64.exe"
FRAMES = 600


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    case: str
    execution: str
    repeat: int


def specs(mode: str) -> tuple[RunSpec, ...]:
    if mode == "authority":
        return (
            RunSpec("tgs-parallel-r1", "tgs", "impact", "parallel", 1),
        )
    base = (
        ("tgs", "impact", "parallel"),
        ("avbd", "impact", "parallel"),
        ("avbd", "impact", "sequential"),
        ("avbd", "projectile", "parallel"),
        ("avbd", "projectile", "sequential"),
    )
    repeats = (1, 2) if mode == "acceptance" else (1,)
    return tuple(
        RunSpec(
            f"{solver}-{case}-{execution}-r{repeat}",
            solver,
            case,
            execution,
            repeat,
        )
        for repeat in repeats
        for solver, case, execution in base
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
        f"--case={spec.case}",
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
        "snippet": "SnippetChainmail",
        "solver": spec.solver,
        "case": spec.case,
        "execution": spec.execution,
        "frames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
        "gridBodies": "900",
        "jointCount": "1740",
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
        "pvd": "0",
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")

    for key in (
        "initialBallY",
        "finalBallY",
        "minBallY",
        "maxBallSpeed",
        "finalCenterY",
        "minNetY",
        "maxNetSpeed",
        "maxAnchorError",
        "maxCornerDrift",
    ):
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"{key}={fields.get(key)!r}, expected finite float"
            )
    for key in (
        "callbackCount",
        "pairCount",
        "pointCount",
        "ballNetPairs",
        "ballNetPoints",
        "movingNetBodies",
    ):
        try:
            if int(fields[key]) <= 0:
                errors.append(f"{key}={fields[key]!r}, expected positive")
        except (KeyError, ValueError):
            errors.append(
                f"{key}={fields.get(key)!r}, expected positive integer"
            )

    print(
        f"[CHAINMAIL_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[CHAINMAIL_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def compare_repeats(
    results: dict[str, dict[str, str]]
) -> bool:
    passed = True
    tolerances = {
        "finalBallY": 0.25,
        "minBallY": 0.25,
        "maxBallSpeed": 0.1,
        "finalCenterY": 0.25,
        "minNetY": 0.25,
        "maxNetSpeed": 5.0,
        "maxAnchorError": 0.05,
    }
    for solver, case, execution in (
        ("tgs", "impact", "parallel"),
        ("avbd", "impact", "parallel"),
        ("avbd", "impact", "sequential"),
        ("avbd", "projectile", "parallel"),
        ("avbd", "projectile", "sequential"),
    ):
        first = results[f"{solver}-{case}-{execution}-r1"]
        second = results[f"{solver}-{case}-{execution}-r2"]
        mismatches = []
        for key, tolerance in tolerances.items():
            try:
                if abs(float(first[key]) - float(second[key])) > tolerance:
                    mismatches.append(key)
            except (KeyError, ValueError):
                mismatches.append(key)
        pair_ok = not mismatches
        passed = passed and pair_ok
        print(
            f"[CHAINMAIL_REPEAT] pair={solver}-{case}-{execution} "
            f"status={'PASS' if pair_ok else 'FAIL'} "
            f"mismatches={','.join(mismatches) if mismatches else 'none'}"
        )
    return passed


def compare_physics(results: dict[str, dict[str, str]]) -> bool:
    try:
        tgs = results["tgs-impact-parallel-r1"]
        avbd_parallel = results["avbd-impact-parallel-r1"]
        avbd_sequential = results["avbd-impact-sequential-r1"]
        tgs_fell_through = (
            float(tgs["finalBallY"]) < 3.0
            and float(tgs["minBallY"]) < 3.0
        )
        avbd_caught = all(
            float(fields["finalBallY"]) > 20.0
            and float(fields["minBallY"]) > 20.0
            and float(fields["maxAnchorError"]) < 1.0
            for fields in (avbd_parallel, avbd_sequential)
        )
        execution_close = (
            abs(
                float(avbd_parallel["finalBallY"])
                - float(avbd_sequential["finalBallY"])
            )
            < 0.1
            and abs(
                float(avbd_parallel["minBallY"])
                - float(avbd_sequential["minBallY"])
            )
            < 0.2
            and abs(
                float(avbd_parallel["maxAnchorError"])
                - float(avbd_sequential["maxAnchorError"])
            )
            < 0.05
        )
    except (KeyError, ValueError) as exc:
        print(f"[CHAINMAIL_PHYSICS_ERROR] error={exc}")
        return False
    passed = tgs_fell_through and avbd_caught and execution_close
    print(
        "[CHAINMAIL_PHYSICS] "
        f"status={'PASS' if passed else 'FAIL'} "
        f"tgsFellThrough={int(tgs_fell_through)} "
        f"avbdCaught={int(avbd_caught)} "
        f"executionClose={int(execution_close)}"
    )
    return passed


def compare_projectile(results: dict[str, dict[str, str]]) -> bool:
    try:
        parallel = results["avbd-projectile-parallel-r1"]
        sequential = results["avbd-projectile-sequential-r1"]
        caught = all(
            float(fields["finalBallY"]) > 20.0
            and float(fields["minBallY"]) > 20.0
            and float(fields["maxAnchorError"]) < 1.0
            and int(fields["ballNetPoints"]) > 0
            for fields in (parallel, sequential)
        )
        execution_close = (
            abs(
                float(parallel["finalBallY"])
                - float(sequential["finalBallY"])
            )
            < 0.1
            and abs(
                float(parallel["minBallY"])
                - float(sequential["minBallY"])
            )
            < 0.2
            and abs(
                float(parallel["maxAnchorError"])
                - float(sequential["maxAnchorError"])
            )
            < 0.05
        )
    except (KeyError, ValueError) as exc:
        print(f"[CHAINMAIL_PROJECTILE_ERROR] error={exc}")
        return False
    passed = caught and execution_close
    print(
        "[CHAINMAIL_PROJECTILE] "
        f"status={'PASS' if passed else 'FAIL'} "
        f"caught={int(caught)} executionClose={int(execution_close)}"
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
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[CHAINMAIL_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0:
        print("[CHAINMAIL_RUNNER_ERROR] --timeout must be positive")
        return 2

    infrastructure_ok = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(spec, bin_dir, args.timeout)
        infrastructure_ok = infrastructure_ok and passed
        results[spec.name] = fields
        if not passed:
            break

    physics_ok = (
        compare_physics(results) and compare_projectile(results)
        if infrastructure_ok and args.mode != "authority"
        else True
    )
    repeat_ok = (
        compare_repeats(results)
        if infrastructure_ok and args.mode == "acceptance"
        else True
    )
    accepted = infrastructure_ok and physics_ok and repeat_ok
    print(
        f"[CHAINMAIL_MATRIX] mode={args.mode} "
        f"infrastructure={'PASS' if infrastructure_ok else 'FAIL'} "
        f"physics={'PASS' if physics_ok else 'FAIL'} "
        f"repeatability={'PASS' if repeat_ok else 'FAIL'} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
