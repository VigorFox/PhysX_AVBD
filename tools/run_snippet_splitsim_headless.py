#!/usr/bin/env python3
"""Run normal and split-simulation authority/AVBD headless coverage."""

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
EXECUTABLE = "SnippetSplitSim_64.exe"
DEFAULT_FRAMES = 240


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    case_name: str


def specs(mode: str) -> tuple[RunSpec, ...]:
    authority = tuple(
        RunSpec(f"tgs-{case}", "tgs", "parallel", case)
        for case in ("simulate", "split")
    )
    if mode == "authority":
        return authority
    return authority + tuple(
        RunSpec(
            f"avbd-{execution}-{case}",
            "avbd",
            execution,
            case,
        )
        for execution in ("parallel", "sequential")
        for case in ("simulate", "split")
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


def number(fields: dict[str, str], key: str) -> float:
    return float(fields[key])


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float, frames: int
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        f"--frames={frames}",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(frames)
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

    split = spec.case_name == "split"
    required = {
        "schema": "1",
        "snippet": "SnippetSplitSim",
        "solver": spec.solver,
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": str(frames),
        "completedFrames": str(frames),
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
        "dynamicBodies": "32",
        "simulateCalls": "0" if split else str(frames),
        "collideCalls": str(frames) if split else "0",
        "fetchCollisionCalls": str(frames) if split else "0",
        "advanceCalls": str(frames) if split else "0",
        "fetchResultsCalls": str(frames),
        "fetchFailures": "0",
        "nonFinite": "0",
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

    numeric_keys = (
        "maxCollisionPhasePoseDelta",
        "maxTargetPositionError",
        "sumX",
        "sumY",
        "sumZ",
        "minY",
        "maxY",
        "sumSpeed",
        "maxSpeed",
    )
    for key in numeric_keys:
        try:
            value = number(fields, key)
            if not math.isfinite(value):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"{key}={fields.get(key)!r}, expected finite float"
            )
    for key in ("callbackCount", "pairCount", "pointCount", "movingBodies"):
        try:
            if int(fields[key]) <= 0:
                errors.append(f"{key}={fields[key]!r}, expected positive")
        except (KeyError, ValueError):
            errors.append(
                f"{key}={fields.get(key)!r}, expected positive integer"
            )

    print(
        f"[SPLITSIM_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[SPLITSIM_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def compare_pair(
    label: str,
    simulate: dict[str, str],
    split: dict[str, str],
) -> bool:
    try:
        keys = ("sumX", "sumY", "sumZ", "minY", "maxY", "sumSpeed")
        deltas = {
            key: abs(number(simulate, key) - number(split, key))
            for key in keys
        }
        max_delta = max(deltas.values())
    except (KeyError, ValueError) as exc:
        print(f"[SPLITSIM_PAIR_ERROR] pair={label} error={exc}")
        return False

    passed = max_delta <= 1e-3
    print(
        f"[SPLITSIM_PAIR] pair={label} "
        f"status={'PASS' if passed else 'FAIL'} "
        f"reason={'none' if passed else 'split_not_equivalent'} "
        f"sumXDelta={deltas['sumX']:.9g} "
        f"sumYDelta={deltas['sumY']:.9g} "
        f"sumZDelta={deltas['sumZ']:.9g} "
        f"minYDelta={deltas['minY']:.9g} "
        f"maxYDelta={deltas['maxY']:.9g} "
        f"sumSpeedDelta={deltas['sumSpeed']:.9g}"
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
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[SPLITSIM_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0:
        print("[SPLITSIM_RUNNER_ERROR] --timeout must be positive")
        return 2
    if args.frames < 120:
        print("[SPLITSIM_RUNNER_ERROR] --frames must be at least 120")
        return 2

    infrastructure_ok = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(spec, bin_dir, args.timeout, args.frames)
        infrastructure_ok = infrastructure_ok and passed
        results[spec.name] = fields
        if not passed:
            break

    pair_results: list[bool] = []
    if infrastructure_ok:
        pair_results.append(
            compare_pair(
                "tgs", results["tgs-simulate"], results["tgs-split"]
            )
        )
        if args.mode != "authority":
            for execution in ("parallel", "sequential"):
                pair_results.append(
                    compare_pair(
                        f"avbd-{execution}",
                        results[f"avbd-{execution}-simulate"],
                        results[f"avbd-{execution}-split"],
                    )
                )

    physics_ok = all(pair_results)
    accepted = infrastructure_ok and (
        physics_ok or args.mode == "probe"
    )
    print(
        f"[SPLITSIM_MATRIX] mode={args.mode} "
        f"infrastructure={'PASS' if infrastructure_ok else 'FAIL'} "
        f"physics={'PASS' if physics_ok else 'FAIL'} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
