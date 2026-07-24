#!/usr/bin/env python3
"""Run SnippetDelayLoadHook without a window and validate its platform gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetDelayLoadHook_64.exe"
DEFAULT_FRAMES = 180


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
    spec: RunSpec, bin_dir: Path, timeout: float, frames: int
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=delay-load-scene",
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

    gate_lines = [
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
    if len(gate_lines) != 1:
        errors.append(f"gate count is {len(gate_lines)}, expected exactly 1")
    else:
        fields, parse_errors = parse_gate(gate_lines[0])
        errors.extend(parse_errors)

    exact = {
        "schema": "1",
        "snippet": "SnippetDelayLoadHook",
        "solver": spec.solver,
        "case": "delay-load-scene",
        "execution": spec.execution,
        "frames": str(frames),
        "completedFrames": str(frames),
        "foundationLoaded": "1",
        "commonLoaded": "1",
        "physxLoaded": "1",
        "foundationPathMatched": "1",
        "commonPathMatched": "1",
        "physxPathMatched": "1",
        "exportsResolved": "4",
        "hooksRegistered": "2",
        "initialized": "1",
        "solverReadbackMatched": "1",
        "sceneStatics": "1",
        "sceneDynamics": "276",
        "nonFiniteActorSamples": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "unloadCompleted": "1",
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

    for key in ("warnings",):
        try:
            value = int(fields[key])
            if value < 0:
                errors.append(f"{key}={value}, expected non-negative")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected integer")

    displacement = parse_float(fields, "projectileDisplacement", errors)
    speed = parse_float(fields, "maxProjectileSpeed", errors)
    if displacement is not None and not 1.0 < displacement < 100000.0:
        errors.append(
            f"projectileDisplacement={displacement}, expected 1..100000"
        )
    if speed is not None and not 1.0 < speed < 100000.0:
        errors.append(f"maxProjectileSpeed={speed}, expected 1..100000")
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")

    print(
        f"[DELAY_LOAD_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"displacement={fields.get('projectileDisplacement', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[DELAY_LOAD_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    keys = (
        "completedFrames",
        "foundationLoaded",
        "commonLoaded",
        "physxLoaded",
        "foundationPathMatched",
        "commonPathMatched",
        "physxPathMatched",
        "exportsResolved",
        "hooksRegistered",
        "initialized",
        "solverReadbackMatched",
        "sceneStatics",
        "sceneDynamics",
        "nonFiniteActorSamples",
        "fetchFailures",
        "fatalErrors",
        "warnings",
        "unloadCompleted",
        "cleanupComplete",
    )
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        first_name = f"{solver}-{execution}-r1"
        second_name = f"{solver}-{execution}-r2"
        if first_name not in results or second_name not in results:
            continue
        mismatches = [
            key
            for key in keys
            if results[first_name].get(key) != results[second_name].get(key)
        ]
        if mismatches:
            passed = False
            print(
                f"[DELAY_LOAD_REPEAT_ERROR] lane={solver}-{execution} "
                f"fields={','.join(mismatches)}"
            )
        else:
            print(
                f"[DELAY_LOAD_REPEAT] lane={solver}-{execution} "
                "status=PASS invariantsExact=1"
            )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Windows delay-load hook and AVBD scene smoke "
            "without a window."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "acceptance"),
        default="probe",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        parser.error(f"executable not found: {executable}")
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    selected_specs = specs(args.mode)
    results: dict[str, dict[str, str]] = {}
    passed_runs = 0
    for spec in selected_specs:
        passed, fields = run_one(
            spec, bin_dir, args.timeout, args.frames
        )
        results[spec.name] = fields
        passed_runs += int(passed)

    repeats_passed = (
        compare_repeats(results) if args.mode == "acceptance" else True
    )
    passed = passed_runs == len(selected_specs) and repeats_passed
    print(
        f"[DELAY_LOAD_SUMMARY] mode={args.mode} "
        f"runs={passed_runs}/{len(selected_specs)} "
        f"repeatInvariantsExact={1 if repeats_passed else 0} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
