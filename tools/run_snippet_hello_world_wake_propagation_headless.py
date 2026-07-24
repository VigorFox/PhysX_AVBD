#!/usr/bin/env python3
"""Run SnippetHelloWorld contact-island wake propagation coverage."""

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
EXECUTABLE = "SnippetHelloWorld_64.exe"
FRAMES = 360
WAKE_FRAME = 60
@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    expected_status: str


def specs(mode: str) -> tuple[RunSpec, ...]:
    authority = (RunSpec("tgs-parallel", "tgs", "parallel", "PASS"),)
    if mode == "authority":
        return authority
    return authority + tuple(
        RunSpec(f"avbd-{execution}", "avbd", execution, "PASS")
        for execution in ("parallel", "sequential")
    )


def parse_fields(line: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append(f"malformed token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate key: {key}")
        fields[key] = value
    return fields, errors


def finite_number(
    fields: dict[str, str], key: str, errors: list[str]
) -> float:
    try:
        value = float(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key}={fields.get(key)!r}, expected finite float")
        return math.nan
    if not math.isfinite(value):
        errors.append(f"{key} is non-finite")
    return value


def run_one(spec: RunSpec, bin_dir: Path, timeout: float) -> bool:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=sleep-wake",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
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
        errors.append(f"gate count is {len(gate_lines)}, expected 1")
    else:
        fields, parse_errors = parse_fields(gate_lines[0])
        errors.extend(parse_errors)

    required = {
        "schema": "1",
        "snippet": "SnippetHelloWorld",
        "case": "sleep-wake",
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "capability": "SUPPORTED",
        "validation": "GATED",
        "status": spec.expected_status,
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "actorCount": "2",
        "sceneSleepingDisabled": "0",
        "wakeFrame": str(WAKE_FRAME),
        "wakeImpulseApplied": "1",
        "sleepingBeforeWake": "1",
        "awakeImmediatelyAfterWake": "1",
        "freeInitialSleeping": "1",
        "staticInitialSleeping": "1",
        "unexpectedNotifications": "0",
        "freeAwakeBeforeWakeSamples": "0",
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )

    expected_returncode = 0 if spec.expected_status == "PASS" else 1
    if result.returncode != expected_returncode:
        errors.append(
            f"exit code {result.returncode}, expected {expected_returncode}"
        )
    if spec.expected_status == "PASS":
        if fields.get("reason") != "none":
            errors.append(f"reason={fields.get('reason')!r}, expected 'none'")
        pass_fields = {
            "probeFinding": "wake-propagation-resleep-observed",
            "freeFinalSleeping": "1",
            "staticFinalSleeping": "1",
        }
        for key, expected in pass_fields.items():
            if fields.get(key) != expected:
                errors.append(
                    f"{key}={fields.get(key)!r}, expected {expected!r}"
                )
    numeric = {
        key: finite_number(fields, key, errors)
        for key in (
            "wakeCounterAfterImpulse",
            "freeFirstSleepAfterWakeFrame",
            "freeFirstWakeNotifyFrame",
            "freeMaxWakeVelocityX",
            "freeMaxWakeDisplacementX",
            "staticFirstSleepAfterWakeFrame",
            "staticFirstWakeNotifyFrame",
            "staticMaxWakeVelocityX",
            "staticMaxWakeDisplacementX",
        )
    }
    if numeric["wakeCounterAfterImpulse"] <= 0:
        errors.append("direct impulse did not reset the wake counter")
    if spec.expected_status == "PASS":
        if not WAKE_FRAME < numeric["freeFirstSleepAfterWakeFrame"] <= FRAMES:
            errors.append("wake source did not re-sleep")
        if not WAKE_FRAME < numeric["staticFirstSleepAfterWakeFrame"] <= FRAMES:
            errors.append("wake peer did not re-sleep")
        if numeric["freeFirstWakeNotifyFrame"] > WAKE_FRAME + 2:
            errors.append("wake source notification was late")
        if numeric["staticFirstWakeNotifyFrame"] > WAKE_FRAME + 2:
            errors.append("wake propagation notification was late")
        if (
            numeric["freeMaxWakeVelocityX"] < 0.5
            or numeric["freeMaxWakeDisplacementX"] < 0.25
        ):
            errors.append("wake source response is below threshold")
        if (
            numeric["staticMaxWakeVelocityX"] < 0.05
            or numeric["staticMaxWakeDisplacementX"] < 0.05
        ):
            errors.append("propagated peer response is below threshold")

    print(
        f"[HELLO_WAKE_RUN] name={spec.name} "
        f"expected={spec.expected_status} "
        f"actual={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[HELLO_WAKE_RUN_ERROR] name={spec.name} error={error}")
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "acceptance"),
        default="acceptance",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[HELLO_WAKE_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0:
        print("[HELLO_WAKE_RUNNER_ERROR] --timeout must be positive")
        return 2

    run_results = [
        run_one(spec, bin_dir, args.timeout) for spec in specs(args.mode)
    ]
    passed = all(run_results)
    print(
        f"[HELLO_WAKE_MATRIX] mode={args.mode} "
        f"runs={len(run_results)} "
        "expectedAvbdPropagationFailures=0 "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
