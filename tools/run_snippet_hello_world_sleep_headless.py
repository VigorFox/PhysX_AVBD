#!/usr/bin/env python3
"""Run SnippetHelloWorld static-contact sleep authority and AVBD coverage."""

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
FRAMES = 180


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    expect_static_sleep: bool


def specs(mode: str) -> tuple[RunSpec, ...]:
    authority = (RunSpec("tgs-parallel", "tgs", "parallel", True),)
    if mode == "authority":
        return authority
    avbd_expectation = mode == "acceptance"
    return authority + tuple(
        RunSpec(
            f"avbd-{execution}",
            "avbd",
            execution,
            avbd_expectation,
        )
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
        "--case=sleep-idle",
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
        "case": "sleep-idle",
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "validation": "GATED",
        "status": "PASS",
        "reason": "none",
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "actorCount": "2",
        "capability": "SUPPORTED",
        "sceneSleepingDisabled": "0",
        "freeSleepLifecycleGate": "HARD",
        "staticTouchSleepGate": "HARD",
        "freeInitialSleeping": "0",
        "freeFinalSleeping": "1",
        "staticInitialSleeping": "0",
        "staticFinalSleeping": "1"
        if spec.expect_static_sleep
        else "0",
        "unexpectedNotifications": "0",
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )

    expected_finding = (
        "auto-sleep-observed"
        if spec.expect_static_sleep
        else "free-sleep-static-touch-awake"
    )
    if fields.get("probeFinding") != expected_finding:
        errors.append(
            f"probeFinding={fields.get('probeFinding')!r}, "
            f"expected {expected_finding!r}"
        )
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")

    free_first_sleep = finite_number(fields, "freeFirstSleepFrame", errors)
    static_first_sleep = finite_number(
        fields, "staticFirstSleepFrame", errors
    )
    static_sleep_samples = finite_number(
        fields, "staticSleepSamples", errors
    )
    static_awake_samples = finite_number(
        fields, "staticAwakeSamples", errors
    )
    static_min_top = finite_number(fields, "staticMinTopY", errors)
    if math.isfinite(free_first_sleep) and not 1 <= free_first_sleep <= 60:
        errors.append("free witness did not auto-sleep in the authority window")
    if spec.expect_static_sleep:
        if not 1 <= static_first_sleep <= 60:
            errors.append(
                "static-touch witness did not auto-sleep in authority window"
            )
        if static_sleep_samples <= 0:
            errors.append("static-touch witness has no sleeping samples")
    else:
        if static_first_sleep != 4294967295:
            errors.append("AVBD static-touch witness unexpectedly slept")
        if static_sleep_samples != 0 or static_awake_samples != FRAMES:
            errors.append("AVBD failure witness sample counts are inconsistent")
    if static_min_top < -0.05:
        errors.append("static-touch witness escaped below the ground")

    print(
        f"[HELLO_SLEEP_RUN] name={spec.name} "
        f"expectStaticSleep={int(spec.expect_static_sleep)} "
        f"actualStaticSleep={fields.get('staticFinalSleeping', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[HELLO_SLEEP_RUN_ERROR] name={spec.name} error={error}")
    return not errors


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
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[HELLO_SLEEP_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0:
        print("[HELLO_SLEEP_RUNNER_ERROR] --timeout must be positive")
        return 2

    run_results = [
        run_one(spec, bin_dir, args.timeout) for spec in specs(args.mode)
    ]
    passed = all(run_results)
    expected_failures = 2 if args.mode == "probe" else 0
    print(
        f"[HELLO_SLEEP_MATRIX] mode={args.mode} "
        f"runs={len(run_results)} "
        f"expectedAvbdStaticTouchFailures={expected_failures} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
