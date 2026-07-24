#!/usr/bin/env python3
"""Run SnippetHelloWorld rigid-body lock-flag authority and AVBD coverage."""

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
WITNESSES = (
    "linear-x",
    "linear-y",
    "linear-z",
    "angular-x",
    "angular-y",
    "angular-z",
)


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
    avbd_status = "FAIL" if mode == "probe" else "PASS"
    return authority + tuple(
        RunSpec(f"avbd-{execution}", "avbd", execution, avbd_status)
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
    frames = 120
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=lock-flags",
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

    gate_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]
    detail_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[SnippetHelloWorldLock] ")
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
        "case": "lock-flags",
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(frames),
        "completedFrames": str(frames),
        "capability": "SUPPORTED",
        "validation": "GATED",
        "status": spec.expected_status,
        "reason": "none"
        if spec.expected_status == "PASS"
        else "locked_axis_motion",
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "actorCount": "12",
        "lockWitnessCount": "6",
        "lockFlagsReadback": "6",
        "runtimeImpulseFrame": "30",
        "runtimeExcitations": "12",
        "finiteSamples": str(frames * 12),
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

    numeric_keys = (
        "maxLockedAxisMotion",
        "maxLockedAxisSpeed",
        "minControlAxisMotion",
        "minControlAxisSpeed",
        "maxControlAxisMotion",
        "maxControlAxisSpeed",
        "lockMotionTolerance",
        "lockSpeedTolerance",
        "controlMotionMinimum",
        "controlSpeedMinimum",
    )
    numeric = {
        key: finite_number(fields, key, errors) for key in numeric_keys
    }

    details: dict[str, dict[str, str]] = {}
    for line in detail_lines:
        detail, parse_errors = parse_fields(line)
        errors.extend(parse_errors)
        witness = detail.get("witness")
        if witness is None:
            errors.append("detail missing witness")
        elif witness in details:
            errors.append(f"duplicate detail witness: {witness}")
        else:
            details[witness] = detail
    if set(details) != set(WITNESSES):
        errors.append(
            f"detail witnesses={sorted(details)}, expected={list(WITNESSES)}"
        )

    for witness in WITNESSES:
        detail = details.get(witness)
        if detail is None:
            continue
        values = {
            key: finite_number(detail, key, errors)
            for key in (
                "lockedMotion",
                "lockedSpeed",
                "controlMotion",
                "controlSpeed",
            )
        }
        if (
            values["controlMotion"] < 1.0
            or values["controlSpeed"] < 1.0
        ):
            errors.append(f"{witness} control did not respond")
        if spec.expected_status == "PASS":
            if (
                values["lockedMotion"] > 1e-4
                or values["lockedSpeed"] > 1e-4
            ):
                errors.append(f"{witness} locked axis moved")
        elif (
            values["lockedMotion"] < 1.0
            or values["lockedSpeed"] < 1.0
        ):
            errors.append(f"{witness} did not expose the AVBD lock failure")

    if math.isfinite(numeric["minControlAxisMotion"]) and (
        numeric["minControlAxisMotion"] < 1.0
        or numeric["minControlAxisSpeed"] < 1.0
    ):
        errors.append("aggregate control response is below threshold")
    if spec.expected_status == "PASS" and (
        numeric["maxLockedAxisMotion"] > 1e-4
        or numeric["maxLockedAxisSpeed"] > 1e-4
    ):
        errors.append("aggregate locked-axis response exceeds tolerance")

    print(
        f"[HELLO_LOCK_RUN] name={spec.name} "
        f"expected={spec.expected_status} "
        f"actual={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[HELLO_LOCK_RUN_ERROR] name={spec.name} error={error}")
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
        print(f"[HELLO_LOCK_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0:
        print("[HELLO_LOCK_RUNNER_ERROR] --timeout must be positive")
        return 2

    run_results = [
        run_one(spec, bin_dir, args.timeout) for spec in specs(args.mode)
    ]
    passed = all(run_results)
    expected_failures = (
        0 if args.mode == "authority" else 12 if args.mode == "probe" else 0
    )
    print(
        f"[HELLO_LOCK_MATRIX] mode={args.mode} "
        f"runs={len(run_results)} expectedAvbdDofFailures={expected_failures} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
