#!/usr/bin/env python3
"""Run gyroscopic flag off/on authority and AVBD failure-first coverage."""

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
EXECUTABLE = "SnippetGyroscopic_64.exe"


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    case_name: str


def specs(mode: str) -> tuple[RunSpec, ...]:
    authority = (
        RunSpec("tgs-off", "tgs", "parallel", "gyro-off"),
        RunSpec("tgs-on", "tgs", "parallel", "gyro-on"),
    )
    if mode == "authority":
        return authority
    return authority + tuple(
        RunSpec(f"avbd-{execution}-{state}", "avbd", execution, f"gyro-{state}")
        for execution in ("parallel", "sequential")
        for state in ("off", "on")
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
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        "--frames=600",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = "600"
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
        "snippet": "SnippetGyroscopic",
        "solver": spec.solver,
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": "600",
        "completedFrames": "600",
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
        "gyroEnabled": "1" if spec.case_name == "gyro-on" else "0",
        "sleepingFrames": "0",
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

    numeric_keys = (
        "finalQx", "finalQy", "finalQz", "finalQw",
        "finalWx", "finalWy", "finalWz",
        "initialEnergy", "finalEnergy", "maxEnergyDrift",
        "initialMomentumMagnitude", "finalMomentumMagnitude",
        "maxMomentumVectorDrift",
    )
    for key in numeric_keys:
        try:
            value = number(fields, key)
            if not math.isfinite(value):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected finite float")

    print(
        f"[GYROSCOPIC_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[GYROSCOPIC_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def compare_pair(
    label: str,
    off: dict[str, str],
    on: dict[str, str],
    require_distinct: bool,
) -> tuple[bool, dict[str, float]]:
    try:
        off_q = tuple(number(off, f"finalQ{axis}") for axis in "xyzw")
        on_q = tuple(number(on, f"finalQ{axis}") for axis in "xyzw")
        quaternion_dot = abs(sum(a * b for a, b in zip(off_q, on_q)))
        quaternion_dot = min(1.0, max(0.0, quaternion_dot))
        orientation_delta = 2.0 * math.acos(quaternion_dot)
        off_w = tuple(number(off, f"finalW{axis}") for axis in "xyz")
        on_w = tuple(number(on, f"finalW{axis}") for axis in "xyz")
        angular_velocity_delta = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(off_w, on_w))
        )
        on_momentum_drift = number(on, "maxMomentumVectorDrift")
        on_energy_drift = number(on, "maxEnergyDrift")
    except (KeyError, ValueError) as exc:
        print(f"[GYROSCOPIC_PAIR_ERROR] pair={label} error={exc}")
        return False, {}

    distinct = orientation_delta > 0.1 and angular_velocity_delta > 0.1
    conservation_ok = (
        on_momentum_drift < 0.25 and on_energy_drift < 0.25
    )
    passed = conservation_ok and (distinct or not require_distinct)
    metrics = {
        "orientationDelta": orientation_delta,
        "angularVelocityDelta": angular_velocity_delta,
        "onMomentumDrift": on_momentum_drift,
        "onEnergyDrift": on_energy_drift,
    }
    reason = (
        "none" if passed else
        "gyro_flag_not_distinct" if not distinct else
        "gyro_conservation_out_of_bounds"
    )
    print(
        f"[GYROSCOPIC_PAIR] pair={label} "
        f"status={'PASS' if passed else 'FAIL'} reason={reason} "
        f"orientationDelta={orientation_delta:.9g} "
        f"angularVelocityDelta={angular_velocity_delta:.9g} "
        f"onMomentumDrift={on_momentum_drift:.9g} "
        f"onEnergyDrift={on_energy_drift:.9g}"
    )
    return passed, metrics


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
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[GYROSCOPIC_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0:
        print("[GYROSCOPIC_RUNNER_ERROR] --timeout must be positive")
        return 2

    infrastructure_ok = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(spec, bin_dir, args.timeout)
        infrastructure_ok = infrastructure_ok and passed
        results[spec.name] = fields
        if not passed:
            break

    pair_results: list[bool] = []
    if infrastructure_ok:
        passed, _ = compare_pair(
            "tgs", results["tgs-off"], results["tgs-on"], True
        )
        pair_results.append(passed)
        if args.mode != "authority":
            for execution in ("parallel", "sequential"):
                passed, _ = compare_pair(
                    f"avbd-{execution}",
                    results[f"avbd-{execution}-off"],
                    results[f"avbd-{execution}-on"],
                    True,
                )
                pair_results.append(passed)

    physics_ok = all(pair_results)
    accepted = infrastructure_ok and (
        physics_ok or args.mode == "probe"
    )
    print(
        f"[GYROSCOPIC_MATRIX] mode={args.mode} "
        f"infrastructure={'PASS' if infrastructure_ok else 'FAIL'} "
        f"physics={'PASS' if physics_ok else 'FAIL'} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
