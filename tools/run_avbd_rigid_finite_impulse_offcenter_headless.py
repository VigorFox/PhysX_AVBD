#!/usr/bin/env python3
"""Gate the single-row spatial finite-impulse body-static response."""

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
EXECUTABLE = "SnippetContactModification_64.exe"
CASE_NAME = "finite-max-impulse-offcenter"
FRAMES = 120
AUTHORITY_ANGULAR_GAP_MIN = 0.025
CHECKPOINT_VELOCITY_GAP_LIMIT = 0.005
CHECKPOINT_ANGULAR_GAP_LIMIT = 0.015


@dataclass(frozen=True)
class RunSpec:
    solver: str
    execution: str
    repeat: int

    @property
    def name(self) -> str:
        return f"{self.solver}-{self.execution}-r{self.repeat}"


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


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, dict[str, float]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={CASE_NAME}",
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
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")
    if len(gate_lines) != 1:
        errors.append(f"gate count is {len(gate_lines)}, expected 1")
    else:
        fields, parse_errors = parse_fields(gate_lines[0])
        errors.extend(parse_errors)

    required = {
        "schema": "2",
        "snippet": "SnippetContactModification",
        "solver": spec.solver,
        "case": CASE_NAME,
        "execution": spec.execution,
        "frames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
        "maxImpulseReadbackErrors": "0",
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

    metrics = {
        key: finite_number(fields, key, errors)
        for key in (
            "modifyCallbackCount",
            "modifiedPointCount",
            "maxReportedImpulse",
            "finiteCheckpointVelocityY",
            "finiteCheckpointAngularSpeed",
        )
    }
    if metrics["modifyCallbackCount"] != metrics["modifiedPointCount"]:
        errors.append("fixture did not remain a single-point contact oracle")
    if not 0.999 <= metrics["maxReportedImpulse"] <= 1.001:
        errors.append("reported impulse did not consume the 1 Ns cap")

    print(
        "[RIGID_FINITE_IMPULSE_OFFCENTER_RUN] "
        f"name={spec.name} callbacks={metrics['modifyCallbackCount']:.0f} "
        f"points={metrics['modifiedPointCount']:.0f} "
        f"checkpointVy={metrics['finiteCheckpointVelocityY']:.9g} "
        f"checkpointAngular="
        f"{metrics['finiteCheckpointAngularSpeed']:.9g} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(
            "[RIGID_FINITE_IMPULSE_OFFCENTER_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "acceptance"),
        default="acceptance",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[RIGID_FINITE_IMPULSE_OFFCENTER_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0.0 or args.repeats <= 0:
        print(
            "[RIGID_FINITE_IMPULSE_OFFCENTER_RUNNER_ERROR] "
            "--timeout and --repeats must be positive"
        )
        return 2

    specs = [
        RunSpec(solver, execution, repeat)
        for repeat in range(1, args.repeats + 1)
        for solver, execution in (
            ("tgs", "parallel"),
            ("avbd", "parallel"),
            ("avbd", "sequential"),
        )
    ]
    results: list[tuple[RunSpec, dict[str, float]]] = []
    passed = 0
    for spec in specs:
        ok, metrics = run_one(spec, bin_dir, args.timeout)
        results.append((spec, metrics))
        if not ok:
            break
        passed += 1

    physics_ok = passed == len(specs)
    if physics_ok:
        by_lane: dict[tuple[str, str], list[dict[str, float]]] = {}
        for spec, metrics in results:
            by_lane.setdefault((spec.solver, spec.execution), []).append(
                metrics
            )
        metric_names = (
            "modifyCallbackCount",
            "modifiedPointCount",
            "maxReportedImpulse",
            "finiteCheckpointVelocityY",
            "finiteCheckpointAngularSpeed",
        )
        for lane, lane_results in by_lane.items():
            for metric_name in metric_names:
                values = [result[metric_name] for result in lane_results]
                if max(values) - min(values) > 1.0e-7:
                    physics_ok = False
                    print(
                        "[RIGID_FINITE_IMPULSE_OFFCENTER_ERROR] "
                        f"name={lane[0]}-{lane[1]} "
                        f"error=repeat mismatch for {metric_name}"
                    )

        tgs = by_lane[("tgs", "parallel")][0]
        avbd_parallel = by_lane[("avbd", "parallel")][0]
        avbd_sequential = by_lane[("avbd", "sequential")][0]
        for metric_name in metric_names:
            if abs(
                avbd_parallel[metric_name] - avbd_sequential[metric_name]
            ) > 1.0e-7:
                physics_ok = False
                print(
                    "[RIGID_FINITE_IMPULSE_OFFCENTER_ERROR] "
                    f"name=avbd error=execution mismatch for {metric_name}"
                )

        velocity_gap = abs(
            avbd_parallel["finiteCheckpointVelocityY"]
            - tgs["finiteCheckpointVelocityY"]
        )
        angular_gap = abs(
            avbd_parallel["finiteCheckpointAngularSpeed"]
            - tgs["finiteCheckpointAngularSpeed"]
        )
        if args.mode == "authority":
            if angular_gap < AUTHORITY_ANGULAR_GAP_MIN:
                physics_ok = False
                print(
                    "[RIGID_FINITE_IMPULSE_OFFCENTER_ERROR] "
                    "name=matrix error=COM-only authority gap not observed"
                )
        elif (
            velocity_gap > CHECKPOINT_VELOCITY_GAP_LIMIT
            or angular_gap > CHECKPOINT_ANGULAR_GAP_LIMIT
        ):
            physics_ok = False
            print(
                "[RIGID_FINITE_IMPULSE_OFFCENTER_ERROR] "
                "name=matrix error=single-row spatial parity outside limits"
            )

    print(
        "[RIGID_FINITE_IMPULSE_OFFCENTER_MATRIX] "
        f"passed={passed} failed={len(specs) - passed} "
        f"expected={len(specs)} "
        f"mode={args.mode} status={'PASS' if physics_ok else 'FAIL'}"
    )
    return 0 if physics_ok else 1


if __name__ == "__main__":
    sys.exit(main())
