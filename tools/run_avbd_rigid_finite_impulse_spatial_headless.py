#!/usr/bin/env python3
"""Gate spatial finite-impulse ownership for a tilted body-static impact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetContactModification_64.exe"
CASE_NAME = "finite-max-impulse-tilted"
FRAMES = 120
AUTHORITY_PEAK_ANGULAR_GAP_MIN = 3.0
AUTHORITY_MIN_Y_GAP_MIN = 0.075
PEAK_ANGULAR_PARITY_LIMIT = 0.50
FINAL_ANGULAR_PARITY_LIMIT = 0.50
PEAK_UP_PARITY_LIMIT = 0.50
MIN_Y_PARITY_LIMIT = 0.05
ITER_PATTERN = re.compile(
    r"\[avbd:iters\] frame=(\d+).*?"
    r"normalOwnership\(alRows=(\d+) alEvals=(\d+)"
)
FRAME_PATTERN = re.compile(
    r"\[FINITE_SPATIAL_FRAME\] frame=(\d+) y=([^\s]+) "
    r"vy=([^\s]+) angular=([^\s]+)"
)


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
    env["PHYSX_AVBD_ITER_DIAG"] = "1" if spec.solver == "avbd" else "0"
    env["PHYSX_AVBD_ITER_DIAG_EVERY"] = "1"
    env["PHYSX_AVBD_ITER_DIAG_SEQUENTIAL"] = (
        "1" if spec.execution == "sequential" else "0"
    )
    env["PHYSX_AVBD_NORMAL_ROW_DIAG"] = (
        "1" if spec.solver == "avbd" else "0"
    )
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
            "peakBody0VelocityY",
            "peakBody0AngularSpeed",
            "finalBody0AngularSpeed",
            "finalBody0Speed",
            "minBody0Y",
            "maxBody0Y",
            "maxReportedImpulse",
        )
    }
    if not 0.999 <= metrics["maxReportedImpulse"] <= 1.001:
        errors.append(
            "reported per-point impulse did not consume the authored 1 Ns cap"
        )

    if spec.repeat == 1:
        iteration_rows: dict[int, tuple[int, int]] = {}
        frame_rows: dict[int, tuple[float, float, float]] = {}
        for line in combined.splitlines():
            iteration_match = ITER_PATTERN.search(line)
            if iteration_match:
                iteration_rows[int(iteration_match.group(1))] = (
                    int(iteration_match.group(2)),
                    int(iteration_match.group(3)),
                )
            frame_match = FRAME_PATTERN.search(line)
            if frame_match:
                frame_rows[int(frame_match.group(1))] = (
                    float(frame_match.group(2)),
                    float(frame_match.group(3)),
                    float(frame_match.group(4)),
                )
        for frame in sorted(frame_rows)[:16]:
            y, vy, angular = frame_rows[frame]
            al_rows, al_evals = iteration_rows.get(
                frame + 1, iteration_rows.get(frame, (0, 0))
            )
            print(
                "[RIGID_FINITE_IMPULSE_FRAME] "
                f"name={spec.name} frame={frame} y={y:.9g} "
                f"vy={vy:.9g} angular={angular:.9g} "
                f"alRows={al_rows} alEvals={al_evals}"
            )
    print(
        "[RIGID_FINITE_IMPULSE_SPATIAL_RUN] "
        f"name={spec.name} callbacks={metrics['modifyCallbackCount']:.0f} "
        f"points={metrics['modifiedPointCount']:.0f} "
        f"peakUp={metrics['peakBody0VelocityY']:.9g} "
        f"peakAngular={metrics['peakBody0AngularSpeed']:.9g} "
        f"finalAngular={metrics['finalBody0AngularSpeed']:.9g} "
        f"minY={metrics['minBody0Y']:.9g} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(
            "[RIGID_FINITE_IMPULSE_SPATIAL_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "acceptance"),
        default="authority",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[RIGID_FINITE_IMPULSE_SPATIAL_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0.0 or args.repeats <= 0:
        print(
            "[RIGID_FINITE_IMPULSE_SPATIAL_RUNNER_ERROR] "
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
            "peakBody0VelocityY",
            "peakBody0AngularSpeed",
            "finalBody0AngularSpeed",
            "minBody0Y",
            "maxReportedImpulse",
        )
        for lane, lane_results in by_lane.items():
            for metric_name in metric_names:
                values = [result[metric_name] for result in lane_results]
                if max(values) - min(values) > 1.0e-7:
                    physics_ok = False
                    print(
                        "[RIGID_FINITE_IMPULSE_SPATIAL_ERROR] "
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
                    "[RIGID_FINITE_IMPULSE_SPATIAL_ERROR] "
                    f"name=avbd error=execution mismatch for {metric_name}"
                )

        peak_angular_gap = abs(
            avbd_parallel["peakBody0AngularSpeed"]
            - tgs["peakBody0AngularSpeed"]
        )
        final_angular_gap = abs(
            avbd_parallel["finalBody0AngularSpeed"]
            - tgs["finalBody0AngularSpeed"]
        )
        peak_up_gap = abs(
            avbd_parallel["peakBody0VelocityY"]
            - tgs["peakBody0VelocityY"]
        )
        min_y_gap = abs(avbd_parallel["minBody0Y"] - tgs["minBody0Y"])
        if args.mode == "authority":
            if (
                peak_angular_gap < AUTHORITY_PEAK_ANGULAR_GAP_MIN
                or min_y_gap < AUTHORITY_MIN_Y_GAP_MIN
            ):
                physics_ok = False
                print(
                    "[RIGID_FINITE_IMPULSE_SPATIAL_ERROR] "
                    "name=matrix error=finite spatial authority gap not observed"
                )
        elif (
            peak_up_gap > PEAK_UP_PARITY_LIMIT
            or peak_angular_gap > PEAK_ANGULAR_PARITY_LIMIT
            or final_angular_gap > FINAL_ANGULAR_PARITY_LIMIT
            or min_y_gap > MIN_Y_PARITY_LIMIT
        ):
            physics_ok = False
            print(
                "[RIGID_FINITE_IMPULSE_SPATIAL_ERROR] "
                "name=matrix error=finite spatial parity outside limits"
            )

    print(
        "[RIGID_FINITE_IMPULSE_SPATIAL_MATRIX] "
        f"passed={passed} failed={len(specs) - passed} "
        f"expected={len(specs)} "
        f"mode={args.mode} status={'PASS' if physics_ok else 'FAIL'}"
    )
    return 0 if physics_ok else 1


if __name__ == "__main__":
    sys.exit(main())
