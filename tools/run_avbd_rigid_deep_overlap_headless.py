#!/usr/bin/env python3
"""Gate deep-overlap recovery without launching a render window."""

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
CASE_NAME = "ownership-deep-tilted"
FRAMES = 120
SETTLED_ANGULAR_LIMIT = 0.01
RECOVERY_ANGULAR_LIMIT = 0.05
AUTHORITY_AVBD_GAP_MIN = 5.0


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
            "peakAbsBody0VelocityY",
            "peakBody0AngularSpeed",
            "peakBody0AngularFrame",
            "finalBody0Speed",
            "finalBody0AngularSpeed",
            "minBody0Y",
            "maxBody0Y",
        )
    }
    if metrics["finalBody0AngularSpeed"] > SETTLED_ANGULAR_LIMIT:
        errors.append("tilted overlap did not settle by the final frame")
    if metrics["minBody0Y"] < 0.1 or metrics["maxBody0Y"] < 0.45:
        errors.append("tilted overlap did not recover above the plane")

    print(
        "[RIGID_DEEP_OVERLAP_RUN] "
        f"name={spec.name} peakAngular={metrics['peakBody0AngularSpeed']:.9g} "
        f"peakFrame={metrics['peakBody0AngularFrame']:.0f} "
        f"finalAngular={metrics['finalBody0AngularSpeed']:.9g} "
        f"peakLinearY={metrics['peakAbsBody0VelocityY']:.9g} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(
            "[RIGID_DEEP_OVERLAP_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "acceptance"),
        default="authority",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[RIGID_DEEP_OVERLAP_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0.0 or args.repeats <= 0:
        print(
            "[RIGID_DEEP_OVERLAP_RUNNER_ERROR] "
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
    infrastructure_ok = True
    for spec in specs:
        ok, metrics = run_one(spec, bin_dir, args.timeout)
        infrastructure_ok = infrastructure_ok and ok
        results.append((spec, metrics))
        if not ok:
            break

    physics_ok = infrastructure_ok and len(results) == len(specs)
    if physics_ok:
        tgs_peaks = [
            metrics["peakBody0AngularSpeed"]
            for spec, metrics in results
            if spec.solver == "tgs"
        ]
        avbd_peaks = [
            metrics["peakBody0AngularSpeed"]
            for spec, metrics in results
            if spec.solver == "avbd"
        ]
        if max(tgs_peaks) > RECOVERY_ANGULAR_LIMIT:
            physics_ok = False
            print(
                "[RIGID_DEEP_OVERLAP_ERROR] "
                "name=matrix error=TGS authority injected angular recovery"
            )
        if args.mode == "authority":
            if min(avbd_peaks) < AUTHORITY_AVBD_GAP_MIN:
                physics_ok = False
                print(
                    "[RIGID_DEEP_OVERLAP_ERROR] "
                    "name=matrix error=AVBD angular gap was not observed"
                )
        elif max(avbd_peaks) > RECOVERY_ANGULAR_LIMIT:
            physics_ok = False
            print(
                "[RIGID_DEEP_OVERLAP_ERROR] "
                "name=matrix error=AVBD retained pose-derived angular launch"
            )

        by_lane: dict[tuple[str, str], list[float]] = {}
        for spec, metrics in results:
            by_lane.setdefault((spec.solver, spec.execution), []).append(
                metrics["peakBody0AngularSpeed"]
            )
        for lane, values in by_lane.items():
            if max(values) - min(values) > 1e-7:
                physics_ok = False
                print(
                    "[RIGID_DEEP_OVERLAP_ERROR] "
                    f"name={lane[0]}-{lane[1]} error=repeat mismatch"
                )
        avbd_parallel = by_lane[("avbd", "parallel")][0]
        avbd_sequential = by_lane[("avbd", "sequential")][0]
        if abs(avbd_parallel - avbd_sequential) > 1e-7:
            physics_ok = False
            print(
                "[RIGID_DEEP_OVERLAP_ERROR] "
                "name=avbd-execution error=parallel/sequential mismatch"
            )

    print(
        "[RIGID_DEEP_OVERLAP_MATRIX] "
        f"mode={args.mode} runs={len(results)}/{len(specs)} "
        f"infrastructure={'PASS' if infrastructure_ok else 'FAIL'} "
        f"physics={'PASS' if physics_ok else 'FAIL'} "
        f"status={'PASS' if infrastructure_ok and physics_ok else 'FAIL'}"
    )
    return 0 if infrastructure_ok and physics_ok else 1


if __name__ == "__main__":
    sys.exit(main())
