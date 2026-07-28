#!/usr/bin/env python3
"""Gate rigid body-static restitution on both sides of the scene threshold."""

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
CASES = (
    "restitution-threshold-below",
    "restitution-threshold-above",
)
FRAMES = 120
OWNERSHIP_PATTERN = re.compile(r"normalOwnership\(([^)]*)\)")


@dataclass(frozen=True)
class RunSpec:
    case_name: str
    solver: str
    execution: str
    repeat: int

    @property
    def name(self) -> str:
        return (
            f"{self.case_name}-{self.solver}-{self.execution}-r{self.repeat}"
        )


def parse_fields(text: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in text.split():
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


def restitution_count(lines: list[str], errors: list[str]) -> int:
    total = 0
    for line in lines:
        match = OWNERSHIP_PATTERN.search(line)
        if not match:
            errors.append("diagnostic line missing normalOwnership group")
            continue
        fields, parse_errors = parse_fields(match.group(1))
        errors.extend(parse_errors)
        try:
            total += int(fields["restitutionCorrections"])
        except (KeyError, ValueError):
            errors.append("missing/non-integer restitutionCorrections")
    return total


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, dict[str, float]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
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
    diagnostic_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[avbd:iters] ")
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
        fields, parse_errors = parse_fields(
            " ".join(gate_lines[0].split()[1:])
        )
        errors.extend(parse_errors)

    required = {
        "schema": "2",
        "snippet": "SnippetContactModification",
        "solver": spec.solver,
        "case": spec.case_name,
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
            "peakBody0VelocityY",
            "maxBody0Y",
            "minBody0Y",
            "finalBody0Speed",
        )
    }
    corrections = 0
    if spec.solver == "avbd":
        if not diagnostic_lines:
            errors.append("missing AVBD iteration diagnostics")
        else:
            corrections = restitution_count(diagnostic_lines, errors)
            if (
                spec.case_name == "restitution-threshold-below"
                and corrections != 0
            ):
                errors.append("below-threshold path applied restitution")
            if (
                spec.case_name == "restitution-threshold-above"
                and corrections != 1
            ):
                errors.append(
                    "above-threshold path did not apply exactly one onset "
                    "restitution correction"
                )
    elif diagnostic_lines:
        errors.append("TGS unexpectedly emitted AVBD diagnostics")

    metrics["restitutionCorrections"] = float(corrections)
    print(
        "[RIGID_RESTITUTION_THRESHOLD_RUN] "
        f"name={spec.name} "
        f"peakUp={metrics['peakBody0VelocityY']:.9g} "
        f"maxY={metrics['maxBody0Y']:.9g} "
        f"finalSpeed={metrics['finalBody0Speed']:.9g} "
        f"restitutionCorrections={corrections} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(
            "[RIGID_RESTITUTION_THRESHOLD_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[RIGID_RESTITUTION_THRESHOLD_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0.0 or args.repeats <= 0:
        print(
            "[RIGID_RESTITUTION_THRESHOLD_RUNNER_ERROR] "
            "--timeout and --repeats must be positive"
        )
        return 2

    specs = [
        RunSpec(case_name, solver, execution, repeat)
        for repeat in range(1, args.repeats + 1)
        for case_name in CASES
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
        for case_name in CASES:
            values = [
                metrics["peakBody0VelocityY"]
                for spec, metrics in results
                if spec.case_name == case_name
            ]
            if max(values) - min(values) > 2.0e-2:
                physics_ok = False
                print(
                    "[RIGID_RESTITUTION_THRESHOLD_ERROR] "
                    f"name={case_name} error=solver/execution/repeat mismatch"
                )

    print(
        "[RIGID_RESTITUTION_THRESHOLD_MATRIX] "
        f"passed={passed} failed={len(specs) - passed} "
        f"expected={len(specs)} "
        f"status={'PASS' if physics_ok else 'FAIL'}"
    )
    return 0 if physics_ok else 1


if __name__ == "__main__":
    sys.exit(main())
