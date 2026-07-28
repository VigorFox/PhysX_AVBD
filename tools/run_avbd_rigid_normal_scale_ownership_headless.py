#!/usr/bin/env python3
"""Run the minimal rigid body-static normal scale-ownership oracle."""

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
EXECUTABLE = "SnippetToleranceScale_64.exe"
CASE_NAME = "normal-ownership-scale-pair"
FRAMES = 120
SCALED_LENGTH = 100.0
OWNERSHIP_PATTERN = re.compile(r"normalOwnership\(([^)]*)\)")
FRAME_PATTERN = re.compile(r"\bframe=(\d+)\b")


@dataclass(frozen=True)
class RunSpec:
    solver: str
    execution: str
    repeat: int

    @property
    def name(self) -> str:
        return f"{self.solver}-{self.execution}-r{self.repeat}"


def specs() -> tuple[RunSpec, ...]:
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    return tuple(
        RunSpec(solver, execution, repeat)
        for repeat in (1, 2)
        for solver, execution in lanes
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


def summarize_ownership(
    lines: list[str],
) -> tuple[dict[str, float], list[str]]:
    count_fields = (
        "alRows",
        "alEvals",
        "depenEligibleRows",
        "depenCorrections",
        "finiteImpulseSkips",
        "authoredFiniteSkips",
        "velocityCorrections",
        "restitutionCorrections",
    )
    sum_fields = ("depenDistance", "velocityDelta")
    max_fields = ("depenMax", "velocityMax")
    summary = {key: 0.0 for key in count_fields + sum_fields + max_fields}
    errors: list[str] = []
    for line in lines:
        match = OWNERSHIP_PATTERN.search(line)
        if not match:
            errors.append("diagnostic line missing normalOwnership group")
            continue
        fields, parse_errors = parse_fields(match.group(1))
        errors.extend(parse_errors)
        for key in count_fields + sum_fields + max_fields:
            try:
                value = float(fields[key])
            except (KeyError, ValueError):
                errors.append(f"missing/non-numeric diagnostic field: {key}")
                continue
            if not math.isfinite(value):
                errors.append(f"non-finite diagnostic field: {key}")
                continue
            if key in max_fields:
                summary[key] = max(summary[key], value)
            else:
                summary[key] += value
    return summary, errors


def split_scene_diagnostics(
    lines: list[str],
) -> tuple[list[list[str]], list[str]]:
    scenes: list[list[str]] = []
    errors: list[str] = []
    previous_frame: int | None = None
    for line in lines:
        match = FRAME_PATTERN.search(line)
        if not match:
            errors.append("diagnostic line missing frame")
            continue
        frame = int(match.group(1))
        if previous_frame is None or frame <= previous_frame:
            scenes.append([])
        scenes[-1].append(line)
        previous_frame = frame
    # The production diagnostic frame counter can remain process-monotonic
    # across the two sequential PxPhysics instances.  Both fixture runs have
    # identical modified-contact lifetimes, so an even single stream is split
    # into its base/scaled halves.
    if len(scenes) == 1 and len(lines) > 0 and len(lines) % 2 == 0:
        midpoint = len(lines) // 2
        scenes = [lines[:midpoint], lines[midpoint:]]
    if len(scenes) != 2:
        errors.append(
            f"diagnostic scene count is {len(scenes)}, expected 2"
        )
    if any(not scene for scene in scenes):
        errors.append("empty diagnostic scene")
    return scenes, errors


def compare_scale_ownership(
    base: dict[str, float], scaled: dict[str, float]
) -> list[str]:
    errors: list[str] = []
    count_fields = (
        "alRows",
        "alEvals",
        "depenEligibleRows",
        "depenCorrections",
        "finiteImpulseSkips",
        "authoredFiniteSkips",
        "velocityCorrections",
        "restitutionCorrections",
    )
    scaled_fields = (
        "depenDistance",
        "depenMax",
        "velocityDelta",
        "velocityMax",
    )
    for key in count_fields:
        if base[key] != scaled[key]:
            errors.append(
                f"scale ownership count mismatch: "
                f"{key}={base[key]:.9g}/{scaled[key]:.9g}"
            )
    for key in scaled_fields:
        normalized = scaled[key] / SCALED_LENGTH
        tolerance = 2e-6 + 1e-4 * max(abs(base[key]), abs(normalized))
        if abs(base[key] - normalized) > tolerance:
            errors.append(
                f"scale ownership magnitude mismatch: "
                f"{key}={base[key]:.9g}/{normalized:.9g} "
                f"tolerance={tolerance:.9g}"
            )
    if base["alRows"] <= 0 or base["alEvals"] <= 0:
        errors.append("body-static AL ownership was not observed")
    if base["depenEligibleRows"] <= 0:
        errors.append("depenetration eligibility was not observed")
    if base["finiteImpulseSkips"] != 0:
        errors.append("explicit unlimited contact was classified as finite")
    if base["velocityCorrections"] <= 0:
        errors.append("material normal-velocity ownership was not observed")
    if base["restitutionCorrections"] != 0:
        errors.append("inelastic fixture unexpectedly used restitution")
    return errors


def run_one(
    spec: RunSpec, bin_dir: Path, timeout_seconds: float
) -> tuple[
    bool,
    dict[str, str],
    dict[str, float],
    dict[str, float],
]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={CASE_NAME}",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
        "--dt=0.0166666675",
        "--dispatcher-threads=4",
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
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout_seconds
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
    base_ownership: dict[str, float] = {}
    scaled_ownership: dict[str, float] = {}

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
        errors.append(f"gate count is {len(gate_lines)}, expected exactly 1")
    else:
        fields, parse_errors = parse_fields(
            " ".join(gate_lines[0].split()[1:])
        )
        errors.extend(parse_errors)

    required = {
        "schema": "2",
        "snippet": "SnippetToleranceScale",
        "solver": spec.solver,
        "case": CASE_NAME,
        "execution": spec.execution,
        "frames": str(FRAMES),
        "runs": "2",
        "baseCompleted": str(FRAMES),
        "scaledCompleted": str(FRAMES),
        "baseBodies": "1",
        "scaledBodies": "1",
        "baseLength": "1",
        "scaledLength": "100",
        "baseNonFinite": "0",
        "scaledNonFinite": "0",
        "baseFetchFailures": "0",
        "scaledFetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "2",
        "pvd": "0",
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )
    for key in (
        "baseModifyCallbacks",
        "scaledModifyCallbacks",
        "baseModifiedPoints",
        "scaledModifiedPoints",
    ):
        try:
            if int(fields[key]) <= 0:
                errors.append(f"{key} did not observe modified contacts")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected positive int")

    if spec.solver == "avbd":
        scenes, split_errors = split_scene_diagnostics(diagnostic_lines)
        errors.extend(split_errors)
        if len(scenes) == 2 and all(scenes):
            base_ownership, base_errors = summarize_ownership(
                scenes[0]
            )
            scaled_ownership, scaled_errors = summarize_ownership(
                scenes[1]
            )
            errors.extend(base_errors)
            errors.extend(scaled_errors)
            errors.extend(
                compare_scale_ownership(base_ownership, scaled_ownership)
            )
    elif diagnostic_lines:
        errors.append("TGS unexpectedly emitted AVBD diagnostics")

    print(
        "[RIGID_NORMAL_SCALE_RUN] "
        f"name={spec.name} status={fields.get('status', 'MISSING')} "
        f"diagLines={len(diagnostic_lines)} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if base_ownership:
        normalized_scaled = dict(scaled_ownership)
        for key in (
            "depenDistance",
            "depenMax",
            "velocityDelta",
            "velocityMax",
        ):
            normalized_scaled[key] /= SCALED_LENGTH
        print(
            "[RIGID_NORMAL_SCALE_STATS] "
            f"name={spec.name} "
            + " ".join(
                f"base_{key}={value:.9g}"
                for key, value in sorted(base_ownership.items())
            )
            + " "
            + " ".join(
                f"scaledN_{key}={value:.9g}"
                for key, value in sorted(normalized_scaled.items())
            )
        )
    concise_output = "\n".join(
        line
        for line in combined.splitlines()
        if not line.startswith("[avbd:iters] ")
    )
    if concise_output:
        print(concise_output.rstrip())
    for error in errors:
        print(
            "[RIGID_NORMAL_SCALE_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, fields, base_ownership, scaled_ownership


def ownership_equal(
    left: dict[str, float], right: dict[str, float]
) -> bool:
    return left.keys() == right.keys() and all(
        abs(left[key] - right[key]) <= 2e-6 for key in left
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(
            "[RIGID_NORMAL_SCALE_RUNNER_ERROR] "
            f"missing executable: {executable}"
        )
        return 2
    if args.timeout <= 0.0:
        print(
            "[RIGID_NORMAL_SCALE_RUNNER_ERROR] "
            "--timeout must be positive"
        )
        return 2

    passed = 0
    avbd_results: list[tuple[dict[str, float], dict[str, float]]] = []
    for spec in specs():
        run_passed, _, base, scaled = run_one(
            spec, bin_dir, args.timeout
        )
        passed += run_passed
        if not run_passed:
            break
        if spec.solver == "avbd":
            avbd_results.append((base, scaled))

    consistency_ok = len(avbd_results) == 4
    if consistency_ok:
        reference_base, reference_scaled = avbd_results[0]
        consistency_ok = all(
            ownership_equal(reference_base, base)
            and ownership_equal(reference_scaled, scaled)
            for base, scaled in avbd_results[1:]
        )
    print(
        "[RIGID_NORMAL_SCALE_CONSISTENCY] "
        f"avbdRuns={len(avbd_results)} "
        f"status={'PASS' if consistency_ok else 'FAIL'}"
    )

    expected = len(specs())
    matrix_passed = passed == expected and consistency_ok
    print(
        "[RIGID_NORMAL_SCALE_MATRIX] "
        f"passed={passed} expected={expected} "
        f"status={'PASS' if matrix_passed else 'FAIL'}"
    )
    return 0 if matrix_passed else 1


if __name__ == "__main__":
    sys.exit(main())
