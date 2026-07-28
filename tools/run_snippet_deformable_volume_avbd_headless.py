#!/usr/bin/env python3
"""Run CPU AVBD soft-body component/coexistence cases without a window."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetDeformableVolumeAVBD_64.exe"
CASES = (
    "volume-ground",
    "volume-static-box",
    "soft-soft",
    "cone-ground",
    "current-all",
)
INT_KEYS = (
    "frames",
    "fetchFailures",
    "particles",
    "softBodies",
    "tetElements",
    "surfaceTriangles",
    "rigidBoxes",
    "sceneStatics",
    "sceneDynamics",
    "sceneDeformableVolumes",
    "groundContactFrames",
    "rigidContactFrames",
    "softContactFrames",
    "maxGroundContacts",
    "maxRigidContacts",
    "maxSoftContacts",
    "finalInsideParticles",
    "nonFiniteParticleSamples",
    "invertedElementSamples",
    "firstInversionFrame",
    "firstInversionBody",
    "firstInversionElement",
    "invertedBodiesMask",
    "fatalErrors",
    "warningErrors",
    "cleanupComplete",
)
FLOAT_KEYS = (
    "minDetF",
    "maxDetF",
    "minBodyVolumeRatio",
    "maxBodyVolumeRatio",
    "minY",
    "maxY",
    "finalMinY",
    "finalMaxY",
    "maxParticleSpeed",
    "finalMaxParticleSpeed",
    "maxCentroidDrop",
)
PERF_INT_KEYS = (
    "warmupFrames",
    "profileFrames",
    "softWorkers",
    "requestedOuterIterations",
    "requestedInnerIterations",
    "executedOuterIterations",
    "executedInnerIterations",
    "particleSweeps",
    "workspaceGrowthEvents",
    "workspaceGrowthBytes",
    "detectionCalls",
    "bodyPairs",
    "overlappingBodyPairs",
    "particleSurfaceCandidates",
    "insideTriangleTests",
    "closestTriangleTests",
    "selfTriangleTests",
    "rigidParticleBoxTests",
    "generatedGroundContacts",
    "generatedRigidContacts",
    "generatedSoftContacts",
    "generatedSelfContacts",
)
PERF_FLOAT_KEYS = (
    "avgStepMs",
    "p50StepMs",
    "p95StepMs",
    "maxStepMs",
    "initialContactMs",
    "solverMs",
    "sceneMs",
    "metricsMs",
    "predictionMs",
    "contactIndexMs",
    "bodyPrecomputeMs",
    "bodySolveMs",
    "particleSolveMs",
    "projectionMs",
    "dualMs",
    "redetectMs",
    "velocityMs",
    "frictionMs",
    "solverUnattributedMs",
    "closureMs",
    "finalMaxDisplacement",
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


def run_one(
    case_name: str,
    repeat: int,
    bin_dir: Path,
    frames: int,
    timeout: float,
    execution: str,
    warmup: int,
) -> tuple[bool, dict[str, str], dict[str, str]]:
    name = f"{case_name}-r{repeat}"
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        "--solver=avbd",
        f"--case={case_name}",
        f"--execution={execution}",
        f"--frames={frames}",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = "avbd"
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(frames)
    env["PHYSX_AVBD_PROFILE_WARMUP"] = str(warmup)
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
    perf_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_PERF] ")
    ]
    errors: list[str] = []
    fields: dict[str, str] = {}
    perf_fields: dict[str, str] = {}
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
    if len(perf_lines) != 1:
        errors.append(f"perf count is {len(perf_lines)}, expected exactly 1")
    else:
        perf_fields, parse_errors = parse_gate(perf_lines[0])
        errors.extend(parse_errors)
    required = {
        "schema": "1",
        "snippet": "SnippetDeformableVolumeAVBD",
        "case": case_name,
        "solver": "avbd",
        "validation": "COMPONENT_GATED",
        "sceneSoftIntegration": "0",
        "status": "PASS",
        "initialized": "1",
        "frames": str(frames),
        "fetchFailures": "0",
        "sceneDynamics": "0",
        "sceneDeformableVolumes": "0",
        "nonFiniteParticleSamples": "0",
        "invertedElementSamples": "0",
        "solverReadbackMatched": "1",
        "fatalErrors": "0",
        "cleanupComplete": "1",
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )
    for key in INT_KEYS:
        try:
            if int(fields[key]) < 0:
                errors.append(f"{key} is negative")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected integer")
    for key in FLOAT_KEYS:
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"{key}={fields.get(key)!r}, expected finite float"
            )
    perf_required = {
        "schema": "1",
        "snippet": "SnippetDeformableVolumeAVBD",
        "case": case_name,
        "softExecution": "serial",
        "softWorkers": "1",
        "warmupFrames": str(warmup),
        "profileFrames": str(frames - warmup),
    }
    for key, expected in perf_required.items():
        if perf_fields.get(key) != expected:
            errors.append(
                f"perf {key}={perf_fields.get(key)!r}, "
                f"expected {expected!r}"
            )
    for key in PERF_INT_KEYS:
        try:
            if int(perf_fields[key]) < 0:
                errors.append(f"perf {key} is negative")
        except (KeyError, ValueError):
            errors.append(
                f"perf {key}={perf_fields.get(key)!r}, expected integer"
            )
    for key in PERF_FLOAT_KEYS:
        try:
            if not math.isfinite(float(perf_fields[key])):
                errors.append(f"perf {key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"perf {key}={perf_fields.get(key)!r}, "
                "expected finite float"
            )
    try:
        if float(perf_fields["closureMs"]) > float(
            perf_fields["avgStepMs"]
        ) * 1.05:
            errors.append("perf closure exceeds avgStepMs by more than 5%")
    except (KeyError, ValueError):
        pass
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")
    print(
        "[DEFORMABLE_VOLUME_AVBD_RUN] "
        f"name={name} status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUN_ERROR] "
            f"name={name} error={error}"
        )
    return not errors, fields, perf_fields


def compare_repeats(
    case_name: str, first: dict[str, str], second: dict[str, str]
) -> bool:
    mismatches = [
        key for key in INT_KEYS if first.get(key) != second.get(key)
    ]
    for key in FLOAT_KEYS:
        try:
            if abs(float(first[key]) - float(second[key])) > 1e-5:
                mismatches.append(key)
        except (KeyError, ValueError):
            mismatches.append(key)
    passed = not mismatches
    print(
        "[DEFORMABLE_VOLUME_AVBD_REPEAT] "
        f"case={case_name} status={'PASS' if passed else 'FAIL'} "
        f"mismatches={','.join(mismatches) if mismatches else 'none'}"
    )
    return passed


def summarize_performance(
    case_name: str,
    results: list[dict[str, str]],
    enforce_gate: bool,
) -> bool:
    try:
        avg_median = statistics.median(
            float(result["avgStepMs"]) for result in results
        )
        p95_median = statistics.median(
            float(result["p95StepMs"]) for result in results
        )
        solver_median = statistics.median(
            float(result["solverMs"]) for result in results
        )
        particle_median = statistics.median(
            float(result["particleSolveMs"]) for result in results
        )
    except (KeyError, ValueError, statistics.StatisticsError) as error:
        print(
            "[DEFORMABLE_VOLUME_AVBD_PERF_ERROR] "
            f"case={case_name} error={error}"
        )
        return False
    gate_passed = (
        not enforce_gate
        or (avg_median <= 16.67 and p95_median <= 33.33)
    )
    print(
        "[DEFORMABLE_VOLUME_AVBD_PERF_SUMMARY] "
        f"case={case_name} repeats={len(results)} "
        f"medianAvgStepMs={avg_median:.9g} "
        f"medianP95StepMs={p95_median:.9g} "
        f"medianSolverMs={solver_median:.9g} "
        f"medianParticleSolveMs={particle_median:.9g} "
        f"gate={'ENFORCED' if enforce_gate else 'BASELINE'} "
        f"status={'PASS' if gate_passed else 'FAIL'}"
    )
    return gate_passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "probe",
            "acceptance",
            "performance-baseline",
            "performance-acceptance",
        ),
        default="probe",
    )
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument(
        "--execution",
        choices=("parallel", "sequential"),
        default="sequential",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        print("[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] frames must be positive")
        return 2
    performance_mode = args.mode.startswith("performance-")
    warmup = args.warmup if args.warmup is not None else (
        30 if performance_mode else 0
    )
    if warmup < 0 or warmup >= args.frames:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "warmup must be non-negative and less than frames"
        )
        return 2
    repeats = args.repeats if args.repeats is not None else (
        3 if performance_mode else (2 if args.mode == "acceptance" else 1)
    )
    if repeats <= 0:
        print("[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] repeats must be positive")
        return 2
    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            f"missing executable: {executable}"
        )
        return 2
    selected_cases = (args.case,) if args.case else (
        ("current-all",) if performance_mode else CASES
    )
    passed = True
    results: dict[tuple[str, int], dict[str, str]] = {}
    perf_results: dict[tuple[str, int], dict[str, str]] = {}
    for repeat in range(1, repeats + 1):
        for case_name in selected_cases:
            run_passed, fields, perf_fields = run_one(
                case_name,
                repeat,
                bin_dir,
                args.frames,
                args.timeout,
                args.execution,
                warmup,
            )
            passed = passed and run_passed
            results[(case_name, repeat)] = fields
            perf_results[(case_name, repeat)] = perf_fields
    if args.mode == "acceptance":
        for case_name in selected_cases:
            passed = (
                compare_repeats(
                    case_name,
                    results[(case_name, 1)],
                    results[(case_name, 2)],
                )
                and passed
            )
    if performance_mode:
        for case_name in selected_cases:
            passed = (
                summarize_performance(
                    case_name,
                    [
                        perf_results[(case_name, repeat)]
                        for repeat in range(1, repeats + 1)
                    ],
                    args.mode == "performance-acceptance",
                )
                and passed
            )
    print(
        "[DEFORMABLE_VOLUME_AVBD_SUMMARY] "
        f"mode={args.mode} cases={len(selected_cases)} runs="
        f"{len(selected_cases) * repeats} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
