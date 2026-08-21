#!/usr/bin/env python3
"""Run reproducible CPU AVBD deformable-surface scene-step benchmarks.

This runner is deliberately separate from the broad correctness runner.  It
only consumes ``[AVBD_PERF] schema=2`` emitted by the snippet, so executable
startup, teardown, and gate sampling never become solver timing data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "profile"
)
EXECUTABLE = "SnippetDeformableSurfaceAVBD_64.exe"
MEASUREMENT_SOURCE_PATHS = (
    "physx/snippets/snippetdeformablesurfaceavbd/"
    "SnippetDeformableSurfaceAVBD.cpp",
    "tools/run_snippet_deformable_surface_avbd_performance.py",
    "tools/snippet_headless_process.py",
    "physx/include/PxAvbdCpuIsa.h",
    "physx/source/physx/src/NpAvbdCpuIsa.cpp",
    "physx/include/PxSimulationStatistics.h",
    "physx/source/simulationcontroller/src/ScScene.cpp",
    "physx/source/lowleveldynamics/src/DyAvbdSoftBodyComponent.h",
)

# These are correctness-owned, named fixtures.  The corpus is intentionally
# explicit: later phase work must not silently substitute a smaller topology
# or a flag-off control when comparing results.
PERFORMANCE_CASES = (
    "surface-lifecycle",
    "surface-performance-dense-no-contact",
    "surface-ground",
    "surface-soft-soft-wake",
    "surface-self-collision",
)
MIN_TOTAL_FRAMES = {
    # The gates are evaluated over all frames (including warmup), so this is
    # deliberately a total-frame requirement rather than a sample-count rule.
    "surface-ground": 150,
    "surface-soft-soft-wake": 150,
    "surface-self-collision": 125,
}
STRING_KEYS = (
    "schema",
    "snippet",
    "case",
    "buildProfile",
    "requestedIsa",
    "selectedIsa",
    "compiledIsaBackends",
    "isaKernelSelfTest",
    "sceneExecution",
    "softScheduler",
    "softExecution",
    "status",
)
INT_KEYS = (
    "fmaSupported",
    "fmaUsed",
    "forceIsaRejected",
    "dispatcherThreads",
    "physicalCores",
    "actualSoftWorkers",
    "taskCount",
    "barrierCount",
    "topologySoftBodies",
    "topologySoftParticles",
    "topologyTriElements",
    "topologySurfaceTriangles",
    "topologySurfaceVertices",
    "warmupFrames",
    "profileFrames",
    "workspaceGrowthEvents",
    "workspaceGrowthBytes",
    "contactWorkspaceGrowthEvents",
    "contactWorkspaceGrowthBytes",
    "contactSweepScratchGrowthEvents",
    "contactSweepScratchGrowthBytes",
    "contactOutputGrowthEvents",
    "contactOutputGrowthBytes",
    "peakContactOutputCount",
    "peakContactOutputCapacity",
    "peakContactIncidenceCount",
    "peakContactIncidenceCapacity",
    "peakStateTransferContactCount",
    "peakStateTransferContactCapacity",
    "peakStateTransferUsedCapacity",
    "collisionDetectionCalls",
    "collisionBodyPairs",
    "collisionOverlappingBodyPairs",
    "collisionParticleSurfaceCandidates",
    "collisionInsideTriangleTests",
    "collisionClosestTriangleTests",
    "collisionSelfTriangleTests",
    "collisionSelfTriangleBoundsBuilt",
    "collisionSelfVertexSweepEntriesBuilt",
    "collisionSelfEdgeBoundsBuilt",
    "collisionSurfaceBvhRefitNodes",
    "collisionSurfaceBvhCandidates",
    "collisionSurfaceEdgeBvhRefitNodes",
    "collisionSurfaceEdgeBvhCandidates",
    "collisionRigidParticleTests",
    "collisionGeneratedGroundContacts",
    "collisionGeneratedRigidContacts",
    "collisionGeneratedSoftContacts",
    "collisionGeneratedSelfContacts",
    "componentFallbackSteps",
    "nativeIslandSteps",
)
FLOAT_KEYS = (
    "isaProbeValue",
    "avgStepMs",
    "p50StepMs",
    "p95StepMs",
    "maxStepMs",
    "sceneMs",
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


def get_source_revision() -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def get_measurement_source_hash() -> str:
    """Hash the files which define the observed surface measurement contract."""
    digest = hashlib.sha256()
    try:
        for relative_path in MEASUREMENT_SOURCE_PATHS:
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update((REPO_ROOT / relative_path).read_bytes())
            digest.update(b"\0")
    except OSError:
        return "unknown"
    return digest.hexdigest()


def validate(
    fields: dict[str, str],
    case_name: str,
    execution: str,
    dispatcher_threads: int,
    warmup_frames: int,
    profile_frames: int,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema": "2",
        "snippet": "SnippetDeformableSurfaceAVBD",
        "case": case_name,
        "sceneExecution": execution,
        "status": "PASS",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            errors.append(f"{key}={fields.get(key)!r}, expected {value!r}")
    for key in STRING_KEYS:
        if not fields.get(key):
            errors.append(f"{key} is missing or empty")
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
            errors.append(f"{key}={fields.get(key)!r}, expected finite float")
    expected_ints = {
        "dispatcherThreads": dispatcher_threads,
        "warmupFrames": warmup_frames,
        "profileFrames": profile_frames,
    }
    for key, value in expected_ints.items():
        try:
            actual = int(fields[key])
        except (KeyError, ValueError):
            continue
        if actual != value:
            errors.append(f"{key}={actual}, expected {value}")
    if fields.get("requestedIsa") not in {"auto", "sse2", "avx2fma", "invalid"}:
        errors.append(
            f"requestedIsa={fields.get('requestedIsa')!r}, expected known ISA mode")
    if fields.get("selectedIsa") not in {"sse2", "avx2fma"}:
        errors.append(
            f"selectedIsa={fields.get('selectedIsa')!r}, expected executable ISA mode")
    if "sse2" not in fields.get("compiledIsaBackends", "").split(","):
        errors.append("compiledIsaBackends must include sse2")
    if fields.get("isaKernelSelfTest") != "pass":
        errors.append(
            f"isaKernelSelfTest={fields.get('isaKernelSelfTest')!r}, expected 'pass'")
    try:
        if float(fields["isaProbeValue"]) != 36.0:
            errors.append(f"isaProbeValue={fields['isaProbeValue']!r}, expected 36.0")
    except (KeyError, ValueError):
        pass
    try:
        selected_avx2_fma = fields["selectedIsa"] == "avx2fma"
        if int(fields["fmaUsed"]) != int(selected_avx2_fma):
            errors.append(
                f"fmaUsed={fields['fmaUsed']!r}, inconsistent with "
                f"selectedIsa={fields['selectedIsa']!r}")
    except (KeyError, ValueError):
        pass
    scheduler_execution = {
        "sceneAvbd": "scene",
        "sceneTaskgraph": "serial",
        "componentSerial": "serial",
    }
    actual_scheduler = fields.get("softScheduler")
    if actual_scheduler not in scheduler_execution:
        errors.append(
            f"softScheduler={actual_scheduler!r}, expected known AVBD scheduler")
    elif fields.get("softExecution") != scheduler_execution[actual_scheduler]:
        errors.append(
            f"softExecution={fields.get('softExecution')!r}, inconsistent with "
            f"softScheduler={actual_scheduler!r}")
    if int(fields.get("topologySurfaceVertices", "0")) == 0:
        errors.append("topologySurfaceVertices must be positive")
    if int(fields.get("topologySurfaceTriangles", "0")) == 0:
        errors.append("topologySurfaceTriangles must be positive")
    if case_name == "surface-performance-dense-no-contact":
        for key, expected in (
            ("topologySoftBodies", 1),
            ("topologySoftParticles", 4225),
            ("topologyTriElements", 8192),
            ("topologySurfaceVertices", 4225),
            ("topologySurfaceTriangles", 8192),
        ):
            try:
                if int(fields[key]) != expected:
                    errors.append(f"{key}={fields[key]}, expected {expected}")
            except (KeyError, ValueError):
                pass
    if float(fields.get("avgStepMs", "nan")) <= 0.0:
        errors.append("avgStepMs must be positive")
    # P0 records growth after the fixed warmup.  Zero is P1's optimization
    # exit criterion, not a P0 admission filter: rejecting a current-source
    # baseline here would hide the capacity-growth work P1 must remove.
    if case_name in (
        "surface-performance-dense-no-contact",
        "surface-self-collision",
        "surface-soft-soft-wake",
    ):
        component_steps = int(fields.get("componentFallbackSteps", "0"))
        native_steps = int(fields.get("nativeIslandSteps", "0"))
        if component_steps + native_steps == 0:
            errors.append(
                "no AVBD soft execution owner stepped the active corpus"
            )
    return errors


def run_once(args: argparse.Namespace, repeat: int) -> tuple[bool, dict[str, str]]:
    total_frames = args.warmup_frames + args.profile_frames
    executable = args.bin_dir / EXECUTABLE
    argv = (
        str(executable),
        "--headless",
        "--solver=avbd",
        f"--case={args.case}",
        f"--execution={args.execution}",
        f"--frames={total_frames}",
        "--dt=0.0166666675",
        f"--dispatcher-threads={args.dispatcher_threads}",
        "--seed=1",
    )
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = "avbd"
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(total_frames)
    env["PHYSX_AVBD_PROFILE_WARMUP"] = str(args.warmup_frames)
    if args.collision_telemetry:
        env["PHYSX_AVBD_COLLISION_TELEMETRY"] = "1"
    if args.surface_triangle_bvh == "off":
        env["PHYSX_AVBD_SURFACE_TRIANGLE_BVH"] = "0"
    else:
        env.pop("PHYSX_AVBD_SURFACE_TRIANGLE_BVH", None)
    if args.surface_edge_bvh == "off":
        env["PHYSX_AVBD_SURFACE_EDGE_BVH"] = "0"
    else:
        env.pop("PHYSX_AVBD_SURFACE_EDGE_BVH", None)
    result = run_headless_process(
        argv, cwd=args.bin_dir, env=env, timeout_seconds=args.timeout
    )
    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + result.stderr
    errors: list[str] = []
    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append("visible window detected")
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")
    lines = [
        line.strip() for line in output.splitlines()
        if line.startswith("[AVBD_PERF] ")
    ]
    fields: dict[str, str] = {}
    if len(lines) != 1:
        errors.append(f"performance line count {len(lines)}, expected 1")
    else:
        fields, parse_errors = parse_fields(lines[0])
        errors.extend(parse_errors)
        errors.extend(
            validate(
                fields, args.case, args.execution,
                args.dispatcher_threads, args.warmup_frames,
                args.profile_frames,
            )
        )
    label = f"{args.case}-r{repeat}"
    if errors:
        print(f"[FAIL] {label}: " + "; ".join(errors))
        if output:
            print(output.rstrip())
        return False, fields
    print(
        f"[PASS] {label}: avgStepMs={fields['avgStepMs']} "
        f"p95StepMs={fields['p95StepMs']} "
        f"sceneMs={fields['sceneMs']} "
        f"profileFrames={fields['profileFrames']}"
    )
    return True, fields


def write_json(
    path: Path, args: argparse.Namespace, results: list[dict[str, str]]
) -> None:
    summary = {
        f"median{key[0].upper()}{key[1:]}": statistics.median(
            float(fields[key]) for fields in results
        )
        for key in ("avgStepMs", "p50StepMs", "p95StepMs", "maxStepMs", "sceneMs")
    }
    payload = {
        "schema": 1,
        "sourceRevision": get_source_revision(),
        "measurementSourceSha256": get_measurement_source_hash(),
        "measurementSourcePaths": MEASUREMENT_SOURCE_PATHS,
        "runner": "run_snippet_deformable_surface_avbd_performance.py",
        "config": {
            "case": args.case,
            "warmupFrames": args.warmup_frames,
            "profileFrames": args.profile_frames,
            "repeats": args.repeats,
            "collisionTelemetry": args.collision_telemetry,
            "surfaceTriangleBvh": args.surface_triangle_bvh,
            "surfaceEdgeBvh": args.surface_edge_bvh,
            "execution": args.execution,
            "dispatcherThreads": args.dispatcher_threads,
            "binDir": str(args.bin_dir.resolve()),
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logicalCores": os.cpu_count(),
            "physicalCores": results[0].get("physicalCores", "unknown"),
        },
        "summary": summary,
        "repeats": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DEFORMABLE_SURFACE_AVBD_PERF_JSON] path={path.resolve()} status=PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--case", choices=PERFORMANCE_CASES, default="surface-lifecycle")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--profile-frames", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dispatcher-threads", type=int, default=1)
    parser.add_argument(
        "--collision-telemetry",
        action="store_true",
        help="Enable diagnostic OGC counters; exclude these runs from timing denominators.",
    )
    parser.add_argument(
        "--surface-triangle-bvh",
        choices=("on", "off"),
        default="on",
        help=(
            "Use the compiled surface-triangle BVH, or retain the exact "
            "legacy traversal for same-binary diagnostic comparison."
        ),
    )
    parser.add_argument(
        "--surface-edge-bvh",
        choices=("on", "off"),
        default="on",
        help=(
            "Use the compiled surface-edge BVH, or retain the exact legacy "
            "edge sweep for same-binary diagnostic comparison."
        ),
    )
    parser.add_argument("--execution", choices=("parallel", "sequential"), default="sequential")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--performance-json", type=Path)
    args = parser.parse_args()
    if args.warmup_frames < 0 or args.profile_frames <= 0:
        parser.error("warmup must be non-negative and profile frames positive")
    min_total_frames = MIN_TOTAL_FRAMES.get(args.case, 1)
    if args.warmup_frames + args.profile_frames < min_total_frames:
        parser.error(
            f"{args.case} requires at least {min_total_frames} total frames "
            "to exercise its correctness event"
        )
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not 1 <= args.dispatcher_threads <= 256:
        parser.error("--dispatcher-threads must be in [1, 256]")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.bin_dir = args.bin_dir.resolve()
    if not (args.bin_dir / EXECUTABLE).is_file():
        print(f"[FAIL] executable not found: {args.bin_dir / EXECUTABLE}")
        return 2
    passed = True
    results: list[dict[str, str]] = []
    for repeat in range(1, args.repeats + 1):
        one_passed, fields = run_once(args, repeat)
        passed = one_passed and passed
        if one_passed:
            results.append(fields)
    if passed and args.performance_json is not None:
        write_json(args.performance_json, args, results)
    print(
        "[DEFORMABLE_SURFACE_AVBD_PERF_SUMMARY] "
        f"case={args.case} repeats={args.repeats} status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
