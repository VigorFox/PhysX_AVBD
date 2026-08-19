#!/usr/bin/env python3
"""Capture an interleaved CPU TGS/AVBD baseline on the HelloGRB workload.

The snippet owns the fixed fixture and times only ``simulate + fetchResults``.
This runner alternates solver order to reduce thermal/order bias.  It records a
baseline; it deliberately does not require either solver to win.  The default
``independent`` layout preserves the historical authority, while
``--rigid-stress-layout=connected`` exercises the large-island CPU backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess

from snippet_headless_process import run_headless_process


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = ROOT / "physx/bin/win.x86_64.vc143.md/release"
EXECUTABLE = "SnippetHelloWorld_64.exe"
CASE = "rigid-stress"
SCHEMA = "4"
AVBD_ITERATION_POLICY = "scene-desc"
AVBD_ITERATION_SOURCE = "default"
AVBD_ITERATIONS = 4
AVBD_ITERATION_SEMANTICS = "budgeted-complete-primal-dual-stiffness"
AVBD_JOINT_ITERATION_OVERRIDE_SOURCE = "default"
AVBD_JOINT_ITERATION_OVERRIDE = 8
AVBD_EARLY_STOP_SOURCE = "default"
AVBD_EARLY_STOP_ENABLED = 1
AVBD_EARLY_STOP_ACTIVE = 0
FIXTURE_FIELDS = {
    "schema": SCHEMA,
    "snippet": "SnippetHelloWorld",
    "case": CASE,
    "buildProfile": "release",
    "cpuOnly": "1",
    "gpuDynamics": "0",
    "broadphase": "cpu",
    "pvd": "0",
    "stacks": "40",
    "stackSize": "20",
    "rigidBoxes": "8400",
    "projectileCount": "1",
    "projectileRadius": "5",
    "projectileDensity": "1000",
    "actorPositionIterations": "4",
    "actorVelocityIterations": "1",
    "avbdIterationPolicy": AVBD_ITERATION_POLICY,
    "avbdIterationSource": AVBD_ITERATION_SOURCE,
    "avbdIterations": str(AVBD_ITERATIONS),
    "avbdIterationSemantics": AVBD_ITERATION_SEMANTICS,
    "avbdJointIterationOverrideSource": (
        AVBD_JOINT_ITERATION_OVERRIDE_SOURCE
    ),
    "avbdJointIterationOverride": str(AVBD_JOINT_ITERATION_OVERRIDE),
    "avbdJointIterationOverrideActive": "0",
    "avbdEarlyStopSource": AVBD_EARLY_STOP_SOURCE,
    "avbdEarlyStopEnabled": str(AVBD_EARLY_STOP_ENABLED),
    "avbdEarlyStopActive": str(AVBD_EARLY_STOP_ACTIVE),
    "measurement": "simulate-fetch",
    "instrumentation": "none",
    "status": "PASS",
}
TIMING_FIELDS = ("avgStepMs", "p50StepMs", "p95StepMs", "maxStepMs")
STATE_DIGEST_PATTERN = re.compile(r"[0-9a-f]{32}")
BUILD_ARTIFACT_NAMES = (
    EXECUTABLE,
    "PhysX_64.dll",
    "PhysXCommon_64.dll",
    "PhysXFoundation_64.dll",
)
CONTRACT_EXACT_PATHS = (
    "physx/include/PxSceneDesc.h",
    "physx/include/PxSimulationStatistics.h",
    "physx/snippets/snippethelloworld/SnippetHelloWorld.cpp",
    "physx/snippets/snippethelloworld/SnippetHelloWorldRender.cpp",
    "physx/snippets/snippetcommon/SnippetHeadless.h",
    "physx/CMakeLists.txt",
    "physx/source/compiler/cmake/CMakeLists.txt",
    "physx/source/compiler/cmake/LowLevelDynamics.cmake",
    "physx/source/compiler/cmake/PhysX.cmake",
    "physx/source/compiler/cmake/windows/CMakeLists.txt",
    "physx/source/simulationcontroller/src/ScScene.cpp",
    "tools/run_avbd_rigid_stress_baseline.py",
    "tools/snippet_headless_process.py",
)
CONTRACT_GLOBS = (
    "physx/source/lowleveldynamics/src/DyAvbd*.cpp",
    "physx/source/lowleveldynamics/src/DyAvbd*.h",
    "physx/source/lowleveldynamics/src/DyAvbd*.inl",
    "physx/source/lowleveldynamics/src/DyTGS*.cpp",
    "physx/source/lowleveldynamics/src/DyTGS*.h",
    "physx/source/lowleveldynamics/src/DyDynamics*.cpp",
    "physx/source/lowleveldynamics/src/DyDynamics*.h",
    "physx/source/lowleveldynamics/src/DySolver*.cpp",
    "physx/source/lowleveldynamics/src/DySolver*.h",
    "physx/source/lowleveldynamics/shared/**/*.h",
)


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            raise RuntimeError(f"malformed result token: {token!r}")
        key, value = token.split("=", 1)
        if key in fields:
            raise RuntimeError(f"duplicate result field: {key}")
        fields[key] = value
    return fields


def only_result(stdout: str, prefix: str) -> dict[str, str]:
    lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(
            f"expected one {prefix.strip()} line, observed {len(lines)}"
        )
    return parse_fields(lines[0])


def integer(fields: dict[str, str], key: str) -> int:
    try:
        return int(fields[key])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"missing/integer-invalid field {key!r}") from error


def number(fields: dict[str, str], key: str) -> float:
    try:
        value = float(fields[key])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"missing/float-invalid field {key!r}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite field {key!r}")
    return value


def source_identity() -> dict[str, object]:
    try:
        head = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            (
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ),
            cwd=ROOT,
        )
        tracked_diff = subprocess.check_output(
            ("git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"),
            cwd=ROOT,
        )
        untracked_output = subprocess.check_output(
            (
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ),
            cwd=ROOT,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("unable to capture git source identity") from error
    status_hash = hashlib.sha256(status).hexdigest()
    diff_hash = hashlib.sha256(tracked_diff).hexdigest()
    untracked_names = sorted(
        name for name in untracked_output.split(b"\0") if name
    )
    untracked_path_bytes = b"\0".join(untracked_names)
    untracked_hash = hashlib.sha256(untracked_path_bytes).hexdigest()
    identity_digest = hashlib.sha256()
    identity_digest.update(status)
    identity_digest.update(b"\0")
    identity_digest.update(tracked_diff)
    identity_digest.update(b"\0")
    identity_digest.update(untracked_path_bytes)
    identity_hash = identity_digest.hexdigest()
    dirty = bool(status)
    revision = f"{head}+dirty.{identity_hash[:12]}" if dirty else head
    return {
        "headRevision": head,
        "revision": revision,
        "worktreeDirty": dirty,
        "worktreeStatusHash": status_hash,
        "worktreeDiffHash": diff_hash,
        "untrackedPathHash": untracked_hash,
        "untrackedFileCount": len(untracked_names),
        "worktreeIdentityHash": identity_hash,
        "worktreeIdentityScope": "tracked-diff+untracked-paths",
    }


def hash_named_files(paths: list[Path]) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    entries: dict[str, str] = {}
    for path in sorted(paths, key=lambda item: item.as_posix()):
        data = path.read_bytes()
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            name = str(path.resolve())
        file_hash = hashlib.sha256(data).hexdigest()
        entries[name] = file_hash
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest(), entries


def contract_paths() -> list[Path]:
    paths = {ROOT / relative for relative in CONTRACT_EXACT_PATHS}
    for pattern in CONTRACT_GLOBS:
        paths.update(ROOT.glob(pattern))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "missing measurement contract input(s): "
            + ", ".join(str(path) for path in sorted(missing))
        )
    return sorted(paths)


def contract_identity() -> tuple[str, dict[str, str]]:
    return hash_named_files(contract_paths())


def build_identity(executable: Path) -> tuple[str, dict[str, str]]:
    artifacts = [executable.parent / name for name in BUILD_ARTIFACT_NAMES]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise RuntimeError(
            "missing build identity artifact(s): "
            + ", ".join(str(path) for path in missing)
        )
    return hash_named_files(artifacts)


def correctness_fingerprint(gate: dict[str, str]) -> str:
    # Fingerprint every correctness/gate field, including the all-actor state
    # digest, but never include timing output from AVBD_RIGID_PERF.
    encoded = json.dumps(gate, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_result(
    gate: dict[str, str],
    perf: dict[str, str],
    work: dict[str, str],
    solver: str,
    frames: int,
    warmup_frames: int,
    dispatcher_threads: int,
    work_attribution: bool,
    layout: str,
) -> None:
    for key, expected in FIXTURE_FIELDS.items():
        if key == "buildProfile" and work_attribution:
            if perf.get(key) not in {"debug", "checked", "profile"}:
                raise RuntimeError(
                    "work attribution requires an instrumented build, "
                    f"observed buildProfile={perf.get(key)!r}"
                )
            continue
        if key == "instrumentation" and work_attribution:
            expected = "work-attribution"
        if perf.get(key) != expected:
            raise RuntimeError(
                f"perf {key}={perf.get(key)!r}, expected {expected!r}"
            )
    expected_fields = {
        "solver": solver,
        "sceneExecution": "parallel",
        "layout": layout,
        "enhancedDeterminismRequested": "0",
        "enhancedDeterminismObserved": "0",
        "avbdBackendPolicy": "fast",
        "dispatcherThreads": str(dispatcher_threads),
        "avbdIterationActive": "1" if solver == "avbd" else "0",
        "avbdEarlyStopActive": str(AVBD_EARLY_STOP_ACTIVE),
        "warmupFrames": str(warmup_frames),
        "profileFrames": str(frames - warmup_frames),
    }
    for key, expected in expected_fields.items():
        if perf.get(key) != expected:
            raise RuntimeError(
                f"perf {key}={perf.get(key)!r}, expected {expected!r}"
            )
    for key in TIMING_FIELDS:
        if number(perf, key) <= 0.0:
            raise RuntimeError(f"perf {key} must be positive")
    gate_expected = {
        "schema": SCHEMA,
        "snippet": "SnippetHelloWorld",
        "case": CASE,
        "solver": solver,
        "execution": "parallel",
        "layout": layout,
        "enhancedDeterminismRequested": "0",
        "enhancedDeterminismObserved": "0",
        "avbdBackendPolicy": "fast",
        "requestedFrames": str(frames),
        "completedFrames": str(frames),
        "dispatcherThreads": str(dispatcher_threads),
        "avbdIterationPolicy": AVBD_ITERATION_POLICY,
        "avbdIterationSource": AVBD_ITERATION_SOURCE,
        "avbdIterations": str(AVBD_ITERATIONS),
        "avbdIterationActive": "1" if solver == "avbd" else "0",
        "avbdIterationSemantics": AVBD_ITERATION_SEMANTICS,
        "avbdJointIterationOverrideSource": (
            AVBD_JOINT_ITERATION_OVERRIDE_SOURCE
        ),
        "avbdJointIterationOverride": str(AVBD_JOINT_ITERATION_OVERRIDE),
        "avbdJointIterationOverrideActive": "0",
        "avbdEarlyStopSource": AVBD_EARLY_STOP_SOURCE,
        "avbdEarlyStopEnabled": str(AVBD_EARLY_STOP_ENABLED),
        "avbdEarlyStopActive": str(AVBD_EARLY_STOP_ACTIVE),
        "status": "PASS",
        "reason": "none",
        "boxCount": "8400",
        "projectileCount": "1",
        "finiteBoxes": "8400",
        "finalBallFinite": "1",
        "stateDigestAlgorithm": "fnv1a64x2-v1",
        "stateDigestActorCount": "8401",
    }
    for key, expected in gate_expected.items():
        if gate.get(key) != expected:
            raise RuntimeError(
                f"gate {key}={gate.get(key)!r}, expected {expected!r}"
            )
    if integer(gate, "movedBoxes") <= 0:
        raise RuntimeError("stress fixture did not move any box")
    state_digest = gate.get("stateDigest", "")
    if not STATE_DIGEST_PATTERN.fullmatch(state_digest):
        raise RuntimeError(f"gate stateDigest is invalid: {state_digest!r}")

    work_expected = {
        "schema": SCHEMA,
        "snippet": "SnippetHelloWorld",
        "case": CASE,
        "solver": solver,
        "profileFrames": str(frames - warmup_frames),
        "actorPositionIterations": "4",
        "actorVelocityIterations": "1",
        "avbdIterationPolicy": AVBD_ITERATION_POLICY,
        "avbdIterationSource": AVBD_ITERATION_SOURCE,
        "avbdIterations": str(AVBD_ITERATIONS),
        "avbdIterationActive": "1" if solver == "avbd" else "0",
        "avbdIterationSemantics": AVBD_ITERATION_SEMANTICS,
        "avbdJointIterationOverrideSource": (
            AVBD_JOINT_ITERATION_OVERRIDE_SOURCE
        ),
        "avbdJointIterationOverride": str(AVBD_JOINT_ITERATION_OVERRIDE),
        "avbdJointIterationOverrideActive": "0",
        "avbdEarlyStopSource": AVBD_EARLY_STOP_SOURCE,
        "avbdEarlyStopEnabled": str(AVBD_EARLY_STOP_ENABLED),
        "avbdEarlyStopActive": str(AVBD_EARLY_STOP_ACTIVE),
        "avbdLocalSolveCount": "UNAVAILABLE",
        "localSolveTelemetry": "UNAVAILABLE",
    }
    for key, expected in work_expected.items():
        if work.get(key) != expected:
            raise RuntimeError(
                f"work {key}={work.get(key)!r}, expected {expected!r}"
            )
    if work_attribution:
        for key, expected in {
            "workTelemetry": "OBSERVED",
            "attributionMode": "profiler-zones",
            "profilerZoneBuild": "1",
        }.items():
            if work.get(key) != expected:
                raise RuntimeError(
                    f"work {key}={work.get(key)!r}, expected {expected!r}"
                )
        for key in (
            "awakeDynamicBodies",
            "discreteContactPairs",
            "contactPairsWithContacts",
        ):
            if integer(work, key) <= 0:
                raise RuntimeError(f"work {key} must be positive")
        if solver == "avbd":
            if integer(work, "avbdIslandSolves") <= 0:
                raise RuntimeError("AVBD attribution observed no island solves")
            if layout == "connected":
                for key in (
                    "avbdBodyColorPlans",
                    "avbdBodyColorPasses",
                    "avbdDualPasses",
                    "avbdDualRanges",
                ):
                    if integer(work, key) <= 0:
                        raise RuntimeError(
                            f"connected AVBD attribution observed no {key}"
                        )
            elif integer(work, "avbdInnerSweeps") <= 0:
                raise RuntimeError("AVBD attribution observed no inner sweeps")
        else:
            for key in (
                "avbdIslandSolves",
                "avbdInnerSweeps",
                "avbdBlockDescentZones",
                "avbdBodyColorPlans",
                "avbdBodyColorPasses",
                "avbdDualPasses",
                "avbdDualRanges",
            ):
                if integer(work, key) != 0:
                    raise RuntimeError(f"TGS unexpectedly emitted {key}")
    else:
        for key, expected in {
            "workTelemetry": "UNAVAILABLE",
            "attributionMode": "none",
            "avbdIslandSolves": "UNAVAILABLE",
            "avbdInnerSweeps": "UNAVAILABLE",
            "avbdBlockDescentZones": "UNAVAILABLE",
            "avbdBodyColorPlans": "UNAVAILABLE",
            "avbdBodyColorPasses": "UNAVAILABLE",
            "avbdDualPasses": "UNAVAILABLE",
            "avbdDualRanges": "UNAVAILABLE",
            "awakeDynamicBodies": "UNAVAILABLE",
            "sceneStatsActiveDynamicBodies": "UNAVAILABLE",
            "discreteContactPairs": "UNAVAILABLE",
            "contactPairsWithContacts": "UNAVAILABLE",
        }.items():
            if work.get(key) != expected:
                raise RuntimeError(
                    f"work {key}={work.get(key)!r}, expected {expected!r}"
                )


def run_once(
    executable: Path,
    solver: str,
    frames: int,
    warmup_frames: int,
    dispatcher_threads: int,
    timeout: float,
    affinity_mask: int,
    work_attribution: bool = False,
    layout: str = "independent",
) -> tuple[dict[str, str], dict[str, str], dict[str, str], int]:
    env = os.environ.copy()
    # Formal runs own the AVBD/Snippet process contract. Remove every legacy,
    # candidate, ISA-force and diagnostic knob before setting explicit args.
    for name in tuple(env):
        if name.startswith(
            ("PHYSX_AVBD_", "PX_AVBD_", "AVBD_", "PHYSX_SNIPPET_")
        ):
            env.pop(name, None)
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    command = [
        str(executable),
        "--headless",
        f"--solver={solver}",
        f"--case={CASE}",
        f"--rigid-stress-layout={layout}",
        "--enhanced-determinism=off",
        f"--frames={frames}",
        f"--warmup-frames={warmup_frames}",
        f"--dispatcher-threads={dispatcher_threads}",
        "--execution=parallel",
        "--dt=0.0166666675",
        "--seed=1",
    ]
    if work_attribution:
        command.append("--work-attribution")
    result = run_headless_process(
        command,
        cwd=executable.parent,
        env=env,
        timeout_seconds=timeout,
        cpu_affinity_mask=affinity_mask,
    )
    if (
        result.returncode
        or result.timed_out
        or result.visible_window_detected
        or result.cpu_affinity_mask != affinity_mask
    ):
        raise RuntimeError(
            f"solver={solver} exit={result.returncode} "
            f"timeout={int(result.timed_out)} "
            f"visibleWindow={int(result.visible_window_detected)} "
            f"affinity={result.cpu_affinity_mask!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    gate = only_result(result.stdout, "[AVBD_GATE] ")
    perf = only_result(result.stdout, "[AVBD_RIGID_PERF] ")
    work = only_result(result.stdout, "[AVBD_RIGID_WORK] ")
    validate_result(
        gate,
        perf,
        work,
        solver,
        frames,
        warmup_frames,
        dispatcher_threads,
        work_attribution,
        layout,
    )
    return gate, perf, work, affinity_mask


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values),
    }


def format_distribution(values: list[float]) -> str:
    item = distribution(values)
    return " ".join(f"{key}={value:.9g}" for key, value in item.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--dispatcher-threads", type=int, default=4)
    parser.add_argument(
        "--rigid-stress-layout",
        choices=("independent", "connected"),
        default="independent",
        dest="layout",
        help="rigid-stress topology; default preserves the historical layout",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--affinity-mask",
        type=lambda value: int(value, 0),
        help="Windows process affinity mask; default is all visible CPUs",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--attribution-bin-dir",
        type=Path,
        help=(
            "Checked/Profile/Debug bin directory for one untimed, "
            "profiler-zone work-attribution run per solver"
        ),
    )
    parser.add_argument(
        "--require-work-attribution",
        action="store_true",
        help="fail unless --attribution-bin-dir yields observed work telemetry",
    )
    args = parser.parse_args()
    if args.pairs < 7:
        raise RuntimeError("formal baseline requires at least 7 interleaved pairs")
    if args.frames < 90:
        raise RuntimeError("formal baseline requires at least 90 frames")
    if args.warmup_frames < 0 or args.frames - args.warmup_frames < 60:
        raise RuntimeError("formal baseline requires at least 60 profile frames")
    if args.dispatcher_threads < 1:
        raise RuntimeError("dispatcher thread count must be positive")
    if args.require_work_attribution and args.attribution_bin_dir is None:
        raise RuntimeError(
            "--require-work-attribution requires --attribution-bin-dir"
        )
    executable = args.bin_dir.resolve() / EXECUTABLE
    if not executable.is_file():
        raise RuntimeError(f"missing executable: {executable}")
    visible_cpus = min(os.cpu_count() or 1, 64)
    affinity_mask = (
        args.affinity_mask
        if args.affinity_mask is not None
        else (1 << visible_cpus) - 1
    )
    if affinity_mask <= 0:
        raise RuntimeError("affinity mask must be positive")

    source = source_identity()
    measurement_hash, contract_files = contract_identity()
    build_hash, build_artifacts = build_identity(executable)

    samples: dict[str, list[dict[str, str]]] = {"tgs": [], "avbd": []}
    gate_samples: dict[str, list[dict[str, str]]] = {"tgs": [], "avbd": []}
    work_samples: dict[str, list[dict[str, str]]] = {"tgs": [], "avbd": []}
    for pair in range(args.pairs):
        order = ("tgs", "avbd") if pair % 2 == 0 else ("avbd", "tgs")
        for solver in order:
            gate, perf, work, effective_mask = run_once(
                executable,
                solver,
                args.frames,
                args.warmup_frames,
                args.dispatcher_threads,
                args.timeout,
                affinity_mask,
                layout=args.layout,
            )
            samples[solver].append(perf)
            gate_samples[solver].append(gate)
            work_samples[solver].append(work)
            fingerprint = correctness_fingerprint(gate)
            print(
                "[AVBD_RIGID_BASELINE_RUN] "
                f"pair={pair + 1} solver={solver} "
                f"avgStepMs={number(perf, 'avgStepMs'):.9g} "
                f"p95StepMs={number(perf, 'p95StepMs'):.9g} "
                f"movedBoxes={integer(gate, 'movedBoxes')} "
                f"awakeBoxes={integer(gate, 'awakeBoxes')} "
                f"finalBallZ={number(gate, 'finalBallZ'):.9g} "
                f"stateDigest={gate['stateDigest']} "
                f"correctnessFingerprint={fingerprint} "
                f"affinityMask=0x{effective_mask:X} status=PASS",
                flush=True,
            )

    correctness_fingerprints = {
        solver: [correctness_fingerprint(gate) for gate in solver_gates]
        for solver, solver_gates in gate_samples.items()
    }
    state_digests = {
        solver: [gate["stateDigest"] for gate in solver_gates]
        for solver, solver_gates in gate_samples.items()
    }
    for solver in ("tgs", "avbd"):
        if len(set(correctness_fingerprints[solver])) != 1:
            raise RuntimeError(
                f"{solver} correctness fingerprint changed across paired runs"
            )
        if len(set(state_digests[solver])) != 1:
            raise RuntimeError(
                f"{solver} all-actor state digest changed across paired runs"
            )

    work_attribution: dict[str, object] = {
        "status": "UNAVAILABLE",
        "reason": "not-requested",
        "buildHash": None,
        "buildArtifacts": {},
        "samples": {},
    }
    if args.attribution_bin_dir is not None:
        attribution_executable = (
            args.attribution_bin_dir.resolve() / EXECUTABLE
        )
        if not attribution_executable.is_file():
            raise RuntimeError(
                f"missing attribution executable: {attribution_executable}"
            )
        attribution_build_hash, attribution_build_artifacts = build_identity(
            attribution_executable
        )
        attribution_samples: dict[str, object] = {}
        for solver in ("tgs", "avbd"):
            gate, perf, work, effective_mask = run_once(
                attribution_executable,
                solver,
                args.frames,
                args.warmup_frames,
                args.dispatcher_threads,
                args.timeout,
                affinity_mask,
                work_attribution=True,
                layout=args.layout,
            )
            attribution_samples[solver] = {
                "gate": gate,
                "performance": perf,
                "work": work,
                "correctnessFingerprint": correctness_fingerprint(gate),
                "affinityMask": f"0x{effective_mask:X}",
            }
            print(
                "[AVBD_RIGID_ATTRIBUTION] "
                f"solver={solver} "
                f"avbdIterationPolicy={work['avbdIterationPolicy']} "
                f"avbdIterationSource={work['avbdIterationSource']} "
                f"avbdIterations={work['avbdIterations']} "
                f"avbdIterationActive={work['avbdIterationActive']} "
                "avbdJointIterationOverride="
                f"{work['avbdJointIterationOverride']} "
                "avbdJointIterationOverrideActive="
                f"{work['avbdJointIterationOverrideActive']} "
                f"avbdEarlyStopEnabled={work['avbdEarlyStopEnabled']} "
                f"avbdEarlyStopActive={work['avbdEarlyStopActive']} "
                f"innerSweeps={work['avbdInnerSweeps']} "
                f"islandSolves={work['avbdIslandSolves']} "
                f"localSolves={work['avbdLocalSolveCount']} "
                f"contactPairs={work['contactPairsWithContacts']} "
                "status=OBSERVED",
                flush=True,
            )
        work_attribution = {
            "status": "OBSERVED",
            "reason": "none",
            "buildHash": attribution_build_hash,
            "buildArtifacts": attribution_build_artifacts,
            "samples": attribution_samples,
        }

    source_after = source_identity()
    contract_hash_after, _ = contract_identity()
    build_hash_after, _ = build_identity(executable)
    if source_after != source:
        raise RuntimeError("source/worktree identity changed during measurement")
    if contract_hash_after != measurement_hash:
        raise RuntimeError("measurement contract changed during measurement")
    if build_hash_after != build_hash:
        raise RuntimeError("release build identity changed during measurement")

    timings = {
        solver: {
            metric: [number(item, metric) for item in solver_samples]
            for metric in TIMING_FIELDS
        }
        for solver, solver_samples in samples.items()
    }
    tgs_median = statistics.median(timings["tgs"]["avgStepMs"])
    avbd_median = statistics.median(timings["avbd"]["avgStepMs"])
    avbd_over_tgs = avbd_median / tgs_median
    avbd_gain_percent = (tgs_median - avbd_median) * 100.0 / tgs_median
    revision = str(source["revision"])
    dirty_value = source["worktreeDirty"]
    dirty_token = "unknown" if dirty_value is None else str(int(bool(dirty_value)))
    print(
        "[AVBD_RIGID_BASELINE_SUMMARY] "
        f"schema={SCHEMA} revision={revision} "
        f"headRevision={source['headRevision']} worktreeDirty={dirty_token} "
        f"worktreeStatusHash={source['worktreeStatusHash']} "
        f"worktreeDiffHash={source['worktreeDiffHash']} "
        f"untrackedPathHash={source['untrackedPathHash']} "
        f"untrackedFileCount={source['untrackedFileCount']} "
        f"worktreeIdentityHash={source['worktreeIdentityHash']} "
        f"worktreeIdentityScope={source['worktreeIdentityScope']} "
        f"contractHash={measurement_hash} buildHash={build_hash} "
        f"host={platform.node() or 'unknown'} pairs={args.pairs} "
        f"frames={args.frames} warmupFrames={args.warmup_frames} "
        f"profileFrames={args.frames - args.warmup_frames} "
        f"layout={args.layout} "
        f"dispatcherThreads={args.dispatcher_threads} "
        f"affinityMask=0x{affinity_mask:X} "
        f"avbdIterationPolicy={AVBD_ITERATION_POLICY} "
        f"avbdIterationSource={AVBD_ITERATION_SOURCE} "
        f"avbdIterations={AVBD_ITERATIONS} "
        f"avbdIterationSemantics={AVBD_ITERATION_SEMANTICS} "
        "avbdJointIterationOverrideSource="
        f"{AVBD_JOINT_ITERATION_OVERRIDE_SOURCE} "
        f"avbdJointIterationOverride={AVBD_JOINT_ITERATION_OVERRIDE} "
        "avbdJointIterationOverrideActive=0 "
        f"avbdEarlyStopSource={AVBD_EARLY_STOP_SOURCE} "
        f"avbdEarlyStopEnabled={AVBD_EARLY_STOP_ENABLED} "
        f"avbdEarlyStopActive={AVBD_EARLY_STOP_ACTIVE} "
        f"tgsAvgMs({format_distribution(timings['tgs']['avgStepMs'])}) "
        f"avbdAvgMs({format_distribution(timings['avbd']['avgStepMs'])}) "
        f"avbdOverTgs={avbd_over_tgs:.9g} "
        f"avbdGainPercent={avbd_gain_percent:.9g} "
        "comparisonBasis=not-work-normalized "
        f"workTelemetry={work_attribution['status']} status=PASS"
    )
    for metric in TIMING_FIELDS[1:]:
        print(
            "[AVBD_RIGID_BASELINE_DISTRIBUTION] "
            f"metric={metric} "
            f"tgs({format_distribution(timings['tgs'][metric])}) "
            f"avbd({format_distribution(timings['avbd'][metric])})"
        )

    if args.json_out:
        payload = {
            "schema": int(SCHEMA),
            "revision": revision,
            "headRevision": source["headRevision"],
            "worktreeDirty": source["worktreeDirty"],
            "worktreeStatusHash": source["worktreeStatusHash"],
            "worktreeDiffHash": source["worktreeDiffHash"],
            "untrackedPathHash": source["untrackedPathHash"],
            "untrackedFileCount": source["untrackedFileCount"],
            "worktreeIdentityHash": source["worktreeIdentityHash"],
            "worktreeIdentityScope": source["worktreeIdentityScope"],
            "contractHash": measurement_hash,
            "contractFileCount": len(contract_files),
            "contractFiles": contract_files,
            "buildHash": build_hash,
            "buildArtifacts": build_artifacts,
            "host": platform.node() or "unknown",
            "platform": platform.platform(),
            "case": CASE,
            "layout": args.layout,
            "pairs": args.pairs,
            "frames": args.frames,
            "warmupFrames": args.warmup_frames,
            "profileFrames": args.frames - args.warmup_frames,
            "dispatcherThreads": args.dispatcher_threads,
            "affinityMask": f"0x{affinity_mask:X}",
            "iterationContract": {
                "actorPositionIterations": 4,
                "actorVelocityIterations": 1,
                "avbdIterationPolicy": AVBD_ITERATION_POLICY,
                "avbdIterationSource": AVBD_ITERATION_SOURCE,
                "avbdIterations": AVBD_ITERATIONS,
                "avbdIterationSemantics": AVBD_ITERATION_SEMANTICS,
                "avbdJointIterationOverrideSource": (
                    AVBD_JOINT_ITERATION_OVERRIDE_SOURCE
                ),
                "avbdJointIterationOverride": (
                    AVBD_JOINT_ITERATION_OVERRIDE
                ),
                "avbdJointIterationOverrideActive": False,
                "avbdEarlyStopSource": AVBD_EARLY_STOP_SOURCE,
                "avbdEarlyStopEnabled": bool(AVBD_EARLY_STOP_ENABLED),
                "avbdEarlyStopActiveForFixture": bool(
                    AVBD_EARLY_STOP_ACTIVE
                ),
                "avbdIterationActiveSolver": "avbd",
            },
            "samples": samples,
            "gateSamples": gate_samples,
            "workSamples": work_samples,
            "correctnessFingerprints": correctness_fingerprints,
            "stateDigests": state_digests,
            "workAttribution": work_attribution,
            "comparisonBasis": "not-work-normalized",
            "distributions": {
                solver: {
                    metric: distribution(values)
                    for metric, values in solver_timings.items()
                }
                for solver, solver_timings in timings.items()
            },
            "avbdOverTgs": avbd_over_tgs,
            "avbdGainPercent": avbd_gain_percent,
            "status": "PASS",
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"AVBD_RIGID_STRESS_BASELINE=FAIL error={error}")
        raise SystemExit(1)
