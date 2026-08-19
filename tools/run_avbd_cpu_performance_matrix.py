#!/usr/bin/env python3
"""Run the formal CPU-only rigid-stress authority across worker counts.

This is deliberately an orchestration layer.  It does not duplicate the
rigid-stress fixture or parse the Snippet directly: every measured row is
produced by ``run_avbd_rigid_stress_baseline.py``.  The matrix then checks the
authority artifact again for CPU-only provenance and reports solver gaps,
real-time budget multiples, and worker scaling.

Fixtures that do not yet have admissible performance telemetry are listed as
unsupported (or correctness-only) instead of being silently substituted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX_RUNNER = Path(__file__).resolve()
AUTHORITY_RUNNER = ROOT / "tools" / "run_avbd_rigid_stress_baseline.py"
DEFAULT_BIN_DIR = ROOT / "physx/bin/win.x86_64.vc143.md/release"
REQUIRED_WORKERS = (4, 14)
FORMAL_MIN_PAIRS = 7
AUTHORITY_SCHEMA = 4
MATRIX_SCHEMA = 3
AVBD_ITERATION_POLICY = "scene-desc"
AVBD_ITERATION_SOURCE = "default"
AVBD_ITERATIONS = 4
AVBD_ITERATION_SEMANTICS = "budgeted-complete-primal-dual-stiffness"
AVBD_JOINT_ITERATION_OVERRIDE_SOURCE = "default"
AVBD_JOINT_ITERATION_OVERRIDE = 8
AVBD_EARLY_STOP_SOURCE = "default"
AVBD_EARLY_STOP_ENABLED = True
AVBD_EARLY_STOP_ACTIVE = False
CPU_FIXTURE_FIELDS = {
    "schema": str(AUTHORITY_SCHEMA),
    "case": "rigid-stress",
    "cpuOnly": "1",
    "gpuDynamics": "0",
    "broadphase": "cpu",
    "pvd": "0",
    "sceneExecution": "parallel",
    "buildProfile": "release",
    "instrumentation": "none",
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
    "avbdEarlyStopEnabled": "1",
    "avbdEarlyStopActive": "0",
    "status": "PASS",
}
TIMING_METRICS = ("avgStepMs", "p95StepMs")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
STATE_DIGEST_PATTERN = re.compile(r"[0-9a-f]{32}")
IDENTITY_FIELDS = (
    "revision",
    "headRevision",
    "contractHash",
    "buildHash",
    "worktreeDirty",
    "worktreeStatusHash",
    "worktreeDiffHash",
    "untrackedPathHash",
    "untrackedFileCount",
    "worktreeIdentityHash",
    "worktreeIdentityScope",
)


@dataclass(frozen=True)
class FixtureAssessment:
    fixture: str
    performance_status: str
    next_gate: str
    evidence: tuple[str, ...]
    reason: str


FIXTURE_ASSESSMENTS = (
    FixtureAssessment(
        fixture="rigid-stress-8400",
        performance_status="SUPPORTED_AUTHORITY",
        next_gate="none",
        evidence=(
            "tools/run_avbd_rigid_stress_baseline.py",
            "physx/snippets/snippethelloworld/SnippetHelloWorld.cpp",
        ),
        reason=(
            "CPU broadphase, no GPU dynamics, interleaved TGS/AVBD timing "
            "and 4/14-worker coverage are available."
        ),
    ),
    FixtureAssessment(
        fixture="rigid-stress-connected-8400",
        performance_status="SUPPORTED_STANDALONE_AUTHORITY",
        next_gate="add_strict_single_island_quality_metrics",
        evidence=(
            "tools/run_avbd_rigid_stress_baseline.py",
            "tools/run_avbd_cpu_colored_backend_matrix.py",
            "physx/snippets/snippethelloworld/SnippetHelloWorld.cpp",
        ),
        reason=(
            "The explicit connected layout now has CPU-only 4/14-worker "
            "interleaved TGS/AVBD timing, fast/ordered timing, state "
            "digests, and colored/dual work attribution.  It contains one "
            "large authority island plus smaller islands, so a strict "
            "single-island quality fixture remains follow-up work."
        ),
    ),
    FixtureAssessment(
        fixture="joint-mixed-chainmail-900",
        performance_status="CORRECTNESS_ONLY",
        next_gate="add_simulate_fetch_timing_and_configurable_4_14_workers",
        evidence=(
            "tools/run_snippet_chainmail_headless.py",
            "physx/snippets/snippetchainmail/SnippetChainmail.cpp",
        ),
        reason=(
            "The 900-body/1740-joint impact fixture covers a large mixed "
            "joint/contact island, but its runner emits correctness gates, "
            "has a fixed four-thread command, and has no formal timing line."
        ),
    ),
    FixtureAssessment(
        fixture="rigid-stress-deterministic",
        performance_status="CORRECTNESS_AND_EXPLORATORY_PERFORMANCE",
        next_gate="add_formal_repeated_ordered_matrix_and_scene_coverage",
        evidence=(
            "tools/run_avbd_cpu_colored_backend_matrix.py",
            "physx/source/lowleveldynamics/src/DyAvbdTasks.cpp",
        ),
        reason=(
            "The scene flag now selects the ordered backend, the connected "
            "runner times that policy, and the independent 90-frame digest "
            "matches at 4 and 14 workers.  Broader scenes and a dedicated "
            "formal repeated determinism authority are still required."
        ),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not a number: {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive")
    return parsed


def parse_workers(values: list[str]) -> tuple[int, ...]:
    workers: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                worker = int(token)
            except ValueError as error:
                raise RuntimeError(
                    f"invalid worker count: {token!r}"
                ) from error
            if worker < 1:
                raise RuntimeError("worker counts must be positive")
            if worker not in workers:
                workers.append(worker)
    missing = [worker for worker in REQUIRED_WORKERS if worker not in workers]
    if missing:
        raise RuntimeError(
            "CPU matrix must include required workers: "
            + ",".join(str(worker) for worker in missing)
        )
    return tuple(workers)


def authority_command(
    bin_dir: Path,
    artifact: Path,
    worker: int,
    pairs: int,
    frames: int,
    warmup_frames: int,
    timeout: float,
    affinity_mask: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        str(AUTHORITY_RUNNER),
        "--bin-dir",
        str(bin_dir),
        "--pairs",
        str(pairs),
        "--frames",
        str(frames),
        "--warmup-frames",
        str(warmup_frames),
        "--dispatcher-threads",
        str(worker),
        "--timeout",
        str(timeout),
        "--json-out",
        str(artifact),
    ]
    if affinity_mask is not None:
        command.extend(("--affinity-mask", hex(affinity_mask)))
    return command


def validate_authority_payload(
    payload: dict[str, Any], worker: int, pairs: int
) -> None:
    expected_top_level = {
        "schema": AUTHORITY_SCHEMA,
        "case": "rigid-stress",
        "dispatcherThreads": worker,
        "pairs": pairs,
        "status": "PASS",
    }
    for key, expected in expected_top_level.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"worker={worker} authority {key}={payload.get(key)!r}, "
                f"expected {expected!r}"
            )

    samples = payload.get("samples")
    if not isinstance(samples, dict):
        raise RuntimeError(f"worker={worker} authority samples are missing")
    for solver in ("tgs", "avbd"):
        solver_samples = samples.get(solver)
        if not isinstance(solver_samples, list) or len(solver_samples) != pairs:
            raise RuntimeError(
                f"worker={worker} solver={solver} expected {pairs} samples"
            )
        for sample_index, sample in enumerate(solver_samples, start=1):
            if not isinstance(sample, dict):
                raise RuntimeError(
                    f"worker={worker} solver={solver} sample={sample_index} "
                    "is malformed"
                )
            expected_fields = dict(CPU_FIXTURE_FIELDS)
            expected_fields.update(
                {
                    "solver": solver,
                    "dispatcherThreads": str(worker),
                    "avbdIterationActive": (
                        "1" if solver == "avbd" else "0"
                    ),
                    "avbdEarlyStopActive": "0",
                }
            )
            for key, expected in expected_fields.items():
                if sample.get(key) != expected:
                    raise RuntimeError(
                        f"GPU/non-authority result rejected: worker={worker} "
                        f"solver={solver} sample={sample_index} "
                        f"{key}={sample.get(key)!r}, expected {expected!r}"
                    )
            for metric in TIMING_METRICS:
                finite_number(
                    sample.get(metric),
                    f"worker={worker} solver={solver} {metric}",
                )

    for identity_key in ("revision", "headRevision"):
        identity = payload.get(identity_key)
        if not isinstance(identity, str) or not identity:
            raise RuntimeError(
                f"worker={worker} authority {identity_key} is missing"
            )
    for identity_key in (
        "contractHash",
        "buildHash",
        "worktreeStatusHash",
        "worktreeDiffHash",
        "untrackedPathHash",
        "worktreeIdentityHash",
    ):
        identity = payload.get(identity_key)
        if not isinstance(identity, str) or not SHA256_PATTERN.fullmatch(
            identity
        ):
            raise RuntimeError(
                f"worker={worker} authority {identity_key} is not a SHA-256"
            )
    if not isinstance(payload.get("worktreeDirty"), bool):
        raise RuntimeError(
            f"worker={worker} authority worktreeDirty is not boolean"
        )
    untracked_count = payload.get("untrackedFileCount")
    if (
        not isinstance(untracked_count, int)
        or isinstance(untracked_count, bool)
        or untracked_count < 0
    ):
        raise RuntimeError(
            f"worker={worker} authority untrackedFileCount is invalid"
        )
    if (
        payload.get("worktreeIdentityScope")
        != "tracked-diff+untracked-paths"
    ):
        raise RuntimeError(
            f"worker={worker} authority worktreeIdentityScope is invalid"
        )
    if payload.get("comparisonBasis") != "not-work-normalized":
        raise RuntimeError(
            f"worker={worker} authority comparison basis is not explicit"
        )
    expected_iteration_contract = {
        "actorPositionIterations": 4,
        "actorVelocityIterations": 1,
        "avbdIterationPolicy": AVBD_ITERATION_POLICY,
        "avbdIterationSource": AVBD_ITERATION_SOURCE,
        "avbdIterations": AVBD_ITERATIONS,
        "avbdIterationSemantics": AVBD_ITERATION_SEMANTICS,
        "avbdJointIterationOverrideSource": (
            AVBD_JOINT_ITERATION_OVERRIDE_SOURCE
        ),
        "avbdJointIterationOverride": AVBD_JOINT_ITERATION_OVERRIDE,
        "avbdJointIterationOverrideActive": False,
        "avbdEarlyStopSource": AVBD_EARLY_STOP_SOURCE,
        "avbdEarlyStopEnabled": AVBD_EARLY_STOP_ENABLED,
        "avbdEarlyStopActiveForFixture": AVBD_EARLY_STOP_ACTIVE,
        "avbdIterationActiveSolver": "avbd",
    }
    if payload.get("iterationContract") != expected_iteration_contract:
        raise RuntimeError(
            f"worker={worker} authority iteration contract is invalid: "
            f"{payload.get('iterationContract')!r}"
        )
    work_attribution = payload.get("workAttribution")
    if not isinstance(work_attribution, dict) or (
        work_attribution.get("status") != "UNAVAILABLE"
        or work_attribution.get("reason") != "not-requested"
    ):
        raise RuntimeError(
            f"worker={worker} formal timing unexpectedly contains work "
            "attribution"
        )

    work_samples = payload.get("workSamples")
    if not isinstance(work_samples, dict):
        raise RuntimeError(
            f"worker={worker} authority work-policy evidence is missing"
        )
    for solver in ("tgs", "avbd"):
        solver_work = work_samples.get(solver)
        if not isinstance(solver_work, list) or len(solver_work) != pairs:
            raise RuntimeError(
                f"worker={worker} solver={solver} work samples are incomplete"
            )
        for pair_index, work in enumerate(solver_work, start=1):
            expected_work = {
                "schema": str(AUTHORITY_SCHEMA),
                "solver": solver,
                "workTelemetry": "UNAVAILABLE",
                "attributionMode": "none",
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
                "avbdJointIterationOverride": str(
                    AVBD_JOINT_ITERATION_OVERRIDE
                ),
                "avbdJointIterationOverrideActive": "0",
                "avbdEarlyStopSource": AVBD_EARLY_STOP_SOURCE,
                "avbdEarlyStopEnabled": "1",
                "avbdEarlyStopActive": "0",
                "avbdLocalSolveCount": "UNAVAILABLE",
                "localSolveTelemetry": "UNAVAILABLE",
            }
            if not isinstance(work, dict):
                raise RuntimeError(
                    f"worker={worker} solver={solver} pair={pair_index} "
                    "work sample is malformed"
                )
            for key, expected in expected_work.items():
                if work.get(key) != expected:
                    raise RuntimeError(
                        f"worker={worker} solver={solver} pair={pair_index} "
                        f"work {key}={work.get(key)!r}, expected {expected!r}"
                    )

    gate_samples = payload.get("gateSamples")
    state_digests = payload.get("stateDigests")
    correctness_fingerprints = payload.get("correctnessFingerprints")
    if (
        not isinstance(gate_samples, dict)
        or not isinstance(state_digests, dict)
        or not isinstance(correctness_fingerprints, dict)
    ):
        raise RuntimeError(
            f"worker={worker} authority state-digest evidence is missing"
        )
    # Stability is a within-solver repeatability condition.  TGS and AVBD
    # solve different trajectories, so their digests are intentionally not
    # compared with each other.
    for solver in ("tgs", "avbd"):
        solver_gates = gate_samples.get(solver)
        solver_digests = state_digests.get(solver)
        solver_fingerprints = correctness_fingerprints.get(solver)
        if not isinstance(solver_gates, list) or len(solver_gates) != pairs:
            raise RuntimeError(
                f"worker={worker} solver={solver} gate samples are incomplete"
            )
        if not isinstance(solver_digests, list) or len(solver_digests) != pairs:
            raise RuntimeError(
                f"worker={worker} solver={solver} state digests are incomplete"
            )
        if not isinstance(solver_fingerprints, list) or len(
            solver_fingerprints
        ) != pairs:
            raise RuntimeError(
                f"worker={worker} solver={solver} correctness fingerprints "
                "are incomplete"
            )
        gate_digests: list[str] = []
        for pair_index, gate in enumerate(solver_gates, start=1):
            if not isinstance(gate, dict):
                raise RuntimeError(
                    f"worker={worker} solver={solver} pair={pair_index} "
                    "gate sample is malformed"
                )
            if gate.get("solver") != solver:
                raise RuntimeError(
                    f"worker={worker} solver={solver} pair={pair_index} "
                    "gate solver identity is invalid"
                )
            expected_gate_iteration = {
                "schema": str(AUTHORITY_SCHEMA),
                "avbdIterationPolicy": AVBD_ITERATION_POLICY,
                "avbdIterationSource": AVBD_ITERATION_SOURCE,
                "avbdIterations": str(AVBD_ITERATIONS),
                "avbdIterationActive": "1" if solver == "avbd" else "0",
                "avbdIterationSemantics": AVBD_ITERATION_SEMANTICS,
                "avbdJointIterationOverrideSource": (
                    AVBD_JOINT_ITERATION_OVERRIDE_SOURCE
                ),
                "avbdJointIterationOverride": str(
                    AVBD_JOINT_ITERATION_OVERRIDE
                ),
                "avbdJointIterationOverrideActive": "0",
                "avbdEarlyStopSource": AVBD_EARLY_STOP_SOURCE,
                "avbdEarlyStopEnabled": "1",
                "avbdEarlyStopActive": "0",
            }
            for key, expected in expected_gate_iteration.items():
                if gate.get(key) != expected:
                    raise RuntimeError(
                        f"worker={worker} solver={solver} pair={pair_index} "
                        f"gate {key}={gate.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            digest = gate.get("stateDigest")
            if not isinstance(digest, str) or not STATE_DIGEST_PATTERN.fullmatch(
                digest
            ):
                raise RuntimeError(
                    f"worker={worker} solver={solver} pair={pair_index} "
                    f"stateDigest={digest!r} is invalid"
                )
            gate_digests.append(digest)
        if solver_digests != gate_digests:
            raise RuntimeError(
                f"worker={worker} solver={solver} stateDigests do not match "
                "gate samples"
            )
        if len(set(gate_digests)) != 1:
            raise RuntimeError(
                f"worker={worker} solver={solver} stateDigest changed "
                "across pairs"
            )
        if any(
            not isinstance(fingerprint, str)
            or not SHA256_PATTERN.fullmatch(fingerprint)
            for fingerprint in solver_fingerprints
        ):
            raise RuntimeError(
                f"worker={worker} solver={solver} correctness fingerprint "
                "is invalid"
            )
        if len(set(solver_fingerprints)) != 1:
            raise RuntimeError(
                f"worker={worker} solver={solver} correctness fingerprint "
                "changed across pairs"
            )


def metric_median(
    payload: dict[str, Any], solver: str, metric: str
) -> float:
    try:
        value = payload["distributions"][solver][metric]["median"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"authority distribution is missing {solver}/{metric}/median"
        ) from error
    return finite_number(value, f"{solver}/{metric}/median")


def summarize_row(
    payload: dict[str, Any], worker: int, frame_budget_ms: float
) -> dict[str, Any]:
    tgs_avg = metric_median(payload, "tgs", "avgStepMs")
    tgs_p95 = metric_median(payload, "tgs", "p95StepMs")
    avbd_avg = metric_median(payload, "avbd", "avgStepMs")
    avbd_p95 = metric_median(payload, "avbd", "p95StepMs")
    tgs_work = payload["workSamples"]["tgs"][0]
    avbd_work = payload["workSamples"]["avbd"][0]
    return {
        "workers": worker,
        "tgsAvgMs": tgs_avg,
        "tgsP95Ms": tgs_p95,
        "avbdAvgMs": avbd_avg,
        "avbdP95Ms": avbd_p95,
        "avbdOverTgsAvg": avbd_avg / tgs_avg,
        "avbdOverTgsP95": avbd_p95 / tgs_p95,
        "tgsAvgBudgetMultiple": tgs_avg / frame_budget_ms,
        "tgsP95BudgetMultiple": tgs_p95 / frame_budget_ms,
        "avbdAvgBudgetMultiple": avbd_avg / frame_budget_ms,
        "avbdP95BudgetMultiple": avbd_p95 / frame_budget_ms,
        "tgsAvgRealtime": tgs_avg <= frame_budget_ms,
        "tgsP95Realtime": tgs_p95 <= frame_budget_ms,
        "avbdAvgRealtime": avbd_avg <= frame_budget_ms,
        "avbdP95Realtime": avbd_p95 <= frame_budget_ms,
        "comparisonBasis": payload["comparisonBasis"],
        "actorPositionIterations": int(
            avbd_work["actorPositionIterations"]
        ),
        "actorVelocityIterations": int(
            avbd_work["actorVelocityIterations"]
        ),
        "avbdIterationPolicy": avbd_work["avbdIterationPolicy"],
        "avbdIterationSource": avbd_work["avbdIterationSource"],
        "avbdIterations": int(avbd_work["avbdIterations"]),
        "avbdIterationSemantics": avbd_work[
            "avbdIterationSemantics"
        ],
        "avbdJointIterationOverrideSource": avbd_work[
            "avbdJointIterationOverrideSource"
        ],
        "avbdJointIterationOverride": int(
            avbd_work["avbdJointIterationOverride"]
        ),
        "avbdJointIterationOverrideActive": bool(
            int(avbd_work["avbdJointIterationOverrideActive"])
        ),
        "avbdEarlyStopSource": avbd_work["avbdEarlyStopSource"],
        "avbdEarlyStopEnabled": bool(
            int(avbd_work["avbdEarlyStopEnabled"])
        ),
        "tgsAvbdEarlyStopActive": bool(
            int(tgs_work["avbdEarlyStopActive"])
        ),
        "avbdEarlyStopActive": bool(
            int(avbd_work["avbdEarlyStopActive"])
        ),
        "tgsAvbdIterationActive": bool(
            int(tgs_work["avbdIterationActive"])
        ),
        "avbdIterationActive": bool(
            int(avbd_work["avbdIterationActive"])
        ),
        "localSolveTelemetry": avbd_work["localSolveTelemetry"],
    }


def scaling_summary(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    low, high = REQUIRED_WORKERS
    low_row = rows[low]
    high_row = rows[high]
    worker_ratio = high / low
    tgs_speedup = low_row["tgsAvgMs"] / high_row["tgsAvgMs"]
    avbd_speedup = low_row["avbdAvgMs"] / high_row["avbdAvgMs"]
    return {
        "fromWorkers": low,
        "toWorkers": high,
        "workerRatio": worker_ratio,
        "tgsAvgSpeedup": tgs_speedup,
        "tgsParallelEfficiency": tgs_speedup / worker_ratio,
        "avbdAvgSpeedup": avbd_speedup,
        "avbdParallelEfficiency": avbd_speedup / worker_ratio,
    }


def consistent_authority_identity(
    payloads: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for key in IDENTITY_FIELDS:
        observed = {
            json.dumps(payload[key], sort_keys=True)
            for payload in payloads.values()
            if key in payload
        }
        present_count = sum(key in payload for payload in payloads.values())
        if present_count != len(payloads):
            raise RuntimeError(
                f"authority identity field {key} is missing from the worker "
                "matrix"
            )
        if len(observed) > 1:
            raise RuntimeError(
                f"authority identity field {key} changed while the worker "
                "matrix was running"
            )
        identity[key] = next(iter(payloads.values()))[key]
    return identity


def fixture_payload() -> list[dict[str, Any]]:
    return [
        {
            "fixture": assessment.fixture,
            "performanceStatus": assessment.performance_status,
            "nextGate": assessment.next_gate,
            "evidence": list(assessment.evidence),
            "reason": assessment.reason,
        }
        for assessment in FIXTURE_ASSESSMENTS
    ]


def print_fixture_assessments() -> None:
    for assessment in FIXTURE_ASSESSMENTS:
        print(
            "[AVBD_CPU_MATRIX_FIXTURE] "
            f"fixture={assessment.fixture} "
            f"performanceStatus={assessment.performance_status} "
            f"nextGate={assessment.next_gate}"
        )


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument(
        "--workers",
        nargs="+",
        default=[str(worker) for worker in REQUIRED_WORKERS],
        metavar="N",
        help="worker counts (space- or comma-separated); must include 4 and 14",
    )
    parser.add_argument("--pairs", type=int, default=FORMAL_MIN_PAIRS)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--affinity-mask",
        type=lambda value: int(value, 0),
        help="affinity forwarded to every authority invocation",
    )
    parser.add_argument("--target-hz", type=float, default=60.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and print the formal plan without running the binary",
    )
    args = parser.parse_args()

    workers = parse_workers(args.workers)
    if args.pairs < FORMAL_MIN_PAIRS:
        raise RuntimeError(
            f"formal CPU matrix requires at least {FORMAL_MIN_PAIRS} pairs"
        )
    if args.frames < 90:
        raise RuntimeError("formal CPU matrix requires at least 90 frames")
    if args.warmup_frames < 0 or args.frames - args.warmup_frames < 60:
        raise RuntimeError(
            "formal CPU matrix requires at least 60 profile frames"
        )
    if args.timeout <= 0.0:
        raise RuntimeError("timeout must be positive")
    if args.target_hz <= 0.0 or not math.isfinite(args.target_hz):
        raise RuntimeError("target Hz must be finite and positive")
    if args.affinity_mask is not None and args.affinity_mask <= 0:
        raise RuntimeError("affinity mask must be positive")
    if not AUTHORITY_RUNNER.is_file():
        raise RuntimeError(f"missing authority runner: {AUTHORITY_RUNNER}")

    bin_dir = args.bin_dir.resolve()
    frame_budget_ms = 1000.0 / args.target_hz
    visible_cpus = min(os.cpu_count() or 1, 64)
    affinity_cpu_count = (
        bin(args.affinity_mask).count("1")
        if args.affinity_mask is not None
        else visible_cpus
    )
    if affinity_cpu_count < max(workers):
        raise RuntimeError(
            f"worker matrix requests {max(workers)} workers but affinity "
            f"exposes only {affinity_cpu_count} CPUs"
        )
    authority_hash = sha256_file(AUTHORITY_RUNNER)
    matrix_hash = sha256_file(MATRIX_RUNNER)
    print_fixture_assessments()

    common_payload: dict[str, Any] = {
        "schema": MATRIX_SCHEMA,
        "requiredAuthoritySchema": AUTHORITY_SCHEMA,
        "matrixRunner": str(MATRIX_RUNNER.relative_to(ROOT)),
        "matrixRunnerSha256": matrix_hash,
        "authorityRunner": str(AUTHORITY_RUNNER.relative_to(ROOT)),
        "authorityRunnerSha256": authority_hash,
        "cpuOnly": True,
        "gpuDynamics": False,
        "correctnessPolicy": {
            "sameSolverPairsRequireStableDigest": True,
            "crossSolverDigestEqualityRequired": False,
            "crossWorkerDigestEqualityRequired": False,
        },
        "visibleLogicalCpus": visible_cpus,
        "affinityCpuCount": affinity_cpu_count,
        "workers": list(workers),
        "pairs": args.pairs,
        "frames": args.frames,
        "warmupFrames": args.warmup_frames,
        "profileFrames": args.frames - args.warmup_frames,
        "targetHz": args.target_hz,
        "frameBudgetMs": frame_budget_ms,
        "comparisonBasis": "not-work-normalized",
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
            "avbdJointIterationOverride": AVBD_JOINT_ITERATION_OVERRIDE,
            "avbdJointIterationOverrideActive": False,
            "avbdEarlyStopSource": AVBD_EARLY_STOP_SOURCE,
            "avbdEarlyStopEnabled": AVBD_EARLY_STOP_ENABLED,
            "avbdEarlyStopActiveForFixture": AVBD_EARLY_STOP_ACTIVE,
            "avbdIterationActiveSolver": "avbd",
        },
        "fixtures": fixture_payload(),
    }

    if args.plan_only:
        for worker in workers:
            print(
                "[AVBD_CPU_MATRIX_PLAN] "
                f"fixture=rigid-stress-8400 workers={worker} "
                f"pairs={args.pairs} authoritySchema={AUTHORITY_SCHEMA} "
                f"avbdIterationPolicy={AVBD_ITERATION_POLICY} "
                f"avbdIterationSource={AVBD_ITERATION_SOURCE} "
                f"avbdIterations={AVBD_ITERATIONS} "
                "avbdJointIterationOverrideSource="
                f"{AVBD_JOINT_ITERATION_OVERRIDE_SOURCE} "
                "avbdJointIterationOverride="
                f"{AVBD_JOINT_ITERATION_OVERRIDE} "
                f"avbdEarlyStopSource={AVBD_EARLY_STOP_SOURCE} "
                f"avbdEarlyStopEnabled={int(AVBD_EARLY_STOP_ENABLED)} "
                f"avbdEarlyStopActive={int(AVBD_EARLY_STOP_ACTIVE)} "
                "identity=required cpuOnly=1 gpuDynamics=0 status=READY"
            )
        common_payload.update({"rows": {}, "status": "PLAN"})
        write_json(args.json_out, common_payload)
        print("[AVBD_CPU_PERFORMANCE_MATRIX] status=PLAN")
        return 0

    executable = bin_dir / "SnippetHelloWorld_64.exe"
    if not executable.is_file():
        raise RuntimeError(f"missing executable: {executable}")
    executable_hash = sha256_file(executable)

    authority_payloads: dict[int, dict[str, Any]] = {}
    rows: dict[int, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="avbd_cpu_matrix_") as temp_dir:
        artifact_dir = Path(temp_dir)
        for worker in workers:
            artifact = artifact_dir / f"rigid_stress_workers{worker}.json"
            command = authority_command(
                bin_dir,
                artifact,
                worker,
                args.pairs,
                args.frames,
                args.warmup_frames,
                args.timeout,
                args.affinity_mask,
            )
            env = os.environ.copy()
            for name in (
                "PHYSX_SNIPPET_SOLVER",
                "PHYSX_SNIPPET_FRAME_COUNT",
                "PHYSX_AVBD_ITER_DIAG_SEQUENTIAL",
                "PHYSX_AVBD_TASKGRAPH_SERIAL",
                "AVBD_HELLOWORLD_TRACE",
            ):
                env.pop(name, None)
            print(
                "[AVBD_CPU_MATRIX_AUTHORITY] "
                f"workers={worker} cpuOnly=1 gpuDynamics=0 status=START",
                flush=True,
            )
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                sys.stdout.write(result.stdout)
            if result.stderr:
                sys.stderr.write(result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"authority runner failed for workers={worker}: "
                    f"exit={result.returncode}"
                )
            if not artifact.is_file():
                raise RuntimeError(
                    f"authority runner produced no artifact for workers={worker}"
                )
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"invalid authority artifact for workers={worker}"
                ) from error
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"authority artifact for workers={worker} is not an object"
                )
            validate_authority_payload(payload, worker, args.pairs)
            authority_payloads[worker] = payload
            rows[worker] = summarize_row(payload, worker, frame_budget_ms)
            row = rows[worker]
            print(
                "[AVBD_CPU_MATRIX_ROW] "
                f"workers={worker} "
                f"tgsAvgMs={row['tgsAvgMs']:.9g} "
                f"tgsP95Ms={row['tgsP95Ms']:.9g} "
                f"avbdAvgMs={row['avbdAvgMs']:.9g} "
                f"avbdP95Ms={row['avbdP95Ms']:.9g} "
                f"avbdOverTgsAvg={row['avbdOverTgsAvg']:.9g} "
                f"avbdOverTgsP95={row['avbdOverTgsP95']:.9g} "
                f"tgsAvgBudgetMultiple={row['tgsAvgBudgetMultiple']:.9g} "
                f"tgsP95BudgetMultiple={row['tgsP95BudgetMultiple']:.9g} "
                f"avbdAvgBudgetMultiple={row['avbdAvgBudgetMultiple']:.9g} "
                f"avbdP95BudgetMultiple={row['avbdP95BudgetMultiple']:.9g} "
                f"avgRealtime={'PASS' if row['avbdAvgRealtime'] else 'MISS'} "
                f"p95Realtime={'PASS' if row['avbdP95Realtime'] else 'MISS'} "
                f"avbdIterationPolicy={row['avbdIterationPolicy']} "
                f"avbdIterationSource={row['avbdIterationSource']} "
                f"avbdIterations={row['avbdIterations']} "
                "avbdIterationSemantics="
                f"{row['avbdIterationSemantics']} "
                "avbdJointIterationOverride="
                f"{row['avbdJointIterationOverride']} "
                "avbdJointIterationOverrideActive="
                f"{int(row['avbdJointIterationOverrideActive'])} "
                "avbdEarlyStopEnabled="
                f"{int(row['avbdEarlyStopEnabled'])} "
                "avbdEarlyStopActive="
                f"{int(row['avbdEarlyStopActive'])} "
                f"comparisonBasis={row['comparisonBasis']} "
                "cpuOnly=1 gpuDynamics=0 status=PASS"
            )

    identity = consistent_authority_identity(authority_payloads)

    scaling = scaling_summary(rows)
    print(
        "[AVBD_CPU_MATRIX_SCALING] "
        f"fromWorkers={scaling['fromWorkers']} "
        f"toWorkers={scaling['toWorkers']} "
        f"tgsAvgSpeedup={scaling['tgsAvgSpeedup']:.9g} "
        f"tgsParallelEfficiency={scaling['tgsParallelEfficiency']:.9g} "
        f"avbdAvgSpeedup={scaling['avbdAvgSpeedup']:.9g} "
        f"avbdParallelEfficiency={scaling['avbdParallelEfficiency']:.9g} "
        "status=PASS"
    )

    common_payload.update(
        {
            **identity,
            "executable": str(executable),
            "executableSha256": executable_hash,
            "rows": {str(worker): row for worker, row in rows.items()},
            "scaling": scaling,
            "authorityArtifacts": {
                str(worker): payload
                for worker, payload in authority_payloads.items()
            },
            "status": "PASS",
        }
    )
    write_json(args.json_out, common_payload)
    print(
        "[AVBD_CPU_PERFORMANCE_MATRIX] "
        f"workers={','.join(str(worker) for worker in workers)} "
        f"pairs={args.pairs} targetHz={args.target_hz:.9g} "
        f"avbdIterationPolicy={AVBD_ITERATION_POLICY} "
        f"avbdIterationSource={AVBD_ITERATION_SOURCE} "
        f"avbdIterations={AVBD_ITERATIONS} "
        "avbdJointIterationOverrideSource="
        f"{AVBD_JOINT_ITERATION_OVERRIDE_SOURCE} "
        f"avbdJointIterationOverride={AVBD_JOINT_ITERATION_OVERRIDE} "
        f"avbdEarlyStopSource={AVBD_EARLY_STOP_SOURCE} "
        f"avbdEarlyStopEnabled={int(AVBD_EARLY_STOP_ENABLED)} "
        f"avbdEarlyStopActive={int(AVBD_EARLY_STOP_ACTIVE)} "
        "cpuOnly=1 gpuDynamics=0 status=PASS"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"AVBD_CPU_PERFORMANCE_MATRIX=FAIL error={error}")
        raise SystemExit(1)
