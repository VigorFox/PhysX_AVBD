#!/usr/bin/env python3
"""Compare AVBD's CPU fast-colored and ordered backends on one large island.

The runner executes the connected ``SnippetHelloWorld`` rigid-stress fixture
in interleaved pairs.  ``--enhanced-determinism=off`` selects the CPU fast
policy; ``on`` selects the ordered authority.  Both sides use the same AVBD
iteration defaults and Release executable.

State digests are retained as diagnostic evidence, but are deliberately not a
cross-run gate: the fast backend is non-deterministic by contract, and this
performance runner does not promote the ordered backend's digest to a formal
determinism authority either.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
from typing import Any

from snippet_headless_process import run_headless_process


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = (
    ROOT
    / "physx/bin/win.x86_64.vc143.md/release/SnippetHelloWorld_64.exe"
)
SCHEMA = 1
SNIPPET_SCHEMA = "4"
CASE = "rigid-stress"
LAYOUT = "connected"
DEFAULT_WORKERS = (4, 14)
TIMING_FIELDS = ("avgStepMs", "p50StepMs", "p95StepMs", "maxStepMs")
STATE_DIGEST_PATTERN = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True)
class Backend:
    name: str
    enhanced_determinism: str
    requested: str
    policy: str


BACKENDS = {
    "fast": Backend("fast", "off", "0", "fast"),
    "ordered": Backend("ordered", "on", "1", "ordered"),
}


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


def require_fields(
    fields: dict[str, str], expected: dict[str, str], label: str
) -> None:
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(
                f"{label} {key}={fields.get(key)!r}, expected {value!r}"
            )


def parse_workers(values: list[str] | None) -> tuple[int, ...]:
    workers: list[int] = []
    for value in values or [",".join(str(item) for item in DEFAULT_WORKERS)]:
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
    if not workers:
        raise RuntimeError("at least one worker count is required")
    return tuple(workers)


def validate_result(
    gate: dict[str, str],
    perf: dict[str, str],
    backend: Backend,
    worker: int,
    frames: int,
    warmup_frames: int,
) -> None:
    route = {
        "layout": LAYOUT,
        "enhancedDeterminismRequested": backend.requested,
        "enhancedDeterminismObserved": backend.requested,
        "avbdBackendPolicy": backend.policy,
    }
    require_fields(
        gate,
        {
            "schema": SNIPPET_SCHEMA,
            "snippet": "SnippetHelloWorld",
            "case": CASE,
            "solver": "avbd",
            "execution": "parallel",
            "requestedFrames": str(frames),
            "completedFrames": str(frames),
            "dispatcherThreads": str(worker),
            "status": "PASS",
            "reason": "none",
            "physicsErrors": "0",
            "physicsWarnings": "0",
            "fetchFailures": "0",
            "fetchErrorState": "0",
            "boxCount": "8400",
            "projectileCount": "1",
            "finiteBoxes": "8400",
            "finalBallFinite": "1",
            "stateDigestAlgorithm": "fnv1a64x2-v1",
            "stateDigestActorCount": "8401",
            "avbdIterationPolicy": "scene-desc",
            "avbdIterationSource": "default",
            "avbdIterations": "4",
            "avbdIterationActive": "1",
            "avbdJointIterationOverrideSource": "default",
            "avbdJointIterationOverride": "8",
            "avbdJointIterationOverrideActive": "0",
            "avbdEarlyStopSource": "default",
            "avbdEarlyStopEnabled": "1",
            "avbdEarlyStopActive": "0",
            **route,
        },
        f"worker={worker} backend={backend.name} gate",
    )
    if integer(gate, "movedBoxes") <= 0:
        raise RuntimeError(
            f"worker={worker} backend={backend.name} moved no boxes"
        )
    state_digest = gate.get("stateDigest", "")
    if not STATE_DIGEST_PATTERN.fullmatch(state_digest):
        raise RuntimeError(
            f"worker={worker} backend={backend.name} "
            f"invalid stateDigest={state_digest!r}"
        )

    require_fields(
        perf,
        {
            "schema": SNIPPET_SCHEMA,
            "snippet": "SnippetHelloWorld",
            "case": CASE,
            "buildProfile": "release",
            "solver": "avbd",
            "sceneExecution": "parallel",
            "dispatcherThreads": str(worker),
            "cpuOnly": "1",
            "gpuDynamics": "0",
            "broadphase": "cpu",
            "pvd": "0",
            "rigidBoxes": "8400",
            "projectileCount": "1",
            "actorPositionIterations": "4",
            "actorVelocityIterations": "1",
            "avbdIterationPolicy": "scene-desc",
            "avbdIterationSource": "default",
            "avbdIterations": "4",
            "avbdIterationActive": "1",
            "avbdJointIterationOverrideSource": "default",
            "avbdJointIterationOverride": "8",
            "avbdJointIterationOverrideActive": "0",
            "avbdEarlyStopSource": "default",
            "avbdEarlyStopEnabled": "1",
            "avbdEarlyStopActive": "0",
            "warmupFrames": str(warmup_frames),
            "profileFrames": str(frames - warmup_frames),
            "measurement": "simulate-fetch",
            "instrumentation": "none",
            "status": "PASS",
            **route,
        },
        f"worker={worker} backend={backend.name} perf",
    )
    for key in TIMING_FIELDS:
        if number(perf, key) <= 0.0:
            raise RuntimeError(
                f"worker={worker} backend={backend.name} {key} "
                "must be positive"
            )


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith(
            ("PHYSX_AVBD_", "PX_AVBD_", "AVBD_", "PHYSX_SNIPPET_")
        ):
            env.pop(name, None)
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    return env


def run_once(
    executable: Path,
    backend: Backend,
    worker: int,
    frames: int,
    warmup_frames: int,
    timeout: float,
    affinity_mask: int | None,
) -> tuple[dict[str, str], dict[str, str], int | None]:
    command = [
        str(executable),
        "--headless",
        "--solver=avbd",
        f"--case={CASE}",
        f"--rigid-stress-layout={LAYOUT}",
        f"--enhanced-determinism={backend.enhanced_determinism}",
        f"--frames={frames}",
        f"--warmup-frames={warmup_frames}",
        f"--dispatcher-threads={worker}",
        "--execution=parallel",
        "--dt=0.0166666675",
        "--seed=1",
    ]
    result = run_headless_process(
        command,
        cwd=executable.parent,
        env=clean_environment(),
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
            f"worker={worker} backend={backend.name} "
            f"exit={result.returncode} timeout={int(result.timed_out)} "
            f"visibleWindow={int(result.visible_window_detected)} "
            f"affinity={result.cpu_affinity_mask!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    gate = only_result(result.stdout, "[AVBD_GATE] ")
    perf = only_result(result.stdout, "[AVBD_RIGID_PERF] ")
    validate_result(gate, perf, backend, worker, frames, warmup_frames)
    return gate, perf, result.cpu_affinity_mask


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument(
        "--workers",
        action="append",
        help="comma-separated worker counts; default: 4,14",
    )
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--affinity-mask",
        type=lambda value: int(value, 0),
        help="optional Windows process affinity mask",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="optional full JSON artifact (a compact JSON summary is always printed)",
    )
    args = parser.parse_args()

    workers = parse_workers(args.workers)
    if args.frames < 2:
        raise RuntimeError("frames must be at least 2")
    if args.warmup_frames < 0 or args.warmup_frames >= args.frames:
        raise RuntimeError("warmup frames must be in [0, frames)")
    if args.runs < 1:
        raise RuntimeError("runs must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise RuntimeError("timeout must be finite and positive")
    if args.affinity_mask is not None and args.affinity_mask <= 0:
        raise RuntimeError("affinity mask must be positive")

    executable = args.exe.resolve()
    if not executable.is_file():
        raise RuntimeError(f"missing executable: {executable}")

    samples: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(worker): {name: [] for name in BACKENDS} for worker in workers
    }
    pairs: dict[str, list[dict[str, Any]]] = {
        str(worker): [] for worker in workers
    }
    for worker in workers:
        worker_key = str(worker)
        for run_index in range(args.runs):
            order = (
                ("fast", "ordered")
                if run_index % 2 == 0
                else ("ordered", "fast")
            )
            pair: dict[str, Any] = {
                "run": run_index + 1,
                "order": list(order),
            }
            for order_index, backend_name in enumerate(order, start=1):
                backend = BACKENDS[backend_name]
                gate, perf, effective_mask = run_once(
                    executable,
                    backend,
                    worker,
                    args.frames,
                    args.warmup_frames,
                    args.timeout,
                    args.affinity_mask,
                )
                sample = {
                    "run": run_index + 1,
                    "orderInPair": order_index,
                    "gate": gate,
                    "performance": perf,
                    "stateDigest": gate["stateDigest"],
                    "affinityMask": (
                        None
                        if effective_mask is None
                        else f"0x{effective_mask:X}"
                    ),
                }
                samples[worker_key][backend_name].append(sample)
                pair[backend_name] = {
                    key: number(perf, key) for key in TIMING_FIELDS
                }
                print(
                    "[AVBD_CPU_COLORED_MATRIX_RUN] "
                    f"worker={worker} run={run_index + 1} "
                    f"backend={backend_name} order={order_index} "
                    f"avgStepMs={number(perf, 'avgStepMs'):.9g} "
                    f"p95StepMs={number(perf, 'p95StepMs'):.9g} "
                    f"stateDigest={gate['stateDigest']} status=PASS",
                    flush=True,
                )
            pair["orderedOverFastAvg"] = (
                pair["ordered"]["avgStepMs"] / pair["fast"]["avgStepMs"]
            )
            pair["orderedOverFastP95"] = (
                pair["ordered"]["p95StepMs"] / pair["fast"]["p95StepMs"]
            )
            pairs[worker_key].append(pair)

    summaries: dict[str, Any] = {}
    for worker in workers:
        worker_key = str(worker)
        backend_summaries: dict[str, Any] = {}
        for backend_name in BACKENDS:
            backend_samples = samples[worker_key][backend_name]
            digests = [item["stateDigest"] for item in backend_samples]
            backend_summaries[backend_name] = {
                "timings": {
                    key: distribution(
                        [
                            number(item["performance"], key)
                            for item in backend_samples
                        ]
                    )
                    for key in TIMING_FIELDS
                },
                "stateDigests": digests,
                "stateDigestUniqueCount": len(set(digests)),
                "stateDigestStable": len(set(digests)) == 1,
                "stateDigestPolicy": "record-only",
            }
        pair_avg_ratios = [
            item["orderedOverFastAvg"] for item in pairs[worker_key]
        ]
        pair_p95_ratios = [
            item["orderedOverFastP95"] for item in pairs[worker_key]
        ]
        backend_summaries["pairedComparison"] = {
            "orderedOverFastAvg": distribution(pair_avg_ratios),
            "orderedOverFastP95": distribution(pair_p95_ratios),
            "interpretation": (
                "values above 1 mean the fast-colored backend is faster"
            ),
        }
        summaries[worker_key] = backend_summaries
        print(
            "[AVBD_CPU_COLORED_MATRIX_WORKER] "
            f"worker={worker} runs={args.runs} "
            "fastMedianMs="
            f"{backend_summaries['fast']['timings']['avgStepMs']['median']:.9g} "
            "orderedMedianMs="
            f"{backend_summaries['ordered']['timings']['avgStepMs']['median']:.9g} "
            "pairedOrderedOverFastMedian="
            f"{backend_summaries['pairedComparison']['orderedOverFastAvg']['median']:.9g} "
            "fastDigestUnique="
            f"{backend_summaries['fast']['stateDigestUniqueCount']} "
            "orderedDigestUnique="
            f"{backend_summaries['ordered']['stateDigestUniqueCount']} "
            "digestPolicy=record-only status=PASS",
            flush=True,
        )

    reference_worker = str(workers[0])
    scaling: dict[str, dict[str, float | int]] = {}
    for backend_name in BACKENDS:
        reference_median = summaries[reference_worker][backend_name][
            "timings"
        ]["avgStepMs"]["median"]
        scaling[backend_name] = {
            str(worker): reference_median
            / summaries[str(worker)][backend_name]["timings"]["avgStepMs"][
                "median"
            ]
            for worker in workers
        }

    payload = {
        "schema": SCHEMA,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node() or "unknown",
        "platform": platform.platform(),
        "executable": str(executable),
        "executableSha256": sha256_file(executable),
        "fixture": {
            "snippet": "SnippetHelloWorld",
            "case": CASE,
            "layout": LAYOUT,
            "solver": "avbd",
            "cpuOnly": True,
            "gpuDynamics": False,
            "frames": args.frames,
            "warmupFrames": args.warmup_frames,
            "profileFrames": args.frames - args.warmup_frames,
            "runs": args.runs,
            "workers": list(workers),
            "affinityMask": (
                None
                if args.affinity_mask is None
                else f"0x{args.affinity_mask:X}"
            ),
        },
        "backendContract": {
            name: {
                "enhancedDeterminism": backend.enhanced_determinism,
                "enhancedDeterminismRequested": int(backend.requested),
                "enhancedDeterminismObserved": int(backend.requested),
                "avbdBackendPolicy": backend.policy,
            }
            for name, backend in BACKENDS.items()
        },
        "digestPolicy": {
            "fast": "record-only; non-deterministic by contract",
            "ordered": "record-only; not a hard gate in this performance tool",
        },
        "samples": samples,
        "pairs": pairs,
        "summaries": summaries,
        "workerScaling": {
            "referenceWorker": int(reference_worker),
            "medianAvgStepSpeedup": scaling,
        },
        "status": "PASS",
    }
    if args.json_out is not None:
        json_out = args.json_out.resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        json_path = str(json_out)
    else:
        json_path = "none"

    compact_summary = {
        "schema": SCHEMA,
        "workers": list(workers),
        "runs": args.runs,
        "summaries": summaries,
        "workerScaling": payload["workerScaling"],
        "jsonOut": json_path,
        "status": "PASS",
    }
    print(
        "[AVBD_CPU_COLORED_MATRIX_JSON] "
        + json.dumps(compact_summary, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    print(
        "[AVBD_CPU_COLORED_MATRIX_SUMMARY] "
        f"workers={','.join(str(worker) for worker in workers)} "
        f"runs={args.runs} referenceWorker={reference_worker} "
        f"jsonOut={json_path} status=PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"AVBD_CPU_COLORED_BACKEND_MATRIX=FAIL error={error}")
        raise SystemExit(1)
