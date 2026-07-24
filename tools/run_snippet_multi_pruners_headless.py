#!/usr/bin/env python3
"""Run the SnippetMultiPruners streaming/query/solver gate headlessly."""

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
EXECUTABLE = "SnippetMultiPruners_64.exe"
FRAMES = 120


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    repeat: int


def specs(mode: str) -> tuple[RunSpec, ...]:
    lanes = (("tgs", "parallel"),)
    if mode != "authority":
        lanes = (
            ("tgs", "parallel"),
            ("avbd", "parallel"),
            ("avbd", "sequential"),
        )
    repeats = (1, 2) if mode == "acceptance" else (1,)
    return tuple(
        RunSpec(f"{solver}-{execution}-r{repeat}", solver, execution, repeat)
        for repeat in repeats
        for solver, execution in lanes
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


def parse_int(
    fields: dict[str, str], key: str, errors: list[str]
) -> int | None:
    try:
        return int(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key}={fields.get(key)!r}, expected integer")
        return None


def parse_float(
    fields: dict[str, str], key: str, errors: list[str]
) -> float | None:
    try:
        value = float(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key}={fields.get(key)!r}, expected float")
        return None
    if not math.isfinite(value):
        errors.append(f"{key}={value}, expected finite")
        return None
    return value


def run_one(
    spec: RunSpec,
    bin_dir: Path,
    timeout: float,
    frames: int,
    debugger: Path | None,
) -> tuple[bool, dict[str, str]]:
    snippet_argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=streaming-custom-pruners",
        f"--execution={spec.execution}",
        f"--frames={frames}",
        "--dt=0.0166666675",
        "--dispatcher-threads=8",
        "--seed=42",
    ]
    argv = snippet_argv
    if debugger is not None:
        argv = [
            str(debugger),
            "-o",
            "-g",
            "-G",
            "-c",
            "sxe av;g;.ecxr;kb;q",
            *snippet_argv,
        ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    result = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout
    )
    combined = result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr
    lines = [
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
    if len(lines) != 1:
        errors.append(f"gate count is {len(lines)}, expected exactly 1")
    else:
        fields, parse_errors = parse_gate(lines[0])
        errors.extend(parse_errors)

    required = {
        "schema": "1",
        "snippet": "SnippetMultiPruners",
        "solver": spec.solver,
        "case": "streaming-custom-pruners",
        "execution": spec.execution,
        "frames": str(frames),
        "prunersCreated": "82",
        "minRegions": "64",
        "maxRegions": "81",
        "finalRegions": "81",
        "maxStreamingObjects": "2673",
        "finalStreamingObjects": "2673",
        "activePruners": "82",
        "assignedObjects": "2675",
        "finalStaticActors": "2674",
        "finalDynamicActors": "1",
        "buildStarts": str(frames),
        "buildFinishes": str(frames),
        "buildTaskSubmissions": str(frames * 82),
        "streamingRaycasts": str(frames),
        "solverQueryHits": str(frames),
        "completedFrames": str(frames),
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
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
        "regionAdds",
        "regionRemoves",
        "regionUpdates",
        "getPrunerIndexCalls",
        "processPrunerCalls",
    ):
        value = parse_int(fields, key, errors)
        if value is not None and value <= 0:
            errors.append(f"{key}={value}, expected > 0")
    region_adds = parse_int(fields, "regionAdds", errors)
    if region_adds is not None and region_adds <= 64:
        errors.append(f"regionAdds={region_adds}, expected > 64")
    streaming_hits = parse_int(fields, "streamingRaycastHits", errors)
    if streaming_hits is not None and streaming_hits <= frames // 2:
        errors.append(
            f"streamingRaycastHits={streaming_hits}, "
            f"expected > {frames // 2}"
        )

    min_y = parse_float(fields, "minBodyY", errors)
    final_y = parse_float(fields, "finalBodyY", errors)
    max_speed = parse_float(fields, "maxBodySpeed", errors)
    displacement = parse_float(fields, "bodyDisplacement", errors)
    if min_y is not None and not 0.0 < min_y < 1.0:
        errors.append(f"minBodyY={min_y}, expected 0 < value < 1")
    if final_y is not None and not 0.0 < final_y < 1.0:
        errors.append(f"finalBodyY={final_y}, expected 0 < value < 1")
    if max_speed is not None and not 0.0 < max_speed < 50.0:
        errors.append(
            f"maxBodySpeed={max_speed}, expected 0 < value < 50"
        )
    if displacement is not None and not 2.0 < displacement < 5.0:
        errors.append(
            f"bodyDisplacement={displacement}, expected 2 < value < 5"
        )
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")

    print(
        f"[MULTI_PRUNERS_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[MULTI_PRUNERS_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    keys = (
        "prunersCreated",
        "regionAdds",
        "regionRemoves",
        "regionUpdates",
        "minRegions",
        "maxRegions",
        "finalRegions",
        "maxStreamingObjects",
        "finalStreamingObjects",
        "activePruners",
        "assignedObjects",
        "finalStaticActors",
        "finalDynamicActors",
        "buildStarts",
        "buildFinishes",
        "buildTaskSubmissions",
        "streamingRaycasts",
        "streamingRaycastHits",
        "solverQueryHits",
        "getPrunerIndexCalls",
        "processPrunerCalls",
        "completedFrames",
        "minBodyY",
        "finalBodyY",
        "maxBodySpeed",
        "bodyDisplacement",
        "nonFinite",
        "fetchFailures",
        "fatalErrors",
        "cleanupComplete",
    )
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        first = results[f"{solver}-{execution}-r1"]
        second = results[f"{solver}-{execution}-r2"]
        mismatches = [key for key in keys if first.get(key) != second.get(key)]
        pair_ok = not mismatches
        passed = passed and pair_ok
        print(
            f"[MULTI_PRUNERS_REPEAT] pair={solver}-{execution} "
            f"status={'PASS' if pair_ok else 'FAIL'} "
            f"mismatches={','.join(mismatches) if mismatches else 'none'}"
        )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "acceptance"),
        default="probe",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument(
        "--debugger",
        type=Path,
        help="diagnostic-only cdb.exe wrapper; acceptance leaves this unset",
    )
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            f"[MULTI_PRUNERS_RUNNER_ERROR] missing executable: "
            f"{bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0:
        print("[MULTI_PRUNERS_RUNNER_ERROR] --timeout must be positive")
        return 2
    if args.frames <= 0:
        print("[MULTI_PRUNERS_RUNNER_ERROR] --frames must be positive")
        return 2
    debugger = args.debugger.resolve() if args.debugger else None
    if debugger is not None and not debugger.is_file():
        print(
            f"[MULTI_PRUNERS_RUNNER_ERROR] missing debugger: {debugger}"
        )
        return 2

    accepted = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(
            spec, bin_dir, args.timeout, args.frames, debugger
        )
        accepted = accepted and passed
        results[spec.name] = fields
    if accepted and args.mode == "acceptance":
        accepted = compare_repeats(results)
    print(
        f"[MULTI_PRUNERS_MATRIX] mode={args.mode} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
