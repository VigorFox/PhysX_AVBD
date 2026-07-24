#!/usr/bin/env python3
"""Run SnippetLoadCollection file/reference identity headlessly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import tempfile

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetLoadCollection_64.exe"
FRAMES = 240
CASES = {
    "complete-xml": ("collection.xml",),
    "complete-bin": ("collection.bin",),
    "split-xml": ("collectionA.xml", "collectionB.xml"),
    "split-bin": ("collectionA.bin", "collectionB.bin"),
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    repeat: int
    case: str


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
        RunSpec(
            f"{solver}-{execution}-{case}-r{repeat}",
            solver,
            execution,
            repeat,
            case,
        )
        for repeat in repeats
        for solver, execution in lanes
        for case in CASES
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


def invoke(
    argv: list[str], work_dir: Path, timeout: float
) -> tuple[object, str, dict[str, str], list[str]]:
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    result = run_headless_process(
        argv, cwd=work_dir, env=env, timeout_seconds=timeout
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
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")
    return result, combined, fields, errors


def generate_files(
    executable: Path, work_dir: Path, timeout: float
) -> bool:
    argv = [
        str(executable),
        "--headless",
        "--solver=tgs",
        "--case=generate",
        "--execution=parallel",
        "--dispatcher-threads=1",
        "--generateExampleFiles",
    ]
    result, combined, fields, errors = invoke(argv, work_dir, timeout)
    required = {
        "schema": "1",
        "snippet": "SnippetLoadCollection",
        "solver": "tgs",
        "case": "generate",
        "execution": "parallel",
        "filesGenerated": "6",
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
    for filename in (
        "collection.xml",
        "collectionA.xml",
        "collectionB.xml",
        "collection.bin",
        "collectionA.bin",
        "collectionB.bin",
    ):
        path = work_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty generated file: {filename}")
    print(
        "[LOAD_COLLECTION_GENERATE] "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[LOAD_COLLECTION_GENERATE_ERROR] error={error}")
    return not errors


def run_one(
    spec: RunSpec,
    executable: Path,
    work_dir: Path,
    timeout: float,
) -> tuple[bool, dict[str, str]]:
    filenames = CASES[spec.case]
    argv = [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case}",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
        "--dt=0.0166666675",
        "--dispatcher-threads=4",
        "--seed=1",
        *filenames,
    ]
    result, combined, fields, errors = invoke(argv, work_dir, timeout)
    binary_count = sum(filename.endswith(".bin") for filename in filenames)
    xml_count = len(filenames) - binary_count
    required = {
        "schema": "1",
        "snippet": "SnippetLoadCollection",
        "solver": spec.solver,
        "case": spec.case,
        "execution": spec.execution,
        "frames": str(FRAMES),
        "filesRequested": str(len(filenames)),
        "filesLoaded": str(len(filenames)),
        "binaryFiles": str(binary_count),
        "xmlFiles": str(xml_count),
        "collectionsAdded": str(len(filenames)),
        "staticActors": "1",
        "dynamicActors": "1",
        "constraints": "0",
        "actorShapeRefs": "2",
        "externalShapeRefs": "2",
        "materialIdentity": "1",
        "initialDynamicY": "8",
        "completedFrames": str(FRAMES),
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
    for key in ("finalDynamicY", "minDynamicY", "maxDynamicSpeed"):
        try:
            value = float(fields[key])
            if not math.isfinite(value):
                errors.append(f"{key}={value}, expected finite")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected float")
    try:
        final_y = float(fields["finalDynamicY"])
        min_y = float(fields["minDynamicY"])
        if final_y >= 7.5:
            errors.append(f"finalDynamicY={final_y}, expected < 7.5")
        if min_y <= 1.0:
            errors.append(f"minDynamicY={min_y}, expected > 1.0")
    except (KeyError, ValueError):
        pass

    print(
        f"[LOAD_COLLECTION_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[LOAD_COLLECTION_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    keys = (
        "filesLoaded",
        "collectionsAdded",
        "staticActors",
        "dynamicActors",
        "actorShapeRefs",
        "externalShapeRefs",
        "materialIdentity",
        "initialDynamicY",
        "finalDynamicY",
        "minDynamicY",
        "maxDynamicSpeed",
    )
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        for case in CASES:
            first = results[f"{solver}-{execution}-{case}-r1"]
            second = results[f"{solver}-{execution}-{case}-r2"]
            mismatches = [
                key for key in keys if first.get(key) != second.get(key)
            ]
            pair_ok = not mismatches
            passed = passed and pair_ok
            print(
                f"[LOAD_COLLECTION_REPEAT] pair={solver}-{execution}-{case} "
                f"status={'PASS' if pair_ok else 'FAIL'} "
                "mismatches="
                f"{','.join(mismatches) if mismatches else 'none'}"
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
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[LOAD_COLLECTION_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0:
        print("[LOAD_COLLECTION_RUNNER_ERROR] --timeout must be positive")
        return 2

    accepted = True
    results: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(
        prefix="physx_avbd_load_collection_"
    ) as temp_dir:
        work_dir = Path(temp_dir)
        accepted = generate_files(executable, work_dir, args.timeout)
        if accepted:
            for spec in specs(args.mode):
                passed, fields = run_one(
                    spec, executable, work_dir, args.timeout
                )
                accepted = accepted and passed
                results[spec.name] = fields
        if accepted and args.mode == "acceptance":
            accepted = compare_repeats(results)
    print(
        f"[LOAD_COLLECTION_MATRIX] mode={args.mode} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
