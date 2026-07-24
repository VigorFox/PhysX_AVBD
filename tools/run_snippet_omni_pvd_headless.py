#!/usr/bin/env python3
"""Run SnippetOmniPvd recording/readback coverage headlessly."""

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
EXECUTABLE = "SnippetOmniPvd_64.exe"
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
) -> tuple[bool, dict[str, str]]:
    with tempfile.TemporaryDirectory(
        prefix="PhysX_AVBD_omnipvd_headless_"
    ) as temp_name:
        output_path = Path(temp_name) / f"{spec.name}.ovd"
        argv = [
            str(bin_dir / EXECUTABLE),
            "--headless",
            f"--solver={spec.solver}",
            "--case=record-scene",
            f"--execution={spec.execution}",
            f"--frames={frames}",
            "--dt=0.0166666675",
            "--dispatcher-threads=2",
            "--seed=1",
            f"--omnipvdfile={output_path}",
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

        exact = {
            "schema": "1",
            "snippet": "SnippetOmniPvd",
            "solver": spec.solver,
            "case": "record-scene",
            "execution": spec.execution,
            "frames": str(frames),
            "completedFrames": str(frames),
            "samplingStarted": "1",
            "readerStarted": "1",
            "startFrames": str(frames * 2 + 1),
            "stopFrames": str(frames * 2 + 1),
            "requiredClasses": "6",
            "physicsCreates": "1",
            "sceneCreates": "1",
            "materialCreates": "1",
            "rigidStaticCreates": "1",
            "rigidDynamicCreates": "1",
            "physicsDestroys": "1",
            "sceneDestroys": "1",
            "materialDestroys": "1",
            "rigidStaticDestroys": "1",
            "rigidDynamicDestroys": "1",
            "solverTypeMismatches": "0",
            "initialStaticActors": "1",
            "initialDynamicActors": "1",
            "nonFinite": "0",
            "fetchFailures": "0",
            "fatalErrors": "0",
            "cleanupComplete": "1",
            "pvd": "0",
            "status": "PASS",
            "reason": "none",
            "validation": "GATED",
        }
        for key, expected in exact.items():
            if fields.get(key) != expected:
                errors.append(
                    f"{key}={fields.get(key)!r}, expected {expected!r}"
                )

        file_bytes = parse_int(fields, "fileBytes", errors)
        actual_bytes = (
            output_path.stat().st_size if output_path.is_file() else 0
        )
        if file_bytes is not None and file_bytes != actual_bytes:
            errors.append(
                f"fileBytes={file_bytes}, actual size is {actual_bytes}"
            )
        if actual_bytes < 1024:
            errors.append(
                f"output file size is {actual_bytes}, expected >= 1024"
            )

        version = fields.get("version", "")
        version_parts = version.split(".")
        if (
            len(version_parts) != 3
            or not all(part.isdigit() for part in version_parts)
        ):
            errors.append(f"version={version!r}, expected major.minor.patch")

        for key in (
            "commands",
            "classRegistrations",
            "attributeRegistrations",
            "setAttributes",
            "createObjects",
            "destroyObjects",
            "shapeCreates",
            "shapeDestroys",
            "solverTypeSamples",
            "solverTypeMatches",
        ):
            value = parse_int(fields, key, errors)
            if value is not None and value <= 0:
                errors.append(f"{key}={value}, expected positive")
        samples = parse_int(fields, "solverTypeSamples", errors)
        matches = parse_int(fields, "solverTypeMatches", errors)
        if (
            samples is not None
            and matches is not None
            and samples != matches
        ):
            errors.append(
                f"solver metadata samples/matches={samples}/{matches}"
            )
        shape_creates = parse_int(fields, "shapeCreates", errors)
        shape_destroys = parse_int(fields, "shapeDestroys", errors)
        if shape_creates is not None and shape_creates < 2:
            errors.append(f"shapeCreates={shape_creates}, expected >= 2")
        if shape_destroys is not None and shape_destroys < 2:
            errors.append(f"shapeDestroys={shape_destroys}, expected >= 2")

        min_y = parse_float(fields, "minProjectileY", errors)
        max_speed = parse_float(fields, "maxProjectileSpeed", errors)
        displacement = parse_float(
            fields, "projectileDisplacement", errors
        )
        if min_y is not None and abs(min_y) >= 100000.0:
            errors.append(
                f"minProjectileY={min_y}, expected bounded scene"
            )
        if max_speed is not None and not 1.0 < max_speed < 100000.0:
            errors.append(
                f"maxProjectileSpeed={max_speed}, expected 1..100000"
            )
        if displacement is not None and not 1.0 < displacement < 100000.0:
            errors.append(
                f"projectileDisplacement={displacement}, "
                "expected 1..100000"
            )
        if result.returncode != 0:
            errors.append(f"exit code {result.returncode}, expected 0")

        print(
            f"[OMNI_PVD_RUN] name={spec.name} "
            f"status={fields.get('status', 'MISSING')} "
            f"fileBytes={actual_bytes} exit={result.returncode} "
            f"runner={'PASS' if not errors else 'FAIL'}"
        )
        if combined:
            print(combined.rstrip())
        for error in errors:
            print(f"[OMNI_PVD_RUN_ERROR] name={spec.name} error={error}")
        return not errors, fields


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    keys = (
        "version",
        "commands",
        "classRegistrations",
        "attributeRegistrations",
        "setAttributes",
        "createObjects",
        "destroyObjects",
        "startFrames",
        "stopFrames",
        "requiredClasses",
        "physicsCreates",
        "sceneCreates",
        "materialCreates",
        "rigidStaticCreates",
        "rigidDynamicCreates",
        "shapeCreates",
        "physicsDestroys",
        "sceneDestroys",
        "materialDestroys",
        "rigidStaticDestroys",
        "rigidDynamicDestroys",
        "shapeDestroys",
        "solverTypeSamples",
        "solverTypeMatches",
        "solverTypeMismatches",
        "completedFrames",
        "minProjectileY",
        "maxProjectileSpeed",
        "projectileDisplacement",
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
        pair_passed = not mismatches
        passed = passed and pair_passed
        print(
            f"[OMNI_PVD_REPEAT] pair={solver}-{execution} "
            f"status={'PASS' if pair_passed else 'FAIL'} "
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
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--frames", type=int, default=FRAMES)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            f"[OMNI_PVD_RUNNER_ERROR] missing executable: "
            f"{bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0:
        print("[OMNI_PVD_RUNNER_ERROR] --timeout must be positive")
        return 2
    if args.frames < 60:
        print(
            "[OMNI_PVD_RUNNER_ERROR] --frames must be at least 60"
        )
        return 2

    accepted = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(
            spec, bin_dir, args.timeout, args.frames
        )
        accepted = accepted and passed
        results[spec.name] = fields
    if accepted and args.mode == "acceptance":
        accepted = compare_repeats(results)
    print(
        f"[OMNI_PVD_MATRIX] mode={args.mode} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
