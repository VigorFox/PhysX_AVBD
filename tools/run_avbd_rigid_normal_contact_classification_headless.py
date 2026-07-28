#!/usr/bin/env python3
"""Classify body-static normal work in the full ToleranceScale scene."""

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
EXECUTABLE = "SnippetToleranceScale_64.exe"
FRAMES = 150
CLASS_PREFIX = "[avbd:normal-class] "

COUNT_FIELDS = (
    "onsetRows",
    "supportRows",
    "rowAge0",
    "rowAge1",
    "rowAge2",
    "rowAge3",
    "managerAge0",
    "managerAge1",
    "managerAge2",
    "managerAge3",
    "rowMissOnSupport",
    "onsetFinalizeBodies",
    "supportFinalizeBodies",
    "onsetFinalizeCorrections",
    "supportFinalizeCorrections",
    "onsetDepenEligible",
    "supportDepenEligible",
    "onsetDepenCorrections",
    "supportDepenCorrections",
    "onsetShallowDepenCorrections",
    "onsetDeepDepenCorrections",
    "supportShallowDepenCorrections",
    "supportDeepDepenCorrections",
)
DISTANCE_FIELDS = (
    "onsetPreRaw",
    "onsetPreEffective",
    "onsetPostRaw",
    "onsetPostEffective",
    "onsetAlphaC0",
    "onsetAlOutward",
    "supportPreRaw",
    "supportPreEffective",
    "supportPostRaw",
    "supportPostEffective",
    "supportAlphaC0",
    "supportAlOutward",
    "onsetDepenDistance",
    "supportDepenDistance",
    "onsetShallowDepenDistance",
    "onsetDeepDepenDistance",
    "supportShallowDepenDistance",
    "supportDeepDepenDistance",
)
VELOCITY_FIELDS = (
    "onsetPoseSeparatingVelocity",
    "supportPoseSeparatingVelocity",
    "onsetFinalizeDelta",
    "supportFinalizeDelta",
)
VALUE_FIELDS = DISTANCE_FIELDS + VELOCITY_FIELDS


@dataclass(frozen=True)
class RunSpec:
    execution: str
    repeat: int

    @property
    def name(self) -> str:
        return f"avbd-{self.execution}-r{self.repeat}"


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


def summarize_segment(
    lines: list[str], label: str, first_frame: int,
) -> tuple[dict[str, float], list[str]]:
    summary = {key: 0.0 for key in COUNT_FIELDS + VALUE_FIELDS}
    errors: list[str] = []
    frames: list[int] = []
    for line in lines:
        fields, parse_errors = parse_fields(line[len(CLASS_PREFIX) :])
        errors.extend(f"{label}: {error}" for error in parse_errors)
        try:
            frames.append(int(fields["frame"]))
        except (KeyError, ValueError):
            errors.append(f"{label}: missing/non-integer frame")
        for key in COUNT_FIELDS + VALUE_FIELDS:
            try:
                value = float(fields[key])
            except (KeyError, ValueError):
                errors.append(f"{label}: missing/non-numeric field {key}")
                continue
            if not math.isfinite(value):
                errors.append(f"{label}: non-finite field {key}")
                continue
            summary[key] += value

    if frames != list(range(first_frame, first_frame + FRAMES)):
        errors.append(
            f"{label}: diagnostic frames are not exactly "
            f"{first_frame}..{first_frame + FRAMES - 1}"
        )
    if (
        summary["managerAge0"]
        + summary["managerAge1"]
        + summary["managerAge2"]
        + summary["managerAge3"]
        != summary["supportRows"]
    ):
        errors.append(f"{label}: manager-age buckets do not cover support rows")
    if (
        summary["rowAge0"]
        + summary["rowAge1"]
        + summary["rowAge2"]
        + summary["rowAge3"]
        > summary["onsetRows"] + summary["supportRows"]
    ):
        errors.append(f"{label}: row-cache ages exceed total rows")
    if summary["onsetRows"] <= 0 or summary["supportRows"] <= 0:
        errors.append(f"{label}: onset/support evidence is incomplete")
    if (
        summary["onsetFinalizeCorrections"]
        > summary["onsetFinalizeBodies"]
    ):
        errors.append(f"{label}: onset corrections exceed finalize bodies")
    if (
        summary["supportFinalizeCorrections"]
        > summary["supportFinalizeBodies"]
    ):
        errors.append(f"{label}: support corrections exceed finalize bodies")
    for class_name in ("onset", "support"):
        if (
            summary[f"{class_name}ShallowDepenCorrections"]
            + summary[f"{class_name}DeepDepenCorrections"]
            != summary[f"{class_name}DepenCorrections"]
        ):
            errors.append(
                f"{label}: {class_name} depen correction buckets "
                "do not match total"
            )
        if (
            abs(
                summary[f"{class_name}ShallowDepenDistance"]
                + summary[f"{class_name}DeepDepenDistance"]
                - summary[f"{class_name}DepenDistance"]
            )
            > 2e-5
        ):
            errors.append(
                f"{label}: {class_name} depen distance buckets "
                "do not match total"
            )
    return summary, errors


def normalized(summary: dict[str, float], length_scale: float) -> dict[str, float]:
    result = dict(summary)
    for key in VALUE_FIELDS:
        result[key] /= length_scale
    return result


def format_summary(summary: dict[str, float]) -> str:
    return " ".join(
        f"{key}={summary[key]:.9g}"
        for key in COUNT_FIELDS + VALUE_FIELDS
    )


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float, dispatcher_threads: int,
) -> tuple[bool, dict[str, dict[str, float]]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        "--solver=avbd",
        "--case=scale-pair",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
        "--dt=0.0166666675",
        f"--dispatcher-threads={dispatcher_threads}",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = "avbd"
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(FRAMES)
    env["PHYSX_AVBD_ITER_DIAG"] = "1"
    env["PHYSX_AVBD_ITER_DIAG_EVERY"] = "1"
    env["PHYSX_AVBD_ITER_DIAG_SEQUENTIAL"] = (
        "1" if spec.execution == "sequential" else "0"
    )
    env["PHYSX_AVBD_NORMAL_ROW_DIAG"] = "1"
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
    class_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(CLASS_PREFIX)
    ]
    errors: list[str] = []
    gate: dict[str, str] = {}
    summaries: dict[str, dict[str, float]] = {}
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
        gate, parse_errors = parse_fields(
            " ".join(gate_lines[0].split()[1:])
        )
        errors.extend(parse_errors)
    required = {
        "schema": "1",
        "snippet": "SnippetToleranceScale",
        "solver": "avbd",
        "case": "scale-pair",
        "execution": spec.execution,
        "frames": str(FRAMES),
        "runs": "2",
        "baseCompleted": str(FRAMES),
        "scaledCompleted": str(FRAMES),
        "baseBodies": "276",
        "scaledBodies": "276",
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
        "fatalErrors": "0",
        "cleanupComplete": "2",
        "pvd": "0",
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            errors.append(f"{key}={gate.get(key)!r}, expected {expected!r}")
    if len(class_lines) != FRAMES * 2:
        errors.append(
            f"class diagnostic count is {len(class_lines)}, "
            f"expected {FRAMES * 2}"
        )
    else:
        base, base_errors = summarize_segment(
            class_lines[:FRAMES], "base", 1
        )
        scaled, scaled_errors = summarize_segment(
            class_lines[FRAMES:], "scaled", FRAMES + 1
        )
        errors.extend(base_errors)
        errors.extend(scaled_errors)
        summaries["base"] = normalized(base, 1.0)
        summaries["scaledNormalized"] = normalized(scaled, 100.0)

    print(
        "[RIGID_NORMAL_CLASS_RUN] "
        f"name={spec.name} classLines={len(class_lines)} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for label, summary in summaries.items():
        print(
            "[RIGID_NORMAL_CLASS_SCALE] "
            f"name={spec.name} scale={label} {format_summary(summary)}"
        )
    for error in errors:
        print(f"[RIGID_NORMAL_CLASS_ERROR] name={spec.name} error={error}")
    return not errors, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dispatcher-threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[RIGID_NORMAL_CLASS_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if (
        args.timeout <= 0.0
        or args.dispatcher_threads <= 0
        or args.repeats <= 0
    ):
        print(
            "[RIGID_NORMAL_CLASS_RUNNER_ERROR] "
            "--timeout, --dispatcher-threads, and --repeats must be positive"
        )
        return 2

    run_specs = [
        RunSpec(execution, repeat)
        for repeat in range(1, args.repeats + 1)
        for execution in ("parallel", "sequential")
    ]
    passed = 0
    for spec in run_specs:
        ok, _ = run_one(
            spec, bin_dir, args.timeout, args.dispatcher_threads
        )
        if not ok:
            break
        passed += 1
    expected = len(run_specs)
    matrix_passed = passed == expected
    print(
        "[RIGID_NORMAL_CLASS_MATRIX] "
        f"passed={passed} failed={expected - passed} expected={expected} "
        f"status={'PASS' if matrix_passed else 'FAIL'}"
    )
    return 0 if matrix_passed else 1


if __name__ == "__main__":
    sys.exit(main())
