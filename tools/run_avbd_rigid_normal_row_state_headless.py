#!/usr/bin/env python3
"""Capture rigid body-static normal row evidence headlessly."""

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
EXECUTABLE = "SnippetContactModification_64.exe"
CASES = ("ownership-shallow", "ownership-deep")
FRAMES = 120
ROW_PREFIX = "[avbd:normal-row] "
CLASS_PREFIX = "[avbd:normal-class] "


@dataclass(frozen=True)
class RunSpec:
    case_name: str
    execution: str
    repeat: int

    @property
    def name(self) -> str:
        return f"{self.case_name}-avbd-{self.execution}-r{self.repeat}"


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


def summarize_rows(
    lines: list[str], expect_alpha_c0: str, case_name: str,
) -> tuple[dict[str, float], list[str]]:
    count_fields = (
        "warmHits",
        "warmMisses",
        "finiteBudgetRows",
        "unlimitedBudgetRows",
        "finalizeCorrections",
    )
    sum_fields = (
        "preAlRawPenetration",
        "postAlRawPenetration",
        "alphaC0Offset",
        "preAlPenetration",
        "postAlPenetration",
        "postAlSeparation",
        "alOutwardDistance",
        "alInwardDistance",
        "poseSeparatingVelocity",
        "allowedSeparatingVelocity",
        "finiteRemainingImpulse",
        "finalizeDelta",
    )
    max_fields = (
        "restoredLambdaMax",
        "restoredPenaltyMax",
        "initialPenaltyMax",
    )
    summary = {key: 0.0 for key in count_fields + sum_fields + max_fields}
    errors: list[str] = []
    frames: list[int] = []

    for line in lines:
        fields, parse_errors = parse_fields(line[len(ROW_PREFIX) :])
        errors.extend(parse_errors)
        try:
            frame = int(fields["frame"])
            frames.append(frame)
        except (KeyError, ValueError):
            errors.append("missing/non-integer frame")
        for key in count_fields + sum_fields + max_fields:
            try:
                value = float(fields[key])
            except (KeyError, ValueError):
                errors.append(f"missing/non-numeric row field: {key}")
                continue
            if not math.isfinite(value):
                errors.append(f"non-finite row field: {key}")
                continue
            if key in max_fields:
                summary[key] = max(summary[key], value)
            else:
                summary[key] += value

    if frames != list(range(1, FRAMES + 1)):
        errors.append("row diagnostic frames are not exactly 1..120")
    if summary["warmHits"] + summary["warmMisses"] != FRAMES:
        errors.append("warmstart ownership does not cover every normal row")
    if summary["warmHits"] <= 0 or summary["warmMisses"] <= 0:
        errors.append("warmstart hit/miss evidence is incomplete")
    if summary["preAlPenetration"] <= 0.0:
        errors.append("pre-AL penetration was not observed")
    if expect_alpha_c0 == "split":
        if summary["preAlRawPenetration"] <= summary["preAlPenetration"]:
            errors.append("alpha-softened AL residual was not distinguished")
        if summary["postAlRawPenetration"] <= summary["postAlPenetration"]:
            errors.append("raw post-AL penetration was not distinguished")
        if summary["alphaC0Offset"] <= 0.0:
            errors.append("alpha*C0 ownership offset was not observed")
    else:
        zero_tolerance = 1e-7
        if summary["alphaC0Offset"] > zero_tolerance:
            errors.append("body-static normal retained a nonzero alpha*C0 offset")
        if (
            abs(
                summary["preAlRawPenetration"]
                - summary["preAlPenetration"]
            )
            > zero_tolerance
        ):
            errors.append("raw/effective pre-AL penetration diverged")
        if (
            abs(
                summary["postAlRawPenetration"]
                - summary["postAlPenetration"]
            )
            > zero_tolerance
        ):
            errors.append("raw/effective post-AL penetration diverged")
    if summary["alOutwardDistance"] <= 0.0:
        errors.append("outward AL position ownership was not observed")
    if summary["alInwardDistance"] > 1e-8:
        errors.append("unexpected inward AL normal displacement")
    if summary["allowedSeparatingVelocity"] > 1e-8:
        errors.append("inelastic fixture allowed separating velocity")
    if case_name == "ownership-deep":
        if summary["poseSeparatingVelocity"] > 2e-7:
            errors.append("deep geometric recovery leaked into velocity")
        if summary["finalizeCorrections"] != 0.0:
            errors.append("deep recovery still required material finalize")
        if summary["finalizeDelta"] > 2e-7:
            errors.append("deep recovery retained a finalize delta")
    else:
        if summary["poseSeparatingVelocity"] <= 0.0:
            errors.append("pose-derived separating velocity was not observed")
        if summary["finalizeCorrections"] <= 0.0:
            errors.append("inelastic finalize correction was not observed")
        if (
            abs(summary["poseSeparatingVelocity"] - summary["finalizeDelta"])
            > 2e-7
        ):
            errors.append("finalize delta did not consume pose separation")
    if summary["finiteBudgetRows"] != 0.0:
        errors.append("unlimited fixture reported finite impulse budget")
    if summary["unlimitedBudgetRows"] != FRAMES:
        errors.append("unlimited impulse ownership does not cover every row")
    return summary, errors


def summarize_classes(
    lines: list[str], row_summary: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    count_fields = (
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
    sum_fields = (
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
        "onsetPoseSeparatingVelocity",
        "supportPoseSeparatingVelocity",
        "onsetFinalizeDelta",
        "supportFinalizeDelta",
        "onsetDepenDistance",
        "supportDepenDistance",
        "onsetShallowDepenDistance",
        "onsetDeepDepenDistance",
        "supportShallowDepenDistance",
        "supportDeepDepenDistance",
    )
    summary = {key: 0.0 for key in count_fields + sum_fields}
    errors: list[str] = []
    frames: list[int] = []
    for line in lines:
        fields, parse_errors = parse_fields(line[len(CLASS_PREFIX) :])
        errors.extend(parse_errors)
        try:
            frames.append(int(fields["frame"]))
        except (KeyError, ValueError):
            errors.append("missing/non-integer class frame")
        for key in count_fields + sum_fields:
            try:
                value = float(fields[key])
            except (KeyError, ValueError):
                errors.append(f"missing/non-numeric class field: {key}")
                continue
            if not math.isfinite(value):
                errors.append(f"non-finite class field: {key}")
                continue
            summary[key] += value

    if frames != list(range(1, FRAMES + 1)):
        errors.append("class diagnostic frames are not exactly 1..120")
    if summary["onsetRows"] + summary["supportRows"] != FRAMES:
        errors.append("onset/support rows do not cover every normal row")
    if (
        summary["rowAge0"]
        + summary["rowAge1"]
        + summary["rowAge2"]
        + summary["rowAge3"]
        != row_summary["warmHits"]
    ):
        errors.append("row-cache age buckets do not cover warm hits")
    if (
        summary["managerAge0"]
        + summary["managerAge1"]
        + summary["managerAge2"]
        + summary["managerAge3"]
        != summary["supportRows"]
    ):
        errors.append("manager-age buckets do not cover support rows")
    if summary["rowMissOnSupport"] > row_summary["warmMisses"]:
        errors.append("row misses on support exceed total warmstart misses")
    if summary["onsetRows"] <= 0 or summary["supportRows"] <= 0:
        errors.append("onset/support evidence is incomplete")
    if (
        summary["onsetFinalizeBodies"]
        + summary["supportFinalizeBodies"]
        != FRAMES
    ):
        errors.append("onset/support finalize bodies do not cover every frame")

    class_to_row = {
        "preAlRawPenetration": ("onsetPreRaw", "supportPreRaw"),
        "preAlPenetration": (
            "onsetPreEffective",
            "supportPreEffective",
        ),
        "postAlRawPenetration": ("onsetPostRaw", "supportPostRaw"),
        "postAlPenetration": (
            "onsetPostEffective",
            "supportPostEffective",
        ),
        "alphaC0Offset": ("onsetAlphaC0", "supportAlphaC0"),
        "alOutwardDistance": ("onsetAlOutward", "supportAlOutward"),
        "poseSeparatingVelocity": (
            "onsetPoseSeparatingVelocity",
            "supportPoseSeparatingVelocity",
        ),
        "finalizeDelta": ("onsetFinalizeDelta", "supportFinalizeDelta"),
    }
    for row_key, class_keys in class_to_row.items():
        class_total = sum(summary[key] for key in class_keys)
        if abs(class_total - row_summary[row_key]) > 2e-7:
            errors.append(f"class total does not match row total: {row_key}")
    if (
        summary["onsetFinalizeCorrections"]
        + summary["supportFinalizeCorrections"]
        != row_summary["finalizeCorrections"]
    ):
        errors.append("class finalize corrections do not match row total")
    for class_name in ("onset", "support"):
        if (
            summary[f"{class_name}ShallowDepenCorrections"]
            + summary[f"{class_name}DeepDepenCorrections"]
            != summary[f"{class_name}DepenCorrections"]
        ):
            errors.append(
                f"{class_name} depen correction buckets do not match total"
            )
        if (
            abs(
                summary[f"{class_name}ShallowDepenDistance"]
                + summary[f"{class_name}DeepDepenDistance"]
                - summary[f"{class_name}DepenDistance"]
            )
            > 2e-7
        ):
            errors.append(
                f"{class_name} depen distance buckets do not match total"
            )
    return summary, errors


def run_one(
    spec: RunSpec,
    bin_dir: Path,
    timeout_seconds: float,
    expect_alpha_c0: str,
) -> tuple[bool, dict[str, float]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        "--solver=avbd",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
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
    row_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(ROW_PREFIX)
    ]
    class_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(CLASS_PREFIX)
    ]
    errors: list[str] = []
    gate: dict[str, str] = {}
    summary: dict[str, float] = {}
    class_summary: dict[str, float] = {}

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
        "schema": "2",
        "snippet": "SnippetContactModification",
        "solver": "avbd",
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
        "pvd": "0",
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            errors.append(f"{key}={gate.get(key)!r}, expected {expected!r}")
    if len(row_lines) != FRAMES:
        errors.append(
            f"row diagnostic count is {len(row_lines)}, expected {FRAMES}"
        )
    else:
        summary, row_errors = summarize_rows(
            row_lines, expect_alpha_c0, spec.case_name
        )
        errors.extend(row_errors)
    if len(class_lines) != FRAMES:
        errors.append(
            f"class diagnostic count is {len(class_lines)}, expected {FRAMES}"
        )
    elif summary:
        class_summary, class_errors = summarize_classes(class_lines, summary)
        errors.extend(class_errors)

    print(
        "[RIGID_NORMAL_ROW_RUN] "
        f"name={spec.name} rowLines={len(row_lines)} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if summary:
        print(
            "[RIGID_NORMAL_ROW_STATS] "
            f"name={spec.name} "
            + " ".join(
                f"{key}={value:.9g}"
                for key, value in sorted(summary.items())
            )
        )
    if class_summary:
        print(
            "[RIGID_NORMAL_CLASS_STATS] "
            f"name={spec.name} "
            + " ".join(
                f"{key}={value:.9g}"
                for key, value in sorted(class_summary.items())
            )
        )
    for error in errors:
        print(f"[RIGID_NORMAL_ROW_ERROR] name={spec.name} error={error}")
    return not errors, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--case",
        choices=CASES,
        default="ownership-shallow",
    )
    parser.add_argument(
        "--expect-alpha-c0",
        choices=("auto", "split", "zero"),
        default="auto",
    )
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[RIGID_NORMAL_ROW_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0.0 or args.repeats <= 0:
        print(
            "[RIGID_NORMAL_ROW_RUNNER_ERROR] "
            "--timeout and --repeats must be positive"
        )
        return 2

    run_specs = [
        RunSpec(args.case, execution, repeat)
        for repeat in range(1, args.repeats + 1)
        for execution in ("parallel", "sequential")
    ]
    passed = 0
    for spec in run_specs:
        expect_alpha_c0 = (
            "zero"
            if args.expect_alpha_c0 == "auto"
            and spec.case_name == "ownership-deep"
            else "split"
            if args.expect_alpha_c0 == "auto"
            else args.expect_alpha_c0
        )
        ok, _ = run_one(
            spec, bin_dir, args.timeout, expect_alpha_c0
        )
        if not ok:
            break
        passed += 1
    expected = len(run_specs)
    matrix_passed = passed == expected
    print(
        "[RIGID_NORMAL_ROW_MATRIX] "
        f"passed={passed} failed={expected - passed} expected={expected} "
        f"status={'PASS' if matrix_passed else 'FAIL'}"
    )
    return 0 if matrix_passed else 1


if __name__ == "__main__":
    sys.exit(main())
