#!/usr/bin/env python3
"""Reproduce the SnippetJoint dynamic-dynamic force-pair authority headlessly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetJoint_64.exe"
FREQUENCIES = (30, 60, 120)
ORDERS = ("normal", "swapped")


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    frequency: int
    actor_order: str
    expected_status: str


def make_specs(mode: str) -> tuple[RunSpec, ...]:
    if mode == "authority":
        lanes = (("pgs", "parallel"), ("tgs", "parallel"))
    elif mode == "probe":
        lanes = (("avbd", "parallel"), ("avbd", "sequential"))
    else:
        lanes = (
            ("pgs", "parallel"),
            ("tgs", "parallel"),
            ("avbd", "parallel"),
            ("avbd", "sequential"),
        )
    specs: list[RunSpec] = []
    for solver, execution in lanes:
        for frequency in FREQUENCIES:
            for actor_order in ORDERS:
                expected_status = "PASS"
                if mode == "baseline" and solver == "avbd":
                    expected_status = "FAIL"
                specs.append(
                    RunSpec(
                        (
                            f"{solver}-{execution}-{frequency}hz-"
                            f"{actor_order}"
                        ),
                        solver,
                        execution,
                        frequency,
                        actor_order,
                        expected_status,
                    )
                )
    return tuple(specs)


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
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, dict[str, str]]:
    frames = spec.frequency * 10
    dt = 1.0 / spec.frequency
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        "--case=force-pair",
        f"--execution={spec.execution}",
        f"--actor-order={spec.actor_order}",
        f"--frames={frames}",
        f"--dt={dt:.12g}",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
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
    errors: list[str] = []
    fields: dict[str, str] = {}
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

    exact = {
        "schema": "1",
        "snippet": "SnippetJoint",
        "case": "force-pair",
        "joint": "fixed",
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(frames),
        "completedFrames": str(frames),
        "seed": "1",
        "dispatcherThreads": "2",
        "capability": "PARTIAL",
        "validation": "PROBE",
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "launchFailures": "0",
        "topologyDynamicActors": "2",
        "topologyStaticActors": "0",
        "topologyConstraints": "1",
        "finalDynamicActors": "2",
        "finalStaticActors": "0",
        "finalConstraints": "1",
        "cleanupDynamicActors": "0",
        "cleanupStaticActors": "0",
        "cleanupConstraints": "0",
        "cleanupComplete": "1",
        "forceFixture": "dynamic-pair",
        "forceActorOrder": spec.actor_order,
        "forcePairActorOrderValid": "1",
        "status": spec.expected_status,
    }
    for key, expected in exact.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )

    expected_exit = 0 if spec.expected_status == "PASS" else 1
    if result.returncode != expected_exit:
        errors.append(
            f"exit code {result.returncode}, expected {expected_exit}"
        )
    reason = fields.get("reason")
    if spec.expected_status == "PASS":
        if reason != "none":
            errors.append(f"reason={reason!r}, expected 'none'")
    elif reason not in {
        "force_unit_dt",
        "force_body_stability",
        "force_pair_momentum",
        "force_pair_relative_stability",
    }:
        errors.append(f"unexpected AVBD failure reason: {reason!r}")
    metric_keys = (
        "forceRatio",
        "meanSampleForceRatio",
        "forcePairMaxSeparationError",
        "forcePairMaxRelativeSpeed",
        "forcePairMaxCenterOfMassError",
        "forcePairMaxTotalMomentum",
        "forcePairFinalCenterOfMassError",
        "forcePairFinalTotalMomentum",
    )
    metrics = {
        key: parse_float(fields, key, errors) for key in metric_keys
    }
    for key in (
        "forcePairTotalMomentumMax",
        "forcePairSeparationErrorMax",
        "forcePairRelativeSpeedMax",
        "forcePairCenterOfMassErrorMax",
    ):
        threshold = parse_float(fields, key, errors)
        if threshold is not None and abs(threshold - 0.001) > 1e-8:
            errors.append(f"{key}={threshold}, expected 0.001")

    if spec.expected_status == "FAIL" and all(
        value is not None
        for value in (
            metrics["forceRatio"],
            metrics["meanSampleForceRatio"],
            metrics["forcePairMaxCenterOfMassError"],
            metrics["forcePairMaxTotalMomentum"],
        )
    ):
        physical_failure = (
            not 0.9 <= metrics["forceRatio"] <= 1.1
            or not 0.9 <= metrics["meanSampleForceRatio"] <= 1.1
            or metrics["forcePairMaxCenterOfMassError"] > 0.001
            or metrics["forcePairMaxTotalMomentum"] > 0.001
        )
        if not physical_failure:
            errors.append("FAIL status lacks an independent red metric")

    print(
        f"[FORCE_PAIR_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"forceRatio={fields.get('forceRatio', 'MISSING')} "
        f"maxPosition={fields.get('forceMaxPositionError', 'MISSING')} "
        f"maxLinearSpeed={fields.get('forceMaxLinearSpeed', 'MISSING')} "
        f"maxSeparation={fields.get('forcePairMaxSeparationError', 'MISSING')} "
        f"maxRelativeSpeed={fields.get('forcePairMaxRelativeSpeed', 'MISSING')} "
        f"maxCOM={fields.get('forcePairMaxCenterOfMassError', 'MISSING')} "
        f"maxMomentum={fields.get('forcePairMaxTotalMomentum', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(f"[FORCE_PAIR_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SnippetJoint dynamic-dynamic fixed reaction matrix "
            "without a window."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "baseline", "acceptance"),
        default="baseline",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        parser.error(f"executable not found: {executable}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    selected_specs = make_specs(args.mode)
    passed_runs = 0
    status_counts = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    for spec in selected_specs:
        passed, fields = run_one(spec, bin_dir, args.timeout)
        passed_runs += int(passed)
        status = fields.get("status", "ERROR")
        status_counts[status if status in status_counts else "ERROR"] += 1

    passed = passed_runs == len(selected_specs)
    print(
        f"[FORCE_PAIR_SUMMARY] mode={args.mode} "
        f"runs={passed_runs}/{len(selected_specs)} "
        f"physicalPass={status_counts['PASS']} "
        f"physicalFail={status_counts['FAIL']} "
        f"error={status_counts['ERROR']} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
