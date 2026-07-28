#!/usr/bin/env python3
"""Freeze AVBD friction/target-velocity ownership evidence headlessly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetContactModification_64.exe"
BASE_CASES = (
    "ownership-target-normal-mu0",
    "ownership-target-normal-mu1",
    "ownership-target-tangent-mu0",
    "ownership-target-tangent-mu1",
    "ownership-target-combined-mu1-finite",
)
DIAG_PREFIX = "[avbd:friction-target] "
ITER_DIAG_PREFIX = "[avbd:iters] "
RESTITUTION_CORRECTIONS_PATTERN = re.compile(
    r"\brestitutionCorrections=(\d+)"
)


@dataclass(frozen=True)
class RunSpec:
    solver: str
    execution: str
    case_name: str
    repeat: int


def parse_fields(line: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append(f"malformed token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate field: {key}")
        fields[key] = value
    return fields, errors


def as_float(fields: dict[str, str], key: str, errors: list[str]) -> float:
    try:
        return float(fields[key])
    except (KeyError, ValueError):
        errors.append(f"missing/non-numeric {key}")
        return 0.0


def as_int(fields: dict[str, str], key: str, errors: list[str]) -> int:
    try:
        return int(fields[key])
    except (KeyError, ValueError):
        errors.append(f"missing/non-integer {key}")
        return 0


def run_one(
    spec: RunSpec,
    bin_dir: Path,
    timeout_seconds: float,
    artifact_dir: Path | None = None,
) -> tuple[bool, dict[str, str], dict[str, float]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        "--frames=120",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = "120"
    env["PHYSX_AVBD_ITER_DIAG"] = "1" if spec.solver == "avbd" else "0"
    env["PHYSX_AVBD_ITER_DIAG_EVERY"] = "1"
    env["PHYSX_AVBD_ITER_DIAG_SEQUENTIAL"] = (
        "1" if spec.execution == "sequential" else "0"
    )
    result = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout_seconds
    )

    combined = result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_name = (
            f"{spec.case_name}-{spec.solver}-{spec.execution}-"
            f"repeat{spec.repeat}.log"
        )
        (artifact_dir / log_name).write_text(
            combined, encoding="utf-8", errors="replace"
        )
    gate_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]
    diag_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(DIAG_PREFIX)
    ]
    iter_diag_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(ITER_DIAG_PREFIX)
    ]
    errors: list[str] = []
    gate: dict[str, str] = {}
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
        errors.append(f"gate count={len(gate_lines)}, expected 1")
    else:
        gate, parse_errors = parse_fields(gate_lines[0])
        errors.extend(parse_errors)

    required = {
        "schema": "2",
        "snippet": "SnippetContactModification",
        "solver": spec.solver,
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": "120",
        "completedFrames": "120",
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
        "identityErrors": "0",
        "scaleReadbackErrors": "0",
        "maxImpulseReadbackErrors": "0",
        "targetVelocityReadbackErrors": "0",
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
        "pvd": "0",
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            errors.append(f"{key}={gate.get(key)!r}, expected {expected!r}")

    diagnostic_sum_fields = (
        "positionAlTargetEvals",
        "bodyStaticSweepTargetRows",
        "bodyStaticSweepTargetCorrections",
        "bodyStaticSweepTargetImpulse",
        "bodyStaticFallbackRows",
        "bodyStaticFallbackCorrections",
        "bodyStaticFallbackImpulse",
        "genericNormalRows",
        "genericNormalCorrections",
        "genericNormalImpulse",
        "genericTangentRows",
        "genericTangentCorrections",
        "genericTangentImpulse",
    )
    diagnostics = {
        "positionAlTargetEvals": 0.0,
        "bodyStaticSweepTargetRows": 0.0,
        "bodyStaticSweepTargetCorrections": 0.0,
        "bodyStaticSweepTargetImpulse": 0.0,
        "bodyStaticFallbackRows": 0.0,
        "bodyStaticFallbackCorrections": 0.0,
        "bodyStaticFallbackImpulse": 0.0,
        "genericNormalRows": 0.0,
        "genericNormalCorrections": 0.0,
        "genericNormalImpulse": 0.0,
        "genericTangentRows": 0.0,
        "genericTangentCorrections": 0.0,
        "genericTangentImpulse": 0.0,
        "genericNormalImpulseFrameMax": 0.0,
        "genericTangentImpulseFrameMax": 0.0,
        "restitutionCorrections": 0.0,
    }
    if spec.solver == "avbd":
        if not diag_lines:
            errors.append("missing AVBD friction-target diagnostics")
        for line in diag_lines:
            fields, parse_errors = parse_fields(line)
            errors.extend(parse_errors)
            for key in diagnostic_sum_fields:
                value = as_float(fields, key, errors)
                diagnostics[key] += value
                if key == "genericNormalImpulse":
                    diagnostics["genericNormalImpulseFrameMax"] = max(
                        diagnostics["genericNormalImpulseFrameMax"],
                        value,
                    )
                elif key == "genericTangentImpulse":
                    diagnostics["genericTangentImpulseFrameMax"] = max(
                        diagnostics["genericTangentImpulseFrameMax"],
                        value,
                    )
        for line in iter_diag_lines:
            match = RESTITUTION_CORRECTIONS_PATTERN.search(line)
            if match is None:
                errors.append(
                    "missing restitutionCorrections in AVBD iteration "
                    "diagnostic"
                )
                continue
            diagnostics["restitutionCorrections"] += float(match.group(1))
    elif diag_lines:
        errors.append("unexpected AVBD diagnostics on TGS lane")

    actor0_count = as_int(gate, "body0Actor0Count", errors)
    actor1_count = as_int(gate, "body0Actor1Count", errors)
    multi_pair_component = spec.case_name.startswith(
        "ownership-passive-friction-component"
    ) or spec.case_name.startswith(
        "ownership-restitution-friction-component"
    )
    if actor0_count + actor1_count <= 0:
        errors.append("contact actor slot was not observed")
    if not multi_pair_component and actor0_count > 0 and actor1_count > 0:
        errors.append("body0 changed contact actor slot within one run")

    print(
        "[RIGID_FRICTION_TARGET_RUN] "
        f"solver={spec.solver} execution={spec.execution} "
        f"case={spec.case_name} repeat={spec.repeat} "
        f"actor0={actor0_count} actor1={actor1_count} "
        f"peakVx={gate.get('peakAbsBody0VelocityX', 'missing')} "
        f"peakVy={gate.get('peakBody0VelocityY', 'missing')} "
        f"peakW={gate.get('peakBody0AngularSpeed', 'missing')} "
        f"finalV={gate.get('finalBody0Speed', 'missing')} "
        f"finalW={gate.get('finalBody0AngularSpeed', 'missing')} "
        f"finalVx={gate.get('finalBody0VelocityX', 'missing')} "
        f"finalWx={gate.get('finalBody0AngularX', 'missing')} "
        f"finalWy={gate.get('finalBody0AngularY', 'missing')} "
        f"finalWz={gate.get('finalBody0AngularZ', 'missing')} "
        f"finalContactVx="
        f"{gate.get('finalBody0ContactVelocityX', 'missing')} "
        f"reported={gate.get('maxReportedImpulse', 'missing')} "
        f"al={diagnostics['positionAlTargetEvals']:.9g} "
        f"sweepRows={diagnostics['bodyStaticSweepTargetRows']:.9g} "
        f"sweepCorrections="
        f"{diagnostics['bodyStaticSweepTargetCorrections']:.9g} "
        f"sweepImpulse={diagnostics['bodyStaticSweepTargetImpulse']:.9g} "
        f"fallbackRows={diagnostics['bodyStaticFallbackRows']:.9g} "
        f"fallbackCorrections="
        f"{diagnostics['bodyStaticFallbackCorrections']:.9g} "
        f"fallbackImpulse="
        f"{diagnostics['bodyStaticFallbackImpulse']:.9g} "
        f"normalRows={diagnostics['genericNormalRows']:.9g} "
        f"normalCorrections={diagnostics['genericNormalCorrections']:.9g} "
        f"normalImpulse={diagnostics['genericNormalImpulse']:.9g} "
        f"tangentRows={diagnostics['genericTangentRows']:.9g} "
        f"tangentCorrections="
        f"{diagnostics['genericTangentCorrections']:.9g} "
        f"tangentImpulse={diagnostics['genericTangentImpulse']:.9g} "
        f"restitutionCorrections="
        f"{diagnostics['restitutionCorrections']:.9g} "
        f"status={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(
            "[RIGID_FRICTION_TARGET_ERROR] "
            f"solver={spec.solver} execution={spec.execution} "
            f"case={spec.case_name} repeat={spec.repeat} error={error}"
        )
    if errors and combined:
        print(combined.rstrip())
    return not errors, gate, diagnostics


def validate_failure_first_authority(
    observed: dict[tuple[str, str, str, int], tuple[dict[str, str], dict[str, float]]],
    repeats: int,
    expected_owner: str,
) -> bool:
    errors: list[str] = []
    for execution in ("parallel", "sequential"):
        for reverse in (False, True):
            suffix = "-reverse" if reverse else ""
            tangent_mu0 = observed[
                ("avbd", execution, f"ownership-target-tangent-mu0{suffix}", 1)
            ][1]
            tangent_mu1 = observed[
                ("avbd", execution, f"ownership-target-tangent-mu1{suffix}", 1)
            ][1]
            combined = observed[
                (
                    "avbd",
                    execution,
                    f"ownership-target-combined-mu1-finite{suffix}",
                    1,
                )
            ][1]
            if tangent_mu0["positionAlTargetEvals"] != 0:
                errors.append(f"{execution}{suffix}: mu0 unexpectedly entered AL")
            if tangent_mu0["bodyStaticSweepTargetRows"] != 0:
                errors.append(f"{execution}{suffix}: mu0 unexpectedly entered sweep")
            if tangent_mu0["genericTangentRows"] <= 0:
                errors.append(f"{execution}{suffix}: mu0 generic tangent row missing")
            if expected_owner == "current-mixed":
                if tangent_mu1["positionAlTargetEvals"] <= 0:
                    errors.append(f"{execution}{suffix}: mu1 AL owner missing")
                if tangent_mu1["bodyStaticSweepTargetRows"] <= 0:
                    errors.append(f"{execution}{suffix}: mu1 sweep owner missing")
                if tangent_mu1["genericTangentRows"] <= 0:
                    errors.append(
                        f"{execution}{suffix}: mu1 generic owner missing"
                    )
            elif expected_owner == "position-owner":
                if tangent_mu1["positionAlTargetEvals"] <= 0:
                    errors.append(f"{execution}{suffix}: mu1 AL owner missing")
                if tangent_mu1["bodyStaticSweepTargetRows"] != 0:
                    errors.append(
                        f"{execution}{suffix}: position-owned mu1 entered sweep"
                    )
                if tangent_mu1["genericTangentRows"] != 0:
                    errors.append(
                        f"{execution}{suffix}: position-owned mu1 entered generic"
                    )
            else:
                if tangent_mu1["positionAlTargetEvals"] != 0:
                    errors.append(
                        f"{execution}{suffix}: velocity-owned mu1 entered AL"
                    )
                if tangent_mu1["bodyStaticSweepTargetRows"] != 0:
                    errors.append(
                        f"{execution}{suffix}: velocity-owned mu1 entered sweep"
                    )
                if tangent_mu1["genericTangentRows"] <= 0:
                    errors.append(
                        f"{execution}{suffix}: velocity owner missing"
                    )
                if tangent_mu1["genericNormalRows"] != 0:
                    errors.append(
                        f"{execution}{suffix}: pure tangent owner entered normal"
                    )
            if expected_owner == "finite-combined-owner":
                if combined["positionAlTargetEvals"] != 0:
                    errors.append(
                        f"{execution}{suffix}: finite target entered AL"
                    )
                if combined["bodyStaticSweepTargetRows"] != 0:
                    errors.append(
                        f"{execution}{suffix}: finite target entered sweep"
                    )
                if combined["genericNormalRows"] <= 0:
                    errors.append(
                        f"{execution}{suffix}: finite normal owner missing"
                    )
                if combined["genericTangentRows"] <= 0:
                    errors.append(
                        f"{execution}{suffix}: finite tangent owner missing"
                    )
            else:
                if combined["positionAlTargetEvals"] <= 0:
                    errors.append(
                        f"{execution}{suffix}: finite AL owner missing"
                    )
                if combined["bodyStaticSweepTargetRows"] <= 0:
                    errors.append(
                        f"{execution}{suffix}: finite sweep owner missing"
                    )
                if combined["genericTangentRows"] <= 0:
                    errors.append(
                        f"{execution}{suffix}: finite generic owner missing"
                    )

    for base_case in BASE_CASES:
        forward = observed[("tgs", "parallel", base_case, 1)][0]
        reverse = observed[
            ("tgs", "parallel", f"{base_case}-reverse", 1)
        ][0]
        forward_slot = (
            as_int(forward, "body0Actor0Count", errors) > 0,
            as_int(forward, "body0Actor1Count", errors) > 0,
        )
        reverse_slot = (
            as_int(reverse, "body0Actor0Count", errors) > 0,
            as_int(reverse, "body0Actor1Count", errors) > 0,
        )
        if forward_slot != reverse_slot:
            errors.append(
                f"{base_case}: body-static callback actor canonicalization changed"
            )
        if forward_slot != (True, False):
            errors.append(
                f"{base_case}: dynamic body was not canonicalized to actor0"
            )

    if expected_owner in ("velocity-owner", "finite-combined-owner"):
        for execution in ("parallel", "sequential"):
            for reverse in (False, True):
                suffix = "-reverse" if reverse else ""
                case_name = f"ownership-target-tangent-mu1{suffix}"
                tgs = observed[("tgs", "parallel", case_name, 1)][0]
                avbd_gate, avbd_diag = observed[
                    ("avbd", execution, case_name, 1)
                ]
                numeric_errors: list[str] = []
                tgs_final_x = as_float(
                    tgs, "finalBody0VelocityX", numeric_errors
                )
                avbd_final_x = as_float(
                    avbd_gate, "finalBody0VelocityX", numeric_errors
                )
                tgs_contact_x = as_float(
                    tgs, "finalBody0ContactVelocityX", numeric_errors
                )
                avbd_contact_x = as_float(
                    avbd_gate,
                    "finalBody0ContactVelocityX",
                    numeric_errors,
                )
                avbd_final_w = as_float(
                    avbd_gate, "finalBody0AngularSpeed", numeric_errors
                )
                body_mass = as_float(avbd_gate, "body0Mass", numeric_errors)
                expected_impulse = body_mass * 3.0
                errors.extend(
                    f"{execution}{suffix}: {error}"
                    for error in numeric_errors
                )
                if abs(avbd_final_x - 3.0) > 2.0e-3:
                    errors.append(
                        f"{execution}{suffix}: final vx={avbd_final_x:.9g}"
                    )
                if abs(avbd_contact_x - 3.0) > 2.0e-3:
                    errors.append(
                        f"{execution}{suffix}: final contact vx="
                        f"{avbd_contact_x:.9g}"
                    )
                if abs(avbd_final_x - tgs_final_x) > 2.0e-3:
                    errors.append(
                        f"{execution}{suffix}: final vx parity gap="
                        f"{abs(avbd_final_x - tgs_final_x):.9g}"
                    )
                if abs(avbd_contact_x - tgs_contact_x) > 2.0e-3:
                    errors.append(
                        f"{execution}{suffix}: contact vx parity gap="
                        f"{abs(avbd_contact_x - tgs_contact_x):.9g}"
                    )
                if avbd_final_w > 1.0e-4:
                    errors.append(
                        f"{execution}{suffix}: final angular="
                        f"{avbd_final_w:.9g}"
                    )
                if (
                    abs(
                        avbd_diag["genericTangentImpulse"]
                        - expected_impulse
                    )
                    > 5.0e-3
                ):
                    errors.append(
                        f"{execution}{suffix}: tangent impulse="
                        f"{avbd_diag['genericTangentImpulse']:.9g}, "
                        f"expected {expected_impulse:.9g}"
                    )

    deterministic_fields = (
        "peakAbsBody0VelocityX",
        "peakBody0VelocityY",
        "maxReportedImpulse",
        "finalBody0VelocityX",
        "finalBody0AngularSpeed",
        "finalBody0ContactVelocityX",
        "body0Actor0Count",
        "body0Actor1Count",
    )
    for solver in ("tgs", "avbd"):
        executions = ("parallel",) if solver == "tgs" else ("parallel", "sequential")
        for execution in executions:
            for base_case in BASE_CASES:
                for reverse in (False, True):
                    case_name = base_case + ("-reverse" if reverse else "")
                    first = observed[(solver, execution, case_name, 1)][0]
                    for repeat in range(2, repeats + 1):
                        current = observed[
                            (solver, execution, case_name, repeat)
                        ][0]
                        for field in deterministic_fields:
                            if current.get(field) != first.get(field):
                                errors.append(
                                    f"{solver}/{execution}/{case_name}: "
                                    f"{field} repeat mismatch"
                                )

    print(
        "[RIGID_FRICTION_TARGET_AUTHORITY] "
        f"expectedOwner={expected_owner} "
        f"status={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(f"[RIGID_FRICTION_TARGET_AUTHORITY_ERROR] {error}")
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--expect",
        choices=(
            "current-mixed",
            "position-owner",
            "velocity-owner",
            "finite-combined-owner",
        ),
        default="velocity-owner",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=BASE_CASES,
        dest="selected_cases",
        help="run only the selected base case; may be repeated",
    )
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[RIGID_FRICTION_TARGET_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0 or args.repeats <= 0:
        print(
            "[RIGID_FRICTION_TARGET_RUNNER_ERROR] "
            "--timeout and --repeats must be positive"
        )
        return 2

    specs: list[RunSpec] = []
    selected_cases = tuple(args.selected_cases or BASE_CASES)
    for repeat in range(1, args.repeats + 1):
        for base_case in selected_cases:
            for case_name in (base_case, f"{base_case}-reverse"):
                specs.append(RunSpec("tgs", "parallel", case_name, repeat))
                specs.append(RunSpec("avbd", "parallel", case_name, repeat))
                specs.append(RunSpec("avbd", "sequential", case_name, repeat))

    observed: dict[
        tuple[str, str, str, int], tuple[dict[str, str], dict[str, float]]
    ] = {}
    passed = True
    for spec in specs:
        run_passed, gate, diagnostics = run_one(
            spec, bin_dir, args.timeout
        )
        observed[
            (spec.solver, spec.execution, spec.case_name, spec.repeat)
        ] = (gate, diagnostics)
        passed = passed and run_passed
        if not run_passed:
            break

    expected_count = len(specs)
    if len(observed) != expected_count:
        passed = False
    elif selected_cases == BASE_CASES:
        if not validate_failure_first_authority(
            observed, args.repeats, args.expect
        ):
            passed = False
    else:
        print(
            "[RIGID_FRICTION_TARGET_AUTHORITY] "
            f"expectedOwner={args.expect} validation=SUBSET "
            "status=PASS"
        )

    print(
        "[RIGID_FRICTION_TARGET_MATRIX] "
        f"completed={len(observed)} expected={expected_count} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
