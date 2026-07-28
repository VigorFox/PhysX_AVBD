#!/usr/bin/env python3
"""Freeze multi-row rigid-static friction-target ownership evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_avbd_rigid_friction_target_ownership_headless import (
    DEFAULT_BIN_DIR,
    RunSpec,
    as_float,
    as_int,
    run_one,
)


BASE_CASES = (
    "ownership-target-tangent-mu1-manifold",
    "ownership-target-tangent-mu1-manifold-yaw",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("baseline", "acceptance"),
        default="baseline",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    if args.timeout <= 0.0 or args.repeats <= 0:
        print(
            "[RIGID_FRICTION_MANIFOLD_TARGET_RUNNER_ERROR] "
            "--timeout and --repeats must be positive"
        )
        return 2

    specs = [
        RunSpec(solver, execution, case_name, repeat)
        for repeat in range(1, args.repeats + 1)
        for base_case in BASE_CASES
        for case_name in (base_case, f"{base_case}-reverse")
        for solver, execution in (
            ("tgs", "parallel"),
            ("avbd", "parallel"),
            ("avbd", "sequential"),
        )
    ]
    observed: dict[
        tuple[str, str, str, int],
        tuple[dict[str, str], dict[str, float]],
    ] = {}
    passed = True
    for spec in specs:
        ok, gate, diagnostics = run_one(spec, bin_dir, args.timeout)
        observed[
            (spec.solver, spec.execution, spec.case_name, spec.repeat)
        ] = (gate, diagnostics)
        passed = passed and ok
        if not ok:
            break

    errors: list[str] = []
    deterministic_fields = (
        "modifyCallbackCount",
        "modifiedPointCount",
        "body0Actor0Count",
        "body0Actor1Count",
        "peakAbsBody0VelocityX",
        "peakBody0VelocityY",
        "peakBody0AngularSpeed",
        "peakBody0AngularFrame",
        "finalBody0Speed",
        "finalBody0AngularSpeed",
        "finalBody0VelocityX",
        "finalBody0AngularX",
        "finalBody0AngularY",
        "finalBody0AngularZ",
        "finalBody0ContactVelocityX",
        "maxReportedImpulse",
    )
    if len(observed) == len(specs):
        for solver in ("tgs", "avbd"):
            executions = (
                ("parallel",)
                if solver == "tgs"
                else ("parallel", "sequential")
            )
            for execution in executions:
                for base_case in BASE_CASES:
                    for reverse in (False, True):
                        case_name = base_case + (
                            "-reverse" if reverse else ""
                        )
                        gate, diag = observed[
                            (solver, execution, case_name, 1)
                        ]
                        prefix = f"{solver}/{execution}/{case_name}"
                        numeric_errors: list[str] = []
                        callbacks = as_int(
                            gate, "modifyCallbackCount", numeric_errors
                        )
                        points = as_int(
                            gate, "modifiedPointCount", numeric_errors
                        )
                        peak_w = as_float(
                            gate, "peakBody0AngularSpeed", numeric_errors
                        )
                        final_w = as_float(
                            gate, "finalBody0AngularSpeed", numeric_errors
                        )
                        final_vx = as_float(
                            gate, "finalBody0VelocityX", numeric_errors
                        )
                        contact_vx = as_float(
                            gate,
                            "finalBody0ContactVelocityX",
                            numeric_errors,
                        )
                        body_mass = as_float(
                            gate, "body0Mass", numeric_errors
                        )
                        errors.extend(
                            f"{prefix}: {error}"
                            for error in numeric_errors
                        )
                        if callbacks <= 0 or points < callbacks * 2:
                            errors.append(
                                f"{prefix}: manifold evidence "
                                f"callbacks={callbacks} points={points}"
                            )
                        if points != callbacks * 4:
                            errors.append(
                                f"{prefix}: expected four-point fixture "
                                f"callbacks={callbacks} points={points}"
                            )
                        if solver == "avbd":
                            if args.mode == "baseline":
                                for field in (
                                    "positionAlTargetEvals",
                                    "bodyStaticSweepTargetRows",
                                    "genericTangentRows",
                                ):
                                    if diag[field] <= 0.0:
                                        errors.append(
                                            f"{prefix}: mixed baseline "
                                            f"missing {field}"
                                        )
                            else:
                                if diag["positionAlTargetEvals"] != 0.0:
                                    errors.append(
                                        f"{prefix}: target entered AL"
                                    )
                                if diag["bodyStaticSweepTargetRows"] != 0.0:
                                    errors.append(
                                        f"{prefix}: target entered sweep"
                                    )
                                if diag["genericTangentRows"] <= 0.0:
                                    errors.append(
                                        f"{prefix}: manifold owner missing"
                                    )
                                if diag["genericTangentRows"] != float(points):
                                    errors.append(
                                        f"{prefix}: tangent row accounting "
                                        f"rows={diag['genericTangentRows']} "
                                        f"points={points}"
                                    )
                                if abs(final_vx - 3.0) > 2.0e-3:
                                    errors.append(
                                        f"{prefix}: final COM target "
                                        f"vx={final_vx}"
                                    )
                                if abs(contact_vx - 3.0) > 2.0e-3:
                                    errors.append(
                                        f"{prefix}: final contact target "
                                        f"vx={contact_vx}"
                                    )
                                if final_w > 1.0e-3:
                                    errors.append(
                                        f"{prefix}: final angular speed "
                                        f"{final_w}"
                                    )
                                expected_impulse = body_mass * 3.0
                                actual_impulse = diag[
                                    "genericTangentImpulse"
                                ]
                                if (
                                    abs(actual_impulse - expected_impulse)
                                    > 5.0e-3
                                ):
                                    errors.append(
                                        f"{prefix}: tangent impulse "
                                        f"actual={actual_impulse} "
                                        f"expected={expected_impulse}"
                                    )
                        print(
                            "[RIGID_FRICTION_MANIFOLD_TARGET_METRIC] "
                            f"solver={solver} execution={execution} "
                            f"case={case_name} callbacks={callbacks} "
                            f"points={points} peakW={peak_w:.9g} "
                            f"finalW={final_w:.9g} "
                            f"finalVx={final_vx:.9g} "
                            f"contactVx={contact_vx:.9g} "
                            f"al={diag['positionAlTargetEvals']:.9g} "
                            f"sweep={diag['bodyStaticSweepTargetRows']:.9g} "
                            f"tangent={diag['genericTangentRows']:.9g}"
                        )

                        for repeat in range(2, args.repeats + 1):
                            repeat_gate, repeat_diag = observed[
                                (
                                    solver,
                                    execution,
                                    case_name,
                                    repeat,
                                )
                            ]
                            for field in deterministic_fields:
                                if (
                                    repeat_gate.get(field)
                                    != gate.get(field)
                                ):
                                    errors.append(
                                        f"{prefix}: {field} "
                                        "repeat mismatch"
                                    )
                            if repeat_diag != diag:
                                errors.append(
                                    f"{prefix}: diagnostic repeat mismatch"
                                )

        for base_case in BASE_CASES:
            for reverse in (False, True):
                case_name = base_case + (
                    "-reverse" if reverse else ""
                )
                parallel_gate, parallel_diag = observed[
                    ("avbd", "parallel", case_name, 1)
                ]
                sequential_gate, sequential_diag = observed[
                    ("avbd", "sequential", case_name, 1)
                ]
                for field in deterministic_fields:
                    if (
                        parallel_gate.get(field)
                        != sequential_gate.get(field)
                    ):
                        errors.append(
                            f"{case_name}: {field} "
                            "parallel/sequential mismatch"
                        )
                if parallel_diag != sequential_diag:
                    errors.append(
                        f"{case_name}: diagnostic "
                        "parallel/sequential mismatch"
                    )

                if args.mode == "acceptance":
                    avbd_peak_w = as_float(
                        parallel_gate,
                        "peakBody0AngularSpeed",
                        errors,
                    )
                    tgs_gate = observed[
                        ("tgs", "parallel", case_name, 1)
                    ][0]
                    tgs_peak_w = as_float(
                        tgs_gate,
                        "peakBody0AngularSpeed",
                        errors,
                    )
                    if avbd_peak_w > tgs_peak_w + 0.1:
                        errors.append(
                            f"{case_name}: AVBD peak angular gap "
                            f"avbd={avbd_peak_w} tgs={tgs_peak_w}"
                        )

        if args.mode == "acceptance":
            reference_gate = observed[
                ("avbd", "parallel", BASE_CASES[0], 1)
            ][0]
            yaw_gate = observed[
                ("avbd", "parallel", BASE_CASES[1], 1)
            ][0]
            reference_peak_w = as_float(
                reference_gate, "peakBody0AngularSpeed", errors
            )
            yaw_peak_w = as_float(
                yaw_gate, "peakBody0AngularSpeed", errors
            )
            if abs(reference_peak_w - yaw_peak_w) > 1.0e-2:
                errors.append(
                    "avbd manifold yaw-order peak angular mismatch "
                    f"reference={reference_peak_w} yaw={yaw_peak_w}"
                )

        equivalent_fields = (
            "peakAbsBody0VelocityX",
            "peakBody0VelocityY",
            "peakBody0AngularSpeed",
            "finalBody0Speed",
            "finalBody0AngularSpeed",
            "finalBody0VelocityX",
            "finalBody0ContactVelocityX",
            "maxReportedImpulse",
        )
        for solver in ("tgs", "avbd"):
            executions = (
                ("parallel",)
                if solver == "tgs"
                else ("parallel", "sequential")
            )
            for execution in executions:
                reference = observed[
                    (
                        solver,
                        execution,
                        BASE_CASES[0],
                        1,
                    )
                ][0]
                for base_case in BASE_CASES:
                    for reverse in (False, True):
                        case_name = base_case + (
                            "-reverse" if reverse else ""
                        )
                        gate = observed[
                            (solver, execution, case_name, 1)
                        ][0]
                        for field in equivalent_fields:
                            if gate.get(field) != reference.get(field):
                                print(
                                    "[RIGID_FRICTION_MANIFOLD_TARGET_ORDER] "
                                    f"solver={solver} "
                                    f"execution={execution} "
                                    f"case={case_name} field={field} "
                                    f"reference={reference.get(field)} "
                                    f"actual={gate.get(field)}"
                                )
    else:
        errors.append(
            f"incomplete matrix {len(observed)}/{len(specs)}"
        )

    passed = passed and not errors
    for error in errors:
        print(f"[RIGID_FRICTION_MANIFOLD_TARGET_ERROR] {error}")
    print(
        "[RIGID_FRICTION_MANIFOLD_TARGET_MATRIX] "
        f"mode={args.mode} completed={len(observed)} "
        f"expected={len(specs)} status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
