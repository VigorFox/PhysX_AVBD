#!/usr/bin/env python3
"""Gate the strict finite combined normal/tangent target ownership slice."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from run_avbd_rigid_friction_target_ownership_headless import (
    DEFAULT_BIN_DIR,
    RunSpec,
    as_float,
    run_one,
)


BASE_CASE = "ownership-target-combined-mu1-finite"


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
            "[RIGID_FINITE_COMBINED_TARGET_RUNNER_ERROR] "
            "--timeout and --repeats must be positive"
        )
        return 2

    specs = [
        RunSpec(solver, execution, case_name, repeat)
        for repeat in range(1, args.repeats + 1)
        for case_name in (BASE_CASE, f"{BASE_CASE}-reverse")
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
    if len(observed) == len(specs):
        for execution in ("parallel", "sequential"):
            for reverse in (False, True):
                case_name = BASE_CASE + ("-reverse" if reverse else "")
                gate, diag = observed[
                    ("avbd", execution, case_name, 1)
                ]
                prefix = f"{execution}/{'reverse' if reverse else 'forward'}"
                if args.mode == "baseline":
                    for field in (
                        "positionAlTargetEvals",
                        "bodyStaticSweepTargetRows",
                        "genericNormalRows",
                        "genericTangentRows",
                    ):
                        if diag[field] <= 0.0:
                            errors.append(
                                f"{prefix}: mixed baseline missing {field}"
                            )
                else:
                    if diag["positionAlTargetEvals"] != 0.0:
                        errors.append(
                            f"{prefix}: target still entered position AL"
                        )
                    if diag["bodyStaticSweepTargetRows"] != 0.0:
                        errors.append(
                            f"{prefix}: target still entered friction sweep"
                        )
                    if diag["genericNormalRows"] <= 0.0:
                        errors.append(
                            f"{prefix}: finite normal velocity owner missing"
                        )
                    if diag["genericTangentRows"] <= 0.0:
                        errors.append(
                            f"{prefix}: finite tangent velocity owner missing"
                        )

                numeric_errors: list[str] = []
                reported = as_float(
                    gate, "maxReportedImpulse", numeric_errors
                )
                final_w = as_float(
                    gate, "finalBody0AngularSpeed", numeric_errors
                )
                peak_vx = as_float(
                    gate, "peakAbsBody0VelocityX", numeric_errors
                )
                final_vx = as_float(
                    gate, "finalBody0VelocityX", numeric_errors
                )
                final_contact_vx = as_float(
                    gate, "finalBody0ContactVelocityX", numeric_errors
                )
                peak_vy = as_float(
                    gate, "peakBody0VelocityY", numeric_errors
                )
                final_speed = as_float(
                    gate, "finalBody0Speed", numeric_errors
                )
                body_mass = as_float(
                    gate, "body0Mass", numeric_errors
                )
                errors.extend(
                    f"{prefix}: {error}" for error in numeric_errors
                )
                values = (
                    reported,
                    final_w,
                    peak_vx,
                    final_vx,
                    final_contact_vx,
                    peak_vy,
                    final_speed,
                    body_mass,
                )
                if not all(math.isfinite(value) for value in values):
                    errors.append(f"{prefix}: non-finite physical metric")
                if abs(reported - 0.25) > 2.0e-4:
                    errors.append(
                        f"{prefix}: reported max impulse={reported:.9g}"
                    )
                if final_w > 1.0e-5:
                    errors.append(
                        f"{prefix}: angular lock drift={final_w:.9g}"
                    )
                if args.mode == "acceptance":
                    if (
                        diag["genericNormalImpulseFrameMax"]
                        > 0.2502
                    ):
                        errors.append(
                            f"{prefix}: per-frame normal impulse="
                            f"{diag['genericNormalImpulseFrameMax']:.9g}"
                        )
                    if (
                        diag["genericTangentImpulseFrameMax"]
                        > 0.2502
                    ):
                        errors.append(
                            f"{prefix}: per-frame tangent impulse="
                            f"{diag['genericTangentImpulseFrameMax']:.9g}"
                        )
                    if abs(final_vx - 3.0) > 2.0e-3:
                        errors.append(
                            f"{prefix}: final vx={final_vx:.9g}"
                        )
                    if abs(final_contact_vx - 3.0) > 2.0e-3:
                        errors.append(
                            f"{prefix}: final contact vx="
                            f"{final_contact_vx:.9g}"
                        )
                    expected_tangent_impulse = body_mass * 3.0
                    if (
                        abs(
                            diag["genericTangentImpulse"]
                            - expected_tangent_impulse
                        )
                        > 5.0e-3
                    ):
                        errors.append(
                            f"{prefix}: tangent impulse="
                            f"{diag['genericTangentImpulse']:.9g}, "
                            f"expected {expected_tangent_impulse:.9g}"
                        )
                    if (
                        diag["objectivePointRows"] <= 0.0 or
                        diag["objectivePointRows"] !=
                        diag["objectiveRows"]
                    ):
                        errors.append(
                            f"{prefix}: compiled PointFinalize partition "
                            f"point={diag['objectivePointRows']:.9g} "
                            f"rows={diag['objectiveRows']:.9g}"
                        )
                    tgs_gate = observed[
                        ("tgs", "parallel", case_name, 1)
                    ][0]
                    tgs_peak_vy = as_float(
                        tgs_gate, "peakBody0VelocityY", errors
                    )
                    if abs(peak_vy - tgs_peak_vy) > 0.1:
                        errors.append(
                            f"{prefix}: peak vy parity gap="
                            f"{abs(peak_vy - tgs_peak_vy):.9g}"
                        )

        deterministic_fields = (
            "peakAbsBody0VelocityX",
            "peakBody0VelocityY",
            "peakBody0AngularSpeed",
            "finalBody0Speed",
            "finalBody0VelocityX",
            "finalBody0AngularSpeed",
            "finalBody0ContactVelocityX",
            "maxReportedImpulse",
            "body0Actor0Count",
            "body0Actor1Count",
        )
        for solver in ("tgs", "avbd"):
            executions = (
                ("parallel",)
                if solver == "tgs"
                else ("parallel", "sequential")
            )
            for execution in executions:
                for reverse in (False, True):
                    case_name = BASE_CASE + (
                        "-reverse" if reverse else ""
                    )
                    first_gate, first_diag = observed[
                        (solver, execution, case_name, 1)
                    ]
                    for repeat in range(2, args.repeats + 1):
                        gate, diag = observed[
                            (solver, execution, case_name, repeat)
                        ]
                        for field in deterministic_fields:
                            if gate.get(field) != first_gate.get(field):
                                errors.append(
                                    f"{solver}/{execution}/{case_name}: "
                                    f"{field} repeat mismatch"
                                )
                        if diag != first_diag:
                            errors.append(
                                f"{solver}/{execution}/{case_name}: "
                                "diagnostic repeat mismatch"
                            )

        for reverse in (False, True):
            case_name = BASE_CASE + ("-reverse" if reverse else "")
            parallel_gate, parallel_diag = observed[
                ("avbd", "parallel", case_name, 1)
            ]
            sequential_gate, sequential_diag = observed[
                ("avbd", "sequential", case_name, 1)
            ]
            for field in deterministic_fields:
                if parallel_gate.get(field) != sequential_gate.get(field):
                    errors.append(
                        f"{case_name}: {field} "
                        "parallel/sequential mismatch"
                    )
            if parallel_diag != sequential_diag:
                errors.append(
                    f"{case_name}: diagnostic parallel/sequential mismatch"
                )
        for execution in ("parallel", "sequential"):
            forward_diag = observed[
                ("avbd", execution, BASE_CASE, 1)
            ][1]
            reverse_diag = observed[
                ("avbd", execution, f"{BASE_CASE}-reverse", 1)
            ][1]
            if (
                forward_diag["objectiveFingerprint"] !=
                reverse_diag["objectiveFingerprint"]
            ):
                errors.append(
                    f"{execution}: compiled-objective actor-order "
                    "fingerprint mismatch"
                )
    else:
        errors.append(
            f"incomplete matrix {len(observed)}/{len(specs)}"
        )

    passed = passed and not errors
    for error in errors:
        print(f"[RIGID_FINITE_COMBINED_TARGET_ERROR] {error}")
    print(
        "[RIGID_FINITE_COMBINED_TARGET_MATRIX] "
        f"mode={args.mode} completed={len(observed)} "
        f"expected={len(specs)} status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
