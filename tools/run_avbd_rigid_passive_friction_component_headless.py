#!/usr/bin/env python3
"""Audit passive friction ownership across a connected rigid contact component."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import tempfile

from run_avbd_rigid_friction_target_ownership_headless import (
    DEFAULT_BIN_DIR,
    RunSpec,
    as_float,
    as_int,
    run_one,
)


BASE_CASES = (
    "ownership-passive-friction-component",
    "ownership-passive-friction-component-yaw",
)
SOLVER_EXECUTIONS = (
    ("tgs", "parallel"),
    ("avbd", "parallel"),
    ("avbd", "sequential"),
)
COMPARISON_FIELDS = (
    "peakBody0AngularSpeed",
    "peakBody1AngularSpeed",
    "finalBody0Speed",
    "finalBody1Speed",
    "finalBody0AngularSpeed",
    "finalBody1AngularSpeed",
    "finalBody0VelocityX",
    "finalBody1VelocityX",
    "finalBody0AngularX",
    "finalBody0AngularY",
    "finalBody0AngularZ",
    "finalBody1AngularX",
    "finalBody1AngularY",
    "finalBody1AngularZ",
    "minBody0Y",
    "minBody1Y",
)


def compare(
    label: str,
    lhs: dict[str, str],
    rhs: dict[str, str],
    errors: list[str],
    *,
    enforce: bool,
    tolerance: float,
) -> None:
    max_delta = 0.0
    max_field = "none"
    for field in COMPARISON_FIELDS:
        try:
            delta = abs(float(lhs[field]) - float(rhs[field]))
        except (KeyError, ValueError):
            errors.append(f"{label}: missing/non-numeric {field}")
            continue
        if delta > max_delta:
            max_delta = delta
            max_field = field
    print(
        "[RIGID_PASSIVE_FRICTION_COMPONENT_DELTA] "
        f"comparison={label} maxField={max_field} "
        f"maxDelta={max_delta:.9g}"
    )
    if enforce and max_delta > tolerance:
        errors.append(
            f"{label}: max {max_field} delta "
            f"{max_delta} > {tolerance}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inventory", "acceptance"),
        default="inventory",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    if args.timeout <= 0.0 or args.repeats <= 0:
        print(
            "[RIGID_PASSIVE_FRICTION_COMPONENT_RUNNER_ERROR] "
            "--timeout and --repeats must be positive"
        )
        return 2

    bin_dir = args.bin_dir.resolve()
    artifact_dir = (
        Path(tempfile.gettempdir())
        / (
            "PhysX_AVBD_passive_friction_component_"
            + datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        )
    )
    specs = [
        RunSpec(solver, execution, case_name, repeat)
        for repeat in range(1, args.repeats + 1)
        for base_case in BASE_CASES
        for case_name in (base_case, f"{base_case}-reverse")
        for solver, execution in SOLVER_EXECUTIONS
    ]
    observed: dict[
        tuple[str, str, str, int],
        tuple[dict[str, str], dict[str, float]],
    ] = {}
    infrastructure_ok = True
    for spec in specs:
        ok, gate, diagnostics = run_one(
            spec, bin_dir, args.timeout, artifact_dir
        )
        observed[
            (spec.solver, spec.execution, spec.case_name, spec.repeat)
        ] = (gate, diagnostics)
        infrastructure_ok = infrastructure_ok and ok
        if not ok:
            break

    errors: list[str] = []
    if len(observed) == len(specs):
        for repeat in range(1, args.repeats + 1):
            for solver, execution in SOLVER_EXECUTIONS:
                for base_case in BASE_CASES:
                    for reverse in (False, True):
                        case_name = base_case + (
                            "-reverse" if reverse else ""
                        )
                        gate, diagnostics = observed[
                            (solver, execution, case_name, repeat)
                        ]
                        numeric_errors: list[str] = []
                        callbacks = as_int(
                            gate, "modifyCallbackCount", numeric_errors
                        )
                        pairs = as_int(
                            gate, "modifiedPairCount", numeric_errors
                        )
                        points = as_int(
                            gate, "modifiedPointCount", numeric_errors
                        )
                        reported_pairs = as_int(
                            gate, "reportCallbackCount", numeric_errors
                        )
                        reported_points = as_int(
                            gate, "reportPointCount", numeric_errors
                        )
                        peak_w0 = as_float(
                            gate, "peakBody0AngularSpeed", numeric_errors
                        )
                        peak_w1 = as_float(
                            gate, "peakBody1AngularSpeed", numeric_errors
                        )
                        final_speed0 = as_float(
                            gate, "finalBody0Speed", numeric_errors
                        )
                        final_speed1 = as_float(
                            gate, "finalBody1Speed", numeric_errors
                        )
                        final_w0 = as_float(
                            gate, "finalBody0AngularSpeed", numeric_errors
                        )
                        final_w1 = as_float(
                            gate, "finalBody1AngularSpeed", numeric_errors
                        )
                        final_vx0 = as_float(
                            gate, "finalBody0VelocityX", numeric_errors
                        )
                        final_vx1 = as_float(
                            gate, "finalBody1VelocityX", numeric_errors
                        )
                        min_y0 = as_float(
                            gate, "minBody0Y", numeric_errors
                        )
                        min_y1 = as_float(
                            gate, "minBody1Y", numeric_errors
                        )
                        errors.extend(
                            f"{solver}/{execution}/{case_name}/"
                            f"repeat{repeat}: {error}"
                            for error in numeric_errors
                        )
                        if (
                            callbacks <= 0
                            or pairs != callbacks * 2
                            or points < pairs * 2
                            or reported_pairs != pairs
                            or reported_points != points
                        ):
                            errors.append(
                                f"{solver}/{execution}/{case_name}/"
                                f"repeat{repeat}: connected multi-row "
                                f"contacts missing callbacks={callbacks} "
                                f"pairs={pairs} points={points} "
                                f"reportedPairs={reported_pairs} "
                                f"reportedPoints={reported_points}"
                            )
                        if args.mode == "acceptance":
                            if max(final_speed0, final_speed1) > 0.1:
                                errors.append(
                                    f"{solver}/{execution}/{case_name}/"
                                    f"repeat{repeat}: passive component "
                                    "did not settle "
                                    f"speeds={final_speed0},{final_speed1}"
                                )
                            if max(final_w0, final_w1) > 0.05:
                                errors.append(
                                    f"{solver}/{execution}/{case_name}/"
                                    f"repeat{repeat}: residual angular "
                                    f"speeds={final_w0},{final_w1}"
                                )
                        print(
                            "[RIGID_PASSIVE_FRICTION_COMPONENT_METRIC] "
                            f"solver={solver} execution={execution} "
                            f"case={case_name} repeat={repeat} "
                            f"callbacks={callbacks} pairs={pairs} "
                            f"points={points} "
                            f"peakW0={peak_w0:.9g} peakW1={peak_w1:.9g} "
                            f"finalSpeed0={final_speed0:.9g} "
                            f"finalSpeed1={final_speed1:.9g} "
                            f"finalW0={final_w0:.9g} "
                            f"finalW1={final_w1:.9g} "
                            f"finalVx0={final_vx0:.9g} "
                            f"finalVx1={final_vx1:.9g} "
                            f"minY0={min_y0:.9g} minY1={min_y1:.9g} "
                            f"materialNormalRows="
                            f"{diagnostics['genericNormalRows']:.9g} "
                            f"materialTangentRows="
                            f"{diagnostics['genericTangentRows']:.9g} "
                            f"fallbackRows="
                            f"{diagnostics['bodyStaticFallbackRows']:.9g} "
                            f"fallbackCorrections="
                            f"{diagnostics['bodyStaticFallbackCorrections']:.9g} "
                            f"fallbackImpulse="
                            f"{diagnostics['bodyStaticFallbackImpulse']:.9g}"
                        )
                        if solver == "avbd":
                            fallback_rows = diagnostics[
                                "bodyStaticFallbackRows"
                            ]
                            material_tangent_rows = diagnostics[
                                "genericTangentRows"
                            ]
                            if args.mode == "inventory":
                                if fallback_rows <= 0:
                                    errors.append(
                                        f"{solver}/{execution}/{case_name}/"
                                        f"repeat{repeat}: legacy fallback "
                                        "was not observed"
                                    )
                                if material_tangent_rows != 0:
                                    errors.append(
                                        f"{solver}/{execution}/{case_name}/"
                                        f"repeat{repeat}: complete material "
                                        "owner unexpectedly accepted rows="
                                        f"{material_tangent_rows}"
                                    )
                            else:
                                if fallback_rows != 0:
                                    errors.append(
                                        f"{solver}/{execution}/{case_name}/"
                                        f"repeat{repeat}: legacy fallback "
                                        f"rows={fallback_rows}"
                                    )
                                if material_tangent_rows != float(points):
                                    errors.append(
                                        f"{solver}/{execution}/{case_name}/"
                                        f"repeat{repeat}: component row "
                                        "accounting "
                                        f"rows={material_tangent_rows} "
                                        f"points={points}"
                                    )

                reference = observed[
                    (
                        solver,
                        execution,
                        BASE_CASES[0],
                        repeat,
                    )
                ][0]
                reverse = observed[
                    (
                        solver,
                        execution,
                        f"{BASE_CASES[0]}-reverse",
                        repeat,
                    )
                ][0]
                yaw = observed[
                    (
                        solver,
                        execution,
                        BASE_CASES[1],
                        repeat,
                    )
                ][0]
                yaw_reverse = observed[
                    (
                        solver,
                        execution,
                        f"{BASE_CASES[1]}-reverse",
                        repeat,
                    )
                ][0]
                compare(
                    f"{solver}-{execution}-actor-order-repeat{repeat}",
                    reference,
                    reverse,
                    errors,
                    enforce=args.mode == "acceptance" and solver == "avbd",
                    # The material owner is velocity-only.  The frozen AVBD
                    # position solve retains a 2.77e-4 m actor-order delta in
                    # the upper body's minimum Y; velocity/owner metrics are
                    # otherwise substantially tighter.
                    tolerance=5.0e-4,
                )
                compare(
                    f"{solver}-{execution}-yaw-robustness-repeat{repeat}",
                    reference,
                    yaw,
                    errors,
                    enforce=False,
                    tolerance=0.0,
                )
                compare(
                    f"{solver}-{execution}-yaw-order-repeat{repeat}",
                    yaw,
                    yaw_reverse,
                    errors,
                    enforce=args.mode == "acceptance" and solver == "avbd",
                    tolerance=5.0e-4,
                )

            for case_name in (
                BASE_CASES[0],
                f"{BASE_CASES[0]}-reverse",
                BASE_CASES[1],
                f"{BASE_CASES[1]}-reverse",
            ):
                parallel = observed[
                    ("avbd", "parallel", case_name, repeat)
                ][0]
                sequential = observed[
                    ("avbd", "sequential", case_name, repeat)
                ][0]
                tgs = observed[
                    ("tgs", "parallel", case_name, repeat)
                ][0]
                compare(
                    f"avbd-execution-{case_name}-repeat{repeat}",
                    parallel,
                    sequential,
                    errors,
                    enforce=args.mode == "acceptance",
                    tolerance=1.0e-6,
                )
                compare(
                    f"avbd-vs-tgs-{case_name}-repeat{repeat}",
                    parallel,
                    tgs,
                    errors,
                    enforce=False,
                    tolerance=0.0,
                )

    for error in errors:
        print(f"[RIGID_PASSIVE_FRICTION_COMPONENT_ERROR] {error}")
    passed = infrastructure_ok and not errors
    summary = {
        "mode": args.mode,
        "passed": passed,
        "infrastructure_ok": infrastructure_ok,
        "expected_runs": len(specs),
        "observed_runs": len(observed),
        "errors": errors,
        "runs": [
            {
                "solver": solver,
                "execution": execution,
                "case": case_name,
                "repeat": repeat,
                "gate": gate,
                "diagnostics": diagnostics,
            }
            for (
                solver,
                execution,
                case_name,
                repeat,
            ), (gate, diagnostics) in observed.items()
        ],
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[RIGID_PASSIVE_FRICTION_COMPONENT_RESULT] "
        f"mode={args.mode} total={len(observed)}/{len(specs)} "
        f"status={'PASS' if passed else 'FAIL'} "
        f"artifact={artifact_dir}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
