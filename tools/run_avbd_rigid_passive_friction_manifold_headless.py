#!/usr/bin/env python3
"""Audit passive multi-row rigid-static friction ownership headlessly."""

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
    "ownership-passive-friction-mu1-manifold",
    "ownership-passive-friction-mu1-manifold-yaw",
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
            "[RIGID_PASSIVE_FRICTION_MANIFOLD_RUNNER_ERROR] "
            "--timeout and --repeats must be positive"
        )
        return 2

    bin_dir = args.bin_dir.resolve()
    artifact_dir = (
        Path(tempfile.gettempdir())
        / (
            "PhysX_AVBD_passive_friction_manifold_"
            + datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        )
    )
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
                        gate, diagnostics = observed[
                            (solver, execution, case_name, 1)
                        ]
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
                        final_speed = as_float(
                            gate, "finalBody0Speed", numeric_errors
                        )
                        final_w = as_float(
                            gate, "finalBody0AngularSpeed", numeric_errors
                        )
                        final_vx = as_float(
                            gate, "finalBody0VelocityX", numeric_errors
                        )
                        final_contact_vx = as_float(
                            gate,
                            "finalBody0ContactVelocityX",
                            numeric_errors,
                        )
                        errors.extend(
                            f"{solver}/{execution}/{case_name}: {error}"
                            for error in numeric_errors
                        )
                        if callbacks <= 0 or points < callbacks * 2:
                            errors.append(
                                f"{solver}/{execution}/{case_name}: "
                                "multi-row manifold missing "
                                f"callbacks={callbacks} points={points}"
                            )
                        if args.mode == "acceptance":
                            if final_speed > 5.0e-3:
                                errors.append(
                                    f"{solver}/{execution}/{case_name}: "
                                    f"passive friction did not settle "
                                    f"speed={final_speed}"
                                )
                            if final_w > 5.0e-3:
                                errors.append(
                                    f"{solver}/{execution}/{case_name}: "
                                    f"residual angular speed={final_w}"
                                )
                        print(
                            "[RIGID_PASSIVE_FRICTION_MANIFOLD_METRIC] "
                            f"solver={solver} execution={execution} "
                            f"case={case_name} callbacks={callbacks} "
                            f"points={points} peakW={peak_w:.9g} "
                            f"finalSpeed={final_speed:.9g} "
                            f"finalW={final_w:.9g} "
                            f"finalVx={final_vx:.9g} "
                            f"finalContactVx={final_contact_vx:.9g} "
                            f"targetAl="
                            f"{diagnostics['positionAlTargetEvals']:.9g} "
                            f"targetSweep="
                            f"{diagnostics['bodyStaticSweepTargetRows']:.9g} "
                            f"targetProjection="
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
                            manifold_rows = diagnostics[
                                "genericTangentRows"
                            ]
                            if args.mode == "inventory":
                                if fallback_rows <= 0:
                                    errors.append(
                                        f"{solver}/{execution}/{case_name}: "
                                        "ordinary rigid-static friction "
                                        "fallback was not observed"
                                    )
                            else:
                                if fallback_rows != 0:
                                    errors.append(
                                        f"{solver}/{execution}/{case_name}: "
                                        "ordinary rigid-static friction "
                                        f"replayed fallback rows={fallback_rows}"
                                    )
                                if manifold_rows != float(points):
                                    errors.append(
                                        f"{solver}/{execution}/{case_name}: "
                                        "complete manifold row accounting "
                                        f"rows={manifold_rows} points={points}"
                                    )
                                if (
                                    diagnostics["objectiveManifoldRows"]
                                    <= 0.0 or
                                    diagnostics["objectiveManifoldRows"] !=
                                    diagnostics["objectiveRows"]
                                ):
                                    errors.append(
                                        f"{solver}/{execution}/{case_name}: "
                                        "compiled ManifoldFinalize partition "
                                        f"manifold="
                                        f"{diagnostics['objectiveManifoldRows']:.9g} "
                                        f"rows="
                                        f"{diagnostics['objectiveRows']:.9g}"
                                    )

        comparison_fields = (
            "peakBody0AngularSpeed",
            "finalBody0Speed",
            "finalBody0AngularSpeed",
            "finalBody0VelocityX",
            "finalBody0AngularX",
            "finalBody0AngularY",
            "finalBody0AngularZ",
            "finalBody0ContactVelocityX",
        )

        def compare(
            label: str,
            lhs: dict[str, str],
            rhs: dict[str, str],
            tolerance: float,
        ) -> None:
            max_delta = 0.0
            max_field = "none"
            for field in comparison_fields:
                try:
                    delta = abs(float(lhs[field]) - float(rhs[field]))
                except (KeyError, ValueError):
                    errors.append(f"{label}: missing/non-numeric {field}")
                    continue
                if delta > max_delta:
                    max_delta = delta
                    max_field = field
            print(
                "[RIGID_PASSIVE_FRICTION_MANIFOLD_DELTA] "
                f"comparison={label} maxField={max_field} "
                f"maxDelta={max_delta:.9g}"
            )
            if args.mode == "acceptance" and max_delta > tolerance:
                errors.append(
                    f"{label}: max {max_field} delta "
                    f"{max_delta} > {tolerance}"
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
                reverse = observed[
                    (
                        solver,
                        execution,
                        f"{BASE_CASES[0]}-reverse",
                        1,
                    )
                ][0]
                yaw = observed[
                    (
                        solver,
                        execution,
                        BASE_CASES[1],
                        1,
                    )
                ][0]
                yaw_reverse = observed[
                    (
                        solver,
                        execution,
                        f"{BASE_CASES[1]}-reverse",
                        1,
                    )
                ][0]
                compare(
                    f"{solver}-{execution}-actor-order",
                    reference,
                    reverse,
                    1.0e-6,
                )
                compare(
                    f"{solver}-{execution}-yaw-equivalence",
                    reference,
                    yaw,
                    0.15 if solver == "tgs" else 1.0e-4,
                )
                compare(
                    f"{solver}-{execution}-yaw-actor-order",
                    yaw,
                    yaw_reverse,
                    1.0e-6,
                )
                if solver == "avbd":
                    reference_diag = observed[
                        (solver, execution, BASE_CASES[0], 1)
                    ][1]
                    reverse_diag = observed[
                        (
                            solver,
                            execution,
                            f"{BASE_CASES[0]}-reverse",
                            1,
                        )
                    ][1]
                    yaw_diag = observed[
                        (solver, execution, BASE_CASES[1], 1)
                    ][1]
                    yaw_reverse_diag = observed[
                        (
                            solver,
                            execution,
                            f"{BASE_CASES[1]}-reverse",
                            1,
                        )
                    ][1]
                    if (
                        reference_diag["objectiveFingerprint"] !=
                        reverse_diag["objectiveFingerprint"]
                    ):
                        errors.append(
                            f"{solver}-{execution}: compiled-objective "
                            "actor-order fingerprint mismatch"
                        )
                    if (
                        yaw_diag["objectiveFingerprint"] !=
                        yaw_reverse_diag["objectiveFingerprint"]
                    ):
                        errors.append(
                            f"{solver}-{execution}: yaw "
                            "compiled-objective actor-order "
                            "fingerprint mismatch"
                        )

        for case_name in (
            BASE_CASES[0],
            f"{BASE_CASES[0]}-reverse",
            BASE_CASES[1],
            f"{BASE_CASES[1]}-reverse",
        ):
            parallel = observed[
                ("avbd", "parallel", case_name, 1)
            ][0]
            sequential = observed[
                ("avbd", "sequential", case_name, 1)
            ][0]
            compare(
                f"avbd-execution-{case_name}",
                parallel,
                sequential,
                1.0e-6,
            )
            tgs = observed[
                ("tgs", "parallel", case_name, 1)
            ][0]
            avbd_peak_w = as_float(
                parallel, "peakBody0AngularSpeed", errors
            )
            tgs_peak_w = as_float(
                tgs, "peakBody0AngularSpeed", errors
            )
            peak_gap = avbd_peak_w - tgs_peak_w
            print(
                "[RIGID_PASSIVE_FRICTION_MANIFOLD_DELTA] "
                f"comparison=avbd-vs-tgs-{case_name} "
                f"maxField=peakBody0AngularSpeed "
                f"maxDelta={peak_gap:.9g}"
            )
            if args.mode == "acceptance" and peak_gap > 0.1:
                errors.append(
                    f"avbd-vs-tgs-{case_name}: peak angular gap "
                    f"{peak_gap} > 0.1"
                )

    for error in errors:
        print(f"[RIGID_PASSIVE_FRICTION_MANIFOLD_ERROR] {error}")
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
        "[RIGID_PASSIVE_FRICTION_MANIFOLD_RESULT] "
        f"mode={args.mode} total={len(observed)}/{len(specs)} "
        f"status={'PASS' if passed else 'FAIL'} "
        f"artifact={artifact_dir}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
