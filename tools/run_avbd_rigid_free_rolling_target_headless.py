#!/usr/bin/env python3
"""Gate the free-rolling single-contact tangential target response headlessly."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from run_avbd_rigid_friction_target_ownership_headless import (
    DEFAULT_BIN_DIR,
    EXECUTABLE,
    RunSpec,
    as_float,
    as_int,
    run_one,
)


BASE_CASE = "ownership-target-tangent-mu1-free"
RADIUS = 0.5
TARGET_SPEED = 3.0


def expected_response(gate: dict[str, str]) -> tuple[float, float, float]:
    errors: list[str] = []
    mass = as_float(gate, "body0Mass", errors)
    inertia = as_float(gate, "body0InertiaZ", errors)
    if errors or mass <= 0.0 or inertia <= 0.0:
        raise ValueError("invalid sphere mass/inertia readback")
    response = 1.0 / mass + RADIUS * RADIUS / inertia
    impulse = TARGET_SPEED / response
    return impulse / mass, RADIUS * impulse / inertia, impulse


def numeric_metrics(
    gate: dict[str, str], diagnostics: dict[str, float]
) -> tuple[float, float, float, float]:
    errors: list[str] = []
    vx = as_float(gate, "finalBody0VelocityX", errors)
    wz = as_float(gate, "finalBody0AngularZ", errors)
    contact_vx = as_float(gate, "finalBody0ContactVelocityX", errors)
    impulse = diagnostics.get("genericTangentImpulse", 0.0)
    if errors:
        raise ValueError("; ".join(errors))
    return vx, wz, contact_vx, impulse


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
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[RIGID_FREE_ROLLING_TARGET_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0.0 or args.repeats <= 0:
        print(
            "[RIGID_FREE_ROLLING_TARGET_RUNNER_ERROR] "
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
    errors: list[str] = []
    for spec in specs:
        passed, gate, diagnostics = run_one(spec, bin_dir, args.timeout)
        if not passed:
            errors.append(
                f"{spec.solver}/{spec.execution}/{spec.case_name}/"
                f"r{spec.repeat}: infrastructure or Snippet gate failed"
            )
            break
        observed[
            (spec.solver, spec.execution, spec.case_name, spec.repeat)
        ] = (gate, diagnostics)

    if len(observed) == len(specs):
        for reverse in (False, True):
            case_name = BASE_CASE + ("-reverse" if reverse else "")
            tgs_gate, _ = observed[("tgs", "parallel", case_name, 1)]
            expected_vx, expected_wz, expected_impulse = expected_response(
                tgs_gate
            )
            tgs_vx, tgs_wz, tgs_contact_vx, _ = numeric_metrics(
                tgs_gate, {}
            )
            if abs(tgs_vx - expected_vx) > 5.0e-3:
                errors.append(
                    f"{case_name}: TGS vx={tgs_vx:.9g}, "
                    f"analytic={expected_vx:.9g}"
                )
            if abs(tgs_wz - expected_wz) > 5.0e-3:
                errors.append(
                    f"{case_name}: TGS wz={tgs_wz:.9g}, "
                    f"analytic={expected_wz:.9g}"
                )
            if abs(tgs_contact_vx - TARGET_SPEED) > 5.0e-3:
                errors.append(
                    f"{case_name}: TGS contact vx={tgs_contact_vx:.9g}"
                )

            for execution in ("parallel", "sequential"):
                gate, diagnostics = observed[
                    ("avbd", execution, case_name, 1)
                ]
                vx, wz, contact_vx, impulse = numeric_metrics(
                    gate, diagnostics
                )
                label = f"{execution}/{'reverse' if reverse else 'forward'}"
                if diagnostics["positionAlTargetEvals"] != 0.0:
                    errors.append(f"{label}: target entered position AL")
                if diagnostics["bodyStaticSweepTargetRows"] != 0.0:
                    errors.append(f"{label}: target entered friction sweep")
                if diagnostics["genericNormalRows"] != 0.0:
                    errors.append(f"{label}: target entered generic normal")
                if diagnostics["genericTangentRows"] <= 0.0:
                    errors.append(f"{label}: tangent velocity owner missing")
                if abs(contact_vx - TARGET_SPEED) > 0.1:
                    errors.append(
                        f"{label}: contact target not held, vx={contact_vx:.9g}"
                    )

                decomposition_gap = max(
                    abs(vx - expected_vx), abs(wz - expected_wz)
                )
                if args.mode == "baseline":
                    if decomposition_gap <= 0.1:
                        errors.append(
                            f"{label}: expected baseline spatial failure "
                            "was not reproduced"
                        )
                else:
                    if abs(vx - expected_vx) > 1.0e-2:
                        errors.append(
                            f"{label}: vx={vx:.9g}, "
                            f"analytic={expected_vx:.9g}"
                        )
                    if abs(wz - expected_wz) > 1.0e-2:
                        errors.append(
                            f"{label}: wz={wz:.9g}, "
                            f"analytic={expected_wz:.9g}"
                        )
                    if abs(impulse - expected_impulse) > 2.0e-2:
                        errors.append(
                            f"{label}: accumulated impulse={impulse:.9g}, "
                            f"analytic={expected_impulse:.9g}"
                        )

                actor0 = as_int(gate, "body0Actor0Count", errors)
                actor1 = as_int(gate, "body0Actor1Count", errors)
                if actor0 <= 0 or actor1 != 0:
                    errors.append(
                        f"{label}: body-static callback was not "
                        "canonicalized to actor0"
                    )

        repeat_fields = (
            "finalBody0VelocityX",
            "finalBody0AngularZ",
            "finalBody0ContactVelocityX",
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
                        gate, diagnostics = observed[
                            (solver, execution, case_name, repeat)
                        ]
                        for field in repeat_fields:
                            if gate.get(field) != first_gate.get(field):
                                errors.append(
                                    f"{solver}/{execution}/{case_name}: "
                                    f"{field} repeat mismatch"
                                )
                        if diagnostics != first_diag:
                            errors.append(
                                f"{solver}/{execution}/{case_name}: "
                                "diagnostic repeat mismatch"
                            )

    expected = len(specs)
    status = "PASS" if not errors and len(observed) == expected else "FAIL"
    print(
        "[RIGID_FREE_ROLLING_TARGET_MATRIX] "
        f"mode={args.mode} completed={len(observed)} expected={expected} "
        f"status={status}"
    )
    for error in errors:
        print(f"[RIGID_FREE_ROLLING_TARGET_ERROR] {error}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
