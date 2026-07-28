#!/usr/bin/env python3
"""Run the dedicated moving-deformable-surface ownership matrix headlessly."""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import tempfile

from run_avbd_cross_snippets_headless import (
    DEFAULT_BIN_DIR,
    RunSpec,
    fields,
    parse_authority,
    run_one,
)


CASES = {
    "stack": (
        "moving-mesh-stack",
        7200,
        fields(
            simulateFailures="0",
            fetchFailures="0",
            fetchErrorState="0",
            cleanupCompleted="1",
            maxFullFallThroughBodies="0",
            settledSunkBoxes="0",
            stackMaxOutOfFootprintBoxes="0",
            softBody="0",
            cloth="0",
        ),
    ),
    "sphere": (
        "sphere-shot",
        180,
        fields(
            simulateFailures="0",
            fetchFailures="0",
            fetchErrorState="0",
            cleanupCompleted="1",
            maxFullFallThroughBodies="0",
            sphereFirstContactObserved="1",
            sphereOutOfFootprintFrames="0",
            softBody="0",
            cloth="0",
        ),
    ),
    "shell": (
        "stress-diagnostic",
        600,
        fields(
            simulateFailures="0",
            fetchFailures="0",
            fetchErrorState="0",
            cleanupCompleted="1",
            stressMetricsObserved="1",
            softBody="0",
            cloth="0",
        ),
    ),
    "mass-light": (
        "surface-owner-light",
        240,
        fields(
            simulateFailures="0",
            fetchFailures="0",
            fetchErrorState="0",
            cleanupCompleted="1",
            dynamicBodies="1",
            ownershipMetricsObserved="1",
            ownershipProbeMass="39",
            ownershipOutOfFootprintFrames="0",
            softBody="0",
            cloth="0",
        ),
    ),
    "mass-heavy": (
        "surface-owner-heavy",
        240,
        fields(
            simulateFailures="0",
            fetchFailures="0",
            fetchErrorState="0",
            cleanupCompleted="1",
            dynamicBodies="1",
            ownershipMetricsObserved="1",
            ownershipProbeMass="41",
            ownershipOutOfFootprintFrames="0",
            softBody="0",
            cloth="0",
        ),
    ),
    "shell-post": (
        "shell-post-ownership",
        240,
        fields(
            simulateFailures="0",
            fetchFailures="0",
            fetchErrorState="0",
            cleanupCompleted="1",
            dynamicBodies="1",
            ownershipMetricsObserved="1",
            ownershipProbeMass="10",
            ownershipOutOfFootprintFrames="0",
            softBody="0",
            cloth="0",
        ),
    ),
    "history": (
        "surface-history",
        180,
        fields(
            simulateFailures="0",
            fetchFailures="0",
            fetchErrorState="0",
            cleanupCompleted="1",
            dynamicBodies="1",
            ownershipMetricsObserved="1",
            ownershipProbeMass="10",
            ownershipOutOfFootprintFrames="0",
            historyMetricsObserved="1",
            historyMotionSamples="60",
            softBody="0",
            cloth="0",
        ),
    ),
    "normal-post": (
        "normal-post-ownership",
        240,
        fields(
            simulateFailures="0",
            fetchFailures="0",
            fetchErrorState="0",
            cleanupCompleted="1",
            dynamicBodies="1",
            ownershipMetricsObserved="1",
            ownershipProbeMass="10",
            ownershipOutOfFootprintFrames="0",
            softBody="0",
            cloth="0",
        ),
    ),
    "broad-authority": (
        "broad-component-authority",
        16,
        fields(
            simulateFailures="0",
            fetchFailures="0",
            fetchErrorState="0",
            cleanupCompleted="1",
            dynamicBodies="2",
            softBody="0",
            cloth="0",
        ),
    ),
}

METRIC_FIELDS = (
    "dynamicBodies",
    "maxLinearSpeed",
    "maxAngularSpeed",
    "maxSpeed",
    "maxSettledSpeed",
    "settledSunkBoxes",
    "maxSettledSpreadXZ",
    "sphereContactEvents",
    "sphereContactPoints",
    "sphereResponseFraction",
    "sphereFinalGap",
    "sphereLateralDrift",
    "maxImpactVerticalGeomOverlap",
    "maxSettledVerticalGeomOverlap",
    "settledProximityFrames",
    "stressShots",
    "expectedStressShots",
    "stressContactEvents",
    "stressContactPoints",
    "stressMaxSunkBoxes",
    "stressWorstMinRelToSurface",
    "stressMaxPassThroughShots",
    "ownershipProbeMass",
    "ownershipMaxHorizontalSpeed",
    "ownershipMaxAngularSpeed",
    "ownershipMinBottomGap",
    "ownershipMaxBottomGap",
    "ownershipFinalDx",
    "ownershipFinalDy",
    "ownershipFinalDz",
    "historyContactEvents",
    "historyContactPoints",
    "historyMotionSamples",
    "historyTargetVelocityY",
    "historySurfaceRise",
    "historyBodyRise",
    "historyPoseFollowError",
    "historyMeanBodyVelocityY",
    "historyMeanAbsRelativeVelocityY",
    "historyMinBodyVelocityY",
    "historyMaxBodyVelocityY",
    "historyMaxAbsRelativeVelocityY",
    "historyVelocityFollowRatio",
)

SURFACE_PREFIX = "[avbd:surface-ownership] "
SURFACE_INT_FIELDS = (
    "deformAlRows",
    "deformAlEvals",
    "deformStrippedRows",
    "deformPositionTangentCandidates",
    "deformPositionTangentRows",
    "deformPositionTangentEvals",
    "deformPositionTangentMixedRejectRows",
    "deformPositionTangentShellRejectRows",
    "deformPositionTangentTargetRejectRows",
    "deformPositionTangentRestitutionRejectRows",
    "deformPositionTangentFiniteRejectRows",
    "deformPositionTangentScaleRejectRows",
    "deformShellSuppressedPrimalRows",
    "deformDepenCorrections",
    "deformFrictionRawRows",
    "deformFrictionDominantRows",
    "deformFrictionFewRows",
    "deformFrictionMultiRows",
    "deformFrictionCorrections",
    "deformFinalizeBodies",
    "deformFinalizeCorrections",
    "deformFinalizeSpatialCorrections",
    "deformFinalizeComFallbackCorrections",
    "deformFinalizeSecondaryRows",
    "deformFinalizeSecondaryResidualSeparationRows",
    "deformFinalizeManifoldBodies",
    "deformFinalizeManifoldOneRowBodies",
    "deformFinalizeManifoldTwoRowBodies",
    "deformFinalizeManifoldThreeRowBodies",
    "deformFinalizeManifoldFourRowBodies",
    "deformFinalizeManifoldOverFourRowBodies",
    "deformFinalizeManifoldFiveToEightRowBodies",
    "deformFinalizeManifoldNineToSixteenRowBodies",
    "deformFinalizeManifoldOverSixteenRowBodies",
    "deformFinalizeManifoldMixedScaleBodies",
    "deformFinalizeManifoldRankDeficientBodies",
    "deformFinalizeManifoldAliasRows",
    "deformFinalizeManifoldDynamicIncidentBodies",
    "deformFinalizeManifoldRigidStaticIncidentBodies",
    "deformFinalizeManifoldNonOwnerDeformableIncidentBodies",
    "deformFinalizeComponents",
    "deformFinalizeComponentOneBody",
    "deformFinalizeComponentTwoBodies",
    "deformFinalizeComponentThreeToFourBodies",
    "deformFinalizeComponentFiveToEightBodies",
    "deformFinalizeComponentNineToSixteenBodies",
    "deformFinalizeComponentSeventeenToThirtyTwoBodies",
    "deformFinalizeComponentOverThirtyTwoBodies",
    "deformFinalizeComponentOneToEightRows",
    "deformFinalizeComponentNineToSixteenRows",
    "deformFinalizeComponentSeventeenToThirtyTwoRows",
    "deformFinalizeComponentThirtyThreeToSixtyFourRows",
    "deformFinalizeComponentOverSixtyFourRows",
    "deformFinalizeComponentRestitution",
    "deformFinalizeComponentFiniteImpulse",
    "deformFinalizeComponentTargetVelocity",
    "deformFinalizeComponentMixedScale",
    "deformFinalizeComponentRigidStatic",
    "deformFinalizeComponentNonOwnerDeformable",
    "deformFinalizeComponentJointIsland",
    "deformFinalizeComponentLockedDof",
    "deformFinalizeComponentNonDynamicBody",
    "deformFinalizeBudgetDiagRows",
    "deformFinalizeBudgetDiagNoCorrectionRows",
    "deformFinalizeBudgetDiagZeroBudgetRequiredRows",
    "deformFinalizeBudgetDiagWithinBudgetRows",
    "deformFinalizeBudgetDiagOverBudgetRows",
    "deformFinalizeBudgetDiagUnsupportedRows",
    "deformFinalizeBudgetDiagComponentsWithinBudget",
    "deformFinalizeBudgetDiagComponentsOverBudget",
    "deformFinalizeBudgetDiagComponentsUnsupported",
    "deformFinalizeShadowComponents",
    "deformFinalizeShadowRows",
    "deformFinalizeShadowNoCorrection",
    "deformFinalizeShadowSolved",
    "deformFinalizeShadowCommitCapable",
    "deformFinalizeShadowBudgetExhausted",
    "deformFinalizeShadowInfeasible",
    "deformFinalizeShadowResidualUnclassified",
    "deformFinalizeShadowNumericalFailure",
    "deformFinalizeShadowIterationLimit",
    "deformFinalizeShadowUnsupported",
    "deformFinalizeShadowUnsupportedFastImpact",
    "deformFinalizeShadowUnsupportedSnapshot",
    "deformFinalizeShadowLowerRows",
    "deformFinalizeShadowFreeRows",
    "deformFinalizeShadowUpperRows",
    "deformFinalizeShadowMatrixFreeComponents",
    "deformFinalizeShadowMatrixFreeRows",
    "deformFinalizeShadowMatrixFreeNoCorrection",
    "deformFinalizeShadowMatrixFreeSolved",
    "deformFinalizeShadowMatrixFreeBudgetExhausted",
    "deformFinalizeShadowMatrixFreeInfeasible",
    "deformFinalizeShadowMatrixFreeResidualUnclassified",
    "deformFinalizeShadowMatrixFreeNumericalFailure",
    "deformFinalizeShadowMatrixFreeIterationLimit",
    "deformFinalizeShadowMatrixFreeIterations",
    "deformFinalizeShadowMatrixFreeIterationLimitKktAtMost2x",
    "deformFinalizeShadowMatrixFreeIterationLimitKktAtMost16x",
    "deformFinalizeShadowMatrixFreeIterationLimitKktOver16x",
    "deformFinalizeShadowMatrixFreeCommittedComponents",
    "deformFinalizeShadowMatrixFreeOracleComponents",
    "deformFinalizeShadowMatrixFreeOracleRows",
    "deformFinalizeShadowMatrixFreeOracleMatched",
    "deformFinalizeShadowMatrixFreeOracleMismatched",
    "deformFinalizeShadowMatrixFreeOracleSkipped",
    "deformFinalizePreOwnerBodies",
    "deformFinalizeLegacyOwnerBodies",
    "deformFinalizeOwnerDiscoveryMismatchBodies",
    "deformFinalizeProbeEligibleComponents",
    "deformFinalizeProbeCommittedComponents",
    "deformFinalizeProbeCommittedRows",
    "deformFinalizeProbeCommittedBodies",
    "deformFinalizeProbeReplacedOwnerBodies",
    "deformAlDepenRows",
    "deformAlFinalizeRows",
    "deformDepenFinalizeRows",
    "deformAlDepenFinalizeRows",
    "deformFinalizeContactFalsePositive",
    "deformFinalizeContactResidualSeparation",
    "deformFinalizeContactReversal",
    "shellContacts",
    "shellDepenCorrections",
    "shellFrictionRows",
    "shellFrictionCorrections",
    "shellFinalizeBodies",
    "shellFinalizeCorrections",
)
SURFACE_REAL_FIELDS = (
    "deformDepenDistance",
    "deformFrictionImpulse",
    "deformFinalizeDelta",
    "deformFinalizeContactPreSeparation",
    "deformFinalizeContactPostSeparation",
    "deformFinalizeContactPostApproach",
    "deformFinalizeSecondaryResidualSeparation",
    "shellDepenDistance",
    "shellFrictionImpulse",
    "shellFinalizeDelta",
)


def authority_from_log(path: Path) -> dict[str, str]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]
    if len(lines) != 1:
        return {}
    authority, errors = parse_authority(lines[0])
    return {} if errors else authority


def surface_ownership_from_log(
    path: Path,
) -> tuple[dict[str, int | Decimal], list[str]]:
    totals: dict[str, int | Decimal] = {
        "diagnosticFrames": 0,
        **{field: 0 for field in SURFACE_INT_FIELDS},
        **{field: Decimal(0) for field in SURFACE_REAL_FIELDS},
    }
    errors: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.startswith(SURFACE_PREFIX):
            continue
        values: dict[str, str] = {}
        for token in line[len(SURFACE_PREFIX) :].split():
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value
        missing = [
            field
            for field in SURFACE_INT_FIELDS + SURFACE_REAL_FIELDS
            if field not in values
        ]
        if missing:
            errors.append(
                f"surface diagnostic line {line_number} missing "
                + ",".join(missing)
            )
            continue
        try:
            totals["diagnosticFrames"] += 1
            for field in SURFACE_INT_FIELDS:
                totals[field] += int(values[field])
            for field in SURFACE_REAL_FIELDS:
                totals[field] += Decimal(values[field])
        except (ValueError, InvalidOperation):
            errors.append(
                f"surface diagnostic line {line_number} has invalid number"
            )
    return totals, errors


def surface_summary(values: dict[str, int | Decimal]) -> str:
    fields_to_print = ("diagnosticFrames",) + SURFACE_INT_FIELDS + SURFACE_REAL_FIELDS
    return " ".join(f"{field}={values[field]}" for field in fields_to_print)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("baseline", "acceptance"), default="baseline"
    )
    parser.add_argument(
        "--history-contract",
        choices=("failure-first", "acceptance"),
        default="acceptance",
        help=(
            "failure-first requires the retained-NP moving-surface velocity "
            "mismatch; acceptance requires coherent point history"
        ),
    )
    parser.add_argument(
        "--normal-post-contract",
        choices=("failure-first", "acceptance"),
        default="failure-first",
        help=(
            "failure-first requires one retained NP contact identity to be "
            "corrected by position AL, geometric depenetration, and material "
            "finalize in the same substep; acceptance requires that triple "
            "ownership to be absent"
        ),
    )
    parser.add_argument(
        "--finalize-contact-contract",
        choices=("observe", "failure-first", "acceptance"),
        default="observe",
        help=(
            "audit or gate dominant-contact material finalize against "
            "contact-point spatial normal velocity"
        ),
    )
    parser.add_argument(
        "--finalize-manifold-contract",
        choices=("observe", "failure-first", "acceptance"),
        default="observe",
        help=(
            "audit or gate residual separation on strict secondary "
            "retained-NP contacts after dominant-point material finalize"
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASES),
        dest="selected_cases",
        help="select a case; default is stack+sphere (shell is an explicit slow gate)",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--bounded-component-probe",
        action="store_true",
        help="enable the opt-in atomic bounded-component P3K replacement probe",
    )
    parser.add_argument(
        "--matrix-free-component-oracle",
        action="store_true",
        help=(
            "enable read-only same-component dense/matrix-free operator "
            "and bounded-result comparison"
        ),
    )
    parser.add_argument(
        "--execution-order",
        choices=("matrix", "parallel", "sequential"),
        default="matrix",
        help=(
            "matrix runs TGS parallel plus AVBD parallel/sequential; "
            "parallel or sequential selects one formulation-probe smoke lane"
        ),
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.timeout <= 0.0:
        parser.error("--repeats and --timeout must be positive")

    selected = args.selected_cases or ["stack", "sphere"]
    bin_dir = args.bin_dir.resolve()
    if args.output_root:
        output_root = args.output_root.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        output_root = (
            Path(tempfile.gettempdir())
            / f"PhysX_AVBD_moving_deformable_surface_{stamp}"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    specs: list[RunSpec] = []
    if args.execution_order == "parallel":
        run_lanes = (("tgs", "parallel"), ("avbd", "parallel"))
    elif args.execution_order == "sequential":
        run_lanes = (("avbd", "sequential"),)
    else:
        run_lanes = (
            ("tgs", "parallel"),
            ("avbd", "parallel"),
            ("avbd", "sequential"),
        )
    for repeat in range(1, args.repeats + 1):
        for short_case in selected:
            case_name, frames, required = CASES[short_case]
            for solver, execution in run_lanes:
                specs.append(
                    RunSpec(
                        f"{short_case}-{solver}-{execution}-r{repeat}",
                        "SnippetDeformableMesh_64.exe",
                        "SnippetDeformableMesh",
                        case_name,
                        execution,
                        frames,
                        "SUPPORTED",
                        (
                            "PROBE"
                            if short_case
                            in ("shell", "mass-light", "mass-heavy",
                                "shell-post", "history", "normal-post",
                                "broad-authority")
                            else "GATED"
                        ),
                        required_fields=required,
                        solver=solver,
                    )
                )

    observed: dict[tuple[str, str, str, int], dict[str, str]] = {}
    observed_surface: dict[
        tuple[str, str, str, int], dict[str, int | Decimal]
    ] = {}
    passed = True
    for index, spec in enumerate(specs, start=1):
        diagnostic_environment = {
            "PHYSX_AVBD_ITER_DIAG": "1",
            "PHYSX_AVBD_ITER_DIAG_EVERY": (
                "60" if spec.case == "moving-mesh-stack" else "1"
            ),
        }
        if args.bounded_component_probe:
            diagnostic_environment[
                "PHYSX_AVBD_BOUNDED_COMPONENT_PROBE"
            ] = "1"
        if args.matrix_free_component_oracle:
            diagnostic_environment[
                "PHYSX_AVBD_MATRIX_FREE_COMPONENT_ORACLE"
            ] = "1"
        previous_environment = {
            key: os.environ.get(key) for key in diagnostic_environment
        }
        os.environ.update(diagnostic_environment)
        try:
            result = run_one(bin_dir, output_root, spec, args.timeout)
        finally:
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        short_case = next(
            key for key, (case_name, _, _) in CASES.items()
            if case_name == spec.case
        )
        repeat = int(spec.name.rsplit("-r", 1)[1])
        log_path = Path(result.log)
        authority = authority_from_log(log_path)
        key = (short_case, spec.solver, spec.execution, repeat)
        observed[key] = authority
        surface, surface_errors = surface_ownership_from_log(log_path)
        observed_surface[key] = surface
        if spec.solver == "tgs":
            if surface["diagnosticFrames"] != 0:
                surface_errors.append(
                    "TGS unexpectedly emitted AVBD surface diagnostics"
                )
        else:
            if surface["diagnosticFrames"] == 0:
                surface_errors.append("AVBD surface diagnostics missing")
            if surface["deformAlRows"] == 0:
                surface_errors.append("deformable position-AL ownership missing")
            if surface["deformAlEvals"] < surface["deformAlRows"]:
                surface_errors.append(
                    "deformable AL evaluations are below owned rows"
                )
            if (
                surface["deformPositionTangentRows"]
                > 2
                * (
                    surface["deformAlRows"]
                    - surface["deformShellSuppressedPrimalRows"]
                )
            ):
                surface_errors.append(
                    "deformable position tangent rows exceed retained "
                    "two-axis contact capacity"
                )
            if (
                surface["deformPositionTangentEvals"]
                < surface["deformPositionTangentRows"]
            ):
                surface_errors.append(
                    "deformable position tangent evaluations are below "
                    "owned rows"
                )
            position_owner_contacts = (
                surface["deformPositionTangentRows"] // 2
            )
            rejected_position_contacts = sum(
                surface[field]
                for field in (
                    "deformPositionTangentMixedRejectRows",
                    "deformPositionTangentShellRejectRows",
                    "deformPositionTangentTargetRejectRows",
                    "deformPositionTangentRestitutionRejectRows",
                    "deformPositionTangentFiniteRejectRows",
                    "deformPositionTangentScaleRejectRows",
                )
            )
            if (
                surface["deformPositionTangentRows"] % 2 != 0
                or position_owner_contacts + rejected_position_contacts
                != surface["deformPositionTangentCandidates"]
            ):
                surface_errors.append(
                    "deformable position tangent candidate accounting "
                    "does not close"
                )
            if surface["deformStrippedRows"] != 0:
                surface_errors.append(
                    "deformable NP rows were stripped from the current path"
                )
            if (
                surface["deformShellSuppressedPrimalRows"]
                > surface["deformAlRows"]
            ):
                surface_errors.append(
                    "shell-suppressed deformable primal rows exceed AL rows"
                )
            if (
                surface["deformFrictionDominantRows"]
                > surface["deformFrictionRawRows"]
            ):
                surface_errors.append(
                    "deformable dominant friction rows exceed raw rows"
                )
            if (
                surface["deformFrictionFewRows"]
                + surface["deformFrictionMultiRows"]
                != surface["deformFrictionDominantRows"]
            ):
                surface_errors.append(
                    "deformable friction support classes do not own all "
                    "dominant rows"
                )
            if surface["shellFrictionRows"] > surface["shellContacts"]:
                surface_errors.append(
                    "shell dominant friction rows exceed shell contacts"
                )
            if (
                short_case in ("shell", "shell-post")
                and args.mode == "baseline"
            ):
                if surface["shellContacts"] == 0:
                    surface_errors.append(
                        "kinematic-shell fixture produced no shell contacts"
                    )
                if surface["deformShellSuppressedPrimalRows"] == 0:
                    surface_errors.append(
                        "kinematic-shell fixture did not suppress retained "
                        "deformable NP primal rows"
                    )
                if short_case == "shell-post":
                    if surface["shellDepenCorrections"] == 0:
                        surface_errors.append(
                            "failure-first shell fixture did not activate "
                            "shell depenetration"
                        )
                    for field in (
                        "shellFrictionRows",
                        "shellFrictionCorrections",
                        "shellFinalizeBodies",
                        "shellFinalizeCorrections",
                    ):
                        if surface[field] != 0:
                            surface_errors.append(
                                f"baseline shell launch unexpectedly reached "
                                f"{field}"
                            )
                    try:
                        peak_speed = Decimal(
                            authority.get("maxLinearSpeed", "NaN")
                        )
                        final_dy = Decimal(
                            authority.get("ownershipFinalDy", "NaN")
                        )
                    except InvalidOperation:
                        peak_speed = final_dy = Decimal("NaN")
                    if not peak_speed.is_finite() or peak_speed < Decimal("100"):
                        surface_errors.append(
                            "failure-first shell launch speed was not observed"
                        )
                    if not final_dy.is_finite() or final_dy < Decimal("100"):
                        surface_errors.append(
                            "failure-first shell launch displacement was not "
                            "observed"
                        )
            elif args.mode == "baseline":
                if surface["deformPositionTangentRows"] != 0:
                    surface_errors.append(
                        "baseline unexpectedly selected deformable position "
                        "tangent ownership"
                    )
                if short_case == "mass-light":
                    if surface["deformFrictionFewRows"] == 0:
                        surface_errors.append(
                            "light-mass fixture did not select few-contact "
                            "physics"
                        )
                    if surface["deformFrictionMultiRows"] != 0:
                        surface_errors.append(
                            "light-mass fixture selected multi-corner physics"
                        )
                elif short_case == "mass-heavy":
                    if surface["deformFrictionMultiRows"] == 0:
                        surface_errors.append(
                            "heavy-mass fixture did not expose multi-corner "
                            "physics"
                        )
            else:
                if surface["deformPositionTangentRows"] == 0:
                    surface_errors.append(
                        "accepted deformable position tangent owner missing"
                    )
                if (
                    surface["deformPositionTangentCandidates"]
                    != surface["deformAlRows"]
                    or surface["deformPositionTangentRows"]
                    != 2 * surface["deformAlRows"]
                ):
                    surface_errors.append(
                        "accepted fixture did not assign both tangent axes "
                        "to every deformable AL row"
                    )
                for field in (
                    "deformFrictionRawRows",
                    "deformFrictionDominantRows",
                    "deformFrictionFewRows",
                    "deformFrictionMultiRows",
                    "deformFrictionCorrections",
                ):
                    if surface[field] != 0:
                        surface_errors.append(
                            f"accepted position tangent row replayed by {field}"
                        )
                if surface["deformFrictionImpulse"] != 0:
                    surface_errors.append(
                        "accepted position tangent row replayed friction impulse"
                    )
                if short_case in (
                    "shell",
                    "shell-post",
                    "history",
                    "normal-post",
                ):
                    for field in (
                        "deformShellSuppressedPrimalRows",
                        "shellContacts",
                        "shellDepenCorrections",
                        "shellFrictionRows",
                        "shellFrictionCorrections",
                        "shellFinalizeBodies",
                        "shellFinalizeCorrections",
                    ):
                        if surface[field] != 0:
                            surface_errors.append(
                                f"retired direct-shell path remained active "
                                f"in {field}"
                            )
                if short_case == "shell-post":
                    try:
                        peak_speed = Decimal(
                            authority.get("maxLinearSpeed", "NaN")
                        )
                        final_dy = Decimal(
                            authority.get("ownershipFinalDy", "NaN")
                        )
                    except InvalidOperation:
                        peak_speed = final_dy = Decimal("NaN")
                    if not peak_speed.is_finite() or peak_speed > Decimal("50"):
                        surface_errors.append(
                            "focused shell replacement retirement did not "
                            "remove the launch speed"
                        )
                    if not final_dy.is_finite() or abs(final_dy) > Decimal("10"):
                        surface_errors.append(
                            "focused shell replacement retirement did not "
                            "remove the launch displacement"
                        )
                if short_case == "history":
                    try:
                        history_contacts = int(
                            authority.get("historyContactPoints", "0")
                        )
                        surface_rise = Decimal(
                            authority.get("historySurfaceRise", "NaN")
                        )
                        body_rise = Decimal(
                            authority.get("historyBodyRise", "NaN")
                        )
                        pose_follow_error = Decimal(
                            authority.get("historyPoseFollowError", "NaN")
                        )
                        mean_relative_velocity = Decimal(
                            authority.get(
                                "historyMeanAbsRelativeVelocityY", "NaN"
                            )
                        )
                        follow_ratio = Decimal(
                            authority.get(
                                "historyVelocityFollowRatio", "NaN"
                            )
                        )
                    except (ValueError, InvalidOperation):
                        history_contacts = 0
                        surface_rise = body_rise = pose_follow_error = Decimal(
                            "NaN"
                        )
                        mean_relative_velocity = follow_ratio = Decimal("NaN")
                    if history_contacts <= 0:
                        surface_errors.append(
                            "moving-surface point-history fixture had no "
                            "contact-point witness"
                        )
                    if (
                        not surface_rise.is_finite()
                        or surface_rise < Decimal("0.9")
                    ):
                        surface_errors.append(
                            "moving-surface point-history fixture did not "
                            "publish the calibrated surface rise"
                        )
                    if (
                        not body_rise.is_finite()
                        or not pose_follow_error.is_finite()
                    ):
                        surface_errors.append(
                            "moving-surface point-history pose metrics are "
                            "not finite"
                        )
                    elif pose_follow_error > Decimal("0.25"):
                        surface_errors.append(
                            "moving-surface body did not remain on the "
                            "translating support"
                        )
                    if args.history_contract == "failure-first":
                        if (
                            not follow_ratio.is_finite()
                            or follow_ratio > Decimal("0.35")
                        ):
                            surface_errors.append(
                                "failure-first fixture did not expose the "
                                "zero-history body-velocity mismatch"
                            )
                        if (
                            not mean_relative_velocity.is_finite()
                            or mean_relative_velocity < Decimal("0.65")
                        ):
                            surface_errors.append(
                                "failure-first fixture did not retain a "
                                "material relative-velocity mismatch"
                            )
                    else:
                        if (
                            not follow_ratio.is_finite()
                            or follow_ratio < Decimal("0.75")
                            or follow_ratio > Decimal("1.25")
                        ):
                            surface_errors.append(
                                "accepted point history did not preserve "
                                "moving-surface body velocity"
                            )
                        if (
                            not mean_relative_velocity.is_finite()
                            or mean_relative_velocity > Decimal("0.25")
                        ):
                            surface_errors.append(
                                "accepted point history retained excessive "
                                "material relative velocity"
                            )
                if short_case == "normal-post":
                    triple_rows = surface[
                        "deformAlDepenFinalizeRows"
                    ]
                    depen_finalize_rows = surface[
                        "deformDepenFinalizeRows"
                    ]
                    if args.normal_post_contract == "failure-first":
                        if triple_rows == 0:
                            surface_errors.append(
                                "failure-first fixture did not expose "
                                "same-contact AL/depenetration/finalize "
                                "ownership"
                            )
                        if depen_finalize_rows != triple_rows:
                            surface_errors.append(
                                "normal post-stage overlap escaped retained "
                                "deformable AL ownership"
                            )
                    elif triple_rows != 0:
                        surface_errors.append(
                            "accepted normal owner still replayed the same "
                            "contact through AL/depenetration/finalize"
                        )
                    finalize_mismatch_corrections = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeContactFalsePositive",
                            "deformFinalizeContactResidualSeparation",
                            "deformFinalizeContactReversal",
                        )
                    )
                    if args.finalize_contact_contract == "failure-first":
                        if finalize_mismatch_corrections == 0:
                            surface_errors.append(
                                "failure-first fixture did not expose a "
                                "COM/contact-point material-finalize mismatch"
                            )
                    elif args.finalize_contact_contract == "acceptance":
                        if surface["deformFinalizeCorrections"] == 0:
                            surface_errors.append(
                                "accepted material finalize was not exercised"
                            )
                        if (
                            surface["deformFinalizeSpatialCorrections"]
                            != surface["deformFinalizeCorrections"]
                        ):
                            surface_errors.append(
                                "accepted material finalize escaped the "
                                "strict spatial owner"
                            )
                        if (
                            surface[
                                "deformFinalizeComFallbackCorrections"
                            ]
                            != 0
                        ):
                            surface_errors.append(
                                "accepted material finalize used the "
                                "COM fallback"
                            )
                        if finalize_mismatch_corrections != 0:
                            surface_errors.append(
                                "accepted material finalize retained "
                                "COM/contact-point mismatch corrections"
                            )
                        for field in (
                            "deformFinalizeContactPostSeparation",
                            "deformFinalizeContactPostApproach",
                        ):
                            if surface[field] != 0:
                                surface_errors.append(
                                    "accepted material finalize retained "
                                    f"nonzero {field}"
                                )
                    secondary_rows = surface[
                        "deformFinalizeSecondaryRows"
                    ]
                    secondary_residual_rows = surface[
                        "deformFinalizeSecondaryResidualSeparationRows"
                    ]
                    secondary_residual = surface[
                        "deformFinalizeSecondaryResidualSeparation"
                    ]
                    manifold_bodies = surface[
                        "deformFinalizeManifoldBodies"
                    ]
                    manifold_histogram_bodies = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeManifoldOneRowBodies",
                            "deformFinalizeManifoldTwoRowBodies",
                            "deformFinalizeManifoldThreeRowBodies",
                            "deformFinalizeManifoldFourRowBodies",
                            "deformFinalizeManifoldOverFourRowBodies",
                        )
                    )
                    if manifold_bodies == 0:
                        surface_errors.append(
                            "deformable finalize manifold shape was not "
                            "observed"
                        )
                    if manifold_histogram_bodies != manifold_bodies:
                        surface_errors.append(
                            "deformable finalize manifold histogram does "
                            "not match observed bodies"
                        )
                    extended_manifold_bodies = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeManifoldFiveToEightRowBodies",
                            "deformFinalizeManifoldNineToSixteenRowBodies",
                            "deformFinalizeManifoldOverSixteenRowBodies",
                        )
                    )
                    if (
                        extended_manifold_bodies
                        != surface[
                            "deformFinalizeManifoldOverFourRowBodies"
                        ]
                    ):
                        surface_errors.append(
                            "extended deformable manifold histogram does "
                            "not match over-four-row bodies"
                        )
                    components = surface["deformFinalizeComponents"]
                    pre_owner_bodies = surface[
                        "deformFinalizePreOwnerBodies"
                    ]
                    legacy_owner_bodies = surface[
                        "deformFinalizeLegacyOwnerBodies"
                    ]
                    owner_mismatch_bodies = surface[
                        "deformFinalizeOwnerDiscoveryMismatchBodies"
                    ]
                    if owner_mismatch_bodies != 0:
                        surface_errors.append(
                            "pre-P3K strict owner discovery disagrees with "
                            "the legacy P3K owner"
                        )
                    probe_eligible = surface[
                        "deformFinalizeProbeEligibleComponents"
                    ]
                    probe_committed = surface[
                        "deformFinalizeProbeCommittedComponents"
                    ]
                    probe_replaced_owners = surface[
                        "deformFinalizeProbeReplacedOwnerBodies"
                    ]
                    if args.bounded_component_probe:
                        if probe_eligible == 0:
                            surface_errors.append(
                                "bounded component probe had no eligible "
                                "component"
                            )
                        if probe_eligible != probe_committed:
                            surface_errors.append(
                                "bounded component probe did not atomically "
                                "commit every eligible component"
                            )
                        if (
                            legacy_owner_bodies + probe_replaced_owners
                            != pre_owner_bodies
                        ):
                            surface_errors.append(
                                "legacy plus replaced owners do not close "
                                "the pre-P3K owner ledger"
                            )
                        for field in (
                            "deformFinalizeProbeCommittedRows",
                            "deformFinalizeProbeCommittedBodies",
                            "deformFinalizeProbeReplacedOwnerBodies",
                        ):
                            if surface[field] == 0:
                                surface_errors.append(
                                    f"{field} was not exercised"
                                )
                    else:
                        if pre_owner_bodies != legacy_owner_bodies:
                            surface_errors.append(
                                "pre-P3K and legacy strict owner totals differ"
                            )
                        if legacy_owner_bodies != manifold_bodies:
                            surface_errors.append(
                                "legacy strict owner total does not match "
                                "the manifold owner total"
                            )
                        for field in (
                            "deformFinalizeProbeEligibleComponents",
                            "deformFinalizeProbeCommittedComponents",
                            "deformFinalizeProbeCommittedRows",
                            "deformFinalizeProbeCommittedBodies",
                            "deformFinalizeProbeReplacedOwnerBodies",
                        ):
                            if surface[field] != 0:
                                surface_errors.append(
                                    f"{field} active without probe request"
                                )
                    component_body_histogram = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeComponentOneBody",
                            "deformFinalizeComponentTwoBodies",
                            "deformFinalizeComponentThreeToFourBodies",
                            "deformFinalizeComponentFiveToEightBodies",
                            "deformFinalizeComponentNineToSixteenBodies",
                            "deformFinalizeComponentSeventeenToThirtyTwoBodies",
                            "deformFinalizeComponentOverThirtyTwoBodies",
                        )
                    )
                    component_row_histogram = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeComponentOneToEightRows",
                            "deformFinalizeComponentNineToSixteenRows",
                            "deformFinalizeComponentSeventeenToThirtyTwoRows",
                            "deformFinalizeComponentThirtyThreeToSixtyFourRows",
                            "deformFinalizeComponentOverSixtyFourRows",
                        )
                    )
                    if components == 0:
                        surface_errors.append(
                            "strict deformable finalize component "
                            "topology was not observed"
                        )
                    if component_body_histogram != components:
                        surface_errors.append(
                            "deformable finalize component body histogram "
                            "does not match observed components"
                        )
                    if component_row_histogram != components:
                        surface_errors.append(
                            "deformable finalize component row histogram "
                            "does not match observed components"
                        )
                    budget_rows = surface["deformFinalizeBudgetDiagRows"]
                    budget_row_classes = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeBudgetDiagNoCorrectionRows",
                            "deformFinalizeBudgetDiagZeroBudgetRequiredRows",
                            "deformFinalizeBudgetDiagWithinBudgetRows",
                            "deformFinalizeBudgetDiagOverBudgetRows",
                            "deformFinalizeBudgetDiagUnsupportedRows",
                        )
                    )
                    if budget_row_classes != budget_rows:
                        surface_errors.append(
                            "deformable finalize budget diagnostic row "
                            "classes do not match observed rows"
                        )
                    budget_component_classes = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeBudgetDiagComponentsWithinBudget",
                            "deformFinalizeBudgetDiagComponentsOverBudget",
                            "deformFinalizeBudgetDiagComponentsUnsupported",
                        )
                    )
                    if budget_component_classes != components:
                        surface_errors.append(
                            "deformable finalize budget diagnostic component "
                            "classes do not match observed components"
                        )
                    shadow_components = surface[
                        "deformFinalizeShadowComponents"
                    ]
                    shadow_rows = surface["deformFinalizeShadowRows"]
                    shadow_outcomes = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeShadowNoCorrection",
                            "deformFinalizeShadowSolved",
                            "deformFinalizeShadowBudgetExhausted",
                            "deformFinalizeShadowInfeasible",
                            "deformFinalizeShadowResidualUnclassified",
                            "deformFinalizeShadowNumericalFailure",
                            "deformFinalizeShadowIterationLimit",
                            "deformFinalizeShadowUnsupported",
                        )
                    )
                    if shadow_components != components:
                        surface_errors.append(
                            "bounded component shadow count does not match "
                            "the strict topology"
                        )
                    if shadow_rows != budget_rows:
                        surface_errors.append(
                            "bounded component shadow row count does not "
                            "match the frozen budget snapshot"
                        )
                    if shadow_outcomes != shadow_components:
                        surface_errors.append(
                            "bounded component shadow outcomes do not "
                            "partition the strict topology"
                        )
                    if (
                        surface["deformFinalizeShadowCommitCapable"]
                        != surface["deformFinalizeShadowSolved"]
                    ):
                        surface_errors.append(
                            "bounded component shadow commit-capable count "
                            "does not match solved candidates"
                        )
                    if (
                        surface["deformFinalizeProbeCommittedComponents"]
                        > surface["deformFinalizeShadowSolved"]
                    ):
                        surface_errors.append(
                            "bounded component probe committed more "
                            "components than the shadow solved"
                        )
                    for field in (
                        "deformFinalizeShadowUnsupportedFastImpact",
                        "deformFinalizeShadowUnsupportedSnapshot",
                    ):
                        if (
                            surface[field]
                            > surface["deformFinalizeShadowUnsupported"]
                        ):
                            surface_errors.append(
                                f"{field} exceeds unsupported shadow "
                                "components"
                            )
                    shadow_state_rows = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeShadowLowerRows",
                            "deformFinalizeShadowFreeRows",
                            "deformFinalizeShadowUpperRows",
                        )
                    )
                    if shadow_state_rows > shadow_rows:
                        surface_errors.append(
                            "bounded component shadow row states exceed "
                            "the complete input row set"
                        )
                    matrix_free_components = surface[
                        "deformFinalizeShadowMatrixFreeComponents"
                    ]
                    matrix_free_rows = surface[
                        "deformFinalizeShadowMatrixFreeRows"
                    ]
                    matrix_free_outcomes = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeShadowMatrixFreeNoCorrection",
                            "deformFinalizeShadowMatrixFreeSolved",
                            "deformFinalizeShadowMatrixFreeBudgetExhausted",
                            "deformFinalizeShadowMatrixFreeInfeasible",
                            "deformFinalizeShadowMatrixFreeResidualUnclassified",
                            "deformFinalizeShadowMatrixFreeNumericalFailure",
                            "deformFinalizeShadowMatrixFreeIterationLimit",
                        )
                    )
                    matrix_free_committed = surface[
                        "deformFinalizeShadowMatrixFreeCommittedComponents"
                    ]
                    matrix_free_iterations = surface[
                        "deformFinalizeShadowMatrixFreeIterations"
                    ]
                    matrix_free_iteration_limit_buckets = sum(
                        surface[field]
                        for field in (
                            "deformFinalizeShadowMatrixFreeIterationLimitKktAtMost2x",
                            "deformFinalizeShadowMatrixFreeIterationLimitKktAtMost16x",
                            "deformFinalizeShadowMatrixFreeIterationLimitKktOver16x",
                        )
                    )
                    if matrix_free_outcomes != matrix_free_components:
                        surface_errors.append(
                            "matrix-free shadow outcomes do not partition "
                            "the invoked backend components"
                        )
                    if (
                        matrix_free_iteration_limit_buckets
                        != surface[
                            "deformFinalizeShadowMatrixFreeIterationLimit"
                        ]
                    ):
                        surface_errors.append(
                            "matrix-free iteration-limit KKT buckets do not "
                            "partition iteration-limit outcomes"
                        )
                    if (
                        matrix_free_components == 0
                        and matrix_free_iterations != 0
                    ):
                        surface_errors.append(
                            "matrix-free iterations recorded without an "
                            "invoked backend component"
                        )
                    if (
                        matrix_free_iterations
                        < surface[
                            "deformFinalizeShadowMatrixFreeIterationLimit"
                        ]
                    ):
                        surface_errors.append(
                            "matrix-free iteration ledger is below its "
                            "iteration-limit component count"
                        )
                    if (
                        matrix_free_components > shadow_components
                        or matrix_free_rows > shadow_rows
                    ):
                        surface_errors.append(
                            "matrix-free shadow ledger exceeds the complete "
                            "shadow input"
                        )
                    if (
                        matrix_free_components > 0
                        and matrix_free_rows
                        <= 128 * matrix_free_components
                    ):
                        surface_errors.append(
                            "matrix-free backend was recorded below its "
                            "implementation-selection boundary"
                        )
                    if (
                        matrix_free_committed
                        > surface["deformFinalizeShadowMatrixFreeSolved"]
                        or matrix_free_committed > probe_committed
                    ):
                        surface_errors.append(
                            "matrix-free commit ledger exceeds solved or "
                            "global committed components"
                        )
                    if (
                        not args.bounded_component_probe
                        and matrix_free_committed != 0
                    ):
                        surface_errors.append(
                            "matrix-free component committed without the "
                            "explicit probe request"
                        )
                    oracle_components = surface[
                        "deformFinalizeShadowMatrixFreeOracleComponents"
                    ]
                    oracle_rows = surface[
                        "deformFinalizeShadowMatrixFreeOracleRows"
                    ]
                    oracle_matched = surface[
                        "deformFinalizeShadowMatrixFreeOracleMatched"
                    ]
                    oracle_mismatched = surface[
                        "deformFinalizeShadowMatrixFreeOracleMismatched"
                    ]
                    oracle_skipped = surface[
                        "deformFinalizeShadowMatrixFreeOracleSkipped"
                    ]
                    if args.matrix_free_component_oracle:
                        if oracle_components == 0 or oracle_rows == 0:
                            surface_errors.append(
                                "matrix-free same-component oracle was not "
                                "exercised"
                            )
                        if oracle_matched == 0:
                            surface_errors.append(
                                "matrix-free same-component oracle had no "
                                "comparable match"
                            )
                        if oracle_mismatched != 0:
                            surface_errors.append(
                                "matrix-free same-component oracle observed "
                                "a backend mismatch"
                            )
                        if (
                            oracle_matched
                            + oracle_mismatched
                            + oracle_skipped
                            != oracle_components
                        ):
                            surface_errors.append(
                                "matrix-free same-component oracle outcomes "
                                "do not partition sampled components"
                            )
                    else:
                        for field in (
                            "deformFinalizeShadowMatrixFreeOracleComponents",
                            "deformFinalizeShadowMatrixFreeOracleRows",
                            "deformFinalizeShadowMatrixFreeOracleMatched",
                            "deformFinalizeShadowMatrixFreeOracleMismatched",
                            "deformFinalizeShadowMatrixFreeOracleSkipped",
                        ):
                            if surface[field] != 0:
                                surface_errors.append(
                                    f"{field} active without oracle request"
                                )
                    component_owner_bound = (
                        pre_owner_bodies
                        if args.bounded_component_probe
                        else manifold_bodies
                    )
                    if components > component_owner_bound:
                        surface_errors.append(
                            "deformable finalize component count exceeds "
                            "strict owner body count"
                        )
                    for field in (
                        "deformFinalizeComponentRestitution",
                        "deformFinalizeComponentFiniteImpulse",
                        "deformFinalizeComponentTargetVelocity",
                        "deformFinalizeComponentMixedScale",
                        "deformFinalizeComponentRigidStatic",
                        "deformFinalizeComponentNonOwnerDeformable",
                        "deformFinalizeComponentJointIsland",
                        "deformFinalizeComponentLockedDof",
                        "deformFinalizeComponentNonDynamicBody",
                    ):
                        if surface[field] > components:
                            surface_errors.append(
                                f"{field} exceeds observed component count"
                            )
                    if (
                        args.finalize_manifold_contract
                        == "failure-first"
                    ):
                        if secondary_rows == 0:
                            surface_errors.append(
                                "failure-first fixture did not expose "
                                "secondary retained-NP contacts"
                            )
                        if (
                            secondary_residual_rows == 0
                            or secondary_residual == 0
                        ):
                            surface_errors.append(
                                "failure-first fixture did not expose "
                                "secondary contact-point separation"
                            )
                    elif (
                        args.finalize_manifold_contract == "acceptance"
                    ):
                        if secondary_rows == 0:
                            surface_errors.append(
                                "accepted manifold finalize was not "
                                "exercised on secondary contacts"
                            )
                        if (
                            secondary_residual_rows != 0
                            or secondary_residual != 0
                        ):
                            surface_errors.append(
                                "accepted manifold finalize retained "
                                "secondary contact-point separation"
                            )
                if short_case == "broad-authority":
                    matrix_free_components = surface[
                        "deformFinalizeShadowMatrixFreeComponents"
                    ]
                    matrix_free_solved = surface[
                        "deformFinalizeShadowMatrixFreeSolved"
                    ]
                    matrix_free_committed = surface[
                        "deformFinalizeShadowMatrixFreeCommittedComponents"
                    ]
                    if matrix_free_components == 0:
                        surface_errors.append(
                            "broad-authority fixture did not invoke the "
                            "matrix-free backend"
                        )
                    if matrix_free_solved == 0:
                        surface_errors.append(
                            "broad-authority fixture produced no matrix-free "
                            "Solved component"
                        )
                    if matrix_free_committed == 0:
                        surface_errors.append(
                            "broad-authority fixture produced no atomic "
                            "matrix-free commit"
                        )
                    if matrix_free_committed != matrix_free_solved:
                        surface_errors.append(
                            "broad-authority matrix-free solved/commit ledger "
                            "is not exact"
                        )
                if short_case == "shell":
                    try:
                        stress_shots = int(
                            authority.get("stressShots", "0")
                        )
                        expected_stress_shots = int(
                            authority.get("expectedStressShots", "0")
                        )
                        stress_contact_events = int(
                            authority.get("stressContactEvents", "0")
                        )
                        stress_contact_points = int(
                            authority.get("stressContactPoints", "0")
                        )
                        stress_pass_through = int(
                            authority.get(
                                "stressMaxPassThroughShots", "-1"
                            )
                        )
                        stress_sunk_boxes = int(
                            authority.get("stressMaxSunkBoxes", "-1")
                        )
                    except ValueError:
                        stress_shots = expected_stress_shots = 0
                        stress_contact_events = stress_contact_points = 0
                        stress_pass_through = stress_sunk_boxes = -1
                        surface_errors.append(
                            "stress authority contained a non-integer metric"
                        )
                    if (
                        expected_stress_shots <= 0
                        or stress_shots != expected_stress_shots
                    ):
                        surface_errors.append(
                            "stress fixture did not complete every "
                            "scheduled shot"
                        )
                    if (
                        stress_contact_events <= 0
                        or stress_contact_points <= 0
                    ):
                        surface_errors.append(
                            "stress fixture did not observe contact"
                        )
                    if stress_pass_through != 0:
                        surface_errors.append(
                            "stress fixture retained pass-through shots"
                        )
                    if stress_sunk_boxes != 0:
                        surface_errors.append(
                            "stress fixture allowed boxes to sink below "
                            "the accepted surface boundary"
                        )

                    secondary_rows = surface[
                        "deformFinalizeSecondaryRows"
                    ]
                    secondary_residual_rows = surface[
                        "deformFinalizeSecondaryResidualSeparationRows"
                    ]
                    secondary_residual = surface[
                        "deformFinalizeSecondaryResidualSeparation"
                    ]
                    if (
                        args.finalize_manifold_contract
                        == "failure-first"
                    ):
                        if secondary_rows == 0:
                            surface_errors.append(
                                "failure-first stress did not expose "
                                "secondary retained-NP contacts"
                            )
                        if (
                            secondary_residual_rows == 0
                            or secondary_residual == 0
                        ):
                            surface_errors.append(
                                "failure-first stress did not expose "
                                "secondary contact-point separation"
                            )
                    elif (
                        args.finalize_manifold_contract == "acceptance"
                    ):
                        if secondary_rows == 0:
                            surface_errors.append(
                                "accepted stress manifold finalize was "
                                "not exercised"
                            )
                        if (
                            secondary_residual_rows != 0
                            or secondary_residual != 0
                        ):
                            surface_errors.append(
                                "accepted stress manifold finalize retained "
                                "secondary contact-point separation"
                            )
        metrics = " ".join(
            f"{field}={authority.get(field, 'MISSING')}"
            for field in METRIC_FIELDS
        )
        print(
            "[MOVING_DEFORMABLE_SURFACE_RUN] "
            f"index={index}/{len(specs)} name={spec.name} "
            f"runner={'PASS' if result.passed else 'FAIL'} {metrics}",
            flush=True,
        )
        print(
            "[MOVING_DEFORMABLE_SURFACE_OWNERSHIP] "
            f"name={spec.name} {surface_summary(surface)}",
            flush=True,
        )
        for error in result.errors:
            print(
                f"[MOVING_DEFORMABLE_SURFACE_ERROR] "
                f"name={spec.name} error={error}",
                flush=True,
            )
        for error in surface_errors:
            print(
                f"[MOVING_DEFORMABLE_SURFACE_ERROR] "
                f"name={spec.name} error={error}",
                flush=True,
            )
        passed = passed and result.passed and not surface_errors
        if result.visible_window_detected:
            break

    deterministic = METRIC_FIELDS + (
        "status",
        "reason",
        "completedFrames",
        "nonFinite",
        "fetchFailures",
        "cleanupCompleted",
    )
    if len(observed) == len(specs):
        for short_case in selected:
            if args.execution_order == "matrix":
                parallel_surface = observed_surface[
                    (short_case, "avbd", "parallel", 1)
                ]
                parallel = observed[(short_case, "avbd", "parallel", 1)]
                sequential = observed[
                    (short_case, "avbd", "sequential", 1)
                ]
                for field in deterministic:
                    if parallel.get(field) != sequential.get(field):
                        passed = False
                        print(
                            "[MOVING_DEFORMABLE_SURFACE_ERROR] "
                            f"case={short_case} field={field} "
                            "parallel/sequential mismatch",
                            flush=True,
                        )
                sequential_surface = observed_surface[
                    (short_case, "avbd", "sequential", 1)
                ]
                for field in (
                    "diagnosticFrames",
                    *SURFACE_INT_FIELDS,
                    *SURFACE_REAL_FIELDS,
                ):
                    if parallel_surface[field] != sequential_surface[field]:
                        passed = False
                        print(
                            "[MOVING_DEFORMABLE_SURFACE_ERROR] "
                            f"case={short_case} surfaceField={field} "
                            "parallel/sequential mismatch",
                            flush=True,
                        )
            for repeat in range(2, args.repeats + 1):
                for solver, execution in run_lanes:
                    reference = observed[
                        (short_case, solver, execution, 1)
                    ]
                    actual = observed[
                        (short_case, solver, execution, repeat)
                    ]
                    for field in deterministic:
                        if reference.get(field) != actual.get(field):
                            passed = False
                            print(
                                "[MOVING_DEFORMABLE_SURFACE_ERROR] "
                                f"case={short_case} solver={solver} "
                                f"execution={execution} field={field} "
                                "repeat mismatch",
                                flush=True,
                            )
                    surface_reference = observed_surface[
                        (short_case, solver, execution, 1)
                    ]
                    surface_actual = observed_surface[
                        (short_case, solver, execution, repeat)
                    ]
                    for field in (
                        "diagnosticFrames",
                        *SURFACE_INT_FIELDS,
                        *SURFACE_REAL_FIELDS,
                    ):
                        if surface_reference[field] != surface_actual[field]:
                            passed = False
                            print(
                                "[MOVING_DEFORMABLE_SURFACE_ERROR] "
                                f"case={short_case} solver={solver} "
                                f"execution={execution} "
                                f"surfaceField={field} repeat mismatch",
                                flush=True,
                            )
        if "mass-light" in selected and "mass-heavy" in selected:
            light_surface = observed_surface[
                ("mass-light", "avbd", "parallel", 1)
            ]
            heavy_surface = observed_surface[
                ("mass-heavy", "avbd", "parallel", 1)
            ]
            mass_pair_equal_fields = (
                (
                    "diagnosticFrames",
                    "deformAlRows",
                    "deformFrictionRawRows",
                    "deformFrictionDominantRows",
                    "deformFinalizeBodies",
                )
                if args.mode == "baseline"
                else ("diagnosticFrames",)
            )
            for field in mass_pair_equal_fields:
                if light_surface[field] != heavy_surface[field]:
                    passed = False
                    print(
                        "[MOVING_DEFORMABLE_SURFACE_ERROR] "
                        f"massSplitField={field} light/heavy contact "
                        "population mismatch",
                        flush=True,
                    )
            light_avbd = observed[
                ("mass-light", "avbd", "parallel", 1)
            ]
            heavy_avbd = observed[
                ("mass-heavy", "avbd", "parallel", 1)
            ]
            light_tgs = observed[
                ("mass-light", "tgs", "parallel", 1)
            ]
            heavy_tgs = observed[
                ("mass-heavy", "tgs", "parallel", 1)
            ]
            mass_summary_fields = (
                "ownershipProbeMass",
                "ownershipFinalDx",
            )
            missing_mass_summary_fields = [
                f"{lane_name}.{field}"
                for lane_name, lane in (
                    ("lightAvbd", light_avbd),
                    ("heavyAvbd", heavy_avbd),
                    ("lightTgs", light_tgs),
                    ("heavyTgs", heavy_tgs),
                )
                for field in mass_summary_fields
                if field not in lane
            ]
            if missing_mass_summary_fields:
                passed = False
                print(
                    "[MOVING_DEFORMABLE_SURFACE_ERROR] "
                    "mass split summary unavailable; missing "
                    + ",".join(missing_mass_summary_fields),
                    flush=True,
                )
            else:
                avbd_dx_jump = abs(
                    Decimal(heavy_avbd["ownershipFinalDx"])
                    - Decimal(light_avbd["ownershipFinalDx"])
                )
                tgs_dx_jump = abs(
                    Decimal(heavy_tgs["ownershipFinalDx"])
                    - Decimal(light_tgs["ownershipFinalDx"])
                )
                if args.mode == "acceptance" and avbd_dx_jump > tgs_dx_jump:
                    passed = False
                    print(
                        "[MOVING_DEFORMABLE_SURFACE_ERROR] "
                        f"massPairAvbdDxJump={avbd_dx_jump} exceeds "
                        f"same-fixture TGS variation={tgs_dx_jump}",
                        flush=True,
                    )
                print(
                    "[MOVING_DEFORMABLE_SURFACE_MASS_SPLIT] "
                    f"lightMass={light_avbd['ownershipProbeMass']} "
                    f"heavyMass={heavy_avbd['ownershipProbeMass']} "
                    f"lightContactRows={light_surface['deformAlRows']} "
                    f"heavyContactRows={heavy_surface['deformAlRows']} "
                    f"lightPositionTangentRows="
                    f"{light_surface['deformPositionTangentRows']} "
                    f"heavyPositionTangentRows="
                    f"{heavy_surface['deformPositionTangentRows']} "
                    f"lightFinalizeBodies="
                    f"{light_surface['deformFinalizeBodies']} "
                    f"heavyFinalizeBodies="
                    f"{heavy_surface['deformFinalizeBodies']} "
                    f"dominantRows="
                    f"{light_surface['deformFrictionDominantRows']} "
                    f"lightFewRows={light_surface['deformFrictionFewRows']} "
                    f"heavyMultiRows="
                    f"{heavy_surface['deformFrictionMultiRows']} "
                    f"lightAvbdFinalDx={light_avbd['ownershipFinalDx']} "
                    f"heavyAvbdFinalDx={heavy_avbd['ownershipFinalDx']} "
                    f"avbdDxJump={avbd_dx_jump} tgsDxJump={tgs_dx_jump}",
                    flush=True,
                )
    else:
        passed = False

    print(
        "[MOVING_DEFORMABLE_SURFACE_MATRIX] "
        f"mode={args.mode} completed={len(observed)} "
        f"expected={len(specs)} artifactRoot={output_root} "
        f"status={'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
