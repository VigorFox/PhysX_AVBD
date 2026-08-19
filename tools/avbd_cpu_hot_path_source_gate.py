#!/usr/bin/env python3
"""Reject diagnostics and experimental controls in the scalar primal hot path."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "physx/source/lowleveldynamics/src/DyAvbdSoftBodyComponent.h"
STEP_STATE_SOURCE = (
    ROOT / "physx/source/lowleveldynamics/src/DyAvbdSoftBodyStepState.cpp")
SCALAR_STEP_SOURCE = (
    ROOT / "physx/source/lowleveldynamics/src/DyAvbdSoftBodyScalar.cpp")
LOW_LEVEL_CMAKE = ROOT / "physx/source/compiler/cmake/LowLevelDynamics.cmake"


def extract_braced_block(text: str, anchor: str) -> str:
    start = text.find(anchor)
    if start < 0:
        raise RuntimeError(f"missing source anchor: {anchor!r}")
    open_brace = text.find("{", start)
    if open_brace < 0:
        raise RuntimeError(f"missing opening brace after: {anchor!r}")
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise RuntimeError(f"unterminated source block: {anchor!r}")


def reject_tokens(label: str, block: str, tokens: tuple[str, ...]) -> list[str]:
    return [f"{label}: forbidden token {token!r}"
            for token in tokens if token in block]


def require_ordered_tokens(
        label: str, block: str, tokens: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    previous = -1
    for token in tokens:
        position = block.find(token, previous + 1)
        if position < 0:
            errors.append(f"{label}: missing ordered token {token!r}")
            continue
        if position <= previous:
            errors.append(f"{label}: reordered token {token!r}")
            continue
        previous = position
    return errors


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    errors: list[str] = []

    solve_context = extract_braced_block(
        text, "PX_FORCE_INLINE void solve(\n\t\tconst AvbdSoftBody& sb")
    errors.extend(reject_tokens(
        "particle-primal solve", solve_context,
        ("std::getenv", "PHYSX_AVBD_", "PxTime", "recordWorkCensus",
         "AvbdParticlePrimalWorkCensus", "census")))
    errors.extend(require_ordered_tokens(
        "particle-primal authority", solve_context,
        (
            "const PxReal massDtSq",
            "PxMat33 H = PxMat33::createDiagonal",
            "PxVec3 f = (particle.predictedPosition - particle.position)",
            "elementAdjacency.triRefs.size()",
            "avbdEvaluateStVKForceHessian(",
            "elementAdjacency.tetRefs.size()",
            "avbdEvaluateCorotationalForceHessianPrepared(",
            "avbdEvaluateNeoHookeanForceHessianPrepared(",
            "elementAdjacency.bendRefs.size()",
            "avbdEvaluateBendingForceHessian(",
            "contactStarts[particleIndex]",
            "avbdEvaluateContactParticleBlock(",
            "objectiveAdjacency.objectiveIndices.size()",
            "avbdEvaluatePinForceHessian(",
            "if(particle.damping > 0.0f)",
            "if(particle.elasticK > 0.0f)",
            "avbdSolveSymmetric33(H, f)",
            "avbdLimitTetDisplacementFromLinearizations(",
            "particle.position += limitResult.appliedDisplacement",
            "avbdTruncateDisplacement(",
            "observation.sweepObservation.observe(",
        )))

    finite_helper = extract_braced_block(
        text, "PX_FORCE_INLINE bool avbdIsFiniteVector(")
    for component in ("value.x", "value.y", "value.z"):
        if f"PxIsFinite({component})" not in finite_helper:
            errors.append(
                "corotational finite helper no longer force-inlines "
                f"{component}")

    polar_rotation = extract_braced_block(
        text, "PX_FORCE_INLINE PxMat33 avbdExtractCorotationalRotation(")
    errors.extend(reject_tokens(
        "corotational polar loop", polar_rotation,
        ("rotation = (rotation + inverseTranspose) * 0.5f;",
         ".isFinite()")))
    for column in ("column0", "column1", "column2"):
        expected = (
            f"rotation.{column} =\n"
            f"\t\t\t\t(rotation.{column} + inverseTranspose.{column}) * 0.5f;")
        if expected not in polar_rotation:
            errors.append(
                "corotational polar loop no longer uses the explicit "
                f"force-inlined {column} average")

    scalar_loop_start = text.find(
        "// Canonical scalar reference traversal.")
    scalar_loop_end = text.find(
        "maxDxSq =", scalar_loop_start)
    if scalar_loop_start < 0 or scalar_loop_end < scalar_loop_start:
        errors.append("scalar reference traversal anchors are missing or reordered")
    else:
        scalar_loop = text[scalar_loop_start:scalar_loop_end]
        errors.extend(reject_tokens(
            "scalar reference traversal", scalar_loop,
            ("std::getenv", "PHYSX_AVBD_", "PxTime", "particlePrimalKernel",
             "recordWorkCensus", "AvbdParticlePrimalWorkCensus", "census")))

    canonical_call = re.compile(
        r"particlePrimalSolveContext\.solve\(\s*"
        r"body,\s*localIndex,\s*particlePrimalObservation\s*\);")
    if not canonical_call.search(scalar_loop):
        errors.append("scalar reference traversal no longer calls the canonical primal kernel")

    if "avbdSolveParticlePrimalCorotationalTetPacketBodyRange(" not in scalar_loop:
        errors.append("SIMD candidate is no longer selected at the sweep boundary")

    if "useColoredSerialPrimal" in scalar_loop:
        errors.append("default-off colored primal remains in scalar traversal")

    if "AvbdSoftBodyStepState persistentStepState;" in text[
            text.find("void avbdStepSoftBodies("):]:
        errors.append("default scalar step still carries persistent P4 state")

    step_state_guard = "#if defined(DY_AVBD_SOFT_BODY_STEP_STATE_IMPLEMENTATION)"
    step_state_end = "#endif // DY_AVBD_SOFT_BODY_STEP_STATE_IMPLEMENTATION"
    guard_start = text.find(step_state_guard)
    guard_end = text.find(step_state_end, guard_start)
    scalar_step_start = text.find("void avbdStepSoftBodies(")
    if not (0 <= guard_start < guard_end < scalar_step_start):
        errors.append(
            "Scene step-state methods are not isolated from scalar header consumers")
    else:
        state_implementation = text[guard_start:guard_end]
        if "AvbdSoftBodyStepState::runToCompletionSerial()" not in state_implementation:
            errors.append("step-state implementation guard is incomplete")
    if not STEP_STATE_SOURCE.is_file():
        errors.append("missing dedicated step-state implementation TU")
    else:
        step_state_source = STEP_STATE_SOURCE.read_text(encoding="utf-8")
        expected_owner = (
            "#define DY_AVBD_SOFT_BODY_STEP_STATE_IMPLEMENTATION\n"
            "#include \"DyAvbdSoftBodyComponent.h\"")
        if expected_owner not in step_state_source:
            errors.append("dedicated step-state TU no longer owns the guarded methods")
    cmake_text = LOW_LEVEL_CMAKE.read_text(encoding="utf-8")
    if "${LLDYNAMICS_BASE_DIR}/src/DyAvbdSoftBodyStepState.cpp" not in cmake_text:
        errors.append("dedicated step-state TU is missing from LowLevelDynamics")

    scalar_step_guard = (
        "#if defined(DY_AVBD_SOFT_BODY_SCALAR_STEP_IMPLEMENTATION)")
    scalar_step_end = (
        "#endif // DY_AVBD_SOFT_BODY_SCALAR_STEP_IMPLEMENTATION")
    scalar_impl_start = text.find(scalar_step_guard)
    scalar_impl_end = text.find(scalar_step_end, scalar_impl_start)
    if not (scalar_step_start < scalar_impl_start < scalar_impl_end):
        errors.append("scalar step implementation guard is missing or reordered")
    if not SCALAR_STEP_SOURCE.is_file():
        errors.append("missing dedicated scalar-step implementation TU")
    else:
        scalar_step_source = SCALAR_STEP_SOURCE.read_text(encoding="utf-8")
        expected_scalar_owner = (
            "#define DY_AVBD_SOFT_BODY_SCALAR_STEP_IMPLEMENTATION\n"
            "#include \"DyAvbdSoftBodyComponent.h\"")
        if expected_scalar_owner not in scalar_step_source:
            errors.append("dedicated scalar-step TU no longer owns the implementation")
    if "${LLDYNAMICS_BASE_DIR}/src/DyAvbdSoftBodyScalar.cpp" not in cmake_text:
        errors.append("dedicated scalar-step TU is missing from LowLevelDynamics")

    contact_index = extract_braced_block(
        text, "inline void avbdBuildSoftParticleContactIndex(")
    errors.extend(reject_tokens(
        "contact-index epoch rebuild", contact_index,
        ("std::getenv", "avbdGetParticlePrimalSchedule()",
         "avbdValidateParticlePrimalAccessPlan()")))

    serial_redetection = extract_braced_block(
        text, "inline void avbdDetectAllOGCContacts(")
    errors.extend(reject_tokens(
        "serial OGC redetection", serial_redetection,
        ("avbdBuildSoftContactRedetectionPhasePlan(",
         "avbdBeginSoftContactRedetection(",
         "avbdCompleteSoftContactRedetection(")))

    for policy_name in (
            "avbdUseSurfaceTriangleBvh",
            "avbdUseSurfaceEdgeBvh",
            "avbdUseRigidTriangleSurfaceBvh"):
        policy = extract_braced_block(
            text, f"PX_FORCE_INLINE bool {policy_name}()")
        if "static const bool enabled = []()" not in policy:
            errors.append(
                f"{policy_name}: OGC policy is not cached at a cold boundary")
        if policy.count("std::getenv") != 1:
            errors.append(
                f"{policy_name}: expected exactly one cold environment sample")

    cached_task_policies = (
        "avbdUseRigidTriangleSurfaceContactTaskFanIn",
        "avbdForceRigidTriangleSurfaceContactTaskFanIn",
        "avbdUseRigidTriangleSurfaceContactTaskThreshold96",
        "avbdUseRigidTriangleSurfaceFeatureRoundRobinTaskPlan",
        "avbdUseRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan",
        "avbdUseRigidTriangleSurfaceFeatureSweptSubstageTiming",
        "avbdUseRigidTriangleSurfaceFeatureForwardOwnerQueryStats",
        "avbdUseRigidTriangleSurfaceFeatureDiscreteQueryStats",
        "avbdUseRigidTriangleSurfaceFeatureDiscreteBodyLocalBoundsCull",
        "avbdDisableRigidTriangleSurfaceFeatureDiscreteBodyLocalBoundsCull",
        "avbdUseRigidTriangleSurfaceFeatureForwardOwnerResultCache",
        "avbdDisableRigidTriangleSurfaceFeatureForwardOwnerResultCache",
        "avbdUseSoftPairContactTaskFanIn",
        "avbdForceSoftPairContactTaskFanIn",
        "avbdUseSelfBvhContactTaskFanIn",
        "avbdForceSelfBvhContactTaskFanIn",
        "avbdValidateRedetectionPhasePlan",
        "avbdUsePersistentStepStateSerial",
        "avbdUseCausalLayerTaskFanIn",
        "avbdForceCausalLayerTaskFanIn",
        "avbdForceCausalLayerTaskGraphReference",
        "avbdUseCausalLayerTaskPartition",
        "avbdForceCausalLayerTaskPartition",
        "avbdUseSceneRedetectionBridge",
        "avbdUseWorldPlaneContactTaskFanIn",
        "avbdForceWorldPlaneContactTaskFanIn",
        "avbdUseRigidBoxSdfContactTaskFanIn",
        "avbdForceRigidBoxSdfContactTaskFanIn",
        "avbdUseRigidSphereSdfContactTaskFanIn",
        "avbdForceRigidSphereSdfContactTaskFanIn",
        "avbdUseRigidCapsuleSdfContactTaskFanIn",
        "avbdForceRigidCapsuleSdfContactTaskFanIn",
        "avbdUseRigidConvexSdfContactTaskFanIn",
        "avbdForceRigidConvexSdfContactTaskFanIn",
    )
    for policy_name in cached_task_policies:
        policy = extract_braced_block(
            text, f"PX_FORCE_INLINE bool {policy_name}()")
        if "static const bool enabled" not in policy:
            errors.append(
                f"{policy_name}: task policy is not cached at a cold boundary")
        if "std::getenv" in policy and "[]()" not in policy:
            errors.append(
                f"{policy_name}: direct environment query remains task-reachable")

    cold_census = extract_braced_block(
        text, "PX_NOINLINE inline void avbdAccumulateParticlePrimalWorkCensusForOuterEpoch(")
    errors.extend(reject_tokens(
        "cold census helper", cold_census,
        ("std::getenv", "PHYSX_AVBD_")))

    if errors:
        for error in errors:
            print(f"AVBD_CPU_HOT_PATH_SOURCE_GATE=FAIL error={error}")
        return 1
    print(
        "[AVBD_CPU_HOT_PATH_SOURCE_GATE] "
        "scalar-primal=clean authority-order=guarded polar-codegen=guarded "
        "diagnostics=isolated step-state-tu=isolated scalar-step-tu=isolated "
        "epoch-policy=cold ogc-policy=cold task-policy=cold status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
