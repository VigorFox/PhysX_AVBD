#!/usr/bin/env python3
"""Run CPU AVBD soft-body component/coexistence cases without a window."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetDeformableVolumeAVBD_64.exe"
CASES = (
    "volume-ground",
    "volume-static-box",
    "soft-soft",
    "cone-ground",
    "scene-volume-lifecycle",
    "scene-volume-corotational",
    "scene-volume-ground",
    "scene-volume-static-box",
    "scene-volume-static-churn",
    "scene-volume-dynamic-box",
    "scene-volume-dynamic-sphere",
    "scene-volume-dynamic-capsule",
    "scene-volume-dynamic-convex",
    "scene-volume-dynamic-churn",
    "scene-volume-multi-dynamic-box",
    "scene-volume-multi-soft-islands",
    "scene-volume-sleep-wake",
    "scene-volume-rigid-wake",
    "scene-volume-mixed-sleep-islands",
    "scene-volume-soft-churn",
    "scene-volume-buffer-mutation",
    "scene-volume-world-pin",
    "scene-volume-world-element-attachment",
    "scene-volume-rigid-attachment",
    "scene-volume-rigid-element-attachment",
    "scene-volume-static-attachment",
    "scene-volume-static-element-attachment",
    "scene-volume-kinematic-attachment",
    "scene-volume-kinematic-element-attachment",
    "scene-volume-articulation-attachment",
    "scene-volume-articulation-element-attachment",
    "scene-volume-element-filter",
    "scene-volume-partial-element-filter",
    "scene-volume-kinematic-box",
    "scene-volume-kinematic-sphere",
    "scene-volume-kinematic-capsule",
    "scene-volume-kinematic-convex",
    "scene-volume-kinematic-triangle-mesh",
    "scene-volume-kinematic-heightfield",
    "scene-volume-multi-scene-isolation",
    "scene-volume-soft-soft-wake",
    "scene-volume-volume-attachment",
    "scene-volume-full-kinematic-target",
    "scene-volume-partial-kinematic-target",
    "scene-volume-skinning",
    "scene-volume-motion-controls",
    "scene-volume-max-depenetration-velocity",
    "scene-volume-speculative-ccd",
    "scene-volume-plane-speculative-ccd",
    "scene-volume-sphere-speculative-ccd",
    "scene-volume-capsule-speculative-ccd",
    "scene-volume-convex-speculative-ccd",
    "scene-volume-moving-kinematic-sphere-speculative-ccd",
    "scene-volume-moving-kinematic-capsule-speculative-ccd",
    "scene-volume-rotating-kinematic-capsule-speculative-ccd",
    "scene-volume-rotating-kinematic-convex-speculative-ccd",
    "scene-volume-moving-kinematic-convex-speculative-ccd",
    "scene-volume-dynamic-sphere-relative-swept-ccd",
    "scene-volume-dynamic-capsule-relative-swept-ccd",
    "scene-volume-dynamic-rotating-capsule-relative-swept-ccd",
    "scene-volume-dynamic-rotating-convex-relative-swept-ccd",
    "scene-volume-dynamic-convex-relative-swept-ccd",
    "scene-volume-deforming-sphere-reverse-swept-ccd",
    "scene-volume-deforming-capsule-reverse-swept-ccd",
    "scene-volume-deforming-convex-reverse-swept-ccd",
    "scene-volume-static-sphere-reverse-swept-ccd",
    "scene-volume-kinematic-sphere-reverse-swept-ccd",
    "scene-volume-dynamic-sphere-reverse-swept-ccd",
    "scene-volume-static-capsule-reverse-swept-ccd",
    "scene-volume-kinematic-capsule-reverse-swept-ccd",
    "scene-volume-dynamic-capsule-reverse-swept-ccd",
    "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
    "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
    "scene-volume-rotating-kinematic-convex-reverse-swept-ccd",
    "scene-volume-dynamic-rotating-convex-reverse-swept-ccd",
    "scene-volume-static-convex-reverse-swept-ccd",
    "scene-volume-kinematic-convex-reverse-swept-ccd",
    "scene-volume-dynamic-convex-reverse-swept-ccd",
    "scene-volume-static-triangle-mesh-speculative-ccd",
    "scene-volume-kinematic-triangle-mesh-speculative-ccd",
    "scene-volume-static-heightfield-speculative-ccd",
    "scene-volume-kinematic-heightfield-speculative-ccd",
    "scene-volume-static-triangle-mesh-reverse-swept-ccd",
    "scene-volume-kinematic-triangle-mesh-reverse-swept-ccd",
    "scene-volume-static-heightfield-reverse-swept-ccd",
    "scene-volume-kinematic-heightfield-reverse-swept-ccd",
    "scene-volume-rotating-kinematic-triangle-mesh-speculative-ccd",
    "scene-volume-rotating-kinematic-heightfield-speculative-ccd",
    "scene-volume-rotating-kinematic-triangle-mesh-reverse-swept-ccd",
    "scene-volume-rotating-kinematic-heightfield-reverse-swept-ccd",
    "scene-volume-sphere-reverse-feature",
    "scene-volume-capsule-reverse-feature",
    "scene-volume-convex-reverse-feature",
    "scene-volume-triangle-mesh-reverse-feature",
    "scene-volume-heightfield-reverse-feature",
    "current-all",
)
INT_KEYS = (
    "frames",
    "fetchFailures",
    "particles",
    "softBodies",
    "tetElements",
    "surfaceTriangles",
    "rigidBoxes",
    "sceneStatics",
    "sceneDynamics",
    "sceneDeformableVolumes",
    "sceneActorCreated",
    "sceneShapeAttached",
    "sceneSimulationMeshAttached",
    "sceneHostBuffersInitialized",
    "sceneActorAdded",
    "sceneActorRemoved",
    "sceneActorReleased",
    "sceneBoundsFinite",
    "sceneSecondVolumeActorCreated",
    "sceneSecondVolumeHostBuffersInitialized",
    "sceneSecondVolumeActorAdded",
    "sceneSecondVolumeActorRemoved",
    "sceneSecondVolumeActorReleased",
    "sceneSecondVolumeBoundsFinite",
    "sceneSoftInitiallyAwake",
    "sceneSoftFirstSlept",
    "sceneSoftFirstSleepFrame",
    "sceneSoftSleepWakeCounterZero",
    "sceneSoftSleepVelocitiesZero",
    "sceneSoftStableWhileSleeping",
    "sceneSoftCounterWakeIssued",
    "sceneSoftWokeByCounter",
    "sceneSoftCounterWakeFrame",
    "sceneSoftSecondSlept",
    "sceneSoftSecondSleepFrame",
    "sceneSoftVelocityWakeIssued",
    "sceneSoftWokeByVelocity",
    "sceneSoftVelocityWakeFrame",
    "sceneSoftMovedAfterVelocityWake",
    "sceneSoftVelocityStopIssued",
    "sceneSoftFinalSlept",
    "sceneSoftFinalSleepFrame",
    "sceneSoftRigidWakeActorAdded",
    "sceneSoftWokeByRigid",
    "sceneSoftRigidWakeFrame",
    "sceneSoftMovedAfterRigidWake",
    "sceneMixedFirstSlept",
    "sceneMixedFirstSleepFrame",
    "sceneMixedFirstStable",
    "sceneMixedSecondStayedAwake",
    "sceneMixedSecondMoved",
    "sceneSoftChurnRemoveCount",
    "sceneSoftChurnReaddCount",
    "sceneSoftChurnCycles",
    "sceneSoftChurnPostCompactMoveCount",
    "sceneSoftChurnStable",
    "sceneBufferMutationIssued",
    "sceneBufferMutationWoke",
    "sceneBufferMutationApplied",
    "sceneBufferDriveIssued",
    "sceneBufferPinHeld",
    "sceneBufferDynamicMoved",
    "sceneBufferInvMassRestored",
    "sceneBufferRestoredMoved",
    "sceneBufferResetIssued",
    "sceneWorldPinCreated",
    "sceneWorldPinHeld",
    "sceneWorldPinActorReadded",
    "sceneWorldPinReleased",
    "sceneWorldPinMovedAfterRelease",
    "sceneRigidAttachmentActorAdded",
    "sceneRigidAttachmentInitiallySleeping",
    "sceneRigidAttachmentCreated",
    "sceneRigidAttachmentRigidWoke",
    "sceneRigidAttachmentRigidMoved",
    "sceneRigidAttachmentHeldAcrossReadd",
    "sceneRigidAttachmentReleased",
    "sceneRigidAttachmentSeparatedAfterRelease",
    "sceneArticulationCreated",
    "sceneArticulationAdded",
    "sceneArticulationInitiallySleeping",
    "sceneArticulationWoke",
    "sceneArticulationJointSubspaceHeld",
    "sceneArticulationRootStable",
    "sceneElementFilterCreated",
    "sceneElementFilterActorReadded",
    "sceneElementFilterSuppressedContact",
    "sceneElementFilterReleased",
    "sceneElementFilterContactRestored",
    "scenePartialFilterUnfilteredContactHeld",
    "scenePartialFilterExactOwnership",
    "sceneKinematicActorAdded",
    "sceneKinematicTargetIssued",
    "sceneKinematicTargetReached",
    "sceneKinematicSoftWoke",
    "sceneKinematicSoftMoved",
    "sceneKinematicContactObserved",
    "sceneVolumeTargetBound",
    "sceneVolumeTargetMutated",
    "sceneVolumeTargetWoke",
    "sceneVolumeTargetReached",
    "sceneVolumePartialInactiveIgnored",
    "sceneVolumePartialActivated",
    "sceneVolumePartialActivatedReached",
    "sceneSecondSceneCreated",
    "sceneSecondSceneSolverMatched",
    "scenePrimarySceneReleased",
    "sceneSecondSceneReleased",
    "sceneMultiPrimaryStable",
    "sceneMultiPrimaryDetachedStable",
    "sceneMultiSecondaryUpdatedBeforeRelease",
    "sceneMultiSecondaryUpdatedAfterRelease",
    "sceneSoftSoftBothSlept",
    "sceneSoftSoftDriveIssued",
    "sceneSoftSoftDriverWoke",
    "sceneSoftSoftTargetWoke",
    "sceneSoftSoftTargetWakeFrame",
    "sceneSoftSoftTargetMoved",
    "sceneSoftSoftResetIssued",
    "sceneSoftSoftBothFinalSlept",
    "motionMaxVelocityBounded",
    "motionSettlingApplied",
    "motionSettlingSlept",
    "motionControlStayedAwake",
    "depenetrationLimitApplied",
    "depenetrationFirstStepBounded",
    "depenetrationControlSeparated",
    "depenetrationGradualRecovery",
    "speculativeCcdFlagApplied",
    "speculativeCcdPreventedTunneling",
    "speculativeCcdNegativeControlTunneled",
    "sceneStaticShapeDetached",
    "sceneStaticShapeReattached",
    "sceneStaticActorRemoved",
    "sceneStaticActorReadded",
    "sceneDynamicActorAdded",
    "sceneDynamicActorReleased",
    "sceneDynamicInitiallySleeping",
    "sceneDynamicWokeBySoft",
    "sceneDynamicFirstWakeFrame",
    "sceneDynamicShapeDetached",
    "sceneDynamicShapeReattached",
    "sceneDynamicActorRemoved",
    "sceneDynamicActorReadded",
    "sceneDynamicReaddedSleeping",
    "sceneDynamicRewokeBySoft",
    "sceneDynamicSecondWakeFrame",
    "sceneSecondDynamicActorAdded",
    "sceneSecondDynamicActorRemoved",
    "sceneSecondDynamicActorReleased",
    "sceneSecondDynamicInitiallySleeping",
    "sceneSecondDynamicWokeBySoft",
    "sceneSecondDynamicFirstWakeFrame",
    "groundContactFrames",
    "rigidContactFrames",
    "softContactFrames",
    "maxGroundContacts",
    "maxRigidContacts",
    "maxSoftContacts",
    "invalidContactSourceSamples",
    "finalInsideParticles",
    "nonFiniteParticleSamples",
    "invertedElementSamples",
    "firstInversionFrame",
    "firstInversionBody",
    "firstInversionElement",
    "invertedBodiesMask",
    "fatalErrors",
    "warningErrors",
    "cleanupComplete",
    "movingSphereTargetIssued",
    "movingSphereCcdResponseObserved",
    "movingSphereNegativeControlHeld",
    "dynamicSphereSweepLaunched",
    "dynamicSphereSweepResponseObserved",
    "dynamicSphereSweepNegativeControlTunneled",
    "dynamicSphereSweepTwoSidedResponseObserved",
)
FLOAT_KEYS = (
    "minDetF",
    "maxDetF",
    "minBodyVolumeRatio",
    "maxBodyVolumeRatio",
    "minY",
    "maxY",
    "finalMinY",
    "finalMaxY",
    "maxParticleSpeed",
    "finalMaxParticleSpeed",
    "maxCentroidDrop",
    "sceneSecondVolumeMaxCentroidDrop",
    "sceneSecondVolumeFinalCentroidY",
    "sceneWorldPinMaxDrift",
    "sceneWorldPinReleasedMaxDisplacement",
    "sceneRigidAttachmentMaxDrift",
    "sceneRigidAttachmentMaxRigidDisplacement",
    "sceneRigidAttachmentMaxRigidSpeed",
    "sceneRigidAttachmentReleasedSeparation",
    "sceneArticulationRootMaxDisplacement",
    "sceneArticulationChildMaxForbiddenDisplacement",
    "sceneArticulationChildMaxAngularDisplacement",
    "sceneElementFilterMinY",
    "sceneElementFilterFinalMinY",
    "scenePartialFilterUnfilteredMinY",
    "sceneKinematicMaxPoseError",
    "sceneKinematicSoftDisplacement",
    "sceneKinematicFinalY",
    "sceneVolumeTargetFinalMaxError",
    "sceneVolumeTargetMaxDisplacement",
    "sceneVolumePartialInactiveDecoyDistance",
    "sceneDynamicInitialY",
    "sceneDynamicFinalY",
    "sceneDynamicMinY",
    "sceneDynamicMaxDrop",
    "sceneDynamicPreContactMaxDrop",
    "sceneDynamicMaxDownSpeed",
    "sceneSecondDynamicInitialY",
    "sceneSecondDynamicMinY",
    "sceneSecondDynamicFinalY",
    "sceneSecondDynamicMaxDrop",
    "sceneSecondDynamicPreContactMaxDrop",
    "sceneSecondDynamicMaxDownSpeed",
    "minDynamicSurfaceSeparation",
    "finalDynamicSurfaceSeparation",
    "motionMaxVelocityFirstStepDisplacement",
    "motionMaxVelocityFirstStepSpeed",
    "motionSettlingFinalSpeed",
    "motionControlFinalSpeed",
    "depenetrationLimitedFirstStepRise",
    "depenetrationControlFirstStepRise",
    "depenetrationLimitedFinalRise",
    "depenetrationLimitedMaxSpeed",
    "speculativeCcdPositiveMinY",
    "speculativeCcdPositiveMinSeparation",
    "speculativeCcdNegativeMaxY",
    "movingSpherePositiveDisplacement",
    "movingSphereNegativeDisplacement",
    "movingSpherePositiveMinSeparation",
    "dynamicSphereSweepPositiveSoftDisplacement",
    "dynamicSphereSweepNegativeSoftDisplacement",
    "dynamicSphereSweepPositiveRigidDrop",
    "dynamicSphereSweepNegativeRigidDrop",
    "dynamicSphereSweepPositiveMinSeparation",
)
PERF_INT_KEYS = (
    "warmupFrames",
    "profileFrames",
    "softWorkers",
    "convergenceSweeps",
    "requestedOuterIterations",
    "requestedInnerIterations",
    "executedOuterIterations",
    "executedInnerIterations",
    "particleSweeps",
    "trustRegionLimitedParticleSteps",
    "positiveJLimitedParticleSteps",
    "positiveJRejectedParticleSteps",
    "nonFiniteRejectedParticleSteps",
    "tetLinearizationCacheFallbackParticleSteps",
    "legacyAppliedConvergedOuterIterations",
    "residualConvergedOuterIterations",
    "unsafeAppliedConvergenceCandidates",
    "budgetExhaustedOuterIterations",
    "shadowResidual1e5ConvergedOuterIterations",
    "shadowResidual1e5SavedInnerIterations",
    "shadowResidual1e4ConvergedOuterIterations",
    "shadowResidual1e4SavedInnerIterations",
    "workspaceGrowthEvents",
    "workspaceGrowthBytes",
    "contactWorkspaceGrowthEvents",
    "contactWorkspaceGrowthBytes",
    "contactOutputGrowthEvents",
    "contactOutputGrowthBytes",
    "detectionCalls",
    "bodyPairs",
    "overlappingBodyPairs",
    "particleSurfaceCandidates",
    "insideTriangleTests",
    "closestTriangleTests",
    "selfTriangleTests",
    "rigidParticleBoxTests",
    "generatedGroundContacts",
    "generatedRigidContacts",
    "generatedSoftContacts",
    "generatedSelfContacts",
)
PERF_FLOAT_KEYS = (
    "avgStepMs",
    "p50StepMs",
    "p95StepMs",
    "maxStepMs",
    "initialContactMs",
    "solverMs",
    "sceneMs",
    "metricsMs",
    "predictionMs",
    "contactIndexMs",
    "bodyPrecomputeMs",
    "bodySolveMs",
    "particleSolveMs",
    "projectionMs",
    "dualMs",
    "redetectMs",
    "velocityMs",
    "frictionMs",
    "solverUnattributedMs",
    "closureMs",
    "convergenceTolerance",
    "finalMaxDisplacement",
    "finalMaxLocalSolveDisplacement",
    "finalMaxAppliedDisplacement",
)


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


def run_one(
    case_name: str,
    repeat: int,
    bin_dir: Path,
    frames: int,
    timeout: float,
    execution: str,
    warmup: int,
) -> tuple[bool, dict[str, str], dict[str, str]]:
    name = f"{case_name}-r{repeat}"
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        "--solver=avbd",
        f"--case={case_name}",
        f"--execution={execution}",
        f"--frames={frames}",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = "avbd"
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(frames)
    env["PHYSX_AVBD_PROFILE_WARMUP"] = str(warmup)
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
    perf_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_PERF] ")
    ]
    sphere_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_SPHERE_REVERSE_FEATURE] ")
    ]
    sphere_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_SPHERE_REVERSE_SWEPT] ")
    ]
    capsule_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CAPSULE_REVERSE_SWEPT] ")
    ]
    capsule_rotational_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_CAPSULE_ROTATIONAL_REVERSE_SWEPT] "
        )
    ]
    capsule_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CAPSULE_ROTATIONAL_SWEPT] ")
    ]
    capsule_dynamic_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CAPSULE_DYNAMIC_ROTATIONAL_SWEPT] ")
    ]
    convex_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CONVEX_REVERSE_SWEPT] ")
    ]
    deforming_volume_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_DEFORMING_VOLUME_REVERSE_SWEPT] "
        )
    ]
    convex_rotational_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_CONVEX_ROTATIONAL_REVERSE_SWEPT] "
        )
    ]
    convex_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CONVEX_ROTATIONAL_SWEPT] ")
    ]
    convex_dynamic_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CONVEX_DYNAMIC_ROTATIONAL_SWEPT] ")
    ]
    triangle_surface_forward_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_TRIANGLE_SURFACE_FORWARD_SWEPT] ")
    ]
    triangle_surface_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_TRIANGLE_SURFACE_REVERSE_SWEPT] ")
    ]
    triangle_surface_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_TRIANGLE_SURFACE_ROTATIONAL_SWEPT] "
        )
    ]
    capsule_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CAPSULE_REVERSE_FEATURE] ")
    ]
    convex_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CONVEX_REVERSE_FEATURE] ")
    ]
    triangle_mesh_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_TRIANGLE_MESH_REVERSE_FEATURE] ")
    ]
    heightfield_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_HEIGHTFIELD_REVERSE_FEATURE] ")
    ]
    errors: list[str] = []
    fields: dict[str, str] = {}
    perf_fields: dict[str, str] = {}
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
    if len(perf_lines) != 1:
        errors.append(f"perf count is {len(perf_lines)}, expected exactly 1")
    else:
        perf_fields, parse_errors = parse_gate(perf_lines[0])
        errors.extend(parse_errors)
    scene_lifecycle = case_name in (
        "scene-volume-lifecycle",
        "scene-volume-corotational",
    )
    scene_ground = case_name == "scene-volume-ground"
    scene_static_churn = case_name == "scene-volume-static-churn"
    scene_static_box = (
        case_name == "scene-volume-static-box" or scene_static_churn
    )
    scene_static = scene_ground or scene_static_box
    scene_dynamic_churn = case_name == "scene-volume-dynamic-churn"
    scene_multi_dynamic = case_name == "scene-volume-multi-dynamic-box"
    scene_multi_soft = case_name == "scene-volume-multi-soft-islands"
    scene_soft_sleep_wake = case_name == "scene-volume-sleep-wake"
    scene_soft_rigid_wake = case_name == "scene-volume-rigid-wake"
    scene_mixed_sleep = case_name == "scene-volume-mixed-sleep-islands"
    scene_soft_churn = case_name == "scene-volume-soft-churn"
    scene_buffer_mutation = case_name == "scene-volume-buffer-mutation"
    scene_world_pin = case_name in (
        "scene-volume-world-pin",
        "scene-volume-world-element-attachment",
    )
    scene_rigid_attachment = (
        case_name
        in (
            "scene-volume-rigid-attachment",
            "scene-volume-rigid-element-attachment",
        )
    )
    scene_static_attachment = (
        case_name
        in (
            "scene-volume-static-attachment",
            "scene-volume-static-element-attachment",
        )
    )
    scene_kinematic_attachment = (
        case_name
        in (
            "scene-volume-kinematic-attachment",
            "scene-volume-kinematic-element-attachment",
        )
    )
    scene_articulation_attachment = (
        case_name
        in (
            "scene-volume-articulation-attachment",
            "scene-volume-articulation-element-attachment",
        )
    )
    scene_attachment = (
        scene_rigid_attachment
        or scene_static_attachment
        or scene_kinematic_attachment
        or scene_articulation_attachment
    )
    scene_partial_element_filter = (
        case_name == "scene-volume-partial-element-filter"
    )
    scene_element_filter = (
        case_name == "scene-volume-element-filter"
        or scene_partial_element_filter
    )
    scene_kinematic = case_name in (
        "scene-volume-kinematic-box",
        "scene-volume-kinematic-sphere",
        "scene-volume-kinematic-capsule",
        "scene-volume-kinematic-convex",
        "scene-volume-kinematic-triangle-mesh",
        "scene-volume-kinematic-heightfield",
    )
    scene_multi_scene = case_name == "scene-volume-multi-scene-isolation"
    scene_soft_soft_wake = case_name == "scene-volume-soft-soft-wake"
    scene_soft_pair_attachment = (
        case_name == "scene-volume-volume-attachment"
    )
    scene_full_kinematic_target = (
        case_name == "scene-volume-full-kinematic-target"
    )
    scene_partial_kinematic_target = (
        case_name == "scene-volume-partial-kinematic-target"
    )
    scene_volume_kinematic_target = (
        scene_full_kinematic_target or scene_partial_kinematic_target
    )
    scene_motion_controls = case_name == "scene-volume-motion-controls"
    scene_max_depenetration_velocity = (
        case_name == "scene-volume-max-depenetration-velocity"
    )
    scene_rotating_kinematic_capsule_ccd = (
        case_name
        == "scene-volume-rotating-kinematic-capsule-speculative-ccd"
    )
    scene_dynamic_rotating_capsule_ccd = (
        case_name
        == "scene-volume-dynamic-rotating-capsule-relative-swept-ccd"
    )
    scene_rotating_kinematic_convex_ccd = (
        case_name
        == "scene-volume-rotating-kinematic-convex-speculative-ccd"
    )
    scene_dynamic_rotating_convex_ccd = (
        case_name
        == "scene-volume-dynamic-rotating-convex-relative-swept-ccd"
    )
    scene_rotating_kinematic_finite_ccd = (
        scene_rotating_kinematic_capsule_ccd
        or scene_rotating_kinematic_convex_ccd
    )
    scene_dynamic_rotating_finite_ccd = (
        scene_dynamic_rotating_capsule_ccd
        or scene_dynamic_rotating_convex_ccd
    )
    scene_moving_kinematic_finite_ccd = case_name in (
        "scene-volume-moving-kinematic-sphere-speculative-ccd",
        "scene-volume-moving-kinematic-capsule-speculative-ccd",
        "scene-volume-rotating-kinematic-capsule-speculative-ccd",
        "scene-volume-rotating-kinematic-convex-speculative-ccd",
        "scene-volume-moving-kinematic-convex-speculative-ccd",
    )
    scene_dynamic_finite_swept_ccd = case_name in (
        "scene-volume-dynamic-sphere-relative-swept-ccd",
        "scene-volume-dynamic-capsule-relative-swept-ccd",
        "scene-volume-dynamic-rotating-capsule-relative-swept-ccd",
        "scene-volume-dynamic-rotating-convex-relative-swept-ccd",
        "scene-volume-dynamic-convex-relative-swept-ccd",
    )
    scene_triangle_surface_swept_ccd = case_name in (
        "scene-volume-static-triangle-mesh-speculative-ccd",
        "scene-volume-kinematic-triangle-mesh-speculative-ccd",
        "scene-volume-static-heightfield-speculative-ccd",
        "scene-volume-kinematic-heightfield-speculative-ccd",
        "scene-volume-static-triangle-mesh-reverse-swept-ccd",
        "scene-volume-kinematic-triangle-mesh-reverse-swept-ccd",
        "scene-volume-static-heightfield-reverse-swept-ccd",
        "scene-volume-kinematic-heightfield-reverse-swept-ccd",
        "scene-volume-rotating-kinematic-triangle-mesh-speculative-ccd",
        "scene-volume-rotating-kinematic-heightfield-speculative-ccd",
        "scene-volume-rotating-kinematic-triangle-mesh-reverse-swept-ccd",
        "scene-volume-rotating-kinematic-heightfield-reverse-swept-ccd",
    )
    scene_rotational_triangle_surface_swept_ccd = (
        scene_triangle_surface_swept_ccd
        and "rotating-kinematic" in case_name
    )
    scene_triangle_surface_reverse_swept_ccd = (
        scene_triangle_surface_swept_ccd
        and "reverse-swept" in case_name
    )
    scene_static_triangle_surface_swept_ccd = (
        scene_triangle_surface_swept_ccd
        and case_name.startswith("scene-volume-static-")
    )
    scene_kinematic_triangle_surface_swept_ccd = (
        scene_triangle_surface_swept_ccd
        and not scene_static_triangle_surface_swept_ccd
    )
    scene_heightfield_swept_ccd = (
        scene_triangle_surface_swept_ccd
        and "heightfield" in case_name
    )
    scene_sphere_reverse_swept_ccd = case_name in (
        "scene-volume-deforming-sphere-reverse-swept-ccd",
        "scene-volume-deforming-capsule-reverse-swept-ccd",
        "scene-volume-deforming-convex-reverse-swept-ccd",
        "scene-volume-static-sphere-reverse-swept-ccd",
        "scene-volume-kinematic-sphere-reverse-swept-ccd",
        "scene-volume-dynamic-sphere-reverse-swept-ccd",
        "scene-volume-static-capsule-reverse-swept-ccd",
        "scene-volume-kinematic-capsule-reverse-swept-ccd",
        "scene-volume-dynamic-capsule-reverse-swept-ccd",
        "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
        "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
        "scene-volume-rotating-kinematic-convex-reverse-swept-ccd",
        "scene-volume-dynamic-rotating-convex-reverse-swept-ccd",
        "scene-volume-static-convex-reverse-swept-ccd",
        "scene-volume-kinematic-convex-reverse-swept-ccd",
        "scene-volume-dynamic-convex-reverse-swept-ccd",
    )
    scene_capsule_reverse_swept_ccd = (
        "-capsule-reverse-swept-ccd" in case_name
    )
    scene_convex_reverse_swept_ccd = (
        "-convex-reverse-swept-ccd" in case_name
    )
    scene_rotational_capsule_reverse_swept_ccd = case_name in (
        "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
        "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
    )
    scene_rotational_convex_reverse_swept_ccd = case_name in (
        "scene-volume-rotating-kinematic-convex-reverse-swept-ccd",
        "scene-volume-dynamic-rotating-convex-reverse-swept-ccd",
    )
    scene_rotational_finite_reverse_swept_ccd = (
        scene_rotational_capsule_reverse_swept_ccd
        or scene_rotational_convex_reverse_swept_ccd
    )
    scene_deforming_volume_reverse_swept_ccd = case_name in (
        "scene-volume-deforming-sphere-reverse-swept-ccd",
        "scene-volume-deforming-capsule-reverse-swept-ccd",
        "scene-volume-deforming-convex-reverse-swept-ccd",
    )
    scene_static_sphere_reverse_swept_ccd = (
        case_name
        in (
            "scene-volume-deforming-sphere-reverse-swept-ccd",
            "scene-volume-deforming-capsule-reverse-swept-ccd",
            "scene-volume-deforming-convex-reverse-swept-ccd",
            "scene-volume-static-sphere-reverse-swept-ccd",
            "scene-volume-static-capsule-reverse-swept-ccd",
            "scene-volume-static-convex-reverse-swept-ccd",
        )
    )
    scene_kinematic_sphere_reverse_swept_ccd = (
        case_name
        in (
            "scene-volume-kinematic-sphere-reverse-swept-ccd",
            "scene-volume-kinematic-capsule-reverse-swept-ccd",
            "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
            "scene-volume-rotating-kinematic-convex-reverse-swept-ccd",
            "scene-volume-kinematic-convex-reverse-swept-ccd",
        )
    )
    scene_dynamic_sphere_reverse_swept_ccd = (
        case_name
        in (
            "scene-volume-dynamic-sphere-reverse-swept-ccd",
            "scene-volume-dynamic-capsule-reverse-swept-ccd",
            "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
            "scene-volume-dynamic-rotating-convex-reverse-swept-ccd",
            "scene-volume-dynamic-convex-reverse-swept-ccd",
        )
    )
    scene_moving_sphere_reverse_swept_ccd = (
        scene_kinematic_sphere_reverse_swept_ccd
        or scene_dynamic_sphere_reverse_swept_ccd
    )
    scene_sphere_reverse_feature = (
        case_name == "scene-volume-sphere-reverse-feature"
    )
    scene_capsule_reverse_feature = (
        case_name == "scene-volume-capsule-reverse-feature"
    )
    scene_convex_reverse_feature = (
        case_name == "scene-volume-convex-reverse-feature"
    )
    scene_triangle_mesh_reverse_feature = (
        case_name == "scene-volume-triangle-mesh-reverse-feature"
    )
    scene_heightfield_reverse_feature = (
        case_name == "scene-volume-heightfield-reverse-feature"
    )
    scene_smooth_reverse_feature = (
        scene_sphere_reverse_feature
        or scene_capsule_reverse_feature
        or scene_convex_reverse_feature
        or scene_triangle_mesh_reverse_feature
        or scene_heightfield_reverse_feature
    )
    scene_speculative_ccd = (
        scene_triangle_surface_swept_ccd
        or case_name
        in (
            "scene-volume-speculative-ccd",
            "scene-volume-plane-speculative-ccd",
            "scene-volume-sphere-speculative-ccd",
            "scene-volume-capsule-speculative-ccd",
            "scene-volume-convex-speculative-ccd",
            "scene-volume-moving-kinematic-sphere-speculative-ccd",
            "scene-volume-moving-kinematic-capsule-speculative-ccd",
            "scene-volume-rotating-kinematic-capsule-speculative-ccd",
            "scene-volume-rotating-kinematic-convex-speculative-ccd",
            "scene-volume-moving-kinematic-convex-speculative-ccd",
            "scene-volume-dynamic-sphere-relative-swept-ccd",
            "scene-volume-dynamic-capsule-relative-swept-ccd",
            "scene-volume-dynamic-rotating-capsule-relative-swept-ccd",
            "scene-volume-dynamic-rotating-convex-relative-swept-ccd",
            "scene-volume-dynamic-convex-relative-swept-ccd",
            "scene-volume-deforming-sphere-reverse-swept-ccd",
            "scene-volume-deforming-capsule-reverse-swept-ccd",
            "scene-volume-deforming-convex-reverse-swept-ccd",
            "scene-volume-static-sphere-reverse-swept-ccd",
            "scene-volume-kinematic-sphere-reverse-swept-ccd",
            "scene-volume-dynamic-sphere-reverse-swept-ccd",
            "scene-volume-static-capsule-reverse-swept-ccd",
            "scene-volume-kinematic-capsule-reverse-swept-ccd",
            "scene-volume-dynamic-capsule-reverse-swept-ccd",
            "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
            "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
            "scene-volume-rotating-kinematic-convex-reverse-swept-ccd",
            "scene-volume-dynamic-rotating-convex-reverse-swept-ccd",
            "scene-volume-static-convex-reverse-swept-ccd",
            "scene-volume-kinematic-convex-reverse-swept-ccd",
            "scene-volume-dynamic-convex-reverse-swept-ccd",
        )
    )
    scene_skinning = case_name == "scene-volume-skinning"
    scene_static = (
        scene_static
        or scene_element_filter
        or scene_max_depenetration_velocity
        or scene_smooth_reverse_feature
        or (
            scene_speculative_ccd
            and not scene_moving_kinematic_finite_ccd
            and not scene_dynamic_finite_swept_ccd
            and not scene_moving_sphere_reverse_swept_ccd
            and not scene_kinematic_triangle_surface_swept_ccd
        )
    )
    scene_two_soft = (
        scene_multi_soft
        or scene_mixed_sleep
        or scene_soft_churn
        or scene_multi_scene
        or scene_soft_soft_wake
        or scene_soft_pair_attachment
        or scene_motion_controls
        or scene_max_depenetration_velocity
        or scene_speculative_ccd
        or scene_smooth_reverse_feature
    )
    scene_two_dynamic = scene_multi_dynamic or scene_multi_soft
    scene_dynamic = (
        case_name == "scene-volume-dynamic-box"
        or case_name == "scene-volume-dynamic-sphere"
        or case_name == "scene-volume-dynamic-capsule"
        or case_name == "scene-volume-dynamic-convex"
        or scene_dynamic_churn
        or scene_two_dynamic
    )
    scene_integrated = (
        scene_lifecycle
        or scene_static
        or scene_dynamic
        or scene_soft_sleep_wake
        or scene_soft_rigid_wake
        or scene_mixed_sleep
        or scene_soft_churn
        or scene_buffer_mutation
        or scene_world_pin
        or scene_attachment
        or scene_element_filter
        or scene_kinematic
        or scene_multi_scene
        or scene_soft_soft_wake
        or scene_soft_pair_attachment
        or scene_volume_kinematic_target
        or scene_skinning
        or scene_motion_controls
        or scene_max_depenetration_velocity
        or scene_speculative_ccd
        or scene_smooth_reverse_feature
    )
    required = {
        "schema": "1",
        "snippet": "SnippetDeformableVolumeAVBD",
        "case": case_name,
        "solver": "avbd",
        "validation": (
            "SCENE_MIXED_SLEEP_ISLANDS_GATED"
            if scene_mixed_sleep
            else (
                "SCENE_SOFT_RIGID_WAKE_GATED"
                if scene_soft_rigid_wake
                else (
                    "SCENE_SOFT_SLEEP_WAKE_GATED"
                    if scene_soft_sleep_wake
                    else (
                        "SCENE_LIFECYCLE_GATED"
                        if scene_lifecycle
                        else (
                            "SCENE_STATIC_LIFECYCLE_GATED"
                            if scene_static_churn
                            else (
                                (
                                    (
                                        "SCENE_DYNAMIC_LIFECYCLE_GATED"
                                        if scene_dynamic_churn
                                        else (
                                            (
                                                "SCENE_MULTI_SOFT_ISLANDS_GATED"
                                                if scene_multi_soft
                                                else (
                                                    "SCENE_MULTI_DYNAMIC_COUPLING_GATED"
                                                )
                                            )
                                            if scene_two_dynamic
                                            else "SCENE_DYNAMIC_COUPLING_GATED"
                                        )
                                    )
                                    if scene_dynamic
                                    else (
                                        "SCENE_STATIC_CONTACT_GATED"
                                        if scene_static
                                        else "COMPONENT_GATED"
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "sceneSoftIntegration": "1" if scene_integrated else "0",
        "status": "PASS",
        "initialized": "1",
        "frames": str(frames),
        "fetchFailures": "0",
        "sceneDynamics": (
            "2"
            if (
                scene_two_dynamic
                or scene_kinematic_triangle_surface_swept_ccd
            )
            else (
                "1"
                if (
                    scene_dynamic
                    or scene_soft_rigid_wake
                    or scene_kinematic
                    or scene_rigid_attachment
                    or scene_kinematic_attachment
                )
                else "0"
            )
        ),
        "sceneDeformableVolumes": (
            "2"
            if scene_two_soft
            else ("1" if scene_integrated else "0")
        ),
        "nonFiniteParticleSamples": "0",
        "invertedElementSamples": "0",
        "invalidContactSourceSamples": "0",
        "solverReadbackMatched": "1",
        "fatalErrors": "0",
        "cleanupComplete": "1",
    }
    if scene_soft_churn:
        required["validation"] = "SCENE_SOFT_CHURN_GATED"
    if scene_buffer_mutation:
        required["validation"] = "SCENE_BUFFER_MUTATION_GATED"
    if scene_world_pin:
        required["validation"] = "SCENE_WORLD_PIN_GATED"
    if scene_rigid_attachment:
        required["validation"] = "SCENE_RIGID_ATTACHMENT_GATED"
    if scene_static_attachment:
        required["validation"] = "SCENE_STATIC_ATTACHMENT_GATED"
    if scene_kinematic_attachment:
        required["validation"] = "SCENE_KINEMATIC_ATTACHMENT_GATED"
    if scene_articulation_attachment:
        required["validation"] = "SCENE_ARTICULATION_ATTACHMENT_GATED"
    if scene_element_filter:
        required["validation"] = (
            "SCENE_PARTIAL_ELEMENT_FILTER_GATED"
            if scene_partial_element_filter
            else "SCENE_ELEMENT_FILTER_GATED"
        )
    if scene_kinematic:
        required["validation"] = "SCENE_KINEMATIC_COUPLING_GATED"
    if scene_multi_scene:
        required["validation"] = "SCENE_MULTI_SCENE_ISOLATION_GATED"
    if scene_soft_soft_wake:
        required["validation"] = "SCENE_SOFT_SOFT_WAKE_GATED"
    if scene_soft_pair_attachment:
        required["validation"] = "SCENE_SOFT_PAIR_ATTACHMENT_GATED"
    if scene_volume_kinematic_target:
        required["validation"] = (
            "SCENE_VOLUME_KINEMATIC_TARGET_GATED"
        )
    if scene_motion_controls:
        required["validation"] = (
            "SCENE_DEFORMABLE_MOTION_CONTROLS_GATED"
        )
    if scene_max_depenetration_velocity:
        required["validation"] = (
            "SCENE_MAX_DEPENETRATION_VELOCITY_GATED"
        )
    if scene_speculative_ccd:
        required["validation"] = "SCENE_SPECULATIVE_CCD_GATED"
    if scene_sphere_reverse_feature:
        required["validation"] = (
            "SCENE_SPHERE_REVERSE_FEATURE_GATED"
        )
    if scene_capsule_reverse_feature:
        required["validation"] = (
            "SCENE_CAPSULE_REVERSE_FEATURE_GATED"
        )
    if scene_convex_reverse_feature:
        required["validation"] = (
            "SCENE_CONVEX_REVERSE_FEATURE_GATED"
        )
    if scene_skinning:
        required["validation"] = "SCENE_CPU_SKINNING_GATED"
    if scene_integrated:
        required.update(
            {
                "sceneStatics": (
                    "1"
                    if (
                        scene_static
                        or scene_static_attachment
                        or case_name == "scene-volume-dynamic-sphere"
                        or case_name == "scene-volume-dynamic-capsule"
                        or case_name == "scene-volume-dynamic-convex"
                    )
                    else "0"
                ),
                "softBodies": "2" if scene_two_soft else "1",
                "sceneActorCreated": "1",
                "sceneShapeAttached": "1",
                "sceneSimulationMeshAttached": "1",
                "sceneHostBuffersInitialized": "1",
                "sceneActorAdded": "1",
                "sceneActorRemoved": "1",
                "sceneActorReleased": "1",
                "sceneBoundsFinite": "1",
            }
        )
    if scene_dynamic:
        required.update(
            {
                "sceneDynamicActorAdded": "1",
                "sceneDynamicActorReleased": "1",
                "sceneDynamicInitiallySleeping": "1",
                "sceneDynamicWokeBySoft": "1",
            }
        )
    if scene_speculative_ccd:
        required.update(
            {
                "sceneSecondVolumeActorCreated": "1",
                "sceneSecondVolumeHostBuffersInitialized": "1",
                "sceneSecondVolumeActorAdded": "1",
                "sceneSecondVolumeActorRemoved": "1",
                "sceneSecondVolumeActorReleased": "1",
                "sceneSecondVolumeBoundsFinite": "1",
                "speculativeCcdFlagApplied": "1",
                "speculativeCcdPreventedTunneling": "1",
            }
        )
        if scene_triangle_surface_swept_ccd:
            required.update(
                {
                    "sceneStatics": (
                        "1"
                        if scene_static_triangle_surface_swept_ccd
                        else "0"
                    ),
                    "sceneDynamics": (
                        "0"
                        if scene_static_triangle_surface_swept_ccd
                        else "2"
                    ),
                    "speculativeCcdNegativeControlTunneled": "1",
                }
            )
            if scene_kinematic_triangle_surface_swept_ccd:
                required.update(
                    {
                        "sceneDynamicActorAdded": "1",
                        "sceneSecondDynamicActorAdded": "1",
                        "sceneDynamicActorRemoved": "1",
                        "sceneSecondDynamicActorRemoved": "1",
                        "sceneDynamicActorReleased": "1",
                        "sceneSecondDynamicActorReleased": "1",
                    }
                )
            swept_lines = (
                triangle_surface_reverse_swept_lines
                if scene_triangle_surface_reverse_swept_ccd
                else triangle_surface_forward_swept_lines
            )
            if len(swept_lines) != 1:
                errors.append(
                    "triangle-surface swept gate count is "
                    f"{len(swept_lines)}, expected exactly 1"
                )
                triangle_swept_fields: dict[str, str] = {}
            else:
                triangle_swept_fields, parse_errors = parse_gate(
                    swept_lines[0]
                )
                errors.extend(parse_errors)
            expected_target = (
                "static"
                if scene_static_triangle_surface_swept_ccd
                else "kinematic"
            )
            expected_geometry = (
                "heightfield"
                if scene_heightfield_swept_ccd
                else "triangle-mesh"
            )
            for key, expected in (
                ("frames", str(frames)),
                ("target", expected_target),
                ("geometry", expected_geometry),
                ("responseObserved", "1"),
                ("negativeControlPassed", "1"),
                ("vertexSweepExcluded", "1"),
                ("nonFiniteSamples", "0"),
                ("result", "PASS"),
            ):
                if triangle_swept_fields.get(key) != expected:
                    errors.append(
                        f"triangle-surface swept {key}="
                        f"{triangle_swept_fields.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            triangle_swept_values: dict[str, float] = {}
            for key in (
                "positiveDisplacement",
                "negativeDisplacement",
                "positiveDrop",
                "negativeDrop",
                "minimumVertexSweepSeparation",
            ):
                try:
                    value = float(triangle_swept_fields[key])
                    if not math.isfinite(value):
                        raise ValueError
                    triangle_swept_values[key] = value
                    fields[f"triangleSurfaceSwept.{key}"] = (
                        triangle_swept_fields[key]
                    )
                except (KeyError, ValueError):
                    errors.append(
                        f"triangle-surface swept {key}="
                        f"{triangle_swept_fields.get(key)!r}, "
                        "expected finite float"
                    )
            positive_displacement = triangle_swept_values.get(
                "positiveDisplacement"
            )
            negative_displacement = triangle_swept_values.get(
                "negativeDisplacement"
            )
            positive_drop = triangle_swept_values.get("positiveDrop")
            negative_drop = triangle_swept_values.get("negativeDrop")
            vertex_separation = triangle_swept_values.get(
                "minimumVertexSweepSeparation"
            )
            if scene_static_triangle_surface_swept_ccd:
                minimum_negative_drop = (
                    0.8
                    if (
                        scene_triangle_surface_reverse_swept_ccd
                        and scene_heightfield_swept_ccd
                    )
                    else 1.5
                )
                if (
                    negative_drop is not None
                    and negative_drop <= minimum_negative_drop
                ):
                    errors.append(
                        "flag-off static triangle-surface volume "
                        "did not tunnel"
                    )
                control_separation = (
                    0.01
                    if scene_triangle_surface_reverse_swept_ccd
                    else 0.10
                )
                if (
                    positive_drop is not None
                    and negative_drop is not None
                    and positive_drop + control_separation
                    >= negative_drop
                ):
                    errors.append(
                        "static triangle-surface sweep did not "
                        "separate controls"
                    )
            else:
                response_threshold = (
                    0.002
                    if scene_rotational_triangle_surface_swept_ccd
                    else (
                        0.005
                        if scene_triangle_surface_reverse_swept_ccd
                        else 0.02
                    )
                )
                if (
                    positive_displacement is not None
                    and positive_displacement <= response_threshold
                ):
                    errors.append(
                        "moving triangle surface did not move "
                        "the swept volume"
                    )
                if (
                    negative_displacement is not None
                    and negative_displacement >= 0.01
                ):
                    errors.append(
                        "flag-off kinematic triangle-surface "
                        "volume moved"
                    )
            if (
                scene_triangle_surface_reverse_swept_ccd
                and vertex_separation is not None
                and vertex_separation
                <= (
                    0.10
                    if scene_rotational_triangle_surface_swept_ccd
                    else 0.05
                )
            ):
                errors.append(
                    "forward volume vertex sweep was not "
                    "geometrically excluded"
                )
            if scene_rotational_triangle_surface_swept_ccd:
                if len(triangle_surface_rotational_swept_lines) != 1:
                    errors.append(
                        "triangle-surface rotational swept gate count is "
                        f"{len(triangle_surface_rotational_swept_lines)}, "
                        "expected exactly 1"
                    )
                    rotational_fields: dict[str, str] = {}
                else:
                    rotational_fields, parse_errors = parse_gate(
                        triangle_surface_rotational_swept_lines[0]
                    )
                    errors.extend(parse_errors)
                expected_owner = (
                    "reverse"
                    if scene_triangle_surface_reverse_swept_ccd
                    else "forward"
                )
                for key, expected in (
                    ("frames", str(frames)),
                    ("target", "kinematic"),
                    ("geometry", expected_geometry),
                    ("owner", expected_owner),
                    ("responseObserved", "1"),
                    ("negativeControlPassed", "1"),
                    ("vertexSweepExcluded", "1"),
                    ("result", "PASS"),
                ):
                    if rotational_fields.get(key) != expected:
                        errors.append(
                            f"triangle-surface rotational {key}="
                            f"{rotational_fields.get(key)!r}, "
                            f"expected {expected!r}"
                        )
                rotational_values: dict[str, float] = {}
                for key in (
                    "endpointMinSeparation",
                    "midSweepMinSeparation",
                    "minimumVertexSweepSeparation",
                    "positiveDisplacement",
                    "negativeDisplacement",
                    "positiveAngularTravel",
                    "negativeAngularTravel",
                ):
                    try:
                        value = float(rotational_fields[key])
                        if not math.isfinite(value):
                            raise ValueError
                        rotational_values[key] = value
                        fields[
                            f"triangleSurfaceRotationalSwept.{key}"
                        ] = rotational_fields[key]
                    except (KeyError, ValueError):
                        errors.append(
                            f"triangle-surface rotational {key}="
                            f"{rotational_fields.get(key)!r}, "
                            "expected finite float"
                        )
                endpoint_separation = rotational_values.get(
                    "endpointMinSeparation"
                )
                mid_separation = rotational_values.get(
                    "midSweepMinSeparation"
                )
                if (
                    endpoint_separation is not None
                    and endpoint_separation <= 0.10
                ):
                    errors.append(
                        "triangle-surface rotational endpoints overlap "
                        "the volume"
                    )
                if mid_separation is not None and mid_separation >= (
                    -0.05
                    if scene_triangle_surface_reverse_swept_ccd
                    else 0.01
                ):
                    errors.append(
                        "triangle-surface rotational arc did not hit "
                        "between endpoints"
                    )
                expected_travel = 2.0 * math.pi / 3.0
                for key in (
                    "positiveAngularTravel",
                    "negativeAngularTravel",
                ):
                    value = rotational_values.get(key)
                    if (
                        value is not None
                        and abs(value - expected_travel) > 0.002
                    ):
                        errors.append(
                            f"triangle-surface rotational {key}={value} "
                            f"did not reach {expected_travel}"
                        )
        elif scene_sphere_reverse_swept_ccd:
            required.update(
                {
                    "sceneStatics": (
                        "1"
                        if scene_static_sphere_reverse_swept_ccd
                        else "0"
                    ),
                    "sceneDynamics": (
                        "0"
                        if scene_static_sphere_reverse_swept_ccd
                        else "2"
                    ),
                }
            )
            if scene_moving_sphere_reverse_swept_ccd:
                required.update(
                    {
                        "sceneDynamicActorAdded": "1",
                        "sceneSecondDynamicActorAdded": "1",
                        "sceneDynamicActorRemoved": "1",
                        "sceneSecondDynamicActorRemoved": "1",
                        "sceneDynamicActorReleased": "1",
                        "sceneSecondDynamicActorReleased": "1",
                    }
                )
            reverse_swept_lines = (
                convex_reverse_swept_lines
                if scene_convex_reverse_swept_ccd
                else (
                    capsule_reverse_swept_lines
                    if scene_capsule_reverse_swept_ccd
                    else sphere_reverse_swept_lines
                )
            )
            geometry_name = (
                "convex"
                if scene_convex_reverse_swept_ccd
                else (
                    "capsule"
                    if scene_capsule_reverse_swept_ccd
                    else "sphere"
                )
            )
            if len(reverse_swept_lines) != 1:
                errors.append(
                    f"{geometry_name} reverse-swept gate count is "
                    f"{len(reverse_swept_lines)}, "
                    "expected exactly 1"
                )
                reverse_swept_fields: dict[str, str] = {}
            else:
                reverse_swept_fields, parse_errors = parse_gate(
                    reverse_swept_lines[0]
                )
                errors.extend(parse_errors)
            target = (
                "static"
                if scene_static_sphere_reverse_swept_ccd
                else (
                    "kinematic"
                    if scene_kinematic_sphere_reverse_swept_ccd
                    else "dynamic"
                )
            )
            for key, expected in (
                ("frames", str(frames)),
                ("target", target),
                ("responseObserved", "1"),
                ("negativeControlPassed", "1"),
                ("twoSidedResponseObserved", "1"),
                ("vertexSweepExcluded", "1"),
                ("nonFiniteSamples", "0"),
                ("result", "PASS"),
            ):
                if reverse_swept_fields.get(key) != expected:
                    errors.append(
                        f"{geometry_name} reverse-swept {key}="
                        f"{reverse_swept_fields.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            reverse_swept_values: dict[str, float] = {}
            for key in (
                "positiveDisplacement",
                "negativeDisplacement",
                "positiveDrop",
                "negativeDrop",
                "positiveRigidDrop",
                "negativeRigidDrop",
                "faceSeparation",
                "minimumVertexSweepSeparation",
            ):
                try:
                    value = float(reverse_swept_fields[key])
                    if not math.isfinite(value):
                        raise ValueError
                    reverse_swept_values[key] = value
                    fields[f"sphereReverseSwept.{key}"] = (
                        reverse_swept_fields[key]
                    )
                except (KeyError, ValueError):
                    errors.append(
                        f"{geometry_name} reverse-swept {key}="
                        f"{reverse_swept_fields.get(key)!r}, "
                        "expected finite float"
                    )
            positive_displacement = reverse_swept_values.get(
                "positiveDisplacement"
            )
            negative_displacement = reverse_swept_values.get(
                "negativeDisplacement"
            )
            positive_drop = reverse_swept_values.get("positiveDrop")
            negative_drop = reverse_swept_values.get("negativeDrop")
            positive_rigid_drop = reverse_swept_values.get(
                "positiveRigidDrop"
            )
            negative_rigid_drop = reverse_swept_values.get(
                "negativeRigidDrop"
            )
            face_separation = reverse_swept_values.get(
                "faceSeparation"
            )
            vertex_sweep_separation = reverse_swept_values.get(
                "minimumVertexSweepSeparation"
            )
            if scene_static_sphere_reverse_swept_ccd:
                if negative_drop is not None and negative_drop <= 0.8:
                    errors.append(
                        "flag-off static reverse-swept volume "
                        "did not tunnel"
                    )
                if (
                    positive_drop is not None
                    and negative_drop is not None
                    and positive_drop + 0.03 >= negative_drop
                ):
                    errors.append(
                        f"static {geometry_name} reverse sweep did not "
                        "separate from its flag-off control"
                    )
            else:
                displacement_floor = (
                    0.02
                    if scene_rotational_finite_reverse_swept_ccd
                    else (
                    0.01
                    if scene_dynamic_sphere_reverse_swept_ccd
                    else 0.02
                    )
                )
                if (
                    positive_displacement is not None
                    and positive_displacement <= displacement_floor
                ):
                    errors.append(
                        f"moving {geometry_name} reverse sweep did not move "
                        "the positive volume"
                    )
                if (
                    negative_displacement is not None
                    and negative_displacement >= 0.005
                ):
                    errors.append(
                        f"flag-off moving {geometry_name} reverse-swept "
                        "volume moved"
                    )
            if scene_dynamic_sphere_reverse_swept_ccd:
                if (
                    negative_rigid_drop is not None
                    and negative_rigid_drop
                    <= (
                        0.8
                        if scene_rotational_finite_reverse_swept_ccd
                        else 1.5
                    )
                ):
                    errors.append(
                        f"flag-off dynamic reverse-swept {geometry_name} "
                        "did not tunnel ballistically"
                    )
                if (
                    positive_rigid_drop is not None
                    and negative_rigid_drop is not None
                    and positive_rigid_drop + 0.05
                    >= negative_rigid_drop
                ):
                    errors.append(
                        f"dynamic {geometry_name} reverse sweep did not "
                        "produce a two-sided response"
                    )
            if face_separation is not None and face_separation <= -0.15:
                errors.append(
                    f"{geometry_name} reverse sweep left excessive "
                    "face penetration"
                )
            if (
                vertex_sweep_separation is not None
                and vertex_sweep_separation
                <= (
                    0.10
                    if scene_rotational_finite_reverse_swept_ccd
                    else (
                    0.05
                    if (
                        scene_capsule_reverse_swept_ccd
                        or scene_convex_reverse_swept_ccd
                    )
                    else 0.10
                    )
                )
            ):
                errors.append(
                    f"{geometry_name} reverse sweep did not geometrically "
                    "exclude every vertex path"
                )
            if scene_deforming_volume_reverse_swept_ccd:
                if len(deforming_volume_reverse_swept_lines) != 1:
                    errors.append(
                        "deforming-volume reverse-swept gate count is "
                        f"{len(deforming_volume_reverse_swept_lines)}, "
                        "expected exactly 1"
                    )
                    deforming_fields: dict[str, str] = {}
                else:
                    deforming_fields, parse_errors = parse_gate(
                        deforming_volume_reverse_swept_lines[0]
                    )
                    errors.extend(parse_errors)
                for key, expected in (
                    ("frames", str(frames)),
                    ("geometry", geometry_name),
                    ("target", "static"),
                    ("owner", "reverse"),
                    ("responseObserved", "1"),
                    ("negativeControlPassed", "1"),
                    ("geometricSweepIsolated", "1"),
                    ("vertexSweepExcluded", "1"),
                    ("nonFiniteSamples", "0"),
                    ("result", "PASS"),
                ):
                    if deforming_fields.get(key) != expected:
                        errors.append(
                            f"deforming-volume reverse-swept {key}="
                            f"{deforming_fields.get(key)!r}, "
                            f"expected {expected!r}"
                        )
                deforming_values: dict[str, float] = {}
                for key in (
                    "endpointMinSeparation",
                    "midSweepMinSeparation",
                    "minimumVertexSweepSeparation",
                    "responseDelta",
                    "positiveDrop",
                    "negativeDrop",
                ):
                    try:
                        value = float(deforming_fields[key])
                        if not math.isfinite(value):
                            raise ValueError
                        deforming_values[key] = value
                        fields[
                            f"deformingVolumeReverseSwept.{key}"
                        ] = deforming_fields[key]
                    except (KeyError, ValueError):
                        errors.append(
                            f"deforming-volume reverse-swept {key}="
                            f"{deforming_fields.get(key)!r}, "
                            "expected finite float"
                        )
                endpoint_separation = deforming_values.get(
                    "endpointMinSeparation"
                )
                mid_separation = deforming_values.get(
                    "midSweepMinSeparation"
                )
                deforming_vertex_separation = deforming_values.get(
                    "minimumVertexSweepSeparation"
                )
                response_delta = deforming_values.get("responseDelta")
                deforming_positive_drop = deforming_values.get(
                    "positiveDrop"
                )
                deforming_negative_drop = deforming_values.get(
                    "negativeDrop"
                )
                if (
                    endpoint_separation is not None
                    and endpoint_separation <= 0.02
                ):
                    errors.append(
                        "deforming-volume reverse-swept endpoints "
                        "overlap the rigid target"
                    )
                if (
                    mid_separation is not None
                    and mid_separation
                    >= (0.02 if geometry_name == "convex" else 0.0)
                ):
                    errors.append(
                        "deforming-volume boundary does not cross the "
                        "rigid target between endpoints"
                    )
                if (
                    deforming_vertex_separation is not None
                    and deforming_vertex_separation <= 0.05
                ):
                    errors.append(
                        "deforming-volume reverse sweep was not isolated "
                        "from every vertex path"
                    )
                if response_delta is not None and response_delta <= 0.01:
                    errors.append(
                        "deforming-volume flag-on/off response delta "
                        "was not observed"
                    )
                if (
                    deforming_positive_drop is not None
                    and deforming_negative_drop is not None
                    and deforming_positive_drop + 0.01
                    >= deforming_negative_drop
                ):
                    errors.append(
                        "deforming-volume flag-on result did not separate "
                        "from its flag-off control"
                    )
            if scene_rotational_finite_reverse_swept_ccd:
                rotational_geometry = (
                    "convex"
                    if scene_rotational_convex_reverse_swept_ccd
                    else "capsule"
                )
                rotational_reverse_swept_lines = (
                    convex_rotational_reverse_swept_lines
                    if scene_rotational_convex_reverse_swept_ccd
                    else capsule_rotational_reverse_swept_lines
                )
                if len(rotational_reverse_swept_lines) != 1:
                    errors.append(
                        f"{rotational_geometry} rotational reverse-swept "
                        "gate count is "
                        f"{len(rotational_reverse_swept_lines)}, "
                        "expected exactly 1"
                    )
                    rotational_fields: dict[str, str] = {}
                else:
                    rotational_fields, parse_errors = parse_gate(
                        rotational_reverse_swept_lines[0]
                    )
                    errors.extend(parse_errors)
                for key, expected in (
                    ("frames", str(frames)),
                    ("target", target),
                    ("owner", "reverse"),
                    ("responseObserved", "1"),
                    ("negativeControlPassed", "1"),
                    ("twoSidedResponseObserved", "1"),
                    ("vertexSweepExcluded", "1"),
                    ("result", "PASS"),
                ):
                    if rotational_fields.get(key) != expected:
                        errors.append(
                            f"{rotational_geometry} rotational "
                            "reverse-swept "
                            f"{key}={rotational_fields.get(key)!r}, "
                            f"expected {expected!r}"
                        )
                rotational_values: dict[str, float] = {}
                for key in (
                    "endpointMinSeparation",
                    "midSweepMinSeparation",
                    "positiveDisplacement",
                    "negativeDisplacement",
                    "positiveAngularTravel",
                    "negativeAngularTravel",
                ):
                    try:
                        value = float(rotational_fields[key])
                        if not math.isfinite(value):
                            raise ValueError
                        rotational_values[key] = value
                        fields[
                            f"{rotational_geometry}RotationalReverseSwept."
                            f"{key}"
                        ] = rotational_fields[key]
                    except (KeyError, ValueError):
                        errors.append(
                            f"{rotational_geometry} rotational "
                            "reverse-swept "
                            f"{key}={rotational_fields.get(key)!r}, "
                            "expected finite float"
                        )
                endpoint_separation = rotational_values.get(
                    "endpointMinSeparation"
                )
                mid_separation = rotational_values.get(
                    "midSweepMinSeparation"
                )
                rotational_positive_displacement = rotational_values.get(
                    "positiveDisplacement"
                )
                rotational_negative_displacement = rotational_values.get(
                    "negativeDisplacement"
                )
                positive_angular_travel = rotational_values.get(
                    "positiveAngularTravel"
                )
                negative_angular_travel = rotational_values.get(
                    "negativeAngularTravel"
                )
                if (
                    endpoint_separation is not None
                    and endpoint_separation <= 0.05
                ):
                    errors.append(
                        f"{rotational_geometry} rotational reverse-swept "
                        "endpoints "
                        "overlap the volume boundary"
                    )
                if (
                    mid_separation is not None
                    and mid_separation >= -0.05
                ):
                    errors.append(
                        f"{rotational_geometry} rotational reverse sweep "
                        "does not cross "
                        "the volume boundary between endpoints"
                    )
                if (
                    rotational_positive_displacement is not None
                    and rotational_positive_displacement <= 0.02
                ):
                    errors.append(
                        f"{rotational_geometry} rotational reverse owner "
                        "did not move "
                        "the positive volume"
                    )
                if (
                    rotational_negative_displacement is not None
                    and rotational_negative_displacement >= 0.005
                ):
                    errors.append(
                        f"flag-off {rotational_geometry} rotational "
                        "reverse control moved"
                    )
                if target == "kinematic":
                    expected_travel = 2.0 * math.pi / 3.0
                    for key, value in (
                        (
                            "positiveAngularTravel",
                            positive_angular_travel,
                        ),
                        (
                            "negativeAngularTravel",
                            negative_angular_travel,
                        ),
                    ):
                        if (
                            value is not None
                            and abs(value - expected_travel) > 0.002
                        ):
                            errors.append(
                                f"kinematic {key}={value} did not "
                                f"reach target {expected_travel}"
                            )
                elif (
                    negative_angular_travel is not None
                    and negative_angular_travel <= 0.8
                ):
                    errors.append(
                        f"flag-off dynamic {rotational_geometry} did not "
                        "rotate "
                        "ballistically"
                    )
                elif (
                    positive_angular_travel is not None
                    and negative_angular_travel is not None
                    and positive_angular_travel + 0.05
                    >= negative_angular_travel
                ):
                    errors.append(
                        f"dynamic {rotational_geometry} rotational reverse "
                        "sweep lacked "
                        "two-sided angular response"
                    )
        elif scene_dynamic_finite_swept_ccd:
            required.update(
                {
                    "sceneStatics": "0",
                    "sceneDynamics": "2",
                    "sceneDynamicActorAdded": "1",
                    "sceneSecondDynamicActorAdded": "1",
                    "sceneDynamicActorRemoved": "1",
                    "sceneSecondDynamicActorRemoved": "1",
                    "sceneDynamicActorReleased": "1",
                    "sceneSecondDynamicActorReleased": "1",
                    "dynamicSphereSweepLaunched": "1",
                    "dynamicSphereSweepResponseObserved": "1",
                    "dynamicSphereSweepNegativeControlTunneled": "1",
                    "dynamicSphereSweepTwoSidedResponseObserved": "1",
                }
            )
            try:
                positive_soft_displacement = float(
                    fields["dynamicSphereSweepPositiveSoftDisplacement"]
                )
                negative_soft_displacement = float(
                    fields["dynamicSphereSweepNegativeSoftDisplacement"]
                )
                positive_rigid_drop = float(
                    fields["dynamicSphereSweepPositiveRigidDrop"]
                )
                negative_rigid_drop = float(
                    fields["dynamicSphereSweepNegativeRigidDrop"]
                )
                min_separation = float(
                    fields["dynamicSphereSweepPositiveMinSeparation"]
                )
                if (
                    not math.isfinite(positive_soft_displacement)
                    or positive_soft_displacement <= 0.02
                ):
                    errors.append(
                        "dynamic finite-geometry sweep did not move the "
                        "positive volume"
                    )
                if (
                    not math.isfinite(negative_soft_displacement)
                    or negative_soft_displacement >= 0.005
                ):
                    errors.append(
                        "flag-off dynamic finite-geometry volume control moved"
                    )
                if (
                    not math.isfinite(negative_rigid_drop)
                    or negative_rigid_drop
                    <= (0.8 if scene_dynamic_rotating_finite_ccd else 1.5)
                ):
                    errors.append(
                        "flag-off dynamic finite geometry did not tunnel "
                        "ballistically"
                    )
                if (
                    not math.isfinite(positive_rigid_drop)
                    or positive_rigid_drop + 0.05 >= negative_rigid_drop
                ):
                    errors.append(
                        "dynamic finite-geometry sweep did not produce "
                        "a two-sided response"
                    )
                if (
                    not math.isfinite(min_separation)
                    or min_separation >= 1.0e30
                    or min_separation <= -0.15
                ):
                    errors.append(
                        "dynamic finite-geometry volume sweep missed "
                        "or penetrated"
                    )
            except (KeyError, ValueError):
                pass
            if scene_dynamic_rotating_finite_ccd:
                rotational_geometry = (
                    "convex"
                    if scene_dynamic_rotating_convex_ccd
                    else "capsule"
                )
                dynamic_rotational_swept_lines = (
                    convex_dynamic_rotational_swept_lines
                    if scene_dynamic_rotating_convex_ccd
                    else capsule_dynamic_rotational_swept_lines
                )
                if len(dynamic_rotational_swept_lines) != 1:
                    errors.append(
                        f"{rotational_geometry} dynamic rotational swept "
                        "gate count is "
                        f"{len(dynamic_rotational_swept_lines)}, "
                        "expected exactly 1"
                    )
                    dynamic_rotational_fields: dict[str, str] = {}
                else:
                    dynamic_rotational_fields, parse_errors = parse_gate(
                        dynamic_rotational_swept_lines[0]
                    )
                    errors.extend(parse_errors)
                for key, expected in (
                    ("frames", str(frames)),
                    ("target", "dynamic"),
                    ("owner", "forward"),
                    ("responseObserved", "1"),
                    ("negativeControlPassed", "1"),
                    ("twoSidedResponseObserved", "1"),
                    ("result", "PASS"),
                ):
                    if dynamic_rotational_fields.get(key) != expected:
                        errors.append(
                            f"{rotational_geometry} dynamic rotational "
                            f"swept {key}="
                            f"{dynamic_rotational_fields.get(key)!r}, "
                            f"expected {expected!r}"
                        )
                dynamic_rotational_values: dict[str, float] = {}
                for key in (
                    "endpointMinSeparation",
                    "midSweepMinSeparation",
                    "positiveDisplacement",
                    "negativeDisplacement",
                    "positiveAngularTravel",
                    "negativeAngularTravel",
                ):
                    try:
                        value = float(dynamic_rotational_fields[key])
                        if not math.isfinite(value):
                            raise ValueError
                        dynamic_rotational_values[key] = value
                        fields[
                            f"{rotational_geometry}DynamicRotational.{key}"
                        ] = (
                            dynamic_rotational_fields[key]
                        )
                    except (KeyError, ValueError):
                        errors.append(
                            f"{rotational_geometry} dynamic rotational "
                            f"swept {key}="
                            f"{dynamic_rotational_fields.get(key)!r}, "
                            "expected finite float"
                        )
                if (
                    dynamic_rotational_values.get(
                        "endpointMinSeparation", -math.inf
                    )
                    <= 0.05
                ):
                    errors.append(
                        f"{rotational_geometry} dynamic rotational swept "
                        "fixture endpoints "
                        "are not separated"
                    )
                if (
                    dynamic_rotational_values.get(
                        "midSweepMinSeparation", math.inf
                    )
                    >= (
                        1.0e-5
                        if scene_dynamic_rotating_convex_ccd
                        else -0.05
                    )
                ):
                    errors.append(
                        f"{rotational_geometry} dynamic rotational swept "
                        "fixture does not "
                        "isolate an intermediate arc hit"
                    )
                if (
                    dynamic_rotational_values.get(
                        "positiveDisplacement", -math.inf
                    )
                    <= 0.02
                ):
                    errors.append(
                        f"{rotational_geometry} dynamic rotational swept "
                        "flag-on response "
                        "was not observed"
                    )
                if (
                    dynamic_rotational_values.get(
                        "negativeDisplacement", math.inf
                    )
                    >= 0.005
                ):
                    errors.append(
                        f"{rotational_geometry} dynamic rotational swept "
                        "flag-off control "
                        "moved"
                    )
                positive_angular_travel = dynamic_rotational_values.get(
                    "positiveAngularTravel", math.inf
                )
                negative_angular_travel = dynamic_rotational_values.get(
                    "negativeAngularTravel", -math.inf
                )
                if negative_angular_travel <= 0.8:
                    errors.append(
                        f"flag-off dynamic rotating {rotational_geometry} "
                        "did not travel "
                        "ballistically"
                    )
                if positive_angular_travel + 0.05 >= negative_angular_travel:
                    errors.append(
                        f"dynamic rotating {rotational_geometry} did not "
                        "produce a "
                        "two-sided response"
                    )
        elif scene_moving_kinematic_finite_ccd:
            required.update(
                {
                    "sceneStatics": "0",
                    "sceneDynamics": "2",
                    "sceneDynamicActorAdded": "1",
                    "sceneSecondDynamicActorAdded": "1",
                    "sceneDynamicActorRemoved": "1",
                    "sceneSecondDynamicActorRemoved": "1",
                    "sceneDynamicActorReleased": "1",
                    "sceneSecondDynamicActorReleased": "1",
                    "movingSphereTargetIssued": "1",
                    "movingSphereCcdResponseObserved": "1",
                    "movingSphereNegativeControlHeld": "1",
                }
            )
            try:
                positive_displacement = float(
                    fields["movingSpherePositiveDisplacement"]
                )
                negative_displacement = float(
                    fields["movingSphereNegativeDisplacement"]
                )
                min_separation = float(
                    fields["movingSpherePositiveMinSeparation"]
                )
                if (
                    not math.isfinite(positive_displacement)
                    or positive_displacement <= 0.02
                ):
                    errors.append(
                        "moving finite geometry did not produce a volume "
                        "CCD response"
                    )
                if (
                    not math.isfinite(negative_displacement)
                    or negative_displacement >= 0.005
                ):
                    errors.append(
                        "flag-off moving finite-geometry volume control moved"
                    )
                if (
                    not math.isfinite(min_separation)
                    or min_separation >= 1.0e30
                    or min_separation <= -0.10
                ):
                    errors.append(
                        "moving finite-geometry volume response missed "
                        "or penetrated"
                    )
            except (KeyError, ValueError):
                pass
            if scene_rotating_kinematic_finite_ccd:
                rotational_geometry = (
                    "convex"
                    if scene_rotating_kinematic_convex_ccd
                    else "capsule"
                )
                rotational_swept_lines = (
                    convex_rotational_swept_lines
                    if scene_rotating_kinematic_convex_ccd
                    else capsule_rotational_swept_lines
                )
                if len(rotational_swept_lines) != 1:
                    errors.append(
                        f"{rotational_geometry} rotational swept gate "
                        "count is "
                        f"{len(rotational_swept_lines)}, "
                        "expected exactly 1"
                    )
                    rotational_fields: dict[str, str] = {}
                else:
                    rotational_fields, parse_errors = parse_gate(
                        rotational_swept_lines[0]
                    )
                    errors.extend(parse_errors)
                for key, expected in (
                    ("frames", str(frames)),
                    ("target", "kinematic"),
                    ("owner", "forward"),
                    ("responseObserved", "1"),
                    ("negativeControlPassed", "1"),
                    ("result", "PASS"),
                ):
                    if rotational_fields.get(key) != expected:
                        errors.append(
                            f"{rotational_geometry} rotational swept {key}="
                            f"{rotational_fields.get(key)!r}, "
                            f"expected {expected!r}"
                        )
                rotational_values: dict[str, float] = {}
                for key in (
                    "endpointMinSeparation",
                    "midSweepMinSeparation",
                    "positiveDisplacement",
                    "negativeDisplacement",
                ):
                    try:
                        value = float(rotational_fields[key])
                        if not math.isfinite(value):
                            raise ValueError
                        rotational_values[key] = value
                        fields[f"{rotational_geometry}Rotational.{key}"] = (
                            rotational_fields[key]
                        )
                    except (KeyError, ValueError):
                        errors.append(
                            f"{rotational_geometry} rotational swept {key}="
                            f"{rotational_fields.get(key)!r}, "
                            "expected finite float"
                        )
                if (
                    rotational_values.get(
                        "endpointMinSeparation", -math.inf
                    )
                    <= 0.05
                ):
                    errors.append(
                        f"{rotational_geometry} rotational swept fixture "
                        "endpoints "
                        "are not separated"
                    )
                if (
                    rotational_values.get(
                        "midSweepMinSeparation", math.inf
                    )
                    >= (
                        1.0e-5
                        if scene_rotating_kinematic_convex_ccd
                        else -0.05
                    )
                ):
                    errors.append(
                        f"{rotational_geometry} rotational swept fixture "
                        "does not "
                        "isolate an intermediate arc hit"
                    )
                if (
                    rotational_values.get(
                        "positiveDisplacement", -math.inf
                    )
                    <= 0.02
                ):
                    errors.append(
                        f"{rotational_geometry} rotational swept flag-on "
                        "response "
                        "was not observed"
                    )
                if (
                    rotational_values.get(
                        "negativeDisplacement", math.inf
                    )
                    >= 0.005
                ):
                    errors.append(
                        f"{rotational_geometry} rotational swept flag-off "
                        "control moved"
                    )
        else:
            plane_case = (
                case_name == "scene-volume-plane-speculative-ccd"
            )
            finite_smooth_case = case_name in (
                "scene-volume-sphere-speculative-ccd",
                "scene-volume-capsule-speculative-ccd",
                "scene-volume-convex-speculative-ccd",
            )
            if not plane_case:
                required["speculativeCcdNegativeControlTunneled"] = "1"
            try:
                positive_min_y = float(
                    fields["speculativeCcdPositiveMinY"]
                )
                positive_min_separation = float(
                    fields["speculativeCcdPositiveMinSeparation"]
                )
                negative_max_y = float(
                    fields["speculativeCcdNegativeMaxY"]
                )
                positive_floor = 0.49 if plane_case else 0.50
                if (
                    not finite_smooth_case
                    and positive_min_y < positive_floor
                ):
                    errors.append(
                        "speculative volume crossed the collision boundary"
                    )
                if (
                    finite_smooth_case
                    and (
                        not math.isfinite(positive_min_separation)
                        or positive_min_separation >= 1.0e30
                        or positive_min_separation < -0.05
                    )
                ):
                    errors.append(
                        "speculative volume finite-geometry separation "
                        "was missing "
                        "or penetrated the boundary"
                    )
                if not plane_case and negative_max_y > 0.44:
                    errors.append(
                        "discrete volume control did not tunnel"
                    )
            except (KeyError, ValueError):
                pass
    if scene_smooth_reverse_feature:
        required.update(
            {
                "sceneStatics": "1",
                "sceneDynamics": "0",
                "sceneSecondVolumeActorCreated": "1",
                "sceneSecondVolumeHostBuffersInitialized": "1",
                "sceneSecondVolumeActorAdded": "1",
                "sceneSecondVolumeActorRemoved": "1",
                "sceneSecondVolumeActorReleased": "1",
                "sceneSecondVolumeBoundsFinite": "1",
            }
        )
        reverse_feature_lines = (
            triangle_mesh_reverse_feature_lines
            if scene_triangle_mesh_reverse_feature
            else (
                heightfield_reverse_feature_lines
                if scene_heightfield_reverse_feature
                else (
                    convex_reverse_feature_lines
                    if scene_convex_reverse_feature
                    else (
                        capsule_reverse_feature_lines
                        if scene_capsule_reverse_feature
                        else sphere_reverse_feature_lines
                    )
                )
            )
        )
        geometry_name = (
            "triangle-mesh"
            if scene_triangle_mesh_reverse_feature
            else (
                "heightfield"
                if scene_heightfield_reverse_feature
                else (
                    "convex"
                    if scene_convex_reverse_feature
                    else (
                        "capsule"
                        if scene_capsule_reverse_feature
                        else "sphere"
                    )
                )
            )
        )
        if len(reverse_feature_lines) != 1:
            errors.append(
                f"{geometry_name} reverse-feature gate count is "
                f"{len(reverse_feature_lines)}, expected exactly 1"
            )
            reverse_fields: dict[str, str] = {}
        else:
            reverse_fields, parse_errors = parse_gate(
                reverse_feature_lines[0]
            )
            errors.extend(parse_errors)
        for key, expected in (
            ("frames", str(frames)),
            ("faceResponseObserved", "1"),
            ("vertexSdfExcluded", "1"),
            ("negativeControlPassed", "1"),
            ("nonFiniteSamples", "0"),
            ("result", "PASS"),
        ):
            if reverse_fields.get(key) != expected:
                errors.append(
                    f"{geometry_name} reverse-feature {key}="
                    f"{reverse_fields.get(key)!r}, expected {expected!r}"
                )
        reverse_values: dict[str, float] = {}
        for key in (
            "positiveDisplacement",
            "positiveDrop",
            "negativeDrop",
            "faceSeparation",
            "minimumVertexSeparation",
        ):
            try:
                value = float(reverse_fields[key])
                if not math.isfinite(value):
                    raise ValueError
                reverse_values[key] = value
                fields[f"{geometry_name}Reverse.{key}"] = (
                    reverse_fields[key]
                )
            except (KeyError, ValueError):
                errors.append(
                    f"{geometry_name} reverse-feature {key}="
                    f"{reverse_fields.get(key)!r}, expected finite float"
                )
        positive_displacement = reverse_values.get(
            "positiveDisplacement"
        )
        positive_drop = reverse_values.get("positiveDrop")
        negative_drop = reverse_values.get("negativeDrop")
        face_separation = reverse_values.get("faceSeparation")
        vertex_separation = reverse_values.get(
            "minimumVertexSeparation"
        )
        if (
            positive_displacement is not None
            and positive_displacement <= 0.001
        ):
            errors.append("reverse feature did not move the volume")
        if negative_drop is not None and negative_drop <= 0.02:
            errors.append("free volume control did not move")
        if (
            positive_drop is not None
            and negative_drop is not None
            and positive_drop + 0.01 >= negative_drop
        ):
            errors.append(
                "reverse feature did not separate from free control"
            )
        if face_separation is not None and face_separation <= 0.02:
            errors.append(
                f"{geometry_name} crossed the soft edge/face"
            )
        if vertex_separation is not None and vertex_separation <= 0.10:
            errors.append("vertex SDF was not geometrically excluded")
    if scene_soft_sleep_wake:
        required.update(
            {
                "sceneSoftInitiallyAwake": "1",
                "sceneSoftFirstSlept": "1",
                "sceneSoftSleepWakeCounterZero": "1",
                "sceneSoftSleepVelocitiesZero": "1",
                "sceneSoftStableWhileSleeping": "1",
                "sceneSoftCounterWakeIssued": "1",
                "sceneSoftWokeByCounter": "1",
                "sceneSoftSecondSlept": "1",
                "sceneSoftVelocityWakeIssued": "1",
                "sceneSoftWokeByVelocity": "1",
                "sceneSoftMovedAfterVelocityWake": "1",
                "sceneSoftVelocityStopIssued": "1",
                "sceneSoftFinalSlept": "1",
            }
        )
        try:
            first_sleep = int(fields["sceneSoftFirstSleepFrame"])
            counter_wake = int(fields["sceneSoftCounterWakeFrame"])
            second_sleep = int(fields["sceneSoftSecondSleepFrame"])
            velocity_wake = int(fields["sceneSoftVelocityWakeFrame"])
            final_sleep = int(fields["sceneSoftFinalSleepFrame"])
            if not (
                first_sleep
                < counter_wake
                < second_sleep
                < velocity_wake
                < final_sleep
                < frames
            ):
                errors.append(
                    "soft sleep/wake frame ordering is not strictly causal"
                )
        except (KeyError, ValueError):
            pass
        try:
            if float(fields["maxParticleSpeed"]) >= 3.0:
                errors.append(
                    "soft velocity wake amplified beyond the bounded gate"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 1.0e-6:
                errors.append(
                    "soft final sleep did not zero all vertex velocities"
                )
        except (KeyError, ValueError):
            pass
    if scene_soft_rigid_wake:
        required.update(
            {
                "sceneSoftInitiallyAwake": "1",
                "sceneSoftFirstSlept": "1",
                "sceneSoftSleepWakeCounterZero": "1",
                "sceneSoftSleepVelocitiesZero": "1",
                "sceneSoftStableWhileSleeping": "1",
                "sceneSoftRigidWakeActorAdded": "1",
                "sceneDynamicActorAdded": "1",
                "sceneDynamicInitiallySleeping": "0",
                "sceneSoftWokeByRigid": "1",
                "sceneSoftVelocityStopIssued": "1",
                "sceneSoftFinalSlept": "1",
                "sceneDynamicActorRemoved": "1",
                "sceneDynamicActorReleased": "1",
            }
        )
        try:
            first_sleep = int(fields["sceneSoftFirstSleepFrame"])
            rigid_wake = int(fields["sceneSoftRigidWakeFrame"])
            final_sleep = int(fields["sceneSoftFinalSleepFrame"])
            if not first_sleep < rigid_wake < final_sleep < frames:
                errors.append(
                    "soft rigid-wake frame ordering is not strictly causal"
                )
            if int(fields["rigidContactFrames"]) <= 0:
                errors.append("rigidContactFrames did not prove contact")
            if int(fields["maxRigidContacts"]) <= 0:
                errors.append("maxRigidContacts did not prove contact")
            if float(fields["maxParticleSpeed"]) >= 2.0:
                errors.append(
                    "rigid wake response exceeded the bounded speed gate"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 1.0e-6:
                errors.append(
                    "rigid wake teardown did not return soft body to sleep"
                )
            if float(fields["finalMinY"]) <= 3.5:
                errors.append("rigid wake soft body fell out of bounds")
        except (KeyError, ValueError):
            pass
    if scene_mixed_sleep:
        required.update(
            {
                "sceneSecondVolumeActorCreated": "1",
                "sceneSecondVolumeHostBuffersInitialized": "1",
                "sceneSecondVolumeActorAdded": "1",
                "sceneSecondVolumeActorRemoved": "1",
                "sceneSecondVolumeActorReleased": "1",
                "sceneSecondVolumeBoundsFinite": "1",
                "sceneMixedFirstSlept": "1",
                "sceneMixedFirstStable": "1",
                "sceneMixedSecondStayedAwake": "1",
                "sceneMixedSecondMoved": "1",
            }
        )
        try:
            first_sleep = int(fields["sceneMixedFirstSleepFrame"])
            if not first_sleep < frames:
                errors.append("mixed-island first soft body did not sleep")
            final_speed = float(fields["finalMaxParticleSpeed"])
            if not 0.01 < final_speed < 0.3:
                errors.append(
                    "mixed-island awake soft speed left its bounded range"
                )
            if float(fields["finalMinY"]) <= 3.5:
                errors.append("mixed-island soft body fell out of bounds")
        except (KeyError, ValueError):
            pass
    if scene_soft_churn:
        expected_cycles = (
            ((frames - 1 - 33) // 6) + 1 if frames > 33 else 0
        )
        expected_events = 2 * expected_cycles
        required.update(
            {
                "sceneSecondVolumeActorCreated": "1",
                "sceneSecondVolumeHostBuffersInitialized": "1",
                "sceneSecondVolumeActorAdded": "1",
                "sceneSecondVolumeActorRemoved": "1",
                "sceneSecondVolumeActorReleased": "1",
                "sceneSecondVolumeBoundsFinite": "1",
                "sceneSoftChurnRemoveCount": str(expected_events),
                "sceneSoftChurnReaddCount": str(expected_events),
                "sceneSoftChurnCycles": str(expected_cycles),
                "sceneSoftChurnPostCompactMoveCount": str(expected_events),
                "sceneSoftChurnStable": "1",
            }
        )
        try:
            if float(fields["finalMinY"]) <= 3.5:
                errors.append("soft churn body fell out of bounds")
            if float(fields["finalMaxParticleSpeed"]) >= 1.0e-4:
                errors.append(
                    "soft churn final speed exceeded its bounded range"
                )
        except (KeyError, ValueError):
            pass
    if scene_buffer_mutation:
        required.update(
            {
                "sceneSoftInitiallyAwake": "1",
                "sceneSoftFirstSlept": "1",
                "sceneSoftSleepWakeCounterZero": "1",
                "sceneSoftSleepVelocitiesZero": "1",
                "sceneSoftStableWhileSleeping": "1",
                "sceneBufferMutationIssued": "1",
                "sceneBufferMutationWoke": "1",
                "sceneBufferMutationApplied": "1",
                "sceneBufferDriveIssued": "1",
                "sceneBufferPinHeld": "1",
                "sceneBufferDynamicMoved": "1",
                "sceneBufferInvMassRestored": "1",
                "sceneBufferRestoredMoved": "1",
                "sceneBufferResetIssued": "1",
                "sceneSoftFinalSlept": "1",
            }
        )
        try:
            first_sleep = int(fields["sceneSoftFirstSleepFrame"])
            final_sleep = int(fields["sceneSoftFinalSleepFrame"])
            if not first_sleep < final_sleep < frames:
                errors.append(
                    "buffer mutation sleep ordering is not strictly causal"
                )
            if float(fields["maxParticleSpeed"]) >= 3.0:
                errors.append(
                    "buffer mutation response exceeded its bounded speed gate"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 1.0e-6:
                errors.append(
                    "buffer mutation reset did not return the actor to sleep"
                )
            if float(fields["finalMinY"]) <= 4.0:
                errors.append("buffer mutation reset fell out of bounds")
        except (KeyError, ValueError):
            pass
    if scene_world_pin:
        required.update(
            {
                "sceneWorldPinCreated": "1",
                "sceneWorldPinHeld": "1",
                "sceneWorldPinActorReadded": "1",
                "sceneWorldPinReleased": "1",
                "sceneWorldPinMovedAfterRelease": "1",
            }
        )
        try:
            if float(fields["sceneWorldPinMaxDrift"]) > 1.0e-4:
                errors.append("volume world-attached vertex drifted")
            if (
                float(fields["sceneWorldPinReleasedMaxDisplacement"])
                <= 1.0e-3
            ):
                errors.append(
                    "released volume world-attached vertex did not move"
                )
        except (KeyError, ValueError):
            pass
    if scene_soft_pair_attachment:
        required.update(
            {
                "sceneSecondVolumeActorCreated": "1",
                "sceneSecondVolumeHostBuffersInitialized": "1",
                "sceneSecondVolumeActorAdded": "1",
                "sceneSecondVolumeActorRemoved": "1",
                "sceneSecondVolumeActorReleased": "1",
                "sceneSecondVolumeBoundsFinite": "1",
                "sceneRigidAttachmentActorAdded": "1",
                "sceneRigidAttachmentCreated": "1",
                "sceneRigidAttachmentRigidWoke": "1",
                "sceneRigidAttachmentRigidMoved": "1",
                "sceneRigidAttachmentHeldAcrossReadd": "1",
                "sceneRigidAttachmentReleased": "1",
                "sceneRigidAttachmentSeparatedAfterRelease": "1",
            }
        )
        try:
            if float(fields["sceneRigidAttachmentMaxDrift"]) >= 0.05:
                errors.append(
                    "volume-pair attachment drift exceeded 5cm"
                )
            if (
                float(fields["sceneRigidAttachmentMaxRigidDisplacement"])
                <= 0.02
            ):
                errors.append(
                    "volume-pair target did not move"
                )
            if (
                float(fields["sceneRigidAttachmentMaxRigidSpeed"])
                >= 10.0
            ):
                errors.append(
                    "volume-pair target speed exceeded the bound"
                )
            if (
                float(fields["sceneRigidAttachmentReleasedSeparation"])
                <= 0.2
            ):
                errors.append(
                    "volume pair did not separate after release"
                )
            if float(fields["maxParticleSpeed"]) >= 10.0:
                errors.append(
                    "volume-pair response exceeded the speed bound"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 2.0:
                errors.append(
                    "volume-pair release left excessive tail speed"
                )
        except (KeyError, ValueError):
            pass
    if scene_volume_kinematic_target:
        required.update(
            {
                "sceneVolumeTargetBound": "1",
                "sceneVolumeTargetMutated": "1",
                "sceneVolumeTargetWoke": "1",
                "sceneVolumeTargetReached": "1",
            }
        )
        if scene_partial_kinematic_target:
            required.update(
                {
                    "sceneVolumePartialInactiveIgnored": "1",
                    "sceneVolumePartialActivated": "1",
                    "sceneVolumePartialActivatedReached": "1",
                }
            )
        try:
            if (
                float(fields["sceneVolumeTargetFinalMaxError"])
                > 5.0e-3
            ):
                errors.append(
                    "volume kinematic target error exceeded 5mm"
                )
            if (
                float(fields["sceneVolumeTargetMaxDisplacement"])
                <= 0.2
            ):
                errors.append(
                    "volume kinematic target did not move the actor"
                )
            if float(fields["maxParticleSpeed"]) >= 5.0:
                errors.append(
                    "volume kinematic target exceeded speed bound"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 0.5:
                errors.append(
                    "volume kinematic target left excessive tail speed"
                )
            if (
                scene_partial_kinematic_target
                and float(
                    fields[
                        "sceneVolumePartialInactiveDecoyDistance"
                    ]
                )
                <= 2.0
            ):
                errors.append(
                    "partial inactive target was not demonstrably ignored"
                )
        except (KeyError, ValueError):
            pass
    if scene_motion_controls:
        required.update(
            {
                "motionMaxVelocityBounded": "1",
                "motionSettlingApplied": "1",
                "motionSettlingSlept": "1",
                "motionControlStayedAwake": "1",
            }
        )
        try:
            first_displacement = float(
                fields["motionMaxVelocityFirstStepDisplacement"]
            )
            first_speed = float(fields["motionMaxVelocityFirstStepSpeed"])
            settling_speed = float(fields["motionSettlingFinalSpeed"])
            control_speed = float(fields["motionControlFinalSpeed"])
            if first_displacement > 0.0166666675 * 1.01:
                errors.append(
                    "volume maxLinearVelocity did not bound "
                    "first-frame displacement"
                )
            # The public limit owns free preintegration. Position-AL can add a
            # small constraint velocity before the host buffer is sampled.
            if first_speed > 1.02:
                errors.append(
                    "volume post-solve first-frame speed escaped tolerance"
                )
            if settling_speed > 1.0e-6:
                errors.append(
                    "volume settling actor did not reach zero-speed sleep"
                )
            if not 0.07 <= control_speed <= 0.09:
                errors.append(
                    "volume no-settling control did not retain commanded speed"
                )
            if settling_speed >= control_speed:
                errors.append(
                    "volume settling policy did not separate target "
                    "from control"
                )
        except (KeyError, ValueError):
            pass
    if scene_max_depenetration_velocity:
        required.update(
            {
                "depenetrationLimitApplied": "1",
                "depenetrationFirstStepBounded": "1",
                "depenetrationControlSeparated": "1",
                "depenetrationGradualRecovery": "1",
            }
        )
        try:
            limited_rise = float(
                fields["depenetrationLimitedFirstStepRise"]
            )
            control_rise = float(
                fields["depenetrationControlFirstStepRise"]
            )
            final_rise = float(
                fields["depenetrationLimitedFinalRise"]
            )
            limited_speed = float(
                fields["depenetrationLimitedMaxSpeed"]
            )
            step_limit = 0.12 * 0.0166666675
            if not -1.0e-6 <= limited_rise <= step_limit * 1.25:
                errors.append(
                    "volume limited first-step rise escaped the public cap"
                )
            if control_rise <= limited_rise + 5.0e-3:
                errors.append(
                    "volume unlimited control did not separate "
                    "from limited actor"
                )
            if final_rise <= limited_rise + 4.0e-3:
                errors.append(
                    "volume limited actor did not recover gradually"
                )
            if limited_speed > 0.25:
                errors.append(
                    "volume limited actor's total elastic speed "
                    "escaped the bounded recovery envelope"
                )
        except (KeyError, ValueError):
            pass
    if scene_attachment:
        required.update(
            {
                "sceneRigidAttachmentActorAdded": "1",
                "sceneRigidAttachmentInitiallySleeping": "1",
                "sceneRigidAttachmentCreated": "1",
                "sceneRigidAttachmentRigidMoved": "1",
                "sceneRigidAttachmentHeldAcrossReadd": "1",
                "sceneRigidAttachmentReleased": "1",
                "sceneRigidAttachmentSeparatedAfterRelease": "1",
            }
        )
        if scene_articulation_attachment:
            required.update(
                {
                    "sceneArticulationCreated": "1",
                    "sceneArticulationAdded": "1",
                    "sceneArticulationInitiallySleeping": "1",
                    "sceneArticulationWoke": "1",
                    "sceneArticulationJointSubspaceHeld": "1",
                    "sceneArticulationRootStable": "1",
                }
            )
        elif not scene_static_attachment:
            required.update(
                {
                    "sceneDynamicActorAdded": "1",
                    "sceneDynamicActorRemoved": "1",
                    "sceneDynamicActorReleased": "1",
                    "sceneDynamicInitiallySleeping": "1",
                }
            )
        if scene_kinematic_attachment or scene_static_attachment:
            required.update(
                {
                    "sceneSoftFirstSlept": "1",
                    "sceneKinematicActorAdded": "1",
                    "sceneKinematicTargetIssued": "1",
                    "sceneKinematicTargetReached": "1",
                    "sceneKinematicSoftWoke": "1",
                    "sceneKinematicSoftMoved": "1",
                }
            )
        elif scene_rigid_attachment:
            required["sceneRigidAttachmentRigidWoke"] = "1"
        try:
            if float(fields["sceneRigidAttachmentMaxDrift"]) >= 0.05:
                errors.append(
                    "volume vertex-to-rigid attachment drift exceeded 5cm"
                )
            if (
                float(
                    fields[
                        "sceneRigidAttachmentMaxRigidDisplacement"
                    ]
                )
                <= 0.02
            ):
                errors.append(
                    "volume attachment did not move the dynamic rigid"
                )
            if (
                float(fields["sceneRigidAttachmentMaxRigidSpeed"])
                >= 5.0
            ):
                errors.append(
                    "volume attachment amplified rigid speed"
                )
            if (
                float(
                    fields[
                        "sceneRigidAttachmentReleasedSeparation"
                    ]
                )
                <= 0.2
            ):
                errors.append(
                    "released volume rigid attachment did not separate"
                )
            if scene_kinematic_attachment or scene_static_attachment:
                if float(fields["sceneKinematicMaxPoseError"]) > 1.0e-4:
                    errors.append(
                        "prescribed attachment actor missed its target"
                    )
                if (
                    float(fields["sceneKinematicSoftDisplacement"])
                    <= 0.02
                ):
                    errors.append(
                        "prescribed attachment did not move the volume"
                    )
                if float(fields["maxParticleSpeed"]) >= 20.0:
                    errors.append(
                        "prescribed attachment amplified volume speed"
                    )
                if float(fields["finalMaxParticleSpeed"]) >= 2.0:
                    errors.append(
                        "prescribed attachment left excessive volume speed"
                    )
            if scene_articulation_attachment:
                if (
                    float(fields["sceneArticulationRootMaxDisplacement"])
                    > 1.0e-4
                ):
                    errors.append(
                        "articulation attachment moved the fixed root"
                    )
                if (
                    float(
                        fields[
                            "sceneArticulationChildMaxForbiddenDisplacement"
                        ]
                    )
                    > 1.0e-3
                ):
                    errors.append(
                        "articulation attachment escaped the prismatic subspace"
                    )
                if (
                    float(
                        fields[
                            "sceneArticulationChildMaxAngularDisplacement"
                        ]
                    )
                    > 1.0e-3
                ):
                    errors.append(
                        "articulation attachment rotated a prismatic child"
                    )
                if float(fields["maxParticleSpeed"]) >= 20.0:
                    errors.append(
                        "articulation attachment amplified volume speed"
                    )
                if float(fields["finalMaxParticleSpeed"]) >= 2.0:
                    errors.append(
                        "articulation attachment left excessive volume speed"
                    )
        except (KeyError, ValueError):
            pass
    if scene_element_filter:
        required.update(
            {
                "sceneElementFilterCreated": "1",
                "sceneElementFilterActorReadded": "1",
                "sceneElementFilterSuppressedContact": "1",
                "sceneElementFilterReleased": "1",
                "sceneElementFilterContactRestored": "1",
            }
        )
        try:
            if float(fields["sceneElementFilterMinY"]) >= -0.2:
                errors.append(
                    "volume element filter did not suppress rigid contact"
                )
            final_min_y = float(
                fields["sceneElementFilterFinalMinY"]
            )
            if not -0.05 < final_min_y < 0.05:
                errors.append(
                    "volume rigid contact did not recover after filter release"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 0.1:
                errors.append(
                    "volume did not settle after element filter release"
                )
            if scene_partial_element_filter:
                if fields.get(
                    "scenePartialFilterUnfilteredContactHeld"
                ) != "1":
                    errors.append(
                        "unfiltered volume component lost rigid contact"
                    )
                if fields.get(
                    "scenePartialFilterExactOwnership"
                ) != "1":
                    errors.append(
                        "partial volume filter ownership was not exact"
                    )
                if float(
                    fields["scenePartialFilterUnfilteredMinY"]
                ) <= -0.05:
                    errors.append(
                        "unfiltered volume component penetrated the ground"
                    )
        except (KeyError, ValueError):
            pass
    if scene_kinematic:
        required.update(
            {
                "sceneKinematicActorAdded": "1",
                "sceneKinematicTargetIssued": "1",
                "sceneKinematicTargetReached": "1",
                "sceneKinematicSoftWoke": "1",
                "sceneKinematicSoftMoved": "1",
                "sceneKinematicContactObserved": "1",
                "sceneSoftFirstSlept": "1",
            }
        )
        try:
            if float(fields["sceneKinematicMaxPoseError"]) > 1.0e-4:
                errors.append(
                    "kinematic rigid did not follow its prescribed target"
                )
            if float(fields["sceneKinematicSoftDisplacement"]) <= 0.02:
                errors.append(
                    "kinematic rigid did not move the sleeping volume"
                )
            if abs(float(fields["sceneKinematicFinalY"]) - 4.10) > 1.0e-4:
                errors.append(
                    "kinematic rigid ended away from its prescribed target"
                )
            if float(fields["maxParticleSpeed"]) >= 2.0:
                errors.append(
                    "kinematic coupling amplified the volume speed"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 0.5:
                errors.append(
                    "kinematic coupling left excessive residual speed"
                )
        except (KeyError, ValueError):
            pass
    if scene_multi_scene:
        required.update(
            {
                "sceneSecondSceneCreated": "1",
                "sceneSecondSceneSolverMatched": "1",
                "scenePrimarySceneReleased": "1",
                "sceneSecondSceneReleased": "1",
                "sceneSecondVolumeActorCreated": "1",
                "sceneSecondVolumeHostBuffersInitialized": "1",
                "sceneSecondVolumeActorAdded": "1",
                "sceneSecondVolumeActorRemoved": "1",
                "sceneSecondVolumeActorReleased": "1",
                "sceneSecondVolumeBoundsFinite": "1",
                "sceneSoftFirstSlept": "1",
                "sceneMultiPrimaryStable": "1",
                "sceneMultiPrimaryDetachedStable": "1",
                "sceneMultiSecondaryUpdatedBeforeRelease": "1",
                "sceneMultiSecondaryUpdatedAfterRelease": "1",
            }
        )
        try:
            if int(fields["sceneSoftFirstSleepFrame"]) >= 60:
                errors.append(
                    "multi-scene primary actor did not sleep before removal"
                )
            if float(fields["sceneSecondVolumeMaxCentroidDrop"]) <= 0.1:
                errors.append(
                    "multi-scene secondary actor did not accept both updates"
                )
            if float(fields["finalMinY"]) <= 3.5:
                errors.append("multi-scene actor fell out of bounds")
            if float(fields["maxParticleSpeed"]) >= 0.3:
                errors.append(
                    "multi-scene response exceeded its bounded speed gate"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 1.0e-6:
                errors.append(
                    "multi-scene final sleep did not zero vertex velocities"
                )
        except (KeyError, ValueError):
            pass
    if scene_soft_soft_wake:
        required.update(
            {
                "sceneSecondVolumeActorCreated": "1",
                "sceneSecondVolumeHostBuffersInitialized": "1",
                "sceneSecondVolumeActorAdded": "1",
                "sceneSecondVolumeActorRemoved": "1",
                "sceneSecondVolumeActorReleased": "1",
                "sceneSecondVolumeBoundsFinite": "1",
                "sceneSoftFirstSlept": "1",
                "sceneSoftSoftBothSlept": "1",
                "sceneSoftSoftDriveIssued": "1",
                "sceneSoftSoftDriverWoke": "1",
                "sceneSoftSoftTargetWoke": "1",
                "sceneSoftSoftTargetMoved": "1",
                "sceneSoftSoftResetIssued": "1",
                "sceneSoftSoftBothFinalSlept": "1",
            }
        )
        try:
            first_sleep = int(fields["sceneSoftFirstSleepFrame"])
            target_wake = int(fields["sceneSoftSoftTargetWakeFrame"])
            if not first_sleep < target_wake < frames:
                errors.append(
                    "soft-soft target wake ordering is not strictly causal"
                )
            if float(fields["finalMinY"]) <= 3.5:
                errors.append("soft-soft wake actors fell out of bounds")
            if float(fields["maxParticleSpeed"]) >= 10.0:
                errors.append(
                    "soft-soft wake response exceeded its bounded speed gate"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 1.0e-6:
                errors.append(
                    "soft-soft wake reset did not return both actors to sleep"
                )
        except (KeyError, ValueError):
            pass
    if scene_dynamic_churn:
        required.update(
            {
                "sceneDynamicShapeDetached": "1",
                "sceneDynamicShapeReattached": "1",
                "sceneDynamicActorRemoved": "1",
                "sceneDynamicActorReadded": "1",
                "sceneDynamicReaddedSleeping": "1",
                "sceneDynamicRewokeBySoft": "1",
            }
        )
        try:
            first_wake = int(fields["sceneDynamicFirstWakeFrame"])
            second_wake = int(fields["sceneDynamicSecondWakeFrame"])
            if second_wake <= first_wake or second_wake >= frames:
                errors.append(
                    "sceneDynamicSecondWakeFrame did not prove rebind wake"
                )
        except (KeyError, ValueError):
            pass
    if scene_two_dynamic:
        required.update(
            {
                "sceneSecondDynamicActorAdded": "1",
                "sceneSecondDynamicActorRemoved": "1",
                "sceneSecondDynamicActorReleased": "1",
                "sceneSecondDynamicInitiallySleeping": "1",
                "sceneSecondDynamicWokeBySoft": "1",
            }
        )
        try:
            first_wake = int(fields["sceneDynamicFirstWakeFrame"])
            second_first_wake = int(
                fields["sceneSecondDynamicFirstWakeFrame"]
            )
            if second_first_wake >= frames:
                errors.append(
                    "sceneSecondDynamicFirstWakeFrame did not prove wake"
                )
            if second_first_wake != first_wake:
                errors.append(
                    (
                        "multi-soft islands did not wake in the same "
                        "scheduling pass"
                        if scene_multi_soft
                        else
                        "multi-dynamic targets did not wake in one island pass"
                    )
                )
            if float(fields["sceneSecondDynamicMaxDrop"]) <= 0.01:
                errors.append(
                    "sceneSecondDynamicMaxDrop did not prove rigid response"
                )
            if (
                float(fields["sceneSecondDynamicPreContactMaxDrop"])
                >= 1.0e-4
            ):
                errors.append(
                    "second gravity-disabled rigid moved before soft contact"
                )
            if float(fields["sceneSecondDynamicMaxDownSpeed"]) <= 0.01:
                errors.append(
                    "sceneSecondDynamicMaxDownSpeed did not prove response"
                )
            if float(fields["sceneDynamicMaxDownSpeed"]) >= 6.0:
                errors.append(
                    "primary rigid exceeded bounded multi-target speed"
                )
            if float(fields["sceneSecondDynamicMaxDownSpeed"]) >= 6.0:
                errors.append(
                    "second rigid exceeded bounded multi-target speed"
                )
        except (KeyError, ValueError):
            pass
    if scene_multi_soft:
        required.update(
            {
                "sceneSecondVolumeActorCreated": "1",
                "sceneSecondVolumeHostBuffersInitialized": "1",
                "sceneSecondVolumeActorAdded": "1",
                "sceneSecondVolumeActorRemoved": "1",
                "sceneSecondVolumeActorReleased": "1",
                "sceneSecondVolumeBoundsFinite": "1",
            }
        )
        try:
            if float(fields["sceneSecondVolumeMaxCentroidDrop"]) <= 0.5:
                errors.append(
                    "second soft island did not prove independent motion"
                )
            if float(fields["finalMinY"]) <= -10.0:
                errors.append(
                    "multi-soft teardown fallback became unbounded"
                )
            if float(fields["finalMaxParticleSpeed"]) >= 1.0e-4:
                errors.append(
                    "multi-soft teardown fallback did not honor zero velocity"
                )
        except (KeyError, ValueError):
            pass
    if scene_ground:
        for key in ("groundContactFrames", "maxGroundContacts"):
            try:
                if int(fields[key]) <= 0:
                    errors.append(f"{key} must be positive")
            except (KeyError, ValueError):
                pass
        try:
            if float(fields["finalMinY"]) <= -0.1:
                errors.append("finalMinY crossed the static ground gate")
        except (KeyError, ValueError):
            pass
    if scene_static_box:
        for key in ("rigidContactFrames", "maxRigidContacts"):
            try:
                if int(fields[key]) <= 0:
                    errors.append(f"{key} must be positive")
            except (KeyError, ValueError):
                pass
        try:
            if float(fields["finalMinY"]) <= 0.7:
                errors.append("finalMinY crossed the static box gate")
        except (KeyError, ValueError):
            pass
    if scene_static_churn:
        for key in (
            "sceneStaticShapeDetached",
            "sceneStaticShapeReattached",
            "sceneStaticActorRemoved",
            "sceneStaticActorReadded",
        ):
            if fields.get(key) != "1":
                errors.append(f"{key}={fields.get(key)!r}, expected '1'")
    if scene_dynamic:
        try:
            if float(fields["sceneDynamicMaxDrop"]) <= 0.05:
                errors.append(
                    "sceneDynamicMaxDrop did not prove rigid response"
                )
            if (
                case_name
                not in (
                    "scene-volume-dynamic-capsule",
                    "scene-volume-dynamic-convex",
                )
                and float(fields["sceneDynamicPreContactMaxDrop"])
                >= 1.0e-4
            ):
                errors.append(
                    "gravity-disabled rigid moved before soft contact"
                )
            if (
                float(fields["sceneDynamicInitialY"])
                - float(fields["sceneDynamicFinalY"])
                <= 0.05
            ):
                errors.append(
                    "sceneDynamicFinalY did not prove downward rigid motion"
                )
            if float(fields["minDynamicSurfaceSeparation"]) <= -0.15:
                errors.append(
                    "soft volume crossed the moving rigid surface"
                )
            if float(fields["finalDynamicSurfaceSeparation"]) <= -0.15:
                errors.append(
                    "final soft/rigid separation is penetrating"
                )
            if case_name in (
                "scene-volume-dynamic-sphere",
                "scene-volume-dynamic-capsule",
                "scene-volume-dynamic-convex",
            ):
                if float(fields["finalMinY"]) <= -0.15:
                    errors.append(
                        "soft volume crossed the dynamic-sphere ground"
                    )
                if float(fields["maxParticleSpeed"]) >= 10.0:
                    errors.append(
                        "dynamic-sphere soft response exceeded speed bound"
                    )
                if float(fields["finalMaxParticleSpeed"]) >= 0.5:
                    errors.append(
                        "dynamic-sphere soft response did not settle"
                    )
                if float(fields["sceneDynamicMaxDownSpeed"]) >= 5.0:
                    errors.append(
                        "dynamic sphere exceeded its rigid speed bound"
                    )
                final_y = float(fields["sceneDynamicFinalY"])
                if not 0.70 < final_y < 0.90:
                    errors.append(
                        "dynamic sphere did not settle on the public plane"
                    )
        except (KeyError, ValueError):
            pass
    for key, expected in required.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )
    for key in INT_KEYS:
        try:
            if int(fields[key]) < 0:
                errors.append(f"{key} is negative")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected integer")
    for key in FLOAT_KEYS:
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"{key}={fields.get(key)!r}, expected finite float"
            )
    perf_required = {
        "schema": "1",
        "snippet": "SnippetDeformableVolumeAVBD",
        "case": case_name,
        "softExecution": "serial",
        "softWorkers": "1",
        "convergenceAuthority": "localSolveResidualConsecutive",
        "convergenceSweeps": "2",
        "warmupFrames": str(warmup),
        "profileFrames": str(frames - warmup),
    }
    for key, expected in perf_required.items():
        if perf_fields.get(key) != expected:
            errors.append(
                f"perf {key}={perf_fields.get(key)!r}, "
                f"expected {expected!r}"
            )
    for key in PERF_INT_KEYS:
        try:
            if int(perf_fields[key]) < 0:
                errors.append(f"perf {key} is negative")
        except (KeyError, ValueError):
            errors.append(
                f"perf {key}={perf_fields.get(key)!r}, expected integer"
            )
    if warmup > 0:
        for key in (
            "workspaceGrowthEvents",
            "workspaceGrowthBytes",
            "contactWorkspaceGrowthEvents",
            "contactWorkspaceGrowthBytes",
            "contactOutputGrowthEvents",
            "contactOutputGrowthBytes",
        ):
            if perf_fields.get(key) != "0":
                errors.append(
                    f"perf {key}={perf_fields.get(key)!r}, "
                    "expected zero after warm-up"
                )
    for key in PERF_FLOAT_KEYS:
        try:
            if not math.isfinite(float(perf_fields[key])):
                errors.append(f"perf {key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"perf {key}={perf_fields.get(key)!r}, "
                "expected finite float"
            )
    try:
        legacy = int(
            perf_fields["legacyAppliedConvergedOuterIterations"]
        )
        residual = int(
            perf_fields["residualConvergedOuterIterations"]
        )
        unsafe = int(perf_fields["unsafeAppliedConvergenceCandidates"])
        full = int(perf_fields["budgetExhaustedOuterIterations"])
        outer = int(perf_fields["executedOuterIterations"])
        if not (0 <= unsafe <= legacy <= outer):
            errors.append(
                "perf legacy/unsafe convergence candidate counts are invalid"
            )
        if residual + full != outer:
            errors.append(
                "perf converged/budget inner-loop accounting does not match "
                "executedOuterIterations"
            )
        shadow_1e5_outer = int(
            perf_fields["shadowResidual1e5ConvergedOuterIterations"]
        )
        shadow_1e5_saved = int(
            perf_fields["shadowResidual1e5SavedInnerIterations"]
        )
        shadow_1e4_outer = int(
            perf_fields["shadowResidual1e4ConvergedOuterIterations"]
        )
        shadow_1e4_saved = int(
            perf_fields["shadowResidual1e4SavedInnerIterations"]
        )
        requested_inner = int(perf_fields["requestedInnerIterations"])
        if not (0 <= shadow_1e5_outer <= shadow_1e4_outer <= outer):
            errors.append(
                "perf shadow convergence counts are not monotonic"
            )
        if not (
            0
            <= shadow_1e5_saved
            <= shadow_1e4_saved
            <= requested_inner
        ):
            errors.append(
                "perf shadow saved-iteration counts are not monotonic"
            )
        if shadow_1e4_outer != residual:
            errors.append(
                "perf active 1e-4 policy disagrees with shadow outer count"
            )
        if shadow_1e4_saved != requested_inner - int(
            perf_fields["executedInnerIterations"]
        ):
            errors.append(
                "perf active 1e-4 policy disagrees with saved sweep count"
            )
        if int(perf_fields["nonFiniteRejectedParticleSteps"]) != 0:
            errors.append("perf nonFiniteRejectedParticleSteps is non-zero")
        if int(
            perf_fields["tetLinearizationCacheFallbackParticleSteps"]
        ) != 0:
            errors.append(
                "perf tet linearization cache fallback is non-zero"
            )
        if float(perf_fields["finalMaxAppliedDisplacement"]) > float(
            perf_fields["finalMaxLocalSolveDisplacement"]
        ) + 1e-7:
            errors.append(
                "perf applied displacement exceeds local solve displacement"
            )
    except (KeyError, ValueError):
        pass
    try:
        if abs(float(perf_fields["convergenceTolerance"]) - 1e-4) > 1e-11:
            errors.append("perf convergenceTolerance is not 1e-4")
    except (KeyError, ValueError):
        pass
    try:
        if float(perf_fields["closureMs"]) > float(
            perf_fields["avgStepMs"]
        ) * 1.05:
            errors.append("perf closure exceeds avgStepMs by more than 5%")
    except (KeyError, ValueError):
        pass
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")
    print(
        "[DEFORMABLE_VOLUME_AVBD_RUN] "
        f"name={name} status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUN_ERROR] "
            f"name={name} error={error}"
        )
    return not errors, fields, perf_fields


def compare_repeats(
    case_name: str, first: dict[str, str], second: dict[str, str]
) -> bool:
    mismatches = [
        key for key in INT_KEYS if first.get(key) != second.get(key)
    ]
    for key in FLOAT_KEYS:
        try:
            if abs(float(first[key]) - float(second[key])) > 1e-5:
                mismatches.append(key)
        except (KeyError, ValueError):
            mismatches.append(key)
    if case_name in (
        "scene-volume-sphere-reverse-feature",
        "scene-volume-capsule-reverse-feature",
        "scene-volume-convex-reverse-feature",
        "scene-volume-triangle-mesh-reverse-feature",
        "scene-volume-heightfield-reverse-feature",
    ):
        geometry_name = (
            "triangle-mesh"
            if case_name == "scene-volume-triangle-mesh-reverse-feature"
            else (
                "heightfield"
                if case_name == "scene-volume-heightfield-reverse-feature"
                else (
                    "convex"
                    if case_name == "scene-volume-convex-reverse-feature"
                    else (
                        "capsule"
                        if case_name
                        == "scene-volume-capsule-reverse-feature"
                        else "sphere"
                    )
                )
            )
        )
        for key in (
            "positiveDisplacement",
            "positiveDrop",
            "negativeDrop",
            "faceSeparation",
            "minimumVertexSeparation",
        ):
            metric_key = f"{geometry_name}Reverse.{key}"
            try:
                if (
                    abs(
                        float(first[metric_key]) -
                        float(second[metric_key])
                    )
                    > 1e-5
                ):
                    mismatches.append(metric_key)
            except (KeyError, ValueError):
                mismatches.append(metric_key)
    if case_name in (
        "scene-volume-deforming-sphere-reverse-swept-ccd",
        "scene-volume-deforming-capsule-reverse-swept-ccd",
        "scene-volume-deforming-convex-reverse-swept-ccd",
        "scene-volume-static-sphere-reverse-swept-ccd",
        "scene-volume-kinematic-sphere-reverse-swept-ccd",
        "scene-volume-dynamic-sphere-reverse-swept-ccd",
        "scene-volume-static-capsule-reverse-swept-ccd",
        "scene-volume-kinematic-capsule-reverse-swept-ccd",
        "scene-volume-dynamic-capsule-reverse-swept-ccd",
        "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
        "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
        "scene-volume-static-convex-reverse-swept-ccd",
        "scene-volume-kinematic-convex-reverse-swept-ccd",
        "scene-volume-dynamic-convex-reverse-swept-ccd",
    ):
        for key in (
            "positiveDisplacement",
            "negativeDisplacement",
            "positiveDrop",
            "negativeDrop",
            "positiveRigidDrop",
            "negativeRigidDrop",
            "faceSeparation",
            "minimumVertexSweepSeparation",
        ):
            metric_key = f"sphereReverseSwept.{key}"
            try:
                if (
                    abs(
                        float(first[metric_key])
                        - float(second[metric_key])
                    )
                    > 1e-5
                ):
                    mismatches.append(metric_key)
            except (KeyError, ValueError):
                mismatches.append(metric_key)
        if case_name.startswith("scene-volume-deforming-"):
            for key in (
                "endpointMinSeparation",
                "midSweepMinSeparation",
                "minimumVertexSweepSeparation",
                "responseDelta",
                "positiveDrop",
                "negativeDrop",
            ):
                metric_key = f"deformingVolumeReverseSwept.{key}"
                try:
                    if (
                        abs(
                            float(first[metric_key])
                            - float(second[metric_key])
                        )
                        > 1e-5
                    ):
                        mismatches.append(metric_key)
                except (KeyError, ValueError):
                    mismatches.append(metric_key)
    if case_name in (
        "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
        "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
    ):
        for key in (
            "endpointMinSeparation",
            "midSweepMinSeparation",
            "positiveDisplacement",
            "negativeDisplacement",
            "positiveAngularTravel",
            "negativeAngularTravel",
        ):
            metric_key = f"capsuleRotationalReverseSwept.{key}"
            try:
                if (
                    abs(
                        float(first[metric_key])
                        - float(second[metric_key])
                    )
                    > 1e-5
                ):
                    mismatches.append(metric_key)
            except (KeyError, ValueError):
                mismatches.append(metric_key)
    if (
        case_name
        == "scene-volume-rotating-kinematic-capsule-speculative-ccd"
    ):
        for key in (
            "endpointMinSeparation",
            "midSweepMinSeparation",
            "positiveDisplacement",
            "negativeDisplacement",
        ):
            metric_key = f"capsuleRotational.{key}"
            try:
                if (
                    abs(
                        float(first[metric_key])
                        - float(second[metric_key])
                    )
                    > 1e-5
                ):
                    mismatches.append(metric_key)
            except (KeyError, ValueError):
                mismatches.append(metric_key)
    if (
        case_name
        == "scene-volume-dynamic-rotating-capsule-relative-swept-ccd"
    ):
        for key in (
            "endpointMinSeparation",
            "midSweepMinSeparation",
            "positiveDisplacement",
            "negativeDisplacement",
            "positiveAngularTravel",
            "negativeAngularTravel",
        ):
            metric_key = f"capsuleDynamicRotational.{key}"
            try:
                if (
                    abs(
                        float(first[metric_key])
                        - float(second[metric_key])
                    )
                    > 1e-5
                ):
                    mismatches.append(metric_key)
            except (KeyError, ValueError):
                mismatches.append(metric_key)
    passed = not mismatches
    print(
        "[DEFORMABLE_VOLUME_AVBD_REPEAT] "
        f"case={case_name} status={'PASS' if passed else 'FAIL'} "
        f"mismatches={','.join(mismatches) if mismatches else 'none'}"
    )
    return passed


def summarize_performance(
    case_name: str,
    results: list[dict[str, str]],
    enforce_gate: bool,
) -> bool:
    try:
        avg_median = statistics.median(
            float(result["avgStepMs"]) for result in results
        )
        p95_median = statistics.median(
            float(result["p95StepMs"]) for result in results
        )
        solver_median = statistics.median(
            float(result["solverMs"]) for result in results
        )
        particle_median = statistics.median(
            float(result["particleSolveMs"]) for result in results
        )
    except (KeyError, ValueError, statistics.StatisticsError) as error:
        print(
            "[DEFORMABLE_VOLUME_AVBD_PERF_ERROR] "
            f"case={case_name} error={error}"
        )
        return False
    gate_passed = (
        not enforce_gate
        or (avg_median <= 16.67 and p95_median <= 33.33)
    )
    print(
        "[DEFORMABLE_VOLUME_AVBD_PERF_SUMMARY] "
        f"case={case_name} repeats={len(results)} "
        f"medianAvgStepMs={avg_median:.9g} "
        f"medianP95StepMs={p95_median:.9g} "
        f"medianSolverMs={solver_median:.9g} "
        f"medianParticleSolveMs={particle_median:.9g} "
        f"gate={'ENFORCED' if enforce_gate else 'BASELINE'} "
        f"status={'PASS' if gate_passed else 'FAIL'}"
    )
    return gate_passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "probe",
            "acceptance",
            "performance-baseline",
            "performance-acceptance",
        ),
        default="probe",
    )
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument(
        "--execution",
        choices=("parallel", "sequential"),
        default="sequential",
    )
    parser.add_argument(
        "--debug-crash",
        action="store_true",
        help="Run the selected case under hidden cdb and print the first AV stack.",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        print("[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] frames must be positive")
        return 2
    if (
        args.case == "scene-volume-max-depenetration-velocity"
        and args.frames < 8
    ):
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "scene-volume-max-depenetration-velocity requires at least 8 frames"
        )
        return 2
    if (
        args.case
        in (
            "scene-volume-speculative-ccd",
            "scene-volume-plane-speculative-ccd",
            "scene-volume-sphere-speculative-ccd",
            "scene-volume-capsule-speculative-ccd",
            "scene-volume-convex-speculative-ccd",
            "scene-volume-moving-kinematic-sphere-speculative-ccd",
            "scene-volume-moving-kinematic-capsule-speculative-ccd",
            "scene-volume-rotating-kinematic-capsule-speculative-ccd",
            "scene-volume-rotating-kinematic-convex-speculative-ccd",
            "scene-volume-moving-kinematic-convex-speculative-ccd",
            "scene-volume-dynamic-sphere-relative-swept-ccd",
            "scene-volume-dynamic-capsule-relative-swept-ccd",
            "scene-volume-dynamic-rotating-capsule-relative-swept-ccd",
            "scene-volume-dynamic-rotating-convex-relative-swept-ccd",
            "scene-volume-dynamic-convex-relative-swept-ccd",
            "scene-volume-deforming-sphere-reverse-swept-ccd",
            "scene-volume-deforming-capsule-reverse-swept-ccd",
            "scene-volume-deforming-convex-reverse-swept-ccd",
            "scene-volume-static-sphere-reverse-swept-ccd",
            "scene-volume-kinematic-sphere-reverse-swept-ccd",
            "scene-volume-dynamic-sphere-reverse-swept-ccd",
            "scene-volume-static-capsule-reverse-swept-ccd",
            "scene-volume-kinematic-capsule-reverse-swept-ccd",
            "scene-volume-dynamic-capsule-reverse-swept-ccd",
            "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
            "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
            "scene-volume-rotating-kinematic-convex-reverse-swept-ccd",
            "scene-volume-dynamic-rotating-convex-reverse-swept-ccd",
            "scene-volume-static-convex-reverse-swept-ccd",
            "scene-volume-kinematic-convex-reverse-swept-ccd",
            "scene-volume-dynamic-convex-reverse-swept-ccd",
        )
        and args.frames < 3
    ):
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "speculative CCD cases require at least 3 frames"
        )
        return 2
    if (
        (args.case is None or args.case == "scene-volume-motion-controls")
        and args.frames < 30
    ):
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "scene-volume-motion-controls requires at least 30 frames"
        )
        return 2
    performance_mode = args.mode.startswith("performance-")
    warmup = args.warmup if args.warmup is not None else (
        30 if performance_mode else 0
    )
    if warmup < 0 or warmup >= args.frames:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "warmup must be non-negative and less than frames"
        )
        return 2
    repeats = args.repeats if args.repeats is not None else (
        3 if performance_mode else (2 if args.mode == "acceptance" else 1)
    )
    if repeats <= 0:
        print("[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] repeats must be positive")
        return 2
    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            f"missing executable: {executable}"
        )
        return 2
    if args.debug_crash:
        if not args.case:
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                "--debug-crash requires --case"
            )
            return 2
        debugger = Path(
            r"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe"
        )
        if not debugger.is_file():
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                f"missing debugger: {debugger}"
            )
            return 2
        env = os.environ.copy()
        env["PHYSX_SNIPPET_HEADLESS"] = "1"
        env["PHYSX_SNIPPET_SOLVER"] = "avbd"
        env["PHYSX_SNIPPET_FRAME_COUNT"] = str(args.frames)
        debug_argv = [
            str(debugger),
            "-o",
            "-g",
            "-G",
            "-c",
            "sxe av;g;kv;q",
            str(executable),
            "--headless",
            "--solver=avbd",
            f"--case={args.case}",
            "--execution=sequential",
            f"--frames={args.frames}",
            "--dt=0.0166666675",
            "--dispatcher-threads=2",
            "--seed=1",
        ]
        result = run_headless_process(
            debug_argv,
            cwd=bin_dir,
            env=env,
            timeout_seconds=args.timeout,
        )
        print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())
        if result.timed_out or result.visible_window_detected:
            return 1
        return 0
    selected_cases = (args.case,) if args.case else (
        ("current-all",) if performance_mode else CASES
    )
    passed = True
    results: dict[tuple[str, int], dict[str, str]] = {}
    perf_results: dict[tuple[str, int], dict[str, str]] = {}
    for repeat in range(1, repeats + 1):
        for case_name in selected_cases:
            run_passed, fields, perf_fields = run_one(
                case_name,
                repeat,
                bin_dir,
                args.frames,
                args.timeout,
                args.execution,
                warmup,
            )
            passed = passed and run_passed
            results[(case_name, repeat)] = fields
            perf_results[(case_name, repeat)] = perf_fields
    if args.mode == "acceptance" and repeats >= 2:
        for case_name in selected_cases:
            passed = (
                compare_repeats(
                    case_name,
                    results[(case_name, 1)],
                    results[(case_name, 2)],
                )
                and passed
            )
    if performance_mode:
        for case_name in selected_cases:
            passed = (
                summarize_performance(
                    case_name,
                    [
                        perf_results[(case_name, repeat)]
                        for repeat in range(1, repeats + 1)
                    ],
                    args.mode == "performance-acceptance",
                )
                and passed
            )
    print(
        "[DEFORMABLE_VOLUME_AVBD_SUMMARY] "
        f"mode={args.mode} cases={len(selected_cases)} runs="
        f"{len(selected_cases) * repeats} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
