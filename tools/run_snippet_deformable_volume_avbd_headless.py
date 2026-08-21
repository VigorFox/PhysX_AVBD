#!/usr/bin/env python3
"""Run CPU AVBD soft-body component/coexistence cases without a window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetDeformableVolumeAVBD_64.exe"
CASE_REGISTRY_PATH = (
    REPO_ROOT
    / "physx"
    / "snippets"
    / "snippetdeformablevolumeavbd"
    / "SnippetDeformableVolumeAVBDCases.inc"
)
MEASUREMENT_SOURCE_PATHS = (
    "physx/snippets/snippetdeformablevolumeavbd/"
    "SnippetDeformableVolumeAVBD.cpp",
    "physx/snippets/snippetdeformablevolumeavbd/"
    "SnippetDeformableVolumeAVBDCases.inc",
    "physx/snippets/snippetdeformablevolumeavbd/"
    "SnippetDeformableVolumeAVBDValidation.cpp",
    "physx/snippets/snippetdeformablevolumeavbd/"
    "SnippetDeformableVolumeAVBDReport.cpp",
    "tools/run_snippet_deformable_volume_avbd_headless.py",
    "tools/snippet_headless_process.py",
    "physx/include/PxAvbdCpuIsa.h",
    "physx/source/physx/src/NpAvbdCpuIsa.cpp",
    "physx/include/PxSimulationStatistics.h",
    "physx/source/simulationcontroller/src/ScScene.cpp",
    "physx/source/simulationcontroller/src/ScPipeline.cpp",
    "physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp",
    "physx/source/lowleveldynamics/src/DyAvbdSoftBodyComponent.h",
)
_CASE_REGISTRY_PATTERN = re.compile(
    r'^AVBD_VOLUME_CASE\('
    r'(COMPONENT|SCENE|CORE_SCENE|INTERNAL_SCENE|META),\s*"([^"]+)"'
    r'(?:,\s*([1-9][0-9]*))?\)$'
)


def load_case_registry(
    path: Path = CASE_REGISTRY_PATH,
) -> tuple[tuple[str, str, int | None], ...]:
    rows: list[tuple[str, str, int | None]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        match = _CASE_REGISTRY_PATTERN.fullmatch(line)
        if match is None:
            raise RuntimeError(
                f"invalid AVBD volume case registry row at "
                f"{path}:{line_number}: {line}"
            )
        kind, case_name, default_frames_text = match.groups()
        if case_name in seen:
            raise RuntimeError(
                f"duplicate AVBD volume case '{case_name}' at "
                f"{path}:{line_number}"
            )
        seen.add(case_name)
        default_frames = (
            int(default_frames_text)
            if default_frames_text is not None
            else None
        )
        if (kind == "CORE_SCENE") != (default_frames is not None):
            raise RuntimeError(
                f"AVBD volume case '{case_name}' must provide default "
                f"frames exactly when registered as CORE_SCENE"
            )
        rows.append((kind, case_name, default_frames))
    if not rows:
        raise RuntimeError(f"empty AVBD volume case registry: {path}")
    return tuple(rows)


CASE_REGISTRY = load_case_registry()
CASES = tuple(
    case_name
    for kind, case_name, _ in CASE_REGISTRY
    if kind != "INTERNAL_SCENE"
)
CORRECTNESS_CASE_FRAMES = {
    case_name: default_frames
    for kind, case_name, default_frames in CASE_REGISTRY
    if kind == "CORE_SCENE" and default_frames is not None
}
CORRECTNESS_CASES = tuple(CORRECTNESS_CASE_FRAMES)
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
    "rigidParticleTests",
    "rigidTriangleFaceCandidates",
    "rigidTriangleFaceTests",
    "rigidTriangleEdgeCandidates",
    "rigidTriangleEdgeTests",
    "rigidTriangleVertexCandidates",
    "rigidTriangleVertexTests",
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
PERF_SCHEMA2_INT_KEYS = (
	"fmaSupported",
    "fmaUsed",
	"forceIsaRejected",
    "dispatcherThreads",
    "physicalCores",
    "actualSoftWorkers",
    "taskCount",
    "barrierCount",
    "taskGraphRequestedWorkers",
    "taskGraphCompletedTasks",
    "taskGraphSerialTasks",
    "taskGraphPureSoftEligibleIslands",
    "taskGraphPureSoftEligibleParticles",
    "predictionTaskCount",
    "predictionCompletedTasks",
    "predictionPeakActiveTasks",
    "predictionSerialStages",
    "writeBackTaskCount",
    "writeBackCompletedTasks",
    "writeBackPeakActiveTasks",
    "writeBackSerialStages",
    "topologySoftBodies",
    "topologySoftParticles",
    "topologyTriElements",
    "topologyTetElements",
    "topologyBendElements",
    "topologySurfaceTriangles",
    "topologySurfaceVertices",
    "topologySurfaceEdges",
    "topologyRigidBoxes",
    "topologyRigidTriangleMeshTriangles",
    "rigidParticleSphereTests",
    "rigidParticleCapsuleTests",
    "rigidParticleConvexTests",
    "rigidParticleTriangleSurfaceTests",
)
PERF_SCHEMA2_FLOAT_KEYS = (
	"isaProbeValue",
)
PERF_SCHEMA2_STRING_KEYS = (
	"isaKernelSelfTest",
    "requestedIsa",
    "selectedIsa",
    "compiledIsaBackends",
    "sceneExecution",
    "softScheduler",
)

SPHERE_LONG_ROLL_INT_KEYS = (
    "frames",
    "maxAngularSpeedFrame",
    "windowBegin",
    "windowEnd",
    "windowSamples",
    "longRunBounded",
    "regressionBounded",
)
SPHERE_LONG_ROLL_FLOAT_KEYS = (
    "maxOrientationChange",
    "maxAngularSpeed",
    "finalAngularSpeed",
    "windowMinAngularSpeed",
    "windowMeanAngularSpeed",
    "windowMaxAngularSpeed",
    "finalLinearSpeed",
    "windowMinLinearSpeed",
    "windowMeanLinearSpeed",
    "windowMaxLinearSpeed",
)
SPHERE_LONG_ROLL_CASE = "scene-volume-sphere-long-roll"
SPHERE_SOFT_SOFT_GLANCING_CASE = "scene-volume-sphere-soft-soft-glancing"
SOFT_SOFT_TORQUE_CASE = "scene-volume-soft-soft-torque"
GROUND_EMBEDDED_TET_PROBE_CASE = "scene-volume-ground-embedded-tet-probe"
ROTATION_QUALITY_ACCEPTANCE_MIN_FRAMES = 600
ROTATION_QUALITY_ACCEPTANCE_MIN_FAST_OVER_ORDERED = 0.75
SPHERE_SOFT_SOFT_GLANCING_MIN_FRAMES = 120
SPHERE_SOFT_SOFT_GLANCING_MIN_FAST_OVER_ORDERED = 0.75
SPHERE_SOFT_SOFT_GLANCING_MIN_DELTA_SPEED = 1.0e-3
SPHERE_SOFT_SOFT_GLANCING_MIN_DIRECTION_COSINE = 0.75
SOFT_SOFT_TORQUE_ACCEPTANCE_MIN_FRAMES = 120
SOFT_SOFT_TORQUE_ACCEPTANCE_MIN_RETENTION_SAMPLES = 16
SOFT_SOFT_TORQUE_ACCEPTANCE_MIN_FAST_OVER_ORDERED = 0.75

SOFT_SOFT_TORQUE_INT_KEYS = (
    "frames",
    "isolated",
    "targetDistinctCollisionSimulation",
    "driverDistinctCollisionSimulation",
    "targetSimulationVertices",
    "targetCollisionVertices",
    "driverSimulationVertices",
    "driverCollisionVertices",
    "softContactFrames",
    "generatedSoftContacts",
    "generatedGroundContacts",
    "generatedRigidContacts",
    "generatedSelfContacts",
    "firstContactFrame",
    "firstRotationFrame",
    "retainedRotationSamples",
    "retentionPassed",
)
SOFT_SOFT_TORQUE_FLOAT_KEYS = (
    "firstContactCentroidLeverArm",
    "maxCentroidLeverArm",
    "targetMaxAngularMomentum",
    "targetFinalAngularMomentum",
    "targetMaxAngularSpeed",
    "targetFinalAngularSpeed",
)
GROUND_EMBEDDED_TET_PROBE_INT_KEYS = (
    "frames",
    "simVertices",
    "simTetrahedra",
    "collisionVertices",
    "collisionTetrahedra",
    "distinctCollisionSimulation",
    "strictInteriorEmbedding",
    "selfCollisionDisabled",
    "speculativeCcdDisabled",
    "contactTelemetryEnabled",
    "preGroundSample",
    "firstGroundContactFrame",
    "lastGroundContactFrame",
    "peakGroundRollFrame",
    "groundContactWindowFrames",
    "generatedGroundContacts",
    "generatedRigidContacts",
    "generatedSoftContacts",
    "generatedSelfContacts",
    "groundContactFrames",
    "maxGroundContacts",
    "groundTetPatchGroundRows",
    "groundTetPatchFourSupportRows",
    "groundTetPatchSingleTetRows",
    "groundTetPatchActiveRows",
    "velocityTangentOwnerRows",
    "velocityTangentAppliedRows",
)
GROUND_EMBEDDED_TET_PROBE_FLOAT_KEYS = (
    "launchSpeed",
    "initialMass",
    "initialRmsRadius",
    "preGroundAngularMomentum",
    "preGroundAngularSpeed",
    "peakDeltaAngularMomentumX",
    "peakDeltaAngularMomentumY",
    "peakDeltaAngularMomentumZ",
    "peakDeltaAngularVelocityX",
    "peakDeltaAngularVelocityY",
    "peakDeltaAngularVelocityZ",
    "peakExpectedRollAngularMomentum",
    "peakExpectedRollAngularSpeed",
    "peakNormalizedRollMomentum",
    "peakNormalizedRollOmega",
    "minDetF",
    "maxDetF",
    "minBodyVolumeRatio",
    "maxBodyVolumeRatio",
)
SPHERE_SOFT_SOFT_GLANCING_INT_KEYS = (
    "frames",
    "preSoftContactSample",
    "firstSoftContactFrame",
    "lastSoftContactFrame",
    "peakSoftContactFrame",
    "softContactFrames",
    "generatedGroundContacts",
    "generatedSoftContacts",
    "maxAngularSpeedFrame",
)
SPHERE_SOFT_SOFT_GLANCING_FLOAT_KEYS = (
    "preSoftAngularMomentum",
    "preSoftAngularSpeed",
    "peakSoftContactAngularMomentum",
    "peakSoftContactAngularSpeed",
    "deltaAngularMomentum",
    "deltaAngularSpeed",
    "preSoftAngularVelocityX",
    "preSoftAngularVelocityY",
    "preSoftAngularVelocityZ",
    "peakSoftContactAngularVelocityX",
    "peakSoftContactAngularVelocityY",
    "peakSoftContactAngularVelocityZ",
    "deltaAngularVelocityX",
    "deltaAngularVelocityY",
    "deltaAngularVelocityZ",
    "maxOrientationChange",
    "maxAngularSpeed",
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


def parse_sphere_long_roll(
    lines: list[str], expected_frames: int
) -> tuple[dict[str, str], list[str]]:
    """Parse the long-roll rotation telemetry without assigning pass thresholds."""
    errors: list[str] = []
    if len(lines) != 1:
        return {}, [
            "sphere long-roll telemetry count is "
            f"{len(lines)}, expected exactly 1"
        ]
    fields, parse_errors = parse_gate(lines[0])
    errors.extend(parse_errors)
    for key in SPHERE_LONG_ROLL_INT_KEYS:
        try:
            value = int(fields[key])
            if value < 0:
                errors.append(f"sphere long-roll {key} is negative")
        except (KeyError, ValueError):
            errors.append(
                f"sphere long-roll {key}={fields.get(key)!r}, "
                "expected integer"
            )
    for key in SPHERE_LONG_ROLL_FLOAT_KEYS:
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"sphere long-roll {key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"sphere long-roll {key}={fields.get(key)!r}, "
                "expected finite float"
            )
    if fields.get("frames") != str(expected_frames):
        errors.append(
            f"sphere long-roll frames={fields.get('frames')!r}, "
            f"expected {expected_frames!r}"
        )
    for key in ("longRunBounded", "regressionBounded"):
        if fields.get(key) not in ("0", "1"):
            errors.append(
                f"sphere long-roll {key}={fields.get(key)!r}, expected '0' or '1'"
            )
    if fields.get("result") not in ("PASS", "FAIL"):
        errors.append(
            f"sphere long-roll result={fields.get('result')!r}, "
            "expected 'PASS' or 'FAIL'"
        )
    return fields, errors


def parse_sphere_soft_soft_glancing(
    lines: list[str], expected_frames: int
) -> tuple[dict[str, str], list[str]]:
    """Parse the no-ground cube/sphere contact-phase control telemetry."""
    errors: list[str] = []
    if len(lines) != 1:
        return {}, [
            "sphere soft-soft glancing telemetry count is "
            f"{len(lines)}, expected exactly 1"
        ]
    fields, parse_errors = parse_gate(lines[0])
    errors.extend(parse_errors)
    for key in SPHERE_SOFT_SOFT_GLANCING_INT_KEYS:
        try:
            value = int(fields[key])
            if value < 0:
                errors.append(f"sphere soft-soft glancing {key} is negative")
        except (KeyError, ValueError):
            errors.append(
                f"sphere soft-soft glancing {key}={fields.get(key)!r}, "
                "expected integer"
            )
    for key in SPHERE_SOFT_SOFT_GLANCING_FLOAT_KEYS:
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(
                    f"sphere soft-soft glancing {key} is non-finite"
                )
        except (KeyError, ValueError):
            errors.append(
                f"sphere soft-soft glancing {key}={fields.get(key)!r}, "
                "expected finite float"
            )
    if fields.get("frames") != str(expected_frames):
        errors.append(
            f"sphere soft-soft glancing frames={fields.get('frames')!r}, "
            f"expected {expected_frames!r}"
        )
    if fields.get("contactTelemetry") not in ("enabled", "disabled"):
        errors.append(
            "sphere soft-soft glancing contactTelemetry="
            f"{fields.get('contactTelemetry')!r}, expected 'enabled' or 'disabled'"
        )
    if fields.get("result") not in ("PASS", "FAIL"):
        errors.append(
            "sphere soft-soft glancing result="
            f"{fields.get('result')!r}, expected 'PASS' or 'FAIL'"
        )
    return fields, errors


def parse_soft_soft_torque(
    lines: list[str], expected_frames: int
) -> tuple[dict[str, str], list[str]]:
    """Parse the isolated true-boundary soft/soft torque telemetry."""
    errors: list[str] = []
    if len(lines) != 1:
        return {}, [
            "soft-soft torque telemetry count is "
            f"{len(lines)}, expected exactly 1"
        ]
    fields, parse_errors = parse_gate(lines[0])
    errors.extend(parse_errors)
    for key in SOFT_SOFT_TORQUE_INT_KEYS:
        try:
            value = int(fields[key])
            if value < 0:
                errors.append(f"soft-soft torque {key} is negative")
        except (KeyError, ValueError):
            errors.append(
                f"soft-soft torque {key}={fields.get(key)!r}, "
                "expected integer"
            )
    for key in SOFT_SOFT_TORQUE_FLOAT_KEYS:
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"soft-soft torque {key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"soft-soft torque {key}={fields.get(key)!r}, "
                "expected finite float"
            )
    if fields.get("frames") != str(expected_frames):
        errors.append(
            f"soft-soft torque frames={fields.get('frames')!r}, "
            f"expected {expected_frames!r}"
        )
    if fields.get("supportExpansionInstrumentation") not in (
        "available",
        "unavailable",
    ):
        errors.append(
            "soft-soft torque supportExpansionInstrumentation="
            f"{fields.get('supportExpansionInstrumentation')!r}, "
            "expected 'available' or 'unavailable'"
        )
    if fields.get("supportEvidence") != "embeddedCollisionSimulationTopology":
        errors.append(
            "soft-soft torque supportEvidence="
            f"{fields.get('supportEvidence')!r}, "
            "expected 'embeddedCollisionSimulationTopology'"
        )
    if fields.get("result") not in ("PASS", "FAIL"):
        errors.append(
            f"soft-soft torque result={fields.get('result')!r}, "
            "expected 'PASS' or 'FAIL'"
        )
    return fields, errors


def parse_ground_embedded_tet_probe(
    lines: list[str], expected_frames: int
) -> tuple[dict[str, str], list[str]]:
    """Parse the read-only four-support/single-tet qualification fixture."""
    errors: list[str] = []
    if len(lines) != 1:
        return {}, [
            "ground embedded-tet probe telemetry count is "
            f"{len(lines)}, expected exactly 1"
        ]
    fields, parse_errors = parse_gate(lines[0])
    errors.extend(parse_errors)
    for key in GROUND_EMBEDDED_TET_PROBE_INT_KEYS:
        try:
            if int(fields[key]) < 0:
                errors.append(f"ground embedded-tet probe {key} is negative")
        except (KeyError, ValueError):
            errors.append(
                f"ground embedded-tet probe {key}={fields.get(key)!r}, "
                "expected integer"
            )
    for key in GROUND_EMBEDDED_TET_PROBE_FLOAT_KEYS:
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"ground embedded-tet probe {key} is non-finite")
        except (KeyError, ValueError):
            errors.append(
                f"ground embedded-tet probe {key}={fields.get(key)!r}, "
                "expected finite float"
            )
    if fields.get("frames") != str(expected_frames):
        errors.append(
            f"ground embedded-tet probe frames={fields.get('frames')!r}, "
            f"expected {expected_frames!r}"
        )
    if fields.get("health") not in ("PASS", "FAIL"):
        errors.append(
            f"ground embedded-tet probe health={fields.get('health')!r}, "
            "expected 'PASS' or 'FAIL'"
        )
    if fields.get("result") not in ("PASS", "FAIL"):
        errors.append(
            f"ground embedded-tet probe result={fields.get('result')!r}, "
            "expected 'PASS' or 'FAIL'"
        )
    return fields, errors


def run_one(
    case_name: str,
    repeat: int,
    bin_dir: Path,
    frames: int,
    timeout: float,
    execution: str,
    warmup: int,
    dispatcher_threads: int,
    required_perf_schema: str,
    collision_telemetry: bool,
    surface_triangle_bvh: str,
    rigid_triangle_bvh: str,
    rigid_triangle_grid_dim: int,
    enforce_performance_contract: bool,
    *,
    rotation_quality_lane: str | None = None,
    rotation_quality_acceptance: bool = False,
    soft_soft_torque_lane: str | None = None,
    sphere_soft_soft_glancing_lane: str | None = None,
) -> tuple[bool, dict[str, str], dict[str, str]]:
    schedule_lane = (
        rotation_quality_lane
        or soft_soft_torque_lane
        or sphere_soft_soft_glancing_lane
    )
    name = (
        f"{case_name}-{rotation_quality_lane}-r{repeat}"
        if rotation_quality_lane is not None
        else (
            f"{case_name}-{soft_soft_torque_lane}-r{repeat}"
            if soft_soft_torque_lane is not None
            else (
                f"{case_name}-{sphere_soft_soft_glancing_lane}-r{repeat}"
                if sphere_soft_soft_glancing_lane is not None
                else f"{case_name}-r{repeat}"
            )
        )
    )
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        "--solver=avbd",
        f"--case={case_name}",
        f"--execution={execution}",
        f"--frames={frames}",
        "--dt=0.0166666675",
        f"--dispatcher-threads={dispatcher_threads}",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = "avbd"
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(frames)
    env["PHYSX_AVBD_PROFILE_WARMUP"] = str(warmup)
    if schedule_lane == "ordered":
        # This lane is intentionally explicit so an inherited process switch
        # cannot turn the quality reference into the relaxed fast schedule.
        for env_name in (
            "PHYSX_AVBD_P4_STEP_STATE_SERIAL",
            "PHYSX_AVBD_P4_FORCE_CAUSAL_LAYER_TASKGRAPH_REFERENCE",
            "PHYSX_AVBD_P5_SCENE_REDETECTION_BRIDGE",
            "PHYSX_AVBD_P5_SOFT_PAIR_TASK_FANIN",
            "PHYSX_AVBD_SOFT_ADAPTIVE_INITIALIZATION",
			"PHYSX_AVBD_SOFT_RIGID_PRIMAL_INITIALIZATION",
			"PHYSX_AVBD_SOFT_GROUND_TET_PATCH_PROBE",
			"PHYSX_AVBD_WORLD_STATIC_VELOCITY_TANGENT_OWNER",
            "PHYSX_AVBD_SOFT_ELASTIC_PROXIMAL",
			"PHYSX_AVBD_SCENE_VOLUME_DYNAMIC_FRICTION",
            "PHYSX_AVBD_VOLUME_TEST_3X3",
        ):
            env.pop(env_name, None)
        env["PHYSX_AVBD_TASKGRAPH_SERIAL"] = "1"
        env["PHYSX_AVBD_SOFT_FAST_PATH"] = "0"
        env["PHYSX_AVBD_P4_PRIMAL_SCHEDULE"] = "serial"
    elif schedule_lane == "fast":
        # Keep the measured production lane explicit as well; each probe lane
        # is a fresh process, so the schedule's static policy cache is isolated.
        for env_name in (
            "PHYSX_AVBD_P4_STEP_STATE_SERIAL",
            "PHYSX_AVBD_P4_FORCE_CAUSAL_LAYER_TASKGRAPH_REFERENCE",
            "PHYSX_AVBD_P5_SCENE_REDETECTION_BRIDGE",
            "PHYSX_AVBD_P5_SOFT_PAIR_TASK_FANIN",
            "PHYSX_AVBD_SOFT_ADAPTIVE_INITIALIZATION",
			"PHYSX_AVBD_SOFT_RIGID_PRIMAL_INITIALIZATION",
			"PHYSX_AVBD_SOFT_GROUND_TET_PATCH_PROBE",
			"PHYSX_AVBD_WORLD_STATIC_VELOCITY_TANGENT_OWNER",
            "PHYSX_AVBD_SOFT_ELASTIC_PROXIMAL",
			"PHYSX_AVBD_SCENE_VOLUME_DYNAMIC_FRICTION",
            "PHYSX_AVBD_VOLUME_TEST_3X3",
        ):
            env.pop(env_name, None)
        env.pop("PHYSX_AVBD_TASKGRAPH_SERIAL", None)
        env["PHYSX_AVBD_SOFT_FAST_PATH"] = "1"
        env["PHYSX_AVBD_P4_PRIMAL_SCHEDULE"] = "relaxed-color"
    elif execution == "sequential":
        env["PHYSX_AVBD_TASKGRAPH_SERIAL"] = "1"
        env["PHYSX_AVBD_SOFT_FAST_PATH"] = "0"
    else:
        env.pop("PHYSX_AVBD_TASKGRAPH_SERIAL", None)
        # Parallel acceptance exercises the relaxed color path by default.  A
        # caller can still override this with an explicit environment value.
        env.setdefault("PHYSX_AVBD_SOFT_FAST_PATH", "1")
    if schedule_lane is not None and case_name not in (
        SOFT_SOFT_TORQUE_CASE,
        SPHERE_SOFT_SOFT_GLANCING_CASE,
    ):
        # The long-roll comparison needs the optional collision counters;
        # the two dedicated soft/soft fixtures enable their own accounting in
        # C++ before constructing the Scene.
        env["PHYSX_AVBD_COLLISION_TELEMETRY"] = "1"
    elif collision_telemetry:
        env["PHYSX_AVBD_COLLISION_TELEMETRY"] = "1"
    else:
        env.pop("PHYSX_AVBD_COLLISION_TELEMETRY", None)
    if surface_triangle_bvh == "off":
        env["PHYSX_AVBD_SURFACE_TRIANGLE_BVH"] = "0"
    else:
        env.pop("PHYSX_AVBD_SURFACE_TRIANGLE_BVH", None)
    if rigid_triangle_bvh == "off":
        env["PHYSX_AVBD_RIGID_TRIANGLE_BVH"] = "0"
    else:
        env.pop("PHYSX_AVBD_RIGID_TRIANGLE_BVH", None)
    if rigid_triangle_grid_dim > 1:
        env["PHYSX_AVBD_RIGID_TRIANGLE_GRID_DIM"] = str(
            rigid_triangle_grid_dim
        )
    else:
        env.pop("PHYSX_AVBD_RIGID_TRIANGLE_GRID_DIM", None)
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
    sphere_long_roll_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_SPHERE_LONG_ROLL] ")
    ]
    soft_soft_torque_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_SOFT_SOFT_TORQUE] ")
    ]
    ground_embedded_tet_probe_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_GROUND_EMBEDDED_TET_PROBE] ")
    ]
    sphere_soft_soft_glancing_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_SPHERE_SOFT_SOFT_GLANCING] ")
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
    rotation_quality_infrastructure_errors: list[str] | None = None
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
    scene_visual_showcase = case_name == "scene-volume-visual-showcase"
    scene_ogc_sandwich = case_name == "scene-volume-ogc-sandwich"
    scene_sphere_long_roll = case_name == SPHERE_LONG_ROLL_CASE
    scene_sphere_soft_soft_glancing = (
        case_name == "scene-volume-sphere-soft-soft-glancing"
    )
    scene_soft_soft_torque = case_name == SOFT_SOFT_TORQUE_CASE
    scene_ground_embedded_tet_probe = (
        case_name == GROUND_EMBEDDED_TET_PROBE_CASE
    )
    if rotation_quality_lane is not None:
        if not scene_sphere_long_roll:
            errors.append(
                "rotation-quality lane requires "
                f"case={SPHERE_LONG_ROLL_CASE!r}"
            )
        else:
            sphere_long_roll_fields, parse_errors = parse_sphere_long_roll(
                sphere_long_roll_lines, frames
            )
            errors.extend(parse_errors)
            for key, value in sphere_long_roll_fields.items():
                fields[f"sphereLongRoll.{key}"] = value
        # The comparison lane is intentionally a measurement, not a new
        # physics acceptance gate.  Retain only failures which mean the
        # measurement itself is untrustworthy: malformed/missing telemetry,
        # process timeout/window leakage, or a mismatched run identity.
        for key, expected in (
            ("snippet", "SnippetDeformableVolumeAVBD"),
            ("solver", "avbd"),
            ("case", SPHERE_LONG_ROLL_CASE),
        ):
            if fields.get(key) != expected:
                errors.append(
                    f"rotation-quality gate {key}={fields.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if perf_fields.get("sceneExecution") != execution:
            errors.append(
                "rotation-quality perf sceneExecution="
                f"{perf_fields.get('sceneExecution')!r}, expected {execution!r}"
            )
        if fields.get("status") not in ("PASS", "FAIL"):
            errors.append(
                f"rotation-quality gate status={fields.get('status')!r}, "
                "expected 'PASS' or 'FAIL'"
            )
        rotation_quality_infrastructure_errors = list(errors)
    if soft_soft_torque_lane is not None:
        if not scene_soft_soft_torque:
            errors.append(
                "soft-soft torque lane requires "
                f"case={SOFT_SOFT_TORQUE_CASE!r}"
            )
        else:
            soft_soft_torque_fields, parse_errors = parse_soft_soft_torque(
                soft_soft_torque_lines, frames
            )
            errors.extend(parse_errors)
            for key, value in soft_soft_torque_fields.items():
                fields[f"softSoftTorque.{key}"] = value
        for key, expected in (
            ("snippet", "SnippetDeformableVolumeAVBD"),
            ("solver", "avbd"),
            ("case", SOFT_SOFT_TORQUE_CASE),
        ):
            if fields.get(key) != expected:
                errors.append(
                    f"soft-soft torque gate {key}={fields.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if perf_fields.get("sceneExecution") != execution:
            errors.append(
                "soft-soft torque perf sceneExecution="
                f"{perf_fields.get('sceneExecution')!r}, expected {execution!r}"
            )
    if scene_ground_embedded_tet_probe:
        ground_tet_fields, parse_errors = parse_ground_embedded_tet_probe(
            ground_embedded_tet_probe_lines, frames
        )
        errors.extend(parse_errors)
        for key, value in ground_tet_fields.items():
            fields[f"groundEmbeddedTetProbe.{key}"] = value
        for key in (
            "simVertices",
            "simTetrahedra",
            "collisionVertices",
            "collisionTetrahedra",
            "distinctCollisionSimulation",
            "strictInteriorEmbedding",
            "selfCollisionDisabled",
            "speculativeCcdDisabled",
        ):
            expected = "1" if key not in ("simVertices", "collisionVertices") else "4"
            if key in ("simTetrahedra", "collisionTetrahedra"):
                expected = "1"
            if ground_tet_fields.get(key) != expected:
                errors.append(
                    "ground embedded-tet probe "
                    f"{key}={ground_tet_fields.get(key)!r}, expected {expected!r}"
                )
        if ground_tet_fields.get("health") != "PASS":
            errors.append("ground embedded-tet probe health gate failed")
        if ground_tet_fields.get("result") != "PASS":
            errors.append("ground embedded-tet probe rotation gate failed")
        for key in (
            "contactTelemetryEnabled",
            "preGroundSample",
        ):
            if ground_tet_fields.get(key) != "1":
                errors.append(
                    "ground embedded-tet probe "
                    f"{key}={ground_tet_fields.get(key)!r}, expected '1'"
                )
        try:
            first_ground_frame = int(ground_tet_fields["firstGroundContactFrame"])
            last_ground_frame = int(ground_tet_fields["lastGroundContactFrame"])
            peak_roll_frame = int(ground_tet_fields["peakGroundRollFrame"])
            if not (
                0 <= first_ground_frame <= peak_roll_frame <= last_ground_frame
                <= frames
            ):
                errors.append(
                    "ground embedded-tet probe invalid ground-contact/roll "
                    "frame ordering"
                )
        except (KeyError, ValueError):
            errors.append(
                "ground embedded-tet probe ground-contact/roll frame fields "
                "are missing or invalid"
            )
        for key in (
            "groundContactWindowFrames",
            "generatedGroundContacts",
        ):
            try:
                if int(ground_tet_fields[key]) <= 0:
                    errors.append(
                        "ground embedded-tet probe "
                        f"{key} must be positive"
                    )
            except (KeyError, ValueError):
                errors.append(
                    "ground embedded-tet probe "
                    f"{key} is missing or invalid"
                )
        for key in (
            "generatedRigidContacts",
            "generatedSoftContacts",
            "generatedSelfContacts",
        ):
            if ground_tet_fields.get(key) != "0":
                errors.append(
                    "ground embedded-tet probe must remain ground-only: "
                    f"{key}={ground_tet_fields.get(key)!r}"
                )
        for key in (
            "peakNormalizedRollMomentum",
            "peakNormalizedRollOmega",
        ):
            try:
                if float(ground_tet_fields[key]) <= 128.0 * 1.1920928955078125e-7:
                    errors.append(
                        "ground embedded-tet probe "
                        f"{key} lacks measurable expected-axis roll"
                    )
            except (KeyError, ValueError):
                errors.append(
                    "ground embedded-tet probe "
                    f"{key} is missing or invalid"
                )
        # Validate against the environment actually passed to this child.  The
        # ordered/fast quality lanes deliberately remove experimental switches
        # from their child environment, even when the runner's own parent was
        # launched with one enabled.
        qualification_enabled = (
            env.get("PHYSX_AVBD_SOFT_GROUND_TET_PATCH_PROBE") == "1"
        )
        if qualification_enabled:
            for key in (
                "groundTetPatchGroundRows",
                "groundTetPatchFourSupportRows",
                "groundTetPatchSingleTetRows",
                "groundTetPatchActiveRows",
            ):
                try:
                    if int(ground_tet_fields[key]) <= 0:
                        errors.append(
                            "ground embedded-tet qualification "
                            f"{key} must be positive when enabled"
                        )
                except (KeyError, ValueError):
                    errors.append(
                        "ground embedded-tet qualification "
                        f"{key} is missing or invalid"
                    )
        # World-static split tangent ownership is production-default.  An
        # exact zero is retained only as a diagnostic rollback lane.
        velocity_tangent_owner_enabled = (
            env.get("PHYSX_AVBD_WORLD_STATIC_VELOCITY_TANGENT_OWNER") != "0"
        )
        if velocity_tangent_owner_enabled:
            for key in (
                "velocityTangentOwnerRows",
                "velocityTangentAppliedRows",
            ):
                try:
                    if int(ground_tet_fields[key]) <= 0:
                        errors.append(
                            "world-static velocity tangent-owner "
                            f"{key} must be positive when enabled"
                        )
                except (KeyError, ValueError):
                    errors.append(
                        "world-static velocity tangent-owner "
                        f"{key} is missing or invalid"
                    )
    if sphere_soft_soft_glancing_lane is not None:
        if not scene_sphere_soft_soft_glancing:
            errors.append(
                "sphere soft-soft glancing lane requires "
                f"case={SPHERE_SOFT_SOFT_GLANCING_CASE!r}"
            )
        else:
            glancing_fields, parse_errors = parse_sphere_soft_soft_glancing(
                sphere_soft_soft_glancing_lines, frames
            )
            errors.extend(parse_errors)
            for key, value in glancing_fields.items():
                fields[f"sphereSoftSoftGlancing.{key}"] = value
        for key, expected in (
            ("snippet", "SnippetDeformableVolumeAVBD"),
            ("solver", "avbd"),
            ("case", SPHERE_SOFT_SOFT_GLANCING_CASE),
        ):
            if fields.get(key) != expected:
                errors.append(
                    f"sphere soft-soft glancing gate {key}={fields.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if perf_fields.get("sceneExecution") != execution:
            errors.append(
                "sphere soft-soft glancing perf sceneExecution="
                f"{perf_fields.get('sceneExecution')!r}, expected {execution!r}"
            )
    component_dense_no_contact = (
        case_name == "volume-performance-dense-no-contact"
    )
    component_many_small_no_contact = (
        case_name == "volume-performance-many-small-no-contact"
    )
    component_no_contact = (
        component_dense_no_contact or component_many_small_no_contact
    )
    scene_ground_embedded_tet_probe = (
        case_name == GROUND_EMBEDDED_TET_PROBE_CASE
    )
    scene_ground = case_name == "scene-volume-ground"
    scene_static_churn = case_name == "scene-volume-static-churn"
    scene_static_box = (
        case_name == "scene-volume-static-box" or scene_static_churn
    )
    scene_static = (
        scene_ground or scene_ground_embedded_tet_probe or scene_static_box
    )
    scene_dynamic_churn = case_name == "scene-volume-dynamic-churn"
    scene_multi_dynamic = case_name == "scene-volume-multi-dynamic-box"
    scene_multi_soft = case_name == "scene-volume-multi-soft-islands"
    scene_taskgraph_pure_soft = (
        case_name in (
            "scene-volume-taskgraph-pure-soft",
            "scene-volume-taskgraph-pure-soft-corotational",
        )
    )
    scene_taskgraph_writeback_four_way = (
        case_name == "scene-volume-taskgraph-writeback-four-way"
        or case_name == "scene-volume-taskgraph-writeback-heterogeneous"
        or case_name == "scene-volume-taskgraph-pipeline"
    )
    scene_taskgraph_writeback = (
        case_name == "scene-volume-taskgraph-writeback"
        or scene_taskgraph_writeback_four_way
    )
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
        "scene-volume-kinematic-triangle-mesh-speculative-ccd",
        "scene-volume-kinematic-heightfield-speculative-ccd",
        "scene-volume-kinematic-triangle-mesh-reverse-swept-ccd",
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
    scene_kinematic_triangle_surface_swept_ccd = (
        scene_triangle_surface_swept_ccd
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
    scene_soft_soft_torque = case_name == SOFT_SOFT_TORQUE_CASE
    scene_rigid_triangle_steady_contact = (
        case_name == "scene-volume-rigid-triangle-steady-contact"
    )
    scene_static = (
        scene_static
        or scene_element_filter
        or scene_max_depenetration_velocity
        or scene_rigid_triangle_steady_contact
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
        scene_sphere_long_roll
        or scene_sphere_soft_soft_glancing
        or scene_soft_soft_torque
        or scene_ogc_sandwich
        or scene_multi_soft
        or scene_mixed_sleep
        or scene_soft_churn
        or scene_multi_scene
        or scene_soft_soft_wake
        or scene_soft_pair_attachment
        or scene_motion_controls
        or scene_max_depenetration_velocity
        or scene_speculative_ccd
        or scene_smooth_reverse_feature
        or scene_taskgraph_writeback
    )
    scene_two_dynamic = scene_multi_dynamic or scene_multi_soft
    scene_dynamic = (
        scene_ogc_sandwich
        or case_name == "scene-volume-dynamic-box"
        or case_name == "scene-volume-true-boundary-dynamic-box"
        or case_name == "scene-volume-dynamic-sphere"
        or case_name == "scene-volume-dynamic-capsule"
        or case_name == "scene-volume-dynamic-convex"
        or scene_dynamic_churn
        or scene_two_dynamic
    )
    scene_integrated = (
        scene_lifecycle
        or scene_visual_showcase
        or scene_ogc_sandwich
        or scene_sphere_long_roll
        or scene_sphere_soft_soft_glancing
        or scene_soft_soft_torque
        or scene_taskgraph_pure_soft
        or scene_taskgraph_writeback
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
            "SCENE_TASKGRAPH_WRITEBACK_GATED"
            if scene_taskgraph_writeback
            else (
                "SCENE_TASKGRAPH_PURE_SOFT_GATED"
                if scene_taskgraph_pure_soft
                else (
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
                    scene_visual_showcase
                    or scene_ogc_sandwich
                    or scene_dynamic
                    or scene_soft_rigid_wake
                    or scene_kinematic
                    or scene_rigid_attachment
                    or scene_kinematic_attachment
                )
                else "0"
            )
        ),
        "sceneDeformableVolumes": (
            "5"
            if scene_visual_showcase
            else (
                "4"
                if scene_taskgraph_writeback_four_way
                else (
                    "2"
                    if scene_two_soft
                    else ("1" if scene_integrated else "0")
                )
            )
        ),
        "nonFiniteParticleSamples": "0",
        "invertedElementSamples": "0",
        "invalidContactSourceSamples": "0",
        "solverReadbackMatched": "1",
        "fatalErrors": "0",
        "cleanupComplete": "1",
    }
    if case_name == "scene-volume-taskgraph-pipeline":
        required["validation"] = "SCENE_TASKGRAPH_PIPELINE_GATED"
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
    if scene_rigid_triangle_steady_contact:
        required["validation"] = (
            "SCENE_RIGID_TRIANGLE_STEADY_CONTACT_GATED"
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
    if scene_visual_showcase:
        required["validation"] = "SCENE_VISUAL_SHOWCASE_GATED"
        # The showcase owns one free rigid body in addition to its two static
        # actors. Collision response itself is checked by the snippet once the
        # visual gate reaches its minimum frame count.
        required.update(
            {
                "sceneDynamicActorAdded": "1",
                "sceneDynamicInitiallySleeping": "0",
                "sceneDynamicActorRemoved": "1",
                "sceneDynamicActorReleased": "1",
            }
        )
    if scene_ogc_sandwich:
        required.update(
            {
                "validation": "SCENE_OGC_SANDWICH_GATED",
                "sceneDynamicActorAdded": "1",
                "sceneDynamicInitiallySleeping": "0",
                "sceneDynamicActorRemoved": "1",
                "sceneDynamicActorReleased": "1",
            }
        )
    if scene_sphere_long_roll:
        required["validation"] = "SCENE_SPHERE_LONG_ROLL_GATED"
    if scene_sphere_soft_soft_glancing:
        required["validation"] = "SCENE_SPHERE_SOFT_SOFT_GLANCING_GATED"
    if scene_soft_soft_torque:
        required["validation"] = "SCENE_SOFT_SOFT_TORQUE_GATED"
    if scene_ground_embedded_tet_probe:
        required["validation"] = "SCENE_GROUND_EMBEDDED_TET_PROBE_GATED"
    if scene_integrated:
        required.update(
            {
                "sceneStatics": (
                    "2"
                    if scene_visual_showcase
                    else (
                        "1"
                        if (
                            scene_sphere_long_roll
                            or scene_static
                            or scene_static_attachment
                            or case_name == "scene-volume-taskgraph-pipeline"
                            or case_name == "scene-volume-dynamic-sphere"
                            or case_name == "scene-volume-dynamic-capsule"
                            or case_name == "scene-volume-dynamic-convex"
                        )
                        else "0"
                    )
                ),
                "softBodies": (
                    "5"
                    if scene_visual_showcase
                    else (
                        "4"
                        if scene_taskgraph_writeback_four_way
                        else ("2" if scene_two_soft else "1")
                    )
                ),
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
    if component_no_contact:
        required.update(
            {
                "softBodies": (
                    "16" if component_many_small_no_contact else "1"
                ),
                "rigidBoxes": "0",
                "sceneStatics": "0",
                "sceneDynamics": "0",
                "sceneDeformableVolumes": "0",
                "groundContactFrames": "0",
                "rigidContactFrames": "0",
                "softContactFrames": "0",
            }
        )
    if component_dense_no_contact:
        required.update({"particles": "2197", "tetElements": "8640"})
    if component_many_small_no_contact:
        required.update({"particles": "1024", "tetElements": "2160"})
    if scene_dynamic and not scene_ogc_sandwich:
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
                    "sceneStatics": "0",
                    "sceneDynamics": "2",
                    "speculativeCcdNegativeControlTunneled": "1",
                }
            )
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
            expected_target = "kinematic"
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
                    and not (
                        scene_dynamic_sphere_reverse_swept_ccd
                        and positive_rigid_drop is not None
                        and negative_rigid_drop is not None
                        and positive_rigid_drop + 0.05
                        < negative_rigid_drop
                    )
                ):
                    errors.append(
                        f"moving {geometry_name} reverse sweep produced "
                        "neither a soft nor a rigid response"
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
                    or (
                        positive_soft_displacement <= 0.02
                        and (
                            not math.isfinite(positive_rigid_drop)
                            or not math.isfinite(negative_rigid_drop)
                            or positive_rigid_drop + 0.05
                            >= negative_rigid_drop
                        )
                    )
                ):
                    errors.append(
                        "dynamic finite-geometry sweep produced neither "
                        "a soft nor a rigid response"
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
            if float(fields["sceneElementFilterMinY"]) > -0.02:
                errors.append(
                    "volume element filter did not suppress rigid contact"
                )
            final_min_y = float(
                fields["sceneElementFilterFinalMinY"]
            )
            if not -0.005 <= final_min_y <= 0.06:
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
                ) < -0.005:
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
            # The fast scheduler may hand the two independent island wake
            # notifications back on adjacent frames.  Bound the latency, but
            # do not make a physical-response gate depend on one exact task
            # completion order.
            if abs(second_first_wake - first_wake) > 1:
                errors.append(
                    (
                        "multi-soft islands did not wake within one "
                        "scheduling frame"
                        if scene_multi_soft
                        else "multi-dynamic targets did not wake within "
                        "one scheduling frame"
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
    if component_no_contact:
        try:
            if component_dense_no_contact:
                if int(fields["particles"]) < 1000:
                    errors.append("dense no-contact corpus is undersized")
                if int(fields["tetElements"]) < 5000:
                    errors.append(
                        "dense no-contact corpus has too few tetrahedra"
                    )
            if component_many_small_no_contact:
                if int(fields["particles"]) != 1024:
                    errors.append("many-small corpus particle count changed")
                if int(fields["tetElements"]) != 2160:
                    errors.append("many-small corpus tet count changed")
            if abs(float(fields["maxCentroidDrop"])) > 1.0e-5:
                errors.append("no-contact volume moved under zero gravity")
            if float(fields["maxParticleSpeed"]) > 1.0e-4:
                errors.append("no-contact volume accumulated velocity")
        except (KeyError, ValueError):
            pass
    # The OGC sandwich is a zero-gravity, symmetric contact-law fixture.
    # Its C++ gate owns the contact and deformation assertions; the generic
    # dynamic-volume checks below intentionally require a gravity-driven box
    # and therefore do not apply to this unit test.
    if scene_dynamic and not scene_ogc_sandwich:
        try:
            dynamic_convex = case_name == "scene-volume-dynamic-convex"
            minimum_rigid_drop = 5.0e-4 if dynamic_convex else 0.05
            if float(fields["sceneDynamicMaxDrop"]) <= minimum_rigid_drop:
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
                <= minimum_rigid_drop
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
                maximum_final_particle_speed = (
                    0.75 if dynamic_convex else 0.5
                )
                if (
                    float(fields["finalMaxParticleSpeed"])
                    >= maximum_final_particle_speed
                ):
                    errors.append(
                        "dynamic-sphere soft response did not settle"
                    )
                if float(fields["sceneDynamicMaxDownSpeed"]) >= 5.0:
                    errors.append(
                        "dynamic sphere exceeded its rigid speed bound"
                    )
                final_y = float(fields["sceneDynamicFinalY"])
                valid_final_height = (
                    0.90 < final_y < 1.10
                    if dynamic_convex
                    else 0.70 < final_y < 0.90
                )
                if not valid_final_height:
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
    perf_schema = perf_fields.get("schema")
    if perf_schema not in ("1", "2"):
        errors.append(
            f"perf schema={perf_schema!r}, expected one of ('1', '2')"
        )
    elif (
        required_perf_schema != "auto"
        and perf_schema != required_perf_schema
    ):
        errors.append(
            f"perf schema={perf_schema!r}, expected {required_perf_schema!r}"
        )
    perf_required = {
        "snippet": "SnippetDeformableVolumeAVBD",
        "case": case_name,
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
    if perf_schema == "2":
        if perf_fields.get("sceneExecution") != execution:
            errors.append(
                f"perf sceneExecution={perf_fields.get('sceneExecution')!r}, "
                f"expected {execution!r}"
            )
        if perf_fields.get("dispatcherThreads") != str(dispatcher_threads):
            errors.append(
                f"perf dispatcherThreads="
                f"{perf_fields.get('dispatcherThreads')!r}, "
                f"expected {dispatcher_threads!r}"
            )
        for key in PERF_SCHEMA2_STRING_KEYS:
            if not perf_fields.get(key):
                errors.append(f"perf {key} is missing or empty")
        for key in PERF_SCHEMA2_INT_KEYS:
            try:
                if int(perf_fields[key]) < 0:
                    errors.append(f"perf {key} is negative")
            except (KeyError, ValueError):
                errors.append(
                    f"perf {key}={perf_fields.get(key)!r}, expected integer"
                )
        for key in PERF_SCHEMA2_FLOAT_KEYS:
            try:
                if not math.isfinite(float(perf_fields[key])):
                    errors.append(f"perf {key} is non-finite")
            except (KeyError, ValueError):
                errors.append(
                    f"perf {key}={perf_fields.get(key)!r}, "
                    "expected finite float"
                )
        try:
            submitted_tasks = int(perf_fields["taskCount"])
            completed_tasks = int(perf_fields["taskGraphCompletedTasks"])
            serial_tasks = int(perf_fields["taskGraphSerialTasks"])
            submitted_prediction_tasks = int(
                perf_fields["predictionTaskCount"]
            )
            completed_prediction_tasks = int(
                perf_fields["predictionCompletedTasks"]
            )
            serial_prediction_stages = int(
                perf_fields["predictionSerialStages"]
            )
            submitted_writeback_tasks = int(perf_fields["writeBackTaskCount"])
            completed_writeback_tasks = int(
                perf_fields["writeBackCompletedTasks"]
            )
            serial_writeback_stages = int(
                perf_fields["writeBackSerialStages"]
            )
            requested_workers = int(
                perf_fields["taskGraphRequestedWorkers"]
            )
            if submitted_tasks != completed_tasks:
                errors.append(
                    "taskgraph submitted/completed mismatch: "
                    f"{submitted_tasks}!={completed_tasks}"
                )
            if submitted_prediction_tasks != completed_prediction_tasks:
                errors.append(
                    "prediction submitted/completed mismatch: "
                    f"{submitted_prediction_tasks}!="
                    f"{completed_prediction_tasks}"
                )
            if submitted_writeback_tasks != completed_writeback_tasks:
                errors.append(
                    "writeback submitted/completed mismatch: "
                    f"{submitted_writeback_tasks}!="
                    f"{completed_writeback_tasks}"
                )
            # The standalone component corpus deliberately has no PxScene,
            # hence no PxTaskManager.  Only Scene-integrated fixtures can
            # prove dispatcher ownership; standalone runs must keep these
            # counters at zero rather than pretending to have workers.
            scene_taskgraph_available = (
                fields.get("sceneSoftIntegration") == "1"
            )
            scene_taskgraph_owner_released = (
                scene_multi_scene
                and fields.get("scenePrimarySceneReleased") == "1"
                and fields.get("sceneSecondSceneReleased") == "1"
            )
            if (
                scene_taskgraph_available
                and not scene_taskgraph_owner_released
                and requested_workers != dispatcher_threads
            ):
                errors.append(
                    "taskgraph requested workers="
                    f"{requested_workers}, expected {dispatcher_threads}"
                )
            if scene_taskgraph_owner_released and requested_workers != 0:
                errors.append(
                    "released multi-scene fixture retained taskgraph workers="
                    f"{requested_workers}"
                )
            if not scene_taskgraph_available and (
                requested_workers != 0
                or submitted_tasks != 0
                or completed_tasks != 0
                or serial_tasks != 0
                or submitted_prediction_tasks != 0
                or completed_prediction_tasks != 0
                or serial_prediction_stages != 0
                or submitted_writeback_tasks != 0
                or completed_writeback_tasks != 0
                or serial_writeback_stages != 0
            ):
                errors.append(
                    "standalone component fixture reported Scene taskgraph work"
                )
            if execution == "sequential" and submitted_tasks != 0:
                errors.append(
                    "sequential taskgraph mode submitted island tasks"
                )
            if execution == "sequential" and (
                perf_fields.get("softExecution") != "serial"
                or perf_fields.get("softWorkers") != "1"
            ):
                errors.append(
                    "sequential taskgraph mode did not report one serial worker"
                )
            if execution == "parallel" and serial_tasks != 0:
                errors.append(
                    "parallel taskgraph mode used serial island tasks"
                )
            if scene_taskgraph_pure_soft:
                pure_soft_particles = int(
                    perf_fields["taskGraphPureSoftEligibleParticles"]
                )
                if pure_soft_particles <= 0:
                    errors.append(
                        "large pure-soft Scene fixture did not report "
                        "submitted pure-soft task work"
                    )
                if execution == "parallel" and submitted_tasks <= 0:
                    errors.append(
                        "large pure-soft Scene fixture did not submit a "
                        "dispatcher task"
                    )
                if execution == "sequential" and serial_tasks <= 0:
                    errors.append(
                        "large pure-soft Scene fixture did not execute the "
                        "serial taskgraph authority path"
                    )
            if scene_taskgraph_writeback:
                p3_entry_count = 4 if scene_taskgraph_writeback_four_way else 2
                p3_profiled_frames = frames - warmup
                p3_stage_task_count = min(dispatcher_threads, p3_entry_count)
                if execution == "parallel" and dispatcher_threads >= 2:
                    expected_p3_task_count = (
                        p3_profiled_frames * p3_stage_task_count
                    )
                    if submitted_prediction_tasks != expected_p3_task_count:
                        errors.append(
                            "P3 prediction task count="
                            f"{submitted_prediction_tasks}, expected "
                            f"{expected_p3_task_count}"
                        )
                    if serial_prediction_stages != 0:
                        errors.append(
                            "P3 parallel prediction fixture used serial ranges"
                        )
                    if submitted_writeback_tasks != expected_p3_task_count:
                        errors.append(
                            "P3 writeback task count="
                            f"{submitted_writeback_tasks}, expected "
                            f"{expected_p3_task_count}"
                        )
                    if serial_writeback_stages != 0:
                        errors.append(
                            "P3 parallel writeback fixture used serial ranges"
                        )
                elif execution == "parallel":
                    if submitted_prediction_tasks != 0:
                        errors.append(
                            "single-worker P3 fixture submitted prediction ranges"
                        )
                    if serial_prediction_stages <= 0:
                        errors.append(
                            "single-worker P3 fixture missed serial prediction fallback"
                        )
                    if submitted_writeback_tasks != 0:
                        errors.append(
                            "single-worker P3 fixture submitted writeback ranges"
                        )
                    if serial_writeback_stages <= 0:
                        errors.append(
                            "single-worker P3 fixture missed serial writeback fallback"
                        )
                elif (
                    submitted_prediction_tasks != 0
                    or serial_prediction_stages != 0
                    or
                    submitted_writeback_tasks != 0
                    or serial_writeback_stages != 0
                ):
                    errors.append(
                        "explicit serial P3 reference unexpectedly used P3 staging"
                    )
        except (KeyError, ValueError):
            errors.append("taskgraph telemetry is missing or invalid")
    for key in PERF_INT_KEYS:
        try:
            if int(perf_fields[key]) < 0:
                errors.append(f"perf {key} is negative")
        except (KeyError, ValueError):
            errors.append(
                f"perf {key}={perf_fields.get(key)!r}, expected integer"
            )
    if component_no_contact:
        for key in (
            "overlappingBodyPairs",
            "particleSurfaceCandidates",
            "insideTriangleTests",
            "closestTriangleTests",
            "selfTriangleTests",
            "generatedGroundContacts",
            "generatedRigidContacts",
            "generatedSoftContacts",
            "generatedSelfContacts",
        ):
            if perf_fields.get(key) != "0":
                errors.append(
                    f"perf {key}={perf_fields.get(key)!r}, expected zero "
                    "for a no-contact corpus"
                )
    if collision_telemetry and case_name == "scene-volume-soft-soft-wake":
        try:
            if int(perf_fields["detectionCalls"]) <= 0:
                errors.append(
                    "scene soft-soft telemetry reported no OGC detections"
                )
            if int(perf_fields["generatedSoftContacts"]) <= 0:
                errors.append(
                    "scene soft-soft telemetry reported no soft contacts"
                )
        except (KeyError, ValueError):
            errors.append(
                "scene soft-soft telemetry counters are missing or invalid"
            )
    if case_name == "scene-volume-rigid-triangle-steady-contact":
        for key in (
            "rigidTriangleFaceTests",
            "rigidTriangleEdgeTests",
            "rigidTriangleVertexTests",
        ):
            try:
                if int(perf_fields.get(key, "0")) <= 0:
                    errors.append(
                        f"steady rigid-triangle corpus {key} must be positive"
                    )
            except ValueError:
                errors.append(
                    f"steady rigid-triangle corpus {key} is invalid"
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
        # High-valence meshes have a correct observed-limiter fallback when
        # their incident tet count exceeds the fixed stack cache.  That is a
        # performance property, not a correctness failure.
        if enforce_performance_contract and int(
            perf_fields["tetLinearizationCacheFallbackParticleSteps"]
        ) != 0:
            errors.append(
                "perf tet linearization cache fallback is non-zero"
            )
        local_displacement = float(
            perf_fields["finalMaxLocalSolveDisplacement"]
        )
        applied_displacement = float(
            perf_fields["finalMaxAppliedDisplacement"]
        )
        # The applied displacement is reconstructed by subtracting two
        # world-space float positions.  Allow the corresponding cancellation
        # error while retaining a tight check against genuine amplification.
        displacement_tolerance = max(
            1e-7, abs(local_displacement) * 1e-3
        )
        if applied_displacement > (
            local_displacement + displacement_tolerance
        ):
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
    if rotation_quality_lane is not None and rotation_quality_acceptance:
        # A rotation retention comparison is meaningful only if the two
        # explicitly configured schedules actually reached their intended
        # execution paths.  In particular, do not allow a nominally parallel
        # lane that silently fell back to a serial task graph to pass because
        # it happens to resemble the ordered reference trajectory.
        try:
            soft_scheduler = perf_fields["softScheduler"]
            soft_execution = perf_fields["softExecution"]
            soft_workers = int(perf_fields["softWorkers"])
            actual_soft_workers = int(perf_fields["actualSoftWorkers"])
            completed_tasks = int(perf_fields["taskGraphCompletedTasks"])
            serial_tasks = int(perf_fields["taskGraphSerialTasks"])
            causal_tasks = int(perf_fields["causalLayerTaskCount"])
            completed_causal_tasks = int(
                perf_fields["causalLayerCompletedTasks"]
            )
            causal_peak = int(perf_fields["causalLayerPeakActiveTasks"])
            causal_fallbacks = int(
                perf_fields["causalLayerSerialFallbacks"]
            )
            if soft_scheduler != "sceneTaskgraph":
                errors.append(
                    "rotation-quality acceptance softScheduler="
                    f"{soft_scheduler!r}, expected 'sceneTaskgraph'"
                )
            if rotation_quality_lane == "ordered":
                if soft_execution != "serial" or soft_workers != 1:
                    errors.append(
                        "rotation-quality ordered lane did not report one "
                        "serial soft worker"
                    )
                if actual_soft_workers != 1 or serial_tasks <= 0:
                    errors.append(
                        "rotation-quality ordered lane did not execute the "
                        "serial Scene-taskgraph authority"
                    )
            elif rotation_quality_lane == "fast":
                if (
                    soft_execution != "parallel"
                    or soft_workers < 2
                    or actual_soft_workers < 2
                ):
                    errors.append(
                        "rotation-quality fast lane did not report parallel "
                        "soft execution"
                    )
                if completed_tasks <= 0 or serial_tasks != 0:
                    errors.append(
                        "rotation-quality fast lane did not complete a pure "
                        "parallel Scene taskgraph route"
                    )
                if (
                    causal_tasks <= 0
                    or completed_causal_tasks != causal_tasks
                    or causal_peak < 2
                    or causal_fallbacks != 0
                ):
                    errors.append(
                        "rotation-quality fast lane did not prove a completed "
                        "parallel causal-layer color schedule"
                    )
            else:
                errors.append(
                    "rotation-quality acceptance has unknown lane="
                    f"{rotation_quality_lane!r}"
                )
            if int(perf_fields["generatedGroundContacts"]) <= 0:
                errors.append(
                    "rotation-quality acceptance observed no generated "
                    "ground contacts"
                )
            if int(perf_fields["generatedSoftContacts"]) <= 0:
                errors.append(
                    "rotation-quality acceptance observed no generated "
                    "soft-soft contacts"
                )
        except (KeyError, ValueError):
            errors.append(
                "rotation-quality acceptance is missing or has invalid "
                "scheduler telemetry"
            )
        # The C++ short-run rotation result deliberately does not require the
        # optional long-run damping check.  When an explicit soak reaches the
        # window, make both reported long-horizon bounds part of the strict
        # contract instead of merely printing them as diagnostics.
        if frames >= 4000:
            if fields.get("sphereLongRoll.longRunBounded") != "1":
                errors.append(
                    "rotation-quality long-run longRunBounded is not PASS"
                )
            if fields.get("sphereLongRoll.regressionBounded") != "1":
                errors.append(
                    "rotation-quality long-run regressionBounded is not PASS"
                )
            try:
                if int(fields["sphereLongRoll.windowSamples"]) <= 0:
                    errors.append(
                        "rotation-quality long-run emitted no angular-speed "
                        "window samples"
                    )
            except (KeyError, ValueError):
                errors.append(
                    "rotation-quality long-run windowSamples is invalid"
                )
    try:
        if float(perf_fields["closureMs"]) > float(
            perf_fields["avgStepMs"]
        ) * 1.05:
            errors.append("perf closure exceeds avgStepMs by more than 5%")
    except (KeyError, ValueError):
        pass
    if rotation_quality_lane is not None and not rotation_quality_acceptance:
        # A long-roll physics gate intentionally returns 1 on a failed
        # trajectory.  Keep that value visible in the telemetry but do not
        # conflate it with probe infrastructure failure; a crash/other exit
        # code is still fatal to the measurement.
        errors = list(rotation_quality_infrastructure_errors or ())
        if result.returncode not in (0, 1):
            errors.append(
                f"rotation-quality exit code {result.returncode}, "
                "expected 0 or 1"
            )
    elif rotation_quality_lane is not None:
        # The explicit acceptance lane retains the complete ordinary gate
        # validation above, then makes its two long-roll sources and process
        # outcome non-negotiable.  This must not inherit the diagnostic
        # probe's deliberate rc=1 tolerance.
        if fields.get("status") != "PASS":
            errors.append(
                "rotation-quality acceptance gate status="
                f"{fields.get('status')!r}, expected 'PASS'"
            )
        if fields.get("sphereLongRoll.result") != "PASS":
            errors.append(
                "rotation-quality acceptance long-roll result="
                f"{fields.get('sphereLongRoll.result')!r}, expected 'PASS'"
            )
        if result.returncode != 0:
            errors.append(
                "rotation-quality acceptance exit code "
                f"{result.returncode}, expected 0"
            )
    elif result.returncode != 0:
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


def get_source_revision() -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def get_measurement_source_hash() -> str:
    """Hash the files which define the observed Volume measurement contract."""
    digest = hashlib.sha256()
    try:
        for relative_path in MEASUREMENT_SOURCE_PATHS:
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update((REPO_ROOT / relative_path).read_bytes())
            digest.update(b"\0")
    except OSError:
        return "unknown"
    return digest.hexdigest()


def write_performance_json(
    path: Path,
    args: argparse.Namespace,
    selected_cases: tuple[str, ...],
    warmup: int,
    required_perf_schema: str,
    perf_results: dict[tuple[str, int], dict[str, str]],
) -> None:
    results: list[dict[str, object]] = []
    for case_name in selected_cases:
        repeats = [
            perf_results[(case_name, repeat)]
            for repeat in range(1, args.repeats_effective + 1)
        ]
        summary: dict[str, float] = {}
        for key in ("avgStepMs", "p50StepMs", "p95StepMs", "maxStepMs",
                    "solverMs", "particleSolveMs", "redetectMs"):
            try:
                summary[f"median{key[0].upper()}{key[1:]}"] = (
                    statistics.median(float(result[key]) for result in repeats)
                )
            except (KeyError, ValueError, statistics.StatisticsError):
                summary[f"median{key[0].upper()}{key[1:]}"] = math.nan
        results.append(
            {
                "case": case_name,
                "summary": summary,
                "repeats": repeats,
            }
        )
    first = next(iter(perf_results.values()), {})
    payload = {
        "schema": 1,
        "sourceRevision": get_source_revision(),
        "measurementSourceSha256": get_measurement_source_hash(),
        "measurementSourcePaths": MEASUREMENT_SOURCE_PATHS,
        "runner": "run_snippet_deformable_volume_avbd_headless.py",
        "mode": args.mode,
        "config": {
            "frames": args.frames,
            "warmupFrames": warmup,
            "repeats": args.repeats_effective,
            "execution": args.execution,
            "dispatcherThreads": args.dispatcher_threads,
            "requiredPerfSchema": required_perf_schema,
            "collisionTelemetry": args.collision_telemetry,
            "surfaceTriangleBvh": args.surface_triangle_bvh,
            "rigidTriangleBvh": args.rigid_triangle_bvh,
            "rigidTriangleGridDim": args.rigid_triangle_grid_dim,
            "binDir": str(args.bin_dir.resolve()),
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logicalCores": os.cpu_count(),
            "physicalCores": first.get("physicalCores", "unknown"),
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[DEFORMABLE_VOLUME_AVBD_PERF_JSON] "
        f"path={path.resolve()} status=PASS"
    )


def print_rotation_quality_measurement(
    lane: str,
    repeat: int,
    execution: str,
    primal_schedule: str,
    run_passed: bool,
    fields: dict[str, str],
    perf_fields: dict[str, str],
    strict_acceptance: bool,
) -> None:
    """Print parsed telemetry for one explicitly configured rotation lane."""
    metric = lambda key: fields.get(f"sphereLongRoll.{key}", "MISSING")
    measured_status = (
        f"acceptancePreRatioStatus={'PASS' if run_passed else 'FAIL'}"
        if strict_acceptance
        else f"infrastructureStatus={'PASS' if run_passed else 'FAIL'}"
    )
    print(
        "[DEFORMABLE_VOLUME_AVBD_ROTATION_QUALITY] "
        f"lane={lane} repeat={repeat} execution={execution} "
        f"primalSchedule={primal_schedule} "
        "optionalSoftOverrides=cleared "
        f"{measured_status} "
        f"gateStatus={fields.get('status', 'MISSING')} "
        f"frames={metric('frames')} "
        f"maxOrientationChange={metric('maxOrientationChange')} "
        f"maxAngularSpeed={metric('maxAngularSpeed')} "
        f"maxAngularSpeedFrame={metric('maxAngularSpeedFrame')} "
        f"finalAngularSpeed={metric('finalAngularSpeed')} "
        f"windowBegin={metric('windowBegin')} "
        f"windowEnd={metric('windowEnd')} "
        f"windowSamples={metric('windowSamples')} "
        f"windowMinAngularSpeed={metric('windowMinAngularSpeed')} "
        f"windowMeanAngularSpeed={metric('windowMeanAngularSpeed')} "
        f"windowMaxAngularSpeed={metric('windowMaxAngularSpeed')} "
        f"finalLinearSpeed={metric('finalLinearSpeed')} "
        f"windowMinLinearSpeed={metric('windowMinLinearSpeed')} "
        f"windowMeanLinearSpeed={metric('windowMeanLinearSpeed')} "
        f"windowMaxLinearSpeed={metric('windowMaxLinearSpeed')} "
        f"longRunBounded={metric('longRunBounded')} "
        f"regressionBounded={metric('regressionBounded')} "
        f"trajectoryResult={metric('result')} "
        f"generatedGroundContacts="
        f"{perf_fields.get('generatedGroundContacts', 'MISSING')} "
        f"generatedSoftContacts="
        f"{perf_fields.get('generatedSoftContacts', 'MISSING')} "
        f"softScheduler={perf_fields.get('softScheduler', 'MISSING')} "
        f"softWorkers={perf_fields.get('softWorkers', 'MISSING')} "
        f"avgStepMs={perf_fields.get('avgStepMs', 'MISSING')}"
    )


def median_rotation_quality_metric(
    results: list[dict[str, str]], key: str
) -> float:
    return statistics.median(
        float(result[f"sphereLongRoll.{key}"])
        for result in results
    )


def run_rotation_quality_probe(
    args: argparse.Namespace,
    bin_dir: Path,
    warmup: int,
    required_perf_schema: str,
    repeats: int,
    *,
    strict_acceptance: bool = False,
) -> int:
    """Measure the two long-roll lanes, optionally as an explicit gate."""
    lanes = (
        ("ordered", "sequential", "serial"),
        ("fast", "parallel", "relaxed-color"),
    )
    lane_results: dict[str, list[dict[str, str]]] = {
        lane: [] for lane, _, _ in lanes
    }
    lane_perf_results: dict[str, list[dict[str, str]]] = {
        lane: [] for lane, _, _ in lanes
    }
    passed = True
    for repeat in range(1, repeats + 1):
        for lane, execution, primal_schedule in lanes:
            run_passed, fields, perf_fields = run_one(
                SPHERE_LONG_ROLL_CASE,
                repeat,
                bin_dir,
                args.frames,
                args.timeout,
                execution,
                warmup,
                args.dispatcher_threads,
                required_perf_schema,
                args.collision_telemetry,
                args.surface_triangle_bvh,
                args.rigid_triangle_bvh,
                args.rigid_triangle_grid_dim,
                False,
                rotation_quality_lane=lane,
                rotation_quality_acceptance=strict_acceptance,
            )
            passed = passed and run_passed
            lane_results[lane].append(fields)
            lane_perf_results[lane].append(perf_fields)
            print_rotation_quality_measurement(
                lane,
                repeat,
                execution,
                primal_schedule,
                run_passed,
                fields,
                perf_fields,
                strict_acceptance,
            )

    if strict_acceptance:
        try:
            for repeat, (ordered, fast) in enumerate(
                zip(lane_results["ordered"], lane_results["fast"]), start=1
            ):
                ordered_orientation = float(
                    ordered["sphereLongRoll.maxOrientationChange"]
                )
                fast_orientation = float(
                    fast["sphereLongRoll.maxOrientationChange"]
                )
                ordered_omega = float(ordered["sphereLongRoll.maxAngularSpeed"])
                fast_omega = float(fast["sphereLongRoll.maxAngularSpeed"])
                orientation_ratio = (
                    fast_orientation / ordered_orientation
                    if ordered_orientation > 0.0
                    else math.nan
                )
                omega_ratio = (
                    fast_omega / ordered_omega
                    if ordered_omega > 0.0
                    else math.nan
                )
                ratios_passed = (
                    math.isfinite(orientation_ratio)
                    and math.isfinite(omega_ratio)
                    and orientation_ratio
                    >= ROTATION_QUALITY_ACCEPTANCE_MIN_FAST_OVER_ORDERED
                    and omega_ratio
                    >= ROTATION_QUALITY_ACCEPTANCE_MIN_FAST_OVER_ORDERED
                )
                passed = passed and ratios_passed
                print(
                    "[DEFORMABLE_VOLUME_AVBD_ROTATION_QUALITY_ACCEPTANCE] "
                    f"repeat={repeat} "
                    f"orderedMaxOrientationChange={ordered_orientation:.9g} "
                    f"fastMaxOrientationChange={fast_orientation:.9g} "
                    f"fastOverOrderedOrientation={orientation_ratio:.9g} "
                    f"orderedMaxAngularSpeed={ordered_omega:.9g} "
                    f"fastMaxAngularSpeed={fast_omega:.9g} "
                    f"fastOverOrderedMaxAngularSpeed={omega_ratio:.9g} "
                    "minimumFastOverOrdered="
                    f"{ROTATION_QUALITY_ACCEPTANCE_MIN_FAST_OVER_ORDERED:.9g} "
                    f"status={'PASS' if ratios_passed else 'FAIL'}"
                )
        except (KeyError, ValueError):
            passed = False
            print(
                "[DEFORMABLE_VOLUME_AVBD_ROTATION_QUALITY_ACCEPTANCE] "
                "status=FAIL error=missing-or-invalid-telemetry"
            )

    # Probe ratios are diagnostic only.  The strict mode above evaluates the
    # same two metrics per repeat, so a bad run cannot be hidden by a median.
    try:
        ordered = lane_results["ordered"]
        fast = lane_results["fast"]
        ordered_orientation = median_rotation_quality_metric(
            ordered, "maxOrientationChange"
        )
        fast_orientation = median_rotation_quality_metric(
            fast, "maxOrientationChange"
        )
        # The standard 600-frame reproduction finishes before the optional
        # long-run window begins, so max angular speed is the comparable
        # rotation measure for both the short and long probe invocations.
        ordered_omega = median_rotation_quality_metric(
            ordered, "maxAngularSpeed"
        )
        fast_omega = median_rotation_quality_metric(
            fast, "maxAngularSpeed"
        )
        ordered_step = statistics.median(
            float(result["avgStepMs"])
            for result in lane_perf_results["ordered"]
        )
        fast_step = statistics.median(
            float(result["avgStepMs"])
            for result in lane_perf_results["fast"]
        )
        orientation_ratio = (
            fast_orientation / ordered_orientation
            if ordered_orientation != 0.0
            else math.nan
        )
        omega_ratio = fast_omega / ordered_omega if ordered_omega != 0.0 else math.nan
        step_ratio = fast_step / ordered_step if ordered_step != 0.0 else math.nan
        print(
            "[DEFORMABLE_VOLUME_AVBD_ROTATION_QUALITY_COMPARISON] "
            f"repeats={repeats} "
            f"orderedMedianMaxOrientationChange={ordered_orientation:.9g} "
            f"fastMedianMaxOrientationChange={fast_orientation:.9g} "
            f"fastOverOrderedOrientation={orientation_ratio:.9g} "
            f"orderedMedianMaxAngularSpeed={ordered_omega:.9g} "
            f"fastMedianMaxAngularSpeed={fast_omega:.9g} "
            f"fastOverOrderedMaxAngularSpeed={omega_ratio:.9g} "
            f"orderedMedianAvgStepMs={ordered_step:.9g} "
            f"fastMedianAvgStepMs={fast_step:.9g} "
            f"fastOverOrderedAvgStepMs={step_ratio:.9g} "
            f"acceptance={'ENFORCED' if strict_acceptance else 'UNCHANGED'}"
        )
    except (KeyError, ValueError, statistics.StatisticsError):
        # The per-lane run has already emitted its precise parser error.  Keep
        # the probe's terminal status honest when no comparison is possible.
        passed = False
        print(
            "[DEFORMABLE_VOLUME_AVBD_ROTATION_QUALITY_COMPARISON] "
            f"{'status' if strict_acceptance else 'infrastructureStatus'}=FAIL "
            "error=missing-or-invalid-telemetry "
            f"acceptance={'ENFORCED' if strict_acceptance else 'UNCHANGED'}"
        )
    if strict_acceptance:
        print(
            "[DEFORMABLE_VOLUME_AVBD_ROTATION_QUALITY_ACCEPTANCE_SUMMARY] "
            f"runs={2 * repeats} status={'PASS' if passed else 'FAIL'} "
            "acceptance=ENFORCED"
        )
    else:
        print(
            "[DEFORMABLE_VOLUME_AVBD_ROTATION_QUALITY_SUMMARY] "
            f"runs={2 * repeats} "
            f"infrastructureStatus={'PASS' if passed else 'FAIL'} "
            "acceptance=UNCHANGED"
        )
    return 0 if passed else 1


def soft_soft_torque_lane_errors(
    lane: str,
    execution: str,
    fields: dict[str, str],
    perf_fields: dict[str, str],
    *,
    strict_acceptance: bool,
) -> list[str]:
    """Verify that a torque observation came only from the intended OGC path."""
    errors: list[str] = []
    metric = lambda key: fields.get(f"softSoftTorque.{key}")
    for key, expected in (
        ("isolated", "1"),
        ("targetDistinctCollisionSimulation", "1"),
        ("driverDistinctCollisionSimulation", "1"),
        ("generatedGroundContacts", "0"),
        ("generatedRigidContacts", "0"),
        ("generatedSelfContacts", "0"),
        ("retentionPassed", "1"),
        ("result", "PASS"),
    ):
        if metric(key) != expected:
            errors.append(
                f"soft-soft torque {lane} {key}={metric(key)!r}, "
                f"expected {expected!r}"
            )
    try:
        contacts = int(metric("generatedSoftContacts") or "")
        contact_frames = int(metric("softContactFrames") or "")
        first_contact = int(metric("firstContactFrame") or "")
        first_rotation = int(metric("firstRotationFrame") or "")
        retained = int(metric("retainedRotationSamples") or "")
        angular_momentum = float(metric("targetMaxAngularMomentum") or "")
        angular_speed = float(metric("targetMaxAngularSpeed") or "")
        lever_arm = float(metric("firstContactCentroidLeverArm") or "")
        if contacts <= 0 or contact_frames <= 0:
            errors.append(
                f"soft-soft torque {lane} reported no soft contact"
            )
        if first_contact < 0 or first_rotation < first_contact:
            errors.append(
                f"soft-soft torque {lane} contact/rotation ordering is invalid"
            )
        if angular_momentum <= 0.0 or angular_speed <= 0.0 or lever_arm <= 0.0:
            errors.append(
                f"soft-soft torque {lane} reported no target torque state"
            )
        if strict_acceptance and retained < SOFT_SOFT_TORQUE_ACCEPTANCE_MIN_RETENTION_SAMPLES:
            errors.append(
                f"soft-soft torque {lane} retainedRotationSamples={retained}, "
                "below the acceptance minimum"
            )
    except ValueError:
        errors.append(f"soft-soft torque {lane} telemetry is invalid")
    try:
        if int(perf_fields["detectionCalls"]) <= 0:
            errors.append(f"soft-soft torque {lane} observed no OGC detections")
        if int(perf_fields["generatedSoftContacts"]) <= 0:
            errors.append(f"soft-soft torque {lane} observed no OGC soft contact")
        for key in (
            "generatedGroundContacts",
            "generatedRigidContacts",
            "generatedSelfContacts",
        ):
            if int(perf_fields[key]) != 0:
                errors.append(
                    f"soft-soft torque {lane} perf {key}="
                    f"{perf_fields[key]!r}, expected '0'"
                )
        soft_scheduler = perf_fields["softScheduler"]
        soft_execution = perf_fields["softExecution"]
        soft_workers = int(perf_fields["softWorkers"])
        actual_soft_workers = int(perf_fields["actualSoftWorkers"])
        submitted_tasks = int(perf_fields["taskCount"])
        completed_tasks = int(perf_fields["taskGraphCompletedTasks"])
        serial_tasks = int(perf_fields["taskGraphSerialTasks"])
        causal_tasks = int(perf_fields["causalLayerTaskCount"])
        completed_causal_tasks = int(perf_fields["causalLayerCompletedTasks"])
        causal_fallbacks = int(perf_fields["causalLayerSerialFallbacks"])
        if perf_fields.get("sceneExecution") != execution:
            errors.append(
                f"soft-soft torque {lane} sceneExecution="
                f"{perf_fields.get('sceneExecution')!r}, expected {execution!r}"
            )
        if soft_scheduler != "sceneTaskgraph":
            errors.append(
                f"soft-soft torque {lane} softScheduler={soft_scheduler!r}, "
                "expected 'sceneTaskgraph'"
            )
        if lane == "ordered":
            if soft_execution != "serial" or soft_workers != 1:
                errors.append(
                    "soft-soft torque ordered lane did not report serial "
                    "soft execution"
                )
            if actual_soft_workers != 1 or serial_tasks <= 0:
                errors.append(
                    "soft-soft torque ordered lane did not execute the serial "
                    "Scene-taskgraph authority"
                )
        elif lane == "fast":
            if (
                soft_execution != "parallel"
                or soft_workers < 2
                or actual_soft_workers < 2
            ):
                errors.append(
                    "soft-soft torque fast lane did not report relaxed parallel "
                    "soft execution"
                )
            if submitted_tasks <= 0 or completed_tasks != submitted_tasks or serial_tasks != 0:
                errors.append(
                    "soft-soft torque fast lane did not complete a parallel "
                    "Scene-taskgraph route"
                )
            if (
                causal_tasks <= 0
                or completed_causal_tasks != causal_tasks
                or causal_fallbacks != 0
            ):
                errors.append(
                    "soft-soft torque fast lane did not prove a completed "
                    "relaxed causal-layer schedule"
                )
        else:
            errors.append(f"soft-soft torque has unknown lane={lane!r}")
    except (KeyError, ValueError):
        errors.append(f"soft-soft torque {lane} scheduler telemetry is invalid")
    return errors


def print_soft_soft_torque_measurement(
    lane: str,
    repeat: int,
    execution: str,
    primal_schedule: str,
    run_passed: bool,
    fields: dict[str, str],
    perf_fields: dict[str, str],
    lane_errors: list[str],
    strict_acceptance: bool,
) -> None:
    metric = lambda key: fields.get(f"softSoftTorque.{key}", "MISSING")
    print(
        "[DEFORMABLE_VOLUME_AVBD_SOFT_SOFT_TORQUE] "
        f"lane={lane} repeat={repeat} execution={execution} "
        f"primalSchedule={primal_schedule} "
        f"runStatus={'PASS' if run_passed else 'FAIL'} "
        f"gateStatus={fields.get('status', 'MISSING')} "
        f"isolated={metric('isolated')} "
        "targetDistinctCollisionSimulation="
        f"{metric('targetDistinctCollisionSimulation')} "
        "driverDistinctCollisionSimulation="
        f"{metric('driverDistinctCollisionSimulation')} "
        f"softContactFrames={metric('softContactFrames')} "
        f"generatedSoftContacts={metric('generatedSoftContacts')} "
        f"generatedGroundContacts={metric('generatedGroundContacts')} "
        f"generatedRigidContacts={metric('generatedRigidContacts')} "
        f"generatedSelfContacts={metric('generatedSelfContacts')} "
        f"firstContactFrame={metric('firstContactFrame')} "
        f"firstRotationFrame={metric('firstRotationFrame')} "
        "firstContactCentroidLeverArm="
        f"{metric('firstContactCentroidLeverArm')} "
        f"targetMaxAngularMomentum={metric('targetMaxAngularMomentum')} "
        f"targetMaxAngularSpeed={metric('targetMaxAngularSpeed')} "
        f"retainedRotationSamples={metric('retainedRotationSamples')} "
        f"retentionPassed={metric('retentionPassed')} "
        "supportExpansionInstrumentation="
        f"{metric('supportExpansionInstrumentation')} "
        f"perfGeneratedSoftContacts={perf_fields.get('generatedSoftContacts', 'MISSING')} "
        f"softExecution={perf_fields.get('softExecution', 'MISSING')} "
        f"softWorkers={perf_fields.get('softWorkers', 'MISSING')} "
        f"validation={'PASS' if not lane_errors else 'FAIL'} "
        f"acceptance={'ENFORCED' if strict_acceptance else 'PROBE'}"
    )
    for error in lane_errors:
        print(f"[DEFORMABLE_VOLUME_AVBD_SOFT_SOFT_TORQUE_ERROR] lane={lane} {error}")


def run_soft_soft_torque_probe(
    args: argparse.Namespace,
    bin_dir: Path,
    warmup: int,
    required_perf_schema: str,
    repeats: int,
    *,
    strict_acceptance: bool = False,
) -> int:
    """Run the no-ground true-boundary torque fixture through both schedules."""
    lanes = (
        ("ordered", "sequential", "serial"),
        ("fast", "parallel", "relaxed-color"),
    )
    lane_results: dict[str, list[dict[str, str]]] = {
        lane: [] for lane, _, _ in lanes
    }
    lane_passed = True
    for repeat in range(1, repeats + 1):
        for lane, execution, primal_schedule in lanes:
            run_passed, fields, perf_fields = run_one(
                SOFT_SOFT_TORQUE_CASE,
                repeat,
                bin_dir,
                args.frames,
                args.timeout,
                execution,
                warmup,
                args.dispatcher_threads,
                required_perf_schema,
                False,
                args.surface_triangle_bvh,
                args.rigid_triangle_bvh,
                args.rigid_triangle_grid_dim,
                False,
                soft_soft_torque_lane=lane,
            )
            lane_errors = soft_soft_torque_lane_errors(
                lane,
                execution,
                fields,
                perf_fields,
                strict_acceptance=strict_acceptance,
            )
            lane_passed = lane_passed and run_passed and not lane_errors
            lane_results[lane].append(fields)
            print_soft_soft_torque_measurement(
                lane,
                repeat,
                execution,
                primal_schedule,
                run_passed,
                fields,
                perf_fields,
                lane_errors,
                strict_acceptance,
            )

    # The individual lanes prove that both schedules generate torque.  Strict
    # acceptance also requires the relaxed parallel schedule to retain the
    # ordered schedule's physically observable torque response on every
    # repeat, so one outlier cannot be hidden by the diagnostic median below.
    ratios_passed = True
    try:
        for repeat, (ordered, fast) in enumerate(
            zip(lane_results["ordered"], lane_results["fast"]), start=1
        ):
            ordered_momentum = float(
                ordered["softSoftTorque.targetMaxAngularMomentum"]
            )
            fast_momentum = float(
                fast["softSoftTorque.targetMaxAngularMomentum"]
            )
            ordered_omega = float(
                ordered["softSoftTorque.targetMaxAngularSpeed"]
            )
            fast_omega = float(
                fast["softSoftTorque.targetMaxAngularSpeed"]
            )
            momentum_ratio = (
                fast_momentum / ordered_momentum
                if ordered_momentum > 0.0
                else math.nan
            )
            omega_ratio = (
                fast_omega / ordered_omega if ordered_omega > 0.0 else math.nan
            )
            this_ratio_passed = (
                math.isfinite(momentum_ratio)
                and math.isfinite(omega_ratio)
                and momentum_ratio
                >= SOFT_SOFT_TORQUE_ACCEPTANCE_MIN_FAST_OVER_ORDERED
                and omega_ratio
                >= SOFT_SOFT_TORQUE_ACCEPTANCE_MIN_FAST_OVER_ORDERED
            )
            if strict_acceptance:
                ratios_passed = ratios_passed and this_ratio_passed
            print(
                "[DEFORMABLE_VOLUME_AVBD_SOFT_SOFT_TORQUE_ACCEPTANCE] "
                f"repeat={repeat} "
                f"orderedTargetMaxAngularMomentum={ordered_momentum:.9g} "
                f"fastTargetMaxAngularMomentum={fast_momentum:.9g} "
                f"fastOverOrderedAngularMomentum={momentum_ratio:.9g} "
                f"orderedTargetMaxAngularSpeed={ordered_omega:.9g} "
                f"fastTargetMaxAngularSpeed={fast_omega:.9g} "
                f"fastOverOrderedAngularSpeed={omega_ratio:.9g} "
                "minimumFastOverOrdered="
                f"{SOFT_SOFT_TORQUE_ACCEPTANCE_MIN_FAST_OVER_ORDERED:.9g} "
                f"status={'PASS' if this_ratio_passed else 'FAIL'} "
                f"acceptance={'ENFORCED' if strict_acceptance else 'PROBE'}"
            )
    except (KeyError, ValueError):
        ratios_passed = False
        print(
            "[DEFORMABLE_VOLUME_AVBD_SOFT_SOFT_TORQUE_ACCEPTANCE] "
            "status=FAIL error=missing-or-invalid-telemetry "
            f"acceptance={'ENFORCED' if strict_acceptance else 'PROBE'}"
        )
    lane_passed = lane_passed and (ratios_passed or not strict_acceptance)
    try:
        ordered_momentum = statistics.median(
            float(item["softSoftTorque.targetMaxAngularMomentum"])
            for item in lane_results["ordered"]
        )
        fast_momentum = statistics.median(
            float(item["softSoftTorque.targetMaxAngularMomentum"])
            for item in lane_results["fast"]
        )
        ordered_omega = statistics.median(
            float(item["softSoftTorque.targetMaxAngularSpeed"])
            for item in lane_results["ordered"]
        )
        fast_omega = statistics.median(
            float(item["softSoftTorque.targetMaxAngularSpeed"])
            for item in lane_results["fast"]
        )
        print(
            "[DEFORMABLE_VOLUME_AVBD_SOFT_SOFT_TORQUE_COMPARISON] "
            f"repeats={repeats} "
            f"orderedMedianTargetMaxAngularMomentum={ordered_momentum:.9g} "
            f"fastMedianTargetMaxAngularMomentum={fast_momentum:.9g} "
            "fastOverOrderedAngularMomentum="
            f"{(fast_momentum / ordered_momentum if ordered_momentum else math.nan):.9g} "
            f"orderedMedianTargetMaxAngularSpeed={ordered_omega:.9g} "
            f"fastMedianTargetMaxAngularSpeed={fast_omega:.9g} "
            "fastOverOrderedAngularSpeed="
            f"{(fast_omega / ordered_omega if ordered_omega else math.nan):.9g} "
            f"acceptance={'ENFORCED' if strict_acceptance else 'PROBE'}"
        )
    except (KeyError, ValueError, statistics.StatisticsError):
        lane_passed = False
        print(
            "[DEFORMABLE_VOLUME_AVBD_SOFT_SOFT_TORQUE_COMPARISON] "
            "status=FAIL error=missing-or-invalid-telemetry"
        )
    print(
        "[DEFORMABLE_VOLUME_AVBD_SOFT_SOFT_TORQUE_SUMMARY] "
        f"runs={2 * repeats} status={'PASS' if lane_passed else 'FAIL'} "
        f"acceptance={'ENFORCED' if strict_acceptance else 'PROBE'}"
    )
    return 0 if lane_passed else 1


def sphere_soft_soft_glancing_lane_errors(
    lane: str,
    execution: str,
    fields: dict[str, str],
    perf_fields: dict[str, str],
) -> list[str]:
    """Validate the no-ground public cube/sphere contact-phase control."""
    errors: list[str] = []
    metric = lambda key: fields.get(f"sphereSoftSoftGlancing.{key}")
    for key, expected in (
        ("contactTelemetry", "enabled"),
        ("preSoftContactSample", "1"),
        ("generatedGroundContacts", "0"),
        ("result", "PASS"),
    ):
        if metric(key) != expected:
            errors.append(
                f"sphere soft-soft glancing {lane} {key}={metric(key)!r}, "
                f"expected {expected!r}"
            )
    try:
        contacts = int(metric("generatedSoftContacts") or "")
        contact_frames = int(metric("softContactFrames") or "")
        first_contact = int(metric("firstSoftContactFrame") or "")
        last_contact = int(metric("lastSoftContactFrame") or "")
        peak_contact = int(metric("peakSoftContactFrame") or "")
        delta_momentum = float(metric("deltaAngularMomentum") or "")
        delta_speed = float(metric("deltaAngularSpeed") or "")
        delta_velocity = tuple(
            float(metric(f"deltaAngularVelocity{axis}") or "")
            for axis in ("X", "Y", "Z")
        )
        delta_velocity_norm = math.sqrt(
            sum(component * component for component in delta_velocity)
        )
        if contacts <= 0 or contact_frames <= 0 or contact_frames > int(
            metric("frames") or ""
        ):
            errors.append(
                f"sphere soft-soft glancing {lane} reported no soft contact"
            )
        if (
            first_contact < 0
            or last_contact < first_contact
            or peak_contact < first_contact
            or peak_contact > last_contact
            or first_contact >= int(metric("frames") or "")
            or last_contact >= int(metric("frames") or "")
        ):
            errors.append(
                f"sphere soft-soft glancing {lane} contact/peak ordering is invalid"
            )
        if (
            delta_momentum <= 0.0
            or delta_speed < SPHERE_SOFT_SOFT_GLANCING_MIN_DELTA_SPEED
            or delta_velocity_norm
            < SPHERE_SOFT_SOFT_GLANCING_MIN_DELTA_SPEED
        ):
            errors.append(
                f"sphere soft-soft glancing {lane} reported no contact-phase torque"
            )
    except ValueError:
        errors.append(f"sphere soft-soft glancing {lane} telemetry is invalid")
    try:
        if int(perf_fields["detectionCalls"]) <= 0:
            errors.append(
                f"sphere soft-soft glancing {lane} observed no OGC detections"
            )
        if int(perf_fields["generatedSoftContacts"]) <= 0:
            errors.append(
                f"sphere soft-soft glancing {lane} observed no OGC soft contact"
            )
        for key in (
            "generatedGroundContacts",
            "generatedRigidContacts",
            "generatedSelfContacts",
        ):
            if int(perf_fields[key]) != 0:
                errors.append(
                    f"sphere soft-soft glancing {lane} perf {key}="
                    f"{perf_fields[key]!r}, expected '0'"
                )
        soft_scheduler = perf_fields["softScheduler"]
        soft_execution = perf_fields["softExecution"]
        soft_workers = int(perf_fields["softWorkers"])
        actual_soft_workers = int(perf_fields["actualSoftWorkers"])
        submitted_tasks = int(perf_fields["taskCount"])
        completed_tasks = int(perf_fields["taskGraphCompletedTasks"])
        serial_tasks = int(perf_fields["taskGraphSerialTasks"])
        causal_tasks = int(perf_fields["causalLayerTaskCount"])
        completed_causal_tasks = int(perf_fields["causalLayerCompletedTasks"])
        causal_peak_tasks = int(perf_fields["causalLayerPeakActiveTasks"])
        causal_fallbacks = int(perf_fields["causalLayerSerialFallbacks"])
        if perf_fields.get("sceneExecution") != execution:
            errors.append(
                f"sphere soft-soft glancing {lane} sceneExecution="
                f"{perf_fields.get('sceneExecution')!r}, expected {execution!r}"
            )
        if soft_scheduler != "sceneTaskgraph":
            errors.append(
                "sphere soft-soft glancing "
                f"{lane} softScheduler={soft_scheduler!r}, "
                "expected 'sceneTaskgraph'"
            )
        if lane == "ordered":
            if soft_execution != "serial" or soft_workers != 1:
                errors.append(
                    "sphere soft-soft glancing ordered lane did not report serial "
                    "soft execution"
                )
            if actual_soft_workers != 1 or serial_tasks <= 0:
                errors.append(
                    "sphere soft-soft glancing ordered lane did not execute the "
                    "serial Scene-taskgraph authority"
                )
        elif lane == "fast":
            if (
                soft_execution != "parallel"
                or soft_workers < 2
                or actual_soft_workers < 2
            ):
                errors.append(
                    "sphere soft-soft glancing fast lane did not report relaxed "
                    "parallel soft execution"
                )
            if (
                submitted_tasks <= 0
                or completed_tasks != submitted_tasks
                or serial_tasks != 0
            ):
                errors.append(
                    "sphere soft-soft glancing fast lane did not complete a "
                    "parallel Scene-taskgraph route"
                )
            if (
                causal_tasks <= 0
                or completed_causal_tasks != causal_tasks
                or causal_fallbacks != 0
                or causal_peak_tasks < 2
            ):
                errors.append(
                    "sphere soft-soft glancing fast lane did not prove a completed "
                    "relaxed causal-layer schedule"
                )
        else:
            errors.append(f"sphere soft-soft glancing has unknown lane={lane!r}")
    except (KeyError, ValueError):
        errors.append(
            f"sphere soft-soft glancing {lane} scheduler telemetry is invalid"
        )
    return errors


def print_sphere_soft_soft_glancing_measurement(
    lane: str,
    repeat: int,
    execution: str,
    primal_schedule: str,
    run_passed: bool,
    fields: dict[str, str],
    perf_fields: dict[str, str],
    lane_errors: list[str],
    strict_acceptance: bool,
) -> None:
    metric = lambda key: fields.get(f"sphereSoftSoftGlancing.{key}", "MISSING")
    print(
        "[DEFORMABLE_VOLUME_AVBD_SPHERE_SOFT_SOFT_GLANCING] "
        f"lane={lane} repeat={repeat} execution={execution} "
        f"primalSchedule={primal_schedule} "
        f"runStatus={'PASS' if run_passed else 'FAIL'} "
        f"gateStatus={fields.get('status', 'MISSING')} "
        f"contactTelemetry={metric('contactTelemetry')} "
        f"preSoftContactSample={metric('preSoftContactSample')} "
        f"firstSoftContactFrame={metric('firstSoftContactFrame')} "
        f"lastSoftContactFrame={metric('lastSoftContactFrame')} "
        f"peakSoftContactFrame={metric('peakSoftContactFrame')} "
        f"softContactFrames={metric('softContactFrames')} "
        f"generatedSoftContacts={metric('generatedSoftContacts')} "
        f"generatedGroundContacts={metric('generatedGroundContacts')} "
        f"preSoftAngularMomentum={metric('preSoftAngularMomentum')} "
        f"peakSoftContactAngularMomentum={metric('peakSoftContactAngularMomentum')} "
        f"deltaAngularMomentum={metric('deltaAngularMomentum')} "
        f"preSoftAngularSpeed={metric('preSoftAngularSpeed')} "
        f"peakSoftContactAngularSpeed={metric('peakSoftContactAngularSpeed')} "
        f"deltaAngularSpeed={metric('deltaAngularSpeed')} "
        f"deltaAngularVelocityX={metric('deltaAngularVelocityX')} "
        f"deltaAngularVelocityY={metric('deltaAngularVelocityY')} "
        f"deltaAngularVelocityZ={metric('deltaAngularVelocityZ')} "
        f"perfGeneratedSoftContacts={perf_fields.get('generatedSoftContacts', 'MISSING')} "
        f"softExecution={perf_fields.get('softExecution', 'MISSING')} "
        f"softWorkers={perf_fields.get('softWorkers', 'MISSING')} "
        f"validation={'PASS' if not lane_errors else 'FAIL'} "
        f"acceptance={'ENFORCED' if strict_acceptance else 'PROBE'}"
    )
    for error in lane_errors:
        print(
            "[DEFORMABLE_VOLUME_AVBD_SPHERE_SOFT_SOFT_GLANCING_ERROR] "
            f"lane={lane} {error}"
        )


def run_sphere_soft_soft_glancing_probe(
    args: argparse.Namespace,
    bin_dir: Path,
    warmup: int,
    required_perf_schema: str,
    repeats: int,
    *,
    strict_acceptance: bool = False,
) -> int:
    """Compare contact-phase torque transfer under ordered and relaxed schedules."""
    lanes = (
        ("ordered", "sequential", "serial"),
        ("fast", "parallel", "relaxed-color"),
    )
    lane_results: dict[str, list[dict[str, str]]] = {
        lane: [] for lane, _, _ in lanes
    }
    lane_passed = True
    for repeat in range(1, repeats + 1):
        for lane, execution, primal_schedule in lanes:
            run_passed, fields, perf_fields = run_one(
                SPHERE_SOFT_SOFT_GLANCING_CASE,
                repeat,
                bin_dir,
                args.frames,
                args.timeout,
                execution,
                warmup,
                args.dispatcher_threads,
                required_perf_schema,
                False,
                args.surface_triangle_bvh,
                args.rigid_triangle_bvh,
                args.rigid_triangle_grid_dim,
                False,
                sphere_soft_soft_glancing_lane=lane,
            )
            lane_errors = sphere_soft_soft_glancing_lane_errors(
                lane, execution, fields, perf_fields
            )
            lane_passed = lane_passed and run_passed and not lane_errors
            lane_results[lane].append(fields)
            print_sphere_soft_soft_glancing_measurement(
                lane,
                repeat,
                execution,
                primal_schedule,
                run_passed,
                fields,
                perf_fields,
                lane_errors,
                strict_acceptance,
            )

    ratios_passed = True
    try:
        for repeat, (ordered, fast) in enumerate(
            zip(lane_results["ordered"], lane_results["fast"]), start=1
        ):
            ordered_momentum = float(
                ordered["sphereSoftSoftGlancing.deltaAngularMomentum"]
            )
            fast_momentum = float(
                fast["sphereSoftSoftGlancing.deltaAngularMomentum"]
            )
            ordered_speed = float(
                ordered["sphereSoftSoftGlancing.deltaAngularSpeed"]
            )
            fast_speed = float(
                fast["sphereSoftSoftGlancing.deltaAngularSpeed"]
            )
            ordered_direction = tuple(
                float(
                    ordered[
                        f"sphereSoftSoftGlancing.deltaAngularVelocity{axis}"
                    ]
                )
                for axis in ("X", "Y", "Z")
            )
            fast_direction = tuple(
                float(
                    fast[
                        f"sphereSoftSoftGlancing.deltaAngularVelocity{axis}"
                    ]
                )
                for axis in ("X", "Y", "Z")
            )
            ordered_direction_norm = math.sqrt(
                sum(component * component for component in ordered_direction)
            )
            fast_direction_norm = math.sqrt(
                sum(component * component for component in fast_direction)
            )
            direction_cosine = (
                sum(
                    ordered_component * fast_component
                    for ordered_component, fast_component in zip(
                        ordered_direction, fast_direction
                    )
                )
                / (ordered_direction_norm * fast_direction_norm)
                if ordered_direction_norm > 0.0 and fast_direction_norm > 0.0
                else math.nan
            )
            momentum_ratio = (
                fast_momentum / ordered_momentum
                if ordered_momentum > 0.0
                else math.nan
            )
            speed_ratio = (
                fast_speed / ordered_speed if ordered_speed > 0.0 else math.nan
            )
            # Compare a physical magnitude and its direction independently:
            # the fast lane may differ numerically from ordered GS, but it may
            # not shed either the contact-generated angular-momentum increment
            # or its angular-velocity increment, nor reverse the latter.
            this_ratio_passed = (
                math.isfinite(momentum_ratio)
                and momentum_ratio
                >= SPHERE_SOFT_SOFT_GLANCING_MIN_FAST_OVER_ORDERED
                and math.isfinite(speed_ratio)
                and speed_ratio >= SPHERE_SOFT_SOFT_GLANCING_MIN_FAST_OVER_ORDERED
                and math.isfinite(direction_cosine)
                and direction_cosine
                >= SPHERE_SOFT_SOFT_GLANCING_MIN_DIRECTION_COSINE
            )
            if strict_acceptance:
                ratios_passed = ratios_passed and this_ratio_passed
            print(
                "[DEFORMABLE_VOLUME_AVBD_SPHERE_SOFT_SOFT_GLANCING_COMPARISON] "
                f"repeat={repeat} "
                f"orderedDeltaAngularMomentum={ordered_momentum:.9g} "
                f"fastDeltaAngularMomentum={fast_momentum:.9g} "
                f"fastOverOrderedAngularMomentum={momentum_ratio:.9g} "
                "minimumFastOverOrderedAngularMomentum="
                f"{SPHERE_SOFT_SOFT_GLANCING_MIN_FAST_OVER_ORDERED:.9g} "
                f"orderedDeltaAngularSpeed={ordered_speed:.9g} "
                f"fastDeltaAngularSpeed={fast_speed:.9g} "
                f"fastOverOrderedAngularSpeed={speed_ratio:.9g} "
                "minimumFastOverOrderedAngularSpeed="
                f"{SPHERE_SOFT_SOFT_GLANCING_MIN_FAST_OVER_ORDERED:.9g} "
                f"angularVelocityDirectionCosine={direction_cosine:.9g} "
                "minimumDirectionCosine="
                f"{SPHERE_SOFT_SOFT_GLANCING_MIN_DIRECTION_COSINE:.9g} "
                f"status={'PASS' if this_ratio_passed else 'FAIL'} "
                f"acceptance={'ENFORCED' if strict_acceptance else 'PROBE'}"
            )
    except (KeyError, ValueError):
        ratios_passed = False
        print(
            "[DEFORMABLE_VOLUME_AVBD_SPHERE_SOFT_SOFT_GLANCING_COMPARISON] "
            "status=FAIL error=missing-or-invalid-telemetry"
        )
    lane_passed = lane_passed and (ratios_passed or not strict_acceptance)
    print(
        "[DEFORMABLE_VOLUME_AVBD_SPHERE_SOFT_SOFT_GLANCING_SUMMARY] "
        f"runs={2 * repeats} status={'PASS' if lane_passed else 'FAIL'} "
        f"acceptance={'ENFORCED' if strict_acceptance else 'PROBE'}"
    )
    return 0 if lane_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "probe",
            "acceptance",
            "correctness",
            "rotation-quality-probe",
            "rotation-quality-acceptance",
            "sphere-soft-soft-glancing-probe",
            "sphere-soft-soft-glancing-acceptance",
            "soft-soft-torque-probe",
            "soft-soft-torque-acceptance",
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
        "--dispatcher-threads",
        type=int,
        default=2,
        help="Headless PxDefaultCpuDispatcher worker count (1..256).",
    )
    parser.add_argument(
        "--require-perf-schema",
        choices=("auto", "1", "2"),
        default="auto",
        help=(
            "Require a specific [AVBD_PERF] schema. Performance modes "
            "default to schema=2 when this remains auto."
        ),
    )
    parser.add_argument(
        "--performance-json",
        type=Path,
        help="Write machine-readable performance data (performance modes only).",
    )
    parser.add_argument(
        "--collision-telemetry",
        action="store_true",
        help=(
            "Enable Scene OGC work counters; use telemetry runs as work "
            "evidence, not timing denominators."
        ),
    )
    parser.add_argument(
        "--surface-triangle-bvh",
        choices=("on", "off"),
        default="on",
        help=(
            "Use the refittable surface-triangle hierarchy (on) or the "
            "internal exact full-traversal reference (off)."
        ),
    )
    parser.add_argument(
        "--rigid-triangle-bvh",
        choices=("on", "off"),
        default="on",
        help=(
            "Use the immutable rigid-triangle hierarchy (on) or the "
            "same-binary exact full-traversal reference (off)."
        ),
    )
    parser.add_argument(
        "--rigid-triangle-grid-dim",
        type=int,
        default=1,
        help=(
            "Headless rigid-triangle corpus grid dimension (1 keeps the "
            "legacy mesh; valid range is 1..128)."
        ),
    )
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
    rotation_quality_mode = args.mode in (
        "rotation-quality-probe",
        "rotation-quality-acceptance",
    )
    soft_soft_torque_mode = args.mode in (
        "soft-soft-torque-probe",
        "soft-soft-torque-acceptance",
    )
    sphere_soft_soft_glancing_mode = args.mode in (
        "sphere-soft-soft-glancing-probe",
        "sphere-soft-soft-glancing-acceptance",
    )
    correctness_mode = args.mode == "correctness"
    if (
        correctness_mode
        and args.case is not None
        and args.case not in CORRECTNESS_CASES
    ):
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            f"{args.case} is not registered in the correctness suite"
        )
        return 2
    if rotation_quality_mode:
        if args.case not in (None, SPHERE_LONG_ROLL_CASE):
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                f"{args.mode} only measures {SPHERE_LONG_ROLL_CASE}"
            )
            return 2
    if soft_soft_torque_mode:
        if args.case not in (None, SOFT_SOFT_TORQUE_CASE):
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                f"{args.mode} only measures {SOFT_SOFT_TORQUE_CASE}"
            )
            return 2
    if sphere_soft_soft_glancing_mode:
        if args.case not in (None, SPHERE_SOFT_SOFT_GLANCING_CASE):
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                f"{args.mode} only measures {SPHERE_SOFT_SOFT_GLANCING_CASE}"
            )
            return 2
    if args.mode == "rotation-quality-acceptance":
        if args.frames < ROTATION_QUALITY_ACCEPTANCE_MIN_FRAMES:
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                "rotation-quality-acceptance requires at least "
                f"{ROTATION_QUALITY_ACCEPTANCE_MIN_FRAMES} frames"
            )
            return 2
        if args.dispatcher_threads < 2:
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                "rotation-quality-acceptance requires at least 2 "
                "dispatcher threads"
            )
            return 2
    if args.mode == "soft-soft-torque-acceptance":
        if args.frames < SOFT_SOFT_TORQUE_ACCEPTANCE_MIN_FRAMES:
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                "soft-soft-torque-acceptance requires at least "
                f"{SOFT_SOFT_TORQUE_ACCEPTANCE_MIN_FRAMES} frames"
            )
            return 2
    if args.mode == "sphere-soft-soft-glancing-acceptance":
        if args.frames < SPHERE_SOFT_SOFT_GLANCING_MIN_FRAMES:
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                "sphere-soft-soft-glancing-acceptance requires at least "
                f"{SPHERE_SOFT_SOFT_GLANCING_MIN_FRAMES} frames"
            )
            return 2
        if args.dispatcher_threads < 2:
            print(
                "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
                "sphere-soft-soft-glancing-acceptance requires at least 2 "
                "dispatcher threads"
            )
            return 2
    if not 1 <= args.dispatcher_threads <= 256:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "dispatcher-threads must be in [1, 256]"
        )
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
    if args.performance_json is not None and not performance_mode:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "--performance-json requires a performance-* mode"
        )
        return 2
    required_perf_schema = (
        "2"
        if performance_mode and args.require_perf_schema == "auto"
        else args.require_perf_schema
    )
    warmup = args.warmup if args.warmup is not None else (
        30 if performance_mode else 0
    )
    if warmup < 0 or warmup >= args.frames:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "warmup must be non-negative and less than frames"
        )
        return 2
    if not 1 <= args.rigid_triangle_grid_dim <= 128:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "rigid-triangle-grid-dim must be in [1, 128]"
        )
        return 2
    repeats = args.repeats if args.repeats is not None else (
        3
        if performance_mode
        else (
            2
            if args.mode in (
                "acceptance",
                "rotation-quality-acceptance",
                "sphere-soft-soft-glancing-acceptance",
                "soft-soft-torque-acceptance",
            )
            else 1
        )
    )
    if repeats <= 0:
        print("[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] repeats must be positive")
        return 2
    if args.mode == "rotation-quality-acceptance" and repeats < 2:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "rotation-quality-acceptance requires at least 2 repeats"
        )
        return 2
    if args.mode == "soft-soft-torque-acceptance" and repeats < 2:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "soft-soft-torque-acceptance requires at least 2 repeats"
        )
        return 2
    if args.mode == "sphere-soft-soft-glancing-acceptance" and repeats < 2:
        print(
            "[DEFORMABLE_VOLUME_AVBD_RUNNER_ERROR] "
            "sphere-soft-soft-glancing-acceptance requires at least 2 repeats"
        )
        return 2
    args.repeats_effective = repeats
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
    if rotation_quality_mode:
        return run_rotation_quality_probe(
            args,
            bin_dir,
            warmup,
            required_perf_schema,
            repeats,
            strict_acceptance=args.mode == "rotation-quality-acceptance",
        )
    if soft_soft_torque_mode:
        return run_soft_soft_torque_probe(
            args,
            bin_dir,
            warmup,
            required_perf_schema,
            repeats,
            strict_acceptance=args.mode == "soft-soft-torque-acceptance",
        )
    if sphere_soft_soft_glancing_mode:
        return run_sphere_soft_soft_glancing_probe(
            args,
            bin_dir,
            warmup,
            required_perf_schema,
            repeats,
            strict_acceptance=(
                args.mode == "sphere-soft-soft-glancing-acceptance"
            ),
        )
    selected_cases = (args.case,) if args.case else (
        ("current-all",) if performance_mode else
        CORRECTNESS_CASES if correctness_mode else CASES
    )
    passed = True
    results: dict[tuple[str, int], dict[str, str]] = {}
    perf_results: dict[tuple[str, int], dict[str, str]] = {}
    for repeat in range(1, repeats + 1):
        for case_name in selected_cases:
            case_frames = (
                CORRECTNESS_CASE_FRAMES[case_name]
                if correctness_mode
                else args.frames
            )
            run_passed, fields, perf_fields = run_one(
                case_name,
                repeat,
                bin_dir,
                case_frames,
                args.timeout,
                args.execution,
                warmup,
                args.dispatcher_threads,
                required_perf_schema,
                args.collision_telemetry or correctness_mode,
                args.surface_triangle_bvh,
                args.rigid_triangle_bvh,
                args.rigid_triangle_grid_dim,
                performance_mode,
            )
            passed = passed and run_passed
            results[(case_name, repeat)] = fields
            perf_results[(case_name, repeat)] = perf_fields
    # Exact repeat equality is retained for the explicit sequential reference
    # lane.  Parallel fast execution is accepted per run through the physical
    # stability, penetration, finite-value and ownership gates, because task
    # partitioning is permitted to change the floating-point trajectory.
    if (
        args.mode == "acceptance"
        and args.execution == "sequential"
        and repeats >= 2
    ):
        for case_name in selected_cases:
            passed = (
                compare_repeats(
                    case_name,
                    results[(case_name, 1)],
                    results[(case_name, 2)],
                )
                and passed
            )
    elif args.mode == "acceptance" and repeats >= 2:
        print(
            "[DEFORMABLE_VOLUME_AVBD_REPEAT] "
            "mode=parallel status=SKIPPED "
            "reason=relaxed-fast-path-per-run-physics-gates"
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
        if args.performance_json is not None:
            write_performance_json(
                args.performance_json,
                args,
                selected_cases,
                warmup,
                required_perf_schema,
                perf_results,
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
