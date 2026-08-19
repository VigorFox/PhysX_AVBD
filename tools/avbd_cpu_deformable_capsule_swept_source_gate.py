#!/usr/bin/env python3
"""Lock E40/E45/E46/E47 CPU AVBD capsule swept OGC ownership."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def section(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))
    if start_index < 0 or end_index < 0:
        return ""
    return text[start_index:end_index]


def require_all(
    errors: list[str],
    scope: str,
    text: str,
    fragments: tuple[str, ...],
) -> None:
    for fragment in fragments:
        if fragment not in text:
            errors.append(f"{scope} lost {fragment!r}")


def main() -> int:
    errors: list[str] = []
    soft = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
    )
    scene = read(
        "physx/source/simulationcontroller/src/ScScene.cpp"
    )
    surface = read(
        "physx/snippets/snippetdeformablesurfaceavbd/"
        "SnippetDeformableSurfaceAVBD.cpp"
    )
    volume = read(
        "physx/snippets/snippetdeformablevolumeavbd/"
        "SnippetDeformableVolumeAVBD.cpp"
    )
    surface_runner = read(
        "tools/run_snippet_deformable_surface_avbd_headless.py"
    )
    volume_runner = read(
        "tools/run_snippet_deformable_volume_avbd_headless.py"
    )

    descriptor = section(
        soft,
        "struct AvbdRigidCapsule",
        "// =============================================================================",
    )
    require_all(
        errors,
        "capsule current/previous/predicted pose descriptor",
        descriptor,
        (
            "PxVec3 previousCenter;",
            "PxQuat previousRotation;",
            "PxVec3 predictedCenter;",
            "PxQuat predictedRotation;",
            "bool predictedPoseValid;",
            "PxTransform shapeToRigidBody;",
        ),
    )

    sweep_entry = section(
        soft,
        "PX_FORCE_INLINE bool avbdAreSweepRotationsEquivalent(",
        "inline void avbdDetectSoftRigidCapsuleSweptSDF(",
    )
    require_all(
        errors,
        "exact-translation and conservative-rotation capsule entry",
        sweep_entry,
        (
            "PxAbs(startRotation.dot(endRotation))",
            "alignment >= 1.0f - tolerance",
            "PX_FORCE_INLINE bool avbdGetSweepAngularDistance(",
            "angularDistance = 2.0f * PxAcos(alignment);",
            "PX_FORCE_INLINE bool avbdGetRigidCapsuleSweepPose(",
            "AvbdSoftContactTargetKind::eWORLD_STATIC",
            "struct AvbdSweptRotatingCapsulePointEntry",
            "PX_FORCE_INLINE bool "
            "avbdSegmentEnterExpandedRotatingCapsule(",
            "shortest-path quaternion slerp",
            "(pointEnd - pointStart) - (centerEnd - centerStart)",
            "halfHeight * angularDistance",
            "PxSlerp(time, normalizedStart, normalizedEnd)",
            "for(PxU32 iteration = 0; iteration < 64; ++iteration)",
            "if(iteration == 0 && distance < expandedRadius)",
            "const PxReal nextTime = time + gap / speed;",
            "result.surfaceLocal =",
            "PX_FORCE_INLINE bool avbdSegmentEnterExpandedCapsule(",
            "direction.y * direction.y",
            "direction.z * direction.z",
            "candidate.x >= -halfHeight",
            "candidate.x <= halfHeight",
            "avbdSegmentEnterExpandedSphere(",
            "candidateTime < bestTime",
            "entryNormalLocal = bestNormal.getNormalized();",
        ),
    )

    swept = section(
        soft,
        "inline void avbdDetectSoftRigidCapsuleSweptSDFRange(",
        "inline void avbdDetectSoftRigidCapsuleSweptSDF(",
    )
    require_all(
        errors,
        "capsule relative swept detector",
        swept,
        (
            "!sourceBody->compiled.speculativeCCDEnabled",
            "avbdIsSoftBodySurfaceVertex(",
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "avbdGetRigidCapsuleSweepPose(",
            "const bool dynamicTarget =",
            "!kinematicTarget && !dynamicTarget",
            "particle.position - centerStart",
            "particle.predictedPosition - centerEnd",
            "currentSdf < margin",
            "avbdSegmentEnterExpandedCapsule(",
            "avbdSegmentEnterExpandedRotatingCapsule(",
            "capsule.radius + margin",
            "avbdConfigureRigidCapsuleTarget(",
            "avbdAppendPreparedSoftContact(",
        ),
    )
    if "particle.predictedPosition - particle.position" in swept:
        errors.append(
            "capsule sweep regressed to soft-only displacement"
        )
    if "rotationEnd.rotate(entryNormalLocal)" not in swept:
        errors.append(
            "capsule sweep lost shape-frame normal reconstruction"
        )

    reverse_swept = section(
        soft,
        "inline void avbdDetectSoftRigidCapsuleSweptOGCFeatures(",
        "inline void avbdDetectSoftRigidCapsuleOGCFeatures(",
    )
    reverse_forward_owner = section(
        soft,
        "PX_NOINLINE inline bool "
        "avbdRigidCapsuleForwardVertexOwnsSweptFeature(",
        "inline void avbdDetectSoftRigidCapsuleSweptOGCFeatures(",
    )
    require_all(
        errors,
        "capsule reverse forward-owner predicate",
        reverse_forward_owner,
        (
            "avbdSegmentEnterExpandedCapsule(",
            "avbdSegmentEnterExpandedRotatingCapsule(",
            "currentSdf < margin",
        ),
    )
    require_all(
        errors,
        "capsule reverse rotation uses conservative feature entry",
        reverse_swept,
        (
            "const bool rotationsEquivalent =",
            "(displacement1 - displacement0)",
            "(displacement2 - displacement0)",
            "avbdRigidCapsuleForwardVertexOwnsSweptFeature(",
            "const PxVec3 relativeCenterEnd =",
            "capsule.halfHeight + expandedRadius",
            "avbdRotatingSegmentEnterExpandedTriangleNonVertex(",
            "const PxQuat entryRotation =",
            "entry.entryTime",
        ),
    )
    if "if(!avbdAreSweepRotationsEquivalent(" in reverse_swept:
        errors.append(
            "capsule reverse rotation regressed to a global fail-closed gate"
        )

    aggregate = section(
        soft,
        "inline void avbdDetectAllOGCContacts(",
        "inline void avbdBuildAllSelfCollisionAdjacencies(",
    )
    require_all(
        errors,
        "aggregate capsule discrete+swept+reverse ordering",
        aggregate,
        (
            "avbdDetectSoftRigidCapsuleSDF(",
            "avbdDetectSoftRigidCapsuleSweptSDF(",
            "avbdDetectSoftRigidCapsuleOGCFeatures(",
        ),
    )

    topology = section(
        scene,
        "void prepareIslandGeneration(",
        "virtual bool prepareSoftIslandSelections(",
    )
    require_all(
        errors,
        "public-flag capsule swept topology",
        topology,
        (
            "const bool speculativeCCDEnabled =",
            "compileDynamicCapsule(",
            "previousCapsule.center =",
            "capsule.previousCenter;",
            "previousCapsule.rotation =",
            "capsule.previousRotation;",
            "avbdAreSweepRotationsEquivalent(",
            "const PxVec3 rotationExtent(",
            "capsule.radius +",
            "capsule.halfHeight);",
            "capsule.previousCenter -",
            "capsule.previousCenter +",
            "capsule.center -",
            "capsule.center +",
            "capsule.radius +",
            "capsule.halfHeight +",
            "const PxVec3 predictedBodyCenter =",
            "bodyCore.linearVelocity * dt",
            "rigidBounds.include(",
            "predictedBodyCenter -",
            "ensureNativeIslandEdge(",
        ),
    )

    compile_capsule = section(
        scene,
        "void compileDynamicCapsulesForIsland(",
        "void compileDynamicConvexesForIsland(",
    )
    require_all(
        errors,
        "current-frame dynamic capsule prediction",
        compile_capsule,
        (
            "Dy::AvbdSolverBody* solverBodies",
            "capsule.targetKind =",
            "eRIGID_BODY;",
            "capsule.shapeToRigidBody =",
            "solverBody.computePrediction(dt, gravity);",
            "solverBody.predictedPosition",
            "solverBody.predictedRotation",
            "capsule.predictedCenter =",
            "capsule.predictedRotation =",
            "capsule.predictedPoseValid = true;",
        ),
    )

    selection = section(
        scene,
        "bool buildIslandSelectionStorage(",
        "void copyIslandSelectionResults(",
    )
    require_all(
        errors,
        "selected dynamic capsule discrete+swept route",
        selection,
        (
            "compileDynamicCapsulesForIsland(",
            "Dy::avbdDetectSoftRigidCapsuleSDF(",
            "Dy::avbdDetectSoftRigidCapsuleSweptSDF(",
            "Dy::avbdDetectSoftRigidCapsuleOGCFeatures(",
            "storage.selectedDynamicCapsules.begin()",
            "storage.probeContacts",
        ),
    )

    public_metrics = (
        "speculativeCcdFlagApplied",
        "speculativeCcdPreventedTunneling",
        "speculativeCcdNegativeControlTunneled",
        "movingSphereCcdResponseObserved",
        "movingSphereNegativeControlHeld",
        "dynamicSphereSweepResponseObserved",
        "dynamicSphereSweepNegativeControlTunneled",
        "dynamicSphereSweepTwoSidedResponseObserved",
        "eENABLE_SPECULATIVE_CCD",
    )
    require_all(
        errors,
        "Surface capsule swept public gates",
        surface + surface_runner,
        (
            "surface-capsule-speculative-ccd",
            "surface-moving-kinematic-capsule-speculative-ccd",
            "surface-rotating-kinematic-capsule-speculative-ccd",
            "surface-dynamic-capsule-relative-swept-ccd",
            "surface-dynamic-rotating-capsule-relative-swept-ccd",
            "surface-rotating-kinematic-capsule-reverse-swept-ccd",
            "surface-dynamic-rotating-capsule-reverse-swept-ccd",
            "[AVBD_CAPSULE_ROTATIONAL_SWEPT]",
            "[AVBD_CAPSULE_DYNAMIC_ROTATIONAL_SWEPT]",
            "[AVBD_CAPSULE_ROTATIONAL_REVERSE_SWEPT]",
            "endpointMinSeparation",
            "midSweepMinSeparation",
            "positiveAngularTravel",
            "negativeAngularTravel",
            "rotationalFreeEnd",
            "PxPi * 10.0f / 9.0f",
            "abs(positive_angular - negative_angular) <= 0.05",
            "PxCapsuleGeometry(radius, halfHeight)",
            "getCapsuleSignedSeparation(",
        )
        + public_metrics,
    )
    require_all(
        errors,
        "Volume capsule swept public gates",
        volume + volume_runner,
        (
            "scene-volume-capsule-speculative-ccd",
            "scene-volume-moving-kinematic-capsule-speculative-ccd",
            "scene-volume-rotating-kinematic-capsule-speculative-ccd",
            "scene-volume-dynamic-capsule-relative-swept-ccd",
            "scene-volume-dynamic-rotating-capsule-relative-swept-ccd",
            "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
            "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
            "[AVBD_CAPSULE_ROTATIONAL_SWEPT]",
            "[AVBD_CAPSULE_DYNAMIC_ROTATIONAL_SWEPT]",
            "[AVBD_CAPSULE_ROTATIONAL_REVERSE_SWEPT]",
            "endpointMinSeparation",
            "midSweepMinSeparation",
            "positiveAngularTravel",
            "negativeAngularTravel",
            "capsuleDynamicRotational.",
            "addSceneRotatingKinematicCapsulePair(",
            "addSceneDynamicRotatingCapsulePair(",
            "addSceneStaticCapsuleCluster(",
            "addSceneMovingKinematicFinitePair(",
            "addSceneDynamicFiniteSweepPair(",
            "getSceneCpuVolumeSingleCapsuleMinSeparation(",
        )
        + public_metrics,
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume E40 gates bypass the public Scene API"
        )

    if errors:
        print("AVBD_CPU_DEFORMABLE_CAPSULE_SWEPT_SOURCE_GATE=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_CAPSULE_SWEPT_SOURCE_GATE=PASS "
        "geometry=capsule sweep=translation-exact+kinematic-dynamic-"
        "rotation-conservative rotation=forward+reverse "
        "targets=static+kinematic+dynamic "
        "actors=surface+volume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
