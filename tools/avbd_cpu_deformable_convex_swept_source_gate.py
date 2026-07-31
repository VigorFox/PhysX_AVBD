#!/usr/bin/env python3
"""Lock E43/E48 CPU AVBD convex continuous OGC ownership."""

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
    scene = read("physx/source/simulationcontroller/src/ScScene.cpp")
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
        "struct AvbdRigidConvex",
        "// =============================================================================\n"
        "// OGC (Offset Geometric Contact)",
    )
    require_all(
        errors,
        "convex continuous-pose descriptor",
        descriptor,
        (
            "PxVec3 previousCenter;",
            "PxQuat previousRotation;",
            "PxVec3 predictedCenter;",
            "PxQuat predictedRotation;",
            "bool predictedPoseValid;",
            "PxTransform shapeToRigidBody;",
            "PxReal localRadius;",
        ),
    )

    point_entry = section(
        soft,
        "struct AvbdSweptConvexPointEntry",
        "inline void avbdDetectSoftRigidConvexSweptSDF(",
    )
    require_all(
        errors,
        "convex point conservative advancement",
        point_entry,
        (
            "avbdSegmentEnterExpandedConvex(",
            "avbdSegmentEnterExpandedRotatingConvex(",
            "avbdQueryRigidConvexLocal(",
            "currentQuery.signedDistance < margin",
            "for(PxU32 iteration = 0; iteration < 48; ++iteration)",
            "for(PxU32 iteration = 0; iteration < 64; ++iteration)",
            "const PxReal nextTime = time + gap / speed;",
            "convex.localRadius * angularDistance",
            "PxSlerp(",
            "result.normalLocal = query.normalLocal;",
            "result.surfaceLocal = query.surfaceLocal;",
            "result.featureKey = query.featureKey;",
        ),
    )

    swept_sdf = section(
        soft,
        "inline void avbdDetectSoftRigidConvexSweptSDF(",
        "struct AvbdSweptConvexEdgeEntry",
    )
    require_all(
        errors,
        "forward convex swept SDF",
        swept_sdf,
        (
            "sourceBody->compiled.speculativeCCDEnabled",
            "avbdIsSoftBodySurfaceVertex(",
            "avbdGetRigidConvexSweepPose(",
            "particle.position - centerStart",
            "particle.predictedPosition - centerEnd",
            "avbdSegmentEnterExpandedConvex(",
            "rotationsEquivalent",
            "avbdSegmentEnterExpandedRotatingConvex(",
            "avbdConfigureRigidConvexTarget(",
            "avbdAppendPreparedSoftContact(",
        ),
    )

    edge_entry = section(
        soft,
        "struct AvbdSweptConvexEdgeEntry",
        "inline void avbdDetectSoftRigidConvexSweptOGCFeatures(",
    )
    require_all(
        errors,
        "convex edge conservative advancement",
        edge_entry,
        (
            "avbdTranslatedSegmentEnterExpandedSegmentInteriors(",
            "avbdRotatingSegmentEnterExpandedSegmentInteriors(",
            "avbdRotatingPointEnterExpandedTriangleFace(",
            "avbdClosestPointsOnSegments(",
            "for(PxU32 iteration = 0; iteration < 48; ++iteration)",
            "for(PxU32 iteration = 0; iteration < 64; ++iteration)",
            "edgeRadius * angularDistance",
            "rigidLocalPoint.magnitude() * angularDistance",
            "softWeight1 <= featureEpsilon",
            "rigidWeight1 <= featureEpsilon",
            "const PxReal nextTime = time + gap / speed;",
            "result.softWeight1 = softWeight1;",
            "result.rigidWeight1 = rigidWeight1;",
        ),
    )

    reverse = section(
        soft,
        "inline void avbdDetectSoftRigidConvexSweptOGCFeatures(",
        "inline void avbdDetectSoftRigidConvexOGCFeatures(",
    )
    require_all(
        errors,
        "reverse convex swept OGC",
        reverse,
        (
            "body.compiled.speculativeCCDEnabled",
            "avbdGetRigidConvexSweepPose(",
            "translationToleranceSq",
            "(displacement1 - displacement0)",
            "(displacement2 - displacement0)",
            "bool forwardVertexOwns = false;",
            "if(currentQuery.signedDistance < margin)",
            "avbdSegmentEnterExpandedConvex(",
            "avbdSegmentEnterExpandedRotatingConvex(",
            "if(forwardVertexOwns)",
            "centerEnd - centerStart - displacement0",
            "avbdTranslatedSegmentEnterExpandedSegmentInteriors(",
            "avbdRotatingSegmentEnterExpandedSegmentInteriors(",
            "normal.dot(outward) <= 0.0f",
            "0x43564545u",
            "avbdSegmentEnterExpandedTriangleNonVertex(",
            "avbdRotatingPointEnterExpandedTriangleFace(",
            "entry.feature != AVBD_FEATURE_FACE",
            "0x43565646u",
            "geometry.queryWeights[0] =",
            "geometry.queryWeights[1] =",
            "geometry.queryWeights[2] =",
            "avbdConfigureRigidConvexTarget(",
            "geometry, 1.0e7f, 1.0e6f",
            "avbdAppendPreparedSoftContact(",
        ),
    )

    pose = section(
        soft,
        "PX_FORCE_INLINE bool avbdGetRigidConvexSweepPose(",
        "struct AvbdSweptConvexPointEntry",
    )
    require_all(
        errors,
        "convex target pose ownership",
        pose,
        (
            "AvbdSoftContactTargetKind::eWORLD_STATIC",
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "!convex.previousCenter.isFinite()",
            "!convex.predictedPoseValid",
            "kinematicTarget ? convex.previousCenter : convex.center",
            "dynamicTarget ? convex.predictedCenter : convex.center",
            "avbdAreSweepRotationsEquivalent(",
            "bool& rotationsEquivalent",
            "rotationsEquivalent =",
        ),
    )
    require_all(
        errors,
        "convex rotation comparison",
        soft,
        (
            "PX_FORCE_INLINE bool avbdAreSweepRotationsEquivalent(",
            "PxReal tolerance = 0.0f",
        ),
    )

    aggregate = section(
        soft,
        "inline void avbdDetectAllOGCContacts(",
        "// Build all per-body self-collision adjacencies",
    )
    require_all(
        errors,
        "aggregate convex continuous ordering",
        aggregate,
        (
            "avbdDetectSoftRigidConvexSDF(",
            "avbdDetectSoftRigidConvexSweptSDF(",
            "avbdDetectSoftRigidConvexSweptOGCFeatures(",
            "avbdDetectSoftRigidConvexOGCFeatures(",
        ),
    )

    selection = section(
        scene,
        "bool buildIslandSelectionStorage(",
        "void copyIslandSelectionResults(",
    )
    require_all(
        errors,
        "selected dynamic-convex ordering",
        selection,
        (
            "Dy::avbdDetectSoftRigidConvexSDF(",
            "Dy::avbdDetectSoftRigidConvexSweptSDF(",
            "Dy::avbdDetectSoftRigidConvexSweptOGCFeatures(",
            "Dy::avbdDetectSoftRigidConvexOGCFeatures(",
            "storage.selectedDynamicConvexes.begin()",
            "storage.probeContacts",
        ),
    )

    compile_dynamic = section(
        scene,
        "void compileDynamicConvexesForIsland(",
        "void refreshSelfCollisionEnabled()",
    )
    require_all(
        errors,
        "dynamic convex predicted pose",
        compile_dynamic,
        (
            "Dy::AvbdSolverBody* solverBodies",
            "PxReal dt",
            "const PxVec3& gravity",
            "convex.shapeToRigidBody",
            "solverBody.computePrediction(dt, gravity);",
            "solverBody.predictedPosition",
            "solverBody.predictedRotation",
            "predictedBodyToWorld *",
            "convex.shapeToRigidBody",
            "convex.predictedCenter",
            "convex.predictedRotation",
            "convex.predictedPoseValid = true;",
        ),
    )
    require_all(
        errors,
        "Scene convex swept overlap envelope",
        scene,
        (
            "previousConvex.center =",
            "convex.previousCenter;",
            "previousConvex.rotation =",
            "convex.previousRotation;",
            "computeConvexBounds(",
            "convex.localRadius +",
            "bodyCore.linearVelocity *",
            "predictedBodyCenter -",
            "predictedBodyCenter +",
            "avbdAreSweepRotationsEquivalent(",
            "convex.previousCenter -",
            "convex.previousCenter +",
            "convex.center -",
            "convex.center +",
        ),
    )

    public_common = (
        "AVBD_CONVEX_REVERSE_SWEPT",
        "responseObserved",
        "negativeControlPassed",
        "twoSidedResponseObserved",
        "vertexSweepExcluded",
        "faceSeparation",
        "minimumVertexSweepSeparation",
        "eENABLE_SPECULATIVE_CCD",
        "PxConvexMeshGeometry",
        "PxGeometryQuery::pointDistance",
    )
    require_all(
        errors,
        "Surface convex continuous public gates",
        surface + surface_runner,
        (
            "surface-convex-speculative-ccd",
            "surface-moving-kinematic-convex-speculative-ccd",
            "surface-rotating-kinematic-convex-speculative-ccd",
            "surface-dynamic-convex-relative-swept-ccd",
            "surface-dynamic-rotating-convex-relative-swept-ccd",
            "surface-static-convex-reverse-swept-ccd",
            "surface-kinematic-convex-reverse-swept-ccd",
            "surface-dynamic-convex-reverse-swept-ccd",
            "surface-rotating-kinematic-convex-reverse-swept-ccd",
            "surface-dynamic-rotating-convex-reverse-swept-ccd",
            "AVBD_CONVEX_ROTATIONAL_SWEPT",
            "AVBD_CONVEX_DYNAMIC_ROTATIONAL_SWEPT",
            "AVBD_CONVEX_ROTATIONAL_REVERSE_SWEPT",
            "runFiniteReverseSweptCcdCase(",
        )
        + public_common,
    )
    require_all(
        errors,
        "Volume convex continuous public gates",
        volume + volume_runner,
        (
            "scene-volume-convex-speculative-ccd",
            "scene-volume-moving-kinematic-convex-speculative-ccd",
            "scene-volume-rotating-kinematic-convex-speculative-ccd",
            "scene-volume-dynamic-convex-relative-swept-ccd",
            "scene-volume-dynamic-rotating-convex-relative-swept-ccd",
            "scene-volume-static-convex-reverse-swept-ccd",
            "scene-volume-kinematic-convex-reverse-swept-ccd",
            "scene-volume-dynamic-convex-reverse-swept-ccd",
            "scene-volume-rotating-kinematic-convex-reverse-swept-ccd",
            "scene-volume-dynamic-rotating-convex-reverse-swept-ccd",
            "AVBD_CONVEX_ROTATIONAL_SWEPT",
            "AVBD_CONVEX_DYNAMIC_ROTATIONAL_SWEPT",
            "AVBD_CONVEX_ROTATIONAL_REVERSE_SWEPT",
            "addSceneStaticConvexCluster(",
            "addSceneMovingKinematicConvexPair(",
            "addSceneDynamicConvexSweepPair(",
            "addSceneRotatingKinematicConvexPair(",
            "addSceneDynamicRotatingConvexPair(",
            "getSceneCpuVolumeConvexMinSeparation(",
            "getSceneCpuVolumeRotationalConvexPointSweepSeparations(",
            "getSceneCpuVolumeRotationalConvexReverseSweptSeparations(",
            "isSceneCpuVolumeConvexReverseSweptCcdCase(",
        )
        + public_common,
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume E43 gates bypass the public Scene API"
        )

    if errors:
        print("AVBD_CPU_DEFORMABLE_CONVEX_SWEPT_SOURCE_GATE=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_CONVEX_SWEPT_SOURCE_GATE=PASS "
        "geometry=convex forward=vertex-sdf "
        "reverse=edge-edge+vertex-face "
        "vertexOwner=forward-current-or-swept-sdf "
        "softMotion=translation-only "
        "rigidMotion=translation+rotation "
        "rotation=forward+reverse target=static+kinematic+dynamic "
        "actors=surface+volume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
