#!/usr/bin/env python3
"""Lock E42/E47 CPU AVBD capsule-to-soft reverse swept OGC ownership."""

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

    entry = section(
        soft,
        "struct AvbdSweptCapsuleTriangleEntry",
        "inline void avbdDetectSoftRigidCapsuleSweptOGCFeatures(",
    )
    require_all(
        errors,
        "translated segment/triangle entry",
        entry,
        (
            "avbdTranslatedSegmentEnterExpandedTriangleNonVertex(",
            "avbdRotatingSegmentEnterExpandedTriangleNonVertex(",
            "avbdClosestSegmentTriangleOGC(",
            "currentClosest.distance < expandedRadius",
            "const PxReal nextTime = time + gap / speed;",
            "for(PxU32 iteration = 0; iteration < 48; ++iteration)",
            "halfHeight * angularDistance",
            "PxSlerp(time, normalizedStart, normalizedEnd)",
            "for(PxU32 iteration = 0; iteration < 64; ++iteration)",
            "closest.feature != AVBD_FEATURE_FACE",
            "closest.feature != AVBD_FEATURE_EDGE",
            "closest.trianglePoint - closest.segmentPoint",
            "soft-vertex/capsule swept SDF is their unique owner",
        ),
    )

    swept = section(
        soft,
        "inline void avbdDetectSoftRigidCapsuleSweptOGCFeatures(",
        "inline void avbdDetectSoftRigidCapsuleOGCFeatures(",
    )
    forward_owner = section(
        soft,
        "PX_NOINLINE inline bool "
        "avbdRigidCapsuleForwardVertexOwnsSweptFeature(",
        "inline void avbdDetectSoftRigidCapsuleSweptOGCFeatures(",
    )
    require_all(
        errors,
        "capsule forward swept owner predicate",
        forward_owner,
        (
            "particle.initialPosition",
            "particle.predictedPosition",
            "avbdSegmentEnterExpandedCapsule(",
            "avbdSegmentEnterExpandedRotatingCapsule(",
            "currentSdf < margin",
        ),
    )
    require_all(
        errors,
        "capsule reverse swept detector",
        swept,
        (
            "body.compiled.speculativeCCDEnabled",
            "AvbdSoftContactTargetKind::eWORLD_STATIC",
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "!capsule.previousCenter.isFinite()",
            "!capsule.predictedPoseValid",
            "avbdAreSweepRotationsEquivalent(",
            "const bool rotationsEquivalent =",
            "translationToleranceSq",
            "(displacement1 - displacement0)",
            "(displacement2 - displacement0)",
            "PxArray<PxU8>* persistentForwardOwnerScratch = NULL",
            "body.compiled.surfaceVertices.size()",
            "avbdRigidCapsuleForwardVertexOwnsSweptFeature(",
            "forwardOwnerScratch[v0] != 0",
            "forwardOwnerScratch[v1] != 0",
            "forwardOwnerScratch[v2] != 0",
            "if(forwardVertexOwns)",
            "centerEnd - centerStart - displacement0",
            "const PxVec3 relativeCenterEnd =",
            "PxVec3 sweptMinimum(0.0f);",
            "PxVec3 sweptMaximum(0.0f);",
            "capsule.halfHeight + expandedRadius",
            "avbdTranslatedSegmentEnterExpandedTriangleNonVertex(",
            "avbdRotatingSegmentEnterExpandedTriangleNonVertex(",
            "avbdSoftTriangleFeatureKey(",
            "0x43505257u",
            "geometry.queryWeights[0] = entry.barycentric.x;",
            "geometry.queryWeights[1] = entry.barycentric.y;",
            "geometry.queryWeights[2] = entry.barycentric.z;",
            "const PxQuat entryRotation =",
            "entry.entryTime",
            "entryRotation.getConjugate().",
            "avbdConfigureRigidCapsuleTarget(",
            "avbdAppendPreparedSoftContact(",
        ),
    )

    aggregate = section(
        soft,
        "inline void avbdDetectAllOGCContacts(",
        "// Build all per-body self-collision adjacencies",
    )
    require_all(
        errors,
        "aggregate capsule ordering",
        aggregate,
        (
            "avbdDetectSoftRigidCapsuleSDF(",
            "avbdDetectSoftRigidCapsuleSweptSDF(",
            "avbdDetectSoftRigidCapsuleSweptOGCFeatures(",
            "avbdDetectSoftRigidCapsuleOGCFeatures(",
        ),
    )

    selection = section(
        scene,
        "bool buildIslandSelectionStorage(",
        "void copyIslandSelectionResults(",
    )
    require_all(
        errors,
        "selected dynamic-capsule ordering",
        selection,
        (
            "Dy::avbdDetectSoftRigidCapsuleSDF(",
            "Dy::avbdDetectSoftRigidCapsuleSweptSDF(",
            "Dy::avbdDetectSoftRigidCapsuleSweptOGCFeatures(",
            "Dy::avbdDetectSoftRigidCapsuleOGCFeatures(",
            "storage.selectedDynamicCapsules.begin()",
            "storage.probeContacts",
        ),
    )

    public_fragments = (
        "AVBD_CAPSULE_REVERSE_SWEPT",
        "responseObserved",
        "negativeControlPassed",
        "twoSidedResponseObserved",
        "vertexSweepExcluded",
        "faceSeparation",
        "minimumVertexSweepSeparation",
        "eENABLE_SPECULATIVE_CCD",
    )
    require_all(
        errors,
        "Surface capsule reverse-swept gates",
        surface + surface_runner,
        (
            "surface-static-capsule-reverse-swept-ccd",
            "surface-kinematic-capsule-reverse-swept-ccd",
            "surface-dynamic-capsule-reverse-swept-ccd",
            "surface-rotating-kinematic-capsule-reverse-swept-ccd",
            "surface-dynamic-rotating-capsule-reverse-swept-ccd",
            "AVBD_CAPSULE_ROTATIONAL_REVERSE_SWEPT",
            "owner=reverse",
            "endpointMinSeparation",
            "midSweepMinSeparation",
            "positiveAngularTravel",
            "negativeAngularTravel",
            "runFiniteReverseSweptCcdCase(",
            "PxCapsuleGeometry(radius, halfHeight)",
        )
        + public_fragments,
    )
    require_all(
        errors,
        "Volume capsule reverse-swept gates",
        volume + volume_runner,
        (
            "scene-volume-static-capsule-reverse-swept-ccd",
            "scene-volume-kinematic-capsule-reverse-swept-ccd",
            "scene-volume-dynamic-capsule-reverse-swept-ccd",
            "scene-volume-rotating-kinematic-capsule-reverse-swept-ccd",
            "scene-volume-dynamic-rotating-capsule-reverse-swept-ccd",
            "AVBD_CAPSULE_ROTATIONAL_REVERSE_SWEPT",
            "owner=reverse",
            "endpointMinSeparation",
            "midSweepMinSeparation",
            "positiveAngularTravel",
            "negativeAngularTravel",
            "getSceneCpuVolumeSphereReverseSweptSeparations(",
            "getSceneCpuVolumeRotationalCapsuleReverseSweptSeparations(",
            "isSceneCpuVolumeCapsuleReverseSweptCcdCase(",
        )
        + public_fragments,
    )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_CAPSULE_REVERSE_SWEPT_SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_CAPSULE_REVERSE_SWEPT_SOURCE_GATE=PASS "
        "geometry=capsule softFeature=face+finite-edge "
        "vertexOwner=forward-swept-sdf softMotion=translation-only "
        "rigidMotion=translation+rotation-conservative "
        "target=static+kinematic+dynamic "
        "actors=surface+volume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
