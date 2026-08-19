#!/usr/bin/env python3
"""Lock E41 CPU AVBD sphere-to-soft reverse swept OGC ownership."""

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

    entry = section(
        soft,
        "struct AvbdSweptTriangleEntry",
        "inline void avbdDetectSoftRigidSphereSweptOGCFeatures(",
    )
    require_all(
        errors,
        "expanded-triangle non-vertex entry",
        entry,
        (
            "AvbdClosestFeature feature;",
            "avbdSegmentEnterExpandedTriangleNonVertex(",
            "currentClosest.distance < expandedRadius",
            "const PxVec3 triangleNormal =",
            "side * expandedRadius - startPlaneDistance",
            "faceClosest.feature == AVBD_FEATURE_FACE",
            "-triangleNormal * side",
            "const PxVec3 edgeStart[3] = {a, a, b};",
            "const PxVec3 edgeEnd[3] = {b, c, c};",
            "startRadial.magnitudeSquared() -",
            "expandedRadius * expandedRadius",
            "const PxReal discriminant =",
            "axial <= endpointEpsilon",
            "axial >= edgeLength - endpointEpsilon",
            "AVBD_FEATURE_EDGE, edgeIndex",
            "result.feature == AVBD_FEATURE_FACE",
            "result.feature == AVBD_FEATURE_EDGE",
            "rounded vertex caps are intentionally",
            "soft-vertex/sphere swept SDF is their unique owner",
        ),
    )

    swept = section(
        soft,
        "inline void avbdDetectSoftRigidSphereSweptOGCFeatures(",
        "inline void avbdDetectSoftRigidSphereOGCFeatures(",
    )
    forward_owner = section(
        soft,
        "PX_FORCE_INLINE bool "
        "avbdRigidSphereForwardVertexOwnsSweptFeature(",
        "inline void avbdDetectSoftRigidSphereSweptOGCFeatures(",
    )
    require_all(
        errors,
        "sphere forward swept owner predicate",
        forward_owner,
        (
            "particle.initialPosition - centerStart",
            "particle.predictedPosition - centerEnd",
            "currentSdf < margin",
            "avbdSegmentEnterExpandedSphere(",
            "sphere.radius + margin",
        ),
    )
    require_all(
        errors,
        "sphere reverse swept detector",
        swept,
        (
            "body.compiled.speculativeCCDEnabled",
            "AvbdSoftContactTargetKind::eWORLD_STATIC",
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "!sphere.previousCenter.isFinite()",
            "!sphere.predictedPoseValid",
            "const PxVec3 centerStart =",
            "? sphere.previousCenter : sphere.center;",
            "const PxVec3 centerEnd =",
            "? sphere.predictedCenter : sphere.center;",
            "body.compiled.surfaceTriangles",
            "const PxVec3 displacement0 =",
            "translationToleranceSq",
            "(displacement1 - displacement0)",
            "(displacement2 - displacement0)",
            "PxArray<PxU8>* persistentForwardOwnerScratch = NULL",
            "body.compiled.surfaceVertices.size()",
            "avbdRigidSphereForwardVertexOwnsSweptFeature(",
            "forwardOwnerScratch[v0] != 0",
            "forwardOwnerScratch[v1] != 0",
            "forwardOwnerScratch[v2] != 0",
            "if(forwardVertexOwns)",
            "const PxVec3 relativeEnd =",
            "centerEnd - displacement0;",
            "avbdSegmentEnterExpandedTriangleNonVertex(",
            "sphere.radius + margin",
            "avbdSoftTriangleFeatureKey(",
            "entry.feature, entry.featureIndex",
            "0x53505357u",
            "geometry.queryWeights[0] = entry.barycentric.x;",
            "geometry.queryWeights[1] = entry.barycentric.y;",
            "geometry.queryWeights[2] = entry.barycentric.z;",
            "geometry.normal = entry.normal;",
            "geometry.depth = 0.0f;",
            "geometry.margin = margin;",
            "avbdConfigureRigidSphereTarget(",
            "avbdAppendPreparedSoftContact(",
        ),
    )
    if "particle.predictedPosition - particle.position" in swept:
        errors.append(
            "reverse sphere sweep regressed to a single-particle owner"
        )

    target = section(
        soft,
        "PX_FORCE_INLINE void avbdConfigureRigidSphereTarget(",
        "inline void avbdDetectSoftRigidSphereSDF(",
    )
    require_all(
        errors,
        "shared sphere target finalization",
        target,
        (
            "AvbdVelocityObjectiveOwner::PositionAL",
            "AvbdVelocityObjectiveOwner::ComponentFinalize",
            "AvbdVelocityObjectiveOwner::ManifoldFinalize",
            "geometry.rigidLocalPoint =",
            "sphere.shapeToRigidBody.transform(surfaceLocal);",
        ),
    )

    aggregate = section(
        soft,
        "inline void avbdDetectAllOGCContacts(",
        "// Build all per-body self-collision adjacencies",
    )
    require_all(
        errors,
        "aggregate sphere contact route",
        aggregate,
        (
            "avbdDetectSoftRigidSphereSDF(",
            "avbdDetectSoftRigidSphereSweptSDF(",
            "avbdDetectSoftRigidSphereSweptOGCFeatures(",
            "avbdDetectSoftRigidSphereOGCFeatures(",
        ),
    )

    selection = section(
        scene,
        "bool buildIslandSelectionStorage(",
        "void copyIslandSelectionResults(",
    )
    require_all(
        errors,
        "selected dynamic-sphere route",
        selection,
        (
            "Dy::avbdDetectSoftRigidSphereSDF(",
            "Dy::avbdDetectSoftRigidSphereSweptSDF(",
            "Dy::avbdDetectSoftRigidSphereSweptOGCFeatures(",
            "Dy::avbdDetectSoftRigidSphereOGCFeatures(",
            "storage.selectedDynamicSpheres.begin()",
            "storage.probeContacts",
        ),
    )

    public_fragments = (
        "AVBD_SPHERE_REVERSE_SWEPT",
        "responseObserved",
        "negativeControlPassed",
        "twoSidedResponseObserved",
        "vertexSweepExcluded",
        "nonFiniteSamples",
        "positiveDisplacement",
        "negativeDisplacement",
        "positiveRigidDrop",
        "negativeRigidDrop",
        "faceSeparation",
        "minimumVertexSweepSeparation",
        "eENABLE_SPECULATIVE_CCD",
    )
    require_all(
        errors,
        "Surface reverse-swept public gates",
        surface + surface_runner,
        (
            "surface-static-sphere-reverse-swept-ccd",
            "surface-kinematic-sphere-reverse-swept-ccd",
            "surface-dynamic-sphere-reverse-swept-ccd",
            "runFiniteReverseSweptCcdCase(",
            "minimumVertexSweepSeparation > 0.10f",
        )
        + public_fragments,
    )
    require_all(
        errors,
        "Volume reverse-swept public gates",
        volume + volume_runner,
        (
            "scene-volume-static-sphere-reverse-swept-ccd",
            "scene-volume-kinematic-sphere-reverse-swept-ccd",
            "scene-volume-dynamic-sphere-reverse-swept-ccd",
            "getSceneCpuVolumeSphereReverseSweptSeparations(",
            "minimumVertexSweepSeparation >",
            "? 0.05f : 0.10f",
            "neither a soft nor a rigid response",
            "positive_rigid_drop + 0.05",
        )
        + public_fragments,
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume E41 gates bypass the public Scene API"
        )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_SPHERE_REVERSE_SWEPT_SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_SPHERE_REVERSE_SWEPT_SOURCE_GATE=PASS "
        "geometry=sphere softFeature=face+finite-edge "
        "vertexOwner=forward-swept-sdf softMotion=translation-only "
        "target=static+kinematic+dynamic actors=surface+volume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
