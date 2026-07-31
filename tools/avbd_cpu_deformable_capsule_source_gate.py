#!/usr/bin/env python3
"""Lock E37 CPU AVBD capsule discrete OGC and rigid-owner routing."""

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
        "// =============================================================================\n"
        "// OGC (Offset Geometric Contact)",
    )
    require_all(
        errors,
        "capsule descriptor",
        descriptor,
        (
            "PxReal radius;",
            "PxReal halfHeight;",
            "PxU64 primitiveKey;",
            "AvbdSoftContactTargetKind targetKind;",
            "PxU32 targetIndex;",
            "PxVec3 previousCenter;",
            "PxQuat previousRotation;",
            "PxTransform shapeToRigidBody;",
        ),
    )

    closest = section(
        soft,
        "struct AvbdClosestSegmentTriangleResult",
        "PX_FORCE_INLINE void avbdGetRigidBoxEdgeLocal(",
    )
    require_all(
        errors,
        "capsule segment-triangle feature query",
        closest,
        (
            "avbdClosestSegmentTriangleOGC(",
            "avbdClosestPointOnTriangleOGC(",
            "avbdClosestPointsOnSegments(",
            "planeClosest.feature",
            "AVBD_FEATURE_EDGE",
            "AVBD_FEATURE_VERTEX",
            "barycentric",
            "segmentWeight1",
        ),
    )

    capsule = section(
        soft,
        "PX_FORCE_INLINE void avbdConfigureRigidCapsuleTarget(",
        "inline void avbdDetectSoftRigidOGCFeatures(",
    )
    require_all(
        errors,
        "capsule discrete OGC",
        capsule,
        (
            "AvbdVelocityObjectiveOwner::ComponentFinalize",
            "AvbdVelocityObjectiveOwner::ManifoldFinalize",
            "capsule.shapeToRigidBody.transform(surfaceLocal);",
            "inline void avbdDetectSoftRigidCapsuleSDF(",
            "avbdIsSoftBodySurfaceVertex(",
            "PxClamp(particleLocal.x,",
            "axisLocal + normalLocal * capsule.radius",
            "inline void avbdDetectSoftRigidCapsuleOGCFeatures(",
            "body.compiled.surfaceTriangles",
            "avbdClosestSegmentTriangleOGC(",
            "closest.feature == AVBD_FEATURE_VERTEX",
            "avbdSoftTriangleFeatureKey(",
            "PxArray<PxU64> emittedFeatureKeys;",
            "geometry.queryParticleIndices[0] = v0;",
            "geometry.queryWeights[0] =",
            "closest.barycentric.x;",
            "queryRadius - closest.distance",
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
        "aggregate capsule route",
        aggregate,
        (
            "const AvbdRigidCapsule* rigidCapsules = NULL",
            "PxU32 numRigidCapsules = 0",
            "stats->rigidParticleCapsuleTests +=",
            "avbdDetectSoftRigidCapsuleSDF(",
            "avbdDetectSoftRigidCapsuleOGCFeatures(",
        ),
    )

    require_all(
        errors,
        "Scene capsule compilation",
        scene,
        (
            "PxGeometryType::eCAPSULE",
            "static PxBounds3 computeCapsuleBounds(",
            "bool compileDynamicCapsule(",
            "void compileDynamicCapsulesForIsland(",
            "mRigidCapsules.pushBack(capsule);",
            "eKINEMATIC_RIGID;",
            "eRIGID_BODY;",
            "storage.selectedDynamicCapsules",
            "Dy::avbdDetectSoftRigidCapsuleSDF(",
            "Dy::avbdDetectSoftRigidCapsuleOGCFeatures(",
            "storage.rigidCapsules",
            "rigidCapsules, numRigidCapsules",
        ),
    )

    require_all(
        errors,
        "Surface capsule public gates",
        surface + surface_runner,
        (
            "surface-capsule-reverse-feature",
            "surface-dynamic-capsule",
            "surface-kinematic-capsule",
            "PxCapsuleGeometry(radius, halfHeight)",
            "PxCapsuleGeometry(0.8f, 0.3f)",
            "[AVBD_CAPSULE_REVERSE_FEATURE]",
            "vertexSdfExcluded",
            "negativeControlPassed",
            "minimumVertexSeparation",
        ),
    )
    require_all(
        errors,
        "Volume capsule public gates",
        volume + volume_runner,
        (
            "scene-volume-capsule-reverse-feature",
            "scene-volume-dynamic-capsule",
            "scene-volume-kinematic-capsule",
            "addSceneStaticCapsule(",
            "addSceneDynamicCapsule(",
            "addSceneKinematicCapsule(",
            "getSceneCpuVolumeSmoothReverseSeparations(",
            "[AVBD_CAPSULE_REVERSE_FEATURE]",
            "SCENE_CAPSULE_REVERSE_FEATURE_GATED",
        ),
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume E37 gates bypass the public Scene API"
        )

    if errors:
        print("AVBD_CPU_DEFORMABLE_CAPSULE_SOURCE_GATE=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_CAPSULE_SOURCE_GATE=PASS "
        "geometry=capsule discrete=vertex-sdf+reverse-edge-face "
        "target=static+kinematic+dynamic actors=surface+volume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
