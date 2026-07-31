#!/usr/bin/env python3
"""Lock E38 CPU AVBD convex discrete OGC and rigid-owner routing."""

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
        "struct AvbdRigidConvexFace",
        "// =============================================================================\n"
        "// OGC (Offset Geometric Contact)",
    )
    require_all(
        errors,
        "convex descriptor",
        descriptor,
        (
            "struct AvbdRigidConvexEdge",
            "struct AvbdRigidConvexTriangle",
            "struct AvbdRigidConvex",
            "PxReal localRadius;",
            "PxU64 primitiveKey;",
            "AvbdSoftContactTargetKind targetKind;",
            "PxU32 targetIndex;",
            "PxTransform shapeToRigidBody;",
            "PxArray<PxVec3> vertices;",
            "PxArray<PxVec3> vertexNormals;",
            "PxArray<AvbdRigidConvexFace> faces;",
            "PxArray<AvbdRigidConvexEdge> edges;",
            "PxArray<AvbdRigidConvexTriangle> triangles;",
            "detector independent of PxConvexMesh",
        ),
    )

    convex = section(
        soft,
        "PX_FORCE_INLINE bool avbdIsRigidConvexValid(",
        "inline void avbdDetectSoftRigidOGCFeatures(",
    )
    require_all(
        errors,
        "convex discrete OGC",
        convex,
        (
            "PX_FORCE_INLINE void avbdConfigureRigidConvexTarget(",
            "AvbdVelocityObjectiveOwner::ComponentFinalize",
            "AvbdVelocityObjectiveOwner::ManifoldFinalize",
            "AvbdVelocityObjectiveOwner::PositionAL",
            "convex.shapeToRigidBody.transform(surfaceLocal);",
            "inline void avbdDetectSoftRigidConvexSDF(",
            "avbdIsSoftBodySurfaceVertex(",
            "maximumPlaneDistance",
            "avbdClosestPointOnTriangleOGC(",
            "inline void avbdDetectSoftRigidConvexOGCFeatures(",
            "body.compiled.surfaceEdges",
            "convex.edges",
            "avbdClosestPointsOnSegments(",
            "body.compiled.surfaceTriangles",
            "convex.vertices",
            "geometry.queryParticleIndices[0]",
            "geometry.queryWeights[0]",
            "avbdConfigureRigidConvexTarget(",
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
        "aggregate convex route",
        aggregate,
        (
            "const AvbdRigidConvex* rigidConvexes = NULL",
            "PxU32 numRigidConvexes = 0",
            "stats->rigidParticleConvexTests +=",
            "avbdDetectSoftRigidConvexSDF(",
            "avbdDetectSoftRigidConvexOGCFeatures(",
        ),
    )

    require_all(
        errors,
        "Scene convex compilation",
        scene,
        (
            "PxGeometryType::eCONVEXMESH",
            "static bool compileConvexTopology(",
            "geometry.scale.isValidForConvexMesh()",
            "bool compileDynamicConvex(",
            "void compileDynamicConvexesForIsland(",
            "mRigidConvexes.pushBack(convex);",
            "eKINEMATIC_RIGID;",
            "eRIGID_BODY;",
            "storage.selectedDynamicConvexes",
            "Dy::avbdDetectSoftRigidConvexSDF(",
            "Dy::avbdDetectSoftRigidConvexOGCFeatures(",
            "storage.rigidConvexes",
            "rigidConvexes, numRigidConvexes",
        ),
    )

    require_all(
        errors,
        "Surface convex public gates",
        surface + surface_runner,
        (
            "surface-convex-reverse-feature",
            "surface-dynamic-convex",
            "surface-kinematic-convex",
            "createAvbdTestConvexMesh(",
            "PxConvexMeshGeometry(convexMesh)",
            "[AVBD_CONVEX_REVERSE_FEATURE]",
            "vertexSdfExcluded",
            "negativeControlPassed",
            "minimumVertexSeparation",
        ),
    )
    require_all(
        errors,
        "Volume convex public gates",
        volume + volume_runner,
        (
            "scene-volume-convex-reverse-feature",
            "scene-volume-dynamic-convex",
            "scene-volume-kinematic-convex",
            "createSceneCpuRigidConvexMesh(",
            "addSceneStaticConvex(",
            "addSceneDynamicConvex(",
            "addSceneKinematicConvex(",
            "[AVBD_CONVEX_REVERSE_FEATURE]",
            "SCENE_CONVEX_REVERSE_FEATURE_GATED",
        ),
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume E38 gates bypass the public Scene API"
        )

    if errors:
        print("AVBD_CPU_DEFORMABLE_CONVEX_SOURCE_GATE=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_CONVEX_SOURCE_GATE=PASS "
        "geometry=convex discrete=vertex-sdf+reverse-edge-face "
        "target=static+kinematic+dynamic actors=surface+volume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
