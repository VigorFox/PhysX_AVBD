#!/usr/bin/env python3
"""Lock E39/E44/E49 triangle-surface discrete and swept OGC semantics."""

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
        "struct AvbdRigidTriangleSurfaceVertex",
        "// =============================================================================\n"
        "// OGC (Offset Geometric Contact)",
    )
    require_all(
        errors,
        "triangle-surface descriptor",
        descriptor,
        (
            "struct AvbdRigidTriangleSurfaceEdge",
            "struct AvbdRigidTriangleSurfaceTriangle",
            "struct AvbdRigidTriangleSurface",
            "PxBounds3 localBounds;",
            "PxReal localRadius;",
            "PxU64 primitiveKey;",
            "AvbdSoftContactTargetKind targetKind;",
            "PxU32 targetIndex;",
            "PxTransform shapeToRigidBody;",
            "PxArray<AvbdRigidTriangleSurfaceVertex> vertices;",
            "PxArray<AvbdRigidTriangleSurfaceEdge> edges;",
            "PxArray<AvbdRigidTriangleSurfaceTriangle> triangles;",
            "one-sided simulation semantics",
        ),
    )

    detector = section(
        soft,
        "PX_FORCE_INLINE bool avbdIsRigidTriangleSurfaceValid(",
        "inline void avbdDetectSelfCollisionOGC(",
    )
    require_all(
        errors,
        "triangle-surface discrete OGC",
        detector,
        (
            "PX_FORCE_INLINE void "
            "avbdConfigureRigidTriangleSurfaceTarget(",
            "struct AvbdRigidTriangleSurfacePointQuery",
            "avbdQueryRigidTriangleSurfaceLocal(",
            "AvbdVelocityObjectiveOwner::ComponentFinalize",
            "AvbdVelocityObjectiveOwner::PositionAL",
            "inline void avbdDetectSoftRigidTriangleSurface(",
            "avbdIsSoftBodySurfaceVertex(",
            "signedPlaneDistance < 0.0f",
            "avbdClosestPointOnTriangleOGC(",
            "featureProjectionTolerance",
            "!edge.active &&",
            "!vertex.active &&",
            "signedPlaneDistance +",
            "inline void "
            "avbdDetectSoftRigidTriangleSurfaceOGCFeatures(",
            "body.compiled.surfaceEdges",
            "surface.edges",
            "avbdClosestPointsOnSegments(",
            "body.compiled.surfaceTriangles",
            "surface.vertices",
            "geometry.queryParticleIndices[0]",
            "geometry.queryWeights[0]",
            "avbdConfigureRigidTriangleSurfaceTarget(",
            "avbdAppendPreparedSoftContact(",
            "avbdGetRigidTriangleSurfaceSweepPose(",
            "bool& rotationsEquivalent)",
            "AvbdSweptTriangleSurfacePointEntry",
            "avbdSegmentEnterExpandedSegmentInterior(",
            "avbdSegmentEnterExpandedTriangleSurface(",
            "avbdSegmentEnterExpandedRotatingTriangleSurface(",
            "surface.localRadius * angularDistance",
            "for(PxU32 iteration = 0; iteration < 64; ++iteration)",
            "PxSlerp(time, normalizedStart, normalizedEnd)",
            "avbdQueryRigidTriangleSurfaceLocal(",
            "inline void avbdDetectSoftRigidTriangleSurfaceSwept(",
            "compiled.speculativeCCDEnabled",
            "inline void "
            "avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(",
            "avbdTriangleSurfaceForwardVertexOwnsSweep(",
            "avbdTranslatedSegmentEnterExpandedSegmentInteriors(",
            "avbdRotatingSegmentEnterExpandedSegmentInteriors(",
            "avbdSegmentEnterExpandedTriangleNonVertex(",
            "avbdRotatingPointEnterExpandedTriangleFace(",
            "translationToleranceSq",
            "0x54534553u",
            "0x54535653u",
        ),
    )

    aggregate = section(
        soft,
        "inline void avbdDetectAllOGCContacts(",
        "// Build all per-body self-collision adjacencies",
    )
    require_all(
        errors,
        "aggregate triangle-surface route",
        aggregate,
        (
            "const AvbdRigidTriangleSurface* "
            "rigidTriangleSurfaces = NULL",
            "PxU32 numRigidTriangleSurfaces = 0",
            "stats->rigidParticleTriangleSurfaceTests +=",
            "avbdDetectSoftRigidTriangleSurface(",
            "avbdDetectSoftRigidTriangleSurfaceSwept(",
            "avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(",
            "avbdDetectSoftRigidTriangleSurfaceOGCFeatures(",
        ),
    )

    topology = section(
        scene,
        "static PxBounds3 computeTriangleSurfaceBounds(",
        "static bool compileConvexTopology(",
    )
    require_all(
        errors,
        "Scene triangle-surface topology",
        topology,
        (
            "static void getRigidMaterialValues(",
            "static bool appendTriangleSurfaceTriangle(",
            "static bool finalizeTriangleSurfaceTopology(",
            "static bool compileTriangleMeshTopology(",
            "static bool compileHeightFieldTopology(",
            "PxMeshQuery::getTriangle(",
            "mesh->getTriangleMaterialIndex(",
            "PxHeightFieldMaterial::eHOLE",
            "PxHeightFieldFlag::eNO_BOUNDARY_EDGES",
            "oppositePlaneDistance < 0.0f",
            "normalDot < 0.999999f",
        ),
    )
    if "eDOUBLE_SIDED" in topology or "getMeshFlags(" in topology:
        errors.append(
            "Scene topology compilation must preserve native one-sided "
            "simulation winding instead of treating query double-sided as "
            "a simulation-contact flag"
        )

    require_all(
        errors,
        "Scene triangle-surface ownership",
        scene,
        (
            "PxGeometryType::eTRIANGLEMESH",
            "PxGeometryType::eHEIGHTFIELD",
            "bool compileDynamicTriangleSurface(",
            "if(!bodySim || !bodySim->isKinematic() ||",
            "surface.targetKind =",
            "eKINEMATIC_RIGID;",
            "mRigidTriangleSurfaces.pushBack(",
            "mRigidTriangleSurfaces.begin()",
            "mRigidTriangleSurfaces.size()",
            "previousSurface.center =",
            "previousCenter;",
            "previousSurface.rotation =",
            "previousRotation;",
            "computeTriangleSurfaceBounds(",
            "avbdAreSweepRotationsEquivalent(",
            "shape-center localRadius sphere.",
            "const PxVec3 rotationExtent(",
            "previousCenter -",
            "triangleSurface.center +",
        ),
    )

    require_all(
        errors,
        "Surface triangle-surface public gates",
        surface + surface_runner,
        (
            "surface-triangle-mesh-reverse-feature",
            "surface-heightfield-reverse-feature",
            "surface-kinematic-triangle-mesh",
            "surface-kinematic-heightfield",
            "createAvbdRigidTriangleMesh(",
            "createAvbdRigidHeightField(",
            "PxTriangleMeshGeometry(",
            "PxHeightFieldGeometry(",
            "[AVBD_TRIANGLE_MESH_REVERSE_FEATURE]",
            "[AVBD_HEIGHTFIELD_REVERSE_FEATURE]",
            "surface-static-triangle-mesh-speculative-ccd",
            "surface-kinematic-triangle-mesh-speculative-ccd",
            "surface-static-heightfield-speculative-ccd",
            "surface-kinematic-heightfield-speculative-ccd",
            "surface-static-triangle-mesh-reverse-swept-ccd",
            "surface-kinematic-triangle-mesh-reverse-swept-ccd",
            "surface-static-heightfield-reverse-swept-ccd",
            "surface-kinematic-heightfield-reverse-swept-ccd",
            "surface-rotating-kinematic-triangle-mesh-speculative-ccd",
            "surface-rotating-kinematic-heightfield-speculative-ccd",
            "surface-rotating-kinematic-triangle-mesh-reverse-swept-ccd",
            "surface-rotating-kinematic-heightfield-reverse-swept-ccd",
            "createAvbdRotationalRigidTriangleMesh(",
            "runTriangleSurfaceSweptCcdCase(",
            "[AVBD_TRIANGLE_SURFACE_FORWARD_SWEPT]",
            "[AVBD_TRIANGLE_SURFACE_REVERSE_SWEPT]",
            "[AVBD_TRIANGLE_SURFACE_ROTATIONAL_SWEPT]",
            "positiveAngularTravel",
            "minimumVertexSweepSeparation",
            "minimumVertexSeparation",
        ),
    )
    require_all(
        errors,
        "Volume triangle-surface public gates",
        volume + volume_runner,
        (
            "scene-volume-triangle-mesh-reverse-feature",
            "scene-volume-heightfield-reverse-feature",
            "scene-volume-kinematic-triangle-mesh",
            "scene-volume-kinematic-heightfield",
            "createSceneCpuRigidTriangleMesh(",
            "createSceneCpuRigidHeightField(",
            "addSceneStaticTriangleMesh(",
            "addSceneStaticHeightField(",
            "addSceneKinematicTriangleMesh(",
            "addSceneKinematicHeightField(",
            "[AVBD_TRIANGLE_MESH_REVERSE_FEATURE]",
            "[AVBD_HEIGHTFIELD_REVERSE_FEATURE]",
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
            "createSceneCpuRotationalTriangleMesh(",
            "addSceneStaticTriangleSurfacePair(",
            "addSceneMovingKinematicTriangleSurfacePair(",
            "addSceneRotatingKinematicTriangleSurfacePair(",
            "getSceneCpuVolumeRotationalTriangleSurfaceSweepSeparations(",
            "[AVBD_TRIANGLE_SURFACE_FORWARD_SWEPT]",
            "[AVBD_TRIANGLE_SURFACE_REVERSE_SWEPT]",
            "[AVBD_TRIANGLE_SURFACE_ROTATIONAL_SWEPT]",
            "minimumVertexSweepSeparation",
            "positiveAngularTravel",
            "SCENE_KINEMATIC_COUPLING_GATED",
        ),
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume E39 gates bypass the public Scene API"
        )

    if errors:
        print("AVBD_CPU_DEFORMABLE_TRIANGLE_SURFACE_SOURCE_GATE=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_TRIANGLE_SURFACE_SOURCE_GATE=PASS "
        "geometry=triangle-mesh+heightfield "
        "discrete=vertex-face+reverse-edge-face "
        "continuous=forward+reverse-translation+rotation "
        "target=static+kinematic actors=surface+volume "
        "seamOwnership=projection-preserved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
