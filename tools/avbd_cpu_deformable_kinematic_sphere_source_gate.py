#!/usr/bin/env python3
"""Lock E32 CPU AVBD kinematic-sphere prescribed target ownership."""

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
        "// Analytical sphere descriptor.",
        "// =============================================================================",
    )
    require_all(
        errors,
        "moving-sphere descriptor",
        descriptor,
        (
            "AvbdSoftContactTargetKind targetKind;",
            "PxU32 targetIndex;",
            "PxVec3 previousCenter;",
            "PxQuat previousRotation;",
            "PxTransform shapeToRigidBody;",
            "targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC)",
        ),
    )

    target = section(
        soft,
        "PX_FORCE_INLINE void avbdConfigureRigidSphereTarget(",
        "inline void avbdDetectSoftRigidSphereSDF(",
    )
    require_all(
        errors,
        "kinematic-sphere contact target",
        target,
        (
            "geometry.targetKind = sphere.targetKind;",
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdVelocityObjectiveOwner::ComponentFinalize",
            "sphere.previousCenter +",
            "sphere.previousRotation.rotate(surfaceLocal)",
        ),
    )

    discrete = section(
        soft,
        "inline void avbdDetectSoftRigidSphereSDF(",
        "PX_FORCE_INLINE bool avbdSegmentEnterExpandedSphere(",
    )
    require_all(
        errors,
        "kinematic-sphere discrete SDF",
        discrete,
        (
            "avbdIsSoftBodySurfaceVertex(",
            "const PxReal sdf = distance - sphere.radius;",
            "sphere.rotation.getConjugate().rotate(",
            "avbdConfigureRigidSphereTarget(",
        ),
    )

    swept = section(
        soft,
        "inline void avbdDetectSoftRigidSphereSweptSDF(",
        "inline void avbdDetectSoftRigidOGCFeatures(",
    )
    require_all(
        errors,
        "kinematic-sphere relative swept scope",
        swept,
        (
            "AvbdSoftContactTargetKind::eWORLD_STATIC",
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "sphere.previousCenter : sphere.center;",
            "particle.position - sphereCenterStart;",
            "particle.predictedPosition - sphereCenterEnd;",
            "avbdSegmentEnterExpandedSphere(",
            "avbdConfigureRigidSphereTarget(",
        ),
    )

    compile_sphere = section(
        scene,
        "bool compileDynamicSphere(",
        "static const PxsDeformableVolumeMaterialCore* getMaterial(",
    )
    require_all(
        errors,
        "Scene moving-sphere extraction",
        compile_sphere,
        (
            "PxGeometryType::eSPHERE",
            "previousShapeToWorld",
            "entry.core->getKinematicTarget(targetPose)",
            "sphere.center = shapeToWorld.p;",
            "sphere.rotation = shapeToWorld.q;",
            "sphere.previousCenter = previousShapeToWorld.p;",
            "sphere.previousRotation = previousShapeToWorld.q;",
            "sphere.radius = geometry.radius;",
            "sphere.primitiveKey = entry.primitiveKey;",
        ),
    )

    world_compile = section(
        scene,
        "void compileWorldStatics(",
        "void compileDynamicBoxesForIsland(",
    )
    require_all(
        errors,
        "Scene kinematic-sphere prescribed route",
        world_compile,
        (
            "bodySim->isKinematic()",
            "compileDynamicSphere(entry, sphere)",
            "sphere.targetKind =",
            "eKINEMATIC_RIGID;",
            "mRigidSpheres.pushBack(sphere);",
        ),
    )

    island_refresh = section(
        scene,
        "void prepareIslandGeneration(",
        "virtual bool prepareSoftIslandSelections(",
    )
    require_all(
        errors,
        "kinematic-sphere overlap wake route",
        island_refresh,
        (
            "compileDynamicSphere(",
            "rigidBounds = computeSphereBounds(sphere);",
            "candidateBounds.intersects(rigidBounds)",
            "if(softEntry.sleeping && bodySim &&",
            "bodySim->isActive()",
            "if(bodySim && !bodySim->isKinematic())",
        ),
    )

    require_all(
        errors,
        "Surface kinematic-sphere public gate",
        surface + surface_runner,
        (
            "surface-kinematic-sphere",
            "PxSphereGeometry(0.8f)",
            "kinematicSmoothCase ? 1.1f",
            "localPosition.magnitude() <= 0.88f",
            "kinematicSurfaceWoke",
            "kinematicSurfaceMoved",
            "kinematicContactObserved",
            "metrics.maxSpeed < 2.0f",
        ),
    )
    require_all(
        errors,
        "Volume kinematic-sphere public gate",
        volume + volume_runner,
        (
            "scene-volume-kinematic-sphere",
            "addSceneKinematicSphere(",
            "PxVec3(0.0f, 3.4f, 0.0f), 0.5f",
            "localPosition.magnitude() <= 0.58f",
            "sceneKinematicSoftWoke",
            "sceneKinematicSoftMoved",
            "sceneKinematicContactObserved",
            "gMetrics.maxParticleSpeed < 2.0f",
        ),
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume kinematic-sphere gates bypass "
            "the public Scene API"
        )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_KINEMATIC_SPHERE_SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_KINEMATIC_SPHERE_SOURCE_GATE=PASS "
        "geometry=sphere target=kinematic-prescribed "
        "actors=surface+volume swept=relative-prescribed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
