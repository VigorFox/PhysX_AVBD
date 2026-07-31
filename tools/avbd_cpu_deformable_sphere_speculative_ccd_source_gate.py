#!/usr/bin/env python3
"""Lock E31 CPU AVBD static-sphere contacts to their accepted scope."""

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

    sphere_descriptor = section(
        soft,
        "// Analytical sphere descriptor.",
        "// =============================================================================",
    )
    require_all(
        errors,
        "analytical sphere descriptor with world-static default",
        sphere_descriptor,
        (
            "struct AvbdRigidSphere",
            "PxVec3 center;",
            "PxReal radius;",
            "PxReal friction;",
            "PxU8 frictionCombineMode;",
            "PxU64 primitiveKey;",
            "targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC)",
        ),
    )

    discrete = section(
        soft,
        "inline void avbdDetectSoftRigidSphereSDF(",
        "PX_FORCE_INLINE bool avbdSegmentEnterExpandedSphere(",
    )
    target = section(
        soft,
        "PX_FORCE_INLINE void avbdConfigureRigidSphereTarget(",
        "inline void avbdDetectSoftRigidSphereSDF(",
    )
    require_all(
        errors,
        "world-static sphere target ownership",
        target,
        (
            "AvbdVelocityObjectiveOwner::PositionAL",
            "geometry.surfacePoint =",
            ": geometry.surfacePoint;",
        ),
    )
    require_all(
        errors,
        "static-sphere discrete SDF",
        discrete,
        (
            "avbdIsSoftBodySurfaceVertex(",
            "const PxReal sdf = distance - sphere.radius;",
            "AvbdSoftContactSource::eRIGID_SDF",
            "const PxVec3 surfaceLocal =",
            "avbdConfigureRigidSphereTarget(",
            "geometry, 1e5f, 1e6f",
        ),
    )

    segment_query = section(
        soft,
        "PX_FORCE_INLINE bool avbdSegmentEnterExpandedSphere(",
        "inline void avbdDetectSoftRigidSphereSweptSDF(",
    )
    require_all(
        errors,
        "analytical segment-sphere entry",
        segment_query,
        (
            "halfB * halfB - directionMagnitudeSq * c",
            "(-halfB - PxSqrt(discriminant))",
            "if(entryTime < 0.0f || entryTime > 1.0f)",
            "entryOffset * PxRecipSqrt(entryMagnitudeSq)",
        ),
    )

    swept = section(
        soft,
        "inline void avbdDetectSoftRigidSphereSweptSDF(",
        "inline void avbdDetectSoftRigidOGCFeatures(",
    )
    require_all(
        errors,
        "public-flag-gated swept sphere SDF",
        swept,
        (
            "!particle.predictedPosition.isFinite()",
            "if(!sourceBody ||",
            "!sourceBody->compiled.speculativeCCDEnabled",
            "!avbdIsSoftBodySurfaceVertex(",
            "if(!PxIsFinite(currentSdf) || currentSdf < margin)",
            "avbdSegmentEnterExpandedSphere(",
            "sphere.radius + margin",
            "AvbdSoftContactTargetKind::eWORLD_STATIC",
            "avbdConfigureRigidSphereTarget(",
            "geometry, 1e6f, 1e6f",
        ),
    )

    all_contacts = section(
        soft,
        "inline void avbdDetectAllOGCContacts(",
        "// Build all per-body self-collision adjacencies",
    )
    require_all(
        errors,
        "OGC aggregate sphere route",
        all_contacts,
        (
            "const AvbdRigidSphere* rigidSpheres = NULL",
            "PxU32 numRigidSpheres = 0",
            "stats->rigidParticleSphereTests +=",
            "avbdDetectSoftRigidSphereSDF(",
            "avbdDetectSoftRigidSphereSweptSDF(",
            "avbdDetectSoftRigidSphereOGCFeatures(",
        ),
    )

    compile_world = section(
        scene,
        "void compileWorldStatics(",
        "void detectContacts(",
    )
    static_compile = section(
        compile_world,
        "mRigidSpheres.clear();",
        "for(PxU32 i = 0; i < mDynamicShapes.size(); i++)",
    )
    require_all(
        errors,
        "Scene static-sphere extraction",
        static_compile,
        (
            "const StaticShapeEntry& entry",
            "PxGeometryType::eSPHERE",
            "const PxSphereGeometry& geometry",
            "Dy::AvbdRigidSphere sphere;",
            "sphere.center = shapeToWorld.p;",
            "sphere.radius = geometry.radius;",
            "mRigidSpheres.pushBack(sphere);",
        ),
    )
    require_all(
        errors,
        "Scene sphere contact route",
        scene,
        (
            "!mRigidBoxes.empty() || !mRigidSpheres.empty() ||",
            "rigidSpheres = mRigidSpheres.begin();",
            "numRigidSpheres = mRigidSpheres.size();",
            "rigidSpheres, numRigidSpheres,",
        ),
    )

    require_all(
        errors,
        "Surface static-sphere public gate",
        surface + surface_runner,
        (
            "surface-sphere-speculative-ccd",
            "PxSphereGeometry(0.3f)",
            "finiteGeometryCase ? -80.0f : -120.0f",
            "eENABLE_SPECULATIVE_CCD",
            "speculativeCcdPositiveMinSeparation",
            "obstacleCenters[obstacleIndex]",
            "positive_min_separation >= 1.0e30",
            "speculativeCcdNegativeControlTunneled",
        ),
    )
    require_all(
        errors,
        "Volume static-sphere public gate",
        volume + volume_runner,
        (
            "scene-volume-sphere-speculative-ccd",
            "addSceneStaticSphereCluster(",
            "0.3f",
            "sphereSpeculativeCcdCase",
            "? -160.0f : -120.0f",
            "finalizeNegativeControl",
            "? gMetrics.completedFrames == 1",
            "getSceneCpuVolumeSphereMinSeparation(",
            "positive_min_separation >= 1.0e30",
            "speculativeCcdNegativeControlTunneled",
        ),
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume sphere gates bypass the public Scene API"
        )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_SPHERE_SPECULATIVE_CCD_SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_SPHERE_SPECULATIVE_CCD_SOURCE_GATE=PASS "
        "geometry=static-sphere actors=surface+volume "
        "negativeControl=discrete scope=world-static"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
