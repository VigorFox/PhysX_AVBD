#!/usr/bin/env python3
"""Lock E36 CPU AVBD sphere-to-soft reverse edge/face ownership."""

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

    reverse = section(
        soft,
        "inline void avbdDetectSoftRigidSphereOGCFeatures(",
        "inline void avbdDetectSoftRigidOGCFeatures(",
    )
    require_all(
        errors,
        "sphere reverse edge/face detector",
        reverse,
        (
            "const PxReal queryRadius =",
            "sphere.radius + margin;",
            "body.compiled.surfaceTriangles",
            "avbdClosestPointOnTriangleOGC(",
            "closest.feature == AVBD_FEATURE_VERTEX",
            "closest.feature == AVBD_FEATURE_UNKNOWN",
            "closest.distance >= queryRadius",
            "avbdSoftTriangleFeatureKey(",
            "PxArray<PxU64> emittedFeatureKeys;",
            "if(duplicate)",
            "emittedFeatureKeys.pushBack(featureKey);",
            "PxVec3 normal = -closest.normal;",
            "geometry.queryParticleIndices[0] = v0;",
            "geometry.queryParticleIndices[1] = v1;",
            "geometry.queryParticleIndices[2] = v2;",
            "geometry.queryWeights[0] =",
            "closest.barycentric.x;",
            "geometry.queryWeights[1] =",
            "closest.barycentric.y;",
            "geometry.queryWeights[2] =",
            "closest.barycentric.z;",
            "geometry.depth =",
            "queryRadius - closest.distance;",
            "geometry.margin = margin;",
            "const PxVec3 surfaceLocal =",
            "normal * sphere.radius",
            "avbdConfigureRigidSphereTarget(",
            "avbdCombineDeformableRigidFriction(",
            "avbdAppendPreparedSoftContact(",
        ),
    )
    if "closest.feature == AVBD_FEATURE_VERTEX" not in reverse:
        errors.append(
            "sphere reverse ownership no longer excludes vertex-SDF features"
        )

    target = section(
        soft,
        "PX_FORCE_INLINE void avbdConfigureRigidSphereTarget(",
        "inline void avbdDetectSoftRigidSphereSDF(",
    )
    require_all(
        errors,
        "shared sphere target ownership",
        target,
        (
            "sphere.targetKind",
            "AvbdSoftContactTargetKind::eRIGID_BODY",
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
        "aggregate sphere reverse route",
        aggregate,
        (
            "avbdDetectSoftRigidSphereSDF(",
            "avbdDetectSoftRigidSphereSweptSDF(",
            "avbdDetectSoftRigidSphereOGCFeatures(",
            "rigidSpheres, numRigidSpheres",
            "softBodies, numSoftBodies",
            "contacts, params.contactRadius",
        ),
    )

    selection = section(
        scene,
        "bool buildIslandSelectionStorage(",
        "void copyIslandSelectionResults(",
    )
    require_all(
        errors,
        "selected dynamic-sphere reverse probe",
        selection,
        (
            "Dy::avbdDetectSoftRigidSphereSDF(",
            "Dy::avbdDetectSoftRigidSphereSweptSDF(",
            "Dy::avbdDetectSoftRigidSphereOGCFeatures(",
            "storage.selectedDynamicSpheres.begin()",
            "storage.bodies.begin()",
            "storage.probeContacts",
            "mContactParams.contactRadius",
        ),
    )

    public_fragments = (
        "[AVBD_SPHERE_REVERSE_FEATURE]",
        "faceResponseObserved",
        "vertexSdfExcluded",
        "negativeControlPassed",
        "nonFiniteSamples",
        "positiveDisplacement",
        "positiveDrop",
        "negativeDrop",
        "faceSeparation",
        "minimumVertexSeparation",
    )
    require_all(
        errors,
        "Surface reverse-feature public gate",
        surface + surface_runner,
        (
            "surface-sphere-reverse-feature",
            "PxSphereGeometry(radius)",
            "minimumVertexSeparation > 0.10f",
            "faceSeparation > 0.02f",
            "positiveDrop + 0.01f < negativeDrop",
        )
        + public_fragments,
    )
    require_all(
        errors,
        "Volume reverse-feature public gate",
        volume + volume_runner,
        (
            "scene-volume-sphere-reverse-feature",
            "addSceneStaticSphereCluster(",
            "getSceneCpuVolumeSmoothReverseSeparations(",
            "minimumVertexSeparation > 0.10f",
            "faceSeparation > 0.02f",
            "positiveDrop + 0.01f <",
            "SCENE_SPHERE_REVERSE_FEATURE_GATED",
        )
        + public_fragments,
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume E36 gates bypass the public Scene API"
        )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_SPHERE_REVERSE_FEATURE_SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_SPHERE_REVERSE_FEATURE_SOURCE_GATE=PASS "
        "geometry=sphere softFeature=edge+face vertexOwner=forward-sdf "
        "target=static+kinematic+dynamic actors=surface+volume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
