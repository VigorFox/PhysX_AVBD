#!/usr/bin/env python3
"""Lock E33 CPU AVBD dynamic-sphere two-sided island ownership."""

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

    target = section(
        soft,
        "PX_FORCE_INLINE void avbdConfigureRigidSphereTarget(",
        "inline void avbdDetectSoftRigidSphereSDF(",
    )
    require_all(
        errors,
        "dynamic-sphere two-sided target",
        target,
        (
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "AvbdVelocityObjectiveOwner::ManifoldFinalize",
            "? sphere.targetIndex : sphereIndex;",
            "geometry.rigidLocalPoint =",
            "sphere.shapeToRigidBody.transform(surfaceLocal);",
        ),
    )

    storage = section(
        scene,
        "struct IslandSelectionStorage",
        "public:",
    )
    require_all(
        errors,
        "dynamic-sphere island storage",
        storage,
        (
            "PxArray<Dy::AvbdRigidSphere>",
            "rigidSpheres;",
            "selectedDynamicSpheres;",
        ),
    )

    selection = section(
        scene,
        "bool buildIslandSelectionStorage(",
        "void copyIslandSelectionResults(",
    )
    require_all(
        errors,
        "dynamic-sphere selection and contact aggregation",
        selection,
        (
            "storage.selectedDynamicSpheres.clear();",
            "compileDynamicSpheresForIsland(",
            "Dy::avbdDetectSoftRigidSphereSDF(",
            "Dy::avbdDetectSoftRigidSphereOGCFeatures(",
            "storage.selectedDynamicSpheres.begin()",
            "storage.probeContacts",
            "storage.rigidSpheres.pushBack(mRigidSpheres[i]);",
            "storage.rigidSpheres.pushBack(",
            "storage.selectedDynamicSpheres[i]);",
            "storage.rigidSpheres.begin()",
            "storage.rigidSpheres.size(),",
            "geometry.hasRigidBodyTarget()",
        ),
    )

    compile_sphere = section(
        scene,
        "void compileDynamicSpheresForIsland(",
        "void refreshSelfCollisionEnabled(",
    )
    require_all(
        errors,
        "dynamic-sphere native-island owner compile",
        compile_sphere,
        (
            "bodySim->isKinematic()",
            "bodySim->isArticulationLink()",
            "compileDynamicSphere(entry, sphere)",
            "rigidBodies[candidateIndex] ==",
            "solverBodies[globalBodyIndex].isStatic()",
            "sphere.targetKind =",
            "eRIGID_BODY;",
            "sphere.targetIndex =",
            "globalBodyIndex - bodyStart;",
            "sphere.shapeToRigidBody =",
            "bodyCore.body2World.getInverse() *",
            "shapeToWorld;",
            "spheres.pushBack(sphere);",
        ),
    )

    detect = section(
        scene,
        "void detectContacts(",
        "static void redetectContacts(",
    )
    require_all(
        errors,
        "per-island sphere contact route",
        detect,
        (
            "const Dy::AvbdRigidSphere* rigidSpheres = NULL",
            "PxU32 numRigidSpheres = 0",
            "rigidSpheres = mRigidSpheres.begin();",
            "numRigidSpheres = mRigidSpheres.size();",
            "rigidSpheres, numRigidSpheres,",
        ),
    )

    swept = section(
        soft,
        "inline void avbdDetectSoftRigidSphereSweptSDF(",
        "inline void avbdDetectSoftRigidOGCFeatures(",
    )
    require_all(
        errors,
        "dynamic-sphere predicted swept scope",
        swept,
        (
            "AvbdSoftContactTargetKind::eWORLD_STATIC",
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "const bool dynamicTarget =",
            "sphere.predictedPoseValid",
            "? sphere.predictedCenter : sphere.center;",
        ),
    )

    require_all(
        errors,
        "Surface dynamic-sphere public gate",
        surface + surface_runner,
        (
            "surface-dynamic-sphere",
            "dynamicSphereCase",
            "PxSphereGeometry(0.8f)",
            "dynamicBoxInitiallySleeping",
            "dynamicBoxWoke",
            "dynamicBoxMaxDrop",
            "dynamicBoxMaxLinearSpeed",
            "dynamicBoxMaxAngularSpeed",
        ),
    )
    require_all(
        errors,
        "Volume dynamic-sphere public gate",
        volume + volume_runner,
        (
            "scene-volume-dynamic-sphere",
            "addSceneDynamicSphere(",
            "PxVec3(0.0f, 1.0f, 0.0f), 0.8f",
            "sceneDynamicInitiallySleeping",
            "sceneDynamicWokeBySoft",
            "radius - 0.8f",
            "minDynamicSurfaceSeparation",
            "finalDynamicSurfaceSeparation",
            "gMetrics.finalMaxParticleSpeed < 0.5f",
            "sceneDynamicFinalY > 0.70f",
        ),
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume dynamic-sphere gates bypass "
            "the public Scene API"
        )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_DYNAMIC_SPHERE_SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_DYNAMIC_SPHERE_SOURCE_GATE=PASS "
        "geometry=sphere target=dynamic-rigid-two-sided "
        "actors=surface+volume swept=dynamic-predicted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
