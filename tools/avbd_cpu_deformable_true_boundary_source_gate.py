#!/usr/bin/env python3
"""Fail closed if CPU AVBD Volume stops using the cooked collision boundary."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOFT_IR = ROOT / "physx/source/lowleveldynamics/src/DyAvbdSoftBodyComponent.h"
SCENE = ROOT / "physx/source/simulationcontroller/src/ScScene.cpp"
SNIPPET = (
    ROOT
    / "physx/snippets/snippetdeformablevolumeavbd/"
    "SnippetDeformableVolumeAVBD.cpp"
)
RENDER = (
    ROOT
    / "physx/snippets/snippetdeformablevolumeavbd/"
    "SnippetDeformableVolumeAVBDRender.cpp"
)
PVD_BINDING = ROOT / "physx/source/physx/src/PvdMetaDataPvdBinding.cpp"
PVD_CLIENT = ROOT / "physx/source/physx/src/NpPvdSceneClient.cpp"
PVD_TYPES = ROOT / "physx/source/physx/src/PvdTypeNames.h"
FETCH_RESULTS = ROOT / "physx/source/physx/src/NpSceneFetchResults.cpp"
RUNNER = ROOT / "tools/run_snippet_deformable_volume_avbd_headless.py"


def require_all(
    errors: list[str], text: str, fragments: tuple[str, ...], scope: str
) -> None:
    for fragment in fragments:
        if fragment not in text:
            errors.append(f"{scope} lost {fragment!r}")


def main() -> int:
    errors: list[str] = []
    soft_ir = SOFT_IR.read_text(encoding="utf-8")
    scene = SCENE.read_text(encoding="utf-8")
    snippet = SNIPPET.read_text(encoding="utf-8")
    render = RENDER.read_text(encoding="utf-8")
    pvd_binding = PVD_BINDING.read_text(encoding="utf-8")
    pvd_client = PVD_CLIENT.read_text(encoding="utf-8")
    pvd_types = PVD_TYPES.read_text(encoding="utf-8")
    fetch_results = FETCH_RESULTS.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    require_all(
        errors,
        soft_ir,
        (
            "AVBD_EMBEDDED_VERTEX_SUPPORT = 4",
            "AVBD_CONTACT_POINT_MAX_SUPPORT = 12",
            "AVBD_CONTACT_MAX_PARTICLES = 24",
            "struct AvbdWeightedContactPoint",
            "AvbdWeightedContactPoint queryPoint;",
            "AvbdWeightedContactPoint targetPoint;",
            "queryCollisionElementIndex",
            "targetCollisionElementIndex",
            "avbdIsSoftContactQueryFullyKinematic(",
            "geometry.queryPoint.particleIndices[i]",
            "geometry.targetPoint.particleIndices[i]",
            "targetAbsoluteAngularMomentum",
            "contact.state.alLambdaTangent[0]",
        ),
        "expanded collision-contact IR",
    )
    require_all(
        errors,
        scene,
        (
            "validateVolumeCollisionEmbedding(",
            "mVertsRemapInGridModel",
            "mVertsBarycentricInGridModel",
            "rebuildCollisionDetectionScene()",
            "refreshCollisionDetectionScene(",
            "rebuildSubsetCollisionDetectionScene(",
            "mCollisionVertexMappings",
            "mSubsetCollisionVertexMappings",
            "mCollisionParticles.begin(), mCollisionParticles.size()",
            "mSubsetCollisionParticles.begin()",
            "expandCollisionDetectionContacts(",
            "resolveCollisionElementForFeature(",
            "CPU AVBD requires cooked collision-to-simulation vertex embedding",
            "Never reinterpret the first N simulation vertices as",
            "entry.collisionMesh != entry.simulationMesh",
            "fullSceneCollisionRequest",
            "CPU AVBD collision-domain proxy scene is incomplete before contact detection.",
            "CPU AVBD failed to build the collision-domain proxy for a public actor subset.",
            "This direct-domain path is reserved for legacy low-level callers",
        ),
        "Scene collision-boundary authority",
    )
    direct_domain_marker = scene.find(
        "This direct-domain path is reserved for legacy low-level callers"
    )
    full_scene_marker = scene.find("if(fullSceneCollisionRequest)")
    subset_marker = scene.find("if(useSubsetCookedCollisionDomain)")
    direct_ogc = scene.find(
        "Dy::avbdDetectAllOGCContacts(", direct_domain_marker
    )
    if not (
        0 <= full_scene_marker < subset_marker < direct_domain_marker < direct_ogc
    ):
        errors.append(
            "public collision requests can reach the direct Simulation Mesh OGC path"
        )
    writeback_start = scene.find("\n\t\tvoid writeBack(Entry& entry)")
    if writeback_start < 0:
        errors.append("Scene lost CPU AVBD writeBack")
    else:
        writeback = scene[writeback_start : writeback_start + 7000]
        for forbidden in (
            "PxMin(collisionVertexCount, particleCount)",
            "copyCount",
            "i < particleCount ?",
        ):
            if forbidden in writeback:
                errors.append(
                    "Volume writeBack restored a copy-first-N fallback: "
                    f"{forbidden!r}"
                )

    require_all(
        errors,
        snippet,
        (
            '#define AVBD_VOLUME_DEFAULT_CASE "scene-volume-partial-element-filter"',
            '#define AVBD_VOLUME_VISUAL_CASE "scene-volume-visual-showcase"',
            'caseName == "scene-volume-visual-showcase"',
            "initSceneCpuVolumeVisualShowcase()",
            "createSubdividedCubeSurface(",
            "createLayeredConeSurface(",
            "cubeCollisionSubdivisions = 4",
            "cubeSimulationVoxels = 5",
            "sphereSimulationVoxels = 6",
            "coneHeightSegments = 3",
            "coneSimulationVoxels = 8",
            "coneMesh->getCollisionMesh() == coneMesh->getSimulationMesh()",
            "avbdGenerateConeTets(",
            "every exterior side vertex lies on the analytic cone",
            "SCENE_CPU_VISUAL_MIN_ORIENTATION_CHANGE",
            "SCENE_CPU_VISUAL_MIN_ANGULAR_SPEED",
            "SCENE_CPU_VISUAL_MIN_SURFACE_TRIANGLES",
            "createdVolume->setSolverIterationCounts(\n\t\tSCENE_CPU_VISUAL_POSITION_ITERATIONS, 1)",
            "[AVBD_VISUAL_ROTATION]",
            "sampleSceneCpuVolumeHealth()",
            "SCENE_CPU_VISUAL_MIN_DET_F",
            "SCENE_CPU_VISUAL_MIN_VOLUME_RATIO",
            "[AVBD_VISUAL_VOLUME_HEALTH]",
            "[AVBD_VISUAL_VOLUME_BODY]",
            "getPositionInvMassBufferH()",
            "collisionRestVertices[collisionVertex].x",
            "scenePartialFilterExactOwnership",
        ),
        "public true-boundary defaults and visual showcase",
    )
    for forbidden in (
        "generateConeTetsViaTetMaker(",
        "TetMaker voxel cone:",
    ):
        if forbidden in snippet:
            errors.append(
                "visual cone restored a voxel-silhouette path: "
                f"{forbidden!r}"
            )
    require_all(
        errors,
        runner,
        (
            '"scene-volume-visual-showcase"',
            'scene_visual_showcase = case_name == "scene-volume-visual-showcase"',
            'required["validation"] = "SCENE_VISUAL_SHOWCASE_GATED"',
        ),
        "visual showcase headless acceptance routing",
    )
    require_all(
        errors,
        render,
        (
            "PxTetrahedronMeshExt::extractTetMeshSurface(",
            "getPublicVolumeSurfaceTriangles(",
            "const PxTetrahedronMesh* mesh = volume.getCollisionMesh()",
            "volume.getPositionInvMassBufferH()",
            "gScene->getNbDeformableVolumes()",
            "gScene->getDeformableVolumes(",
        ),
        "public true-boundary rendering",
    )
    for forbidden in (
        "renderSimulationBoundary",
        "collisionMesh->getNbVertices() == 8",
    ):
        if forbidden in render:
            errors.append(
                "renderer restored a simulation-boundary visual substitute: "
                f"{forbidden!r}"
            )

    require_all(
        errors,
        pvd_binding,
        (
            'createProperty<PxScene, ObjectRef>("DeformableVolumes"',
            "inStream.createClass<PxDeformableVolume>()",
            '"CollisionPositions"',
            '"CollisionTetrahedrons"',
            '"SimulationPositions"',
            '"SimulationVelocities"',
            '"SimulationTetrahedrons"',
            'addSceneGroupProperty(inStream, "DeformableVolumes"',
            "mutableObj.getPositionInvMassBufferH()",
            "mutableObj.getSimPositionInvMassBufferH()",
            'removeSceneGroupProperty(\n\t\tinStream, "DeformableVolumes"',
            'createProperty<PxTetrahedronMesh, PxVec3>("Points"',
            'createProperty<PxTetrahedronMesh, PxU32>("NbTetrahedrons"',
            'setPropertyValue(&inData, "Points"',
            'setPropertyValue(&inData, "NbTetrahedrons"',
        ),
        "legacy PVD deformable-volume schema and streaming",
    )
    require_all(
        errors,
        pvd_client,
        (
            "buildPvdTetBoundaryTriangles(",
            "visualizeCpuDeformableVolume(",
            "getNbDeformableVolumes()",
            "getDeformableVolumes(",
            "mMetaDataBinding.sendAllProperties(\n\t\t\t\t*mPvdDataStream, *volumes[i])",
            "mMetaDataBinding.createInstance(\n\t\t\t*mPvdDataStream, *deformableVolume",
            "mMetaDataBinding.destroyInstance(\n\t\t\t*mPvdDataStream, *deformableVolume",
        ),
        "legacy PVD deformable-volume lifetime and visualization",
    )
    require_all(
        errors,
        pvd_types,
        ("DEFINE_NATIVE_PVD_PHYSX3_TYPE_MAP(PxDeformableVolume)",),
        "legacy PVD type registration",
    )
    require_all(
        errors,
        fetch_results,
        (
            "CPU AVBD owns host buffers directly",
            "dv->getPositionInvMassBufferH()",
            "dv->getSimPositionInvMassBufferH()",
            "dv->getSimVelocityBufferH()",
        ),
        "OmniPVD CPU AVBD host streaming",
    )

    if errors:
        print("AVBD_CPU_DEFORMABLE_TRUE_BOUNDARY_SOURCE_GATE=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("AVBD_CPU_DEFORMABLE_TRUE_BOUNDARY_SOURCE_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
