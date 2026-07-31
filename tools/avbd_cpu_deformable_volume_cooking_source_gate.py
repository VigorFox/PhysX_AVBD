#!/usr/bin/env python3
"""Freeze CPU deformable-volume cooking ownership and stream compatibility."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MESH_DATA = ROOT / "physx/source/geomutils/src/mesh/GuMeshData.h"
COOKING = (
    ROOT
    / "physx/source/geomutils/src/cooking/GuCookingTetrahedronMesh.cpp"
)
LOADER = ROOT / "physx/source/geomutils/src/GuMeshFactory.cpp"
UNIT = (
    ROOT
    / "physx/snippets/snippetsoftbodyavbd/SnippetSoftBodyAVBD.cpp"
)
VOLUME = (
    ROOT
    / "physx/snippets/snippetdeformablevolumeavbd/"
    "SnippetDeformableVolumeAVBD.cpp"
)
SURFACE = (
    ROOT
    / "physx/snippets/snippetdeformablesurfaceavbd/"
    "SnippetDeformableSurfaceAVBD.cpp"
)


def require(errors: list[str], condition: bool, description: str) -> None:
    if not condition:
        errors.append(description)


def require_all(
    errors: list[str], text: str, fragments: tuple[str, ...], scope: str
) -> None:
    for fragment in fragments:
        require(errors, fragment in text, f"{scope} lost {fragment!r}")


def main() -> int:
    errors: list[str] = []
    mesh_data = MESH_DATA.read_text(encoding="utf-8")
    cooking = COOKING.read_text(encoding="utf-8")
    loader = LOADER.read_text(encoding="utf-8")
    unit = UNIT.read_text(encoding="utf-8")
    volume = VOLUME.read_text(encoding="utf-8")
    surface = SURFACE.read_text(encoding="utf-8")

    require_all(
        errors,
        mesh_data,
        (
            "#define PX_DEFORMABLE_VOLUME_MESH_VERSION 4",
            "IMSF_DEFORMABLE_DATA",
            "(1<<10)",
            "backend-neutral deformable simulation and mapping data",
        ),
        "cooked format declaration",
    )
    require_all(
        errors,
        cooking,
        (
            "collisionData.allocateCollisionData("
            "collisionMeshDesc.tetrahedrons.count)",
            "recordTetrahedronIndices(collisionMesh, collisionData)",
            "serialFlags |= IMSF_DEFORMABLE_DATA",
            "if (serialFlags & IMSF_DEFORMABLE_DATA)",
            "if (serialFlags & IMSF_GRB_DATA)",
            "BV32TriangleMeshBuilder::saveMidPhaseStructure",
            "writeFloatBuffer("
            "mappingData.mVertsBarycentricInGridModel",
            "writeIntBuffer(mappingData.mVertsRemapInGridModel",
        ),
        "deformable-volume writer",
    )
    require(
        errors,
        "recordTetrahedronIndices("
        "collisionMesh, collisionData, params.buildGPUData)" not in cooking,
        "shared collision topology is still conditional on buildGPUData",
    )
    require(
        errors,
        "computeModelsMapping(simulationMesh, collisionMesh, collisionData, "
        "mappingData, params.buildGPUData" not in cooking,
        "shared collision embedding is still conditional on buildGPUData",
    )
    require_all(
        errors,
        loader,
        (
            "const bool hasGRBData",
            "const bool hasDeformableData",
            "(serialFlags & IMSF_DEFORMABLE_DATA) != 0 || hasGRBData",
            "if (hasDeformableData)",
            "if (hasGRBData)",
            "data.mCollisionData.allocateCollisionData(nbTetrahedrons)",
            "data.mSimulationMesh.allocateTetrahedrons("
            "nbGridModelTetrahedrons, 1)",
            "data.mMappingData.allocatemappingData(",
        ),
        "deformable-volume loader",
    )
    require_all(
        errors,
        unit,
        (
            "deformableCookingParams.buildGPUData = false",
            "PxCookDeformableVolumeMesh(",
            "CPU-only cooked stream reloads complete shared deformable data",
            "Optional GRB cooked stream retains the shared deformable payload",
            "Version-3 GRB stream remains backward compatible",
            "secondaryVolume->attachSimulationMesh(",
            "*streamedVolumeMesh->getSimulationMesh()",
        ),
        "runtime stream gate",
    )
    require(
        errors,
        "cookingParams.buildGPUData = false" in volume,
        "Volume Scene fixture no longer freezes CPU-only cooking",
    )
    require(
        errors,
        "cookingParams.buildGPUData = false" in surface,
        "Surface mixed fixture no longer freezes CPU-only cooking",
    )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_DEFORMABLE_VOLUME_COOKING_SOURCE_GATE_ERROR] "
                f"{error}"
            )
        print("[AVBD_CPU_DEFORMABLE_VOLUME_COOKING_SOURCE_GATE] status=FAIL")
        return 1

    print(
        "[AVBD_CPU_DEFORMABLE_VOLUME_COOKING_SOURCE_GATE] status=PASS "
        "immediate=cpu-only stream=v4-shared grb=optional legacy=v3"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
