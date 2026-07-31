#!/usr/bin/env python3
"""Fail closed if CPU AVBD Volume OGC loses its boundary query domain."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOFT_COMPONENT = (
    ROOT / "physx/source/lowleveldynamics/src/DyAvbdSoftBodyComponent.h"
)
SC_SCENE = ROOT / "physx/source/simulationcontroller/src/ScScene.cpp"
SC_VOLUME_CORE_HEADER = (
    ROOT
    / "physx/source/simulationcontroller/include/ScDeformableVolumeCore.h"
)
SC_VOLUME_CORE_SOURCE = (
    ROOT
    / "physx/source/simulationcontroller/src/ScDeformableVolumeCore.cpp"
)
NP_VOLUME = ROOT / "physx/source/physx/src/NpDeformableVolume.cpp"
SURFACE_SNIPPET = (
    ROOT
    / "physx/snippets/snippetdeformablesurfaceavbd/"
    "SnippetDeformableSurfaceAVBD.cpp"
)


def require_all(
    errors: list[str], text: str, fragments: tuple[str, ...], scope: str
) -> None:
    for fragment in fragments:
        if fragment not in text:
            errors.append(f"{scope} lost {fragment!r}")


def main() -> int:
    errors: list[str] = []
    soft_component = SOFT_COMPONENT.read_text(encoding="utf-8")
    sc_scene = SC_SCENE.read_text(encoding="utf-8")
    sc_volume_core_header = SC_VOLUME_CORE_HEADER.read_text(encoding="utf-8")
    sc_volume_core_source = SC_VOLUME_CORE_SOURCE.read_text(encoding="utf-8")
    np_volume = NP_VOLUME.read_text(encoding="utf-8")
    surface_snippet = SURFACE_SNIPPET.read_text(encoding="utf-8")

    require_all(
        errors,
        soft_component,
        (
            "PxArray<PxU32> surfaceVertices;",
            "surfaceVertices = surfaceTriangles;",
            "PxSort(surfaceVertices.begin(), surfaceVertices.size());",
            "avbdIsSoftBodySurfaceVertex(",
            "!avbdIsSoftBodySurfaceVertex(*sourceBody, i)",
            "!avbdIsSoftBodySurfaceVertex(*sourceBody, pi)",
            "testBody.compiled.surfaceVertices.size()",
            "sb.compiled.surfaceVertices.size()",
        ),
        "compiled Volume boundary-only OGC query domain",
    )
    require_all(
        errors,
        sc_scene,
        (
            "body.compiled.surfaceVertices.size()",
            "body.compiled.surfaceEdges.size()",
            "initializeCpuAvbdSimulationRestPositions(",
            "getCpuAvbdSimulationRestPositions()",
        ),
        "Scene rebase and Volume rest-state preservation",
    )
    require_all(
        errors,
        sc_volume_core_header + sc_volume_core_source,
        (
            "mCpuAvbdSimulationRestPositions",
            "initializeCpuAvbdSimulationRestPositions(",
            "clearCpuAvbdSimulationRestPositions()",
        ),
        "persistent CPU AVBD Volume simulation rest state",
    )
    require_all(
        errors,
        np_volume,
        (
            "attachSimulationMesh(",
            "detachSimulationMesh()",
            "mCore.clearCpuAvbdSimulationRestPositions();",
        ),
        "Volume simulation-mesh lifecycle rest-state invalidation",
    )
    require_all(
        errors,
        surface_snippet,
        (
            "position.y += 0.049f;",
            "3.0f * local.x",
            '"volume-volume-element-filter"',
            "partialFilterExactOwnership",
            "elementFilterContactRestored",
        ),
        "Volume--Volume exact-filter recovery fixture",
    )

    if errors:
        print("AVBD_CPU_VOLUME_OGC_BOUNDARY_SOURCE_GATE=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("AVBD_CPU_VOLUME_OGC_BOUNDARY_SOURCE_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
