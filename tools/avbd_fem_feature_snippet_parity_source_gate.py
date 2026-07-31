#!/usr/bin/env python3
"""Lock the CPU AVBD counterparts of the remaining FEM feature snippets."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    cmake = read("physx/snippets/compiler/cmake/CMakeLists.txt")
    template = read(
        "physx/snippets/compiler/cmake/SnippetTemplate.cmake"
    )
    skinning = read(
        "physx/snippets/snippetcommon/"
        "SnippetDeformableAVBDSkinning.h"
    )
    volume = read(
        "physx/snippets/snippetdeformablevolumeavbd/"
        "SnippetDeformableVolumeAVBD.cpp"
    )
    volume_render = read(
        "physx/snippets/snippetdeformablevolumeavbd/"
        "SnippetDeformableVolumeAVBDRender.cpp"
    )
    surface = read(
        "physx/snippets/snippetdeformablesurfaceavbd/"
        "SnippetDeformableSurfaceAVBD.cpp"
    )
    surface_render = read(
        "physx/snippets/snippetdeformablesurfaceavbd/"
        "SnippetDeformableSurfaceAVBDRender.cpp"
    )
    runner = read(
        "tools/run_snippet_deformable_avbd_feature_demos_headless.py"
    )

    required = {
        "triangle binding": (
            skinning,
            "struct AvbdTriangleSkinningBinding",
        ),
        "tetrahedron binding": (
            skinning,
            "struct AvbdTetrahedronSkinningBinding",
        ),
        "triangle barycentric evaluation": (
            skinning,
            "evaluateTriangleSkinning(",
        ),
        "tetrahedron barycentric evaluation": (
            skinning,
            "evaluateTetrahedronSkinning(",
        ),
        "finite normalized render normals": (
            skinning,
            "updateSkinningNormals(",
        ),
        "Surface skinning case": (
            surface,
            '"surface-skinning"',
        ),
        "Surface per-frame skinning": (
            surface,
            "evaluateTriangleSkinning(",
        ),
        "Surface skinning render route": (
            surface_render,
            "skinnedPositions",
        ),
        "Volume skinning case": (
            volume,
            '"scene-volume-skinning"',
        ),
        "Volume per-frame skinning": (
            volume,
            "evaluateTetrahedronSkinning(",
        ),
        "Volume skinning render route": (
            volume_render,
            "gVolumeAvbdSkinningRenderData",
        ),
        "dedicated no-window runner": (
            runner,
            "[AVBD_FEATURE_DEMOS]",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    demos = {
        "DeformableVolumeAttachmentAVBD": (
            "snippetdeformablevolumeattachmentavbd/"
            "SnippetDeformableVolumeAttachmentAVBD.cpp",
            "scene-volume-rigid-attachment",
        ),
        "DeformableVolumeKinematicAVBD": (
            "snippetdeformablevolumekinematicavbd/"
            "SnippetDeformableVolumeKinematicAVBD.cpp",
            "scene-volume-partial-kinematic-target",
        ),
        "DeformableVolumeSkinningAVBD": (
            "snippetdeformablevolumeskinningavbd/"
            "SnippetDeformableVolumeSkinningAVBD.cpp",
            "scene-volume-skinning",
        ),
        "DeformableSurfaceSkinningAVBD": (
            "snippetdeformablesurfaceskinningavbd/"
            "SnippetDeformableSurfaceSkinningAVBD.cpp",
            "surface-skinning",
        ),
    }
    wrapper_root = "physx/snippets/"
    wrapper_text = ""
    for demo, (relative, default_case) in demos.items():
        source = read(wrapper_root + relative)
        wrapper_text += source
        if demo not in cmake:
            errors.append(f"CMake target lost {demo!r}")
        if demo not in template:
            errors.append(
                f"LowLevelDynamics include route lost {demo!r}"
            )
        if f"Snippet{demo}" not in source:
            errors.append(f"wrapper identity lost {demo!r}")
        if default_case not in source:
            errors.append(
                f"{demo} default case lost {default_case!r}"
            )
        if f"Snippet{demo}_64.exe" not in runner:
            errors.append(f"runner lost {demo!r}")

    for forbidden in (
        "PxCudaContextManager",
        "copyToDevice",
        "PxDeformableSkinning",
        "cuda.h",
    ):
        if forbidden in skinning + wrapper_text:
            errors.append(
                "CPU feature snippets regained GPU dependency "
                f"{forbidden!r}"
            )

    if errors:
        for error in errors:
            print(
                "[AVBD_FEM_FEATURE_SNIPPET_PARITY_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_FEM_FEATURE_SNIPPET_PARITY_SOURCE_GATE] "
            "status=FAIL"
        )
        return 1

    print(
        "[AVBD_FEM_FEATURE_SNIPPET_PARITY_SOURCE_GATE] status=PASS "
        "targets=4 skinning=triangle,tetrahedron "
        "backend=cpu-scene gpuDependency=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
