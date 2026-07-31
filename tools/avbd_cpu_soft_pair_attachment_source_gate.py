#!/usr/bin/env python3
"""Fail closed if CPU AVBD soft-pair attachments lose one coupled owner."""

from __future__ import annotations

import sys
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


def require(
    errors: list[str], condition: bool, description: str
) -> None:
    if not condition:
        errors.append(description)


def require_all(
    errors: list[str],
    text: str,
    tokens: tuple[str, ...],
    scope: str,
) -> None:
    for token in tokens:
        require(errors, token in text, f"{scope} lost {token!r}")


def main() -> int:
    errors: list[str] = []
    component = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
    )
    solver_header = read(
        "physx/source/lowleveldynamics/src/DyAvbdSolver.h"
    )
    solver = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSolverJointPath.cpp"
    )
    dynamics = read(
        "physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp"
    )
    scene_header = read(
        "physx/source/simulationcontroller/include/ScScene.h"
    )
    scene = read(
        "physx/source/simulationcontroller/src/ScScene.cpp"
    )
    np_header = read(
        "physx/source/physx/src/NpDeformableAttachment.h"
    )
    np_attachment = read(
        "physx/source/physx/src/NpDeformableAttachment.cpp"
    )
    factory = read("physx/source/physx/src/NpFactory.cpp")
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

    require_all(
        errors,
        component,
        (
            "eDYNAMIC_SOFT",
            "eSOFT_PAIR_ATTACHMENT_POSITION_AL",
            "AvbdSoftPoint targetPoint",
            "objective.targetPoint = attachment.targetPoint",
        ),
        "compiled soft-pair owner",
    )
    require_all(
        errors,
        solver_header + solver,
        (
            "solveSoftPairAttachmentsCoupled(",
            "avbdUpdateSoftPairAttachmentDual(",
            "eSOFT_PAIR_ATTACHMENT_POSITION_AL",
        ),
        "soft-pair primal/dual owner",
    )
    require_all(
        errors,
        dynamics + solver,
        (
            "bodyAllocationCount = "
            "PxMax<PxU32>(totalBodyCount, 1u)",
            "hasCompleteSoftSelection",
            "numBodies == 0 && !hasCompleteSoftSelection",
        ),
        "soft-only main-island dispatch",
    )
    require(
        errors,
        "if (totalBodyCount == 0)\n    return;" not in dynamics,
        "zero-rigid soft island returned before main AVBD selection",
    )
    soft_pair_solve = section(
        solver,
        "void AvbdSolver::solveSoftPairAttachmentsCoupled(",
        "//",
    )
    require(
        errors,
        bool(soft_pair_solve),
        "could not isolate soft-pair coupled solve",
    )
    for forbidden in (
        "applyImpulse(",
        "setLinearVelocity(",
        "setAngularVelocity(",
        "AvbdSoftContact",
    ):
        require(
            errors,
            forbidden not in soft_pair_solve,
            "soft-pair attachment entered contact/velocity path "
            f"{forbidden!r}",
        )

    require_all(
        errors,
        scene,
        (
            "struct SoftPairAttachmentEntry",
            "mSoftPairAttachments",
            "addSoftPairAttachment(",
            "removeSoftPairAttachment(",
            "ensureNativeSoftSoftIslandEdge(",
            "eDYNAMIC_SOFT",
        ),
        "Scene canonical soft-pair storage",
    )
    require_all(
        errors,
        scene_header + scene,
        (
            "addAvbdCpuDeformablePairAttachment(",
            "removeAvbdCpuDeformablePairAttachment(",
        ),
        "Scene public soft-pair bridge",
    )
    for forbidden in (
        "SurfaceSurfaceAttachmentEntry",
        "VolumeVolumeAttachmentEntry",
        "VolumeSurfaceAttachmentEntry",
        "mSurfaceSurfaceAttachments",
        "mVolumeVolumeAttachments",
        "mVolumeSurfaceAttachments",
    ):
        require(
            errors,
            forbidden not in scene,
            f"duplicate combination storage returned: {forbidden}",
        )

    require_all(
        errors,
        np_header + np_attachment,
        (
            "isCpuAvbdSoftPairAttachment()",
            "CpuAvbdRoute::eSOFT_PAIR",
            "addAvbdCpuDeformablePairAttachment(",
            "removeAvbdCpuDeformablePairAttachment(",
        ),
        "Np typed soft-pair route",
    )
    require_all(
        errors,
        np_attachment,
        (
            "mCpuAvbdRoute != CpuAvbdRoute::eSOFT_PAIR",
            "updatePose is not defined for a ",
        ),
        "Np soft-pair updatePose policy",
    )
    soft_pair_predicate = section(
        np_attachment,
        "bool NpDeformableAttachment::"
        "isCpuAvbdSoftPairAttachment() const",
        "NpScene* NpDeformableAttachment::getSceneFromActors()",
    )
    require_all(
        errors,
        soft_pair_predicate,
        (
            "eSURFACE_VTX_SURFACE_VTX",
            "eSURFACE_TRI_SURFACE_VTX",
            "eSURFACE_TRI_SURFACE_TRI",
            "eVOLUME_VTX_SURFACE_VTX",
            "eVOLUME_TET_SURFACE_VTX",
            "eVOLUME_TET_SURFACE_TRI",
            "eVOLUME_VTX_VOLUME_VTX",
            "eVOLUME_TET_VOLUME_VTX",
            "eVOLUME_TET_VOLUME_TET",
        ),
        "Np public soft-pair combination matrix",
    )
    require_all(
        errors,
        factory,
        (
            "cpuSoftPair",
            "hasValidCpuAvbdSoftPairAttachmentData(",
        ),
        "Factory soft-pair fail-closed support",
    )

    require_all(
        errors,
        surface + surface_runner,
        (
            "surface-surface-attachment",
            "surface-volume-attachment",
        ),
        "Surface public soft-pair cases",
    )
    require_all(
        errors,
        volume + volume_runner,
        ("scene-volume-volume-attachment",),
        "Volume public soft-pair case",
    )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_SOFT_PAIR_ATTACHMENT_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_SOFT_PAIR_ATTACHMENT_SOURCE_GATE] status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_SOFT_PAIR_ATTACHMENT_SOURCE_GATE] status=PASS "
        "representation=two-weighted-points "
        "owner=soft-pair-position-al reaction=two-sided "
        "velocityImpulse=none duplicateStorage=none "
        "softOnlyMainIsland=owned"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
