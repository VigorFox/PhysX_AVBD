#!/usr/bin/env python3
"""Fail closed if CPU AVBD element attachments lose weighted-point ownership."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    component = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
    )
    internal = read(
        "physx/source/lowleveldynamics/src/DyAvbdSoftBody.h"
    )
    solver = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSolverJointPath.cpp"
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
    soft_tests = read(
        "physx/snippets/snippetsoftbodyavbd/"
        "SnippetSoftBodyAVBD.cpp"
    )
    surface = read(
        "physx/snippets/snippetdeformablesurfaceavbd/"
        "SnippetDeformableSurfaceAVBD.cpp"
    )
    volume = read(
        "physx/snippets/snippetdeformablevolumeavbd/"
        "SnippetDeformableVolumeAVBD.cpp"
    )
    errors: list[str] = []

    required = {
        "weighted soft point": (
            component,
            "struct AvbdSoftPoint",
        ),
        "four endpoint indices": (
            component,
            "particleIndices[4]",
        ),
        "four endpoint weights": (
            component,
            "weights[4]",
        ),
        "weighted point position": (
            component,
            "avbdGetSoftPointPosition(",
        ),
        "weighted point Jacobian": (
            component,
            "avbdGetSoftPointJacobianWeight(",
        ),
        "compiled point snapshot": (
            component,
            "objective.point = attachment.point",
        ),
        "compiled pin point snapshot": (
            component,
            "objective.point = pin.point",
        ),
        "weighted inverse response": (
            internal,
            "avbdGetSoftPointInverseMass(",
        ),
        "coupled weighted corrections": (
            internal + solver,
            "particleCorrections",
        ),
        "solver consumes compiled point": (
            solver,
            "objective.point",
        ),
        "Surface world element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableSurfaceWorldElementAttachment(",
        ),
        "Volume world element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableVolumeWorldElementAttachment(",
        ),
        "Surface rigid element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableSurfaceRigidElementAttachment(",
        ),
        "Volume rigid element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableVolumeRigidElementAttachment(",
        ),
        "Np world element route": (
            np_header + np_attachment,
            "isCpuAvbdWorldElementAttachment()",
        ),
        "Np rigid element route": (
            np_header + np_attachment,
            "isCpuAvbdRigidElementAttachment()",
        ),
        "factory world element support": (
            factory,
            "cpuWorldElement",
        ),
        "factory rigid element support": (
            factory,
            "cpuDynamicRigidElement",
        ),
        "factory normalized barycentric rejection": (
            factory,
            "isValidCpuAvbdElementBarycentric(",
        ),
        "factory element topology bounds rejection": (
            factory,
            "getCpuAvbdElementCount(",
        ),
        "factory element data fail closed": (
            factory,
            "hasValidCpuAvbdElementAttachmentData(",
        ),
        "component weighted test": (
            soft_tests,
            "Weighted-Point Attachment Position AL",
        ),
        "Surface public world gate": (
            surface,
            "surface-world-element-attachment",
        ),
        "Surface public rigid gate": (
            surface,
            "surface-rigid-element-attachment",
        ),
        "Volume public world gate": (
            volume,
            "scene-volume-world-element-attachment",
        ),
        "Volume public rigid gate": (
            volume,
            "scene-volume-rigid-element-attachment",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_BARYCENTRIC_ATTACHMENT_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_BARYCENTRIC_ATTACHMENT_SOURCE_GATE] "
            "status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_BARYCENTRIC_ATTACHMENT_SOURCE_GATE] status=PASS "
        "owner=weighted-point-position-al approximation=none "
        "routes=surface-triangle,volume-tetrahedron "
        "targets=world,dynamic-rigid"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
