#!/usr/bin/env python3
"""Fail closed if CPU AVBD kinematic attachments regain rigid reaction."""

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


def main() -> int:
    errors: list[str] = []
    soft = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
    )
    joint = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSolverJointPath.cpp"
    )
    sc_header = read(
        "physx/source/simulationcontroller/include/ScScene.h"
    )
    sc = read("physx/source/simulationcontroller/src/ScScene.cpp")
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

    required = {
        "typed pin target kind": (
            soft,
            "enum class AvbdSoftPinTargetKind",
        ),
        "prescribed target kind": (soft, "ePRESCRIBED_RIGID"),
        "unique compiled owner": (
            soft,
            "eKINEMATIC_ATTACHMENT_POSITION_AL",
        ),
        "target history": (soft, "previousWorldTarget"),
        "owner compiler": (soft, "avbdGetPinObjectiveOwner("),
        "Scene entry": (sc, "struct PrescribedAttachmentEntry"),
        "Scene storage": (sc, "mPrescribedAttachments"),
        "target refresh": (
            sc,
            "refreshPrescribedAttachmentTargets()",
        ),
        "body target read": (sc, "getKinematicTarget("),
        "COM local conversion": (
            sc,
            "getBody2Actor().getInverse()",
        ),
        "Surface Scene API": (
            sc_header + sc,
            "addAvbdCpuDeformableSurfaceKinematicAttachment(",
        ),
        "Volume Scene API": (
            sc_header + sc,
            "addAvbdCpuDeformableVolumeKinematicAttachment(",
        ),
        "Np typed route": (
            np_attachment,
            "isCpuAvbdKinematicVertexAttachment()",
        ),
        "Np persistent route enum": (
            np_attachment,
            "CpuAvbdRoute::eKINEMATIC_RIGID",
        ),
        "Np persistent route state": (
            np_attachment,
            "mCpuAvbdRoute ==",
        ),
        "factory kinematic route": (
            factory,
            "cpuKinematicRigidVertex",
        ),
        "Surface public gate": (
            surface,
            "surface-kinematic-attachment",
        ),
        "Volume public gate": (
            volume,
            "scene-volume-kinematic-attachment",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    add_path = section(
        sc,
        "PxU32 addKinematicAttachment(",
        "PxU32 addRigidAttachment(",
    )
    if not add_path:
        errors.append("could not isolate kinematic Scene path")
    for forbidden in (
        "ensureNativeIslandEdge(",
        "mRigidAttachments",
        "rigidCore.wakeUp(",
        "AvbdSoftAttachment",
    ):
        if forbidden in add_path:
            errors.append(
                "kinematic Scene path regained rigid coupling "
                f"{forbidden!r}"
            )

    coupled = section(
        joint,
        "void AvbdSolver::solveSoftRigidAttachmentsCoupled(",
        "void AvbdSolver::",
    )
    if "eKINEMATIC_ATTACHMENT_POSITION_AL" in coupled:
        errors.append(
            "kinematic attachment entered the coupled rigid block"
        )
    if joint.count("eKINEMATIC_ATTACHMENT_POSITION_AL") < 3:
        errors.append(
            "kinematic owner is not covered by warmstart/primal/dual"
        )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_KINEMATIC_ATTACHMENT_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_KINEMATIC_ATTACHMENT_SOURCE_GATE] status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_KINEMATIC_ATTACHMENT_SOURCE_GATE] "
        "status=PASS owner=position-al reaction=none "
        "targetHistory=current-previous"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
