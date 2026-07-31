#!/usr/bin/env python3
"""Fail closed if element attachments fork prescribed/generalized owners."""

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
    component = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
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
        "Surface kinematic element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableSurfaceKinematicElementAttachment(",
        ),
        "Volume kinematic element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableVolumeKinematicElementAttachment(",
        ),
        "Surface articulation element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableSurfaceArticulationElementAttachment(",
        ),
        "Volume articulation element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableVolumeArticulationElementAttachment(",
        ),
        "Np kinematic element predicate": (
            np_header + np_attachment,
            "isCpuAvbdKinematicElementAttachment()",
        ),
        "Np articulation element predicate": (
            np_header + np_attachment,
            "isCpuAvbdArticulationElementAttachment()",
        ),
        "Np kinematic element route": (
            np_header + np_attachment,
            "eKINEMATIC_RIGID_ELEMENT",
        ),
        "Np articulation element route": (
            np_header + np_attachment,
            "eARTICULATION_LINK_ELEMENT",
        ),
        "factory kinematic element support": (
            factory,
            "cpuKinematicRigidElement",
        ),
        "factory articulation element support": (
            factory,
            "cpuArticulationRigidElement",
        ),
        "Surface kinematic public gate": (
            surface,
            "surface-kinematic-element-attachment",
        ),
        "Surface articulation public gate": (
            surface,
            "surface-articulation-element-attachment",
        ),
        "Volume kinematic public gate": (
            volume,
            "scene-volume-kinematic-element-attachment",
        ),
        "Volume articulation public gate": (
            volume,
            "scene-volume-articulation-element-attachment",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    articulation_entry = section(
        scene,
        "struct ArticulationAttachmentEntry",
        "struct PrescribedAttachmentEntry",
    )
    kinematic_entry = section(
        scene,
        "struct PrescribedAttachmentEntry",
        "struct RigidActorFilterEntry",
    )
    for description, entry in (
        ("articulation canonical entry", articulation_entry),
        ("kinematic canonical entry", kinematic_entry),
    ):
        if not entry:
            errors.append(f"could not isolate {description}")
            continue
        if "Dy::AvbdSoftPoint" not in entry or "localPoint" not in entry:
            errors.append(f"{description} does not own AvbdSoftPoint")
        if "localVertex;" in entry:
            errors.append(f"{description} retains vertex-only shadow state")

    kinematic_compile = section(
        scene,
        "bool rebuildEntryPins(Entry& entry)",
        "void refreshPrescribedAttachmentTargets()",
    )
    for token in (
        "pin.point = source.localPoint",
        "endpoint < pin.point.particleCount",
        "pin.point.particleIndices[endpoint] +=",
        "ePRESCRIBED_RIGID",
    ):
        if token not in kinematic_compile:
            errors.append(
                "kinematic element did not reuse weighted pin owner "
                f"{token!r}"
            )

    articulation_solve = section(
        solver,
        "if (articulationOwner)",
        "AvbdSoftRigidAttachmentCoupledStep step;",
    )
    for token in (
        "avbdGetSoftPointPosition(",
        "objective.point.particleCount",
        "objective.point.weights[endpoint]",
        "eARTICULATION_ATTACHMENT_POSITION_AL",
    ):
        if token not in solver:
            errors.append(
                "articulation element did not reuse generalized "
                f"weighted owner {token!r}"
            )
    for forbidden in (
        "pxcFsApplyImpulse",
        "applyImpulse(",
        "setLinearVelocity(",
        "setAngularVelocity(",
    ):
        if forbidden in articulation_solve:
            errors.append(
                "articulation element entered velocity/impulse path "
                f"{forbidden!r}"
            )

    if component.count("eKINEMATIC_ATTACHMENT_POSITION_AL") < 3:
        errors.append("kinematic Position AL owner is no longer canonical")
    if component.count("eARTICULATION_ATTACHMENT_POSITION_AL") < 3:
        errors.append(
            "articulation Position AL owner is no longer canonical"
        )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_PRESCRIBED_GENERALIZED_ELEMENT_"
                "ATTACHMENT_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_PRESCRIBED_GENERALIZED_ELEMENT_"
            "ATTACHMENT_SOURCE_GATE] status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_PRESCRIBED_GENERALIZED_ELEMENT_"
        "ATTACHMENT_SOURCE_GATE] status=PASS "
        "representation=weighted-point "
        "kinematicOwner=prescribed-position-al "
        "articulationOwner=generalized-position-al "
        "velocityImpulse=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
