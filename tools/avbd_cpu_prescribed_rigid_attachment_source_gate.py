#!/usr/bin/env python3
"""Fail closed if static and kinematic attachments fork prescribed owners."""

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
    errors: list[str] = []

    required = {
        "prescribed pin target kind": (
            component,
            "ePRESCRIBED_RIGID",
        ),
        "canonical prescribed Scene entry": (
            scene,
            "struct PrescribedAttachmentEntry",
        ),
        "canonical prescribed core": (
            scene,
            "RigidCore*\t\t\t\tprescribedCore",
        ),
        "prescribed target compiler": (
            scene,
            "computePrescribedAttachmentWorldTarget(",
        ),
        "static transform source": (
            scene,
            "getActor2World()",
        ),
        "kinematic command source": (
            scene,
            "getKinematicTarget(",
        ),
        "Surface static vertex Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableSurfaceStaticAttachment(",
        ),
        "Surface static element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableSurfaceStaticElementAttachment(",
        ),
        "Volume static vertex Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableVolumeStaticAttachment(",
        ),
        "Volume static element Scene API": (
            scene_header + scene,
            "addAvbdCpuDeformableVolumeStaticElementAttachment(",
        ),
        "Np static vertex predicate": (
            np_header + np_attachment,
            "isCpuAvbdStaticVertexAttachment()",
        ),
        "Np static element predicate": (
            np_header + np_attachment,
            "isCpuAvbdStaticElementAttachment()",
        ),
        "Np static vertex route": (
            np_header + np_attachment,
            "eSTATIC_RIGID",
        ),
        "Np static element route": (
            np_header + np_attachment,
            "eSTATIC_RIGID_ELEMENT",
        ),
        "factory static vertex support": (
            factory,
            "cpuStaticRigidVertex",
        ),
        "factory static element support": (
            factory,
            "cpuStaticRigidElement",
        ),
        "Surface static vertex public case": (
            surface + surface_runner,
            "surface-static-attachment",
        ),
        "Surface static element public case": (
            surface + surface_runner,
            "surface-static-element-attachment",
        ),
        "Volume static vertex public case": (
            volume + volume_runner,
            "scene-volume-static-attachment",
        ),
        "Volume static element public case": (
            volume + volume_runner,
            "scene-volume-static-element-attachment",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    pin_kinds = section(
        component,
        "enum class AvbdSoftPinTargetKind",
        "struct AvbdKinematicPin",
    )
    if "eKINEMATIC_RIGID" in pin_kinds:
        errors.append(
            "pin target kind still encodes actor class instead of "
            "prescribed ownership"
        )
    pin_owner = section(
        component,
        "PX_FORCE_INLINE AvbdSoftObjectiveOwner "
        "avbdGetPinObjectiveOwner(",
        "PX_FORCE_INLINE bool avbdIsPinPositionOwner(",
    )
    for token in (
        "ePRESCRIBED_RIGID",
        "eKINEMATIC_ATTACHMENT_POSITION_AL",
    ):
        if token not in pin_owner:
            errors.append(
                "prescribed rigid did not reuse Position AL owner "
                f"{token!r}"
            )

    prescribed_entry = section(
        scene,
        "struct PrescribedAttachmentEntry",
        "struct RigidActorFilterEntry",
    )
    for token in (
        "Dy::AvbdSoftPoint",
        "localPoint",
        "actorLocalTarget",
        "worldTarget",
        "previousWorldTarget",
        "alLambda",
    ):
        if token not in prescribed_entry:
            errors.append(
                "prescribed entry lost canonical state "
                f"{token!r}"
            )
    for forbidden in (
        "struct KinematicAttachmentEntry",
        "mStaticAttachments",
        "StaticAttachmentEntry",
    ):
        if forbidden in scene:
            errors.append(
                "prescribed attachment storage forked "
                f"{forbidden!r}"
            )

    target_compile = section(
        scene,
        "bool computePrescribedAttachmentWorldTarget(",
        "PxU32 addKinematicAttachment(",
    )
    for token in (
        "PxActorType::eRIGID_STATIC",
        "PxActorType::eRIGID_DYNAMIC",
        "getActor2World()",
        "getKinematicTarget(",
    ):
        if token not in target_compile:
            errors.append(
                "prescribed target compiler lost "
                f"{token!r}"
            )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_PRESCRIBED_RIGID_ATTACHMENT_"
                "SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_PRESCRIBED_RIGID_ATTACHMENT_"
            "SOURCE_GATE] status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_PRESCRIBED_RIGID_ATTACHMENT_SOURCE_GATE] "
        "status=PASS representation=weighted-point "
        "targets=kinematic,static owner=prescribed-position-al "
        "reaction=none duplicateStorage=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
