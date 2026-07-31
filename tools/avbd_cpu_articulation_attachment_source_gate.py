#!/usr/bin/env python3
"""Fail closed if CPU AVBD articulation attachments lose generalized ownership."""

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
    component = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
    )
    solver = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSolverJointPath.cpp"
    )
    dynamics_header = read(
        "physx/source/lowleveldynamics/src/DyAvbdDynamics.h"
    )
    dynamics = read(
        "physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp"
    )
    featherstone_header = read(
        "physx/source/lowleveldynamics/include/"
        "DyFeatherstoneArticulation.h"
    )
    featherstone = read(
        "physx/source/lowleveldynamics/src/"
        "DyFeatherstoneForwardDynamic.cpp"
    )
    sc_header = read(
        "physx/source/simulationcontroller/include/ScScene.h"
    )
    sc = read("physx/source/simulationcontroller/src/ScScene.cpp")
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

    required = {
        "typed attachment target": (
            component,
            "enum class AvbdSoftAttachmentTargetKind",
        ),
        "articulation target kind": (
            component,
            "eARTICULATION_LINK",
        ),
        "unique compiled owner": (
            component,
            "eARTICULATION_ATTACHMENT_POSITION_AL",
        ),
        "attachment owner compiler": (
            component,
            "avbdGetAttachmentObjectiveOwner(",
        ),
        "Scene independent entry": (
            sc,
            "struct ArticulationAttachmentEntry",
        ),
        "Scene independent storage": (
            sc,
            "mArticulationAttachments",
        ),
        "Scene articulation island lookup": (
            sc,
            "findArticulationBodyIndexInIsland(",
        ),
        "Surface Scene API": (
            sc_header + sc,
            "addAvbdCpuDeformableSurfaceArticulationAttachment(",
        ),
        "Volume Scene API": (
            sc_header + sc,
            "addAvbdCpuDeformableVolumeArticulationAttachment(",
        ),
        "Np typed detection": (
            np_header + np_attachment,
            "isCpuAvbdArticulationVertexAttachment()",
        ),
        "Np persistent route": (
            np_header + np_attachment,
            "CpuAvbdRoute::eARTICULATION_LINK",
        ),
        "factory articulation route": (
            factory,
            "cpuArticulationRigidVertex",
        ),
        "island articulation mapping": (
            dynamics_header + dynamics,
            "articulationForBody",
        ),
        "island link mapping": (
            dynamics_header + dynamics,
            "linkIndexForBody",
        ),
        "generalized response prep": (
            featherstone_header + featherstone,
            "prepareAvbdGeneralizedPositionResponse(",
        ),
        "Surface public gate": (
            surface,
            "surface-articulation-attachment",
        ),
        "Volume public gate": (
            volume,
            "scene-volume-articulation-attachment",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    scene_path = section(
        sc,
        "PxU32 addArticulationAttachment(",
        "PxU32 addRigidActorFilter(",
    )
    if not scene_path:
        errors.append("could not isolate articulation Scene path")
    for forbidden in (
        "mRigidAttachments",
        "addRigidAttachment(",
        "updateRigidAttachment(",
        "removeRigidAttachment(",
    ):
        if forbidden in scene_path:
            errors.append(
                "articulation Scene path aliases dynamic-rigid storage "
                f"{forbidden!r}"
            )

    generalized = section(
        solver,
        "if (articulationOwner)",
        "AvbdSoftRigidAttachmentCoupledStep step;",
    )
    if not generalized:
        errors.append("could not isolate articulation generalized primal")
    for token in (
        "getImpulseResponse(",
        "getImpulseSelfResponse(",
        "articulationPointInverseMass",
        "effectiveMass",
        "articulationForBody[linkBodyIndex] != articulation",
        "linkBody.position += linkPoseResponse.linear",
        "endpoint < objective.point.particleCount",
        "objective.point.particleIndices[endpoint]].position +=",
        "particleCorrections[endpoint]",
    ):
        if token not in generalized:
            errors.append(
                "articulation generalized primal lost "
                f"{token!r}"
            )
    for forbidden in (
        "avbdEvaluateSoftRigidAttachmentCoupledStep(",
        "pxcFsApplyImpulse",
        "applyImpulse(",
        "setLinearVelocity(",
        "setAngularVelocity(",
        "linearVelocity +=",
        "angularVelocity +=",
    ):
        if forbidden in generalized:
            errors.append(
                "articulation position owner regained a rigid/velocity "
                f"path {forbidden!r}"
            )

    response_prep = section(
        featherstone,
        "prepareAvbdGeneralizedPositionResponse()",
        "void FeatherstoneArticulation::computeArticulatedSpatialZ(",
    )
    if not response_prep:
        errors.append("could not isolate generalized response prep")
    for token in (
        "initializeCommonData();",
        "computeArticulatedSpatialInertia(",
        "computeArticulatedResponseMatrix(",
        "savedCfms",
    ):
        if token not in response_prep:
            errors.append(
                "generalized response prep lost "
                f"{token!r}"
            )
    for forbidden in (
        "computeUnconstrainedVelocities",
        "solveInternalConstraints",
        "applyImpulse",
        "setRootLinearVelocity",
        "setRootAngularVelocity",
    ):
        if forbidden in response_prep:
            errors.append(
                "generalized response prep entered a velocity path "
                f"{forbidden!r}"
            )

    if (
        component.count(
            "eARTICULATION_ATTACHMENT_POSITION_AL"
        )
        < 3
    ):
        errors.append(
            "articulation owner is not declared/compiled/validated"
        )
    if solver.count(
        "eARTICULATION_ATTACHMENT_POSITION_AL"
    ) < 4:
        errors.append(
            "articulation owner is not covered by "
            "warmstart/primal/dual/finalize ownership"
        )

    # The compiled attachment already owns a stable typed identity.  Scene
    # copyback must use that identity directly instead of maintaining
    # order-coupled shadow arrays and cursors beside the compiled IR.
    for forbidden in (
        "rigidAttachmentHandles",
        "articulationAttachmentHandles",
        "rigidAttachmentCursor",
        "articulationAttachmentCursor",
    ):
        if forbidden in sc:
            errors.append(
                "attachment state copyback retains order-coupled shadow "
                f"identity {forbidden!r}"
            )
    if "sourceAttachment.sourceHandle" not in sc:
        errors.append(
            "attachment state copyback lost compiled sourceHandle identity"
        )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_ARTICULATION_ATTACHMENT_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_ARTICULATION_ATTACHMENT_SOURCE_GATE] "
            "status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_ARTICULATION_ATTACHMENT_SOURCE_GATE] "
        "status=PASS owner=generalized-position-al "
        "response=articulation-tangent-space velocityImpulse=none "
        "stateIdentity=compiled-source-handle"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
