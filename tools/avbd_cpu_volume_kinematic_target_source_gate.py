#!/usr/bin/env python3
"""Lock CPU AVBD deformable-volume kinematic targets to Position AL."""

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
    sc = read("physx/source/simulationcontroller/src/ScScene.cpp")
    volume_api = read("physx/include/PxDeformableVolume.h")
    volume_flag = read("physx/include/PxDeformableVolumeFlag.h")
    body_flag = read("physx/include/PxDeformableBodyFlag.h")
    np_volume = read("physx/source/physx/src/NpDeformableVolume.cpp")
    np_scene = read("physx/source/physx/src/NpScene.cpp")
    snippet = read(
        "physx/snippets/snippetdeformablevolumeavbd/"
        "SnippetDeformableVolumeAVBD.cpp"
    )
    runner = read(
        "tools/run_snippet_deformable_volume_avbd_headless.py"
    )

    required = {
        "host target API": (
            volume_api + np_volume,
            "setKinematicTargetBufferH(",
        ),
        "full kinematic flag": (
            body_flag,
            "eKINEMATIC",
        ),
        "partial kinematic flag": (
            volume_flag,
            "ePARTIALLY_KINEMATIC",
        ),
        "typed target": (
            soft,
            "eDEFORMABLE_KINEMATIC",
        ),
        "single Position-AL owner": (
            soft,
            "eDEFORMABLE_KINEMATIC_POSITION_AL",
        ),
        "target pin builder": (
            sc,
            "appendVolumeKinematicTargetPins(",
        ),
        "per-frame target refresh": (
            sc,
            "refreshVolumeKinematicTargets()",
        ),
        "persistent host target read": (
            sc,
            "core.kinematicTarget",
        ),
        "full target selection": (
            sc,
            "PxDeformableBodyFlag::eKINEMATIC",
        ),
        "partial target selection": (
            sc,
            "PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC",
        ),
        "partial w contract": (
            sc,
            "target.w == 0.0f",
        ),
        "vertex source identity": (
            sc,
            "pin.sourceHandle = localIndex",
        ),
        "persistent target AL state": (
            joint,
            "eDEFORMABLE_KINEMATIC_POSITION_AL",
        ),
        "sleep residual ownership": (
            sc,
            "kinematicTargetResidualSquared",
        ),
        "full headless case": (
            snippet + runner,
            "scene-volume-full-kinematic-target",
        ),
        "partial headless case": (
            snippet + runner,
            "scene-volume-partial-kinematic-target",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    refresh = section(
        sc,
        "void refreshVolumeKinematicTargets()",
        "void refreshPrescribedAttachmentTargets()",
    )
    if not refresh:
        errors.append("could not isolate volume target refresh")
    for forbidden in (
        "applyImpulse",
        "velocityImpulse",
        "mRigidAttachments",
        "ensureNativeIslandEdge(",
    ):
        if forbidden in refresh:
            errors.append(
                "volume kinematic target regained non-prescribed path "
                f"{forbidden!r}"
            )

    pin_owner = section(
        soft,
        "PX_FORCE_INLINE AvbdSoftObjectiveOwner "
        "avbdGetPinObjectiveOwner(",
        "PX_FORCE_INLINE bool avbdIsPinPositionOwner(",
    )
    if (
        "case AvbdSoftPinTargetKind::eDEFORMABLE_KINEMATIC:"
        not in pin_owner
        or "eDEFORMABLE_KINEMATIC_POSITION_AL" not in pin_owner
    ):
        errors.append(
            "deformable target is not compiled to its unique Position-AL "
            "owner"
        )

    warmstart = section(
        joint,
        "// Soft body AVBD warmstart.",
        "for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)",
    )
    if (
        "eDEFORMABLE_KINEMATIC_POSITION_AL" not in warmstart
        or "1.0f, 1.0f, 1.0e8f" not in warmstart
    ):
        errors.append(
            "persistent deformable target multiplier/penalty retention "
            "was lost"
        )

    if (
        "CPU AVBD deformable-volume kinematic targets are not enabled"
        in np_scene
    ):
        errors.append(
            "NpScene still rejects CPU AVBD volume kinematic targets"
        )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_VOLUME_KINEMATIC_TARGET_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_VOLUME_KINEMATIC_TARGET_SOURCE_GATE] "
            "status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_VOLUME_KINEMATIC_TARGET_SOURCE_GATE] "
        "status=PASS owner=position-al targets=full,partial "
        "response=prescribed-particle velocityImpulse=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
