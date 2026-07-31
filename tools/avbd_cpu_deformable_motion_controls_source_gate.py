#!/usr/bin/env python3
"""Lock CPU AVBD deformable motion controls to one actor-level policy."""

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
    public_api = read("physx/include/PxDeformableBody.h")
    body_core = read(
        "physx/source/lowleveldynamics/include/"
        "DyDeformableBodyCore.h"
    )
    scene = read("physx/source/simulationcontroller/src/ScScene.cpp")
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

    required = {
        "public max velocity": (public_api, "setMaxLinearVelocity"),
        "public settling threshold": (
            public_api,
            "setSettlingThreshold",
        ),
        "public settling damping": (
            public_api,
            "setSettlingDamping",
        ),
        "public sleep threshold": (public_api, "setSleepThreshold"),
        "core max velocity": (body_core, "maxLinearVelocity"),
        "core settling threshold": (body_core, "settlingThreshold"),
        "core settling damping": (body_core, "settlingDamping"),
        "preintegration policy": (
            scene,
            "applyDeformablePreintegrationControls(",
        ),
        "single finalization policy": (
            scene,
            "finalizeDeformableMotionControls(",
        ),
        "surface behavior case": (
            surface + surface_runner,
            "surface-motion-controls",
        ),
        "volume behavior case": (
            volume + volume_runner,
            "scene-volume-motion-controls",
        ),
        "negative control": (
            surface + volume + surface_runner + volume_runner,
            "motionControlStayedAwake",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    preintegration = section(
        scene,
        "void applyDeformablePreintegrationControls(",
        "void syncHostInputs(",
    )
    if not preintegration:
        errors.append("could not isolate preintegration policy")
    else:
        for token in (
            "maxLinearVelocity",
            "particle.velocity",
            "particle.invMass",
            "particle.prevVelocity",
        ):
            if token not in preintegration:
                errors.append(
                    f"preintegration policy lost {token!r}"
                )
        for forbidden in (
            "particle.position =",
            "predictedPosition =",
            "applyImpulse",
            "maxPenetrationBias",
            "objectiveFlags",
        ):
            if forbidden in preintegration:
                errors.append(
                    "preintegration policy gained forbidden ownership "
                    f"{forbidden!r}"
                )

    finalization = section(
        scene,
        "void finalizeDeformableMotionControls(",
        "void updateSleepStates(",
    )
    if not finalization:
        errors.append("could not isolate motion finalization policy")
    else:
        for token in (
            "settlingThreshold",
            "settlingDamping",
            "particle.velocity *=",
            "particle.invMass <= 0.0f",
        ):
            if token not in finalization:
                errors.append(
                    f"motion finalization lost {token!r}"
                )
        for forbidden in (
            "particle.position =",
            "applyImpulse",
            "maxPenetrationBias",
            "objectiveFlags",
        ):
            if forbidden in finalization:
                errors.append(
                    "motion finalization gained forbidden ownership "
                    f"{forbidden!r}"
                )

    step = section(scene, "\t\tvoid step(", "\n\tprivate:")
    if not step:
        errors.append("could not isolate CPU deformable Scene step")
    elif step.count("finalizeDeformableMotionControls(dt);") != 2:
        errors.append(
            "main/fallback paths do not share exactly two finalizer calls"
        )

    sync = section(
        scene,
        "void syncHostInputs(",
        "void writeBack(",
    )
    if not sync:
        errors.append("could not isolate host-input synchronization")
    elif sync.count(
        "applyDeformablePreintegrationControls(entry);"
    ) != 1:
        errors.append(
            "host-input synchronization lost its single "
            "preintegration policy call"
        )

    if scene.count("sleepThreshold * sleepThreshold") != 3:
        errors.append(
            "sleep threshold is not squared at both actor adds "
            "and runtime sleep update"
        )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_DEFORMABLE_MOTION_CONTROLS_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_DEFORMABLE_MOTION_CONTROLS_SOURCE_GATE] "
            "status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_DEFORMABLE_MOTION_CONTROLS_SOURCE_GATE] "
        "status=PASS preintegration=max-linear-velocity "
        "finalization=settling sleepUnits=linear-squared "
        "positionClamp=none velocityImpulse=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
