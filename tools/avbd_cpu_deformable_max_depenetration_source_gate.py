#!/usr/bin/env python3
"""Lock CPU AVBD max depenetration velocity to prepared contact rows."""

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


def require_tokens(
    errors: list[str],
    description: str,
    text: str,
    tokens: tuple[str, ...],
) -> None:
    for token in tokens:
        if token not in text:
            errors.append(f"{description} lost {token!r}")


def main() -> int:
    errors: list[str] = []
    public_api = read("physx/include/PxDeformableBody.h")
    body_core = read(
        "physx/source/lowleveldynamics/include/"
        "DyDeformableBodyCore.h"
    )
    np_surface = read(
        "physx/source/physx/src/NpDeformableSurface.cpp"
    )
    np_volume = read(
        "physx/source/physx/src/NpDeformableVolume.cpp"
    )
    soft = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
    )
    joint = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSolverJointPath.cpp"
    )
    scene = read(
        "physx/source/simulationcontroller/src/ScScene.cpp"
    )
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

    require_tokens(
        errors,
        "public API",
        public_api,
        (
            "setMaxDepenetrationVelocity",
            "getMaxDepenetrationVelocity",
        ),
    )
    require_tokens(
        errors,
        "deformable core",
        body_core,
        (
            "PxReal",
            "maxPenetrationBias",
            "maxPenetrationBias(-1e32f)",
        ),
    )
    for description, np_source in (
        ("Surface public adapter", np_surface),
        ("Volume public adapter", np_volume),
    ):
        require_tokens(
            errors,
            description,
            np_source,
            (
                "mCore.setMaxPenetrationBias(-v)",
                "return -mCore.getMaxPenetrationBias()",
            ),
        )

    compiled = section(
        soft,
        "struct AvbdSoftBodyCompiledData",
        "void compileBendingRestAngles",
    )
    require_tokens(
        errors,
        "compiled actor policy",
        compiled,
        (
            "PxReal maxDepenetrationVelocity;",
            "maxDepenetrationVelocity(PX_MAX_F32)",
        ),
    )
    sync = section(scene, "void syncHostInputs(", "void writeBack(")
    require_tokens(
        errors,
        "Scene host synchronization",
        sync,
        (
            "body.compiled.maxDepenetrationVelocity =",
            "PxMax(-bodyCore.maxPenetrationBias, 0.0f)",
        ),
    )

    state = section(
        soft,
        "struct AvbdSoftContactAugmentedState",
        "struct AvbdSoftContact",
    )
    require_tokens(
        errors,
        "prepared contact state",
        state,
        (
            "depenetrationConstraintOffset",
            "depenetrationLimitInitialized",
        ),
    )
    initializer = section(
        soft,
        "avbdInitializeSoftContactDepenetrationLimitAtSurfacePoint(",
        "PX_FORCE_INLINE void avbdInitializeSoftContactDepenetrationLimits(",
    )
    require_tokens(
        errors,
        "frame-start contact target",
        initializer,
        (
            "maxDepenetrationVelocity",
            "maxRecoveryDistance",
            "initialConstraint + maxRecoveryDistance",
            "avbdGetSoftContactInitialQueryPoint",
            "state.alLambda = 0.0f",
        ),
    )
    require_tokens(
        errors,
        "contact constraint evaluator",
        soft,
        (
            "avbdEvaluateSoftContactNormalConstraint(",
            "state.depenetrationConstraintOffset",
        ),
    )
    if soft.count(
        "state.depenetrationConstraintOffset;"
    ) < 2:
        errors.append(
            "primal force and dual update do not both consume "
            "the shifted contact target"
        )

    transfer = section(
        soft,
        "PX_FORCE_INLINE void avbdTransferSoftContactState(",
        "inline void avbdDetectAllOGCContacts(",
    )
    require_tokens(
        errors,
        "same-frame redetection transfer",
        transfer,
        (
            "bestState.depenetrationConstraintOffset",
            "bestState.depenetrationLimitInitialized",
        ),
    )

    fallback = section(
        soft,
        "inline void avbdStepSoftBodies(",
        "} // namespace Dy",
    )
    require_tokens(
        errors,
        "fallback component path",
        fallback,
        (
            "avbdResetSoftContactDepenetrationLimits(",
            "avbdInitializeSoftContactDepenetrationLimits(",
            "only contacts born at this redetection are initialized",
        ),
    )
    require_tokens(
        errors,
        "native joint path",
        joint,
        (
            "avbdResetSoftContactDepenetrationLimits(",
            "avbdInitializeSoftContactDepenetrationLimitAtSurfacePoint(",
            "rigidBody.prevPosition",
            "geometry.rigidLocalPoint",
        ),
    )

    finalizer = section(
        scene,
        "void finalizeDeformableMotionControls(",
        "void updateSleepStates(",
    )
    if "maxPenetrationBias" in finalizer or (
        "maxDepenetrationVelocity" in finalizer
    ):
        errors.append(
            "generic motion finalizer stole contact-bias ownership"
        )
    for forbidden in (
        "applyImpulse",
        "remainingDepenetrationDistance",
    ):
        if forbidden in soft + joint:
            errors.append(
                "contact bias gained forbidden implementation "
                f"{forbidden!r}"
            )

    require_tokens(
        errors,
        "Surface behavior gate",
        surface + surface_runner,
        (
            "surface-max-depenetration-velocity",
            "depenetrationFirstStepBounded",
            "depenetrationControlSeparated",
            "depenetrationGradualRecovery",
        ),
    )
    require_tokens(
        errors,
        "Volume behavior gate",
        volume + volume_runner,
        (
            "scene-volume-max-depenetration-velocity",
            "SCENE_MAX_DEPENETRATION_VELOCITY_GATED",
            "depenetrationLimitedFirstStepRise",
        ),
    )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_DEFORMABLE_MAX_DEPENETRATION_"
                f"SOURCE_GATE_ERROR] {error}"
            )
        print(
            "[AVBD_CPU_DEFORMABLE_MAX_DEPENETRATION_SOURCE_GATE] "
            "status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_DEFORMABLE_MAX_DEPENETRATION_SOURCE_GATE] "
        "status=PASS owner=prepared-contact "
        "frameAnchor=initial-relative-gap "
        "paths=native+fallback velocityImpulse=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
