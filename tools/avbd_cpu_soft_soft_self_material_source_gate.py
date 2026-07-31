#!/usr/bin/env python3
"""Lock the final pre-performance CPU AVBD soft-body correctness slice."""

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
    soft = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
    )
    joint = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSolverJointPath.cpp"
    )
    scene = read("physx/source/simulationcontroller/src/ScScene.cpp")
    component_snippet = read(
        "physx/snippets/snippetsoftbodyavbd/"
        "SnippetSoftBodyAVBD.cpp"
    )
    surface_snippet = read(
        "physx/snippets/snippetdeformablesurfaceavbd/"
        "SnippetDeformableSurfaceAVBD.cpp"
    )
    component_runner = read(
        "tools/run_snippet_soft_body_avbd_headless.py"
    )
    surface_runner = read(
        "tools/run_snippet_deformable_surface_avbd_headless.py"
    )

    soft_soft = section(
        soft,
        "inline void avbdDetectSoftSoftOGC(",
        "inline void avbdDetectSelfCollisionOGC(",
    )
    if not soft_soft:
        errors.append("could not isolate soft-soft OGC detection")
    else:
        require_tokens(
            errors,
            "soft-soft swept owner",
            soft_soft,
            (
                "pairSpeculative",
                "particle.initialPosition",
                "bestEntryTime",
                "avbdRotatingPointEnterExpandedDeformingTriangleFace(",
                "avbdDeformingSegmentsEnterExpandedInteriors(",
                "geometry.depth = 0.0f",
                "continue;",
            ),
        )

    self_collision = section(
        soft,
        "inline void avbdDetectSelfCollisionOGC(",
        "PX_FORCE_INLINE void avbdTransferSoftContactState(",
    )
    if not self_collision:
        errors.append("could not isolate self-collision OGC detection")
    else:
        require_tokens(
            errors,
            "self swept owner",
            self_collision,
            (
                "sweepEnabled",
                "bestEntryTime",
                "avbdRotatingPointEnterExpandedDeformingTriangleFace(",
                "avbdDeformingSegmentsEnterExpandedInteriors(",
                "stabilizeEdgeNormal",
                "selfCollisionStressTolerance",
                "avbdComputeTetStressCoefficient(",
            ),
        )

    require_tokens(
        errors,
        "continuous deforming edge kernel",
        soft,
        (
            "avbdDeformingSegmentsEnterExpandedInteriors(",
            "iteration < 64",
        ),
    )
    require_tokens(
        errors,
        "volume material parity",
        soft + joint + scene,
        (
            "shapeGradients[4]",
            "avbdExtractCorotationalRotation(",
            "avbdEvaluateCorotationalForceHessianPrepared(",
            "coRotationalVolumeModel",
            "PxDeformableVolumeMaterialModel::",
            "eCO_ROTATIONAL",
        ),
    )
    if (
        soft.count("avbdEvaluateCorotationalForceHessianPrepared(") < 2
        or joint.count("avbdEvaluateCorotationalForceHessianPrepared(") < 1
    ):
        errors.append(
            "co-rotational evaluator is not dispatched by both solve paths"
        )

    require_tokens(
        errors,
        "surface bending damping",
        soft + joint + scene,
        (
            "avbdApplyBendingDamping(",
            "body.material.bendingDamping",
            "material->bendingDamping",
        ),
    )
    if (
        soft.count("avbdApplyBendingDamping(") < 2
        or joint.count("avbdApplyBendingDamping(") < 1
    ):
        errors.append(
            "bending damping is not finalized by both solve paths"
        )

    fallback = section(
        scene,
        "void stepComponentFallback(",
        "void syncHostInputs(",
    )
    if not fallback:
        errors.append("could not isolate component fallback")
    else:
        require_tokens(
            errors,
            "surface collision update/substep semantics",
            fallback,
            (
                "getNbCollisionPairUpdatesPerTimestep()",
                "getNbCollisionSubsteps()",
                "hasExplicitCollisionPairUpdates",
                "requestedRedetectionStages",
                "minimumContactIterations",
            ),
        )

    sync = section(scene, "void syncHostInputs(", "void writeBack(")
    if not sync:
        errors.append("could not isolate host input synchronization")
    else:
        require_tokens(
            errors,
            "compiled public material and stress inputs",
            sync,
            (
                "body.compiled.selfCollisionStressTolerance",
                "bodyCore.selfCollisionStressTolerance",
                "material->materialModel",
                "material->bendingDamping",
            ),
        )

    require_tokens(
        errors,
        "component runtime regressions",
        component_snippet,
        (
            "Test 34: Soft-Soft Swept OGC Features",
            "Test 35: Self Swept OGC Features",
            "Test 36: Deformable Material Semantics",
            "setMaterialModel(",
            "getMaterialModel()",
        ),
    )
    require_tokens(
        errors,
        "public Surface swept regressions",
        surface_snippet + surface_runner,
        (
            "surface-soft-soft-swept-ccd",
            "surface-self-collision-swept-ccd",
            "setNbCollisionPairUpdatesPerTimestep(4)",
            "setNbCollisionSubsteps(2)",
        ),
    )
    require_tokens(
        errors,
        "public Neo-Hookean Scene route",
        surface_snippet + read(
            "physx/snippets/snippetdeformablevolumeavbd/"
            "SnippetDeformableVolumeAVBD.cpp"
        ),
        (
            "volumeMaterial->setMaterialModel(",
            "PxDeformableVolumeMaterialModel::eNEO_HOOKEAN",
            "scene-volume-corotational",
            "PxDeformableVolumeMaterialModel::eCO_ROTATIONAL",
        ),
    )
    require_tokens(
        errors,
        "component runner inventory",
        component_runner,
        (
            "tuple(range(1, 37))",
            "choices=range(1, 37)",
        ),
    )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_SOFT_SOFT_SELF_MATERIAL_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_SOFT_SOFT_SELF_MATERIAL_SOURCE_GATE] status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_SOFT_SOFT_SELF_MATERIAL_SOURCE_GATE] status=PASS "
        "soft-soft=vf,ee self=vf,ee material=corot,neo "
        "bending-damping=enabled stress-filter=enabled "
        "surface-collision-controls=compiled"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
