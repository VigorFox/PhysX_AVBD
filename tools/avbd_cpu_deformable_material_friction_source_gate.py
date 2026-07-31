#!/usr/bin/env python3
"""Lock CPU AVBD deformable friction to material-owned contact prep."""

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
    public_api = read("physx/include/PxDeformableMaterial.h")
    soft = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
    )
    scene = read("physx/source/simulationcontroller/src/ScScene.cpp")
    snippet = read(
        "physx/snippets/snippetdeformablesurfaceavbd/"
        "SnippetDeformableSurfaceAVBD.cpp"
    )
    runner = read(
        "tools/run_snippet_deformable_surface_avbd_headless.py"
    )

    require_tokens(
        errors,
        "public deformable material API",
        public_api,
        ("setDynamicFriction", "getDynamicFriction"),
    )
    require_tokens(
        errors,
        "behavior fixture",
        snippet + runner,
        (
            "surface-material-friction",
            "materialFrictionLowApplied",
            "materialFrictionHighApplied",
            "materialFrictionResponseObserved",
            "materialFrictionLowDisplacement",
            "materialFrictionHighDisplacement",
        ),
    )
    require_tokens(
        errors,
        "low/high material controls",
        snippet,
        (
            "surfaceMaterial->setDynamicFriction(0.0f)",
            "surfaceMaterial->setDynamicFriction(2.0f)",
            "physics->createMaterial(0.0f, 0.0f, 0.0f)",
        ),
    )

    material = section(
        soft,
        "struct AvbdSoftBodyMaterialData",
        "struct AvbdSoftBodyRuntimeState",
    )
    require_tokens(
        errors,
        "compiled material",
        material,
        ("PxReal dynamicFriction;", "dynamicFriction(0.5f)"),
    )

    combine = section(
        soft,
        "PX_FORCE_INLINE PxReal "
        "avbdCombineDeformableRigidFriction(",
        "// =============================================================================\n"
        "// VBD Force/Hessian evaluators",
    )
    require_tokens(
        errors,
        "rigid/deformable combine policy",
        combine,
        (
            "PxCombineMode::eAVERAGE",
            "PxCombineMode::eMIN",
            "PxCombineMode::eMULTIPLY",
            "PxCombineMode::eMAX",
        ),
    )

    world = section(
        soft,
        "inline void avbdDetectSoftWorldPlaneContacts(",
        "inline void avbdDetectSoftGroundContacts(",
    )
    rigid = section(
        soft,
        "inline void avbdDetectSoftRigidSDF(",
        "inline void avbdDetectSoftSoftOGC(",
    )
    soft_soft = section(
        soft,
        "inline void avbdDetectSoftSoftOGC(",
        "inline void avbdBuildSelfCollisionAdjacency(",
    )
    self_collision = section(
        soft,
        "inline void avbdDetectSelfCollisionOGC(",
        "PX_FORCE_INLINE void avbdTransferSoftContactState(",
    )
    require_tokens(
        errors,
        "world contact prep",
        world,
        (
            "sourceBody->material.dynamicFriction",
            "plane.frictionCombineMode",
            "avbdCombineDeformableRigidFriction(",
        ),
    )
    require_tokens(
        errors,
        "rigid contact prep",
        rigid,
        (
            "sourceBody->material.dynamicFriction",
            "box.frictionCombineMode",
            "avbdCombineDeformableRigidFriction(",
        ),
    )
    require_tokens(
        errors,
        "soft-soft contact prep",
        soft_soft,
        (
            "bodyA.material.dynamicFriction",
            "bodyB.material.dynamicFriction",
            "geometry.friction = pairFriction;",
        ),
    )
    require_tokens(
        errors,
        "self-collision contact prep",
        self_collision,
        (
            "sb.material.dynamicFriction",
            "geometry.friction =",
        ),
    )
    for description, contact_section in (
        ("world", world),
        ("rigid", rigid),
        ("soft-soft", soft_soft),
        ("self-collision", self_collision),
    ):
        for forbidden in (
            "bodyFlags",
            "objectiveFlags",
            "applyImpulse",
            "finalizeDeformableMotionControls",
        ):
            if forbidden in contact_section:
                errors.append(
                    f"{description} contact prep gained forbidden "
                    f"friction ownership {forbidden!r}"
                )
    if "params.friction" in soft_soft:
        errors.append(
            "soft-soft contact prep regained global friction ownership"
        )
    if "params.friction" in self_collision:
        errors.append(
            "self-collision prep regained global friction ownership"
        )

    create = section(
        soft,
        "inline PxU32 avbdCreateSoftBody(",
        "typedef void (*AvbdContactRedetectFn)",
    )
    require_tokens(
        errors,
        "soft-body creation",
        create,
        (
            "PxReal dynamicFriction = 0.5f",
            "sb.material.dynamicFriction =",
        ),
    )

    sync = section(scene, "void syncHostInputs(", "void writeBack(")
    require_tokens(
        errors,
        "runtime material synchronization",
        sync,
        (
            "body.material.dynamicFriction = material",
            "PxMax(material->dynamicFriction, 0.0f)",
        ),
    )
    if sync.count("body.material.dynamicFriction = material") != 2:
        errors.append(
            "Volume and Surface do not both refresh dynamic friction"
        )
    if scene.count(
        "material ? material->dynamicFriction : 0.5f"
    ) != 3:
        errors.append(
            "Volume add, Surface add, and Surface rest rebuild do not "
            "all compile dynamic friction"
        )
    require_tokens(
        errors,
        "rigid material compilation",
        scene,
        (
            "getStaticFrictionCombineMode(",
            "material->getFrictionCombineMode()",
            "plane.frictionCombineMode =",
            "box.frictionCombineMode =",
        ),
    )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_DEFORMABLE_MATERIAL_FRICTION_"
                f"SOURCE_GATE_ERROR] {error}"
            )
        print(
            "[AVBD_CPU_DEFORMABLE_MATERIAL_FRICTION_SOURCE_GATE] "
            "status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_DEFORMABLE_MATERIAL_FRICTION_SOURCE_GATE] "
        "status=PASS owner=contact-prep material=per-body "
        "soft-soft=average rigid=combine-mode "
        "velocity-finalizer=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
