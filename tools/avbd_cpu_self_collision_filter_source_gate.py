#!/usr/bin/env python3
"""Lock CPU AVBD self-collision filtering to rest-space prep data."""

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

    required = {
        "public setter": (
            public_api,
            "setSelfCollisionFilterDistance",
        ),
        "public getter": (
            public_api,
            "getSelfCollisionFilterDistance",
        ),
        "shared body-core value": (
            body_core,
            "selfCollisionFilterDistance",
        ),
        "compiled rest positions": (
            soft,
            "selfCollisionRestPositions",
        ),
        "compiled filter distance": (
            soft,
            "selfCollisionFilterDistance",
        ),
        "filter behavior case": (
            snippet + runner,
            "surface-self-collision-filter",
        ),
        "zero-distance control": (
            snippet,
            "selfCollisionFilterCase ? 0.1f : 0.0f",
        ),
        "behavior result": (
            snippet + runner,
            "selfCollisionFilterExcludedPair",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    creation = section(
        soft,
        "inline PxU32 avbdCreateSoftBody(",
        "typedef void (*AvbdContactRedetectFn)",
    )
    if not creation:
        errors.append("could not isolate soft-body creation")
    else:
        for token in (
            "selfCollisionRestPositions.resize(numVertices)",
            "selfCollisionRestPositions[i] = vertices[i]",
            "selfCollisionFilterDistance =",
        ):
            if token not in creation:
                errors.append(
                    f"soft-body creation lost compiled input {token!r}"
                )

    detection = section(
        soft,
        "inline void avbdDetectSelfCollisionOGC(",
        "inline void avbdBuildAllSelfCollisionAdjacencies(",
    )
    if not detection:
        errors.append("could not isolate self-collision detection")
    else:
        for token in (
            "sb.compiled.selfCollisionFilterDistance",
            "sb.compiled.selfCollisionRestPositions",
            "avbdClosestPointOnTriangleOGC(",
            "restClosest.distance <= filterDistance",
        ):
            if token not in detection:
                errors.append(
                    f"self-collision prep lost rest filter {token!r}"
                )
        for forbidden in (
            "bodyFlags",
            "eDISABLE_SELF_COLLISION",
            "filterDistance = params.contactRadius",
        ):
            if forbidden in detection:
                errors.append(
                    "self-collision prep regained non-compiled policy "
                    f"{forbidden!r}"
                )

    sync = section(scene, "void syncHostInputs(", "void writeBack(")
    if not sync:
        errors.append("could not isolate host-input synchronization")
    elif (
        "body.compiled.selfCollisionFilterDistance =" not in sync
        or "bodyCore.selfCollisionFilterDistance" not in sync
    ):
        errors.append(
            "runtime public value is not compiled during host prep"
        )

    if scene.count("getSelfCollisionFilterDistance()") != 3:
        errors.append(
            "Volume add, Surface add, and Surface rest rebuild do not "
            "all compile the public filter distance"
        )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_SELF_COLLISION_FILTER_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_SELF_COLLISION_FILTER_SOURCE_GATE] status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_SELF_COLLISION_FILTER_SOURCE_GATE] "
        "status=PASS space=rest owner=contact-prep "
        "policy=per-body global-radius-leak=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
