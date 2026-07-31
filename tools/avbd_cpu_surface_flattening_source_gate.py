#!/usr/bin/env python3
"""Lock CPU AVBD surface flattening to prep-compiled bending targets."""

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
    scene = read("physx/source/simulationcontroller/src/ScScene.cpp")
    surface_flag = read("physx/include/PxDeformableSurfaceFlag.h")
    snippet = read(
        "physx/snippets/snippetdeformablesurfaceavbd/"
        "SnippetDeformableSurfaceAVBD.cpp"
    )
    runner = read(
        "tools/run_snippet_deformable_surface_avbd_headless.py"
    )

    required = {
        "public flattening flag": (
            surface_flag,
            "eENABLE_FLATTENING",
        ),
        "geometric rest-angle retention": (
            soft,
            "restShapeAngle",
        ),
        "compiled flattening state": (
            soft,
            "flatteningEnabled",
        ),
        "single prep compiler": (
            soft,
            "compileBendingRestAngles(",
        ),
        "creation-time compilation": (
            scene,
            "PxDeformableSurfaceFlag::eENABLE_FLATTENING",
        ),
        "runtime flag refresh": (
            scene,
            "refreshSurfaceFlattening(",
        ),
        "headless behavior case": (
            snippet + runner,
            "surface-flattening",
        ),
        "negative control": (
            snippet + runner,
            "flatteningControlHeld",
        ),
        "runtime retarget": (
            snippet + runner,
            "flatteningRetargetObserved",
        ),
    }
    for description, (text, token) in required.items():
        if token not in text:
            errors.append(f"{description} lost {token!r}")

    bending_kernel = section(
        soft,
        "PX_FORCE_INLINE void avbdEvaluateBendingForceHessian(",
        "PX_FORCE_INLINE AvbdSoftContactRowForces",
    )
    if not bending_kernel:
        errors.append("could not isolate bending kernel")
    else:
        if "be.restAngle" not in bending_kernel:
            errors.append("bending kernel lost compiled rest-angle input")
        for forbidden in (
            "surfaceFlags",
            "eENABLE_FLATTENING",
            "flatteningEnabled",
        ):
            if forbidden in bending_kernel:
                errors.append(
                    "bending kernel regained runtime flag branch "
                    f"{forbidden!r}"
                )

    refresh = section(
        scene,
        "void refreshSurfaceFlattening(",
        "void syncHostInputs(",
    )
    if not refresh:
        errors.append("could not isolate surface flattening refresh")
    else:
        if "compileBendingRestAngles(" not in refresh:
            errors.append(
                "runtime surface flag is not recompiled before solving"
            )
        for forbidden in (
            "applyImpulse",
            "velocityImpulse",
            "surfaceFlags &=",
        ):
            if forbidden in refresh:
                errors.append(
                    "flattening refresh regained forbidden solve path "
                    f"{forbidden!r}"
                )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_SURFACE_FLATTENING_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_SURFACE_FLATTENING_SOURCE_GATE] status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_SURFACE_FLATTENING_SOURCE_GATE] "
        "status=PASS target=compiled-rest-angle "
        "runtimeBranch=none velocityImpulse=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
