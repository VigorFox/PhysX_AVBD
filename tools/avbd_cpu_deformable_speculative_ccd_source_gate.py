#!/usr/bin/env python3
"""Lock CPU AVBD speculative rigid-box CCD to the public body flag."""

from __future__ import annotations

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


def require_all(
    errors: list[str],
    scope: str,
    text: str,
    fragments: tuple[str, ...],
) -> None:
    for fragment in fragments:
        if fragment not in text:
            errors.append(f"{scope} lost {fragment!r}")


def main() -> int:
    errors: list[str] = []
    public_flags = read("physx/include/PxDeformableBodyFlag.h")
    soft = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
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

    require_all(
        errors,
        "public API",
        public_flags,
        (
            "eENABLE_SPECULATIVE_CCD = 1 << 1",
            "Enables support for speculative contact generation",
        ),
    )

    compiled = section(
        soft,
        "struct AvbdSoftBodyCompiledData",
        "void compileBendingRestAngles",
    )
    require_all(
        errors,
        "compiled actor policy",
        compiled,
        (
            "bool speculativeCCDEnabled;",
            "speculativeCCDEnabled(false)",
        ),
    )

    sync = section(scene, "void syncHostInputs(", "void writeBack(")
    require_all(
        errors,
        "Scene flag synchronization",
        sync,
        (
            "body.compiled.speculativeCCDEnabled =",
            "bodyCore.bodyFlags.isSet(",
            "eENABLE_SPECULATIVE_CCD",
        ),
    )

    swept = section(
        soft,
        "inline void avbdDetectSoftRigidSweptSDF(",
        "inline void avbdDetectSoftRigidOGCFeatures(",
    )
    require_all(
        errors,
        "swept rigid-box query",
        swept,
        (
            "particle.predictedPosition",
            "sourceBody->compiled.speculativeCCDEnabled",
            "avbdSegmentEnterExpandedBox(",
            "avbdAppendPreparedSoftContact(",
        ),
    )
    if (
        "if(sourceBody &&\n"
        "\t\t\t!sourceBody->compiled.speculativeCCDEnabled)"
        not in swept
    ):
        errors.append(
            "swept rigid-box query is no longer fail-closed by actor flag"
        )

    require_all(
        errors,
        "Surface positive/negative behavior gate",
        surface + surface_runner,
        (
            "surface-speculative-ccd",
            "speculativeCcdFlagApplied",
            "speculativeCcdPreventedTunneling",
            "speculativeCcdNegativeControlTunneled",
            "speculativeCcdPositiveMinY",
            "speculativeCcdNegativeMaxY",
        ),
    )
    require_all(
        errors,
        "Volume positive/negative behavior gate",
        volume + volume_runner,
        (
            "scene-volume-speculative-ccd",
            "SCENE_SPECULATIVE_CCD_GATED",
            "speculativeCcdFlagApplied",
            "speculativeCcdPreventedTunneling",
            "speculativeCcdNegativeControlTunneled",
            "speculativeCcdPositiveMinY",
            "speculativeCcdNegativeMaxY",
        ),
    )

    if errors:
        print("AVBD_CPU_DEFORMABLE_SPECULATIVE_CCD_SOURCE_GATE=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_SPECULATIVE_CCD_SOURCE_GATE=PASS "
        "geometry=rigid-box actors=surface+volume "
        "negativeControl=discrete"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
