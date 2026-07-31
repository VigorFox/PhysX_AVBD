#!/usr/bin/env python3
"""Lock CPU AVBD speculative plane contacts to the public body flag."""

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
    soft = read(
        "physx/source/lowleveldynamics/src/"
        "DyAvbdSoftBodyComponent.h"
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
    component = read(
        "physx/snippets/snippetsoftbodyavbd/"
        "SnippetSoftBodyAVBD.cpp"
    )
    component_runner = read(
        "tools/run_snippet_soft_body_avbd_headless.py"
    )

    plane_query = section(
        soft,
        "inline void avbdDetectSoftWorldPlaneContacts(",
        "inline void avbdDetectSoftGroundContacts(",
    )
    require_all(
        errors,
        "world-plane contact query",
        plane_query,
        (
            "const PxReal distance =",
            "if(distance >= margin)",
            "!sourceBody->compiled.speculativeCCDEnabled",
            "particles[i].predictedPosition.isFinite()",
            "const PxReal predictedDistance =",
            "if(predictedDistance >= margin)",
            "speculativeCandidate = true;",
            "speculativeCandidate ? 1e6f : 1e4f",
            "geometry.source = AvbdSoftContactSource(",
            "AvbdSoftContactSource::eGROUND",
            "geometry.velocityOwner =",
            "AvbdVelocityObjectiveOwner::PositionAL",
            "avbdAppendPreparedSoftContact(",
        ),
    )
    if (
        "if(!sourceBody ||\n"
        "\t\t\t\t\t!sourceBody->compiled.speculativeCCDEnabled"
        not in plane_query
    ):
        errors.append(
            "speculative plane query is no longer fail-closed "
            "without an opted-in source body"
        )

    solve = section(
        soft,
        "// Stage 1: prediction",
        "avbdInitializeSoftContactDepenetrationLimits(",
    )
    require_all(
        errors,
        "same-timestep contact refresh",
        solve,
        (
            "particles[i].computePrediction(dt, gravity);",
            "speculative plane",
            "redetectFn(",
            "compileVelocityObjectives(contacts, numContacts);",
        ),
    )

    require_all(
        errors,
        "Surface plane positive/negative gate",
        surface + surface_runner,
        (
            "surface-plane-speculative-ccd",
            "PxCreatePlane(",
            "speculativeCcdFlagApplied",
            "speculativeCcdPreventedTunneling",
            "planeCase ? 0.49f : 0.54f",
        ),
    )
    require_all(
        errors,
        "Volume plane positive/negative gate",
        volume + volume_runner,
        (
            "scene-volume-plane-speculative-ccd",
            "isSceneCpuVolumeSpeculativeCcdCase(",
            "PxCreatePlane(",
            "speculativeCcdFlagApplied",
            "speculativeCcdPreventedTunneling",
            "getSceneCpuVolumeCollisionMinY(",
            "getSceneCpuVolumeCollisionMaxY(",
            "getPositionInvMassBufferH()",
            "planeSpeculativeCcdCase",
            "? 0.49f : 0.50f",
        ),
    )
    volume_runtime = section(
        volume,
        "else if(speculativeCcdCase &&",
        "if(rigidAttachmentCase &&",
    )
    positive_update = volume_runtime.find(
        "gMetrics.speculativeCcdPositiveMinY ="
    )
    negative_window = volume_runtime.find(
        "if(gMetrics.completedFrames <= 3)"
    )
    if (
        positive_update < 0
        or negative_window < 0
        or positive_update > negative_window
    ):
        errors.append(
            "Volume collision-boundary minimum is no longer accumulated "
            "for the full requested horizon"
        )
    require_all(
        errors,
        "component plane positive/negative active-set gate",
        component + component_runner,
        (
            "testPlaneSpeculativeCcdActiveSet()",
            "bodies[0].compiled.speculativeCCDEnabled = true;",
            "bodies[1].compiled.speculativeCCDEnabled = false;",
            "avbdDetectSoftWorldPlaneContacts(",
            "positivePrepared == 1",
            "negativePrepared == 0",
            "tuple(range(1, 37))",
            "choices=range(1, 37)",
        ),
    )
    if (
        "Dy::" in surface
        or "Dy::" in volume
        or "probePlaneSpeculativeCcdActiveSet(" in surface
        or "probePlaneSpeculativeCcdActiveSet(" in volume
    ):
        errors.append(
            "public Surface/Volume plane gates bypass the public Scene API"
        )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_PLANE_SPECULATIVE_CCD_SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_PLANE_SPECULATIVE_CCD_SOURCE_GATE=PASS "
        "geometry=plane actors=surface+volume "
        "negativeControl=component-active-set"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
