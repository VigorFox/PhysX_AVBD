#!/usr/bin/env python3
"""Lock E50 deforming-soft reverse swept OGC ownership and public gates."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require_all(
    errors: list[str],
    scope: str,
    text: str,
    fragments: tuple[str, ...],
) -> None:
    for fragment in fragments:
        if fragment not in text:
            errors.append(f"{scope} lost {fragment!r}")


def require_count(
    errors: list[str],
    scope: str,
    text: str,
    fragment: str,
    minimum: int,
) -> None:
    count = text.count(fragment)
    if count < minimum:
        errors.append(
            f"{scope} has {count} copies of {fragment!r}, "
            f"expected at least {minimum}"
        )


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

    require_all(
        errors,
        "deforming reverse swept kernels",
        soft,
        (
            "avbdLinearPointEnterExpandedDeformingTriangleNonVertex(",
            "avbdRotatingSegmentEnterExpandedDeformingTriangleNonVertex(",
            "avbdRotatingSegmentEnterExpandedDeformingSegmentInteriors(",
            "avbdRotatingPointEnterExpandedDeformingTriangleFace(",
            "linearly deforming soft edge",
            "linearly deforming soft face",
            "const PxReal nextTime = time + gap / speed;",
            "for(PxU32 iteration = 0; iteration < 64; ++iteration)",
            "entry.entryTime",
            "entry.barycentric",
            "entry.rigidWeight1",
        ),
    )
    for fragment, minimum in (
        (
            "avbdLinearPointEnterExpandedDeformingTriangleNonVertex(",
            2,
        ),
        (
            "avbdRotatingSegmentEnterExpandedDeformingTriangleNonVertex(",
            2,
        ),
        (
            "avbdRotatingSegmentEnterExpandedDeformingSegmentInteriors(",
            3,
        ),
        (
            "avbdRotatingPointEnterExpandedDeformingTriangleFace(",
            3,
        ),
        ("particles[v0].initialPosition", 4),
        ("particles[v1].initialPosition", 4),
        ("particles[v2].initialPosition", 4),
    ):
        require_count(
            errors,
            "deforming reverse swept integration",
            soft,
            fragment,
            minimum,
        )

    require_all(
        errors,
        "Surface deforming reverse swept public gates",
        surface + surface_runner,
        (
            "surface-deforming-sphere-reverse-swept-ccd",
            "surface-deforming-capsule-reverse-swept-ccd",
            "surface-deforming-convex-reverse-swept-ccd",
            "surface-deforming-triangle-mesh-reverse-swept-ccd",
            "surface-deforming-heightfield-reverse-swept-ccd",
            "AVBD_DEFORMING_SOFT_REVERSE_SWEPT",
            "AVBD_DEFORMING_SOFT_TRIANGLE_SURFACE_REVERSE_SWEPT",
            "endpointMinSeparation",
            "midSweepMinSeparation",
            "minimumVertexSweepSeparation",
            "responseDelta",
            "eENABLE_SPECULATIVE_CCD",
        ),
    )
    require_all(
        errors,
        "Volume deforming reverse swept public gates",
        volume + volume_runner,
        (
            "scene-volume-deforming-sphere-reverse-swept-ccd",
            "scene-volume-deforming-capsule-reverse-swept-ccd",
            "scene-volume-deforming-convex-reverse-swept-ccd",
            "AVBD_DEFORMING_VOLUME_REVERSE_SWEPT",
            "getSceneCpuVolumeDeformingReverseSweptProof(",
            "gSceneCpuDeformingReverseSweptFreeEndPositions",
            "geometricSweepIsolated",
            "endpointMinSeparation",
            "midSweepMinSeparation",
            "minimumVertexSweepSeparation",
            "responseDelta",
            "eENABLE_SPECULATIVE_CCD",
        ),
    )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_DEFORMING_REVERSE_SWEPT_"
            "SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_DEFORMING_REVERSE_SWEPT_SOURCE_GATE=PASS "
        "softMotion=linear-deformation rigidMotion=static+translation+rotation "
        "owners=face+finite-edge forwardOwner=swept-sdf "
        "actors=surface+volume geometries=sphere+capsule+convex+"
        "triangle-mesh+heightfield"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
