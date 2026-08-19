#!/usr/bin/env python3
"""Lock E34 CPU AVBD moving-kinematic-sphere relative swept ownership."""

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

    target = section(
        soft,
        "PX_FORCE_INLINE void avbdConfigureRigidSphereTarget(",
        "inline void avbdDetectSoftRigidSphereSDF(",
    )
    require_all(
        errors,
        "prescribed sphere target ownership",
        target,
        (
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdVelocityObjectiveOwner::ComponentFinalize",
            "sphere.previousCenter +",
            "sphere.previousRotation.rotate(surfaceLocal)",
        ),
    )

    swept = section(
        soft,
        "inline void avbdDetectSoftRigidSphereSweptSDFRange(",
        "inline void avbdDetectSoftRigidSphereSweptSDF(",
    )
    require_all(
        errors,
        "moving-kinematic-sphere relative sweep",
        swept,
        (
            "!sourceBody->compiled.speculativeCCDEnabled",
            "AvbdSoftContactTargetKind::eWORLD_STATIC",
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "const bool dynamicTarget =",
            "const PxVec3 sphereCenterStart =",
            "? sphere.previousCenter : sphere.center;",
            "const PxVec3 sphereCenterEnd =",
            "? sphere.predictedCenter : sphere.center;",
            "const PxVec3 relativeStart =",
            "particle.position - sphereCenterStart;",
            "const PxVec3 relativeEnd =",
            "particle.predictedPosition - sphereCenterEnd;",
            "relativeEnd - relativeStart;",
            "relativeStart.magnitude() - sphere.radius;",
            "avbdSegmentEnterExpandedSphere(",
            "relativeStart,",
            "relativeEnd,",
            "PxVec3(0.0f),",
            "sphere.radius + margin",
            "sphere.rotation.getConjugate().rotate(",
            "avbdConfigureRigidSphereTarget(",
        ),
    )
    if "particle.predictedPosition - particle.position" in swept:
        errors.append(
            "E34 swept detector regressed to soft-only displacement"
        )

    require_all(
        errors,
        "Surface moving-sphere public positive/negative gate",
        surface + surface_runner,
        (
            "surface-moving-kinematic-sphere-speculative-ccd",
            "PxSphereGeometry(0.8f)",
            "setKinematicTarget(",
            "eENABLE_SPECULATIVE_CCD",
            "movingSphereTargetIssued",
            "movingSphereCcdResponseObserved",
            "movingSphereNegativeControlHeld",
            "movingSpherePositiveDisplacement",
            "movingSphereNegativeDisplacement",
            "movingSpherePositiveMinSeparation",
            "positive_displacement <= 0.02",
            "negative_displacement >= 0.005",
            "min_separation <= -0.10",
        ),
    )
    require_all(
        errors,
        "Volume moving-sphere public positive/negative gate",
        volume + volume_runner,
        (
            "scene-volume-moving-kinematic-sphere-speculative-ccd",
            "addSceneMovingKinematicFinitePair(",
            "0.0f, 0.8f, 0.0f",
            "setKinematicTarget(",
            "eENABLE_SPECULATIVE_CCD",
            "movingSphereTargetIssued",
            "movingSphereCcdResponseObserved",
            "movingSphereNegativeControlHeld",
            "movingSpherePositiveDisplacement",
            "movingSphereNegativeDisplacement",
            "movingSpherePositiveMinSeparation",
            "if(gMetrics.completedFrames == 1)",
            "gScene->removeActor(",
            "*gSceneCpuSecondVolume",
            "*gSceneCpuSecondDynamicActor",
            "positive_displacement <= 0.02",
            "negative_displacement >= 0.005",
            "min_separation <= -0.10",
        ),
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume E34 gates bypass the public Scene API"
        )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_MOVING_KINEMATIC_SPHERE_SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_MOVING_KINEMATIC_SPHERE_SOURCE_GATE=PASS "
        "geometry=sphere target=kinematic-prescribed "
        "sweep=relative actors=surface+volume "
        "dynamicScope=predicted-valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
