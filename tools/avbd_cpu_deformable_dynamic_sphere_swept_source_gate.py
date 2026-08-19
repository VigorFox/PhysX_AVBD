#!/usr/bin/env python3
"""Lock E35 CPU AVBD dynamic-sphere relative swept OGC ownership."""

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

    descriptor = section(
        soft,
        "struct AvbdRigidSphere",
        "// =============================================================================",
    )
    require_all(
        errors,
        "dynamic sphere predicted-pose descriptor",
        descriptor,
        (
            "PxVec3 predictedCenter;",
            "PxQuat predictedRotation;",
            "bool predictedPoseValid;",
            "predictedCenter(0.0f)",
            "predictedRotation(PxIdentity)",
            "predictedPoseValid(false)",
        ),
    )

    target = section(
        soft,
        "PX_FORCE_INLINE void avbdConfigureRigidSphereTarget(",
        "inline void avbdDetectSoftRigidSphereSDF(",
    )
    require_all(
        errors,
        "dynamic sphere two-sided response owner",
        target,
        (
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "AvbdVelocityObjectiveOwner::ManifoldFinalize",
            "geometry.rigidLocalPoint =",
            "sphere.shapeToRigidBody.transform(surfaceLocal);",
        ),
    )

    swept = section(
        soft,
        "inline void avbdDetectSoftRigidSphereSweptSDFRange(",
        "inline void avbdDetectSoftRigidOGCFeatures(",
    )
    require_all(
        errors,
        "dynamic sphere relative swept detector",
        swept,
        (
            "!sourceBody->compiled.speculativeCCDEnabled",
            "const bool dynamicTarget =",
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "dynamicTarget &&",
            "!sphere.predictedPoseValid",
            "!sphere.predictedCenter.isFinite()",
            "!sphere.predictedRotation.isFinite()",
            "const PxVec3 sphereCenterStart =",
            "? sphere.previousCenter : sphere.center;",
            "const PxVec3 sphereCenterEnd =",
            "? sphere.predictedCenter : sphere.center;",
            "const PxVec3 relativeStart =",
            "particle.position - sphereCenterStart;",
            "const PxVec3 relativeEnd =",
            "particle.predictedPosition - sphereCenterEnd;",
            "relativeEnd - relativeStart;",
            "avbdSegmentEnterExpandedSphere(",
            "avbdConfigureRigidSphereTarget(",
        ),
    )
    if "particle.predictedPosition - particle.position" in swept:
        errors.append(
            "dynamic sphere sweep regressed to soft-only displacement"
        )

    topology = section(
        scene,
        "void prepareIslandGeneration(",
        "virtual bool prepareSoftIslandSelections(",
    )
    require_all(
        errors,
        "public-flag-gated predicted topology",
        topology,
        (
            "PxReal dt, const PxVec3& gravity",
            "const bool speculativeCCDEnabled =",
            "compiled.",
            "speculativeCCDEnabled;",
            "expandSoftBoundsForPrediction(",
            "softEntry, dt, gravity, softBounds",
            "const PxVec3 predictedBodyCenter =",
            "bodyCore.linearVelocity * dt",
            "gravity * (dt * dt)",
            "const PxReal envelopeRadius =",
            "sphere.radius + shapeOffset;",
            "rigidBounds.include(",
            "predictedBodyCenter -",
            "ensureNativeIslandEdge(",
        ),
    )

    selection = section(
        scene,
        "virtual bool prepareSoftIslandSelections(",
        "void step(",
    )
    require_all(
        errors,
        "native-edge bridge and partial selection ownership",
        selection,
        (
            "mDynamicsSelectedEntryCount = 0;",
            "edge.touched",
            "bridgeIsland",
            "bridgeAmbiguous",
            "islandBodyCounts[islandIndex] == 0",
            "storage.touched = false;",
            "without discarding independent complete selections",
            "mDynamicsOwnsStep = !selections.empty();",
            "mDynamicsSelectedEntryCount =",
            "selectedEntryCount",
        ),
    )

    mixed_step = section(
        scene,
        "void step(",
        "const PxsDeformableSurfaceMaterialCore*",
    )
    require_all(
        errors,
        "mixed native/fallback scheduling",
        mixed_step,
        (
            "PxU32 awakeEntryCount = 0;",
            "if(mDynamicsOwnsStep)",
            "mDynamicsSelectedEntryCount < awakeEntryCount",
            "stepComponentFallback(",
            "copyIslandSelectionResults(storage);",
            "void stepComponentFallback(",
            "Dy::avbdStepSoftBodies(",
            "mDynamicsSelectedEntryCount = 0;",
        ),
    )

    compile_sphere = section(
        scene,
        "void compileDynamicSpheresForIsland(",
        "void refreshSelfCollisionEnabled(",
    )
    require_all(
        errors,
        "current-frame dynamic sphere prediction",
        compile_sphere,
        (
            "Dy::AvbdSolverBody* solverBodies",
            "PxReal dt",
            "const PxVec3& gravity",
            "sphere.targetKind =",
            "eRIGID_BODY;",
            "sphere.shapeToRigidBody =",
            "solverBody.computePrediction(dt, gravity);",
            "solverBody.predictedPosition",
            "solverBody.predictedRotation",
            "predictedBodyToWorld * sphere.shapeToRigidBody",
            "sphere.predictedCenter =",
            "sphere.predictedRotation =",
            "sphere.predictedPoseValid = true;",
        ),
    )

    island_contacts = section(
        scene,
        "bool buildIslandSelectionStorage(",
        "void copyIslandSelectionResults(",
    )
    require_all(
        errors,
        "selected dynamic sphere discrete+swept route",
        island_contacts,
        (
            "compileDynamicSpheresForIsland(",
            "Dy::avbdDetectSoftRigidSphereSDF(",
            "Dy::avbdDetectSoftRigidSphereSweptSDF(",
            "Dy::avbdDetectSoftRigidSphereOGCFeatures(",
            "storage.selectedDynamicSpheres.begin()",
            "storage.probeContacts",
        ),
    )

    public_gate_fragments = (
        "dynamicSphereSweepLaunched",
        "dynamicSphereSweepResponseObserved",
        "dynamicSphereSweepNegativeControlTunneled",
        "dynamicSphereSweepTwoSidedResponseObserved",
        "dynamicSphereSweepPositiveSoftDisplacement",
        "dynamicSphereSweepNegativeSoftDisplacement",
        "dynamicSphereSweepPositiveRigidDrop",
        "dynamicSphereSweepNegativeRigidDrop",
        "dynamicSphereSweepPositiveMinSeparation",
        "eENABLE_SPECULATIVE_CCD",
    )
    require_all(
        errors,
        "Surface dynamic relative-sweep public gate",
        surface + surface_runner,
        (
            "surface-dynamic-sphere-relative-swept-ccd",
            "PxSphereGeometry(radius)",
            "rotationalFiniteCase ? 0.0f : -132.0f",
        )
        + public_gate_fragments,
    )
    require_all(
        errors,
        "Volume dynamic relative-sweep public gate",
        volume + volume_runner,
        (
            "scene-volume-dynamic-sphere-relative-swept-ccd",
            "addSceneDynamicFiniteSweepPair(",
            "0.8f, 0.0f, -132.0f",
            "a soft nor a rigid response",
            "positive_rigid_drop + 0.05",
        )
        + public_gate_fragments,
    )
    if "Dy::" in surface or "Dy::" in volume:
        errors.append(
            "public Surface/Volume E35 gates bypass the public Scene API"
        )

    if errors:
        print(
            "AVBD_CPU_DEFORMABLE_DYNAMIC_SPHERE_SWEPT_SOURCE_GATE=FAIL"
        )
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "AVBD_CPU_DEFORMABLE_DYNAMIC_SPHERE_SWEPT_SOURCE_GATE=PASS "
        "geometry=sphere target=dynamic-rigid-two-sided "
        "sweep=relative-predicted topology=native-edge "
        "scheduling=mixed-main-fallback actors=surface+volume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
