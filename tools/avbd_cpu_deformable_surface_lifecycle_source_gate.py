#!/usr/bin/env python3
"""Fail closed if the public CPU AVBD deformable-surface slice regresses."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SURFACE = ROOT / "physx/include/PxDeformableSurface.h"
PUBLIC_PHYSICS = ROOT / "physx/include/PxPhysics.h"
NP_SURFACE_HEADER = ROOT / "physx/source/physx/src/NpDeformableSurface.h"
NP_SURFACE_SOURCE = ROOT / "physx/source/physx/src/NpDeformableSurface.cpp"
NP_FACTORY_HEADER = ROOT / "physx/source/physx/src/NpFactory.h"
NP_FACTORY_SOURCE = ROOT / "physx/source/physx/src/NpFactory.cpp"
NP_PHYSICS_HEADER = ROOT / "physx/source/physx/src/NpPhysics.h"
NP_PHYSICS_SOURCE = ROOT / "physx/source/physx/src/NpPhysics.cpp"
NP_SCENE = ROOT / "physx/source/physx/src/NpScene.cpp"
NP_ATTACHMENT_HEADER = (
    ROOT / "physx/source/physx/src/NpDeformableAttachment.h"
)
NP_ATTACHMENT_SOURCE = (
    ROOT / "physx/source/physx/src/NpDeformableAttachment.cpp"
)
NP_ELEMENT_FILTER_HEADER = (
    ROOT / "physx/source/physx/src/NpDeformableElementFilter.h"
)
NP_ELEMENT_FILTER_SOURCE = (
    ROOT / "physx/source/physx/src/NpDeformableElementFilter.cpp"
)
SC_SURFACE_CORE = (
    ROOT / "physx/source/simulationcontroller/src/ScDeformableSurfaceCore.cpp"
)
SC_SCENE_HEADER = ROOT / "physx/source/simulationcontroller/include/ScScene.h"
SC_SCENE_SOURCE = ROOT / "physx/source/simulationcontroller/src/ScScene.cpp"
SOFT_COMPONENT = (
    ROOT
    / "physx/source/lowleveldynamics/src/DyAvbdSoftBodyComponent.h"
)
SNIPPET_CMAKE = ROOT / "physx/snippets/compiler/cmake/CMakeLists.txt"
SNIPPET_SOURCE = (
    ROOT
    / "physx/snippets/snippetdeformablesurfaceavbd"
    / "SnippetDeformableSurfaceAVBD.cpp"
)
HEADLESS_RUNNER = (
    ROOT / "tools/run_snippet_deformable_surface_avbd_headless.py"
)


def require(errors: list[str], condition: bool, description: str) -> None:
    if not condition:
        errors.append(description)


def require_all(
    errors: list[str], text: str, fragments: tuple[str, ...], scope: str
) -> None:
    for fragment in fragments:
        require(errors, fragment in text, f"{scope} lost {fragment!r}")


def main() -> int:
    errors: list[str] = []
    public_surface = PUBLIC_SURFACE.read_text(encoding="utf-8")
    public_physics = PUBLIC_PHYSICS.read_text(encoding="utf-8")
    np_surface_header = NP_SURFACE_HEADER.read_text(encoding="utf-8")
    np_surface_source = NP_SURFACE_SOURCE.read_text(encoding="utf-8")
    np_factory_header = NP_FACTORY_HEADER.read_text(encoding="utf-8")
    np_factory_source = NP_FACTORY_SOURCE.read_text(encoding="utf-8")
    np_physics_header = NP_PHYSICS_HEADER.read_text(encoding="utf-8")
    np_physics_source = NP_PHYSICS_SOURCE.read_text(encoding="utf-8")
    np_scene = NP_SCENE.read_text(encoding="utf-8")
    np_attachment_header = NP_ATTACHMENT_HEADER.read_text(encoding="utf-8")
    np_attachment_source = NP_ATTACHMENT_SOURCE.read_text(encoding="utf-8")
    np_element_filter_header = NP_ELEMENT_FILTER_HEADER.read_text(
        encoding="utf-8"
    )
    np_element_filter_source = NP_ELEMENT_FILTER_SOURCE.read_text(
        encoding="utf-8"
    )
    sc_surface_core = SC_SURFACE_CORE.read_text(encoding="utf-8")
    sc_scene_header = SC_SCENE_HEADER.read_text(encoding="utf-8")
    sc_scene_source = SC_SCENE_SOURCE.read_text(encoding="utf-8")
    soft_component = SOFT_COMPONENT.read_text(encoding="utf-8")
    snippet_cmake = SNIPPET_CMAKE.read_text(encoding="utf-8")
    snippet_source = SNIPPET_SOURCE.read_text(encoding="utf-8")
    headless_runner = HEADLESS_RUNNER.read_text(encoding="utf-8")
    np_scene_compact = " ".join(np_scene.split())

    require_all(
        errors,
        public_surface,
        (
            "struct PxDeformableSurfaceBackend",
            "eCPU_AVBD",
            "getDeformableSurfaceBackend() const",
            "getPositionInvMassBufferH()",
            "getVelocityBufferH()",
            "getRestPositionBufferH()",
        ),
        "public CPU deformable-surface contract",
    )
    require(
        errors,
        "createDeformableSurface(PxDeformableSurfaceBackend::Enum backend)"
        in public_physics,
        "PxPhysics lost the explicit deformable-surface backend factory",
    )
    require_all(
        errors,
        np_surface_header,
        (
            "NpDeformableSurface();",
            "mBackend",
            "mPositionInvMassBufferH",
            "mVelocityBufferH",
            "mRestPositionBufferH",
        ),
        "Np CPU deformable-surface storage",
    )
    require_all(
        errors,
        np_surface_source,
        (
            "mBackend(PxDeformableSurfaceBackend::eCPU_AVBD)",
            "mPositionInvMassBufferH.resize(numVerts)",
            "mVelocityBufferH.resize(numVerts)",
            "mRestPositionBufferH.resize(numVerts)",
            "core.cpuAvbdSleeping = false;",
            "core.cpuAvbdWakeRequested = true;",
        ),
        "Np CPU deformable-surface implementation",
    )
    require(
        errors,
        (
            "if(mBackend == PxDeformableSurfaceBackend::eGPU)"
            in np_surface_source
            and "getNbTriangleReferences()" in np_surface_source
        ),
        "CPU deformable-surface shape attachment regained a GPU vertex-mapping dependency",
    )
    require(
        errors,
        "createDeformableSurface()" in np_factory_header
        and "NpFactory::createDeformableSurface()" in np_factory_source
        and (
            "createShapeInternal<PxDeformableSurfaceMaterial, "
            "NpDeformableSurfaceMaterial>"
        )
        in np_factory_source,
        "NpFactory lost CPU deformable-surface construction",
    )
    require(
        errors,
        "createDeformableSurface(PxDeformableSurfaceBackend::Enum backend)"
        in np_physics_header
        and "NpPhysics::createDeformableSurface("
        "PxDeformableSurfaceBackend::Enum backend)"
        in np_physics_source,
        "NpPhysics lost CPU deformable-surface backend dispatch",
    )
    require_all(
        errors,
        sc_surface_core,
        (
            "requestCpuAvbdWake(",
            "core.cpuAvbdSleeping = false;",
            "core.cpuAvbdWakeRequested = true;",
        ),
        "CPU deformable-surface property wake requests",
    )
    require_all(
        errors,
        np_scene,
        (
            "CPU AVBD deformable surfaces require PxSolverType::eAVBD",
            "CPU AVBD deformable-surface host buffers are incomplete",
            "s.addAvbdCpuDeformableSurface(",
            "s.removeAvbdCpuDeformableSurface(v.getCore())",
            "return mDeformableSurfaces.size();",
            "mDeformableSurfaces.getEntries()",
        ),
        "Np CPU deformable-surface Scene lifecycle",
    )
    require(
        errors,
        (
            "case (PxConcreteType::eDEFORMABLE_SURFACE): { "
            "return addDeformableSurface("
            "static_cast<PxDeformableSurface&>(actor)); } "
            "#if PX_SUPPORT_GPU_PHYSX "
            "case (PxConcreteType::ePBD_PARTICLESYSTEM):"
        )
        in np_scene_compact,
        "PxScene::addActor CPU deformable-surface dispatch is GPU-guarded",
    )
    require(
        errors,
        (
            "case PxActorType::eDEFORMABLE_SURFACE: { "
            "NpDeformableSurface& npDeformableSurface = "
            "static_cast<NpDeformableSurface&>(actor); "
            "removeDeformableSurface(npDeformableSurface, "
            "wakeOnLostTouch); } break; "
            "#if PX_SUPPORT_GPU_PHYSX "
            "case PxActorType::ePBD_PARTICLESYSTEM:"
        )
        in np_scene_compact,
        "PxScene::removeActor CPU deformable-surface dispatch is GPU-guarded",
    )
    require_all(
        errors,
        sc_scene_header,
        (
            "addAvbdCpuDeformableSurface(",
            "removeAvbdCpuDeformableSurface(",
        ),
        "Sc CPU deformable-surface declaration",
    )
    require_all(
        errors,
        sc_scene_source,
        (
            "Sc::Scene::addAvbdCpuDeformableSurface(",
            "Sc::Scene::removeAvbdCpuDeformableSurface(",
            "PxTriangleMesh&",
            "PxDeformableSurfaceDataFlag::ePOSITION_INVMASS",
            "PxDeformableSurfaceDataFlag::eVELOCITY",
            "PxDeformableSurfaceDataFlag::eREST_POSITION",
            "IG::Node::eDEFORMABLE_SURFACE_TYPE",
        ),
        "Sc CPU deformable-surface Scene implementation",
    )
    require_all(
        errors,
        sc_scene_source,
        (
            "enum EntryKind",
            "eVOLUME",
            "eSURFACE",
            "PxArray<Entry>",
            "entry.getBodyCore()",
            "entry.getPositionInvMass()",
            "entry.getVelocity()",
        ),
        "unified CPU soft Scene storage",
    )
    require_all(
        errors,
        sc_scene_source,
        (
            "compileDynamicBox(",
            "entry.core->getKinematicTarget(targetPose)",
            "box.previousCenter = previousShapeToWorld.p;",
            "box.previousRotation = previousShapeToWorld.q;",
            "eKINEMATIC_RIGID",
        ),
        "CPU soft Scene kinematic target preparation",
    )
    for legacy_fragment in (
        "SurfaceEntry",
        "mSurfaceEntries",
        "mSurfaceParticles",
        "mSurfaceBodies",
        "mSurfaceContacts",
        "mSurfaceWorkspace",
    ):
        require(
            errors,
            legacy_fragment not in sc_scene_source,
            f"duplicate CPU surface path returned: {legacy_fragment}",
        )
    require_all(
        errors,
        sc_scene_source,
        (
            "mSelfCollisionAdjacencies",
            "mSelfCollisionEnabled",
            "avbdBuildSelfCollisionAdjacency(",
            "refreshSelfCollisionEnabled()",
            "PxDeformableBodyFlag::",
            "eDISABLE_SELF_COLLISION",
            "storage.selfCollisionAdjacencies",
            "storage.selfCollisionEnabled",
        ),
        "CPU soft Scene self-collision routing",
    )
    require_all(
        errors,
        np_attachment_header + np_attachment_source,
        (
            "isCpuAvbdWorldVertexAttachment",
            "addAvbdCpuDeformableSurfaceWorldPin",
            "updateAvbdCpuDeformableSurfaceWorldPin",
            "removeAvbdCpuDeformableSurfaceWorldPin",
            "eSURFACE_VTX_GLOBAL_POSE",
            "isCpuAvbdRigidVertexAttachment",
            "isCpuAvbdKinematicVertexAttachment",
            "eSURFACE_VTX_RIGID_BODY",
            "addAvbdCpuDeformableSurfaceRigidAttachment",
            "updateAvbdCpuDeformableSurfaceRigidAttachment",
            "removeAvbdCpuDeformableSurfaceRigidAttachment",
            "addAvbdCpuDeformableSurfaceKinematicAttachment",
            "updateAvbdCpuDeformableSurfaceKinematicAttachment",
            "removeAvbdCpuDeformableSurfaceKinematicAttachment",
        ),
        "Np CPU AVBD surface vertex attachment",
    )
    require_all(
        errors,
        np_factory_header + np_factory_source,
        (
            "createDeformableAttachment(",
            "mAttachmentTracking",
            "mAttachmentPool",
            "cpuSupported",
        ),
        "CPU-capable deformable attachment factory",
    )
    attachment_factory = np_physics_source.split(
        "PxDeformableAttachment* NpPhysics::createDeformableAttachment",
        1,
    )
    require(
        errors,
        len(attachment_factory) == 2
        and "PX_SUPPORT_GPU_PHYSX"
        not in attachment_factory[1].split(
            "PxDeformableElementFilter* "
            "NpPhysics::createDeformableElementFilter",
            1,
        )[0],
        "NpPhysics put deformable attachment creation back behind the GPU build",
    )
    require_all(
        errors,
        sc_scene_header + sc_scene_source,
        (
            "WorldPinEntry",
            "mWorldPins",
            "rebuildEntryPins",
            "AvbdKinematicPin",
            "addAvbdCpuDeformableSurfaceWorldPin",
            "updateAvbdCpuDeformableSurfaceWorldPin",
            "removeAvbdCpuDeformableSurfaceWorldPin",
            "RigidAttachmentEntry",
            "mRigidAttachments",
            "ensureNativeIslandEdge",
            "addAvbdCpuDeformableSurfaceRigidAttachment",
            "updateAvbdCpuDeformableSurfaceRigidAttachment",
            "removeAvbdCpuDeformableSurfaceRigidAttachment",
            "PrescribedAttachmentEntry",
            "mPrescribedAttachments",
            "AvbdSoftPinTargetKind",
            "ePRESCRIBED_RIGID",
            "addAvbdCpuDeformableSurfaceKinematicAttachment",
            "updateAvbdCpuDeformableSurfaceKinematicAttachment",
            "removeAvbdCpuDeformableSurfaceKinematicAttachment",
        ),
        "Sc CPU AVBD attachment objective routing",
    )
    require_all(
        errors,
        np_element_filter_header + np_element_filter_source,
        (
            "cpuAvbdRigidActorFilter",
            "cpuAvbdDeformablePairFilter",
            "cpuAvbdFilterAllElements",
            "hasValidDeformableElementGroups(",
            "hasValidDeformablePairGroups(",
            "coversEveryDeformableElement(",
            "deformable-pair element filters",
            "eSURFACE_TRI_RIGID_BODY",
            "eSURFACE_TRI_SURFACE_TRI",
            "eVOLUME_TET_SURFACE_TRI",
            "eVOLUME_TET_VOLUME_TET",
            "mTetsAccumulatedRemapColToSim",
            "mTetsRemapColToSim",
            "addAvbdCpuDeformableSurfaceRigidActorFilter(",
            "removeAvbdCpuDeformableSurfaceRigidActorFilter(",
            "addAvbdCpuDeformableSurfaceSurfaceFilter(",
            "addAvbdCpuDeformableVolumeSurfaceFilter(",
            "addAvbdCpuDeformableVolumeVolumeFilter(",
            "removeAvbdCpuDeformablePairFilter(",
        ),
        "Np CPU AVBD exact deformable-pair element-filter routes",
    )
    require_all(
        errors,
        sc_scene_header + sc_scene_source,
        (
            "RigidActorFilterEntry",
            "mRigidActorFilters",
            "removeRigidActorFilteredContacts(",
            "isRigidActorContactFiltered(",
            "filterAllElements",
            "containsElement(",
            "elementAdjacency[",
            "sourceElementIndex",
            "findRigidCoreForPrimitive(",
            "addAvbdCpuDeformableSurfaceRigidActorFilter(",
            "removeAvbdCpuDeformableSurfaceRigidActorFilter(",
            "DeformablePairFilterEntry",
            "mDeformablePairFilters",
            "containsPair(",
            "isDeformablePairContactFiltered(",
            "removeDeformablePairFilteredContacts(",
            "targetSourceElementIndex",
            "expandVolumeCollisionElement(",
            "mTetsAccumulatedRemapColToSim",
            "mTetsRemapColToSim",
            "addAvbdCpuDeformableSurfaceSurfaceFilter(",
            "addAvbdCpuDeformableVolumeSurfaceFilter(",
            "addAvbdCpuDeformableVolumeVolumeFilter(",
            "removeAvbdCpuDeformablePairFilter(",
        ),
        "Sc CPU AVBD prep-time element filter routing",
    )
    require_all(
        errors,
        soft_component,
        (
            "const PxU8* selfCollisionEnabled = NULL",
            "(!selfCollisionEnabled || selfCollisionEnabled[si])",
            "avbdDetectSelfCollisionOGC(",
        ),
        "low-level self-collision enable mask",
    )
    require_all(
        errors,
        soft_component,
        (
            "const bool targetIsShell =",
            "const bool isInside = !targetIsShell",
            "PxReal bestDistance = r;",
            "if(bestTriangle != PX_MAX_U32)",
            "appendOutwardContact(",
            "targetSourceElementIndex",
            "geometry.targetSourceElementIndex =",
            "surfaceTriangleElementIndices",
            "faceElementIndices",
        ),
        "unique soft-soft contact objective preparation and target ownership",
    )
    require_all(
        errors,
        soft_component,
        (
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdVelocityObjectiveOwner::ComponentFinalize",
            "struct AvbdCompiledSoftVelocityObjective",
            "compileVelocityObjectives",
            "avbdFinalizeSoftComponentVelocities(",
        ),
        "CPU soft kinematic component-finalize routing",
    )
    require_all(
        errors,
        sc_scene_source,
        (
            "restVertices[i] = restPosition.getXYZ();",
            "material ? material->bendingStiffness : 0.0f",
            "Dy::avbdCreateSoftBody(",
        ),
        "public surface rest-shape and bending material compilation",
    )
    require_all(
        errors,
        soft_component,
        (
            "void buildBendingElements(",
            "avbdEvaluateBendingForceHessian(",
            "sb.material.bendingStiffness",
        ),
        "position-level AVBD bending objective",
    )
    require(
        errors,
        "DeformableSurfaceAVBD" in snippet_cmake,
        "CPU deformable-surface snippet is missing from the CPU snippet list",
    )
    require_all(
        errors,
        snippet_source,
        (
            "createDeformableSurface(",
            "PxDeformableSurfaceBackend::eCPU_AVBD",
            "cookingParams.buildGPUData = false;",
            "getPositionInvMassBufferH()",
            "getVelocityBufferH()",
            "getRestPositionBufferH()",
            "scene->removeActor(*surface)",
            "scene->addActor(*surface)",
            'options.caseName == "surface-ground"',
            'options.caseName == "surface-sleep-wake"',
            'options.caseName == "surface-buffer-mutation"',
            'options.caseName == "surface-dynamic-box"',
            'options.caseName == "surface-kinematic-box"',
            'options.caseName == "surface-soft-soft-wake"',
            'options.caseName == "surface-volume-wake"',
            'options.caseName == "surface-self-collision"',
            'options.caseName == "surface-world-pin"',
            'options.caseName == "surface-rigid-attachment"',
            'options.caseName == "surface-element-filter"',
            'options.caseName == "surface-partial-element-filter"',
            'options.caseName == "surface-soft-soft-element-filter"',
            'options.caseName == "surface-volume-element-filter"',
            'options.caseName == "volume-volume-element-filter"',
            'options.caseName == "surface-bending"',
            "surface->isSleeping()",
            "PxDeformableSurfaceDataFlag::eVELOCITY",
            "restPositions[mutationIndex]",
            "PxDeformableSurfaceDataFlag::eALL",
            "bufferInvMassRestored",
            "selfCollisionPreventedCrossing",
            "selfCollisionDisabledCrossed",
            "mixedVolumeWoke",
            "mixedVolumeMoved",
            "attachmentCreated",
            "attachmentPinned",
            "attachmentReleased",
            "attachmentMovedAfterRelease",
            "rigidAttachmentRigidMoved",
            "rigidAttachmentRigidRotated",
            "rigidAttachmentMaxAngularDisplacement",
            "rigidAttachmentMaxAngularSpeed",
            "createDeformableElementFilter(",
            "elementFilterHeldAcrossReadd",
            "elementFilterSuppressedContact",
            "elementFilterContactRestored",
            "partialFilterExactOwnership",
            "partialFilterUnfilteredContactHeld",
            "partialFilterFilteredMinY",
            "partialFilterUnfilteredMinY",
            "bendingMaterialPairCreated",
            "bendingZeroControlHeld",
            "bendingResponseObserved",
            "bendingMembraneIsolated",
            "bendingMaxEdgeStrain",
            "kinematicBoxAdded",
            "kinematicTargetIssued",
            "kinematicTargetReached",
            "kinematicSurfaceWoke",
            "kinematicSurfaceMoved",
            "kinematicContactObserved",
            "kinematicMaxPoseError",
            "kinematicSurfaceDisplacement",
            "kinematicFinalY",
            "[AVBD_GATE]",
        ),
        "public CPU deformable-surface runtime gate",
    )
    require(
        errors,
        "Dy::" not in snippet_source
        and "lowleveldynamics" not in snippet_source,
        "CPU deformable-surface snippet bypasses the public API",
    )
    require_all(
        errors,
        headless_runner,
        (
            "run_headless_process(",
            "visible_window_detected",
            "SnippetDeformableSurfaceAVBD_64.exe",
            "surface-lifecycle",
            "surface-ground",
            "surface-sleep-wake",
            "surface-buffer-mutation",
            "surface-dynamic-box",
            "surface-kinematic-box",
            "surface-soft-soft-wake",
            "surface-volume-wake",
            "surface-self-collision",
            "surface-world-pin",
            "surface-rigid-attachment",
            "surface-element-filter",
            "surface-partial-element-filter",
            "surface-soft-soft-element-filter",
            "surface-volume-element-filter",
            "volume-volume-element-filter",
            "surface-bending",
            "selfCollisionMinEnabledSeparation",
            "selfCollisionMinDisabledSeparation",
            "attachmentPinMaxDrift",
            "attachmentReleasedMaxDisplacement",
            "rigidAttachmentMaxRigidDisplacement",
            "rigidAttachmentMaxAngularDisplacement",
            "rigidAttachmentMaxAngularSpeed",
            "off-center surface attachment did not publish ",
            "elementFilterMinY",
            "elementFilterFinalMinY",
            "filtered deformable elements did not suppress contact",
            "filtered element did not pass through its contact target",
            "unfiltered element lost its contact target",
            "partialFilterExactOwnership",
            "positive bending stiffness did not reduce dihedral error",
            "zero-bending negative control unexpectedly moved",
            "bendingMaxEdgeStrain",
            "dynamicBoxFinalLinearSpeed",
            "kinematicMaxPoseError",
            "kinematicSurfaceDisplacement",
            "kinematicFinalY",
            "kinematic coupling amplified the surface speed",
            "kinematic coupling left excessive residual speed",
            "secondSurfaceFinalMaxSpeed",
            "mixedVolumeFinalMaxSpeed",
            "mixed-contact cases require at least 600 frames",
            "DETERMINISM_KEYS",
            "deterministic metrics differ from repeat 1",
        ),
        "CPU deformable-surface hidden runner",
    )

    if errors:
        print("AVBD_CPU_DEFORMABLE_SURFACE_LIFECYCLE_SOURCE_GATE=FAIL")
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("AVBD_CPU_DEFORMABLE_SURFACE_LIFECYCLE_SOURCE_GATE=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
