#!/usr/bin/env python3
"""Fail closed if the public CPU AVBD volume Scene slice regresses."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_VOLUME = ROOT / "physx/include/PxDeformableVolume.h"
PUBLIC_PHYSICS = ROOT / "physx/include/PxPhysics.h"
NP_VOLUME_HEADER = ROOT / "physx/source/physx/src/NpDeformableVolume.h"
NP_VOLUME_SOURCE = ROOT / "physx/source/physx/src/NpDeformableVolume.cpp"
NP_ATTACHMENT_SOURCE = (
    ROOT / "physx/source/physx/src/NpDeformableAttachment.cpp"
)
NP_ELEMENT_FILTER_HEADER = (
    ROOT / "physx/source/physx/src/NpDeformableElementFilter.h"
)
NP_ELEMENT_FILTER_SOURCE = (
    ROOT / "physx/source/physx/src/NpDeformableElementFilter.cpp"
)
NP_FACTORY_SOURCE = ROOT / "physx/source/physx/src/NpFactory.cpp"
NP_PHYSICS = ROOT / "physx/source/physx/src/NpPhysics.cpp"
NP_SCENE = ROOT / "physx/source/physx/src/NpScene.cpp"
SC_SCENE_HEADER = ROOT / "physx/source/simulationcontroller/include/ScScene.h"
SC_SCENE_SOURCE = ROOT / "physx/source/simulationcontroller/src/ScScene.cpp"
SC_PIPELINE = ROOT / "physx/source/simulationcontroller/src/ScPipeline.cpp"
DEFORMABLE_BODY_CORE = (
    ROOT / "physx/source/lowleveldynamics/include/DyDeformableBodyCore.h"
)
DEFORMABLE_VOLUME_CORE_SOURCE = (
    ROOT
    / "physx/source/simulationcontroller/src/ScDeformableVolumeCore.cpp"
)
SOFT_COMPONENT = (
    ROOT / "physx/source/lowleveldynamics/src/DyAvbdSoftBodyComponent.h"
)
AVBD_DYNAMICS_HEADER = (
    ROOT / "physx/source/lowleveldynamics/src/DyAvbdDynamics.h"
)
AVBD_DYNAMICS_SOURCE = (
    ROOT / "physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp"
)
AVBD_SOLVER_BODY = (
    ROOT / "physx/source/lowleveldynamics/src/DyAvbdSolverBody.h"
)
AVBD_BODY_CONVERSION = (
    ROOT / "physx/source/lowleveldynamics/src/DyAvbdBodyConversion.h"
)
SNIPPET = (
    ROOT
    / "physx/snippets/snippetdeformablevolumeavbd/"
    "SnippetDeformableVolumeAVBD.cpp"
)
RUNNER = ROOT / "tools/run_snippet_deformable_volume_avbd_headless.py"
UNIT_SNIPPET = (
    ROOT / "physx/snippets/snippetsoftbodyavbd/SnippetSoftBodyAVBD.cpp"
)
UNIT_RUNNER = ROOT / "tools/run_snippet_soft_body_avbd_headless.py"


def require(errors: list[str], condition: bool, description: str) -> None:
    if not condition:
        errors.append(description)


def require_all(
    errors: list[str], text: str, fragments: tuple[str, ...], scope: str
) -> None:
    for fragment in fragments:
        require(
            errors,
            fragment in text,
            f"{scope} lost {fragment!r}",
        )


def main() -> int:
    errors: list[str] = []
    public_volume = PUBLIC_VOLUME.read_text(encoding="utf-8")
    public_physics = PUBLIC_PHYSICS.read_text(encoding="utf-8")
    np_volume_header = NP_VOLUME_HEADER.read_text(encoding="utf-8")
    np_volume_source = NP_VOLUME_SOURCE.read_text(encoding="utf-8")
    np_attachment_source = NP_ATTACHMENT_SOURCE.read_text(encoding="utf-8")
    np_element_filter_header = NP_ELEMENT_FILTER_HEADER.read_text(
        encoding="utf-8"
    )
    np_element_filter_source = NP_ELEMENT_FILTER_SOURCE.read_text(
        encoding="utf-8"
    )
    np_factory_source = NP_FACTORY_SOURCE.read_text(encoding="utf-8")
    np_physics = NP_PHYSICS.read_text(encoding="utf-8")
    np_scene = NP_SCENE.read_text(encoding="utf-8")
    sc_scene_header = SC_SCENE_HEADER.read_text(encoding="utf-8")
    sc_scene_source = SC_SCENE_SOURCE.read_text(encoding="utf-8")
    sc_pipeline = SC_PIPELINE.read_text(encoding="utf-8")
    deformable_body_core = DEFORMABLE_BODY_CORE.read_text(encoding="utf-8")
    deformable_volume_core_source = DEFORMABLE_VOLUME_CORE_SOURCE.read_text(
        encoding="utf-8"
    )
    soft_component = SOFT_COMPONENT.read_text(encoding="utf-8")
    avbd_dynamics_header = AVBD_DYNAMICS_HEADER.read_text(encoding="utf-8")
    avbd_dynamics_source = AVBD_DYNAMICS_SOURCE.read_text(encoding="utf-8")
    avbd_solver_body = AVBD_SOLVER_BODY.read_text(encoding="utf-8")
    avbd_body_conversion = AVBD_BODY_CONVERSION.read_text(encoding="utf-8")
    snippet = SNIPPET.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    unit_snippet = UNIT_SNIPPET.read_text(encoding="utf-8")
    unit_runner = UNIT_RUNNER.read_text(encoding="utf-8")

    require_all(
        errors,
        public_volume,
        (
            "struct PxDeformableVolumeBackend",
            "eCPU_AVBD",
            "getDeformableVolumeBackend() const",
            "getPositionInvMassBufferH()",
            "getRestPositionBufferH()",
            "getSimPositionInvMassBufferH()",
            "getSimVelocityBufferH()",
            "setKinematicTargetBufferH(const PxVec4* positions)",
        ),
        "public CPU deformable-volume contract",
    )
    require(
        errors,
        "createDeformableVolume(PxDeformableVolumeBackend::Enum backend)"
        in public_physics,
        "PxPhysics lost the explicit deformable-volume backend factory",
    )

    require_all(
        errors,
        np_volume_header,
        (
            "NpDeformableVolume();",
            "mBackend",
            "mPositionInvMassBufferH",
            "mRestPositionBufferH",
            "mSimPositionInvMassBufferH",
            "mSimVelocityBufferH",
        ),
        "Np CPU deformable-volume storage",
    )
    require_all(
        errors,
        np_volume_source,
        (
            "mCudaContextManager(NULL)",
            "mBackend(PxDeformableVolumeBackend::eCPU_AVBD)",
            "mPositionInvMassBufferH.resize(numVerts)",
            "mSimPositionInvMassBufferH.resize(numVertsGM)",
            "mBackend == PxDeformableVolumeBackend::eGPU",
            "mBackend == PxDeformableVolumeBackend::eCPU_AVBD",
            "PxBounds3 NpDeformableVolume::getWorldBounds",
            "bool NpDeformableVolume::isSleeping() const",
            "cpuAvbdSleeping",
            "cpuAvbdWakeRequested",
            "core.dirtyFlags |= flags;",
            "core.cpuAvbdSleeping = false;",
            "core.cpuAvbdWakeRequested = true;",
        ),
        "Np CPU deformable-volume implementation",
    )
    require_all(
        errors,
        deformable_body_core,
        (
            "bool\t\t\t\t\tcpuAvbdSleeping;",
            "bool\t\t\t\t\tcpuAvbdWakeRequested;",
            "cpuAvbdSleeping(false)",
            "cpuAvbdWakeRequested(false)",
        ),
        "CPU AVBD deformable sleep state",
    )
    require_all(
        errors,
        deformable_volume_core_source,
        (
            "requestCpuAvbdWake(",
            "mCore.cpuAvbdSleeping = false;",
            "mCore.cpuAvbdWakeRequested = true;",
            "if(v > 0.0f)",
        ),
        "CPU AVBD sleep-affecting property wake requests",
    )
    require(
        errors,
        "if(backend == PxDeformableVolumeBackend::eCPU_AVBD)" in np_physics
        and "NpFactory::getInstance().createDeformableVolume()" in np_physics,
        "explicit CPU backend factory no longer creates an Np actor",
    )
    require_all(
        errors,
        np_attachment_source,
        (
            "eVOLUME_VTX_GLOBAL_POSE",
            "eVOLUME_VTX_RIGID_BODY",
            "PxDeformableVolumeBackend::eCPU_AVBD",
            "isCpuAvbdRigidVertexAttachment(",
            "isCpuAvbdKinematicVertexAttachment(",
            "addAvbdCpuDeformableVolumeWorldPin(",
            "updateAvbdCpuDeformableVolumeWorldPin(",
            "removeAvbdCpuDeformableVolumeWorldPin(",
            "addAvbdCpuDeformableVolumeRigidAttachment(",
            "updateAvbdCpuDeformableVolumeRigidAttachment(",
            "removeAvbdCpuDeformableVolumeRigidAttachment(",
            "addAvbdCpuDeformableVolumeArticulationAttachment(",
            "updateAvbdCpuDeformableVolumeArticulationAttachment(",
            "removeAvbdCpuDeformableVolumeArticulationAttachment(",
            "addAvbdCpuDeformableVolumeKinematicAttachment(",
            "updateAvbdCpuDeformableVolumeKinematicAttachment(",
            "removeAvbdCpuDeformableVolumeKinematicAttachment(",
        ),
        "CPU AVBD deformable-volume attachment routing",
    )
    require_all(
        errors,
        np_factory_source,
        (
            "cpuAvbdVolume",
            "cpuAvbdSurface",
            "eVOLUME_VTX_GLOBAL_POSE",
            "eVOLUME_VTX_RIGID_BODY",
            "eSURFACE_VTX_RIGID_BODY",
            "nonKinematicDynamic",
            "kinematicDynamic",
            "articulationLink",
            "cpuDynamicRigidVertex",
            "cpuWorldElement",
            "cpuDynamicRigidElement",
            "cpuKinematicRigidVertex",
            "cpuKinematicRigidElement",
            "cpuArticulationRigidVertex",
            "cpuArticulationRigidElement",
            "softPairType",
            "cpuSoftPair",
            "cpuSoftPairDataValid",
            "hasValidCpuAvbdSoftPairAttachmentData(",
            "hasValidCpuAvbdElementAttachmentData(",
            "AVBD currently supports only non-empty deformable ",
            "surface/volume vertex or element attachments to ",
            "world, PxRigidStatic, PxRigidDynamic, or ",
            "PxArticulationLink, plus two-sided deformable-pair ",
        ),
        "CPU AVBD deformable-volume attachment factory gate",
    )
    require_all(
        errors,
        sc_scene_header,
        (
            "addAvbdCpuDeformableVolumeWorldPin(",
            "updateAvbdCpuDeformableVolumeWorldPin(",
            "removeAvbdCpuDeformableVolumeWorldPin(",
            "addAvbdCpuDeformableVolumeRigidAttachment(",
            "updateAvbdCpuDeformableVolumeRigidAttachment(",
            "removeAvbdCpuDeformableVolumeRigidAttachment(",
            "addAvbdCpuDeformableVolumeKinematicAttachment(",
            "updateAvbdCpuDeformableVolumeKinematicAttachment(",
            "removeAvbdCpuDeformableVolumeKinematicAttachment(",
        ),
        "CPU AVBD deformable-volume Scene attachment API",
    )
    require_all(
        errors,
        sc_scene_source,
        (
            "struct WorldPinEntry",
            "mWorldPins",
            "rebuildEntryPins(",
            "Dy::AvbdKinematicPin",
            "Scene::addAvbdCpuDeformableVolumeWorldPin(",
            "Scene::updateAvbdCpuDeformableVolumeWorldPin(",
            "Scene::removeAvbdCpuDeformableVolumeWorldPin(",
            "RigidAttachmentEntry",
            "mRigidAttachments",
            "ensureNativeIslandEdge(",
            "Scene::addAvbdCpuDeformableVolumeRigidAttachment(",
            "Scene::updateAvbdCpuDeformableVolumeRigidAttachment(",
            "Scene::removeAvbdCpuDeformableVolumeRigidAttachment(",
            "ArticulationAttachmentEntry",
            "mArticulationAttachments",
            "eARTICULATION_LINK",
            "Scene::addAvbdCpuDeformableVolumeArticulationAttachment(",
            "Scene::updateAvbdCpuDeformableVolumeArticulationAttachment(",
            "Scene::removeAvbdCpuDeformableVolumeArticulationAttachment(",
            "PrescribedAttachmentEntry",
            "mPrescribedAttachments",
            "AvbdSoftPinTargetKind::",
            "ePRESCRIBED_RIGID",
            "Scene::addAvbdCpuDeformableVolumeKinematicAttachment(",
            "Scene::updateAvbdCpuDeformableVolumeKinematicAttachment(",
            "Scene::removeAvbdCpuDeformableVolumeKinematicAttachment(",
        ),
        "CPU AVBD deformable-volume compiled world-pin ownership",
    )
    require_all(
        errors,
        np_element_filter_header + np_element_filter_source,
        (
            "cpuAvbdRigidActorFilter",
            "cpuAvbdFilterAllElements",
            "eVOLUME_TET_RIGID_BODY",
            "hasValidDeformableElementGroups(",
            "coversEveryDeformableElement(",
            "hasCpuAvbdVolumeCollisionOwnership(",
            "mTetsAccumulatedRemapColToSim",
            "mTetsRemapColToSim",
            "collision-to-simulation element ownership.",
            "addAvbdCpuDeformableVolumeRigidActorFilter(",
            "removeAvbdCpuDeformableVolumeRigidActorFilter(",
        ),
        "Np CPU AVBD exact volume rigid filter",
    )
    require_all(
        errors,
        sc_scene_header + sc_scene_source,
        (
            "RigidActorFilterEntry",
            "mRigidActorFilters",
            "filterAllElements",
            "addVolumeRigidActorFilter(",
            "mTetsAccumulatedRemapColToSim",
            "mTetsRemapColToSim",
            "simulationElements",
            "elementAdjacency",
            "tetRefs",
            "sourceElementIndex",
            "removeRigidActorFilteredContacts(",
            "findRigidCoreForPrimitive(",
            "addAvbdCpuDeformableVolumeRigidActorFilter(",
            "removeAvbdCpuDeformableVolumeRigidActorFilter(",
        ),
        "Sc CPU AVBD collision-to-simulation volume filter ownership",
    )
    require(
        errors,
        "eGPU requires the PxCudaContextManager overload" in np_physics,
        "GPU creation no longer stays on the CUDA-context overload",
    )

    require_all(
        errors,
        np_scene,
        (
            "CPU AVBD deformable volumes require PxSolverType::eAVBD",
            "CPU AVBD deformable-volume host buffers are incomplete",
            "mDeformableVolumes.insert(&deformableVolume)",
            "getNbDeformableVolumes() const",
            "getDeformableVolumes(",
            "s.addAvbdCpuDeformableVolume(",
            "s.removeAvbdCpuDeformableVolume(v.getCore())",
            "registerAvbdCpuStaticShapes(",
            "scene.addAvbdCpuStaticShape(",
            "s.removeAvbdCpuStatic(v.getCore())",
            "for(PxU32 i = 0; i < mRigidStatics.size(); i++)",
        ),
        "Np CPU Scene lifecycle/static registration",
    )
    require(
        errors,
        "CPU AVBD deformable-volume Scene ownership is not enabled"
        not in np_scene,
        "obsolete CPU Scene fail-closed path returned",
    )

    require_all(
        errors,
        sc_scene_header,
        (
            "addAvbdCpuDeformableVolume(",
            "removeAvbdCpuDeformableVolume(",
            "addAvbdCpuStaticShape(",
            "removeAvbdCpuStaticShape(",
            "removeAvbdCpuStatic(",
            "addAvbdCpuDynamicShape(",
            "removeAvbdCpuDynamicShape(",
            "removeAvbdCpuDynamic(",
            "stepAvbdCpuDeformableVolumes()",
            "prepareAvbdCpuSoftIslandGeneration()",
            "AvbdCpuSoftScene*",
        ),
        "Sc CPU Scene declaration",
    )
    require_all(
        errors,
        sc_scene_source,
        (
            "class AvbdCpuSoftScene",
            "public Dy::AvbdSoftIslandProvider",
            "Dy::avbdCreateSoftBody(",
            "Dy::avbdStepSoftBodies(",
            "syncHostInputs(",
            "writeBack(",
            "bool syncPositions = false;",
            "eSIM_POSITION_INVMASS",
            "particle.invMass =",
            "PxMax(positionInvMass.w, 0.0f)",
            "struct StaticShapeEntry",
            "struct DynamicShapeEntry",
            "struct NativeIslandEdgeEntry",
            "Dy::DeformableVolume*",
            "PX_PLACEMENT_NEW(",
            ")->~DeformableVolume();",
            "softEntry.islandNode",
            "entry.softCore == softCore",
            "void addStaticShape(StaticCore& core, ShapeCore& shape)",
            "void removeStaticShape(",
            "void removeStatic(StaticCore& core)",
            "void compileWorldStatics(",
            "void compileDynamicBoxesForIsland(",
            "Dy::AvbdSoftContactTargetKind::eRIGID_BODY",
            "eKINEMATIC_RIGID",
            "entry.core->getKinematicTarget(targetPose)",
            "box.previousCenter = previousShapeToWorld.p;",
            "box.previousRotation = previousShapeToWorld.q;",
            "globalBodyIndex - bodyStart",
            "box.shapeToRigidBody =",
            "Dy::AvbdWorldPlane plane;",
            "Dy::AvbdRigidBox box;",
            "plane.primitiveKey = entry.primitiveKey",
            "box.primitiveKey = entry.primitiveKey",
            "Dy::avbdDetectAllOGCContacts(",
            "redetectContacts, &mContacts, this",
            "Sc::Scene::addAvbdCpuDeformableVolume(",
            "Sc::Scene::removeAvbdCpuDeformableVolume(",
            "Sc::Scene::addAvbdCpuStaticShape(",
            "Sc::Scene::removeAvbdCpuStaticShape(",
            "Sc::Scene::removeAvbdCpuStatic(",
            "Sc::Scene::addAvbdCpuDynamicShape(",
            "Sc::Scene::removeAvbdCpuDynamicShape(",
            "Sc::Scene::removeAvbdCpuDynamic(",
            "Sc::Scene::stepAvbdCpuDeformableVolumes()",
            "Sc::Scene::prepareAvbdCpuSoftIslandGeneration()",
            "void prepareIslandGeneration(",
            "PxReal dt, const PxVec3& gravity, bool sleepingEnabled",
            "IG::Node::eDEFORMABLE_VOLUME_TYPE",
            "mIslandManager.activateNode(islandNode)",
            "mIslandManager.addContactManager(",
            "IG::Edge::eSOFT_BODY_CONTACT",
            "NativeSoftSoftIslandEdgeEntry",
            "ensureNativeSoftSoftIslandEdge(",
            "mNativeSoftSoftIslandEdges",
            "soft0WasSleeping && !soft1WasSleeping",
            "soft1WasSleeping && !soft0WasSleeping",
            "mIslandManager.setEdgeConnected(",
            "mIslandManager.removeConnection(",
            "removeNativeIslandEdgesForRigid(",
            "removeNativeIslandEdgesForSoft(",
            "prepareSoftIslandSelections(",
            "const PxU32* activeIslandIds",
            "const IG::IslandId entryIslandId",
            "activeIslandIds[islandIndex]",
            "struct IslandSelectionStorage",
            "acquireIslandSelectionStorage(entryIslandId)",
            "buildIslandSelectionStorage(",
            "copyAndRebaseSoftBody(",
            "rebaseSoftBodyParticleRangeInPlace(",
            "storage.globalParticleIndices.pushBack(globalIndex)",
            "selection.islandIndex = storage.selectedIsland",
            "mDynamicsSelectedEntryCount =",
            "selectedEntryCount",
            "copyIslandSelectionResults(storage)",
            "stepComponentFallback(",
            "void sleepEntry(Entry& entry)",
            "void wakeEntry(Entry& entry, PxReal wakeCounter)",
            "void updateSleepStates(",
            "core.cpuAvbdSleeping = true;",
            "core.cpuAvbdSleeping = false;",
            "core.cpuAvbdWakeRequested",
            "mIslandManager.deactivateNode(entry.islandNode)",
            "mIslandManager.activateNode(entry.islandNode)",
            "if(awakeEntryCount == 0 && sleepingEnabled)",
            "islandBodyCounts[storage.selectedIsland] > 0",
            "bodySim->isActive()",
            "rebaseSoftBodyParticleRangeInPlace(",
            "body.runtime.compileObjectiveProgram(",
            "body.runtime.isObjectiveProgramCurrent(",
            "mParticles.removeRange(",
            "removedParticleStart",
            "removedParticleCount",
            "const PxU32 remainingParticleStart =",
            "getParticleStart(remainingEntry)",
            "getParticleCount(remainingEntry)",
            "const PxU32 rebasedParticleStart",
            "mContacts.clear();",
            "mWorkspace.reset();",
        ),
        "Sc CPU Scene implementation and compact remove storage",
    )
    require(
        errors,
        "IG::Node::eDEFORMABLE_VOLUME_TYPE,\n\t\t\t\t\tthis"
        not in sc_scene_source,
        "CPU soft island node object regressed to an invalid manager pointer",
    )
    require(
        errors,
        sc_scene_source.count(
            "owner.getActorType() == PxActorType::eRIGID_STATIC"
        )
        >= 2,
        "Sc attach/detach no longer owns per-shape static registration",
    )
    require(
        errors,
        sc_scene_source.count("addAvbdCpuDynamicShape(") >= 4
        and sc_scene_source.count("removeAvbdCpuDynamicShape(") >= 2,
        "Sc regular/batch insertion or detach lost dynamic-shape registration",
    )
    require(
        errors,
        "stepAvbdCpuDeformableVolumes();" in sc_pipeline,
        "CPU deformable volumes are no longer stepped by the Scene pipeline",
    )
    require(
        errors,
        "prepareAvbdCpuSoftIslandGeneration();" in sc_pipeline
        and sc_pipeline.index("prepareAvbdCpuSoftIslandGeneration();")
        < sc_pipeline.index("mSimpleIslandManager->firstPassIslandGen();"),
        "CPU soft contact edges are no longer prepared before first-pass "
        "island generation",
    )
    require(
        errors,
        "gravityScale" in soft_component
        and "gravity * (gravityScale * dt * dt)" in soft_component
        and "PxActorFlag::eDISABLE_GRAVITY" in sc_scene_source,
        "CPU Scene lost per-actor eDISABLE_GRAVITY semantics",
    )
    require_all(
        errors,
        avbd_solver_body,
        (
            "physx::PxReal gravityScale;",
            "gravity * (gravityScale * dt)",
        ),
        "rigid AVBD per-body gravity prediction",
    )
    require(
        errors,
        "body.gravityScale = core.disableGravity ? 0.0f : 1.0f;"
        in avbd_body_conversion,
        "rigid AVBD conversion lost eDISABLE_GRAVITY",
    )

    require_all(
        errors,
        soft_component,
        (
            "struct AvbdWorldPlane",
            "inline void avbdDetectSoftWorldPlaneContacts(",
            "PxU64 primitiveKey;",
            "AvbdSoftContactTargetKind::eWORLD_STATIC",
            "AvbdSoftContactTargetKind::eRIGID_BODY",
            "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
            "AvbdVelocityObjectiveOwner::ComponentFinalize",
            "struct AvbdCompiledSoftVelocityObjective",
            "compileVelocityObjectives",
            "avbdFinalizeSoftComponentVelocities(",
            "PxTransform shapeToRigidBody;",
            "geometry.rigidLocalPoint =",
            "box.shapeToRigidBody.transform(surfaceLocal)",
            "This Scene-external component has no rigid 6x6 block",
            "inline void avbdDetectAllOGCContacts(",
            "const AvbdWorldPlane* worldPlanes = NULL",
            "bool includeLegacyGround = true",
        ),
        "low-level static contact objective IR",
    )
    require_all(
        errors,
        avbd_dynamics_header,
        (
            "struct AvbdSoftIslandSelection",
            "class AvbdSoftIslandProvider",
            "prepareSoftIslandSelections(",
            "bool isComplete() const",
            "setSoftIslandProvider(",
            "mSoftIslandProvider",
        ),
        "main AVBD soft-island provider contract",
    )
    require_all(
        errors,
        avbd_dynamics_source,
        (
            "PxArray<AvbdSoftIslandSelection> softSelections;",
            "mSoftIslandProvider->prepareSoftIslandSelections(",
            "selection.isComplete()",
            "softSelections[selectionIndex].islandIndex == i",
            "batch.softParticles =",
            "batch.softBodies =",
            "batch.softContacts =",
            "ownsSoftSelection ? softSelection->iterationOverride : 0",
            "selection.islandIndex >= islandCount || !uniqueIsland",
            "info.bodyCount == 0 && !ownsSoftSelection",
            "for (PxU32 i = 0; i < islandCount; ++i)",
            "including native",
            "soft-only island records uninitialized",
        ),
        "main AVBD island gather soft tuple routing",
    )

    require_all(
        errors,
        snippet,
        (
            'caseName == "scene-volume-lifecycle"',
            'caseName == "scene-volume-ground"',
            'caseName == "scene-volume-static-box"',
            'caseName == "scene-volume-static-churn"',
            'caseName == "scene-volume-dynamic-box"',
            'caseName == "scene-volume-dynamic-churn"',
            'caseName == "scene-volume-multi-dynamic-box"',
            'caseName == "scene-volume-multi-soft-islands"',
            'caseName == "scene-volume-sleep-wake"',
            'caseName == "scene-volume-rigid-wake"',
            'caseName == "scene-volume-mixed-sleep-islands"',
            'caseName == "scene-volume-soft-churn"',
            'caseName == "scene-volume-buffer-mutation"',
            'caseName == "scene-volume-multi-scene-isolation"',
            'caseName == "scene-volume-soft-soft-wake"',
            'caseName == "scene-volume-world-pin"',
            'caseName == "scene-volume-element-filter"',
            'caseName == "scene-volume-partial-element-filter"',
            'caseName == "scene-volume-kinematic-box"',
            "initSceneCpuVolumeLifecycle()",
            "createAdditionalSceneCpuVolume(",
            "addSceneStaticBox(",
            "addSceneDynamicBox(",
            "addSceneSecondDynamicBox(",
            "updateSceneDynamicChurn()",
            "updateSceneMultiDynamicGate()",
            "updateSceneSoftActorChurn()",
            "updateSceneSoftBufferMutation()",
            "updateSceneSoftSoftWake()",
            "updateSceneVolumeWorldPin()",
            "updateSceneVolumeElementFilter()",
            "addSceneKinematicBox(",
            "updateSceneKinematicBox()",
            "stepSceneMultiSceneIsolation(",
            "createDeformableAttachment(attachmentData)",
            "createDeformableElementFilter(filterData)",
            "gPhysics->createScene(sceneDesc)",
            "gMetrics.scenePrimarySceneReleased = 1",
            "cookingParams.buildGPUData = false",
            "PxDeformableVolumeBackend::eCPU_AVBD",
            "getSimPositionInvMassBufferH()",
            "gScene->addActor(*gSceneCpuVolume)",
            "PxCreatePlane(",
            "gScene->removeActor(*gSceneCpuVolume)",
            "gMetrics.sceneActorRemoved = 1",
            "gMetrics.sceneActorReleased = 1",
            "SCENE_LIFECYCLE_GATED",
            "SCENE_STATIC_CONTACT_GATED",
            "SCENE_STATIC_LIFECYCLE_GATED",
            "SCENE_DYNAMIC_COUPLING_GATED",
            "SCENE_DYNAMIC_LIFECYCLE_GATED",
            "SCENE_MULTI_SOFT_ISLANDS_GATED",
            "SCENE_SOFT_SLEEP_WAKE_GATED",
            "SCENE_SOFT_RIGID_WAKE_GATED",
            "SCENE_MIXED_SLEEP_ISLANDS_GATED",
            "SCENE_SOFT_CHURN_GATED",
            "SCENE_BUFFER_MUTATION_GATED",
            "SCENE_MULTI_SCENE_ISOLATION_GATED",
            "SCENE_SOFT_SOFT_WAKE_GATED",
            "SCENE_WORLD_PIN_GATED",
            "SCENE_ELEMENT_FILTER_GATED",
            "SCENE_PARTIAL_ELEMENT_FILTER_GATED",
            "SCENE_KINEMATIC_COUPLING_GATED",
            "gMetrics.sceneSoftFirstSlept == 1",
            "gMetrics.sceneSoftWokeByCounter == 1",
            "gMetrics.sceneSoftWokeByVelocity == 1",
            "gMetrics.sceneSoftFinalSlept == 1",
            "gMetrics.sceneSoftWokeByRigid == 1",
            "gMetrics.sceneMixedFirstSlept == 1",
            "gMetrics.sceneMixedFirstStable == 1",
            "gMetrics.sceneMixedSecondStayedAwake == 1",
            "gMetrics.sceneBufferMutationWoke == 1",
            "gMetrics.sceneBufferPinHeld == 1",
            "gMetrics.sceneBufferDynamicMoved == 1",
            "gMetrics.sceneBufferRestoredMoved == 1",
            "gMetrics.sceneSecondSceneCreated == 1",
            "gMetrics.sceneSecondSceneSolverMatched == 1",
            "gMetrics.scenePrimarySceneReleased == 1",
            "gMetrics.sceneSecondSceneReleased == 1",
            "gMetrics.sceneMultiPrimaryStable == 1",
            "gMetrics.sceneMultiPrimaryDetachedStable == 1",
            "gMetrics.sceneMultiSecondaryUpdatedBeforeRelease == 1",
            "gMetrics.sceneMultiSecondaryUpdatedAfterRelease == 1",
            "gMetrics.sceneSoftSoftBothSlept == 1",
            "gMetrics.sceneSoftSoftDriveIssued == 1",
            "gMetrics.sceneSoftSoftDriverWoke == 1",
            "gMetrics.sceneSoftSoftTargetWoke == 1",
            "gMetrics.sceneSoftSoftTargetMoved == 1",
            "gMetrics.sceneSoftSoftResetIssued == 1",
            "gMetrics.sceneSoftSoftBothFinalSlept == 1",
            "gMetrics.sceneWorldPinCreated == 1",
            "gMetrics.sceneWorldPinHeld == 1",
            "gMetrics.sceneWorldPinActorReadded == 1",
            "gMetrics.sceneWorldPinReleased == 1",
            "gMetrics.sceneWorldPinMovedAfterRelease == 1",
            "gMetrics.sceneWorldPinMaxDrift <= 1.0e-4f",
            "sceneWorldPinReleasedMaxDisplacement >",
            "gMetrics.sceneElementFilterCreated == 1",
            "gMetrics.sceneElementFilterActorReadded == 1",
            "sceneElementFilterSuppressedContact == 1",
            "gMetrics.sceneElementFilterReleased == 1",
            "gMetrics.sceneElementFilterContactRestored == 1",
            "gMetrics.sceneElementFilterMinY < -0.2f",
            "gMetrics.sceneElementFilterFinalMinY > -0.05f",
            "scenePartialFilterUnfilteredContactHeld =",
            "scenePartialFilterExactOwnership =",
            "gMetrics.scenePartialFilterUnfilteredMinY > -0.05f",
            "gMetrics.sceneMixedSecondMoved == 1",
            "gMetrics.sceneSoftChurnRemoveCount ==",
            "gMetrics.sceneSoftChurnReaddCount ==",
            "gMetrics.sceneSoftChurnPostCompactMoveCount ==",
            "gMetrics.sceneSoftChurnStable == 1",
            "actor->putToSleep()",
            "gMetrics.sceneDynamicInitiallySleeping == 1",
            "gMetrics.sceneDynamicWokeBySoft == 1",
            "gMetrics.sceneDynamicFirstWakeFrame != PX_MAX_U32",
            "gMetrics.sceneDynamicShapeDetached == 1",
            "gMetrics.sceneDynamicShapeReattached == 1",
            "gMetrics.sceneDynamicActorReadded == 1",
            "gMetrics.sceneDynamicReaddedSleeping == 1",
            "gMetrics.sceneDynamicRewokeBySoft == 1",
            "gMetrics.sceneDynamicSecondWakeFrame >",
            "gMetrics.sceneSecondDynamicInitiallySleeping == 1",
            "gMetrics.sceneSecondDynamicWokeBySoft == 1",
            "gMetrics.sceneSecondDynamicFirstWakeFrame ==",
            "gMetrics.sceneSecondVolumeActorCreated == 1",
            "gMetrics.sceneSecondVolumeActorRemoved == 1",
            "gMetrics.sceneSecondVolumeActorReleased == 1",
            "gMetrics.sceneSecondVolumeBoundsFinite == 1",
            "sceneSecondVolumeMaxCentroidDrop > 0.5f",
            "PxDeformableVolumeDataFlag::eSIM_VELOCITY",
            "gMetrics.finalMaxParticleSpeed < 1.0e-4f",
            "sceneSoftIntegration=%u",
            "gMetrics.groundContactFrames > 0",
            "gMetrics.maxGroundContacts > 0",
            "gMetrics.rigidContactFrames > 0",
            "gMetrics.maxRigidContacts > 0",
            "gMetrics.finalMinY > 0.7f",
            "gMetrics.sceneStaticShapeDetached == 1",
            "gMetrics.sceneStaticShapeReattached == 1",
            "gMetrics.sceneStaticActorRemoved == 1",
            "gMetrics.sceneStaticActorReadded == 1",
            "gMetrics.sceneDynamicMaxDrop > 0.05f",
            "dynamicCapsuleCase ||",
            "gMetrics.sceneDynamicPreContactMaxDrop <",
            "1.0e-4f",
            "gMetrics.minDynamicSurfaceSeparation > -0.15f",
            "gMetrics.finalDynamicSurfaceSeparation > -0.15f",
            "gMetrics.sceneKinematicActorAdded == 1",
            "gMetrics.sceneKinematicTargetIssued == 1",
            "gMetrics.sceneKinematicTargetReached == 1",
            "gMetrics.sceneKinematicSoftWoke == 1",
            "gMetrics.sceneKinematicSoftMoved == 1",
            "gMetrics.sceneKinematicContactObserved == 1",
            "gMetrics.sceneKinematicMaxPoseError <= 1.0e-4f",
            "gMetrics.sceneKinematicSoftDisplacement > 0.02f",
            "gMetrics.cleanupComplete",
        ),
        "SnippetDeformableVolumeAVBD public Scene gate",
    )
    require_all(
        errors,
        runner,
        (
            '"scene-volume-lifecycle"',
            '"scene-volume-ground"',
            '"scene-volume-static-box"',
            '"scene-volume-static-churn"',
            '"scene-volume-dynamic-box"',
            '"scene-volume-dynamic-churn"',
            '"scene-volume-multi-dynamic-box"',
            '"scene-volume-multi-soft-islands"',
            '"scene-volume-sleep-wake"',
            '"scene-volume-rigid-wake"',
            '"scene-volume-mixed-sleep-islands"',
            '"scene-volume-soft-churn"',
            '"scene-volume-buffer-mutation"',
            '"scene-volume-multi-scene-isolation"',
            '"scene-volume-soft-soft-wake"',
            '"scene-volume-world-pin"',
            '"scene-volume-element-filter"',
            '"scene-volume-partial-element-filter"',
            '"scene-volume-kinematic-box"',
            '"SCENE_LIFECYCLE_GATED"',
            '"SCENE_STATIC_CONTACT_GATED"',
            '"SCENE_STATIC_LIFECYCLE_GATED"',
            '"SCENE_DYNAMIC_COUPLING_GATED"',
            '"SCENE_DYNAMIC_LIFECYCLE_GATED"',
            '"SCENE_MULTI_DYNAMIC_COUPLING_GATED"',
            '"SCENE_MULTI_SOFT_ISLANDS_GATED"',
            '"SCENE_SOFT_SLEEP_WAKE_GATED"',
            '"SCENE_SOFT_RIGID_WAKE_GATED"',
            '"SCENE_MIXED_SLEEP_ISLANDS_GATED"',
            '"SCENE_SOFT_CHURN_GATED"',
            '"SCENE_BUFFER_MUTATION_GATED"',
            '"SCENE_MULTI_SCENE_ISOLATION_GATED"',
            '"SCENE_SOFT_SOFT_WAKE_GATED"',
            '"SCENE_WORLD_PIN_GATED"',
            '"SCENE_ELEMENT_FILTER_GATED"',
            '"SCENE_PARTIAL_ELEMENT_FILTER_GATED"',
            '"SCENE_KINEMATIC_COUPLING_GATED"',
            '"sceneSoftFirstSlept": "1"',
            '"sceneSoftWokeByCounter": "1"',
            '"sceneSoftWokeByVelocity": "1"',
            '"sceneSoftFinalSlept": "1"',
            '"sceneSoftWokeByRigid": "1"',
            '"rigid wake response exceeded the bounded speed gate"',
            '"sceneMixedFirstSlept": "1"',
            '"sceneMixedFirstStable": "1"',
            '"sceneMixedSecondStayedAwake": "1"',
            '"sceneMixedSecondMoved": "1"',
            '"sceneSoftChurnRemoveCount": str(expected_events)',
            '"sceneSoftChurnReaddCount": str(expected_events)',
            '"sceneSoftChurnCycles": str(expected_cycles)',
            '"sceneSoftChurnPostCompactMoveCount": str(expected_events)',
            '"sceneSoftChurnStable": "1"',
            '"sceneBufferMutationWoke": "1"',
            '"sceneBufferPinHeld": "1"',
            '"sceneBufferDynamicMoved": "1"',
            '"sceneBufferRestoredMoved": "1"',
            '"sceneSecondSceneCreated": "1"',
            '"sceneMultiSecondaryUpdatedAfterRelease": "1"',
            '"sceneSoftSoftBothSlept": "1"',
            '"sceneSoftSoftDriveIssued": "1"',
            '"sceneSoftSoftDriverWoke": "1"',
            '"sceneSoftSoftTargetWoke": "1"',
            '"sceneSoftSoftTargetMoved": "1"',
            '"sceneSoftSoftResetIssued": "1"',
            '"sceneSoftSoftBothFinalSlept": "1"',
            '"sceneWorldPinCreated": "1"',
            '"sceneWorldPinHeld": "1"',
            '"sceneWorldPinActorReadded": "1"',
            '"sceneWorldPinReleased": "1"',
            '"sceneWorldPinMovedAfterRelease": "1"',
            '"volume world-attached vertex drifted"',
            '"released volume world-attached vertex did not move"',
            '"sceneElementFilterCreated": "1"',
            '"sceneElementFilterActorReadded": "1"',
            '"sceneElementFilterSuppressedContact": "1"',
            '"sceneElementFilterReleased": "1"',
            '"sceneElementFilterContactRestored": "1"',
            '"sceneElementFilterMinY"',
            '"sceneElementFilterFinalMinY"',
            '"volume element filter did not suppress rigid contact"',
            '"volume rigid contact did not recover after filter release"',
            '"scenePartialFilterUnfilteredContactHeld"',
            '"scenePartialFilterExactOwnership"',
            '"scenePartialFilterUnfilteredMinY"',
            '"unfiltered volume component lost rigid contact"',
            '"partial volume filter ownership was not exact"',
            '"unfiltered volume component penetrated the ground"',
            '"sceneKinematicActorAdded": "1"',
            '"sceneKinematicTargetIssued": "1"',
            '"sceneKinematicTargetReached": "1"',
            '"sceneKinematicSoftWoke": "1"',
            '"sceneKinematicSoftMoved": "1"',
            '"sceneKinematicContactObserved": "1"',
            '"sceneKinematicMaxPoseError"',
            '"sceneKinematicSoftDisplacement"',
            '"sceneKinematicFinalY"',
            '"kinematic coupling amplified the volume speed"',
            '"kinematic coupling left excessive residual speed"',
            '"sceneSoftIntegration": "1" if scene_integrated else "0"',
            '"sceneActorRemoved": "1"',
            '"sceneActorReleased": "1"',
            '"cleanupComplete": "1"',
            'for key in ("groundContactFrames", "maxGroundContacts")',
            'for key in ("rigidContactFrames", "maxRigidContacts")',
            'float(fields["finalMinY"]) <= -0.1',
            'float(fields["finalMinY"]) <= 0.7',
            '"sceneStaticShapeDetached"',
            '"sceneStaticShapeReattached"',
            '"sceneStaticActorRemoved"',
            '"sceneStaticActorReadded"',
            '"sceneDynamicActorAdded": "1"',
            '"sceneDynamicActorReleased": "1"',
            '"sceneDynamicInitiallySleeping": "1"',
            '"sceneDynamicWokeBySoft": "1"',
            '"sceneDynamicFirstWakeFrame"',
            '"sceneDynamicShapeDetached": "1"',
            '"sceneDynamicShapeReattached": "1"',
            '"sceneDynamicActorReadded": "1"',
            '"sceneDynamicReaddedSleeping": "1"',
            '"sceneDynamicRewokeBySoft": "1"',
            '"sceneDynamicSecondWakeFrame"',
            '"sceneSecondDynamicActorAdded": "1"',
            '"sceneSecondDynamicActorRemoved": "1"',
            '"sceneSecondDynamicActorReleased": "1"',
            '"sceneSecondDynamicInitiallySleeping": "1"',
            '"sceneSecondDynamicWokeBySoft": "1"',
            '"sceneSecondDynamicFirstWakeFrame"',
            '"multi-dynamic targets did not wake in one island pass"',
            '"sceneSecondVolumeActorCreated": "1"',
            '"sceneSecondVolumeActorRemoved": "1"',
            '"sceneSecondVolumeActorReleased": "1"',
            '"sceneSecondVolumeBoundsFinite": "1"',
            'float(fields["sceneSecondVolumeMaxCentroidDrop"]) <= 0.5',
            'float(fields["finalMaxParticleSpeed"]) >= 1.0e-4',
            'float(fields["sceneDynamicMaxDrop"]) <= 0.05',
            'case_name\n                not in (',
            '"scene-volume-dynamic-capsule",',
            '"scene-volume-dynamic-convex",',
            'float(fields["sceneDynamicPreContactMaxDrop"])',
            ">= 1.0e-4",
            'float(fields["minDynamicSurfaceSeparation"]) <= -0.15',
        ),
        "dedicated deformable-volume headless runner",
    )
    require_all(
        errors,
        unit_snippet,
        (
            "--- Test 30: CPU AVBD Deformable Volume Scene Lifecycle ---",
            "CPU AVBD Scene owns multiple deformable volumes",
            "Removing the first CPU AVBD body preserves the second",
            "Remaining CPU AVBD body keeps stepping after middle removal",
            "CPU AVBD deformable volume can re-enter the Scene",
            "CPU AVBD Scene honors eDISABLE_GRAVITY after re-entry",
            "CPU AVBD Scene re-entry leaves no stale ownership",
            "CPU AVBD multi-island fixture owns both host tuples",
            "Independent CPU AVBD soft islands each own a complete dynamic contact tuple",
        ),
        "Test30 multi-volume Scene lifecycle fixture",
    )
    require(
        errors,
        "tuple(range(1, 37))" in unit_runner
        and "choices=range(1, 37)" in unit_runner,
        "soft-body runner no longer requires Tests 1..36",
    )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_DEFORMABLE_VOLUME_LIFECYCLE_SOURCE_GATE_ERROR] "
                f"{error}"
            )
        print(
            "[AVBD_CPU_DEFORMABLE_VOLUME_LIFECYCLE_SOURCE_GATE] status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_DEFORMABLE_VOLUME_LIFECYCLE_SOURCE_GATE] "
        "status=PASS slices=scene-lifecycle,scene-static-plane,"
        "scene-static-box,scene-static-churn,scene-dynamic-box,"
        "scene-dynamic-churn,scene-multi-dynamic,scene-multi-soft,"
        "scene-soft-sleep-wake,scene-soft-rigid-wake,"
        "scene-mixed-sleep-islands,scene-soft-churn,"
        "scene-buffer-mutation,scene-multi-scene-isolation,"
        "scene-soft-soft-wake,scene-world-pin,scene-element-filter,"
        "scene-partial-element-filter,scene-kinematic-box "
        "backend=explicit storage=host scene=owned "
        "contactOwner=position-al-or-component-finalize-ir "
        "dynamicCoupling=native-island-partitioned-complete-tuples"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
