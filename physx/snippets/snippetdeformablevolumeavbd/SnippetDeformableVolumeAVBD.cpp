// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// ****************************************************************************
// SnippetDeformableVolumeAVBD
//
// CPU-only AVBD equivalent of SnippetDeformableVolume (GPU FEM).
// Demonstrates multiple VBD soft bodies -- a cube, a sphere, and a tall
// cube (cone substitute) -- dropping onto a rigid ground plane.  All elastic
// forces use Neo-Hookean energy via VBD; contacts (ground, soft-soft,
// soft-rigid) are enforced through AVBD adaptive penalty.
//
// Scene layout:
//   Body 0 : cuboid at (-1.8, 8.0, 0.0) -- tilted, falls onto sphere edge and spins
//   Body 1 : sphere at (-3.8, 2.0, 0.0) -- restored visual anchor for soft-soft collision
//   Body 2 : cone   at (-0.8,11.0, 1.2) -- glancing hit into the left stack
//   Body 3 : cuboid at ( 7.0, 4.2, 0.0) -- tilted on a narrow rigid box edge
//   Body 4 : cube   at ( 5.4, 8.8, 0.3) -- off-center follower amplifies body 3 rotation
//   Rigid  : box    at ( 7.6, 0.55,0.0) -- narrow step, uses SDF contact path
//
// No GPU or CUDA dependency -- runs entirely on the CPU.
// ****************************************************************************

#include <cstdio>
#include <cmath>
#include "PxPhysicsAPI.h"
#include "cooking/PxCooking.h"
#include "DyAvbdSoftBodyComponent.h"
#include "extensions/PxDeformableVolumeExt.h"
#include "extensions/PxTetMakerExt.h"

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetDeformableAVBDSkinning.h"

#include "SnippetDeformableVolumeAVBD.h"

#include <cfloat>
#include <cstring>
#include <string>

using namespace physx;
using namespace physx::Dy;

#ifndef AVBD_VOLUME_SNIPPET_NAME
#define AVBD_VOLUME_SNIPPET_NAME "SnippetDeformableVolumeAVBD"
#endif

#ifndef AVBD_VOLUME_DEFAULT_CASE
#define AVBD_VOLUME_DEFAULT_CASE "current-all"
#endif

#ifndef AVBD_VOLUME_VISUAL_CASE
#define AVBD_VOLUME_VISUAL_CASE "current-all"
#endif

// ---------------------------------------------------------------------------
// Generate cone surface triangles, then use PxTetMaker conforming->voxel
// pipeline to produce a uniform voxel tet mesh.
// ---------------------------------------------------------------------------
static void rotateVerticesAroundZ(
	PxArray<PxVec3>& verts,
	const PxVec3& center,
	PxReal angle)
{
	const PxReal cs = PxCos(angle);
	const PxReal sn = PxSin(angle);
	for (PxU32 i = 0; i < verts.size(); i++)
	{
		const PxVec3 r = verts[i] - center;
		verts[i].x = center.x + r.x * cs - r.y * sn;
		verts[i].y = center.y + r.x * sn + r.y * cs;
	}
}

static void scaleVerticesAboutCenter(
	PxArray<PxVec3>& verts,
	const PxVec3& center,
	const PxVec3& scale)
{
	for (PxU32 i = 0; i < verts.size(); i++)
	{
		const PxVec3 r = verts[i] - center;
		verts[i] = center + PxVec3(r.x * scale.x, r.y * scale.y, r.z * scale.z);
	}
}

static void generateConeTetsViaTetMaker(
	const PxVec3& center, PxReal radius, PxReal height,
	PxU32 numVoxels,
	PxArray<PxVec3>& outVerts, PxArray<PxU32>& outTets)
{
	// Build a cone surface mesh (triangle fan base + lateral)
	const PxU32 N = 16; // ring segments
	PxArray<PxVec3> surfVerts;
	PxArray<PxU32>  surfTris;

	// vertex 0 = apex
	surfVerts.pushBack(center + PxVec3(0, height, 0));
	// vertices 1..N = base ring
	for (PxU32 i = 0; i < N; i++)
	{
		PxReal a = 2.0f * 3.14159265f * i / N;
		surfVerts.pushBack(center + PxVec3(radius * cosf(a), 0, radius * sinf(a)));
	}
	// vertex N+1 = base center
	surfVerts.pushBack(center);

	// Lateral triangles (apex -> ring[i] -> ring[i+1])
	for (PxU32 i = 0; i < N; i++)
	{
		surfTris.pushBack(0);
		surfTris.pushBack(1 + i);
		surfTris.pushBack(1 + (i + 1) % N);
	}
	// Base triangles (center -> ring[i+1] -> ring[i])
	for (PxU32 i = 0; i < N; i++)
	{
		surfTris.pushBack(N + 1);
		surfTris.pushBack(1 + (i + 1) % N);
		surfTris.pushBack(1 + i);
	}

	// Step 1: conforming tet mesh from surface
	PxArray<PxVec3> confVerts;
	PxArray<PxU32>  confTets;
	{
		PxSimpleTriangleMesh surfMesh;
		surfMesh.points.count  = surfVerts.size();
		surfMesh.points.data   = surfVerts.begin();
		surfMesh.points.stride = sizeof(PxVec3);
		surfMesh.triangles.count  = surfTris.size() / 3;
		surfMesh.triangles.data   = surfTris.begin();
		surfMesh.triangles.stride = sizeof(PxU32) * 3;

		if (!PxTetMaker::createConformingTetrahedronMesh(surfMesh, confVerts, confTets))
		{
			printf("TetMaker: conforming mesh failed, falling back to hand-made cone\n");
			avbdGenerateConeTets(center, radius, height, 4, outVerts, outTets);
			return;
		}
	}

	// Step 2: voxel tet mesh from the conforming mesh
	{
		PxTetrahedronMeshDesc meshDesc;
		meshDesc.points.count  = confVerts.size();
		meshDesc.points.data   = confVerts.begin();
		meshDesc.points.stride = sizeof(PxVec3);
		meshDesc.tetrahedrons.count  = confTets.size() / 4;
		meshDesc.tetrahedrons.data   = confTets.begin();
		meshDesc.tetrahedrons.stride = sizeof(PxU32) * 4;

		if (!PxTetMaker::createVoxelTetrahedronMesh(meshDesc, numVoxels,
				outVerts, outTets))
		{
			printf("TetMaker: voxel mesh failed, falling back to hand-made cone\n");
			avbdGenerateConeTets(center, radius, height, 4, outVerts, outTets);
			return;
		}
	}

	printf("TetMaker voxel cone: %u verts, %u tets\n",
		outVerts.size(), outTets.size() / 4);
}

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static PxDefaultAllocator      gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*           gFoundation  = NULL;
static PxPhysics*              gPhysics     = NULL;
static PxDefaultCpuDispatcher* gDispatcher  = NULL;
static PxMaterial*             gMaterial    = NULL;
static PxPvd*                  gPvd         = NULL;
static bool                    gExtensionsInitialized = false;
static Snippets::HeadlessOptions gHeadlessOptions;

PxScene*                       gScene       = NULL;
static PxScene*                gSceneCpuSecondScene = NULL;

PxArray<AvbdSoftParticle>      gParticles;
PxArray<AvbdSoftBody>          gSoftBodies;
PxArray<SoftBodyRenderData>    gSoftBodyRenderData;
VolumeAvbdSkinningRenderData   gVolumeAvbdSkinningRenderData;

static PxArray<AvbdSoftContact> gContacts;
static PxArray<AvbdRigidBox>     gRigidBoxes;
static AvbdSoftBodyWorkspace     gSoftWorkspace;
static PxDeformableVolume*       gSceneCpuVolume = NULL;
static PxDeformableVolume*       gSceneCpuSecondVolume = NULL;
static PxDeformableAttachment*   gSceneCpuWorldAttachment = NULL;
static PxDeformableAttachment*   gSceneCpuRigidAttachment = NULL;
static PxDeformableElementFilter* gSceneCpuElementFilter = NULL;
static PxDeformableVolumeMesh*   gSceneCpuVolumeMesh = NULL;
static PxDeformableVolumeMaterial*
								   gSceneCpuVolumeMaterial = NULL;
static PxRigidStatic*               gSceneCpuStaticActor = NULL;
static PxRigidDynamic*              gSceneCpuDynamicActor = NULL;
static PxRigidDynamic*              gSceneCpuSecondDynamicActor = NULL;
static PxConvexMesh*                gSceneCpuRigidConvexMesh = NULL;
static PxTriangleMesh*              gSceneCpuRigidTriangleMesh = NULL;
static PxHeightField*               gSceneCpuRigidHeightField = NULL;
static PxArticulationReducedCoordinate*
								   gSceneCpuAttachmentArticulation = NULL;
static PxArticulationLink*           gSceneCpuAttachmentRoot = NULL;
static PxArticulationLink*           gSceneCpuAttachmentLink = NULL;
static PxRigidBody*                  gSceneCpuAttachmentBody = NULL;
static PxReal                     gSceneCpuVolumeInitialCentroidY = 0.0f;
static PxReal                     gSceneCpuSecondVolumeInitialCentroidY = 0.0f;
static PxReal                     gSceneCpuDepenetrationInitialMinY =
								   0.0f;
static PxReal                     gSceneCpuDepenetrationControlInitialMinY =
								   0.0f;
static PxVec3                     gSceneCpuMotionInitialCentroid(0.0f);
static PxReal                     gSceneCpuDynamicInitialY = 0.0f;
static PxReal                     gSceneCpuSecondDynamicInitialY = 0.0f;
static PxReal                     gSceneCpuKinematicCommandY = 0.0f;
static PxReal                     gSceneCpuKinematicSoftBaselineY = 0.0f;
static PxVec3                     gSceneCpuMovingSpherePositiveInitial(
	0.0f);
static PxVec3                     gSceneCpuMovingSphereNegativeInitial(
	0.0f);
static PxVec3                     gSceneCpuSphereReversePositiveInitial(
	0.0f);
static PxVec3                     gSceneCpuSphereReverseNegativeInitial(
	0.0f);
static PxReal                     gSceneCpuFirstSleepCentroidY = 0.0f;
static PxReal                     gSceneCpuSecondSleepCentroidY = 0.0f;
static PxReal                     gSceneCpuVelocityWakeCentroidY = 0.0f;
static PxReal                     gSceneCpuRigidWakeCentroidY = 0.0f;
static PxReal                     gSceneCpuSoftChurnCentroidY = 0.0f;
static bool                       gSceneCpuSoftChurnMovePending = false;
static PxVec3                     gSceneCpuBufferPinTarget(0.0f);
static PxVec3                     gSceneCpuBufferDynamicBaseline(0.0f);
static PxVec3                     gSceneCpuBufferRestoredBaseline(0.0f);
static PxReal                     gSceneCpuBufferOriginalInvMass = 0.0f;
static PxReal                     gSceneCpuMultiPrimaryRemovedCentroidY = 0.0f;
static PxReal                     gSceneCpuMultiSecondaryAtReleaseCentroidY = 0.0f;
static PxVec3                     gSceneCpuSoftSoftTargetBaseline(0.0f);
static PxVec3                     gSceneCpuWorldPinTarget(0.0f);
static PxArray<PxVec4>            gSceneCpuVolumeKinematicTargets;
static PxArray<PxVec3>            gSceneCpuVolumeKinematicInitial;
static PxArray<PxVec3>            gSceneCpuSphereReverseSweptInitialPositions;
static PxArray<PxVec3>
	gSceneCpuDeformingReverseSweptFreeEndPositions;
static PxArray<PxVec3>
	gSceneCpuCapsuleRotationalSweptInitialPositions;
static PxU32                      gSceneCpuVolumePartialProbe =
	PX_MAX_U32;
static PxVec3                     gSceneCpuVolumePartialActivationStart(
	0.0f);
static bool                       gSceneCpuElementAttachment = false;
static PxU32                      gSceneCpuAttachmentVertices[4] =
	{0, 0, 0, 0};
static PxVec4                     gSceneCpuAttachmentBarycentric(
	0.1f, 0.2f, 0.3f, 0.4f);
static PxVec3                     gSceneCpuRigidAttachmentInitialPosition(
	0.0f);
static PxVec3                     gSceneCpuRigidAttachmentLocalOffset(
	0.0f);
static PxReal                     gSceneCpuKinematicAttachmentProgress = 0.0f;
static PxVec3                     gSceneCpuKinematicAttachmentSoftBaseline(
	0.0f);
static PxTransform                gSceneCpuKinematicAttachmentCommand(
	PxIdentity);
static PxTransform                gSceneCpuArticulationRootInitialPose(
	PxIdentity);
static PxTransform                gSceneCpuArticulationChildInitialPose(
	PxIdentity);
static bool                       gSceneCpuPartialFilterSelectedPositiveX =
	false;
static PxArray<Snippets::AvbdTetrahedronSkinningBinding>
								   gVolumeSkinningBindings;
static PxArray<PxU32>              gVolumeSkinningTriangles;
static PxArray<PxVec3>             gVolumeSkinningPositions;
static PxArray<PxVec3>             gVolumeSkinningNormals;
static PxArray<PxVec3>             gVolumeSkinningInitialPositions;

struct VolumeSkinningMetrics
{
	PxU32 initialized;
	PxU32 finiteFrames;
	PxU32 evaluatedFrames;
	PxU32 vertices;
	PxU32 triangles;
	PxReal maxDisplacement;

	VolumeSkinningMetrics()
		: initialized(0), finiteFrames(0), evaluatedFrames(0),
		  vertices(0), triangles(0), maxDisplacement(0.0f)
	{
	}
};

static VolumeSkinningMetrics gVolumeSkinningMetrics;

struct SphereReverseFeatureMetrics
{
	PxU32 faceResponseObserved;
	PxU32 vertexSdfExcluded;
	PxU32 negativeControlPassed;
	PxU32 nonFiniteSamples;
	PxReal positiveDisplacement;
	PxReal positiveDrop;
	PxReal negativeDrop;
	PxReal faceSeparation;
	PxReal minimumVertexSeparation;

	SphereReverseFeatureMetrics()
		: faceResponseObserved(0), vertexSdfExcluded(0),
		  negativeControlPassed(0), nonFiniteSamples(0),
		  positiveDisplacement(0.0f), positiveDrop(0.0f),
		  negativeDrop(0.0f), faceSeparation(PX_MAX_F32),
		  minimumVertexSeparation(PX_MAX_F32)
	{
	}
};

static SphereReverseFeatureMetrics gSphereReverseFeatureMetrics;

struct SphereReverseSweptMetrics
{
	PxU32 responseObserved;
	PxU32 negativeControlPassed;
	PxU32 twoSidedResponseObserved;
	PxU32 vertexSweepExcluded;
	PxU32 nonFiniteSamples;
	PxReal positiveDisplacement;
	PxReal negativeDisplacement;
	PxReal positiveDrop;
	PxReal negativeDrop;
	PxReal positiveRigidDrop;
	PxReal negativeRigidDrop;
	PxReal faceSeparation;
	PxReal minimumVertexSweepSeparation;

	SphereReverseSweptMetrics()
		: responseObserved(0), negativeControlPassed(0),
		  twoSidedResponseObserved(0), vertexSweepExcluded(0),
		  nonFiniteSamples(0), positiveDisplacement(0.0f),
		  negativeDisplacement(0.0f), positiveDrop(0.0f),
		  negativeDrop(0.0f), positiveRigidDrop(0.0f),
		  negativeRigidDrop(0.0f), faceSeparation(PX_MAX_F32),
		  minimumVertexSweepSeparation(PX_MAX_F32)
	{
	}
};

static SphereReverseSweptMetrics gSphereReverseSweptMetrics;

struct DeformingVolumeReverseSweptMetrics
{
	PxU32 geometricSweepIsolated;
	PxReal endpointMinSeparation;
	PxReal midSweepMinSeparation;
	PxReal responseDelta;

	DeformingVolumeReverseSweptMetrics()
		: geometricSweepIsolated(0),
		  endpointMinSeparation(PX_MAX_F32),
		  midSweepMinSeparation(PX_MAX_F32),
		  responseDelta(0.0f)
	{
	}
};

static DeformingVolumeReverseSweptMetrics
	gDeformingVolumeReverseSweptMetrics;

struct CapsuleRotationalSweepMetrics
{
	PxU32 sweepIsolated;
	PxU32 nonFiniteSamples;
	PxReal endpointMinSeparation;
	PxReal midSweepMinSeparation;
	PxReal positiveAngularTravel;
	PxReal negativeAngularTravel;

	CapsuleRotationalSweepMetrics()
		: sweepIsolated(0), nonFiniteSamples(0),
		  endpointMinSeparation(PX_MAX_F32),
		  midSweepMinSeparation(PX_MAX_F32),
		  positiveAngularTravel(0.0f),
		  negativeAngularTravel(0.0f)
	{
	}
};

static CapsuleRotationalSweepMetrics
	gCapsuleRotationalSweepMetrics;

struct DeformableVolumeMetrics
{
	PxU32 initialized;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFiniteParticleSamples;
	PxU32 invertedElementSamples;
	PxU32 firstInversionFrame;
	PxU32 firstInversionBody;
	PxU32 firstInversionElement;
	PxU32 invertedBodiesMask;
	PxU32 particles;
	PxU32 softBodies;
	PxU32 tetElements;
	PxU32 surfaceTriangles;
	PxU32 rigidBoxes;
	PxU32 sceneStatics;
	PxU32 sceneDynamics;
	PxU32 sceneDeformableVolumes;
	PxU32 groundContactFrames;
	PxU32 rigidContactFrames;
	PxU32 softContactFrames;
	PxU32 maxGroundContacts;
	PxU32 maxRigidContacts;
	PxU32 maxSoftContacts;
	PxU32 invalidContactSourceSamples;
	PxU32 finalInsideParticles;
	PxU32 cleanupComplete;
	PxU32 sceneActorCreated;
	PxU32 sceneShapeAttached;
	PxU32 sceneSimulationMeshAttached;
	PxU32 sceneHostBuffersInitialized;
	PxU32 sceneActorAdded;
	PxU32 sceneActorRemoved;
	PxU32 sceneActorReleased;
	PxU32 sceneBoundsFinite;
	PxU32 sceneStaticShapeDetached;
	PxU32 sceneStaticShapeReattached;
	PxU32 sceneStaticActorRemoved;
	PxU32 sceneStaticActorReadded;
	PxU32 sceneDynamicActorAdded;
	PxU32 sceneDynamicActorRemoved;
	PxU32 sceneDynamicActorReleased;
	PxU32 sceneDynamicInitiallySleeping;
	PxU32 sceneDynamicWokeBySoft;
	PxU32 sceneDynamicFirstWakeFrame;
	PxU32 sceneDynamicShapeDetached;
	PxU32 sceneDynamicShapeReattached;
	PxU32 sceneDynamicActorReadded;
	PxU32 sceneDynamicReaddedSleeping;
	PxU32 sceneDynamicRewokeBySoft;
	PxU32 sceneDynamicSecondWakeFrame;
	PxU32 sceneSecondDynamicActorAdded;
	PxU32 sceneSecondDynamicActorRemoved;
	PxU32 sceneSecondDynamicActorReleased;
	PxU32 sceneSecondDynamicInitiallySleeping;
	PxU32 sceneSecondDynamicWokeBySoft;
	PxU32 sceneSecondDynamicFirstWakeFrame;
	PxU32 sceneSecondVolumeActorCreated;
	PxU32 sceneSecondVolumeHostBuffersInitialized;
	PxU32 sceneSecondVolumeActorAdded;
	PxU32 sceneSecondVolumeActorRemoved;
	PxU32 sceneSecondVolumeActorReleased;
	PxU32 sceneSecondVolumeBoundsFinite;
	PxU32 sceneSoftInitiallyAwake;
	PxU32 sceneSoftFirstSlept;
	PxU32 sceneSoftFirstSleepFrame;
	PxU32 sceneSoftSleepWakeCounterZero;
	PxU32 sceneSoftSleepVelocitiesZero;
	PxU32 sceneSoftStableWhileSleeping;
	PxU32 sceneSoftCounterWakeIssued;
	PxU32 sceneSoftWokeByCounter;
	PxU32 sceneSoftCounterWakeFrame;
	PxU32 sceneSoftSecondSlept;
	PxU32 sceneSoftSecondSleepFrame;
	PxU32 sceneSoftVelocityWakeIssued;
	PxU32 sceneSoftWokeByVelocity;
	PxU32 sceneSoftVelocityWakeFrame;
	PxU32 sceneSoftMovedAfterVelocityWake;
	PxU32 sceneSoftVelocityStopIssued;
	PxU32 sceneSoftFinalSlept;
	PxU32 sceneSoftFinalSleepFrame;
	PxU32 sceneSoftRigidWakeActorAdded;
	PxU32 sceneSoftWokeByRigid;
	PxU32 sceneSoftRigidWakeFrame;
	PxU32 sceneSoftMovedAfterRigidWake;
	PxU32 sceneMixedFirstSlept;
	PxU32 sceneMixedFirstSleepFrame;
	PxU32 sceneMixedFirstStable;
	PxU32 sceneMixedSecondStayedAwake;
	PxU32 sceneMixedSecondMoved;
	PxU32 sceneSoftChurnRemoveCount;
	PxU32 sceneSoftChurnReaddCount;
	PxU32 sceneSoftChurnCycles;
	PxU32 sceneSoftChurnPostCompactMoveCount;
	PxU32 sceneSoftChurnStable;
	PxU32 sceneBufferMutationIssued;
	PxU32 sceneBufferMutationWoke;
	PxU32 sceneBufferMutationApplied;
	PxU32 sceneBufferDriveIssued;
	PxU32 sceneBufferPinHeld;
	PxU32 sceneBufferDynamicMoved;
	PxU32 sceneBufferInvMassRestored;
	PxU32 sceneBufferRestoredMoved;
	PxU32 sceneBufferResetIssued;
	PxU32 sceneWorldPinCreated;
	PxU32 sceneWorldPinHeld;
	PxU32 sceneWorldPinActorReadded;
	PxU32 sceneWorldPinReleased;
	PxU32 sceneWorldPinMovedAfterRelease;
	PxU32 sceneRigidAttachmentActorAdded;
	PxU32 sceneRigidAttachmentInitiallySleeping;
	PxU32 sceneRigidAttachmentCreated;
	PxU32 sceneRigidAttachmentRigidWoke;
	PxU32 sceneRigidAttachmentRigidMoved;
	PxU32 sceneRigidAttachmentHeldAcrossReadd;
	PxU32 sceneRigidAttachmentReleased;
	PxU32 sceneRigidAttachmentSeparatedAfterRelease;
	PxU32 sceneArticulationCreated;
	PxU32 sceneArticulationAdded;
	PxU32 sceneArticulationInitiallySleeping;
	PxU32 sceneArticulationWoke;
	PxU32 sceneArticulationJointSubspaceHeld;
	PxU32 sceneArticulationRootStable;
	PxU32 sceneElementFilterCreated;
	PxU32 sceneElementFilterActorReadded;
	PxU32 sceneElementFilterSuppressedContact;
	PxU32 sceneElementFilterReleased;
	PxU32 sceneElementFilterContactRestored;
	PxU32 scenePartialFilterUnfilteredContactHeld;
	PxU32 scenePartialFilterExactOwnership;
	PxU32 sceneKinematicActorAdded;
	PxU32 sceneKinematicTargetIssued;
	PxU32 sceneKinematicTargetReached;
	PxU32 sceneKinematicSoftWoke;
	PxU32 sceneKinematicSoftMoved;
	PxU32 sceneKinematicContactObserved;
	PxU32 sceneVolumeTargetBound;
	PxU32 sceneVolumeTargetMutated;
	PxU32 sceneVolumeTargetWoke;
	PxU32 sceneVolumeTargetReached;
	PxU32 sceneVolumePartialInactiveIgnored;
	PxU32 sceneVolumePartialActivated;
	PxU32 sceneVolumePartialActivatedReached;
	PxU32 sceneSecondSceneCreated;
	PxU32 sceneSecondSceneSolverMatched;
	PxU32 scenePrimarySceneReleased;
	PxU32 sceneSecondSceneReleased;
	PxU32 sceneMultiPrimaryStable;
	PxU32 sceneMultiPrimaryDetachedStable;
	PxU32 sceneMultiSecondaryUpdatedBeforeRelease;
	PxU32 sceneMultiSecondaryUpdatedAfterRelease;
	PxU32 sceneSoftSoftBothSlept;
	PxU32 sceneSoftSoftDriveIssued;
	PxU32 sceneSoftSoftDriverWoke;
	PxU32 sceneSoftSoftTargetWoke;
	PxU32 sceneSoftSoftTargetWakeFrame;
	PxU32 sceneSoftSoftTargetMoved;
	PxU32 sceneSoftSoftResetIssued;
	PxU32 sceneSoftSoftBothFinalSlept;
	PxU32 motionMaxVelocityBounded;
	PxU32 motionSettlingApplied;
	PxU32 motionSettlingSlept;
	PxU32 motionControlStayedAwake;
	PxU32 depenetrationLimitApplied;
	PxU32 depenetrationFirstStepBounded;
	PxU32 depenetrationControlSeparated;
	PxU32 depenetrationGradualRecovery;
	PxU32 speculativeCcdFlagApplied;
	PxU32 speculativeCcdPreventedTunneling;
	PxU32 speculativeCcdNegativeControlTunneled;
	PxU32 movingSphereTargetIssued;
	PxU32 movingSphereCcdResponseObserved;
	PxU32 movingSphereNegativeControlHeld;
	PxU32 dynamicSphereSweepLaunched;
	PxU32 dynamicSphereSweepResponseObserved;
	PxU32 dynamicSphereSweepNegativeControlTunneled;
	PxU32 dynamicSphereSweepTwoSidedResponseObserved;
	PxReal minDetF;
	PxReal maxDetF;
	PxReal minBodyVolumeRatio;
	PxReal maxBodyVolumeRatio;
	PxReal minY;
	PxReal maxY;
	PxReal maxParticleSpeed;
	PxReal finalMinY;
	PxReal finalMaxY;
	PxReal finalMaxParticleSpeed;
	PxReal maxCentroidDrop;
	PxReal sceneDynamicMinY;
	PxReal sceneDynamicFinalY;
	PxReal sceneDynamicMaxDrop;
	PxReal sceneDynamicPreContactMaxDrop;
	PxReal sceneDynamicMaxDownSpeed;
	PxReal sceneSecondDynamicMinY;
	PxReal sceneSecondDynamicFinalY;
	PxReal sceneSecondDynamicMaxDrop;
	PxReal sceneSecondDynamicPreContactMaxDrop;
	PxReal sceneSecondDynamicMaxDownSpeed;
	PxReal sceneSecondVolumeMaxCentroidDrop;
	PxReal sceneSecondVolumeFinalCentroidY;
	PxReal sceneWorldPinMaxDrift;
	PxReal sceneWorldPinReleasedMaxDisplacement;
	PxReal sceneRigidAttachmentMaxDrift;
	PxReal sceneRigidAttachmentMaxRigidDisplacement;
	PxReal sceneRigidAttachmentMaxRigidSpeed;
	PxReal sceneRigidAttachmentReleasedSeparation;
	PxReal sceneArticulationRootMaxDisplacement;
	PxReal sceneArticulationChildMaxForbiddenDisplacement;
	PxReal sceneArticulationChildMaxAngularDisplacement;
	PxReal sceneElementFilterMinY;
	PxReal sceneElementFilterFinalMinY;
	PxReal scenePartialFilterUnfilteredMinY;
	PxReal sceneKinematicMaxPoseError;
	PxReal sceneKinematicSoftDisplacement;
	PxReal sceneKinematicFinalY;
	PxReal sceneVolumeTargetFinalMaxError;
	PxReal sceneVolumeTargetMaxDisplacement;
	PxReal sceneVolumePartialInactiveDecoyDistance;
	PxReal minDynamicSurfaceSeparation;
	PxReal finalDynamicSurfaceSeparation;
	PxReal motionMaxVelocityFirstStepDisplacement;
	PxReal motionMaxVelocityFirstStepSpeed;
	PxReal motionSettlingFinalSpeed;
	PxReal motionControlFinalSpeed;
	PxReal depenetrationLimitedFirstStepRise;
	PxReal depenetrationControlFirstStepRise;
	PxReal depenetrationLimitedFinalRise;
	PxReal depenetrationLimitedMaxSpeed;
	PxReal speculativeCcdPositiveMinY;
	PxReal speculativeCcdPositiveMinSeparation;
	PxReal speculativeCcdNegativeMaxY;
	PxReal movingSpherePositiveDisplacement;
	PxReal movingSphereNegativeDisplacement;
	PxReal movingSpherePositiveMinSeparation;
	PxReal dynamicSphereSweepPositiveSoftDisplacement;
	PxReal dynamicSphereSweepNegativeSoftDisplacement;
	PxReal dynamicSphereSweepPositiveRigidDrop;
	PxReal dynamicSphereSweepNegativeRigidDrop;
	PxReal dynamicSphereSweepPositiveMinSeparation;
	bool solverReadbackMatched;

	DeformableVolumeMetrics()
	: initialized(0), completedFrames(0), fetchFailures(0),
	  nonFiniteParticleSamples(0), invertedElementSamples(0),
	  firstInversionFrame(PX_MAX_U32), firstInversionBody(PX_MAX_U32),
	  firstInversionElement(PX_MAX_U32), invertedBodiesMask(0),
	  particles(0), softBodies(0), tetElements(0), surfaceTriangles(0),
	  rigidBoxes(0),
	  sceneStatics(0), sceneDynamics(0), sceneDeformableVolumes(0),
	  groundContactFrames(0), rigidContactFrames(0), softContactFrames(0),
	  maxGroundContacts(0), maxRigidContacts(0), maxSoftContacts(0),
	  invalidContactSourceSamples(0), finalInsideParticles(0),
	  cleanupComplete(0), sceneActorCreated(0), sceneShapeAttached(0),
	  sceneSimulationMeshAttached(0), sceneHostBuffersInitialized(0),
	  sceneActorAdded(0), sceneActorRemoved(0), sceneActorReleased(0),
	  sceneBoundsFinite(0), sceneStaticShapeDetached(0),
	  sceneStaticShapeReattached(0), sceneStaticActorRemoved(0),
	  sceneStaticActorReadded(0), sceneDynamicActorAdded(0),
	  sceneDynamicActorRemoved(0),
	  sceneDynamicActorReleased(0), sceneDynamicInitiallySleeping(0),
	  sceneDynamicWokeBySoft(0),
	  sceneDynamicFirstWakeFrame(PX_MAX_U32),
	  sceneDynamicShapeDetached(0),
	  sceneDynamicShapeReattached(0),
	  sceneDynamicActorReadded(0),
	  sceneDynamicReaddedSleeping(0),
	  sceneDynamicRewokeBySoft(0),
	  sceneDynamicSecondWakeFrame(PX_MAX_U32),
	  sceneSecondDynamicActorAdded(0),
	  sceneSecondDynamicActorRemoved(0),
	  sceneSecondDynamicActorReleased(0),
	  sceneSecondDynamicInitiallySleeping(0),
	  sceneSecondDynamicWokeBySoft(0),
	  sceneSecondDynamicFirstWakeFrame(PX_MAX_U32),
	  sceneSecondVolumeActorCreated(0),
	  sceneSecondVolumeHostBuffersInitialized(0),
	  sceneSecondVolumeActorAdded(0),
	  sceneSecondVolumeActorRemoved(0),
	  sceneSecondVolumeActorReleased(0),
	  sceneSecondVolumeBoundsFinite(0),
	  sceneSoftInitiallyAwake(0),
	  sceneSoftFirstSlept(0),
	  sceneSoftFirstSleepFrame(PX_MAX_U32),
	  sceneSoftSleepWakeCounterZero(0),
	  sceneSoftSleepVelocitiesZero(0),
	  sceneSoftStableWhileSleeping(0),
	  sceneSoftCounterWakeIssued(0),
	  sceneSoftWokeByCounter(0),
	  sceneSoftCounterWakeFrame(PX_MAX_U32),
	  sceneSoftSecondSlept(0),
	  sceneSoftSecondSleepFrame(PX_MAX_U32),
	  sceneSoftVelocityWakeIssued(0),
	  sceneSoftWokeByVelocity(0),
	  sceneSoftVelocityWakeFrame(PX_MAX_U32),
	  sceneSoftMovedAfterVelocityWake(0),
	  sceneSoftVelocityStopIssued(0),
	  sceneSoftFinalSlept(0),
	  sceneSoftFinalSleepFrame(PX_MAX_U32),
	  sceneSoftRigidWakeActorAdded(0),
	  sceneSoftWokeByRigid(0),
	  sceneSoftRigidWakeFrame(PX_MAX_U32),
	  sceneSoftMovedAfterRigidWake(0),
	  sceneMixedFirstSlept(0),
	  sceneMixedFirstSleepFrame(PX_MAX_U32),
	  sceneMixedFirstStable(0),
	  sceneMixedSecondStayedAwake(0),
	  sceneMixedSecondMoved(0),
	  sceneSoftChurnRemoveCount(0),
	  sceneSoftChurnReaddCount(0),
	  sceneSoftChurnCycles(0),
	  sceneSoftChurnPostCompactMoveCount(0),
	  sceneSoftChurnStable(0),
	  sceneBufferMutationIssued(0),
	  sceneBufferMutationWoke(0),
	  sceneBufferMutationApplied(0),
	  sceneBufferDriveIssued(0),
	  sceneBufferPinHeld(0),
	  sceneBufferDynamicMoved(0),
	  sceneBufferInvMassRestored(0),
	  sceneBufferRestoredMoved(0),
	  sceneBufferResetIssued(0),
	  sceneWorldPinCreated(0),
	  sceneWorldPinHeld(0),
	  sceneWorldPinActorReadded(0),
	  sceneWorldPinReleased(0),
	  sceneWorldPinMovedAfterRelease(0),
	  sceneRigidAttachmentActorAdded(0),
	  sceneRigidAttachmentInitiallySleeping(0),
	  sceneRigidAttachmentCreated(0),
	  sceneRigidAttachmentRigidWoke(0),
	  sceneRigidAttachmentRigidMoved(0),
	  sceneRigidAttachmentHeldAcrossReadd(0),
	  sceneRigidAttachmentReleased(0),
	  sceneRigidAttachmentSeparatedAfterRelease(0),
	  sceneArticulationCreated(0),
	  sceneArticulationAdded(0),
	  sceneArticulationInitiallySleeping(0),
	  sceneArticulationWoke(0),
	  sceneArticulationJointSubspaceHeld(0),
	  sceneArticulationRootStable(0),
	  sceneElementFilterCreated(0),
	  sceneElementFilterActorReadded(0),
	  sceneElementFilterSuppressedContact(0),
	  sceneElementFilterReleased(0),
	  sceneElementFilterContactRestored(0),
	  scenePartialFilterUnfilteredContactHeld(0),
	  scenePartialFilterExactOwnership(0),
	  sceneKinematicActorAdded(0),
	  sceneKinematicTargetIssued(0),
	  sceneKinematicTargetReached(0),
	  sceneKinematicSoftWoke(0),
	  sceneKinematicSoftMoved(0),
	  sceneKinematicContactObserved(0),
	  sceneVolumeTargetBound(0),
	  sceneVolumeTargetMutated(0),
	  sceneVolumeTargetWoke(0),
	  sceneVolumeTargetReached(0),
	  sceneVolumePartialInactiveIgnored(0),
	  sceneVolumePartialActivated(0),
	  sceneVolumePartialActivatedReached(0),
	  sceneSecondSceneCreated(0),
	  sceneSecondSceneSolverMatched(0),
	  scenePrimarySceneReleased(0),
	  sceneSecondSceneReleased(0),
	  sceneMultiPrimaryStable(0),
	  sceneMultiPrimaryDetachedStable(0),
	  sceneMultiSecondaryUpdatedBeforeRelease(0),
	  sceneMultiSecondaryUpdatedAfterRelease(0),
	  sceneSoftSoftBothSlept(0),
	  sceneSoftSoftDriveIssued(0),
	  sceneSoftSoftDriverWoke(0),
	  sceneSoftSoftTargetWoke(0),
	  sceneSoftSoftTargetWakeFrame(PX_MAX_U32),
	  sceneSoftSoftTargetMoved(0),
	  sceneSoftSoftResetIssued(0),
	  sceneSoftSoftBothFinalSlept(0),
	  motionMaxVelocityBounded(0),
	  motionSettlingApplied(0),
	  motionSettlingSlept(0),
	  motionControlStayedAwake(0),
	  depenetrationLimitApplied(0),
	  depenetrationFirstStepBounded(0),
	  depenetrationControlSeparated(0),
	  depenetrationGradualRecovery(0),
	  speculativeCcdFlagApplied(0),
	  speculativeCcdPreventedTunneling(0),
	  speculativeCcdNegativeControlTunneled(0),
	  movingSphereTargetIssued(0),
	  movingSphereCcdResponseObserved(0),
	  movingSphereNegativeControlHeld(0),
	  dynamicSphereSweepLaunched(0),
	  dynamicSphereSweepResponseObserved(0),
	  dynamicSphereSweepNegativeControlTunneled(0),
	  dynamicSphereSweepTwoSidedResponseObserved(0),
	  minDetF(FLT_MAX),
	  maxDetF(-FLT_MAX), minBodyVolumeRatio(FLT_MAX),
	  maxBodyVolumeRatio(-FLT_MAX), minY(FLT_MAX), maxY(-FLT_MAX),
	  maxParticleSpeed(0.0f), finalMinY(FLT_MAX), finalMaxY(-FLT_MAX),
	  finalMaxParticleSpeed(0.0f),
	  maxCentroidDrop(0.0f), sceneDynamicMinY(FLT_MAX),
	  sceneDynamicFinalY(FLT_MAX),
	  sceneDynamicMaxDrop(0.0f),
	  sceneDynamicPreContactMaxDrop(0.0f),
	  sceneDynamicMaxDownSpeed(0.0f),
	  sceneSecondDynamicMinY(FLT_MAX),
	  sceneSecondDynamicFinalY(FLT_MAX),
	  sceneSecondDynamicMaxDrop(0.0f),
	  sceneSecondDynamicPreContactMaxDrop(0.0f),
	  sceneSecondDynamicMaxDownSpeed(0.0f),
	  sceneSecondVolumeMaxCentroidDrop(0.0f),
	  sceneSecondVolumeFinalCentroidY(FLT_MAX),
	  sceneWorldPinMaxDrift(0.0f),
	  sceneWorldPinReleasedMaxDisplacement(0.0f),
	  sceneRigidAttachmentMaxDrift(0.0f),
	  sceneRigidAttachmentMaxRigidDisplacement(0.0f),
	  sceneRigidAttachmentMaxRigidSpeed(0.0f),
	  sceneRigidAttachmentReleasedSeparation(0.0f),
	  sceneArticulationRootMaxDisplacement(0.0f),
	  sceneArticulationChildMaxForbiddenDisplacement(0.0f),
	  sceneArticulationChildMaxAngularDisplacement(0.0f),
	  sceneElementFilterMinY(FLT_MAX),
	  sceneElementFilterFinalMinY(FLT_MAX),
	  scenePartialFilterUnfilteredMinY(FLT_MAX),
	  sceneKinematicMaxPoseError(0.0f),
	  sceneKinematicSoftDisplacement(0.0f),
	  sceneKinematicFinalY(FLT_MAX),
	  sceneVolumeTargetFinalMaxError(FLT_MAX),
	  sceneVolumeTargetMaxDisplacement(0.0f),
	  sceneVolumePartialInactiveDecoyDistance(0.0f),
	  minDynamicSurfaceSeparation(FLT_MAX),
	  finalDynamicSurfaceSeparation(FLT_MAX),
	  motionMaxVelocityFirstStepDisplacement(0.0f),
	  motionMaxVelocityFirstStepSpeed(0.0f),
	  motionSettlingFinalSpeed(0.0f),
	  motionControlFinalSpeed(0.0f),
	  depenetrationLimitedFirstStepRise(0.0f),
	  depenetrationControlFirstStepRise(0.0f),
	  depenetrationLimitedFinalRise(0.0f),
	  depenetrationLimitedMaxSpeed(0.0f),
	  speculativeCcdPositiveMinY(PX_MAX_F32),
	  speculativeCcdPositiveMinSeparation(PX_MAX_F32),
	  speculativeCcdNegativeMaxY(-PX_MAX_F32),
	  movingSpherePositiveDisplacement(0.0f),
	  movingSphereNegativeDisplacement(0.0f),
	  movingSpherePositiveMinSeparation(PX_MAX_F32),
	  dynamicSphereSweepPositiveSoftDisplacement(0.0f),
	  dynamicSphereSweepNegativeSoftDisplacement(0.0f),
	  dynamicSphereSweepPositiveRigidDrop(0.0f),
	  dynamicSphereSweepNegativeRigidDrop(0.0f),
	  dynamicSphereSweepPositiveMinSeparation(PX_MAX_F32),
	  solverReadbackMatched(false)
	{
	}
};

static DeformableVolumeMetrics gMetrics;
static PxArray<PxVec3> gInitialCentroids;

struct DeformableVolumePerformanceMetrics
{
	PxU32 warmupFrames;
	PxU32 profiledFrames;
	PxU32 softWorkers;
	PxArray<PxReal> stepSamplesMs;
	PxF64 initialContactMs;
	PxF64 solverMs;
	PxF64 sceneMs;
	PxF64 metricsMs;
	PxReal avgStepMs;
	PxReal p50StepMs;
	PxReal p95StepMs;
	PxReal maxStepMs;
	AvbdSoftBodyStepStats solverStages;
	AvbdSoftCollisionStats collision;

	DeformableVolumePerformanceMetrics()
		: warmupFrames(0), profiledFrames(0), softWorkers(1),
		  initialContactMs(0.0),
		  solverMs(0.0), sceneMs(0.0), metricsMs(0.0),
		  avgStepMs(0.0f), p50StepMs(0.0f), p95StepMs(0.0f),
		  maxStepMs(0.0f)
	{
	}
};

static DeformableVolumePerformanceMetrics gPerformance;
static AvbdSoftCollisionStats gFrameCollisionStats;
static PxU32 gProfileWarmupFrames = 0;

static void accumulateStepStats(
	AvbdSoftBodyStepStats& total, const AvbdSoftBodyStepStats& frame)
{
	total.predictionMs += frame.predictionMs;
	total.contactIndexMs += frame.contactIndexMs;
	total.bodyPrecomputeMs += frame.bodyPrecomputeMs;
	total.bodySolveMs += frame.bodySolveMs;
	total.particleSolveMs += frame.particleSolveMs;
	total.projectionMs += frame.projectionMs;
	total.dualMs += frame.dualMs;
	total.redetectMs += frame.redetectMs;
	total.velocityMs += frame.velocityMs;
	total.frictionMs += frame.frictionMs;
	total.requestedOuterIterations += frame.requestedOuterIterations;
	total.requestedInnerIterations += frame.requestedInnerIterations;
	total.executedOuterIterations += frame.executedOuterIterations;
	total.executedInnerIterations += frame.executedInnerIterations;
	total.particleSweeps += frame.particleSweeps;
	total.trustRegionLimitedParticleSteps +=
		frame.trustRegionLimitedParticleSteps;
	total.positiveJLimitedParticleSteps +=
		frame.positiveJLimitedParticleSteps;
	total.positiveJRejectedParticleSteps +=
		frame.positiveJRejectedParticleSteps;
	total.nonFiniteRejectedParticleSteps +=
		frame.nonFiniteRejectedParticleSteps;
	total.tetLinearizationCacheFallbackParticleSteps +=
		frame.tetLinearizationCacheFallbackParticleSteps;
	total.legacyAppliedConvergedOuterIterations +=
		frame.legacyAppliedConvergedOuterIterations;
	total.residualConvergedOuterIterations +=
		frame.residualConvergedOuterIterations;
	total.unsafeAppliedConvergenceCandidates +=
		frame.unsafeAppliedConvergenceCandidates;
	total.budgetExhaustedOuterIterations +=
		frame.budgetExhaustedOuterIterations;
	total.shadowResidual1e5ConvergedOuterIterations +=
		frame.shadowResidual1e5ConvergedOuterIterations;
	total.shadowResidual1e5SavedInnerIterations +=
		frame.shadowResidual1e5SavedInnerIterations;
	total.shadowResidual1e4ConvergedOuterIterations +=
		frame.shadowResidual1e4ConvergedOuterIterations;
	total.shadowResidual1e4SavedInnerIterations +=
		frame.shadowResidual1e4SavedInnerIterations;
	total.workspaceGrowthEvents += frame.workspaceGrowthEvents;
	total.workspaceGrowthBytes += frame.workspaceGrowthBytes;
	total.contactWorkspaceGrowthEvents +=
		frame.contactWorkspaceGrowthEvents;
	total.contactWorkspaceGrowthBytes +=
		frame.contactWorkspaceGrowthBytes;
	total.contactOutputGrowthEvents +=
		frame.contactOutputGrowthEvents;
	total.contactOutputGrowthBytes +=
		frame.contactOutputGrowthBytes;
	total.finalMaxLocalSolveDisplacement =
		frame.finalMaxLocalSolveDisplacement;
	total.finalMaxAppliedDisplacement =
		frame.finalMaxAppliedDisplacement;
	total.finalMaxDisplacement = frame.finalMaxDisplacement;
}

// ---------------------------------------------------------------------------
// Push AVBD soft-body surface triangles to PVD as debug geometry.
// ---------------------------------------------------------------------------
static void sendSoftBodiesToPvd()
{
	PxPvdSceneClient* pvdClient = gScene ? gScene->getScenePvdClient() : NULL;
	if (!pvdClient)
		return;

	static const PxU32 bodyColors[] = { 0xFFFF8000, 0xFF0080FF, 0xFF00FF80 };

	PxArray<PxDebugTriangle> tris;
	for (PxU32 b = 0; b < gSoftBodies.size(); b++)
	{
		const AvbdSoftBody& sb = gSoftBodies[b];
		const PxU32* idx = sb.compiled.surfaceTriangles.begin();
		const PxU32 numTris = sb.compiled.surfaceTriangles.size() / 3;
		const PxU32 color = bodyColors[b % (sizeof(bodyColors) / sizeof(bodyColors[0]))];

		for (PxU32 t = 0; t < numTris; t++)
		{
			const PxVec3& p0 = gParticles[idx[t * 3 + 0]].position;
			const PxVec3& p1 = gParticles[idx[t * 3 + 1]].position;
			const PxVec3& p2 = gParticles[idx[t * 3 + 2]].position;
			tris.pushBack(PxDebugTriangle(p0, p1, p2, color));
		}
	}

	if (tris.size())
		pvdClient->drawTriangles(tris.begin(), tris.size());
}

// ---------------------------------------------------------------------------
static AvbdOGCParams gOGCParams;

static void initOGCParams()
{
	gOGCParams.contactRadius    = 0.20f;
	gOGCParams.contactStiffness = 3e5f;
	gOGCParams.friction         = 0.35f;
}

// ---------------------------------------------------------------------------
static void updateRenderData()
{
	gSoftBodyRenderData.clear();
	for (PxU32 i = 0; i < gSoftBodies.size(); i++)
	{
		SoftBodyRenderData rd;
		rd.surfaceTriIndices = gSoftBodies[i].compiled.surfaceTriangles.begin();
		rd.numSurfaceTris    = gSoftBodies[i].compiled.surfaceTriangles.size() / 3;
		gSoftBodyRenderData.pushBack(rd);
	}
}

static PxVec3 getSoftBodyCentroid(const AvbdSoftBody& body)
{
	PxVec3 centroid(0.0f);
	PxReal totalMass = 0.0f;
	for(PxU32 localId = 0; localId < body.compiled.particleCount; ++localId)
	{
		const AvbdSoftParticle& particle =
			gParticles[body.compiled.particleStart + localId];
		centroid += particle.position * particle.mass;
		totalMass += particle.mass;
	}
	return totalMass > 0.0f ?
		centroid * (1.0f / totalMass) : PxVec3(0.0f);
}

static void addCubeSoftBody(
	const PxVec3& center, PxReal halfExtent, PxU32 subdivisions,
	PxReal youngsModulus = 2e5f, PxReal density = 500.0f,
	PxReal damping = 0.015f)
{
	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	avbdGenerateSubdividedCubeTets(
		center, halfExtent, int(subdivisions), verts, tets);
	avbdCreateSoftBody(
		verts.begin(), verts.size(), tets.begin(), tets.size(), NULL, 0,
		youngsModulus, 0.3f, density, damping, 0.0f, 0.01f,
		gParticles, gSoftBodies);
}

static void addConeSoftBody(const PxVec3& baseCenter)
{
	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	generateConeTetsViaTetMaker(
		baseCenter, 0.8f, 3.0f, 14, verts, tets);
	avbdCreateSoftBody(
		verts.begin(), verts.size(), tets.begin(), tets.size(), NULL, 0,
		2e5f, 0.3f, 100.0f, 0.015f, 0.0f, 0.01f,
		gParticles, gSoftBodies);
}

static bool addRigidBox(
	const PxVec3& center, const PxVec3& halfExtent)
{
	AvbdRigidBox rigidBox;
	rigidBox.center = center;
	rigidBox.halfExtent = halfExtent;
	rigidBox.friction = 0.5f;
	gRigidBoxes.pushBack(rigidBox);

	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	return true;
}

static bool isSceneCpuVolumeSpeculativeCcdCase(
	const std::string& caseName)
{
	return caseName == "scene-volume-speculative-ccd" ||
		caseName == "scene-volume-plane-speculative-ccd" ||
		caseName == "scene-volume-sphere-speculative-ccd" ||
		caseName == "scene-volume-capsule-speculative-ccd" ||
		caseName == "scene-volume-convex-speculative-ccd" ||
		caseName ==
			"scene-volume-moving-kinematic-sphere-speculative-ccd" ||
		caseName ==
			"scene-volume-moving-kinematic-capsule-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-capsule-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-convex-speculative-ccd" ||
		caseName ==
			"scene-volume-moving-kinematic-convex-speculative-ccd" ||
		caseName ==
			"scene-volume-dynamic-sphere-relative-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-capsule-relative-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-capsule-relative-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-convex-relative-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-convex-relative-swept-ccd" ||
		caseName ==
			"scene-volume-static-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-kinematic-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-static-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-kinematic-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-static-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-heightfield-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-heightfield-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-heightfield-reverse-swept-ccd";
}

static bool isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
	const std::string& caseName)
{
	return
		caseName ==
			"scene-volume-static-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-kinematic-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-static-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-kinematic-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-static-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-heightfield-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-heightfield-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-heightfield-reverse-swept-ccd";
}

static bool isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
	const std::string& caseName)
{
	return isSceneCpuVolumeTriangleSurfaceSweptCcdCase(caseName) &&
		caseName.find("reverse-swept") != std::string::npos;
}

static bool isSceneCpuVolumeStaticTriangleSurfaceSweptCcdCase(
	const std::string& caseName)
{
	return isSceneCpuVolumeTriangleSurfaceSweptCcdCase(caseName) &&
		caseName.find("static") != std::string::npos;
}

static bool isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
	const std::string& caseName)
{
	return isSceneCpuVolumeTriangleSurfaceSweptCcdCase(caseName) &&
		caseName.find("rotating-kinematic") !=
			std::string::npos;
}

static bool isSceneCpuVolumeHeightFieldSweptCcdCase(
	const std::string& caseName)
{
	return isSceneCpuVolumeTriangleSurfaceSweptCcdCase(caseName) &&
		caseName.find("heightfield") != std::string::npos;
}

static bool isSceneCpuVolumeSphereReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-deforming-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-deforming-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-deforming-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-convex-reverse-swept-ccd";
}

static bool isSceneCpuVolumeCapsuleReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-deforming-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd";
}

static bool isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd";
}

static bool isSceneCpuVolumeConvexReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-deforming-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd";
}

static bool isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd";
}

static bool isSceneCpuVolumeDeformingReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-deforming-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-deforming-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-deforming-convex-reverse-swept-ccd";
}

static bool isSceneCpuVolumeKinematicRigidCase(
	const std::string& caseName)
{
	return caseName == "scene-volume-kinematic-box" ||
		caseName == "scene-volume-kinematic-sphere" ||
		caseName == "scene-volume-kinematic-capsule" ||
		caseName == "scene-volume-kinematic-convex" ||
		caseName == "scene-volume-kinematic-triangle-mesh" ||
		caseName == "scene-volume-kinematic-heightfield";
}

static bool isSceneCpuVolumeCase(const std::string& caseName)
{
	return caseName == "scene-volume-lifecycle" ||
		caseName == "scene-volume-corotational" ||
		caseName == "scene-volume-ground" ||
		caseName == "scene-volume-static-box" ||
		caseName == "scene-volume-static-churn" ||
		caseName == "scene-volume-dynamic-box" ||
		caseName == "scene-volume-dynamic-sphere" ||
		caseName == "scene-volume-dynamic-capsule" ||
		caseName == "scene-volume-dynamic-convex" ||
		caseName == "scene-volume-dynamic-churn" ||
		caseName == "scene-volume-multi-dynamic-box" ||
		caseName == "scene-volume-multi-soft-islands" ||
		caseName == "scene-volume-sleep-wake" ||
		caseName == "scene-volume-rigid-wake" ||
		caseName == "scene-volume-mixed-sleep-islands" ||
		caseName == "scene-volume-soft-churn" ||
		caseName == "scene-volume-buffer-mutation" ||
		caseName == "scene-volume-world-pin" ||
		caseName == "scene-volume-world-element-attachment" ||
		caseName == "scene-volume-rigid-attachment" ||
		caseName == "scene-volume-rigid-element-attachment" ||
		caseName == "scene-volume-static-attachment" ||
		caseName == "scene-volume-static-element-attachment" ||
		caseName == "scene-volume-kinematic-attachment" ||
		caseName == "scene-volume-kinematic-element-attachment" ||
		caseName == "scene-volume-articulation-attachment" ||
		caseName == "scene-volume-articulation-element-attachment" ||
		caseName == "scene-volume-element-filter" ||
		caseName == "scene-volume-partial-element-filter" ||
		isSceneCpuVolumeKinematicRigidCase(caseName) ||
		caseName == "scene-volume-full-kinematic-target" ||
		caseName == "scene-volume-partial-kinematic-target" ||
		caseName == "scene-volume-multi-scene-isolation" ||
		caseName == "scene-volume-soft-soft-wake" ||
		caseName == "scene-volume-volume-attachment" ||
		caseName == "scene-volume-skinning" ||
		caseName == "scene-volume-motion-controls" ||
		caseName ==
			"scene-volume-max-depenetration-velocity" ||
		caseName == "scene-volume-sphere-reverse-feature" ||
		caseName == "scene-volume-capsule-reverse-feature" ||
		caseName == "scene-volume-convex-reverse-feature" ||
		caseName ==
			"scene-volume-triangle-mesh-reverse-feature" ||
		caseName ==
			"scene-volume-heightfield-reverse-feature" ||
		isSceneCpuVolumeSpeculativeCcdCase(caseName) ||
		isSceneCpuVolumeSphereReverseSweptCcdCase(caseName);
}

static bool addSceneStaticBox(
	const PxVec3& center, const PxVec3& halfExtent)
{
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return true;
}

static bool addSceneStaticSphereCluster(
	const PxVec3* centers, PxU32 centerCount, PxReal radius)
{
	if(!centers || centerCount == 0 || radius <= 0.0f)
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(PxIdentity));
	if(!actor)
		return false;
	for(PxU32 i = 0; i < centerCount; ++i)
	{
		PxShape* shape = PxRigidActorExt::createExclusiveShape(
			*actor, PxSphereGeometry(radius), *gMaterial);
		if(!shape)
		{
			actor->release();
			return false;
		}
		shape->setLocalPose(PxTransform(centers[i]));
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return true;
}

static bool addSceneStaticCapsule(
	const PxVec3& center, PxReal radius, PxReal halfHeight)
{
	if(radius <= 0.0f || halfHeight < 0.0f)
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxCapsuleGeometry(radius, halfHeight), *gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return true;
}

static bool addSceneStaticCapsuleCluster(
	const PxVec3* centers, PxU32 centerCount,
	PxReal radius, PxReal halfHeight)
{
	if(!centers || centerCount == 0 ||
		radius <= 0.0f || halfHeight < 0.0f)
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(PxIdentity));
	if(!actor)
		return false;
	for(PxU32 i = 0; i < centerCount; ++i)
	{
		PxShape* shape = PxRigidActorExt::createExclusiveShape(
			*actor, PxCapsuleGeometry(radius, halfHeight), *gMaterial);
		if(!shape)
		{
			actor->release();
			return false;
		}
		shape->setLocalPose(PxTransform(centers[i]));
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return true;
}

enum SceneCpuRigidConvexFixture
{
	eSCENE_CPU_CONVEX_OWNER,
	eSCENE_CPU_CONVEX_REVERSE_FEATURE,
	eSCENE_CPU_CONVEX_SWEPT_BOX,
	eSCENE_CPU_CONVEX_REVERSE_SWEPT,
	eSCENE_CPU_CONVEX_DEFORMING_REVERSE_SWEPT,
	eSCENE_CPU_CONVEX_ROTATIONAL
};

static bool createSceneCpuRigidConvexMesh(
	SceneCpuRigidConvexFixture fixture)
{
	if(gSceneCpuRigidConvexMesh || !gPhysics)
		return false;
	const PxVec3 reverseVertices[] =
	{
		PxVec3(0.0f, 0.3f, 0.0f),
		PxVec3(-0.3f, -0.3f, -0.3f),
		PxVec3(0.3f, -0.3f, -0.3f),
		PxVec3(0.0f, -0.3f, 0.35f)
	};
	const PxVec3 ownerVertices[] =
	{
		PxVec3(-0.8f, 0.0f, 0.0f),
		PxVec3(0.8f, 0.0f, 0.0f),
		PxVec3(0.0f, -0.8f, 0.0f),
		PxVec3(0.0f, 0.8f, 0.0f),
		PxVec3(0.0f, 0.0f, -0.8f),
		PxVec3(0.0f, 0.0f, 0.8f)
	};
	const PxVec3 sweptBoxVertices[] =
	{
		PxVec3(-0.8f, -0.05f, -0.8f),
		PxVec3(0.8f, -0.05f, -0.8f),
		PxVec3(0.8f, -0.05f, 0.8f),
		PxVec3(-0.8f, -0.05f, 0.8f),
		PxVec3(-0.8f, 0.05f, -0.8f),
		PxVec3(0.8f, 0.05f, -0.8f),
		PxVec3(0.8f, 0.05f, 0.8f),
		PxVec3(-0.8f, 0.05f, 0.8f)
	};
	const PxVec3 reverseSweptVertices[] =
	{
		PxVec3(0.0f, -0.3f, 0.0f),
		PxVec3(0.0f, 0.3f, 0.0f),
		PxVec3(-0.25f, 0.0f, -0.2f),
		PxVec3(0.25f, 0.0f, -0.2f),
		PxVec3(0.25f, 0.0f, 0.2f),
		PxVec3(-0.25f, 0.0f, 0.2f)
	};
	const PxVec3 deformingReverseSweptVertices[] =
	{
		PxVec3(0.0f, -0.2f, 0.0f),
		PxVec3(0.0f, 0.2f, 0.0f),
		PxVec3(-0.15f, 0.0f, -0.12f),
		PxVec3(0.15f, 0.0f, -0.12f),
		PxVec3(0.15f, 0.0f, 0.12f),
		PxVec3(-0.15f, 0.0f, 0.12f)
	};
	const PxVec3 rotationalVertices[] =
	{
		PxVec3(-1.0f, -0.1f, -0.1f),
		PxVec3(1.0f, -0.1f, -0.1f),
		PxVec3(1.0f, -0.1f, 0.1f),
		PxVec3(-1.0f, -0.1f, 0.1f),
		PxVec3(-1.0f, 0.1f, -0.1f),
		PxVec3(1.0f, 0.1f, -0.1f),
		PxVec3(1.0f, 0.1f, 0.1f),
		PxVec3(-1.0f, 0.1f, 0.1f)
	};
	const PxVec3* vertices = ownerVertices;
	PxU32 vertexCount =
		sizeof(ownerVertices) / sizeof(ownerVertices[0]);
	if(fixture == eSCENE_CPU_CONVEX_REVERSE_FEATURE)
	{
		vertices = reverseVertices;
		vertexCount =
			sizeof(reverseVertices) / sizeof(reverseVertices[0]);
	}
	else if(fixture == eSCENE_CPU_CONVEX_SWEPT_BOX)
	{
		vertices = sweptBoxVertices;
		vertexCount =
			sizeof(sweptBoxVertices) / sizeof(sweptBoxVertices[0]);
	}
	else if(fixture == eSCENE_CPU_CONVEX_REVERSE_SWEPT)
	{
		vertices = reverseSweptVertices;
		vertexCount =
			sizeof(reverseSweptVertices) /
			sizeof(reverseSweptVertices[0]);
	}
	else if(
		fixture ==
			eSCENE_CPU_CONVEX_DEFORMING_REVERSE_SWEPT)
	{
		vertices = deformingReverseSweptVertices;
		vertexCount =
			sizeof(deformingReverseSweptVertices) /
			sizeof(deformingReverseSweptVertices[0]);
	}
	else if(fixture == eSCENE_CPU_CONVEX_ROTATIONAL)
	{
		vertices = rotationalVertices;
		vertexCount =
			sizeof(rotationalVertices) /
			sizeof(rotationalVertices[0]);
	}
	PxConvexMeshDesc convexDesc;
	convexDesc.points.count = vertexCount;
	convexDesc.points.stride = sizeof(PxVec3);
	convexDesc.points.data = vertices;
	convexDesc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
	PxCookingParams cookingParams(
		gPhysics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	gSceneCpuRigidConvexMesh = PxCreateConvexMesh(
		cookingParams, convexDesc,
		gPhysics->getPhysicsInsertionCallback());
	return gSceneCpuRigidConvexMesh != NULL;
}

static bool addSceneStaticConvex(
	const PxVec3& center, bool reverseFeature)
{
	if(!createSceneCpuRigidConvexMesh(
			reverseFeature
				? eSCENE_CPU_CONVEX_REVERSE_FEATURE
				: eSCENE_CPU_CONVEX_OWNER))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor,
		PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
		*gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool addSceneStaticConvexCluster(
	const PxVec3* centers, PxU32 centerCount,
	SceneCpuRigidConvexFixture fixture)
{
	if(!centers || centerCount == 0 ||
		!createSceneCpuRigidConvexMesh(fixture))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(PxIdentity));
	if(!actor)
		return false;
	for(PxU32 i = 0; i < centerCount; ++i)
	{
		PxShape* shape = PxRigidActorExt::createExclusiveShape(
			*actor,
			PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
			*gMaterial);
		if(!shape)
		{
			actor->release();
			return false;
		}
		shape->setLocalPose(PxTransform(centers[i]));
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool createSceneCpuRigidTriangleMesh(bool reverseFeature)
{
	if(gSceneCpuRigidTriangleMesh || !gPhysics)
		return false;
	const PxVec3 reverseVertices[] =
	{
		PxVec3(0.0f, 0.3f, 0.0f),
		PxVec3(-0.3f, -0.3f, -0.3f),
		PxVec3(0.3f, -0.3f, -0.3f),
		PxVec3(0.0f, -0.3f, 0.35f)
	};
	const PxU32 reverseTriangles[] =
	{
		0, 2, 1,
		0, 3, 2,
		0, 1, 3,
		1, 2, 3
	};
	const PxVec3 ownerVertices[] =
	{
		PxVec3(-4.0f, 0.0f, -4.0f),
		PxVec3(-4.0f, 0.0f, 4.0f),
		PxVec3(4.0f, 0.0f, 4.0f),
		PxVec3(4.0f, 0.0f, -4.0f)
	};
	const PxU32 ownerTriangles[] =
	{
		0, 1, 2,
		0, 2, 3
	};
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = 4;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = reverseFeature
		? reverseVertices : ownerVertices;
	meshDesc.triangles.count =
		reverseFeature ? 4u : 2u;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = reverseFeature
		? reverseTriangles : ownerTriangles;
	PxCookingParams cookingParams(
		gPhysics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	gSceneCpuRigidTriangleMesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		gPhysics->getPhysicsInsertionCallback());
	return gSceneCpuRigidTriangleMesh != NULL;
}

static bool createSceneCpuRotationalTriangleMesh()
{
	if(gSceneCpuRigidTriangleMesh || !gPhysics)
		return false;
	const PxVec3 vertices[] =
	{
		PxVec3(-1.0f, 0.0f, -0.1f),
		PxVec3(-1.0f, 0.0f, 0.1f),
		PxVec3(1.0f, 0.0f, 0.1f),
		PxVec3(1.0f, 0.0f, -0.1f)
	};
	const PxU32 triangles[] =
	{
		0, 1, 2,
		0, 2, 3
	};
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = 4;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = vertices;
	meshDesc.triangles.count = 2;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = triangles;
	PxCookingParams cookingParams(
		gPhysics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	gSceneCpuRigidTriangleMesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		gPhysics->getPhysicsInsertionCallback());
	return gSceneCpuRigidTriangleMesh != NULL;
}

static bool createSceneCpuRigidHeightField(bool reverseFeature)
{
	if(gSceneCpuRigidHeightField || !gPhysics)
		return false;
	PxHeightFieldSample samples[9];
	for(PxU32 sampleIndex = 0;
		sampleIndex < 9; ++sampleIndex)
	{
		samples[sampleIndex].height =
			reverseFeature ? PxI16(-3) : PxI16(0);
		samples[sampleIndex].materialIndex0 =
			PxBitAndByte(0);
		samples[sampleIndex].materialIndex1 =
			PxBitAndByte(0);
		samples[sampleIndex].setTessFlag();
	}
	if(reverseFeature)
		samples[4].height = PxI16(3);
	PxHeightFieldDesc heightFieldDesc;
	heightFieldDesc.nbRows = 3;
	heightFieldDesc.nbColumns = 3;
	heightFieldDesc.samples.data = samples;
	heightFieldDesc.samples.stride =
		sizeof(PxHeightFieldSample);
	gSceneCpuRigidHeightField = PxCreateHeightField(
		heightFieldDesc,
		gPhysics->getPhysicsInsertionCallback());
	return gSceneCpuRigidHeightField != NULL;
}

static bool addSceneStaticTriangleMesh(
	const PxVec3& center, bool reverseFeature)
{
	if(!createSceneCpuRigidTriangleMesh(reverseFeature))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxTriangleMeshGeometry(
				gSceneCpuRigidTriangleMesh),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool addSceneStaticHeightField(
	const PxVec3& center, bool reverseFeature)
{
	if(!createSceneCpuRigidHeightField(reverseFeature))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	const PxHeightFieldGeometry geometry(
		gSceneCpuRigidHeightField,
		PxMeshGeometryFlags(), 0.1f,
		reverseFeature ? 0.3f : 4.0f,
		reverseFeature ? 0.3f : 4.0f);
	if(!PxRigidActorExt::createExclusiveShape(
			*actor, geometry, *gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool addSceneStaticTriangleSurfacePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	bool heightField, bool reverseFeature)
{
	if(heightField
			? !createSceneCpuRigidHeightField(reverseFeature)
			: !createSceneCpuRigidTriangleMesh(reverseFeature))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(PxIdentity));
	if(!actor)
		return false;
	const PxVec3 centers[2] =
		{positiveCenter, negativeCenter};
	for(PxU32 i = 0; i < 2; ++i)
	{
		PxShape* shape = heightField
			? PxRigidActorExt::createExclusiveShape(
				*actor,
				PxHeightFieldGeometry(
					gSceneCpuRigidHeightField,
					PxMeshGeometryFlags(), 0.1f,
					reverseFeature ? 0.3f : 4.0f,
					reverseFeature ? 0.3f : 4.0f),
				*gMaterial)
			: PxRigidActorExt::createExclusiveShape(
				*actor,
				PxTriangleMeshGeometry(
					gSceneCpuRigidTriangleMesh),
				*gMaterial);
		if(!shape)
		{
			actor->release();
			return false;
		}
		shape->setLocalPose(PxTransform(centers[i]));
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool addSceneDynamicBox(
	const PxVec3& center, const PxVec3& halfExtent,
	bool startSleeping = true)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	if(!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	if(startSleeping)
		actor->putToSleep();
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = actor->getGlobalPose().p.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneDynamicSphere(
	const PxVec3& center, PxReal radius)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxSphereGeometry(radius), *gMaterial))
	{
		actor->release();
		return false;
	}
	if(!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	actor->putToSleep();
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneDynamicCapsule(
	const PxVec3& center, PxReal radius, PxReal halfHeight)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxCapsuleGeometry(radius, halfHeight), *gMaterial) ||
		!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	actor->putToSleep();
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneDynamicConvex(
	const PxVec3& center,
	SceneCpuRigidConvexFixture fixture =
		eSCENE_CPU_CONVEX_OWNER)
{
	if(!createSceneCpuRigidConvexMesh(fixture))
		return false;
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
			*gMaterial) ||
		!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	actor->putToSleep();
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicBox(
	const PxVec3& center, const PxVec3& halfExtent)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicSphere(
	const PxVec3& center, PxReal radius)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxSphereGeometry(radius), *gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicCapsule(
	const PxVec3& center, PxReal radius, PxReal halfHeight)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxCapsuleGeometry(radius, halfHeight), *gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicConvex(
	const PxVec3& center,
	SceneCpuRigidConvexFixture fixture =
		eSCENE_CPU_CONVEX_OWNER)
{
	if(!createSceneCpuRigidConvexMesh(fixture))
		return false;
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor,
			PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicTriangleMesh(const PxVec3& center)
{
	if(!createSceneCpuRigidTriangleMesh(false))
		return false;
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxTriangleMeshGeometry(
				gSceneCpuRigidTriangleMesh),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicHeightField(const PxVec3& center)
{
	if(!createSceneCpuRigidHeightField(false))
		return false;
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxHeightFieldGeometry(
				gSceneCpuRigidHeightField,
				PxMeshGeometryFlags(),
				0.1f, 4.0f, 4.0f),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneMovingKinematicTriangleSurfacePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal targetY, bool heightField,
	bool reverseFeature)
{
	if(heightField
			? !createSceneCpuRigidHeightField(reverseFeature)
			: !createSceneCpuRigidTriangleMesh(reverseFeature))
		return false;
	const PxVec3 centers[2] =
		{positiveCenter, negativeCenter};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i]));
		if(!actors[i])
			break;
		actors[i]->setRigidBodyFlag(
			PxRigidBodyFlag::eKINEMATIC, true);
		PxShape* shape = heightField
			? PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxHeightFieldGeometry(
					gSceneCpuRigidHeightField,
					PxMeshGeometryFlags(), 0.1f,
					reverseFeature ? 0.3f : 4.0f,
					reverseFeature ? 0.3f : 4.0f),
				*gMaterial)
			: PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxTriangleMeshGeometry(
					gSceneCpuRigidTriangleMesh),
				*gMaterial);
		if(!shape || !gScene->addActor(*actors[i]))
			break;
	}
	if(!actors[0] || !actors[1] ||
		actors[0]->getScene() != gScene ||
		actors[1]->getScene() != gScene)
	{
		for(PxU32 i = 0; i < 2; ++i)
			PX_RELEASE(actors[i]);
		return false;
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gSceneCpuKinematicCommandY = targetY;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.movingSphereTargetIssued = 1;
	for(PxU32 i = 0; i < 2; ++i)
		actors[i]->setKinematicTarget(
			PxTransform(PxVec3(
				centers[i].x, targetY, centers[i].z)));
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneRotatingKinematicTriangleSurfacePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	bool heightField)
{
	if(heightField
			? !createSceneCpuRigidHeightField(false)
			: !createSceneCpuRotationalTriangleMesh())
		return false;
	const PxReal startAngle = 0.5235987756f;
	const PxReal endAngle = 2.6179938780f;
	const PxQuat startRotation(
		startAngle, PxVec3(0.0f, 0.0f, 1.0f));
	const PxQuat endRotation(
		endAngle, PxVec3(0.0f, 0.0f, 1.0f));
	const PxVec3 centers[2] =
		{positiveCenter, negativeCenter};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i])
			break;
		actors[i]->setRigidBodyFlag(
			PxRigidBodyFlag::eKINEMATIC, true);
		PxShape* shape = heightField
			? PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxHeightFieldGeometry(
					gSceneCpuRigidHeightField,
					PxMeshGeometryFlags(), 0.1f,
					1.0f, 0.1f),
				*gMaterial)
			: PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxTriangleMeshGeometry(
					gSceneCpuRigidTriangleMesh),
				*gMaterial);
		if(!shape || !gScene->addActor(*actors[i]))
			break;
	}
	if(!actors[0] || !actors[1] ||
		actors[0]->getScene() != gScene ||
		actors[1]->getScene() != gScene)
	{
		for(PxU32 i = 0; i < 2; ++i)
			PX_RELEASE(actors[i]);
		return false;
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gSceneCpuKinematicCommandY = positiveCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.movingSphereTargetIssued = 1;
	for(PxU32 i = 0; i < 2; ++i)
		actors[i]->setKinematicTarget(
			PxTransform(centers[i], endRotation));
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneMovingKinematicFinitePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal targetY,
	PxReal radius,
	PxReal capsuleHalfHeight)
{
	const bool capsule = capsuleHalfHeight > 0.0f;
	if(!(capsule
			? addSceneKinematicCapsule(
				positiveCenter, radius, capsuleHalfHeight)
			: addSceneKinematicSphere(positiveCenter, radius)))
		return false;

	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(negativeCenter));
	if(!actor)
		return false;
	PxShape* shape = capsule
		? PxRigidActorExt::createExclusiveShape(
			*actor,
			PxCapsuleGeometry(radius, capsuleHalfHeight),
			*gMaterial)
		: PxRigidActorExt::createExclusiveShape(
			*actor, PxSphereGeometry(radius), *gMaterial);
	if(!shape)
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
	if(!gScene->addActor(*actor))
	{
		actor->release();
		return false;
	}
	gSceneCpuSecondDynamicActor = actor;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);

	gSceneCpuDynamicActor->setKinematicTarget(
		PxTransform(PxVec3(
			positiveCenter.x, targetY, positiveCenter.z)));
	gSceneCpuSecondDynamicActor->setKinematicTarget(
		PxTransform(PxVec3(
			negativeCenter.x, targetY, negativeCenter.z)));
	gMetrics.movingSphereTargetIssued = 1;
	return true;
}

static bool addSceneRotatingKinematicCapsulePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal radius,
	PxReal capsuleHalfHeight,
	const PxQuat& startRotation,
	const PxQuat& endRotation)
{
	const PxVec3 centers[2] =
	{
		positiveCenter,
		negativeCenter
	};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i])
		{
			for(PxU32 j = 0; j < i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
		if(!PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxCapsuleGeometry(radius, capsuleHalfHeight),
				*gMaterial))
		{
			for(PxU32 j = 0; j <= i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
		actors[i]->setRigidBodyFlag(
			PxRigidBodyFlag::eKINEMATIC, true);
		if(!gScene->addActor(*actors[i]))
		{
			for(PxU32 j = 0; j <= i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	for(PxU32 i = 0; i < 2; ++i)
		actors[i]->setKinematicTarget(
			PxTransform(centers[i], endRotation));
	gMetrics.movingSphereTargetIssued = 1;
	return true;
}

static bool addSceneRotatingKinematicConvexPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	const PxQuat& startRotation,
	const PxQuat& endRotation)
{
	if(!createSceneCpuRigidConvexMesh(
			eSCENE_CPU_CONVEX_ROTATIONAL))
		return false;
	const PxVec3 centers[2] =
	{
		positiveCenter,
		negativeCenter
	};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i] ||
			!PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
				*gMaterial))
		{
			for(PxU32 j = 0; j <= i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
		actors[i]->setRigidBodyFlag(
			PxRigidBodyFlag::eKINEMATIC, true);
		if(!gScene->addActor(*actors[i]))
		{
			for(PxU32 j = 0; j <= i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	for(PxU32 i = 0; i < 2; ++i)
		actors[i]->setKinematicTarget(
			PxTransform(centers[i], endRotation));
	gMetrics.movingSphereTargetIssued = 1;
	return true;
}

static bool addSceneDynamicFiniteSweepPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal radius,
	PxReal capsuleHalfHeight,
	PxReal launchSpeed)
{
	const bool capsule = capsuleHalfHeight > 0.0f;
	PxRigidDynamic* actors[2] = {NULL, NULL};
	const PxVec3 centers[2] = {
		positiveCenter, negativeCenter
	};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] =
			gPhysics->createRigidDynamic(PxTransform(centers[i]));
		PxShape* shape = actors[i]
			? (capsule
				? PxRigidActorExt::createExclusiveShape(
					*actors[i],
					PxCapsuleGeometry(radius, capsuleHalfHeight),
					*gMaterial)
				: PxRigidActorExt::createExclusiveShape(
					*actors[i],
					PxSphereGeometry(radius), *gMaterial))
			: NULL;
		if(!actors[i] || !shape ||
			!PxRigidBodyExt::setMassAndUpdateInertia(
				*actors[i], 1.0f))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
				PX_RELEASE(actors[0]);
			return false;
		}
		actors[i]->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		actors[i]->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
		actors[i]->setSolverIterationCounts(16, 1);
		actors[i]->setLinearVelocity(
			PxVec3(0.0f, launchSpeed, 0.0f));
		if(!gScene->addActor(*actors[i]))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
	}

	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.dynamicSphereSweepLaunched = 1;
	return gSceneCpuDynamicActor->getScene() == gScene &&
		gSceneCpuSecondDynamicActor->getScene() == gScene;
}

static bool addSceneDynamicRotatingCapsulePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal radius,
	PxReal capsuleHalfHeight,
	const PxQuat& startRotation,
	PxReal mass)
{
	const PxReal angularSpeed =
		(PxPi * 10.0f / 9.0f) / gHeadlessOptions.dt;
	const PxVec3 centers[2] =
	{
		positiveCenter,
		negativeCenter
	};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i] ||
			!PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxCapsuleGeometry(radius, capsuleHalfHeight),
				*gMaterial) ||
			!PxRigidBodyExt::setMassAndUpdateInertia(
				*actors[i], mass))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
		actors[i]->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		actors[i]->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y);
		actors[i]->setSolverIterationCounts(16, 1);
		actors[i]->setMaxAngularVelocity(300.0f);
		actors[i]->setAngularVelocity(
			PxVec3(0.0f, 0.0f, angularSpeed));
		if(!gScene->addActor(*actors[i]))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.dynamicSphereSweepLaunched = 1;
	return gSceneCpuDynamicActor->getScene() == gScene &&
		gSceneCpuSecondDynamicActor->getScene() == gScene;
}

static bool addSceneDynamicRotatingConvexPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	const PxQuat& startRotation,
	PxReal mass)
{
	if(!createSceneCpuRigidConvexMesh(
			eSCENE_CPU_CONVEX_ROTATIONAL))
		return false;
	const PxReal angularSpeed =
		(PxPi * 10.0f / 9.0f) / gHeadlessOptions.dt;
	const PxVec3 centers[2] =
	{
		positiveCenter,
		negativeCenter
	};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i] ||
			!PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
				*gMaterial) ||
			!PxRigidBodyExt::setMassAndUpdateInertia(
				*actors[i], mass))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
		actors[i]->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		actors[i]->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y);
		actors[i]->setSolverIterationCounts(16, 1);
		actors[i]->setMaxAngularVelocity(300.0f);
		actors[i]->setAngularVelocity(
			PxVec3(0.0f, 0.0f, angularSpeed));
		if(!gScene->addActor(*actors[i]))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.dynamicSphereSweepLaunched = 1;
	return gSceneCpuDynamicActor->getScene() == gScene &&
		gSceneCpuSecondDynamicActor->getScene() == gScene;
}

static bool addSceneMovingKinematicConvexPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal targetY,
	SceneCpuRigidConvexFixture fixture)
{
	if(!addSceneKinematicConvex(positiveCenter, fixture))
		return false;

	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(negativeCenter));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
	if(!gScene->addActor(*actor))
	{
		actor->release();
		return false;
	}
	gSceneCpuSecondDynamicActor = actor;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);

	gSceneCpuDynamicActor->setKinematicTarget(
		PxTransform(PxVec3(
			positiveCenter.x, targetY, positiveCenter.z)));
	gSceneCpuSecondDynamicActor->setKinematicTarget(
		PxTransform(PxVec3(
			negativeCenter.x, targetY, negativeCenter.z)));
	gMetrics.movingSphereTargetIssued = 1;
	return true;
}

static bool addSceneDynamicConvexSweepPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal launchSpeed,
	SceneCpuRigidConvexFixture fixture)
{
	if(!createSceneCpuRigidConvexMesh(fixture))
		return false;
	PxRigidDynamic* actors[2] = {NULL, NULL};
	const PxVec3 centers[2] = {
		positiveCenter, negativeCenter
	};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] =
			gPhysics->createRigidDynamic(PxTransform(centers[i]));
		PxShape* shape = actors[i]
			? PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
				*gMaterial)
			: NULL;
		if(!actors[i] || !shape ||
			!PxRigidBodyExt::setMassAndUpdateInertia(
				*actors[i], 1.0f))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
				PX_RELEASE(actors[0]);
			return false;
		}
		actors[i]->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		actors[i]->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
		actors[i]->setSolverIterationCounts(16, 1);
		actors[i]->setLinearVelocity(
			PxVec3(0.0f, launchSpeed, 0.0f));
		if(!gScene->addActor(*actors[i]))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
	}

	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.dynamicSphereSweepLaunched = 1;
	return gSceneCpuDynamicActor->getScene() == gScene &&
		gSceneCpuSecondDynamicActor->getScene() == gScene;
}

static bool addSceneSecondDynamicBox(
	const PxVec3& center, const PxVec3& halfExtent)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	if(!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	actor->putToSleep();
	gSceneCpuSecondDynamicActor = actor;
	gSceneCpuSecondDynamicInitialY = actor->getGlobalPose().p.y;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	return true;
}

static PxVec3 getSceneCpuVolumeCentroid(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getSimulationMesh())
		return PxVec3(0.0f);
	const PxU32 vertexCount =
		volume->getSimulationMesh()->getNbVertices();
	const PxVec4* positions =
		volume->getSimPositionInvMassBufferH();
	if(!positions || vertexCount == 0)
		return PxVec3(0.0f);
	PxVec3 centroid(0.0f);
	for(PxU32 i = 0; i < vertexCount; i++)
		centroid += positions[i].getXYZ();
	return centroid / PxReal(vertexCount);
}

static PxReal getSceneCpuVolumeCentroidY(
	PxDeformableVolume* volume)
{
	return getSceneCpuVolumeCentroid(volume).y;
}

static PxReal getSceneCpuVolumeCentroidY()
{
	return getSceneCpuVolumeCentroidY(gSceneCpuVolume);
}

static PxVec3 getSceneCpuAttachmentPoint(
	PxDeformableVolume* volume)
{
	if(!volume)
		return PxVec3(0.0f);
	const PxVec4* positions =
		volume->getSimPositionInvMassBufferH();
	if(!positions)
		return PxVec3(0.0f);
	if(!gSceneCpuElementAttachment)
		return positions[0].getXYZ();
	return
		positions[gSceneCpuAttachmentVertices[0]].getXYZ() *
			gSceneCpuAttachmentBarycentric.x +
		positions[gSceneCpuAttachmentVertices[1]].getXYZ() *
			gSceneCpuAttachmentBarycentric.y +
		positions[gSceneCpuAttachmentVertices[2]].getXYZ() *
			gSceneCpuAttachmentBarycentric.z +
		positions[gSceneCpuAttachmentVertices[3]].getXYZ() *
			gSceneCpuAttachmentBarycentric.w;
}

static PxVec3 getSceneCpuAttachmentPoint()
{
	return getSceneCpuAttachmentPoint(gSceneCpuVolume);
}

static bool updateSceneStaticChurn()
{
	if(gHeadlessOptions.caseName != "scene-volume-static-churn")
		return true;
	if(!gSceneCpuStaticActor)
		return false;

	if(gMetrics.completedFrames == 1)
	{
		PxShape* shape = NULL;
		if(gSceneCpuStaticActor->getNbShapes() != 1 ||
			gSceneCpuStaticActor->getShapes(&shape, 1) != 1 || !shape)
			return false;
		gSceneCpuStaticActor->detachShape(*shape);
		if(gSceneCpuStaticActor->getNbShapes() != 0)
			return false;
		gMetrics.sceneStaticShapeDetached = 1;

		if(!PxRigidActorExt::createExclusiveShape(
			*gSceneCpuStaticActor,
			PxBoxGeometry(PxVec3(20.0f, 0.5f, 20.0f)),
			*gMaterial))
			return false;
		gMetrics.sceneStaticShapeReattached =
			gSceneCpuStaticActor->getNbShapes() == 1 ? 1u : 0u;
		return gMetrics.sceneStaticShapeReattached == 1;
	}
	if(gMetrics.completedFrames == 2)
	{
		if(gSceneCpuStaticActor->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuStaticActor);
		gMetrics.sceneStaticActorRemoved =
			gSceneCpuStaticActor->getScene() == NULL ? 1u : 0u;
		return gMetrics.sceneStaticActorRemoved == 1;
	}
	if(gMetrics.completedFrames == 3)
	{
		if(gSceneCpuStaticActor->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuStaticActor);
		gMetrics.sceneStaticActorReadded =
			gSceneCpuStaticActor->getScene() == gScene ? 1u : 0u;
		return gMetrics.sceneStaticActorReadded == 1;
	}
	return true;
}

static bool updateSceneDynamicChurn()
{
	if(gHeadlessOptions.caseName != "scene-volume-dynamic-churn")
		return true;
	if(!gSceneCpuDynamicActor)
		return false;

	// Start churn only after the original sleeping actor has been woken by
	// the native soft contact edge.  Each mutation is separated by one Scene
	// step so stale edge/node ownership cannot be hidden by same-frame reuse.
	if(!gMetrics.sceneDynamicWokeBySoft)
		return true;

	if(!gMetrics.sceneDynamicShapeDetached)
	{
		PxShape* shape = NULL;
		if(gSceneCpuDynamicActor->getNbShapes() != 1 ||
			gSceneCpuDynamicActor->getShapes(&shape, 1) != 1 || !shape)
			return false;
		gSceneCpuDynamicActor->detachShape(*shape);
		if(gSceneCpuDynamicActor->getNbShapes() != 0)
			return false;
		gMetrics.sceneDynamicShapeDetached = 1;

		if(!PxRigidActorExt::createExclusiveShape(
			*gSceneCpuDynamicActor,
			PxBoxGeometry(PxVec3(4.0f, 0.25f, 4.0f)),
			*gMaterial))
			return false;
		gMetrics.sceneDynamicShapeReattached =
			gSceneCpuDynamicActor->getNbShapes() == 1 ? 1u : 0u;
		return gMetrics.sceneDynamicShapeReattached == 1;
	}

	if(!gMetrics.sceneDynamicActorRemoved)
	{
		if(gSceneCpuDynamicActor->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuDynamicActor);
		gMetrics.sceneDynamicActorRemoved =
			gSceneCpuDynamicActor->getScene() == NULL ? 1u : 0u;
		return gMetrics.sceneDynamicActorRemoved == 1;
	}

	if(!gMetrics.sceneDynamicActorReadded)
	{
		if(gSceneCpuDynamicActor->getScene() != NULL)
			return false;
		const PxReal reentryY =
			getSceneCpuVolumeCentroidY() - 0.75f;
		gSceneCpuDynamicActor->setGlobalPose(
			PxTransform(PxVec3(0.0f, reentryY, 0.0f)));
		gSceneCpuDynamicActor->setLinearVelocity(PxVec3(0.0f));
		gSceneCpuDynamicActor->setAngularVelocity(PxVec3(0.0f));
		gScene->addActor(*gSceneCpuDynamicActor);
		gMetrics.sceneDynamicActorReadded =
			gSceneCpuDynamicActor->getScene() == gScene ? 1u : 0u;
		if(!gMetrics.sceneDynamicActorReadded)
			return false;
		gSceneCpuDynamicActor->putToSleep();
		gMetrics.sceneDynamicReaddedSleeping =
			gSceneCpuDynamicActor->isSleeping() ? 1u : 0u;
		return gMetrics.sceneDynamicReaddedSleeping == 1;
	}

	return true;
}

static bool updateSceneMultiDynamicGate()
{
	const bool multiDynamicCase =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-dynamic-box";
	const bool multiSoftIslandCase =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-soft-islands";
	if(!multiDynamicCase && !multiSoftIslandCase)
		return true;
	if(!gSceneCpuDynamicActor || !gSceneCpuSecondDynamicActor)
		return false;

	// The fixture is a bounded ownership/merge gate, not a long-lived bridge
	// support scene.  Once both targets have woken and responded, remove both
	// actors on a frame boundary and keep simulating to exercise two-edge
	// teardown without allowing later asymmetric rebound to redefine the gate.
	if(gMetrics.completedFrames != 60)
		return true;
	if(!gMetrics.sceneDynamicWokeBySoft ||
		!gMetrics.sceneSecondDynamicWokeBySoft ||
		gMetrics.sceneDynamicMaxDrop <= 0.05f ||
		gMetrics.sceneSecondDynamicMaxDrop <= 0.05f)
		return false;
	if(gSceneCpuDynamicActor->getScene() != gScene ||
		gSceneCpuSecondDynamicActor->getScene() != gScene)
		return false;

	gScene->removeActor(*gSceneCpuDynamicActor);
	gMetrics.sceneDynamicActorRemoved =
		gSceneCpuDynamicActor->getScene() == NULL ? 1u : 0u;
	gScene->removeActor(*gSceneCpuSecondDynamicActor);
	gMetrics.sceneSecondDynamicActorRemoved =
		gSceneCpuSecondDynamicActor->getScene() == NULL ? 1u : 0u;
	if(multiSoftIslandCase)
	{
		PxDeformableVolume* volumes[2] =
		{
			gSceneCpuVolume,
			gSceneCpuSecondVolume
		};
		for(PxU32 volumeId = 0; volumeId < 2; ++volumeId)
		{
			PxDeformableVolume* volume = volumes[volumeId];
			if(!volume || !volume->getSimulationMesh())
				return false;
			volume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			PxVec4* velocities =
				volume->getSimVelocityBufferH();
			const PxU32 vertexCount =
				volume->getSimulationMesh()->getNbVertices();
			if(!velocities)
				return false;
			for(PxU32 i = 0; i < vertexCount; ++i)
			{
				const PxReal invMass = velocities[i].w;
				velocities[i] =
					PxVec4(0.0f, 0.0f, 0.0f, invMass);
			}
			volume->markDirty(
				PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		}
	}
	return gMetrics.sceneDynamicActorRemoved == 1 &&
		gMetrics.sceneSecondDynamicActorRemoved == 1;
}

static bool createAdditionalSceneCpuVolume(
	PxDeformableVolume*& volume, const PxVec3& translation,
	PxScene* targetScene = NULL)
{
	if(!gSceneCpuVolumeMesh || !gSceneCpuVolumeMaterial)
		return false;
	PxScene* scene = targetScene ? targetScene : gScene;
	if(!scene)
		return false;

	volume = gPhysics->createDeformableVolume(
		PxDeformableVolumeBackend::eCPU_AVBD);
	if(!volume ||
		volume->getDeformableVolumeBackend() !=
			PxDeformableVolumeBackend::eCPU_AVBD ||
		volume->getCudaContextManager() != NULL)
		return false;

	const PxShapeFlags shapeFlags =
		PxShapeFlag::eVISUALIZATION |
		PxShapeFlag::eSCENE_QUERY_SHAPE |
		PxShapeFlag::eSIMULATION_SHAPE;
	const PxTetrahedronMeshGeometry geometry(
		gSceneCpuVolumeMesh->getCollisionMesh());
	PxDeformableVolumeMaterial* material =
		gSceneCpuVolumeMaterial;
	PxShape* shape = gPhysics->createShape(
		geometry, &material, 1, true, shapeFlags);
	if(!shape)
		return false;
	const bool shapeAttached = volume->attachShape(*shape);
	shape->release();
	if(!shapeAttached ||
		!volume->attachSimulationMesh(
			*gSceneCpuVolumeMesh->getSimulationMesh(),
			*gSceneCpuVolumeMesh->getDeformableVolumeAuxData()))
		return false;

	PxVec4* simPositionInvMass =
		volume->getSimPositionInvMassBufferH();
	PxVec4* simVelocity =
		volume->getSimVelocityBufferH();
	PxVec4* collisionPositionInvMass =
		volume->getPositionInvMassBufferH();
	PxVec4* collisionRestPosition =
		volume->getRestPositionBufferH();
	if(!simPositionInvMass || !simVelocity ||
		!collisionPositionInvMass || !collisionRestPosition)
		return false;

	const PxTetrahedronMesh* simulationMesh =
		volume->getSimulationMesh();
	const PxVec3* simulationVertices =
		simulationMesh->getVertices();
	const PxU32 simulationVertexCount =
		simulationMesh->getNbVertices();
	PxReal* cookedInvMass =
		volume->getDeformableVolumeAuxData()->
			getGridModelInvMass();
	for(PxU32 i = 0; i < simulationVertexCount; i++)
	{
		const PxReal invMass = cookedInvMass
			? PxMax(cookedInvMass[i], 0.0f) : 1.0f;
		simPositionInvMass[i] =
			PxVec4(simulationVertices[i] + translation, invMass);
		simVelocity[i] =
			PxVec4(0.0f, 0.0f, 0.0f, invMass);
	}
	PxDeformableVolumeExt::updateMass(
		*volume, 100.0f, 50.0f, simPositionInvMass);
	for(PxU32 i = 0; i < simulationVertexCount; i++)
		simVelocity[i].w = simPositionInvMass[i].w;
	PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
		*volume, simPositionInvMass, collisionPositionInvMass);
	const PxU32 collisionVertexCount =
		volume->getCollisionMesh()->getNbVertices();
	for(PxU32 i = 0; i < collisionVertexCount; i++)
		collisionRestPosition[i] = collisionPositionInvMass[i];
	volume->markDirty(PxDeformableVolumeDataFlag::eALL);
	volume->setSolverIterationCounts(8, 1);
	scene->addActor(*volume);
	return volume->getScene() == scene;
}

static bool initializeVolumeSkinning()
{
	gVolumeSkinningBindings.clear();
	gVolumeSkinningTriangles.clear();
	gVolumeSkinningPositions.clear();
	gVolumeSkinningNormals.clear();
	gVolumeSkinningInitialPositions.clear();
	gVolumeAvbdSkinningRenderData =
		VolumeAvbdSkinningRenderData();
	gVolumeSkinningMetrics = VolumeSkinningMetrics();

	if(!gSceneCpuVolume ||
		!gSceneCpuVolume->getSimulationMesh())
		return false;
	const PxTetrahedronMesh* simulationMesh =
		gSceneCpuVolume->getSimulationMesh();
	if(simulationMesh->getNbTetrahedrons() == 0)
		return false;

	PxU32 tetrahedron[4];
	const bool has16BitIndices =
		simulationMesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	if(has16BitIndices)
	{
		const PxU16* indices =
			static_cast<const PxU16*>(
				simulationMesh->getTetrahedrons());
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
			tetrahedron[endpoint] = indices[endpoint];
	}
	else
	{
		const PxU32* indices =
			static_cast<const PxU32*>(
				simulationMesh->getTetrahedrons());
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
			tetrahedron[endpoint] = indices[endpoint];
	}

	Snippets::appendTetrahedronSurfaceSkinning(
		tetrahedron, 8, gVolumeSkinningBindings,
		gVolumeSkinningTriangles);
	const PxVec4* positions =
		gSceneCpuVolume->getSimPositionInvMassBufferH();
	if(!Snippets::evaluateTetrahedronSkinning(
		positions, simulationMesh->getNbVertices(),
		gVolumeSkinningBindings, gVolumeSkinningTriangles,
		gVolumeSkinningPositions, gVolumeSkinningNormals))
		return false;
	gVolumeSkinningInitialPositions =
		gVolumeSkinningPositions;
	gVolumeAvbdSkinningRenderData.positions =
		gVolumeSkinningPositions.begin();
	gVolumeAvbdSkinningRenderData.normals =
		gVolumeSkinningNormals.begin();
	gVolumeAvbdSkinningRenderData.triangles =
		gVolumeSkinningTriangles.begin();
	gVolumeAvbdSkinningRenderData.numVertices =
		gVolumeSkinningPositions.size();
	gVolumeAvbdSkinningRenderData.numTriangles =
		gVolumeSkinningTriangles.size() / 3;
	gVolumeSkinningMetrics.initialized = 1;
	gVolumeSkinningMetrics.vertices =
		gVolumeSkinningPositions.size();
	gVolumeSkinningMetrics.triangles =
		gVolumeSkinningTriangles.size() / 3;
	return true;
}

static bool updateVolumeSkinning()
{
	if(gHeadlessOptions.caseName != "scene-volume-skinning")
		return true;
	if(!gSceneCpuVolume ||
		!gSceneCpuVolume->getSimulationMesh() ||
		!gVolumeSkinningMetrics.initialized)
		return false;
	const PxTetrahedronMesh* simulationMesh =
		gSceneCpuVolume->getSimulationMesh();
	if(!Snippets::evaluateTetrahedronSkinning(
		gSceneCpuVolume->getSimPositionInvMassBufferH(),
		simulationMesh->getNbVertices(),
		gVolumeSkinningBindings, gVolumeSkinningTriangles,
		gVolumeSkinningPositions, gVolumeSkinningNormals))
		return false;

	gVolumeSkinningMetrics.evaluatedFrames++;
	bool finite = true;
	for(PxU32 i = 0; i < gVolumeSkinningPositions.size(); ++i)
	{
		if(!gVolumeSkinningPositions[i].isFinite() ||
			!gVolumeSkinningNormals[i].isFinite())
		{
			finite = false;
			break;
		}
		gVolumeSkinningMetrics.maxDisplacement = PxMax(
			gVolumeSkinningMetrics.maxDisplacement,
			(gVolumeSkinningPositions[i] -
				gVolumeSkinningInitialPositions[i]).magnitude());
	}
	if(finite)
		gVolumeSkinningMetrics.finiteFrames++;
	gVolumeAvbdSkinningRenderData.positions =
		gVolumeSkinningPositions.begin();
	gVolumeAvbdSkinningRenderData.normals =
		gVolumeSkinningNormals.begin();
	return finite;
}

static bool setSceneCpuVolumeVelocity(
	PxDeformableVolume& volume, const PxVec3& velocity);
static bool setSceneCpuVolumeDeformingReverseVelocity(
	PxDeformableVolume& volume, PxReal speed);
static PxReal getSceneCpuVolumeMinY(
	PxDeformableVolume* volume);

static bool initSceneCpuVolumeLifecycle()
{
	gSceneCpuSphereReverseSweptInitialPositions.reset();
	gSceneCpuDeformingReverseSweptFreeEndPositions.reset();
	gSceneCpuCapsuleRotationalSweptInitialPositions.reset();
	const bool partialElementFilterCase =
		gHeadlessOptions.caseName ==
			"scene-volume-partial-element-filter";
	PxArray<PxVec3> surfaceVertices;
	PxArray<PxU32> surfaceTriangles;
	auto appendTetraSurface = [&](
		const PxVec3& offset)
	{
		const PxU32 vertexBase = surfaceVertices.size();
		surfaceVertices.pushBack(
			offset + PxVec3(0.0f, 0.0f, 0.0f));
		surfaceVertices.pushBack(
			offset + PxVec3(1.0f, 0.0f, 0.0f));
		surfaceVertices.pushBack(
			offset + PxVec3(0.0f, 1.0f, 0.0f));
		surfaceVertices.pushBack(
			offset + PxVec3(0.0f, 0.0f, 1.0f));
		const PxU32 indices[] =
		{
			0, 2, 1,
			0, 1, 3,
			0, 3, 2,
			1, 2, 3
		};
		for(PxU32 i = 0;
			i < sizeof(indices) / sizeof(indices[0]); ++i)
			surfaceTriangles.pushBack(vertexBase + indices[i]);
	};
	if(partialElementFilterCase)
	{
		appendTetraSurface(PxVec3(-3.0f, 0.0f, 0.0f));
		appendTetraSurface(PxVec3(2.0f, 0.0f, 0.0f));
	}
	else
		appendTetraSurface(PxVec3(0.0f));

	PxSimpleTriangleMesh surfaceMesh;
	surfaceMesh.points.count = surfaceVertices.size();
	surfaceMesh.points.data = surfaceVertices.begin();
	surfaceMesh.points.stride = sizeof(PxVec3);
	surfaceMesh.triangles.count = surfaceTriangles.size() / 3;
	surfaceMesh.triangles.data = surfaceTriangles.begin();
	surfaceMesh.triangles.stride = 3 * sizeof(PxU32);

	PxCookingParams cookingParams(gPhysics->getTolerancesScale());
	// CPU AVBD consumes shared deformable topology and embedding data without
	// requesting the optional GRB/BV32 payload.
	cookingParams.buildGPUData = false;
	cookingParams.meshWeldTolerance = 0.001f;
	cookingParams.meshPreprocessParams =
		PxMeshPreprocessingFlag::eWELD_VERTICES;
	if(partialElementFilterCase)
	{
		const PxU32 collisionTetrahedra[] =
		{
			0, 1, 2, 3,
			4, 5, 6, 7
		};
		PxTetrahedronMeshDesc collisionMeshDesc;
		collisionMeshDesc.points.count = surfaceVertices.size();
		collisionMeshDesc.points.data = surfaceVertices.begin();
		collisionMeshDesc.points.stride = sizeof(PxVec3);
		collisionMeshDesc.tetrahedrons.count = 2;
		collisionMeshDesc.tetrahedrons.data =
			collisionTetrahedra;
		collisionMeshDesc.tetrahedrons.stride =
			4 * sizeof(PxU32);

		PxArray<PxVec3> simulationVertices =
			surfaceVertices;
		simulationVertices.pushBack(
			PxVec3(-2.75f, 0.25f, 0.25f));
		simulationVertices.pushBack(
			PxVec3(2.25f, 0.25f, 0.25f));
		const PxU32 simulationTetrahedra[] =
		{
			0, 1, 2, 8,
			0, 1, 8, 3,
			0, 8, 2, 3,
			8, 1, 2, 3,
			4, 5, 6, 9,
			4, 5, 9, 7,
			4, 9, 6, 7,
			9, 5, 6, 7
		};
		PxTetrahedronMeshDesc simulationMeshDesc;
		simulationMeshDesc.points.count =
			simulationVertices.size();
		simulationMeshDesc.points.data =
			simulationVertices.begin();
		simulationMeshDesc.points.stride = sizeof(PxVec3);
		simulationMeshDesc.tetrahedrons.count = 8;
		simulationMeshDesc.tetrahedrons.data =
			simulationTetrahedra;
		simulationMeshDesc.tetrahedrons.stride =
			4 * sizeof(PxU32);
		PxDeformableVolumeSimulationDataDesc simulationDataDesc;
		gSceneCpuVolumeMesh = PxCreateDeformableVolumeMesh(
			cookingParams, simulationMeshDesc,
			collisionMeshDesc,
			simulationDataDesc,
			gPhysics->getPhysicsInsertionCallback());
	}
	else
	{
		gSceneCpuVolumeMesh =
			PxDeformableVolumeExt::createDeformableVolumeMeshNoVoxels(
				cookingParams, surfaceMesh,
				gPhysics->getPhysicsInsertionCallback(), 1.5f, true);
	}
	if(!gSceneCpuVolumeMesh)
		return false;

	gSceneCpuVolumeMaterial =
		gPhysics->createDeformableVolumeMaterial(
			2.0e5f, 0.3f,
			(gHeadlessOptions.caseName ==
					"scene-volume-motion-controls" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-max-depenetration-velocity" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-sphere-reverse-feature" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-capsule-reverse-feature" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-convex-reverse-feature" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-triangle-mesh-reverse-feature" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-heightfield-reverse-feature" ||
			 isSceneCpuVolumeSpeculativeCcdCase(
				gHeadlessOptions.caseName) ||
			 isSceneCpuVolumeSphereReverseSweptCcdCase(
				gHeadlessOptions.caseName))
				? 0.0f : 0.2f,
			0.01f);
	if(!gSceneCpuVolumeMaterial)
		return false;
	const PxDeformableVolumeMaterialModel::Enum materialModel =
		gHeadlessOptions.caseName == "scene-volume-corotational"
			? PxDeformableVolumeMaterialModel::eCO_ROTATIONAL
			: PxDeformableVolumeMaterialModel::eNEO_HOOKEAN;
	gSceneCpuVolumeMaterial->setMaterialModel(materialModel);
	if(gSceneCpuVolumeMaterial->getMaterialModel() != materialModel)
		return false;

	gSceneCpuVolume = gPhysics->createDeformableVolume(
		PxDeformableVolumeBackend::eCPU_AVBD);
	if(!gSceneCpuVolume)
		return false;
	gMetrics.sceneActorCreated = 1;
	if(gSceneCpuVolume->getDeformableVolumeBackend() !=
		PxDeformableVolumeBackend::eCPU_AVBD ||
		gSceneCpuVolume->getCudaContextManager() != NULL)
		return false;

	PxShapeFlags shapeFlags =
		PxShapeFlag::eVISUALIZATION |
		PxShapeFlag::eSCENE_QUERY_SHAPE |
		PxShapeFlag::eSIMULATION_SHAPE;
	PxTetrahedronMeshGeometry geometry(
		gSceneCpuVolumeMesh->getCollisionMesh());
	PxDeformableVolumeMaterial* material = gSceneCpuVolumeMaterial;
	PxShape* shape = gPhysics->createShape(
		geometry, &material, 1, true, shapeFlags);
	if(!shape)
		return false;
	const bool shapeAttached = gSceneCpuVolume->attachShape(*shape);
	shape->release();
	if(!shapeAttached)
		return false;
	gMetrics.sceneShapeAttached = 1;

	if(!gSceneCpuVolume->attachSimulationMesh(
		*gSceneCpuVolumeMesh->getSimulationMesh(),
		*gSceneCpuVolumeMesh->getDeformableVolumeAuxData()))
		return false;
	gMetrics.sceneSimulationMeshAttached = 1;

	PxVec4* simPositionInvMass =
		gSceneCpuVolume->getSimPositionInvMassBufferH();
	PxVec4* simVelocity =
		gSceneCpuVolume->getSimVelocityBufferH();
	PxVec4* collisionPositionInvMass =
		gSceneCpuVolume->getPositionInvMassBufferH();
	PxVec4* collisionRestPosition =
		gSceneCpuVolume->getRestPositionBufferH();
	if(!simPositionInvMass || !simVelocity ||
		!collisionPositionInvMass || !collisionRestPosition)
		return false;

	const bool multiSoftIslandCase =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-soft-islands";
	const bool mixedSleepIslandCase =
		gHeadlessOptions.caseName ==
			"scene-volume-mixed-sleep-islands";
	const bool softChurnCase =
		gHeadlessOptions.caseName ==
			"scene-volume-soft-churn";
	const bool bufferMutationCase =
		gHeadlessOptions.caseName ==
			"scene-volume-buffer-mutation";
	const bool worldElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-world-element-attachment";
	const bool rigidElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-rigid-element-attachment";
	const bool staticElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-static-element-attachment";
	const bool kinematicElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-element-attachment";
	const bool articulationElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-element-attachment";
	const bool rigidAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-rigid-attachment" ||
		rigidElementAttachmentCase;
	const bool staticAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-static-attachment" ||
		staticElementAttachmentCase;
	const bool kinematicAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-attachment" ||
		kinematicElementAttachmentCase;
	const bool articulationAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-attachment" ||
		articulationElementAttachmentCase;
	const bool attachmentCase =
		rigidAttachmentCase || staticAttachmentCase ||
		kinematicAttachmentCase ||
		articulationAttachmentCase;
	const bool multiSceneIsolationCase =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-scene-isolation";
	const bool softSoftWakeCase =
		gHeadlessOptions.caseName ==
			"scene-volume-soft-soft-wake";
	const bool softPairAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-volume-attachment";
	const bool fullKinematicTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-full-kinematic-target";
	const bool partialKinematicTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-partial-kinematic-target";
	const bool volumeKinematicTargetCase =
		fullKinematicTargetCase || partialKinematicTargetCase;
	const bool motionControlsCase =
		gHeadlessOptions.caseName ==
			"scene-volume-motion-controls";
	const bool maxDepenetrationVelocityCase =
		gHeadlessOptions.caseName ==
			"scene-volume-max-depenetration-velocity";
	const bool triangleSurfaceSweptCcdCase =
		isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool triangleSurfaceReverseSweptCcdCase =
		isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool staticTriangleSurfaceSweptCcdCase =
		isSceneCpuVolumeStaticTriangleSurfaceSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool heightFieldSweptCcdCase =
		isSceneCpuVolumeHeightFieldSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool rotationalTriangleSurfaceSweptCcdCase =
		isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool sphereReverseSweptCcdCase =
		isSceneCpuVolumeSphereReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool deformingReverseSweptCcdCase =
		isSceneCpuVolumeDeformingReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool convexReverseSweptCcdCase =
		isSceneCpuVolumeConvexReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool rotationalCapsuleReverseSweptCcdCase =
		isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool rotationalConvexReverseSweptCcdCase =
		isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool rotationalFiniteReverseSweptCcdCase =
		rotationalCapsuleReverseSweptCcdCase ||
		rotationalConvexReverseSweptCcdCase;
	const bool staticSphereReverseSweptCcdCase =
		deformingReverseSweptCcdCase ||
		gHeadlessOptions.caseName ==
			"scene-volume-static-sphere-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-static-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-static-convex-reverse-swept-ccd";
	const bool kinematicSphereReverseSweptCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-convex-reverse-swept-ccd";
	const bool dynamicSphereReverseSweptCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-sphere-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-convex-reverse-swept-ccd";
	const bool movingSphereReverseSweptCcdCase =
		kinematicSphereReverseSweptCcdCase ||
		dynamicSphereReverseSweptCcdCase;
	const bool speculativeCcdCase =
		isSceneCpuVolumeSpeculativeCcdCase(
			gHeadlessOptions.caseName) ||
		sphereReverseSweptCcdCase;
	const bool sphereSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-sphere-speculative-ccd";
	const bool capsuleSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-capsule-speculative-ccd";
	const bool convexSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-convex-speculative-ccd";
	const bool finiteSmoothSpeculativeCcdCase =
		sphereSpeculativeCcdCase || capsuleSpeculativeCcdCase ||
		convexSpeculativeCcdCase;
	const bool rotatingKinematicCapsuleSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-capsule-speculative-ccd";
	const bool dynamicRotatingCapsuleSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-capsule-relative-swept-ccd";
	const bool rotatingKinematicConvexSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-convex-speculative-ccd";
	const bool dynamicRotatingConvexSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-convex-relative-swept-ccd";
	const bool rotationalConvexSpeculativeCcdCase =
		rotatingKinematicConvexSpeculativeCcdCase ||
		dynamicRotatingConvexSpeculativeCcdCase;
	const bool rotationalCapsuleSpeculativeCcdCase =
		rotatingKinematicCapsuleSpeculativeCcdCase ||
		dynamicRotatingCapsuleSpeculativeCcdCase;
	const bool rotationalFiniteSpeculativeCcdCase =
		rotationalCapsuleSpeculativeCcdCase ||
		rotationalConvexSpeculativeCcdCase;
	const bool movingKinematicFiniteSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-moving-kinematic-sphere-speculative-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-moving-kinematic-capsule-speculative-ccd" ||
		rotatingKinematicCapsuleSpeculativeCcdCase ||
		rotatingKinematicConvexSpeculativeCcdCase ||
		gHeadlessOptions.caseName ==
			"scene-volume-moving-kinematic-convex-speculative-ccd";
	const bool dynamicFiniteRelativeSweptCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-sphere-relative-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-capsule-relative-swept-ccd" ||
		dynamicRotatingCapsuleSpeculativeCcdCase ||
		dynamicRotatingConvexSpeculativeCcdCase ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-convex-relative-swept-ccd";
	const bool sphereReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-sphere-reverse-feature";
	const bool capsuleReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-capsule-reverse-feature";
	const bool convexReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-convex-reverse-feature";
	const bool triangleMeshReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-triangle-mesh-reverse-feature";
	const bool heightFieldReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-heightfield-reverse-feature";
	const bool smoothReverseFeatureCase =
		sphereReverseFeatureCase || capsuleReverseFeatureCase ||
		convexReverseFeatureCase ||
		triangleMeshReverseFeatureCase ||
		heightFieldReverseFeatureCase;
	const bool twoSoftVolumeCase =
		multiSoftIslandCase || mixedSleepIslandCase ||
		softChurnCase || multiSceneIsolationCase ||
		softSoftWakeCase || softPairAttachmentCase ||
		motionControlsCase || maxDepenetrationVelocityCase ||
		speculativeCcdCase || smoothReverseFeatureCase;
	const bool softSleepCase =
		gHeadlessOptions.caseName == "scene-volume-sleep-wake" ||
		gHeadlessOptions.caseName == "scene-volume-rigid-wake" ||
		isSceneCpuVolumeKinematicRigidCase(
			gHeadlessOptions.caseName) ||
		mixedSleepIslandCase || softChurnCase ||
		bufferMutationCase || attachmentCase ||
		multiSceneIsolationCase ||
		softSoftWakeCase || softPairAttachmentCase ||
		volumeKinematicTargetCase;
	const PxVec3 translation =
		smoothReverseFeatureCase
			? PxVec3(-2.0f, 0.34f, 0.0f)
			: triangleSurfaceSweptCcdCase
			? PxVec3(
				-2.0f,
				staticTriangleSurfaceSweptCcdCase
					? 1.1f : 0.0f,
				0.0f)
			: sphereReverseSweptCcdCase
			? PxVec3(
				-2.0f,
				deformingReverseSweptCcdCase
					? 0.45f
				: staticSphereReverseSweptCcdCase
					? (convexReverseSweptCcdCase
						? 1.1f : 0.55f)
					: 0.0f,
				0.0f)
			: rotationalFiniteSpeculativeCcdCase
			? PxVec3(-2.0f, 0.0f, 0.0f)
			: speculativeCcdCase
			? PxVec3(-2.0f, 1.2f, 0.0f)
			: maxDepenetrationVelocityCase
			? PxVec3(-3.0f, -1.05f, 0.0f)
			: PxVec3(
				twoSoftVolumeCase && !softSoftWakeCase &&
					!softPairAttachmentCase ? -10.0f : 0.0f,
				4.0f, 0.0f);
	const PxTetrahedronMesh* simulationMesh =
		gSceneCpuVolume->getSimulationMesh();
	const PxVec3* simulationVertices = simulationMesh->getVertices();
	const PxU32 simulationVertexCount = simulationMesh->getNbVertices();
	gSceneCpuElementAttachment =
		worldElementAttachmentCase || rigidElementAttachmentCase ||
		staticElementAttachmentCase ||
		kinematicElementAttachmentCase ||
		articulationElementAttachmentCase ||
		softPairAttachmentCase;
	if(gSceneCpuElementAttachment)
	{
		if(simulationMesh->getNbTetrahedrons() == 0)
			return false;
		const bool has16BitIndices =
			simulationMesh->getTetrahedronMeshFlags() &
				PxTetrahedronMeshFlag::e16_BIT_INDICES;
		if(has16BitIndices)
		{
			const PxU16* tetrahedra =
				static_cast<const PxU16*>(
					simulationMesh->getTetrahedrons());
			for(PxU32 endpoint = 0; endpoint < 4; endpoint++)
				gSceneCpuAttachmentVertices[endpoint] =
					tetrahedra[endpoint];
		}
		else
		{
			const PxU32* tetrahedra =
				static_cast<const PxU32*>(
					simulationMesh->getTetrahedrons());
			for(PxU32 endpoint = 0; endpoint < 4; endpoint++)
				gSceneCpuAttachmentVertices[endpoint] =
					tetrahedra[endpoint];
		}
		for(PxU32 endpoint = 0; endpoint < 4; endpoint++)
		{
			if(gSceneCpuAttachmentVertices[endpoint] >=
				simulationVertexCount)
				return false;
		}
	}
	PxReal* cookedInvMass =
		gSceneCpuVolume->getDeformableVolumeAuxData()->
			getGridModelInvMass();
	PxU32 deformingReverseMovingVertex = 0;
	if(deformingReverseSweptCcdCase &&
		simulationVertexCount > 0)
	{
		PxReal minimumRestRadiusSq =
			simulationVertices[0].magnitudeSquared();
		for(PxU32 i = 1; i < simulationVertexCount; ++i)
		{
			const PxReal restRadiusSq =
				simulationVertices[i].magnitudeSquared();
			if(restRadiusSq < minimumRestRadiusSq)
			{
				minimumRestRadiusSq = restRadiusSq;
				deformingReverseMovingVertex = i;
			}
		}
	}
	for(PxU32 i = 0; i < simulationVertexCount; i++)
	{
		const PxReal invMass = cookedInvMass
			? PxMax(cookedInvMass[i], 0.0f) : 1.0f;
		simPositionInvMass[i] =
			PxVec4(simulationVertices[i] + translation, invMass);
		simVelocity[i] = PxVec4(
			motionControlsCase
				? PxVec3(10.0f, 0.0f, 0.0f)
				: smoothReverseFeatureCase
				? PxVec3(0.0f, -2.0f, 0.0f)
				: deformingReverseSweptCcdCase
				? (i == deformingReverseMovingVertex
					? PxVec3(0.0f, -720.0f, 0.0f)
					: PxVec3(0.0f))
				: speculativeCcdCase
				? PxVec3(
					0.0f,
					triangleSurfaceSweptCcdCase
						? (staticTriangleSurfaceSweptCcdCase
							? (triangleSurfaceReverseSweptCcdCase &&
									heightFieldSweptCcdCase
								? -80.0f : -132.0f)
							: 0.0f) :
					staticSphereReverseSweptCcdCase
						? -132.0f :
					(movingKinematicFiniteSpeculativeCcdCase ||
					 dynamicFiniteRelativeSweptCcdCase ||
					 movingSphereReverseSweptCcdCase)
						? 0.0f :
					capsuleSpeculativeCcdCase
						? -170.0f :
					convexSpeculativeCcdCase
						? -220.0f :
					finiteSmoothSpeculativeCcdCase
						? -160.0f : -120.0f,
					0.0f)
				: PxVec3(0.0f),
			invMass);
	}
	PxDeformableVolumeExt::updateMass(
		*gSceneCpuVolume, 100.0f, 50.0f, simPositionInvMass);
	for(PxU32 i = 0; i < simulationVertexCount; i++)
		simVelocity[i].w = simPositionInvMass[i].w;
	PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
		*gSceneCpuVolume, simPositionInvMass,
		collisionPositionInvMass);
	const PxU32 collisionVertexCount =
		gSceneCpuVolume->getCollisionMesh()->getNbVertices();
	for(PxU32 i = 0; i < collisionVertexCount; i++)
		collisionRestPosition[i] = collisionPositionInvMass[i];
	if(sphereReverseSweptCcdCase ||
		triangleSurfaceReverseSweptCcdCase ||
		rotationalTriangleSurfaceSweptCcdCase)
	{
		gSceneCpuSphereReverseSweptInitialPositions.resize(
			collisionVertexCount);
		for(PxU32 i = 0; i < collisionVertexCount; ++i)
			gSceneCpuSphereReverseSweptInitialPositions[i] =
				collisionPositionInvMass[i].getXYZ();
	}
	if(deformingReverseSweptCcdCase)
	{
		PxArray<PxVec4> freeEndSimulationPositions;
		PxArray<PxVec4> freeEndCollisionPositions;
		freeEndSimulationPositions.resize(simulationVertexCount);
		freeEndCollisionPositions.resize(collisionVertexCount);
		for(PxU32 i = 0; i < simulationVertexCount; ++i)
		{
			freeEndSimulationPositions[i] = PxVec4(
				simPositionInvMass[i].getXYZ() +
					simVelocity[i].getXYZ() *
						gHeadlessOptions.dt,
				simPositionInvMass[i].w);
		}
		PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
			*gSceneCpuVolume,
			freeEndSimulationPositions.begin(),
			freeEndCollisionPositions.begin());
		gSceneCpuDeformingReverseSweptFreeEndPositions.resize(
			collisionVertexCount);
		for(PxU32 i = 0; i < collisionVertexCount; ++i)
			gSceneCpuDeformingReverseSweptFreeEndPositions[i] =
				freeEndCollisionPositions[i].getXYZ();
	}
	if(rotationalFiniteSpeculativeCcdCase ||
		rotationalFiniteReverseSweptCcdCase)
	{
		gSceneCpuCapsuleRotationalSweptInitialPositions.resize(
			collisionVertexCount);
		for(PxU32 i = 0; i < collisionVertexCount; ++i)
			gSceneCpuCapsuleRotationalSweptInitialPositions[i] =
				collisionPositionInvMass[i].getXYZ();
	}
	if(volumeKinematicTargetCase)
	{
		gSceneCpuVolumeKinematicTargets.resize(
			simulationVertexCount);
		gSceneCpuVolumeKinematicInitial.resize(
			simulationVertexCount);
		gSceneCpuVolumePartialProbe =
			partialKinematicTargetCase
				? simulationVertexCount - 1
				: PX_MAX_U32;
		for(PxU32 i = 0; i < simulationVertexCount; ++i)
		{
			const PxVec3 initial =
				simPositionInvMass[i].getXYZ();
			gSceneCpuVolumeKinematicInitial[i] = initial;
			const bool inactiveProbe =
				partialKinematicTargetCase &&
				i == gSceneCpuVolumePartialProbe;
			gSceneCpuVolumeKinematicTargets[i] =
				PxVec4(
					initial +
						(inactiveProbe
							? PxVec3(0.0f, 5.0f, 0.0f)
							: PxVec3(0.0f)),
					inactiveProbe ? 1.0f : 0.0f);
		}
		gSceneCpuVolume->setKinematicTargetBufferH(
			gSceneCpuVolumeKinematicTargets.begin());
		if(fullKinematicTargetCase)
			gSceneCpuVolume->setDeformableBodyFlag(
				PxDeformableBodyFlag::eKINEMATIC, true);
		else
			gSceneCpuVolume->setDeformableVolumeFlag(
				PxDeformableVolumeFlag::
					ePARTIALLY_KINEMATIC, true);
		gMetrics.sceneVolumeTargetBound = 1;
	}
	gSceneCpuVolume->markDirty(PxDeformableVolumeDataFlag::eALL);
	gSceneCpuVolume->setSolverIterationCounts(
		partialElementFilterCase ? 16u : 8u, 1);
	if(gHeadlessOptions.caseName == "scene-volume-skinning" &&
		!initializeVolumeSkinning())
		return false;
	if(motionControlsCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setLinearDamping(0.0f);
		gSceneCpuVolume->setMaxLinearVelocity(1.0f);
		gSceneCpuVolume->setSettlingThreshold(0.0f);
		gSceneCpuVolume->setSettlingDamping(0.0f);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(maxDepenetrationVelocityCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setLinearDamping(0.0f);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
		gSceneCpuVolume->setMaxDepenetrationVelocity(0.12f);
	}
	if(speculativeCcdCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		gSceneCpuVolume->setLinearDamping(0.0f);
		gSceneCpuVolume->setSettlingThreshold(0.0f);
		gSceneCpuVolume->setSettlingDamping(0.0f);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(smoothReverseFeatureCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setLinearDamping(0.0f);
		gSceneCpuVolume->setSettlingThreshold(0.0f);
		gSceneCpuVolume->setSettlingDamping(0.0f);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(softSleepCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setSleepThreshold(1.0e-4f);
	}
	gMetrics.sceneHostBuffersInitialized = 1;

	gSceneCpuVolumeInitialCentroidY = getSceneCpuVolumeCentroidY();
	gScene->addActor(*gSceneCpuVolume);
	if(gSceneCpuVolume->getScene() != gScene)
		return false;
	gMetrics.sceneActorAdded = 1;
	if(gHeadlessOptions.caseName == "scene-volume-world-pin" ||
		worldElementAttachmentCase)
	{
		const PxU32 attachmentVertex = 0;
		gSceneCpuWorldPinTarget = getSceneCpuAttachmentPoint();
		PxDeformableAttachmentData attachmentData;
		PxVec4 worldCoordinate(gSceneCpuWorldPinTarget, 0.0f);
		attachmentData.actor[0] = gSceneCpuVolume;
		attachmentData.type[0] =
			worldElementAttachmentCase
				? PxDeformableAttachmentTargetType::eTETRAHEDRON
				: PxDeformableAttachmentTargetType::eVERTEX;
		attachmentData.indices[0].data = &attachmentVertex;
		attachmentData.indices[0].count = 1;
		if(worldElementAttachmentCase)
		{
			attachmentData.coords[0].data =
				&gSceneCpuAttachmentBarycentric;
			attachmentData.coords[0].count = 1;
		}
		attachmentData.actor[1] = NULL;
		attachmentData.type[1] =
			PxDeformableAttachmentTargetType::eWORLD;
		attachmentData.coords[1].data = &worldCoordinate;
		attachmentData.coords[1].count = 1;
		gSceneCpuWorldAttachment =
			gPhysics->createDeformableAttachment(attachmentData);
		if(!gSceneCpuWorldAttachment)
			return false;
		gMetrics.sceneWorldPinCreated = 1;
	}
	if(attachmentCase)
	{
		const PxU32 attachmentVertex = 0;
		gSceneCpuRigidAttachmentLocalOffset =
			(kinematicAttachmentCase || staticAttachmentCase)
				? PxVec3(0.0f, 0.5f, 0.0f)
				: PxVec3(0.0f);
		gSceneCpuRigidAttachmentInitialPosition =
			getSceneCpuAttachmentPoint() -
				gSceneCpuRigidAttachmentLocalOffset;
		PxRigidActor* attachmentActor = NULL;
		if(articulationAttachmentCase)
		{
			gSceneCpuAttachmentArticulation =
				gPhysics->createArticulationReducedCoordinate();
			if(!gSceneCpuAttachmentArticulation)
				return false;
			gMetrics.sceneArticulationCreated = 1;
			gSceneCpuAttachmentArticulation->setArticulationFlag(
				PxArticulationFlag::eFIX_BASE, true);
			gSceneCpuAttachmentArticulation->
				setSolverIterationCounts(16, 1);
			gSceneCpuAttachmentArticulation->
				setSleepThreshold(5.0e-4f);
			gSceneCpuArticulationRootInitialPose =
				PxTransform(gSceneCpuRigidAttachmentInitialPosition);
			gSceneCpuArticulationChildInitialPose =
				gSceneCpuArticulationRootInitialPose;
			gSceneCpuAttachmentRoot =
				gSceneCpuAttachmentArticulation->createLink(
					NULL, gSceneCpuArticulationRootInitialPose);
			gSceneCpuAttachmentLink =
				gSceneCpuAttachmentArticulation->createLink(
					gSceneCpuAttachmentRoot,
					gSceneCpuArticulationChildInitialPose);
			if(!gSceneCpuAttachmentRoot ||
				!gSceneCpuAttachmentLink)
				return false;
			PxArticulationLink* links[2] = {
				gSceneCpuAttachmentRoot,
				gSceneCpuAttachmentLink
			};
			for(PxU32 linkIndex = 0; linkIndex < 2; ++linkIndex)
			{
				links[linkIndex]->setMass(1.0f);
				links[linkIndex]->setMassSpaceInertiaTensor(
					PxVec3(1.0f));
				links[linkIndex]->setActorFlag(
					PxActorFlag::eDISABLE_GRAVITY, true);
				links[linkIndex]->setLinearDamping(0.5f);
				links[linkIndex]->setAngularDamping(0.5f);
			}
			PxArticulationJointReducedCoordinate* joint =
				gSceneCpuAttachmentLink->getInboundJoint();
			if(!joint)
				return false;
			joint->setJointType(
				PxArticulationJointType::ePRISMATIC);
			joint->setMotion(
				PxArticulationAxis::eX,
				PxArticulationMotion::eFREE);
			joint->setParentPose(PxTransform(PxIdentity));
			joint->setChildPose(PxTransform(PxIdentity));
			if(!gScene->addArticulation(
				*gSceneCpuAttachmentArticulation))
				return false;
			gSceneCpuAttachmentArticulation->putToSleep();
			gMetrics.sceneArticulationAdded = 1;
			gMetrics.sceneArticulationInitiallySleeping =
				gSceneCpuAttachmentArticulation->isSleeping()
					? 1u : 0u;
			gSceneCpuAttachmentBody = gSceneCpuAttachmentLink;
			attachmentActor = gSceneCpuAttachmentLink;
		}
		else if(staticAttachmentCase)
		{
			gSceneCpuStaticActor =
				gPhysics->createRigidStatic(
					PxTransform(
						gSceneCpuRigidAttachmentInitialPosition));
			if(!gSceneCpuStaticActor ||
				!gScene->addActor(*gSceneCpuStaticActor))
				return false;
			attachmentActor = gSceneCpuStaticActor;
			gMetrics.sceneKinematicActorAdded = 1;
		}
		else
		{
			gSceneCpuDynamicActor =
				gPhysics->createRigidDynamic(
					PxTransform(
						gSceneCpuRigidAttachmentInitialPosition));
			if(!gSceneCpuDynamicActor)
				return false;
			if(kinematicAttachmentCase)
				gSceneCpuDynamicActor->setRigidBodyFlag(
					PxRigidBodyFlag::eKINEMATIC, true);
			else
			{
				gSceneCpuDynamicActor->setMass(1.0f);
				gSceneCpuDynamicActor->setMassSpaceInertiaTensor(
					PxVec3(1.0f));
			}
			gSceneCpuDynamicActor->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuDynamicActor->setLinearDamping(0.5f);
			gSceneCpuDynamicActor->setAngularDamping(0.5f);
			if(!gScene->addActor(*gSceneCpuDynamicActor))
				return false;
			if(!kinematicAttachmentCase)
				gSceneCpuDynamicActor->putToSleep();
			else
				gMetrics.sceneKinematicActorAdded = 1;
			gMetrics.sceneDynamicActorAdded = 1;
			gMetrics.sceneDynamicInitiallySleeping =
				gSceneCpuDynamicActor->isSleeping() ? 1u : 0u;
			gSceneCpuAttachmentBody = gSceneCpuDynamicActor;
			attachmentActor = gSceneCpuDynamicActor;
		}
		gMetrics.sceneRigidAttachmentActorAdded = 1;
		gMetrics.sceneRigidAttachmentInitiallySleeping =
			staticAttachmentCase
				? 1u
				: articulationAttachmentCase
				? gMetrics.sceneArticulationInitiallySleeping
				: (gSceneCpuDynamicActor->isSleeping() ? 1u : 0u);

		PxDeformableAttachmentData attachmentData;
		PxVec4 rigidCoordinate(
			gSceneCpuRigidAttachmentLocalOffset, 0.0f);
		attachmentData.actor[0] = gSceneCpuVolume;
		attachmentData.type[0] =
			gSceneCpuElementAttachment
				? PxDeformableAttachmentTargetType::eTETRAHEDRON
				: PxDeformableAttachmentTargetType::eVERTEX;
		attachmentData.indices[0].data = &attachmentVertex;
		attachmentData.indices[0].count = 1;
		if(gSceneCpuElementAttachment)
		{
			attachmentData.coords[0].data =
				&gSceneCpuAttachmentBarycentric;
			attachmentData.coords[0].count = 1;
		}
		attachmentData.actor[1] = attachmentActor;
		attachmentData.type[1] =
			PxDeformableAttachmentTargetType::eRIGID;
		attachmentData.coords[1].data = &rigidCoordinate;
		attachmentData.coords[1].count = 1;
		gSceneCpuRigidAttachment =
			gPhysics->createDeformableAttachment(attachmentData);
		if(!gSceneCpuRigidAttachment)
			return false;
		gMetrics.sceneRigidAttachmentCreated = 1;
		if(!kinematicAttachmentCase &&
			!staticAttachmentCase)
		{
			for(PxU32 i = 0; i < simulationVertexCount; ++i)
				simVelocity[i] =
					PxVec4(
						0.5f, 0.0f, 0.0f,
						simVelocity[i].w);
			gSceneCpuVolume->markDirty(
				PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		}
	}
	if(gHeadlessOptions.caseName ==
			"scene-volume-element-filter" ||
		partialElementFilterCase)
	{
		if(!gSceneCpuStaticActor)
			return false;
		const PxTetrahedronMesh* collisionMesh =
			gSceneCpuVolume->getCollisionMesh();
		const PxU32 elementCount =
			collisionMesh->getNbTetrahedrons();
		if(partialElementFilterCase && elementCount < 2)
			return false;
		const PxU32 filteredElementCount =
			partialElementFilterCase ? 1u : elementCount;
		PxArray<PxU32> filteredElements;
		filteredElements.resize(filteredElementCount);
		for(PxU32 i = 0; i < filteredElementCount; ++i)
			filteredElements[i] = i;
		if(partialElementFilterCase)
		{
			const bool has16BitIndices =
				collisionMesh->getTetrahedronMeshFlags() &
					PxTetrahedronMeshFlag::e16_BIT_INDICES;
			const PxU16* tets16 = has16BitIndices
				? static_cast<const PxU16*>(
					collisionMesh->getTetrahedrons())
				: NULL;
			const PxU32* tets32 = has16BitIndices
				? NULL
				: static_cast<const PxU32*>(
					collisionMesh->getTetrahedrons());
			const PxVec3* vertices = collisionMesh->getVertices();
			PxVec3 selectedCentroid(0.0f);
			for(PxU32 i = 0; i < 4; ++i)
			{
				const PxU32 vertex =
					has16BitIndices ? tets16[i] : tets32[i];
				if(vertex >= collisionMesh->getNbVertices())
					return false;
				selectedCentroid += vertices[vertex];
			}
			gSceneCpuPartialFilterSelectedPositiveX =
				selectedCentroid.x > 0.0f;
		}
		PxDeformableElementFilterData filterData;
		filterData.actor[0] = gSceneCpuVolume;
		filterData.actor[1] = gSceneCpuStaticActor;
		filterData.groupElementCounts[0].data =
			&filteredElementCount;
		filterData.groupElementCounts[0].count = 1;
		filterData.groupElementIndices[0].data =
			filteredElements.begin();
		filterData.groupElementIndices[0].count =
			filteredElementCount;
		gSceneCpuElementFilter =
			gPhysics->createDeformableElementFilter(filterData);
		if(!gSceneCpuElementFilter)
			return false;
		gMetrics.sceneElementFilterCreated = 1;
	}
	if(softSleepCase)
		gMetrics.sceneSoftInitiallyAwake =
			gSceneCpuVolume->isSleeping() ? 0u : 1u;

	if(twoSoftVolumeCase)
	{
		if(!createAdditionalSceneCpuVolume(
			gSceneCpuSecondVolume,
			smoothReverseFeatureCase
				? PxVec3(1.0f, 0.34f, 0.0f)
				: triangleSurfaceSweptCcdCase
				? PxVec3(
					1.0f,
					staticTriangleSurfaceSweptCcdCase
						? 1.1f : 0.0f,
					0.0f)
				: sphereReverseSweptCcdCase
				? PxVec3(
					1.0f,
					deformingReverseSweptCcdCase
						? 0.45f
					: staticSphereReverseSweptCcdCase
						? (convexReverseSweptCcdCase
							? 1.1f : 0.55f)
						: 0.0f,
					0.0f)
				: rotationalCapsuleSpeculativeCcdCase
				? PxVec3(1.0f, 0.0f, 0.0f)
				: speculativeCcdCase
				? PxVec3(1.0f, 1.2f, 0.0f)
				: maxDepenetrationVelocityCase
				? PxVec3(3.0f, -1.05f, 0.0f)
				: PxVec3(
					softSoftWakeCase ? 3.0f :
						(softPairAttachmentCase ? 0.0f : 10.0f),
					multiSceneIsolationCase ? 6.0f :
						4.0f,
					0.0f),
			multiSceneIsolationCase ? gSceneCpuSecondScene : NULL))
			return false;
		gMetrics.sceneSecondVolumeActorCreated = 1;
		gMetrics.sceneSecondVolumeHostBuffersInitialized = 1;
		gMetrics.sceneSecondVolumeActorAdded = 1;
		gSceneCpuSecondVolumeInitialCentroidY =
			getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
		if(mixedSleepIslandCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setSettlingThreshold(0.0f);
			gSceneCpuSecondVolume->setSettlingDamping(0.0f);
			gSceneCpuSecondVolume->setSleepThreshold(0.0f);
			PxVec4* secondVelocities =
				gSceneCpuSecondVolume->getSimVelocityBufferH();
			const PxU32 secondVertexCount =
				gSceneCpuSecondVolume->getSimulationMesh()->
					getNbVertices();
			if(!secondVelocities)
				return false;
			for(PxU32 i = 0; i < secondVertexCount; ++i)
				secondVelocities[i] = PxVec4(
					PxVec3(0.0f, 0.05f, 0.0f),
					secondVelocities[i].w);
			gSceneCpuSecondVolume->markDirty(
				PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		}
		else if(softChurnCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setSleepThreshold(1.0e-4f);
		}
		else if(multiSceneIsolationCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setSleepThreshold(1.0e-4f);
			PxVec4* secondVelocities =
				gSceneCpuSecondVolume->getSimVelocityBufferH();
			const PxU32 secondVertexCount =
				gSceneCpuSecondVolume->getSimulationMesh()->
					getNbVertices();
			if(!secondVelocities)
				return false;
			for(PxU32 i = 0; i < secondVertexCount; ++i)
				secondVelocities[i] = PxVec4(
					PxVec3(0.0f),
					secondVelocities[i].w);
			gSceneCpuSecondVolume->markDirty(
				PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		}
		else if(softSoftWakeCase || softPairAttachmentCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setSleepThreshold(1.0e-4f);
		}
		else if(motionControlsCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setMaxLinearVelocity(
				PX_MAX_F32);
			gSceneCpuSecondVolume->setSettlingThreshold(0.1f);
			gSceneCpuSecondVolume->setSettlingDamping(10.0f);
			gSceneCpuSecondVolume->setSleepThreshold(0.05f);
			gSceneCpuSecondVolume->setWakeCounter(0.2f);
			if(!setSceneCpuVolumeVelocity(
					*gSceneCpuSecondVolume,
					PxVec3(0.08f, 0.0f, 0.0f)))
				return false;
			gSceneCpuMotionInitialCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			gMetrics.motionControlStayedAwake = 1;
		}
		else if(maxDepenetrationVelocityCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setDeformableBodyFlag(
				PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setSleepThreshold(0.0f);
			gSceneCpuSecondVolume->setWakeCounter(1.0f);
			gSceneCpuSecondVolume->setMaxDepenetrationVelocity(
				PX_MAX_F32);
			gSceneCpuDepenetrationInitialMinY =
				getSceneCpuVolumeMinY(gSceneCpuVolume);
			gSceneCpuDepenetrationControlInitialMinY =
				getSceneCpuVolumeMinY(gSceneCpuSecondVolume);
			gMetrics.depenetrationLimitApplied =
				PxAbs(
					gSceneCpuVolume->
						getMaxDepenetrationVelocity() -
					0.12f) <= 1.0e-6f &&
				gSceneCpuSecondVolume->
					getMaxDepenetrationVelocity() >
						1.0e20f ? 1u : 0u;
		}
		else if(speculativeCcdCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setDeformableBodyFlag(
				PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setSettlingThreshold(0.0f);
			gSceneCpuSecondVolume->setSettlingDamping(0.0f);
			gSceneCpuSecondVolume->setSleepThreshold(0.0f);
			gSceneCpuSecondVolume->setWakeCounter(1.0f);
			if(deformingReverseSweptCcdCase
				? !setSceneCpuVolumeDeformingReverseVelocity(
					*gSceneCpuSecondVolume, -720.0f)
				: !setSceneCpuVolumeVelocity(
					*gSceneCpuSecondVolume,
					PxVec3(
						0.0f,
						triangleSurfaceSweptCcdCase
							? (staticTriangleSurfaceSweptCcdCase
								? (triangleSurfaceReverseSweptCcdCase &&
										heightFieldSweptCcdCase
									? -80.0f : -132.0f)
								: 0.0f) :
						staticSphereReverseSweptCcdCase
							? -132.0f :
						(movingKinematicFiniteSpeculativeCcdCase ||
						 dynamicFiniteRelativeSweptCcdCase ||
						 movingSphereReverseSweptCcdCase)
							? 0.0f :
						capsuleSpeculativeCcdCase
							? -170.0f :
						convexSpeculativeCcdCase
							? -220.0f :
						finiteSmoothSpeculativeCcdCase
							? -160.0f : -120.0f,
						0.0f)))
				return false;
			gMetrics.speculativeCcdFlagApplied =
				gSceneCpuVolume->getDeformableBodyFlags().isSet(
					PxDeformableBodyFlag::
						eENABLE_SPECULATIVE_CCD) &&
				!gSceneCpuSecondVolume->
					getDeformableBodyFlags().isSet(
						PxDeformableBodyFlag::
							eENABLE_SPECULATIVE_CCD)
					? 1u : 0u;
			if(movingKinematicFiniteSpeculativeCcdCase ||
				dynamicFiniteRelativeSweptCcdCase ||
				sphereReverseSweptCcdCase ||
				triangleSurfaceSweptCcdCase)
			{
				gSceneCpuMovingSpherePositiveInitial =
					getSceneCpuVolumeCentroid(gSceneCpuVolume);
				gSceneCpuMovingSphereNegativeInitial =
					getSceneCpuVolumeCentroid(
						gSceneCpuSecondVolume);
			}
		}
		else if(smoothReverseFeatureCase)
		{
			PxDeformableVolume* volumes[2] =
			{
				gSceneCpuVolume,
				gSceneCpuSecondVolume
			};
			for(PxU32 volumeIndex = 0;
				volumeIndex < 2; ++volumeIndex)
			{
				PxDeformableVolume* volume = volumes[volumeIndex];
				volume->setActorFlag(
					PxActorFlag::eDISABLE_GRAVITY, true);
				volume->setDeformableBodyFlag(
					PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,
					true);
				volume->setLinearDamping(0.0f);
				volume->setSettlingThreshold(0.0f);
				volume->setSettlingDamping(0.0f);
				volume->setSleepThreshold(0.0f);
				volume->setWakeCounter(1.0f);
				if(!setSceneCpuVolumeVelocity(
						*volume, PxVec3(0.0f, -2.0f, 0.0f)))
					return false;
			}
			gSceneCpuSphereReversePositiveInitial =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			gSceneCpuSphereReverseNegativeInitial =
				getSceneCpuVolumeCentroid(gSceneCpuSecondVolume);
		}
		if(softPairAttachmentCase)
		{
			const PxU32 collisionElementCount =
				gSceneCpuVolume->getCollisionMesh()->
					getNbTetrahedrons();
			if(collisionElementCount == 0 ||
				gSceneCpuSecondVolume->getCollisionMesh()->
					getNbTetrahedrons() != collisionElementCount)
				return false;
			PxArray<PxU32> filteredElements;
			filteredElements.resize(collisionElementCount);
			for(PxU32 i = 0; i < collisionElementCount; ++i)
				filteredElements[i] = i;
			PxDeformableElementFilterData filterData;
			for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
			{
				filterData.actor[endpoint] = endpoint == 0
					? static_cast<PxActor*>(gSceneCpuVolume)
					: static_cast<PxActor*>(
						gSceneCpuSecondVolume);
				filterData.groupElementCounts[endpoint].data =
					&collisionElementCount;
				filterData.groupElementCounts[endpoint].count = 1;
				filterData.groupElementIndices[endpoint].data =
					filteredElements.begin();
				filterData.groupElementIndices[endpoint].count =
					collisionElementCount;
			}
			gSceneCpuElementFilter =
				gPhysics->createDeformableElementFilter(
					filterData);
			if(!gSceneCpuElementFilter)
				return false;

			const PxU32 elementIndex = 0;
			PxDeformableAttachmentData attachmentData;
			for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
			{
				attachmentData.actor[endpoint] = endpoint == 0
					? static_cast<PxActor*>(gSceneCpuVolume)
					: static_cast<PxActor*>(
						gSceneCpuSecondVolume);
				attachmentData.type[endpoint] =
					PxDeformableAttachmentTargetType::
						eTETRAHEDRON;
				attachmentData.indices[endpoint].data =
					&elementIndex;
				attachmentData.indices[endpoint].count = 1;
				attachmentData.coords[endpoint].data =
					&gSceneCpuAttachmentBarycentric;
				attachmentData.coords[endpoint].count = 1;
			}
			gSceneCpuRigidAttachment =
				gPhysics->createDeformableAttachment(
					attachmentData);
			if(!gSceneCpuRigidAttachment)
				return false;
			gMetrics.sceneRigidAttachmentActorAdded = 1;
			gMetrics.sceneRigidAttachmentCreated = 1;
			gSceneCpuVolume->setLinearDamping(0.0f);
			gSceneCpuVolume->setSettlingThreshold(0.0f);
			gSceneCpuVolume->setSettlingDamping(0.0f);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setSettlingThreshold(0.0f);
			gSceneCpuSecondVolume->setSettlingDamping(0.0f);
			if(!setSceneCpuVolumeVelocity(
					*gSceneCpuVolume,
					PxVec3(0.0f, -0.1f, 0.0f)) ||
				!setSceneCpuVolumeVelocity(
					*gSceneCpuSecondVolume,
					PxVec3(0.0f, 0.1f, 0.0f)))
				return false;
		}
	}
	const PxU32 expectedVolumeCount =
		twoSoftVolumeCase ? 2u : 1u;
	if(multiSceneIsolationCase)
	{
		if(!gSceneCpuSecondScene ||
			gScene->getNbDeformableVolumes() != 1 ||
			gSceneCpuSecondScene->getNbDeformableVolumes() != 1)
			return false;
	}
	else if(gScene->getNbDeformableVolumes() != expectedVolumeCount)
		return false;

	gMetrics.initialized = 1;
	gMetrics.particles =
		simulationVertexCount * expectedVolumeCount;
	gMetrics.softBodies = expectedVolumeCount;
	gMetrics.tetElements =
		simulationMesh->getNbTetrahedrons() * expectedVolumeCount;
	gMetrics.surfaceTriangles =
		(surfaceTriangles.size() / 3) * expectedVolumeCount;
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes =
		multiSceneIsolationCase ? 2u :
			gScene->getNbDeformableVolumes();
	gMetrics.minDetF = 1.0f;
	gMetrics.maxDetF = 1.0f;
	gMetrics.minBodyVolumeRatio = 1.0f;
	gMetrics.maxBodyVolumeRatio = 1.0f;
	return true;
}

static PxReal getSceneCpuVolumeMaxSpeed(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getSimulationMesh())
		return PX_MAX_F32;
	const PxVec4* velocities =
		volume->getSimVelocityBufferH();
	const PxU32 vertexCount =
		volume->getSimulationMesh()->getNbVertices();
	if(!velocities)
		return PX_MAX_F32;
	PxReal maxSpeed = 0.0f;
	for(PxU32 i = 0; i < vertexCount; ++i)
		maxSpeed = PxMax(
			maxSpeed, velocities[i].getXYZ().magnitude());
	return maxSpeed;
}

static PxReal getSceneCpuVolumeMaxSpeed()
{
	return getSceneCpuVolumeMaxSpeed(gSceneCpuVolume);
}

static PxReal getSceneCpuVolumeMinY(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getSimulationMesh())
		return PX_MAX_F32;
	const PxVec4* positions =
		volume->getSimPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getSimulationMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return PX_MAX_F32;
	PxReal minY = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
		minY = PxMin(minY, positions[i].y);
	return minY;
}

static PxReal getSceneCpuVolumeCollisionMinY(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getCollisionMesh())
		return PX_MAX_F32;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return PX_MAX_F32;
	PxReal minY = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
		minY = PxMin(minY, positions[i].y);
	return minY;
}

static PxReal getSceneCpuVolumeCollisionMaxY(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getCollisionMesh())
		return -PX_MAX_F32;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return -PX_MAX_F32;
	PxReal maxY = -PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
		maxY = PxMax(maxY, positions[i].y);
	return maxY;
}

static bool getSceneCpuVolumeSphereMinSeparation(
	PxDeformableVolume* volume, PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh())
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	const PxVec3 sphereCenters[] =
	{
		PxVec3(-2.0f, 0.25f, 0.0f),
		PxVec3(-1.0f, 0.25f, 0.0f),
		PxVec3(-2.0f, 0.25f, 1.0f),
		PxVec3(-1.0f, 0.25f, 1.0f),
		PxVec3(1.0f, 0.25f, 0.0f),
		PxVec3(2.0f, 0.25f, 0.0f),
		PxVec3(1.0f, 0.25f, 1.0f),
		PxVec3(2.0f, 0.25f, 1.0f)
	};
	minimumSeparation = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		for(PxU32 sphereIndex = 0;
			sphereIndex < sizeof(sphereCenters) /
				sizeof(sphereCenters[0]);
			++sphereIndex)
		{
			minimumSeparation = PxMin(
				minimumSeparation,
				(positions[i].getXYZ() -
					sphereCenters[sphereIndex]).magnitude() -
					0.3f);
		}
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeSingleSphereMinSeparation(
	PxDeformableVolume* volume,
	const PxVec3& sphereCenter,
	PxReal sphereRadius,
	PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh() ||
		!sphereCenter.isFinite() ||
		!PxIsFinite(sphereRadius) || sphereRadius <= 0.0f)
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	minimumSeparation = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		const PxVec3 position = positions[i].getXYZ();
		if(!position.isFinite())
			return false;
		minimumSeparation = PxMin(
			minimumSeparation,
			(position - sphereCenter).magnitude() -
				sphereRadius);
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static PxReal getCapsuleSignedSeparation(
	const PxVec3& point,
	const PxTransform& capsulePose,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight)
{
	const PxVec3 localPoint = capsulePose.transformInv(point);
	const PxReal axisCoordinate = PxClamp(
		localPoint.x, -capsuleHalfHeight, capsuleHalfHeight);
	return (
		localPoint - PxVec3(axisCoordinate, 0.0f, 0.0f)).
			magnitude() - capsuleRadius;
}

static PxReal getCapsuleSignedSeparation(
	const PxVec3& point,
	const PxVec3& capsuleCenter,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight)
{
	return getCapsuleSignedSeparation(
		point, PxTransform(capsuleCenter),
		capsuleRadius, capsuleHalfHeight);
}

static bool getSceneCpuVolumeCapsuleClusterMinSeparation(
	PxDeformableVolume* volume, PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh())
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	const PxVec3 capsuleCenters[] =
	{
		PxVec3(-2.0f, 0.25f, 0.0f),
		PxVec3(-1.0f, 0.25f, 0.0f),
		PxVec3(-2.0f, 0.25f, 1.0f),
		PxVec3(-1.0f, 0.25f, 1.0f),
		PxVec3(1.0f, 0.25f, 0.0f),
		PxVec3(2.0f, 0.25f, 0.0f),
		PxVec3(1.0f, 0.25f, 1.0f),
		PxVec3(2.0f, 0.25f, 1.0f)
	};
	minimumSeparation = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		const PxVec3 position = positions[i].getXYZ();
		if(!position.isFinite())
			return false;
		for(PxU32 capsuleIndex = 0;
			capsuleIndex < sizeof(capsuleCenters) /
				sizeof(capsuleCenters[0]);
			++capsuleIndex)
		{
			minimumSeparation = PxMin(
				minimumSeparation,
				getCapsuleSignedSeparation(
					position, capsuleCenters[capsuleIndex],
					0.3f, 0.2f));
		}
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeSingleCapsuleMinSeparation(
	PxDeformableVolume* volume,
	const PxTransform& capsulePose,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight,
	PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh() ||
		!capsulePose.isValid() ||
		!PxIsFinite(capsuleRadius) || capsuleRadius <= 0.0f ||
		!PxIsFinite(capsuleHalfHeight) || capsuleHalfHeight < 0.0f)
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	minimumSeparation = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		const PxVec3 position = positions[i].getXYZ();
		if(!position.isFinite())
			return false;
		minimumSeparation = PxMin(
			minimumSeparation,
			getCapsuleSignedSeparation(
				position, capsulePose,
				capsuleRadius, capsuleHalfHeight));
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeConvexMinSeparation(
	PxDeformableVolume* volume,
	const PxVec3* convexCenters,
	PxU32 convexCount,
	PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh() ||
		!convexCenters || convexCount == 0 ||
		!gSceneCpuRigidConvexMesh)
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	const PxConvexMeshGeometry geometry(gSceneCpuRigidConvexMesh);
	minimumSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex < vertexCount; ++vertexIndex)
	{
		const PxVec3 position =
			positions[vertexIndex].getXYZ();
		if(!position.isFinite())
			return false;
		for(PxU32 convexIndex = 0;
			convexIndex < convexCount; ++convexIndex)
		{
			if(!convexCenters[convexIndex].isFinite())
				return false;
			const PxReal squaredDistance =
				PxGeometryQuery::pointDistance(
					position, geometry,
					PxTransform(convexCenters[convexIndex]));
			if(!PxIsFinite(squaredDistance) ||
				squaredDistance < 0.0f)
				return false;
			minimumSeparation = PxMin(
				minimumSeparation,
				PxSqrt(squaredDistance));
		}
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeSingleConvexMinSeparation(
	PxDeformableVolume* volume,
	const PxTransform& convexPose,
	PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh() ||
		!convexPose.isValid() || !gSceneCpuRigidConvexMesh)
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	const PxConvexMeshGeometry geometry(gSceneCpuRigidConvexMesh);
	minimumSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex < vertexCount; ++vertexIndex)
	{
		const PxVec3 point = positions[vertexIndex].getXYZ();
		if(!point.isFinite())
			return false;
		const PxReal squaredDistance =
			PxGeometryQuery::pointDistance(
				point, geometry, convexPose);
		if(!PxIsFinite(squaredDistance) ||
			squaredDistance < 0.0f)
			return false;
		minimumSeparation = PxMin(
			minimumSeparation, PxSqrt(squaredDistance));
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static PxVec3 closestPointOnTriangleForGate(
	const PxVec3& point,
	const PxVec3& a,
	const PxVec3& b,
	const PxVec3& c)
{
	const PxVec3 ab = b - a;
	const PxVec3 ac = c - a;
	const PxVec3 ap = point - a;
	const PxReal d1 = ab.dot(ap);
	const PxReal d2 = ac.dot(ap);
	if(d1 <= 0.0f && d2 <= 0.0f)
		return a;

	const PxVec3 bp = point - b;
	const PxReal d3 = ab.dot(bp);
	const PxReal d4 = ac.dot(bp);
	if(d3 >= 0.0f && d4 <= d3)
		return b;

	const PxReal vc = d1 * d4 - d3 * d2;
	if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		const PxReal denominator = d1 - d3;
		return denominator > 1.0e-20f
			? a + ab * (d1 / denominator) : a;
	}

	const PxVec3 cp = point - c;
	const PxReal d5 = ab.dot(cp);
	const PxReal d6 = ac.dot(cp);
	if(d6 >= 0.0f && d5 <= d6)
		return c;

	const PxReal vb = d5 * d2 - d1 * d6;
	if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		const PxReal denominator = d2 - d6;
		return denominator > 1.0e-20f
			? a + ac * (d2 / denominator) : a;
	}

	const PxReal va = d3 * d6 - d5 * d4;
	if(va <= 0.0f && d4 - d3 >= 0.0f &&
		d5 - d6 >= 0.0f)
	{
		const PxReal edgeTerm = d4 - d3;
		const PxReal denominator =
			edgeTerm + d5 - d6;
		return denominator > 1.0e-20f
			? b + (c - b) * (edgeTerm / denominator) : b;
	}

	const PxReal denominator = va + vb + vc;
	if(PxAbs(denominator) <= 1.0e-20f)
		return a;
	const PxReal inverseDenominator = 1.0f / denominator;
	return a + ab * (vb * inverseDenominator) +
		ac * (vc * inverseDenominator);
}

static bool getSceneCpuVolumeSmoothReverseSeparations(
	PxDeformableVolume* volume,
	const PxVec3& rigidCenter,
	PxReal rigidRadius,
	PxReal capsuleHalfHeight,
	bool convexCase,
	bool triangleMeshCase,
	bool heightFieldCase,
	PxReal& faceSeparation,
	PxReal& minimumVertexSeparation);

static bool getSceneCpuVolumeSphereReverseSweptSeparations(
	PxDeformableVolume* volume,
	const PxVec3& sphereCenterCurrent,
	const PxVec3& sphereCenterStart,
	const PxVec3& sphereCenterEnd,
	PxReal sphereRadius,
	PxReal capsuleHalfHeight,
	bool convexCase,
	PxReal& faceSeparation,
	PxReal& minimumVertexSweepSeparation)
{
	PxReal currentVertexSeparation = PX_MAX_F32;
	if(!getSceneCpuVolumeSmoothReverseSeparations(
			volume, sphereCenterCurrent, sphereRadius,
			capsuleHalfHeight,
			convexCase, false, false,
			faceSeparation, currentVertexSeparation) ||
		!sphereCenterCurrent.isFinite() ||
		!sphereCenterStart.isFinite() ||
		!sphereCenterEnd.isFinite())
		return false;

	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	if(vertexCount == 0 ||
		gSceneCpuSphereReverseSweptInitialPositions.size() !=
			vertexCount)
		return false;
	minimumVertexSweepSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex < vertexCount; ++vertexIndex)
	{
		const PxVec3 initialPosition =
			gSceneCpuSphereReverseSweptInitialPositions[vertexIndex];
		if(!initialPosition.isFinite())
			return false;
		if(convexCase)
		{
			if(!gSceneCpuRigidConvexMesh)
				return false;
			const PxConvexMeshGeometry geometry(
				gSceneCpuRigidConvexMesh);
			for(PxU32 sampleIndex = 0;
				sampleIndex <= 64; ++sampleIndex)
			{
				const PxReal alpha =
					PxReal(sampleIndex) / 64.0f;
				const PxVec3 center =
					sphereCenterStart +
					(sphereCenterEnd - sphereCenterStart) *
						alpha;
				const PxReal squaredDistance =
					PxGeometryQuery::pointDistance(
						initialPosition, geometry,
						PxTransform(center));
				if(!PxIsFinite(squaredDistance) ||
					squaredDistance < 0.0f)
					return false;
				minimumVertexSweepSeparation = PxMin(
					minimumVertexSweepSeparation,
					PxSqrt(squaredDistance));
			}
		}
		else
		{
			const PxVec3 centerPath =
				sphereCenterEnd - sphereCenterStart;
			const PxReal centerPathLengthSq =
				centerPath.magnitudeSquared();
			const PxReal centerPathWeight =
				centerPathLengthSq > 1.0e-20f
					? PxClamp(
						(initialPosition - sphereCenterStart).
							dot(centerPath) /
							centerPathLengthSq,
						0.0f, 1.0f)
					: 0.0f;
			const PxVec3 closestCenter =
				sphereCenterStart +
				centerPath * centerPathWeight;
			const PxVec3 closestMedialPoint =
				closestCenter + PxVec3(
					PxClamp(
						initialPosition.x - closestCenter.x,
						-capsuleHalfHeight,
						capsuleHalfHeight),
					0.0f, 0.0f);
			minimumVertexSweepSeparation = PxMin(
				minimumVertexSweepSeparation,
				(initialPosition - closestMedialPoint).
					magnitude() - sphereRadius);
		}
	}
	return PxIsFinite(minimumVertexSweepSeparation) &&
		minimumVertexSweepSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeDeformingReverseSweptProof(
	PxDeformableVolume* volume,
	const PxTransform& rigidPose,
	PxReal radius,
	PxReal capsuleHalfHeight,
	bool convexCase,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation,
	PxReal& minimumVertexSweepSeparation)
{
	if(!volume || !rigidPose.isValid() ||
		!PxIsFinite(radius) || radius <= 0.0f ||
		!PxIsFinite(capsuleHalfHeight) ||
		capsuleHalfHeight < 0.0f ||
		(convexCase && !gSceneCpuRigidConvexMesh))
		return false;
	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	const PxU32 tetrahedronCount =
		mesh ? mesh->getNbTetrahedrons() : 0;
	if(!mesh || vertexCount == 0 || tetrahedronCount == 0 ||
		gSceneCpuSphereReverseSweptInitialPositions.size() !=
			vertexCount ||
		gSceneCpuDeformingReverseSweptFreeEndPositions.size() !=
			vertexCount)
		return false;
	const bool has16BitIndices =
		mesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* indices16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTetrahedrons())
		: NULL;
	const PxU32* indices32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTetrahedrons());
	if((has16BitIndices && !indices16) ||
		(!has16BitIndices && !indices32))
		return false;
	const PxU32 faceEndpoints[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};
	struct IndexedBoundaryFace
	{
		PxU32 vertices[3];
		PxU32 sortedVertices[3];
		PxU32 ownerCount;
	};
	PxArray<IndexedBoundaryFace> indexedFaces;
	for(PxU32 tetrahedronIndex = 0;
		tetrahedronIndex < tetrahedronCount;
		++tetrahedronIndex)
	{
		PxU32 tetrahedron[4];
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
		{
			const PxU32 flatIndex =
				tetrahedronIndex * 4 + endpoint;
			tetrahedron[endpoint] = has16BitIndices
				? PxU32(indices16[flatIndex])
				: indices32[flatIndex];
			if(tetrahedron[endpoint] >= vertexCount)
				return false;
		}
		for(PxU32 faceIndex = 0;
			faceIndex < 4; ++faceIndex)
		{
			IndexedBoundaryFace face;
			for(PxU32 endpoint = 0; endpoint < 3; ++endpoint)
			{
				face.vertices[endpoint] =
					tetrahedron[
						faceEndpoints[faceIndex][endpoint]];
				face.sortedVertices[endpoint] =
					face.vertices[endpoint];
			}
			for(PxU32 i = 0; i < 2; ++i)
			{
				for(PxU32 j = i + 1; j < 3; ++j)
				{
					if(face.sortedVertices[j] <
						face.sortedVertices[i])
					{
						const PxU32 temporary =
							face.sortedVertices[i];
						face.sortedVertices[i] =
							face.sortedVertices[j];
						face.sortedVertices[j] =
							temporary;
					}
				}
			}
			face.ownerCount = 1;
			bool matched = false;
			for(PxU32 existingIndex = 0;
				existingIndex < indexedFaces.size();
				++existingIndex)
			{
				IndexedBoundaryFace& existing =
					indexedFaces[existingIndex];
				if(existing.sortedVertices[0] ==
						face.sortedVertices[0] &&
					existing.sortedVertices[1] ==
						face.sortedVertices[1] &&
					existing.sortedVertices[2] ==
						face.sortedVertices[2])
				{
					existing.ownerCount++;
					matched = true;
					break;
				}
			}
			if(!matched)
				indexedFaces.pushBack(face);
		}
	}
	const PxVec3* convexVertices = convexCase
		? gSceneCpuRigidConvexMesh->getVertices()
		: NULL;
	const PxU32 convexVertexCount = convexCase
		? gSceneCpuRigidConvexMesh->getNbVertices()
		: 0;
	const PxConvexMeshGeometry convexGeometry(
		convexCase ? gSceneCpuRigidConvexMesh : NULL);
	const PxVec3 capsuleAxis = rigidPose.q.getBasisVector0();

	auto getFaceSeparation = [&](
		const PxVec3& a,
		const PxVec3& b,
		const PxVec3& c)
	{
		PxReal separation = PX_MAX_F32;
		if(convexCase)
		{
			for(PxU32 rigidVertex = 0;
				rigidVertex < convexVertexCount; ++rigidVertex)
			{
				const PxVec3 point =
					rigidPose.transform(
						convexVertices[rigidVertex]);
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						point, a, b, c);
				separation = PxMin(
					separation,
					(point - closest).magnitude());
			}
		}
		else
		{
			const PxU32 sampleCount =
				capsuleHalfHeight > 0.0f ? 128u : 0u;
			for(PxU32 sample = 0;
				sample <= sampleCount; ++sample)
			{
				const PxReal axisCoordinate =
					sampleCount > 0
						? -capsuleHalfHeight +
							2.0f * capsuleHalfHeight *
								(PxReal(sample) /
								 PxReal(sampleCount))
						: 0.0f;
				const PxVec3 medialPoint =
					rigidPose.p +
						capsuleAxis * axisCoordinate;
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						medialPoint, a, b, c);
				separation = PxMin(
					separation,
					(medialPoint - closest).magnitude() -
						radius);
			}
		}
		return separation;
	};

	endpointMinSeparation = PX_MAX_F32;
	midSweepMinSeparation = PX_MAX_F32;
	minimumVertexSweepSeparation = PX_MAX_F32;
	PxArray<PxVec3> samplePositions;
	samplePositions.resize(vertexCount);
	for(PxU32 sample = 0; sample <= 128; ++sample)
	{
		const PxReal alpha =
			PxReal(sample) / 128.0f;
		for(PxU32 vertexIndex = 0;
			vertexIndex < vertexCount; ++vertexIndex)
		{
			const PxVec3 start =
				gSceneCpuSphereReverseSweptInitialPositions[
					vertexIndex];
			const PxVec3 end =
				gSceneCpuDeformingReverseSweptFreeEndPositions[
					vertexIndex];
			if(!start.isFinite() || !end.isFinite())
				return false;
			const PxVec3 point =
				start + (end - start) * alpha;
			samplePositions[vertexIndex] = point;
			PxReal vertexSeparation = PX_MAX_F32;
			if(convexCase)
			{
				const PxReal squaredDistance =
					PxGeometryQuery::pointDistance(
						point, convexGeometry, rigidPose);
				if(!PxIsFinite(squaredDistance) ||
					squaredDistance < 0.0f)
					return false;
				vertexSeparation = PxSqrt(squaredDistance);
			}
			else
				vertexSeparation = getCapsuleSignedSeparation(
					point, rigidPose, radius,
					capsuleHalfHeight);
			minimumVertexSweepSeparation = PxMin(
				minimumVertexSweepSeparation,
				vertexSeparation);
		}

		PxReal sampleFaceSeparation = PX_MAX_F32;
		for(PxU32 faceIndex = 0;
			faceIndex < indexedFaces.size(); ++faceIndex)
		{
			const IndexedBoundaryFace& face =
				indexedFaces[faceIndex];
			if(face.ownerCount != 1)
				continue;
			const PxVec3& a =
				samplePositions[face.vertices[0]];
			const PxVec3& b =
				samplePositions[face.vertices[1]];
			const PxVec3& c =
				samplePositions[face.vertices[2]];
			const PxReal candidateFaceSeparation =
				getFaceSeparation(a, b, c);
			sampleFaceSeparation = PxMin(
				sampleFaceSeparation,
				candidateFaceSeparation);
		}
		if(sample == 0 || sample == 128)
		{
			endpointMinSeparation = PxMin(
				endpointMinSeparation,
				sampleFaceSeparation);
		}
		else
			midSweepMinSeparation = PxMin(
				midSweepMinSeparation,
				sampleFaceSeparation);
	}
	return PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation) &&
		PxIsFinite(minimumVertexSweepSeparation);
}

static bool getSceneCpuVolumeCapsuleFaceSeparation(
	PxDeformableVolume* volume,
	const PxTransform& capsulePose,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight,
	bool useInitialPositions,
	PxReal& faceSeparation)
{
	if(!volume || !capsulePose.isValid() ||
		!PxIsFinite(capsuleRadius) || capsuleRadius <= 0.0f ||
		!PxIsFinite(capsuleHalfHeight) || capsuleHalfHeight < 0.0f)
		return false;
	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxVec4* positions = volume->getPositionInvMassBufferH();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	const PxU32 tetrahedronCount =
		mesh ? mesh->getNbTetrahedrons() : 0;
	if(!mesh || !positions || vertexCount == 0 ||
		tetrahedronCount == 0 ||
		(useInitialPositions &&
		 gSceneCpuSphereReverseSweptInitialPositions.size() !=
			vertexCount))
		return false;

	const bool has16BitIndices =
		mesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* indices16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTetrahedrons())
		: NULL;
	const PxU32* indices32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTetrahedrons());
	if((has16BitIndices && !indices16) ||
		(!has16BitIndices && !indices32))
		return false;
	const PxU32 faceEndpoints[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};
	const PxVec3 capsuleAxis = capsulePose.q.getBasisVector0();
	faceSeparation = PX_MAX_F32;
	for(PxU32 tetrahedronIndex = 0;
		tetrahedronIndex < tetrahedronCount;
		++tetrahedronIndex)
	{
		PxU32 tetrahedron[4];
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
		{
			const PxU32 flatIndex =
				tetrahedronIndex * 4 + endpoint;
			tetrahedron[endpoint] = has16BitIndices
				? PxU32(indices16[flatIndex])
				: indices32[flatIndex];
			if(tetrahedron[endpoint] >= vertexCount)
				return false;
		}
		for(PxU32 faceIndex = 0; faceIndex < 4; ++faceIndex)
		{
			PxVec3 face[3];
			for(PxU32 endpoint = 0; endpoint < 3; ++endpoint)
			{
				const PxU32 vertexIndex =
					tetrahedron[faceEndpoints[faceIndex][endpoint]];
				face[endpoint] = useInitialPositions
					? gSceneCpuSphereReverseSweptInitialPositions[
						vertexIndex]
					: positions[vertexIndex].getXYZ();
				if(!face[endpoint].isFinite())
					return false;
			}
			for(PxU32 sample = 0; sample <= 128; ++sample)
			{
				const PxReal axisCoordinate =
					-capsuleHalfHeight +
					2.0f * capsuleHalfHeight *
						(PxReal(sample) / 128.0f);
				const PxVec3 medialPoint =
					capsulePose.p + capsuleAxis * axisCoordinate;
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						medialPoint, face[0], face[1], face[2]);
				if(!closest.isFinite())
					return false;
				faceSeparation = PxMin(
					faceSeparation,
					(medialPoint - closest).magnitude() -
						capsuleRadius);
			}
		}
	}
	return PxIsFinite(faceSeparation) &&
		faceSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeConvexFaceSeparation(
	PxDeformableVolume* volume,
	const PxTransform& convexPose,
	bool useInitialPositions,
	PxReal& faceSeparation)
{
	if(!volume || !convexPose.isValid() ||
		!gSceneCpuRigidConvexMesh)
		return false;
	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxVec4* positions = volume->getPositionInvMassBufferH();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	const PxU32 tetrahedronCount =
		mesh ? mesh->getNbTetrahedrons() : 0;
	const PxVec3* convexVertices =
		gSceneCpuRigidConvexMesh->getVertices();
	const PxU32 convexVertexCount =
		gSceneCpuRigidConvexMesh->getNbVertices();
	if(!mesh || !positions || vertexCount == 0 ||
		tetrahedronCount == 0 || !convexVertices ||
		convexVertexCount == 0 ||
		(useInitialPositions &&
		 gSceneCpuSphereReverseSweptInitialPositions.size() !=
			vertexCount))
		return false;

	const bool has16BitIndices =
		mesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* indices16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTetrahedrons())
		: NULL;
	const PxU32* indices32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTetrahedrons());
	if((has16BitIndices && !indices16) ||
		(!has16BitIndices && !indices32))
		return false;
	const PxU32 faceEndpoints[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};

	faceSeparation = PX_MAX_F32;
	for(PxU32 convexVertexIndex = 0;
		convexVertexIndex < convexVertexCount;
		++convexVertexIndex)
	{
		const PxVec3 point =
			convexPose.transform(convexVertices[convexVertexIndex]);
		if(!point.isFinite())
			return false;
		PxReal pointDistance = PX_MAX_F32;
		bool pointInside = false;
		for(PxU32 tetrahedronIndex = 0;
			tetrahedronIndex < tetrahedronCount;
			++tetrahedronIndex)
		{
			PxVec3 tetrahedron[4];
			for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
			{
				const PxU32 flatIndex =
					tetrahedronIndex * 4 + endpoint;
				const PxU32 vertexIndex = has16BitIndices
					? PxU32(indices16[flatIndex])
					: indices32[flatIndex];
				if(vertexIndex >= vertexCount)
					return false;
				tetrahedron[endpoint] = useInitialPositions
					? gSceneCpuSphereReverseSweptInitialPositions[
						vertexIndex]
					: positions[vertexIndex].getXYZ();
				if(!tetrahedron[endpoint].isFinite())
					return false;
			}

			const PxVec3 edge0 =
				tetrahedron[1] - tetrahedron[0];
			const PxVec3 edge1 =
				tetrahedron[2] - tetrahedron[0];
			const PxVec3 edge2 =
				tetrahedron[3] - tetrahedron[0];
			const PxVec3 relative =
				point - tetrahedron[0];
			const PxReal determinant =
				edge0.dot(edge1.cross(edge2));
			if(PxAbs(determinant) > 1.0e-20f)
			{
				const PxReal inverseDeterminant =
					1.0f / determinant;
				const PxReal b1 =
					relative.dot(edge1.cross(edge2)) *
						inverseDeterminant;
				const PxReal b2 =
					edge0.dot(relative.cross(edge2)) *
						inverseDeterminant;
				const PxReal b3 =
					edge0.dot(edge1.cross(relative)) *
						inverseDeterminant;
				const PxReal b0 = 1.0f - b1 - b2 - b3;
				const PxReal tolerance = 1.0e-5f;
				pointInside = pointInside ||
					(b0 >= -tolerance &&
					 b1 >= -tolerance &&
					 b2 >= -tolerance &&
					 b3 >= -tolerance);
			}

			for(PxU32 faceIndex = 0;
				faceIndex < 4; ++faceIndex)
			{
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						point,
						tetrahedron[
							faceEndpoints[faceIndex][0]],
						tetrahedron[
							faceEndpoints[faceIndex][1]],
						tetrahedron[
							faceEndpoints[faceIndex][2]]);
				if(!closest.isFinite())
					return false;
				pointDistance = PxMin(
					pointDistance,
					(point - closest).magnitude());
			}
		}
		if(!PxIsFinite(pointDistance) ||
			pointDistance >= PX_MAX_F32)
			return false;
		faceSeparation = PxMin(
			faceSeparation,
			pointInside ? -pointDistance : pointDistance);
	}
	return PxIsFinite(faceSeparation) &&
		faceSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeRotationalConvexPointSweepSeparations(
	const PxTransform& startPose,
	const PxTransform& endPose,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation)
{
	if(!startPose.isValid() || !endPose.isValid() ||
		!gSceneCpuRigidConvexMesh ||
		gSceneCpuCapsuleRotationalSweptInitialPositions.empty())
		return false;
	const PxConvexMeshGeometry geometry(
		gSceneCpuRigidConvexMesh);
	endpointMinSeparation = PX_MAX_F32;
	midSweepMinSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex <
			gSceneCpuCapsuleRotationalSweptInitialPositions.size();
		++vertexIndex)
	{
		const PxVec3 point =
			gSceneCpuCapsuleRotationalSweptInitialPositions[
				vertexIndex];
		if(!point.isFinite())
			return false;
		const PxReal startDistanceSq =
			PxGeometryQuery::pointDistance(
				point, geometry, startPose);
		const PxReal endDistanceSq =
			PxGeometryQuery::pointDistance(
				point, geometry, endPose);
		if(!PxIsFinite(startDistanceSq) ||
			!PxIsFinite(endDistanceSq) ||
			startDistanceSq < 0.0f ||
			endDistanceSq < 0.0f)
			return false;
		endpointMinSeparation = PxMin(
			endpointMinSeparation,
			PxSqrt(PxMin(startDistanceSq, endDistanceSq)));
		for(PxU32 sample = 1; sample < 64; ++sample)
		{
			const PxReal time =
				PxReal(sample) / 64.0f;
			const PxTransform samplePose(
				startPose.p +
					(endPose.p - startPose.p) * time,
				PxSlerp(
					time, startPose.q, endPose.q).
						getNormalized());
			const PxReal sampleDistanceSq =
				PxGeometryQuery::pointDistance(
					point, geometry, samplePose);
			if(!PxIsFinite(sampleDistanceSq) ||
				sampleDistanceSq < 0.0f)
				return false;
			midSweepMinSeparation = PxMin(
				midSweepMinSeparation,
				PxSqrt(sampleDistanceSq));
		}
	}
	return PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation);
}

static bool getSceneCpuVolumeRotationalCapsuleReverseSweptSeparations(
	PxDeformableVolume* volume,
	const PxTransform& currentPose,
	const PxTransform& startPose,
	const PxTransform& endPose,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight,
	PxReal& faceSeparation,
	PxReal& minimumVertexSweepSeparation,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation)
{
	if(!volume || !currentPose.isValid() ||
		!startPose.isValid() || !endPose.isValid() ||
		!getSceneCpuVolumeCapsuleFaceSeparation(
			volume, currentPose, capsuleRadius,
			capsuleHalfHeight, false, faceSeparation))
		return false;
	PxReal startFaceSeparation = PX_MAX_F32;
	PxReal endFaceSeparation = PX_MAX_F32;
	if(!getSceneCpuVolumeCapsuleFaceSeparation(
			volume, startPose, capsuleRadius,
			capsuleHalfHeight, true, startFaceSeparation) ||
		!getSceneCpuVolumeCapsuleFaceSeparation(
			volume, endPose, capsuleRadius,
			capsuleHalfHeight, true, endFaceSeparation))
		return false;

	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	if(vertexCount == 0 ||
		gSceneCpuSphereReverseSweptInitialPositions.size() !=
			vertexCount)
		return false;
	endpointMinSeparation =
		PxMin(startFaceSeparation, endFaceSeparation);
	midSweepMinSeparation = PX_MAX_F32;
	minimumVertexSweepSeparation = PX_MAX_F32;
	for(PxU32 sample = 0; sample <= 64; ++sample)
	{
		const PxReal time = PxReal(sample) / 64.0f;
		const PxTransform samplePose(
			startPose.p + (endPose.p - startPose.p) * time,
			PxSlerp(
				time, startPose.q, endPose.q).
					getNormalized());
		if(sample > 0 && sample < 64)
		{
			PxReal sampleFaceSeparation = PX_MAX_F32;
			if(!getSceneCpuVolumeCapsuleFaceSeparation(
					volume, samplePose, capsuleRadius,
					capsuleHalfHeight, true,
					sampleFaceSeparation))
				return false;
			midSweepMinSeparation = PxMin(
				midSweepMinSeparation, sampleFaceSeparation);
		}
		for(PxU32 vertexIndex = 0;
			vertexIndex < vertexCount;
			++vertexIndex)
		{
			const PxVec3 point =
				gSceneCpuSphereReverseSweptInitialPositions[
					vertexIndex];
			if(!point.isFinite())
				return false;
			minimumVertexSweepSeparation = PxMin(
				minimumVertexSweepSeparation,
				getCapsuleSignedSeparation(
					point, samplePose,
					capsuleRadius, capsuleHalfHeight));
		}
	}
	return PxIsFinite(faceSeparation) &&
		PxIsFinite(minimumVertexSweepSeparation) &&
		PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation);
}

static bool getSceneCpuVolumeRotationalConvexReverseSweptSeparations(
	PxDeformableVolume* volume,
	const PxTransform& currentPose,
	const PxTransform& startPose,
	const PxTransform& endPose,
	PxReal& faceSeparation,
	PxReal& minimumVertexSweepSeparation,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation)
{
	if(!volume || !currentPose.isValid() ||
		!startPose.isValid() || !endPose.isValid() ||
		!gSceneCpuRigidConvexMesh ||
		!getSceneCpuVolumeConvexFaceSeparation(
			volume, currentPose, false, faceSeparation))
		return false;
	PxReal startFaceSeparation = PX_MAX_F32;
	PxReal endFaceSeparation = PX_MAX_F32;
	if(!getSceneCpuVolumeConvexFaceSeparation(
			volume, startPose, true, startFaceSeparation) ||
		!getSceneCpuVolumeConvexFaceSeparation(
			volume, endPose, true, endFaceSeparation))
		return false;

	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	if(vertexCount == 0 ||
		gSceneCpuSphereReverseSweptInitialPositions.size() !=
			vertexCount)
		return false;
	const PxConvexMeshGeometry geometry(
		gSceneCpuRigidConvexMesh);
	endpointMinSeparation =
		PxMin(startFaceSeparation, endFaceSeparation);
	midSweepMinSeparation = PX_MAX_F32;
	minimumVertexSweepSeparation = PX_MAX_F32;
	for(PxU32 sample = 0; sample <= 64; ++sample)
	{
		const PxReal time = PxReal(sample) / 64.0f;
		const PxTransform samplePose(
			startPose.p + (endPose.p - startPose.p) * time,
			PxSlerp(
				time, startPose.q, endPose.q).
					getNormalized());
		if(sample > 0 && sample < 64)
		{
			PxReal sampleFaceSeparation = PX_MAX_F32;
			if(!getSceneCpuVolumeConvexFaceSeparation(
					volume, samplePose, true,
					sampleFaceSeparation))
				return false;
			midSweepMinSeparation = PxMin(
				midSweepMinSeparation, sampleFaceSeparation);
		}
		for(PxU32 vertexIndex = 0;
			vertexIndex < vertexCount;
			++vertexIndex)
		{
			const PxVec3 point =
				gSceneCpuSphereReverseSweptInitialPositions[
					vertexIndex];
			if(!point.isFinite())
				return false;
			const PxReal squaredDistance =
				PxGeometryQuery::pointDistance(
					point, geometry, samplePose);
			if(!PxIsFinite(squaredDistance) ||
				squaredDistance < 0.0f)
				return false;
			minimumVertexSweepSeparation = PxMin(
				minimumVertexSweepSeparation,
				PxSqrt(squaredDistance));
		}
	}
	return PxIsFinite(faceSeparation) &&
		PxIsFinite(minimumVertexSweepSeparation) &&
		PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation);
}

static bool getSceneCpuVolumeRotationalTriangleSurfaceSweepSeparations(
	const PxTransform& startPose,
	const PxTransform& endPose,
	bool heightField, bool reverseCase,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation,
	PxReal& minimumVertexSweepSeparation)
{
	if(!startPose.isValid() || !endPose.isValid() ||
		gSceneCpuSphereReverseSweptInitialPositions.size() < 4)
		return false;
	const PxVec3 bladeLocal[4] =
	{
		heightField
			? PxVec3(0.0f, 0.0f, 0.0f)
			: PxVec3(-1.0f, 0.0f, -0.1f),
		heightField
			? PxVec3(0.0f, 0.0f, 0.2f)
			: PxVec3(-1.0f, 0.0f, 0.1f),
		heightField
			? PxVec3(2.0f, 0.0f, 0.2f)
			: PxVec3(1.0f, 0.0f, 0.1f),
		heightField
			? PxVec3(2.0f, 0.0f, 0.0f)
			: PxVec3(1.0f, 0.0f, -0.1f)
	};
	auto getBladeWorld = [&](
		PxReal time, PxVec3 world[4])
	{
		const PxVec3 center =
			startPose.p + (endPose.p - startPose.p) * time;
		const PxQuat rotation = PxSlerp(
			time, startPose.q, endPose.q).getNormalized();
		for(PxU32 vertex = 0; vertex < 4; ++vertex)
			world[vertex] =
				center + rotation.rotate(bladeLocal[vertex]);
	};
	auto getPointBladeDistance = [&](
		const PxVec3& point, PxReal time)
	{
		PxVec3 world[4];
		getBladeWorld(time, world);
		const PxVec3 closest0 =
			closestPointOnTriangleForGate(
				point, world[0], world[1], world[2]);
		const PxVec3 closest1 =
			closestPointOnTriangleForGate(
				point, world[0], world[2], world[3]);
		return PxMin(
			(point - closest0).magnitude(),
			(point - closest1).magnitude());
	};
	const PxU32 boundaryFaces[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};
	endpointMinSeparation = PX_MAX_F32;
	midSweepMinSeparation = PX_MAX_F32;
	minimumVertexSweepSeparation = PX_MAX_F32;
	for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
	{
		const PxReal time = PxReal(endpoint);
		for(PxU32 softVertex = 0;
			softVertex <
				gSceneCpuSphereReverseSweptInitialPositions.size();
			++softVertex)
		{
			const PxVec3 point =
				gSceneCpuSphereReverseSweptInitialPositions[
					softVertex];
			if(!point.isFinite())
				return false;
			endpointMinSeparation = PxMin(
				endpointMinSeparation,
				getPointBladeDistance(point, time));
		}
		if(reverseCase)
		{
			PxVec3 world[4];
			getBladeWorld(time, world);
			for(PxU32 rigidVertex = 0;
				rigidVertex < 4; ++rigidVertex)
			{
				for(PxU32 face = 0; face < 4; ++face)
				{
					const PxVec3& a =
						gSceneCpuSphereReverseSweptInitialPositions[
							boundaryFaces[face][0]];
					const PxVec3& b =
						gSceneCpuSphereReverseSweptInitialPositions[
							boundaryFaces[face][1]];
					const PxVec3& c =
						gSceneCpuSphereReverseSweptInitialPositions[
							boundaryFaces[face][2]];
					const PxVec3 closest =
						closestPointOnTriangleForGate(
							world[rigidVertex], a, b, c);
					endpointMinSeparation = PxMin(
						endpointMinSeparation,
						(world[rigidVertex] -
							closest).magnitude());
				}
			}
		}
	}

	const PxVec3& bottom0 =
		gSceneCpuSphereReverseSweptInitialPositions[0];
	const PxVec3& bottom1 =
		gSceneCpuSphereReverseSweptInitialPositions[1];
	const PxVec3& bottom2 =
		gSceneCpuSphereReverseSweptInitialPositions[3];
	for(PxU32 sample = 0; sample <= 256; ++sample)
	{
		const PxReal time = PxReal(sample) / 256.0f;
		for(PxU32 softVertex = 0;
			softVertex <
				gSceneCpuSphereReverseSweptInitialPositions.size();
			++softVertex)
		{
			const PxReal distance = getPointBladeDistance(
				gSceneCpuSphereReverseSweptInitialPositions[
					softVertex],
				time);
			if(!PxIsFinite(distance))
				return false;
			if(reverseCase)
				minimumVertexSweepSeparation = PxMin(
					minimumVertexSweepSeparation, distance);
			else
				midSweepMinSeparation = PxMin(
					midSweepMinSeparation, distance);
		}
		if(reverseCase)
		{
			PxVec3 world[4];
			getBladeWorld(time, world);
			for(PxU32 rigidVertex = 0;
				rigidVertex < 4; ++rigidVertex)
			{
				const PxVec3 projected(
					world[rigidVertex].x,
					bottom0.y,
					world[rigidVertex].z);
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						projected,
						bottom0, bottom1, bottom2);
				if((projected - closest).magnitudeSquared() <=
					1.0e-8f)
					midSweepMinSeparation = PxMin(
						midSweepMinSeparation,
						bottom0.y -
							world[rigidVertex].y);
			}
		}
	}
	if(!reverseCase)
		minimumVertexSweepSeparation = PX_MAX_F32;
	return PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation) &&
		(!reverseCase ||
		 PxIsFinite(minimumVertexSweepSeparation));
}

static bool getSceneCpuVolumeSmoothReverseSeparations(
	PxDeformableVolume* volume,
	const PxVec3& rigidCenter,
	PxReal rigidRadius,
	PxReal capsuleHalfHeight,
	bool convexCase,
	bool triangleMeshCase,
	bool heightFieldCase,
	PxReal& faceSeparation,
	PxReal& minimumVertexSeparation)
{
	if(!volume || !volume->getCollisionMesh() ||
		!rigidCenter.isFinite() ||
		!PxIsFinite(rigidRadius) || rigidRadius <= 0.0f ||
		!PxIsFinite(capsuleHalfHeight) ||
		capsuleHalfHeight < 0.0f ||
		(convexCase && !gSceneCpuRigidConvexMesh) ||
		(triangleMeshCase && !gSceneCpuRigidTriangleMesh) ||
		(heightFieldCase && !gSceneCpuRigidHeightField))
		return false;
	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount = mesh->getNbVertices();
	const PxU32 tetrahedronCount = mesh->getNbTetrahedrons();
	if(!positions || vertexCount == 0 || tetrahedronCount == 0)
		return false;

	minimumVertexSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex < vertexCount; ++vertexIndex)
	{
		const PxVec3 position = positions[vertexIndex].getXYZ();
		if(!position.isFinite())
			return false;
		minimumVertexSeparation = PxMin(
			minimumVertexSeparation,
			[&]()
			{
				if(triangleMeshCase || heightFieldCase)
				{
					const PxTriangleMeshGeometry triangleGeometry(
						gSceneCpuRigidTriangleMesh);
					const PxHeightFieldGeometry heightGeometry(
						gSceneCpuRigidHeightField,
						PxMeshGeometryFlags(),
						0.1f, 0.3f, 0.3f);
					const PxU32 heightFieldTriangles[] =
						{0, 1, 2, 3, 6, 7, 8, 9};
					const PxU32 triangleCount =
						triangleMeshCase
							? gSceneCpuRigidTriangleMesh->
								getNbTriangles()
							: 8u;
					PxReal minimumDistance = PX_MAX_F32;
					for(PxU32 triangleIndex = 0;
						triangleIndex < triangleCount;
						++triangleIndex)
					{
						PxTriangle triangle;
						if(triangleMeshCase)
							PxMeshQuery::getTriangle(
								triangleGeometry,
								PxTransform(rigidCenter),
								triangleIndex, triangle);
						else
							PxMeshQuery::getTriangle(
								heightGeometry,
								PxTransform(rigidCenter),
								heightFieldTriangles[
									triangleIndex],
								triangle);
						const PxVec3 closest =
							closestPointOnTriangleForGate(
								position,
								triangle.verts[0],
								triangle.verts[1],
								triangle.verts[2]);
						minimumDistance = PxMin(
							minimumDistance,
							(position - closest).magnitude());
					}
					return minimumDistance;
				}
				if(convexCase)
				{
					const PxReal squaredDistance =
						PxGeometryQuery::pointDistance(
							position,
							PxConvexMeshGeometry(
								gSceneCpuRigidConvexMesh),
							PxTransform(rigidCenter));
					return squaredDistance >= 0.0f
						? PxSqrt(squaredDistance)
						: -PX_MAX_F32;
				}
				PxVec3 radial = position - rigidCenter;
				radial.x -= PxClamp(
					radial.x,
					-capsuleHalfHeight,
					capsuleHalfHeight);
				return radial.magnitude() - rigidRadius;
			}());
	}

	const PxU32 faceEndpoints[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};
	const bool has16BitIndices =
		mesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* indices16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTetrahedrons())
		: NULL;
	const PxU32* indices32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTetrahedrons());
	if((has16BitIndices && !indices16) ||
		(!has16BitIndices && !indices32))
		return false;

	faceSeparation = PX_MAX_F32;
	for(PxU32 tetrahedronIndex = 0;
		tetrahedronIndex < tetrahedronCount;
		++tetrahedronIndex)
	{
		PxU32 tetrahedron[4];
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
		{
			const PxU32 flatIndex =
				tetrahedronIndex * 4 + endpoint;
			tetrahedron[endpoint] = has16BitIndices
				? PxU32(indices16[flatIndex])
				: indices32[flatIndex];
			if(tetrahedron[endpoint] >= vertexCount)
				return false;
		}
		for(PxU32 faceIndex = 0; faceIndex < 4; ++faceIndex)
		{
			const PxVec3 a =
				positions[tetrahedron[
					faceEndpoints[faceIndex][0]]].getXYZ();
			const PxVec3 b =
				positions[tetrahedron[
					faceEndpoints[faceIndex][1]]].getXYZ();
			const PxVec3 c =
				positions[tetrahedron[
					faceEndpoints[faceIndex][2]]].getXYZ();
			const PxVec3 axisSamples[3] =
			{
				triangleMeshCase
					? rigidCenter +
						PxVec3(0.0f, 0.3f, 0.0f)
					: heightFieldCase
					? rigidCenter +
						PxVec3(0.3f, 0.3f, 0.3f)
					: convexCase
					? rigidCenter +
						PxVec3(0.0f, 0.3f, 0.0f)
					: rigidCenter -
						PxVec3(
							capsuleHalfHeight,
							0.0f, 0.0f),
				rigidCenter,
				rigidCenter +
					PxVec3(capsuleHalfHeight, 0.0f, 0.0f)
			};
			for(PxU32 sampleIndex = 0;
				sampleIndex <
					((convexCase || triangleMeshCase ||
					  heightFieldCase) ? 1u : 3u);
				++sampleIndex)
			{
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						axisSamples[sampleIndex], a, b, c);
				if(!closest.isFinite())
					return false;
				faceSeparation = PxMin(
					faceSeparation,
					(closest - axisSamples[sampleIndex]).
						magnitude() -
						((convexCase || triangleMeshCase ||
						  heightFieldCase)
							? 0.0f : rigidRadius));
			}
		}
	}
	return PxIsFinite(faceSeparation) &&
		faceSeparation < PX_MAX_F32 &&
		PxIsFinite(minimumVertexSeparation) &&
		minimumVertexSeparation < PX_MAX_F32;
}

static bool updateSceneSoftSleepWake()
{
	if(gHeadlessOptions.caseName != "scene-volume-sleep-wake")
		return true;
	if(!gSceneCpuVolume)
		return false;

	if(gMetrics.sceneSoftFirstSlept &&
		!gMetrics.sceneSoftCounterWakeIssued &&
		gMetrics.completedFrames >=
			gMetrics.sceneSoftFirstSleepFrame + 2)
	{
		if(PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneSoftStableWhileSleeping = 1;
		gSceneCpuVolume->setWakeCounter(0.1f);
		gMetrics.sceneSoftCounterWakeIssued = 1;
	}

	if(gMetrics.sceneSoftSecondSlept &&
		!gMetrics.sceneSoftVelocityWakeIssued &&
		gMetrics.completedFrames >=
			gMetrics.sceneSoftSecondSleepFrame + 2)
	{
		if(PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuSecondSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneSoftStableWhileSleeping = 1;
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxTetrahedronMesh* sceneSimulationMesh =
			gSceneCpuVolume->getSimulationMesh();
		const PxU32 vertexCount =
			sceneSimulationMesh->getNbVertices();
		if(!velocities)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] = PxVec4(
				PxVec3(0.0f, 0.5f, 0.0f), velocities[i].w);
		gSceneCpuVelocityWakeCentroidY =
			getSceneCpuVolumeCentroidY();
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		gMetrics.sceneSoftVelocityWakeIssued = 1;
	}
	if(gMetrics.sceneSoftMovedAfterVelocityWake &&
		!gMetrics.sceneSoftVelocityStopIssued)
	{
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxTetrahedronMesh* sceneSimulationMesh =
			gSceneCpuVolume->getSimulationMesh();
		const PxU32 vertexCount =
			sceneSimulationMesh->getNbVertices();
		if(!velocities)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] =
				PxVec4(PxVec3(0.0f), velocities[i].w);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		gSceneCpuVolume->setWakeCounter(0.0f);
		gMetrics.sceneSoftVelocityStopIssued = 1;
	}
	return true;
}

static bool updateSceneSoftRigidWake()
{
	if(gHeadlessOptions.caseName != "scene-volume-rigid-wake")
		return true;
	if(!gSceneCpuVolume)
		return false;

	if(gMetrics.sceneSoftFirstSlept &&
		!gMetrics.sceneSoftRigidWakeActorAdded &&
		gMetrics.completedFrames >=
			gMetrics.sceneSoftFirstSleepFrame + 2)
	{
		if(PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneSoftStableWhileSleeping = 1;
		gSceneCpuRigidWakeCentroidY =
			getSceneCpuVolumeCentroidY();
		if(!addSceneDynamicBox(
			PxVec3(0.0f, 3.70f, 0.0f),
			PxVec3(4.0f, 0.25f, 4.0f), false))
			return false;
		if(!PxRigidBodyExt::setMassAndUpdateInertia(
			*gSceneCpuDynamicActor, 1000.0f))
			return false;
		gSceneCpuDynamicActor->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
		gSceneCpuDynamicActor->setSleepThreshold(0.0f);
		gSceneCpuDynamicActor->wakeUp();
		gMetrics.sceneSoftRigidWakeActorAdded = 1;
	}
	if(gMetrics.sceneSoftWokeByRigid &&
		!gMetrics.sceneSoftVelocityStopIssued)
	{
		if(!gSceneCpuDynamicActor ||
			gSceneCpuDynamicActor->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuDynamicActor);
		gMetrics.sceneDynamicActorRemoved =
			gSceneCpuDynamicActor->getScene() == NULL ? 1u : 0u;
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxU32 vertexCount =
			gSceneCpuVolume->getSimulationMesh()->getNbVertices();
		if(!velocities)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] =
				PxVec4(PxVec3(0.0f), velocities[i].w);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		gSceneCpuVolume->setWakeCounter(0.0f);
		gMetrics.sceneSoftVelocityStopIssued = 1;
	}
	return true;
}

static bool setSceneCpuVolumeVelocity(
	PxDeformableVolume& volume, const PxVec3& velocity)
{
	PxVec4* velocities = volume.getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		volume.getSimulationMesh();
	if(!velocities || !simulationMesh)
		return false;
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	for(PxU32 i = 0; i < vertexCount; ++i)
		velocities[i] = PxVec4(velocity, velocities[i].w);
	volume.markDirty(PxDeformableVolumeDataFlag::eSIM_VELOCITY);
	return true;
}

static bool setSceneCpuVolumeDeformingReverseVelocity(
	PxDeformableVolume& volume, PxReal speed)
{
	PxVec4* velocities = volume.getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		volume.getSimulationMesh();
	if(!velocities || !simulationMesh || !PxIsFinite(speed))
		return false;
	const PxVec3* restVertices = simulationMesh->getVertices();
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	if(!restVertices || vertexCount == 0)
		return false;
	PxU32 movingVertex = 0;
	PxReal minimumRestRadiusSq =
		restVertices[0].magnitudeSquared();
	for(PxU32 i = 1; i < vertexCount; ++i)
	{
		const PxReal restRadiusSq =
			restVertices[i].magnitudeSquared();
		if(restRadiusSq < minimumRestRadiusSq)
		{
			minimumRestRadiusSq = restRadiusSq;
			movingVertex = i;
		}
	}
	for(PxU32 i = 0; i < vertexCount; ++i)
		velocities[i] = PxVec4(
			i == movingVertex
				? PxVec3(0.0f, speed, 0.0f)
				: PxVec3(0.0f),
			velocities[i].w);
	volume.markDirty(PxDeformableVolumeDataFlag::eSIM_VELOCITY);
	return true;
}

static bool updateSceneSoftActorChurn()
{
	if(gHeadlessOptions.caseName != "scene-volume-soft-churn")
		return true;
	if(!gSceneCpuVolume || !gSceneCpuSecondVolume || !gScene)
		return false;
	if(gMetrics.completedFrames < 30)
		return true;

	const PxU32 phase = (gMetrics.completedFrames - 30) % 6;
	if(gSceneCpuSoftChurnMovePending)
	{
		PxDeformableVolume* movingVolume =
			phase == 1 ? gSceneCpuSecondVolume :
			(phase == 3 ? gSceneCpuVolume : NULL);
		if(!movingVolume || movingVolume->getScene() != gScene)
			return false;
		const PxReal displacement =
			getSceneCpuVolumeCentroidY(movingVolume) -
			gSceneCpuSoftChurnCentroidY;
		if(!PxIsFinite(displacement) ||
			displacement <= 1.0e-5f ||
			displacement >= 0.01f)
			return false;
		gMetrics.sceneSoftChurnPostCompactMoveCount++;
		gSceneCpuSoftChurnMovePending = false;
	}
	if(phase == 0)
	{
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gMetrics.sceneSoftChurnRemoveCount++;
		gSceneCpuSoftChurnCentroidY =
			getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
		if(!setSceneCpuVolumeVelocity(
			*gSceneCpuSecondVolume, PxVec3(0.0f, 0.05f, 0.0f)))
			return false;
		gSceneCpuSoftChurnMovePending = true;
	}
	else if(phase == 1)
	{
		if(!setSceneCpuVolumeVelocity(
			*gSceneCpuSecondVolume, PxVec3(0.0f)))
			return false;
		gSceneCpuSecondVolume->setWakeCounter(0.0f);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		gMetrics.sceneSoftChurnReaddCount++;
	}
	else if(phase == 2)
	{
		if(gSceneCpuSecondVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuSecondVolume);
		if(gSceneCpuSecondVolume->getScene() != NULL)
			return false;
		gMetrics.sceneSoftChurnRemoveCount++;
		gSceneCpuSoftChurnCentroidY =
			getSceneCpuVolumeCentroidY(gSceneCpuVolume);
		if(!setSceneCpuVolumeVelocity(
			*gSceneCpuVolume, PxVec3(0.0f, 0.05f, 0.0f)))
			return false;
		gSceneCpuSoftChurnMovePending = true;
	}
	else if(phase == 3)
	{
		if(!setSceneCpuVolumeVelocity(
			*gSceneCpuVolume, PxVec3(0.0f)))
			return false;
		gSceneCpuVolume->setWakeCounter(0.0f);
		if(gSceneCpuSecondVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuSecondVolume);
		if(gSceneCpuSecondVolume->getScene() != gScene ||
			gScene->getNbDeformableVolumes() != 2)
			return false;
		gMetrics.sceneSoftChurnReaddCount++;
		gMetrics.sceneSoftChurnCycles++;
		const PxBounds3 firstBounds =
			gSceneCpuVolume->getWorldBounds();
		const PxBounds3 secondBounds =
			gSceneCpuSecondVolume->getWorldBounds();
		const bool stable =
			firstBounds.isValid() &&
			firstBounds.minimum.isFinite() &&
			firstBounds.maximum.isFinite() &&
			secondBounds.isValid() &&
			secondBounds.minimum.isFinite() &&
			secondBounds.maximum.isFinite();
		if(gMetrics.sceneSoftChurnCycles == 1)
			gMetrics.sceneSoftChurnStable = stable ? 1u : 0u;
		else if(!stable)
			gMetrics.sceneSoftChurnStable = 0;
	}
	return true;
}

static bool updateSceneSoftBufferMutation()
{
	if(gHeadlessOptions.caseName !=
		"scene-volume-buffer-mutation")
		return true;
	if(!gSceneCpuVolume ||
		!gSceneCpuVolume->getSimulationMesh())
		return false;

	PxVec4* positions =
		gSceneCpuVolume->getSimPositionInvMassBufferH();
	PxVec4* velocities =
		gSceneCpuVolume->getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		gSceneCpuVolume->getSimulationMesh();
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	if(!positions || !velocities || vertexCount < 2)
		return false;

	if(gMetrics.sceneSoftFirstSlept &&
		!gMetrics.sceneBufferMutationIssued &&
		gMetrics.completedFrames >=
			gMetrics.sceneSoftFirstSleepFrame + 2)
	{
		if(PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneSoftStableWhileSleeping = 1;
		gSceneCpuBufferOriginalInvMass = positions[0].w;
		if(!PxIsFinite(gSceneCpuBufferOriginalInvMass) ||
			gSceneCpuBufferOriginalInvMass <= 0.0f)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			const PxVec3 translated =
				positions[i].getXYZ() + PxVec3(0.0f, 0.25f, 0.0f);
			positions[i] = PxVec4(translated, positions[i].w);
			velocities[i] =
				PxVec4(PxVec3(0.0f), velocities[i].w);
		}
		positions[0].w = 0.0f;
		gSceneCpuBufferPinTarget = positions[0].getXYZ();
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlags(
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_POSITION_INVMASS) |
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_VELOCITY)));
		gMetrics.sceneBufferMutationIssued = 1;
	}
	else if(gMetrics.sceneBufferMutationApplied &&
		gMetrics.sceneBufferPinHeld &&
		!gMetrics.sceneBufferDriveIssued)
	{
		gSceneCpuBufferDynamicBaseline =
			positions[1].getXYZ();
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			const PxVec3 velocity =
				i == 0 ? PxVec3(0.0f) :
					PxVec3(0.05f, 0.0f, 0.0f);
			velocities[i] =
				PxVec4(velocity, velocities[i].w);
		}
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		gMetrics.sceneBufferDriveIssued = 1;
	}
	else if(gMetrics.sceneBufferDynamicMoved &&
		gMetrics.sceneBufferPinHeld &&
		!gMetrics.sceneBufferInvMassRestored)
	{
		positions[0].w = gSceneCpuBufferOriginalInvMass;
		gSceneCpuBufferRestoredBaseline =
			positions[0].getXYZ();
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] = PxVec4(
				PxVec3(0.05f, 0.0f, 0.0f),
				velocities[i].w);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlags(
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_POSITION_INVMASS) |
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_VELOCITY)));
		gMetrics.sceneBufferInvMassRestored = 1;
	}
	else if(gMetrics.sceneBufferRestoredMoved &&
		!gMetrics.sceneBufferResetIssued)
	{
		const PxVec3* simulationVertices =
			simulationMesh->getVertices();
		if(!simulationVertices)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			positions[i] = PxVec4(
				simulationVertices[i] +
					PxVec3(0.0f, 4.25f, 0.0f),
				i == 0 ? gSceneCpuBufferOriginalInvMass :
					positions[i].w);
			velocities[i] =
				PxVec4(PxVec3(0.0f), velocities[i].w);
		}
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlags(
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_POSITION_INVMASS) |
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_VELOCITY)));
		gSceneCpuVolume->setWakeCounter(0.0f);
		gMetrics.sceneBufferResetIssued = 1;
	}
	return true;
}

// ---------------------------------------------------------------------------
static bool initPhysicsInternal(
	bool interactive, const std::string& caseName)
{
	gMetrics = DeformableVolumeMetrics();
	gPerformance = DeformableVolumePerformanceMetrics();
	gPerformance.warmupFrames = gProfileWarmupFrames;
	gSceneCpuSoftChurnCentroidY = 0.0f;
	gSceneCpuSoftChurnMovePending = false;
	gSceneCpuBufferPinTarget = PxVec3(0.0f);
	gSceneCpuBufferDynamicBaseline = PxVec3(0.0f);
	gSceneCpuBufferRestoredBaseline = PxVec3(0.0f);
	gSceneCpuBufferOriginalInvMass = 0.0f;
	gSceneCpuMultiPrimaryRemovedCentroidY = 0.0f;
	gSceneCpuMultiSecondaryAtReleaseCentroidY = 0.0f;
	gSceneCpuSoftSoftTargetBaseline = PxVec3(0.0f);
	gSceneCpuVolumeKinematicTargets.clear();
	gSceneCpuVolumeKinematicInitial.clear();
	gVolumeSkinningBindings.clear();
	gVolumeSkinningTriangles.clear();
	gVolumeSkinningPositions.clear();
	gVolumeSkinningNormals.clear();
	gVolumeSkinningInitialPositions.clear();
	gVolumeAvbdSkinningRenderData =
		VolumeAvbdSkinningRenderData();
	gVolumeSkinningMetrics = VolumeSkinningMetrics();
	gSphereReverseFeatureMetrics =
		SphereReverseFeatureMetrics();
	gSphereReverseSweptMetrics =
		SphereReverseSweptMetrics();
	gDeformingVolumeReverseSweptMetrics =
		DeformingVolumeReverseSweptMetrics();
	gCapsuleRotationalSweepMetrics =
		CapsuleRotationalSweepMetrics();
	gSceneCpuDeformingReverseSweptFreeEndPositions.reset();
	gSceneCpuCapsuleRotationalSweptInitialPositions.reset();
	gSceneCpuVolumePartialProbe = PX_MAX_U32;
	gSceneCpuVolumePartialActivationStart = PxVec3(0.0f);
	gSceneCpuKinematicCommandY = 0.0f;
	gSceneCpuKinematicSoftBaselineY = 0.0f;
	gSceneCpuMovingSpherePositiveInitial = PxVec3(0.0f);
	gSceneCpuMovingSphereNegativeInitial = PxVec3(0.0f);
	gSceneCpuSphereReversePositiveInitial = PxVec3(0.0f);
	gSceneCpuSphereReverseNegativeInitial = PxVec3(0.0f);
	gSceneCpuElementAttachment = false;
	for(PxU32 i = 0; i < 4; ++i)
		gSceneCpuAttachmentVertices[i] = 0;
	gSceneCpuRigidAttachmentInitialPosition = PxVec3(0.0f);
	gSceneCpuRigidAttachmentLocalOffset = PxVec3(0.0f);
	gSceneCpuKinematicAttachmentProgress = 0.0f;
	gSceneCpuKinematicAttachmentSoftBaseline = PxVec3(0.0f);
	gSceneCpuKinematicAttachmentCommand = PxTransform(PxIdentity);
	gSceneCpuAttachmentArticulation = NULL;
	gSceneCpuAttachmentRoot = NULL;
	gSceneCpuAttachmentLink = NULL;
	gSceneCpuAttachmentBody = NULL;
	gSceneCpuArticulationRootInitialPose = PxTransform(PxIdentity);
	gSceneCpuArticulationChildInitialPose = PxTransform(PxIdentity);
	gErrorCallback.reset();
	gParticles.clear();
	gSoftBodies.clear();
	gContacts.clear();
	gRigidBoxes.clear();
	gSoftBodyRenderData.clear();
	gInitialCentroids.clear();
	initOGCParams();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;

	if(interactive)
	{
		gPvd = PxCreatePvd(*gFoundation);
		if(gPvd)
		{
			PxPvdTransport* transport =
				PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
			if(transport)
				gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
		}
	}

	gPhysics    = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	if(!gExtensionsInitialized)
		return false;

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	sceneDesc.solverType = interactive ?
		PxSolverType::eAVBD : gHeadlessOptions.solverType;
	if(caseName == "scene-volume-element-filter" ||
		caseName == "scene-volume-partial-element-filter")
		sceneDesc.flags |= PxSceneFlag::eDISABLE_SLEEPING;
	const PxU32 workerCount =
		interactive ? 2u : gHeadlessOptions.dispatcherThreads;
	gDispatcher = PxDefaultCpuDispatcherCreate(workerCount);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader  = PxDefaultSimulationFilterShader;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;
	// Vertex-block nonlinear GS is intentionally serial until a colored
	// schedule has a numerical-equivalence gate.
	gPerformance.softWorkers = 1;
	gMetrics.solverReadbackMatched =
		gScene->getSolverType() == sceneDesc.solverType;
	if(caseName == "scene-volume-multi-scene-isolation")
	{
		gSceneCpuSecondScene = gPhysics->createScene(sceneDesc);
		if(!gSceneCpuSecondScene)
			return false;
		gMetrics.sceneSecondSceneCreated = 1;
		gMetrics.sceneSecondSceneSolverMatched =
			gSceneCpuSecondScene->getSolverType() ==
				sceneDesc.solverType ? 1u : 0u;
	}

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if (interactive && pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
	if(!gMaterial)
		return false;

	if(isSceneCpuVolumeCase(caseName))
	{
		if(caseName == "scene-volume-ground" ||
			caseName == "scene-volume-element-filter" ||
			caseName == "scene-volume-partial-element-filter" ||
			caseName ==
				"scene-volume-max-depenetration-velocity")
		{
			PxRigidStatic* ground = PxCreatePlane(
				*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
			if(!ground)
				return false;
			gScene->addActor(*ground);
			if(caseName == "scene-volume-element-filter" ||
				caseName ==
					"scene-volume-partial-element-filter")
				gSceneCpuStaticActor = ground;
		}
		else if(caseName == "scene-volume-static-box" ||
			caseName == "scene-volume-static-churn")
		{
			if(!addSceneStaticBox(
				PxVec3(0.0f, 0.5f, 0.0f),
				PxVec3(20.0f, 0.5f, 20.0f)))
				return false;
		}
		else if(isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
					caseName))
		{
			const bool reverseFeature =
				isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
					caseName);
			const bool staticTarget =
				isSceneCpuVolumeStaticTriangleSurfaceSweptCcdCase(
					caseName);
			const bool heightField =
				isSceneCpuVolumeHeightFieldSweptCcdCase(
					caseName);
			const bool rotationalTarget =
				isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
					caseName);
			const PxReal startY = staticTarget ? 0.0f : -1.1f;
			const PxVec3 positiveCenter = rotationalTarget
				? (reverseFeature
					? (heightField
						? PxVec3(
							-3.22791624f, -1.25f,
							0.233333334f)
						: PxVec3(
							-2.32810450f, -0.75f,
							0.333333343f))
					: (heightField
						? PxVec3(
							-3.56124949f, -1.25f, -0.1f)
						: PxVec3(
							-2.66143775f, -0.75f, 0.0f)))
				: reverseFeature
					? (heightField
						? PxVec3(-1.93f, startY, 0.03f)
						: PxVec3(-1.63f, startY, 0.33f))
					: (heightField
						? PxVec3(-2.0f, startY, 0.0f)
						: PxVec3(-1.5f, startY, 0.5f));
			const PxVec3 negativeCenter = rotationalTarget
				? (reverseFeature
					? (heightField
						? PxVec3(
							-0.227916166f, -1.25f,
							0.233333334f)
						: PxVec3(
							0.671895504f, -0.75f,
							0.333333343f))
					: (heightField
						? PxVec3(
							-0.561249495f, -1.25f, -0.1f)
						: PxVec3(
							0.338562191f, -0.75f, 0.0f)))
				: reverseFeature
					? (heightField
						? PxVec3(1.07f, startY, 0.03f)
						: PxVec3(1.37f, startY, 0.33f))
					: (heightField
						? PxVec3(1.0f, startY, 0.0f)
						: PxVec3(1.5f, startY, 0.5f));
			if(rotationalTarget
				? !addSceneRotatingKinematicTriangleSurfacePair(
					positiveCenter, negativeCenter,
					heightField)
				: staticTarget
				? !addSceneStaticTriangleSurfacePair(
					positiveCenter, negativeCenter,
					heightField, reverseFeature)
				: !addSceneMovingKinematicTriangleSurfacePair(
					positiveCenter, negativeCenter, 1.1f,
					heightField, reverseFeature))
				return false;
		}
		else if(caseName ==
			"scene-volume-deforming-sphere-reverse-swept-ccd")
		{
			const PxVec3 sphereCenters[] =
			{
				PxVec3(-1.68f, 0.0f, 0.32f),
				PxVec3(1.32f, 0.0f, 0.32f)
			};
			if(!addSceneStaticSphereCluster(
					sphereCenters, 2, 0.18f))
				return false;
		}
		else if(caseName ==
			"scene-volume-deforming-capsule-reverse-swept-ccd")
		{
			const PxVec3 capsuleCenters[] =
			{
				PxVec3(-1.68f, 0.0f, 0.32f),
				PxVec3(1.32f, 0.0f, 0.32f)
			};
			if(!addSceneStaticCapsuleCluster(
					capsuleCenters, 2, 0.18f, 0.02f))
				return false;
		}
		else if(caseName ==
			"scene-volume-deforming-convex-reverse-swept-ccd")
		{
			const PxVec3 convexCenters[] =
			{
				PxVec3(-1.68f, 0.0f, 0.32f),
				PxVec3(1.32f, 0.0f, 0.32f)
			};
			if(!addSceneStaticConvexCluster(
					convexCenters, 2,
					eSCENE_CPU_CONVEX_DEFORMING_REVERSE_SWEPT))
				return false;
		}
		else if(caseName ==
			"scene-volume-static-sphere-reverse-swept-ccd")
		{
			const PxVec3 sphereCenters[] =
			{
				PxVec3(-1.35f, 0.0f, 0.15f),
				PxVec3(1.65f, 0.0f, 0.15f)
			};
			if(!addSceneStaticSphereCluster(
					sphereCenters, 2, 0.25f))
				return false;
		}
		else if(caseName ==
			"scene-volume-static-capsule-reverse-swept-ccd")
		{
			const PxVec3 capsuleCenters[] =
			{
				PxVec3(-1.35f, 0.0f, 0.15f),
				PxVec3(1.65f, 0.0f, 0.15f)
			};
			if(!addSceneStaticCapsuleCluster(
					capsuleCenters, 2, 0.25f, 0.02f))
				return false;
		}
		else if(caseName ==
			"scene-volume-static-convex-reverse-swept-ccd")
		{
			const PxVec3 convexCenters[] =
			{
				PxVec3(-1.35f, 0.0f, 0.15f),
				PxVec3(1.65f, 0.0f, 0.15f)
			};
			if(!addSceneStaticConvexCluster(
					convexCenters, 2,
					eSCENE_CPU_CONVEX_REVERSE_SWEPT))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-sphere-reverse-swept-ccd")
		{
			if(!addSceneMovingKinematicFinitePair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					-1.1f, 0.25f, 0.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-capsule-reverse-swept-ccd")
		{
			if(!addSceneMovingKinematicFinitePair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					-1.1f, 0.25f, 0.02f))
				return false;
		}
		else if(caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd")
		{
			if(!addSceneRotatingKinematicCapsulePair(
					PxVec3(-1.45f, -1.0f, 0.25f),
					PxVec3(1.55f, -1.0f, 0.25f),
					0.1f, 1.0f,
					PxQuat(
						PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					PxQuat(
						PxPi * 5.0f / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f))))
				return false;
		}
		else if(caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd")
		{
			if(!addSceneRotatingKinematicConvexPair(
					PxVec3(-1.45f, -0.85f, 0.25f),
					PxVec3(1.55f, -0.85f, 0.25f),
					PxQuat(
						PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					PxQuat(
						PxPi * 5.0f / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f))))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-convex-reverse-swept-ccd")
		{
			if(!addSceneMovingKinematicConvexPair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					-1.1f,
					eSCENE_CPU_CONVEX_REVERSE_SWEPT))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-sphere-reverse-swept-ccd")
		{
			if(!addSceneDynamicFiniteSweepPair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					0.25f, 0.0f, -132.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-capsule-reverse-swept-ccd")
		{
			if(!addSceneDynamicFiniteSweepPair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					0.25f, 0.02f, -132.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd")
		{
			if(!addSceneDynamicRotatingCapsulePair(
					PxVec3(-1.45f, -1.0f, 0.25f),
					PxVec3(1.55f, -1.0f, 0.25f),
					0.1f, 1.0f,
					PxQuat(
						PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					10.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd")
		{
			if(!addSceneDynamicRotatingConvexPair(
					PxVec3(-1.45f, -0.85f, 0.25f),
					PxVec3(1.55f, -0.85f, 0.25f),
					PxQuat(
						PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					10.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-convex-reverse-swept-ccd")
		{
			if(!addSceneDynamicConvexSweepPair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					-132.0f,
					eSCENE_CPU_CONVEX_REVERSE_SWEPT))
				return false;
		}
		else if(caseName == "scene-volume-speculative-ccd")
		{
			if(!addSceneStaticBox(
				PxVec3(0.0f, 0.5f, 0.5f),
				PxVec3(4.0f, 0.05f, 2.0f)))
				return false;
		}
		else if(caseName ==
			"scene-volume-plane-speculative-ccd")
		{
			PxRigidStatic* plane = PxCreatePlane(
				*gPhysics,
				PxPlane(0.0f, 1.0f, 0.0f, -0.5f),
				*gMaterial);
			if(!plane)
				return false;
			gScene->addActor(*plane);
		}
		else if(caseName ==
			"scene-volume-sphere-speculative-ccd")
		{
			const PxVec3 sphereCenters[] =
			{
				PxVec3(-2.0f, 0.25f, 0.0f),
				PxVec3(-1.0f, 0.25f, 0.0f),
				PxVec3(-2.0f, 0.25f, 1.0f),
				PxVec3(-1.0f, 0.25f, 1.0f),
				PxVec3(1.0f, 0.25f, 0.0f),
				PxVec3(2.0f, 0.25f, 0.0f),
				PxVec3(1.0f, 0.25f, 1.0f),
				PxVec3(2.0f, 0.25f, 1.0f)
			};
			if(!addSceneStaticSphereCluster(
				sphereCenters,
				sizeof(sphereCenters) /
					sizeof(sphereCenters[0]),
				0.3f))
				return false;
		}
		else if(caseName ==
			"scene-volume-capsule-speculative-ccd")
		{
			const PxVec3 capsuleCenters[] =
			{
				PxVec3(-2.0f, 0.25f, 0.0f),
				PxVec3(-1.0f, 0.25f, 0.0f),
				PxVec3(-2.0f, 0.25f, 1.0f),
				PxVec3(-1.0f, 0.25f, 1.0f),
				PxVec3(1.0f, 0.25f, 0.0f),
				PxVec3(2.0f, 0.25f, 0.0f),
				PxVec3(1.0f, 0.25f, 1.0f),
				PxVec3(2.0f, 0.25f, 1.0f)
			};
			if(!addSceneStaticCapsuleCluster(
				capsuleCenters,
				sizeof(capsuleCenters) /
					sizeof(capsuleCenters[0]),
				0.3f, 0.2f))
				return false;
		}
		else if(caseName ==
			"scene-volume-convex-speculative-ccd")
		{
			const PxVec3 convexCenters[] =
			{
				PxVec3(-1.5f, 0.25f, 0.5f),
				PxVec3(1.5f, 0.25f, 0.5f)
			};
			if(!addSceneStaticConvexCluster(
					convexCenters, 2,
					eSCENE_CPU_CONVEX_SWEPT_BOX))
				return false;
		}
		else if(caseName ==
			"scene-volume-sphere-reverse-feature")
		{
			const PxVec3 sphereCenter(
				-1.75f, 0.0f, 0.25f);
			if(!addSceneStaticSphereCluster(
					&sphereCenter, 1, 0.3f))
				return false;
		}
		else if(caseName ==
			"scene-volume-capsule-reverse-feature")
		{
			if(!addSceneStaticCapsule(
				PxVec3(-1.75f, 0.0f, 0.25f),
				0.3f, 0.15f))
				return false;
		}
		else if(caseName ==
			"scene-volume-convex-reverse-feature")
		{
			if(!addSceneStaticConvex(
				PxVec3(-1.75f, 0.0f, 0.25f), true))
				return false;
		}
		else if(caseName ==
			"scene-volume-triangle-mesh-reverse-feature")
		{
			if(!addSceneStaticTriangleMesh(
				PxVec3(-1.75f, 0.0f, 0.25f), true))
				return false;
		}
		else if(caseName ==
			"scene-volume-heightfield-reverse-feature")
		{
			if(!addSceneStaticHeightField(
				PxVec3(-2.05f, 0.0f, -0.05f), true))
				return false;
		}
		else if(caseName ==
			"scene-volume-moving-kinematic-sphere-speculative-ccd")
		{
			if(!addSceneMovingKinematicFinitePair(
					PxVec3(-1.5f, 2.8f, 0.5f),
					PxVec3(1.5f, 2.8f, 0.5f),
					0.0f, 0.8f, 0.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-moving-kinematic-capsule-speculative-ccd")
		{
			if(!addSceneMovingKinematicFinitePair(
					PxVec3(-1.5f, 2.8f, 0.5f),
					PxVec3(1.5f, 2.8f, 0.5f),
					0.0f, 0.8f, 0.3f))
				return false;
		}
		else if(caseName ==
			"scene-volume-rotating-kinematic-capsule-speculative-ccd")
		{
			if(!addSceneRotatingKinematicCapsulePair(
					PxVec3(-2.9f, 0.0f, 0.0f),
					PxVec3(0.1f, 0.0f, 0.0f),
					0.1f, 1.0f,
					PxQuat(
						-PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					PxQuat(
						PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f))))
				return false;
		}
		else if(caseName ==
			"scene-volume-rotating-kinematic-convex-speculative-ccd")
		{
			if(!addSceneRotatingKinematicConvexPair(
					PxVec3(-2.9f, 0.0f, 0.0f),
					PxVec3(0.1f, 0.0f, 0.0f),
					PxQuat(
						-PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					PxQuat(
						PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f))))
				return false;
		}
		else if(caseName ==
			"scene-volume-moving-kinematic-convex-speculative-ccd")
		{
			if(!addSceneMovingKinematicConvexPair(
					PxVec3(-1.5f, 2.8f, 0.5f),
					PxVec3(1.5f, 2.8f, 0.5f),
					0.0f,
					eSCENE_CPU_CONVEX_SWEPT_BOX))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-sphere-relative-swept-ccd")
		{
			if(!addSceneDynamicFiniteSweepPair(
					PxVec3(-1.5f, 2.8f, 0.5f),
					PxVec3(1.5f, 2.8f, 0.5f),
					0.8f, 0.0f, -132.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-capsule-relative-swept-ccd")
		{
			if(!addSceneDynamicFiniteSweepPair(
					PxVec3(-1.5f, 3.1f, 0.5f),
					PxVec3(1.5f, 3.1f, 0.5f),
					0.8f, 0.3f, -150.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-rotating-capsule-relative-swept-ccd")
		{
			if(!addSceneDynamicRotatingCapsulePair(
					PxVec3(-2.9f, 0.0f, 0.0f),
					PxVec3(0.1f, 0.0f, 0.0f),
					0.1f, 1.0f,
					PxQuat(
						-PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					1.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-rotating-convex-relative-swept-ccd")
		{
			if(!addSceneDynamicRotatingConvexPair(
					PxVec3(-2.9f, 0.0f, 0.0f),
					PxVec3(0.1f, 0.0f, 0.0f),
					PxQuat(
						-PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					1.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-convex-relative-swept-ccd")
		{
			if(!addSceneDynamicConvexSweepPair(
					PxVec3(-1.5f, 2.8f, 0.5f),
					PxVec3(1.5f, 2.8f, 0.5f),
					-132.0f,
					eSCENE_CPU_CONVEX_SWEPT_BOX))
				return false;
		}
		else if(caseName == "scene-volume-multi-soft-islands")
		{
			if(!addSceneDynamicBox(
				PxVec3(-10.0f, 2.5f, 0.0f),
				PxVec3(4.0f, 0.25f, 4.0f)) ||
				!addSceneSecondDynamicBox(
					PxVec3(10.0f, 2.5f, 0.0f),
					PxVec3(4.0f, 0.25f, 4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-multi-dynamic-box")
		{
			if(!addSceneDynamicBox(
				PxVec3(0.0f, 2.5f, 0.0f),
				PxVec3(0.45f, 0.25f, 4.0f)) ||
				!addSceneSecondDynamicBox(
					PxVec3(1.0f, 2.5f, 0.0f),
					PxVec3(0.45f, 0.25f, 4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-kinematic-box")
		{
			if(!addSceneKinematicBox(
				PxVec3(0.0f, 3.5f, 0.0f),
				PxVec3(4.0f, 0.25f, 4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-kinematic-sphere")
		{
			if(!addSceneKinematicSphere(
				PxVec3(0.0f, 3.4f, 0.0f), 0.5f))
				return false;
		}
		else if(caseName == "scene-volume-kinematic-capsule")
		{
			if(!addSceneKinematicCapsule(
				PxVec3(0.0f, 3.4f, 0.0f), 0.5f, 0.3f))
				return false;
		}
		else if(caseName == "scene-volume-kinematic-convex")
		{
			if(!addSceneKinematicConvex(
				PxVec3(0.0f, 3.2f, 0.0f)))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-triangle-mesh")
		{
			if(!addSceneKinematicTriangleMesh(
				PxVec3(0.0f, 3.2f, 0.0f)))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-heightfield")
		{
			if(!addSceneKinematicHeightField(
				PxVec3(-4.0f, 3.2f, -4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-dynamic-box" ||
			caseName == "scene-volume-dynamic-churn")
		{
			if(!addSceneDynamicBox(
				PxVec3(0.0f, 2.5f, 0.0f),
				PxVec3(4.0f, 0.25f, 4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-dynamic-sphere")
		{
			PxRigidStatic* ground = PxCreatePlane(
				*gPhysics,
				PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
				*gMaterial);
			if(!ground)
				return false;
			gScene->addActor(*ground);
			if(!addSceneDynamicSphere(
				PxVec3(0.0f, 1.0f, 0.0f), 0.8f))
				return false;
		}
		else if(caseName == "scene-volume-dynamic-capsule")
		{
			PxRigidStatic* ground = PxCreatePlane(
				*gPhysics,
				PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
				*gMaterial);
			if(!ground)
				return false;
			gScene->addActor(*ground);
			if(!addSceneDynamicCapsule(
				PxVec3(0.0f, 1.0f, 0.0f), 0.8f, 0.3f))
				return false;
		}
		else if(caseName == "scene-volume-dynamic-convex")
		{
			PxRigidStatic* ground = PxCreatePlane(
				*gPhysics,
				PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
				*gMaterial);
			if(!ground)
				return false;
			gScene->addActor(*ground);
			if(!addSceneDynamicConvex(
				PxVec3(0.0f, 1.0f, 0.0f)))
				return false;
		}
		return initSceneCpuVolumeLifecycle();
	}

	// Ground plane
	PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	if(!ground)
		return false;
	gScene->addActor(*ground);

	if(caseName == "volume-ground")
	{
		addCubeSoftBody(PxVec3(0.0f, 3.0f, 0.0f), 0.5f, 3);
	}
	else if(caseName == "volume-static-box")
	{
		addCubeSoftBody(PxVec3(0.0f, 4.0f, 0.0f), 0.5f, 3);
		if(!addRigidBox(
			PxVec3(0.0f, 0.5f, 0.0f),
			PxVec3(2.0f, 0.5f, 2.0f)))
		{
			return false;
		}
	}
	else if(caseName == "soft-soft")
	{
		addCubeSoftBody(PxVec3(0.0f, 1.0f, 0.0f), 0.5f, 3);
		addCubeSoftBody(PxVec3(0.0f, 4.0f, 0.0f), 0.5f, 3);
	}
	else if(caseName == "cone-ground")
	{
		addConeSoftBody(PxVec3(-0.8f, 11.0f, 1.2f));
	}
	else
	{

	// ------------------------------------------------------------------
	// Body 0: Tilted cuboid for visible soft-soft tumbling
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		const PxVec3 center(-1.8f, 8.0f, 0.0f);
		avbdGenerateSubdividedCubeTets(center, 1.0f, 4, verts, tets);
		scaleVerticesAboutCenter(verts, center, PxVec3(1.8f, 0.65f, 0.9f));
		rotateVerticesAroundZ(verts, center, -0.55f);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 160.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 1: Sphere restored as the soft-soft support body
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		avbdGenerateSubdividedSphereTets(PxVec3(-3.8f, 2.0f, 0.0f), 1.8f, 4, verts, tets);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 130.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 2: Cone glancing into the left stack
	//   Uses PxTetMaker conforming->voxel pipeline for uniform voxel tets.
	// ------------------------------------------------------------------
	addConeSoftBody(PxVec3(-0.8f, 11.0f, 1.2f));

	// ------------------------------------------------------------------
	// Body 3: Tilted cuboid (rigid-soft toppling rotation)
	//   Pre-rotated and offset on a narrow edge so rigid-soft torque is obvious.
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		PxVec3 center(7.0f, 4.2f, 0.0f);
		avbdGenerateSubdividedCubeTets(center, 1.0f, 3, verts, tets);
		scaleVerticesAboutCenter(verts, center, PxVec3(1.7f, 0.7f, 0.9f));
		rotateVerticesAroundZ(verts, center, 0.95f);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 160.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 4: Off-center follower that keeps Body 3 rotating after impact.
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		const PxVec3 center(5.4f, 8.8f, 0.3f);
		avbdGenerateSubdividedCubeTets(center, 0.85f, 3, verts, tets);
		rotateVerticesAroundZ(verts, center, -0.28f);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 120.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Rigid box obstacle (narrow support edge for Body 3)
	// ------------------------------------------------------------------
	if(!addRigidBox(
		PxVec3(7.6f, 0.55f, 0.0f),
		PxVec3(0.7f, 0.55f, 2.2f)))
	{
		return false;
	}
	}

	if(gSoftBodies.empty() || gParticles.empty())
		return false;

	// Contact prep can emit more than one surface feature per particle.  Keep
	// the capacity policy caller-owned and reserve outside the timed loop.
	const PxU32 contactCapacity = gParticles.size() * 4;
	gSoftWorkspace.reserve(gParticles.size(), contactCapacity);
	gContacts.reserve(contactCapacity);

	updateRenderData();
	gInitialCentroids.reserve(gSoftBodies.size());
	for(PxU32 bodyId = 0; bodyId < gSoftBodies.size(); ++bodyId)
		gInitialCentroids.pushBack(getSoftBodyCentroid(gSoftBodies[bodyId]));

	gMetrics.initialized = 1;
	gMetrics.particles = gParticles.size();
	gMetrics.softBodies = gSoftBodies.size();
	gMetrics.rigidBoxes = gRigidBoxes.size();
	for(PxU32 bodyId = 0; bodyId < gSoftBodies.size(); ++bodyId)
	{
		gMetrics.tetElements += gSoftBodies[bodyId].compiled.tetElements.size();
		gMetrics.surfaceTriangles +=
			gSoftBodies[bodyId].compiled.surfaceTriangles.size() / 3;
	}
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes = gScene->getNbDeformableVolumes();

	printf("%s: %u particles, %u soft bodies, %u rigid boxes\n",
		AVBD_VOLUME_SNIPPET_NAME,
		gParticles.size(), gSoftBodies.size(), gRigidBoxes.size());
	printf(
		"[AVBD_COMPONENT_TOPOLOGY] particles=%u softBodies=%u "
		"tetElements=%u surfaceTriangles=%u rigidBoxes=%u "
		"sceneStatics=%u sceneDynamics=%u sceneDeformableVolumes=%u\n",
		gMetrics.particles, gMetrics.softBodies, gMetrics.tetElements,
		gMetrics.surfaceTriangles, gMetrics.rigidBoxes,
		gMetrics.sceneStatics, gMetrics.sceneDynamics,
		gMetrics.sceneDeformableVolumes);
	return true;
}

void initPhysics(bool interactive)
{
	gHeadlessOptions.solverType = PxSolverType::eAVBD;
	gHeadlessOptions.caseName = AVBD_VOLUME_VISUAL_CASE;
	gHeadlessOptions.frames = 600;
	gHeadlessOptions.dispatcherThreads = 2;
	if(!initPhysicsInternal(
		interactive, gHeadlessOptions.caseName))
		printf("%s initialization failed.\n",
			AVBD_VOLUME_SNIPPET_NAME);
}

// ---------------------------------------------------------------------------
// Contact re-detection callback for use inside avbdStepSoftBodies outer loop.
// Re-creates all ground + soft-soft contacts with fresh surface positions.
// ---------------------------------------------------------------------------
static void redetectContacts(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts, void* /*userData*/)
{
	AvbdSoftCollisionStats stats;
	avbdDetectAllOGCContacts(
		particles, numParticles,
		softBodies, numSoftBodies,
		gRigidBoxes.begin(), gRigidBoxes.size(),
		NULL, 0,
		contacts, gOGCParams, 0.0f, &stats,
		&gSoftWorkspace.contact);
	gFrameCollisionStats.accumulate(stats);
}

static void recordContactMetrics()
{
	PxU32 groundContacts = 0;
	PxU32 rigidContacts = 0;
	PxU32 softContacts = 0;
	for(PxU32 contactId = 0; contactId < gContacts.size(); ++contactId)
	{
		const AvbdSoftContact& contact = gContacts[contactId];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		if(!geometry.source.isValid())
			gMetrics.invalidContactSourceSamples++;
		if(geometry.source.type == AvbdSoftContactSource::eGROUND)
			groundContacts++;
		else if(
			geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE ||
			geometry.source.type == AvbdSoftContactSource::eSELF_SURFACE)
			softContacts++;
		else if(geometry.source.type == AvbdSoftContactSource::eRIGID_SDF)
			rigidContacts++;
		else
		{
			// Legacy fallback remains diagnostic-only while all production
			// detectors migrate to explicit source identities.
			if(geometry.targetKind ==
				AvbdSoftContactTargetKind::eWORLD_STATIC)
				groundContacts++;
			else if(geometry.targetKind ==
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE)
				softContacts++;
			else
				rigidContacts++;
		}
	}
	if(groundContacts)
		gMetrics.groundContactFrames++;
	if(rigidContacts)
		gMetrics.rigidContactFrames++;
	if(softContacts)
		gMetrics.softContactFrames++;
	gMetrics.maxGroundContacts =
		PxMax(gMetrics.maxGroundContacts, groundContacts);
	gMetrics.maxRigidContacts =
		PxMax(gMetrics.maxRigidContacts, rigidContacts);
	gMetrics.maxSoftContacts =
		PxMax(gMetrics.maxSoftContacts, softContacts);
}

static void recordStateMetrics()
{
	for(PxU32 particleId = 0; particleId < gParticles.size(); ++particleId)
	{
		const AvbdSoftParticle& particle = gParticles[particleId];
		if(!particle.position.isFinite() || !particle.velocity.isFinite())
		{
			gMetrics.nonFiniteParticleSamples++;
			continue;
		}
		gMetrics.minY = PxMin(gMetrics.minY, particle.position.y);
		gMetrics.maxY = PxMax(gMetrics.maxY, particle.position.y);
		gMetrics.maxParticleSpeed =
			PxMax(gMetrics.maxParticleSpeed, particle.velocity.magnitude());
	}

	for(PxU32 bodyId = 0; bodyId < gSoftBodies.size(); ++bodyId)
	{
		const AvbdSoftBody& body = gSoftBodies[bodyId];
		PxReal restVolume = 0.0f;
		PxReal currentVolume = 0.0f;
		for(PxU32 elementId = 0;
			elementId < body.compiled.tetElements.size(); ++elementId)
		{
			const AvbdTetElement& tet = body.compiled.tetElements[elementId];
			const PxVec3& x0 = gParticles[tet.p0].position;
			const PxMat33 ds(
				gParticles[tet.p1].position - x0,
				gParticles[tet.p2].position - x0,
				gParticles[tet.p3].position - x0);
			const PxReal detF = (ds * tet.DmInv).getDeterminant();
			restVolume += tet.restVolume;
			if(!PxIsFinite(detF))
			{
				gMetrics.invertedElementSamples++;
				if(gMetrics.firstInversionFrame == PX_MAX_U32)
				{
					gMetrics.firstInversionFrame = gMetrics.completedFrames;
					gMetrics.firstInversionBody = bodyId;
					gMetrics.firstInversionElement = elementId;
				}
				if(bodyId < 32)
					gMetrics.invertedBodiesMask |= 1u << bodyId;
				continue;
			}
			gMetrics.minDetF = PxMin(gMetrics.minDetF, detF);
			gMetrics.maxDetF = PxMax(gMetrics.maxDetF, detF);
			if(detF <= 0.0f)
			{
				gMetrics.invertedElementSamples++;
				if(gMetrics.firstInversionFrame == PX_MAX_U32)
				{
					gMetrics.firstInversionFrame = gMetrics.completedFrames;
					gMetrics.firstInversionBody = bodyId;
					gMetrics.firstInversionElement = elementId;
				}
				if(bodyId < 32)
					gMetrics.invertedBodiesMask |= 1u << bodyId;
			}
			currentVolume += detF * tet.restVolume;
		}
		if(restVolume > 0.0f)
		{
			const PxReal ratio = currentVolume / restVolume;
			if(PxIsFinite(ratio))
			{
				gMetrics.minBodyVolumeRatio =
					PxMin(gMetrics.minBodyVolumeRatio, ratio);
				gMetrics.maxBodyVolumeRatio =
					PxMax(gMetrics.maxBodyVolumeRatio, ratio);
			}
			else
				gMetrics.nonFiniteParticleSamples++;
		}
		const PxVec3 centroid = getSoftBodyCentroid(body);
		if(centroid.isFinite() && bodyId < gInitialCentroids.size())
		{
			gMetrics.maxCentroidDrop = PxMax(
				gMetrics.maxCentroidDrop,
				gInitialCentroids[bodyId].y - centroid.y);
		}
	}
}

static bool translateSceneCpuVolumeState(
	PxDeformableVolume& volume, const PxVec3& translation)
{
	PxVec4* positions = volume.getSimPositionInvMassBufferH();
	PxVec4* velocities = volume.getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		volume.getSimulationMesh();
	if(!positions || !velocities || !simulationMesh)
		return false;
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		positions[i] = PxVec4(
			positions[i].getXYZ() + translation,
			positions[i].w);
		velocities[i] =
			PxVec4(PxVec3(0.0f), velocities[i].w);
	}
	volume.markDirty(
		PxDeformableVolumeDataFlags(
			PxU32(
				PxDeformableVolumeDataFlag::
					eSIM_POSITION_INVMASS) |
			PxU32(
				PxDeformableVolumeDataFlag::
					eSIM_VELOCITY)));
	return true;
}

static bool resetSceneCpuVolumeState(
	PxDeformableVolume& volume, const PxVec3& translation)
{
	PxVec4* positions = volume.getSimPositionInvMassBufferH();
	PxVec4* velocities = volume.getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		volume.getSimulationMesh();
	if(!positions || !velocities || !simulationMesh)
		return false;
	const PxVec3* restPositions = simulationMesh->getVertices();
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	if(!restPositions)
		return false;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		positions[i] = PxVec4(
			restPositions[i] + translation, positions[i].w);
		velocities[i] =
			PxVec4(PxVec3(0.0f), velocities[i].w);
	}
	volume.markDirty(
		PxDeformableVolumeDataFlags(
			PxU32(
				PxDeformableVolumeDataFlag::
					eSIM_POSITION_INVMASS) |
			PxU32(
				PxDeformableVolumeDataFlag::
					eSIM_VELOCITY)));
	return true;
}

static bool updateSceneSoftSoftWake()
{
	if(gHeadlessOptions.caseName !=
		"scene-volume-soft-soft-wake")
		return true;
	if(!gSceneCpuVolume || !gSceneCpuSecondVolume)
		return false;
	if(gMetrics.sceneSoftSoftTargetMoved &&
		!gMetrics.sceneSoftSoftResetIssued)
	{
		if(!resetSceneCpuVolumeState(
				*gSceneCpuVolume, PxVec3(0.0f, 4.0f, 0.0f)) ||
			!resetSceneCpuVolumeState(
				*gSceneCpuSecondVolume, PxVec3(3.0f, 4.0f, 0.0f)))
			return false;
		gSceneCpuVolume->setWakeCounter(0.0f);
		gSceneCpuSecondVolume->setWakeCounter(0.0f);
		gMetrics.sceneSoftSoftResetIssued = 1;
		return true;
	}
	if(!gMetrics.sceneSoftSoftBothSlept ||
		gMetrics.sceneSoftSoftDriveIssued)
		return true;

	gSceneCpuSoftSoftTargetBaseline =
		getSceneCpuVolumeCentroid(gSceneCpuVolume);
	if(!translateSceneCpuVolumeState(
			*gSceneCpuSecondVolume,
			PxVec3(-2.05f, 0.0f, 0.0f)))
		return false;
	gMetrics.sceneSoftSoftDriveIssued = 1;
	return true;
}

static bool stepSceneMultiSceneIsolation(PxReal dt)
{
	if(!gSceneCpuSecondScene || !gSceneCpuVolume ||
		!gSceneCpuSecondVolume)
		return false;

	const bool profileFrame =
		gMetrics.completedFrames >= gProfileWarmupFrames;
	PxTime frameTimer;
	if((gMetrics.completedFrames == 30 ||
			gMetrics.completedFrames == 92) &&
		!translateSceneCpuVolumeState(
			*gSceneCpuSecondVolume,
			PxVec3(0.0f, -0.08f, 0.0f)))
		return false;
	if(gMetrics.completedFrames == 60)
	{
		if(!gScene || gSceneCpuVolume->getScene() != gScene ||
			gScene->getNbDeformableVolumes() != 1)
			return false;
		gSceneCpuMultiPrimaryRemovedCentroidY =
			getSceneCpuVolumeCentroidY();
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL ||
			gScene->getNbDeformableVolumes() != 0)
			return false;
		gMetrics.sceneActorRemoved = 1;
	}
	if(gMetrics.sceneActorRemoved)
	{
		const PxReal detachedDeviation = PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuMultiPrimaryRemovedCentroidY);
		if(!PxIsFinite(detachedDeviation) ||
			detachedDeviation > 1.0e-6f)
			return false;
		if(gMetrics.completedFrames >= 62)
			gMetrics.sceneMultiPrimaryDetachedStable = 1;
	}
	if(gMetrics.completedFrames == 90)
	{
		if(!gScene || gScene->getNbDeformableVolumes() != 0 ||
			gSceneCpuVolume->getScene() != NULL)
			return false;
		gSceneCpuMultiSecondaryAtReleaseCentroidY =
			getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
		PX_RELEASE(gScene);
		gMetrics.scenePrimarySceneReleased = 1;
	}

	PxTime sceneTimer;
	if(gScene)
	{
		gScene->simulate(dt);
		if(!gScene->fetchResults(true))
		{
			gMetrics.fetchFailures++;
			return false;
		}
	}
	gSceneCpuSecondScene->simulate(dt);
	if(!gSceneCpuSecondScene->fetchResults(true))
	{
		gMetrics.fetchFailures++;
		return false;
	}
	const PxF64 sceneMs =
		sceneTimer.getElapsedSeconds() * 1000.0;
	gMetrics.completedFrames++;

	PxDeformableVolume* volumes[2] =
	{
		gSceneCpuVolume,
		gSceneCpuSecondVolume
	};
	for(PxU32 volumeId = 0; volumeId < 2; ++volumeId)
	{
		PxDeformableVolume* volume = volumes[volumeId];
		const PxVec4* positions =
			volume->getSimPositionInvMassBufferH();
		const PxVec4* velocities =
			volume->getSimVelocityBufferH();
		const PxU32 vertexCount =
			volume->getSimulationMesh()->getNbVertices();
		if(!positions || !velocities)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			const PxVec3 position = positions[i].getXYZ();
			const PxVec3 velocity = velocities[i].getXYZ();
			if(!position.isFinite() || !velocity.isFinite() ||
				!PxIsFinite(positions[i].w) ||
				!PxIsFinite(velocities[i].w))
			{
				gMetrics.nonFiniteParticleSamples++;
				continue;
			}
			gMetrics.minY = PxMin(gMetrics.minY, position.y);
			gMetrics.maxY = PxMax(gMetrics.maxY, position.y);
			gMetrics.maxParticleSpeed = PxMax(
				gMetrics.maxParticleSpeed, velocity.magnitude());
		}
	}

	if(gScene && gSceneCpuVolume->getScene() == gScene)
	{
		const PxBounds3 bounds = gSceneCpuVolume->getWorldBounds();
		if(bounds.isValid() && bounds.minimum.isFinite() &&
			bounds.maximum.isFinite())
			gMetrics.sceneBoundsFinite = 1;
		const bool sleeping = gSceneCpuVolume->isSleeping();
		if(!gMetrics.sceneSoftFirstSlept && sleeping)
		{
			gMetrics.sceneSoftFirstSlept = 1;
			gMetrics.sceneSoftFirstSleepFrame =
				gMetrics.completedFrames;
			gSceneCpuFirstSleepCentroidY =
				getSceneCpuVolumeCentroidY();
		}
		if(gMetrics.sceneSoftFirstSlept &&
			gMetrics.completedFrames >=
				gMetrics.sceneSoftFirstSleepFrame + 2 &&
			PxAbs(getSceneCpuVolumeCentroidY() -
				gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneMultiPrimaryStable = 1;
	}

	if(gSceneCpuSecondVolume->getScene() !=
			gSceneCpuSecondScene ||
		gSceneCpuSecondScene->getNbDeformableVolumes() != 1)
		return false;
	const PxReal secondCentroidY =
		getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
	gMetrics.sceneSecondVolumeFinalCentroidY =
		secondCentroidY;
	gMetrics.sceneSecondVolumeMaxCentroidDrop = PxMax(
		gMetrics.sceneSecondVolumeMaxCentroidDrop,
		gSceneCpuSecondVolumeInitialCentroidY - secondCentroidY);
	gMetrics.maxCentroidDrop = PxMax(
		gMetrics.maxCentroidDrop,
		gSceneCpuSecondVolumeInitialCentroidY - secondCentroidY);
	const PxBounds3 secondBounds =
		gSceneCpuSecondVolume->getWorldBounds();
	if(secondBounds.isValid() &&
		secondBounds.minimum.isFinite() &&
		secondBounds.maximum.isFinite())
		gMetrics.sceneSecondVolumeBoundsFinite = 1;
	if(!gMetrics.scenePrimarySceneReleased &&
		gSceneCpuSecondVolumeInitialCentroidY -
			secondCentroidY > 0.02f)
		gMetrics.sceneMultiSecondaryUpdatedBeforeRelease = 1;
	if(gMetrics.scenePrimarySceneReleased &&
		gSceneCpuMultiSecondaryAtReleaseCentroidY -
			secondCentroidY > 0.02f)
		gMetrics.sceneMultiSecondaryUpdatedAfterRelease = 1;

	if(profileFrame)
	{
		gPerformance.profiledFrames++;
		gPerformance.sceneMs += sceneMs;
		gPerformance.stepSamplesMs.pushBack(
			PxReal(frameTimer.getElapsedSeconds() * 1000.0));
	}
	return true;
}

static bool updateSceneVolumeWorldPin()
{
	if(gHeadlessOptions.caseName != "scene-volume-world-pin" &&
		gHeadlessOptions.caseName !=
			"scene-volume-world-element-attachment")
		return true;
	if(!gScene || !gSceneCpuVolume ||
		!gSceneCpuWorldAttachment)
		return gMetrics.sceneWorldPinReleased != 0;

	const PxU32 churnFrame = PxMax<PxU32>(
		1, gHeadlessOptions.frames / 3);
	const PxU32 releaseFrame = PxMax<PxU32>(
		churnFrame + 1, 2 * gHeadlessOptions.frames / 3);
	if(gMetrics.completedFrames == churnFrame)
	{
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		gMetrics.sceneWorldPinActorReadded = 1;
	}
	if(gMetrics.completedFrames == releaseFrame)
	{
		gSceneCpuWorldAttachment->release();
		gSceneCpuWorldAttachment = NULL;
		gMetrics.sceneWorldPinReleased = 1;
	}
	return true;
}

static bool updateSceneVolumeRigidAttachment()
{
	const bool staticAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-static-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-static-element-attachment";
	const bool kinematicAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-element-attachment";
	const bool articulationAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-element-attachment";
	if(gHeadlessOptions.caseName !=
			"scene-volume-rigid-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-rigid-element-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-static-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-static-element-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-kinematic-element-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-articulation-element-attachment" &&
		!kinematicAttachmentCase &&
		!articulationAttachmentCase)
		return true;
	if(!gScene || !gSceneCpuVolume ||
		(!gSceneCpuAttachmentBody &&
		 !gSceneCpuStaticActor))
		return false;

	const PxU32 churnFrame = PxMax<PxU32>(
		1, gHeadlessOptions.frames / 3);
	const PxU32 releaseFrame = PxMax<PxU32>(
		churnFrame + 1, 2 * gHeadlessOptions.frames / 3);
	if(gMetrics.completedFrames == churnFrame)
	{
		if(!gSceneCpuRigidAttachment ||
			gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
	}
	if(gMetrics.completedFrames == releaseFrame)
	{
		if(!gSceneCpuRigidAttachment)
			return false;
		gSceneCpuRigidAttachment->release();
		gSceneCpuRigidAttachment = NULL;
		gMetrics.sceneRigidAttachmentReleased = 1;
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxU32 vertexCount =
			gSceneCpuVolume->getSimulationMesh()->
				getNbVertices();
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] =
				PxVec4(1.0f, 0.0f, 0.0f, velocities[i].w);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		if(!kinematicAttachmentCase &&
			!staticAttachmentCase &&
			!articulationAttachmentCase)
			gSceneCpuDynamicActor->setLinearVelocity(
				PxVec3(-1.0f, 0.0f, 0.0f));
	}
	else if((kinematicAttachmentCase || staticAttachmentCase) &&
		gMetrics.sceneSoftFirstSlept &&
		gSceneCpuRigidAttachment)
	{
		if(!gMetrics.sceneKinematicTargetIssued)
		{
			gSceneCpuKinematicAttachmentSoftBaseline =
				getSceneCpuAttachmentPoint();
			gSceneCpuKinematicAttachmentCommand =
				PxTransform(
					gSceneCpuRigidAttachmentInitialPosition);
			gMetrics.sceneKinematicTargetIssued = 1;
		}
		gSceneCpuKinematicAttachmentProgress = PxMin(
			gSceneCpuKinematicAttachmentProgress + 0.01f,
			1.0f);
		gSceneCpuKinematicAttachmentCommand =
			PxTransform(
				gSceneCpuRigidAttachmentInitialPosition +
					PxVec3(
						gSceneCpuKinematicAttachmentProgress,
						0.0f, 0.0f),
				PxQuat(
					0.5f *
						gSceneCpuKinematicAttachmentProgress,
					PxVec3(0.0f, 0.0f, 1.0f)));
		if(staticAttachmentCase)
			gSceneCpuStaticActor->setGlobalPose(
				gSceneCpuKinematicAttachmentCommand);
		else
			gSceneCpuDynamicActor->setKinematicTarget(
				gSceneCpuKinematicAttachmentCommand);
	}
	return true;
}

static bool updateSceneVolumeSoftPairAttachment()
{
	if(gHeadlessOptions.caseName !=
		"scene-volume-volume-attachment")
		return true;
	if(!gScene || !gSceneCpuVolume ||
		!gSceneCpuSecondVolume)
		return false;

	const PxU32 churnFrame = PxMax<PxU32>(
		1, gHeadlessOptions.frames / 3);
	const PxU32 releaseFrame = PxMax<PxU32>(
		churnFrame + 1, 2 * gHeadlessOptions.frames / 3);
	if(gMetrics.completedFrames == churnFrame)
	{
		if(!gSceneCpuRigidAttachment ||
			gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
	}
	if(gMetrics.completedFrames == releaseFrame)
	{
		if(!gSceneCpuRigidAttachment)
			return false;
		gSceneCpuRigidAttachment->release();
		gSceneCpuRigidAttachment = NULL;
		gMetrics.sceneRigidAttachmentReleased = 1;
		if(!setSceneCpuVolumeVelocity(
				*gSceneCpuVolume,
				PxVec3(-0.5f, 0.0f, 0.0f)) ||
			!setSceneCpuVolumeVelocity(
				*gSceneCpuSecondVolume,
				PxVec3(0.5f, 0.0f, 0.0f)))
			return false;
	}
	return true;
}

static bool updateSceneVolumeElementFilter()
{
	const bool elementFilterCase =
		gHeadlessOptions.caseName ==
			"scene-volume-element-filter" ||
		gHeadlessOptions.caseName ==
			"scene-volume-partial-element-filter";
	if(!elementFilterCase)
		return true;
	if(!gScene || !gSceneCpuVolume)
		return false;

	const PxU32 churnFrame = PxMax<PxU32>(
		1, 2 * gHeadlessOptions.frames / 5);
	const PxU32 releaseFrame = PxMin<PxU32>(
		gHeadlessOptions.frames - 1,
		PxMax<PxU32>(churnFrame + 1,
			gHeadlessOptions.frames / 2));
	if(gMetrics.completedFrames == churnFrame)
	{
		if(!gSceneCpuElementFilter ||
			gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxU32 vertexCount =
			gSceneCpuVolume->getSimulationMesh()->
				getNbVertices();
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] =
				PxVec4(
					0.0f,
					gHeadlessOptions.caseName ==
						"scene-volume-partial-element-filter"
						? 0.0f : -2.0f,
					0.0f, velocities[i].w);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		gMetrics.sceneElementFilterActorReadded = 1;
	}
	if(gMetrics.completedFrames == releaseFrame)
	{
		if(!gSceneCpuElementFilter)
			return false;
		PxVec4* positions =
			gSceneCpuVolume->getSimPositionInvMassBufferH();
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		PxVec4* collisionPositions =
			gSceneCpuVolume->getPositionInvMassBufferH();
		const PxTetrahedronMesh* simulationMesh =
			gSceneCpuVolume->getSimulationMesh();
		const PxVec3* vertices = simulationMesh->getVertices();
		const PxU32 vertexCount = simulationMesh->getNbVertices();
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			positions[i] = PxVec4(
				vertices[i] + PxVec3(0.0f, 4.0f, 0.0f),
				positions[i].w);
			velocities[i] =
				PxVec4(0.0f, 0.0f, 0.0f, velocities[i].w);
		}
		PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
			*gSceneCpuVolume, positions, collisionPositions);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eALL);
		gSceneCpuElementFilter->release();
		gSceneCpuElementFilter = NULL;
		gMetrics.sceneElementFilterReleased = 1;
	}
	return true;
}

static bool updateSceneVolumeKinematicTargets()
{
	const bool fullTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-full-kinematic-target";
	const bool partialTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-partial-kinematic-target";
	if(!fullTargetCase && !partialTargetCase)
		return true;
	if(!gSceneCpuVolume ||
		gSceneCpuVolumeKinematicTargets.empty() ||
		gSceneCpuVolumeKinematicTargets.size() !=
			gSceneCpuVolumeKinematicInitial.size())
		return false;

	const PxU32 driveStart = PxMax<PxU32>(
		1, gHeadlessOptions.frames / 5);
	const PxU32 driveEnd = PxMax<PxU32>(
		driveStart + 1, gHeadlessOptions.frames / 2);
	if(gMetrics.completedFrames >= driveStart)
	{
		const PxReal progress = PxClamp(
			PxReal(gMetrics.completedFrames - driveStart) /
				PxReal(driveEnd - driveStart),
			0.0f, 1.0f);
		const PxVec3 translation(
			0.5f * progress, 0.25f * progress, 0.0f);
		for(PxU32 i = 0;
			i < gSceneCpuVolumeKinematicTargets.size(); ++i)
		{
			if(partialTargetCase &&
				i == gSceneCpuVolumePartialProbe)
				continue;
			gSceneCpuVolumeKinematicTargets[i] =
				PxVec4(
					gSceneCpuVolumeKinematicInitial[i] +
						translation,
					0.0f);
		}
		gMetrics.sceneVolumeTargetMutated = 1;
	}

	const PxU32 activationFrame = PxMax<PxU32>(
		driveEnd + 1, 2 * gHeadlessOptions.frames / 3);
	if(partialTargetCase &&
		!gMetrics.sceneVolumePartialActivated &&
		gMetrics.completedFrames >= activationFrame)
	{
		if(gSceneCpuVolumePartialProbe >=
				gSceneCpuVolumeKinematicTargets.size())
			return false;
		const PxVec4* positions =
			gSceneCpuVolume->getSimPositionInvMassBufferH();
		if(!positions)
			return false;
		const PxU32 probe = gSceneCpuVolumePartialProbe;
		const PxVec3 oldDecoy =
			gSceneCpuVolumeKinematicTargets[probe].getXYZ();
		const PxVec3 current = positions[probe].getXYZ();
		gMetrics.sceneVolumePartialInactiveDecoyDistance =
			(oldDecoy - current).magnitude();
		if(gMetrics.
				sceneVolumePartialInactiveDecoyDistance > 2.0f)
			gMetrics.sceneVolumePartialInactiveIgnored = 1;
		gSceneCpuVolumePartialActivationStart = current;
		gSceneCpuVolumeKinematicTargets[probe] =
			PxVec4(current, 0.0f);
		gMetrics.sceneVolumePartialActivated = 1;
	}
	if(partialTargetCase &&
		gMetrics.sceneVolumePartialActivated)
	{
		const PxU32 probe = gSceneCpuVolumePartialProbe;
		if(probe >= gSceneCpuVolumeKinematicTargets.size())
			return false;
		const PxU32 activationDuration = PxMax<PxU32>(
			1, gHeadlessOptions.frames / 10);
		const PxReal activationProgress = PxClamp(
			PxReal(gMetrics.completedFrames - activationFrame) /
				PxReal(activationDuration),
			0.0f, 1.0f);
		gSceneCpuVolumeKinematicTargets[probe] =
			PxVec4(
				gSceneCpuVolumePartialActivationStart +
					PxVec3(
						// The probe is the interior vertex of the
						// five-vertex fixture. A 0.25 m x displacement
						// lands it on the outer tetrahedron face and
						// correctly triggers the positive-J limiter.
						// Keep this behavioral gate strictly inside
						// the non-singular domain.
						0.1f * activationProgress,
						0.0f, 0.0f),
				0.0f);
	}
	return true;
}

static bool sampleSceneVolumeKinematicTargets()
{
	const bool fullTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-full-kinematic-target";
	const bool partialTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-partial-kinematic-target";
	if(!fullTargetCase && !partialTargetCase)
		return true;
	if(!gSceneCpuVolume ||
		gSceneCpuVolumeKinematicTargets.empty() ||
		gSceneCpuVolumeKinematicTargets.size() !=
			gSceneCpuVolumeKinematicInitial.size())
		return false;
	const PxVec4* positions =
		gSceneCpuVolume->getSimPositionInvMassBufferH();
	if(!positions)
		return false;

	PxVec3 initialCentroid(0.0f);
	PxVec3 currentCentroid(0.0f);
	PxReal maxActiveError = 0.0f;
	for(PxU32 i = 0;
		i < gSceneCpuVolumeKinematicTargets.size(); ++i)
	{
		initialCentroid += gSceneCpuVolumeKinematicInitial[i];
		currentCentroid += positions[i].getXYZ();
		const bool active =
			fullTargetCase ||
			gSceneCpuVolumeKinematicTargets[i].w == 0.0f;
		if(active)
			maxActiveError = PxMax(
				maxActiveError,
				(positions[i].getXYZ() -
					gSceneCpuVolumeKinematicTargets[i].
						getXYZ()).magnitude());
	}
	const PxReal invCount = 1.0f /
		PxReal(gSceneCpuVolumeKinematicTargets.size());
	initialCentroid *= invCount;
	currentCentroid *= invCount;
	gMetrics.sceneVolumeTargetMaxDisplacement = PxMax(
		gMetrics.sceneVolumeTargetMaxDisplacement,
		(currentCentroid - initialCentroid).magnitude());
	gMetrics.sceneVolumeTargetFinalMaxError =
		maxActiveError;

	if(gMetrics.sceneVolumeTargetMutated &&
		!gSceneCpuVolume->isSleeping())
		gMetrics.sceneVolumeTargetWoke = 1;
	const PxU32 driveEnd = PxMax<PxU32>(
		PxMax<PxU32>(1, gHeadlessOptions.frames / 5) + 1,
		gHeadlessOptions.frames / 2);
	if(gMetrics.completedFrames > driveEnd &&
		maxActiveError <= 5.0e-3f)
		gMetrics.sceneVolumeTargetReached = 1;
	if(partialTargetCase &&
		gMetrics.sceneVolumePartialActivated &&
		gSceneCpuVolumePartialProbe <
			gSceneCpuVolumeKinematicTargets.size())
	{
		const PxU32 probe = gSceneCpuVolumePartialProbe;
		const PxReal probeError =
			(positions[probe].getXYZ() -
				gSceneCpuVolumeKinematicTargets[probe].
					getXYZ()).magnitude();
		if(probeError <= 5.0e-3f)
			gMetrics.sceneVolumePartialActivatedReached = 1;
	}
	return true;
}

static bool updateSceneKinematicBox()
{
	if(!isSceneCpuVolumeKinematicRigidCase(
		gHeadlessOptions.caseName))
		return true;
	if(!gSceneCpuVolume || !gSceneCpuDynamicActor ||
		gSceneCpuDynamicActor->getScene() != gScene)
		return false;
	if(!gMetrics.sceneSoftFirstSlept)
		return true;

	if(!gMetrics.sceneKinematicTargetIssued)
	{
		gSceneCpuKinematicSoftBaselineY =
			getSceneCpuVolumeCentroidY();
		gMetrics.sceneKinematicTargetIssued = 1;
	}

	const PxReal kinematicStep =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-convex"
		? 0.0025f : 0.005f;
	const PxReal nextY = PxMin(
		gSceneCpuKinematicCommandY + kinematicStep, 4.10f);
	if(nextY > gSceneCpuKinematicCommandY)
	{
		gSceneCpuKinematicCommandY = nextY;
		gSceneCpuDynamicActor->setKinematicTarget(
			PxTransform(
				gHeadlessOptions.caseName ==
					"scene-volume-kinematic-heightfield"
					? PxVec3(-4.0f, nextY, -4.0f)
					: PxVec3(0.0f, nextY, 0.0f)));
	}
	return true;
}

static bool stepPhysicsInternal(PxReal dt)
{
	if(gHeadlessOptions.caseName ==
		"scene-volume-multi-scene-isolation")
		return stepSceneMultiSceneIsolation(dt);

	const bool profileFrame =
		gMetrics.completedFrames >= gProfileWarmupFrames;
	PxTime frameTimer;
	if(isSceneCpuVolumeCase(gHeadlessOptions.caseName))
	{
		if(!updateSceneStaticChurn() ||
			!updateSceneDynamicChurn() ||
			!updateSceneMultiDynamicGate() ||
			!updateSceneSoftSleepWake() ||
			!updateSceneSoftRigidWake() ||
			!updateSceneSoftActorChurn() ||
			!updateSceneSoftBufferMutation() ||
			!updateSceneSoftSoftWake() ||
			!updateSceneVolumeWorldPin() ||
			!updateSceneVolumeRigidAttachment() ||
			!updateSceneVolumeSoftPairAttachment() ||
			!updateSceneVolumeElementFilter() ||
			!updateSceneVolumeKinematicTargets() ||
			!updateSceneKinematicBox())
			return false;
		PxTime sceneTimer;
		gScene->simulate(dt);
		if(!gScene->fetchResults(true))
		{
			gMetrics.fetchFailures++;
			return false;
		}
		const PxF64 sceneMs =
			sceneTimer.getElapsedSeconds() * 1000.0;
		gMetrics.completedFrames++;
		if(!sampleSceneVolumeKinematicTargets())
			return false;
		if(!updateVolumeSkinning())
			return false;

		PxVec4* positions =
			gSceneCpuVolume->getSimPositionInvMassBufferH();
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxTetrahedronMesh* sceneSimulationMesh =
			gSceneCpuVolume->getSimulationMesh();
		const PxU32 vertexCount =
			sceneSimulationMesh->getNbVertices();
		const PxVec3* simulationRestVertices =
			sceneSimulationMesh->getVertices();
		PxU32 nearGroundParticles = 0;
		PxU32 nearRigidParticles = 0;
		PxReal frameMinY = FLT_MAX;
		PxReal frameDynamicSurfaceSeparation = FLT_MAX;
		const bool multiDynamicBoxCase =
			gHeadlessOptions.caseName ==
				"scene-volume-multi-dynamic-box";
		const bool multiSoftIslandCase =
			gHeadlessOptions.caseName ==
				"scene-volume-multi-soft-islands";
		const bool mixedSleepIslandCase =
			gHeadlessOptions.caseName ==
				"scene-volume-mixed-sleep-islands";
		const bool softChurnCase =
			gHeadlessOptions.caseName ==
				"scene-volume-soft-churn";
		const bool bufferMutationCase =
			gHeadlessOptions.caseName ==
				"scene-volume-buffer-mutation";
		const bool partialElementFilterCase =
			gHeadlessOptions.caseName ==
				"scene-volume-partial-element-filter";
		const bool elementFilterCase =
			gHeadlessOptions.caseName ==
				"scene-volume-element-filter" ||
			partialElementFilterCase;
		const bool rigidAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-rigid-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-rigid-element-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-element-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-element-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-articulation-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-articulation-element-attachment";
		const bool staticAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-static-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-element-attachment";
		const bool kinematicAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-element-attachment";
		const bool articulationAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-articulation-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-articulation-element-attachment";
		const bool softSoftWakeCase =
			gHeadlessOptions.caseName ==
				"scene-volume-soft-soft-wake";
		const bool softPairAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-volume-attachment";
		const bool motionControlsCase =
			gHeadlessOptions.caseName ==
				"scene-volume-motion-controls";
		const bool maxDepenetrationVelocityCase =
			gHeadlessOptions.caseName ==
				"scene-volume-max-depenetration-velocity";
		const bool triangleSurfaceSweptCcdCase =
			isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool triangleSurfaceReverseSweptCcdCase =
			isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool staticTriangleSurfaceSweptCcdCase =
			isSceneCpuVolumeStaticTriangleSurfaceSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool heightFieldSweptCcdCase =
			isSceneCpuVolumeHeightFieldSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalTriangleSurfaceSweptCcdCase =
			isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool sphereReverseSweptCcdCase =
			isSceneCpuVolumeSphereReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool deformingReverseSweptCcdCase =
			isSceneCpuVolumeDeformingReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool capsuleReverseSweptCcdCase =
			isSceneCpuVolumeCapsuleReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool convexReverseSweptCcdCase =
			isSceneCpuVolumeConvexReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalCapsuleReverseSweptCcdCase =
			isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalConvexReverseSweptCcdCase =
			isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalFiniteReverseSweptCcdCase =
			rotationalCapsuleReverseSweptCcdCase ||
			rotationalConvexReverseSweptCcdCase;
		const bool staticSphereReverseSweptCcdCase =
			deformingReverseSweptCcdCase ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-sphere-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-convex-reverse-swept-ccd";
		const bool kinematicSphereReverseSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-convex-reverse-swept-ccd";
		const bool dynamicSphereReverseSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-sphere-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-rotating-convex-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-convex-reverse-swept-ccd";
		const bool speculativeCcdCase =
			isSceneCpuVolumeSpeculativeCcdCase(
				gHeadlessOptions.caseName) ||
			sphereReverseSweptCcdCase;
		const bool planeSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-plane-speculative-ccd";
		const bool sphereSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-sphere-speculative-ccd";
		const bool capsuleSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-capsule-speculative-ccd";
		const bool convexSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-convex-speculative-ccd";
		const bool finiteSmoothSpeculativeCcdCase =
			sphereSpeculativeCcdCase || capsuleSpeculativeCcdCase ||
			convexSpeculativeCcdCase;
		const bool rotatingKinematicCapsuleSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-rotating-kinematic-capsule-speculative-ccd";
		const bool rotatingKinematicConvexSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-rotating-kinematic-convex-speculative-ccd";
		const bool movingKinematicCapsuleSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-moving-kinematic-capsule-speculative-ccd" ||
			rotatingKinematicCapsuleSpeculativeCcdCase;
		const bool movingKinematicConvexSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-moving-kinematic-convex-speculative-ccd" ||
			rotatingKinematicConvexSpeculativeCcdCase;
		const bool movingKinematicFiniteSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-moving-kinematic-sphere-speculative-ccd" ||
			movingKinematicCapsuleSpeculativeCcdCase ||
			movingKinematicConvexSpeculativeCcdCase;
		const bool dynamicRotatingCapsuleSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-rotating-capsule-relative-swept-ccd";
		const bool dynamicRotatingConvexSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-rotating-convex-relative-swept-ccd";
		const bool dynamicCapsuleRelativeSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule-relative-swept-ccd" ||
			dynamicRotatingCapsuleSpeculativeCcdCase;
		const bool dynamicConvexRelativeSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-convex-relative-swept-ccd" ||
			dynamicRotatingConvexSpeculativeCcdCase;
		const bool dynamicFiniteRelativeSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-sphere-relative-swept-ccd" ||
			dynamicCapsuleRelativeSweptCcdCase ||
			dynamicConvexRelativeSweptCcdCase;
		const bool sphereReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-sphere-reverse-feature";
		const bool capsuleReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-capsule-reverse-feature";
		const bool convexReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-convex-reverse-feature";
		const bool triangleMeshReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-triangle-mesh-reverse-feature";
		const bool heightFieldReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-heightfield-reverse-feature";
		const bool smoothReverseFeatureCase =
			sphereReverseFeatureCase ||
			capsuleReverseFeatureCase ||
			convexReverseFeatureCase ||
			triangleMeshReverseFeatureCase ||
			heightFieldReverseFeatureCase;
		const bool twoSoftVolumeCase =
			multiSoftIslandCase || mixedSleepIslandCase ||
			softChurnCase || softSoftWakeCase ||
			softPairAttachmentCase || motionControlsCase ||
			maxDepenetrationVelocityCase ||
			speculativeCcdCase || smoothReverseFeatureCase;
		const bool softSleepWakeCase =
			gHeadlessOptions.caseName ==
				"scene-volume-sleep-wake";
		const bool softRigidWakeCase =
			gHeadlessOptions.caseName ==
				"scene-volume-rigid-wake";
		const bool kinematicBoxCase =
			isSceneCpuVolumeKinematicRigidCase(
				gHeadlessOptions.caseName);
		const bool kinematicSphereCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-sphere";
		const bool kinematicCapsuleCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-capsule";
		const bool kinematicConvexCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-convex";
		const bool kinematicTriangleMeshCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-triangle-mesh";
		const bool kinematicHeightFieldCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-heightfield";
		const bool twoDynamicActorsCase =
			multiDynamicBoxCase || multiSoftIslandCase ||
			movingKinematicFiniteSpeculativeCcdCase ||
			dynamicFiniteRelativeSweptCcdCase ||
			kinematicSphereReverseSweptCcdCase ||
			dynamicSphereReverseSweptCcdCase ||
			(triangleSurfaceSweptCcdCase &&
			 !staticTriangleSurfaceSweptCcdCase);
		const bool dynamicBoxCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-box" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-sphere" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-convex" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-churn" ||
			twoDynamicActorsCase;
		const bool dynamicSphereCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-sphere" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule";
		const bool dynamicCapsuleCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule";
		const bool dynamicConvexCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-convex";
		const bool dynamicActorInScene =
			(dynamicBoxCase || softRigidWakeCase) &&
			gSceneCpuDynamicActor &&
			gSceneCpuDynamicActor->getScene() == gScene;
		const bool secondDynamicActorInScene =
			twoDynamicActorsCase && gSceneCpuSecondDynamicActor &&
			gSceneCpuSecondDynamicActor->getScene() == gScene;
		if(gHeadlessOptions.caseName == "scene-volume-world-pin" ||
			gHeadlessOptions.caseName ==
				"scene-volume-world-element-attachment")
		{
			const PxReal displacement =
				(getSceneCpuAttachmentPoint() -
					gSceneCpuWorldPinTarget).magnitude();
			if(!gMetrics.sceneWorldPinReleased)
			{
				gMetrics.sceneWorldPinMaxDrift = PxMax(
					gMetrics.sceneWorldPinMaxDrift,
					displacement);
				gMetrics.sceneWorldPinHeld =
					gMetrics.sceneWorldPinMaxDrift <= 1.0e-4f
						? 1u : 0u;
			}
			else
			{
				gMetrics.sceneWorldPinReleasedMaxDisplacement =
					PxMax(
						gMetrics.
							sceneWorldPinReleasedMaxDisplacement,
						displacement);
				if(gMetrics.
					sceneWorldPinReleasedMaxDisplacement >
					1.0e-3f)
					gMetrics.sceneWorldPinMovedAfterRelease = 1;
			}
		}
		if(softPairAttachmentCase && gSceneCpuSecondVolume)
		{
			const PxReal separation =
				(getSceneCpuAttachmentPoint() -
					getSceneCpuAttachmentPoint(
						gSceneCpuSecondVolume)).magnitude();
			const PxReal targetDisplacement = PxAbs(
				getSceneCpuVolumeCentroidY(
					gSceneCpuSecondVolume) -
				gSceneCpuSecondVolumeInitialCentroidY);
			const PxVec4* targetVelocities =
				gSceneCpuSecondVolume->
					getSimVelocityBufferH();
			const PxU32 targetVertexCount =
				gSceneCpuSecondVolume->getSimulationMesh()->
					getNbVertices();
			PxReal targetMaxSpeed = 0.0f;
			if(!targetVelocities)
				return false;
			for(PxU32 i = 0; i < targetVertexCount; ++i)
				targetMaxSpeed = PxMax(
					targetMaxSpeed,
					targetVelocities[i].getXYZ().magnitude());
			if(!PxIsFinite(separation) ||
				!PxIsFinite(targetDisplacement) ||
				!PxIsFinite(targetMaxSpeed))
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.sceneRigidAttachmentMaxRigidSpeed = PxMax(
					gMetrics.sceneRigidAttachmentMaxRigidSpeed,
					targetMaxSpeed);
				gMetrics.
					sceneRigidAttachmentMaxRigidDisplacement = PxMax(
						gMetrics.
							sceneRigidAttachmentMaxRigidDisplacement,
						targetDisplacement);
				if(targetDisplacement > 0.02f)
					gMetrics.sceneRigidAttachmentRigidMoved = 1;
				if(!gSceneCpuSecondVolume->isSleeping())
					gMetrics.sceneRigidAttachmentRigidWoke = 1;
				const PxU32 churnFrame = PxMax<PxU32>(
					1, gHeadlessOptions.frames / 3);
				if(!gMetrics.sceneRigidAttachmentReleased &&
					gMetrics.completedFrames >= churnFrame)
				{
					gMetrics.sceneRigidAttachmentMaxDrift = PxMax(
						gMetrics.sceneRigidAttachmentMaxDrift,
						separation);
					if(gMetrics.completedFrames > churnFrame &&
						separation < 0.05f)
						gMetrics.
							sceneRigidAttachmentHeldAcrossReadd = 1;
				}
				else if(gMetrics.sceneRigidAttachmentReleased)
				{
					gMetrics.
						sceneRigidAttachmentReleasedSeparation = PxMax(
							gMetrics.
								sceneRigidAttachmentReleasedSeparation,
							separation);
					if(separation > 0.2f)
						gMetrics.
							sceneRigidAttachmentSeparatedAfterRelease =
								1;
				}
			}
		}
		if(motionControlsCase && gSceneCpuSecondVolume)
		{
			const PxVec3 controlCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxReal controlSpeed =
				getSceneCpuVolumeMaxSpeed(gSceneCpuVolume);
			const PxReal settlingSpeed =
				getSceneCpuVolumeMaxSpeed(gSceneCpuSecondVolume);
			if(!controlCentroid.isFinite() ||
				!PxIsFinite(controlSpeed) ||
				!PxIsFinite(settlingSpeed))
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.motionControlFinalSpeed = controlSpeed;
				gMetrics.motionSettlingFinalSpeed = settlingSpeed;
				if(gMetrics.completedFrames == 1)
				{
					gMetrics.
						motionMaxVelocityFirstStepDisplacement =
						(controlCentroid -
							gSceneCpuMotionInitialCentroid).
								magnitude();
					gMetrics.motionMaxVelocityFirstStepSpeed =
						controlSpeed;
					gMetrics.motionMaxVelocityBounded =
						gMetrics.
							motionMaxVelocityFirstStepDisplacement <=
								dt * 1.01f &&
						// maxLinearVelocity owns free preintegration;
						// Position-AL may add a small post-solve
						// constraint velocity.
						controlSpeed <= 1.02f ? 1u : 0u;
					if(settlingSpeed <= 0.07f)
						gMetrics.motionSettlingApplied = 1;
					if(!setSceneCpuVolumeVelocity(
							*gSceneCpuVolume,
							PxVec3(0.08f, 0.0f, 0.0f)))
						return false;
					gSceneCpuVolume->setMaxLinearVelocity(
						PX_MAX_F32);
					gSceneCpuVolume->setSettlingThreshold(0.1f);
					gSceneCpuVolume->setSettlingDamping(0.0f);
					gSceneCpuVolume->setSleepThreshold(0.05f);
					gSceneCpuVolume->setWakeCounter(0.2f);
				}
				else if(gMetrics.completedFrames > 1)
				{
					if(gSceneCpuVolume->isSleeping())
						gMetrics.motionControlStayedAwake = 0;
				}
				if(gSceneCpuSecondVolume->isSleeping())
					gMetrics.motionSettlingSlept = 1;
			}
		}
		if(maxDepenetrationVelocityCase &&
			gSceneCpuSecondVolume)
		{
			const PxReal limitedMinY =
				getSceneCpuVolumeMinY(gSceneCpuVolume);
			const PxReal controlMinY =
				getSceneCpuVolumeMinY(gSceneCpuSecondVolume);
			const PxReal limitedRise =
				limitedMinY -
				gSceneCpuDepenetrationInitialMinY;
			const PxReal controlRise =
				controlMinY -
				gSceneCpuDepenetrationControlInitialMinY;
			const PxReal limitedSpeed =
				getSceneCpuVolumeMaxSpeed(gSceneCpuVolume);
			if(!PxIsFinite(limitedRise) ||
				!PxIsFinite(controlRise) ||
				!PxIsFinite(limitedSpeed))
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.depenetrationLimitedMaxSpeed = PxMax(
					gMetrics.depenetrationLimitedMaxSpeed,
					limitedSpeed);
				if(gMetrics.completedFrames == 1)
				{
					gMetrics.
						depenetrationLimitedFirstStepRise =
							limitedRise;
					gMetrics.
						depenetrationControlFirstStepRise =
							controlRise;
					gMetrics.depenetrationFirstStepBounded =
						limitedRise >= -1.0e-6f &&
						limitedRise <= 0.12f * dt * 1.25f &&
						limitedSpeed <= 0.25f
							? 1u : 0u;
					gMetrics.depenetrationControlSeparated =
						controlRise >
							limitedRise + 5.0e-3f
								? 1u : 0u;
				}
				gMetrics.depenetrationLimitedFinalRise =
					limitedRise;
				if(limitedRise >
					gMetrics.
						depenetrationLimitedFirstStepRise +
							4.0e-3f)
					gMetrics.depenetrationGradualRecovery = 1;
			}
		}
		if(smoothReverseFeatureCase &&
			gSceneCpuSecondVolume &&
			gSceneCpuStaticActor &&
			gMetrics.completedFrames == 1)
		{
			const PxVec3 rigidCenter =
				heightFieldReverseFeatureCase
					? PxVec3(-2.05f, 0.0f, -0.05f)
					: PxVec3(-1.75f, 0.0f, 0.25f);
			const PxReal rigidRadius = 0.3f;
			const PxReal capsuleHalfHeight =
				capsuleReverseFeatureCase ? 0.15f : 0.0f;
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuSecondVolume);
			PxReal faceSeparation = PX_MAX_F32;
			PxReal minimumVertexSeparation = PX_MAX_F32;
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite() &&
				getSceneCpuVolumeSmoothReverseSeparations(
					gSceneCpuVolume, rigidCenter, rigidRadius,
					capsuleHalfHeight,
					convexReverseFeatureCase,
					triangleMeshReverseFeatureCase,
					heightFieldReverseFeatureCase,
					faceSeparation,
					minimumVertexSeparation);
			if(!finite)
				gSphereReverseFeatureMetrics.nonFiniteSamples++;
			else
			{
				gSphereReverseFeatureMetrics.
					positiveDisplacement =
						(positiveCentroid -
							gSceneCpuSphereReversePositiveInitial).
								magnitude();
				gSphereReverseFeatureMetrics.positiveDrop =
					gSceneCpuSphereReversePositiveInitial.y -
						positiveCentroid.y;
				gSphereReverseFeatureMetrics.negativeDrop =
					gSceneCpuSphereReverseNegativeInitial.y -
						negativeCentroid.y;
				gSphereReverseFeatureMetrics.faceSeparation =
					faceSeparation;
				gSphereReverseFeatureMetrics.
					minimumVertexSeparation =
						minimumVertexSeparation;
				gSphereReverseFeatureMetrics.vertexSdfExcluded =
					minimumVertexSeparation > 0.10f ? 1u : 0u;
				gSphereReverseFeatureMetrics.
					negativeControlPassed =
						gSphereReverseFeatureMetrics.
							negativeDrop > 0.02f ? 1u : 0u;
				gSphereReverseFeatureMetrics.
					faceResponseObserved =
						gSphereReverseFeatureMetrics.
							positiveDisplacement > 1.0e-3f &&
						faceSeparation > 0.02f &&
						gSphereReverseFeatureMetrics.
								positiveDrop + 0.01f <
							gSphereReverseFeatureMetrics.
								negativeDrop
							? 1u : 0u;
			}

			if(gSceneCpuVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuVolume);
				gMetrics.sceneActorRemoved =
					gSceneCpuVolume->getScene() == NULL ? 1u : 0u;
			}
			if(gSceneCpuSecondVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuSecondVolume);
				gMetrics.sceneSecondVolumeActorRemoved =
					gSceneCpuSecondVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuStaticActor->getScene() == gScene)
				gScene->removeActor(*gSceneCpuStaticActor);
		}
		else if(triangleSurfaceSweptCcdCase &&
			gSceneCpuSecondVolume &&
			(staticTriangleSurfaceSweptCcdCase
				? gSceneCpuStaticActor != NULL
				: (gSceneCpuDynamicActor &&
				   gSceneCpuSecondDynamicActor)) &&
			gMetrics.completedFrames == 1)
		{
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(
					gSceneCpuSecondVolume);
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite();
			if(!finite)
			{
				gSphereReverseSweptMetrics.nonFiniteSamples++;
				if(rotationalTriangleSurfaceSweptCcdCase)
					gCapsuleRotationalSweepMetrics.
						nonFiniteSamples++;
			}
			else
			{
				gSphereReverseSweptMetrics.
					positiveDisplacement =
						(positiveCentroid -
							gSceneCpuMovingSpherePositiveInitial).
								magnitude();
				gSphereReverseSweptMetrics.
					negativeDisplacement =
						(negativeCentroid -
							gSceneCpuMovingSphereNegativeInitial).
								magnitude();
				gSphereReverseSweptMetrics.positiveDrop =
					gSceneCpuMovingSpherePositiveInitial.y -
						positiveCentroid.y;
				gSphereReverseSweptMetrics.negativeDrop =
					gSceneCpuMovingSphereNegativeInitial.y -
						negativeCentroid.y;
				gSphereReverseSweptMetrics.positiveRigidDrop =
					0.0f;
				gSphereReverseSweptMetrics.negativeRigidDrop =
					0.0f;
				gSphereReverseSweptMetrics.faceSeparation =
					0.0f;
				gSphereReverseSweptMetrics.
					minimumVertexSweepSeparation =
						PX_MAX_F32;
				if(rotationalTriangleSurfaceSweptCcdCase)
				{
					const PxQuat startRotation(
						0.5235987756f,
						PxVec3(0.0f, 0.0f, 1.0f));
					const PxQuat endRotation(
						2.6179938780f,
						PxVec3(0.0f, 0.0f, 1.0f));
					const PxTransform positivePose =
						gSceneCpuDynamicActor->getGlobalPose();
					const PxTransform negativePose =
						gSceneCpuSecondDynamicActor->
							getGlobalPose();
					const PxTransform startPose(
						positivePose.p, startRotation);
					const PxTransform endPose(
						positivePose.p, endRotation);
					const bool geometryFinite =
						getSceneCpuVolumeRotationalTriangleSurfaceSweepSeparations(
							startPose, endPose,
							heightFieldSweptCcdCase,
							triangleSurfaceReverseSweptCcdCase,
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation,
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation,
							gSphereReverseSweptMetrics.
								minimumVertexSweepSeparation);
					auto getAngularTravel = [&](
						const PxQuat& end)
					{
						const PxReal alignment = PxClamp(
							PxAbs(
								startRotation.dot(
									end.getNormalized())),
							0.0f, 1.0f);
						return 2.0f * PxAcos(alignment);
					};
					gCapsuleRotationalSweepMetrics.
						positiveAngularTravel =
							getAngularTravel(positivePose.q);
					gCapsuleRotationalSweepMetrics.
						negativeAngularTravel =
							getAngularTravel(negativePose.q);
					if(!geometryFinite)
						gCapsuleRotationalSweepMetrics.
							nonFiniteSamples++;
					gCapsuleRotationalSweepMetrics.
						sweepIsolated =
							geometryFinite &&
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation > 0.10f &&
							(triangleSurfaceReverseSweptCcdCase
								? gCapsuleRotationalSweepMetrics.
										midSweepMinSeparation <
									-0.05f
								: gCapsuleRotationalSweepMetrics.
										midSweepMinSeparation <
									0.01f)
								? 1u : 0u;
				}
				else if(triangleSurfaceReverseSweptCcdCase)
				{
					const PxReal startY =
						staticTriangleSurfaceSweptCcdCase
							? 0.0f : -1.1f;
					const PxVec3 rigidCenter =
						heightFieldSweptCcdCase
							? PxVec3(
								-1.93f, startY, 0.03f)
							: PxVec3(
								-1.63f, startY, 0.33f);
					const PxVec3 rigidTipStart =
						rigidCenter +
							(heightFieldSweptCcdCase
								? PxVec3(0.3f, 0.3f, 0.3f)
								: PxVec3(0.0f, 0.3f, 0.0f));
					const PxVec3 rigidTipSegment(
						0.0f, 2.2f, 0.0f);
					const PxReal denominator =
						rigidTipSegment.magnitudeSquared();
					for(PxU32 vertexIndex = 0;
						vertexIndex <
							gSceneCpuSphereReverseSweptInitialPositions.
								size();
						++vertexIndex)
					{
						const PxVec3 softVertex =
							gSceneCpuSphereReverseSweptInitialPositions[
								vertexIndex];
						const PxReal time = PxClamp(
							(softVertex - rigidTipStart).
								dot(rigidTipSegment) /
								denominator,
							0.0f, 1.0f);
						gSphereReverseSweptMetrics.
							minimumVertexSweepSeparation =
								PxMin(
									gSphereReverseSweptMetrics.
										minimumVertexSweepSeparation,
									(softVertex -
										(rigidTipStart +
											rigidTipSegment * time)).
										magnitude());
					}
				}
				gSphereReverseSweptMetrics.vertexSweepExcluded =
					!triangleSurfaceReverseSweptCcdCase ||
					gSphereReverseSweptMetrics.
							minimumVertexSweepSeparation > 0.05f
						? 1u : 0u;
				const PxReal staticControlSeparation =
					triangleSurfaceReverseSweptCcdCase
						? 0.01f : 0.10f;
				const PxReal kinematicResponseThreshold =
					rotationalTriangleSurfaceSweptCcdCase
						? 2.0e-3f
						: triangleSurfaceReverseSweptCcdCase
							? 5.0e-3f : 0.02f;
				gSphereReverseSweptMetrics.responseObserved =
					staticTriangleSurfaceSweptCcdCase
						? gSphereReverseSweptMetrics.positiveDrop +
								staticControlSeparation <
							gSphereReverseSweptMetrics.negativeDrop
						: gSphereReverseSweptMetrics.
								positiveDisplacement >
									kinematicResponseThreshold &&
							gSphereReverseSweptMetrics.
								negativeDisplacement < 1.0e-2f
							? 1u : 0u;
				gSphereReverseSweptMetrics.negativeControlPassed =
					staticTriangleSurfaceSweptCcdCase
						? gSphereReverseSweptMetrics.negativeDrop >
							(triangleSurfaceReverseSweptCcdCase &&
							 heightFieldSweptCcdCase
								? 0.8f : 1.5f)
						: gSphereReverseSweptMetrics.
							negativeDisplacement < 1.0e-2f
							? 1u : 0u;
				gSphereReverseSweptMetrics.
					twoSidedResponseObserved = 1;
				gMetrics.speculativeCcdPreventedTunneling =
					gSphereReverseSweptMetrics.responseObserved;
				gMetrics.speculativeCcdNegativeControlTunneled =
					gSphereReverseSweptMetrics.
						negativeControlPassed;
			}

			if(gSceneCpuVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuVolume);
				gMetrics.sceneActorRemoved =
					gSceneCpuVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuSecondVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuSecondVolume);
				gMetrics.sceneSecondVolumeActorRemoved =
					gSceneCpuSecondVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(staticTriangleSurfaceSweptCcdCase)
			{
				if(gSceneCpuStaticActor->getScene() == gScene)
					gScene->removeActor(*gSceneCpuStaticActor);
			}
			else
			{
				if(gSceneCpuDynamicActor->getScene() == gScene)
				{
					gScene->removeActor(*gSceneCpuDynamicActor);
					gMetrics.sceneDynamicActorRemoved =
						gSceneCpuDynamicActor->getScene() == NULL
							? 1u : 0u;
				}
				if(gSceneCpuSecondDynamicActor->getScene() == gScene)
				{
					gScene->removeActor(
						*gSceneCpuSecondDynamicActor);
					gMetrics.sceneSecondDynamicActorRemoved =
						gSceneCpuSecondDynamicActor->getScene() ==
							NULL ? 1u : 0u;
				}
			}
		}
		else if(sphereReverseSweptCcdCase &&
			gSceneCpuSecondVolume &&
			(staticSphereReverseSweptCcdCase ||
				(gSceneCpuDynamicActor &&
				 gSceneCpuSecondDynamicActor)) &&
			gMetrics.completedFrames == 1)
		{
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuSecondVolume);
			const PxVec3 positiveSphereStart(
				deformingReverseSweptCcdCase
					? -1.68f
				: rotationalFiniteReverseSweptCcdCase
					? -1.45f : -1.35f,
				rotationalFiniteReverseSweptCcdCase
					? (rotationalConvexReverseSweptCcdCase
						? -0.85f : -1.0f) :
					(staticSphereReverseSweptCcdCase ? 0.0f : 1.1f),
				deformingReverseSweptCcdCase
					? 0.32f
				: rotationalFiniteReverseSweptCcdCase
					? 0.25f : 0.15f);
			const PxVec3 negativeSphereStart(
				deformingReverseSweptCcdCase
					? 1.32f
				: rotationalFiniteReverseSweptCcdCase
					? 1.55f : 1.65f,
				rotationalFiniteReverseSweptCcdCase
					? (rotationalConvexReverseSweptCcdCase
						? -0.85f : -1.0f) :
					(staticSphereReverseSweptCcdCase ? 0.0f : 1.1f),
				deformingReverseSweptCcdCase
					? 0.32f
				: rotationalFiniteReverseSweptCcdCase
					? 0.25f : 0.15f);
			const PxTransform positiveRigidPose =
				staticSphereReverseSweptCcdCase
					? PxTransform(positiveSphereStart)
					: gSceneCpuDynamicActor->getGlobalPose();
			const PxTransform negativeRigidPose =
				staticSphereReverseSweptCcdCase
					? PxTransform(negativeSphereStart)
					: gSceneCpuSecondDynamicActor->getGlobalPose();
			const PxVec3 positiveSphereCurrent =
				positiveRigidPose.p;
			const PxVec3 negativeSphereCurrent =
				negativeRigidPose.p;
			const PxVec3 sweepEnd =
				deformingReverseSweptCcdCase
					? positiveSphereStart
				: staticSphereReverseSweptCcdCase
					? positiveSphereStart +
						PxVec3(
							0.0f,
							convexReverseSweptCcdCase
								? 2.2f : 1.1f,
							0.0f)
					: positiveSphereCurrent;
			PxReal faceSeparation = PX_MAX_F32;
			PxReal minimumVertexSweepSeparation = PX_MAX_F32;
			if(rotationalFiniteReverseSweptCcdCase)
			{
				const PxQuat startRotation(
					PxPi / 6.0f,
					PxVec3(0.0f, 0.0f, 1.0f));
				auto getAngularTravel = [&](
					const PxQuat& endRotation)
				{
					const PxReal alignment = PxClamp(
						PxAbs(
							startRotation.dot(
								endRotation.getNormalized())),
						0.0f, 1.0f);
					return 2.0f * PxAcos(alignment);
				};
				gCapsuleRotationalSweepMetrics.
					positiveAngularTravel =
						getAngularTravel(positiveRigidPose.q);
				gCapsuleRotationalSweepMetrics.
					negativeAngularTravel =
						getAngularTravel(negativeRigidPose.q);
			}
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite() &&
				positiveSphereCurrent.isFinite() &&
				negativeSphereCurrent.isFinite() &&
				(deformingReverseSweptCcdCase
					? getSceneCpuVolumeDeformingReverseSweptProof(
						gSceneCpuVolume,
						positiveRigidPose,
						convexReverseSweptCcdCase
							? 0.25f : 0.18f,
						capsuleReverseSweptCcdCase
							? 0.02f : 0.0f,
						convexReverseSweptCcdCase,
						gDeformingVolumeReverseSweptMetrics.
							endpointMinSeparation,
						gDeformingVolumeReverseSweptMetrics.
							midSweepMinSeparation,
						minimumVertexSweepSeparation)
				: rotationalFiniteReverseSweptCcdCase
					? (rotationalConvexReverseSweptCcdCase
						? getSceneCpuVolumeRotationalConvexReverseSweptSeparations(
							gSceneCpuVolume,
							positiveRigidPose,
							PxTransform(
								positiveSphereStart,
								PxQuat(
									PxPi / 6.0f,
									PxVec3(0.0f, 0.0f, 1.0f))),
							PxTransform(
								negativeRigidPose.p -
									(negativeSphereStart -
									 positiveSphereStart),
								negativeRigidPose.q),
							faceSeparation,
							minimumVertexSweepSeparation,
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation,
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation)
						: getSceneCpuVolumeRotationalCapsuleReverseSweptSeparations(
						gSceneCpuVolume,
						positiveRigidPose,
						PxTransform(
							positiveSphereStart,
							PxQuat(
								PxPi / 6.0f,
								PxVec3(0.0f, 0.0f, 1.0f))),
						PxTransform(
							negativeRigidPose.p -
								(negativeSphereStart -
								 positiveSphereStart),
							negativeRigidPose.q),
						0.1f, 1.0f,
						faceSeparation,
						minimumVertexSweepSeparation,
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation,
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation))
					: getSceneCpuVolumeSphereReverseSweptSeparations(
						gSceneCpuVolume,
						positiveSphereCurrent,
						positiveSphereStart,
						sweepEnd,
						0.25f,
						capsuleReverseSweptCcdCase ? 0.02f : 0.0f,
						convexReverseSweptCcdCase,
						faceSeparation,
						minimumVertexSweepSeparation));
			if(!finite)
				gSphereReverseSweptMetrics.nonFiniteSamples++;
			else
			{
				gSphereReverseSweptMetrics.positiveDisplacement =
					(positiveCentroid -
						gSceneCpuMovingSpherePositiveInitial).
							magnitude();
				gSphereReverseSweptMetrics.negativeDisplacement =
					(negativeCentroid -
						gSceneCpuMovingSphereNegativeInitial).
							magnitude();
				gSphereReverseSweptMetrics.positiveDrop =
					gSceneCpuMovingSpherePositiveInitial.y -
						positiveCentroid.y;
				gSphereReverseSweptMetrics.negativeDrop =
					gSceneCpuMovingSphereNegativeInitial.y -
						negativeCentroid.y;
				gSphereReverseSweptMetrics.positiveRigidDrop =
					staticSphereReverseSweptCcdCase
						? 0.0f
					: rotationalFiniteReverseSweptCcdCase
						? gCapsuleRotationalSweepMetrics.
							positiveAngularTravel
						: gSceneCpuDynamicInitialY -
							positiveSphereCurrent.y;
				gSphereReverseSweptMetrics.negativeRigidDrop =
					staticSphereReverseSweptCcdCase
						? 0.0f
					: rotationalFiniteReverseSweptCcdCase
						? gCapsuleRotationalSweepMetrics.
							negativeAngularTravel
						: gSceneCpuSecondDynamicInitialY -
							negativeSphereCurrent.y;
				if(deformingReverseSweptCcdCase)
				{
					const PxVec4* positivePositions =
						gSceneCpuVolume->
							getPositionInvMassBufferH();
					const PxVec4* negativePositions =
						gSceneCpuSecondVolume->
							getPositionInvMassBufferH();
					const PxU32 positiveVertexCount =
						gSceneCpuVolume->getCollisionMesh()->
							getNbVertices();
					const PxU32 negativeVertexCount =
						gSceneCpuSecondVolume->getCollisionMesh()->
							getNbVertices();
					if(!positivePositions || !negativePositions ||
						positiveVertexCount != negativeVertexCount)
					{
						gSphereReverseSweptMetrics.
							nonFiniteSamples++;
					}
					else
					{
						for(PxU32 vertexIndex = 0;
							vertexIndex < positiveVertexCount;
							++vertexIndex)
						{
							const PxVec3 positiveLocal =
								positivePositions[vertexIndex].
									getXYZ() -
								PxVec3(-2.0f, 0.0f, 0.0f);
							const PxVec3 negativeLocal =
								negativePositions[vertexIndex].
									getXYZ() -
								PxVec3(1.0f, 0.0f, 0.0f);
							gDeformingVolumeReverseSweptMetrics.
								responseDelta = PxMax(
									gDeformingVolumeReverseSweptMetrics.
										responseDelta,
									(positiveLocal - negativeLocal).
										magnitude());
						}
					}
				}
				gSphereReverseSweptMetrics.faceSeparation =
					deformingReverseSweptCcdCase
						? gDeformingVolumeReverseSweptMetrics.
							endpointMinSeparation
						: faceSeparation;
				gSphereReverseSweptMetrics.
					minimumVertexSweepSeparation =
						minimumVertexSweepSeparation;
				gSphereReverseSweptMetrics.vertexSweepExcluded =
					minimumVertexSweepSeparation >
						(deformingReverseSweptCcdCase
							? 0.05f
						: rotationalFiniteReverseSweptCcdCase
							? 0.10f :
						((capsuleReverseSweptCcdCase ||
						  convexReverseSweptCcdCase)
							? 0.05f : 0.10f))
						? 1u : 0u;
				if(deformingReverseSweptCcdCase)
					gDeformingVolumeReverseSweptMetrics.
						geometricSweepIsolated =
							gDeformingVolumeReverseSweptMetrics.
								endpointMinSeparation > 0.02f &&
							gDeformingVolumeReverseSweptMetrics.
								midSweepMinSeparation <
									(convexReverseSweptCcdCase
										? 0.02f : 0.0f) &&
							minimumVertexSweepSeparation > 0.05f
								? 1u : 0u;
				if(rotationalFiniteReverseSweptCcdCase)
					gCapsuleRotationalSweepMetrics.sweepIsolated =
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation > 0.05f &&
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation < -0.05f &&
						minimumVertexSweepSeparation > 0.10f
							? 1u : 0u;
				gSphereReverseSweptMetrics.responseObserved =
					(deformingReverseSweptCcdCase
						? gDeformingVolumeReverseSweptMetrics.
								responseDelta > 0.01f &&
							gSphereReverseSweptMetrics.
								positiveDrop + 0.01f <
							gSphereReverseSweptMetrics.
								negativeDrop
					: staticSphereReverseSweptCcdCase
						? gSphereReverseSweptMetrics.positiveDrop +
								0.03f <
							gSphereReverseSweptMetrics.negativeDrop
						: gSphereReverseSweptMetrics.
								positiveDisplacement >
									(rotationalFiniteReverseSweptCcdCase
										? 0.02f :
									 dynamicSphereReverseSweptCcdCase
										? 0.01f : 0.02f) &&
							gSphereReverseSweptMetrics.
								negativeDisplacement < 5.0e-3f) &&
					(deformingReverseSweptCcdCase ||
						faceSeparation > -0.15f) ? 1u : 0u;
				gSphereReverseSweptMetrics.negativeControlPassed =
					(deformingReverseSweptCcdCase
						? gSphereReverseSweptMetrics.
								negativeDrop > 0.15f
					: staticSphereReverseSweptCcdCase
						? gSphereReverseSweptMetrics.negativeDrop >
							0.8f
						: gSphereReverseSweptMetrics.
							negativeDisplacement < 5.0e-3f) &&
					(!dynamicSphereReverseSweptCcdCase ||
						gSphereReverseSweptMetrics.
							negativeRigidDrop >
								(rotationalFiniteReverseSweptCcdCase
									? 0.8f : 1.5f))
						? 1u : 0u;
				gSphereReverseSweptMetrics.
					twoSidedResponseObserved =
						!dynamicSphereReverseSweptCcdCase ||
						gSphereReverseSweptMetrics.
								positiveRigidDrop + 0.05f <
							gSphereReverseSweptMetrics.
								negativeRigidDrop
							? 1u : 0u;
				gMetrics.speculativeCcdPreventedTunneling =
					gSphereReverseSweptMetrics.responseObserved;
				gMetrics.speculativeCcdNegativeControlTunneled =
					gSphereReverseSweptMetrics.
						negativeControlPassed;
			}

			if(gSceneCpuVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuVolume);
				gMetrics.sceneActorRemoved =
					gSceneCpuVolume->getScene() == NULL ? 1u : 0u;
			}
			if(gSceneCpuSecondVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuSecondVolume);
				gMetrics.sceneSecondVolumeActorRemoved =
					gSceneCpuSecondVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuStaticActor &&
				gSceneCpuStaticActor->getScene() == gScene)
				gScene->removeActor(*gSceneCpuStaticActor);
			if(gSceneCpuDynamicActor &&
				gSceneCpuDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuDynamicActor);
				gMetrics.sceneDynamicActorRemoved =
					gSceneCpuDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuSecondDynamicActor &&
				gSceneCpuSecondDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(
					*gSceneCpuSecondDynamicActor);
				gMetrics.sceneSecondDynamicActorRemoved =
					gSceneCpuSecondDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
		}
		else if(dynamicFiniteRelativeSweptCcdCase &&
			gSceneCpuSecondVolume &&
			gSceneCpuDynamicActor &&
			gSceneCpuSecondDynamicActor &&
			gMetrics.completedFrames == 1)
		{
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuSecondVolume);
			const PxTransform positiveRigidPose =
				gSceneCpuDynamicActor->getGlobalPose();
			const PxTransform negativeRigidPose =
				gSceneCpuSecondDynamicActor->getGlobalPose();
			const PxVec3 positiveRigidCenter =
				positiveRigidPose.p;
			const PxVec3 negativeRigidCenter =
				negativeRigidPose.p;
			if(dynamicRotatingCapsuleSpeculativeCcdCase ||
				dynamicRotatingConvexSpeculativeCcdCase)
			{
				const PxQuat startRotation(
					-PxPi / 6.0f,
					PxVec3(0.0f, 0.0f, 1.0f));
				auto getAngularTravel = [&](
					const PxQuat& endRotation)
				{
					const PxReal alignment = PxClamp(
						PxAbs(
							startRotation.dot(
								endRotation.getNormalized())),
						0.0f, 1.0f);
					return 2.0f * PxAcos(alignment);
				};
				gCapsuleRotationalSweepMetrics.
					positiveAngularTravel =
						getAngularTravel(positiveRigidPose.q);
				gCapsuleRotationalSweepMetrics.
					negativeAngularTravel =
						getAngularTravel(negativeRigidPose.q);
				const PxTransform startPose(
					PxVec3(-2.9f, 0.0f, 0.0f),
					startRotation);
				const PxTransform endPose(
					startPose.p,
					negativeRigidPose.q.getNormalized());
				bool rotationalMetricsFinite =
					dynamicRotatingConvexSpeculativeCcdCase
						? getSceneCpuVolumeRotationalConvexPointSweepSeparations(
							startPose, endPose,
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation,
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation)
						: !gSceneCpuCapsuleRotationalSweptInitialPositions.
							empty();
				if(!dynamicRotatingConvexSpeculativeCcdCase)
				{
					for(PxU32 i = 0;
						i <
							gSceneCpuCapsuleRotationalSweptInitialPositions.
								size();
						++i)
					{
						const PxVec3 point =
							gSceneCpuCapsuleRotationalSweptInitialPositions[
								i];
						if(!point.isFinite())
						{
							rotationalMetricsFinite = false;
							break;
						}
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation = PxMin(
								gCapsuleRotationalSweepMetrics.
									endpointMinSeparation,
								PxMin(
									getCapsuleSignedSeparation(
										point, startPose, 0.1f, 1.0f),
									getCapsuleSignedSeparation(
										point, endPose, 0.1f, 1.0f)));
						for(PxU32 sample = 1; sample < 64; ++sample)
						{
							const PxReal time =
								PxReal(sample) / 64.0f;
							const PxTransform samplePose(
								startPose.p,
								PxSlerp(
									time, startRotation,
									negativeRigidPose.q.getNormalized()).
										getNormalized());
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation = PxMin(
									gCapsuleRotationalSweepMetrics.
										midSweepMinSeparation,
									getCapsuleSignedSeparation(
										point, samplePose,
										0.1f, 1.0f));
						}
					}
				}
				rotationalMetricsFinite =
					rotationalMetricsFinite &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation) &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation) &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							positiveAngularTravel) &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							negativeAngularTravel);
				if(!rotationalMetricsFinite)
				{
					gCapsuleRotationalSweepMetrics.nonFiniteSamples++;
					gMetrics.nonFiniteParticleSamples++;
				}
				else
				{
					gCapsuleRotationalSweepMetrics.sweepIsolated =
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation > 0.05f &&
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation <
								(dynamicRotatingConvexSpeculativeCcdCase
									? 1.0e-5f : -0.05f)
							? 1u : 0u;
				}
			}
			PxReal positiveMinSeparation = PX_MAX_F32;
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite() &&
				positiveRigidCenter.isFinite() &&
				negativeRigidCenter.isFinite() &&
				(dynamicConvexRelativeSweptCcdCase
					? getSceneCpuVolumeSingleConvexMinSeparation(
						gSceneCpuVolume,
						positiveRigidPose,
						positiveMinSeparation)
					: dynamicCapsuleRelativeSweptCcdCase
					? getSceneCpuVolumeSingleCapsuleMinSeparation(
						gSceneCpuVolume,
						positiveRigidPose,
						dynamicRotatingCapsuleSpeculativeCcdCase
							? 0.1f : 0.8f,
						dynamicRotatingCapsuleSpeculativeCcdCase
							? 1.0f : 0.3f,
						positiveMinSeparation)
					: getSceneCpuVolumeSingleSphereMinSeparation(
						gSceneCpuVolume,
						positiveRigidCenter,
						0.8f,
						positiveMinSeparation));
			if(!finite)
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.
					dynamicSphereSweepPositiveSoftDisplacement =
						(positiveCentroid -
							gSceneCpuMovingSpherePositiveInitial).
								magnitude();
				gMetrics.
					dynamicSphereSweepNegativeSoftDisplacement =
						(negativeCentroid -
							gSceneCpuMovingSphereNegativeInitial).
								magnitude();
				gMetrics.dynamicSphereSweepPositiveRigidDrop =
					(dynamicRotatingCapsuleSpeculativeCcdCase ||
					 dynamicRotatingConvexSpeculativeCcdCase)
						? gCapsuleRotationalSweepMetrics.
							positiveAngularTravel
						: gSceneCpuDynamicInitialY -
							positiveRigidCenter.y;
				gMetrics.dynamicSphereSweepNegativeRigidDrop =
					(dynamicRotatingCapsuleSpeculativeCcdCase ||
					 dynamicRotatingConvexSpeculativeCcdCase)
						? gCapsuleRotationalSweepMetrics.
							negativeAngularTravel
						: gSceneCpuSecondDynamicInitialY -
							negativeRigidCenter.y;
				gMetrics.
					dynamicSphereSweepPositiveMinSeparation =
						positiveMinSeparation;
				gMetrics.dynamicSphereSweepResponseObserved =
					gMetrics.
						dynamicSphereSweepPositiveSoftDisplacement >
							0.02f &&
					gMetrics.
						dynamicSphereSweepPositiveMinSeparation >
							-0.15f ? 1u : 0u;
				gMetrics.
					dynamicSphereSweepNegativeControlTunneled =
						gMetrics.
							dynamicSphereSweepNegativeSoftDisplacement <
								5.0e-3f &&
						gMetrics.
							dynamicSphereSweepNegativeRigidDrop >
								((dynamicRotatingCapsuleSpeculativeCcdCase ||
								  dynamicRotatingConvexSpeculativeCcdCase)
									? 0.8f : 1.5f)
							? 1u : 0u;
				gMetrics.
					dynamicSphereSweepTwoSidedResponseObserved =
						gMetrics.
							dynamicSphereSweepPositiveRigidDrop +
								0.05f <
						gMetrics.
							dynamicSphereSweepNegativeRigidDrop
							? 1u : 0u;
				gMetrics.speculativeCcdPreventedTunneling =
					gMetrics.dynamicSphereSweepResponseObserved;
				gMetrics.speculativeCcdNegativeControlTunneled =
					gMetrics.
						dynamicSphereSweepNegativeControlTunneled;
			}

			if(gSceneCpuVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuVolume);
				gMetrics.sceneActorRemoved =
					gSceneCpuVolume->getScene() == NULL ? 1u : 0u;
			}
			if(gSceneCpuSecondVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuSecondVolume);
				gMetrics.sceneSecondVolumeActorRemoved =
					gSceneCpuSecondVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuDynamicActor);
				gMetrics.sceneDynamicActorRemoved =
					gSceneCpuDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuSecondDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(
					*gSceneCpuSecondDynamicActor);
				gMetrics.sceneSecondDynamicActorRemoved =
					gSceneCpuSecondDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
		}
		else if(movingKinematicFiniteSpeculativeCcdCase &&
			gSceneCpuSecondVolume &&
			gSceneCpuDynamicActor)
		{
			if((rotatingKinematicCapsuleSpeculativeCcdCase ||
				rotatingKinematicConvexSpeculativeCcdCase) &&
				gMetrics.completedFrames == 1)
			{
				const PxVec3 rigidCenter(-2.9f, 0.0f, 0.0f);
				const PxTransform startPose(
					rigidCenter,
					PxQuat(
						-PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f)));
				const PxTransform endPose(
					rigidCenter,
					PxQuat(
						PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f)));
				const PxTransform midPose(rigidCenter);
				bool rotationalMetricsFinite =
					rotatingKinematicConvexSpeculativeCcdCase
						? getSceneCpuVolumeRotationalConvexPointSweepSeparations(
							startPose, endPose,
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation,
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation)
						: !gSceneCpuCapsuleRotationalSweptInitialPositions.
							empty();
				if(!rotatingKinematicConvexSpeculativeCcdCase)
				{
					for(PxU32 i = 0;
						i <
							gSceneCpuCapsuleRotationalSweptInitialPositions.
								size();
						++i)
					{
						const PxVec3 point =
							gSceneCpuCapsuleRotationalSweptInitialPositions[
								i];
						if(!point.isFinite())
						{
							rotationalMetricsFinite = false;
							break;
						}
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation = PxMin(
								gCapsuleRotationalSweepMetrics.
									endpointMinSeparation,
								PxMin(
									getCapsuleSignedSeparation(
										point, startPose, 0.1f, 1.0f),
									getCapsuleSignedSeparation(
										point, endPose, 0.1f, 1.0f)));
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation = PxMin(
								gCapsuleRotationalSweepMetrics.
									midSweepMinSeparation,
								getCapsuleSignedSeparation(
									point, midPose, 0.1f, 1.0f));
					}
				}
				rotationalMetricsFinite =
					rotationalMetricsFinite &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation) &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation);
				if(!rotationalMetricsFinite)
				{
					gCapsuleRotationalSweepMetrics.nonFiniteSamples++;
					gMetrics.nonFiniteParticleSamples++;
				}
				else
				{
					gCapsuleRotationalSweepMetrics.sweepIsolated =
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation > 0.05f &&
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation <
								(rotatingKinematicConvexSpeculativeCcdCase
									? 1.0e-5f : -0.05f)
							? 1u : 0u;
				}
			}
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(
					gSceneCpuSecondVolume);
			PxReal positiveMinSeparation = PX_MAX_F32;
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite() &&
				(movingKinematicConvexSpeculativeCcdCase
					? getSceneCpuVolumeSingleConvexMinSeparation(
						gSceneCpuVolume,
						gSceneCpuDynamicActor->getGlobalPose(),
						positiveMinSeparation)
					: movingKinematicCapsuleSpeculativeCcdCase
					? getSceneCpuVolumeSingleCapsuleMinSeparation(
						gSceneCpuVolume,
						gSceneCpuDynamicActor->getGlobalPose(),
						rotatingKinematicCapsuleSpeculativeCcdCase
							? 0.1f : 0.8f,
						rotatingKinematicCapsuleSpeculativeCcdCase
							? 1.0f : 0.3f,
						positiveMinSeparation)
					: getSceneCpuVolumeSingleSphereMinSeparation(
						gSceneCpuVolume,
						gSceneCpuDynamicActor->getGlobalPose().p,
						0.8f, positiveMinSeparation));
			if(!finite)
				gMetrics.nonFiniteParticleSamples++;
			else if(gMetrics.completedFrames <= 3)
			{
				gMetrics.movingSpherePositiveDisplacement =
					PxMax(
						gMetrics.movingSpherePositiveDisplacement,
						(positiveCentroid -
							gSceneCpuMovingSpherePositiveInitial).
								magnitude());
				gMetrics.movingSphereNegativeDisplacement =
					PxMax(
						gMetrics.movingSphereNegativeDisplacement,
						(negativeCentroid -
							gSceneCpuMovingSphereNegativeInitial).
								magnitude());
				gMetrics.movingSpherePositiveMinSeparation =
					PxMin(
						gMetrics.movingSpherePositiveMinSeparation,
						positiveMinSeparation);
				if(gMetrics.completedFrames == 1)
				{
					gMetrics.movingSphereNegativeControlHeld =
						gMetrics.
							movingSphereNegativeDisplacement <
								5.0e-3f ? 1u : 0u;
					if(gSceneCpuSecondVolume->getScene() == gScene)
					{
						gScene->removeActor(
							*gSceneCpuSecondVolume);
						gMetrics.
							sceneSecondVolumeActorRemoved =
								gSceneCpuSecondVolume->
									getScene() == NULL ? 1u : 0u;
					}
					if(gSceneCpuSecondDynamicActor &&
						gSceneCpuSecondDynamicActor->getScene() ==
							gScene)
					{
						gScene->removeActor(
							*gSceneCpuSecondDynamicActor);
						gMetrics.
							sceneSecondDynamicActorRemoved =
								gSceneCpuSecondDynamicActor->
									getScene() == NULL ? 1u : 0u;
					}
				}
				if(gMetrics.completedFrames == 3)
				{
					gMetrics.movingSphereCcdResponseObserved =
						gMetrics.
							movingSpherePositiveDisplacement >
								0.02f &&
						gMetrics.
							movingSpherePositiveMinSeparation >
								-0.10f ? 1u : 0u;
					gMetrics.speculativeCcdPreventedTunneling =
						gMetrics.
							movingSphereCcdResponseObserved;
					gMetrics.
						speculativeCcdNegativeControlTunneled =
							gMetrics.
								movingSphereNegativeControlHeld;
					if(gSceneCpuVolume->getScene() == gScene)
					{
						gScene->removeActor(*gSceneCpuVolume);
						gMetrics.sceneActorRemoved =
							gSceneCpuVolume->getScene() == NULL
								? 1u : 0u;
					}
					if(gSceneCpuSecondVolume->getScene() == gScene)
					{
						gScene->removeActor(
							*gSceneCpuSecondVolume);
						gMetrics.
							sceneSecondVolumeActorRemoved =
								gSceneCpuSecondVolume->
									getScene() == NULL ? 1u : 0u;
					}
				}
			}
		}
		else if(speculativeCcdCase &&
			!sphereReverseSweptCcdCase &&
			!triangleSurfaceSweptCcdCase &&
			!dynamicFiniteRelativeSweptCcdCase &&
			gSceneCpuSecondVolume)
		{
			const PxReal positiveMinY =
				getSceneCpuVolumeCollisionMinY(gSceneCpuVolume);
			const PxReal negativeMaxY =
				getSceneCpuVolumeCollisionMaxY(
					gSceneCpuSecondVolume);
			PxReal positiveMinSeparation = PX_MAX_F32;
			const PxVec3 convexCenters[] =
			{
				PxVec3(-1.5f, 0.25f, 0.5f),
				PxVec3(1.5f, 0.25f, 0.5f)
			};
			const bool hasPositiveMinSeparation =
				!finiteSmoothSpeculativeCcdCase ||
				(convexSpeculativeCcdCase
					? getSceneCpuVolumeConvexMinSeparation(
						gSceneCpuVolume,
						convexCenters, 2,
						positiveMinSeparation)
					: capsuleSpeculativeCcdCase
					? getSceneCpuVolumeCapsuleClusterMinSeparation(
						gSceneCpuVolume, positiveMinSeparation)
					: getSceneCpuVolumeSphereMinSeparation(
						gSceneCpuVolume, positiveMinSeparation));
			if(!PxIsFinite(positiveMinY) ||
				!PxIsFinite(negativeMaxY) ||
				!hasPositiveMinSeparation)
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.speculativeCcdPositiveMinY =
					PxMin(
						gMetrics.speculativeCcdPositiveMinY,
						positiveMinY);
				if(finiteSmoothSpeculativeCcdCase)
					gMetrics.
						speculativeCcdPositiveMinSeparation =
							PxMin(
								gMetrics.
									speculativeCcdPositiveMinSeparation,
								positiveMinSeparation);
				gMetrics.speculativeCcdPreventedTunneling =
					(finiteSmoothSpeculativeCcdCase
						? gMetrics.
							speculativeCcdPositiveMinSeparation >=
								-0.05f
						: gMetrics.speculativeCcdPositiveMinY >=
							(planeSpeculativeCcdCase
								? 0.49f : 0.50f))
						? 1u : 0u;
				if(gMetrics.completedFrames <= 3)
				{
					if((!planeSpeculativeCcdCase &&
							!finiteSmoothSpeculativeCcdCase) ||
						convexSpeculativeCcdCase ||
						gMetrics.completedFrames == 1)
						gMetrics.speculativeCcdNegativeMaxY =
							negativeMaxY;
					const bool finalizeNegativeControl =
						convexSpeculativeCcdCase
							? gMetrics.completedFrames == 3
							: finiteSmoothSpeculativeCcdCase
							? gMetrics.completedFrames == 1
							: gMetrics.completedFrames == 3;
					if(finalizeNegativeControl)
					{
						gMetrics.
							speculativeCcdNegativeControlTunneled =
								gMetrics.
									speculativeCcdNegativeMaxY <=
									(planeSpeculativeCcdCase
										? 0.45f : 0.44f)
									? 1u : 0u;
						if(gSceneCpuSecondVolume->getScene() == gScene)
						{
							gScene->removeActor(
								*gSceneCpuSecondVolume);
							gMetrics.
								sceneSecondVolumeActorRemoved =
								gSceneCpuSecondVolume->
									getScene() == NULL ? 1u : 0u;
						}
					}
				}
			}
		}
		if(rigidAttachmentCase &&
			(gSceneCpuAttachmentBody || gSceneCpuStaticActor))
		{
			const PxTransform rigidPose =
				staticAttachmentCase
					? gSceneCpuStaticActor->getGlobalPose()
					: gSceneCpuAttachmentBody->getGlobalPose();
			const PxReal rigidSpeed =
				staticAttachmentCase
					? 0.0f
					: gSceneCpuAttachmentBody->getLinearVelocity().
						magnitude();
			const PxReal separation =
				(getSceneCpuAttachmentPoint() -
					rigidPose.transform(
						gSceneCpuRigidAttachmentLocalOffset)).
					magnitude();
			if(!rigidPose.isValid() ||
				!PxIsFinite(rigidSpeed) ||
				!PxIsFinite(separation))
			{
				gMetrics.nonFiniteParticleSamples++;
			}
			else
			{
				gMetrics.sceneRigidAttachmentMaxRigidSpeed =
					PxMax(
						gMetrics.sceneRigidAttachmentMaxRigidSpeed,
						rigidSpeed);
				if(!gMetrics.sceneRigidAttachmentReleased)
				{
					gMetrics.
						sceneRigidAttachmentMaxRigidDisplacement =
						PxMax(
							gMetrics.
								sceneRigidAttachmentMaxRigidDisplacement,
							(rigidPose.p -
								gSceneCpuRigidAttachmentInitialPosition).
								magnitude());
					if(gMetrics.
						sceneRigidAttachmentMaxRigidDisplacement >
							0.02f)
						gMetrics.
							sceneRigidAttachmentRigidMoved = 1;
					gMetrics.sceneRigidAttachmentMaxDrift =
						PxMax(
							gMetrics.
								sceneRigidAttachmentMaxDrift,
							separation);
					const PxU32 churnFrame = PxMax<PxU32>(
						1, gHeadlessOptions.frames / 3);
					if(gMetrics.completedFrames > churnFrame &&
						separation < 0.05f)
						gMetrics.
							sceneRigidAttachmentHeldAcrossReadd =
								1;
				}
				else
				{
					gMetrics.
						sceneRigidAttachmentReleasedSeparation =
						PxMax(
							gMetrics.
								sceneRigidAttachmentReleasedSeparation,
							separation);
					if(gMetrics.
						sceneRigidAttachmentReleasedSeparation >
							0.2f)
						gMetrics.
							sceneRigidAttachmentSeparatedAfterRelease =
								1;
				}
			}
			if(articulationAttachmentCase)
			{
				const PxTransform rootPose =
					gSceneCpuAttachmentRoot->getGlobalPose();
				const PxVec3 childDelta =
					rigidPose.p -
						gSceneCpuArticulationChildInitialPose.p;
				const PxReal rootDisplacement =
					(rootPose.p -
						gSceneCpuArticulationRootInitialPose.p).
							magnitude();
				const PxReal forbiddenDisplacement =
					PxSqrt(
						childDelta.y * childDelta.y +
						childDelta.z * childDelta.z);
				const PxQuat childOrientationError =
					rigidPose.q *
						gSceneCpuArticulationChildInitialPose.q.
							getConjugate();
				const PxReal childAngularDisplacement =
					2.0f * childOrientationError.
						getImaginaryPart().magnitude();
				if(!rootPose.isValid() ||
					!PxIsFinite(rootDisplacement) ||
					!PxIsFinite(forbiddenDisplacement) ||
					!PxIsFinite(childAngularDisplacement))
				{
					gMetrics.nonFiniteParticleSamples++;
				}
				else
				{
					gMetrics.sceneArticulationRootMaxDisplacement =
						PxMax(
							gMetrics.
								sceneArticulationRootMaxDisplacement,
							rootDisplacement);
					gMetrics.
						sceneArticulationChildMaxForbiddenDisplacement =
						PxMax(
							gMetrics.
								sceneArticulationChildMaxForbiddenDisplacement,
							forbiddenDisplacement);
					gMetrics.
						sceneArticulationChildMaxAngularDisplacement =
						PxMax(
							gMetrics.
								sceneArticulationChildMaxAngularDisplacement,
							childAngularDisplacement);
					if(!gSceneCpuAttachmentArticulation->isSleeping())
						gMetrics.sceneArticulationWoke = 1;
					if(rootDisplacement <= 1.0e-4f)
						gMetrics.sceneArticulationRootStable = 1;
					if(forbiddenDisplacement <= 1.0e-3f &&
						childAngularDisplacement <= 1.0e-3f)
						gMetrics.
							sceneArticulationJointSubspaceHeld = 1;
				}
			}
			else if(kinematicAttachmentCase ||
				staticAttachmentCase)
			{
				if(gMetrics.sceneKinematicTargetIssued)
				{
					if(!gSceneCpuVolume->isSleeping())
						gMetrics.sceneKinematicSoftWoke = 1;
					const PxReal softDisplacement =
						(getSceneCpuAttachmentPoint() -
							gSceneCpuKinematicAttachmentSoftBaseline).
								magnitude();
					gMetrics.sceneKinematicSoftDisplacement =
						PxMax(
							gMetrics.
								sceneKinematicSoftDisplacement,
							softDisplacement);
					if(softDisplacement > 0.02f)
						gMetrics.sceneKinematicSoftMoved = 1;
					const PxQuat orientationError =
						rigidPose.q *
							gSceneCpuKinematicAttachmentCommand.q.
								getConjugate();
					const PxReal poseError =
						(rigidPose.p -
							gSceneCpuKinematicAttachmentCommand.p).
								magnitude() +
						2.0f * orientationError.
							getImaginaryPart().magnitude();
					gMetrics.sceneKinematicMaxPoseError =
						PxMax(
							gMetrics.sceneKinematicMaxPoseError,
							poseError);
					if(gSceneCpuKinematicAttachmentProgress >= 1.0f &&
						poseError <= 1.0e-4f)
						gMetrics.sceneKinematicTargetReached = 1;
				}
			}
			else if(!staticAttachmentCase &&
				!gSceneCpuDynamicActor->isSleeping())
				gMetrics.sceneRigidAttachmentRigidWoke = 1;
		}
		if(softSleepWakeCase || softRigidWakeCase ||
			kinematicBoxCase ||
			bufferMutationCase || softSoftWakeCase ||
			kinematicAttachmentCase || staticAttachmentCase)
		{
			const bool sleeping = gSceneCpuVolume->isSleeping();
			const PxReal maxSpeed = getSceneCpuVolumeMaxSpeed();
			if(!gMetrics.sceneSoftFirstSlept && sleeping)
			{
				gMetrics.sceneSoftFirstSlept = 1;
				gMetrics.sceneSoftFirstSleepFrame =
					gMetrics.completedFrames;
				gSceneCpuFirstSleepCentroidY =
					getSceneCpuVolumeCentroidY();
				gMetrics.sceneSoftSleepWakeCounterZero =
					gSceneCpuVolume->getWakeCounter() == 0.0f
						? 1u : 0u;
				gMetrics.sceneSoftSleepVelocitiesZero =
					maxSpeed <= 1.0e-7f ? 1u : 0u;
			}
			else if(softSleepWakeCase &&
				gMetrics.sceneSoftCounterWakeIssued &&
				!gMetrics.sceneSoftWokeByCounter && !sleeping)
			{
				gMetrics.sceneSoftWokeByCounter = 1;
				gMetrics.sceneSoftCounterWakeFrame =
					gMetrics.completedFrames;
			}
			else if(softSleepWakeCase &&
				gMetrics.sceneSoftWokeByCounter &&
				!gMetrics.sceneSoftSecondSlept && sleeping)
			{
				gMetrics.sceneSoftSecondSlept = 1;
				gMetrics.sceneSoftSecondSleepFrame =
					gMetrics.completedFrames;
				gSceneCpuSecondSleepCentroidY =
					getSceneCpuVolumeCentroidY();
			}
			else if(softSleepWakeCase &&
				gMetrics.sceneSoftVelocityWakeIssued &&
				!gMetrics.sceneSoftWokeByVelocity && !sleeping)
			{
				gMetrics.sceneSoftWokeByVelocity = 1;
				gMetrics.sceneSoftVelocityWakeFrame =
					gMetrics.completedFrames;
			}
			if(softSleepWakeCase &&
				gMetrics.sceneSoftWokeByVelocity &&
				getSceneCpuVolumeCentroidY() -
					gSceneCpuVelocityWakeCentroidY > 0.02f)
				gMetrics.sceneSoftMovedAfterVelocityWake = 1;
			if(((softSleepWakeCase || softRigidWakeCase) &&
					gMetrics.sceneSoftVelocityStopIssued ||
				(bufferMutationCase &&
					gMetrics.sceneBufferResetIssued)) &&
				!gMetrics.sceneSoftFinalSlept && sleeping)
			{
				gMetrics.sceneSoftFinalSlept = 1;
				gMetrics.sceneSoftFinalSleepFrame =
					gMetrics.completedFrames;
			}
			if(softRigidWakeCase &&
				gMetrics.sceneSoftRigidWakeActorAdded &&
				!gMetrics.sceneSoftWokeByRigid && !sleeping)
			{
				gMetrics.sceneSoftWokeByRigid = 1;
				gMetrics.sceneSoftRigidWakeFrame =
					gMetrics.completedFrames;
			}
			if(softRigidWakeCase &&
				gMetrics.sceneSoftWokeByRigid &&
				PxAbs(
					getSceneCpuVolumeCentroidY() -
					gSceneCpuRigidWakeCentroidY) > 0.005f)
				gMetrics.sceneSoftMovedAfterRigidWake = 1;
			if(bufferMutationCase &&
				gMetrics.sceneBufferMutationIssued)
			{
				if(!gMetrics.sceneBufferMutationWoke &&
					!sleeping)
					gMetrics.sceneBufferMutationWoke = 1;
				const PxReal pinDeviation =
					(positions[0].getXYZ() -
						gSceneCpuBufferPinTarget).magnitude();
				const bool pinValid =
					PxIsFinite(pinDeviation) &&
					pinDeviation <= 1.0e-4f &&
					PxAbs(positions[0].w) <= 1.0e-7f;
				if(!gMetrics.sceneBufferMutationApplied &&
					pinValid)
				{
					gMetrics.sceneBufferMutationApplied = 1;
					gMetrics.sceneBufferPinHeld = 1;
				}
				else if(gMetrics.sceneBufferDriveIssued &&
					!gMetrics.sceneBufferInvMassRestored &&
					!pinValid)
					gMetrics.sceneBufferPinHeld = 0;
				if(gMetrics.sceneBufferDriveIssued &&
					!gMetrics.sceneBufferInvMassRestored &&
					(positions[1].getXYZ() -
						gSceneCpuBufferDynamicBaseline).
							magnitude() > 1.0e-5f)
					gMetrics.sceneBufferDynamicMoved = 1;
				if(gMetrics.sceneBufferInvMassRestored &&
					!gMetrics.sceneBufferResetIssued &&
					PxAbs(
						positions[0].w -
						gSceneCpuBufferOriginalInvMass) <=
							1.0e-7f &&
					(positions[0].getXYZ() -
						gSceneCpuBufferRestoredBaseline).
							magnitude() > 1.0e-5f)
					gMetrics.sceneBufferRestoredMoved = 1;
			}
		}
		if(softSoftWakeCase && gSceneCpuSecondVolume)
		{
			const bool targetSleeping =
				gSceneCpuVolume->isSleeping();
			const bool driverSleeping =
				gSceneCpuSecondVolume->isSleeping();
			if(!gMetrics.sceneSoftSoftDriveIssued &&
				targetSleeping && driverSleeping)
				gMetrics.sceneSoftSoftBothSlept = 1;
			if(gMetrics.sceneSoftSoftDriveIssued)
			{
				if(!driverSleeping)
					gMetrics.sceneSoftSoftDriverWoke = 1;
				if(!gMetrics.sceneSoftSoftTargetWoke &&
					!targetSleeping)
				{
					gMetrics.sceneSoftSoftTargetWoke = 1;
					gMetrics.sceneSoftSoftTargetWakeFrame =
						gMetrics.completedFrames;
				}
				if(gMetrics.sceneSoftSoftTargetWoke &&
					(getSceneCpuVolumeCentroid(
						gSceneCpuVolume) -
						gSceneCpuSoftSoftTargetBaseline).
							magnitude() > 0.002f)
					gMetrics.sceneSoftSoftTargetMoved = 1;
				if(gMetrics.sceneSoftSoftResetIssued &&
					targetSleeping && driverSleeping)
					gMetrics.sceneSoftSoftBothFinalSlept = 1;
			}
		}
		if(mixedSleepIslandCase && gSceneCpuSecondVolume)
		{
			const bool firstSleeping =
				gSceneCpuVolume->isSleeping();
			const bool secondSleeping =
				gSceneCpuSecondVolume->isSleeping();
			if(!gMetrics.sceneMixedFirstSlept && firstSleeping)
			{
				gMetrics.sceneMixedFirstSlept = 1;
				gMetrics.sceneMixedFirstSleepFrame =
					gMetrics.completedFrames;
				gSceneCpuFirstSleepCentroidY =
					getSceneCpuVolumeCentroidY();
			}
			if(gMetrics.sceneMixedFirstSlept && !secondSleeping)
				gMetrics.sceneMixedSecondStayedAwake = 1;
			if(gMetrics.sceneMixedFirstSlept &&
				gMetrics.completedFrames >=
					gMetrics.sceneMixedFirstSleepFrame + 2 &&
				PxAbs(
					getSceneCpuVolumeCentroidY() -
					gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
				gMetrics.sceneMixedFirstStable = 1;
			if(getSceneCpuVolumeCentroidY(
					gSceneCpuSecondVolume) -
				gSceneCpuSecondVolumeInitialCentroidY > 0.05f)
				gMetrics.sceneMixedSecondMoved = 1;
		}
		const PxReal dynamicHalfX =
			multiDynamicBoxCase ? 0.45f : 4.0f;
		PxTransform dynamicBoxPose(PxIdentity);
		PxTransform secondDynamicBoxPose(PxIdentity);
		PxTransform kinematicBoxPose(PxIdentity);
		if(kinematicBoxCase && gSceneCpuDynamicActor &&
			gSceneCpuDynamicActor->getScene() == gScene)
		{
			kinematicBoxPose =
				gSceneCpuDynamicActor->getGlobalPose();
			if(!kinematicBoxPose.isValid())
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.sceneKinematicFinalY =
					kinematicBoxPose.p.y;
				if(gMetrics.sceneKinematicTargetIssued)
				{
					const PxReal poseError = PxAbs(
						kinematicBoxPose.p.y -
							gSceneCpuKinematicCommandY);
					gMetrics.sceneKinematicMaxPoseError =
						PxMax(
							gMetrics.
								sceneKinematicMaxPoseError,
							poseError);
					if(gSceneCpuKinematicCommandY >= 4.10f &&
						poseError <= 1.0e-4f)
						gMetrics.sceneKinematicTargetReached =
							1;
					if(!gSceneCpuVolume->isSleeping())
						gMetrics.sceneKinematicSoftWoke = 1;
					gMetrics.sceneKinematicSoftDisplacement =
						PxMax(
							gMetrics.
								sceneKinematicSoftDisplacement,
							PxAbs(
								getSceneCpuVolumeCentroidY() -
								gSceneCpuKinematicSoftBaselineY));
					if(gMetrics.
						sceneKinematicSoftDisplacement >
						0.02f)
						gMetrics.sceneKinematicSoftMoved = 1;
				}
			}
		}
		if(dynamicActorInScene)
		{
			if(gMetrics.sceneDynamicReaddedSleeping &&
				!gMetrics.sceneDynamicRewokeBySoft &&
				!gSceneCpuDynamicActor->isSleeping())
			{
				gMetrics.sceneDynamicRewokeBySoft = 1;
				gMetrics.sceneDynamicSecondWakeFrame =
					gMetrics.completedFrames;
			}
			else if(gMetrics.sceneDynamicInitiallySleeping &&
				!gMetrics.sceneDynamicWokeBySoft &&
				!gSceneCpuDynamicActor->isSleeping())
			{
				gMetrics.sceneDynamicWokeBySoft = 1;
				gMetrics.sceneDynamicFirstWakeFrame =
					gMetrics.completedFrames;
			}
			dynamicBoxPose =
				gSceneCpuDynamicActor->getGlobalPose();
			const PxVec3 dynamicVelocity =
				gSceneCpuDynamicActor->getLinearVelocity();
			if(!dynamicBoxPose.isValid() ||
				!dynamicVelocity.isFinite())
			{
				gMetrics.nonFiniteParticleSamples++;
			}
			else
			{
				gMetrics.sceneDynamicMinY = PxMin(
					gMetrics.sceneDynamicMinY,
					dynamicBoxPose.p.y);
				gMetrics.sceneDynamicFinalY =
					dynamicBoxPose.p.y;
				gMetrics.sceneDynamicMaxDrop = PxMax(
					gMetrics.sceneDynamicMaxDrop,
					gSceneCpuDynamicInitialY -
						dynamicBoxPose.p.y);
				gMetrics.sceneDynamicMaxDownSpeed = PxMax(
					gMetrics.sceneDynamicMaxDownSpeed,
					PxMax(-dynamicVelocity.y, 0.0f));
			}
		}
		if(secondDynamicActorInScene)
		{
			if(gMetrics.sceneSecondDynamicInitiallySleeping &&
				!gMetrics.sceneSecondDynamicWokeBySoft &&
				!gSceneCpuSecondDynamicActor->isSleeping())
			{
				gMetrics.sceneSecondDynamicWokeBySoft = 1;
				gMetrics.sceneSecondDynamicFirstWakeFrame =
					gMetrics.completedFrames;
			}
			secondDynamicBoxPose =
				gSceneCpuSecondDynamicActor->getGlobalPose();
			const PxVec3 dynamicVelocity =
				gSceneCpuSecondDynamicActor->getLinearVelocity();
			if(!secondDynamicBoxPose.isValid() ||
				!dynamicVelocity.isFinite())
			{
				gMetrics.nonFiniteParticleSamples++;
			}
			else
			{
				gMetrics.sceneSecondDynamicMinY = PxMin(
					gMetrics.sceneSecondDynamicMinY,
					secondDynamicBoxPose.p.y);
				gMetrics.sceneSecondDynamicFinalY =
					secondDynamicBoxPose.p.y;
				gMetrics.sceneSecondDynamicMaxDrop = PxMax(
					gMetrics.sceneSecondDynamicMaxDrop,
					gSceneCpuSecondDynamicInitialY -
						secondDynamicBoxPose.p.y);
				gMetrics.sceneSecondDynamicMaxDownSpeed = PxMax(
					gMetrics.sceneSecondDynamicMaxDownSpeed,
					PxMax(-dynamicVelocity.y, 0.0f));
			}
		}
		for(PxU32 i = 0; i < vertexCount; i++)
		{
			const PxVec3 position = positions[i].getXYZ();
			const PxVec3 velocity = velocities[i].getXYZ();
			if(!position.isFinite() || !velocity.isFinite() ||
				!PxIsFinite(positions[i].w))
			{
				gMetrics.nonFiniteParticleSamples++;
				continue;
			}
			gMetrics.minY = PxMin(gMetrics.minY, position.y);
			gMetrics.maxY = PxMax(gMetrics.maxY, position.y);
			frameMinY = PxMin(frameMinY, position.y);
			gMetrics.maxParticleSpeed = PxMax(
				gMetrics.maxParticleSpeed, velocity.magnitude());
			if(partialElementFilterCase &&
				!gMetrics.sceneElementFilterReleased)
			{
				const bool positiveX =
					simulationRestVertices[i].x > 0.0f;
				if(positiveX ==
					gSceneCpuPartialFilterSelectedPositiveX)
				{
					gMetrics.sceneElementFilterMinY = PxMin(
						gMetrics.sceneElementFilterMinY,
						position.y);
				}
				else
				{
					gMetrics.
						scenePartialFilterUnfilteredMinY = PxMin(
							gMetrics.
								scenePartialFilterUnfilteredMinY,
							position.y);
					if(position.y > -0.05f &&
						position.y < 0.05f)
						gMetrics.
							scenePartialFilterUnfilteredContactHeld =
								1;
				}
			}
			if(gHeadlessOptions.caseName == "scene-volume-ground" &&
				position.y <= 0.08f)
				nearGroundParticles++;
			if((gHeadlessOptions.caseName == "scene-volume-static-box" ||
				gHeadlessOptions.caseName == "scene-volume-static-churn") &&
				position.y <= 1.08f &&
				PxAbs(position.x) <= 20.08f &&
				PxAbs(position.z) <= 20.08f)
				nearRigidParticles++;
			if(kinematicBoxCase && kinematicBoxPose.isValid())
			{
				const PxVec3 localPosition =
					kinematicBoxPose.transformInv(position);
				PxVec3 capsuleRadial = localPosition;
				capsuleRadial.x -= PxClamp(
					capsuleRadial.x, -0.3f, 0.3f);
				const bool nearKinematicShape =
					kinematicCapsuleCase
						? capsuleRadial.magnitude() <= 0.58f
						: kinematicTriangleMeshCase
						? PxAbs(localPosition.x) <= 4.08f &&
							PxAbs(localPosition.z) <= 4.08f &&
							PxAbs(localPosition.y) <= 0.12f
						: kinematicHeightFieldCase
						? localPosition.x >= -0.08f &&
							localPosition.x <= 8.08f &&
							localPosition.z >= -0.08f &&
							localPosition.z <= 8.08f &&
							PxAbs(localPosition.y) <= 0.12f
						: kinematicConvexCase &&
							gSceneCpuRigidConvexMesh
						? PxGeometryQuery::pointDistance(
							position,
							PxConvexMeshGeometry(
								gSceneCpuRigidConvexMesh),
							kinematicBoxPose) <=
								0.08f * 0.08f
						: kinematicSphereCase
						? localPosition.magnitude() <= 0.58f
						: PxAbs(localPosition.x) <= 4.08f &&
							PxAbs(localPosition.z) <= 4.08f &&
							PxAbs(localPosition.y) <= 0.35f;
				if(nearKinematicShape)
					gMetrics.sceneKinematicContactObserved = 1;
			}
			if(dynamicActorInScene && dynamicBoxPose.isValid())
			{
				const PxVec3 localPosition =
					dynamicBoxPose.transformInv(position);
				if(dynamicConvexCase &&
					gSceneCpuRigidConvexMesh)
				{
					const PxReal squaredDistance =
						PxGeometryQuery::pointDistance(
							position,
							PxConvexMeshGeometry(
								gSceneCpuRigidConvexMesh),
							dynamicBoxPose);
					if(squaredDistance >= 0.0f &&
						PxIsFinite(squaredDistance))
					{
						const PxReal separation =
							PxSqrt(squaredDistance);
						frameDynamicSurfaceSeparation =
							PxMin(
								frameDynamicSurfaceSeparation,
								separation);
						gMetrics.
							minDynamicSurfaceSeparation =
							PxMin(
								gMetrics.
									minDynamicSurfaceSeparation,
								separation);
						if(separation <= 0.08f)
							nearRigidParticles++;
					}
				}
				else if(dynamicSphereCase)
				{
					PxVec3 radial = localPosition;
					if(dynamicCapsuleCase)
						radial.x -= PxClamp(
							radial.x, -0.3f, 0.3f);
					const PxReal radius = radial.magnitude();
					const PxReal separation = radius - 0.8f;
					frameDynamicSurfaceSeparation = PxMin(
						frameDynamicSurfaceSeparation,
						separation);
					gMetrics.minDynamicSurfaceSeparation = PxMin(
						gMetrics.minDynamicSurfaceSeparation,
						separation);
					if(radius >= 0.65f && radius <= 0.88f)
						nearRigidParticles++;
				}
				else if(
					PxAbs(localPosition.x) <= dynamicHalfX + 0.08f &&
					PxAbs(localPosition.z) <= 4.08f)
				{
					const PxReal separation =
						localPosition.y - 0.25f;
					frameDynamicSurfaceSeparation = PxMin(
						frameDynamicSurfaceSeparation,
						separation);
					gMetrics.minDynamicSurfaceSeparation = PxMin(
						gMetrics.minDynamicSurfaceSeparation,
						separation);
				}
				if(localPosition.y >= 0.10f &&
					localPosition.y <= 0.33f &&
					PxAbs(localPosition.x) <= dynamicHalfX + 0.08f &&
					PxAbs(localPosition.z) <= 4.08f)
					nearRigidParticles++;
			}
			if(secondDynamicActorInScene &&
				secondDynamicBoxPose.isValid())
			{
				const PxVec3 localPosition =
					secondDynamicBoxPose.transformInv(position);
				if(PxAbs(localPosition.x) <= dynamicHalfX + 0.08f &&
					PxAbs(localPosition.z) <= 4.08f)
				{
					const PxReal separation =
						localPosition.y - 0.25f;
					frameDynamicSurfaceSeparation = PxMin(
						frameDynamicSurfaceSeparation,
						separation);
					gMetrics.minDynamicSurfaceSeparation = PxMin(
						gMetrics.minDynamicSurfaceSeparation,
						separation);
				}
				if(localPosition.y >= 0.10f &&
					localPosition.y <= 0.33f &&
					PxAbs(localPosition.x) <= dynamicHalfX + 0.08f &&
					PxAbs(localPosition.z) <= 4.08f)
					nearRigidParticles++;
			}
		}
		if(elementFilterCase && frameMinY != FLT_MAX)
		{
			if(!gMetrics.sceneElementFilterReleased)
			{
				if(!partialElementFilterCase)
					gMetrics.sceneElementFilterMinY = PxMin(
						gMetrics.sceneElementFilterMinY,
						frameMinY);
				if(gMetrics.sceneElementFilterMinY < -0.2f)
					gMetrics.sceneElementFilterSuppressedContact =
						1;
			}
			else
				gMetrics.sceneElementFilterFinalMinY = frameMinY;
		}
		if(twoSoftVolumeCase && gSceneCpuSecondVolume)
		{
			PxVec4* secondPositions =
				gSceneCpuSecondVolume->
					getSimPositionInvMassBufferH();
			PxVec4* secondVelocities =
				gSceneCpuSecondVolume->
					getSimVelocityBufferH();
			const PxU32 secondVertexCount =
				gSceneCpuSecondVolume->getSimulationMesh()->
					getNbVertices();
			if(!secondPositions || !secondVelocities)
				return false;
			for(PxU32 i = 0; i < secondVertexCount; i++)
			{
				const PxVec3 position =
					secondPositions[i].getXYZ();
				const PxVec3 velocity =
					secondVelocities[i].getXYZ();
				if(!position.isFinite() ||
					!velocity.isFinite() ||
					!PxIsFinite(secondPositions[i].w))
				{
					gMetrics.nonFiniteParticleSamples++;
					continue;
				}
				gMetrics.minY =
					PxMin(gMetrics.minY, position.y);
				gMetrics.maxY =
					PxMax(gMetrics.maxY, position.y);
				gMetrics.maxParticleSpeed = PxMax(
					gMetrics.maxParticleSpeed,
					velocity.magnitude());
				if(secondDynamicActorInScene &&
					secondDynamicBoxPose.isValid())
				{
					const PxVec3 localPosition =
						secondDynamicBoxPose.transformInv(
							position);
					if(PxAbs(localPosition.x) <=
							dynamicHalfX + 0.08f &&
						PxAbs(localPosition.z) <= 4.08f)
					{
						const PxReal separation =
							localPosition.y - 0.25f;
						frameDynamicSurfaceSeparation =
							PxMin(
								frameDynamicSurfaceSeparation,
								separation);
						gMetrics.minDynamicSurfaceSeparation =
							PxMin(
								gMetrics.
									minDynamicSurfaceSeparation,
								separation);
					}
					if(localPosition.y >= 0.10f &&
						localPosition.y <= 0.33f &&
						PxAbs(localPosition.x) <=
							dynamicHalfX + 0.08f &&
						PxAbs(localPosition.z) <= 4.08f)
						nearRigidParticles++;
				}
			}
			const PxReal secondCentroidY =
				getSceneCpuVolumeCentroidY(
					gSceneCpuSecondVolume);
			gMetrics.sceneSecondVolumeFinalCentroidY =
				secondCentroidY;
			gMetrics.sceneSecondVolumeMaxCentroidDrop = PxMax(
				gMetrics.sceneSecondVolumeMaxCentroidDrop,
				gSceneCpuSecondVolumeInitialCentroidY -
					secondCentroidY);
			const PxBounds3 secondBounds =
				gSceneCpuSecondVolume->getWorldBounds();
			if(secondBounds.isValid() &&
				secondBounds.minimum.isFinite() &&
				secondBounds.maximum.isFinite())
				gMetrics.sceneSecondVolumeBoundsFinite = 1;
		}
		if(dynamicActorInScene &&
			PxIsFinite(frameDynamicSurfaceSeparation) &&
			frameDynamicSurfaceSeparation != FLT_MAX)
		{
			gMetrics.finalDynamicSurfaceSeparation =
				frameDynamicSurfaceSeparation;
		}
		if(dynamicActorInScene && gMetrics.rigidContactFrames == 0 &&
			nearRigidParticles == 0 && dynamicBoxPose.isValid())
		{
			gMetrics.sceneDynamicPreContactMaxDrop = PxMax(
				gMetrics.sceneDynamicPreContactMaxDrop,
				PxAbs(gSceneCpuDynamicInitialY -
					dynamicBoxPose.p.y));
			if(secondDynamicActorInScene &&
				secondDynamicBoxPose.isValid())
			{
				gMetrics.sceneSecondDynamicPreContactMaxDrop = PxMax(
					gMetrics.sceneSecondDynamicPreContactMaxDrop,
					PxAbs(gSceneCpuSecondDynamicInitialY -
						secondDynamicBoxPose.p.y));
			}
		}
		if(nearGroundParticles > 0)
		{
			gMetrics.groundContactFrames++;
			gMetrics.maxGroundContacts = PxMax(
				gMetrics.maxGroundContacts, nearGroundParticles);
		}
		if(nearRigidParticles > 0)
		{
			gMetrics.rigidContactFrames++;
			gMetrics.maxRigidContacts = PxMax(
				gMetrics.maxRigidContacts, nearRigidParticles);
		}
		const PxReal centroidY = getSceneCpuVolumeCentroidY();
		gMetrics.maxCentroidDrop = PxMax(
			gMetrics.maxCentroidDrop,
			gSceneCpuVolumeInitialCentroidY - centroidY);
		const PxBounds3 bounds = gSceneCpuVolume->getWorldBounds();
		if(bounds.isValid() &&
			bounds.minimum.isFinite() && bounds.maximum.isFinite())
			gMetrics.sceneBoundsFinite = 1;

		if(profileFrame)
		{
			gPerformance.profiledFrames++;
			gPerformance.sceneMs += sceneMs;
			gPerformance.stepSamplesMs.pushBack(
				PxReal(frameTimer.getElapsedSeconds() * 1000.0));
		}
		return true;
	}

	PxTime stageTimer;
	gFrameCollisionStats = AvbdSoftCollisionStats();
	gSoftWorkspace.contact.beginStep();

	// Initial contact detection: ground + soft-soft OGC + rigid-soft SDF.
	AvbdSoftCollisionStats initialCollisionStats;
	avbdDetectAllOGCContacts(
		gParticles.begin(), gParticles.size(),
		gSoftBodies.begin(), gSoftBodies.size(),
		gRigidBoxes.begin(), gRigidBoxes.size(),
		NULL, 0,
		gContacts, gOGCParams, 0.0f, &initialCollisionStats,
		&gSoftWorkspace.contact);
	gFrameCollisionStats.accumulate(initialCollisionStats);
	const PxF64 initialContactMs =
		stageTimer.getElapsedSeconds() * 1000.0;
	const PxU64 initialContactWorkspaceGrowthEvents =
		gSoftWorkspace.contact.growthEvents;
	const PxU64 initialContactWorkspaceGrowthBytes =
		gSoftWorkspace.contact.growthBytes;
	const PxU64 initialContactOutputGrowthEvents =
		gSoftWorkspace.contact.outputGrowthEvents;
	const PxU64 initialContactOutputGrowthBytes =
		gSoftWorkspace.contact.outputGrowthBytes;
	recordContactMetrics();

	// 8 outer iterations with contact re-detection between each.
	// Contacts are re-detected via callback so surface-point anchors
	// track the deforming geometry instead of going stale.
	AvbdSoftBodyStepStats stepStats;
	PxTime solverTimer;
	avbdStepSoftBodies(
		gParticles.begin(), gParticles.size(),
		gSoftBodies.begin(), gSoftBodies.size(),
		gContacts.begin(), gContacts.size(),
		dt, PxVec3(0.0f, -9.81f, 0.0f), 8, 20, 1000.0f,
		redetectContacts, &gContacts, NULL, 0.92f, &stepStats,
		&gSoftWorkspace);
	// avbdStepSoftBodies starts its own solver-stage counters.  Preserve the
	// contact-prep growth observed before entering the solver and combine it
	// with the outer-loop redetection growth reported by the solver.
	stepStats.contactWorkspaceGrowthEvents +=
		initialContactWorkspaceGrowthEvents;
	stepStats.contactWorkspaceGrowthBytes +=
		initialContactWorkspaceGrowthBytes;
	stepStats.contactOutputGrowthEvents +=
		initialContactOutputGrowthEvents;
	stepStats.contactOutputGrowthBytes +=
		initialContactOutputGrowthBytes;
	const PxF64 solverMs = solverTimer.getElapsedSeconds() * 1000.0;

	PxTime sceneTimer;
	gScene->simulate(dt);
	if(!gScene->fetchResults(true))
	{
		gMetrics.fetchFailures++;
		return false;
	}
	const PxF64 sceneMs = sceneTimer.getElapsedSeconds() * 1000.0;

	PxTime metricsTimer;
	gMetrics.completedFrames++;
	recordStateMetrics();
	sendSoftBodiesToPvd();
	const PxF64 metricsMs = metricsTimer.getElapsedSeconds() * 1000.0;
	if(profileFrame)
	{
		gPerformance.profiledFrames++;
		gPerformance.initialContactMs += initialContactMs;
		gPerformance.solverMs += solverMs;
		gPerformance.sceneMs += sceneMs;
		gPerformance.metricsMs += metricsMs;
		accumulateStepStats(gPerformance.solverStages, stepStats);
		gPerformance.collision.accumulate(gFrameCollisionStats);
		gPerformance.stepSamplesMs.pushBack(
			PxReal(frameTimer.getElapsedSeconds() * 1000.0));
	}
	return true;
}

void stepPhysics(bool /*interactive*/)
{
	stepPhysicsInternal(1.0f / 60.0f);
}

static void finalizeMetrics()
{
	gMetrics.finalInsideParticles = 0;
	gMetrics.finalMinY = FLT_MAX;
	gMetrics.finalMaxY = -FLT_MAX;
	gMetrics.finalMaxParticleSpeed = 0.0f;
	if(isSceneCpuVolumeCase(gHeadlessOptions.caseName) &&
		gSceneCpuVolume)
	{
		PxDeformableVolume* volumes[2] =
		{
			gSceneCpuVolume,
			gSceneCpuSecondVolume
		};
		const PxU32 volumeCount =
			gSceneCpuSecondVolume ? 2u : 1u;
		for(PxU32 volumeId = 0;
			volumeId < volumeCount; ++volumeId)
		{
			PxDeformableVolume* volume = volumes[volumeId];
			const PxVec4* positions =
				volume->getSimPositionInvMassBufferH();
			const PxVec4* velocities =
				volume->getSimVelocityBufferH();
			const PxU32 vertexCount =
				volume->getSimulationMesh()->getNbVertices();
			for(PxU32 i = 0; i < vertexCount; i++)
			{
				const PxVec3 position =
					positions[i].getXYZ();
				const PxVec3 velocity =
					velocities[i].getXYZ();
				if(!position.isFinite() ||
					!velocity.isFinite())
					continue;
				gMetrics.finalMinY =
					PxMin(gMetrics.finalMinY, position.y);
				gMetrics.finalMaxY =
					PxMax(gMetrics.finalMaxY, position.y);
				gMetrics.finalMaxParticleSpeed = PxMax(
					gMetrics.finalMaxParticleSpeed,
					velocity.magnitude());
			}
		}
		if(gSceneCpuSecondVolume)
			gMetrics.sceneSecondVolumeFinalCentroidY =
				getSceneCpuVolumeCentroidY(
					gSceneCpuSecondVolume);
		return;
	}

	for(PxU32 particleId = 0; particleId < gParticles.size(); ++particleId)
	{
		const AvbdSoftParticle& particle = gParticles[particleId];
		if(!particle.position.isFinite() || !particle.velocity.isFinite())
			continue;
		gMetrics.finalMinY =
			PxMin(gMetrics.finalMinY, particle.position.y);
		gMetrics.finalMaxY =
			PxMax(gMetrics.finalMaxY, particle.position.y);
		gMetrics.finalMaxParticleSpeed = PxMax(
			gMetrics.finalMaxParticleSpeed, particle.velocity.magnitude());
	}

	for(PxU32 bodyA = 0; bodyA < gSoftBodies.size(); ++bodyA)
	{
		const AvbdSoftBody& source = gSoftBodies[bodyA];
		for(PxU32 bodyB = 0; bodyB < gSoftBodies.size(); ++bodyB)
		{
			if(bodyA == bodyB)
				continue;
			const AvbdSoftBody& target = gSoftBodies[bodyB];
			for(PxU32 localId = 0;
				localId < source.compiled.particleCount; ++localId)
			{
				const PxU32 particleId = source.compiled.particleStart + localId;
				if(avbdIsPointInsideTetMesh(
					gParticles[particleId].position,
					target.compiled.surfaceTriangles, gParticles.begin()))
				{
					gMetrics.finalInsideParticles++;
				}
			}
		}
	}
}

void cleanupPhysics(bool /*interactive*/)
{
	// PxArray uses the PhysX foundation allocator. Interactive shutdown does
	// not pass through finalizePerformanceMetrics(), so release the sample
	// storage here while the foundation broadcast allocator is still alive.
	if(gSceneCpuElementFilter)
	{
		gSceneCpuElementFilter->release();
		gSceneCpuElementFilter = NULL;
	}
	if(gSceneCpuWorldAttachment)
	{
		gSceneCpuWorldAttachment->release();
		gSceneCpuWorldAttachment = NULL;
	}
	if(gSceneCpuRigidAttachment)
	{
		gSceneCpuRigidAttachment->release();
		gSceneCpuRigidAttachment = NULL;
	}
	if(gSceneCpuSecondVolume)
	{
		PxScene* volumeScene =
			gSceneCpuSecondVolume->getScene();
		if(volumeScene)
		{
			volumeScene->removeActor(*gSceneCpuSecondVolume);
			if(gSceneCpuSecondVolume->getScene() == NULL)
				gMetrics.sceneSecondVolumeActorRemoved = 1;
		}
		gSceneCpuSecondVolume->release();
		gSceneCpuSecondVolume = NULL;
		gMetrics.sceneSecondVolumeActorReleased = 1;
	}
	if(gSceneCpuVolume)
	{
		PxScene* volumeScene = gSceneCpuVolume->getScene();
		if(volumeScene)
		{
			volumeScene->removeActor(*gSceneCpuVolume);
			if(gSceneCpuVolume->getScene() == NULL &&
				volumeScene->getNbDeformableVolumes() == 0)
				gMetrics.sceneActorRemoved = 1;
		}
		gSceneCpuVolume->release();
		gSceneCpuVolume = NULL;
		gMetrics.sceneActorReleased = 1;
	}
	PX_RELEASE(gSceneCpuVolumeMesh);
	PX_RELEASE(gSceneCpuVolumeMaterial);
	if(gSceneCpuAttachmentArticulation)
	{
		gSceneCpuAttachmentBody = NULL;
		gSceneCpuAttachmentLink = NULL;
		gSceneCpuAttachmentRoot = NULL;
		gSceneCpuAttachmentArticulation->release();
		gSceneCpuAttachmentArticulation = NULL;
	}
	if(gSceneCpuStaticActor)
	{
		if(gScene &&
			gSceneCpuStaticActor->getScene() == gScene)
			gScene->removeActor(*gSceneCpuStaticActor);
		gSceneCpuStaticActor->release();
		gSceneCpuStaticActor = NULL;
	}
	if(gSceneCpuDynamicActor)
	{
		if(gScene &&
			gSceneCpuDynamicActor->getScene() == gScene)
		{
			gScene->removeActor(*gSceneCpuDynamicActor);
			if(gSceneCpuDynamicActor->getScene() == NULL)
				gMetrics.sceneDynamicActorRemoved = 1;
		}
		gSceneCpuDynamicActor->release();
		gSceneCpuDynamicActor = NULL;
		gSceneCpuAttachmentBody = NULL;
		gMetrics.sceneDynamicActorReleased = 1;
	}
	if(gSceneCpuSecondDynamicActor)
	{
		if(gScene &&
			gSceneCpuSecondDynamicActor->getScene() == gScene)
		{
			gScene->removeActor(*gSceneCpuSecondDynamicActor);
			if(gSceneCpuSecondDynamicActor->getScene() == NULL)
				gMetrics.sceneSecondDynamicActorRemoved = 1;
		}
		gSceneCpuSecondDynamicActor->release();
		gSceneCpuSecondDynamicActor = NULL;
		gMetrics.sceneSecondDynamicActorReleased = 1;
	}
	PX_RELEASE(gSceneCpuRigidConvexMesh);
	PX_RELEASE(gSceneCpuRigidHeightField);
	PX_RELEASE(gSceneCpuRigidTriangleMesh);

	gPerformance.stepSamplesMs.reset();
	gSoftBodyRenderData.reset();
	gContacts.reset();
	gRigidBoxes.reset();
	gSoftBodies.reset();
	gParticles.reset();
	gInitialCentroids.reset();
	gSceneCpuVolumeKinematicTargets.reset();
	gSceneCpuVolumeKinematicInitial.reset();
	gSceneCpuSphereReverseSweptInitialPositions.reset();
	gSceneCpuDeformingReverseSweptFreeEndPositions.reset();
	gSceneCpuCapsuleRotationalSweptInitialPositions.reset();
	gVolumeSkinningBindings.reset();
	gVolumeSkinningTriangles.reset();
	gVolumeSkinningPositions.reset();
	gVolumeSkinningNormals.reset();
	gVolumeSkinningInitialPositions.reset();
	gVolumeAvbdSkinningRenderData =
		VolumeAvbdSkinningRenderData();
	gSoftWorkspace.reset();

	PX_RELEASE(gScene);
	if(gSceneCpuSecondScene)
	{
		PX_RELEASE(gSceneCpuSecondScene);
		gMetrics.sceneSecondSceneReleased = 1;
	}
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gMaterial);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	if (gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gScene && !gSceneCpuSecondScene &&
		!gDispatcher && !gMaterial && !gPhysics &&
		!gPvd && !gFoundation && !gSceneCpuVolume &&
		!gSceneCpuSecondVolume &&
		!gSceneCpuWorldAttachment &&
		!gSceneCpuRigidAttachment &&
		!gSceneCpuElementFilter &&
		!gSceneCpuVolumeMesh && !gSceneCpuVolumeMaterial &&
		!gSceneCpuStaticActor && !gSceneCpuDynamicActor &&
		!gSceneCpuSecondDynamicActor &&
		!gSceneCpuAttachmentArticulation &&
		!gSceneCpuAttachmentRoot &&
		!gSceneCpuAttachmentLink &&
		!gSceneCpuAttachmentBody ? 1u : 0u;

	printf("%s done.\n", AVBD_VOLUME_SNIPPET_NAME);
}

void keyPress(unsigned char /*key*/, const PxTransform& /*camera*/)
{
}

PxDeformableVolume* getPrimaryCpuAvbdVolume()
{
	return gSceneCpuVolume;
}

static bool isKnownCase(const std::string& caseName)
{
	return caseName == "volume-ground" ||
		caseName == "volume-static-box" ||
		caseName == "soft-soft" ||
		caseName == "cone-ground" ||
		caseName == "scene-volume-lifecycle" ||
		caseName == "scene-volume-corotational" ||
		caseName == "scene-volume-ground" ||
		caseName == "scene-volume-static-box" ||
		caseName == "scene-volume-static-churn" ||
		caseName == "scene-volume-dynamic-box" ||
		caseName == "scene-volume-dynamic-sphere" ||
		caseName == "scene-volume-dynamic-capsule" ||
		caseName == "scene-volume-dynamic-convex" ||
		caseName == "scene-volume-dynamic-churn" ||
		caseName == "scene-volume-multi-dynamic-box" ||
		caseName == "scene-volume-multi-soft-islands" ||
		caseName == "scene-volume-sleep-wake" ||
		caseName == "scene-volume-rigid-wake" ||
		caseName == "scene-volume-mixed-sleep-islands" ||
		caseName == "scene-volume-soft-churn" ||
		caseName == "scene-volume-buffer-mutation" ||
		caseName == "scene-volume-world-pin" ||
		caseName == "scene-volume-world-element-attachment" ||
		caseName == "scene-volume-rigid-attachment" ||
		caseName == "scene-volume-rigid-element-attachment" ||
		caseName == "scene-volume-static-attachment" ||
		caseName == "scene-volume-static-element-attachment" ||
		caseName == "scene-volume-kinematic-attachment" ||
		caseName == "scene-volume-kinematic-element-attachment" ||
		caseName == "scene-volume-articulation-attachment" ||
		caseName == "scene-volume-articulation-element-attachment" ||
		caseName == "scene-volume-element-filter" ||
		caseName == "scene-volume-partial-element-filter" ||
		isSceneCpuVolumeKinematicRigidCase(caseName) ||
		caseName == "scene-volume-full-kinematic-target" ||
		caseName == "scene-volume-partial-kinematic-target" ||
		caseName == "scene-volume-multi-scene-isolation" ||
		caseName == "scene-volume-soft-soft-wake" ||
		caseName == "scene-volume-volume-attachment" ||
		caseName == "scene-volume-skinning" ||
		caseName == "scene-volume-motion-controls" ||
		caseName ==
			"scene-volume-max-depenetration-velocity" ||
		caseName == "scene-volume-sphere-reverse-feature" ||
		caseName == "scene-volume-capsule-reverse-feature" ||
		caseName == "scene-volume-convex-reverse-feature" ||
		caseName ==
			"scene-volume-triangle-mesh-reverse-feature" ||
		caseName ==
			"scene-volume-heightfield-reverse-feature" ||
		isSceneCpuVolumeSpeculativeCcdCase(caseName) ||
		isSceneCpuVolumeSphereReverseSweptCcdCase(caseName) ||
		caseName == "current-all";
}

static bool validateHeadlessResult(const std::string& caseName)
{
	if(isSceneCpuVolumeCase(caseName))
	{
		const bool dynamicChurnCase =
			caseName == "scene-volume-dynamic-churn";
		const bool multiDynamicBoxCase =
			caseName == "scene-volume-multi-dynamic-box";
		const bool multiSoftIslandCase =
			caseName == "scene-volume-multi-soft-islands";
		const bool mixedSleepIslandCase =
			caseName == "scene-volume-mixed-sleep-islands";
		const bool softChurnCase =
			caseName == "scene-volume-soft-churn";
		const bool bufferMutationCase =
			caseName == "scene-volume-buffer-mutation";
		const bool worldPinCase =
			caseName == "scene-volume-world-pin" ||
			caseName == "scene-volume-world-element-attachment";
		const bool rigidAttachmentCase =
			caseName == "scene-volume-rigid-attachment" ||
			caseName == "scene-volume-rigid-element-attachment";
		const bool staticAttachmentCase =
			caseName == "scene-volume-static-attachment" ||
			caseName == "scene-volume-static-element-attachment";
		const bool kinematicAttachmentCase =
			caseName == "scene-volume-kinematic-attachment" ||
			caseName ==
				"scene-volume-kinematic-element-attachment";
		const bool articulationAttachmentCase =
			caseName == "scene-volume-articulation-attachment" ||
			caseName ==
				"scene-volume-articulation-element-attachment";
		const bool attachmentCase =
			rigidAttachmentCase || staticAttachmentCase ||
			kinematicAttachmentCase ||
			articulationAttachmentCase;
		const bool partialElementFilterCase =
			caseName ==
				"scene-volume-partial-element-filter";
		const bool elementFilterCase =
			caseName == "scene-volume-element-filter" ||
			partialElementFilterCase;
		const bool kinematicBoxCase =
			isSceneCpuVolumeKinematicRigidCase(caseName);
		const bool multiSceneIsolationCase =
			caseName == "scene-volume-multi-scene-isolation";
		const bool softSoftWakeCase =
			caseName == "scene-volume-soft-soft-wake";
		const bool softPairAttachmentCase =
			caseName == "scene-volume-volume-attachment";
		const bool fullKinematicTargetCase =
			caseName ==
				"scene-volume-full-kinematic-target";
		const bool partialKinematicTargetCase =
			caseName ==
				"scene-volume-partial-kinematic-target";
		const bool volumeKinematicTargetCase =
			fullKinematicTargetCase ||
			partialKinematicTargetCase;
		const bool motionControlsCase =
			caseName == "scene-volume-motion-controls";
		const bool maxDepenetrationVelocityCase =
			caseName ==
				"scene-volume-max-depenetration-velocity";
		const bool triangleSurfaceSweptCcdCase =
			isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
				caseName);
		const bool triangleSurfaceReverseSweptCcdCase =
			isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
				caseName);
		const bool staticTriangleSurfaceSweptCcdCase =
			isSceneCpuVolumeStaticTriangleSurfaceSweptCcdCase(
				caseName);
		const bool heightFieldSweptCcdCase =
			isSceneCpuVolumeHeightFieldSweptCcdCase(
				caseName);
		const bool rotationalTriangleSurfaceSweptCcdCase =
			isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
				caseName);
		const bool sphereReverseSweptCcdCase =
			isSceneCpuVolumeSphereReverseSweptCcdCase(caseName);
		const bool deformingReverseSweptCcdCase =
			isSceneCpuVolumeDeformingReverseSweptCcdCase(
				caseName);
		const bool capsuleReverseSweptCcdCase =
			isSceneCpuVolumeCapsuleReverseSweptCcdCase(caseName);
		const bool convexReverseSweptCcdCase =
			isSceneCpuVolumeConvexReverseSweptCcdCase(caseName);
		const bool rotationalCapsuleReverseSweptCcdCase =
			isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
				caseName);
		const bool rotationalConvexReverseSweptCcdCase =
			isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
				caseName);
		const bool rotationalFiniteReverseSweptCcdCase =
			rotationalCapsuleReverseSweptCcdCase ||
			rotationalConvexReverseSweptCcdCase;
		const bool staticSphereReverseSweptCcdCase =
			deformingReverseSweptCcdCase ||
			caseName ==
				"scene-volume-static-sphere-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-static-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-static-convex-reverse-swept-ccd";
		const bool kinematicSphereReverseSweptCcdCase =
			caseName ==
				"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-kinematic-convex-reverse-swept-ccd";
		const bool dynamicSphereReverseSweptCcdCase =
			caseName ==
				"scene-volume-dynamic-sphere-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-rotating-convex-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-convex-reverse-swept-ccd";
		const bool speculativeCcdCase =
			isSceneCpuVolumeSpeculativeCcdCase(caseName) ||
			sphereReverseSweptCcdCase;
		const bool planeSpeculativeCcdCase =
			caseName == "scene-volume-plane-speculative-ccd";
		const bool sphereSpeculativeCcdCase =
			caseName == "scene-volume-sphere-speculative-ccd";
		const bool capsuleSpeculativeCcdCase =
			caseName == "scene-volume-capsule-speculative-ccd";
		const bool convexSpeculativeCcdCase =
			caseName == "scene-volume-convex-speculative-ccd";
		const bool finiteSmoothSpeculativeCcdCase =
			sphereSpeculativeCcdCase || capsuleSpeculativeCcdCase ||
			convexSpeculativeCcdCase;
		const bool rotatingKinematicCapsuleSpeculativeCcdCase =
			caseName ==
				"scene-volume-rotating-kinematic-capsule-speculative-ccd";
		const bool rotatingKinematicConvexSpeculativeCcdCase =
			caseName ==
				"scene-volume-rotating-kinematic-convex-speculative-ccd";
		const bool dynamicRotatingCapsuleSpeculativeCcdCase =
			caseName ==
				"scene-volume-dynamic-rotating-capsule-relative-swept-ccd";
		const bool dynamicRotatingConvexSpeculativeCcdCase =
			caseName ==
				"scene-volume-dynamic-rotating-convex-relative-swept-ccd";
		const bool movingKinematicFiniteSpeculativeCcdCase =
			caseName ==
				"scene-volume-moving-kinematic-sphere-speculative-ccd" ||
			caseName ==
				"scene-volume-moving-kinematic-capsule-speculative-ccd" ||
			rotatingKinematicCapsuleSpeculativeCcdCase ||
			rotatingKinematicConvexSpeculativeCcdCase ||
			caseName ==
				"scene-volume-moving-kinematic-convex-speculative-ccd";
		const bool dynamicFiniteRelativeSweptCcdCase =
			caseName ==
				"scene-volume-dynamic-sphere-relative-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-capsule-relative-swept-ccd" ||
			dynamicRotatingCapsuleSpeculativeCcdCase ||
			dynamicRotatingConvexSpeculativeCcdCase ||
			caseName ==
				"scene-volume-dynamic-convex-relative-swept-ccd";
		const bool sphereReverseFeatureCase =
			caseName == "scene-volume-sphere-reverse-feature";
		const bool capsuleReverseFeatureCase =
			caseName == "scene-volume-capsule-reverse-feature";
		const bool convexReverseFeatureCase =
			caseName == "scene-volume-convex-reverse-feature";
		const bool triangleMeshReverseFeatureCase =
			caseName ==
				"scene-volume-triangle-mesh-reverse-feature";
		const bool heightFieldReverseFeatureCase =
			caseName ==
				"scene-volume-heightfield-reverse-feature";
		const bool smoothReverseFeatureCase =
			sphereReverseFeatureCase ||
			capsuleReverseFeatureCase ||
			convexReverseFeatureCase ||
			triangleMeshReverseFeatureCase ||
			heightFieldReverseFeatureCase;
		const bool skinningCase =
			caseName == "scene-volume-skinning";
		const bool twoSoftVolumeCase =
			multiSoftIslandCase || mixedSleepIslandCase ||
			softChurnCase || multiSceneIsolationCase ||
			softSoftWakeCase || softPairAttachmentCase ||
			motionControlsCase ||
			maxDepenetrationVelocityCase ||
			speculativeCcdCase || smoothReverseFeatureCase;
		const bool softSleepWakeCase =
			caseName == "scene-volume-sleep-wake";
		const bool softRigidWakeCase =
			caseName == "scene-volume-rigid-wake";
		const bool twoDynamicActorsCase =
			multiDynamicBoxCase || multiSoftIslandCase ||
			movingKinematicFiniteSpeculativeCcdCase ||
			dynamicFiniteRelativeSweptCcdCase ||
			kinematicSphereReverseSweptCcdCase ||
			dynamicSphereReverseSweptCcdCase ||
			(triangleSurfaceSweptCcdCase &&
			 !staticTriangleSurfaceSweptCcdCase);
		const bool dynamicBoxCase =
			caseName == "scene-volume-dynamic-box" ||
			caseName == "scene-volume-dynamic-sphere" ||
			caseName == "scene-volume-dynamic-capsule" ||
			caseName == "scene-volume-dynamic-convex" ||
			dynamicChurnCase || twoDynamicActorsCase;
		const bool dynamicSphereCase =
			caseName == "scene-volume-dynamic-sphere" ||
			caseName == "scene-volume-dynamic-capsule";
		const bool dynamicCapsuleCase =
			caseName == "scene-volume-dynamic-capsule";
		const bool dynamicConvexCase =
			caseName == "scene-volume-dynamic-convex";
		const bool dynamicSmoothCase =
			dynamicSphereCase || dynamicConvexCase;
		const bool commonPassed =
			gMetrics.initialized == 1 &&
			gMetrics.completedFrames == gHeadlessOptions.frames &&
			gMetrics.fetchFailures == 0 &&
			gMetrics.nonFiniteParticleSamples == 0 &&
			gMetrics.sceneActorCreated == 1 &&
			gMetrics.sceneShapeAttached == 1 &&
			gMetrics.sceneSimulationMeshAttached == 1 &&
			gMetrics.sceneHostBuffersInitialized == 1 &&
			gMetrics.sceneActorAdded == 1 &&
			gMetrics.sceneActorRemoved == 1 &&
			gMetrics.sceneActorReleased == 1 &&
			gMetrics.sceneBoundsFinite == 1 &&
			gMetrics.sceneDynamics ==
				(twoDynamicActorsCase ? 2u :
					(dynamicBoxCase || softRigidWakeCase ||
						kinematicBoxCase ||
						rigidAttachmentCase ||
						kinematicAttachmentCase ? 1u : 0u)) &&
			gMetrics.sceneDeformableVolumes ==
				(twoSoftVolumeCase ? 2u : 1u) &&
			gMetrics.particles > 0 &&
			gMetrics.softBodies ==
				(twoSoftVolumeCase ? 2u : 1u) &&
			gMetrics.tetElements > 0 &&
			gMetrics.solverReadbackMatched &&
			(softSleepWakeCase || softRigidWakeCase ||
				mixedSleepIslandCase || softChurnCase ||
				bufferMutationCase || worldPinCase ||
				attachmentCase || elementFilterCase ||
				kinematicBoxCase ||
				multiSceneIsolationCase ||
				softSoftWakeCase ||
				softPairAttachmentCase ||
				volumeKinematicTargetCase ||
				motionControlsCase ||
				maxDepenetrationVelocityCase ||
				speculativeCcdCase ||
				smoothReverseFeatureCase ||
				gMetrics.maxCentroidDrop > 0.0001f) &&
			PxIsFinite(gMetrics.finalMinY) &&
			PxIsFinite(gMetrics.finalMaxY) &&
			PxIsFinite(gMetrics.finalMaxParticleSpeed) &&
			gErrorCallback.getFatalCount() == 0 &&
			gMetrics.cleanupComplete == 1;
		if(!commonPassed)
			return false;
		if(skinningCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gVolumeSkinningMetrics.initialized == 1 &&
				gVolumeSkinningMetrics.evaluatedFrames ==
					gHeadlessOptions.frames &&
				gVolumeSkinningMetrics.finiteFrames ==
					gHeadlessOptions.frames &&
				gVolumeSkinningMetrics.vertices > 4 &&
				gVolumeSkinningMetrics.triangles >= 4 &&
				PxIsFinite(
					gVolumeSkinningMetrics.maxDisplacement) &&
				gVolumeSkinningMetrics.maxDisplacement > 0.05f;
		}
		if(motionControlsCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.motionMaxVelocityBounded == 1 &&
				gMetrics.motionSettlingApplied == 1 &&
				gMetrics.motionSettlingSlept == 1 &&
				gMetrics.motionControlStayedAwake == 1 &&
				gMetrics.
					motionMaxVelocityFirstStepDisplacement <=
						gHeadlessOptions.dt * 1.01f &&
				gMetrics.motionMaxVelocityFirstStepSpeed <= 1.02f &&
				gMetrics.motionSettlingFinalSpeed <= 1.0e-6f &&
				gMetrics.motionControlFinalSpeed >= 0.07f &&
				gMetrics.motionControlFinalSpeed <= 0.09f;
		}
		if(maxDepenetrationVelocityCase)
		{
			return
				gMetrics.sceneStatics == 1 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.depenetrationLimitApplied == 1 &&
				gMetrics.depenetrationFirstStepBounded == 1 &&
				gMetrics.depenetrationControlSeparated == 1 &&
				gMetrics.depenetrationGradualRecovery == 1 &&
				gMetrics.depenetrationLimitedFirstStepRise >=
					-1.0e-6f &&
				gMetrics.depenetrationLimitedFirstStepRise <=
					0.12f * gHeadlessOptions.dt * 1.25f &&
				gMetrics.depenetrationLimitedMaxSpeed <=
					0.25f;
		}
		if(smoothReverseFeatureCase)
		{
			return
				gMetrics.sceneStatics == 1 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gSphereReverseFeatureMetrics.
					faceResponseObserved == 1 &&
				gSphereReverseFeatureMetrics.vertexSdfExcluded == 1 &&
				gSphereReverseFeatureMetrics.
					negativeControlPassed == 1 &&
				gSphereReverseFeatureMetrics.nonFiniteSamples == 0 &&
				PxIsFinite(
					gSphereReverseFeatureMetrics.
						positiveDisplacement) &&
				gSphereReverseFeatureMetrics.
					positiveDisplacement > 1.0e-3f &&
				PxIsFinite(
					gSphereReverseFeatureMetrics.positiveDrop) &&
				PxIsFinite(
					gSphereReverseFeatureMetrics.negativeDrop) &&
				gSphereReverseFeatureMetrics.negativeDrop > 0.02f &&
				gSphereReverseFeatureMetrics.positiveDrop + 0.01f <
					gSphereReverseFeatureMetrics.negativeDrop &&
				PxIsFinite(
					gSphereReverseFeatureMetrics.faceSeparation) &&
				gSphereReverseFeatureMetrics.faceSeparation > 0.02f &&
				PxIsFinite(
					gSphereReverseFeatureMetrics.
						minimumVertexSeparation) &&
				gSphereReverseFeatureMetrics.
					minimumVertexSeparation > 0.10f;
		}
		if(triangleSurfaceSweptCcdCase)
		{
			const bool rigidLifecyclePassed =
				staticTriangleSurfaceSweptCcdCase
					? gMetrics.sceneStatics == 1 &&
						gMetrics.sceneDynamics == 0
					: gMetrics.sceneStatics == 0 &&
						gMetrics.sceneDynamics == 2 &&
						gMetrics.sceneDynamicActorAdded == 1 &&
						gMetrics.sceneSecondDynamicActorAdded == 1 &&
						gMetrics.sceneDynamicActorRemoved == 1 &&
						gMetrics.sceneSecondDynamicActorRemoved == 1 &&
						gMetrics.sceneDynamicActorReleased == 1 &&
						gMetrics.sceneSecondDynamicActorReleased == 1;
			return
				rigidLifecyclePassed &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.speculativeCcdFlagApplied == 1 &&
				gMetrics.speculativeCcdPreventedTunneling == 1 &&
				gMetrics.speculativeCcdNegativeControlTunneled == 1 &&
				gSphereReverseSweptMetrics.responseObserved == 1 &&
				gSphereReverseSweptMetrics.
					negativeControlPassed == 1 &&
				gSphereReverseSweptMetrics.
					twoSidedResponseObserved == 1 &&
				gSphereReverseSweptMetrics.vertexSweepExcluded == 1 &&
				gSphereReverseSweptMetrics.nonFiniteSamples == 0 &&
				PxIsFinite(
					gSphereReverseSweptMetrics.
						positiveDisplacement) &&
				PxIsFinite(
					gSphereReverseSweptMetrics.
						negativeDisplacement) &&
				PxIsFinite(
					gSphereReverseSweptMetrics.positiveDrop) &&
				PxIsFinite(
					gSphereReverseSweptMetrics.negativeDrop) &&
				(!triangleSurfaceReverseSweptCcdCase ||
					(PxIsFinite(
						gSphereReverseSweptMetrics.
							minimumVertexSweepSeparation) &&
					 gSphereReverseSweptMetrics.
							minimumVertexSweepSeparation >
							(rotationalTriangleSurfaceSweptCcdCase
								? 0.10f : 0.05f))) &&
				(!rotationalTriangleSurfaceSweptCcdCase ||
					(gCapsuleRotationalSweepMetrics.
							sweepIsolated == 1 &&
					 gCapsuleRotationalSweepMetrics.
							nonFiniteSamples == 0 &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation) &&
					 gCapsuleRotationalSweepMetrics.
							endpointMinSeparation > 0.10f &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation) &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							positiveAngularTravel) &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							negativeAngularTravel) &&
					 PxAbs(
						gCapsuleRotationalSweepMetrics.
							positiveAngularTravel -
							2.0f * PxPi / 3.0f) < 0.002f &&
					 PxAbs(
						gCapsuleRotationalSweepMetrics.
							negativeAngularTravel -
							2.0f * PxPi / 3.0f) < 0.002f)) &&
				(staticTriangleSurfaceSweptCcdCase
					? gSphereReverseSweptMetrics.negativeDrop >
							(triangleSurfaceReverseSweptCcdCase &&
							 heightFieldSweptCcdCase
								? 0.8f : 1.5f) &&
						gSphereReverseSweptMetrics.positiveDrop +
								(triangleSurfaceReverseSweptCcdCase
									? 0.01f : 0.10f) <
							gSphereReverseSweptMetrics.negativeDrop
					: gSphereReverseSweptMetrics.
							positiveDisplacement >
								(rotationalTriangleSurfaceSweptCcdCase
									? 2.0e-3f
									: triangleSurfaceReverseSweptCcdCase
										? 5.0e-3f : 0.02f) &&
						gSphereReverseSweptMetrics.
							negativeDisplacement < 1.0e-2f);
		}
		if(sphereReverseSweptCcdCase)
		{
			const bool rigidLifecyclePassed =
				staticSphereReverseSweptCcdCase
					? gMetrics.sceneStatics == 1 &&
						gMetrics.sceneDynamics == 0
					: gMetrics.sceneStatics == 0 &&
						gMetrics.sceneDynamics == 2 &&
						gMetrics.sceneDynamicActorAdded == 1 &&
						gMetrics.sceneSecondDynamicActorAdded == 1 &&
						gMetrics.sceneDynamicActorRemoved == 1 &&
						gMetrics.sceneSecondDynamicActorRemoved == 1 &&
						gMetrics.sceneDynamicActorReleased == 1 &&
						gMetrics.sceneSecondDynamicActorReleased == 1;
			return
				rigidLifecyclePassed &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.speculativeCcdFlagApplied == 1 &&
				gMetrics.speculativeCcdPreventedTunneling == 1 &&
				gMetrics.speculativeCcdNegativeControlTunneled == 1 &&
				gSphereReverseSweptMetrics.responseObserved == 1 &&
				gSphereReverseSweptMetrics.
					negativeControlPassed == 1 &&
				gSphereReverseSweptMetrics.
					twoSidedResponseObserved == 1 &&
				gSphereReverseSweptMetrics.vertexSweepExcluded == 1 &&
				gSphereReverseSweptMetrics.nonFiniteSamples == 0 &&
				PxIsFinite(
					gSphereReverseSweptMetrics.
						positiveDisplacement) &&
				PxIsFinite(
					gSphereReverseSweptMetrics.
						negativeDisplacement) &&
				PxIsFinite(
					gSphereReverseSweptMetrics.positiveDrop) &&
				PxIsFinite(
					gSphereReverseSweptMetrics.negativeDrop) &&
				PxIsFinite(
					gSphereReverseSweptMetrics.
						positiveRigidDrop) &&
				PxIsFinite(
					gSphereReverseSweptMetrics.
						negativeRigidDrop) &&
				PxIsFinite(
					gSphereReverseSweptMetrics.faceSeparation) &&
				gSphereReverseSweptMetrics.faceSeparation >
					-0.15f &&
				PxIsFinite(
					gSphereReverseSweptMetrics.
						minimumVertexSweepSeparation) &&
				gSphereReverseSweptMetrics.
					minimumVertexSweepSeparation >
						(rotationalFiniteReverseSweptCcdCase
							? 0.10f :
						((capsuleReverseSweptCcdCase ||
						  convexReverseSweptCcdCase)
							? 0.05f : 0.10f)) &&
				(!deformingReverseSweptCcdCase ||
					(gDeformingVolumeReverseSweptMetrics.
							geometricSweepIsolated == 1 &&
					 PxIsFinite(
						gDeformingVolumeReverseSweptMetrics.
							endpointMinSeparation) &&
					 PxIsFinite(
						gDeformingVolumeReverseSweptMetrics.
							midSweepMinSeparation) &&
					 PxIsFinite(
						gDeformingVolumeReverseSweptMetrics.
							responseDelta) &&
					 gDeformingVolumeReverseSweptMetrics.
							responseDelta > 0.01f)) &&
				(!rotationalFiniteReverseSweptCcdCase ||
					(gCapsuleRotationalSweepMetrics.sweepIsolated == 1 &&
					 gCapsuleRotationalSweepMetrics.nonFiniteSamples == 0 &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation) &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation) &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							positiveAngularTravel) &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							negativeAngularTravel) &&
					 (!kinematicSphereReverseSweptCcdCase ||
						(PxAbs(
							gCapsuleRotationalSweepMetrics.
								positiveAngularTravel -
							2.0f * PxPi / 3.0f) < 0.002f &&
						 PxAbs(
							gCapsuleRotationalSweepMetrics.
								negativeAngularTravel -
							2.0f * PxPi / 3.0f) < 0.002f)))) &&
				(staticSphereReverseSweptCcdCase
					? gSphereReverseSweptMetrics.negativeDrop >
							(deformingReverseSweptCcdCase
								? 0.15f : 0.8f) &&
						gSphereReverseSweptMetrics.positiveDrop +
								(deformingReverseSweptCcdCase
									? 0.01f : 0.03f) <
							gSphereReverseSweptMetrics.negativeDrop
					: gSphereReverseSweptMetrics.
							positiveDisplacement >
								(rotationalFiniteReverseSweptCcdCase
									? 0.02f :
								 dynamicSphereReverseSweptCcdCase
									? 0.01f : 0.02f) &&
						gSphereReverseSweptMetrics.
							negativeDisplacement < 5.0e-3f) &&
				(!dynamicSphereReverseSweptCcdCase ||
					(gSphereReverseSweptMetrics.negativeRigidDrop >
							(rotationalFiniteReverseSweptCcdCase
								? 0.8f : 1.5f) &&
					 gSphereReverseSweptMetrics.positiveRigidDrop +
							0.05f <
						gSphereReverseSweptMetrics.
							negativeRigidDrop));
		}
		if(dynamicFiniteRelativeSweptCcdCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 2 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.sceneDynamicActorAdded == 1 &&
				gMetrics.sceneSecondDynamicActorAdded == 1 &&
				gMetrics.sceneDynamicActorRemoved == 1 &&
				gMetrics.sceneSecondDynamicActorRemoved == 1 &&
				gMetrics.sceneDynamicActorReleased == 1 &&
				gMetrics.sceneSecondDynamicActorReleased == 1 &&
				gMetrics.speculativeCcdFlagApplied == 1 &&
				gMetrics.dynamicSphereSweepLaunched == 1 &&
				gMetrics.dynamicSphereSweepResponseObserved == 1 &&
				gMetrics.
					dynamicSphereSweepNegativeControlTunneled == 1 &&
				gMetrics.
					dynamicSphereSweepTwoSidedResponseObserved == 1 &&
				PxIsFinite(
					gMetrics.
						dynamicSphereSweepPositiveSoftDisplacement) &&
				gMetrics.
					dynamicSphereSweepPositiveSoftDisplacement >
						0.02f &&
				PxIsFinite(
					gMetrics.
						dynamicSphereSweepNegativeSoftDisplacement) &&
				gMetrics.
					dynamicSphereSweepNegativeSoftDisplacement <
						5.0e-3f &&
				PxIsFinite(
					gMetrics.dynamicSphereSweepPositiveRigidDrop) &&
				PxIsFinite(
					gMetrics.dynamicSphereSweepNegativeRigidDrop) &&
				gMetrics.dynamicSphereSweepNegativeRigidDrop >
					((dynamicRotatingCapsuleSpeculativeCcdCase ||
					  dynamicRotatingConvexSpeculativeCcdCase)
						? 0.8f : 1.5f) &&
				gMetrics.dynamicSphereSweepPositiveRigidDrop +
						0.05f <
					gMetrics.dynamicSphereSweepNegativeRigidDrop &&
				PxIsFinite(
					gMetrics.
						dynamicSphereSweepPositiveMinSeparation) &&
				gMetrics.dynamicSphereSweepPositiveMinSeparation >
					-0.15f &&
				gMetrics.dynamicSphereSweepPositiveMinSeparation <
					PX_MAX_F32 &&
				(!(dynamicRotatingCapsuleSpeculativeCcdCase ||
				   dynamicRotatingConvexSpeculativeCcdCase) ||
					(gCapsuleRotationalSweepMetrics.sweepIsolated == 1 &&
					 gCapsuleRotationalSweepMetrics.nonFiniteSamples == 0 &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation) &&
					 gCapsuleRotationalSweepMetrics.
							endpointMinSeparation > 0.05f &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation) &&
					 gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation <
								(dynamicRotatingConvexSpeculativeCcdCase
									? 1.0e-5f : -0.05f) &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							positiveAngularTravel) &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							negativeAngularTravel) &&
					 gCapsuleRotationalSweepMetrics.
							negativeAngularTravel > 0.8f &&
					 gCapsuleRotationalSweepMetrics.
								positiveAngularTravel +
							0.05f <
						gCapsuleRotationalSweepMetrics.
							negativeAngularTravel));
		}
		if(movingKinematicFiniteSpeculativeCcdCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 2 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.sceneDynamicActorAdded == 1 &&
				gMetrics.sceneSecondDynamicActorAdded == 1 &&
				gMetrics.sceneDynamicActorRemoved == 1 &&
				gMetrics.sceneSecondDynamicActorRemoved == 1 &&
				gMetrics.sceneDynamicActorReleased == 1 &&
				gMetrics.sceneSecondDynamicActorReleased == 1 &&
				gMetrics.speculativeCcdFlagApplied == 1 &&
				gMetrics.movingSphereTargetIssued == 1 &&
				gMetrics.movingSphereCcdResponseObserved == 1 &&
				gMetrics.movingSphereNegativeControlHeld == 1 &&
				PxIsFinite(
					gMetrics.movingSpherePositiveDisplacement) &&
				gMetrics.movingSpherePositiveDisplacement >
					0.02f &&
				PxIsFinite(
					gMetrics.movingSphereNegativeDisplacement) &&
				gMetrics.movingSphereNegativeDisplacement <
					5.0e-3f &&
				PxIsFinite(
					gMetrics.
						movingSpherePositiveMinSeparation) &&
				gMetrics.movingSpherePositiveMinSeparation >
					-0.10f &&
				gMetrics.movingSpherePositiveMinSeparation <
					PX_MAX_F32 &&
				(!(rotatingKinematicCapsuleSpeculativeCcdCase ||
				   rotatingKinematicConvexSpeculativeCcdCase) ||
					(gCapsuleRotationalSweepMetrics.sweepIsolated == 1 &&
					 gCapsuleRotationalSweepMetrics.nonFiniteSamples == 0 &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation) &&
					 gCapsuleRotationalSweepMetrics.
							endpointMinSeparation > 0.05f &&
					 PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation) &&
					 gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation <
								(rotatingKinematicConvexSpeculativeCcdCase
									? 1.0e-5f : -0.05f)));
		}
		if(speculativeCcdCase)
		{
			return
				gMetrics.sceneStatics == 1 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.speculativeCcdFlagApplied == 1 &&
				gMetrics.speculativeCcdPreventedTunneling == 1 &&
				(planeSpeculativeCcdCase ||
					gMetrics.
						speculativeCcdNegativeControlTunneled == 1) &&
				PxIsFinite(
					gMetrics.speculativeCcdPositiveMinY) &&
				(finiteSmoothSpeculativeCcdCase
					? PxIsFinite(
						gMetrics.
							speculativeCcdPositiveMinSeparation) &&
						gMetrics.
							speculativeCcdPositiveMinSeparation >=
								-0.05f &&
						gMetrics.
							speculativeCcdPositiveMinSeparation <
								PX_MAX_F32
					: gMetrics.speculativeCcdPositiveMinY >=
						(planeSpeculativeCcdCase
							? 0.49f : 0.50f)) &&
				PxIsFinite(
					gMetrics.speculativeCcdNegativeMaxY) &&
				(planeSpeculativeCcdCase ||
					gMetrics.speculativeCcdNegativeMaxY <= 0.44f);
		}
		if(volumeKinematicTargetCase)
		{
			const bool commonTargetPassed =
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneVolumeTargetBound == 1 &&
				gMetrics.sceneVolumeTargetMutated == 1 &&
				gMetrics.sceneVolumeTargetWoke == 1 &&
				gMetrics.sceneVolumeTargetReached == 1 &&
				PxIsFinite(
					gMetrics.
						sceneVolumeTargetFinalMaxError) &&
				gMetrics.sceneVolumeTargetFinalMaxError <=
					5.0e-3f &&
				gMetrics.
					sceneVolumeTargetMaxDisplacement > 0.2f &&
				gMetrics.maxParticleSpeed < 5.0f &&
				gMetrics.finalMaxParticleSpeed < 0.5f;
			if(!commonTargetPassed)
				return false;
			if(fullKinematicTargetCase)
				return true;
			return
				gMetrics.
					sceneVolumePartialInactiveIgnored == 1 &&
				gMetrics.sceneVolumePartialActivated == 1 &&
				gMetrics.
					sceneVolumePartialActivatedReached == 1 &&
				PxIsFinite(
					gMetrics.
						sceneVolumePartialInactiveDecoyDistance) &&
				gMetrics.
					sceneVolumePartialInactiveDecoyDistance >
						2.0f;
		}
		if(multiSceneIsolationCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSecondSceneCreated == 1 &&
				gMetrics.sceneSecondSceneSolverMatched == 1 &&
				gMetrics.scenePrimarySceneReleased == 1 &&
				gMetrics.sceneSecondSceneReleased == 1 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.sceneSoftFirstSlept == 1 &&
				gMetrics.sceneSoftFirstSleepFrame < 60 &&
				gMetrics.sceneMultiPrimaryStable == 1 &&
				gMetrics.sceneMultiPrimaryDetachedStable == 1 &&
				gMetrics.sceneMultiSecondaryUpdatedBeforeRelease == 1 &&
				gMetrics.sceneMultiSecondaryUpdatedAfterRelease == 1 &&
				gMetrics.sceneSecondVolumeMaxCentroidDrop > 0.1f &&
				gMetrics.finalMinY > 3.5f &&
				gMetrics.maxParticleSpeed < 0.3f &&
				gMetrics.finalMaxParticleSpeed < 1.0e-6f;
		}
		if(softSoftWakeCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.sceneSoftFirstSlept == 1 &&
				gMetrics.sceneSoftSoftBothSlept == 1 &&
				gMetrics.sceneSoftSoftDriveIssued == 1 &&
				gMetrics.sceneSoftSoftDriverWoke == 1 &&
				gMetrics.sceneSoftSoftTargetWoke == 1 &&
				gMetrics.sceneSoftSoftTargetWakeFrame >
					gMetrics.sceneSoftFirstSleepFrame &&
				gMetrics.sceneSoftSoftTargetWakeFrame <
					gMetrics.completedFrames &&
				gMetrics.sceneSoftSoftTargetMoved == 1 &&
				gMetrics.sceneSoftSoftResetIssued == 1 &&
				gMetrics.sceneSoftSoftBothFinalSlept == 1 &&
				gMetrics.finalMinY > 3.5f &&
				gMetrics.maxParticleSpeed < 10.0f &&
				gMetrics.finalMaxParticleSpeed < 1.0e-6f;
		}
		if(softPairAttachmentCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.sceneRigidAttachmentActorAdded == 1 &&
				gMetrics.sceneRigidAttachmentCreated == 1 &&
				gMetrics.sceneRigidAttachmentRigidWoke == 1 &&
				gMetrics.sceneRigidAttachmentRigidMoved == 1 &&
				gMetrics.
					sceneRigidAttachmentHeldAcrossReadd == 1 &&
				gMetrics.sceneRigidAttachmentReleased == 1 &&
				gMetrics.
					sceneRigidAttachmentSeparatedAfterRelease == 1 &&
				gMetrics.sceneRigidAttachmentMaxDrift < 0.05f &&
				gMetrics.sceneRigidAttachmentMaxRigidSpeed < 10.0f &&
				gMetrics.
					sceneRigidAttachmentMaxRigidDisplacement >
						0.02f &&
				gMetrics.
					sceneRigidAttachmentReleasedSeparation > 0.2f &&
				gMetrics.maxParticleSpeed < 10.0f &&
				gMetrics.finalMaxParticleSpeed < 2.0f;
		}
		if(kinematicBoxCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneKinematicActorAdded == 1 &&
				gMetrics.sceneSoftFirstSlept == 1 &&
				gMetrics.sceneKinematicTargetIssued == 1 &&
				gMetrics.sceneKinematicTargetReached == 1 &&
				gMetrics.sceneKinematicSoftWoke == 1 &&
				gMetrics.sceneKinematicSoftMoved == 1 &&
				gMetrics.sceneKinematicContactObserved == 1 &&
				gMetrics.sceneKinematicMaxPoseError <= 1.0e-4f &&
				gMetrics.sceneKinematicSoftDisplacement > 0.02f &&
				PxIsFinite(gMetrics.sceneKinematicFinalY) &&
				PxAbs(
					gMetrics.sceneKinematicFinalY - 4.10f) <=
						1.0e-4f &&
				gMetrics.maxParticleSpeed < 2.0f &&
				gMetrics.finalMaxParticleSpeed < 0.5f &&
				gMetrics.sceneDynamicActorRemoved == 1 &&
				gMetrics.sceneDynamicActorReleased == 1;
		}
		if(dynamicBoxCase)
		{
			const bool dynamicPassed =
				gMetrics.sceneStatics ==
					(dynamicSmoothCase ? 1u : 0u) &&
				gMetrics.rigidContactFrames > 0 &&
				gMetrics.maxRigidContacts > 0 &&
				gMetrics.maxCentroidDrop > 0.5f &&
				gMetrics.sceneDynamicActorRemoved == 1 &&
				gMetrics.sceneDynamicActorReleased == 1 &&
				gMetrics.sceneDynamicInitiallySleeping == 1 &&
				gMetrics.sceneDynamicWokeBySoft == 1 &&
				gMetrics.sceneDynamicFirstWakeFrame != PX_MAX_U32 &&
				PxIsFinite(gMetrics.sceneDynamicMinY) &&
				PxIsFinite(gMetrics.sceneDynamicFinalY) &&
				gMetrics.sceneDynamicMaxDrop > 0.05f &&
				gSceneCpuDynamicInitialY -
					gMetrics.sceneDynamicFinalY > 0.05f &&
				(dynamicCapsuleCase || dynamicConvexCase ||
					gMetrics.sceneDynamicPreContactMaxDrop <
						1.0e-4f) &&
				gMetrics.sceneDynamicMaxDownSpeed > 0.01f &&
				PxIsFinite(
					gMetrics.minDynamicSurfaceSeparation) &&
				gMetrics.minDynamicSurfaceSeparation > -0.15f &&
				PxIsFinite(
					gMetrics.finalDynamicSurfaceSeparation) &&
				gMetrics.finalDynamicSurfaceSeparation > -0.15f &&
				(!dynamicSmoothCase ||
					(gMetrics.finalMinY > -0.15f &&
					 gMetrics.maxParticleSpeed < 10.0f &&
					 gMetrics.finalMaxParticleSpeed < 0.5f &&
					 gMetrics.sceneDynamicMaxDownSpeed < 5.0f &&
					 gMetrics.sceneDynamicFinalY > 0.70f &&
					 gMetrics.sceneDynamicFinalY < 0.90f));
			if(!dynamicPassed)
				return false;
			if(twoDynamicActorsCase)
			{
				const bool secondDynamicPassed =
					gMetrics.sceneSecondDynamicActorAdded == 1 &&
					gMetrics.sceneSecondDynamicActorRemoved == 1 &&
					gMetrics.sceneSecondDynamicActorReleased == 1 &&
					gMetrics.sceneSecondDynamicInitiallySleeping == 1 &&
					gMetrics.sceneSecondDynamicWokeBySoft == 1 &&
					gMetrics.sceneSecondDynamicFirstWakeFrame !=
						PX_MAX_U32 &&
					gMetrics.sceneSecondDynamicFirstWakeFrame ==
						gMetrics.sceneDynamicFirstWakeFrame &&
					PxIsFinite(gMetrics.sceneSecondDynamicMinY) &&
					PxIsFinite(gMetrics.sceneSecondDynamicFinalY) &&
					gMetrics.sceneSecondDynamicMaxDrop > 0.01f &&
					gMetrics.sceneSecondDynamicPreContactMaxDrop <
						1.0e-4f &&
					gMetrics.sceneSecondDynamicMaxDownSpeed > 0.01f &&
					gMetrics.sceneDynamicMaxDownSpeed < 6.0f &&
					gMetrics.sceneSecondDynamicMaxDownSpeed < 6.0f;
				if(!secondDynamicPassed)
					return false;
				if(!multiSoftIslandCase)
					return true;
				return
					gMetrics.sceneSecondVolumeActorCreated == 1 &&
					gMetrics.
						sceneSecondVolumeHostBuffersInitialized == 1 &&
					gMetrics.sceneSecondVolumeActorAdded == 1 &&
					gMetrics.sceneSecondVolumeActorRemoved == 1 &&
					gMetrics.sceneSecondVolumeActorReleased == 1 &&
					gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
					gMetrics.
						sceneSecondVolumeMaxCentroidDrop > 0.5f &&
					PxIsFinite(
						gMetrics.
							sceneSecondVolumeFinalCentroidY) &&
					gMetrics.finalMinY > -10.0f &&
					gMetrics.finalMaxParticleSpeed < 1.0e-4f;
			}
			if(!dynamicChurnCase)
				return true;
			return
				gMetrics.sceneDynamicShapeDetached == 1 &&
				gMetrics.sceneDynamicShapeReattached == 1 &&
				gMetrics.sceneDynamicActorRemoved == 1 &&
				gMetrics.sceneDynamicActorReadded == 1 &&
				gMetrics.sceneDynamicReaddedSleeping == 1 &&
				gMetrics.sceneDynamicRewokeBySoft == 1 &&
				gMetrics.sceneDynamicSecondWakeFrame >
					gMetrics.sceneDynamicFirstWakeFrame &&
				gMetrics.sceneDynamicSecondWakeFrame <
					gMetrics.completedFrames;
		}
		if(softSleepWakeCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneSoftInitiallyAwake == 1 &&
				gMetrics.sceneSoftFirstSlept == 1 &&
				gMetrics.sceneSoftFirstSleepFrame <
					gMetrics.completedFrames &&
				gMetrics.sceneSoftSleepWakeCounterZero == 1 &&
				gMetrics.sceneSoftSleepVelocitiesZero == 1 &&
				gMetrics.sceneSoftStableWhileSleeping == 1 &&
				gMetrics.sceneSoftCounterWakeIssued == 1 &&
				gMetrics.sceneSoftWokeByCounter == 1 &&
				gMetrics.sceneSoftCounterWakeFrame >
					gMetrics.sceneSoftFirstSleepFrame &&
				gMetrics.sceneSoftSecondSlept == 1 &&
				gMetrics.sceneSoftSecondSleepFrame >
					gMetrics.sceneSoftCounterWakeFrame &&
				gMetrics.sceneSoftVelocityWakeIssued == 1 &&
				gMetrics.sceneSoftWokeByVelocity == 1 &&
				gMetrics.sceneSoftVelocityWakeFrame >
					gMetrics.sceneSoftSecondSleepFrame &&
				gMetrics.sceneSoftMovedAfterVelocityWake == 1 &&
				gMetrics.sceneSoftVelocityStopIssued == 1 &&
				gMetrics.sceneSoftFinalSlept == 1 &&
				gMetrics.sceneSoftFinalSleepFrame >
					gMetrics.sceneSoftVelocityWakeFrame &&
				gMetrics.maxParticleSpeed < 3.0f &&
				gMetrics.finalMaxParticleSpeed < 1.0e-6f;
		}
		if(softRigidWakeCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneSoftInitiallyAwake == 1 &&
				gMetrics.sceneSoftFirstSlept == 1 &&
				gMetrics.sceneSoftFirstSleepFrame <
					gMetrics.completedFrames &&
				gMetrics.sceneSoftSleepWakeCounterZero == 1 &&
				gMetrics.sceneSoftSleepVelocitiesZero == 1 &&
				gMetrics.sceneSoftStableWhileSleeping == 1 &&
				gMetrics.sceneSoftRigidWakeActorAdded == 1 &&
				gMetrics.sceneDynamicActorAdded == 1 &&
				gMetrics.sceneDynamicInitiallySleeping == 0 &&
				gMetrics.sceneSoftWokeByRigid == 1 &&
				gMetrics.sceneSoftRigidWakeFrame >
					gMetrics.sceneSoftFirstSleepFrame &&
				gMetrics.sceneSoftVelocityStopIssued == 1 &&
				gMetrics.sceneSoftFinalSlept == 1 &&
				gMetrics.sceneSoftFinalSleepFrame >
					gMetrics.sceneSoftRigidWakeFrame &&
				gMetrics.rigidContactFrames > 0 &&
				gMetrics.maxRigidContacts > 0 &&
				gMetrics.finalMinY > 3.5f &&
				gMetrics.maxParticleSpeed < 2.0f &&
				gMetrics.finalMaxParticleSpeed < 1.0e-6f &&
				gMetrics.sceneDynamicActorRemoved == 1 &&
				gMetrics.sceneDynamicActorReleased == 1;
		}
		if(bufferMutationCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSoftInitiallyAwake == 1 &&
				gMetrics.sceneSoftFirstSlept == 1 &&
				gMetrics.sceneSoftFirstSleepFrame <
					gMetrics.completedFrames &&
				gMetrics.sceneSoftSleepWakeCounterZero == 1 &&
				gMetrics.sceneSoftSleepVelocitiesZero == 1 &&
				gMetrics.sceneSoftStableWhileSleeping == 1 &&
				gMetrics.sceneBufferMutationIssued == 1 &&
				gMetrics.sceneBufferMutationWoke == 1 &&
				gMetrics.sceneBufferMutationApplied == 1 &&
				gMetrics.sceneBufferDriveIssued == 1 &&
				gMetrics.sceneBufferPinHeld == 1 &&
				gMetrics.sceneBufferDynamicMoved == 1 &&
				gMetrics.sceneBufferInvMassRestored == 1 &&
				gMetrics.sceneBufferRestoredMoved == 1 &&
				gMetrics.sceneBufferResetIssued == 1 &&
				gMetrics.sceneSoftFinalSlept == 1 &&
				gMetrics.sceneSoftFinalSleepFrame >
					gMetrics.sceneSoftFirstSleepFrame &&
				gMetrics.sceneSoftFinalSleepFrame <
					gMetrics.completedFrames &&
				gMetrics.finalMinY > 4.0f &&
				gMetrics.maxParticleSpeed < 3.0f &&
				gMetrics.finalMaxParticleSpeed < 1.0e-6f;
		}
		if(worldPinCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneWorldPinCreated == 1 &&
				gMetrics.sceneWorldPinHeld == 1 &&
				gMetrics.sceneWorldPinActorReadded == 1 &&
				gMetrics.sceneWorldPinReleased == 1 &&
				gMetrics.sceneWorldPinMovedAfterRelease == 1 &&
				gMetrics.sceneWorldPinMaxDrift <= 1.0e-4f &&
				gMetrics.
					sceneWorldPinReleasedMaxDisplacement >
						1.0e-3f;
		}
		if(attachmentCase)
		{
			if(articulationAttachmentCase)
			{
				return
					gMetrics.sceneStatics == 0 &&
					gMetrics.sceneDynamics == 0 &&
					gMetrics.sceneArticulationCreated == 1 &&
					gMetrics.sceneArticulationAdded == 1 &&
					gMetrics.sceneArticulationInitiallySleeping == 1 &&
					gMetrics.sceneArticulationWoke == 1 &&
					gMetrics.sceneArticulationJointSubspaceHeld == 1 &&
					gMetrics.sceneArticulationRootStable == 1 &&
					gMetrics.sceneRigidAttachmentActorAdded == 1 &&
					gMetrics.
						sceneRigidAttachmentInitiallySleeping == 1 &&
					gMetrics.sceneRigidAttachmentCreated == 1 &&
					gMetrics.sceneRigidAttachmentRigidMoved == 1 &&
					gMetrics.
						sceneRigidAttachmentHeldAcrossReadd == 1 &&
					gMetrics.sceneRigidAttachmentReleased == 1 &&
					gMetrics.
						sceneRigidAttachmentSeparatedAfterRelease == 1 &&
					gMetrics.sceneRigidAttachmentMaxDrift < 0.05f &&
					gMetrics.sceneRigidAttachmentMaxRigidSpeed < 5.0f &&
					gMetrics.
						sceneRigidAttachmentMaxRigidDisplacement >
							0.02f &&
					gMetrics.
						sceneRigidAttachmentReleasedSeparation > 0.2f &&
					gMetrics.
						sceneArticulationRootMaxDisplacement <=
							1.0e-4f &&
					gMetrics.
						sceneArticulationChildMaxForbiddenDisplacement <=
							1.0e-3f &&
					gMetrics.
						sceneArticulationChildMaxAngularDisplacement <=
							1.0e-3f &&
					gMetrics.maxParticleSpeed < 20.0f &&
					gMetrics.finalMaxParticleSpeed < 2.0f;
			}
			return
				gMetrics.sceneStatics ==
					(staticAttachmentCase ? 1u : 0u) &&
				gMetrics.sceneDynamics ==
					(staticAttachmentCase ? 0u : 1u) &&
				gMetrics.sceneRigidAttachmentActorAdded == 1 &&
				gMetrics.
					sceneRigidAttachmentInitiallySleeping == 1 &&
				gMetrics.sceneRigidAttachmentCreated == 1 &&
				(!rigidAttachmentCase ||
					gMetrics.sceneRigidAttachmentRigidWoke == 1) &&
				gMetrics.sceneRigidAttachmentRigidMoved == 1 &&
				gMetrics.
					sceneRigidAttachmentHeldAcrossReadd == 1 &&
				gMetrics.sceneRigidAttachmentReleased == 1 &&
				gMetrics.
					sceneRigidAttachmentSeparatedAfterRelease == 1 &&
				gMetrics.sceneRigidAttachmentMaxDrift < 0.05f &&
				gMetrics.sceneRigidAttachmentMaxRigidSpeed < 5.0f &&
				gMetrics.sceneRigidAttachmentMaxRigidDisplacement >
					0.02f &&
				(!(kinematicAttachmentCase ||
					staticAttachmentCase) ||
					(gMetrics.sceneKinematicActorAdded == 1 &&
					 gMetrics.sceneSoftFirstSlept == 1 &&
					 gMetrics.sceneKinematicTargetIssued == 1 &&
					 gMetrics.sceneKinematicTargetReached == 1 &&
					 gMetrics.sceneKinematicSoftWoke == 1 &&
					 gMetrics.sceneKinematicSoftMoved == 1 &&
					 gMetrics.sceneKinematicMaxPoseError <=
						1.0e-4f &&
					 gMetrics.sceneKinematicSoftDisplacement >
						0.02f &&
					 gMetrics.maxParticleSpeed < 20.0f &&
					 gMetrics.finalMaxParticleSpeed < 2.0f)) &&
				gMetrics.
					sceneRigidAttachmentReleasedSeparation > 0.2f &&
				(staticAttachmentCase ||
				 (gMetrics.sceneDynamicActorRemoved == 1 &&
				  gMetrics.sceneDynamicActorReleased == 1));
		}
		if(elementFilterCase)
		{
			gMetrics.sceneElementFilterContactRestored =
				gMetrics.sceneElementFilterReleased == 1 &&
				PxIsFinite(gMetrics.sceneElementFilterFinalMinY) &&
				gMetrics.sceneElementFilterFinalMinY > -0.05f &&
				gMetrics.sceneElementFilterFinalMinY < 0.05f &&
				gMetrics.finalMaxParticleSpeed < 0.1f
					? 1u : 0u;
			gMetrics.scenePartialFilterExactOwnership =
				partialElementFilterCase &&
				gMetrics.sceneElementFilterSuppressedContact == 1 &&
				gMetrics.
					scenePartialFilterUnfilteredContactHeld == 1 &&
				PxIsFinite(
					gMetrics.scenePartialFilterUnfilteredMinY) &&
				gMetrics.scenePartialFilterUnfilteredMinY > -0.05f
					? 1u : 0u;
			return
				gMetrics.sceneStatics == 1 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneElementFilterCreated == 1 &&
				gMetrics.sceneElementFilterActorReadded == 1 &&
				gMetrics.
					sceneElementFilterSuppressedContact == 1 &&
				gMetrics.sceneElementFilterReleased == 1 &&
				gMetrics.sceneElementFilterContactRestored == 1 &&
				gMetrics.sceneElementFilterMinY < -0.2f &&
				gMetrics.sceneElementFilterFinalMinY > -0.05f &&
				gMetrics.sceneElementFilterFinalMinY < 0.05f &&
				(!partialElementFilterCase ||
					(gMetrics.
						scenePartialFilterExactOwnership == 1 &&
					 gMetrics.
						scenePartialFilterUnfilteredMinY > -0.05f));
		}
		if(mixedSleepIslandCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.sceneMixedFirstSlept == 1 &&
				gMetrics.sceneMixedFirstSleepFrame <
					gMetrics.completedFrames &&
				gMetrics.sceneMixedFirstStable == 1 &&
				gMetrics.sceneMixedSecondStayedAwake == 1 &&
				gMetrics.sceneMixedSecondMoved == 1 &&
				gMetrics.finalMinY > 3.5f &&
				gMetrics.finalMaxParticleSpeed < 0.3f;
		}
		if(softChurnCase)
		{
			return
				gMetrics.sceneStatics == 0 &&
				gMetrics.sceneDynamics == 0 &&
				gMetrics.sceneSecondVolumeActorCreated == 1 &&
				gMetrics.
					sceneSecondVolumeHostBuffersInitialized == 1 &&
				gMetrics.sceneSecondVolumeActorAdded == 1 &&
				gMetrics.sceneSecondVolumeActorRemoved == 1 &&
				gMetrics.sceneSecondVolumeActorReleased == 1 &&
				gMetrics.sceneSecondVolumeBoundsFinite == 1 &&
				gMetrics.sceneSoftChurnCycles > 0 &&
				gMetrics.sceneSoftChurnRemoveCount ==
					2 * gMetrics.sceneSoftChurnCycles &&
				gMetrics.sceneSoftChurnReaddCount ==
					2 * gMetrics.sceneSoftChurnCycles &&
				gMetrics.sceneSoftChurnPostCompactMoveCount ==
					2 * gMetrics.sceneSoftChurnCycles &&
				gMetrics.sceneSoftChurnStable == 1 &&
				gMetrics.finalMinY > 3.5f &&
				gMetrics.finalMaxParticleSpeed < 1.0e-4f;
		}
		if(caseName == "scene-volume-lifecycle" ||
			caseName == "scene-volume-corotational")
			return gMetrics.sceneStatics == 0;
		if(caseName == "scene-volume-ground")
		{
			return
				gMetrics.sceneStatics == 1 &&
				gMetrics.groundContactFrames > 0 &&
				gMetrics.maxGroundContacts > 0 &&
				gMetrics.maxCentroidDrop > 2.0f &&
				gMetrics.finalMinY > -0.1f;
		}
		const bool staticBoxPassed =
			gMetrics.sceneStatics == 1 &&
			gMetrics.rigidContactFrames > 0 &&
			gMetrics.maxRigidContacts > 0 &&
			gMetrics.maxCentroidDrop > 2.0f &&
			gMetrics.finalMinY > 0.7f;
		if(!staticBoxPassed)
			return false;
		if(caseName != "scene-volume-static-churn")
			return true;
		return
			gMetrics.sceneStaticShapeDetached == 1 &&
			gMetrics.sceneStaticShapeReattached == 1 &&
			gMetrics.sceneStaticActorRemoved == 1 &&
			gMetrics.sceneStaticActorReadded == 1;
	}

	bool passed =
		gMetrics.initialized == 1 &&
		gMetrics.completedFrames == gHeadlessOptions.frames &&
		gMetrics.fetchFailures == 0 &&
		gMetrics.nonFiniteParticleSamples == 0 &&
		gMetrics.invertedElementSamples == 0 &&
		gMetrics.invalidContactSourceSamples == 0 &&
		gMetrics.particles > 0 &&
		gMetrics.softBodies > 0 &&
		gMetrics.tetElements > 0 &&
		gMetrics.surfaceTriangles > 0 &&
		gMetrics.sceneStatics == gMetrics.rigidBoxes + 1 &&
		gMetrics.sceneDynamics == 0 &&
		gMetrics.sceneDeformableVolumes == 0 &&
		gMetrics.solverReadbackMatched &&
		gMetrics.cleanupComplete == 1 &&
		PxIsFinite(gMetrics.minDetF) && gMetrics.minDetF > 0.0f &&
		PxIsFinite(gMetrics.maxDetF) && gMetrics.maxDetF < 20.0f &&
		PxIsFinite(gMetrics.minBodyVolumeRatio) &&
		gMetrics.minBodyVolumeRatio > 0.01f &&
		PxIsFinite(gMetrics.maxBodyVolumeRatio) &&
		gMetrics.maxBodyVolumeRatio < 20.0f &&
		PxIsFinite(gMetrics.minY) && gMetrics.minY > -0.25f &&
		PxIsFinite(gMetrics.maxY) && gMetrics.maxY < 100.0f &&
		PxIsFinite(gMetrics.maxParticleSpeed) &&
		gMetrics.maxParticleSpeed < 250.0f &&
		gErrorCallback.getFatalCount() == 0;

	if(caseName == "volume-ground")
	{
		passed = passed &&
			gMetrics.softBodies == 1 &&
			gMetrics.rigidBoxes == 0 &&
			gMetrics.groundContactFrames > 0 &&
			gMetrics.maxGroundContacts > 0 &&
			gMetrics.rigidContactFrames == 0 &&
			gMetrics.softContactFrames == 0 &&
			gMetrics.maxCentroidDrop > 1.0f;
	}
	else if(caseName == "volume-static-box")
	{
		passed = passed &&
			gMetrics.softBodies == 1 &&
			gMetrics.rigidBoxes == 1 &&
			gMetrics.rigidContactFrames > 0 &&
			gMetrics.maxRigidContacts > 0 &&
			gMetrics.softContactFrames == 0 &&
			gMetrics.maxCentroidDrop > 1.0f &&
			gMetrics.finalMinY > 0.70f;
	}
	else if(caseName == "soft-soft")
	{
		passed = passed &&
			gMetrics.softBodies == 2 &&
			gMetrics.rigidBoxes == 0 &&
			gMetrics.softContactFrames > 0 &&
			gMetrics.maxSoftContacts > 0 &&
			gMetrics.finalInsideParticles == 0 &&
			gMetrics.maxCentroidDrop > 1.0f;
	}
	else if(caseName == "cone-ground")
	{
		passed = passed &&
			gMetrics.softBodies == 1 &&
			gMetrics.rigidBoxes == 0 &&
			gMetrics.groundContactFrames > 0 &&
			gMetrics.maxGroundContacts > 0 &&
			gMetrics.rigidContactFrames == 0 &&
			gMetrics.softContactFrames == 0 &&
			gMetrics.maxCentroidDrop > 5.0f;
	}
	else
	{
		passed = passed &&
			gMetrics.softBodies == 5 &&
			gMetrics.rigidBoxes == 1 &&
			gMetrics.groundContactFrames > 0 &&
			gMetrics.rigidContactFrames > 0 &&
			gMetrics.softContactFrames > 0 &&
			gMetrics.maxCentroidDrop > 1.0f;
	}
	return passed;
}

static void finalizePerformanceMetrics()
{
	if(gPerformance.stepSamplesMs.empty())
		return;
	PxF64 sumStepMs = 0.0;
	for(PxU32 i = 0; i < gPerformance.stepSamplesMs.size(); ++i)
		sumStepMs += gPerformance.stepSamplesMs[i];
	PxSort(
		gPerformance.stepSamplesMs.begin(),
		gPerformance.stepSamplesMs.size());
	gPerformance.avgStepMs = PxReal(
		sumStepMs / PxF64(gPerformance.stepSamplesMs.size()));
	const PxU32 last = gPerformance.stepSamplesMs.size() - 1;
	gPerformance.p50StepMs =
		gPerformance.stepSamplesMs[PxU32(PxCeil(0.5f * PxReal(last)))];
	gPerformance.p95StepMs =
		gPerformance.stepSamplesMs[PxU32(PxCeil(0.95f * PxReal(last)))];
	gPerformance.maxStepMs = gPerformance.stepSamplesMs[last];
}

static void printPerformanceResult()
{
#if PX_DEBUG
	const char* buildProfile = "debug";
#elif PX_CHECKED
	const char* buildProfile = "checked";
#else
	const char* buildProfile = "release";
#endif
	const bool softParallel = gPerformance.softWorkers > 1;
	const char* softExecution = softParallel ? "parallel" : "serial";
	const PxU32 softWorkers = gPerformance.softWorkers;
	const PxF64 divisor = gPerformance.profiledFrames ?
		PxF64(gPerformance.profiledFrames) : 1.0;
	const AvbdSoftBodyStepStats& stages = gPerformance.solverStages;
	const PxF64 solverStageMs =
		stages.predictionMs + stages.contactIndexMs +
		stages.bodyPrecomputeMs + stages.bodySolveMs +
		stages.particleSolveMs + stages.projectionMs + stages.dualMs +
		stages.redetectMs + stages.velocityMs + stages.frictionMs;
	const PxF64 closureMs =
		gPerformance.initialContactMs + gPerformance.solverMs +
		gPerformance.sceneMs + gPerformance.metricsMs;

	printf(
		"[AVBD_PERF] schema=1 snippet=" AVBD_VOLUME_SNIPPET_NAME " "
		"case=%s buildProfile=%s softExecution=%s softWorkers=%u "
		"warmupFrames=%u profileFrames=%u "
		"avgStepMs=%.9g p50StepMs=%.9g p95StepMs=%.9g maxStepMs=%.9g "
		"initialContactMs=%.9g solverMs=%.9g sceneMs=%.9g metricsMs=%.9g "
		"predictionMs=%.9g contactIndexMs=%.9g bodyPrecomputeMs=%.9g "
		"bodySolveMs=%.9g particleSolveMs=%.9g projectionMs=%.9g "
		"dualMs=%.9g redetectMs=%.9g velocityMs=%.9g frictionMs=%.9g "
		"solverUnattributedMs=%.9g closureMs=%.9g "
		"requestedOuterIterations=%llu requestedInnerIterations=%llu "
		"executedOuterIterations=%llu executedInnerIterations=%llu "
		"particleSweeps=%llu "
		"convergenceAuthority=localSolveResidualConsecutive "
		"convergenceTolerance=0.0001 convergenceSweeps=2 "
		"trustRegionLimitedParticleSteps=%llu "
		"positiveJLimitedParticleSteps=%llu "
		"positiveJRejectedParticleSteps=%llu "
		"nonFiniteRejectedParticleSteps=%llu "
		"tetLinearizationCacheFallbackParticleSteps=%llu "
		"legacyAppliedConvergedOuterIterations=%llu "
		"residualConvergedOuterIterations=%llu "
		"unsafeAppliedConvergenceCandidates=%llu "
		"budgetExhaustedOuterIterations=%llu "
		"shadowResidual1e5ConvergedOuterIterations=%llu "
		"shadowResidual1e5SavedInnerIterations=%llu "
		"shadowResidual1e4ConvergedOuterIterations=%llu "
		"shadowResidual1e4SavedInnerIterations=%llu "
		"workspaceGrowthEvents=%llu "
		"workspaceGrowthBytes=%llu contactWorkspaceGrowthEvents=%llu "
		"contactWorkspaceGrowthBytes=%llu contactOutputGrowthEvents=%llu "
		"contactOutputGrowthBytes=%llu finalMaxDisplacement=%.9g "
		"finalMaxLocalSolveDisplacement=%.9g "
		"finalMaxAppliedDisplacement=%.9g "
		"detectionCalls=%llu bodyPairs=%llu overlappingBodyPairs=%llu "
		"particleSurfaceCandidates=%llu insideTriangleTests=%llu "
		"closestTriangleTests=%llu selfTriangleTests=%llu "
		"rigidParticleBoxTests=%llu generatedGroundContacts=%llu "
		"generatedRigidContacts=%llu generatedSoftContacts=%llu "
		"generatedSelfContacts=%llu\n",
		gHeadlessOptions.caseName.c_str(), buildProfile,
		softExecution, softWorkers,
		gPerformance.warmupFrames,
		gPerformance.profiledFrames, double(gPerformance.avgStepMs),
		double(gPerformance.p50StepMs),
		double(gPerformance.p95StepMs), double(gPerformance.maxStepMs),
		double(gPerformance.initialContactMs / divisor),
		double(gPerformance.solverMs / divisor),
		double(gPerformance.sceneMs / divisor),
		double(gPerformance.metricsMs / divisor),
		double(stages.predictionMs / divisor),
		double(stages.contactIndexMs / divisor),
		double(stages.bodyPrecomputeMs / divisor),
		double(stages.bodySolveMs / divisor),
		double(stages.particleSolveMs / divisor),
		double(stages.projectionMs / divisor),
		double(stages.dualMs / divisor),
		double(stages.redetectMs / divisor),
		double(stages.velocityMs / divisor),
		double(stages.frictionMs / divisor),
		double((gPerformance.solverMs - solverStageMs) / divisor),
		double(closureMs / divisor),
		static_cast<unsigned long long>(stages.requestedOuterIterations),
		static_cast<unsigned long long>(stages.requestedInnerIterations),
		static_cast<unsigned long long>(stages.executedOuterIterations),
		static_cast<unsigned long long>(stages.executedInnerIterations),
		static_cast<unsigned long long>(stages.particleSweeps),
		static_cast<unsigned long long>(
			stages.trustRegionLimitedParticleSteps),
		static_cast<unsigned long long>(
			stages.positiveJLimitedParticleSteps),
		static_cast<unsigned long long>(
			stages.positiveJRejectedParticleSteps),
		static_cast<unsigned long long>(
			stages.nonFiniteRejectedParticleSteps),
		static_cast<unsigned long long>(
			stages.tetLinearizationCacheFallbackParticleSteps),
		static_cast<unsigned long long>(
			stages.legacyAppliedConvergedOuterIterations),
		static_cast<unsigned long long>(
			stages.residualConvergedOuterIterations),
		static_cast<unsigned long long>(
			stages.unsafeAppliedConvergenceCandidates),
		static_cast<unsigned long long>(
			stages.budgetExhaustedOuterIterations),
		static_cast<unsigned long long>(
			stages.shadowResidual1e5ConvergedOuterIterations),
		static_cast<unsigned long long>(
			stages.shadowResidual1e5SavedInnerIterations),
		static_cast<unsigned long long>(
			stages.shadowResidual1e4ConvergedOuterIterations),
		static_cast<unsigned long long>(
			stages.shadowResidual1e4SavedInnerIterations),
		static_cast<unsigned long long>(stages.workspaceGrowthEvents),
		static_cast<unsigned long long>(stages.workspaceGrowthBytes),
		static_cast<unsigned long long>(
			stages.contactWorkspaceGrowthEvents),
		static_cast<unsigned long long>(
			stages.contactWorkspaceGrowthBytes),
		static_cast<unsigned long long>(
			stages.contactOutputGrowthEvents),
		static_cast<unsigned long long>(
			stages.contactOutputGrowthBytes),
		double(stages.finalMaxDisplacement),
		double(stages.finalMaxLocalSolveDisplacement),
		double(stages.finalMaxAppliedDisplacement),
		static_cast<unsigned long long>(
			gPerformance.collision.detectionCalls),
		static_cast<unsigned long long>(gPerformance.collision.bodyPairs),
		static_cast<unsigned long long>(
			gPerformance.collision.overlappingBodyPairs),
		static_cast<unsigned long long>(
			gPerformance.collision.particleSurfaceCandidates),
		static_cast<unsigned long long>(
			gPerformance.collision.insideTriangleTests),
		static_cast<unsigned long long>(
			gPerformance.collision.closestTriangleTests),
		static_cast<unsigned long long>(
			gPerformance.collision.selfTriangleTests),
		static_cast<unsigned long long>(
			gPerformance.collision.rigidParticleBoxTests),
		static_cast<unsigned long long>(
			gPerformance.collision.generatedGroundContacts),
		static_cast<unsigned long long>(
			gPerformance.collision.generatedRigidContacts),
		static_cast<unsigned long long>(
			gPerformance.collision.generatedSoftContacts),
		static_cast<unsigned long long>(
			gPerformance.collision.generatedSelfContacts));
}

static void printHeadlessResult(bool passed)
{
	if(gHeadlessOptions.caseName == "scene-volume-skinning")
	{
		printf(
			"[AVBD_CPU_SKINNING] schema=1 snippet=%s "
			"kind=volume vertices=%u triangles=%u "
			"evaluatedFrames=%u finiteFrames=%u "
			"maxDisplacement=%.9g status=%s\n",
			AVBD_VOLUME_SNIPPET_NAME,
			gVolumeSkinningMetrics.vertices,
			gVolumeSkinningMetrics.triangles,
			gVolumeSkinningMetrics.evaluatedFrames,
			gVolumeSkinningMetrics.finiteFrames,
			double(gVolumeSkinningMetrics.maxDisplacement),
			passed ? "PASS" : "FAIL");
	}
	if(isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
			gHeadlessOptions.caseName))
	{
		const bool reverseCase =
			isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool staticTarget =
			isSceneCpuVolumeStaticTriangleSurfaceSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool heightField =
			isSceneCpuVolumeHeightFieldSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalCase =
			isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
				gHeadlessOptions.caseName);
		printf(
			"[%s] frames=%u target=%s geometry=%s "
			"responseObserved=%u negativeControlPassed=%u "
			"vertexSweepExcluded=%u nonFiniteSamples=%u "
			"positiveDisplacement=%.9g negativeDisplacement=%.9g "
			"positiveDrop=%.9g negativeDrop=%.9g "
			"minimumVertexSweepSeparation=%.9g result=%s\n",
			reverseCase
				? "AVBD_TRIANGLE_SURFACE_REVERSE_SWEPT"
				: "AVBD_TRIANGLE_SURFACE_FORWARD_SWEPT",
			gHeadlessOptions.frames,
			staticTarget ? "static" : "kinematic",
			heightField ? "heightfield" : "triangle-mesh",
			gSphereReverseSweptMetrics.responseObserved,
			gSphereReverseSweptMetrics.negativeControlPassed,
			gSphereReverseSweptMetrics.vertexSweepExcluded,
			gSphereReverseSweptMetrics.nonFiniteSamples,
			double(
				gSphereReverseSweptMetrics.
					positiveDisplacement),
			double(
				gSphereReverseSweptMetrics.
					negativeDisplacement),
			double(gSphereReverseSweptMetrics.positiveDrop),
			double(gSphereReverseSweptMetrics.negativeDrop),
			double(
				gSphereReverseSweptMetrics.
					minimumVertexSweepSeparation),
			passed ? "PASS" : "FAIL");
		if(rotationalCase)
		{
			printf(
				"[AVBD_TRIANGLE_SURFACE_ROTATIONAL_SWEPT] "
				"frames=%u target=kinematic geometry=%s owner=%s "
				"responseObserved=%u negativeControlPassed=%u "
				"vertexSweepExcluded=%u "
				"endpointMinSeparation=%.9g "
				"midSweepMinSeparation=%.9g "
				"minimumVertexSweepSeparation=%.9g "
				"positiveDisplacement=%.9g "
				"negativeDisplacement=%.9g "
				"positiveAngularTravel=%.9g "
				"negativeAngularTravel=%.9g result=%s\n",
				gHeadlessOptions.frames,
				heightField ? "heightfield" : "triangle-mesh",
				reverseCase ? "reverse" : "forward",
				gSphereReverseSweptMetrics.responseObserved,
				gSphereReverseSweptMetrics.
					negativeControlPassed,
				gSphereReverseSweptMetrics.vertexSweepExcluded,
				double(
					gCapsuleRotationalSweepMetrics.
						endpointMinSeparation),
				double(
					gCapsuleRotationalSweepMetrics.
						midSweepMinSeparation),
				double(
					gSphereReverseSweptMetrics.
						minimumVertexSweepSeparation),
				double(
					gSphereReverseSweptMetrics.
						positiveDisplacement),
				double(
					gSphereReverseSweptMetrics.
						negativeDisplacement),
				double(
					gCapsuleRotationalSweepMetrics.
						positiveAngularTravel),
				double(
					gCapsuleRotationalSweepMetrics.
						negativeAngularTravel),
				passed ? "PASS" : "FAIL");
		}
	}
	if(isSceneCpuVolumeSphereReverseSweptCcdCase(
			gHeadlessOptions.caseName))
	{
		const bool deformingSoftTarget =
			isSceneCpuVolumeDeformingReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool capsuleTarget =
			isSceneCpuVolumeCapsuleReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool convexTarget =
			isSceneCpuVolumeConvexReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const char* target =
			deformingSoftTarget ||
			gHeadlessOptions.caseName ==
					"scene-volume-static-sphere-reverse-swept-ccd" ||
				gHeadlessOptions.caseName ==
					"scene-volume-static-capsule-reverse-swept-ccd" ||
				gHeadlessOptions.caseName ==
					"scene-volume-static-convex-reverse-swept-ccd"
				? "static"
				: (gHeadlessOptions.caseName ==
						"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
				   gHeadlessOptions.caseName ==
						"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
				   gHeadlessOptions.caseName ==
						"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
				   gHeadlessOptions.caseName ==
						"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
				   gHeadlessOptions.caseName ==
						"scene-volume-kinematic-convex-reverse-swept-ccd")
					? "kinematic"
					: "dynamic";
		printf(
			"[%s] frames=%u target=%s "
			"responseObserved=%u negativeControlPassed=%u "
			"twoSidedResponseObserved=%u vertexSweepExcluded=%u "
			"nonFiniteSamples=%u positiveDisplacement=%.9g "
			"negativeDisplacement=%.9g positiveDrop=%.9g "
			"negativeDrop=%.9g positiveRigidDrop=%.9g "
			"negativeRigidDrop=%.9g faceSeparation=%.9g "
			"minimumVertexSweepSeparation=%.9g result=%s\n",
			convexTarget
				? "AVBD_CONVEX_REVERSE_SWEPT"
				: capsuleTarget
				? "AVBD_CAPSULE_REVERSE_SWEPT"
				: "AVBD_SPHERE_REVERSE_SWEPT",
			gHeadlessOptions.frames, target,
			gSphereReverseSweptMetrics.responseObserved,
			gSphereReverseSweptMetrics.negativeControlPassed,
			gSphereReverseSweptMetrics.twoSidedResponseObserved,
			gSphereReverseSweptMetrics.vertexSweepExcluded,
			gSphereReverseSweptMetrics.nonFiniteSamples,
			double(gSphereReverseSweptMetrics.positiveDisplacement),
			double(gSphereReverseSweptMetrics.negativeDisplacement),
			double(gSphereReverseSweptMetrics.positiveDrop),
			double(gSphereReverseSweptMetrics.negativeDrop),
			double(gSphereReverseSweptMetrics.positiveRigidDrop),
			double(gSphereReverseSweptMetrics.negativeRigidDrop),
			double(gSphereReverseSweptMetrics.faceSeparation),
			double(
				gSphereReverseSweptMetrics.
					minimumVertexSweepSeparation),
			passed ? "PASS" : "FAIL");
		if(deformingSoftTarget)
		{
			printf(
				"[AVBD_DEFORMING_VOLUME_REVERSE_SWEPT] "
				"frames=%u geometry=%s target=static owner=reverse "
				"responseObserved=%u negativeControlPassed=%u "
				"geometricSweepIsolated=%u "
				"vertexSweepExcluded=%u nonFiniteSamples=%u "
				"endpointMinSeparation=%.9g "
				"midSweepMinSeparation=%.9g "
				"minimumVertexSweepSeparation=%.9g "
				"responseDelta=%.9g positiveDrop=%.9g "
				"negativeDrop=%.9g result=%s\n",
				gHeadlessOptions.frames,
				convexTarget ? "convex" :
					(capsuleTarget ? "capsule" : "sphere"),
				gSphereReverseSweptMetrics.responseObserved,
				gSphereReverseSweptMetrics.
					negativeControlPassed,
				gDeformingVolumeReverseSweptMetrics.
					geometricSweepIsolated,
				gSphereReverseSweptMetrics.vertexSweepExcluded,
				gSphereReverseSweptMetrics.nonFiniteSamples,
				double(
					gDeformingVolumeReverseSweptMetrics.
						endpointMinSeparation),
				double(
					gDeformingVolumeReverseSweptMetrics.
						midSweepMinSeparation),
				double(
					gSphereReverseSweptMetrics.
						minimumVertexSweepSeparation),
				double(
					gDeformingVolumeReverseSweptMetrics.
						responseDelta),
				double(gSphereReverseSweptMetrics.positiveDrop),
				double(gSphereReverseSweptMetrics.negativeDrop),
				passed ? "PASS" : "FAIL");
		}
		const bool rotationalCapsuleTarget =
			isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalConvexTarget =
			isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		if(rotationalCapsuleTarget || rotationalConvexTarget)
		{
			printf(
				"[%s] "
				"frames=%u target=%s owner=reverse "
				"responseObserved=%u negativeControlPassed=%u "
				"twoSidedResponseObserved=%u vertexSweepExcluded=%u "
				"endpointMinSeparation=%.9g "
				"midSweepMinSeparation=%.9g "
				"positiveDisplacement=%.9g "
				"negativeDisplacement=%.9g "
				"positiveAngularTravel=%.9g "
				"negativeAngularTravel=%.9g result=%s\n",
				rotationalConvexTarget
					? "AVBD_CONVEX_ROTATIONAL_REVERSE_SWEPT"
					: "AVBD_CAPSULE_ROTATIONAL_REVERSE_SWEPT",
				gHeadlessOptions.frames, target,
				gSphereReverseSweptMetrics.responseObserved,
				gSphereReverseSweptMetrics.negativeControlPassed,
				gSphereReverseSweptMetrics.twoSidedResponseObserved,
				gSphereReverseSweptMetrics.vertexSweepExcluded,
				double(
					gCapsuleRotationalSweepMetrics.
						endpointMinSeparation),
				double(
					gCapsuleRotationalSweepMetrics.
						midSweepMinSeparation),
				double(
					gSphereReverseSweptMetrics.
						positiveDisplacement),
				double(
					gSphereReverseSweptMetrics.
						negativeDisplacement),
				double(
					gCapsuleRotationalSweepMetrics.
						positiveAngularTravel),
				double(
					gCapsuleRotationalSweepMetrics.
						negativeAngularTravel),
				passed ? "PASS" : "FAIL");
		}
	}
	if(gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-capsule-speculative-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-convex-speculative-ccd")
	{
		printf(
			"[%s] "
			"frames=%u target=kinematic owner=forward "
			"responseObserved=%u negativeControlPassed=%u "
			"endpointMinSeparation=%.9g "
			"midSweepMinSeparation=%.9g "
			"positiveDisplacement=%.9g "
			"negativeDisplacement=%.9g result=%s\n",
			gHeadlessOptions.caseName ==
				"scene-volume-rotating-kinematic-convex-speculative-ccd"
				? "AVBD_CONVEX_ROTATIONAL_SWEPT"
				: "AVBD_CAPSULE_ROTATIONAL_SWEPT",
			gHeadlessOptions.frames,
			gMetrics.movingSphereCcdResponseObserved,
			gMetrics.movingSphereNegativeControlHeld,
			double(
				gCapsuleRotationalSweepMetrics.
					endpointMinSeparation),
			double(
				gCapsuleRotationalSweepMetrics.
					midSweepMinSeparation),
			double(gMetrics.movingSpherePositiveDisplacement),
			double(gMetrics.movingSphereNegativeDisplacement),
			passed ? "PASS" : "FAIL");
	}
	if(gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-capsule-relative-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-convex-relative-swept-ccd")
	{
		printf(
			"[%s] "
			"frames=%u target=dynamic owner=forward "
			"responseObserved=%u negativeControlPassed=%u "
			"twoSidedResponseObserved=%u "
			"endpointMinSeparation=%.9g "
			"midSweepMinSeparation=%.9g "
			"positiveDisplacement=%.9g "
			"negativeDisplacement=%.9g "
			"positiveAngularTravel=%.9g "
			"negativeAngularTravel=%.9g result=%s\n",
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-rotating-convex-relative-swept-ccd"
				? "AVBD_CONVEX_DYNAMIC_ROTATIONAL_SWEPT"
				: "AVBD_CAPSULE_DYNAMIC_ROTATIONAL_SWEPT",
			gHeadlessOptions.frames,
			gMetrics.dynamicSphereSweepResponseObserved,
			gMetrics.dynamicSphereSweepNegativeControlTunneled,
			gMetrics.dynamicSphereSweepTwoSidedResponseObserved,
			double(
				gCapsuleRotationalSweepMetrics.
					endpointMinSeparation),
			double(
				gCapsuleRotationalSweepMetrics.
					midSweepMinSeparation),
			double(
				gMetrics.
					dynamicSphereSweepPositiveSoftDisplacement),
			double(
				gMetrics.
					dynamicSphereSweepNegativeSoftDisplacement),
			double(
				gCapsuleRotationalSweepMetrics.
					positiveAngularTravel),
			double(
				gCapsuleRotationalSweepMetrics.
					negativeAngularTravel),
			passed ? "PASS" : "FAIL");
	}
	const bool capsuleReverseFeatureOutput =
		gHeadlessOptions.caseName ==
			"scene-volume-capsule-reverse-feature";
	const bool convexReverseFeatureOutput =
		gHeadlessOptions.caseName ==
			"scene-volume-convex-reverse-feature";
	const bool triangleMeshReverseFeatureOutput =
		gHeadlessOptions.caseName ==
			"scene-volume-triangle-mesh-reverse-feature";
	const bool heightFieldReverseFeatureOutput =
		gHeadlessOptions.caseName ==
			"scene-volume-heightfield-reverse-feature";
	if(gHeadlessOptions.caseName ==
			"scene-volume-sphere-reverse-feature" ||
		capsuleReverseFeatureOutput ||
		convexReverseFeatureOutput ||
		triangleMeshReverseFeatureOutput ||
		heightFieldReverseFeatureOutput)
	{
		const char* reverseFeatureTag =
			triangleMeshReverseFeatureOutput
				? "AVBD_TRIANGLE_MESH_REVERSE_FEATURE"
				: heightFieldReverseFeatureOutput
				? "AVBD_HEIGHTFIELD_REVERSE_FEATURE"
				: convexReverseFeatureOutput
				? "AVBD_CONVEX_REVERSE_FEATURE"
				: capsuleReverseFeatureOutput
				? "AVBD_CAPSULE_REVERSE_FEATURE"
				: "AVBD_SPHERE_REVERSE_FEATURE";
		printf(
			"[%s] frames=%u "
			"faceResponseObserved=%u vertexSdfExcluded=%u "
			"negativeControlPassed=%u nonFiniteSamples=%u "
			"positiveDisplacement=%.9g positiveDrop=%.9g "
			"negativeDrop=%.9g faceSeparation=%.9g "
			"minimumVertexSeparation=%.9g result=%s\n",
			reverseFeatureTag, gHeadlessOptions.frames,
			gSphereReverseFeatureMetrics.faceResponseObserved,
			gSphereReverseFeatureMetrics.vertexSdfExcluded,
			gSphereReverseFeatureMetrics.negativeControlPassed,
			gSphereReverseFeatureMetrics.nonFiniteSamples,
			double(
				gSphereReverseFeatureMetrics.
					positiveDisplacement),
			double(gSphereReverseFeatureMetrics.positiveDrop),
			double(gSphereReverseFeatureMetrics.negativeDrop),
			double(gSphereReverseFeatureMetrics.faceSeparation),
			double(
				gSphereReverseFeatureMetrics.
					minimumVertexSeparation),
			passed ? "PASS" : "FAIL");
	}
	const bool sceneLifecycle =
		gHeadlessOptions.caseName == "scene-volume-lifecycle" ||
		gHeadlessOptions.caseName == "scene-volume-corotational";
	const bool sceneGround =
		gHeadlessOptions.caseName == "scene-volume-ground";
	const bool sceneMaxDepenetrationVelocity =
		gHeadlessOptions.caseName ==
			"scene-volume-max-depenetration-velocity";
	const bool sceneSpeculativeCcd =
		isSceneCpuVolumeSpeculativeCcdCase(
			gHeadlessOptions.caseName) ||
		isSceneCpuVolumeSphereReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool sceneSphereReverseFeature =
		gHeadlessOptions.caseName ==
			"scene-volume-sphere-reverse-feature";
	const bool sceneCapsuleReverseFeature =
		gHeadlessOptions.caseName ==
			"scene-volume-capsule-reverse-feature";
	const bool sceneConvexReverseFeature =
		gHeadlessOptions.caseName ==
			"scene-volume-convex-reverse-feature";
	const bool sceneTriangleMeshReverseFeature =
		gHeadlessOptions.caseName ==
			"scene-volume-triangle-mesh-reverse-feature";
	const bool sceneHeightFieldReverseFeature =
		gHeadlessOptions.caseName ==
			"scene-volume-heightfield-reverse-feature";
	const bool sceneStaticChurn =
		gHeadlessOptions.caseName == "scene-volume-static-churn";
	const bool sceneStaticBox =
		gHeadlessOptions.caseName == "scene-volume-static-box" ||
		sceneStaticChurn;
	const bool sceneStatic =
		sceneGround || sceneStaticBox ||
		sceneMaxDepenetrationVelocity ||
		sceneSpeculativeCcd ||
		sceneSphereReverseFeature ||
		sceneCapsuleReverseFeature ||
		sceneConvexReverseFeature ||
		sceneTriangleMeshReverseFeature ||
		sceneHeightFieldReverseFeature;
	const bool sceneDynamicChurn =
		gHeadlessOptions.caseName == "scene-volume-dynamic-churn";
	const bool sceneMultiDynamic =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-dynamic-box";
	const bool sceneMultiSoft =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-soft-islands";
	const bool sceneSoftSleepWake =
		gHeadlessOptions.caseName ==
			"scene-volume-sleep-wake";
	const bool sceneSoftRigidWake =
		gHeadlessOptions.caseName ==
			"scene-volume-rigid-wake";
	const bool sceneMixedSleepIslands =
		gHeadlessOptions.caseName ==
			"scene-volume-mixed-sleep-islands";
	const bool sceneSoftChurn =
		gHeadlessOptions.caseName ==
			"scene-volume-soft-churn";
	const bool sceneBufferMutation =
		gHeadlessOptions.caseName ==
			"scene-volume-buffer-mutation";
	const bool sceneWorldPin =
		gHeadlessOptions.caseName ==
			"scene-volume-world-pin" ||
		gHeadlessOptions.caseName ==
			"scene-volume-world-element-attachment";
	const bool sceneRigidAttachment =
		gHeadlessOptions.caseName ==
			"scene-volume-rigid-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-rigid-element-attachment";
	const bool sceneStaticAttachment =
		gHeadlessOptions.caseName ==
			"scene-volume-static-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-static-element-attachment";
	const bool sceneKinematicAttachment =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-element-attachment";
	const bool sceneArticulationAttachment =
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-element-attachment";
	const bool scenePartialElementFilter =
		gHeadlessOptions.caseName ==
			"scene-volume-partial-element-filter";
	const bool sceneElementFilter =
		gHeadlessOptions.caseName ==
			"scene-volume-element-filter" ||
		scenePartialElementFilter;
	const bool sceneKinematicBox =
		isSceneCpuVolumeKinematicRigidCase(
			gHeadlessOptions.caseName);
	const bool sceneMultiSceneIsolation =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-scene-isolation";
	const bool sceneSoftSoftWake =
		gHeadlessOptions.caseName ==
			"scene-volume-soft-soft-wake";
	const bool sceneSoftPairAttachment =
		gHeadlessOptions.caseName ==
			"scene-volume-volume-attachment";
	const bool sceneMotionControls =
		gHeadlessOptions.caseName ==
			"scene-volume-motion-controls";
	const bool sceneSkinning =
		gHeadlessOptions.caseName ==
			"scene-volume-skinning";
	const bool sceneVolumeKinematicTarget =
		gHeadlessOptions.caseName ==
			"scene-volume-full-kinematic-target" ||
		gHeadlessOptions.caseName ==
			"scene-volume-partial-kinematic-target";
	const bool sceneDynamic =
		gHeadlessOptions.caseName == "scene-volume-dynamic-box" ||
		gHeadlessOptions.caseName == "scene-volume-dynamic-sphere" ||
		gHeadlessOptions.caseName == "scene-volume-dynamic-capsule" ||
		gHeadlessOptions.caseName == "scene-volume-dynamic-convex" ||
		sceneDynamicChurn || sceneMultiDynamic ||
		sceneMultiSoft || sceneRigidAttachment ||
		sceneKinematicAttachment;
	const bool sceneIntegrated =
		sceneLifecycle || sceneStatic || sceneDynamic ||
		sceneSoftSleepWake || sceneSoftRigidWake ||
		sceneMixedSleepIslands || sceneSoftChurn ||
		sceneBufferMutation || sceneWorldPin ||
		sceneRigidAttachment || sceneStaticAttachment ||
		sceneKinematicAttachment ||
		sceneArticulationAttachment ||
		sceneElementFilter ||
		sceneKinematicBox ||
		sceneMultiSceneIsolation ||
		sceneSoftSoftWake ||
		sceneSoftPairAttachment ||
		sceneVolumeKinematicTarget ||
		sceneSkinning ||
		sceneMotionControls ||
		sceneMaxDepenetrationVelocity ||
		sceneSpeculativeCcd;
	const char* validation =
		sceneSkinning ? "SCENE_CPU_SKINNING_GATED" :
		(sceneMotionControls ?
			"SCENE_DEFORMABLE_MOTION_CONTROLS_GATED" :
		sceneVolumeKinematicTarget ?
			"SCENE_VOLUME_KINEMATIC_TARGET_GATED" :
		(sceneArticulationAttachment ?
			"SCENE_ARTICULATION_ATTACHMENT_GATED" :
		(sceneStaticAttachment ?
			"SCENE_STATIC_ATTACHMENT_GATED" :
		(sceneKinematicAttachment ?
			"SCENE_KINEMATIC_ATTACHMENT_GATED" :
		(sceneRigidAttachment ? "SCENE_RIGID_ATTACHMENT_GATED" :
		(sceneSoftPairAttachment ?
			"SCENE_SOFT_PAIR_ATTACHMENT_GATED" :
		(sceneSoftSoftWake ? "SCENE_SOFT_SOFT_WAKE_GATED" :
		(sceneMultiSceneIsolation ?
			"SCENE_MULTI_SCENE_ISOLATION_GATED" :
		(sceneKinematicBox ? "SCENE_KINEMATIC_COUPLING_GATED" :
		(sceneElementFilter ?
			(scenePartialElementFilter ?
				"SCENE_PARTIAL_ELEMENT_FILTER_GATED" :
				"SCENE_ELEMENT_FILTER_GATED") :
		(sceneWorldPin ? "SCENE_WORLD_PIN_GATED" :
		(sceneBufferMutation ? "SCENE_BUFFER_MUTATION_GATED" :
		(sceneSoftChurn ? "SCENE_SOFT_CHURN_GATED" :
		(sceneMixedSleepIslands ?
			"SCENE_MIXED_SLEEP_ISLANDS_GATED" :
		(sceneSoftRigidWake ? "SCENE_SOFT_RIGID_WAKE_GATED" :
		(sceneSoftSleepWake ? "SCENE_SOFT_SLEEP_WAKE_GATED" :
		(sceneLifecycle ? "SCENE_LIFECYCLE_GATED" :
			(sceneDynamicChurn ?
				"SCENE_DYNAMIC_LIFECYCLE_GATED" :
			(sceneMultiSoft ?
				"SCENE_MULTI_SOFT_ISLANDS_GATED" :
			(sceneMultiDynamic ?
				"SCENE_MULTI_DYNAMIC_COUPLING_GATED" :
			(sceneDynamic ? "SCENE_DYNAMIC_COUPLING_GATED" :
				(sceneStaticChurn ?
					"SCENE_STATIC_LIFECYCLE_GATED" :
					(sceneStatic ? "SCENE_STATIC_CONTACT_GATED" :
						"COMPONENT_GATED")))))))))))))))))))))));
	if(sceneMaxDepenetrationVelocity)
		validation =
			"SCENE_MAX_DEPENETRATION_VELOCITY_GATED";
	if(sceneSpeculativeCcd)
		validation =
			"SCENE_SPECULATIVE_CCD_GATED";
	if(sceneSphereReverseFeature)
		validation =
			"SCENE_SPHERE_REVERSE_FEATURE_GATED";
	if(sceneCapsuleReverseFeature)
		validation =
			"SCENE_CAPSULE_REVERSE_FEATURE_GATED";
	if(sceneConvexReverseFeature)
		validation =
			"SCENE_CONVEX_REVERSE_FEATURE_GATED";
	printf(
		"[AVBD_GATE] schema=1 snippet=" AVBD_VOLUME_SNIPPET_NAME " "
		"case=%s solver=%s validation=%s "
		"sceneSoftIntegration=%u status=%s initialized=%u "
		"frames=%u fetchFailures=%u particles=%u softBodies=%u "
		"tetElements=%u surfaceTriangles=%u rigidBoxes=%u "
		"sceneStatics=%u sceneDynamics=%u sceneDeformableVolumes=%u "
		"sceneActorCreated=%u sceneShapeAttached=%u "
		"sceneSimulationMeshAttached=%u "
		"sceneHostBuffersInitialized=%u sceneActorAdded=%u "
		"sceneActorRemoved=%u sceneActorReleased=%u "
		"sceneBoundsFinite=%u "
		"sceneSecondVolumeActorCreated=%u "
		"sceneSecondVolumeHostBuffersInitialized=%u "
		"sceneSecondVolumeActorAdded=%u "
		"sceneSecondVolumeActorRemoved=%u "
		"sceneSecondVolumeActorReleased=%u "
		"sceneSecondVolumeBoundsFinite=%u "
		"sceneSoftInitiallyAwake=%u "
		"sceneSoftFirstSlept=%u "
		"sceneSoftFirstSleepFrame=%u "
		"sceneSoftSleepWakeCounterZero=%u "
		"sceneSoftSleepVelocitiesZero=%u "
		"sceneSoftStableWhileSleeping=%u "
		"sceneSoftCounterWakeIssued=%u "
		"sceneSoftWokeByCounter=%u "
		"sceneSoftCounterWakeFrame=%u "
		"sceneSoftSecondSlept=%u "
		"sceneSoftSecondSleepFrame=%u "
		"sceneSoftVelocityWakeIssued=%u "
		"sceneSoftWokeByVelocity=%u "
		"sceneSoftVelocityWakeFrame=%u "
		"sceneSoftMovedAfterVelocityWake=%u "
		"sceneSoftVelocityStopIssued=%u "
		"sceneSoftFinalSlept=%u "
		"sceneSoftFinalSleepFrame=%u "
		"sceneSoftRigidWakeActorAdded=%u "
		"sceneSoftWokeByRigid=%u "
		"sceneSoftRigidWakeFrame=%u "
		"sceneSoftMovedAfterRigidWake=%u "
		"sceneMixedFirstSlept=%u "
		"sceneMixedFirstSleepFrame=%u "
		"sceneMixedFirstStable=%u "
		"sceneMixedSecondStayedAwake=%u "
		"sceneMixedSecondMoved=%u "
		"sceneSoftChurnRemoveCount=%u "
		"sceneSoftChurnReaddCount=%u "
		"sceneSoftChurnCycles=%u "
		"sceneSoftChurnPostCompactMoveCount=%u "
		"sceneSoftChurnStable=%u "
		"sceneBufferMutationIssued=%u "
		"sceneBufferMutationWoke=%u "
		"sceneBufferMutationApplied=%u "
		"sceneBufferDriveIssued=%u "
		"sceneBufferPinHeld=%u "
		"sceneBufferDynamicMoved=%u "
		"sceneBufferInvMassRestored=%u "
		"sceneBufferRestoredMoved=%u "
		"sceneBufferResetIssued=%u "
		"sceneWorldPinCreated=%u sceneWorldPinHeld=%u "
		"sceneWorldPinActorReadded=%u "
		"sceneWorldPinReleased=%u "
		"sceneWorldPinMovedAfterRelease=%u "
		"sceneRigidAttachmentActorAdded=%u "
		"sceneRigidAttachmentInitiallySleeping=%u "
		"sceneRigidAttachmentCreated=%u "
		"sceneRigidAttachmentRigidWoke=%u "
		"sceneRigidAttachmentRigidMoved=%u "
		"sceneRigidAttachmentHeldAcrossReadd=%u "
		"sceneRigidAttachmentReleased=%u "
		"sceneRigidAttachmentSeparatedAfterRelease=%u "
		"sceneArticulationCreated=%u "
		"sceneArticulationAdded=%u "
		"sceneArticulationInitiallySleeping=%u "
		"sceneArticulationWoke=%u "
		"sceneArticulationJointSubspaceHeld=%u "
		"sceneArticulationRootStable=%u "
		"sceneElementFilterCreated=%u "
		"sceneElementFilterActorReadded=%u "
		"sceneElementFilterSuppressedContact=%u "
		"sceneElementFilterReleased=%u "
		"sceneElementFilterContactRestored=%u "
		"scenePartialFilterUnfilteredContactHeld=%u "
		"scenePartialFilterExactOwnership=%u "
		"sceneKinematicActorAdded=%u "
		"sceneKinematicTargetIssued=%u "
		"sceneKinematicTargetReached=%u "
		"sceneKinematicSoftWoke=%u "
		"sceneKinematicSoftMoved=%u "
		"sceneKinematicContactObserved=%u "
		"sceneVolumeTargetBound=%u "
		"sceneVolumeTargetMutated=%u "
		"sceneVolumeTargetWoke=%u "
		"sceneVolumeTargetReached=%u "
		"sceneVolumePartialInactiveIgnored=%u "
		"sceneVolumePartialActivated=%u "
		"sceneVolumePartialActivatedReached=%u "
		"sceneSecondSceneCreated=%u "
		"sceneSecondSceneSolverMatched=%u "
		"scenePrimarySceneReleased=%u "
		"sceneSecondSceneReleased=%u "
		"sceneMultiPrimaryStable=%u "
		"sceneMultiPrimaryDetachedStable=%u "
		"sceneMultiSecondaryUpdatedBeforeRelease=%u "
		"sceneMultiSecondaryUpdatedAfterRelease=%u "
		"sceneSoftSoftBothSlept=%u "
		"sceneSoftSoftDriveIssued=%u "
		"sceneSoftSoftDriverWoke=%u "
		"sceneSoftSoftTargetWoke=%u "
		"sceneSoftSoftTargetWakeFrame=%u "
		"sceneSoftSoftTargetMoved=%u "
		"sceneSoftSoftResetIssued=%u "
		"sceneSoftSoftBothFinalSlept=%u "
		"motionMaxVelocityBounded=%u "
		"motionSettlingApplied=%u "
		"motionSettlingSlept=%u "
		"motionControlStayedAwake=%u "
		"depenetrationLimitApplied=%u "
		"depenetrationFirstStepBounded=%u "
		"depenetrationControlSeparated=%u "
		"depenetrationGradualRecovery=%u "
		"speculativeCcdFlagApplied=%u "
		"speculativeCcdPreventedTunneling=%u "
		"speculativeCcdNegativeControlTunneled=%u "
		"movingSphereTargetIssued=%u "
		"movingSphereCcdResponseObserved=%u "
		"movingSphereNegativeControlHeld=%u "
		"dynamicSphereSweepLaunched=%u "
		"dynamicSphereSweepResponseObserved=%u "
		"dynamicSphereSweepNegativeControlTunneled=%u "
		"dynamicSphereSweepTwoSidedResponseObserved=%u "
		"sceneStaticShapeDetached=%u "
		"sceneStaticShapeReattached=%u "
		"sceneStaticActorRemoved=%u sceneStaticActorReadded=%u "
		"sceneDynamicActorAdded=%u "
		"sceneDynamicActorRemoved=%u "
		"sceneDynamicActorReleased=%u "
		"sceneDynamicInitiallySleeping=%u "
		"sceneDynamicWokeBySoft=%u "
		"sceneDynamicFirstWakeFrame=%u "
		"sceneDynamicShapeDetached=%u "
		"sceneDynamicShapeReattached=%u "
		"sceneDynamicActorReadded=%u "
		"sceneDynamicReaddedSleeping=%u "
		"sceneDynamicRewokeBySoft=%u "
		"sceneDynamicSecondWakeFrame=%u "
		"sceneSecondDynamicActorAdded=%u "
		"sceneSecondDynamicActorRemoved=%u "
		"sceneSecondDynamicActorReleased=%u "
		"sceneSecondDynamicInitiallySleeping=%u "
		"sceneSecondDynamicWokeBySoft=%u "
		"sceneSecondDynamicFirstWakeFrame=%u "
		"groundContactFrames=%u rigidContactFrames=%u "
		"softContactFrames=%u maxGroundContacts=%u "
		"maxRigidContacts=%u maxSoftContacts=%u "
		"invalidContactSourceSamples=%u finalInsideParticles=%u "
		"nonFiniteParticleSamples=%u "
		"invertedElementSamples=%u firstInversionFrame=%u "
		"firstInversionBody=%u firstInversionElement=%u "
		"invertedBodiesMask=%u minDetF=%.9g maxDetF=%.9g "
		"minBodyVolumeRatio=%.9g maxBodyVolumeRatio=%.9g "
		"minY=%.9g maxY=%.9g finalMinY=%.9g finalMaxY=%.9g "
		"maxParticleSpeed=%.9g finalMaxParticleSpeed=%.9g "
		"maxCentroidDrop=%.9g "
		"sceneSecondVolumeMaxCentroidDrop=%.9g "
		"sceneSecondVolumeFinalCentroidY=%.9g "
		"sceneWorldPinMaxDrift=%.9g "
		"sceneWorldPinReleasedMaxDisplacement=%.9g "
		"sceneRigidAttachmentMaxDrift=%.9g "
		"sceneRigidAttachmentMaxRigidDisplacement=%.9g "
		"sceneRigidAttachmentMaxRigidSpeed=%.9g "
		"sceneRigidAttachmentReleasedSeparation=%.9g "
		"sceneArticulationRootMaxDisplacement=%.9g "
		"sceneArticulationChildMaxForbiddenDisplacement=%.9g "
		"sceneArticulationChildMaxAngularDisplacement=%.9g "
		"sceneElementFilterMinY=%.9g "
		"sceneElementFilterFinalMinY=%.9g "
		"scenePartialFilterUnfilteredMinY=%.9g "
		"sceneKinematicMaxPoseError=%.9g "
		"sceneKinematicSoftDisplacement=%.9g "
		"sceneKinematicFinalY=%.9g "
		"sceneVolumeTargetFinalMaxError=%.9g "
		"sceneVolumeTargetMaxDisplacement=%.9g "
		"sceneVolumePartialInactiveDecoyDistance=%.9g "
		"sceneDynamicInitialY=%.9g "
		"sceneDynamicMinY=%.9g sceneDynamicFinalY=%.9g "
		"sceneDynamicMaxDrop=%.9g "
		"sceneDynamicPreContactMaxDrop=%.9g "
		"sceneDynamicMaxDownSpeed=%.9g "
		"sceneSecondDynamicInitialY=%.9g "
		"sceneSecondDynamicMinY=%.9g "
		"sceneSecondDynamicFinalY=%.9g "
		"sceneSecondDynamicMaxDrop=%.9g "
		"sceneSecondDynamicPreContactMaxDrop=%.9g "
		"sceneSecondDynamicMaxDownSpeed=%.9g "
		"minDynamicSurfaceSeparation=%.9g "
		"finalDynamicSurfaceSeparation=%.9g "
		"motionMaxVelocityFirstStepDisplacement=%.9g "
		"motionMaxVelocityFirstStepSpeed=%.9g "
		"motionSettlingFinalSpeed=%.9g "
		"motionControlFinalSpeed=%.9g "
		"depenetrationLimitedFirstStepRise=%.9g "
		"depenetrationControlFirstStepRise=%.9g "
		"depenetrationLimitedFinalRise=%.9g "
		"depenetrationLimitedMaxSpeed=%.9g "
		"speculativeCcdPositiveMinY=%.9g "
		"speculativeCcdPositiveMinSeparation=%.9g "
		"speculativeCcdNegativeMaxY=%.9g "
		"movingSpherePositiveDisplacement=%.9g "
		"movingSphereNegativeDisplacement=%.9g "
		"movingSpherePositiveMinSeparation=%.9g "
		"dynamicSphereSweepPositiveSoftDisplacement=%.9g "
		"dynamicSphereSweepNegativeSoftDisplacement=%.9g "
		"dynamicSphereSweepPositiveRigidDrop=%.9g "
		"dynamicSphereSweepNegativeRigidDrop=%.9g "
		"dynamicSphereSweepPositiveMinSeparation=%.9g "
		"solverReadbackMatched=%u "
		"fatalErrors=%u warningErrors=%u cleanupComplete=%u\n",
		gHeadlessOptions.caseName.c_str(),
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		validation,
		sceneIntegrated ? 1u : 0u,
		passed ? "PASS" : "FAIL", gMetrics.initialized,
		gMetrics.completedFrames, gMetrics.fetchFailures,
		gMetrics.particles, gMetrics.softBodies, gMetrics.tetElements,
		gMetrics.surfaceTriangles, gMetrics.rigidBoxes,
		gMetrics.sceneStatics, gMetrics.sceneDynamics,
		gMetrics.sceneDeformableVolumes,
		gMetrics.sceneActorCreated, gMetrics.sceneShapeAttached,
		gMetrics.sceneSimulationMeshAttached,
		gMetrics.sceneHostBuffersInitialized, gMetrics.sceneActorAdded,
		gMetrics.sceneActorRemoved, gMetrics.sceneActorReleased,
		gMetrics.sceneBoundsFinite,
		gMetrics.sceneSecondVolumeActorCreated,
		gMetrics.sceneSecondVolumeHostBuffersInitialized,
		gMetrics.sceneSecondVolumeActorAdded,
		gMetrics.sceneSecondVolumeActorRemoved,
		gMetrics.sceneSecondVolumeActorReleased,
		gMetrics.sceneSecondVolumeBoundsFinite,
		gMetrics.sceneSoftInitiallyAwake,
		gMetrics.sceneSoftFirstSlept,
		gMetrics.sceneSoftFirstSleepFrame,
		gMetrics.sceneSoftSleepWakeCounterZero,
		gMetrics.sceneSoftSleepVelocitiesZero,
		gMetrics.sceneSoftStableWhileSleeping,
		gMetrics.sceneSoftCounterWakeIssued,
		gMetrics.sceneSoftWokeByCounter,
		gMetrics.sceneSoftCounterWakeFrame,
		gMetrics.sceneSoftSecondSlept,
		gMetrics.sceneSoftSecondSleepFrame,
		gMetrics.sceneSoftVelocityWakeIssued,
		gMetrics.sceneSoftWokeByVelocity,
		gMetrics.sceneSoftVelocityWakeFrame,
		gMetrics.sceneSoftMovedAfterVelocityWake,
		gMetrics.sceneSoftVelocityStopIssued,
		gMetrics.sceneSoftFinalSlept,
		gMetrics.sceneSoftFinalSleepFrame,
		gMetrics.sceneSoftRigidWakeActorAdded,
		gMetrics.sceneSoftWokeByRigid,
		gMetrics.sceneSoftRigidWakeFrame,
		gMetrics.sceneSoftMovedAfterRigidWake,
		gMetrics.sceneMixedFirstSlept,
		gMetrics.sceneMixedFirstSleepFrame,
		gMetrics.sceneMixedFirstStable,
		gMetrics.sceneMixedSecondStayedAwake,
		gMetrics.sceneMixedSecondMoved,
		gMetrics.sceneSoftChurnRemoveCount,
		gMetrics.sceneSoftChurnReaddCount,
		gMetrics.sceneSoftChurnCycles,
		gMetrics.sceneSoftChurnPostCompactMoveCount,
		gMetrics.sceneSoftChurnStable,
		gMetrics.sceneBufferMutationIssued,
		gMetrics.sceneBufferMutationWoke,
		gMetrics.sceneBufferMutationApplied,
		gMetrics.sceneBufferDriveIssued,
		gMetrics.sceneBufferPinHeld,
		gMetrics.sceneBufferDynamicMoved,
		gMetrics.sceneBufferInvMassRestored,
		gMetrics.sceneBufferRestoredMoved,
		gMetrics.sceneBufferResetIssued,
		gMetrics.sceneWorldPinCreated,
		gMetrics.sceneWorldPinHeld,
		gMetrics.sceneWorldPinActorReadded,
		gMetrics.sceneWorldPinReleased,
		gMetrics.sceneWorldPinMovedAfterRelease,
		gMetrics.sceneRigidAttachmentActorAdded,
		gMetrics.sceneRigidAttachmentInitiallySleeping,
		gMetrics.sceneRigidAttachmentCreated,
		gMetrics.sceneRigidAttachmentRigidWoke,
		gMetrics.sceneRigidAttachmentRigidMoved,
		gMetrics.sceneRigidAttachmentHeldAcrossReadd,
		gMetrics.sceneRigidAttachmentReleased,
		gMetrics.sceneRigidAttachmentSeparatedAfterRelease,
		gMetrics.sceneArticulationCreated,
		gMetrics.sceneArticulationAdded,
		gMetrics.sceneArticulationInitiallySleeping,
		gMetrics.sceneArticulationWoke,
		gMetrics.sceneArticulationJointSubspaceHeld,
		gMetrics.sceneArticulationRootStable,
		gMetrics.sceneElementFilterCreated,
		gMetrics.sceneElementFilterActorReadded,
		gMetrics.sceneElementFilterSuppressedContact,
		gMetrics.sceneElementFilterReleased,
		gMetrics.sceneElementFilterContactRestored,
		gMetrics.scenePartialFilterUnfilteredContactHeld,
		gMetrics.scenePartialFilterExactOwnership,
		gMetrics.sceneKinematicActorAdded,
		gMetrics.sceneKinematicTargetIssued,
		gMetrics.sceneKinematicTargetReached,
		gMetrics.sceneKinematicSoftWoke,
		gMetrics.sceneKinematicSoftMoved,
		gMetrics.sceneKinematicContactObserved,
		gMetrics.sceneVolumeTargetBound,
		gMetrics.sceneVolumeTargetMutated,
		gMetrics.sceneVolumeTargetWoke,
		gMetrics.sceneVolumeTargetReached,
		gMetrics.sceneVolumePartialInactiveIgnored,
		gMetrics.sceneVolumePartialActivated,
		gMetrics.sceneVolumePartialActivatedReached,
		gMetrics.sceneSecondSceneCreated,
		gMetrics.sceneSecondSceneSolverMatched,
		gMetrics.scenePrimarySceneReleased,
		gMetrics.sceneSecondSceneReleased,
		gMetrics.sceneMultiPrimaryStable,
		gMetrics.sceneMultiPrimaryDetachedStable,
		gMetrics.sceneMultiSecondaryUpdatedBeforeRelease,
		gMetrics.sceneMultiSecondaryUpdatedAfterRelease,
		gMetrics.sceneSoftSoftBothSlept,
		gMetrics.sceneSoftSoftDriveIssued,
		gMetrics.sceneSoftSoftDriverWoke,
		gMetrics.sceneSoftSoftTargetWoke,
		gMetrics.sceneSoftSoftTargetWakeFrame,
		gMetrics.sceneSoftSoftTargetMoved,
		gMetrics.sceneSoftSoftResetIssued,
		gMetrics.sceneSoftSoftBothFinalSlept,
		gMetrics.motionMaxVelocityBounded,
		gMetrics.motionSettlingApplied,
		gMetrics.motionSettlingSlept,
		gMetrics.motionControlStayedAwake,
		gMetrics.depenetrationLimitApplied,
		gMetrics.depenetrationFirstStepBounded,
		gMetrics.depenetrationControlSeparated,
		gMetrics.depenetrationGradualRecovery,
		gMetrics.speculativeCcdFlagApplied,
		gMetrics.speculativeCcdPreventedTunneling,
		gMetrics.speculativeCcdNegativeControlTunneled,
		gMetrics.movingSphereTargetIssued,
		gMetrics.movingSphereCcdResponseObserved,
		gMetrics.movingSphereNegativeControlHeld,
		gMetrics.dynamicSphereSweepLaunched,
		gMetrics.dynamicSphereSweepResponseObserved,
		gMetrics.dynamicSphereSweepNegativeControlTunneled,
		gMetrics.dynamicSphereSweepTwoSidedResponseObserved,
		gMetrics.sceneStaticShapeDetached,
		gMetrics.sceneStaticShapeReattached,
		gMetrics.sceneStaticActorRemoved,
		gMetrics.sceneStaticActorReadded,
		gMetrics.sceneDynamicActorAdded,
		gMetrics.sceneDynamicActorRemoved,
		gMetrics.sceneDynamicActorReleased,
		gMetrics.sceneDynamicInitiallySleeping,
		gMetrics.sceneDynamicWokeBySoft,
		gMetrics.sceneDynamicFirstWakeFrame,
		gMetrics.sceneDynamicShapeDetached,
		gMetrics.sceneDynamicShapeReattached,
		gMetrics.sceneDynamicActorReadded,
		gMetrics.sceneDynamicReaddedSleeping,
		gMetrics.sceneDynamicRewokeBySoft,
		gMetrics.sceneDynamicSecondWakeFrame,
		gMetrics.sceneSecondDynamicActorAdded,
		gMetrics.sceneSecondDynamicActorRemoved,
		gMetrics.sceneSecondDynamicActorReleased,
		gMetrics.sceneSecondDynamicInitiallySleeping,
		gMetrics.sceneSecondDynamicWokeBySoft,
		gMetrics.sceneSecondDynamicFirstWakeFrame,
		gMetrics.groundContactFrames,
		gMetrics.rigidContactFrames, gMetrics.softContactFrames,
		gMetrics.maxGroundContacts, gMetrics.maxRigidContacts,
		gMetrics.maxSoftContacts, gMetrics.invalidContactSourceSamples,
		gMetrics.finalInsideParticles,
		gMetrics.nonFiniteParticleSamples, gMetrics.invertedElementSamples,
		gMetrics.firstInversionFrame, gMetrics.firstInversionBody,
		gMetrics.firstInversionElement, gMetrics.invertedBodiesMask,
		double(gMetrics.minDetF), double(gMetrics.maxDetF),
		double(gMetrics.minBodyVolumeRatio),
		double(gMetrics.maxBodyVolumeRatio), double(gMetrics.minY),
		double(gMetrics.maxY), double(gMetrics.finalMinY),
		double(gMetrics.finalMaxY), double(gMetrics.maxParticleSpeed),
		double(gMetrics.finalMaxParticleSpeed),
		double(gMetrics.maxCentroidDrop),
		double(gMetrics.sceneSecondVolumeMaxCentroidDrop),
		double(gMetrics.sceneSecondVolumeFinalCentroidY),
		double(gMetrics.sceneWorldPinMaxDrift),
		double(gMetrics.sceneWorldPinReleasedMaxDisplacement),
		double(gMetrics.sceneRigidAttachmentMaxDrift),
		double(gMetrics.sceneRigidAttachmentMaxRigidDisplacement),
		double(gMetrics.sceneRigidAttachmentMaxRigidSpeed),
		double(gMetrics.sceneRigidAttachmentReleasedSeparation),
		double(gMetrics.sceneArticulationRootMaxDisplacement),
		double(
			gMetrics.sceneArticulationChildMaxForbiddenDisplacement),
		double(
			gMetrics.sceneArticulationChildMaxAngularDisplacement),
		double(gMetrics.sceneElementFilterMinY),
		double(gMetrics.sceneElementFilterFinalMinY),
		double(gMetrics.scenePartialFilterUnfilteredMinY),
		double(gMetrics.sceneKinematicMaxPoseError),
		double(gMetrics.sceneKinematicSoftDisplacement),
		double(gMetrics.sceneKinematicFinalY),
		double(gMetrics.sceneVolumeTargetFinalMaxError),
		double(gMetrics.sceneVolumeTargetMaxDisplacement),
		double(
			gMetrics.
				sceneVolumePartialInactiveDecoyDistance),
		double(gSceneCpuDynamicInitialY),
		double(gMetrics.sceneDynamicMinY),
		double(gMetrics.sceneDynamicFinalY),
		double(gMetrics.sceneDynamicMaxDrop),
		double(gMetrics.sceneDynamicPreContactMaxDrop),
		double(gMetrics.sceneDynamicMaxDownSpeed),
		double(gSceneCpuSecondDynamicInitialY),
		double(gMetrics.sceneSecondDynamicMinY),
		double(gMetrics.sceneSecondDynamicFinalY),
		double(gMetrics.sceneSecondDynamicMaxDrop),
		double(gMetrics.sceneSecondDynamicPreContactMaxDrop),
		double(gMetrics.sceneSecondDynamicMaxDownSpeed),
		double(gMetrics.minDynamicSurfaceSeparation),
		double(gMetrics.finalDynamicSurfaceSeparation),
		double(gMetrics.motionMaxVelocityFirstStepDisplacement),
		double(gMetrics.motionMaxVelocityFirstStepSpeed),
		double(gMetrics.motionSettlingFinalSpeed),
		double(gMetrics.motionControlFinalSpeed),
		double(gMetrics.depenetrationLimitedFirstStepRise),
		double(gMetrics.depenetrationControlFirstStepRise),
		double(gMetrics.depenetrationLimitedFinalRise),
		double(gMetrics.depenetrationLimitedMaxSpeed),
		double(gMetrics.speculativeCcdPositiveMinY),
		double(gMetrics.speculativeCcdPositiveMinSeparation),
		double(gMetrics.speculativeCcdNegativeMaxY),
		double(gMetrics.movingSpherePositiveDisplacement),
		double(gMetrics.movingSphereNegativeDisplacement),
		double(gMetrics.movingSpherePositiveMinSeparation),
		double(gMetrics.dynamicSphereSweepPositiveSoftDisplacement),
		double(gMetrics.dynamicSphereSweepNegativeSoftDisplacement),
		double(gMetrics.dynamicSphereSweepPositiveRigidDrop),
		double(gMetrics.dynamicSphereSweepNegativeRigidDrop),
		double(gMetrics.dynamicSphereSweepPositiveMinSeparation),
		gMetrics.solverReadbackMatched ? 1u : 0u,
		gErrorCallback.getFatalCount(), gErrorCallback.getWarningCount(),
		gMetrics.cleanupComplete);
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.caseName = AVBD_VOLUME_DEFAULT_CASE;
	defaults.frames = 600;
	defaults.dispatcherThreads = 2;
	std::string parseError;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, parseError))
	{
		printf("[AVBD_GATE_CONFIG_ERROR] %s\n", parseError.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	for(int argId = 1; argId < argc; ++argId)
	{
		if(!Snippets::isCommonHeadlessOption(argv[argId]))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] unknown option: %s\n",
				argv[argId]);
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
	}
	if(gHeadlessOptions.headless)
	{
		if(gHeadlessOptions.solverType != PxSolverType::eAVBD)
		{
			printf(
				"[AVBD_GATE_UNSUPPORTED] reason=component-is-avbd-only\n");
			return Snippets::eHEADLESS_UNSUPPORTED;
		}
		if(!isKnownCase(gHeadlessOptions.caseName))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] unknown case: %s\n",
				gHeadlessOptions.caseName.c_str());
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		const char* warmupEnvironment =
			std::getenv("PHYSX_AVBD_PROFILE_WARMUP");
		gProfileWarmupFrames = 0;
		if(warmupEnvironment && warmupEnvironment[0] &&
			(!Snippets::parseU32(
				warmupEnvironment, 0, 100000000u,
				gProfileWarmupFrames) ||
			 gProfileWarmupFrames >= gHeadlessOptions.frames))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] "
				"invalid PHYSX_AVBD_PROFILE_WARMUP\n");
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] "
				"failed to apply execution environment\n");
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		Snippets::printHeadlessConfig(
			AVBD_VOLUME_SNIPPET_NAME, gHeadlessOptions);
		bool initialized = initPhysicsInternal(
			false, gHeadlessOptions.caseName);
		if(initialized)
		{
			for(PxU32 frame = 0;
				frame < gHeadlessOptions.frames; ++frame)
			{
				if(!stepPhysicsInternal(gHeadlessOptions.dt))
					break;
			}
			finalizeMetrics();
			finalizePerformanceMetrics();
		}
		cleanupPhysics(false);
		const bool passed =
			initialized &&
			validateHeadlessResult(gHeadlessOptions.caseName);
		printPerformanceResult();
		printHeadlessResult(passed);
		return passed ?
			Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
	}

#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	printf("%s: No render snippet, nothing to do.\n",
		AVBD_VOLUME_SNIPPET_NAME);
#endif

	return 0;
}
