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
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

// ****************************************************************************
// This snippet demonstrates the use of Reduced Coordinates articulations.
// ****************************************************************************

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"

using namespace physx;

static PxDefaultAllocator						gAllocator;
static Snippets::TrackingErrorCallback		gErrorCallback;
static PxFoundation*							gFoundation		= NULL;
static PxPhysics*								gPhysics		= NULL;
static PxDefaultCpuDispatcher*					gDispatcher		= NULL;
static PxScene*									gScene			= NULL;
static PxMaterial*								gMaterial		= NULL;
static PxPvd*									gPvd			= NULL;
static PxPvdTransport*							gPvdTransport	= NULL;
static PxArticulationReducedCoordinate*			gArticulation	= NULL;
static PxArticulationJointReducedCoordinate*	gDriveJoint		= NULL;
static PxSolverType::Enum						gSolverType		= PxSolverType::eAVBD;
static PxArticulationLink*						gBaseLink		= NULL;
static PxArticulationLink*						gTopLink		= NULL;
static PxArticulationLink*						gDriveLink		= NULL;
static PxD6Joint*								gLoopJoints[8]	= {};
static PxU32									gLoopJointCount	= 0;
static Snippets::HeadlessOptions				gHeadlessOptions;
static bool									gExtensionsInitialized = false;
static bool									gInitializationFailed = false;
static bool									gRuntimeInvariantFailed = false;
static bool									gCleanupFailed = false;

static void resetScissorState();
void cleanupPhysics(bool interactive);

static bool failInitialization()
{
	gInitializationFailed = true;
	return false;
}

static bool trackLoopJoint(PxD6Joint* joint)
{
	PX_ASSERT(gLoopJointCount < sizeof(gLoopJoints) / sizeof(gLoopJoints[0]));
	if(!joint || gLoopJointCount >= sizeof(gLoopJoints) / sizeof(gLoopJoints[0]))
	{
		PX_RELEASE(joint);
		return failInitialization();
	}

	gLoopJoints[gLoopJointCount++] = joint;
	joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
	joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
	joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
	return true;
}

static PxArticulationLink* createLinkWithShape(PxArticulationLink* parent,
	const PxTransform& pose, const PxGeometry& geometry, PxReal density)
{
	if(!gArticulation || !gMaterial)
	{
		failInitialization();
		return NULL;
	}

	PxArticulationLink* link = gArticulation->createLink(parent, pose);
	if(!link)
	{
		failInitialization();
		return NULL;
	}
	PxShape* shape = PxRigidActorExt::createExclusiveShape(
		*link, geometry, *gMaterial);
	if(!shape || !PxRigidBodyExt::updateMassAndInertia(*link, density))
	{
		failInitialization();
		return NULL;
	}
	return link;
}

static PxArticulationJointReducedCoordinate* getInboundJointChecked(
	PxArticulationLink* link)
{
	PxArticulationJointReducedCoordinate* joint =
		link ? link->getInboundJoint() : NULL;
	if(!joint)
		failInitialization();
	return joint;
}

static bool createFallingBox(const PxTransform& pose, const PxVec3& halfExt,
	PxReal density, PxShape*& shape)
{
	shape = NULL;
	if(!gPhysics || !gScene || !gMaterial)
		return failInitialization();

	PxRigidDynamic* actor = gPhysics->createRigidDynamic(pose);
	if(!actor)
		return failInitialization();
	shape = PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExt), *gMaterial);
	if(!shape || !PxRigidBodyExt::updateMassAndInertia(*actor, density))
	{
		PX_RELEASE(actor);
		shape = NULL;
		return failInitialization();
	}
	if(!gScene->addActor(*actor))
	{
		PX_RELEASE(actor);
		shape = NULL;
		return failInitialization();
	}
	return true;
}

static PxFilterFlags scissorFilter(	PxFilterObjectAttributes attributes0, PxFilterData filterData0,
									PxFilterObjectAttributes attributes1, PxFilterData filterData1,
									PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(constantBlock);
	PX_UNUSED(constantBlockSize);
	if (filterData0.word2 != 0 && filterData0.word2 == filterData1.word2)
		return PxFilterFlag::eKILL;
	pairFlags |= PxPairFlag::eCONTACT_DEFAULT;
	return PxFilterFlag::eDEFAULT;
}

static void createScissorLift()
{
	const PxReal runnerLength = 2.f;
	const PxReal placementDistance = 1.8f;

	const PxReal cosAng = (placementDistance) / (runnerLength);

	const PxReal angle = PxAcos(cosAng);

	const PxReal sinAng = PxSin(angle);

	const PxQuat leftRot(-angle, PxVec3(1.f, 0.f, 0.f));
	const PxQuat rightRot(angle, PxVec3(1.f, 0.f, 0.f));

	//(1) Create base...
	PxArticulationLink* base = createLinkWithShape(NULL,
		PxTransform(PxVec3(0.f, 0.25f, 0.f)),
		PxBoxGeometry(0.5f, 0.25f, 1.5f), 3.f);
	if(!base)
		return;
	gBaseLink = base;

	//Now create the slider and fixed joints...

	gArticulation->setSolverIterationCounts(10);

	PxArticulationLink* leftRoot = createLinkWithShape(base,
		PxTransform(PxVec3(0.f, 0.55f, -0.9f)),
		PxBoxGeometry(0.5f, 0.05f, 0.05f), 1.f);
	if(!leftRoot)
		return;

	PxArticulationLink* rightRoot = createLinkWithShape(base,
		PxTransform(PxVec3(0.f, 0.55f, 0.9f)),
		PxBoxGeometry(0.5f, 0.05f, 0.05f), 1.f);
	if(!rightRoot)
		return;
	gDriveLink = rightRoot;

	PxArticulationJointReducedCoordinate* joint =
		getInboundJointChecked(leftRoot);
	if(!joint)
		return;
	joint->setJointType(PxArticulationJointType::eFIX);
	joint->setParentPose(PxTransform(PxVec3(0.f, 0.25f, -0.9f)));
	joint->setChildPose(PxTransform(PxVec3(0.f, -0.05f, 0.f)));

	//Set up the drive joint...	
	gDriveJoint = getInboundJointChecked(rightRoot);
	if(!gDriveJoint)
		return;
	gDriveJoint->setJointType(PxArticulationJointType::ePRISMATIC);
	gDriveJoint->setMotion(PxArticulationAxis::eZ, PxArticulationMotion::eLIMITED);
	gDriveJoint->setLimitParams(PxArticulationAxis::eZ, PxArticulationLimit(-1.4f, 0.2f));
	gDriveJoint->setDriveParams(PxArticulationAxis::eZ, PxArticulationDrive(100000.f, 0.f, PX_MAX_F32));

	gDriveJoint->setParentPose(PxTransform(PxVec3(0.f, 0.25f, 0.9f)));
	gDriveJoint->setChildPose(PxTransform(PxVec3(0.f, -0.05f, 0.f)));


	const PxU32 linkHeight = 3;
	PxArticulationLink* currLeft = leftRoot, *currRight = rightRoot;

	PxQuat rightParentRot(PxIdentity);
	PxQuat leftParentRot(PxIdentity);
	for (PxU32 i = 0; i < linkHeight; ++i)
	{
		const PxVec3 pos(0.5f, 0.55f + 0.1f*(1 + i), 0.f);
		PxArticulationLink* leftLink = createLinkWithShape(currLeft,
			PxTransform(pos + PxVec3(0.f, sinAng*(2 * i + 1), 0.f), leftRot),
			PxBoxGeometry(0.05f, 0.05f, 1.f), 1.f);
		if(!leftLink)
			return;

		const PxVec3 leftAnchorLocation = pos + PxVec3(0.f, sinAng*(2 * i), -0.9f);

		joint = getInboundJointChecked(leftLink);
		if(!joint)
			return;
		joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchorLocation), leftParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, -1.f), rightRot));
		joint->setJointType(PxArticulationJointType::eREVOLUTE);

		leftParentRot = leftRot;

		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));


		PxArticulationLink* rightLink = createLinkWithShape(currRight,
			PxTransform(pos + PxVec3(0.f, sinAng*(2 * i + 1), 0.f), rightRot),
			PxBoxGeometry(0.05f, 0.05f, 1.f), 1.f);
		if(!rightLink)
			return;

		const PxVec3 rightAnchorLocation = pos + PxVec3(0.f, sinAng*(2 * i), 0.9f);

		joint = getInboundJointChecked(rightLink);
		if(!joint)
			return;
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchorLocation), rightParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, 1.f), leftRot));
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));

		rightParentRot = rightRot;

		PxD6Joint* d6joint = PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity), rightLink, PxTransform(PxIdentity));
		if(!trackLoopJoint(d6joint))
			return;

		currLeft = rightLink;
		currRight = leftLink;
	}

	
	PxArticulationLink* leftTop = createLinkWithShape(currLeft,
		currLeft->getGlobalPose().transform(PxTransform(
			PxVec3(-0.5f, 0.f, -1.0f), leftParentRot)),
		PxBoxGeometry(0.5f, 0.05f, 0.05f), 1.f);
	if(!leftTop)
		return;

	PxArticulationLink* rightTop = createLinkWithShape(currRight,
		currRight->getGlobalPose().transform(PxTransform(
			PxVec3(-0.5f, 0.f, 1.0f), rightParentRot)),
		PxCapsuleGeometry(0.05f, 0.8f), 1.f);
	if(!rightTop)
		return;

	joint = getInboundJointChecked(leftTop);
	if(!joint)
		return;
	joint->setParentPose(PxTransform(PxVec3(0.f, 0.f, -1.f), currLeft->getGlobalPose().q.getConjugate()));
	joint->setChildPose(PxTransform(PxVec3(0.5f, 0.f, 0.f), leftTop->getGlobalPose().q.getConjugate()));
	joint->setJointType(PxArticulationJointType::eREVOLUTE);
	joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);

	joint = getInboundJointChecked(rightTop);
	if(!joint)
		return;
	joint->setParentPose(PxTransform(PxVec3(0.f, 0.f, 1.f), currRight->getGlobalPose().q.getConjugate()));
	joint->setChildPose(PxTransform(PxVec3(0.5f, 0.f, 0.f), rightTop->getGlobalPose().q.getConjugate()));
	joint->setJointType(PxArticulationJointType::eREVOLUTE);
	joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);


	currLeft = leftRoot;
	currRight = rightRoot;

	rightParentRot = PxQuat(PxIdentity);
	leftParentRot = PxQuat(PxIdentity);

	for (PxU32 i = 0; i < linkHeight; ++i)
	{
		const PxVec3 pos(-0.5f, 0.55f + 0.1f*(1 + i), 0.f);
		PxArticulationLink* leftLink = createLinkWithShape(currLeft,
			PxTransform(pos + PxVec3(0.f, sinAng*(2 * i + 1), 0.f), leftRot),
			PxBoxGeometry(0.05f, 0.05f, 1.f), 1.f);
		if(!leftLink)
			return;

		const PxVec3 leftAnchorLocation = pos + PxVec3(0.f, sinAng*(2 * i), -0.9f);

		joint = getInboundJointChecked(leftLink);
		if(!joint)
			return;
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchorLocation), leftParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, -1.f), rightRot));

		leftParentRot = leftRot;

		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));

		PxArticulationLink* rightLink = createLinkWithShape(currRight,
			PxTransform(pos + PxVec3(0.f, sinAng*(2 * i + 1), 0.f), rightRot),
			PxBoxGeometry(0.05f, 0.05f, 1.f), 1.f);
		if(!rightLink)
			return;

		const PxVec3 rightAnchorLocation = pos + PxVec3(0.f, sinAng*(2 * i), 0.9f);

		/*joint = PxD6JointCreate(getPhysics(), currRight, PxTransform(currRight->getGlobalPose().transformInv(rightAnchorLocation)),
		rightLink, PxTransform(PxVec3(0.f, 0.f, 1.f)));*/

		joint = getInboundJointChecked(rightLink);
		if(!joint)
			return;
		joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchorLocation), rightParentRot));
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, 1.f), leftRot));
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));

		rightParentRot = rightRot;

		PxD6Joint* d6joint = PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity), rightLink, PxTransform(PxIdentity));
		if(!trackLoopJoint(d6joint))
			return;

		currLeft = rightLink;
		currRight = leftLink;
	}

	PxD6Joint* d6joint = PxD6JointCreate(*gPhysics, currLeft, PxTransform(PxVec3(0.f, 0.f, -1.f)), leftTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
	if(!trackLoopJoint(d6joint))
		return;

	d6joint = PxD6JointCreate(*gPhysics, currRight, PxTransform(PxVec3(0.f, 0.f, 1.f)), rightTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
	if(!trackLoopJoint(d6joint))
		return;


	const PxTransform topPose(PxVec3(0.f, leftTop->getGlobalPose().p.y + 0.15f, 0.f));

	PxArticulationLink* top = createLinkWithShape(leftTop, topPose,
		PxBoxGeometry(0.5f, 0.1f, 1.5f), 1.f);
	if(!top)
		return;
	gTopLink = top;

	joint = getInboundJointChecked(top);
	if(!joint)
		return;
	joint->setJointType(PxArticulationJointType::eFIX);
	joint->setParentPose(PxTransform(PxVec3(0.f, 0.0f, 0.f)));
	joint->setChildPose(PxTransform(PxVec3(0.f, -0.15f, -0.9f)));

	if(!gScene->addArticulation(*gArticulation))
	{
		failInitialization();
		return;
	}

	for (PxU32 i = 0; i < gArticulation->getNbLinks(); ++i)
	{
		PxArticulationLink* link = NULL;
		if(gArticulation->getLinks(&link, 1, i) != 1 || !link
			|| link->getNbShapes() == 0)
		{
			failInitialization();
			return;
		}

		link->setLinearDamping(0.2f);
		link->setAngularDamping(0.2f);

		link->setMaxAngularVelocity(20.f);
		link->setMaxLinearVelocity(100.f);

		if (link != top)
		{
			for (PxU32 b = 0; b < link->getNbShapes(); ++b)
			{
				PxShape* shape = NULL;
				if(link->getShapes(&shape, 1, b) != 1 || !shape)
				{
					failInitialization();
					return;
				}

				shape->setSimulationFilterData(PxFilterData(0, 0, 1, 0));
			}
		}
	}

	const PxVec3 halfExt(0.25f);
	const PxReal density(0.5f);
	const PxVec3 boxPositions[] = {
		PxVec3(-0.25f, 5.f, 0.5f), PxVec3(0.25f, 5.f, 0.5f),
		PxVec3(-0.25f, 4.5f, 0.5f), PxVec3(0.25f, 4.5f, 0.5f),
		PxVec3(-0.25f, 5.f, 0.f), PxVec3(0.25f, 5.f, 0.f),
		PxVec3(-0.25f, 4.5f, 0.f), PxVec3(0.25f, 4.5f, 0.f)
	};
	PxShape* boxShapes[8] = {};
	for(PxU32 i = 0; i < sizeof(boxPositions) / sizeof(boxPositions[0]); ++i)
	{
		if(!createFallingBox(PxTransform(boxPositions[i]), halfExt, density,
			boxShapes[i]))
			return;
	}

	const float contactOffset = 0.2f;
	for(PxU32 i = 0; i < sizeof(boxShapes) / sizeof(boxShapes[0]); ++i)
	{
		if(!boxShapes[i])
		{
			failInitialization();
			return;
		}
		boxShapes[i]->setContactOffset(contactOffset);
	}
}

static bool hasOwnedPhysicsResources()
{
	return gFoundation || gPhysics || gDispatcher || gScene || gMaterial || gPvd
		|| gPvdTransport || gArticulation || gLoopJointCount
		|| gExtensionsInitialized;
}

static bool validateScissorSetup()
{
	static const PxU32 expectedLinkCount = 18;
	static const PxU32 expectedLoopJointCount = 8;
	if(gInitializationFailed || !gFoundation || !gPhysics || !gDispatcher
		|| !gScene || !gMaterial || !gArticulation || !gBaseLink || !gTopLink
		|| !gDriveLink || !gDriveJoint
		|| gArticulation->getScene() != gScene
		|| gArticulation->getNbLinks() != expectedLinkCount
		|| gLoopJointCount != expectedLoopJointCount
		|| gDriveLink->getInboundJoint() != gDriveJoint
		|| gDriveJoint->getJointType() != PxArticulationJointType::ePRISMATIC
		|| gDriveJoint->getMotion(PxArticulationAxis::eZ)
			!= PxArticulationMotion::eLIMITED)
		return false;

	for(PxU32 i = 0; i < expectedLinkCount; ++i)
	{
		PxArticulationLink* link = NULL;
		if(gArticulation->getLinks(&link, 1, i) != 1 || !link
			|| !link->getGlobalPose().isSane() || link->getNbShapes() == 0
			|| (link != gBaseLink && !link->getInboundJoint()))
			return false;
	}
	for(PxU32 i = 0; i < expectedLoopJointCount; ++i)
	{
		if(!gLoopJoints[i])
			return false;
		PxRigidActor* actor0 = NULL;
		PxRigidActor* actor1 = NULL;
		gLoopJoints[i]->getActors(actor0, actor1);
		if(!actor0 || !actor1)
			return false;
	}
	return true;
}

static bool initializePhysics(bool interactive)
{
	if(hasOwnedPhysicsResources())
		return failInitialization();

	resetScissorState();
	gErrorCallback.reset();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return failInitialization();

	if(interactive)
	{
		gPvd = PxCreatePvd(*gFoundation);
		if(gPvd)
		{
			gPvdTransport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
			if(gPvdTransport)
				gPvd->connect(*gPvdTransport, PxPvdInstrumentationFlag::eALL);
		}
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
		PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return failInitialization();
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	if(!gExtensionsInitialized)
		return failInitialization();

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);

	const PxU32 numCores = SnippetUtils::getNbPhysicalCores();
	const PxU32 dispatcherThreads = interactive
		? (numCores == 0 ? 0 : numCores - 1)
		: gHeadlessOptions.dispatcherThreads;
	gDispatcher = PxDefaultCpuDispatcherCreate(dispatcherThreads);
	if(!gDispatcher)
		return failInitialization();
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.solverType = gSolverType;
	sceneDesc.filterShader = scissorFilter;

	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return failInitialization();

	printf("[SnippetArticulationRCConfig] solver=%s dispatcherThreads=%u "
		"pvd=%s\n", Snippets::getSolverTypeName(sceneDesc.solverType),
		dispatcherThreads, gPvd ? "enabled" : "disabled");
	if(interactive)
	{
		PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
		if(pvdClient)
		{
			pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
			pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
			pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
		}
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.f);
	if(!gMaterial)
		return failInitialization();

	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	if(!groundPlane || groundPlane->getNbShapes() == 0)
	{
		PX_RELEASE(groundPlane);
		return failInitialization();
	}
	if(!gScene->addActor(*groundPlane))
	{
		PX_RELEASE(groundPlane);
		return failInitialization();
	}

	// TODO(AVBD): PhysX 5 removed the old solver-neutral PxArticulation layer,
	// so AVBD currently keeps the reduced-coordinate factory name even when the
	// articulation/joint handling behind it is maximal-coordinate oriented.
	gArticulation = gPhysics->createArticulationReducedCoordinate();
	if(!gArticulation)
		return failInitialization();

	createScissorLift();
	if(!validateScissorSetup() || gErrorCallback.getFatalCount())
		return failInitialization();
	return true;
}

void initPhysics(bool interactive)
{
	if(initializePhysics(interactive))
		return;

	if(interactive)
	{
		std::fprintf(stderr,
			"SnippetArticulationRC interactive initialization failed.\n");
		cleanupPhysics(false);
		std::exit(Snippets::eHEADLESS_CONFIG_ERROR);
	}
}

static bool gClosing = true;
static PxU32 gFrame = 0;
static PxU32 gSimulateFailures = 0;
static PxU32 gFetchFailures = 0;
static PxU32 gFetchErrorState = 0;
static bool gFetchPending = false;
static PxReal gPendingDt = 0.0f;
static PxReal gPendingDriveValue = 0.0f;

struct ScissorStats
{
	PxReal topYInitial = 0.0f;
	PxReal topYMin = PX_MAX_F32;
	PxReal topYMax = -PX_MAX_F32;
	PxReal topYLast = 0.0f;
	PxReal baseYDriftMax = 0.0f;
	PxReal baseTiltDegMax = 0.0f;
	PxU32 nonFiniteFrame = PX_MAX_U32;
	PxU32 firstReportedFrame = PX_MAX_U32;
	PxReal driveCoord = 0.0f;
	PxReal driveCoordVelocity = 0.0f;
	PxReal driveErrorMax = 0.0f;
	PxU32 firstDriveErrorFrame = PX_MAX_U32;
	PxU32 firstStallFrame = PX_MAX_U32;
	PxReal stallWindowCoord = 0.0f;
	PxReal stallWindowTarget = 0.0f;
	PxU32 stallWindowFrame = PX_MAX_U32;
	PxReal internalAnchorError = 0.0f;
	PxU32 internalAnchorLink = PX_MAX_U32;
	PxReal internalAnchorErrorMax = 0.0f;
	PxU32 internalAnchorErrorMaxLink = PX_MAX_U32;
	PxU32 firstInternalAnchorBadFrame = PX_MAX_U32;
	PxU32 firstInternalAnchorBadLink = PX_MAX_U32;
	PxReal internalAngularErrorDeg = 0.0f;
	PxU32 internalAngularLink = PX_MAX_U32;
	PxReal internalAngularErrorDegMax = 0.0f;
	PxU32 internalAngularErrorMaxLink = PX_MAX_U32;
	PxReal twistLimitViolationDeg = 0.0f;
	PxU32 twistLimitViolationLink = PX_MAX_U32;
	PxReal twistLimitViolationDegMax = 0.0f;
	PxU32 twistLimitViolationMaxLink = PX_MAX_U32;
	PxU32 firstInternalAngularBadFrame = PX_MAX_U32;
	PxU32 firstInternalAngularBadLink = PX_MAX_U32;
	PxReal loopAnchorError = 0.0f;
	PxU32 loopAnchorIndex = PX_MAX_U32;
	PxReal loopAnchorErrorMax = 0.0f;
	PxU32 loopAnchorErrorMaxIndex = PX_MAX_U32;
	PxU32 firstLoopAnchorBadFrame = PX_MAX_U32;
	PxU32 firstLoopAnchorBadIndex = PX_MAX_U32;
	PxU32 phaseHeightCount = 0;
	PxReal phaseFirstHeight = 0.0f;
	PxReal phaseReferenceHeight = 0.0f;
	PxReal phaseBaselineSpread = PX_MAX_F32;
	PxReal phaseMaxRelativeDrift = 0.0f;
	PxU32 firstBadPhaseSample = PX_MAX_U32;
	PxU32 consecutiveBadPhaseSamples = 0;
	bool phaseRegressionFailed = false;
	PxReal previousPhaseCoord = 0.0f;
	PxReal previousPhaseHeight = 0.0f;
	bool phaseStateValid = false;
};

static ScissorStats gScissorStats;

static bool areScissorStatsFinite()
{
	return PxIsFinite(gScissorStats.topYInitial)
		&& PxIsFinite(gScissorStats.topYMin)
		&& PxIsFinite(gScissorStats.topYMax)
		&& PxIsFinite(gScissorStats.topYLast)
		&& PxIsFinite(gScissorStats.baseYDriftMax)
		&& PxIsFinite(gScissorStats.baseTiltDegMax)
		&& PxIsFinite(gScissorStats.driveCoord)
		&& PxIsFinite(gScissorStats.driveCoordVelocity)
		&& PxIsFinite(gScissorStats.driveErrorMax)
		&& PxIsFinite(gScissorStats.stallWindowCoord)
		&& PxIsFinite(gScissorStats.stallWindowTarget)
		&& PxIsFinite(gScissorStats.internalAnchorError)
		&& PxIsFinite(gScissorStats.internalAnchorErrorMax)
		&& PxIsFinite(gScissorStats.internalAngularErrorDeg)
		&& PxIsFinite(gScissorStats.internalAngularErrorDegMax)
		&& PxIsFinite(gScissorStats.twistLimitViolationDeg)
		&& PxIsFinite(gScissorStats.twistLimitViolationDegMax)
		&& PxIsFinite(gScissorStats.loopAnchorError)
		&& PxIsFinite(gScissorStats.loopAnchorErrorMax)
		&& PxIsFinite(gScissorStats.phaseFirstHeight)
		&& PxIsFinite(gScissorStats.phaseReferenceHeight)
		&& PxIsFinite(gScissorStats.phaseBaselineSpread)
		&& PxIsFinite(gScissorStats.phaseMaxRelativeDrift)
		&& PxIsFinite(gScissorStats.previousPhaseCoord)
		&& PxIsFinite(gScissorStats.previousPhaseHeight);
}

static void resetScissorState()
{
	gInitializationFailed = false;
	gRuntimeInvariantFailed = false;
	gCleanupFailed = false;
	gExtensionsInitialized = false;
	gClosing = true;
	gFrame = 0;
	gSimulateFailures = 0;
	gFetchFailures = 0;
	gFetchErrorState = 0;
	gFetchPending = false;
	gPendingDt = 0.0f;
	gPendingDriveValue = 0.0f;
	gScissorStats = ScissorStats();
	gBaseLink = NULL;
	gTopLink = NULL;
	gDriveLink = NULL;
	gDriveJoint = NULL;
	gLoopJointCount = 0;
	for(PxU32 i = 0; i < sizeof(gLoopJoints) / sizeof(gLoopJoints[0]); ++i)
		gLoopJoints[i] = NULL;
}

static bool isArticulationStateFinite()
{
	if(!gArticulation || !gDriveJoint || !PxIsFinite(gScissorStats.driveCoord)
		|| !PxIsFinite(gScissorStats.driveCoordVelocity)
		|| !PxIsFinite(gDriveJoint->getDriveTarget(PxArticulationAxis::eZ))
		|| !PxIsFinite(gDriveJoint->getJointPosition(PxArticulationAxis::eZ))
		|| !PxIsFinite(gDriveJoint->getJointVelocity(PxArticulationAxis::eZ)))
		return false;

	const PxU32 linkCount = gArticulation->getNbLinks();
	for(PxU32 i = 0; i < linkCount; ++i)
	{
		PxArticulationLink* link = NULL;
		if(gArticulation->getLinks(&link, 1, i) != 1 || !link)
			return false;
		const PxTransform pose = link->getGlobalPose();
		if(!pose.isSane() || !link->getLinearVelocity().isFinite()
			|| !link->getAngularVelocity().isFinite())
			return false;
	}
	return true;
}

static void recordPhaseHeightSample(PxReal sampleHeight)
{
	const PxU32 sampleNumber = ++gScissorStats.phaseHeightCount;
	if(sampleNumber == 1)
	{
		gScissorStats.phaseFirstHeight = sampleHeight;
		return;
	}
	if(sampleNumber == 2)
	{
		gScissorStats.phaseReferenceHeight = 0.5f
			* (gScissorStats.phaseFirstHeight + sampleHeight);
		const PxReal referenceScale = PxMax(
			PxAbs(gScissorStats.phaseReferenceHeight), 0.1f);
		gScissorStats.phaseBaselineSpread = PxAbs(sampleHeight
			- gScissorStats.phaseFirstHeight) / referenceScale;
		return;
	}

	const PxReal referenceScale = PxMax(
		PxAbs(gScissorStats.phaseReferenceHeight), 0.1f);
	const PxReal drift = PxAbs(sampleHeight
		- gScissorStats.phaseReferenceHeight) / referenceScale;
	gScissorStats.phaseMaxRelativeDrift = PxMax(
		gScissorStats.phaseMaxRelativeDrift, drift);
	if(drift > 0.20f)
	{
		if(gScissorStats.firstBadPhaseSample == PX_MAX_U32)
			gScissorStats.firstBadPhaseSample = sampleNumber;
		if(++gScissorStats.consecutiveBadPhaseSamples >= 2)
			gScissorStats.phaseRegressionFailed = true;
	}
	else
	{
		gScissorStats.consecutiveBadPhaseSamples = 0;
	}
}

static PxReal getDriveCoordinate()
{
	if(!gBaseLink || !gDriveLink || !gDriveJoint)
		return 0.0f;
	const PxTransform parentFrame = gBaseLink->getGlobalPose() * gDriveJoint->getParentPose();
	const PxTransform childFrame = gDriveLink->getGlobalPose() * gDriveJoint->getChildPose();
	return parentFrame.q.getBasisVector2().dot(childFrame.p - parentFrame.p);
}

static bool updateAnchorErrors(PxU32 frame)
{
	gScissorStats.internalAnchorError = 0.0f;
	gScissorStats.internalAnchorLink = PX_MAX_U32;
	gScissorStats.internalAngularErrorDeg = 0.0f;
	gScissorStats.internalAngularLink = PX_MAX_U32;
	gScissorStats.twistLimitViolationDeg = 0.0f;
	gScissorStats.twistLimitViolationLink = PX_MAX_U32;
	const PxU32 linkCount = gArticulation->getNbLinks();
	for(PxU32 i = 1; i < linkCount; ++i)
	{
		PxArticulationLink* child = NULL;
		if(gArticulation->getLinks(&child, 1, i) != 1 || !child)
		{
			gRuntimeInvariantFailed = true;
			return false;
		}
		PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
		if(!joint)
		{
			gRuntimeInvariantFailed = true;
			return false;
		}
		const PxTransform parentFrame = joint->getParentArticulationLink().getGlobalPose()
			* joint->getParentPose();
		const PxTransform childFrame = child->getGlobalPose() * joint->getChildPose();
		PxVec3 anchorDelta = childFrame.p - parentFrame.p;
		if(joint == gDriveJoint)
		{
			const PxVec3 driveAxis = parentFrame.q.getBasisVector2();
			anchorDelta -= driveAxis * driveAxis.dot(anchorDelta);
		}
		const PxReal error = anchorDelta.magnitude();
		if(error > gScissorStats.internalAnchorError)
		{
			gScissorStats.internalAnchorError = error;
			gScissorStats.internalAnchorLink = i;
		}

		if(joint->getJointType() == PxArticulationJointType::eREVOLUTE)
		{
			PxQuat relativeQ = parentFrame.q.getConjugate() * childFrame.q;
			if(relativeQ.w < 0.0f)
				relativeQ = PxQuat(-relativeQ.x, -relativeQ.y, -relativeQ.z, -relativeQ.w);
			relativeQ.normalize();
			PxQuat twist(relativeQ.x, 0.0f, 0.0f, relativeQ.w);
			if(twist.magnitudeSquared() > 1.0e-12f)
				twist.normalize();
			else
				twist = PxQuat(PxIdentity);
			PxQuat swing = relativeQ * twist.getConjugate();
			if(swing.w < 0.0f)
				swing = PxQuat(-swing.x, -swing.y, -swing.z, -swing.w);
			swing.normalize();
			const PxReal swingErrorDeg = swing.getAngle() * 180.0f / PxPi;
			if(swingErrorDeg > gScissorStats.internalAngularErrorDeg)
			{
				gScissorStats.internalAngularErrorDeg = swingErrorDeg;
				gScissorStats.internalAngularLink = i;
			}

			if(joint->getMotion(PxArticulationAxis::eTWIST) ==
				PxArticulationMotion::eLIMITED)
			{
				const PxReal twistAngle = 2.0f * PxAtan2(twist.x, twist.w);
				const PxArticulationLimit limit =
					joint->getLimitParams(PxArticulationAxis::eTWIST);
				PxReal violation = 0.0f;
				if(twistAngle < limit.low)
					violation = limit.low - twistAngle;
				else if(twistAngle > limit.high)
					violation = twistAngle - limit.high;
				const PxReal violationDeg = violation * 180.0f / PxPi;
				if(violationDeg > gScissorStats.twistLimitViolationDeg)
				{
					gScissorStats.twistLimitViolationDeg = violationDeg;
					gScissorStats.twistLimitViolationLink = i;
				}
			}
		}
	}
	if(gScissorStats.internalAnchorError > gScissorStats.internalAnchorErrorMax)
	{
		gScissorStats.internalAnchorErrorMax = gScissorStats.internalAnchorError;
		gScissorStats.internalAnchorErrorMaxLink = gScissorStats.internalAnchorLink;
	}
	if(gScissorStats.firstInternalAnchorBadFrame == PX_MAX_U32
		&& gScissorStats.internalAnchorError > 0.1f)
	{
		gScissorStats.firstInternalAnchorBadFrame = frame;
		gScissorStats.firstInternalAnchorBadLink = gScissorStats.internalAnchorLink;
		printf("[ScissorInternalAnchor] firstFrame=%u link=%u error=%.4f\n",
			frame, gScissorStats.internalAnchorLink, gScissorStats.internalAnchorError);
	}
	if(gScissorStats.internalAngularErrorDeg > gScissorStats.internalAngularErrorDegMax)
	{
		gScissorStats.internalAngularErrorDegMax = gScissorStats.internalAngularErrorDeg;
		gScissorStats.internalAngularErrorMaxLink = gScissorStats.internalAngularLink;
	}
	if(gScissorStats.twistLimitViolationDeg > gScissorStats.twistLimitViolationDegMax)
	{
		gScissorStats.twistLimitViolationDegMax = gScissorStats.twistLimitViolationDeg;
		gScissorStats.twistLimitViolationMaxLink = gScissorStats.twistLimitViolationLink;
	}
	if(gScissorStats.firstInternalAngularBadFrame == PX_MAX_U32
		&& gScissorStats.internalAngularErrorDeg > 5.0f)
	{
		gScissorStats.firstInternalAngularBadFrame = frame;
		gScissorStats.firstInternalAngularBadLink = gScissorStats.internalAngularLink;
		printf("[ScissorRevoluteSwing] firstFrame=%u link=%u errorDeg=%.3f\n",
			frame, gScissorStats.internalAngularLink,
			gScissorStats.internalAngularErrorDeg);
	}

	gScissorStats.loopAnchorError = 0.0f;
	gScissorStats.loopAnchorIndex = PX_MAX_U32;
	for(PxU32 i = 0; i < gLoopJointCount; ++i)
	{
		if(!gLoopJoints[i])
		{
			gRuntimeInvariantFailed = true;
			return false;
		}
		PxRigidActor* actor0 = NULL;
		PxRigidActor* actor1 = NULL;
		gLoopJoints[i]->getActors(actor0, actor1);
		if(!actor0 || !actor1)
		{
			gRuntimeInvariantFailed = true;
			return false;
		}
		const PxTransform frame0 = actor0->getGlobalPose()
			* gLoopJoints[i]->getLocalPose(PxJointActorIndex::eACTOR0);
		const PxTransform frame1 = actor1->getGlobalPose()
			* gLoopJoints[i]->getLocalPose(PxJointActorIndex::eACTOR1);
		const PxReal error = (frame1.p - frame0.p).magnitude();
		if(error > gScissorStats.loopAnchorError)
		{
			gScissorStats.loopAnchorError = error;
			gScissorStats.loopAnchorIndex = i;
		}
	}
	if(gScissorStats.loopAnchorError > gScissorStats.loopAnchorErrorMax)
	{
		gScissorStats.loopAnchorErrorMax = gScissorStats.loopAnchorError;
		gScissorStats.loopAnchorErrorMaxIndex = gScissorStats.loopAnchorIndex;
	}
	if(gScissorStats.firstLoopAnchorBadFrame == PX_MAX_U32
		&& gScissorStats.loopAnchorError > 0.1f)
	{
		gScissorStats.firstLoopAnchorBadFrame = frame;
		gScissorStats.firstLoopAnchorBadIndex = gScissorStats.loopAnchorIndex;
		printf("[ScissorLoopAnchor] firstFrame=%u loop=%u error=%.4f\n",
			frame, gScissorStats.loopAnchorIndex, gScissorStats.loopAnchorError);
	}
	return true;
}

static void dumpScissorState(PxU32 frame)
{
	if(!gBaseLink || !gTopLink || !gDriveJoint)
		return;
	const PxTransform basePose = gBaseLink->getGlobalPose();
	const PxTransform topPose  = gTopLink->getGlobalPose();
	const PxReal driveTarget  = gDriveJoint->getDriveTarget(PxArticulationAxis::eZ);
	const PxReal jointPos     = gDriveJoint->getJointPosition(PxArticulationAxis::eZ);
	const PxReal jointVel     = gDriveJoint->getJointVelocity(PxArticulationAxis::eZ);
	printf("[Scissor] frame=%u base=(%.3f,%.3f,%.3f) top=(%.3f,%.3f,%.3f) "
		"driveTarget=%.4f driveCoord=%.4f driveCoordVel=%.4f driveError=%.4f "
		"internalAnchor=%.4f(link=%u) revoluteSwingDeg=%.3f(link=%u) "
		"twistLimitViolationDeg=%.3f(link=%u) "
		"loopAnchor=%.4f(loop=%u) "
		"jointPos=%.4f jointVel=%.4f\n",
		frame, basePose.p.x, basePose.p.y, basePose.p.z,
		topPose.p.x, topPose.p.y, topPose.p.z,
		driveTarget, gScissorStats.driveCoord, gScissorStats.driveCoordVelocity,
		gScissorStats.driveCoord - driveTarget,
		gScissorStats.internalAnchorError, gScissorStats.internalAnchorLink,
		gScissorStats.internalAngularErrorDeg, gScissorStats.internalAngularLink,
		gScissorStats.twistLimitViolationDeg, gScissorStats.twistLimitViolationLink,
		gScissorStats.loopAnchorError, gScissorStats.loopAnchorIndex,
		jointPos, jointVel);
}

static void sampleFetchedFrame(PxReal dt, PxReal driveValue)
{
	if(!gScene || !gArticulation || !gDriveJoint || !gBaseLink || !gTopLink
		|| !gDriveLink)
	{
		gRuntimeInvariantFailed = true;
		return;
	}

	const PxReal previousDriveCoord = gScissorStats.driveCoord;
	gScissorStats.driveCoord = getDriveCoordinate();
	gScissorStats.driveCoordVelocity = gFrame
		? (gScissorStats.driveCoord - previousDriveCoord) / dt : 0.0f;
	if(!isArticulationStateFinite())
	{
		if(gScissorStats.nonFiniteFrame == PX_MAX_U32)
			gScissorStats.nonFiniteFrame = gFrame;
	}
	else
	{
		if(!updateAnchorErrors(gFrame))
			return;
		const PxReal driveError = gScissorStats.driveCoord - driveValue;
		gScissorStats.driveErrorMax = PxMax(gScissorStats.driveErrorMax,
			PxAbs(driveError));
		if(gScissorStats.firstDriveErrorFrame == PX_MAX_U32
			&& PxAbs(driveError) > 0.1f)
			gScissorStats.firstDriveErrorFrame = gFrame;

		// Detect a full one-second window in which the target moves by at least
		// 0.2 m while the geometric prismatic coordinate moves by at most 0.02 m.
		if(gScissorStats.stallWindowFrame == PX_MAX_U32)
		{
			gScissorStats.stallWindowFrame = gFrame;
			gScissorStats.stallWindowCoord = gScissorStats.driveCoord;
			gScissorStats.stallWindowTarget = driveValue;
		}
		else if(gFrame - gScissorStats.stallWindowFrame >= 60)
		{
			const PxReal targetDelta = driveValue
				- gScissorStats.stallWindowTarget;
			const PxReal coordDelta = gScissorStats.driveCoord
				- gScissorStats.stallWindowCoord;
			if(gScissorStats.firstStallFrame == PX_MAX_U32
				&& PxAbs(targetDelta) >= 0.2f && PxAbs(coordDelta) <= 0.02f
				&& PxAbs(driveError) > 0.1f)
			{
				gScissorStats.firstStallFrame = gScissorStats.stallWindowFrame;
				printf("[ScissorStall] firstFrame=%u endFrame=%u targetDelta=%.4f "
					"driveCoordDelta=%.4f driveError=%.4f\n",
					gScissorStats.firstStallFrame, gFrame, targetDelta,
					coordDelta, driveError);
			}
			gScissorStats.stallWindowFrame = gFrame;
			gScissorStats.stallWindowCoord = gScissorStats.driveCoord;
			gScissorStats.stallWindowTarget = driveValue;
		}

		if(gBaseLink && gTopLink)
		{
			const PxTransform basePose = gBaseLink->getGlobalPose();
			const PxTransform topPose  = gTopLink->getGlobalPose();
			const PxVec3 baseUp = basePose.q.rotate(PxVec3(0.f, 1.f, 0.f));
			const PxReal relativeHeight = baseUp.dot(topPose.p - basePose.p);
			if(!PxIsFinite(relativeHeight))
			{
				if(gScissorStats.nonFiniteFrame == PX_MAX_U32)
					gScissorStats.nonFiniteFrame = gFrame;
			}
			else
			{
				const PxReal sampleCoord = -0.6f;
				if(gScissorStats.phaseStateValid && gClosing
					&& gScissorStats.previousPhaseCoord > sampleCoord
					&& gScissorStats.driveCoord <= sampleCoord)
				{
					const PxReal denominator = gScissorStats.previousPhaseCoord
						- gScissorStats.driveCoord;
					const PxReal alpha = denominator > 1.0e-6f
						? PxClamp((gScissorStats.previousPhaseCoord - sampleCoord)
							/ denominator, 0.0f, 1.0f)
						: 1.0f;
					const PxReal sampleHeight = gScissorStats.previousPhaseHeight
						+ alpha * (relativeHeight
							- gScissorStats.previousPhaseHeight);
					if(PxIsFinite(sampleHeight))
					{
						recordPhaseHeightSample(sampleHeight);
						printf("[ScissorCycle] sample=%u frame=%u q=%.3f "
							"relativeHeight=%.4f\n",
							gScissorStats.phaseHeightCount, gFrame,
							sampleCoord, sampleHeight);
					}
					else if(gScissorStats.nonFiniteFrame == PX_MAX_U32)
					{
						gScissorStats.nonFiniteFrame = gFrame;
					}
				}
				gScissorStats.previousPhaseCoord = gScissorStats.driveCoord;
				gScissorStats.previousPhaseHeight = relativeHeight;
				gScissorStats.phaseStateValid = true;

				if(gScissorStats.firstReportedFrame == PX_MAX_U32)
				{
					gScissorStats.firstReportedFrame = gFrame;
					gScissorStats.topYInitial = topPose.p.y;
				}
				gScissorStats.topYLast = topPose.p.y;
				gScissorStats.topYMin = PxMin(gScissorStats.topYMin, topPose.p.y);
				gScissorStats.topYMax = PxMax(gScissorStats.topYMax, topPose.p.y);
				gScissorStats.baseYDriftMax = PxMax(gScissorStats.baseYDriftMax,
					PxAbs(basePose.p.y - 0.25f));
				// Base should stay flat on ground (no tilt). Compute tilt as angle
				// between local +Y and world +Y.
				const PxReal dotUp = PxClamp(baseUp.y, -1.0f, 1.0f);
				const PxReal tiltDeg = PxAcos(dotUp) * 180.0f / PxPi;
				gScissorStats.baseTiltDegMax = PxMax(
					gScissorStats.baseTiltDegMax, tiltDeg);
			}
		}
	}

	if(!areScissorStatsFinite()
		&& gScissorStats.nonFiniteFrame == PX_MAX_U32)
		gScissorStats.nonFiniteFrame = gFrame;

	const bool snapshotFrame = (gFrame % 600) == 0 || gFrame < 5;
	if(snapshotFrame)
		dumpScissorState(gFrame);
	++gFrame;
}

static bool finishPendingSimulation()
{
	if(!gFetchPending)
		return true;
	if(!gScene)
	{
		gRuntimeInvariantFailed = true;
		++gFetchFailures;
		return false;
	}

	PxU32 errorState = 0;
	if(!gScene->fetchResults(true, &errorState))
	{
		++gFetchFailures;
		gFetchErrorState |= errorState;
		return false;
	}

	gFetchPending = false;
	gFetchErrorState |= errorState;
	const PxReal dt = gPendingDt;
	const PxReal driveValue = gPendingDriveValue;
	gPendingDt = 0.0f;
	gPendingDriveValue = 0.0f;
	if(!PxIsFinite(dt) || !PxIsFinite(driveValue) || dt <= 0.0f)
	{
		gRuntimeInvariantFailed = true;
		return true;
	}
	sampleFetchedFrame(dt, driveValue);
	return true;
}

void stepPhysics(bool interactive)
{
	if(gFetchPending)
	{
		finishPendingSimulation();
		return;
	}
	if(!gScene || !gArticulation || !gDriveJoint || !gBaseLink || !gTopLink
		|| !gDriveLink)
	{
		if(!interactive)
			gRuntimeInvariantFailed = true;
		return;
	}

	const PxReal dt = interactive ? (1.0f / 60.f) : gHeadlessOptions.dt;
	PxReal driveValue = gDriveJoint->getDriveTarget(PxArticulationAxis::eZ);
	if (gClosing && driveValue < -1.2f)
		gClosing = false;
	else if (!gClosing && driveValue > 0.f)
		gClosing = true;

	if(gClosing)
		driveValue -= dt * 0.25f;
	else
		driveValue += dt * 0.25f;
	gDriveJoint->setDriveTarget(PxArticulationAxis::eZ, driveValue);
	if(!gScene->simulate(dt))
	{
		++gSimulateFailures;
		return;
	}

	gFetchPending = true;
	gPendingDt = dt;
	gPendingDriveValue = driveValue;
	finishPendingSimulation();
}

struct ScissorGateEvaluation
{
	Snippets::HeadlessExitCode exitCode;
	const char* status;
	const char* reason;
	PxU32 completedFrames;
	PxU32 oraclePassed;
	PxU32 linkCount;
	PxU32 loopJointCount;
	PxU32 nonFinite;
	PxU32 nonFiniteFrame;
	PxU32 firstStallFrame;
	PxReal twistLimitViolationDegMax;
	PxU32 twistLimitViolationMaxLink;
	PxReal internalAnchorErrorMax;
	PxU32 internalAnchorErrorMaxLink;
	PxU32 firstInternalAnchorBadFrame;
	PxU32 firstInternalAnchorBadLink;
	PxReal internalAngularErrorDegMax;
	PxU32 internalAngularErrorMaxLink;
	PxU32 firstInternalAngularBadFrame;
	PxU32 firstInternalAngularBadLink;
	PxReal loopAnchorErrorMax;
	PxU32 loopAnchorErrorMaxIndex;
	PxU32 firstLoopAnchorBadFrame;
	PxU32 firstLoopAnchorBadIndex;
	PxU32 phaseHeightCount;
	PxU32 phaseBaselineValid;
	PxReal phaseReferenceHeight;
	PxReal phaseBaselineSpread;
	PxReal phaseMaxRelativeDrift;
	PxU32 phaseRegressionFailed;
	PxU32 firstBadPhaseSample;
	PxU32 topSampleValid;
	PxReal topYInitial;
	PxReal topYLast;
	PxReal topYMin;
	PxReal topYMax;
	PxReal topYRange;
	PxReal baseYDriftMax;
	PxReal baseTiltDegMax;
	PxReal driveCoord;
	PxReal driveErrorMax;
	PxU32 firstDriveErrorFrame;
};

static PxReal finiteMetricOrZero(PxReal value)
{
	return PxIsFinite(value) ? value : 0.0f;
}

static void setGateFailure(ScissorGateEvaluation& evaluation,
	const char* reason)
{
	if(evaluation.exitCode != Snippets::eHEADLESS_PASS)
		return;
	evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
	evaluation.status = "FAIL";
	evaluation.reason = reason;
}

static void setGateError(ScissorGateEvaluation& evaluation,
	const char* reason)
{
	evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
	evaluation.status = "ERROR";
	evaluation.reason = reason;
	evaluation.oraclePassed = 0;
}

static ScissorGateEvaluation evaluateScissorGate(bool physicsReadable)
{
	ScissorGateEvaluation evaluation = {};
	evaluation.exitCode = Snippets::eHEADLESS_PASS;
	evaluation.status = "PASS";
	evaluation.reason = "none";
	evaluation.completedFrames = gFrame;
	evaluation.linkCount = physicsReadable && gArticulation
		? gArticulation->getNbLinks() : 0;
	evaluation.loopJointCount = gLoopJointCount;
	const bool statsFinite = areScissorStatsFinite();
	evaluation.nonFinite = (gScissorStats.nonFiniteFrame != PX_MAX_U32
		|| !statsFinite) ? 1u : 0u;
	evaluation.nonFiniteFrame = !statsFinite
		&& gScissorStats.nonFiniteFrame == PX_MAX_U32
		? gFrame : gScissorStats.nonFiniteFrame;
	evaluation.firstStallFrame = gScissorStats.firstStallFrame;
	evaluation.twistLimitViolationDegMax = finiteMetricOrZero(
		gScissorStats.twistLimitViolationDegMax);
	evaluation.twistLimitViolationMaxLink =
		gScissorStats.twistLimitViolationMaxLink;
	evaluation.internalAnchorErrorMax = finiteMetricOrZero(
		gScissorStats.internalAnchorErrorMax);
	evaluation.internalAnchorErrorMaxLink =
		gScissorStats.internalAnchorErrorMaxLink;
	evaluation.firstInternalAnchorBadFrame =
		gScissorStats.firstInternalAnchorBadFrame;
	evaluation.firstInternalAnchorBadLink =
		gScissorStats.firstInternalAnchorBadLink;
	evaluation.internalAngularErrorDegMax = finiteMetricOrZero(
		gScissorStats.internalAngularErrorDegMax);
	evaluation.internalAngularErrorMaxLink =
		gScissorStats.internalAngularErrorMaxLink;
	evaluation.firstInternalAngularBadFrame =
		gScissorStats.firstInternalAngularBadFrame;
	evaluation.firstInternalAngularBadLink =
		gScissorStats.firstInternalAngularBadLink;
	evaluation.loopAnchorErrorMax = finiteMetricOrZero(
		gScissorStats.loopAnchorErrorMax);
	evaluation.loopAnchorErrorMaxIndex = gScissorStats.loopAnchorErrorMaxIndex;
	evaluation.firstLoopAnchorBadFrame = gScissorStats.firstLoopAnchorBadFrame;
	evaluation.firstLoopAnchorBadIndex = gScissorStats.firstLoopAnchorBadIndex;
	evaluation.phaseHeightCount = gScissorStats.phaseHeightCount;
	evaluation.phaseBaselineValid = gScissorStats.phaseHeightCount >= 2 ? 1u : 0u;
	evaluation.phaseReferenceHeight = finiteMetricOrZero(
		gScissorStats.phaseReferenceHeight);
	evaluation.phaseBaselineSpread = evaluation.phaseBaselineValid
		? finiteMetricOrZero(gScissorStats.phaseBaselineSpread) : 0.0f;
	evaluation.phaseMaxRelativeDrift = finiteMetricOrZero(
		gScissorStats.phaseMaxRelativeDrift);
	evaluation.phaseRegressionFailed =
		gScissorStats.phaseRegressionFailed ? 1u : 0u;
	evaluation.firstBadPhaseSample = gScissorStats.firstBadPhaseSample;
	evaluation.topSampleValid =
		gScissorStats.firstReportedFrame != PX_MAX_U32 ? 1u : 0u;
	evaluation.topYInitial = finiteMetricOrZero(gScissorStats.topYInitial);
	evaluation.topYLast = finiteMetricOrZero(gScissorStats.topYLast);
	evaluation.topYMin = evaluation.topSampleValid
		? finiteMetricOrZero(gScissorStats.topYMin) : 0.0f;
	evaluation.topYMax = evaluation.topSampleValid
		? finiteMetricOrZero(gScissorStats.topYMax) : 0.0f;
	evaluation.topYRange = evaluation.topSampleValid
		? finiteMetricOrZero(evaluation.topYMax - evaluation.topYMin) : 0.0f;
	evaluation.baseYDriftMax = finiteMetricOrZero(gScissorStats.baseYDriftMax);
	evaluation.baseTiltDegMax = finiteMetricOrZero(gScissorStats.baseTiltDegMax);
	evaluation.driveCoord = finiteMetricOrZero(gScissorStats.driveCoord);
	evaluation.driveErrorMax = finiteMetricOrZero(gScissorStats.driveErrorMax);
	evaluation.firstDriveErrorFrame = gScissorStats.firstDriveErrorFrame;

	const bool finite = evaluation.nonFinite == 0;
	const bool moving = evaluation.firstStallFrame == PX_MAX_U32;
	const bool limitsHeld = evaluation.twistLimitViolationDegMax <= 5.0f;
	const bool anchorsHeld = evaluation.internalAnchorErrorMax <= 0.1f
		&& evaluation.internalAngularErrorDegMax <= 5.0f
		&& evaluation.loopAnchorErrorMax <= 0.1f;
	bool oraclePassed = finite && moving && limitsHeld && anchorsHeld
		&& !evaluation.phaseRegressionFailed;
	if(evaluation.completedFrames >= 3000
		&& (evaluation.phaseHeightCount < 5
			|| !evaluation.phaseBaselineValid
			|| evaluation.phaseBaselineSpread > 0.10f))
		oraclePassed = false;
	evaluation.oraclePassed = oraclePassed ? 1u : 0u;

	if(gInitializationFailed)
		setGateError(evaluation, "initialization");
	else if(gRuntimeInvariantFailed)
		setGateError(evaluation, "runtime_invariant");
	else if(!physicsReadable || gFetchPending || gSimulateFailures
		|| gFetchFailures)
		setGateError(evaluation, "lifecycle");
	else if(evaluation.completedFrames != gHeadlessOptions.frames)
		setGateError(evaluation, "incomplete_simulation");
	else if(gFetchErrorState)
		setGateFailure(evaluation, "fetch_error_state");
	else if(!finite)
		setGateFailure(evaluation, "non_finite");
	else if(!moving)
		setGateFailure(evaluation, "stall");
	else if(!limitsHeld)
		setGateFailure(evaluation, "twist_limit");
	else if(evaluation.internalAnchorErrorMax > 0.1f)
		setGateFailure(evaluation, "internal_anchor");
	else if(evaluation.internalAngularErrorDegMax > 5.0f)
		setGateFailure(evaluation, "revolute_swing");
	else if(evaluation.loopAnchorErrorMax > 0.1f)
		setGateFailure(evaluation, "loop_anchor");
	else if(evaluation.phaseRegressionFailed)
		setGateFailure(evaluation, "phase_regression");
	else if(evaluation.completedFrames >= 3000
		&& evaluation.phaseHeightCount < 5)
		setGateFailure(evaluation, "phase_sample_count");
	else if(evaluation.completedFrames >= 3000
		&& (!evaluation.phaseBaselineValid
			|| evaluation.phaseBaselineSpread > 0.10f))
		setGateFailure(evaluation, "phase_baseline_spread");

	return evaluation;
}

static void printScissorDiagnostics(const ScissorGateEvaluation& evaluation)
{
	if(evaluation.completedFrames < 3000)
	{
		printf("[ScissorRegression] ok=%u skipped=short-run frames=%u "
			"nonFiniteFrame=%u firstStallFrame=%u twistLimitViolationDegMax=%.3f "
			"internalAnchorMax=%.4f revoluteSwingDegMax=%.3f loopAnchorMax=%.4f\n",
			evaluation.oraclePassed, evaluation.completedFrames,
			evaluation.nonFiniteFrame, evaluation.firstStallFrame,
			double(evaluation.twistLimitViolationDegMax),
			double(evaluation.internalAnchorErrorMax),
			double(evaluation.internalAngularErrorDegMax),
			double(evaluation.loopAnchorErrorMax));
	}
	else
	{
		printf("[ScissorRegression] ok=%u samples=%u referenceHeight=%.4f "
			"baselineSpread=%.3f maxRelativeDrift=%.3f firstBadSample=%u "
			"nonFiniteFrame=%u firstStallFrame=%u twistLimitViolationDegMax=%.3f "
			"internalAnchorMax=%.4f revoluteSwingDegMax=%.3f loopAnchorMax=%.4f\n",
			evaluation.oraclePassed, evaluation.phaseHeightCount,
			double(evaluation.phaseReferenceHeight),
			double(evaluation.phaseBaselineSpread),
			double(evaluation.phaseMaxRelativeDrift),
			evaluation.firstBadPhaseSample, evaluation.nonFiniteFrame,
			evaluation.firstStallFrame,
			double(evaluation.twistLimitViolationDegMax),
			double(evaluation.internalAnchorErrorMax),
			double(evaluation.internalAngularErrorDegMax),
			double(evaluation.loopAnchorErrorMax));
	}

	printf("[ScissorDiag] frames=%u topY initial=%.4f last=%.4f min=%.4f max=%.4f "
		"range=%.4f baseDriftYMax=%.4f baseTiltDegMax=%.3f "
		"driveCoord=%.4f driveErrorMax=%.4f firstDriveErrorFrame=%u "
		"firstStallFrame=%u internalAnchorMax=%.4f(link=%u) "
		"firstInternalAnchorBadFrame=%u(link=%u) revoluteSwingDegMax=%.3f(link=%u) "
		"firstRevoluteSwingBadFrame=%u(link=%u) twistLimitViolationDegMax=%.3f(link=%u) "
		"loopAnchorMax=%.4f(loop=%u) firstLoopAnchorBadFrame=%u(loop=%u) "
		"nonFiniteFrame=%u\n",
		evaluation.completedFrames, double(evaluation.topYInitial),
		double(evaluation.topYLast), double(evaluation.topYMin),
		double(evaluation.topYMax), double(evaluation.topYRange),
		double(evaluation.baseYDriftMax), double(evaluation.baseTiltDegMax),
		double(evaluation.driveCoord), double(evaluation.driveErrorMax),
		evaluation.firstDriveErrorFrame, evaluation.firstStallFrame,
		double(evaluation.internalAnchorErrorMax),
		evaluation.internalAnchorErrorMaxLink,
		evaluation.firstInternalAnchorBadFrame,
		evaluation.firstInternalAnchorBadLink,
		double(evaluation.internalAngularErrorDegMax),
		evaluation.internalAngularErrorMaxLink,
		evaluation.firstInternalAngularBadFrame,
		evaluation.firstInternalAngularBadLink,
		double(evaluation.twistLimitViolationDegMax),
		evaluation.twistLimitViolationMaxLink,
		double(evaluation.loopAnchorErrorMax),
		evaluation.loopAnchorErrorMaxIndex,
		evaluation.firstLoopAnchorBadFrame,
		evaluation.firstLoopAnchorBadIndex, evaluation.nonFiniteFrame);
}

void cleanupPhysics(bool interactive)
{
	if(gFetchPending && !finishPendingSimulation())
	{
		gCleanupFailed = true;
		if(interactive)
			std::fprintf(stderr,
				"SnippetArticulationRC fetch remained pending during cleanup.\n");
		return;
	}

	if(interactive)
	{
		const ScissorGateEvaluation evaluation = evaluateScissorGate(true);
		printScissorDiagnostics(evaluation);
	}

	for(PxU32 i = 0; i < sizeof(gLoopJoints) / sizeof(gLoopJoints[0]); ++i)
		PX_RELEASE(gLoopJoints[i]);
	gLoopJointCount = 0;

	PX_RELEASE(gArticulation);
	gDriveJoint = NULL;
	gDriveLink = NULL;
	gBaseLink = NULL;
	gTopLink = NULL;
	PX_RELEASE(gScene);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	PX_RELEASE(gPvd);
	PX_RELEASE(gPvdTransport);
	PX_RELEASE(gFoundation);

	gCleanupFailed = hasOwnedPhysicsResources();
	if(interactive)
		printf("SnippetArticulation done.\n");
}

static void finalizeAfterCleanup(ScissorGateEvaluation& evaluation,
	PxU32 physicsErrors)
{
	if(gCleanupFailed)
	{
		if(evaluation.exitCode != Snippets::eHEADLESS_CONFIG_ERROR)
			setGateError(evaluation, "cleanup");
		else
			evaluation.oraclePassed = 0;
	}
	else if(physicsErrors)
		setGateFailure(evaluation, "physx_error");
}

static void printGateResult(const ScissorGateEvaluation& evaluation,
	PxU32 physicsErrors, PxU32 physicsWarnings)
{
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetArticulationRC case=%s solver=%s "
		"execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
		"dispatcherThreads=%u capability=SUPPORTED validation=ACCEPTED status=%s "
		"reason=%s nonFinite=%u physicsErrors=%u physicsWarnings=%u "
		"simulateFailures=%u fetchFailures=%u fetchPending=%u fetchErrorState=%u "
		"runtimeInvariantFailed=%u "
		"cleanupFailed=%u oraclePass=%u linkCount=%u loopJointCount=%u "
		"firstNonFiniteFrame=%u firstStallFrame=%u "
		"twistLimitViolationDegMax=%.9g internalAnchorMax=%.9g "
		"revoluteSwingDegMax=%.9g loopAnchorMax=%.9g "
		"phaseHeightCount=%u phaseBaselineValid=%u phaseReferenceHeight=%.9g "
		"phaseBaselineSpread=%.9g phaseMaxRelativeDrift=%.9g "
		"phaseRegressionFailed=%u firstBadPhaseSample=%u "
		"twistLimitCapDeg=5 internalAnchorCap=0.1 revoluteSwingCapDeg=5 "
		"loopAnchorCap=0.1 phaseMinSamples=5 phaseBaselineSpreadCap=0.1 "
		"phaseDriftCap=0.2 phaseDriftConsecutiveSamples=2 "
		"driveCoordDiagnostic=%.9g driveErrorMaxDiagnostic=%.9g "
		"topYInitialDiagnostic=%.9g topYLastDiagnostic=%.9g "
		"topYMinDiagnostic=%.9g topYMaxDiagnostic=%.9g "
		"topYRangeDiagnostic=%.9g baseYDriftMaxDiagnostic=%.9g "
		"baseTiltDegMaxDiagnostic=%.9g pvd=0\n",
		gHeadlessOptions.caseName.c_str(),
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, evaluation.completedFrames,
		double(gHeadlessOptions.dt), gHeadlessOptions.seed,
		gHeadlessOptions.dispatcherThreads, evaluation.status, evaluation.reason,
		evaluation.nonFinite, physicsErrors, physicsWarnings, gSimulateFailures,
		gFetchFailures, gFetchPending ? 1u : 0u, gFetchErrorState,
		gRuntimeInvariantFailed ? 1u : 0u,
		gCleanupFailed ? 1u : 0u, evaluation.oraclePassed,
		evaluation.linkCount, evaluation.loopJointCount,
		evaluation.nonFiniteFrame, evaluation.firstStallFrame,
		double(evaluation.twistLimitViolationDegMax),
		double(evaluation.internalAnchorErrorMax),
		double(evaluation.internalAngularErrorDegMax),
		double(evaluation.loopAnchorErrorMax), evaluation.phaseHeightCount,
		evaluation.phaseBaselineValid,
		double(evaluation.phaseReferenceHeight),
		double(evaluation.phaseBaselineSpread),
		double(evaluation.phaseMaxRelativeDrift),
		evaluation.phaseRegressionFailed, evaluation.firstBadPhaseSample,
		double(evaluation.driveCoord), double(evaluation.driveErrorMax),
		double(evaluation.topYInitial), double(evaluation.topYLast),
		double(evaluation.topYMin), double(evaluation.topYMax),
		double(evaluation.topYRange), double(evaluation.baseYDriftMax),
		double(evaluation.baseTiltDegMax));
}

static int reportConfigurationError(const Snippets::HeadlessOptions& options,
	const char* message)
{
	printf("[AVBD_GATE_ERROR] snippet=SnippetArticulationRC message=%s\n", message);
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetArticulationRC case=config-error "
		"solver=%s execution=%s requestedFrames=%u completedFrames=0 dt=%.9g "
		"seed=%u dispatcherThreads=%u capability=SUPPORTED validation=ACCEPTED "
		"status=ERROR reason=config nonFinite=0 physicsErrors=0 "
		"physicsWarnings=0 simulateFailures=0 fetchFailures=0 fetchPending=0 "
		"fetchErrorState=0 pvd=0\n",
		Snippets::getSolverTypeName(options.solverType),
		Snippets::getExecutionName(options.execution), options.frames,
		double(options.dt), options.seed, options.dispatcherThreads);
	return Snippets::eHEADLESS_CONFIG_ERROR;
}

int snippetMain(int argc, const char*const* argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);

	Snippets::HeadlessOptions defaults;
	defaults.caseName = "scissor-cycle";
	defaults.frames = 3600;
	defaults.seed = 1;
	defaults.dispatcherThreads = 2;
	defaults.dt = 1.0f / 60.0f;

	Snippets::HeadlessOptions options;
	std::string parseError;
	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options,
		parseError))
		return reportConfigurationError(options, parseError.c_str());

	bool headlessOnlyOptionSeen = false;
	for(int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if(!arg)
			continue;
		if(Snippets::isCommonHeadlessOption(arg))
		{
			if(std::strcmp(arg, "--headless") != 0
				&& !Snippets::hasOptionPrefix(arg, "--solver="))
				headlessOnlyOptionSeen = true;
			continue;
		}
		return reportConfigurationError(options, "unknown_argument");
	}

#ifndef RENDER_SNIPPET
	options.headless = true;
#endif

	if(!Snippets::equalsIgnoreCase(options.caseName.c_str(), "scissor-cycle"))
		return reportConfigurationError(options, "invalid_--case_value");
	options.caseName = "scissor-cycle";
	if(!options.headless && headlessOnlyOptionSeen)
		return reportConfigurationError(options, "gate_option_requires_--headless");
	if(options.headless && options.frames < 3600)
		return reportConfigurationError(options,
			"frames_must_be_at_least_3600");
	if(options.execution == Snippets::eHEADLESS_SEQUENTIAL
		&& options.solverType != PxSolverType::eAVBD)
		return reportConfigurationError(options, "sequential_requires_avbd");
	if(PxAbs(options.dt - (1.0f / 60.0f)) > 1.0e-7f)
		return reportConfigurationError(options, "dt_requires_60hz_calibration");
	if(!Snippets::applyExecutionEnvironment(options))
		return reportConfigurationError(options, "execution_environment_failed");

	gHeadlessOptions = options;
	gSolverType = options.solverType;

#ifdef RENDER_SNIPPET
	if(!options.headless)
	{
		extern void renderLoop();
		renderLoop();
		return 0;
	}
#endif

	Snippets::printHeadlessConfig("SnippetArticulationRC", gHeadlessOptions);
	initPhysics(false);
	if(!gInitializationFailed)
	{
		for(PxU32 i = 0; i < gHeadlessOptions.frames; ++i)
		{
			stepPhysics(false);
			if(gSimulateFailures || gFetchFailures
				|| gRuntimeInvariantFailed)
				break;
		}
	}

	if(gFetchPending)
		finishPendingSimulation();
	const bool physicsReadable = !gFetchPending;
	ScissorGateEvaluation evaluation = evaluateScissorGate(physicsReadable);
	printScissorDiagnostics(evaluation);
	cleanupPhysics(false);
	const PxU32 physicsErrors = gErrorCallback.getFatalCount();
	const PxU32 physicsWarnings = gErrorCallback.getWarningCount();
	finalizeAfterCleanup(evaluation, physicsErrors);
	printGateResult(evaluation, physicsErrors, physicsWarnings);
	return static_cast<int>(evaluation.exitCode);
}
