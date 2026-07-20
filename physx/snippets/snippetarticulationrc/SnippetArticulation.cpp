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

#include <ctype.h>
#include <cstdlib>
#include <cstring>
#include "PxPhysicsAPI.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"

using namespace physx;

static PxDefaultAllocator						gAllocator;
static PxDefaultErrorCallback					gErrorCallback;
static PxFoundation*							gFoundation		= NULL;
static PxPhysics*								gPhysics		= NULL;
static PxDefaultCpuDispatcher*					gDispatcher		= NULL;
static PxScene*									gScene			= NULL;
static PxMaterial*								gMaterial		= NULL;
static PxPvd*									gPvd			= NULL;
static PxArticulationReducedCoordinate*			gArticulation	= NULL;
static PxArticulationJointReducedCoordinate*	gDriveJoint		= NULL;
static PxSolverType::Enum						gSolverType		= PxSolverType::eAVBD;
static PxArticulationLink*						gBaseLink		= NULL;
static PxArticulationLink*						gTopLink		= NULL;
static PxArticulationLink*						gDriveLink		= NULL;
static PxD6Joint*								gLoopJoints[8]	= {};
static PxU32									gLoopJointCount	= 0;

static void trackLoopJoint(PxD6Joint* joint)
{
	PX_ASSERT(gLoopJointCount < sizeof(gLoopJoints) / sizeof(gLoopJoints[0]));
	if(gLoopJointCount < sizeof(gLoopJoints) / sizeof(gLoopJoints[0]))
		gLoopJoints[gLoopJointCount++] = joint;
}

static const char* getSolverTypeName(PxSolverType::Enum solverType)
{
	switch(solverType)
	{
	case PxSolverType::ePGS:	return "pgs";
	case PxSolverType::eTGS:	return "tgs";
	case PxSolverType::eAVBD:	return "avbd";
	default:					return "unknown";
	}
}

static bool tryParseSolverType(const char* value, PxSolverType::Enum& solverType)
{
	if(!value || !value[0])
		return false;
	if(_stricmp(value, "pgs") == 0)		{ solverType = PxSolverType::ePGS;  return true; }
	if(_stricmp(value, "tgs") == 0)		{ solverType = PxSolverType::eTGS;  return true; }
	if(_stricmp(value, "avbd") == 0)	{ solverType = PxSolverType::eAVBD; return true; }
	return false;
}

static PxSolverType::Enum getRequestedSolverType(int argc, const char*const* argv)
{
	for(int i = 1; i < argc; ++i)
	{
		if(!argv[i])
			continue;
		static const char prefix[] = "--solver=";
		if(std::strncmp(argv[i], prefix, sizeof(prefix) - 1) == 0)
		{
			PxSolverType::Enum solverType = PxSolverType::eAVBD;
			if(tryParseSolverType(argv[i] + sizeof(prefix) - 1, solverType))
				return solverType;
		}
	}
	const char* value = std::getenv("PHYSX_SNIPPET_SOLVER");
	PxSolverType::Enum solverType = PxSolverType::eAVBD;
	if(tryParseSolverType(value, solverType))
		return solverType;
	return PxSolverType::eAVBD;
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
	PxArticulationLink* base = gArticulation->createLink(NULL, PxTransform(PxVec3(0.f, 0.25f, 0.f)));
	PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.5f, 0.25f, 1.5f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*base, 3.f);
	gBaseLink = base;

	//Now create the slider and fixed joints...

	gArticulation->setSolverIterationCounts(10);

	PxArticulationLink* leftRoot = gArticulation->createLink(base, PxTransform(PxVec3(0.f, 0.55f, -0.9f)));
	PxRigidActorExt::createExclusiveShape(*leftRoot, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*leftRoot, 1.f);

	PxArticulationLink* rightRoot = gArticulation->createLink(base, PxTransform(PxVec3(0.f, 0.55f, 0.9f)));
	PxRigidActorExt::createExclusiveShape(*rightRoot, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*rightRoot, 1.f);
	gDriveLink = rightRoot;

	PxArticulationJointReducedCoordinate* joint = leftRoot->getInboundJoint();
	joint->setJointType(PxArticulationJointType::eFIX);
	joint->setParentPose(PxTransform(PxVec3(0.f, 0.25f, -0.9f)));
	joint->setChildPose(PxTransform(PxVec3(0.f, -0.05f, 0.f)));

	//Set up the drive joint...	
	gDriveJoint = rightRoot->getInboundJoint();
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
		PxArticulationLink* leftLink = gArticulation->createLink(currLeft, PxTransform(pos + PxVec3(0.f, sinAng*(2 * i + 1), 0.f), leftRot));
		PxRigidActorExt::createExclusiveShape(*leftLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *gMaterial);
		PxRigidBodyExt::updateMassAndInertia(*leftLink, 1.f);

		const PxVec3 leftAnchorLocation = pos + PxVec3(0.f, sinAng*(2 * i), -0.9f);

		joint = leftLink->getInboundJoint();
		joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchorLocation), leftParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, -1.f), rightRot));
		joint->setJointType(PxArticulationJointType::eREVOLUTE);

		leftParentRot = leftRot;

		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));


		PxArticulationLink* rightLink = gArticulation->createLink(currRight, PxTransform(pos + PxVec3(0.f, sinAng*(2 * i + 1), 0.f), rightRot));
		PxRigidActorExt::createExclusiveShape(*rightLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *gMaterial);
		PxRigidBodyExt::updateMassAndInertia(*rightLink, 1.f);

		const PxVec3 rightAnchorLocation = pos + PxVec3(0.f, sinAng*(2 * i), 0.9f);

		joint = rightLink->getInboundJoint();
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchorLocation), rightParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, 1.f), leftRot));
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));

		rightParentRot = rightRot;

		PxD6Joint* d6joint = PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity), rightLink, PxTransform(PxIdentity));
		trackLoopJoint(d6joint);

		d6joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
		d6joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
		d6joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

		currLeft = rightLink;
		currRight = leftLink;
	}

	
	PxArticulationLink* leftTop = gArticulation->createLink(currLeft, currLeft->getGlobalPose().transform(PxTransform(PxVec3(-0.5f, 0.f, -1.0f), leftParentRot)));
	PxRigidActorExt::createExclusiveShape(*leftTop, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*leftTop, 1.f);

	PxArticulationLink* rightTop = gArticulation->createLink(currRight, currRight->getGlobalPose().transform(PxTransform(PxVec3(-0.5f, 0.f, 1.0f), rightParentRot)));
	PxRigidActorExt::createExclusiveShape(*rightTop, PxCapsuleGeometry(0.05f, 0.8f), *gMaterial);
	//PxRigidActorExt::createExclusiveShape(*rightTop, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*rightTop, 1.f);

	joint = leftTop->getInboundJoint();
	joint->setParentPose(PxTransform(PxVec3(0.f, 0.f, -1.f), currLeft->getGlobalPose().q.getConjugate()));
	joint->setChildPose(PxTransform(PxVec3(0.5f, 0.f, 0.f), leftTop->getGlobalPose().q.getConjugate()));
	joint->setJointType(PxArticulationJointType::eREVOLUTE);
	joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);

	joint = rightTop->getInboundJoint();
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
		PxArticulationLink* leftLink = gArticulation->createLink(currLeft, PxTransform(pos + PxVec3(0.f, sinAng*(2 * i + 1), 0.f), leftRot));
		PxRigidActorExt::createExclusiveShape(*leftLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *gMaterial);
		PxRigidBodyExt::updateMassAndInertia(*leftLink, 1.f);

		const PxVec3 leftAnchorLocation = pos + PxVec3(0.f, sinAng*(2 * i), -0.9f);

		joint = leftLink->getInboundJoint();
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchorLocation), leftParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, -1.f), rightRot));

		leftParentRot = leftRot;

		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));

		PxArticulationLink* rightLink = gArticulation->createLink(currRight, PxTransform(pos + PxVec3(0.f, sinAng*(2 * i + 1), 0.f), rightRot));
		PxRigidActorExt::createExclusiveShape(*rightLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *gMaterial);
		PxRigidBodyExt::updateMassAndInertia(*rightLink, 1.f);

		const PxVec3 rightAnchorLocation = pos + PxVec3(0.f, sinAng*(2 * i), 0.9f);

		/*joint = PxD6JointCreate(getPhysics(), currRight, PxTransform(currRight->getGlobalPose().transformInv(rightAnchorLocation)),
		rightLink, PxTransform(PxVec3(0.f, 0.f, 1.f)));*/

		joint = rightLink->getInboundJoint();
		joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchorLocation), rightParentRot));
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, 1.f), leftRot));
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));

		rightParentRot = rightRot;

		PxD6Joint* d6joint = PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity), rightLink, PxTransform(PxIdentity));
		trackLoopJoint(d6joint);

		d6joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
		d6joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
		d6joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

		currLeft = rightLink;
		currRight = leftLink;
	}

	PxD6Joint* d6joint = PxD6JointCreate(*gPhysics, currLeft, PxTransform(PxVec3(0.f, 0.f, -1.f)), leftTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
	trackLoopJoint(d6joint);

	d6joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
	d6joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
	d6joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

	d6joint = PxD6JointCreate(*gPhysics, currRight, PxTransform(PxVec3(0.f, 0.f, 1.f)), rightTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
	trackLoopJoint(d6joint);

	d6joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
	d6joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
	d6joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);


	const PxTransform topPose(PxVec3(0.f, leftTop->getGlobalPose().p.y + 0.15f, 0.f));

	PxArticulationLink* top = gArticulation->createLink(leftTop, topPose);
	PxRigidActorExt::createExclusiveShape(*top, PxBoxGeometry(0.5f, 0.1f, 1.5f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*top, 1.f);
	gTopLink = top;

	joint = top->getInboundJoint();
	joint->setJointType(PxArticulationJointType::eFIX);
	joint->setParentPose(PxTransform(PxVec3(0.f, 0.0f, 0.f)));
	joint->setChildPose(PxTransform(PxVec3(0.f, -0.15f, -0.9f)));

	gScene->addArticulation(*gArticulation);

	for (PxU32 i = 0; i < gArticulation->getNbLinks(); ++i)
	{
		PxArticulationLink* link;
		gArticulation->getLinks(&link, 1, i);

		link->setLinearDamping(0.2f);
		link->setAngularDamping(0.2f);

		link->setMaxAngularVelocity(20.f);
		link->setMaxLinearVelocity(100.f);

		if (link != top)
		{
			for (PxU32 b = 0; b < link->getNbShapes(); ++b)
			{
				PxShape* shape;
				link->getShapes(&shape, 1, b);

				shape->setSimulationFilterData(PxFilterData(0, 0, 1, 0));
			}
		}
	}

	const PxVec3 halfExt(0.25f);
	const PxReal density(0.5f);

	PxRigidDynamic* box0 = gPhysics->createRigidDynamic(PxTransform(PxVec3(-0.25f, 5.f, 0.5f)));
	PxShape* shape0 = PxRigidActorExt::createExclusiveShape(*box0, PxBoxGeometry(halfExt), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*box0, density);
	gScene->addActor(*box0);

	PxRigidDynamic* box1 = gPhysics->createRigidDynamic(PxTransform(PxVec3(0.25f, 5.f, 0.5f)));
	PxShape* shape1 = PxRigidActorExt::createExclusiveShape(*box1, PxBoxGeometry(halfExt), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*box1, density);
	gScene->addActor(*box1);

	PxRigidDynamic* box2 = gPhysics->createRigidDynamic(PxTransform(PxVec3(-0.25f, 4.5f, 0.5f)));
	PxShape* shape2 = PxRigidActorExt::createExclusiveShape(*box2, PxBoxGeometry(halfExt), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*box2, density);
	gScene->addActor(*box2);

	PxRigidDynamic* box3 = gPhysics->createRigidDynamic(PxTransform(PxVec3(0.25f, 4.5f, 0.5f)));
	PxShape* shape3 = PxRigidActorExt::createExclusiveShape(*box3, PxBoxGeometry(halfExt), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*box3, density);
	gScene->addActor(*box3);

	PxRigidDynamic* box4 = gPhysics->createRigidDynamic(PxTransform(PxVec3(-0.25f, 5.f, 0.f)));
	PxShape* shape4 = PxRigidActorExt::createExclusiveShape(*box4, PxBoxGeometry(halfExt), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*box4, density);
	gScene->addActor(*box4);

	PxRigidDynamic* box5 = gPhysics->createRigidDynamic(PxTransform(PxVec3(0.25f, 5.f, 0.f)));
	PxShape* shape5 = PxRigidActorExt::createExclusiveShape(*box5, PxBoxGeometry(halfExt), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*box5, density);
	gScene->addActor(*box5);

	PxRigidDynamic* box6 = gPhysics->createRigidDynamic(PxTransform(PxVec3(-0.25f, 4.5f, 0.f)));
	PxShape* shape6 = PxRigidActorExt::createExclusiveShape(*box6, PxBoxGeometry(halfExt), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*box6, density);
	gScene->addActor(*box6);

	PxRigidDynamic* box7 = gPhysics->createRigidDynamic(PxTransform(PxVec3(0.25f, 4.5f, 0.f)));
	PxShape* shape7 = PxRigidActorExt::createExclusiveShape(*box7, PxBoxGeometry(halfExt), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*box7, density);
	gScene->addActor(*box7);

	const float contactOffset = 0.2f;
	shape0->setContactOffset(contactOffset);
	shape1->setContactOffset(contactOffset);
	shape2->setContactOffset(contactOffset);
	shape3->setContactOffset(contactOffset);
	shape4->setContactOffset(contactOffset);
	shape5->setContactOffset(contactOffset);
	shape6->setContactOffset(contactOffset);
	shape7->setContactOffset(contactOffset);
}

void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPvd = PxCreatePvd(*gFoundation);
	PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
	gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	PxInitExtensions(*gPhysics, gPvd);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	
	PxU32 numCores = SnippetUtils::getNbPhysicalCores();
	gDispatcher = PxDefaultCpuDispatcherCreate(numCores == 0 ? 0 : numCores - 1);
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;

	sceneDesc.solverType = gSolverType;
	sceneDesc.filterShader = scissorFilter;

	gScene = gPhysics->createScene(sceneDesc);

	printf("[SnippetArticulationRCConfig] solver=%s\n",
		getSolverTypeName(sceneDesc.solverType));
	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.f);

	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	gScene->addActor(*groundPlane);

	// TODO(AVBD): PhysX 5 removed the old solver-neutral PxArticulation layer,
	// so AVBD currently keeps the reduced-coordinate factory name even when the
	// articulation/joint handling behind it is maximal-coordinate oriented.
	gArticulation = gPhysics->createArticulationReducedCoordinate();

	createScissorLift();
}

static bool gClosing = true;
static PxU32 gFrame = 0;

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
static bool gScissorRegressionOK = true;

static bool isArticulationStateFinite()
{
	if(!gArticulation || !PxIsFinite(gScissorStats.driveCoord)
		|| !PxIsFinite(gScissorStats.driveCoordVelocity))
		return false;

	const PxU32 linkCount = gArticulation->getNbLinks();
	for(PxU32 i = 0; i < linkCount; ++i)
	{
		PxArticulationLink* link = NULL;
		gArticulation->getLinks(&link, 1, i);
		if(!link)
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

static void updateAnchorErrors(PxU32 frame)
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
		gArticulation->getLinks(&child, 1, i);
		PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
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
		PxRigidActor* actor0 = NULL;
		PxRigidActor* actor1 = NULL;
		gLoopJoints[i]->getActors(actor0, actor1);
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
}

static void dumpScissorState(PxU32 frame)
{
	if(!gBaseLink || !gTopLink)
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

void stepPhysics(bool /*interactive*/)
{
	const PxReal dt = 1.0f / 60.f;
	PxReal driveValue = gDriveJoint->getDriveTarget(PxArticulationAxis::eZ);

	if (gClosing && driveValue < -1.2f)
		gClosing = false;
	else if (!gClosing && driveValue > 0.f)
		gClosing = true;

	if (gClosing)
		driveValue -= dt*0.25f;
	else
		driveValue += dt*0.25f;
	gDriveJoint->setDriveTarget(PxArticulationAxis::eZ, driveValue);

	gScene->simulate(dt);
	gScene->fetchResults(true);

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
		updateAnchorErrors(gFrame);
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

	const bool snapshotFrame = (gFrame % 600) == 0 || gFrame < 5;
	if(snapshotFrame)
		dumpScissorState(gFrame);
	++gFrame;
}

static bool evaluateScissorRegression()
{
	const bool finite = gScissorStats.nonFiniteFrame == PX_MAX_U32;
	const bool moving = gScissorStats.firstStallFrame == PX_MAX_U32;
	const bool limitsHeld = gScissorStats.twistLimitViolationDegMax <= 5.0f;
	const bool anchorsHeld = gScissorStats.internalAnchorErrorMax <= 0.1f
		&& gScissorStats.internalAngularErrorDegMax <= 5.0f
		&& gScissorStats.loopAnchorErrorMax <= 0.1f;
	bool ok = finite && moving && limitsHeld && anchorsHeld
		&& !gScissorStats.phaseRegressionFailed;

	if(gFrame < 3000)
	{
		printf("[ScissorRegression] ok=%d skipped=short-run frames=%u "
			"nonFiniteFrame=%u firstStallFrame=%u twistLimitViolationDegMax=%.3f "
			"internalAnchorMax=%.4f revoluteSwingDegMax=%.3f loopAnchorMax=%.4f\n",
			ok ? 1 : 0, gFrame, gScissorStats.nonFiniteFrame,
			gScissorStats.firstStallFrame, gScissorStats.twistLimitViolationDegMax,
			gScissorStats.internalAnchorErrorMax,
			gScissorStats.internalAngularErrorDegMax,
			gScissorStats.loopAnchorErrorMax);
		return ok;
	}

	const PxU32 sampleCount = gScissorStats.phaseHeightCount;
	if(sampleCount < 5 || gScissorStats.phaseBaselineSpread > 0.10f)
		ok = false;

	printf("[ScissorRegression] ok=%d samples=%u referenceHeight=%.4f "
		"baselineSpread=%.3f maxRelativeDrift=%.3f firstBadSample=%u "
		"nonFiniteFrame=%u firstStallFrame=%u twistLimitViolationDegMax=%.3f "
		"internalAnchorMax=%.4f revoluteSwingDegMax=%.3f loopAnchorMax=%.4f\n",
		ok ? 1 : 0, sampleCount, gScissorStats.phaseReferenceHeight,
		gScissorStats.phaseBaselineSpread, gScissorStats.phaseMaxRelativeDrift,
		gScissorStats.firstBadPhaseSample, gScissorStats.nonFiniteFrame,
		gScissorStats.firstStallFrame, gScissorStats.twistLimitViolationDegMax,
		gScissorStats.internalAnchorErrorMax,
		gScissorStats.internalAngularErrorDegMax,
		gScissorStats.loopAnchorErrorMax);
	return ok;
}
	
void cleanupPhysics(bool /*interactive*/)
{
	gScissorRegressionOK = evaluateScissorRegression();
	printf("[ScissorDiag] frames=%u topY initial=%.4f last=%.4f min=%.4f max=%.4f "
		"range=%.4f baseDriftYMax=%.4f baseTiltDegMax=%.3f "
		"driveCoord=%.4f driveErrorMax=%.4f firstDriveErrorFrame=%u "
		"firstStallFrame=%u internalAnchorMax=%.4f(link=%u) "
		"firstInternalAnchorBadFrame=%u(link=%u) revoluteSwingDegMax=%.3f(link=%u) "
		"firstRevoluteSwingBadFrame=%u(link=%u) twistLimitViolationDegMax=%.3f(link=%u) "
		"loopAnchorMax=%.4f(loop=%u) "
		"firstLoopAnchorBadFrame=%u(loop=%u) nonFiniteFrame=%u\n",
		gFrame, gScissorStats.topYInitial, gScissorStats.topYLast,
		gScissorStats.topYMin == PX_MAX_F32 ? 0.0f : gScissorStats.topYMin,
		gScissorStats.topYMax == -PX_MAX_F32 ? 0.0f : gScissorStats.topYMax,
		(gScissorStats.topYMax > -PX_MAX_F32 && gScissorStats.topYMin < PX_MAX_F32)
			? (gScissorStats.topYMax - gScissorStats.topYMin) : 0.0f,
		gScissorStats.baseYDriftMax, gScissorStats.baseTiltDegMax,
		gScissorStats.driveCoord, gScissorStats.driveErrorMax,
		gScissorStats.firstDriveErrorFrame, gScissorStats.firstStallFrame,
		gScissorStats.internalAnchorErrorMax, gScissorStats.internalAnchorErrorMaxLink,
		gScissorStats.firstInternalAnchorBadFrame, gScissorStats.firstInternalAnchorBadLink,
		gScissorStats.internalAngularErrorDegMax, gScissorStats.internalAngularErrorMaxLink,
		gScissorStats.firstInternalAngularBadFrame, gScissorStats.firstInternalAngularBadLink,
		gScissorStats.twistLimitViolationDegMax, gScissorStats.twistLimitViolationMaxLink,
		gScissorStats.loopAnchorErrorMax, gScissorStats.loopAnchorErrorMaxIndex,
		gScissorStats.firstLoopAnchorBadFrame, gScissorStats.firstLoopAnchorBadIndex,
		gScissorStats.nonFiniteFrame);

	gArticulation->release();
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	PxPvdTransport* transport = gPvd->getTransport();
	PX_RELEASE(gPvd);
	PX_RELEASE(transport);
	PxCloseExtensions();  
	PX_RELEASE(gFoundation);

	printf("SnippetArticulation done.\n");
}

static bool hasHeadlessArg(int argc, const char*const* argv)
{
	for(PxI32 i = 1; i < argc; ++i)
	{
		if(!argv[i])
			continue;
		if(std::strcmp(argv[i], "--headless") == 0)
			return true;
	}
	return false;
}

static bool isHeadlessRequested(int argc, const char*const* argv)
{
	if(hasHeadlessArg(argc, argv))
		return true;

	const char* value = std::getenv("PHYSX_SNIPPET_HEADLESS");
	return value && value[0] && value[0] != '0';
}

int snippetMain(int argc, const char*const* argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	gSolverType = getRequestedSolverType(argc, argv);
#ifdef RENDER_SNIPPET
	if(!isHeadlessRequested(argc, argv))
	{
		extern void renderLoop();
		renderLoop();
		return 0;
	}
#endif

	// Cover enough complete cycles to catch branch changes caused by an
	// incorrectly mapped asymmetric articulation limit.
	PxU32 frameCount = 3600;
	if(const char* override = std::getenv("PHYSX_SNIPPET_FRAME_COUNT"))
	{
		const long value = std::strtol(override, nullptr, 10);
		if(value > 0 && value < 1000000)
			frameCount = static_cast<PxU32>(value);
	}
	initPhysics(false);
	for(PxU32 i=0; i<frameCount; i++)
		stepPhysics(false);
	cleanupPhysics(false);

	return gScissorRegressionOK ? 0 : 1;
}
