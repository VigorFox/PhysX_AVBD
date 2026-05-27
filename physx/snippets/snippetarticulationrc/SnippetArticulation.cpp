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

		d6joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
		d6joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
		d6joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

		currLeft = rightLink;
		currRight = leftLink;
	}

	PxD6Joint* d6joint = PxD6JointCreate(*gPhysics, currLeft, PxTransform(PxVec3(0.f, 0.f, -1.f)), leftTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));

	d6joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
	d6joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
	d6joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

	d6joint = PxD6JointCreate(*gPhysics, currRight, PxTransform(PxVec3(0.f, 0.f, 1.f)), rightTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));

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
};

static ScissorStats gScissorStats;

static bool isFiniteVec(const PxVec3& v)
{
	return PxIsFinite(v.x) && PxIsFinite(v.y) && PxIsFinite(v.z);
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
		"driveTarget=%.4f jointPos=%.4f jointVel=%.4f\n",
		frame, basePose.p.x, basePose.p.y, basePose.p.z,
		topPose.p.x, topPose.p.y, topPose.p.z,
		driveTarget, jointPos, jointVel);
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

	if(gBaseLink && gTopLink)
	{
		const PxTransform basePose = gBaseLink->getGlobalPose();
		const PxTransform topPose  = gTopLink->getGlobalPose();
		if(!isFiniteVec(basePose.p) || !isFiniteVec(topPose.p))
		{
			if(gScissorStats.nonFiniteFrame == PX_MAX_U32)
				gScissorStats.nonFiniteFrame = gFrame;
		}
		else
		{
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
			const PxVec3 baseUp = basePose.q.rotate(PxVec3(0.f, 1.f, 0.f));
			const PxReal dotUp = PxClamp(baseUp.y, -1.0f, 1.0f);
			const PxReal tiltDeg = PxAcos(dotUp) * 180.0f / PxPi;
			gScissorStats.baseTiltDegMax = PxMax(gScissorStats.baseTiltDegMax, tiltDeg);
		}
	}

	const bool snapshotFrame = (gFrame % 60) == 0 || gFrame < 30;
	if(snapshotFrame)
		dumpScissorState(gFrame);
	++gFrame;
}
	
void cleanupPhysics(bool /*interactive*/)
{
	printf("[ScissorDiag] frames=%u topY initial=%.4f last=%.4f min=%.4f max=%.4f "
		"range=%.4f baseDriftYMax=%.4f baseTiltDegMax=%.3f nonFiniteFrame=%u\n",
		gFrame, gScissorStats.topYInitial, gScissorStats.topYLast,
		gScissorStats.topYMin == PX_MAX_F32 ? 0.0f : gScissorStats.topYMin,
		gScissorStats.topYMax == -PX_MAX_F32 ? 0.0f : gScissorStats.topYMax,
		(gScissorStats.topYMax > -PX_MAX_F32 && gScissorStats.topYMin < PX_MAX_F32)
			? (gScissorStats.topYMax - gScissorStats.topYMin) : 0.0f,
		gScissorStats.baseYDriftMax, gScissorStats.baseTiltDegMax,
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

	// Cover at least one full open/close cycle of the drive joint (~10s) so
	// regressions in the AVBD articulation path show up in the diag rather
	// than being masked by a short 1.5s smoke window.
	PxU32 frameCount = 900;
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

	return 0;
}
