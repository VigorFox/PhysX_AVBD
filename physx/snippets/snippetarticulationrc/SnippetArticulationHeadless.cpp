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

// Distilled SnippetArticulationRC headless gates. Scene contract from
// PhysX_AVBD_2 scissor-cycle; not a copy of that 1600-line demo.
// Visual path stays in SnippetArticulation.cpp.

#include "SnippetArticulationHeadless.h"

#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace physx;

namespace
{

enum ArticHeadlessCase
{
	eARTIC_CASE_SCISSOR_UNLOADED,
	eARTIC_CASE_SCISSOR_CYCLE
};

static const PxU32 gExpectedLinkCount = 18;
static const PxU32 gExpectedLoopCount = 8;
static const PxU32 gExpectedCargoCount = 8;
static const PxReal gTwistLimitCapDeg = 5.0f;
static const PxReal gInternalAnchorCap = 0.1f;
static const PxReal gRevoluteSwingCapDeg = 5.0f;
static const PxReal gLoopAnchorCap = 0.1f;
static const PxReal gMaxAbsPositionCap = 50.0f;
static const PxReal gMaxLinearSpeedCap = 80.0f;
static const PxReal gMaxAngularSpeedCap = 120.0f;
static const PxReal gStrokeMin = 0.25f;
static const PxReal gCargoMinYCap = -0.1f;
static const PxReal gTopTiltDegCap = 25.0f;
static const PxReal gCargoMinTopClearanceCap = 0.15f;
static const PxReal gCargoPlanarOffsetCap = 1.2f;
static const PxReal gCargoLinearSpeedCap = 8.0f;

static Snippets::HeadlessOptions gOptions;
static ArticHeadlessCase gCase = eARTIC_CASE_SCISSOR_CYCLE;
static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation* gFoundation = NULL;
static PxPhysics* gPhysics = NULL;
static PxDefaultCpuDispatcher* gDispatcher = NULL;
static PxScene* gScene = NULL;
static PxMaterial* gMaterial = NULL;
static PxArticulationReducedCoordinate* gArticulation = NULL;
static PxArticulationJointReducedCoordinate* gDriveJoint = NULL;
static PxArticulationLink* gBaseLink = NULL;
static PxArticulationLink* gDriveLink = NULL;
static PxArticulationLink* gTopLink = NULL;
static std::vector<PxD6Joint*> gLoopJoints;
static std::vector<PxRigidDynamic*> gCargo;
static bool gExtensionsInitialized = false;
static bool gInitializationFailed = false;
static bool gClosing = true;

static PxU32 gCompletedFrames = 0;
static PxU32 gFetchFailures = 0;
static PxU32 gFetchErrorState = 0;
static PxU32 gNonFinite = 0;
static PxU32 gFirstFailFrame = PX_MAX_U32;
static const char* gFirstFailReason = "none";
static PxU32 gFirstStallFrame = PX_MAX_U32;
static PxU32 gStallWindowFrame = PX_MAX_U32;
static PxReal gStallWindowCoord = 0.0f;
static PxReal gStallWindowTarget = 0.0f;
static PxReal gDriveCoord = 0.0f;
static PxReal gDriveErrorMax = 0.0f;
static PxReal gMaxAbsPosition = 0.0f;
static PxReal gMaxLinearSpeed = 0.0f;
static PxReal gMaxAngularSpeed = 0.0f;
static PxReal gInternalAnchorMax = 0.0f;
static PxReal gRevoluteSwingDegMax = 0.0f;
static PxReal gLockedAngularDegMax = 0.0f;
static PxReal gTwistLimitViolationDegMax = 0.0f;
static PxReal gLoopAnchorMax = 0.0f;
static PxReal gBaseYMin = PX_MAX_F32;
static PxReal gBaseYMax = -PX_MAX_F32;
static PxReal gBaseTiltDegMax = 0.0f;
static PxReal gTopYMin = PX_MAX_F32;
static PxReal gTopYMax = -PX_MAX_F32;
static PxReal gTopTiltDegMax = 0.0f;
static PxReal gCargoMinY = PX_MAX_F32;
static PxReal gCargoCurrentMinY = PX_MAX_F32;
static PxReal gCargoCurrentMinTopClearance = PX_MAX_F32;
static PxReal gCargoCurrentMaxPlanarOffset = 0.0f;
static PxReal gCargoMaxAbsPosition = 0.0f;
static PxReal gCargoMaxLinearSpeed = 0.0f;

static PxFilterFlags scissorFilter(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
	PxFilterObjectAttributes attributes1, PxFilterData filterData1,
	PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(constantBlock);
	PX_UNUSED(constantBlockSize);
	if(filterData0.word2 != 0 && filterData0.word2 == filterData1.word2)
		return PxFilterFlag::eKILL;
	pairFlags |= PxPairFlag::eCONTACT_DEFAULT;
	return PxFilterFlag::eDEFAULT;
}

static bool tryParseHeadlessCase(const char* value, ArticHeadlessCase& headlessCase)
{
	if(Snippets::equalsIgnoreCase(value, "scissor-unloaded"))
	{
		headlessCase = eARTIC_CASE_SCISSOR_UNLOADED;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "scissor-cycle") ||
		Snippets::equalsIgnoreCase(value, "default"))
	{
		headlessCase = eARTIC_CASE_SCISSOR_CYCLE;
		return true;
	}
	return false;
}

static const char* getHeadlessCaseName(ArticHeadlessCase headlessCase)
{
	return headlessCase == eARTIC_CASE_SCISSOR_UNLOADED ? "scissor-unloaded" : "scissor-cycle";
}

static bool isFiniteVec(const PxVec3& v)
{
	return PxIsFinite(v.x) && PxIsFinite(v.y) && PxIsFinite(v.z);
}

static bool isFiniteQuat(const PxQuat& q)
{
	return PxIsFinite(q.x) && PxIsFinite(q.y) && PxIsFinite(q.z) && PxIsFinite(q.w);
}

static bool isFinitePose(const PxTransform& pose)
{
	return isFiniteVec(pose.p) && isFiniteQuat(pose.q);
}

static void noteFirstFail(const char* reason)
{
	if(gFirstFailFrame != PX_MAX_U32)
		return;
	gFirstFailFrame = gCompletedFrames + 1;
	gFirstFailReason = reason;
	printf("[ScissorFirstFail] frame=%u reason=%s driveCoord=%.4f "
		"internalAnchor=%.4f loopAnchor=%.4f twistDeg=%.3f swingDeg=%.3f "
		"maxAbsPosition=%.3f maxLin=%.3f maxAng=%.3f cargoMinY=%.3f\n",
		gFirstFailFrame, reason, double(gDriveCoord), double(gInternalAnchorMax),
		double(gLoopAnchorMax), double(gTwistLimitViolationDegMax),
		double(gRevoluteSwingDegMax), double(gMaxAbsPosition),
		double(gMaxLinearSpeed), double(gMaxAngularSpeed), double(gCargoMinY));
}

static void addLoopJoint(PxD6Joint* joint)
{
	if(!joint)
	{
		gInitializationFailed = true;
		return;
	}
	joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
	joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
	joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
	gLoopJoints.push_back(joint);
}

static void createCargoBoxes()
{
	const PxVec3 halfExt(0.25f);
	const PxReal density = 0.5f;
	const PxReal contactOffset = 0.2f;
	const PxVec3 positions[gExpectedCargoCount] = {
		PxVec3(-0.25f, 5.0f, 0.5f),
		PxVec3(0.25f, 5.0f, 0.5f),
		PxVec3(-0.25f, 4.5f, 0.5f),
		PxVec3(0.25f, 4.5f, 0.5f),
		PxVec3(-0.25f, 5.0f, 0.0f),
		PxVec3(0.25f, 5.0f, 0.0f),
		PxVec3(-0.25f, 4.5f, 0.0f),
		PxVec3(0.25f, 4.5f, 0.0f)
	};
	for(PxU32 i = 0; i < gExpectedCargoCount; ++i)
	{
		PxRigidDynamic* box = gPhysics->createRigidDynamic(PxTransform(positions[i]));
		PxShape* shape = PxRigidActorExt::createExclusiveShape(*box, PxBoxGeometry(halfExt), *gMaterial);
		if(!box || !shape || !PxRigidBodyExt::updateMassAndInertia(*box, density))
		{
			gInitializationFailed = true;
			return;
		}
		shape->setContactOffset(contactOffset);
		gScene->addActor(*box);
		gCargo.push_back(box);
	}
}

static void createScissorLift(bool loadCargo)
{
	const PxReal runnerLength = 2.0f;
	const PxReal placementDistance = 1.8f;
	const PxReal angle = PxAcos(placementDistance / runnerLength);
	const PxReal sinAng = PxSin(angle);
	const PxQuat leftRot(-angle, PxVec3(1.0f, 0.0f, 0.0f));
	const PxQuat rightRot(angle, PxVec3(1.0f, 0.0f, 0.0f));

	gBaseLink = gArticulation->createLink(NULL, PxTransform(PxVec3(0.0f, 0.25f, 0.0f)));
	PxRigidActorExt::createExclusiveShape(*gBaseLink, PxBoxGeometry(0.5f, 0.25f, 1.5f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*gBaseLink, 3.0f);
	gArticulation->setSolverIterationCounts(32);

	PxArticulationLink* leftRoot = gArticulation->createLink(gBaseLink, PxTransform(PxVec3(0.0f, 0.55f, -0.9f)));
	PxRigidActorExt::createExclusiveShape(*leftRoot, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*leftRoot, 1.0f);
	gDriveLink = gArticulation->createLink(gBaseLink, PxTransform(PxVec3(0.0f, 0.55f, 0.9f)));
	PxRigidActorExt::createExclusiveShape(*gDriveLink, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*gDriveLink, 1.0f);

	PxArticulationJointReducedCoordinate* joint = leftRoot->getInboundJoint();
	joint->setJointType(PxArticulationJointType::eFIX);
	joint->setParentPose(PxTransform(PxVec3(0.0f, 0.25f, -0.9f)));
	joint->setChildPose(PxTransform(PxVec3(0.0f, -0.05f, 0.0f)));

	gDriveJoint = gDriveLink->getInboundJoint();
	gDriveJoint->setJointType(PxArticulationJointType::ePRISMATIC);
	gDriveJoint->setMotion(PxArticulationAxis::eZ, PxArticulationMotion::eLIMITED);
	gDriveJoint->setLimitParams(PxArticulationAxis::eZ, PxArticulationLimit(-1.4f, 0.2f));
	gDriveJoint->setDriveParams(PxArticulationAxis::eZ, PxArticulationDrive(100000.0f, 0.0f, PX_MAX_F32));
	gDriveJoint->setParentPose(PxTransform(PxVec3(0.0f, 0.25f, 0.9f)));
	gDriveJoint->setChildPose(PxTransform(PxVec3(0.0f, -0.05f, 0.0f)));

	const PxU32 linkHeight = 3;
	PxArticulationLink* currLeft = leftRoot;
	PxArticulationLink* currRight = gDriveLink;
	PxQuat rightParentRot(PxIdentity);
	PxQuat leftParentRot(PxIdentity);
	for(PxU32 i = 0; i < linkHeight; ++i)
	{
		const PxVec3 pos(0.5f, 0.55f + 0.1f * (1 + i), 0.0f);
		PxArticulationLink* leftLink = gArticulation->createLink(currLeft,
			PxTransform(pos + PxVec3(0.0f, sinAng * (2 * i + 1), 0.0f), leftRot));
		PxRigidActorExt::createExclusiveShape(*leftLink, PxBoxGeometry(0.05f, 0.05f, 1.0f), *gMaterial);
		PxRigidBodyExt::updateMassAndInertia(*leftLink, 1.0f);
		const PxVec3 leftAnchor = pos + PxVec3(0.0f, sinAng * (2 * i), -0.9f);
		joint = leftLink->getInboundJoint();
		joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchor), leftParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.0f, 0.0f, -1.0f), rightRot));
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		leftParentRot = leftRot;
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));

		PxArticulationLink* rightLink = gArticulation->createLink(currRight,
			PxTransform(pos + PxVec3(0.0f, sinAng * (2 * i + 1), 0.0f), rightRot));
		PxRigidActorExt::createExclusiveShape(*rightLink, PxBoxGeometry(0.05f, 0.05f, 1.0f), *gMaterial);
		PxRigidBodyExt::updateMassAndInertia(*rightLink, 1.0f);
		const PxVec3 rightAnchor = pos + PxVec3(0.0f, sinAng * (2 * i), 0.9f);
		joint = rightLink->getInboundJoint();
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchor), rightParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.0f, 0.0f, 1.0f), leftRot));
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));
		rightParentRot = rightRot;

		addLoopJoint(PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity),
			rightLink, PxTransform(PxIdentity)));
		currLeft = rightLink;
		currRight = leftLink;
	}

	PxArticulationLink* leftTop = gArticulation->createLink(currLeft,
		currLeft->getGlobalPose().transform(PxTransform(PxVec3(-0.5f, 0.0f, -1.0f), leftParentRot)));
	PxRigidActorExt::createExclusiveShape(*leftTop, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*leftTop, 1.0f);
	PxArticulationLink* rightTop = gArticulation->createLink(currRight,
		currRight->getGlobalPose().transform(PxTransform(PxVec3(-0.5f, 0.0f, 1.0f), rightParentRot)));
	PxRigidActorExt::createExclusiveShape(*rightTop, PxCapsuleGeometry(0.05f, 0.8f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*rightTop, 1.0f);

	joint = leftTop->getInboundJoint();
	joint->setParentPose(PxTransform(PxVec3(0.0f, 0.0f, -1.0f), currLeft->getGlobalPose().q.getConjugate()));
	joint->setChildPose(PxTransform(PxVec3(0.5f, 0.0f, 0.0f), leftTop->getGlobalPose().q.getConjugate()));
	joint->setJointType(PxArticulationJointType::eREVOLUTE);
	joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
	joint = rightTop->getInboundJoint();
	joint->setParentPose(PxTransform(PxVec3(0.0f, 0.0f, 1.0f), currRight->getGlobalPose().q.getConjugate()));
	joint->setChildPose(PxTransform(PxVec3(0.5f, 0.0f, 0.0f), rightTop->getGlobalPose().q.getConjugate()));
	joint->setJointType(PxArticulationJointType::eREVOLUTE);
	joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);

	currLeft = leftRoot;
	currRight = gDriveLink;
	rightParentRot = PxQuat(PxIdentity);
	leftParentRot = PxQuat(PxIdentity);
	for(PxU32 i = 0; i < linkHeight; ++i)
	{
		const PxVec3 pos(-0.5f, 0.55f + 0.1f * (1 + i), 0.0f);
		PxArticulationLink* leftLink = gArticulation->createLink(currLeft,
			PxTransform(pos + PxVec3(0.0f, sinAng * (2 * i + 1), 0.0f), leftRot));
		PxRigidActorExt::createExclusiveShape(*leftLink, PxBoxGeometry(0.05f, 0.05f, 1.0f), *gMaterial);
		PxRigidBodyExt::updateMassAndInertia(*leftLink, 1.0f);
		const PxVec3 leftAnchor = pos + PxVec3(0.0f, sinAng * (2 * i), -0.9f);
		joint = leftLink->getInboundJoint();
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchor), leftParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.0f, 0.0f, -1.0f), rightRot));
		leftParentRot = leftRot;
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));

		PxArticulationLink* rightLink = gArticulation->createLink(currRight,
			PxTransform(pos + PxVec3(0.0f, sinAng * (2 * i + 1), 0.0f), rightRot));
		PxRigidActorExt::createExclusiveShape(*rightLink, PxBoxGeometry(0.05f, 0.05f, 1.0f), *gMaterial);
		PxRigidBodyExt::updateMassAndInertia(*rightLink, 1.0f);
		const PxVec3 rightAnchor = pos + PxVec3(0.0f, sinAng * (2 * i), 0.9f);
		joint = rightLink->getInboundJoint();
		joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchor), rightParentRot));
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setChildPose(PxTransform(PxVec3(0.0f, 0.0f, 1.0f), leftRot));
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));
		rightParentRot = rightRot;

		addLoopJoint(PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity),
			rightLink, PxTransform(PxIdentity)));
		currLeft = rightLink;
		currRight = leftLink;
	}

	addLoopJoint(PxD6JointCreate(*gPhysics, currLeft, PxTransform(PxVec3(0.0f, 0.0f, -1.0f)),
		leftTop, PxTransform(PxVec3(-0.5f, 0.0f, 0.0f))));
	addLoopJoint(PxD6JointCreate(*gPhysics, currRight, PxTransform(PxVec3(0.0f, 0.0f, 1.0f)),
		rightTop, PxTransform(PxVec3(-0.5f, 0.0f, 0.0f))));

	const PxTransform topPose(PxVec3(0.0f, leftTop->getGlobalPose().p.y + 0.15f, 0.0f));
	gTopLink = gArticulation->createLink(leftTop, topPose);
	PxRigidActorExt::createExclusiveShape(*gTopLink, PxBoxGeometry(0.5f, 0.1f, 1.5f), *gMaterial);
	PxRigidBodyExt::updateMassAndInertia(*gTopLink, 1.0f);
	joint = gTopLink->getInboundJoint();
	joint->setJointType(PxArticulationJointType::eFIX);
	joint->setParentPose(PxTransform(PxVec3(0.0f, 0.0f, 0.0f)));
	joint->setChildPose(PxTransform(PxVec3(0.0f, -0.15f, -0.9f)));

	gScene->addArticulation(*gArticulation);
	for(PxU32 i = 0; i < gArticulation->getNbLinks(); ++i)
	{
		PxArticulationLink* link = NULL;
		gArticulation->getLinks(&link, 1, i);
		link->setLinearDamping(0.2f);
		link->setAngularDamping(0.2f);
		link->setMaxAngularVelocity(20.0f);
		link->setMaxLinearVelocity(100.0f);
		if(link != gTopLink)
		{
			for(PxU32 b = 0; b < link->getNbShapes(); ++b)
			{
				PxShape* shape = NULL;
				link->getShapes(&shape, 1, b);
				shape->setSimulationFilterData(PxFilterData(0, 0, 1, 0));
			}
		}
	}

	if(loadCargo)
		createCargoBoxes();
}

static PxReal getDriveCoordinate()
{
	if(!gBaseLink || !gDriveLink || !gDriveJoint)
		return 0.0f;
	const PxTransform parentFrame = gBaseLink->getGlobalPose() * gDriveJoint->getParentPose();
	const PxTransform childFrame = gDriveLink->getGlobalPose() * gDriveJoint->getChildPose();
	return parentFrame.q.getBasisVector2().dot(childFrame.p - parentFrame.p);
}

static void sampleBody(const PxTransform& pose, const PxVec3& lin, const PxVec3& ang)
{
	if(!isFinitePose(pose) || !isFiniteVec(lin) || !isFiniteVec(ang))
	{
		gNonFinite++;
		noteFirstFail("non_finite");
		return;
	}
	gMaxAbsPosition = PxMax(gMaxAbsPosition, PxAbs(pose.p.x));
	gMaxAbsPosition = PxMax(gMaxAbsPosition, PxAbs(pose.p.y));
	gMaxAbsPosition = PxMax(gMaxAbsPosition, PxAbs(pose.p.z));
	gMaxLinearSpeed = PxMax(gMaxLinearSpeed, lin.magnitude());
	gMaxAngularSpeed = PxMax(gMaxAngularSpeed, ang.magnitude());
}

static void sampleGateState(PxReal driveTarget)
{
	if(!gArticulation || !gDriveJoint || !gBaseLink || !gTopLink || !gDriveLink)
	{
		noteFirstFail("runtime_invariant");
		return;
	}

	gDriveCoord = getDriveCoordinate();
	if(!PxIsFinite(gDriveCoord))
	{
		gNonFinite++;
		noteFirstFail("non_finite");
		return;
	}
	gDriveErrorMax = PxMax(gDriveErrorMax, PxAbs(gDriveCoord - driveTarget));

	PxReal maxLinkAng = 0.0f;
	PxU32 maxLinkAngIndex = PX_MAX_U32;
	const PxU32 linkCount = gArticulation->getNbLinks();
	for(PxU32 i = 0; i < linkCount; ++i)
	{
		PxArticulationLink* link = NULL;
		if(gArticulation->getLinks(&link, 1, i) != 1 || !link)
		{
			noteFirstFail("runtime_invariant");
			return;
		}
		const PxVec3 ang = link->getAngularVelocity();
		sampleBody(link->getGlobalPose(), link->getLinearVelocity(), ang);
		const PxReal angMag = isFiniteVec(ang) ? ang.magnitude() : 0.0f;
		if(angMag > maxLinkAng)
		{
			maxLinkAng = angMag;
			maxLinkAngIndex = i;
		}
		if(i == 0)
			continue;
		PxArticulationJointReducedCoordinate* joint = link->getInboundJoint();
		if(!joint)
		{
			noteFirstFail("runtime_invariant");
			return;
		}
		const PxTransform parentFrame = joint->getParentArticulationLink().getGlobalPose() * joint->getParentPose();
		const PxTransform childFrame = link->getGlobalPose() * joint->getChildPose();
		if(!isFinitePose(parentFrame) || !isFinitePose(childFrame))
		{
			gNonFinite++;
			noteFirstFail("non_finite");
			return;
		}
		PxVec3 anchorDelta = childFrame.p - parentFrame.p;
		if(joint == gDriveJoint)
		{
			const PxVec3 driveAxis = parentFrame.q.getBasisVector2();
			anchorDelta -= driveAxis * driveAxis.dot(anchorDelta);
		}
		gInternalAnchorMax = PxMax(gInternalAnchorMax, anchorDelta.magnitude());

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
			gRevoluteSwingDegMax = PxMax(gRevoluteSwingDegMax, swing.getAngle() * 180.0f / PxPi);
			if(joint->getMotion(PxArticulationAxis::eTWIST) == PxArticulationMotion::eLIMITED)
			{
				const PxReal twistAngle = 2.0f * PxAtan2(twist.x, twist.w);
				const PxArticulationLimit limit = joint->getLimitParams(PxArticulationAxis::eTWIST);
				PxReal violation = 0.0f;
				if(twistAngle < limit.low)
					violation = limit.low - twistAngle;
				else if(twistAngle > limit.high)
					violation = twistAngle - limit.high;
				gTwistLimitViolationDegMax = PxMax(gTwistLimitViolationDegMax, violation * 180.0f / PxPi);
			}
		}
		else if(joint->getMotion(PxArticulationAxis::eTWIST) == PxArticulationMotion::eLOCKED &&
			joint->getMotion(PxArticulationAxis::eSWING1) == PxArticulationMotion::eLOCKED &&
			joint->getMotion(PxArticulationAxis::eSWING2) == PxArticulationMotion::eLOCKED)
		{
			PxQuat relativeQ = parentFrame.q.getConjugate() * childFrame.q;
			if(relativeQ.w < 0.0f)
				relativeQ = -relativeQ;
			relativeQ.normalize();
			gLockedAngularDegMax = PxMax(gLockedAngularDegMax,
				relativeQ.getAngle() * 180.0f / PxPi);
		}
	}

	for(PxU32 i = 0; i < gLoopJoints.size(); ++i)
	{
		PxRigidActor* actor0 = NULL;
		PxRigidActor* actor1 = NULL;
		gLoopJoints[i]->getActors(actor0, actor1);
		if(!actor0 || !actor1)
		{
			noteFirstFail("runtime_invariant");
			return;
		}
		const PxTransform frame0 = actor0->getGlobalPose() * gLoopJoints[i]->getLocalPose(PxJointActorIndex::eACTOR0);
		const PxTransform frame1 = actor1->getGlobalPose() * gLoopJoints[i]->getLocalPose(PxJointActorIndex::eACTOR1);
		if(!isFinitePose(frame0) || !isFinitePose(frame1))
		{
			gNonFinite++;
			noteFirstFail("non_finite");
			return;
		}
		gLoopAnchorMax = PxMax(gLoopAnchorMax, (frame1.p - frame0.p).magnitude());
	}

	const PxTransform topPose = gTopLink->getGlobalPose();
	gCargoCurrentMinY = PX_MAX_F32;
	gCargoCurrentMinTopClearance = PX_MAX_F32;
	gCargoCurrentMaxPlanarOffset = 0.0f;
	for(PxU32 i = 0; i < gCargo.size(); ++i)
	{
		PxRigidDynamic* box = gCargo[i];
		const PxTransform pose = box->getGlobalPose();
		const PxVec3 lin = box->getLinearVelocity();
		const PxVec3 ang = box->getAngularVelocity();
		sampleBody(pose, lin, ang);
		if(isFinitePose(pose))
		{
			gCargoMinY = PxMin(gCargoMinY, pose.p.y);
			gCargoCurrentMinY = PxMin(gCargoCurrentMinY, pose.p.y);
			gCargoCurrentMinTopClearance = PxMin(gCargoCurrentMinTopClearance,
				pose.p.y - topPose.p.y);
			const PxVec3 local = topPose.transformInv(pose.p);
			gCargoCurrentMaxPlanarOffset = PxMax(gCargoCurrentMaxPlanarOffset,
				PxMax(PxAbs(local.x), PxAbs(local.z)));
			gCargoMaxAbsPosition = PxMax(gCargoMaxAbsPosition, PxAbs(pose.p.x));
			gCargoMaxAbsPosition = PxMax(gCargoMaxAbsPosition, PxAbs(pose.p.y));
			gCargoMaxAbsPosition = PxMax(gCargoMaxAbsPosition, PxAbs(pose.p.z));
			gCargoMaxLinearSpeed = PxMax(gCargoMaxLinearSpeed, lin.magnitude());
		}
	}

	const PxTransform basePose = gBaseLink->getGlobalPose();
	const bool posesFinite = isFinitePose(basePose) && isFinitePose(topPose);
	if(posesFinite)
	{
		gBaseYMin = PxMin(gBaseYMin, basePose.p.y);
		gBaseYMax = PxMax(gBaseYMax, basePose.p.y);
		gTopYMin = PxMin(gTopYMin, topPose.p.y);
		gTopYMax = PxMax(gTopYMax, topPose.p.y);
		const PxVec3 topUp = topPose.q.rotate(PxVec3(0.0f, 1.0f, 0.0f));
		gTopTiltDegMax = PxMax(gTopTiltDegMax,
			PxAcos(PxClamp(topUp.y, -1.0f, 1.0f)) * 180.0f / PxPi);
		const PxVec3 baseUp = basePose.q.rotate(PxVec3(0.0f, 1.0f, 0.0f));
		const PxReal tiltDeg = PxAcos(PxClamp(baseUp.y, -1.0f, 1.0f)) * 180.0f / PxPi;
		gBaseTiltDegMax = PxMax(gBaseTiltDegMax, tiltDeg);
		if(basePose.p.y < 0.15f || basePose.p.y > 2.5f ||
			PxAbs(basePose.p.x) > 4.0f || PxAbs(basePose.p.z) > 4.0f ||
			baseUp.y < 0.5f)
			noteFirstFail("base_pose");
		if(topPose.p.y < 1.5f || topPose.p.y > 8.5f ||
			topPose.p.y + 0.5f < basePose.p.y || topPose.p.magnitude() > 12.0f)
			noteFirstFail("top_pose");
	}

	if(gInternalAnchorMax > gInternalAnchorCap)
		noteFirstFail("internal_anchor");
	if(gRevoluteSwingDegMax > gRevoluteSwingCapDeg)
		noteFirstFail("revolute_swing");
	if(gTwistLimitViolationDegMax > gTwistLimitCapDeg)
		noteFirstFail("twist_limit");
	if(gLoopAnchorMax > gLoopAnchorCap)
		noteFirstFail("loop_anchor");
	if(gMaxAbsPosition > gMaxAbsPositionCap)
		noteFirstFail("position_cap");
	if(gMaxLinearSpeed > gMaxLinearSpeedCap)
		noteFirstFail("linear_speed");
	if(gMaxAngularSpeed > gMaxAngularSpeedCap)
		noteFirstFail("angular_speed");
	if(!gCargo.empty() && gCargoMinY < gCargoMinYCap)
		noteFirstFail("cargo_min_y");
	if(gTopTiltDegMax > gTopTiltDegCap)
		noteFirstFail("top_tilt");
	if(!gCargo.empty() && gCargoCurrentMinTopClearance < gCargoMinTopClearanceCap)
		noteFirstFail("cargo_top_clearance");
	if(!gCargo.empty() && gCargoCurrentMaxPlanarOffset > gCargoPlanarOffsetCap)
		noteFirstFail("cargo_planar_offset");
	if(!gCargo.empty() && gCargoMaxLinearSpeed > gCargoLinearSpeedCap)
		noteFirstFail("cargo_linear_speed");

	if(gStallWindowFrame == PX_MAX_U32)
	{
		gStallWindowFrame = gCompletedFrames;
		gStallWindowCoord = gDriveCoord;
		gStallWindowTarget = driveTarget;
	}
	else if(gCompletedFrames - gStallWindowFrame >= 60)
	{
		const PxReal targetDelta = driveTarget - gStallWindowTarget;
		const PxReal coordDelta = gDriveCoord - gStallWindowCoord;
		if(gFirstStallFrame == PX_MAX_U32 &&
			PxAbs(targetDelta) >= 0.2f && PxAbs(coordDelta) <= 0.02f &&
			PxAbs(gDriveCoord - driveTarget) > 0.1f)
		{
			gFirstStallFrame = gStallWindowFrame + 1;
			noteFirstFail("stall");
			printf("[ScissorStall] firstFrame=%u endFrame=%u targetDelta=%.4f "
				"driveCoordDelta=%.4f\n",
				gFirstStallFrame, gCompletedFrames + 1, double(targetDelta),
				double(coordDelta));
		}
		gStallWindowFrame = gCompletedFrames;
		gStallWindowCoord = gDriveCoord;
		gStallWindowTarget = driveTarget;
	}

	if(gCompletedFrames < 20)
	{
		printf("[ScissorDiag] frame=%u baseY=%.3f topY=%.3f topTilt=%.3f drive=%.3f "
			"anchor=%.4f loop=%.4f twistDeg=%.3f swingDeg=%.3f lockedDeg=%.3f "
			"maxLin=%.3f maxAng=%.3f maxAngLink=%u cargoMinY=%.3f "
			"cargoNowY=%.3f cargoTopClearance=%.3f cargoPlanar=%.3f\n",
			gCompletedFrames + 1,
			posesFinite ? double(basePose.p.y) : 0.0,
			posesFinite ? double(topPose.p.y) : 0.0,
			double(gTopTiltDegMax),
			double(gDriveCoord), double(gInternalAnchorMax),
			double(gLoopAnchorMax), double(gTwistLimitViolationDegMax),
			double(gRevoluteSwingDegMax), double(gLockedAngularDegMax), double(gMaxLinearSpeed),
			double(gMaxAngularSpeed), maxLinkAngIndex,
			gCargo.empty() ? 0.0 : double(gCargoMinY),
			gCargo.empty() ? 0.0 : double(gCargoCurrentMinY),
			gCargo.empty() ? 0.0 : double(gCargoCurrentMinTopClearance),
			double(gCargoCurrentMaxPlanarOffset));
	}
}

static void resetRuntimeState()
{
	gArticulation = NULL;
	gDriveJoint = NULL;
	gBaseLink = NULL;
	gDriveLink = NULL;
	gTopLink = NULL;
	gLoopJoints.clear();
	gCargo.clear();
	gInitializationFailed = false;
	gExtensionsInitialized = false;
	gClosing = true;
	gCompletedFrames = 0;
	gFetchFailures = 0;
	gFetchErrorState = 0;
	gNonFinite = 0;
	gFirstFailFrame = PX_MAX_U32;
	gFirstFailReason = "none";
	gFirstStallFrame = PX_MAX_U32;
	gStallWindowFrame = PX_MAX_U32;
	gStallWindowCoord = 0.0f;
	gStallWindowTarget = 0.0f;
	gDriveCoord = 0.0f;
	gDriveErrorMax = 0.0f;
	gMaxAbsPosition = 0.0f;
	gMaxLinearSpeed = 0.0f;
	gMaxAngularSpeed = 0.0f;
	gInternalAnchorMax = 0.0f;
	gRevoluteSwingDegMax = 0.0f;
	gLockedAngularDegMax = 0.0f;
	gTwistLimitViolationDegMax = 0.0f;
	gLoopAnchorMax = 0.0f;
	gBaseYMin = PX_MAX_F32;
	gBaseYMax = -PX_MAX_F32;
	gBaseTiltDegMax = 0.0f;
	gTopYMin = PX_MAX_F32;
	gTopYMax = -PX_MAX_F32;
	gTopTiltDegMax = 0.0f;
	gCargoMinY = PX_MAX_F32;
	gCargoCurrentMinY = PX_MAX_F32;
	gCargoCurrentMinTopClearance = PX_MAX_F32;
	gCargoCurrentMaxPlanarOffset = 0.0f;
	gCargoMaxAbsPosition = 0.0f;
	gCargoMaxLinearSpeed = 0.0f;
}

static void cleanupHeadlessPhysics()
{
	if(gArticulation)
	{
		gArticulation->release();
		gArticulation = NULL;
	}
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	PX_RELEASE(gFoundation);
	gMaterial = NULL;
	gLoopJoints.clear();
	gCargo.clear();
}

static bool initHeadlessPhysics()
{
	resetRuntimeState();
	gErrorCallback.reset();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, NULL);
	if(!gPhysics)
		return false;
	gExtensionsInitialized = PxInitExtensions(*gPhysics, NULL);
	if(!gExtensionsInitialized)
		return false;

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(gOptions.dispatcherThreads);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = scissorFilter;
	sceneDesc.solverType = gOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
	if(!gMaterial)
		return false;
	PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	if(!ground)
		return false;
	gScene->addActor(*ground);

	gArticulation = gPhysics->createArticulationReducedCoordinate();
	if(!gArticulation)
		return false;
	createScissorLift(gCase == eARTIC_CASE_SCISSOR_CYCLE);
	return !gInitializationFailed && gDriveJoint && gBaseLink && gTopLink && gDriveLink;
}

static void stepHeadlessPhysics()
{
	if(!gScene || !gDriveJoint)
		return;
	PxReal driveValue = gDriveJoint->getDriveTarget(PxArticulationAxis::eZ);
	if(gClosing && driveValue < -1.2f)
		gClosing = false;
	else if(!gClosing && driveValue > 0.0f)
		gClosing = true;
	if(gClosing)
		driveValue -= gOptions.dt * 0.25f;
	else
		driveValue += gOptions.dt * 0.25f;
	gDriveJoint->setDriveTarget(PxArticulationAxis::eZ, driveValue);

	gScene->simulate(gOptions.dt);
	PxU32 errorState = 0;
	if(!gScene->fetchResults(true, &errorState))
	{
		gFetchFailures++;
		noteFirstFail("fetch_failure");
	}
	gFetchErrorState |= errorState;
	if(gFetchErrorState)
		noteFirstFail("fetch_error_state");
	sampleGateState(driveValue);
	gCompletedFrames++;
}

struct GateEvaluation
{
	bool pass;
	const char* reason;
	GateEvaluation() : pass(true), reason("none") {}
};

static void failGate(GateEvaluation& evaluation, const char* reason)
{
	if(!evaluation.pass)
		return;
	evaluation.pass = false;
	evaluation.reason = reason;
}

static GateEvaluation evaluateGate()
{
	GateEvaluation evaluation;
	if(gNonFinite)
		failGate(evaluation, "non_finite");
	if(gErrorCallback.getFatalCount())
		failGate(evaluation, "physics_error");
	if(gFetchFailures)
		failGate(evaluation, "fetch_failure");
	if(gFetchErrorState)
		failGate(evaluation, "fetch_error_state");
	if(gArticulation && gArticulation->getNbLinks() != gExpectedLinkCount)
		failGate(evaluation, "link_count");
	if(gLoopJoints.size() != gExpectedLoopCount)
		failGate(evaluation, "loop_joint_count");
	if(gCase == eARTIC_CASE_SCISSOR_CYCLE && gCargo.size() != gExpectedCargoCount)
		failGate(evaluation, "cargo_count");
	if(gCase == eARTIC_CASE_SCISSOR_UNLOADED && !gCargo.empty())
		failGate(evaluation, "cargo_count");
	if(gMaxAbsPosition > gMaxAbsPositionCap)
		failGate(evaluation, "position_cap");
	if(gMaxLinearSpeed > gMaxLinearSpeedCap)
		failGate(evaluation, "linear_speed");
	if(gMaxAngularSpeed > gMaxAngularSpeedCap)
		failGate(evaluation, "angular_speed");
	if(gInternalAnchorMax > gInternalAnchorCap)
		failGate(evaluation, "internal_anchor");
	if(gRevoluteSwingDegMax > gRevoluteSwingCapDeg)
		failGate(evaluation, "revolute_swing");
	if(gTwistLimitViolationDegMax > gTwistLimitCapDeg)
		failGate(evaluation, "twist_limit");
	if(gLoopAnchorMax > gLoopAnchorCap)
		failGate(evaluation, "loop_anchor");
	if(gFirstStallFrame != PX_MAX_U32)
		failGate(evaluation, "stall");
	if(gBaseYMin < 0.15f || gBaseYMax > 2.5f || gBaseTiltDegMax > 60.0f)
		failGate(evaluation, "base_pose");
	if(gTopYMin < 1.5f || gTopYMax > 8.5f)
		failGate(evaluation, "top_pose");
	if(gTopTiltDegMax > gTopTiltDegCap)
		failGate(evaluation, "top_tilt");
	if(gCompletedFrames == gOptions.frames && (gTopYMax - gTopYMin) < gStrokeMin)
		failGate(evaluation, "stroke");
	if(!gCargo.empty() && gCargoMinY < gCargoMinYCap)
		failGate(evaluation, "cargo_min_y");
	if(!gCargo.empty() && gCargoCurrentMinTopClearance < gCargoMinTopClearanceCap)
		failGate(evaluation, "cargo_top_clearance");
	if(!gCargo.empty() && gCargoCurrentMaxPlanarOffset > gCargoPlanarOffsetCap)
		failGate(evaluation, "cargo_planar_offset");
	if(!gCargo.empty() && gCargoMaxLinearSpeed > gCargoLinearSpeedCap)
		failGate(evaluation, "cargo_linear_speed");
	if(gCompletedFrames != gOptions.frames && evaluation.pass)
		failGate(evaluation, gFirstFailReason && std::strcmp(gFirstFailReason, "none") != 0
			? gFirstFailReason : "incomplete_simulation");
	return evaluation;
}

static void printGate(const GateEvaluation& evaluation)
{
	const PxReal stroke = (gTopYMax > gTopYMin) ? (gTopYMax - gTopYMin) : 0.0f;
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetArticulationRC case=%s solver=%s "
		"execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
		"dispatcherThreads=%u capability=SUPPORTED validation=GATED status=%s "
		"reason=%s nonFinite=%u physicsErrors=%u physicsWarnings=%u "
		"fetchFailures=%u fetchErrorState=%u "
		"linkCount=%u loopJointCount=%u cargoCount=%u "
		"firstFailFrame=%u firstStallFrame=%u "
		"twistLimitViolationDegMax=%.9g internalAnchorMax=%.9g "
		"revoluteSwingDegMax=%.9g lockedAngularDegMax=%.9g loopAnchorMax=%.9g "
		"maxAbsPosition=%.9g maxLinearSpeed=%.9g maxAngularSpeed=%.9g "
		"driveCoord=%.9g driveErrorMax=%.9g "
		"baseYMin=%.9g baseYMax=%.9g baseTiltDegMax=%.9g "
		"topYMin=%.9g topYMax=%.9g topTiltDegMax=%.9g stroke=%.9g "
		"cargoMinY=%.9g cargoCurrentMinY=%.9g cargoCurrentMinTopClearance=%.9g "
		"cargoCurrentMaxPlanarOffset=%.9g "
		"cargoMaxAbsPosition=%.9g cargoMaxLinearSpeed=%.9g "
		"twistLimitCapDeg=5 internalAnchorCap=0.1 revoluteSwingCapDeg=5 "
		"loopAnchorCap=0.1 topTiltDegCap=%.9g cargoMinTopClearanceCap=%.9g "
		"cargoPlanarOffsetCap=%.9g cargoLinearSpeedCap=%.9g\n",
		getHeadlessCaseName(gCase),
		Snippets::getSolverTypeName(gOptions.solverType),
		Snippets::getExecutionName(gOptions.execution),
		gOptions.frames, gCompletedFrames, double(gOptions.dt), gOptions.seed,
		gOptions.dispatcherThreads, evaluation.pass ? "PASS" : "FAIL",
		evaluation.reason, gNonFinite, gErrorCallback.getFatalCount(),
		gErrorCallback.getWarningCount(), gFetchFailures, gFetchErrorState,
		gArticulation ? gArticulation->getNbLinks() : 0u,
		PxU32(gLoopJoints.size()), PxU32(gCargo.size()),
		gFirstFailFrame, gFirstStallFrame,
		double(gTwistLimitViolationDegMax), double(gInternalAnchorMax),
		double(gRevoluteSwingDegMax), double(gLockedAngularDegMax), double(gLoopAnchorMax),
		double(gMaxAbsPosition), double(gMaxLinearSpeed), double(gMaxAngularSpeed),
		double(gDriveCoord), double(gDriveErrorMax),
		double(gBaseYMin), double(gBaseYMax), double(gBaseTiltDegMax),
		double(gTopYMin), double(gTopYMax), double(gTopTiltDegMax), double(stroke),
		gCargo.empty() ? 0.0 : double(gCargoMinY),
		gCargo.empty() ? 0.0 : double(gCargoCurrentMinY),
		gCargo.empty() ? 0.0 : double(gCargoCurrentMinTopClearance),
		double(gCargoCurrentMaxPlanarOffset),
		double(gCargoMaxAbsPosition), double(gCargoMaxLinearSpeed),
		double(gTopTiltDegCap), double(gCargoMinTopClearanceCap),
		double(gCargoPlanarOffsetCap), double(gCargoLinearSpeedCap));
}

static int reportConfigurationError(const Snippets::HeadlessOptions& options,
	const char* message)
{
	printf("[AVBD_GATE_ERROR] snippet=SnippetArticulationRC message=%s\n", message);
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetArticulationRC case=config-error "
		"solver=%s execution=%s requestedFrames=%u completedFrames=0 dt=%.9g "
		"seed=%u dispatcherThreads=%u capability=SUPPORTED validation=GATED "
		"status=ERROR reason=config nonFinite=0 physicsErrors=0 physicsWarnings=0\n",
		Snippets::getSolverTypeName(options.solverType),
		Snippets::getExecutionName(options.execution), options.frames,
		double(options.dt), options.seed, options.dispatcherThreads);
	return Snippets::eHEADLESS_CONFIG_ERROR;
}

} // namespace

bool snippetArticulationHeadlessShouldRun(int argc, const char* const* argv)
{
#ifndef RENDER_SNIPPET
	PX_UNUSED(argc);
	PX_UNUSED(argv);
	return true;
#else
	if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS")))
		return true;
	for(int i = 1; i < argc; ++i)
	{
		if(argv[i] && std::strcmp(argv[i], "--headless") == 0)
			return true;
	}
	return false;
#endif
}

int snippetArticulationHeadlessMain(int argc, const char* const* argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);

	Snippets::HeadlessOptions defaults;
	defaults.caseName = "scissor-cycle";
	defaults.frames = 600;
	defaults.seed = 1;
	defaults.dispatcherThreads = 2;
	defaults.dt = 1.0f / 60.0f;

	Snippets::HeadlessOptions options;
	std::string parseError;
	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options, parseError))
		return reportConfigurationError(options, parseError.c_str());

	for(int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if(!arg)
			continue;
		if(Snippets::isCommonHeadlessOption(arg))
			continue;
		return reportConfigurationError(options, "unknown_argument");
	}

#ifndef RENDER_SNIPPET
	options.headless = true;
#endif

	ArticHeadlessCase headlessCase = eARTIC_CASE_SCISSOR_CYCLE;
	if(!tryParseHeadlessCase(options.caseName.c_str(), headlessCase))
		return reportConfigurationError(options, "invalid_--case_value");
	options.caseName = getHeadlessCaseName(headlessCase);

	if(!Snippets::applyExecutionEnvironment(options))
		return reportConfigurationError(options, "execution_environment");

	Snippets::printHeadlessConfig("SnippetArticulationRC", options);
	gOptions = options;
	gCase = headlessCase;

	if(!initHeadlessPhysics())
	{
		cleanupHeadlessPhysics();
		return reportConfigurationError(options, "initialization_failed");
	}

	for(PxU32 i = 0; i < gOptions.frames; ++i)
	{
		stepHeadlessPhysics();
		if(gNonFinite || gFetchFailures || gFetchErrorState)
			break;
	}

	const GateEvaluation evaluation = evaluateGate();
	printGate(evaluation);
	cleanupHeadlessPhysics();
	return evaluation.pass ? Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
}
