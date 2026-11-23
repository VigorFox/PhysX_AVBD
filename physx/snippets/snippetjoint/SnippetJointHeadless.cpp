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

// Distilled SnippetJoint headless gates. Scene contract from PhysX_AVBD_2;
// not a copy of that 10k-line demo. Visual path stays in SnippetJoint.cpp.

#include "SnippetJointHeadless.h"

#include "PxPhysicsAPI.h"
#include "extensions/PxConstraintExt.h"
#include "../snippetcommon/SnippetHeadless.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace physx;

namespace
{

enum JointHeadlessCase
{
	eJOINT_CASE_HANG,
	eJOINT_CASE_HANG_SPHERICAL,
	eJOINT_CASE_HANG_FIXED,
	eJOINT_CASE_HANG_D6,
	eJOINT_CASE_FIXED_NO_BREAK,
	eJOINT_CASE_FIXED_BREAK,
	eJOINT_CASE_UNSUPPORTED
};

enum
{
	eFILTER_CHAIN = 1,
	eFILTER_PROJECTILE = 2
};

static const PxU32 gHangChainLength = 5;
static const PxU32 gHangChainCount = 3;
static const PxU32 gImpactLaunchFrame = 120;
static const PxReal gImpactRadius = 0.75f;
static const PxReal gImpactHeight = 6.0f;
static const PxReal gImpactSpeed = 20.0f;
static const PxReal gImpactTransverseOffset = 0.25f;
static const PxReal gAnchorErrorCap = 0.25f;
static const PxReal gMaxAbsPositionCap = 500.0f;
static const PxReal gMaxLinearSpeedCap = 200.0f;
static const PxReal gMaxAngularSpeedCap = 200.0f;

static Snippets::HeadlessOptions gOptions;
static JointHeadlessCase gCase = eJOINT_CASE_HANG;
static const char* gUnsupportedReason = "not_ported_this_slice";
static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation* gFoundation = NULL;
static PxPhysics* gPhysics = NULL;
static PxDefaultCpuDispatcher* gDispatcher = NULL;
static PxScene* gScene = NULL;
static PxMaterial* gMaterial = NULL;
static bool gExtensionsInitialized = false;
static bool gInitializationFailed = false;

static std::vector<PxRigidDynamic*> gBodies;
static std::vector<PxJoint*> gJoints;
static std::vector<PxJoint*> gBreakableJoints;
static std::vector<PxU8> gBreakPollReported;
static std::vector<PxU8> gBreakCallbackReported;
static PxRigidDynamic* gTargetBody = NULL;
static PxRigidDynamic* gProjectile = NULL;

static PxU32 gSimFrame = 0;
static PxU32 gCompletedFrames = 0;
static PxU32 gFetchFailures = 0;
static PxU32 gFetchErrorState = 0;
static PxU32 gLaunchFailures = 0;
static PxU32 gNonFinite = 0;
static PxU32 gBrokenCount = 0;
static PxU32 gFirstBrokenFrame = PX_MAX_U32;
static PxU32 gFirstContactFrame = PX_MAX_U32;
static PxU32 gBreakCallbackCount = 0;
static PxU32 gBreakCallbackIdentityMatches = 0;
static PxU32 gBreakCallbackConstraintMismatches = 0;
static PxU32 gBreakCallbackExternalReferenceMismatches = 0;
static PxU32 gBreakCallbackTypeMismatches = 0;
static PxU32 gBreakCallbackBrokenFlagMismatches = 0;
static PxU32 gBreakCallbackDuplicateMismatches = 0;
static PxU32 gBreakCallbackPollMismatches = 0;
static PxReal gMaxAnchorError = 0.0f;
static PxReal gMaxAbsPosition = 0.0f;
static PxReal gMaxLinearSpeed = 0.0f;
static PxReal gMaxAngularSpeed = 0.0f;
static PxReal gLinearBreakForce = 1000.0f;
static PxReal gAngularBreakForce = 100000.0f;

static PxI32 findBreakableJoint(const PxConstraint* constraint)
{
	for(PxU32 i = 0; i < gBreakableJoints.size(); ++i)
	{
		if(gBreakableJoints[i] && gBreakableJoints[i]->getConstraint() == constraint)
			return PxI32(i);
	}
	return -1;
}

class JointGateCallback : public PxSimulationEventCallback
{
public:
	virtual void onConstraintBreak(PxConstraintInfo* constraints, PxU32 count) PX_OVERRIDE
	{
		for(PxU32 i = 0; i < count; ++i)
		{
			gBreakCallbackCount++;
			const PxConstraintInfo& info = constraints[i];
			const PxI32 index = findBreakableJoint(info.constraint);
			const bool constraintMatches = index >= 0;
			const bool externalReferenceMatches =
				constraintMatches &&
				info.externalReference == static_cast<PxJoint*>(gBreakableJoints[PxU32(index)]);
			const bool typeMatches = info.type == PxU32(PxConstraintExtIDs::eJOINT);
			const bool brokenFlagMatches =
				info.constraint &&
				info.constraint->getFlags().isSet(PxConstraintFlag::eBROKEN);
			const bool duplicate =
				constraintMatches &&
				PxU32(index) < gBreakCallbackReported.size() &&
				gBreakCallbackReported[PxU32(index)] != 0;
			if(!constraintMatches)
				gBreakCallbackConstraintMismatches++;
			if(!externalReferenceMatches)
				gBreakCallbackExternalReferenceMismatches++;
			if(!typeMatches)
				gBreakCallbackTypeMismatches++;
			if(!brokenFlagMatches)
				gBreakCallbackBrokenFlagMismatches++;
			if(duplicate)
				gBreakCallbackDuplicateMismatches++;
			if(constraintMatches && externalReferenceMatches && typeMatches &&
				brokenFlagMatches && !duplicate)
			{
				gBreakCallbackReported[PxU32(index)] = 1;
				gBreakCallbackIdentityMatches++;
			}
		}
	}

	virtual void onWake(PxActor**, PxU32) PX_OVERRIDE {}
	virtual void onSleep(PxActor**, PxU32) PX_OVERRIDE {}
	virtual void onTrigger(PxTriggerPair*, PxU32) PX_OVERRIDE {}
	virtual void onAdvance(const PxRigidBody* const*, const PxTransform*, PxU32) PX_OVERRIDE {}

	virtual void onContact(const PxContactPairHeader& pairHeader,
		const PxContactPair*, PxU32) PX_OVERRIDE
	{
		const PxActor* a0 = pairHeader.actors[0];
		const PxActor* a1 = pairHeader.actors[1];
		const bool hit =
			(a0 == gProjectile && a1 == gTargetBody) ||
			(a1 == gProjectile && a0 == gTargetBody);
		if(hit && gFirstContactFrame == PX_MAX_U32)
			gFirstContactFrame = gCompletedFrames + 1;
	}
};

static JointGateCallback gCallback;

static PxFilterFlags jointHeadlessFilterShader(
	PxFilterObjectAttributes attributes0, PxFilterData filterData0,
	PxFilterObjectAttributes attributes1, PxFilterData filterData1,
	PxPairFlags& pairFlags, const void*, PxU32)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	pairFlags = PxPairFlag::eCONTACT_DEFAULT;
	if(filterData0.word0 == eFILTER_PROJECTILE || filterData1.word0 == eFILTER_PROJECTILE)
		pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND;
	return PxFilterFlag::eDEFAULT;
}

static bool tryParseHeadlessCase(const char* value, JointHeadlessCase& headlessCase,
	const char** unsupportedReason)
{
	if(Snippets::equalsIgnoreCase(value, "hang") ||
		Snippets::equalsIgnoreCase(value, "passive") ||
		Snippets::equalsIgnoreCase(value, "hang-fixed"))
	{
		headlessCase = eJOINT_CASE_UNSUPPORTED;
		*unsupportedReason = "multi_body_fixed_lock_chain";
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "hang-spherical"))
	{
		headlessCase = eJOINT_CASE_HANG_SPHERICAL;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "hang-d6"))
	{
		headlessCase = eJOINT_CASE_HANG_D6;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "fixed-no-break"))
	{
		headlessCase = eJOINT_CASE_FIXED_NO_BREAK;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "fixed-break"))
	{
		headlessCase = eJOINT_CASE_FIXED_BREAK;
		return true;
	}

	static const char* kUnsupported[][2] = {
		{ "impact-all", "not_ported_this_slice" },
		{ "wide-joint-stress", "not_ported_this_slice" },
		{ "impact", "not_ported_this_slice" },
		{ "force-static", "not_ported_this_slice" },
		{ "force-offset", "not_ported_this_slice" },
		{ "force-pair", "not_ported_this_slice" },
		{ "force-pair-disabled", "not_ported_this_slice" },
		{ "spherical-cone-inside", "not_ported_this_slice" },
		{ "spherical-cone-outside", "not_ported_this_slice" },
		{ "native-reaction", "not_ported_this_slice" },
		{ "native-no-break", "not_ported_this_slice" },
		{ "native-break", "not_ported_this_slice" },
		{ "revolute-motor", "not_ported_this_slice" },
		{ "revolute-motor-limit", "not_ported_this_slice" },
		{ "revolute-motor-dynamic-limit", "not_ported_this_slice" },
		{ "revolute-motor-freespin", "not_ported_this_slice" },
		{ "revolute-motor-dynamic-freespin", "not_ported_this_slice" },
		{ "revolute-motor-ratio", "not_ported_this_slice" },
		{ "revolute-motor-contact", "not_ported_this_slice" },
		{ "revolute-motor-kinematic", "not_ported_this_slice" },
		{ "revolute-motor-off-principal", "not_ported_this_slice" },
		{ "revolute-motor-off-center", "not_ported_this_slice" },
		{ "revolute-motor-spatial", "not_ported_this_slice" },
		{ "revolute-motor-dynamic-off-principal", "not_ported_this_slice" },
		{ "revolute-motor-dynamic-off-center", "not_ported_this_slice" },
		{ "revolute-motor-dynamic-spatial", "not_ported_this_slice" }
	};
	for(PxU32 i = 0; i < sizeof(kUnsupported) / sizeof(kUnsupported[0]); ++i)
	{
		if(Snippets::equalsIgnoreCase(value, kUnsupported[i][0]))
		{
			headlessCase = eJOINT_CASE_UNSUPPORTED;
			*unsupportedReason = kUnsupported[i][1];
			return true;
		}
	}
	return false;
}

static const char* getHeadlessCaseName(JointHeadlessCase headlessCase)
{
	switch(headlessCase)
	{
	case eJOINT_CASE_HANG: return "hang";
	case eJOINT_CASE_HANG_SPHERICAL: return "hang-spherical";
	case eJOINT_CASE_HANG_FIXED: return "hang-fixed";
	case eJOINT_CASE_HANG_D6: return "hang-d6";
	case eJOINT_CASE_FIXED_NO_BREAK: return "fixed-no-break";
	case eJOINT_CASE_FIXED_BREAK: return "fixed-break";
	case eJOINT_CASE_UNSUPPORTED: return gOptions.caseName.c_str();
	default: return "unknown";
	}
}

static const char* getGateJointName(JointHeadlessCase headlessCase)
{
	switch(headlessCase)
	{
	case eJOINT_CASE_FIXED_NO_BREAK:
	case eJOINT_CASE_FIXED_BREAK:
		return "fixed";
	default:
		return "all";
	}
}

static bool usesProjectile(JointHeadlessCase headlessCase)
{
	return headlessCase == eJOINT_CASE_FIXED_NO_BREAK ||
		headlessCase == eJOINT_CASE_FIXED_BREAK;
}

static void registerBreakableJoint(PxJoint* joint)
{
	if(!joint)
		return;
	gBreakableJoints.push_back(joint);
	gBreakPollReported.push_back(0);
	gBreakCallbackReported.push_back(0);
}

static void setFilterWord(PxRigidActor* actor, PxU32 word0)
{
	if(!actor)
		return;
	const PxU32 nb = actor->getNbShapes();
	for(PxU32 i = 0; i < nb; ++i)
	{
		PxShape* shape = NULL;
		actor->getShapes(&shape, 1, i);
		if(!shape)
			continue;
		PxFilterData data = shape->getSimulationFilterData();
		data.word0 = word0;
		shape->setSimulationFilterData(data);
	}
}

static PxJoint* createLimitedSpherical(PxRigidActor* a0, const PxTransform& t0,
	PxRigidActor* a1, const PxTransform& t1)
{
	PxSphericalJoint* j = PxSphericalJointCreate(*gPhysics, a0, t0, a1, t1);
	if(!j)
		return NULL;
	j->setLimitCone(PxJointLimitCone(PxPi / 4, PxPi / 4));
	j->setSphericalJointFlag(PxSphericalJointFlag::eLIMIT_ENABLED, true);
	return j;
}

static PxJoint* createBreakableFixed(PxRigidActor* a0, const PxTransform& t0,
	PxRigidActor* a1, const PxTransform& t1)
{
	PxFixedJoint* j = PxFixedJointCreate(*gPhysics, a0, t0, a1, t1);
	if(!j)
		return NULL;
	j->setBreakForce(gLinearBreakForce, gAngularBreakForce);
	j->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
	j->setConstraintFlag(PxConstraintFlag::eDISABLE_PREPROCESSING, true);
	registerBreakableJoint(j);
	return j;
}

static PxJoint* createDampedD6(PxRigidActor* a0, const PxTransform& t0,
	PxRigidActor* a1, const PxTransform& t1)
{
	PxD6Joint* j = PxD6JointCreate(*gPhysics, a0, t0, a1, t1);
	if(!j)
		return NULL;
	j->setAngularDriveConfig(PxD6AngularDriveConfig::eSLERP);
	j->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
	j->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
	j->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
	j->setDrive(PxD6Drive::eSLERP, PxD6JointDrive(0, 1000, FLT_MAX, true));
	return j;
}

typedef PxJoint* (*JointCreateFunction)(PxRigidActor*, const PxTransform&,
	PxRigidActor*, const PxTransform&);

static void createChain(const PxTransform& t, JointCreateFunction createJoint)
{
	PxVec3 offset(2.0f, 0.0f, 0.0f);
	PxTransform localTm(offset);
	PxRigidDynamic* prev = NULL;
	for(PxU32 i = 0; i < gHangChainLength; ++i)
	{
		PxRigidDynamic* current =
			PxCreateDynamic(*gPhysics, t * localTm, PxBoxGeometry(2.0f, 0.5f, 0.5f),
				*gMaterial, 1.0f);
		if(!current)
		{
			gInitializationFailed = true;
			return;
		}
		PxJoint* joint = (*createJoint)(prev, prev ? PxTransform(offset) : t,
			current, PxTransform(-offset));
		if(!joint)
		{
			current->release();
			gInitializationFailed = true;
			return;
		}
		setFilterWord(current, eFILTER_CHAIN);
		gScene->addActor(*current);
		gBodies.push_back(current);
		gJoints.push_back(joint);
		prev = current;
		localTm.p.x += 4.0f;
	}
}

static void createHangScene()
{
	if(gCase == eJOINT_CASE_HANG || gCase == eJOINT_CASE_HANG_SPHERICAL)
		createChain(PxTransform(PxVec3(0.0f, 20.0f, 0.0f)), createLimitedSpherical);
	if(gInitializationFailed)
		return;
	if(gCase == eJOINT_CASE_HANG || gCase == eJOINT_CASE_HANG_FIXED)
		createChain(PxTransform(PxVec3(0.0f, 20.0f, -10.0f)), createBreakableFixed);
	if(gInitializationFailed)
		return;
	if(gCase == eJOINT_CASE_HANG || gCase == eJOINT_CASE_HANG_D6)
		createChain(PxTransform(PxVec3(0.0f, 20.0f, -20.0f)), createDampedD6);
}

static void createIsolatedFixedTarget()
{
	const PxTransform targetPose(PxVec3(0.0f, 10.0f, 0.0f));
	gTargetBody = PxCreateDynamic(*gPhysics, targetPose,
		PxBoxGeometry(2.0f, 0.5f, 0.5f), *gMaterial, 1.0f);
	if(!gTargetBody)
	{
		gInitializationFailed = true;
		return;
	}
	PxJoint* joint = createBreakableFixed(NULL, targetPose, gTargetBody, PxTransform(PxIdentity));
	if(!joint)
	{
		gTargetBody->release();
		gTargetBody = NULL;
		gInitializationFailed = true;
		return;
	}
	setFilterWord(gTargetBody, eFILTER_CHAIN);
	gScene->addActor(*gTargetBody);
	gBodies.push_back(gTargetBody);
	gJoints.push_back(joint);
}

static void launchProjectile()
{
	if(gProjectile || !gTargetBody)
	{
		gLaunchFailures++;
		return;
	}
	const PxVec3 targetPosition = gTargetBody->getGlobalPose().p;
	const PxTransform projectilePose(
		targetPosition + PxVec3(0.0f, gImpactTransverseOffset, gImpactHeight));
	gProjectile = PxCreateDynamic(*gPhysics, projectilePose,
		PxSphereGeometry(gImpactRadius), *gMaterial, 1.0f);
	if(!gProjectile ||
		!PxRigidBodyExt::setMassAndUpdateInertia(*gProjectile, gTargetBody->getMass()))
	{
		PX_RELEASE(gProjectile);
		gLaunchFailures++;
		return;
	}
	gProjectile->setAngularDamping(0.5f);
	gProjectile->setLinearVelocity(gTargetBody->getLinearVelocity() +
		PxVec3(0.0f, 0.0f, -gImpactSpeed));
	setFilterWord(gProjectile, eFILTER_PROJECTILE);
	gScene->addActor(*gProjectile);
	gBodies.push_back(gProjectile);
}

static PxTransform actorPose(PxRigidActor* actor, const PxTransform& local)
{
	return actor ? actor->getGlobalPose() * local : local;
}

static bool isFiniteVec(const PxVec3& v)
{
	return PxIsFinite(v.x) && PxIsFinite(v.y) && PxIsFinite(v.z);
}

static bool isFiniteQuat(const PxQuat& q)
{
	return PxIsFinite(q.x) && PxIsFinite(q.y) && PxIsFinite(q.z) && PxIsFinite(q.w);
}

static void sampleGateState()
{
	for(PxU32 i = 0; i < gBodies.size(); ++i)
	{
		PxRigidDynamic* body = gBodies[i];
		if(!body)
			continue;
		const PxTransform pose = body->getGlobalPose();
		const PxVec3 lin = body->getLinearVelocity();
		const PxVec3 ang = body->getAngularVelocity();
		if(!isFiniteVec(pose.p) || !isFiniteQuat(pose.q) ||
			!isFiniteVec(lin) || !isFiniteVec(ang))
		{
			gNonFinite++;
			continue;
		}
		gMaxAbsPosition = PxMax(gMaxAbsPosition, PxAbs(pose.p.x));
		gMaxAbsPosition = PxMax(gMaxAbsPosition, PxAbs(pose.p.y));
		gMaxAbsPosition = PxMax(gMaxAbsPosition, PxAbs(pose.p.z));
		gMaxLinearSpeed = PxMax(gMaxLinearSpeed, lin.magnitude());
		gMaxAngularSpeed = PxMax(gMaxAngularSpeed, ang.magnitude());
	}

	for(PxU32 i = 0; i < gJoints.size(); ++i)
	{
		PxJoint* joint = gJoints[i];
		if(!joint)
			continue;
		PxConstraint* constraint = joint->getConstraint();
		if(constraint && constraint->getFlags().isSet(PxConstraintFlag::eBROKEN))
			continue;
		PxRigidActor* a0 = NULL;
		PxRigidActor* a1 = NULL;
		joint->getActors(a0, a1);
		const PxTransform c0 = actorPose(a0, joint->getLocalPose(PxJointActorIndex::eACTOR0));
		const PxTransform c1 = actorPose(a1, joint->getLocalPose(PxJointActorIndex::eACTOR1));
		if(!isFiniteVec(c0.p) || !isFiniteVec(c1.p))
		{
			gNonFinite++;
			continue;
		}
		gMaxAnchorError = PxMax(gMaxAnchorError, (c0.p - c1.p).magnitude());
	}

	PxU32 polled = 0;
	for(PxU32 i = 0; i < gBreakableJoints.size(); ++i)
	{
		PxJoint* joint = gBreakableJoints[i];
		PxConstraint* constraint = joint ? joint->getConstraint() : NULL;
		const bool broken = constraint &&
			constraint->getFlags().isSet(PxConstraintFlag::eBROKEN);
		if(broken && !gBreakPollReported[i])
		{
			gBreakPollReported[i] = 1;
			gBrokenCount++;
			if(gFirstBrokenFrame == PX_MAX_U32)
				gFirstBrokenFrame = gCompletedFrames + 1;
		}
		if(gBreakPollReported[i])
			polled++;
	}
	gBreakCallbackPollMismatches = (polled != gBreakCallbackCount) ? 1u : 0u;
}

static void resetRuntimeState()
{
	gBodies.clear();
	gJoints.clear();
	gBreakableJoints.clear();
	gBreakPollReported.clear();
	gBreakCallbackReported.clear();
	gTargetBody = NULL;
	gProjectile = NULL;
	gSimFrame = 0;
	gCompletedFrames = 0;
	gFetchFailures = 0;
	gFetchErrorState = 0;
	gLaunchFailures = 0;
	gNonFinite = 0;
	gBrokenCount = 0;
	gFirstBrokenFrame = PX_MAX_U32;
	gFirstContactFrame = PX_MAX_U32;
	gBreakCallbackCount = 0;
	gBreakCallbackIdentityMatches = 0;
	gBreakCallbackConstraintMismatches = 0;
	gBreakCallbackExternalReferenceMismatches = 0;
	gBreakCallbackTypeMismatches = 0;
	gBreakCallbackBrokenFlagMismatches = 0;
	gBreakCallbackDuplicateMismatches = 0;
	gBreakCallbackPollMismatches = 0;
	gMaxAnchorError = 0.0f;
	gMaxAbsPosition = 0.0f;
	gMaxLinearSpeed = 0.0f;
	gMaxAngularSpeed = 0.0f;
	gInitializationFailed = false;
	gExtensionsInitialized = false;
}

static void cleanupHeadlessPhysics()
{
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
	gBodies.clear();
	gJoints.clear();
	gBreakableJoints.clear();
	gTargetBody = NULL;
	gProjectile = NULL;
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
	sceneDesc.filterShader = jointHeadlessFilterShader;
	sceneDesc.simulationEventCallback = &gCallback;
	sceneDesc.solverType = gOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	if(!gMaterial)
		return false;
	PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	if(!ground)
		return false;
	gScene->addActor(*ground);

	if(gCase == eJOINT_CASE_HANG || gCase == eJOINT_CASE_HANG_SPHERICAL ||
		gCase == eJOINT_CASE_HANG_FIXED || gCase == eJOINT_CASE_HANG_D6)
		createHangScene();
	else
		createIsolatedFixedTarget();
	return !gInitializationFailed;
}

static void stepHeadlessPhysics()
{
	if(!gScene)
		return;
	if(usesProjectile(gCase) && gSimFrame == gImpactLaunchFrame)
		launchProjectile();
	gScene->simulate(gOptions.dt);
	PxU32 errorState = 0;
	if(!gScene->fetchResults(true, &errorState))
		gFetchFailures++;
	gFetchErrorState |= errorState;
	sampleGateState();
	gCompletedFrames++;
	gSimFrame++;
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
	if(gLaunchFailures)
		failGate(evaluation, "launch_failure");
	if(gMaxAbsPosition > gMaxAbsPositionCap)
		failGate(evaluation, "position_cap");
	if(gMaxLinearSpeed > gMaxLinearSpeedCap)
		failGate(evaluation, "linear_speed");
	if(gMaxAngularSpeed > gMaxAngularSpeedCap)
		failGate(evaluation, "angular_speed");
	if(gMaxAnchorError > gAnchorErrorCap)
		failGate(evaluation, "anchor_error");
	if(gBreakCallbackPollMismatches)
		failGate(evaluation, "break_callback_mismatch");
	if(gBreakCallbackConstraintMismatches ||
		gBreakCallbackExternalReferenceMismatches ||
		gBreakCallbackTypeMismatches ||
		gBreakCallbackBrokenFlagMismatches ||
		gBreakCallbackDuplicateMismatches)
		failGate(evaluation, "break_callback_identity");

	if(gCase == eJOINT_CASE_HANG)
	{
		if(gBodies.size() != gHangChainLength * gHangChainCount)
			failGate(evaluation, "actor_count");
		if(gBrokenCount || gBreakCallbackCount)
			failGate(evaluation, "unexpected_break");
	}
	else if(gCase == eJOINT_CASE_HANG_SPHERICAL ||
		gCase == eJOINT_CASE_HANG_FIXED ||
		gCase == eJOINT_CASE_HANG_D6)
	{
		if(gBodies.size() != gHangChainLength)
			failGate(evaluation, "actor_count");
		if(gBrokenCount || gBreakCallbackCount)
			failGate(evaluation, "unexpected_break");
	}
	else if(gCase == eJOINT_CASE_FIXED_NO_BREAK)
	{
		if(gBrokenCount || gBreakCallbackCount)
			failGate(evaluation, "unexpected_break");
		if(gFirstContactFrame == PX_MAX_U32)
			failGate(evaluation, "expected_hits");
	}
	else if(gCase == eJOINT_CASE_FIXED_BREAK)
	{
		if(gBrokenCount != 1 || gBreakCallbackCount != 1)
			failGate(evaluation, "break_count");
		if(gBreakCallbackIdentityMatches != 1)
			failGate(evaluation, "break_callback_identity");
		if(gFirstContactFrame == PX_MAX_U32)
			failGate(evaluation, "expected_hits");
		if(gFirstBrokenFrame == PX_MAX_U32 || gFirstBrokenFrame < 1 ||
			gFirstBrokenFrame > gOptions.frames)
			failGate(evaluation, "first_broken_frame");
		if(gFirstContactFrame != PX_MAX_U32 && gFirstBrokenFrame < gFirstContactFrame)
			failGate(evaluation, "break_before_contact");
	}
	return evaluation;
}

static void printGate(const GateEvaluation& evaluation)
{
	const PxU32 expectedHits = usesProjectile(gCase) ? 1u : 0u;
	const PxU32 hitChains = gFirstContactFrame == PX_MAX_U32 ? 0u : 1u;
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetJoint case=%s joint=%s solver=%s "
		"execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
		"dispatcherThreads=%u capability=SUPPORTED validation=GATED status=%s "
		"reason=%s nonFinite=%u physicsErrors=%u physicsWarnings=%u "
		"fetchFailures=%u fetchErrorState=%u launchFailures=%u "
		"expectedHits=%u hitChains=%u responseProjectiles=%u "
		"brokenCount=%u breakCallbacks=%u breakCallbackIdentityMatches=%u "
		"breakCallbackConstraintMismatches=%u "
		"breakCallbackExternalReferenceMismatches=%u "
		"breakCallbackTypeMismatches=%u breakCallbackBrokenFlagMismatches=%u "
		"breakCallbackDuplicateMismatches=%u breakCallbackPollMismatches=%u "
		"firstBrokenFrame=%u firstContactFrame=%u dynamicActors=%u "
		"maxAnchorError=%.9g maxAbsPosition=%.9g maxLinearSpeed=%.9g "
		"maxAngularSpeed=%.9g linearBreakForce=%.9g\n",
		getHeadlessCaseName(gCase), getGateJointName(gCase),
		Snippets::getSolverTypeName(gOptions.solverType),
		Snippets::getExecutionName(gOptions.execution),
		gOptions.frames, gCompletedFrames, double(gOptions.dt), gOptions.seed,
		gOptions.dispatcherThreads, evaluation.pass ? "PASS" : "FAIL",
		evaluation.reason, gNonFinite, gErrorCallback.getFatalCount(),
		gErrorCallback.getWarningCount(), gFetchFailures, gFetchErrorState,
		gLaunchFailures, expectedHits, hitChains, hitChains, gBrokenCount,
		gBreakCallbackCount, gBreakCallbackIdentityMatches,
		gBreakCallbackConstraintMismatches,
		gBreakCallbackExternalReferenceMismatches,
		gBreakCallbackTypeMismatches, gBreakCallbackBrokenFlagMismatches,
		gBreakCallbackDuplicateMismatches, gBreakCallbackPollMismatches,
		gFirstBrokenFrame, gFirstContactFrame, PxU32(gBodies.size()),
		double(gMaxAnchorError), double(gMaxAbsPosition),
		double(gMaxLinearSpeed), double(gMaxAngularSpeed),
		double(gLinearBreakForce));
}

static int reportConfigurationError(const Snippets::HeadlessOptions& options,
	const char* message)
{
	printf("[AVBD_GATE_ERROR] snippet=SnippetJoint message=%s\n", message);
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetJoint case=config-error "
		"solver=%s execution=%s requestedFrames=%u completedFrames=0 dt=%.9g "
		"seed=%u dispatcherThreads=%u capability=SUPPORTED validation=GATED "
		"status=ERROR reason=config nonFinite=0 physicsErrors=0 physicsWarnings=0\n",
		Snippets::getSolverTypeName(options.solverType),
		Snippets::getExecutionName(options.execution), options.frames,
		double(options.dt), options.seed, options.dispatcherThreads);
	return Snippets::eHEADLESS_CONFIG_ERROR;
}

static int reportUnsupported(const Snippets::HeadlessOptions& options, const char* reason)
{
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetJoint case=%s solver=%s "
		"execution=%s requestedFrames=%u completedFrames=0 dt=%.9g seed=%u "
		"dispatcherThreads=%u capability=UNSUPPORTED validation=GATED status=ERROR "
		"reason=%s nonFinite=0 physicsErrors=0 physicsWarnings=0 "
		"fetchFailures=0 fetchErrorState=0 launchFailures=0\n",
		options.caseName.c_str(), Snippets::getSolverTypeName(options.solverType),
		Snippets::getExecutionName(options.execution), options.frames,
		double(options.dt), options.seed, options.dispatcherThreads, reason);
	return Snippets::eHEADLESS_UNSUPPORTED;
}

} // namespace

bool snippetJointHeadlessShouldRun(int argc, const char* const* argv)
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

int snippetJointHeadlessMain(int argc, const char* const* argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);

	Snippets::HeadlessOptions defaults;
	defaults.caseName = "hang-spherical";
	defaults.frames = 180;
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

	JointHeadlessCase headlessCase = eJOINT_CASE_HANG;
	const char* unsupportedReason = "not_ported_this_slice";
	if(!tryParseHeadlessCase(options.caseName.c_str(), headlessCase, &unsupportedReason))
		return reportConfigurationError(options, "invalid_--case_value");
	gUnsupportedReason = unsupportedReason;
	if(headlessCase != eJOINT_CASE_UNSUPPORTED)
		options.caseName = getHeadlessCaseName(headlessCase);

	if(usesProjectile(headlessCase) && !options.framesExplicit)
		options.frames = 600u;

	if(!Snippets::applyExecutionEnvironment(options))
		return reportConfigurationError(options, "execution_environment");

	Snippets::printHeadlessConfig("SnippetJoint", options);
	if(headlessCase == eJOINT_CASE_UNSUPPORTED)
		return reportUnsupported(options, unsupportedReason);

	gOptions = options;
	gCase = headlessCase;
	gLinearBreakForce = 1000.0f;
	gAngularBreakForce = 100000.0f;
	if(gCase == eJOINT_CASE_FIXED_NO_BREAK)
	{
		gLinearBreakForce = PX_MAX_F32;
		gAngularBreakForce = PX_MAX_F32;
	}
	else if(gCase == eJOINT_CASE_FIXED_BREAK)
	{
		gLinearBreakForce = 250.0f;
		gAngularBreakForce = PX_MAX_F32;
	}

	if(!initHeadlessPhysics())
	{
		cleanupHeadlessPhysics();
		return reportConfigurationError(options, "initialization_failed");
	}

	for(PxU32 i = 0; i < gOptions.frames; ++i)
		stepHeadlessPhysics();

	const GateEvaluation evaluation = evaluateGate();
	printGate(evaluation);
	cleanupHeadlessPhysics();
	return evaluation.pass ? Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
}
