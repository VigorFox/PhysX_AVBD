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
// This snippet illustrates the use of simple contact reports and contact modification.
//
// It defines a filter shader function that requests contact modification and 
// touch reports for all pairs, and a contact callback function that saves 
// the contact points. It configures the scene to use this filter and callback, 
// and prints the number of contact reports each frame. If rendering, it renders 
// each contact as a line whose length and direction are defined by the contact 
// impulse.
// This test sets up a situation that would be unstable without contact modification
// due to very large mass ratios. This test uses local mass modification to make
// the configuration stable. It also demonstrates how to interpret contact impulses
// when local mass modification is used.
// Local mass modification can be disabled with the MODIFY_MASS_PROPERTIES #define 
// to demonstrate the instability if it was not used.
// 
// ****************************************************************************

#include "PxPhysicsAPI.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include <cstdio>
#include <string>

using namespace physx;

#define MODIFY_MASS_PROPERTIES 1

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;
static PxRigidDynamic*			gHeadlessBody0 = NULL;
static PxRigidDynamic*			gHeadlessBody1 = NULL;
static PxRigidStatic*			gHeadlessGround = NULL;
static bool						gExtensionsInitialized = false;

struct ContactModificationMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 modifyCallbackCount;
	PxU32 modifiedPairCount;
	PxU32 modifiedPointCount;
	PxU32 reportCallbackCount;
	PxU32 reportPointCount;
	PxU32 identityErrors;
	PxU32 scaleReadbackErrors;
	PxU32 maxImpulseReadbackErrors;
	PxU32 targetVelocityReadbackErrors;
	PxU32 body0Actor0Count;
	PxU32 body0Actor1Count;
	PxU32 nonFinite;
	PxReal maxAbsBody0X;
	PxReal peakAbsBody0VelocityX;
	PxReal peakAbsBody0VelocityY;
	PxReal peakBody0AngularSpeed;
	PxU32 peakBody0AngularFrame;
	PxReal maxBody0Y;
	PxReal peakBody0VelocityY;
	PxReal minBody0Y;
	PxReal peakBody1Speed;
	PxReal peakAbsBody1VelocityY;
	PxReal peakBody1VelocityY;
	PxReal minBody1VelocityY;
	PxReal peakBody1AngularSpeed;
	PxReal peakBody0MinusBody1VelocityY;
	PxReal minBody0SpeedAfterModify;
	PxReal finalBody0Speed;
	PxReal finalBody0AngularSpeed;
	PxReal finalBody0VelocityX;
	PxVec3 finalBody0AngularVelocity;
	PxReal finalBody0ContactVelocityX;
	PxReal finalBody1Speed;
	PxReal finalBody1AngularSpeed;
	PxReal finalBody1VelocityX;
	PxVec3 finalBody1AngularVelocity;
	PxReal minBody1Y;
	PxReal maxBody1Y;
	PxReal finiteCheckpointVelocityY;
	PxReal finiteCheckpointAngularSpeed;
	PxReal body0Mass;
	PxReal body0InertiaZ;
	PxReal maxReportedImpulse;
	PxReal expectedBody0LinearDelta;
	PxReal expectedBody1LinearDelta;
	PxReal expectedBody0AngularDelta;
	PxReal expectedBody1AngularDelta;
	PxReal actualBody0AngularDelta;
	PxReal actualBody1AngularDelta;
	PxReal scaleLinearVelocityResidual;
	PxReal scaleAngularVelocityResidual;
	PxU32 cleanupComplete;

	ContactModificationMetrics()
	: completedFrames(0), fetchFailures(0), modifyCallbackCount(0),
	  modifiedPairCount(0), modifiedPointCount(0), reportCallbackCount(0),
	  reportPointCount(0), identityErrors(0), scaleReadbackErrors(0),
	  maxImpulseReadbackErrors(0), targetVelocityReadbackErrors(0),
	  body0Actor0Count(0), body0Actor1Count(0), nonFinite(0),
	  maxAbsBody0X(0.0f), peakAbsBody0VelocityX(0.0f),
	  peakAbsBody0VelocityY(0.0f), peakBody0AngularSpeed(0.0f),
	  peakBody0AngularFrame(0),
	  maxBody0Y(-PX_MAX_F32), peakBody0VelocityY(-PX_MAX_F32),
	  minBody0Y(PX_MAX_F32), peakBody1Speed(0.0f),
	  peakAbsBody1VelocityY(0.0f), peakBody1VelocityY(-PX_MAX_F32),
	  minBody1VelocityY(PX_MAX_F32), peakBody1AngularSpeed(0.0f),
	  peakBody0MinusBody1VelocityY(-PX_MAX_F32),
	  minBody0SpeedAfterModify(PX_MAX_F32), finalBody0Speed(0.0f),
	  finalBody0AngularSpeed(0.0f), finalBody0VelocityX(0.0f),
	  finalBody0AngularVelocity(0.0f), finalBody0ContactVelocityX(0.0f),
	  finalBody1Speed(0.0f), finalBody1AngularSpeed(0.0f),
	  finalBody1VelocityX(0.0f), finalBody1AngularVelocity(0.0f),
	  minBody1Y(PX_MAX_F32), maxBody1Y(-PX_MAX_F32),
	  finiteCheckpointVelocityY(PX_MAX_F32),
	  finiteCheckpointAngularSpeed(PX_MAX_F32),
	  body0Mass(0.0f), body0InertiaZ(0.0f),
	  maxReportedImpulse(0.0f),
	  expectedBody0LinearDelta(0.0f), expectedBody1LinearDelta(0.0f),
	  expectedBody0AngularDelta(0.0f), expectedBody1AngularDelta(0.0f),
	  actualBody0AngularDelta(0.0f), actualBody1AngularDelta(0.0f),
	  scaleLinearVelocityResidual(0.0f),
	  scaleAngularVelocityResidual(0.0f), cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static ContactModificationMetrics gMetrics;
static PxVec3 gHeadlessInitialLinearVelocity[2];
static PxVec3 gHeadlessInitialAngularVelocity[2];
static PxVec3 gExpectedLinearVelocityDelta[2];
static PxVec3 gExpectedAngularVelocityDelta[2];

PxArray<PxVec3> gContactPositions;
PxArray<PxVec3> gContactImpulses;
PxArray<PxVec3> gContactLinearImpulses[2];
PxArray<PxVec3> gContactAngularImpulses[2];

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(), name);
}

static bool isTargetOwnershipAuthorityCase()
{
	return isHeadlessCase("ownership-target-normal-mu0") ||
		isHeadlessCase("ownership-target-normal-mu0-reverse") ||
		isHeadlessCase("ownership-target-normal-mu1") ||
		isHeadlessCase("ownership-target-normal-mu1-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu0") ||
		isHeadlessCase("ownership-target-tangent-mu0-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1") ||
		isHeadlessCase("ownership-target-tangent-mu1-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-free") ||
		isHeadlessCase("ownership-target-tangent-mu1-free-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw-reverse") ||
		isHeadlessCase("ownership-passive-friction-component") ||
		isHeadlessCase("ownership-passive-friction-component-reverse") ||
		isHeadlessCase("ownership-passive-friction-component-yaw") ||
		isHeadlessCase("ownership-passive-friction-component-yaw-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component") ||
		isHeadlessCase("ownership-restitution-friction-component-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw-reverse") ||
		isHeadlessCase("ownership-target-combined-mu1-finite") ||
		isHeadlessCase("ownership-target-combined-mu1-finite-reverse");
}

static bool isTargetOwnershipReverseCase()
{
	return isHeadlessCase("ownership-target-normal-mu0-reverse") ||
		isHeadlessCase("ownership-target-normal-mu1-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu0-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-free-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw-reverse") ||
		isHeadlessCase("ownership-passive-friction-component-reverse") ||
		isHeadlessCase("ownership-passive-friction-component-yaw-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw-reverse") ||
		isHeadlessCase("ownership-target-combined-mu1-finite-reverse");
}

static bool targetOwnershipHasFriction()
{
	return isHeadlessCase("ownership-target-normal-mu1") ||
		isHeadlessCase("ownership-target-normal-mu1-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1") ||
		isHeadlessCase("ownership-target-tangent-mu1-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-free") ||
		isHeadlessCase("ownership-target-tangent-mu1-free-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw-reverse") ||
		isHeadlessCase("ownership-passive-friction-component") ||
		isHeadlessCase("ownership-passive-friction-component-reverse") ||
		isHeadlessCase("ownership-passive-friction-component-yaw") ||
		isHeadlessCase("ownership-passive-friction-component-yaw-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component") ||
		isHeadlessCase("ownership-restitution-friction-component-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw-reverse") ||
		isHeadlessCase("ownership-target-combined-mu1-finite") ||
		isHeadlessCase("ownership-target-combined-mu1-finite-reverse");
}

static bool targetOwnershipHasNormalTarget()
{
	return isHeadlessCase("ownership-target-normal-mu0") ||
		isHeadlessCase("ownership-target-normal-mu0-reverse") ||
		isHeadlessCase("ownership-target-normal-mu1") ||
		isHeadlessCase("ownership-target-normal-mu1-reverse") ||
		isHeadlessCase("ownership-target-combined-mu1-finite") ||
		isHeadlessCase("ownership-target-combined-mu1-finite-reverse");
}

static bool targetOwnershipHasTangentTarget()
{
	return isHeadlessCase("ownership-target-tangent-mu0") ||
		isHeadlessCase("ownership-target-tangent-mu0-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1") ||
		isHeadlessCase("ownership-target-tangent-mu1-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-free") ||
		isHeadlessCase("ownership-target-tangent-mu1-free-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw-reverse") ||
		isHeadlessCase("ownership-target-combined-mu1-finite") ||
		isHeadlessCase("ownership-target-combined-mu1-finite-reverse");
}

static bool targetOwnershipLocksAngularMotion()
{
	return !isHeadlessCase("ownership-target-tangent-mu1-free") &&
		!isHeadlessCase("ownership-target-tangent-mu1-free-reverse") &&
		!isHeadlessCase("ownership-target-tangent-mu1-manifold") &&
		!isHeadlessCase("ownership-target-tangent-mu1-manifold-reverse") &&
		!isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw") &&
		!isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw-reverse") &&
		!isHeadlessCase("ownership-passive-friction-mu1-manifold") &&
		!isHeadlessCase("ownership-passive-friction-mu1-manifold-reverse") &&
		!isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw") &&
		!isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw-reverse") &&
		!isHeadlessCase("ownership-passive-friction-component") &&
		!isHeadlessCase("ownership-passive-friction-component-reverse") &&
		!isHeadlessCase("ownership-passive-friction-component-yaw") &&
		!isHeadlessCase("ownership-passive-friction-component-yaw-reverse") &&
		!isHeadlessCase("ownership-restitution-friction-component") &&
		!isHeadlessCase("ownership-restitution-friction-component-reverse") &&
		!isHeadlessCase("ownership-restitution-friction-component-yaw") &&
		!isHeadlessCase("ownership-restitution-friction-component-yaw-reverse");
}

static bool isTargetOwnershipManifoldCase()
{
	return isHeadlessCase("ownership-target-tangent-mu1-manifold") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw-reverse");
}

static bool isTargetOwnershipManifoldYawCase()
{
	return isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-target-tangent-mu1-manifold-yaw-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw-reverse");
}

static bool isPassiveFrictionManifoldCase()
{
	return isHeadlessCase("ownership-passive-friction-mu1-manifold") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-reverse") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw") ||
		isHeadlessCase("ownership-passive-friction-mu1-manifold-yaw-reverse");
}

static bool isPassiveFrictionComponentCase()
{
	return isHeadlessCase("ownership-passive-friction-component") ||
		isHeadlessCase("ownership-passive-friction-component-reverse") ||
		isHeadlessCase("ownership-passive-friction-component-yaw") ||
		isHeadlessCase("ownership-passive-friction-component-yaw-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component") ||
		isHeadlessCase("ownership-restitution-friction-component-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw-reverse");
}

static bool isRestitutionFrictionComponentCase()
{
	return isHeadlessCase("ownership-restitution-friction-component") ||
		isHeadlessCase("ownership-restitution-friction-component-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw-reverse");
}

static bool isPassiveFrictionComponentYawCase()
{
	return isHeadlessCase("ownership-passive-friction-component-yaw") ||
		isHeadlessCase("ownership-passive-friction-component-yaw-reverse") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw") ||
		isHeadlessCase("ownership-restitution-friction-component-yaw-reverse");
}

static bool targetOwnershipHasFiniteCap()
{
	return isHeadlessCase("ownership-target-combined-mu1-finite") ||
		isHeadlessCase("ownership-target-combined-mu1-finite-reverse");
}

static bool isRestitutionThresholdCase()
{
	return isHeadlessCase("restitution-threshold-below") ||
		isHeadlessCase("restitution-threshold-above");
}

static bool isTiltedRestitutionCase()
{
	return isHeadlessCase("restitution-tilted");
}

static bool isTiltedFiniteImpulseCase()
{
	return isHeadlessCase("finite-max-impulse-tilted");
}

static bool isOffCenterFiniteImpulseCase()
{
	return isHeadlessCase("finite-max-impulse-offcenter");
}

static bool isSpatialFiniteImpulseCase()
{
	return isTiltedFiniteImpulseCase() ||
		isOffCenterFiniteImpulseCase();
}

static bool isDynamicPairCase()
{
	return isHeadlessCase("mass-scale") ||
		isHeadlessCase("finite-scales-control") ||
		isHeadlessCase("finite-scales") ||
		isHeadlessCase("tangent-target");
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 120;
	defaults.caseName = "normal";
	defaults.solverType = PxSolverType::eAVBD;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	for(int i = 1; i < argc; ++i)
	{
		if(!Snippets::isCommonHeadlessOption(argv[i]))
		{
			error = std::string("unknown option: ") +
				(argv[i] ? argv[i] : "<null>");
			return false;
		}
	}
	if(!isHeadlessCase("normal") &&
		!isHeadlessCase("target-velocity") &&
		!isHeadlessCase("max-impulse") &&
		!isHeadlessCase("ownership-shallow") &&
		!isHeadlessCase("ownership-deep") &&
		!isHeadlessCase("ownership-deep-tilted") &&
		!isHeadlessCase("ownership-bounce") &&
		!isHeadlessCase("restitution-threshold-below") &&
		!isHeadlessCase("restitution-threshold-above") &&
		!isHeadlessCase("restitution-tilted") &&
		!isHeadlessCase("mass-scale") &&
		!isHeadlessCase("finite-max-impulse-control") &&
		!isHeadlessCase("finite-max-impulse") &&
		!isHeadlessCase("finite-max-impulse-tilted") &&
		!isHeadlessCase("finite-max-impulse-offcenter") &&
		!isHeadlessCase("finite-scales-control") &&
		!isHeadlessCase("finite-scales") &&
		!isHeadlessCase("tangent-target") &&
		!isTargetOwnershipAuthorityCase())
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.frames != 120)
	{
		error = "--frames must be 120";
		return false;
	}
	return true;
}

static PxFilterFlags contactReportFilterShader(	PxFilterObjectAttributes attributes0, PxFilterData filterData0, 
												PxFilterObjectAttributes attributes1, PxFilterData filterData1,
												PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(filterData0);
	PX_UNUSED(filterData1);
	PX_UNUSED(constantBlockSize);
	PX_UNUSED(constantBlock);

	// all initial and persisting reports for everything, with per-point data
	pairFlags = PxPairFlag::eSOLVE_CONTACT | PxPairFlag::eDETECT_DISCRETE_CONTACT
			  |	PxPairFlag::eNOTIFY_TOUCH_FOUND 
			  | PxPairFlag::eNOTIFY_TOUCH_PERSISTS
			  | PxPairFlag::eNOTIFY_CONTACT_POINTS
			  | PxPairFlag::eMODIFY_CONTACTS;
	return PxFilterFlag::eDEFAULT;
}

static bool isUnorderedActorPair(
	const PxActor* actor0, const PxActor* actor1,
	const PxActor* expected0, const PxActor* expected1)
{
	return (actor0 == expected0 && actor1 == expected1) ||
		(actor0 == expected1 && actor1 == expected0);
}

static bool isExpectedHeadlessPair(
	const PxActor* actor0, const PxActor* actor1)
{
	if(isPassiveFrictionComponentCase())
	{
		return isUnorderedActorPair(
				actor0, actor1, gHeadlessBody0, gHeadlessGround) ||
			isUnorderedActorPair(
				actor0, actor1, gHeadlessBody0, gHeadlessBody1);
	}
	if(isDynamicPairCase())
		return isUnorderedActorPair(
			actor0, actor1, gHeadlessBody0, gHeadlessBody1);
	return isUnorderedActorPair(
		actor0, actor1, gHeadlessBody0, gHeadlessGround);
}

class ContactModifyCallback: public PxContactModifyCallback
{
	void onContactModify(PxContactModifyPair* const pairs, PxU32 count)
	{
		if(gHeadlessOptions.headless)
		{
			++gMetrics.modifyCallbackCount;
			gMetrics.modifiedPairCount += count;
			for(PxU32 i = 0; i < count; ++i)
			{
				const bool body0IsActor0 =
					pairs[i].actor[0] == gHeadlessBody0;
				const bool body0IsActor1 =
					pairs[i].actor[1] == gHeadlessBody0;
				gMetrics.body0Actor0Count += body0IsActor0 ? 1u : 0u;
				gMetrics.body0Actor1Count += body0IsActor1 ? 1u : 0u;
				const bool expectedPair = isExpectedHeadlessPair(
					pairs[i].actor[0], pairs[i].actor[1]);
				if(!expectedPair)
				++gMetrics.identityErrors;

				for(PxU32 c = 0; c < pairs[i].contacts.size(); ++c)
				{
					if(isHeadlessCase("normal"))
					{
						const PxVec3 body0Normal =
							PxVec3(0.6f, 0.8f, 0.0f);
						pairs[i].contacts.setNormal(
							c, body0IsActor0 ? body0Normal : -body0Normal);
					}
					else if(isHeadlessCase("target-velocity"))
					{
						const PxVec3 body0Target(0.0f, 3.0f, 0.0f);
						const PxVec3 authoredTarget =
							body0IsActor0 ? body0Target : -body0Target;
						pairs[i].contacts.setTargetVelocity(c, authoredTarget);
						if((pairs[i].contacts.getTargetVelocity(c) -
							authoredTarget).magnitudeSquared() > 1.0e-12f)
							++gMetrics.targetVelocityReadbackErrors;
					}
					else if(isHeadlessCase("tangent-target"))
					{
						const PxVec3 body0Target(0.0f, 3.0f, 0.0f);
						const PxVec3 authoredTarget =
							body0IsActor0 ? body0Target : -body0Target;
						pairs[i].contacts.setTargetVelocity(c, authoredTarget);
						if((pairs[i].contacts.getTargetVelocity(c) -
							authoredTarget).magnitudeSquared() > 1.0e-12f)
							++gMetrics.targetVelocityReadbackErrors;
					}
					else if(isTargetOwnershipAuthorityCase())
					{
						PxVec3 body0Target(0.0f);
						if(targetOwnershipHasNormalTarget())
							body0Target.y = 3.0f;
						if(targetOwnershipHasTangentTarget())
							body0Target.x = 3.0f;
						const PxVec3 authoredTarget =
							body0IsActor0 ? body0Target : -body0Target;
						pairs[i].contacts.setTargetVelocity(c, authoredTarget);
						if((pairs[i].contacts.getTargetVelocity(c) -
							authoredTarget).magnitudeSquared() > 1.0e-12f)
							++gMetrics.targetVelocityReadbackErrors;
						if(targetOwnershipHasFiniteCap())
						{
							const PxReal maxImpulse = 0.25f;
							pairs[i].contacts.setMaxImpulse(c, maxImpulse);
							if(PxAbs(pairs[i].contacts.getMaxImpulse(c) -
								maxImpulse) > 1.0e-6f)
								++gMetrics.maxImpulseReadbackErrors;
						}
					}
					else if(isHeadlessCase("max-impulse"))
					{
						pairs[i].contacts.setMaxImpulse(c, 0.0f);
						if(pairs[i].contacts.getMaxImpulse(c) != 0.0f)
							++gMetrics.maxImpulseReadbackErrors;
					}
					else if(isHeadlessCase("finite-max-impulse") ||
						isSpatialFiniteImpulseCase())
					{
						const PxReal maxImpulse =
							isSpatialFiniteImpulseCase() ? 1.0f : 0.25f;
						pairs[i].contacts.setMaxImpulse(c, maxImpulse);
						if(PxAbs(pairs[i].contacts.getMaxImpulse(c) -
							maxImpulse) > 1.0e-6f)
							++gMetrics.maxImpulseReadbackErrors;
					}
					else if(isHeadlessCase("ownership-shallow") ||
						isHeadlessCase("ownership-deep") ||
						isHeadlessCase("ownership-deep-tilted") ||
						isHeadlessCase("ownership-bounce") ||
						isRestitutionThresholdCase() ||
						isTiltedRestitutionCase())
					{
						pairs[i].contacts.setMaxImpulse(c, PX_MAX_REAL);
						if(pairs[i].contacts.getMaxImpulse(c) != PX_MAX_REAL)
							++gMetrics.maxImpulseReadbackErrors;
					}
					else if(isHeadlessCase("mass-scale"))
					{
						const bool body1IsActor0 =
							pairs[i].actor[0] == gHeadlessBody1;
						if(body1IsActor0)
						{
							pairs[i].contacts.setInvMassScale0(0.0f);
							pairs[i].contacts.setInvInertiaScale0(0.0f);
							if(pairs[i].contacts.getInvMassScale0() != 0.0f ||
								pairs[i].contacts.getInvInertiaScale0() != 0.0f)
								++gMetrics.scaleReadbackErrors;
						}
						else
						{
							pairs[i].contacts.setInvMassScale1(0.0f);
							pairs[i].contacts.setInvInertiaScale1(0.0f);
							if(pairs[i].contacts.getInvMassScale1() != 0.0f ||
								pairs[i].contacts.getInvInertiaScale1() != 0.0f)
								++gMetrics.scaleReadbackErrors;
						}
					}
					else if(isHeadlessCase("finite-scales"))
					{
						const bool body1IsActor0 =
							pairs[i].actor[0] == gHeadlessBody1;
						const PxReal body0InvMassScale = 0.25f;
						const PxReal body0InvInertiaScale = 0.6f;
						const PxReal body1InvMassScale = 1.5f;
						const PxReal body1InvInertiaScale = 0.35f;
						if(body0IsActor0)
						{
							pairs[i].contacts.setInvMassScale0(
								body0InvMassScale);
							pairs[i].contacts.setInvInertiaScale0(
								body0InvInertiaScale);
						}
						else
						{
							pairs[i].contacts.setInvMassScale1(
								body0InvMassScale);
							pairs[i].contacts.setInvInertiaScale1(
								body0InvInertiaScale);
						}
						if(body1IsActor0)
						{
							pairs[i].contacts.setInvMassScale0(
								body1InvMassScale);
							pairs[i].contacts.setInvInertiaScale0(
								body1InvInertiaScale);
						}
						else
						{
							pairs[i].contacts.setInvMassScale1(
								body1InvMassScale);
							pairs[i].contacts.setInvInertiaScale1(
								body1InvInertiaScale);
						}
						const PxReal actor0MassScale =
							pairs[i].contacts.getInvMassScale0();
						const PxReal actor0InertiaScale =
							pairs[i].contacts.getInvInertiaScale0();
						const PxReal actor1MassScale =
							pairs[i].contacts.getInvMassScale1();
						const PxReal actor1InertiaScale =
							pairs[i].contacts.getInvInertiaScale1();
						const PxReal expectedActor0MassScale =
							body0IsActor0 ? body0InvMassScale :
								body1InvMassScale;
						const PxReal expectedActor0InertiaScale =
							body0IsActor0 ? body0InvInertiaScale :
								body1InvInertiaScale;
						const PxReal expectedActor1MassScale =
							body0IsActor1 ? body0InvMassScale :
								body1InvMassScale;
						const PxReal expectedActor1InertiaScale =
							body0IsActor1 ? body0InvInertiaScale :
								body1InvInertiaScale;
						if(PxAbs(actor0MassScale -
								expectedActor0MassScale) > 1.0e-6f ||
							PxAbs(actor0InertiaScale -
								expectedActor0InertiaScale) > 1.0e-6f ||
							PxAbs(actor1MassScale -
								expectedActor1MassScale) > 1.0e-6f ||
							PxAbs(actor1InertiaScale -
								expectedActor1InertiaScale) > 1.0e-6f)
							++gMetrics.scaleReadbackErrors;
					}
					++gMetrics.modifiedPointCount;
				}
			}
			return;
		}
#if MODIFY_MASS_PROPERTIES
		//We define a maximum mass ratio that we will accept in this test, which is a ratio of 2
		const PxReal maxMassRatio = 2.f;

		for(PxU32 i = 0; i < count; i++)
		{
			const PxRigidDynamic* dynamic0 = pairs[i].actor[0]->is<PxRigidDynamic>();
			const PxRigidDynamic* dynamic1 = pairs[i].actor[1]->is<PxRigidDynamic>();
			if(dynamic0 != NULL && dynamic1 != NULL)
			{
				//We only want to perform local mass modification between 2 dynamic bodies because we intend on 
				//normalizing the mass ratios between the pair within a tolerable range

				PxReal mass0 = dynamic0->getMass();
				PxReal mass1 = dynamic1->getMass();

				if(mass0 > mass1)
				{
					//dynamic0 is heavier than dynamic1 so we will locally increase the mass of dynamic1 
					//to be half the mass of dynamic0.
					PxReal ratio = mass0/mass1;
					if(ratio > maxMassRatio)
					{
						PxReal invMassScale = maxMassRatio/ratio;
						pairs[i].contacts.setInvMassScale1(invMassScale);
						pairs[i].contacts.setInvInertiaScale1(invMassScale);
					}
				}
				else
				{
					//dynamic1 is heavier than dynamic0 so we will locally increase the mass of dynamic0 
					//to be half the mass of dynamic1.
					PxReal ratio = mass1/mass0;
					if(ratio > maxMassRatio)
					{
						PxReal invMassScale = maxMassRatio/ratio;
						pairs[i].contacts.setInvMassScale0(invMassScale);
						pairs[i].contacts.setInvInertiaScale0(invMassScale);
					}
				}
			}
		}
#endif
	}
};

ContactModifyCallback gContactModifyCallback;

static PxU32 extractContactsWithMassScale(
	const PxContactPair& pair, PxContactPairPoint* userBuffer,
	PxU32 bufferSize, PxReal& invMassScale0, PxReal& invMassScale1,
	PxReal& invInertiaScale0, PxReal& invInertiaScale1)
{
	const PxU8* contactStream = pair.contactPoints;
	const PxU8* patchStream = pair.contactPatches;
	const PxU32* faceIndices = pair.getInternalFaceIndices();

	PxU32 nbContacts = 0;

	if(pair.contactCount && bufferSize)
	{
		PxContactStreamIterator iter(patchStream, contactStream, faceIndices, pair.patchCount, pair.contactCount);

		const PxReal* impulses = reinterpret_cast<const PxReal*>(pair.contactImpulses);

		PxU32 flippedContacts = (pair.flags & PxContactPairFlag::eINTERNAL_CONTACTS_ARE_FLIPPED);
		PxU32 hasImpulses = (pair.flags & PxContactPairFlag::eINTERNAL_HAS_IMPULSES);


		invMassScale0 = iter.getInvMassScale0();
		invMassScale1 = iter.getInvMassScale1();
		invInertiaScale0 = iter.getInvInertiaScale0();
		invInertiaScale1 = iter.getInvInertiaScale1();
		while(iter.hasNextPatch())
		{
			iter.nextPatch();
			while(iter.hasNextContact())
			{
				iter.nextContact();
				PxContactPairPoint& dst = userBuffer[nbContacts];
				dst.position = iter.getContactPoint();
				dst.separation = iter.getSeparation();
				dst.normal = iter.getContactNormal();
				if (!flippedContacts)
				{
					dst.internalFaceIndex0 = iter.getFaceIndex0();
					dst.internalFaceIndex1 = iter.getFaceIndex1();
				}
				else
				{
					dst.internalFaceIndex0 = iter.getFaceIndex1();
					dst.internalFaceIndex1 = iter.getFaceIndex0();
				}

				if (hasImpulses)
				{
					PxReal impulse = impulses[nbContacts];
					dst.impulse = dst.normal * impulse;
				}
				else
					dst.impulse = PxVec3(0.0f);
				++nbContacts;
				if(nbContacts == bufferSize)
					return nbContacts;
			}
		}
	}

	return nbContacts;
}

class ContactReportCallback: public PxSimulationEventCallback
{
	void onConstraintBreak(PxConstraintInfo* constraints, PxU32 count)	{ PX_UNUSED(constraints); PX_UNUSED(count); }
	void onWake(PxActor** actors, PxU32 count)							{ PX_UNUSED(actors); PX_UNUSED(count); }
	void onSleep(PxActor** actors, PxU32 count)							{ PX_UNUSED(actors); PX_UNUSED(count); }
	void onTrigger(PxTriggerPair* pairs, PxU32 count)					{ PX_UNUSED(pairs); PX_UNUSED(count); }
	void onAdvance(const PxRigidBody*const*, const PxTransform*, const PxU32) {}
	void onContact(const PxContactPairHeader& pairHeader, const PxContactPair* pairs, PxU32 nbPairs) 
	{
		if(gHeadlessOptions.headless)
		{
			++gMetrics.reportCallbackCount;
			const bool expectedPair = isExpectedHeadlessPair(
				pairHeader.actors[0], pairHeader.actors[1]);
			if(!expectedPair)
				++gMetrics.identityErrors;

			PxArray<PxContactPairPoint> points;
			for(PxU32 i = 0; i < nbPairs; ++i)
			{
				if(!pairs[i].contactCount)
					continue;
				points.resize(pairs[i].contactCount);
				PxReal invMassScale0 = 1.0f;
				PxReal invMassScale1 = 1.0f;
				PxReal invInertiaScale0 = 1.0f;
				PxReal invInertiaScale1 = 1.0f;
				const PxU32 extracted = isHeadlessCase("finite-scales") ?
					extractContactsWithMassScale(
						pairs[i], points.begin(), points.size(),
						invMassScale0, invMassScale1,
						invInertiaScale0, invInertiaScale1) :
					pairs[i].extractContacts(points.begin(), points.size());
				gMetrics.reportPointCount += extracted;
				for(PxU32 j = 0; j < extracted; ++j)
				{
					if(!points[j].position.isFinite() ||
						!points[j].normal.isFinite() ||
						!points[j].impulse.isFinite() ||
						!PxIsFinite(points[j].separation))
						++gMetrics.nonFinite;
					gMetrics.maxReportedImpulse = PxMax(
						gMetrics.maxReportedImpulse,
						points[j].impulse.magnitude());
					if(isHeadlessCase("finite-scales"))
					{
						for(PxU32 actorIndex = 0; actorIndex < 2;
							++actorIndex)
						{
							const PxRigidDynamic* dynamic =
								pairHeader.actors[actorIndex]->
									is<PxRigidDynamic>();
							if(!dynamic)
								continue;
							const PxVec3 signedImpulse = actorIndex == 0 ?
								points[j].impulse : -points[j].impulse;
							const PxReal invMassScale = actorIndex == 0 ?
								invMassScale0 : invMassScale1;
							const PxReal invInertiaScale = actorIndex == 0 ?
								invInertiaScale0 : invInertiaScale1;
							PxVec3 linearImpulse(0.0f);
							PxVec3 angularImpulse(0.0f);
							PxRigidBodyExt::computeLinearAngularImpulse(
								*dynamic, dynamic->getGlobalPose(),
								points[j].position, signedImpulse,
								invMassScale, invInertiaScale,
								linearImpulse, angularImpulse);
							PxVec3 deltaLinearVelocity(0.0f);
							PxVec3 deltaAngularVelocity(0.0f);
							PxRigidBodyExt::computeVelocityDeltaFromImpulse(
								*dynamic, linearImpulse, angularImpulse,
								deltaLinearVelocity, deltaAngularVelocity);
							const PxU32 physicalIndex =
								dynamic == gHeadlessBody0 ? 0u : 1u;
							gExpectedLinearVelocityDelta[physicalIndex] +=
								deltaLinearVelocity;
							gExpectedAngularVelocityDelta[physicalIndex] +=
								deltaAngularVelocity;
						}
					}
				}
			}
			return;
		}
		PX_UNUSED((pairHeader));
		PxArray<PxContactPairPoint> contactPoints;
	

		for(PxU32 i=0;i<nbPairs;i++)
		{
			PxU32 contactCount = pairs[i].contactCount;
			if(contactCount)
			{
				contactPoints.resize(contactCount);
				PxReal invMassScale[2];
				PxReal invInertiaScale[2];
				extractContactsWithMassScale(
					pairs[i], &contactPoints[0], contactCount,
					invMassScale[0], invMassScale[1],
					invInertiaScale[0], invInertiaScale[1]);

				for(PxU32 j=0;j<contactCount;j++)
				{
					gContactPositions.pushBack(contactPoints[j].position);
					//Push back reported contact impulses
					gContactImpulses.pushBack(contactPoints[j].impulse);

					//Compute the effective linear/angular impulses for each body.
					//Note that the local mass scaling permits separate scales for invMass and invInertia.
					for(PxU32 k = 0; k < 2; ++k)
					{
						const PxRigidDynamic* dynamic = pairHeader.actors[k]->is<PxRigidDynamic>();
						PxVec3 linImpulse(0.f), angImpulse(0.f);
						if(dynamic != NULL)
						{
							PxRigidBodyExt::computeLinearAngularImpulse(*dynamic, dynamic->getGlobalPose(), contactPoints[j].position, 
								k == 0 ? contactPoints[j].impulse : -contactPoints[j].impulse, invMassScale[k], invInertiaScale[k], linImpulse, angImpulse);
						}
						gContactLinearImpulses[k].pushBack(linImpulse);
						gContactAngularImpulses[k].pushBack(angImpulse);
					}
				}
			}
		}
	}
};

static ContactReportCallback gContactReportCallback;

static void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent)
{
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	for(PxU32 i=0; i<size;i++)
	{
		PxTransform localTm(PxVec3(0, PxReal(i*2+1), 0) * halfExtent);
		PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
		body->attachShape(*shape);
		PxRigidBodyExt::updateMassAndInertia(*body, (i+1)*(i+1)*(i+1)*10.0f);
		gScene->addActor(*body);
	}
	shape->release();
}

void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gHeadlessOptions.headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	
	if(gHeadlessOptions.headless)
		gDispatcher = PxDefaultCpuDispatcherCreate(
			gHeadlessOptions.dispatcherThreads);
	else
	{
		PxU32 numCores = SnippetUtils::getNbPhysicalCores();
		gDispatcher = PxDefaultCpuDispatcherCreate(
			numCores == 0 ? 0 : numCores - 1);
	}
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.gravity =
		gHeadlessOptions.headless && !isHeadlessCase("max-impulse") &&
			!isTargetOwnershipAuthorityCase() ?
			PxVec3(0.0f) : PxVec3(0, -9.81f, 0);
	sceneDesc.filterShader	= contactReportFilterShader;			
	sceneDesc.simulationEventCallback = &gContactReportCallback;	
	sceneDesc.contactModifyCallback = &gContactModifyCallback;
	sceneDesc.solverType = gHeadlessOptions.solverType;
	if(isRestitutionThresholdCase() || isTiltedRestitutionCase() ||
		isSpatialFiniteImpulseCase() || isRestitutionFrictionComponentCase())
		sceneDesc.bounceThresholdVelocity = 2.0f;
	gScene = gPhysics->createScene(sceneDesc);

	if(!gHeadlessOptions.headless)
	{
		PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
		if(pvdClient)
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
	}

	gMaterial = gHeadlessOptions.headless ?
		gPhysics->createMaterial(
			(isHeadlessCase("tangent-target") ||
			 targetOwnershipHasFriction()) ? 1.0f : 0.0f,
			(isHeadlessCase("tangent-target") ||
			 targetOwnershipHasFriction()) ? 1.0f : 0.0f,
			(isHeadlessCase("ownership-bounce") ||
			 isRestitutionThresholdCase() ||
			 isTiltedRestitutionCase() ||
			 isSpatialFiniteImpulseCase() ||
			 isRestitutionFrictionComponentCase()) ? 0.5f : 0.0f) :
		gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	if(gHeadlessOptions.headless)
	{
		if(isPassiveFrictionComponentCase())
		{
			const bool restitutionComponent =
				isRestitutionFrictionComponentCase();
			const PxQuat rotation = isPassiveFrictionComponentYawCase() ?
				PxQuat(PxPi * 0.25f, PxVec3(0.0f, 1.0f, 0.0f)) :
				PxQuat(PxIdentity);
			// Keep the intended ground--lower--upper contact component connected
			// for the full audit.  The broad square footprint is rotationally
			// symmetric about Y and prevents the legacy friction mismatch from
			// changing the fixture topology by simply throwing the upper body
			// onto the ground.
			const PxBoxGeometry boxGeometry(PxVec3(2.0f, 0.5f, 2.0f));
			gHeadlessGround =
				PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
			gHeadlessBody0 = PxCreateDynamic(
				*gPhysics, PxTransform(PxVec3(0.0f, 0.49f, 0.0f), rotation),
				boxGeometry, *gMaterial, 1.0f);
			gHeadlessBody1 = PxCreateDynamic(
				*gPhysics,
				PxTransform(
					PxVec3(
						0.0f, restitutionComponent ? 2.0f : 1.48f, 0.0f),
					rotation),
				boxGeometry, *gMaterial, 1.0f);
			if(isTargetOwnershipReverseCase())
			{
				gScene->addActor(*gHeadlessBody1);
				gScene->addActor(*gHeadlessBody0);
				gScene->addActor(*gHeadlessGround);
			}
			else
			{
				gScene->addActor(*gHeadlessGround);
				gScene->addActor(*gHeadlessBody0);
				gScene->addActor(*gHeadlessBody1);
			}
			gHeadlessBody0->setLinearVelocity(
				PxVec3(3.0f, 0.0f, 0.0f));
			gHeadlessBody1->setLinearVelocity(
				PxVec3(3.0f, restitutionComponent ? -4.0f : 0.0f, 0.0f));
		}
		else if(isTargetOwnershipAuthorityCase())
		{
			const PxTransform targetPose(
				PxVec3(0.0f, 0.45f, 0.0f),
				isTargetOwnershipManifoldYawCase() ?
					PxQuat(PxPi, PxVec3(0.0f, 1.0f, 0.0f)) :
					PxQuat(PxIdentity));
			if(isTargetOwnershipReverseCase())
			{
				gHeadlessBody0 = isTargetOwnershipManifoldCase() ?
					PxCreateDynamic(
						*gPhysics, targetPose,
						PxBoxGeometry(PxVec3(0.5f)), *gMaterial, 1.0f) :
					PxCreateDynamic(
						*gPhysics, targetPose,
						PxSphereGeometry(0.5f), *gMaterial, 1.0f);
				gScene->addActor(*gHeadlessBody0);
				gHeadlessGround =
					PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
				gScene->addActor(*gHeadlessGround);
			}
			else
			{
				gHeadlessGround =
					PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
				gScene->addActor(*gHeadlessGround);
				gHeadlessBody0 = isTargetOwnershipManifoldCase() ?
					PxCreateDynamic(
						*gPhysics, targetPose,
						PxBoxGeometry(PxVec3(0.5f)), *gMaterial, 1.0f) :
					PxCreateDynamic(
						*gPhysics, targetPose,
						PxSphereGeometry(0.5f), *gMaterial, 1.0f);
				gScene->addActor(*gHeadlessBody0);
			}
		}
		else if(isDynamicPairCase())
		{
			if(isHeadlessCase("finite-scales-control") ||
				isHeadlessCase("finite-scales"))
			{
				gHeadlessBody0 = PxCreateDynamic(
					*gPhysics,
					PxTransform(PxVec3(-3.0f, 0.75f, 0.0f)),
					PxBoxGeometry(0.5f, 0.5f, 0.5f), *gMaterial, 1.0f);
				gHeadlessBody1 = PxCreateDynamic(
					*gPhysics, PxTransform(PxVec3(0.0f)),
					PxBoxGeometry(0.5f, 0.5f, 0.5f), *gMaterial, 1.0f);
				gHeadlessBody0->setLinearVelocity(
					PxVec3(10.0f, 0.0f, 0.0f));
			}
			else
			{
				const PxReal initialX =
					isHeadlessCase("tangent-target") ? -2.0f : -3.0f;
				const PxReal initialSpeed =
					isHeadlessCase("tangent-target") ? 5.0f : 10.0f;
				gHeadlessBody0 = PxCreateDynamic(
					*gPhysics, PxTransform(PxVec3(initialX, 0.0f, 0.0f)),
					PxSphereGeometry(0.5f), *gMaterial, 1.0f);
				gHeadlessBody1 = PxCreateDynamic(
					*gPhysics, PxTransform(PxVec3(0.0f)),
					PxSphereGeometry(0.5f), *gMaterial, 1.0f);
				gHeadlessBody0->setLinearVelocity(
					PxVec3(initialSpeed, 0.0f, 0.0f));
			}
			gScene->addActor(*gHeadlessBody0);
			gScene->addActor(*gHeadlessBody1);
		}
		else
		{
			gHeadlessGround = isOffCenterFiniteImpulseCase() ?
				PxCreateStatic(
					*gPhysics, PxTransform(PxVec3(-0.6f, 0.0f, 0.0f)),
					PxSphereGeometry(0.25f), *gMaterial) :
				PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
			gScene->addActor(*gHeadlessGround);
			const PxReal initialY =
				isHeadlessCase("target-velocity") ? 0.45f :
				isHeadlessCase("ownership-shallow") ? 0.49f :
				isHeadlessCase("ownership-deep") ? 0.30f :
				isHeadlessCase("ownership-deep-tilted") ? 0.25f :
				isTiltedRestitutionCase() ? 0.90f :
				isTiltedFiniteImpulseCase() ? 0.90f :
				isOffCenterFiniteImpulseCase() ? 1.35f :
				isRestitutionThresholdCase() ? 0.55f :
				(isHeadlessCase("normal") ||
				 isHeadlessCase("ownership-bounce") ||
				 isHeadlessCase("finite-max-impulse-control") ||
				 isHeadlessCase("finite-max-impulse") ? 2.0f : 3.0f);
			if(isHeadlessCase("ownership-deep-tilted") ||
				isTiltedRestitutionCase() ||
				isTiltedFiniteImpulseCase())
			{
				const PxTransform pose(
					PxVec3(0.0f, initialY, 0.0f),
					PxQuat(PxPi * 0.125f, PxVec3(0.0f, 0.0f, 1.0f)));
				gHeadlessBody0 = PxCreateDynamic(
					*gPhysics, pose,
					PxBoxGeometry(PxVec3(1.0f, 0.5f, 0.5f)),
					*gMaterial, 1.0f);
			}
			else if(isOffCenterFiniteImpulseCase())
			{
				gHeadlessBody0 = PxCreateDynamic(
					*gPhysics, PxTransform(PxVec3(0.0f, initialY, 0.0f)),
					PxBoxGeometry(PxVec3(1.0f, 0.5f, 0.5f)),
					*gMaterial, 1.0f);
			}
			else
			{
				gHeadlessBody0 = PxCreateDynamic(
					*gPhysics, PxTransform(PxVec3(0.0f, initialY, 0.0f)),
					PxSphereGeometry(0.5f), *gMaterial, 1.0f);
			}
			if(isHeadlessCase("normal"))
				gHeadlessBody0->setLinearVelocity(
					PxVec3(0.0f, -8.0f, 0.0f));
			else if(isHeadlessCase("ownership-bounce"))
				gHeadlessBody0->setLinearVelocity(
					PxVec3(0.0f, -8.0f, 0.0f));
			else if(isHeadlessCase("restitution-threshold-below"))
				gHeadlessBody0->setLinearVelocity(
					PxVec3(0.0f, -1.0f, 0.0f));
			else if(isHeadlessCase("restitution-threshold-above"))
				gHeadlessBody0->setLinearVelocity(
					PxVec3(0.0f, -6.0f, 0.0f));
			else if(isTiltedRestitutionCase())
				gHeadlessBody0->setLinearVelocity(
					PxVec3(0.0f, -6.0f, 0.0f));
			else if(isTiltedFiniteImpulseCase())
				gHeadlessBody0->setLinearVelocity(
					PxVec3(0.0f, -6.0f, 0.0f));
			else if(isOffCenterFiniteImpulseCase())
				gHeadlessBody0->setLinearVelocity(
					PxVec3(0.0f, -6.0f, 0.0f));
			else if(isHeadlessCase("finite-max-impulse-control") ||
				isHeadlessCase("finite-max-impulse"))
				gHeadlessBody0->setLinearVelocity(
					PxVec3(0.0f, -10.0f, 0.0f));
			gScene->addActor(*gHeadlessBody0);
		}
		if(isTargetOwnershipAuthorityCase() &&
			targetOwnershipLocksAngularMotion())
		{
			gHeadlessBody0->setRigidDynamicLockFlags(
				PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
				PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
				PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
		}
		if(isPassiveFrictionManifoldCase())
			gHeadlessBody0->setLinearVelocity(PxVec3(3.0f, 0.0f, 0.0f));
		gHeadlessBody0->setLinearDamping(0.0f);
		gHeadlessBody0->setAngularDamping(0.0f);
		gHeadlessBody0->setSleepThreshold(0.0f);
		if(gHeadlessBody1)
		{
			gHeadlessBody1->setLinearDamping(0.0f);
			gHeadlessBody1->setAngularDamping(0.0f);
			gHeadlessBody1->setSleepThreshold(0.0f);
		}
		gHeadlessInitialLinearVelocity[0] =
			gHeadlessBody0->getLinearVelocity();
		gHeadlessInitialAngularVelocity[0] =
			gHeadlessBody0->getAngularVelocity();
		gMetrics.body0Mass = gHeadlessBody0->getMass();
		gMetrics.body0InertiaZ =
			gHeadlessBody0->getMassSpaceInertiaTensor().z;
		if(gHeadlessBody1)
		{
			gHeadlessInitialLinearVelocity[1] =
				gHeadlessBody1->getLinearVelocity();
			gHeadlessInitialAngularVelocity[1] =
				gHeadlessBody1->getAngularVelocity();
		}
	}
	else
	{
		PxRigidStatic* groundPlane =
			PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
		gScene->addActor(*groundPlane);
		createStack(PxTransform(PxVec3(0,0.0f,10.0f)), 5, 2.0f);
	}
}

void stepPhysics(bool /*interactive*/)
{
	gContactPositions.clear();
	gContactImpulses.clear();

	const PxReal dt = gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f/60.0f;
	gScene->simulate(dt);
	const bool fetched = gScene->fetchResults(true);
	if(gHeadlessOptions.headless)
	{
		if(!fetched)
			++gMetrics.fetchFailures;
		if(gHeadlessBody0)
		{
			const PxTransform pose = gHeadlessBody0->getGlobalPose();
			const PxVec3 velocity = gHeadlessBody0->getLinearVelocity();
			const PxVec3 angularVelocity =
				gHeadlessBody0->getAngularVelocity();
			if(!pose.isFinite() || !velocity.isFinite() ||
				!angularVelocity.isFinite())
				++gMetrics.nonFinite;
			gMetrics.maxAbsBody0X =
				PxMax(gMetrics.maxAbsBody0X, PxAbs(pose.p.x));
			gMetrics.peakAbsBody0VelocityX =
				PxMax(gMetrics.peakAbsBody0VelocityX, PxAbs(velocity.x));
			gMetrics.peakAbsBody0VelocityY =
				PxMax(gMetrics.peakAbsBody0VelocityY, PxAbs(velocity.y));
			const PxReal angularSpeed = angularVelocity.magnitude();
			if(angularSpeed > gMetrics.peakBody0AngularSpeed)
			{
				gMetrics.peakBody0AngularSpeed = angularSpeed;
				gMetrics.peakBody0AngularFrame = gMetrics.completedFrames;
			}
			gMetrics.maxBody0Y = PxMax(gMetrics.maxBody0Y, pose.p.y);
			gMetrics.peakBody0VelocityY =
				PxMax(gMetrics.peakBody0VelocityY, velocity.y);
			gMetrics.minBody0Y = PxMin(gMetrics.minBody0Y, pose.p.y);
			gMetrics.finalBody0Speed = velocity.magnitude();
			gMetrics.finalBody0AngularSpeed =
				angularVelocity.magnitude();
			gMetrics.finalBody0VelocityX = velocity.x;
			gMetrics.finalBody0AngularVelocity = angularVelocity;
			if(isTargetOwnershipAuthorityCase())
			{
				const PxVec3 bottomOffset(0.0f, -0.5f, 0.0f);
				gMetrics.finalBody0ContactVelocityX =
					(velocity + angularVelocity.cross(bottomOffset)).x;
			}
			if(isOffCenterFiniteImpulseCase() &&
				gMetrics.completedFrames == 6)
			{
				gMetrics.finiteCheckpointVelocityY = velocity.y;
				gMetrics.finiteCheckpointAngularSpeed = angularSpeed;
			}
			if(isSpatialFiniteImpulseCase() &&
				gMetrics.completedFrames <= 16)
			{
				std::printf(
					"[FINITE_SPATIAL_FRAME] frame=%u y=%.9g vy=%.9g "
					"angular=%.9g\n",
					gMetrics.completedFrames, double(pose.p.y),
					double(velocity.y), double(angularSpeed));
			}
			if(gMetrics.modifyCallbackCount > 0)
				gMetrics.minBody0SpeedAfterModify = PxMin(
					gMetrics.minBody0SpeedAfterModify,
					velocity.magnitude());
		}
		if(gHeadlessBody1)
		{
			const PxTransform pose = gHeadlessBody1->getGlobalPose();
			const PxVec3 velocity = gHeadlessBody1->getLinearVelocity();
			const PxVec3 angularVelocity =
				gHeadlessBody1->getAngularVelocity();
			if(!pose.isFinite() || !velocity.isFinite() ||
				!angularVelocity.isFinite())
				++gMetrics.nonFinite;
			gMetrics.peakBody1Speed =
				PxMax(gMetrics.peakBody1Speed, velocity.magnitude());
			gMetrics.peakAbsBody1VelocityY =
				PxMax(gMetrics.peakAbsBody1VelocityY, PxAbs(velocity.y));
			gMetrics.peakBody1VelocityY =
				PxMax(gMetrics.peakBody1VelocityY, velocity.y);
			gMetrics.minBody1VelocityY =
				PxMin(gMetrics.minBody1VelocityY, velocity.y);
			gMetrics.peakBody1AngularSpeed = PxMax(
				gMetrics.peakBody1AngularSpeed,
				angularVelocity.magnitude());
			gMetrics.finalBody1Speed = velocity.magnitude();
			gMetrics.finalBody1AngularSpeed =
				angularVelocity.magnitude();
			gMetrics.finalBody1VelocityX = velocity.x;
			gMetrics.finalBody1AngularVelocity = angularVelocity;
			gMetrics.minBody1Y = PxMin(gMetrics.minBody1Y, pose.p.y);
			gMetrics.maxBody1Y = PxMax(gMetrics.maxBody1Y, pose.p.y);
			if(gHeadlessBody0)
			{
				const PxReal relativeVelocityY =
					gHeadlessBody0->getLinearVelocity().y - velocity.y;
				gMetrics.peakBody0MinusBody1VelocityY = PxMax(
					gMetrics.peakBody0MinusBody1VelocityY,
					relativeVelocityY);
			}
		}
		++gMetrics.completedFrames;
	}
	else
		printf("%u contact reports\n", PxU32(gContactPositions.size()));
}
	
void cleanupPhysics(bool /*interactive*/)
{
    gContactPositions.reset();
    gContactImpulses.reset();
    gContactLinearImpulses[0].reset();
    gContactAngularImpulses[0].reset();
    gContactLinearImpulses[1].reset();
    gContactAngularImpulses[1].reset();
    
	PX_RELEASE(gScene);
	gHeadlessBody0 = NULL;
	gHeadlessBody1 = NULL;
	gHeadlessGround = NULL;
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gScene && !gMaterial && !gDispatcher && !gPhysics && !gPvd &&
		!gFoundation && !gExtensionsInitialized ? 1u : 0u;
	
	printf("SnippetContactModification done.\n");
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig(
		"SnippetContactModification", gHeadlessOptions);

	initPhysics(false);
	const bool massScaleCase = isHeadlessCase("mass-scale");
	const bool dynamicPairCase = isDynamicPairCase();
	const bool passiveFrictionComponentCase =
		isPassiveFrictionComponentCase();
	const bool initialized =
		gFoundation && gPhysics && gExtensionsInitialized && gDispatcher &&
		gScene && gMaterial && gHeadlessBody0 &&
		(dynamicPairCase ? (gHeadlessBody1 != NULL) :
			(passiveFrictionComponentCase ?
				(gHeadlessBody1 != NULL && gHeadlessGround != NULL) :
				(gHeadlessGround != NULL)));
	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics(false);
	}
	if(initialized && (isHeadlessCase("finite-scales-control") ||
		isHeadlessCase("finite-scales")))
	{
		const PxVec3 actualLinearDelta0 =
			gHeadlessBody0->getLinearVelocity() -
			gHeadlessInitialLinearVelocity[0];
		const PxVec3 actualLinearDelta1 =
			gHeadlessBody1->getLinearVelocity() -
			gHeadlessInitialLinearVelocity[1];
		const PxVec3 actualAngularDelta0 =
			gHeadlessBody0->getAngularVelocity() -
			gHeadlessInitialAngularVelocity[0];
		const PxVec3 actualAngularDelta1 =
			gHeadlessBody1->getAngularVelocity() -
			gHeadlessInitialAngularVelocity[1];
		gMetrics.actualBody0AngularDelta = actualAngularDelta0.magnitude();
		gMetrics.actualBody1AngularDelta = actualAngularDelta1.magnitude();
		if(isHeadlessCase("finite-scales"))
		{
			gMetrics.expectedBody0LinearDelta =
				gExpectedLinearVelocityDelta[0].magnitude();
			gMetrics.expectedBody1LinearDelta =
				gExpectedLinearVelocityDelta[1].magnitude();
			gMetrics.expectedBody0AngularDelta =
				gExpectedAngularVelocityDelta[0].magnitude();
			gMetrics.expectedBody1AngularDelta =
				gExpectedAngularVelocityDelta[1].magnitude();
			gMetrics.scaleLinearVelocityResidual = PxMax(
				(actualLinearDelta0 -
				 gExpectedLinearVelocityDelta[0]).magnitude(),
				(actualLinearDelta1 -
				 gExpectedLinearVelocityDelta[1]).magnitude());
			gMetrics.scaleAngularVelocityResidual = PxMax(
				(actualAngularDelta0 -
				 gExpectedAngularVelocityDelta[0]).magnitude(),
				(actualAngularDelta1 -
				 gExpectedAngularVelocityDelta[1]).magnitude());
		}
	}

	const char* reason = "none";
	bool passed = true;
	if(!initialized)
	{
		passed = false;
		reason = "initialization_failed";
	}
	else if(gMetrics.completedFrames != gHeadlessOptions.frames ||
		gMetrics.fetchFailures != 0)
	{
		passed = false;
		reason = "incomplete_simulation";
	}
	else if(gMetrics.nonFinite != 0 ||
		gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.modifyCallbackCount == 0 ||
		gMetrics.modifiedPairCount == 0 ||
		gMetrics.modifiedPointCount == 0)
	{
		passed = false;
		reason = "missing_modify_callback";
	}
	else if(gMetrics.reportCallbackCount == 0 ||
		(!isHeadlessCase("max-impulse") &&
		 gMetrics.reportPointCount == 0))
	{
		passed = false;
		reason = "missing_contact_report";
	}
	else if(gMetrics.identityErrors != 0 ||
		gMetrics.scaleReadbackErrors != 0 ||
		gMetrics.maxImpulseReadbackErrors != 0 ||
		gMetrics.targetVelocityReadbackErrors != 0)
	{
		passed = false;
		reason = "callback_payload_error";
	}
	else if(isTargetOwnershipAuthorityCase() &&
		targetOwnershipHasNormalTarget() &&
		gMetrics.peakBody0VelocityY < 0.25f)
	{
		passed = false;
		reason = "ownership_normal_target_missing";
	}
	else if(isTargetOwnershipAuthorityCase() &&
		targetOwnershipHasTangentTarget() &&
		!targetOwnershipHasFriction() &&
		gMetrics.peakAbsBody0VelocityX > 0.05f)
	{
		passed = false;
		reason = "ownership_zero_friction_leaked_tangent_target";
	}
	else if(isTargetOwnershipAuthorityCase() &&
		targetOwnershipHasTangentTarget() &&
		targetOwnershipHasFriction() &&
		gMetrics.peakAbsBody0VelocityX < 0.25f)
	{
		passed = false;
		reason = "ownership_friction_tangent_target_missing";
	}
	else if(targetOwnershipHasFiniteCap() &&
		(gMetrics.maxReportedImpulse <= 0.0f ||
		 gMetrics.maxReportedImpulse > 0.36f))
	{
		passed = false;
		reason = "ownership_finite_cap_not_consumed";
	}
	else if(isHeadlessCase("normal") &&
		(gMetrics.peakAbsBody0VelocityX < 0.5f ||
		 gMetrics.maxAbsBody0X < 0.25f))
	{
		passed = false;
		reason = "modified_normal_not_consumed";
	}
	else if(isHeadlessCase("target-velocity") &&
		(gMetrics.peakBody0VelocityY < 1.0f ||
		 gMetrics.maxBody0Y < 1.0f))
	{
		passed = false;
		reason = "target_velocity_not_consumed";
	}
	else if(isHeadlessCase("max-impulse") &&
		gMetrics.minBody0Y > -1.0f)
	{
		passed = false;
		reason = "max_impulse_not_consumed";
	}
	else if(isHeadlessCase("ownership-shallow") &&
		(gMetrics.maxBody0Y < 0.49f ||
		 gMetrics.minBody0Y < 0.20f ||
		 gMetrics.peakAbsBody0VelocityY > 20.0f))
	{
		passed = false;
		reason = "shallow_overlap_unstable";
	}
	else if(isHeadlessCase("ownership-deep") &&
		(gMetrics.maxBody0Y < 0.45f ||
		 gMetrics.minBody0Y < 0.20f ||
		 gMetrics.peakAbsBody0VelocityY > 20.0f))
	{
		passed = false;
		reason = "deep_overlap_unstable";
	}
	else if(isHeadlessCase("ownership-deep-tilted") &&
		(gMetrics.maxBody0Y < 0.45f ||
		 gMetrics.minBody0Y < 0.10f ||
		 gMetrics.peakAbsBody0VelocityY > 40.0f ||
		 gMetrics.peakBody0AngularSpeed > 100.0f))
	{
		passed = false;
		reason = "deep_tilted_overlap_unstable";
	}
	else if(isHeadlessCase("ownership-bounce") &&
		(gMetrics.peakBody0VelocityY < 2.0f ||
		 gMetrics.maxBody0Y < 2.0f))
	{
		passed = false;
		reason = "material_bounce_missing";
	}
	else if(isHeadlessCase("restitution-threshold-below") &&
		(gMetrics.peakBody0VelocityY > 0.25f ||
		 gMetrics.maxBody0Y > 0.60f))
	{
		passed = false;
		reason = "below_threshold_restitution_applied";
	}
	else if(isHeadlessCase("restitution-threshold-above") &&
		(gMetrics.peakBody0VelocityY < 2.5f ||
		 gMetrics.maxBody0Y < 2.0f))
	{
		passed = false;
		reason = "above_threshold_restitution_missing";
	}
	else if(isTiltedRestitutionCase() &&
		(gMetrics.peakBody0VelocityY < 0.25f ||
		 gMetrics.peakBody0AngularSpeed < 0.05f ||
		 gMetrics.peakBody0AngularSpeed > 100.0f))
	{
		passed = false;
		reason = "tilted_restitution_not_exercised";
	}
	else if(isHeadlessCase("finite-max-impulse-control") &&
		(gMetrics.maxReportedImpulse < 1.0f ||
		 gMetrics.minBody0Y < 0.45f))
	{
		passed = false;
		reason = "finite_impulse_control_not_stopped";
	}
	else if(isHeadlessCase("finite-max-impulse") &&
		(gMetrics.maxReportedImpulse < 0.20f ||
		 gMetrics.maxReportedImpulse > 0.251f ||
		 gMetrics.minBody0Y > -1.0f))
	{
		passed = false;
		reason = "finite_max_impulse_not_consumed";
	}
	else if(isTiltedFiniteImpulseCase() &&
		(gMetrics.maxReportedImpulse < 0.75f ||
		 gMetrics.maxReportedImpulse > 1.001f ||
		 gMetrics.peakBody0AngularSpeed < 0.05f ||
		 gMetrics.peakBody0AngularSpeed > 100.0f))
	{
		passed = false;
		reason = "tilted_finite_impulse_not_exercised";
	}
	else if(isOffCenterFiniteImpulseCase() &&
		(gMetrics.maxReportedImpulse < 0.75f ||
		 gMetrics.maxReportedImpulse > 1.001f ||
		 gMetrics.finiteCheckpointVelocityY < -5.52f ||
		 gMetrics.finiteCheckpointVelocityY > -5.48f ||
		 gMetrics.finiteCheckpointAngularSpeed < 0.70f ||
		 gMetrics.finiteCheckpointAngularSpeed > 0.74f ||
		 gMetrics.peakBody0AngularSpeed < 0.05f ||
		 gMetrics.peakBody0AngularSpeed > 100.0f))
	{
		passed = false;
		reason = "offcenter_finite_impulse_not_exercised";
	}
	else if(massScaleCase &&
		(gMetrics.peakBody1Speed > 0.5f ||
		 gMetrics.minBody0SpeedAfterModify > 2.0f))
	{
		passed = false;
		reason = "mass_scale_not_consumed";
	}
	else if(isHeadlessCase("finite-scales") &&
		(gMetrics.expectedBody0LinearDelta < 0.1f ||
		 gMetrics.expectedBody1LinearDelta < 0.1f ||
		 gMetrics.expectedBody0AngularDelta < 0.01f ||
		 gMetrics.expectedBody1AngularDelta < 0.01f ||
		 gMetrics.actualBody0AngularDelta < 0.1f ||
		 gMetrics.actualBody1AngularDelta < 0.1f ||
		 gMetrics.scaleLinearVelocityResidual > 0.1f))
	{
		passed = false;
		reason = "finite_scales_not_consumed";
	}
	else if(isHeadlessCase("tangent-target") &&
		(gMetrics.peakAbsBody0VelocityY < 0.1f ||
		 gMetrics.peakAbsBody1VelocityY < 0.1f ||
		 gMetrics.peakBody0MinusBody1VelocityY < 0.5f))
	{
		passed = false;
		reason = "tangent_target_not_consumed";
	}
	else if(isRestitutionFrictionComponentCase() &&
		(gMetrics.minBody1VelocityY > -3.0f ||
		 gMetrics.peakBody1VelocityY < 1.0f ||
		 gMetrics.minBody0Y < 0.20f ||
		 gMetrics.minBody1Y < 0.80f ||
		 gMetrics.peakBody0AngularSpeed > 50.0f ||
		 gMetrics.peakBody1AngularSpeed > 50.0f))
	{
		passed = false;
		reason = "restitution_friction_component_not_exercised";
	}
	else if(passiveFrictionComponentCase &&
		(gMetrics.minBody0Y < 0.20f ||
		 gMetrics.minBody1Y < 0.80f ||
		 gMetrics.peakBody0AngularSpeed > 50.0f ||
		 gMetrics.peakBody1AngularSpeed > 50.0f ||
		 gMetrics.finalBody0Speed > 10.0f ||
		 gMetrics.finalBody1Speed > 10.0f))
	{
		passed = false;
		reason = "passive_friction_component_unstable";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}

	std::printf(
		"[AVBD_GATE] schema=2 snippet=SnippetContactModification solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED modifyCallbackCount=%u "
		"modifiedPairCount=%u modifiedPointCount=%u "
		"reportCallbackCount=%u reportPointCount=%u identityErrors=%u "
		"body0Actor0Count=%u body0Actor1Count=%u "
		"scaleReadbackErrors=%u maxImpulseReadbackErrors=%u "
		"targetVelocityReadbackErrors=%u maxAbsBody0X=%.9g "
		"peakAbsBody0VelocityX=%.9g peakAbsBody0VelocityY=%.9g "
		"peakBody0AngularSpeed=%.9g peakBody0AngularFrame=%u "
		"maxBody0Y=%.9g "
		"peakBody0VelocityY=%.9g minBody0Y=%.9g peakBody1Speed=%.9g "
		"peakAbsBody1VelocityY=%.9g "
		"peakBody0MinusBody1VelocityY=%.9g maxReportedImpulse=%.9g "
		"minBody0SpeedAfterModify=%.9g finalBody0Speed=%.9g "
		"finalBody0AngularSpeed=%.9g finalBody0VelocityX=%.9g "
		"finalBody0AngularX=%.9g finalBody0AngularY=%.9g "
		"finalBody0AngularZ=%.9g finalBody0ContactVelocityX=%.9g "
		"finalBody1Speed=%.9g peakBody1AngularSpeed=%.9g "
		"finalBody1AngularSpeed=%.9g finalBody1VelocityX=%.9g "
		"finalBody1AngularX=%.9g finalBody1AngularY=%.9g "
		"finalBody1AngularZ=%.9g minBody1Y=%.9g "
		"maxBody1Y=%.9g peakBody1VelocityY=%.9g "
		"minBody1VelocityY=%.9g "
		"finiteCheckpointVelocityY=%.9g "
		"finiteCheckpointAngularSpeed=%.9g "
		"body0Mass=%.9g body0InertiaZ=%.9g "
		"expectedBody0LinearDelta=%.9g "
		"expectedBody1LinearDelta=%.9g expectedBody0AngularDelta=%.9g "
		"expectedBody1AngularDelta=%.9g "
		"actualBody0AngularDelta=%.9g actualBody1AngularDelta=%.9g "
		"scaleLinearVelocityResidual=%.9g "
		"scaleAngularVelocityResidual=%.9g "
		"nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason,
		gMetrics.modifyCallbackCount, gMetrics.modifiedPairCount,
		gMetrics.modifiedPointCount, gMetrics.reportCallbackCount,
		gMetrics.reportPointCount, gMetrics.identityErrors,
		gMetrics.body0Actor0Count, gMetrics.body0Actor1Count,
		gMetrics.scaleReadbackErrors, gMetrics.maxImpulseReadbackErrors,
		gMetrics.targetVelocityReadbackErrors,
		double(gMetrics.maxAbsBody0X),
		double(gMetrics.peakAbsBody0VelocityX),
		double(gMetrics.peakAbsBody0VelocityY),
		double(gMetrics.peakBody0AngularSpeed),
		gMetrics.peakBody0AngularFrame,
		double(gMetrics.maxBody0Y),
		double(gMetrics.peakBody0VelocityY),
		double(gMetrics.minBody0Y), double(gMetrics.peakBody1Speed),
		double(gMetrics.peakAbsBody1VelocityY),
		double(gMetrics.peakBody0MinusBody1VelocityY),
		double(gMetrics.maxReportedImpulse),
		double(gMetrics.minBody0SpeedAfterModify),
		double(gMetrics.finalBody0Speed),
		double(gMetrics.finalBody0AngularSpeed),
		double(gMetrics.finalBody0VelocityX),
		double(gMetrics.finalBody0AngularVelocity.x),
		double(gMetrics.finalBody0AngularVelocity.y),
		double(gMetrics.finalBody0AngularVelocity.z),
		double(gMetrics.finalBody0ContactVelocityX),
		double(gMetrics.finalBody1Speed),
		double(gMetrics.peakBody1AngularSpeed),
		double(gMetrics.finalBody1AngularSpeed),
		double(gMetrics.finalBody1VelocityX),
		double(gMetrics.finalBody1AngularVelocity.x),
		double(gMetrics.finalBody1AngularVelocity.y),
		double(gMetrics.finalBody1AngularVelocity.z),
		double(gMetrics.minBody1Y),
		double(gMetrics.maxBody1Y),
		double(gMetrics.peakBody1VelocityY),
		double(gMetrics.minBody1VelocityY),
		double(gMetrics.finiteCheckpointVelocityY),
		double(gMetrics.finiteCheckpointAngularSpeed),
		double(gMetrics.body0Mass), double(gMetrics.body0InertiaZ),
		double(gMetrics.expectedBody0LinearDelta),
		double(gMetrics.expectedBody1LinearDelta),
		double(gMetrics.expectedBody0AngularDelta),
		double(gMetrics.expectedBody1AngularDelta),
		double(gMetrics.actualBody0AngularDelta),
		double(gMetrics.actualBody1AngularDelta),
		double(gMetrics.scaleLinearVelocityResidual),
		double(gMetrics.scaleAngularVelocityResidual), gMetrics.nonFinite,
		gMetrics.fetchFailures, gErrorCallback.getFatalCount(),
		gMetrics.cleanupComplete);
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char*const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetContactModification "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetContactModification "
			"reason=execution_environment_failed\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();

#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	initPhysics(false);
	for(PxU32 i=0; i<250; i++)
		stepPhysics(false);
	cleanupPhysics(false);
#endif

	return 0;
}
