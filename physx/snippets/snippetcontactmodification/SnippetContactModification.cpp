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
	PxU32 nonFinite;
	PxReal maxAbsBody0X;
	PxReal peakAbsBody0VelocityX;
	PxReal maxBody0Y;
	PxReal peakBody0VelocityY;
	PxReal minBody0Y;
	PxReal peakBody1Speed;
	PxReal minBody0SpeedAfterModify;
	PxReal finalBody0Speed;
	PxReal finalBody1Speed;
	PxU32 cleanupComplete;

	ContactModificationMetrics()
	: completedFrames(0), fetchFailures(0), modifyCallbackCount(0),
	  modifiedPairCount(0), modifiedPointCount(0), reportCallbackCount(0),
	  reportPointCount(0), identityErrors(0), scaleReadbackErrors(0),
	  nonFinite(0), maxAbsBody0X(0.0f), peakAbsBody0VelocityX(0.0f),
	  maxBody0Y(-PX_MAX_F32), peakBody0VelocityY(-PX_MAX_F32),
	  minBody0Y(PX_MAX_F32), peakBody1Speed(0.0f),
	  minBody0SpeedAfterModify(PX_MAX_F32), finalBody0Speed(0.0f),
	  finalBody1Speed(0.0f), cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static ContactModificationMetrics gMetrics;

PxArray<PxVec3> gContactPositions;
PxArray<PxVec3> gContactImpulses;
PxArray<PxVec3> gContactLinearImpulses[2];
PxArray<PxVec3> gContactAngularImpulses[2];

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(), name);
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
		!isHeadlessCase("mass-scale"))
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
				const bool expectedPair = isHeadlessCase("mass-scale") ?
					((body0IsActor0 &&
					  pairs[i].actor[1] == gHeadlessBody1) ||
					 (body0IsActor1 &&
					  pairs[i].actor[0] == gHeadlessBody1)) :
					((body0IsActor0 &&
					  pairs[i].actor[1] == gHeadlessGround) ||
					 (body0IsActor1 &&
					  pairs[i].actor[0] == gHeadlessGround));
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
						pairs[i].contacts.setTargetVelocity(
							c, body0IsActor0 ? body0Target : -body0Target);
					}
					else if(isHeadlessCase("max-impulse"))
					{
						pairs[i].contacts.setMaxImpulse(c, 0.0f);
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

static PxU32 extractContactsWithMassScale(const PxContactPair& pair, PxContactPairPoint* userBuffer, PxU32 bufferSize, PxReal& invMassScale0, PxReal& invMassScale1)
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
			const bool body0IsActor0 =
				pairHeader.actors[0] == gHeadlessBody0;
			const bool body0IsActor1 =
				pairHeader.actors[1] == gHeadlessBody0;
			const bool expectedPair = isHeadlessCase("mass-scale") ?
				((body0IsActor0 &&
				  pairHeader.actors[1] == gHeadlessBody1) ||
				 (body0IsActor1 &&
				  pairHeader.actors[0] == gHeadlessBody1)) :
				((body0IsActor0 &&
				  pairHeader.actors[1] == gHeadlessGround) ||
				 (body0IsActor1 &&
				  pairHeader.actors[0] == gHeadlessGround));
			if(!expectedPair)
				++gMetrics.identityErrors;

			PxArray<PxContactPairPoint> points;
			for(PxU32 i = 0; i < nbPairs; ++i)
			{
				if(!pairs[i].contactCount)
					continue;
				points.resize(pairs[i].contactCount);
				const PxU32 extracted = pairs[i].extractContacts(
					points.begin(), points.size());
				gMetrics.reportPointCount += extracted;
				for(PxU32 j = 0; j < extracted; ++j)
				{
					if(!points[j].position.isFinite() ||
						!points[j].normal.isFinite() ||
						!points[j].impulse.isFinite() ||
						!PxIsFinite(points[j].separation))
						++gMetrics.nonFinite;
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
				extractContactsWithMassScale(pairs[i], &contactPoints[0], contactCount, invMassScale[0], invMassScale[1]);

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
								k == 0 ? contactPoints[j].impulse : -contactPoints[j].impulse, invMassScale[k], invMassScale[k], linImpulse, angImpulse);
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
		gHeadlessOptions.headless && !isHeadlessCase("max-impulse") ?
			PxVec3(0.0f) : PxVec3(0, -9.81f, 0);
	sceneDesc.filterShader	= contactReportFilterShader;			
	sceneDesc.simulationEventCallback = &gContactReportCallback;	
	sceneDesc.contactModifyCallback = &gContactModifyCallback;
	sceneDesc.solverType = gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);

	if(!gHeadlessOptions.headless)
	{
		PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
		if(pvdClient)
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
	}

	gMaterial = gHeadlessOptions.headless ?
		gPhysics->createMaterial(0.0f, 0.0f, 0.0f) :
		gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	if(gHeadlessOptions.headless)
	{
		if(isHeadlessCase("mass-scale"))
		{
			gHeadlessBody0 = PxCreateDynamic(
				*gPhysics, PxTransform(PxVec3(-3.0f, 0.0f, 0.0f)),
				PxSphereGeometry(0.5f), *gMaterial, 1.0f);
			gHeadlessBody1 = PxCreateDynamic(
				*gPhysics, PxTransform(PxVec3(0.0f, 0.0f, 0.0f)),
				PxSphereGeometry(0.5f), *gMaterial, 1.0f);
			gHeadlessBody0->setLinearVelocity(PxVec3(10.0f, 0.0f, 0.0f));
			gScene->addActor(*gHeadlessBody0);
			gScene->addActor(*gHeadlessBody1);
		}
		else
		{
			gHeadlessGround =
				PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
			gScene->addActor(*gHeadlessGround);
			const PxReal initialY = isHeadlessCase("target-velocity") ?
				0.45f : (isHeadlessCase("normal") ? 2.0f : 3.0f);
			gHeadlessBody0 = PxCreateDynamic(
				*gPhysics, PxTransform(PxVec3(0.0f, initialY, 0.0f)),
				PxSphereGeometry(0.5f), *gMaterial, 1.0f);
			if(isHeadlessCase("normal"))
				gHeadlessBody0->setLinearVelocity(
					PxVec3(0.0f, -8.0f, 0.0f));
			gScene->addActor(*gHeadlessBody0);
		}
		gHeadlessBody0->setLinearDamping(0.0f);
		gHeadlessBody0->setAngularDamping(0.0f);
		gHeadlessBody0->setSleepThreshold(0.0f);
		if(gHeadlessBody1)
		{
			gHeadlessBody1->setLinearDamping(0.0f);
			gHeadlessBody1->setAngularDamping(0.0f);
			gHeadlessBody1->setSleepThreshold(0.0f);
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
			if(!pose.isFinite() || !velocity.isFinite())
				++gMetrics.nonFinite;
			gMetrics.maxAbsBody0X =
				PxMax(gMetrics.maxAbsBody0X, PxAbs(pose.p.x));
			gMetrics.peakAbsBody0VelocityX =
				PxMax(gMetrics.peakAbsBody0VelocityX, PxAbs(velocity.x));
			gMetrics.maxBody0Y = PxMax(gMetrics.maxBody0Y, pose.p.y);
			gMetrics.peakBody0VelocityY =
				PxMax(gMetrics.peakBody0VelocityY, velocity.y);
			gMetrics.minBody0Y = PxMin(gMetrics.minBody0Y, pose.p.y);
			gMetrics.finalBody0Speed = velocity.magnitude();
			if(gMetrics.modifyCallbackCount > 0)
				gMetrics.minBody0SpeedAfterModify = PxMin(
					gMetrics.minBody0SpeedAfterModify,
					velocity.magnitude());
		}
		if(gHeadlessBody1)
		{
			const PxTransform pose = gHeadlessBody1->getGlobalPose();
			const PxVec3 velocity = gHeadlessBody1->getLinearVelocity();
			if(!pose.isFinite() || !velocity.isFinite())
				++gMetrics.nonFinite;
			gMetrics.peakBody1Speed =
				PxMax(gMetrics.peakBody1Speed, velocity.magnitude());
			gMetrics.finalBody1Speed = velocity.magnitude();
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
	const bool initialized =
		gFoundation && gPhysics && gExtensionsInitialized && gDispatcher &&
		gScene && gMaterial && gHeadlessBody0 &&
		(massScaleCase ? (gHeadlessBody1 != NULL) :
			(gHeadlessGround != NULL));
	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics(false);
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
		gMetrics.scaleReadbackErrors != 0)
	{
		passed = false;
		reason = "callback_payload_error";
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
	else if(massScaleCase &&
		(gMetrics.peakBody1Speed > 0.5f ||
		 gMetrics.minBody0SpeedAfterModify > 2.0f))
	{
		passed = false;
		reason = "mass_scale_not_consumed";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}

	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetContactModification solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED modifyCallbackCount=%u "
		"modifiedPairCount=%u modifiedPointCount=%u "
		"reportCallbackCount=%u reportPointCount=%u identityErrors=%u "
		"scaleReadbackErrors=%u maxAbsBody0X=%.9g "
		"peakAbsBody0VelocityX=%.9g maxBody0Y=%.9g "
		"peakBody0VelocityY=%.9g minBody0Y=%.9g peakBody1Speed=%.9g "
		"minBody0SpeedAfterModify=%.9g finalBody0Speed=%.9g "
		"finalBody1Speed=%.9g nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason,
		gMetrics.modifyCallbackCount, gMetrics.modifiedPairCount,
		gMetrics.modifiedPointCount, gMetrics.reportCallbackCount,
		gMetrics.reportPointCount, gMetrics.identityErrors,
		gMetrics.scaleReadbackErrors, double(gMetrics.maxAbsBody0X),
		double(gMetrics.peakAbsBody0VelocityX),
		double(gMetrics.maxBody0Y),
		double(gMetrics.peakBody0VelocityY),
		double(gMetrics.minBody0Y), double(gMetrics.peakBody1Speed),
		double(gMetrics.minBody0SpeedAfterModify),
		double(gMetrics.finalBody0Speed),
		double(gMetrics.finalBody1Speed), gMetrics.nonFinite,
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
