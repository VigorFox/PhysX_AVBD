// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_ANALYTIC_CONTACT_KERNELS_H
#define SC_AVBD_ANALYTIC_CONTACT_KERNELS_H

#include "foundation/PxArray.h"

namespace physx
{
namespace Dy
{
struct AvbdRigidBox;
struct AvbdRigidCapsule;
struct AvbdRigidConvex;
struct AvbdRigidSphere;
struct AvbdRigidTriangleSurface;
struct AvbdRigidTriangleSurfaceDiscreteOGCQueryStats;
struct AvbdRigidTriangleSurfaceFeaturePlan;
struct AvbdRigidTriangleSurfaceFeaturePlanRangeTiming;
struct AvbdRigidTriangleSurfaceForwardOwnerQueryStats;
struct AvbdRigidTriangleSurfaceForwardOwnerResultCache;
struct AvbdRigidTriangleSurfaceQueryScratch;
struct AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming;
struct AvbdSoftBody;
struct AvbdSoftCollisionStats;
struct AvbdSoftContact;
struct AvbdSoftContactWorkspace;
struct AvbdSoftSoftPairQueryScratch;
struct AvbdSoftParticle;
typedef PxArray<PxArray<PxU32> > AvbdSelfCollisionAdjacency;
struct AvbdOGCParams;
struct AvbdWorldPlane;
}

namespace Sc
{

struct AvbdWorldPlaneContactRange
{
	const Dy::AvbdSoftParticle*	particles;
	PxU32					numParticles;
	PxU32					particleBegin;
	PxU32					particleEnd;
	const Dy::AvbdWorldPlane*	planes;
	PxU32					numPlanes;
	const Dy::AvbdSoftBody*		bodies;
	PxU32					numBodies;
	PxArray<Dy::AvbdSoftContact>* contacts;
	PxReal					margin;
};

struct AvbdRigidBoxSdfContactRange
{
	const Dy::AvbdSoftParticle*	particles;
	PxU32					numParticles;
	PxU32					particleBegin;
	PxU32					particleEnd;
	const Dy::AvbdRigidBox*		boxes;
	PxU32					numBoxes;
	const Dy::AvbdSoftContact*	previousContacts;
	PxU32					numPreviousContacts;
	const Dy::AvbdSoftBody*		bodies;
	PxU32					numBodies;
	PxArray<Dy::AvbdSoftContact>* contacts;
	PxArray<Dy::AvbdSoftContact>* sweptContacts;
	PxReal					margin;
};

struct AvbdRigidSphereSdfContactRange
{
	const Dy::AvbdSoftParticle*	particles;
	PxU32					numParticles;
	PxU32					particleBegin;
	PxU32					particleEnd;
	const Dy::AvbdRigidSphere*	spheres;
	PxU32					numSpheres;
	const Dy::AvbdSoftBody*		bodies;
	PxU32					numBodies;
	PxArray<Dy::AvbdSoftContact>* contacts;
	PxArray<Dy::AvbdSoftContact>* sweptContacts;
	PxReal					margin;
};

struct AvbdRigidCapsuleSdfContactRange
{
	const Dy::AvbdSoftParticle*	particles;
	PxU32					numParticles;
	PxU32					particleBegin;
	PxU32					particleEnd;
	const Dy::AvbdRigidCapsule*	capsules;
	PxU32					numCapsules;
	const Dy::AvbdSoftBody*		bodies;
	PxU32					numBodies;
	PxArray<Dy::AvbdSoftContact>* contacts;
	PxArray<Dy::AvbdSoftContact>* sweptContacts;
	PxReal					margin;
};

struct AvbdRigidConvexSdfContactRange
{
	const Dy::AvbdSoftParticle*	particles;
	PxU32					numParticles;
	PxU32					particleBegin;
	PxU32					particleEnd;
	const Dy::AvbdRigidConvex*	convexes;
	PxU32					numConvexes;
	const Dy::AvbdSoftBody*		bodies;
	PxU32					numBodies;
	PxArray<Dy::AvbdSoftContact>* contacts;
	PxArray<Dy::AvbdSoftContact>* sweptContacts;
	PxReal					margin;
};

struct AvbdRigidTriangleSurfaceContactRange
{
	const Dy::AvbdSoftParticle*	particles;
	PxU32					numParticles;
	PxU32					particleBegin;
	PxU32					particleEnd;
	const Dy::AvbdRigidTriangleSurface* surfaces;
	PxU32					numSurfaces;
	const Dy::AvbdSoftBody*		bodies;
	PxU32					numBodies;
	PxArray<Dy::AvbdSoftContact>* contacts;
	PxArray<Dy::AvbdSoftContact>* sweptContacts;
	const Dy::AvbdRigidTriangleSurfaceFeaturePlan* featurePlan;
	PxU32					featurePlanBegin;
	PxU32					featurePlanEnd;
	PxArray<Dy::AvbdSoftContact>* featureContacts;
	PxArray<PxArray<Dy::AvbdSoftContact> >* featurePlanOutputs;
	bool					featurePlanRowPrivateOutputs;
	bool					featurePlanRoundRobin;
	PxU32					featurePlanTaskIndex;
	PxU32					featurePlanTaskCount;
	Dy::AvbdRigidTriangleSurfaceQueryScratch* queryScratch;
	Dy::AvbdRigidTriangleSurfaceForwardOwnerQueryStats*
						forwardOwnerQueryStats;
	Dy::AvbdRigidTriangleSurfaceForwardOwnerResultCache*
						forwardOwnerResultCache;
	bool					discreteQueryStatsEnabled;
	bool					discreteBodyLocalBoundsCullEnabled;
	PxReal					margin;
	Dy::AvbdSoftCollisionStats*	collisionStats;
};

struct AvbdRigidTriangleSurfaceContactRangeTiming
{
	PxU64 currentSdfNanos;
	PxU64 sweptSdfNanos;
	PxU64 featureNanos;
};

struct AvbdSelfBvhContactRange
{
	const Dy::AvbdSoftParticle*	particles;
	const Dy::AvbdSoftBody*		body;
	PxU32					softBodyIndex;
	const Dy::AvbdSelfCollisionAdjacency* adjacency;
	const Dy::AvbdSoftContactWorkspace* parentWorkspace;
	Dy::AvbdSoftContactWorkspace*	rangeWorkspace;
	PxU32					vertexBegin;
	PxU32					vertexEnd;
	PxU32					edgeBegin;
	PxU32					edgeEnd;
	PxArray<Dy::AvbdSoftContact>* contacts;
	const Dy::AvbdOGCParams*	params;
	Dy::AvbdSoftCollisionStats*	collisionStats;
};

void executeAvbdWorldPlaneContactRange(
	const AvbdWorldPlaneContactRange& range);
void executeAvbdRigidBoxSdfContactRange(
	const AvbdRigidBoxSdfContactRange& range);
void executeAvbdRigidSphereSdfContactRange(
	const AvbdRigidSphereSdfContactRange& range);
void executeAvbdRigidCapsuleSdfContactRange(
	const AvbdRigidCapsuleSdfContactRange& range);
void executeAvbdRigidConvexSdfContactRange(
	const AvbdRigidConvexSdfContactRange& range);
void executeAvbdRigidTriangleSurfaceContactRange(
	const AvbdRigidTriangleSurfaceContactRange& range,
	AvbdRigidTriangleSurfaceContactRangeTiming* rangeTiming,
	Dy::AvbdRigidTriangleSurfaceFeaturePlanRangeTiming* featurePlanTiming,
	Dy::AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming*
		featureSweptSubstageTiming,
	Dy::AvbdRigidTriangleSurfaceDiscreteOGCQueryStats* discreteQueryStats);
void executeAvbdSelfBvhContactRange(
	const AvbdSelfBvhContactRange& range);

} // namespace Sc
} // namespace physx

#endif // SC_AVBD_ANALYTIC_CONTACT_KERNELS_H
