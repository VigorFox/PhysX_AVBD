// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

#ifndef DY_AVBD_DETECTION_PLAN_H
#define DY_AVBD_DETECTION_PLAN_H

#include "avbd/contact/DyAvbdContactBounds.h"
#include "foundation/PxArray.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftParticle;
struct AvbdSoftBody;
struct AvbdRigidBox;
struct AvbdRigidSphere;
struct AvbdRigidCapsule;
struct AvbdRigidConvex;
struct AvbdRigidTriangleSurface;
struct AvbdWorldPlane;

typedef PxArray<PxArray<PxU32> > AvbdSelfCollisionAdjacency;

// Immutable geometry/source view for one OGC detection transaction. Contact
// output, params, statistics and workspace remain explicit call-owned state;
// this view only describes the collision domain and optional primitive families.
struct AvbdSoftContactDetectionView
{
	AvbdSoftParticle* particles;
	PxU32 numParticles;
	AvbdSoftBody* softBodies;
	PxU32 numSoftBodies;
	const AvbdRigidBox* rigidBoxes;
	PxU32 numRigidBoxes;
	const AvbdSelfCollisionAdjacency* selfCollisionAdjacencies;
	PxU32 numSelfCollisionAdjacencies;
	const AvbdWorldPlane* worldPlanes;
	PxU32 numWorldPlanes;
	bool includeLegacyGround;
	const PxU8* selfCollisionEnabled;
	const AvbdRigidSphere* rigidSpheres;
	PxU32 numRigidSpheres;
	const AvbdRigidCapsule* rigidCapsules;
	PxU32 numRigidCapsules;
	const AvbdRigidConvex* rigidConvexes;
	PxU32 numRigidConvexes;
	const AvbdRigidTriangleSurface* rigidTriangleSurfaces;
	PxU32 numRigidTriangleSurfaces;
	bool includeSoftTargets;

	AvbdSoftContactDetectionView()
		: particles(NULL), numParticles(0), softBodies(NULL),
		  numSoftBodies(0), rigidBoxes(NULL), numRigidBoxes(0),
		  selfCollisionAdjacencies(NULL), numSelfCollisionAdjacencies(0),
		  worldPlanes(NULL), numWorldPlanes(0), includeLegacyGround(true),
		  selfCollisionEnabled(NULL), rigidSpheres(NULL), numRigidSpheres(0),
		  rigidCapsules(NULL), numRigidCapsules(0), rigidConvexes(NULL),
		  numRigidConvexes(0), rigidTriangleSurfaces(NULL),
		  numRigidTriangleSurfaces(0), includeSoftTargets(true)
	{
	}
};

struct AvbdSoftPairDetectionPlan
{
	PxU32 bodyA;
	PxU32 bodyB;
	bool swept;
	PxVec3 minimumA;
	PxVec3 maximumA;
	PxVec3 minimumB;
	PxVec3 maximumB;
};

struct AvbdSurfaceBvhNodeBounds
{
	PxVec3 minimum;
	PxVec3 maximum;
};

struct AvbdSoftPairBvhEpochSpans
{
	PxArray<AvbdSurfaceBvhNodeBounds> currentBounds;
	PxArray<AvbdSurfaceBvhNodeBounds> sweptBounds;
	PxU32 currentRequiredEpoch;
	PxU32 sweptRequiredEpoch;
	PxU32 currentRefitEpoch;
	PxU32 sweptRefitEpoch;

	AvbdSoftPairBvhEpochSpans()
		: currentRequiredEpoch(0), sweptRequiredEpoch(0),
		  currentRefitEpoch(0), sweptRefitEpoch(0)
	{
	}
};

// Immutable source-order plan for one complete OGC redetection epoch.  The
// phase remains a data-only value; validation and execution policy belong to
// the owning step/orchestration layer.
struct AvbdSoftContactRedetectionPhase
{
	enum Type : PxU8
	{
		eWORLD_PLANES,
		eLEGACY_GROUND,
		eRIGID_BOXES,
		eRIGID_SPHERES,
		eRIGID_CAPSULES,
		eRIGID_CONVEXES,
		eRIGID_TRIANGLE_SURFACES,
		eSOFT_SOFT,
		eSELF_BODY
	};

	Type type;
	PxU32 sourceBegin;
	PxU32 sourceEnd;

	AvbdSoftContactRedetectionPhase()
		: type(eLEGACY_GROUND), sourceBegin(0), sourceEnd(0)
	{
	}

	AvbdSoftContactRedetectionPhase(
		Type inputType, PxU32 inputSourceBegin, PxU32 inputSourceEnd)
		: type(inputType), sourceBegin(inputSourceBegin),
		  sourceEnd(inputSourceEnd)
	{
	}
};

struct AvbdSurfaceTriangleBvhNode
{
	PxVec3 minimum;
	PxVec3 maximum;
	PxU32 leftChild;
	PxU32 rightChild;
	PxU32 firstTriangle;
	PxU32 triangleCount;

	PX_FORCE_INLINE bool isLeaf() const
	{
		return leftChild == PX_MAX_U32;
	}
};

struct AvbdSurfaceEdgeBvhNode
{
	PxVec3 minimum;
	PxVec3 maximum;
	PxU32 leftChild;
	PxU32 rightChild;
	PxU32 firstEdge;
	PxU32 edgeCount;

	PX_FORCE_INLINE bool isLeaf() const
	{
		return leftChild == PX_MAX_U32;
	}
};

struct AvbdSoftSoftPairQueryScratch
{
	PxArray<AvbdSoftPairEdgeBounds> edgeBoundsA;
	PxArray<AvbdSoftPairEdgeBounds> edgeBoundsB;
	PxArray<PxU32> triangleCandidates;

	void reserve(PxU32 edgeCountA, PxU32 edgeCountB,
		PxU32 triangleCandidateCapacity = 0)
	{
		edgeBoundsA.reserve(edgeCountA);
		edgeBoundsB.reserve(edgeCountB);
		triangleCandidates.reserve(triangleCandidateCapacity);
	}

	void reset()
	{
		edgeBoundsA.reset();
		edgeBoundsB.reset();
		triangleCandidates.reset();
	}
};

struct AvbdRigidConvexEdgeBounds
{
	PxVec3 point0;
	PxVec3 point1;
	PxVec3 minimum;
	PxVec3 maximum;
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_DETECTION_PLAN_H
