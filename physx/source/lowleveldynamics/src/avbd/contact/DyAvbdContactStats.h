// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions are met.

#ifndef DY_AVBD_CONTACT_STATS_H
#define DY_AVBD_CONTACT_STATS_H

#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Dy
{

// Caller-owned detector counters. The contact pipeline may omit the object
// entirely in production; keeping the POD outside the soft-body component
// prevents diagnostic state from defining detector ownership.
struct AvbdSoftCollisionStats
{
	PxU64 detectionCalls;
	PxU64 bodyPairs;
	PxU64 overlappingBodyPairs;
	PxU64 particleSurfaceCandidates;
	PxU64 insideTriangleTests;
	PxU64 closestTriangleTests;
	PxU64 selfTriangleTests;
	PxU64 selfTriangleBoundsBuilt;
	PxU64 selfVertexSweepEntriesBuilt;
	PxU64 selfEdgeBoundsBuilt;
	PxU64 surfaceTriangleBvhRefitNodes;
	PxU64 surfaceTriangleBvhCandidateTriangles;
	PxU64 surfaceEdgeBvhRefitNodes;
	PxU64 surfaceEdgeBvhCandidateEdges;
	PxU64 rigidParticleBoxTests;
	PxU64 rigidParticleSphereTests;
	PxU64 rigidParticleCapsuleTests;
	PxU64 rigidParticleConvexTests;
	PxU64 rigidParticleTriangleSurfaceTests;
	PxU64 rigidTriangleSurfaceFaceCandidates;
	PxU64 rigidTriangleSurfaceFaceTests;
	PxU64 rigidTriangleSurfaceEdgeCandidates;
	PxU64 rigidTriangleSurfaceEdgeTests;
	PxU64 rigidTriangleSurfaceVertexCandidates;
	PxU64 rigidTriangleSurfaceVertexTests;
	PxU64 generatedGroundContacts;
	PxU64 generatedRigidContacts;
	PxU64 generatedSoftContacts;
	PxU64 generatedSelfContacts;

	AvbdSoftCollisionStats()
		: detectionCalls(0), bodyPairs(0), overlappingBodyPairs(0),
		  particleSurfaceCandidates(0), insideTriangleTests(0),
		  closestTriangleTests(0), selfTriangleTests(0),
		  selfTriangleBoundsBuilt(0), selfVertexSweepEntriesBuilt(0),
		  selfEdgeBoundsBuilt(0), surfaceTriangleBvhRefitNodes(0),
		  surfaceTriangleBvhCandidateTriangles(0),
		  surfaceEdgeBvhRefitNodes(0), surfaceEdgeBvhCandidateEdges(0),
		  rigidParticleBoxTests(0), rigidParticleSphereTests(0),
		  rigidParticleCapsuleTests(0), rigidParticleConvexTests(0),
		  rigidParticleTriangleSurfaceTests(0),
		  rigidTriangleSurfaceFaceCandidates(0),
		  rigidTriangleSurfaceFaceTests(0),
		  rigidTriangleSurfaceEdgeCandidates(0),
		  rigidTriangleSurfaceEdgeTests(0),
		  rigidTriangleSurfaceVertexCandidates(0),
		  rigidTriangleSurfaceVertexTests(0), generatedGroundContacts(0),
		  generatedRigidContacts(0), generatedSoftContacts(0),
		  generatedSelfContacts(0)
	{
	}

	void accumulate(const AvbdSoftCollisionStats& other)
	{
		detectionCalls += other.detectionCalls;
		bodyPairs += other.bodyPairs;
		overlappingBodyPairs += other.overlappingBodyPairs;
		particleSurfaceCandidates += other.particleSurfaceCandidates;
		insideTriangleTests += other.insideTriangleTests;
		closestTriangleTests += other.closestTriangleTests;
		selfTriangleTests += other.selfTriangleTests;
		selfTriangleBoundsBuilt += other.selfTriangleBoundsBuilt;
		selfVertexSweepEntriesBuilt += other.selfVertexSweepEntriesBuilt;
		selfEdgeBoundsBuilt += other.selfEdgeBoundsBuilt;
		surfaceTriangleBvhRefitNodes += other.surfaceTriangleBvhRefitNodes;
		surfaceTriangleBvhCandidateTriangles +=
			other.surfaceTriangleBvhCandidateTriangles;
		surfaceEdgeBvhRefitNodes += other.surfaceEdgeBvhRefitNodes;
		surfaceEdgeBvhCandidateEdges += other.surfaceEdgeBvhCandidateEdges;
		rigidParticleBoxTests += other.rigidParticleBoxTests;
		rigidParticleSphereTests += other.rigidParticleSphereTests;
		rigidParticleCapsuleTests += other.rigidParticleCapsuleTests;
		rigidParticleConvexTests += other.rigidParticleConvexTests;
		rigidParticleTriangleSurfaceTests +=
			other.rigidParticleTriangleSurfaceTests;
		rigidTriangleSurfaceFaceCandidates +=
			other.rigidTriangleSurfaceFaceCandidates;
		rigidTriangleSurfaceFaceTests += other.rigidTriangleSurfaceFaceTests;
		rigidTriangleSurfaceEdgeCandidates +=
			other.rigidTriangleSurfaceEdgeCandidates;
		rigidTriangleSurfaceEdgeTests += other.rigidTriangleSurfaceEdgeTests;
		rigidTriangleSurfaceVertexCandidates +=
			other.rigidTriangleSurfaceVertexCandidates;
		rigidTriangleSurfaceVertexTests += other.rigidTriangleSurfaceVertexTests;
		generatedGroundContacts += other.generatedGroundContacts;
		generatedRigidContacts += other.generatedRigidContacts;
		generatedSoftContacts += other.generatedSoftContacts;
		generatedSelfContacts += other.generatedSelfContacts;
	}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_STATS_H
