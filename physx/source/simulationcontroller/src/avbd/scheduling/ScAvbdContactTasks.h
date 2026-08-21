// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_CONTACT_TASKS_H
#define SC_AVBD_CONTACT_TASKS_H

#include "avbd/scene/ScAvbdCpuSoftScene.h"
#include "ScAvbdAnalyticContactKernels.h"

namespace physx
{
namespace Sc
{

		// P3 keeps the nonlinear component solve serial for now.  Its first
		// task-graph slice is deliberately limited to post-solve output ranges:
		// each range owns whole deformable entries and therefore disjoint host
		// particle/output buffers.  The parent Scene task joins these before it
		// mutates sleep/island state or continues the simulation pipeline.
		class AvbdCpuSoftScene::WriteBackTask : public Cm::Task, public PxUserAllocated
		{
		public:
			WriteBackTask(PxU64 contextId, AvbdCpuSoftScene& scene);

			void configure(
				PxU32 entryBegin, PxU32 entryEnd);

			virtual void runInternal() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&					mScene;
			PxU32							mEntryBegin;
			PxU32							mEntryEnd;
		};

		// The P3 pre-solve task owns only whole-entry particle ranges.  It runs
		// after the low-level ePREPARE prefix and before eRESUME's predicted
		// position OGC redetection, so no task observes or mutates contact state.
		class AvbdCpuSoftScene::PredictionTask : public Cm::Task, public PxUserAllocated
		{
		public:
			PredictionTask(PxU64 contextId, AvbdCpuSoftScene& scene);

			void configure(
				PxU32 entryBegin, PxU32 entryEnd, PxReal dt,
				const PxVec3& gravity);

			virtual void runInternal() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&					mScene;
			PxU32							mEntryBegin;
			PxU32							mEntryEnd;
			PxReal							mDt;
			PxVec3							mGravity;
		};

		// A primal child owns either one stable packed causal-layer subrange or a
		// contiguous range of complete independent bodies. It receives no
		// Scene/contact/workspace mutable state beyond the frozen solve context,
		// and writes one private observation for the parent fan-in.
		class AvbdCpuSoftScene::CausalLayerTask : public Cm::Task, public PxUserAllocated
		{
		public:
			CausalLayerTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex);

			void configure(
				const Dy::AvbdParticlePrimalSolveContext& solveContext,
				const Dy::AvbdSoftBody* bodies, PxU32 bodyCount,
				const PxU32* particleBodyIndices, PxU32 numParticles,
				const PxU32* packedParticleIndices,
				PxU32 packedBegin, PxU32 packedEnd,
				Dy::AvbdParticlePrimalRangeObservation& observation);

			void configureIndependentBodyRange(
				const Dy::AvbdParticlePrimalSolveContext& solveContext,
				const Dy::AvbdSoftBody* bodies, PxU32 bodyCount,
				PxU32 bodyBegin, PxU32 bodyEnd,
				Dy::AvbdParticlePrimalRangeObservation& observation);

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&					mScene;
			PxU32						mPoolIndex;
			const Dy::AvbdParticlePrimalSolveContext*	mSolveContext;
			const Dy::AvbdSoftBody*			mBodies;
			PxU32						mBodyCount;
			const PxU32*					mParticleBodyIndices;
			PxU32						mNumParticles;
			const PxU32*					mPackedParticleIndices;
			PxU32						mPackedBegin;
			PxU32						mPackedEnd;
			bool						mIndependentBodyRange;
			PxU32						mBodyBegin;
			PxU32						mBodyEnd;
			Dy::AvbdParticlePrimalRangeObservation*	mObservation;
		};

		// A parent fan-in task owns no particle range. It merges private
		// observations in fixed range order, advances the persistent step state,
		// and asks Scene to publish the next primal range or resume the existing
		// post-solve/write-back continuation.
		class AvbdCpuSoftScene::CausalLayerFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			CausalLayerFinishTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				Scene& owner, PxU32 poolIndex);

			virtual void runInternal() PX_OVERRIDE;

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.3b: the first collision worker is intentionally narrower than the
		// generic OGC rebuild.  It owns one particle interval of a world-plane
		// only transaction and may append solely to its pre-reserved private
		// stream.  The parent performs the fixed-order merge and every mutable
		// contact/workspace transition after the fan-in.
		class AvbdCpuSoftScene::WorldPlaneContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			WorldPlaneContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex);

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdWorldPlane* planes, PxU32 numPlanes,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts, PxReal margin);

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&			mScene;
			PxU32					mPoolIndex;
			AvbdWorldPlaneContactRange	mRange;
		};

		class AvbdCpuSoftScene::WorldPlaneContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			WorldPlaneContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex);

			virtual void runInternal() PX_OVERRIDE;

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.12b range-owns both static-box SDF families, but retains independent
		// private streams. The parent stable-merges all current ranges before all
		// swept ranges, then appends the feature suffix. Both leaves read only
		// immutable inputs, so a child may compute them back-to-back.
		class AvbdCpuSoftScene::RigidBoxSdfContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidBoxSdfContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex);

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidBox* boxes, PxU32 numBoxes,
				const Dy::AvbdSoftContact* previousContacts,
				PxU32 numPreviousContacts,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin);

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32						mPoolIndex;
			AvbdRigidBoxSdfContactRange	mRange;
		};

		class AvbdCpuSoftScene::RigidBoxSdfContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidBoxSdfContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex);

			virtual void runInternal() PX_OVERRIDE;

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.13b range-owns the current and swept static-sphere SDF families,
		// with independent private streams. The parent merges the complete
		// current family before the swept family and retains both feature suffixes.
		class AvbdCpuSoftScene::RigidSphereSdfContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidSphereSdfContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex);

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidSphere* spheres, PxU32 numSpheres,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin);

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&					mScene;
			PxU32							mPoolIndex;
			AvbdRigidSphereSdfContactRange	mRange;
		};

		class AvbdCpuSoftScene::RigidSphereSdfContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidSphereSdfContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex);

			virtual void runInternal() PX_OVERRIDE;

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.14b range-owns current and swept static-capsule SDF in independent
		// streams. The Scene continuation remains mutually exclusive with spheres;
		// neither geometry, telemetry, nor output storage is shared.
		class AvbdCpuSoftScene::RigidCapsuleSdfContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidCapsuleSdfContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex);

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidCapsule* capsules, PxU32 numCapsules,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin);

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&					mScene;
			PxU32							mPoolIndex;
			AvbdRigidCapsuleSdfContactRange	mRange;
		};

		class AvbdCpuSoftScene::RigidCapsuleSdfContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidCapsuleSdfContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex);

			virtual void runInternal() PX_OVERRIDE;

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.15b range-owns current and swept static-convex SDF in independent
		// streams. Only the continuation readiness bit is shared with other
		// mutually exclusive smooth-rigid SDF transactions.
		class AvbdCpuSoftScene::RigidConvexSdfContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidConvexSdfContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex);

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidConvex* convexes, PxU32 numConvexes,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin);

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&					mScene;
			PxU32							mPoolIndex;
			AvbdRigidConvexSdfContactRange	mRange;
		};

		class AvbdCpuSoftScene::RigidConvexSdfContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidConvexSdfContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex);

			virtual void runInternal() PX_OVERRIDE;

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.17d range-owns every static triangle contact family. Current/swept
		// SDF stay particle-range streams; OGC features consume contiguous rows
		// of the immutable P5.17b parent plan. Every child carries complete BVH
		// query scratch, so the Scene descriptor remains read only while sibling
		// ranges run. Historical P5.16b: Both OGC feature suffixes remain parent-owned.
		class AvbdCpuSoftScene::RigidTriangleSurfaceContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidTriangleSurfaceContactTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, PxU32 poolIndex);

			PxU64 reserveBvhCandidateScratch(PxU32 triangleCapacity,
				PxU32 edgeCapacity, PxU32 vertexCapacity,
				PxU32 forwardOwnerQueryStampCapacity = 0,
				PxU32 forwardOwnerResultCacheCapacity = 0,
				PxU32 forwardOwnerResultCacheSurfaceSlotCapacity = 0);

			PxU64 getBvhCandidateScratchResidentPayloadBytes() const;

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidTriangleSurface* surfaces,
				PxU32 numSurfaces, const Dy::AvbdSoftBody* bodies,
				PxU32 numBodies, PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts,
				const Dy::AvbdRigidTriangleSurfaceFeaturePlan& featurePlan,
				PxU32 featurePlanBegin, PxU32 featurePlanEnd,
				PxArray<Dy::AvbdSoftContact>& featureContacts,
				PxReal margin, Dy::AvbdSoftCollisionStats* collisionStats);

			void configureFeaturePlanRoundRobin(
				PxArray<PxArray<Dy::AvbdSoftContact> >& featurePlanOutputs,
				PxU32 taskIndex, PxU32 taskCount);

			void configureFeaturePlanRowPrivateOutputs(
				PxArray<PxArray<Dy::AvbdSoftContact> >& featurePlanOutputs);

			void configureForwardOwnerQueryStats();

			void configureDiscreteQueryStats();

			void configureDiscreteBodyLocalBoundsCull();

			void configureForwardOwnerResultCache();

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32						mPoolIndex;
			AvbdRigidTriangleSurfaceContactRange mRange;
			Dy::AvbdRigidTriangleSurfaceQueryScratch mQueryScratch;
			PxArray<PxU32>						mForwardOwnerQueryStamps;
			Dy::AvbdRigidTriangleSurfaceForwardOwnerQueryStats
										mForwardOwnerQueryStats;
			PxU32						mForwardOwnerQueryStamp;
			PxArray<PxU32>						mForwardOwnerResultCacheEntries;
			PxArray<PxU32>						mForwardOwnerResultCacheSurfaceSlots;
			Dy::AvbdRigidTriangleSurfaceForwardOwnerResultCache
										mForwardOwnerResultCache;
			PxU32						mForwardOwnerResultCacheStamp;
		};

		class AvbdCpuSoftScene::RigidTriangleSurfaceContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidTriangleSurfaceContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex);

			virtual void runInternal() PX_OVERRIDE;

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.10b consumes a single body whose parent has already prepared its
		// self stress and triangle/edge BVH epoch. The child may own only a
		// contiguous VF or EE outer range; its query workspace is never shared.
		class AvbdCpuSoftScene::SelfBvhContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			SelfBvhContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex);

			void reserveQueryScratch(const Dy::AvbdSoftBody& body);

			void configure(const Dy::AvbdSoftParticle* particles,
				const Dy::AvbdSoftBody& body, PxU32 softBodyIndex,
				const Dy::AvbdSelfCollisionAdjacency& adjacency,
				const Dy::AvbdSoftContactWorkspace& parentWorkspace,
				PxU32 vertexBegin, PxU32 vertexEnd,
				PxU32 edgeBegin, PxU32 edgeEnd,
				PxArray<Dy::AvbdSoftContact>& contacts,
				const Dy::AvbdOGCParams& params,
				Dy::AvbdSoftCollisionStats* collisionStats);

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32					mPoolIndex;
			Dy::AvbdOGCParams				mParams;
			Dy::AvbdSoftContactWorkspace	mRangeWorkspace;
			AvbdSelfBvhContactRange		mRange;
		};

		class AvbdCpuSoftScene::SelfBvhContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			SelfBvhContactFinishTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				Scene& owner, PxU32 poolIndex);

			virtual void runInternal() PX_OVERRIDE;

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&			mOwner;
			PxU32			mPoolIndex;
		};

		// One aggregate child owns matching disjoint ranges for all three source
		// families.  It writes one private stream per canonical source substage;
		// the parent never merges by child completion order.
		class AvbdCpuSoftScene::StaticWorldSelfOgcContactTask :
			public Cm::Task, public PxUserAllocated
		{
		public:
			StaticWorldSelfOgcContactTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, PxU32 poolIndex);

			void reserveQueryScratch(const Dy::AvbdSoftBody& body);

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdWorldPlane* planes, PxU32 numPlanes,
				const Dy::AvbdRigidBox* boxes, PxU32 numBoxes,
				const Dy::AvbdSoftContact* previousContacts,
				PxU32 numPreviousContacts, const Dy::AvbdSoftBody& body,
				const Dy::AvbdSelfCollisionAdjacency& adjacency,
				const Dy::AvbdSoftContactWorkspace& preparedWorkspace,
				PxU32 vertexBegin, PxU32 vertexEnd,
				PxU32 edgeBegin, PxU32 edgeEnd,
				PxArray<Dy::AvbdSoftContact>& worldContacts,
				PxArray<Dy::AvbdSoftContact>& boxContacts,
				PxArray<Dy::AvbdSoftContact>& boxSweptContacts,
				PxArray<Dy::AvbdSoftContact>& selfVertexContacts,
				PxArray<Dy::AvbdSoftContact>& selfEdgeContacts,
				const Dy::AvbdOGCParams& params,
				Dy::AvbdSoftCollisionStats* taskStats,
				PxReal margin);

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32					mPoolIndex;
			Dy::AvbdOGCParams				mParams;
			Dy::AvbdSoftContactWorkspace	mRangeWorkspace;
			AvbdWorldPlaneContactRange	mWorldRange;
			AvbdRigidBoxSdfContactRange	mBoxRange;
			AvbdSelfBvhContactRange		mSelfVertexRange;
			AvbdSelfBvhContactRange		mSelfEdgeRange;
		};

		// A single parent fan-in joins the private plane, box, and self-BVH
		// streams.  It deliberately uses the existing mutually-exclusive
		// smooth-rigid continuation slot: the pending aggregate transaction is
		// dispatched before any individual smooth-rigid leaf, while this distinct
		// pool keeps its lifetime separate from those source-specific pools.
		class AvbdCpuSoftScene::StaticWorldSelfOgcContactFinishTask :
			public Cm::Task, public PxUserAllocated
		{
		public:
			StaticWorldSelfOgcContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex);

			virtual void runInternal() PX_OVERRIDE;

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE;

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&			mOwner;
			PxU32			mPoolIndex;
		};

} // namespace Sc
} // namespace physx

#endif // SC_AVBD_CONTACT_TASKS_H
