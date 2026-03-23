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

#include "DyAvbdDynamics.h"
#include "../../common/include/utils/PxcScratchAllocator.h"
#include "DyArticulationCore.h"
#include "DyAvbdBodyConversion.h"
#include "DyAvbdConstraint.h"
#include "DyAvbdTasks.h"
#include "DyConstraint.h"
#include "DyFeatherstoneArticulation.h"
#include "DyIslandManager.h"
#include "DyVArticulation.h"
#include "PxContact.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsIslandManagerTypes.h"
#include "PxsRigidBody.h"
#include "PxsSimpleIslandManager.h"
#include "common/PxProfileZone.h"
#include "foundation/PxMath.h"

using namespace physx;
using namespace physx::Dy;

// Global frame counter for motor deduplication
// This is incremented at the start of each update() call
static physx::PxU64 gAvbdMotorFrameCounter = 0;

// Debug: set to 1 to process all islands sequentially (no task parallelism)
// This makes output deterministic and easier to debug.
#define AVBD_DEBUG_SEQUENTIAL 0

// Accessor for the solver to get current frame
physx::PxU64 getAvbdMotorFrameCounter() { return gAvbdMotorFrameCounter; }

#ifndef AVBD_JOINT_DEBUG
#define AVBD_JOINT_DEBUG 0
#endif
#ifndef AVBD_JOINT_DEBUG_FRAMES
#define AVBD_JOINT_DEBUG_FRAMES 2
#endif



//=============================================================================
// Articulation Internal Joints Helper (forward declaration)
//=============================================================================
static void prepareArticulationInternalJoints(
    FeatherstoneArticulation *articulation, PxU32 firstBodyIndex,
    AvbdD6JointConstraint *d6Constraints, PxU32 &numD6, PxU32 maxD6,
    AvbdGearJointConstraint *gearConstraints, PxU32 &numGear, PxU32 maxGear,
    PxReal dt = 1.0f / 60.0f);

//=============================================================================
// Helper function to find articulation link index from rigid core
//=============================================================================
static PxU32 findArticulationLinkIndex(FeatherstoneArticulation *articulation,
                                       const PxsRigidCore *rigidCore) {

  if (!articulation || !rigidCore)
    return PX_MAX_U32;

  ArticulationData &artData = articulation->getArticulationData();
  const PxU32 linkCount = artData.getLinkCount();

  for (PxU32 linkIdx = 0; linkIdx < linkCount; ++linkIdx) {
    const ArticulationLink &link = artData.getLink(linkIdx);
    if (link.bodyCore == rigidCore) {
      return linkIdx;
    }
  }

  return PX_MAX_U32;
}

//=============================================================================
// Helper: Allocate from scratch, fallback to main allocator if needed
// Note: This is called only from update() which runs on a single thread,
// so we can use a simpler tracking mechanism without mutex for the common case.
//=============================================================================
static void *allocWithFallback(PxcScratchAllocator &scratch,
                               PxVirtualAllocatorCallback *mainAllocator,
                               PxArray<void *> &fallbackAllocations, PxU32 size,
                               const char *name) {

  // First try scratch allocator (no heap fallback) - fast path
  void *ptr = scratch.alloc(size, false);
  if (ptr) {
    return ptr;
  }

  // Scratch memory exhausted - use main allocator (slower path)
  if (mainAllocator) {
    PxAllocatorCallback *allocator =
        reinterpret_cast<PxAllocatorCallback *>(mainAllocator);
    ptr = allocator->allocate(size, name, __FILE__, __LINE__);
    if (ptr) {
      // Track for cleanup in mergeResults()
      // No mutex needed here since update() is called from a single thread
      // context
      fallbackAllocations.pushBack(ptr);
    }
  }
  return ptr;
}

//=============================================================================
// Adapter Implementation
//=============================================================================

AvbdDynamicsContext::ScratchAllocatorAdapter::ScratchAllocatorAdapter(
    PxcScratchAllocator &scratch)
    : mScratch(scratch) {}

void *AvbdDynamicsContext::ScratchAllocatorAdapter::allocate(size_t size,
                                                             const char *,
                                                             const char *,
                                                             int) {
  // Use scratch allocator WITHOUT heap fallback to avoid memory leaks.
  // Scratch memory is automatically reset at frame end.
  return mScratch.alloc(static_cast<PxU32>(size), false);
}

void AvbdDynamicsContext::ScratchAllocatorAdapter::deallocate(void *) {}

//=============================================================================
// Constructor / Destructor
//=============================================================================

AvbdDynamicsContext::AvbdDynamicsContext(
    PxcNpMemBlockPool *memBlockPool, PxcScratchAllocator &scratchAllocator,
    Cm::FlushPool &taskPool, PxvSimStats &simStats, PxTaskManager *taskManager,
    PxVirtualAllocatorCallback *allocatorCallback,
    PxsMaterialManager *materialManager, IG::SimpleIslandManager &islandManager,
    PxU64 contextID, bool enableStabilization, bool useEnhancedDeterminism,
    bool solveArticulationContactLast, PxReal maxBiasCoefficient,
    bool frictionEveryIteration, PxReal lengthScale,
    bool isResidualReportingEnabled)
    : DynamicsContextBase(memBlockPool, taskPool, simStats, allocatorCallback,
                          materialManager, islandManager, contextID,
                          maxBiasCoefficient, lengthScale, enableStabilization,
                          useEnhancedDeterminism, solveArticulationContactLast,
                          isResidualReportingEnabled),
      mScratchAllocator(scratchAllocator), mTaskManager(taskManager),
      mScratchAdapter(scratchAllocator),
      mFrictionEveryIteration(frictionEveryIteration),
      mAllocatorCallback(allocatorCallback) {
  PX_UNUSED(frictionEveryIteration);
  mSolverInitialized = false;

  createThresholdStream(*allocatorCallback);
  createForceChangeThresholdStream(*allocatorCallback);

  mExceededForceThresholdStream[0] =
      PX_NEW(ThresholdStream)(*allocatorCallback);
  mExceededForceThresholdStream[1] =
      PX_NEW(ThresholdStream)(*allocatorCallback);

  // Use the main allocator callback for tasks, NOT the scratch adapter.
  // Tasks need explicit deallocation, which ScratchAllocatorAdapter doesn't
  // provide.
  mTaskFactory = new AvbdTaskFactory(
      mTaskManager,
      *reinterpret_cast<PxAllocatorCallback *>(mAllocatorCallback));

  // Initialize lambda warm-starting cache
  mEnableLambdaWarmStart = true;
  // Pre-allocate for ~1000 contact managers x 4 contacts each
  mLambdaCache.resize(4096);
  memset(mLambdaCache.begin(), 0, sizeof(CachedLambda) * mLambdaCache.size());
}

AvbdDynamicsContext::~AvbdDynamicsContext() {
  delete mTaskFactory;

  PX_DELETE(mExceededForceThresholdStream[1]);
  PX_DELETE(mExceededForceThresholdStream[0]);

  if (mSolverInitialized && mAllocatorCallback) {
    mConstraintColoring.release();
    mSolver.release();
  }
}

//=============================================================================
// Context Interface Implementation
//=============================================================================

void AvbdDynamicsContext::destroy() {
  this->~AvbdDynamicsContext();
  PX_FREE_THIS;
}

void AvbdDynamicsContext::destroyTask(AvbdTask *task) {
  if (mTaskFactory) {
    mTaskFactory->destroyTask(task);
  }
}

//=============================================================================
// Lambda Warm-Starting Cache Write-Back
//=============================================================================

namespace physx {
namespace Dy {
void writeLambdaToCache(AvbdDynamicsContext &ctx,
                        AvbdContactConstraint *constraints,
                        PxU32 numConstraints) {
  if (!ctx.mEnableLambdaWarmStart || !constraints || numConstraints == 0) {
    return;
  }

  PxArray<AvbdDynamicsContext::CachedLambda> &cache = ctx.mLambdaCache;

  for (PxU32 i = 0; i < numConstraints; ++i) {
    const AvbdContactConstraint &constraint = constraints[i];
    const PxU32 cacheIdx = constraint.cacheIndex;

    // Skip invalid cache indices
    if (cacheIdx >= cache.size()) {
      continue;
    }

    // Write back solved lambda and penalty values
    AvbdDynamicsContext::CachedLambda &cached = cache[cacheIdx];
    cached.lambda = constraint.header.lambda;
    cached.tangentLambda0 = constraint.tangentLambda0;
    cached.tangentLambda1 = constraint.tangentLambda1;
    cached.penalty = constraint.header.penalty;
    cached.tangentPenalty0 = constraint.tangentPenalty0;
    cached.tangentPenalty1 = constraint.tangentPenalty1;
    cached.frameAge = 0; // Reset age on update
  }
}
} // namespace Dy
} // namespace physx

void AvbdDynamicsContext::update(
    Cm::FlushPool &flushPool, PxBaseTask *continuation,
    PxBaseTask *postPartitioningTask, PxBaseTask *processLostTouchTask,
    PxvNphaseImplementationContext *nPhaseContext, PxU32 maxPatchesPerCM,
    PxU32 maxArticulationLinks, PxReal dt, const PxVec3 &gravity,
    PxBitMapPinned &changedHandleMap) {

  PX_PROFILE_ZONE("AVBD.update", mContextID);

  // Increment global frame counter for motor deduplication
  gAvbdMotorFrameCounter++;

  // Lambda warm-starting: age all cached entries at frame start
  if (mEnableLambdaWarmStart) {
    PX_PROFILE_ZONE("AVBD.ageLambdaCache", mContextID);
    for (PxU32 i = 0; i < mLambdaCache.size(); ++i) {
      if (mLambdaCache[i].frameAge < 255) {
        mLambdaCache[i].frameAge++;
      }
    }
  }

  PX_UNUSED(flushPool);
  PX_UNUSED(postPartitioningTask);
  PX_UNUSED(processLostTouchTask);
  PX_UNUSED(maxPatchesPerCM);
  PX_UNUSED(changedHandleMap);

  mOutputIterator = nPhaseContext->getContactManagerOutputs();

  mDt = dt;
  mInvDt = dt > 0.0f ? 1.0f / dt : 0.0f;
  mGravity = gravity;

  const IG::IslandSim &islandSim = mIslandManager.getAccurateIslandSim();
  const PxU32 islandCount = islandSim.getNbActiveIslands();
  const PxU32 numDynamicBodies =
      islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);
  const PxU32 numArticulations =
      islandSim.getNbActiveNodes(IG::Node::eARTICULATION_TYPE);

  if (islandCount == 0) {
    return;
  }

  // Calculate total body count including articulation links
  PxU32 totalBodyCount = numDynamicBodies + numArticulations * maxArticulationLinks;

  // Allocate global arrays - use scratch with main allocator fallback
  AvbdSolverBody *avbdBodies = nullptr;
  PxsRigidBody **rigidBodies = nullptr;
  {
    PX_PROFILE_ZONE("AVBD.allocateMemory", mContextID);
    avbdBodies = reinterpret_cast<AvbdSolverBody *>(allocWithFallback(
        mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
        sizeof(AvbdSolverBody) * totalBodyCount, "AvbdSolverBody"));

    rigidBodies = reinterpret_cast<PxsRigidBody **>(allocWithFallback(
        mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
        sizeof(PxsRigidBody *) * totalBodyCount, "RigidBodies"));
  }

  // Check if allocation failed completely
  if (!avbdBodies || !rigidBodies) {
    return;
  }

  // Track articulation info for writeback
  FeatherstoneArticulation **articulationForBody = nullptr;
  PxU32 *linkIndexForBody = nullptr;
  if (numArticulations > 0 && maxArticulationLinks > 0) {
    articulationForBody =
        reinterpret_cast<FeatherstoneArticulation **>(allocWithFallback(
            mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
            sizeof(FeatherstoneArticulation *) * totalBodyCount,
            "ArticulationForBody"));
    linkIndexForBody = reinterpret_cast<PxU32 *>(allocWithFallback(
        mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
        sizeof(PxU32) * totalBodyCount, "LinkIndexForBody"));

    if (articulationForBody && linkIndexForBody) {
      for (PxU32 i = 0; i < totalBodyCount; ++i) {
        articulationForBody[i] = nullptr;
        linkIndexForBody[i] = PX_MAX_U32;
      }
    }
  }

  const PxU32 maxActiveNodes = numDynamicBodies + numArticulations + 1;
  PxU32 *bodyRemapTable = reinterpret_cast<PxU32 *>(allocWithFallback(
      mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
      sizeof(PxU32) * maxActiveNodes, "BodyRemapTable"));

  if (!bodyRemapTable) {
    return;
  }

  for (PxU32 i = 0; i < maxActiveNodes; ++i) {
    bodyRemapTable[i] = PX_MAX_U32;
  }

  // Track articulation first link indices
  PxU32 *articulationFirstLinkIndex = nullptr;
  FeatherstoneArticulation **articulationByActiveIdx = nullptr;
  if (numArticulations > 0) {
    articulationFirstLinkIndex = reinterpret_cast<PxU32 *>(allocWithFallback(
        mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
        sizeof(PxU32) * (numArticulations + 1), "ArticulationFirstLinkIndex"));
    articulationByActiveIdx =
        reinterpret_cast<FeatherstoneArticulation **>(allocWithFallback(
            mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
            sizeof(FeatherstoneArticulation *) * (numArticulations + 1),
            "ArticulationByActiveIdx"));

    if (articulationFirstLinkIndex && articulationByActiveIdx) {
      for (PxU32 i = 0; i <= numArticulations; ++i) {
        articulationFirstLinkIndex[i] = PX_MAX_U32;
        articulationByActiveIdx[i] = nullptr;
      }
    }
  }

  // Track per-island info
  struct AvbdIslandInfo {
    PxU32 bodyStart;
    PxU32 bodyCount;
    PxU32 cmStart;
    PxU32 cmCount;
    PxU32 constraintCount;
    PxU32 articulationJointCount;
  };

  AvbdIslandInfo *islandInfos =
      reinterpret_cast<AvbdIslandInfo *>(allocWithFallback(
          mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
          sizeof(AvbdIslandInfo) * islandCount, "IslandInfos"));

  if (!islandInfos) {
    return;
  }

  PxU32 bodyIndex = 0;
  const IG::IslandId *islandIds = islandSim.getActiveIslands();

  // 1. Gather bodies per island (including articulation links)
  for (PxU32 i = 0; i < islandCount && bodyIndex < totalBodyCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];
    info.bodyStart = bodyIndex;
    info.articulationJointCount = 0;

    const IG::Island &island = islandSim.getIsland(islandIds[i]);
    PxNodeIndex currentIndex = island.mRootNode;

    while (currentIndex.isValid() && bodyIndex < totalBodyCount) {
      const IG::Node &node = islandSim.getNode(currentIndex);

      if (node.getNodeType() == IG::Node::eRIGID_BODY_TYPE) {
        rigidBodies[bodyIndex] = getRigidBodyFromIG(islandSim, currentIndex);
        const PxU32 activeNodeIdx = islandSim.getActiveNodeIndex(currentIndex);
        if (activeNodeIdx < maxActiveNodes) {
          bodyRemapTable[activeNodeIdx] = bodyIndex;
        }
        bodyIndex++;
      } else if (node.getNodeType() == IG::Node::eARTICULATION_TYPE) {
        FeatherstoneArticulation *articulation =
            static_cast<FeatherstoneArticulation *>(islandSim.getObject(
                currentIndex, IG::Node::eARTICULATION_TYPE));

        if (articulation) {
          const PxU32 activeNodeIdx =
              islandSim.getActiveNodeIndex(currentIndex);

          // Store first link index and articulation pointer
          if (articulationFirstLinkIndex &&
              activeNodeIdx < numArticulations + 1) {
            articulationFirstLinkIndex[activeNodeIdx] = bodyIndex;
            articulationByActiveIdx[activeNodeIdx] = articulation;
          }

          ArticulationData &artData = articulation->getArticulationData();
          const PxU32 linkCount = artData.getLinkCount();

          for (PxU32 linkIdx = 0;
               linkIdx < linkCount && bodyIndex < totalBodyCount; ++linkIdx) {
            const ArticulationLink &link = artData.getLink(linkIdx);
            AvbdSolverBody &solverBody = avbdBodies[bodyIndex];

            const PxsBodyCore *bodyCore = link.bodyCore;
            if (bodyCore) {
              solverBody.position = bodyCore->body2World.p;
              solverBody.rotation = bodyCore->body2World.q;
              solverBody.linearVelocity = bodyCore->linearVelocity;
              solverBody.angularVelocity = bodyCore->angularVelocity;
              solverBody.invMass = bodyCore->inverseMass;

              PxMat33 R(bodyCore->body2World.q);
              PxMat33 invInertiaLocal =
                  PxMat33::createDiagonal(bodyCore->inverseInertia);
              solverBody.invInertiaWorld =
                  R * invInertiaLocal * R.getTranspose();

              solverBody.nodeIndex = bodyIndex;
              solverBody.colorGroup = 0;

              // Copy per-body damping and velocity caps from body core
              solverBody.linearDamping = bodyCore->linearDamping;
              solverBody.angularDampingBody = bodyCore->angularDamping;
              solverBody.maxLinearVelocitySq = bodyCore->maxLinearVelocitySq;
              solverBody.maxAngularVelocitySq = bodyCore->maxAngularVelocitySq;

              // Store for writeback
              if (articulationForBody && linkIndexForBody) {
                articulationForBody[bodyIndex] = articulation;
                linkIndexForBody[bodyIndex] = linkIdx;
              }
            } else {
              // Fallback: initialize as static
              initializeStaticAvbdBody(PxTransform(PxIdentity), solverBody,
                                       bodyIndex);
            }

            rigidBodies[bodyIndex] = nullptr; // Mark as articulation link
            bodyIndex++;
          }

          // Count internal articulation joints
          info.articulationJointCount += (linkCount > 1) ? (linkCount - 1) : 0;

          // Offset articulation active index by numDynamicBodies to avoid
          // namespace collision -- getActiveNodeIndex() returns per-TYPE indices
          const PxU32 artRemapIdx = numDynamicBodies + activeNodeIdx;
          if (artRemapIdx < maxActiveNodes && articulationFirstLinkIndex) {
            bodyRemapTable[artRemapIdx] =
                articulationFirstLinkIndex[activeNodeIdx];
          }
        }
      }
      currentIndex = node.mNextNode;
    }
    info.bodyCount = bodyIndex - info.bodyStart;
  }

  // 2. Gather contact edges per island
  const PxU32 nbActiveContacts =
      islandSim.getNbActiveEdges(IG::Edge::eCONTACT_MANAGER);
  mContactList.forceSize_Unsafe(0);
  mContactList.reserve((nbActiveContacts + 63u) & (~63u));

  PxU32 contactIndex = 0;
  for (PxU32 i = 0; i < islandCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];
    info.cmStart = contactIndex;

    const IG::Island &island = islandSim.getIsland(islandIds[i]);
    IG::EdgeIndex contactEdgeIndex =
        island.mFirstEdge[IG::Edge::eCONTACT_MANAGER];

    while (contactEdgeIndex != IG_INVALID_EDGE) {
      const IG::Edge &edge = islandSim.getEdge(contactEdgeIndex);
      PxsContactManager *contactManager =
          mIslandManager.getContactManager(contactEdgeIndex);

      if (contactManager) {
        const PxNodeIndex nodeIndex1 =
            islandSim.mCpuData.getNodeIndex1(contactEdgeIndex);
        const PxNodeIndex nodeIndex2 =
            islandSim.mCpuData.getNodeIndex2(contactEdgeIndex);
        const PxcNpWorkUnit &workUnit = contactManager->getWorkUnit();

        mContactList.pushBack(PxsIndexedContactManager(contactManager));
        PxsIndexedContactManager &icm = mContactList.back();

        // Set up body0
        if (!nodeIndex1.isStaticBody()) {
          const PxU32 activeIdx = islandSim.getActiveNodeIndex(nodeIndex1);
          const bool isArt0 = (workUnit.mFlags & PxcNpWorkUnitFlag::eARTICULATION_BODY0) != 0;
          const PxU32 remapIdx0 = isArt0 ? (numDynamicBodies + activeIdx) : activeIdx;
          if (remapIdx0 < maxActiveNodes &&
              bodyRemapTable[remapIdx0] != PX_MAX_U32) {
            // Check if this is an articulation link
            if (isArt0 &&
                articulationByActiveIdx && articulationFirstLinkIndex &&
                activeIdx < numArticulations + 1) {
              // Find the actual link index for this contact
              FeatherstoneArticulation *art =
                  articulationByActiveIdx[activeIdx];
              PxU32 linkIdx =
                  findArticulationLinkIndex(art, workUnit.mRigidCore0);
              if (linkIdx != PX_MAX_U32) {
                icm.indexType0 = PxsIndexedInteraction::eBODY;
                icm.solverBody0 =
                    articulationFirstLinkIndex[activeIdx] + linkIdx;
              } else {
                // Fallback to first link if not found
                icm.indexType0 = PxsIndexedInteraction::eBODY;
                icm.solverBody0 = bodyRemapTable[remapIdx0];
              }
            } else {
              icm.indexType0 = PxsIndexedInteraction::eBODY;
              icm.solverBody0 = bodyRemapTable[remapIdx0];
            }
          } else {
            icm.indexType0 = PxsIndexedInteraction::eWORLD;
            icm.solverBody0 = 0;
          }
        } else {
          icm.indexType0 = PxsIndexedInteraction::eWORLD;
          icm.solverBody0 = 0;
        }

        // Set up body1
        if (nodeIndex2.isStaticBody()) {
          icm.indexType1 = PxsIndexedInteraction::eWORLD;
          icm.solverBody1 = 0;
        } else {
          const PxU32 activeIdx = islandSim.getActiveNodeIndex(nodeIndex2);
          const bool isArt1 = (workUnit.mFlags & PxcNpWorkUnitFlag::eARTICULATION_BODY1) != 0;
          const PxU32 remapIdx1 = isArt1 ? (numDynamicBodies + activeIdx) : activeIdx;
          if (remapIdx1 < maxActiveNodes &&
              bodyRemapTable[remapIdx1] != PX_MAX_U32) {
            // Check if this is an articulation link
            if (isArt1 &&
                articulationByActiveIdx && articulationFirstLinkIndex &&
                activeIdx < numArticulations + 1) {
              // Find the actual link index for this contact
              FeatherstoneArticulation *art =
                  articulationByActiveIdx[activeIdx];
              PxU32 linkIdx =
                  findArticulationLinkIndex(art, workUnit.mRigidCore1);
              if (linkIdx != PX_MAX_U32) {
                icm.indexType1 = PxsIndexedInteraction::eBODY;
                icm.solverBody1 =
                    articulationFirstLinkIndex[activeIdx] + linkIdx;
              } else {
                // Fallback to first link if not found
                icm.indexType1 = PxsIndexedInteraction::eBODY;
                icm.solverBody1 = bodyRemapTable[remapIdx1];
              }
            } else {
              icm.indexType1 = PxsIndexedInteraction::eBODY;
              icm.solverBody1 = bodyRemapTable[remapIdx1];
            }
          } else {
            icm.indexType1 = PxsIndexedInteraction::eWORLD;
            icm.solverBody1 = 0;
          }
        }
        contactIndex++;
      }
      contactEdgeIndex = edge.mNextIslandEdge;
    }
    info.cmCount = contactIndex - info.cmStart;

    // Count constraint joints
    info.constraintCount = 0;
    IG::EdgeIndex constraintEdge = island.mFirstEdge[IG::Edge::eCONSTRAINT];
    while (constraintEdge != IG_INVALID_EDGE) {
      info.constraintCount++;
      constraintEdge = islandSim.getEdge(constraintEdge).mNextIslandEdge;
    }
  }

  // 3. Setup bodies (for rigid bodies only, articulation links already set up)
  if (avbdBodies && rigidBodies) {
    setupBodies(avbdBodies, rigidBodies, bodyIndex, dt, gravity);
  }

  // 4. Initialize Solver
  if (!mSolverInitialized && mAllocatorCallback) {
    AvbdSolverConfig config;
    config.outerIterations = 1;
    config.innerIterations = 4; // Default for contact-only islands; articulations use per-body overrides
    config.initialRho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
    config.maxRho = AvbdConstants::AVBD_MAX_PENALTY_RHO;
    config.enableLocal6x6Solve = true;
    config.contactCompliance = 1e-2f;
    // AVBD reference parameters
    config.avbdAlpha = 0.95f;
    config.avbdBeta = 1000.0f;
    config.avbdGamma = 0.99f;
    config.avbdPenaltyMin = 1000.0f;
    config.avbdPenaltyMax = 1e9f;
    mSolver.initialize(
        config, *reinterpret_cast<PxAllocatorCallback *>(mAllocatorCallback));
    mSolverInitialized = true;
  }

  // 5. Allocate Constraints
  const PxU32 actualContactCount = mContactList.size();
  const PxU32 maxConstraints = actualContactCount * 4;
  AvbdContactConstraint *avbdConstraints = nullptr;
  if (maxConstraints > 0) {
    avbdConstraints =
        reinterpret_cast<AvbdContactConstraint *>(allocWithFallback(
            mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
            sizeof(AvbdContactConstraint) * maxConstraints,
            "AvbdContactConstraint"));
  }

  PxU32 totalJoints = 0;
  PxU32 totalArticulationJoints = 0;
  for (PxU32 i = 0; i < islandCount; ++i) {
    totalJoints += islandInfos[i].constraintCount;
    totalArticulationJoints += islandInfos[i].articulationJointCount;
  }

  PxU32 totalJointCapacity = totalJoints + totalArticulationJoints;
  if (totalJointCapacity == 0)
    totalJointCapacity = 1;

  AvbdD6JointConstraint *d6Joints =
      reinterpret_cast<AvbdD6JointConstraint *>(allocWithFallback(
          mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
          sizeof(AvbdD6JointConstraint) * totalJointCapacity, "D6Joints"));
  AvbdGearJointConstraint *gearJoints =
      reinterpret_cast<AvbdGearJointConstraint *>(allocWithFallback(
          mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
          sizeof(AvbdGearJointConstraint) * totalJointCapacity, "GearJoints"));

  // 6. Create Task Chain
  AvbdWriteBackTask *wbTask = mTaskFactory->createWriteBackTask(
      *this, avbdBodies, rigidBodies, bodyIndex, articulationForBody,
      linkIndexForBody);
  wbTask->setContinuation(continuation);

  AvbdCoordinatorTask *coordTask =
      mTaskFactory->createCoordinatorTask(*this, wbTask);
  coordTask->setContinuation(wbTask);

  PxU32 currentConstraintIdx = 0;
  PxU32 currD6Idx = 0;
  PxU32 currGearIdx = 0;
  PxU32 tasksSpawned = 0;

  // Track color batch allocations for cleanup
  PxArray<AvbdColorBatch *> colorBatchAllocations;

  // 7. Iterate Islands
  for (PxU32 i = 0; i < islandCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];

    // Prepare contact constraints
    PxU32 numConstraints = 0;
    if (avbdConstraints && info.cmCount > 0) {
      numConstraints =
          prepareAvbdContacts(&avbdBodies[info.bodyStart], info.bodyCount,
                              avbdConstraints + currentConstraintIdx,
                              maxConstraints - currentConstraintIdx,
                              info.cmStart, info.cmCount, info.bodyStart);
    }

    PxU32 numD6 = 0;
    PxU32 numGear = 0;

    // Prepare external joint constraints
    if (info.constraintCount > 0) {
      prepareAvbdConstraints(
          islandSim, &avbdBodies[info.bodyStart], info.bodyCount,
          info.bodyStart, d6Joints + currD6Idx, numD6,
          totalJointCapacity - currD6Idx, gearJoints + currGearIdx, numGear,
          totalJointCapacity - currGearIdx, i, bodyRemapTable,
          articulationFirstLinkIndex, articulationByActiveIdx,
          numArticulations);
    }

    // Prepare articulation internal joints
    PxU32 islandArticIterations = 0; // max per-articulation iteration count
    if (info.articulationJointCount > 0 && articulationFirstLinkIndex) {
      const IG::Island &island = islandSim.getIsland(islandIds[i]);
      PxNodeIndex currentNodeIndex = island.mRootNode;

      while (currentNodeIndex.isValid()) {
        const IG::Node &node = islandSim.getNode(currentNodeIndex);

        if (node.getNodeType() == IG::Node::eARTICULATION_TYPE) {
          FeatherstoneArticulation *articulation =
              static_cast<FeatherstoneArticulation *>(islandSim.getObject(
                  currentNodeIndex, IG::Node::eARTICULATION_TYPE));

          if (articulation) {
            // Read per-articulation position iteration count (low byte)
            // Format: high byte = velocityIters, low byte = positionIters
            const PxU16 iterWord = articulation->getIterationCounts();
            const PxU32 posIters = iterWord & 0xFF;
            if (posIters > islandArticIterations)
              islandArticIterations = posIters;

            const PxU32 activeNodeIdx =
                islandSim.getActiveNodeIndex(currentNodeIndex);
            PxU32 artFirstBodyIdx = PX_MAX_U32;
            if (activeNodeIdx < numArticulations + 1) {
              artFirstBodyIdx = articulationFirstLinkIndex[activeNodeIdx];
            }

            if (artFirstBodyIdx >= info.bodyStart &&
                artFirstBodyIdx < info.bodyStart + info.bodyCount) {
              PxU32 localFirstBodyIdx = artFirstBodyIdx - info.bodyStart;

              // Prepare articulation internal joints as unified D6
              PxU32 artD6 = 0, artGear = 0;
              prepareArticulationInternalJoints(
                  articulation, localFirstBodyIdx,
                  d6Joints + currD6Idx + numD6, artD6,
                  totalJointCapacity - currD6Idx - numD6,
                  gearJoints + currGearIdx + numGear, artGear,
                  totalJointCapacity - currGearIdx - numGear,
                  dt);

              numD6 += artD6;
              numGear += artGear;
            }
          }
        }
        currentNodeIndex = node.mNextNode;
      }
    }

    // Skip empty islands
    if (numConstraints == 0 && numD6 == 0 && numGear == 0 &&
        info.bodyCount == 0) {
      continue;
    }

    // Create Island Batch
    AvbdIslandBatch batch;
    batch.bodies = &avbdBodies[info.bodyStart];
    batch.numBodies = info.bodyCount;
    batch.constraints =
        avbdConstraints ? &avbdConstraints[currentConstraintIdx] : nullptr;
    batch.numConstraints = numConstraints;

    batch.d6Joints = &d6Joints[currD6Idx];
    batch.numD6 = numD6;
    batch.gearJoints = &gearJoints[currGearIdx];
    batch.numGear = numGear;

    batch.softParticles = nullptr;
    batch.numSoftParticles = 0;
    batch.softBodies = nullptr;
    batch.numSoftBodies = 0;
    batch.softContacts = nullptr;
    batch.numSoftContacts = 0;

    batch.islandStart = i;
    batch.islandEnd = i + 1;
    batch.iterationOverride = islandArticIterations;
    batch.colorBatches = nullptr;
    batch.numColors = 0;

    // Build constraint-to-body mappings for O(1) lookup in solver
    // This eliminates O(N^2) complexity in the inner loop
    if (batch.numBodies > 0) {
      PxAllocatorCallback &allocator =
          *reinterpret_cast<PxAllocatorCallback *>(mAllocatorCallback);
      if (batch.numConstraints > 0 && batch.constraints) {
        batch.contactMap.build(batch.numBodies, batch.constraints,
                               batch.numConstraints, allocator);
      }
      if (batch.numD6 > 0 && batch.d6Joints) {
        batch.d6Map.build(batch.numBodies, batch.d6Joints, batch.numD6,
                          allocator);
      }
      if (batch.numGear > 0 && batch.gearJoints) {
        batch.gearMap.build(batch.numBodies, batch.gearJoints, batch.numGear,
                            allocator);
      }
    }

    // Constraint coloring for large islands
    const PxU32 largeIslandThreshold = mSolver.getConfig().largeIslandThreshold;
    if (mSolver.getConfig().enableParallelization &&
        numConstraints >= largeIslandThreshold) {
      if (!mConstraintColoring.isInitialized()) {
        mConstraintColoring.initialize(numConstraints, mScratchAdapter);
      }

      PxU32 numColors = mConstraintColoring.colorConstraints(
          batch.constraints, numConstraints, batch.bodies, batch.numBodies);

      if (numColors > 0) {
        batch.colorBatches = static_cast<AvbdColorBatch *>(allocWithFallback(
            mScratchAllocator, mAllocatorCallback, mHeapFallbackAllocations,
            sizeof(AvbdColorBatch) * numColors, "ColorBatches"));

        // Skip coloring if allocation failed
        if (!batch.colorBatches) {
          batch.numColors = 0;
        } else {
          batch.numColors = numColors;

          for (PxU32 c = 0; c < numColors; ++c) {
            const AvbdColorBatch &src = mConstraintColoring.getBatch(c);
            AvbdColorBatch &dst = batch.colorBatches[c];

            dst.numConstraints = src.numConstraints;
            dst.capacity = src.numConstraints;

            if (src.numConstraints > 0) {
              dst.constraintIndices = static_cast<PxU32 *>(allocWithFallback(
                  mScratchAllocator, mAllocatorCallback,
                  mHeapFallbackAllocations, sizeof(PxU32) * src.numConstraints,
                  "ConstraintIndices"));

              if (dst.constraintIndices) {
                memcpy(dst.constraintIndices, src.constraintIndices,
                       sizeof(PxU32) * src.numConstraints);
              } else {
                dst.numConstraints = 0;
                dst.capacity = 0;
              }
            } else {
              dst.constraintIndices = nullptr;
            }
          }
        }
      }
    }

    currentConstraintIdx += numConstraints;
    currD6Idx += numD6;
    currGearIdx += numGear;

    // Fix body nodeIndex: setupBodies() sets global indices, but constraints
    // use island-local indices. Remap so solveLocalSystem() can match bodies
    // to their constraints. Without this, islands after the first one have
    // nodeIndex != local constraint bodyIndex, causing ALL constraints to be
    // skipped and bodies to freefall through the ground.
    for (PxU32 j = 0; j < batch.numBodies; ++j) {
      batch.bodies[j].nodeIndex = j;
    }

    // Spawn Solve Task
#if AVBD_DEBUG_SEQUENTIAL
    // ===== Sequential debug mode: run solver inline (no task parallelism)
    // =====
    {
      const bool hasJoints = (batch.numD6 > 0 || batch.numGear > 0);
      if (hasJoints) {
        mSolver.solveWithJoints(
            dt, batch.bodies, batch.numBodies, batch.constraints,
            batch.numConstraints, batch.d6Joints, batch.numD6,
            batch.gearJoints, batch.numGear, gravity, &batch.contactMap,
            &batch.d6Map, &batch.gearMap, batch.colorBatches,
            batch.numColors, batch.iterationOverride);
      } else {
        mSolver.solve(dt, batch.bodies, batch.numBodies, batch.constraints,
                      batch.numConstraints, gravity, &batch.contactMap,
                      batch.colorBatches, batch.numColors,
                      batch.iterationOverride);
      }
      // Write back lambda cache inline
      writeLambdaToCache(*this, batch.constraints, batch.numConstraints);
      // Release constraint maps
      PxAllocatorCallback &alloc = getAllocator();
      batch.contactMap.release(alloc);
      batch.d6Map.release(alloc);
      batch.gearMap.release(alloc);
    }
#else
    AvbdSolveIslandTask *solveTask =
        mTaskFactory->createSolveTask(*this, mSolver, batch, dt, gravity);
    solveTask->setContinuation(coordTask);
    solveTask->removeReference();
#endif
    tasksSpawned++;
  }

  coordTask->removeReference();
  wbTask->removeReference();

  // NOTE: Do NOT free scratch allocations here!
  // The body arrays (avbdBodies, rigidBodies), constraint arrays, and other
  // data are used by async tasks (AvbdSolveIslandTask, AvbdWriteBackTask).
  // These will be automatically cleaned up when the scratch allocator is reset
  // at frame end by the PhysX simulation framework.
  //
  // The scratch allocator uses a stack-based approach and will be reset via
  // setBlock() at the beginning of the next frame, which handles both
  // scratch memory and any heap fallback allocations.
}

//=============================================================================
// Internal Methods
//=============================================================================

void AvbdDynamicsContext::setupBodies(AvbdSolverBody *avbdBodies,
                                      PxsRigidBody **rigidBodies,
                                      PxU32 numBodies, PxReal dt,
                                      const PxVec3 &gravity) {
  PX_UNUSED(dt);
  PX_UNUSED(gravity);

  for (PxU32 i = 0; i < numBodies; ++i) {
    PxsRigidBody *rigidBody = rigidBodies[i];
    if (rigidBody) {
      const PxsBodyCore &core = rigidBody->getCore();
      copyToAvbdSolverBody(core, avbdBodies[i], i);
    }
  }
}

void AvbdDynamicsContext::writeBackBodies(AvbdSolverBody *avbdBodies,
                                          PxsRigidBody **rigidBodies,
                                          PxU32 numBodies) {
  for (PxU32 i = 0; i < numBodies; ++i) {
    PxsRigidBody *rigidBody = rigidBodies[i];
    if (rigidBody && !avbdBodies[i].isStatic()) {
      PxsBodyCore &core = rigidBody->getCore();
      writeBackAvbdSolverBody(avbdBodies[i], core);
    }
  }
}



//=============================================================================
// Articulation Internal Joints Preparation (Multi-Type)
//=============================================================================

static void prepareArticulationInternalJoints(
    FeatherstoneArticulation *articulation, PxU32 firstBodyIndex,
    AvbdD6JointConstraint *d6Constraints, PxU32 &numD6, PxU32 maxD6,
    AvbdGearJointConstraint *gearConstraints, PxU32 &numGear, PxU32 maxGear,
    PxReal dt) {

  PX_UNUSED(gearConstraints);
  PX_UNUSED(maxGear);

  numD6 = 0;
  numGear = 0;

  if (!articulation)
    return;

  ArticulationData &artData = articulation->getArticulationData();
  const PxU32 linkCount = artData.getLinkCount();

  if (linkCount <= 1)
    return;

  for (PxU32 linkIdx = 1; linkIdx < linkCount; ++linkIdx) {
    const ArticulationLink &link = artData.getLink(linkIdx);
    const PxU32 parentIdx = link.parent;

    if (parentIdx == DY_ARTICULATION_LINK_NONE)
      continue;

    ArticulationJointCore *jointCore = link.inboundJoint;
    if (!jointCore)
      continue;

    const PxsBodyCore *parentBodyCore = artData.getLink(parentIdx).bodyCore;
    const PxsBodyCore *childBodyCore = link.bodyCore;

    if (!parentBodyCore || !childBodyCore)
      continue;

    const PxVec3 anchorInParent = jointCore->parentPose.p;
    const PxVec3 anchorInChild = jointCore->childPose.p;
    const PxU32 bodyIndexA = firstBodyIndex + parentIdx;
    const PxU32 bodyIndexB = firstBodyIndex + linkIdx;

    // Create a D6 constraint to represent the articulation joint
    if (numD6 < maxD6 && d6Constraints) {
      AvbdD6JointConstraint &c = d6Constraints[numD6];
      c.initDefaults();

      c.header.bodyIndexA = bodyIndexA;
      c.header.bodyIndexB = bodyIndexB;

      // Use mass-proportional rho to avoid overwhelming contact penalties.
      // With fixed rho=2e6 and contact penalty ~m/h^2, the 100:1 imbalance
      // causes block-descent to ignore contacts, leading to base drift.
      const PxReal artInvDt = (dt > 0.0f) ? (1.0f / dt) : 60.0f;
      const PxReal artInvDt2 = artInvDt * artInvDt;
      const PxReal massA_art = parentBodyCore->inverseMass > 0.0f
          ? 1.0f / parentBodyCore->inverseMass : 100.0f;
      const PxReal massB_art = childBodyCore->inverseMass > 0.0f
          ? 1.0f / childBodyCore->inverseMass : 100.0f;
      const PxReal massMax = PxMax(massA_art, massB_art);
      c.header.rho = PxMax(10.0f * massMax * artInvDt2, 1e5f);
      c.header.compliance = 0.0f;
      c.header.damping = AvbdConstants::AVBD_CONSTRAINT_DAMPING;

      c.anchorA = anchorInParent;
      c.anchorB = anchorInChild;

      c.localFrameA = jointCore->parentPose.q;
      c.localFrameB = jointCore->childPose.q;

      // Translate articulation motion limits to D6 limit bits
      // 2-bit-per-axis encoding: bits[1:0]=axisX, bits[3:2]=axisY,
      // bits[5:4]=axisZ.  Values: 0=LOCKED, 1=LIMITED, 2=FREE.
      c.linearMotion = 0;
      if (jointCore->motion[PxArticulationAxis::eX] ==
          PxArticulationMotion::eLIMITED)
        c.linearMotion |= (1u << 0);
      else if (jointCore->motion[PxArticulationAxis::eX] ==
               PxArticulationMotion::eFREE)
        c.linearMotion |= (2u << 0);
      if (jointCore->motion[PxArticulationAxis::eY] ==
          PxArticulationMotion::eLIMITED)
        c.linearMotion |= (1u << 2);
      else if (jointCore->motion[PxArticulationAxis::eY] ==
               PxArticulationMotion::eFREE)
        c.linearMotion |= (2u << 2);
      if (jointCore->motion[PxArticulationAxis::eZ] ==
          PxArticulationMotion::eLIMITED)
        c.linearMotion |= (1u << 4);
      else if (jointCore->motion[PxArticulationAxis::eZ] ==
               PxArticulationMotion::eFREE)
        c.linearMotion |= (2u << 4);

      // Set limits
      c.linearLimitLower = PxVec3(0.0f);
      c.linearLimitUpper = PxVec3(0.0f);
      c.angularLimitLower = PxVec3(0.0f);
      c.angularLimitUpper = PxVec3(0.0f);

      if (jointCore->motion[PxArticulationAxis::eX] ==
          PxArticulationMotion::eLIMITED) {
        c.linearLimitLower.x = jointCore->limits[PxArticulationAxis::eX].low;
        c.linearLimitUpper.x = jointCore->limits[PxArticulationAxis::eX].high;
      }
      if (jointCore->motion[PxArticulationAxis::eY] ==
          PxArticulationMotion::eLIMITED) {
        c.linearLimitLower.y = jointCore->limits[PxArticulationAxis::eY].low;
        c.linearLimitUpper.y = jointCore->limits[PxArticulationAxis::eY].high;
      }
      if (jointCore->motion[PxArticulationAxis::eZ] ==
          PxArticulationMotion::eLIMITED) {
        c.linearLimitLower.z = jointCore->limits[PxArticulationAxis::eZ].low;
        c.linearLimitUpper.z = jointCore->limits[PxArticulationAxis::eZ].high;
      }

      c.angularMotion = 0;
      if (jointCore->motion[PxArticulationAxis::eTWIST] ==
          PxArticulationMotion::eLIMITED)
        c.angularMotion |= (1u << 0);
      else if (jointCore->motion[PxArticulationAxis::eTWIST] ==
               PxArticulationMotion::eFREE)
        c.angularMotion |= (2u << 0);
      if (jointCore->motion[PxArticulationAxis::eSWING1] ==
          PxArticulationMotion::eLIMITED)
        c.angularMotion |= (1u << 2);
      else if (jointCore->motion[PxArticulationAxis::eSWING1] ==
               PxArticulationMotion::eFREE)
        c.angularMotion |= (2u << 2);
      if (jointCore->motion[PxArticulationAxis::eSWING2] ==
          PxArticulationMotion::eLIMITED)
        c.angularMotion |= (1u << 4);
      else if (jointCore->motion[PxArticulationAxis::eSWING2] ==
               PxArticulationMotion::eFREE)
        c.angularMotion |= (2u << 4);

      if (jointCore->motion[PxArticulationAxis::eTWIST] ==
          PxArticulationMotion::eLIMITED) {
        c.angularLimitLower.x =
            jointCore->limits[PxArticulationAxis::eTWIST].low;
        c.angularLimitUpper.x =
            jointCore->limits[PxArticulationAxis::eTWIST].high;
      }
      if (jointCore->motion[PxArticulationAxis::eSWING1] ==
          PxArticulationMotion::eLIMITED) {
        c.angularLimitLower.y =
            jointCore->limits[PxArticulationAxis::eSWING1].low;
        c.angularLimitUpper.y =
            jointCore->limits[PxArticulationAxis::eSWING1].high;
      }
      if (jointCore->motion[PxArticulationAxis::eSWING2] ==
          PxArticulationMotion::eLIMITED) {
        c.angularLimitLower.z =
            jointCore->limits[PxArticulationAxis::eSWING2].low;
        c.angularLimitUpper.z =
            jointCore->limits[PxArticulationAxis::eSWING2].high;
      }

      // ---------------------------------------------------------------
      // Boost penalty for fully-locked joints (eFIX equivalent) so they
      // can resist drive forces transmitted through cross-links.
      // ---------------------------------------------------------------
      if (c.linearMotion == 0 && c.angularMotion == 0) {
        c.header.rho = PxMax(c.header.rho,
                             AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH);
      }

      // ---------------------------------------------------------------
      // Copy articulation drive parameters to D6 drive fields
      //
      // The drive uses position-error (targetP - currentQ) rather than
      // the raw target, matching the standalone articulation solver.
      // This prevents the drive from applying full-position displacement
      // each step which would overpower fixed joints.
      //   v_target = stiffness/(S+D) * (targetP - currentQ)/dt
      //            + damping/(S+D)  * targetV
      //   rho_drive = (S+D) / dt^2   (capped)
      // ---------------------------------------------------------------
      const PxReal maxDriveStiffness = 100.0f;
      const PxReal invDt = (dt > 0.0f) ? (1.0f / dt) : 60.0f;

      // Precompute world-space anchor separation and joint frame for
      // position-error drives.
      const PxVec3 worldAnchorA =
          parentBodyCore->body2World.transform(anchorInParent);
      const PxVec3 worldAnchorB =
          childBodyCore->body2World.transform(anchorInChild);
      const PxVec3 anchorSep = worldAnchorB - worldAnchorA;
      const PxQuat worldFrameA_drive =
          parentBodyCore->body2World.q * jointCore->parentPose.q;

      // Angular position error: relative rotation in joint frame.
      const PxQuat worldFrameB_drive =
          childBodyCore->body2World.q * jointCore->childPose.q;
      PxQuat relRotDrive =
          worldFrameA_drive.getConjugate() * worldFrameB_drive;
      if (relRotDrive.w < 0.0f)
        relRotDrive = -relRotDrive;

      // Linear drives (eX, eY, eZ)
      {
        const PxArticulationAxis::Enum linAxes[3] = {
            PxArticulationAxis::eX, PxArticulationAxis::eY,
            PxArticulationAxis::eZ};
        for (int a = 0; a < 3; ++a) {
          const PxArticulationDrive &drive = jointCore->drives[linAxes[a]];
          if (drive.driveType == PxArticulationDriveType::eNONE)
            continue;
          PxReal totalSD =
              PxMin(drive.stiffness + drive.damping, maxDriveStiffness);
          if (totalSD <= 0.0f)
            continue;
          c.driveFlags |= (1u << a); // bit 0=X, 1=Y, 2=Z
          (&c.linearDamping.x)[a] = totalSD;

          // Compute current joint displacement along the driven axis
          PxVec3 localAxis(0.0f);
          (&localAxis.x)[a] = 1.0f;
          PxVec3 worldAxis = worldFrameA_drive.rotate(localAxis);
          PxReal currentQ = anchorSep.dot(worldAxis);

          // Position-error spring: drive toward (targetP - currentQ)
          PxReal targetP = jointCore->targetP[linAxes[a]];
          PxReal posVel = (targetP - currentQ) * invDt;
          PxReal velVel = jointCore->targetV[linAxes[a]];
          PxReal invSD = 1.0f / totalSD;
          PxReal sClamped = PxMin(drive.stiffness, maxDriveStiffness);
          PxReal dClamped = totalSD - sClamped;
          (&c.driveLinearVelocity.x)[a] =
              sClamped * invSD * posVel + dClamped * invSD * velVel;
        }
      }

      // Angular drives (eTWIST, eSWING1, eSWING2)
      {
        const PxArticulationAxis::Enum angAxes[3] = {
            PxArticulationAxis::eTWIST, PxArticulationAxis::eSWING1,
            PxArticulationAxis::eSWING2};
        // Drive flags: bit3=twist, bit4=swing1, bit5=swing2
        const PxU32 angBits[3] = {(1u << 3), (1u << 4), (1u << 5)};
        for (int a = 0; a < 3; ++a) {
          const PxArticulationDrive &drive = jointCore->drives[angAxes[a]];
          if (drive.driveType == PxArticulationDriveType::eNONE)
            continue;
          PxReal totalSD =
              PxMin(drive.stiffness + drive.damping, maxDriveStiffness);
          if (totalSD <= 0.0f)
            continue;
          c.driveFlags |= angBits[a];
          (&c.angularDamping.x)[a] = totalSD;

          // Compute current joint angle for the driven axis.
          // For twist (a=0) use atan2(x,w); for swings approximate
          // with atan2(y,w) / atan2(z,w).
          PxReal currentAngle;
          if (a == 0)
            currentAngle = 2.0f * PxAtan2(relRotDrive.x, relRotDrive.w);
          else if (a == 1)
            currentAngle = 2.0f * PxAtan2(relRotDrive.y, relRotDrive.w);
          else
            currentAngle = 2.0f * PxAtan2(relRotDrive.z, relRotDrive.w);

          PxReal targetAng = jointCore->targetP[angAxes[a]];
          PxReal posVel = (targetAng - currentAngle) * invDt;
          PxReal velVel = jointCore->targetV[angAxes[a]];
          PxReal invSD = 1.0f / totalSD;
          PxReal sClamped = PxMin(drive.stiffness, maxDriveStiffness);
          PxReal dClamped = totalSD - sClamped;
          (&c.driveAngularVelocity.x)[a] =
              sClamped * invSD * posVel + dClamped * invSD * velVel;
        }
      }

      numD6++;
    }
  }
}

void AvbdDynamicsContext::solveIsland(const IG::IslandSim &islandSim, PxReal dt,
                                      const PxVec3 &gravity) {
  PX_UNUSED(islandSim);
  PX_UNUSED(dt);
  PX_UNUSED(gravity);
}

void AvbdDynamicsContext::prepareContacts(const IG::IslandSim &islandSim) {
  PX_UNUSED(islandSim);
}

void AvbdDynamicsContext::mergeResults() {
  // Clean up any heap fallback allocations from this frame
  // No mutex needed since mergeResults() is called from a single thread
  // context
  if (mHeapFallbackAllocations.size() > 0 && mAllocatorCallback) {
    PxAllocatorCallback *allocator =
        reinterpret_cast<PxAllocatorCallback *>(mAllocatorCallback);
    for (PxU32 i = 0; i < mHeapFallbackAllocations.size(); ++i) {
      if (mHeapFallbackAllocations[i]) {
        allocator->deallocate(mHeapFallbackAllocations[i]);
      }
    }
    mHeapFallbackAllocations.clear();
  }
}

void AvbdDynamicsContext::setSimulationController(
    PxsSimulationController *simulationController) {
  mSimulationController = simulationController;
}

Dy::Context *Dy::createAVBDDynamicsContext(
    PxcNpMemBlockPool *memBlockPool, PxcScratchAllocator &scratchAllocator,
    Cm::FlushPool &taskPool, PxvSimStats &simStats, PxTaskManager *taskManager,
    PxVirtualAllocatorCallback *allocatorCallback,
    PxsMaterialManager *materialManager, IG::SimpleIslandManager &islandManager,
    PxU64 contextID, bool enableStabilization, bool useEnhancedDeterminism,
    bool solveArticulationContactLast, PxReal maxBiasCoefficient,
    bool frictionEveryIteration, PxReal lengthScale,
    bool isResidualReportingEnabled) {
  return PX_PLACEMENT_NEW(
      PX_ALLOC(sizeof(Dy::AvbdDynamicsContext), "AvbdDynamicsContext"),
      Dy::AvbdDynamicsContext)(
      memBlockPool, scratchAllocator, taskPool, simStats, taskManager,
      allocatorCallback, materialManager, islandManager, contextID,
      enableStabilization, useEnhancedDeterminism, solveArticulationContactLast,
      maxBiasCoefficient, frictionEveryIteration, lengthScale,
      isResidualReportingEnabled);
}
