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
#include "DyArticulationMimicJointCore.h"
#include "DyArticulationTendon.h"
#include "DyAvbdBodyConversion.h"
#include "DyAvbdConstraint.h"
#include "DyAvbdTasks.h"
#include "DyConstraint.h"
#include "DyConstraintPrep.h"
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

#include <cstdlib>
#include <cstdint>
#include <cstring>

using namespace physx;
using namespace physx::Dy;

// Global frame counter for motor deduplication
// This is incremented at the start of each update() call
static physx::PxU64 gAvbdMotorFrameCounter = 0;

static PX_FORCE_INLINE PxU32 getJointLambdaCacheIndex(PxU64 key,
                                                      PxU32 cacheSize) {
  const PxU64 mixed = key ^ (key >> 33) ^ (key >> 17);
  return cacheSize ? static_cast<PxU32>(mixed % cacheSize) : 0;
}

static PX_FORCE_INLINE PxU32 getBodyVelocityHistoryCacheIndex(PxU64 key,
                                                              PxU32 cacheSize) {
  const PxU64 mixed = key ^ (key >> 33) ^ (key >> 17);
  return cacheSize ? static_cast<PxU32>(mixed % cacheSize) : 0;
}

static bool isEnvFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] && value[0] != '0';
}

static PX_FORCE_INLINE void mixAvbdContactManagerKey(PxU64 &hash, PxU32 value) {
  hash ^= value;
  hash *= 1099511628211ull;
}

static PxU64 makeAvbdContactManagerKey(const PxsContactManager *manager,
                                       PxU32 globalBody0Idx,
                                       PxU32 globalBody1Idx) {
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(manager);
  PxU64 hash = 14695981039346656037ull;
  mixAvbdContactManagerKey(hash, static_cast<PxU32>(address));
  mixAvbdContactManagerKey(
      hash, static_cast<PxU32>(static_cast<PxU64>(address) >> 32));
  mixAvbdContactManagerKey(hash, globalBody0Idx);
  mixAvbdContactManagerKey(hash, globalBody1Idx);
  return hash != 0 ? hash : 1;
}

// Accessor for the solver to get current frame
physx::PxU64 getAvbdMotorFrameCounter() { return gAvbdMotorFrameCounter; }

//=============================================================================
// Articulation Internal Joints Helper (forward declaration)
//=============================================================================
static void prepareArticulationInternalJoints(
  AvbdDynamicsContext &context, FeatherstoneArticulation *articulation,
  PxU32 firstBodyIndex,
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
                               PxAllocatorCallback &mainAllocator,
                               PxArray<void *> &fallbackAllocations, PxU32 size,
                               const char *name) {

  // First try scratch allocator (no heap fallback) - fast path
  void *ptr = scratch.alloc(size, false);
  if (ptr) {
    return ptr;
  }

  // Scratch memory exhausted - use main allocator (slower path)
  ptr = mainAllocator.allocate(size, name, __FILE__, __LINE__);
  if (ptr) {
    // Track for cleanup in mergeResults()
    // No mutex needed here since update() is called from a single thread
    // context
    fallbackAllocations.pushBack(ptr);
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

AvbdDynamicsContext::VirtualAllocatorAdapter::VirtualAllocatorAdapter(
    Cm::VirtualAllocatorCallback &allocator)
    : mAllocator(allocator) {}

void *AvbdDynamicsContext::VirtualAllocatorAdapter::allocate(
    size_t size, const char *, const char *file, int line) {
  return mAllocator.allocate(size, PxsHeapStats::eSOLVER, file, line);
}

void AvbdDynamicsContext::VirtualAllocatorAdapter::deallocate(void *ptr) {
  mAllocator.deallocate(ptr);
}

//=============================================================================
// Constructor / Destructor
//=============================================================================

AvbdDynamicsContext::AvbdDynamicsContext(
    PxcNpMemBlockPool *memBlockPool, PxcScratchAllocator &scratchAllocator,
    Cm::FlushPool &taskPool, PxvSimStats &simStats, PxTaskManager *taskManager,
    Cm::VirtualAllocatorCallback &allocator,
    PxsMaterialManager *materialManager, IG::SimpleIslandManager &islandManager,
    PxU64 contextID, PxReal maxBiasCoefficient, PxReal lengthScale,
    PxU32 avbdIterations, PxU32 avbdJointIterationOverride,
    bool avbdEnableEarlyStop, PxSceneFlags sceneFlags)
    : DynamicsContextBase(memBlockPool, taskPool, simStats, allocator,
                          materialManager, islandManager, contextID,
                          maxBiasCoefficient, lengthScale, sceneFlags),
      mScratchAllocator(scratchAllocator), mSoftIslandProvider(nullptr),
      mRigidGpuWaveBackend(nullptr),
      mAvbdIterations(avbdIterations),
      mAvbdJointIterationOverride(avbdJointIterationOverride),
      mAvbdEnableEarlyStop(avbdEnableEarlyStop),
      mScratchAdapter(scratchAllocator), mAllocatorAdapter(allocator),
      mTaskManager(taskManager),
      mTaskGraphSerialMode(
          isEnvFlagEnabled("PHYSX_AVBD_TASKGRAPH_SERIAL")),
      mFrictionEveryIteration(
          sceneFlags & PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION) {
  PX_ASSERT(mAvbdIterations > 0 && mAvbdIterations <= 255);
  PX_ASSERT(mAvbdJointIterationOverride <= 255);
  mSolverInitialized = false;

  // Use the main allocator callback for tasks, NOT the scratch adapter.
  // Tasks need explicit deallocation, which ScratchAllocatorAdapter doesn't
  // provide.
  mTaskFactory = new AvbdTaskFactory(mTaskManager, mAllocatorAdapter);

  // Initialize lambda warm-starting cache
  mEnableLambdaWarmStart = true;
  // Small-scene seed; update() grows this to cover every active contact
  // manager before any island task can access the backing storage.
  mLambdaCache.resize(4096);
  memset(mLambdaCache.begin(), 0, sizeof(CachedLambda) * mLambdaCache.size());
  mJointLambdaCache.resize(JOINT_LAMBDA_CACHE_SIZE);
  memset(mJointLambdaCache.begin(), 0,
         sizeof(CachedJointLambda) * mJointLambdaCache.size());
  mBodyVelocityHistoryCache.resize(BODY_VELOCITY_HISTORY_CACHE_SIZE);
  memset(mBodyVelocityHistoryCache.begin(), 0,
         sizeof(CachedBodyVelocityHistory) *
             mBodyVelocityHistoryCache.size());
  mBodyVelocityHistoryFrame = 0;
}

void AvbdDynamicsContext::restoreAndUpdateBodyVelocityHistory(
    const PxsBodyCore &bodyCore, AvbdSolverBody &solverBody) {
  const PxU64 bodyCoreKey = reinterpret_cast<PxU64>(&bodyCore);
  const PxU32 cacheSize = mBodyVelocityHistoryCache.size();
  if (cacheSize == 0)
    return;

  const PxU32 firstIndex =
      getBodyVelocityHistoryCacheIndex(bodyCoreKey, cacheSize);
  CachedBodyVelocityHistory *cached = NULL;
  CachedBodyVelocityHistory *firstReusable = NULL;

  // Gather is serial.  Resolve pointer-hash collisions instead of allowing
  // one body to evict another active body's previous-frame velocity.  A slot
  // absent for at least one complete update is reusable, but remains a
  // tombstone while the exact key may still exist later in the probe chain.
  for (PxU32 probe = 0; probe < cacheSize; ++probe) {
    CachedBodyVelocityHistory &candidate =
        mBodyVelocityHistoryCache[(firstIndex + probe) % cacheSize];
    if (candidate.bodyCoreKey == bodyCoreKey) {
      cached = &candidate;
      break;
    }
    if (candidate.bodyCoreKey == 0) {
      cached = firstReusable ? firstReusable : &candidate;
      break;
    }
    if (!firstReusable &&
        candidate.lastSeenFrame + 1 < mBodyVelocityHistoryFrame) {
      firstReusable = &candidate;
    }
  }
  if (!cached)
    cached = firstReusable;
  if (!cached)
    return;

  // initialize()/copyToAvbdSolverBody() deliberately remain the source of all
  // current-frame state.  Only replace the history sample, and only when the
  // exact body was gathered in the immediately preceding update.  A sleeping
  // or otherwise absent frame therefore falls back to full initialization.
  if (cached->bodyCoreKey == bodyCoreKey &&
      cached->lastSeenFrame + 1 == mBodyVelocityHistoryFrame) {
    solverBody.prevLinearVelocity = cached->linearVelocity;
    solverBody.projectLockedLinearVector(solverBody.prevLinearVelocity);
  }

  // Gather is serial, so it is safe to prepare next frame's history here.
  cached->bodyCoreKey = bodyCoreKey;
  cached->lastSeenFrame = mBodyVelocityHistoryFrame;
  cached->linearVelocity = bodyCore.linearVelocity;
}

void AvbdDynamicsContext::ensureBodyVelocityHistoryCapacity(
    PxU32 bodyCount) {
  const PxU64 required64 = PxU64(bodyCount) * 2u;
  if (required64 <= mBodyVelocityHistoryCache.size() ||
      required64 > PX_MAX_U32)
    return;

  PxU32 newCapacity = mBodyVelocityHistoryCache.size();
  while (newCapacity < required64 && newCapacity <= PX_MAX_U32 / 2u)
    newCapacity *= 2u;
  if (newCapacity < required64)
    return;

  PxArray<CachedBodyVelocityHistory> previous(
      mBodyVelocityHistoryCache);
  mBodyVelocityHistoryCache.resize(newCapacity);
  memset(mBodyVelocityHistoryCache.begin(), 0,
         sizeof(CachedBodyVelocityHistory) *
             mBodyVelocityHistoryCache.size());

  // Preserve exact previous-frame samples while rebuilding the probe table.
  // Growth keeps load at or below 0.5, so every retained key has an empty
  // destination.
  for (PxU32 i = 0; i < previous.size(); ++i) {
    const CachedBodyVelocityHistory &source = previous[i];
    if (source.bodyCoreKey == 0)
      continue;
    const PxU32 firstIndex = getBodyVelocityHistoryCacheIndex(
        source.bodyCoreKey, newCapacity);
    for (PxU32 probe = 0; probe < newCapacity; ++probe) {
      CachedBodyVelocityHistory &destination =
          mBodyVelocityHistoryCache[(firstIndex + probe) % newCapacity];
      if (destination.bodyCoreKey == 0) {
        destination = source;
        break;
      }
    }
  }
}

AvbdDynamicsContext::~AvbdDynamicsContext() {
  delete mTaskFactory;

  if (mSolverInitialized) {
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
                        PxU32 numConstraints, PxU32 numBodies) {
  PX_UNUSED(numBodies);
  if (!ctx.mEnableLambdaWarmStart || !constraints || numConstraints == 0) {
    return;
  }

  PxArray<AvbdDynamicsContext::CachedLambda> &cache = ctx.mLambdaCache;
  const PxU16 cacheFrameStamp = ctx.getAvbdFrameStamp();

  for (PxU32 i = 0; i < numConstraints; ++i) {
    const AvbdContactConstraint &constraint = constraints[i];
    const PxU32 cacheIdx = constraint.cacheIndex;

    // Skip invalid cache indices
    if (cacheIdx >= cache.size()) {
      continue;
    }

    AvbdDynamicsContext::CachedLambda &cached = cache[cacheIdx];

    // Dual warmstart policy:
    //   deformable mesh NP anchor -> no cross-frame lambda / staticPrev
    //   rigid plane / dyn-dyn -> write dual state
    // Entry 108/153: CM-index staticPrev aliasing caused long-run heave energy.
    if (hasDeformableStaticAnchor(constraint)) {
      cached.key = 0;
      cached.lambda = 0.0f;
      cached.tangentLambda0 = 0.0f;
      cached.tangentLambda1 = 0.0f;
      cached.penalty = 1000.0f;
      cached.tangentPenalty0 = 1000.0f;
      cached.tangentPenalty1 = 1000.0f;
      cached.stick = 0;
      cached.prevStaticWorldPoint = PxVec3(0.0f);
      // Invalidate the stamp so a non-deformable pair reusing this CM index
      // does not inherit mesh dual garbage if the pair type changes.
      cached.frameStamp = 0;
      continue;
    }

    // Rigid / dyn-dyn contacts: standard dual warmstart write-back.
    cached.key = constraint.cacheKey;
    cached.lambda = constraint.header.lambda;
    cached.tangentLambda0 = constraint.tangentLambda0;
    cached.tangentLambda1 = constraint.tangentLambda1;
    cached.penalty = constraint.header.penalty;
    cached.tangentPenalty0 = constraint.tangentPenalty0;
    cached.tangentPenalty1 = constraint.tangentPenalty1;
    cached.stick = hasFrictionStick(constraint) ? 1u : 0u;
    cached.frameStamp = cacheFrameStamp;
  }
}

void writeContactImpulseToOutput(const AvbdContactConstraint *constraints,
                                 PxU32 numConstraints, PxU32 numBodies,
                                 PxReal dt) {
  if (!constraints || numConstraints == 0 || dt <= 0.0f)
    return;

  for (PxU32 i = 0; i < numConstraints; ++i) {
    const AvbdContactConstraint &constraint = constraints[i];

    // PxContactPairPoint::impulse is a scalar normal impulse expanded along
    // the contact normal by extractContacts(). AVBD's unilateral normal
    // multiplier uses the opposite sign: compression is lambda < 0.
    if (constraint.contactImpulseWriteback) {
      *constraint.contactImpulseWriteback =
          constraint.velocityNormalImpulse >= 0.0f
              ? constraint.velocityNormalImpulse
              : PxMax(0.0f, -constraint.header.lambda) * dt;
    }

    if (constraint.frictionImpulseWriteback) {
      PxVec3 frictionImpulse = constraint.frictionSweepImpulse;
      const bool dynamicDynamic =
          constraint.header.bodyIndexA < numBodies &&
          constraint.header.bodyIndexB < numBodies;
      if (dynamicDynamic) {
        // Dynamic-dynamic tangents are part of the AVBD primal/dual solve.
        // Their multipliers are forces, matching the normal multiplier, so
        // convert the final force on body A to a per-step impulse.
        frictionImpulse +=
            (constraint.tangent0 * constraint.tangentLambda0 +
             constraint.tangent1 * constraint.tangentLambda1) *
            dt;
      }
      if (frictionImpulse.isFinite())
        *constraint.frictionImpulseWriteback += frictionImpulse;
    }
  }
}

void writeContactImpulseToOutputTokens(
    const AvbdContactConstraint *constraints,
    const AvbdContactOutputToken *tokens,
    AvbdContactOutputResult *results, PxU32 numConstraints, PxU32 numBodies,
    PxReal dt) {
  if (!constraints || !tokens || !results || numConstraints == 0 ||
      dt <= 0.0f)
    return;

  for (PxU32 i = 0; i < numConstraints; ++i) {
    const AvbdContactOutputToken &token = tokens[i];
    if ((token.flags & (AvbdContactOutputToken::eNORMAL_IMPULSE |
                       AvbdContactOutputToken::eFRICTION_IMPULSE)) == 0 ||
        token.targetIndex == PX_MAX_U32)
      continue;

    const AvbdContactConstraint &constraint = constraints[i];
    AvbdContactOutputResult &result = results[token.targetIndex];
    if ((token.flags & AvbdContactOutputToken::eNORMAL_IMPULSE) != 0) {
      result.normalImpulse =
          constraint.velocityNormalImpulse >= 0.0f
              ? constraint.velocityNormalImpulse
              : PxMax(0.0f, -constraint.header.lambda) * dt;
      result.flags |= AvbdContactOutputToken::eNORMAL_IMPULSE;
    }

    if ((token.flags & AvbdContactOutputToken::eFRICTION_IMPULSE) != 0) {
      PxVec3 frictionImpulse = constraint.frictionSweepImpulse;
      const bool dynamicDynamic =
          constraint.header.bodyIndexA < numBodies &&
          constraint.header.bodyIndexB < numBodies;
      if (dynamicDynamic) {
        frictionImpulse +=
            (constraint.tangent0 * constraint.tangentLambda0 +
             constraint.tangent1 * constraint.tangentLambda1) *
            dt;
      }
      if (frictionImpulse.isFinite()) {
        result.frictionImpulse += frictionImpulse;
        result.flags |= AvbdContactOutputToken::eFRICTION_IMPULSE;
      }
    }
  }
}

void commitContactOutputTokens(const AvbdContactOutputTarget *targets,
                               AvbdContactOutputResult *results,
                               PxU32 count) {
  if (!targets || !results)
    return;
  for (PxU32 i = 0; i < count; ++i) {
    if ((results[i].flags & AvbdContactOutputToken::eNORMAL_IMPULSE) != 0 &&
        targets[i].normalImpulse)
      *targets[i].normalImpulse = results[i].normalImpulse;
    if ((results[i].flags & AvbdContactOutputToken::eFRICTION_IMPULSE) != 0 &&
        targets[i].frictionImpulse)
      *targets[i].frictionImpulse += results[i].frictionImpulse;
  }
}

void restoreJointLambdaFromCache(AvbdDynamicsContext &ctx,
                                 AvbdD6JointConstraint &constraint,
                                 PxU64 cacheKey) {
  constraint.cacheIndex = PX_MAX_U32;

  if (!ctx.mEnableLambdaWarmStart || cacheKey == 0 ||
      ctx.mJointLambdaCache.empty()) {
    return;
  }

  const PxU32 cacheIdx =
      getJointLambdaCacheIndex(cacheKey, ctx.mJointLambdaCache.size());
  constraint.cacheIndex = cacheIdx;
    constraint.cacheKey = cacheKey;

  AvbdDynamicsContext::CachedJointLambda &cached =
      ctx.mJointLambdaCache[cacheIdx];
  const PxU16 currentStamp = ctx.getAvbdFrameStamp();
  const PxU16 frameAge =
      static_cast<PxU16>(currentStamp - cached.frameStamp);
  if (cached.key != cacheKey || cached.frameStamp == 0 ||
      frameAge > AvbdDynamicsContext::LAMBDA_MAX_AGE) {
    return;
  }

  const PxReal warmScale = 0.95f * 0.99f;
  constraint.lambdaLinear = cached.lambdaLinear * warmScale;
  constraint.lambdaAngular = cached.lambdaAngular * warmScale;
  constraint.lambdaDriveLinear = cached.lambdaDriveLinear * warmScale;
  constraint.lambdaDriveAngular = cached.lambdaDriveAngular * warmScale;
  constraint.coneLambda = cached.coneLambda * warmScale;

  // Native PxRevoluteJoint: clear the hinge swing-lock warmstart rows so
  // they do not accumulate cross-frame and amplify the SnippetJoint
  // revolute chain's late-window swing burst (AVBD audit Entry 093).
  //
  // Match strictly on the original native joint type instead of the
  // motion-mask shape, because articulation internal joints created by
  // prepareArticulationInternalJoints() share the same all-linear-locked /
  // single-angular-movable mask and they need to keep their warmstart for
  // the articulation chain to remain stable.
  if (constraint.header.type == AvbdConstraintType::eJOINT_REVOLUTE) {
    constraint.lambdaAngular.y = 0.0f;
    constraint.lambdaAngular.z = 0.0f;
  }

  // Reseed locked/limited row lambdas every frame for joint patterns where
  // cross-frame accumulation pushes the AVBD dual integrator (lambda <-
  // 0.99*lambda + rho_dual*C, 10 iters) into a positive-feedback regime.
  //
  // 1. Breakable joints (SnippetJoint's fixed chain et al.): the writeback
  //    in DyAvbdTasks.cpp compares lambda directly against the authored
  //    break force. With warmScale~=0.94 the steady-state lambda ends up
  //    an order of magnitude above the physical reaction force for stiff
  //    chains and crosses authored thresholds spuriously. Mainstream
  //    PhysX TGS/PGS reseed from prior-substep converged impulses for the
  //    same reason; avbd_standalone simply skips joint lambdas in
  //    Solver::warmstart().
  //
  // 2. Native PxPrismaticJoint: the chain's slow sliding cascades into
  //    limit / angular-lock pumping when prior-frame lambda is
  //    reintroduced (SnippetJoint prismatic chain swings into the wrong
  //    half-space within a few seconds). Standalone addPrismaticJoint
  //    persists rest-relative localFrameB but does not carry joint
  //    lambdas across frames, so match that. Articulation internal
  //    prismatic joints intentionally retain warmstart because the
  //    articulation reduced-coord coupling depends on cross-frame
  //    dual-state continuity.
  //
  // Drive rows and unbreakable / loop-closure / articulation D6 caches are
  // left untouched to preserve steady-state motor tracking and loop tension
  // continuity.
  const bool breakable =
      (constraint.linBreakImpulse < PX_MAX_F32) ||
      (constraint.angBreakImpulse < PX_MAX_F32);
  const bool isPrismaticSource =
      (constraint.header.type == AvbdConstraintType::eJOINT_PRISMATIC);

  if (breakable || isPrismaticSource) {
    constraint.lambdaLinear = physx::PxVec3(0.0f);
    constraint.lambdaAngular = physx::PxVec3(0.0f);
    constraint.coneLambda = 0.0f;
  }
}

void writeJointLambdaToCache(AvbdDynamicsContext &ctx,
                             AvbdD6JointConstraint *constraints,
                             PxU32 numConstraints) {
  if (!ctx.mEnableLambdaWarmStart || !constraints || numConstraints == 0 ||
      ctx.mJointLambdaCache.empty()) {
    return;
  }

  for (PxU32 i = 0; i < numConstraints; ++i) {
    const AvbdD6JointConstraint &constraint = constraints[i];
    const PxU32 cacheIdx = constraint.cacheIndex;
    if (cacheIdx >= ctx.mJointLambdaCache.size()) {
      continue;
    }

    AvbdDynamicsContext::CachedJointLambda &cached =
        ctx.mJointLambdaCache[cacheIdx];
    cached.key = constraint.cacheKey;
    cached.lambdaLinear = constraint.lambdaLinear;
    cached.lambdaAngular = constraint.lambdaAngular;
    cached.lambdaDriveLinear = constraint.lambdaDriveLinear;
    cached.lambdaDriveAngular = constraint.lambdaDriveAngular;
    cached.coneLambda = constraint.coneLambda;
    cached.frameStamp = ctx.getAvbdFrameStamp();
  }
}

void writeJointConstraintWriteback(
    AvbdDynamicsContext &ctx, const AvbdD6JointConstraint *constraints,
    PxU32 numConstraints, PxReal dt) {
  if (!constraints || numConstraints == 0)
    return;

  Cm::PinnableArray<Dy::ConstraintWriteback> &writeBackPool =
      ctx.getConstraintWriteBackPool();
  for (PxU32 i = 0; i < numConstraints; ++i) {
    const AvbdD6JointConstraint &constraint = constraints[i];
    if (constraint.writeBackIndex == PX_MAX_U32)
      continue;

    const bool positionDriveOwned =
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::LinearPositionDrive) ||
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::CoupledLinearPositionDrive);
    const bool angularAxisVelocityDriveOwned =
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::AngularAxisVelocityDrive);
    const bool slerpVelocityDriveOwned =
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::SlerpVelocityDrive);
    const bool genericPhysical1DOwned =
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::GenericHard1D) ||
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::ArticulationHardMimic) ||
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::GenericForceSpring1D) ||
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::GenericRestitution1D);
    const bool passiveNativeReactionOwned =
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::NativePassiveReaction) ||
        hasAvbdJointObjective(
            constraint.objectiveProgram,
            AvbdJointObjectiveKind::CoupledFixedD6);
    const bool physicalWritebackOwned =
        positionDriveOwned || angularAxisVelocityDriveOwned ||
        slerpVelocityDriveOwned || genericPhysical1DOwned ||
        passiveNativeReactionOwned;
    const PxReal linearForce =
        physicalWritebackOwned && constraint.writebackLinearImpulseValid &&
                dt > 0.0f
            ? constraint.writebackLinearImpulse.magnitude() / dt
            : constraint.lambdaLinear.magnitude();
    const PxReal angularTorque =
        physicalWritebackOwned && constraint.writebackAngularImpulseValid &&
                dt > 0.0f
            ? constraint.writebackAngularImpulse.magnitude() / dt
            : constraint.lambdaAngular.magnitude();
    Dy::ConstraintWriteback &writeback =
        writeBackPool[constraint.writeBackIndex];
    const bool linearBreakable = constraint.linBreakImpulse < PX_MAX_F32;
    const bool angularBreakable = constraint.angBreakImpulse < PX_MAX_F32;
    const PxVec3 linearImpulse =
        constraint.writebackLinearImpulseValid
            ? constraint.writebackLinearImpulse
            : constraint.lambdaLinear;
    const PxVec3 angularImpulse =
        constraint.writebackAngularImpulseValid
            ? constraint.writebackAngularImpulse
            : constraint.lambdaAngular;
    const bool genericMultiRow =
        (constraint.sourceFlags &
         AvbdD6JointConstraint::eGENERIC_MULTI_ROW) != 0;
    if (genericMultiRow) {
      if ((constraint.sourceFlags &
           AvbdD6JointConstraint::eGENERIC_MULTI_ROW_LEADER) != 0) {
        writeback.linearImpulse = PxVec3(0.0f);
        writeback.angularImpulse = PxVec3(0.0f);
        writeback.broken = 0;
      }
      writeback.linearImpulse += linearImpulse;
      writeback.angularImpulse += angularImpulse;
      const PxReal aggregateLinearForce =
          dt > 0.0f ? writeback.linearImpulse.magnitude() / dt : 0.0f;
      const PxReal aggregateAngularTorque =
          dt > 0.0f ? writeback.angularImpulse.magnitude() / dt : 0.0f;
      if ((linearBreakable &&
           aggregateLinearForce > constraint.linBreakImpulse) ||
          (angularBreakable &&
           aggregateAngularTorque > constraint.angBreakImpulse))
        writeback.broken = 1;
      continue;
    }
    writeback.linearImpulse = linearImpulse;
    writeback.angularImpulse = angularImpulse;
    writeback.broken =
        ((linearBreakable && linearForce > constraint.linBreakImpulse) ||
         (angularBreakable && angularTorque > constraint.angBreakImpulse))
            ? 1u
            : 0u;
  }
}
} // namespace Dy
} // namespace physx

void AvbdDynamicsContext::update(
    Cm::FlushPool &flushPool, PxBaseTask *continuation,
    PxBaseTask *postPartitioningTask, PxBaseTask *processLostTouchTask,
    PxvNphaseImplementationContext *nPhaseContext, PxU32 maxPatchesPerCM,
    PxU32 maxArticulationLinks, PxReal dt, const PxVec3 &gravity,
    Cm::PinnableBitMap &changedHandleMap) {

  PX_PROFILE_ZONE("AVBD.update", mContextID);

  // Increment global frame counter for motor deduplication
  gAvbdMotorFrameCounter++;

  // Advance independently of active islands.  If a body sleeps or is absent
  // for one update, its cached velocity must not be treated as contiguous
  // history when it next appears.
  ++mBodyVelocityHistoryFrame;
  if (mBodyVelocityHistoryFrame == 0) {
    memset(mBodyVelocityHistoryCache.begin(), 0,
           sizeof(CachedBodyVelocityHistory) *
               mBodyVelocityHistoryCache.size());
    ++mBodyVelocityHistoryFrame;
  }

  // Warm-start age is derived lazily from the per-context frame stamp at
  // lookup time. The previous dense scans touched every cache slot every
  // frame, including mostly empty direct-mapped contact capacity.
  if (mBodyVelocityHistoryFrame > 1u &&
      ((mBodyVelocityHistoryFrame - 1u) % 65535u) == 0u) {
    // Clear stamps at the 16-bit epoch boundary so an entry older than 65535
    // updates cannot alias a fresh stamp after wraparound.
    for (PxU32 i = 0; i < mLambdaCache.size(); ++i)
      mLambdaCache[i].frameStamp = 0;
    for (PxU32 i = 0; i < mJointLambdaCache.size(); ++i)
      mJointLambdaCache[i].frameStamp = 0;
    for (PxU32 i = 0; i < mContactManagerStateCache.size(); ++i)
      mContactManagerStateCache[i].frameStamp = 0;
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
  // A complete soft selection is a first-class main-solver island even when
  // it has no rigid endpoint. Keep one non-addressable allocation slot so the
  // gather/provider/task plumbing stays valid while every physical loop still
  // observes totalBodyCount == 0.
  const PxU32 bodyAllocationCount = PxMax<PxU32>(totalBodyCount, 1u);
  ensureBodyVelocityHistoryCapacity(totalBodyCount);

  // Allocate global arrays - use scratch with main allocator fallback
  AvbdSolverBody *avbdBodies = nullptr;
  PxsRigidBody **rigidBodies = nullptr;
  PxU32 *staticTouchCounts = nullptr;
  {
    PX_PROFILE_ZONE("AVBD.allocateMemory", mContextID);
    avbdBodies = reinterpret_cast<AvbdSolverBody *>(allocWithFallback(
        mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
        sizeof(AvbdSolverBody) * bodyAllocationCount, "AvbdSolverBody"));

    rigidBodies = reinterpret_cast<PxsRigidBody **>(allocWithFallback(
        mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
        sizeof(PxsRigidBody *) * bodyAllocationCount, "RigidBodies"));

    staticTouchCounts = reinterpret_cast<PxU32 *>(allocWithFallback(
        mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
        sizeof(PxU32) * bodyAllocationCount, "StaticTouchCounts"));
  }

  // Check if allocation failed completely
  if (!avbdBodies || !rigidBodies || !staticTouchCounts) {
    return;
  }

  memset(staticTouchCounts, 0, sizeof(PxU32) * bodyAllocationCount);

  // Track articulation info for writeback
  FeatherstoneArticulation **articulationForBody = nullptr;
  PxU32 *linkIndexForBody = nullptr;
  if (numArticulations > 0 && maxArticulationLinks > 0) {
    articulationForBody =
        reinterpret_cast<FeatherstoneArticulation **>(allocWithFallback(
            mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
            sizeof(FeatherstoneArticulation *) * totalBodyCount,
            "ArticulationForBody"));
    linkIndexForBody = reinterpret_cast<PxU32 *>(allocWithFallback(
        mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
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
      mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
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
        mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
        sizeof(PxU32) * (numArticulations + 1),
        "ArticulationFirstLinkIndex"));
    articulationByActiveIdx =
        reinterpret_cast<FeatherstoneArticulation **>(allocWithFallback(
            mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
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
    // Frozen preparation-table metadata. The first P35 slice only consumes
    // these capacities on the serial path; offsets remain legacy compact
    // cursors until a task-owned output plan is admitted.
    PxU32 contactOutputOffset;
    PxU32 contactOutputCapacity;
    PxU32 d6OutputOffset;
    PxU32 d6OutputCapacity;
    PxU32 gearOutputOffset;
    PxU32 gearOutputCapacity;
    bool hasArticulationBodies;
    bool deferredContactPrepEligible;
    AvbdMapStorage mapStorage[3]; // contact, D6, gear
  };

  AvbdIslandInfo *islandInfos =
      reinterpret_cast<AvbdIslandInfo *>(allocWithFallback(
          mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
          sizeof(AvbdIslandInfo) * islandCount, "IslandInfos"));

  if (!islandInfos) {
    return;
  }

  PxU32 bodyIndex = 0;
  const IG::IslandId *islandIds = islandSim.getActiveIslands();

  // 1. Gather bodies per island (including articulation links)
  // Every active island needs a defined body range, including native
  // deformable-only islands.  Stopping once the rigid-body array is full
  // leaves later soft-only island records uninitialized and lets a provider
  // interpret arbitrary body starts/counts as rigid-array indices.
  for (PxU32 i = 0; i < islandCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];
    info.bodyStart = bodyIndex;
    info.articulationJointCount = 0;
    info.contactOutputOffset = 0;
    info.contactOutputCapacity = 0;
    info.d6OutputOffset = 0;
    info.d6OutputCapacity = 0;
    info.gearOutputOffset = 0;
    info.gearOutputCapacity = 0;
    info.hasArticulationBodies = false;
    info.deferredContactPrepEligible = false;
    for (PxU32 mapIndex = 0; mapIndex < 3; ++mapIndex) {
      info.mapStorage[mapIndex].counts = nullptr;
      info.mapStorage[mapIndex].offsets = nullptr;
      info.mapStorage[mapIndex].indices = nullptr;
      info.mapStorage[mapIndex].indexCapacity = 0;
    }
    const IG::Island &island = islandSim.getIsland(islandIds[i]);
    PxNodeIndex currentIndex = island.mRootNode;

    while (currentIndex.isValid() && bodyIndex < totalBodyCount) {
      const IG::Node &node = islandSim.getNode(currentIndex);

      if (node.getNodeType() == IG::Node::eRIGID_BODY_TYPE) {
        rigidBodies[bodyIndex] = getRigidBodyFromIG(islandSim, currentIndex);
        staticTouchCounts[bodyIndex] =
            islandSim.getIslandStaticTouchCount(currentIndex);
        const PxU32 activeNodeIdx = islandSim.getActiveNodeIndex(currentIndex);
        if (activeNodeIdx < maxActiveNodes) {
          bodyRemapTable[activeNodeIdx] = bodyIndex;
        }
        bodyIndex++;
      } else if (node.getNodeType() == IG::Node::eARTICULATION_TYPE) {
        info.hasArticulationBodies = true;
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
              if (bodyCore->fixedBaseLink) {
                // PhysX filters fixed-base articulation roots as static.  AVBD
                // must preserve the same semantics; otherwise the root is
                // integrated under gravity while its static contacts are
                // deliberately absent from narrow phase.
                initializeStaticAvbdBody(bodyCore->body2World, solverBody,
                                         bodyIndex);
              } else {
                PxMat33 R(bodyCore->body2World.q);
                const PxMat33 invInertiaLocal =
                    PxMat33::createDiagonal(bodyCore->inverseInertia);
                const PxMat33 invInertiaWorld =
                    R * invInertiaLocal * R.getTranspose();
                solverBody.initialize(
                    bodyCore->body2World, bodyCore->linearVelocity,
                    computeAvbdAngularVelocity(*bodyCore, dt),
                    bodyCore->inverseMass,
                    invInertiaWorld, bodyIndex);
              }

              restoreAndUpdateBodyVelocityHistory(*bodyCore, solverBody);

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

          // AVBD represents both inbound articulation joints and supported
          // articulation-internal coupling rows in the unified row array.
          // A spatial tendon contributes one constraint per leaf attachment,
          // matching Featherstone's internal-tendon representation.
          PxU32 spatialTendonRowCount = 0;
          for (PxU32 tendonIndex = 0;
               tendonIndex < artData.getSpatialTendonCount();
               ++tendonIndex) {
            ArticulationSpatialTendon *tendon =
                artData.getSpatialTendon(tendonIndex);
            if (!tendon)
              continue;
            ArticulationAttachment *attachments =
                tendon->getAttachments();
            const PxU32 attachmentCount =
                tendon->getNumAttachments();
            for (PxU32 attachmentIndex = 0;
                 attachmentIndex < attachmentCount;
                 ++attachmentIndex) {
              const ArticulationAttachment &attachment =
                  attachments[attachmentIndex];
              if (attachment.parent !=
                      DY_ARTICULATION_ATTACHMENT_NONE &&
                  attachment.childCount == 0)
                ++spatialTendonRowCount;
            }
          }
          info.articulationJointCount +=
              ((linkCount > 1) ? (linkCount - 1) : 0) +
              artData.getMimicJointCount() +
              artData.getFixedTendonCount() +
              spatialTendonRowCount;

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
    START_ENUMERATING_ISLAND_EDGES(IG::Edge::eCONTACT_MANAGER) {
      GET_CURRENT_ISLAND_EDGE
      PxsContactManager *contactManager =
          mIslandManager.getContactManager(edgeId);

      if (contactManager) {
#if IG_CACHE_CONTACT_MANAGER_DATA
        const PxNodeIndex nodeIndex1 = edgeIndices[j].getNodeIndex0();
        const PxNodeIndex nodeIndex2 = edgeIndices[j].getNodeIndex1();
#else
        const PxNodeIndex nodeIndex1 =
            islandSim.mCpuData.getNodeIndex1(edgeId);
        const PxNodeIndex nodeIndex2 =
            islandSim.mCpuData.getNodeIndex2(edgeId);
#endif
        const PxcNpWorkUnit &workUnit = contactManager->getWorkUnit();

        mContactList.pushBack(PxsIndexedContactManager(contactManager));
        PxsIndexedContactManager &icm = mContactList.back();

        // Set up body0
        if (!nodeIndex1.isStaticBody()) {
          const PxU32 activeIdx = islandSim.getActiveNodeIndex(nodeIndex1);
          if (islandSim.getNode(nodeIndex1).isKinematic()) {
            // Preserve the island manager's kinematic namespace.  AVBD does
            // not allocate a dynamic solver body for kinematics, but contact
            // prep still needs the active-kinematic index to recover its
            // target-derived point velocity.
            icm.indexType0 = PxsIndexedInteraction::eKINEMATIC;
            icm.solverBody0 = activeIdx;
          } else {
            const bool isArt0 =
                (workUnit.mFlags &
                 PxcNpWorkUnitFlag::eARTICULATION_BODY0) != 0;
            const PxU32 remapIdx0 =
                isArt0 ? (numDynamicBodies + activeIdx) : activeIdx;
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
          if (islandSim.getNode(nodeIndex2).isKinematic()) {
            icm.indexType1 = PxsIndexedInteraction::eKINEMATIC;
            icm.solverBody1 = activeIdx;
          } else {
            const bool isArt1 =
                (workUnit.mFlags &
                 PxcNpWorkUnitFlag::eARTICULATION_BODY1) != 0;
            const PxU32 remapIdx1 =
                isArt1 ? (numDynamicBodies + activeIdx) : activeIdx;
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
        }
        contactIndex++;
      }
      GET_NEXT_ISLAND_EDGE
    }
    info.cmCount = contactIndex - info.cmStart;

    // Count constraint joints
    info.constraintCount = island.mEdges.getCount(IG::Edge::eCONSTRAINT);
  }

  // Freeze the per-island contact response capacities before any preparation
  // can be dispatched.  NP output is immutable for this update; response
  // filtering may reduce the actual row count returned by prepareAvbdContacts.
  // The first P35 slice keeps the legacy compact cursor for the actual pointer
  // so this metadata is a serial differential with no layout change.
  PxU32 contactOutputOffset = 0;
  for (PxU32 i = 0; i < islandCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];
    info.contactOutputOffset = contactOutputOffset;
    PxU32 capacity = 0;
    for (PxU32 contactOrder = 0; contactOrder < info.cmCount;
         ++contactOrder) {
      const PxsIndexedContactManager &icm =
          mContactList[info.cmStart + contactOrder];
      if (!icm.contactManager)
        continue;
      const PxU32 npIndex = icm.contactManager->getWorkUnit().mNpIndex;
      const PxsContactManagerOutput &output =
          mOutputIterator.getContactManagerOutput(npIndex);
      if (output.nbContacts > PX_MAX_U32 - capacity) {
        capacity = PX_MAX_U32;
        break;
      }
      capacity += output.nbContacts;
    }
    info.contactOutputCapacity = capacity;
    contactOutputOffset =
        capacity > PX_MAX_U32 - contactOutputOffset
            ? PX_MAX_U32
            : contactOutputOffset + capacity;
  }

  // 3. Setup bodies (for rigid bodies only, articulation links already set up)
  if (avbdBodies && rigidBodies) {
    setupBodies(avbdBodies, rigidBodies, bodyIndex, dt, gravity);
  }

  // The Scene may bind one complete soft/VBD tuple to each already gathered
  // native island. Selection happens before any solve task is submitted, so
  // the provider can compile stable island-local particle/body/contact slices
  // and rigid target indices without racing later islands.
  PxArray<AvbdSoftIslandSelection> softSelections;
  if (mSoftIslandProvider) {
    PxArray<PxU32> islandBodyStarts;
    PxArray<PxU32> islandBodyCounts;
    islandBodyStarts.resize(islandCount);
    islandBodyCounts.resize(islandCount);
    for (PxU32 i = 0; i < islandCount; ++i) {
      islandBodyStarts[i] = islandInfos[i].bodyStart;
      islandBodyCounts[i] = islandInfos[i].bodyCount;
    }
    if (!mSoftIslandProvider->prepareSoftIslandSelections(
            avbdBodies, rigidBodies, articulationForBody, linkIndexForBody,
            islandBodyStarts.begin(),
            islandBodyCounts.begin(), islandIds, islandCount, dt,
            gravity, softSelections)) {
      softSelections.clear();
    }
    for (PxU32 selectionIndex = 0;
         selectionIndex < softSelections.size(); ++selectionIndex) {
      const AvbdSoftIslandSelection &selection =
          softSelections[selectionIndex];
      bool uniqueIsland = true;
      for (PxU32 priorIndex = 0; priorIndex < selectionIndex; ++priorIndex)
        uniqueIsland = uniqueIsland &&
                       softSelections[priorIndex].islandIndex !=
                           selection.islandIndex;
      if (!selection.isComplete() ||
          selection.islandIndex >= islandCount || !uniqueIsland) {
        softSelections.clear();
        break;
      }
    }
  }

  // 4. Initialize Solver
  if (!mSolverInitialized) {
    AvbdSolverConfig config;
    config.lengthScale = getLengthScale();
    config.positionTolerance *= config.lengthScale;
    config.iterations = mAvbdIterations;
    config.jointIterationOverride = mAvbdJointIterationOverride;
    config.enableEarlyStop = mAvbdEnableEarlyStop;
    // Enhanced determinism selects the existing ordered AVBD authority.  Do
    // not enable the dormant sort/Kahan experiment here: it changes the
    // frozen trajectory and its contact ordering is not part of this backend
    // selection contract.
    if (isEnhancedDeterminismEnabled())
      config.enableOrderedBackend();
    config.initialRho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
    config.maxRho = AvbdConstants::AVBD_MAX_PENALTY_RHO;
    config.enableLocal6x6Solve = true;
    config.enableStageOwnershipDiagnostics = false;
    config.enableBoundedComponentProductionProbe =
        isEnvFlagEnabled("PHYSX_AVBD_BOUNDED_COMPONENT_PROBE");
    config.contactCompliance = 1e-2f;
    // AVBD reference parameters
    config.avbdAlpha = 0.95f;
    config.avbdBeta = 1000.0f;
    config.avbdGamma = 0.99f;
    config.avbdPenaltyMin = 1000.0f;
    config.avbdPenaltyMax = 1e9f;
    mSolver.initialize(config, mAllocatorAdapter);
    mSolverInitialized = true;
  }

  // Material response: scene bounce threshold (PhysX default -2) for restitution.
  if (mSolverInitialized)
    mSolver.getConfigMutable().bounceThresholdVelocity = getBounceThreshold();

  // 5. Allocate Constraints
  // Allocate one AVBD row for every contact emitted by narrow phase.  Four is
  // only the size of the old reduced-manifold helper, not the CPU contact
  // manager limit (PxContactBuffer can emit up to 256).  Truncating the global
  // row array here silently dropped later mesh support contacts.
  PxU64 maxConstraintCount64 = 0;
  for (PxU32 i = 0; i < mContactList.size(); ++i) {
    PxsContactManager *cm = mContactList[i].contactManager;
    if (!cm)
      continue;
    const PxU32 npIndex = cm->getWorkUnit().mNpIndex;
    const PxsContactManagerOutput &output =
        mOutputIterator.getContactManagerOutput(npIndex);
    maxConstraintCount64 += output.nbContacts;
  }
  const PxU64 maxContactRowsByAllocation =
      static_cast<PxU64>(PX_MAX_U32) / sizeof(AvbdContactConstraint);
  if (maxConstraintCount64 > maxContactRowsByAllocation) {
    PX_WARN_ONCE(
        "AVBD contact row storage exceeds the 32-bit allocator limit; "
        "skipping the step instead of truncating contacts or overflowing the allocation.");
    return;
  }
  const PxU32 maxConstraints = static_cast<PxU32>(maxConstraintCount64);
  // The serial task-graph differential retains its full-frame snapshot table.
  // The normal parallel path allocates a narrower, fail-closed table below
  // only when at least two rigid contact-only islands can consume it.
  const bool contactSnapshotMode = mTaskGraphSerialMode;
  auto allocContactPrepStorage = [&](PxU32 size, const char *name) -> void * {
    return allocWithFallback(mScratchAllocator, mAllocatorAdapter,
                             mHeapFallbackAllocations, size, name);
  };
  AvbdContactConstraint *avbdConstraints = nullptr;
  if (maxConstraints > 0) {
    avbdConstraints = reinterpret_cast<AvbdContactConstraint *>(
        allocContactPrepStorage(sizeof(AvbdContactConstraint) * maxConstraints,
                                "AvbdContactConstraint"));
  }

  // Serial-only staging storage for deferred contact-force commits. The
  // default task-graph path writes through the established direct outputs and
  // never allocates these arrays.
  AvbdContactOutputToken *contactOutputTokens = nullptr;
  AvbdContactOutputTarget *contactOutputTargets = nullptr;
  AvbdContactOutputResult *contactOutputResults = nullptr;
  bool hasDeferredContactOutput = false;
  if (contactSnapshotMode && maxConstraints > 0) {
    for (PxU32 outputScanIndex = 0;
         outputScanIndex < mContactList.size(); ++outputScanIndex) {
      PxsContactManager *cm =
          mContactList[outputScanIndex].contactManager;
      if (!cm)
        continue;
      const PxsContactManagerOutput &output =
          mOutputIterator.getContactManagerOutput(
              cm->getWorkUnit().mNpIndex);
      if (output.contactForces || output.frictionPatches) {
        hasDeferredContactOutput = true;
        break;
      }
    }
  }
  if (hasDeferredContactOutput) {
    contactOutputTokens = reinterpret_cast<AvbdContactOutputToken *>(
        allocContactPrepStorage(sizeof(AvbdContactOutputToken) * maxConstraints,
                                "AvbdContactOutputTokens"));
    contactOutputTargets = reinterpret_cast<AvbdContactOutputTarget *>(
        allocContactPrepStorage(sizeof(AvbdContactOutputTarget) * maxConstraints,
                                "AvbdContactOutputTargets"));
    contactOutputResults = reinterpret_cast<AvbdContactOutputResult *>(
        allocContactPrepStorage(sizeof(AvbdContactOutputResult) * maxConstraints,
                                "AvbdContactOutputResults"));
    if (contactOutputTokens && contactOutputTargets && contactOutputResults) {
      for (PxU32 tokenIndex = 0; tokenIndex < maxConstraints; ++tokenIndex) {
        PX_PLACEMENT_NEW(&contactOutputTokens[tokenIndex],
                         AvbdContactOutputToken)();
        PX_PLACEMENT_NEW(&contactOutputTargets[tokenIndex],
                         AvbdContactOutputTarget)();
        PX_PLACEMENT_NEW(&contactOutputResults[tokenIndex],
                         AvbdContactOutputResult)();
      }
    } else {
      contactOutputTokens = nullptr;
      contactOutputTargets = nullptr;
      contactOutputResults = nullptr;
    }
  }

  // Reserve the complete contact-cache range before any island task is
  // submitted.  Islands are prepared and launched in one loop below; growing
  // PxArray from a later island while an earlier task writes its lambdas would
  // invalidate the earlier task's cache storage and race the reallocation.
  if (mEnableLambdaWarmStart && !mContactList.empty()) {
    PxU64 requiredCacheSize = 0;
    for (PxU32 i = 0; i < mContactList.size(); ++i) {
      PxsContactManager *cm = mContactList[i].contactManager;
      if (!cm)
        continue;
      const PxU64 managerEnd =
          (static_cast<PxU64>(cm->getIndex()) + 1u) *
          CONTACT_CACHE_SLOTS_PER_CM;
      requiredCacheSize = PxMax(requiredCacheSize, managerEnd);
    }
    if (requiredCacheSize <= PX_MAX_U32 &&
        requiredCacheSize > mLambdaCache.size()) {
      const PxU32 oldSize = mLambdaCache.size();
      const PxU32 requested = static_cast<PxU32>(requiredCacheSize);
      const PxU32 newSize = requested > PX_MAX_U32 - 1023u
                                ? requested
                                : (requested + 1023u) & ~1023u;
      mLambdaCache.resize(newSize);
      memset(mLambdaCache.begin() + oldSize, 0,
             sizeof(CachedLambda) * (newSize - oldSize));
    }
  }
  if (!mContactList.empty()) {
    PxU64 requiredManagerStateSize = 0;
    for (PxU32 i = 0; i < mContactList.size(); ++i) {
      PxsContactManager *cm = mContactList[i].contactManager;
      if (!cm)
        continue;
      requiredManagerStateSize = PxMax(
          requiredManagerStateSize,
          static_cast<PxU64>(cm->getIndex()) + 1u);
    }
    if (requiredManagerStateSize <= PX_MAX_U32 &&
        requiredManagerStateSize > mContactManagerStateCache.size()) {
      const PxU32 oldSize = mContactManagerStateCache.size();
      const PxU32 newSize =
          static_cast<PxU32>(requiredManagerStateSize);
      mContactManagerStateCache.resize(newSize);
      memset(mContactManagerStateCache.begin() + oldSize, 0,
             sizeof(CachedContactManagerState) * (newSize - oldSize));
    }
  }

  // P44 serial differential: capture the complete narrow-phase byte ranges
  // before contact preparation.  This path is intentionally fail-closed for
  // kinematic/deformable endpoints; those still use the legacy builder until
  // their commit ABI is explicit.  Contact-force and friction-report output
  // is parent-owned through the token table below. The borrowed stream is not
  // dispatched yet. It only proves that preparation can consume immutable
  // bytes without changing the solver row contract.
  AvbdContactPrepSnapshot *contactSnapshots = nullptr;
  const PxU16 frameStamp = getAvbdFrameStamp();
  if (contactSnapshotMode && !mContactList.empty()) {
    contactSnapshots = reinterpret_cast<AvbdContactPrepSnapshot *>(
        allocContactPrepStorage(
            sizeof(AvbdContactPrepSnapshot) * mContactList.size(),
            "AvbdContactPrepSnapshots"));
    if (contactSnapshots) {
      PxU32 outputTargetCursor = 0;
      for (PxU32 snapshotIndex = 0; snapshotIndex < mContactList.size();
           ++snapshotIndex) {
        PX_PLACEMENT_NEW(&contactSnapshots[snapshotIndex],
                         AvbdContactPrepSnapshot)();
        const PxsIndexedContactManager &icm = mContactList[snapshotIndex];
        PxsContactManager *cm = icm.contactManager;
        if (!cm) {
          contactSnapshots[snapshotIndex].eligible = 1;
          continue;
        }

        AvbdContactPrepSnapshot &snapshot = contactSnapshots[snapshotIndex];
        snapshot.contactManagerIndex = cm->getIndex();
        snapshot.npIndex = cm->getWorkUnit().mNpIndex;
        snapshot.solverBody0 = icm.solverBody0;
        snapshot.solverBody1 = icm.solverBody1;
        snapshot.indexType0 = icm.indexType0;
        snapshot.indexType1 = icm.indexType1;

        const PxsContactManagerOutput &output =
            mOutputIterator.getContactManagerOutput(snapshot.npIndex);
        snapshot.outputTargetBase = outputTargetCursor;
        snapshot.outputTargetCount = output.nbContacts;
        outputTargetCursor += output.nbContacts;
        if (output.contactForces || output.frictionPatches) {
          if (!contactOutputTargets)
            continue;
        }
        if (output.contactForces) {
          snapshot.outputTargetFlags =
              AvbdContactOutputToken::eNORMAL_IMPULSE;
          for (PxU32 contactOffset = 0; contactOffset < output.nbContacts;
               ++contactOffset) {
            contactOutputTargets[snapshot.outputTargetBase + contactOffset]
                .normalImpulse = output.contactForces + contactOffset;
          }
        }

        // The first worker-ready ABI supports rigid body/world endpoints only.
        // Kinematic and deformable/shell rows retain the serial fallback.
        if ((snapshot.indexType0 != PxsIndexedInteraction::eBODY &&
             snapshot.indexType0 != PxsIndexedInteraction::eWORLD) ||
            (snapshot.indexType1 != PxsIndexedInteraction::eBODY &&
             snapshot.indexType1 != PxsIndexedInteraction::eWORLD))
          continue;

        const PxU32 globalBody0Idx =
            snapshot.indexType0 == PxsIndexedInteraction::eBODY
                ? static_cast<PxU32>(snapshot.solverBody0)
                : PX_MAX_U32;
        const PxU32 globalBody1Idx =
            snapshot.indexType1 == PxsIndexedInteraction::eBODY
                ? static_cast<PxU32>(snapshot.solverBody1)
                : PX_MAX_U32;
        snapshot.contactManagerKey = makeAvbdContactManagerKey(
            cm, globalBody0Idx, globalBody1Idx);
        snapshot.restDistance = cm->getRestDistance();
        if (isDeformableStaticAnchorContact(
                (globalBody0Idx != PX_MAX_U32) !=
                    (globalBody1Idx != PX_MAX_U32),
                snapshot.restDistance))
          continue;

        if (snapshot.contactManagerIndex < mContactManagerStateCache.size()) {
          const CachedContactManagerState &state =
              mContactManagerStateCache[snapshot.contactManagerIndex];
          if (state.key == snapshot.contactManagerKey &&
              state.frameStamp != 0) {
            const PxU16 age = static_cast<PxU16>(frameStamp - state.frameStamp);
            snapshot.contactManagerAge =
                age >= 255u ? PxU8(255) : PxU8(age);
          }
        }

        if (!output.contactPatches || !output.contactPoints ||
            output.nbContacts == 0 || output.nbPatches == 0)
          continue;

        const PxContactStreamIterator stream(
            output.contactPatches, output.contactPoints,
            output.getInternalFaceIndice(), output.nbPatches,
            output.nbContacts);
        const PxU32 patchBytes =
            static_cast<PxU32>(output.nbPatches) * sizeof(PxContactPatch);
        const PxU32 pointBytes =
            static_cast<PxU32>(output.nbContacts) * stream.contactPointSize;
        if (stream.contactPointSize == 0 ||
            patchBytes > PX_MAX_U32 || pointBytes > PX_MAX_U32)
          continue;

        // Capture the exact response-row cardinality from the immutable
        // narrow-phase stream. The pure parser records its emitted count and
        // the parent compares the two before committing snapshot state.
        PxU32 expectedResponseRows = 0;
        if (!stream.forceNoResponse) {
          const bool hasExtendedContact =
              stream.mStreamFormat != PxContactStreamIterator::eSIMPLE_STREAM;
          for (PxU8 patchIdx = 0; patchIdx < output.nbPatches; ++patchIdx) {
            const PxContactPatch *patch = reinterpret_cast<const PxContactPatch *>(
                output.contactPatches + patchIdx * sizeof(PxContactPatch));
            for (PxU16 contactIdx = 0; contactIdx < patch->nbContacts;
                 ++contactIdx) {
              if (!hasExtendedContact) {
                ++expectedResponseRows;
                continue;
              }
              const PxExtendedContact *contact =
                  reinterpret_cast<const PxExtendedContact *>(
                      output.contactPoints +
                      (patch->startContactIndex + contactIdx) *
                          stream.contactPointSize);
              if (contact->maxImpulse > 0.0f)
                ++expectedResponseRows;
            }
          }
        }
        snapshot.expectedResponseRows = expectedResponseRows > PX_MAX_U16
                                            ? PX_MAX_U16
                                            : static_cast<PxU16>(expectedResponseRows);

        if (output.frictionPatches) {
          PxFrictionPatch *frictionPatches =
              reinterpret_cast<PxFrictionPatch *>(output.frictionPatches);
          for (PxU8 patchIdx = 0; patchIdx < output.nbPatches; ++patchIdx) {
            const PxContactPatch *patch = reinterpret_cast<const PxContactPatch *>(
                output.contactPatches + patchIdx * sizeof(PxContactPatch));
            PxFrictionPatch &frictionPatch = frictionPatches[patchIdx];
            frictionPatch.anchorCount = 0;
            frictionPatch.anchorPositions[0] = PxVec3(0.0f);
            frictionPatch.anchorPositions[1] = PxVec3(0.0f);
            frictionPatch.anchorImpulses[0] = PxVec3(0.0f);
            frictionPatch.anchorImpulses[1] = PxVec3(0.0f);

            PxU32 anchorCount = 0;
            if (patch->nbContacts > 0 &&
                (patch->dynamicFriction > 0.0f ||
                 patch->staticFriction > 0.0f)) {
              const PxContact *firstContact =
                  reinterpret_cast<const PxContact *>(
                      output.contactPoints +
                      patch->startContactIndex * stream.contactPointSize);
              frictionPatch.anchorPositions[0] = firstContact->contact;
              anchorCount = 1;

              PxReal farthestDistanceSq = 0.0f;
              PxVec3 farthestPosition = firstContact->contact;
              for (PxU16 contactIdx = 1; contactIdx < patch->nbContacts;
                   ++contactIdx) {
                const PxContact *candidate =
                    reinterpret_cast<const PxContact *>(
                        output.contactPoints +
                        (patch->startContactIndex + contactIdx) *
                            stream.contactPointSize);
                const PxReal distanceSq =
                    (candidate->contact - firstContact->contact)
                        .magnitudeSquared();
                if (distanceSq > farthestDistanceSq) {
                  farthestDistanceSq = distanceSq;
                  farthestPosition = candidate->contact;
                }
              }
              if (farthestDistanceSq > 1.0e-12f) {
                frictionPatch.anchorPositions[1] = farthestPosition;
                anchorCount = 2;
              }
              frictionPatch.anchorCount = anchorCount;
            }

            if (anchorCount > 0) {
              snapshot.outputTargetFlags |=
                  AvbdContactOutputToken::eFRICTION_IMPULSE;
              for (PxU16 contactIdx = 0; contactIdx < patch->nbContacts;
                   ++contactIdx) {
                const PxContact *contact = reinterpret_cast<const PxContact *>(
                    output.contactPoints +
                    (patch->startContactIndex + contactIdx) *
                        stream.contactPointSize);
                PxU32 anchorIndex = 0;
                if (anchorCount > 1) {
                  const PxReal distance0 =
                      (contact->contact - frictionPatch.anchorPositions[0])
                          .magnitudeSquared();
                  const PxReal distance1 =
                      (contact->contact - frictionPatch.anchorPositions[1])
                          .magnitudeSquared();
                  if (distance1 < distance0)
                    anchorIndex = 1;
                }
                contactOutputTargets[snapshot.outputTargetBase +
                                     patch->startContactIndex + contactIdx]
                    .frictionImpulse =
                    &frictionPatch.anchorImpulses[anchorIndex];
              }
            }
          }
        }

        // The cache belongs to this contact manager's disjoint slot range.
        // Borrow it read-only until the solve task writes the same range back;
        // this avoids rebuilding per-contact compact cache payloads during
        // the parent preparation prefix.
        if (mEnableLambdaWarmStart) {
          const PxU64 cacheBase = static_cast<PxU64>(
              snapshot.contactManagerIndex) * CONTACT_CACHE_SLOTS_PER_CM;
          const PxU64 cacheEnd = cacheBase + CONTACT_CACHE_SLOTS_PER_CM;
          if (cacheEnd <= mLambdaCache.size())
            snapshot.lambdaCacheDirect = &mLambdaCache[static_cast<PxU32>(cacheBase)];
        }
        // The narrow-phase output remains immutable until the post-solver
        // continuation consumes it. Borrow the already frozen byte ranges
        // instead of copying every patch/point into scratch storage; this is
        // the same lifetime used by the legacy parent-side parser.
        snapshot.contactPatches = output.contactPatches;
        snapshot.contactPoints = output.contactPoints;
        snapshot.nbPatches = output.nbPatches;
        snapshot.nbContacts = output.nbContacts;
        snapshot.contactPointSize =
            static_cast<PxU16>(stream.contactPointSize);
        snapshot.eligible = 1;
      }
    }
  }

  PxU32 totalJoints = 0;
  PxU32 totalArticulationJoints = 0;
  for (PxU32 i = 0; i < islandCount; ++i) {
    totalJoints += islandInfos[i].constraintCount;
    totalArticulationJoints += islandInfos[i].articulationJointCount;
  }

  // An extension/custom PxConstraintSolverPrep may emit multiple independent
  // Px1DConstraint rows (Vehicle emits up to 12 from one PxConstraint).
  // Keep gear/articulation capacity at one object per source constraint, but
  // reserve the public solverPrep maximum for the D6-backed generic row path.
  const PxU64 totalD6Capacity64 =
      static_cast<PxU64>(totalJoints) * MAX_CONSTRAINT_ROWS +
      totalArticulationJoints;
  PxU32 totalD6Capacity =
      totalD6Capacity64 > PX_MAX_U32
          ? PX_MAX_U32
          : static_cast<PxU32>(totalD6Capacity64);
  PxU32 totalGearCapacity = totalJoints + totalArticulationJoints;
  if (totalD6Capacity == 0)
    totalD6Capacity = 1;
  if (totalGearCapacity == 0)
    totalGearCapacity = 1;

  PxU32 d6OutputOffset = 0;
  PxU32 gearOutputOffset = 0;
  for (PxU32 i = 0; i < islandCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];
    const PxU64 d6Capacity64 =
        static_cast<PxU64>(info.constraintCount) * MAX_CONSTRAINT_ROWS +
        info.articulationJointCount;
    const PxU64 gearCapacity64 =
        static_cast<PxU64>(info.constraintCount) +
        info.articulationJointCount;
    info.d6OutputOffset = d6OutputOffset;
    info.d6OutputCapacity =
        d6Capacity64 > PX_MAX_U32 ? PX_MAX_U32
                                  : static_cast<PxU32>(d6Capacity64);
    info.gearOutputOffset = gearOutputOffset;
    info.gearOutputCapacity =
        gearCapacity64 > PX_MAX_U32 ? PX_MAX_U32
                                    : static_cast<PxU32>(gearCapacity64);
    d6OutputOffset =
        info.d6OutputCapacity > PX_MAX_U32 - d6OutputOffset
            ? PX_MAX_U32
            : d6OutputOffset + info.d6OutputCapacity;
    gearOutputOffset =
        info.gearOutputCapacity > PX_MAX_U32 - gearOutputOffset
            ? PX_MAX_U32
            : gearOutputOffset + info.gearOutputCapacity;
  }

  // The capacity tables above are parent-owned and immutable for this update.
  // When their aggregate fits the backing arrays, they form disjoint
  // island-local output ranges used by the serial preparation pass without
  // advancing a shared cursor. Keep the compact cursor fallback
  // fail-closed for pathological overflow/allocator configurations.
  const bool useFixedPreparationRanges =
      contactOutputOffset <= maxConstraints &&
      d6OutputOffset <= totalD6Capacity &&
      gearOutputOffset <= totalGearCapacity;

  AvbdD6JointConstraint *d6Joints =
      reinterpret_cast<AvbdD6JointConstraint *>(allocWithFallback(
          mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
          sizeof(AvbdD6JointConstraint) * totalD6Capacity, "D6Joints"));
  AvbdGearJointConstraint *gearJoints =
      reinterpret_cast<AvbdGearJointConstraint *>(allocWithFallback(
          mScratchAllocator, mAllocatorAdapter, mHeapFallbackAllocations,
          sizeof(AvbdGearJointConstraint) * totalGearCapacity, "GearJoints"));

  // Reserve one disjoint map slot per island before any solve task can be
  // submitted.  A failed reservation is deliberately non-fatal: that island
  // falls back to the legacy allocator-backed build() below.
  PxAllocatorCallback &mapAllocator = mAllocatorAdapter;
  auto releaseMapStorage = [&mapAllocator](AvbdMapStorage &storage) {
    if (storage.counts)
      mapAllocator.deallocate(storage.counts);
    if (storage.offsets)
      mapAllocator.deallocate(storage.offsets);
    if (storage.indices)
      mapAllocator.deallocate(storage.indices);
    storage.counts = nullptr;
    storage.offsets = nullptr;
    storage.indices = nullptr;
    storage.indexCapacity = 0;
  };
  auto clearMapStorage = [](AvbdMapStorage &storage) {
    storage.counts = nullptr;
    storage.offsets = nullptr;
    storage.indices = nullptr;
    storage.indexCapacity = 0;
  };
  auto reserveMapStorage = [&mapAllocator](AvbdMapStorage &storage,
                                            PxU32 numBodies,
                                            PxU32 indexCapacity,
                                            const char *label) {
    if (numBodies == 0 || indexCapacity == 0 ||
        numBodies > PX_MAX_U32 / sizeof(PxU32) - 1u)
      return false;
    storage.counts = static_cast<PxU32 *>(mapAllocator.allocate(
        sizeof(PxU32) * numBodies, label, __FILE__, __LINE__));
    storage.offsets = static_cast<PxU32 *>(mapAllocator.allocate(
        sizeof(PxU32) * (numBodies + 1u), label, __FILE__, __LINE__));
    storage.indices = static_cast<PxU32 *>(mapAllocator.allocate(
        sizeof(PxU32) * indexCapacity, label, __FILE__, __LINE__));
    if (!storage.counts || !storage.offsets || !storage.indices) {
      if (storage.counts)
        mapAllocator.deallocate(storage.counts);
      if (storage.offsets)
        mapAllocator.deallocate(storage.offsets);
      if (storage.indices)
        mapAllocator.deallocate(storage.indices);
      storage.counts = nullptr;
      storage.offsets = nullptr;
      storage.indices = nullptr;
      storage.indexCapacity = 0;
      return false;
    }
    storage.indexCapacity = indexCapacity;
    return true;
  };
  for (PxU32 i = 0; i < islandCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];
    if (info.bodyCount > 0 && info.contactOutputCapacity <= PX_MAX_U32 / 2u)
      reserveMapStorage(info.mapStorage[0], info.bodyCount,
                        info.contactOutputCapacity * 2u,
                        "AvbdContactMapStorage");
    if (info.bodyCount > 0 && info.d6OutputCapacity <= PX_MAX_U32 / 2u)
      reserveMapStorage(info.mapStorage[1], info.bodyCount,
                        info.d6OutputCapacity * 2u,
                        "AvbdD6MapStorage");
    if (info.bodyCount > 0 && info.gearOutputCapacity <= PX_MAX_U32 / 2u)
      reserveMapStorage(info.mapStorage[2], info.bodyCount,
                        info.gearOutputCapacity * 2u,
                        "AvbdGearMapStorage");
  }

  // Freeze the narrow-phase input for contact-only rigid islands before any
  // solve task is submitted. The parent performs only an O(manager/patch)
  // eligibility pass; row materialization then runs inside the already
  // existing island tasks, with no extra task edge or global join.
  AvbdContactPrepSnapshot *deferredContactSnapshots = nullptr;
  auto islandOwnsCompleteSoftSelection = [&](PxU32 islandIndex) {
    for (PxU32 selectionIndex = 0; selectionIndex < softSelections.size();
         ++selectionIndex) {
      if (softSelections[selectionIndex].islandIndex == islandIndex &&
          softSelections[selectionIndex].isComplete())
        return true;
    }
    return false;
  };
  PxTaskManager *taskManager =
      mTaskFactory ? mTaskFactory->getTaskManager() : nullptr;
  // Below this crossover the established parent-prep/worker-solve pipeline is
  // faster than moving both phases into the same island task.
  const PxU32 minDeferredContactPrepWorkers = 8;
  const bool canDeferContactPreparation =
      !mTaskGraphSerialMode && useFixedPreparationRanges && avbdConstraints &&
      !mSolver.getConfig().requiresOrderedBackend() && taskManager &&
      taskManager->getCpuDispatcher() &&
      taskManager->getCpuDispatcher()->getWorkerCount() >=
          minDeferredContactPrepWorkers;
  PxU32 coarseDeferredIslandCount = 0;
  if (canDeferContactPreparation) {
    for (PxU32 islandIndex = 0; islandIndex < islandCount; ++islandIndex) {
      const AvbdIslandInfo &info = islandInfos[islandIndex];
      if (info.bodyCount > 0 && info.cmCount > 0 &&
          info.constraintCount == 0 && info.articulationJointCount == 0 &&
          !info.hasArticulationBodies &&
          !islandOwnsCompleteSoftSelection(islandIndex))
        ++coarseDeferredIslandCount;
    }
  }

  if (coarseDeferredIslandCount >= 2 &&
      mContactList.size() <=
          PX_MAX_U32 / sizeof(AvbdContactPrepSnapshot)) {
    deferredContactSnapshots =
        reinterpret_cast<AvbdContactPrepSnapshot *>(allocContactPrepStorage(
            sizeof(AvbdContactPrepSnapshot) * mContactList.size(),
            "AvbdDeferredContactPrepSnapshots"));
  }
  if (deferredContactSnapshots) {
    for (PxU32 snapshotIndex = 0; snapshotIndex < mContactList.size();
         ++snapshotIndex)
      PX_PLACEMENT_NEW(&deferredContactSnapshots[snapshotIndex],
                       AvbdContactPrepSnapshot)();

    const PxU16 unsupportedFlags = static_cast<PxU16>(
        PxcNpWorkUnitFlag::eOUTPUT_CONTACTS |
        PxcNpWorkUnitFlag::eFORCE_THRESHOLD |
        PxcNpWorkUnitFlag::eMODIFIABLE_CONTACT |
        PxcNpWorkUnitFlag::eARTICULATION_BODY0 |
        PxcNpWorkUnitFlag::eARTICULATION_BODY1 |
        PxcNpWorkUnitFlag::eSOFT_BODY |
        PxcNpWorkUnitFlag::eHAS_KINEMATIC_ACTOR |
        PxcNpWorkUnitFlag::eDISABLE_RESPONSE |
        PxcNpWorkUnitFlag::eDETECT_CCD_CONTACTS);
    PxU32 exactDeferredIslandCount = 0;

    for (PxU32 islandIndex = 0; islandIndex < islandCount; ++islandIndex) {
      AvbdIslandInfo &info = islandInfos[islandIndex];
      if (info.bodyCount == 0 || info.cmCount == 0 ||
          info.constraintCount != 0 || info.articulationJointCount != 0 ||
          info.hasArticulationBodies ||
          islandOwnsCompleteSoftSelection(islandIndex))
        continue;

      bool eligible = true;
      PxU32 capturedRows = 0;
      const PxU64 bodyEnd =
          static_cast<PxU64>(info.bodyStart) + info.bodyCount;
      for (PxU32 contactOrder = 0;
           eligible && contactOrder < info.cmCount; ++contactOrder) {
        const PxU32 snapshotIndex = info.cmStart + contactOrder;
        const PxsIndexedContactManager &icm = mContactList[snapshotIndex];
        PxsContactManager *cm = icm.contactManager;
        if (!cm ||
            (icm.indexType0 != PxsIndexedInteraction::eBODY &&
             icm.indexType0 != PxsIndexedInteraction::eWORLD) ||
            (icm.indexType1 != PxsIndexedInteraction::eBODY &&
             icm.indexType1 != PxsIndexedInteraction::eWORLD)) {
          eligible = false;
          break;
        }

        const bool body0 = icm.indexType0 == PxsIndexedInteraction::eBODY;
        const bool body1 = icm.indexType1 == PxsIndexedInteraction::eBODY;
        const PxU64 globalBody0 = body0 ? icm.solverBody0 : PX_MAX_U32;
        const PxU64 globalBody1 = body1 ? icm.solverBody1 : PX_MAX_U32;
        if ((!body0 && !body1) ||
            (body0 && (globalBody0 < info.bodyStart ||
                       globalBody0 >= bodyEnd)) ||
            (body1 && (globalBody1 < info.bodyStart ||
                       globalBody1 >= bodyEnd))) {
          eligible = false;
          break;
        }

        const PxcNpWorkUnit &workUnit = cm->getWorkUnit();
        const PxU16 requiredFlags = static_cast<PxU16>(
            PxcNpWorkUnitFlag::eOUTPUT_CONSTRAINTS |
            PxcNpWorkUnitFlag::eDETECT_DISCRETE_CONTACT);
        if ((workUnit.mFlags & unsupportedFlags) != 0 ||
            (workUnit.mFlags & requiredFlags) != requiredFlags ||
            workUnit.mCCDContacts) {
          eligible = false;
          break;
        }
        const PxsContactManagerOutput &output =
            mOutputIterator.getContactManagerOutput(workUnit.mNpIndex);
        // Snapshot parsing deliberately has no mutable narrow-phase output.
        // Require the direct path when an otherwise unreported stream still
        // carries normal/friction writeback storage, so even internal output
        // buffers retain byte-for-byte legacy ownership.
        if (output.contactForces || output.frictionPatches ||
            !output.contactPatches || !output.contactPoints ||
            output.nbContacts == 0 || output.nbPatches == 0 ||
            output.nbContacts > info.contactOutputCapacity - capturedRows) {
          eligible = false;
          break;
        }

        const PxContactStreamIterator stream(
            output.contactPatches, output.contactPoints,
            output.getInternalFaceIndice(), output.nbPatches,
            output.nbContacts);
        if (stream.forceNoResponse ||
            stream.mStreamFormat != PxContactStreamIterator::eSIMPLE_STREAM ||
            stream.contactPointSize != sizeof(PxContact)) {
          eligible = false;
          break;
        }

        PxU32 runningContact = 0;
        for (PxU32 patchIndex = 0;
             eligible && patchIndex < output.nbPatches; ++patchIndex) {
          const PxContactPatch *patch =
              reinterpret_cast<const PxContactPatch *>(
                  output.contactPatches +
                  patchIndex * sizeof(PxContactPatch));
          const PxU64 patchEnd = static_cast<PxU64>(patch->startContactIndex) +
                                 patch->nbContacts;
          if (patch->nbContacts == 0 ||
              patch->startContactIndex != runningContact ||
              patchEnd > output.nbContacts) {
            eligible = false;
            break;
          }
          runningContact = static_cast<PxU32>(patchEnd);
        }
        if (!eligible || runningContact != output.nbContacts) {
          eligible = false;
          break;
        }

        AvbdContactPrepSnapshot &snapshot =
            deferredContactSnapshots[snapshotIndex];
        snapshot.contactManagerIndex = cm->getIndex();
        snapshot.npIndex = workUnit.mNpIndex;
        snapshot.solverBody0 = icm.solverBody0;
        snapshot.solverBody1 = icm.solverBody1;
        snapshot.indexType0 = icm.indexType0;
        snapshot.indexType1 = icm.indexType1;
        snapshot.contactManagerKey = makeAvbdContactManagerKey(
            cm, body0 ? static_cast<PxU32>(globalBody0) : PX_MAX_U32,
            body1 ? static_cast<PxU32>(globalBody1) : PX_MAX_U32);
        snapshot.restDistance = cm->getRestDistance();
        if (isDeformableStaticAnchorContact(body0 != body1,
                                            snapshot.restDistance)) {
          eligible = false;
          break;
        }

        if (snapshot.contactManagerIndex <
            mContactManagerStateCache.size()) {
          const CachedContactManagerState &state =
              mContactManagerStateCache[snapshot.contactManagerIndex];
          if (state.key == snapshot.contactManagerKey &&
              state.frameStamp != 0) {
            const PxU16 age =
                static_cast<PxU16>(frameStamp - state.frameStamp);
            snapshot.contactManagerAge =
                age >= 255u ? PxU8(255) : PxU8(age);
          }
        }
        if (mEnableLambdaWarmStart) {
          const PxU64 cacheBase =
              static_cast<PxU64>(snapshot.contactManagerIndex) *
              CONTACT_CACHE_SLOTS_PER_CM;
          const PxU64 cacheEnd = cacheBase + CONTACT_CACHE_SLOTS_PER_CM;
          if (cacheEnd <= mLambdaCache.size())
            snapshot.lambdaCacheDirect =
                &mLambdaCache[static_cast<PxU32>(cacheBase)];
        }
        snapshot.contactPatches = output.contactPatches;
        snapshot.contactPoints = output.contactPoints;
        snapshot.nbPatches = output.nbPatches;
        snapshot.nbContacts = output.nbContacts;
        snapshot.contactPointSize =
            static_cast<PxU16>(stream.contactPointSize);
        snapshot.expectedResponseRows = output.nbContacts;
        snapshot.eligible = 1;
        capturedRows += output.nbContacts;
      }

      if (eligible && capturedRows == info.contactOutputCapacity) {
        info.deferredContactPrepEligible = true;
        ++exactDeferredIslandCount;
      }
    }

    // A single island would merely move serial work to another thread while
    // retaining all capture cost. Require actual cross-island parallelism.
    if (exactDeferredIslandCount < 2) {
      for (PxU32 islandIndex = 0; islandIndex < islandCount; ++islandIndex)
        islandInfos[islandIndex].deferredContactPrepEligible = false;
    }
  }

  auto buildContactMap = [&](AvbdBodyConstraintMap &map,
                             AvbdMapStorage &storage,
                             const AvbdContactConstraint *constraints,
                             PxU32 numBodies, PxU32 numConstraints) {
    if (storage.counts && storage.offsets && storage.indices &&
        map.buildInPlace(numBodies, constraints, numConstraints,
                         storage.counts, storage.offsets, storage.indices,
                         storage.indexCapacity)) {
      clearMapStorage(storage);
      return;
    }
    releaseMapStorage(storage);
    map.build(numBodies, constraints, numConstraints, mapAllocator);
  };
  auto buildD6Map = [&](AvbdBodyConstraintMap &map,
                        AvbdMapStorage &storage,
                        const AvbdD6JointConstraint *constraints,
                        PxU32 numBodies, PxU32 numConstraints) {
    if (storage.counts && storage.offsets && storage.indices &&
        map.buildInPlace(numBodies, constraints, numConstraints,
                         storage.counts, storage.offsets, storage.indices,
                         storage.indexCapacity)) {
      clearMapStorage(storage);
      return;
    }
    releaseMapStorage(storage);
    map.build(numBodies, constraints, numConstraints, mapAllocator);
  };
  auto buildGearMap = [&](AvbdBodyConstraintMap &map,
                          AvbdMapStorage &storage,
                          const AvbdGearJointConstraint *constraints,
                          PxU32 numBodies, PxU32 numConstraints) {
    if (storage.counts && storage.offsets && storage.indices &&
        map.buildInPlace(numBodies, constraints, numConstraints,
                         storage.counts, storage.offsets, storage.indices,
                         storage.indexCapacity)) {
      clearMapStorage(storage);
      return;
    }
    releaseMapStorage(storage);
    map.build(numBodies, constraints, numConstraints, mapAllocator);
  };

  // 6. Create Task Chain
  AvbdWriteBackTask *wbTask = mTaskFactory->createWriteBackTask(
      *this, avbdBodies, rigidBodies, staticTouchCounts, bodyIndex, dt,
      mEnableStabilization, isSleepingDisabled(), articulationForBody,
      linkIndexForBody, contactOutputTargets, contactOutputResults,
      maxConstraints);
  wbTask->setContinuation(continuation);

  AvbdCoordinatorTask *coordTask =
      mTaskFactory->createCoordinatorTask(*this, wbTask);
  coordTask->setContinuation(wbTask);

  PxU32 currentConstraintIdx = 0;
  PxU32 currD6Idx = 0;
  PxU32 currGearIdx = 0;
  PxU32 tasksSpawned = 0;
  // 7. Iterate Islands
  for (PxU32 i = 0; i < islandCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];
    const PxU32 d6StorageOffset =
        useFixedPreparationRanges ? info.d6OutputOffset : currD6Idx;
    const PxU32 gearStorageOffset =
        useFixedPreparationRanges ? info.gearOutputOffset : currGearIdx;

    // Prepare contact constraints
    PxU32 numConstraints = 0;
    const bool deferContactPrep =
        info.deferredContactPrepEligible && deferredContactSnapshots;
    AvbdContactPrepSnapshot *snapshotsForIsland = nullptr;
    if (mTaskGraphSerialMode && contactSnapshots && info.cmCount > 0) {
      bool eligible = true;
      for (PxU32 contactOrder = 0; contactOrder < info.cmCount;
           ++contactOrder) {
        if (!contactSnapshots[info.cmStart + contactOrder].eligible) {
          eligible = false;
          break;
        }
      }
      if (eligible)
        snapshotsForIsland = contactSnapshots;
    }
    const PxU32 contactStorageOffset =
        useFixedPreparationRanges ? info.contactOutputOffset
                                  : currentConstraintIdx;

    bool ownsCompleteSoftSelection = false;
    for (PxU32 selectionIndex = 0;
         selectionIndex < softSelections.size(); ++selectionIndex) {
      if (softSelections[selectionIndex].islandIndex == i &&
          softSelections[selectionIndex].isComplete()) {
        ownsCompleteSoftSelection = true;
        break;
      }
    }
    if (!deferContactPrep && avbdConstraints && info.cmCount > 0) {
      numConstraints = prepareAvbdContacts(
          islandSim, dt, &avbdBodies[info.bodyStart], info.bodyCount,
          avbdConstraints + contactStorageOffset,
          info.contactOutputCapacity, info.cmStart, info.cmCount,
          info.bodyStart, snapshotsForIsland, contactOutputTokens,
          contactStorageOffset);
      if (snapshotsForIsland) {
        bool rowCardinalityMatch = true;
        for (PxU32 contactOrder = 0; contactOrder < info.cmCount;
             ++contactOrder) {
          const AvbdContactPrepSnapshot &snapshot =
              contactSnapshots[info.cmStart + contactOrder];
          if (snapshot.expectedResponseRows != snapshot.emittedResponseRows) {
            rowCardinalityMatch = false;
            break;
          }
        }
        if (!rowCardinalityMatch) {
          // The immutable payload did not satisfy the row contract. Rebuild
          // this island through the direct context path instead of allowing a
          // partial snapshot result to reach the solver.
          snapshotsForIsland = nullptr;
          numConstraints = prepareAvbdContacts(
              islandSim, dt, &avbdBodies[info.bodyStart], info.bodyCount,
              avbdConstraints + contactStorageOffset,
              info.contactOutputCapacity, info.cmStart, info.cmCount,
              info.bodyStart, nullptr, nullptr, 0);
        } else {
          // Persistent contact-manager state is a parent-owned commit. The
          // snapshot parser only emits a token, so no worker can retain or
          // mutate the context cache slot.
          commitContactPrepSnapshots(contactSnapshots, info.cmStart,
                                     info.cmCount, frameStamp);
        }
      }
    }

    PxU32 numD6 = 0;
    PxU32 numGear = 0;

    // Prepare external joint constraints
    if (info.constraintCount > 0) {
      prepareAvbdConstraints(
           islandSim, dt, &avbdBodies[info.bodyStart], info.bodyCount,
           info.bodyStart, d6Joints + d6StorageOffset, numD6,
           info.d6OutputCapacity, gearJoints + gearStorageOffset, numGear,
           info.gearOutputCapacity, i, bodyRemapTable,
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
                   *this, articulation, localFirstBodyIdx,
                   d6Joints + d6StorageOffset + numD6, artD6,
                   info.d6OutputCapacity - PxMin(
                       info.d6OutputCapacity, numD6),
                   gearJoints + gearStorageOffset + numGear, artGear,
                   info.gearOutputCapacity - PxMin(
                       info.gearOutputCapacity, numGear),
                   dt);

              numD6 += artD6;
              numGear += artGear;
            }
          }
        }
        currentNodeIndex = node.mNextNode;
      }
    }

    const AvbdSoftIslandSelection *softSelection = nullptr;
    for (PxU32 selectionIndex = 0;
         selectionIndex < softSelections.size(); ++selectionIndex) {
      if (softSelections[selectionIndex].islandIndex == i) {
        softSelection = &softSelections[selectionIndex];
        break;
      }
    }
    const bool ownsSoftSelection =
        softSelection && softSelection->isComplete();

    // Skip empty islands
    if (numConstraints == 0 && numD6 == 0 && numGear == 0 &&
        info.bodyCount == 0 && !ownsSoftSelection) {
      continue;
    }

    // Create Island Batch
    AvbdIslandBatch batch;
    batch.bodies =
        info.bodyCount > 0 ? &avbdBodies[info.bodyStart] : nullptr;
    batch.numBodies = info.bodyCount;
    batch.hasArticulationBodies = info.hasArticulationBodies;
    batch.articulationForBody =
        info.bodyCount > 0 && articulationForBody
            ? &articulationForBody[info.bodyStart]
            : nullptr;
    batch.linkIndexForBody =
        info.bodyCount > 0 && linkIndexForBody
            ? &linkIndexForBody[info.bodyStart]
            : nullptr;
    batch.constraints = avbdConstraints
                            ? &avbdConstraints[contactStorageOffset]
                            : nullptr;
    batch.numConstraints = numConstraints;
    batch.contactOutputTokens =
        snapshotsForIsland && contactOutputTokens
            ? &contactOutputTokens[contactStorageOffset]
            : nullptr;
    batch.contactOutputResults =
        batch.contactOutputTokens ? contactOutputResults : nullptr;
    if (deferContactPrep) {
      batch.deferredContactPrep.snapshots = deferredContactSnapshots;
      batch.deferredContactPrep.lengthScale = getLengthScale();
      batch.deferredContactPrep.startContactIdx = info.cmStart;
      batch.deferredContactPrep.numContactManagers = info.cmCount;
      batch.deferredContactPrep.bodyOffset = info.bodyStart;
      batch.deferredContactPrep.constraintCapacity =
          info.contactOutputCapacity;
      batch.deferredContactPrep.frameStamp = frameStamp;
      batch.deferredContactPrep.enableLambdaWarmStart =
          mEnableLambdaWarmStart;
    }

    batch.d6Joints = &d6Joints[d6StorageOffset];
    batch.numD6 = numD6;
    batch.gearJoints = &gearJoints[gearStorageOffset];
    batch.numGear = numGear;

    batch.softParticles =
        ownsSoftSelection ? softSelection->particles : nullptr;
    batch.numSoftParticles =
        ownsSoftSelection ? softSelection->numParticles : 0;
    batch.softBodies = ownsSoftSelection ? softSelection->bodies : nullptr;
    batch.numSoftBodies =
        ownsSoftSelection ? softSelection->numBodies : 0;
    batch.softContacts =
        ownsSoftSelection ? softSelection->contacts : nullptr;
    batch.numSoftContacts =
        ownsSoftSelection ? softSelection->numContacts : 0;
    batch.softExecutionPlan = ownsSoftSelection
                                  ? softSelection->executionPlan
                                  : AvbdSoftIslandExecutionPlan();

    batch.islandStart = i;
    batch.islandEnd = i + 1;
    batch.colorBatches = nullptr;
    batch.numColors = 0;

    const PxU32 softIterations =
        ownsSoftSelection ? softSelection->iterationOverride : 0;
    batch.iterationOverride =
        PxMax(islandArticIterations, softIterations);

    // Build constraint-to-body mappings for the serial path.  Parallel solve
    // tasks consume their disjoint preallocated map slots on workers instead,
    // removing this preparation work from the update thread.
    if (mTaskGraphSerialMode) {
      if (batch.numBodies > 0) {
        if (batch.numConstraints > 0 && batch.constraints) {
          buildContactMap(batch.contactMap, info.mapStorage[0],
                          batch.constraints, batch.numBodies,
                          batch.numConstraints);
        } else
          releaseMapStorage(info.mapStorage[0]);
        if (batch.numD6 > 0 && batch.d6Joints) {
          buildD6Map(batch.d6Map, info.mapStorage[1], batch.d6Joints,
                     batch.numBodies, batch.numD6);
        } else
          releaseMapStorage(info.mapStorage[1]);
        if (batch.numGear > 0 && batch.gearJoints) {
          buildGearMap(batch.gearMap, info.mapStorage[2], batch.gearJoints,
                       batch.numBodies, batch.numGear);
        } else
          releaseMapStorage(info.mapStorage[2]);
      } else {
        releaseMapStorage(info.mapStorage[0]);
        releaseMapStorage(info.mapStorage[1]);
        releaseMapStorage(info.mapStorage[2]);
      }
    }

    if (!mTaskGraphSerialMode) {
      for (PxU32 mapIndex = 0; mapIndex < 3; ++mapIndex) {
        batch.mapStorage[mapIndex] = info.mapStorage[mapIndex];
        clearMapStorage(info.mapStorage[mapIndex]);
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

    if (mTaskGraphSerialMode) {
      AvbdSolverStats stats = {};
      // Single island entry: classification + shared post-AL inside solveIsland.
      mSolver.solveIsland(
          dt, batch.bodies, batch.numBodies, batch.constraints,
          batch.numConstraints, gravity, batch.d6Joints, batch.numD6,
          batch.gearJoints, batch.numGear, &batch.contactMap, &batch.d6Map,
          &batch.gearMap, batch.colorBatches, batch.numColors,
          batch.iterationOverride, batch.softParticles,
          batch.numSoftParticles, batch.softBodies, batch.numSoftBodies,
          batch.softContacts, batch.numSoftContacts,
          &batch.softExecutionPlan,
          batch.articulationForBody, batch.linkIndexForBody, stats);

      // Write back lambda cache inline
      writeLambdaToCache(*this, batch.constraints, batch.numConstraints,
                         batch.numBodies);
      if (batch.contactOutputTokens && batch.contactOutputResults) {
        writeContactImpulseToOutputTokens(
            batch.constraints, batch.contactOutputTokens,
            batch.contactOutputResults, batch.numConstraints, batch.numBodies,
            dt);
      } else {
        writeContactImpulseToOutput(batch.constraints, batch.numConstraints,
                                    batch.numBodies, dt);
      }
      writeJointLambdaToCache(*this, batch.d6Joints, batch.numD6);
      writeJointConstraintWriteback(*this, batch.d6Joints, batch.numD6, dt);
      // Release constraint maps
      PxAllocatorCallback &alloc = getAllocator();
      batch.contactMap.release(alloc);
      batch.d6Map.release(alloc);
      batch.gearMap.release(alloc);
    } else {
      AvbdSolveIslandTask *solveTask =
          mTaskFactory->createSolveTask(*this, mSolver, batch, dt, gravity);
      solveTask->setContinuation(coordTask);
      solveTask->removeReference();
    }
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

void AvbdDynamicsContext::commitContactPrepSnapshots(
    AvbdContactPrepSnapshot *snapshots, PxU32 startContactIdx,
    PxU32 numContacts, PxU16 frameStamp) {
  if (!snapshots)
    return;
  for (PxU32 contactOrder = 0; contactOrder < numContacts; ++contactOrder) {
    AvbdContactPrepSnapshot &snapshot = snapshots[startContactIdx + contactOrder];
    if (!snapshot.managerStateCommit)
      continue;
    snapshot.managerStateCommit = 0;
    if (snapshot.contactManagerIndex >= mContactManagerStateCache.size())
      continue;
    CachedContactManagerState &state =
        mContactManagerStateCache[snapshot.contactManagerIndex];
    state.key = snapshot.contactManagerKey;
    state.frameStamp = frameStamp;
  }
}

//=============================================================================
// Internal Methods
//=============================================================================

void AvbdDynamicsContext::setupBodies(AvbdSolverBody *avbdBodies,
                                      PxsRigidBody **rigidBodies,
                                      PxU32 numBodies, PxReal dt,
                                      const PxVec3 &gravity) {
  PX_UNUSED(gravity);

  for (PxU32 i = 0; i < numBodies; ++i) {
    PxsRigidBody *rigidBody = rigidBodies[i];
    if (rigidBody) {
      const PxsBodyCore &core = rigidBody->getCore();
      copyToAvbdSolverBody(core, avbdBodies[i], i, dt);
      restoreAndUpdateBodyVelocityHistory(core, avbdBodies[i]);
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

static bool prepareHardMimicAxis(
    const ArticulationJointCore &jointCore,
    PxArticulationAxis::Enum axis, const PxsBodyCore &parentBody,
    const PxsBodyCore &childBody, PxVec3 &worldAxis, PxReal &coordinate,
    bool &linearAxis) {
  if (jointCore.motion[axis] != PxArticulationMotion::eFREE ||
      jointCore.childPose.p.magnitudeSquared() > 1e-8f)
    return false;

  PxU32 activeDofs = 0;
  for (PxU32 i = 0; i < PxArticulationAxis::eCOUNT; ++i)
    activeDofs +=
        jointCore.motion[i] != PxArticulationMotion::eLOCKED ? 1u : 0u;
  if (activeDofs != 1)
    return false;

  PxU32 component = 0;
  switch (axis) {
  case PxArticulationAxis::eX:
  case PxArticulationAxis::eTWIST:
    component = 0;
    break;
  case PxArticulationAxis::eY:
  case PxArticulationAxis::eSWING1:
    component = 1;
    break;
  case PxArticulationAxis::eZ:
  case PxArticulationAxis::eSWING2:
    component = 2;
    break;
  default:
    return false;
  }

  PxVec3 localAxis(0.0f);
  localAxis[component] = 1.0f;
  const PxTransform parentFrame =
      parentBody.body2World * jointCore.parentPose;
  const PxTransform childFrame =
      childBody.body2World * jointCore.childPose;
  worldAxis = parentFrame.q.rotate(localAxis);
  linearAxis = axis >= PxArticulationAxis::eX;
  if (linearAxis) {
    coordinate = (childFrame.p - parentFrame.p).dot(worldAxis);
  } else {
    PxQuat relative = parentFrame.q.getConjugate() * childFrame.q;
    if (relative.w < 0.0f)
      relative = -relative;
    coordinate =
        2.0f * PxAtan2((&relative.x)[component], relative.w);
  }
  return worldAxis.isFinite() && PxIsFinite(coordinate);
}

static void prepareArticulationInternalJoints(
  AvbdDynamicsContext &context, FeatherstoneArticulation *articulation,
  PxU32 firstBodyIndex,
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
        // Articulation coordinates measure frame B relative to A, while the
        // shared D6 angular-error row measures A relative to B.  Convert the
        // interval into that solver convention: [low, high] -> [-high, -low].
        c.angularLimitLower.x =
            -jointCore->limits[PxArticulationAxis::eTWIST].high;
        c.angularLimitUpper.x =
            -jointCore->limits[PxArticulationAxis::eTWIST].low;
      }
      if (jointCore->motion[PxArticulationAxis::eSWING1] ==
          PxArticulationMotion::eLIMITED) {
        c.angularLimitLower.y =
            -jointCore->limits[PxArticulationAxis::eSWING1].high;
        c.angularLimitUpper.y =
            -jointCore->limits[PxArticulationAxis::eSWING1].low;
      }
      if (jointCore->motion[PxArticulationAxis::eSWING2] ==
          PxArticulationMotion::eLIMITED) {
        c.angularLimitLower.z =
            -jointCore->limits[PxArticulationAxis::eSWING2].high;
        c.angularLimitUpper.z =
            -jointCore->limits[PxArticulationAxis::eSWING2].low;
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
      // Force drives keep the legacy capped (S+D)/dt^2 penalty. Angular
      // acceleration drives instead preserve S and D for the native implicit
      // coefficient based on dt*S+D and inverse joint response.
      // ---------------------------------------------------------------
      const PxReal maxDriveStiffness = 100.0f;
      const PxReal maxDrivePenalty = 2.0f * c.header.rho;
      const PxReal maxDriveStiffnessFromPenalty =
          maxDrivePenalty / artInvDt2;
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
          if (drive.driveType == PxArticulationDriveType::eACCELERATION)
            c.driveAccelerationFlags |= (1u << a);

          // Compute current joint displacement along the driven axis
          PxVec3 localAxis(0.0f);
          (&localAxis.x)[a] = 1.0f;
          PxVec3 worldAxis = worldFrameA_drive.rotate(localAxis);
          PxReal totalSD = PxMin(drive.stiffness + drive.damping,
                                 PxMin(maxDriveStiffness,
                                       maxDriveStiffnessFromPenalty));
          if (totalSD <= 0.0f)
            continue;
          c.driveFlags |= (1u << a); // bit 0=X, 1=Y, 2=Z
          (&c.linearDamping.x)[a] = totalSD;
          PxReal currentQ = anchorSep.dot(worldAxis);

          // Position-error spring: drive toward (targetP - currentQ)
          PxReal targetP = jointCore->targetP[linAxes[a]];
          PxReal posVel = (targetP - currentQ) * invDt;
          PxReal velVel = jointCore->targetV[linAxes[a]];
          PxReal invSD = 1.0f / totalSD;
          PxReal sClamped = PxMin(drive.stiffness,
                                  PxMin(maxDriveStiffness,
                                        maxDriveStiffnessFromPenalty));
          PxReal dClamped = totalSD - sClamped;
          (&c.linearStiffness.x)[a] = sClamped;
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

          const PxReal targetAng = jointCore->targetP[angAxes[a]];
          const PxReal targetVel = jointCore->targetV[angAxes[a]];
          const PxReal geomError = targetAng - currentAngle;

          if (drive.driveType == PxArticulationDriveType::eACCELERATION) {
            // Reduced-coordinate acceleration drives are implicit in
            // dt * stiffness + damping, not force-drive penalties scaled by
            // 1 / dt^2. Keep the physical coefficients separate so the
            // solver can apply the matching inverse-response scaling.
            const PxReal stiffness = PxMax(0.0f, drive.stiffness);
            const PxReal damping = PxMax(0.0f, drive.damping);
            const PxReal effectiveRate = dt * stiffness + damping;
            if (effectiveRate <= 0.0f)
              continue;

            c.driveFlags |= angBits[a];
            c.driveAccelerationFlags |= angBits[a];
            (&c.angularStiffness.x)[a] = stiffness;
            (&c.angularDamping.x)[a] = damping;
            // The shared D6 axis row stores angular targets in the PhysX
            // (wA - wB) convention, while articulation coordinates are
            // measured as frame B relative to frame A.
            (&c.driveAngularVelocity.x)[a] =
                -(damping * targetVel + stiffness * geomError) / effectiveRate;
            continue;
          }

          PxReal totalSD = PxMin(drive.stiffness + drive.damping,
                                 PxMin(maxDriveStiffness,
                                       maxDriveStiffnessFromPenalty));
          if (totalSD <= 0.0f)
            continue;
          c.driveFlags |= angBits[a];
          (&c.angularDamping.x)[a] = totalSD;

          PxReal posVel = geomError * invDt;
          PxReal invSD = 1.0f / totalSD;
          PxReal sClamped = PxMin(drive.stiffness,
                                  PxMin(maxDriveStiffness,
                                        maxDriveStiffnessFromPenalty));
          PxReal dClamped = totalSD - sClamped;
          (&c.angularStiffness.x)[a] = sClamped;
          // Convert the articulation (B relative to A) target into the
          // shared D6 axis-drive convention.
          (&c.driveAngularVelocity.x)[a] =
              -(sClamped * invSD * posVel + dClamped * invSD * targetVel);
        }
      }

      restoreJointLambdaFromCache(
          context, c,
          reinterpret_cast<PxU64>(jointCore));

      numD6++;
    }
  }

  ArticulationMimicJointCore **mimicCores =
      artData.getMimicJointCores();
  const PxU32 mimicCount = artData.getMimicJointCount();
  for (PxU32 mimicIndex = 0;
       mimicIndex < mimicCount && numD6 < maxD6;
       ++mimicIndex) {
    const ArticulationMimicJointCore *mimic = mimicCores[mimicIndex];
    if (!mimic || !PxIsFinite(mimic->gearRatio) ||
        !PxIsFinite(mimic->offset) || PxAbs(mimic->gearRatio) < 1e-6f ||
        !PxIsFinite(mimic->naturalFrequency) ||
        !PxIsFinite(mimic->dampingRatio) ||
        mimic->naturalFrequency < 0.0f ||
        mimic->dampingRatio < 0.0f ||
        mimic->axisA >= PxArticulationAxis::eCOUNT ||
        mimic->axisB >= PxArticulationAxis::eCOUNT ||
        mimic->linkA == 0 || mimic->linkB == 0 ||
        mimic->linkA >= linkCount || mimic->linkB >= linkCount ||
        mimic->linkA == mimic->linkB)
      continue;

    const ArticulationLink &linkA = artData.getLink(mimic->linkA);
    const ArticulationLink &linkB = artData.getLink(mimic->linkB);
    if (linkA.parent == DY_ARTICULATION_LINK_NONE ||
        linkA.parent != linkB.parent)
      continue;
    const ArticulationLink &parent = artData.getLink(linkA.parent);
    if (!parent.bodyCore ||
        (!parent.bodyCore->fixedBaseLink &&
         parent.bodyCore->inverseMass > 0.0f) ||
        !linkA.bodyCore || !linkB.bodyCore ||
        !linkA.inboundJoint || !linkB.inboundJoint)
      continue;

    PxVec3 worldAxisA(0.0f), worldAxisB(0.0f);
    PxReal coordinateA = 0.0f, coordinateB = 0.0f;
    bool linearA = false, linearB = false;
    if (!prepareHardMimicAxis(
            *linkA.inboundJoint,
            static_cast<PxArticulationAxis::Enum>(mimic->axisA),
            *parent.bodyCore, *linkA.bodyCore, worldAxisA, coordinateA,
            linearA) ||
        !prepareHardMimicAxis(
            *linkB.inboundJoint,
            static_cast<PxArticulationAxis::Enum>(mimic->axisB),
            *parent.bodyCore, *linkB.bodyCore, worldAxisB, coordinateB,
            linearB))
      continue;

    AvbdD6JointConstraint &c = d6Constraints[numD6];
    c.initDefaults();
    c.header.type = AvbdConstraintType::eJOINT_CUSTOM_1D;
    c.header.bodyIndexA = firstBodyIndex + mimic->linkA;
    c.header.bodyIndexB = firstBodyIndex + mimic->linkB;
    const bool compliant =
        mimic->naturalFrequency > 0.0f && mimic->dampingRatio > 0.0f;
    const PxReal invDt =
        dt > 0.0f ? 1.0f / dt : 60.0f;
    const PxReal massA =
        linkA.bodyCore->inverseMass > 0.0f
            ? 1.0f / linkA.bodyCore->inverseMass
            : 1.0f;
    const PxReal massB =
        linkB.bodyCore->inverseMass > 0.0f
            ? 1.0f / linkB.bodyCore->inverseMass
            : 1.0f;
    c.header.rho = compliant ? 0.0f :
        PxMax(10.0f * PxMax(massA, massB) * invDt * invDt,
              AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH);
    c.header.compliance = 0.0f;
    c.header.damping = compliant ? 0.0f :
        AvbdConstants::AVBD_CONSTRAINT_DAMPING;
    c.linearMotion = 0x2A;
    c.angularMotion = 0x2A;
    c.sourceFlags |= compliant
        ? AvbdD6JointConstraint::eARTICULATION_COMPLIANT_MIMIC_ROW
        : (AvbdD6JointConstraint::eGENERIC_HARD_1D_ROW |
           AvbdD6JointConstraint::eARTICULATION_MIMIC_ROW);

    if (linearA)
      c.genericLinearA = worldAxisA;
    else
      c.genericAngularA = worldAxisA;
    if (linearB)
      c.genericLinearB = worldAxisB * mimic->gearRatio;
    else
      c.genericAngularB = worldAxisB * mimic->gearRatio;
    c.genericGeometricError =
        coordinateA + mimic->gearRatio * coordinateB + mimic->offset;
    c.genericVelocityTarget = 0.0f;
    c.genericMinImpulse = -PX_MAX_REAL;
    c.genericMaxImpulse = PX_MAX_REAL;
    c.genericNaturalFrequency = mimic->naturalFrequency;
    c.genericDampingRatio = mimic->dampingRatio;
    c.genericReferencePositionA = linkA.bodyCore->body2World.p;
    c.genericReferencePositionB = linkB.bodyCore->body2World.p;
    c.genericReferenceRotationA = linkA.bodyCore->body2World.q;
    c.genericReferenceRotationB = linkB.bodyCore->body2World.q;
    c.genericRowFlags = 0;
    ++numD6;
  }

  // Admit the fixed-root, two-active-joint fixed-tendon topologies covered by
  // the headless TGS authority:
  //   * serial centered angular joints
  //   * sibling branch centered angular or linear joints
  // The public tendon length is coeffA*qA + coeffB*qB + offset.  Expanding a
  // serial angular path into maximal body coordinates gives
  //   JA = coeffA*axisA - coeffB*axisB, JB = coeffB*axisB.
  // A sibling branch has one independent coordinate on each endpoint and
  // therefore uses JA = coeffA*axisA, JB = coeffB*axisB.  Length limits share
  // the same row and are expressed below in rest-error coordinates.
  // Asymmetric reciprocal coefficients remain excluded: supporting them
  // requires distinct length and response Jacobians rather than treating
  // recipCoefficient as a synonym for coefficient.
  auto prepareFixedTendon = [&]() {
  if (artData.getFixedTendonCount() != 1 || numD6 >= maxD6 ||
      !d6Constraints)
    return;
  ArticulationFixedTendon *tendon = artData.getFixedTendon(0);
  const bool limitActive =
      tendon && PxIsFinite(tendon->mLimitStiffness) &&
      tendon->mLimitStiffness > 0.0f;
  if (!tendon || tendon->getNumJoints() != 3 ||
      !PxIsFinite(tendon->mStiffness) || tendon->mStiffness < 0.0f ||
      !PxIsFinite(tendon->mDamping) || tendon->mDamping < 0.0f ||
      !PxIsFinite(tendon->mLimitStiffness) ||
      tendon->mLimitStiffness < 0.0f ||
      !PxIsFinite(tendon->mOffset) || !PxIsFinite(tendon->mRestLength) ||
      (tendon->mStiffness <= 0.0f && !limitActive) ||
      (limitActive &&
       (!PxIsFinite(tendon->mLowLimit) ||
        !PxIsFinite(tendon->mHighLimit) ||
        tendon->mLowLimit > tendon->mHighLimit)) ||
      (!limitActive &&
       (tendon->mLowLimit != PX_MAX_F32 ||
        tendon->mHighLimit != -PX_MAX_F32)))
    return;

  ArticulationTendonJoint *tendonJoints = tendon->getTendonJoints();
  const ArticulationTendonJoint &rootTendonJoint = tendonJoints[0];
  const ArticulationTendonJoint &tendonJointA = tendonJoints[1];
  const ArticulationTendonJoint &tendonJointB = tendonJoints[2];
  const ArticulationAttachmentBitField childA =
      ArticulationAttachmentBitField(1) << 1;
  const ArticulationAttachmentBitField childB =
      ArticulationAttachmentBitField(1) << 2;
  const bool serialTopology =
      rootTendonJoint.childCount == 1 &&
      rootTendonJoint.children == childA &&
      tendonJointA.parent == 0 && tendonJointA.childCount == 1 &&
      tendonJointA.children == childB &&
      tendonJointB.parent == 1 && tendonJointB.childCount == 0 &&
      tendonJointB.children == 0;
  const bool branchTopology =
      rootTendonJoint.childCount == 2 &&
      rootTendonJoint.children == (childA | childB) &&
      tendonJointA.parent == 0 && tendonJointA.childCount == 0 &&
      tendonJointA.children == 0 &&
      tendonJointB.parent == 0 && tendonJointB.childCount == 0 &&
      tendonJointB.children == 0;
  if (rootTendonJoint.parent != DY_ARTICULATION_ATTACHMENT_NONE ||
      rootTendonJoint.linkInd != 0 ||
      (!serialTopology && !branchTopology) ||
      tendonJointA.linkInd == 0 || tendonJointB.linkInd == 0 ||
      tendonJointA.linkInd >= linkCount ||
      tendonJointB.linkInd >= linkCount ||
      tendonJointA.axis >= PxArticulationAxis::eCOUNT ||
      tendonJointB.axis >= PxArticulationAxis::eCOUNT ||
      !PxIsFinite(tendonJointA.coefficient) ||
      !PxIsFinite(tendonJointB.coefficient) ||
      !PxIsFinite(tendonJointA.recipCoefficient) ||
      !PxIsFinite(tendonJointB.recipCoefficient) ||
      PxAbs(tendonJointA.coefficient) < 1e-6f ||
      PxAbs(tendonJointB.coefficient) < 1e-6f ||
      PxAbs(tendonJointA.coefficient -
            tendonJointA.recipCoefficient) > 1e-6f ||
      PxAbs(tendonJointB.coefficient -
            tendonJointB.recipCoefficient) > 1e-6f)
    return;

  const ArticulationLink &linkA =
      artData.getLink(tendonJointA.linkInd);
  const ArticulationLink &linkB =
      artData.getLink(tendonJointB.linkInd);
  if (linkA.parent != 0 ||
      (serialTopology && linkB.parent != tendonJointA.linkInd) ||
      (branchTopology && linkB.parent != 0) ||
      !artData.getLink(0).bodyCore ||
      !artData.getLink(0).bodyCore->fixedBaseLink ||
      !linkA.bodyCore || !linkB.bodyCore ||
      !linkA.inboundJoint || !linkB.inboundJoint)
    return;

  PxVec3 worldAxisA(0.0f), worldAxisB(0.0f);
  PxReal coordinateA = 0.0f, coordinateB = 0.0f;
  bool linearA = false, linearB = false;
  if (!prepareHardMimicAxis(
          *linkA.inboundJoint,
          static_cast<PxArticulationAxis::Enum>(tendonJointA.axis),
          *artData.getLink(0).bodyCore, *linkA.bodyCore,
          worldAxisA, coordinateA, linearA) ||
      !prepareHardMimicAxis(
          *linkB.inboundJoint,
          static_cast<PxArticulationAxis::Enum>(tendonJointB.axis),
          *(serialTopology ? linkA.bodyCore
                           : artData.getLink(0).bodyCore),
          *linkB.bodyCore, worldAxisB, coordinateB, linearB) ||
      (serialTopology && (linearA || linearB)) ||
      (branchTopology && linearA != linearB))
    return;

  AvbdD6JointConstraint &c = d6Constraints[numD6];
  c.initDefaults();
  c.header.type = AvbdConstraintType::eJOINT_CUSTOM_1D;
  c.header.bodyIndexA = firstBodyIndex + tendonJointA.linkInd;
  c.header.bodyIndexB = firstBodyIndex + tendonJointB.linkInd;
  const PxReal activeJointScale = 0.5f;
  c.header.rho = tendon->mStiffness * activeJointScale;
  c.header.compliance = 0.0f;
  c.header.damping = tendon->mDamping * activeJointScale;
  c.linearMotion = 0x2A;
  c.angularMotion = 0x2A;
  c.sourceFlags |=
      AvbdD6JointConstraint::eARTICULATION_FIXED_TENDON_ROW;
  if (branchTopology) {
    if (linearA) {
      c.genericLinearA =
          worldAxisA * tendonJointA.coefficient;
      c.genericLinearB =
          worldAxisB * tendonJointB.coefficient;
    } else {
      c.genericAngularA =
          worldAxisA * tendonJointA.coefficient;
      c.genericAngularB =
          worldAxisB * tendonJointB.coefficient;
    }
  } else {
    c.genericAngularA =
        worldAxisA * tendonJointA.coefficient -
        worldAxisB * tendonJointB.coefficient;
    c.genericAngularB =
        worldAxisB * tendonJointB.coefficient;
  }
  c.genericGeometricError =
      tendonJointA.coefficient * coordinateA +
      tendonJointB.coefficient * coordinateB +
      tendon->mOffset - tendon->mRestLength;
  c.genericTendonLowLimit =
      limitActive ? tendon->mLowLimit - tendon->mRestLength : 0.0f;
  c.genericTendonHighLimit =
      limitActive ? tendon->mHighLimit - tendon->mRestLength : 0.0f;
  c.genericTendonLimitStiffness =
      tendon->mLimitStiffness * activeJointScale;
  c.genericVelocityTarget = 0.0f;
  c.genericMinImpulse = -PX_MAX_REAL;
  c.genericMaxImpulse = PX_MAX_REAL;
  c.genericReferencePositionA = linkA.bodyCore->body2World.p;
  c.genericReferencePositionB = linkB.bodyCore->body2World.p;
  c.genericReferenceRotationA = linkA.bodyCore->body2World.q;
  c.genericReferenceRotationB = linkB.bodyCore->body2World.q;
  c.genericRowFlags = 0;
  ++numD6;
  };
  prepareFixedTendon();

  // A spatial tendon produces one constraint per leaf. Intermediate
  // attachments contribute to path length and its current linearization, but
  // PhysX applies the opposing tendon forces only at the root attachment and
  // the leaf attachment. This implementation admits a fixed articulation
  // root with centered, single-DOF sibling endpoints; the endpoint DOFs may
  // be angular or linear, while intermediate attachments may move.
  auto prepareSpatialTendon =
      [&](ArticulationSpatialTendon *spatialTendon) {
    if (!spatialTendon || !d6Constraints)
      return;
    const PxU32 attachmentCount =
        spatialTendon->getNumAttachments();
    const bool limitActive =
        PxIsFinite(spatialTendon->mLimitStiffness) &&
        spatialTendon->mLimitStiffness > 0.0f;
    if (attachmentCount < 2 || attachmentCount > 64 ||
        !PxIsFinite(spatialTendon->mStiffness) ||
        spatialTendon->mStiffness < 0.0f ||
        !PxIsFinite(spatialTendon->mDamping) ||
        spatialTendon->mDamping < 0.0f ||
        !PxIsFinite(spatialTendon->mOffset) ||
        !PxIsFinite(spatialTendon->mLimitStiffness) ||
        spatialTendon->mLimitStiffness < 0.0f ||
        (spatialTendon->mStiffness <= 0.0f && !limitActive))
      return;

    ArticulationAttachment *attachments =
        spatialTendon->getAttachments();
    const ArticulationAttachmentBitField validChildren =
        attachmentCount == 64
            ? ~ArticulationAttachmentBitField(0)
            : (ArticulationAttachmentBitField(1) <<
               attachmentCount) - 1;
    PxU32 leafCount = 0;
    for (PxU32 attachmentIndex = 0;
         attachmentIndex < attachmentCount;
         ++attachmentIndex) {
      const ArticulationAttachment &attachment =
          attachments[attachmentIndex];
      if (attachment.linkInd >= linkCount ||
          !artData.getLink(attachment.linkInd).bodyCore ||
          !attachment.relativeOffset.isFinite() ||
          !PxIsFinite(attachment.coefficient) ||
          PxAbs(attachment.coefficient) < 1e-6f ||
          (attachment.children & ~validChildren) != 0)
        return;

      PxU32 countedChildren = 0;
      for (ArticulationAttachmentBitField children =
               attachment.children;
           children != 0; children &= children - 1) {
        const PxU32 childIndex = PxLowestSetBit(children);
        if (childIndex == attachmentIndex ||
            attachments[childIndex].parent != attachmentIndex)
          return;
        ++countedChildren;
      }
      if (countedChildren != attachment.childCount)
        return;

      if (attachmentIndex == 0) {
        if (attachment.parent !=
                DY_ARTICULATION_ATTACHMENT_NONE ||
            attachment.childCount == 0)
          return;
      } else {
        if (attachment.parent >= attachmentCount ||
            (attachments[attachment.parent].children &
             (ArticulationAttachmentBitField(1) <<
              attachmentIndex)) == 0)
          return;
        PxU32 ancestor = attachmentIndex;
        for (PxU32 depth = 0; depth < attachmentCount; ++depth) {
          ancestor = attachments[ancestor].parent;
          if (ancestor == 0)
            break;
          if (ancestor >= attachmentCount)
            return;
        }
        if (ancestor != 0)
          return;
      }

      if (attachment.childCount == 0 &&
          attachment.parent != DY_ARTICULATION_ATTACHMENT_NONE) {
        if (!PxIsFinite(attachment.restLength) ||
            (limitActive &&
             (!PxIsFinite(attachment.lowLimit) ||
              !PxIsFinite(attachment.highLimit) ||
              attachment.lowLimit > attachment.highLimit)) ||
            (!limitActive &&
             (attachment.lowLimit != PX_MAX_F32 ||
              attachment.highLimit != -PX_MAX_F32)))
          return;
        ++leafCount;
      }
    }
    if (leafCount == 0)
      return;

    const ArticulationAttachment &rootAttachment =
        attachments[0];
    const ArticulationLink &rootEndpoint =
        artData.getLink(rootAttachment.linkInd);
    const ArticulationLink &articulationRoot =
        artData.getLink(0);
    if (rootAttachment.linkInd == 0 ||
        rootEndpoint.parent != 0 ||
        !articulationRoot.bodyCore ||
        !articulationRoot.bodyCore->fixedBaseLink ||
        !rootEndpoint.inboundJoint)
      return;

    auto findSingleFreeAxis =
        [](const ArticulationJointCore &joint,
           PxArticulationAxis::Enum &axis) -> bool {
      axis = PxArticulationAxis::eCOUNT;
      for (PxU32 axisIndex = 0;
           axisIndex < PxArticulationAxis::eCOUNT;
           ++axisIndex) {
        if (joint.motion[axisIndex] ==
            PxArticulationMotion::eLOCKED)
          continue;
        if (axis != PxArticulationAxis::eCOUNT ||
            joint.motion[axisIndex] !=
                PxArticulationMotion::eFREE)
          return false;
        axis =
            static_cast<PxArticulationAxis::Enum>(axisIndex);
      }
      return axis < PxArticulationAxis::eCOUNT;
    };

    PxArticulationAxis::Enum rootAxis;
    if (!findSingleFreeAxis(
            *rootEndpoint.inboundJoint, rootAxis))
      return;
    PxVec3 worldAxisA(0.0f);
    PxReal coordinateA = 0.0f;
    bool linearAxisA = false;
    if (!prepareHardMimicAxis(
            *rootEndpoint.inboundJoint, rootAxis,
            *articulationRoot.bodyCore, *rootEndpoint.bodyCore,
            worldAxisA, coordinateA, linearAxisA))
      return;

    PxVec3 attachmentPoints[64];
    for (PxU32 attachmentIndex = 0;
         attachmentIndex < attachmentCount;
         ++attachmentIndex) {
      const ArticulationAttachment &attachment =
          attachments[attachmentIndex];
      const PxsBodyCore &body =
          *artData.getLink(attachment.linkInd).bodyCore;
      attachmentPoints[attachmentIndex] =
          body.body2World.transform(attachment.relativeOffset);
    }

    const PxVec3 armA =
        rootEndpoint.bodyCore->body2World.q.rotate(
            rootAttachment.relativeOffset);
    for (PxU32 leafIndex = 1;
         leafIndex < attachmentCount;
         ++leafIndex) {
      const ArticulationAttachment &leaf =
          attachments[leafIndex];
      if (leaf.childCount != 0)
        continue;
      if (numD6 >= maxD6)
        return;

      const ArticulationLink &leafEndpoint =
          artData.getLink(leaf.linkInd);
      if (leaf.linkInd == 0 ||
          leaf.linkInd == rootAttachment.linkInd ||
          leafEndpoint.parent != 0 ||
          !leafEndpoint.inboundJoint)
        continue;
      PxArticulationAxis::Enum leafAxis;
      if (!findSingleFreeAxis(
              *leafEndpoint.inboundJoint, leafAxis))
        continue;
      PxVec3 worldAxisB(0.0f);
      PxReal coordinateB = 0.0f;
      bool linearAxisB = false;
      if (!prepareHardMimicAxis(
              *leafEndpoint.inboundJoint, leafAxis,
              *articulationRoot.bodyCore, *leafEndpoint.bodyCore,
              worldAxisB, coordinateB, linearAxisB))
        continue;

      PxReal length =
          rootAttachment.coefficient *
          spatialTendon->mOffset;
      PxU32 currentIndex = leafIndex;
      PxU32 firstChildIndex = leafIndex;
      bool pathValid = true;
      while (currentIndex != 0) {
        const ArticulationAttachment &current =
            attachments[currentIndex];
        const PxU32 parentIndex = current.parent;
        const PxVec3 segment =
            attachmentPoints[currentIndex] -
            attachmentPoints[parentIndex];
        const PxReal distance = segment.magnitude();
        if (!(distance > 1e-4f) || !PxIsFinite(distance)) {
          pathValid = false;
          break;
        }
        length += current.coefficient * distance;
        if (parentIndex == 0)
          firstChildIndex = currentIndex;
        currentIndex = parentIndex;
      }
      if (!pathValid || !PxIsFinite(length))
        continue;

      const PxVec3 rootSegment =
          attachmentPoints[0] -
          attachmentPoints[firstChildIndex];
      const PxVec3 leafSegment =
          attachmentPoints[leafIndex] -
          attachmentPoints[leaf.parent];
      const PxReal rootDistance = rootSegment.magnitude();
      const PxReal leafDistance = leafSegment.magnitude();
      if (!(rootDistance > 1e-4f) ||
          !(leafDistance > 1e-4f) ||
          !PxIsFinite(rootDistance) ||
          !PxIsFinite(leafDistance))
        continue;

      const PxVec3 linearJacobianA =
          rootSegment *
          (attachments[firstChildIndex].coefficient /
           rootDistance);
      const PxVec3 linearJacobianB =
          leafSegment * (leaf.coefficient / leafDistance);
      const PxVec3 armB =
          leafEndpoint.bodyCore->body2World.q.rotate(
              leaf.relativeOffset);
      const PxVec3 angularJacobianA =
          armA.cross(linearJacobianA);
      const PxVec3 angularJacobianB =
          armB.cross(linearJacobianB);

      AvbdD6JointConstraint &spatialRow =
          d6Constraints[numD6];
      spatialRow.initDefaults();
      spatialRow.header.type =
          AvbdConstraintType::eJOINT_CUSTOM_1D;
      spatialRow.header.bodyIndexA =
          firstBodyIndex + rootAttachment.linkInd;
      spatialRow.header.bodyIndexB =
          firstBodyIndex + leaf.linkInd;
      spatialRow.header.rho = spatialTendon->mStiffness;
      spatialRow.header.compliance = 0.0f;
      spatialRow.header.damping = spatialTendon->mDamping;
      spatialRow.linearMotion = 0x2A;
      spatialRow.angularMotion = 0x2A;
      spatialRow.sourceFlags |=
          AvbdD6JointConstraint::
              eARTICULATION_SPATIAL_TENDON_ROW;
      if (linearAxisA)
        spatialRow.genericLinearA =
            worldAxisA * linearJacobianA.dot(worldAxisA);
      else
        spatialRow.genericAngularA =
            worldAxisA * angularJacobianA.dot(worldAxisA);
      if (linearAxisB)
        spatialRow.genericLinearB =
            worldAxisB * linearJacobianB.dot(worldAxisB);
      else
        spatialRow.genericAngularB =
            worldAxisB * angularJacobianB.dot(worldAxisB);
      if (spatialRow.genericLinearA.magnitudeSquared() +
              spatialRow.genericAngularA.magnitudeSquared() <=
              1e-12f ||
          spatialRow.genericLinearB.magnitudeSquared() +
              spatialRow.genericAngularB.magnitudeSquared() <=
              1e-12f)
        continue;

      spatialRow.genericGeometricError =
          length - leaf.restLength;
      spatialRow.genericTendonLowLimit =
          limitActive
              ? leaf.lowLimit - leaf.restLength
              : 0.0f;
      spatialRow.genericTendonHighLimit =
          limitActive
              ? leaf.highLimit - leaf.restLength
              : 0.0f;
      spatialRow.genericTendonLimitStiffness =
          spatialTendon->mLimitStiffness;
      spatialRow.genericVelocityTarget = 0.0f;
      spatialRow.genericMinImpulse = -PX_MAX_REAL;
      spatialRow.genericMaxImpulse = PX_MAX_REAL;
      spatialRow.genericReferencePositionA =
          rootEndpoint.bodyCore->body2World.p;
      spatialRow.genericReferencePositionB =
          leafEndpoint.bodyCore->body2World.p;
      spatialRow.genericReferenceRotationA =
          rootEndpoint.bodyCore->body2World.q;
      spatialRow.genericReferenceRotationB =
          leafEndpoint.bodyCore->body2World.q;
      spatialRow.genericRowFlags = 0;
      ++numD6;
    }
  };
  for (PxU32 tendonIndex = 0;
       tendonIndex < artData.getSpatialTendonCount();
       ++tendonIndex)
    prepareSpatialTendon(
        artData.getSpatialTendon(tendonIndex));
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
  if (mHeapFallbackAllocations.size() > 0) {
    for (PxU32 i = 0; i < mHeapFallbackAllocations.size(); ++i) {
      if (mHeapFallbackAllocations[i]) {
        mAllocatorAdapter.deallocate(mHeapFallbackAllocations[i]);
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
    Cm::VirtualAllocatorCallback &allocator,
    PxsMaterialManager *materialManager, IG::SimpleIslandManager &islandManager,
    PxU64 contextID, PxReal maxBiasCoefficient, PxReal lengthScale,
    PxU32 avbdIterations, PxU32 avbdJointIterationOverride,
    bool avbdEnableEarlyStop, PxSceneFlags sceneFlags) {
  return PX_PLACEMENT_NEW(
      PX_ALLOC(sizeof(Dy::AvbdDynamicsContext), "AvbdDynamicsContext"),
      Dy::AvbdDynamicsContext)(
      memBlockPool, scratchAllocator, taskPool, simStats, taskManager,
      allocator, materialManager, islandManager, contextID, maxBiasCoefficient,
      lengthScale, avbdIterations, avbdJointIterationOverride,
      avbdEnableEarlyStop, sceneFlags);
}
