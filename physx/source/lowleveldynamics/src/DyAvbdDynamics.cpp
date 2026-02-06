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
#include "DyAvbdBodyConversion.h"
#include "DyAvbdConstraint.h"
#include "DyAvbdTasks.h"
#include "DyConstraint.h"
#include "DyIslandManager.h"
#include "PxContact.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsIslandManagerTypes.h"
#include "PxsRigidBody.h"
#include "PxsSimpleIslandManager.h"
#include "foundation/PxMath.h"
#include "DyFeatherstoneArticulation.h"
#include "DyArticulationCore.h"
#include "DyVArticulation.h"
#include "common/PxProfileZone.h"

using namespace physx;
using namespace physx::Dy;

// Global frame counter for motor deduplication
// This is incremented at the start of each update() call
static physx::PxU64 gAvbdMotorFrameCounter = 0;

// Accessor for the solver to get current frame
physx::PxU64 getAvbdMotorFrameCounter() {
  return gAvbdMotorFrameCounter;
}

// Helper struct for joint data protocol (must match SnippetAvbdDx11)
struct AvbdSnippetJointData {
  enum Type { eSPHERICAL = 0, eFIXED, eREVOLUTE, ePRISMATIC, eD6 };
  int type;
  PxVec3 pivot0;
  PxVec3 pivot1;
  PxVec3 axis0;
  PxVec3 axis1;
  float limitLow;
  float limitHigh;
  float padding[2];
};

// Standard PhysX JointData structure (mirrors ExtJointData.h)
// This is the base format for all standard PhysX joints
struct PhysXJointData {
  PxConstraintInvMassScale invMassScale;
  PxTransform32 c2b[2];  // Constraint-to-body transforms
};

// Mirror of PxJointLimitParameters (16 bytes with padding)
struct PhysXJointLimitParameters {
  PxReal restitution;
  PxReal bounceThreshold;
  PxReal stiffness;
  PxReal damping;
};

// Mirror of PxJointLimitCone
struct PhysXJointLimitCone : PhysXJointLimitParameters {
  PxReal yAngle;
  PxReal zAngle;
};

// Mirror of SphericalJointData (ExtSphericalJoint.h)
struct PhysXSphericalJointData : PhysXJointData {
  PhysXJointLimitCone limit;
  PxU16 jointFlags;  // PxSphericalJointFlags
};

// Mirror of FixedJointData (ExtFixedJoint.h) - no extra members
struct PhysXFixedJointData : PhysXJointData {
  // No additional members
};

// Mirror of PxJointAngularLimitPair
struct PhysXJointAngularLimitPair : PhysXJointLimitParameters {
  PxReal upper;
  PxReal lower;
};

// Mirror of PxJointLinearLimitPair
struct PhysXJointLinearLimitPair : PhysXJointLimitParameters {
  PxReal upper;
  PxReal lower;
};

// Mirror of RevoluteJointData (ExtRevoluteJoint.h)
struct PhysXRevoluteJointData : PhysXJointData {
  PxReal driveVelocity;
  PxReal driveForceLimit;
  PxReal driveGearRatio;
  PhysXJointAngularLimitPair limit;
  PxU16 jointFlags;  // PxRevoluteJointFlags
};

// Mirror of PrismaticJointData (ExtPrismaticJoint.h)
struct PhysXPrismaticJointData : PhysXJointData {
  PhysXJointLinearLimitPair limit;
  PxU16 jointFlags;  // PxPrismaticJointFlags
};

// Mirror of PxD6JointDrive
struct PhysXD6JointDrive {
  PxReal stiffness;
  PxReal damping;
  PxReal forceLimit;
  PxU32 flags;
};

// Mirror of PxJointLinearLimit
struct PhysXJointLinearLimit : PhysXJointLimitParameters {
  PxReal value;
};

// Mirror of PxJointLimitPyramid
struct PhysXJointLimitPyramid : PhysXJointLimitParameters {
  PxReal yAngleMin;
  PxReal yAngleMax;
  PxReal zAngleMin;
  PxReal zAngleMax;
};

// Mirror of D6JointData (ExtD6Joint.h) - partial, enough for type detection
struct PhysXD6JointData : PhysXJointData {
  PxU32 motion[6];  // PxD6Motion::Enum
  PhysXJointLinearLimit distanceLimit;
  PhysXJointLinearLimitPair linearLimitX;
  PhysXJointLinearLimitPair linearLimitY;
  PhysXJointLinearLimitPair linearLimitZ;
  PhysXJointAngularLimitPair twistLimit;
  PhysXJointLimitCone swingLimit;
  PhysXJointLimitPyramid pyramidSwingLimit;
  PhysXD6JointDrive drive[6];
  PxTransform drivePosition;
  PxVec3 driveLinearVelocity;
  PxVec3 driveAngularVelocity;
  PxU32 locked;
  PxU32 limited;
  PxU32 driving;
  // More members follow but not needed for detection
};

// Mirror of GearJointData (ExtGearJoint.h)
struct PhysXGearJointData : PhysXJointData {
  const void* hingeJoint0;  // PxBase* - either PxJoint or PxArticulationJointReducedCoordinate
  const void* hingeJoint1;  // PxBase* - either PxJoint or PxArticulationJointReducedCoordinate
  float gearRatio;
  float error;
};

// Enum for detected joint types (local copy for compatibility)
enum PhysXJointType {
  eJOINT_UNKNOWN = -1,
  eJOINT_FIXED = 0,
  eJOINT_SPHERICAL = 1,
  eJOINT_REVOLUTE = 2,
  eJOINT_PRISMATIC = 3,
  eJOINT_D6 = 4,
  eJOINT_GEAR = 5
};

// Helper function to convert Dy::ConstraintJointType to local PhysXJointType
static PhysXJointType getJointTypeFromConstraint(Dy::ConstraintJointType cjt) {
  switch (cjt) {
    case Dy::eCONSTRAINT_JOINT_SPHERICAL:  return eJOINT_SPHERICAL;
    case Dy::eCONSTRAINT_JOINT_REVOLUTE:   return eJOINT_REVOLUTE;
    case Dy::eCONSTRAINT_JOINT_PRISMATIC:  return eJOINT_PRISMATIC;
    case Dy::eCONSTRAINT_JOINT_FIXED:      return eJOINT_FIXED;
    case Dy::eCONSTRAINT_JOINT_D6:         return eJOINT_D6;
    case Dy::eCONSTRAINT_JOINT_GEAR:       return eJOINT_GEAR;
    default:                               return eJOINT_UNKNOWN;
  }
}

//=============================================================================
// Articulation Internal Joints Helper (forward declaration)
//=============================================================================
static void prepareArticulationInternalJoints(
    FeatherstoneArticulation* articulation,
    PxU32 firstBodyIndex,
    AvbdSphericalJointConstraint* sphericalConstraints, PxU32& numSpherical, PxU32 maxSpherical,
    AvbdFixedJointConstraint* fixedConstraints, PxU32& numFixed, PxU32 maxFixed,
    AvbdRevoluteJointConstraint* revoluteConstraints, PxU32& numRevolute, PxU32 maxRevolute,
    AvbdPrismaticJointConstraint* prismaticConstraints, PxU32& numPrismatic, PxU32 maxPrismatic);

//=============================================================================
// Helper function to find articulation link index from rigid core
//=============================================================================
static PxU32 findArticulationLinkIndex(
    FeatherstoneArticulation* articulation,
    const PxsRigidCore* rigidCore) {
    
  if (!articulation || !rigidCore)
    return PX_MAX_U32;
    
  ArticulationData& artData = articulation->getArticulationData();
  const PxU32 linkCount = artData.getLinkCount();
  
  for (PxU32 linkIdx = 0; linkIdx < linkCount; ++linkIdx) {
    const ArticulationLink& link = artData.getLink(linkIdx);
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
static void* allocWithFallback(
    PxcScratchAllocator& scratch,
    PxVirtualAllocatorCallback* mainAllocator,
    PxArray<void*>& fallbackAllocations,
    PxU32 size,
    const char* name) {
  
  // First try scratch allocator (no heap fallback) - fast path
  void* ptr = scratch.alloc(size, false);
  if (ptr) {
    return ptr;
  }
  
  // Scratch memory exhausted - use main allocator (slower path)
  if (mainAllocator) {
    PxAllocatorCallback* allocator = reinterpret_cast<PxAllocatorCallback*>(mainAllocator);
    ptr = allocator->allocate(size, name, __FILE__, __LINE__);
    if (ptr) {
      // Track for cleanup in mergeResults()
      // No mutex needed here since update() is called from a single thread context
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

  mExceededForceThresholdStream[0] = PX_NEW(ThresholdStream)(*allocatorCallback);
  mExceededForceThresholdStream[1] = PX_NEW(ThresholdStream)(*allocatorCallback);

  // Use the main allocator callback for tasks, NOT the scratch adapter.
  // Tasks need explicit deallocation, which ScratchAllocatorAdapter doesn't provide.
  mTaskFactory = new AvbdTaskFactory(mTaskManager, 
      *reinterpret_cast<PxAllocatorCallback*>(mAllocatorCallback));
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

void AvbdDynamicsContext::update(
    Cm::FlushPool &flushPool, PxBaseTask *continuation,
    PxBaseTask *postPartitioningTask, PxBaseTask *processLostTouchTask,
    PxvNphaseImplementationContext *nPhaseContext, PxU32 maxPatchesPerCM,
    PxU32 maxArticulationLinks, PxReal dt, const PxVec3 &gravity,
    PxBitMapPinned &changedHandleMap) {

  PX_PROFILE_ZONE("AVBD.update", mContextID);
  
  // Increment global frame counter for motor deduplication
  gAvbdMotorFrameCounter++;

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
  const PxU32 numDynamicBodies = islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);
  const PxU32 numArticulations = islandSim.getNbActiveNodes(IG::Node::eARTICULATION_TYPE);

  if (islandCount == 0) {
    return;
  }

  // Calculate total body count including articulation links
  PxU32 totalBodyCount = numDynamicBodies + maxArticulationLinks;

  // Allocate global arrays - use scratch with main allocator fallback
  AvbdSolverBody *avbdBodies = nullptr;
  PxsRigidBody **rigidBodies = nullptr;
  {
    PX_PROFILE_ZONE("AVBD.allocateMemory", mContextID);
    avbdBodies = reinterpret_cast<AvbdSolverBody *>
        (allocWithFallback(mScratchAllocator, mAllocatorCallback, 
                          mHeapFallbackAllocations,
                          sizeof(AvbdSolverBody) * totalBodyCount, "AvbdSolverBody"));

    rigidBodies = reinterpret_cast<PxsRigidBody **>
        (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                          mHeapFallbackAllocations,
                          sizeof(PxsRigidBody *) * totalBodyCount, "RigidBodies"));
  }

  // Check if allocation failed completely
  if (!avbdBodies || !rigidBodies) {
    return;
  }

  // Track articulation info for writeback
  FeatherstoneArticulation** articulationForBody = nullptr;
  PxU32* linkIndexForBody = nullptr;
  if (numArticulations > 0 && maxArticulationLinks > 0) {
    articulationForBody = reinterpret_cast<FeatherstoneArticulation**>
        (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                          mHeapFallbackAllocations,
                          sizeof(FeatherstoneArticulation*) * totalBodyCount, "ArticulationForBody"));
    linkIndexForBody = reinterpret_cast<PxU32*>
        (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                          mHeapFallbackAllocations,
                          sizeof(PxU32) * totalBodyCount, "LinkIndexForBody"));
    
    if (articulationForBody && linkIndexForBody) {
      for (PxU32 i = 0; i < totalBodyCount; ++i) {
        articulationForBody[i] = nullptr;
        linkIndexForBody[i] = PX_MAX_U32;
      }
    }
  }

  const PxU32 maxActiveNodes = numDynamicBodies + numArticulations + 1;
  PxU32 *bodyRemapTable = reinterpret_cast<PxU32 *>
      (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                        mHeapFallbackAllocations,
                        sizeof(PxU32) * maxActiveNodes, "BodyRemapTable"));
  
  if (!bodyRemapTable) {
    return;
  }
  
  for (PxU32 i = 0; i < maxActiveNodes; ++i) {
    bodyRemapTable[i] = PX_MAX_U32;
  }

  // Track articulation first link indices
  PxU32* articulationFirstLinkIndex = nullptr;
  FeatherstoneArticulation** articulationByActiveIdx = nullptr;
  if (numArticulations > 0) {
    articulationFirstLinkIndex = reinterpret_cast<PxU32*>(
        allocWithFallback(mScratchAllocator, mAllocatorCallback,
                          mHeapFallbackAllocations,
                          sizeof(PxU32) * (numArticulations + 1), "ArticulationFirstLinkIndex"));
    articulationByActiveIdx = reinterpret_cast<FeatherstoneArticulation**>(
        allocWithFallback(mScratchAllocator, mAllocatorCallback,
                          mHeapFallbackAllocations,
                          sizeof(FeatherstoneArticulation*) * (numArticulations + 1), "ArticulationByActiveIdx"));
    
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

  AvbdIslandInfo *islandInfos = reinterpret_cast<AvbdIslandInfo *>(
      allocWithFallback(mScratchAllocator, mAllocatorCallback,
                        mHeapFallbackAllocations,
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
      }
      else if (node.getNodeType() == IG::Node::eARTICULATION_TYPE) {
        FeatherstoneArticulation* articulation = 
            static_cast<FeatherstoneArticulation*>( 
                islandSim.getObject(currentIndex, IG::Node::eARTICULATION_TYPE));
        
        if (articulation) {
          const PxU32 activeNodeIdx = islandSim.getActiveNodeIndex(currentIndex);
          
          // Store first link index and articulation pointer
          if (articulationFirstLinkIndex && activeNodeIdx < numArticulations + 1) {
            articulationFirstLinkIndex[activeNodeIdx] = bodyIndex;
            articulationByActiveIdx[activeNodeIdx] = articulation;
          }
          
          ArticulationData& artData = articulation->getArticulationData();
          const PxU32 linkCount = artData.getLinkCount();
          
          for (PxU32 linkIdx = 0; linkIdx < linkCount && bodyIndex < totalBodyCount; ++linkIdx) {
            const ArticulationLink& link = artData.getLink(linkIdx);
            AvbdSolverBody& solverBody = avbdBodies[bodyIndex];
            
            const PxsBodyCore* bodyCore = link.bodyCore;
            if (bodyCore) {
              solverBody.position = bodyCore->body2World.p;
              solverBody.rotation = bodyCore->body2World.q;
              solverBody.linearVelocity = bodyCore->linearVelocity;
              solverBody.angularVelocity = bodyCore->angularVelocity;
              solverBody.invMass = bodyCore->inverseMass;
              
              PxMat33 R(bodyCore->body2World.q);
              PxMat33 invInertiaLocal = PxMat33::createDiagonal(bodyCore->inverseInertia);
              solverBody.invInertiaWorld = R * invInertiaLocal * R.getTranspose();
              

              solverBody.nodeIndex = bodyIndex;
              solverBody.colorGroup = 0;
              
              // Store for writeback
              if (articulationForBody && linkIndexForBody) {
                articulationForBody[bodyIndex] = articulation;
                linkIndexForBody[bodyIndex] = linkIdx;
              }
            } else {
              // Fallback: initialize as static
              initializeStaticAvbdBody(PxTransform(PxIdentity), solverBody, bodyIndex);
            }
            
            rigidBodies[bodyIndex] = nullptr;  // Mark as articulation link
            bodyIndex++;
          }
          
          // Count internal articulation joints
          info.articulationJointCount += (linkCount > 1) ? (linkCount - 1) : 0;
          
          if (activeNodeIdx < maxActiveNodes && articulationFirstLinkIndex) {
            bodyRemapTable[activeNodeIdx] = articulationFirstLinkIndex[activeNodeIdx];
          }
        }
      }
      currentIndex = node.mNextNode;
    }
    info.bodyCount = bodyIndex - info.bodyStart;
  }

  // 2. Gather contact edges per island
  const PxU32 nbActiveContacts = islandSim.getNbActiveEdges(IG::Edge::eCONTACT_MANAGER);
  mContactList.forceSize_Unsafe(0);
  mContactList.reserve((nbActiveContacts + 63u) & (~63u));

  PxU32 contactIndex = 0;
  for (PxU32 i = 0; i < islandCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];
    info.cmStart = contactIndex;

    const IG::Island &island = islandSim.getIsland(islandIds[i]);
    IG::EdgeIndex contactEdgeIndex = island.mFirstEdge[IG::Edge::eCONTACT_MANAGER];

    while (contactEdgeIndex != IG_INVALID_EDGE) {
      const IG::Edge &edge = islandSim.getEdge(contactEdgeIndex);
      PxsContactManager *contactManager = mIslandManager.getContactManager(contactEdgeIndex);

      if (contactManager) {
        const PxNodeIndex nodeIndex1 = islandSim.mCpuData.getNodeIndex1(contactEdgeIndex);
        const PxNodeIndex nodeIndex2 = islandSim.mCpuData.getNodeIndex2(contactEdgeIndex);
        const PxcNpWorkUnit& workUnit = contactManager->getWorkUnit();

        mContactList.pushBack(PxsIndexedContactManager(contactManager));
        PxsIndexedContactManager &icm = mContactList.back();

        // Set up body0
        if (!nodeIndex1.isStaticBody()) {
          const PxU32 activeIdx = islandSim.getActiveNodeIndex(nodeIndex1);
          if (activeIdx < maxActiveNodes && bodyRemapTable[activeIdx] != PX_MAX_U32) {
            // Check if this is an articulation link
            if ((workUnit.mFlags & PxcNpWorkUnitFlag::eARTICULATION_BODY0) && 
                articulationByActiveIdx && articulationFirstLinkIndex &&
                activeIdx < numArticulations + 1) {
              // Find the actual link index for this contact
              FeatherstoneArticulation* art = articulationByActiveIdx[activeIdx];
              PxU32 linkIdx = findArticulationLinkIndex(art, workUnit.mRigidCore0);
              if (linkIdx != PX_MAX_U32) {
                icm.indexType0 = PxsIndexedInteraction::eBODY;
                icm.solverBody0 = articulationFirstLinkIndex[activeIdx] + linkIdx;
              } else {
                // Fallback to first link if not found
                icm.indexType0 = PxsIndexedInteraction::eBODY;
                icm.solverBody0 = bodyRemapTable[activeIdx];
              }
            } else {
              icm.indexType0 = PxsIndexedInteraction::eBODY;
              icm.solverBody0 = bodyRemapTable[activeIdx];
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
          if (activeIdx < maxActiveNodes && bodyRemapTable[activeIdx] != PX_MAX_U32) {
            // Check if this is an articulation link
            if ((workUnit.mFlags & PxcNpWorkUnitFlag::eARTICULATION_BODY1) && 
                articulationByActiveIdx && articulationFirstLinkIndex &&
                activeIdx < numArticulations + 1) {
              // Find the actual link index for this contact
              FeatherstoneArticulation* art = articulationByActiveIdx[activeIdx];
              PxU32 linkIdx = findArticulationLinkIndex(art, workUnit.mRigidCore1);
              if (linkIdx != PX_MAX_U32) {
                icm.indexType1 = PxsIndexedInteraction::eBODY;
                icm.solverBody1 = articulationFirstLinkIndex[activeIdx] + linkIdx;
              } else {
                // Fallback to first link if not found
                icm.indexType1 = PxsIndexedInteraction::eBODY;
                icm.solverBody1 = bodyRemapTable[activeIdx];
              }
            } else {
              icm.indexType1 = PxsIndexedInteraction::eBODY;
              icm.solverBody1 = bodyRemapTable[activeIdx];
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
    config.outerIterations = 4;
    config.innerIterations = 8;
    config.initialRho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
    config.maxRho = AvbdConstants::AVBD_MAX_PENALTY_RHO;
    config.baumgarte = 0.3f;
    config.enableLocal6x6Solve = false;
    mSolver.initialize(
        config, *reinterpret_cast<PxAllocatorCallback *>(mAllocatorCallback));
    mSolverInitialized = true;
  }

  // 5. Allocate Constraints
  const PxU32 actualContactCount = mContactList.size();
  const PxU32 maxConstraints = actualContactCount * 4;
  AvbdContactConstraint *avbdConstraints = nullptr;
  if (maxConstraints > 0) {
    avbdConstraints = reinterpret_cast<AvbdContactConstraint *>
        (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                          mHeapFallbackAllocations,
                          sizeof(AvbdContactConstraint) * maxConstraints, "AvbdContactConstraint"));
  }

  PxU32 totalJoints = 0;
  PxU32 totalArticulationJoints = 0;
  for (PxU32 i = 0; i < islandCount; ++i) {
    totalJoints += islandInfos[i].constraintCount;
    totalArticulationJoints += islandInfos[i].articulationJointCount;
  }
  
  PxU32 totalJointCapacity = totalJoints + totalArticulationJoints;
  if (totalJointCapacity == 0) totalJointCapacity = 1;

  AvbdSphericalJointConstraint *sphericalJoints = reinterpret_cast<AvbdSphericalJointConstraint *>
      (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                        mHeapFallbackAllocations,
                        sizeof(AvbdSphericalJointConstraint) * totalJointCapacity, "SphericalJoints"));
  AvbdFixedJointConstraint *fixedJoints = reinterpret_cast<AvbdFixedJointConstraint *>
      (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                        mHeapFallbackAllocations,
                        sizeof(AvbdFixedJointConstraint) * totalJointCapacity, "FixedJoints"));
  AvbdRevoluteJointConstraint *revoluteJoints = reinterpret_cast<AvbdRevoluteJointConstraint *>
      (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                        mHeapFallbackAllocations,
                        sizeof(AvbdRevoluteJointConstraint) * totalJointCapacity, "RevoluteJoints"));
  AvbdPrismaticJointConstraint *prismaticJoints = reinterpret_cast<AvbdPrismaticJointConstraint *>
      (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                        mHeapFallbackAllocations,
                        sizeof(AvbdPrismaticJointConstraint) * totalJointCapacity, "PrismaticJoints"));
  AvbdD6JointConstraint *d6Joints = reinterpret_cast<AvbdD6JointConstraint *>
      (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                        mHeapFallbackAllocations,
                        sizeof(AvbdD6JointConstraint) * totalJointCapacity, "D6Joints"));
  AvbdGearJointConstraint *gearJoints = reinterpret_cast<AvbdGearJointConstraint *>
      (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                        mHeapFallbackAllocations,
                        sizeof(AvbdGearJointConstraint) * totalJointCapacity, "GearJoints"));

  // 6. Create Task Chain
  AvbdWriteBackTask *wbTask = mTaskFactory->createWriteBackTask(
      *this, avbdBodies, rigidBodies, bodyIndex,
      articulationForBody, linkIndexForBody);
  wbTask->setContinuation(continuation);

  AvbdCoordinatorTask *coordTask = mTaskFactory->createCoordinatorTask(*this, wbTask);
  coordTask->setContinuation(wbTask);

  PxU32 currentConstraintIdx = 0;
  PxU32 currSphericalIdx = 0;
  PxU32 currFixedIdx = 0;
  PxU32 currRevoluteIdx = 0;
  PxU32 currPrismaticIdx = 0;
  PxU32 currD6Idx = 0;
  PxU32 currGearIdx = 0;
  PxU32 tasksSpawned = 0;

  // Track color batch allocations for cleanup
  PxArray<AvbdColorBatch*> colorBatchAllocations;

  // 7. Iterate Islands
  for (PxU32 i = 0; i < islandCount; ++i) {
    AvbdIslandInfo &info = islandInfos[i];

    // Prepare contact constraints
    PxU32 numConstraints = 0;
    if (avbdConstraints && info.cmCount > 0) {
      numConstraints = prepareAvbdContacts(
          &avbdBodies[info.bodyStart], info.bodyCount,
          avbdConstraints + currentConstraintIdx,
          maxConstraints - currentConstraintIdx,
          info.cmStart, info.cmCount, info.bodyStart);
    }

    PxU32 numSpherical = 0;
    PxU32 numFixed = 0;
    PxU32 numRevolute = 0;
    PxU32 numPrismatic = 0;
    PxU32 numD6 = 0;
    PxU32 numGear = 0;

    // Prepare external joint constraints
    if (info.constraintCount > 0) {
      prepareAvbdConstraints(
          islandSim, &avbdBodies[info.bodyStart], info.bodyCount,
          info.bodyStart, 
          sphericalJoints + currSphericalIdx, numSpherical, totalJointCapacity - currSphericalIdx,
          fixedJoints + currFixedIdx, numFixed, totalJointCapacity - currFixedIdx,
          revoluteJoints + currRevoluteIdx, numRevolute, totalJointCapacity - currRevoluteIdx,
          prismaticJoints + currPrismaticIdx, numPrismatic, totalJointCapacity - currPrismaticIdx,
          d6Joints + currD6Idx, numD6, totalJointCapacity - currD6Idx,
          gearJoints + currGearIdx, numGear, totalJointCapacity - currGearIdx,
          i, bodyRemapTable);
    }
    
    // Prepare articulation internal joints
    if (info.articulationJointCount > 0 && articulationFirstLinkIndex) {
      const IG::Island &island = islandSim.getIsland(islandIds[i]);
      PxNodeIndex currentNodeIndex = island.mRootNode;
      
      while (currentNodeIndex.isValid()) {
        const IG::Node &node = islandSim.getNode(currentNodeIndex);
        
        if (node.getNodeType() == IG::Node::eARTICULATION_TYPE) {
          FeatherstoneArticulation* articulation = 
              static_cast<FeatherstoneArticulation*>( 
                  islandSim.getObject(currentNodeIndex, IG::Node::eARTICULATION_TYPE));
          
          if (articulation) {
            const PxU32 activeNodeIdx = islandSim.getActiveNodeIndex(currentNodeIndex);
            PxU32 artFirstBodyIdx = PX_MAX_U32;
            if (activeNodeIdx < numArticulations + 1) {
              artFirstBodyIdx = articulationFirstLinkIndex[activeNodeIdx];
            }
            
            if (artFirstBodyIdx >= info.bodyStart && 
                artFirstBodyIdx < info.bodyStart + info.bodyCount) {
              PxU32 localFirstBodyIdx = artFirstBodyIdx - info.bodyStart;
              
              // Prepare articulation internal joints - dispatch to correct joint type
              PxU32 artSpherical = 0, artFixed = 0, artRevolute = 0, artPrismatic = 0;
              prepareArticulationInternalJoints(
                  articulation,
                  localFirstBodyIdx,
                  sphericalJoints + currSphericalIdx + numSpherical, artSpherical, 
                  totalJointCapacity - currSphericalIdx - numSpherical,
                  fixedJoints + currFixedIdx + numFixed, artFixed,
                  totalJointCapacity - currFixedIdx - numFixed,
                  revoluteJoints + currRevoluteIdx + numRevolute, artRevolute,
                  totalJointCapacity - currRevoluteIdx - numRevolute,
                  prismaticJoints + currPrismaticIdx + numPrismatic, artPrismatic,
                  totalJointCapacity - currPrismaticIdx - numPrismatic);
              
              numSpherical += artSpherical;
              numFixed += artFixed;
              numRevolute += artRevolute;
              numPrismatic += artPrismatic;
            }
          }
        }
        currentNodeIndex = node.mNextNode;
      }
    }

    // Skip empty islands
    if (numConstraints == 0 && numSpherical == 0 && numFixed == 0 &&
        numRevolute == 0 && numPrismatic == 0 && numD6 == 0 && numGear == 0 && info.bodyCount == 0) {
      continue;
    }

    // Create Island Batch
    AvbdIslandBatch batch;
    batch.bodies = &avbdBodies[info.bodyStart];
    batch.numBodies = info.bodyCount;
    batch.constraints = avbdConstraints ? &avbdConstraints[currentConstraintIdx] : nullptr;
    batch.numConstraints = numConstraints;

    batch.sphericalJoints = &sphericalJoints[currSphericalIdx];
    batch.numSpherical = numSpherical;
    batch.fixedJoints = &fixedJoints[currFixedIdx];
    batch.numFixed = numFixed;
    batch.revoluteJoints = &revoluteJoints[currRevoluteIdx];
    batch.numRevolute = numRevolute;
    batch.prismaticJoints = &prismaticJoints[currPrismaticIdx];
    batch.numPrismatic = numPrismatic;
    batch.d6Joints = &d6Joints[currD6Idx];
    batch.numD6 = numD6;
    batch.gearJoints = &gearJoints[currGearIdx];
    batch.numGear = numGear;

    batch.islandStart = i;
    batch.islandEnd = i + 1;
    batch.colorBatches = nullptr;
    batch.numColors = 0;
    
    // Build constraint-to-body mappings for O(1) lookup in solver
    // This eliminates O(N^2) complexity in the inner loop
    if (batch.numBodies > 0) {
      PxAllocatorCallback& allocator = *reinterpret_cast<PxAllocatorCallback*>(mAllocatorCallback);
      if (batch.numConstraints > 0 && batch.constraints) {
        batch.contactMap.build(batch.numBodies, batch.constraints, batch.numConstraints, allocator);
      }
      if (batch.numSpherical > 0 && batch.sphericalJoints) {
        batch.sphericalMap.build(batch.numBodies, batch.sphericalJoints, batch.numSpherical, allocator);
      }
      if (batch.numFixed > 0 && batch.fixedJoints) {
        batch.fixedMap.build(batch.numBodies, batch.fixedJoints, batch.numFixed, allocator);
      }
      if (batch.numRevolute > 0 && batch.revoluteJoints) {
        batch.revoluteMap.build(batch.numBodies, batch.revoluteJoints, batch.numRevolute, allocator);
      }
      if (batch.numPrismatic > 0 && batch.prismaticJoints) {
        batch.prismaticMap.build(batch.numBodies, batch.prismaticJoints, batch.numPrismatic, allocator);
      }
      if (batch.numD6 > 0 && batch.d6Joints) {
        batch.d6Map.build(batch.numBodies, batch.d6Joints, batch.numD6, allocator);
      }
      if (batch.numGear > 0 && batch.gearJoints) {
        batch.gearMap.build(batch.numBodies, batch.gearJoints, batch.numGear, allocator);
      }
    }

    // Constraint coloring for large islands
    const PxU32 largeIslandThreshold = mSolver.getConfig().largeIslandThreshold;
    if (mSolver.getConfig().enableParallelization && numConstraints >= largeIslandThreshold) {
      if (!mConstraintColoring.isInitialized()) {
        mConstraintColoring.initialize(numConstraints, mScratchAdapter);
      }

      PxU32 numColors = mConstraintColoring.colorConstraints(
          batch.constraints, numConstraints, batch.bodies, batch.numBodies);

      if (numColors > 0) {
        batch.colorBatches = static_cast<AvbdColorBatch *>
            (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                              mHeapFallbackAllocations,
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
              dst.constraintIndices = static_cast<PxU32 *>
                  (allocWithFallback(mScratchAllocator, mAllocatorCallback,
                                    mHeapFallbackAllocations,
                                    sizeof(PxU32) * src.numConstraints, "ConstraintIndices"));
              
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
    currSphericalIdx += numSpherical;
    currFixedIdx += numFixed;
    currRevoluteIdx += numRevolute;
    currPrismaticIdx += numPrismatic;
    currD6Idx += numD6;
    currGearIdx += numGear;

    // Spawn Solve Task
    AvbdSolveIslandTask *solveTask =
        mTaskFactory->createSolveTask(*this, mSolver, batch, dt, gravity);
    solveTask->setContinuation(coordTask);
    solveTask->removeReference();
    tasksSpawned++;
  }

  coordTask->removeReference();
  wbTask->removeReference();
  
  // NOTE: Do NOT free scratch allocations here!
  // The body arrays (avbdBodies, rigidBodies), constraint arrays, and other data
  // are used by async tasks (AvbdSolveIslandTask, AvbdWriteBackTask).
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

PxU32 AvbdDynamicsContext::prepareAvbdContacts(
    AvbdSolverBody *avbdBodies, PxU32 islandBodyCount,
    AvbdContactConstraint *constraints, PxU32 maxConstraints,
    PxU32 startContactIdx, PxU32 numContactsToProcess, PxU32 bodyOffset) {

  PxU32 constraintIndex = 0;
  const PxU32 endContactIdx = startContactIdx + numContactsToProcess;
  const PxU32 actualMax = PxMin(static_cast<PxU32>(mContactList.size()), endContactIdx);
  const PxU32 bodyEnd = bodyOffset + islandBodyCount;

  for (PxU32 i = startContactIdx; i < actualMax && constraintIndex < maxConstraints; ++i) {
    const PxsIndexedContactManager &icm = mContactList[i];
    PxsContactManager *cm = icm.contactManager;

    if (!cm)
      continue;

    PxU32 globalBody0Idx = PX_MAX_U32;
    PxU32 globalBody1Idx = PX_MAX_U32;

    if (icm.indexType0 == PxsIndexedInteraction::eBODY) {
      globalBody0Idx = static_cast<PxU32>(icm.solverBody0);
    }
    if (icm.indexType1 == PxsIndexedInteraction::eBODY) {
      globalBody1Idx = static_cast<PxU32>(icm.solverBody1);
    }

    if (globalBody0Idx == PX_MAX_U32 && globalBody1Idx == PX_MAX_U32)
      continue;

    PxU32 localBody0Idx = PX_MAX_U32;
    PxU32 localBody1Idx = PX_MAX_U32;

    if (globalBody0Idx != PX_MAX_U32 && globalBody0Idx >= bodyOffset && globalBody0Idx < bodyEnd) {
      localBody0Idx = globalBody0Idx - bodyOffset;
    }
    if (globalBody1Idx != PX_MAX_U32 && globalBody1Idx >= bodyOffset && globalBody1Idx < bodyEnd) {
      localBody1Idx = globalBody1Idx - bodyOffset;
    }

    if (localBody0Idx == PX_MAX_U32 && localBody1Idx == PX_MAX_U32)
      continue;

    const PxU32 npIndex = cm->getWorkUnit().mNpIndex;
    PxsContactManagerOutput &output = mOutputIterator.getContactManagerOutput(npIndex);
    if (output.nbContacts == 0)
      continue;
    if (!(output.statusFlag & PxsContactManagerStatusFlag::eHAS_TOUCH))
      continue;

    AvbdSolverBody *bodyA = (localBody0Idx < islandBodyCount) ? &avbdBodies[localBody0Idx] : nullptr;
    AvbdSolverBody *bodyB = (localBody1Idx < islandBodyCount) ? &avbdBodies[localBody1Idx] : nullptr;

    const PxU8 *contactData = output.contactPoints;
    const PxU8 *patchData = output.contactPatches;

    if (!contactData || !patchData)
      continue;

    for (PxU8 patchIdx = 0; patchIdx < output.nbPatches; ++patchIdx) {
      const PxContactPatch *patch = reinterpret_cast<const PxContactPatch *>(
          patchData + patchIdx * sizeof(PxContactPatch));

      const PxVec3 normal = patch->normal;
      const PxU32 startContact = patch->startContactIndex;
      const PxU16 numContactsInPatch = patch->nbContacts;

      for (PxU16 c = 0; c < numContactsInPatch && constraintIndex < maxConstraints; ++c) {
        const PxContact *contact = reinterpret_cast<const PxContact *>(
            contactData + (startContact + c) * sizeof(PxContact));

        AvbdContactConstraint &constraint = constraints[constraintIndex];

        constraint.header.bodyIndexA = localBody0Idx;
        constraint.header.bodyIndexB = localBody1Idx;
        constraint.header.type = static_cast<PxU16>(AvbdConstraintType::eCONTACT);
        constraint.header.flags = 0;
        constraint.header.compliance = 0.0f;
        constraint.header.damping = AvbdConstants::AVBD_CONSTRAINT_DAMPING;
        constraint.header.lambda = 0.0f;
        constraint.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_LOW;

        if (bodyA) {
          constraint.contactPointA = bodyA->rotation.rotateInv(contact->contact - bodyA->position);
        } else {
          constraint.contactPointA = contact->contact;
        }

        if (bodyB) {
          constraint.contactPointB = bodyB->rotation.rotateInv(contact->contact - bodyB->position);
        } else {
          constraint.contactPointB = contact->contact;
        }

        constraint.contactNormal = normal;
        constraint.penetrationDepth = contact->separation;
        constraint.restitution = 0.0f;
        constraint.friction = 0.5f;

        PxVec3 t0, t1;
        if (PxAbs(normal.x) < 0.9f) {
          t0 = normal.cross(PxVec3(1, 0, 0)).getNormalized();
        } else {
          t0 = normal.cross(PxVec3(0, 1, 0)).getNormalized();
        }
        t1 = normal.cross(t0);

        constraint.tangent0 = t0;
        constraint.tangent1 = t1;
        constraint.tangentLambda0 = 0.0f;
        constraint.tangentLambda1 = 0.0f;

        ++constraintIndex;
      }
    }
  }

  return constraintIndex;
}

void AvbdDynamicsContext::prepareAvbdConstraints(
    const IG::IslandSim &islandSim, AvbdSolverBody *avbdBodies,
    PxU32 islandBodyCount, PxU32 bodyOffset,
    AvbdSphericalJointConstraint *sphericalConstraints, PxU32 &numSpherical,
    PxU32 maxSpherical, AvbdFixedJointConstraint *fixedConstraints,
    PxU32 &numFixed, PxU32 maxFixed,
    AvbdRevoluteJointConstraint *revoluteConstraints, PxU32 &numRevolute,
    PxU32 maxRevolute, AvbdPrismaticJointConstraint *prismaticConstraints,
    PxU32 &numPrismatic, PxU32 maxPrismatic,
    AvbdD6JointConstraint *d6Constraints, PxU32 &numD6, PxU32 maxD6,
    AvbdGearJointConstraint *gearConstraints, PxU32 &numGear, PxU32 maxGear,
    PxU32 islandIndex, PxU32 *bodyRemapTable) {

  PX_UNUSED(avbdBodies);
  PX_UNUSED(islandBodyCount);

  const IG::Island &island = islandSim.getIsland(islandSim.getActiveIslands()[islandIndex]);
  IG::EdgeIndex edgeIndex = island.mFirstEdge[IG::Edge::eCONSTRAINT];

  numSpherical = 0;
  numFixed = 0;
  numRevolute = 0;
  numPrismatic = 0;
  numD6 = 0;
  numGear = 0;

  while (edgeIndex != IG_INVALID_EDGE) {
    const IG::Edge &edge = islandSim.getEdge(edgeIndex);
    Dy::Constraint *constraint = mIslandManager.getConstraint(edgeIndex);

    if (constraint && constraint->constantBlock && constraint->constantBlockSize > 0) {
      
      const PxNodeIndex nodeIndex0 = islandSim.mCpuData.getNodeIndex1(edgeIndex);
      const PxNodeIndex nodeIndex1 = islandSim.mCpuData.getNodeIndex2(edgeIndex);

      PxU32 localBody0 = PX_MAX_U32;
      PxU32 localBody1 = PX_MAX_U32;
      
      // Check if bodies are static using inverseMass (more reliable than nodeIndex.isStaticBody())
      // invMass == 0 means infinite mass (static body)
      // Note: NULL bodyCore means connected to world, which is handled by localBody staying PX_MAX_U32
      bool body0IsStatic = constraint->bodyCore0 && constraint->bodyCore0->inverseMass == 0.0f;
      bool body1IsStatic = constraint->bodyCore1 && constraint->bodyCore1->inverseMass == 0.0f;

      if (!body0IsStatic) {
        if (!nodeIndex0.isStaticBody()) {
          const PxU32 activeIdx = islandSim.getActiveNodeIndex(nodeIndex0);
          if (bodyRemapTable[activeIdx] != PX_MAX_U32) {
            localBody0 = bodyRemapTable[activeIdx] - bodyOffset;
          }
        }
      }

      if (!body1IsStatic) {
        if (!nodeIndex1.isStaticBody()) {
          const PxU32 activeIdx = islandSim.getActiveNodeIndex(nodeIndex1);
          if (bodyRemapTable[activeIdx] != PX_MAX_U32) {
            localBody1 = bodyRemapTable[activeIdx] - bodyOffset;
          }
        }
      }

      // Use the jointType field to detect joint type reliably
      PhysXJointType jointType = getJointTypeFromConstraint(constraint->jointType);
      
      if (jointType == eJOINT_GEAR) {
        // Process GearJoint
        if (numGear < maxGear && gearConstraints) {
          const PhysXGearJointData* gearData = 
              static_cast<const PhysXGearJointData*>(constraint->constantBlock);
          
          AvbdGearJointConstraint &c = gearConstraints[numGear++];
          c.initDefaults();
          c.header.bodyIndexA = localBody0;
          c.header.bodyIndexB = localBody1;
          c.gearRatio = gearData->gearRatio;
          c.geometricError = gearData->error;
          c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
          
          // Store gear axes in LOCAL body space
          // In PhysX, RevoluteJoint rotation axis is the X axis in joint local frame
          PxQuat frameA = gearData->c2b[0].q;
          PxQuat frameB = gearData->c2b[1].q;
          
          // Joint rotation axis is X in joint local frame (PhysX convention)
          PxVec3 jointAxis = PxVec3(1.0f, 0.0f, 0.0f);
          
          // Transform joint axis to body-local space
          // c2b is "constraint to body" transform, so we rotate the joint axis by it
          c.gearAxis0 = frameA.rotate(jointAxis);  // Body A local space
          c.gearAxis1 = frameB.rotate(jointAxis);  // Body B local space
        }
      }
      else if (jointType != eJOINT_UNKNOWN && constraint->constantBlockSize >= sizeof(PhysXJointData)) {
        // Process standard PhysX joints (Spherical, Revolute, Fixed, D6, etc.)
        const PhysXJointData* physXData = 
            static_cast<const PhysXJointData*>(constraint->constantBlock);
        
        // Validate joint data
        const float firstFloat = physXData->invMassScale.linear0;
        if (firstFloat >= 0.5f && firstFloat <= 2.0f && 
            physXData->c2b[0].p.isFinite() && physXData->c2b[1].p.isFinite()) {
        
        PxVec3 anchorA = physXData->c2b[0].p;
        PxVec3 anchorB = physXData->c2b[1].p;
        PxQuat frameA = physXData->c2b[0].q;
        PxQuat frameB = physXData->c2b[1].q;
        
        // Handle anchor transformation based on body type:
        // - NULL bodyCore: connected to world, c2b is already in world space
        // - Static body (inverseMass == 0): transform c2b from body local to world space
        // - Dynamic body: c2b stays in body local space for AVBD solver
        
        if (!constraint->bodyCore0) {
          // Connected to world - c2b[0] is already in world space, keep as is
          // localBody0 = PX_MAX_U32 indicates world anchor
        } else if (body0IsStatic) {
          // Static rigid body - transform anchor to world space
          PxTransform staticPose = constraint->bodyCore0->body2World;
          PxTransform jointFrame = staticPose * physXData->c2b[0];
          anchorA = jointFrame.p;
          frameA = jointFrame.q;
        }
        
        if (!constraint->bodyCore1) {
          // Connected to world - c2b[1] is already in world space, keep as is
          // localBody1 = PX_MAX_U32 indicates world anchor
        } else if (body1IsStatic) {
          // Static rigid body - transform anchor to world space
          PxTransform staticPose = constraint->bodyCore1->body2World;
          PxTransform jointFrame = staticPose * physXData->c2b[1];
          anchorB = jointFrame.p;
          frameB = jointFrame.q;
        }
        
        switch (jointType) {
          case eJOINT_FIXED:
          {
            if (numFixed < maxFixed) {
              AvbdFixedJointConstraint &c = fixedConstraints[numFixed++];
              c.initDefaults();
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              // Compute relative rotation between frames
              c.relativeRotation = frameA.getConjugate() * frameB;
              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
            }
            break;
          }
          
          case eJOINT_REVOLUTE:
          {
            if (numRevolute < maxRevolute) {
              const PhysXRevoluteJointData* revoluteData = 
                  static_cast<const PhysXRevoluteJointData*>(constraint->constantBlock);
              
              AvbdRevoluteJointConstraint &c = revoluteConstraints[numRevolute++];
              c.initDefaults();
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              
              // Revolute joint rotates around X-axis in joint frame
              c.axisA = frameA.getBasisVector0();  // X-axis of joint frame A
              c.axisB = frameB.getBasisVector0();  // X-axis of joint frame B
              
              // Reference axis for angle measurement (perpendicular to rotation axis)
              c.refAxisA = frameA.getBasisVector1();  // Y-axis
              c.refAxisB = frameB.getBasisVector1();
              
              // Check for angle limits
              // PxRevoluteJointFlag::eLIMIT_ENABLED = 1
              if (revoluteData->jointFlags & 0x0001) {
                c.hasAngleLimit = 1;
                c.angleLimitLower = revoluteData->limit.lower;
                c.angleLimitUpper = revoluteData->limit.upper;
              } else {
                c.hasAngleLimit = 0;
                c.angleLimitLower = -PxPi;
                c.angleLimitUpper = PxPi;
              }
              
              // Check for motor/drive
              // PxRevoluteJointFlag::eDRIVE_ENABLED = 2
              if (revoluteData->jointFlags & 0x0002) {
                c.motorEnabled = 1;
                c.motorTargetVelocity = revoluteData->driveVelocity;
                c.motorMaxForce = revoluteData->driveForceLimit;
              } else {
                c.motorEnabled = 0;
                c.motorTargetVelocity = 0.0f;
                c.motorMaxForce = 0.0f;
              }
              
              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
            }
            break;
          }
          
          case eJOINT_D6:
          {
            if (numD6 < maxD6) {
              const PhysXD6JointData* d6Data = 
                  static_cast<const PhysXD6JointData*>(constraint->constantBlock);
              
              AvbdD6JointConstraint &c = d6Constraints[numD6++];
              c.initDefaults();
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              c.localFrameA = frameA;
              c.localFrameB = frameB;
              
              // Set motion flags from D6 data
              // Each axis uses 2 bits: 0=LOCKED, 1=LIMITED, 2=FREE
              c.linearMotion = 0;
              c.angularMotion = 0;
              for (int i = 0; i < 3; ++i) {
                // Linear axes: X, Y, Z (indices 0, 1, 2)
                // Store motion type directly (0=LOCKED, 1=LIMITED, 2=FREE)
                c.linearMotion |= (d6Data->motion[i] << (i * 2));
                // Angular axes: TWIST, SWING1, SWING2 (indices 3, 4, 5)
                c.angularMotion |= (d6Data->motion[i + 3] << (i * 2));
              }
              
              // Set limits from D6 data
              c.linearLimitLower = PxVec3(d6Data->linearLimitX.lower, 
                                          d6Data->linearLimitY.lower, 
                                          d6Data->linearLimitZ.lower);
              c.linearLimitUpper = PxVec3(d6Data->linearLimitX.upper, 
                                          d6Data->linearLimitY.upper, 
                                          d6Data->linearLimitZ.upper);
              c.angularLimitLower = PxVec3(d6Data->twistLimit.lower,
                                           -d6Data->swingLimit.yAngle,
                                           -d6Data->swingLimit.zAngle);
              c.angularLimitUpper = PxVec3(d6Data->twistLimit.upper,
                                           d6Data->swingLimit.yAngle,
                                           d6Data->swingLimit.zAngle);
              
              // Set drive parameters if any drives are active
              c.driveFlags = 0;
              if (d6Data->driving != 0) {
                // Set stiffness and damping from drive parameters
                c.linearStiffness = PxVec3(d6Data->drive[0].stiffness,
                                           d6Data->drive[1].stiffness,
                                           d6Data->drive[2].stiffness);
                c.linearDamping = PxVec3(d6Data->drive[0].damping,
                                         d6Data->drive[1].damping,
                                         d6Data->drive[2].damping);
                // Angular drive data indices:
                // drive[3] = eSWING (deprecated), also used for eSWING1
                // drive[4] = eTWIST
                // drive[5] = eSLERP, also used for eSWING2
                // For AVBD: angularDamping.x = TWIST, .y = SWING1, .z = SWING2/SLERP
                c.angularStiffness = PxVec3(d6Data->drive[4].stiffness,  // TWIST
                                            d6Data->drive[3].stiffness,  // SWING1 (uses SWING slot)
                                            d6Data->drive[5].stiffness); // SWING2/SLERP
                c.angularDamping = PxVec3(d6Data->drive[4].damping,   // TWIST
                                          d6Data->drive[3].damping,   // SWING1 (uses SWING slot)
                                          d6Data->drive[5].damping);  // SWING2/SLERP
                c.driveFlags = d6Data->driving;
                
                // Set target drive velocities
                c.driveLinearVelocity = d6Data->driveLinearVelocity;
                c.driveAngularVelocity = d6Data->driveAngularVelocity;
                
                // Set max drive forces from drive[i].forceLimit
                c.driveLinearForce = PxVec3(d6Data->drive[0].forceLimit,
                                            d6Data->drive[1].forceLimit,
                                            d6Data->drive[2].forceLimit);
                c.driveAngularForce = PxVec3(d6Data->drive[4].forceLimit,  // TWIST
                                             d6Data->drive[3].forceLimit,  // SWING1
                                             d6Data->drive[5].forceLimit); // SWING2/SLERP
              }
              
              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
            }
            break;
          }
          
          case eJOINT_SPHERICAL:
          default:
          {
            // Default to spherical joint
            if (numSpherical < maxSpherical) {
              AvbdSphericalJointConstraint &c = sphericalConstraints[numSpherical++];
              c.initDefaults();
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
              
              // Check for cone limit if it's a spherical joint
              if (jointType == eJOINT_SPHERICAL && 
                  constraint->constantBlockSize >= sizeof(PhysXSphericalJointData)) {
                const PhysXSphericalJointData* sphericalData = 
                    static_cast<const PhysXSphericalJointData*>(constraint->constantBlock);
                
                // Check if limit is enabled (PxSphericalJointFlag::eLIMIT_ENABLED = 0x0002)
                if (sphericalData->jointFlags & 0x0002) {
                  // Use the smaller of the two angles as the cone limit
                  c.coneAngleLimit = PxMin(sphericalData->limit.yAngle, 
                                           sphericalData->limit.zAngle);
                  c.coneAxisA = PxVec3(1.0f, 0.0f, 0.0f);  // X-axis in local frame
                  c.hasConeLimit = 1;
                }
              }
            }
            break;
          }
        }  // end switch (jointType)
      }  // end else if (jointType != eJOINT_UNKNOWN && ...)
      else if (constraint->constantBlockSize >= sizeof(AvbdSnippetJointData)) {
        const AvbdSnippetJointData *data =
            static_cast<const AvbdSnippetJointData *>(constraint->constantBlock);

        if (data->type == AvbdSnippetJointData::eSPHERICAL && numSpherical < maxSpherical) {
          AvbdSphericalJointConstraint &c = sphericalConstraints[numSpherical++];
          c.initDefaults();
          c.header.bodyIndexA = localBody0;
          c.header.bodyIndexB = localBody1;
          c.anchorA = data->pivot0;
          c.anchorB = data->pivot1;
        } else if (data->type == AvbdSnippetJointData::eFIXED && numFixed < maxFixed) {
          AvbdFixedJointConstraint &c = fixedConstraints[numFixed++];
          c.initDefaults();
          c.header.bodyIndexA = localBody0;
          c.header.bodyIndexB = localBody1;
          c.anchorA = data->pivot0;
          c.anchorB = data->pivot1;
        } else if (data->type == AvbdSnippetJointData::eREVOLUTE && numRevolute < maxRevolute) {
          AvbdRevoluteJointConstraint &c = revoluteConstraints[numRevolute++];
          c.initDefaults();
          c.header.bodyIndexA = localBody0;
          c.header.bodyIndexB = localBody1;
          c.anchorA = data->pivot0;
          c.anchorB = data->pivot1;
          c.axisA = data->axis0;
          c.axisB = data->axis1;

          c.refAxisA = (PxAbs(c.axisA.x) < 0.9f) 
              ? c.axisA.cross(PxVec3(1, 0, 0)).getNormalized()
              : c.axisA.cross(PxVec3(0, 1, 0)).getNormalized();
          c.refAxisB = (PxAbs(c.axisB.x) < 0.9f)
              ? c.axisB.cross(PxVec3(1, 0, 0)).getNormalized()
              : c.axisB.cross(PxVec3(0, 1, 0)).getNormalized();

          c.angleLimitLower = data->limitLow;
          c.angleLimitUpper = data->limitHigh;
          c.hasAngleLimit = (c.angleLimitLower > -PxPi - AvbdConstants::AVBD_NUMERICAL_EPSILON ||
                             c.angleLimitUpper < PxPi + AvbdConstants::AVBD_NUMERICAL_EPSILON) ? 1 : 0;
        } else if (data->type == AvbdSnippetJointData::ePRISMATIC && numPrismatic < maxPrismatic) {
          AvbdPrismaticJointConstraint &c = prismaticConstraints[numPrismatic++];
          c.initDefaults();
          c.header.bodyIndexA = localBody0;
          c.header.bodyIndexB = localBody1;
          c.anchorA = data->pivot0;
          c.anchorB = data->pivot1;
          c.axisA = data->axis0;
          c.localFrameA = PxQuat(PxIdentity);
          c.localFrameB = PxQuat(PxIdentity);
          c.limitLower = data->limitLow;
          c.limitUpper = data->limitHigh;
          c.hasLimit = (c.limitLower > -PX_MAX_F32 / 2 || c.limitUpper < PX_MAX_F32 / 2) ? 1 : 0;
        } else if (data->type == AvbdSnippetJointData::eD6 && numD6 < maxD6) {
          AvbdD6JointConstraint &c = d6Constraints[numD6++];
          c.initDefaults();
          c.header.bodyIndexA = localBody0;
          c.header.bodyIndexB = localBody1;
          c.anchorA = data->pivot0;
          c.anchorB = data->pivot1;
          
          PxVec3 xAxis = data->axis0.getNormalized();
          PxVec3 yAxis = data->axis1.getNormalized();
          PxVec3 zAxis = xAxis.cross(yAxis).getNormalized();
          yAxis = zAxis.cross(xAxis).getNormalized();
          
          c.localFrameA = PxQuat(PxMat33(xAxis, yAxis, zAxis));
          c.localFrameB = PxQuat(PxIdentity);
          
          c.linearLimitLower = PxVec3(data->limitLow);
          c.linearLimitUpper = PxVec3(data->limitHigh);
          c.angularLimitLower = PxVec3(-PxPi);
          c.angularLimitUpper = PxVec3(PxPi);
          c.linearMotion = 0;
          c.angularMotion = 0;
          
          if (data->limitLow > -PX_MAX_F32 / 2 || data->limitHigh < PX_MAX_F32 / 2) {
            c.linearMotion = 0b010101;
          }
        }
      }
    }  // end else if (AvbdSnippetJointData)
    }  // end if (constraint && constraint->constantBlock && ...)
    edgeIndex = edge.mNextIslandEdge;
  }  // end while (edgeIndex != IG_INVALID_EDGE)
}  // end prepareAvbdConstraints

//=============================================================================
// Helper: Get joint axis in world frame
//=============================================================================
static PxVec3 getJointAxisWorld(const ArticulationJointCore* jointCore, PxArticulationAxis::Enum axis)
{
  // Determine the local axis based on the axis enum
  PxVec3 localAxis;
  switch (axis) {
    case PxArticulationAxis::eTWIST:
      localAxis = PxVec3(1.0f, 0.0f, 0.0f); // X-axis for twist
      break;
    case PxArticulationAxis::eSWING1:
      localAxis = PxVec3(0.0f, 1.0f, 0.0f); // Y-axis for swing1
      break;
    case PxArticulationAxis::eSWING2:
    default:
      localAxis = PxVec3(0.0f, 0.0f, 1.0f); // Z-axis for swing2
      break;
  }
  
  // Rotate by child pose orientation
  return jointCore->childPose.q.rotate(localAxis);
}

//=============================================================================
// Helper: Get first unlocked rotational axis for revolute joints
//=============================================================================
static PxArticulationAxis::Enum getFirstUnlockedRotationalAxis(const ArticulationJointCore& jointCore)
{
  // Check rotational axes first (TWIST, SWING1, SWING2)
  if (jointCore.motion[PxArticulationAxis::eTWIST] != PxArticulationMotion::eLOCKED)
    return PxArticulationAxis::eTWIST;
  if (jointCore.motion[PxArticulationAxis::eSWING1] != PxArticulationMotion::eLOCKED)
    return PxArticulationAxis::eSWING1;
  if (jointCore.motion[PxArticulationAxis::eSWING2] != PxArticulationMotion::eLOCKED)
    return PxArticulationAxis::eSWING2;
  
  return PxArticulationAxis::eTWIST; // Default
}

//=============================================================================
// Helper: Get first unlocked translational axis for prismatic joints
//=============================================================================
static PxArticulationAxis::Enum getFirstUnlockedTranslationalAxis(const ArticulationJointCore& jointCore)
{
  // Check translational axes (eX, eY, eZ)
  if (jointCore.motion[PxArticulationAxis::eX] != PxArticulationMotion::eLOCKED)
    return PxArticulationAxis::eX;
  if (jointCore.motion[PxArticulationAxis::eY] != PxArticulationMotion::eLOCKED)
    return PxArticulationAxis::eY;
  if (jointCore.motion[PxArticulationAxis::eZ] != PxArticulationMotion::eLOCKED)
    return PxArticulationAxis::eZ;
  
  return PxArticulationAxis::eX; // Default
}

//=============================================================================
// Articulation Internal Joints Preparation (Multi-Type)
//=============================================================================

static void prepareArticulationInternalJoints(
    FeatherstoneArticulation* articulation,
    PxU32 firstBodyIndex,
    AvbdSphericalJointConstraint* sphericalConstraints, PxU32& numSpherical, PxU32 maxSpherical,
    AvbdFixedJointConstraint* fixedConstraints, PxU32& numFixed, PxU32 maxFixed,
    AvbdRevoluteJointConstraint* revoluteConstraints, PxU32& numRevolute, PxU32 maxRevolute,
    AvbdPrismaticJointConstraint* prismaticConstraints, PxU32& numPrismatic, PxU32 maxPrismatic) {
    
  numSpherical = 0;
  numFixed = 0;
  numRevolute = 0;
  numPrismatic = 0;
  
  if (!articulation)
    return;
    
  ArticulationData& artData = articulation->getArticulationData();
  const PxU32 linkCount = artData.getLinkCount();
  
  if (linkCount <= 1)
    return;
  
  for (PxU32 linkIdx = 1; linkIdx < linkCount; ++linkIdx) {
    const ArticulationLink& link = artData.getLink(linkIdx);
    const PxU32 parentIdx = link.parent;
    
    if (parentIdx == DY_ARTICULATION_LINK_NONE)
      continue;
      
    ArticulationJointCore* jointCore = link.inboundJoint;
    if (!jointCore)
      continue;
    
    const PxsBodyCore* parentBodyCore = artData.getLink(parentIdx).bodyCore;
    const PxsBodyCore* childBodyCore = link.bodyCore;
    
    if (!parentBodyCore || !childBodyCore)
      continue;
    
    const PxVec3 anchorInParent = jointCore->parentPose.p;
    const PxVec3 anchorInChild = jointCore->childPose.p;
    const PxU32 bodyIndexA = firstBodyIndex + parentIdx;
    const PxU32 bodyIndexB = firstBodyIndex + linkIdx;
    
    // Dispatch based on joint type
    switch (static_cast<PxArticulationJointType::Enum>(jointCore->jointType)) {
      
      case PxArticulationJointType::eFIX:
      {
        if (numFixed < maxFixed && fixedConstraints) {
          AvbdFixedJointConstraint& c = fixedConstraints[numFixed];
          c.initDefaults();
          
          c.header.bodyIndexA = bodyIndexA;
          c.header.bodyIndexB = bodyIndexB;
          c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH * 2.0f;
          c.header.compliance = 0.0f;
          c.header.damping = AvbdConstants::AVBD_CONSTRAINT_DAMPING;
          
          c.anchorA = anchorInParent;
          c.anchorB = anchorInChild;
          
          // Store relative orientation (child relative to parent)
          c.relativeRotation = jointCore->parentPose.q.getConjugate() * jointCore->childPose.q;
          
          numFixed++;
        }
        break;
      }
      
      case PxArticulationJointType::eREVOLUTE:
      case PxArticulationJointType::eREVOLUTE_UNWRAPPED:
      {
        if (numRevolute < maxRevolute && revoluteConstraints) {
          AvbdRevoluteJointConstraint& c = revoluteConstraints[numRevolute];
          c.initDefaults();
          
          c.header.bodyIndexA = bodyIndexA;
          c.header.bodyIndexB = bodyIndexB;
          c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH * 2.0f;
          c.header.compliance = 0.0f;
          c.header.damping = AvbdConstants::AVBD_CONSTRAINT_DAMPING;
          
          c.anchorA = anchorInParent;
          c.anchorB = anchorInChild;
          
          // Get the rotation axis
          const PxArticulationAxis::Enum primaryAxis = getFirstUnlockedRotationalAxis(*jointCore);
          c.axisA = getJointAxisWorld(jointCore, primaryAxis);
          c.axisB = c.axisA; // Same axis for both bodies initially
          
          // Setup limits if enabled
          if (jointCore->motion[primaryAxis] == PxArticulationMotion::eLIMITED) {
            c.hasAngleLimit = 1;
            c.angleLimitLower = jointCore->limits[primaryAxis].low;
            c.angleLimitUpper = jointCore->limits[primaryAxis].high;
          } else {
            c.hasAngleLimit = 0;
            c.angleLimitLower = -PxPi;
            c.angleLimitUpper = PxPi;
          }
          
          numRevolute++;
        }
        break;
      }
      
      case PxArticulationJointType::ePRISMATIC:
      {
        if (numPrismatic < maxPrismatic && prismaticConstraints) {
          AvbdPrismaticJointConstraint& c = prismaticConstraints[numPrismatic];
          c.initDefaults();
          
          c.header.bodyIndexA = bodyIndexA;
          c.header.bodyIndexB = bodyIndexB;
          c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH * 2.0f;
          c.header.compliance = 0.0f;
          c.header.damping = AvbdConstants::AVBD_CONSTRAINT_DAMPING;
          
          c.anchorA = anchorInParent;
          c.anchorB = anchorInChild;
          
          // Get the translation axis
          const PxArticulationAxis::Enum primaryAxis = getFirstUnlockedTranslationalAxis(*jointCore);
          PxVec3 slideAxis;
          switch (primaryAxis) {
            case PxArticulationAxis::eX: slideAxis = PxVec3(1.0f, 0.0f, 0.0f); break;
            case PxArticulationAxis::eY: slideAxis = PxVec3(0.0f, 1.0f, 0.0f); break;
            case PxArticulationAxis::eZ: slideAxis = PxVec3(0.0f, 0.0f, 1.0f); break;
            default: slideAxis = PxVec3(1.0f, 0.0f, 0.0f); break;
          }
          c.axisA = jointCore->childPose.q.rotate(slideAxis);
          
          // Setup limits if enabled
          if (jointCore->motion[primaryAxis] == PxArticulationMotion::eLIMITED) {
            c.hasLimit = 1;
            c.limitLower = jointCore->limits[primaryAxis].low;
            c.limitUpper = jointCore->limits[primaryAxis].high;
          } else {
            c.hasLimit = 0;
            c.limitLower = -PX_MAX_F32;
            c.limitUpper = PX_MAX_F32;
          }
          
          // Store local frame orientations for rotation constraint
          c.localFrameA = jointCore->parentPose.q;
          c.localFrameB = jointCore->childPose.q;
          
          numPrismatic++;
        }
        break;
      }
      
      case PxArticulationJointType::eSPHERICAL:
      default:
      {
        // Default to spherical joint (ball-and-socket)
        if (numSpherical < maxSpherical && sphericalConstraints) {
          AvbdSphericalJointConstraint& c = sphericalConstraints[numSpherical];
          c.initDefaults();
          
          c.header.bodyIndexA = bodyIndexA;
          c.header.bodyIndexB = bodyIndexB;
          c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH * 2.0f;
          c.header.compliance = 0.0f;
          c.header.damping = AvbdConstants::AVBD_CONSTRAINT_DAMPING;
          
          c.anchorA = anchorInParent;
          c.anchorB = anchorInChild;
          
          // Setup cone limit if any rotational axis is limited
          c.hasConeLimit = false;
          for (PxU32 axis = PxArticulationAxis::eTWIST; axis <= PxArticulationAxis::eSWING2; ++axis) {
            if (jointCore->motion[axis] == PxArticulationMotion::eLIMITED) {
              c.hasConeLimit = true;
              c.coneAxisA = getJointAxisWorld(jointCore, static_cast<PxArticulationAxis::Enum>(axis));
              // Use the average of limits as cone angle
              c.coneAngleLimit = (PxAbs(jointCore->limits[axis].high) + 
                                  PxAbs(jointCore->limits[axis].low)) * 0.5f;
              break;
            }
          }
          
          numSpherical++;
        }
        break;
      }
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
  // No mutex needed since mergeResults() is called from a single thread context
  if (mHeapFallbackAllocations.size() > 0 && mAllocatorCallback) {
    PxAllocatorCallback* allocator = reinterpret_cast<PxAllocatorCallback*>(mAllocatorCallback);
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
