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

#include "DyAvbdTasks.h"
#include "DyAvbdBodyConversion.h"
#include "DyAvbdDynamics.h"
#include "DyAvbdKinematicShell.h"
#include "DyFeatherstoneArticulation.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DySleep.h"
#include "DyVArticulation.h"
#include "PxsRigidBody.h"
#include <cstdio>

// Debug logging macro
#if defined(AVBD_ENABLE_LOG)
#define AVBD_LOG(fmt, ...)                                                     \
  printf("[AVBD] " fmt "\n", ##__VA_ARGS__);                                   \
  fflush(stdout)
#else
#define AVBD_LOG(...)                                                          \
  do {                                                                         \
  } while (0)
#endif

namespace physx {
namespace Dy {

static void syncSingleDofArticulationJointState(
    ArticulationData &artData, PxU32 linkIndex) {
  if (linkIndex == 0 || linkIndex >= artData.getLinkCount())
    return;
  const ArticulationLink &link = artData.getLink(linkIndex);
  if (link.parent == DY_ARTICULATION_LINK_NONE ||
      link.parent >= artData.getLinkCount() || !link.inboundJoint ||
      !link.bodyCore)
    return;
  const ArticulationLink &parent = artData.getLink(link.parent);
  if (!parent.bodyCore)
    return;

  ArticulationJointCore &joint = *link.inboundJoint;
  ArticulationJointCoreData &jointData =
      artData.getJointData(linkIndex);
  if (jointData.nbDof != 1 || jointData.jointOffset == PX_MAX_U32)
    return;
  const PxArticulationAxis::Enum axis =
      static_cast<PxArticulationAxis::Enum>(joint.dofIds[0]);
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
    return;
  }

  const PxTransform parentFrame =
      parent.bodyCore->body2World * joint.parentPose;
  const PxTransform childFrame =
      link.bodyCore->body2World * joint.childPose;
  PxVec3 localAxis(0.0f);
  localAxis[component] = 1.0f;
  const PxVec3 worldAxis = parentFrame.q.rotate(localAxis);
  const bool linearAxis = axis >= PxArticulationAxis::eX;
  PxReal position = 0.0f;
  PxReal velocity = 0.0f;
  if (linearAxis) {
    position = (childFrame.p - parentFrame.p).dot(worldAxis);
    const PxVec3 parentPointVelocity =
        parent.bodyCore->linearVelocity +
        parent.bodyCore->angularVelocity.cross(
            parentFrame.p - parent.bodyCore->body2World.p);
    const PxVec3 childPointVelocity =
        link.bodyCore->linearVelocity +
        link.bodyCore->angularVelocity.cross(
            childFrame.p - link.bodyCore->body2World.p);
    velocity =
        (childPointVelocity - parentPointVelocity).dot(worldAxis);
  } else {
    PxQuat relative = parentFrame.q.getConjugate() * childFrame.q;
    if (relative.w < 0.0f)
      relative = -relative;
    position =
        2.0f * PxAtan2((&relative.x)[component], relative.w);
    velocity =
        (link.bodyCore->angularVelocity -
         parent.bodyCore->angularVelocity).dot(worldAxis);
  }
  if (!PxIsFinite(position) || !PxIsFinite(velocity))
    return;

  artData.getJointPositions()[jointData.jointOffset] = position;
  artData.getJointVelocities()[jointData.jointOffset] = velocity;
  artData.getPosIterJointVelocities()[jointData.jointOffset] = velocity;
  joint.jointPos[axis] = position;
  joint.jointVel[axis] = velocity;
}

void AvbdTask::release() {
  AVBD_LOG("Task release: %s", getName());
  // CRITICAL: Must call base class release() which calls
  // mCont->removeReference() to notify the continuation that this task is done.
  PxLightCpuTask::release();
  // After base class release, destroy ourselves
  mContext.destroyTask(this);
}

void AvbdSolveIslandTask::release() {
  const bool hasJointConstraints = (mBatch.numD6 > 0 || mBatch.numGear > 0);
  const bool usesJointPath =
      hasJointConstraints ||
      (mBatch.numSoftParticles > 0 && mBatch.numSoftBodies > 0 &&
       mBatch.softContacts && mBatch.numSoftContacts > 0);
  const PxU32 baseIterations =
      (mBatch.iterationOverride > 0)
          ? mBatch.iterationOverride
          : mSolver.getConfig().innerIterations;
  const PxU32 requestedIterations =
      (usesJointPath && hasJointConstraints)
        ? PxMax(baseIterations, PxU32(10))
          : baseIterations;
  mContext.recordIterationDiagnostics(requestedIterations, mStats,
                      hasJointConstraints, mBatch.d6Joints,
                      mBatch.numD6);

  // Write back lambda values to the cache for warm-starting next frame
  // This is thread-safe because each island writes to disjoint cache indices
  {
    AvbdContactConstraint *constraints = mBatch.constraints;
    PxU32 numConstraints = mBatch.numConstraints;

    // Function declared as friend in AvbdDynamicsContext class
    writeLambdaToCache(mContext, constraints, numConstraints, mBatch.numBodies);
    writeContactImpulseToOutput(constraints, numConstraints, mBatch.numBodies,
                                mDt);
    writeJointLambdaToCache(mContext, mBatch.d6Joints, mBatch.numD6);
  }

  // Shared by the task and forced-sequential paths so constraint force and
  // breakage semantics do not depend on scheduling mode. Scoped drive
  // families with a validated physical writeback contract compare their
  // force*dt or torque*dt impulse divided by dt; every other family retains
  // the legacy lambda comparison until it has an independent reaction gate.
  writeJointConstraintWriteback(mContext, mBatch.d6Joints, mBatch.numD6,
                                mDt);

  // Release constraint maps to prevent memory leak
  // Each frame builds new maps, so we must free them when task completes
  PxAllocatorCallback &allocator = mContext.getAllocator();
  mBatch.contactMap.release(allocator);
  mBatch.d6Map.release(allocator);
  mBatch.gearMap.release(allocator);

  // Call base class release
  AvbdTask::release();
}

void AvbdWriteBackTask::run() {
  AVBD_LOG("AvbdWriteBackTask::run() START - numBodies=%u", mNumBodies);

  for (PxU32 i = 0; i < mNumBodies; ++i) {
    if (mRigidBodies[i]) {
      // Regular rigid body - writeback to body core
      if (!mAvbdBodies[i].isStatic()) {
        PxsRigidBody &rigidBody = *mRigidBodies[i];
        PxsBodyCore &bodyCore = rigidBody.getCore();
        const PxTransform oldTransform = bodyCore.body2World;

        // Match the PGS/TGS writeback contract: CCD consumes the pre-step COM
        // pose, while sleeping consumes the motion that actually changed the
        // COM pose this step (which is not necessarily AVBD's final API
        // velocity after depenetration, damping, and velocity limiting).
        rigidBody.mLastTransform = oldTransform;
        writeBackAvbdSolverBody(mAvbdBodies[i], bodyCore);

        if (!mSleepingDisabled) {
          PxVec3 linearMotionVelocity(0.0f);
          PxVec3 angularMotionVelocity(0.0f);
          if (mDt > 0.0f) {
            calculateNewVelocity(bodyCore.body2World, oldTransform, mDt,
                                 linearMotionVelocity,
                                 angularMotionVelocity);
          }

          const Cm::SpatialVector motionVelocity(linearMotionVelocity,
                                                  angularMotionVelocity);
          const PxU32 staticTouchCount =
              mStaticTouchCounts ? mStaticTouchCounts[i] : 0;
          sleepCheck(&rigidBody, mDt, mEnableStabilization, motionVelocity,
                     staticTouchCount);
        }
      }
    } else if (mArticulationForBody && mLinkIndexForBody) {
      // Articulation link - writeback to articulation link body core
      FeatherstoneArticulation *articulation = mArticulationForBody[i];
      PxU32 linkIndex = mLinkIndexForBody[i];

      if (articulation && linkIndex != PX_MAX_U32) {
        ArticulationData &artData = articulation->getArticulationData();
        if (linkIndex < artData.getLinkCount()) {
          const ArticulationLink &link = artData.getLink(linkIndex);
          PxsBodyCore *bodyCore = link.bodyCore;

          if (bodyCore) {
            // Write back position and rotation
            bodyCore->body2World.p = mAvbdBodies[i].position;
            bodyCore->body2World.q = mAvbdBodies[i].rotation;

            // Write back velocities
            bodyCore->linearVelocity = mAvbdBodies[i].linearVelocity;
            bodyCore->angularVelocity = mAvbdBodies[i].angularVelocity;

            // PhysX 5.9 public link-velocity queries consume mMotionVelocities,
            // while articulation sleeping consumes mPosIterMotionVelocities.
            // Keep both buffers and PxsBodyCore synchronized with AVBD's final
            // velocity so driven/falling links cannot sleep on stale zeros.
            Cm::SpatialVectorF &motionVelocity =
                artData.getMotionVelocity(linkIndex);
            motionVelocity.top = mAvbdBodies[i].angularVelocity;
            motionVelocity.bottom = mAvbdBodies[i].linearVelocity;
            artData.getPosIterMotionVelocity(linkIndex) = motionVelocity;
          }
        }
      }
    }
  }

  // Joint coordinates are relative parent/child state. Rebuild them only
  // after every AVBD link pose and velocity has been written back; doing this
  // inside the loop can observe a new child against a parent that has not yet
  // reached its final pose when body ordering changes.
  FeatherstoneArticulation *lastArticulation = nullptr;
  for (PxU32 i = 0; i < mNumBodies; ++i) {
    if (!mArticulationForBody || !mLinkIndexForBody)
      break;
    FeatherstoneArticulation *articulation = mArticulationForBody[i];
    if (!articulation || articulation == lastArticulation)
      continue;
    lastArticulation = articulation;
    ArticulationData &artData = articulation->getArticulationData();
    for (PxU32 linkIndex = 1; linkIndex < artData.getLinkCount();
         ++linkIndex)
      syncSingleDofArticulationJointState(artData, linkIndex);
  }

  mContext.flushIterationDiagnosticsFrame();

  AVBD_LOG("AvbdWriteBackTask::run() END");
}

} // namespace Dy
} // namespace physx
