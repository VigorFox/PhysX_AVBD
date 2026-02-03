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

#include "DyAvbdArticulationAdapter.h"
#include "DyFeatherstoneArticulation.h"
#include "DyVArticulation.h"
#include "DyArticulationJointCore.h"
#include "DyAvbdBodyConversion.h"
#include "foundation/PxMemory.h"

namespace physx {
namespace Dy {

//=============================================================================
// Constructor / Destructor
//=============================================================================

AvbdArticulationAdapter::AvbdArticulationAdapter()
    : mArticulation(nullptr), mArticulationData(nullptr), mLinkData(nullptr),
      mNumLinks(0), mDofCount(0), mAllocator(nullptr), mInitialized(false) {}

AvbdArticulationAdapter::~AvbdArticulationAdapter() { release(); }

//=============================================================================
// Initialization
//=============================================================================

bool AvbdArticulationAdapter::initialize(FeatherstoneArticulation *articulation,
                                         PxAllocatorCallback &allocator) {
  if (!articulation) {
    return false;
  }

  mArticulation = articulation;
  mAllocator = &allocator;

  // Get articulation data
  mArticulationData = &mArticulation->getArticulationData();
  mNumLinks = mArticulationData->getLinkCount();
  mDofCount = mArticulationData->getDofs();

  // Allocate link data array
  mLinkData = reinterpret_cast<AvbdArticulationLinkData *>(
      mAllocator->allocate(sizeof(AvbdArticulationLinkData) * mNumLinks,
                          "AvbdArticulationLinkData", __FILE__, __LINE__));

  // Initialize link data
  for (PxU32 i = 0; i < mNumLinks; ++i) {
    mLinkData[i].avbdBodyIndex = i;
    mLinkData[i].parentLinkIndex = 0xFFFFFFFF;
    mLinkData[i].numChildLinks = 0;
    mLinkData[i].childLinkIndices = nullptr;
    mLinkData[i].jointPosition = 0.0f;
    mLinkData[i].jointVelocity = 0.0f;
    mLinkData[i].jointTarget = 0.0f;
    mLinkData[i].jointTargetVelocity = 0.0f;
    mLinkData[i].driveStiffness = 0.0f;
    mLinkData[i].driveDamping = 0.0f;
    mLinkData[i].maxForce = 0.0f;
    mLinkData[i].driveEnabled = false;
    mLinkData[i].limitEnabled = false;
    mLinkData[i].limitLower = 0.0f;
    mLinkData[i].limitUpper = 0.0f;

    // Extract joint parameters
    extractJointParameters(i, mLinkData[i]);
  }

  // Build parent-child relationships
  ArticulationLink *links = mArticulationData->getLinks();
  for (PxU32 i = 0; i < mNumLinks; ++i) {
    PxU32 parentIndex = links[i].parent;
    if (parentIndex < mNumLinks) {
      mLinkData[i].parentLinkIndex = parentIndex;
      mLinkData[parentIndex].numChildLinks++;
    }
  }

  // Allocate child link index arrays
  for (PxU32 i = 0; i < mNumLinks; ++i) {
    if (mLinkData[i].numChildLinks > 0) {
      mLinkData[i].childLinkIndices = reinterpret_cast<PxU32 *>(
          mAllocator->allocate(sizeof(PxU32) * mLinkData[i].numChildLinks,
                              "AvbdArticulationChildIndices", __FILE__, __LINE__));
    }
  }

  // Fill child link index arrays
  PxU32 *childCounters = reinterpret_cast<PxU32 *>(
      mAllocator->allocate(sizeof(PxU32) * mNumLinks, "TempCounters",
                          __FILE__, __LINE__));
  PxMemZero(childCounters, sizeof(PxU32) * mNumLinks);

  for (PxU32 i = 0; i < mNumLinks; ++i) {
    PxU32 parentIndex = mLinkData[i].parentLinkIndex;
    if (parentIndex < mNumLinks) {
      mLinkData[parentIndex].childLinkIndices[childCounters[parentIndex]++] = i;
    }
  }

  mAllocator->deallocate(childCounters);

  mInitialized = true;
  return true;
}

void AvbdArticulationAdapter::release() {
  if (mLinkData) {
    for (PxU32 i = 0; i < mNumLinks; ++i) {
      if (mLinkData[i].childLinkIndices) {
        mAllocator->deallocate(mLinkData[i].childLinkIndices);
      }
    }
    mAllocator->deallocate(mLinkData);
    mLinkData = nullptr;
  }

  mArticulation = nullptr;
  mArticulationData = nullptr;
  mNumLinks = 0;
  mDofCount = 0;
  mInitialized = false;
}

//=============================================================================
// State Synchronization
//=============================================================================

void AvbdArticulationAdapter::syncDriveTargetsToAvbd() {
  if (!mInitialized || !mArticulationData) {
    return;
  }

  // Get joint positions and velocities from Featherstone articulation
  const PxReal *jointPositions = mArticulationData->getJointPositions();
  const PxReal *jointVelocities = mArticulationData->getJointVelocities();

  for (PxU32 i = 0; i < mNumLinks; ++i) {
    // Update joint state
    if (jointPositions) {
      mLinkData[i].jointPosition = jointPositions[i];
    }
    if (jointVelocities) {
      mLinkData[i].jointVelocity = jointVelocities[i];
    }

    // Update drive targets (if available)
    // Note: This is a placeholder - actual implementation depends on
    // how drive targets are stored in the articulation
    mLinkData[i].jointTarget = mLinkData[i].jointPosition;
    mLinkData[i].jointTargetVelocity = mLinkData[i].jointVelocity;
  }
}

void AvbdArticulationAdapter::syncStateToArticulation() {
  if (!mInitialized || !mArticulationData) {
    return;
  }

  // Get joint positions and velocities arrays from Featherstone articulation
  PxReal *jointPositions = mArticulationData->getJointPositions();
  PxReal *jointVelocities = mArticulationData->getJointVelocities();

  // Copy solved state from AVBD to Featherstone
  for (PxU32 i = 0; i < mNumLinks; ++i) {
    if (jointPositions) {
      jointPositions[i] = mLinkData[i].jointPosition;
    }
    if (jointVelocities) {
      jointVelocities[i] = mLinkData[i].jointVelocity;
    }
  }
}

//=============================================================================
// Forward Dynamics (AVBD)
//=============================================================================

void AvbdArticulationAdapter::solveForwardDynamics(
    PxReal dt, const PxVec3 &gravity, AvbdSolver &solver,
    AvbdSolverBody *bodies, PxU32 numBodies,
    AvbdSphericalJointConstraint *joints, PxU32 numJoints) {
  if (!mInitialized) {
    return;
  }

  // Sync drive targets before solving
  syncDriveTargetsToAvbd();

  // Apply joint drives
  applyJointDrives(bodies, numBodies, dt);

  // Apply joint limits
  applyJointLimits(bodies, numBodies);

  // Solve using AVBD solver
  solver.solveWithJoints(dt, bodies, numBodies, nullptr, 0, joints, numJoints,
                        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, gravity);

  // Sync state back to articulation
  syncStateToArticulation();
}

//=============================================================================
// Inverse Dynamics (Featherstone)
//=============================================================================

bool AvbdArticulationAdapter::computeMassMatrix(PxReal *massMatrix,
                                                 const PxReal *jointPositions,
                                                 const PxReal *jointVelocities) {
  if (!mInitialized || !mArticulation) {
    return false;
  }
  PX_UNUSED(massMatrix);
  PX_UNUSED(jointPositions);
  PX_UNUSED(jointVelocities);
  // Not exposed by FeatherstoneArticulation in this branch.
  return false;
}

bool AvbdArticulationAdapter::computeJointForces(
    const PxReal *jointAccelerations, PxReal *jointForces) {
  if (!mInitialized || !mArticulation) {
    return false;
  }
  PX_UNUSED(jointAccelerations);
  PX_UNUSED(jointForces);
  // Not exposed by FeatherstoneArticulation in this branch.
  return false;
}

bool AvbdArticulationAdapter::computeGeneralizedForces(
    const PxReal *jointPositions, const PxReal *jointVelocities,
    PxReal *jointForces) {
  if (!mInitialized || !mArticulation) {
    return false;
  }
  PX_UNUSED(jointPositions);
  PX_UNUSED(jointVelocities);
  PX_UNUSED(jointForces);
  // Not exposed by FeatherstoneArticulation in this branch.
  return false;
}

//=============================================================================
// Internal Methods
//=============================================================================

void AvbdArticulationAdapter::buildAvbdBodies(AvbdSolverBody *bodies,
                                              PxU32 numBodies) {
  if (!mInitialized || !mArticulationData) {
    return;
  }

  ArticulationLink *links = mArticulationData->getLinks();

  for (PxU32 i = 0; i < mNumLinks && i < numBodies; ++i) {
    // Copy link state to AVBD body using the body core
    if (links[i].bodyCore) {
      copyToAvbdSolverBody(*links[i].bodyCore, bodies[i], i);
    }
  }
}

void AvbdArticulationAdapter::buildAvbdJoints(
    AvbdSphericalJointConstraint *joints, PxU32 numJoints) {
  if (!mInitialized) {
    return;
  }

  for (PxU32 i = 1; i < mNumLinks && i < numJoints; ++i) {
    joints[i - 1].initDefaults();
    // Create spherical joint constraint for each link (except root)
    joints[i - 1].header.bodyIndexA = mLinkData[i].parentLinkIndex;
    joints[i - 1].header.bodyIndexB = i;
    joints[i - 1].header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_LOW;
    joints[i - 1].header.compliance = 0.0f;
    joints[i - 1].header.damping = AvbdConstants::AVBD_CONSTRAINT_DAMPING;
    joints[i - 1].anchorA = mLinkData[i].localAnchor;
    joints[i - 1].anchorB = PxVec3(0); // Child anchor at origin
    joints[i - 1].coneAxisA = mLinkData[i].localAxis;
    joints[i - 1].coneAngleLimit = 0.0f;
    joints[i - 1].hasConeLimit = false;
    joints[i - 1].coneLambda = 0.0f;
  }
}

PxU32 AvbdArticulationAdapter::getJointType(
    const ArticulationJointCore &jointCore) {
  // Map PhysX articulation joint type to AVBD constraint type
  switch (static_cast<PxArticulationJointType::Enum>(jointCore.jointType)) {
  case PxArticulationJointType::eSPHERICAL:
    return 0; // Spherical (ball-and-socket)
  case PxArticulationJointType::eFIX:
    return 1; // Fixed (welded)
  case PxArticulationJointType::eREVOLUTE:
  case PxArticulationJointType::eREVOLUTE_UNWRAPPED:
    return 2; // Revolute (hinge)
  case PxArticulationJointType::ePRISMATIC:
    return 3; // Prismatic (slider)
  default:
    return 0; // Default to spherical
  }
}

static PX_FORCE_INLINE PxArticulationAxis::Enum getFirstUnlockedAxis(
    const ArticulationJointCore& jointCore)
{
  for (PxU32 i = 0; i < PxArticulationAxis::eCOUNT; ++i)
  {
    if (jointCore.motion[i] != PxArticulationMotion::eLOCKED)
      return PxArticulationAxis::Enum(i);
  }
  return PxArticulationAxis::eTWIST;
}

static PX_FORCE_INLINE PxVec3 axisToVector(PxArticulationAxis::Enum axis)
{
  switch (axis)
  {
  case PxArticulationAxis::eTWIST:
    return PxVec3(1.0f, 0.0f, 0.0f);
  case PxArticulationAxis::eSWING1:
    return PxVec3(0.0f, 1.0f, 0.0f);
  case PxArticulationAxis::eSWING2:
    return PxVec3(0.0f, 0.0f, 1.0f);
  default:
    return PxVec3(0.0f, 0.0f, 1.0f);
  }
}

void AvbdArticulationAdapter::extractJointParameters(
    PxU32 linkIndex, AvbdArticulationLinkData &linkData) {
  if (!mArticulationData) {
    return;
  }

  ArticulationLink *links = mArticulationData->getLinks();
  ArticulationJointCore *jointCore = links[linkIndex].inboundJoint;

  if (!jointCore) {
    return;
  }

  const PxArticulationAxis::Enum primaryAxis = getFirstUnlockedAxis(*jointCore);

  // Extract joint anchor and axis
  linkData.localAnchor = jointCore->childPose.p;
  linkData.localAxis = axisToVector(primaryAxis);

  // Extract drive parameters
  const PxArticulationDrive& drive = jointCore->drives[primaryAxis];
  linkData.driveStiffness = drive.stiffness;
  linkData.driveDamping = drive.damping;
  linkData.maxForce = drive.maxForce;
  linkData.driveEnabled = (drive.driveType != PxArticulationDriveType::eNONE);
  linkData.jointTarget = jointCore->targetP[primaryAxis];
  linkData.jointTargetVelocity = jointCore->targetV[primaryAxis];

  // Extract limit parameters
  linkData.limitEnabled = (jointCore->motion[primaryAxis] == PxArticulationMotion::eLIMITED);
  linkData.limitLower = jointCore->limits[primaryAxis].low;
  linkData.limitUpper = jointCore->limits[primaryAxis].high;
}

void AvbdArticulationAdapter::applyJointDrives(AvbdSolverBody *bodies,
                                               PxU32 numBodies, PxReal dt) {
  if (!mInitialized) {
    return;
  }

  // Apply PD control forces for each joint
  for (PxU32 i = 1; i < mNumLinks && i < numBodies; ++i) {
    if (!mLinkData[i].driveEnabled) {
      continue;
    }

    // PD control: F = Kp * (target - current) + Kd * (targetVel - currentVel)
    PxReal positionError = mLinkData[i].jointTarget - mLinkData[i].jointPosition;
    PxReal velocityError =
        mLinkData[i].jointTargetVelocity - mLinkData[i].jointVelocity;

    PxReal driveForce = mLinkData[i].driveStiffness * positionError +
                        mLinkData[i].driveDamping * velocityError;

    // Clamp to max force
    driveForce = PxClamp(driveForce, -mLinkData[i].maxForce,
                         mLinkData[i].maxForce);

    // Apply force to bodies (simplified - actual implementation needs proper
    // torque application)
    PxVec3 force = mLinkData[i].localAxis * driveForce;
    bodies[i].linearVelocity += force * bodies[i].invMass * dt;
  }
}

void AvbdArticulationAdapter::applyJointLimits(AvbdSolverBody *bodies,
                                               PxU32 numBodies) {
  if (!mInitialized) {
    return;
  }

  PX_UNUSED(bodies);

  // Apply position limits for each joint
  for (PxU32 i = 1; i < mNumLinks && i < numBodies; ++i) {
    if (!mLinkData[i].limitEnabled) {
      continue;
    }

    // Clamp joint position to limits
    if (mLinkData[i].jointPosition < mLinkData[i].limitLower) {
      mLinkData[i].jointPosition = mLinkData[i].limitLower;
      mLinkData[i].jointVelocity = 0.0f; // Stop at limit
    } else if (mLinkData[i].jointPosition > mLinkData[i].limitUpper) {
      mLinkData[i].jointPosition = mLinkData[i].limitUpper;
      mLinkData[i].jointVelocity = 0.0f; // Stop at limit
    }
  }
}

} // namespace Dy

} // namespace physx
