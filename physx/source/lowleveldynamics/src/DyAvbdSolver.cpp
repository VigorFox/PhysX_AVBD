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

#include "DyAvbdSolver.h"
#include "foundation/PxAssert.h"
#include "foundation/PxArray.h"
#include "DyAvbdJointSolver.h"
#include "common/PxProfileZone.h"

#include <algorithm>

// External frame counter from DyAvbdDynamics.cpp
extern physx::PxU64 getAvbdMotorFrameCounter();

namespace physx {
namespace Dy {

namespace {
struct KahanSum {
  physx::PxReal sum{0.0f};
  physx::PxReal c{0.0f};

  void add(physx::PxReal value) {
    physx::PxReal y = value - c;
    physx::PxReal t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};
} // namespace

//=============================================================================
// Main Solver Entry Point
//=============================================================================

void AvbdSolver::solve(physx::PxReal dt, AvbdSolverBody *bodies,
                       physx::PxU32 numBodies, AvbdContactConstraint *contacts,
                       physx::PxU32 numContacts, const physx::PxVec3 &gravity,
                       AvbdColorBatch *colorBatches, physx::PxU32 numColors) {
  PX_PROFILE_ZONE("AVBD.solve", 0);
  
  if (!mInitialized || numBodies == 0) {
    return;
  }

  mStats.reset();
  mStats.numBodies = numBodies;
  mStats.numContacts = numContacts;

  physx::PxReal invDt = 1.0f / dt;

  // Stage 1: Prediction
  {
    PX_PROFILE_ZONE("AVBD.prediction", 0);
    computePrediction(bodies, numBodies, dt, gravity);
  }

  // Stage 2: Graph Coloring (skip if pre-computed coloring is provided)
  if (mConfig.enableParallelization && numColors == 0) {
    PX_PROFILE_ZONE("AVBD.graphColoring", 0);
    computeGraphColoring(bodies, numBodies, contacts, numContacts);
  }

  // Initialize positions to predicted values, save current as previous
  {
    PX_PROFILE_ZONE("AVBD.initPositions", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      // Save current state before prediction for velocity calculation
      bodies[i].prevPosition = bodies[i].position;
      bodies[i].prevRotation = bodies[i].rotation;
      // Start from predicted position
      bodies[i].position = bodies[i].predictedPosition;
      bodies[i].rotation = bodies[i].predictedRotation;
    }
  }

  // Outer loop: Augmented Lagrangian iterations
  if (mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eSORT_CONSTRAINTS) &&
      numContacts > 1) {
    PX_PROFILE_ZONE("AVBD.sortConstraints", 0);
    std::sort(contacts, contacts + numContacts,
              [](const AvbdContactConstraint &a,
                 const AvbdContactConstraint &b) {
                if (a.header.bodyIndexA != b.header.bodyIndexA)
                  return a.header.bodyIndexA < b.header.bodyIndexA;
                if (a.header.bodyIndexB != b.header.bodyIndexB)
                  return a.header.bodyIndexB < b.header.bodyIndexB;
                return a.header.type < b.header.type;
              });
  }

  // Build constraint-to-body mapping for O(1) lookup (eliminates O(N²))
  if (numContacts > 0 && numBodies > 0) {
    PX_PROFILE_ZONE("AVBD.buildConstraintMap", 0);
    buildConstraintMapping(contacts, numContacts, numBodies);
  }

  {
    PX_PROFILE_ZONE("AVBD.solveIterations", 0);
    for (physx::PxU32 outerIter = 0; outerIter < mConfig.outerIterations;
         ++outerIter) {

      // Inner loop: Block descent iterations
      for (physx::PxU32 innerIter = 0; innerIter < mConfig.innerIterations;
           ++innerIter) {
        PX_PROFILE_ZONE("AVBD.blockDescent", 0);
        // Pass pre-computed coloring if available
        blockDescentIteration(bodies, numBodies, contacts, numContacts, dt,
                              colorBatches, numColors);
        mStats.totalIterations++;
      }

      // Update Lagrangian multipliers
      {
        PX_PROFILE_ZONE("AVBD.updateLambda", 0);
        updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts);
      }
    }
  }

  // Stage 5: Update velocities from position/rotation change
  {
    PX_PROFILE_ZONE("AVBD.updateVelocities", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f) {
        // Linear velocity: v = (x - x_old) / dt
        bodies[i].linearVelocity = (bodies[i].position - bodies[i].prevPosition) * invDt;

        // Angular velocity from quaternion difference:  = 2 * (q * q_old^-1).xyz / dt
        physx::PxQuat deltaQ = bodies[i].rotation * bodies[i].prevRotation.getConjugate();
        if (deltaQ.w < 0.0f) {
          deltaQ = -deltaQ; // Ensure shortest path
        }
        bodies[i].angularVelocity = physx::PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * (2.0f * invDt);
        
        // Apply angular damping
        bodies[i].angularVelocity *= mConfig.angularDamping;
      }
    }
  }

  // Note: No Stage 6 rotation update needed - rotation was already updated during constraint solving
}

//=============================================================================
// Graph Coloring
//=============================================================================

void AvbdSolver::computeGraphColoring(AvbdSolverBody *bodies,
                                      physx::PxU32 numBodies,
                                      AvbdContactConstraint *contacts,
                                      physx::PxU32 numContacts) {
  PX_ASSERT(mAllocator != nullptr);

  // Build adjacency from contacts
  // Two bodies are adjacent if they share a constraint

  // Reset colors
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].colorGroup = 0xFFFFFFFF; // Uncolored
  }

  // Simple greedy coloring
  physx::PxU32 numColors = 0;

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].isStatic()) {
      bodies[i].colorGroup = 0; // Static bodies go to color 0
      continue;
    }

    // Find colors used by neighbors
    physx::PxU32 usedColors = 0;

    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
      physx::PxU32 bodyB = contacts[c].header.bodyIndexB;

      if (bodyA == i && bodyB < numBodies && bodies[bodyB].colorGroup < 32) {
        usedColors |= (1u << bodies[bodyB].colorGroup);
      }
      if (bodyB == i && bodyA < numBodies && bodies[bodyA].colorGroup < 32) {
        usedColors |= (1u << bodies[bodyA].colorGroup);
      }
    }

    // Find first available color (skip color 0 for static bodies)
    physx::PxU32 color = 1;
    while ((usedColors & (1u << color)) != 0 && color < 32) {
      color++;
    }

    bodies[i].colorGroup = color;
    if (color + 1 > numColors) {
      numColors = color + 1;
    }
  }

  mStats.numColorGroups = numColors;
}

//=============================================================================
// Body-Based Graph Coloring for Block Coordinate Descent
//=============================================================================

void AvbdSolver::computeBodyColoring(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                     AvbdContactConstraint *contacts, physx::PxU32 numContacts) {
  PX_ASSERT(mAllocator != nullptr);

  // Initialize body coloring if not already done
  if (!mBodyColoring.isInitialized()) {
    mBodyColoring.initialize(numBodies, *mAllocator);
  }

  // Perform body-based coloring
  physx::PxU32 numColors = mBodyColoring.colorBodies(contacts, numContacts, bodies, numBodies);
  
  mStats.numColorGroups = numColors;
}

//=============================================================================
// Augmented Lagrangian Multiplier Update
//=============================================================================

void AvbdSolver::updateLagrangianMultipliers(AvbdSolverBody *bodies,
                                             physx::PxU32 numBodies,
                                             AvbdContactConstraint *contacts,
                                             physx::PxU32 numContacts) {
  physx::PxReal totalError = 0.0f;
  KahanSum totalErrorKahan;
  const bool useKahan =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eUSE_KAHAN_SUMMATION);
  physx::PxU32 numActive = 0;

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyB = contacts[c].header.bodyIndexB;

    // Compute current violation
    physx::PxReal violation = 0.0f;

    if (bodyA < numBodies && bodyB < numBodies) {
      violation =
          computeContactViolation(contacts[c], bodies[bodyA], bodies[bodyB]);
    } else if (bodyA < numBodies) {
      // B is world/static
      AvbdSolverBody staticBody;
      staticBody.position = physx::PxVec3(0.0f);
      staticBody.rotation = physx::PxQuat(physx::PxIdentity);
      violation =
          computeContactViolation(contacts[c], bodies[bodyA], staticBody);
    }

    // Update multiplier: lambda <- lambda + rho * C
    // For inequality (contact): lambda <- max(0, lambda + rho * C)
    physx::PxReal newLambda =
        contacts[c].header.lambda + contacts[c].header.rho * violation;
    contacts[c].header.lambda = physx::PxMax(0.0f, newLambda);

    // Track convergence
    if (violation < 0.0f) { // Penetrating
      physx::PxReal err = violation * violation;
      if (useKahan) {
        totalErrorKahan.add(err);
      } else {
        totalError += err;
      }
      numActive++;
    }
  }

  if (useKahan) {
    totalError = totalErrorKahan.sum;
  }

  mStats.constraintError =
      (numActive > 0) ? sqrtf(totalError / (physx::PxReal)numActive) : 0.0f;
  mStats.activeConstraints = numActive;
}

//=============================================================================
// Local 6x6 System Solver
//=============================================================================

void AvbdSolver::solveLocalSystem(AvbdSolverBody &body,
                                   AvbdContactConstraint *contacts,
                                   physx::PxU32 numContacts,
                                   physx::PxReal /*dt*/,
                                   physx::PxReal invDt2) {
  
  // Skip static bodies
  if (body.invMass <= 0.0f) {
    return;
  }
  
  // 1. Build Hessian matrix H
  AvbdBlock6x6 H;
  buildHessianMatrix(body, contacts, numContacts, invDt2, H);
  
  // 2. Build gradient vector g
  AvbdVec6 g;
  buildGradientVector(body, contacts, numContacts, invDt2, g);
  
  // 3. Solve H * x = -g using LDLT decomposition
  AvbdLDLT ldlt;
  if (ldlt.decompose(H)) {
    AvbdVec6 negG(-g.linear, -g.angular);
    AvbdVec6 deltaX = ldlt.solve(negG);
    
    // 4. Update position
    body.position += deltaX.linear;
    
    // 5. Update rotation using exponential map
    physx::PxReal angle = deltaX.angular.magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      physx::PxVec3 axis = deltaX.angular / angle;
      physx::PxReal sinHalf = physx::PxSin(angle * 0.5f);
      physx::PxReal cosHalf = physx::PxCos(angle * 0.5f);
      physx::PxQuat deltaQ(axis.x * sinHalf, axis.y * sinHalf,
                             axis.z * sinHalf, cosHalf);
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
  } else {
    // Fallback to gradient descent if LDLT fails
    physx::PxReal stepSize = mConfig.damping;
    physx::PxVec3 deltaPos = -g.linear * body.invMass * stepSize;
    
    // Clamp step size for stability
    physx::PxReal maxStep = AvbdConstants::AVBD_MAX_POSITION_STEP;
    physx::PxReal stepMag = deltaPos.magnitude();
    if (stepMag > maxStep) {
      deltaPos *= maxStep / stepMag;
    }
    
    body.position += deltaPos;
  }
}

//=============================================================================
// Helper Methods for 6x6 System Solver
//=============================================================================

physx::PxU32 AvbdSolver::collectBodyConstraints(
    physx::PxU32 bodyIndex,
    AvbdContactConstraint *contacts,
    physx::PxU32 numContacts,
    physx::PxU32 *constraintIndices) {
    
    physx::PxU32 count = 0;
    for (physx::PxU32 c = 0; c < numContacts && count < 64; ++c) {
        physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
        physx::PxU32 bodyB = contacts[c].header.bodyIndexB;
        
        if (bodyA == bodyIndex || bodyB == bodyIndex) {
            constraintIndices[count++] = c;
        }
    }
    return count;
}

void AvbdSolver::buildHessianMatrix(
    const AvbdSolverBody &body,
    AvbdContactConstraint *contacts,
    physx::PxU32 numContacts,
    physx::PxReal invDt2,
    AvbdBlock6x6 &H) {
    
    // Initialize with mass/inertia contribution: M/h^2
    H.initializeDiagonal(body.invMass, body.invInertiaWorld, invDt2);
    
    // Add constraint contributions
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
        physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
        physx::PxU32 bodyB = contacts[c].header.bodyIndexB;
        
        // Skip if this body is not involved
        if (bodyA != body.nodeIndex && bodyB != body.nodeIndex) {
            continue;
        }
        
        // Compute constraint violation
        physx::PxReal violation = contacts[c].penetrationDepth;
        
        // Only process active contacts
        if (violation >= 0.0f && contacts[c].header.lambda <= 0.0f) {
            continue;
        }
        
        // Compute constraint gradient (Jacobian)
        physx::PxVec3 gradPos, gradRot;
        if (bodyA == body.nodeIndex) {
            gradPos = contacts[c].contactNormal;
            physx::PxVec3 rA = body.rotation.rotate(contacts[c].contactPointA);
            gradRot = rA.cross(contacts[c].contactNormal);
        } else {
            gradPos = -contacts[c].contactNormal;
            physx::PxVec3 rB = body.rotation.rotate(contacts[c].contactPointB);
            gradRot = -rB.cross(contacts[c].contactNormal);
        }
        
        // Add constraint contribution: H += rho * J^T * J
        physx::PxReal rho = contacts[c].header.rho;
        H.addConstraintContribution(gradPos, gradRot, rho);
    }
}

void AvbdSolver::buildGradientVector(
    const AvbdSolverBody &body,
    AvbdContactConstraint *contacts,
    physx::PxU32 numContacts,
    physx::PxReal invDt2,
    AvbdVec6 &g) {
    
    // Initialize with inertia term: M/h^2 * (x - x_tilde)
    physx::PxReal massContrib = (body.invMass > 0.0f) ?
        (1.0f / body.invMass) * invDt2 : 0.0f;
    g.linear = (body.position - body.predictedPosition) * massContrib;
    
    // For rotation, use simplified inertia contribution
    g.angular = physx::PxVec3(0.0f);
    
    // Add constraint gradient contributions
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
        physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
        physx::PxU32 bodyB = contacts[c].header.bodyIndexB;
        
        // Skip if this body is not involved
        if (bodyA != body.nodeIndex && bodyB != body.nodeIndex) {
            continue;
        }
        
        // Compute constraint violation
        physx::PxReal violation = contacts[c].penetrationDepth;
        
        // Only process active contacts
        if (violation >= 0.0f && contacts[c].header.lambda <= 0.0f) {
            continue;
        }
        
        // Compute constraint force: F = rho * C + lambda
        physx::PxReal force = contacts[c].header.rho * violation + contacts[c].header.lambda;
        
        // Compute constraint gradient (Jacobian)
        physx::PxVec3 gradPos, gradRot;
        if (bodyA == body.nodeIndex) {
            gradPos = contacts[c].contactNormal;
            physx::PxVec3 rA = body.rotation.rotate(contacts[c].contactPointA);
            gradRot = rA.cross(contacts[c].contactNormal);
        } else {
            gradPos = -contacts[c].contactNormal;
            physx::PxVec3 rB = body.rotation.rotate(contacts[c].contactPointB);
            gradRot = -rB.cross(contacts[c].contactNormal);
        }
        
        // Add contribution: g += J^T * F
        g.linear += gradPos * force;
        g.angular += gradRot * force;
    }
}

//=============================================================================
// Block Descent Iteration - Position-Based Constraint Solving
//=============================================================================

void AvbdSolver::blockDescentIteration(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, AvbdColorBatch *colorBatches, physx::PxU32 numColors) {
  
  PX_UNUSED(colorBatches);
  PX_UNUSED(numColors);
  
  // True Block Coordinate Descent: iterate over bodies, not constraints
  // For each body, solve a local optimization problem considering all
  // constraints that affect this body.
  
  const bool useDeterministicOrder =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eSORT_BODIES);

  physx::PxArray<physx::PxU32> bodyOrder;
  if (useDeterministicOrder) {
    bodyOrder.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      bodyOrder[i] = i;
    }
    std::sort(bodyOrder.begin(), bodyOrder.end(),
              [bodies](physx::PxU32 a, physx::PxU32 b) {
                if (bodies[a].nodeIndex != bodies[b].nodeIndex)
                  return bodies[a].nodeIndex < bodies[b].nodeIndex;
                return a < b;
              });
  }

  const physx::PxReal invDt2 = mConfig.enableLocal6x6Solve ? (1.0f / (dt * dt)) : 0.0f;
  const physx::PxU32 *orderPtr = useDeterministicOrder ? bodyOrder.begin() : nullptr;

  if (mConfig.enableLocal6x6Solve) {
    for (physx::PxU32 idx = 0; idx < numBodies; ++idx) {
      const physx::PxU32 i = orderPtr ? orderPtr[idx] : idx;
      if (bodies[i].invMass <= 0.0f) {
        continue;
      }

      // Solve local 6x6 system for this body
      solveLocalSystem(bodies[i], contacts, numContacts, dt, invDt2);
    }
  } else {
    // Use fast O(1) lookup if constraint mapping is available
    const bool useFastPath = (mContactMap.numBodies > 0 && mContactMap.constraintIndices != nullptr);
    
    for (physx::PxU32 idx = 0; idx < numBodies; ++idx) {
      const physx::PxU32 i = orderPtr ? orderPtr[idx] : idx;
      // Skip static bodies
      if (bodies[i].invMass <= 0.0f) {
        continue;
      }

      if (useFastPath) {
        // Optimized path: O(constraints per body) instead of O(all constraints)
        solveBodyLocalConstraintsFast(bodies, numBodies, i, contacts);
      } else {
        // Fallback: O(N²) path
        solveBodyLocalConstraints(bodies, numBodies, i, contacts, numContacts);
      }
    }
  }
}

/**
 * @brief Solve all constraints affecting a single body (Block Descent step)
 * 
 * This is the core of AVBD: for a single body, we consider all constraints
 * affecting it and compute the optimal position/rotation update.
 */
void AvbdSolver::solveBodyLocalConstraints(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts) {
  
  AvbdSolverBody &body = bodies[bodyIndex];
  
  if (body.invMass <= 0.0f) {
    return;
  }
  
  // Accumulate position and rotation corrections from all constraints
  physx::PxVec3 totalDeltaPos(0.0f);
  physx::PxVec3 totalDeltaTheta(0.0f);
  physx::PxReal totalWeight = 0.0f;

  const bool useKahan =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eUSE_KAHAN_SUMMATION);
  AvbdKahanAccumulator kahanDeltaPos;
  AvbdKahanAccumulator kahanDeltaTheta;
  KahanSum kahanWeight;
  
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
    const physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;
    
    // Skip if this body is not involved
    if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
      continue;
    }
    
    // Get the other body
    AvbdSolverBody *otherBody = nullptr;
    bool isBodyA = (bodyAIdx == bodyIndex);
    
    if (isBodyA && bodyBIdx < numBodies) {
      otherBody = &bodies[bodyBIdx];
    } else if (!isBodyA && bodyAIdx < numBodies) {
      otherBody = &bodies[bodyAIdx];
    }
    
    // Compute world positions of contact points
    physx::PxVec3 worldPosA, worldPosB;
    physx::PxVec3 r;  // Contact arm for this body
    
    if (isBodyA) {
      r = body.rotation.rotate(contacts[c].contactPointA);
      worldPosA = body.position + r;
      if (otherBody) {
        worldPosB = otherBody->position + otherBody->rotation.rotate(contacts[c].contactPointB);
      } else {
        worldPosB = contacts[c].contactPointB;
      }
    } else {
      r = body.rotation.rotate(contacts[c].contactPointB);
      if (otherBody) {
        worldPosA = otherBody->position + otherBody->rotation.rotate(contacts[c].contactPointA);
      } else {
        worldPosA = contacts[c].contactPointA;
      }
      worldPosB = body.position + r;
    }
    
    // Compute separation (negative = penetration)
    const physx::PxVec3 &normal = contacts[c].contactNormal;
    physx::PxReal separation = (worldPosA - worldPosB).dot(normal) + contacts[c].penetrationDepth;
    
    // Only correct if penetrating
    if (separation >= 0.0f) {
      continue;
    }
    
    // Compute generalized inverse mass for normal direction
    physx::PxVec3 rCrossN = r.cross(normal);
    physx::PxReal w = body.invMass + rCrossN.dot(body.invInertiaWorld * rCrossN);
    
    // Add other body's contribution if dynamic
    physx::PxVec3 rOther(0.0f);
    if (otherBody && otherBody->invMass > 0.0f) {
      rOther = isBodyA ?
          otherBody->rotation.rotate(contacts[c].contactPointB) :
          otherBody->rotation.rotate(contacts[c].contactPointA);
      physx::PxVec3 rOtherCrossN = rOther.cross(normal);
      w += otherBody->invMass + rOtherCrossN.dot(otherBody->invInertiaWorld * rOtherCrossN);
    }
    
    if (w <= 1e-6f) {
      continue;
    }
    
    // Compute normal correction magnitude
    physx::PxReal normalCorrectionMag = -separation / w;
    normalCorrectionMag *= mConfig.baumgarte;
    
    // Direction sign for this body
    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
    
    // Normal position and rotation corrections
    physx::PxVec3 deltaPos = normal * (normalCorrectionMag * body.invMass * sign);
    physx::PxVec3 deltaTheta = (body.invInertiaWorld * rCrossN) * (normalCorrectionMag * sign);
    
    //=========================================================================
    // FRICTION CONSTRAINT (Position-based using previous position delta)
    //=========================================================================
    if (contacts[c].friction > 0.0f) {
      // Compute relative position displacement at contact point
      // This is more appropriate for position-based dynamics than velocity
      physx::PxVec3 prevWorldPosA, prevWorldPosB;
      
      if (isBodyA) {
        // Use previous position to compute displacement
        prevWorldPosA = body.prevPosition + body.prevRotation.rotate(contacts[c].contactPointA);
        if (otherBody) {
          prevWorldPosB = otherBody->prevPosition + otherBody->prevRotation.rotate(contacts[c].contactPointB);
        } else {
          prevWorldPosB = contacts[c].contactPointB;
        }
      } else {
        if (otherBody) {
          prevWorldPosA = otherBody->prevPosition + otherBody->prevRotation.rotate(contacts[c].contactPointA);
        } else {
          prevWorldPosA = contacts[c].contactPointA;
        }
        prevWorldPosB = body.prevPosition + body.prevRotation.rotate(contacts[c].contactPointB);
      }
      
      // Relative displacement from previous to current position
      physx::PxVec3 dispA = worldPosA - prevWorldPosA;
      physx::PxVec3 dispB = worldPosB - prevWorldPosB;
      physx::PxVec3 relDisp = isBodyA ? (dispA - dispB) : (dispB - dispA);
      
      // Extract tangential component (remove normal component)
      physx::PxVec3 tangentDisp = relDisp - normal * relDisp.dot(normal);
      physx::PxReal tangentDispMag = tangentDisp.magnitude();
      
      if (tangentDispMag > 1e-6f) {
        physx::PxVec3 tangent = tangentDisp / tangentDispMag;
        
        // Compute generalized inverse mass for tangent direction
        physx::PxVec3 rCrossT = r.cross(tangent);
        physx::PxReal wT = body.invMass + rCrossT.dot(body.invInertiaWorld * rCrossT);
        
        if (otherBody && otherBody->invMass > 0.0f) {
          physx::PxVec3 rOtherCrossT = rOther.cross(tangent);
          wT += otherBody->invMass + rOtherCrossT.dot(otherBody->invInertiaWorld * rOtherCrossT);
        }
        
        if (wT > 1e-6f) {
          // Maximum friction correction based on Coulomb friction model
          // Friction force magnitude <= mu * normal force magnitude
          physx::PxReal maxFrictionCorrection = contacts[c].friction * physx::PxAbs(normalCorrectionMag);
          
          // Compute friction correction to oppose tangent displacement
          physx::PxReal frictionCorrection = tangentDispMag / wT;
          
          // Clamp to Coulomb friction cone
          frictionCorrection = physx::PxMin(frictionCorrection, maxFrictionCorrection);
          
          // Apply friction correction (opposes tangent displacement direction)
          physx::PxVec3 frictionDeltaPos = -tangent * (frictionCorrection * body.invMass);
          physx::PxVec3 frictionDeltaTheta = -(body.invInertiaWorld * rCrossT) * frictionCorrection;
          
          deltaPos += frictionDeltaPos;
          deltaTheta += frictionDeltaTheta;
        }
      }
    }
    
    // Weight by constraint stiffness (rho)
    physx::PxReal weight = contacts[c].header.rho;

    if (useKahan) {
      kahanDeltaPos.add(deltaPos * weight);
      kahanDeltaTheta.add(deltaTheta * weight);
      kahanWeight.add(weight);
    } else {
      totalDeltaPos += deltaPos * weight;
      totalDeltaTheta += deltaTheta * weight;
      totalWeight += weight;
    }
  }

  if (useKahan) {
    totalDeltaPos = kahanDeltaPos.getSum();
    totalDeltaTheta = kahanDeltaTheta.getSum();
    totalWeight = kahanWeight.sum;
  }
  
  // Apply averaged corrections
  if (totalWeight > 0.0f) {
    // Normalize by total weight for stability
    physx::PxReal invWeight = 1.0f / totalWeight;
    
    // Apply position correction
    body.position += totalDeltaPos * invWeight;
    
    // Apply rotation correction using exponential map
    physx::PxVec3 avgDeltaTheta = totalDeltaTheta * invWeight;
    physx::PxReal angle = avgDeltaTheta.magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      angle = physx::PxMin(angle, 0.1f); // Clamp for stability
      physx::PxVec3 axis = avgDeltaTheta.getNormalized();
      physx::PxReal halfAngle = angle * 0.5f;
      physx::PxQuat deltaQ(axis.x * physx::PxSin(halfAngle),
                            axis.y * physx::PxSin(halfAngle),
                            axis.z * physx::PxSin(halfAngle),
                            physx::PxCos(halfAngle));
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
  }
}

/**
 * @brief Solve all constraints affecting a single body
 * 
 * This is the core AVBD block descent step. For a given body, we:
 * 1. Collect all constraint violations and gradients
 * 2. Compute the optimal position/rotation update
 * 3. Apply the update
 */
void AvbdSolver::solveBodyAllConstraints(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
    AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
    AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
    AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxReal dt) {
  
  PX_UNUSED(dt);
  
  AvbdSolverBody &body = bodies[bodyIndex];
  
  if (body.invMass <= 0.0f) {
    return;
  }
  
  // Accumulate corrections from all constraint types
  physx::PxVec3 totalDeltaPos(0.0f);
  physx::PxVec3 totalDeltaTheta(0.0f);
  physx::PxU32 numActiveConstraints = 0;
  
  // Process contact constraints
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computeContactCorrection(contacts[c], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }
  
  // Process spherical joint constraints
  for (physx::PxU32 j = 0; j < numSpherical; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computeSphericalJointCorrection(sphericalJoints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }
  
  // Process fixed joint constraints
  for (physx::PxU32 j = 0; j < numFixed; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computeFixedJointCorrection(fixedJoints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }
  
  // Process revolute joint constraints
  for (physx::PxU32 j = 0; j < numRevolute; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computeRevoluteJointCorrection(revoluteJoints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }
  
  // Process prismatic joint constraints
  for (physx::PxU32 j = 0; j < numPrismatic; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computePrismaticJointCorrection(prismaticJoints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }
  
  // Process D6 joint constraints
  for (physx::PxU32 j = 0; j < numD6; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computeD6JointCorrection(d6Joints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }
  
  // Apply averaged corrections
  if (numActiveConstraints > 0) {
    // Average the corrections (simple averaging, could use weighted average)
    physx::PxReal invCount = 1.0f / static_cast<physx::PxReal>(numActiveConstraints);
    
    // Apply position correction with SOR
    body.position += totalDeltaPos * invCount * mConfig.baumgarte;
    
    // Apply rotation correction
    physx::PxVec3 avgDeltaTheta = totalDeltaTheta * invCount * mConfig.baumgarte;
    physx::PxReal angle = avgDeltaTheta.magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      angle = physx::PxMin(angle, 0.1f);
      physx::PxVec3 axis = avgDeltaTheta.getNormalized();
      physx::PxReal halfAngle = angle * 0.5f;
      physx::PxQuat deltaQ(axis.x * physx::PxSin(halfAngle),
                            axis.y * physx::PxSin(halfAngle),
                            axis.z * physx::PxSin(halfAngle),
                            physx::PxCos(halfAngle));
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
  }
}

/**
 * @brief Compute position/rotation correction for a contact constraint
 * @return true if constraint is active and correction was computed
 */
bool AvbdSolver::computeContactCorrection(
    const AvbdContactConstraint &contact,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta) {
  
  const physx::PxU32 bodyAIdx = contact.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = contact.header.bodyIndexB;
  
  // Skip if this body is not involved
  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }
  
  AvbdSolverBody &body = bodies[bodyIndex];
  bool isBodyA = (bodyAIdx == bodyIndex);
  
  // Get other body
  AvbdSolverBody *otherBody = nullptr;
  if (isBodyA && bodyBIdx < numBodies) {
    otherBody = &bodies[bodyBIdx];
  } else if (!isBodyA && bodyAIdx < numBodies) {
    otherBody = &bodies[bodyAIdx];
  }
  
  // Compute world positions and contact arms
  physx::PxVec3 worldPosA, worldPosB;
  physx::PxVec3 r;  // Contact arm for this body
  
  if (isBodyA) {
    r = body.rotation.rotate(contact.contactPointA);
    worldPosA = body.position + r;
    worldPosB = otherBody ? 
        otherBody->position + otherBody->rotation.rotate(contact.contactPointB) :
        contact.contactPointB;
  } else {
    r = body.rotation.rotate(contact.contactPointB);
    worldPosA = otherBody ?
        otherBody->position + otherBody->rotation.rotate(contact.contactPointA) :
        contact.contactPointA;
    worldPosB = body.position + r;
  }
  
  // Compute separation
  const physx::PxVec3 &normal = contact.contactNormal;
  physx::PxReal separation = (worldPosA - worldPosB).dot(normal) + contact.penetrationDepth;
  
  // Only correct if penetrating
  if (separation >= 0.0f) {
    return false;
  }
  
  // Compute generalized inverse mass for normal
  physx::PxVec3 rCrossN = r.cross(normal);
  physx::PxReal w = body.invMass + rCrossN.dot(body.invInertiaWorld * rCrossN);
  
  physx::PxVec3 rOther(0.0f);
  if (otherBody && otherBody->invMass > 0.0f) {
    rOther = isBodyA ?
        otherBody->rotation.rotate(contact.contactPointB) :
        otherBody->rotation.rotate(contact.contactPointA);
    physx::PxVec3 rOtherCrossN = rOther.cross(normal);
    w += otherBody->invMass + rOtherCrossN.dot(otherBody->invInertiaWorld * rOtherCrossN);
  }
  
  if (w <= 1e-6f) {
    return false;
  }
  
  // Compute normal correction
  physx::PxReal correctionMag = -separation / w;
  physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
  
  deltaPos = normal * (correctionMag * body.invMass * sign);
  deltaTheta = (body.invInertiaWorld * rCrossN) * (correctionMag * sign);
  
  //=========================================================================
  // FRICTION CONSTRAINT (Position-based)
  //=========================================================================
  if (contact.friction > 0.0f) {
    // Compute relative position displacement at contact point
    physx::PxVec3 prevWorldPosA, prevWorldPosB;
    
    if (isBodyA) {
      // Use previous position to compute displacement
      prevWorldPosA = body.prevPosition + body.prevRotation.rotate(contact.contactPointA);
      if (otherBody) {
        prevWorldPosB = otherBody->prevPosition + otherBody->prevRotation.rotate(contact.contactPointB);
      } else {
        prevWorldPosB = contact.contactPointB;
      }
    } else {
      if (otherBody) {
        prevWorldPosA = otherBody->prevPosition + otherBody->prevRotation.rotate(contact.contactPointA);
      } else {
        prevWorldPosA = contact.contactPointA;
      }
      prevWorldPosB = body.prevPosition + body.prevRotation.rotate(contact.contactPointB);
    }
    
    // Relative displacement from previous to current position
    physx::PxVec3 dispA = worldPosA - prevWorldPosA;
    physx::PxVec3 dispB = worldPosB - prevWorldPosB;
    physx::PxVec3 relDisp = isBodyA ? (dispA - dispB) : (dispB - dispA);
    
    // Extract tangential component (remove normal component)
    physx::PxVec3 tangentDisp = relDisp - normal * relDisp.dot(normal);
    physx::PxReal tangentDispMag = tangentDisp.magnitude();
    
    if (tangentDispMag > 1e-6f) {
      physx::PxVec3 tangent = tangentDisp / tangentDispMag;
      
      // Compute generalized inverse mass for tangent direction
      physx::PxVec3 rCrossT = r.cross(tangent);
      physx::PxReal wT = body.invMass + rCrossT.dot(body.invInertiaWorld * rCrossT);
      
      if (otherBody && otherBody->invMass > 0.0f) {
        physx::PxVec3 rOtherCrossT = rOther.cross(tangent);
        wT += otherBody->invMass + rOtherCrossT.dot(otherBody->invInertiaWorld * rOtherCrossT);
      }
      
      if (wT > 1e-6f) {
        // Maximum friction correction based on Coulomb friction model
        // Friction force magnitude <= mu * normal force magnitude
        physx::PxReal maxFrictionCorrection = contact.friction * physx::PxAbs(correctionMag);
        
        // Compute friction correction to oppose tangent displacement
        physx::PxReal frictionCorrection = tangentDispMag / wT;
        frictionCorrection = physx::PxMin(frictionCorrection, maxFrictionCorrection);
        
        // Apply friction correction (opposes tangent displacement direction)
        deltaPos += -tangent * (frictionCorrection * body.invMass);
        deltaTheta += -(body.invInertiaWorld * rCrossT) * frictionCorrection;
      }
    }
  }
  
  return true;
}

/**
 * @brief Compute position/rotation correction for a spherical joint
 */
bool AvbdSolver::computeSphericalJointCorrection(
    const AvbdSphericalJointConstraint &joint,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta) {
  
  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;
  
  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }
  
  AvbdSolverBody &body = bodies[bodyIndex];
  bool isBodyA = (bodyAIdx == bodyIndex);
  
  // Get other body
  AvbdSolverBody *otherBody = nullptr;
  if (isBodyA && bodyBIdx < numBodies) {
    otherBody = &bodies[bodyBIdx];
  } else if (!isBodyA && bodyAIdx < numBodies) {
    otherBody = &bodies[bodyAIdx];
  }
  
  // Compute world anchor positions
  physx::PxVec3 worldAnchorA, worldAnchorB;
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB = otherBody ?
        otherBody->position + otherBody->rotation.rotate(joint.anchorB) :
        joint.anchorB;
  } else {
    worldAnchorA = otherBody ?
        otherBody->position + otherBody->rotation.rotate(joint.anchorA) :
        joint.anchorA;
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
  }
  
  // Compute constraint error
  physx::PxVec3 error = worldAnchorA - worldAnchorB;
  physx::PxReal errorMag = error.magnitude();
  
  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);
  bool hasCorrection = false;
  
  if (errorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxVec3 direction = error / errorMag;
    
    // Compute generalized inverse mass
    physx::PxVec3 r = isBodyA ?
        body.rotation.rotate(joint.anchorA) :
        body.rotation.rotate(joint.anchorB);
    physx::PxVec3 rCrossD = r.cross(direction);
    physx::PxReal w = body.invMass + rCrossD.dot(body.invInertiaWorld * rCrossD);
    
    if (otherBody && otherBody->invMass > 0.0f) {
      physx::PxVec3 rOther = isBodyA ?
          otherBody->rotation.rotate(joint.anchorB) :
          otherBody->rotation.rotate(joint.anchorA);
      physx::PxVec3 rOtherCrossD = rOther.cross(direction);
      w += otherBody->invMass + rOtherCrossD.dot(otherBody->invInertiaWorld * rOtherCrossD);
    }
    
    if (w > 1e-6f) {
      // Compute correction
      physx::PxReal correctionMag = -errorMag / w;
      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
      
      deltaPos = direction * (correctionMag * body.invMass * sign);
      deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
      hasCorrection = true;
    }
  }
  
  // Process cone limit if enabled
  if (joint.hasConeLimit && joint.coneAngleLimit > 0.0f) {
    physx::PxQuat rotA = isBodyA ? body.rotation : (otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity));
    physx::PxQuat rotB = isBodyA ? (otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity)) : body.rotation;
    
    // Compute cone violation
    physx::PxVec3 worldAxisA = rotA.rotate(joint.coneAxisA);
    physx::PxVec3 worldAxisB = rotB.rotate(joint.coneAxisA);
    
    physx::PxReal cosAngle = worldAxisA.dot(worldAxisB);
    cosAngle = physx::PxClamp(cosAngle, -1.0f, 1.0f);
    physx::PxReal angle = physx::PxAcos(cosAngle);
    
    if (angle > joint.coneAngleLimit) {
      // Cone limit violated
      physx::PxReal violation = angle - joint.coneAngleLimit;
      
      // Correction axis is perpendicular to both
      physx::PxVec3 corrAxis = worldAxisA.cross(worldAxisB);
      physx::PxReal corrAxisMag = corrAxis.magnitude();
      
      if (corrAxisMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
        corrAxis /= corrAxisMag;
        
        // Compute angular correction
        physx::PxReal angularW = 0.0f;
        angularW += (body.invInertiaWorld * corrAxis).dot(corrAxis);
        if (otherBody && otherBody->invMass > 0.0f) {
          angularW += (otherBody->invInertiaWorld * corrAxis).dot(corrAxis);
        }
        
        if (angularW > 1e-6f) {
          physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
          physx::PxVec3 angCorrection = corrAxis * (violation * 0.5f / angularW * sign);
          deltaTheta += body.invInertiaWorld * angCorrection;
          hasCorrection = true;
        }
      }
    }
  }
  
  return hasCorrection;
}

/**
 * @brief Compute position/rotation correction for a fixed joint
 */
bool AvbdSolver::computeFixedJointCorrection(
    const AvbdFixedJointConstraint &joint,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta) {
  
  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;
  
  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }
  
  AvbdSolverBody &body = bodies[bodyIndex];
  bool isBodyA = (bodyAIdx == bodyIndex);
  
  AvbdSolverBody *otherBody = nullptr;
  if (isBodyA && bodyBIdx < numBodies) {
    otherBody = &bodies[bodyBIdx];
  } else if (!isBodyA && bodyAIdx < numBodies) {
    otherBody = &bodies[bodyAIdx];
  }
  
  // Position constraint (same as spherical)
  physx::PxVec3 worldAnchorA, worldAnchorB;
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB = otherBody ?
        otherBody->position + otherBody->rotation.rotate(joint.anchorB) :
        joint.anchorB;
  } else {
    worldAnchorA = otherBody ?
        otherBody->position + otherBody->rotation.rotate(joint.anchorA) :
        joint.anchorA;
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
  }
  
  physx::PxVec3 posError = worldAnchorA - worldAnchorB;
  physx::PxReal posErrorMag = posError.magnitude();
  
  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);
  
  if (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxVec3 direction = posError / posErrorMag;
    
    physx::PxVec3 r = isBodyA ?
        body.rotation.rotate(joint.anchorA) :
        body.rotation.rotate(joint.anchorB);
    physx::PxVec3 rCrossD = r.cross(direction);
    physx::PxReal w = body.invMass + rCrossD.dot(body.invInertiaWorld * rCrossD);
    
    if (otherBody && otherBody->invMass > 0.0f) {
      physx::PxVec3 rOther = isBodyA ?
          otherBody->rotation.rotate(joint.anchorB) :
          otherBody->rotation.rotate(joint.anchorA);
      physx::PxVec3 rOtherCrossD = rOther.cross(direction);
      w += otherBody->invMass + rOtherCrossD.dot(otherBody->invInertiaWorld * rOtherCrossD);
    }
    
    if (w > 1e-6f) {
      physx::PxReal correctionMag = -posErrorMag / w;
      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
      
      deltaPos = direction * (correctionMag * body.invMass * sign);
      deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
    }
  }
  
  // Rotation constraint for fixed joint - lock all 3 rotation axes
  physx::PxQuat rotA, rotB;
  if (isBodyA) {
    rotA = body.rotation;
    rotB = otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity);
  } else {
    rotA = otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity);
    rotB = body.rotation;
  }
  
  // Target: rotB = rotA * relativeRotation
  // Current relative rotation vs target
  physx::PxQuat targetRotB = rotA * joint.relativeRotation;
  physx::PxQuat errorQ = targetRotB.getConjugate() * rotB;
  if (errorQ.w < 0.0f) {
    errorQ = -errorQ;  // Shortest path
  }
  
  // Convert to axis-angle in world space (small angle approximation)
  // errorQ represents the rotation needed to align rotB with target
  physx::PxVec3 rotErrorLocal(errorQ.x * 2.0f, errorQ.y * 2.0f, errorQ.z * 2.0f);
  // Transform to world space
  physx::PxVec3 rotError = rotB.rotate(rotErrorLocal);
  physx::PxReal rotErrorMag = rotError.magnitude();
  
  if (rotErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    // Compute effective mass for rotation
    physx::PxVec3 axis = rotError / rotErrorMag;
    physx::PxReal w = axis.dot(body.invInertiaWorld * axis);
    
    if (otherBody && otherBody->invMass > 0.0f) {
      w += axis.dot(otherBody->invInertiaWorld * axis);
    }
    
    if (w > 1e-6f) {
      // Apply damped correction to avoid oscillation
      physx::PxReal compliance = 0.0001f;  // Small compliance for stability
      physx::PxReal correctionMag = rotErrorMag / (w + compliance);
      correctionMag = physx::PxMin(correctionMag, rotErrorMag * 0.8f);  // Limit correction
      
      // Body B should rotate towards target (negative correction)
      // Body A should rotate away (positive correction)
      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
      physx::PxVec3 deltaOmega = axis * (correctionMag * sign);
      deltaTheta += body.invInertiaWorld * deltaOmega / w;
    }
  }
  
  return (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON || 
          rotErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON);
}

/**
 * @brief Compute correction for revolute joint
 */
bool AvbdSolver::computeRevoluteJointCorrection(
    const AvbdRevoluteJointConstraint &joint,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta) {
  
  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;
  
  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }
  
  AvbdSolverBody &body = bodies[bodyIndex];
  bool isBodyA = (bodyAIdx == bodyIndex);
  
  AvbdSolverBody *otherBody = nullptr;
  if (isBodyA && bodyBIdx < numBodies) {
    otherBody = &bodies[bodyBIdx];
  } else if (!isBodyA && bodyAIdx < numBodies) {
    otherBody = &bodies[bodyAIdx];
  }
  
  // Position constraint
  physx::PxVec3 worldAnchorA, worldAnchorB;
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB = otherBody ?
        otherBody->position + otherBody->rotation.rotate(joint.anchorB) :
        joint.anchorB;
  } else {
    worldAnchorA = otherBody ?
        otherBody->position + otherBody->rotation.rotate(joint.anchorA) :
        joint.anchorA;
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
  }
  
  physx::PxVec3 posError = worldAnchorA - worldAnchorB;
  physx::PxReal posErrorMag = posError.magnitude();
  
  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);
  
  if (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxVec3 direction = posError / posErrorMag;
    
    physx::PxVec3 r = isBodyA ?
        body.rotation.rotate(joint.anchorA) :
        body.rotation.rotate(joint.anchorB);
    physx::PxVec3 rCrossD = r.cross(direction);
    physx::PxReal w = body.invMass + rCrossD.dot(body.invInertiaWorld * rCrossD);
    
    if (otherBody && otherBody->invMass > 0.0f) {
      physx::PxVec3 rOther = isBodyA ?
          otherBody->rotation.rotate(joint.anchorB) :
          otherBody->rotation.rotate(joint.anchorA);
      physx::PxVec3 rOtherCrossD = rOther.cross(direction);
      w += otherBody->invMass + rOtherCrossD.dot(otherBody->invInertiaWorld * rOtherCrossD);
    }
    
    if (w > 1e-6f) {
      physx::PxReal correctionMag = -posErrorMag / w;
      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
      
      deltaPos = direction * (correctionMag * body.invMass * sign);
      deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
    }
  }
  
  // Axis alignment constraint
  physx::PxVec3 worldAxisA = isBodyA ? 
      body.rotation.rotate(joint.axisA) :
      (otherBody ? otherBody->rotation.rotate(joint.axisA) : joint.axisA);
  physx::PxVec3 worldAxisB = isBodyA ?
      (otherBody ? otherBody->rotation.rotate(joint.axisB) : joint.axisB) :
      body.rotation.rotate(joint.axisB);
  
  physx::PxVec3 axisCross = worldAxisA.cross(worldAxisB);
  physx::PxReal axisError = axisCross.magnitude();
  
  if (axisError > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxReal sign = isBodyA ? -1.0f : 1.0f;
    deltaTheta += axisCross * (0.5f * sign);
  }
  
  return (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON ||
          axisError > AvbdConstants::AVBD_NUMERICAL_EPSILON);
}

/**
 * @brief Compute correction for prismatic joint
 */
bool AvbdSolver::computePrismaticJointCorrection(
    const AvbdPrismaticJointConstraint &joint,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta) {
  
  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;
  
  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }
  
  AvbdSolverBody &body = bodies[bodyIndex];
  bool isBodyA = (bodyAIdx == bodyIndex);
  
  AvbdSolverBody *otherBody = nullptr;
  if (isBodyA && bodyBIdx < numBodies) {
    otherBody = &bodies[bodyBIdx];
  } else if (!isBodyA && bodyAIdx < numBodies) {
    otherBody = &bodies[bodyAIdx];
  }
  
  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);
  
  physx::PxVec3 worldAnchorA, worldAnchorB;
  physx::PxVec3 worldAxis;
  
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB = otherBody ? 
        otherBody->position + otherBody->rotation.rotate(joint.anchorB) :
        joint.anchorB;
    worldAxis = body.rotation.rotate(joint.axisA);
  } else {
    worldAnchorA = otherBody ?
        otherBody->position + otherBody->rotation.rotate(joint.anchorA) :
        joint.anchorA;
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
    worldAxis = otherBody ? 
        otherBody->rotation.rotate(joint.axisA) : joint.axisA;
  }
  
  physx::PxVec3 diff = worldAnchorA - worldAnchorB;
  physx::PxVec3 perpError = diff - worldAxis * diff.dot(worldAxis);
  physx::PxReal perpErrorMag = perpError.magnitude();
  
  if (perpErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxVec3 direction = perpError / perpErrorMag;
    
    physx::PxVec3 r = isBodyA ?
        body.rotation.rotate(joint.anchorA) :
        body.rotation.rotate(joint.anchorB);
    physx::PxVec3 rCrossD = r.cross(direction);
    physx::PxReal w = body.invMass + rCrossD.dot(body.invInertiaWorld * rCrossD);
    
    if (otherBody && otherBody->invMass > 0.0f) {
      physx::PxVec3 rOther = isBodyA ?
          otherBody->rotation.rotate(joint.anchorB) :
          otherBody->rotation.rotate(joint.anchorA);
      physx::PxVec3 rOtherCrossD = rOther.cross(direction);
      w += otherBody->invMass + rOtherCrossD.dot(otherBody->invInertiaWorld * rOtherCrossD);
    }
    
    if (w > 1e-6f) {
      physx::PxReal correctionMag = -perpErrorMag / w;
      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
      
      deltaPos = direction * (correctionMag * body.invMass * sign);
      deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
    }
  }
  
  return perpErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON;
}

/**
 * @brief Compute correction for D6 joint
 */
bool AvbdSolver::computeD6JointCorrection(
    const AvbdD6JointConstraint &joint,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta) {
  
  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;
  
  bool bodyAIsStatic = (bodyAIdx >= numBodies);
  
  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }
  
  AvbdSolverBody &body = bodies[bodyIndex];
  bool isBodyA = (bodyAIdx == bodyIndex);
  
  AvbdSolverBody *otherBody = nullptr;
  if (isBodyA && bodyBIdx < numBodies) {
    otherBody = &bodies[bodyBIdx];
  } else if (!isBodyA && bodyAIdx < numBodies) {
    otherBody = &bodies[bodyAIdx];
  }
  
  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);
  
  bool hasCorrection = false;
  
  // Check if bodies are static (index >= numBodies means static body, frame already in world space)
  // Note: bodyAIsStatic already defined above for debug purposes
  bool bodyBIsStatic = (bodyBIdx >= numBodies);
  
  // Get rotations for frame transforms
  physx::PxQuat rotA = bodyAIsStatic ? physx::PxQuat(physx::PxIdentity) : 
                       (isBodyA ? body.rotation : (otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity)));
  physx::PxQuat rotB = bodyBIsStatic ? physx::PxQuat(physx::PxIdentity) :
                       (isBodyA ? (otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity)) : body.rotation);
  
  physx::PxVec3 worldAnchorA, worldAnchorB;
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB = otherBody ?
        otherBody->position + otherBody->rotation.rotate(joint.anchorB) :
        joint.anchorB;  // anchorB already in world space for static
  } else {
    worldAnchorA = otherBody ?
        otherBody->position + otherBody->rotation.rotate(joint.anchorA) :
        joint.anchorA;  // anchorA already in world space for static
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
  }
  
  physx::PxVec3 posError = worldAnchorA - worldAnchorB;
  
  // Position constraint (linear locked) - but skip axes with velocity drive
  // When velocity drive is active, we want the body to move, not be constrained
  if (joint.linearMotion == 0) {
    // Determine which axes have velocity drive (we'll skip position constraint on those)
    physx::PxU32 linearDriveAxes = joint.driveFlags & 0x7;  // bits 0,1,2 for X,Y,Z
    
    // If we have velocity drive, project out the position error along driven axes
    physx::PxVec3 constrainedPosError = posError;
    
    if (linearDriveAxes != 0 && !isBodyA) {
      // Get joint frame in world space
      physx::PxQuat jointFrameA = bodyAIsStatic ? joint.localFrameA : (rotA * joint.localFrameA);
      physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
      if (qMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON && PxIsFinite(qMag2)) {
        jointFrameA *= 1.0f / physx::PxSqrt(qMag2);
        
        // Remove position error component along driven axes
        for (int axis = 0; axis < 3; ++axis) {
          if ((linearDriveAxes & (1 << axis)) != 0) {
            physx::PxVec3 localAxis(0.0f);
            (&localAxis.x)[axis] = 1.0f;
            physx::PxVec3 worldAxis = jointFrameA.rotate(localAxis);
            // Remove the component of position error along this driven axis
            constrainedPosError -= worldAxis * constrainedPosError.dot(worldAxis);
          }
        }
      }
    }
    
    physx::PxReal posErrorMag = constrainedPosError.magnitude();
    if (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      physx::PxVec3 direction = constrainedPosError / posErrorMag;
      
      physx::PxVec3 r = isBodyA ?
          body.rotation.rotate(joint.anchorA) :
          body.rotation.rotate(joint.anchorB);
      physx::PxVec3 rCrossD = r.cross(direction);
      physx::PxReal w = body.invMass + rCrossD.dot(body.invInertiaWorld * rCrossD);
      
      if (otherBody && otherBody->invMass > 0.0f) {
        physx::PxVec3 rOther = isBodyA ?
            otherBody->rotation.rotate(joint.anchorB) :
            otherBody->rotation.rotate(joint.anchorA);
        physx::PxVec3 rOtherCrossD = rOther.cross(direction);
        w += otherBody->invMass + rOtherCrossD.dot(otherBody->invInertiaWorld * rOtherCrossD);
      }
      
      if (w > 1e-6f) {
        physx::PxReal correctionMag = -posErrorMag / w;
        physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
        
        deltaPos = direction * (correctionMag * body.invMass * sign);
        deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
      }
      hasCorrection = true;
    }
  }
  
  // Linear velocity drive
  // PxD6Drive: eX=0, eY=1, eZ=2 -> bits 0,1,2
  // Only apply delta when processing body B (the driven body)
  if ((joint.driveFlags & 0x7) != 0 && body.invMass > 0.0f) {
    // Drive target velocity is in joint frame A's space
    // If body A is static, use localFrameA directly; otherwise transform it
    physx::PxQuat jointFrameA = bodyAIsStatic ? joint.localFrameA : (rotA * joint.localFrameA);
    
    // Validate and normalize quaternion
    physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
    if (qMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON && PxIsFinite(qMag2)) {
      jointFrameA *= 1.0f / physx::PxSqrt(qMag2);
      
      for (int axis = 0; axis < 3; ++axis) {
        if ((joint.driveFlags & (1 << axis)) == 0) continue;
        
        physx::PxReal targetVel = (&joint.driveLinearVelocity.x)[axis];
        physx::PxReal damping = (&joint.linearDamping.x)[axis];
        
        // Skip if no damping or invalid values
        if (damping <= 0.0f || !PxIsFinite(targetVel) || !PxIsFinite(damping)) continue;
        
        // Get axis direction in world space
        physx::PxVec3 localAxis(0.0f);
        (&localAxis.x)[axis] = 1.0f;
        physx::PxVec3 worldAxis = jointFrameA.rotate(localAxis);
        
        // Validate world axis
        physx::PxReal axisMag2 = worldAxis.magnitudeSquared();
        if (axisMag2 < 0.9f || axisMag2 > 1.1f || !PxIsFinite(axisMag2)) continue;
        
        // Current relative velocity along this axis
        // Body B should move in +axis direction for positive target velocity
        physx::PxVec3 velA = otherBody ? otherBody->linearVelocity : physx::PxVec3(0.0f);
        physx::PxVec3 velB = body.linearVelocity;
        physx::PxReal relVel = (velB - velA).dot(worldAxis);
        
        if (!PxIsFinite(relVel)) continue;
        
        // Velocity error: positive error means we need to speed up in target direction
        physx::PxReal velError = targetVel - relVel;
        
        if (physx::PxAbs(velError) > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
          // Position-based velocity drive: deltaX = targetVel * dt
          // Scale by damping factor and dt
          physx::PxReal dt = 1.0f / 60.0f;  // Approximate timestep
          // Normalize damping: 1000 -> factor for reasonable speed
          physx::PxReal dampingFactor = physx::PxMin(damping / 30000.0f, 0.05f);
          
          // Direct position change based on target velocity
          // Positive targetVel -> body B moves in +worldAxis direction
          physx::PxVec3 posDelta = worldAxis * (targetVel * dt * dampingFactor);
          
          // Only apply to body B, not body A
          // When processing body A (isBodyA=true), skip applying the drive
          if (!isBodyA) {
            // Validate correction before applying
            if (PxIsFinite(posDelta.x) && PxIsFinite(posDelta.y) && PxIsFinite(posDelta.z)) {
              // Clamp max correction per frame
              physx::PxReal maxCorrection = 0.005f;
              posDelta.x = physx::PxClamp(posDelta.x, -maxCorrection, maxCorrection);
              posDelta.y = physx::PxClamp(posDelta.y, -maxCorrection, maxCorrection);
              posDelta.z = physx::PxClamp(posDelta.z, -maxCorrection, maxCorrection);
              deltaPos += posDelta;
              hasCorrection = true;
            }
          }
        }
      }
    }
  }
  
  // Angular velocity drive
  // PxD6Drive: eTWIST=4 (bit 4=0x10), eSLERP=5 (bit 5=0x20), 
  //            eSWING1=6 (bit 6=0x40), eSWING2=7 (bit 7=0x80)
  // Combined angular mask: 0xF0 (bits 4-7)
  if ((joint.driveFlags & 0xF0) != 0) {
    // Drive target velocity is in joint frame A's space
    // If body A is static, use localFrameA directly; otherwise transform it
    physx::PxQuat jointFrameA = bodyAIsStatic ? joint.localFrameA : (rotA * joint.localFrameA);
    
    // Validate and normalize quaternion
    physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
    if (qMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON && PxIsFinite(qMag2)) {
      jointFrameA *= 1.0f / physx::PxSqrt(qMag2);
      
      // SLERP drive (PxD6Drive::eSLERP = 5, bit 5 = 0x20)
      bool slerpDrive = (joint.driveFlags & 0x20) != 0;
      
      physx::PxVec3 targetAngVel = joint.driveAngularVelocity;
      
      // Validate target angular velocity
      if (!PxIsFinite(targetAngVel.x) || !PxIsFinite(targetAngVel.y) || !PxIsFinite(targetAngVel.z)) {
        targetAngVel = physx::PxVec3(0.0f);
      }
      
      // Transform target velocity from joint frame to world frame
      physx::PxVec3 worldTargetAngVel = jointFrameA.rotate(targetAngVel);
      
      physx::PxVec3 thetaDelta(0.0f);
      physx::PxReal maxDamping = 0.0f;
      physx::PxReal dt = 1.0f / 60.0f;
      // Normalize damping: 1000 -> factor for reasonable speed
      
      if (slerpDrive) {
        // SLERP applies to all angular axes
        // angularDamping.z = SLERP damping
        maxDamping = joint.angularDamping.z;
        if (maxDamping > 0.0f && PxIsFinite(maxDamping)) {
          physx::PxReal dampingFactor = physx::PxMin(maxDamping / 8000.0f, 0.15f);
          // Apply angular velocity in world space
          thetaDelta = worldTargetAngVel * (dt * dampingFactor);
        }
      } else {
        // Individual axis drives: TWIST (bit 4), SWING1 (bit 6), SWING2 (bit 7)
        // TWIST = X axis (PxD6Drive::eTWIST = 4), damping from angularDamping.x
        if (joint.driveFlags & 0x10) {  // bit 4 = TWIST
          physx::PxReal damping = joint.angularDamping.x;
          if (damping > 0.0f && PxIsFinite(damping)) {
            maxDamping = physx::PxMax(maxDamping, damping);
            physx::PxVec3 worldAxisX = jointFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
            physx::PxReal dampingFactor = physx::PxMin(damping / 8000.0f, 0.15f);
            physx::PxReal targetOnAxis = worldTargetAngVel.dot(worldAxisX);
            // Negate for correct direction
            thetaDelta -= worldAxisX * (targetOnAxis * dt * dampingFactor);
          }
        }
        
        // SWING1 = Y axis (PxD6Drive::eSWING1 = 6), damping from angularDamping.y
        if (joint.driveFlags & 0x40) {  // bit 6 = SWING1
          physx::PxReal damping = joint.angularDamping.y;
          if (damping > 0.0f && PxIsFinite(damping)) {
            maxDamping = physx::PxMax(maxDamping, damping);
            physx::PxVec3 worldAxisY = jointFrameA.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f));
            physx::PxReal dampingFactor = physx::PxMin(damping / 8000.0f, 0.15f);
            physx::PxReal targetOnAxis = worldTargetAngVel.dot(worldAxisY);
            // Negate for correct direction
            thetaDelta -= worldAxisY * (targetOnAxis * dt * dampingFactor);
          }
        }
        
        // SWING2 = Z axis (PxD6Drive::eSWING2 = 7), damping from angularDamping.z
        if (joint.driveFlags & 0x80) {  // bit 7 = SWING2
          physx::PxReal damping = joint.angularDamping.z;
          if (damping > 0.0f && PxIsFinite(damping)) {
            maxDamping = physx::PxMax(maxDamping, damping);
            physx::PxVec3 worldAxisZ = jointFrameA.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f));
            physx::PxReal dampingFactor = physx::PxMin(damping / 8000.0f, 0.15f);
            physx::PxReal targetOnAxis = worldTargetAngVel.dot(worldAxisZ);
            // Negate for correct direction
            thetaDelta -= worldAxisZ * (targetOnAxis * dt * dampingFactor);
          }
        }
      }
      
      physx::PxReal thetaMag2 = thetaDelta.magnitudeSquared();
      if (thetaMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON && PxIsFinite(thetaMag2)) {
        // Only apply to body B, not body A
        if (!isBodyA) {
          // Validate correction before applying
          if (PxIsFinite(thetaDelta.x) && PxIsFinite(thetaDelta.y) && PxIsFinite(thetaDelta.z)) {
            // Clamp max angular correction
            physx::PxReal maxAngCorrection = 0.01f;
            thetaDelta.x = physx::PxClamp(thetaDelta.x, -maxAngCorrection, maxAngCorrection);
            thetaDelta.y = physx::PxClamp(thetaDelta.y, -maxAngCorrection, maxAngCorrection);
            thetaDelta.z = physx::PxClamp(thetaDelta.z, -maxAngCorrection, maxAngCorrection);
            deltaTheta += thetaDelta;
            hasCorrection = true;
          }
        }
      }
    }
  }
  
  return hasCorrection;
}

/**
 * @brief Compute correction for gear joint
 *
 * The gear joint constrains the angular velocities of two bodies around their
 * respective axes such that they maintain a fixed ratio:
 *   omega0 * gearRatio + omega1 = 0
 *
 * In position-based form, we constrain the angular displacement:
 *   deltaTheta0 * gearRatio + deltaTheta1 = 0
 * 
 * Where deltaTheta is the change in angle from prevRotation to current rotation.
 */
bool AvbdSolver::computeGearJointCorrection(
    const AvbdGearJointConstraint &joint,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta) {
  
  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;
  
  // Check if this body is involved in the constraint
  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }
  
  // Check for valid body indices - both must be dynamic for gear to work
  if (bodyAIdx >= numBodies || bodyBIdx >= numBodies) {
    return false;  // Gear joint needs both bodies to be dynamic
  }
  
  AvbdSolverBody &bodyA = bodies[bodyAIdx];
  AvbdSolverBody &bodyB = bodies[bodyBIdx];
  
  // Skip if either body is static
  if (bodyA.invMass <= 0.0f || bodyB.invMass <= 0.0f) {
    return false;
  }
  
  bool isBodyA = (bodyAIdx == bodyIndex);
  
  // Initialize outputs
  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);
  
  // Use joint.gearAxis0/1 as local axes, transform to world space using CURRENT rotation
  physx::PxVec3 worldAxis0 = bodyA.rotation.rotate(joint.gearAxis0);
  physx::PxVec3 worldAxis1 = bodyB.rotation.rotate(joint.gearAxis1);
  
  // Normalize axes
  physx::PxReal axis0Len = worldAxis0.magnitude();
  physx::PxReal axis1Len = worldAxis1.magnitude();
  if (axis0Len < AvbdConstants::AVBD_NUMERICAL_EPSILON ||
      axis1Len < AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    return false;
  }
  worldAxis0 /= axis0Len;
  worldAxis1 /= axis1Len;
  
  // Compute angle changes since start of frame (prevRotation -> rotation)
  // deltaQ = rotation * prevRotation.conjugate()
  physx::PxQuat deltaQA = bodyA.rotation * bodyA.prevRotation.getConjugate();
  physx::PxQuat deltaQB = bodyB.rotation * bodyB.prevRotation.getConjugate();
  
  // Ensure shortest path
  if (deltaQA.w < 0.0f) deltaQA = -deltaQA;
  if (deltaQB.w < 0.0f) deltaQB = -deltaQB;
  
  // Extract angle around the gear axis
  // Use local axis for extraction since we want rotation around the hinge axis
  physx::PxVec3 localAxis0 = joint.gearAxis0;
  physx::PxVec3 localAxis1 = joint.gearAxis1;
  localAxis0.normalize();
  localAxis1.normalize();
  
  // Convert deltaQ to axis-angle and project onto gear axis
  physx::PxVec3 deltaAngA(deltaQA.x, deltaQA.y, deltaQA.z);
  physx::PxVec3 deltaAngB(deltaQB.x, deltaQB.y, deltaQB.z);
  
  // For small angles: angle ≈ 2 * |imaginary part|, direction = imaginary part / |imaginary part|
  // So angular displacement vector ≈ 2 * imaginary part
  deltaAngA *= 2.0f;
  deltaAngB *= 2.0f;
  
  // Transform to local space to get rotation around gear axis
  physx::PxVec3 localDeltaAngA = bodyA.prevRotation.getConjugate().rotate(deltaAngA);
  physx::PxVec3 localDeltaAngB = bodyB.prevRotation.getConjugate().rotate(deltaAngB);
  
  physx::PxReal thetaA = localDeltaAngA.dot(localAxis0);  // Rotation of body A around its gear axis
  physx::PxReal thetaB = localDeltaAngB.dot(localAxis1);  // Rotation of body B around its gear axis
  
  // Gear constraint: thetaA * gearRatio - thetaB = 0
  // This means: if A rotates by angle theta, B should rotate by theta * gearRatio (OPPOSITE direction)
  // The minus sign ensures opposite rotation direction (gears mesh)
  physx::PxReal violation = thetaA * joint.gearRatio - thetaB + joint.geometricError;
  
  // Skip if violation is negligible
  if (physx::PxAbs(violation) < 1e-6f) {
    return false;
  }
  
  // Compute effective inertia for angular constraint along axis
  physx::PxVec3 Iinv_axis0 = bodyA.invInertiaWorld * worldAxis0;
  physx::PxVec3 Iinv_axis1 = bodyB.invInertiaWorld * worldAxis1;
  physx::PxReal w0 = worldAxis0.dot(Iinv_axis0);
  physx::PxReal w1 = worldAxis1.dot(Iinv_axis1);
  
  // Effective inverse inertia for gear constraint:
  // w_eff = gearRatio^2 * w0 + w1
  physx::PxReal wEff = joint.gearRatio * joint.gearRatio * w0 + w1;
  
  if (wEff < 1e-10f) {
    return false;
  }
  
  // Compute Lagrange multiplier (correction impulse magnitude)
  physx::PxReal lambda = -violation / wEff;
  
  // Apply correction to this body
  // Constraint: C = thetaA * gearRatio - thetaB = 0
  // Gradient for A: ∂C/∂thetaA = +gearRatio
  // Gradient for B: ∂C/∂thetaB = -1
  if (isBodyA) {
    // Body A gets impulse along axis0, scaled by gearRatio
    physx::PxReal impulseMag = joint.gearRatio * lambda;
    deltaTheta = (bodyA.invInertiaWorld * worldAxis0) * impulseMag;
  } else {
    // Body B gets impulse along axis1, with NEGATIVE sign (opposite direction)
    physx::PxReal impulseMag = -lambda;  // Note: negative because gradient is -1
    deltaTheta = (bodyB.invInertiaWorld * worldAxis1) * impulseMag;
  }
  
  // Clamp the correction to prevent instability
  physx::PxReal maxAngCorrection = 0.1f;
  physx::PxReal thetaMag = deltaTheta.magnitude();
  if (thetaMag > maxAngCorrection) {
    deltaTheta *= maxAngCorrection / thetaMag;
  }
  
  return true;
}

//=============================================================================
// Solver with Joint Constraints
//=============================================================================

void AvbdSolver::solveWithJoints(
    physx::PxReal dt, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
    AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
    AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
    AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    const physx::PxVec3 &gravity,
    const AvbdBodyConstraintMap *contactMap,
    const AvbdBodyConstraintMap *sphericalMap,
    const AvbdBodyConstraintMap *fixedMap,
    const AvbdBodyConstraintMap *revoluteMap,
    const AvbdBodyConstraintMap *prismaticMap,
    const AvbdBodyConstraintMap *d6Map,
    const AvbdBodyConstraintMap *gearMap,
    AvbdColorBatch *colorBatches,
    physx::PxU32 numColors) {

  PX_PROFILE_ZONE("AVBD.solveWithJoints", 0);

  PX_UNUSED(colorBatches);
  PX_UNUSED(numColors);
  
  // Debug: Track solver entry
  if (!mInitialized || numBodies == 0) {
    return;
  }
  
  // Check if we have pre-computed constraint mappings for O(M) optimization
  const bool hasConstraintMappings = (contactMap != nullptr) || 
                                     (sphericalMap != nullptr) ||
                                     (fixedMap != nullptr) ||
                                     (revoluteMap != nullptr) ||
                                     (prismaticMap != nullptr) ||
                                     (d6Map != nullptr) ||
                                     (gearMap != nullptr);

  mStats.reset();
  mStats.numBodies = numBodies;
  mStats.numContacts = numContacts;
  mStats.numJoints = numSpherical + numFixed + numRevolute + numPrismatic + numD6 + numGear;

  physx::PxReal invDt = 1.0f / dt;

  {
    PX_PROFILE_ZONE("AVBD.prediction", 0);
    computePrediction(bodies, numBodies, dt, gravity);
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].prevPosition = bodies[i].position;
    bodies[i].prevRotation = bodies[i].rotation;
    bodies[i].position = bodies[i].predictedPosition;
    bodies[i].rotation = bodies[i].predictedRotation;
  }

  // Empty mappings for fallback when nullptr is passed
  static const AvbdBodyConstraintMap emptyMap;

  {
    PX_PROFILE_ZONE("AVBD.solveIterations", 0);
    
    for (physx::PxU32 outerIter = 0; outerIter < mConfig.outerIterations; ++outerIter) {
      for (physx::PxU32 innerIter = 0; innerIter < mConfig.innerIterations; ++innerIter) {
        PX_PROFILE_ZONE("AVBD.blockDescentWithJoints", 0);
        
        for (physx::PxU32 bodyIdx = 0; bodyIdx < numBodies; ++bodyIdx) {
          if (bodies[bodyIdx].invMass <= 0.0f) {
            continue;
          }
          
          if (hasConstraintMappings) {
            // Use optimized O(M) version with pre-computed constraint mappings
            solveBodyAllConstraintsFast(
                bodies, numBodies, bodyIdx,
                contacts, numContacts,
                sphericalJoints, numSpherical,
                fixedJoints, numFixed,
                revoluteJoints, numRevolute,
                prismaticJoints, numPrismatic,
                d6Joints, numD6,
                gearJoints, numGear,
                contactMap ? *contactMap : emptyMap,
                sphericalMap ? *sphericalMap : emptyMap,
                fixedMap ? *fixedMap : emptyMap,
                revoluteMap ? *revoluteMap : emptyMap,
                prismaticMap ? *prismaticMap : emptyMap,
                d6Map ? *d6Map : emptyMap,
                gearMap ? *gearMap : emptyMap,
                dt);
          } else {
            // Fallback to original O(N*M) version
            solveBodyAllConstraints(
                bodies, numBodies, bodyIdx,
                contacts, numContacts,
                sphericalJoints, numSpherical,
                fixedJoints, numFixed,
                revoluteJoints, numRevolute,
                prismaticJoints, numPrismatic,
                d6Joints, numD6,
                dt);
          }
        }
        mStats.totalIterations++;
      }

      {
        PX_PROFILE_ZONE("AVBD.updateLambda", 0);
        updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts);

        for (physx::PxU32 j = 0; j < numSpherical; ++j) {
          updateSphericalJointMultiplier(sphericalJoints[j], bodies, numBodies, mConfig);
        }
        for (physx::PxU32 j = 0; j < numFixed; ++j) {
          updateFixedJointMultiplier(fixedJoints[j], bodies, numBodies, mConfig);
        }
        for (physx::PxU32 j = 0; j < numRevolute; ++j) {
          updateRevoluteJointMultiplier(revoluteJoints[j], bodies, numBodies, mConfig);
        }
        for (physx::PxU32 j = 0; j < numPrismatic; ++j) {
          updatePrismaticJointMultiplier(prismaticJoints[j], bodies, numBodies, mConfig);
        }
        for (physx::PxU32 j = 0; j < numD6; ++j) {
          updateD6JointMultiplier(d6Joints[j], bodies, numBodies, mConfig);
        }
        for (physx::PxU32 j = 0; j < numGear; ++j) {
          updateGearJointMultiplier(gearJoints[j], bodies, numBodies, mConfig);
        }
      }
    }
  }

  // Process motor drives for RevoluteJoints AFTER all constraint iterations
  // Motor applies torque (limited by maxForce) to accelerate toward target velocity
  // This matches TGS behavior where motor gradually accelerates bodies
  // 
  // IMPORTANT: The solver may be called multiple times per simulation step
  // (once per island). We use a global frame counter to track which frame
  // we're in and only apply motor once per frame per body.
  {
    PX_PROFILE_ZONE("AVBD.motorDrives", 0);
    
    // Get current frame from global counter (incremented in AvbdDynamicsContext::update)
    physx::PxU64 currentFrame = getAvbdMotorFrameCounter();
    
    // Static tracking for frame-based deduplication
    static physx::PxU64 lastMotorFrame = 0;
    static physx::PxU32 processedBodyFlags = 0;  // Bitmask for up to 32 bodies
    
    // Reset flags when entering a new frame
    if (currentFrame != lastMotorFrame) {
      lastMotorFrame = currentFrame;
      processedBodyFlags = 0;
    }
    
    for (physx::PxU32 j = 0; j < numRevolute; ++j) {
      AvbdRevoluteJointConstraint &joint = revoluteJoints[j];
      if (joint.motorEnabled && joint.motorMaxForce > 0.0f) {
        const physx::PxU32 idxA = joint.header.bodyIndexA;
        const physx::PxU32 idxB = joint.header.bodyIndexB;
        
        const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
        const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);
        
        if (isAStatic && isBStatic) continue;
        
        // Check if this body already had motor applied this frame
        if (idxB < 32 && (processedBodyFlags & (1u << idxB))) {
          continue;  // Skip - already processed this frame
        }
        if (idxB < 32) {
          processedBodyFlags |= (1u << idxB);  // Mark as processed
        }
        
        AvbdSolverBody &bodyB = bodies[idxB];
        
        // Get world-space joint axis
        physx::PxVec3 worldAxis = isAStatic ? joint.axisA : bodies[idxA].rotation.rotate(joint.axisA);
        worldAxis.normalize();
        
        // Compute current angular velocity around the joint axis
        // Note: In AVBD, velocity is computed from position difference (prevPosition/rotation)
        // We need to get the actual angular velocity from the body
        physx::PxQuat deltaQ = bodyB.rotation * bodyB.prevRotation.getConjugate();
        if (deltaQ.w < 0.0f) deltaQ = -deltaQ;
        physx::PxVec3 currentAngVel = physx::PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * (2.0f * invDt);
        
        // Project to joint axis to get current rotation speed around axis
        physx::PxReal currentAxisVel = currentAngVel.dot(worldAxis);
        
        // Velocity error
        physx::PxReal velocityError = joint.motorTargetVelocity - currentAxisVel;
        
        // Compute effective inertia around the joint axis
        // For a rotation around axis 'n', effective inertia = n^T * I * n
        // Since we have invInertiaWorld, effective invInertia = n^T * invI * n
        physx::PxVec3 invITimesAxis = bodyB.invInertiaWorld * worldAxis;
        physx::PxReal effectiveInvInertia = worldAxis.dot(invITimesAxis);
        
        if (effectiveInvInertia < 1e-10f) continue;  // Body is static around this axis
        
        physx::PxReal effectiveInertia = 1.0f / effectiveInvInertia;
        
        // Required torque to achieve target velocity change
        // torque = inertia * angular_acceleration = inertia * (deltaVel / dt)
        physx::PxReal requiredTorque = effectiveInertia * velocityError * invDt;
        
        // Clamp torque to maxForce (which is actually max torque for revolute joints)
        physx::PxReal clampedTorque = physx::PxClamp(requiredTorque, -joint.motorMaxForce, joint.motorMaxForce);
        
        // Angular acceleration from clamped torque
        physx::PxReal angularAccel = clampedTorque * effectiveInvInertia;
        
        // Delta angle = 0.5 * alpha * dt^2 (integration from torque to position)
        // But since we already have velocity changes, use: deltaAngle = deltaVel * dt
        physx::PxReal deltaVel = angularAccel * dt;
        physx::PxReal deltaAngle = deltaVel * dt;
        
        // Apply rotation around the joint axis
        physx::PxReal halfAngle = deltaAngle * 0.5f;
        physx::PxReal sinHalf = physx::PxSin(halfAngle);
        physx::PxReal cosHalf = physx::PxCos(halfAngle);
        physx::PxQuat deltaRot(worldAxis.x * sinHalf, worldAxis.y * sinHalf, worldAxis.z * sinHalf, cosHalf);
        
        // Apply rotation to body B
        bodyB.rotation = (deltaRot * bodyB.rotation).getNormalized();
        
        // If bodyA is also dynamic, apply opposite rotation (scaled by inertia ratio)
        if (!isAStatic) {
          AvbdSolverBody &bodyA = bodies[idxA];
          physx::PxVec3 invITimesAxisA = bodyA.invInertiaWorld * worldAxis;
          physx::PxReal effectiveInvInertiaA = worldAxis.dot(invITimesAxisA);
          
          if (effectiveInvInertiaA > 1e-10f) {
            physx::PxReal angularAccelA = clampedTorque * effectiveInvInertiaA;
            physx::PxReal deltaAngleA = angularAccelA * dt * dt;
            
            physx::PxReal halfAngleA = -deltaAngleA * 0.5f;  // Opposite direction
            physx::PxReal sinHalfA = physx::PxSin(halfAngleA);
            physx::PxReal cosHalfA = physx::PxCos(halfAngleA);
            physx::PxQuat deltaRotA(worldAxis.x * sinHalfA, worldAxis.y * sinHalfA, worldAxis.z * sinHalfA, cosHalfA);
            
            bodyA.rotation = (deltaRotA * bodyA.rotation).getNormalized();
          }
        }
      }
    }
  }

  {
    PX_PROFILE_ZONE("AVBD.updateVelocities", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f) {
        bodies[i].linearVelocity = (bodies[i].position - bodies[i].prevPosition) * invDt;
        physx::PxQuat deltaQ = bodies[i].rotation * bodies[i].prevRotation.getConjugate();
        if (deltaQ.w < 0.0f) {
          deltaQ = -deltaQ;
        }
        bodies[i].angularVelocity = physx::PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * (2.0f * invDt);
        bodies[i].angularVelocity *= mConfig.angularDamping;
      }
    }
    
    // Apply D6 joint angular damping to connected bodies
    for (physx::PxU32 j = 0; j < numD6; ++j) {
      const AvbdD6JointConstraint &joint = d6Joints[j];
      physx::PxReal maxDamping = physx::PxMax(joint.angularDamping.x, 
                                  physx::PxMax(joint.angularDamping.y, joint.angularDamping.z));
      if (maxDamping > 0.0f && joint.driveFlags != 0) {
        // Compute damping factor (normalize to reasonable range)
        // For damping=1000, we want strong damping ~0.9
        physx::PxReal dampingFactor = 1.0f / (1.0f + maxDamping * dt * 0.1f);
        dampingFactor = physx::PxMax(dampingFactor, 0.1f);  // Minimum 10% velocity retained
        
        // Apply damping to body A
        if (joint.header.bodyIndexA < numBodies && bodies[joint.header.bodyIndexA].invMass > 0.0f) {
          bodies[joint.header.bodyIndexA].angularVelocity *= dampingFactor;
        }
        // Apply damping to body B
        if (joint.header.bodyIndexB < numBodies && bodies[joint.header.bodyIndexB].invMass > 0.0f) {
          bodies[joint.header.bodyIndexB].angularVelocity *= dampingFactor;
        }
      }
    }
  }
}

//=============================================================================
// Energy Minimization Framework
//=============================================================================

physx::PxReal AvbdSolver::computeTotalEnergy(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                              AvbdContactConstraint *contacts, physx::PxU32 numContacts,
                                              const physx::PxVec3 &gravity) {
  physx::PxReal kineticEnergy = computeKineticEnergy(bodies, numBodies);
  physx::PxReal potentialEnergy = computePotentialEnergy(bodies, numBodies, gravity);
  physx::PxReal constraintEnergy = computeConstraintEnergy(contacts, numContacts, bodies, numBodies);
  return kineticEnergy + potentialEnergy + constraintEnergy;
}

physx::PxReal AvbdSolver::computeKineticEnergy(AvbdSolverBody *bodies, physx::PxU32 numBodies) {
  physx::PxReal totalEnergy = 0.0f;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f) continue;
    physx::PxReal mass = 1.0f / bodies[i].invMass;
    physx::PxReal linearKE = 0.5f * mass * bodies[i].linearVelocity.magnitudeSquared();
    physx::PxVec3 angVel = bodies[i].angularVelocity;
    physx::PxVec3 angMomentum = bodies[i].invInertiaWorld.transformTranspose(angVel);
    physx::PxReal angularKE = 0.5f * angVel.dot(angMomentum);
    totalEnergy += linearKE + angularKE;
  }
  return totalEnergy;
}

physx::PxReal AvbdSolver::computePotentialEnergy(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                                  const physx::PxVec3 &gravity) {
  physx::PxReal totalEnergy = 0.0f;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f) continue;
    physx::PxReal mass = 1.0f / bodies[i].invMass;
    physx::PxReal height = bodies[i].position.dot(gravity.getNormalized());
    totalEnergy -= mass * gravity.magnitude() * height;
  }
  return totalEnergy;
}

physx::PxReal AvbdSolver::computeConstraintEnergy(AvbdContactConstraint *contacts, physx::PxU32 numContacts,
                                                   AvbdSolverBody *bodies, physx::PxU32 numBodies) {
  physx::PxReal totalEnergy = 0.0f;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;
    AvbdSolverBody *bodyA = (bodyAIdx < numBodies) ? &bodies[bodyAIdx] : nullptr;
    AvbdSolverBody *bodyB = (bodyBIdx < numBodies) ? &bodies[bodyBIdx] : nullptr;
    physx::PxReal violation = 0.0f;
    if (bodyA && bodyB) {
      violation = computeContactViolation(contacts[c], *bodyA, *bodyB);
    } else if (bodyA) {
      AvbdSolverBody staticBody;
      staticBody.position = physx::PxVec3(0.0f);
      staticBody.rotation = physx::PxQuat(physx::PxIdentity);
      violation = computeContactViolation(contacts[c], *bodyA, staticBody);
    } else if (bodyB) {
      AvbdSolverBody staticBody;
      staticBody.position = physx::PxVec3(0.0f);
      staticBody.rotation = physx::PxQuat(physx::PxIdentity);
      violation = computeContactViolation(contacts[c], staticBody, *bodyB);
    }
    if (violation >= 0.0f && contacts[c].header.lambda <= 0.0f) continue;
    physx::PxReal rho = contacts[c].header.rho;
    physx::PxReal lambda = contacts[c].header.lambda;
    totalEnergy += 0.5f * rho * violation * violation + lambda * violation;
  }
  return totalEnergy;
}

void AvbdSolver::computeEnergyGradient(physx::PxU32 bodyIndex, AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
                                      physx::PxReal invDt2, AvbdVec6 &gradient) {
  PX_UNUSED(numBodies);
  AvbdSolverBody &body = bodies[bodyIndex];
  physx::PxReal massContrib = (body.invMass > 0.0f) ? (1.0f / body.invMass) * invDt2 : 0.0f;
  gradient.linear = (body.position - body.predictedPosition) * massContrib;
  gradient.angular = physx::PxVec3(0.0f);
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyB = contacts[c].header.bodyIndexB;
    if (bodyA != bodyIndex && bodyB != bodyIndex) continue;
    physx::PxReal violation = contacts[c].penetrationDepth;
    if (violation >= 0.0f && contacts[c].header.lambda <= 0.0f) continue;
    physx::PxReal force = contacts[c].header.rho * violation + contacts[c].header.lambda;
    physx::PxVec3 gradPos, gradRot;
    if (bodyA == bodyIndex) {
      gradPos = contacts[c].contactNormal;
      physx::PxVec3 rA = body.rotation.rotate(contacts[c].contactPointA);
      gradRot = rA.cross(contacts[c].contactNormal);
    } else {
      gradPos = -contacts[c].contactNormal;
      physx::PxVec3 rB = body.rotation.rotate(contacts[c].contactPointB);
      gradRot = -rB.cross(contacts[c].contactNormal);
    }
    gradient.linear += gradPos * force;
    gradient.angular += gradRot * force;
  }
}

bool AvbdSolver::checkEnergyConvergence(physx::PxReal oldEnergy, physx::PxReal newEnergy,
                                       physx::PxReal tolerance) const {
  physx::PxReal energyChange = physx::PxAbs(newEnergy - oldEnergy);
  physx::PxReal relativeChange = (physx::PxAbs(oldEnergy) > AvbdConstants::AVBD_LDLT_SINGULAR_THRESHOLD) ?
      energyChange / physx::PxAbs(oldEnergy) : energyChange;
  return relativeChange < tolerance;
}

physx::PxReal AvbdSolver::performLineSearch(AvbdSolverBody &body, const AvbdVec6 &direction,
                                             physx::PxReal initialStep, physx::PxReal energy,
                                             physx::PxReal c1, physx::PxReal rho) {
  PX_UNUSED(energy);
  physx::PxReal alpha = initialStep;
  physx::PxReal maxIter = 10;
  physx::PxVec3 savedPosition = body.position;
  physx::PxQuat savedRotation = body.rotation;
  for (physx::PxU32 i = 0; i < maxIter; ++i) {
    body.position += direction.linear * alpha;
    physx::PxReal angle = (direction.angular * alpha).magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      physx::PxVec3 axis = direction.angular / angle;
      physx::PxReal sinHalf = physx::PxSin(angle * 0.5f);
      physx::PxReal cosHalf = physx::PxCos(angle * 0.5f);
      physx::PxQuat deltaQ(axis.x * sinHalf, axis.y * sinHalf, axis.z * sinHalf, cosHalf);
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
    physx::PxReal positionChange = (body.position - savedPosition).magnitude();
    if (positionChange < initialStep * c1) {
      return alpha;
    }
    alpha *= rho;
    body.position = savedPosition;
    body.rotation = savedRotation;
  }
  body.position = savedPosition;
  body.rotation = savedRotation;
  return alpha;
}

//=============================================================================
// Optimized Constraint Mapping and Solving
//=============================================================================

void AvbdSolver::buildConstraintMapping(AvbdContactConstraint *contacts, 
                                        physx::PxU32 numContacts,
                                        physx::PxU32 numBodies) {
  if (!mAllocator || numContacts == 0 || numBodies == 0) {
    return;
  }
  
  mContactMap.build(numBodies, contacts, numContacts, *mAllocator);
}

void AvbdSolver::solveBodyLocalConstraintsFast(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    AvbdContactConstraint *contacts) {
  
  AvbdSolverBody &body = bodies[bodyIndex];
  
  if (body.invMass <= 0.0f) {
    return;
  }
  
  // Get only the constraints affecting this body - O(1) lookup!
  const physx::PxU32 *constraintIndices = nullptr;
  physx::PxU32 numBodyConstraints = 0;
  mContactMap.getBodyConstraints(bodyIndex, constraintIndices, numBodyConstraints);
  
  if (numBodyConstraints == 0 || constraintIndices == nullptr) {
    return;
  }
  
  // Accumulate position and rotation corrections
  physx::PxVec3 totalDeltaPos(0.0f);
  physx::PxVec3 totalDeltaTheta(0.0f);
  physx::PxReal totalWeight = 0.0f;
  
  // Only iterate over constraints that affect this body
  for (physx::PxU32 ci = 0; ci < numBodyConstraints; ++ci) {
    const physx::PxU32 c = constraintIndices[ci];
    AvbdContactConstraint &contact = contacts[c];
    
    const physx::PxU32 bodyAIdx = contact.header.bodyIndexA;
    const physx::PxU32 bodyBIdx = contact.header.bodyIndexB;
    
    // Determine which body we are
    bool isBodyA = (bodyAIdx == bodyIndex);
    
    // Get the other body
    AvbdSolverBody *otherBody = nullptr;
    if (isBodyA && bodyBIdx < numBodies) {
      otherBody = &bodies[bodyBIdx];
    } else if (!isBodyA && bodyAIdx < numBodies) {
      otherBody = &bodies[bodyAIdx];
    }
    
    // Compute world positions of contact points
    physx::PxVec3 worldPosA, worldPosB;
    physx::PxVec3 r;  // Contact arm for this body
    
    if (isBodyA) {
      r = body.rotation.rotate(contact.contactPointA);
      worldPosA = body.position + r;
      if (otherBody) {
        worldPosB = otherBody->position + otherBody->rotation.rotate(contact.contactPointB);
      } else {
        worldPosB = contact.contactPointB;
      }
    } else {
      r = body.rotation.rotate(contact.contactPointB);
      if (otherBody) {
        worldPosA = otherBody->position + otherBody->rotation.rotate(contact.contactPointA);
      } else {
        worldPosA = contact.contactPointA;
      }
      worldPosB = body.position + r;
    }
    
    // Compute separation (negative = penetration)
    const physx::PxVec3 &normal = contact.contactNormal;
    physx::PxReal separation = (worldPosA - worldPosB).dot(normal) + contact.penetrationDepth;
    
    // Only correct if penetrating
    if (separation >= 0.0f) {
      continue;
    }
    
    // Compute generalized inverse mass for normal direction
    physx::PxVec3 rCrossN = r.cross(normal);
    physx::PxReal w = body.invMass + rCrossN.dot(body.invInertiaWorld * rCrossN);
    
    // Add other body's contribution if dynamic
    physx::PxVec3 rOther(0.0f);
    if (otherBody && otherBody->invMass > 0.0f) {
      rOther = isBodyA ?
          otherBody->rotation.rotate(contact.contactPointB) :
          otherBody->rotation.rotate(contact.contactPointA);
      physx::PxVec3 rOtherCrossN = rOther.cross(normal);
      w += otherBody->invMass + rOtherCrossN.dot(otherBody->invInertiaWorld * rOtherCrossN);
    }
    
    if (w <= 1e-6f) {
      continue;
    }
    
    // Compute normal correction magnitude
    physx::PxReal normalCorrectionMag = -separation / w;
    normalCorrectionMag *= mConfig.baumgarte;
    
    // Direction sign for this body
    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
    
    // Normal position and rotation corrections
    physx::PxVec3 deltaPos = normal * (normalCorrectionMag * body.invMass * sign);
    physx::PxVec3 deltaTheta = (body.invInertiaWorld * rCrossN) * (normalCorrectionMag * sign);
    
    // Weight by constraint stiffness (rho)
    physx::PxReal weight = contact.header.rho;
    totalDeltaPos += deltaPos * weight;
    totalDeltaTheta += deltaTheta * weight;
    totalWeight += weight;
  }
  
  // Apply averaged corrections
  if (totalWeight > 0.0f) {
    physx::PxReal invWeight = 1.0f / totalWeight;
    
    // Apply position correction
    body.position += totalDeltaPos * invWeight;
    
    // Apply rotation correction using exponential map
    physx::PxVec3 avgDeltaTheta = totalDeltaTheta * invWeight;
    physx::PxReal angle = avgDeltaTheta.magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      angle = physx::PxMin(angle, 0.1f); // Clamp for stability
      physx::PxVec3 axis = avgDeltaTheta.getNormalized();
      physx::PxReal halfAngle = angle * 0.5f;
      physx::PxQuat deltaQ(axis.x * physx::PxSin(halfAngle),
                            axis.y * physx::PxSin(halfAngle),
                            axis.z * physx::PxSin(halfAngle),
                            physx::PxCos(halfAngle));
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
  }
}

//=============================================================================
// Optimized Constraint Mapping Functions
//=============================================================================

void AvbdSolver::buildAllConstraintMappings(
    physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
    AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
    AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
    AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6) {
  
  if (!mAllocator || numBodies == 0) return;
  
  // IMPORTANT: Release all old mappings first to avoid stale data
  // This prevents accessing old indices when constraint counts change between frames
  mContactMap.release(*mAllocator);
  mSphericalMap.release(*mAllocator);
  mFixedMap.release(*mAllocator);
  mRevoluteMap.release(*mAllocator);
  mPrismaticMap.release(*mAllocator);
  mD6Map.release(*mAllocator);
  
  // Build contact mapping
  if (numContacts > 0 && contacts) {
    mContactMap.build(numBodies, contacts, numContacts, *mAllocator);
  }
  
  // Build joint mappings
  if (numSpherical > 0 && sphericalJoints) {
    mSphericalMap.build(numBodies, sphericalJoints, numSpherical, *mAllocator);
  }
  if (numFixed > 0 && fixedJoints) {
    mFixedMap.build(numBodies, fixedJoints, numFixed, *mAllocator);
  }
  if (numRevolute > 0 && revoluteJoints) {
    mRevoluteMap.build(numBodies, revoluteJoints, numRevolute, *mAllocator);
  }
  if (numPrismatic > 0 && prismaticJoints) {
    mPrismaticMap.build(numBodies, prismaticJoints, numPrismatic, *mAllocator);
  }
  if (numD6 > 0 && d6Joints) {
    mD6Map.build(numBodies, d6Joints, numD6, *mAllocator);
  }
}

void AvbdSolver::solveBodyAllConstraintsFast(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 bodyIndex,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
    AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
    AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
    AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    const AvbdBodyConstraintMap &contactMap,
    const AvbdBodyConstraintMap &sphericalMap,
    const AvbdBodyConstraintMap &fixedMap,
    const AvbdBodyConstraintMap &revoluteMap,
    const AvbdBodyConstraintMap &prismaticMap,
    const AvbdBodyConstraintMap &d6Map,
    const AvbdBodyConstraintMap &gearMap,
    physx::PxReal dt) {
  
  PX_UNUSED(dt);
  
  AvbdSolverBody &body = bodies[bodyIndex];
  
  if (body.invMass <= 0.0f) {
    return;
  }
  
  // Accumulate corrections from all constraint types
  physx::PxVec3 totalDeltaPos(0.0f);
  physx::PxVec3 totalDeltaTheta(0.0f);
  physx::PxU32 numActiveConstraints = 0;
  
  // Process contact constraints - O(contacts connected to this body)
  if (contacts && numContacts > 0) {
    const physx::PxU32 *contactIndices = nullptr;
    physx::PxU32 numBodyContacts = 0;
    contactMap.getBodyConstraints(bodyIndex, contactIndices, numBodyContacts);
    
    for (physx::PxU32 i = 0; i < numBodyContacts; ++i) {
      physx::PxU32 c = contactIndices[i];
      // Bounds check to prevent out-of-range access
      if (c >= numContacts) continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeContactCorrection(contacts[c], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }
  
  // Process spherical joint constraints
  if (sphericalJoints && numSpherical > 0) {
    const physx::PxU32 *sphericalIndices = nullptr;
    physx::PxU32 numBodySpherical = 0;
    sphericalMap.getBodyConstraints(bodyIndex, sphericalIndices, numBodySpherical);
    
    for (physx::PxU32 i = 0; i < numBodySpherical; ++i) {
      physx::PxU32 j = sphericalIndices[i];
      if (j >= numSpherical) continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeSphericalJointCorrection(sphericalJoints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }
  
  // Process fixed joint constraints
  if (fixedJoints && numFixed > 0) {
    const physx::PxU32 *fixedIndices = nullptr;
    physx::PxU32 numBodyFixed = 0;
    fixedMap.getBodyConstraints(bodyIndex, fixedIndices, numBodyFixed);
    
    for (physx::PxU32 i = 0; i < numBodyFixed; ++i) {
      physx::PxU32 j = fixedIndices[i];
      if (j >= numFixed) continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeFixedJointCorrection(fixedJoints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }
  
  // Process revolute joint constraints
  if (revoluteJoints && numRevolute > 0) {
    const physx::PxU32 *revoluteIndices = nullptr;
    physx::PxU32 numBodyRevolute = 0;
    revoluteMap.getBodyConstraints(bodyIndex, revoluteIndices, numBodyRevolute);
    
    for (physx::PxU32 i = 0; i < numBodyRevolute; ++i) {
      physx::PxU32 j = revoluteIndices[i];
      if (j >= numRevolute) continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeRevoluteJointCorrection(revoluteJoints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }
  
  // Process prismatic joint constraints
  if (prismaticJoints && numPrismatic > 0) {
    const physx::PxU32 *prismaticIndices = nullptr;
    physx::PxU32 numBodyPrismatic = 0;
    prismaticMap.getBodyConstraints(bodyIndex, prismaticIndices, numBodyPrismatic);
    
    for (physx::PxU32 i = 0; i < numBodyPrismatic; ++i) {
      physx::PxU32 j = prismaticIndices[i];
      if (j >= numPrismatic) continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computePrismaticJointCorrection(prismaticJoints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }
  
  // Process D6 joint constraints
  if (d6Joints && numD6 > 0) {
    const physx::PxU32 *d6Indices = nullptr;
    physx::PxU32 numBodyD6 = 0;
    d6Map.getBodyConstraints(bodyIndex, d6Indices, numBodyD6);
    
    for (physx::PxU32 i = 0; i < numBodyD6; ++i) {
      physx::PxU32 j = d6Indices[i];
      if (j >= numD6) continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeD6JointCorrection(d6Joints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }
  
  // Process gear joint constraints
  if (gearJoints && numGear > 0) {
    const physx::PxU32 *gearIndices = nullptr;
    physx::PxU32 numBodyGear = 0;
    gearMap.getBodyConstraints(bodyIndex, gearIndices, numBodyGear);
    
    for (physx::PxU32 i = 0; i < numBodyGear; ++i) {
      physx::PxU32 j = gearIndices[i];
      if (j >= numGear) continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeGearJointCorrection(gearJoints[j], bodies, numBodies, bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }
  
  // Apply averaged corrections
  if (numActiveConstraints > 0) {
    physx::PxReal invCount = 1.0f / static_cast<physx::PxReal>(numActiveConstraints);
    
    // Apply position correction with Baumgarte stabilization
    body.position += totalDeltaPos * invCount * mConfig.baumgarte;
    
    // Apply rotation correction
    physx::PxVec3 avgDeltaTheta = totalDeltaTheta * invCount * mConfig.baumgarte;
    physx::PxReal angle = avgDeltaTheta.magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      angle = physx::PxMin(angle, 0.1f);
      physx::PxVec3 axis = avgDeltaTheta.getNormalized();
      physx::PxReal halfAngle = angle * 0.5f;
      physx::PxQuat deltaQ(axis.x * physx::PxSin(halfAngle),
                            axis.y * physx::PxSin(halfAngle),
                            axis.z * physx::PxSin(halfAngle),
                            physx::PxCos(halfAngle));
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
  }
}

} // namespace Dy
} // namespace physx
