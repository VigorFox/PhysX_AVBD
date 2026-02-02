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

#include <algorithm>

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
  if (!mInitialized || numBodies == 0) {
    return;
  }

  mStats.reset();

  physx::PxReal invDt = 1.0f / dt;

  // Stage 1: Prediction
  computePrediction(bodies, numBodies, dt, gravity);

  // Stage 2: Graph Coloring (skip if pre-computed coloring is provided)
  if (mConfig.enableParallelization && numColors == 0) {
    computeGraphColoring(bodies, numBodies, contacts, numContacts);
  }

  // Initialize positions to predicted values, save current as previous
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    // Save current state before prediction for velocity calculation
    bodies[i].prevPosition = bodies[i].position;
    bodies[i].prevRotation = bodies[i].rotation;
    // Start from predicted position
    bodies[i].position = bodies[i].predictedPosition;
    bodies[i].rotation = bodies[i].predictedRotation;
  }

  // Outer loop: Augmented Lagrangian iterations
  if (mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eSORT_CONSTRAINTS) &&
      numContacts > 1) {
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

  for (physx::PxU32 outerIter = 0; outerIter < mConfig.outerIterations;
       ++outerIter) {

    // Inner loop: Block descent iterations
    for (physx::PxU32 innerIter = 0; innerIter < mConfig.innerIterations;
         ++innerIter) {
      // Pass pre-computed coloring if available
      blockDescentIteration(bodies, numBodies, contacts, numContacts, dt,
                            colorBatches, numColors);
      mStats.totalIterations++;
    }

    // Update Lagrangian multipliers
    updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts);
  }

  // Stage 5: Update velocities from position/rotation change
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
    for (physx::PxU32 idx = 0; idx < numBodies; ++idx) {
      const physx::PxU32 i = orderPtr ? orderPtr[idx] : idx;
      // Skip static bodies
      if (bodies[i].invMass <= 0.0f) {
        continue;
      }

      // Solve local system for this body using Gauss-Seidel on all its constraints
      solveBodyLocalConstraints(bodies, numBodies, i, contacts, numContacts);
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
  
  if (errorMag < AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    return false;
  }
  
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
  
  if (w <= 1e-6f) {
    return false;
  }
  
  // Compute correction
  physx::PxReal correctionMag = -errorMag / w;
  physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
  
  deltaPos = direction * (correctionMag * body.invMass * sign);
  deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
  
  return true;
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
  
  return (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON);
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
  
  if (joint.linearMotion == 0) {
    physx::PxReal posErrorMag = posError.magnitude();
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
      return true;
    }
  }
  
  return false;
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
    const physx::PxVec3 &gravity, AvbdColorBatch *colorBatches,
    physx::PxU32 numColors) {

  PX_UNUSED(colorBatches);
  PX_UNUSED(numColors);

  if (!mInitialized || numBodies == 0) {
    return;
  }

  mStats.reset();
  mStats.numJoints = numSpherical + numFixed + numRevolute + numPrismatic + numD6;

  physx::PxReal invDt = 1.0f / dt;

  computePrediction(bodies, numBodies, dt, gravity);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].prevPosition = bodies[i].position;
    bodies[i].prevRotation = bodies[i].rotation;
    bodies[i].position = bodies[i].predictedPosition;
    bodies[i].rotation = bodies[i].predictedRotation;
  }

  for (physx::PxU32 outerIter = 0; outerIter < mConfig.outerIterations; ++outerIter) {
    for (physx::PxU32 innerIter = 0; innerIter < mConfig.innerIterations; ++innerIter) {
      for (physx::PxU32 bodyIdx = 0; bodyIdx < numBodies; ++bodyIdx) {
        if (bodies[bodyIdx].invMass <= 0.0f) {
          continue;
        }
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
      mStats.totalIterations++;
    }

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
  }

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

} // namespace Dy
} // namespace physx
