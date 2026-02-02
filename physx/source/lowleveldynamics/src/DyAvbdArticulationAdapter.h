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

#ifndef DY_AVBD_ARTICULATION_ADAPTER_H
#define DY_AVBD_ARTICULATION_ADAPTER_H

#include "DyAvbdTypes.h"
#include "DyAvbdSolver.h"
#include "DyAvbdConstraint.h"
#include "DyAvbdSolverBody.h"
#include "foundation/PxAllocator.h"

namespace physx {

// Forward declarations
namespace Dy {
class FeatherstoneArticulation;
class ArticulationData;
struct ArticulationLink;
class ArticulationJointCoreData;
} // namespace Dy

namespace Dy {

/**
 * @brief Articulation link data for AVBD solver
 * 
 * Stores the mapping between PhysX articulation links and AVBD solver bodies.
 */
struct AvbdArticulationLinkData {
  PxU32 avbdBodyIndex;      //!< Index in AVBD solver body array
  PxU32 parentLinkIndex;     //!< Parent link index (0xFFFFFFFF for root)
  PxU32 numChildLinks;       //!< Number of child links
  PxU32 *childLinkIndices;   //!< Array of child link indices
  PxVec3 localAnchor;        //!< Joint anchor in local body frame
  PxVec3 localAxis;          //!< Joint axis in local body frame
  PxReal jointPosition;      //!< Current joint position
  PxReal jointVelocity;      //!< Current joint velocity
  PxReal jointTarget;        //!< Target joint position (for drive)
  PxReal jointTargetVelocity; //!< Target joint velocity (for drive)
  PxReal driveStiffness;     //!< Drive stiffness (PD parameter)
  PxReal driveDamping;       //!< Drive damping (PD parameter)
  PxReal maxForce;           //!< Maximum drive force
  bool driveEnabled;         //!< Whether drive is enabled
  bool limitEnabled;         //!< Whether limit is enabled
  PxReal limitLower;         //!< Lower limit
  PxReal limitUpper;         //!< Upper limit
};

/**
 * @brief AVBD Articulation Adapter
 * 
 * This adapter bridges PhysX Articulation (Featherstone) with the AVBD solver.
 * It implements a hybrid architecture where:
 * - Forward dynamics uses AVBD solver for better constraint handling
 * - Inverse dynamics uses native Featherstone implementation
 * - State is synchronized between the two representations
 * 
 * Architecture:
 * - User API: PxArticulationRC
 * - Adapter: AvbdArticulationAdapter
 *   - AVBD representation (forward dynamics)
 *     - AvbdSolverBody[] (rigid bodies)
 *     - AvbdJointConstraint[] (joint constraints)
 *     - AvbdSolver (constraint solver)
 *   - Featherstone representation (inverse dynamics)
 *     - FeatherstoneArticulation (native)
 *     - ArticulationData (state)
 * - State synchronization:
 *   - syncDriveTargetsToAvbd(): AVBD <- Featherstone
 *   - syncStateToArticulation(): Featherstone <- AVBD
 */
class AvbdArticulationAdapter {
public:
  AvbdArticulationAdapter();
  ~AvbdArticulationAdapter();

  //-------------------------------------------------------------------------
  // Initialization
  //-------------------------------------------------------------------------

  /**
   * @brief Initialize adapter from PhysX articulation
   * @param articulation Source Featherstone articulation
   * @param allocator Memory allocator
   * @return true if initialization succeeded
   */
  bool initialize(FeatherstoneArticulation *articulation,
                  PxAllocatorCallback &allocator);

  /**
   * @brief Release all allocated resources
   */
  void release();

  //-------------------------------------------------------------------------
  // State Synchronization
  //-------------------------------------------------------------------------

  /**
   * @brief Sync drive targets from Featherstone to AVBD
   * 
   * Copies joint drive targets (position/velocity) from the native
   * Featherstone articulation to the AVBD representation.
   * 
   * This is called before each AVBD solve step.
   */
  void syncDriveTargetsToAvbd();

  /**
   * @brief Sync state from AVBD to Featherstone
   * 
   * Copies solved positions and velocities from AVBD back to the
   * native Featherstone articulation.
   * 
   * This is called after each AVBD solve step.
   */
  void syncStateToArticulation();

  //-------------------------------------------------------------------------
  // Forward Dynamics (AVBD)
  //-------------------------------------------------------------------------

  /**
   * @brief Solve forward dynamics using AVBD
   * @param dt Time step
   * @param gravity Gravity vector
   * @param solver AVBD solver instance
   * @param bodies AVBD solver bodies array
   * @param numBodies Number of bodies
   * @param joints AVBD joint constraints array
   * @param numJoints Number of joints
   */
  void solveForwardDynamics(PxReal dt, const PxVec3 &gravity,
                            AvbdSolver &solver, AvbdSolverBody *bodies,
                            PxU32 numBodies, AvbdSphericalJointConstraint *joints,
                            PxU32 numJoints);

  //-------------------------------------------------------------------------
  // Inverse Dynamics (Featherstone)
  //-------------------------------------------------------------------------

  /**
   * @brief Compute mass matrix using native Featherstone
   * @param massMatrix Output mass matrix (DOF x DOF)
   * @param jointPositions Joint positions
   * @param jointVelocities Joint velocities
   * @return true if computation succeeded
   */
  bool computeMassMatrix(PxReal *massMatrix, const PxReal *jointPositions,
                         const PxReal *jointVelocities);

  /**
   * @brief Compute joint forces for given accelerations
   * @param jointAccelerations Input joint accelerations
   * @param jointForces Output joint forces
   * @return true if computation succeeded
   */
  bool computeJointForces(const PxReal *jointAccelerations,
                          PxReal *jointForces);

  /**
   * @brief Compute generalized forces for given joint positions/velocities
   * @param jointPositions Joint positions
   * @param jointVelocities Joint velocities
   * @param jointForces Output joint forces
   * @return true if computation succeeded
   */
  bool computeGeneralizedForces(const PxReal *jointPositions,
                                const PxReal *jointVelocities,
                                PxReal *jointForces);

  //-------------------------------------------------------------------------
  // Accessors
  //-------------------------------------------------------------------------

  /**
   * @brief Get number of links in articulation
   */
  PxU32 getLinkCount() const { return mNumLinks; }

  /**
   * @brief Get number of degrees of freedom
   */
  PxU32 getDofCount() const { return mDofCount; }

  /**
   * @brief Get link data for a specific link
   */
  const AvbdArticulationLinkData &getLinkData(PxU32 linkIndex) const {
    return mLinkData[linkIndex];
  }

  /**
   * @brief Get AVBD body index for a link
   */
  PxU32 getAvbdBodyIndex(PxU32 linkIndex) const {
    return mLinkData[linkIndex].avbdBodyIndex;
  }

  /**
   * @brief Check if adapter is initialized
   */
  bool isInitialized() const { return mInitialized; }

private:
  //-------------------------------------------------------------------------
  // Internal Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Build AVBD bodies from articulation links
   */
  void buildAvbdBodies(AvbdSolverBody *bodies, PxU32 numBodies);

  /**
   * @brief Build AVBD joint constraints from articulation joints
   */
  void buildAvbdJoints(AvbdSphericalJointConstraint *joints, PxU32 numJoints);

  /**
   * @brief Convert joint type to AVBD constraint type
   */
  PxU32 getJointType(const ArticulationJointCoreData &jointData);

  /**
   * @brief Extract joint parameters from articulation
   */
  void extractJointParameters(PxU32 linkIndex,
                              AvbdArticulationLinkData &linkData);

  /**
   * @brief Apply joint drive forces in AVBD representation
   */
  void applyJointDrives(AvbdSolverBody *bodies, PxU32 numBodies, PxReal dt);

  /**
   * @brief Apply joint limits in AVBD representation
   */
  void applyJointLimits(AvbdSolverBody *bodies, PxU32 numBodies);

  //-------------------------------------------------------------------------
  // Member Variables
  //-------------------------------------------------------------------------

  FeatherstoneArticulation *mArticulation;  //!< Native Featherstone articulation
  ArticulationData *mArticulationData;      //!< Articulation data pointer

  AvbdArticulationLinkData *mLinkData;      //!< Per-link data
  PxU32 mNumLinks;                          //!< Number of links
  PxU32 mDofCount;                          //!< Number of degrees of freedom

  PxAllocatorCallback *mAllocator;          //!< Memory allocator
  bool mInitialized;                        //!< Initialization flag
};

} // namespace Dy

} // namespace physx

#endif // DY_AVBD_ARTICULATION_ADAPTER_H
