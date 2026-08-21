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
#ifndef DY_AVBD_SOFT_BODY_COMPONENT_H
#define DY_AVBD_SOFT_BODY_COMPONENT_H

// =============================================================================
// Internal AVBD Soft Body / Cloth -- energy-based deformable component
//
// Elastic energies use position-level VBD blocks. Contacts use a persistent
// augmented-Lagrangian normal/tangent state; pins currently use adaptive
// penalty state. The authoritative global schedule is serial vertex-block
// nonlinear Gauss-Seidel.
//
// Elastic forces: StVK (triangles), Neo-Hookean (tetrahedra), dihedral bending
// Constraints: contact (ground/soft-soft/soft-rigid), kinematic pins
//
// This Scene-external component remains private until a real CPU deformable
// actor/factory/buffer contract exists. Validation Snippets may consume it
// through an explicit private include path; it is not a public PhysX API.
//
// References: VBD (SIGGRAPH 2024), AVBD (SIGGRAPH 2025)
// =============================================================================

#include "foundation/PxAllocator.h"
#include "foundation/PxAlignedMalloc.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"
#include "foundation/PxBounds3.h"
#include "foundation/PxMat33.h"
#include "foundation/PxMathUtils.h"
#include "foundation/PxQuat.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxSort.h"
#include "foundation/PxTime.h"
#include "foundation/PxVec3.h"
#include "PxMaterial.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "avbd/core/DyAvbdConstraint.h"
#include "avbd/backend/cpu/DyAvbdCpuIsa.h"
#include "avbd/contact/DyAvbdContact.h"
#include "avbd/contact/DyAvbdContactBounds.h"
#include "avbd/contact/DyAvbdContactFeature.h"
#include "avbd/contact/DyAvbdContactFeatureGeometry.h"
#include "avbd/contact/DyAvbdContactRigidPrimitives.h"
#include "avbd/contact/DyAvbdContactStats.h"
#include "avbd/contact/DyAvbdContactGeometry.h"
#include "avbd/contact/DyAvbdContactGeometryQueries.h"
#include "avbd/contact/DyAvbdContactMaterial.h"
#include "avbd/contact/DyAvbdContactDetection.h"
#include "avbd/contact/DyAvbdContactEpoch.h"
#include "avbd/contact/DyAvbdContactPlane.h"
#include "avbd/contact/DyAvbdContactMeshQueries.h"
#include "avbd/contact/DyAvbdContactRigidSoft.h"
#include "avbd/contact/DyAvbdContactSelf.h"
#include "avbd/contact/DyAvbdContactSoftPair.h"
#include "avbd/contact/DyAvbdContactVelocityEpoch.h"
#include "avbd/contact/DyAvbdContactTriangleSurfaceDiagnostics.h"
#include "avbd/contact/DyAvbdContactWorkspace.h"
#include "avbd/contact/DyAvbdDetectionPlan.h"
#include "avbd/contact/DyAvbdSoftContactWorkspace.h"
#include "avbd/contact/DyAvbdSoftContactPrep.h"
#include "avbd/contact/DyAvbdSelfCollisionTopology.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/ogc/DyAvbdOgcParameters.h"
#include "avbd/solver/soft/DyAvbdSoftBodyGeometry.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"
#include "avbd/solver/soft/DyAvbdSoftBodyData.h"
#include "avbd/solver/soft/DyAvbdSoftBodyCompiledData.h"
#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"
#include "avbd/solver/soft/DyAvbdSoftContactGeometry.h"
#include "avbd/solver/soft/DyAvbdSoftBodyCreation.h"
#include "avbd/solver/soft/DyAvbdSoftBodyStep.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPolicy.h"
#include "avbd/solver/soft/DyAvbdSoftBodyMechanics.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPrimal.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPrimalPolicy.h"
#include "avbd/solver/soft/DyAvbdSoftBodyScheduling.h"
#include "avbd/solver/soft/DyAvbdSoftBodyFinalization.h"
#include "avbd/solver/soft/DyAvbdSoftBodyWorkspace.h"
#include "avbd/solver/soft/DyAvbdSoftBodyEpochSafety.h"
#include "avbd/diagnostics/DyAvbdSoftBodyDiagnostics.h"

namespace physx
{
namespace Dy
{

// Private bridge supplied by PhysX core for component headers that are also
// compiled by validation Snippets.  Deliberately undecorated here: static
// PhysX sub-libraries resolve it locally, while the Snippet import library
// resolves the exported C symbol.
extern "C" AvbdCpuIsaCorotationalTetPacket8Fn PX_CALL_CONV
PxAvbdCpuIsaCorotationalTetPacket8FunctionInternal();
extern "C" AvbdCpuIsaNeoHookeanTetPacket8Fn PX_CALL_CONV
PxAvbdCpuIsaNeoHookeanTetPacket8FunctionInternal();

// PxMat33 helper utilities (column-major <-> element access)
// =============================================================================

#include "avbd/solver/soft/DyAvbdSoftBodyMechanics.inl"

#include "avbd/solver/soft/DyAvbdSoftBodyContactSolve.inl"

// =============================================================================
#include "avbd/solver/soft/DyAvbdSoftBodyPrimalKernel.inl"

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_COMPONENT_H
