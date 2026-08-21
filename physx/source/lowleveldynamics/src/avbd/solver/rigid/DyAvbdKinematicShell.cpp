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

#include "avbd/solver/rigid/DyAvbdKinematicShell.h"
#include "avbd/core/DyAvbdConstraint.h"
#include "avbd/solver/rigid/DyAvbdSolverBody.h"

namespace physx {
namespace Dy {

bool AvbdKinematicShell::sActive = false;
PxU32 AvbdKinematicShell::sGridSize = 0;
PxReal AvbdKinematicShell::sGridStep = 0.0f;
PxVec3 AvbdKinematicShell::sWorld[AvbdKinematicShell::kMaxVerts];
PxVec3 AvbdKinematicShell::sPrevWorld[AvbdKinematicShell::kMaxVerts];

bool AvbdKinematicShell::applyShellNormalAndPrev(
    AvbdContactConstraint &constraint, AvbdSolverBody *bodyA,
    AvbdSolverBody *bodyB, PxReal /*restDist*/, PxVec3 &tangent0,
    PxVec3 &tangent1) {
  if (!isActive())
    return false;

  AvbdSolverBody *dynBody = nullptr;
  if (bodyA && bodyA->invMass > 0.0f)
    dynBody = bodyA;
  else if (bodyB && bodyB->invMass > 0.0f)
    dynBody = bodyB;
  if (!dynBody)
    return false;

  const PxVec3 dynCenter = dynBody->position;
  const PxVec3 shellN =
      outwardNormal(dynCenter.x, dynCenter.z, dynCenter);
  // Do NOT rewrite NP contact points here: overriding static anchors caused
  // sphere launch / lateral explosions (Entry 129, headless 2026-07-13).
  constraint.contactNormal = shellN;
  // Preserve the authoritative NP anchor and transport only its mesh motion
  // back to t-dt. Sampling current/previous grid height at the NP anchor x/z
  // keeps the material feature stable; applying the delta (rather than
  // replacing the anchor with an absolute grid sample) also preserves any
  // narrow-phase contact offset. A body-centre sample is not equivalent here:
  // on a sloped surface it aliases spatial slope into fictitious velocity.
  const PxVec3 staticNow =
      bodyA == dynBody ? constraint.contactPointB : constraint.contactPointA;
  const PxReal currentSurfaceY =
      sampleSurfaceY(staticNow.x, staticNow.z);
  const PxReal previousSurfaceY =
      sampleSurfacePrevY(staticNow.x, staticNow.z);
  constraint.staticPrevWorldPoint =
      staticNow + PxVec3(0.0f, previousSurfaceY - currentSurfaceY, 0.0f);
  constraint.header.flags |= AvbdContactConstraintFlags::eKINEMATIC_SHELL_ANCHOR;

  if (PxAbs(shellN.y) > 0.9f)
    tangent0 = shellN.cross(PxVec3(1, 0, 0)).getNormalized();
  else
    tangent0 = shellN.cross(PxVec3(0, 1, 0)).getNormalized();
  tangent1 = shellN.cross(tangent0);
  constraint.tangent0 = tangent0;
  constraint.tangent1 = tangent1;
  (void)bodyA;
  (void)bodyB;
  return true;
}

} // namespace Dy
} // namespace physx
