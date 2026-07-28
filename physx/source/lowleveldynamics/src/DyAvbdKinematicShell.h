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

#ifndef DY_AVBD_KINEMATIC_SHELL_H
#define DY_AVBD_KINEMATIC_SHELL_H

#include "foundation/PxMath.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"
#include <cstdlib>

namespace physx {
namespace Dy {

struct AvbdContactConstraint;
struct AvbdSolverBody;

/** SnippetDeformableMesh grid history used to align authoritative NP rows. */
class AvbdKinematicShell {
public:
  static constexpr PxU32 kMaxVerts = 64;

  static bool isEnabled() {
    const char *env = std::getenv("AVBD_KINEMATIC_SHELL");
    return !env || env[0] == '\0' || env[0] != '0';
  }

  static bool isActive() { return sActive && isEnabled(); }

  /**
   * Compatibility no-op. The NP-backed synthesized box-corner shell was
   * retired in P3G; callers may continue to request it while validating that
   * no direct-shell rows are emitted.
   */
  static void setBoxCornerShellEnabled(bool enabled) {
    PX_UNUSED(enabled);
  }

  static void reset() {
    sActive = false;
    sGridSize = 0;
    sGridStep = 0.0f;
  }

  static void updateFromMeshGrid(const PxVec3 *localVerts, PxU32 gridSize,
                               PxReal gridStep, const PxTransform &actorPose) {
    if (!localVerts || gridSize < 2 || gridSize > kMaxVerts / gridSize ||
        !PxIsFinite(gridStep) || gridStep <= 0.0f)
      return;
    sGridSize = gridSize;
    sGridStep = gridStep;
    const PxU32 n = gridSize * gridSize;
    const bool wasActive = sActive;
    for (PxU32 i = 0; i < n; ++i) {
      const PxVec3 world = actorPose.transform(localVerts[i]);
      sPrevWorld[i] = wasActive ? sWorld[i] : world;
      sWorld[i] = world;
    }
    sActive = true;
  }

  static PxReal sampleSurfaceY(PxReal worldX, PxReal worldZ) {
    if (!sActive)
      return 0.0f;
    const PxReal gx = PxClamp((worldX + 400.0f) / sGridStep, 0.0f,
                              PxReal(sGridSize - 1u));
    const PxReal gz = PxClamp((worldZ + 400.0f) / sGridStep, 0.0f,
                              PxReal(sGridSize - 1u));
    const PxU32 b0 = PxMin(PxU32(gx), sGridSize - 2u);
    const PxU32 a0 = PxMin(PxU32(gz), sGridSize - 2u);
    const PxReal tx = gx - PxReal(b0);
    const PxReal tz = gz - PxReal(a0);
    const auto yAt = [](PxU32 a, PxU32 b) -> PxReal {
      return sWorld[a * sGridSize + b].y;
    };
    const PxReal y0 = yAt(a0, b0) * (1.0f - tx) + yAt(a0, b0 + 1) * tx;
    const PxReal y1 = yAt(a0 + 1, b0) * (1.0f - tx) + yAt(a0 + 1, b0 + 1) * tx;
    return y0 * (1.0f - tz) + y1 * tz;
  }

  static PxReal sampleSurfacePrevY(PxReal worldX, PxReal worldZ) {
    if (!sActive)
      return sampleSurfaceY(worldX, worldZ);
    const PxReal gx = PxClamp((worldX + 400.0f) / sGridStep, 0.0f,
                              PxReal(sGridSize - 1u));
    const PxReal gz = PxClamp((worldZ + 400.0f) / sGridStep, 0.0f,
                              PxReal(sGridSize - 1u));
    const PxU32 b0 = PxMin(PxU32(gx), sGridSize - 2u);
    const PxU32 a0 = PxMin(PxU32(gz), sGridSize - 2u);
    const PxReal tx = gx - PxReal(b0);
    const PxReal tz = gz - PxReal(a0);
    const auto yAt = [](PxU32 a, PxU32 b) -> PxReal {
      return sPrevWorld[a * sGridSize + b].y;
    };
    const PxReal y0 = yAt(a0, b0) * (1.0f - tx) + yAt(a0, b0 + 1) * tx;
    const PxReal y1 = yAt(a0 + 1, b0) * (1.0f - tx) + yAt(a0 + 1, b0 + 1) * tx;
    return y0 * (1.0f - tz) + y1 * tz;
  }

  static PxVec3 outwardNormal(PxReal worldX, PxReal worldZ,
                              const PxVec3 &dynamicHint) {
    if (!sActive)
      return PxVec3(0.0f, 1.0f, 0.0f);
    const PxReal gx = PxClamp((worldX + 400.0f) / sGridStep, 0.0f,
                              PxReal(sGridSize - 1u));
    const PxReal gz = PxClamp((worldZ + 400.0f) / sGridStep, 0.0f,
                              PxReal(sGridSize - 1u));
    const PxU32 b0 = PxMin(PxU32(gx), sGridSize - 2u);
    const PxU32 a0 = PxMin(PxU32(gz), sGridSize - 2u);
    const PxVec3 &v00 = sWorld[a0 * sGridSize + b0];
    const PxVec3 &v01 = sWorld[a0 * sGridSize + b0 + 1];
    const PxVec3 &v10 = sWorld[(a0 + 1) * sGridSize + b0];
    const PxVec3 dx = v01 - v00;
    const PxVec3 dz = v10 - v00;
    PxVec3 n = dz.cross(dx);
    const PxReal len = n.magnitude();
    if (len > 1e-6f)
      n *= (1.0f / len);
    else
      n = PxVec3(0.0f, 1.0f, 0.0f);
    // The shell mirrors a one-sided triangle-mesh ground.  Its outward side is
    // fixed by the mesh winding (dz x dx), not by whichever side currently
    // contains the dynamic body.  Flipping toward dynamicHint turns the normal
    // downward as soon as a body crosses the heaving surface, so the next
    // contact correction drives it farther through the ground.
    PX_UNUSED(dynamicHint);
    return n;
  }

  static PxVec3 bilinearSurfacePoint(PxReal worldX, PxReal worldZ) {
    return PxVec3(worldX, sampleSurfaceY(worldX, worldZ), worldZ);
  }

  static PxVec3 bilinearSurfacePointPrev(PxReal worldX, PxReal worldZ) {
    return PxVec3(worldX, sampleSurfacePrevY(worldX, worldZ), worldZ);
  }

  /** Publish grid normal/history metadata while preserving NP contact points. */
  static bool applyShellNormalAndPrev(
      struct AvbdContactConstraint &constraint, struct AvbdSolverBody *bodyA,
      struct AvbdSolverBody *bodyB, PxReal restDist, PxVec3 &tangent0,
      PxVec3 &tangent1);

private:
  static bool sActive;
  static PxU32 sGridSize;
  static PxReal sGridStep;
  static PxVec3 sWorld[kMaxVerts];
  static PxVec3 sPrevWorld[kMaxVerts];
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_KINEMATIC_SHELL_H
