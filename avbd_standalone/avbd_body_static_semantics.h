#pragma once
// Shared body-vs-static constants; keep in sync with the PhysX DyAvbd* path.

#include <cstdint>

namespace AvbdRef {

/** Penalty floor scale * M_eff / dt^2 for body-vs-static (PhysX DyAvbdSolver.cpp). */
static constexpr float kPenScaleBodyVsStatic = 2.0f;
/** Penalty floor scale for dynamic-dynamic contacts. */
static constexpr float kPenScaleDynDyn = 0.05f;
/** Per-body primal row boost: fraction of M/dt^2. */
static constexpr float kContactBoostFraction = 0.005f;
/** Minimum inner iterations when island is all body-vs-static. */
static constexpr int kMinBodyVsStaticInnerIters = 16;
static constexpr uint32_t kBodyStatic6x6FrictionMaxIslandContacts = 4u;
static constexpr uint32_t kBodyStatic6x6FrictionMaxBodyContacts = 8u;
/** Dyn-dyn tangents in aggregated 6x6 over-constrain stacks/pyramids. */
static constexpr uint32_t kDynDyn6x6FrictionMaxIslandContacts = 6u;
static constexpr uint32_t kDynDyn6x6FrictionMaxBodyContacts = 4u;

/** Standalone: static partner is UINT32_MAX on bodyB. */
inline bool isBodyVsStaticContact(uint32_t bodyA, uint32_t bodyB) {
  return bodyB == UINT32_MAX;
}

/** PhysX production: never stack body-static tangents in 6x6. */
inline bool useBodyVsStaticFrictionIn6x6(uint32_t bodyA, uint32_t bodyB) {
  return !isBodyVsStaticContact(bodyA, bodyB);
}

/** Standalone flat-ground tests only (see semantics doc). */
inline bool useBodyVsStaticFrictionIn6x6LowContactIsland(
    uint32_t bodyA, uint32_t bodyB, uint32_t staticContactsOnBody,
    uint32_t numContactsInIsland) {
  if (!isBodyVsStaticContact(bodyA, bodyB))
    return true;
  if (numContactsInIsland > kBodyStatic6x6FrictionMaxIslandContacts)
    return false;
  return staticContactsOnBody <= kBodyStatic6x6FrictionMaxBodyContacts;
}

inline int bodyVsStaticPrimalRowCount(uint32_t bodyA, uint32_t bodyB,
                                      uint32_t staticContactsOnBody,
                                      uint32_t numContactsInIsland,
                                      bool useLowContactIslandException) {
  const bool in6x6 =
      useLowContactIslandException
          ? useBodyVsStaticFrictionIn6x6LowContactIsland(
                bodyA, bodyB, staticContactsOnBody, numContactsInIsland)
          : useBodyVsStaticFrictionIn6x6(bodyA, bodyB);
  return in6x6 ? 3 : 1;
}

inline bool useDynDynFrictionIn6x6(uint32_t dynContactsOnBody,
                                   uint32_t numDynContactsInIsland) {
  if (numDynContactsInIsland > kDynDyn6x6FrictionMaxIslandContacts)
    return false;
  return dynContactsOnBody <= kDynDyn6x6FrictionMaxBodyContacts;
}

inline int contactPrimalRowCount(uint32_t bodyA, uint32_t bodyB,
                                 uint32_t staticContactsOnBody,
                                 uint32_t dynContactsOnBody,
                                 uint32_t numDynContactsInIsland,
                                 bool useLowContactIslandException) {
  if (isBodyVsStaticContact(bodyA, bodyB))
    return bodyVsStaticPrimalRowCount(bodyA, bodyB, staticContactsOnBody,
                                      numDynContactsInIsland,
                                      useLowContactIslandException);
  return useDynDynFrictionIn6x6(dynContactsOnBody, numDynContactsInIsland) ? 3
                                                                           : 1;
}

} // namespace AvbdRef
