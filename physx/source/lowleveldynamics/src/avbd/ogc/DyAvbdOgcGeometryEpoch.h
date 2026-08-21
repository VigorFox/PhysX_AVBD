// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_GEOMETRY_EPOCH_H
#define DY_AVBD_OGC_GEOMETRY_EPOCH_H

#include "avbd/contact/DyAvbdContact.h"
#include "foundation/PxArray.h"

namespace physx {
namespace Dy {

struct AvbdOgcRigidBoxGeometry;
struct AvbdOgcPairState;
struct AvbdSoftIslandExecutionPlan;

// Sparse per-triangle witness. Ordinary ground, analytic rigid and soft-soft
// manifold rows never instantiate this payload.
struct AvbdOgcTriangleCoreCertificate {
  AvbdWeightedContactPoint points[3];
  // Assigned by the geometry sidecar when the detector publishes this
  // certificate. A sparse witness is valid only inside that exact DCD epoch.
  physx::PxU32 geometryEpoch = 0u;

  PX_FORCE_INLINE bool isValid() const {
    for (physx::PxU32 vertex = 0u; vertex < 3u; ++vertex) {
      const AvbdWeightedContactPoint &point = points[vertex];
      if (point.count == 0u || point.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
        return false;
      physx::PxReal weightSum = 0.0f;
      for (physx::PxU32 support = 0u; support < point.count; ++support) {
        if (point.particleIndices[support] == PX_MAX_U32 ||
            !physx::PxIsFinite(point.weights[support]))
          return false;
        weightSum += point.weights[support];
      }
      if (!physx::PxIsFinite(weightSum) ||
          physx::PxAbs(weightSum - 1.0f) > 1.0e-3f)
        return false;
    }
    return true;
  }
};

// Resolve one TBIX triangle vertex from the sparse epoch certificate. A TBIX
// row without an epoch certificate is incomplete and fails closed.
bool resolveOgcTriangleCorePoint(
    const AvbdSoftContactGeometry &geometry,
    const AvbdOgcTriangleCoreCertificate *certificate,
    physx::PxU32 vertex, AvbdWeightedContactPoint &point);

// Owner for geometry that is sparse at contact-row granularity. Pair-shared
// shape descriptors live in AvbdOgcPairGeometryState; this sidecar owns only
// per-triangle certificates and the dense contact->sparse mapping for one DCD
// epoch. Capacity is reused by the native/component/terminal epoch owner.
struct AvbdOgcGeometryEpochSidecar {
  physx::PxArray<AvbdOgcTriangleCoreCertificate> triangleCoreCertificates;
  physx::PxArray<physx::PxU32> contactTriangleCoreIndices;
  physx::PxU32 geometryEpoch = 0u;

  // Starts one detector-owned DCD epoch while retaining allocated capacity.
  // Plain clear() only invalidates the current payload and is used by storage
  // teardown/failure paths; it must not fabricate a new published epoch.
  PX_FORCE_INLINE void beginEpoch() {
    triangleCoreCertificates.clear();
    contactTriangleCoreIndices.clear();
    geometryEpoch = geometryEpoch == PX_MAX_U32
        ? 1u : geometryEpoch + 1u;
  }
  void clear();
  void reset();
  PX_FORCE_INLINE bool resizeContactMapping(physx::PxU32 numContacts) {
    const physx::PxU32 previousSize = contactTriangleCoreIndices.size();
    contactTriangleCoreIndices.resize(numContacts);
    for (physx::PxU32 contactIndex = previousSize;
         contactIndex < numContacts; ++contactIndex)
      contactTriangleCoreIndices[contactIndex] = PX_MAX_U32;
    return contactTriangleCoreIndices.size() == numContacts;
  }
  PX_FORCE_INLINE bool publishTriangleCore(
      physx::PxU32 contactIndex,
      const AvbdOgcTriangleCoreCertificate &certificate) {
    if (geometryEpoch == 0u || !certificate.isValid() ||
        !resizeContactMapping(contactIndex + 1u) ||
        contactTriangleCoreIndices[contactIndex] != PX_MAX_U32)
      return false;
    const physx::PxU32 certificateIndex =
        triangleCoreCertificates.size();
    triangleCoreCertificates.pushBack(certificate);
    triangleCoreCertificates[certificateIndex].geometryEpoch = geometryEpoch;
    contactTriangleCoreIndices[contactIndex] = certificateIndex;
    return true;
  }
  bool moveContactMapping(
      physx::PxU32 sourceContactIndex,
      physx::PxU32 destinationContactIndex);
  bool finalizeContactCompaction(physx::PxU32 numContacts);

  AvbdOgcTriangleCoreCertificate *getTriangleCoreMutable(
      physx::PxU32 contactIndex);
  const AvbdOgcTriangleCoreCertificate *getTriangleCore(
      physx::PxU32 contactIndex) const;
};

// Immutable geometry view for one published DCD epoch. Contact force/velocity
// ownership remains in AvbdOgcPairState; this view only resolves pair-shared
// shape descriptors and sparse per-triangle certificates. Native selection,
// component fallback and terminal requery all publish this same contract.
struct AvbdOgcGeometryEpochView {
  const AvbdOgcPairState *pairStates = nullptr;
  physx::PxU32 numPairStates = 0u;
  const physx::PxU32 *contactPairIndices = nullptr;
  physx::PxU32 numContactPairIndices = 0u;
  const AvbdOgcTriangleCoreCertificate *triangleCoreCertificates = nullptr;
  physx::PxU32 numTriangleCoreCertificates = 0u;
  const physx::PxU32 *contactTriangleCoreIndices = nullptr;
  physx::PxU32 numContactTriangleCoreIndices = 0u;
  physx::PxU32 geometryEpoch = 0u;

  bool hasPairPlan(physx::PxU32 numContacts) const;
  PX_FORCE_INLINE bool hasTriangleCorePlan(
      physx::PxU32 numContacts) const {
    return geometryEpoch != 0u && contactTriangleCoreIndices &&
        numContactTriangleCoreIndices == numContacts &&
        (numTriangleCoreCertificates == 0u || triangleCoreCertificates);
  }
  const AvbdOgcRigidBoxGeometry *getRigidBox(
      physx::PxU32 contactIndex, physx::PxU32 numContacts) const;
  PX_FORCE_INLINE const AvbdOgcTriangleCoreCertificate *getTriangleCore(
      physx::PxU32 contactIndex, physx::PxU32 numContacts) const {
    if (contactIndex >= numContacts || !hasTriangleCorePlan(numContacts))
      return nullptr;
    const physx::PxU32 certificateIndex =
        contactTriangleCoreIndices[contactIndex];
    if (certificateIndex >= numTriangleCoreCertificates)
      return nullptr;
    const AvbdOgcTriangleCoreCertificate &certificate =
        triangleCoreCertificates[certificateIndex];
    return certificate.geometryEpoch == geometryEpoch
        ? &certificate : nullptr;
  }
};

AvbdOgcGeometryEpochView makeOgcGeometryEpochView(
    const AvbdSoftIslandExecutionPlan *plan);

AvbdOgcGeometryEpochView makeOgcGeometryEpochView(
    const AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    const physx::PxU32 *contactPairIndices,
    physx::PxU32 numContactPairIndices,
    const AvbdOgcGeometryEpochSidecar *sidecar);

const AvbdOgcTriangleCoreCertificate *getOgcTriangleCoreCertificate(
    const AvbdSoftIslandExecutionPlan *plan, physx::PxU32 contactIndex,
    physx::PxU32 numContacts);

const AvbdOgcRigidBoxGeometry *getOgcRigidBoxGeometry(
    const AvbdSoftIslandExecutionPlan *plan, physx::PxU32 contactIndex,
    physx::PxU32 numContacts);

} // namespace Dy
} // namespace physx

#endif
