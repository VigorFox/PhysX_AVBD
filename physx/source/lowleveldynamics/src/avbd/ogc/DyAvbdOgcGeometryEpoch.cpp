// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/solver/soft/DyAvbdSoftIslandPlan.h"

namespace physx {
namespace Dy {

bool resolveOgcTriangleCorePoint(
    const AvbdSoftContactGeometry &geometry,
    const AvbdOgcTriangleCoreCertificate *certificate,
    physx::PxU32 vertex, AvbdWeightedContactPoint &point) {
  (void)geometry;
  point.clear();
  if (vertex >= 3u)
    return false;
  if (certificate) {
    point = certificate->points[vertex];
    return point.count > 0u;
  }
  return false;
}

void AvbdOgcGeometryEpochSidecar::clear() {
  triangleCoreCertificates.clear();
  contactTriangleCoreIndices.clear();
}

void AvbdOgcGeometryEpochSidecar::reset() {
  triangleCoreCertificates.reset();
  contactTriangleCoreIndices.reset();
  geometryEpoch = 0u;
}

bool AvbdOgcGeometryEpochSidecar::moveContactMapping(
    physx::PxU32 sourceContactIndex,
    physx::PxU32 destinationContactIndex) {
  if (sourceContactIndex >= contactTriangleCoreIndices.size() ||
      destinationContactIndex >= contactTriangleCoreIndices.size())
    return false;
  if (sourceContactIndex != destinationContactIndex)
    contactTriangleCoreIndices[destinationContactIndex] =
        contactTriangleCoreIndices[sourceContactIndex];
  return true;
}

bool AvbdOgcGeometryEpochSidecar::finalizeContactCompaction(
    physx::PxU32 numContacts) {
  if (numContacts > contactTriangleCoreIndices.size())
    return false;
  physx::PxU32 certificateWriteIndex = 0u;
  physx::PxU32 previousCertificateIndex = PX_MAX_U32;
  for (physx::PxU32 contactIndex = 0u;
       contactIndex < numContacts; ++contactIndex) {
    const physx::PxU32 certificateIndex =
        contactTriangleCoreIndices[contactIndex];
    if (certificateIndex == PX_MAX_U32)
      continue;
    if (certificateIndex >= triangleCoreCertificates.size())
      return false;
    // Contact filters are stable compactions. Enforce that contract here so
    // an invalid reorder cannot overwrite a sparse certificate before it is
    // copied to its new dense slot.
    if (previousCertificateIndex != PX_MAX_U32 &&
        certificateIndex <= previousCertificateIndex)
      return false;
    previousCertificateIndex = certificateIndex;
    if (certificateWriteIndex != certificateIndex)
      triangleCoreCertificates[certificateWriteIndex] =
          triangleCoreCertificates[certificateIndex];
    contactTriangleCoreIndices[contactIndex] = certificateWriteIndex++;
  }
  contactTriangleCoreIndices.resize(numContacts);
  triangleCoreCertificates.resize(certificateWriteIndex);
  return true;
}

const AvbdOgcTriangleCoreCertificate *
AvbdOgcGeometryEpochSidecar::getTriangleCore(
    physx::PxU32 contactIndex) const {
  if (contactIndex >= contactTriangleCoreIndices.size())
    return nullptr;
  const physx::PxU32 certificateIndex =
      contactTriangleCoreIndices[contactIndex];
  if (certificateIndex >= triangleCoreCertificates.size())
    return nullptr;
  const AvbdOgcTriangleCoreCertificate &certificate =
      triangleCoreCertificates[certificateIndex];
  return geometryEpoch != 0u &&
          certificate.geometryEpoch == geometryEpoch
      ? &certificate : nullptr;
}

AvbdOgcTriangleCoreCertificate *
AvbdOgcGeometryEpochSidecar::getTriangleCoreMutable(
    physx::PxU32 contactIndex) {
  if (contactIndex >= contactTriangleCoreIndices.size())
    return nullptr;
  const physx::PxU32 certificateIndex =
      contactTriangleCoreIndices[contactIndex];
  if (certificateIndex >= triangleCoreCertificates.size())
    return nullptr;
  AvbdOgcTriangleCoreCertificate &certificate =
      triangleCoreCertificates[certificateIndex];
  return geometryEpoch != 0u &&
          certificate.geometryEpoch == geometryEpoch
      ? &certificate : nullptr;
}

bool AvbdOgcGeometryEpochView::hasPairPlan(
    physx::PxU32 numContacts) const {
  return pairStates && numPairStates > 0u && contactPairIndices &&
      numContactPairIndices == numContacts && numContacts > 0u;
}

const AvbdOgcRigidBoxGeometry *AvbdOgcGeometryEpochView::getRigidBox(
    physx::PxU32 contactIndex, physx::PxU32 numContacts) const {
  if (contactIndex >= numContacts || !hasPairPlan(numContacts))
    return nullptr;
  const physx::PxU32 pairIndex = contactPairIndices[contactIndex];
  if (pairIndex >= numPairStates)
    return nullptr;
  const AvbdOgcRigidBoxGeometry &rigidBox =
      pairStates[pairIndex].geometry.rigidBox;
  return rigidBox.valid ? &rigidBox : nullptr;
}

AvbdOgcGeometryEpochView makeOgcGeometryEpochView(
    const AvbdSoftIslandExecutionPlan *plan) {
  AvbdOgcGeometryEpochView view;
  if (!plan)
    return view;
  view.pairStates = plan->ogcPairStates;
  view.numPairStates = plan->numOgcPairStates;
  view.contactPairIndices = plan->ogcPairIndices;
  view.numContactPairIndices = plan->numOgcPairIndices;
  view.triangleCoreCertificates = plan->ogcTriangleCoreCertificates;
  view.numTriangleCoreCertificates = plan->numOgcTriangleCoreCertificates;
  view.contactTriangleCoreIndices = plan->ogcContactTriangleCoreIndices;
  view.numContactTriangleCoreIndices =
      plan->numOgcContactTriangleCoreIndices;
  view.geometryEpoch = plan->ogcGeometryEpoch;
  return view;
}

AvbdOgcGeometryEpochView makeOgcGeometryEpochView(
    const AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    const physx::PxU32 *contactPairIndices,
    physx::PxU32 numContactPairIndices,
    const AvbdOgcGeometryEpochSidecar *sidecar) {
  AvbdOgcGeometryEpochView view;
  view.pairStates = pairStates;
  view.numPairStates = numPairStates;
  view.contactPairIndices = contactPairIndices;
  view.numContactPairIndices = numContactPairIndices;
  if (sidecar) {
    view.triangleCoreCertificates =
        sidecar->triangleCoreCertificates.begin();
    view.numTriangleCoreCertificates =
        sidecar->triangleCoreCertificates.size();
    view.contactTriangleCoreIndices =
        sidecar->contactTriangleCoreIndices.begin();
    view.numContactTriangleCoreIndices =
        sidecar->contactTriangleCoreIndices.size();
    view.geometryEpoch = sidecar->geometryEpoch;
  }
  return view;
}

const AvbdOgcTriangleCoreCertificate *getOgcTriangleCoreCertificate(
    const AvbdSoftIslandExecutionPlan *plan, physx::PxU32 contactIndex,
    physx::PxU32 numContacts) {
  return makeOgcGeometryEpochView(plan).getTriangleCore(
      contactIndex, numContacts);
}

const AvbdOgcRigidBoxGeometry *getOgcRigidBoxGeometry(
    const AvbdSoftIslandExecutionPlan *plan, physx::PxU32 contactIndex,
    physx::PxU32 numContacts) {
  return makeOgcGeometryEpochView(plan).getRigidBox(
      contactIndex, numContacts);
}

} // namespace Dy
} // namespace physx
