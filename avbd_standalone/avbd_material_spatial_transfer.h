#pragma once

#include "avbd_material_interface_wrench.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace AvbdRef {

using MaterialWorldSpatialVector = std::array<double, 6>;

struct MaterialSpatialBody {
  uint64_t stableKey = 0;
  Vec3 worldPosition;
  double inverseMass = 0.0;
  Mat33 worldInverseInertia;
};

struct MaterialSpatialInterface {
  uint64_t stableKey = 0;
  uint64_t bodyKeyA = 0;
  uint64_t bodyKeyB = 0;
  bool bodyBStatic = false;
};

struct MaterialSpatialTransfer {
  std::vector<MaterialSpatialBody> bodies;
  std::vector<MaterialSpatialInterface> interfaces;
  std::vector<size_t> bodyCanonicalToInput;
  std::vector<size_t> interfaceCanonicalToInput;
  std::vector<size_t> bodyA;
  std::vector<size_t> bodyB;
  bool finite = false;
};

namespace MaterialSpatialTransferDetail {

inline bool finiteVec3(const Vec3 &value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

inline bool finiteMat33(const Mat33 &value) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      if (!std::isfinite(value.m[row][column]))
        return false;
    }
  }
  return true;
}

inline bool finiteSpatial(const MaterialWorldSpatialVector &value) {
  for (double component : value) {
    if (!std::isfinite(component))
      return false;
  }
  return true;
}

inline MaterialWorldSpatialVector zeroSpatial() {
  return MaterialWorldSpatialVector{};
}

inline void addScaled(MaterialWorldSpatialVector &target,
                      const MaterialWorldSpatialVector &value,
                      double scale) {
  for (size_t component = 0; component < 6u; ++component)
    target[component] += scale * value[component];
}

inline Vec3 linear(const MaterialWorldSpatialVector &value) {
  return Vec3(static_cast<float>(value[0]),
              static_cast<float>(value[1]),
              static_cast<float>(value[2]));
}

inline Vec3 angular(const MaterialWorldSpatialVector &value) {
  return Vec3(static_cast<float>(value[3]),
              static_cast<float>(value[4]),
              static_cast<float>(value[5]));
}

inline MaterialWorldSpatialVector fromVec3(const Vec3 &linearValue,
                                           const Vec3 &angularValue) {
  return {static_cast<double>(linearValue.x),
          static_cast<double>(linearValue.y),
          static_cast<double>(linearValue.z),
          static_cast<double>(angularValue.x),
          static_cast<double>(angularValue.y),
          static_cast<double>(angularValue.z)};
}

} // namespace MaterialSpatialTransferDetail

inline MaterialSpatialTransfer buildMaterialSpatialTransfer(
    const std::vector<MaterialSpatialBody> &inputBodies,
    const std::vector<MaterialSpatialInterface> &inputInterfaces) {
  MaterialSpatialTransfer result;
  if (inputBodies.empty() || inputInterfaces.empty())
    return result;

  std::vector<std::pair<uint64_t, size_t>> bodyOrder(
      inputBodies.size());
  for (size_t body = 0; body < inputBodies.size(); ++body)
    bodyOrder[body] = {inputBodies[body].stableKey, body};
  std::sort(bodyOrder.begin(), bodyOrder.end());
  result.bodies.resize(inputBodies.size());
  result.bodyCanonicalToInput.resize(inputBodies.size());
  for (size_t body = 0; body < bodyOrder.size(); ++body) {
    if ((body > 0u &&
         bodyOrder[body - 1u].first == bodyOrder[body].first) ||
        !MaterialSpatialTransferDetail::finiteVec3(
            inputBodies[bodyOrder[body].second].worldPosition) ||
        !MaterialSpatialTransferDetail::finiteMat33(
            inputBodies[bodyOrder[body].second]
                .worldInverseInertia) ||
        !(inputBodies[bodyOrder[body].second].inverseMass >
          0.0) ||
        !std::isfinite(
            inputBodies[bodyOrder[body].second].inverseMass)) {
      return MaterialSpatialTransfer{};
    }
    result.bodies[body] = inputBodies[bodyOrder[body].second];
    result.bodyCanonicalToInput[body] = bodyOrder[body].second;
  }

  std::vector<std::pair<uint64_t, size_t>> interfaceOrder(
      inputInterfaces.size());
  for (size_t interfaceIndex = 0;
       interfaceIndex < inputInterfaces.size(); ++interfaceIndex) {
    interfaceOrder[interfaceIndex] = {
        inputInterfaces[interfaceIndex].stableKey,
        interfaceIndex};
  }
  std::sort(interfaceOrder.begin(), interfaceOrder.end());
  result.interfaces.resize(inputInterfaces.size());
  result.interfaceCanonicalToInput.resize(inputInterfaces.size());
  result.bodyA.resize(inputInterfaces.size());
  result.bodyB.assign(
      inputInterfaces.size(),
      std::numeric_limits<size_t>::max());
  for (size_t interfaceIndex = 0;
       interfaceIndex < interfaceOrder.size(); ++interfaceIndex) {
    if (interfaceIndex > 0u &&
        interfaceOrder[interfaceIndex - 1u].first ==
            interfaceOrder[interfaceIndex].first) {
      return MaterialSpatialTransfer{};
    }
    const size_t input = interfaceOrder[interfaceIndex].second;
    const MaterialSpatialInterface &interfaceValue =
        inputInterfaces[input];
    const auto findBody =
        [&](uint64_t key) {
          return std::lower_bound(
              result.bodies.begin(), result.bodies.end(), key,
              [](const MaterialSpatialBody &body, uint64_t value) {
                return body.stableKey < value;
              });
        };
    const auto bodyA = findBody(interfaceValue.bodyKeyA);
    if (bodyA == result.bodies.end() ||
        bodyA->stableKey != interfaceValue.bodyKeyA) {
      return MaterialSpatialTransfer{};
    }
    result.bodyA[interfaceIndex] =
        static_cast<size_t>(bodyA - result.bodies.begin());
    if (!interfaceValue.bodyBStatic) {
      const auto bodyB = findBody(interfaceValue.bodyKeyB);
      if (bodyB == result.bodies.end() ||
          bodyB->stableKey != interfaceValue.bodyKeyB ||
          bodyB == bodyA) {
        return MaterialSpatialTransfer{};
      }
      result.bodyB[interfaceIndex] =
          static_cast<size_t>(bodyB - result.bodies.begin());
    }
    result.interfaces[interfaceIndex] = interfaceValue;
    result.interfaceCanonicalToInput[interfaceIndex] = input;
  }
  result.finite = true;
  return result;
}

inline MaterialWorldSpatialVector materialBodyComTwistToWorldOrigin(
    const Vec6 &bodyComTwist, const Vec3 &worldPosition) {
  const Vec3 angular = bodyComTwist.angular();
  const Vec3 linear =
      bodyComTwist.linear() - angular.cross(worldPosition);
  return MaterialSpatialTransferDetail::fromVec3(linear, angular);
}

inline Vec6 materialBodyWorldOriginTwistToCom(
    const MaterialWorldSpatialVector &worldOriginTwist,
    const Vec3 &worldPosition) {
  const Vec3 angular =
      MaterialSpatialTransferDetail::angular(worldOriginTwist);
  const Vec3 linear =
      MaterialSpatialTransferDetail::linear(worldOriginTwist) +
      angular.cross(worldPosition);
  return Vec6(linear, angular);
}

inline bool scatterMaterialInterfaceWrenchesToBodies(
    const MaterialSpatialTransfer &transfer,
    const std::vector<MaterialWorldSpatialVector>
        &inputInterfaceWrenches,
    std::vector<MaterialWorldSpatialVector> &inputBodyLoads) {
  if (!transfer.finite ||
      inputInterfaceWrenches.size() !=
          transfer.interfaces.size()) {
    return false;
  }
  std::vector<MaterialWorldSpatialVector> canonicalLoads(
      transfer.bodies.size(),
      MaterialSpatialTransferDetail::zeroSpatial());
  for (size_t interfaceIndex = 0;
       interfaceIndex < transfer.interfaces.size();
       ++interfaceIndex) {
    const MaterialWorldSpatialVector &wrench =
        inputInterfaceWrenches[
            transfer
                .interfaceCanonicalToInput[interfaceIndex]];
    if (!MaterialSpatialTransferDetail::finiteSpatial(wrench))
      return false;
    MaterialSpatialTransferDetail::addScaled(
        canonicalLoads[transfer.bodyA[interfaceIndex]],
        wrench, 1.0);
    if (transfer.bodyB[interfaceIndex] !=
        std::numeric_limits<size_t>::max()) {
      MaterialSpatialTransferDetail::addScaled(
          canonicalLoads[transfer.bodyB[interfaceIndex]],
          wrench, -1.0);
    }
  }
  inputBodyLoads.assign(
      transfer.bodies.size(),
      MaterialSpatialTransferDetail::zeroSpatial());
  for (size_t body = 0; body < transfer.bodies.size(); ++body) {
    inputBodyLoads[transfer.bodyCanonicalToInput[body]] =
        canonicalLoads[body];
  }
  return true;
}

inline bool gatherMaterialBodyTwistsToInterfaces(
    const MaterialSpatialTransfer &transfer,
    const std::vector<MaterialWorldSpatialVector>
        &inputBodyWorldOriginTwists,
    std::vector<MaterialWorldSpatialVector>
        &inputInterfaceRelativeTwists) {
  if (!transfer.finite ||
      inputBodyWorldOriginTwists.size() !=
          transfer.bodies.size()) {
    return false;
  }
  std::vector<MaterialWorldSpatialVector> canonicalTwists(
      transfer.bodies.size(),
      MaterialSpatialTransferDetail::zeroSpatial());
  for (size_t body = 0; body < transfer.bodies.size(); ++body) {
    canonicalTwists[body] =
        inputBodyWorldOriginTwists[
            transfer.bodyCanonicalToInput[body]];
    if (!MaterialSpatialTransferDetail::finiteSpatial(
            canonicalTwists[body])) {
      return false;
    }
  }
  inputInterfaceRelativeTwists.assign(
      transfer.interfaces.size(),
      MaterialSpatialTransferDetail::zeroSpatial());
  for (size_t interfaceIndex = 0;
       interfaceIndex < transfer.interfaces.size();
       ++interfaceIndex) {
    MaterialWorldSpatialVector relative =
        canonicalTwists[transfer.bodyA[interfaceIndex]];
    if (transfer.bodyB[interfaceIndex] !=
        std::numeric_limits<size_t>::max()) {
      MaterialSpatialTransferDetail::addScaled(
          relative,
          canonicalTwists[transfer.bodyB[interfaceIndex]],
          -1.0);
    }
    inputInterfaceRelativeTwists[
        transfer.interfaceCanonicalToInput[interfaceIndex]] =
        relative;
  }
  return true;
}

inline bool applyMaterialBodyWorldOriginMobility(
    const MaterialSpatialTransfer &transfer,
    const std::vector<MaterialWorldSpatialVector> &inputBodyLoads,
    std::vector<MaterialWorldSpatialVector>
        &inputBodyWorldOriginTwists) {
  if (!transfer.finite ||
      inputBodyLoads.size() != transfer.bodies.size()) {
    return false;
  }
  inputBodyWorldOriginTwists.assign(
      transfer.bodies.size(),
      MaterialSpatialTransferDetail::zeroSpatial());
  for (size_t canonical = 0; canonical < transfer.bodies.size();
       ++canonical) {
    const size_t input = transfer.bodyCanonicalToInput[canonical];
    const MaterialWorldSpatialVector &load = inputBodyLoads[input];
    if (!MaterialSpatialTransferDetail::finiteSpatial(load))
      return false;
    const MaterialSpatialBody &body = transfer.bodies[canonical];
    const Vec3 force =
        MaterialSpatialTransferDetail::linear(load);
    const Vec3 worldMoment =
        MaterialSpatialTransferDetail::angular(load);
    const Vec3 comTorque =
        worldMoment - body.worldPosition.cross(force);
    const Vec3 angular =
        body.worldInverseInertia * comTorque;
    const Vec3 comLinear =
        force * static_cast<float>(body.inverseMass);
    const Vec3 worldOriginLinear =
        comLinear - angular.cross(body.worldPosition);
    inputBodyWorldOriginTwists[input] =
        MaterialSpatialTransferDetail::fromVec3(
            worldOriginLinear, angular);
  }
  return true;
}

inline double materialSpatialWork(
    const std::vector<MaterialWorldSpatialVector> &a,
    const std::vector<MaterialWorldSpatialVector> &b) {
  if (a.size() != b.size())
    return std::numeric_limits<double>::infinity();
  double work = 0.0;
  for (size_t entry = 0; entry < a.size(); ++entry) {
    for (size_t component = 0; component < 6u; ++component)
      work += a[entry][component] * b[entry][component];
  }
  return work;
}

} // namespace AvbdRef
