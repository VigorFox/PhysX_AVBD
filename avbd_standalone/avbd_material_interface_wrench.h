#pragma once

#include "avbd_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace AvbdRef {

using MaterialSpatialWrench = std::array<double, 6>;

struct MaterialInterfacePoint {
  Vec3 worldPoint;
  Vec3 normal;
  Vec3 tangent0;
  Vec3 tangent1;
  uint64_t stableKey = 0;
};

struct MaterialInterfaceWrenchMap {
  std::vector<MaterialInterfacePoint> points;
  // Row-major 6 x (3 * pointCount): point impulses -> world-origin wrench.
  std::vector<double> restriction;
  // Row-major (3 * pointCount) x 6: minimum-norm range prolongation.
  std::vector<double> prolongation;
  // Row-major 6 x 6 orthogonal projector onto range(restriction).
  std::array<double, 36> rangeProjector{};
  int rank = 0;
  bool finite = false;
};

namespace MaterialInterfaceWrenchDetail {

inline double component(const Vec3 &value, int index) {
  return index == 0 ? static_cast<double>(value.x)
                    : (index == 1 ? static_cast<double>(value.y)
                                  : static_cast<double>(value.z));
}

inline bool finiteVec3(const Vec3 &value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

inline bool pointLess(const MaterialInterfacePoint &a,
                      const MaterialInterfacePoint &b) {
  return a.stableKey < b.stableKey;
}

inline bool symmetricPseudoInverse6(
    const std::array<double, 36> &input,
    std::array<double, 36> &pseudoInverse, int &rank,
    double relativeRankTolerance) {
  std::array<double, 36> matrix = input;
  std::array<double, 36> eigenvectors{};
  for (int diagonal = 0; diagonal < 6; ++diagonal)
    eigenvectors[diagonal * 6 + diagonal] = 1.0;

  for (int iteration = 0; iteration < 128; ++iteration) {
    int pivot = 0;
    int target = 1;
    double maximum = std::fabs(matrix[1]);
    for (int row = 0; row < 6; ++row) {
      for (int column = row + 1; column < 6; ++column) {
        const double magnitude =
            std::fabs(matrix[row * 6 + column]);
        if (magnitude > maximum) {
          maximum = magnitude;
          pivot = row;
          target = column;
        }
      }
    }
    double diagonalScale = 0.0;
    for (int diagonal = 0; diagonal < 6; ++diagonal) {
      diagonalScale =
          std::max(diagonalScale,
                   std::fabs(matrix[diagonal * 6 + diagonal]));
    }
    if (maximum <=
        1.0e-15 * std::max(1.0, diagonalScale)) {
      break;
    }

    const double app = matrix[pivot * 6 + pivot];
    const double aqq = matrix[target * 6 + target];
    const double apq = matrix[pivot * 6 + target];
    const double angle =
        0.5 * std::atan2(2.0 * apq, aqq - app);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);

    for (int entry = 0; entry < 6; ++entry) {
      if (entry == pivot || entry == target)
        continue;
      const double aep = matrix[entry * 6 + pivot];
      const double aeq = matrix[entry * 6 + target];
      const double rotatedP = cosine * aep - sine * aeq;
      const double rotatedQ = sine * aep + cosine * aeq;
      matrix[entry * 6 + pivot] = rotatedP;
      matrix[pivot * 6 + entry] = rotatedP;
      matrix[entry * 6 + target] = rotatedQ;
      matrix[target * 6 + entry] = rotatedQ;
    }
    matrix[pivot * 6 + pivot] =
        cosine * cosine * app - 2.0 * cosine * sine * apq +
        sine * sine * aqq;
    matrix[target * 6 + target] =
        sine * sine * app + 2.0 * cosine * sine * apq +
        cosine * cosine * aqq;
    matrix[pivot * 6 + target] = 0.0;
    matrix[target * 6 + pivot] = 0.0;

    for (int row = 0; row < 6; ++row) {
      const double vip = eigenvectors[row * 6 + pivot];
      const double viq = eigenvectors[row * 6 + target];
      eigenvectors[row * 6 + pivot] =
          cosine * vip - sine * viq;
      eigenvectors[row * 6 + target] =
          sine * vip + cosine * viq;
    }
  }

  double maximumEigenvalue = 0.0;
  for (int diagonal = 0; diagonal < 6; ++diagonal) {
    const double eigenvalue = matrix[diagonal * 6 + diagonal];
    if (!std::isfinite(eigenvalue))
      return false;
    maximumEigenvalue = std::max(maximumEigenvalue, eigenvalue);
  }
  if (!(maximumEigenvalue > 0.0) ||
      !std::isfinite(maximumEigenvalue)) {
    return false;
  }
  const double threshold =
      std::max(std::numeric_limits<double>::epsilon(),
               relativeRankTolerance * maximumEigenvalue);
  pseudoInverse.fill(0.0);
  rank = 0;
  for (int mode = 0; mode < 6; ++mode) {
    const double eigenvalue = matrix[mode * 6 + mode];
    if (!(eigenvalue > threshold))
      continue;
    ++rank;
    const double inverse = 1.0 / eigenvalue;
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < 6; ++column) {
        pseudoInverse[row * 6 + column] +=
            eigenvectors[row * 6 + mode] * inverse *
            eigenvectors[column * 6 + mode];
      }
    }
  }
  return rank > 0;
}

} // namespace MaterialInterfaceWrenchDetail

inline MaterialInterfaceWrenchMap buildMaterialInterfaceWrenchMap(
    const std::vector<MaterialInterfacePoint> &inputPoints,
    double relativeRankTolerance = 1.0e-12) {
  MaterialInterfaceWrenchMap result;
  if (inputPoints.empty() || !(relativeRankTolerance > 0.0) ||
      !std::isfinite(relativeRankTolerance)) {
    return result;
  }
  result.points = inputPoints;
  std::sort(result.points.begin(), result.points.end(),
            MaterialInterfaceWrenchDetail::pointLess);
  for (size_t point = 0; point < result.points.size(); ++point) {
    const MaterialInterfacePoint &value = result.points[point];
    if (!MaterialInterfaceWrenchDetail::finiteVec3(
            value.worldPoint) ||
        !MaterialInterfaceWrenchDetail::finiteVec3(value.normal) ||
        !MaterialInterfaceWrenchDetail::finiteVec3(
            value.tangent0) ||
        !MaterialInterfaceWrenchDetail::finiteVec3(
            value.tangent1) ||
        value.normal.length2() <= 1.0e-12f ||
        value.tangent0.length2() <= 1.0e-12f ||
        value.tangent1.length2() <= 1.0e-12f ||
        (point > 0 &&
         result.points[point - 1].stableKey == value.stableKey)) {
      return result;
    }
  }

  const size_t coordinateCount = result.points.size() * 3u;
  result.restriction.assign(6u * coordinateCount, 0.0);
  for (size_t point = 0; point < result.points.size(); ++point) {
    const Vec3 axes[3] = {result.points[point].normal,
                          result.points[point].tangent0,
                          result.points[point].tangent1};
    for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
      const size_t column = point * 3u +
                            static_cast<size_t>(axisIndex);
      const Vec3 moment =
          result.points[point].worldPoint.cross(axes[axisIndex]);
      for (int componentIndex = 0; componentIndex < 3;
           ++componentIndex) {
        result.restriction[
            static_cast<size_t>(componentIndex) *
                coordinateCount +
            column] =
            MaterialInterfaceWrenchDetail::component(
                axes[axisIndex], componentIndex);
        result.restriction[
            static_cast<size_t>(componentIndex + 3) *
                coordinateCount +
            column] =
            MaterialInterfaceWrenchDetail::component(
                moment, componentIndex);
      }
    }
  }

  std::array<double, 36> gram{};
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      double value = 0.0;
      for (size_t coordinate = 0; coordinate < coordinateCount;
           ++coordinate) {
        value +=
            result.restriction[
                static_cast<size_t>(row) * coordinateCount +
                coordinate] *
            result.restriction[
                static_cast<size_t>(column) * coordinateCount +
                coordinate];
      }
      gram[row * 6 + column] = value;
    }
  }

  std::array<double, 36> pseudoInverse{};
  if (!MaterialInterfaceWrenchDetail::symmetricPseudoInverse6(
          gram, pseudoInverse, result.rank,
          relativeRankTolerance)) {
    return result;
  }
  result.prolongation.assign(coordinateCount * 6u, 0.0);
  for (size_t coordinate = 0; coordinate < coordinateCount;
       ++coordinate) {
    for (int column = 0; column < 6; ++column) {
      double value = 0.0;
      for (int row = 0; row < 6; ++row) {
        value +=
            result.restriction[
                static_cast<size_t>(row) * coordinateCount +
                coordinate] *
            pseudoInverse[row * 6 + column];
      }
      result.prolongation[coordinate * 6u +
                          static_cast<size_t>(column)] = value;
    }
  }
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      double value = 0.0;
      for (size_t coordinate = 0; coordinate < coordinateCount;
           ++coordinate) {
        value +=
            result.restriction[
                static_cast<size_t>(row) * coordinateCount +
                coordinate] *
            result.prolongation[
                coordinate * 6u +
                static_cast<size_t>(column)];
      }
      result.rangeProjector[row * 6 + column] = value;
    }
  }
  result.finite = true;
  for (double value : result.restriction)
    result.finite = result.finite && std::isfinite(value);
  for (double value : result.prolongation)
    result.finite = result.finite && std::isfinite(value);
  for (double value : result.rangeProjector)
    result.finite = result.finite && std::isfinite(value);
  return result;
}

inline bool restrictMaterialPointImpulses(
    const MaterialInterfaceWrenchMap &map,
    const std::vector<double> &pointImpulses,
    MaterialSpatialWrench &wrench) {
  const size_t coordinateCount = map.points.size() * 3u;
  if (!map.finite || pointImpulses.size() != coordinateCount)
    return false;
  wrench.fill(0.0);
  for (int row = 0; row < 6; ++row) {
    for (size_t coordinate = 0; coordinate < coordinateCount;
         ++coordinate) {
      wrench[static_cast<size_t>(row)] +=
          map.restriction[
              static_cast<size_t>(row) * coordinateCount +
              coordinate] *
          pointImpulses[coordinate];
    }
  }
  for (double value : wrench) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

inline bool prolongMaterialInterfaceWrench(
    const MaterialInterfaceWrenchMap &map,
    const MaterialSpatialWrench &wrench,
    std::vector<double> &pointImpulses) {
  if (!map.finite)
    return false;
  const size_t coordinateCount = map.points.size() * 3u;
  pointImpulses.assign(coordinateCount, 0.0);
  for (size_t coordinate = 0; coordinate < coordinateCount;
       ++coordinate) {
    for (int column = 0; column < 6; ++column) {
      pointImpulses[coordinate] +=
          map.prolongation[
              coordinate * 6u +
              static_cast<size_t>(column)] *
          wrench[static_cast<size_t>(column)];
    }
    if (!std::isfinite(pointImpulses[coordinate]))
      return false;
  }
  return true;
}

inline MaterialSpatialWrench projectMaterialInterfaceWrenchToRange(
    const MaterialInterfaceWrenchMap &map,
    const MaterialSpatialWrench &wrench) {
  MaterialSpatialWrench projected{};
  if (!map.finite)
    return projected;
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      projected[static_cast<size_t>(row)] +=
          map.rangeProjector[row * 6 + column] *
          wrench[static_cast<size_t>(column)];
    }
  }
  return projected;
}

inline bool projectMaterialPointImpulses(
    const MaterialInterfaceWrenchMap &map,
    const std::vector<double> &friction,
    std::vector<double> &pointImpulses) {
  if (!map.finite || friction.size() != map.points.size() ||
      pointImpulses.size() != map.points.size() * 3u) {
    return false;
  }
  for (size_t point = 0; point < map.points.size(); ++point) {
    if (!(friction[point] >= 0.0) ||
        !std::isfinite(friction[point])) {
      return false;
    }
    const size_t row = point * 3u;
    const double normal = std::max(0.0, pointImpulses[row]);
    double tangent0 = pointImpulses[row + 1u];
    double tangent1 = pointImpulses[row + 2u];
    const double magnitude =
        std::sqrt(tangent0 * tangent0 + tangent1 * tangent1);
    const double cap = friction[point] * normal;
    if (magnitude > cap && magnitude > 0.0) {
      const double scale = cap / magnitude;
      tangent0 *= scale;
      tangent1 *= scale;
    }
    pointImpulses[row] = normal;
    pointImpulses[row + 1u] = tangent0;
    pointImpulses[row + 2u] = tangent1;
  }
  return true;
}

} // namespace AvbdRef
