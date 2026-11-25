// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_POINT_RESPONSE_H
#define DY_AVBD_POINT_RESPONSE_H

#include "foundation/PxQuat.h"
#include "foundation/PxMath.h"

namespace physx
{
namespace Dy
{
// A contact endpoint and its first ring of point joints. Only the lower
// triangle of the square root is stored. Each contact uses private scratch.
struct AvbdPointResponse
{
    enum { MaxLinks = 4, MaxBodies = MaxLinks + 1, MaxIncidences = 32, Dimension = 6 * MaxBodies };
    typedef PxReal Real;
    Real lower[Dimension * (Dimension + 1) / 2];

    static PX_FORCE_INLINE PxU32 offset(PxU32 i) { return i * (i + 1) / 2; }
    void clear(PxU32 count)
    {
        for (PxU32 i = 0; i < offset(6 * count); ++i)
            lower[i] = 0;
    }
    // Insert one row into R with orthogonal rotations. Keeping the square
    // root avoids subtracting two O(K) terms to recover the O(m/h^2) pivot
    // of a nearly free mechanism. R is stored transposed in lower[].
    void insert(Real (&v)[Dimension], PxU32 first, PxU32 count)
    {
        const PxU32 n = 6 * count;
        for (PxU32 i = first; i < n; ++i)
        {
            if (v[i] == 0)
                continue;
            Real& pivot = lower[offset(i) + i];
            const Real mag = PxMax(PxAbs(v[i]), pivot);
            const Real a = pivot / mag, b = v[i] / mag;
            const Real l = PxSqrt(a * a + b * b);
            const Real cosine = a / l, sine = b / l;
            pivot = mag * l;
            for (PxU32 k = i + 1; k < n; ++k)
            {
                Real& coefficient = lower[offset(k) + i];
                const Real next = cosine * coefficient + sine * v[k];
                v[k] = cosine * v[k] - sine * coefficient;
                coefficient = next;
            }
        }
    }
    void addInertia(PxU32 slot, PxReal massInvH2, const PxVec3& inverseInertia, const PxQuat& axes,
                    PxReal inverseDt)
    {
        for (PxU32 d = 0; d < 3; ++d)
        {
            const PxU32 i = 6 * slot + d;
            lower[offset(i) + i] = PxSqrt(massInvH2);
            Real angular[Dimension] = {};
            PxVec3 direction(0);
            direction[d] = inverseDt / PxSqrt(inverseInertia[d]);
            direction = axes.rotate(direction);
            for (PxU32 k = 0; k < 3; ++k)
                angular[6 * slot + 3 + k] = direction[k];
            insert(angular, 6 * slot + 3, slot + 1);
        }
    }
    void addPointAxis(PxU32 slot, PxU32 other, PxU32 count, const Real (&j)[6], const Real (&k)[6], Real penalty)
    {
        Real row[Dimension] = {};
        const Real scale = PxSqrt(penalty);
        for (PxU32 d = 0; d < 6; ++d)
        {
            row[6 * slot + d] = scale * j[d];
            if (other < count)
                row[6 * other + d] = -scale * k[d];
        }
        insert(row, 6 * slot, count);
    }
    bool compliance(PxU32 count, const Real (&j)[6], Real& result) const
    {
        Real x[Dimension];
        result = 0;
        for (PxU32 i = 0; i < 6 * count; ++i)
        {
            const Real* row = lower + offset(i);
            if (!(row[i] > 0))
                return false;
            Real rhs = i < 6 ? j[i] : 0;
            for (PxU32 k = 0; k < i; ++k)
                rhs -= row[k] * x[k];
            x[i] = rhs / row[i];
            result += x[i] * x[i];
        }
        return result > 0 && PxIsFinite(result);
    }
};
// Bounded input to the Table. Boundary axes have second == count.
struct AvbdPointPatch
{
    struct Body { PxReal massInvH2; PxVec3 inverseInertia; PxQuat axes; };
    struct Axis { PxU32 first, second; PxVec3 normal, angularA, angularB; PxReal penalty; };
    enum { MaxAxes = 3 * AvbdPointResponse::MaxLinks * AvbdPointResponse::MaxBodies };
    Body bodies[AvbdPointResponse::MaxBodies];
    Axis axes[MaxAxes];
    PxU32 count, axisCount;
    PxReal inverseDt;

    bool evaluate(const PxReal (&normalJacobian)[6], PxReal& result) const
    {
        AvbdPointResponse factor;
        factor.clear(count);
        for (PxU32 i = 0; i < count; ++i)
        {
            const Body& b = bodies[i];
            factor.addInertia(i, b.massInvH2, b.inverseInertia, b.axes, inverseDt);
        }
        for (PxU32 i = 0; i < axisCount; ++i)
        {
            const Axis& axis = axes[i];
            const PxReal a[] = {axis.normal.x, axis.normal.y, axis.normal.z,
                                axis.angularA.x, axis.angularA.y, axis.angularA.z};
            const PxReal b[] = {axis.normal.x, axis.normal.y, axis.normal.z,
                                axis.angularB.x, axis.angularB.y, axis.angularB.z};
            factor.addPointAxis(axis.first, axis.second, count, a, b, axis.penalty);
        }
        return factor.compliance(count, normalJacobian, result);
    }
};
} // namespace Dy
} // namespace physx
#endif
