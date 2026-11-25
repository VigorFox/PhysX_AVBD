#include "avbd/DyAvbdPointResponse.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace physx;
using namespace physx::Dy;

namespace
{
enum { MaxDofs = AvbdPointResponse::Dimension };
using Dense = double[MaxDofs][MaxDofs];

// Independent reference for J H^-1 J^T. The kernel factors square-root
// rows in PxReal. This reference solves the assembled normal matrix in double.
double solveDense(Dense& h, PxU32 size, const PxReal (&normal)[6])
{
    double rhs[MaxDofs] = {}, x[MaxDofs] = {};
    std::copy(normal, normal + 6, rhs);
    for (PxU32 i = 0; i < size; ++i)
    {
        PxU32 pivot = i;
        for (PxU32 k = i + 1; k < size; ++k)
            if (std::abs(h[k][i]) > std::abs(h[pivot][i]))
                pivot = k;
        std::swap(h[pivot], h[i]);
        std::swap(rhs[pivot], rhs[i]);
        for (PxU32 k = i + 1; k < size; ++k)
        {
            const double scale = h[k][i] / h[i][i];
            for (PxU32 t = i; t < size; ++t)
                h[k][t] -= scale * h[i][t];
            rhs[k] -= scale * rhs[i];
        }
    }
    for (PxU32 k = size; k > 0; --k)
    {
        const PxU32 i = k - 1;
        double value = rhs[i];
        for (PxU32 t = i + 1; t < size; ++t)
            value -= h[i][t] * x[t];
        x[i] = value / h[i][i];
    }
    double response = 0;
    for (PxU32 i = 0; i < 6; ++i)
        response += normal[i] * x[i];
    return response;
}

bool check(PxU32 count, bool supported, PxReal massScale, PxReal inverseDt)
{
    Dense h = {};
    AvbdPointPatch patch;
    patch.inverseDt = inverseDt;
    patch.count = count;
    patch.axisCount = 0;
    for (PxU32 i = 0; i < count; ++i)
    {
        const PxReal mass = (78 + 9 * i) * massScale;
        const PxReal inertia = (.48f + .1f * i) * massScale;
        const PxQuat pose(.27f + .31f * i, PxVec3(1, 2, -1).getNormalized());
        patch.bodies[i] = {mass, PxVec3(1 / inertia, 1 / (2 * inertia), 1 / (3 * inertia)), pose};
        for (PxU32 d = 0; d < 3; ++d)
        {
            h[6 * i + d][6 * i + d] = mass;
            PxVec3 basis(0);
            basis[d] = 1;
            const PxVec3 world = pose.rotate(basis);
            const double diagonal = double(inertia) * (d + 1) * patch.inverseDt * patch.inverseDt;
            for (PxU32 j = 0; j < 3; ++j)
                for (PxU32 k = 0; k < 3; ++k)
                    h[6 * i + j + 3][6 * i + k + 3] += diagonal * world[j] * world[k];
        }
    }
    const auto link = [&](PxU32 first, PxU32 second, PxVec3 armA, PxVec3 armB)
    {
        for (PxU32 d = 0; d < 3; ++d)
        {
            PxVec3 axis(0);
            axis[d] = 1;
            const PxVec3 a = armA.cross(axis), b = armB.cross(axis);
            const PxReal stiffness = 1.e6f;
            patch.axes[patch.axisCount++] = {first, second, axis, a, b, stiffness};
            double row[MaxDofs] = {};
            for (PxU32 k = 0; k < 3; ++k)
            {
                row[6 * first + k] = axis[k];
                row[6 * first + k + 3] = a[k];
                if (second < count)
                {
                    row[6 * second + k] = -axis[k];
                    row[6 * second + k + 3] = -b[k];
                }
            }
            for (PxU32 j = 0; j < 6 * count; ++j)
                for (PxU32 k = 0; k < 6 * count; ++k)
                    h[j][k] += stiffness * row[j] * row[k];
        }
    };
    for (PxU32 i = 1; i < count; ++i)
    {
        const PxVec3 arm(.27f * PxCos(PxReal(i)), .06f, .27f * PxSin(PxReal(i)));
        link(0, i, arm, -arm);
        if (supported)
            link(i, count, arm, PxVec3(0));
    }
    if (count > 2)
        link(1, 2, PxVec3(.19f, .1f, 0), PxVec3(-.13f, .04f, 0));
    if (supported)
        link(0, count, PxVec3(.1f, 0, .24f), PxVec3(0));
    const PxReal normal[] = {.48f, .8f, .36f, -.06f, .03f, .12f};
    PxReal measured = 0;
    const double expected = solveDense(h, 6 * count, normal);
    if (!patch.evaluate(normal, measured) || !(expected > 0))
        return false;
    const double error = std::abs(measured / expected - 1);
    std::printf("[POINT_RESPONSE] bodies=%u supported=%u massScale=%g inverseDt=%g relativeError=%g\n",
                count, unsigned(supported), double(massScale), double(inverseDt), error);
    return std::isfinite(error) && error < .005;
}
}

int main()
{
    bool passed = true;
    for (PxU32 count = 1; count <= AvbdPointResponse::MaxBodies; ++count)
        for (bool supported : {false, true})
            for (PxReal scale : {1.f, .001f, .00001f})
                for (PxReal inverseDt : {1.f, 60.f})
                    passed = check(count, supported, scale, inverseDt) && passed;
    return passed ? 0 : 1;
}
