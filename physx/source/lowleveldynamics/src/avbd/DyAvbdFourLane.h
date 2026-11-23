// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_FOUR_LANE_H
#define DY_AVBD_FOUR_LANE_H
#include "foundation/PxVecMath.h"
namespace physx { namespace Dy {
// Four independent rows, with coordinates in separate registers.
struct AvbdFourLane
{
    using Lane=aos::Vec4V;
    struct Vector {Lane x,y,z;};
    static Vector sub(const Vector& a,const Vector& b)
    {return {aos::V4Sub(a.x,b.x),aos::V4Sub(a.y,b.y),aos::V4Sub(a.z,b.z)};}
    static Vector add(const Vector& a,const Vector& b)
    {return {aos::V4Add(a.x,b.x),aos::V4Add(a.y,b.y),aos::V4Add(a.z,b.z)};}
    static Vector mul(const Vector& a,Lane b)
    {return {aos::V4Mul(a.x,b),aos::V4Mul(a.y,b),aos::V4Mul(a.z,b)};}
    static Lane dot(const Vector& a,const Vector& b)
    {return aos::V4Add(aos::V4Add(aos::V4Mul(a.x,b.x),aos::V4Mul(a.y,b.y)),aos::V4Mul(a.z,b.z));}
    static Vector cross(const Vector& a,const Vector& b)
    {
        return {aos::V4Sub(aos::V4Mul(a.y,b.z),aos::V4Mul(a.z,b.y)),
                aos::V4Sub(aos::V4Mul(a.z,b.x),aos::V4Mul(a.x,b.z)),
                aos::V4Sub(aos::V4Mul(a.x,b.y),aos::V4Mul(a.y,b.x))};
    }
};
} }
#endif
