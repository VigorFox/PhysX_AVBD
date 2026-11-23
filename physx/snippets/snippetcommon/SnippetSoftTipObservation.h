// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "foundation/PxVec3.h"
#include <vector>
#include <cmath>

namespace Snippets
{
// Mass-weighted material long axis, observed from published positions. The
// rate is the derivative of its elevation, so compression along it drops out.
struct SoftTipObservation
{
    std::vector<double> weights;
    double initialAngle = 0;
    template<class Rest, class Mass>
    void bake(physx::PxU32 count, Rest restX, Mass mass)
    {
        double total=0, moment=0;
        for(physx::PxU32 i=0;i<count;++i){total+=mass(i);moment+=mass(i)*restX(i);}
        weights.resize(count);
        for(physx::PxU32 i=0;i<count;++i)weights[i]=total>0 ? mass(i)*(restX(i)-moment/total) : 0;
    }
    template<class Position, class Velocity>
    bool sample(Position position, Velocity velocity, double& angle, double& rate) const
    {
        using namespace physx;
        PxVec3d axis(0), derivative(0);
        for(PxU32 i=0;i<weights.size();++i)
        {
            const auto p=position(i),v=velocity(i);
            axis+=PxVec3d(p.x,p.y,p.z)*weights[i];
            derivative+=PxVec3d(v.x,v.y,v.z)*weights[i];
        }
        const double h=std::hypot(axis.x,axis.z);
        if(!(axis.magnitudeSquared()>0))return false;
        angle=std::atan2(axis.y,h)-initialAngle;
        const double dh=h>0?(axis.x*derivative.x+axis.z*derivative.z)/h:std::hypot(derivative.x,derivative.z);
        rate=(h*derivative.y-axis.y*dh)/axis.magnitudeSquared();
        return std::isfinite(angle) && std::isfinite(rate);
    }
};
}
