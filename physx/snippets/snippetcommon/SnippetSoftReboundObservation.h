// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "PxPhysicsAPI.h"

namespace Snippets
{
using namespace physx;
struct ReboundObservation
{
    PxU32 contactFrame=0,apexFrame=0;
    PxU32 departureFrame=0,outgoingFrames=0;
    PxReal incoming=0,up=0,lowest=0,rise=0,minVolume=1,apexVolume=1,clearance=0;
    void sample(PxU32 frame,bool near,PxReal height,PxReal velocity,PxReal ratio,PxReal gap)
    {
        if(apexFrame || (contactFrame && frame-contactFrame>60))return;
        if(!contactFrame)
        {
            incoming=PxMax(incoming,-velocity);
            if(!near)return;
            contactFrame=frame;lowest=height;
        }
        minVolume=PxMin(minVolume,ratio);
        if(up>0 && velocity<=0){apexFrame=frame;return;}
        lowest=PxMin(lowest,height);
        up=PxMax(up,velocity);
        if(velocity>0){rise=PxMax(rise,height-lowest);apexVolume=ratio;clearance=PxMax(clearance,gap);}
        // A glancing collision can separate monotonically along the incident
        // normal. Measure sustained outward flight of the recovered material.
        if(!near && velocity>.1f*incoming && ratio>minVolume && height>lowest)
        {
            if(++outgoingFrames==3 && !departureFrame)departureFrame=frame;
        }
        else outgoingFrames=0;
    }
};
}
