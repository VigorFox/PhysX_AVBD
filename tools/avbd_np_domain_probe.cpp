// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "avbd/DyAvbdNpDomain.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
using namespace physx;
using namespace physx::Dy;

static bool report(const char* name,bool pass)
{ std::printf("[NP_DOMAIN_TEST] case=%s status=%s\n",name,pass?"PASS":"FAIL");return pass; }

int main()
{
    AvbdProbeFoundation foundation;bool pass=true;
    AvbdNpDomain d;d.normal=PxVec3d(0,1,0);d.radius[0]=d.radius[1]=0;
    // The old center witness misses a rotated corner of the complete box.
    for(PxU32 i=0;i<8;++i) d.points[0].pushBack(PxVec3d((i&1)?1:-1,(i&2)?.1:-.1,(i&4)?.1:-.1));
    d.points[1].pushBack(PxVec3d(0));
    PxTransform start[2]={PxTransform(PxVec3(0,.2f,0)),PxTransform(PxIdentity)};
    PxTransform end[2]={PxTransform(PxVec3(0,.2f,0),PxQuat(.4f,PxVec3(0,0,1))),start[1]};
    pass&=report("omitted_manifold_corner",end[0].p.y>0 && avbdNpGap(d,end)<-.2 && !avbdNpDomainPath(d,start,end));
    // Both endpoints are clear of a ceiling, but the middle of the shortest
    // rotation intersects. Endpoint-only admission would incorrectly pass.
    d.normal=PxVec3d(0,-1,0);d.points[0].clear();d.points[0].pushBack(PxVec3d(1,0,0));
    start[0]=PxTransform(PxVec3(0),PxQuat(PxPi/6,PxVec3(0,0,1)));
    end[0]=PxTransform(PxVec3(0),PxQuat(5*PxPi/6,PxVec3(0,0,1)));
    start[1]=PxTransform(PxVec3(0,.6f,0));end[1]=start[1];
    pass&=report("intermediate_rotation",avbdNpGap(d,start)>.09 && avbdNpGap(d,end)>.09 && !avbdNpDomainPath(d,start,end));
    // This arc stays at least 5 cm above the floor, but falls below the
    // artificial midpoint of its two endpoint gaps.
    d.normal=PxVec3d(0,1,0);start[1]=PxTransform(PxIdentity);end[1]=start[1];
    start[0]=PxTransform(PxVec3(0,1.05f,0),PxQuat(-2*PxPi/3,PxVec3(0,0,1)));
    end[0]=PxTransform(PxVec3(0,1.05f,0),PxQuat(-PxPi/3,PxVec3(0,0,1)));
    pass&=report("clear_arc_no_midplane_margin",avbdNpGap(d,start)>.18 && avbdNpGap(d,end)>.18 && avbdNpDomainPath(d,start,end));
    d.normal=PxVec3d(0,1,0);d.points[0][0]=PxVec3d(0);d.radius[0]=d.radius[1]=.5;
    start[0]=PxTransform(PxVec3(0,1,0));start[1]=PxTransform(PxIdentity);
    end[0]=PxTransform(PxVec3(12,6,-3));end[1]=PxTransform(PxVec3(12,5,-3));
    pass&=report("touching_pair_common_translation",avbdNpDomainPath(d,start,end));
    end[0].p.y=std::nextafter(6.f,0.f);
    pass&=report("one_float_ulp_overlap_rejected",!avbdNpDomainPath(d,start,end));
    d.radius[0]=d.radius[1]=0;d.points[0][0]=PxVec3d(1,0,0);
    unsigned state=12345;unsigned admitted=0,violations=0;
    auto random=[&]() { state=1664525*state+1013904223;return double(state)/4294967296.0; };
    for(unsigned i=0;i<2000;++i)
    {
        start[0]=PxTransform(PxVec3(0,float(.2+random()),0),PxQuat(float(random()*6.28),PxVec3(0,0,1)));
        end[0]=PxTransform(PxVec3(0,float(.2+random()),0),PxQuat(float(random()*6.28),PxVec3(0,0,1)));
        start[1]=PxTransform(PxIdentity);end[1]=start[1];
        if(!avbdNpDomainPath(d,start,end)) continue;
        ++admitted;PxQuatd q0=avbdNpRotation(start[0].q),q1=avbdNpRotation(end[0].q);
        if(q0.dot(q1)<0) q1=-q1;
        for(unsigned j=0;j<=100;++j)
        {
            const double t=double(j)/100;
            const PxQuatd q=(q0*(1-t)+q1*t).getNormalized();
            const double y=(1-t)*start[0].p.y+t*end[0].p.y+q.rotate(d.points[0][0]).y;
            if(y < -1e-12) ++violations;
        }
    }
    pass&=report("admitted_paths_dense_oracle",admitted>100 && violations==0);
    std::printf("[NP_DOMAIN_SAMPLES] admitted=%u violations=%u\n",admitted,violations);
    // Independent finite differences of the complete support oracle. The
    // radius changes the gap, never its orientation derivative.
    double gradientError=0;unsigned gradients=0;
    d.normal=PxVec3d(.3,1,.2).getNormalized();d.radius[0]=.17;d.radius[1]=.08;d.restDistance=.02;
    d.points[0].clear();d.points[0].pushBack(PxVec3d(.3,-.2,.1));
    d.points[1][0]=PxVec3d(-.1,.15,.2);
    for(unsigned test=0;test<120;++test)
    {
        PxTransform poses[2]={PxTransform(PxVec3(.1f,.7f,.3f),PxQuat(float(random()),PxVec3(0,0,1))),
            PxTransform(PxVec3(0),PxQuat(float(random()),PxVec3(0,1,0)))};
        PxVec3d arms[2];const double gap=avbdNpGapGradient(d,poses,arms);
        pass&=std::abs(gap-(avbdNpGap(d,poses)-d.restDistance))<1.e-12;
        for(unsigned side=0;side<2;++side)for(unsigned axis=0;axis<3;++axis)
        {
            const double h=1.e-3;PxVec3 basis(0);basis[axis]=1;
            PxTransform plus[2]={poses[0],poses[1]},minus[2]={poses[0],poses[1]};
            plus[side].q=(PxQuat(float(h),basis)*poses[side].q).getNormalized();
            minus[side].q=(PxQuat(float(-h),basis)*poses[side].q).getNormalized();
            const double fd=(avbdNpGap(d,plus)-avbdNpGap(d,minus))/(2*h);
            const double derivative=arms[side].cross(d.normal*(side?-1.:1.))[axis];
            gradientError=PxMax(gradientError,std::abs(fd-derivative));++gradients;
        }
    }
    pass&=report("rounded_core_gradient",gradientError<5.e-5);
    std::printf("[NP_SUPPORT_GRADIENT] samples=%u maximumError=%.9g\n",gradients,gradientError);
    return pass?0:1;
}
