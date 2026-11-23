// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_FRAME_ASSEMBLY_H
#define DY_AVBD_FRAME_ASSEMBLY_H
#include "DyAvbdElasticFrame.h"
#include "DyKernelTable.h"
#include <new>

namespace physx { namespace Dy {
// A borrowed destination in the existing local block/group. No state or
// integrator is owned here. Matrices are full 6x6 blocks with a row stride.
struct AvbdFrameAssemblyTarget
{
    double *gradient,*K,*M,*energy;
    PxU32 count,matrixRowStride;
};
struct AvbdFrameAssemblyItem
{
    PxU32 material;
    PxU8 poses[2],members[2]; // 255 means outside this local group.
};
struct AvbdFrameAssemblyArgs
{
    enum { MaxItems=8,MaxPoses=16,Outside=255 };
    const AvbdElasticFrameMaterials* materials;
    AvbdFrameAssemblyTarget target;
    AvbdFrameAssemblyItem items[MaxItems];
    PxTransformT<double> poses[MaxPoses];
    PxU32 count=0,poseCount=0;
    bool valid=false;
};
// Per-item outputs and sparse destination lists are local assembly scratch.
// Evaluation lanes write distinct values; after their barrier, each matrix
// entry has one reduction lane. Aliased endpoints retain all four cross terms.
// Nothing here persists across a candidate or color, or owns physical state.
struct AvbdFramePacketScratch
{
    union Value
    {
        AvbdElasticFrameContribution coupled;
        AvbdElasticFrameBlockContribution endpoint;
    } values[AvbdFrameAssemblyArgs::MaxItems];
    struct Term { PxU8 item,a,b,next; } terms[32];
    struct Destination { PxU8 a,b,head; } destinations[32];
    PxU32 termCount=0,destinationCount=0;
    void add(PxU8 item,PxU8 a,PxU8 b,PxU8 memberA,PxU8 memberB)
    {
        PxU32 d=0;
        while(d<destinationCount && (destinations[d].a!=memberA || destinations[d].b!=memberB))++d;
        if(d==destinationCount)destinations[destinationCount++]={memberA,memberB,255};
        terms[termCount]={item,a,b,destinations[d].head};destinations[d].head=PxU8(termCount++);
    }
};
inline bool avbdPrepareFramePacket(const AvbdFrameAssemblyArgs& args,AvbdFramePacketScratch& scratch)
{
    scratch.termCount=scratch.destinationCount=0;const auto& dst=args.target;
    if(!args.materials || args.count>args.MaxItems || args.poseCount>args.MaxPoses ||
        !dst.gradient || !dst.K || !dst.M || !dst.energy || !dst.count || dst.count>16 ||
        dst.matrixRowStride<36*dst.count || dst.matrixRowStride>576)return false;
    for(PxU32 n=0;n<args.count;++n)
    {
        const auto& item=args.items[n];
        if(item.material>=args.materials->size())return false;
        if(item.members[0]==args.Outside && item.members[1]==args.Outside)return false;
        for(PxU32 b=0;b<2;++b)
            if(item.poses[b]>=args.poseCount || (item.members[b]!=args.Outside && item.members[b]>=dst.count))return false;
        for(PxU32 a=0;a<2;++a)if(item.members[a]!=args.Outside)
            for(PxU32 b=0;b<2;++b)if(item.members[b]!=args.Outside && item.members[a]<=item.members[b])
                scratch.add(PxU8(n),PxU8(a),PxU8(b),item.members[a],item.members[b]);
    }
    return true;
}
// One independent item, maximum two poses, one bounded material evaluation.
inline bool avbdEvaluateFramePacketItem(const AvbdFrameAssemblyArgs& args,PxU32 n,AvbdFramePacketScratch::Value& value)
{
    const auto& item=args.items[n];const auto material=(*args.materials)[item.material];
    const PxTransformT<double> bodies[2]={args.poses[item.poses[0]],args.poses[item.poses[1]]};
    if(item.members[0]==args.Outside || item.members[1]==args.Outside)
    {
        new (&value.endpoint) AvbdElasticFrameBlockContribution;
        avbdEvaluateElasticFrameBlock(material,bodies,item.members[0]==args.Outside?1:0,value.endpoint);
        return value.endpoint.valid;
    }
    new (&value.coupled) AvbdElasticFrameContribution;
    avbdEvaluateElasticFrameContribution(material,bodies,value.coupled);return value.coupled.valid;
}
// One exclusively owned symmetric matrix-entry pair. A device workgroup may assign
// 36 lanes per destination, with at most 32 terms per lane and no float atomics.
inline void avbdReduceFramePacketEntry(const AvbdFrameAssemblyArgs& args,const AvbdFramePacketScratch& scratch,
    PxU32 destination,PxU32 i,PxU32 j)
{
    const auto& dst=args.target;const auto& d=scratch.destinations[destination];
    if(d.a==d.b && j<i)return;
    double K=0,M=0,g=0;
    for(PxU32 t=d.head;t!=255;t=scratch.terms[t].next)
    {
        const auto& term=scratch.terms[t];const auto& item=args.items[term.item];
        const auto& value=scratch.values[term.item];
        if(item.members[0]==args.Outside || item.members[1]==args.Outside)
        {
            const PxU32 k=avbdFramePackedIndex(PxMin(i,j),PxMax(i,j));
            K+=value.endpoint.hessian[k];M+=value.endpoint.gaussNewton[k];
            if(j==i)g+=value.endpoint.gradient[i];
        }
        else
        {
            const PxU32 k=9*(i<3?i:i+3*term.a)+(j<3?j:j+3*term.b);
            const double sign=(term.a && i<3?-1:1)*(term.b && j<3?-1:1);
            K+=sign*value.coupled.hessian[k];M+=sign*value.coupled.gaussNewton[k];
            if(j==i && term.a==term.b)g+=value.coupled.gradient[6*term.a+i];
        }
    }
    const PxU32 out=d.a*dst.matrixRowStride+36*d.b+6*i+j;
    dst.K[out]+=K;dst.M[out]+=M;
    if(d.a!=d.b || i!=j)
    {
        const PxU32 transpose=d.b*dst.matrixRowStride+36*d.a+6*j+i;
        dst.K[transpose]+=K;dst.M[transpose]+=M;
    }
    if(j==i && d.a==d.b)dst.gradient[6*d.a+i]+=g;
}
// A CPU lane owns a contiguous row of the same symmetric destination. Hoist
// term classification and support indexing out of its six entry reductions;
// the two three-column runs expose contiguous inputs to SIMD. Device lanes
// may still use avbdReduceFramePacketEntry. Both have exclusive outputs.
inline void avbdReduceFramePacketRow(const AvbdFrameAssemblyArgs& args,const AvbdFramePacketScratch& scratch,
    PxU32 destination,PxU32 i)
{
    const auto& dst=args.target;const auto& d=scratch.destinations[destination];
    const PxU32 begin=d.a==d.b?i:0;
    double K[6]={},M[6]={},g=0;
    for(PxU32 t=d.head;t!=255;t=scratch.terms[t].next)
    {
        const auto& term=scratch.terms[t];const auto& item=args.items[term.item];
        const auto& value=scratch.values[term.item];
        if(item.members[0]==args.Outside || item.members[1]==args.Outside)
        {
            const PxU32 offset=avbdFramePackedIndex(i,i)-i;
            for(PxU32 j=begin;j<6;++j)
            {K[j]+=value.endpoint.hessian[offset+j];M[j]+=value.endpoint.gaussNewton[offset+j];}
            g+=value.endpoint.gradient[i];
        }
        else
        {
            const PxU32 row=9*(i<3?i:i+3*term.a);
            const double rotationSign=term.a && i<3?-1:1;
            const double translationSign=term.b?-rotationSign:rotationSign;
            for(PxU32 j=begin;j<3;++j)
            {K[j]+=translationSign*value.coupled.hessian[row+j];M[j]+=translationSign*value.coupled.gaussNewton[row+j];}
            for(PxU32 j=PxMax(3u,begin);j<6;++j)
            {K[j]+=rotationSign*value.coupled.hessian[row+j+3*term.b];M[j]+=rotationSign*value.coupled.gaussNewton[row+j+3*term.b];}
            if(term.a==term.b)g+=value.coupled.gradient[6*term.a+i];
        }
    }
    for(PxU32 j=begin;j<6;++j)
    {
        const PxU32 out=d.a*dst.matrixRowStride+36*d.b+6*i+j;
        dst.K[out]+=K[j];dst.M[out]+=M[j];
        if(d.a!=d.b || i!=j)
        {
            const PxU32 transpose=d.b*dst.matrixRowStride+36*d.a+6*j+i;
            dst.K[transpose]+=K[j];dst.M[transpose]+=M[j];
        }
    }
    if(d.a==d.b)dst.gradient[6*d.a+i]+=g;
}
inline void avbdAssembleFramePacket(AvbdFrameAssemblyArgs& args)
{
    args.valid=false;AvbdFramePacketScratch scratch;
    if(!avbdPrepareFramePacket(args,scratch))return;
    for(PxU32 n=0;n<args.count;++n)
        if(!avbdEvaluateFramePacketItem(args,n,scratch.values[n]))return;
    // All contributions are complete before any destination is reduced.
    for(PxU32 d=0;d<scratch.destinationCount;++d)
        for(PxU32 i=0;i<6;++i)avbdReduceFramePacketRow(args,scratch,d,i);
    double energy=0;
    for(PxU32 n=0;n<args.count;++n)
    {
        const auto& item=args.items[n];
        energy+=item.members[0]==args.Outside || item.members[1]==args.Outside?
            scratch.values[n].endpoint.energy:scratch.values[n].coupled.energy;
    }
    *args.target.energy+=energy;args.valid=true;
}

// Scratch lives for one assembly call only. Flush at material-run boundaries
// and before any pose change; no cross-trial or cross-color transform cache.
class AvbdFrameAssemblyPacket
{
    AvbdFrameAssemblyArgs mArgs;
    const KernelTable& mKernels;
    const PxU32* mMembers;
    PxU32 mBodyIds[AvbdFrameAssemblyArgs::MaxPoses];
    bool mValid=true;
public:
    AvbdFrameAssemblyPacket(const AvbdElasticFrameMaterials& materials,const KernelTable& kernels,
        const PxU32* members,const AvbdFrameAssemblyTarget& target) : mKernels(kernels),mMembers(members)
    {mArgs.materials=&materials;mArgs.target=target;}
    bool flush()
    {
        if(!mValid)return false;
        if(mArgs.count){mKernels.launch(AVBD_ELASTIC_FRAME_BATCH,&mArgs);mValid=mArgs.valid;}
        mArgs.count=mArgs.poseCount=0;return mValid;
    }
    template<class Bodies> bool append(PxU32 material,PxU32 bodyA,PxU32 bodyB,const Bodies& bodies)
    {
        if(!mValid || (mArgs.count==mArgs.MaxItems && !flush()))return false;
        AvbdFrameAssemblyItem item;item.material=material;const PxU32 ids[2]={bodyA,bodyB};
        for(PxU32 end=0;end<2;++end)
        {
            PxU32 slot=0;while(slot<mArgs.poseCount && mBodyIds[slot]!=ids[end])++slot;
            if(slot==mArgs.poseCount)
            {
                if(slot==mArgs.MaxPoses){mValid=false;return false;}
                const auto& body=bodies[ids[end]];
                mBodyIds[slot]=ids[end];mArgs.poses[slot]=avbdElasticFrameDouble(PxTransform(body.p,body.q));++mArgs.poseCount;
            }
            item.poses[end]=PxU8(slot);item.members[end]=mArgs.Outside;
            for(PxU32 s=0;s<mArgs.target.count;++s)if(mMembers[s]==ids[end])item.members[end]=PxU8(s);
        }
        mArgs.items[mArgs.count++]=item;return true;
    }
};
} }
#endif
