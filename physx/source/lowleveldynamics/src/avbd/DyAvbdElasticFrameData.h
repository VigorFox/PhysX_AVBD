// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_ELASTIC_FRAME_DATA_H
#define DY_AVBD_ELASTIC_FRAME_DATA_H
#include "DyAvbdElasticFramePotential.h"
#include "foundation/PxArray.h"
#include "foundation/PxHashMap.h"
#include "foundation/PxMemory.h"

namespace physx { namespace Dy {
struct AvbdElasticFrameGeometry
{
    PxVec3 positions[2]; // Authored PxReal coordinates are lossless here.
    PxQuatT<double> rotations[2]; // Normalize once, keep the cooked result.
};
struct AvbdElasticFrameParameters
{
    PxReal stiffness[6];
    double dampingOverDt[6];
    bool active() const
    {for(PxU32 i=0;i<6;++i)if(stiffness[i]>0 || dampingOverDt[i]>0)return true;return false;}
};
// Borrowed only for a synchronous Table launch. Indices, not these references,
// are retained in the step program; the pool is immutable during primal.
struct AvbdElasticFrameMaterial
{
    const AvbdElasticFrameGeometry& geometry;
    const AvbdElasticFrameParameters& parameters;
    const AvbdElasticFrameReference& reference;
};
// Owned preparation packet for standalone operator callers. Production rows
// use the shared pool below and do not allocate one of these per connection.
struct AvbdElasticFramePrepared
{
    AvbdElasticFrameGeometry geometry;
    AvbdElasticFrameParameters parameters;
    AvbdElasticFrameReference reference;
    operator AvbdElasticFrameMaterial() const {return {geometry,parameters,reference};}
};
inline PxTransformT<double> avbdElasticFrameDouble(const PxTransform& p)
{
    return PxTransformT<double>(PxVec3d(p.p.x,p.p.y,p.p.z),
        PxQuatT<double>(p.q.x,p.q.y,p.q.z,p.q.w).getNormalized());
}
inline bool avbdElasticFrameDataValid(const PxConstraintElasticFrameData& data,double h)
{
    if(!(h>0) || !PxIsFinite(h))return false;
    for(PxU32 i=0;i<2;++i)if(!data.frames[i].isSane())return false;
    for(PxU32 i=0;i<6;++i)
        if(!PxIsFinite(data.stiffness[i]) || data.stiffness[i]<0 ||
           !PxIsFinite(data.damping[i]) || data.damping[i]<0 || !PxIsFinite(double(data.damping[i])/h))return false;
    return true;
}
inline void avbdElasticFrameWorldFrames(const AvbdElasticFrameGeometry& geometry,
    const PxTransformT<double>* bodies,PxTransformT<double>* frames)
{
    for(PxU32 i=0;i<2;++i)
    {
        const auto& p=geometry.positions[i];
        frames[i]=bodies[i]*PxTransformT<double>(PxVec3d(p.x,p.y,p.z),geometry.rotations[i]);
    }
}
inline void avbdElasticFrameSections(const AvbdElasticFrameGeometry& geometry,
    const PxTransform* poses,PxTransformT<double>* bodies,PxTransformT<double>* frames)
{
    for(PxU32 i=0;i<2;++i)bodies[i]=avbdElasticFrameDouble(poses[i]);
    avbdElasticFrameWorldFrames(geometry,bodies,frames);
}
inline AvbdElasticFrameGeometry avbdCookFrameGeometry(const PxConstraintElasticFrameData& data)
{
    AvbdElasticFrameGeometry result;
    for(PxU32 i=0;i<2;++i)
    {result.positions[i]=data.frames[i].p;result.rotations[i]=avbdElasticFrameDouble(data.frames[i]).q;}
    return result;
}
inline AvbdElasticFrameParameters avbdCookFrameParameters(const PxConstraintElasticFrameData& data,double h)
{
    AvbdElasticFrameParameters result;
    for(PxU32 i=0;i<6;++i){result.stiffness[i]=data.stiffness[i];result.dampingOverDt[i]=double(data.damping[i])/h;}
    return result;
}
inline AvbdElasticFrameReference avbdFrameStepReference(const AvbdElasticFrameGeometry& geometry,const PxTransform* poses)
{
    PxTransformT<double> bodies[2],frames[2];avbdElasticFrameSections(geometry,poses,bodies,frames);
    return avbdPrepareFrameReference(frames);
}
inline bool avbdPrepareElasticFrame(AvbdElasticFramePrepared& m,
    const PxConstraintElasticFrameData& data,const PxTransform* poses,double h)
{
    if(!avbdElasticFrameDataValid(data,h) || !poses[0].isSane() || !poses[1].isSane())return false;
    m.geometry=avbdCookFrameGeometry(data);m.parameters=avbdCookFrameParameters(data,h);
    m.reference=avbdFrameStepReference(m.geometry,poses);return true;
}

// Keys contain explicit authored scalar bits, never struct padding, pointers,
// or rounded approximations. Exact signed zeros may occupy separate entries.
template<PxU32 N> struct AvbdFrameKey
{
    PxU32 words[N];
    void set(PxU32 i,PxReal value){PxMemCopy(&words[i],&value,sizeof(value));}
};
template<PxU32 N> struct AvbdFrameKeyHash
{
    PxU32 operator()(const AvbdFrameKey<N>& k) const
    {PxU32 h=2166136261u;for(PxU32 i=0;i<N;++i)h=PxHash<PxU32>()(h^k.words[i]);return h;}
    bool equal(const AvbdFrameKey<N>& a,const AvbdFrameKey<N>& b) const
    {for(PxU32 i=0;i<N;++i)if(a.words[i]!=b.words[i])return false;return true;}
};
struct AvbdElasticFrameInstance
{
    PxU32 geometry,parameters;
    AvbdElasticFrameReference reference;
};
class AvbdElasticFrameMaterials
{
    PxArray<AvbdElasticFrameInstance> mInstances;
    PxArray<AvbdElasticFrameGeometry> mGeometry;
    PxArray<AvbdElasticFrameParameters> mParameters;
    PxHashMap<AvbdFrameKey<14>,PxU32,AvbdFrameKeyHash<14> > mGeometryLookup;
    PxHashMap<AvbdFrameKey<12>,PxU32,AvbdFrameKeyHash<12> > mParameterLookup;
    double mDt=0;
public:
    AvbdElasticFrameMaterials() : mGeometryLookup(0),mParameterLookup(0) {}
    void beginStep(double dt)
    {
        mInstances.clear();mGeometry.clear();mParameters.clear();
        mGeometryLookup.clear();mParameterLookup.clear();mDt=dt;
    }
    PxU32 append(const PxConstraintElasticFrameData& data,const PxTransform* poses)
    {
        if(!avbdElasticFrameDataValid(data,mDt) || !poses[0].isSane() || !poses[1].isSane())return PX_MAX_U32;
        AvbdFrameKey<14> geometryKey;
        for(PxU32 i=0;i<2;++i)
        {
            const auto& f=data.frames[i];
            for(PxU32 j=0;j<3;++j)geometryKey.set(7*i+j,f.p[j]);
            geometryKey.set(7*i+3,f.q.x);geometryKey.set(7*i+4,f.q.y);
            geometryKey.set(7*i+5,f.q.z);geometryKey.set(7*i+6,f.q.w);
        }
        AvbdFrameKey<12> parameterKey;
        for(PxU32 i=0;i<6;++i){parameterKey.set(i,data.stiffness[i]);parameterKey.set(i+6,data.damping[i]);}
        const auto* geometry=mGeometryLookup.find(geometryKey);PxU32 gi;
        if(geometry)gi=geometry->second;
        else {gi=mGeometry.size();mGeometry.pushBack(avbdCookFrameGeometry(data));mGeometryLookup.insert(geometryKey,gi);}
        const auto* parameters=mParameterLookup.find(parameterKey);PxU32 pi;
        if(parameters)pi=parameters->second;
        else {pi=mParameters.size();mParameters.pushBack(avbdCookFrameParameters(data,mDt));mParameterLookup.insert(parameterKey,pi);}
        const PxU32 index=mInstances.size();
        const AvbdElasticFrameInstance instance={gi,pi,avbdFrameStepReference(mGeometry[gi],poses)};
        mInstances.pushBack(instance);return index;
    }
    PxU32 size() const{return mInstances.size();}
    bool empty() const{return mInstances.empty();}
    PxU32 geometryCount() const{return mGeometry.size();}
    PxU32 parameterCount() const{return mParameters.size();}
    AvbdElasticFrameMaterial operator[](PxU32 index) const
    {const auto& i=mInstances[index];return {mGeometry[i.geometry],mParameters[i.parameters],i.reference};}
    PxU64 bytes() const
    {return PxU64(size())*sizeof(AvbdElasticFrameInstance)+PxU64(geometryCount())*sizeof(AvbdElasticFrameGeometry)+PxU64(parameterCount())*sizeof(AvbdElasticFrameParameters);}
    PxU64 capacityBytes() const
    {return PxU64(mInstances.capacity())*sizeof(AvbdElasticFrameInstance)+PxU64(mGeometry.capacity())*sizeof(AvbdElasticFrameGeometry)+PxU64(mParameters.capacity())*sizeof(AvbdElasticFrameParameters);}
    // Hash bucket/index/key buffers at the fixed .75 load factor, excluding
    // allocator metadata/alignment. Compilation scratch, not numerical data.
    PxU64 internCapacityEstimate() const
    {
        const PxU64 g=mGeometryLookup.capacity(),p=mParameterLookup.capacity();
        return 4*(g+p)+(3*g/4)*(4+sizeof(AvbdFrameKey<14>)+4)+(3*p/4)*(4+sizeof(AvbdFrameKey<12>)+4);
    }
};
} }
#endif
