// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_MATERIAL_H
#define DY_AVBD_CONTACT_MATERIAL_H

#include "foundation/PxArray.h"
#include "foundation/PxVec3.h"
#include "foundation/PxBitUtils.h"
#if PX_X64
#include <emmintrin.h>
#endif

namespace physx { namespace Dy {

// The projection scale also bounds the tangential Jacobian at fixed load:
// identity in stick, rank-one times bound/length in slip, zero on release.
PX_FORCE_INLINE PxReal avbdProjectCoulomb(PxReal Fn, PxReal mu, PxReal& Ft0, PxReal& Ft1)
{
	if(mu <= 0.0f || Fn == 0.0f)
	{
		Ft0 = 0.0f;
		Ft1 = 0.0f;
		return 0.f;
	}
	const PxReal bound = PxAbs(Fn) * mu;
	const PxReal len = PxSqrt(Ft0 * Ft0 + Ft1 * Ft1);
	if(len > bound && len > 1.0e-20f)
	{
		const PxReal s = bound / len;
		Ft0 *= s;
		Ft1 *= s;
		return s;
	}
	return 1.f;
}

struct AvbdMaterialRow
{
	PxU32 rowIndex;
	// One common finalized-pose application point, expressed as a world arm
	// from each endpoint COM.  Equal/opposite material impulses at distinct
	// persistent AL anchors would manufacture an internal couple.
	PxVec3 armA;
	PxVec3 armB;
	// Normal load already supplied by the accepted OGC pose objective. It
	// contributes Coulomb capacity and the public maxImpulse budget, but is not
	// scattered a second time by the velocity complementarity solve.
	PxReal positionNormalImpulse;
	PxReal normalImpulse;
	PxReal normalCapacity;
	PxReal tangentImpulse0;
	PxReal tangentImpulse1;
	PxReal normalTarget;
	PxReal normalResponse;
	PxReal maxNormalImpulse;
	PxReal tangentStep;
    // Inverse of the 2D tangential mobility in the frozen contact frame.
    PxVec3 tangentInverse;
	PxU8 solveNormal;
	PxU8 solveTangent;
	// Joint-coupled fallback normal uses the COM projection until the generic
	// hard-joint Schur mobility is available; tangents retain spatial response.
	PxU8 normalLinearOnly;
};

struct AvbdMaterialVelocityDelta
{
	double linear[3];
	double angular[3];
	AvbdMaterialVelocityDelta()
	{
		linear[0] = linear[1] = linear[2] = 0.0;
		angular[0] = angular[1] = angular[2] = 0.0;
	}
	void addLinear(const PxVec3& value, double scale = 1.0)
	{
		linear[0] += double(value.x) * scale;
		linear[1] += double(value.y) * scale;
		linear[2] += double(value.z) * scale;
	}
	void addAngular(const PxVec3& value, double scale = 1.0)
	{
		angular[0] += double(value.x) * scale;
		angular[1] += double(value.y) * scale;
		angular[2] += double(value.z) * scale;
	}
	PxVec3 linearPx() const
	{
		return PxVec3(PxReal(linear[0]), PxReal(linear[1]), PxReal(linear[2]));
	}
	PxVec3 angularPx() const
	{
		return PxVec3(PxReal(angular[0]), PxReal(angular[1]), PxReal(angular[2]));
	}
};

struct AvbdMaterialAxis
{
    PxVec3 direction;
    PxVec3 angular[2]={PxVec3(0.f),PxVec3(0.f)};
    PxVec3 angularResponse[2]={PxVec3(0.f),PxVec3(0.f)};
    double baseSpeed;
};

struct AvbdMaterialPoint
{
    AvbdMaterialAxis axes[3];
    PxU32 state;
    PxReal friction;
};

// A span only shares writable endpoints. Every point retains its own arms,
// Jacobians, force limit, two tangents and original NP row identity.
struct AvbdMaterialPatch
{
    PxU32 begin, count, body[2];
    PxReal invMass[2];
};

struct AvbdMaterialResidual
{
    PxReal error = 0, scale = 1;
    void include(PxReal e, PxReal s) { error=PxMax(error,e); scale=PxMax(scale,s); }
};

struct AvbdContactMaterial
{
    enum { PointsPerPatch=4, PatchesPerTask=32, IslandPatchLimit=192 };
    PxArray<AvbdMaterialVelocityDelta> delta;
    PxArray<AvbdMaterialPoint> points;
    PxArray<AvbdMaterialPatch> patches;
    PxArray<PxU32> offsets, order;
    PxArray<PxU32> islandOffsets, islandPatches;
    PxArray<AvbdMaterialResidual> residual;
    PxU32 color=0, cursor=0, iteration=0, islandCursor=0;
    bool friction=false;

    void begin(PxU32 bodies)
    {
        delta.clear(); delta.resize(bodies);
        points.clear(); patches.clear(); offsets.clear(); order.clear(); residual.clear();
        islandOffsets.clear(); islandPatches.clear();
        color=cursor=iteration=islandCursor=0; friction=false;
    }
    void schedule(const PxArray<PxU32>& components, PxU32 workers);
};

// Compact independent components share a single job per sweep. Larger components
// expose disjoint endpoint pairs through colors; bits 0..31 have a bounded
// first-fit search and higher colors advance monotonically per endpoint.
inline void AvbdContactMaterial::schedule(const PxArray<PxU32>& components,PxU32 workers)
{
    if(patches.empty()) return;
    struct BodyColors {PxU32 mask=0, spill=32;};
    PxArray<BodyColors> used(delta.size());
    PxArray<PxU32> colors(patches.size(),PX_MAX_U32), counts(components.size(),0), starts(components.size());
    const auto root=[&](const AvbdMaterialPatch& patch)
    {return components[patch.body[patch.invMass[0]>0 ? 0 : 1]];};
    for(const auto& patch:patches) ++counts[root(patch)];
    PxU32 total=0;
    for(PxU32 n:counts) if(n<=IslandPatchLimit) total+=n;
    const PxU32 target=PxClamp(total/(2*PxClamp(workers,1u,32u)),8u,PxU32(IslandPatchLimit));
    islandPatches.resize(total);
    PxU32 offset=0;
    for(PxU32 i=0;i<counts.size();++i)
    {
        const PxU32 n=counts[i];
        if(!n || n>IslandPatchLimit) continue;
        if(islandOffsets.empty() || n>target || offset-islandOffsets.back()>target-n)
            islandOffsets.pushBack(offset);
        starts[i]=offset; offset+=n;
    }
    if(total) islandOffsets.pushBack(total);
    for(PxU32 i=0;i<patches.size();++i)
    {
        const auto& patch=patches[i];
        const PxU32 component=root(patch);
        if(counts[component]<=IslandPatchLimit)
        {
            islandPatches[starts[component]++]=i;
            continue;
        }
        PxU32 mask=0, spill=32;
        for(PxU32 s=0;s<2;++s) if(patch.invMass[s]>0)
        {
            const auto& body=used[patch.body[s]];
            mask|=body.mask; spill=PxMax(spill,body.spill);
        }
        const PxU32 c=(~mask) ? PxLowestSetBit(~mask) : spill;
        for(PxU32 s=0;s<2;++s) if(patch.invMass[s]>0)
        {
            auto& body=used[patch.body[s]];
            if(c<32) body.mask|=1u<<c;
            else body.spill=c+1;
        }
        colors[i]=c;
        if(offsets.size()<c+2) offsets.resize(c+2,0);
        ++offsets[c+1];
    }
    for(PxU32 i=1;i<offsets.size();++i) offsets[i]+=offsets[i-1];
    PxArray<PxU32> colorStarts(offsets);
    order.resize(patches.size()-total); residual.resize(patches.size());
    for(PxU32 i=0;i<patches.size();++i)
        if(colors[i]!=PX_MAX_U32) order[colorStarts[colors[i]]++]=i;
}

struct AvbdMaterialBatch
{
    AvbdContactMaterial& program;
    AvbdMaterialRow* states;
    PxU32 begin, end;
    enum Operation { Update, Residual, IndependentIslands } operation;
};

namespace contactMaterial {
PX_FORCE_INLINE double project(const PxVec3& j, const double* v)
{
    return j.x*v[0]+j.y*v[1]+j.z*v[2];
}

PX_FORCE_INLINE PxReal speed(const AvbdMaterialAxis& axis, const AvbdMaterialVelocityDelta* const v[2])
{
    double value=axis.baseSpeed;
    for(PxU32 s=0;s<2;++s) if(v[s])
    {
        const double dv=project(axis.direction,v[s]->linear)+project(axis.angular[s],v[s]->angular);
        value+=s ? -dv : dv;
    }
    return PxReal(value);
}

// Both tangent velocities observe the same pre-projection velocity. Evaluate
// them together in double precision before projecting the accumulated impulse.
PX_FORCE_INLINE void tangentSpeed(const AvbdMaterialPoint& point,
    const AvbdMaterialVelocityDelta* const v[2], PxReal& t0, PxReal& t1)
{
#if PX_X64
    const AvbdMaterialAxis& a=point.axes[1];
    const AvbdMaterialAxis& b=point.axes[2];
    __m128d value=_mm_set_pd(b.baseSpeed,a.baseSpeed);
    for(PxU32 s=0;s<2;++s) if(v[s])
    {
        __m128d dv=_mm_setzero_pd();
        for(PxU32 k=0;k<3;++k)
        {
            dv=_mm_add_pd(dv,_mm_mul_pd(_mm_set_pd(b.direction[k],a.direction[k]),_mm_set1_pd(v[s]->linear[k])));
            dv=_mm_add_pd(dv,_mm_mul_pd(_mm_set_pd(b.angular[s][k],a.angular[s][k]),_mm_set1_pd(v[s]->angular[k])));
        }
        value=s ? _mm_sub_pd(value,dv) : _mm_add_pd(value,dv);
    }
    double lanes[2]; _mm_storeu_pd(lanes,value);
    t0=PxReal(lanes[0]); t1=PxReal(lanes[1]);
#else
    t0=speed(point.axes[1],v); t1=speed(point.axes[2],v);
#endif
}

PX_FORCE_INLINE void scatter(const AvbdMaterialAxis& axis, const AvbdMaterialPatch& patch,
    AvbdMaterialVelocityDelta* const v[2], PxReal impulse)
{
    if(!PxIsFinite(impulse) || PxAbs(impulse)<1e-12f) return;
    for(PxU32 s=0;s<2;++s) if(v[s])
    {
        const double dj=s ? -double(impulse) : double(impulse);
        v[s]->addLinear(axis.direction,dj*patch.invMass[s]);
        v[s]->addAngular(axis.angularResponse[s],dj);
    }
}

template<bool Measure>
inline void visit(AvbdContactMaterial& p, const AvbdMaterialPatch& patch,
    AvbdMaterialRow* states, AvbdMaterialResidual& residual)
{
    AvbdMaterialVelocityDelta* v[2];
    for(PxU32 s=0;s<2;++s)
        v[s]=patch.invMass[s]>0 ? &p.delta[patch.body[s]] : NULL;
    for(PxU32 i=patch.begin;i<patch.begin+patch.count;++i)
    {
        const AvbdMaterialPoint& point=p.points[i];
        AvbdMaterialRow& state=states[point.state];
        if(state.solveNormal)
        {
            const PxReal gap=speed(point.axes[0],v)-state.normalTarget;
            if(Measure)
            {
                // At the authored impulse limit, a closing residual satisfies
                // the upper-bound KKT condition. A zero-width interval is fixed.
                PxReal e=state.normalImpulse>1e-8f ? PxAbs(gap) : PxMax(0.f,-gap);
                if(state.normalImpulse>=state.maxNormalImpulse)
                    e=state.maxNormalImpulse==0 ? 0.f : PxMax(0.f,gap);
                residual.include(e,PxMax(PxAbs(gap),PxAbs(state.normalTarget)));
            }
            else
            {
                const PxReal old=state.normalImpulse;
                state.normalImpulse=PxClamp(old-gap/state.normalResponse,0.f,state.maxNormalImpulse);
                state.normalCapacity=state.positionNormalImpulse+state.normalImpulse;
                scatter(point.axes[0],patch,v,state.normalImpulse-old);
            }
        }
        if(p.friction && state.solveTangent)
        {
            PxReal vt0,vt1;
            tangentSpeed(point,v,vt0,vt1);
            PxReal j0=state.tangentImpulse0-state.tangentStep*vt0;
            PxReal j1=state.tangentImpulse1-state.tangentStep*vt1;
            bool stick=false;
            if(!Measure && state.tangentInverse.x>0)
            {
                const PxVec3& inv=state.tangentInverse;
                const PxReal a=state.tangentImpulse0-inv.x*vt0-inv.y*vt1;
                const PxReal b=state.tangentImpulse1-inv.y*vt0-inv.z*vt1;
                const double radius=double(state.normalCapacity)*point.friction;
                stick=PxIsFinite(a) && PxIsFinite(b) && double(a)*a+double(b)*b<=radius*radius;
                if(stick){j0=a; j1=b;}
            }
            if(!stick) avbdProjectCoulomb(state.normalCapacity,point.friction,j0,j1);
            const PxReal d0=j0-state.tangentImpulse0, d1=j1-state.tangentImpulse1;
            if(Measure)
                residual.include(PxSqrt(d0*d0+d1*d1)/state.tangentStep,PxSqrt(vt0*vt0+vt1*vt1));
            else
            {
                state.tangentImpulse0=j0; state.tangentImpulse1=j1;
                scatter(point.axes[1],patch,v,d0); scatter(point.axes[2],patch,v,d1);
            }
        }
    }
}
}

inline void avbdContactMaterialBatch(void* data)
{
    AvbdMaterialBatch& args=*static_cast<AvbdMaterialBatch*>(data);
    auto& p=args.program;
    for(PxU32 i=args.begin;i<args.end;++i)
    {
        AvbdMaterialResidual result;
        if(args.operation==AvbdMaterialBatch::IndependentIslands)
        {
            const PxU32 first=p.islandOffsets[i],last=p.islandOffsets[i+1];
            for(PxU32 j=first;j<last;++j)
                contactMaterial::visit<false>(p,p.patches[p.islandPatches[j]],args.states,result);
            for(PxU32 j=first;j<last;++j)
            {
                const PxU32 index=p.islandPatches[j];
                AvbdMaterialResidual local;
                contactMaterial::visit<true>(p,p.patches[index],args.states,local);
                p.residual[index]=local;
            }
        }
        else if(args.operation==AvbdMaterialBatch::Residual)
        {
            const PxU32 index=p.order[i];
            contactMaterial::visit<true>(p,p.patches[index],args.states,result);
            p.residual[index]=result;
        }
        else
            contactMaterial::visit<false>(p,p.patches[p.order[i]],args.states,result);
    }
}

} }
#endif
