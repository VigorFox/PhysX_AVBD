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
	PxReal normalEffectiveMass;
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
    // One adjacent pair per coordinate can be loaded by either SIMD width.
    double coordinate[3][2] = {};
    void addLinear(const PxVec3& value, double scale=1.)
    {
        for(PxU32 k=0;k<3;++k) coordinate[k][0]+=double(value[k])*scale;
    }
    void addAngular(const PxVec3& value, double scale=1.)
    {
        for(PxU32 k=0;k<3;++k) coordinate[k][1]+=double(value[k])*scale;
    }
    PxVec3 linearPx() const {return PxVec3(PxReal(coordinate[0][0]),PxReal(coordinate[1][0]),PxReal(coordinate[2][0]));}
    PxVec3 angularPx() const {return PxVec3(PxReal(coordinate[0][1]),PxReal(coordinate[1][1]),PxReal(coordinate[2][1]));}
};

struct AvbdMaterialAxis
{
    // Lanes are linear A, angular A, linear B, angular B (including signs).
    // Gather uses J. Scatter uses M^-1 J^T prepared at the material epoch.
    PxReal jacobian[3][4], response[3][4];
    double baseSpeed;

    void setEndpoint(PxU32 side, const PxVec3& direction, const PxVec3& angular,
        const PxVec3& angularResponse, PxReal inverseMass)
    {
        const PxReal sign=side ? -1.f : 1.f;
        for(PxU32 k=0;k<3;++k)
        {
            jacobian[k][2*side]=sign*direction[k];
            response[k][2*side]=jacobian[k][2*side]*inverseMass;
            jacobian[k][2*side+1]=sign*angular[k];
            response[k][2*side+1]=inverseMass>0.f ? sign*angularResponse[k] : 0.f;
        }
    }
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
    bool converged() const { return error <= 1.0e-5f * scale; }
    void include(PxReal e, PxReal s) { error=PxMax(error,e); scale=PxMax(scale,s); }
};

struct AvbdMaterialComponent
{
    PxU32 begin, end;  // Patch span inside islandPatches.
    AvbdMaterialResidual residual;
    bool certified = false;
    bool friction = false;
};

struct AvbdMaterialJob
{
    PxU32 begin, end;
    AvbdMaterialResidual residual;
};

struct AvbdContactMaterial
{
    enum { PointsPerPatch=4, PatchesPerTask=32, IslandPatchLimit=192, MaxSweepsPerCheck=4, MaxColorsPerWave=4 };
    PxArray<AvbdMaterialVelocityDelta> delta;
    PxArray<AvbdMaterialPoint> points;
    PxArray<AvbdMaterialPatch> patches;
    // After scheduling: wave -> job, job -> ordered patches.
    PxArray<PxU32> offsets, order;
    PxArray<PxU32> colorOffsets, islandPatches;
    // Colored jobs index order; independent jobs index contiguous components.
    PxArray<AvbdMaterialJob> jobs;
    PxArray<AvbdMaterialComponent> islands;
    PxU32 color=0, cursor=0, iteration=0, islandCursor=0;
    PxU8 sweepsPerCheck=1, coloredSweeps=0;
    // Separate convergence checks from the bounded per-job update window.
    PxU32 nextCheck=1;
    bool checksResidual() const {return iteration+sweepsPerCheck==nextCheck;}
    bool friction=false;

    void begin(PxU32 bodies)
    {
        delta.clear(); delta.resize(bodies);
        points.clear(); patches.clear(); offsets.clear(); order.clear(); jobs.clear();
        colorOffsets.clear(); islandPatches.clear(); islands.clear();
        color=cursor=iteration=islandCursor=coloredSweeps=0;
        sweepsPerCheck=1; nextCheck=1; friction=false;
    }
    void schedule(const PxArray<PxU32>& components, PxU32 workers);
    void packColorWaves(const PxArray<PxU32>& colored);
    PxU32 coloredJobs() const { return offsets.empty() ? 0 : offsets.back(); }
};

// In a short run of colors, patches that touch the same writable endpoint
// stay in original GS order on one lane. Separate connected components of
// that run commute. Split the run if a component cannot fit a bounded lane.
inline void AvbdContactMaterial::packColorWaves(const PxArray<PxU32>& colored)
{
    if(colored.empty()) return;
    order.reserve(colored.size());
    offsets.pushBack(0);
    PxU32 firstInJob=0;
    const auto finishJob=[&](PxU32 end)
    {
        jobs.pushBack({firstInJob,end,{}});
        firstInJob=end;
    };
    struct Owner { PxU32 epoch=0, patch=0; };
    PxArray<Owner> owner(delta.size());
    PxArray<PxU32> parent(colored.size()), size(colored.size()), next(colored.size());
    PxU32 epoch=0;
    const auto root=[&](PxU32 i)
    {
        while(i!=parent[i])
        {
            parent[i]=parent[parent[i]];
            i=parent[i];
        }
        return i;
    };
    const auto connect=[&](PxU32 first, PxU32 last)
    {
        ++epoch;
        for(PxU32 i=first;i<last;++i)
        {
            parent[i]=i;
            size[i]=1;
            PxU32 joined=i;
            const auto& patch=patches[colored[i]];
            for(PxU32 s=0;s<2;++s) if(patch.invMass[s]>0)
            {
                Owner& entry=owner[patch.body[s]];
                if(entry.epoch==epoch)
                {
                    PxU32 other=root(entry.patch);
                    if(other==joined) continue;
                    if(size[other]>size[joined]) PxSwap(other,joined);
                    size[joined]+=size[other];
                    size[other]=0;
                    parent[other]=joined;
                    if(size[joined]>PatchesPerTask) return false;
                }
                entry.epoch=epoch;
                entry.patch=joined;
            }
        }
        return true;
    };
    for(PxU32 firstColor=0;firstColor+1<colorOffsets.size();)
    {
        const PxU32 first=colorOffsets[firstColor];
        PxU32 span=PxMin(PxU32(MaxColorsPerWave),colorOffsets.size()-1-firstColor);
        while(span>1 && !connect(first,colorOffsets[firstColor+span])) span=(span+1)/2;
        const PxU32 last=colorOffsets[firstColor+span];
        if(span==1)
        {
            for(PxU32 i=first;i<last;++i) order.pushBack(colored[i]);
            for(PxU32 i=first;i<last;i+=PatchesPerTask)
                finishJob(i+PxMin(PxU32(PatchesPerTask),last-i));
        }
        else
        {
            PxU32 write=first;
            for(PxU32 i=first;i<last;++i) if(size[i])
            {
                if(write-firstInJob+size[i]>PatchesPerTask) finishJob(write);
                next[i]=write;
                write+=size[i];
            }
            if(write!=firstInJob) finishJob(write);
            order.resize(last);
            for(PxU32 i=first;i<last;++i)
                order[next[root(i)]++]=colored[i];
        }
        firstColor+=span;
        offsets.pushBack(jobs.size());
    }
}

// Compact independent components share a job up to the next residual check. Larger components
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
    PxArray<PxU32> islandOffsets;
    islandPatches.resize(total);
    PxU32 offset=0,firstPatch=0;
    for(PxU32 i=0;i<counts.size();++i)
    {
        const PxU32 n=counts[i];
        if(!n || n>IslandPatchLimit) continue;
        if(islandOffsets.empty() || n>target || offset-firstPatch>target-n)
        {
            firstPatch=offset;
            islandOffsets.pushBack(islands.size());
        }
        starts[i]=offset;
        islands.pushBack({offset,offset+n,{}});
        offset+=n;
    }
    if(total) islandOffsets.pushBack(islands.size());
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
        if(colorOffsets.size()<c+2) colorOffsets.resize(c+2,0);
        ++colorOffsets[c+1];
    }
    for(PxU32 i=1;i<colorOffsets.size();++i) colorOffsets[i]+=colorOffsets[i-1];
    PxArray<PxU32> colorStarts(colorOffsets), colored(patches.size()-total);
    for(PxU32 i=0;i<patches.size();++i)
        if(colors[i]!=PX_MAX_U32) colored[colorStarts[colors[i]]++]=i;
    packColorWaves(colored);
    for(PxU32 i=1;i<islandOffsets.size();++i)
        jobs.pushBack({islandOffsets[i-1],islandOffsets[i],{}});
}

struct AvbdMaterialBatch
{
    AvbdContactMaterial& program;
    AvbdMaterialRow* states;
    PxU32 begin, end;
    enum Operation { Update, Residual, IndependentIslands } operation;
};

namespace contactMaterial {
// All backends evaluate the same four spatial coefficients and the same contact law.
struct ScalarSpatial
{
    double x[4];
    static ScalarSpatial load(const PxReal* f) {return {{f[0],f[1],f[2],f[3]}};}
    static ScalarSpatial join(const double* a,const double* b) {return {{a[0],a[1],b[0],b[1]}};}
    static ScalarSpatial splat(double v) {return {{v,v,v,v}};}
    void split(double* a,double* b) const {a[0]=x[0];a[1]=x[1];b[0]=x[2];b[1]=x[3];}
    double sum() const {return (x[0]+x[2])+(x[1]+x[3]);}
    ScalarSpatial operator+(const ScalarSpatial& r) const {return {{x[0]+r.x[0],x[1]+r.x[1],x[2]+r.x[2],x[3]+r.x[3]}};}
    ScalarSpatial operator*(const ScalarSpatial& r) const {return {{x[0]*r.x[0],x[1]*r.x[1],x[2]*r.x[2],x[3]*r.x[3]}};}
};

#if PX_X64
struct Sse2Spatial
{
    __m128d a,b;
    static Sse2Spatial load(const PxReal* f)
    {
        const __m128 x=_mm_loadu_ps(f);
        return {_mm_cvtps_pd(x),_mm_cvtps_pd(_mm_movehl_ps(x,x))};
    }
    static Sse2Spatial join(const double* a,const double* b) {return {_mm_loadu_pd(a),_mm_loadu_pd(b)};}
    static Sse2Spatial splat(double v) {return {_mm_set1_pd(v),_mm_set1_pd(v)};}
    void split(double* lo,double* hi) const {_mm_storeu_pd(lo,a);_mm_storeu_pd(hi,b);}
    double sum() const
    {
        const __m128d pair=_mm_add_pd(a,b);
        return _mm_cvtsd_f64(_mm_add_sd(pair,_mm_unpackhi_pd(pair,pair)));
    }
    Sse2Spatial operator+(const Sse2Spatial& r) const {return {_mm_add_pd(a,r.a),_mm_add_pd(b,r.b)};}
    Sse2Spatial operator*(const Sse2Spatial& r) const {return {_mm_mul_pd(a,r.a),_mm_mul_pd(b,r.b)};}
};
typedef Sse2Spatial NativeSpatial;
#else
typedef ScalarSpatial NativeSpatial;
#endif

template<class V> struct PatchVelocity
{
    V coordinate[3];
    AvbdMaterialVelocityDelta zero;
    AvbdMaterialVelocityDelta *a,*b;

    PatchVelocity(AvbdContactMaterial& p,const AvbdMaterialPatch& patch)
    {
        a=patch.invMass[0]>0 ? &p.delta[patch.body[0]] : &zero;
        b=patch.invMass[1]>0 && patch.body[1]!=patch.body[0] ? &p.delta[patch.body[1]] : &zero;
        for(PxU32 k=0;k<3;++k) coordinate[k]=V::join(a->coordinate[k],b->coordinate[k]);
    }

    PxReal speed(const AvbdMaterialAxis& axis) const
    {
        const V dot=V::load(axis.jacobian[0])*coordinate[0]+V::load(axis.jacobian[1])*coordinate[1]+V::load(axis.jacobian[2])*coordinate[2];
        return PxReal(axis.baseSpeed+dot.sum());
    }

    void scatter(const AvbdMaterialAxis& axis,PxReal impulse)
    {
        if(!PxIsFinite(impulse) || PxAbs(impulse)<1e-12f) return;
        const V force=V::splat(impulse);
        for(PxU32 k=0;k<3;++k) coordinate[k]=coordinate[k]+V::load(axis.response[k])*force;
    }

    void store()
    {
        for(PxU32 k=0;k<3;++k) coordinate[k].split(a->coordinate[k],b->coordinate[k]);
    }
};

template<bool Measure,bool Friction,class V>
inline void visit(AvbdContactMaterial& p, const AvbdMaterialPatch& patch,
    AvbdMaterialRow* states, AvbdMaterialResidual& residual)
{
    PatchVelocity<V> v(p,patch);
    for(PxU32 i=patch.begin;i<patch.begin+patch.count;++i)
    {
        const AvbdMaterialPoint& point=p.points[i];
        AvbdMaterialRow& state=states[point.state];
        if(state.solveNormal)
        {
            const PxReal gap=v.speed(point.axes[0])-state.normalTarget;
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
                state.normalImpulse=PxClamp(old-gap*state.normalEffectiveMass,0.f,state.maxNormalImpulse);
                state.normalCapacity=state.positionNormalImpulse+state.normalImpulse;
                v.scatter(point.axes[0],state.normalImpulse-old);
            }
        }
        if(Friction && state.solveTangent)
        {
            const PxReal vt0=v.speed(point.axes[1]), vt1=v.speed(point.axes[2]);
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
                v.scatter(point.axes[1],d0); v.scatter(point.axes[2],d1);
            }
        }
    }
    if(!Measure) v.store();
}
}

#include "DyAvbdSingleContact.inl"

template<class V,bool Friction> inline void avbdRunContactMaterialBatch(void* data)
{
    AvbdMaterialBatch& args=*static_cast<AvbdMaterialBatch*>(data);
    auto& p=args.program;
    for(PxU32 i=args.begin;i<args.end;++i)
    {
        AvbdMaterialResidual result;
        if(args.operation==AvbdMaterialBatch::IndependentIslands)
        {
            auto& job=p.jobs[p.coloredJobs()+i];
            // Once verified, a disconnected component has no external velocity
            // writes. Its residual stays valid until the contact law changes.
            for(PxU32 id=job.begin; id<job.end; ++id)
            {
                AvbdMaterialComponent& local=p.islands[id];
                if(local.friction!=Friction)
                {
                    local.certified=false;
                    local.friction=Friction;
                }
                if(!local.certified)
                {
                    PxU32 firstSweep=0;
                    if(Friction && !p.iteration && local.end-local.begin==1)
                    {
                        const AvbdMaterialPatch& patch=p.patches[p.islandPatches[local.begin]];
                        if(patch.count==1 && contactMaterial::solveSinglePoint<V>(p,patch,args.states))
                            firstSweep=1;
                    }
                    for(PxU32 sweep=firstSweep; sweep<p.sweepsPerCheck; ++sweep)
                        for(PxU32 j=local.begin; j<local.end; ++j)
                            contactMaterial::visit<false,Friction,V>(p,p.patches[p.islandPatches[j]],args.states,result);
                    if(p.checksResidual())
                    {
                        AvbdMaterialResidual measured;
                        for(PxU32 j=local.begin; j<local.end; ++j)
                            contactMaterial::visit<true,Friction,V>(p,p.patches[p.islandPatches[j]],args.states,measured);
                        local.residual=measured;
                        local.certified=measured.converged();
                    }
                }
                if(p.checksResidual()) result.include(local.residual.error,local.residual.scale);
            }
            if(p.checksResidual()) job.residual=result;
        }
        else if(args.operation==AvbdMaterialBatch::Residual)
        {
            auto& job=p.jobs[i];
            for(PxU32 j=job.begin;j<job.end;++j)
                contactMaterial::visit<true,Friction,V>(p,p.patches[p.order[j]],args.states,result);
            job.residual=result;
        }
        else
            for(PxU32 j=p.jobs[i].begin;j<p.jobs[i].end;++j)
                contactMaterial::visit<false,Friction,V>(p,p.patches[p.order[j]],args.states,result);
    }
}

template<class V> inline void avbdContactMaterialBatchImpl(void* data)
{
    if(static_cast<AvbdMaterialBatch*>(data)->program.friction) avbdRunContactMaterialBatch<V,true>(data);
    else avbdRunContactMaterialBatch<V,false>(data);
}

inline void avbdContactMaterialBatch(void* data) {avbdContactMaterialBatchImpl<contactMaterial::NativeSpatial>(data);}

} }
#endif
