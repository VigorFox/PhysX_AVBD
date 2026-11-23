// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_LOCAL_WORK_H
#define DY_AVBD_LOCAL_WORK_H
#include "DyAvbdNonlinearBlock.h"
#include "DyAvbdNodalShell.h"
#include "foundation/PxArray.h"
#include "foundation/PxAtomic.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec4.h"
#include "foundation/PxUserAllocated.h"
namespace physx { namespace Dy {
// Numerical scratch for one color, never a second pose/dual owner. The same
// leaf functions serve inline and dispatcher execution of a Rigid6 update.
struct AvbdLocalPartial
{
    AvbdNonlinearBlock model;
    double energy=0;
    bool scalar=true,valid=true;
    void merge(const AvbdLocalPartial& a,const AvbdLocalPartial& b,bool trial)
    {
        energy=a.energy+b.energy;scalar=a.scalar && b.scalar;valid=a.valid && b.valid;
        if(!trial)
        {
            for(PxU32 k=0;k<36;++k){model.K[k]=a.model.K[k]+b.model.K[k];model.M[k]=a.model.M[k]+b.model.M[k];}
            for(PxU32 k=0;k<6;++k)model.gradient[k]=a.model.gradient[k]+b.model.gradient[k];
        }
    }
};
// Additive local terms. The first leaf includes inertia. The block solves
// and checks its single candidate only after all leaf contributions arrive.
struct AvbdNodalAssembly
{
    AvbdNodalPlaneRange plane;
    AvbdNodalVec3 gradient=AvbdNodalVec3(0),polynomialGradient=AvbdNodalVec3(0),cubic=AvbdNodalVec3(0);
    AvbdNodalMat33 metric=AvbdNodalMat33(PxZero),polynomialMetric=AvbdNodalMat33(PxZero);
    AvbdNodalReal quartic=0,nonlinearEnergy=0;
    PX_FORCE_INLINE bool decreases(const AvbdNodalVec3& accepted,AvbdNodalReal nextEnergy) const
    {
        const auto length2=accepted.magnitudeSquared();
        const auto change=accepted.dot(polynomialGradient+polynomialMetric*accepted*AvbdNodalReal(.5))+
            length2*(cubic.dot(accepted)+quartic*length2)+(nextEnergy-nonlinearEnergy);
        return gradient.dot(accepted)<0 && change<=AvbdNodalReal(1.e-4)*gradient.dot(accepted);
    }
    void add(const AvbdNodalAssembly& b)
    {
        plane.merge(b.plane);
        gradient+=b.gradient;metric+=b.metric;polynomialGradient+=b.polynomialGradient;
        polynomialMetric+=b.polynomialMetric;cubic+=b.cubic;quartic+=b.quartic;
        nonlinearEnergy+=b.nonlinearEnergy;
    }
};

struct AvbdNodalEnergy
{
    AvbdNodalReal energy=0;
    bool valid=true;
};
struct AvbdNodalTrial
{
    AvbdNodalVec3 direction=AvbdNodalVec3(0);
    PxVec3 original=PxVec3(0),candidate=PxVec3(0);
    AvbdNodalReal fraction=1;
    bool done=false;
};
struct AvbdLocalBlockState
{
    PxU32 body=0,root=0,search=0;
    PxTransform original=PxTransform(PxIdentity),candidate=PxTransform(PxIdentity);
    PxVec3 dp=PxVec3(0),dth=PxVec3(0),gradientLinear=PxVec3(0),gradientAngular=PxVec3(0);
    PxReal metric[6][6]={},fraction=1,result=0;
    double initialEnergy=0;
    bool scalar=true,initialFeasible=true,done=false;
};
struct AvbdLocalWork : PxUserAllocated
{
    enum { RowLimit=256, SearchLimit=16 };
    enum Phase { Assemble, ReduceAssembly, Direction, Trial, ReduceTrial, Accept };
    struct Node { PxU32 state,begin,end,left,right; };
    PxArray<AvbdLocalBlockState> states;
    PxArray<Node> nodes;
    PxArray<AvbdLocalPartial> partials;
    PxArray<AvbdNodalAssembly> nodal;
    PxArray<AvbdNodalEnergy> nodalEnergy;
    PxArray<AvbdNodalTrial> nodalTrials;
    PxArray<PxU32> leaves;
    PxArray<PxArray<PxU32>> levels;
    PxU32 cursor=0,level=0,search=0;
    Phase phase=Assemble;
    void add(PxU32 body,PxU32 begin,PxU32 end)
    {
        AvbdLocalBlockState state;state.body=body;
        const PxU32 owner=states.size();PxArray<PxU32> frontier;
        for(PxU32 row=begin;row<end;)
        {
            const PxU32 stop=row+PxMin(PxU32(RowLimit),end-row),id=nodes.size();
            nodes.pushBack({owner,row,stop,0,0});leaves.pushBack(id);frontier.pushBack(id);row=stop;
        }
        PxU32 depth=0;
        while(frontier.size()>1)
        {
            if(depth==levels.size())levels.pushBack(PxArray<PxU32>());
            PxArray<PxU32> next;
            for(PxU32 k=0;k<frontier.size();k+=2)
            {
                if(k+1==frontier.size()){next.pushBack(frontier[k]);continue;}
                const PxU32 id=nodes.size();nodes.pushBack({owner,0,0,frontier[k],frontier[k+1]});
                levels[depth].pushBack(id);next.pushBack(id);
            }
            frontier=next;++depth;
        }
        state.root=frontier[0];states.pushBack(state);
    }
    PxU32 jobs() const
    {return phase==Assemble || phase==Trial?leaves.size():
        phase==ReduceAssembly || phase==ReduceTrial?levels[level].size():states.size();}
};
// Rare over-valent support. Each leaf writes its own incident load. The last
// child at a binary join publishes the sum, with at most 32 ancestors.
struct AvbdWideNomination : PxUserAllocated
{
    AvbdLocalWork tree;
    PxArray<PxU32> parent;
    PxArray<PxI32> pending;
    PxArray<PxVec4> load;
    bool reduce(PxU32 id,PxVec4 value)
    {
        load[id]=value;
        while(parent[id]!=PX_MAX_U32)
        {
            id=parent[id];if(PxAtomicDecrement(&pending[id]))return false;
            const auto& join=tree.nodes[id];load[id]=load[join.left]+load[join.right];
            pending[id]=2; // Reuse after the phase join.
        }
        return true;
    }
    void prepare()
    {
        parent.resize(tree.nodes.size(),PX_MAX_U32);
        pending.resize(tree.nodes.size(),2);
        load.resize(tree.nodes.size());
        for(const auto& level:tree.levels)for(PxU32 id:level)
        {
            const auto& n=tree.nodes[id];parent[n.left]=parent[n.right]=id;
        }
    }
};
} }
#endif
