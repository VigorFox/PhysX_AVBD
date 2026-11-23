// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_BOUNDED_RESPONSE_H
#define DY_AVBD_BOUNDED_RESPONSE_H
#include "DyAvbdCommonProgram.h"
#include "foundation/PxUserAllocated.h"
#include "DyKernelTable.h"
#include "DyAvbdForceExchange.h"
#include "DyAvbdNormalBounds.h"

namespace physx { namespace Dy {
// A transient executor for the existing frozen row program. No physical state.
// Every dispatched batch and every reduction node has an input-independent cap.
// The caller joins a wave before advance(); no waiting or task creation here.
struct AvbdBoundedResponse : public PxUserAllocated
{
    enum { LeafWidth=32, FanIn=8, BatchWidth=32, ColorLimit=32, SweepLimit=1024 };
    enum Status { Running, Converged, BudgetExhausted, Unsupported, NumericalFailure, RowsRequired };
    enum Kind { Inverse, Penalty, Save, Accumulate, Sum, Primal, Dual, Station, Root, Maximum,
        Remember, History, QrLeaf, QrSum, Coefficients, Propose, ForceBounds, Choose, Restore, SelectLeaf, SelectSum, Enable,
        AnchorLeaf, ExchangeLeaf, ExchangeSum, Exchange, VerifyRows, VerifySum };
    struct Node {PxU32 body,begin,count,level;};
    struct SumValue {double a[36],rhs[6],reaction[6];};
    struct Metric {double error,activeError,station,scale,meritSquared,baseScale;PxU32 failed;};
    struct Choice {double score;PxU32 row;};
    struct Wave {Kind kind;PxU32 begin,count;bool check;};
    struct QrValue {double r[25];};
    const AvbdCommonProgram& p;
    const AvbdCommonProgram* complete=NULL;
    const PxArray<PxU32>* completeMap=NULL;
    const PxArray<AvbdNormalBound>* normalBounds=NULL;
    struct Verification {double error;PxU32 missing,failed;};
    PxArray<PxU8> requests;
    PxArray<Verification> verification;
    PxArray<Node> verificationNodes;
    PxU32 sweepEnd=0,sweepLimit=SweepLimit;
    Status reducedStatus=Unsupported;
    PxArray<double> step,force,penalty,inverse;
    PxArray<PxU8> enabled,failed;
    PxArray<PxU32> colors,diagonal,roots,items;
    PxArray<Node> nodes,maxNodes;
    PxArray<SumValue> sums;
    PxArray<Choice> choices;PxArray<double> defects;
    PxArray<Choice> anchors;
    PxArray<Metric> metrics;
    PxArray<Wave> waves;
    PxArray<double> input,output,previousResidual,previousOutput,dr[4],dy[4];
    PxArray<Node> qrNodes;
    PxArray<QrValue> qr;
    double gamma[4]={};Metric originalMetric={},chosenMetric={};bool accept=false,extend=false;PxU32 historyCount=0;
    Status status=Unsupported;
    PxU32 wave=0,repeatWave=0,sweeps=0,colorCount=0,maxDepth=0;
    double error=0;bool equilibrate=false;PxU32 equilibriumSweeps=0,equilibriumEntries=0;

    // Compilation is currently a CPU setup operation, explicitly not an
    // end-to-end parallel qualification. All iteration work uses these waves.
    explicit AvbdBoundedResponse(const AvbdCommonProgram& program,const double* initialForce=NULL,
        const AvbdCommonProgram* full=NULL,const PxArray<PxU32>* map=NULL,PxU32 limit=SweepLimit,
        const double* initialStep=NULL,const PxArray<AvbdNormalBound>* bounds=NULL):p(program),complete(full),completeMap(map),normalBounds(bounds),sweepLimit(limit)
    {
        if(!p.valid || p.dimensions.empty() || !limit || limit>SweepLimit)return;
        if(complete && (!complete->valid || !map || map->size()!=complete->mu.size() ||
            complete->mu.size()>PX_MAX_U32/72 || complete->base.size()!=p.base.size() || complete->dimensions.size()!=p.dimensions.size()))return;
        if(complete)
        {
            for(PxU32 b=0;b<p.dimensions.size();++b)if(complete->dimensions[b]!=p.dimensions[b])return;
            for(PxU32 r=0;r<map->size();++r)if(complete->jOffsets[r+1]-complete->jOffsets[r]>4 ||
                ((*map)[r]!=PX_MAX_U32 && (*map)[r]>=p.mu.size()))return;
            if(normalBounds)
            {
                PxU32 covered=0;
                for(const auto& b:*normalBounds)
                {if(b.first!=covered || !b.count || b.count>LeafWidth || b.arity>4)return;covered+=b.count;}
                if(covered!=complete->mu.size())return;
            }
        }
        const PxU32 nb=p.dimensions.size(),nr=p.mu.size();
        // Guard all flattened indices before allocation, including sum trees.
        if(nb>PX_MAX_U32/72 || nr>PX_MAX_U32/72 || p.H.size()>PX_MAX_U32/4 || p.J.size()>PX_MAX_U32/4)return;
        for(PxU32 r=0;r<nr;++r)if(p.jOffsets[r+1]-p.jOffsets[r]>4)return;
        // The compiler supplies full H, not just a triangle. A missing reverse
        // edge would make the coloring unsafe for simultaneous readers/writers.
        for(const auto& h:p.H)if(h.row!=h.column)
        {
            const auto* end=p.H.begin()+p.hOffsets[h.column+1];
            const auto* other=std::lower_bound(p.H.begin()+p.hOffsets[h.column],end,h.row,
                [](const AvbdCommonHBlock& entry,PxU32 column){return entry.column<column;});
            if(other==end || other->column!=h.row)return;
        }
        colors.resize(nb,PX_MAX_U32);diagonal.resize(nb,PX_MAX_U32);roots.resize(nb);
        for(PxU32 b=0;b<nb;++b)
        {
            if(p.dimensions[b]!=3 && p.dimensions[b]!=6)return;
            PxU32 used=0;
            auto mark=[&](PxU32 other){if(other<b)used|=PxU32(1)<<colors[other];};
            for(PxU32 e=p.hOffsets[b];e<p.hOffsets[b+1];++e)
            {mark(p.H[e].column);if(p.H[e].column==b)diagonal[b]=e;}
            if(diagonal[b]==PX_MAX_U32)return;
            for(PxU32 e=p.transposeOffsets[b];e<p.transposeOffsets[b+1];++e)
            {const PxU32 r=p.J[p.transposeEntries[e]].row;for(PxU32 j=p.jOffsets[r];j<p.jOffsets[r+1];++j)mark(p.J[j].column);}
            PxU32 c=0;while(c<ColorLimit && (used&(PxU32(1)<<c)))++c;
            if(c==ColorLimit)return;colors[b]=c;colorCount=std::max(colorCount,c+1);
            const PxU32 count=p.hOffsets[b+1]-p.hOffsets[b]+p.transposeOffsets[b+1]-p.transposeOffsets[b];
            PxU32 first=nodes.size(),width=0,level=0;
            for(PxU32 start=0;start<count;start+=LeafWidth)
            {Node node={b,start,std::min(PxU32(LeafWidth),count-start),0};nodes.pushBack(node);++width;}
            while(width>1)
            {
                const PxU32 next=nodes.size();++level;
                for(PxU32 start=0;start<width;start+=FanIn)
                {Node node={b,first+start,std::min(PxU32(FanIn),width-start),level};nodes.pushBack(node);}
                first=next;width=nodes.size()-next;
            }
            roots[b]=first;maxDepth=std::max(maxDepth,level);
        }
        step.resize(p.base.size(),0);force.resize(p.offset.size(),0);penalty.resize(force.size(),1);
        if(initialStep)for(PxU32 i=0;i<step.size();++i)
        {if(!std::isfinite(initialStep[i])){status=NumericalFailure;return;}step[i]=initialStep[i];}
        if(initialForce)for(PxU32 i=0;i<force.size();++i)
        {if(!std::isfinite(initialForce[i])){status=NumericalFailure;return;}force[i]=initialForce[i];}
        inverse.resize(36*nb,0);
        enabled.resize(nr,0);failed.resize(nb,0);sums.resize(nodes.size());choices.resize(nodes.size());defects.resize(nr,0);anchors.resize(nb);
        for(PxU32 r=0;r<nr;++r)
        {
            bool contact=p.lower[3*r]==0;
            if(!p.coulomb[r])for(PxU32 q=1;q<3;++q)contact=contact && p.lower[3*r+q]==0 && p.upper[3*r+q]==0;
            enabled[r]=!contact;
            for(PxU32 q=0;q<3;++q)enabled[r]=enabled[r] || force[3*r+q]!=0;
        }
        PxU32 first=0,width=nb+nr,metricCount=width;
        while(width>1)
        {
            const PxU32 next=metricCount;
            for(PxU32 start=0;start<width;start+=FanIn)
            {Node node={metricCount++,first+start,std::min(PxU32(FanIn),width-start),0};maxNodes.pushBack(node);}
            first=next;width=metricCount-next;
        }
        metrics.resize(metricCount);
        const PxU32 size=step.size()+force.size();input.resize(size);output.resize(size);
        previousResidual.resize(size);previousOutput.resize(size);
        for(PxU32 k=0;k<4;++k){dr[k].resize(size,0);dy[k].resize(size,0);}
        for(PxU32 start=0;start<size;start+=LeafWidth)
        {Node node={0,start,std::min(PxU32(LeafWidth),size-start),0};qrNodes.pushBack(node);}
        first=0;width=qrNodes.size();PxU32 qrDepth=0;
        while(width>1)
        {
            const PxU32 next=qrNodes.size();++qrDepth;
            for(PxU32 start=0;start<width;start+=FanIn)
            {Node node={0,first+start,std::min(PxU32(FanIn),width-start),qrDepth};qrNodes.pushBack(node);}
            first=next;width=qrNodes.size()-next;
        }
        qr.resize(qrNodes.size());
        auto range=[&](Kind kind,PxU32 count,bool check=false)
        {Wave w={kind,items.size(),count,check};for(PxU32 i=0;i<count;++i)items.pushBack(i);waves.pushBack(w);};
        // Only pairs of single-support Coulomb rows can use this exchange.
        // This is structural scheduling, never a filter on physical rows.
        PxArray<PxU32> exchangeCount(nb,0);
        for(PxU32 r=0;r<nr;++r)if(p.coulomb[r] && p.jOffsets[r+1]-p.jOffsets[r]==1)
            ++exchangeCount[p.J[p.jOffsets[r]].column];
        range(Inverse,nb);if(nr)range(Penalty,nr);repeatWave=waves.size();
        range(Save,(size+LeafWidth-1)/LeafWidth);
        for(PxU32 c=0;c<colorCount;++c)
        {
            for(PxU32 level=0;level<=maxDepth;++level)
            {
                Wave w={level?Sum:Accumulate,items.size(),0,false};
                for(PxU32 i=0;i<nodes.size();++i)if(nodes[i].level==level && colors[nodes[i].body]==c && nodes[roots[nodes[i].body]].level)
                {items.pushBack(i);++w.count;}waves.pushBack(w);
            }
            Wave w={Primal,items.size(),0,false};
            for(PxU32 b=0;b<nb;++b)if(colors[b]==c){items.pushBack(b);++w.count;}waves.pushBack(w);
        }
        if(nr)range(Dual,nr);
        for(PxU32 pass=0;nr && pass<2;++pass)
        {
            for(PxU32 level=0;level<=maxDepth;++level)
            {
                Wave w={level?ExchangeSum:(pass?ExchangeLeaf:AnchorLeaf),items.size(),0,false};
                for(PxU32 i=0;i<nodes.size();++i)if(nodes[i].level==level && exchangeCount[nodes[i].body]>1){items.pushBack(i);++w.count;}waves.pushBack(w);
            }
            Wave w={Exchange,items.size(),0,pass==0};
            for(PxU32 b=0;b<nb;++b)if(exchangeCount[b]>1){items.pushBack(b);++w.count;}waves.pushBack(w);
        }
        auto check=[&]()
        {
            for(PxU32 level=0;level<=maxDepth;++level)
            {
                Wave w={level?Sum:Accumulate,items.size(),0,true};
                for(PxU32 i=0;i<nodes.size();++i)if(nodes[i].level==level && nodes[roots[nodes[i].body]].level){items.pushBack(i);++w.count;}waves.pushBack(w);
            }
            range(Station,nb);if(nr)range(Root,nr);
            // Contiguous levels of the max tree; no O(N) convergence scan.
            PxU32 remaining=nb+nr,cursor=0;
            while(remaining>1)
            {
                const PxU32 next=(remaining+FanIn-1)/FanIn;Wave w={Maximum,items.size(),next,false};
                for(PxU32 i=0;i<next;++i)items.pushBack(cursor++);waves.pushBack(w);remaining=next;
            }
        };
        check();range(Remember,1);range(History,(size+LeafWidth-1)/LeafWidth);
        for(PxU32 level=0;level<=qrDepth;++level)
        {
            Wave w={level?QrSum:QrLeaf,items.size(),0,false};
            for(PxU32 i=0;i<qrNodes.size();++i)if(qrNodes[i].level==level){items.pushBack(i);++w.count;}waves.pushBack(w);
        }
        range(Coefficients,1);range(Propose,(size+LeafWidth-1)/LeafWidth);if(nr)range(ForceBounds,nr);
        check();range(Choose,1);range(Restore,(size+LeafWidth-1)/LeafWidth);
        if(nr)range(Root,nr);
        for(PxU32 level=0;nr && level<=maxDepth;++level)
        {
            Wave w={level?SelectSum:SelectLeaf,items.size(),0,false};
            for(PxU32 i=0;i<nodes.size();++i)if(nodes[i].level==level){items.pushBack(i);++w.count;}waves.pushBack(w);
        }
        if(nr)range(Enable,nr);
        // Empty levels carry no dependency. Initialization waves are nonempty,
        // so compacting the sweep leaves repeatWave unchanged.
        PxArray<Wave> compact;for(const auto& w:waves)if(w.count)compact.pushBack(w);waves.swap(compact);
        sweepEnd=waves.size();
        if(complete && complete->mu.size()>p.mu.size())
        {
            requests.resize(complete->mu.size(),0);
            PxU32 verificationWidth=normalBounds?normalBounds->size():(requests.size()+LeafWidth-1)/LeafWidth,verificationFirst=0;
            verification.resize(verificationWidth);range(VerifyRows,verificationWidth);
            while(verificationWidth>1)
            {
                const PxU32 next=verification.size(),start=verificationNodes.size();
                for(PxU32 i=0;i<verificationWidth;i+=FanIn)
                {Node n={verification.size(),verificationFirst+i,std::min(PxU32(FanIn),verificationWidth-i),0};
                    verificationNodes.pushBack(n);verification.pushBack({});}
                Wave w={VerifySum,items.size(),verificationNodes.size()-start,false};
                for(PxU32 i=start;i<verificationNodes.size();++i)items.pushBack(i);waves.pushBack(w);
                verificationFirst=next;verificationWidth=verification.size()-next;
            }
        }
        status=Running;
    }
    // Reuse a publication-local H-only certificate. Its owner freezes H and
    // the support graph for this object's lifetime; ONLY p.base may change.
    // This is a cold numerical restart: no previous solution/history/force is
    // carried into the next certificate. History overwrites its scratch on
    // the first sweep; Inverse and all reduction leaves overwrite theirs too.
    bool restartHOnly()
    {
        if(!p.valid || complete || !p.mu.empty() || waves.empty() || p.base.size()!=step.size())return false;
        for(double value:p.base)if(!std::isfinite(value))return false;
        for(double& value:step)value=0;
        for(PxU8& value:failed)value=0;
        wave=sweeps=historyCount=equilibriumSweeps=equilibriumEntries=0;
        error=0;equilibrate=accept=extend=false;originalMetric={};chosenMetric={};
        for(double& value:gamma)value=0;
        status=Running;return true;
    }
    // Streaming square root reduction, never A^T A. The leaf has at most 32
    // coordinates; a parent merges at most eight 5x5 triangular factors.
    static void qrAdd(QrValue& qrValue,double* row)
    {
        for(PxU32 j=0;j<5;++j)
        {
            if(row[j]==0)continue;
            const double length=std::hypot(qrValue.r[5*j+j],row[j]);if(!(length>0))continue;
            const double c=qrValue.r[5*j+j]/length,s=row[j]/length;
            for(PxU32 k=j;k<5;++k)
            {const double a=qrValue.r[5*j+k],b=row[k];qrValue.r[5*j+k]=c*a+s*b;row[k]=c*b-s*a;}
        }
    }
    static double merit(const Metric& m) {return m.meritSquared;}
    // Exact shared tangents use a normal exchange; incompatible sticking pairs
    // use a weak wrench coordinate. Both move force and pose together and keep
    // every original row. This is numerical preconditioning, not publication.
    double exchange(PxU32 a,PxU32 b,AvbdForceExchange* move=NULL) const
    {
        if(a==b || !enabled[b] || p.jOffsets[b+1]-p.jOffsets[b]!=1)return 0;
        const auto& ja=p.J[p.jOffsets[a]];const auto& jb=p.J[p.jOffsets[b]];
        if(ja.column!=jb.column)return 0;const PxU32 block=ja.column,n=p.dimensions[block];
        if(!p.sameTangent(a,b))
        {const auto candidate=avbdStickingExchange(p,inverse.begin()+36*block,step.begin()+p.starts[block],force.begin(),a,b);
            if(move)*move=candidate;return candidate.decrease;}
        double v[6]={},dx[6]={},gap=p.offset[3*a]-p.offset[3*b]+p.compliance[3*a]*force[3*a]-p.compliance[3*b]*force[3*b];
        for(PxU32 d=0;d<n;++d){v[d]=ja.values[d]-jb.values[d];gap+=v[d]*step[p.starts[block]+d];}
        double curvature=p.compliance[3*a]+p.compliance[3*b];
        for(PxU32 d=0;d<n;++d){for(PxU32 k=0;k<n;++k)dx[d]+=inverse[36*block+6*d+k]*v[k];curvature+=v[d]*dx[d];}
        if(curvature<0 || !std::isfinite(curvature))return 0;
        const double low=std::max(p.lower[3*a]-force[3*a],force[3*b]-p.upper[3*b]);
        const double high=std::min(p.upper[3*a]-force[3*a],force[3*b]-p.lower[3*b]);
        const double change=std::max(low,std::min(high,curvature>0?-gap/curvature:(gap>0?low:high)));
        if(!std::isfinite(change))return 0;
        const double decrease=std::max(0.,-change*(gap+.5*curvature*change));
        if(move)
        {
            *move={};move->decrease=decrease;move->force[0]=change;move->force[3]=-change;
            const double total=force[3*a]+force[3*b],weight=total>0?(force[3*a]+change)/total:0;
            for(PxU32 q=1;q<3;++q)
            {const double tangent=force[3*a+q]+force[3*b+q];move->force[q]=weight*tangent-force[3*a+q];move->force[q+3]=(1-weight)*tangent-force[3*b+q];}
            for(PxU32 d=0;d<n;++d)move->motion[d]=dx[d]*change;
        }
        return decrease;
    }
    PxU32 count() const {return status==Running?waves[wave].count:0;}
    void accumulate(PxU32 id,bool check)
    {
        const Node& node=nodes[id];const PxU32 b=node.body,n=p.dimensions[b],hc=p.hOffsets[b+1]-p.hOffsets[b];
        const bool fixed=check || equilibrate;
        SumValue v={};
        for(PxU32 entry=node.begin;entry<node.begin+node.count;++entry)
        {
            if(entry<hc)
            {
                const auto& h=p.H[p.hOffsets[b]+entry];
                if(!check && h.column==b)for(PxU32 d=0;d<36;++d)v.a[d]+=p.metricBlock(p.hOffsets[b]+entry).values[d];
                for(PxU32 d=0;d<n;++d)for(PxU32 k=0;k<p.dimensions[h.column];++k)
                    v.rhs[d]-=h.values[6*d+k]*step[p.starts[h.column]+k];
            }
            else
            {
                const auto& j=p.J[p.transposeEntries[p.transposeOffsets[b]+entry-hc]];
                if(!fixed && !enabled[j.row])continue;
                if(fixed && force[3*j.row]==0 && force[3*j.row+1]==0 && force[3*j.row+2]==0)continue;
                double projected[3]={},stiffness[3]={};
                if(!fixed)p.projected(j.row,penalty.begin(),step.begin(),force.begin(),projected,stiffness);
                for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<n;++d)
                {
                    const double reaction=j.values[6*q+d]*(fixed?force[3*j.row+q]:projected[q]);
                    v.rhs[d]+=reaction;v.reaction[d]+=reaction;
                    if(!fixed && stiffness[q]!=0)for(PxU32 k=0;k<n;++k)v.a[6*d+k]+=j.values[6*q+d]*stiffness[q]*j.values[6*q+k];
                }
            }
        }
        sums[id]=v;
    }
    const char* statusName() const
    {
        switch(status)
        {case Running:return "running";case Converged:return "converged";case BudgetExhausted:return "budget_exhausted";
         case NumericalFailure:return "numerical_failure";case RowsRequired:return "rows_required";default:return "unsupported";}
    }
    bool run(PxU32 begin,PxU32 end)
    {
        PX_ASSERT(status==Running && begin<=end && end<=count() && end-begin<=BatchWidth);
        // A bad launch must not turn into an unbounded kernel in release builds.
        if(status!=Running || begin>end || end>count() || end-begin>BatchWidth)return false;
        const Wave& w=waves[wave];
        for(PxU32 slot=begin;slot<end;++slot)
        {
            const PxU32 id=items[w.begin+slot];
            if(equilibrate && (w.kind==Dual || w.kind==ForceBounds || w.kind==AnchorLeaf || w.kind==ExchangeLeaf || w.kind==ExchangeSum || w.kind==Exchange))continue;
            if(w.kind==Inverse)
            {
                const PxU32 n=p.dimensions[id];
                for(PxU32 k=0;k<n;++k)
                {double rhs[6]={},x[6]={};rhs[k]=1;
                    if(!avbdSolveSmallBlock(p.metricBlock(diagonal[id]).values,rhs,x,n)){failed[id]=1;break;}
                    for(PxU32 d=0;d<n;++d)inverse[36*id+6*d+k]=x[d];}
            }
            else if(w.kind==Penalty)
            {
                p.rowPenalty(id,inverse.begin(),penalty.begin()+3*id);
            }
            else if(w.kind==VerifyRows)
            {
                Verification result={};
                const auto* bound=normalBounds?&(*normalBounds)[id]:NULL;
                if(bound && bound->nonnegative(step.begin(),p.starts))
                {verification[id]=result;continue;}
                const PxU32 first=bound?bound->first:id*LeafWidth;
                const PxU32 stop=bound?first+bound->count:std::min(requests.size(),first+PxU32(LeafWidth));
                for(PxU32 r=first;r<stop;++r)
                {
                    const PxU32 local=(*completeMap)[r];double f[3]={},rho[3],projected[3],stiffness[3],root[3];
                    // Selected rows were checked by Root in this sweep.
                    // Omitted rows have exactly zero force; query that root.
                    requests[r]=0;
                    if(local!=PX_MAX_U32 || complete->zeroReactionAdmissible(r,step.begin()))continue;
                    complete->rowPenalty(r,inverse.begin(),rho);
                    complete->projectedRow(r,rho,step.begin(),f,projected,stiffness,true,NULL,root);
                    double defect=0;for(PxU32 q=0;q<3;++q)
                    {const double v=std::abs(root[q])/rho[q];result.failed|=!std::isfinite(v) || !std::isfinite(f[q]);defect=std::max(defect,v);}
                    result.error=std::max(result.error,defect);
                    requests[r]=local==PX_MAX_U32 && defect>=1.e-10;result.missing+=requests[r];
                }
                verification[id]=result;
            }
            else if(w.kind==VerifySum)
            {
                const auto& node=verificationNodes[id];Verification result={};
                for(PxU32 i=node.begin;i<node.begin+node.count;++i)
                {const auto& child=verification[i];result.error=std::max(result.error,child.error);result.missing+=child.missing;result.failed|=child.failed;}
                verification[node.body]=result;
            }
            else if(w.kind==Dual || w.kind==Root)
            {
                double f[3],k[3],root[3];p.projected(id,penalty.begin(),step.begin(),force.begin(),f,k,true,NULL,w.kind==Root?root:NULL);
                if(w.kind==Root)
                {
                    Metric m={};m.scale=1;for(PxU32 q=0;q<3;++q)
                    {const double v=std::abs(root[q])/penalty[3*id+q];m.failed|=!std::isfinite(v);m.error=std::max(m.error,v);if(enabled[id]){m.activeError=std::max(m.activeError,v);m.meritSquared+=v*v;}}
                    metrics[p.dimensions.size()+id]=m;defects[id]=m.error;
                }
                else
                {
                    for(PxU32 q=0;q<3;++q)force[3*id+q]=enabled[id]?f[q]:0;
                }
            }
            else if(w.kind==Accumulate)
            {
                accumulate(id,w.check);
            }
            else if(w.kind==Sum)
            {
                const auto& node=nodes[id];SumValue v={};
                for(PxU32 i=node.begin;i<node.begin+node.count;++i)
                {for(PxU32 k=0;k<36;++k)v.a[k]+=sums[i].a[k];
                    for(PxU32 k=0;k<6;++k){v.rhs[k]+=sums[i].rhs[k];v.reaction[k]+=sums[i].reaction[k];}}
                sums[id]=v;
            }
            else if(w.kind==Primal || w.kind==Station)
            {
                // One leaf has at most 32 neighbors: fuse its local work with
                // the block operation. Same-color blocks cannot share an H/J
                // edge; Station only reads poses, so both launches are safe.
                if(!nodes[roots[id]].level)accumulate(roots[id],w.kind==Station);
                const auto& v=sums[roots[id]];const PxU32 n=p.dimensions[id],start=p.starts[id];double rhs[6]={},dx[6]={};
                for(PxU32 d=0;d<n;++d)rhs[d]=v.rhs[d]+p.base[start+d];
                if(w.kind==Primal)
                {
                    // H-only requests reuse the compiled physical diagonal;
                    // its factor does not change between these colored sweeps.
                    if(p.mu.empty() || equilibrate)for(PxU32 d=0;d<n;++d)
                        for(PxU32 k=0;k<n;++k)dx[d]+=inverse[36*id+6*d+k]*rhs[k];
                    else if(!avbdSolveSmallBlock(v.a,rhs,dx,n))failed[id]=1;
                    if(!failed[id])for(PxU32 d=0;d<n;++d)step[start+d]+=dx[d];
                }
                else
                {
                    Metric m={};m.scale=1;m.failed=failed[id];
                    for(PxU32 d=0;d<n;++d)
                    {
                        for(PxU32 k=0;k<n;++k)dx[d]+=inverse[36*id+6*d+k]*rhs[k];
                        m.failed|=!std::isfinite(dx[d]) || !std::isfinite(rhs[d]);m.error=std::max(m.error,std::abs(dx[d]));m.activeError=m.error;m.meritSquared+=dx[d]*dx[d];
                        m.station=std::max(m.station,std::abs(rhs[d]));m.scale=std::max(m.scale,std::max(std::abs(p.base[start+d]),std::abs(v.reaction[d])));
                        m.baseScale=std::max(m.baseScale,std::abs(p.base[start+d]));
                    }
                    metrics[id]=m;
                }
            }
            else if(w.kind==Maximum)
            {
                const auto& node=maxNodes[id];Metric m={};m.scale=1;
                for(PxU32 i=node.begin;i<node.begin+node.count;++i)
                {m.error=std::max(m.error,metrics[i].error);m.activeError=std::max(m.activeError,metrics[i].activeError);m.meritSquared+=metrics[i].meritSquared;m.station=std::max(m.station,metrics[i].station);
                    m.scale=std::max(m.scale,metrics[i].scale);m.baseScale=std::max(m.baseScale,metrics[i].baseScale);m.failed|=metrics[i].failed;}metrics[node.body]=m;
            }
            else if(w.kind==Remember)originalMetric=metrics.back();
            else if(w.kind==Save || w.kind==History || w.kind==Propose || w.kind==Restore)
            {
                const PxU32 coordinateEnd=std::min(input.size(),(id+1)*PxU32(LeafWidth));
                for(PxU32 i=id*LeafWidth;i<coordinateEnd;++i)
                {
                    if(equilibrate && i>=step.size() && (w.kind==Propose || w.kind==Restore))continue;
                    double& value=i<step.size()?step[i]:force[i-step.size()];const double scale=i<step.size()?1:penalty[i-step.size()];
                    if(w.kind==Save)input[i]=value/scale;
                    else if(w.kind==History)
                    {
                        output[i]=value/scale;const double residual=output[i]-input[i];
                        if(historyCount){dr[(historyCount-1)%4][i]=residual-previousResidual[i];dy[(historyCount-1)%4][i]=output[i]-previousOutput[i];}
                        else for(PxU32 k=0;k<4;++k){dr[k][i]=0;dy[k][i]=0;}
                        previousResidual[i]=residual;previousOutput[i]=output[i];
                    }
                    else if(w.kind==Propose)
                    {double v=output[i];for(PxU32 k=0;k<4;++k)v-=gamma[k]*dy[k][i];value=v*scale;}
                    else if(!accept)value=output[i]*scale;
                }
            }
            else if(w.kind==QrLeaf || w.kind==QrSum)
            {
                const Node& node=qrNodes[id];QrValue value={};
                if(w.kind==QrLeaf)
                    for(PxU32 i=node.begin;i<node.begin+node.count;++i)
                    {double row[5]={};for(PxU32 k=0;k<4;++k)row[k]=dr[k][i];row[4]=previousResidual[i];qrAdd(value,row);}
                else for(PxU32 child=node.begin;child<node.begin+node.count;++child)for(PxU32 r=0;r<5;++r)
                {double row[5];for(PxU32 k=0;k<5;++k)row[k]=qr[child].r[5*r+k];qrAdd(value,row);}
                qr[id]=value;
            }
            else if(w.kind==Coefficients)
            {
                // Twice reorthogonalize only this bounded root factor, newest
                // independent columns first, matching the existing secant rule.
                double q[4][5]={},R[4][4]={},rhs[4]={},g[4]={};PxU32 selected[4]={},rank=0;
                for(PxU32 k=0;k<4;++k)gamma[k]=0;
                for(PxU32 age=0;age<std::min(historyCount,PxU32(4));++age)
                {
                    const PxU32 k=(historyCount-1-age)%4;double v[5],norm=0;
                    for(PxU32 i=0;i<5;++i){v[i]=qr.back().r[5*i+k];norm+=v[i]*v[i];}
                    for(PxU32 pass=0;pass<2;++pass)for(PxU32 j=0;j<rank;++j)
                    {double a=0;for(PxU32 i=0;i<5;++i)a+=q[j][i]*v[i];R[j][rank]+=a;for(PxU32 i=0;i<5;++i)v[i]-=a*q[j][i];}
                    double length=0;for(double v0:v)length=std::hypot(length,v0);
                    if(!(length>1.e-8*std::sqrt(norm)) || !std::isfinite(length))
                    {for(PxU32 j=0;j<rank;++j)R[j][rank]=0;continue;}
                    R[rank][rank]=length;for(PxU32 i=0;i<5;++i){q[rank][i]=v[i]/length;rhs[rank]+=q[rank][i]*qr.back().r[5*i+4];}
                    selected[rank]=k;++rank;
                }
                for(PxU32 jj=rank;jj>0;--jj)
                {const PxU32 j=jj-1;double v=rhs[j];for(PxU32 k=j+1;k<rank;++k)v-=R[j][k]*g[k];g[j]=v/R[j][j];gamma[selected[j]]=g[j];}
            }
            else if(w.kind==ForceBounds)
            {
                for(PxU32 q=0;q<3;++q){const PxU32 i=3*id+q;force[i]=std::max(p.lower[i],std::min(p.upper[i],force[i]));}
                if(p.coulomb[id])
                {const double bound=p.mu[id]*std::max(0.,force[3*id]),length=std::hypot(force[3*id+1],force[3*id+2]);
                    if(length>bound)for(PxU32 q=1;q<3;++q)force[3*id+q]*=bound/length;}
            }
            else if(w.kind==Choose)
            {
                accept=!metrics.back().failed && merit(metrics.back())<merit(originalMetric);chosenMetric=accept?metrics.back():originalMetric;
                // Do not oversolve a reduced work set while an omitted row
                // dominates the original mixed defect. Final stopping still
                // checks every row at the same strict tolerance.
                extend=chosenMetric.error>std::max(chosenMetric.activeError,chosenMetric.station/chosenMetric.scale) && chosenMetric.error>=1.e-10;
            }
            else if(w.kind==SelectLeaf || w.kind==SelectSum)
            {
                const Node& node=nodes[id];Choice choice={0,PX_MAX_U32};
                if(extend)for(PxU32 i=node.begin;i<node.begin+node.count;++i)
                {
                    Choice trial={0,PX_MAX_U32};
                    if(w.kind==SelectSum)trial=choices[i];
                    else
                    {
                        const PxU32 b=node.body,hc=p.hOffsets[b+1]-p.hOffsets[b];if(i<hc)continue;
                        const auto& j=p.J[p.transposeEntries[p.transposeOffsets[b]+i-hc]];
                        if(enabled[j.row] || p.J[p.jOffsets[j.row]].column!=b)continue;
                        trial={defects[j.row],j.row};
                    }
                    if(trial.score>choice.score || (trial.score==choice.score && trial.row<choice.row))choice=trial;
                }
                choices[id]=choice;
            }
            else if(w.kind==Enable && extend && !enabled[id] && defects[id]>1.e-10)
            {
                const PxU32 first=p.jOffsets[id];
                if(first==p.jOffsets[id+1] || choices[roots[p.J[first].column]].row==id)enabled[id]=1;
            }
            else if(w.kind==AnchorLeaf || w.kind==ExchangeLeaf || w.kind==ExchangeSum)
            {
                const Node& node=nodes[id];Choice choice={0,PX_MAX_U32};
                for(PxU32 i=node.begin;i<node.begin+node.count;++i)
                {
                    Choice trial={0,PX_MAX_U32};
                    if(w.kind==ExchangeSum)trial=choices[i];
                    else
                    {
                        const PxU32 b=node.body,hc=p.hOffsets[b+1]-p.hOffsets[b];if(i<hc)continue;
                        const auto& j=p.J[p.transposeEntries[p.transposeOffsets[b]+i-hc]];const PxU32 r=j.row;
                        if(!enabled[r] || p.jOffsets[r+1]-p.jOffsets[r]!=1)continue;
                        if(w.kind==AnchorLeaf)
                        {
                            if(!(force[3*r]>0) || !p.coulomb[r])continue;
                            double f[3],k[3],root[3],defect=0;p.projected(r,penalty.begin(),step.begin(),force.begin(),f,k,true,NULL,root);
                            for(PxU32 q=0;q<3;++q)defect=std::max(defect,std::abs(root[q])/penalty[3*r+q]);
                            trial={defect,r};
                        }
                        else if(anchors[b].row!=PX_MAX_U32)trial={exchange(anchors[b].row,r),r};
                    }
                    if(trial.score>choice.score || (trial.score==choice.score && trial.row<choice.row))choice=trial;
                }
                choices[id]=choice;
            }
            else if(w.kind==Exchange)
            {
                if(w.check){anchors[id]=choices[roots[id]];continue;}
                const PxU32 a=anchors[id].row,b=choices[roots[id]].row;
                if(a==PX_MAX_U32 || b==PX_MAX_U32 || !(choices[roots[id]].score>0))continue;
                AvbdForceExchange move={};if(!(exchange(a,b,&move)>0))continue;
                for(PxU32 q=0;q<3;++q){force[3*a+q]+=move.force[q];force[3*b+q]+=move.force[q+3];}
                for(PxU32 d=0;d<p.dimensions[id];++d)step[p.starts[id]+d]+=move.motion[d];
            }
        }
        return true;
    }
    // Caller must complete every item in the current wave exactly once first.
    void advance()
    {
        if(status!=Running)return;
        if(wave>=sweepEnd)
        {
            if(++wave<waves.size())return;
            const auto& v=verification.back();error=std::max(error,v.error);
            status=v.failed?NumericalFailure:v.missing?RowsRequired:
                (v.error>=1.e-10?BudgetExhausted:reducedStatus);
            // A periodic separation query may see unconverged active roots.
            // Without missing rows, continue the SAME numerical iteration.
            if(!v.failed && !v.missing && reducedStatus==Running)
            {status=Running;wave=repeatWave;}
            return;
        }
        if(++wave<sweepEnd)return;
        equilibriumSweeps+=equilibrate;
        const Metric& m=chosenMetric;error=std::max(m.error,m.station/m.scale);++sweeps;historyCount=extend?0:historyCount+1;
        if(m.failed)status=NumericalFailure;
        // A force-free request also serves as the full-H publication certificate.
        // Preserve that oracle's original true-residual tolerance, including
        // tiny RHS values; the mixed program's scale floor is not sufficient.
        else if(error<1.e-10 && (!p.mu.empty() || m.station<=std::max(1.e-14,1.e-12*m.baseScale)))status=Converged;
        else if(sweeps==sweepLimit)status=BudgetExhausted;
        else
        {
            // Once original contact roots are small, AL penalties can damp a
            // tiny remaining displacement while leaving a large force defect.
            // Balance the SAME Hx=base+J^T f with colored physical blocks and
            // fixed f; recheck every original row after every sweep. If any
            // root worsens, resume AL within this unchanged total budget.
            const bool next=!p.mu.empty() && m.error<1.e-10;
            if(next!=equilibrate)historyCount=0;
            if(next && !equilibrate)++equilibriumEntries;
            equilibrate=next;wave=repeatWave;
        }
        if(!verification.empty() && (status==Converged || status==BudgetExhausted || status==Running))
        {reducedStatus=status;status=Running;wave=sweepEnd;}
    }
    bool solve(const KernelTable& table)
    {
        while(status==Running)
        {for(PxU32 begin=0;begin<count();begin+=BatchWidth)
            {AvbdBoundedBatchArgs args={this,begin,std::min(begin+PxU32(BatchWidth),count()),false};table.launch(AVBD_BOUNDED_BATCH,&args);
                if(!args.executed){status=NumericalFailure;return false;}}advance();}
        return status==Converged;
    }
};
} }
#endif
