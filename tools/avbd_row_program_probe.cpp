// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "avbd_probe_foundation.h"
#include "avbd/DyAvbdRowProgram.h"
#include "avbd/DyAvbdColoring.h"
#include "avbd/DyAvbdContactMaterial.h"
#include <vector>
#include <algorithm>
#include <random>
using namespace physx;
using namespace physx::Dy;

static bool materialPatchSchedule()
{
    // A dynamic star above the small-island bound forces the color spill.
    // Independent bodies touching the same fixed world index share no writes.
    AvbdContactMaterial program;
    const PxU32 spokes=211, isolated=65, bodies=1+spokes+isolated;
    program.begin(bodies);
    PxArray<PxU32> components(bodies,0);
    std::vector<unsigned> visits(spokes+isolated);
    for(PxU32 i=1;i<bodies;++i)
    {
        const bool large=i<=spokes;
        if(!large) components[i]=i;
        program.patches.pushBack({0,1,{i,large ? 0 : PX_MAX_U32},{1,large ? 1.f : 0.f}});
    }
    program.schedule(components,8);
    if(program.offsets.size()!=spokes+1 || program.order.size()!=spokes ||
        program.islandPatches.size()!=isolated) return false;
    for(PxU32 c=1;c<program.offsets.size();++c)
    {
        std::vector<bool> written(bodies);
        for(PxU32 j=program.offsets[c-1];j<program.offsets[c];++j)
        {
            const PxU32 index=program.order[j];
            ++visits[index];
            for(PxU32 b:program.patches[index].body)
            {
                if(written[b]) return false;
                written[b]=true;
            }
        }
    }
    for(PxU32 i=1;i<program.islandOffsets.size();++i)
    {
        if(program.islandOffsets[i]-program.islandOffsets[i-1]>AvbdContactMaterial::IslandPatchLimit) return false;
        for(PxU32 j=program.islandOffsets[i-1];j<program.islandOffsets[i];++j)
            ++visits[program.islandPatches[j]];
    }
    return std::all_of(visits.begin(),visits.end(),[](unsigned n){return n==1;});
}

static bool materialContactLaw()
{
    AvbdContactMaterial p;
    p.begin(2);
    p.patches.pushBack({0,1,{0,1},{1,1}});
    p.order.pushBack(0); p.residual.resize(1); p.friction=true;
    AvbdMaterialPoint point={};
    point.friction=.5f;
    const PxVec3 axes[3]={PxVec3(1,0,0),PxVec3(0,1,0),PxVec3(0,0,1)};
    for(PxU32 k=0;k<3;++k)
    {
        auto& axis=point.axes[k];
        axis.direction=axes[k];
        for(PxU32 s=0;s<2;++s) axis.angular[s]=axis.angularResponse[s]=PxVec3(0.f);
    }
    point.axes[0].baseSpeed=-3; point.axes[1].baseSpeed=2;
    p.points.pushBack(point);
    AvbdMaterialRow state={};
    state.solveNormal=state.solveTangent=1;
    state.normalResponse=2; state.maxNormalImpulse=1; state.tangentStep=.5f;
    state.tangentInverse=PxVec3(.5f,0,.5f);
    AvbdMaterialBatch args={p,&state,0,1,AvbdMaterialBatch::Update};
    avbdContactMaterialBatch(&args);
    if(state.normalImpulse!=1 || state.tangentImpulse0!=-.5f || state.tangentImpulse1!=0 ||
        p.delta[0].linearPx()!=PxVec3(1,-.5f,0) || p.delta[1].linearPx()!=-p.delta[0].linearPx()) return false;
    args.operation=AvbdMaterialBatch::Residual;
    avbdContactMaterialBatch(&args);
    if(p.residual[0].error!=0) return false;
    // Releasing the normal also releases the stored friction impulse.
    args.operation=AvbdMaterialBatch::Update;
    p.points[0].axes[0].baseSpeed=1;
    avbdContactMaterialBatch(&args);
    return state.normalImpulse==0 && state.tangentImpulse0==0 &&
        p.delta[0].linearPx().isZero() && p.delta[1].linearPx().isZero();
}

int main()
{
    AvbdProbeFoundation foundation;
    AvbdRowProgram program;
    std::mt19937 random(20261016);
    bool pass = materialPatchSchedule() && materialContactLaw();
    std::printf("[MATERIAL_PATCH] spill_islands_cone_release=%s\n",pass ? "PASS" : "FAIL");
    for(PxU32 trial = 0; trial < 600; ++trial)
    {
        const PxU32 n = 1 + random() % 40, nr = random() % 60;
        std::vector<std::vector<PxU32>> scopes(nr);
        std::vector<bool> active(nr), responds(n);
        std::vector<std::vector<bool>> incident(nr, std::vector<bool>(n));
        std::vector<std::vector<bool>> reach(n, std::vector<bool>(n));
        for(PxU32 b = 0; b < n; ++b) { responds[b] = random() % 3 != 0; reach[b][b] = true; }
        for(PxU32 r = 0; r < nr; ++r)
        {
            active[r] = random() % 4 != 0;
            const PxU32 count = random() % 18;
            for(PxU32 k = 0; k < count; ++k)
            {
                const PxU32 b = random() % (n + 1);
                scopes[r].push_back(b == n ? PX_MAX_U32 : b);
                if(b < n) incident[r][b] = true;
            }
            if(active[r]) for(PxU32 a = 0; a < n; ++a) for(PxU32 b = 0; b < n; ++b)
                if(incident[r][a] && incident[r][b] && responds[a] && responds[b]) reach[a][b] = true;
        }
        pass &= program.compile(nr, n, [&](PxU32 r, const auto& emit)
        { for(PxU32 b : scopes[r]) emit(b); return true; });
        PxU32 total = 0;
        for(PxU32 r = 0; r < nr; ++r)
        {
            std::vector<PxU32> visits(n);
            program.forEachBlock(r, [&](PxU32 b) { ++visits[b]; });
            for(PxU32 b = 0; b < n; ++b) { pass &= visits[b] == PxU32(incident[r][b]); total += visits[b]; }
        }
        pass &= program.rowBlocks.size() == total && program.blockRows.size() == total;
        for(PxU32 b = 0; b < n; ++b)
        {
            std::vector<PxU32> actual;
            for(PxU32 k = program.blockOffsets[b]; k < program.blockOffsets[b + 1]; ++k) actual.push_back(program.blockRows[k]);
            std::vector<PxU32> expected;
            for(PxU32 r = 0; r < nr; ++r) if(incident[r][b]) expected.push_back(r);
            pass &= actual == expected;
        }
        // Independent transitive-closure oracle, including static hubs and inactive rows.
        for(PxU32 k = 0; k < n; ++k) for(PxU32 a = 0; a < n; ++a) for(PxU32 b = 0; b < n; ++b)
            reach[a][b] = reach[a][b] || (reach[a][k] && reach[k][b]);
        PxArray<PxU32> parents;
        program.components(parents, [&](PxU32 b) { return responds[b]; }, [&](PxU32 r) { return active[r]; });
        for(PxU32 a = 0; a < n; ++a) for(PxU32 b = 0; b < n; ++b)
            pass &= (parents[a] == parents[b]) == reach[a][b];
        PxArray<PxU8> writable(n, 0); PxArray<PxU32> order;
        for(PxU32 b = 0; b < n; ++b) if(responds[b]) { writable[b] = 1; order.pushBack(b); }
        AvbdBlockColoring borrowed, copied;
        borrowed.addProgram(program);
        for(const auto& scope : scopes) copied.addScope(scope.data(), PxU32(scope.size()), writable);
        // Extra ordinary scopes share the same graph, rather than a second schedule.
        const PxU32 extra[2] = {0, n - 1}; borrowed.addScope(extra, 2, writable); copied.addScope(extra, 2, writable);
        for(PxU32 prefix : {0u, PxU32(order.size() / 2), order.size()})
        {
            PxArray<PxU32> a, b, c, d;
            borrowed.build(order, n, a, b, prefix); copied.build(order, n, c, d, prefix);
            const auto validColoring=[&](const PxArray<PxU32>& offsets,const PxArray<PxU32>& list)
            {
                if(offsets.empty() || offsets[0]!=0 || offsets.back()!=list.size() || list.size()!=order.size())return false;
                std::vector<PxU32> color(n,PX_MAX_U32);
                for(PxU32 label=0;label+1<offsets.size();++label)
                {
                    if(offsets[label]>offsets[label+1])return false;
                    for(PxU32 k=offsets[label];k<offsets[label+1];++k)
                    {
                        const PxU32 v=list[k];
                        if(v>=n || !responds[v] || color[v]!=PX_MAX_U32)return false;
                        color[v]=label;
                    }
                }
                for(PxU32 r=0;r<nr;++r)for(PxU32 u=0;u<n;++u)for(PxU32 v=u+1;v<n;++v)
                    if(incident[r][u] && incident[r][v] && responds[u] && responds[v] && color[u]==color[v])return false;
                return n==1 || !responds[0] || !responds[n-1] || color[0]!=color[n-1];
            };
            // Each legal schedule has complete dependency coverage. Its color
            // labels, packing and order need not match the other representation.
            pass &= validColoring(a,b) && validColoring(c,d);
        }
        if(!pass) { std::printf("[ROW_PROGRAM] failed_trial=%u\n", trial); return 1; }
    }
    // Every endpoint of a four-support material term participates; repeated
    // supports (including canceling coefficients) must never hide a dependency.
    const PxU32 four[] = {0, 1, 2, 3, 2, 3, PX_MAX_U32};
    pass &= program.compile(1, 4, [&](PxU32, const auto& emit)
        { for(PxU32 b : four) emit(b); return true; });
    pass &= program.rowBlocks.size() == 4;
    pass &= !program.compile(1, 4, [&](PxU32, const auto& emit) { emit(0); emit(4); return true; });
    pass &= program.blockOffsets.empty() && program.rowOffsets.empty();
    pass &= !program.compile(1, 4, [&](PxU32, const auto& emit) { emit(0); return false; });
    pass &= program.blockOffsets.empty() && program.rowOffsets.empty();
    pass &= program.compile(0, 0, [&](PxU32, const auto&) { return true; });
    pass &= program.rowOffsets.size() == 1 && program.blockOffsets.size() == 1;
    AvbdRowProgram::Ranges ranges;std::vector<PxU32> selected;
    for(PxU32 r=0;r<1300;++r)if(r<17 || (r>=50 && r<651) || r>=1200)
    {ranges.append(r);selected.push_back(r);}
    pass &= ranges.rows==selected.size() && ranges.spans.size()<=3;
    PxU32 slot=0;
    while(slot<ranges.rows)
    {
        PxU32 physical;
        const PxU32 count=PxMin(256u,ranges.locate(slot,physical));
        pass &= count>0 && count<=256;
        for(PxU32 k=0;k<count;++k)pass &= selected[slot+k]==physical+k;
        slot+=count;
    }
    // Invalid compilation cannot expose an operation prefix for dispatch.
    pass &= !program.compile(1,1,[&](PxU32 r,const auto& emit)
    {program.dualRows.append(r);program.historyRows.append(r);emit(1);return true;});
    pass &= program.dualRows.rows==0 && program.historyRows.rows==0;
    // Only the declared immutable prefix may be skipped. Change the suffix,
    // layout and asset revision independently, including a failed compilation.
    AvbdRowProgram packed,reference;
    PxArray<AvbdRowProgram::PrefixPart> parts;
    parts.pushBack({1,0,2});parts.pushBack({2,3,1});
    for(PxU32 step=0;step<9;++step)
    {
        if(step==3)parts[0].revision++;
        if(step==4)parts[1].firstBlock=4;
        const PxU32 rows=step==2 || step==7 || step==8 ? 3u : 6u;
        const auto source=[&](AvbdRowProgram& target,PxU32 r,const auto& emit)
        {
            if(r<2){emit(PxU32(parts[0].revision));emit(0);emit(0);target.historyRows.append(r);}
            else if(r==2){emit(parts[1].firstBlock);target.dualRows.append(r);}
            else {emit(r);emit((step+r)%7);emit(PX_MAX_U32);target.dualRows.append(r);}
            return !(step==5 && r==4);
        };
        PxU32 visited=0;
        const bool ok=packed.compile(rows,7,[&](PxU32 r,const auto& emit)
        {++visited;return source(packed,r,emit);},&parts);
        pass &= reference.compile(rows,7,[&](PxU32 r,const auto& emit){return source(reference,r,emit);})==ok;
        if(!ok){pass &= step==5 && packed.blockRows.empty();continue;}
        const bool cold=step==0 || step==3 || step==4 || step==6;
        pass &= visited==rows-(cold?0:3);
        const auto same=[](const PxArray<PxU32>& a,const PxArray<PxU32>& b)
        {return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin());};
        pass &= same(packed.rowOffsets,reference.rowOffsets) && same(packed.rowBlocks,reference.rowBlocks) &&
            same(packed.blockOffsets,reference.blockOffsets) && same(packed.blockRows,reference.blockRows);
        for(PxU32 op=0;op<2;++op)
        {
            const auto& a=op?packed.historyRows:packed.dualRows;
            const auto& b=op?reference.historyRows:reference.dualRows;
            pass &= a.rows==b.rows;
            for(PxU32 k=0;k<PxMin(a.rows,b.rows);++k)
            {PxU32 x,y;a.locate(k,x);b.locate(k,y);pass &= x==y;}
        }
    }
    std::printf("[ROW_PROGRAM] random_lifecycle=600 coloring_modes=1800 four_supports_static_inactive_invalid_empty=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
