// Compact persistent rows: lifecycle, copied NP seeds, unique dual and parallel access.
#include "avbd/DyAvbdRows.h"
#include "avbd_probe_foundation.h"
#include "foundation/PxMemory.h"
#include <vector>
#include <thread>
#include <type_traits>
#include <cstdio>
using namespace physx;using namespace physx::Dy;
static AvbdRowBuilder makeRow(PxU32 id,PxReal* output)
{
    AvbdRowBuilder row;PxMemZero(&row,sizeof(row));
    row.header().kernelId=PxU8(id%7);row.header().bodyA=id;row.header().bodyB=id+1;
    row.header().supportStart=4*id;row.header().supportCount=row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON?4:0;
    row.header().geometryIndex=id+1;row.header().flags=PxU8(id%256);
    auto& p=row.scalar();p.rho=float(id+1);p.lambda=-float(id+2);
    p.n=PxVec3(float(id),2,3);p.rA=PxVec3(1,float(id),3);p.rB=PxVec3(1,2,float(id));
    p.jLinA=PxVec3(1,2,3);p.jAngB=PxVec3(4,5,6);p.minLambda=-10;p.maxLambda=20;
    p.t0=PxVec3(1,0,0);p.t1=PxVec3(0,0,1);p.mu=.5f;p.rhoT0=5;p.rhoT1=6;
    p.lambdaT0=7;p.lambdaT1=8;p.articAxis=2;p.articFlip=1;p.integratedAngular=1;
    p.accelerationDrive=1;p.driveStiffness=9;p.driveDampingOverDt=10;p.forceWriteback=output;
    return row;
}
static bool same(const AvbdRowConstRef& row,const AvbdRowBuilder& expected)
{
    if(row.header().kernelId!=expected.kernelId || row.header().bodyA!=expected.bodyA || row.header().bodyB!=expected.bodyB ||
       row.header().supportStart!=expected.supportStart || row.header().supportCount!=expected.supportCount ||
       row.header().geometryIndex!=expected.geometryIndex || row.header().flags!=expected.flags)return false;
    const bool scalar=row.header().kernelId!=eAVBD_ROW_KERNEL_ELASTIC_FRAME && row.header().kernelId!=eAVBD_ROW_KERNEL_TETRAHEDRON;
    if(row.hasScalar()!=scalar)return false;if(!scalar)return row.header().scalarIndex==0;
    const auto& a=row.scalar();const auto& b=expected.scalar();
    return a.rho==b.rho && a.lambda==b.lambda && a.n==b.n && a.rA==b.rA && a.rB==b.rB &&
        a.jLinA==b.jLinA && a.jAngB==b.jAngB && a.minLambda==b.minLambda && a.maxLambda==b.maxLambda &&
        a.t0==b.t0 && a.t1==b.t1 && a.mu==b.mu && a.rhoT0==b.rhoT0 && a.rhoT1==b.rhoT1 &&
        a.lambdaT0==b.lambdaT0 && a.lambdaT1==b.lambdaT1 && a.articAxis==b.articAxis && a.articFlip==b.articFlip &&
        a.integratedAngular==b.integratedAngular && a.accelerationDrive==b.accelerationDrive &&
        a.driveStiffness==b.driveStiffness && a.driveDampingOverDt==b.driveDampingOverDt && a.forceWriteback==b.forceWriteback;
}
int main()
{
    AvbdProbeFoundation foundation;AvbdRows rows;unsigned failures=0;
    static_assert(std::is_const<typename std::remove_reference<decltype((std::declval<AvbdRowConstRef>().header().bodyA))>::type>::value,"const header");
    static_assert(std::is_const<typename std::remove_reference<decltype(std::declval<AvbdRowConstRef>().scalar())>::type>::value,"const payload");
    static_assert(sizeof(AvbdRow)==24,"common header ABI");
    static_assert(sizeof(AvbdRowRef)==2*sizeof(void*),"borrowed view must stay thin");
    std::vector<PxReal> outputs(8192);std::vector<AvbdRowBuilder> expected;
    for(PxU32 i=0;i<outputs.size();++i){expected.push_back(makeRow(i,&outputs[i]));rows.pushBack(expected.back());}
    PxU32 scalarCount=0;
    for(PxU32 i=0;i<rows.size();++i){failures+=!same(rows[i],expected[i]);scalarCount+=rows[i].hasScalar();}
    failures+=rows.scalarCount()!=scalarCount;
    // Disjoint row and dual writers, with all indices and pools frozen.
    std::vector<std::thread> workers;
    for(PxU32 t=0;t<4;++t)workers.emplace_back([&,t]()
    {
        for(PxU32 i=t;i<rows.size();i+=4)
        {auto row=rows[i];row.header().flags^=eAVBD_ROW_FRESH;if(row.hasScalar()){row.scalar().lambda-=3;*row.scalar().forceWriteback=row.scalar().lambda;}}
    });
    for(auto& worker:workers)worker.join();
    for(PxU32 i=0;i<rows.size();++i)
    {expected[i].flags^=eAVBD_ROW_FRESH;expected[i].scalar().lambda-=3;failures+=!same(rows[i],expected[i]);}
    // Copy a reduced contact manifold before truncating/replacing the suffix.
    std::vector<AvbdRowBuilder> seeds;
    for(PxU32 i=100;i<150;++i)seeds.emplace_back(rows[i]);
    rows.truncate(100);expected.resize(100);
    for(PxU32 i=0;i<seeds.size();++i){rows.pushBack(seeds[seeds.size()-1-i]);expected.push_back(seeds[seeds.size()-1-i]);}
    for(PxU32 i=0;i<rows.size();++i)failures+=!same(rows[i],expected[i]);
    // Repeated relinearization truncates at both scalar and material boundaries.
    for(PxU32 count=rows.size();count>0;--count)
    {rows.truncate(count);scalarCount=0;for(const auto row:rows)scalarCount+=row.hasScalar();failures+=rows.scalarCount()!=scalarCount;}
    rows.truncate(0);failures+=!rows.empty() || rows.scalarCount()!=0;
    for(PxU32 i=0;i<4096;++i)
    {AvbdRow row={};row.header().bodyA=i;row.header().bodyB=i+1;row.header().kernelId=i%2?eAVBD_ROW_KERNEL_ELASTIC_FRAME:eAVBD_ROW_KERNEL_TETRAHEDRON;rows.pushBack(row);}
    failures+=rows.scalarCount()!=0 || rows.bytes()!=PxU64(4096)*sizeof(AvbdRow);
    const AvbdRows& immutable=rows;PxU32 visited=0;for(const auto row:immutable){failures+=row.hasScalar();++visited;}
    failures+=visited!=rows.size();
    std::printf("[ROW_STORAGE] header=%llu scalar=%llu materialRows=%u bytes=%llu workers=4 failures=%u\n",
        static_cast<unsigned long long>(sizeof(AvbdRow)),static_cast<unsigned long long>(sizeof(AvbdScalarRowPayload)),
        rows.size(),static_cast<unsigned long long>(rows.bytes()),failures);
    return failures?1:0;
}
