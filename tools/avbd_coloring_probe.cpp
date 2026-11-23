// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "avbd/DyAvbdColoring.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
using namespace physx;
using namespace physx::Dy;

int main()
{
 AvbdProbeFoundation foundation;
 bool pass=true;
 for(PxU32 trial=0;trial<100;++trial)
 {
  const PxU32 count=67;
  AvbdBlockColoring graph;
  PxArray<PxU8> writable(count,1);writable[0]=0;
  PxArray<PxU32> order,offsets,blocks,rank(count,0),colors(count,0);
  PxArray<PxArray<PxU32>> scopes;
  for(PxU32 i=1;i<count;++i) order.pushBack(1+(i-1+trial)%(count-1));
  for(PxU32 i=0;i<order.size();++i) rank[order[i]]=i;
  PxU32 random=trial+1;
  for(PxU32 s=0;s<40;++s)
  {
   PxArray<PxU32> scope;
   for(PxU32 j=0;j<2+s%4;++j)
   { random=random*1664525u+1013904223u;scope.pushBack(random%count); }
   graph.addScope(scope.begin(),scope.size(),writable);scopes.pushBack(scope);
  }
  graph.build(order,count,offsets,blocks);
  pass &= blocks.size()==order.size();
  PxArray<PxU8> seen(count,0);
  for(PxU32 c=0;c+1<offsets.size();++c)
   for(PxU32 i=offsets[c];i<offsets[c+1];++i)
   { pass &= !seen[blocks[i]];seen[blocks[i]]=1;colors[blocks[i]]=c; }
  pass &= !seen[0];for(PxU32 b:order)pass &= seen[b]!=0;
  for(PxU32 s=0;s<scopes.size();++s)
   for(PxU32 i=0;i<scopes[s].size();++i) for(PxU32 j=0;j<scopes[s].size();++j)
   {
    PxU32 a=scopes[s][i],b=scopes[s][j];
    if(writable[a] && writable[b] && rank[a]<rank[b]) pass &= colors[a]<colors[b];
   }
  // An independent clique representation is the oracle for the borrowed
  // rest-material CSR. Include duplicate supports and changing writable sets.
  AvbdBlockColoring baked;
  PxU8 edges[67][67]={};
  for(PxU32 s=0;s<scopes.size();++s)
  {
   if(s&1) baked.addScope(scopes[s].begin(),scopes[s].size(),writable);
   else for(PxU32 i=0;i<scopes[s].size();++i) for(PxU32 j=0;j<scopes[s].size();++j)
    if(scopes[s][i]!=scopes[s][j]) edges[scopes[s][i]][scopes[s][j]]=1;
  }
  PxArray<PxU32> restOffsets,restNeighbors;
  for(PxU32 i=0;i<count;++i)
  {
   restOffsets.pushBack(restNeighbors.size());
   for(PxU32 j=0;j<count;++j) if(edges[i][j]) restNeighbors.pushBack(j);
  }
  restOffsets.pushBack(restNeighbors.size());
  baked.addAdjacency(restOffsets.begin(),restNeighbors.begin(),0,count);
  const PxU32 prefixes[3]={0,trial%order.size(),order.size()};
  for(PxU32 mode=0;mode<3;++mode)
  {
   const PxU32 prefix=prefixes[mode];
   graph.build(order,count,offsets,blocks,prefix);
   PxArray<PxU32> bakedOffsets,bakedBlocks;
   baked.build(order,count,bakedOffsets,bakedBlocks,prefix);
   // Both representations must independently cover the same dependencies;
   // color IDs, ordering and repeatability are not acceptance requirements.
   for(unsigned representation=0;representation<2;++representation)
   {
    if(representation){offsets=bakedOffsets;blocks=bakedBlocks;}
    pass &= blocks.size()==order.size();
    for(PxU32 i=0;i<count;++i) seen[i]=0;
    for(PxU32 c=0;c+1<offsets.size();++c)
     for(PxU32 i=offsets[c];i<offsets[c+1];++i)
     { pass &= !seen[blocks[i]];seen[blocks[i]]=1;colors[blocks[i]]=c; }
    pass &= !seen[0];for(PxU32 b:order)pass &= seen[b]!=0;
    for(PxU32 s=0;s<scopes.size();++s)
     for(PxU32 i=0;i<scopes[s].size();++i) for(PxU32 j=0;j<scopes[s].size();++j)
     {
      const PxU32 a=scopes[s][i],b=scopes[s][j];
      if(a==b || !writable[a] || !writable[b]) continue;
      pass &= colors[a]!=colors[b];
      if(rank[a]<rank[b] && rank[b]>=prefix) pass &= colors[a]<colors[b];
     }
   }
  }
  // Group projection retains the complete row conflict clique, including
  // repeated mapped endpoints. Validate coverage against original supports.
  AvbdRowProgram projected;PxArray<PxU32> groupOrder;
  auto root=[](PxU32 b){return ((b-1)/4)*4+1;};
  for(PxU32 b=1;b<count;b+=4)groupOrder.pushBack(b);
  pass &= projected.compile(scopes.size(),count,[&](PxU32 r,const auto& emit)
  {for(PxU32 b:scopes[r])if(writable[b])emit(root(b));return true;});
  AvbdBlockColoring grouped;grouped.addProgram(projected);
  grouped.build(groupOrder,count,offsets,blocks,groupOrder.size());
  for(PxU32 b=0;b<count;++b)seen[b]=0;
  for(PxU32 c=0;c+1<offsets.size();++c)for(PxU32 i=offsets[c];i<offsets[c+1];++i)
  {pass &= !seen[blocks[i]];seen[blocks[i]]=1;colors[blocks[i]]=c;}
  for(PxU32 b:groupOrder)pass &= seen[b]!=0;
  for(const auto& scope:scopes)for(PxU32 a:scope)for(PxU32 b:scope)
   if(writable[a] && writable[b] && root(a)!=root(b))pass &= colors[root(a)]!=colors[root(b)];
 }
 AvbdBlockColoring staticSupport;
 PxArray<PxU8> writable(3,1);writable[0]=0;
 const PxU32 a[3]={1,0,1},b[2]={2,0};
 staticSupport.addScope(a,3,writable);staticSupport.addScope(b,2,writable);
 PxArray<PxU32> order,offsets,blocks;order.pushBack(1);order.pushBack(2);
 staticSupport.build(order,3,offsets,blocks);
 pass &= offsets.size()==2 && blocks.size()==2;
 order.clear();staticSupport.build(order,3,offsets,blocks);
 pass &= offsets.size()==1 && blocks.empty();
 AvbdBlockColoring chain,clique;
 writable.resize(128,1);writable[0]=1;
 for(PxU32 i=0;i<128;++i)
 {
  order.pushBack(i);
  if(i) { const PxU32 edge[2]={i-1,i};chain.addScope(edge,2,writable); }
 }
 chain.build(order,128,offsets,blocks,128);
 pass &= offsets.size()==3;
 clique.addScope(order.begin(),order.size(),writable);
 clique.build(order,128,offsets,blocks,128);
 pass &= offsets.size()==129 && blocks.size()==128;
 std::printf("[COLORING_TEST] coverage_ordered_api_compact_static_duplicates_empty_chain_uncapped_clique_result_equality_not_required=%s\n",pass?"PASS":"FAIL");
 return pass?0:1;
}
