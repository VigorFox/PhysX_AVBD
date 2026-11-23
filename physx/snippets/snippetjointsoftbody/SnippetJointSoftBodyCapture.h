// Read-only mechanism fixtures. Captured data is never consumed by the solver.
#ifndef SNIPPET_JOINT_SOFT_BODY_CAPTURE_H
#define SNIPPET_JOINT_SOFT_BODY_CAPTURE_H
#include <cstdio>
#include <cstdlib>
#include <memory>
inline void captureVec(FILE* f,const physx::PxVec3& v)
{ std::fprintf(f,"[%.9g,%.9g,%.9g]",v.x,v.y,v.z); }
inline void capturePose(FILE* f,const physx::PxTransform& p)
{ std::fprintf(f,"{\"p\":");captureVec(f,p.p);std::fprintf(f,",\"q\":[%.9g,%.9g,%.9g,%.9g]}",p.q.x,p.q.y,p.q.z,p.q.w); }
inline void captureShape(FILE* f,const physx::PxShape& shape)
{
 using namespace physx;const PxGeometry& g=shape.getGeometry();
 if(g.getType()==PxGeometryType::ePLANE) { std::fprintf(f,"{\"kind\":\"plane\"}");return; }
 if(g.getType()!=PxGeometryType::eBOX && g.getType()!=PxGeometryType::eCONVEXMESH)
 { std::fprintf(f,"{\"kind\":\"unsupported\",\"type\":%u}",unsigned(g.getType()));return; }
 std::fprintf(f,"{\"kind\":\"polytope\",\"vertices\":[");
 if(g.getType()==PxGeometryType::eBOX)
 {
  const PxVec3 e=static_cast<const PxBoxGeometry&>(g).halfExtents;
  for(PxU32 i=0;i<8;++i) { if(i) std::fprintf(f,",");captureVec(f,PxVec3(i&1?e.x:-e.x,i&2?e.y:-e.y,i&4?e.z:-e.z)); }
  std::fprintf(f,"],\"faces\":[[0,2,6,4],[1,5,7,3],[0,4,5,1],[2,3,7,6],[0,1,3,2],[4,6,7,5]]}");
 }
 else
 {
  const PxConvexMeshGeometry& c=static_cast<const PxConvexMeshGeometry&>(g);const PxConvexMesh& mesh=*c.convexMesh;
  for(PxU32 i=0;i<mesh.getNbVertices();++i) { if(i) std::fprintf(f,",");captureVec(f,c.scale.toMat33()*mesh.getVertices()[i]); }
  std::fprintf(f,"],\"faces\":[");
  for(PxU32 i=0;i<mesh.getNbPolygons();++i)
  {
   PxHullPolygon polygon;mesh.getPolygonData(i,polygon);if(i) std::fprintf(f,",");std::fprintf(f,"[");
   for(PxU32 j=0;j<polygon.mNbVerts;++j) std::fprintf(f,"%s%u",j?",":"",unsigned(mesh.getIndexBuffer()[polygon.mIndexBase+j]));
   std::fprintf(f,"]");
  }
  std::fprintf(f,"]}");
 }
}
template<class Witnesses>
void capturePair(FILE* f,physx::PxU32 frame,const physx::PxRigidActor* a,const physx::PxRigidActor* b,
 physx::PxReal observedDepth,const Witnesses& witnesses)
{
 using namespace physx;if(!f || !a || !b) return;const PxRigidActor* actors[2]={a,b};
 std::fprintf(f,"{\"schema\":1,\"frame\":%u,\"observedDepth\":%.9g,\"actors\":[",frame,observedDepth);
 for(PxU32 k=0;k<2;++k)
 {
  if(k) std::fprintf(f,",");PxShape* shape=NULL;actors[k]->getShapes(&shape,1);
  const auto it=witnesses.before.find(actors[k]);const PxTransform before=it==witnesses.before.end()?actors[k]->getGlobalPose():it->second;
  std::fprintf(f,"{\"shape\":");captureShape(f,*shape);
  std::fprintf(f,",\"before\":");capturePose(f,before*shape->getLocalPose());
  std::fprintf(f,",\"after\":");capturePose(f,PxShapeExt::getGlobalPose(*shape,*actors[k]));
  const PxRigidDynamic* dynamic=actors[k]->is<PxRigidDynamic>();
  std::fprintf(f,",\"mass\":%.9g}",dynamic?dynamic->getMass():0.f);
 }
 std::fprintf(f,"],\"witnesses\":[");bool first=true;
 for(size_t i=0;i<witnesses.samples.size();++i)
 {
  const auto& s=witnesses.samples[i];if(!((s.a==a && s.b==b)||(s.a==b && s.b==a))) continue;
  if(!first) std::fprintf(f,",");first=false;
  const PxVec3 delta=s.a->getGlobalPose().transform(s.pa)-s.b->getGlobalPose().transform(s.pb);
  std::fprintf(f,"{\"gap0\":%.9g,\"gap1\":%.9g}",s.separation,s.n.dot(delta)+s.separation);
 }
 std::fprintf(f,"]}\n");std::fflush(f);
}
#endif
