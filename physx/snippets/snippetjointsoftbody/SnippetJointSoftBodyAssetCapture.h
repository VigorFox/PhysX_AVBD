// Read-only full input capture for the independent migration laboratory.
#ifndef SNIPPET_JOINT_SOFT_BODY_ASSET_CAPTURE_H
#define SNIPPET_JOINT_SOFT_BODY_ASSET_CAPTURE_H
#include "SnippetJointSoftBodyCapture.h"
#include <map>
#include <vector>

namespace JointSoftBodySnippet
{
inline void captureArray(FILE* f,const PxReal* x,PxU32 n)
{ std::fprintf(f,"[");for(PxU32 i=0;i<n;++i) std::fprintf(f,"%s%.9g",i?",":"",x[i]);std::fprintf(f,"]"); }
inline void captureAssetShape(FILE* f,const PxShape& shape)
{
 std::fprintf(f,"{\"geometry\":");captureShape(f,shape);std::fprintf(f,",\"localPose\":");capturePose(f,shape.getLocalPose());
 const PxFilterData d=shape.getSimulationFilterData();
 std::fprintf(f,",\"geometryType\":%u,\"filter\":[%u,%u,%u,%u],\"flags\":%u,\"contactOffset\":%.9g,\"restOffset\":%.9g,\"materials\":[",
  unsigned(shape.getGeometry().getType()),d.word0,d.word1,d.word2,d.word3,unsigned(shape.getFlags()),shape.getContactOffset(),shape.getRestOffset());
 std::vector<PxMaterial*> materials(shape.getNbMaterials());shape.getMaterials(materials.data(),PxU32(materials.size()));
 for(size_t i=0;i<materials.size();++i)
 {
  const PxMaterial& m=*materials[i];if(i) std::fprintf(f,",");
  std::fprintf(f,"{\"staticFriction\":%.9g,\"dynamicFriction\":%.9g,\"restitution\":%.9g,\"frictionCombine\":%u,\"restitutionCombine\":%u}",
   m.getStaticFriction(),m.getDynamicFriction(),m.getRestitution(),unsigned(m.getFrictionCombineMode()),unsigned(m.getRestitutionCombineMode()));
 }
 std::fprintf(f,"]}");
}
inline void captureAssetActor(FILE* f,const PxRigidActor& actor)
{
 std::fprintf(f,"{\"actorFlags\":%u,\"pose\":",unsigned(actor.getActorFlags()));capturePose(f,actor.getGlobalPose());
 const PxRigidDynamic* b=actor.is<PxRigidDynamic>();
 if(b)
 {
  std::fprintf(f,",\"comLocal\":");capturePose(f,b->getCMassLocalPose());
  std::fprintf(f,",\"comWorld\":");capturePose(f,b->getGlobalPose()*b->getCMassLocalPose());
  std::fprintf(f,",\"mass\":%.9g,\"inertia\":",b->getMass());captureVec(f,b->getMassSpaceInertiaTensor());
  std::fprintf(f,",\"v\":");captureVec(f,b->getLinearVelocity());std::fprintf(f,",\"w\":");captureVec(f,b->getAngularVelocity());
  std::fprintf(f,",\"sleeping\":%s,\"stabilizationThreshold\":%.9g,\"maxDepenetrationVelocity\":%.9g,\"maxContactImpulse\":%.9g",
   b->isSleeping()?"true":"false",b->getStabilizationThreshold(),b->getMaxDepenetrationVelocity(),b->getMaxContactImpulse());
  PxU32 positionIterations,velocityIterations;b->getSolverIterationCounts(positionIterations,velocityIterations);
  std::fprintf(f,",\"linearDamping\":%.9g,\"angularDamping\":%.9g,\"sleepThreshold\":%.9g,\"wakeCounter\":%.9g,\"bodyFlags\":%u,\"lockFlags\":%u,\"maxLinearVelocity\":%.9g,\"maxAngularVelocity\":%.9g,\"iterations\":[%u,%u]",
   b->getLinearDamping(),b->getAngularDamping(),b->getSleepThreshold(),b->getWakeCounter(),unsigned(b->getRigidBodyFlags()),unsigned(b->getRigidDynamicLockFlags()),
   b->getMaxLinearVelocity(),b->getMaxAngularVelocity(),positionIterations,velocityIterations);
 }
 std::fprintf(f,",\"shapes\":[");std::vector<PxShape*> shapes(actor.getNbShapes());actor.getShapes(shapes.data(),PxU32(shapes.size()));
 for(size_t i=0;i<shapes.size();++i) { if(i) std::fprintf(f,",");captureAssetShape(f,*shapes[i]); }
 std::fprintf(f,"]}");
}
inline bool captureShowcaseAsset(const ShowcaseWorld& world,const Settings& settings,const char* sceneName,double initialEnergy)
{
 const char* path=std::getenv("PHYSX_AVBD_ASSET_CAPTURE");if(!path) return true;
 std::unique_ptr<FILE,int(*)(FILE*)> file(std::fopen(path,"wb"),std::fclose);if(!file) return false;FILE* f=file.get();
 std::vector<const PxRigidActor*> actors;std::map<const PxRigidActor*,PxU32> ids;
 for(PxU32 i=0;i<world.bodies.size();++i) for(PxU32 j=0;j<world.bodies[i]->lattice.nodes.size();++j) actors.push_back(world.bodies[i]->lattice.nodes[j]);
 if(world.host.impactor) actors.push_back(world.host.impactor);
 for(PxU32 i=0;i<actors.size();++i) ids[actors[i]]=i;
 std::map<const void*,PxConstraint*> constraints;
 for(size_t i=0;i<actors.size();++i)
 {
  std::vector<PxConstraint*> list(actors[i]->getNbConstraints());actors[i]->getConstraints(list.data(),PxU32(list.size()));
  for(size_t j=0;j<list.size();++j) { PxU32 type=0;constraints[list[j]->getExternalReference(type)]=list[j]; }
 }
 std::fprintf(f,"{\"schema\":1,\"scene\":\"%s\",\"dt\":%.9g,\"dampingRatio\":%.9g,\"volumeMaterial\":%s,\"initialMechanicalEnergy\":%.17g,\"gravity\":",
  sceneName,1.f/60.f,settings.dampingRatio,settings.volumeMaterial?"true":"false",initialEnergy);captureVec(f,world.host.scene->getGravity());
 const PxScene& scene=*world.host.scene;
 std::fprintf(f,",\"sceneParameters\":{\"solverType\":%u,\"flags\":%u,\"bounceThresholdVelocity\":%.9g,\"frictionOffsetThreshold\":%.9g,\"frictionCorrelationDistance\":%.9g,\"ccdMaxPasses\":%u}",
  unsigned(scene.getSolverType()),unsigned(scene.getFlags()),scene.getBounceThresholdVelocity(),scene.getFrictionOffsetThreshold(),scene.getFrictionCorrelationDistance(),scene.getCCDMaxPasses());
 std::fprintf(f,",\"actors\":[");for(size_t i=0;i<actors.size();++i) { if(i) std::fprintf(f,",");captureAssetActor(f,*actors[i]); }
 std::fprintf(f,"],\"staticActors\":[");bool first=true;
 const PxRigidActor* fixed[2]={world.host.ground,world.pedestal};
 for(PxU32 i=0;i<2;++i) if(fixed[i]) { if(!first) std::fprintf(f,",");first=false;captureAssetActor(f,*fixed[i]); }
 std::fprintf(f,"],\"assets\":[");PxU32 start=0;
 for(PxU32 i=0;i<world.bodies.size();++i)
 {
  const JointAsset& a=*world.bodies[i];const Lattice& l=a.lattice;if(i) std::fprintf(f,",");
  std::fprintf(f,"{\"firstActor\":%u,\"actorCount\":%u,\"grid\":%u,\"cell\":%.9g,\"mass\":%.9g,\"proxyVolume\":%.9g,\"beamCount\":%u,\"tetCount\":%u}",
   start,l.nodes.size(),l.grid,l.cell,l.totalMass,a.proxyVolume,l.joints.size(),l.materials.size());start+=l.nodes.size();
 }
 std::fprintf(f,"],\"beams\":[");first=true;
 for(PxU32 i=0;i<world.bodies.size();++i) for(PxU32 j=0;j<world.bodies[i]->lattice.joints.size();++j)
 {
  const FiniteJoint& joint=*world.bodies[i]->lattice.joints[j];PxRigidActor* a;PxRigidActor* b;joint.getActors(a,b);
  const FiniteJointData& data=*static_cast<const FiniteJointData*>(joint.getConstantBlock());
  if(!first) std::fprintf(f,",");first=false;
  std::fprintf(f,"{\"ids\":[%u,%u],\"frames\":[",ids.at(a),ids.at(b));capturePose(f,data.c2b[0]);std::fprintf(f,",");capturePose(f,data.c2b[1]);
  std::fprintf(f,"],\"actorFrames\":[");capturePose(f,joint.getLocalPose(PxJointActorIndex::eACTOR0));std::fprintf(f,",");capturePose(f,joint.getLocalPose(PxJointActorIndex::eACTOR1));
  const char* labels[3]={"stiffness","damping","forceLimit"};
  for(PxU32 k=0;k<3;++k) { std::fprintf(f,"%s\"%s\":[",k?",":"],",labels[k]);for(PxU32 r=0;r<6;++r) { const PxD6JointDrive& d=data.drives[r];std::fprintf(f,"%s%.9g",r?",":"",k==0?d.stiffness:k==1?d.damping:d.forceLimit); }std::fprintf(f,"]"); }
  std::fprintf(f,",\"driveFlags\":[");for(PxU32 r=0;r<6;++r) std::fprintf(f,"%s%u",r?",":"",unsigned(data.drives[r].flags));std::fprintf(f,"]");
  Px1DConstraint rows[FINITE_JOINT_ROWS];PxVec3p offset,anchorA,anchorB;PxConstraintInvMassScale scale;
  const PxTransform pa=a->getGlobalPose()*a->is<PxRigidBody>()->getCMassLocalPose(),pb=b->getGlobalPose()*b->is<PxRigidBody>()->getCMassLocalPose();
  if(finiteJointPrep(rows,offset,FINITE_JOINT_ROWS,scale,&data,pa,pb,false,anchorA,anchorB)!=FINITE_JOINT_ROWS) return false;
  std::fprintf(f,",\"coordinates\":[");for(PxU32 r=0;r<FINITE_JOINT_ROWS;++r) std::fprintf(f,"%s%.9g",r?",":"",rows[r].geometricError);
  std::fprintf(f,"],\"jacobian\":[");
  for(PxU32 r=0;r<FINITE_JOINT_ROWS;++r) { if(r) std::fprintf(f,",");PxReal v[12];for(PxU32 k=0;k<3;++k) { v[k]=rows[r].linear0[k];v[k+3]=rows[r].angular0[k];v[k+6]=-rows[r].linear1[k];v[k+9]=-rows[r].angular1[k]; }captureArray(f,v,12); }
  std::fprintf(f,"],\"elasticEnergy\":%.17g}",joint.elasticEnergy());
 }
 std::fprintf(f,"],\"tetrahedra\":[");first=true;
 for(PxU32 i=0;i<world.bodies.size();++i) for(PxU32 j=0;j<world.bodies[i]->lattice.materials.size();++j)
 {
  const VolumeMaterial& m=*world.bodies[i]->lattice.materials[j];const PxConstraintTetrahedronData& d=*static_cast<const PxConstraintTetrahedronData*>(m.getConstantBlock());
  const auto found=constraints.find(&m);if(found==constraints.end()) return false;PxRigidActor* support[4];
  if(found->second->getNbActors()!=4 || found->second->getActors(support,4)!=4) return false;
  if(!first) std::fprintf(f,",");first=false;std::fprintf(f,"{\"ids\":[");for(PxU32 k=0;k<4;++k) std::fprintf(f,"%s%u",k?",":"",ids.at(support[k]));
  std::fprintf(f,"],\"localPoints\":[");for(PxU32 k=0;k<4;++k) { if(k) std::fprintf(f,",");captureVec(f,d.localPoints[k]); }
  std::fprintf(f,"],\"restDeterminant\":%.9g,\"energyScale\":%.9g,\"elasticEnergy\":%.17g}",d.restDeterminant,d.energyScale,m.elasticEnergy());
 }
 std::fprintf(f,"],\"suppressedActorPairs\":[");first=true;
 // Current showcase uses exactly one simulation shape per actor. Refuse an
 // ambiguous actor-level filter capture rather than silently dropping shapes.
 for(size_t a=0;a<actors.size();++a)
 {
  if(actors[a]->getNbShapes()!=1) return false;PxShape* sa;actors[a]->getShapes(&sa,1);
  for(size_t b=a+1;b<actors.size();++b)
  {
   if(actors[b]->getNbShapes()!=1) return false;PxShape* sb;actors[b]->getShapes(&sb,1);
   if(excludedCellPair(sa->getSimulationFilterData(),sb->getSimulationFilterData()))
   { if(!first) std::fprintf(f,",");first=false;std::fprintf(f,"[%u,%u]",unsigned(a),unsigned(b)); }
  }
 }
 std::fprintf(f,"]}\n");return std::fflush(f)==0 && !std::ferror(f);
}
}
#endif
