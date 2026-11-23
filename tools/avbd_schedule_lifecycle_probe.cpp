// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "PxPhysicsAPI.h"
#include "../physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDCommon.h"
#include <atomic>
#include <cstdio>
using namespace physx;
// A dispatcher may execute submitted work immediately. Reporting capacity
// exercises the parallel fan-in path while every child completes inline.
struct InlineDispatcher : PxCpuDispatcher
{
 void submitTask(PxBaseTask& task) PX_OVERRIDE { task.run();task.release(); }
 PxU32 getWorkerCount() const PX_OVERRIDE { return 4; }
};
struct Errors : PxErrorCallback
{
 std::atomic<unsigned> count{0};
 void reportError(PxErrorCode::Enum code,const char* message,const char*,int) PX_OVERRIDE
 {
  if(code&(PxErrorCode::eINVALID_PARAMETER|PxErrorCode::eINVALID_OPERATION|
   PxErrorCode::eOUT_OF_MEMORY|PxErrorCode::eINTERNAL_ERROR))
  { ++count;std::fprintf(stderr,"%s\n",message); }
 }
};
int main()
{
 PxDefaultAllocator allocator;Errors errors;
 PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
 if(!foundation) return 2;
 PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale());
 if(!physics) { foundation->release();return 2; }
 PxMaterial* material=physics->createMaterial(0.5f,0.5f,0.0f);
 bool pass=true;
 const PxU32 counts[5]={0,1,2,4,4};
 for(PxU32 mode=0;mode<5;++mode)
 {
  InlineDispatcher inlineDispatcher;
  PxDefaultCpuDispatcher* dispatcher=mode==4 ? NULL : PxDefaultCpuDispatcherCreate(counts[mode]);
  for(PxU32 repeat=0;repeat<8;++repeat)
  {
   PxScene* scenes[2];PxRigidDynamic* bodies[2][32];PxJoint* joints[2][16];
   PxDeformableVolume* volumes[2][2]={};PxDeformableVolumeMesh* meshes[2][2]={};
   for(PxU32 s=0;s<2;++s)
   {
    PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0,-9.81f,0);
    desc.solverType=PxSolverType::eAVBD;
    desc.cpuDispatcher=dispatcher ? static_cast<PxCpuDispatcher*>(dispatcher) : &inlineDispatcher;
    desc.filterShader=PxDefaultSimulationFilterShader;
    scenes[s]=physics->createScene(desc);
    if(!scenes[s]) return 2;
    for(PxU32 i=0;i<32;++i)
    {
     bodies[s][i]=PxCreateDynamic(*physics,PxTransform(PxVec3(float(i)*3,10,0)),PxBoxGeometry(0.4f,0.4f,0.4f),*material,1);
     bodies[s][i]->setLinearVelocity(PxVec3(float(i%3)*0.2f,0,0));
     scenes[s]->addActor(*bodies[s][i]);
    }
    for(PxU32 i=0;i<16;++i)
     joints[s][i]=PxSphericalJointCreate(*physics,bodies[s][i*2],PxTransform(PxVec3(1.5f,0,0)),
      bodies[s][i*2+1],PxTransform(PxVec3(-1.5f,0,0)));
   }
   for(PxU32 frame=0;frame<30;++frame)
   {
    // Exercise nonempty triangle certificates, then an empty particle
    // program, then recreate it in the same context and shared dispatcher.
    if(repeat==0 && (frame==0 || frame==15))
     for(PxU32 s=0;s<2;++s)
     {
      meshes[s][0]=VolumeAvbdSnippet::cookCube(*physics,PxVec3(0),1.0f);
      if(!meshes[s][0]) return 2;
      meshes[s][1]=meshes[s][0];meshes[s][1]->acquireReference();
      PxDeformableVolumeMaterial* softMaterial=physics->createDeformableVolumeMaterial(2.e5f,0.3f,0.1f);
      for(PxU32 v=0;v<2;++v)
      {
       volumes[s][v]=VolumeAvbdSnippet::createVisualVolume(*physics,*scenes[s],*meshes[s][v],*softMaterial,
        PxTransform(PxVec3(-10.0f+float(v)*1.5f,10,0),PxQuat(float(v)*0.1f,PxVec3(0,1,0))),
        PxVec3(1.0f+float(v)*0.2f,1.0f,1.0f),100.0f+float(v)*50.0f);
       if(!volumes[s][v]) return 2;
      }
      softMaterial->release();
     }
    if(repeat==0 && (frame==6 || frame==9 || frame==22 || frame==24))
     for(PxU32 s=0;s<2;++s)
     {
      const PxU32 v=(frame==9 || frame==24) ? 1 : 0;
      PX_RELEASE(volumes[s][v]);PX_RELEASE(meshes[s][v]);
     }
    if(repeat==0 && frame==4)
     for(PxU32 s=0;s<2;++s)
     {
      // Mass changes alter the writable graph while the rest topology remains
      // shared. Author the same pin in both scenes through the public buffer.
      PxVec4* positions=volumes[s][1]->getSimPositionInvMassBufferD();
      positions[0].w=0.0f;
      volumes[s][1]->markDirty(PxDeformableVolumeDataFlag::eSIM_POSITION_INVMASS);
     }
    // Two simultaneous scenes share one dispatcher. No color may block a
    // worker waiting for another queued task, including the one-worker case.
    const PxReal dt=frame%5==0 ? 1.0f/120 : 1.0f/60;
    scenes[0]->simulate(dt);scenes[1]->simulate(dt);
    pass &= scenes[0]->fetchResults(true) && scenes[1]->fetchResults(true);
    if(repeat==0)
     for(PxU32 v=0;v<2;++v)
     {
      pass &= (volumes[0][v]!=NULL)==(volumes[1][v]!=NULL);
      if(!volumes[0][v] || !volumes[1][v]) continue;
      const PxVec4* a=volumes[0][v]->getSimPositionInvMassBufferD();
      const PxVec4* b=volumes[1][v]->getSimPositionInvMassBufferD();
      for(PxU32 i=0;i<volumes[0][v]->getSimulationMesh()->getNbVertices();++i)
       pass &= a[i].isFinite() && b[i].isFinite() && a[i]==b[i];
     }
    for(PxU32 i=0;i<32;++i)
    {
     const PxTransform a=bodies[0][i]->getGlobalPose(),b=bodies[1][i]->getGlobalPose();
     pass &= a.isFinite() && a.p==b.p && a.q==b.q;
    }
   }
   for(PxU32 s=0;s<2;++s)
   {
    for(PxU32 i=0;i<16;++i) joints[s][i]->release();
    for(PxU32 i=0;i<32;++i) bodies[s][i]->release();
    scenes[s]->release();
   }
  }
  if(dispatcher) dispatcher->release();
  std::printf("[SCHEDULE_LIFECYCLE] workers=%u dispatch=%s twoScenes=8 status=%s\n",
   counts[mode],mode==4?"inline":"default",pass?"PASS":"FAIL");
 }
 pass &= errors.count==0;
 material->release();physics->release();foundation->release();
 return pass?0:1;
}
