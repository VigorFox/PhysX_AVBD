// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Public-API experiment only. Both models execute the production AVBD backend.
#include "../physx/snippets/snippetjointsoftbody/SnippetJointSoftBodyCommon.h"
#include "../physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDCommon.h"
#include "GuIntersectionTriangleBox.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <vector>

using namespace JointSoftBodySnippet;
namespace
{
struct Options
{
	Settings settings;
	const char* model;
	const char* caseName;
	PxU32 frames;
	bool overlapProbe;
	Options() : model("joint"), caseName("drop"), frames(360),overlapProbe(false) { settings.dampingRatio=0; }
};
bool readNumber(const char* s, PxU32& value)
{
	if(!s || !*s || *s=='-') return false;
	char* end=NULL; const unsigned long n=std::strtoul(s,&end,10);
	if(*end || n>10000) return false; value=PxU32(n); return true;
}
bool parse(int argc, char** argv, Options& o)
{
	for(int i=1;i<argc;++i)
	{
		const char* a=argv[i];
		if(!std::strcmp(a,"--probe-overlap")) o.overlapProbe=true;
		else if(std::strncmp(a,"--model=",8)==0) o.model=a+8;
		else if(std::strncmp(a,"--case=",7)==0) o.caseName=a+7;
		else if(std::strncmp(a,"--grid=",7)==0)
		{ if(!readNumber(a+7,o.settings.grid) || o.settings.grid<3 || o.settings.grid>10) return false; }
		else if(std::strncmp(a,"--frames=",9)==0)
		{ if(!readNumber(a+9,o.frames) || o.frames<360) return false; }
		else if(std::strncmp(a,"--dispatcher-threads=",21)==0)
		{ if(!readNumber(a+21,o.settings.workers) || o.settings.workers>16) return false; }
		else return false;
	}
	return (!std::strcmp(o.model,"joint") || !std::strcmp(o.model,"fem")) &&
		(!std::strcmp(o.caseName,"free-flight") || !std::strcmp(o.caseName,"drop") ||
		 !std::strcmp(o.caseName,"compression-load") || !std::strcmp(o.caseName,"showcase"));
}
struct Tet
{
	PxU32 ids[4];
	double restDet;
};
double determinant(const PxVec3& a,const PxVec3& b,const PxVec3& c,const PxVec3& d)
{
	const PxVec3 u=b-a,v=c-a,w=d-a;
	return double(u.x)*(double(v.y)*w.z-double(v.z)*w.y) -
		double(u.y)*(double(v.x)*w.z-double(v.z)*w.x) +
		double(u.z)*(double(v.x)*w.y-double(v.y)*w.x);
}
struct State
{
	std::vector<PxVec3> points;
	PxVec3 center;
	double mass, kinetic, minDet, volumeRatio, groundGap, height, width, externalDepth;
	PxU32 awake, boxCoreHits;
	bool finite;
	State() : center(0),mass(0),kinetic(0),minDet(1),volumeRatio(1),groundGap(PX_MAX_F32),height(0),width(0),externalDepth(0),awake(0),boxCoreHits(0),finite(true) {}
};
struct Experiment
{
	World world;
	PxDeformableVolume* volume;
	PxDeformableVolumeMesh* mesh;
	std::vector<Tet> tets;
	std::vector<Tet> collisionTets;
	std::vector<std::array<PxU32,3> > boundary;
	std::vector<PxU32> topIds;
	std::vector<PxReal> topAreas;
	double restVolume, loadArea;
	Experiment() : volume(NULL),mesh(NULL),restVolume(0),loadArea(0) {}
	~Experiment() { PX_RELEASE(volume); PX_RELEASE(mesh); }
	bool init(const Options& o)
	{
		const bool joint=!std::strcmp(o.model,"joint");
		if(joint)
		{
			if(!world.init(o.settings,o.caseName)) return false;
			if(o.overlapProbe && world.impactor) world.impactor->setGlobalPose(PxTransform(PxVec3(0,1,0)));
			const PxU32 g=world.lattice.grid;
			static const PxU32 perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
			for(PxU32 z=0;z+1<g;++z) for(PxU32 y=0;y+1<g;++y) for(PxU32 x=0;x+1<g;++x)
				for(PxU32 t=0;t<6;++t)
				{
					PxU32 c[3]={x,y,z}; Tet tet;
					tet.ids[0]=world.lattice.index(x,y,z);
					++c[perm[t][0]]; tet.ids[1]=world.lattice.index(c[0],c[1],c[2]);
					++c[perm[t][1]]; tet.ids[2]=world.lattice.index(c[0],c[1],c[2]);
					tet.ids[3]=world.lattice.index(x+1,y+1,z+1); tets.push_back(tet);
				}
			loadArea=double(g)*g*world.lattice.cell*world.lattice.cell;
		}
		else
		{
			world.settings=o.settings; world.caseName=o.caseName;
			world.foundation=PxCreateFoundation(PX_PHYSICS_VERSION,world.allocator,world.errors);
			if(!world.foundation) return false;
			world.physics=PxCreatePhysics(PX_PHYSICS_VERSION,*world.foundation,PxTolerancesScale());
			if(!world.physics || !PxInitExtensions(*world.physics,NULL)) return false;
			world.extensions=true;
			world.dispatcher=PxDefaultCpuDispatcherCreate(o.settings.workers);
			if(!world.dispatcher) return false;
			const bool free=!std::strcmp(o.caseName,"free-flight");
			PxSceneDesc desc(world.physics->getTolerancesScale());
			desc.solverType=PxSolverType::eAVBD; desc.cpuDispatcher=world.dispatcher;
			desc.gravity=free ? PxVec3(0) : PxVec3(0,-9.81f,0);
			desc.filterShader=filter; desc.simulationEventCallback=&world.events;
			world.scene=world.physics->createScene(desc);
			world.material=world.physics->createMaterial(0.5f,0.5f,0);
			if(!world.scene || !world.material) return false;
			if(!free)
			{
				world.ground=PxCreatePlane(*world.physics,PxPlane(0,1,0,0),*world.material);
				if(!world.ground || !world.scene->addActor(*world.ground)) return false;
			}
			const PxU32 g=o.settings.grid, side=g+1;
			const PxReal h=o.settings.size/PxReal(g);
			PxArray<PxVec3> vertices; PxArray<PxU32> indices;
			for(PxU32 z=0;z<=g;++z) for(PxU32 y=0;y<=g;++y) for(PxU32 x=0;x<=g;++x)
				vertices.pushBack(PxVec3(PxReal(x),PxReal(y),PxReal(z))*h-PxVec3(o.settings.size*0.5f));
			static const PxU32 perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
			for(PxU32 z=0;z<g;++z) for(PxU32 y=0;y<g;++y) for(PxU32 x=0;x<g;++x)
				for(PxU32 t=0;t<6;++t)
				{
					PxU32 c[3]={x,y,z}; PxU32 ids[4]={x+side*(y+side*z),0,0,(x+1)+side*((y+1)+side*(z+1))};
					++c[perm[t][0]]; ids[1]=c[0]+side*(c[1]+side*c[2]);
					++c[perm[t][1]]; ids[2]=c[0]+side*(c[1]+side*c[2]);
					if(determinant(vertices[ids[0]],vertices[ids[1]],vertices[ids[2]],vertices[ids[3]])<0)
						std::swap(ids[1],ids[2]);
					for(PxU32 j=0;j<4;++j) indices.pushBack(ids[j]);
				}
			PxCookingParams params(world.physics->getTolerancesScale()); params.buildGPUData=false;
			const PxTetrahedronMeshDesc meshDesc(vertices,indices);
			mesh=PxCreateDeformableVolumeMesh(params,meshDesc,meshDesc,PxDeformableVolumeSimulationDataDesc(),world.physics->getPhysicsInsertionCallback());
			PxDeformableVolumeMaterial* material=world.physics->createDeformableVolumeMaterial(o.settings.young,o.settings.poisson,0.5f,0.0f);
			if(!mesh || !material) { PX_RELEASE(material); return false; }
			// Match the current volume showcase rather than PhysX's default model.
			material->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
			const PxReal centerY=(!std::strcmp(o.caseName,"drop") || free) ? 3.0f : 1.0f;
			const PxQuat rotation=free ? PxQuat(0.37f,PxVec3(1,2,3).getNormalized()) : PxQuat(PxIdentity);
			volume=VolumeAvbdSnippet::createVisualVolume(*world.physics,*world.scene,*mesh,*material,
				PxTransform(PxVec3(0,centerY,0),rotation),PxVec3(1),o.settings.density);
			PX_RELEASE(material);
			if(!volume) return false;
			volume->setLinearDamping(0);
			if(free && !VolumeAvbdSnippet::setVisualVelocity(*volume,PxVec3(0.5f,-0.25f,0.2f))) return false;
			const PxTetrahedronMesh& cooked=*volume->getSimulationMesh();
			const bool small=cooked.getTetrahedronMeshFlags() & PxTetrahedronMeshFlag::e16_BIT_INDICES;
			for(PxU32 i=0;i<cooked.getNbTetrahedrons();++i)
			{
				Tet tet;
				for(PxU32 j=0;j<4;++j) tet.ids[j]=small ?
					static_cast<const PxU16*>(cooked.getTetrahedrons())[4*i+j] : static_cast<const PxU32*>(cooked.getTetrahedrons())[4*i+j];
				tets.push_back(tet);
			}
			const PxTetrahedronMesh& collision=*volume->getCollisionMesh();
			const bool smallCollision=collision.getTetrahedronMeshFlags() & PxTetrahedronMeshFlag::e16_BIT_INDICES;
			std::map<std::array<PxU32,3>,PxU32> faces;
			static const PxU32 faceCorners[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
			for(PxU32 i=0;i<collision.getNbTetrahedrons();++i)
			{
				Tet tet;
				for(PxU32 j=0;j<4;++j) tet.ids[j]=smallCollision ?
					static_cast<const PxU16*>(collision.getTetrahedrons())[4*i+j] : static_cast<const PxU32*>(collision.getTetrahedrons())[4*i+j];
				collisionTets.push_back(tet);
				for(PxU32 f=0;f<4;++f)
				{
					std::array<PxU32,3> key={{tet.ids[faceCorners[f][0]],tet.ids[faceCorners[f][1]],tet.ids[faceCorners[f][2]]}};
					std::sort(key.begin(),key.end()); ++faces[key];
				}
			}
			for(const auto& entry:faces)
			{
				if(entry.second>2) return false;
				if(entry.second==1) boundary.push_back(entry.first);
			}
			// Integrate uniform pressure over the top boundary of the cooked mesh.
			// This handles cooking remaps without assuming source vertex IDs survive.
			const PxVec3* rest=cooked.getVertices();
			for(PxU32 i=0;i<cooked.getNbVertices();++i)
				if(PxAbs(rest[i].y-o.settings.size*0.5f)<1.e-5f)
				{
					const PxReal wx=PxAbs(PxAbs(rest[i].x)-o.settings.size*0.5f)<1.e-5f ? 0.5f : 1.0f;
					const PxReal wz=PxAbs(PxAbs(rest[i].z)-o.settings.size*0.5f)<1.e-5f ? 0.5f : 1.0f;
					topIds.push_back(i); topAreas.push_back(h*h*wx*wz); loadArea+=double(topAreas.back());
				}
			if(!std::strcmp(o.caseName,"showcase"))
			{
				world.impactor=PxCreateDynamic(*world.physics,PxTransform(PxVec3(0,o.overlapProbe?1.0f:3.1f,0)),PxBoxGeometry(world.impactHalf),*world.material,1);
				if(!world.impactor) return false;
				PxRigidBodyExt::setMassAndUpdateInertia(*world.impactor,80.0f);
				world.impactor->setLinearDamping(0); world.impactor->setAngularDamping(0);
				if(!world.scene->addActor(*world.impactor)) return false;
			}
		}
		State initial=sample();
		for(Tet& tet:tets)
		{
			tet.restDet=determinant(initial.points[tet.ids[0]],initial.points[tet.ids[1]],initial.points[tet.ids[2]],initial.points[tet.ids[3]]);
			if(std::abs(tet.restDet)<1.e-12) return false;
			restVolume+=std::abs(tet.restDet)/6.0;
		}
		return initial.finite && std::abs(initial.mass-800.0)<0.1 && std::abs(loadArea-4.0)<1.e-4;
	}
	void observeImpactor(State& s) const
	{
		if(!world.impactor) return;
		const PxTransform box=world.impactor->getGlobalPose();
		const PxVec3 v=world.impactor->getLinearVelocity(),w=box.q.rotateInv(world.impactor->getAngularVelocity());
		s.awake+=!world.impactor->isSleeping();
		s.kinetic+=0.5*(double(world.impactor->getMass())*v.magnitudeSquared()+double(w.dot(world.impactor->getMassSpaceInertiaTensor().multiply(w))));
		s.finite &= box.isFinite() && v.isFinite() && w.isFinite();
		if(!volume)
		{
			for(PxRigidDynamic* body:world.lattice.nodes)
			{
				PxVec3 direction; PxReal depth=0;
				if(PxGeometryQuery::computePenetration(direction,depth,PxBoxGeometry(PxVec3(0.5f*world.lattice.cell)),body->getGlobalPose(),
					PxBoxGeometry(world.impactHalf),box)) s.externalDepth=std::max(s.externalDepth,double(depth));
			}
			return;
		}
		const PxVec4* p=volume->getPositionInvMassBufferD();
		// Deep-overlap test on the actual collision boundary, plus containment.
		// The inset is the same 1%-of-cell allowance as the rigid-cell gate.
		const PxVec3 core=world.impactHalf-PxVec3(0.01f*world.settings.size/world.settings.grid);
		for(const auto& face:boundary)
		{
			const PxVec3 a=box.transformInv(p[face[0]].getXYZ()),b=box.transformInv(p[face[1]].getXYZ()),c=box.transformInv(p[face[2]].getXYZ());
			if(Gu::intersectTriangleBox_ReferenceCode(PxVec3(0),core,a,b,c)) ++s.boxCoreHits;
		}
		for(PxU32 i=0;i<volume->getCollisionMesh()->getNbVertices();++i)
		{
			const PxVec3 local=box.transformInv(p[i].getXYZ());
			const double depth=std::min(double(world.impactHalf.x-PxAbs(local.x)),std::min(double(world.impactHalf.y-PxAbs(local.y)),double(world.impactHalf.z-PxAbs(local.z))));
			s.externalDepth=std::max(s.externalDepth,depth);
		}
		for(PxU32 corner=0;corner<8;++corner)
		{
			const PxVec3 q=box.transform(PxVec3((corner&1)?core.x:-core.x,(corner&2)?core.y:-core.y,(corner&4)?core.z:-core.z));
			for(const Tet& t:collisionTets)
			{
				const PxVec3 a=p[t.ids[0]].getXYZ(),b=p[t.ids[1]].getXYZ(),c=p[t.ids[2]].getXYZ(),d=p[t.ids[3]].getXYZ();
				const double det=determinant(a,b,c,d);
				if(std::abs(det)<1.e-12) { s.finite=false; continue; }
				const double weights[4]={determinant(q,b,c,d)/det,determinant(a,q,c,d)/det,determinant(a,b,q,d)/det,determinant(a,b,c,q)/det};
				bool inside=true;
				for(double weight:weights) inside &= weight>=-1.e-7 && weight<=1.0000001;
				if(inside) { ++s.boxCoreHits; break; }
			}
		}
	}
	State sample() const
	{
		State s;
		double minY=PX_MAX_F32,maxY=-PX_MAX_F32,minX=PX_MAX_F32,maxX=-PX_MAX_F32;
		if(volume)
		{
			const PxU32 n=volume->getSimulationMesh()->getNbVertices();
			const PxVec4* p=volume->getSimPositionInvMassBufferD();
			const PxVec4* v=volume->getSimVelocityBufferD();
			for(PxU32 i=0;i<n;++i)
			{
				const PxReal mass=p[i].w>0 ? 1.0f/p[i].w : 0;
				s.points.push_back(p[i].getXYZ()); s.mass+=mass; s.center+=p[i].getXYZ()*mass;
				s.kinetic+=0.5*double(mass)*v[i].getXYZ().magnitudeSquared();
				s.finite &= p[i].isFinite() && v[i].isFinite() && mass>0;
			}
			s.awake=!volume->isSleeping();
			const PxVec4* collision=volume->getPositionInvMassBufferD();
			for(PxU32 i=0;i<volume->getCollisionMesh()->getNbVertices();++i)
			{
				minY=std::min(minY,double(collision[i].y)); maxY=std::max(maxY,double(collision[i].y));
				minX=std::min(minX,double(collision[i].x)); maxX=std::max(maxX,double(collision[i].x));
				s.finite &= collision[i].isFinite();
			}
		}
		else
		{
			for(PxRigidDynamic* body:world.lattice.nodes)
			{
				const PxTransform p=body->getGlobalPose(); const PxVec3 v=body->getLinearVelocity();
				const PxVec3 w=p.q.rotateInv(body->getAngularVelocity()); const PxReal m=body->getMass();
				s.points.push_back(p.p); s.mass+=m; s.center+=p.p*m; s.awake+=!body->isSleeping();
				s.kinetic+=0.5*(double(m)*v.magnitudeSquared()+double(w.dot(body->getMassSpaceInertiaTensor().multiply(w))));
				s.finite &= p.isFinite() && v.isFinite() && w.isFinite();
				const PxMat33 r(p.q);
				const double extent=0.5*world.lattice.cell*(PxAbs(r.column0.y)+PxAbs(r.column1.y)+PxAbs(r.column2.y));
				minY=std::min(minY,double(p.p.y)-extent); maxY=std::max(maxY,double(p.p.y)+extent);
				const double extentX=0.5*world.lattice.cell*(PxAbs(r.column0.x)+PxAbs(r.column1.x)+PxAbs(r.column2.x));
				minX=std::min(minX,double(p.p.x)-extentX); maxX=std::max(maxX,double(p.p.x)+extentX);
			}
		}
		s.height=maxY-minY;
		s.width=maxX-minX;
		if(world.ground) s.groundGap=minY;
		if(s.mass>0) s.center/=PxReal(s.mass); else s.finite=false;
		if(restVolume>0)
		{
			double currentVolume=0;
			for(const Tet& tet:tets)
			{
				const double d=determinant(s.points[tet.ids[0]],s.points[tet.ids[1]],s.points[tet.ids[2]],s.points[tet.ids[3]]);
				s.minDet=std::min(s.minDet,d/tet.restDet); currentVolume+=d/tet.restDet*std::abs(tet.restDet)/6.0;
			}
			s.volumeRatio=currentVolume/restVolume;
		}
		observeImpactor(s);
		s.finite &= s.center.isFinite() && std::isfinite(s.minDet) && std::isfinite(s.volumeRatio) && std::isfinite(s.kinetic);
		return s;
	}
	bool step(PxU32 frame)
	{
		if(volume && !std::strcmp(world.caseName,"compression-load") && frame>=60 && frame<180)
		{
			const PxReal pressure=world.settings.young*0.1f*PxClamp((PxReal(frame)-60.0f)/60.0f,0.0f,1.0f);
			PxVec4* velocity=volume->getSimVelocityBufferD(); const PxVec4* p=volume->getSimPositionInvMassBufferD();
			for(PxU32 i=0;i<topIds.size();++i) velocity[topIds[i]].y-=pressure*topAreas[i]*p[topIds[i]].w/60.0f;
			volume->markDirty(PxDeformableVolumeDataFlag::eSIM_VELOCITY); volume->setWakeCounter(0.4f);
		}
		return world.step(frame);
	}
};
void hashValue(PxU64& h,PxReal value)
{
	PxU32 bits=0; std::memcpy(&bits,&value,4);
	for(PxU32 i=0;i<4;++i) { h^=(bits>>(8*i))&255; h*=1099511628211ULL; }
}
}
int main(int argc,char** argv)
{
	Options o;
	if(!parse(argc,argv,o)) { std::fprintf(stderr,"Invalid soft benchmark options.\n"); return 2; }
	Experiment e;
	const auto initStart=std::chrono::steady_clock::now();
	if(!e.init(o)) { std::fprintf(stderr,"Soft benchmark initialization failed.\n"); return 2; }
	const double initMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-initStart).count();
	const State initial=e.sample(); State previous=initial,current=initial;
	if(o.overlapProbe)
	{
		const bool detected=e.world.impactor && (initial.boxCoreHits || initial.externalDepth>0.01*o.settings.size/o.settings.grid);
		std::printf("[SOFT_OVERLAP_PROBE] model=%s status=%s externalDepth=%.9g boxCoreHits=%u\n",o.model,detected?"PASS":"FAIL",initial.externalDepth,initial.boxCoreHits);
		return detected?0:1;
	}
	double totalMs=0,activeMs=0,groundMs=0,holdMs=0,impactMs=0,maxMs=0,minDet=1,minVolume=1,maxGround=0;
	double maxBallisticError=0,maxPointBallisticError=0,maxEnergy=initial.kinetic,holdEndVolume=1,holdEndHeight=1,holdEndWidth=1,maxMotion=0,maxExternal=initial.externalDepth;
	PxU32 maxBoxCoreHits=initial.boxCoreHits;
	std::vector<double> activeTimes;
	PxU32 completed=0,activeFrames=0,groundFrames=0,zeroActiveMotion=0;
	const PxU32 impactBegin=!std::strcmp(o.caseName,"showcase") ? 15 : 30;
	const PxU32 impactEnd=!std::strcmp(o.caseName,"showcase") ? 60 : 90;
	PxU32 impactActiveFrames=0,holdActiveFrames=0;
	PxU64 hash=14695981039346656037ULL;
	for(PxU32 frame=0;frame<o.frames;++frame)
	{
		const auto start=std::chrono::steady_clock::now();
		if(!e.step(frame)) break;
		const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
		current=e.sample(); ++completed; totalMs+=ms; maxMs=std::max(maxMs,ms);
		if(!current.finite) break;
		minDet=std::min(minDet,current.minDet); minVolume=std::min(minVolume,current.volumeRatio);
		maxGround=std::max(maxGround,-current.groundGap); maxEnergy=std::max(maxEnergy,current.kinetic);
		maxExternal=std::max(maxExternal,current.externalDepth); maxBoxCoreHits=std::max(maxBoxCoreHits,current.boxCoreHits);
		maxBallisticError=std::max(maxBallisticError,double((current.center-initial.center-
			PxVec3(0.5f,-0.25f,0.2f)*(PxReal(frame+1)/60.0f)).magnitude()));
		double motion=0;
		for(PxU32 i=0;i<current.points.size();++i)
		{
			motion=std::max(motion,double((current.points[i]-previous.points[i]).magnitude()));
			maxPointBallisticError=std::max(maxPointBallisticError,double((current.points[i]-initial.points[i]-
				PxVec3(0.5f,-0.25f,0.2f)*(PxReal(frame+1)/60.0f)).magnitude()));
			for(PxU32 k=0;k<3;++k) hashValue(hash,current.points[i][k]);
		}
		for(PxRigidDynamic* body:e.world.lattice.nodes)
		{
			const PxQuat q=body->getGlobalPose().q;
			hashValue(hash,q.x); hashValue(hash,q.y); hashValue(hash,q.z); hashValue(hash,q.w);
		}
		if(e.world.impactor)
		{
			const PxTransform p=e.world.impactor->getGlobalPose();
			for(PxU32 i=0;i<3;++i) hashValue(hash,p.p[i]);
			hashValue(hash,p.q.x); hashValue(hash,p.q.y); hashValue(hash,p.q.z); hashValue(hash,p.q.w);
		}
		maxMotion=std::max(maxMotion,motion);
		if(previous.awake || current.awake)
		{
			++activeFrames; activeMs+=ms; activeTimes.push_back(ms); if(motion==0) ++zeroActiveMotion;
			// Shared geometric observation, not an NP-contact or penetration test.
			// Production FEM has a 0.04 m contact radius at the default length scale.
			if(current.groundGap<=0.0401) { ++groundFrames; groundMs+=ms; }
		}
		if(frame>=120 && frame<180) { holdMs+=ms; holdActiveFrames+=bool(previous.awake || current.awake); }
		if(frame>=impactBegin && frame<impactEnd) { impactMs+=ms; impactActiveFrames+=bool(previous.awake || current.awake); }
		if(frame==179) { holdEndVolume=current.volumeRatio; holdEndHeight=current.height/initial.height; holdEndWidth=current.width/initial.width; }
		if((frame+1)%60==0)
		{
			std::printf("[SOFT_MODEL_PROGRESS] model=%s case=%s grid=%u frame=%u stepMs=%.9g volumeRatio=%.9g minDet=%.9g heightRatio=%.9g groundGap=%.9g awake=%u\n",
				o.model,o.caseName,o.settings.grid,frame+1,ms,current.volumeRatio,current.minDet,current.height/initial.height,current.groundGap,current.awake);
			std::fflush(stdout);
		}
		previous=current;
	}
	const bool free=!std::strcmp(o.caseName,"free-flight"), pressure=!std::strcmp(o.caseName,"compression-load");
	const double cell=double(o.settings.size)/o.settings.grid;
	const double recovery=(current.height/initial.height-holdEndHeight)/std::max(1.0-holdEndHeight,1.e-6);
	std::sort(activeTimes.begin(),activeTimes.end());
	const double p95=activeTimes.empty() ? 0 : activeTimes[PxU32(std::ceil(0.95*activeTimes.size()))-1];
	const char* reason="none";
	if(completed!=o.frames || !current.finite || e.world.errors.fatal) reason="runtime";
	else if(minDet<0.2 || minVolume<0.55 || current.volumeRatio<0.9) reason="shape_or_volume";
	else if(maxGround>0.01*cell) reason="penetration";
	else if(maxExternal>0.01*cell || maxBoxCoreHits) reason="external_penetration";
	else if(free && (maxBallisticError>0.002 || maxPointBallisticError>0.002 || maxEnergy>initial.kinetic*1.05 || current.kinetic<initial.kinetic*0.99)) reason="free_flight";
	else if(!free && (groundFrames==0 || maxMotion<1.e-4)) reason="contact_motion";
	// Poisson expansion makes volume response different; test axial compression.
	else if(pressure && (holdEndHeight>0.95 || current.height/initial.height<0.94 || current.volumeRatio<0.94 || recovery<0.7)) reason="pressure_recovery";
	const bool pass=!std::strcmp(reason,"none");
	std::printf("[AVBD_GATE] schema=1 snippet=AvbdSoftModelBenchmark capability=EXPERIMENTAL validation=GATED model=%s case=%s solver=avbd grid=%u workers=%u frames=%u completedFrames=%u status=%s reason=%s "
		"constitutiveModel=%s blocks=%u dofs=%u materialElements=%u mass=%.9g size=%.9g cellSize=%.9g young=%.9g poisson=%.9g damping=0 dt=%.9g loadArea=%.9g "
		"initMs=%.9g averageStepMs=%.9g averageActiveMs=%.9g averageGroundedActiveMs=%.9g holdAverageMs=%.9g impactWindowAverageMs=%.9g activeP95Ms=%.9g maxStepMs=%.9g "
		"activeFrames=%u groundedActiveFrames=%u zeroActiveMotionFrames=%u impactWindowBegin=%u impactWindowEnd=%u impactActiveFrames=%u holdActiveFrames=%u minDet=%.9g minVolumeRatio=%.9g finalVolumeRatio=%.9g "
		"holdEndVolumeRatio=%.9g holdEndHeightRatio=%.9g holdEndWidthRatio=%.9g heightRecoveredFraction=%.9g finalHeightRatio=%.9g groundObservationDistance=0.0401 finalGroundGap=%.9g maxGroundPenetration=%.9g maxExternalPenetration=%.9g maxBoxCoreHits=%u impactorMass=%.9g maxKinetic=%.9g maxMotion=%.9g maxBallisticError=%.9g maxPointBallisticError=%.9g kineticRatio=%.9g fatalErrors=%u trajectoryHash=%016llx\n",
		o.model,o.caseName,o.settings.grid,o.settings.workers,o.frames,completed,pass?"PASS":"FAIL",reason,
		e.volume?"stable_neo_hookean":"face_beam_lattice",
		PxU32(initial.points.size())+(e.world.impactor?1:0),PxU32(initial.points.size())*(e.volume?3:6)+(e.world.impactor?6:0),e.volume?PxU32(e.tets.size()):e.world.lattice.joints.size(),
		initial.mass,double(o.settings.size),cell,double(o.settings.young),double(o.settings.poisson),1.0/60.0,e.loadArea,
		initMs,completed?totalMs/completed:0,activeFrames?activeMs/activeFrames:0,groundFrames?groundMs/groundFrames:0,holdMs/60.0,impactMs/double(impactEnd-impactBegin),p95,maxMs,
		activeFrames,groundFrames,zeroActiveMotion,impactBegin,impactEnd,impactActiveFrames,holdActiveFrames,minDet,minVolume,current.volumeRatio,holdEndVolume,holdEndHeight,holdEndWidth,recovery,current.height/initial.height,current.groundGap,maxGround,maxExternal,maxBoxCoreHits,
		e.world.impactor?double(e.world.impactor->getMass()):0,maxEnergy,maxMotion,maxBallisticError,maxPointBallisticError,
		initial.kinetic>0 ? current.kinetic/initial.kinetic : 0,e.world.errors.fatal.load(),static_cast<unsigned long long>(hash));
	return pass?0:1;
}
