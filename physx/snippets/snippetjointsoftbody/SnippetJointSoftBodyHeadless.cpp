// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "SnippetJointSoftBodyCommon.h"
#include "SnippetJointSoftBodyHeadless.h"
#include "SnippetJointSoftBodyShowcase.h"
#include "SnippetJointSoftBodyApp.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace JointSoftBodySnippet;
namespace
{
const char* solverName(PxSolverType::Enum s)
{
	return s==PxSolverType::eAVBD ? "avbd" : s==PxSolverType::eTGS ? "tgs" : "pgs";
}
bool number(const char* s,PxU32& result)
{
	if(!s || !*s || *s=='-') return false;
	char* end=NULL; const unsigned long n=std::strtoul(s,&end,10);
	if(*end || n>10000) return false; result=PxU32(n); return true;
}
}
namespace JointSoftBodySnippet
{
bool parseOptions(int argc,const char*const* argv,Options& options)
{
	for(int i=1;i<argc;++i)
	{
		const char* a=argv[i];
		if(std::strcmp(a,"--headless")==0) continue;
		if(std::strncmp(a,"--solver=",9)==0)
		{
			a+=9;
			if(std::strcmp(a,"avbd")==0) options.settings.solver=PxSolverType::eAVBD;
			else if(std::strcmp(a,"pgs")==0) options.settings.solver=PxSolverType::ePGS;
			else if(std::strcmp(a,"tgs")==0) options.settings.solver=PxSolverType::eTGS;
			else return false;
		}
		else if(std::strcmp(a,"--volume-material")==0) options.settings.volumeMaterial=true;
		else if(std::strcmp(a,"--pvd")==0) options.settings.pvd=true;
		else if(std::strcmp(a,"--visual-showcase")==0) options.caseName="visual-showcase";
		else if(std::strcmp(a,"--disable-sleep")==0) options.disableSleep=true;
		else if(std::strncmp(a,"--case=",7)==0) options.caseName=a+7;
		else if(std::strncmp(a,"--frames=",9)==0)
		{
			if(!number(a+9,options.frames) || options.frames==0) return false;
		}
		else if(std::strncmp(a,"--grid=",7)==0)
		{
			if(!number(a+7,options.settings.grid) || options.settings.grid<3 || options.settings.grid>10) return false;
		}
		else if(std::strncmp(a,"--dispatcher-threads=",21)==0)
		{
			if(!number(a+21,options.settings.workers) || options.settings.workers>16) return false;
		}
		else if(std::strncmp(a,"--compression-hold-frames=",26)==0)
		{
			if(!number(a+26,options.settings.compressionHoldFrames) || options.settings.compressionHoldFrames<60) return false;
		}
		else return false;
	}
	return std::strcmp(options.caseName,"showcase")==0 || std::strcmp(options.caseName,"drop")==0 ||
		std::strcmp(options.caseName,"free-flight")==0 || std::strcmp(options.caseName,"reaction")==0 ||
		std::strcmp(options.caseName,"compression-recovery")==0 || std::strcmp(options.caseName,"compression-load")==0 ||
		std::strcmp(options.caseName,"wake-recovery")==0 || std::strcmp(options.caseName,"multi-collision")==0 ||
		std::strcmp(options.caseName,"visual-showcase")==0;
}

}
namespace
{
struct Snapshot
{
	std::vector<PxTransform> poses;
	PxVec3 center, systemCenter, momentum;
	PxReal volumeRatio, minDet, shapeError, groundPenetration, externalPenetration;
	PxReal mechanicalEnergy, maxSpeed, momentumScale, maxJointAngle;
	PxU32 awake;
	bool finite;
	Snapshot() : center(0),systemCenter(0),momentum(0),volumeRatio(1),minDet(PX_MAX_F32),shapeError(0),
		groundPenetration(0),externalPenetration(0),mechanicalEnergy(0),maxSpeed(0),momentumScale(0),maxJointAngle(0),awake(0),finite(true) {}
};

// Observations only: these signed center-grid tetrahedra do not exert forces.
Snapshot sample(World& world)
{
	Snapshot s;
	const Lattice& l=world.lattice;
	s.poses.resize(l.nodes.size());
	for(PxU32 i=0;i<l.nodes.size();++i)
	{
		const PxRigidDynamic& body=*l.nodes[i];
		const PxTransform p=body.getGlobalPose();
		const PxVec3 v=body.getLinearVelocity(),w=body.getAngularVelocity();
		s.poses[i]=p; s.finite &= p.isFinite() && v.isFinite() && w.isFinite();
		s.center+=p.p*body.getMass();
		s.momentum+=v*body.getMass(); s.momentumScale+=v.magnitude()*body.getMass();
		s.awake+=!body.isSleeping();
		const PxVec3 localW=p.q.rotateInv(w);
		s.mechanicalEnergy+=0.5f*(body.getMass()*v.magnitudeSquared()+
			localW.dot(body.getMassSpaceInertiaTensor().multiply(localW)));
		s.maxSpeed=PxMax(s.maxSpeed,v.magnitude());
		if(world.ground)
		{
			const PxMat33 r(p.q);
			const PxReal halfY=0.5f*l.cell*(PxAbs(r.column0.y)+PxAbs(r.column1.y)+PxAbs(r.column2.y));
			s.groundPenetration=PxMax(s.groundPenetration,halfY-p.p.y);
		}
		if(world.platen || world.impactor)
		{
			PxVec3 direction; PxReal depth=0;
			const PxRigidDynamic* other=world.platen ? world.platen : world.impactor;
			const PxVec3 half=world.platen ? PxVec3(1.2f,0.15f,1.2f) : world.impactHalf;
			if(PxGeometryQuery::computePenetration(direction,depth,PxBoxGeometry(PxVec3(0.5f*l.cell)),p,
				PxBoxGeometry(half),other->getGlobalPose()))
				s.externalPenetration=PxMax(s.externalPenetration,depth);
		}
	}
	s.center/=l.totalMass;
	s.systemCenter=s.center;
	if(world.impactor)
	{
		const PxReal m=world.impactor->getMass();
		const PxTransform p=world.impactor->getGlobalPose();
		const PxVec3 v=world.impactor->getLinearVelocity();
		const PxVec3 w=p.q.rotateInv(world.impactor->getAngularVelocity());
		s.systemCenter=(s.center*l.totalMass+p.p*m)/(l.totalMass+m);
		s.momentum+=v*m; s.momentumScale+=v.magnitude()*m;
		s.awake+=!world.impactor->isSleeping();
		s.mechanicalEnergy+=0.5f*(m*v.magnitudeSquared()+
			w.dot(world.impactor->getMassSpaceInertiaTensor().multiply(w)));
		s.finite &= p.isFinite() && v.isFinite() && w.isFinite();
	}
	const PxQuat orientation=s.poses[s.poses.size()/2].q;
	double squaredError=0;
	for(PxU32 i=0;i<s.poses.size();++i)
		squaredError+=double((orientation.rotateInv(s.poses[i].p-s.center)-l.restLocal[i]).magnitudeSquared());
	s.shapeError=PxReal(std::sqrt(squaredError/double(s.poses.size())));
	for(PxU32 i=0;i<l.joints.size();++i)
	{
		PxRigidActor* a=NULL; PxRigidActor* b=NULL; l.joints[i]->getActors(a,b);
		const PxTransform pa=a->getGlobalPose()*l.joints[i]->getLocalPose(PxJointActorIndex::eACTOR0);
		const PxTransform pb=b->getGlobalPose()*l.joints[i]->getLocalPose(PxJointActorIndex::eACTOR1);
		const PxQuat relative=pa.q.getConjugate()*pb.q;
		s.maxJointAngle=PxMax(s.maxJointAngle,2.0f*PxAcos(PxClamp(PxAbs(relative.w),0.0f,1.0f)));
		// The observer uses the same finite material coordinates as solverPrep.
		s.mechanicalEnergy+=PxReal(l.joints[i]->elasticEnergy());
	}
	for(PxU32 i=0;i<l.materials.size();++i) s.mechanicalEnergy+=PxReal(l.materials[i]->elasticEnergy());
	static const PxU32 permutations[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
	double volume=0;
	for(PxU32 z=0;z+1<l.grid;++z)
		for(PxU32 y=0;y+1<l.grid;++y)
			for(PxU32 x=0;x+1<l.grid;++x)
				for(PxU32 t=0;t<6;++t)
				{
					PxU32 c[3]={x,y,z}; const PxU32 a=l.index(x,y,z);
					++c[permutations[t][0]]; const PxU32 b=l.index(c[0],c[1],c[2]);
					++c[permutations[t][1]]; const PxU32 d=l.index(c[0],c[1],c[2]);
					const PxU32 e=l.index(x+1,y+1,z+1);
					const PxReal current=(s.poses[b].p-s.poses[a].p).cross(s.poses[d].p-s.poses[a].p).dot(s.poses[e].p-s.poses[a].p);
					const PxReal rest=(l.restLocal[b]-l.restLocal[a]).cross(l.restLocal[d]-l.restLocal[a]).dot(l.restLocal[e]-l.restLocal[a]);
					const PxReal ratio=current/rest;
					s.minDet=PxMin(s.minDet,ratio); volume+=double(ratio)*double(PxAbs(rest))/6.0;
				}
	const double side=double(l.grid-1)*double(l.cell);
	s.volumeRatio=PxReal(volume/(side*side*side));
	if(world.impactor) s.poses.push_back(world.impactor->getGlobalPose());
	s.finite &= PxIsFinite(s.volumeRatio) && PxIsFinite(s.minDet) && PxIsFinite(s.mechanicalEnergy);
	return s;
}

void hashValue(PxU64& hash,PxReal value)
{
	PxU32 bits=0; std::memcpy(&bits,&value,sizeof(bits));
	for(PxU32 b=0;b<4;++b) { hash^=(bits>>(8*b))&255; hash*=1099511628211ULL; }
}
void hashSample(PxU64& hash,const Snapshot& sample)
{
	for(PxU32 i=0;i<sample.poses.size();++i)
	{
		const PxTransform& p=sample.poses[i];
		hashValue(hash,p.p.x); hashValue(hash,p.p.y); hashValue(hash,p.p.z);
		hashValue(hash,p.q.x); hashValue(hash,p.q.y); hashValue(hash,p.q.z); hashValue(hash,p.q.w);
	}
}
}

bool snippetJointSoftBodyHeadlessShouldRun(int argc,const char*const* argv)
{
	for(int i=1;i<argc;++i) if(std::strcmp(argv[i],"--headless")==0 || std::strncmp(argv[i],"--case=",7)==0) return true;
	return std::getenv("PHYSX_SNIPPET_HEADLESS")!=NULL;
}
int snippetJointSoftBodyHeadlessMain(int argc,const char*const* argv)
{
	Options options;
	if(!parseOptions(argc,argv,options)) { std::fprintf(stderr,"Invalid joint-soft options.\n"); return 2; }
	if(options.disableSleep && std::strcmp(options.caseName,"reaction")!=0)
	{ std::fprintf(stderr,"--disable-sleep is a separate reaction diagnostic.\n"); return 2; }
	if(std::strcmp(options.caseName,"multi-collision")==0 || std::strcmp(options.caseName,"visual-showcase")==0)
		return runJointSoftBodyMultiHeadless(options.settings,options.caseName,options.frames);
	World world;
	if(!world.init(options.settings,options.caseName)) { std::fprintf(stderr,"Joint-soft initialization failed.\n"); return 2; }
	if(options.disableSleep)
	{
		for(PxU32 i=0;i<world.lattice.nodes.size();++i) world.lattice.nodes[i]->setSleepThreshold(0);
		world.impactor->setSleepThreshold(0);
	}
	const Snapshot initial=sample(world);
	Snapshot previous=initial,current=initial;
	PxU64 hash=14695981039346656037ULL;
	hashSample(hash,initial);
	PxReal minVolume=1,minDet=1,maxShape=0,maxGround=0,maxExternal=0,maxEnergy=initial.mechanicalEnergy;
	PxReal maxDrift=0,maxPostContactStep=0,maxPostImpactVy=-PX_MAX_F32,compressionMinimum=1;
	PxReal maxStepMs=0,maxMomentumError=0,maxBallisticError=0,maxWakeMotion=0,maxJointAngle=0;
	double sumStepMs=0,sumFrameMs=0,activeStepMs=0,contactStepMs=0;
	PxU32 activeFrames=0,activeContactFrames=0,sleepingBeforeWake=0;
	PxU32 completed=0,contacts=0,firstContact=PX_MAX_U32,fetchFailures=0,nonFinite=0,zeroMotionFrames=0;
	PxU32 nodes=world.lattice.nodes.size(),joints=world.lattice.joints.size();
	const PxReal cell=world.lattice.cell, mass=world.lattice.totalMass;
	for(PxU32 frame=0;frame<options.frames;++frame)
	{
		if(frame==240) sleepingBeforeWake=nodes+(world.impactor?1:0)-previous.awake;
		const auto start=std::chrono::steady_clock::now();
		if(!world.step(frame)) { ++fetchFailures; break; }
		const auto fetched=std::chrono::steady_clock::now();
		const double stepMs=std::chrono::duration<double,std::milli>(fetched-start).count();
		current=sample(world); ++completed;
		if(std::getenv("PHYSX_AVBD_MOMENTUM_TRACE"))
			std::printf("[MOMENTUM] frame=%u p=(%.9g,%.9g,%.9g) awake=%u impactorVy=%.9g contacts=%u\n",
				frame,current.momentum.x,current.momentum.y,current.momentum.z,current.awake,
				world.impactor ? world.impactor->getLinearVelocity().y : 0,world.events.contacts);
		if(!current.finite) { ++nonFinite; break; }
		hashSample(hash,current);
		minVolume=PxMin(minVolume,current.volumeRatio); minDet=PxMin(minDet,current.minDet);
		maxShape=PxMax(maxShape,current.shapeError); maxGround=PxMax(maxGround,current.groundPenetration);
		maxExternal=PxMax(maxExternal,current.externalPenetration); maxEnergy=PxMax(maxEnergy,current.mechanicalEnergy);
		maxDrift=PxMax(maxDrift,(current.systemCenter-initial.systemCenter).magnitude());
		maxMomentumError=PxMax(maxMomentumError,(current.momentum-initial.momentum).magnitude()/PxMax(initial.momentumScale,1.0f));
		const PxVec3 ballistic=initial.center+PxVec3(0.5f,-0.25f,0.2f)*(PxReal(frame+1)/60.0f);
		maxBallisticError=PxMax(maxBallisticError,(current.center-ballistic).magnitude());
		if(frame>=120 && frame<120+options.settings.compressionHoldFrames)
			compressionMinimum=PxMin(compressionMinimum,current.volumeRatio);
		maxJointAngle=PxMax(maxJointAngle,current.maxJointAngle);
		contacts+=world.events.contacts;
		if(world.events.contacts && firstContact==PX_MAX_U32) firstContact=frame;
		PxReal maxMotion=0;
		for(PxU32 i=0;i<nodes;++i)
			maxMotion=PxMax(maxMotion,(current.poses[i].p-previous.poses[i].p).magnitude());
		if(firstContact!=PX_MAX_U32)
		{
			maxPostContactStep=PxMax(maxPostContactStep,maxMotion);
			if(world.impactor) maxPostImpactVy=PxMax(maxPostImpactVy,world.impactor->getLinearVelocity().y);
		}
		if(maxMotion==0) ++zeroMotionFrames;
		if(frame>=240 && frame<300) maxWakeMotion=PxMax(maxWakeMotion,maxMotion);
		if(previous.awake || current.awake)
		{
			++activeFrames; activeStepMs+=stepMs;
			if(world.events.contacts) { ++activeContactFrames; contactStepMs+=stepMs; }
		}
		maxStepMs=PxMax(maxStepMs,PxReal(stepMs)); sumStepMs+=stepMs;
		previous=current;
		sumFrameMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
		if(std::getenv("PHYSX_AVBD_HEADLESS_PROGRESS") && ((frame+1)%60==0 || frame+1==options.frames))
		{
			std::printf("[JOINT_SOFT_PROGRESS] frame=%u stepMs=%.6g volumeRatio=%.9g minDet=%.9g shapeError=%.9g contacts=%u centerY=%.9g\n",
				frame+1,stepMs,current.volumeRatio,current.minDet,current.shapeError,world.events.contacts,current.center.y);
			std::fflush(stdout);
		}
	}
	const bool free=std::strcmp(options.caseName,"free-flight")==0;
	const bool reaction=std::strcmp(options.caseName,"reaction")==0;
	const bool compression=std::strcmp(options.caseName,"compression-recovery")==0 || std::strcmp(options.caseName,"compression-load")==0;
	const bool wake=std::strcmp(options.caseName,"wake-recovery")==0;
	const PxReal recovered=compression ?
		(current.volumeRatio-compressionMinimum)/PxMax(1.0f-compressionMinimum,1.e-6f) : 0.0f;
	const PxReal travel=(current.center-initial.center).magnitude();
	const PxU32 materialElements=world.lattice.materials.size();
	world.release();
	const char* reason="none";
	if(completed!=options.frames || options.frames<360 ||
		(compression && options.frames<300+options.settings.compressionHoldFrames)) reason="incomplete";
	else if(nonFinite || fetchFailures || world.errors.fatal) reason="runtime";
	else if(minDet<0.2f || minVolume<0.55f || current.volumeRatio<0.9f) reason="shape_or_volume";
	else if(maxGround>0.01f*cell || maxExternal>0.01f*cell) reason="penetration";
	else if(free && (maxShape>0.002f*cell || maxEnergy>initial.mechanicalEnergy*1.05f ||
		current.mechanicalEnergy<initial.mechanicalEnergy*0.99f || maxMomentumError>0.01f || maxBallisticError>0.002f || travel<0.5f)) reason="free_flight";
	else if(!free && (!contacts || maxPostContactStep<1.e-4f)) reason="contact_motion";
	else if(compression && (compressionMinimum>0.9f || current.volumeRatio<0.94f || recovered<0.7f)) reason="recovery";
	else if(reaction && (maxDrift>0.02f || maxMomentumError>0.01f || maxPostImpactVy<-1.5f)) reason="reaction";
	else if(wake && (sleepingBeforeWake!=nodes+1 || maxWakeMotion<0.002f)) reason="wake_recovery";
	const bool pass=std::strcmp(reason,"none")==0;
	std::printf("[AVBD_GATE] schema=1 snippet=SnippetJointSoftBody solver=%s case=%s frames=%u completedFrames=%u status=%s reason=%s "
		"capability=EXPERIMENTAL validation=GATED model=%s materialElements=%u sleepEnabled=%u grid=%u nodes=%u joints=%u mass=%.9g cellSize=%.9g "
		"dispatcherThreads=%u compressionHoldFrames=%u maxJointAngle=%.9g trajectoryHash=%016llx minVolumeRatio=%.9g minCellDet=%.9g finalVolumeRatio=%.9g "
		"maxShapeError=%.9g finalShapeError=%.9g maxGroundPenetration=%.9g maxExternalPenetration=%.9g "
		"compressionMinimum=%.9g recoveredFraction=%.9g contacts=%u firstContactFrame=%u maxPostContactStep=%.9g "
		"zeroMotionFrames=%u systemComDrift=%.9g maxPostImpactVy=%.9g maxMechanicalEstimate=%.9g "
		"finalMechanicalEstimate=%.9g averageStepMs=%.9g maxStepMs=%.9g averageFrameMs=%.9g "
		"activeFrames=%u activeContactFrames=%u averageActiveStepMs=%.9g averageActiveContactMs=%.9g "
		"maxMomentumRelativeError=%.9g maxBallisticError=%.9g sleepingBeforeWake=%u maxWakeMotion=%.9g "
		"nonFinite=%u fetchFailures=%u fatalErrors=%u\n",
		solverName(options.settings.solver),options.caseName,options.frames,completed,pass?"PASS":"FAIL",reason,
		options.settings.volumeMaterial ? "rigid_cell_joint_volume" : "rigid_cell_joint",materialElements,PxU32(!options.disableSleep),
		options.settings.grid,nodes,joints,mass,cell,options.settings.workers,options.settings.compressionHoldFrames,maxJointAngle,static_cast<unsigned long long>(hash),
		minVolume,minDet,current.volumeRatio,maxShape,current.shapeError,maxGround,maxExternal,compressionMinimum,recovered,
		contacts,firstContact,maxPostContactStep,zeroMotionFrames,maxDrift,maxPostImpactVy,maxEnergy,current.mechanicalEnergy,
		completed?sumStepMs/completed:0,maxStepMs,completed?sumFrameMs/completed:0,
		activeFrames,activeContactFrames,activeFrames?activeStepMs/activeFrames:0,activeContactFrames?contactStepMs/activeContactFrames:0,
		maxMomentumError,maxBallisticError,sleepingBeforeWake,maxWakeMotion,nonFinite,fetchFailures,world.errors.fatal.load());
	std::fflush(stdout);
	return pass?0:1;
}
