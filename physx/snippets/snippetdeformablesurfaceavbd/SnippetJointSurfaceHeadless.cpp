// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "SnippetJointSurface.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace JointSurfaceSnippet
{
namespace
{
struct Observation
{
	PxVec3 center,momentum;
    PxVec3d angular=PxVec3d(0);
	double energy;
	PxReal mass,minY,penetration,selfPenetration,principal,area;
	PxU32 awake;
	bool finite;
	Observation() : center(0),momentum(0),energy(0),mass(0),minY(PX_MAX_F32),penetration(0),selfPenetration(0),principal(0),area(0),awake(0),finite(true) {}
};
PxVec3d asDouble(const PxVec3& v){return PxVec3d(v.x,v.y,v.z);}
void bodyState(const PxRigidDynamic& body,const PxVec3& gravity,Observation& o)
{
	const PxTransform pose=body.getGlobalPose()*body.getCMassLocalPose();
	const PxVec3 v=body.getLinearVelocity(),w=pose.q.rotateInv(body.getAngularVelocity());
	const PxReal mass=body.getMass();
	o.center+=pose.p*mass;o.mass+=mass;o.momentum+=v*mass;o.awake+=!body.isSleeping();
    o.angular+=asDouble(pose.p).cross(asDouble(v))*double(mass)+asDouble(pose.q.rotate(body.getMassSpaceInertiaTensor().multiply(w)));
	o.energy+=.5*double(mass*v.magnitudeSquared()+w.dot(body.getMassSpaceInertiaTensor().multiply(w)))-double(mass*gravity.dot(pose.p));
	o.finite&=pose.isFinite() && v.isFinite() && w.isFinite();
}
PxReal penetration(const PxShape& a,const PxTransform& pa,const PxShape& b,const PxTransform& pb)
{
	PxVec3 direction;PxReal depth=0;
	return PxGeometryQuery::computePenetration(direction,depth,a.getGeometry(),pa,b.getGeometry(),pb)?depth:0;
}
Observation observe(SheetWorld& world)
{
	Observation o;world.updateSkin();
	const auto& nodes=world.host.lattice.nodes;
	std::vector<PxShape*> shapes(nodes.size());std::vector<PxTransform> poses(nodes.size());std::vector<PxBounds3> bounds(nodes.size());
	PxRigidActor* external[3]={world.host.ground,world.obstacle,world.host.impactor};
	for(PxU32 i=0;i<nodes.size();++i)
	{
		bodyState(*nodes[i],world.host.scene->getGravity(),o);
		nodes[i]->getShapes(&shapes[i],1);poses[i]=nodes[i]->getGlobalPose()*shapes[i]->getLocalPose();
		o.finite&=PxGeometryQuery::computeGeomBounds(bounds[i],shapes[i]->getGeometry(),poses[i]);
		o.minY=PxMin(o.minY,bounds[i].minimum.y);
		for(PxU32 e=0;e<3;++e) if(external[e])
		{
			PxShape* shape=NULL;external[e]->getShapes(&shape,1);
			o.penetration=PxMax(o.penetration,penetration(*shapes[i],poses[i],*shape,external[e]->getGlobalPose()*shape->getLocalPose()));
		}
		for(PxU32 j=0;j<i;++j)
			if(bounds[i].intersects(bounds[j]) && !JointSoftBodySnippet::excludedCellPair(shapes[i]->getSimulationFilterData(),shapes[j]->getSimulationFilterData()))
				o.selfPenetration=PxMax(o.selfPenetration,penetration(*shapes[i],poses[i],*shapes[j],poses[j]));
	}
	if(world.host.impactor) bodyState(*world.host.impactor,world.host.scene->getGravity(),o);
	for(PxU32 j=0;j<world.host.lattice.joints.size();++j) o.energy+=world.host.lattice.joints[j]->elasticEnergy();
	o.center/=o.mass;
	// Independent display deformation observation; it never writes actor state.
	double restArea=0,area=0;
	for(PxU32 i=0;i<world.triangles.size();i+=3)
	{
		const PxU32 a=world.triangles[i],b=world.triangles[i+1],c=world.triangles[i+2];
		const PxVec3 r0=world.restVertices[b]-world.restVertices[a],r1=world.restVertices[c]-world.restVertices[a];
		const PxVec3 e0=world.vertices[b]-world.vertices[a],e1=world.vertices[c]-world.vertices[a];
		const PxReal length=r0.magnitude(),twiceArea=r0.cross(r1).magnitude();
		const PxVec3 f0=e0/length,f1=(e1-f0*(r0.dot(r1)/length))*(length/twiceArea);
		const double aa=f0.dot(f0),bb=f1.dot(f1),ab=f0.dot(f1);
		o.principal=PxMax(o.principal,PxReal(std::sqrt(.5*(aa+bb+std::sqrt((aa-bb)*(aa-bb)+4*ab*ab)))));
		restArea+=twiceArea;area+=e0.cross(e1).magnitude();
		o.finite&=world.vertices[a].isFinite() && world.vertices[b].isFinite() && world.vertices[c].isFinite();
	}
	o.area=PxReal(area/restArea);
	o.finite&=std::isfinite(o.energy) && o.center.isFinite() && o.momentum.isFinite() && PxIsFinite(o.principal) && PxIsFinite(o.area);
	return o;
}
}
int runHeadless(const Options& options)
{
	SheetWorld world;if(!world.init(options)) return 2;
	const Observation initial=observe(world);Observation current=initial;
    double maxAngularDrift=0;
	PxReal peakStrain=initial.principal,peakPenetration=initial.penetration,peakSelf=initial.selfPenetration,maxMomentum=0,maxBallistic=0;
	double peakEnergy=initial.energy,observerMs=0;std::vector<double> timings;
	PxU32 completed=0,nonFinite=initial.finite?0:1,fetchFailures=0;
	for(PxU32 frame=0;frame<options.frames && !nonFinite;++frame)
	{
		const auto start=std::chrono::steady_clock::now();const bool ok=world.step();const auto fetched=std::chrono::steady_clock::now();
		timings.push_back(std::chrono::duration<double,std::milli>(fetched-start).count());
		if(!ok) { ++fetchFailures;break; }
		++completed;current=observe(world);if(!current.finite) ++nonFinite;
        const auto angularDrift=current.angular-initial.angular-asDouble(initial.center).cross(asDouble(current.momentum-initial.momentum));
        maxAngularDrift=PxMax(maxAngularDrift,angularDrift.magnitude());
		observerMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fetched).count();
		peakStrain=PxMax(peakStrain,current.principal);peakEnergy=PxMax(peakEnergy,current.energy);
		peakPenetration=PxMax(peakPenetration,current.penetration);peakSelf=PxMax(peakSelf,current.selfPenetration);
		maxMomentum=PxMax(maxMomentum,(current.momentum-initial.momentum).magnitude()/PxMax(1.0f,initial.momentum.magnitude()));
		maxBallistic=PxMax(maxBallistic,(current.center-initial.center-initial.momentum/initial.mass*(PxReal(completed)/60)).magnitude());
		if(std::getenv("PHYSX_AVBD_MOMENTUM_TRACE"))
			std::printf("[MOMENTUM] frame=%u p=(%.9g,%.9g,%.9g) center=(%.9g,%.9g,%.9g) contacts=%u\n",
				completed,current.momentum.x,current.momentum.y,current.momentum.z,current.center.x,current.center.y,current.center.z,world.host.events.contacts);
		if(std::getenv("PHYSX_AVBD_HEADLESS_PROGRESS"))
		{
			std::printf("[JOINT_SURFACE_FRAME] frame=%u stepMs=%.9g minY=%.9g strain=%.9g penetration=%.9g contacts=%u\n",completed,timings.back(),current.minY,current.principal,current.penetration,world.host.events.contacts);std::fflush(stdout);
		}
	}
	const bool release=std::strcmp(options.caseName,"strain-release")==0;
	const bool free=std::strcmp(options.caseName,"free-flight")==0 || release,reaction=std::strcmp(options.caseName,"reaction")==0;
	// A flat sheet arrested at the box top is not a successful drape. Use a
	// scene-scale requirement independent of solver residuals and sleep state.
	double drapeDepth=0,requiredDrapeDepth=0;
	if(world.obstacle)
	{
		const PxBounds3 support=world.obstacle->getWorldBounds(1.0f);
		const PxVec3 half=support.getExtents();
		drapeDepth=double(support.maximum.y)-current.minY;
		requiredDrapeDepth=.5*PxMax(0.0f,SheetWorld::size()*.5f-PxMax(half.x,half.z));
	}
	const bool draped=drapeDepth>=requiredDrapeDepth;
	const PxU32 fatal=world.host.errors.fatal.load();
	const char* reason="none";
	if(fatal || fetchFailures) reason="simulation_failure";
	else if(nonFinite) reason="non_finite";
	else if(completed!=options.frames || options.frames<(free && !release?60u:180u)) reason="incomplete";
	else if(peakPenetration>1e-3f || peakSelf>1e-3f) reason="penetration";
	else if(peakStrain>1.6f || current.principal>1.3f) reason="overstretch";
	else if(peakEnergy>1.05*initial.energy+1e-4) reason="energy_growth";
	else if((free || reaction) && (maxMomentum>5e-3f || maxBallistic>1e-3f)) reason="momentum_or_ballistic";
	else if(reaction && world.host.events.crossContacts==0) reason="missing_reaction";
	else if(!free && !reaction && (world.host.events.contacts==0 || current.center.y>2.5f || current.minY<-.001f || !draped)) reason="missing_drape";
	std::printf("[JOINT_SURFACE_QUALITY] acceptance=engineering-drape-20261111 drapeDepth=%.9g requiredDrapeDepth=%.9g drapePass=%u\n",
		drapeDepth,requiredDrapeDepth,unsigned(draped));
	double sum=0;for(double ms:timings) sum+=ms;std::sort(timings.begin(),timings.end());
	std::printf("[AVBD_GATE] schema=1 snippet=SnippetDeformableSurfaceAVBD solver=%s case=%s model=rigid_patch_joint_sheet capability=EXPERIMENTAL validation=GATED status=%s reason=%s frames=%u completedFrames=%u grid=%u cells=%u joints=%u dispatcherThreads=%u mass=%.9g contacts=%u crossContacts=%u centerY=%.9g minY=%.9g peakPrincipalStrain=%.9g finalPrincipalStrain=%.9g finalAreaRatio=%.9g maxPenetration=%.9g maxSelfPenetration=%.9g initialEnergy=%.9g peakEnergy=%.9g maxMomentumError=%.9g maxBallisticError=%.9g finalAwake=%u averageStepMs=%.9g p95StepMs=%.9g maxStepMs=%.9g averageObserverMs=%.9g nonFinite=%u fetchFailures=%u fatalErrors=%u\n",
		options.solver.solver==PxSolverType::eAVBD?"avbd":options.solver.solver==PxSolverType::ePGS?"pgs":"tgs",options.caseName,!std::strcmp(reason,"none")?"PASS":"FAIL",reason,options.frames,completed,options.grid,world.host.lattice.nodes.size(),world.host.lattice.joints.size(),options.solver.workers,world.host.lattice.totalMass,world.host.events.contacts,world.host.events.crossContacts,current.center.y,current.minY,peakStrain,current.principal,current.area,peakPenetration,peakSelf,initial.energy,peakEnergy,maxMomentum,maxBallistic,current.awake,timings.empty()?0:sum/timings.size(),timings.empty()?0:timings[(timings.size()-1)*95/100],timings.empty()?0:timings.back(),completed?observerMs/completed:0,nonFinite,fetchFailures,fatal);
    if(free || reaction)std::printf("[SHEET_ANGULAR_MOMENTUM] maxDrift=%.9g units=kg*m^2/s\n",maxAngularDrift);
	return !std::strcmp(reason,"none")?0:1;
}
}
