// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "SnippetJointSurface.h"
#include "../snippetjointsoftbody/SnippetJointSoftBodyPreset.h"
#include <cstdlib>

namespace JointSurfaceSnippet
{
bool SheetWorld::init(const Options& config)
{
	options=config;
	const bool release=std::strcmp(options.caseName,"strain-release")==0;
	const bool free=release || std::strcmp(options.caseName,"free-flight")==0;
	const bool reaction=std::strcmp(options.caseName,"reaction")==0;
	if(!host.initScene(options.solver,options.caseName,(free || reaction)?PxVec3(0):PxVec3(0,-9.81f,0),!free && !reaction,.6f)) return false;
	if(std::strcmp(options.caseName,"static-box")==0)
	{
		obstacle=PxCreateStatic(*host.physics,PxTransform(PxVec3(0,1,0)),PxBoxGeometry(1.15f,1,1.15f),*host.material);
		if(!obstacle || !host.scene->addActor(*obstacle)) return false;
	}
	auto& lattice=host.lattice;
	const PxU32 n=options.grid;
	const PxReal cell=size()/PxReal(n),t=thickness(),cellMass=mass()/PxReal(n*n);
	const PxReal arealYoung=2.0e4f,young=arealYoung/t,shear=young/(2*(1+.3f));
	lattice.grid=n;lattice.cell=cell;
	// Rectangular section: local X joins centers, Y is the sheet normal,
	// Z spans the edge. Only author coefficients; the shared shader/backend
	// owns all strain, derivatives, damping, primal/dual and integration.
	const PxReal stiffness[6]={arealYoung,shear*t,shear*t,
		shear*t*t*t/3,arealYoung*cell*cell/12,young*t*t*t/12};
	const PxReal inertia[3]={cellMass*(cell*cell+t*t)/12,cellMass*cell*cell/6,cellMass*(cell*cell+t*t)/12};
	for(PxU32 z=0;z<n;++z) for(PxU32 x=0;x<n;++x)
	{
		const PxVec3 p((PxReal(x)+.5f)*cell-size()*.5f,4.2f,(PxReal(z)+.5f)*cell-size()*.5f);
		PxRigidDynamic* body=PxCreateDynamic(*host.physics,PxTransform(p),PxBoxGeometry(cell*.5f,t*.5f,cell*.5f),*host.material,cellMass/(cell*cell*t));
		if(!body) return false;
		lattice.nodes.pushBack(body);lattice.restLocal.pushBack(p);lattice.totalMass+=body->getMass();
		body->setLinearDamping(free || reaction?0:.2f);body->setAngularDamping(0);
		body->setSleepThreshold(5.0e-4f);
		if(free) body->setLinearVelocity(PxVec3(.5f,-.25f,.2f));
		PxShape* shape=NULL;body->getShapes(&shape,1);
		const PxFilterData data(1,x+1,1,z+1);
		shape->setSimulationFilterData(data);shape->setQueryFilterData(data);
		shape->setContactOffset(.04f);shape->setRestOffset(0);
		if(!host.scene->addActor(*body)) return false;
	}
	for(PxU32 z=0;z<n;++z) for(PxU32 x=0;x<n;++x) for(PxU32 axis=0;axis<2;++axis)
	{
		const PxU32 nx=x+(axis==0),nz=z+(axis==1);if(nx>=n || nz>=n) continue;
		const PxVec3 anchor=axis==0?PxVec3(cell*.5f,0,0):PxVec3(0,0,cell*.5f);
		const PxQuat frame=axis==0?PxQuat(PxIdentity):PxQuat(-PxHalfPi,PxVec3(0,1,0));
		auto* joint=JointSoftBodySnippet::FiniteJoint::create(*host.physics,*lattice.nodes[x+n*z],PxTransform(anchor,frame),*lattice.nodes[nx+n*nz],PxTransform(-anchor,frame));
		if(!joint) return false;lattice.joints.pushBack(joint);
		for(PxU32 d=0;d<6;++d)
		{
			const PxReal reducedMass=.5f*(d<3?cellMass:inertia[d-3]);
			joint->setDrive(PxD6Drive::Enum(d),PxD6JointDrive(stiffness[d],2*.08f*PxSqrt(stiffness[d]*reducedMass),PX_MAX_F32,false));
		}
	}
	if(reaction)
	{
		host.impactor=PxCreateDynamic(*host.physics,PxTransform(PxVec3(.1f,5.0f,.1f)),PxBoxGeometry(.3f,.3f,.3f),*host.material,20/(.6f*.6f*.6f));
		if(!host.impactor) return false;
		host.impactor->setLinearDamping(0);host.impactor->setAngularDamping(0);host.impactor->setLinearVelocity(PxVec3(0,-2,0));
		PxShape* shape=NULL;host.impactor->getShapes(&shape,1);shape->setSimulationFilterData(PxFilterData(2,0,0,0));
		shape->setRestOffset(0);shape->setContactOffset(.04f);
		if(!host.scene->addActor(*host.impactor)) return false;
	}
	vertices.resize((n+1)*(n+1));
	for(PxU32 z=0;z<n;++z) for(PxU32 x=0;x<n;++x)
	{
		const PxU32 a=x+(n+1)*z,b=a+1,c=a+n+1,d=c+1;
		if((x+z)&1) { triangles.pushBack(a);triangles.pushBack(c);triangles.pushBack(d);triangles.pushBack(a);triangles.pushBack(d);triangles.pushBack(b); }
		else { triangles.pushBack(a);triangles.pushBack(c);triangles.pushBack(b);triangles.pushBack(c);triangles.pushBack(d);triangles.pushBack(b); }
	}
	updateSkin();restVertices=vertices;
	if(release) for(auto* node:lattice.nodes)
	{
		PxTransform pose=node->getGlobalPose();
		pose.p.x*=1.05f;pose.p.z*=.98f;
		pose.p.y+=.15f*PxSin(1.7f*(pose.p.x+.4f))*PxCos(1.1f*(pose.p.z-.3f));
		node->setGlobalPose(pose);
	}
	std::printf("[JOINT_SURFACE_ASSET] grid=%u cells=%u joints=%u mass=%.9g width=%.9g thickness=%.9g axial=%.9g shear=%.9g twist=%.9g inPlaneBend=%.9g outOfPlaneBend=%.9g\n",
		n,lattice.nodes.size(),lattice.joints.size(),lattice.totalMass,size(),t,stiffness[0],stiffness[1],stiffness[3],stiffness[4],stiffness[5]);
	return host.errors.fatal.load()==0;
}
bool SheetWorld::step()
{
	sendPvdSkin();
	host.scene->simulate(1.0f/60);
	return host.scene->fetchResults(true) && host.errors.fatal.load()==0;
}
void SheetWorld::sendPvdSkin()
{
	if(!host.pvd.connected()) return;
	updateSkin();
	host.pvd.wireMesh(*host.scene,vertices,triangles,0xfff25729);
}
void SheetWorld::updateSkin()
{
	const PxU32 n=options.grid;const PxReal cell=host.lattice.cell;
	for(PxU32 z=0;z<=n;++z) for(PxU32 x=0;x<=n;++x)
	{
		PxVec3 p(0);PxU32 count=0;
		for(PxU32 cz=z?z-1:0;cz<=z && cz<n;++cz) for(PxU32 cx=x?x-1:0;cx<=x && cx<n;++cx)
		{
			p+=host.lattice.nodes[cx+n*cz]->getGlobalPose().transform(PxVec3((PxReal(x-cx)-.5f)*cell,0,(PxReal(z-cz)-.5f)*cell));++count;
		}
		vertices[x+(n+1)*z]=p/PxReal(count);
	}
}
namespace
{
bool number(const char* text,PxU32& value,PxU32 minimum,PxU32 maximum)
{
	if(!*text || *text=='-') return false;char* end=NULL;const unsigned long n=std::strtoul(text,&end,10);
	if(*end || n<minimum || n>maximum) return false;value=PxU32(n);return true;
}
}
int runApplication(int argc,const char*const* argv)
{
	Options options;options.headless=std::getenv("PHYSX_SNIPPET_HEADLESS")!=NULL;
	for(int i=1;i<argc;++i)
	{
		const char* a=argv[i];bool valid=true;
		if(!std::strcmp(a,"--headless")) options.headless=true;
		else if(!std::strcmp(a,"--pvd")) options.solver.pvd=true;
		else if(!std::strncmp(a,"--case=",7)) { options.caseName=a+7;options.headless=true; }
		else if(!std::strncmp(a,"--grid=",7)) valid=number(a+7,options.grid,4,32);
		else if(!std::strncmp(a,"--frames=",9)) valid=number(a+9,options.frames,1,10000);
		else if(!std::strncmp(a,"--dispatcher-threads=",21)) valid=number(a+21,options.solver.workers,0,16);
		else if(!std::strcmp(a,"--solver=avbd")) options.solver.solver=PxSolverType::eAVBD;
		else if(!std::strcmp(a,"--solver=pgs")) options.solver.solver=PxSolverType::ePGS;
		else if(!std::strcmp(a,"--solver=tgs")) options.solver.solver=PxSolverType::eTGS;
		else valid=false;
		if(!valid) { std::fprintf(stderr,"Invalid joint surface option: %s.\n",a);return 2; }
	}
	if(std::strcmp(options.caseName,"static-box") && std::strcmp(options.caseName,"ground") && std::strcmp(options.caseName,"free-flight") &&
		std::strcmp(options.caseName,"strain-release") && std::strcmp(options.caseName,"reaction"))
	{
		std::fprintf(stderr,"Unsupported joint surface case: %s.\n",options.caseName);return 2;
	}
	const bool avbd=options.solver.solver==PxSolverType::eAVBD;
	if(avbd && !JointSoftBodySnippet::configureNativeJointStep()) return 2;
	std::printf("[AVBD_SURFACE_ROUTE] model=joint_sheet commonStep=%s case=%s grid=%u workers=%u\n",avbd?std::getenv("PHYSX_AVBD_COMMON_STEP"):"physx",options.caseName,options.grid,options.solver.workers);std::fflush(stdout);
	if(options.headless) return runHeadless(options);
#ifdef RENDER_SNIPPET
	options.solver.pvd=true;
	return renderLoop(options);
#else
	return runHeadless(options);
#endif
}
}
