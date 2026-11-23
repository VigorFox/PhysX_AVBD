// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef SNIPPET_JOINT_SOFT_BODY_COMMON_H
#define SNIPPET_JOINT_SOFT_BODY_COMMON_H

#include "PxPhysicsAPI.h"
#include "SnippetFiniteJoint.h"
#include "SnippetVolumeMaterial.h"
#include "SnippetJointSoftBodyPvd.h"
#include "../snippetcommon/SnippetAllocationStats.h"
#include <atomic>
#include <cstdio>
#include <cstring>

namespace JointSoftBodySnippet
{
using namespace physx;

struct Settings
{
	PxU32 grid, workers, compressionHoldFrames;
	PxReal size, density, young, poisson, dampingRatio;
	PxSolverType::Enum solver;
    bool volumeMaterial;
	bool pvd;
	Settings() : grid(5), workers(4), compressionHoldFrames(60), size(2.0f), density(100.0f),
		young(1.2e5f), poisson(0.4f), dampingRatio(0.15f), solver(PxSolverType::eAVBD), volumeMaterial(false), pvd(false) {}
};

struct Errors : PxErrorCallback
{
	std::atomic<PxU32> fatal;
	Errors() : fatal(0) {}
	void reportError(PxErrorCode::Enum code, const char* message, const char*, int) PX_OVERRIDE
	{
		if(code & (PxErrorCode::eINVALID_PARAMETER | PxErrorCode::eINVALID_OPERATION |
			PxErrorCode::eOUT_OF_MEMORY | PxErrorCode::eINTERNAL_ERROR | PxErrorCode::eABORT))
		{
			++fatal;
			std::fprintf(stderr, "[JOINT_SOFT_ERROR] %s\n", message);
		}
	}
};

struct Events : PxSimulationEventCallback
{
	PxU32 contacts, crossContacts, crossMask;
	Events() : contacts(0),crossContacts(0),crossMask(0) {}
	void onConstraintBreak(PxConstraintInfo*, PxU32) PX_OVERRIDE {}
	void onWake(PxActor**, PxU32) PX_OVERRIDE {}
	void onSleep(PxActor**, PxU32) PX_OVERRIDE {}
	void onTrigger(PxTriggerPair*, PxU32) PX_OVERRIDE {}
	void onAdvance(const PxRigidBody*const*, const PxTransform*, const PxU32) PX_OVERRIDE {}
	void onContact(const PxContactPairHeader& header, const PxContactPair* pairs, PxU32 count) PX_OVERRIDE
	{
		PX_UNUSED(header);
		for(PxU32 i=0; i<count; ++i)
		{
			contacts += pairs[i].contactCount;
			if(pairs[i].flags & (PxContactPairFlag::eREMOVED_SHAPE_0|PxContactPairFlag::eREMOVED_SHAPE_1)) continue;
			const PxU32 ids[2]={pairs[i].shapes[0]->getSimulationFilterData().word0,pairs[i].shapes[1]->getSimulationFilterData().word0};
			if(ids[0] && ids[1] && ids[0]!=ids[1])
			{
				crossContacts+=pairs[i].contactCount;
				if(ids[0]<=5 && ids[1]<=5 && pairs[i].contactCount)
					crossMask|=1u<<((ids[0]-1)*5+ids[1]-1);
			}
		}
	}
};

// Adjacent finite cells overlap under compression. Non-neighboring material
// must still collide through NP. Zero grid words retain the original model's
// authored whole-instance exclusion; one-based words carry baked cell indices.
inline bool excludedCellPair(const PxFilterData& a,const PxFilterData& b)
{
	if(!a.word0 || a.word0!=b.word0) return false;
	if(!a.word1 || !b.word1) return true;
	return PxMax(a.word1,b.word1)-PxMin(a.word1,b.word1)<=1 &&
		PxMax(a.word2,b.word2)-PxMin(a.word2,b.word2)<=1 &&
		PxMax(a.word3,b.word3)-PxMin(a.word3,b.word3)<=1;
}
inline PxFilterFlags filter(PxFilterObjectAttributes a0, PxFilterData d0,
	PxFilterObjectAttributes a1, PxFilterData d1, PxPairFlags& flags, const void*, PxU32)
{
	if(PxFilterObjectIsTrigger(a0) || PxFilterObjectIsTrigger(a1))
	{
		flags=PxPairFlag::eTRIGGER_DEFAULT;
		return PxFilterFlag::eDEFAULT;
	}
	if(excludedCellPair(d0,d1)) return PxFilterFlag::eSUPPRESS;
	flags=PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_FOUND |
		PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
	return PxFilterFlag::eDEFAULT;
}

struct Lattice
{
	PxArray<PxRigidDynamic*> nodes;
	PxArray<FiniteJoint*> joints;
    PxArray<VolumeMaterial*> materials;
	PxArray<PxVec3> restLocal;
	PxU32 grid;
	PxReal cell, linearStiffness, shearStiffness, bendingStiffness, torsionStiffness, totalMass;
	Lattice() : grid(0), cell(0), linearStiffness(0), shearStiffness(0),
		bendingStiffness(0), torsionStiffness(0), totalMass(0) {}
	PxU32 index(PxU32 x, PxU32 y, PxU32 z) const { return x+grid*(y+grid*z); }
	void release()
	{
		for(PxU32 i=0; i<materials.size(); ++i) materials[i]->release();
        materials.reset();
		for(PxU32 i=0; i<joints.size(); ++i) joints[i]->release();
		for(PxU32 i=0; i<nodes.size(); ++i) nodes[i]->release();
		joints.reset(); nodes.reset(); restLocal.reset();
		totalMass=0;
	}
    bool buildMaterials(PxPhysics& physics,const Settings& settings,const PxArray<PxU32>& map)
    {
        if(!settings.volumeMaterial) return true;
        struct Cell { PxU32 ids[4];PxReal rest; };
        PxArray<Cell> cells;double volume=0;
        const PxU32 permutations[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
        // Physical center-lattice connectivity is authored here, before any
        // headless observer exists; only rest operators persist in constraints.
        for(PxU32 z=0;z+1<grid;++z) for(PxU32 y=0;y+1<grid;++y) for(PxU32 x=0;x+1<grid;++x)
            for(PxU32 t=0;t<6;++t)
            {
                Cell c;PxU32 v[3]={x,y,z};c.ids[0]=map[index(x,y,z)];
                ++v[permutations[t][0]];c.ids[1]=map[index(v[0],v[1],v[2])];
                ++v[permutations[t][1]];c.ids[2]=map[index(v[0],v[1],v[2])];c.ids[3]=map[index(x+1,y+1,z+1)];
                bool complete=true;for(PxU32 s=0;s<4;++s) complete&=c.ids[s]!=PX_MAX_U32;
                if(!complete) continue;
                PxVec3 p[4];for(PxU32 s=0;s<4;++s) p[s]=restLocal[c.ids[s]]+nodes[c.ids[s]]->getCMassLocalPose().p;
                c.rest=(p[1]-p[0]).dot((p[2]-p[0]).cross(p[3]-p[0]));
                if(!PxIsFinite(c.rest) || c.rest==0) return false;
                volume+=PxAbs(double(c.rest))/6;cells.pushBack(c);
            }
        if(volume<=0) return false;
        const double nonlinearBulk=double(settings.young)/(3*(1-2*double(settings.poisson)));
        if(!(nonlinearBulk>0)) return false;
        std::printf("[JOINT_VOLUME_MATERIAL] elements=%u volume=%.9g nonlinearBulk=%.9g addedLinearBulk=0\n",cells.size(),volume,nonlinearBulk);
        for(PxU32 t=0;t<cells.size();++t)
        {
            PxRigidActor* actors[4];for(PxU32 s=0;s<4;++s) actors[s]=nodes[cells[t].ids[s]];
            VolumeMaterial* material=VolumeMaterial::create(physics,actors,cells[t].rest,PxReal(nonlinearBulk*PxAbs(double(cells[t].rest))/6));
            if(!material) return false;materials.pushBack(material);
        }
        return true;
    }
	bool build(PxPhysics& physics, PxScene& scene, PxMaterial& material,
		const Settings& settings, const PxTransform& pose, PxU32 collisionIdentity)
	{
		if(!collisionIdentity) return false;
		grid=settings.grid; cell=settings.size/PxReal(grid);
		const PxReal mass=settings.density*cell*cell*cell;
		linearStiffness=settings.young*cell;
		const PxReal shear=settings.young/(2.0f*(1.0f+settings.poisson));
		shearStiffness=shear*cell;
		// Condense a square face's distributed spring into its centroid force
		// and section moments: EA/L, GA/L, EI/L, GIp/L, with L=cell.
		// This is a finite beam lattice, not an isotropic tetrahedral FEM.
		bendingStiffness=settings.young*cell*cell*cell/12.0f;
		torsionStiffness=shear*cell*cell*cell/6.0f;
		for(PxU32 z=0; z<grid; ++z)
			for(PxU32 y=0; y<grid; ++y)
				for(PxU32 x=0; x<grid; ++x)
				{
					const PxVec3 p=(PxVec3(PxReal(x),PxReal(y),PxReal(z))-
						PxVec3(0.5f*PxReal(grid-1)))*cell;
					PxRigidDynamic* body=PxCreateDynamic(physics, pose*PxTransform(p),
						PxBoxGeometry(PxVec3(0.5f*cell)), material, settings.density);
					if(!body) return false;
					body->setLinearDamping(0); body->setAngularDamping(0);
					PxShape* shape=NULL; body->getShapes(&shape,1);
					shape->setSimulationFilterData(PxFilterData(collisionIdentity,0,0,0));
					shape->setQueryFilterData(PxFilterData(collisionIdentity,0,0,0));
					shape->setContactOffset(0.02f*cell);
					shape->setRestOffset(0);
					scene.addActor(*body); nodes.pushBack(body); restLocal.pushBack(p);
					totalMass+=body->getMass();
				}
		const PxReal inertia=mass*cell*cell/6.0f;
		for(PxU32 z=0; z<grid; ++z)
			for(PxU32 y=0; y<grid; ++y)
				for(PxU32 x=0; x<grid; ++x)
					for(PxU32 axis=0; axis<3; ++axis)
					{
						PxU32 next[3]={x,y,z}; if(++next[axis]>=grid) continue;
						PxVec3 anchor(0); anchor[axis]=cell*0.5f;
						// Local X is the face normal, Y/Z are section tangents.
						const PxQuat frame=axis==0 ? PxQuat(PxIdentity) :
							(axis==1 ? PxQuat(PxHalfPi,PxVec3(0,0,1)) : PxQuat(-PxHalfPi,PxVec3(0,1,0)));
						FiniteJoint* joint=FiniteJoint::create(physics, *nodes[index(x,y,z)],
							PxTransform(anchor,frame), *nodes[index(next[0],next[1],next[2])], PxTransform(-anchor,frame));
						if(!joint) return false;
						joints.pushBack(joint);
						for(PxU32 drive=0; drive<3; ++drive)
						{
							const PxReal k=drive==0 ? linearStiffness : shearStiffness;
							const PxReal d=2.0f*settings.dampingRatio*PxSqrt(k*mass*0.5f);
							joint->setDrive(PxD6Drive::Enum(drive),PxD6JointDrive(k,d,PX_MAX_F32,false));
						}
						for(PxU32 drive=0; drive<3; ++drive)
						{
							const PxReal k=drive==0 ? torsionStiffness : bendingStiffness;
							const PxReal d=2.0f*settings.dampingRatio*PxSqrt(k*inertia*0.5f);
							joint->setDrive(PxD6Drive::Enum(PxD6Drive::eTWIST+drive),PxD6JointDrive(k,d,PX_MAX_F32,false));
						}
					}
        PxArray<PxU32> map(nodes.size());for(PxU32 i=0;i<map.size();++i) map[i]=i;
        return buildMaterials(physics,settings,map);
	}
};

struct World
{
	Snippets::AllocationStats allocator;
	Errors errors; Events events;
	PvdVisualization pvd;
	PxFoundation* foundation; PxPhysics* physics; PxScene* scene;
	PxDefaultCpuDispatcher* dispatcher;
	PxMaterial* material;
	PxRigidStatic* ground;
	PxRigidDynamic* platen; PxRigidDynamic* impactor;
	Lattice lattice;
	Settings settings;
	const char* caseName;
	PxVec3 impactHalf;
	bool extensions;
	World() : foundation(NULL),physics(NULL),scene(NULL),dispatcher(NULL),material(NULL),
		ground(NULL),platen(NULL),impactor(NULL),caseName(NULL),impactHalf(0.4f),extensions(false) {}
	~World() { release(); }
	void release()
	{
		lattice.release();
		PX_RELEASE(platen); PX_RELEASE(impactor); PX_RELEASE(ground);
		PX_RELEASE(scene); PX_RELEASE(dispatcher); PX_RELEASE(material);
		if(extensions) { PxCloseExtensions(); extensions=false; }
		PX_RELEASE(physics); pvd.release(); PX_RELEASE(foundation);
	}
	bool initScene(const Settings& config, const char* name, const PxVec3& gravity,
		bool groundEnabled, PxReal friction=0.5f)
	{
		settings=config; caseName=name;
		foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
		if(!foundation) return false;
		if(!pvd.init(*foundation,settings.pvd)) return false;
		physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,pvd.get());
		if(!physics || !PxInitExtensions(*physics,pvd.get())) return false;
		extensions=true;
		dispatcher=PxDefaultCpuDispatcherCreate(settings.workers);
		if(!dispatcher) return false;
		PxSceneDesc desc(physics->getTolerancesScale());
		desc.solverType=settings.solver; desc.cpuDispatcher=dispatcher;
		desc.gravity=gravity;
		desc.filterShader=filter; desc.simulationEventCallback=&events;
		scene=physics->createScene(desc); if(!scene) return false;
		pvd.configureScene(*scene);
		material=physics->createMaterial(friction,friction,0); if(!material) return false;
		if(groundEnabled)
		{
			ground=PxCreatePlane(*physics,PxPlane(0,1,0,0),*material);
			if(!ground) return false; scene->addActor(*ground);
		}
		return true;
	}
	bool init(const Settings& config, const char* name)
	{
		const bool free=std::strcmp(name,"free-flight")==0;
		const bool reaction=std::strcmp(name,"reaction")==0;
		const bool compression=std::strcmp(name,"compression-recovery")==0;
		if(!initScene(config,name,(free || reaction)?PxVec3(0):PxVec3(0,-9.81f,0),!free && !reaction)) return false;
		const PxReal centerY=std::strcmp(name,"drop")==0 ? 3.0f : free ? 3.0f : 1.0f;
		const PxQuat rotation=free ? PxQuat(0.37f,PxVec3(1,2,3).getNormalized()) : PxQuat(PxIdentity);
		if(!lattice.build(*physics,*scene,*material,settings,PxTransform(PxVec3(0,centerY,0),rotation),1)) return false;
		if(free)
			for(PxU32 i=0; i<lattice.nodes.size(); ++i) lattice.nodes[i]->setLinearVelocity(PxVec3(0.5f,-0.25f,0.2f));
		if(compression)
		{
			platen=PxCreateDynamic(*physics,PxTransform(PxVec3(0,2.35f,0)),PxBoxGeometry(1.2f,0.15f,1.2f),*material,100);
			if(!platen) return false;
			platen->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true); scene->addActor(*platen);
		}
		if(reaction || std::strcmp(name,"showcase")==0 || std::strcmp(name,"wake-recovery")==0)
		{
			impactor=PxCreateDynamic(*physics,PxTransform(PxVec3(0,3.1f,0)),PxBoxGeometry(impactHalf),*material,1);
			if(!impactor) return false;
			PxRigidBodyExt::setMassAndUpdateInertia(*impactor,80.0f);
			impactor->setLinearDamping(0); impactor->setAngularDamping(0);
			scene->addActor(*impactor);
			if(reaction)
			{
				impactor->setLinearVelocity(PxVec3(0,-2,0));
				for(PxU32 i=0; i<lattice.nodes.size(); ++i)
					lattice.nodes[i]->setLinearVelocity(PxVec3(0,160.0f/lattice.totalMass,0));
			}
		}
		return true;
	}
	bool step(PxU32 frame)
	{
		const PxU32 releaseFrame=120+settings.compressionHoldFrames;
		if(std::strcmp(caseName,"compression-load")==0 && frame>=60 && frame<releaseFrame)
		{
			const PxReal fraction=PxClamp((PxReal(frame)-60.0f)/60.0f,0.0f,1.0f);
			// Prescribed external stress, E * 0.1. No geometric correction:
			// each exposed top cell carries its authored area share of the load.
			const PxReal force=settings.young*0.1f*lattice.cell*lattice.cell*fraction;
			for(PxU32 z=0;z<lattice.grid;++z)
				for(PxU32 x=0;x<lattice.grid;++x)
					lattice.nodes[lattice.index(x,lattice.grid-1,z)]->addForce(PxVec3(0,-force,0));
		}
		if(std::strcmp(caseName,"wake-recovery")==0 && frame==240)
			for(PxU32 i=0;i<lattice.nodes.size();++i)
				lattice.nodes[i]->addForce(PxVec3(0,0.5f*lattice.nodes[i]->getMass(),0),PxForceMode::eIMPULSE);
		if(platen)
		{
			if(frame==releaseFrame) { PX_RELEASE(platen); }
			else
			{
				const PxReal fraction=PxClamp((PxReal(frame)-60.0f)/60.0f,0.0f,1.0f);
				platen->setKinematicTarget(PxTransform(PxVec3(0,2.35f-0.7f*fraction,0)));
			}
		}
		events.contacts=events.crossContacts=events.crossMask=0;
		scene->simulate(1.0f/60.0f);
		return scene->fetchResults(true);
	}
};
}
#endif
