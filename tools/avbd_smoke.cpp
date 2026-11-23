// Headless AVBD smoke: box-on-plane, D6 hang, artic pendulum/slider, unloaded scissor 10s.
// Compile against the checked CPU-only PhysX_AVBD_3 tree.

#include <cstdio>
#include "PxPhysicsAPI.h"
#include "extensions/PxDeformableVolumeExt.h"
#include "cooking/PxCooking.h"
#include "cooking/PxTetrahedronMeshDesc.h"
#include "avbd_host_access.h"

using namespace physx;

class SmokeErrorCallback : public PxErrorCallback
{
public:
	virtual void reportError(PxErrorCode::Enum code, const char* message,
		const char* file, int line) PX_OVERRIDE
	{
		std::printf("PhysX [%d] %s (%s:%d)\n", int(code), message, file, line);
		std::fflush(stdout);
	}
};

static bool finitePose(const PxTransform& t)
{
	return t.p.isFinite() && t.q.isFinite();
}

static PxFilterFlags scissorFilter(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
	PxFilterObjectAttributes attributes1, PxFilterData filterData1,
	PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(constantBlock);
	PX_UNUSED(constantBlockSize);
	if(filterData0.word2 != 0 && filterData0.word2 == filterData1.word2)
		return PxFilterFlag::eKILL;
	pairFlags |= PxPairFlag::eCONTACT_DEFAULT;
	return PxFilterFlag::eDEFAULT;
}

static PxFilterFlags detectOnlyFilter(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
	PxFilterObjectAttributes attributes1, PxFilterData filterData1,
	PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(filterData0);
	PX_UNUSED(filterData1);
	PX_UNUSED(constantBlock);
	PX_UNUSED(constantBlockSize);
	pairFlags = PxPairFlag::eDETECT_DISCRETE_CONTACT | PxPairFlag::eNOTIFY_TOUCH_FOUND;
	return PxFilterFlag::eDEFAULT;
}

static PxFilterFlags modifyContactFilter(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
	PxFilterObjectAttributes attributes1, PxFilterData filterData1,
	PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(filterData0);
	PX_UNUSED(filterData1);
	PX_UNUSED(constantBlock);
	PX_UNUSED(constantBlockSize);
	pairFlags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eMODIFY_CONTACTS;
	return PxFilterFlag::eDEFAULT;
}

static PxFilterFlags ogcDcdFilter(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
	PxFilterObjectAttributes attributes1, PxFilterData filterData1,
	PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(filterData0);
	PX_UNUSED(filterData1);
	PX_UNUSED(constantBlock);
	PX_UNUSED(constantBlockSize);
	pairFlags = PxPairFlag::eSOLVE_CONTACT | PxPairFlag::eDETECT_DISCRETE_CONTACT;
	return PxFilterFlag::eDEFAULT;
}

enum SmokeModifyMode
{
	eSMOKE_MODIFY_IGNORE = 0,
	eSMOKE_MODIFY_TARGET_VEL,
	eSMOKE_MODIFY_MASS_SCALE,
	eSMOKE_MODIFY_FINITE_ONE
};

class SmokeModifyCallback : public PxContactModifyCallback
{
public:
	SmokeModifyMode mode;
	SmokeModifyCallback() : mode(eSMOKE_MODIFY_IGNORE) {}
	virtual void onContactModify(PxContactModifyPair* const pairs, PxU32 count) PX_OVERRIDE
	{
		for(PxU32 i = 0; i < count; ++i)
		{
			PxContactSet& contacts = pairs[i].contacts;
			for(PxU32 k = 0; k < contacts.size(); ++k)
			{
				if(mode == eSMOKE_MODIFY_IGNORE)
					contacts.ignore(k);
				else if(mode == eSMOKE_MODIFY_TARGET_VEL)
					contacts.setTargetVelocity(k, PxVec3(0.0f, 80.0f, 0.0f));
				else if(mode == eSMOKE_MODIFY_FINITE_ONE)
					contacts.setMaxImpulse(k, 1.0f);
			}
			if(mode == eSMOKE_MODIFY_MASS_SCALE)
			{
				contacts.setInvMassScale0(0.0f);
				contacts.setInvMassScale1(0.0f);
				contacts.setInvInertiaScale0(0.0f);
				contacts.setInvInertiaScale1(0.0f);
			}
		}
	}
};

static PxScene* smokeAvbdScene(PxPhysics* physics, PxDefaultCpuDispatcher* dispatcher,
	PxSimulationFilterShader filter, PxContactModifyCallback* modify, bool ccd)
{
	PxSceneDesc desc(physics->getTolerancesScale());
	desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	desc.cpuDispatcher = dispatcher;
	desc.filterShader = filter;
	desc.solverType = PxSolverType::eAVBD;
	if(modify)
		desc.contactModifyCallback = modify;
	if(ccd)
		desc.flags |= PxSceneFlag::eENABLE_CCD;
	return physics->createScene(desc);
}

static PxRigidDynamic* smokeRigidProbeBox(PxPhysics* physics, PxMaterial* material,
	const PxVec3& position, const PxVec3& velocity,
	const PxVec3& halfExtent = PxVec3(0.5f))
{
	PxRigidDynamic* body = physics->createRigidDynamic(PxTransform(position));
	PxShape* shape = physics->createShape(PxBoxGeometry(halfExtent), *material, true);
	if(!body || !shape)
	{
		if(shape) shape->release();
		if(body) body->release();
		return NULL;
	}
	body->attachShape(*shape);
	shape->release();
	body->setMass(1.0f);
	body->setMassSpaceInertiaTensor(PxVec3(
		(halfExtent.y * halfExtent.y + halfExtent.z * halfExtent.z) / 3.0f,
		(halfExtent.x * halfExtent.x + halfExtent.z * halfExtent.z) / 3.0f,
		(halfExtent.x * halfExtent.x + halfExtent.y * halfExtent.y) / 3.0f));
	body->setLinearVelocity(velocity);
	body->setLinearDamping(0.0f);
	body->setAngularDamping(0.0f);
	body->setSleepThreshold(0.0f);
	body->setSolverIterationCounts(8, 2);
	return body;
}

static PxReal smokeQuatStepAngle(const PxQuat& previous, const PxQuat& current)
{
	const PxReal cosineHalfAngle = PxClamp(PxAbs(previous.dot(current)), 0.0f, 1.0f);
	return 2.0f * PxAcos(cosineHalfAngle);
}

static int smokeAbort(PxScene* scene, PxDefaultCpuDispatcher* dispatcher, PxPhysics* physics,
	PxFoundation* foundation)
{
	if(scene)
		scene->release();
	dispatcher->release();
	PxCloseExtensions();
	physics->release();
	foundation->release();
	return 1;
}

static PxArticulationReducedCoordinate* createScissorLift(PxPhysics* physics, PxScene* scene,
	PxMaterial* material, PxArticulationJointReducedCoordinate** driveOut,
	PxArticulationLink** baseOut, PxArticulationLink** topOut)
{
	PxArticulationReducedCoordinate* art = physics->createArticulationReducedCoordinate();
	const PxReal runnerLength = 2.0f;
	const PxReal placementDistance = 1.8f;
	const PxReal angle = PxAcos(placementDistance / runnerLength);
	const PxReal sinAng = PxSin(angle);
	const PxQuat leftRot(-angle, PxVec3(1.0f, 0.0f, 0.0f));
	const PxQuat rightRot(angle, PxVec3(1.0f, 0.0f, 0.0f));

	PxArticulationLink* base = art->createLink(NULL, PxTransform(PxVec3(0.0f, 0.25f, 0.0f)));
	PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.5f, 0.25f, 1.5f), *material);
	PxRigidBodyExt::updateMassAndInertia(*base, 3.0f);
	art->setSolverIterationCounts(32);

	PxArticulationLink* leftRoot = art->createLink(base, PxTransform(PxVec3(0.0f, 0.55f, -0.9f)));
	PxRigidActorExt::createExclusiveShape(*leftRoot, PxBoxGeometry(0.5f, 0.05f, 0.05f), *material);
	PxRigidBodyExt::updateMassAndInertia(*leftRoot, 1.0f);
	PxArticulationLink* rightRoot = art->createLink(base, PxTransform(PxVec3(0.0f, 0.55f, 0.9f)));
	PxRigidActorExt::createExclusiveShape(*rightRoot, PxBoxGeometry(0.5f, 0.05f, 0.05f), *material);
	PxRigidBodyExt::updateMassAndInertia(*rightRoot, 1.0f);

	PxArticulationJointReducedCoordinate* joint = leftRoot->getInboundJoint();
	joint->setJointType(PxArticulationJointType::eFIX);
	joint->setParentPose(PxTransform(PxVec3(0.0f, 0.25f, -0.9f)));
	joint->setChildPose(PxTransform(PxVec3(0.0f, -0.05f, 0.0f)));

	PxArticulationJointReducedCoordinate* drive = rightRoot->getInboundJoint();
	drive->setJointType(PxArticulationJointType::ePRISMATIC);
	drive->setMotion(PxArticulationAxis::eZ, PxArticulationMotion::eLIMITED);
	drive->setLimitParams(PxArticulationAxis::eZ, PxArticulationLimit(-1.4f, 0.2f));
	drive->setDriveParams(PxArticulationAxis::eZ, PxArticulationDrive(100000.0f, 0.0f, PX_MAX_F32));
	drive->setParentPose(PxTransform(PxVec3(0.0f, 0.25f, 0.9f)));
	drive->setChildPose(PxTransform(PxVec3(0.0f, -0.05f, 0.0f)));

	const PxU32 linkHeight = 3;
	PxArticulationLink* currLeft = leftRoot;
	PxArticulationLink* currRight = rightRoot;
	PxQuat rightParentRot(PxIdentity);
	PxQuat leftParentRot(PxIdentity);
	for(PxU32 i = 0; i < linkHeight; ++i)
	{
		const PxVec3 pos(0.5f, 0.55f + 0.1f * (1 + i), 0.0f);
		PxArticulationLink* leftLink = art->createLink(currLeft,
			PxTransform(pos + PxVec3(0.0f, sinAng * (2 * i + 1), 0.0f), leftRot));
		PxRigidActorExt::createExclusiveShape(*leftLink, PxBoxGeometry(0.05f, 0.05f, 1.0f), *material);
		PxRigidBodyExt::updateMassAndInertia(*leftLink, 1.0f);
		const PxVec3 leftAnchor = pos + PxVec3(0.0f, sinAng * (2 * i), -0.9f);
		joint = leftLink->getInboundJoint();
		joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchor), leftParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.0f, 0.0f, -1.0f), rightRot));
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		leftParentRot = leftRot;
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));

		PxArticulationLink* rightLink = art->createLink(currRight,
			PxTransform(pos + PxVec3(0.0f, sinAng * (2 * i + 1), 0.0f), rightRot));
		PxRigidActorExt::createExclusiveShape(*rightLink, PxBoxGeometry(0.05f, 0.05f, 1.0f), *material);
		PxRigidBodyExt::updateMassAndInertia(*rightLink, 1.0f);
		const PxVec3 rightAnchor = pos + PxVec3(0.0f, sinAng * (2 * i), 0.9f);
		joint = rightLink->getInboundJoint();
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchor), rightParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.0f, 0.0f, 1.0f), leftRot));
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));
		rightParentRot = rightRot;

		PxD6Joint* d6 = PxD6JointCreate(*physics, leftLink, PxTransform(PxIdentity),
			rightLink, PxTransform(PxIdentity));
		d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
		d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
		d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
		currLeft = rightLink;
		currRight = leftLink;
	}

	PxArticulationLink* leftTop = art->createLink(currLeft,
		currLeft->getGlobalPose().transform(PxTransform(PxVec3(-0.5f, 0.0f, -1.0f), leftParentRot)));
	PxRigidActorExt::createExclusiveShape(*leftTop, PxBoxGeometry(0.5f, 0.05f, 0.05f), *material);
	PxRigidBodyExt::updateMassAndInertia(*leftTop, 1.0f);
	PxArticulationLink* rightTop = art->createLink(currRight,
		currRight->getGlobalPose().transform(PxTransform(PxVec3(-0.5f, 0.0f, 1.0f), rightParentRot)));
	PxRigidActorExt::createExclusiveShape(*rightTop, PxCapsuleGeometry(0.05f, 0.8f), *material);
	PxRigidBodyExt::updateMassAndInertia(*rightTop, 1.0f);

	joint = leftTop->getInboundJoint();
	joint->setParentPose(PxTransform(PxVec3(0.0f, 0.0f, -1.0f), currLeft->getGlobalPose().q.getConjugate()));
	joint->setChildPose(PxTransform(PxVec3(0.5f, 0.0f, 0.0f), leftTop->getGlobalPose().q.getConjugate()));
	joint->setJointType(PxArticulationJointType::eREVOLUTE);
	joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
	joint = rightTop->getInboundJoint();
	joint->setParentPose(PxTransform(PxVec3(0.0f, 0.0f, 1.0f), currRight->getGlobalPose().q.getConjugate()));
	joint->setChildPose(PxTransform(PxVec3(0.5f, 0.0f, 0.0f), rightTop->getGlobalPose().q.getConjugate()));
	joint->setJointType(PxArticulationJointType::eREVOLUTE);
	joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);

	currLeft = leftRoot;
	currRight = rightRoot;
	rightParentRot = PxQuat(PxIdentity);
	leftParentRot = PxQuat(PxIdentity);
	for(PxU32 i = 0; i < linkHeight; ++i)
	{
		const PxVec3 pos(-0.5f, 0.55f + 0.1f * (1 + i), 0.0f);
		PxArticulationLink* leftLink = art->createLink(currLeft,
			PxTransform(pos + PxVec3(0.0f, sinAng * (2 * i + 1), 0.0f), leftRot));
		PxRigidActorExt::createExclusiveShape(*leftLink, PxBoxGeometry(0.05f, 0.05f, 1.0f), *material);
		PxRigidBodyExt::updateMassAndInertia(*leftLink, 1.0f);
		const PxVec3 leftAnchor = pos + PxVec3(0.0f, sinAng * (2 * i), -0.9f);
		joint = leftLink->getInboundJoint();
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchor), leftParentRot));
		joint->setChildPose(PxTransform(PxVec3(0.0f, 0.0f, -1.0f), rightRot));
		leftParentRot = leftRot;
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));

		PxArticulationLink* rightLink = art->createLink(currRight,
			PxTransform(pos + PxVec3(0.0f, sinAng * (2 * i + 1), 0.0f), rightRot));
		PxRigidActorExt::createExclusiveShape(*rightLink, PxBoxGeometry(0.05f, 0.05f, 1.0f), *material);
		PxRigidBodyExt::updateMassAndInertia(*rightLink, 1.0f);
		const PxVec3 rightAnchor = pos + PxVec3(0.0f, sinAng * (2 * i), 0.9f);
		joint = rightLink->getInboundJoint();
		joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchor), rightParentRot));
		joint->setJointType(PxArticulationJointType::eREVOLUTE);
		joint->setChildPose(PxTransform(PxVec3(0.0f, 0.0f, 1.0f), leftRot));
		joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
		joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));
		rightParentRot = rightRot;

		PxD6Joint* d6 = PxD6JointCreate(*physics, leftLink, PxTransform(PxIdentity),
			rightLink, PxTransform(PxIdentity));
		d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
		d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
		d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
		currLeft = rightLink;
		currRight = leftLink;
	}

	PxD6Joint* d6 = PxD6JointCreate(*physics, currLeft, PxTransform(PxVec3(0.0f, 0.0f, -1.0f)),
		leftTop, PxTransform(PxVec3(-0.5f, 0.0f, 0.0f)));
	d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
	d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
	d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
	d6 = PxD6JointCreate(*physics, currRight, PxTransform(PxVec3(0.0f, 0.0f, 1.0f)),
		rightTop, PxTransform(PxVec3(-0.5f, 0.0f, 0.0f)));
	d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
	d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
	d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

	const PxTransform topPose(PxVec3(0.0f, leftTop->getGlobalPose().p.y + 0.15f, 0.0f));
	PxArticulationLink* top = art->createLink(leftTop, topPose);
	PxRigidActorExt::createExclusiveShape(*top, PxBoxGeometry(0.5f, 0.1f, 1.5f), *material);
	PxRigidBodyExt::updateMassAndInertia(*top, 1.0f);
	joint = top->getInboundJoint();
	joint->setJointType(PxArticulationJointType::eFIX);
	joint->setParentPose(PxTransform(PxVec3(0.0f, 0.0f, 0.0f)));
	joint->setChildPose(PxTransform(PxVec3(0.0f, -0.15f, -0.9f)));

	scene->addArticulation(*art);
	for(PxU32 i = 0; i < art->getNbLinks(); ++i)
	{
		PxArticulationLink* link = NULL;
		art->getLinks(&link, 1, i);
		link->setLinearDamping(0.2f);
		link->setAngularDamping(0.2f);
		link->setMaxAngularVelocity(20.0f);
		link->setMaxLinearVelocity(100.0f);
		for(PxU32 b = 0; b < link->getNbShapes(); ++b)
		{
			PxShape* shape = NULL;
			link->getShapes(&shape, 1, b);
			shape->setSimulationFilterData(PxFilterData(0, 0, 1, 0));
		}
	}

	*driveOut = drive;
	*baseOut = base;
	*topOut = top;
	return art;
}

static void hostAllocFree(void* p)
{
	if(p)
		PxGetFoundation().getAllocatorCallback().deallocate(p);
}

static PxDeformableVolumeMesh* cookCubeVolumeMesh(PxPhysics& physics, const PxVec3& center, PxReal size)
{
	PxArray<PxVec3> verts;
	verts.reserve(8);
	const PxReal h = size * 0.5f;
	for(PxU32 i = 0; i < 8; ++i)
	{
		verts.pushBack(center + PxVec3(
			(i & 1) ? h : -h,
			(i & 2) ? h : -h,
			(i & 4) ? h : -h));
	}
	PxArray<PxU32> tets;
	const PxU32 tet5[5][4] = {
		{0, 1, 2, 4},
		{1, 3, 2, 7},
		{1, 4, 5, 7},
		{2, 4, 6, 7},
		{1, 2, 4, 7}
	};
	for(PxU32 i = 0; i < 5; ++i)
	{
		tets.pushBack(tet5[i][0]);
		tets.pushBack(tet5[i][1]);
		tets.pushBack(tet5[i][2]);
		tets.pushBack(tet5[i][3]);
	}
	PxTetrahedronMeshDesc meshDesc(verts, tets);
	PxDeformableVolumeSimulationDataDesc simDesc;
	PxCookingParams params(physics.getTolerancesScale());
	return PxCreateDeformableVolumeMesh(params, meshDesc, meshDesc, simDesc, physics.getPhysicsInsertionCallback());
}

static bool initVolumeHostBuffers(PxDeformableVolume& volume, PxReal density)
{
	PxVec4* simPos = NULL;
	PxVec4* simVel = NULL;
	PxVec4* collPos = NULL;
	PxVec4* restPos = NULL;
	PxDeformableVolumeExt::allocateAndInitializeHostMirror(volume, NULL, simPos, simVel, collPos, restPos);
	if(!simPos || !simVel)
		return false;
	PxDeformableVolumeExt::updateMass(volume, density, 50.0f, simPos);
	PxDeformableVolumeExt::copyToDevice(volume, PxDeformableVolumeDataFlag::eALL, simPos, simVel, collPos, restPos);
	hostAllocFree(simPos);
	hostAllocFree(simVel);
	hostAllocFree(collPos);
	hostAllocFree(restPos);
	return true;
}

static void releaseIslandVolume(PxDeformableVolume* vol, PxDeformableVolumeMesh* mesh,
	PxDeformableVolumeMaterial* mat)
{
	PxShape* shape = vol ? vol->getShape() : NULL;
	if(vol)
	{
		if(PxScene* scene = vol->getScene())
			scene->removeActor(*vol);
		vol->release();
	}
	if(shape)
		shape->release();
	if(mesh)
		mesh->release();
	if(mat)
		mat->release();
}

static PxReal tetMeshAbsVolume(const PxTetrahedronMesh& mesh, const PxVec4* pos)
{
	const PxU32 nbTets = mesh.getNbTetrahedrons();
	const bool idx16 = mesh.getTetrahedronMeshFlags() & PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* i16 = idx16 ? reinterpret_cast<const PxU16*>(mesh.getTetrahedrons()) : NULL;
	const PxU32* i32 = idx16 ? NULL : reinterpret_cast<const PxU32*>(mesh.getTetrahedrons());
	PxReal vol = 0.0f;
	for(PxU32 t = 0; t < nbTets; ++t)
	{
		PxU32 i0, i1, i2, i3;
		if(idx16)
		{
			i0 = i16[t * 4 + 0]; i1 = i16[t * 4 + 1];
			i2 = i16[t * 4 + 2]; i3 = i16[t * 4 + 3];
		}
		else
		{
			i0 = i32[t * 4 + 0]; i1 = i32[t * 4 + 1];
			i2 = i32[t * 4 + 2]; i3 = i32[t * 4 + 3];
		}
		const PxVec3 e1 = pos[i1].getXYZ() - pos[i0].getXYZ();
		const PxVec3 e2 = pos[i2].getXYZ() - pos[i0].getXYZ();
		const PxVec3 e3 = pos[i3].getXYZ() - pos[i0].getXYZ();
		vol += PxAbs(e1.dot(e2.cross(e3))) / 6.0f;
	}
	return vol;
}

int main()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	std::printf("avbd_smoke start\n");
	PxDefaultAllocator allocator;
	SmokeErrorCallback error;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, error);
	if(!foundation)
	{
		std::printf("FAIL create foundation\n");
		return 1;
	}

	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation,
		PxTolerancesScale(), false, NULL);
	if(!physics)
	{
		std::printf("FAIL create physics\n");
		return 1;
	}
	PxInitExtensions(*physics, NULL);

	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(0);
	PxSceneDesc desc(physics->getTolerancesScale());
	desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	desc.cpuDispatcher = dispatcher;
	desc.filterShader = PxDefaultSimulationFilterShader;
	desc.solverType = PxSolverType::eAVBD;
	PxScene* scene = physics->createScene(desc);
	if(!scene)
	{
		std::printf("FAIL create AVBD scene\n");
		return 1;
	}

	PxU32 kDevice = PX_MAX_U32;
	bool kLdlt6 = false;
	bool kLdlt3 = false;
	if(!avbdHostKernelTableInfo(scene, &kDevice, &kLdlt6, &kLdlt3) || kDevice != 0 ||
		!kLdlt6 || !kLdlt3 || !avbdHostKernelLdltSelfCheck())
	{
		std::printf("FAIL kernel table device=%u ldlt6=%d ldlt3=%d\n",
			unsigned(kDevice), int(kLdlt6), int(kLdlt3));
		scene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	std::printf("PASS kernel table device=SCALAR LDLT6 LDLT3\n");

	PxMaterial* mat = physics->createMaterial(0.5f, 0.5f, 0.0f);
	PxRigidStatic* ground = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *mat);
	scene->addActor(*ground);

	PxRigidDynamic* box = PxCreateDynamic(*physics, PxTransform(PxVec3(0.0f, 4.0f, 0.0f)),
		PxBoxGeometry(0.5f, 0.5f, 0.5f), *mat, 1.0f);
	scene->addActor(*box);

	PxRigidDynamic* hang = PxCreateDynamic(*physics, PxTransform(PxVec3(4.0f, 3.0f, 0.0f)),
		PxBoxGeometry(0.4f, 0.4f, 0.4f), *mat, 1.0f);
	scene->addActor(*hang);
	PxD6Joint* joint = PxD6JointCreate(*physics,
		NULL, PxTransform(PxVec3(4.0f, 6.0f, 0.0f)),
		hang, PxTransform(PxVec3(0.0f, 3.0f, 0.0f)));
	joint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
	joint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
	joint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
	joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
	joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
	joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

	PxArticulationReducedCoordinate* pendulum = physics->createArticulationReducedCoordinate();
	pendulum->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
	PxArticulationLink* pRoot = pendulum->createLink(NULL, PxTransform(PxVec3(-4.0f, 5.0f, 0.0f)));
	PxRigidActorExt::createExclusiveShape(*pRoot, PxBoxGeometry(0.1f, 0.1f, 0.1f), *mat);
	PxRigidBodyExt::updateMassAndInertia(*pRoot, 1.0f);
	PxArticulationLink* pChild = pendulum->createLink(pRoot, PxTransform(PxVec3(-4.0f, 3.5f, 0.0f)));
	PxRigidActorExt::createExclusiveShape(*pChild, PxBoxGeometry(0.2f, 0.2f, 0.2f), *mat);
	PxRigidBodyExt::updateMassAndInertia(*pChild, 1.0f);
	PxArticulationJointReducedCoordinate* pJoint = pChild->getInboundJoint();
	pJoint->setJointType(PxArticulationJointType::eREVOLUTE);
	pJoint->setParentPose(PxTransform(PxVec3(0.0f, -0.1f, 0.0f)));
	pJoint->setChildPose(PxTransform(PxVec3(0.0f, 1.4f, 0.0f)));
	pJoint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
	scene->addArticulation(*pendulum);

	PxArticulationReducedCoordinate* slider = physics->createArticulationReducedCoordinate();
	slider->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
	PxArticulationLink* sRoot = slider->createLink(NULL, PxTransform(PxVec3(8.0f, 1.5f, 0.0f)));
	PxRigidActorExt::createExclusiveShape(*sRoot, PxBoxGeometry(0.1f, 0.1f, 0.1f), *mat);
	PxRigidBodyExt::updateMassAndInertia(*sRoot, 1.0f);
	PxArticulationLink* sChild = slider->createLink(sRoot, PxTransform(PxVec3(8.0f, 1.5f, 0.0f)));
	PxRigidActorExt::createExclusiveShape(*sChild, PxBoxGeometry(0.2f, 0.2f, 0.2f), *mat);
	PxRigidBodyExt::updateMassAndInertia(*sChild, 1.0f);
	PxArticulationJointReducedCoordinate* sJoint = sChild->getInboundJoint();
	sJoint->setJointType(PxArticulationJointType::ePRISMATIC);
	sJoint->setParentPose(PxTransform(PxIdentity));
	sJoint->setChildPose(PxTransform(PxIdentity));
	sJoint->setMotion(PxArticulationAxis::eZ, PxArticulationMotion::eFREE);
	sJoint->setDriveParams(PxArticulationAxis::eZ, PxArticulationDrive(1.0e5f, 0.0f, PX_MAX_F32));
	sJoint->setDriveTarget(PxArticulationAxis::eZ, 0.8f);
	scene->addArticulation(*slider);

	std::printf("basic scene created, simulating 3s\n");
	const PxU32 frames = 180;
	for(PxU32 i = 0; i < frames; ++i)
	{
		scene->simulate(1.0f / 60.0f);
		scene->fetchResults(true);
		const PxTransform bp = box->getGlobalPose();
		const PxTransform hp = hang->getGlobalPose();
		const PxTransform pp = pChild->getGlobalPose();
		const PxTransform sp = sChild->getGlobalPose();
		if(!finitePose(bp) || !finitePose(hp) || !finitePose(pp) || !finitePose(sp))
		{
			std::printf("FAIL NaN at frame %u\n", i);
			return 1;
		}
		if((i % 60) == 59)
		{
			std::printf("t=%.2f box y=%.3f hang y=%.3f pend=(%.3f,%.3f,%.3f) slide z=%.3f\n",
				(i + 1) / 60.0f, bp.p.y, hp.p.y, pp.p.x, pp.p.y, pp.p.z, sp.p.z);
		}
	}

	const PxVec3 boxP = box->getGlobalPose().p;
	const PxVec3 hangP = hang->getGlobalPose().p;
	const PxVec3 pendP = pChild->getGlobalPose().p;
	const PxVec3 slideP = sChild->getGlobalPose().p;
	int fail = 0;
	if(boxP.y < 0.35f || boxP.y > 1.2f)
	{
		std::printf("FAIL box height y=%.4f (want ~0.5 on plane)\n", boxP.y);
		fail = 1;
	}
	if(PxAbs(boxP.x) > 1.5f || PxAbs(boxP.z) > 1.5f)
	{
		std::printf("FAIL box slid away xz=(%.4f,%.4f)\n", boxP.x, boxP.z);
		fail = 1;
	}
	if(hangP.y < 2.0f || hangP.y > 4.5f)
	{
		std::printf("FAIL hang height y=%.4f (want ~3 on D6)\n", hangP.y);
		fail = 1;
	}
	if(PxAbs(hangP.x - 4.0f) > 1.5f)
	{
		std::printf("FAIL hang x=%.4f drifted from 4\n", hangP.x);
		fail = 1;
	}
	if(pendP.y < 2.5f || pendP.y > 4.5f)
	{
		std::printf("FAIL pendulum y=%.4f (want ~3.5 hanging)\n", pendP.y);
		fail = 1;
	}
	if(PxAbs(pendP.x + 4.0f) > 0.8f)
	{
		std::printf("FAIL pendulum x=%.4f drifted from -4\n", pendP.x);
		fail = 1;
	}
	if(PxAbs(slideP.z - 0.8f) > 0.25f)
	{
		std::printf("FAIL slider z=%.4f (want ~0.8 drive target)\n", slideP.z);
		fail = 1;
	}

	joint->release();
	slider->release();
	pendulum->release();
	scene->release();

	if(fail)
	{
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	std::printf("PASS box y=%.4f hang y=%.4f pend y=%.4f slide z=%.4f\n",
		boxP.y, hangP.y, pendP.y, slideP.z);

	PxSceneDesc scissorDesc(physics->getTolerancesScale());
	scissorDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	scissorDesc.cpuDispatcher = dispatcher;
	scissorDesc.filterShader = scissorFilter;
	scissorDesc.solverType = PxSolverType::eAVBD;
	PxScene* scissorScene = physics->createScene(scissorDesc);
	if(!scissorScene)
	{
		std::printf("FAIL create scissor AVBD scene\n");
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	PxRigidStatic* scissorGround = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *mat);
	scissorScene->addActor(*scissorGround);
	PxArticulationJointReducedCoordinate* driveJoint = NULL;
	PxArticulationLink* scissorBase = NULL;
	PxArticulationLink* scissorTop = NULL;
	PxArticulationReducedCoordinate* scissor =
		createScissorLift(physics, scissorScene, mat, &driveJoint, &scissorBase, &scissorTop);

	std::printf("scissor scene created, simulating 10s with drive cycle\n");
	bool closing = true;
	int scissorFail = 0;
	PxReal topYMin = PX_MAX_F32;
	PxReal topYMax = -PX_MAX_F32;
	const PxU32 scissorFrames = 600;
	for(PxU32 i = 0; i < scissorFrames; ++i)
	{
		const PxReal dt = 1.0f / 60.0f;
		PxReal driveValue = driveJoint->getDriveTarget(PxArticulationAxis::eZ);
		if(closing && driveValue < -1.2f)
			closing = false;
		else if(!closing && driveValue > 0.0f)
			closing = true;
		if(closing)
			driveValue -= dt * 0.25f;
		else
			driveValue += dt * 0.25f;
		driveJoint->setDriveTarget(PxArticulationAxis::eZ, driveValue);

		scissorScene->simulate(dt);
		scissorScene->fetchResults(true);

		const PxTransform basePose = scissorBase->getGlobalPose();
		const PxTransform topPose = scissorTop->getGlobalPose();
		if(!finitePose(basePose) || !finitePose(topPose))
		{
			std::printf("FAIL scissor NaN at frame %u\n", i);
			scissorFail = 1;
			break;
		}
		topYMin = PxMin(topYMin, topPose.p.y);
		topYMax = PxMax(topYMax, topPose.p.y);
		const PxVec3 baseUp = basePose.q.rotate(PxVec3(0.0f, 1.0f, 0.0f));
		if(basePose.p.y < 0.15f || basePose.p.y > 2.5f ||
			PxAbs(basePose.p.x) > 4.0f || PxAbs(basePose.p.z) > 4.0f ||
			baseUp.y < 0.5f ||
			topPose.p.y < 1.5f || topPose.p.y > 8.5f ||
			topPose.p.y + 0.5f < basePose.p.y ||
			topPose.p.magnitude() > 12.0f)
		{
			std::printf("FAIL scissor fell frame %u base=(%.3f,%.3f,%.3f) top=(%.3f,%.3f,%.3f) drive=%.3f upY=%.3f\n",
				i, basePose.p.x, basePose.p.y, basePose.p.z,
				topPose.p.x, topPose.p.y, topPose.p.z, driveValue, baseUp.y);
			scissorFail = 1;
			break;
		}
		if((i % 120) == 119)
			std::printf("scissor t=%.1f base y=%.3f top y=%.3f drive z=%.3f\n",
				(i + 1) / 60.0f, basePose.p.y, topPose.p.y, driveValue);
	}

	const PxTransform baseEnd = scissorBase->getGlobalPose();
	const PxTransform topEnd = scissorTop->getGlobalPose();
	scissor->release();

	if(scissorFail)
	{
		scissorScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	const PxReal stroke = topYMax - topYMin;
	if(stroke < 0.25f)
	{
		std::printf("FAIL scissor stroke %.3f (want >0.25) top y in [%.3f, %.3f]\n",
			stroke, topYMin, topYMax);
		scissorScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	std::printf("PASS scissor 10s base y=%.4f top y=%.4f stroke=%.3f\n",
		baseEnd.p.y, topEnd.p.y, stroke);
	scissorScene->release();

	// P4b: VBD from island PxDeformableVolume (not Dy sidecar).
	// OGC anti-tunneling contract: scene CCD, body CCD, and pair CCD are all
	// deliberately disabled. Host DCD publishes the inertial query neighborhood;
	// the AVBD 6x6 block remains inside the resulting conservative region.
	PxScene* ogcScene = smokeAvbdScene(physics, dispatcher, ogcDcdFilter, NULL, false);
	if(!ogcScene)
	{
		std::printf("FAIL create rigid OGC scene\n");
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	PxRigidStatic* ogcWall = PxCreateStatic(*physics, PxTransform(PxVec3(0.5f, 1.0f, 0.0f)),
		PxBoxGeometry(0.02f, 1.0f, 1.0f), *mat);
	ogcScene->addActor(*ogcWall);
	PxRigidDynamic* ogcBox = PxCreateDynamic(*physics, PxTransform(PxVec3(-0.5f, 1.0f, 0.0f)),
		PxBoxGeometry(0.2f, 0.2f, 0.2f), *mat, 1.0f);
	ogcBox->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	ogcBox->setLinearVelocity(PxVec3(80.0f, 0.0f, 0.0f));
	ogcScene->addActor(*ogcBox);
	for(PxU32 f = 0; f < 30; ++f)
	{
		ogcScene->simulate(1.0f / 60.0f);
		ogcScene->fetchResults(true);
		if(std::getenv("PHYSX_AVBD_OGC_REGION_TRACE"))
		{
			const PxTransform tracePose = ogcBox->getGlobalPose();
			const PxVec3 traceVelocity = ogcBox->getLinearVelocity();
			std::printf("[RIGID_OGC_TEST] frame=%u x=%.9g vx=%.9g\n",
				f + 1, double(tracePose.p.x), double(traceVelocity.x));
			std::fflush(stdout);
		}
	}
	const PxTransform ogcP = ogcBox->getGlobalPose();
	// The old x<=2 gate accepted a frozen body at its initial x=-0.5.  A valid
	// OGC solve must consume the free-space domain and settle immediately before
	// the wall (wall face x=0.48, box support radius 0.2).
	if(!finitePose(ogcP) || ogcP.p.x < 0.15f || ogcP.p.x > 0.35f)
	{
		std::printf("FAIL rigid OGC no-CCD x=%.3f (DCD region must stop tunnel)\n", ogcP.p.x);
		ogcScene->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS rigid OGC no-CCD x=%.4f\n", ogcP.p.x);
	ogcScene->release();

	std::printf("Particle3 volume-ground start\n");
	PxSceneDesc softDesc(physics->getTolerancesScale());
	softDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	softDesc.cpuDispatcher = dispatcher;
	softDesc.filterShader = PxDefaultSimulationFilterShader;
	softDesc.solverType = PxSolverType::eAVBD;
	PxScene* softScene = physics->createScene(softDesc);
	if(!softScene)
	{
		std::printf("FAIL create soft AVBD scene\n");
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	PxRigidStatic* softGround = PxCreatePlane(*physics, PxPlane(0, 1, 0, 0), *mat);
	softScene->addActor(*softGround);

	PxDeformableVolume* islandVol = physics->createDeformableVolume();
	PxDeformableVolumeMesh* islandMesh = cookCubeVolumeMesh(*physics, PxVec3(0.5f, 2.5f, 0.5f), 1.0f);
	PxDeformableVolumeMaterial* volMat = physics->createDeformableVolumeMaterial(1.0e5f, 0.3f, 0.1f);
	if(volMat) volMat->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
	PxShape* volShape = NULL;
	if(islandVol && islandMesh && volMat)
	{
		PxTetrahedronMeshGeometry geom(islandMesh->getCollisionMesh());
		const PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE;
		volShape = physics->createShape(geom, &volMat, 1, true, shapeFlags);
	}
	const bool attachedShape = volShape && islandVol && islandVol->attachShape(*volShape);
	const bool attachedSim = attachedShape && islandVol->attachSimulationMesh(
		*islandMesh->getSimulationMesh(), *islandMesh->getDeformableVolumeAuxData());
	const bool buffersOk = attachedSim && initVolumeHostBuffers(*islandVol, 100.0f);
	const bool added = buffersOk && softScene->addActor(*islandVol);
	if(!islandVol || !islandMesh || !volMat || !volShape || !attachedShape || !attachedSim || !buffersOk || !added)
	{
		std::printf("FAIL island volume create/attach/addActor\n");
		if(islandVol) islandVol->release();
		if(islandMesh) islandMesh->release();
		if(volMat) volMat->release();
		softScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	const PxTetrahedronMesh* islandSimMesh = islandVol->getSimulationMesh();
	const PxU32 islandNbVerts = islandSimMesh->getNbVertices();
	const PxReal restVol = tetMeshAbsVolume(*islandSimMesh, islandVol->getSimPositionInvMassBufferD());

	for(PxU32 f = 0; f < 180; ++f)
	{
		softScene->simulate(1.0f / 60.0f);
		softScene->fetchResults(true);
	}

	const PxU32 softNodes = avbdHostActiveDeformableVolumeNodeCount(softScene);
	const PxVec4* simPos = islandVol->getSimPositionInvMassBufferD();
	if(!simPos || islandNbVerts == 0)
	{
		std::printf("FAIL island volume fetch\n");
		softScene->removeActor(*islandVol);
		islandVol->release();
		islandMesh->release();
		volMat->release();
		softScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	PxVec3 com(0.0f);
	PxReal massSum = 0.0f;
	PxReal minY = PX_MAX_F32;
	PxReal maxY = -PX_MAX_F32;
	for(PxU32 i = 0; i < islandNbVerts; ++i)
	{
		const PxVec3 p = simPos[i].getXYZ();
		if(!p.isFinite())
		{
			std::printf("FAIL island NaN particle %u\n", i);
			softScene->removeActor(*islandVol);
			islandVol->release();
			islandMesh->release();
			volMat->release();
			softScene->release();
			dispatcher->release();
			PxCloseExtensions();
			physics->release();
			foundation->release();
			return 1;
		}
		const PxReal invMass = simPos[i].w;
		const PxReal mass = (invMass > 1.0e-8f) ? (1.0f / invMass) : 0.0f;
		com += p * mass;
		massSum += mass;
		minY = PxMin(minY, p.y);
		maxY = PxMax(maxY, p.y);
	}
	if(massSum > 0.0f)
		com *= 1.0f / massSum;
	const PxReal liveVol = tetMeshAbsVolume(*islandSimMesh, simPos);
	const PxReal volRatio = (restVol > 1.0e-8f) ? (liveVol / restVol) : 0.0f;
	if(softNodes < 1 || com.y < -0.5f || com.y > 1.5f || minY < -0.15f ||
		(maxY - minY) < 0.3f || volRatio < 0.5f || volRatio > 1.5f)
	{
		std::printf("FAIL island volume settle nodes=%u com=(%.3f,%.3f,%.3f) y=[%.3f,%.3f] volRatio=%.3f\n",
			softNodes, com.x, com.y, com.z, minY, maxY, volRatio);
		softScene->removeActor(*islandVol);
		islandVol->release();
		islandMesh->release();
		volMat->release();
		softScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	std::printf("PASS island volume 3s nodes=%u com y=%.4f minY=%.4f volRatio=%.3f\n",
		softNodes, com.y, minY, volRatio);

	releaseIslandVolume(islandVol, islandMesh, volMat);
	softScene->release();

	// P4b: static-box SDF from Host PxRigidStatic (not Dy sidecar).
	std::printf("Particle3 volume-on-static-box start\n");
	PxSceneDesc softBoxDesc(physics->getTolerancesScale());
	softBoxDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	softBoxDesc.cpuDispatcher = dispatcher;
	softBoxDesc.filterShader = PxDefaultSimulationFilterShader;
	softBoxDesc.solverType = PxSolverType::eAVBD;
	PxScene* softBoxScene = physics->createScene(softBoxDesc);
	if(!softBoxScene)
	{
		std::printf("FAIL create island-box AVBD scene\n");
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	PxRigidStatic* softBoxGround = PxCreatePlane(*physics, PxPlane(0, 1, 0, 0), *mat);
	softBoxScene->addActor(*softBoxGround);

	const PxReal platformTop = 1.0f;
	PxRigidStatic* platform = physics->createRigidStatic(PxTransform(PxVec3(0.5f, 0.5f, 0.5f)));
	PxShape* platformShape = PxRigidActorExt::createExclusiveShape(*platform, PxBoxGeometry(2.0f, 0.5f, 2.0f), *mat);
	if(platformShape)
		platformShape->setContactOffset(0.05f);
	softBoxScene->addActor(*platform);

	PxDeformableVolume* boxVol = physics->createDeformableVolume();
	PxDeformableVolumeMesh* boxMesh = cookCubeVolumeMesh(*physics, PxVec3(0.5f, 3.5f, 0.5f), 1.0f);
	PxDeformableVolumeMaterial* boxVolMat = physics->createDeformableVolumeMaterial(1.0e5f, 0.3f, 0.1f);
	if(boxVolMat) boxVolMat->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
	PxShape* boxVolShape = NULL;
	if(boxVol && boxMesh && boxVolMat)
	{
		PxTetrahedronMeshGeometry geom(boxMesh->getCollisionMesh());
		const PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE;
		boxVolShape = physics->createShape(geom, &boxVolMat, 1, true, shapeFlags);
		if(boxVolShape)
			boxVolShape->setContactOffset(0.05f);
	}
	const bool boxAttachedShape = boxVolShape && boxVol && boxVol->attachShape(*boxVolShape);
	const bool boxAttachedSim = boxAttachedShape && boxVol->attachSimulationMesh(
		*boxMesh->getSimulationMesh(), *boxMesh->getDeformableVolumeAuxData());
	const bool boxBuffersOk = boxAttachedSim && initVolumeHostBuffers(*boxVol, 100.0f);
	const bool boxAdded = boxBuffersOk && softBoxScene->addActor(*boxVol);
	if(!boxVol || !boxMesh || !boxVolMat || !boxVolShape || !boxAttachedShape ||
		!boxAttachedSim || !boxBuffersOk || !boxAdded)
	{
		std::printf("FAIL island volume on-box create/attach/addActor\n");
		if(boxVol) boxVol->release();
		if(boxMesh) boxMesh->release();
		if(boxVolMat) boxVolMat->release();
		softBoxScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	const PxTetrahedronMesh* boxSimMesh = boxVol->getSimulationMesh();
	const PxU32 boxNbVerts = boxSimMesh->getNbVertices();
	const PxReal boxRestVol = tetMeshAbsVolume(*boxSimMesh, boxVol->getSimPositionInvMassBufferD());

	for(PxU32 f = 0; f < 180; ++f)
	{
		softBoxScene->simulate(1.0f / 60.0f);
		softBoxScene->fetchResults(true);
	}

	const PxVec4* boxSimPos = boxVol->getSimPositionInvMassBufferD();
	if(!boxSimPos || boxNbVerts == 0)
	{
		std::printf("FAIL island volume on-box fetch\n");
		softBoxScene->removeActor(*boxVol);
		boxVol->release();
		boxMesh->release();
		boxVolMat->release();
		softBoxScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	PxVec3 boxCom(0.0f);
	PxReal boxMassSum = 0.0f;
	PxReal boxMinY = PX_MAX_F32;
	PxReal boxMaxY = -PX_MAX_F32;
	for(PxU32 i = 0; i < boxNbVerts; ++i)
	{
		const PxVec3 p = boxSimPos[i].getXYZ();
		if(!p.isFinite())
		{
			std::printf("FAIL island-box NaN particle %u\n", i);
			softBoxScene->removeActor(*boxVol);
			boxVol->release();
			boxMesh->release();
			boxVolMat->release();
			softBoxScene->release();
			dispatcher->release();
			PxCloseExtensions();
			physics->release();
			foundation->release();
			return 1;
		}
		const PxReal invMass = boxSimPos[i].w;
		const PxReal mass = (invMass > 1.0e-8f) ? (1.0f / invMass) : 0.0f;
		boxCom += p * mass;
		boxMassSum += mass;
		boxMinY = PxMin(boxMinY, p.y);
		boxMaxY = PxMax(boxMaxY, p.y);
	}
	if(boxMassSum > 0.0f)
		boxCom *= 1.0f / boxMassSum;
	const PxReal boxLiveVol = tetMeshAbsVolume(*boxSimMesh, boxSimPos);
	const PxReal boxVolRatio = (boxRestVol > 1.0e-8f) ? (boxLiveVol / boxRestVol) : 0.0f;
	// Soft must rest on platform top (~1.0), not fall through to ground (~0.12).
	if(boxMinY < platformTop - 0.2f || boxCom.y < platformTop || boxCom.y > 3.0f ||
		(boxMaxY - boxMinY) < 0.3f || boxVolRatio < 0.5f || boxVolRatio > 1.5f)
	{
		std::printf("FAIL island-box settle com=(%.3f,%.3f,%.3f) y=[%.3f,%.3f] volRatio=%.3f (want minY>~%.2f)\n",
			boxCom.x, boxCom.y, boxCom.z, boxMinY, boxMaxY, boxVolRatio, platformTop - 0.2f);
		softBoxScene->removeActor(*boxVol);
		boxVol->release();
		boxMesh->release();
		boxVolMat->release();
		softBoxScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	std::printf("PASS island volume on-box 3s com y=%.4f minY=%.4f volRatio=%.3f\n",
		boxCom.y, boxMinY, boxVolRatio);

	releaseIslandVolume(boxVol, boxMesh, boxVolMat);
	softBoxScene->release();

	// P6: rotated static box is an OBB, not a world AABB.
	std::printf("Particle3 volume-on-rotated-box start\n");
	PxSceneDesc rotBoxDesc(physics->getTolerancesScale());
	rotBoxDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	rotBoxDesc.cpuDispatcher = dispatcher;
	rotBoxDesc.filterShader = PxDefaultSimulationFilterShader;
	rotBoxDesc.solverType = PxSolverType::eAVBD;
	PxScene* rotBoxScene = physics->createScene(rotBoxDesc);
	if(!rotBoxScene)
	{
		std::printf("FAIL create island-rotbox AVBD scene\n");
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	PxRigidStatic* rotGround = PxCreatePlane(*physics, PxPlane(0, 1, 0, 0), *mat);
	rotBoxScene->addActor(*rotGround);

	const PxQuat rotY(PxPi * 0.25f, PxVec3(0.0f, 1.0f, 0.0f));
	PxRigidStatic* rotPlatform = physics->createRigidStatic(PxTransform(PxVec3(0.5f, 0.5f, 0.5f), rotY));
	PxShape* rotPlatShape = PxRigidActorExt::createExclusiveShape(*rotPlatform, PxBoxGeometry(2.0f, 0.5f, 2.0f), *mat);
	if(rotPlatShape)
		rotPlatShape->setContactOffset(0.05f);
	rotBoxScene->addActor(*rotPlatform);

	// Drop in an xz corner that sits inside the world AABB but outside the OBB diamond.
	// AABB SDF would catch the cube at y~1.0; OBB SDF lets it fall to the ground.
	PxDeformableVolume* rotVol = physics->createDeformableVolume();
	PxDeformableVolumeMesh* rotMesh = cookCubeVolumeMesh(*physics, PxVec3(2.7f, 2.5f, 2.7f), 1.0f);
	PxDeformableVolumeMaterial* rotVolMat = physics->createDeformableVolumeMaterial(1.0e5f, 0.3f, 0.1f);
	if(rotVolMat) rotVolMat->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
	PxShape* rotVolShape = NULL;
	if(rotVol && rotMesh && rotVolMat)
	{
		PxTetrahedronMeshGeometry geom(rotMesh->getCollisionMesh());
		const PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE;
		rotVolShape = physics->createShape(geom, &rotVolMat, 1, true, shapeFlags);
		if(rotVolShape)
			rotVolShape->setContactOffset(0.6f);
	}
	const bool rotAttachedShape = rotVolShape && rotVol && rotVol->attachShape(*rotVolShape);
	const bool rotAttachedSim = rotAttachedShape && rotVol->attachSimulationMesh(
		*rotMesh->getSimulationMesh(), *rotMesh->getDeformableVolumeAuxData());
	const bool rotBuffersOk = rotAttachedSim && initVolumeHostBuffers(*rotVol, 100.0f);
	const bool rotAdded = rotBuffersOk && rotBoxScene->addActor(*rotVol);
	if(!rotVol || !rotMesh || !rotVolMat || !rotVolShape || !rotAttachedShape ||
		!rotAttachedSim || !rotBuffersOk || !rotAdded)
	{
		std::printf("FAIL island volume on-rotbox create/attach/addActor\n");
		if(rotVol) rotVol->release();
		if(rotMesh) rotMesh->release();
		if(rotVolMat) rotVolMat->release();
		rotBoxScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	const PxTetrahedronMesh* rotSimMesh = rotVol->getSimulationMesh();
	const PxU32 rotNbVerts = rotSimMesh->getNbVertices();
	const PxReal rotRestVol = tetMeshAbsVolume(*rotSimMesh, rotVol->getSimPositionInvMassBufferD());

	for(PxU32 f = 0; f < 180; ++f)
	{
		rotBoxScene->simulate(1.0f / 60.0f);
		rotBoxScene->fetchResults(true);
	}

	const PxVec4* rotSimPos = rotVol->getSimPositionInvMassBufferD();
	if(!rotSimPos || rotNbVerts == 0)
	{
		std::printf("FAIL island volume on-rotbox fetch\n");
		rotBoxScene->removeActor(*rotVol);
		rotVol->release();
		rotMesh->release();
		rotVolMat->release();
		rotBoxScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	PxVec3 rotCom(0.0f);
	PxReal rotMassSum = 0.0f;
	PxReal rotMinY = PX_MAX_F32;
	PxReal rotMaxY = -PX_MAX_F32;
	for(PxU32 i = 0; i < rotNbVerts; ++i)
	{
		const PxVec3 p = rotSimPos[i].getXYZ();
		if(!p.isFinite())
		{
			std::printf("FAIL island-rotbox NaN particle %u\n", i);
			rotBoxScene->removeActor(*rotVol);
			rotVol->release();
			rotMesh->release();
			rotVolMat->release();
			rotBoxScene->release();
			dispatcher->release();
			PxCloseExtensions();
			physics->release();
			foundation->release();
			return 1;
		}
		const PxReal invMass = rotSimPos[i].w;
		const PxReal mass = (invMass > 1.0e-8f) ? (1.0f / invMass) : 0.0f;
		rotCom += p * mass;
		rotMassSum += mass;
		rotMinY = PxMin(rotMinY, p.y);
		rotMaxY = PxMax(rotMaxY, p.y);
	}
	if(rotMassSum > 0.0f)
		rotCom *= 1.0f / rotMassSum;
	const PxReal rotLiveVol = tetMeshAbsVolume(*rotSimMesh, rotSimPos);
	const PxReal rotVolRatio = (rotRestVol > 1.0e-8f) ? (rotLiveVol / rotRestVol) : 0.0f;
	// Y-rotated box: OBB is a diamond in xz. This drop is inside the world AABB
	// (would rest at y~1.0) but outside the OBB, so it must hit the ground.
	if(rotMinY > 0.70f || rotMinY < -1.0e-5f || rotCom.y > 1.6f ||
		rotCom.x < 1.5f || rotCom.z < 1.5f ||
		(rotMaxY - rotMinY) < 0.3f || rotVolRatio < 0.5f || rotVolRatio > 1.5f)
	{
		std::printf("FAIL island-rotbox settle com=(%.3f,%.3f,%.3f) y=[%.3f,%.3f] volRatio=%.3f (want ground, not AABB shelf y~1)\n",
			rotCom.x, rotCom.y, rotCom.z, rotMinY, rotMaxY, rotVolRatio);
		rotBoxScene->removeActor(*rotVol);
		rotVol->release();
		rotMesh->release();
		rotVolMat->release();
		rotBoxScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	std::printf("PASS island volume on-rotbox 3s com y=%.4f minY=%.4f volRatio=%.3f\n",
		rotCom.y, rotMinY, rotVolRatio);

	releaseIslandVolume(rotVol, rotMesh, rotVolMat);
	rotBoxScene->release();

	// P4b: dynamic box 6x6 reaction (DISABLE_GRAVITY hover; soft weight must push it down).
	std::printf("Particle3 volume-on-dynamic-box start\n");
	PxSceneDesc dynBoxDesc(physics->getTolerancesScale());
	dynBoxDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	dynBoxDesc.cpuDispatcher = dispatcher;
	dynBoxDesc.filterShader = PxDefaultSimulationFilterShader;
	dynBoxDesc.solverType = PxSolverType::eAVBD;
	PxScene* dynBoxScene = physics->createScene(dynBoxDesc);
	if(!dynBoxScene)
	{
		std::printf("FAIL create island-dynbox AVBD scene\n");
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	PxRigidStatic* dynGround = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *mat);
	dynBoxScene->addActor(*dynGround);

	PxRigidDynamic* dynPlatform = physics->createRigidDynamic(PxTransform(PxVec3(0.5f, 2.0f, 0.5f)));
	PxShape* dynPlatShape = PxRigidActorExt::createExclusiveShape(*dynPlatform, PxBoxGeometry(2.0f, 0.5f, 2.0f), *mat);
	if(dynPlatShape)
		dynPlatShape->setContactOffset(0.05f);
	PxRigidBodyExt::updateMassAndInertia(*dynPlatform, 20.0f);
	dynPlatform->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	dynPlatform->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	dynPlatform->setSleepThreshold(0.0f);
	dynPlatform->setStabilizationThreshold(0.0f);
	dynBoxScene->addActor(*dynPlatform);
	dynPlatform->setWakeCounter(100.0f);
	dynPlatform->wakeUp();

	PxDeformableVolume* dynVol = physics->createDeformableVolume();
	PxDeformableVolumeMesh* dynMesh = cookCubeVolumeMesh(*physics, PxVec3(0.5f, 3.5f, 0.5f), 1.0f);
	PxDeformableVolumeMaterial* dynVolMat = physics->createDeformableVolumeMaterial(1.0e5f, 0.3f, 0.1f);
	if(dynVolMat) dynVolMat->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
	PxShape* dynVolShape = NULL;
	if(dynVol && dynMesh && dynVolMat)
	{
		PxTetrahedronMeshGeometry geom(dynMesh->getCollisionMesh());
		const PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE;
		dynVolShape = physics->createShape(geom, &dynVolMat, 1, true, shapeFlags);
		if(dynVolShape)
			dynVolShape->setContactOffset(0.05f);
	}
	const bool dynAttachedShape = dynVolShape && dynVol && dynVol->attachShape(*dynVolShape);
	const bool dynAttachedSim = dynAttachedShape && dynVol->attachSimulationMesh(
		*dynMesh->getSimulationMesh(), *dynMesh->getDeformableVolumeAuxData());
	const bool dynBuffersOk = dynAttachedSim && initVolumeHostBuffers(*dynVol, 100.0f);
	const bool dynAdded = dynBuffersOk && dynBoxScene->addActor(*dynVol);
	if(!dynVol || !dynMesh || !dynVolMat || !dynVolShape || !dynAttachedShape ||
		!dynAttachedSim || !dynBuffersOk || !dynAdded || !dynPlatform)
	{
		std::printf("FAIL island volume on-dynbox create/attach/addActor\n");
		if(dynVol) dynVol->release();
		if(dynMesh) dynMesh->release();
		if(dynVolMat) dynVolMat->release();
		dynBoxScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	const PxTetrahedronMesh* dynSimMesh = dynVol->getSimulationMesh();
	const PxU32 dynNbVerts = dynSimMesh->getNbVertices();
	const PxReal dynRestVol = tetMeshAbsVolume(*dynSimMesh, dynVol->getSimPositionInvMassBufferD());

	for(PxU32 f = 0; f < 180; ++f)
	{
		dynBoxScene->simulate(1.0f / 60.0f);
		dynBoxScene->fetchResults(true);
		if((f + 1) % 60 == 0)
			std::printf("dynbox t=%.2f box y=%.4f\n", PxReal(f + 1) / 60.0f, dynPlatform->getGlobalPose().p.y);
	}

	const PxVec4* dynSimPos = dynVol->getSimPositionInvMassBufferD();
	const PxTransform dynPose = dynPlatform->getGlobalPose();
	if(!dynSimPos || dynNbVerts == 0 || !dynPose.isValid())
	{
		std::printf("FAIL island volume on-dynbox fetch\n");
		dynBoxScene->removeActor(*dynVol);
		dynVol->release();
		dynMesh->release();
		dynVolMat->release();
		dynBoxScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	PxVec3 dynCom(0.0f);
	PxReal dynMassSum = 0.0f;
	PxReal dynMinY = PX_MAX_F32;
	PxReal dynMaxY = -PX_MAX_F32;
	for(PxU32 i = 0; i < dynNbVerts; ++i)
	{
		const PxVec3 p = dynSimPos[i].getXYZ();
		if(!p.isFinite())
		{
			std::printf("FAIL island-dynbox NaN particle %u\n", i);
			dynBoxScene->removeActor(*dynVol);
			dynVol->release();
			dynMesh->release();
			dynVolMat->release();
			dynBoxScene->release();
			dispatcher->release();
			PxCloseExtensions();
			physics->release();
			foundation->release();
			return 1;
		}
		const PxReal invMass = dynSimPos[i].w;
		const PxReal mass = (invMass > 1.0e-8f) ? (1.0f / invMass) : 0.0f;
		dynCom += p * mass;
		dynMassSum += mass;
		dynMinY = PxMin(dynMinY, p.y);
		dynMaxY = PxMax(dynMaxY, p.y);
	}
	if(dynMassSum > 0.0f)
		dynCom *= 1.0f / dynMassSum;
	const PxReal dynLiveVol = tetMeshAbsVolume(*dynSimMesh, dynSimPos);
	const PxReal dynVolRatio = (dynRestVol > 1.0e-8f) ? (dynLiveVol / dynRestVol) : 0.0f;
	const PxReal dynBoxTop = dynPose.p.y + 0.5f;
	// Two-way 6x6: gravity-disabled hover box must be pushed down onto the ground.
	// One-way SDF would leave it at y=2 with the soft sitting on top.
	if(dynPose.p.y > 1.2f || dynPose.p.y < 0.2f ||
		dynMinY < dynBoxTop - 0.35f || dynCom.y < dynBoxTop - 0.1f ||
		(dynMaxY - dynMinY) < 0.3f || dynVolRatio < 0.5f || dynVolRatio > 1.5f)
	{
		std::printf("FAIL island-dynbox settle boxY=%.3f com=(%.3f,%.3f,%.3f) y=[%.3f,%.3f] volRatio=%.3f (want boxY<1.2 on ground, soft on box)\n",
			dynPose.p.y, dynCom.x, dynCom.y, dynCom.z, dynMinY, dynMaxY, dynVolRatio);
		dynBoxScene->removeActor(*dynVol);
		dynVol->release();
		dynMesh->release();
		dynVolMat->release();
		dynBoxScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	std::printf("PASS island volume on-dynbox 3s box y=%.4f com y=%.4f minY=%.4f volRatio=%.3f\n",
		dynPose.p.y, dynCom.y, dynMinY, dynVolRatio);

	releaseIslandVolume(dynVol, dynMesh, dynVolMat);
	dynBoxScene->release();

	// P6: a broad-phase overlap without an active NP witness must not wake a sleeping body.
	std::printf("soft island overlap-sleep start\n");
	PxSceneDesc sleepDesc(physics->getTolerancesScale());
	sleepDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	sleepDesc.cpuDispatcher = dispatcher;
	sleepDesc.filterShader = PxDefaultSimulationFilterShader;
	sleepDesc.solverType = PxSolverType::eAVBD;
	PxScene* sleepScene = physics->createScene(sleepDesc);
	if(!sleepScene)
	{
		std::printf("FAIL create overlap-sleep AVBD scene\n");
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	PxRigidStatic* sleepGround = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *mat);
	sleepScene->addActor(*sleepGround);

	// Both axis gaps (0.18 m) are inside the summed 0.22 m offsets. The actual
	// corner distance (0.255 m) is outside that shell: BP overlap alone must not wake.
	PxRigidDynamic* sleepBox = physics->createRigidDynamic(PxTransform(PxVec3(0.5f, 2.0f, 0.5f)));
	PxShape* sleepBoxShape = PxRigidActorExt::createExclusiveShape(*sleepBox, PxBoxGeometry(0.5f, 0.5f, 0.5f), *mat);
	if(sleepBoxShape)
		sleepBoxShape->setContactOffset(0.02f);
	PxRigidBodyExt::updateMassAndInertia(*sleepBox, 10.0f);
	sleepBox->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	sleepBox->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	sleepScene->addActor(*sleepBox);
	sleepBox->putToSleep();

	PxDeformableVolume* sleepVol = physics->createDeformableVolume();
	PxDeformableVolumeMesh* sleepMesh = cookCubeVolumeMesh(*physics, PxVec3(1.68f, 3.18f, 0.5f), 1.0f);
	PxDeformableVolumeMaterial* sleepVolMat = physics->createDeformableVolumeMaterial(1.0e5f, 0.3f, 0.1f);
	if(sleepVolMat) sleepVolMat->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
	PxShape* sleepVolShape = NULL;
	if(sleepVol && sleepMesh && sleepVolMat)
	{
		PxTetrahedronMeshGeometry geom(sleepMesh->getCollisionMesh());
		const PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE;
		sleepVolShape = physics->createShape(geom, &sleepVolMat, 1, true, shapeFlags);
		if(sleepVolShape)
			sleepVolShape->setContactOffset(0.2f);
	}
	const bool sleepAttachedShape = sleepVolShape && sleepVol && sleepVol->attachShape(*sleepVolShape);
	const bool sleepAttachedSim = sleepAttachedShape && sleepVol->attachSimulationMesh(
		*sleepMesh->getSimulationMesh(), *sleepMesh->getDeformableVolumeAuxData());
	const bool sleepBuffersOk = sleepAttachedSim && initVolumeHostBuffers(*sleepVol, 100.0f);
	if(sleepVol)
		sleepVol->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	const bool sleepVolAdded = sleepBuffersOk && sleepScene->addActor(*sleepVol);
	if(!sleepVol || !sleepMesh || !sleepVolMat || !sleepVolShape || !sleepAttachedShape ||
		!sleepAttachedSim || !sleepBuffersOk || !sleepVolAdded || !sleepBox)
	{
		std::printf("FAIL overlap-sleep create/attach/addActor\n");
		if(sleepVol) sleepVol->release();
		if(sleepMesh) sleepMesh->release();
		if(sleepVolMat) sleepVolMat->release();
		sleepScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	for(PxU32 f = 0; f < 60; ++f)
	{
		sleepScene->simulate(1.0f / 60.0f);
		sleepScene->fetchResults(true);
	}
	const PxReal sleepWc = sleepBox->getWakeCounter();
	const PxReal sleepY = sleepBox->getGlobalPose().p.y;
	// Overlap-only must not raise wakeCounter (no Host internalWakeUp, no SDF wake).
	// The box may still be island-active next to an awake volume; pose must not drift.
	if(sleepWc > 1.0e-4f || PxAbs(sleepY - 2.0f) > 0.02f)
	{
		std::printf("FAIL overlap-sleep wakeCounter=%.4f pose y=%.4f sleeping=%d (want wakeCounter=0, y=2)\n",
			sleepWc, sleepY, int(sleepBox->isSleeping()));
		releaseIslandVolume(sleepVol, sleepMesh, sleepVolMat);
		sleepScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	std::printf("PASS overlap-sleep 1s wakeCounter=%.4f sleeping=%d box y=%.4f\n",
		sleepWc, int(sleepBox->isSleeping()), sleepY);

	releaseIslandVolume(sleepVol, sleepMesh, sleepVolMat);
	sleepScene->release();

	// Particle3 fixed simulation vertices retain their positions while the rest of the volume sags.
	std::printf("Particle3 volume world-pin start\n");
	PxSceneDesc pinDesc(physics->getTolerancesScale());
	pinDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	pinDesc.cpuDispatcher = dispatcher;
	pinDesc.filterShader = PxDefaultSimulationFilterShader;
	pinDesc.solverType = PxSolverType::eAVBD;
	PxScene* pinScene = physics->createScene(pinDesc);
	if(!pinScene)
	{
		std::printf("FAIL create island-pin AVBD scene\n");
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	PxDeformableVolume* pinVol = physics->createDeformableVolume();
	PxDeformableVolumeMesh* pinMesh = cookCubeVolumeMesh(*physics, PxVec3(0.5f, 3.5f, 0.5f), 1.0f);
	PxDeformableVolumeMaterial* pinVolMat = physics->createDeformableVolumeMaterial(1.0e5f, 0.3f, 0.1f);
	if(pinVolMat) pinVolMat->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
	PxShape* pinVolShape = NULL;
	if(pinVol && pinMesh && pinVolMat)
	{
		PxTetrahedronMeshGeometry pinGeom(pinMesh->getCollisionMesh());
		const PxShapeFlags pinFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE;
		pinVolShape = physics->createShape(pinGeom, &pinVolMat, 1, true, pinFlags);
		if(pinVolShape)
			pinVolShape->setContactOffset(0.05f);
	}
	const bool pinAttachedShape = pinVolShape && pinVol && pinVol->attachShape(*pinVolShape);
	const bool pinAttachedSim = pinAttachedShape && pinVol->attachSimulationMesh(
		*pinMesh->getSimulationMesh(), *pinMesh->getDeformableVolumeAuxData());
	const bool pinBuffersOk = pinAttachedSim && initVolumeHostBuffers(*pinVol, 100.0f);
	if(pinVol)
	{
		pinVol->setSleepThreshold(0.0f);
		pinVol->setWakeCounter(100.0f);
	}
	const bool pinAdded = pinBuffersOk && pinScene->addActor(*pinVol);
	if(!pinVol || !pinMesh || !pinVolMat || !pinVolShape || !pinAttachedShape ||
		!pinAttachedSim || !pinBuffersOk || !pinAdded)
	{
		std::printf("FAIL island volume pin create/attach/addActor\n");
		if(pinVol) pinVol->release();
		if(pinMesh) pinMesh->release();
		if(pinVolMat) pinVolMat->release();
		pinScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	const PxTetrahedronMesh* pinSimMesh = pinVol->getSimulationMesh();
	const PxU32 pinNbVerts = pinSimMesh->getNbVertices();
	PxVec4* pinSim0 = pinVol->getSimPositionInvMassBufferD();
	const PxReal pinRestVol = tetMeshAbsVolume(*pinSimMesh, pinSim0);
	PxU32 pinIdx[8];
	PxVec4 pinWorld[8];
	PxU32 nPin = 0;
	PxReal pinTopY = -PX_MAX_F32, restBottomY = PX_MAX_F32;
    for(PxU32 i = 0; i < pinNbVerts; ++i)
    {
        pinTopY = PxMax(pinTopY, pinSim0[i].y);
        restBottomY = PxMin(restBottomY, pinSim0[i].y);
    }
	for(PxU32 i = 0; i < pinNbVerts && nPin < 8; ++i)
	{
		if(PxAbs(pinSim0[i].y - pinTopY) < 0.01f)
		{
			pinIdx[nPin] = i;
			pinWorld[nPin] = PxVec4(pinSim0[i].getXYZ(), 0.0f);
			++nPin;
		}
	}
    PxDeformableAttachmentData attachment;
    attachment.actor[0]=pinVol;
    attachment.type[0]=PxDeformableAttachmentTargetType::eVERTEX;
    attachment.indices[0].data=pinIdx;
    attachment.indices[0].count=nPin;
    attachment.actor[1]=NULL;
    attachment.type[1]=PxDeformableAttachmentTargetType::eWORLD;
    attachment.coords[1].data=pinWorld;
    attachment.coords[1].count=nPin;
    PxDeformableAttachment* unsupported=physics->createDeformableAttachment(attachment);
    const bool rejected=unsupported==NULL;
    if(unsupported)unsupported->release();
    for(PxU32 i = 0; i < nPin; ++i)
        pinSim0[pinIdx[i]].w = 0;
    pinVol->markDirty(PxDeformableVolumeDataFlag::eSIM_POSITION_INVMASS);
    if(!rejected || nPin == 0 || nPin == pinNbVerts)
    {
        std::printf("FAIL CPU attachment rejection or fixed-vertex fixture\n");
		pinScene->removeActor(*pinVol);
		pinVol->release();
		pinMesh->release();
		pinVolMat->release();
		pinScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	for(PxU32 f = 0; f < 180; ++f)
	{
		pinScene->simulate(1.0f / 60.0f);
		pinScene->fetchResults(true);
	}

	const PxVec4* pinSimPos = pinVol->getSimPositionInvMassBufferD();
	if(!pinSimPos || pinNbVerts == 0)
	{
		std::printf("FAIL island volume pin fetch\n");

		pinScene->removeActor(*pinVol);
		pinVol->release();
		pinMesh->release();
		pinVolMat->release();
		pinScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}

	PxReal pinMaxDrift = 0.0f;
	PxReal pinMinY = PX_MAX_F32;
	PxReal pinMaxY = -PX_MAX_F32;
	for(PxU32 i = 0; i < pinNbVerts; ++i)
	{
		const PxVec3 p = pinSimPos[i].getXYZ();
		if(!p.isFinite())
		{
			std::printf("FAIL island-pin NaN particle %u\n", i);

			pinScene->removeActor(*pinVol);
			pinVol->release();
			pinMesh->release();
			pinVolMat->release();
			pinScene->release();
			dispatcher->release();
			PxCloseExtensions();
			physics->release();
			foundation->release();
			return 1;
		}
		pinMinY = PxMin(pinMinY, p.y);
		pinMaxY = PxMax(pinMaxY, p.y);
	}
	for(PxU32 i = 0; i < nPin; ++i)
	{
		const PxVec3 p = pinSimPos[pinIdx[i]].getXYZ();
		const PxVec3 t = pinWorld[i].getXYZ();
		pinMaxDrift = PxMax(pinMaxDrift, (p - t).magnitude());
	}
	const PxReal pinLiveVol = tetMeshAbsVolume(*pinSimMesh, pinSimPos);
	const PxReal pinVolRatio = (pinRestVol > 1.0e-8f) ? (pinLiveVol / pinRestVol) : 0.0f;
	if(pinMinY >= restBottomY || pinMaxDrift > 1.e-5f || (pinMaxY - pinMinY) < 0.1f ||
		pinVolRatio < 0.2f || pinVolRatio > 1.5f)
	{
		std::printf("FAIL island-pin 3s drift=%.3f y=[%.3f,%.3f] volRatio=%.3f (want fixed vertices and downward strain)\n",
			pinMaxDrift, pinMinY, pinMaxY, pinVolRatio);

		pinScene->removeActor(*pinVol);
		pinVol->release();
		pinMesh->release();
		pinVolMat->release();
		pinScene->release();
		dispatcher->release();
		PxCloseExtensions();
		physics->release();
		foundation->release();
		return 1;
	}
	std::printf("PASS island volume pin 3s drift=%.4f y=[%.4f,%.4f] volRatio=%.3f nPin=%u\n",
		pinMaxDrift, pinMinY, pinMaxY, pinVolRatio, nPin);

	releaseIslandVolume(pinVol, pinMesh, pinVolMat);
	pinScene->release();

	// Distilled AVBD_2 rigid matrix: ownership-deep-tilted.  A position
	// recovery may rotate the deeply overlapped box, but that AL correction is
	// not angular momentum.  The body must settle after the recovery.
	PxMaterial* tiltedMat = physics->createMaterial(0.5f, 0.5f, 0.0f);
	PxScene* tiltedScene = smokeAvbdScene(physics, dispatcher,
		PxDefaultSimulationFilterShader, NULL, false);
	if(!tiltedScene || !tiltedMat)
	{
		std::printf("FAIL rigid matrix deep-tilted create\n");
		return smokeAbort(tiltedScene, dispatcher, physics, foundation);
	}
	PxRigidStatic* tiltedGround = PxCreatePlane(*physics,
		PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *tiltedMat);
	tiltedScene->addActor(*tiltedGround);
	const PxTransform tiltedPose(PxVec3(0.0f, 0.25f, 0.0f),
		PxQuat(PxPi * 0.125f, PxVec3(0.0f, 0.0f, 1.0f)));
	PxRigidDynamic* tiltedBox = PxCreateDynamic(*physics, tiltedPose,
		PxBoxGeometry(PxVec3(1.0f, 0.5f, 0.5f)), *tiltedMat, 1.0f);
	tiltedBox->setLinearDamping(0.0f);
	tiltedBox->setAngularDamping(0.0f);
	tiltedBox->setSleepThreshold(0.0f);
	tiltedScene->addActor(*tiltedBox);
	PxReal tiltedPeakAngular = 0.0f;
	PxReal tiltedMinY = PX_MAX_F32;
	PxReal tiltedMaxY = -PX_MAX_F32;
	for(PxU32 f = 0; f < 120; ++f)
	{
		tiltedScene->simulate(1.0f / 60.0f);
		tiltedScene->fetchResults(true);
		const PxTransform pose = tiltedBox->getGlobalPose();
		const PxVec3 lin = tiltedBox->getLinearVelocity();
		const PxVec3 ang = tiltedBox->getAngularVelocity();
		if(!finitePose(pose) || !lin.isFinite() || !ang.isFinite())
		{
			std::printf("FAIL rigid matrix deep-tilted NaN\n");
			tiltedScene->release();
			tiltedMat->release();
			return smokeAbort(NULL, dispatcher, physics, foundation);
		}
		tiltedPeakAngular = PxMax(tiltedPeakAngular, ang.magnitude());
		tiltedMinY = PxMin(tiltedMinY, pose.p.y);
		tiltedMaxY = PxMax(tiltedMaxY, pose.p.y);
		if(std::getenv("PHYSX_AVBD_RIGID_MATRIX_TRACE"))
			std::printf("[DEEP_TILTED] frame=%u y=%.9g v=(%.9g,%.9g,%.9g) "
				"w=(%.9g,%.9g,%.9g)\n", f + 1, double(pose.p.y),
				double(lin.x), double(lin.y), double(lin.z), double(ang.x),
				double(ang.y), double(ang.z));
	}
	const PxReal tiltedFinalSpeed = tiltedBox->getLinearVelocity().magnitude();
	const PxReal tiltedFinalAngular = tiltedBox->getAngularVelocity().magnitude();
	if(tiltedMinY < 0.10f || tiltedMaxY < 0.45f ||
		tiltedFinalSpeed > 0.01f || tiltedFinalAngular > 0.01f)
	{
		std::printf("FAIL rigid matrix deep-tilted peakW=%.6f finalV=%.6f finalW=%.6f y=[%.6f,%.6f]\n",
			tiltedPeakAngular, tiltedFinalSpeed, tiltedFinalAngular,
			tiltedMinY, tiltedMaxY);
		tiltedScene->release();
		tiltedMat->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS rigid matrix deep-tilted peakW=%.6f finalV=%.6f finalW=%.6f y=[%.6f,%.6f]\n",
		tiltedPeakAngular, tiltedFinalSpeed, tiltedFinalAngular,
		tiltedMinY, tiltedMaxY);
	tiltedScene->release();
	tiltedMat->release();

	// Distilled rigid matrix: a one-newton-second off-center contact budget is
	// a spatial impulse.  It must change both COM and angular velocity by the
	// same capped impulse, not cap the two responses independently.
	SmokeModifyCallback finiteCb;
	finiteCb.mode = eSMOKE_MODIFY_FINITE_ONE;
	PxMaterial* finiteMat = physics->createMaterial(0.0f, 0.0f, 0.5f);
	PxScene* finiteScene = smokeAvbdScene(physics, dispatcher,
		modifyContactFilter, &finiteCb, false);
	if(!finiteScene || !finiteMat)
	{
		std::printf("FAIL rigid matrix finite-offcenter create\n");
		return smokeAbort(finiteScene, dispatcher, physics, foundation);
	}
	finiteScene->setBounceThresholdVelocity(2.0f);
	finiteScene->setGravity(PxVec3(0.0f));
	PxRigidStatic* finiteAnchor = PxCreateStatic(*physics,
		PxTransform(PxVec3(-0.6f, 0.0f, 0.0f)),
		PxSphereGeometry(0.25f), *finiteMat);
	finiteScene->addActor(*finiteAnchor);
	PxRigidDynamic* finiteBox = PxCreateDynamic(*physics,
		PxTransform(PxVec3(0.0f, 1.35f, 0.0f)),
		PxBoxGeometry(PxVec3(1.0f, 0.5f, 0.5f)), *finiteMat, 1.0f);
	finiteBox->setLinearDamping(0.0f);
	finiteBox->setAngularDamping(0.0f);
	finiteBox->setLinearVelocity(PxVec3(0.0f, -6.0f, 0.0f));
	finiteScene->addActor(*finiteBox);
	PxReal finiteVy = 0.0f;
	PxReal finiteW = 0.0f;
	for(PxU32 f = 0; f < 30; ++f)
	{
		finiteScene->simulate(1.0f / 60.0f);
		finiteScene->fetchResults(true);
		// AVBD_2 records this after completedFrames==6. This zero-based loop's
		// matching sample is f==5; f==6 observes a second, independently capped
		// simulation step and is not the one-impulse oracle.
		if(f == 5)
		{
			finiteVy = finiteBox->getLinearVelocity().y;
			finiteW = finiteBox->getAngularVelocity().magnitude();
		}
	}
	if(!PxIsFinite(finiteVy) || !PxIsFinite(finiteW) ||
		finiteVy < -5.52f || finiteVy > -5.48f ||
		finiteW < 0.70f || finiteW > 0.74f)
	{
		std::printf("FAIL rigid matrix finite-offcenter checkpointVy=%.9f checkpointW=%.9f\n",
			finiteVy, finiteW);
		finiteScene->release();
		finiteMat->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS rigid matrix finite-offcenter checkpointVy=%.9f checkpointW=%.9f\n",
		finiteVy, finiteW);
	finiteScene->release();
	finiteMat->release();

	// Distilled rigid matrix: a tilted contact must reconstruct restitution as
	// one spatial manifold response, retaining the coupled angular component.
	PxMaterial* spatialMat = physics->createMaterial(0.0f, 0.0f, 0.5f);
	PxScene* spatialScene = smokeAvbdScene(physics, dispatcher,
		PxDefaultSimulationFilterShader, NULL, false);
	if(!spatialScene || !spatialMat)
	{
		std::printf("FAIL rigid matrix restitution-tilted create\n");
		return smokeAbort(spatialScene, dispatcher, physics, foundation);
	}
	spatialScene->setBounceThresholdVelocity(2.0f);
	spatialScene->setGravity(PxVec3(0.0f));
	PxRigidStatic* spatialGround = PxCreatePlane(*physics,
		PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *spatialMat);
	spatialScene->addActor(*spatialGround);
	PxRigidDynamic* spatialBox = PxCreateDynamic(*physics,
		PxTransform(PxVec3(0.0f, 0.9f, 0.0f),
			PxQuat(PxPi * 0.125f, PxVec3(0.0f, 0.0f, 1.0f))),
		PxBoxGeometry(PxVec3(1.0f, 0.5f, 0.5f)), *spatialMat, 1.0f);
	spatialBox->setLinearDamping(0.0f);
	spatialBox->setAngularDamping(0.0f);
	spatialBox->setLinearVelocity(PxVec3(0.0f, -6.0f, 0.0f));
	spatialScene->addActor(*spatialBox);
	PxReal spatialPeakUp = 0.0f;
	PxReal spatialPeakW = 0.0f;
	for(PxU32 f = 0; f < 120; ++f)
	{
		spatialScene->simulate(1.0f / 60.0f);
		spatialScene->fetchResults(true);
		spatialPeakUp = PxMax(spatialPeakUp,
			spatialBox->getLinearVelocity().y);
		spatialPeakW = PxMax(spatialPeakW,
			spatialBox->getAngularVelocity().magnitude());
	}
	const PxReal spatialFinalW = spatialBox->getAngularVelocity().magnitude();
	if(!PxIsFinite(spatialPeakUp) || !PxIsFinite(spatialPeakW) ||
		!PxIsFinite(spatialFinalW) || spatialPeakUp < 0.25f ||
		spatialPeakW < 0.05f || spatialPeakW > 100.0f)
	{
		std::printf("FAIL rigid matrix restitution-tilted peakUp=%.9f peakW=%.9f finalW=%.9f\n",
			spatialPeakUp, spatialPeakW, spatialFinalW);
		spatialScene->release();
		spatialMat->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS rigid matrix restitution-tilted peakUp=%.9f peakW=%.9f finalW=%.9f\n",
		spatialPeakUp, spatialPeakW, spatialFinalW);
	spatialScene->release();
	spatialMat->release();

	// Distilled AVBD reference matrix: a tumbling box crosses one-point and
	// multi-point ground manifolds while retaining one inertial prediction
	// epoch. The landing must converge instead of publishing pose corrections
	// as persistent angular motion.
	PxMaterial* landingMat = physics->createMaterial(0.5f, 0.5f, 0.6f);
	PxScene* landingScene = smokeAvbdScene(physics, dispatcher,
		PxDefaultSimulationFilterShader, NULL, false);
	if(!landingScene || !landingMat)
	{
		std::printf("FAIL rigid matrix tumbling-landing create\n");
		return smokeAbort(landingScene, dispatcher, physics, foundation);
	}
	PxRigidStatic* landingGround = PxCreatePlane(*physics,
		PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *landingMat);
	PxRigidDynamic* landingBox = smokeRigidProbeBox(physics, landingMat,
		PxVec3(0.0f, 4.0f, 0.0f), PxVec3(8.0f, 0.0f, 1.0f));
	if(!landingGround || !landingBox)
	{
		std::printf("FAIL rigid matrix tumbling-landing actors\n");
		landingScene->release();
		landingMat->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	landingBox->setGlobalPose(PxTransform(PxVec3(0.0f, 4.0f, 0.0f),
		PxQuat(0.65f, PxVec3(0.0f, 0.0f, 1.0f))));
	landingBox->setAngularVelocity(PxVec3(2.0f, 4.0f, 6.0f));
	landingScene->addActor(*landingGround);
	landingScene->addActor(*landingBox);
	PxQuat landingPrevious = landingBox->getGlobalPose().q;
	PxReal landingMaxPoseStep = 0.0f;
	PxReal landingX5 = 0.0f;
	bool landingFinite = true;
	for(PxU32 f = 0; f < 1200; ++f)
	{
		landingScene->simulate(1.0f / 60.0f);
		landingScene->fetchResults(true);
		const PxTransform pose = landingBox->getGlobalPose();
		landingMaxPoseStep = PxMax(landingMaxPoseStep,
			smokeQuatStepAngle(landingPrevious, pose.q));
		landingPrevious = pose.q;
		landingFinite = landingFinite && finitePose(pose) &&
			landingBox->getLinearVelocity().isFinite() &&
			landingBox->getAngularVelocity().isFinite();
		if(f == 299)
			landingX5 = pose.p.x;
	}
	const PxReal landingFinalX = landingBox->getGlobalPose().p.x;
	const PxReal landingFinalV = landingBox->getLinearVelocity().magnitude();
	const PxReal landingFinalW = landingBox->getAngularVelocity().magnitude();
	if(!landingFinite || landingFinalV >= 0.02f || landingFinalW >= 0.02f ||
		PxAbs(landingFinalX - landingX5) >= 1.0f)
	{
		std::printf("FAIL rigid matrix tumbling-landing x5=%.9f x20=%.9f finalV=%.9f finalW=%.9f maxPoseStep=%.9f\n",
			landingX5, landingFinalX, landingFinalV, landingFinalW,
			landingMaxPoseStep);
		landingScene->release();
		landingMat->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS rigid matrix tumbling-landing x5=%.9f x20=%.9f finalV=%.9f finalW=%.9f maxPoseStep=%.9f\n",
		landingX5, landingFinalX, landingFinalV, landingFinalW,
		landingMaxPoseStep);
	landingScene->release();
	landingMat->release();

	// The large high-spin box stresses contact-point migration under a sparse
	// solver budget. This exposes landing angular discontinuities in isolation
	// from a stack.
	PxMaterial* highSpinMat = physics->createMaterial(0.5f, 0.5f, 0.6f);
	PxScene* highSpinScene = smokeAvbdScene(physics, dispatcher,
		PxDefaultSimulationFilterShader, NULL, false);
	if(!highSpinScene || !highSpinMat)
	{
		std::printf("FAIL rigid matrix high-spin-landing create\n");
		return smokeAbort(highSpinScene, dispatcher, physics, foundation);
	}
	PxRigidStatic* highSpinGround = PxCreatePlane(*physics,
		PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *highSpinMat);
	PxRigidDynamic* highSpinBox = smokeRigidProbeBox(physics, highSpinMat,
		PxVec3(0.0f, 7.0f, 0.0f), PxVec3(9.37f, 0.0f, -0.257f),
		PxVec3(2.0f));
	if(!highSpinGround || !highSpinBox)
	{
		std::printf("FAIL rigid matrix high-spin-landing actors\n");
		highSpinScene->release();
		highSpinMat->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	highSpinBox->setMass(640.0f);
	highSpinBox->setMassSpaceInertiaTensor(
		highSpinBox->getMassSpaceInertiaTensor() * 640.0f);
	highSpinBox->setAngularVelocity(PxVec3(0.0f, 24.0f, 0.0f));
	highSpinBox->setSolverIterationCounts(4, 1);
	highSpinScene->addActor(*highSpinGround);
	highSpinScene->addActor(*highSpinBox);
	PxQuat highSpinPrevious = highSpinBox->getGlobalPose().q;
	PxReal highSpinMaxPoseStep = 0.0f;
	PxReal highSpinV5 = 0.0f;
	PxReal highSpinW5 = 0.0f;
	bool highSpinFinite = true;
	for(PxU32 f = 0; f < 1200; ++f)
	{
		highSpinScene->simulate(1.0f / 60.0f);
		highSpinScene->fetchResults(true);
		const PxTransform pose = highSpinBox->getGlobalPose();
		highSpinMaxPoseStep = PxMax(highSpinMaxPoseStep,
			smokeQuatStepAngle(highSpinPrevious, pose.q));
		highSpinPrevious = pose.q;
		highSpinFinite = highSpinFinite && finitePose(pose) &&
			highSpinBox->getLinearVelocity().isFinite() &&
			highSpinBox->getAngularVelocity().isFinite();
		if(f == 299)
		{
			highSpinV5 = highSpinBox->getLinearVelocity().magnitude();
			highSpinW5 = highSpinBox->getAngularVelocity().magnitude();
		}
	}
	const PxReal highSpinFinalX = highSpinBox->getGlobalPose().p.x;
	const PxReal highSpinFinalV = highSpinBox->getLinearVelocity().magnitude();
	const PxReal highSpinFinalW = highSpinBox->getAngularVelocity().magnitude();
	if(!highSpinFinite || highSpinFinalV >= 0.02f || highSpinFinalW >= 0.02f ||
		highSpinV5 >= 6.0f || highSpinW5 >= 3.0f ||
		PxAbs(highSpinFinalX) >= 50.0f)
	{
		std::printf("FAIL rigid matrix high-spin-landing v5=%.9f w5=%.9f x20=%.9f finalV=%.9f finalW=%.9f maxPoseStep=%.9f\n",
			highSpinV5, highSpinW5, highSpinFinalX, highSpinFinalV,
			highSpinFinalW, highSpinMaxPoseStep);
		highSpinScene->release();
		highSpinMat->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS rigid matrix high-spin-landing v5=%.9f w5=%.9f x20=%.9f finalV=%.9f finalW=%.9f maxPoseStep=%.9f\n",
		highSpinV5, highSpinW5, highSpinFinalX, highSpinFinalV,
		highSpinFinalW, highSpinMaxPoseStep);
	highSpinScene->release();
	highSpinMat->release();

	PxMaterial* bounceMat = physics->createMaterial(0.0f, 0.0f, 0.8f);
	PxScene* bounceScene = smokeAvbdScene(physics, dispatcher, PxDefaultSimulationFilterShader,
		NULL, false);
	if(!bounceScene)
	{
		std::printf("FAIL create restitution scene\n");
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	bounceScene->setBounceThresholdVelocity(0.2f);
	PxRigidStatic* bounceGround = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *bounceMat);
	bounceScene->addActor(*bounceGround);
	PxRigidDynamic* ball = PxCreateDynamic(*physics, PxTransform(PxVec3(0.0f, 3.0f, 0.0f)),
		PxSphereGeometry(0.5f), *bounceMat, 1.0f);
	ball->setRigidDynamicLockFlags(PxRigidDynamicLockFlag::eLOCK_LINEAR_X
		| PxRigidDynamicLockFlag::eLOCK_LINEAR_Z
		| PxRigidDynamicLockFlag::eLOCK_ANGULAR_X
		| PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y
		| PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	bounceScene->addActor(*ball);
	ball->setLinearDamping(0.0f);
	ball->setAngularDamping(0.0f);
	bool bounceHit = false;
	PxReal bounceMin = PX_MAX_F32;
	PxReal bouncePeak = 0.0f;
	for(PxU32 f = 0; f < 120; ++f)
	{
		bounceScene->simulate(1.0f / 60.0f);
		bounceScene->fetchResults(true);
		const PxTransform bp = ball->getGlobalPose();
		if(!finitePose(bp))
		{
			std::printf("FAIL restitution NaN\n");
			bounceScene->release();
			return smokeAbort(NULL, dispatcher, physics, foundation);
		}
		const PxReal y = bp.p.y;
		if(y < 0.9f)
			bounceHit = true;
		if(bounceHit)
		{
			bounceMin = PxMin(bounceMin, y);
			bouncePeak = PxMax(bouncePeak, y);
		}
	}
	if(!bounceHit || bounceMin > 0.8f || bouncePeak < 1.2f)
	{
		std::printf("FAIL restitution e*v_in min=%.3f peak=%.3f hit=%d (want min<0.8 peak>1.2)\n",
			bounceMin, bouncePeak, int(bounceHit));
		bounceScene->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS restitution e*v_in min y=%.4f peak y=%.4f\n", bounceMin, bouncePeak);
	bounceScene->release();

	PxScene* noSolveScene = smokeAvbdScene(physics, dispatcher, detectOnlyFilter, NULL, false);
	if(!noSolveScene)
	{
		std::printf("FAIL create no-solve scene\n");
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	PxRigidStatic* noSolveGround = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *mat);
	noSolveScene->addActor(*noSolveGround);
	PxRigidDynamic* noSolveBox = PxCreateDynamic(*physics, PxTransform(PxVec3(0.0f, 2.0f, 0.0f)),
		PxBoxGeometry(0.5f, 0.5f, 0.5f), *mat, 1.0f);
	noSolveScene->addActor(*noSolveBox);
	for(PxU32 f = 0; f < 90; ++f)
	{
		noSolveScene->simulate(1.0f / 60.0f);
		noSolveScene->fetchResults(true);
	}
	const PxReal noSolveY = noSolveBox->getGlobalPose().p.y;
	if(!PxIsFinite(noSolveY) || noSolveY >= 0.0f)
	{
		std::printf("FAIL no-solve polarity y=%.3f (want fall-through y<0)\n", noSolveY);
		noSolveScene->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS no-solve polarity y=%.4f\n", noSolveY);
	noSolveScene->release();

	SmokeModifyCallback ignoreCb;
	ignoreCb.mode = eSMOKE_MODIFY_IGNORE;
	PxScene* ignoreScene = smokeAvbdScene(physics, dispatcher, modifyContactFilter, &ignoreCb, false);
	if(!ignoreScene)
	{
		std::printf("FAIL create ignore scene\n");
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	PxRigidStatic* ignoreGround = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *mat);
	ignoreScene->addActor(*ignoreGround);
	PxRigidDynamic* ignoreBox = PxCreateDynamic(*physics, PxTransform(PxVec3(0.0f, 2.0f, 0.0f)),
		PxBoxGeometry(0.5f, 0.5f, 0.5f), *mat, 1.0f);
	ignoreScene->addActor(*ignoreBox);
	for(PxU32 f = 0; f < 90; ++f)
	{
		ignoreScene->simulate(1.0f / 60.0f);
		ignoreScene->fetchResults(true);
	}
	const PxReal ignoreY = ignoreBox->getGlobalPose().p.y;
	if(!PxIsFinite(ignoreY) || ignoreY >= 0.0f)
	{
		std::printf("FAIL modify ignore/maxImpulse0 y=%.3f (want fall-through y<0)\n", ignoreY);
		ignoreScene->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS modify ignore/maxImpulse0 y=%.4f\n", ignoreY);
	ignoreScene->release();

	SmokeModifyCallback targetCb;
	targetCb.mode = eSMOKE_MODIFY_TARGET_VEL;
	PxScene* targetScene = smokeAvbdScene(physics, dispatcher, modifyContactFilter, &targetCb, false);
	if(!targetScene)
	{
		std::printf("FAIL create targetVelocity scene\n");
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	PxRigidStatic* targetGround = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *mat);
	targetScene->addActor(*targetGround);
	PxRigidDynamic* targetBox = PxCreateDynamic(*physics, PxTransform(PxVec3(0.0f, 4.0f, 0.0f)),
		PxBoxGeometry(0.5f, 0.5f, 0.5f), *mat, 1.0f);
	targetScene->addActor(*targetBox);
	for(PxU32 f = 0; f < 180; ++f)
	{
		targetScene->simulate(1.0f / 60.0f);
		targetScene->fetchResults(true);
	}
	const PxReal targetY = targetBox->getGlobalPose().p.y;
	if(!PxIsFinite(targetY) || targetY > 2.0f || targetY < 0.2f)
	{
		std::printf("FAIL unsupported contact-targetVelocity y=%.3f (must rest, not launch)\n",
			targetY);
		targetScene->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS unsupported contact-targetVelocity y=%.4f\n", targetY);
	targetScene->release();

	SmokeModifyCallback massCb;
	massCb.mode = eSMOKE_MODIFY_MASS_SCALE;
	PxScene* massScene = smokeAvbdScene(physics, dispatcher, modifyContactFilter, &massCb, false);
	if(!massScene)
	{
		std::printf("FAIL create mass-scale scene\n");
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	PxRigidStatic* massGround = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *mat);
	massScene->addActor(*massGround);
	PxRigidDynamic* massBox = PxCreateDynamic(*physics, PxTransform(PxVec3(0.0f, 4.0f, 0.0f)),
		PxBoxGeometry(0.5f, 0.5f, 0.5f), *mat, 1.0f);
	massScene->addActor(*massBox);
	for(PxU32 f = 0; f < 180; ++f)
	{
		massScene->simulate(1.0f / 60.0f);
		massScene->fetchResults(true);
	}
	const PxReal massY = massBox->getGlobalPose().p.y;
	if(!PxIsFinite(massY) || massY > 2.0f || massY < 0.2f)
	{
		std::printf("FAIL unsupported mass-inertia scale y=%.3f (must rest as finite mass)\n",
			massY);
		massScene->release();
		return smokeAbort(NULL, dispatcher, physics, foundation);
	}
	std::printf("PASS unsupported mass-inertia scale y=%.4f\n", massY);
	massScene->release();

	bounceMat->release();

	dispatcher->release();
	PxCloseExtensions();
	physics->release();
	foundation->release();
	return 0;
}
