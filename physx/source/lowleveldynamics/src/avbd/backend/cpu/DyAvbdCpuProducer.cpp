#include "avbd/backend/cpu/DyAvbdCpuProducer.h"

namespace physx {
namespace Dy {

#if !defined(PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD)

namespace {

PX_FORCE_INLINE bool isNormalDynamicLaneEligible(
	const AvbdContactConstraint& contact, PxU32 ownerBody, PxU32 numBodies,
	PxF32& linearScale, PxF32& angularScale, bool& ownerIsA, PxU32& targetBody)
{
	const PxU32 bodyA = contact.header.bodyIndexA;
	const PxU32 bodyB = contact.header.bodyIndexB;
	ownerIsA = bodyA == ownerBody;
	const bool ownerIsB = bodyB == ownerBody;
	if(ownerIsA == ownerIsB)
		return false;
	targetBody = ownerIsA ? bodyB : bodyA;
	if(targetBody >= numBodies || ownerBody >= numBodies)
		return false;
	if((contact.header.flags & AvbdContactConstraintFlags::eDEFORMABLE_STATIC_ANCHOR) != 0 ||
		contact.friction > 0.0f || contact.staticFriction > 0.0f ||
		contact.targetVelocity.magnitudeSquared() > 1.0e-12f ||
		contact.C0 != 0.0f)
		return false;
	linearScale = ownerIsA ? contact.invMassScaleA : contact.invMassScaleB;
	angularScale = ownerIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
	return linearScale >= 0.0f && angularScale >= 0.0f &&
		(linearScale > 0.0f || angularScale > 0.0f);
}

PX_FORCE_INLINE void writeOwnerFields(
	AvbdRigidNormalContactSoA& soa, PxU32 row,
	const AvbdContactConstraint& contact, const AvbdSolverBody& owner,
	bool ownerIsA, PxF32 linearScale, PxF32 angularScale, PxF32 dt,
	PxF32 contactBoostFloor)
{
	const PxVec3 localPoint = ownerIsA ? contact.contactPointA : contact.contactPointB;
	soa.bodyPosition[0][row] = owner.position.x;
	soa.bodyPosition[1][row] = owner.position.y;
	soa.bodyPosition[2][row] = owner.position.z;
	soa.bodyRotation[0][row] = owner.rotation.x;
	soa.bodyRotation[1][row] = owner.rotation.y;
	soa.bodyRotation[2][row] = owner.rotation.z;
	soa.bodyRotation[3][row] = owner.rotation.w;
	soa.bodyContactPoint[0][row] = localPoint.x;
	soa.bodyContactPoint[1][row] = localPoint.y;
	soa.bodyContactPoint[2][row] = localPoint.z;
	soa.normal[0][row] = contact.contactNormal.x;
	soa.normal[1][row] = contact.contactNormal.y;
	soa.normal[2][row] = contact.contactNormal.z;
	soa.penetration[row] = contact.penetrationDepth;
	soa.penalty[row] = PxMax(contact.header.penalty, contactBoostFloor);
	soa.lambda[row] = contact.header.lambda;
	soa.maxImpulse[row] = contact.maxImpulse;
	soa.dt[row] = dt;
	soa.linearResponseScale[row] = linearScale;
	soa.angularResponseScale[row] = angularScale;
	soa.sign[row] = ownerIsA ? 1.0f : -1.0f;
}

} // namespace

bool avbdClassifyRigidNormalContactDynamicLane(
	const AvbdContactConstraint& contact, PxU32 ownerBody, PxU32 numBodies,
	PxF32& linearScale, PxF32& angularScale, bool& ownerIsA,
	PxU32& targetBody)
{
	return isNormalDynamicLaneEligible(contact, ownerBody, numBodies,
		linearScale, angularScale, ownerIsA, targetBody);
}

bool avbdBuildRigidNormalContactWave(
	const AvbdRigidNormalContactWaveBuildInput& input)
{
	AvbdRigidNormalContactWaveStorage* storage = input.storage;
	if(!storage || !input.bodies || !input.contacts || !input.contactMap ||
		!input.endpointKernel || input.numBodies == 0 ||
		input.numContacts == 0 || !(input.dt > 0.0f) ||
		!(input.invDt2 > 0.0f) || input.poseStamp == 0 ||
		!storage->ranges || !storage->dynamicTargets ||
		!storage->bodyRangeStarts || !storage->bodyRangeCounts ||
		!storage->soa.floatStorage || storage->soa.capacity < 8u ||
		storage->rangeCapacity == 0 ||
		storage->dynamicTargetCapacity < storage->rangeCapacity ||
		storage->bodyCapacity < input.numBodies ||
		!input.contactMap->constraintOffsets ||
		!input.contactMap->constraintCounts ||
		!input.contactMap->constraintIndices ||
		input.contactMap->numBodies < input.numBodies)
		return false;

	const PxU32 rowCapacity = storage->soa.capacity;
	storage->rangeCount = 0;
	storage->waveStamp = 0;
	storage->soa.numRows = rowCapacity;
	for(PxU32 body = 0; body < input.numBodies; ++body)
	{
		storage->bodyRangeStarts[body] = 0;
		storage->bodyRangeCounts[body] = 0;
	}

	PxU32 rowCursor = 0;
	PxU32 sourceLanes[eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];
	PxU8 sourceMask = 0;
	auto emitRange = [&](PxU32 ownerBody) -> bool
	{
		if(sourceMask == 0)
			return true;
		if(storage->rangeCount >= storage->rangeCapacity ||
			rowCursor > rowCapacity - 8u)
			return false;
		AvbdRigidNormalContactDynamicRangeInput8 rangeInput;
		rangeInput.bodies = input.bodies;
		rangeInput.numBodies = input.numBodies;
		rangeInput.contacts = input.contacts;
		rangeInput.numContacts = input.numContacts;
		rangeInput.sourceConstraints = sourceLanes;
		rangeInput.ownerBody = ownerBody;
		rangeInput.rowBase = rowCursor;
		rangeInput.dynamicTargetStorageIndex = storage->rangeCount;
		rangeInput.activeMask = sourceMask;
		rangeInput.dt = input.dt;
		rangeInput.contactBoostFloor =
			AvbdConstants::AVBD_CONTACT_BOOST_FRACTION *
			((1.0f / input.bodies[ownerBody].invMass) * input.invDt2);
		rangeInput.poseStamp = input.poseStamp;
		rangeInput.soa = &storage->soa;
		rangeInput.dynamicTargetStorage =
			&storage->dynamicTargets[storage->rangeCount];
		rangeInput.endpointKernel = input.endpointKernel;
		AvbdRigidNormalContactProducerRange8& range =
			storage->ranges[storage->rangeCount];
		const PxU8 dynamicMask =
			avbdPrepareRigidNormalContactDynamicRange8(rangeInput, range);
		if(dynamicMask != sourceMask || !range.validate())
			return false;
		++storage->rangeCount;
		++storage->bodyRangeCounts[ownerBody];
		rowCursor += 8u;
		sourceMask = 0;
		return true;
	};

	for(PxU32 ownerBody = 0; ownerBody < input.numBodies; ++ownerBody)
	{
		storage->bodyRangeStarts[ownerBody] = storage->rangeCount;
		if(input.bodies[ownerBody].invMass <= 0.0f)
			continue;

		const PxU32* mapIndices = nullptr;
		PxU32 mapCount = 0;
		input.contactMap->getBodyConstraints(ownerBody, mapIndices, mapCount);
		if(!mapIndices && mapCount != 0)
			return false;

		sourceMask = 0;
		for(PxU32 mapIndex = 0; mapIndex < mapCount; ++mapIndex)
		{
			const PxU32 source = mapIndices[mapIndex];
			if(source >= input.numContacts)
				return false;
			PxF32 linearScale = 0.0f, angularScale = 0.0f;
			bool ownerIsA = false;
			PxU32 targetBody = PX_MAX_U32;
			if(!isNormalDynamicLaneEligible(input.contacts[source], ownerBody,
				input.numBodies, linearScale, angularScale, ownerIsA,
				targetBody))
				continue;

			const PxU32 lane = [&]() {
				for(PxU32 candidate = 0;
					candidate < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH;
					++candidate)
					if((sourceMask & PxU8(1u << candidate)) == 0)
						return candidate;
				return eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH;
			}();
			if(lane >= eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH)
				return false;
			sourceLanes[lane] = source;
			sourceMask |= PxU8(1u << lane);

			if(sourceMask == PxU8(0xffu) && !emitRange(ownerBody))
				return false;
		}
		if(sourceMask != 0 && !emitRange(ownerBody))
			return false;
	}

	storage->soa.numRows = rowCursor;
	storage->waveStamp = input.poseStamp;
	return storage->validate(input.numBodies);
}

PxU8 avbdPrepareRigidNormalContactDynamicRange8(
	const AvbdRigidNormalContactDynamicRangeInput8& input,
	AvbdRigidNormalContactProducerRange8& range)
{
	range = AvbdRigidNormalContactProducerRange8();
	range.rowBase = input.rowBase;
	range.ownerBody = input.ownerBody;
	range.dynamicTargetStorageIndex = input.dynamicTargetStorageIndex;
	range.activeMask = input.activeMask;
	range.scalarFallbackMask = input.activeMask;
	if(!input.bodies || !input.contacts || !input.sourceConstraints ||
		!input.soa || !input.dynamicTargetStorage || !input.endpointKernel ||
		input.ownerBody >= input.numBodies ||
		input.rowBase > input.soa->numRows ||
		8u > input.soa->numRows - input.rowBase)
		return 0;

	for(PxU32 lane = 0;
		lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH; ++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((input.activeMask & bit) == 0)
			continue;
		const PxU32 source = input.sourceConstraints[lane];
		range.sourceConstraint[lane] = source;
		if(source >= input.numContacts)
			continue;
		const AvbdContactConstraint& contact = input.contacts[source];
		PxF32 linearScale = 0.0f, angularScale = 0.0f;
		bool ownerIsA = false;
		PxU32 targetBody = PX_MAX_U32;
		if(!isNormalDynamicLaneEligible(contact, input.ownerBody,
			input.numBodies, linearScale, angularScale, ownerIsA, targetBody))
			continue;

		const AvbdSolverBody& owner = input.bodies[input.ownerBody];
		const AvbdSolverBody& target = input.bodies[targetBody];
		const PxU32 row = input.rowBase + lane;
		writeOwnerFields(*input.soa, row, contact, owner, ownerIsA,
			linearScale, angularScale, input.dt, input.contactBoostFloor);
		input.dynamicTargetStorage->position[0][lane] = target.position.x;
		input.dynamicTargetStorage->position[1][lane] = target.position.y;
		input.dynamicTargetStorage->position[2][lane] = target.position.z;
		input.dynamicTargetStorage->rotation[0][lane] = target.rotation.x;
		input.dynamicTargetStorage->rotation[1][lane] = target.rotation.y;
		input.dynamicTargetStorage->rotation[2][lane] = target.rotation.z;
		input.dynamicTargetStorage->rotation[3][lane] = target.rotation.w;
		const PxVec3 targetPoint = ownerIsA ? contact.contactPointB : contact.contactPointA;
		input.dynamicTargetStorage->contactPoint[0][lane] = targetPoint.x;
		input.dynamicTargetStorage->contactPoint[1][lane] = targetPoint.y;
		input.dynamicTargetStorage->contactPoint[2][lane] = targetPoint.z;
		range.dynamicTargetBody[lane] = targetBody;
		range.targetPoseStamp[lane] = input.poseStamp;
		range.dynamicTargetMask |= bit;
		range.scalarFallbackMask &= PxU8(~bit);
	}

	if(range.dynamicTargetMask != 0)
	{
		const AvbdRigidNormalContactDynamicTarget8 view =
			input.dynamicTargetStorage->view(range.dynamicTargetMask);
		if(!view.validate(range.activeMask))
			return 0;
		input.endpointKernel(view,
			input.soa->staticContactPoint[0] + input.rowBase,
			input.soa->staticContactPoint[1] + input.rowBase,
			input.soa->staticContactPoint[2] + input.rowBase);
	}
	return range.dynamicTargetMask;
}

namespace {

PX_FORCE_INLINE bool isWideD6SlerpLaneEligible(
	const AvbdD6JointConstraint& joint, PxU32 ownerBody, PxU32 numBodies,
	PxU32& otherBody, bool& ownerIsA)
{
	if(ownerBody >= numBodies)
		return false;
	const PxU32 bodyA = joint.header.bodyIndexA;
	const PxU32 bodyB = joint.header.bodyIndexB;
	ownerIsA = bodyA == ownerBody;
	const bool ownerIsB = bodyB == ownerBody;
	if(ownerIsA == ownerIsB)
		return false;
	otherBody = ownerIsA ? bodyB : bodyA;
	if(otherBody >= numBodies || otherBody == ownerBody)
		return false;
	if(joint.getLinearMotion(0) != 0u || joint.getLinearMotion(1) != 0u ||
		joint.getLinearMotion(2) != 0u ||
		joint.getAngularMotion(0) != 2u ||
		joint.getAngularMotion(1) != 2u ||
		joint.getAngularMotion(2) != 2u)
		return false;
	if((joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) == 0 ||
		joint.driveFlags != (1u << 5) ||
		(joint.driveAccelerationFlags & (1u << 5)) != 0 ||
		joint.angularDamping.x != 0.0f ||
		joint.angularDamping.y != 0.0f ||
		!(joint.angularDamping.z > 0.0f) ||
		joint.angularStiffness.z != 0.0f)
		return false;
	if(joint.header.bodyIndexA >= numBodies ||
		joint.header.bodyIndexB >= numBodies)
		return false;
	return true;
}

PX_FORCE_INLINE PxVec3 wideD6AngularStep(const AvbdSolverBody& body)
{
	PxQuat dq = body.rotation * body.prevRotation.getConjugate();
	if(dq.w < 0.0f)
		dq = -dq;
	return PxVec3(dq.x, dq.y, dq.z) * 2.0f;
}

PX_FORCE_INLINE void clearWideD6RowMasks(
	AvbdRigidLocalResponsePacket8Input* rows, PxU8 activeMask)
{
	for(PxU32 row = 0;
		row < eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT; ++row)
	{
		rows[row].activeMask = activeMask;
		rows[row].touchingMask = activeMask;
		rows[row].padding[0] = rows[row].padding[1] = 0;
	}
}

} // namespace

static PxU8 avbdPrepareRigidD6WideRange8Impl(
	const AvbdRigidD6WideRangeInput8& input,
	AvbdRigidLocalSystemAoSoA8& target,
	AvbdRigidLocalResponsePacket8Input* rows,
	AvbdRigidD6ResponsePacket8View& view)
{
	if(!input.bodies || !input.joints || !input.ownerBodies ||
		!input.jointIndices || !rows || input.numBodies == 0 ||
		input.numJoints == 0 || !(input.dt > 0.0f) ||
		!(input.invDt2 > 0.0f))
		return 0;

	target.resetLanes(input.activeMask);
	PxU8 eligibleMask = 0;
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((input.activeMask & bit) == 0)
			continue;
		const PxU32 ownerBody = input.ownerBodies[lane];
		const PxU32 jointIndex = input.jointIndices[lane];
		if(ownerBody >= input.numBodies || jointIndex >= input.numJoints)
			continue;
		const AvbdD6JointConstraint& joint = input.joints[jointIndex];
		PxU32 otherBody = PX_MAX_U32;
		bool ownerIsA = false;
		if(!isWideD6SlerpLaneEligible(joint, ownerBody,
			input.numBodies, otherBody, ownerIsA))
			continue;
		if(otherBody >= input.numBodies)
			continue;
		const AvbdSolverBody& owner = input.bodies[ownerBody];
		const AvbdSolverBody& other = input.bodies[otherBody];
		if(!(owner.invMass > 0.0f) || !(other.invMass > 0.0f))
			continue;

		const PxF32 mass = 1.0f / owner.invMass;
		const PxF32 massInvDt2 = mass * input.invDt2;
		const PxVec3 rhsLinear =
			(owner.position - owner.inertialPosition) * massInvDt2;
		PxQuat deltaQ = owner.rotation * owner.inertialRotation.getConjugate();
		if(deltaQ.w < 0.0f)
			deltaQ = -deltaQ;
		const PxVec3 rotError = PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * 2.0f;
		const PxMat33 inertiaTensor = owner.invInertiaWorld.getInverse();
		const PxVec3 rhsAngular = (inertiaTensor * rotError) * input.invDt2;
		PxF32 angularInertia[3][3];
		for(PxU32 i = 0; i < 3; ++i)
			for(PxU32 j = 0; j < 3; ++j)
				angularInertia[i][j] = inertiaTensor(i, j) * input.invDt2;
		const PxF32 rhsLinearSoA[3] = { rhsLinear.x, rhsLinear.y, rhsLinear.z };
		const PxF32 rhsAngularSoA[3] = { rhsAngular.x, rhsAngular.y, rhsAngular.z };
		if(!target.seedInertialSoA(lane, massInvDt2, angularInertia,
			rhsLinearSoA, rhsAngularSoA, ownerBody, true))
			continue;

		const AvbdSolverBody& bodyA = input.bodies[joint.header.bodyIndexA];
		const AvbdSolverBody& bodyB = input.bodies[joint.header.bodyIndexB];
		const PxVec3 worldAnchorA = bodyA.position +
			bodyA.rotation.rotate(joint.anchorA);
		const PxVec3 worldAnchorB = bodyB.position +
			bodyB.rotation.rotate(joint.anchorB);
		const PxVec3 positionViolation = worldAnchorA - worldAnchorB;
		const PxVec3 rArm = owner.rotation.rotate(
			ownerIsA ? joint.anchorA : joint.anchorB);
		const PxF32 massA = 1.0f / bodyA.invMass;
		const PxF32 massB = 1.0f / bodyB.invMass;
		const PxF32 penalty = PxMax(joint.header.rho,
			PxMax(massA, massB) * input.invDt2);
		const PxF32 signJ = ownerIsA ? 1.0f : -1.0f;

		for(PxU32 axis = 0; axis < 3; ++axis)
		{
			const PxVec3 axisVector = axis == 0 ? PxVec3(1.0f, 0.0f, 0.0f) :
				(axis == 1 ? PxVec3(0.0f, 1.0f, 0.0f) :
				 PxVec3(0.0f, 0.0f, 1.0f));
			const PxVec3 gradPos = axisVector * signJ;
			const PxVec3 gradRot = rArm.cross(axisVector) * signJ;
			const PxF32 force = penalty * positionViolation.dot(axisVector) +
				joint.lambdaLinear[axis];
			AvbdRigidLocalResponsePacket8Input& row = rows[axis];
			for(PxU32 i = 0; i < 3; ++i)
			{
				row.gradPos[i][lane] = gradPos[i];
				row.gradRot[i][lane] = gradRot[i];
			}
			row.invCompliance[lane] = penalty;
			row.linearScale[lane] = 1.0f;
			row.angularScale[lane] = 1.0f;
			row.force[lane] = force;
		}

		const PxQuat frameA = bodyA.rotation * joint.localFrameA;
		PxQuat normalizedFrameA = frameA;
		const PxF32 frameMagnitudeSquared = normalizedFrameA.magnitudeSquared();
		if(frameMagnitudeSquared > 1e-8f && PxIsFinite(frameMagnitudeSquared))
			normalizedFrameA *= 1.0f / PxSqrt(frameMagnitudeSquared);
		const PxVec3 dThetaA = wideD6AngularStep(bodyA);
		const PxVec3 dThetaB = wideD6AngularStep(bodyB);
		const PxVec3 relativeAngularDisplacement = dThetaB - dThetaA;
		const PxVec3 worldAngularTarget = normalizedFrameA.rotate(
			joint.driveAngularVelocity) * input.dt;
		const PxF32 angularPenalty = joint.angularDamping.z / input.dt /
			input.dt;
		const PxF32 angularSign = ownerIsA ? -1.0f : 1.0f;
		for(PxU32 axis = 0; axis < 3; ++axis)
		{
			AvbdRigidLocalResponsePacket8Input& row = rows[3u + axis];
			for(PxU32 i = 0; i < 3; ++i)
			{
				row.gradPos[i][lane] = 0.0f;
				row.gradRot[i][lane] = i == axis ? 1.0f : 0.0f;
			}
			const PxF32 constraint = relativeAngularDisplacement[axis] -
				worldAngularTarget[axis];
			row.invCompliance[lane] = angularPenalty;
			row.linearScale[lane] = 0.0f;
			row.angularScale[lane] = 1.0f;
			row.force[lane] = angularSign *
				(angularPenalty * constraint + joint.lambdaDriveAngular[axis]);
		}
		eligibleMask |= bit;
	}

	clearWideD6RowMasks(rows, eligibleMask);
	view.bindRows(rows, eligibleMask, eligibleMask);
	return eligibleMask;
}

PxU8 avbdPrepareRigidD6WideRange8(
	const AvbdRigidD6WideRangeInput8& input,
	AvbdRigidLocalSystemAoSoA8& target,
	AvbdRigidLocalResponsePacket8Input* rows,
	AvbdRigidD6ResponsePacket8View& view)
{
	return avbdPrepareRigidD6WideRange8Impl(input, target, rows, view);
}

#endif // !PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD

namespace {

PX_FORCE_INLINE bool ownerMajorContactSupported(
	const AvbdContactConstraint& contact, PxU32 ownerBody, PxU32 numBodies,
	const AvbdSolverBody* bodies)
{
	const PxU32 bodyA = contact.header.bodyIndexA;
	const PxU32 bodyB = contact.header.bodyIndexB;
	if(bodyA >= numBodies || bodyB >= numBodies || bodyA == bodyB ||
		(ownerBody != bodyA && ownerBody != bodyB) ||
		bodies[bodyA].invMass <= 0.0f || bodies[bodyB].invMass <= 0.0f)
		return false;
	// The first owner-major slice is intentionally rigid dynamic-dynamic only.
	// Static/deformable rows and their specialized post-stage ownership stay
	// wholly scalar rather than being mixed into a packet lane.
	if(hasDeformableStaticAnchor(contact))
		return false;
	return true;
}

PX_FORCE_INLINE void seedOwnerMajorInertial(
	const AvbdSolverBody& body, PxF32 invDt2,
	AvbdRigidLocalSystemAoSoA8& target, PxU32 lane, PxU32 bodyIndex)
{
	const PxF32 mass = body.invMass > 1.0e-8f ? 1.0f / body.invMass : 0.0f;
	const PxF32 massInvDt2 = mass * invDt2;
	const PxF32 linearDiagonal = body.invMass > 0.0f ?
		(1.0f / body.invMass) * invDt2 : 0.0f;
	const PxVec3 rhsLinear =
		(body.position - body.inertialPosition) * massInvDt2;
	PxQuat deltaQ = body.rotation * body.inertialRotation.getConjugate();
	if(deltaQ.w < 0.0f)
		deltaQ = -deltaQ;
	const PxVec3 rotError(deltaQ.x * 2.0f, deltaQ.y * 2.0f,
		deltaQ.z * 2.0f);
	const PxMat33 inertiaTensor = body.invInertiaWorld.getInverse();
	const PxVec3 rhsAngular = (inertiaTensor * rotError) * invDt2;
	PxF32 angularDiagonal[3][3];
	for(PxU32 i = 0; i < 3; ++i)
		for(PxU32 j = 0; j < 3; ++j)
			angularDiagonal[i][j] = inertiaTensor(i, j) * invDt2;
	const PxF32 rhsLinearSoA[3] =
		{rhsLinear.x, rhsLinear.y, rhsLinear.z};
	const PxF32 rhsAngularSoA[3] =
		{rhsAngular.x, rhsAngular.y, rhsAngular.z};
	(void)target.seedInertialSoA(lane, linearDiagonal, angularDiagonal,
		rhsLinearSoA, rhsAngularSoA, bodyIndex, true);
}

PX_FORCE_INLINE bool accumulateOwnerMajorContact(
	const AvbdContactConstraint& contact, const AvbdSolverBody& body,
	const AvbdSolverBody& other, PxU32 bodyIndex, PxU32 numBodies,
	PxF32 dt, PxF32 contactBoostFloor, PxF32 avbdAlpha,
	AvbdRigidLocalSystemAoSoA8& target, PxU32 lane)
{
	const PxU32 bodyAIdx = contact.header.bodyIndexA;
	const PxU32 bodyBIdx = contact.header.bodyIndexB;
	if(bodyAIdx >= numBodies || bodyBIdx >= numBodies ||
		bodyAIdx == bodyBIdx ||
		(bodyAIdx != bodyIndex && bodyBIdx != bodyIndex))
		return false;
	const bool isBodyA = bodyAIdx == bodyIndex;
	const PxF32 linearScale = isBodyA ? contact.invMassScaleA
		: contact.invMassScaleB;
	const PxF32 angularScale = isBodyA ? contact.invInertiaScaleA
		: contact.invInertiaScaleB;
	if(linearScale <= 0.0f && angularScale <= 0.0f)
		return true;

	PxVec3 worldPosA, worldPosB, r;
	if(isBodyA)
	{
		r = body.rotation.rotate(contact.contactPointA);
		worldPosA = body.position + r;
		worldPosB = other.position + other.rotation.rotate(contact.contactPointB);
	}
	else
	{
		r = body.rotation.rotate(contact.contactPointB);
		worldPosA = other.position + other.rotation.rotate(contact.contactPointA);
		worldPosB = body.position + r;
	}
	const PxVec3& normal = contact.contactNormal;
	PxF32 violation = (worldPosA - worldPosB).dot(normal) +
		contact.penetrationDepth;
	violation -= avbdAlpha * contact.C0;
	const PxF32 pen = PxMax(contact.header.penalty, contactBoostFloor);
	const PxF32 lambda = contact.header.lambda;
	const PxF32 sign = isBodyA ? 1.0f : -1.0f;
	const PxVec3 gradPos = normal * sign;
	const PxVec3 gradRot = r.cross(normal) * sign;
	const PxF32 rawForce = PxMin(0.0f, pen * violation + lambda);
	PxF32 force = rawForce;
	bool forceSaturated = false;
	if(contact.maxImpulse < PX_MAX_REAL && dt > 0.0f)
	{
		const PxF32 maxNormalForce =
			PxMax(contact.maxImpulse, PxF32(0.0f)) / dt;
		force = PxMax(force, -maxNormalForce);
		forceSaturated = rawForce < -maxNormalForce;
	}
	const PxF32 normalRhsForce = force < 0.0f ? force : 0.0f;
	if(!forceSaturated)
	{
		if(!target.addResponseContribution(lane, gradPos, gradRot, pen,
			linearScale, angularScale, normalRhsForce, true))
			return false;
	}
	else if(!target.addRhsContribution(lane, gradPos, gradRot,
		linearScale, angularScale, normalRhsForce, true))
		return false;

	// The owner-major slice admits only dynamic-dynamic contacts, therefore the
	// scalar primal path owns both tangents whenever a friction material exists.
	if(contact.friction > 0.0f || contact.staticFriction > 0.0f)
	{
		PxVec3 prevWorldPosA, prevWorldPosB;
		if(isBodyA)
		{
			prevWorldPosA = body.prevPosition +
				body.prevRotation.rotate(contact.contactPointA);
			prevWorldPosB = other.prevPosition +
				other.prevRotation.rotate(contact.contactPointB);
		}
		else
		{
			prevWorldPosA = other.prevPosition +
				other.prevRotation.rotate(contact.contactPointA);
			prevWorldPosB = body.prevPosition +
				body.prevRotation.rotate(contact.contactPointB);
		}
		const PxVec3 relDisp = (worldPosA - prevWorldPosA) -
			(worldPosB - prevWorldPosB);
		const PxF32 tPen0 = PxMax(contact.tangentPenalty0,
			contactBoostFloor);
		const PxF32 tPen1 = PxMax(contact.tangentPenalty1,
			contactBoostFloor);
		const PxF32 tC0 = relDisp.dot(contact.tangent0);
		const PxF32 tC1 = relDisp.dot(contact.tangent1);
		const PxF32 mu = contactCoulombMu(contact);
		PxF32 Fn = 0.0f, Ft0 = 0.0f, Ft1 = 0.0f;
		(void)avbdEvaluateContactForcesCone(
			pen, violation, lambda, tPen0, tC0, contact.tangentLambda0,
			tPen1, tC1, contact.tangentLambda1, mu, Fn, Ft0, Ft1);
		const PxVec3 tangents[2] = {contact.tangent0, contact.tangent1};
		const PxF32 penalties[2] = {tPen0, tPen1};
		const PxF32 forces[2] = {Ft0, Ft1};
		for(PxU32 tangent = 0; tangent < 2u; ++tangent)
		{
			const PxVec3 tangentGradPos = tangents[tangent] * sign;
			const PxVec3 tangentGradRot = r.cross(tangents[tangent]) * sign;
			if(!target.addResponseContribution(lane, tangentGradPos,
				tangentGradRot, penalties[tangent], linearScale,
				angularScale, forces[tangent], true))
				return false;
		}
	}
	return true;
}

#if !defined(PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD)

// Rejected AVX2+FMA candidate assembly, retained only for standalone probes.
// Production keeps the scalar owner-major producer used by the GPU bridge.
static PxU8 prepareOwnerMajorContactBlock(
	const AvbdRigidOwnerMajorWaveInput8& input,
	AvbdRigidLocalSystemAoSoA8& target)
{
	constexpr PxU32 kMaxRows = 16u;
	const PxU8 validMask = PxU8(
		(1u << eAVBD_RIGID_LDLT_PACKET_WIDTH) - 1u);
	const PxU8 activeMask = PxU8(input.activeMask & validMask);
	if(!input.contactBlockKernel || activeMask == 0)
		return 0;

	PxU32 rowIndices[eAVBD_RIGID_LDLT_PACKET_WIDTH][kMaxRows] = {};
	PxU32 rowCounts[eAVBD_RIGID_LDLT_PACKET_WIDTH] = {};
	PxU32 maxRows = 0;
	PxU8 eligibleMask = activeMask;
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((activeMask & bit) == 0)
			continue;
		const PxU32 ownerBody = input.ownerBodies[lane];
		if(ownerBody >= input.numBodies ||
			input.bodies[ownerBody].invMass <= 0.0f)
		{
			eligibleMask &= PxU8(~bit);
			continue;
		}
		const PxU32* mapIndices = nullptr;
		PxU32 mapCount = 0;
		if(!input.contactMap || input.contactMap->numBodies <= ownerBody)
		{
			eligibleMask &= PxU8(~bit);
			continue;
		}
		input.contactMap->getBodyConstraints(ownerBody, mapIndices, mapCount);
		if(!mapIndices || mapCount > kMaxRows)
		{
			eligibleMask &= PxU8(~bit);
			continue;
		}
		bool laneSupported = true;
		for(PxU32 mapIndex = 0; mapIndex < mapCount; ++mapIndex)
		{
			const PxU32 contactIndex = mapIndices[mapIndex];
			if(contactIndex >= input.numContacts)
				return 0;
			const AvbdContactConstraint& contact = input.contacts[contactIndex];
			if(contact.header.bodyIndexA != ownerBody &&
				contact.header.bodyIndexB != ownerBody)
				continue;
			if(!ownerMajorContactSupported(contact, ownerBody,
				input.numBodies, input.bodies))
			{
				laneSupported = false;
				break;
			}
			const PxU32 row = rowCounts[lane]++;
			if(row >= kMaxRows)
			{
				laneSupported = false;
				break;
			}
			rowIndices[lane][row] = contactIndex;
		}
		if(!laneSupported || rowCounts[lane] == 0)
		{
			eligibleMask &= PxU8(~bit);
			rowCounts[lane] = 0;
			continue;
		}
		maxRows = PxMax(maxRows, rowCounts[lane]);
	}
	if(eligibleMask == 0)
		return 0;
	PxU32 eligibleLaneCount = 0;
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
		eligibleLaneCount += (eligibleMask & PxU8(1u << lane)) != 0 ? 1u : 0u;
	// A sparse packet cannot amortize the field-major assembly and packet
	// factor call over its scalar fallbacks.  Let the authoritative scalar
	// producer handle the whole packet when fewer than half the lanes qualify;
	// dense packets retain the partial-lane fallback contract below.
	if(eligibleLaneCount < eAVBD_RIGID_LDLT_PACKET_WIDTH / 2u)
		return 0;

	target.resetLanes(target.activeMask);
	target.activeMask = 0;
	target.touchingMask = 0;
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((eligibleMask & bit) == 0)
			continue;
		seedOwnerMajorInertial(input.bodies[input.ownerBodies[lane]],
			input.invDt2, target, lane, input.ownerBodies[lane]);
		target.activeMask |= bit;
	}

	for(PxU32 row = 0; row < maxRows; ++row)
	{
		AvbdRigidContactBlockPacket8Input block = {};
		PxU8 rowMask = 0;
		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
		{
			const PxU8 bit = PxU8(1u << lane);
			if((eligibleMask & bit) != 0 && row < rowCounts[lane])
				rowMask |= bit;
		}
		if(rowMask == 0)
			continue;
		block.activeMask = rowMask;
		block.touchingMask = rowMask;
		block.hessianMask[0] = rowMask;
		block.hessianMask[1] = rowMask;
		block.hessianMask[2] = rowMask;

		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
		{
			const PxU8 bit = PxU8(1u << lane);
			if((rowMask & bit) == 0)
				continue;
			const PxU32 ownerBody = input.ownerBodies[lane];
			const AvbdContactConstraint& contact = input.contacts[
				rowIndices[lane][row]];
			const bool isBodyA = contact.header.bodyIndexA == ownerBody;
			const PxU32 otherBody = isBodyA ? contact.header.bodyIndexB
				: contact.header.bodyIndexA;
			const AvbdSolverBody& body = input.bodies[ownerBody];
			const AvbdSolverBody& other = input.bodies[otherBody];
			const PxF32 linearScale = isBodyA ? contact.invMassScaleA
				: contact.invMassScaleB;
			const PxF32 angularScale = isBodyA ? contact.invInertiaScaleA
				: contact.invInertiaScaleB;
			block.linearScale[lane] = linearScale;
			block.angularScale[lane] = angularScale;
			if(linearScale <= 0.0f && angularScale <= 0.0f)
			{
				block.touchingMask &= PxU8(~bit);
				block.hessianMask[0] &= PxU8(~bit);
				block.hessianMask[1] &= PxU8(~bit);
				block.hessianMask[2] &= PxU8(~bit);
				continue;
			}

			const PxVec3 r = isBodyA ?
				body.rotation.rotate(contact.contactPointA) :
				body.rotation.rotate(contact.contactPointB);
			const PxVec3 worldA = isBodyA ? body.position + r : other.position +
				other.rotation.rotate(contact.contactPointA);
			const PxVec3 worldB = isBodyA ? other.position +
				other.rotation.rotate(contact.contactPointB) : body.position + r;
			const PxF32 mass = 1.0f / body.invMass;
			const PxF32 floor = AvbdConstants::AVBD_CONTACT_BOOST_FRACTION *
				mass * input.invDt2;
			const PxF32 penalty = PxMax(contact.header.penalty, floor);
			const PxF32 sign = isBodyA ? 1.0f : -1.0f;
			const PxVec3 normalGradPos = contact.contactNormal * sign;
			const PxVec3 normalGradRot = r.cross(contact.contactNormal) * sign;
			const PxF32 violation = (worldA - worldB).dot(contact.contactNormal) +
				contact.penetrationDepth - input.avbdAlpha * contact.C0;
			const PxF32 rawForce = PxMin(0.0f,
				penalty * violation + contact.header.lambda);
			PxF32 force = rawForce;
			bool saturated = false;
			if(contact.maxImpulse < PX_MAX_F32)
			{
				const PxF32 maxForce = PxMax(contact.maxImpulse, PxF32(0.0f)) /
					input.dt;
				force = PxMax(force, -maxForce);
				saturated = rawForce < -maxForce;
			}
			const PxF32 rhsForce = force < 0.0f ? force : 0.0f;
			for(PxU32 component = 0; component < 3u; ++component)
			{
				block.gradPos[0][component][lane] = normalGradPos[component];
				block.gradRot[0][component][lane] = normalGradRot[component];
			}
			block.invCompliance[0][lane] = penalty;
			block.force[0][lane] = rhsForce;
			if(saturated)
				block.hessianMask[0] &= PxU8(~bit);

			const PxVec3 prevA = isBodyA ? body.prevPosition +
				body.prevRotation.rotate(contact.contactPointA) : other.prevPosition +
				other.prevRotation.rotate(contact.contactPointA);
			const PxVec3 prevB = isBodyA ? other.prevPosition +
				other.prevRotation.rotate(contact.contactPointB) : body.prevPosition +
				body.prevRotation.rotate(contact.contactPointB);
			const PxVec3 rel = (worldA - prevA) - (worldB - prevB);
			const PxF32 pen0 = PxMax(contact.tangentPenalty0, floor);
			const PxF32 pen1 = PxMax(contact.tangentPenalty1, floor);
			PxF32 normalForce = 0.0f, tangent0 = 0.0f, tangent1 = 0.0f;
			if(contact.friction > 0.0f || contact.staticFriction > 0.0f)
				(void)avbdEvaluateContactForcesCone(penalty, violation,
					contact.header.lambda, pen0, rel.dot(contact.tangent0),
					contact.tangentLambda0, pen1, rel.dot(contact.tangent1),
					contact.tangentLambda1, contactCoulombMu(contact), normalForce,
					tangent0, tangent1);
			const PxVec3 tangents[2] = {contact.tangent0, contact.tangent1};
			const PxF32 penalties[2] = {pen0, pen1};
			const PxF32 forces[2] = {tangent0, tangent1};
			if(contact.friction <= 0.0f && contact.staticFriction <= 0.0f)
			{
				block.hessianMask[1] &= PxU8(~bit);
				block.hessianMask[2] &= PxU8(~bit);
			}
			for(PxU32 tangent = 0; tangent < 2u; ++tangent)
			{
				const PxVec3 gradPos = tangents[tangent] * sign;
				const PxVec3 gradRot = r.cross(tangents[tangent]) * sign;
				for(PxU32 component = 0; component < 3u; ++component)
				{
					block.gradPos[tangent + 1u][component][lane] =
						gradPos[component];
					block.gradRot[tangent + 1u][component][lane] =
						gradRot[component];
				}
				block.invCompliance[tangent + 1u][lane] = penalties[tangent];
				block.force[tangent + 1u][lane] = forces[tangent];
			}
		}
		input.contactBlockKernel(block, target.factorInput,
			target.touchingMask);
	}
	target.touchingMask = PxU8(target.touchingMask & eligibleMask);
	return eligibleMask;
}

#endif // !PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD

} // namespace

PxU8 avbdPrepareRigidOwnerMajorWave8(
	const AvbdRigidOwnerMajorWaveInput8& input,
	AvbdRigidLocalSystemAoSoA8& target)
{
	const PxU8 validMask = PxU8(
		(1u << eAVBD_RIGID_LDLT_PACKET_WIDTH) - 1u);
	if(!input.bodies || !input.contacts || !input.ownerBodies ||
		input.numBodies == 0 || input.numContacts == 0 ||
		!(input.dt > 0.0f) || !(input.invDt2 > 0.0f) ||
		(input.activeMask & ~validMask) != 0)
		return 0;
#if !defined(PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD)
	if(input.contactBlockKernel)
	{
		const PxU8 blockMask = prepareOwnerMajorContactBlock(input, target);
		if(blockMask != 0)
			return blockMask;
	}
#endif
	// A freshly constructed packet is already zeroed by its constructor.  On
	// reuse, clear only lanes that were active in the previous packet; the
	// producer overwrites every field of accepted lanes below and inactive
	// lanes are never exported.  Clearing the new active mask unconditionally
	// duplicated the full lane/matrix wipe on every eight-owner packet.
	const PxU8 previousActiveMask = target.activeMask;
	if(previousActiveMask != 0)
		target.resetLanes(previousActiveMask);
	PxU8 acceptedMask = 0;
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((input.activeMask & bit) == 0)
			continue;
		const PxU32 ownerBody = input.ownerBodies[lane];
		if(ownerBody >= input.numBodies ||
			input.bodies[ownerBody].invMass <= 0.0f)
			continue;
		bool duplicate = false;
		for(PxU32 prior = 0; prior < lane; ++prior)
			if((input.activeMask & PxU8(1u << prior)) != 0 &&
				input.ownerBodies[prior] == ownerBody)
				duplicate = true;
		if(duplicate)
			continue;

		seedOwnerMajorInertial(input.bodies[ownerBody], input.invDt2,
			target, lane, ownerBody);
		target.activeMask |= bit;
		const PxF32 mass = input.bodies[ownerBody].invMass > 1.0e-8f ?
			1.0f / input.bodies[ownerBody].invMass : 0.0f;
		const PxF32 contactBoostFloor =
			AvbdConstants::AVBD_CONTACT_BOOST_FRACTION * mass * input.invDt2;
		const PxU32* mapIndices = nullptr;
		PxU32 mapCount = 0;
		if(input.contactMap && input.contactMap->numBodies > ownerBody)
			input.contactMap->getBodyConstraints(ownerBody, mapIndices, mapCount);
		const PxU32 loopCount = mapIndices ? mapCount : input.numContacts;
		bool complete = true;
		for(PxU32 loop = 0; loop < loopCount; ++loop)
		{
			const PxU32 contactIndex = mapIndices ? mapIndices[loop] : loop;
			if(contactIndex >= input.numContacts)
			{
				complete = false;
				break;
			}
			const AvbdContactConstraint& contact =
				input.contacts[contactIndex];
			if(contact.header.bodyIndexA != ownerBody &&
				contact.header.bodyIndexB != ownerBody)
				continue;
			if(!ownerMajorContactSupported(contact, ownerBody,
				input.numBodies, input.bodies))
			{
				complete = false;
				break;
			}
			const PxU32 otherBody = contact.header.bodyIndexA == ownerBody ?
				contact.header.bodyIndexB : contact.header.bodyIndexA;
			if(!accumulateOwnerMajorContact(contact, input.bodies[ownerBody],
				input.bodies[otherBody], ownerBody, input.numBodies, input.dt,
				contactBoostFloor, input.avbdAlpha, target, lane))
			{
				complete = false;
				break;
			}
		}
		if(!complete)
		{
			target.resetLanes(bit);
			continue;
		}
		acceptedMask |= bit;
	}
	target.activeMask = PxU8(target.activeMask & acceptedMask);
	target.touchingMask = PxU8(target.touchingMask & acceptedMask);
	return acceptedMask;
}

} // namespace Dy
} // namespace physx
