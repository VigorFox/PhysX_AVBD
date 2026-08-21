// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Position-owned soft-contact force/Hessian evaluation and dual updates.
// This unit keeps the contact AL equations together without owning contact
// discovery, pair identity, or the component step lifecycle.

struct AvbdSoftContactRowForces
{
	PxReal normal;
	PxReal tangent[2];
	bool tangentClamped;

	AvbdSoftContactRowForces()
		: normal(0.0f), tangent{0.0f, 0.0f}, tangentClamped(false)
	{
	}
};

PX_FORCE_INLINE AvbdSoftContactRowForces
avbdEvaluateSoftContactRowForces(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	const PxVec3& currentSurfacePoint)
{
	AvbdSoftContactRowForces forces;
	const PxVec3 queryPoint =
		avbdGetSoftContactQueryPoint(geometry, particles);
	const PxVec3 n = geometry.normal;
	const PxReal normalConstraint =
		avbdEvaluateSoftContactNormalConstraint(
			geometry, queryPoint, currentSurfacePoint) -
		state.depenetrationConstraintOffset;
	forces.normal =
		PxMin(
			0.0f,
			state.k * normalConstraint +
				state.alLambda);

	if(geometry.tangentOwner !=
			AvbdSoftContactTangentOwner::ePOSITION_AL ||
		geometry.friction <= 0.0f || forces.normal >= 0.0f)
		return forces;

	const PxVec3 relativeDisplacement =
		(queryPoint - state.particlePointPrev) -
		(currentSurfacePoint - state.surfacePointPrev);
	const PxReal tangentConstraint[2] =
	{
		relativeDisplacement.dot(geometry.tangent1),
		relativeDisplacement.dot(geometry.tangent2)
	};
	forces.tangent[0] =
		state.penTangent[0] * tangentConstraint[0] +
			state.alLambdaTangent[0];
	forces.tangent[1] =
		state.penTangent[1] * tangentConstraint[1] +
			state.alLambdaTangent[1];
	const PxReal frictionBound =
		geometry.friction * PxAbs(forces.normal);
	const PxReal tangentMagnitude = PxSqrt(
		forces.tangent[0] * forces.tangent[0] +
		forces.tangent[1] * forces.tangent[1]);
	if(tangentMagnitude > frictionBound && tangentMagnitude > 1e-12f)
	{
		const PxReal scale = frictionBound / tangentMagnitude;
		forces.tangent[0] *= scale;
		forces.tangent[1] *= scale;
		forces.tangentClamped = true;
	}
	return forces;
}

PX_FORCE_INLINE void avbdEvaluateContactParticleBlockAtSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	const PxVec3& currentSurfacePoint,
	PxReal jacobianScale,
	PxVec3& outForce, PxMat33& outHessian)
{
	outForce = PxVec3(0.0f);
	outHessian = PxMat33(PxZero);
	if(PxAbs(jacobianScale) <= 1e-12f)
		return;

	const PxVec3 n = geometry.normal;
	const AvbdSoftContactRowForces rowForces =
		avbdEvaluateSoftContactRowForces(
			geometry, state, particles, currentSurfacePoint);
	outForce = n * (-jacobianScale * rowForces.normal);
	outHessian =
		avbdOuter(n, n) *
		(state.k * jacobianScale * jacobianScale);

	if(geometry.tangentOwner !=
			AvbdSoftContactTangentOwner::ePOSITION_AL ||
		geometry.friction <= 0.0f || rowForces.normal >= 0.0f)
		return;

	outForce -=
		(geometry.tangent1 * rowForces.tangent[0] +
		 geometry.tangent2 * rowForces.tangent[1]) * jacobianScale;
	// Once the trial tangent force is projected onto the Coulomb cone the
	// contact is sliding. Keeping the unprojected penalty Hessian here makes
	// Newton see a sticking spring even though the force itself is capped; a
	// dense edge manifold can then numerically pin two visibly sliding bodies.
	// Use a lagged Coulomb force for the sliding row. Inertia and material
	// curvature still regularize the particle block, while a row inside the
	// cone retains the full static-friction curvature below.
	if(!rowForces.tangentClamped)
	{
		outHessian = outHessian +
			avbdOuter(geometry.tangent1, geometry.tangent1) *
				(state.penTangent[0] *
				 jacobianScale * jacobianScale) +
			avbdOuter(geometry.tangent2, geometry.tangent2) *
				(state.penTangent[1] *
				 jacobianScale * jacobianScale);
	}
}
PX_FORCE_INLINE void avbdEvaluateContactParticleBlock(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	PxReal jacobianScale,
	PxVec3& outForce, PxMat33& outHessian)
{
	avbdEvaluateContactParticleBlockAtSurfacePoint(
		geometry, state, particles,
		avbdGetSoftContactSurfacePoint(geometry, particles),
		jacobianScale, outForce, outHessian);
}

PX_FORCE_INLINE void avbdEvaluateContactForceHessian(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	avbdEvaluateContactParticleBlock(
		geometry, state, particles, 1.0f,
		outForce, outHessian);
}

PX_FORCE_INLINE void avbdEvaluatePinForceHessian(
	const AvbdSoftPoint& point,
	const AvbdKinematicPin& kp,
	const AvbdSoftParticle* particles,
	PxU32 particleIndex,
	PxVec3& outForce, PxMat33& outHessian)
{
	const PxReal jacobianWeight =
		avbdGetSoftPointJacobianWeight(point, particleIndex);
	const PxVec3 C =
		avbdGetSoftPointPosition(point, particles) - kp.worldTarget;
	outForce = -(C * kp.k + kp.alLambda) * jacobianWeight;
	outHessian = PxMat33::createDiagonal(
		PxVec3(kp.k * jacobianWeight * jacobianWeight));
}

// =============================================================================
// AVBD Dual updates
// =============================================================================

PX_FORCE_INLINE void avbdWarmstartAttachmentState(
	AvbdSoftAttachment& attachment,
	PxReal alpha, PxReal gamma, PxReal penaltyMin)
{
	attachment.alLambda *= alpha * gamma;
	attachment.k = PxMax(
		penaltyMin,
		PxMin(attachment.kMax, attachment.k * gamma));
}

PX_FORCE_INLINE void avbdWarmstartPinState(
	AvbdKinematicPin& kp,
	PxReal alpha, PxReal gamma, PxReal penaltyMin)
{
	kp.alLambda *= alpha * gamma;
	kp.k = PxMax(penaltyMin, PxMin(kp.kMax, kp.k * gamma));
}

PX_FORCE_INLINE void avbdUpdatePinDual(
	AvbdKinematicPin& kp,
	const AvbdSoftPoint& point,
	const AvbdSoftParticle* particles,
	PxReal beta)
{
	const PxVec3 C =
		avbdGetSoftPointPosition(point, particles) - kp.worldTarget;
	kp.alLambda += C * kp.k;
	const PxReal C_lin = C.magnitude();
	kp.k = PxMin(kp.k + beta * C_lin, kp.kMax);
}

PX_FORCE_INLINE void avbdUpdateSoftPairAttachmentDual(
	AvbdSoftAttachment& attachment,
	const AvbdSoftPoint& point,
	const AvbdSoftPoint& targetPoint,
	const AvbdSoftParticle* particles,
	PxReal beta)
{
	const PxVec3 constraint =
		avbdGetSoftPointPosition(point, particles) -
		avbdGetSoftPointPosition(targetPoint, particles);
	attachment.alLambda += constraint * attachment.k;
	attachment.k = PxMin(
		attachment.k + beta * constraint.magnitude(),
		attachment.kMax);
}

PX_FORCE_INLINE void avbdUpdateSoftContactDualAtSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	const PxVec3& currentSurfacePoint,
	PxReal beta)
{
	PxVec3 n = geometry.normal;
	const PxVec3 queryPoint =
		avbdGetSoftContactQueryPoint(geometry, particles);
	const PxReal normalConstraint =
		avbdEvaluateSoftContactNormalConstraint(
			geometry, queryPoint, currentSurfacePoint) -
		state.depenetrationConstraintOffset;

	state.alLambda =
		PxMin(
			0.0f,
			state.k * normalConstraint +
				state.alLambda);
	if(state.alLambda < 0.0f)
		state.k = PxMin(
			state.k + beta * PxAbs(normalConstraint),
			state.ke);

	if(geometry.tangentOwner !=
		AvbdSoftContactTangentOwner::ePOSITION_AL)
	{
		avbdResetSoftContactTangentState(geometry, state, particles);
		return;
	}

	const PxVec3 relativeDisplacement =
		(queryPoint - state.particlePointPrev) -
		(currentSurfacePoint - state.surfacePointPrev);
	const PxReal tangentConstraint[2] =
	{
		relativeDisplacement.dot(geometry.tangent1),
		relativeDisplacement.dot(geometry.tangent2)
	};
	const PxReal frictionBound =
		geometry.friction * PxAbs(state.alLambda);
	PxReal tangentForce[2] =
	{
		state.penTangent[0] * tangentConstraint[0] +
			state.alLambdaTangent[0],
		state.penTangent[1] * tangentConstraint[1] +
			state.alLambdaTangent[1]
	};
	const PxReal rawTangentMagnitude = PxSqrt(
		tangentForce[0] * tangentForce[0] +
		tangentForce[1] * tangentForce[1]);
	const bool insideFrictionCone =
		rawTangentMagnitude <= frictionBound;
	if(!insideFrictionCone && rawTangentMagnitude > 1e-12f)
	{
		const PxReal scale = frictionBound / rawTangentMagnitude;
		tangentForce[0] *= scale;
		tangentForce[1] *= scale;
	}
	state.alLambdaTangent[0] = tangentForce[0];
	state.alLambdaTangent[1] = tangentForce[1];
	if(insideFrictionCone)
	{
		state.penTangent[0] = PxMin(
			state.penTangent[0] +
				beta * PxAbs(tangentConstraint[0]), state.ke);
		state.penTangent[1] = PxMin(
			state.penTangent[1] +
				beta * PxAbs(tangentConstraint[1]), state.ke);
	}
	state.frictionStick =
		insideFrictionCone &&
		tangentConstraint[0] * tangentConstraint[0] +
		tangentConstraint[1] * tangentConstraint[1] < 1e-10f;
}
