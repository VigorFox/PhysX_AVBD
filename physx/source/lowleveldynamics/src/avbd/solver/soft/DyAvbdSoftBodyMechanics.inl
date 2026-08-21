// AVBD soft-body mechanics and incidence primitives.
//
// This unit owns topology incidence records, constitutive force/Hessian
// helpers, determinant/bending evaluation, and primal displacement limits.
// It contains numerical mechanics only; lifecycle and scheduling stay outside.
// =============================================================================

// =============================================================================
// VBD Force/Hessian evaluators
// =============================================================================

PX_FORCE_INLINE void avbdEvaluateStVKForceHessian(
	const AvbdTriElement& tri, int vOrder,
	PxReal mu, PxReal lam,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	PxVec3 x0 = particles[tri.p0].position;
	PxVec3 x01 = particles[tri.p1].position - x0;
	PxVec3 x02 = particles[tri.p2].position - x0;

	PxReal D00 = tri.DmInv00, D01 = tri.DmInv01;
	PxReal D10 = tri.DmInv10, D11 = tri.DmInv11;

	PxVec3 f0 = x01 * D00 + x02 * D10;
	PxVec3 f1 = x01 * D01 + x02 * D11;

	PxReal f0f0 = f0.dot(f0);
	PxReal f1f1 = f1.dot(f1);
	PxReal f0f1 = f0.dot(f1);

	PxReal G00 = 0.5f * (f0f0 - 1.0f);
	PxReal G11 = 0.5f * (f1f1 - 1.0f);
	PxReal G01 = 0.5f * f0f1;

	PxReal Gfro2 = G00 * G00 + G11 * G11 + 2.0f * G01 * G01;
	if (Gfro2 < 1e-20f)
	{
		outForce = PxVec3(0.0f);
		outHessian = PxMat33(PxZero);
		return;
	}

	PxReal trG = G00 + G11;
	PxReal ltrG = lam * trG;
	PxReal twoMu = 2.0f * mu;
	PxVec3 PK1_0 = f0 * (twoMu * G00 + ltrG) + f1 * (twoMu * G01);
	PxVec3 PK1_1 = f0 * (twoMu * G01) + f1 * (twoMu * G11 + ltrG);

	PxReal df0, df1;
	if (vOrder == 0)      { df0 = -D00 - D10; df1 = -D01 - D11; }
	else if (vOrder == 1) { df0 = D00; df1 = D01; }
	else                  { df0 = D10; df1 = D11; }

	outForce = (PK1_0 * df0 + PK1_1 * df1) * (-tri.restArea);

	PxReal df0sq = df0 * df0;
	PxReal df1sq = df1 * df1;
	PxReal df0df1 = df0 * df1;

	PxReal Ic = f0f0 + f1f1;
	PxReal two_dpsi_dIc = -mu + (0.5f * Ic - 1.0f) * lam;
	PxMat33 I33 = PxMat33(PxIdentity);

	PxMat33 f0f0m = avbdOuter(f0, f0);
	PxMat33 f1f1m = avbdOuter(f1, f1);
	PxMat33 f0f1m = avbdOuter(f0, f1);
	PxMat33 f1f0m = avbdOuter(f1, f0);

	PxMat33 H00 = f0f0m * lam + I33 * two_dpsi_dIc
	            + (I33 * f0f0 + f0f0m * 2.0f + f1f1m) * mu;
	PxMat33 H01 = f0f1m * lam + (I33 * f0f1 + f1f0m) * mu;
	PxMat33 H11 = f1f1m * lam + I33 * two_dpsi_dIc
	            + (I33 * f1f1 + f1f1m * 2.0f + f0f0m) * mu;

	PxReal area = tri.restArea;
	outHessian = H00 * (df0sq * area) + H11 * (df1sq * area)
	           + (H01 + H01.getTranspose()) * (df0df1 * area);
}

PX_FORCE_INLINE void avbdEvaluateTetDeterminantAndGradient(
	const AvbdTetElement& tet, PxU32 vertexOrder,
	const PxVec3& e1, const PxVec3& e2, const PxVec3& e3,
	PxReal& outDeterminant, PxVec3& outGradient)
{
	PxVec3 currentFaceGradient;
	PxReal currentDeterminant;
	switch(vertexOrder)
	{
	case 0:
		currentFaceGradient = (e3 - e1).cross(e2 - e1);
		currentDeterminant = (-e1).dot(currentFaceGradient);
		break;
	case 1:
		currentFaceGradient = e2.cross(e3);
		currentDeterminant = e1.dot(currentFaceGradient);
		break;
	case 2:
		currentFaceGradient = e3.cross(e1);
		currentDeterminant = e2.dot(currentFaceGradient);
		break;
	default:
		currentFaceGradient = e1.cross(e2);
		currentDeterminant = e3.dot(currentFaceGradient);
		break;
	}
	outDeterminant =
		currentDeterminant * tet.inverseRestDeterminant;
	outGradient =
		currentFaceGradient * tet.inverseRestDeterminant;
}

PX_FORCE_INLINE bool avbdIsFiniteVector(const PxVec3& value)
{
	return PxIsFinite(value.x) && PxIsFinite(value.y) &&
		PxIsFinite(value.z);
}

// The polar iteration already computes and validates the determinant before
// inversion.  PxMat33::getInverse() would recompute that same determinant;
// consume the validated value while preserving its scalar cofactor algebra.
PX_FORCE_INLINE PxMat33 avbdGetInverseTransposeWithDeterminant(
	const PxMat33& matrix, PxReal determinant)
{
	const PxReal invDet = 1.0f / determinant;
	return PxMat33(
		PxVec3(
			invDet * (matrix.column1.y * matrix.column2.z -
				matrix.column2.y * matrix.column1.z),
			invDet * -(matrix.column1.x * matrix.column2.z -
				matrix.column1.z * matrix.column2.x),
			invDet * (matrix.column1.x * matrix.column2.y -
				matrix.column1.y * matrix.column2.x)),
		PxVec3(
			invDet * -(matrix.column0.y * matrix.column2.z -
				matrix.column2.y * matrix.column0.z),
			invDet * (matrix.column0.x * matrix.column2.z -
				matrix.column0.z * matrix.column2.x),
			invDet * -(matrix.column0.x * matrix.column2.y -
				matrix.column0.y * matrix.column2.x)),
		PxVec3(
			invDet * (matrix.column0.y * matrix.column1.z -
				matrix.column0.z * matrix.column1.y),
			invDet * -(matrix.column0.x * matrix.column1.z -
				matrix.column0.z * matrix.column1.x),
			invDet * (matrix.column0.x * matrix.column1.y -
				matrix.column1.x * matrix.column0.y)));
}

PX_FORCE_INLINE PxMat33 avbdExtractCorotationalRotation(
	const PxMat33& deformationGradient)
{
	PxMat33 rotation = deformationGradient;
	PxReal determinant = rotation.getDeterminant();
	if(!PxIsFinite(determinant) || PxAbs(determinant) <= 1.0e-9f)
		rotation = PxMat33(PxIdentity);
	else
	{
		for(PxU32 iteration = 0; iteration < 5; iteration++)
		{
			if(!PxIsFinite(determinant) ||
				PxAbs(determinant) <= 1.0e-9f)
				break;
			const PxMat33 inverseTranspose =
				avbdGetInverseTransposeWithDeterminant(
					rotation, determinant);
			if(!avbdIsFiniteVector(inverseTranspose.column0) ||
				!avbdIsFiniteVector(inverseTranspose.column1) ||
				!avbdIsFiniteVector(inverseTranspose.column2))
				break;
			rotation.column0 =
				(rotation.column0 + inverseTranspose.column0) * 0.5f;
			rotation.column1 =
				(rotation.column1 + inverseTranspose.column1) * 0.5f;
			rotation.column2 =
				(rotation.column2 + inverseTranspose.column2) * 0.5f;
			if(iteration + 1 < 5)
				determinant = rotation.getDeterminant();
		}
	}

	// Finish with an explicitly right-handed orthonormal basis.  The polar
	// iteration alone can retain a reflection for inverted configurations;
	// co-rotational elasticity requires the closest proper rotation.
	PxVec3 column0 = rotation.column0;
	if(!avbdIsFiniteVector(column0) ||
		column0.magnitudeSquared() <= 1.0e-12f)
		column0 = deformationGradient.column0;
	if(!avbdIsFiniteVector(column0) ||
		column0.magnitudeSquared() <= 1.0e-12f)
		column0 = PxVec3(1.0f, 0.0f, 0.0f);
	column0.normalize();

	PxVec3 column1 =
		rotation.column1 -
		column0 * rotation.column1.dot(column0);
	if(!avbdIsFiniteVector(column1) ||
		column1.magnitudeSquared() <= 1.0e-12f)
	{
		const PxVec3 reference =
			PxAbs(column0.x) < 0.8f
				? PxVec3(1.0f, 0.0f, 0.0f)
				: PxVec3(0.0f, 1.0f, 0.0f);
		column1 =
			reference - column0 * reference.dot(column0);
	}
	column1.normalize();
	PxVec3 column2 = column0.cross(column1);
	if(column2.dot(rotation.column2) < 0.0f)
	{
		column1 = -column1;
		column2 = -column2;
	}
	return PxMat33(column0, column1, column2);
}

PX_FORCE_INLINE PxReal avbdComputeTetStressCoefficient(
	const AvbdTetElement& tet,
	const AvbdSoftParticle* particles)
{
	const PxVec3 p0 = particles[tet.p0].position;
	const PxMat33 deformationGradient(
		particles[tet.p1].position - p0,
		particles[tet.p2].position - p0,
		particles[tet.p3].position - p0);
	const PxMat33 F = deformationGradient * tet.DmInv;
	const PxMat33 rotation =
		avbdExtractCorotationalRotation(F);
	const PxMat33 coRotatedF =
		rotation.getTranspose() * F;
	const PxMat33 strain =
		(coRotatedF + coRotatedF.getTranspose()) * 0.5f -
		PxMat33(PxIdentity);
	const PxReal q0 = strain.column0.x;
	const PxReal q1 = strain.column1.y;
	const PxReal q2 = strain.column2.z;
	const PxReal q01 = q0 - q1;
	const PxReal q12 = q1 - q2;
	const PxReal q20 = q2 - q0;
	const PxReal coefficient = PxSqrt(
		q01 * q01 + q12 * q12 + q20 * q20) *
		0.7071067811865475244f;
	return PxIsFinite(coefficient)
		? coefficient : PX_MAX_F32;
}

PX_FORCE_INLINE void
avbdEvaluateCorotationalForceHessianPrepared(
	const AvbdTetElement& tet, int vOrder,
	PxReal mu, PxReal lam,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian,
	AvbdTetVertexLinearization* outLinearization = NULL)
{
	const PxVec3 p0 = particles[tet.p0].position;
	const PxVec3 e1 = particles[tet.p1].position - p0;
	const PxVec3 e2 = particles[tet.p2].position - p0;
	const PxVec3 e3 = particles[tet.p3].position - p0;
	const PxU32 vertexOrder =
		vOrder >= 0 && vOrder < 3 ? PxU32(vOrder) : 3;
	if(outLinearization)
	{
		avbdEvaluateTetDeterminantAndGradient(
			tet, vertexOrder, e1, e2, e3,
			outLinearization->determinant,
			outLinearization->determinantGradient);
	}

	const PxMat33 deformationGradient =
		PxMat33(e1, e2, e3) * tet.DmInv;
	const PxMat33 rotation =
		avbdExtractCorotationalRotation(deformationGradient);
	const PxReal strainTrace =
		rotation.column0.dot(deformationGradient.column0) +
		rotation.column1.dot(deformationGradient.column1) +
		rotation.column2.dot(deformationGradient.column2) -
		3.0f;
	const PxMat33 firstPiola =
		(deformationGradient - rotation) * (2.0f * mu) +
		rotation * (lam * strainTrace);
	const PxVec3& shapeGradient =
		tet.shapeGradients[vertexOrder];
	outForce =
		(firstPiola * shapeGradient) * (-tet.restVolume);

	// A frozen-rotation Gauss-Newton block is symmetric positive
	// semi-definite and gives the exact local stiffness for the linearized
	// co-rotational energy.
	const PxVec3 rotatedGradient =
		rotation * shapeGradient;
	const PxReal gradientNormSq =
		tet.shapeGradientNormSq[vertexOrder];
	outHessian =
		PxMat33::createDiagonal(
			PxVec3(2.0f * mu * gradientNormSq *
				tet.restVolume)) +
		avbdOuter(rotatedGradient, rotatedGradient) *
			(lam * tet.restVolume);
}

PX_FORCE_INLINE void avbdEvaluateNeoHookeanForceHessianPrepared(
	const AvbdTetElement& tet, int vOrder,
	PxReal mu, PxReal lam, PxReal alpha,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian,
	AvbdTetVertexLinearization* outLinearization = NULL)
{
	PxVec3 p0 = particles[tet.p0].position;
	PxVec3 e1 = particles[tet.p1].position - p0;
	PxVec3 e2 = particles[tet.p2].position - p0;
	PxVec3 e3 = particles[tet.p3].position - p0;

	const PxU32 vertexOrder =
		vOrder >= 0 && vOrder < 3 ? PxU32(vOrder) : 3;
	PxReal J;
	PxVec3 cofm;
	avbdEvaluateTetDeterminantAndGradient(
		tet, vertexOrder, e1, e2, e3, J, cofm);
	if(outLinearization)
	{
		outLinearization->determinant = J;
		outLinearization->determinantGradient = cofm;
	}

	const PxVec3& deformationWeights =
		tet.deformationGradientWeights[vertexOrder];
	const PxVec3 Fm =
		e1 * deformationWeights.x +
		e2 * deformationWeights.y +
		e3 * deformationWeights.z;

	PxReal V0 = tet.restVolume;

	// Inversion protection: clamp J to a small positive value so that
	// fully inverted tets produce bounded restoration forces instead of
	// catastrophic blowup.  The force direction remains correct (cofactor
	// still points toward un-inverting the tet).
	const PxReal Jmin = 0.05f;
	const PxReal Jsafe = PxMax(J, Jmin);

	outForce =
		(Fm * mu + cofm * (lam * (Jsafe - alpha))) * (-V0);

	const PxReal m2 = tet.shapeGradientNormSq[vertexOrder];
	outHessian =
		PxMat33::createDiagonal(PxVec3(mu * m2 * V0)) +
		avbdOuter(cofm, cofm) * (lam * V0);

	// Extra diagonal regularization for severely compressed / inverted tets
	// to keep the Hessian well-conditioned.
	if(J < 0.5f)
	{
		const PxReal regularization =
			(0.5f - J) * lam * V0 * m2;
		outHessian.column0.x += regularization;
		outHessian.column1.y += regularization;
		outHessian.column2.z += regularization;
	}
}

PX_FORCE_INLINE void avbdEvaluateNeoHookeanForceHessian(
	const AvbdTetElement& tet, int vOrder,
	PxReal mu, PxReal lam,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	const PxReal lambdaSafe =
		PxAbs(lam) < 1e-6f ? 1e-6f : lam;
	avbdEvaluateNeoHookeanForceHessianPrepared(
		tet, vOrder, mu, lam, 1.0f + mu / lambdaSafe,
		particles, outForce, outHessian);
}

PX_FORCE_INLINE AvbdSoftTetDisplacementLimitResult
avbdLimitTetDisplacementFromLinearizations(
	const PxVec3& displacement,
	const AvbdTetVertexLinearization* linearizations,
	PxU32 linearizationCount, PxReal minDetF = 0.05f)
{
	if(!displacement.isFinite())
	{
		return AvbdSoftTetDisplacementLimitResult(
			PxVec3(0.0f), 0.0f,
			AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED);
	}

	PxReal fraction = 1.0f;
	for(PxU32 linearizationId = 0;
		linearizationId < linearizationCount; linearizationId++)
	{
		const AvbdTetVertexLinearization& linearization =
			linearizations[linearizationId];
		const PxReal currentDetF = linearization.determinant;
		const PxReal proposedDetF =
			currentDetF +
			linearization.determinantGradient.dot(displacement);
		if(!PxIsFinite(currentDetF) || !PxIsFinite(proposedDetF))
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					eNONFINITE_REJECTED);
		}
		if(proposedDetF >= minDetF ||
			proposedDetF >= currentDetF)
			continue;
		if(currentDetF <= minDetF)
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					ePOSITIVE_J_REJECTED);
		}
		const PxReal admissible =
			(currentDetF - minDetF) /
			(currentDetF - proposedDetF);
		fraction = PxMin(
			fraction,
			PxMax(0.0f, admissible * 0.99f));
	}
	return AvbdSoftTetDisplacementLimitResult(
		displacement * fraction, fraction,
		fraction < 1.0f
			? AvbdSoftTetDisplacementLimitReason::
				ePOSITIVE_J_LIMITED
			: AvbdSoftTetDisplacementLimitReason::eNONE);
}

// Limit a single-particle displacement so no incident tetrahedron is
// pushed through the same positive-J floor used by the Neo-Hookean model.
// For one moving vertex, det(F) is affine in the displacement, so the
// admissible fraction is available analytically without a global line search.
PX_FORCE_INLINE AvbdSoftTetDisplacementLimitResult
avbdLimitTetDisplacementObserved(
	const AvbdSoftBody& body, PxU32 particleIdx,
	const AvbdSoftParticle* particles, const PxVec3& displacement,
	PxReal minDetF = 0.05f)
{
	if(!displacement.isFinite())
	{
		return AvbdSoftTetDisplacementLimitResult(
			PxVec3(0.0f), 0.0f,
			AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED);
	}
	if(particleIdx < body.compiled.particleStart ||
		particleIdx >= body.compiled.particleStart + body.compiled.particleCount)
	{
		return AvbdSoftTetDisplacementLimitResult(
			displacement, 1.0f,
			AvbdSoftTetDisplacementLimitReason::eNONE);
	}

	const PxU32 localIdx = particleIdx - body.compiled.particleStart;
	const AvbdParticleElementAdjacency& adjacency =
		body.compiled.elementAdjacency[localIdx];
	PxReal fraction = 1.0f;
	for(PxU32 refId = 0; refId < adjacency.tetRefs.size(); ++refId)
	{
		const AvbdParticleElementRef& ref =
			adjacency.tetRefs[refId];
		const AvbdTetElement& tet =
			body.compiled.tetElements[ref.index];
		const PxVec3 current0 = particles[tet.p0].position;
		const PxVec3 e1 =
			particles[tet.p1].position - current0;
		const PxVec3 e2 =
			particles[tet.p2].position - current0;
		const PxVec3 e3 =
			particles[tet.p3].position - current0;
		PxReal currentDetF;
		PxVec3 determinantGradient;
		avbdEvaluateTetDeterminantAndGradient(
			tet, ref.vOrder, e1, e2, e3,
			currentDetF, determinantGradient);
		const PxReal proposedDetF =
			currentDetF + determinantGradient.dot(displacement);

		if(!PxIsFinite(currentDetF) || !PxIsFinite(proposedDetF))
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					eNONFINITE_REJECTED);
		}
		if(proposedDetF >= minDetF || proposedDetF >= currentDetF)
			continue;
		if(currentDetF <= minDetF)
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					ePOSITIVE_J_REJECTED);
		}
		const PxReal admissible =
			(currentDetF - minDetF) /
			(currentDetF - proposedDetF);
		fraction = PxMin(fraction, PxMax(0.0f, admissible * 0.99f));
	}
	const AvbdSoftTetDisplacementLimitReason reason =
		fraction < 1.0f
			? AvbdSoftTetDisplacementLimitReason::ePOSITIVE_J_LIMITED
			: AvbdSoftTetDisplacementLimitReason::eNONE;
	return AvbdSoftTetDisplacementLimitResult(
		displacement * fraction, fraction, reason);
}

PX_FORCE_INLINE PxVec3 avbdLimitTetDisplacement(
	const AvbdSoftBody& body, PxU32 particleIdx,
	const AvbdSoftParticle* particles, const PxVec3& displacement,
	PxReal minDetF = 0.05f)
{
	return avbdLimitTetDisplacementObserved(
		body, particleIdx, particles, displacement, minDetF).
			appliedDisplacement;
}

PX_FORCE_INLINE void avbdEvaluateBendingForceHessian(
	const AvbdBendingElement& be, int vOrder,
	PxReal stiffness,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	const PxReal eps = 1e-6f;

	PxVec3 x0 = particles[be.opp0].position;
	PxVec3 x1 = particles[be.opp1].position;
	PxVec3 x2 = particles[be.edgeStart].position;
	PxVec3 x3 = particles[be.edgeEnd].position;

	PxVec3 e = x3 - x2;
	PxVec3 x02 = x2 - x0, x03 = x3 - x0;
	PxVec3 x13 = x3 - x1, x12 = x2 - x1;

	PxVec3 n1 = x02.cross(x03);
	PxVec3 n2 = x13.cross(x12);

	PxReal n1Norm = n1.magnitude();
	PxReal n2Norm = n2.magnitude();
	PxReal eNorm = e.magnitude();

	if (n1Norm < eps || n2Norm < eps || eNorm < eps)
	{
		outForce = PxVec3(0.0f);
		outHessian = PxMat33(PxZero);
		return;
	}

	PxVec3 n1Hat = n1 * (1.0f / n1Norm);
	PxVec3 n2Hat = n2 * (1.0f / n2Norm);
	PxVec3 eHat = e * (1.0f / eNorm);

	PxReal sinTheta = n1Hat.cross(n2Hat).dot(eHat);
	PxReal cosTheta = PxClamp(n1Hat.dot(n2Hat), -1.0f, 1.0f);
	PxReal theta = PxAtan2(sinTheta, cosTheta);

	PxReal k = stiffness * be.restLength;
	PxReal dE_dtheta = k * (theta - be.restAngle);

	auto normalizedDerivative = [](PxReal unnormLen, const PxVec3& nHat,
	                                const PxMat33& dNdx) -> PxMat33 {
		PxMat33 P = PxMat33(PxIdentity) - avbdOuter(nHat, nHat);
		return (P * dNdx) * (1.0f / unnormLen);
	};

	auto angleDerivative = [](const PxVec3& n1h, const PxVec3& n2h, const PxVec3& eh,
	                          const PxMat33& dn1dx, const PxMat33& dn2dx,
	                          PxReal sinT, PxReal cosT,
	                          const PxMat33& skN1, const PxMat33& skN2) -> PxVec3 {
		PxMat33 dSinMat = skN1 * dn2dx - skN2 * dn1dx;
		PxVec3 dSin = dSinMat.getTranspose() * eh;
		PxVec3 dCos = dn1dx.getTranspose() * n2h + dn2dx.getTranspose() * n1h;
		return dSin * cosT - dCos * sinT;
	};

	PxMat33 skE = avbdSkew(e);
	PxMat33 skX03 = avbdSkew(x03);
	PxMat33 skX02 = avbdSkew(x02);
	PxMat33 skX13 = avbdSkew(x13);
	PxMat33 skX12 = avbdSkew(x12);
	PxMat33 skN1 = avbdSkew(n1Hat);
	PxMat33 skN2 = avbdSkew(n2Hat);

	PxMat33 dn1hat_dx0 = normalizedDerivative(n1Norm, n1Hat, skE);
	PxMat33 dn1hat_dx1(PxZero);
	PxMat33 dn1hat_dx2 = normalizedDerivative(n1Norm, n1Hat, skX03 * (-1.0f));
	PxMat33 dn1hat_dx3 = normalizedDerivative(n1Norm, n1Hat, skX02);

	PxMat33 dn2hat_dx0(PxZero);
	PxMat33 dn2hat_dx1 = normalizedDerivative(n2Norm, n2Hat, skE * (-1.0f));
	PxMat33 dn2hat_dx2 = normalizedDerivative(n2Norm, n2Hat, skX13);
	PxMat33 dn2hat_dx3 = normalizedDerivative(n2Norm, n2Hat, skX12 * (-1.0f));

	PxVec3 dtheta_dx0 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx0, dn2hat_dx0,
	                                     sinTheta, cosTheta, skN1, skN2);
	PxVec3 dtheta_dx1 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx1, dn2hat_dx1,
	                                     sinTheta, cosTheta, skN1, skN2);
	PxVec3 dtheta_dx2 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx2, dn2hat_dx2,
	                                     sinTheta, cosTheta, skN1, skN2);
	PxVec3 dtheta_dx3 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx3, dn2hat_dx3,
	                                     sinTheta, cosTheta, skN1, skN2);

	PxVec3 dtheta_dx;
	switch (vOrder)
	{
		case 0: dtheta_dx = dtheta_dx0; break;
		case 1: dtheta_dx = dtheta_dx1; break;
		case 2: dtheta_dx = dtheta_dx2; break;
		case 3: dtheta_dx = dtheta_dx3; break;
		default:
			outForce = PxVec3(0.0f);
			outHessian = PxMat33(PxZero);
			return;
	}

	outForce = dtheta_dx * (-dE_dtheta);
	outHessian = avbdOuter(dtheta_dx, dtheta_dx) * k;
}

// =============================================================================
// AVBD contact/pin evaluators
// =============================================================================
