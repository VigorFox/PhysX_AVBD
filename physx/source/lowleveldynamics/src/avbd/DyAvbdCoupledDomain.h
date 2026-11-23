// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_COUPLED_DOMAIN_H
#define DY_AVBD_COUPLED_DOMAIN_H

#include "foundation/PxArray.h"
#include "foundation/PxMath.h"
#include "foundation/PxVec3.h"
#include <cfloat>
#include <cmath>

namespace physx
{
namespace Dy
{

// Temporary Schur data for a block-diagonal primal metric.  The physical
// unknowns remain Particle3 blocks; these records only solve the multipliers
// of shared active-domain inequalities A*d >= rhs.
struct AvbdCoupledDomainBlock
{
	PxVec3d step;
};

struct AvbdCoupledDomainTerm
{
	PxU32 block;
	PxVec3d gradient;
	PxVec3d inverseMetricGradient;
};

struct AvbdCoupledDomainConstraint
{
	PxU32 termStart;
	PxU32 termCount;
	double rhs;
	double tolerance;
	double multiplier;
	double response;
};

// Candidate construction in the PxReal lattice. This is a margin for the
// algebraic proposal, never a relaxation of final geometry admission or a
// stationarity test. Account for rounding p+d and the change of the existing
// float-input interval along d. The original nonlinear program remains the
// authority, including when the corrected direction crosses an exponent bin.
PX_INLINE double avbdCoupledPublicationError(const PxVec3& point,
	const PxVec3d& step,const PxVec3d& gradient)
{
	double error=0.0;
	for(PxU32 axis=0;axis<3;++axis)
	{
		const PxReal value=point[axis];
		const double spacing=PxMax(
			double(std::nextafter(value,PX_MAX_F32))-double(value),
			double(value)-double(std::nextafter(value,-PX_MAX_F32)));
		error+=PxAbs(gradient[axis])*(0.5*spacing+
			8.0*double(PX_EPS_F32)*PxAbs(step[axis]));
	}
	return error;
}

// Choose the publication variables before solving. An unconstrained proposal
// outside this support set must not make a plane look feasible and then vanish
// when the candidate is committed. The caller queries the complete domain
// again and grows this set only through newly active non-product constraints.
PX_INLINE bool avbdCoupledDomainPublicationScope(
	const PxArray<AvbdCoupledDomainBlock>& freeBlocks,
	const PxArray<AvbdCoupledDomainConstraint>& constraints,
	const PxArray<AvbdCoupledDomainTerm>& terms,
	PxArray<PxU8>& writable,PxArray<AvbdCoupledDomainBlock>& blocks)
{
	writable.resize(freeBlocks.size());
	for(PxU32 i=0;i<writable.size();++i) writable[i]=0;
	for(PxU32 i=0;i<constraints.size();++i)
	{
		const AvbdCoupledDomainConstraint& c=constraints[i];
		if(c.termStart>terms.size() || c.termCount>terms.size()-c.termStart) return false;
		if(c.termCount<2) continue;
		for(PxU32 j=c.termStart;j<c.termStart+c.termCount;++j)
		{
			if(terms[j].block>=writable.size()) return false;
			writable[terms[j].block]=1;
		}
	}
	blocks=freeBlocks;
	for(PxU32 i=0;i<blocks.size();++i)
		if(!writable[i]) blocks[i].step=PxVec3d(0.0);
	return true;
}

// Hildreth coordinate ascent on the inequality multipliers.  It is the exact
// dual of min 1/2*d^T*H*d-f^T*d for the supplied block-diagonal H.  It does
// not publish a Jacobi pose: callers still validate and commit one atomic
// candidate against the original nonlinear row/domain program.
PX_INLINE bool avbdSolveCoupledDomainKkt(
	PxArray<AvbdCoupledDomainBlock>& blocks,
	PxArray<AvbdCoupledDomainConstraint>& constraints,
	const PxArray<AvbdCoupledDomainTerm>& terms,
	double& maximumViolation,
	bool useInputIntervals=false)
{
	maximumViolation=0.0;
	if(constraints.empty()) return true;
	// Only endpoints present in the active program can change during its dual
	// sweeps. Preserve the original all-block stopping scale, but do not copy
	// and measure every unrelated particle on every multiplier iteration.
	PxArray<PxU8> touched(blocks.size(),0);
	PxArray<PxU32> activeBlocks;
	for(PxU32 i=0;i<constraints.size();++i)
	{
		AvbdCoupledDomainConstraint& constraint=constraints[i];
		constraint.multiplier=0.0;
		constraint.response=0.0;
		if(constraint.termStart+constraint.termCount>terms.size()) return false;
		for(PxU32 j=constraint.termStart;
			j<constraint.termStart+constraint.termCount;++j)
		{
			const AvbdCoupledDomainTerm& term=terms[j];
			if(term.block>=blocks.size() || !term.gradient.isFinite() ||
				!term.inverseMetricGradient.isFinite()) return false;
			if(!touched[term.block])
			{ touched[term.block]=1;activeBlocks.pushBack(term.block); }
			constraint.response+=term.gradient.dot(term.inverseMetricGradient);
		}
		if(!PxIsFinite(constraint.rhs) || !PxIsFinite(constraint.tolerance) ||
			constraint.tolerance<0.0 || !PxIsFinite(constraint.response) ||
			constraint.response<0.0) return false;
	}

	PxArray<PxVec3d> previousSteps;
	previousSteps.resize(activeBlocks.size());
	double unchangedScale=1.0;
	for(PxU32 block=0;block<blocks.size();++block)
		if(!touched[block]) unchangedScale=PxMax(unchangedScale,blocks[block].step.magnitude());
	auto completeIntervalRay=[&]()->bool
	{
		if(!useInputIntervals) return false;
		double fraction=1.0;
		for(PxU32 i=0;i<constraints.size();++i)
		{
			const AvbdCoupledDomainConstraint& constraint=constraints[i];
			const double effectiveRhs=constraint.rhs-constraint.tolerance;
			double value=0.0;
			for(PxU32 j=constraint.termStart;
				j<constraint.termStart+constraint.termCount;++j)
			{
				const AvbdCoupledDomainTerm& term=terms[j];
				value+=term.gradient.dot(blocks[term.block].step);
			}
			const double scale=1.0+PxAbs(effectiveRhs)+PxAbs(value);
			if(effectiveRhs-value<=64.0*DBL_EPSILON*scale) continue;
			// The current committed pose is the zero-step witness of the input
			// interval. Intersect the segment from it to this KKT iterate with every
			// violated half-space; no geometric or scene tolerance is introduced.
			if(effectiveRhs>=0.0 || value>=0.0) return false;
			fraction=PxMin(fraction,effectiveRhs/value);
		}
		if(!PxIsFinite(fraction) || fraction<0.0 || fraction>1.0) return false;
		if(fraction<1.0)
		{
			fraction=std::nextafter(fraction,0.0);
			for(PxU32 block=0;block<blocks.size();++block)
				blocks[block].step*=fraction;
		}
		maximumViolation=0.0;
		double valueScale=1.0;
		for(PxU32 i=0;i<constraints.size();++i)
		{
			const AvbdCoupledDomainConstraint& constraint=constraints[i];
			const double effectiveRhs=constraint.rhs-constraint.tolerance;
			double value=0.0;
			for(PxU32 j=constraint.termStart;
				j<constraint.termStart+constraint.termCount;++j)
			{
				const AvbdCoupledDomainTerm& term=terms[j];
				value+=term.gradient.dot(blocks[term.block].step);
			}
			maximumViolation=PxMax(maximumViolation,effectiveRhs-value);
			valueScale=PxMax(valueScale,PxAbs(effectiveRhs)+PxAbs(value));
		}
		return maximumViolation<=256.0*DBL_EPSILON*valueScale;
	};
	const PxU32 sweepLimit=256*(constraints.size()+1);
	for(PxU32 sweep=0;sweep<sweepLimit;++sweep)
	{
		for(PxU32 i=0;i<activeBlocks.size();++i)
			previousSteps[i]=blocks[activeBlocks[i]].step;
		double maximumMultiplierChange=0.0;
		double multiplierScale=1.0;
		for(PxU32 i=0;i<constraints.size();++i)
		{
			AvbdCoupledDomainConstraint& constraint=constraints[i];
			double value=0.0;
			for(PxU32 j=constraint.termStart;
				j<constraint.termStart+constraint.termCount;++j)
			{
				const AvbdCoupledDomainTerm& term=terms[j];
				value+=term.gradient.dot(blocks[term.block].step);
			}
			const double effectiveRhs=constraint.rhs-
				(useInputIntervals ? constraint.tolerance : 0.0);
			const double violation=effectiveRhs-value;
			if(constraint.response<=1.0e-30)
			{
				if(violation>
					64.0*DBL_EPSILON*(1.0+PxAbs(effectiveRhs)))
					return false;
				continue;
			}
			const double next=PxMax(0.0,
				constraint.multiplier+violation/constraint.response);
			const double change=next-constraint.multiplier;
			constraint.multiplier=next;
			maximumMultiplierChange=PxMax(maximumMultiplierChange,PxAbs(change));
			multiplierScale=PxMax(multiplierScale,PxAbs(next));
			if(change==0.0) continue;
			for(PxU32 j=constraint.termStart;
				j<constraint.termStart+constraint.termCount;++j)
			{
				const AvbdCoupledDomainTerm& term=terms[j];
				blocks[term.block].step+=term.inverseMetricGradient*change;
			}
		}

		double maximumPrimalChange=0.0;
		double primalScale=unchangedScale;
		for(PxU32 i=0;i<activeBlocks.size();++i)
		{
			const PxU32 block=activeBlocks[i];
			maximumPrimalChange=PxMax(maximumPrimalChange,
				(blocks[block].step-previousSteps[i]).magnitude());
			primalScale=PxMax(primalScale,blocks[block].step.magnitude());
		}
		maximumViolation=0.0;
		double valueScale=1.0;
		double maximumComplementarity=0.0,complementarityScale=1.0;
		for(PxU32 i=0;i<constraints.size();++i)
		{
			const AvbdCoupledDomainConstraint& constraint=constraints[i];
			const double effectiveRhs=constraint.rhs-
				(useInputIntervals ? constraint.tolerance : 0.0);
			double value=0.0;
			for(PxU32 j=constraint.termStart;
				j<constraint.termStart+constraint.termCount;++j)
			{
				const AvbdCoupledDomainTerm& term=terms[j];
				value+=term.gradient.dot(blocks[term.block].step);
			}
			maximumViolation=PxMax(maximumViolation,effectiveRhs-value);
			valueScale=PxMax(valueScale,PxAbs(effectiveRhs)+PxAbs(value));
			maximumComplementarity=PxMax(maximumComplementarity,
				PxAbs(constraint.multiplier*(value-effectiveRhs)));
			complementarityScale=PxMax(complementarityScale,
				constraint.multiplier*(PxAbs(value)+PxAbs(effectiveRhs)));
		}
		const double feasibilityTolerance=256.0*DBL_EPSILON*valueScale;
		const double dualTolerance=256.0*DBL_EPSILON*multiplierScale;
		// Coordinate ascent loses meaningful primal progress at the square-root
		// machine-precision scale, especially with dependent contact planes.
		// The caller still publishes through PxReal and revalidates the complete
		// nonlinear domain, so this only terminates the internal double solve.
		const double primalTolerance=PxSqrt(DBL_EPSILON)*primalScale;
		const bool complementary=maximumComplementarity<=
			256.0*DBL_EPSILON*complementarityScale;
		// Redundant active half-spaces have non-unique multipliers.  Their
		// coordinate updates may keep exchanging dual mass after the physical
		// Particle3 step has converged, so accept either dual convergence or
		// convergence of the primal unknown that will actually be published.
		// Nearly dependent planes can exchange large multipliers while producing
		// an almost unchanged primal step. Feasibility and that small change do
		// not imply stationarity: positive multipliers must also have zero slack.
		// Otherwise even a feasible zero origin can yield an uphill "solution".
		if(complementary && maximumViolation<=feasibilityTolerance &&
			(maximumMultiplierChange<=dualTolerance ||
				maximumPrimalChange<=primalTolerance))
			return true;
		if(complementary && useInputIntervals && maximumPrimalChange<=primalTolerance &&
			completeIntervalRay()) return true;
	}
	return completeIntervalRay();
}

} // namespace Dy
} // namespace physx

#endif
