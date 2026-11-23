// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_INTERNAL_H
#define DY_AVBD_INTERNAL_H

#include "avbd/DyAvbdDynamics.h"
#include "avbd/DyAvbdContactPotential.h"
#include "avbd/DyAvbdOgcActivation.h"
#include "avbd/DyAvbdRigidMetric.h"
#include "avbd/DyAvbdVbd.h"
#include "CmSpatialVector.h"
#include "DyArticulationJointCore.h"
#include "DyDeformableVolume.h"
#include "DyFeatherstoneArticulation.h"
#include "PxsRigidBody.h"
#include "foundation/PxMat33.h"
#include "foundation/PxMath.h"
#include "foundation/PxMemory.h"
#include "foundation/PxVec4.h"

namespace physx
{
namespace Dy
{

template<class PoseView>
PX_INLINE bool avbdNpCandidateAdmissible(const AvbdRowConstRef& row,
    const PxArray<AvbdNpDomain>& domains,const PxArray<AvbdSolverBody>& bodies,
    const PoseView& candidate,const KernelTable& kernels,AvbdNpQuery* query=NULL)
{
    if(row.header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT) return true;
    if(!row.header().geometryIndex || row.header().geometryIndex>domains.size()) return false;
    const AvbdNpDomain& domain=domains[row.header().geometryIndex-1];
    const PxU32 indices[2]={row.header().bodyA,row.header().bodyB};
    PxTransform start[2],end[2];
    for(PxU32 s=0;s<2;++s)
    {
        const PxU32 i=indices[s];
        // Primal iterates are not physical time samples. In particular a
        // prescribed boundary already carries its end pose during assembly.
        // Certify the physical DCD-to-candidate path, never that mixed epoch.
        start[s]=i<bodies.size()?PxTransform(bodies[i].pStart,bodies[i].qStart):domain.fixedPose[s];
        end[s]=i<bodies.size()?PxTransform(candidate[i].p,candidate[i].q):domain.fixedPose[s];
    }
    bool accepted=false;
    AvbdNpDomainArgs args={&domain,start,end,&accepted,query};kernels.launch(AVBD_NP_DOMAIN,&args);
    return accepted;
}

PX_INLINE AvbdContactPotential avbdNpContactPotential(const AvbdRowConstRef& row,PxReal gap)
{
    if(row.scalar().contactRadius<=0)
        return avbdUnilateralContactPotential(gap,row.scalar().rho,row.scalar().lambda,row.scalar().minLambda);
    if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)
    {
        const auto value=avbdOGCNormalEnvelope(AvbdNodalReal(gap),AvbdNodalReal(row.scalar().lambda),AvbdNodalReal(row.scalar().rho),
            AvbdNodalReal(row.scalar().contactRadius),AvbdNodalReal(row.scalar().contactStiffness));
        return {PxReal(value.energy),-PxReal(value.force),PxReal(value.hessian)};
    }
    const auto value=avbdOGCNormalEnvelope(double(gap),double(row.scalar().lambda),double(row.scalar().rho),
        double(row.scalar().contactRadius),double(row.scalar().contactStiffness));
    const AvbdContactPotential out={PxReal(value.energy),-PxReal(value.force),PxReal(value.hessian)};
    return out;
}

static const PxReal AVBD_RHO = 1.0e4f;
// AVBD_2's point-lock objective uses a fixed penalty with an inertial-scale
// dual step. It is not the adaptive demo3d hard-row update below.
static const PxReal AVBD_POINT_LOCK_PENALTY = 1.0e6f;
static const PxReal AVBD_POINT_LOCK_DUAL_DECAY = 0.99f;
// AVBD_2 joint/contact closure uses 4 * positionTolerance (1e-4).
static const PxReal AVBD_NP_JOINT_CLOSURE_REL_TOL = 4.0e-4f;
// Other hard joint rows (artic locks/limits, solverPrep 1D rows) start at
// AVBD_RHO and keep a per-row penalty across steps (demo3d joint.cpp:
// penalty*gamma at warmstart, +beta|C| per dual update).  The load is carried
// by lambda, so the penalty only has to bound the transient of a load change;
// fixed 1e5/1e6 floors were 1000x the light links' m/h^2 and stalled block GS
// on the collective modes of a loop chain (levitating scissor base).
static const PxReal AVBD_JOINT_BETA_ANGULAR_SCALE = 0.01f;
static const PxReal AVBD_RHO_MAX = 1.0e8f;
static const PxReal AVBD_RHO_T_MIN = 1.0f;
// PhysX_AVBD_2 keeps the contact AL penalty independent from the OGC query
// stiffness.  The query decides whether a row exists; this value conditions
// the position objective and is the warm-start lower bound.
static const PxReal AVBD_CONTACT_PENALTY_MIN = 1.0e2f;
static const PxReal AVBD_CONTACT_PENALTY_MAX = 1.0e9f;
static const PxReal AVBD_CONTACT_JOINT_BOOST_FRACTION = 0.005f;
static const PxReal AVBD_BETA = 1000.0f;
static const PxReal AVBD_ALPHA = 0.99f;
static const PxReal AVBD_GAMMA = 0.999f;
// Solver-level post-AL velocity stabilization from the AVBD_2 green path.
// This is applied before material complementarity; authored body damping is
// an additional public property, not a replacement for these step factors.
static const PxReal AVBD_VELOCITY_DAMPING = 0.99f;
static const PxReal AVBD_ANGULAR_DAMPING = 0.95f;
// Spectral-radius estimate used by the AVBD_2 fixed-point accelerator.  It
// accelerates the same in-place GS map; it is not a constraint stiffness.
static const PxReal AVBD_JOINT_CHEBYSHEV_RHO = 0.92f;
// AVBD_2 green contact conditioning: rho >= 2 / (J M^-1 J^T dt^2).
// HelloWorld he=2 density-10 boxes have M/dt^2 ~ 2e6; packing rho at
// 1e4-1e5 leaves the stack a soft spring whose BDF1 residual is the
// 1cm overlap / 5.6cm top bounce in the pose log (PGS overlap=0).
static const PxReal AVBD_CONTACT_CONDITIONING = 2.0f;
static const PxReal AVBD_PEN_SCALE_BODY_VS_STATIC = 2.0f;
static const PxU32 AVBD_ITERS_CONTACT = 10;
static const PxU32 AVBD_ITERS_JOINT = 8;
static const PxU32 AVBD_ITERS_CONTACT_JOINT_MAX = 32;
static const PxU32 AVBD_ITERS_ARTIC_MAX = 128;
// This distilled backend currently certifies only the in-place GS pose map;
// keep both coordinates strict until the reference residual certificate is
// migrated together with its operator.  The 128-sweep limit is only a guard.
static const PxReal AVBD_ARTIC_POSITION_REL_TOL = 1.0e-5f;
static const PxReal AVBD_ARTIC_ROTATION_TOL = 1.0e-5f;
// Material iterations terminate on the velocity natural-map certificate.
// The cap is only a finite-work guard for extreme mass-ratio manifolds.
static const PxU32 AVBD_MATERIAL_MAX_ITERS = 32;
static const PxReal AVBD_MATERIAL_REL_TOL = 1.0e-5f;
static const PxReal AVBD_BOUNCE_SOR = 0.5f;
static const PxReal AVBD_REG = 1.0e-6f;
static const PxReal AVBD_SOFT_RIGID_WAKE_COUNTER = 20.0f * 0.02f;


PX_FORCE_INLINE void avbdFreeIslandSoftBlocks(AvbdIslandSoftCache& cache)
{
	PX_FREE(cache.nodalRest);
	PX_FREE(cache.particles);
}

PX_FORCE_INLINE PxVec3 avbdPointVel(const PxVec3& lin, const PxVec3& ang,
	const PxQuat& q, const PxVec3& rLocal)
{
	return lin + ang.cross(q.rotate(rLocal));
}

PX_FORCE_INLINE bool avbdBodyResponds(const AvbdSolverBody& b)
{
	return b.core && b.invMass > 0.0f && !b.isKinematic;
}

template<class BodyView>
PX_FORCE_INLINE PxVec3 avbdBlockSupportPoint(const AvbdBlock& block,
	const AvbdRowSupport& support, const BodyView& bodies)
{
	if(block.type == eAVBD_BLOCK_RIGID6)
	{
		if(block.objectIndex >= bodies.size())
			return PxVec3(0.0f);
		const auto& body = bodies[block.objectIndex];
		return avbdContactWorldPoint(body.p,body.q,support.localPoint);
	}
	return block.particle ? block.particle->p : PxVec3(0.0f);
}


// Remaining BDF1 inertial of a support, from the current committed pose.
// Inertial-target admission adds every other endpoint's remainder so co-fall
// is a relative step.  Primal rows already evaluate all committed endpoints.
PX_FORCE_INLINE PxVec3 avbdBlockSupportInertialDelta(const AvbdBlock& block,
	const AvbdRowSupport& support, const PxArray<AvbdSolverBody>& bodies)
{
	if(block.type == eAVBD_BLOCK_RIGID6)
	{
		if(block.objectIndex >= bodies.size())
			return PxVec3(0.0f);
		const AvbdSolverBody& body = bodies[block.objectIndex];
		const PxVec3 current = body.p + body.q.rotate(support.localPoint);
		const PxVec3 predicted = body.pHat + body.qHat.rotate(support.localPoint);
		return predicted - current;
	}
	if(!block.particle)
		return PxVec3(0.0f);
	return block.particle->pHat - block.particle->p;
}

template<class BodyView>
PX_INLINE AvbdContactFrameEvaluation avbdContactFrameRow(const AvbdRowConstRef& row,
 const PxArray<AvbdRowSupport>& supports,const PxArray<AvbdBlock>& blocks,
 const BodyView& bodies,const AvbdContactGeometryView& geometry,
 PxU32 candidateBlock=PX_MAX_U32,const PxVec3* candidate=NULL,const PxVec3* localAxis=NULL)
{
 AvbdContactFrameEvaluation result={};
 if(!geometry.frames || !geometry.kernels || !row.header().geometryIndex || row.header().geometryIndex>geometry.frames->size()) return result;
 AvbdContactFrameGeometry frame=(*geometry.frames)[row.header().geometryIndex-1];
 if(localAxis) frame.normalLocal=*localAxis;
 PxVec3d points[5];
 avbdContactFramePoints(frame,supports.begin()+row.header().supportStart,
  [&](const AvbdRowSupport& support)
  { return candidate && support.block==candidateBlock ? *candidate : avbdBlockSupportPoint(blocks[support.block],support,bodies); },points);
 AvbdContactFrameArgs args={&frame,points,double(row.scalar().contactOffset),&result};
 geometry.kernels->launch(AVBD_CONTACT_FRAME,&args);
 return result;
}

PX_INLINE PxVec3 avbdContactFrameGradient(const AvbdRowConstRef& row,
 const PxArray<AvbdRowSupport>& supports,const AvbdContactGeometryView& geometry,
 const AvbdContactFrameEvaluation& evaluation,PxU32 block)
{
 const PxVec3d g=avbdContactFrameBlockGradient((*geometry.frames)[row.header().geometryIndex-1],
  supports.begin()+row.header().supportStart,evaluation,block);
 return PxVec3(PxReal(g.x),PxReal(g.y),PxReal(g.z));
}

template<class Visit>
PX_INLINE void avbdForEachContactFrameBlock(const AvbdRowConstRef& row,
 const PxArray<AvbdRowSupport>& supports,const Visit& visit)
{
 for(PxU32 i=row.header().supportStart;i<row.header().supportStart+row.header().supportCount;++i)
 {
  const PxU32 block=supports[i].block;
  bool repeated=false;
  for(PxU32 j=row.header().supportStart;j<i;++j) if(supports[j].block==block) { repeated=true;break; }
  if(!repeated) visit(block);
 }
}

template<class Bodies> PX_INLINE PxVec3 avbdNodalWitnessTravel(const AvbdRowConstRef& row,
    const PxArray<AvbdRowSupport>& supports,const PxArray<AvbdBlock>& blocks,const Bodies& bodies)
{
    PxVec3 result(0);
    for(PxU32 k=row.header().supportStart;k<row.header().supportStart+row.header().supportCount;++k)
    {
        const auto& support=supports[k];const auto& block=blocks[support.block];
        if(block.particle)result+=(block.particle->p-block.particle->pStart)*support.weight;
        else
        {
            const auto& b=bodies[block.objectIndex];
            const auto arm=support.block==row.header().bodyA?row.scalar().rA:row.scalar().rB;
            result+=(b.p-b.pStart+b.q.rotate(arm)-b.qStart.rotate(arm))*support.weight;
        }
    }
    return result;
}
PX_FORCE_INLINE PxU32 avbdNodalRigidFrame(const AvbdRowConstRef& row,const AvbdContactGeometryView& g)
{
    return row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT && row.header().geometryIndex && g.nodal ?
        g.nodal[row.header().geometryIndex-1].rigidFrame : PX_MAX_U32;
}
template<class Bodies> PX_INLINE PxVec3 avbdNodalContactSlip(const AvbdRowConstRef& row,
    const PxArray<AvbdRowSupport>& supports,const PxArray<AvbdBlock>& blocks,const Bodies& bodies,
    const AvbdContactGeometryView& geometry)
{
    const auto travel=avbdNodalWitnessTravel(row,supports,blocks,bodies);
    const auto frame=avbdNodalRigidFrame(row,geometry);
    if(frame==PX_MAX_U32)return travel;
    const auto& b=bodies[frame];
    return avbdNodalMaterialVector(b.qStart,b.q,travel);
}
template<class BodyView>
PX_FORCE_INLINE PxVec3 avbdHeterogeneousNormal(const AvbdRowConstRef& row,
 const BodyView& bodies, const AvbdContactGeometryView& geometry,
 const PxArray<AvbdRowSupport>& supports,const PxArray<AvbdBlock>& blocks)
{
 const auto frame=avbdNodalRigidFrame(row,geometry);
 if(frame!=PX_MAX_U32){const auto& b=bodies[frame];return avbdNodalWorldAxis(b.qStart,b.q,row.scalar().n);}
 if(row.header().kernelId==eAVBD_ROW_KERNEL_CONTACT_FRAME)
 {
  const auto value=avbdContactFrameRow(row,supports,blocks,bodies,geometry);
  return value.valid ? PxVec3(PxReal(value.normal.x),PxReal(value.normal.y),PxReal(value.normal.z)) : PxVec3(PxReal(NAN));
 }
 return row.header().geometryIndex && row.header().kernelId!=eAVBD_ROW_KERNEL_NODAL_CONTACT ? avbdContactWorldPoint(PxVec3(0.0f),bodies[row.header().bodyA].q,
  geometry.planes[row.header().geometryIndex-1].normalLocal) : row.scalar().n;
}

template<class BodyView>
PX_FORCE_INLINE PxVec3 avbdHeterogeneousPlaneOppositePoint(const AvbdRowConstRef& row,
 const PxArray<AvbdRowSupport>& supports, const PxArray<AvbdBlock>& blocks,
 const BodyView& bodies)
{
 PxVec3 point(0.0f);
 for(PxU32 k=row.header().supportStart;k<row.header().supportStart+row.header().supportCount;++k)
 {
  const AvbdRowSupport& support=supports[k];
  if(support.block==row.header().bodyA) continue;
  point+=support.weight*avbdBlockSupportPoint(blocks[support.block],support,bodies);
 }
 return point;
}

template<class BodyView>
PX_FORCE_INLINE AvbdContactPlaneEvaluation avbdHeterogeneousPlaneEvaluation(
 const AvbdRowConstRef& row, const PxArray<AvbdRowSupport>& supports,
 const PxArray<AvbdBlock>& blocks, const BodyView& bodies,
 const AvbdContactGeometryView& geometry)
{
 const auto& body=bodies[row.header().bodyA];
 return avbdEvaluateContactPlane(geometry.planes[row.header().geometryIndex-1].normalLocal,
  row.scalar().rA,body.p,body.q,avbdHeterogeneousPlaneOppositePoint(row,supports,blocks,bodies),
  -row.scalar().contactOffset);
}

template<class BodyView>
PX_FORCE_INLINE PxReal avbdHeterogeneousRowC(const AvbdRowConstRef& row,
 const PxArray<AvbdRowSupport>& supports, const PxArray<AvbdBlock>& blocks,
 const BodyView& bodies, const AvbdContactGeometryView& geometry)
{
 if(row.header().kernelId==eAVBD_ROW_KERNEL_ATTACHMENT)
 {
     const auto& h=row.header();
     const auto rigid=bodies[h.bodyA];
     const PxU32 last=h.supportStart+h.supportCount-1;
     PxVec3 separation=-rigid.q.rotate(supports[last].localPoint);
     for(PxU32 k=h.supportStart;k<last;++k)
         separation+=supports[k].weight*(blocks[supports[k].block].particle->p-rigid.p);
     return row.scalar().n.dot(separation);
 }
 if(avbdNodalRigidFrame(row,geometry)!=PX_MAX_U32)
     return row.scalar().C0+row.scalar().n.dot(avbdNodalContactSlip(row,supports,blocks,bodies,geometry));
 if(row.header().kernelId==eAVBD_ROW_KERNEL_CONTACT_FRAME)
 {
  const auto value=avbdContactFrameRow(row,supports,blocks,bodies,geometry);
  return value.valid ? PxReal(value.gap) : PxReal(NAN);
 }
 if(row.header().geometryIndex && row.header().kernelId!=eAVBD_ROW_KERNEL_NODAL_CONTACT)
  return avbdHeterogeneousPlaneEvaluation(row,supports,blocks,bodies,geometry).gap;
 PxReal C=row.scalar().contactOffset;
 for(PxU32 k=row.header().supportStart;k<row.header().supportStart+row.header().supportCount;++k)
 {
  if(k>=supports.size() || supports[k].block>=blocks.size()) return 0.0f;
  const AvbdRowSupport& support=supports[k];
  C+=support.weight*row.scalar().n.dot(avbdBlockSupportPoint(blocks[support.block],support,bodies));
 }
 return C;
}

// k > 0 selects the offset potential; k = 0 selects unilateral AL.
// Triangle/box rows use complete plane supports. Force/energy consistency
// alone does not prove candidate coverage or final-pose admissibility.
PX_FORCE_INLINE bool avbdHeterogeneousPositionAl(const AvbdRowConstRef& row)
{
	return row.header().supportCount != 0 &&
		(row.header().flags & eAVBD_ROW_CONTACT) != 0 &&
		!(row.scalar().contactStiffness > 0.0f);
}

// A Position-AL geometry row owns a DCD-epoch lower bound.  Comparing every
// commit with the immediately preceding pose lets floating-point roundoff
// accumulate over blocks and sweeps until a separated feature changes side.
// Offset-potential rows keep the existing core-clearance admission rule.
PX_FORCE_INLINE double avbdHeterogeneousAdmissionOffset(const AvbdRowConstRef& row)
{
	return avbdHeterogeneousPositionAl(row) ? 0.0 : double(row.scalar().contactRadius);
}

PX_FORCE_INLINE double avbdHeterogeneousAdmissionLower(const AvbdRowConstRef& row,
	double currentGap)
{
	return avbdHeterogeneousPositionAl(row) ?
		PxMin(0.0, double(row.scalar().C0)) : PxMin(0.0, currentGap);
}

template<class OldPoint, class NewPoint>
PX_INLINE bool avbdHeterogeneousCandidateAdmissiblePoints(const AvbdRowConstRef& row,
 const PxArray<AvbdRowSupport>& supports,const PxVec3& oldNormal,
 const PxVec3& newNormal,const OldPoint& oldPoint,const NewPoint& newPoint)
{
 const double admissionOffset=avbdHeterogeneousAdmissionOffset(row);
  AvbdContactAdmissionState oldState(row.scalar().contactOffset,admissionOffset,row.scalar().C0);
  AvbdContactAdmissionState newState=oldState;
 for(PxU32 k=row.header().supportStart;k<row.header().supportStart+row.header().supportCount;++k)
 {
  const AvbdRowSupport& support=supports[k];
  const PxVec3 oldPosition=oldPoint(support);
  const PxVec3 newPosition=newPoint(support);
  oldState.add(support.weight,oldNormal,oldPosition);
  newState.add(support.weight,newNormal,newPosition);
 }
 const double lower=avbdHeterogeneousAdmissionLower(row,oldState.gap);
 // A published pose becomes the next block's old pose. Its admissibility
 // must survive that change of reference: the old pose's larger input scale
 // cannot authorize a boundary point rejected when evaluated against itself.
 const double candidateLower=avbdHeterogeneousAdmissionLower(row,newState.gap);
 return PxIsFinite(oldState.gap) && PxIsFinite(newState.gap) && PxIsFinite(lower) &&
  newState.gap>=lower-oldState.tolerance(newState) &&
  newState.gap>=candidateLower-newState.tolerance(newState);
}

// Evaluate both complete pose states, including a moving plane normal. Used
// by local rigid acceptance and the read-only all-block inertial candidate.
template<class OldBodies, class NewBodies>
PX_INLINE bool avbdHeterogeneousCandidateAdmissible(const AvbdRowConstRef& row,
 const PxArray<AvbdRowSupport>& supports, const PxArray<AvbdBlock>& oldBlocks,
 const OldBodies& oldBodies, const PxArray<AvbdBlock>& newBlocks,
 const NewBodies& newBodies, const AvbdContactGeometryView& geometry)
{
 if(row.header().kernelId==eAVBD_ROW_KERNEL_CONTACT_FRAME)
  return avbdContactFrameRow(row,supports,newBlocks,newBodies,geometry).valid;
 const PxVec3 oldNormal=avbdHeterogeneousNormal(row,oldBodies,geometry,supports,oldBlocks);
 const PxVec3 newNormal=avbdHeterogeneousNormal(row,newBodies,geometry,supports,newBlocks);
 return avbdHeterogeneousCandidateAdmissiblePoints(row,supports,oldNormal,
  newNormal,
  [&](const AvbdRowSupport& support)
  { return avbdBlockSupportPoint(oldBlocks[support.block],support,oldBodies); },
  [&](const AvbdRowSupport& support)
  { return avbdBlockSupportPoint(newBlocks[support.block],support,newBodies); });
}

PX_FORCE_INLINE bool avbdHeterogeneousSupportJacobian(const AvbdRowConstRef& row,
 const AvbdRowSupport& support, const AvbdBlock& block,
 const PxArray<AvbdSolverBody>& bodies,
 const AvbdContactGeometryView& geometry,
 const PxArray<AvbdRowSupport>& supports, const PxArray<AvbdBlock>& blocks,
 PxVec3& jLin, PxVec3& jAng)
{
 if(row.header().kernelId==eAVBD_ROW_KERNEL_CONTACT_FRAME) return false; // Uses the merged block gradient, not a repeated support gradient.
 jLin=avbdHeterogeneousNormal(row,bodies,geometry,supports,blocks)*support.weight;
 jAng=PxVec3(0.0f);
 if(block.type==eAVBD_BLOCK_RIGID6)
 {
  if(block.objectIndex>=bodies.size()) return false;
  if(row.header().geometryIndex && row.header().kernelId!=eAVBD_ROW_KERNEL_NODAL_CONTACT && block.objectIndex==row.header().bodyA)
   jAng=avbdHeterogeneousPlaneEvaluation(row,supports,blocks,bodies,geometry).rigidAngularGradient;
  else if(avbdNodalRigidFrame(row,geometry)!=PX_MAX_U32)
  {
   const auto arm=block.objectIndex==row.header().bodyA?row.scalar().rA:row.scalar().rB;
   const auto travel=avbdNodalWitnessTravel(row,supports,blocks,bodies);
   jAng=avbdNodalWitnessArm(bodies[block.objectIndex].q,arm,travel,support.weight).cross(jLin);
  }
  else
   jAng=bodies[block.objectIndex].q.rotate(support.localPoint).cross(jLin);
 }
 return jLin.isFinite() && jAng.isFinite();
}

template<class BodyView>
PX_FORCE_INLINE PxReal avbdHeterogeneousAlC(const AvbdRowConstRef& row,
	const PxArray<AvbdRowSupport>& supports,
	const PxArray<AvbdBlock>& blocks,
	const BodyView& bodies, const AvbdContactGeometryView& geometry)
{
	return avbdHeterogeneousRowC(row, supports, blocks, bodies, geometry);
}

PX_FORCE_INLINE AvbdContactPotential avbdHeterogeneousContactPotential(
	const AvbdRowConstRef& row, PxReal C)
{
    if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)return avbdNpContactPotential(row,C);
	const bool al = avbdHeterogeneousPositionAl(row);
	return avbdUnilateralContactPotential(C,
		al ? row.scalar().rho : row.scalar().contactStiffness, al ? row.scalar().lambda : 0.0f,
		row.scalar().minLambda);
}

template<class Bodies> PX_INLINE void avbdNodalContactJacobian(const AvbdRowConstRef& row,PxU32 block,
    const PxVec3& axis,const PxArray<AvbdRowSupport>& supports,const PxArray<AvbdBlock>& blocks,
    const Bodies& bodies,const AvbdContactGeometryView& geometry,PxVec3& linear,PxVec3& angular)
{
    linear=angular=PxVec3(0);
    for(PxU32 k=row.header().supportStart;k<row.header().supportStart+row.header().supportCount;++k)
    {
        const auto& s=supports[k];if(s.block!=block)continue;
        const auto frame=avbdNodalRigidFrame(row,geometry);
        const auto j=(frame==PX_MAX_U32?axis:avbdNodalWorldAxis(bodies[frame].qStart,bodies[frame].q,axis))*s.weight;linear+=j;
        if(!blocks[block].particle)
        {
            const auto arm=block==row.header().bodyA?row.scalar().rA:row.scalar().rB;
            auto lever=bodies[blocks[block].objectIndex].q.rotate(arm);
            if(frame!=PX_MAX_U32)lever=avbdNodalWitnessArm(bodies[frame].q,arm,avbdNodalWitnessTravel(row,supports,blocks,bodies),s.weight);
            angular+=lever.cross(j);
        }
    }
}

PX_FORCE_INLINE void avbdLockCoreVel(PxsBodyCore& core)
{
	if(core.lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_X) core.linearVelocity.x = 0.0f;
	if(core.lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Y) core.linearVelocity.y = 0.0f;
	if(core.lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Z) core.linearVelocity.z = 0.0f;
	if(core.lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_X) core.angularVelocity.x = 0.0f;
	if(core.lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y) core.angularVelocity.y = 0.0f;
	if(core.lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z) core.angularVelocity.z = 0.0f;
}

PX_FORCE_INLINE void avbdSyncArticMotionVel(AvbdSolverBody& b)
{
	if(!b.artic || !b.core)
		return;
	Cm::SpatialVectorF* mv = b.artic->getArticulationData().getMotionVelocities();
	if(mv)
	{
		const Cm::SpatialVectorF velocity(
			b.core->angularVelocity, b.core->linearVelocity);
		mv[b.linkIndex] = velocity;
		b.artic->getArticulationData().getPosIterMotionVelocity(b.linkIndex) =
			velocity;
	}
}

PX_FORCE_INLINE PxReal avbdContactVn(const AvbdRowConstRef& row, const PxArray<AvbdSolverBody>& bodies,
	bool unconstrained)
{
	PxVec3 vA(0.0f), vB(0.0f);
	if(row.header().bodyA != PX_MAX_U32)
	{
		const AvbdSolverBody& a = bodies[row.header().bodyA];
		// Material velocity is evaluated on the frozen solve-start contact arm.
		// Position AL may rotate the local anchor before writeback; using that
		// edited arm here changes the effective mass and manufactures torque.
		const PxQuat q = a.qStart;
		const PxVec3 lin = unconstrained ? a.linVel :
			(a.core ? a.core->linearVelocity : PxVec3(0.0f));
		const PxVec3 ang = unconstrained ? a.angVel :
			(a.core ? a.core->angularVelocity : PxVec3(0.0f));
		vA = avbdPointVel(lin, ang, q, row.scalar().ogcRA);
	}
	if(row.header().bodyB != PX_MAX_U32)
	{
		const AvbdSolverBody& b = bodies[row.header().bodyB];
		const PxQuat q = b.qStart;
		const PxVec3 lin = unconstrained ? b.linVel :
			(b.core ? b.core->linearVelocity : PxVec3(0.0f));
		const PxVec3 ang = unconstrained ? b.angVel :
			(b.core ? b.core->angularVelocity : PxVec3(0.0f));
		vB = avbdPointVel(lin, ang, q, row.scalar().ogcRB);
	}
	return row.scalar().n.dot(vA - vB);
}

PX_FORCE_INLINE PxVec3 avbdWorldInvInertiaMul(const AvbdSolverBody& b,
	const PxVec3& torqueWorld)
{
	const PxVec3 tLocal = b.qStart.rotateInv(torqueWorld);
	const PxVec3 aLocal(tLocal.x * b.invInertia.x,
		tLocal.y * b.invInertia.y, tLocal.z * b.invInertia.z);
	return b.qStart.rotate(aLocal);
}

PX_FORCE_INLINE PxReal avbdEndpointAxisResponse(const AvbdSolverBody& b,
	const PxVec3& axis, const PxVec3& rLocal)
{
	if(!avbdBodyResponds(b))
		return 0.0f;
	PxReal w = b.invMass;
	// The contact Jacobian belongs to the live nonlinear pose while the
	// inverse-inertia metric is frozen at solve start.  Freezing the arm as
	// well conditions Position-AL with a Hessian from a different pose and is
	// especially visible on large-deformation joint graphs such as chainmail.
	const PxVec3 jAng = b.q.rotate(rLocal).cross(axis);
	const PxVec3 jAngLocal = b.qStart.rotateInv(jAng);
	w += jAngLocal.x * jAngLocal.x * b.invInertia.x;
	w += jAngLocal.y * jAngLocal.y * b.invInertia.y;
	w += jAngLocal.z * jAngLocal.z * b.invInertia.z;
	return w;
}

PX_FORCE_INLINE PxReal avbdEndpointCrossResponse(const AvbdSolverBody& b,
	const PxVec3& axis0, const PxVec3& axis1, const PxVec3& rLocal)
{
	if(!avbdBodyResponds(b))
		return 0.0f;
	const PxVec3 rWorld = b.q.rotate(rLocal);
	const PxVec3 j0Local = b.qStart.rotateInv(rWorld.cross(axis0));
	const PxVec3 j1Local = b.qStart.rotateInv(rWorld.cross(axis1));
	return b.invMass * axis0.dot(axis1) +
		j0Local.x * j1Local.x * b.invInertia.x +
		j0Local.y * j1Local.y * b.invInertia.y +
		j0Local.z * j1Local.z * b.invInertia.z;
}

PX_FORCE_INLINE PxReal avbdContactAxisResponse(const AvbdRowConstRef& row,
	const PxArray<AvbdSolverBody>& bodies, const PxVec3& axis)
{
	PxReal w = 0.0f;
	if(row.header().bodyA != PX_MAX_U32 && row.header().bodyA < bodies.size())
		w += avbdEndpointAxisResponse(bodies[row.header().bodyA], axis, row.scalar().ogcRA);
	if(row.header().bodyB != PX_MAX_U32 && row.header().bodyB < bodies.size())
		w += avbdEndpointAxisResponse(bodies[row.header().bodyB], axis, row.scalar().ogcRB);
	return (PxIsFinite(w) && w > 1.0e-12f) ? w : 0.0f;
}

PX_FORCE_INLINE PxReal avbdContactCrossResponse(const AvbdRowConstRef& row,
	const PxArray<AvbdSolverBody>& bodies, const PxVec3& axis0, const PxVec3& axis1)
{
	PxReal w = 0.0f;
	if(row.header().bodyA != PX_MAX_U32 && row.header().bodyA < bodies.size())
		w += avbdEndpointCrossResponse(bodies[row.header().bodyA], axis0, axis1, row.scalar().ogcRA);
	if(row.header().bodyB != PX_MAX_U32 && row.header().bodyB < bodies.size())
		w += avbdEndpointCrossResponse(bodies[row.header().bodyB], axis0, axis1, row.scalar().ogcRB);
	return PxIsFinite(w) ? w : 0.0f;
}

PX_FORCE_INLINE void avbdFloorContactRho(AvbdRowRef row,
	const PxArray<AvbdSolverBody>& bodies, PxReal dt)
{
	if((row.header().flags & eAVBD_ROW_CONTACT) == 0)
		return;
	if(row.header().flags & (eAVBD_ROW_INACTIVE | eAVBD_ROW_SOFT_PAIR))
		return;
	if(dt <= 0.0f)
		return;
	const PxReal w = avbdContactAxisResponse(row, bodies, row.scalar().n);
	if(w <= 0.0f)
		return;
	const PxReal floorRho = AVBD_CONTACT_CONDITIONING / (w * dt * dt);
	if(!PxIsFinite(floorRho) || floorRho <= 0.0f)
		return;
	// The reference cold-start sentinel is replaced by the spatial floor even
	// when the latter is lower.  Once adaptive AL growth has moved above the
	// sentinel, the penalty is monotone.  Using max() unconditionally makes a
	// light endpoint inherit an arbitrary 1e3 stiffness in extreme mass-ratio
	// pairs and destroys the per-body BCD conditioning below.
	if(row.scalar().rho <= AVBD_CONTACT_PENALTY_MIN || row.scalar().rho < floorRho)
		row.scalar().rho = PxMin(AVBD_CONTACT_PENALTY_MAX, floorRho);
}

PX_FORCE_INLINE void avbdFloorLoadedContactTangentRho(AvbdRowRef row,
	const PxArray<AvbdSolverBody>& bodies, PxReal dt)
{
	if((row.header().flags & eAVBD_ROW_CONTACT) == 0 || row.scalar().mu <= 0.0f ||
		(row.header().flags & (eAVBD_ROW_INACTIVE | eAVBD_ROW_SOFT_PAIR)) || dt <= 0.0f)
		return;
	const PxReal w0 = avbdContactAxisResponse(row, bodies, row.scalar().t0);
	const PxReal w1 = avbdContactAxisResponse(row, bodies, row.scalar().t1);
	if(w0 <= 0.0f || w1 <= 0.0f)
		return;
	const PxReal invH2 = 1.0f / (dt * dt);
	const PxReal floor0 = AVBD_CONTACT_CONDITIONING * invH2 / w0;
	const PxReal floor1 = AVBD_CONTACT_CONDITIONING * invH2 / w1;
	if(PxIsFinite(floor0) && floor0 > 0.0f &&
		(row.scalar().rhoT0 <= AVBD_CONTACT_PENALTY_MIN || row.scalar().rhoT0 < floor0))
		row.scalar().rhoT0 = PxMin(AVBD_CONTACT_PENALTY_MAX, floor0);
	if(PxIsFinite(floor1) && floor1 > 0.0f &&
		(row.scalar().rhoT1 <= AVBD_CONTACT_PENALTY_MIN || row.scalar().rhoT1 < floor1))
		row.scalar().rhoT1 = PxMin(AVBD_CONTACT_PENALTY_MAX, floor1);
}

PX_FORCE_INLINE void avbdApplyAxisImpulse(AvbdSolverBody& b, const PxVec3& axis,
	const PxVec3& rLocal, PxReal j)
{
	// Maximal-coordinate articulation links are ordinary 6-DOF solver
	// endpoints.  Their inbound joints share the AvbdRow program and restore
	// the admissible joint velocity in the coupled velocity sweep.
	if(!avbdBodyResponds(b))
		return;
	b.core->linearVelocity += axis * (j * b.invMass);
	b.core->angularVelocity +=
		avbdWorldInvInertiaMul(b, b.qStart.rotate(rLocal).cross(axis) * j);
	avbdLockCoreVel(*b.core);
}

PX_FORCE_INLINE void avbdApplyContactAxisImpulse(const AvbdRowConstRef& row,
	PxArray<AvbdSolverBody>& bodies, const PxVec3& axis, PxReal j)
{
	if(!PxIsFinite(j) || PxAbs(j) < 1.0e-12f)
		return;
	if(row.header().bodyA != PX_MAX_U32 && row.header().bodyA < bodies.size())
		avbdApplyAxisImpulse(bodies[row.header().bodyA], axis, row.scalar().ogcRA, j);
	if(row.header().bodyB != PX_MAX_U32 && row.header().bodyB < bodies.size())
		avbdApplyAxisImpulse(bodies[row.header().bodyB], axis, row.scalar().ogcRB, -j);
}

PX_FORCE_INLINE PxReal avbdContactAxisVel(const AvbdRowConstRef& row,
	const PxArray<AvbdSolverBody>& bodies, const PxVec3& axis, bool unconstrained)
{
	PxVec3 vA(0.0f), vB(0.0f);
	if(row.header().bodyA != PX_MAX_U32)
	{
		const AvbdSolverBody& a = bodies[row.header().bodyA];
		const PxQuat q = a.qStart;
		const PxVec3 lin = unconstrained ? a.linVel :
			(a.core ? a.core->linearVelocity : PxVec3(0.0f));
		const PxVec3 ang = unconstrained ? a.angVel :
			(a.core ? a.core->angularVelocity : PxVec3(0.0f));
		vA = avbdPointVel(lin, ang, q, row.scalar().ogcRA);
	}
	if(row.header().bodyB != PX_MAX_U32)
	{
		const AvbdSolverBody& b = bodies[row.header().bodyB];
		const PxQuat q = b.qStart;
		const PxVec3 lin = unconstrained ? b.linVel :
			(b.core ? b.core->linearVelocity : PxVec3(0.0f));
		const PxVec3 ang = unconstrained ? b.angVel :
			(b.core ? b.core->angularVelocity : PxVec3(0.0f));
		vB = avbdPointVel(lin, ang, q, row.scalar().ogcRB);
	}
	return axis.dot(vA - vB);
}

PX_FORCE_INLINE void avbdOrthonormalBasis(const PxVec3& n, PxVec3& t0, PxVec3& t1)
{
	const PxVec3 tmp = PxAbs(n.z) > 0.7071f ? PxVec3(1.0f, 0.0f, 0.0f)
											: PxVec3(0.0f, 0.0f, 1.0f);
	t0 = n.cross(tmp);
	t0.normalize();
	t1 = n.cross(t0);
}

template<class Pose>
PX_FORCE_INLINE void avbdApplyDelta(Pose& b, const PxVec3& dp,
	const PxVec3& dth)
{
	b.p += dp;
	const PxQuat dq(dth.x * 0.5f, dth.y * 0.5f, dth.z * 0.5f, 0.0f);
	b.q = b.q + dq * b.q;
	b.q.normalize();
}

PX_FORCE_INLINE void avbdProjectLockedDelta(const AvbdSolverBody& b,
	PxVec3& dp, PxVec3& dth)
{
	if(!b.core)
		return;
	const PxU32 flags = b.core->lockFlags;
	if(flags & PxRigidDynamicLockFlag::eLOCK_LINEAR_X) dp.x = 0.0f;
	if(flags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Y) dp.y = 0.0f;
	if(flags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Z) dp.z = 0.0f;
	if(flags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_X) dth.x = 0.0f;
	if(flags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y) dth.y = 0.0f;
	if(flags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z) dth.z = 0.0f;
}

template<class Pose>
PX_FORCE_INLINE PxVec3 avbdWorldPoint(const Pose& b, const PxVec3& rLocal)
{
	return b.p + b.q.rotate(rLocal);
}

// This-step slip, not t*(pA-pB). Absolute NP-anchor stick injects spin on
// large boxes (HelloWorld half-extent 2). AVBD_2 green dual uses relDisp.
template<class BodyView>
PX_FORCE_INLINE PxVec3 avbdContactRelDisp(const AvbdRowConstRef& row,
	const BodyView& bodies)
{
	PxVec3 dA(0.0f);
	PxVec3 dB(0.0f);
	if(row.header().bodyA != PX_MAX_U32)
	{
		const auto& a = bodies[row.header().bodyA];
		dA = avbdWorldPoint(a, row.scalar().rA) - (a.pStart + a.qStart.rotate(row.scalar().rA));
	}
	if(row.header().bodyB != PX_MAX_U32)
	{
		const auto& b = bodies[row.header().bodyB];
		dB = avbdWorldPoint(b, row.scalar().rB) - (b.pStart + b.qStart.rotate(row.scalar().rB));
	}
	return dA - dB;
}

PX_FORCE_INLINE PxVec3 avbdQuatLog(const PxQuat& q0, const PxQuat& q1)
{
	PxQuat qerr = q0.getConjugate() * q1;
	if(qerr.w < 0.0f)
		qerr = -qerr;
	const PxReal w = PxClamp(qerr.w, -1.0f, 1.0f);
	const PxReal s = PxSqrt(PxMax(0.0f, 1.0f - w * w));
	if(s <= 1.0e-6f)
		return PxVec3(0.0f);
	return PxVec3(qerr.x, qerr.y, qerr.z) * (2.0f * PxAtan2(s, w) / s);
}

PX_FORCE_INLINE PxVec3 avbdQuatLogWorld(const PxQuat& q0, const PxQuat& q1)
{
	// avbdApplyDelta and PhysX angular velocity use world-space increments:
	// q1 = exp(dtheta_world) * q0.
	PxQuat qerr = q1 * q0.getConjugate();
	if(qerr.w < 0.0f)
		qerr = -qerr;
	const PxReal w = PxClamp(qerr.w, -1.0f, 1.0f);
	const PxReal s = PxSqrt(PxMax(0.0f, 1.0f - w * w));
	if(s <= 1.0e-6f)
		return PxVec3(0.0f);
	return PxVec3(qerr.x, qerr.y, qerr.z) * (2.0f * PxAtan2(s, w) / s);
}

// Inverse of avbdApplyDelta's first-order left multiply q += (dth/2)*q.
// 1D residual must use this chart, not the geodesic log.
PX_FORCE_INLINE PxVec3 avbdQuatDeltaApply(const PxQuat& q0, const PxQuat& q1)
{
	PxQuat qerr = q1 * q0.getConjugate();
	if(qerr.w < 0.0f)
		qerr = -qerr;
	return PxVec3(qerr.x * 2.0f, qerr.y * 2.0f, qerr.z * 2.0f);
}

template<class Pose>
PX_FORCE_INLINE PxVec3 avbdRowAngDelta(const Pose& b)
{
	// solverPrep J_ang is world and applyDelta is left-multiply.
	// Artic inbound rows do not use this chart; they eval atan2 C live.
	return avbdQuatDeltaApply(b.qLin, b.q);
}

PX_FORCE_INLINE void avbdPoseDeltaVel(const AvbdSolverBody& b, PxReal invH,
	PxVec3& lin, PxVec3& ang)
{
	lin = (b.p - b.pStart) * invH;
	// The primal rotation update uses the first-order left quaternion chart;
	// velocity reconstruction must use its inverse chart as well.  Mixing a
	// geodesic log here turns a large constraint correction into extra angular
	// kinetic energy on the following frame.
	ang = avbdQuatDeltaApply(b.qStart, b.q) * invH;
}

template<class Parent, class Child>
PX_FORCE_INLINE bool avbdArticAxis(const Parent& parent, const Child& child,
	const ArticulationJointCore& joint, PxU32 axis,
	PxVec3& jLinA, PxVec3& jAngA, PxVec3& jLinB, PxVec3& jAngB, PxReal& C)
{
	const PxTransform poseA(parent.p, parent.q);
	const PxTransform poseB(child.p, child.q);
	const PxTransform FA = poseA * joint.parentPose;
	const PxTransform FB = poseB * joint.childPose;
	const bool linear = axis >= PxArticulationAxis::eX;
	const PxU32 component = linear ? (axis - PxArticulationAxis::eX) : axis;
	PxVec3 local(0.0f);
	local[component] = 1.0f;
	PxVec3 worldAxis = FA.q.rotate(local);
	if(worldAxis.magnitudeSquared() < 1.0e-16f)
		return false;
	if(linear)
	{
		// Three locked translations constrain anchor coincidence, not a moving
		// joint-frame coordinate.  A world orthonormal basis is the exact same
		// three-dimensional constraint with a constant Jacobian.  Using FA's
		// rotating basis while omitting d(axis)/d(thetaA) made C and J disagree
		// and injected work into revolute articulation chains.
		const bool allLinearLocked =
			joint.motion[PxArticulationAxis::eX] == PxArticulationMotion::eLOCKED &&
			joint.motion[PxArticulationAxis::eY] == PxArticulationMotion::eLOCKED &&
			joint.motion[PxArticulationAxis::eZ] == PxArticulationMotion::eLOCKED;
		if(allLinearLocked)
		{
			worldAxis = PxVec3(0.0f);
			worldAxis[component] = 1.0f;
		}
		const PxVec3 d = FB.p - FA.p;
		C = d.dot(worldAxis);
		const PxVec3 rA = FA.p - parent.p;
		const PxVec3 rB = FB.p - child.p;
		jLinA = -worldAxis;
		// With a rotating axis: d C / d thetaA = -(rA x a) + (a x d). When
		// the axis rotates with the parent (prismatic / partially locked), the
		// second term scales with the slide distance.  Omitting it made the
		// locked Y row of a prismatic drive apply a pitch torque to the parent
		// that the child never saw (lambda * stroke), lifting a scissor base.
		jAngA = rA.cross(-worldAxis);
		if(!allLinearLocked)
			jAngA += worldAxis.cross(d);
		jLinB = worldAxis;
		jAngB = rB.cross(worldAxis);
	}
	else
	{
		PxVec3 gradient;
		if(!avbdJointAngularCoordinate(FA.q, FB.q, axis, C, gradient))
			return false;
		jLinA = jLinB = PxVec3(0.0f);
		jAngA = -gradient;
		jAngB = gradient;
	}
	return PxIsFinite(C) && worldAxis.isFinite();
}

// Geometric inbound C and J at the current parent/child poses.  Linearized
// atan2 + CLin+J dx is only valid for a tiny GS step; demo3d joints eval
// C from the live pose instead.
template<class BodyView>
PX_FORCE_INLINE bool avbdEvalArticRow(const AvbdRowConstRef& row,
	const BodyView& bodies,
	PxVec3& jLinA, PxVec3& jAngA, PxVec3& jLinB, PxVec3& jAngB, PxReal& C)
{
	if(!row.scalar().articJoint)
		return false;
	if(row.header().bodyA >= bodies.size() || row.header().bodyB >= bodies.size())
		return false;
	if(!avbdArticAxis(bodies[row.header().bodyA], bodies[row.header().bodyB], *row.scalar().articJoint,
		row.scalar().articAxis, jLinA, jAngA, jLinB, jAngB, C))
		return false;
	if(row.scalar().articFlip)
	{
		jLinA = -jLinA;
		jAngA = -jAngA;
		jLinB = -jLinB;
		jAngB = -jAngB;
		C = row.scalar().articTarget - C;
	}
	else
		C = C - row.scalar().articTarget;
	return PxIsFinite(C);
}

template<class BodyView>
PX_FORCE_INLINE PxReal avbdEvalC(const AvbdRowConstRef& row, const BodyView& bodies)
{
	if(row.header().flags & eAVBD_ROW_INACTIVE)
		return 0.0f;
	if(row.scalar().articJoint)
	{
		PxVec3 jLinA, jAngA, jLinB, jAngB;
		PxReal C = 0.0f;
		if(!avbdEvalArticRow(row, bodies, jLinA, jAngA, jLinB, jAngB, C))
			return 0.0f;
		return C;
	}
	if(row.header().flags & eAVBD_ROW_CONTACT)
	{
		PxVec3 pA = row.scalar().ogcRA;
		PxVec3 pB = row.scalar().ogcRB;
		if(row.header().bodyA != PX_MAX_U32)
			pA = avbdWorldPoint(bodies[row.header().bodyA], row.scalar().ogcRA);
		if(row.header().bodyB != PX_MAX_U32)
			pB = avbdWorldPoint(bodies[row.header().bodyB], row.scalar().ogcRB);
		return row.scalar().n.dot(pA - pB) + row.scalar().contactOffset;
	}
	if(row.header().flags & eAVBD_ROW_POINT_LOCK)
	{
		PxVec3 pA = row.scalar().rA;
		PxVec3 pB = row.scalar().rB;
		if(row.header().bodyA != PX_MAX_U32 && row.header().bodyA < bodies.size())
			pA = avbdWorldPoint(bodies[row.header().bodyA], row.scalar().rA);
		if(row.header().bodyB != PX_MAX_U32 && row.header().bodyB < bodies.size())
			pB = avbdWorldPoint(bodies[row.header().bodyB], row.scalar().rB);
		return row.scalar().n.dot(pA - pB);
	}
	if(row.header().flags & eAVBD_ROW_DRIVE)
	{
		PxReal positionC = row.scalar().CLin;
		PxReal velocityC = -row.scalar().driveVelocityTarget;
		if(row.header().bodyA != PX_MAX_U32 && row.header().bodyA < bodies.size())
		{
			const auto& a = bodies[row.header().bodyA];
			positionC += row.scalar().jLinA.dot(a.p - a.pLin);
			positionC += row.scalar().jAngA.dot(avbdRowAngDelta(a));
			velocityC += row.scalar().jLinA.dot(a.p - a.pStart);
			velocityC += row.scalar().jAngA.dot(avbdQuatDeltaApply(a.qStart, a.q));
		}
		if(row.header().bodyB != PX_MAX_U32 && row.header().bodyB < bodies.size())
		{
			const auto& b = bodies[row.header().bodyB];
			positionC += row.scalar().jLinB.dot(b.p - b.pLin);
			positionC += row.scalar().jAngB.dot(avbdRowAngDelta(b));
			velocityC += row.scalar().jLinB.dot(b.p - b.pStart);
			velocityC += row.scalar().jAngB.dot(avbdQuatDeltaApply(b.qStart, b.q));
		}
		if(row.header().kernelId == eAVBD_ROW_KERNEL_FINITE_SPRING)
			return avbdFiniteSpringResidual(row.scalar().driveStiffness,
				row.scalar().driveDampingOverDt, positionC, row.scalar().C0, row.scalar().driveVelocityTarget);
		const PxReal total = row.scalar().driveStiffness + row.scalar().driveDampingOverDt;
		return total > 0.0f ?
			(row.scalar().driveStiffness * positionC +
			 row.scalar().driveDampingOverDt * velocityC) / total : 0.0f;
	}
	if(row.scalar().integratedAngular)
	{
		PxReal C = row.scalar().C0;
		if(row.header().bodyA != PX_MAX_U32 && row.header().bodyA < bodies.size())
			C += row.scalar().jAngA.dot(avbdQuatDeltaApply(
				bodies[row.header().bodyA].qStart, bodies[row.header().bodyA].q));
		if(row.header().bodyB != PX_MAX_U32 && row.header().bodyB < bodies.size())
			C += row.scalar().jAngB.dot(avbdQuatDeltaApply(
				bodies[row.header().bodyB].qStart, bodies[row.header().bodyB].q));
		return C;
	}
	PxReal C = row.scalar().CLin;
	if(row.header().bodyA != PX_MAX_U32 && row.header().bodyA < bodies.size())
	{
		const auto& a = bodies[row.header().bodyA];
		C += row.scalar().jLinA.dot(a.p - a.pLin);
		C += row.scalar().jAngA.dot(avbdRowAngDelta(a));
	}
	if(row.header().bodyB != PX_MAX_U32 && row.header().bodyB < bodies.size())
	{
		const auto& b = bodies[row.header().bodyB];
		C += row.scalar().jLinB.dot(b.p - b.pLin);
		C += row.scalar().jAngB.dot(avbdRowAngDelta(b));
	}
	return C;
}

// Hard inbound rows retain the immutable solve-start error (demo3d Eq. 14).
// A drive is an authored spring and must not relax its target error.
PX_FORCE_INLINE PxReal avbdArticPrimalC(const AvbdRowConstRef& row, PxReal C)
{
	return (row.header().flags & eAVBD_ROW_DRIVE) ? C : C - AVBD_ALPHA * row.scalar().C0;
}

// demo3d manifold.cpp: C = C0*(1-alpha) + J*dq. Alpha softens only an
// existing penetration (demo3d contacts start at C0 <= margin).  A speculative
// NP point with positive solve-start gap keeps C = gap so the row rests at the
// authored rest distance; softening a positive C0 would pin the body at
// alpha*gap and make it creep down at 1% per step.
template<class BodyView>
PX_FORCE_INLINE PxReal avbdPrimalC(const AvbdRowConstRef& row,
	const BodyView& bodies, PxReal /*lengthScale*/)
{
	const PxReal C = avbdEvalC(row, bodies);
	if(row.header().flags & eAVBD_ROW_SOFT_PAIR)
		return C;
	if(row.scalar().articJoint)
		return avbdArticPrimalC(row, C);
	if((row.header().flags & eAVBD_ROW_CONTACT) == 0)
		return C;
	if(!row.header().supportCount && row.scalar().contactRadius>0) return C;
	return C - AVBD_ALPHA * PxMin(row.scalar().C0, 0.0f);
}

} // namespace Dy
} // namespace physx

#endif
