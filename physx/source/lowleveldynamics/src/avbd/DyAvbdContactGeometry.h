// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_CONTACT_GEOMETRY_H
#define DY_AVBD_CONTACT_GEOMETRY_H

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "avbd/DyAvbdOgcProjection.h"

namespace physx
{
namespace Dy
{
// One arithmetic contract for a contact's compiled domain and final admission.
// gap is measured from the authored plane, never from an already tolerated
// boundary point. Recentring there would spend the input interval twice.
struct AvbdContactAdmissionState
{
 double gap, baseScale, pointScale;
 AvbdContactAdmissionState(PxReal offset, double admissionOffset, PxReal initialGap)
 : gap(double(offset)+admissionOffset),
   baseScale(PxAbs(double(offset))+PxAbs(admissionOffset)+PxAbs(double(initialGap))),
   pointScale(0.0) {}
 void add(PxReal weight, const PxVec3& normal, const PxVec3& point)
 {
  for(PxU32 axis=0;axis<3;++axis)
  {
   const double coefficient=double(weight)*double(normal[axis]);
   gap+=coefficient*double(point[axis]);
   pointScale+=PxAbs(coefficient)*PxAbs(double(point[axis]));
  }
 }
 double tolerance(const AvbdContactAdmissionState& other) const
 { return 4.0*double(PX_EPS_F32)*(baseScale+pointScale+other.pointScale); }
};

// Admission must see the same PxReal point before and after publication.
// Fast-math inlining can reassociate q.rotate(local) + p differently in
// Particle3 and Rigid6 callers. Keep this small transform out of those
// expression trees and give it one evaluation order.
#if PX_VC
#pragma float_control(precise, on, push)
#endif
PX_NOINLINE inline PxVec3 avbdContactWorldPoint(const PxVec3& p,
 const PxQuat& q, const PxVec3& local)
{
 const PxReal vx=2.0f*local.x,vy=2.0f*local.y,vz=2.0f*local.z;
 const PxReal w2=q.w*q.w-0.5f;
 const PxReal dot2=q.x*vx+q.y*vy+q.z*vz;
 return PxVec3(p.x+(vx*w2+(q.y*vz-q.z*vy)*q.w+q.x*dot2),
  p.y+(vy*w2+(q.z*vx-q.x*vz)*q.w+q.y*dot2),
  p.z+(vz*w2+(q.x*vy-q.y*vx)*q.w+q.z*dot2));
}
#if PX_VC
#pragma float_control(pop)
#endif

// Payload of a rigid-carried contact plane. Row bodyA is its frame and the
// signed support set supplies its local point and embedded opposite point.
struct AvbdContactPlaneGeometry
{
	PxVec3 normalLocal;
};

// DCD witness coordinates and a normal carried by an actual target triangle.
// Five affine points: query witness, target witness, and frame vertices a/b/c.
// Embedding weights keep their signs; endpoint roles never follow weight sign.
struct AvbdContactFrameGeometry
{
 PxVec3 normalLocal;
 PxU16 supportEnds[5];
};

struct AvbdContactFrameEvaluation
{
 double gap;
 PxVec3d normal;
 PxVec3d gradients[5];
 bool valid;
};

PX_INLINE bool avbdContactFrameBasis(const PxVec3d* points,
 PxVec3d& t,PxVec3d& y,PxVec3d& z,double& edgeLength,double& crossLength)
{
 const PxVec3d edge=points[3]-points[2],other=points[4]-points[2];
 edgeLength=edge.magnitude();
 if(!PxIsFinite(edgeLength) || edgeLength<=0.0) return false;
 t=edge/edgeLength;
 const PxVec3d cross=t.cross(other);
 crossLength=cross.magnitude();
 if(!PxIsFinite(crossLength) || crossLength<=0.0) return false;
 z=cross/crossLength;y=z.cross(t);
 return t.isFinite() && y.isFinite() && z.isFinite();
}

PX_INLINE AvbdContactFrameEvaluation avbdEvaluateContactFrame(
 const AvbdContactFrameGeometry& geometry,const PxVec3d* points,double offset)
{
 AvbdContactFrameEvaluation result={};
 for(PxU32 i=0;i<5;++i) if(!points[i].isFinite()) return result;
 if(!geometry.normalLocal.isFinite() || !PxIsFinite(offset)) return result;
 PxVec3d t,y,z;double edgeLength,crossLength;
 if(!avbdContactFrameBasis(points,t,y,z,edgeLength,crossLength)) return result;
 const PxVec3d local=avbdOgcDouble(geometry.normalLocal);
 const PxVec3d normal=t*local.x+y*local.y+z*local.z;
 const PxVec3d difference=points[0]-points[1];
 result.gap=normal.dot(difference)+offset;
 result.normal=normal;
 result.gradients[0]=normal;result.gradients[1]=-normal;
 // Reverse differentiation of y=z cross t, z=normalize(t cross (c-a)),
 // t=normalize(b-a). Keeping these frame terms is essential for zero torque.
 const PxVec3d gy=difference*local.y;
 const PxVec3d gz=difference*local.z+t.cross(gy);
 PxVec3d gt=difference*local.x+gy.cross(z);
 const PxVec3d gcross=(gz-z*z.dot(gz))/crossLength;
 gt+=(points[4]-points[2]).cross(gcross);
 const PxVec3d gother=gcross.cross(t);
 const PxVec3d gedge=(gt-t*t.dot(gt))/edgeLength;
 result.gradients[2]=-gedge-gother;
 result.gradients[3]=gedge;result.gradients[4]=gother;
 result.valid=PxIsFinite(result.gap) && result.normal.isFinite();
 for(PxU32 i=0;i<5;++i) result.valid &= result.gradients[i].isFinite();
 return result;
}

template<class Support,class PointAt>
PX_INLINE void avbdContactFramePoints(const AvbdContactFrameGeometry& geometry,
 const Support* supports,const PointAt& pointAt,PxVec3d (&points)[5])
{
 PxU32 start=0;
 for(PxU32 point=0;point<5;++point)
 {
  points[point]=PxVec3d(0.0);
  for(PxU32 i=start;i<geometry.supportEnds[point];++i)
   points[point]+=avbdOgcDouble(pointAt(supports[i]))*double(supports[i].weight);
  start=geometry.supportEnds[point];
 }
}

template<class Support>
PX_INLINE PxVec3d avbdContactFrameBlockGradient(const AvbdContactFrameGeometry& geometry,
 const Support* supports,const AvbdContactFrameEvaluation& evaluation,PxU32 block)
{
 PxVec3d gradient(0.0);PxU32 start=0;
 for(PxU32 point=0;point<5;++point)
 {
  for(PxU32 i=start;i<geometry.supportEnds[point];++i)
   if(supports[i].block==block) gradient+=evaluation.gradients[point]*double(supports[i].weight);
  start=geometry.supportEnds[point];
 }
 return gradient;
}

// One full pair certificate, with six affine points in the row's shared
// support array. It is a domain of the physical objective, not nine springs.
struct AvbdTriangleDomainGeometry
{
	PxVec3 normal;
	PxReal lowerGap;
	PxU16 supportEnds[6];
	PxU32 pointIndices[6]; // shared committed evaluations, local primal only
	PxU32 projectionIndices[6]; // point/normal factors of this certificate
};

struct AvbdTrianglePointProjection
{
 PxVec3 normal;
 PxU32 pointIndex;
 double value, scale;
};

struct AvbdTriangleProjectionRef { PxU32 projection; double weight,absoluteWeight; };

// A sufficient certificate for a whole batch while supports move away from
// a snapshot. Drift is observed, never constrained: an uncertain certificate
// falls through to live point factors and then complete candidate evaluation.
struct AvbdTriangleBatchBound
{
 double gap,scale,motionCoefficient,localCoefficient;
 bool admits(double motion,double localMotion) const
 {
  const double drift=motionCoefficient*motion+localCoefficient*localMotion;
  const double error=128.0*DBL_EPSILON*(scale+drift)+DBL_MIN;
  return PxIsFinite(drift) && PxIsFinite(gap) && PxIsFinite(error) &&
   gap>(1.0+4.0*FLT_EPSILON)*drift+error;
 }
};

// A sufficient, never necessary, test for all nine exact affine conditions.
// Factoring dot(n,a-b) into dot(n,a)-dot(n,b) changes double rounding. The
// bound covers both evaluation orders, signed embedding and local motion;
// uncertain/boundary cases must use the original nine-condition kernel.
PX_INLINE bool avbdTriangleDomainSeparated(const AvbdTriangleDomainGeometry& geometry,
 const AvbdTrianglePointProjection* projections,const double* weights,
 const PxVec3d& delta,const AvbdTriangleProjectionRef* refs=NULL,
 PxU32 countA=3,PxU32 count=6,const double* absoluteWeights=NULL,
 bool inputIntervals=false)
{
 const PxVec3d normal(geometry.normal.x,geometry.normal.y,geometry.normal.z);
 const double motion=normal.dot(delta), motionScale=normal.abs().dot(delta.abs());
 double minimum=DBL_MAX,maximum=-DBL_MAX,scaleA=0.0,scaleB=0.0;
 for(PxU32 i=0;i<count;++i)
 {
  const AvbdTrianglePointProjection& point=projections[refs ? refs[i].projection : geometry.projectionIndices[i]];
  const double weight=refs ? refs[i].weight : weights[i];
  const double absoluteWeight=inputIntervals ? (refs ? refs[i].absoluteWeight : absoluteWeights[i]) : PxAbs(weight);
  const double scale=point.scale+absoluteWeight*motionScale;
  // Triangle inequality gives a LOWER bound on the actual candidate's
  // absolute input scale, including duplicate signed supports. Consuming only
  // this proven part of the original interval cannot enlarge admission.
  const double lowerScale=PxMax(0.0,point.scale-absoluteWeight*motionScale-64.0*DBL_EPSILON*scale);
  const double allowance=inputIntervals ? 4.0*FLT_EPSILON*lowerScale : 0.0;
  const double value=point.value+weight*motion+(i<countA ? allowance : -allowance);
  if(!PxIsFinite(value) || !PxIsFinite(scale)) return false;
  if(i<countA) { minimum=PxMin(minimum,value);scaleA=PxMax(scaleA,scale); }
  else { maximum=PxMax(maximum,value);scaleB=PxMax(scaleB,scale); }
 }
 const double gap=minimum-maximum-double(geometry.lowerGap)+
  (inputIntervals ? 4.0*FLT_EPSILON*PxAbs(double(geometry.lowerGap)) : 0.0);
 const double error=64.0*DBL_EPSILON*(scaleA+scaleB+PxAbs(double(geometry.lowerGap)))+DBL_MIN;
 return PxIsFinite(gap) && PxIsFinite(error) && gap>error;
}

// Derived values of an affine row point. Writers recompute the original sum;
// readers never mutate this storage. Incidence is fixed for the DCD step.
struct AvbdAffinePointEvaluation
{
 PxVec3d point, scale;
 PxU32 supportStart, supportCount;
 double absoluteWeight;
};

template<class Support, class PointAt>
PX_INLINE void avbdEvaluateAffinePoint(const Support* supports, PxU32 count,
 const PointAt& pointAt, AvbdAffinePointEvaluation& result)
{
 result.point=result.scale=PxVec3d(0.0);
 for(PxU32 i=0;i<count;++i)
 {
  const PxVec3 position=pointAt(supports[i]);
  const PxVec3d p(position.x,position.y,position.z);
  result.point+=p*double(supports[i].weight);
  result.scale+=p.abs()*PxAbs(double(supports[i].weight));
 }
}

template<class Support, class PointAt>
PX_INLINE void avbdTriangleDomainPoints(const AvbdTriangleDomainGeometry& geometry,
 const Support* supports, const PointAt& pointAt, PxU32 block,
 PxVec3d (&points)[6], PxVec3d (&scales)[6], double (&weights)[6],
 const AvbdAffinePointEvaluation* committed=NULL, const double* localWeights=NULL)
{
 PxU32 start=0;
 for(PxU32 point=0;point<6;++point)
 {
  if(committed)
  {
   const AvbdAffinePointEvaluation& value=committed[geometry.pointIndices[point]];
   points[point]=value.point;scales[point]=value.scale;
   weights[point]=localWeights[point];
   start=geometry.supportEnds[point];
   continue;
  }
  points[point]=scales[point]=PxVec3d(0.0);weights[point]=0.0;
  for(PxU32 i=start;i<geometry.supportEnds[point];++i)
  {
   const Support& support=supports[i];
   const PxVec3 position=pointAt(support);
   const PxVec3d p(position.x,position.y,position.z);
   points[point]+=p*double(support.weight);
   scales[point]+=p.abs()*PxAbs(double(support.weight));
   if(support.block==block) weights[point]+=double(support.weight);
  }
  start=geometry.supportEnds[point];
 }
}

// The local coefficient depends only on fixed DCD normals and embedding
// weights. Bounds and input scales still depend on live or candidate points.
PX_INLINE PxVec3d avbdTriangleDomainCoefficient(const PxVec3d& normal,
 const double* weights,PxU32 a,PxU32 b)
{ return normal*(weights[a]-weights[b]); }

// Exact affine local restrictions for one coordinate block. All other
// supports have their live committed positions. Nine scalar conditions
// represent complete projection separation, including signed embeddings.
PX_INLINE void avbdTriangleDomainPlanes(const AvbdTriangleDomainGeometry& geometry,
 const PxVec3d* points,const PxVec3d* scales,const double* weights,
 AvbdOgcProjectionPlane* planes)
{
 const PxVec3d normal(geometry.normal.x,geometry.normal.y,geometry.normal.z);
 for(PxU32 a=0;a<3;++a) for(PxU32 b=3;b<6;++b)
 {
  AvbdOgcProjectionPlane& plane=planes[a*3+b-3];
  plane.n=avbdTriangleDomainCoefficient(normal,weights,a,b);
  plane.b=double(geometry.lowerGap)-normal.dot(points[a]-points[b]);
  plane.roundoff=4.0*FLT_EPSILON*(normal.abs().dot(scales[a]+scales[b])+PxAbs(double(geometry.lowerGap)));
 }
}

// These raw current-relative conditions are stricter than the normalized
// projection planes' anchor roundoff. Reuse them only inside one block solve,
// while all positions are unchanged, and retain the exact acceptance test.
PX_INLINE bool avbdTriangleDomainAdmits(const AvbdOgcProjectionPlane* planes,
 PxU32 count,const PxVec3d& delta)
{
 for(PxU32 i=0;i<count;++i)
  if(planes[i].n.dot(delta)<planes[i].b-planes[i].roundoff) return false;
 return true;
}

struct AvbdContactPlaneEvaluation
{
	PxReal gap;
	PxVec3 normal;
	PxVec3 rigidAngularGradient;
};

struct AvbdTrianglePairPlane
{
	PxVec3 normal;
	PxReal gap;
};

PX_INLINE AvbdTrianglePairPlane avbdPointSetsPlane(const PxVec3* a,PxU32 na,
 const PxVec3* b,PxU32 nb,const PxVec3& axis)
{
 AvbdTrianglePairPlane result={PxVec3(0.0f),-PX_MAX_F32};
 if(!na || !nb || !axis.isFinite() || axis.magnitudeSquared()<=1e-20f) return result;
 const PxVec3 normal=axis.getNormalized();
 const PxVec3d n(normal.x,normal.y,normal.z);
 double amin=DBL_MAX,amax=-DBL_MAX,bmin=DBL_MAX,bmax=-DBL_MAX;
 for(PxU32 i=0;i<na;++i)
 { const double p=n.dot(PxVec3d(a[i].x,a[i].y,a[i].z));amin=PxMin(amin,p);amax=PxMax(amax,p); }
 for(PxU32 i=0;i<nb;++i)
 { const double p=n.dot(PxVec3d(b[i].x,b[i].y,b[i].z));bmin=PxMin(bmin,p);bmax=PxMax(bmax,p); }
 if(amin-bmax>=bmin-amax) { result.normal=normal;result.gap=PxReal(amin-bmax); }
 else { result.normal=-normal;result.gap=PxReal(bmin-amax); }
 return result;
}

// A positive minimum over all nine signed vertex differences separates both
// convex hulls. SAT axes handle crossing/coplanar triangles; the closest-point
// direction supplies the Euclidean separating direction for disjoint pairs.
PX_INLINE AvbdTrianglePairPlane avbdTrianglePairPlane(const PxVec3 (&a)[3],
	const PxVec3 (&b)[3],const PxVec3& closestDirection)
{
	AvbdTrianglePairPlane result={PxVec3(0,1,0),-PX_MAX_F32};
	auto consider=[&](const PxVec3& axis)
	{
		if(!axis.isFinite() || axis.magnitudeSquared()<=1.0e-20f) return;
		const PxVec3 n=axis.getNormalized();
		PxReal lower=PX_MAX_F32,upper=-PX_MAX_F32;
		for(PxU32 i=0;i<3;++i) for(PxU32 j=0;j<3;++j)
		{
			const PxReal gap=n.dot(a[i]-b[j]);
			lower=PxMin(lower,gap); upper=PxMax(upper,gap);
		}
		if(lower>result.gap) { result.normal=n; result.gap=lower; }
		if(-upper>result.gap) { result.normal=-n; result.gap=-upper; }
	};
	consider(closestDirection);
	const PxVec3 an=(a[1]-a[0]).cross(a[2]-a[0]);
	const PxVec3 bn=(b[1]-b[0]).cross(b[2]-b[0]);
	consider(an); consider(bn);
	for(PxU32 i=0;i<3;++i)
	{
		const PxVec3 ae=a[(i+1)%3]-a[i],be=b[(i+1)%3]-b[i];
		consider(an.cross(ae)); consider(bn.cross(be));
		for(PxU32 j=0;j<3;++j) consider(ae.cross(b[(j+1)%3]-b[j]));
	}
	return result;
}

// Complete triangle/OBB projection separation. A positive gap separates
// both convex hulls, independently of which vertex realizes the minimum.
PX_INLINE AvbdTrianglePairPlane avbdTriangleBoxPlane(const PxVec3 triangle[3],
 const PxVec3& center,const PxQuat& rotation,const PxVec3& halfExtent)
{
 PxVec3 local[3];
 for(PxU32 i=0;i<3;++i) local[i]=rotation.rotateInv(triangle[i]-center);
 AvbdTrianglePairPlane result;
 result.normal=PxVec3(0.0f); result.gap=-PX_MAX_F32;
 auto consider=[&](const PxVec3& axis)
 {
  if(!axis.isFinite() || axis.magnitudeSquared()<=1.0e-20f) return;
  const PxVec3 n=axis.getNormalized();
  const PxReal support=n.abs().dot(halfExtent);
  PxReal lower=PX_MAX_F32,upper=-PX_MAX_F32;
  for(PxU32 i=0;i<3;++i)
  { const PxReal p=n.dot(local[i]); lower=PxMin(lower,p); upper=PxMax(upper,p); }
  if(lower-support>result.gap) { result.normal=n; result.gap=lower-support; }
  if(-upper-support>result.gap) { result.normal=-n; result.gap=-upper-support; }
 };
 const PxVec3 axes[3]={PxVec3(1,0,0),PxVec3(0,1,0),PxVec3(0,0,1)};
 for(PxU32 axis=0;axis<3;++axis) consider(axes[axis]);
 consider((local[1]-local[0]).cross(local[2]-local[0]));
 for(PxU32 i=0;i<3;++i)
  for(PxU32 axis=0;axis<3;++axis) consider((local[(i+1)%3]-local[i]).cross(axes[axis]));
 result.normal=rotation.rotate(result.normal);
 return result;
}

PX_FORCE_INLINE AvbdContactPlaneEvaluation avbdEvaluateContactPlane(
	const PxVec3& normalLocal, const PxVec3& pointLocal,
	const PxVec3& position, const PxQuat& rotation,
	const PxVec3& oppositePoint, PxReal margin)
{
	AvbdContactPlaneEvaluation value;
	value.normal = rotation.rotate(normalLocal);
	const PxVec3 arm = oppositePoint - position;
	value.gap = value.normal.dot(arm - rotation.rotate(pointLocal)) - margin;
	// Includes rotation of the normal as well as the plane support point.
	value.rigidAngularGradient = value.normal.cross(arm);
	return value;
}

PX_FORCE_INLINE PxU32 avbdBoxSupportCorner(const PxVec3& normalLocal)
{
	return (normalLocal.x >= 0.0f ? 1u : 0u) |
		(normalLocal.y >= 0.0f ? 2u : 0u) |
		(normalLocal.z >= 0.0f ? 4u : 0u);
}

PX_FORCE_INLINE PxVec3 avbdBoxCornerLocal(const PxVec3& halfExtent, PxU32 index)
{
	return PxVec3((index & 1) ? halfExtent.x : -halfExtent.x,
		(index & 2) ? halfExtent.y : -halfExtent.y,
		(index & 4) ? halfExtent.z : -halfExtent.z);
}

// If all emitted n.(triangleVertex-boxVertex) are positive, the projected
// convex hull intervals are disjoint. No centroid or fixed-rotation surrogate
// can replace the support set, even when several initial distances coincide.
template<class Emit>
PX_FORCE_INLINE void avbdForEachTriangleBoxPlaneSupport(const PxVec3& halfExtent,
	const Emit& emit)
{
	for(PxU32 triangleVertex = 0; triangleVertex < 3; ++triangleVertex)
		for(PxU32 boxVertex = 0; boxVertex < 8; ++boxVertex)
			emit(triangleVertex, boxVertex, avbdBoxCornerLocal(halfExtent, boxVertex));
}
} // namespace Dy
} // namespace physx
#endif
