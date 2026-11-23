// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_OGC_PROJECTION_H
#define DY_AVBD_OGC_PROJECTION_H

#include "foundation/PxInlineArray.h"
#include "foundation/PxVec3.h"
#include "foundation/PxMat33.h"
#include "foundation/PxSort.h"
#include <cfloat>
#include <cmath>

namespace physx { namespace Dy {

struct AvbdOgcProjectionPlane
{
	PxVec3d n;
	double b; // n.x >= b, in coordinates relative to the DCD anchor
	double roundoff = 0.0;
};
typedef PxInlineArray<AvbdOgcProjectionPlane, 8> AvbdOgcProjectionPlanes;

PX_INLINE PxVec3d avbdOgcDouble(const PxVec3& v)
{ return PxVec3d(v.x, v.y, v.z); }

// /fp:fast may eliminate a float cast promoted back to double. Geometric
// admission depends on the ACTUAL stored value, so make this rounding point
// observable. Only the quantization boundary needs storage, not the solver.
PX_INLINE PxReal avbdOgcStoredFloat(double value)
{
	volatile PxReal stored = PxReal(value);
	return stored;
}

PX_INLINE PxVec3 avbdOgcPublishPoint(const PxVec3& anchor,const PxVec3d& displacement)
{
	const PxVec3d absolute=avbdOgcDouble(anchor)+displacement;
	PxVec3 result(avbdOgcStoredFloat(absolute.x),avbdOgcStoredFloat(absolute.y),avbdOgcStoredFloat(absolute.z));
	// Round toward the ball center if nearest rounding increased a component
	// of displacement. This is at most one float ULP, not a geometric step cap.
	for(PxU32 axis=0;axis<3;++axis)
		if(PxAbs(double(result[axis])-anchor[axis])>PxAbs(displacement[axis]))
			result[axis]=std::nextafter(result[axis],anchor[axis]);
	return result;
}

PX_INLINE double avbdOgcProjectionTolerance(const AvbdOgcProjectionPlane& p,
	const PxVec3d& x)
{
	return p.roundoff + 32.0 * DBL_EPSILON * (1.0 + PxAbs(p.b) +
		PxAbs(p.n.x * x.x) + PxAbs(p.n.y * x.y) + PxAbs(p.n.z * x.z));
}

PX_INLINE bool avbdOgcRegionContains(const AvbdOgcProjectionPlanes& planes,
	const PxVec3d& x, double radius)
{
	if(!x.isFinite() || !PxIsFinite(radius) || radius < 0.0) return false;
	for(PxU32 i=0; i<planes.size(); ++i)
		if(!planes[i].n.isFinite() || !PxIsFinite(planes[i].b) ||
			!PxIsFinite(planes[i].roundoff) || planes[i].roundoff<0.0 ||
			planes[i].b - planes[i].n.dot(x) > avbdOgcProjectionTolerance(planes[i],x))
			return false;
	return radius >= double(PX_MAX_F32)*0.5 ||
		x.magnitudeSquared() <= radius*radius*(1.0+16.0*FLT_EPSILON);
}

// A continuous feasible point need not stay feasible under one selected float
// rounding direction. Inspect the vertices of its containing float cell using
// the original domain, without enlarging its input-error intervals.
PX_INLINE bool avbdOgcPublishRegionPoint(const PxVec3& anchor,
 const PxVec3d& displacement,const AvbdOgcProjectionPlanes& planes,
 double radius,PxVec3& result,const PxVec3* feasiblePoint=NULL)
{
 const PxVec3d absolute=avbdOgcDouble(anchor)+displacement;
 // Only a finite ball requires inward rounding. With no radius restriction,
 // truncating every coordinate toward the anchor can erase a descent step.
 const PxVec3 preferred=radius>=double(PX_MAX_F32)*.5?
  PxVec3(avbdOgcStoredFloat(absolute.x),avbdOgcStoredFloat(absolute.y),avbdOgcStoredFloat(absolute.z)):
  avbdOgcPublishPoint(anchor,displacement);
 if(avbdOgcRegionContains(planes,avbdOgcDouble(preferred)-avbdOgcDouble(anchor),radius))
 { result=preferred; return true; }
 PxReal lower[3],upper[3];
 for(PxU32 axis=0;axis<3;++axis)
 {
  const PxReal nearest=avbdOgcStoredFloat(absolute[axis]);
  lower[axis]=double(nearest)>absolute[axis] ? std::nextafter(nearest,-PX_MAX_F32) : nearest;
  upper[axis]=double(nearest)<absolute[axis] ? std::nextafter(nearest,PX_MAX_F32) : nearest;
 }
 bool found=false;
 double best=DBL_MAX;
 for(PxU32 corner=0;corner<8;++corner)
 {
  const PxVec3 candidate((corner&1)?upper[0]:lower[0],
   (corner&2)?upper[1]:lower[1],(corner&4)?upper[2]:lower[2]);
  const PxVec3d delta=avbdOgcDouble(candidate)-avbdOgcDouble(anchor);
  if(!avbdOgcRegionContains(planes,delta,radius)) continue;
  const double error=(delta-displacement).magnitudeSquared();
  if(error<best) { best=error; result=candidate; found=true; }
 }
 if(found) return true;
 // A feasible continuous optimum can have no feasible vertex in its float
 // cell, even though an already committed float point exists in this domain.
 // Approach that verified point within the same convex domain. This changes
 // only the quantized local proposal; primal still checks its actual descent
 // and energy, and a zero step is not a stationarity certificate.
 if(feasiblePoint)
 {
  const PxVec3d feasible=avbdOgcDouble(*feasiblePoint)-avbdOgcDouble(anchor);
  if(avbdOgcRegionContains(planes,feasible,radius))
   for(double fraction=DBL_EPSILON;fraction<=1.0;fraction*=2.0)
   {
    const PxVec3d trial=feasible+(displacement-feasible)*(1.0-fraction);
    if(avbdOgcPublishRegionPoint(anchor,trial,planes,radius,result)) return true;
   }
 }
 return false;
}

// A local affine domain can contain many different physical features with
// exactly the same coefficient. Keep its strongest represented boundary and
// strongest input-interval boundary. This is local algebraic implication,
// never a merge of feature identities or approximately parallel normals.
PX_INLINE void avbdOgcCompactParallelPlanes(AvbdOgcProjectionPlanes& planes)
{
 if(planes.size()<2) return;
 auto less=[](const AvbdOgcProjectionPlane& a,const AvbdOgcProjectionPlane& b)->bool {
  if(a.n.x!=b.n.x) return a.n.x<b.n.x;
  if(a.n.y!=b.n.y) return a.n.y<b.n.y;
  return a.n.z<b.n.z;
 };
 // Invalid input must remain visible to the verifier, and NaNs cannot be
 // passed to a strict sorting comparator.
 for(PxU32 i=0;i<planes.size();++i)
  if(!planes[i].n.isFinite() || !PxIsFinite(planes[i].b) ||
   !PxIsFinite(planes[i].roundoff) || planes[i].roundoff<0.0) return;
 PxSort(planes.begin(),planes.size(),less);
 PxU32 written=0;
 for(PxU32 start=0;start<planes.size();)
 {
  PxU32 end=start+1,strict=start,interval=start;
  auto lower=[](const AvbdOgcProjectionPlane& p)->double {
   return p.b-p.roundoff-32.0*DBL_EPSILON*(1.0+PxAbs(p.b));
  };
  for(;end<planes.size() && planes[end].n==planes[start].n;++end)
  {
   if(planes[end].b>planes[strict].b) strict=end;
   if(lower(planes[end])>lower(planes[interval])) interval=end;
  }
  const AvbdOgcProjectionPlane a=planes[strict],b=planes[interval];
  planes[written++]=a;
  if(interval!=strict) planes[written++]=b;
  start=end;
 }
 planes.forceSize_Unsafe(written);
}

// Reduction for a precompiled group of exactly equal coefficients. The
// original represented and input-interval bounds are both retained; callers
// keep invalid planes outside the reduction so verification still rejects them.
struct AvbdOgcParallelPlaneBounds
{
 AvbdOgcProjectionPlane strict, interval;
 bool populated;
 AvbdOgcParallelPlaneBounds() : populated(false) {}
 void include(const AvbdOgcProjectionPlane& plane)
 {
  auto lower=[](const AvbdOgcProjectionPlane& p)->double
  { return p.b-p.roundoff-32.0*DBL_EPSILON*(1.0+PxAbs(p.b)); };
  if(!populated) { strict=interval=plane;populated=true;return; }
  if(plane.b>strict.b) strict=plane;
  if(lower(plane)>lower(interval)) interval=plane;
 }
 void append(AvbdOgcProjectionPlanes& planes) const
 {
  if(!populated) return;
  planes.pushBack(strict);
  if(interval.b!=strict.b || interval.roundoff!=strict.roundoff)
   planes.pushBack(interval);
 }
};

// Dual active-set projection of y onto an intersection of affine half-spaces.
// x = y + sum(lambda_i*n_i), lambda_i >= 0. The working normals are independent,
// hence at most three. Dropping a zero dual frees a direction; a dependent
// violated plane with no dual blocking step certifies an empty intersection.
// The iteration bound is a failure bound, never a feasibility certificate.
PX_INLINE bool avbdOgcProjectHalfSpacesFast(const AvbdOgcProjectionPlanes& planes,
	const PxVec3d& y, PxVec3d& result, const PxMat33d& inverseMetric)
{
	PxVec3d x = y;
	if(!avbdOgcRegionContains(planes,y,double(PX_MAX_F32)))
		for(PxU32 i=0;i<planes.size();++i)
			if(!planes[i].n.isFinite() || !PxIsFinite(planes[i].b)) return false;
	PxU32 active[3], count = 0;
	double lambda[3] = {0.0, 0.0, 0.0};
	const PxU32 limit = 64 * (planes.size() + 1);
	for(PxU32 iteration = 0; iteration < limit; ++iteration)
	{
		PxU32 entering = PX_MAX_U32;
		double worst = 0.0;
		for(PxU32 i = 0; i < planes.size(); ++i)
		{
			const double violation = planes[i].b - planes[i].n.dot(x);
			if(violation > avbdOgcProjectionTolerance(planes[i], x) && violation > worst)
			{ entering = i; worst = violation; }
		}
		if(entering == PX_MAX_U32)
		{ result = x; return x.isFinite(); }
		const PxVec3d n = planes[entering].n;
		double enteringLambda = 0.0;
		bool admitted = false;
		for(PxU32 drop = 0; drop < 4; ++drop)
		{
			double a[3] = {0.0, 0.0, 0.0}, G[3][4] = {};
			for(PxU32 i = 0; i < count; ++i)
			{
				for(PxU32 j = 0; j < count; ++j)
					G[i][j] = planes[active[i]].n.dot(inverseMetric*planes[active[j]].n);
				G[i][count] = planes[active[i]].n.dot(inverseMetric*n);
			}
			for(PxU32 col = 0; col < count; ++col)
			{
				PxU32 pivot = col;
				for(PxU32 row = col + 1; row < count; ++row)
					if(PxAbs(G[row][col]) > PxAbs(G[pivot][col])) pivot = row;
				if(PxAbs(G[pivot][col]) <= 64.0 * DBL_EPSILON) return false;
				for(PxU32 j = col; j <= count; ++j)
				{ const double t = G[col][j]; G[col][j] = G[pivot][j]; G[pivot][j] = t; }
				const double divisor = G[col][col];
				for(PxU32 j = col; j <= count; ++j) G[col][j] /= divisor;
				for(PxU32 row = 0; row < count; ++row)
					if(row != col)
					{
						const double factor = G[row][col];
						for(PxU32 j = col; j <= count; ++j) G[row][j] -= factor * G[col][j];
					}
			}
			PxVec3d direction = inverseMetric*n;
			for(PxU32 i = 0; i < count; ++i)
			{ a[i] = G[i][count]; direction -= (inverseMetric*planes[active[i]].n) * a[i]; }
			double dualStep = DBL_MAX;
			PxU32 leaving = PX_MAX_U32;
			for(PxU32 i = 0; i < count; ++i)
				if(a[i] > 64.0 * DBL_EPSILON && lambda[i] / a[i] < dualStep)
				{ dualStep = PxMax(0.0, lambda[i] / a[i]); leaving = i; }
			const double denom = n.dot(direction);
			const double fullStep = denom > 64.0 * DBL_EPSILON ?
				PxMax(0.0, (planes[entering].b - n.dot(x)) / denom) : DBL_MAX;
			const double step = PxMin(dualStep, fullStep);
			if(step == DBL_MAX || !PxIsFinite(step)) return false;
			x += direction * step;
			for(PxU32 i = 0; i < count; ++i) lambda[i] -= step * a[i];
			enteringLambda += step;
			if(fullStep <= dualStep && count < 3)
			{
				active[count] = entering; lambda[count++] = enteringLambda;
				admitted = true; break;
			}
			if(leaving == PX_MAX_U32) return false;
			--count;
			for(PxU32 i = leaving; i < count; ++i)
			{ active[i] = active[i + 1]; lambda[i] = lambda[i + 1]; }
		}
		if(!admitted) return false;
	}
	return false;
}

// In three dimensions a convex quadratic projection has at most three
// linearly independent active planes.  The incremental active-set path above
// is fast, but nearly coincident contact and tet planes can make its current
// working set rank deficient even though another active set is well posed.
// Enumerate those small active sets only after the fast path fails.  This is a
// numerical completion of the same metric projection, not a Euclidean or
// geometry-specific fallback.
PX_INLINE bool avbdOgcProjectHalfSpacesEnumerated(
	const AvbdOgcProjectionPlanes& planes, const PxVec3d& y, PxVec3d& result,
	const PxMat33d& inverseMetric)
{
	if(avbdOgcRegionContains(planes,y,double(PX_MAX_F32)))
	{
		result=y;
		return true;
	}
	for(PxU32 i=0;i<planes.size();++i)
		if(!planes[i].n.isFinite() || !PxIsFinite(planes[i].b) ||
			!PxIsFinite(planes[i].roundoff) || planes[i].roundoff<0.0)
			return false;

	bool found=false;
	double bestObjective=DBL_MAX;
	auto consider=[&](const PxU32* active,PxU32 count)
	{
		double system[3][4]={{0.0}};
		double matrixScale=0.0;
		for(PxU32 row=0;row<count;++row)
		{
			for(PxU32 column=0;column<count;++column)
			{
				system[row][column]=planes[active[row]].n.dot(
					inverseMetric*planes[active[column]].n);
				matrixScale=PxMax(matrixScale,PxAbs(system[row][column]));
			}
			system[row][count]=planes[active[row]].b-
				planes[active[row]].n.dot(y);
		}
		if(!PxIsFinite(matrixScale) || matrixScale<=0.0) return;
		for(PxU32 column=0;column<count;++column)
		{
			PxU32 pivot=column;
			for(PxU32 row=column+1;row<count;++row)
				if(PxAbs(system[row][column])>PxAbs(system[pivot][column]))
					pivot=row;
			if(PxAbs(system[pivot][column])<=256.0*DBL_EPSILON*matrixScale)
				return;
			for(PxU32 entry=column;entry<=count;++entry)
			{
				const double value=system[column][entry];
				system[column][entry]=system[pivot][entry];
				system[pivot][entry]=value;
			}
			const double divisor=system[column][column];
			for(PxU32 entry=column;entry<=count;++entry)
				system[column][entry]/=divisor;
			for(PxU32 row=0;row<count;++row)
				if(row!=column)
				{
					const double factor=system[row][column];
					for(PxU32 entry=column;entry<=count;++entry)
						system[row][entry]-=factor*system[column][entry];
				}
		}

		PxVec3d dualNormal(0.0);
		for(PxU32 i=0;i<count;++i)
		{
			double lambda=system[i][count];
			if(!PxIsFinite(lambda) ||
				lambda < -1024.0*DBL_EPSILON*(1.0+PxAbs(lambda)))
				return;
			lambda=PxMax(0.0,lambda);
			dualNormal+=planes[active[i]].n*lambda;
		}
		const PxVec3d candidate=y+inverseMetric*dualNormal;
		if(!candidate.isFinite() ||
			!avbdOgcRegionContains(planes,candidate,double(PX_MAX_F32)))
			return;
		const PxVec3d displacement=candidate-y;
		const double objective=dualNormal.dot(displacement);
		if(!PxIsFinite(objective) || objective<0.0 || objective>=bestObjective)
			return;
		bestObjective=objective;
		result=candidate;
		found=true;
	};

	for(PxU32 i=0;i<planes.size();++i)
	{
		const PxU32 one[1]={i};
		consider(one,1);
		for(PxU32 j=i+1;j<planes.size();++j)
		{
			const PxU32 two[2]={i,j};
			consider(two,2);
			for(PxU32 k=j+1;k<planes.size();++k)
			{
				const PxU32 three[3]={i,j,k};
				consider(three,3);
			}
		}
	}
	return found;
}

PX_INLINE bool avbdOgcProjectHalfSpacesExact(const AvbdOgcProjectionPlanes& planes,
	const PxVec3d& y, PxVec3d& result, const PxMat33d& inverseMetric)
{
	return avbdOgcProjectHalfSpacesFast(planes,y,result,inverseMetric) ||
		avbdOgcProjectHalfSpacesEnumerated(planes,y,result,inverseMetric);
}

PX_INLINE bool avbdOgcProjectHalfSpaces(const AvbdOgcProjectionPlanes& planes,
	const PxVec3d& y,PxVec3d& result, const PxMat33d& inverseMetric=PxMat33d(PxIdentity))
{
	// Prefer the represented boundaries, leaving their input-error interval
	// available for final float publication. If those boundaries conflict at
	// input precision, solve the interval domain that Contains actually tests.
	if(avbdOgcProjectHalfSpacesExact(planes,y,result,inverseMetric) &&
		avbdOgcRegionContains(planes,result,double(PX_MAX_F32))) return true;
	AvbdOgcProjectionPlanes domain;
	for(PxU32 i=0;i<planes.size();++i)
	{
		AvbdOgcProjectionPlane plane=planes[i];
		if(!PxIsFinite(plane.roundoff) || plane.roundoff<0.0) return false;
		// The same input-rounding interval must define both admission and the
		// active equalities. Otherwise a point can pass Contains while projection
		// from another point incorrectly declares this very domain empty.
		plane.b-=plane.roundoff;
		plane.roundoff=0.0;
		domain.pushBack(plane);
	}
	return avbdOgcProjectHalfSpacesExact(domain,y,result,inverseMetric) &&
		avbdOgcRegionContains(planes,result,double(PX_MAX_F32));
}

PX_INLINE bool avbdOgcInverseMetric(const PxMat33d& metric,PxMat33d& inverse)
{
 if(!metric.column0.isFinite() || !metric.column1.isFinite() || !metric.column2.isFinite() ||
    metric.column0.x<=0.0 || metric.column0.x*metric.column1.y-metric.column0.y*metric.column1.x<=0.0 ||
    metric.getDeterminant()<=0.0) return false;
 inverse=metric.getInverse();
 return inverse.column0.isFinite() && inverse.column1.isFinite() && inverse.column2.isFinite();
}

// One convex block problem, including the isotropic DCD ball. Its multiplier
// adds mu*I to H. Parameter t=mu/(1+mu) compactifies [0,infinity]: at t=1 the
// Euclidean minimum-norm feasible point certifies whether the ball intersects.
PX_INLINE bool avbdOgcProjectRegion(const AvbdOgcProjectionPlanes& planes,
 const PxVec3d& y,double radius,PxVec3d& result,const PxMat33d* metric=NULL)
{
 if(!y.isFinite() || radius<0.0 || !PxIsFinite(radius)) return false;
 PxMat33d H=metric ? *metric : PxMat33d(PxIdentity);
 const double trace=H.column0.x+H.column1.y+H.column2.z;
 if(!PxIsFinite(trace) || trace<=0.0) return false;
 H=H*(3.0/trace);
 PxMat33d inverse;
 if(!avbdOgcInverseMetric(H,inverse)) return false;
 auto project=[&](double t,PxVec3d& point)->bool
 {
  const PxMat33d A=H*(1.0-t)+PxMat33d(PxIdentity)*t;
  PxMat33d K;
  if(!avbdOgcInverseMetric(A,K)) return false;
  const PxVec3d target=K*(H*y)*(1.0-t);
  // Uniform inverse-metric scaling leaves the primal minimizer unchanged.
  K=K*(3.0/(K.column0.x+K.column1.y+K.column2.z));
  return avbdOgcProjectHalfSpaces(planes,target,point,K);
 };
 if(!project(0.0,result)) return false;
 if(radius>=double(PX_MAX_F32)*0.5 || result.magnitudeSquared()<=radius*radius) return true;
 PxVec3d feasible;
 if(!project(1.0,feasible) || feasible.magnitudeSquared()>radius*radius) return false;
 double lo=0.0,hi=1.0;
 for(PxU32 i=0;i<56;++i)
 {
  const double t=(lo+hi)*0.5;
  PxVec3d trial;
  if(!project(t,trial)) return false;
  if(trial.magnitudeSquared()<=radius*radius) { hi=t; feasible=trial; }
  else lo=t;
 }
 result=feasible;
 return avbdOgcRegionContains(planes,result,radius);
}

}} // namespace physx::Dy
#endif
