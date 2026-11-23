// Native checks of the production contact potential, using finite differences.
#include "avbd/DyAvbdContactPotential.h"
#include "avbd/DyAvbdOgc.h"
#include "avbd_probe_foundation.h"
#include "avbd/DyAvbdRigidMetric.h"
#include "avbd/DyAvbdContactGeometry.h"
#include "avbd/DyAvbdVbd.h"
#include "foundation/PxVec3.h"
#include <cmath>
#include <cstdio>

using namespace physx;
using namespace physx::Dy;

static bool close(PxReal a, PxReal b, PxReal relative = 0.003f)
{
	return PxIsFinite(a) && PxIsFinite(b) &&
		PxAbs(a - b) <= relative * PxMax(1.0f, PxMax(PxAbs(a), PxAbs(b)));
}

static bool derivatives(const char* name, PxReal C, PxReal k,
	PxReal lambda, PxReal limit, PxReal expectedGradient)
{
	const PxReal h = 1.0e-5f;
	const AvbdContactPotential value =
		avbdUnilateralContactPotential(C, k, lambda, limit);
	const AvbdContactPotential plus =
		avbdUnilateralContactPotential(C + h, k, lambda, limit);
	const AvbdContactPotential minus =
		avbdUnilateralContactPotential(C - h, k, lambda, limit);
	const PxReal gradient = (plus.energy - minus.energy) / (2.0f * h);
	const PxReal curvature = (plus.gradient - minus.gradient) / (2.0f * h);
	const bool pass = close(value.gradient, gradient) &&
		close(value.curvature, curvature) && close(value.gradient, expectedGradient);
	std::printf("[CONTACT_POTENTIAL_TEST] case=%s status=%s gradient=%.9g finiteDifference=%.9g curvature=%.9g\n",
		name, pass ? "PASS" : "FAIL", double(value.gradient), double(gradient), double(curvature));
	return pass;
}

static bool supportedRow()
{
	// Unequal embedded supports and a rigid endpoint: numerical differentiation
	// of one scalar energy must yield zero net translation force.  Masses do
	// not enter this energy; introducing endpoint-specific stiffness fails it.
	const PxVec3 n = PxVec3(0.3f, 0.8f, -0.2f).getNormalized();
	const PxReal w[4] = {0.2f, 0.3f, 0.5f, -1.0f};
	const PxReal C = -0.01f, h = 1.0e-5f;
	const AvbdContactPotential value = avbdUnilateralContactPotential(C, 1.0e5f, 0.0f, -1.0e30f);
	PxVec3 netForce(0.0f);
	bool pass = true;
	for(PxU32 support = 0; support < 4; ++support)
	{
		PxVec3 numeric;
		for(PxU32 axis = 0; axis < 3; ++axis)
		{
			const PxReal shift = h * w[support] * n[axis];
			numeric[axis] = -(avbdUnilateralContactPotential(C + shift, 1.0e5f, 0.0f, -1.0e30f).energy -
				avbdUnilateralContactPotential(C - shift, 1.0e5f, 0.0f, -1.0e30f).energy) / (2.0f * h);
			pass &= close(numeric[axis], -w[support] * n[axis] * value.gradient);
		}
		netForce += numeric;
	}
	pass &= netForce.magnitude() < 0.2f;
	std::printf("[CONTACT_POTENTIAL_TEST] case=embedded_action_reaction status=%s netForce=%.9g\n",
		pass ? "PASS" : "FAIL", double(netForce.magnitude()));
	return pass;
}

static bool witnessEpoch()
{
	const PxVec3 weights(0.75f, 0.25f, 0.0f);
	const PxVec3 zero(0.0f), translation(0.1f, -0.2f, 0.3f);
	const PxVec3 initial = avbdOgcTriangleWitnessDelta(weights, zero, zero, zero);
	const PxVec3 common = avbdOgcTriangleWitnessDelta(weights,
		translation, translation, translation);
	const PxVec3 deformed = avbdOgcTriangleWitnessDelta(weights,
		PxVec3(0.0f, 4.0f, 0.0f), zero, PxVec3(0.0f, 100.0f, 0.0f));
	const bool pass = initial == zero && (common - translation).magnitude() < 1.0e-7f &&
		deformed == PxVec3(0.0f, 3.0f, 0.0f);
	std::printf("[CONTACT_POTENTIAL_TEST] case=witness_epoch_and_translation status=%s\n",
		pass ? "PASS" : "FAIL");
	return pass;
}

static bool rotationLogResolution()
{
 const PxVec3 axis=PxVec3(1,2,3).getNormalized();
 const PxReal angles[]={0,1e-6f,1e-5f,1e-4f,3e-4f,5e-4f,1e-3f,.01f,.5f,1.2f,3.f};
 const PxQuat base(.83f,PxVec3(-2,1,4).getNormalized());
 PxReal maxError=0;PxU32 lost=0;
 for(PxU32 i=0;i<sizeof(angles)/sizeof(angles[0]);++i)
 {
  const PxQuat increment(angles[i],axis);const PxVec3 expected=axis*angles[i];
  const PxVec3 actual=avbdAcceptedRotationDisplacement(PxQuat(PxIdentity),increment);
  lost+=angles[i]>0 && actual.magnitudeSquared()==0;
  maxError=PxMax(maxError,(actual-expected).magnitude());
  maxError=PxMax(maxError,(avbdAcceptedRotationDisplacement(base,increment*base)-expected).magnitude());
  maxError=PxMax(maxError,(avbdAcceptedRotationDisplacement(-base,-increment*base)-expected).magnitude());
 }
 const bool pass=!lost && maxError<1e-6f;
 std::printf("[CONTACT_POTENTIAL_TEST] case=rotation_log_resolution status=%s lostAngles=%u maxError=%.9g\n",pass?"PASS":"FAIL",lost,maxError);
 return pass;
}

static bool rotationDerivatives()
{
	const PxQuat target(0.6f, PxVec3(0.2f, 0.8f, -0.3f).getNormalized());
	const PxQuat candidate = PxQuat(1.2f, PxVec3(-0.7f, 0.1f, 0.5f).getNormalized()) * target;
	const PxMat33 frame(PxQuat(0.8f, PxVec3(0.4f, 0.6f, 0.3f).getNormalized()));
	const PxMat33 inertia = frame * PxMat33::createDiagonal(PxVec3(2.0f, 7.0f, 13.0f)) * frame.getTranspose();
	const AvbdRotationPotential value = avbdRotationInertialPotential(target, candidate, inertia);
	const PxReal h = 0.0005f;
	bool pass = true;
	PxReal error = 0.0f;
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		PxVec3 unit(0.0f); unit[axis] = 1.0f;
		const PxReal plus = avbdRotationInertialPotential(target, PxQuat(h, unit) * candidate, inertia).energy;
		const PxReal minus = avbdRotationInertialPotential(target, PxQuat(-h, unit) * candidate, inertia).energy;
		const PxReal numeric = (plus - minus) / (2.0f * h);
		error = PxMax(error, PxAbs(numeric - value.gradient[axis]));
		pass &= close(numeric, value.gradient[axis], 0.002f);
		pass &= unit.dot(value.gaussNewton * unit) >= 0.0f;
	}
	std::printf("[CONTACT_POTENTIAL_TEST] case=finite_rotation_metric status=%s maxGradientError=%.9g\n",
		pass ? "PASS" : "FAIL", double(error));
	return pass;
}

static bool completePlaneFeatures()
{
	const PxVec3 normal(0.0f, 1.0f, 0.0f);
	const PxVec3 triangle[3] = {PxVec3(-0.4f, 1.1f, 0.0f),
		PxVec3(0.4f, 1.1f, 0.0f), PxVec3(0.0f, 1.1f, 0.4f)};
	const PxQuat rotation(0.78539816339f, PxVec3(0.0f, 0.0f, 1.0f));
	PxReal gap = PX_MAX_F32;
	avbdForEachTriangleBoxPlaneSupport(PxVec3(1.0f),
		[&](PxU32 vertex, PxU32, const PxVec3& local)
		{ gap = PxMin(gap, normal.dot(triangle[vertex] - rotation.rotate(local))); });
	// All triangle vertices clear the old face-center surrogate, yet the
	// rotated box's top corner crosses their plane by about 0.314 m.
	const PxReal oldGap = 1.1f - rotation.rotate(PxVec3(0.0f, 1.0f, 0.0f)).y;
	bool pass = oldGap > 0.3f && gap < -0.3f;
	PxReal separatedGap = PX_MAX_F32;
	avbdForEachTriangleBoxPlaneSupport(PxVec3(1.0f),
		[&](PxU32 vertex, PxU32, const PxVec3& local)
		{ separatedGap = PxMin(separatedGap,
			normal.dot(triangle[vertex] + PxVec3(0.0f, 1.0f, 0.0f) - rotation.rotate(local))); });
	pass &= separatedGap > 0.6f;
	std::printf("[CONTACT_POTENTIAL_TEST] case=complete_plane_finite_rotation status=%s gap=%.9g oldGap=%.9g\n",
		pass ? "PASS" : "FAIL", double(gap), double(oldGap));
	return pass;
}

static bool boundedRowDerivatives()
{
 bool pass=true;
 const PxReal samples[]={-2.0f,-0.2f,0.0f,0.2f,2.0f};
 const PxReal h=0.0005f, penalty=17.0f, lambda=1.3f, lower=-4.0f, upper=7.0f;
 for(PxU32 i=0;i<5;++i) {
  const PxReal c=samples[i];
  const PxReal numeric=(avbdBoundedRowEnergy(c+h,penalty,lambda,lower,upper)-
   avbdBoundedRowEnergy(c-h,penalty,lambda,lower,upper))/(2.0f*h);
  pass &= close(numeric,PxClamp(penalty*c+lambda,lower,upper),0.003f);
 }
 const PxReal constantNumeric=(avbdBoundedRowEnergy(0.3f,0.0f,9.0f,lower,upper)-
  avbdBoundedRowEnergy(-0.3f,0.0f,9.0f,lower,upper))/0.6f;
 pass &= close(constantNumeric,upper,0.0001f);
 std::printf("[CONTACT_POTENTIAL_TEST] case=bounded_row_primitive status=%s\n",pass ? "PASS" : "FAIL");
 return pass;
}

static bool convectedPlaneDerivatives()
{
 const PxVec3 localNormal=PxVec3(0.2f,1.0f,-0.3f).getNormalized();
 const PxVec3 localPoint=avbdBoxCornerLocal(PxVec3(0.5f,0.7f,0.4f),avbdBoxSupportCorner(localNormal));
 const PxVec3 position(0.3f,1.4f,-0.2f);
 const PxQuat rotation(0.9f,PxVec3(0.2f,0.7f,-0.4f).getNormalized());
 const PxVec3 opposite=position+rotation.rotate(localPoint)+rotation.rotate(PxVec3(0.4f,-0.1f,0.2f));
 const AvbdContactPlaneEvaluation value=avbdEvaluateContactPlane(localNormal,localPoint,position,rotation,opposite,0.01f);
 const PxReal h=0.0005f;
 bool pass=true;
 PxReal error=0.0f;
 for(PxU32 axis=0;axis<3;++axis) {
  PxVec3 unit(0.0f);unit[axis]=1.0f;
  const PxReal plus=avbdEvaluateContactPlane(localNormal,localPoint,position,PxQuat(h,unit)*rotation,opposite,0.01f).gap;
  const PxReal minus=avbdEvaluateContactPlane(localNormal,localPoint,position,PxQuat(-h,unit)*rotation,opposite,0.01f).gap;
  error=PxMax(error,PxAbs((plus-minus)/(2.0f*h)-value.rigidAngularGradient[axis]));
 }
 const PxVec3 moment=opposite.cross(value.normal)+position.cross(-value.normal)+value.rigidAngularGradient;
 const PxQuat common(1.3f,PxVec3(0.7f,-0.2f,0.3f).getNormalized());
 const PxVec3 shift(3.0f,-2.0f,1.0f);
 const AvbdContactPlaneEvaluation moved=avbdEvaluateContactPlane(localNormal,localPoint,
  common.rotate(position)+shift,common*rotation,common.rotate(opposite)+shift,0.01f);
 pass &= error<0.001f && moment.magnitude()<1e-6f && PxAbs(moved.gap-value.gap)<1e-6f;
 // A frozen world normal retains only the support-point angular derivative.
 // That surrogate has a nonzero net moment for this same embedded point.
 const PxVec3 frozenMoment=opposite.cross(value.normal)+
  (position+rotation.rotate(localPoint)).cross(-value.normal);
 pass &= frozenMoment.magnitude()>0.1f;
 std::printf("[CONTACT_POTENTIAL_TEST] case=convected_plane_derivatives status=%s gradientError=%.9g netMoment=%.9g frozenMoment=%.9g\n",
  pass ? "PASS" : "FAIL",double(error),double(moment.magnitude()),double(frozenMoment.magnitude()));
 return pass;
}

static bool convectedPlaneCoverage()
{
 bool pass=true;
 const PxVec3 extent(0.5f,0.7f,0.4f), position(0.3f,0.5f,-0.2f);
 for(PxU32 sample=0;sample<23;++sample) {
  const PxVec3 normalLocal=PxVec3(0.3f+0.01f*sample,1.0f,-0.2f).getNormalized();
  const PxQuat rotation(0.13f*sample,PxVec3(0.2f,0.7f,-0.4f).getNormalized());
  const PxVec3 support=avbdBoxCornerLocal(extent,avbdBoxSupportCorner(normalLocal));
  const PxVec3 triangle[3]={PxVec3(-0.4f,1.1f,0.0f),PxVec3(0.4f,1.1f,0.0f),PxVec3(0.0f,1.1f,0.4f)};
  for(PxU32 vertex=0;vertex<3;++vertex) {
   const AvbdContactPlaneEvaluation value=avbdEvaluateContactPlane(normalLocal,support,position,rotation,triangle[vertex],0.01f);
   PxReal full=PX_MAX_F32;
   for(PxU32 corner=0;corner<8;++corner)
    full=PxMin(full,value.normal.dot(triangle[vertex]-position-rotation.rotate(avbdBoxCornerLocal(extent,corner)))-0.01f);
   pass &= PxAbs(value.gap-full)<1e-6f;
  }
 }
 std::printf("[CONTACT_POTENTIAL_TEST] case=convected_plane_support_equivalence status=%s\n",pass ? "PASS" : "FAIL");
 return pass;
}

static bool trianglePairCoverage()
{
	const PxVec3 a[3]={PxVec3(0,0,0),PxVec3(1,0,0),PxVec3(0,1,0)};
	const PxVec3 b[3]={PxVec3(2,0,0),PxVec3(3,0,0),PxVec3(2,1,0)};
	const AvbdTrianglePairPlane coplanar=avbdTrianglePairPlane(a,b,PxVec3(-1,0,0));
	bool pass=coplanar.gap>0.999f;
	const PxVec3 crossing[3]={PxVec3(0.25f,0.25f,-1),PxVec3(0.25f,0.25f,1),PxVec3(0.75f,0.25f,0)};
	const AvbdTrianglePairPlane intersection=avbdTrianglePairPlane(a,crossing,PxVec3(0,0,1));
	pass &= intersection.gap<=0.0f;
	// Every vertex pair is needed: the centroid can remain above the plane
	// while a single corner moves through the opposite face's projection.
	const PxVec3 lifted[3]={PxVec3(0,0,1),PxVec3(1,0,1),PxVec3(0,1,1)};
	const AvbdTrianglePairPlane plane=avbdTrianglePairPlane(lifted,a,PxVec3(0,0,1));
	PxVec3 trial[3]={lifted[0],lifted[1],PxVec3(0,1,-0.1f)};
	PxReal gap=PX_MAX_F32;
	for(PxU32 i=0;i<3;++i) for(PxU32 j=0;j<3;++j) gap=PxMin(gap,plane.normal.dot(trial[i]-a[j]));
	pass &= ((trial[0]+trial[1]+trial[2])*(1.0f/3.0f)).z>0.0f && gap<0.0f;
	// Complete linear supports preserve separation throughout interpolation,
	// including a common translation much larger than the initial gap.
	for(PxU32 sample=0;sample<=50;++sample)
	{
		const PxReal t=PxReal(sample)/50.0f;
		const PxVec3 da(4*t,-3*t,2*t), db=da+PxVec3(0,0,0.75f*t);
		for(PxU32 i=0;i<3;++i) for(PxU32 j=0;j<3;++j)
			pass &= plane.normal.dot((lifted[i]+da)-(a[j]+db))>=0.24999f;
	}
	std::printf("[CONTACT_POTENTIAL_TEST] case=triangle_pair_complete_supports status=%s\n",pass?"PASS":"FAIL");
	return pass;
}

static bool triangleBoxCoverage()
{
 const PxVec3 half(1.0f),center(0.0f);
 const PxVec3 separated[3]={PxVec3(-2,1.1f,-2),PxVec3(2,1.1f,-2),PxVec3(0,1.1f,2)};
 const PxVec3 crossing[3]={PxVec3(-2,0,-2),PxVec3(2,0,-2),PxVec3(0,0,2)};
 const AvbdTrianglePairPlane plane=avbdTriangleBoxPlane(separated,center,PxQuat(PxIdentity),half);
 bool pass=close(plane.gap,0.1f,1.0e-5f) && plane.normal.y>0.999f;
 // All triangle vertices can be outside while its face crosses the box.
 pass &= avbdTriangleBoxPlane(crossing,center,PxQuat(PxIdentity),half).gap<0.0f;
 const PxQuat rotation(0.73f,PxVec3(1,2,3).getNormalized());
 const PxVec3 translation(4,-3,2);
 PxVec3 rotated[3];
 for(PxU32 i=0;i<3;++i) rotated[i]=rotation.rotate(separated[i])+translation;
 const AvbdTrianglePairPlane moved=avbdTriangleBoxPlane(rotated,translation,rotation,half);
 pass &= close(moved.gap,plane.gap,1.0e-5f) &&
  (moved.normal-rotation.rotate(plane.normal)).magnitude()<1.0e-5f;
 for(PxU32 sample=0;sample<=20;++sample)
 {
  const PxReal t=PxReal(sample)/20.0f;
  for(PxU32 i=0;i<3;++i) for(PxU32 j=0;j<8;++j)
   pass &= plane.normal.dot(separated[i]+PxVec3(50*t,-0.099f*t,0)-avbdBoxCornerLocal(half,j))>0.0f;
 }
 // A centroid above the support plane does not protect a single low corner.
 PxVec3 trial[3]={separated[0],separated[1],separated[2]}; trial[2].y=0.9f;
 pass &= (trial[0].y+trial[1].y+trial[2].y)/3.0f>1.0f &&
  plane.normal.dot(trial[2]-avbdBoxCornerLocal(half,7))<0.0f;
 for(PxU32 seed=1;seed<=100;++seed)
 {
  const PxReal scale=0.01f*PxReal(seed);
  const PxVec3 extent(scale,1.3f*scale,0.8f*scale);
  const PxQuat q(0.071f*seed,PxVec3(1,2,3).getNormalized());
  const PxVec3 c(3*scale,-2*scale,scale);
  const PxVec3 local[3]={PxVec3(-2*scale,1.4f*scale,-scale),
   PxVec3(2*scale,1.6f*scale,-scale),PxVec3(scale,1.5f*scale,2*scale)};
  PxVec3 world[3];
  for(PxU32 i=0;i<3;++i) world[i]=c+q.rotate(local[i]);
  const AvbdTrianglePairPlane certificate=avbdTriangleBoxPlane(world,c,q,extent);
  PxReal full=PX_MAX_F32;
  for(PxU32 i=0;i<3;++i) for(PxU32 j=0;j<8;++j)
   full=PxMin(full,certificate.normal.dot(world[i]-c-q.rotate(avbdBoxCornerLocal(extent,j))));
  pass &= full>0.0f && close(certificate.gap,full,1.0e-5f);
  // A vertex strictly inside the box rules out every separating plane.
  world[0]=c;
  pass &= avbdTriangleBoxPlane(world,c,q,extent).gap<0.0f;
 }
 // Captured Surface epoch 68: changing a barely overlapping diagonal
 // certificate to a top-face recovery plane admits a much deeper crossing.
 const PxVec3 capturedStart[3]={PxVec3(-1.16557777f,1.99201906f,-0.77843529f),
  PxVec3(-1.28180277f,2.00234652f,-0.497449458f),PxVec3(-1.02829754f,2.06233335f,-0.80885303f)};
 const PxVec3 capturedNow[3]={PxVec3(-1.15880859f,1.99652481f,-0.77952987f),
  PxVec3(-1.25360954f,2.00000024f,-0.500928938f),PxVec3(-1.0024178f,2.0403409f,-0.818966091f)};
 const PxVec3 capturedCenter(0,1,0),capturedHalf(1.15f,1,1.15f);
 const AvbdTrianglePairPlane captured=avbdTriangleBoxPlane(capturedStart,
  capturedCenter,PxQuat(PxIdentity),capturedHalf);
 const PxVec3 support=capturedCenter+avbdBoxCornerLocal(capturedHalf,avbdBoxSupportCorner(captured.normal));
 bool retainedAccepts=true,switchedAccepts=true;
 for(PxU32 i=0;i<3;++i)
 {
  AvbdOgcPointRegion retained,switched;
  avbdOgcResetPointRegion(retained,capturedStart[i]);
  avbdOgcResetPointRegion(switched,capturedStart[i]);
  avbdOgcAddHalfSpace(retained.planes,retained.planeCount,captured.normal,
   PxMax(0.0f,captured.normal.dot(capturedStart[i]-support)),PX_MAX_U32,PxVec3(0),PxVec3(0));
  avbdOgcAddHalfSpace(switched.planes,switched.planeCount,PxVec3(0,1,0),
   PxMax(0.0f,capturedStart[i].y-2.0f),PX_MAX_U32,PxVec3(0),PxVec3(0));
  retainedAccepts &= avbdOgcPointContains(retained,capturedNow[i]);
  switchedAccepts &= avbdOgcPointContains(switched,capturedNow[i]);
 }
 pass &= !retainedAccepts && switchedAccepts;
 std::printf("[CONTACT_POTENTIAL_TEST] case=triangle_box_complete_static_domain status=%s\n",pass?"PASS":"FAIL");
 return pass;
}

static bool surfaceEnergyDerivatives()
{
	AvbdSoftParticle particles[4] = {};
	particles[0].p=PxVec3(0,1,0); particles[1].p=PxVec3(0,-1,0);
	particles[2].p=PxVec3(-1,0,0); particles[3].p=PxVec3(1,0,0);
	AvbdSoftTri tri;
	if(!avbdSoftBuildTri(tri,0,2,3,particles)) return false;
	particles[0].p=PxVec3(0.2f,1.15f,0.3f);
	particles[1].p=PxVec3(0.1f,-0.85f,0.2f);
	AvbdSoftBending bend={};
	bend.opp0=0; bend.opp1=1; bend.edgeStart=2; bend.edgeEnd=3; bend.restLength=2.0f;
	bool membrane=true,bending=true;
	const PxReal h=1.0e-4f;
	for(PxU32 order=0;order<4;++order)
	{
		PxVec3 force; PxMat33 H;
		avbdSoftBendingForceHessian(bend,order,3.0f,0.2f,particles,force,H);
		for(PxU32 axis=0;axis<3;++axis)
		{
			const PxReal original=particles[order].p[axis];
			particles[order].p[axis]=original+h;
			const PxReal plus=avbdSoftBendingEnergy(bend,3.0f,0.2f,particles[0].p,particles[1].p,particles[2].p,particles[3].p);
			particles[order].p[axis]=original-h;
			const PxReal minus=avbdSoftBendingEnergy(bend,3.0f,0.2f,particles[0].p,particles[1].p,particles[2].p,particles[3].p);
			particles[order].p[axis]=original;
			bending &= close(-force[axis],(plus-minus)/(2.0f*h),0.005f);
		}
	}
	const PxU32 indices[3]={0,2,3};
	for(PxU32 order=0;order<3;++order)
	{
		PxVec3 force; PxMat33 H;
		avbdSoftStvkTriForceHessian(tri,order,7.0f,11.0f,particles,force,H);
		for(PxU32 axis=0;axis<3;++axis)
		{
			PxVec3& p=particles[indices[order]].p;
			const PxReal original=p[axis];
			p[axis]=original+h;
			const PxReal plus=avbdSoftStvkTriEnergy(tri,7.0f,11.0f,particles[0].p,particles[2].p,particles[3].p);
			p[axis]=original-h;
			const PxReal minus=avbdSoftStvkTriEnergy(tri,7.0f,11.0f,particles[0].p,particles[2].p,particles[3].p);
			p[axis]=original;
			membrane &= close(-force[axis],(plus-minus)/(2.0f*h),0.005f);
		}
	}
	std::printf("[CONTACT_POTENTIAL_TEST] case=surface_membrane_energy status=%s\n",membrane?"PASS":"FAIL");
	std::printf("[CONTACT_POTENTIAL_TEST] case=surface_bending_energy status=%s\n",bending?"PASS":"FAIL");
	return membrane && bending;
}

static bool rowChartDerivative()
{
	PxReal error=0,legacyError=0; bool pass=true;
	const PxQuat reference(.61f,PxVec3(1,3,2).getNormalized());
	const PxVec3 row(.7f,-1.3f,.2f);
	for(PxU32 sample=0;sample<30;++sample)
	{
		const PxQuat q=PxQuat(.1f*sample,PxVec3(-2,1,4).getNormalized())*reference;
		const PxVec3 exact=avbdRotationChartGradient(reference,q,row);
		for(PxU32 a=0;a<3;++a)
		{
			PxVec3 axis(0);axis[a]=1;
			const PxQuat plus=(PxQuat(1e-3f,axis)*q)*reference.getConjugate();
			const PxQuat minus=(PxQuat(-1e-3f,axis)*q)*reference.getConjugate();
			const PxReal numeric=row.dot(plus.getImaginaryPart()-minus.getImaginaryPart())/1e-3f;
			error=PxMax(error,PxAbs(numeric-exact[a]));
			legacyError=PxMax(legacyError,PxAbs(numeric-row[a]));
		}
		pass &= (avbdRotationChartGradient(-reference,-q,row)-exact).magnitude()<1e-6f;
	}
	pass &= error<.0003f;
	std::printf("[CONTACT_POTENTIAL_TEST] case=row_rotation_chart status=%s error=%.9g legacyError=%.9g\n",pass?"PASS":"FAIL",error,legacyError);
	return pass;
}

int main()
{
	AvbdProbeFoundation foundation;
	bool pass = true;
	pass &= rowChartDerivative();
	pass &= derivatives("offset_active", -0.01f, 1.0e5f, 0.0f, -1.0e30f, -1000.0f);
	pass &= derivatives("offset_released", 0.01f, 1.0e5f, 0.0f, -1.0e30f, 0.0f);
	pass &= derivatives("al_penetrating", -0.01f, 1.0e4f, -100.0f, -1.0e30f, -200.0f);
	pass &= derivatives("al_positive_gap_loaded", 0.005f, 1.0e4f, -100.0f, -1.0e30f, -50.0f);
	pass &= derivatives("al_released", 0.02f, 1.0e4f, -100.0f, -1.0e30f, 0.0f);
	pass &= derivatives("offset_force_limit", -0.01f, 1.0e5f, 0.0f, -25.0f, -25.0f);
	pass &= derivatives("al_force_limit", -0.01f, 1.0e4f, -100.0f, -25.0f, -25.0f);
	pass &= derivatives("zero_force_limit", -0.01f, 1.0e4f, -100.0f, 0.0f, 0.0f);
	pass &= supportedRow();
	pass &= witnessEpoch();
	pass &= rotationDerivatives();
	pass &= rotationLogResolution();
	pass &= completePlaneFeatures();
	pass &= boundedRowDerivatives();
	pass &= convectedPlaneDerivatives();
	pass &= convectedPlaneCoverage();
	pass &= surfaceEnergyDerivatives();
	pass &= trianglePairCoverage();
	pass &= triangleBoxCoverage();
	return pass ? 0 : 1;
}
