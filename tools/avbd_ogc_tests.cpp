// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "foundation/PxFoundation.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"
#include "extensions/PxDefaultAllocator.h"
#include "avbd/DyAvbdCoupledDomain.h"
#include "avbd/DyAvbdOgc.h"
#include <cstdio>
#include <cstdlib>

using namespace physx;
using namespace physx::Dy;
struct Errors : PxErrorCallback
{
	void reportError(PxErrorCode::Enum, const char* message, const char*, int) PX_OVERRIDE
	{ std::fprintf(stderr, "%s\n", message); }
};
static void require(bool condition, const char* name)
{
	if(!condition) { std::printf("FAIL %s\n", name); std::exit(1); }
}
static void add(AvbdOgcPlanes& planes, PxU32& count, const PxVec3& n,
	PxReal bound, const PxVec3& point = PxVec3(0), PxU32 partner = PX_MAX_U32)
{
	avbdOgcAddHalfSpace(planes, count, n, bound, partner, PxVec3(0), PxVec3(0),
		PX_MAX_U32, PX_MAX_U32, PX_MAX_U32, PX_MAX_U32, point);
}
static void testCertificates()
{
	AvbdOgcRigidRegion rigid = {};
	rigid.anchorPosition = PxVec3(0);
	rigid.anchorRotation = PxQuat(PxIdentity);
	rigid.bound = PX_MAX_F32;
	rigid.supportRadius = 1;
	add(rigid.planes, rigid.planeCount, PxVec3(0,1,0), 0.05f, PxVec3(1,0,0));
	add(rigid.planes, rigid.planeCount, PxVec3(0,1,0), 0.05f, PxVec3(-1,0,0));
	require(rigid.planeCount == 2, "rotating feature identity");
	PxVec3 p(0); PxQuat q(0.2f, PxVec3(0,0,1));
	require(avbdOgcTruncateRigid(rigid, p, q).feasible, "rigid projection exists");
	for(PxU32 i = 0; i < rigid.planeCount; ++i)
	{
		const AvbdOgcHalfSpace& plane = rigid.planes[i];
		require(-plane.n.dot(p + q.rotate(plane.selfLocalPoint) - plane.selfLocalPoint) <= 0.050001f,
			"both rotated features feasible");
	}
	for(PxU32 i = 0; i < 20; ++i)
		add(rigid.planes, rigid.planeCount, PxVec3(0,1,0), 0.05f, PxVec3(0), i);
	require(rigid.planeCount == 22, "no eight-plane eviction; partner identity");
	AvbdOgcRigidRegion copy = rigid;
	copy.planes[0].partnerDelta = PxVec3(1);
	require(rigid.planes[0].partnerDelta == PxVec3(0), "live view does not mutate DCD epoch");
	AvbdOgcPointRegion point = {};
	avbdOgcResetPointRegion(point, PxVec3(0));
	add(point.planes, point.planeCount, PxVec3(1,0,0), 0);
	add(point.planes, point.planeCount, PxVec3(-0.5f,0.8660254f,0), 0);
	p = PxVec3(-1,-1,0);
	require(avbdOgcTruncatePoint(point, p).feasible, "nonorthogonal intersection exists");
	require(p.x >= -1e-6f && -0.5f*p.x+0.8660254f*p.y >= -1e-6f,
		"later projection preserves earlier plane");
	p = PxVec3(1,1,0); const PxVec3 original = p;
	require(avbdOgcTruncatePoint(point, p).feasible && p == original, "legal candidate unchanged");
}
static void testParallelDomainImplication()
{
	AvbdOgcProjectionPlanes all;
	for(PxU32 i=0;i<100;++i)
	{
		AvbdOgcProjectionPlane p={PxVec3d(1,0,0),double(i)*0.1,2.0}; all.pushBack(p);
	}
	AvbdOgcProjectionPlane strict={PxVec3d(1,0,0),10.0,9.0}; all.pushBack(strict);
	AvbdOgcProjectionPlane interval={PxVec3d(1,0,0),8.0,0.0}; all.pushBack(interval);
	AvbdOgcProjectionPlanes reduced=all;
	avbdOgcCompactParallelPlanes(reduced);
	require(reduced.size()==2,"distinct strongest exact and interval bounds retained");
	for(PxU32 i=0;i<1000;++i)
	{
		const PxVec3d point(double(i)*0.02-2.0,0.7,-0.3);
		require(avbdOgcRegionContains(all,point,PX_MAX_F32)==
			avbdOgcRegionContains(reduced,point,PX_MAX_F32),"parallel implication preserves admission");
	}
	AvbdOgcProjectionPlane tilted={PxVec3d(1,1.0e-8,0),0.0,0.0}; reduced.pushBack(tilted);
	avbdOgcCompactParallelPlanes(reduced);
	require(reduced.size()==3,"near-parallel domain coefficients remain distinct");
	std::puts("PASS exact parallel-domain implication and distinct input intervals");
}

static void testSignedEmbeddedGroundCertificate()
{
	const PxVec3 normal(0,1,0);
	const PxReal weights[2] = {2.0f, -1.0f};
	const PxReal absoluteWeightSum = 3.0f;
	const PxReal featureBound = 0.3f;
	AvbdOgcPointRegion regions[2];
	for(PxU32 i = 0; i < 2; ++i)
	{
		avbdOgcResetPointRegion(regions[i], PxVec3(0));
		PxVec3 supportNormal;
		PxReal supportBound = 0.0f;
		require(avbdOgcEmbeddedSupportHalfSpace(normal, featureBound,
			absoluteWeightSum, weights[i], supportNormal, supportBound),
			"signed embedded support plane exists");
		avbdOgcAddHalfSpace(regions[i].planes, regions[i].planeCount,
			supportNormal, supportBound, PX_MAX_U32, PxVec3(0), PxVec3(0));
	}
	require(regions[0].planes[0].n == normal &&
		regions[1].planes[0].n == -normal,
		"negative embedding reverses the support half-space");
	const PxVec3 boundary[2] = {PxVec3(0,-0.1f,0), PxVec3(0,0.1f,0)};
	require(avbdOgcPointContains(regions[0], boundary[0]) &&
		avbdOgcPointContains(regions[1], boundary[1]),
		"split support boundary is feasible");
	const PxReal embeddedBoundaryDelta =
		weights[0] * boundary[0].y + weights[1] * boundary[1].y;
	require(embeddedBoundaryDelta >= -featureBound - 1.0e-6f,
		"split domains imply the embedded feature certificate");
	const PxVec3 oldFalsePositive(0,1,0);
	require(normal.dot(oldFalsePositive) >= -featureBound / absoluteWeightSum &&
		!avbdOgcPointContains(regions[1], oldFalsePositive) &&
		weights[1] * oldFalsePositive.y < -featureBound,
		"negative support motion rejected instead of penetrating embedded vertex");
	std::puts("PASS signed embedded ground certificate and negative-weight polarity");
}

static void testCapturedFloatPublication()
{
	// Showcase frame 57: the double point passes the original input-error
	// interval. Rounding every coordinate toward the DCD anchor does not.
	const PxVec3 anchor(-0.019712928682565689f,3.4376699924468994f,0.099783375859260559f);
	const PxVec3d projected(0.034155264447988887,0.0088038130689660765,-0.011903783495487142);
	AvbdOgcProjectionPlanes planes;
	AvbdOgcProjectionPlane p={PxVec3d(-0.4354173243045807,0.5087653398513794,0.7426773309707642),
		-0.01923171430826187,1.7528019498058711e-6};
	planes.pushBack(p);
	require(avbdOgcRegionContains(planes,projected,PX_MAX_F32),"captured double point feasible");
	const PxVec3 old=avbdOgcPublishPoint(anchor,projected);
	require(!avbdOgcRegionContains(planes,avbdOgcDouble(old)-avbdOgcDouble(anchor),PX_MAX_F32),
		"captured toward-anchor rounding violates plane");
	PxVec3 result;
	require(avbdOgcPublishRegionPoint(anchor,projected,planes,PX_MAX_F32,result),
		"captured adjacent float cell has admissible point");
	require(avbdOgcRegionContains(planes,avbdOgcDouble(result)-avbdOgcDouble(anchor),PX_MAX_F32),
		"captured publication preserves original tolerance");
	// A real interval strictly between two adjacent floats has no float point.
	// Do not widen the interval or misreport this as publication success.
	planes.clear();
	p={PxVec3d(1,0,0),1.0+0.25*FLT_EPSILON,0.0}; planes.pushBack(p);
	p={PxVec3d(-1,0,0),-1.0-0.75*FLT_EPSILON,0.0}; planes.pushBack(p);
	require(!avbdOgcPublishRegionPoint(PxVec3(0),PxVec3d(1.0+0.5*FLT_EPSILON,0,0),
		planes,PX_MAX_F32,result),"empty float cell remains rejected");
	std::puts("PASS captured showcase float publication and empty float interval");
}

static void testProjection()
{
	AvbdOgcProjectionPlanes planes;
	AvbdOgcProjectionPlane c = {PxVec3d(1,0,0), 0.5}; planes.pushBack(c);
	PxVec3d x;
	require(avbdOgcProjectRegion(planes, PxVec3d(-2,2,0), 1.0, x), "ball and plane intersection");
	require(PxAbs(x.x-0.5) < 1e-8 && PxAbs(x.y-PxSqrt(0.75)) < 1e-8, "ball KKT optimum");
	c.n = PxVec3d(-1,0,0); c.b = 0.5; planes.pushBack(c);
	require(!avbdOgcProjectHalfSpaces(planes, PxVec3d(0), x), "empty polyhedron rejected");
	planes.popBack();
	require(!avbdOgcProjectRegion(planes, PxVec3d(1), 0.25, x), "empty ball intersection rejected");
	// Manufactured KKT solutions: independent active planes and strictly slack
	// distractors. The oracle is the specified minimizer, not a second projector.
	for(PxU32 trial = 0; trial < 200; ++trial)
	{
		planes.clear();
		const double angle = double(trial) * 0.037;
		const PxVec3d n0(PxCos(angle), PxSin(angle), 0);
		const PxVec3d n1(-PxSin(angle), PxCos(angle), 0);
		const PxVec3d n2 = (n0 + n1 + PxVec3d(0,0,1)).getNormalized();
		const PxVec3d expected(0.2, -0.3, 0.4);
		const PxVec3d target = expected - n0*0.7 - n1*1.3 - n2*0.9;
		for(PxU32 j = 0; j < 12; ++j)
		{
			const double a = double(j)*0.63;
			c.n = PxVec3d(PxCos(a), PxSin(a), 0.2).getNormalized();
			c.b = c.n.dot(expected) - 0.1;
			planes.pushBack(c);
		}
		c.n=n0; c.b=n0.dot(expected); planes.pushBack(c);
		c.n=n1; c.b=n1.dot(expected); planes.pushBack(c);
		c.n=n2; c.b=n2.dot(expected); planes.pushBack(c);
		require(avbdOgcProjectHalfSpaces(planes, target, x), "manufactured KKT feasible");
		require((x-expected).magnitude() < 1e-8, "manufactured KKT minimizer");
	}
}
static void testComposedCommit()
{
    AvbdOgcPointRegion region;
    avbdOgcResetPointRegion(region,PxVec3(0));
    // A live partner closes the old origin out of this half-space.
    avbdOgcAddHalfSpace(region.planes,region.planeCount,PxVec3(1,0,0),0,
        PX_MAX_U32,PxVec3(0),PxVec3(1,0,0));
    const PxVec3 target(-1,3,0);
    PxVec3 sequential=target;
    require(avbdOgcTruncatePoint(region,sequential).feasible,"first projection exists");
    sequential*=0.5f/sequential.y;
    require(!avbdOgcPointContains(region,sequential),"tet shrink counterexample exercises invalid origin");
    AvbdOgcProjectionPlanes material;
    AvbdOgcProjectionPlane tet={PxVec3d(0,-1,0),-0.5}; material.pushBack(tet);
    PxVec3 candidate=target;
    require(avbdOgcTruncatePoint(region,candidate,&material).feasible,"joint contact/tet domain exists");
    require((candidate-PxVec3(1,0.5f,0)).magnitude()<1e-6f,"joint domain KKT solution");
    require(avbdOgcPointContains(region,candidate),"published position remains in OGC");
    tet.n=PxVec3d(-1,0,0); tet.b=0; material.pushBack(tet);
    candidate=target;
    require(!avbdOgcTruncatePoint(region,candidate,&material).feasible && candidate==target,
        "failed combined projection does not publish partial state");
    AvbdOgcProjectionPlanes malformed;
    tet.n=PxVec3d(0); tet.b=1; malformed.pushBack(tet);
    PxVec3d result;
    require(!avbdOgcProjectHalfSpaces(malformed,PxVec3d(0),result),"zero-normal contradiction rejected");
    AvbdOgcProjectionPlanes rounded;
    tet.n=PxVec3d(1,0,0); tet.b=1.0+1e-6; tet.roundoff=1e-6; rounded.pushBack(tet);
    tet.n=PxVec3d(-1,0,0); tet.b=-1.0+1e-6; rounded.pushBack(tet);
    require(avbdOgcRegionContains(rounded,PxVec3d(1,0,0),double(PX_MAX_F32)),"input interval has a feasible point");
    require(avbdOgcProjectHalfSpaces(rounded,PxVec3d(0),result) && PxAbs(result.x-1.0)<1e-10,
        "projection and admission use the same input interval");
    const PxVec3 anchor(2.05901361f,4.0f,25.0f);
    const PxVec3d displacement(0.002567891,-0.0036173,0.001907);
    const PxVec3 published=avbdOgcPublishPoint(anchor,displacement);
    require((avbdOgcDouble(published)-avbdOgcDouble(anchor)).magnitudeSquared()<=displacement.magnitudeSquared(),
        "float publication preserves the finite ball");
    std::printf("PASS composed contact/material commit and invalid-origin regression\n");
}

static void testCapturedEmptyDomain()
{
    // Actual Surface static-box initialization capture. Cauchy--Schwarz alone
    // proves this allocated support certificate cannot intersect its OGC ball.
    // Correct rejection here does not make the physical scene a passing case.
    AvbdOgcProjectionPlanes planes;
    AvbdOgcProjectionPlane plane={PxVec3d(0,0.99920785427093506,-0.039794996380805969),
        0.0038999680936791584,3.8576484933552081e-06};
    const double radius=0.00094091321807354689;
    require(plane.b-plane.roundoff>plane.n.magnitude()*radius,"captured domain has an analytic empty-intersection certificate");
    planes.pushBack(plane);
    PxVec3d result;
    require(!avbdOgcProjectRegion(planes,PxVec3d(0,-0.053501725196838379,0),radius,result),
        "captured impossible initialization cannot publish a pose");
    std::printf("PASS captured Surface domain rejects; physical initialization remains unresolved\n");
}

static void testMetricProjection()
{
    const PxMat33d H(PxVec3d(5,2,0),PxVec3d(2,1,0),PxVec3d(0,0,3));
    AvbdOgcProjectionPlanes planes;
    AvbdOgcProjectionPlane plane={PxVec3d(1,0,0),0.0}; planes.pushBack(plane);
    const PxVec3d newton(-1,1,0),force=H*newton;
    PxVec3d euclidean,metric;
    require(avbdOgcProjectRegion(planes,newton,double(PX_MAX_F32),euclidean),"Euclidean counterexample exists");
    require(force.dot(euclidean)<0.0,"Euclidean projection turns Newton descent into ascent");
    require(avbdOgcProjectRegion(planes,newton,double(PX_MAX_F32),metric,&H),"Hessian projection exists");
    require((metric-PxVec3d(0,-1,0)).magnitude()<1e-10 && force.dot(metric)>0.0,
        "Hessian projection preserves a feasible descent direction");
    const PxVec3d expected(0.6,0.8,0);
    planes[0].b=0.6;
    const PxVec3d target=expected+H.getInverse()*(expected*0.7-PxVec3d(1,0,0)*2.0);
    require(avbdOgcProjectRegion(planes,target,1.0,metric,&H) && (metric-expected).magnitude()<1e-8,
        "anisotropic ball and plane KKT solution");
    for(PxU32 trial=0;trial<100;++trial)
    {
        const double angle=trial*0.071;
        const PxMat33d R(PxVec3d(PxCos(angle),PxSin(angle),0),
            PxVec3d(-PxSin(angle),PxCos(angle),0),PxVec3d(0,0,1));
        const PxMat33d rotated=R*H*R.getTranspose();
        const PxVec3d solution=R*expected,normal=R*PxVec3d(1,0,0);
        const double multiplier=0.05+0.09*trial;
        const PxVec3d y=solution+rotated.getInverse()*(solution*multiplier-normal*2.0);
        planes[0].n=normal; planes[0].b=normal.dot(solution);
        require(avbdOgcProjectRegion(planes,y,1.0,metric,&rotated) && (metric-solution).magnitude()<1e-8,
            "rotated metric/ball manufactured KKT solution");
    }
    std::printf("PASS metric descent counterexample and 101 anisotropic ball KKT cases\n");
}

static void testCapturedNearDependentMetricProjection()
{
    // Checked visual-showcase, solve stamp 135. The incremental active set
    // rejected this nonempty domain after selecting nearly coincident contact
    // planes. The exhaustive <=3-plane completion must recover the same
    // Hessian-metric projection; changing to a Euclidean projection is not a
    // valid fallback because it can reverse a Newton descent direction.
    static const double data[][5]={
        {-0.85821647263024736,-0.36076044270217555,-0.36512516906867215,-0.33943826292421381,7.0947577891358509e-06},
        {-0.61637830482699696,-0.77424621609306499,0.14359868454887964,-0.11166230005175151,6.3955124587060105e-06},
        {-0.55806949202342426,-0.7619820463727689,0.32854497877502931,-0.11260125889195374,6.0041550062071407e-06},
        {-0.51969486014790944,0.85399355841517699,-0.024743777424401606,-0.65811424642012828,6.1960152759277349e-06},
        {-0.42646069546905879,-0.18586784114871163,0.88520303933434619,-0.35548533807514454,3.7919529825849052e-06},
        {-0.35782391938829222,0.092112076324452397,-0.92923485089012858,-0.40266235830954805,3.1063038431914738e-06},
        {-0.35579040391660843,0.86483149393291647,0.35423110476441316,-0.62759142246697885,5.1594098291816929e-06},
        {-0.32203828428788406,0.052396542216404864,-0.94527559252139814,-0.41637542112144216,2.7575462464949604e-06},
        {-0.092889097525089934,-0.99393228457840732,-0.058908651603330128,-0.075884889271953443,3.3707971375291674e-06},
        {-0.069593882626077727,0.99725166901770768,0.025412598104563283,-1.2088772843645699,8.9873751876465051e-06},
        {-0.060686831064719497,0.99059745428073265,0.12261236523228046,0.00055159055496383272,3.1296789913083441e-06},
        {-0.057656945644816958,0.99827299315116025,-0.011256454323365908,2.706703837052528e-05,1.5253515005796849e-05},
        {-0.042407276908567618,0.99855972422527073,0.032864875176259106,-0.044443134441946253,2.0716926512540188e-05},
        {-0.031148663092980675,0.99910420994702354,0.028645042392248264,-0.00052215411785717697,8.6586390835442204e-06},
        {-0.031148645386678858,0.99910376234320419,0.028660669256475483,-0.00053848013638741431,8.6586498267882121e-06},
        {-0.031039656103847839,0.99910981550785771,0.028567749382991288,-0.0018098805661301877,1.1602943974829092e-05},
        {-0.018220126783991326,-0.91826905593303321,-0.39553756824836628,-0.17736786459547843,2.7803944640850162e-06},
        {-0.012259751180226125,0.99969112967550666,0.02161813472777516,-0.010596621900976922,8.2812106235692846e-06},
        {-0.012252694781605924,0.99969248200245275,0.021559519901069427,-0.017720082774242342,1.1082208376801284e-05},
        {-0.010796982397391705,-0.98792564986031428,0.15455204792944685,-0.076264287825060992,2.8140782966026194e-06},
        {0,1,0,-1.2564531564712524,3.25343989970861e-06},
        {0.057503106471099012,-0.99828061510854726,0.011366892480987765,5.1218011236358657e-07,2.1141884236634352e-05},
        {0.24672959706115224,-0.9079642292091763,0.33871147664439694,-0.16851624478187355,4.295448596095171e-06},
        {0.26356514910647044,0.060520149426161726,-0.9627412547979437,-0.0099956813811559811,1.1240364855374765e-05},
        {0.26373021561789772,0.060385463891565717,-0.96270450768693172,-0.052569438473387707,1.5529198891673418e-05},
        {0.70458814234601053,0.52552650369795695,0.47684110936075902,-0.23922384547275102,6.4495709619312831e-06},
        {0.86230101504051304,0.39701042639496958,-0.31435597782416247,-7.2072469051445225,1.8058929273786931e-05},
        {0.92523416478522591,-0.031430992865484765,-0.37809236041167926,-0.60721400978497819,6.7793465127248525e-06},
        {0.93343326113611902,0.081430332637871763,-0.34938724637753393,-0.49482467182044776,1.9539716122493648e-05},
        {0.9334562161555171,0.081106216098870365,-0.3494013082842537,-0.49482979074873579,1.9537586841717137e-05},
        {0.96032268091465789,-0.028452317072626541,-0.27743614431808095,-0.60595730806302228,6.9879995677770897e-06}
    };
    AvbdOgcProjectionPlanes planes;
    for(PxU32 i=0;i<sizeof(data)/sizeof(data[0]);++i)
    {
        const AvbdOgcProjectionPlane plane={PxVec3d(data[i][0],data[i][1],data[i][2]),
            data[i][3],data[i][4]};
        planes.pushBack(plane);
    }
    const PxMat33d H(PxVec3d(189781.6875,55715.80078125,-8186.119140625),
        PxVec3d(55715.80078125,355691.71875,-16558.55859375),
        PxVec3d(-8186.119140625,-16558.55859375,164174.640625));
    const PxVec3d target(7.16274977-7.1748275756835938,
        2.73765922-2.8288459777832031,-0.216249749+0.2285371869802475);
    PxVec3d euclidean,metric;
    require(avbdOgcProjectRegion(planes,target,double(PX_MAX_F32),euclidean),
        "captured near-dependent domain has a constructive feasible point");
    require(avbdOgcProjectRegion(planes,target,double(PX_MAX_F32),metric,&H),
        "captured near-dependent Hessian projection exists");
    require(avbdOgcRegionContains(planes,metric,double(PX_MAX_F32)),
        "captured near-dependent Hessian result is admitted");
    const double metricCost=(metric-target).dot(H*(metric-target));
    const double euclideanCost=(euclidean-target).dot(H*(euclidean-target));
    require(metricCost<=euclideanCost+1.0e-9*(1.0+euclideanCost),
        "captured Hessian projection minimizes its metric against a feasible witness");
    std::puts("PASS captured near-dependent Hessian active-set completion");
}

static void testCapturedSurfaceIntervalPublication()
{
 const PxVec3 anchor(-0.688145638f,1.91914737f,-1.26813388f);
 const PxVec3 current(-0.685008228f,1.91926575f,-1.26795733f);
 const PxVec3 candidate(-0.685292304f,1.99752557f,-1.23809552f);
 const PxMat33 H(PxVec3(6730.68311f,-134.57222f,421.288605f),
  PxVec3(-134.57222f,206176.859f,-486.791229f),PxVec3(421.288605f,-486.791229f,5261.55469f));
 const double data[8][5]={
  {1.1920927533992653e-07,1,0,-1.9191474914550781,2.7827361464005662e-06},
  {0,0.82523000240325928,-0.56479698419570923,0,2.2161802917253226e-06},
  {0,0,-1,-0.11813390254974365,1.2513939964264864e-06},
  {1.1920927533992569e-07,0.99999999999999289,0,-1.9191472619757028,2.7550121063429378e-06},
  {0,0,-1,-0.1181339118629694,2.3718111492598837e-06},
  {0,1,0,0.00011837482452392578,3.6993403114138346e-06},
  {0,1,0,0.00011837482452392578,3.6993403114138346e-06},
  {0,-0.84117849934302091,0.54075755403232517,-4.1041746527846053e-06,2.0323462113905537e-05}
 };
 AvbdOgcProjectionPlanes full;
 for(PxU32 i=0;i<8;++i)
 { AvbdOgcProjectionPlane p={PxVec3d(data[i][0],data[i][1],data[i][2]),data[i][3],data[i][4]};full.pushBack(p); }
 const double radius=double(PX_MAX_F32);
 require(avbdOgcRegionContains(full,avbdOgcDouble(current)-avbdOgcDouble(anchor),radius),
  "captured Surface has an already represented feasible point");
 AvbdOgcProjectionPlanes solve=full; avbdOgcCompactParallelPlanes(solve);
 const PxMat33d metric(avbdOgcDouble(H.column0),avbdOgcDouble(H.column1),avbdOgcDouble(H.column2));
 PxVec3d projected;
 require(avbdOgcProjectRegion(solve,avbdOgcDouble(candidate)-avbdOgcDouble(anchor),radius,projected,&metric),
  "captured Surface continuous interval projection exists");
 PxVec3 published;
 require(!avbdOgcPublishRegionPoint(anchor,projected,full,radius,published),
  "captured continuous optimum has no feasible adjacent float vertex");
 require(avbdOgcPublishRegionPoint(anchor,projected,full,radius,published,&current),
  "captured Surface interval projection can publish a feasible float point");
 require(avbdOgcRegionContains(full,avbdOgcDouble(published)-avbdOgcDouble(anchor),radius),
  "captured Surface output stays in original input intervals");
 require(published!=current && (H*(candidate-current)).dot(published-current)>0.0f,
  "captured float publication admits actual Newton descent rather than freezing");
 std::puts("PASS captured Surface interval-domain publication");
}

static void testCoupledInputIntervalCompletion()
{
	PxArray<AvbdCoupledDomainBlock> blocks;
	PxArray<AvbdCoupledDomainConstraint> constraints;
	PxArray<AvbdCoupledDomainTerm> terms;
	blocks.resize(1);
	blocks[0].step=PxVec3d(-2.0,0.0,0.0);
	AvbdCoupledDomainTerm term;
	term.block=0;
	term.gradient=PxVec3d(1.0,0.0,0.0);
	term.inverseMetricGradient=term.gradient;
	terms.pushBack(term);
	AvbdCoupledDomainConstraint constraint={};
	constraint.termStart=0;
	constraint.termCount=1;
	constraint.rhs=0.0;
	constraint.tolerance=0.25;
	constraints.pushBack(constraint);
	double violation=0.0;
	require(avbdSolveCoupledDomainKkt(blocks,constraints,terms,violation,true),
		"coupled interval KKT completes");
	require(blocks[0].step.x>=-0.25-1024.0*DBL_EPSILON &&
		blocks[0].step.x<-0.24,
		"coupled interval KKT solves the admission boundary");
	require(violation<=1024.0*DBL_EPSILON,
		"coupled interval KKT returns a verified feasible direction");
	std::puts("PASS coupled KKT input-interval completion");
}

static void testActivation()
{
	const PxReal r=0.04f, k=1.e5f, tau=r*0.5f;
	const AvbdOGCActivation at=avbdOGCActivationFull(tau,r,k,tau);
	const AvbdOGCActivation below=avbdOGCActivationFull(std::nextafter(tau,0.0f),r,k,tau);
	require(PxAbs(at.energy-below.energy)<1.e-4f && PxAbs(at.force-below.force)<1.e-3f &&
		PxAbs(at.hessian-below.hessian)<0.1f, "OGC C2 stitching in actual PxReal");
	const double distances[] = {0.001,0.019,0.021,0.039,0.05,1.e-11};
	for(double d : distances)
	{
		const double eps=d*1.e-5;
		const auto a=avbdOGCActivationFullT(d,0.04,1.e5,0.02);
		const auto lo=avbdOGCActivationFullT(d-eps,0.04,1.e5,0.02);
		const auto hi=avbdOGCActivationFullT(d+eps,0.04,1.e5,0.02);
		require(std::abs((hi.energy-lo.energy)/(2*eps)+a.force)<1.e-6*(1+a.force), "OGC energy derivative");
		require(std::abs((lo.force-hi.force)/(2*eps)-a.hessian)<1.e-6*(1+a.hessian), "OGC force derivative");
	}
	require(!std::isfinite(avbdOGCActivationFull(0,r,k,tau).energy), "OGC zero gap has no finite energy");
	require(!std::isfinite(avbdOGCActivationFull(-1,r,k,tau).energy), "OGC negative gap rejected");
	std::puts("PASS OGC continuous activation, energy/force/Hessian, positive domain");
}

int main()
{
	PxDefaultAllocator allocator; Errors errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	require(foundation != NULL, "foundation");
	testActivation();
	testCertificates(); testSignedEmbeddedGroundCertificate(); testProjection(); testComposedCommit(); testCapturedEmptyDomain(); testMetricProjection();
	testCapturedNearDependentMetricProjection();
	testCapturedFloatPublication();
	testParallelDomainImplication();
	testCapturedSurfaceIntervalPublication();
	testCoupledInputIntervalCompletion();
	foundation->release();
	std::puts("PASS OGC feature identity, capacity, copy, intersection, ball, infeasibility, 200 KKT cases");
	return 0;
}
