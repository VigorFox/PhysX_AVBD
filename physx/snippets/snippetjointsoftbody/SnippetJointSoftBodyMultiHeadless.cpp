// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "SnippetJointSoftBodyShowcase.h"
#include "SnippetJointSoftBodyGeometry.h"
#include "SnippetJointSoftBodyCapture.h"
#include "SnippetJointSoftBodyAssetCapture.h"
#include "../snippetcommon/SnippetSoftTipObservation.h"
#include "../snippetcommon/SnippetSoftReboundObservation.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <map>

namespace JointSoftBodySnippet
{
namespace
{
typedef std::chrono::steady_clock Clock;
// Freeze the closest reported normal of the first prism/sphere impact.
struct PrismImpactEvents : Events
{
    Events& events;
    bool observing;
    PxReal closest=PX_MAX_F32;
    PxVec3 normal=PxVec3(0);
    PrismImpactEvents(Events& parent,bool enabled) : events(parent),observing(enabled) {}
    void onContact(const PxContactPairHeader& header,const PxContactPair* pairs,PxU32 count) PX_OVERRIDE
    {
        events.onContact(header,pairs,count);
        if(!observing || header.flags)return;
        for(PxU32 i=0;i<count;++i)
        {
            const auto& pair=pairs[i];
            if(!pair.contactCount || (pair.flags & (PxContactPairFlag::eREMOVED_SHAPE_0|PxContactPairFlag::eREMOVED_SHAPE_1)))continue;
            const auto a=pair.shapes[0]->getSimulationFilterData().word0,b=pair.shapes[1]->getSimulationFilterData().word0;
            if(!((a==1 && b==2)||(a==2 && b==1)))continue;
            std::vector<PxContactPairPoint> points(pair.contactCount);
            const PxU32 n=pair.extractContacts(points.data(),pair.contactCount);
            for(PxU32 j=0;j<n;++j)if(points[j].separation<closest)
            {closest=points[j].separation;normal=a==1?points[j].normal:-points[j].normal;}
        }
    }
};
double ms(Clock::time_point a,Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
// Read-only NP witness transport. This observer never supplies solver rows.
struct WitnessTrace : Events
{
 struct Sample { const PxRigidActor* a;const PxRigidActor* b;PxVec3 pa,pb,n;PxReal separation; };
 Events& events;std::map<const PxRigidActor*,PxTransform> before;std::vector<Sample> samples;
 std::map<const PxRigidActor*,PxVec3> linearBefore,angularBefore;
 WitnessTrace(Events& e) : events(e) {}
 void onContact(const PxContactPairHeader& header,const PxContactPair* pairs,PxU32 count) PX_OVERRIDE
 {
  events.onContact(header,pairs,count);
  if(header.flags) return;
  const PxRigidActor* a=header.actors[0]->is<PxRigidActor>();const PxRigidActor* b=header.actors[1]->is<PxRigidActor>();
  if(!a || !b) return;
  const PxTransform ta=before.count(a)?before[a]:a->getGlobalPose(),tb=before.count(b)?before[b]:b->getGlobalPose();
  for(PxU32 i=0;i<count;++i)
  {
   PxArray<PxContactPairPoint> points(pairs[i].contactCount);
   const PxU32 n=pairs[i].extractContacts(points.begin(),points.size());
   for(PxU32 j=0;j<n;++j) samples.push_back({a,b,ta.transformInv(points[j].position),tb.transformInv(points[j].position),points[j].normal,points[j].separation});
  }
 }
 PxReal minimumGap() const
 {
  PxReal gap=PX_MAX_F32;
  for(size_t i=0;i<samples.size();++i)
  {
   const Sample& s=samples[i];
   gap=PxMin(gap,s.n.dot(s.a->getGlobalPose().transform(s.pa)-s.b->getGlobalPose().transform(s.pb))+s.separation);
  }
  return gap;
 }
};
struct ExternalFilter : PxQueryFilterCallback
{
	const PxRigidActor* source;PxFilterData sourceData;
	ExternalFilter(const PxRigidActor* a,const PxFilterData& data) : source(a),sourceData(data) {}
	PxQueryHitType::Enum preFilter(const PxFilterData&,const PxShape* shape,const PxRigidActor* actor,PxHitFlags&) PX_OVERRIDE
	{ return actor==source || excludedCellPair(sourceData,shape->getSimulationFilterData())?PxQueryHitType::eNONE:PxQueryHitType::eTOUCH; }
	PxQueryHitType::Enum postFilter(const PxFilterData&,const PxQueryHit&,const PxShape*,const PxRigidActor*) PX_OVERRIDE { return PxQueryHitType::eTOUCH; }
};
struct Penetration : PxOverlapCallback
{
	PxOverlapHit buffer[64];const PxShape& source;PxTransform pose;PxReal depth,selfDepth;const PxRigidActor* external;
	Penetration(const PxShape& shape,const PxRigidActor& actor) : PxOverlapCallback(buffer,64),source(shape),pose(PxShapeExt::getGlobalPose(shape,actor)),depth(0),selfDepth(0),external(NULL) {}
	void inspect(const PxOverlapHit* hits,PxU32 count)
	{
		for(PxU32 i=0;i<count;++i)
		{
			const PxReal value=observedPenetration(source.getGeometry(),pose,hits[i].shape->getGeometry(),PxShapeExt::getGlobalPose(*hits[i].shape,*hits[i].actor));
			if(source.getSimulationFilterData().word0==hits[i].shape->getSimulationFilterData().word0) selfDepth=PxMax(selfDepth,value);
			else if(value>depth) { depth=value;external=hits[i].actor; }
		}
	}
	PxAgain processTouches(const PxOverlapHit* hits,PxU32 count) PX_OVERRIDE { inspect(hits,count);return true; }
	void finalizeQuery() PX_OVERRIDE { inspect(touches,nbTouches); }
};
struct Observation
{
	std::vector<PxReal> bodyDet,bodyVolume;
	std::vector<PxVec3> bodyCenter,bodyVelocity;
	std::vector<PxU32> bodyMinTet;
	const PxRigidActor* penetrationA=NULL;const PxRigidActor* penetrationB=NULL;
	std::vector<PxTransform> poses;PxVec3 momentum;double energy;PxReal momentumScale,penetration,selfPenetration,minDet,minVolume,maxVolume,jointError,maxSpeed;
	PxU32 awake;bool finite;
	Observation() : momentum(0),energy(0),momentumScale(0),penetration(0),selfPenetration(0),minDet(1),minVolume(1),maxVolume(1),jointError(0),maxSpeed(0),awake(0),finite(true) {}
};
struct CenterTet { PxU32 ids[4];PxReal restDet; };
typedef std::vector<std::vector<CenterTet> > AssetObservations;
void bodyState(const PxRigidDynamic& body,const PxVec3& gravity,Observation& o)
{
	const PxTransform actor=body.getGlobalPose(),center=actor*body.getCMassLocalPose();const PxVec3 v=body.getLinearVelocity(),w=center.q.rotateInv(body.getAngularVelocity());
	o.poses.push_back(actor);o.finite&=actor.isFinite() && v.isFinite() && w.isFinite();o.awake+=!body.isSleeping();
	o.momentum+=v*body.getMass();o.momentumScale+=v.magnitude()*body.getMass();o.maxSpeed=PxMax(o.maxSpeed,v.magnitude());
	o.energy+=0.5*double(body.getMass()*v.magnitudeSquared()+w.dot(body.getMassSpaceInertiaTensor().multiply(w)))-double(body.getMass()*gravity.dot(center.p));
}
Observation observe(ShowcaseWorld& world,const AssetObservations& observations,bool geometry)
{
	Observation o;const PxVec3 gravity=world.host.scene->getGravity();
	for(PxU32 i=0;i<world.bodies.size();++i)
	{
		const JointAsset& asset=*world.bodies[i];const Lattice& l=asset.lattice;const PxU32 start=PxU32(o.poses.size());
		PxVec3d center(0),velocity(0);double mass=0;
		for(PxU32 n=0;n<l.nodes.size();++n)
		{
			bodyState(*l.nodes[n],gravity,o);
			const auto p=(l.nodes[n]->getGlobalPose()*l.nodes[n]->getCMassLocalPose()).p;
			const auto v=l.nodes[n]->getLinearVelocity();const double m=l.nodes[n]->getMass();mass+=m;
			center+=PxVec3d(p.x,p.y,p.z)*m;velocity+=PxVec3d(v.x,v.y,v.z)*m;
			if(geometry)
			{
				PxShape* shape=NULL;l.nodes[n]->getShapes(&shape,1);Penetration penetration(*shape,*l.nodes[n]);ExternalFilter filter(l.nodes[n],shape->getSimulationFilterData());
				const PxQueryFilterData data(PxQueryFlag::eSTATIC|PxQueryFlag::eDYNAMIC|PxQueryFlag::ePREFILTER|PxQueryFlag::eNO_BLOCK);
				world.host.scene->overlap(shape->getGeometry(),penetration.pose,penetration,data,&filter);
				if(penetration.depth>o.penetration) { o.penetration=penetration.depth;o.penetrationA=l.nodes[n];o.penetrationB=penetration.external; }
				o.selfPenetration=PxMax(o.selfPenetration,penetration.selfDepth);
			}
		}
		center/=mass;velocity/=mass;
		o.bodyCenter.push_back(PxVec3(PxReal(center.x),PxReal(center.y),PxReal(center.z)));
		o.bodyVelocity.push_back(PxVec3(PxReal(velocity.x),PxReal(velocity.y),PxReal(velocity.z)));
		for(PxU32 j=0;j<l.joints.size();++j)
		{
			PxRigidActor* a=NULL;PxRigidActor* b=NULL;l.joints[j]->getActors(a,b);
			const PxTransform pa=a->getGlobalPose()*l.joints[j]->getLocalPose(PxJointActorIndex::eACTOR0),pb=b->getGlobalPose()*l.joints[j]->getLocalPose(PxJointActorIndex::eACTOR1);
			const PxVec3 d=pa.q.rotateInv(pb.p-pa.p);o.jointError=PxMax(o.jointError,d.magnitude()/l.cell);
			// Observe the native material coordinates without feeding the solver.
			o.energy+=l.joints[j]->elasticEnergy();
		}
		for(PxU32 m=0;m<l.materials.size();++m) o.energy+=l.materials[m]->elasticEnergy();
		double current=0,rest=0;PxReal minDet=1;PxU32 minTet=0;
		for(PxU32 t=0;t<observations[i].size();++t)
		{
			const CenterTet& tet=observations[i][t];PxVec3 centers[4];
			for(PxU32 k=0;k<4;++k) centers[k]=o.poses[start+tet.ids[k]].transform(l.nodes[tet.ids[k]]->getCMassLocalPose().p);
			const PxReal d=(centers[1]-centers[0]).dot((centers[2]-centers[0]).cross(centers[3]-centers[0]));
			const PxReal ratio=d/tet.restDet;if(ratio<minDet) { minDet=ratio;minTet=t; }current+=double(ratio)*PxAbs(tet.restDet);rest+=PxAbs(tet.restDet);
		}
		o.minDet=PxMin(o.minDet,minDet);o.bodyDet.push_back(minDet);o.bodyMinTet.push_back(minTet);
		const PxReal ratio=rest>0?PxReal(current/rest):1;o.bodyVolume.push_back(ratio);o.minVolume=PxMin(o.minVolume,ratio);o.maxVolume=PxMax(o.maxVolume,ratio);
	}
	if(world.host.impactor) bodyState(*world.host.impactor,gravity,o);
	o.finite&=std::isfinite(o.energy) && o.momentum.isFinite() && PxIsFinite(o.minVolume) && PxIsFinite(o.minDet);return o;
}
std::vector<CenterTet> bakeObservations(const JointAsset& asset)
{
	std::vector<CenterTet> result;const Lattice& l=asset.lattice;std::vector<PxU32> map(l.grid*l.grid*l.grid,PX_MAX_U32);
	if(asset.skinRest.empty()) for(PxU32 i=0;i<l.nodes.size();++i) map[i]=i;
	else
	{
		PxBounds3 bounds=PxBounds3::empty();for(PxU32 i=0;i<asset.skinRest.size();++i) bounds.include(asset.skinRest[i]);const PxVec3 cell=(bounds.maximum-bounds.minimum)/PxReal(l.grid);
		for(PxU32 i=0;i<l.restLocal.size();++i) { const PxVec3 p=l.restLocal[i]-bounds.minimum;const PxU32 x=PxU32(p.x/cell.x),y=PxU32(p.y/cell.y),z=PxU32(p.z/cell.z);map[l.index(x,y,z)]=i; }
	}
	static const PxU32 permutations[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
	for(PxU32 z=0;z+1<l.grid;++z) for(PxU32 y=0;y+1<l.grid;++y) for(PxU32 x=0;x+1<l.grid;++x) for(PxU32 t=0;t<6;++t)
	{
		PxU32 c[3]={x,y,z};CenterTet tet;tet.ids[0]=map[l.index(x,y,z)];++c[permutations[t][0]];tet.ids[1]=map[l.index(c[0],c[1],c[2])];++c[permutations[t][1]];tet.ids[2]=map[l.index(c[0],c[1],c[2])];tet.ids[3]=map[l.index(x+1,y+1,z+1)];
		bool complete=true;for(PxU32 k=0;k<4;++k) complete&=tet.ids[k]!=PX_MAX_U32;if(!complete) continue;
		// Cell origins can be outside a clipped solid. Observe material COMs,
		// preserving the original baked grid incidence and the actor pose hash.
		PxVec3 centers[4];for(PxU32 k=0;k<4;++k) centers[k]=l.restLocal[tet.ids[k]]+l.nodes[tet.ids[k]]->getCMassLocalPose().p;
		tet.restDet=(centers[1]-centers[0]).dot((centers[2]-centers[0]).cross(centers[3]-centers[0]));result.push_back(tet);
	}
	return result;
}
void hashObservation(PxU64& hash,const Observation& o)
{
	for(size_t i=0;i<o.poses.size();++i)
	{
		const PxTransform& p=o.poses[i];const PxReal values[7]={p.p.x,p.p.y,p.p.z,p.q.x,p.q.y,p.q.z,p.q.w};
		for(PxU32 j=0;j<7;++j) { PxU32 bits;std::memcpy(&bits,values+j,4);for(PxU32 k=0;k<4;++k) { hash^=(bits>>(8*k))&255;hash*=1099511628211ULL; } }
	}
}
void traceFailure(PxU32 frame,const ShowcaseWorld& world,const AssetObservations& observations,
 const Observation& current,const WitnessTrace& witnesses)
{
 const PxRigidActor* pair[2]={current.penetrationA,current.penetrationB};
 if(pair[0] && pair[1])
 {
  PxReal gap=PX_MAX_F32,initial=PX_MAX_F32;PxU32 count=0;
  for(size_t i=0;i<witnesses.samples.size();++i)
  {
   const WitnessTrace::Sample& s=witnesses.samples[i];
   if(!((s.a==pair[0] && s.b==pair[1]) || (s.a==pair[1] && s.b==pair[0]))) continue;
   ++count;initial=PxMin(initial,s.separation);
   gap=PxMin(gap,s.n.dot(s.a->getGlobalPose().transform(s.pa)-s.b->getGlobalPose().transform(s.pb))+s.separation);
  }
  PxShape* shapes[2];pair[0]->getShapes(&shapes[0],1);pair[1]->getShapes(&shapes[1],1);
  PxTransform before[2];for(PxU32 k=0;k<2;++k) { const auto it=witnesses.before.find(pair[k]);before[k]=(it==witnesses.before.end()?pair[k]->getGlobalPose():it->second)*shapes[k]->getLocalPose(); }
  const PxReal dcdDepth=observedPenetration(shapes[0]->getGeometry(),before[0],shapes[1]->getGeometry(),before[1]);
  std::printf("[JOINT_PAIR_TRACE] frame=%u depth=%.9g dcdDepth=%.9g samples=%u initialGap=%.9g finalGap=%.9g",frame,current.penetration,dcdDepth,count,initial,gap);
  for(PxU32 k=0;k<2;++k)
  {
   PxShape* shape=NULL;pair[k]->getShapes(&shape,1);const PxFilterData d=shape->getSimulationFilterData();
   std::printf(" actor%u=%u,%u,%u,%u",k,d.word0,d.word1,d.word2,d.word3);
  }
  std::printf("\n");
  for(PxU32 k=0;k<2;++k)
  {
   const PxRigidDynamic* body=pair[k]->is<PxRigidDynamic>();if(!body) continue;
   const auto pose=witnesses.before.find(pair[k]);
   const auto linear=witnesses.linearBefore.find(pair[k]),angular=witnesses.angularBefore.find(pair[k]);
   if(pose==witnesses.before.end() || linear==witnesses.linearBefore.end() || angular==witnesses.angularBefore.end()) continue;
   const PxTransform com0=pose->second*body->getCMassLocalPose(),com1=body->getGlobalPose()*body->getCMassLocalPose();
   PxBounds3 bounds;PxGeometryQuery::computeGeomBounds(bounds,shapes[k]->getGeometry(),before[k]);
   const float radius=bounds.getExtents().magnitude(),offset=(bounds.getCenter()-com0.p).magnitude();
   const PxQuat rotation=(com1.q*com0.q.getConjugate()).getNormalized();
   const float angle=2*PxAtan2(rotation.getImaginaryPart().magnitude(),PxAbs(rotation.w));
   std::printf("[JOINT_QUERY_TRACE] frame=%u side=%u contactOffset=%.9g radius=%.9g comOffset=%.9g linearReach=%.9g angularReach=%.9g actualTranslation=%.9g actualAngle=%.9g\n",
    frame,k,shapes[k]->getContactOffset(),radius,offset,linear->second.magnitude()/60,angular->second.magnitude()*radius/60,
    (com1.p-com0.p).magnitude(),angle);
  }
 }
 for(PxU32 b=0;b<world.bodies.size();++b)
 {
  if(observations[b].empty()) continue;
  const CenterTet& t=observations[b][current.bodyMinTet[b]];const Lattice& l=world.bodies[b]->lattice;
  std::printf("[JOINT_TET_TRACE] frame=%u body=%u tet=%u det=%.9g restDet=%.9g nodes=%u,%u,%u,%u\n",frame,b,current.bodyMinTet[b],current.bodyDet[b],t.restDet,t.ids[0],t.ids[1],t.ids[2],t.ids[3]);
  if(current.bodyDet[b]>=0.2f) continue;
  for(PxU32 k=0;k<4;++k)
  {
   const PxRigidDynamic& node=*l.nodes[t.ids[k]];const PxVec3 p=node.getGlobalPose().transform(node.getCMassLocalPose().p);
   const PxVec3 r=l.restLocal[t.ids[k]]+node.getCMassLocalPose().p;
   std::printf("[JOINT_NODE_TRACE] frame=%u body=%u node=%u mass=%.9g rest=%.9g,%.9g,%.9g center=%.9g,%.9g,%.9g\n",frame,b,t.ids[k],node.getMass(),r.x,r.y,r.z,p.x,p.y,p.z);
  }
 }
}
}
int runJointSoftBodyMultiHeadless(const Settings& settings,const char* name,PxU32 frames)
{
	const bool pair=std::strcmp(name,"multi-collision")==0;ShowcaseWorld world;
	if(!world.init(settings,name)) { std::fprintf(stderr,"Multi joint-soft initialization failed.\n");return 2; }
    const bool observePrism=!pair && world.bodies.size()==5;
    PrismImpactEvents impactEvents(world.host.events,observePrism);
    Snippets::ReboundObservation prismImpact;
	WitnessTrace witnessTrace(impactEvents);
	const char* capturePath=std::getenv("PHYSX_AVBD_MECHANISM_CAPTURE");
	std::unique_ptr<FILE,int(*)(FILE*)> capture(capturePath?std::fopen(capturePath,"wb"):NULL,std::fclose);
	if(capturePath && !capture) { std::fprintf(stderr,"Cannot create mechanism capture.\n");return 2; }
	const bool trace=std::getenv("PHYSX_AVBD_HEADLESS_TRACE")!=NULL || capture.get()!=NULL;
	if(trace) world.host.scene->setSimulationEventCallback(&witnessTrace);
    else if(observePrism)world.host.scene->setSimulationEventCallback(&impactEvents);
	if(!geometryOracleValidated(*world.host.physics)) { std::fprintf(stderr,"Headless geometry oracle validation failed.\n");return 2; }
	AssetObservations observations;PxU32 nodes=0,joints=0,materialElements=0;PxU64 skinBytes=0,assetBytes=0,diagnosticBytes=0;PxReal mass=0,cell=PX_MAX_F32,skinError=0,massError=0;
	for(PxU32 i=0;i<world.bodies.size();++i)
	{
		JointAsset& asset=*world.bodies[i];const Lattice& l=asset.lattice;observations.push_back(bakeObservations(asset));
		nodes+=l.nodes.size();joints+=l.joints.size();materialElements+=l.materials.size();mass+=l.totalMass;cell=PxMin(cell,l.cell);skinError=PxMax(skinError,PxMax(asset.restSkinError,asset.rigidSkinError));
		skinBytes+=asset.skinBytes();assetBytes+=PxU64(l.nodes.size())*(sizeof(PxRigidDynamic*)+sizeof(PxVec3))+PxU64(l.joints.size())*sizeof(FiniteJoint*)+PxU64(l.materials.size())*(sizeof(VolumeMaterial*)+sizeof(VolumeMaterial));
		diagnosticBytes+=PxU64(observations[i].size())*sizeof(CenterTet);
		const PxReal density=i==1?130.f:160.f;const PxReal expected=pair?l.totalMass:asset.proxyVolume*density;
		massError=PxMax(massError,PxAbs(l.totalMass-expected)/expected);
		PxU32 boxes=0;for(PxU32 n=0;n<l.nodes.size();++n) { PxShape* shape=NULL;l.nodes[n]->getShapes(&shape,1);boxes+=shape->getGeometry().getType()==PxGeometryType::eBOX; }
		std::printf("[JOINT_SHOWCASE_BODY] body=%u grid=%u nodes=%u joints=%u boxes=%u convexMeshes=%u sourceVolume=%.9g proxyVolume=%.9g mass=%.9g sourceMass=%.9g proxyMass=%.9g skinVertices=%u skinTriangles=%u skinBytes=%llu assetBytes=%llu diagnosticTets=%u envelopeError=%.9g envelopeExpansion=%.9g rigidSkinError=%.9g\n",i,l.grid,l.nodes.size(),l.joints.size(),boxes,l.nodes.size()-boxes,asset.sourceVolume,asset.proxyVolume,l.totalMass,asset.sourceVolume*density,expected,asset.skinRest.size(),asset.skinTriangles.size()/3,static_cast<unsigned long long>(asset.skinBytes()),static_cast<unsigned long long>(PxU64(l.nodes.size())*(sizeof(PxRigidDynamic*)+sizeof(PxVec3))+PxU64(l.joints.size())*sizeof(FiniteJoint*)+PxU64(l.materials.size())*(sizeof(VolumeMaterial*)+sizeof(VolumeMaterial))),PxU32(observations[i].size()),asset.envelopeError,asset.envelopeExpansion,asset.rigidSkinError);
	}
	const Observation initial=observe(world,observations,true);Observation previous=initial,current=initial;PxU64 hash=14695981039346656037ULL;hashObservation(hash,initial);
	const bool traceTip=!pair && world.bodies.size()==5 && std::getenv("PHYSX_AVBD_SHOWCASE_MOTION");
	Snippets::SoftTipObservation lowerAxis;
	if(traceTip)
	{
		const auto& l=world.bodies[4]->lattice;
		lowerAxis.bake(PxU32(l.nodes.size()),[&](PxU32 i){return l.restLocal[i].x+l.nodes[i]->getCMassLocalPose().p.x;},
			[&](PxU32 i){return double(l.nodes[i]->getMass());});
		double angle=0,rate=0;
		lowerAxis.sample([&](PxU32 i){return (l.nodes[i]->getGlobalPose()*l.nodes[i]->getCMassLocalPose()).p;},
			[](PxU32){return PxVec3(0);},angle,rate);
		lowerAxis.initialAngle=angle;
	}
	// The four elevated assets have no external support during the first half
	// second. Internal material/contact forces must preserve their gravity mode,
	// even while the low sphere is hitting the ground in another component.
	std::vector<double> gravityY,gravityV,gravityDamping;
	for(PxU32 i=0;i<world.bodies.size();++i)
	{ gravityY.push_back(initial.bodyCenter[i].y);gravityV.push_back(initial.bodyVelocity[i].y);gravityDamping.push_back(world.bodies[i]->lattice.nodes[0]->getLinearDamping()); }
	double gravityPositionError=0,gravitySpeedError=0;PxU32 gravitySamples=0;
	if(!captureShowcaseAsset(world,settings,name,initial.energy)) { std::fprintf(stderr,"Cannot capture complete showcase asset.\n");return 2; }
	world.host.allocator.report("joint-initialized");
	PxReal minDet=1,minVolume=1,maxVolume=1,maxPenetration=initial.penetration,maxSelfPenetration=initial.selfPenetration,maxJoint=0,maxMomentum=0,maxMotion=0,maxEnergy=PxReal(initial.energy);
	PxU32 completed=0,contacts=0,crossContacts=0,crossMask=0,activeContacts=0,nonFinite=0,fetchFailures=0;
	double sum=0,contactSum=0,skinSum=0,observationSum=0;std::vector<double> steps;
	for(PxU32 f=0;f<frames;++f)
	{
		if(trace)
		{
			witnessTrace.before.clear();witnessTrace.samples.clear();
			witnessTrace.linearBefore.clear();witnessTrace.angularBefore.clear();
			for(PxU32 i=0;i<world.bodies.size();++i) for(PxU32 j=0;j<world.bodies[i]->lattice.nodes.size();++j)
			{ PxRigidDynamic* b=world.bodies[i]->lattice.nodes[j];witnessTrace.before[b]=b->getGlobalPose();
			  witnessTrace.linearBefore[b]=b->getLinearVelocity();witnessTrace.angularBefore[b]=b->getAngularVelocity(); }
			if(world.host.impactor) witnessTrace.before[world.host.impactor]=world.host.impactor->getGlobalPose();
		}
		const auto start=Clock::now();if(!world.step()) { ++fetchFailures;break; }const auto fetched=Clock::now();
		if(world.host.errors.fatal.load())break;
		const double elapsed=ms(start,fetched);steps.push_back(elapsed);sum+=elapsed;++completed;
		for(PxU32 i=0;i<world.bodies.size();++i) world.bodies[i]->updateSkin();const auto skinned=Clock::now();skinSum+=ms(fetched,skinned);
		current=observe(world,observations,true);observationSum+=ms(skinned,Clock::now());if(!current.finite) { ++nonFinite;break; }hashObservation(hash,current);
		if(!pair && f<30)for(PxU32 i=0;i<world.bodies.size();++i)if(i!=1)
		{
			const double dt=double(1.f/60.f),g=world.host.scene->getGravity().y;
			gravityY[i]+=(gravityV[i]+g*dt)*dt;
			gravityV[i]=(gravityV[i]+g*dt)*PxMax(0.0,1.0-gravityDamping[i]*dt);
			gravityPositionError=PxMax(gravityPositionError,std::abs(double(current.bodyCenter[i].y)-gravityY[i]));
			gravitySpeedError=PxMax(gravitySpeedError,std::abs(double(current.bodyVelocity[i].y)-gravityV[i]));++gravitySamples;
		}
		capturePair(capture.get(),f+1,current.penetrationA,current.penetrationB,current.penetration,witnessTrace);
		minDet=PxMin(minDet,current.minDet);minVolume=PxMin(minVolume,current.minVolume);maxVolume=PxMax(maxVolume,current.maxVolume);
		maxPenetration=PxMax(maxPenetration,current.penetration);maxJoint=PxMax(maxJoint,current.jointError);maxEnergy=PxMax(maxEnergy,PxReal(current.energy));
		maxSelfPenetration=PxMax(maxSelfPenetration,current.selfPenetration);
		maxMomentum=PxMax(maxMomentum,(current.momentum-initial.momentum).magnitude()/PxMax(initial.momentumScale,1.f));
		contacts+=world.host.events.contacts;crossContacts+=world.host.events.crossContacts;crossMask|=world.host.events.crossMask;
		if(world.host.events.contacts && (current.awake || previous.awake)) { ++activeContacts;contactSum+=elapsed; }
		if(contacts) for(PxU32 i=0;i<nodes;++i) maxMotion=PxMax(maxMotion,(current.poses[i].p-previous.poses[i].p).magnitude());
        if(observePrism && !impactEvents.normal.isZero())
        {
            const bool near=(world.host.events.crossMask&((1u<<1)|(1u<<5)))!=0;
            const auto delta=current.bodyCenter[0]-current.bodyCenter[1];
            const auto velocity=current.bodyVelocity[0]-current.bodyVelocity[1];
            if(impactEvents.observing)
            {
                if(impactEvents.normal.dot(delta)<0)impactEvents.normal=-impactEvents.normal;
                prismImpact.incoming=PxMax(0.f,-impactEvents.normal.dot(previous.bodyVelocity[0]-previous.bodyVelocity[1]));
                impactEvents.observing=false;
            }
            prismImpact.sample(f+1,near,impactEvents.normal.dot(delta),impactEvents.normal.dot(velocity),current.bodyVolume[0],0);
        }
		previous=current;
		if(traceTip)
		{
			const auto& l=world.bodies[4]->lattice;
			double angle=0,rate=0;
			if(lowerAxis.sample([&](PxU32 i){return (l.nodes[i]->getGlobalPose()*l.nodes[i]->getCMassLocalPose()).p;},
				[&](PxU32 i){return l.nodes[i]->getLinearVelocity();},angle,rate))
				std::printf("[JAW_TIP] frame=%u angleRad=%.8g rateRadS=%.8g\n",f+1,angle,rate);
		}
		if(trace) { std::printf("[JOINT_WITNESS_TRACE] frame=%u samples=%u minTransportedGap=%.9g penetration=%.9g\n",f+1,PxU32(witnessTrace.samples.size()),witnessTrace.minimumGap(),current.penetration);traceFailure(f+1,world,observations,current,witnessTrace); }
		if(std::getenv("PHYSX_AVBD_HEADLESS_TRACE") || std::getenv("PHYSX_AVBD_SHOWCASE_MOTION") || (std::getenv("PHYSX_AVBD_HEADLESS_PROGRESS") && ((f+1)%60==0 || f+1==frames)))
		{ std::printf("[JOINT_MULTI_PROGRESS] frame=%u stepMs=%.6g minDet=%.9g volume=%.9g penetration=%.9g selfPenetration=%.9g contacts=%u crossContacts=%u awake=%u energy=%.9g jointError=%.9g\n",f+1,elapsed,current.minDet,current.minVolume,current.penetration,current.selfPenetration,world.host.events.contacts,world.host.events.crossContacts,current.awake,current.energy,current.jointError);for(PxU32 i=0;i<world.bodies.size();++i) std::printf("[JOINT_BODY_PROGRESS] frame=%u body=%u minDet=%.9g volume=%.9g centerY=%.9g velocityY=%.9g\n",f+1,i,current.bodyDet[i],current.bodyVolume[i],double(current.bodyCenter[i].y),double(current.bodyVelocity[i].y));if(world.host.impactor)std::printf("[JOINT_RIGID_PROGRESS] frame=%u centerY=%.9g velocityY=%.9g\n",f+1,double(world.host.impactor->getGlobalPose().p.y),double(world.host.impactor->getLinearVelocity().y));std::fflush(stdout); }
	}
	std::sort(steps.begin(),steps.end());const double p95=steps.empty()?0:steps[(steps.size()*95+99)/100-1],peak=steps.empty()?0:steps.back();
	world.host.allocator.report("joint-completed");
	const PxU32 bodyCount=world.bodies.size(),awake=current.awake;const PxU32 fatal=world.host.errors.fatal.load();world.release();world.host.allocator.report("joint-released");
	const bool gravityPass=pair || (gravitySamples==120 && gravityPositionError<.005 && gravitySpeedError<.01);
	if(!pair)std::printf("[JOINT_GRAVITY] samples=%u positionError=%.9g speedError=%.9g status=%s\n",gravitySamples,gravityPositionError,gravitySpeedError,gravityPass?"PASS":"FAIL");
    if(observePrism)std::printf("[FIVE_REBOUND] pair=prism-sphere axis=incidentNormal windowFrames=60 contactFrame=%u apexFrame=%u departureFrame=%u incoming=%.8g up=%.8g rise=%.8g minVolume=%.8g recoveredVolume=%.8g\n",
        prismImpact.contactFrame,prismImpact.apexFrame,prismImpact.departureFrame,prismImpact.incoming,prismImpact.up,
        prismImpact.rise,prismImpact.minVolume,prismImpact.apexVolume);
	const char* reason="none";
	if(nonFinite || fetchFailures || fatal) reason="runtime";
	else if(frames<360 || completed!=frames) reason="incomplete";
	else if(!gravityPass) reason="gravity";
	else if(skinError>1.e-5f || massError>1.e-4f) reason="asset_contract";
	else if(minDet<0.2f || minVolume<0.55f || current.minVolume<0.9f) reason="shape_or_volume";
	else if(maxPenetration>0.01f*cell) reason="penetration";
	else if(maxSelfPenetration>0.01f*cell) reason="self_penetration";
	else if(!crossContacts || maxMotion<1.e-4f) reason="cross_contact_motion";
	else if(maxEnergy>PxMax(PxReal(initial.energy)*2.f,1.f) || (pair && maxMomentum>0.01f)) reason="energy_or_reaction";
	const bool pass=std::strcmp(reason,"none")==0;
	std::printf("[AVBD_GATE] schema=1 snippet=SnippetJointSoftBody solver=%s case=%s frames=%u completedFrames=%u status=%s reason=%s capability=EXPERIMENTAL validation=GATED model=%s materialElements=%u grid=%u dispatcherThreads=%u bodies=%u nodes=%u joints=%u mass=%.9g cellSize=%.9g trajectoryHash=%016llx contacts=%u crossContacts=%u crossMask=%u minCellDet=%.9g minVolumeRatio=%.9g maxVolumeRatio=%.9g finalVolumeRatio=%.9g maxExternalPenetration=%.9g maxSelfPenetration=%.9g maxJointAnchorError=%.9g maxPostContactStep=%.9g maxMomentumRelativeError=%.9g initialMechanicalEstimate=%.9g maxMechanicalEstimate=%.9g finalMechanicalEstimate=%.9g finalAwake=%u skinRestError=%.9g massRelativeError=%.9g assetBytes=%llu skinBytes=%llu diagnosticBytes=%llu averageStepMs=%.9g p95StepMs=%.9g maxStepMs=%.9g activeContactFrames=%u averageActiveContactMs=%.9g averageSkinMs=%.9g averageObserverMs=%.9g averageFrameMs=%.9g nonFinite=%u fetchFailures=%u fatalErrors=%u\n",settings.solver==PxSolverType::eAVBD?"avbd":settings.solver==PxSolverType::eTGS?"tgs":"pgs",name,frames,completed,pass?"PASS":"FAIL",reason,pair?(settings.volumeMaterial?"rigid_cell_joint_volume":"rigid_cell_joint"):(settings.volumeMaterial?"convex_cell_joint_volume":"convex_cell_joint"),materialElements,settings.grid,settings.workers,bodyCount,nodes,joints,mass,cell,static_cast<unsigned long long>(hash),contacts,crossContacts,crossMask,minDet,minVolume,maxVolume,current.minVolume,maxPenetration,maxSelfPenetration,maxJoint,maxMotion,maxMomentum,initial.energy,maxEnergy,current.energy,awake,skinError,massError,static_cast<unsigned long long>(assetBytes),static_cast<unsigned long long>(skinBytes),static_cast<unsigned long long>(diagnosticBytes),completed?sum/completed:0,p95,peak,activeContacts,activeContacts?contactSum/activeContacts:0,completed?skinSum/completed:0,completed?observationSum/completed:0,completed?(sum+skinSum)/completed:0,nonFinite,fetchFailures,fatal);
	std::fflush(stdout);return pass?0:1;
}
}
