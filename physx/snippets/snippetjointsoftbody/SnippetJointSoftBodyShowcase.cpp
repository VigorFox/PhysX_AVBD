// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "SnippetJointSoftBodyShowcase.h"
#include "../snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDCommon.h"
#include "../snippetdeformablevolume/MeshGenerator.h"
#include <algorithm>
#include <cmath>

namespace JointSoftBodySnippet
{
namespace
{
struct Face { PxArray<PxVec3> points; };
typedef PxArray<Face> Polyhedron;
void uniquePoint(PxArray<PxVec3>& points,const PxVec3& p,PxReal epsilon)
{
	for(PxU32 i=0;i<points.size();++i) if((points[i]-p).magnitudeSquared()<=epsilon*epsilon) return;
	points.pushBack(p);
}
void clipPolygon(PxArray<PxVec3>& points,const PxPlane& plane,PxArray<PxVec3>* cap,PxReal epsilon)
{
	PxArray<PxVec3> output;
	for(PxU32 i=0;i<points.size();++i)
	{
		PxVec3 a=points[i],b=points[(i+1)%points.size()];
		PxReal da=plane.distance(a),db=plane.distance(b);
		if(PxAbs(da)<=epsilon) { a-=plane.n*da;da=0; }
		if(PxAbs(db)<=epsilon) { b-=plane.n*db;db=0; }
		if(da<=0) output.pushBack(a);
		if((da<0 && db>0) || (da>0 && db<0))
		{
			const PxVec3 p=a+(b-a)*(da/(da-db)); output.pushBack(p);
			if(cap) uniquePoint(*cap,p,epsilon);
		}
		if(cap && PxAbs(da)<=epsilon) uniquePoint(*cap,a,epsilon);
	}
	points.swap(output);
}
void sortPolygon(PxArray<PxVec3>& points,const PxVec3& normal)
{
	PxVec3 center(0); for(PxU32 i=0;i<points.size();++i) center+=points[i]; center/=PxReal(points.size());
	const PxVec3 u=normal.cross(PxAbs(normal.x)<0.8f?PxVec3(1,0,0):PxVec3(0,1,0)).getNormalized();
	const PxVec3 v=normal.cross(u);
	std::sort(points.begin(),points.end(),[&](const PxVec3& a,const PxVec3& b)
	{ const PxVec3 da=a-center,db=b-center; return std::atan2(da.dot(v),da.dot(u))<std::atan2(db.dot(v),db.dot(u)); });
}
void clipPolyhedron(Polyhedron& poly,const PxPlane& plane,PxReal epsilon)
{
	PxArray<PxVec3> cap; Polyhedron result;bool outside=false,inside=false;
	for(PxU32 f=0;f<poly.size();++f) for(PxU32 p=0;p<poly[f].points.size();++p)
	{ const PxReal d=plane.distance(poly[f].points[p]);outside|=d>epsilon;inside|=d<-epsilon; }
	if(!outside) return; // A coincident source face must not duplicate a cell face.
	for(PxU32 f=0;f<poly.size();++f)
	{
		clipPolygon(poly[f].points,plane,&cap,epsilon);
		if(poly[f].points.size()>=3) result.pushBack(poly[f]);
	}
	if(inside && cap.size()>=3) { sortPolygon(cap,plane.n); Face face;face.points.swap(cap);result.pushBack(face); }
	poly.swap(result);
}
Polyhedron cellPolyhedron(const PxVec3& lo,const PxVec3& hi)
{
	Polyhedron poly;
	for(PxU32 axis=0;axis<3;++axis) for(PxU32 side=0;side<2;++side)
	{
		Face f; const PxU32 u=(axis+1)%3,v=(axis+2)%3;
		for(PxU32 c=0;c<4;++c)
		{
			PxVec3 p=lo;p[axis]=side?hi[axis]:lo[axis];
			p[u]=(c==1 || c==2)?hi[u]:lo[u];p[v]=c>=2?hi[v]:lo[v]; f.points.pushBack(p);
		}
		poly.pushBack(f);
	}
	return poly;
}
PxReal polyVolume(const Polyhedron& poly)
{
	PxVec3 origin(0);PxU32 count=0;
	for(PxU32 f=0;f<poly.size();++f) for(PxU32 p=0;p<poly[f].points.size();++p) { origin+=poly[f].points[p];++count; }
	if(!count) return 0;origin/=PxReal(count);double volume=0;
	for(PxU32 f=0;f<poly.size();++f) for(PxU32 p=1;p+1<poly[f].points.size();++p)
		volume+=PxAbs((poly[f].points[0]-origin).dot((poly[f].points[p]-origin).cross(poly[f].points[p+1]-origin)))/6.0;
	return PxReal(volume);
}
bool section(const PxArray<PxVec3>& polygon,PxU32 axis,PxReal& area,PxVec3& center,PxQuat& frame,PxReal& iy,PxReal& iz)
{
	const PxQuat base=axis==0?PxQuat(PxIdentity):(axis==1?PxQuat(PxHalfPi,PxVec3(0,0,1)):PxQuat(-PxHalfPi,PxVec3(0,1,0)));
	const PxVec3 u=base.rotate(PxVec3(0,1,0)),v=base.rotate(PxVec3(0,0,1));
	double a=0,cy=0,cz=0,my=0,mz=0,myz=0;
	for(PxU32 i=0;i<polygon.size();++i)
	{
		const double y=polygon[i].dot(u),z=polygon[i].dot(v),yy=polygon[(i+1)%polygon.size()].dot(u),zz=polygon[(i+1)%polygon.size()].dot(v);
		const double cross=y*zz-yy*z;a+=cross;cy+=(y+yy)*cross;cz+=(z+zz)*cross;
		my+=(z*z+z*zz+zz*zz)*cross;mz+=(y*y+y*yy+yy*yy)*cross;
		myz+=(2*y*z+y*zz+yy*z+2*yy*zz)*cross;
	}
	if(a==0) return false;const double sign=a>0?1:-1;
	area=PxReal(sign*a*0.5);cy/=3*a;cz/=3*a;
	const double iyy=sign*my/12-area*cz*cz,izz=sign*mz/12-area*cy*cy,iyz=-sign*myz/24+area*cy*cz;
	const PxReal angle=PxReal(0.5*std::atan2(2*iyz,iyy-izz));const double c=std::cos(angle),s=std::sin(angle);
	iy=PxReal(c*c*iyy+2*c*s*iyz+s*s*izz);iz=PxReal(s*s*iyy-2*c*s*iyz+c*c*izz);
	center=u*PxReal(cy)+v*PxReal(cz);center[axis]=polygon[0][axis];
	frame=base*PxQuat(angle,PxVec3(1,0,0));return area>0 && iy>0 && iz>0;
}
}

bool JointAsset::build(PxPhysics& physics,PxScene& scene,PxMaterial& material,const Settings& settings,
	const PxTransform& pose,const PxArray<PxVec3>& vertices,const PxArray<PxU32>& triangles,PxU32 identity)
{
	if(!identity || vertices.empty() || triangles.size()%3) return false;
	skinRest=vertices;skinTriangles=triangles;skinVertices.resize(vertices.size());skinBindings.resize(vertices.size());
	PxBounds3 bounds=PxBounds3::empty();PxVec3 centroid(0);
	for(PxU32 i=0;i<vertices.size();++i) { bounds.include(vertices[i]);centroid+=vertices[i]; } centroid/=PxReal(vertices.size());
	const PxVec3 cell=(bounds.maximum-bounds.minimum)/PxReal(settings.grid);
	const PxReal epsilon=1.e-6f*(bounds.maximum-bounds.minimum).magnitude();
	PxArray<PxPlane> planes;double signedVolume=0;
	for(PxU32 t=0;t<triangles.size();t+=3)
	{
		if(triangles[t]>=vertices.size() || triangles[t+1]>=vertices.size() || triangles[t+2]>=vertices.size()) return false;
		const PxVec3 a=vertices[triangles[t]],b=vertices[triangles[t+1]],c=vertices[triangles[t+2]];
		signedVolume+=double(a.dot(b.cross(c)))/6.0;
	}
	sourceVolume=PxReal(std::abs(signedVolume));if(sourceVolume<=0) return false;
	// The source sphere triangulation is slightly non-convex. Bake the public
	// PhysX convex envelope once; the original triangles remain the display skin.
	PxConvexMeshDesc envelopeDesc;envelopeDesc.points.count=vertices.size();envelopeDesc.points.stride=sizeof(PxVec3);envelopeDesc.points.data=vertices.begin();envelopeDesc.flags=PxConvexFlag::eCOMPUTE_CONVEX;
	PxConvexMesh* envelope=PxCreateConvexMesh(PxCookingParams(physics.getTolerancesScale()),envelopeDesc,physics.getPhysicsInsertionCallback());
	if(!envelope) return false;
	for(PxU32 f=0;f<envelope->getNbPolygons();++f)
	{
		PxHullPolygon polygon;envelope->getPolygonData(f,polygon);const PxVec3 normal(polygon.mPlane[0],polygon.mPlane[1],polygon.mPlane[2]);
		PxReal support=-PX_MAX_F32;for(PxU32 v=0;v<vertices.size();++v) support=PxMax(support,normal.dot(vertices[v]));
		// Cooking's polygon limit can simplify inward. Use the exact source
		// support for each baked axis so the initial display is inside its proxy.
		envelopeExpansion=PxMax(envelopeExpansion,support+polygon.mPlane[3]);planes.pushBack(PxPlane(normal,-support));
	}
	envelope->release();if(planes.size()<4) return false;
	for(PxU32 v=0;v<vertices.size();++v) for(PxU32 p=0;p<planes.size();++p) envelopeError=PxMax(envelopeError,planes[p].distance(vertices[v]));
	Polyhedron envelopePoly=cellPolyhedron(bounds.minimum,bounds.maximum);
	for(PxU32 p=0;p<planes.size();++p) clipPolyhedron(envelopePoly,planes[p],epsilon);proxyVolume=polyVolume(envelopePoly);
	lattice.grid=settings.grid;lattice.cell=PxMin(cell.x,PxMin(cell.y,cell.z));
	const PxU32 g=settings.grid;PxArray<PxU32> map(g*g*g,PX_MAX_U32);
	const PxReal fullVolume=cell.x*cell.y*cell.z;
	for(PxU32 z=0;z<g;++z) for(PxU32 y=0;y<g;++y) for(PxU32 x=0;x<g;++x)
	{
		const PxVec3 lo=bounds.minimum+cell.multiply(PxVec3(PxReal(x),PxReal(y),PxReal(z))),hi=lo+cell,origin=(lo+hi)*0.5f;
		bool rectangle=true;
		for(PxU32 p=0;p<planes.size() && rectangle;++p) for(PxU32 c=0;c<8 && rectangle;++c)
			rectangle=planes[p].distance(PxVec3(c&1?hi.x:lo.x,c&2?hi.y:lo.y,c&4?hi.z:lo.z))<=epsilon;
		PxRigidDynamic* body=NULL;
		if(rectangle) body=PxCreateDynamic(physics,pose*PxTransform(origin),PxBoxGeometry(cell*0.5f),material,settings.density);
		else
		{
			Polyhedron poly=cellPolyhedron(lo,hi);
			for(PxU32 p=0;p<planes.size() && !poly.empty();++p) clipPolyhedron(poly,planes[p],epsilon);
			const PxReal volume=polyVolume(poly);if(volume<=fullVolume*1.e-8f) { discardedVolume+=volume;continue; }
			PxArray<PxVec3> points;
			for(PxU32 f=0;f<poly.size();++f) for(PxU32 p=0;p<poly[f].points.size();++p) uniquePoint(points,poly[f].points[p]-origin,epsilon);
			PxConvexMeshDesc desc;desc.points.count=points.size();desc.points.stride=sizeof(PxVec3);desc.points.data=points.begin();desc.flags=PxConvexFlag::eCOMPUTE_CONVEX;
			PxConvexMesh* mesh=PxCreateConvexMesh(PxCookingParams(physics.getTolerancesScale()),desc,physics.getPhysicsInsertionCallback());
			if(!mesh) { std::fprintf(stderr,"[JOINT_ASSET_CELL] x=%u y=%u z=%u points=%u volume=%.9g\n",x,y,z,points.size(),volume);return false; }
			body=PxCreateDynamic(physics,pose*PxTransform(origin),PxConvexMeshGeometry(mesh),material,settings.density);mesh->release();
		}
		if(!body) return false;
		body->setLinearDamping(0.005f);body->setAngularDamping(0);
		PxShape* shape=NULL;body->getShapes(&shape,1);const PxFilterData filterData(identity,x+1,y+1,z+1);
		shape->setSimulationFilterData(filterData);shape->setQueryFilterData(filterData);
		shape->setContactOffset(0.02f*lattice.cell);shape->setRestOffset(0);scene.addActor(*body);
		map[lattice.index(x,y,z)]=lattice.nodes.size();lattice.nodes.pushBack(body);lattice.restLocal.pushBack(origin);lattice.totalMass+=body->getMass();
	}
	if(lattice.nodes.empty()) return false;
	PxArray<PxU32> components(lattice.nodes.size());for(PxU32 i=0;i<components.size();++i) components[i]=i;
	const PxReal shear=settings.young/(2*(1+settings.poisson));
	for(PxU32 z=0;z<g;++z) for(PxU32 y=0;y<g;++y) for(PxU32 x=0;x<g;++x) for(PxU32 axis=0;axis<3;++axis)
	{
		PxU32 n[3]={x,y,z};if(++n[axis]>=g) continue;
		const PxU32 a=map[lattice.index(x,y,z)],b=map[lattice.index(n[0],n[1],n[2])];if(a==PX_MAX_U32 || b==PX_MAX_U32) continue;
		const PxVec3 lo=bounds.minimum+cell.multiply(PxVec3(PxReal(x),PxReal(y),PxReal(z))),hi=lo+cell;
		PxArray<PxVec3> face;const PxU32 u=(axis+1)%3,v=(axis+2)%3;
		for(PxU32 c=0;c<4;++c) { PxVec3 p=lo;p[axis]=hi[axis];p[u]=(c==1 || c==2)?hi[u]:lo[u];p[v]=c>=2?hi[v]:lo[v];face.pushBack(p); }
		for(PxU32 p=0;p<planes.size() && !face.empty();++p) clipPolygon(face,planes[p],NULL,epsilon);
		PxReal area,iy,iz;PxVec3 center;PxQuat frame;
		if(face.size()<3 || !section(face,axis,area,center,frame,iy,iz)) continue;
		PxRigidDynamic* ba=lattice.nodes[a];PxRigidDynamic* bb=lattice.nodes[b];
		const PxVec3 materialA=lattice.restLocal[a]+ba->getCMassLocalPose().p,materialB=lattice.restLocal[b]+bb->getCMassLocalPose().p;
		const PxReal length=materialB[axis]-materialA[axis];if(!(length>0) || !PxIsFinite(length)) return false;
		FiniteJoint* joint=FiniteJoint::create(physics,*ba,PxTransform(center-lattice.restLocal[a],frame),*bb,PxTransform(center-lattice.restLocal[b],frame));if(!joint) return false;
		lattice.joints.pushBack(joint);
		PxU32 ra=a,rb=b;while(components[ra]!=ra) ra=components[ra];while(components[rb]!=rb) rb=components[rb];components[rb]=ra;
		const PxReal k[6]={settings.young*area/length,shear*area/length,shear*area/length,shear*(iy+iz)/length,settings.young*iy/length,settings.young*iz/length};
		for(PxU32 drive=0;drive<6;++drive)
		{
			PxReal mobility=1/ba->getMass()+1/bb->getMass();
			if(drive>=3)
			{
				mobility=0;PxVec3 basis(0);basis[drive-3]=1;const PxVec3 direction=pose.q.rotate(frame.rotate(basis));
				PxRigidDynamic* endpoints[2]={ba,bb};for(PxU32 e=0;e<2;++e) { const PxVec3 local=(endpoints[e]->getGlobalPose().q*endpoints[e]->getCMassLocalPose().q).rotateInv(direction);mobility+=local.dot(endpoints[e]->getMassSpaceInvInertiaTensor().multiply(local)); }
			}
			const PxReal damping=2*settings.dampingRatio*PxSqrt(k[drive]/mobility);
			joint->setDrive(PxD6Drive::Enum(drive),PxD6JointDrive(k[drive],damping,PX_MAX_F32,false));
		}
	}
	if(!lattice.buildMaterials(physics,settings,map)) return false;
	// Only display weights persist. Clipping and neighbor storage ends here.
	for(PxU32 v=0;v<vertices.size();++v)
	{
		PxU32 closest[8];PxReal distances[8];for(PxU32 k=0;k<8;++k) { closest[k]=0;distances[k]=PX_MAX_F32; }
		for(PxU32 n=0;n<lattice.nodes.size();++n)
		{
			const PxReal d=(vertices[v]-lattice.restLocal[n]).magnitudeSquared();
			for(PxU32 k=0;k<8;++k) if(d<distances[k]) { for(PxU32 j=7;j>k;--j) { closest[j]=closest[j-1];distances[j]=distances[j-1]; }closest[k]=n;distances[k]=d;break; }
		}
		PxReal sum=0;for(PxU32 k=0;k<8;++k) { skinBindings[v].nodes[k]=closest[k];skinBindings[v].weights[k]=distances[k]<PX_MAX_F32?1/PxMax(distances[k],epsilon*epsilon):0;sum+=skinBindings[v].weights[k]; }
		for(PxU32 k=0;k<8;++k) skinBindings[v].weights[k]/=sum;
	}
	poseView.resize(lattice.nodes.size());updateSkin();
	for(PxU32 v=0;v<vertices.size();++v) restSkinError=PxMax(restSkinError,(skinVertices[v]-pose.transform(vertices[v])).magnitude());
	// Read-only common-motion oracle: weights must reproduce a rigid transform,
	// including translation, independently of the physical scene's current pose.
	const PxTransform oracle(PxVec3(2,-3,1),PxQuat(0.7f,PxVec3(1,2,3).getNormalized()));
	for(PxU32 v=0;v<vertices.size();++v)
	{
		PxVec3 p(0);for(PxU32 k=0;k<8;++k) { const PxU32 n=skinBindings[v].nodes[k];p+=(oracle*PxTransform(lattice.restLocal[n])).transform(vertices[v]-lattice.restLocal[n])*skinBindings[v].weights[k]; }
		rigidSkinError=PxMax(rigidSkinError,(p-oracle.transform(vertices[v])).magnitude());
	}
	PxU32 root=0;while(components[root]!=root) root=components[root];
	for(PxU32 i=0;i<components.size();++i) { PxU32 r=i;while(components[r]!=r) r=components[r];if(r!=root) { std::fprintf(stderr,"[JOINT_ASSET_CONNECTIVITY] node=%u nodes=%u joints=%u\n",i,components.size(),lattice.joints.size());return false; } }
	return discardedVolume/proxyVolume<1.e-5f;
}
void JointAsset::updateSkin()
{
	if(skinVertices.empty()) return;
	for(PxU32 n=0;n<lattice.nodes.size();++n) poseView[n]=lattice.nodes[n]->getGlobalPose();
	for(PxU32 v=0;v<skinVertices.size();++v)
	{
		PxVec3 p(0);for(PxU32 k=0;k<8;++k) { const PxU32 n=skinBindings[v].nodes[k];p+=poseView[n].transform(skinRest[v]-lattice.restLocal[n])*skinBindings[v].weights[k]; }skinVertices[v]=p;
	}
}
PxU64 JointAsset::skinBytes() const
{
	return PxU64(skinRest.size()+skinVertices.size())*sizeof(PxVec3)+PxU64(skinTriangles.size())*sizeof(PxU32)+PxU64(skinBindings.size())*sizeof(SkinBinding)+PxU64(poseView.size())*sizeof(PxTransform);
}
void ShowcaseWorld::release()
{
	for(PxU32 i=0;i<bodies.size();++i) { bodies[i]->release();delete bodies[i]; }bodies.reset();
	PX_RELEASE(pedestal);PX_RELEASE(host.impactor);PX_RELEASE(rigidMaterial);host.release();
}
bool ShowcaseWorld::step()
{
	// PVD ends the frame before publishing new actor poses. Queue the skin of
	// the same fetched state before simulate; SDK visualization flushes it with
	// that frame's contacts. Never read actors while simulation is in flight.
	if(host.pvd.connected())
	{
		static const PxU32 colors[5]={0xff66b3e6,0xffb373e6,0xffe6994d,0xff4dcc99,0xff4d99cc};
		for(PxU32 i=0;i<bodies.size();++i)
		{
			bodies[i]->updateSkin();
			host.pvd.wireMesh(*host.scene,bodies[i]->skinVertices,bodies[i]->skinTriangles,colors[i%5]);
		}
	}
	host.events.contacts=host.events.crossContacts=host.events.crossMask=0;
	host.scene->simulate(1.0f/60.0f);return host.scene->fetchResults(true);
}
bool ShowcaseWorld::init(const Settings& settings,const char* name)
{
	const bool pair=std::strcmp(name,"multi-collision")==0;
	if(!host.initScene(settings,name,pair?PxVec3(0):PxVec3(0,-9.81f,0),!pair,pair?0.5f:0.2f)) return false;
	if(pair)
	{
		for(PxU32 i=0;i<2;++i)
		{
			JointAsset* asset=new JointAsset;bodies.pushBack(asset);
			if(!asset->lattice.build(*host.physics,*host.scene,*host.material,settings,PxTransform(PxVec3(i?1.1f:-1.1f,3,0)),i+1)) return false;
			for(PxU32 n=0;n<asset->lattice.nodes.size();++n) asset->lattice.nodes[n]->setLinearVelocity(PxVec3(i?-2.f:2.f,0,0));
		}
		return true;
	}
	rigidMaterial=host.physics->createMaterial(0.5f,0.5f,0);if(!rigidMaterial) return false;
	PxShape* groundShape=NULL;host.ground->getShapes(&groundShape,1);groundShape->setMaterials(&rigidMaterial,1);
	pedestal=host.physics->createRigidStatic(PxTransform(PxVec3(7.6f,0.55f,0)));
	if(!pedestal || !PxRigidActorExt::createExclusiveShape(*pedestal,PxBoxGeometry(0.7f,0.55f,2.2f),*rigidMaterial)) return false;host.scene->addActor(*pedestal);
	const PxQuat primary(-0.62f,PxVec3(0.18f,0.22f,0.96f).getNormalized()),cone(0.48f,PxVec3(0.35f,0.82f,0.45f).getNormalized()),jaw(0.36f,PxVec3(0,1,0));
	const PxVec3 rigidCenter(6.4f,25,0.3f),jawAxis=jaw.rotate(PxVec3(0,1,0));
	const PxTransform poses[5]={PxTransform(PxVec3(-3.2f,8,0),primary),PxTransform(PxVec3(-0.2f,2,0)),PxTransform(PxVec3(3.2f,11,1.2f),cone),PxTransform(rigidCenter+jawAxis*1.1f,jaw),PxTransform(rigidCenter-jawAxis*1.1f,jaw)};
	const PxVec3 scales[5]={PxVec3(1.8f,0.65f,0.9f),PxVec3(1),PxVec3(1),PxVec3(1.7f,0.7f,0.9f),PxVec3(1.7f,0.7f,0.9f)};
	const PxReal density[5]={160,130,160,160,160};const PxU32 resolutions[5]={5,6,8,5,5};
	for(PxU32 i=0;i<5;++i)
	{
		PxArray<PxVec3> vertices;PxArray<PxU32> triangles;
		if(i==1) meshgenerator::createSphere(vertices,triangles,PxVec3(0),1.8f,0.8f);
		else if(i==2) VolumeAvbdSnippet::createLayeredConeSurface(vertices,triangles,0.8f,3,32,3);
		else VolumeAvbdSnippet::createSubdividedCubeSurface(vertices,triangles,2,4);
		for(PxU32 v=0;v<vertices.size();++v) vertices[v]=vertices[v].multiply(scales[i]);
		Settings config=settings;config.grid=PxMax(3u,(resolutions[i]*settings.grid+2)/5);config.density=density[i];
		JointAsset* asset=new JointAsset;bodies.pushBack(asset);
		if(!asset->build(*host.physics,*host.scene,*host.material,config,poses[i],vertices,triangles,i+1)) { std::fprintf(stderr,"[JOINT_ASSET_ERROR] body=%u\n",i);return false; }
		const PxVec3 velocity=i==0?PxVec3(2.5f,0,0):i==2?PxVec3(0.045f,0,0.02f):PxVec3(0);
		for(PxU32 n=0;n<asset->lattice.nodes.size();++n) asset->lattice.nodes[n]->setLinearVelocity(velocity);
	}
	host.impactor=PxCreateDynamic(*host.physics,PxTransform(rigidCenter,jaw),PxBoxGeometry(0.5f,0.35f,0.7f),*rigidMaterial,1);
	if(!host.impactor || !PxRigidBodyExt::setMassAndUpdateInertia(*host.impactor,10)) return false;
	host.impactor->setLinearDamping(0.015f);host.impactor->setSolverIterationCounts(16,1);host.scene->addActor(*host.impactor);return true;
}
}
