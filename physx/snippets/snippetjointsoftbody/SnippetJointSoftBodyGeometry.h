// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef SNIPPET_JOINT_SOFT_BODY_GEOMETRY_H
#define SNIPPET_JOINT_SOFT_BODY_GEOMETRY_H
#include "PxPhysicsAPI.h"
#include <vector>
#include <cmath>
namespace JointSoftBodySnippet
{
// Read-only headless oracle. Face normals and all edge cross products cover
// convex SAT, including containment. It never corrects a physical pose or
// feeds geometry into Backend. No oracle geometry is stored in a scene asset.
struct GeometryVector
{
	double x,y,z;
	GeometryVector() : x(0),y(0),z(0) {}
	explicit GeometryVector(const physx::PxVec3& p) : x(p.x),y(p.y),z(p.z) {}
	GeometryVector(double a,double b,double c) : x(a),y(b),z(c) {}
	GeometryVector operator-(const GeometryVector& b) const { return GeometryVector(x-b.x,y-b.y,z-b.z); }
	double dot(const GeometryVector& b) const { return x*b.x+y*b.y+z*b.z; }
	GeometryVector cross(const GeometryVector& b) const { return GeometryVector(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x); }
};
struct ObservedPolytope
{
	std::vector<GeometryVector> points,normals,edges;
	bool build(const physx::PxGeometry& geometry,const physx::PxTransform& pose)
	{
		using namespace physx;
		if(geometry.getType()==PxGeometryType::eBOX)
		{
			const PxVec3 half=static_cast<const PxBoxGeometry&>(geometry).halfExtents;
			for(PxU32 i=0;i<8;++i) points.push_back(GeometryVector(pose.transform(PxVec3(i&1?half.x:-half.x,i&2?half.y:-half.y,i&4?half.z:-half.z))));
			for(PxU32 i=0;i<3;++i) { PxVec3 n(0);n[i]=1;normals.push_back(GeometryVector(pose.q.rotate(n)));edges.push_back(normals.back()); }
			return true;
		}
		if(geometry.getType()!=PxGeometryType::eCONVEXMESH) return false;
		const PxConvexMeshGeometry& convex=static_cast<const PxConvexMeshGeometry&>(geometry);
		const PxMat33 scale=convex.scale.toMat33(),normalScale=scale.getInverse().getTranspose();
		const PxConvexMesh& mesh=*convex.convexMesh;
		for(PxU32 i=0;i<mesh.getNbVertices();++i) points.push_back(GeometryVector(pose.transform(scale*mesh.getVertices()[i])));
		std::vector<PxU32> usedEdges;
		for(PxU32 f=0;f<mesh.getNbPolygons();++f)
		{
			PxHullPolygon polygon;mesh.getPolygonData(f,polygon);
			normals.push_back(GeometryVector(pose.q.rotate(normalScale*PxVec3(polygon.mPlane[0],polygon.mPlane[1],polygon.mPlane[2]))));
			const PxU8* ids=mesh.getIndexBuffer()+polygon.mIndexBase;
			for(PxU32 k=0;k<polygon.mNbVerts;++k)
			{
				const PxU32 a=ids[k],b=ids[(k+1)%polygon.mNbVerts],key=PxMin(a,b)*256+PxMax(a,b);
				bool used=false;for(size_t i=0;i<usedEdges.size();++i) used|=usedEdges[i]==key;
				if(!used) { usedEdges.push_back(key);edges.push_back(points[b]-points[a]); }
			}
		}
		return !points.empty();
	}
};
inline double polytopeDepth(const ObservedPolytope& a,const ObservedPolytope& b)
{
	double depth=1.e300;
	auto axis=[&](const GeometryVector& n)
	{
		const double length=std::sqrt(n.dot(n));if(length==0) return true;
		double amin=1.e300,amax=-1.e300,bmin=1.e300,bmax=-1.e300;
		for(size_t i=0;i<a.points.size();++i) { const double d=a.points[i].dot(n);amin=std::fmin(amin,d);amax=std::fmax(amax,d); }
		for(size_t i=0;i<b.points.size();++i) { const double d=b.points[i].dot(n);bmin=std::fmin(bmin,d);bmax=std::fmax(bmax,d); }
		const double overlap=std::fmin(amax-bmin,bmax-amin);if(overlap<=0) return false;
		depth=std::fmin(depth,overlap/length);return true;
	};
	for(size_t i=0;i<a.normals.size();++i) if(!axis(a.normals[i])) return 0;
	for(size_t i=0;i<b.normals.size();++i) if(!axis(b.normals[i])) return 0;
	for(size_t i=0;i<a.edges.size();++i) for(size_t j=0;j<b.edges.size();++j) if(!axis(a.edges[i].cross(b.edges[j]))) return 0;
	return depth;
}
inline physx::PxReal observedPenetration(const physx::PxGeometry& a,const physx::PxTransform& pa,const physx::PxGeometry& b,const physx::PxTransform& pb)
{
	using namespace physx;ObservedPolytope first;if(!first.build(a,pa)) return PX_MAX_F32;
	if(b.getType()==PxGeometryType::ePLANE)
	{
		const GeometryVector normal(pb.q.rotate(PxVec3(1,0,0))),origin(pb.p);double minimum=1.e300;
		for(size_t i=0;i<first.points.size();++i) minimum=std::fmin(minimum,(first.points[i]-origin).dot(normal));
		return PxReal(std::fmax(0,-minimum));
	}
	ObservedPolytope second;if(!second.build(b,pb)) return PX_MAX_F32;
	return PxReal(polytopeDepth(first,second));
}
inline bool geometryOracleValidated(physx::PxPhysics& physics)
{
	using namespace physx;
	// Known analytic containment, separation and plane depths. The containing
	// box test rejects the incorrect interval-intersection depth of 2.
	if(PxAbs(observedPenetration(PxBoxGeometry(1,1,1),PxTransform(PxIdentity),PxBoxGeometry(2,2,2),PxTransform(PxIdentity))-3)>1.e-6f) return false;
	if(observedPenetration(PxBoxGeometry(1,1,1),PxTransform(PxIdentity),PxBoxGeometry(1,1,1),PxTransform(PxVec3(3,0,0)))!=0) return false;
	if(PxAbs(observedPenetration(PxBoxGeometry(1,1,1),PxTransform(PxVec3(0.5f,0,0)),PxPlaneGeometry(),PxTransform(PxIdentity))-0.5f)>1.e-6f) return false;
	// Compare independent SAT with SDK MTD on ordinary rotated box pairs.
	for(PxU32 i=0;i<24;++i)
	{
		const PxBoxGeometry a(0.7f,0.4f,0.6f),b(0.6f,0.5f,0.3f);
		const PxTransform pa(PxVec3(0),PxQuat(0.023f*PxReal(i),PxVec3(1,2,3).getNormalized())),pb(PxVec3(0.5f+0.037f*PxReal(i),0.1f,0.1f),PxQuat(-0.031f*PxReal(i),PxVec3(2,1,1).getNormalized()));
		PxVec3 direction;PxReal depth=0;if(!PxGeometryQuery::computePenetration(direction,depth,a,pa,b,pb)) depth=0;
		if(PxAbs(observedPenetration(a,pa,b,pb)-depth)>2.e-5f) return false;
	}
	PxVec3 vertices[8];for(PxU32 i=0;i<8;++i) vertices[i]=PxVec3(i&1?0.7f:-0.7f,i&2?0.4f:-0.4f,i&4?0.6f:-0.6f);
	PxConvexMeshDesc desc;desc.points.count=8;desc.points.stride=sizeof(PxVec3);desc.points.data=vertices;desc.flags=PxConvexFlag::eCOMPUTE_CONVEX;
	PxConvexMesh* mesh=PxCreateConvexMesh(PxCookingParams(physics.getTolerancesScale()),desc,physics.getPhysicsInsertionCallback());if(!mesh) return false;
	bool valid=true;
	for(PxU32 i=0;i<24;++i)
	{
		const PxTransform pa(PxVec3(0),PxQuat(0.023f*PxReal(i),PxVec3(1,2,3).getNormalized())),pb(PxVec3(0.5f+0.037f*PxReal(i),0.1f,0.1f),PxQuat(-0.031f*PxReal(i),PxVec3(2,1,1).getNormalized()));
		const PxBoxGeometry box(0.6f,0.5f,0.3f);PxVec3 direction;PxReal depth=0;
		if(!PxGeometryQuery::computePenetration(direction,depth,PxBoxGeometry(0.7f,0.4f,0.6f),pa,box,pb)) depth=0;
		valid&=PxAbs(observedPenetration(PxConvexMeshGeometry(mesh),pa,box,pb)-depth)<=2.e-5f;
	}
	valid&=PxAbs(observedPenetration(PxConvexMeshGeometry(mesh),PxTransform(PxIdentity),PxBoxGeometry(2,2,2),PxTransform(PxIdentity))-2.4f)<=2.e-5f;
	mesh->release();if(!valid) return false;
	return true;
}
}
#endif
