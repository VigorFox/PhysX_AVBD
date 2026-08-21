// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#include "SnippetDeformableVolumeAVBDFixtures.h"

using namespace physx;

namespace SnippetDeformableVolumeAVBDFixtures
{

bool isComponentDenseNoContactCase(const std::string& caseName)
{
	return caseName == "volume-performance-dense-no-contact";
}

bool isComponentManySmallNoContactCase(const std::string& caseName)
{
	return caseName == "volume-performance-many-small-no-contact";
}

void rotateVerticesAroundZ(
	PxArray<PxVec3>& vertices,
	const PxVec3& center,
	PxReal angle)
{
	const PxReal cs = PxCos(angle);
	const PxReal sn = PxSin(angle);
	for(PxU32 i = 0; i < vertices.size(); ++i)
	{
		const PxVec3 relative = vertices[i] - center;
		vertices[i].x = center.x + relative.x * cs - relative.y * sn;
		vertices[i].y = center.y + relative.x * sn + relative.y * cs;
	}
}

void scaleVerticesAboutCenter(
	PxArray<PxVec3>& vertices,
	const PxVec3& center,
	const PxVec3& scale)
{
	for(PxU32 i = 0; i < vertices.size(); ++i)
	{
		const PxVec3 relative = vertices[i] - center;
		vertices[i] = center + PxVec3(
			relative.x * scale.x,
			relative.y * scale.y,
			relative.z * scale.z);
	}
}

static void appendSubdividedQuad(
	PxArray<PxVec3>& vertices,
	PxArray<PxU32>& triangles,
	const PxVec3& origin,
	const PxVec3& axisU,
	const PxVec3& axisV,
	PxU32 subdivisions)
{
	PX_ASSERT(subdivisions > 0);
	const PxU32 base = vertices.size();
	for(PxU32 v = 0; v <= subdivisions; ++v)
	{
		for(PxU32 u = 0; u <= subdivisions; ++u)
		{
			vertices.pushBack(origin +
				axisU * (PxReal(u) / PxReal(subdivisions)) +
				axisV * (PxReal(v) / PxReal(subdivisions)));
		}
	}
	const PxU32 row = subdivisions + 1;
	for(PxU32 v = 0; v < subdivisions; ++v)
	{
		for(PxU32 u = 0; u < subdivisions; ++u)
		{
			const PxU32 i00 = base + v * row + u;
			const PxU32 i10 = i00 + 1;
			const PxU32 i01 = i00 + row;
			const PxU32 i11 = i01 + 1;
			triangles.pushBack(i00);
			triangles.pushBack(i10);
			triangles.pushBack(i11);
			triangles.pushBack(i00);
			triangles.pushBack(i11);
			triangles.pushBack(i01);
		}
	}
}

void createSubdividedCubeSurface(
	PxArray<PxVec3>& vertices,
	PxArray<PxU32>& triangles,
	const PxVec3& center,
	PxReal sideLength,
	PxU32 subdivisions)
{
	vertices.clear();
	triangles.clear();
	const PxReal halfSide = 0.5f * sideLength;
	// axisU x axisV points outwards on every face. Duplicate seam vertices
	// are intentional cooking input; eWELD_VERTICES merges them into one
	// manifold collision boundary.
	appendSubdividedQuad(vertices, triangles,
		center + PxVec3(halfSide, -halfSide, -halfSide),
		PxVec3(0, sideLength, 0), PxVec3(0, 0, sideLength), subdivisions);
	appendSubdividedQuad(vertices, triangles,
		center + PxVec3(-halfSide, -halfSide, -halfSide),
		PxVec3(0, 0, sideLength), PxVec3(0, sideLength, 0), subdivisions);
	appendSubdividedQuad(vertices, triangles,
		center + PxVec3(-halfSide, halfSide, -halfSide),
		PxVec3(0, 0, sideLength), PxVec3(sideLength, 0, 0), subdivisions);
	appendSubdividedQuad(vertices, triangles,
		center + PxVec3(-halfSide, -halfSide, -halfSide),
		PxVec3(sideLength, 0, 0), PxVec3(0, 0, sideLength), subdivisions);
	appendSubdividedQuad(vertices, triangles,
		center + PxVec3(-halfSide, -halfSide, halfSide),
		PxVec3(sideLength, 0, 0), PxVec3(0, sideLength, 0), subdivisions);
	appendSubdividedQuad(vertices, triangles,
		center + PxVec3(-halfSide, -halfSide, -halfSide),
		PxVec3(0, sideLength, 0), PxVec3(sideLength, 0, 0), subdivisions);
}

void createLayeredConeSurface(
	PxArray<PxVec3>& vertices,
	PxArray<PxU32>& triangles,
	const PxVec3& center,
	PxReal radius,
	PxReal height,
	PxU32 ringSegments,
	PxU32 heightSegments)
{
	vertices.clear();
	triangles.clear();
	PX_ASSERT(ringSegments >= 3 && heightSegments >= 2);
	for(PxU32 layer = 0; layer < heightSegments; ++layer)
	{
		const PxReal t = PxReal(layer) / PxReal(heightSegments);
		const PxReal layerRadius = radius * (1.0f - t);
		for(PxU32 segment = 0; segment < ringSegments; ++segment)
		{
			const PxReal angle = 2.0f * PxPi * PxReal(segment) /
				PxReal(ringSegments);
			vertices.pushBack(center + PxVec3(
				layerRadius * PxSin(angle), height * t,
				layerRadius * PxCos(angle)));
		}
	}
	const PxU32 apex = vertices.size();
	vertices.pushBack(center + PxVec3(0, height, 0));
	for(PxU32 layer = 0; layer + 1 < heightSegments; ++layer)
	{
		const PxU32 lower = layer * ringSegments;
		const PxU32 upper = (layer + 1) * ringSegments;
		for(PxU32 segment = 0; segment < ringSegments; ++segment)
		{
			const PxU32 next = (segment + 1) % ringSegments;
			triangles.pushBack(upper + segment);
			triangles.pushBack(lower + next);
			triangles.pushBack(lower + segment);
			triangles.pushBack(upper + segment);
			triangles.pushBack(upper + next);
			triangles.pushBack(lower + next);
		}
	}
	const PxU32 topRing = (heightSegments - 1) * ringSegments;
	for(PxU32 segment = 0; segment < ringSegments; ++segment)
	{
		triangles.pushBack(apex);
		triangles.pushBack(topRing + (segment + 1) % ringSegments);
		triangles.pushBack(topRing + segment);
	}
	const PxU32 baseCenter = vertices.size();
	vertices.pushBack(center);
	for(PxU32 segment = 0; segment < ringSegments; ++segment)
	{
		triangles.pushBack(baseCenter);
		triangles.pushBack(segment);
		triangles.pushBack((segment + 1) % ringSegments);
	}
}

bool isSceneCpuVolumeSpeculativeCcdCase(
	const std::string& caseName)
{
	return caseName == "scene-volume-speculative-ccd" ||
		caseName == "scene-volume-plane-speculative-ccd" ||
		caseName == "scene-volume-sphere-speculative-ccd" ||
		caseName == "scene-volume-capsule-speculative-ccd" ||
		caseName == "scene-volume-convex-speculative-ccd" ||
		caseName ==
			"scene-volume-moving-kinematic-sphere-speculative-ccd" ||
		caseName ==
			"scene-volume-moving-kinematic-capsule-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-capsule-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-convex-speculative-ccd" ||
		caseName ==
			"scene-volume-moving-kinematic-convex-speculative-ccd" ||
		caseName ==
			"scene-volume-dynamic-sphere-relative-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-capsule-relative-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-capsule-relative-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-convex-relative-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-convex-relative-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-kinematic-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-kinematic-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-heightfield-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-heightfield-reverse-swept-ccd";
}

bool isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
	const std::string& caseName)
{
	return
		caseName ==
			"scene-volume-kinematic-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-kinematic-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-kinematic-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-heightfield-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-triangle-mesh-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-heightfield-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-triangle-mesh-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-heightfield-reverse-swept-ccd";
}

bool isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
	const std::string& caseName)
{
	return isSceneCpuVolumeTriangleSurfaceSweptCcdCase(caseName) &&
		caseName.find("reverse-swept") != std::string::npos;
}

bool isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
	const std::string& caseName)
{
	return isSceneCpuVolumeTriangleSurfaceSweptCcdCase(caseName) &&
		caseName.find("rotating-kinematic") != std::string::npos;
}

bool isSceneCpuVolumeHeightFieldSweptCcdCase(
	const std::string& caseName)
{
	return isSceneCpuVolumeTriangleSurfaceSweptCcdCase(caseName) &&
		caseName.find("heightfield") != std::string::npos;
}

bool isSceneCpuVolumeRigidTriangleSteadyContactCase(
	const std::string& caseName)
{
	return caseName == "scene-volume-rigid-triangle-steady-contact";
}

bool isSceneCpuVolumeTaskGraphRigidTriangleSurfaceLargeCase(
	const std::string& caseName)
{
	return caseName ==
		"scene-volume-taskgraph-rigid-triangle-surface-large";
}

bool isSceneCpuVolumeTaskGraphRigidTriangleSurfaceFeatureOverlapCase(
	const std::string& caseName)
{
	return caseName ==
		"scene-volume-taskgraph-rigid-triangle-surface-feature-overlap";
}

bool isSceneCpuVolumeTaskGraphRigidTriangleSurfaceThresholdCase(
	const std::string& caseName)
{
	return caseName ==
		"scene-volume-taskgraph-rigid-triangle-surface-threshold";
}

bool isSceneCpuVolumeTaskGraphWriteBackCase(
	const std::string& caseName)
{
	return caseName == "scene-volume-taskgraph-writeback" ||
		caseName == "scene-volume-taskgraph-writeback-four-way" ||
		isSceneCpuVolumeTaskGraphPipelineCase(caseName) ||
		caseName ==
			"scene-volume-taskgraph-writeback-heterogeneous";
}

bool isSceneCpuVolumeTaskGraphWriteBackFourWayCase(
	const std::string& caseName)
{
	return caseName == "scene-volume-taskgraph-writeback-four-way" ||
		isSceneCpuVolumeTaskGraphPipelineCase(caseName) ||
		caseName ==
			"scene-volume-taskgraph-writeback-heterogeneous";
}

bool isSceneCpuVolumeTaskGraphPipelineCase(
	const std::string& caseName)
{
	return caseName == "scene-volume-taskgraph-pipeline";
}

bool isSceneCpuVolumeTaskGraphPureSoftCase(
	const std::string& caseName)
{
	return caseName == "scene-volume-taskgraph-pure-soft" ||
		caseName == "scene-volume-taskgraph-pure-soft-corotational";
}

bool isSceneCpuVolumeTaskGraphDirectSimulationDomainCase(
	const std::string& caseName)
{
	return isSceneCpuVolumeTaskGraphPureSoftCase(caseName) ||
		caseName == "scene-volume-taskgraph-world-plane" ||
		caseName == "scene-volume-taskgraph-rigid-box-sdf" ||
		caseName == "scene-volume-taskgraph-rigid-sphere-sdf" ||
		caseName == "scene-volume-taskgraph-rigid-capsule-sdf" ||
		caseName == "scene-volume-taskgraph-rigid-convex-sdf" ||
		caseName == "scene-volume-taskgraph-rigid-triangle-surface" ||
		isSceneCpuVolumeTaskGraphRigidTriangleSurfaceLargeCase(caseName) ||
		isSceneCpuVolumeTaskGraphRigidTriangleSurfaceFeatureOverlapCase(
			caseName) ||
		isSceneCpuVolumeTaskGraphRigidTriangleSurfaceThresholdCase(caseName) ||
		isSceneCpuVolumeTaskGraphWriteBackCase(caseName);
}

bool isSceneCpuVolumeSphereReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-deforming-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-deforming-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-deforming-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-convex-reverse-swept-ccd";
}

bool isSceneCpuVolumeCapsuleReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-deforming-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd";
}

bool isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd";
}

bool isSceneCpuVolumeConvexReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-deforming-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-static-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd";
}

bool isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd";
}

bool isSceneCpuVolumeDeformingReverseSweptCcdCase(
	const std::string& caseName)
{
	return caseName ==
			"scene-volume-deforming-sphere-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-deforming-capsule-reverse-swept-ccd" ||
		caseName ==
			"scene-volume-deforming-convex-reverse-swept-ccd";
}

bool isSceneCpuVolumeKinematicRigidCase(
	const std::string& caseName)
{
	return caseName == "scene-volume-kinematic-box" ||
		caseName == "scene-volume-kinematic-sphere" ||
		caseName == "scene-volume-kinematic-capsule" ||
		caseName == "scene-volume-kinematic-convex" ||
		caseName == "scene-volume-kinematic-triangle-mesh" ||
		caseName == "scene-volume-kinematic-heightfield";
}

bool isSceneCpuVolumeCase(const std::string& caseName)
{
	static const char* const sceneCases[] = {
#define AVBD_VOLUME_CASE(kind, name, ...) AVBD_VOLUME_CASE_##kind(name)
#define AVBD_VOLUME_CASE_COMPONENT(name)
#define AVBD_VOLUME_CASE_SCENE(name) name,
#define AVBD_VOLUME_CASE_CORE_SCENE(name) name,
#define AVBD_VOLUME_CASE_INTERNAL_SCENE(name) name,
#define AVBD_VOLUME_CASE_META(name)
#include "SnippetDeformableVolumeAVBDCases.inc"
#undef AVBD_VOLUME_CASE_META
#undef AVBD_VOLUME_CASE_INTERNAL_SCENE
#undef AVBD_VOLUME_CASE_CORE_SCENE
#undef AVBD_VOLUME_CASE_SCENE
#undef AVBD_VOLUME_CASE_COMPONENT
#undef AVBD_VOLUME_CASE
	};
	for(PxU32 i = 0; i < sizeof(sceneCases) / sizeof(sceneCases[0]); ++i)
	{
		if(caseName == sceneCases[i])
			return true;
	}
	return false;
}

HeadlessGateClassification classifyHeadlessGate(
	const std::string& caseName)
{
	const bool sceneLifecycle =
		caseName == "scene-volume-lifecycle" ||
		caseName == "scene-volume-corotational";
	const bool sceneVisualShowcase =
		caseName == "scene-volume-visual-showcase";
	const bool sceneOgcSandwich =
		caseName == "scene-volume-ogc-sandwich";
	const bool sceneSphereLongRoll =
		caseName == "scene-volume-sphere-long-roll";
	const bool sceneSphereSoftSoftGlancing =
		caseName ==
		"scene-volume-sphere-soft-soft-glancing";
	const bool sceneSoftSoftTorque =
		caseName == "scene-volume-soft-soft-torque";
	const bool sceneTaskGraphPureSoft =
		isSceneCpuVolumeTaskGraphPureSoftCase(caseName);
	const bool sceneTaskGraphRigidTriangleSurfaceFeatureOverlap =
		isSceneCpuVolumeTaskGraphRigidTriangleSurfaceFeatureOverlapCase(
			caseName);
	const bool sceneTaskGraphWriteBack =
		isSceneCpuVolumeTaskGraphWriteBackCase(caseName);
	const bool sceneTaskGraphPipeline =
		isSceneCpuVolumeTaskGraphPipelineCase(caseName);
	const bool sceneGround =
		caseName == "scene-volume-ground";
	const bool sceneGroundEmbeddedTetProbe =
		caseName ==
			"scene-volume-ground-embedded-tet-probe";
	const bool sceneMaxDepenetrationVelocity =
		caseName ==
			"scene-volume-max-depenetration-velocity";
	const bool sceneRigidTriangleSteadyContact =
		isSceneCpuVolumeRigidTriangleSteadyContactCase(
			caseName);
	const bool sceneSpeculativeCcd =
		isSceneCpuVolumeSpeculativeCcdCase(
			caseName) ||
		isSceneCpuVolumeSphereReverseSweptCcdCase(
			caseName);
	const bool sceneSphereReverseFeature =
		caseName ==
			"scene-volume-sphere-reverse-feature";
	const bool sceneCapsuleReverseFeature =
		caseName ==
			"scene-volume-capsule-reverse-feature";
	const bool sceneConvexReverseFeature =
		caseName ==
			"scene-volume-convex-reverse-feature";
	const bool sceneTriangleMeshReverseFeature =
		caseName ==
			"scene-volume-triangle-mesh-reverse-feature";
	const bool sceneHeightFieldReverseFeature =
		caseName ==
			"scene-volume-heightfield-reverse-feature";
	const bool sceneStaticChurn =
		caseName == "scene-volume-static-churn";
	const bool sceneStaticBox =
		caseName == "scene-volume-static-box" ||
		sceneStaticChurn;
	const bool sceneStatic =
		sceneGround || sceneGroundEmbeddedTetProbe || sceneStaticBox ||
		sceneMaxDepenetrationVelocity ||
		sceneRigidTriangleSteadyContact ||
		sceneSpeculativeCcd ||
		sceneSphereReverseFeature ||
		sceneCapsuleReverseFeature ||
		sceneConvexReverseFeature ||
		sceneTriangleMeshReverseFeature ||
		sceneHeightFieldReverseFeature;
	const bool sceneDynamicChurn =
		caseName == "scene-volume-dynamic-churn";
	const bool sceneMultiDynamic =
		caseName ==
			"scene-volume-multi-dynamic-box";
	const bool sceneMultiSoft =
		caseName ==
			"scene-volume-multi-soft-islands";
	const bool sceneSoftSleepWake =
		caseName ==
			"scene-volume-sleep-wake";
	const bool sceneSoftRigidWake =
		caseName ==
			"scene-volume-rigid-wake";
	const bool sceneMixedSleepIslands =
		caseName ==
			"scene-volume-mixed-sleep-islands";
	const bool sceneSoftChurn =
		caseName ==
			"scene-volume-soft-churn";
	const bool sceneBufferMutation =
		caseName ==
			"scene-volume-buffer-mutation";
	const bool sceneWorldPin =
		caseName ==
			"scene-volume-world-pin" ||
		caseName ==
			"scene-volume-world-element-attachment";
	const bool sceneRigidAttachment =
		caseName ==
			"scene-volume-rigid-attachment" ||
		caseName ==
			"scene-volume-rigid-element-attachment";
	const bool sceneStaticAttachment =
		caseName ==
			"scene-volume-static-attachment" ||
		caseName ==
			"scene-volume-static-element-attachment";
	const bool sceneKinematicAttachment =
		caseName ==
			"scene-volume-kinematic-attachment" ||
		caseName ==
			"scene-volume-kinematic-element-attachment";
	const bool sceneArticulationAttachment =
		caseName ==
			"scene-volume-articulation-attachment" ||
		caseName ==
			"scene-volume-articulation-element-attachment";
	const bool scenePartialElementFilter =
		caseName ==
			"scene-volume-partial-element-filter";
	const bool sceneElementFilter =
		caseName ==
			"scene-volume-element-filter" ||
		scenePartialElementFilter;
	const bool sceneKinematicBox =
		isSceneCpuVolumeKinematicRigidCase(
			caseName);
	const bool sceneMultiSceneIsolation =
		caseName ==
			"scene-volume-multi-scene-isolation";
	const bool sceneSoftSoftWake =
		caseName ==
			"scene-volume-soft-soft-wake";
	const bool sceneSoftPairAttachment =
		caseName ==
			"scene-volume-volume-attachment";
	const bool sceneMotionControls =
		caseName ==
			"scene-volume-motion-controls";
	const bool sceneSkinning =
		caseName ==
			"scene-volume-skinning";
	const bool sceneVolumeKinematicTarget =
		caseName ==
			"scene-volume-full-kinematic-target" ||
		caseName ==
			"scene-volume-partial-kinematic-target";
	const bool sceneDynamic =
		sceneOgcSandwich ||
		caseName == "scene-volume-dynamic-box" ||
		caseName ==
			"scene-volume-true-boundary-dynamic-box" ||
		caseName == "scene-volume-dynamic-sphere" ||
		caseName == "scene-volume-dynamic-capsule" ||
		caseName == "scene-volume-dynamic-convex" ||
		sceneDynamicChurn || sceneMultiDynamic ||
		sceneMultiSoft || sceneRigidAttachment ||
		sceneKinematicAttachment;
	const bool sceneIntegrated =
		sceneVisualShowcase || sceneOgcSandwich || sceneSphereLongRoll ||
		sceneSphereSoftSoftGlancing || sceneSoftSoftTorque ||
		sceneLifecycle ||
		sceneTaskGraphPureSoft ||
		sceneTaskGraphRigidTriangleSurfaceFeatureOverlap ||
		sceneTaskGraphWriteBack ||
		sceneStatic || sceneDynamic ||
		sceneSoftSleepWake || sceneSoftRigidWake ||
		sceneMixedSleepIslands || sceneSoftChurn ||
		sceneBufferMutation || sceneWorldPin ||
		sceneRigidAttachment || sceneStaticAttachment ||
		sceneKinematicAttachment ||
		sceneArticulationAttachment ||
		sceneElementFilter ||
		sceneKinematicBox ||
		sceneMultiSceneIsolation ||
		sceneSoftSoftWake ||
		sceneSoftPairAttachment ||
		sceneVolumeKinematicTarget ||
		sceneSkinning ||
		sceneMotionControls ||
		sceneMaxDepenetrationVelocity ||
		sceneSpeculativeCcd;
	const char* validation =
		sceneTaskGraphPureSoft ? "SCENE_TASKGRAPH_PURE_SOFT_GATED" :
		(sceneSkinning ? "SCENE_CPU_SKINNING_GATED" :
		(sceneMotionControls ?
			"SCENE_DEFORMABLE_MOTION_CONTROLS_GATED" :
		sceneVolumeKinematicTarget ?
			"SCENE_VOLUME_KINEMATIC_TARGET_GATED" :
		(sceneArticulationAttachment ?
			"SCENE_ARTICULATION_ATTACHMENT_GATED" :
		(sceneStaticAttachment ?
			"SCENE_STATIC_ATTACHMENT_GATED" :
		(sceneKinematicAttachment ?
			"SCENE_KINEMATIC_ATTACHMENT_GATED" :
		(sceneRigidAttachment ? "SCENE_RIGID_ATTACHMENT_GATED" :
		(sceneSoftPairAttachment ?
			"SCENE_SOFT_PAIR_ATTACHMENT_GATED" :
		(sceneSoftSoftWake ? "SCENE_SOFT_SOFT_WAKE_GATED" :
		(sceneMultiSceneIsolation ?
			"SCENE_MULTI_SCENE_ISOLATION_GATED" :
		(sceneKinematicBox ? "SCENE_KINEMATIC_COUPLING_GATED" :
		(sceneElementFilter ?
			(scenePartialElementFilter ?
				"SCENE_PARTIAL_ELEMENT_FILTER_GATED" :
				"SCENE_ELEMENT_FILTER_GATED") :
		(sceneWorldPin ? "SCENE_WORLD_PIN_GATED" :
		(sceneBufferMutation ? "SCENE_BUFFER_MUTATION_GATED" :
		(sceneSoftChurn ? "SCENE_SOFT_CHURN_GATED" :
		(sceneMixedSleepIslands ?
			"SCENE_MIXED_SLEEP_ISLANDS_GATED" :
		(sceneSoftRigidWake ? "SCENE_SOFT_RIGID_WAKE_GATED" :
		(sceneSoftSleepWake ? "SCENE_SOFT_SLEEP_WAKE_GATED" :
		(sceneLifecycle ? "SCENE_LIFECYCLE_GATED" :
			(sceneDynamicChurn ?
				"SCENE_DYNAMIC_LIFECYCLE_GATED" :
			(sceneMultiSoft ?
				"SCENE_MULTI_SOFT_ISLANDS_GATED" :
			(sceneMultiDynamic ?
				"SCENE_MULTI_DYNAMIC_COUPLING_GATED" :
			(sceneDynamic ? "SCENE_DYNAMIC_COUPLING_GATED" :
				(sceneStaticChurn ?
					"SCENE_STATIC_LIFECYCLE_GATED" :
					(sceneStatic ? "SCENE_STATIC_CONTACT_GATED" :
						"COMPONENT_GATED"))))))))))))))))))))))));
	if(sceneTaskGraphWriteBack)
		validation = "SCENE_TASKGRAPH_WRITEBACK_GATED";
	if(sceneTaskGraphPipeline)
		validation = "SCENE_TASKGRAPH_PIPELINE_GATED";
	if(sceneVisualShowcase)
		validation = "SCENE_VISUAL_SHOWCASE_GATED";
	if(sceneOgcSandwich)
		validation = "SCENE_OGC_SANDWICH_GATED";
	if(sceneSphereLongRoll)
		validation = "SCENE_SPHERE_LONG_ROLL_GATED";
	if(sceneSphereSoftSoftGlancing)
		validation = "SCENE_SPHERE_SOFT_SOFT_GLANCING_GATED";
	if(sceneSoftSoftTorque)
		validation = "SCENE_SOFT_SOFT_TORQUE_GATED";
	if(sceneGroundEmbeddedTetProbe)
		validation = "SCENE_GROUND_EMBEDDED_TET_PROBE_GATED";
	if(sceneTaskGraphRigidTriangleSurfaceFeatureOverlap)
		validation =
			"SCENE_TASKGRAPH_RIGID_TRIANGLE_FEATURE_OVERLAP_GATED";
	if(sceneMaxDepenetrationVelocity)
		validation =
			"SCENE_MAX_DEPENETRATION_VELOCITY_GATED";
	if(sceneRigidTriangleSteadyContact)
		validation = "SCENE_RIGID_TRIANGLE_STEADY_CONTACT_GATED";
	if(sceneSpeculativeCcd)
		validation =
			"SCENE_SPECULATIVE_CCD_GATED";
	if(sceneSphereReverseFeature)
		validation =
			"SCENE_SPHERE_REVERSE_FEATURE_GATED";
	if(sceneCapsuleReverseFeature)
		validation =
			"SCENE_CAPSULE_REVERSE_FEATURE_GATED";
	if(sceneConvexReverseFeature)
		validation =
			"SCENE_CONVEX_REVERSE_FEATURE_GATED";
	HeadlessGateClassification classification;
	classification.sceneIntegrated = sceneIntegrated;
	classification.validation = validation;
	return classification;
}


} // namespace SnippetDeformableVolumeAVBDFixtures
