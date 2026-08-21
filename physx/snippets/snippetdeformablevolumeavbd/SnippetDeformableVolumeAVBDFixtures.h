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

#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_FIXTURES_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_FIXTURES_H

#include "PxPhysicsAPI.h"

#include <string>

namespace SnippetDeformableVolumeAVBDFixtures
{

struct HeadlessGateClassification
{
	bool sceneIntegrated;
	const char* validation;
};

HeadlessGateClassification classifyHeadlessGate(
	const std::string& caseName);

void rotateVerticesAroundZ(
	physx::PxArray<physx::PxVec3>& vertices,
	const physx::PxVec3& center,
	physx::PxReal angle);

void scaleVerticesAboutCenter(
	physx::PxArray<physx::PxVec3>& vertices,
	const physx::PxVec3& center,
	const physx::PxVec3& scale);

void createSubdividedCubeSurface(
	physx::PxArray<physx::PxVec3>& vertices,
	physx::PxArray<physx::PxU32>& triangles,
	const physx::PxVec3& center,
	physx::PxReal sideLength,
	physx::PxU32 subdivisions);

void createLayeredConeSurface(
	physx::PxArray<physx::PxVec3>& vertices,
	physx::PxArray<physx::PxU32>& triangles,
	const physx::PxVec3& center,
	physx::PxReal radius,
	physx::PxReal height,
	physx::PxU32 ringSegments,
	physx::PxU32 heightSegments);

bool isSceneCpuVolumeCase(const std::string& caseName);
bool isSceneCpuVolumeKinematicRigidCase(const std::string& caseName);
bool isSceneCpuVolumeSpeculativeCcdCase(const std::string& caseName);
bool isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
	const std::string& caseName);
bool isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
	const std::string& caseName);
bool isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
	const std::string& caseName);
bool isSceneCpuVolumeHeightFieldSweptCcdCase(
	const std::string& caseName);
bool isSceneCpuVolumeRigidTriangleSteadyContactCase(
	const std::string& caseName);
bool isSceneCpuVolumeTaskGraphRigidTriangleSurfaceLargeCase(
	const std::string& caseName);
bool isSceneCpuVolumeTaskGraphRigidTriangleSurfaceFeatureOverlapCase(
	const std::string& caseName);
bool isSceneCpuVolumeTaskGraphRigidTriangleSurfaceThresholdCase(
	const std::string& caseName);
bool isSceneCpuVolumeTaskGraphWriteBackCase(
	const std::string& caseName);
bool isSceneCpuVolumeTaskGraphWriteBackFourWayCase(
	const std::string& caseName);
bool isSceneCpuVolumeTaskGraphPipelineCase(
	const std::string& caseName);
bool isSceneCpuVolumeTaskGraphPureSoftCase(
	const std::string& caseName);
bool isSceneCpuVolumeTaskGraphDirectSimulationDomainCase(
	const std::string& caseName);
bool isSceneCpuVolumeSphereReverseSweptCcdCase(
	const std::string& caseName);
bool isSceneCpuVolumeCapsuleReverseSweptCcdCase(
	const std::string& caseName);
bool isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
	const std::string& caseName);
bool isSceneCpuVolumeConvexReverseSweptCcdCase(
	const std::string& caseName);
bool isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
	const std::string& caseName);
bool isSceneCpuVolumeDeformingReverseSweptCcdCase(
	const std::string& caseName);
bool isComponentDenseNoContactCase(const std::string& caseName);
bool isComponentManySmallNoContactCase(const std::string& caseName);

} // namespace SnippetDeformableVolumeAVBDFixtures

#endif
