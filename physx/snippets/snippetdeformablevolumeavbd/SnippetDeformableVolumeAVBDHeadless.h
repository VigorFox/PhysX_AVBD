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

#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_HEADLESS_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_HEADLESS_H

#include "SnippetDeformableVolumeAVBDReport.h"

#include <string>

namespace SnippetDeformableVolumeAVBDHeadless
{

struct HeadlessResultContext
{
	const std::string* caseName;
	const char* solverName;
	physx::PxU32 frames;
	physx::PxReal dt;
	physx::PxU32 dispatcherThreads;
	bool parallelExecution;
	bool sequentialExecution;

	SnippetDeformableVolumeAVBDValidation::DeformableVolumeMetrics* metrics;
	const SnippetDeformableVolumeAVBDReport::
		DeformableVolumePerformanceMetrics* performance;
	const SnippetDeformableVolumeAVBDValidation::OgcSandwichMetrics*
		ogcSandwichMetrics;
	const SnippetDeformableVolumeAVBDValidation::RotationMetrics*
		visualRotationMetrics;
	const SnippetDeformableVolumeAVBDValidation::RotationMetrics*
		visualPrimaryCubeRotationMetrics;
	const SnippetDeformableVolumeAVBDValidation::RotationMetrics*
		visualSphereRotationMetrics;
	const SnippetDeformableVolumeAVBDValidation::VisualInteractionMetrics*
		visualInteractionMetrics;
	const SnippetDeformableVolumeAVBDValidation::SoftContactPhaseMetrics*
		softContactPhaseMetrics;
	const SnippetDeformableVolumeAVBDValidation::SoftSoftTorqueMetrics*
		softSoftTorqueMetrics;
	const SnippetDeformableVolumeAVBDValidation::GroundEmbeddedTetProbeMetrics*
		groundEmbeddedTetMetrics;
	const SnippetDeformableVolumeAVBDValidation::VolumeHealthMonitor*
		volumeHealthMonitor;
	const SnippetDeformableVolumeAVBDValidation::VolumeSkinningMetrics*
		volumeSkinningMetrics;
	const SnippetDeformableVolumeAVBDValidation::ReverseFeatureMetrics*
		reverseFeatureMetrics;
	SnippetDeformableVolumeAVBDValidation::ReverseSweptMetrics*
		reverseSweptMetrics;
	const SnippetDeformableVolumeAVBDValidation::DeformingReverseSweptMetrics*
		deformingReverseSweptMetrics;
	const SnippetDeformableVolumeAVBDValidation::RotationalSweepMetrics*
		rotationalSweepMetrics;
	SnippetDeformableVolumeAVBDValidation::KinematicFiniteSweptMetrics
		kinematicFiniteSweptMetrics;
	SnippetDeformableVolumeAVBDValidation::DynamicFiniteSweptMetrics
		dynamicFiniteSweptMetrics;

	physx::PxReal dynamicInitialY;
	physx::PxReal secondDynamicInitialY;
	bool visualSphereLongRunBounded;
	bool sphereRollingKinematicsValid;
	bool sphereLongRollRegressionPassed;
	physx::PxU32 sphereRollWindowBeginFrame;
	physx::PxU32 sphereRollWindowEndFrame;
	physx::PxU32 sphereRollCheckpointCount;
	physx::PxU32 sphereRollCheckpointInterval;
	physx::PxU32 ogcPressureDriveFrames;
	physx::PxReal softSoftTorqueMinAngularMomentum;
	physx::PxReal softSoftTorqueMinAngularSpeed;
	physx::PxU32 fatalErrors;
	physx::PxU32 warningErrors;

	HeadlessResultContext();
};

bool validateHeadlessResult(HeadlessResultContext& context);
void printHeadlessResult(
	const HeadlessResultContext& context,
	bool passed);

} // namespace SnippetDeformableVolumeAVBDHeadless

#endif // SNIPPET_DEFORMABLE_VOLUME_AVBD_HEADLESS_H
