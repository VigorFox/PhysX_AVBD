## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions
## are met:
##  * Redistributions of source code must retain the above copyright
##    notice, this list of conditions and the following disclaimer.
##  * Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  * Neither the name of NVIDIA CORPORATION nor the names of its
##    contributors may be used to endorse or promote products derived
##    from this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
## EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
## OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##
## Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#
# Build LowLevelDynamics common
#

SET(PHYSX_SOURCE_DIR ${PHYSX_ROOT_DIR}/source)
SET(LL_SOURCE_DIR ${PHYSX_SOURCE_DIR}/lowleveldynamics/src)

# Include here after the directories are defined so that the platform specific file can use the variables.
include(${PHYSX_ROOT_DIR}/${PROJECT_CMAKE_FILES_DIR}/${TARGET_BUILD_PLATFORM}/LowLevelDynamics.cmake)


SET(LLDYNAMICS_BASE_DIR ${PHYSX_ROOT_DIR}/source/lowleveldynamics)

SET(LLDYNAMICS_INCLUDES
	${LLDYNAMICS_BASE_DIR}/include/DyArticulationCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyVArticulation.h
	${LLDYNAMICS_BASE_DIR}/include/DyArticulationTendon.h
	${LLDYNAMICS_BASE_DIR}/include/DyArticulationMimicJointCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableBodyCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableSurface.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableSurfaceCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableVolume.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableVolumeCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyFeatherstoneArticulation.h
	${LLDYNAMICS_BASE_DIR}/include/DyFeatherstoneArticulationJointData.h
	${LLDYNAMICS_BASE_DIR}/include/DyFeatherstoneArticulationUtils.h
	${LLDYNAMICS_BASE_DIR}/include/DyConstraint.h
	${LLDYNAMICS_BASE_DIR}/include/DyConstraintWriteBack.h
	${LLDYNAMICS_BASE_DIR}/include/DyContext.h
	${LLDYNAMICS_BASE_DIR}/include/DySleepingConfigulation.h
	${LLDYNAMICS_BASE_DIR}/include/DyThresholdTable.h
	${LLDYNAMICS_BASE_DIR}/include/DyArticulationJointCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyParticleSystemCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyParticleSystem.h
	${LLDYNAMICS_BASE_DIR}/include/DyIslandManager.h
)
SOURCE_GROUP("include" FILES ${LLDYNAMICS_INCLUDES})

SET(LLDYNAMICS_SHARED
	${LLDYNAMICS_BASE_DIR}/shared/DyCpuGpuArticulation.h
	${LLDYNAMICS_BASE_DIR}/shared/DyCpuGpu1dConstraint.h
    ${LLDYNAMICS_BASE_DIR}/shared/DyCpuGpuBiasCoefficient.h
	${LLDYNAMICS_BASE_DIR}/shared/DyAvbdOwnerWaveContract.h
)
SOURCE_GROUP("shared" FILES ${LLDYNAMICS_SHARED})

SET(LLDYNAMICS_SOURCE		
	${LLDYNAMICS_BASE_DIR}/src/DyAllocator.h
	${LLDYNAMICS_BASE_DIR}/src/DyAllocator.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyArticulationContactPrep.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyArticulationMimicJoint.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/rigid/DyAvbdBodyConversion.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/core/DyAvbdConstraint.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactPrep.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactPrep.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactBounds.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactBounds.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactDetection.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactDetection.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactEpoch.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactEpoch.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactFeatureGeometry.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactGeometry.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactGeometryQueries.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactMaterial.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactPlane.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactPlane.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactMeshQueries.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactRigidBoxGeometry.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactRigidSoft.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactRigidSoft.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactSelf.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactSelf.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactSoftPair.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactSoftPair.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactTriangleSurface.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactTriangleSurfaceTypes.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactVelocityEpoch.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactVelocityEpoch.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdSoftContactPrep.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdSelfCollisionTopology.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdSelfCollisionTopology.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcParameters.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcAdmission.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcAdmission.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcCurrentPose.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcCurrentPose.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcDeformableResponse.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcResponse.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcResponse.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcDynamicResponse.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcDynamicTangentVelocity.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcDynamicTriangle.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcDynamicVelocity.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcGeometryEpoch.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcGeometryEpoch.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcGeometryProvider.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcGeometryQueries.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcPairState.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcPairState.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcPostAlTrustRegion.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcTerminal.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcTerminal.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcTerminalAdmission.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcTerminalState.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcTriangleCoreGeometry.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcTriangleCoreGeometry.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcTrustRegion.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcTrustRegion.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/backend/cpu/DyAvbdCpuDispatch.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/backend/cpu/DyAvbdCpuIsa.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/backend/cpu/DyAvbdCpuProducer.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/backend/cpu/DyAvbdCpuProducer.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/backend/gpu/DyAvbdGpuWaveBridge.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/pipeline/DyAvbdDynamics.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/pipeline/DyAvbdDynamics.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/pipeline/DyAvbdDynamicsPrep.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/rigid/DyAvbdKinematicShell.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/rigid/DyAvbdKinematicShell.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/backend/cpu/DyAvbdKernelsSse2.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointPreparation.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointPreparation.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdCoupledD6.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdCoupledD6.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointCoupledMath.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointCoupledSystem.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointCoupledSystem.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointDriveMath.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointDualPhase.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointFinalization.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointFinalization.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointGeometryPolicy.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointIteration.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointIteration.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointOgcPhase.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointOgcPhase.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointObjectiveCompilation.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointObjectiveCompilation.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointPhaseState.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointPhaseState.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointPositionSolves.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointPositionSolves.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdNativeMotorVelocity.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdNativeMotorVelocity.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdSpatialTendon.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdSpatialTendon.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointProjection.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointProjection.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointSoftExecutionData.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointSoftExecutionData.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointSupportPolicies.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointSupportPolicies.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointVelocityPolicies.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdJointVelocityPolicies.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdLinearDriveSolve.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdLinearDriveSolve.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdLocalJointSolve.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/scheduling/DyAvbdParallel.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/DyAvbdSolver.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/DyAvbdSolver.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAl.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAl.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAlBodyVelocity.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAlContactResponse.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAlContactResponse.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAlFrictionPhase.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAlPosePhase.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAlRecoveryPhase.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAlSoftVelocity.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAlTerminalPhase.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/post_al/DyAvbdPostAlVelocityState.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/rigid/DyAvbdRigidPhases.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/rigid/DyAvbdRigidPhases.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcStaticResponse.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcStaticResponse.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBody.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyCompiledData.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyCompiledData.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyCreation.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyCreation.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyData.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyData.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyEpochSafety.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyEpochSafety.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyFinalization.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyFinalization.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyMechanics.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyMechanics.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyPolicy.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyPolicy.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyPrimal.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyPrimal.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyPrimalPolicy.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyPrimalPolicy.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyRuntime.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyRuntime.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftContactGeometry.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyScalar.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyScheduling.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyStep.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyStepState.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyTopology.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyWorkspace.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyWorkspace.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftWarmstart.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftWarmstart.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcPlanValidation.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcPlanValidation.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftDualUpdate.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftJointSolve.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/diagnostics/DyAvbdSoftBodyDiagnostics.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/diagnostics/DyAvbdSoftBodyDiagnostics.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/scheduling/DyAvbdSchedulingPolicy.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/joint/DyAvbdSolverJointPath.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/rigid/DyAvbdSolverBody.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/rigid/DyAvbdSolverBody.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/pipeline/DyAvbdTasks.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/pipeline/DyAvbdTasks.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/backend/gpu/DyAvbdGpuWaveBackend.h
	${LLDYNAMICS_BASE_DIR}/src/avbd/backend/gpu/DyAvbdGpuWaveCallbacks.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/core/DyAvbdTypes.h
	${LLDYNAMICS_BASE_DIR}/src/DyFeatherstoneArticulation.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyFeatherstoneForwardDynamic.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyFeatherstoneInverseDynamic.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintPartition.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintSetup.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintSetupBlock.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyContactPrep.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyContactPrep4.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyDynamicsBase.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyDynamics.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyFrictionCorrelation.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyRigidBodyToSolverBody.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraints.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraintsBlock.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverControl.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraint1DStep.h
	${LLDYNAMICS_BASE_DIR}/src/DyThreadContext.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyThresholdTable.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyTGSDynamics.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyTGSContactPrep.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyTGSContactPrepBlock.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyArticulationContactPrep.h
	${LLDYNAMICS_BASE_DIR}/src/DyBodyCoreIntegrator.h
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintPartition.h
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintPrep.h
	${LLDYNAMICS_BASE_DIR}/src/DyContactPrep.h
	${LLDYNAMICS_BASE_DIR}/src/DyContactPrepShared.h
	${LLDYNAMICS_BASE_DIR}/src/DyContactReduction.h
	${LLDYNAMICS_BASE_DIR}/src/DyCorrelationBuffer.h
	${LLDYNAMICS_BASE_DIR}/src/DyDynamicsBase.h
	${LLDYNAMICS_BASE_DIR}/src/DyDynamics.h
	${LLDYNAMICS_BASE_DIR}/src/DyFrictionPatch.h
	${LLDYNAMICS_BASE_DIR}/src/DyFrictionPatchStreamPair.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverBody.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraint1D.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraint1D4.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraintDesc.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraintExtShared.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraintsShared.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraintTypes.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverContact.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverContact4.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverContext.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverControl.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverCore.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverCore.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverExt.h
	${LLDYNAMICS_BASE_DIR}/src/DyThreadContext.h
	${LLDYNAMICS_BASE_DIR}/src/DyTGSDynamics.h
    ${LLDYNAMICS_BASE_DIR}/src/DyTGSContactPrep.h
    ${LLDYNAMICS_BASE_DIR}/src/DyTGS.h
    ${LLDYNAMICS_BASE_DIR}/src/DyPGS.h
	${LLDYNAMICS_BASE_DIR}/src/DySleep.h
	${LLDYNAMICS_BASE_DIR}/src/DySleep.cpp
)

# P6: AVX2+FMA code lives in its own translation unit.  No target-wide AVX
# switch is permitted: the baseline code must remain executable on every x64
# CPU with SSE2, while runtime dispatch admits this TU only after CPUID/XGETBV.
SET(LLDYNAMICS_AVX2_FMA_SOURCE
	${LLDYNAMICS_BASE_DIR}/src/avbd/backend/cpu/DyAvbdKernelsAvx2Fma.cpp
)
SET(LLDYNAMICS_AVX2_FMA_COMPILED OFF)

# Soft-body component entry points are implemented once in LowLevelDynamics
# and consumed by private validation Snippets through the PhysX shared-library
# boundary. Static SDK builds keep ordinary symbols.
SET_SOURCE_FILES_PROPERTIES(
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactDetection.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactBounds.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactEpoch.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactPlane.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactRigidSoft.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactSelf.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactSoftPair.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdContactTriangleSurface.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/contact/DyAvbdSelfCollisionTopology.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcAdmission.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcPairState.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/ogc/DyAvbdOgcResponse.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyScalar.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyCompiledData.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyCreation.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyData.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyEpochSafety.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyFinalization.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyMechanics.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyPolicy.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyPrimal.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyPrimalPolicy.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyRuntime.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/scheduling/DyAvbdSchedulingPolicy.cpp
	${LLDYNAMICS_BASE_DIR}/src/avbd/solver/soft/DyAvbdSoftBodyWorkspace.cpp
	PROPERTIES COMPILE_DEFINITIONS DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS=1
)

IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
	IF(MSVC)
		SET_SOURCE_FILES_PROPERTIES(${LLDYNAMICS_AVX2_FMA_SOURCE}
			PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
		LIST(APPEND LLDYNAMICS_SOURCE ${LLDYNAMICS_AVX2_FMA_SOURCE})
		SET(LLDYNAMICS_AVX2_FMA_COMPILED ON)
	ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		SET_SOURCE_FILES_PROPERTIES(${LLDYNAMICS_AVX2_FMA_SOURCE}
			PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
		LIST(APPEND LLDYNAMICS_SOURCE ${LLDYNAMICS_AVX2_FMA_SOURCE})
		SET(LLDYNAMICS_AVX2_FMA_COMPILED ON)
	ENDIF()
ENDIF()
SOURCE_GROUP("src" FILES ${LLDYNAMICS_SOURCE})

ADD_LIBRARY(LowLevelDynamics ${LOWLEVELDYNAMICS_LIBTYPE}
	${LLDYNAMICS_INCLUDES}
	${LLDYNAMICS_SHARED}
	${LLDYNAMICS_SOURCE}
)

GET_TARGET_PROPERTY(PHYSXFOUNDATION_INCLUDES PhysXFoundation INTERFACE_INCLUDE_DIRECTORIES)

TARGET_INCLUDE_DIRECTORIES(LowLevelDynamics 
	PRIVATE ${LOWLEVELDYNAMICS_PLATFORM_INCLUDES}

	PRIVATE ${PHYSXFOUNDATION_INCLUDES}

	PRIVATE ${PHYSX_ROOT_DIR}/include

	PRIVATE ${PHYSX_SOURCE_DIR}/common/src
	
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src/contact
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/include
	
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/api/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/common/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/common/include/pipeline
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/common/include/utils
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/software/include

	PRIVATE ${PHYSX_SOURCE_DIR}/lowleveldynamics/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowleveldynamics/shared
	PRIVATE ${PHYSX_SOURCE_DIR}/lowleveldynamics/src
	
	PRIVATE ${PHYSX_SOURCE_DIR}/physxgpu/include
)

# Use generator expressions to set config specific preprocessor definitions
TARGET_COMPILE_DEFINITIONS(LowLevelDynamics 

	# Common to all configurations
	PRIVATE ${LOWLEVELDYNAMICS_COMPILE_DEFS}
	PRIVATE PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD=1
)

IF(LLDYNAMICS_AVX2_FMA_COMPILED)
	TARGET_COMPILE_DEFINITIONS(LowLevelDynamics
		PRIVATE PX_AVBD_CPU_AVX2_FMA_COMPILED=1
	)
ENDIF()

SET_TARGET_PROPERTIES(LowLevelDynamics PROPERTIES 
    ARCHIVE_OUTPUT_NAME_DEBUG "LowLevelDynamics_static"
    ARCHIVE_OUTPUT_NAME_CHECKED "LowLevelDynamics_static"
    ARCHIVE_OUTPUT_NAME_PROFILE "LowLevelDynamics_static"
    ARCHIVE_OUTPUT_NAME_RELEASE "LowLevelDynamics_static"
)

IF(LLDYNAMICS_COMPILE_PDB_NAME_DEBUG)
	SET_TARGET_PROPERTIES(LowLevelDynamics PROPERTIES 
		COMPILE_PDB_NAME_DEBUG "${LLDYNAMICS_COMPILE_PDB_NAME_DEBUG}"
		COMPILE_PDB_NAME_CHECKED "${LLDYNAMICS_COMPILE_PDB_NAME_CHECKED}"
		COMPILE_PDB_NAME_PROFILE "${LLDYNAMICS_COMPILE_PDB_NAME_PROFILE}"
		COMPILE_PDB_NAME_RELEASE "${LLDYNAMICS_COMPILE_PDB_NAME_RELEASE}"
	)
ENDIF()

IF(PX_EXPORT_LOWLEVEL_PDB)
	SET_TARGET_PROPERTIES(LowLevelDynamics PROPERTIES 
		COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG "${PHYSX_ROOT_DIR}/${PX_ROOT_LIB_DIR}/debug/"
		COMPILE_PDB_OUTPUT_DIRECTORY_CHECKED "${PHYSX_ROOT_DIR}/${PX_ROOT_LIB_DIR}/checked/"
		COMPILE_PDB_OUTPUT_DIRECTORY_PROFILE "${PHYSX_ROOT_DIR}/${PX_ROOT_LIB_DIR}/profile/"
		COMPILE_PDB_OUTPUT_DIRECTORY_RELEASE "${PHYSX_ROOT_DIR}/${PX_ROOT_LIB_DIR}/release/"
	)
ENDIF()

IF(PX_GENERATE_SOURCE_DISTRO)
	LIST(APPEND SOURCE_DISTRO_FILE_LIST ${LLDYNAMICS_INCLUDES})
	LIST(APPEND SOURCE_DISTRO_FILE_LIST ${LLDYNAMICS_SHARED})
	LIST(APPEND SOURCE_DISTRO_FILE_LIST ${LLDYNAMICS_SOURCE})	
ENDIF()

# enable -fPIC so we can link static libs with the editor
SET_TARGET_PROPERTIES(LowLevelDynamics PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
