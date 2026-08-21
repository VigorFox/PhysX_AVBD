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
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// ****************************************************************************
// SnippetDeformableVolumeAVBD
//
// CPU-only AVBD equivalent of SnippetDeformableVolume (GPU FEM).
// Demonstrates multiple VBD soft bodies -- cuboids, a sphere, and a true
// collision-surface cone -- dropping onto a rigid ground plane. All elastic
// forces use Neo-Hookean energy via VBD; contacts (ground, soft-soft,
// soft-rigid) are enforced through AVBD adaptive penalty.
//
// Scene layout:
//   Body 0 : cuboid at (-2.0, 8.0, 0.0) -- independent deformable ground impact
//   Body 1 : sphere at (-0.2, 2.0, 0.0) -- glancing overlap with body 0
//   Body 2 : cone   at ( 3.2,11.0, 1.2) -- separate analytic cone drop lane
//   Body 3 : cuboid at ( 7.0, 4.2, 0.0) -- tilted on a narrow rigid box edge
//   Body 4 : cube   at ( 5.4, 8.8, 0.3) -- off-center follower amplifies body 3 rotation
//   Rigid  : box    at ( 7.6, 0.55,0.0) -- narrow step, uses SDF contact path
//
// No GPU or CUDA dependency -- runs entirely on the CPU.
// ****************************************************************************

#include <cstdio>
#include <cmath>
#include "PxPhysicsAPI.h"
#include "cooking/PxCooking.h"
#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "extensions/PxDeformableVolumeExt.h"

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetDeformableAVBDSkinning.h"
#include "../snippetdeformablevolume/MeshGenerator.h"

#include "SnippetDeformableVolumeAVBD.h"
#include "SnippetDeformableVolumeAVBDFixtures.h"
#include "SnippetDeformableVolumeAVBDHeadless.h"
#include "SnippetDeformableVolumeAVBDReport.h"
#include "SnippetDeformableVolumeAVBDValidation.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace physx;
using namespace physx::Dy;
using SnippetDeformableVolumeAVBDFixtures::createLayeredConeSurface;
using SnippetDeformableVolumeAVBDFixtures::createSubdividedCubeSurface;
using SnippetDeformableVolumeAVBDFixtures::isComponentDenseNoContactCase;
using SnippetDeformableVolumeAVBDFixtures::isComponentManySmallNoContactCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeCapsuleReverseSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeConvexReverseSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeDeformingReverseSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeHeightFieldSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeKinematicRigidCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeRigidTriangleSteadyContactCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeRotationalConvexReverseSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeSphereReverseSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeSpeculativeCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTaskGraphDirectSimulationDomainCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTaskGraphPipelineCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTaskGraphPureSoftCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTaskGraphRigidTriangleSurfaceFeatureOverlapCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTaskGraphRigidTriangleSurfaceLargeCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTaskGraphRigidTriangleSurfaceThresholdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTaskGraphWriteBackCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTaskGraphWriteBackFourWayCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::isSceneCpuVolumeTriangleSurfaceSweptCcdCase;
using SnippetDeformableVolumeAVBDFixtures::rotateVerticesAroundZ;
using SnippetDeformableVolumeAVBDFixtures::scaleVerticesAboutCenter;
using SnippetDeformableVolumeAVBDReport::DeformableVolumePerformanceMetrics;
using SnippetDeformableVolumeAVBDReport::PerformanceReportConfig;
using SnippetDeformableVolumeAVBDReport::finalizePerformanceMetrics;
using SnippetDeformableVolumeAVBDReport::printPerformanceResult;
using SnippetDeformableVolumeAVBDValidation::OgcSandwichFrameSample;
using SnippetDeformableVolumeAVBDValidation::OgcSandwichMetrics;
using SnippetDeformableVolumeAVBDValidation::OgcSandwichMonitor;
using SnippetDeformableVolumeAVBDValidation::RotationMetrics;
using SnippetDeformableVolumeAVBDValidation::RotationMonitor;
using SnippetDeformableVolumeAVBDValidation::RotationSamplingConfig;
using SnippetDeformableVolumeAVBDValidation::RotationalSweepMetrics;
using SnippetDeformableVolumeAVBDValidation::ReverseSweptMetrics;
using SnippetDeformableVolumeAVBDValidation::SweptGeometrySampler;
using SnippetDeformableVolumeAVBDValidation::ReverseFeatureMetrics;
using SnippetDeformableVolumeAVBDValidation::DeformingReverseSweptMetrics;
using SnippetDeformableVolumeAVBDValidation::DeformableVolumeMetrics;
using SnippetDeformableVolumeAVBDValidation::DynamicFiniteSweptMetrics;
using SnippetDeformableVolumeAVBDValidation::VolumeSkinningMetrics;
using SnippetDeformableVolumeAVBDValidation::KinematicFiniteSweptMetrics;
using SnippetDeformableVolumeAVBDValidation::SoftContactPhaseFrameSample;
using SnippetDeformableVolumeAVBDValidation::SoftContactPhaseMetrics;
using SnippetDeformableVolumeAVBDValidation::SoftContactPhaseMonitor;
using SceneCpuSoftSoftTorqueMetrics =
	SnippetDeformableVolumeAVBDValidation::SoftSoftTorqueMetrics;
using SnippetDeformableVolumeAVBDValidation::SoftSoftTorqueFrameSample;
using SnippetDeformableVolumeAVBDValidation::SoftSoftTorqueMonitor;
using SnippetDeformableVolumeAVBDValidation::VolumeHealthMonitor;
using SnippetDeformableVolumeAVBDValidation::VolumeHealthSample;
using SnippetDeformableVolumeAVBDValidation::VisualInteractionFrameSample;
using SnippetDeformableVolumeAVBDValidation::VisualInteractionMetrics;
using SnippetDeformableVolumeAVBDValidation::VisualInteractionMonitor;
using SceneCpuGroundEmbeddedTetProbeMetrics =
	SnippetDeformableVolumeAVBDValidation::GroundEmbeddedTetProbeMetrics;
using SnippetDeformableVolumeAVBDValidation::GroundEmbeddedTetProbeFrameSample;
using SnippetDeformableVolumeAVBDValidation::isRollingKinematicsValid;
using SnippetDeformableVolumeAVBDValidation::isRotationLongRunBounded;
using SnippetDeformableVolumeAVBDValidation::getCapsuleSignedSeparation;
using SnippetDeformableVolumeAVBDValidation::sampleGroundEmbeddedTetProbe;
using SnippetDeformableVolumeAVBDValidation::updateReverseFeatureMetrics;

namespace Headless = SnippetDeformableVolumeAVBDHeadless;

#ifndef AVBD_VOLUME_SNIPPET_NAME
#define AVBD_VOLUME_SNIPPET_NAME "SnippetDeformableVolumeAVBD"
#endif

#ifndef AVBD_VOLUME_DEFAULT_CASE
// The default exercises the public CPU AVBD actor with a deliberately
// distinct collision/simulation mesh.  "current-all" remains available as
// an explicit low-level component stress corpus.
#define AVBD_VOLUME_DEFAULT_CASE "scene-volume-partial-element-filter"
#endif

#ifndef AVBD_VOLUME_VISUAL_CASE
#define AVBD_VOLUME_VISUAL_CASE "scene-volume-visual-showcase"
#endif

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static PxDefaultAllocator      gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*           gFoundation  = NULL;
static PxPhysics*              gPhysics     = NULL;
static PxDefaultCpuDispatcher* gDispatcher  = NULL;
static PxMaterial*             gMaterial    = NULL;
static PxPvd*                  gPvd         = NULL;
static bool                    gExtensionsInitialized = false;
static Snippets::HeadlessOptions gHeadlessOptions;

PxScene*                       gScene       = NULL;
static PxScene*                gSceneCpuSecondScene = NULL;

PxArray<AvbdSoftParticle>      gParticles;
PxArray<AvbdSoftBody>          gSoftBodies;
PxArray<SoftBodyRenderData>    gSoftBodyRenderData;
VolumeAvbdSkinningRenderData   gVolumeAvbdSkinningRenderData;

static PxArray<AvbdSoftContact> gContacts;
static PxArray<AvbdRigidBox>     gRigidBoxes;
static AvbdSoftBodyWorkspace     gSoftWorkspace;
static PxDeformableVolume*       gSceneCpuVolume = NULL;
static PxDeformableVolume*       gSceneCpuSecondVolume = NULL;
// Used only by the P3 four-entry taskgraph fixture.  These retain normal
// Scene actor ownership and are released before the shared mesh/material.
static PxArray<PxDeformableVolume*> gSceneCpuTaskGraphExtraVolumes;
// P6 heterogeneous-body fixture meshes outlive their extra actors but remain
// independent from the primary shared mesh.
static PxArray<PxDeformableVolumeMesh*> gSceneCpuTaskGraphExtraMeshes;
// Interactive showcase meshes are released only after every public Volume
// actor that references them. The cube mesh is shared by three actors; sphere
// and cone retain their own cooked collision/simulation pair.
static PxArray<PxDeformableVolumeMesh*> gSceneCpuVisualVolumeMeshes;
static const PxU32 SCENE_CPU_VISUAL_SPHERE_EARLY_END_FRAME = 600;
static const PxU32 SCENE_CPU_VISUAL_SPHERE_LATE_BEGIN_FRAME = 1200;
static const PxU32 SCENE_CPU_VISUAL_SPHERE_LONG_RUN_MIN_FRAMES = 2000;
static const PxU32 SCENE_CPU_SPHERE_ROLL_WINDOW_BEGIN_FRAME = 3000;
static const PxU32 SCENE_CPU_SPHERE_ROLL_WINDOW_END_FRAME = 7000;
static const PxU32 SCENE_CPU_SPHERE_ROLL_CHECKPOINT_INTERVAL = 1000;
static const PxU32 SCENE_CPU_SPHERE_ROLL_CHECKPOINT_COUNT = 8;
static const PxU32 SCENE_CPU_SPHERE_ROLL_REGRESSION_MIN_FRAMES = 4000;
static const PxReal SCENE_CPU_SPHERE_ROLL_MAX_WINDOW_MEAN_SPEED = 0.12f;
static const PxReal SCENE_CPU_SPHERE_ROLL_MAX_FINAL_SPEED = 0.15f;
static const PxReal SCENE_CPU_SPHERE_ROLL_MIN_GENERATED_ANGULAR_SPEED = 0.1f;
static const PxReal SCENE_CPU_SPHERE_ROLL_MIN_ORIENTATION_CHANGE = 0.03f;
static const PxReal SCENE_CPU_SPHERE_ROLL_MAX_RIGID_SLIP_SPEED = 0.03f;
static const PxReal SCENE_CPU_VISUAL_SPHERE_MAX_LATE_SPEED_FLOOR = 0.75f;
static const PxReal SCENE_CPU_VISUAL_SPHERE_MAX_LATE_SPEED_RATIO = 1.5f;
// Keep the showcase visibly compliant while retaining the same collision,
// damping and volume-preservation setup used by its long-roll regression.
static const PxReal SCENE_CPU_VISUAL_YOUNGS_MODULUS = 1.2e5f;
static const PxReal SCENE_CPU_VISUAL_POISSONS_RATIO = 0.4f;
static const PxReal SCENE_CPU_VISUAL_MATERIAL_DAMPING = 0.01f;
// This is deliberately a process-static A/B only for the public showcase and
// its two-body long-roll reproduction.  The isolated soft/soft rotation
// controls keep their fixed material friction so they remain independent.
static PxReal getSceneCpuVisualMaterialDynamicFriction()
{
	static const PxReal dynamicFriction = []() -> PxReal
	{
		const char* const value = std::getenv(
			"PHYSX_AVBD_SCENE_VOLUME_DYNAMIC_FRICTION");
		if(value && std::strcmp(value, "0.2") == 0)
			return 0.2f;
		if(value && std::strcmp(value, "0.5") == 0)
			return 0.5f;
		if(value && std::strcmp(value, "0.8") == 0)
			return 0.8f;
		// Absent and invalid values intentionally preserve the established 0.2
		// material friction.
		return 0.2f;
	}();
	return dynamicFriction;
}
// The public body core contributes its linear damping to every particle in
// addition to the material damping above.  The primary cube/sphere lane keeps
// a light damping value so its rolling response remains visible.
static const PxReal SCENE_CPU_VISUAL_SPHERE_LINEAR_DAMPING = 0.005f;
// Keep the soft jaw's body damping light. Its material contributes another
// 0.01, so the rigid uses the summed 0.015 value below. This preserves normal
// gravitational acceleration rather than making the display look terminally
// damped or kinematic.
static const PxReal SCENE_CPU_VISUAL_JAW_LINEAR_DAMPING = 0.005f;
static const PxReal SCENE_CPU_VISUAL_DYNAMIC_BOX_LINEAR_DAMPING =
	SCENE_CPU_VISUAL_JAW_LINEAR_DAMPING +
		SCENE_CPU_VISUAL_MATERIAL_DAMPING;
// Do not hide contact-pipeline defects behind the public body's low-speed
// settling damper.  A value of 10 removed roughly one sixth of every
// particle velocity on every 60 Hz frame once the sphere slowed below
// 0.1 m/s.  That damps the rigid rotation mode itself and makes a grounded
// deformable look frozen even though it never entered the sleep state.  The
// material/body damping and Coulomb contact own physical settling here.
static const PxReal SCENE_CPU_VISUAL_SPHERE_SETTLING_THRESHOLD = 0.0f;
static const PxReal SCENE_CPU_VISUAL_SPHERE_SETTLING_DAMPING = 0.0f;
// The cube advances by roughly 2 m before its first contact with the grounded
// sphere. Starting at -2.0 therefore turns the intended glancing impact into
// an almost centered one. Keep a finite contact lever arm in both the visual
// showcase and its two-body long-roll reproduction.
static const PxReal SCENE_CPU_VISUAL_PRIMARY_CUBE_START_X = -3.2f;
static const PxReal SCENE_CPU_VISUAL_PRIMARY_SPHERE_START_X = -0.2f;
// The visual OGC sandwich is deliberately a dense coupled soft/rigid contact.
// Sixteen position iterations is the stable budget for the current-pose rows;
// a larger global count made the lower jaw's static landing less healthy.
static const PxU32 SCENE_CPU_VISUAL_POSITION_ITERATIONS = 16u;
// Element-filter fixtures validate ownership at one current pose.  A small
// deterministic overlap keeps the test independent of gravity, CCD, and the
// amount of time spent accelerating before the filter is released.
static const PxReal SCENE_CPU_ELEMENT_FILTER_INITIAL_CLEARANCE = 0.05f;
static const PxReal SCENE_CPU_ELEMENT_FILTER_TEST_PENETRATION = 0.03f;
static const PxReal SCENE_CPU_ELEMENT_FILTER_MIN_SUPPRESSED_DEPTH = 0.02f;
static const PxReal SCENE_CPU_ELEMENT_FILTER_SURFACE_TOLERANCE = 0.005f;
static const PxReal SCENE_CPU_ELEMENT_FILTER_CONTACT_OFFSET_LIMIT = 0.06f;
// Keep the falling dynamic rigid between broad horizontal soft faces: yellow
// above, magenta below. The yaw makes the objects visually non-axis-aligned
// while preserving local +Y as the world-up stacking direction.
static const PxQuat SCENE_CPU_VISUAL_DYNAMIC_BOX_ORIENTATION(
	0.36f, PxVec3(0.0f, 1.0f, 0.0f));
static const PxVec3 SCENE_CPU_VISUAL_DYNAMIC_BOX_CENTER(
	6.40f, 25.00f, 0.30f);
static const PxVec3 SCENE_CPU_VISUAL_VERTICAL_JAW_AXIS(
	SCENE_CPU_VISUAL_DYNAMIC_BOX_ORIENTATION.rotate(PxVec3(0.0f, 1.0f, 0.0f)));
// The soft jaw has a 0.7 m local-Y half extent and the rigid has a 0.35 m
// local-Y half extent. Start exactly on the 5 cm current-pose OGC shell, with
// neither mesh overlap nor a two-sided normal preload. There is no artificial
// closing velocity; gravity supplies the visible motion.
static const PxReal SCENE_CPU_VISUAL_JAW_BOX_INITIAL_GAP = 0.050f;
static const PxReal SCENE_CPU_VISUAL_JAW_CENTER_DISTANCE =
	0.70f + 0.35f + SCENE_CPU_VISUAL_JAW_BOX_INITIAL_GAP;
static const PxVec3 SCENE_CPU_VISUAL_UPPER_SOFT_JAW_CENTER(
	SCENE_CPU_VISUAL_DYNAMIC_BOX_CENTER +
		SCENE_CPU_VISUAL_VERTICAL_JAW_AXIS *
			SCENE_CPU_VISUAL_JAW_CENTER_DISTANCE);
static const PxVec3 SCENE_CPU_VISUAL_LOWER_SOFT_JAW_CENTER(
	SCENE_CPU_VISUAL_DYNAMIC_BOX_CENTER -
		SCENE_CPU_VISUAL_VERTICAL_JAW_AXIS *
			SCENE_CPU_VISUAL_JAW_CENTER_DISTANCE);
static const PxVec3 SCENE_CPU_VISUAL_DYNAMIC_BOX_HALF_EXTENTS(
	0.50f, 0.35f, 0.70f);
// Keep the static collision targets named alongside the dynamic sandwich so
// the visual telemetry measures the exact authored rigid boundaries rather
// than inferring them from a broad scene bound.
static const PxReal SCENE_CPU_VISUAL_GROUND_HEIGHT = 0.0f;
static const PxVec3 SCENE_CPU_VISUAL_STATIC_PEDESTAL_CENTER(
	7.6f, 0.55f, 0.0f);
static const PxVec3 SCENE_CPU_VISUAL_STATIC_PEDESTAL_HALF_EXTENTS(
	0.7f, 0.55f, 2.2f);
// Keep a finite but not negligible inertia between the two very heavy soft
// jaws.  A 1 kg target exaggerated asymmetric OGC row noise in this display.
static const PxReal SCENE_CPU_VISUAL_DYNAMIC_BOX_MASS = 10.0f;
// All three bodies start with no imposed velocity and fall solely under the
// scene gravity. This remains a current-pose OGC test: no swept/CCD row is
// enabled for the visual bodies.
static const PxVec3 SCENE_CPU_VISUAL_DYNAMIC_BOX_INITIAL_LINEAR_VELOCITY(
	0.0f, 0.0f, 0.0f);
static const PxVec3 SCENE_CPU_VISUAL_CONE_INITIAL_LINEAR_VELOCITY(
	0.045f, 0.0f, 0.020f);
static const PxVec3 SCENE_CPU_VISUAL_TILTED_CUBE_INITIAL_LINEAR_VELOCITY(
	0.0f, 0.0f, 0.0f);
static const PxVec3 SCENE_CPU_VISUAL_FOLLOWER_CUBE_INITIAL_LINEAR_VELOCITY(
	0.0f, 0.0f, 0.0f);
// A deliberately small, isolated regression for the mixed OGC path.  It has
// no ground, pedestal, attachments, locks, CCD or visual-only substeps: the
// only possible collision owner is the two soft-volume / free-rigid-box
// manifold.  The jaws retain the public showcase collision skin and material
// so this is a solver test, rather than a low-resolution surrogate.
static const PxVec3 SCENE_CPU_OGC_SANDWICH_BOX_CENTER(0.0f, 25.0f, 0.0f);
static const PxVec3 SCENE_CPU_OGC_SANDWICH_BOX_HALF_EXTENTS(
	0.50f, 0.35f, 0.70f);
static const PxReal SCENE_CPU_OGC_SANDWICH_JAW_BOX_GAP = 0.030f;
static const PxReal SCENE_CPU_OGC_SANDWICH_JAW_CENTER_DISTANCE =
	0.70f + 0.35f + SCENE_CPU_OGC_SANDWICH_JAW_BOX_GAP;
static const PxVec3 SCENE_CPU_OGC_SANDWICH_UPPER_JAW_CENTER(
	SCENE_CPU_OGC_SANDWICH_BOX_CENTER +
		PxVec3(0.0f, SCENE_CPU_OGC_SANDWICH_JAW_CENTER_DISTANCE, 0.0f));
static const PxVec3 SCENE_CPU_OGC_SANDWICH_LOWER_JAW_CENTER(
	SCENE_CPU_OGC_SANDWICH_BOX_CENTER -
		PxVec3(0.0f, SCENE_CPU_OGC_SANDWICH_JAW_CENTER_DISTANCE, 0.0f));
static const PxVec3 SCENE_CPU_OGC_SANDWICH_UPPER_JAW_VELOCITY(
	0.0f, -0.20f, 0.0f);
static const PxVec3 SCENE_CPU_OGC_SANDWICH_LOWER_JAW_VELOCITY(
	0.0f, 0.20f, 0.0f);
// The regression owns a short, explicit external compression window.  The
// dynamic box remains a free six-DOF body; only the two soft jaws receive the
// prescribed uniform approach velocity.  Without this drive the old fixture
// contained just two initial-velocity contact frames, so it could pass after
// the bodies separated and never exercise a sustained mixed OGC manifold.
static const PxU32 SCENE_CPU_OGC_SANDWICH_PRESSURE_DRIVE_FRAMES = 36u;
// Use an inertia comparable to the two volumetric jaws.  The box stays fully
// dynamic (all six DOF free); this merely makes the fixture a compression
// test instead of a one-sided light-projectile rebound test.
static const PxReal SCENE_CPU_OGC_SANDWICH_BOX_MASS = 500.0f;
static RotationMonitor gSceneCpuVisualRotationMonitor;
static RotationMonitor gSceneCpuVisualPrimaryCubeRotationMonitor;
static RotationMonitor gSceneCpuVisualSphereRotationMonitor;
static const RotationMetrics& gSceneCpuVisualRotationMetrics =
	gSceneCpuVisualRotationMonitor.getMetrics();
static const RotationMetrics& gSceneCpuVisualPrimaryCubeRotationMetrics =
	gSceneCpuVisualPrimaryCubeRotationMonitor.getMetrics();
static const RotationMetrics& gSceneCpuVisualSphereRotationMetrics =
	gSceneCpuVisualSphereRotationMonitor.getMetrics();

static VisualInteractionMonitor gSceneCpuVisualInteractionMonitor;
static const VisualInteractionMetrics& gSceneCpuVisualInteractionMetrics =
	gSceneCpuVisualInteractionMonitor.getMetrics();

static OgcSandwichMonitor gSceneCpuOgcSandwichMonitor;
static SoftContactPhaseMonitor gSceneCpuSoftContactPhaseMonitor;
static const SoftContactPhaseMetrics& gSceneCpuSoftContactPhaseMetrics =
	gSceneCpuSoftContactPhaseMonitor.getMetrics();
// Dedicated, gravity-free soft/soft rotation fixture.  Unlike the visual
// showcase, this owns no plane, static actor or rigid actor: any target
// angular momentum observed here must originate in the off-centre soft/soft
// boundary contact below.
static const PxReal SCENE_CPU_SOFT_SOFT_TORQUE_MIN_ANGULAR_SPEED =
	1.0e-3f;
static const PxReal SCENE_CPU_SOFT_SOFT_TORQUE_MIN_ANGULAR_MOMENTUM =
	1.0e-3f;
// The rotating capsule begins and ends well outside the unit collision
// tetrahedron translated to (-2, 0, 0), but its long axis crosses the corner
// (-2, 0, 0) near the middle of the 200 degree sweep.  The old (-2.9, 0, 0)
// center left only a few millimetres at the start pose, so its CCD-off actor
// was legitimately admitted by current-pose OGC and was not a swept negative
// control.
static const PxVec3 SCENE_CPU_DYNAMIC_ROTATING_CAPSULE_SWEEP_CENTER(
	-2.35f, -0.95f, 0.0f);
static const PxVec3 SCENE_CPU_DYNAMIC_ROTATING_CAPSULE_CONTROL_CENTER(
	0.65f, -0.95f, 0.0f);
// The embedded-tet ground fixture begins with pure +X translation.  Its
// expected rolling axis is groundNormal x travelDirection = -Z, so any
// measured angular state is contact-generated rather than injected at setup.
static const PxReal SCENE_CPU_GROUND_EMBEDDED_TET_PROBE_LAUNCH_SPEED = 1.0f;

// The two dedicated soft/soft rotation cases use Scene collision statistics as
// part of their correctness contract.  Enable that test-only accounting before
// the Scene exists so direct invocation has the same semantics as the runner.
static bool enableSceneCpuFixtureCollisionTelemetry()
{
#if defined(_WIN32)
	return _putenv_s("PHYSX_AVBD_COLLISION_TELEMETRY", "1") == 0;
#else
	return setenv("PHYSX_AVBD_COLLISION_TELEMETRY", "1", 1) == 0;
#endif
}

static SoftSoftTorqueMonitor gSceneCpuSoftSoftTorqueMonitor;
static const SceneCpuSoftSoftTorqueMetrics& gSceneCpuSoftSoftTorqueMetrics =
	gSceneCpuSoftSoftTorqueMonitor.getMetrics();

// A deliberately tiny public-Volume ground fixture for the first ground-row
// coupling experiment.  The collision tetrahedron is strictly embedded in one
// larger simulation tetrahedron, so every contact point must expand through
// all four simulation vertices rather than coinciding with a simulation-mesh
// boundary vertex.
static SceneCpuGroundEmbeddedTetProbeMetrics
	gSceneCpuGroundEmbeddedTetProbeMetrics;

static VolumeHealthMonitor gSceneCpuVolumeHealthMonitor;
static PxDeformableAttachment*   gSceneCpuWorldAttachment = NULL;
static PxDeformableAttachment*   gSceneCpuRigidAttachment = NULL;
static PxDeformableElementFilter* gSceneCpuElementFilter = NULL;
static PxDeformableVolumeMesh*   gSceneCpuVolumeMesh = NULL;
static PxDeformableVolumeMaterial*
								   gSceneCpuVolumeMaterial = NULL;
static PxRigidStatic*               gSceneCpuStaticActor = NULL;
static PxRigidDynamic*              gSceneCpuDynamicActor = NULL;
static PxRigidDynamic*              gSceneCpuSecondDynamicActor = NULL;
static PxConvexMesh*                gSceneCpuRigidConvexMesh = NULL;
static PxTriangleMesh*              gSceneCpuRigidTriangleMesh = NULL;
static PxHeightField*               gSceneCpuRigidHeightField = NULL;
static PxArticulationReducedCoordinate*
								   gSceneCpuAttachmentArticulation = NULL;
static PxArticulationLink*           gSceneCpuAttachmentRoot = NULL;
static PxArticulationLink*           gSceneCpuAttachmentLink = NULL;
static PxRigidBody*                  gSceneCpuAttachmentBody = NULL;
static PxReal                     gSceneCpuVolumeInitialCentroidY = 0.0f;
static PxReal                     gSceneCpuSecondVolumeInitialCentroidY = 0.0f;
static PxReal                     gSceneCpuDepenetrationInitialMinY =
								   0.0f;
static PxReal                     gSceneCpuDepenetrationControlInitialMinY =
								   0.0f;
static PxVec3                     gSceneCpuMotionInitialCentroid(0.0f);
static PxReal                     gSceneCpuDynamicInitialY = 0.0f;
static PxReal                     gSceneCpuSecondDynamicInitialY = 0.0f;
static PxReal                     gSceneCpuKinematicCommandY = 0.0f;
static PxReal                     gSceneCpuKinematicSoftBaselineY = 0.0f;
static PxVec3                     gSceneCpuMovingSpherePositiveInitial(
	0.0f);
static PxVec3                     gSceneCpuMovingSphereNegativeInitial(
	0.0f);
static PxVec3                     gSceneCpuSphereReversePositiveInitial(
	0.0f);
static PxVec3                     gSceneCpuSphereReverseNegativeInitial(
	0.0f);
static PxReal                     gSceneCpuFirstSleepCentroidY = 0.0f;
static PxReal                     gSceneCpuSecondSleepCentroidY = 0.0f;
static PxReal                     gSceneCpuVelocityWakeCentroidY = 0.0f;
static PxReal                     gSceneCpuRigidWakeCentroidY = 0.0f;
static PxReal                     gSceneCpuSoftChurnCentroidY = 0.0f;
static bool                       gSceneCpuSoftChurnMovePending = false;
static PxVec3                     gSceneCpuBufferPinTarget(0.0f);
static PxVec3                     gSceneCpuBufferDynamicBaseline(0.0f);
static PxVec3                     gSceneCpuBufferRestoredBaseline(0.0f);
static PxReal                     gSceneCpuBufferOriginalInvMass = 0.0f;
static PxReal                     gSceneCpuMultiPrimaryRemovedCentroidY = 0.0f;
static PxReal                     gSceneCpuMultiSecondaryAtReleaseCentroidY = 0.0f;
static PxVec3                     gSceneCpuSoftSoftTargetBaseline(0.0f);
static PxVec3                     gSceneCpuWorldPinTarget(0.0f);
static PxArray<PxVec4>            gSceneCpuVolumeKinematicTargets;
static PxArray<PxVec3>            gSceneCpuVolumeKinematicInitial;
static PxArray<PxVec3>            gSceneCpuSphereReverseSweptInitialPositions;
static PxArray<PxVec3>
	gSceneCpuDeformingReverseSweptFreeEndPositions;
static PxArray<PxVec3>
	gSceneCpuCapsuleRotationalSweptInitialPositions;
static PxU32                      gSceneCpuVolumePartialProbe =
	PX_MAX_U32;
static PxVec3                     gSceneCpuVolumePartialActivationStart(
	0.0f);
static bool                       gSceneCpuElementAttachment = false;
static PxU32                      gSceneCpuAttachmentVertices[4] =
	{0, 0, 0, 0};
static PxVec4                     gSceneCpuAttachmentBarycentric(
	0.1f, 0.2f, 0.3f, 0.4f);
static PxVec3                     gSceneCpuRigidAttachmentInitialPosition(
	0.0f);
static PxVec3                     gSceneCpuRigidAttachmentLocalOffset(
	0.0f);
static PxReal                     gSceneCpuKinematicAttachmentProgress = 0.0f;
static PxVec3                     gSceneCpuKinematicAttachmentSoftBaseline(
	0.0f);
static PxTransform                gSceneCpuKinematicAttachmentCommand(
	PxIdentity);
static PxTransform                gSceneCpuArticulationRootInitialPose(
	PxIdentity);
static PxTransform                gSceneCpuArticulationChildInitialPose(
	PxIdentity);
static bool                       gSceneCpuPartialFilterSelectedPositiveX =
	false;
static PxArray<Snippets::AvbdTetrahedronSkinningBinding>
								   gVolumeSkinningBindings;
static PxArray<PxU32>              gVolumeSkinningTriangles;
static PxArray<PxVec3>             gVolumeSkinningPositions;
static PxArray<PxVec3>             gVolumeSkinningNormals;
static PxArray<PxVec3>             gVolumeSkinningInitialPositions;

static VolumeSkinningMetrics gVolumeSkinningMetrics;

static ReverseFeatureMetrics gSphereReverseFeatureMetrics;

static ReverseSweptMetrics gSphereReverseSweptMetrics;
static DeformingReverseSweptMetrics gDeformingVolumeReverseSweptMetrics;
static RotationalSweepMetrics gCapsuleRotationalSweepMetrics;


static DeformableVolumeMetrics gMetrics;
static PxArray<PxVec3> gInitialCentroids;

static DeformableVolumePerformanceMetrics gPerformance;
static AvbdSoftCollisionStats gFrameCollisionStats;
static PxU32 gProfileWarmupFrames = 0;
// Headless-only P1 corpus knob. The normal snippet never changes its legacy
// two-triangle mesh unless this validated environment setting is explicit.
static PxU32 gRigidTriangleMeshGridDimension = 1;

static void capturePerformanceTopology()
{
	gPerformance.topologySoftBodies = gSoftBodies.size();
	gPerformance.topologySoftParticles = gParticles.size();
	gPerformance.topologyRigidBoxes = gRigidBoxes.size();
	gPerformance.topologyTriElements = 0;
	gPerformance.topologyTetElements = 0;
	gPerformance.topologyBendElements = 0;
	gPerformance.topologySurfaceTriangles = 0;
	gPerformance.topologySurfaceVertices = 0;
	gPerformance.topologySurfaceEdges = 0;
	gPerformance.topologyRigidTriangleMeshTriangles = 0;
	for(PxU32 bodyIndex = 0; bodyIndex < gSoftBodies.size(); ++bodyIndex)
	{
		const AvbdSoftBody& body = gSoftBodies[bodyIndex];
		gPerformance.topologyTriElements += body.compiled.triElements.size();
		gPerformance.topologyTetElements += body.compiled.tetElements.size();
		gPerformance.topologyBendElements += body.compiled.bendElements.size();
		gPerformance.topologySurfaceTriangles +=
			body.compiled.surfaceTriangles.size() / 3;
		gPerformance.topologySurfaceVertices +=
			body.compiled.surfaceVertices.size();
		gPerformance.topologySurfaceEdges +=
			body.compiled.surfaceEdges.size();
	}
}

// A Scene CPU AVBD actor owns its mesh state rather than the snippet's
// component-fallback arrays.  Capture it from the actor API so schema=2
// remains a truthful workload description for Scene cases too.  Boundary
// edge count is intentionally left zero: PxDeformableVolume exposes the
// tetrahedral meshes but not a precompiled boundary-edge table.
static void captureScenePerformanceTopology()
{
	gPerformance.topologySoftBodies = 0;
	gPerformance.topologySoftParticles = 0;
	gPerformance.topologyTriElements = 0;
	gPerformance.topologyTetElements = 0;
	gPerformance.topologyBendElements = 0;
	gPerformance.topologySurfaceTriangles =
		gMetrics.surfaceTriangles;
	gPerformance.topologySurfaceVertices = 0;
	gPerformance.topologySurfaceEdges = 0;
	gPerformance.topologyRigidBoxes = gMetrics.rigidBoxes;
	gPerformance.topologyRigidTriangleMeshTriangles =
		gSceneCpuRigidTriangleMesh
			? gSceneCpuRigidTriangleMesh->getNbTriangles()
			: 0;
	auto captureVolume = [](const PxDeformableVolume* volume)
	{
		if(!volume)
			return;
		++gPerformance.topologySoftBodies;
		const PxTetrahedronMesh* simulationMesh =
			volume->getSimulationMesh();
		if(simulationMesh)
		{
			gPerformance.topologySoftParticles +=
				simulationMesh->getNbVertices();
			gPerformance.topologyTetElements +=
				simulationMesh->getNbTetrahedrons();
		}
		const PxTetrahedronMesh* collisionMesh =
			volume->getCollisionMesh();
		if(collisionMesh)
			gPerformance.topologySurfaceVertices +=
				collisionMesh->getNbVertices();
	};
	captureVolume(gSceneCpuVolume);
	captureVolume(gSceneCpuSecondVolume);
	for(PxU32 volumeId = 0;
		volumeId < gSceneCpuTaskGraphExtraVolumes.size(); ++volumeId)
		captureVolume(gSceneCpuTaskGraphExtraVolumes[volumeId]);
}

static void accumulateStepStats(
	AvbdSoftBodyStepStats& total, const AvbdSoftBodyStepStats& frame)
{
	total.predictionMs += frame.predictionMs;
	total.contactIndexMs += frame.contactIndexMs;
	total.bodyPrecomputeMs += frame.bodyPrecomputeMs;
	total.bodySolveMs += frame.bodySolveMs;
	total.particleSolveMs += frame.particleSolveMs;
	total.projectionMs += frame.projectionMs;
	total.dualMs += frame.dualMs;
	total.redetectMs += frame.redetectMs;
	total.velocityMs += frame.velocityMs;
	total.frictionMs += frame.frictionMs;
	total.requestedOuterIterations += frame.requestedOuterIterations;
	total.requestedInnerIterations += frame.requestedInnerIterations;
	total.executedOuterIterations += frame.executedOuterIterations;
	total.executedInnerIterations += frame.executedInnerIterations;
	total.particleSweeps += frame.particleSweeps;
	total.groundTetPatchGroundPositionAlRows +=
		frame.groundTetPatchGroundPositionAlRows;
	total.groundTetPatchFourSupportRows +=
		frame.groundTetPatchFourSupportRows;
	total.groundTetPatchSingleTetRows +=
		frame.groundTetPatchSingleTetRows;
	total.groundTetPatchActiveRows +=
		frame.groundTetPatchActiveRows;
	total.worldStaticVelocityTangentOwnerRows +=
		frame.worldStaticVelocityTangentOwnerRows;
	total.worldStaticVelocityTangentAppliedRows +=
		frame.worldStaticVelocityTangentAppliedRows;
	total.trustRegionLimitedParticleSteps +=
		frame.trustRegionLimitedParticleSteps;
	total.positiveJLimitedParticleSteps +=
		frame.positiveJLimitedParticleSteps;
	total.positiveJRejectedParticleSteps +=
		frame.positiveJRejectedParticleSteps;
	total.nonFiniteRejectedParticleSteps +=
		frame.nonFiniteRejectedParticleSteps;
	total.tetLinearizationCacheFallbackParticleSteps +=
		frame.tetLinearizationCacheFallbackParticleSteps;
	total.legacyAppliedConvergedOuterIterations +=
		frame.legacyAppliedConvergedOuterIterations;
	total.residualConvergedOuterIterations +=
		frame.residualConvergedOuterIterations;
	total.unsafeAppliedConvergenceCandidates +=
		frame.unsafeAppliedConvergenceCandidates;
	total.budgetExhaustedOuterIterations +=
		frame.budgetExhaustedOuterIterations;
	total.shadowResidual1e5ConvergedOuterIterations +=
		frame.shadowResidual1e5ConvergedOuterIterations;
	total.shadowResidual1e5SavedInnerIterations +=
		frame.shadowResidual1e5SavedInnerIterations;
	total.shadowResidual1e4ConvergedOuterIterations +=
		frame.shadowResidual1e4ConvergedOuterIterations;
	total.shadowResidual1e4SavedInnerIterations +=
		frame.shadowResidual1e4SavedInnerIterations;
	total.workspaceGrowthEvents += frame.workspaceGrowthEvents;
	total.workspaceGrowthBytes += frame.workspaceGrowthBytes;
	total.contactWorkspaceGrowthEvents +=
		frame.contactWorkspaceGrowthEvents;
	total.contactWorkspaceGrowthBytes +=
		frame.contactWorkspaceGrowthBytes;
	total.contactOutputGrowthEvents +=
		frame.contactOutputGrowthEvents;
	total.contactOutputGrowthBytes +=
		frame.contactOutputGrowthBytes;
	total.finalMaxLocalSolveDisplacement =
		frame.finalMaxLocalSolveDisplacement;
	total.finalMaxAppliedDisplacement =
		frame.finalMaxAppliedDisplacement;
	total.finalMaxDisplacement = frame.finalMaxDisplacement;
	total.particlePrimalColorCount = PxMax(
		total.particlePrimalColorCount,
		frame.particlePrimalColorCount);
	total.particlePrimalDynamicAccessGroupCount = PxMax(
		total.particlePrimalDynamicAccessGroupCount,
		frame.particlePrimalDynamicAccessGroupCount);
	total.particlePrimalColoredSerialSweeps +=
		frame.particlePrimalColoredSerialSweeps;
	total.particlePrimalColoredSerialFallbackSweeps +=
		frame.particlePrimalColoredSerialFallbackSweeps;
	total.particlePrimalCensusDynamicParticleSolves +=
		frame.particlePrimalCensusDynamicParticleSolves;
	total.particlePrimalCensusTriangleEvaluations +=
		frame.particlePrimalCensusTriangleEvaluations;
	total.particlePrimalCensusCorotationalTetEvaluations +=
		frame.particlePrimalCensusCorotationalTetEvaluations;
	total.particlePrimalCensusNeoHookeanTetEvaluations +=
		frame.particlePrimalCensusNeoHookeanTetEvaluations;
	total.particlePrimalCensusBendingEvaluations +=
		frame.particlePrimalCensusBendingEvaluations;
	total.particlePrimalCensusContactEvaluations +=
		frame.particlePrimalCensusContactEvaluations;
	total.particlePrimalCensusTetPacket8FullPackets +=
		frame.particlePrimalCensusTetPacket8FullPackets;
	total.particlePrimalCensusTetPacket8TailLanes +=
		frame.particlePrimalCensusTetPacket8TailLanes;
	// P8.2 records immutable topology metadata, not work executed per frame.
	// Keep its profile value as the per-step peak instead of multiplying one
	// compiled program by the number of profiled frames.
	total.particlePrimalTetPacketIrBodies = PxMax(
		total.particlePrimalTetPacketIrBodies,
		frame.particlePrimalTetPacketIrBodies);
	total.particlePrimalTetPacketIrPackets = PxMax(
		total.particlePrimalTetPacketIrPackets,
		frame.particlePrimalTetPacketIrPackets);
	total.particlePrimalTetPacketIrActiveLanes = PxMax(
		total.particlePrimalTetPacketIrActiveLanes,
		frame.particlePrimalTetPacketIrActiveLanes);
	total.particlePrimalTetPacketIrTailLanes = PxMax(
		total.particlePrimalTetPacketIrTailLanes,
		frame.particlePrimalTetPacketIrTailLanes);
	total.particlePrimalTetPacketIrActiveTailLanes = PxMax(
		total.particlePrimalTetPacketIrActiveTailLanes,
		frame.particlePrimalTetPacketIrActiveTailLanes);
	total.particlePrimalTetPacketIrInvalidBodies = PxMax(
		total.particlePrimalTetPacketIrInvalidBodies,
		frame.particlePrimalTetPacketIrInvalidBodies);
}

// PxScene reports step-local CPU AVBD telemetry.  The Scene branch must use
// that authority rather than gFrameCollisionStats, which belongs exclusively
// to the standalone component-fallback path below.
static void accumulateScenePerformanceStatistics(
	const PxSimulationStatistics& sceneStatistics)
{
	const PxU32 componentFallbackSteps =
		sceneStatistics.avbdCpuSoftBodyComponentFallbackSteps;
	const PxU32 nativeIslandSteps =
		sceneStatistics.avbdCpuSoftBodyNativeIslandSteps;
	const PxU64 causalLayerTasks =
		sceneStatistics.avbdCpuTaskGraphSubmittedCausalLayerTasks;
	AvbdSoftBodyStepStats& stages = gPerformance.solverStages;
	gPerformance.componentFallbackSteps +=
		componentFallbackSteps;
	gPerformance.nativeIslandSteps +=
		nativeIslandSteps;
	if(nativeIslandSteps && componentFallbackSteps)
		gPerformance.nativeIslandComponentFallbackOverlapFrames++;
	if(nativeIslandSteps && causalLayerTasks)
		gPerformance.nativeIslandCausalLayerTaskOverlapFrames++;
	gPerformance.cpuIsaRequested = sceneStatistics.avbdCpuIsaRequested;
	gPerformance.cpuIsaSelected = sceneStatistics.avbdCpuIsaSelected;
	gPerformance.cpuIsaCompiledBackendMask =
		sceneStatistics.avbdCpuIsaCompiledBackendMask;
	gPerformance.cpuIsaCapabilityMask = sceneStatistics.avbdCpuIsaCapabilityMask;
	gPerformance.cpuIsaForceModeRejected =
		sceneStatistics.avbdCpuIsaForceModeRejected;
	gPerformance.cpuIsaKernelSelfTestPassed =
		sceneStatistics.avbdCpuIsaKernelSelfTestPassed;
	gPerformance.cpuIsaFmaUsed = sceneStatistics.avbdCpuIsaFmaUsed;
	gPerformance.cpuIsaKernelSelfTestValue =
		sceneStatistics.avbdCpuIsaKernelSelfTestValue;
	stages.particlePrimalCensusDynamicParticleSolves +=
		sceneStatistics.
			avbdCpuSoftBodyParticlePrimalCensusDynamicParticleSolves;
	stages.particlePrimalCensusTriangleEvaluations +=
		sceneStatistics.avbdCpuSoftBodyParticlePrimalCensusTriangleEvaluations;
	stages.particlePrimalCensusCorotationalTetEvaluations +=
		sceneStatistics.
			avbdCpuSoftBodyParticlePrimalCensusCorotationalTetEvaluations;
	stages.particlePrimalCensusNeoHookeanTetEvaluations +=
		sceneStatistics.
			avbdCpuSoftBodyParticlePrimalCensusNeoHookeanTetEvaluations;
	stages.particlePrimalCensusBendingEvaluations +=
		sceneStatistics.avbdCpuSoftBodyParticlePrimalCensusBendingEvaluations;
	stages.particlePrimalCensusContactEvaluations +=
		sceneStatistics.avbdCpuSoftBodyParticlePrimalCensusContactEvaluations;
	stages.particlePrimalCensusTetPacket8FullPackets +=
		sceneStatistics.
			avbdCpuSoftBodyParticlePrimalCensusTetPacket8FullPackets;
	stages.particlePrimalCensusTetPacket8TailLanes +=
		sceneStatistics.
			avbdCpuSoftBodyParticlePrimalCensusTetPacket8TailLanes;
	stages.particlePrimalTetPacketIrBodies = PxMax(
		stages.particlePrimalTetPacketIrBodies,
		sceneStatistics.avbdCpuSoftBodyParticlePrimalTetPacketIrBodies);
	stages.particlePrimalTetPacketIrPackets = PxMax(
		stages.particlePrimalTetPacketIrPackets,
		sceneStatistics.avbdCpuSoftBodyParticlePrimalTetPacketIrPackets);
	stages.particlePrimalTetPacketIrActiveLanes = PxMax(
		stages.particlePrimalTetPacketIrActiveLanes,
		sceneStatistics.avbdCpuSoftBodyParticlePrimalTetPacketIrActiveLanes);
	stages.particlePrimalTetPacketIrTailLanes = PxMax(
		stages.particlePrimalTetPacketIrTailLanes,
		sceneStatistics.avbdCpuSoftBodyParticlePrimalTetPacketIrTailLanes);
	stages.particlePrimalTetPacketIrActiveTailLanes = PxMax(
		stages.particlePrimalTetPacketIrActiveTailLanes,
		sceneStatistics.
			avbdCpuSoftBodyParticlePrimalTetPacketIrActiveTailLanes);
	stages.particlePrimalTetPacketIrInvalidBodies = PxMax(
		stages.particlePrimalTetPacketIrInvalidBodies,
		sceneStatistics.avbdCpuSoftBodyParticlePrimalTetPacketIrInvalidBodies);
	gPerformance.taskGraphRequestedDispatcherWorkers = PxMax(
		gPerformance.taskGraphRequestedDispatcherWorkers,
		sceneStatistics.avbdCpuTaskGraphRequestedDispatcherWorkers);
	gPerformance.taskGraphPeakActiveSolveTasks = PxMax(
		gPerformance.taskGraphPeakActiveSolveTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActiveSolveTasks);
	gPerformance.taskGraphSubmittedSolveTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedSolveTasks;
	gPerformance.taskGraphCompletedSolveTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedSolveTasks;
	gPerformance.taskGraphBarrierTasks +=
		sceneStatistics.avbdCpuTaskGraphBarrierTasks;
	gPerformance.taskGraphSerialSolveTasks +=
		sceneStatistics.avbdCpuTaskGraphSerialSolveTasks;
	gPerformance.taskGraphSubmittedPredictionTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedPredictionTasks;
	gPerformance.taskGraphCompletedPredictionTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedPredictionTasks;
	gPerformance.taskGraphPeakActivePredictionTasks = PxMax(
		gPerformance.taskGraphPeakActivePredictionTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActivePredictionTasks);
	gPerformance.taskGraphSerialPredictionStages +=
		sceneStatistics.avbdCpuTaskGraphSerialPredictionStages;
	gPerformance.taskGraphSubmittedWriteBackTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedWriteBackTasks;
	gPerformance.taskGraphCompletedWriteBackTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedWriteBackTasks;
	gPerformance.taskGraphPeakActiveWriteBackTasks = PxMax(
		gPerformance.taskGraphPeakActiveWriteBackTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActiveWriteBackTasks);
	gPerformance.taskGraphSerialWriteBackStages +=
		sceneStatistics.avbdCpuTaskGraphSerialWriteBackStages;
	gPerformance.taskGraphSubmittedCausalLayerTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedCausalLayerTasks;
	gPerformance.taskGraphCompletedCausalLayerTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedCausalLayerTasks;
	gPerformance.taskGraphPeakActiveCausalLayerTasks = PxMax(
		gPerformance.taskGraphPeakActiveCausalLayerTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActiveCausalLayerTasks);
	gPerformance.taskGraphCausalLayerFanIns +=
		sceneStatistics.avbdCpuTaskGraphCausalLayerFanIns;
	gPerformance.taskGraphSerialCausalLayerFallbacks +=
		sceneStatistics.avbdCpuTaskGraphSerialCausalLayerFallbacks;
	gPerformance.taskGraphMaxCausalLayerOccupancy = PxMax(
		gPerformance.taskGraphMaxCausalLayerOccupancy,
		sceneStatistics.avbdCpuTaskGraphMaxCausalLayerOccupancy);
	gPerformance.taskGraphTotalCausalLayerOccupancy +=
		sceneStatistics.avbdCpuTaskGraphTotalCausalLayerOccupancy;
	gPerformance.taskGraphSubmittedWorldPlaneContactTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedWorldPlaneContactTasks;
	gPerformance.taskGraphCompletedWorldPlaneContactTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedWorldPlaneContactTasks;
	gPerformance.taskGraphPeakActiveWorldPlaneContactTasks = PxMax(
		gPerformance.taskGraphPeakActiveWorldPlaneContactTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActiveWorldPlaneContactTasks);
	gPerformance.taskGraphWorldPlaneContactFanIns +=
		sceneStatistics.avbdCpuTaskGraphWorldPlaneContactFanIns;
	gPerformance.taskGraphSerialWorldPlaneContactFallbacks +=
		sceneStatistics.avbdCpuTaskGraphSerialWorldPlaneContactFallbacks;
	gPerformance.taskGraphSubmittedRigidBoxSdfContactTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedRigidBoxSdfContactTasks;
	gPerformance.taskGraphCompletedRigidBoxSdfContactTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedRigidBoxSdfContactTasks;
	gPerformance.taskGraphPeakActiveRigidBoxSdfContactTasks = PxMax(
		gPerformance.taskGraphPeakActiveRigidBoxSdfContactTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActiveRigidBoxSdfContactTasks);
	gPerformance.taskGraphRigidBoxSdfContactFanIns +=
		sceneStatistics.avbdCpuTaskGraphRigidBoxSdfContactFanIns;
	gPerformance.taskGraphSerialRigidBoxSdfContactFallbacks +=
		sceneStatistics.avbdCpuTaskGraphSerialRigidBoxSdfContactFallbacks;
	gPerformance.taskGraphSubmittedRigidSphereSdfContactTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedRigidSphereSdfContactTasks;
	gPerformance.taskGraphCompletedRigidSphereSdfContactTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedRigidSphereSdfContactTasks;
	gPerformance.taskGraphPeakActiveRigidSphereSdfContactTasks = PxMax(
		gPerformance.taskGraphPeakActiveRigidSphereSdfContactTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActiveRigidSphereSdfContactTasks);
	gPerformance.taskGraphRigidSphereSdfContactFanIns +=
		sceneStatistics.avbdCpuTaskGraphRigidSphereSdfContactFanIns;
	gPerformance.taskGraphSerialRigidSphereSdfContactFallbacks +=
		sceneStatistics.avbdCpuTaskGraphSerialRigidSphereSdfContactFallbacks;
	gPerformance.taskGraphSubmittedRigidCapsuleSdfContactTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedRigidCapsuleSdfContactTasks;
	gPerformance.taskGraphCompletedRigidCapsuleSdfContactTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedRigidCapsuleSdfContactTasks;
	gPerformance.taskGraphPeakActiveRigidCapsuleSdfContactTasks = PxMax(
		gPerformance.taskGraphPeakActiveRigidCapsuleSdfContactTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActiveRigidCapsuleSdfContactTasks);
	gPerformance.taskGraphRigidCapsuleSdfContactFanIns +=
		sceneStatistics.avbdCpuTaskGraphRigidCapsuleSdfContactFanIns;
	gPerformance.taskGraphSerialRigidCapsuleSdfContactFallbacks +=
		sceneStatistics.avbdCpuTaskGraphSerialRigidCapsuleSdfContactFallbacks;
	gPerformance.taskGraphSubmittedRigidConvexSdfContactTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedRigidConvexSdfContactTasks;
	gPerformance.taskGraphCompletedRigidConvexSdfContactTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedRigidConvexSdfContactTasks;
	gPerformance.taskGraphPeakActiveRigidConvexSdfContactTasks = PxMax(
		gPerformance.taskGraphPeakActiveRigidConvexSdfContactTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActiveRigidConvexSdfContactTasks);
	gPerformance.taskGraphRigidConvexSdfContactFanIns +=
		sceneStatistics.avbdCpuTaskGraphRigidConvexSdfContactFanIns;
	gPerformance.taskGraphSerialRigidConvexSdfContactFallbacks +=
		sceneStatistics.avbdCpuTaskGraphSerialRigidConvexSdfContactFallbacks;
	gPerformance.taskGraphSubmittedRigidTriangleSurfaceContactTasks +=
		sceneStatistics.avbdCpuTaskGraphSubmittedRigidTriangleSurfaceContactTasks;
	gPerformance.taskGraphCompletedRigidTriangleSurfaceContactTasks +=
		sceneStatistics.avbdCpuTaskGraphCompletedRigidTriangleSurfaceContactTasks;
	gPerformance.taskGraphPeakActiveRigidTriangleSurfaceContactTasks = PxMax(
		gPerformance.taskGraphPeakActiveRigidTriangleSurfaceContactTasks,
		sceneStatistics.avbdCpuTaskGraphPeakActiveRigidTriangleSurfaceContactTasks);
	gPerformance.taskGraphRigidTriangleSurfaceContactFanIns +=
		sceneStatistics.avbdCpuTaskGraphRigidTriangleSurfaceContactFanIns;
	gPerformance.taskGraphSerialRigidTriangleSurfaceContactFallbacks +=
		sceneStatistics.avbdCpuTaskGraphSerialRigidTriangleSurfaceContactFallbacks;
	gPerformance.taskGraphPureSoftEligibleIslands +=
		sceneStatistics.avbdCpuTaskGraphPureSoftEligibleIslands;
	gPerformance.taskGraphPureSoftEligibleParticles +=
		sceneStatistics.avbdCpuTaskGraphPureSoftEligibleParticles;
	stages.workspaceGrowthEvents +=
		sceneStatistics.avbdCpuSoftBodyWorkspaceGrowthEvents;
	stages.workspaceGrowthBytes +=
		sceneStatistics.avbdCpuSoftBodyWorkspaceGrowthBytes;
	stages.contactWorkspaceGrowthEvents +=
		sceneStatistics.avbdCpuSoftBodyContactWorkspaceGrowthEvents;
	stages.contactWorkspaceGrowthBytes +=
		sceneStatistics.avbdCpuSoftBodyContactWorkspaceGrowthBytes;
	stages.contactSweepScratchGrowthEvents +=
		sceneStatistics.avbdCpuSoftBodyContactSweepScratchGrowthEvents;
	stages.contactSweepScratchGrowthBytes +=
		sceneStatistics.avbdCpuSoftBodyContactSweepScratchGrowthBytes;
	stages.contactOutputGrowthEvents +=
		sceneStatistics.avbdCpuSoftBodyContactOutputGrowthEvents;
	stages.contactOutputGrowthBytes +=
		sceneStatistics.avbdCpuSoftBodyContactOutputGrowthBytes;
	stages.peakContactOutputCount = PxMax(
		stages.peakContactOutputCount,
		sceneStatistics.avbdCpuSoftBodyPeakContactOutputCount);
	stages.peakContactOutputCapacity = PxMax(
		stages.peakContactOutputCapacity,
		sceneStatistics.avbdCpuSoftBodyPeakContactOutputCapacity);
	stages.peakContactIncidenceCount = PxMax(
		stages.peakContactIncidenceCount,
		sceneStatistics.avbdCpuSoftBodyPeakContactIncidenceCount);
	stages.peakContactIncidenceCapacity = PxMax(
		stages.peakContactIncidenceCapacity,
		sceneStatistics.avbdCpuSoftBodyPeakContactIncidenceCapacity);
	stages.peakStateTransferContactCount = PxMax(
		stages.peakStateTransferContactCount,
		sceneStatistics.avbdCpuSoftBodyPeakStateTransferContactCount);
	stages.peakStateTransferContactCapacity = PxMax(
		stages.peakStateTransferContactCapacity,
		sceneStatistics.avbdCpuSoftBodyPeakStateTransferContactCapacity);
	stages.peakStateTransferUsedCapacity = PxMax(
		stages.peakStateTransferUsedCapacity,
		sceneStatistics.avbdCpuSoftBodyPeakStateTransferUsedCapacity);
	stages.particlePrimalColorCount = PxMax(
		stages.particlePrimalColorCount,
		sceneStatistics.avbdCpuSoftBodyParticlePrimalColorCount);
	stages.particlePrimalDynamicAccessGroupCount = PxMax(
		stages.particlePrimalDynamicAccessGroupCount,
		sceneStatistics.
			avbdCpuSoftBodyParticlePrimalDynamicAccessGroupCount);
	stages.particlePrimalColoredSerialSweeps +=
		sceneStatistics.avbdCpuSoftBodyParticlePrimalColoredSerialSweeps;
	stages.particlePrimalColoredSerialFallbackSweeps +=
		sceneStatistics.
			avbdCpuSoftBodyParticlePrimalColoredSerialFallbackSweeps;
	stages.groundTetPatchGroundPositionAlRows +=
		sceneStatistics.avbdCpuSoftBodyGroundTetPatchGroundPositionAlRows;
	stages.groundTetPatchFourSupportRows +=
		sceneStatistics.avbdCpuSoftBodyGroundTetPatchFourSupportRows;
	stages.groundTetPatchSingleTetRows +=
		sceneStatistics.avbdCpuSoftBodyGroundTetPatchSingleTetRows;
	stages.groundTetPatchActiveRows +=
		sceneStatistics.avbdCpuSoftBodyGroundTetPatchActiveRows;
	stages.worldStaticVelocityTangentOwnerRows +=
		sceneStatistics.
			avbdCpuSoftBodyWorldStaticVelocityTangentOwnerRows;
	stages.worldStaticVelocityTangentAppliedRows +=
		sceneStatistics.
			avbdCpuSoftBodyWorldStaticVelocityTangentAppliedRows;

	AvbdSoftCollisionStats& collision = gPerformance.collision;
	collision.detectionCalls +=
		sceneStatistics.avbdCpuSoftBodyCollisionDetectionCalls;
	collision.bodyPairs +=
		sceneStatistics.avbdCpuSoftBodyCollisionBodyPairs;
	collision.overlappingBodyPairs +=
		sceneStatistics.avbdCpuSoftBodyCollisionOverlappingBodyPairs;
	collision.particleSurfaceCandidates +=
		sceneStatistics.avbdCpuSoftBodyCollisionParticleSurfaceCandidates;
	collision.insideTriangleTests +=
		sceneStatistics.avbdCpuSoftBodyCollisionInsideTriangleTests;
	collision.closestTriangleTests +=
		sceneStatistics.avbdCpuSoftBodyCollisionClosestTriangleTests;
	collision.selfTriangleTests +=
		sceneStatistics.avbdCpuSoftBodyCollisionSelfTriangleTests;
	collision.selfTriangleBoundsBuilt +=
		sceneStatistics.avbdCpuSoftBodyCollisionSelfTriangleBoundsBuilt;
	collision.selfVertexSweepEntriesBuilt +=
		sceneStatistics.avbdCpuSoftBodyCollisionSelfVertexSweepEntriesBuilt;
	collision.selfEdgeBoundsBuilt +=
		sceneStatistics.avbdCpuSoftBodyCollisionSelfEdgeBoundsBuilt;
	collision.surfaceTriangleBvhRefitNodes +=
		sceneStatistics.avbdCpuSoftBodyCollisionSurfaceBvhRefitNodes;
	collision.surfaceTriangleBvhCandidateTriangles +=
		sceneStatistics.avbdCpuSoftBodyCollisionSurfaceBvhCandidates;
	collision.surfaceEdgeBvhRefitNodes +=
		sceneStatistics.avbdCpuSoftBodyCollisionSurfaceEdgeBvhRefitNodes;
	collision.surfaceEdgeBvhCandidateEdges +=
		sceneStatistics.avbdCpuSoftBodyCollisionSurfaceEdgeBvhCandidates;
	gPerformance.collisionRigidParticleTests +=
		sceneStatistics.avbdCpuSoftBodyCollisionRigidParticleTests;
	collision.rigidTriangleSurfaceFaceCandidates +=
		sceneStatistics.avbdCpuSoftBodyCollisionRigidTriangleFaceCandidates;
	collision.rigidTriangleSurfaceFaceTests +=
		sceneStatistics.avbdCpuSoftBodyCollisionRigidTriangleFaceTests;
	collision.rigidTriangleSurfaceEdgeCandidates +=
		sceneStatistics.avbdCpuSoftBodyCollisionRigidTriangleEdgeCandidates;
	collision.rigidTriangleSurfaceEdgeTests +=
		sceneStatistics.avbdCpuSoftBodyCollisionRigidTriangleEdgeTests;
	collision.rigidTriangleSurfaceVertexCandidates +=
		sceneStatistics.avbdCpuSoftBodyCollisionRigidTriangleVertexCandidates;
	collision.rigidTriangleSurfaceVertexTests +=
		sceneStatistics.avbdCpuSoftBodyCollisionRigidTriangleVertexTests;
	collision.generatedGroundContacts +=
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedGroundContacts;
	collision.generatedRigidContacts +=
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedRigidContacts;
	collision.generatedSoftContacts +=
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedSoftContacts;
	collision.generatedSelfContacts +=
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedSelfContacts;
}

// ---------------------------------------------------------------------------
// Push AVBD soft-body surface triangles to PVD as debug geometry.
// ---------------------------------------------------------------------------
static void sendSoftBodiesToPvd()
{
	PxPvdSceneClient* pvdClient = gScene ? gScene->getScenePvdClient() : NULL;
	if (!pvdClient)
		return;

	static const PxU32 bodyColors[] = { 0xFFFF8000, 0xFF0080FF, 0xFF00FF80 };

	PxArray<PxDebugTriangle> tris;
	for (PxU32 b = 0; b < gSoftBodies.size(); b++)
	{
		const AvbdSoftBody& sb = gSoftBodies[b];
		const PxU32* idx = sb.compiled.surfaceTriangles.begin();
		const PxU32 numTris = sb.compiled.surfaceTriangles.size() / 3;
		const PxU32 color = bodyColors[b % (sizeof(bodyColors) / sizeof(bodyColors[0]))];

		for (PxU32 t = 0; t < numTris; t++)
		{
			const PxVec3& p0 = gParticles[idx[t * 3 + 0]].position;
			const PxVec3& p1 = gParticles[idx[t * 3 + 1]].position;
			const PxVec3& p2 = gParticles[idx[t * 3 + 2]].position;
			tris.pushBack(PxDebugTriangle(p0, p1, p2, color));
		}
	}

	if (tris.size())
		pvdClient->drawTriangles(tris.begin(), tris.size());
}

// ---------------------------------------------------------------------------
static AvbdOGCParams gOGCParams;

static void initOGCParams()
{
	gOGCParams.contactRadius    = 0.20f;
	gOGCParams.contactStiffness = 3e5f;
	gOGCParams.friction         = 0.35f;
}

// ---------------------------------------------------------------------------
static void updateRenderData()
{
	gSoftBodyRenderData.clear();
	for (PxU32 i = 0; i < gSoftBodies.size(); i++)
	{
		SoftBodyRenderData rd;
		rd.surfaceTriIndices = gSoftBodies[i].compiled.surfaceTriangles.begin();
		rd.numSurfaceTris    = gSoftBodies[i].compiled.surfaceTriangles.size() / 3;
		gSoftBodyRenderData.pushBack(rd);
	}
}

static PxVec3 getSoftBodyCentroid(const AvbdSoftBody& body)
{
	PxVec3 centroid(0.0f);
	PxReal totalMass = 0.0f;
	for(PxU32 localId = 0; localId < body.compiled.particleCount; ++localId)
	{
		const AvbdSoftParticle& particle =
			gParticles[body.compiled.particleStart + localId];
		centroid += particle.position * particle.mass;
		totalMass += particle.mass;
	}
	return totalMass > 0.0f ?
		centroid * (1.0f / totalMass) : PxVec3(0.0f);
}

static void addCubeSoftBody(
	const PxVec3& center, PxReal halfExtent, PxU32 subdivisions,
	PxReal youngsModulus = 2e5f, PxReal density = 500.0f,
	PxReal damping = 0.015f)
{
	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	avbdGenerateSubdividedCubeTets(
		center, halfExtent, int(subdivisions), verts, tets);
	avbdCreateSoftBody(
		verts.begin(), verts.size(), tets.begin(), tets.size(), NULL, 0,
		youngsModulus, 0.3f, density, damping, 0.0f, 0.01f,
		gParticles, gSoftBodies);
}

static void addConeSoftBody(const PxVec3& baseCenter)
{
	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	// Eight shrinking 32-point rings plus one apex form a real tapered cone.
	// Unlike the old voxel conversion, the collision surface has no stair-step
	// cells and every exterior side vertex lies on the analytic cone.
	avbdGenerateConeTets(
		baseCenter, 0.8f, 3.0f, 8, verts, tets);
	avbdCreateSoftBody(
		verts.begin(), verts.size(), tets.begin(), tets.size(), NULL, 0,
		2e5f, 0.3f, 100.0f, 0.015f, 0.0f, 0.01f,
		gParticles, gSoftBodies);
}

static bool addRigidBox(
	const PxVec3& center, const PxVec3& halfExtent)
{
	AvbdRigidBox rigidBox;
	rigidBox.center = center;
	rigidBox.halfExtent = halfExtent;
	rigidBox.friction = 0.5f;
	gRigidBoxes.pushBack(rigidBox);

	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	return true;
}


static SweptGeometrySampler getSceneCpuVolumeSweptGeometrySampler()
{
	return SweptGeometrySampler(
		gSceneCpuSphereReverseSweptInitialPositions,
		gSceneCpuDeformingReverseSweptFreeEndPositions,
		gSceneCpuCapsuleRotationalSweptInitialPositions,
		gSceneCpuRigidConvexMesh,
		gSceneCpuRigidTriangleMesh,
		gSceneCpuRigidHeightField);
}

static KinematicFiniteSweptMetrics getKinematicFiniteSweptMetrics()
{
	KinematicFiniteSweptMetrics metrics;
	metrics.targetIssued = gMetrics.movingSphereTargetIssued;
	metrics.responseObserved = gMetrics.movingSphereCcdResponseObserved;
	metrics.negativeControlPassed =
		gMetrics.movingSphereNegativeControlHeld;
	metrics.positiveDisplacement =
		gMetrics.movingSpherePositiveDisplacement;
	metrics.negativeDisplacement =
		gMetrics.movingSphereNegativeDisplacement;
	metrics.positiveMinSeparation =
		gMetrics.movingSpherePositiveMinSeparation;
	return metrics;
}

static DynamicFiniteSweptMetrics getDynamicFiniteSweptMetrics()
{
	DynamicFiniteSweptMetrics metrics;
	metrics.launched = gMetrics.dynamicSphereSweepLaunched;
	metrics.responseObserved =
		gMetrics.dynamicSphereSweepResponseObserved;
	metrics.negativeControlPassed =
		gMetrics.dynamicSphereSweepNegativeControlTunneled;
	metrics.twoSidedResponseObserved =
		gMetrics.dynamicSphereSweepTwoSidedResponseObserved;
	metrics.positiveSoftDisplacement =
		gMetrics.dynamicSphereSweepPositiveSoftDisplacement;
	metrics.negativeSoftDisplacement =
		gMetrics.dynamicSphereSweepNegativeSoftDisplacement;
	metrics.positiveRigidDrop =
		gMetrics.dynamicSphereSweepPositiveRigidDrop;
	metrics.negativeRigidDrop =
		gMetrics.dynamicSphereSweepNegativeRigidDrop;
	metrics.positiveMinSeparation =
		gMetrics.dynamicSphereSweepPositiveMinSeparation;
	return metrics;
}


static bool isComponentNoContactCase(const std::string& caseName)
{
	return isComponentDenseNoContactCase(caseName) ||
		isComponentManySmallNoContactCase(caseName);
}

static bool addSceneStaticBox(
	const PxVec3& center, const PxVec3& halfExtent)
{
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return true;
}

static bool addSceneStaticSphereCluster(
	const PxVec3* centers, PxU32 centerCount, PxReal radius)
{
	if(!centers || centerCount == 0 || radius <= 0.0f)
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(PxIdentity));
	if(!actor)
		return false;
	for(PxU32 i = 0; i < centerCount; ++i)
	{
		PxShape* shape = PxRigidActorExt::createExclusiveShape(
			*actor, PxSphereGeometry(radius), *gMaterial);
		if(!shape)
		{
			actor->release();
			return false;
		}
		shape->setLocalPose(PxTransform(centers[i]));
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return true;
}

static bool addSceneStaticCapsule(
	const PxVec3& center, PxReal radius, PxReal halfHeight)
{
	if(radius <= 0.0f || halfHeight < 0.0f)
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxCapsuleGeometry(radius, halfHeight), *gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return true;
}

static bool addSceneStaticCapsuleCluster(
	const PxVec3* centers, PxU32 centerCount,
	PxReal radius, PxReal halfHeight)
{
	if(!centers || centerCount == 0 ||
		radius <= 0.0f || halfHeight < 0.0f)
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(PxIdentity));
	if(!actor)
		return false;
	for(PxU32 i = 0; i < centerCount; ++i)
	{
		PxShape* shape = PxRigidActorExt::createExclusiveShape(
			*actor, PxCapsuleGeometry(radius, halfHeight), *gMaterial);
		if(!shape)
		{
			actor->release();
			return false;
		}
		shape->setLocalPose(PxTransform(centers[i]));
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return true;
}

enum SceneCpuRigidConvexFixture
{
	eSCENE_CPU_CONVEX_OWNER,
	eSCENE_CPU_CONVEX_REVERSE_FEATURE,
	eSCENE_CPU_CONVEX_SWEPT_BOX,
	eSCENE_CPU_CONVEX_REVERSE_SWEPT,
	eSCENE_CPU_CONVEX_DEFORMING_REVERSE_SWEPT,
	eSCENE_CPU_CONVEX_ROTATIONAL
};

static bool createSceneCpuRigidConvexMesh(
	SceneCpuRigidConvexFixture fixture)
{
	if(gSceneCpuRigidConvexMesh || !gPhysics)
		return false;
	const PxVec3 reverseVertices[] =
	{
		PxVec3(0.0f, 0.3f, 0.0f),
		PxVec3(-0.3f, -0.3f, -0.3f),
		PxVec3(0.3f, -0.3f, -0.3f),
		PxVec3(0.0f, -0.3f, 0.35f)
	};
	const PxVec3 ownerVertices[] =
	{
		PxVec3(-0.8f, 0.0f, 0.0f),
		PxVec3(0.8f, 0.0f, 0.0f),
		PxVec3(0.0f, -0.8f, 0.0f),
		PxVec3(0.0f, 0.8f, 0.0f),
		PxVec3(0.0f, 0.0f, -0.8f),
		PxVec3(0.0f, 0.0f, 0.8f)
	};
	const PxVec3 sweptBoxVertices[] =
	{
		PxVec3(-0.8f, -0.05f, -0.8f),
		PxVec3(0.8f, -0.05f, -0.8f),
		PxVec3(0.8f, -0.05f, 0.8f),
		PxVec3(-0.8f, -0.05f, 0.8f),
		PxVec3(-0.8f, 0.05f, -0.8f),
		PxVec3(0.8f, 0.05f, -0.8f),
		PxVec3(0.8f, 0.05f, 0.8f),
		PxVec3(-0.8f, 0.05f, 0.8f)
	};
	const PxVec3 reverseSweptVertices[] =
	{
		PxVec3(0.0f, -0.3f, 0.0f),
		PxVec3(0.0f, 0.3f, 0.0f),
		PxVec3(-0.25f, 0.0f, -0.2f),
		PxVec3(0.25f, 0.0f, -0.2f),
		PxVec3(0.25f, 0.0f, 0.2f),
		PxVec3(-0.25f, 0.0f, 0.2f)
	};
	const PxVec3 deformingReverseSweptVertices[] =
	{
		PxVec3(0.0f, -0.2f, 0.0f),
		PxVec3(0.0f, 0.2f, 0.0f),
		PxVec3(-0.15f, 0.0f, -0.12f),
		PxVec3(0.15f, 0.0f, -0.12f),
		PxVec3(0.15f, 0.0f, 0.12f),
		PxVec3(-0.15f, 0.0f, 0.12f)
	};
	const PxVec3 rotationalVertices[] =
	{
		PxVec3(-1.0f, -0.1f, -0.1f),
		PxVec3(1.0f, -0.1f, -0.1f),
		PxVec3(1.0f, -0.1f, 0.1f),
		PxVec3(-1.0f, -0.1f, 0.1f),
		PxVec3(-1.0f, 0.1f, -0.1f),
		PxVec3(1.0f, 0.1f, -0.1f),
		PxVec3(1.0f, 0.1f, 0.1f),
		PxVec3(-1.0f, 0.1f, 0.1f)
	};
	const PxVec3* vertices = ownerVertices;
	PxU32 vertexCount =
		sizeof(ownerVertices) / sizeof(ownerVertices[0]);
	if(fixture == eSCENE_CPU_CONVEX_REVERSE_FEATURE)
	{
		vertices = reverseVertices;
		vertexCount =
			sizeof(reverseVertices) / sizeof(reverseVertices[0]);
	}
	else if(fixture == eSCENE_CPU_CONVEX_SWEPT_BOX)
	{
		vertices = sweptBoxVertices;
		vertexCount =
			sizeof(sweptBoxVertices) / sizeof(sweptBoxVertices[0]);
	}
	else if(fixture == eSCENE_CPU_CONVEX_REVERSE_SWEPT)
	{
		vertices = reverseSweptVertices;
		vertexCount =
			sizeof(reverseSweptVertices) /
			sizeof(reverseSweptVertices[0]);
	}
	else if(
		fixture ==
			eSCENE_CPU_CONVEX_DEFORMING_REVERSE_SWEPT)
	{
		vertices = deformingReverseSweptVertices;
		vertexCount =
			sizeof(deformingReverseSweptVertices) /
			sizeof(deformingReverseSweptVertices[0]);
	}
	else if(fixture == eSCENE_CPU_CONVEX_ROTATIONAL)
	{
		vertices = rotationalVertices;
		vertexCount =
			sizeof(rotationalVertices) /
			sizeof(rotationalVertices[0]);
	}
	PxConvexMeshDesc convexDesc;
	convexDesc.points.count = vertexCount;
	convexDesc.points.stride = sizeof(PxVec3);
	convexDesc.points.data = vertices;
	convexDesc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
	PxCookingParams cookingParams(
		gPhysics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	gSceneCpuRigidConvexMesh = PxCreateConvexMesh(
		cookingParams, convexDesc,
		gPhysics->getPhysicsInsertionCallback());
	return gSceneCpuRigidConvexMesh != NULL;
}

static bool addSceneStaticConvex(
	const PxVec3& center, bool reverseFeature)
{
	if(!createSceneCpuRigidConvexMesh(
			reverseFeature
				? eSCENE_CPU_CONVEX_REVERSE_FEATURE
				: eSCENE_CPU_CONVEX_OWNER))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor,
		PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
		*gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool addSceneStaticConvexCluster(
	const PxVec3* centers, PxU32 centerCount,
	SceneCpuRigidConvexFixture fixture)
{
	if(!centers || centerCount == 0 ||
		!createSceneCpuRigidConvexMesh(fixture))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(PxIdentity));
	if(!actor)
		return false;
	for(PxU32 i = 0; i < centerCount; ++i)
	{
		PxShape* shape = PxRigidActorExt::createExclusiveShape(
			*actor,
			PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
			*gMaterial);
		if(!shape)
		{
			actor->release();
			return false;
		}
		shape->setLocalPose(PxTransform(centers[i]));
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool createSceneCpuRigidTriangleMesh(bool reverseFeature)
{
	if(gSceneCpuRigidTriangleMesh || !gPhysics)
		return false;
	// P5.19/P5.20 fixtures own the legacy two-triangle hierarchy. Their purpose
	// is to isolate large-body/two-surface partitioning, not to turn a
	// hierarchy-size experiment into a misleading threshold measurement. The
	// ordinary P1 corpus knob stays opt-in for every other headless case.
	const PxU32 meshGridDimension =
		(isSceneCpuVolumeTaskGraphRigidTriangleSurfaceLargeCase(
			gHeadlessOptions.caseName) ||
		 isSceneCpuVolumeTaskGraphRigidTriangleSurfaceFeatureOverlapCase(
			gHeadlessOptions.caseName) ||
		 isSceneCpuVolumeTaskGraphRigidTriangleSurfaceThresholdCase(
			gHeadlessOptions.caseName)) ? 1u :
		gRigidTriangleMeshGridDimension;
	const PxVec3 reverseVertices[] =
	{
		PxVec3(0.0f, 0.3f, 0.0f),
		PxVec3(-0.3f, -0.3f, -0.3f),
		PxVec3(0.3f, -0.3f, -0.3f),
		PxVec3(0.0f, -0.3f, 0.35f)
	};
	const PxU32 reverseTriangles[] =
	{
		0, 2, 1,
		0, 3, 2,
		0, 1, 3,
		1, 2, 3
	};
	const PxVec3 ownerVertices[] =
	{
		PxVec3(-4.0f, 0.0f, -4.0f),
		PxVec3(-4.0f, 0.0f, 4.0f),
		PxVec3(4.0f, 0.0f, 4.0f),
		PxVec3(4.0f, 0.0f, -4.0f)
	};
	const PxU32 ownerTriangles[] =
	{
		0, 1, 2,
		0, 2, 3
	};
	PxArray<PxVec3> corpusVertices;
	PxArray<PxU32> corpusTriangles;
	const PxVec3* meshVertices = reverseFeature
		? reverseVertices : ownerVertices;
	const PxU32* meshTriangles = reverseFeature
		? reverseTriangles : ownerTriangles;
	PxU32 vertexCount = 4;
	PxU32 triangleCount = reverseFeature ? 4u : 2u;
	if(meshGridDimension > 1)
	{
		const PxU32 gridDimension = meshGridDimension;
		const PxU32 vertexDimension = gridDimension + 1;
		const PxReal cellExtent = 8.0f / PxReal(gridDimension);
		const PxVec3 gridOrigin = reverseFeature
			// Preserve the local reverse-feature tetra. The corpus patch is
			// intentionally remote, so its only role is to test hierarchy
			// rejection against a topology-large triangle surface.
			? PxVec3(16.0f, 0.0f, 16.0f)
			: PxVec3(-4.0f, 0.0f, -4.0f);
		const PxU32 baseVertexCount = reverseFeature ? 4u : 0u;
		corpusVertices.reserve(
			baseVertexCount + vertexDimension * vertexDimension);
		corpusTriangles.reserve(
			(reverseFeature ? 12u : 0u) +
				6 * gridDimension * gridDimension);
		if(reverseFeature)
		{
			for(PxU32 vertexIndex = 0; vertexIndex < 4; ++vertexIndex)
				corpusVertices.pushBack(reverseVertices[vertexIndex]);
			for(PxU32 index = 0; index < 12; ++index)
				corpusTriangles.pushBack(reverseTriangles[index]);
		}
		for(PxU32 row = 0; row < vertexDimension; ++row)
		{
			const PxReal z =
				gridOrigin.z + PxReal(row) * cellExtent;
			for(PxU32 column = 0; column < vertexDimension; ++column)
			{
				const PxReal x =
					gridOrigin.x + PxReal(column) * cellExtent;
				corpusVertices.pushBack(PxVec3(x, gridOrigin.y, z));
			}
		}
		for(PxU32 row = 0; row < gridDimension; ++row)
		{
			for(PxU32 column = 0; column < gridDimension; ++column)
			{
				const PxU32 lowerLeft = baseVertexCount +
					row * vertexDimension + column;
				const PxU32 upperLeft = lowerLeft + vertexDimension;
				const PxU32 upperRight = upperLeft + 1;
				const PxU32 lowerRight = lowerLeft + 1;
				// Match the legacy two-triangle plane winding and diagonal.
				corpusTriangles.pushBack(lowerLeft);
				corpusTriangles.pushBack(upperLeft);
				corpusTriangles.pushBack(upperRight);
				corpusTriangles.pushBack(lowerLeft);
				corpusTriangles.pushBack(upperRight);
				corpusTriangles.pushBack(lowerRight);
			}
		}
		meshVertices = corpusVertices.begin();
		meshTriangles = corpusTriangles.begin();
		vertexCount = corpusVertices.size();
		triangleCount = corpusTriangles.size() / 3;
	}
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = vertexCount;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = meshVertices;
	meshDesc.triangles.count = triangleCount;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = meshTriangles;
	PxCookingParams cookingParams(
		gPhysics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	gSceneCpuRigidTriangleMesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		gPhysics->getPhysicsInsertionCallback());
	return gSceneCpuRigidTriangleMesh != NULL;
}

static bool createSceneCpuRotationalTriangleMesh()
{
	if(gSceneCpuRigidTriangleMesh || !gPhysics)
		return false;
	const PxVec3 vertices[] =
	{
		PxVec3(-1.0f, 0.0f, -0.1f),
		PxVec3(-1.0f, 0.0f, 0.1f),
		PxVec3(1.0f, 0.0f, 0.1f),
		PxVec3(1.0f, 0.0f, -0.1f)
	};
	const PxU32 triangles[] =
	{
		0, 1, 2,
		0, 2, 3
	};
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = 4;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = vertices;
	meshDesc.triangles.count = 2;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = triangles;
	PxCookingParams cookingParams(
		gPhysics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	gSceneCpuRigidTriangleMesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		gPhysics->getPhysicsInsertionCallback());
	return gSceneCpuRigidTriangleMesh != NULL;
}

static bool createSceneCpuRigidHeightField(bool reverseFeature)
{
	if(gSceneCpuRigidHeightField || !gPhysics)
		return false;
	PxHeightFieldSample samples[9];
	for(PxU32 sampleIndex = 0;
		sampleIndex < 9; ++sampleIndex)
	{
		samples[sampleIndex].height =
			reverseFeature ? PxI16(-3) : PxI16(0);
		samples[sampleIndex].materialIndex0 =
			PxBitAndByte(0);
		samples[sampleIndex].materialIndex1 =
			PxBitAndByte(0);
		samples[sampleIndex].setTessFlag();
	}
	if(reverseFeature)
		samples[4].height = PxI16(3);
	PxHeightFieldDesc heightFieldDesc;
	heightFieldDesc.nbRows = 3;
	heightFieldDesc.nbColumns = 3;
	heightFieldDesc.samples.data = samples;
	heightFieldDesc.samples.stride =
		sizeof(PxHeightFieldSample);
	gSceneCpuRigidHeightField = PxCreateHeightField(
		heightFieldDesc,
		gPhysics->getPhysicsInsertionCallback());
	return gSceneCpuRigidHeightField != NULL;
}

static bool addSceneStaticTriangleMesh(
	const PxVec3& center, bool reverseFeature)
{
	if(!createSceneCpuRigidTriangleMesh(reverseFeature))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxTriangleMeshGeometry(
				gSceneCpuRigidTriangleMesh),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool addSceneStaticHeightField(
	const PxVec3& center, bool reverseFeature)
{
	if(!createSceneCpuRigidHeightField(reverseFeature))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	const PxHeightFieldGeometry geometry(
		gSceneCpuRigidHeightField,
		PxMeshGeometryFlags(), 0.1f,
		reverseFeature ? 0.3f : 4.0f,
		reverseFeature ? 0.3f : 4.0f);
	if(!PxRigidActorExt::createExclusiveShape(
			*actor, geometry, *gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool addSceneStaticTriangleSurfacePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	bool heightField, bool reverseFeature)
{
	if(heightField
			? !createSceneCpuRigidHeightField(reverseFeature)
			: !createSceneCpuRigidTriangleMesh(reverseFeature))
		return false;
	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(PxIdentity));
	if(!actor)
		return false;
	const PxVec3 centers[2] =
		{positiveCenter, negativeCenter};
	for(PxU32 i = 0; i < 2; ++i)
	{
		PxShape* shape = heightField
			? PxRigidActorExt::createExclusiveShape(
				*actor,
				PxHeightFieldGeometry(
					gSceneCpuRigidHeightField,
					PxMeshGeometryFlags(), 0.1f,
					reverseFeature ? 0.3f : 4.0f,
					reverseFeature ? 0.3f : 4.0f),
				*gMaterial)
			: PxRigidActorExt::createExclusiveShape(
				*actor,
				PxTriangleMeshGeometry(
					gSceneCpuRigidTriangleMesh),
				*gMaterial);
		if(!shape)
		{
			actor->release();
			return false;
		}
		shape->setLocalPose(PxTransform(centers[i]));
	}
	gScene->addActor(*actor);
	gSceneCpuStaticActor = actor;
	return actor->getScene() == gScene;
}

static bool addSceneDynamicBox(
	const PxVec3& center, const PxVec3& halfExtent,
	bool startSleeping = true)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	if(!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	if(startSleeping)
		actor->putToSleep();
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = actor->getGlobalPose().p.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneDynamicSphere(
	const PxVec3& center, PxReal radius)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxSphereGeometry(radius), *gMaterial))
	{
		actor->release();
		return false;
	}
	if(!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	actor->putToSleep();
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneDynamicCapsule(
	const PxVec3& center, PxReal radius, PxReal halfHeight)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxCapsuleGeometry(radius, halfHeight), *gMaterial) ||
		!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	actor->putToSleep();
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneDynamicConvex(
	const PxVec3& center,
	SceneCpuRigidConvexFixture fixture =
		eSCENE_CPU_CONVEX_OWNER)
{
	if(!createSceneCpuRigidConvexMesh(fixture))
		return false;
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
			*gMaterial) ||
		!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	actor->putToSleep();
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicBox(
	const PxVec3& center, const PxVec3& halfExtent)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicSphere(
	const PxVec3& center, PxReal radius)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxSphereGeometry(radius), *gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicCapsule(
	const PxVec3& center, PxReal radius, PxReal halfHeight)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxCapsuleGeometry(radius, halfHeight), *gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicConvex(
	const PxVec3& center,
	SceneCpuRigidConvexFixture fixture =
		eSCENE_CPU_CONVEX_OWNER)
{
	if(!createSceneCpuRigidConvexMesh(fixture))
		return false;
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor,
			PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicTriangleMesh(const PxVec3& center)
{
	if(!createSceneCpuRigidTriangleMesh(false))
		return false;
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxTriangleMeshGeometry(
				gSceneCpuRigidTriangleMesh),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneKinematicHeightField(const PxVec3& center)
{
	if(!createSceneCpuRigidHeightField(false))
		return false;
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eKINEMATIC, true);
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxHeightFieldGeometry(
				gSceneCpuRigidHeightField,
				PxMeshGeometryFlags(),
				0.1f, 4.0f, 4.0f),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = center.y;
	gSceneCpuKinematicCommandY = center.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return actor->getScene() == gScene;
}

static bool addSceneMovingKinematicTriangleSurfacePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal targetY, bool heightField,
	bool reverseFeature)
{
	if(heightField
			? !createSceneCpuRigidHeightField(reverseFeature)
			: !createSceneCpuRigidTriangleMesh(reverseFeature))
		return false;
	const PxVec3 centers[2] =
		{positiveCenter, negativeCenter};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i]));
		if(!actors[i])
			break;
		actors[i]->setRigidBodyFlag(
			PxRigidBodyFlag::eKINEMATIC, true);
		PxShape* shape = heightField
			? PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxHeightFieldGeometry(
					gSceneCpuRigidHeightField,
					PxMeshGeometryFlags(), 0.1f,
					reverseFeature ? 0.3f : 4.0f,
					reverseFeature ? 0.3f : 4.0f),
				*gMaterial)
			: PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxTriangleMeshGeometry(
					gSceneCpuRigidTriangleMesh),
				*gMaterial);
		if(!shape || !gScene->addActor(*actors[i]))
			break;
	}
	if(!actors[0] || !actors[1] ||
		actors[0]->getScene() != gScene ||
		actors[1]->getScene() != gScene)
	{
		for(PxU32 i = 0; i < 2; ++i)
			PX_RELEASE(actors[i]);
		return false;
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gSceneCpuKinematicCommandY = targetY;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.movingSphereTargetIssued = 1;
	for(PxU32 i = 0; i < 2; ++i)
		actors[i]->setKinematicTarget(
			PxTransform(PxVec3(
				centers[i].x, targetY, centers[i].z)));
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneRotatingKinematicTriangleSurfacePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	bool heightField)
{
	if(heightField
			? !createSceneCpuRigidHeightField(false)
			: !createSceneCpuRotationalTriangleMesh())
		return false;
	const PxReal startAngle = 0.5235987756f;
	const PxReal endAngle = 2.6179938780f;
	const PxQuat startRotation(
		startAngle, PxVec3(0.0f, 0.0f, 1.0f));
	const PxQuat endRotation(
		endAngle, PxVec3(0.0f, 0.0f, 1.0f));
	const PxVec3 centers[2] =
		{positiveCenter, negativeCenter};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i])
			break;
		actors[i]->setRigidBodyFlag(
			PxRigidBodyFlag::eKINEMATIC, true);
		PxShape* shape = heightField
			? PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxHeightFieldGeometry(
					gSceneCpuRigidHeightField,
					PxMeshGeometryFlags(), 0.1f,
					1.0f, 0.1f),
				*gMaterial)
			: PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxTriangleMeshGeometry(
					gSceneCpuRigidTriangleMesh),
				*gMaterial);
		if(!shape || !gScene->addActor(*actors[i]))
			break;
	}
	if(!actors[0] || !actors[1] ||
		actors[0]->getScene() != gScene ||
		actors[1]->getScene() != gScene)
	{
		for(PxU32 i = 0; i < 2; ++i)
			PX_RELEASE(actors[i]);
		return false;
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gSceneCpuKinematicCommandY = positiveCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.movingSphereTargetIssued = 1;
	for(PxU32 i = 0; i < 2; ++i)
		actors[i]->setKinematicTarget(
			PxTransform(centers[i], endRotation));
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool addSceneMovingKinematicFinitePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal targetY,
	PxReal radius,
	PxReal capsuleHalfHeight)
{
	const bool capsule = capsuleHalfHeight > 0.0f;
	if(!(capsule
			? addSceneKinematicCapsule(
				positiveCenter, radius, capsuleHalfHeight)
			: addSceneKinematicSphere(positiveCenter, radius)))
		return false;

	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(negativeCenter));
	if(!actor)
		return false;
	PxShape* shape = capsule
		? PxRigidActorExt::createExclusiveShape(
			*actor,
			PxCapsuleGeometry(radius, capsuleHalfHeight),
			*gMaterial)
		: PxRigidActorExt::createExclusiveShape(
			*actor, PxSphereGeometry(radius), *gMaterial);
	if(!shape)
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
	if(!gScene->addActor(*actor))
	{
		actor->release();
		return false;
	}
	gSceneCpuSecondDynamicActor = actor;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);

	gSceneCpuDynamicActor->setKinematicTarget(
		PxTransform(PxVec3(
			positiveCenter.x, targetY, positiveCenter.z)));
	gSceneCpuSecondDynamicActor->setKinematicTarget(
		PxTransform(PxVec3(
			negativeCenter.x, targetY, negativeCenter.z)));
	gMetrics.movingSphereTargetIssued = 1;
	return true;
}

static bool addSceneRotatingKinematicCapsulePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal radius,
	PxReal capsuleHalfHeight,
	const PxQuat& startRotation,
	const PxQuat& endRotation)
{
	const PxVec3 centers[2] =
	{
		positiveCenter,
		negativeCenter
	};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i])
		{
			for(PxU32 j = 0; j < i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
		if(!PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxCapsuleGeometry(radius, capsuleHalfHeight),
				*gMaterial))
		{
			for(PxU32 j = 0; j <= i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
		actors[i]->setRigidBodyFlag(
			PxRigidBodyFlag::eKINEMATIC, true);
		if(!gScene->addActor(*actors[i]))
		{
			for(PxU32 j = 0; j <= i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	for(PxU32 i = 0; i < 2; ++i)
		actors[i]->setKinematicTarget(
			PxTransform(centers[i], endRotation));
	gMetrics.movingSphereTargetIssued = 1;
	return true;
}

static bool addSceneRotatingKinematicConvexPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	const PxQuat& startRotation,
	const PxQuat& endRotation)
{
	if(!createSceneCpuRigidConvexMesh(
			eSCENE_CPU_CONVEX_ROTATIONAL))
		return false;
	const PxVec3 centers[2] =
	{
		positiveCenter,
		negativeCenter
	};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i] ||
			!PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
				*gMaterial))
		{
			for(PxU32 j = 0; j <= i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
		actors[i]->setRigidBodyFlag(
			PxRigidBodyFlag::eKINEMATIC, true);
		if(!gScene->addActor(*actors[i]))
		{
			for(PxU32 j = 0; j <= i; ++j)
				PX_RELEASE(actors[j]);
			return false;
		}
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneKinematicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	for(PxU32 i = 0; i < 2; ++i)
		actors[i]->setKinematicTarget(
			PxTransform(centers[i], endRotation));
	gMetrics.movingSphereTargetIssued = 1;
	return true;
}

static bool addSceneDynamicFiniteSweepPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal radius,
	PxReal capsuleHalfHeight,
	PxReal launchSpeed)
{
	const bool capsule = capsuleHalfHeight > 0.0f;
	PxRigidDynamic* actors[2] = {NULL, NULL};
	const PxVec3 centers[2] = {
		positiveCenter, negativeCenter
	};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] =
			gPhysics->createRigidDynamic(PxTransform(centers[i]));
		PxShape* shape = actors[i]
			? (capsule
				? PxRigidActorExt::createExclusiveShape(
					*actors[i],
					PxCapsuleGeometry(radius, capsuleHalfHeight),
					*gMaterial)
				: PxRigidActorExt::createExclusiveShape(
					*actors[i],
					PxSphereGeometry(radius), *gMaterial))
			: NULL;
		if(!actors[i] || !shape ||
			!PxRigidBodyExt::setMassAndUpdateInertia(
				*actors[i], 1.0f))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
				PX_RELEASE(actors[0]);
			return false;
		}
		actors[i]->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		actors[i]->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
		actors[i]->setSolverIterationCounts(16, 1);
		actors[i]->setLinearVelocity(
			PxVec3(0.0f, launchSpeed, 0.0f));
		if(!gScene->addActor(*actors[i]))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
	}

	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.dynamicSphereSweepLaunched = 1;
	return gSceneCpuDynamicActor->getScene() == gScene &&
		gSceneCpuSecondDynamicActor->getScene() == gScene;
}

static bool addSceneDynamicRotatingCapsulePair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal radius,
	PxReal capsuleHalfHeight,
	const PxQuat& startRotation,
	PxReal mass)
{
	const PxReal angularSpeed =
		(PxPi * 10.0f / 9.0f) / gHeadlessOptions.dt;
	const PxVec3 centers[2] =
	{
		positiveCenter,
		negativeCenter
	};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i] ||
			!PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxCapsuleGeometry(radius, capsuleHalfHeight),
				*gMaterial) ||
			!PxRigidBodyExt::setMassAndUpdateInertia(
				*actors[i], mass))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
		actors[i]->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		actors[i]->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y);
		actors[i]->setSolverIterationCounts(16, 1);
		actors[i]->setMaxAngularVelocity(300.0f);
		actors[i]->setAngularVelocity(
			PxVec3(0.0f, 0.0f, angularSpeed));
		if(!gScene->addActor(*actors[i]))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.dynamicSphereSweepLaunched = 1;
	return gSceneCpuDynamicActor->getScene() == gScene &&
		gSceneCpuSecondDynamicActor->getScene() == gScene;
}

static bool addSceneDynamicRotatingConvexPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	const PxQuat& startRotation,
	PxReal mass)
{
	if(!createSceneCpuRigidConvexMesh(
			eSCENE_CPU_CONVEX_ROTATIONAL))
		return false;
	const PxReal angularSpeed =
		(PxPi * 10.0f / 9.0f) / gHeadlessOptions.dt;
	const PxVec3 centers[2] =
	{
		positiveCenter,
		negativeCenter
	};
	PxRigidDynamic* actors[2] = {NULL, NULL};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] = gPhysics->createRigidDynamic(
			PxTransform(centers[i], startRotation));
		if(!actors[i] ||
			!PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
				*gMaterial) ||
			!PxRigidBodyExt::setMassAndUpdateInertia(
				*actors[i], mass))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
		actors[i]->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		actors[i]->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y);
		actors[i]->setSolverIterationCounts(16, 1);
		actors[i]->setMaxAngularVelocity(300.0f);
		actors[i]->setAngularVelocity(
			PxVec3(0.0f, 0.0f, angularSpeed));
		if(!gScene->addActor(*actors[i]))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
	}
	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.dynamicSphereSweepLaunched = 1;
	return gSceneCpuDynamicActor->getScene() == gScene &&
		gSceneCpuSecondDynamicActor->getScene() == gScene;
}

static bool addSceneMovingKinematicConvexPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal targetY,
	SceneCpuRigidConvexFixture fixture)
{
	if(!addSceneKinematicConvex(positiveCenter, fixture))
		return false;

	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(negativeCenter));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
			*gMaterial))
	{
		actor->release();
		return false;
	}
	actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
	if(!gScene->addActor(*actor))
	{
		actor->release();
		return false;
	}
	gSceneCpuSecondDynamicActor = actor;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);

	gSceneCpuDynamicActor->setKinematicTarget(
		PxTransform(PxVec3(
			positiveCenter.x, targetY, positiveCenter.z)));
	gSceneCpuSecondDynamicActor->setKinematicTarget(
		PxTransform(PxVec3(
			negativeCenter.x, targetY, negativeCenter.z)));
	gMetrics.movingSphereTargetIssued = 1;
	return true;
}

static bool addSceneDynamicConvexSweepPair(
	const PxVec3& positiveCenter,
	const PxVec3& negativeCenter,
	PxReal launchSpeed,
	SceneCpuRigidConvexFixture fixture)
{
	if(!createSceneCpuRigidConvexMesh(fixture))
		return false;
	PxRigidDynamic* actors[2] = {NULL, NULL};
	const PxVec3 centers[2] = {
		positiveCenter, negativeCenter
	};
	for(PxU32 i = 0; i < 2; ++i)
	{
		actors[i] =
			gPhysics->createRigidDynamic(PxTransform(centers[i]));
		PxShape* shape = actors[i]
			? PxRigidActorExt::createExclusiveShape(
				*actors[i],
				PxConvexMeshGeometry(gSceneCpuRigidConvexMesh),
				*gMaterial)
			: NULL;
		if(!actors[i] || !shape ||
			!PxRigidBodyExt::setMassAndUpdateInertia(
				*actors[i], 1.0f))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
				PX_RELEASE(actors[0]);
			return false;
		}
		actors[i]->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		actors[i]->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
		actors[i]->setSolverIterationCounts(16, 1);
		actors[i]->setLinearVelocity(
			PxVec3(0.0f, launchSpeed, 0.0f));
		if(!gScene->addActor(*actors[i]))
		{
			PX_RELEASE(actors[i]);
			if(i > 0)
			{
				if(actors[0]->getScene() == gScene)
					gScene->removeActor(*actors[0]);
				PX_RELEASE(actors[0]);
			}
			return false;
		}
	}

	gSceneCpuDynamicActor = actors[0];
	gSceneCpuSecondDynamicActor = actors[1];
	gSceneCpuDynamicInitialY = positiveCenter.y;
	gSceneCpuSecondDynamicInitialY = negativeCenter.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.dynamicSphereSweepLaunched = 1;
	return gSceneCpuDynamicActor->getScene() == gScene &&
		gSceneCpuSecondDynamicActor->getScene() == gScene;
}

static bool addSceneSecondDynamicBox(
	const PxVec3& center, const PxVec3& halfExtent)
{
	PxRigidDynamic* actor =
		gPhysics->createRigidDynamic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	if(!PxRigidBodyExt::setMassAndUpdateInertia(*actor, 20.0f))
	{
		actor->release();
		return false;
	}
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	actor->setRigidDynamicLockFlags(
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
	actor->setSolverIterationCounts(8, 1);
	gScene->addActor(*actor);
	actor->putToSleep();
	gSceneCpuSecondDynamicActor = actor;
	gSceneCpuSecondDynamicInitialY = actor->getGlobalPose().p.y;
	gMetrics.sceneSecondDynamicActorAdded = 1;
	gMetrics.sceneSecondDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	return true;
}

static PxVec3 getSceneCpuVolumeCentroid(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getSimulationMesh())
		return PxVec3(0.0f);
	const PxU32 vertexCount =
		volume->getSimulationMesh()->getNbVertices();
	const PxVec4* positions =
		volume->getSimPositionInvMassBufferH();
	if(!positions || vertexCount == 0)
		return PxVec3(0.0f);
	PxVec3 centroid(0.0f);
	for(PxU32 i = 0; i < vertexCount; i++)
		centroid += positions[i].getXYZ();
	return centroid / PxReal(vertexCount);
}

static PxReal getSceneCpuVolumeCentroidY(
	PxDeformableVolume* volume)
{
	return getSceneCpuVolumeCentroid(volume).y;
}

static PxReal getSceneCpuVolumeCentroidY()
{
	return getSceneCpuVolumeCentroidY(gSceneCpuVolume);
}

static PxVec3 getSceneCpuAttachmentPoint(
	PxDeformableVolume* volume)
{
	if(!volume)
		return PxVec3(0.0f);
	const PxVec4* positions =
		volume->getSimPositionInvMassBufferH();
	if(!positions)
		return PxVec3(0.0f);
	if(!gSceneCpuElementAttachment)
		return positions[0].getXYZ();
	return
		positions[gSceneCpuAttachmentVertices[0]].getXYZ() *
			gSceneCpuAttachmentBarycentric.x +
		positions[gSceneCpuAttachmentVertices[1]].getXYZ() *
			gSceneCpuAttachmentBarycentric.y +
		positions[gSceneCpuAttachmentVertices[2]].getXYZ() *
			gSceneCpuAttachmentBarycentric.z +
		positions[gSceneCpuAttachmentVertices[3]].getXYZ() *
			gSceneCpuAttachmentBarycentric.w;
}

static PxVec3 getSceneCpuAttachmentPoint()
{
	return getSceneCpuAttachmentPoint(gSceneCpuVolume);
}

static bool updateSceneStaticChurn()
{
	if(gHeadlessOptions.caseName != "scene-volume-static-churn")
		return true;
	if(!gSceneCpuStaticActor)
		return false;

	if(gMetrics.completedFrames == 1)
	{
		PxShape* shape = NULL;
		if(gSceneCpuStaticActor->getNbShapes() != 1 ||
			gSceneCpuStaticActor->getShapes(&shape, 1) != 1 || !shape)
			return false;
		gSceneCpuStaticActor->detachShape(*shape);
		if(gSceneCpuStaticActor->getNbShapes() != 0)
			return false;
		gMetrics.sceneStaticShapeDetached = 1;

		if(!PxRigidActorExt::createExclusiveShape(
			*gSceneCpuStaticActor,
			PxBoxGeometry(PxVec3(20.0f, 0.5f, 20.0f)),
			*gMaterial))
			return false;
		gMetrics.sceneStaticShapeReattached =
			gSceneCpuStaticActor->getNbShapes() == 1 ? 1u : 0u;
		return gMetrics.sceneStaticShapeReattached == 1;
	}
	if(gMetrics.completedFrames == 2)
	{
		if(gSceneCpuStaticActor->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuStaticActor);
		gMetrics.sceneStaticActorRemoved =
			gSceneCpuStaticActor->getScene() == NULL ? 1u : 0u;
		return gMetrics.sceneStaticActorRemoved == 1;
	}
	if(gMetrics.completedFrames == 3)
	{
		if(gSceneCpuStaticActor->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuStaticActor);
		gMetrics.sceneStaticActorReadded =
			gSceneCpuStaticActor->getScene() == gScene ? 1u : 0u;
		return gMetrics.sceneStaticActorReadded == 1;
	}
	return true;
}

static bool updateSceneDynamicChurn()
{
	if(gHeadlessOptions.caseName != "scene-volume-dynamic-churn")
		return true;
	if(!gSceneCpuDynamicActor)
		return false;

	// Start churn only after the original sleeping actor has been woken by
	// the native soft contact edge.  Each mutation is separated by one Scene
	// step so stale edge/node ownership cannot be hidden by same-frame reuse.
	if(!gMetrics.sceneDynamicWokeBySoft)
		return true;

	if(!gMetrics.sceneDynamicShapeDetached)
	{
		PxShape* shape = NULL;
		if(gSceneCpuDynamicActor->getNbShapes() != 1 ||
			gSceneCpuDynamicActor->getShapes(&shape, 1) != 1 || !shape)
			return false;
		gSceneCpuDynamicActor->detachShape(*shape);
		if(gSceneCpuDynamicActor->getNbShapes() != 0)
			return false;
		gMetrics.sceneDynamicShapeDetached = 1;

		if(!PxRigidActorExt::createExclusiveShape(
			*gSceneCpuDynamicActor,
			PxBoxGeometry(PxVec3(4.0f, 0.25f, 4.0f)),
			*gMaterial))
			return false;
		gMetrics.sceneDynamicShapeReattached =
			gSceneCpuDynamicActor->getNbShapes() == 1 ? 1u : 0u;
		return gMetrics.sceneDynamicShapeReattached == 1;
	}

	if(!gMetrics.sceneDynamicActorRemoved)
	{
		if(gSceneCpuDynamicActor->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuDynamicActor);
		gMetrics.sceneDynamicActorRemoved =
			gSceneCpuDynamicActor->getScene() == NULL ? 1u : 0u;
		return gMetrics.sceneDynamicActorRemoved == 1;
	}

	if(!gMetrics.sceneDynamicActorReadded)
	{
		if(gSceneCpuDynamicActor->getScene() != NULL)
			return false;
		const PxReal reentryY =
			getSceneCpuVolumeCentroidY() - 0.75f;
		gSceneCpuDynamicActor->setGlobalPose(
			PxTransform(PxVec3(0.0f, reentryY, 0.0f)));
		gSceneCpuDynamicActor->setLinearVelocity(PxVec3(0.0f));
		gSceneCpuDynamicActor->setAngularVelocity(PxVec3(0.0f));
		gScene->addActor(*gSceneCpuDynamicActor);
		gMetrics.sceneDynamicActorReadded =
			gSceneCpuDynamicActor->getScene() == gScene ? 1u : 0u;
		if(!gMetrics.sceneDynamicActorReadded)
			return false;
		gSceneCpuDynamicActor->putToSleep();
		gMetrics.sceneDynamicReaddedSleeping =
			gSceneCpuDynamicActor->isSleeping() ? 1u : 0u;
		return gMetrics.sceneDynamicReaddedSleeping == 1;
	}

	return true;
}

static bool updateSceneMultiDynamicGate()
{
	const bool multiDynamicCase =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-dynamic-box";
	const bool multiSoftIslandCase =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-soft-islands";
	if(!multiDynamicCase && !multiSoftIslandCase)
		return true;
	if(!gSceneCpuDynamicActor || !gSceneCpuSecondDynamicActor)
		return false;

	// The fixture is a bounded ownership/merge gate, not a long-lived bridge
	// support scene.  Once both targets have woken and responded, remove both
	// actors on a frame boundary and keep simulating to exercise two-edge
	// teardown without allowing later asymmetric rebound to redefine the gate.
	if(gMetrics.completedFrames != 60)
		return true;
	if(!gMetrics.sceneDynamicWokeBySoft ||
		!gMetrics.sceneSecondDynamicWokeBySoft ||
		gMetrics.sceneDynamicMaxDrop <= 0.05f ||
		gMetrics.sceneSecondDynamicMaxDrop <= 0.05f)
		return false;
	if(gSceneCpuDynamicActor->getScene() != gScene ||
		gSceneCpuSecondDynamicActor->getScene() != gScene)
		return false;

	gScene->removeActor(*gSceneCpuDynamicActor);
	gMetrics.sceneDynamicActorRemoved =
		gSceneCpuDynamicActor->getScene() == NULL ? 1u : 0u;
	gScene->removeActor(*gSceneCpuSecondDynamicActor);
	gMetrics.sceneSecondDynamicActorRemoved =
		gSceneCpuSecondDynamicActor->getScene() == NULL ? 1u : 0u;
	if(multiSoftIslandCase)
	{
		PxDeformableVolume* volumes[2] =
		{
			gSceneCpuVolume,
			gSceneCpuSecondVolume
		};
		for(PxU32 volumeId = 0; volumeId < 2; ++volumeId)
		{
			PxDeformableVolume* volume = volumes[volumeId];
			if(!volume || !volume->getSimulationMesh())
				return false;
			volume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			PxVec4* velocities =
				volume->getSimVelocityBufferH();
			const PxU32 vertexCount =
				volume->getSimulationMesh()->getNbVertices();
			if(!velocities)
				return false;
			for(PxU32 i = 0; i < vertexCount; ++i)
			{
				const PxReal invMass = velocities[i].w;
				velocities[i] =
					PxVec4(0.0f, 0.0f, 0.0f, invMass);
			}
			volume->markDirty(
				PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		}
	}
	return gMetrics.sceneDynamicActorRemoved == 1 &&
		gMetrics.sceneSecondDynamicActorRemoved == 1;
}

static bool createAdditionalSceneCpuVolume(
	PxDeformableVolume*& volume, const PxVec3& translation,
	PxScene* targetScene = NULL,
	PxDeformableVolumeMesh* sourceMesh = NULL)
{
	PxDeformableVolumeMesh* const volumeMesh = sourceMesh ?
		sourceMesh : gSceneCpuVolumeMesh;
	if(!volumeMesh || !gSceneCpuVolumeMaterial)
		return false;
	PxScene* scene = targetScene ? targetScene : gScene;
	if(!scene)
		return false;

	volume = gPhysics->createDeformableVolume(
		PxDeformableVolumeBackend::eCPU_AVBD);
	if(!volume ||
		volume->getDeformableVolumeBackend() !=
			PxDeformableVolumeBackend::eCPU_AVBD ||
		volume->getCudaContextManager() != NULL)
		return false;

	const PxShapeFlags shapeFlags =
		PxShapeFlag::eVISUALIZATION |
		PxShapeFlag::eSCENE_QUERY_SHAPE |
		PxShapeFlag::eSIMULATION_SHAPE;
	PxTetrahedronMesh* shapeMesh =
		isSceneCpuVolumeTaskGraphDirectSimulationDomainCase(
			gHeadlessOptions.caseName)
			? volumeMesh->getSimulationMesh()
			: volumeMesh->getCollisionMesh();
	const PxTetrahedronMeshGeometry geometry(shapeMesh);
	PxDeformableVolumeMaterial* material =
		gSceneCpuVolumeMaterial;
	PxShape* shape = gPhysics->createShape(
		geometry, &material, 1, true, shapeFlags);
	if(!shape)
		return false;
	const bool shapeAttached = volume->attachShape(*shape);
	shape->release();
	if(!shapeAttached ||
		!volume->attachSimulationMesh(
			*volumeMesh->getSimulationMesh(),
			*volumeMesh->getDeformableVolumeAuxData()))
		return false;

	PxVec4* simPositionInvMass =
		volume->getSimPositionInvMassBufferH();
	PxVec4* simVelocity =
		volume->getSimVelocityBufferH();
	PxVec4* collisionPositionInvMass =
		volume->getPositionInvMassBufferH();
	PxVec4* collisionRestPosition =
		volume->getRestPositionBufferH();
	if(!simPositionInvMass || !simVelocity ||
		!collisionPositionInvMass || !collisionRestPosition)
		return false;

	const PxTetrahedronMesh* simulationMesh =
		volume->getSimulationMesh();
	const PxVec3* simulationVertices =
		simulationMesh->getVertices();
	const PxU32 simulationVertexCount =
		simulationMesh->getNbVertices();
	PxReal* cookedInvMass =
		volume->getDeformableVolumeAuxData()->
			getGridModelInvMass();
	for(PxU32 i = 0; i < simulationVertexCount; i++)
	{
		const PxReal invMass = cookedInvMass
			? PxMax(cookedInvMass[i], 0.0f) : 1.0f;
		simPositionInvMass[i] =
			PxVec4(simulationVertices[i] + translation, invMass);
		simVelocity[i] =
			PxVec4(0.0f, 0.0f, 0.0f, invMass);
	}
	PxDeformableVolumeExt::updateMass(
		*volume, 100.0f, 50.0f, simPositionInvMass);
	for(PxU32 i = 0; i < simulationVertexCount; i++)
		simVelocity[i].w = simPositionInvMass[i].w;
	PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
		*volume, simPositionInvMass, collisionPositionInvMass);
	const PxU32 collisionVertexCount =
		volume->getCollisionMesh()->getNbVertices();
	for(PxU32 i = 0; i < collisionVertexCount; i++)
		collisionRestPosition[i] = collisionPositionInvMass[i];
	volume->markDirty(PxDeformableVolumeDataFlag::eALL);
	volume->setSolverIterationCounts(8, 1);
	scene->addActor(*volume);
	return volume->getScene() == scene;
}

static PxDeformableVolumeMesh* cookSceneCpuVisualVolumeMesh(
	const PxArray<PxVec3>& surfaceVertices,
	const PxArray<PxU32>& surfaceTriangles,
	PxU32 voxelsAlongLongestAxis)
{
	if(surfaceVertices.empty() || surfaceTriangles.empty() ||
		surfaceTriangles.size() % 3 != 0 || voxelsAlongLongestAxis < 2)
		return NULL;
	PxSimpleTriangleMesh surfaceMesh;
	surfaceMesh.points.count = surfaceVertices.size();
	surfaceMesh.points.data = surfaceVertices.begin();
	surfaceMesh.points.stride = sizeof(PxVec3);
	surfaceMesh.triangles.count = surfaceTriangles.size() / 3;
	surfaceMesh.triangles.data = surfaceTriangles.begin();
	surfaceMesh.triangles.stride = 3 * sizeof(PxU32);

	PxCookingParams cookingParams(gPhysics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	cookingParams.meshWeldTolerance = 0.001f;
	cookingParams.meshPreprocessParams =
		PxMeshPreprocessingFlag::eWELD_VERTICES;
	return PxDeformableVolumeExt::createDeformableVolumeMesh(
		cookingParams, surfaceMesh, voxelsAlongLongestAxis,
		gPhysics->getPhysicsInsertionCallback());
}

static bool setSceneCpuVolumeVelocity(
	PxDeformableVolume& volume, const PxVec3& velocity);

static bool initializeSceneCpuVisualRotationMetrics(
	RotationMonitor& monitor,
	PxDeformableVolume& volume)
{
	RotationSamplingConfig config;
	config.earlyEndFrame = SCENE_CPU_VISUAL_SPHERE_EARLY_END_FRAME;
	config.lateBeginFrame = SCENE_CPU_VISUAL_SPHERE_LATE_BEGIN_FRAME;
	config.windowBeginFrame = SCENE_CPU_SPHERE_ROLL_WINDOW_BEGIN_FRAME;
	config.windowEndFrame = SCENE_CPU_SPHERE_ROLL_WINDOW_END_FRAME;
	config.checkpointInterval = SCENE_CPU_SPHERE_ROLL_CHECKPOINT_INTERVAL;
	config.checkpointCount = SCENE_CPU_SPHERE_ROLL_CHECKPOINT_COUNT;
	return monitor.initialize(volume, config);
}

static bool sampleSceneCpuVisualRotationMetrics(
	RotationMonitor& monitor)
{
	return monitor.sample(gMetrics.completedFrames);
}

static bool initializeSceneCpuVisualInteractionMetrics(
	PxDeformableVolume& upperJaw, PxDeformableVolume& lowerJaw)
{
	if(!gSceneCpuDynamicActor || !gSceneCpuStaticActor ||
		gSceneCpuDynamicActor->getScene() != gScene ||
		gSceneCpuStaticActor->getScene() != gScene)
		return false;
	return gSceneCpuVisualInteractionMonitor.initialize(
		upperJaw, lowerJaw, *gSceneCpuDynamicActor, *gSceneCpuStaticActor,
		SCENE_CPU_VISUAL_DYNAMIC_BOX_HALF_EXTENTS,
		SCENE_CPU_VISUAL_STATIC_PEDESTAL_HALF_EXTENTS,
		SCENE_CPU_VISUAL_GROUND_HEIGHT);
}

static bool sampleSceneCpuVisualInteractionMetrics()
{
	if(!gScene)
		return false;
	VisualInteractionFrameSample sample;
	if(!gSceneCpuVisualInteractionMonitor.sample(
			*gScene, gMetrics.completedFrames, sample))
	{
		gMetrics.nonFiniteParticleSamples++;
		return false;
	}
	if(sample.generatedRigidContacts)
	{
		gMetrics.rigidContactFrames++;
		gMetrics.maxRigidContacts = PxMax(gMetrics.maxRigidContacts,
			static_cast<PxU32>(sample.generatedRigidContacts));
	}
	if(sample.generatedSoftContacts)
	{
		gMetrics.softContactFrames++;
		gMetrics.maxSoftContacts = PxMax(gMetrics.maxSoftContacts,
			static_cast<PxU32>(sample.generatedSoftContacts));
	}
	// The showcase body begins awake under gravity.  Its actual soft/rigid
	// participation is gated below by contact telemetry plus a finite falling
	// trajectory, rather than by the legacy sleeping-body wake signal.
	gMetrics.sceneDynamicMinY = PxMin(
		gMetrics.sceneDynamicMinY, sample.boxPosition.y);
	gMetrics.sceneDynamicFinalY = sample.boxPosition.y;
	gMetrics.sceneDynamicMaxDrop = PxMax(
		gMetrics.sceneDynamicMaxDrop,
		gSceneCpuDynamicInitialY - sample.boxPosition.y);
	gMetrics.sceneDynamicMaxDownSpeed = PxMax(
		gMetrics.sceneDynamicMaxDownSpeed,
		PxMax(-sample.boxLinearVelocity.y, 0.0f));
	return true;
}

static bool initializeSceneCpuOgcSandwichMetrics(
	PxDeformableVolume& upperJaw, PxDeformableVolume& lowerJaw)
{
	if(!gScene || !gSceneCpuDynamicActor ||
		gSceneCpuDynamicActor->getScene() != gScene)
		return false;
	return gSceneCpuOgcSandwichMonitor.initialize(
		upperJaw, lowerJaw, *gSceneCpuDynamicActor,
		SCENE_CPU_OGC_SANDWICH_BOX_HALF_EXTENTS);
}

static bool sampleSceneCpuOgcSandwichMetrics()
{
	if(!gScene)
		return false;
	OgcSandwichFrameSample sample;
	if(!gSceneCpuOgcSandwichMonitor.sample(
			*gScene, gSceneCpuVolumeHealthMonitor, sample))
		return false;
	if(sample.generatedRigidContacts)
	{
		gMetrics.rigidContactFrames++;
		gMetrics.maxRigidContacts = PxMax(gMetrics.maxRigidContacts,
			static_cast<PxU32>(sample.generatedRigidContacts));
	}
	const OgcSandwichMetrics& metrics =
		gSceneCpuOgcSandwichMonitor.getMetrics();
	gMetrics.sceneDynamicMinY = PxMin(
		gMetrics.sceneDynamicMinY, sample.boxPosition.y);
	gMetrics.sceneDynamicFinalY = sample.boxPosition.y;
	gMetrics.sceneDynamicMaxDrop = PxMax(gMetrics.sceneDynamicMaxDrop,
		metrics.initialBoxPosition.y - sample.boxPosition.y);
	gMetrics.sceneDynamicMaxDownSpeed = PxMax(gMetrics.sceneDynamicMaxDownSpeed,
		PxMax(-sample.boxVelocity.y, 0.0f));
	return true;
}

static bool getSceneCpuVolumeMassAndRmsRadius(
	PxDeformableVolume& volume, PxReal& outMass, PxReal& outRmsRadius)
{
	const PxTetrahedronMesh* const simulationMesh =
		volume.getSimulationMesh();
	const PxVec4* const positions = volume.getSimPositionInvMassBufferH();
	if(!simulationMesh || !positions)
		return false;

	outMass = 0.0f;
	PxVec3 centroid(0.0f);
	for(PxU32 vertexIndex = 0;
		vertexIndex < simulationMesh->getNbVertices(); ++vertexIndex)
	{
		const PxReal invMass = positions[vertexIndex].w;
		const PxVec3 position = positions[vertexIndex].getXYZ();
		if(invMass <= 0.0f || !PxIsFinite(invMass) || !position.isFinite())
			return false;
		const PxReal mass = 1.0f / invMass;
		if(!PxIsFinite(mass) || mass <= 0.0f)
			return false;
		centroid += position * mass;
		outMass += mass;
	}
	if(!PxIsFinite(outMass) || outMass <= 0.0f)
		return false;
	centroid *= 1.0f / outMass;

	PxReal weightedRadiusSq = 0.0f;
	for(PxU32 vertexIndex = 0;
		vertexIndex < simulationMesh->getNbVertices(); ++vertexIndex)
	{
		const PxReal mass = 1.0f / positions[vertexIndex].w;
		const PxVec3 offset = positions[vertexIndex].getXYZ() - centroid;
		weightedRadiusSq += mass * offset.magnitudeSquared();
	}
	weightedRadiusSq *= 1.0f / outMass;
	if(!PxIsFinite(weightedRadiusSq) || weightedRadiusSq <= 0.0f)
		return false;
	outRmsRadius = PxSqrt(weightedRadiusSq);
	return PxIsFinite(outRmsRadius) && outRmsRadius > 0.0f;
}

static bool sampleSceneCpuGroundEmbeddedTetProbeMetrics()
{
	SceneCpuGroundEmbeddedTetProbeMetrics& metrics =
		gSceneCpuGroundEmbeddedTetProbeMetrics;
	if(!gScene || !gSceneCpuVolume)
		return false;
	GroundEmbeddedTetProbeFrameSample sample;
	if(!sampleGroundEmbeddedTetProbe(
			*gScene, *gSceneCpuVolume, gMetrics.completedFrames,
			metrics, sample))
		return false;
	if(sample.generatedGroundContacts)
	{
		gMetrics.groundContactFrames++;
		gMetrics.maxGroundContacts = PxMax(gMetrics.maxGroundContacts,
			static_cast<PxU32>(sample.generatedGroundContacts));
	}
	if(sample.generatedRigidContacts)
	{
		gMetrics.rigidContactFrames++;
		gMetrics.maxRigidContacts = PxMax(gMetrics.maxRigidContacts,
			static_cast<PxU32>(sample.generatedRigidContacts));
	}
	if(sample.generatedSoftContacts)
	{
		gMetrics.softContactFrames++;
		gMetrics.maxSoftContacts = PxMax(gMetrics.maxSoftContacts,
			static_cast<PxU32>(sample.generatedSoftContacts));
	}
	return true;
}

static bool sampleSceneCpuSphereLongRollContactPhaseMetrics()
{
	if(!gScene)
		return false;
	SoftContactPhaseFrameSample sample;
	if(!gSceneCpuSoftContactPhaseMonitor.sample(
			*gScene, gMetrics.completedFrames, sample))
		return false;
	// Keep the common gate counters truthful as well as maintaining the
	// contact-phase baseline below.  The dedicated phase telemetry is sampled
	// after fetchResults(), so it is the sole consumer of these per-frame Scene
	// collision statistics for the two sphere controls.
	if(sample.generatedGroundContacts)
	{
		gMetrics.groundContactFrames++;
		gMetrics.maxGroundContacts = PxMax(
			gMetrics.maxGroundContacts,
			static_cast<PxU32>(sample.generatedGroundContacts));
	}
	if(sample.generatedSoftContacts)
	{
		gMetrics.softContactFrames++;
		gMetrics.maxSoftContacts = PxMax(
			gMetrics.maxSoftContacts,
			static_cast<PxU32>(sample.generatedSoftContacts));
	}
	return true;
}

static bool sampleSceneCpuSoftSoftTorqueMetrics()
{
	if(!gSceneCpuVolume || !gScene)
		return false;
	SoftSoftTorqueFrameSample sample;
	if(!gSceneCpuSoftSoftTorqueMonitor.sample(
			*gScene, gMetrics.completedFrames,
			SCENE_CPU_SOFT_SOFT_TORQUE_MIN_ANGULAR_MOMENTUM,
			SCENE_CPU_SOFT_SOFT_TORQUE_MIN_ANGULAR_SPEED, sample))
		return false;
	if(sample.generatedGroundContacts)
	{
		gMetrics.groundContactFrames++;
		gMetrics.maxGroundContacts = PxMax(
			gMetrics.maxGroundContacts,
			static_cast<PxU32>(sample.generatedGroundContacts));
	}
	if(sample.generatedRigidContacts)
	{
		gMetrics.rigidContactFrames++;
		gMetrics.maxRigidContacts = PxMax(
			gMetrics.maxRigidContacts,
			static_cast<PxU32>(sample.generatedRigidContacts));
	}
	if(sample.generatedSoftContacts)
	{
		gMetrics.softContactFrames++;
		gMetrics.maxSoftContacts = PxMax(
			gMetrics.maxSoftContacts,
			static_cast<PxU32>(sample.generatedSoftContacts));
	}
	if(sample.driverBoundsFinite)
		gMetrics.sceneBoundsFinite = 1;
	if(sample.targetBoundsFinite)
		gMetrics.sceneSecondVolumeBoundsFinite = 1;
	return true;
}

static bool isSceneCpuVisualSphereLongRunBounded()
{
	return isRotationLongRunBounded(
		gSceneCpuVisualSphereRotationMetrics, gMetrics.completedFrames,
		SCENE_CPU_VISUAL_SPHERE_LONG_RUN_MIN_FRAMES,
		SCENE_CPU_VISUAL_SPHERE_MAX_LATE_SPEED_FLOOR,
		SCENE_CPU_VISUAL_SPHERE_MAX_LATE_SPEED_RATIO);
}

static bool isSceneCpuSphereLongRollRegressionPassed()
{
	if(gMetrics.completedFrames <
		SCENE_CPU_SPHERE_ROLL_REGRESSION_MIN_FRAMES)
		return true;
	const RotationMetrics& metrics =
		gSceneCpuVisualSphereRotationMetrics;
	if(metrics.windowSampleCount == 0)
		return false;
	const PxF64 windowMean = metrics.windowAngularSpeedSum /
		PxF64(metrics.windowSampleCount);
	return windowMean <=
			SCENE_CPU_SPHERE_ROLL_MAX_WINDOW_MEAN_SPEED &&
		metrics.finalAngularSpeed <=
			SCENE_CPU_SPHERE_ROLL_MAX_FINAL_SPEED;
}

static bool isSceneCpuSphereRollingKinematicsValid()
{
	const RotationMetrics& metrics =
		gSceneCpuVisualSphereRotationMetrics;
	// Headless output is emitted after cleanup clears the non-owning Volume
	// pointer and initialized flag.  The sampled values remain authoritative,
	// so validate their explicit finite/episode witnesses rather than lifetime
	// state that is intentionally false by print time.
	return isRollingKinematicsValid(
		metrics, SCENE_CPU_SPHERE_ROLL_MIN_ORIENTATION_CHANGE,
		SCENE_CPU_SPHERE_ROLL_MIN_GENERATED_ANGULAR_SPEED,
		SCENE_CPU_SPHERE_ROLL_MAX_RIGID_SLIP_SPEED);
}

static bool initializeSceneCpuVolumeRestStates(
	PxU32 expectedVolumeCount = 5)
{
	if(expectedVolumeCount == 0 || expectedVolumeCount > 5)
		return false;
	PxDeformableVolume* volumes[5] =
	{
		gSceneCpuVolume,
		gSceneCpuSecondVolume,
		gSceneCpuTaskGraphExtraVolumes.size() > 0
			? gSceneCpuTaskGraphExtraVolumes[0] : NULL,
		gSceneCpuTaskGraphExtraVolumes.size() > 1
			? gSceneCpuTaskGraphExtraVolumes[1] : NULL,
		gSceneCpuTaskGraphExtraVolumes.size() > 2
			? gSceneCpuTaskGraphExtraVolumes[2] : NULL
	};
	return gSceneCpuVolumeHealthMonitor.initialize(
		volumes, expectedVolumeCount);
}

static bool sampleSceneCpuVolumeHealth()
{
	VolumeHealthSample sample;
	if(!gSceneCpuVolumeHealthMonitor.sample(
			gMetrics.completedFrames, sample))
		return false;
	gMetrics.nonFiniteParticleSamples += sample.nonFiniteParticleSamples;
	gMetrics.invertedElementSamples += sample.invertedElementSamples;
	gMetrics.invertedBodiesMask |= sample.invertedBodiesMask;
	if(gMetrics.firstInversionFrame == PX_MAX_U32 &&
		sample.firstInversionBody != PX_MAX_U32)
	{
		gMetrics.firstInversionFrame = gMetrics.completedFrames;
		gMetrics.firstInversionBody = sample.firstInversionBody;
		gMetrics.firstInversionElement = sample.firstInversionElement;
	}
	gMetrics.minY = PxMin(gMetrics.minY, sample.minY);
	gMetrics.maxY = PxMax(gMetrics.maxY, sample.maxY);
	gMetrics.maxParticleSpeed = PxMax(
		gMetrics.maxParticleSpeed, sample.maxParticleSpeed);
	gMetrics.minDetF = PxMin(gMetrics.minDetF, sample.minDetF);
	gMetrics.maxDetF = PxMax(gMetrics.maxDetF, sample.maxDetF);
	gMetrics.minBodyVolumeRatio = PxMin(
		gMetrics.minBodyVolumeRatio, sample.minBodyVolumeRatio);
	gMetrics.maxBodyVolumeRatio = PxMax(
		gMetrics.maxBodyVolumeRatio, sample.maxBodyVolumeRatio);
	return true;
}

static bool createSceneCpuVisualVolume(
	PxDeformableVolumeMesh& volumeMesh,
	const PxTransform& pose, const PxVec3& scale,
	PxReal density, PxDeformableVolume*& volume)
{
	volume = NULL;
	if(!gScene || !gSceneCpuVolumeMaterial || !pose.isValid() ||
		!scale.isFinite() || scale.x <= 0.0f || scale.y <= 0.0f ||
		scale.z <= 0.0f || density <= 0.0f)
		return false;
	PxDeformableVolume* createdVolume = gPhysics->createDeformableVolume(
		PxDeformableVolumeBackend::eCPU_AVBD);
	if(!createdVolume ||
		createdVolume->getDeformableVolumeBackend() !=
			PxDeformableVolumeBackend::eCPU_AVBD ||
		createdVolume->getCudaContextManager() != NULL)
	{
		PX_RELEASE(createdVolume);
		return false;
	}

	const PxShapeFlags shapeFlags =
		PxShapeFlag::eVISUALIZATION |
		PxShapeFlag::eSCENE_QUERY_SHAPE |
		PxShapeFlag::eSIMULATION_SHAPE;
	PxDeformableVolumeMaterial* material = gSceneCpuVolumeMaterial;
	PxShape* shape = gPhysics->createShape(
		PxTetrahedronMeshGeometry(volumeMesh.getCollisionMesh()),
		&material, 1, true, shapeFlags);
	if(!shape)
	{
		createdVolume->release();
		return false;
	}
	const bool shapeAttached = createdVolume->attachShape(*shape);
	shape->release();
	if(!shapeAttached || !createdVolume->attachSimulationMesh(
			*volumeMesh.getSimulationMesh(),
			*volumeMesh.getDeformableVolumeAuxData()))
	{
		createdVolume->release();
		return false;
	}

	PxVec4* simPositionInvMass =
		createdVolume->getSimPositionInvMassBufferH();
	PxVec4* simVelocity = createdVolume->getSimVelocityBufferH();
	PxVec4* collisionPositionInvMass =
		createdVolume->getPositionInvMassBufferH();
	PxVec4* collisionRestPosition =
		createdVolume->getRestPositionBufferH();
	const PxTetrahedronMesh* simulationMesh =
		createdVolume->getSimulationMesh();
	if(!simPositionInvMass || !simVelocity ||
		!collisionPositionInvMass || !collisionRestPosition ||
		!simulationMesh)
	{
		createdVolume->release();
		return false;
	}

	const PxVec3* simulationVertices = simulationMesh->getVertices();
	const PxU32 simulationVertexCount = simulationMesh->getNbVertices();
	PxReal* cookedInvMass = createdVolume->getDeformableVolumeAuxData()->
		getGridModelInvMass();
	for(PxU32 vertexIndex = 0;
		vertexIndex < simulationVertexCount; ++vertexIndex)
	{
		const PxVec3 local = simulationVertices[vertexIndex];
		const PxVec3 scaled(
			local.x * scale.x, local.y * scale.y, local.z * scale.z);
		const PxReal invMass = cookedInvMass
			? PxMax(cookedInvMass[vertexIndex], 0.0f) : 1.0f;
		simPositionInvMass[vertexIndex] =
			PxVec4(pose.transform(scaled), invMass);
		simVelocity[vertexIndex] = PxVec4(0.0f, 0.0f, 0.0f, invMass);
	}
	PxDeformableVolumeExt::updateMass(
		*createdVolume, density, 50.0f, simPositionInvMass);
	for(PxU32 vertexIndex = 0;
		vertexIndex < simulationVertexCount; ++vertexIndex)
		simVelocity[vertexIndex].w = simPositionInvMass[vertexIndex].w;
	PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
		*createdVolume, simPositionInvMass, collisionPositionInvMass);
	const PxU32 collisionVertexCount =
		createdVolume->getCollisionMesh()->getNbVertices();
	for(PxU32 vertexIndex = 0;
		vertexIndex < collisionVertexCount; ++vertexIndex)
		collisionRestPosition[vertexIndex] =
			collisionPositionInvMass[vertexIndex];
	createdVolume->markDirty(PxDeformableVolumeDataFlag::eALL);
	// Keep iteration count independent from mesh resolution. The visual case
	// now uses a genuinely subdivided simulation mesh; lowering the topology is
	// not an acceptable performance shortcut.
	createdVolume->setSolverIterationCounts(
		SCENE_CPU_VISUAL_POSITION_ITERATIONS, 1);
	// All of the public collision-showcase bodies participate in the same
	// impact chain.  Give each one the legacy total damping budget instead of
	// leaving the impactor at the body-core default, which would silently drain
	// its tangential momentum before it reaches the measured sphere.
	createdVolume->setLinearDamping(
		SCENE_CPU_VISUAL_SPHERE_LINEAR_DAMPING);
	gScene->addActor(*createdVolume);
	if(createdVolume->getScene() != gScene)
	{
		createdVolume->release();
		return false;
	}
	volume = createdVolume;
	return true;
}

static void configureSceneCpuVisualSphereDamping(
	PxDeformableVolume& volume)
{
	// Settling damping is conditional on low particle velocity, unlike linear
	// damping which is applied throughout the impact and rotation event.  This
	// preserves collision-generated spin while still damping the long-run rest
	// state without extra solver iterations.
	volume.setSettlingThreshold(
		SCENE_CPU_VISUAL_SPHERE_SETTLING_THRESHOLD);
	volume.setSettlingDamping(
		SCENE_CPU_VISUAL_SPHERE_SETTLING_DAMPING);
}

// Unlike the bounded dynamic-rigid regression helpers above, this is a free
// rigid body: all translational and angular degrees of freedom remain enabled
// so soft impacts are directly visible in the showcase.
static bool addSceneCpuVisualDynamicBox()
{
	PX_ASSERT(!gSceneCpuDynamicActor);
	if(!gPhysics || !gScene || !gMaterial)
		return false;

	PxRigidDynamic* actor = gPhysics->createRigidDynamic(PxTransform(
		SCENE_CPU_VISUAL_DYNAMIC_BOX_CENTER,
		SCENE_CPU_VISUAL_DYNAMIC_BOX_ORIENTATION));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
			*actor,
			PxBoxGeometry(SCENE_CPU_VISUAL_DYNAMIC_BOX_HALF_EXTENTS),
			*gMaterial) ||
		!PxRigidBodyExt::setMassAndUpdateInertia(
			*actor, SCENE_CPU_VISUAL_DYNAMIC_BOX_MASS))
	{
		actor->release();
		return false;
	}
	// This is an awake, gravity-driven rigid body.  The visual interaction is
	// deliberately owned by the current-pose OGC path; do not enable a swept
	// CCD substitute for this diagnostic scene.
	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, false);
	actor->setRigidBodyFlag(
		PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, false);
	actor->setLinearDamping(
		SCENE_CPU_VISUAL_DYNAMIC_BOX_LINEAR_DAMPING);
	actor->setSolverIterationCounts(
		SCENE_CPU_VISUAL_POSITION_ITERATIONS, 1);
	// Keep every rigid DOF free.  The post-AL recovery is responsible for true
	// current-pose overlap; this visual case must not hide a soft/rigid response
	// by constraining the box's lateral or angular motion.
	gScene->addActor(*actor);
	if(actor->getScene() != gScene)
	{
		actor->release();
		return false;
	}
	actor->setLinearVelocity(
		SCENE_CPU_VISUAL_DYNAMIC_BOX_INITIAL_LINEAR_VELOCITY);
	actor->wakeUp();
	gSceneCpuDynamicActor = actor;
	gSceneCpuDynamicInitialY = SCENE_CPU_VISUAL_DYNAMIC_BOX_CENTER.y;
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping =
		actor->isSleeping() ? 1u : 0u;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
}

static bool initSceneCpuVolumeVisualShowcase()
{
	PX_ASSERT(gSceneCpuTaskGraphExtraVolumes.empty());
	PX_ASSERT(gSceneCpuVisualVolumeMeshes.empty());
	PX_ASSERT(gSceneCpuVolumeHealthMonitor.empty());
	PxRigidStatic* ground = PxCreatePlane(
		*gPhysics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *gMaterial);
	if(!ground)
		return false;
	gScene->addActor(*ground);
	if(!addSceneStaticBox(
			SCENE_CPU_VISUAL_STATIC_PEDESTAL_CENTER,
			SCENE_CPU_VISUAL_STATIC_PEDESTAL_HALF_EXTENTS))
		return false;

	gSceneCpuVolumeMaterial =
		gPhysics->createDeformableVolumeMaterial(
			SCENE_CPU_VISUAL_YOUNGS_MODULUS,
			SCENE_CPU_VISUAL_POISSONS_RATIO,
			getSceneCpuVisualMaterialDynamicFriction(),
			SCENE_CPU_VISUAL_MATERIAL_DAMPING);
	if(!gSceneCpuVolumeMaterial)
		return false;
	gSceneCpuVolumeMaterial->setMaterialModel(
		PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
	printf("[AVBD_SCENE_VOLUME_MATERIAL] dynamicFriction=%.6g\n",
		gSceneCpuVolumeMaterial->getDynamicFriction());

	PxArray<PxVec3> surfaceVertices;
	PxArray<PxU32> surfaceTriangles;
	// Match the pre-public-Volume showcase baseline.  This is the authoritative
	// collision/render surface, not a cosmetic render-only tessellation.
	static const PxU32 cubeCollisionSubdivisions = 4;
	static const PxU32 cubeSimulationVoxels = 5;
	static const PxU32 sphereSimulationVoxels = 6;
	static const PxU32 coneRingSegments = 32;
	static const PxU32 coneHeightSegments = 3;
	static const PxU32 coneSimulationVoxels = 8;
	createSubdividedCubeSurface(
		surfaceVertices, surfaceTriangles, PxVec3(0.0f), 2.0f,
		cubeCollisionSubdivisions);
	const PxU32 cubeSurfaceTriangleCount = surfaceTriangles.size() / 3;
	PxDeformableVolumeMesh* cubeMesh = cookSceneCpuVisualVolumeMesh(
		surfaceVertices, surfaceTriangles, cubeSimulationVoxels);
	meshgenerator::createSphere(
		surfaceVertices, surfaceTriangles,
		PxVec3(0.0f), 1.8f, 0.8f);
	const PxU32 sphereSurfaceTriangleCount = surfaceTriangles.size() / 3;
	PxDeformableVolumeMesh* sphereMesh = cookSceneCpuVisualVolumeMesh(
		surfaceVertices, surfaceTriangles, sphereSimulationVoxels);
	// This surface is the public collision boundary. The voxel mesh cooked
	// behind it is simulation-only, so neither rendering nor contacts acquire
	// the stepped voxel silhouette that the old low-level cone used.
	createLayeredConeSurface(
		surfaceVertices, surfaceTriangles, PxVec3(0.0f), 0.8f, 3.0f,
		coneRingSegments, coneHeightSegments);
	const PxU32 coneSurfaceVertexCount = surfaceVertices.size();
	const PxU32 coneSurfaceTriangleCount = surfaceTriangles.size() / 3;
	PxDeformableVolumeMesh* coneMesh = cookSceneCpuVisualVolumeMesh(
		surfaceVertices, surfaceTriangles, coneSimulationVoxels);
	if(!cubeMesh || !sphereMesh || !coneMesh ||
		cubeMesh->getCollisionMesh() == cubeMesh->getSimulationMesh() ||
		sphereMesh->getCollisionMesh() == sphereMesh->getSimulationMesh() ||
		coneMesh->getCollisionMesh() == coneMesh->getSimulationMesh())
	{
		PX_RELEASE(cubeMesh);
		PX_RELEASE(sphereMesh);
		PX_RELEASE(coneMesh);
		return false;
	}
	gSceneCpuVisualVolumeMeshes.pushBack(cubeMesh);
	gSceneCpuVisualVolumeMeshes.pushBack(sphereMesh);
	gSceneCpuVisualVolumeMeshes.pushBack(coneMesh);
	// Give the showcase bodies visible 3D attitudes. The sphere intentionally
	// remains unrotated: its visual symmetry would hide the change and it is the
	// shared long-roll reference body. The two soft cubes deliberately share
	// the rigid box's yaw-only attitude, giving the OGC-only cluster broad,
	// horizontal faces rather than a one-vertex grazing coincidence.
	const PxQuat primaryCubeOrientation(
		-0.62f, PxVec3(0.18f, 0.22f, 0.96f).getNormalized());
	const PxQuat coneOrientation(
		0.48f, PxVec3(0.35f, 0.82f, 0.45f).getNormalized());
	const PxQuat tiltedCubeOrientation(
		SCENE_CPU_VISUAL_DYNAMIC_BOX_ORIENTATION);
	const PxQuat followerCubeOrientation(
		SCENE_CPU_VISUAL_DYNAMIC_BOX_ORIENTATION);

	if(!createSceneCpuVisualVolume(
			*cubeMesh,
			PxTransform(
				PxVec3(SCENE_CPU_VISUAL_PRIMARY_CUBE_START_X, 8.0f, 0.0f),
				primaryCubeOrientation),
			PxVec3(1.8f, 0.65f, 0.9f), 160.0f,
			gSceneCpuVolume) ||
		!createSceneCpuVisualVolume(
			// The cube crosses this lane with a deliberate finite lateral offset.
			*sphereMesh, PxTransform(PxVec3(
				SCENE_CPU_VISUAL_PRIMARY_SPHERE_START_X, 2.0f, 0.0f)),
			PxVec3(1.0f), 130.0f, gSceneCpuSecondVolume))
		return false;
	if(!initializeSceneCpuVisualRotationMetrics(
			gSceneCpuVisualPrimaryCubeRotationMonitor,
			*gSceneCpuVolume) ||
		!initializeSceneCpuVisualRotationMetrics(
			gSceneCpuVisualSphereRotationMonitor,
			*gSceneCpuSecondVolume))
		return false;
	configureSceneCpuVisualSphereDamping(*gSceneCpuSecondVolume);
	// Make this a finite glancing collision instead of a permanent eccentric
	// stack: the cube crosses the sphere's lane and then clears it. The sphere
	// still begins at rest, so its angular momentum remains collision-generated.
	if(!setSceneCpuVolumeVelocity(
			*gSceneCpuVolume, PxVec3(2.5f, 0.0f, 0.0f)))
		return false;
	PxDeformableVolume* coneVolume = NULL;
	PxDeformableVolume* tiltedVolume = NULL;
	PxDeformableVolume* followerVolume = NULL;
	if(!createSceneCpuVisualVolume(
			*coneMesh, PxTransform(PxVec3(3.2f, 11.0f, 1.2f),
				coneOrientation),
			PxVec3(1.0f), 160.0f, coneVolume))
		return false;
	gSceneCpuTaskGraphExtraVolumes.pushBack(coneVolume);
	if(!setSceneCpuVolumeVelocity(
			*coneVolume,
			SCENE_CPU_VISUAL_CONE_INITIAL_LINEAR_VELOCITY))
		return false;
	if(!createSceneCpuVisualVolume(
			*cubeMesh,
			PxTransform(
				SCENE_CPU_VISUAL_UPPER_SOFT_JAW_CENTER,
				tiltedCubeOrientation),
			PxVec3(1.7f, 0.7f, 0.9f), 160.0f, tiltedVolume))
		return false;
	gSceneCpuTaskGraphExtraVolumes.pushBack(tiltedVolume);
	tiltedVolume->setLinearDamping(
		SCENE_CPU_VISUAL_JAW_LINEAR_DAMPING);
	if(!initializeSceneCpuVisualRotationMetrics(
			gSceneCpuVisualRotationMonitor, *tiltedVolume))
		return false;
	// This is the upper yellow soft jaw. It begins above the rigid with a true
	// mesh gap inside the current-pose OGC shell, without swept rows.
	if(!setSceneCpuVolumeVelocity(
			*tiltedVolume,
			SCENE_CPU_VISUAL_TILTED_CUBE_INITIAL_LINEAR_VELOCITY))
		return false;
	if(!createSceneCpuVisualVolume(
			*cubeMesh,
			PxTransform(
				SCENE_CPU_VISUAL_LOWER_SOFT_JAW_CENTER,
				followerCubeOrientation),
			PxVec3(1.7f, 0.7f, 0.9f), 160.0f, followerVolume))
		return false;
	gSceneCpuTaskGraphExtraVolumes.pushBack(followerVolume);
	followerVolume->setLinearDamping(
		SCENE_CPU_VISUAL_JAW_LINEAR_DAMPING);
	// This is the lower magenta soft jaw. Both soft cubes and the free rigid are
	// fully dynamic under the same gravity, without an imposed closing velocity.
	if(!setSceneCpuVolumeVelocity(
			*followerVolume,
			SCENE_CPU_VISUAL_FOLLOWER_CUBE_INITIAL_LINEAR_VELOCITY))
		return false;
	// Keep the visual bodies on the current-pose OGC path.  The dynamic box is
	// placed so its relative motion is small enough for discrete OGC detection;
	// enabling speculative CCD here would append swept rows instead of exposing
	// the penetration path this showcase is intended to display.
	PxDeformableVolume* const visualVolumes[] =
	{
		gSceneCpuVolume,
		gSceneCpuSecondVolume,
		coneVolume,
		tiltedVolume,
		followerVolume
	};
	for(PxU32 volumeIndex = 0;
		volumeIndex < PX_ARRAY_SIZE(visualVolumes); ++volumeIndex)
	{
		// Make the free-fall contract explicit rather than relying on the
		// default actor flags.  The box below is configured the same way.
		visualVolumes[volumeIndex]->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, false);
		visualVolumes[volumeIndex]->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, false);
		if(visualVolumes[volumeIndex]->getDeformableBodyFlags().isSet(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD))
			return false;
	}
	// Place the free rigid vertically between the two high soft jaws. Both
	// current-pose OGC interfaces start with positive mesh gaps; the lower
	// primary cube/sphere soft/soft lane is independent.
	if(!addSceneCpuVisualDynamicBox())
		return false;
	if(!initializeSceneCpuVisualInteractionMetrics(
			*tiltedVolume, *followerVolume))
		return false;
	if(!initializeSceneCpuVolumeRestStates())
		return false;

	gMetrics.initialized = 1;
	gMetrics.sceneActorCreated = 1;
	gMetrics.sceneShapeAttached = 1;
	gMetrics.sceneSimulationMeshAttached = 1;
	gMetrics.sceneHostBuffersInitialized = 1;
	gMetrics.sceneActorAdded = 1;
	gMetrics.softBodies = 5;
	gMetrics.particles =
		3 * cubeMesh->getSimulationMesh()->getNbVertices() +
		sphereMesh->getSimulationMesh()->getNbVertices() +
		coneMesh->getSimulationMesh()->getNbVertices();
	gMetrics.tetElements =
		3 * cubeMesh->getSimulationMesh()->getNbTetrahedrons() +
		sphereMesh->getSimulationMesh()->getNbTetrahedrons() +
		coneMesh->getSimulationMesh()->getNbTetrahedrons();
	gMetrics.surfaceTriangles =
		3 * cubeSurfaceTriangleCount + sphereSurfaceTriangleCount +
		coneSurfaceTriangleCount;
	gSceneCpuVolumeInitialCentroidY = getSceneCpuVolumeCentroidY();
	gSceneCpuSecondVolumeInitialCentroidY =
		getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes = gScene->getNbDeformableVolumes();
	printf(
		"%s visual showcase: %u public CPU AVBD volumes\n",
		AVBD_VOLUME_SNIPPET_NAME, gMetrics.sceneDeformableVolumes);
	printf(
		"  cube: input=%u triangles (%ux%u face grid), "
		"collision=%u vertices/%u tets, simulation=%u vertices/%u tets "
		"(voxels=%u)\n",
		cubeSurfaceTriangleCount, cubeCollisionSubdivisions,
		cubeCollisionSubdivisions,
		cubeMesh->getCollisionMesh()->getNbVertices(),
		cubeMesh->getCollisionMesh()->getNbTetrahedrons(),
		cubeMesh->getSimulationMesh()->getNbVertices(),
		cubeMesh->getSimulationMesh()->getNbTetrahedrons(),
		cubeSimulationVoxels);
	printf(
		"  sphere: input=%u triangles, collision=%u vertices/%u tets, "
		"simulation=%u vertices/%u tets (voxels=%u)\n",
		sphereSurfaceTriangleCount,
		sphereMesh->getCollisionMesh()->getNbVertices(),
		sphereMesh->getCollisionMesh()->getNbTetrahedrons(),
		sphereMesh->getSimulationMesh()->getNbVertices(),
		sphereMesh->getSimulationMesh()->getNbTetrahedrons(),
		sphereSimulationVoxels);
	printf(
		"  cone: input=%u vertices/%u triangles (%u rings x %u height "
		"segments), collision=%u vertices/%u tets, simulation=%u "
		"vertices/%u tets (voxels=%u)\n",
		coneSurfaceVertexCount, coneSurfaceTriangleCount,
		coneRingSegments, coneHeightSegments,
		coneMesh->getCollisionMesh()->getNbVertices(),
		coneMesh->getCollisionMesh()->getNbTetrahedrons(),
		coneMesh->getSimulationMesh()->getNbVertices(),
		coneMesh->getSimulationMesh()->getNbTetrahedrons(),
		coneSimulationVoxels);
	return gMetrics.sceneDeformableVolumes == 5;
}

static bool initSceneCpuVolumeOgcSandwich()
{
	PX_ASSERT(gSceneCpuTaskGraphExtraVolumes.empty());
	PX_ASSERT(gSceneCpuVisualVolumeMeshes.empty());
	PX_ASSERT(gSceneCpuVolumeHealthMonitor.empty());
	gSceneCpuVolumeMaterial = gPhysics->createDeformableVolumeMaterial(
		SCENE_CPU_VISUAL_YOUNGS_MODULUS,
		SCENE_CPU_VISUAL_POISSONS_RATIO,
		getSceneCpuVisualMaterialDynamicFriction(),
		SCENE_CPU_VISUAL_MATERIAL_DAMPING);
	if(!gSceneCpuVolumeMaterial)
		return false;
	gSceneCpuVolumeMaterial->setMaterialModel(
		PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);

	PxArray<PxVec3> surfaceVertices;
	PxArray<PxU32> surfaceTriangles;
	static const PxU32 cubeCollisionSubdivisions = 4;
	static const PxU32 cubeSimulationVoxels = 5;
	createSubdividedCubeSurface(surfaceVertices, surfaceTriangles,
		PxVec3(0.0f), 2.0f, cubeCollisionSubdivisions);
	const PxU32 surfaceTriangleCount = surfaceTriangles.size() / 3;
	PxDeformableVolumeMesh* cubeMesh = cookSceneCpuVisualVolumeMesh(
		surfaceVertices, surfaceTriangles, cubeSimulationVoxels);
	if(!cubeMesh ||
		cubeMesh->getCollisionMesh() == cubeMesh->getSimulationMesh())
	{
		PX_RELEASE(cubeMesh);
		return false;
	}
	gSceneCpuVisualVolumeMeshes.pushBack(cubeMesh);
	if(!createSceneCpuVisualVolume(*cubeMesh,
			PxTransform(SCENE_CPU_OGC_SANDWICH_UPPER_JAW_CENTER),
			PxVec3(1.7f, 0.7f, 0.9f), 160.0f, gSceneCpuVolume) ||
		!createSceneCpuVisualVolume(*cubeMesh,
			PxTransform(SCENE_CPU_OGC_SANDWICH_LOWER_JAW_CENTER),
			PxVec3(1.7f, 0.7f, 0.9f), 160.0f,
			gSceneCpuSecondVolume))
		return false;
	const PxDeformableVolume* const jaws[] =
	{
		gSceneCpuVolume, gSceneCpuSecondVolume
	};
	for(PxU32 jawIndex = 0; jawIndex < PX_ARRAY_SIZE(jaws); ++jawIndex)
	{
		PxDeformableVolume* const jaw = const_cast<PxDeformableVolume*>(
			jaws[jawIndex]);
		jaw->setLinearDamping(SCENE_CPU_VISUAL_JAW_LINEAR_DAMPING);
		jaw->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, false);
		jaw->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, false);
		if(jaw->getDeformableBodyFlags().isSet(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD))
			return false;
	}
	if(!setSceneCpuVolumeVelocity(*gSceneCpuVolume,
			SCENE_CPU_OGC_SANDWICH_UPPER_JAW_VELOCITY) ||
		!setSceneCpuVolumeVelocity(*gSceneCpuSecondVolume,
			SCENE_CPU_OGC_SANDWICH_LOWER_JAW_VELOCITY))
		return false;

	PxRigidDynamic* box = gPhysics->createRigidDynamic(PxTransform(
		SCENE_CPU_OGC_SANDWICH_BOX_CENTER));
	if(!box || !PxRigidActorExt::createExclusiveShape(*box,
			PxBoxGeometry(SCENE_CPU_OGC_SANDWICH_BOX_HALF_EXTENTS),
			*gMaterial) || !PxRigidBodyExt::setMassAndUpdateInertia(*box,
			SCENE_CPU_OGC_SANDWICH_BOX_MASS))
	{
		PX_RELEASE(box);
		return false;
	}
	box->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, false);
	box->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, false);
	box->setLinearDamping(SCENE_CPU_VISUAL_DYNAMIC_BOX_LINEAR_DAMPING);
	box->setSolverIterationCounts(SCENE_CPU_VISUAL_POSITION_ITERATIONS, 1);
	gScene->addActor(*box);
	if(box->getScene() != gScene)
	{
		box->release();
		return false;
	}
	box->setLinearVelocity(PxVec3(0.0f));
	box->wakeUp();
	gSceneCpuDynamicActor = box;
	gSceneCpuDynamicInitialY = SCENE_CPU_OGC_SANDWICH_BOX_CENTER.y;
	if(!initializeSceneCpuOgcSandwichMetrics(
			*gSceneCpuVolume, *gSceneCpuSecondVolume) ||
		!initializeSceneCpuVolumeRestStates(2))
		return false;

	gMetrics.initialized = 1;
	gMetrics.sceneActorCreated = 1;
	gMetrics.sceneShapeAttached = 1;
	gMetrics.sceneSimulationMeshAttached = 1;
	gMetrics.sceneHostBuffersInitialized = 1;
	gMetrics.sceneActorAdded = 1;
	gMetrics.softBodies = 2;
	gMetrics.particles = 2 * cubeMesh->getSimulationMesh()->getNbVertices();
	gMetrics.tetElements =
		2 * cubeMesh->getSimulationMesh()->getNbTetrahedrons();
	gMetrics.surfaceTriangles = 2 * surfaceTriangleCount;
	gSceneCpuVolumeInitialCentroidY = getSceneCpuVolumeCentroidY();
	gSceneCpuSecondVolumeInitialCentroidY =
		getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
	gMetrics.sceneDynamicActorAdded = 1;
	gMetrics.sceneDynamicInitiallySleeping = box->isSleeping() ? 1u : 0u;
	gMetrics.sceneStatics = 0;
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes = gScene->getNbDeformableVolumes();
	printf("%s OGC sandwich: 2 public CPU AVBD volumes + free box\n",
		AVBD_VOLUME_SNIPPET_NAME);
	return gMetrics.sceneDeformableVolumes == 2 &&
		gMetrics.sceneDynamics == 1;
}

// Minimal public-Volume reproduction for the showcase sphere's long-horizon
// angular drift.  Only the causal chain remains: one eccentric falling cube,
// the sphere and the ground plane.  Meshes, material, transforms and launch
// velocity intentionally match the full showcase so this case can replace the
// five-volume scene in routine regression matrices.
static bool initSceneCpuVolumeSphereLongRoll()
{
	PX_ASSERT(gSceneCpuTaskGraphExtraVolumes.empty());
	PX_ASSERT(gSceneCpuVisualVolumeMeshes.empty());
	PX_ASSERT(gSceneCpuVolumeHealthMonitor.empty());
	gSceneCpuSoftContactPhaseMonitor.releaseSource();
	PxRigidStatic* ground = PxCreatePlane(
		*gPhysics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *gMaterial);
	if(!ground)
		return false;
	gScene->addActor(*ground);

	gSceneCpuVolumeMaterial =
		gPhysics->createDeformableVolumeMaterial(
			SCENE_CPU_VISUAL_YOUNGS_MODULUS,
			SCENE_CPU_VISUAL_POISSONS_RATIO,
			getSceneCpuVisualMaterialDynamicFriction(),
			SCENE_CPU_VISUAL_MATERIAL_DAMPING);
	if(!gSceneCpuVolumeMaterial)
		return false;
	gSceneCpuVolumeMaterial->setMaterialModel(
		PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
	printf("[AVBD_SCENE_VOLUME_MATERIAL] dynamicFriction=%.6g\n",
		gSceneCpuVolumeMaterial->getDynamicFriction());

	PxArray<PxVec3> surfaceVertices;
	PxArray<PxU32> surfaceTriangles;
	static const PxU32 cubeCollisionSubdivisions = 4;
	static const PxU32 cubeSimulationVoxels = 5;
	static const PxU32 sphereSimulationVoxels = 6;
	createSubdividedCubeSurface(
		surfaceVertices, surfaceTriangles, PxVec3(0.0f), 2.0f,
		cubeCollisionSubdivisions);
	const PxU32 cubeSurfaceTriangleCount = surfaceTriangles.size() / 3;
	PxDeformableVolumeMesh* cubeMesh = cookSceneCpuVisualVolumeMesh(
		surfaceVertices, surfaceTriangles, cubeSimulationVoxels);
	meshgenerator::createSphere(
		surfaceVertices, surfaceTriangles,
		PxVec3(0.0f), 1.8f, 0.8f);
	const PxU32 sphereSurfaceTriangleCount = surfaceTriangles.size() / 3;
	PxDeformableVolumeMesh* sphereMesh = cookSceneCpuVisualVolumeMesh(
		surfaceVertices, surfaceTriangles, sphereSimulationVoxels);
	if(!cubeMesh || !sphereMesh ||
		cubeMesh->getCollisionMesh() == cubeMesh->getSimulationMesh() ||
		sphereMesh->getCollisionMesh() == sphereMesh->getSimulationMesh())
	{
		PX_RELEASE(cubeMesh);
		PX_RELEASE(sphereMesh);
		return false;
	}
	gSceneCpuVisualVolumeMeshes.pushBack(cubeMesh);
	gSceneCpuVisualVolumeMeshes.pushBack(sphereMesh);

	if(!createSceneCpuVisualVolume(
			*cubeMesh,
			PxTransform(
				PxVec3(SCENE_CPU_VISUAL_PRIMARY_CUBE_START_X, 8.0f, 0.0f),
				PxQuat(-0.55f, PxVec3(0.0f, 0.0f, 1.0f))),
			PxVec3(1.8f, 0.65f, 0.9f), 160.0f,
			gSceneCpuVolume) ||
		!createSceneCpuVisualVolume(
			*sphereMesh, PxTransform(PxVec3(
				SCENE_CPU_VISUAL_PRIMARY_SPHERE_START_X, 2.0f, 0.0f)),
			PxVec3(1.0f), 130.0f, gSceneCpuSecondVolume))
		return false;
	if(!initializeSceneCpuVisualRotationMetrics(
			gSceneCpuVisualSphereRotationMonitor,
			*gSceneCpuSecondVolume))
		return false;
	const char* const collisionTelemetry =
		std::getenv("PHYSX_AVBD_COLLISION_TELEMETRY");
	if(!gSceneCpuSoftContactPhaseMonitor.initialize(
			*gSceneCpuSecondVolume,
			collisionTelemetry && collisionTelemetry[0] == '1' &&
			collisionTelemetry[1] == '\0'))
		return false;
	configureSceneCpuVisualSphereDamping(*gSceneCpuSecondVolume);
	if(!setSceneCpuVolumeVelocity(
			*gSceneCpuVolume, PxVec3(2.5f, 0.0f, 0.0f)) ||
		!initializeSceneCpuVolumeRestStates(2))
		return false;

	gMetrics.initialized = 1;
	gMetrics.sceneActorCreated = 1;
	gMetrics.sceneShapeAttached = 1;
	gMetrics.sceneSimulationMeshAttached = 1;
	gMetrics.sceneHostBuffersInitialized = 1;
	gMetrics.sceneActorAdded = 1;
	gMetrics.softBodies = 2;
	gMetrics.particles =
		cubeMesh->getSimulationMesh()->getNbVertices() +
		sphereMesh->getSimulationMesh()->getNbVertices();
	gMetrics.tetElements =
		cubeMesh->getSimulationMesh()->getNbTetrahedrons() +
		sphereMesh->getSimulationMesh()->getNbTetrahedrons();
	gMetrics.surfaceTriangles =
		cubeSurfaceTriangleCount + sphereSurfaceTriangleCount;
	gSceneCpuVolumeInitialCentroidY = getSceneCpuVolumeCentroidY();
	gSceneCpuSecondVolumeInitialCentroidY =
		getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes = gScene->getNbDeformableVolumes();
	printf(
		"%s sphere long-roll reproduction: %u public CPU AVBD volumes, "
		"%u surface triangles\n",
		AVBD_VOLUME_SNIPPET_NAME, gMetrics.sceneDeformableVolumes,
		gMetrics.surfaceTriangles);
	return gMetrics.sceneDeformableVolumes == 2;
}

// Contact-only control for the long-roll sphere.  It keeps the public cube
// and sphere meshes, material, damping and eight-sweep budget, but removes
// gravity and the plane so a horizontal, off-centre cube/sphere impact can be
// attributed to soft/soft contact alone.  This is intentionally not the
// high-friction cube/cube torque fixture below.
static bool initSceneCpuVolumeSphereSoftSoftGlancing()
{
	PX_ASSERT(gSceneCpuTaskGraphExtraVolumes.empty());
	PX_ASSERT(gSceneCpuVisualVolumeMeshes.empty());
	PX_ASSERT(gSceneCpuVolumeHealthMonitor.empty());
	gSceneCpuSoftContactPhaseMonitor.releaseSource();

	gSceneCpuVolumeMaterial =
		gPhysics->createDeformableVolumeMaterial(
			SCENE_CPU_VISUAL_YOUNGS_MODULUS,
			SCENE_CPU_VISUAL_POISSONS_RATIO,
			0.2f,
			SCENE_CPU_VISUAL_MATERIAL_DAMPING);
	if(!gSceneCpuVolumeMaterial)
		return false;
	gSceneCpuVolumeMaterial->setMaterialModel(
		PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);

	PxArray<PxVec3> surfaceVertices;
	PxArray<PxU32> surfaceTriangles;
	static const PxU32 cubeCollisionSubdivisions = 4;
	static const PxU32 cubeSimulationVoxels = 5;
	static const PxU32 sphereSimulationVoxels = 6;
	createSubdividedCubeSurface(
		surfaceVertices, surfaceTriangles, PxVec3(0.0f), 2.0f,
		cubeCollisionSubdivisions);
	const PxU32 cubeSurfaceTriangleCount = surfaceTriangles.size() / 3;
	PxDeformableVolumeMesh* cubeMesh = cookSceneCpuVisualVolumeMesh(
		surfaceVertices, surfaceTriangles, cubeSimulationVoxels);
	meshgenerator::createSphere(
		surfaceVertices, surfaceTriangles,
		PxVec3(0.0f), 1.8f, 0.8f);
	const PxU32 sphereSurfaceTriangleCount = surfaceTriangles.size() / 3;
	PxDeformableVolumeMesh* sphereMesh = cookSceneCpuVisualVolumeMesh(
		surfaceVertices, surfaceTriangles, sphereSimulationVoxels);
	if(!cubeMesh || !sphereMesh ||
		cubeMesh->getCollisionMesh() == cubeMesh->getSimulationMesh() ||
		sphereMesh->getCollisionMesh() == sphereMesh->getSimulationMesh())
	{
		PX_RELEASE(cubeMesh);
		PX_RELEASE(sphereMesh);
		return false;
	}
	gSceneCpuVisualVolumeMeshes.pushBack(cubeMesh);
	gSceneCpuVisualVolumeMeshes.pushBack(sphereMesh);

	if(!createSceneCpuVisualVolume(
			*cubeMesh,
			PxTransform(
				PxVec3(-4.5f, 1.15f, 0.0f),
				PxQuat(-0.55f, PxVec3(0.0f, 0.0f, 1.0f))),
			PxVec3(1.8f, 0.65f, 0.9f), 160.0f,
			gSceneCpuVolume) ||
		!createSceneCpuVisualVolume(
			*sphereMesh, PxTransform(PxVec3(0.0f)),
			PxVec3(1.0f), 130.0f, gSceneCpuSecondVolume))
		return false;
	if(!initializeSceneCpuVisualRotationMetrics(
			gSceneCpuVisualSphereRotationMonitor,
			*gSceneCpuSecondVolume))
		return false;
	configureSceneCpuVisualSphereDamping(*gSceneCpuSecondVolume);
	// This fixture's validity depends on its contact-phase proof.  Collect its
	// small test-only statistics unconditionally so direct invocation has the
	// same gate semantics as the headless runner.
	if(!gSceneCpuSoftContactPhaseMonitor.initialize(
			*gSceneCpuSecondVolume, true))
		return false;

	auto configureContactOnlyVolume = [](PxDeformableVolume& volume,
		const PxVec3& velocity)
	{
		volume.setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
		volume.setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		volume.setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, false);
		volume.setSleepThreshold(0.0f);
		volume.setWakeCounter(1.0f);
		volume.setMaxLinearVelocity(PX_MAX_F32);
		return setSceneCpuVolumeVelocity(volume, velocity);
	};
	if(!configureContactOnlyVolume(
			*gSceneCpuVolume, PxVec3(4.0f, 0.0f, 0.0f)) ||
		!configureContactOnlyVolume(*gSceneCpuSecondVolume, PxVec3(0.0f)) ||
		!initializeSceneCpuVolumeRestStates(2))
		return false;

	gMetrics.initialized = 1;
	gMetrics.sceneActorCreated = 1;
	gMetrics.sceneShapeAttached = 1;
	gMetrics.sceneSimulationMeshAttached = 1;
	gMetrics.sceneHostBuffersInitialized = 1;
	gMetrics.sceneActorAdded = 1;
	gMetrics.softBodies = 2;
	gMetrics.particles =
		cubeMesh->getSimulationMesh()->getNbVertices() +
		sphereMesh->getSimulationMesh()->getNbVertices();
	gMetrics.tetElements =
		cubeMesh->getSimulationMesh()->getNbTetrahedrons() +
		sphereMesh->getSimulationMesh()->getNbTetrahedrons();
	gMetrics.surfaceTriangles =
		cubeSurfaceTriangleCount + sphereSurfaceTriangleCount;
	gSceneCpuVolumeInitialCentroidY = getSceneCpuVolumeCentroidY();
	gSceneCpuSecondVolumeInitialCentroidY =
		getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes = gScene->getNbDeformableVolumes();
	printf(
		"%s sphere soft-soft glancing control: %u public CPU AVBD volumes, "
		"%u surface triangles\n",
		AVBD_VOLUME_SNIPPET_NAME, gMetrics.sceneDeformableVolumes,
		gMetrics.surfaceTriangles);
	return gMetrics.sceneDeformableVolumes == 2;
}

// A narrow regression for the actual failure mode: the driver starts left of
// a resting target and translates along +X with a Y offset.  Both volumes use
// a cooked surface collision mesh embedded in a separate voxel simulation
// mesh, so this is not the old simulation-tet boundary shortcut.  There are
// deliberately no world, rigid or self-contact sources in this fixture.
static bool initSceneCpuVolumeSoftSoftTorque()
{
	PX_ASSERT(gSceneCpuTaskGraphExtraVolumes.empty());
	PX_ASSERT(gSceneCpuVisualVolumeMeshes.empty());
	PX_ASSERT(gSceneCpuVolumeHealthMonitor.empty());
	gSceneCpuSoftSoftTorqueMonitor.releaseSources();

	gSceneCpuVolumeMaterial =
		gPhysics->createDeformableVolumeMaterial(
			SCENE_CPU_VISUAL_YOUNGS_MODULUS,
			SCENE_CPU_VISUAL_POISSONS_RATIO, 0.8f, 0.0f);
	if(!gSceneCpuVolumeMaterial)
		return false;
	gSceneCpuVolumeMaterial->setMaterialModel(
		PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);

	PxArray<PxVec3> surfaceVertices;
	PxArray<PxU32> surfaceTriangles;
	static const PxU32 collisionSubdivisions = 4u;
	static const PxU32 simulationVoxels = 6u;
	createSubdividedCubeSurface(
		surfaceVertices, surfaceTriangles, PxVec3(0.0f), 2.0f,
		collisionSubdivisions);
	const PxU32 surfaceTriangleCount = surfaceTriangles.size() / 3;
	PxDeformableVolumeMesh* cubeMesh = cookSceneCpuVisualVolumeMesh(
		surfaceVertices, surfaceTriangles, simulationVoxels);
	if(!cubeMesh ||
		cubeMesh->getCollisionMesh() == cubeMesh->getSimulationMesh())
	{
		PX_RELEASE(cubeMesh);
		return false;
	}
	gSceneCpuVisualVolumeMeshes.pushBack(cubeMesh);

	if(!createSceneCpuVisualVolume(
			*cubeMesh, PxTransform(PxVec3(-3.0f, 0.95f, 0.0f)),
			PxVec3(1.05f, 0.60f, 0.75f), 140.0f,
			gSceneCpuVolume) ||
		!createSceneCpuVisualVolume(
			*cubeMesh, PxTransform(PxVec3(0.0f, 0.0f, 0.0f)),
			PxVec3(1.0f), 140.0f, gSceneCpuSecondVolume))
		return false;

	auto configureIsolatedSoftBody = [](PxDeformableVolume& volume,
		const PxVec3& velocity)
	{
		volume.setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
		volume.setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		volume.setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, false);
		volume.setLinearDamping(0.0f);
		volume.setSettlingThreshold(0.0f);
		volume.setSettlingDamping(0.0f);
		volume.setSleepThreshold(0.0f);
		volume.setWakeCounter(1.0f);
		volume.setMaxLinearVelocity(PX_MAX_F32);
		return setSceneCpuVolumeVelocity(volume, velocity);
	};
	if(!configureIsolatedSoftBody(
			*gSceneCpuVolume, PxVec3(4.0f, 0.0f, 0.0f)) ||
		!configureIsolatedSoftBody(*gSceneCpuSecondVolume, PxVec3(0.0f)) ||
		!initializeSceneCpuVolumeRestStates(2))
		return false;

	// Public Volume exposes the two domains and the embedding, but deliberately
	// does not expose the private weighted AVBD support rows.  Keep this
	// telemetry honest rather than inferring an exact support count from a mesh
	// vertex count.
	const bool isolatedConfiguration =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC) == 0 &&
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC) == 0 &&
		gSceneCpuVolume->getActorFlags().isSet(
			PxActorFlag::eDISABLE_GRAVITY) &&
		gSceneCpuSecondVolume->getActorFlags().isSet(
			PxActorFlag::eDISABLE_GRAVITY) &&
		gSceneCpuVolume->getDeformableBodyFlags().isSet(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION) &&
		gSceneCpuSecondVolume->getDeformableBodyFlags().isSet(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION) &&
		!gSceneCpuVolume->getDeformableBodyFlags().isSet(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) &&
		!gSceneCpuSecondVolume->getDeformableBodyFlags().isSet(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD);
	if(!gSceneCpuSoftSoftTorqueMonitor.initialize(
			*gSceneCpuVolume, *gSceneCpuSecondVolume,
			isolatedConfiguration, false))
		return false;
	const SceneCpuSoftSoftTorqueMetrics& torqueMetrics =
		gSceneCpuSoftSoftTorqueMetrics;

	gMetrics.initialized = 1;
	gMetrics.sceneActorCreated = 1;
	gMetrics.sceneShapeAttached = 1;
	gMetrics.sceneSimulationMeshAttached = 1;
	gMetrics.sceneHostBuffersInitialized = 1;
	gMetrics.sceneActorAdded = 1;
	gMetrics.sceneSecondVolumeActorCreated = 1;
	gMetrics.sceneSecondVolumeHostBuffersInitialized = 1;
	gMetrics.sceneSecondVolumeActorAdded = 1;
	gMetrics.softBodies = 2;
	gMetrics.particles =
		torqueMetrics.driverSimulationVertices +
		torqueMetrics.targetSimulationVertices;
	gMetrics.tetElements =
		gSceneCpuVolume->getSimulationMesh()->getNbTetrahedrons() +
		gSceneCpuSecondVolume->getSimulationMesh()->getNbTetrahedrons();
	gMetrics.surfaceTriangles = 2 * surfaceTriangleCount;
	gSceneCpuVolumeInitialCentroidY = getSceneCpuVolumeCentroidY();
	gSceneCpuSecondVolumeInitialCentroidY =
		getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes = gScene->getNbDeformableVolumes();
	printf(
		"%s soft/soft true-boundary torque fixture: %u volumes, "
		"%u collision triangles per volume\n",
		AVBD_VOLUME_SNIPPET_NAME, gMetrics.sceneDeformableVolumes,
		surfaceTriangleCount);
	return gMetrics.sceneDeformableVolumes == 2 &&
		torqueMetrics.isolatedConfiguration == 1 &&
		torqueMetrics.targetDistinctCollisionSimulation == 1 &&
		torqueMetrics.driverDistinctCollisionSimulation == 1;
}

// Ground-only, single-tet authority fixture for the local coupled-row probe.
// The four collision vertices are the 0.7/0.1 permutations inside the one
// simulation tetrahedron.  Thus they are all strictly interior and each one
// has nonzero support from every simulation vertex.
static bool initSceneCpuVolumeGroundEmbeddedTetProbe()
{
	PX_ASSERT(gSceneCpuTaskGraphExtraVolumes.empty());
	PX_ASSERT(gSceneCpuVisualVolumeMeshes.empty());
	PX_ASSERT(gSceneCpuVolumeHealthMonitor.empty());
	gSceneCpuGroundEmbeddedTetProbeMetrics =
		SceneCpuGroundEmbeddedTetProbeMetrics();

	PxRigidStatic* const ground = PxCreatePlane(
		*gPhysics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *gMaterial);
	if(!ground)
		return false;
	gScene->addActor(*ground);

	const PxVec3 simulationVertices[] =
	{
		PxVec3(-2.0f, -1.0f, -2.0f),
		PxVec3( 2.0f, -1.0f, -2.0f),
		PxVec3( 0.0f,  3.0f, -2.0f),
		PxVec3( 0.0f, -1.0f,  2.0f)
	};
	const PxU32 simulationTetrahedra[] = { 0, 1, 2, 3 };
	const PxVec4 collisionBarycentrics[] =
	{
		PxVec4(0.7f, 0.1f, 0.1f, 0.1f),
		PxVec4(0.1f, 0.7f, 0.1f, 0.1f),
		PxVec4(0.1f, 0.1f, 0.7f, 0.1f),
		PxVec4(0.1f, 0.1f, 0.1f, 0.7f)
	};
	auto evaluateEmbeddedVertex = [&](const PxVec4& weights)
	{
		return simulationVertices[0] * weights.x +
			simulationVertices[1] * weights.y +
			simulationVertices[2] * weights.z +
			simulationVertices[3] * weights.w;
	};
	const PxVec3 collisionVertices[] =
	{
		evaluateEmbeddedVertex(collisionBarycentrics[0]),
		evaluateEmbeddedVertex(collisionBarycentrics[1]),
		evaluateEmbeddedVertex(collisionBarycentrics[2]),
		evaluateEmbeddedVertex(collisionBarycentrics[3])
	};
	const PxU32 collisionTetrahedra[] = { 0, 1, 2, 3 };

	PxTetrahedronMeshDesc simulationMeshDesc;
	simulationMeshDesc.points.count = 4;
	simulationMeshDesc.points.data = simulationVertices;
	simulationMeshDesc.points.stride = sizeof(PxVec3);
	simulationMeshDesc.tetrahedrons.count = 1;
	simulationMeshDesc.tetrahedrons.data = simulationTetrahedra;
	simulationMeshDesc.tetrahedrons.stride = 4 * sizeof(PxU32);
	PxTetrahedronMeshDesc collisionMeshDesc;
	collisionMeshDesc.points.count = 4;
	collisionMeshDesc.points.data = collisionVertices;
	collisionMeshDesc.points.stride = sizeof(PxVec3);
	collisionMeshDesc.tetrahedrons.count = 1;
	collisionMeshDesc.tetrahedrons.data = collisionTetrahedra;
	collisionMeshDesc.tetrahedrons.stride = 4 * sizeof(PxU32);

	// Pin every collision vertex to the sole simulation tet explicitly.  This
	// keeps the test independent of nearest-tet/ambiguity behavior in cooking.
	PxArray<PxI32> collisionVertexToTet;
	collisionVertexToTet.resize(4);
	for(PxU32 vertexIndex = 0; vertexIndex < 4; ++vertexIndex)
		collisionVertexToTet[vertexIndex] = 0;
	PxDeformableVolumeSimulationDataDesc simulationDataDesc(
		collisionVertexToTet);
	PxCookingParams cookingParams(gPhysics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	cookingParams.meshWeldTolerance = 0.001f;
	cookingParams.meshPreprocessParams =
		PxMeshPreprocessingFlag::eWELD_VERTICES;
	gSceneCpuVolumeMesh = PxCreateDeformableVolumeMesh(
		cookingParams, simulationMeshDesc, collisionMeshDesc,
		simulationDataDesc, gPhysics->getPhysicsInsertionCallback());
	if(!gSceneCpuVolumeMesh ||
		gSceneCpuVolumeMesh->getSimulationMesh() ==
			gSceneCpuVolumeMesh->getCollisionMesh())
		return false;

	gSceneCpuVolumeMaterial =
		gPhysics->createDeformableVolumeMaterial(
			1.0e5f, 0.3f, 0.4f, 0.01f);
	if(!gSceneCpuVolumeMaterial)
		return false;
	gSceneCpuVolumeMaterial->setMaterialModel(
		PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);

	gSceneCpuVolume = gPhysics->createDeformableVolume(
		PxDeformableVolumeBackend::eCPU_AVBD);
	if(!gSceneCpuVolume ||
		gSceneCpuVolume->getDeformableVolumeBackend() !=
			PxDeformableVolumeBackend::eCPU_AVBD ||
		gSceneCpuVolume->getCudaContextManager() != NULL)
		return false;
	gMetrics.sceneActorCreated = 1;

	const PxShapeFlags shapeFlags =
		PxShapeFlag::eVISUALIZATION |
		PxShapeFlag::eSCENE_QUERY_SHAPE |
		PxShapeFlag::eSIMULATION_SHAPE;
	PxTetrahedronMeshGeometry geometry(
		gSceneCpuVolumeMesh->getCollisionMesh());
	PxDeformableVolumeMaterial* material = gSceneCpuVolumeMaterial;
	PxShape* const shape = gPhysics->createShape(
		geometry, &material, 1, true, shapeFlags);
	if(!shape)
		return false;
	const bool shapeAttached = gSceneCpuVolume->attachShape(*shape);
	shape->release();
	if(!shapeAttached || !gSceneCpuVolume->attachSimulationMesh(
			*gSceneCpuVolumeMesh->getSimulationMesh(),
			*gSceneCpuVolumeMesh->getDeformableVolumeAuxData()))
		return false;
	gMetrics.sceneShapeAttached = 1;
	gMetrics.sceneSimulationMeshAttached = 1;

	PxVec4* const simPositionInvMass =
		gSceneCpuVolume->getSimPositionInvMassBufferH();
	PxVec4* const simVelocity =
		gSceneCpuVolume->getSimVelocityBufferH();
	PxVec4* const collisionPositionInvMass =
		gSceneCpuVolume->getPositionInvMassBufferH();
	PxVec4* const collisionRestPosition =
		gSceneCpuVolume->getRestPositionBufferH();
	const PxTetrahedronMesh* const simulationMesh =
		gSceneCpuVolume->getSimulationMesh();
	const PxReal* const cookedInvMass =
		gSceneCpuVolume->getDeformableVolumeAuxData()->
			getGridModelInvMass();
	if(!simPositionInvMass || !simVelocity ||
		!collisionPositionInvMass || !collisionRestPosition ||
		!simulationMesh || !cookedInvMass)
		return false;
	const PxVec3 startTranslation(0.0f, 4.0f, 0.0f);
	for(PxU32 vertexIndex = 0; vertexIndex < 4; ++vertexIndex)
	{
		const PxReal invMass = PxMax(cookedInvMass[vertexIndex], 0.0f);
		simPositionInvMass[vertexIndex] = PxVec4(
			simulationVertices[vertexIndex] + startTranslation, invMass);
		// Uniform translation injects no initial angular momentum.  Ground
		// friction is therefore the only source of the expected -Z roll in
		// this isolated world-static contact fixture.
		simVelocity[vertexIndex] = PxVec4(
			PxVec3(SCENE_CPU_GROUND_EMBEDDED_TET_PROBE_LAUNCH_SPEED,
				0.0f, 0.0f), invMass);
	}
	PxDeformableVolumeExt::updateMass(
		*gSceneCpuVolume, 100.0f, 50.0f, simPositionInvMass);
	for(PxU32 vertexIndex = 0; vertexIndex < 4; ++vertexIndex)
		simVelocity[vertexIndex].w = simPositionInvMass[vertexIndex].w;
	PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
		*gSceneCpuVolume, simPositionInvMass, collisionPositionInvMass);
	for(PxU32 vertexIndex = 0; vertexIndex < 4; ++vertexIndex)
		collisionRestPosition[vertexIndex] =
			collisionPositionInvMass[vertexIndex];

	gSceneCpuVolume->setDeformableBodyFlag(
		PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
	gSceneCpuVolume->setDeformableBodyFlag(
		PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, false);
	gSceneCpuVolume->setLinearDamping(0.02f);
	gSceneCpuVolume->setSleepThreshold(0.0f);
	gSceneCpuVolume->setWakeCounter(1.0f);
	gSceneCpuVolume->setSolverIterationCounts(12, 1);
	gSceneCpuVolume->markDirty(PxDeformableVolumeDataFlag::eALL);
	gMetrics.sceneHostBuffersInitialized = 1;
	gScene->addActor(*gSceneCpuVolume);
	if(gSceneCpuVolume->getScene() != gScene ||
		!initializeSceneCpuVolumeRestStates(1))
		return false;
	gMetrics.sceneActorAdded = 1;

	SceneCpuGroundEmbeddedTetProbeMetrics& probeMetrics =
		gSceneCpuGroundEmbeddedTetProbeMetrics;
	probeMetrics.simulationVertices = simulationMesh->getNbVertices();
	probeMetrics.collisionVertices =
		gSceneCpuVolume->getCollisionMesh()->getNbVertices();
	probeMetrics.simulationTetrahedra =
		simulationMesh->getNbTetrahedrons();
	probeMetrics.collisionTetrahedra =
		gSceneCpuVolume->getCollisionMesh()->getNbTetrahedrons();
	probeMetrics.distinctCollisionSimulation =
		gSceneCpuVolume->getCollisionMesh() != simulationMesh ? 1u : 0u;
	bool strictlyInterior = true;
	for(PxU32 vertexIndex = 0; vertexIndex < 4; ++vertexIndex)
	{
		const PxVec4& weights = collisionBarycentrics[vertexIndex];
		const PxReal weightSum =
			weights.x + weights.y + weights.z + weights.w;
		const PxVec3 embedded =
			simPositionInvMass[0].getXYZ() * weights.x +
			simPositionInvMass[1].getXYZ() * weights.y +
			simPositionInvMass[2].getXYZ() * weights.z +
			simPositionInvMass[3].getXYZ() * weights.w;
		strictlyInterior = strictlyInterior &&
			weights.x > 0.0f && weights.y > 0.0f &&
			weights.z > 0.0f && weights.w > 0.0f &&
			PxAbs(weightSum - 1.0f) <= 1.0e-6f &&
			collisionPositionInvMass[vertexIndex].getXYZ().isFinite() &&
			(collisionPositionInvMass[vertexIndex].getXYZ() - embedded).
				magnitude() <= 1.0e-4f;
	}
	probeMetrics.strictInteriorEmbedding = strictlyInterior ? 1u : 0u;
	probeMetrics.selfCollisionDisabled =
		gSceneCpuVolume->getDeformableBodyFlags().isSet(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION) ? 1u : 0u;
	probeMetrics.speculativeCcdDisabled =
		!gSceneCpuVolume->getDeformableBodyFlags().isSet(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) ? 1u : 0u;
	probeMetrics.contactTelemetryEnabled = 1;
	probeMetrics.launchSpeed =
		SCENE_CPU_GROUND_EMBEDDED_TET_PROBE_LAUNCH_SPEED;
	probeMetrics.rollAxis = PxVec3(0.0f, 0.0f, -1.0f);
	if(!getSceneCpuVolumeMassAndRmsRadius(*gSceneCpuVolume,
			probeMetrics.initialMass, probeMetrics.initialRmsRadius))
		return false;
	probeMetrics.initialized =
		probeMetrics.simulationVertices == 4 &&
		probeMetrics.collisionVertices == 4 &&
		probeMetrics.simulationTetrahedra == 1 &&
		probeMetrics.collisionTetrahedra == 1 &&
		probeMetrics.distinctCollisionSimulation == 1 &&
		probeMetrics.strictInteriorEmbedding == 1 &&
		probeMetrics.selfCollisionDisabled == 1 &&
		probeMetrics.speculativeCcdDisabled == 1 &&
		probeMetrics.contactTelemetryEnabled == 1 &&
		probeMetrics.launchSpeed > 0.0f &&
		probeMetrics.initialMass > 0.0f &&
		probeMetrics.initialRmsRadius > 0.0f;
	if(!probeMetrics.initialized)
		return false;

	gMetrics.initialized = 1;
	gMetrics.particles = probeMetrics.simulationVertices;
	gMetrics.softBodies = 1;
	gMetrics.tetElements = probeMetrics.simulationTetrahedra;
	gMetrics.surfaceTriangles = 4;
	gSceneCpuVolumeInitialCentroidY = getSceneCpuVolumeCentroidY();
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes = gScene->getNbDeformableVolumes();
	printf(
		"%s ground embedded-tet probe: sim=%u vertices/%u tet, "
		"collision=%u vertices/%u tet, distinct=%u\\n",
		AVBD_VOLUME_SNIPPET_NAME, probeMetrics.simulationVertices,
		probeMetrics.simulationTetrahedra, probeMetrics.collisionVertices,
		probeMetrics.collisionTetrahedra,
		probeMetrics.distinctCollisionSimulation);
	return gMetrics.sceneStatics == 1 &&
		gMetrics.sceneDynamics == 0 &&
		gMetrics.sceneDeformableVolumes == 1;
}

static bool configureSceneCpuTaskGraphVolume(
	PxDeformableVolume& volume, const PxVec3& velocity)
{
	volume.setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	volume.setDeformableBodyFlag(
		PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
	volume.setSleepThreshold(0.0f);
	volume.setWakeCounter(1.0f);
	PxVec4* velocities = volume.getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh = volume.getSimulationMesh();
	if(!velocities || !simulationMesh)
		return false;
	for(PxU32 vertexId = 0;
		vertexId < simulationMesh->getNbVertices(); ++vertexId)
		velocities[vertexId] = PxVec4(velocity, velocities[vertexId].w);
	volume.markDirty(PxDeformableVolumeDataFlag::eSIM_VELOCITY);
	return true;
}

static bool initializeVolumeSkinning()
{
	gVolumeSkinningBindings.clear();
	gVolumeSkinningTriangles.clear();
	gVolumeSkinningPositions.clear();
	gVolumeSkinningNormals.clear();
	gVolumeSkinningInitialPositions.clear();
	gVolumeAvbdSkinningRenderData =
		VolumeAvbdSkinningRenderData();
	gVolumeSkinningMetrics = VolumeSkinningMetrics();

	if(!gSceneCpuVolume ||
		!gSceneCpuVolume->getSimulationMesh())
		return false;
	const PxTetrahedronMesh* simulationMesh =
		gSceneCpuVolume->getSimulationMesh();
	if(simulationMesh->getNbTetrahedrons() == 0)
		return false;

	PxU32 tetrahedron[4];
	const bool has16BitIndices =
		simulationMesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	if(has16BitIndices)
	{
		const PxU16* indices =
			static_cast<const PxU16*>(
				simulationMesh->getTetrahedrons());
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
			tetrahedron[endpoint] = indices[endpoint];
	}
	else
	{
		const PxU32* indices =
			static_cast<const PxU32*>(
				simulationMesh->getTetrahedrons());
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
			tetrahedron[endpoint] = indices[endpoint];
	}

	Snippets::appendTetrahedronSurfaceSkinning(
		tetrahedron, 8, gVolumeSkinningBindings,
		gVolumeSkinningTriangles);
	const PxVec4* positions =
		gSceneCpuVolume->getSimPositionInvMassBufferH();
	if(!Snippets::evaluateTetrahedronSkinning(
		positions, simulationMesh->getNbVertices(),
		gVolumeSkinningBindings, gVolumeSkinningTriangles,
		gVolumeSkinningPositions, gVolumeSkinningNormals))
		return false;
	gVolumeSkinningInitialPositions =
		gVolumeSkinningPositions;
	gVolumeAvbdSkinningRenderData.positions =
		gVolumeSkinningPositions.begin();
	gVolumeAvbdSkinningRenderData.normals =
		gVolumeSkinningNormals.begin();
	gVolumeAvbdSkinningRenderData.triangles =
		gVolumeSkinningTriangles.begin();
	gVolumeAvbdSkinningRenderData.numVertices =
		gVolumeSkinningPositions.size();
	gVolumeAvbdSkinningRenderData.numTriangles =
		gVolumeSkinningTriangles.size() / 3;
	gVolumeSkinningMetrics.initialized = 1;
	gVolumeSkinningMetrics.vertices =
		gVolumeSkinningPositions.size();
	gVolumeSkinningMetrics.triangles =
		gVolumeSkinningTriangles.size() / 3;
	return true;
}

static bool updateVolumeSkinning()
{
	if(gHeadlessOptions.caseName != "scene-volume-skinning")
		return true;
	if(!gSceneCpuVolume ||
		!gSceneCpuVolume->getSimulationMesh() ||
		!gVolumeSkinningMetrics.initialized)
		return false;
	const PxTetrahedronMesh* simulationMesh =
		gSceneCpuVolume->getSimulationMesh();
	if(!Snippets::evaluateTetrahedronSkinning(
		gSceneCpuVolume->getSimPositionInvMassBufferH(),
		simulationMesh->getNbVertices(),
		gVolumeSkinningBindings, gVolumeSkinningTriangles,
		gVolumeSkinningPositions, gVolumeSkinningNormals))
		return false;

	gVolumeSkinningMetrics.evaluatedFrames++;
	bool finite = true;
	for(PxU32 i = 0; i < gVolumeSkinningPositions.size(); ++i)
	{
		if(!gVolumeSkinningPositions[i].isFinite() ||
			!gVolumeSkinningNormals[i].isFinite())
		{
			finite = false;
			break;
		}
		gVolumeSkinningMetrics.maxDisplacement = PxMax(
			gVolumeSkinningMetrics.maxDisplacement,
			(gVolumeSkinningPositions[i] -
				gVolumeSkinningInitialPositions[i]).magnitude());
	}
	if(finite)
		gVolumeSkinningMetrics.finiteFrames++;
	gVolumeAvbdSkinningRenderData.positions =
		gVolumeSkinningPositions.begin();
	gVolumeAvbdSkinningRenderData.normals =
		gVolumeSkinningNormals.begin();
	return finite;
}

static bool setSceneCpuVolumeVelocity(
	PxDeformableVolume& volume, const PxVec3& velocity);
static bool setSceneCpuVolumeDeformingReverseVelocity(
	PxDeformableVolume& volume, PxReal speed);
static PxReal getSceneCpuVolumeMinY(
	PxDeformableVolume* volume);

static bool initSceneCpuVolumeLifecycle()
{
	PX_ASSERT(gSceneCpuTaskGraphExtraMeshes.empty());
	gSceneCpuSphereReverseSweptInitialPositions.reset();
	gSceneCpuDeformingReverseSweptFreeEndPositions.reset();
	gSceneCpuCapsuleRotationalSweptInitialPositions.reset();
	const bool partialElementFilterCase =
		gHeadlessOptions.caseName ==
			"scene-volume-partial-element-filter";
	const bool elementFilterCase =
		gHeadlessOptions.caseName == "scene-volume-element-filter" ||
		partialElementFilterCase;
	const bool trueBoundaryDynamicBoxCase =
		gHeadlessOptions.caseName ==
			"scene-volume-true-boundary-dynamic-box";
	const bool distinctCollisionMeshCase =
		partialElementFilterCase || trueBoundaryDynamicBoxCase;
	const bool taskGraphPureSoftCase =
		isSceneCpuVolumeTaskGraphPureSoftCase(gHeadlessOptions.caseName);
	const bool taskGraphWorldPlaneCase =
		gHeadlessOptions.caseName ==
			"scene-volume-taskgraph-world-plane";
	const bool taskGraphRigidBoxSdfCase =
		gHeadlessOptions.caseName ==
			"scene-volume-taskgraph-rigid-box-sdf";
	const bool taskGraphRigidSphereSdfCase =
		gHeadlessOptions.caseName ==
			"scene-volume-taskgraph-rigid-sphere-sdf";
	const bool taskGraphRigidCapsuleSdfCase =
		gHeadlessOptions.caseName ==
			"scene-volume-taskgraph-rigid-capsule-sdf";
	const bool taskGraphRigidConvexSdfCase =
		gHeadlessOptions.caseName ==
			"scene-volume-taskgraph-rigid-convex-sdf";
	const bool taskGraphRigidTriangleSurfaceCase =
		gHeadlessOptions.caseName ==
			"scene-volume-taskgraph-rigid-triangle-surface";
	const bool taskGraphRigidTriangleSurfaceLargeCase =
		isSceneCpuVolumeTaskGraphRigidTriangleSurfaceLargeCase(
			gHeadlessOptions.caseName);
	const bool taskGraphRigidTriangleSurfaceFeatureOverlapCase =
		isSceneCpuVolumeTaskGraphRigidTriangleSurfaceFeatureOverlapCase(
			gHeadlessOptions.caseName);
	const bool taskGraphRigidTriangleSurfaceThresholdCase =
		isSceneCpuVolumeTaskGraphRigidTriangleSurfaceThresholdCase(
			gHeadlessOptions.caseName);
	const bool taskGraphRigidTriangleSurfaceAnyCase =
		taskGraphRigidTriangleSurfaceCase ||
		taskGraphRigidTriangleSurfaceLargeCase ||
		taskGraphRigidTriangleSurfaceFeatureOverlapCase ||
		taskGraphRigidTriangleSurfaceThresholdCase;
	const bool taskGraphWriteBackCase =
		isSceneCpuVolumeTaskGraphWriteBackCase(gHeadlessOptions.caseName);
	const bool taskGraphWriteBackFourWayCase =
		isSceneCpuVolumeTaskGraphWriteBackFourWayCase(
			gHeadlessOptions.caseName);
	const bool taskGraphWriteBackHeterogeneousCase =
		gHeadlessOptions.caseName ==
			"scene-volume-taskgraph-writeback-heterogeneous";
	const bool taskGraphPipelineCase =
		isSceneCpuVolumeTaskGraphPipelineCase(gHeadlessOptions.caseName);
	const bool taskGraphDirectSimulationDomainCase =
		isSceneCpuVolumeTaskGraphDirectSimulationDomainCase(
			gHeadlessOptions.caseName);
	PxArray<PxVec3> surfaceVertices;
	PxArray<PxU32> surfaceTriangles;
	auto appendTetraSurface = [&](
		const PxVec3& offset)
	{
		const PxU32 vertexBase = surfaceVertices.size();
		surfaceVertices.pushBack(
			offset + PxVec3(0.0f, 0.0f, 0.0f));
		surfaceVertices.pushBack(
			offset + PxVec3(1.0f, 0.0f, 0.0f));
		surfaceVertices.pushBack(
			offset + PxVec3(0.0f, 1.0f, 0.0f));
		surfaceVertices.pushBack(
			offset + PxVec3(0.0f, 0.0f, 1.0f));
		const PxU32 indices[] =
		{
			0, 2, 1,
			0, 1, 3,
			0, 3, 2,
			1, 2, 3
		};
		for(PxU32 i = 0;
			i < sizeof(indices) / sizeof(indices[0]); ++i)
			surfaceTriangles.pushBack(vertexBase + indices[i]);
	};
	if(distinctCollisionMeshCase)
	{
		appendTetraSurface(PxVec3(-3.0f, 0.0f, 0.0f));
		appendTetraSurface(PxVec3(2.0f, 0.0f, 0.0f));
	}
	else
		appendTetraSurface(PxVec3(0.0f));

	PxSimpleTriangleMesh surfaceMesh;
	surfaceMesh.points.count = surfaceVertices.size();
	surfaceMesh.points.data = surfaceVertices.begin();
	surfaceMesh.points.stride = sizeof(PxVec3);
	surfaceMesh.triangles.count = surfaceTriangles.size() / 3;
	surfaceMesh.triangles.data = surfaceTriangles.begin();
	surfaceMesh.triangles.stride = 3 * sizeof(PxU32);

	PxCookingParams cookingParams(gPhysics->getTolerancesScale());
	// CPU AVBD consumes shared deformable topology and embedding data without
	// requesting the optional GRB/BV32 payload.
	cookingParams.buildGPUData = false;
	cookingParams.meshWeldTolerance = 0.001f;
	cookingParams.meshPreprocessParams =
		PxMeshPreprocessingFlag::eWELD_VERTICES;
	if(taskGraphPureSoftCase || taskGraphWorldPlaneCase ||
		taskGraphRigidBoxSdfCase || taskGraphRigidSphereSdfCase ||
		taskGraphRigidCapsuleSdfCase ||
		taskGraphRigidConvexSdfCase ||
		taskGraphRigidTriangleSurfaceAnyCase ||
		taskGraphWriteBackCase)
	{
		// P2/P3 task-graph authority fixtures share the fixed 12^3 pure-soft
		// topology from the P0 dense corpus. Triangle P5.16b instead uses the
		// smallest two-range topology: its two serial OGC feature suffixes are
		// intentionally retained and far more expensive than the current/swept
		// SDF child leaves, so 12^3 would measure parent work rather than the
		// task transaction contract. P5.19 uses a separate 7^3 large/two-surface
		// fixture; P5.20 uses 6^3 to underfill four workers under the accepted
		// 128-particle policy without hierarchy-size capacity dominating either
		// threshold-policy measurement.
		PxArray<PxVec3> denseVertices;
		PxArray<PxU32> denseTetrahedra;
		const PxU32 taskGraphSubdivision =
			taskGraphRigidTriangleSurfaceLargeCase ? 7u :
			taskGraphRigidTriangleSurfaceThresholdCase ? 6u :
			(taskGraphRigidTriangleSurfaceCase ||
			 taskGraphRigidTriangleSurfaceFeatureOverlapCase) ? 5u : 12u;
		avbdGenerateSubdividedCubeTets(
			PxVec3(0.0f), 2.0f, taskGraphSubdivision,
			denseVertices, denseTetrahedra);
		PxTetrahedronMeshDesc denseMeshDesc;
		denseMeshDesc.points.count = denseVertices.size();
		denseMeshDesc.points.data = denseVertices.begin();
		denseMeshDesc.points.stride = sizeof(PxVec3);
		denseMeshDesc.tetrahedrons.count =
			denseTetrahedra.size() / 4;
		denseMeshDesc.tetrahedrons.data = denseTetrahedra.begin();
		denseMeshDesc.tetrahedrons.stride = 4 * sizeof(PxU32);
		PxDeformableVolumeSimulationDataDesc simulationDataDesc;
		gSceneCpuVolumeMesh = PxCreateDeformableVolumeMesh(
			cookingParams, denseMeshDesc, denseMeshDesc,
			simulationDataDesc,
			gPhysics->getPhysicsInsertionCallback());
		if(taskGraphWriteBackHeterogeneousCase && gSceneCpuVolumeMesh)
		{
			// Body order is deliberately [large, large, small, small]. A
			// two-worker count split therefore exposes P6 range imbalance while
			// preserving complete-body ownership and a fixed no-contact workload.
			PxArray<PxVec3> smallVertices;
			PxArray<PxU32> smallTetrahedra;
			avbdGenerateSubdividedCubeTets(
				PxVec3(0.0f), 2.0f, 4u,
				smallVertices, smallTetrahedra);
			PxTetrahedronMeshDesc smallMeshDesc;
			smallMeshDesc.points.count = smallVertices.size();
			smallMeshDesc.points.data = smallVertices.begin();
			smallMeshDesc.points.stride = sizeof(PxVec3);
			smallMeshDesc.tetrahedrons.count =
				smallTetrahedra.size() / 4;
			smallMeshDesc.tetrahedrons.data = smallTetrahedra.begin();
			smallMeshDesc.tetrahedrons.stride = 4 * sizeof(PxU32);
			PxDeformableVolumeSimulationDataDesc smallSimulationDataDesc;
			PxDeformableVolumeMesh* const smallMesh =
				PxCreateDeformableVolumeMesh(
					cookingParams, smallMeshDesc, smallMeshDesc,
					smallSimulationDataDesc,
					gPhysics->getPhysicsInsertionCallback());
			if(!smallMesh)
				return false;
			gSceneCpuTaskGraphExtraMeshes.pushBack(smallMesh);
		}
	}
	else if(distinctCollisionMeshCase)
	{
		const PxU32 collisionTetrahedra[] =
		{
			0, 1, 2, 3,
			4, 5, 6, 7
		};
		PxTetrahedronMeshDesc collisionMeshDesc;
		collisionMeshDesc.points.count = surfaceVertices.size();
		collisionMeshDesc.points.data = surfaceVertices.begin();
		collisionMeshDesc.points.stride = sizeof(PxVec3);
		collisionMeshDesc.tetrahedrons.count = 2;
		collisionMeshDesc.tetrahedrons.data =
			collisionTetrahedra;
		collisionMeshDesc.tetrahedrons.stride =
			4 * sizeof(PxU32);

		PxArray<PxVec3> simulationVertices =
			surfaceVertices;
		simulationVertices.pushBack(
			PxVec3(-2.75f, 0.25f, 0.25f));
		simulationVertices.pushBack(
			PxVec3(2.25f, 0.25f, 0.25f));
		const PxU32 simulationTetrahedra[] =
		{
			0, 1, 2, 8,
			0, 1, 8, 3,
			0, 8, 2, 3,
			8, 1, 2, 3,
			4, 5, 6, 9,
			4, 5, 9, 7,
			4, 9, 6, 7,
			9, 5, 6, 7
		};
		PxTetrahedronMeshDesc simulationMeshDesc;
		simulationMeshDesc.points.count =
			simulationVertices.size();
		simulationMeshDesc.points.data =
			simulationVertices.begin();
		simulationMeshDesc.points.stride = sizeof(PxVec3);
		simulationMeshDesc.tetrahedrons.count = 8;
		simulationMeshDesc.tetrahedrons.data =
			simulationTetrahedra;
		simulationMeshDesc.tetrahedrons.stride =
			4 * sizeof(PxU32);
		PxDeformableVolumeSimulationDataDesc simulationDataDesc;
		gSceneCpuVolumeMesh = PxCreateDeformableVolumeMesh(
			cookingParams, simulationMeshDesc,
			collisionMeshDesc,
			simulationDataDesc,
			gPhysics->getPhysicsInsertionCallback());
	}
	else
	{
		gSceneCpuVolumeMesh =
			PxDeformableVolumeExt::createDeformableVolumeMeshNoVoxels(
				cookingParams, surfaceMesh,
				gPhysics->getPhysicsInsertionCallback(), 1.5f, true);
	}
	if(!gSceneCpuVolumeMesh)
		return false;

	gSceneCpuVolumeMaterial =
		gPhysics->createDeformableVolumeMaterial(
			2.0e5f, 0.3f,
			(gHeadlessOptions.caseName ==
					"scene-volume-motion-controls" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-max-depenetration-velocity" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-sphere-reverse-feature" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-capsule-reverse-feature" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-convex-reverse-feature" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-triangle-mesh-reverse-feature" ||
			 gHeadlessOptions.caseName ==
					"scene-volume-heightfield-reverse-feature" ||
			 isSceneCpuVolumeSpeculativeCcdCase(
				gHeadlessOptions.caseName) ||
			 isSceneCpuVolumeSphereReverseSweptCcdCase(
				gHeadlessOptions.caseName))
				? 0.0f : 0.2f,
			0.01f);
	if(!gSceneCpuVolumeMaterial)
		return false;
	const PxDeformableVolumeMaterialModel::Enum materialModel =
		(gHeadlessOptions.caseName == "scene-volume-corotational" ||
		 gHeadlessOptions.caseName ==
			"scene-volume-taskgraph-pure-soft-corotational")
			? PxDeformableVolumeMaterialModel::eCO_ROTATIONAL
			: PxDeformableVolumeMaterialModel::eNEO_HOOKEAN;
	gSceneCpuVolumeMaterial->setMaterialModel(materialModel);
	if(gSceneCpuVolumeMaterial->getMaterialModel() != materialModel)
		return false;

	gSceneCpuVolume = gPhysics->createDeformableVolume(
		PxDeformableVolumeBackend::eCPU_AVBD);
	if(!gSceneCpuVolume)
		return false;
	gMetrics.sceneActorCreated = 1;
	if(gSceneCpuVolume->getDeformableVolumeBackend() !=
		PxDeformableVolumeBackend::eCPU_AVBD ||
		gSceneCpuVolume->getCudaContextManager() != NULL)
		return false;

	PxShapeFlags shapeFlags =
		PxShapeFlag::eVISUALIZATION |
		PxShapeFlag::eSCENE_QUERY_SHAPE |
		PxShapeFlag::eSIMULATION_SHAPE;
	// Task-graph collision leaves currently consume simulation-particle
	// topology directly. Give their dedicated fixtures an explicitly identical
	// public collision domain by attaching that same mesh object; ordinary
	// public-volume cases retain the cooked collision mesh and its embedding.
	PxTetrahedronMesh* shapeMesh = taskGraphDirectSimulationDomainCase
		? gSceneCpuVolumeMesh->getSimulationMesh()
		: gSceneCpuVolumeMesh->getCollisionMesh();
	PxTetrahedronMeshGeometry geometry(shapeMesh);
	PxDeformableVolumeMaterial* material = gSceneCpuVolumeMaterial;
	PxShape* shape = gPhysics->createShape(
		geometry, &material, 1, true, shapeFlags);
	if(!shape)
		return false;
	const bool shapeAttached = gSceneCpuVolume->attachShape(*shape);
	shape->release();
	if(!shapeAttached)
		return false;
	gMetrics.sceneShapeAttached = 1;

	if(!gSceneCpuVolume->attachSimulationMesh(
		*gSceneCpuVolumeMesh->getSimulationMesh(),
		*gSceneCpuVolumeMesh->getDeformableVolumeAuxData()))
		return false;
	gMetrics.sceneSimulationMeshAttached = 1;

	PxVec4* simPositionInvMass =
		gSceneCpuVolume->getSimPositionInvMassBufferH();
	PxVec4* simVelocity =
		gSceneCpuVolume->getSimVelocityBufferH();
	PxVec4* collisionPositionInvMass =
		gSceneCpuVolume->getPositionInvMassBufferH();
	PxVec4* collisionRestPosition =
		gSceneCpuVolume->getRestPositionBufferH();
	if(!simPositionInvMass || !simVelocity ||
		!collisionPositionInvMass || !collisionRestPosition)
		return false;

	const bool multiSoftIslandCase =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-soft-islands";
	const bool mixedSleepIslandCase =
		gHeadlessOptions.caseName ==
			"scene-volume-mixed-sleep-islands";
	const bool softChurnCase =
		gHeadlessOptions.caseName ==
			"scene-volume-soft-churn";
	const bool bufferMutationCase =
		gHeadlessOptions.caseName ==
			"scene-volume-buffer-mutation";
	const bool worldElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-world-element-attachment";
	const bool rigidElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-rigid-element-attachment";
	const bool staticElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-static-element-attachment";
	const bool kinematicElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-element-attachment";
	const bool articulationElementAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-element-attachment";
	const bool rigidAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-rigid-attachment" ||
		rigidElementAttachmentCase;
	const bool staticAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-static-attachment" ||
		staticElementAttachmentCase;
	const bool kinematicAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-attachment" ||
		kinematicElementAttachmentCase;
	const bool articulationAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-attachment" ||
		articulationElementAttachmentCase;
	const bool attachmentCase =
		rigidAttachmentCase || staticAttachmentCase ||
		kinematicAttachmentCase ||
		articulationAttachmentCase;
	const bool multiSceneIsolationCase =
		gHeadlessOptions.caseName ==
			"scene-volume-multi-scene-isolation";
	const bool softSoftWakeCase =
		gHeadlessOptions.caseName ==
			"scene-volume-soft-soft-wake";
	const bool softPairAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-volume-attachment";
	const bool rigidTriangleSteadyContactCase =
		isSceneCpuVolumeRigidTriangleSteadyContactCase(
			gHeadlessOptions.caseName);
	const bool fullKinematicTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-full-kinematic-target";
	const bool partialKinematicTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-partial-kinematic-target";
	const bool volumeKinematicTargetCase =
		fullKinematicTargetCase || partialKinematicTargetCase ||
		rigidTriangleSteadyContactCase;
	const bool motionControlsCase =
		gHeadlessOptions.caseName ==
			"scene-volume-motion-controls";
	const bool maxDepenetrationVelocityCase =
		gHeadlessOptions.caseName ==
			"scene-volume-max-depenetration-velocity";
	const bool triangleSurfaceSweptCcdCase =
		isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool triangleSurfaceReverseSweptCcdCase =
		isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool rotationalTriangleSurfaceSweptCcdCase =
		isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool sphereReverseSweptCcdCase =
		isSceneCpuVolumeSphereReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool deformingReverseSweptCcdCase =
		isSceneCpuVolumeDeformingReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool convexReverseSweptCcdCase =
		isSceneCpuVolumeConvexReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool rotationalCapsuleReverseSweptCcdCase =
		isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool rotationalConvexReverseSweptCcdCase =
		isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
			gHeadlessOptions.caseName);
	const bool rotationalFiniteReverseSweptCcdCase =
		rotationalCapsuleReverseSweptCcdCase ||
		rotationalConvexReverseSweptCcdCase;
	const bool staticSphereReverseSweptCcdCase =
		deformingReverseSweptCcdCase ||
		gHeadlessOptions.caseName ==
			"scene-volume-static-sphere-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-static-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-static-convex-reverse-swept-ccd";
	const bool kinematicSphereReverseSweptCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-convex-reverse-swept-ccd";
	const bool dynamicSphereReverseSweptCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-sphere-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-convex-reverse-swept-ccd";
	const bool movingSphereReverseSweptCcdCase =
		kinematicSphereReverseSweptCcdCase ||
		dynamicSphereReverseSweptCcdCase;
	const bool speculativeCcdCase =
		isSceneCpuVolumeSpeculativeCcdCase(
			gHeadlessOptions.caseName) ||
		sphereReverseSweptCcdCase;
	const bool sphereSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-sphere-speculative-ccd";
	const bool capsuleSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-capsule-speculative-ccd";
	const bool convexSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-convex-speculative-ccd";
	const bool finiteSmoothSpeculativeCcdCase =
		sphereSpeculativeCcdCase || capsuleSpeculativeCcdCase ||
		convexSpeculativeCcdCase;
	const bool rotatingKinematicCapsuleSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-capsule-speculative-ccd";
	const bool dynamicRotatingCapsuleSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-capsule-relative-swept-ccd";
	const bool rotatingKinematicConvexSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-rotating-kinematic-convex-speculative-ccd";
	const bool dynamicRotatingConvexSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-rotating-convex-relative-swept-ccd";
	const bool rotationalConvexSpeculativeCcdCase =
		rotatingKinematicConvexSpeculativeCcdCase ||
		dynamicRotatingConvexSpeculativeCcdCase;
	const bool rotationalCapsuleSpeculativeCcdCase =
		rotatingKinematicCapsuleSpeculativeCcdCase ||
		dynamicRotatingCapsuleSpeculativeCcdCase;
	const bool rotationalFiniteSpeculativeCcdCase =
		rotationalCapsuleSpeculativeCcdCase ||
		rotationalConvexSpeculativeCcdCase;
	const bool movingKinematicFiniteSpeculativeCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-moving-kinematic-sphere-speculative-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-moving-kinematic-capsule-speculative-ccd" ||
		rotatingKinematicCapsuleSpeculativeCcdCase ||
		rotatingKinematicConvexSpeculativeCcdCase ||
		gHeadlessOptions.caseName ==
			"scene-volume-moving-kinematic-convex-speculative-ccd";
	const bool dynamicFiniteRelativeSweptCcdCase =
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-sphere-relative-swept-ccd" ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-capsule-relative-swept-ccd" ||
		dynamicRotatingCapsuleSpeculativeCcdCase ||
		dynamicRotatingConvexSpeculativeCcdCase ||
		gHeadlessOptions.caseName ==
			"scene-volume-dynamic-convex-relative-swept-ccd";
	const bool sphereReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-sphere-reverse-feature";
	const bool capsuleReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-capsule-reverse-feature";
	const bool convexReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-convex-reverse-feature";
	const bool triangleMeshReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-triangle-mesh-reverse-feature";
	const bool heightFieldReverseFeatureCase =
		gHeadlessOptions.caseName ==
			"scene-volume-heightfield-reverse-feature";
	const bool smoothReverseFeatureCase =
		sphereReverseFeatureCase || capsuleReverseFeatureCase ||
		convexReverseFeatureCase ||
		triangleMeshReverseFeatureCase ||
		heightFieldReverseFeatureCase;
	const bool twoSoftVolumeCase =
		multiSoftIslandCase || mixedSleepIslandCase ||
		softChurnCase || multiSceneIsolationCase ||
		softSoftWakeCase || softPairAttachmentCase ||
		motionControlsCase || maxDepenetrationVelocityCase ||
		speculativeCcdCase || smoothReverseFeatureCase ||
		taskGraphWriteBackCase;
	gSceneCpuTaskGraphExtraVolumes.reset();
	const bool softSleepCase =
		gHeadlessOptions.caseName == "scene-volume-sleep-wake" ||
		gHeadlessOptions.caseName == "scene-volume-rigid-wake" ||
		isSceneCpuVolumeKinematicRigidCase(
			gHeadlessOptions.caseName) ||
		mixedSleepIslandCase || softChurnCase ||
		bufferMutationCase || attachmentCase ||
		multiSceneIsolationCase ||
		softSoftWakeCase || softPairAttachmentCase ||
		volumeKinematicTargetCase;
	const PxReal taskGraphPipelineHeight = 2.5f;
	const PxVec3 translation =
		rigidTriangleSteadyContactCase
			? PxVec3(-2.0f, 1.1f, 0.0f)
			: taskGraphRigidTriangleSurfaceFeatureOverlapCase
			? PxVec3(-2.0f, -0.1f, 0.0f)
			: smoothReverseFeatureCase
			? PxVec3(-2.0f, 0.34f, 0.0f)
			: triangleSurfaceSweptCcdCase
			? PxVec3(-2.0f, 0.0f, 0.0f)
			: sphereReverseSweptCcdCase
			? PxVec3(
				-2.0f,
				deformingReverseSweptCcdCase
					? 0.45f
				: staticSphereReverseSweptCcdCase
					? (convexReverseSweptCcdCase
						? 1.1f : 0.55f)
					: 0.0f,
				0.0f)
			: rotationalFiniteSpeculativeCcdCase
			? PxVec3(-2.0f, 0.0f, 0.0f)
			: speculativeCcdCase
			? PxVec3(-2.0f, 1.2f, 0.0f)
			: maxDepenetrationVelocityCase
			? PxVec3(-3.0f, -1.05f, 0.0f)
			: elementFilterCase
			? PxVec3(
				0.0f, SCENE_CPU_ELEMENT_FILTER_INITIAL_CLEARANCE, 0.0f)
			: taskGraphPipelineCase
			? PxVec3(-10.0f, taskGraphPipelineHeight, 0.0f)
			: PxVec3(
				twoSoftVolumeCase && !softSoftWakeCase &&
					!softPairAttachmentCase ? -10.0f : 0.0f,
				4.0f, 0.0f);
	const PxTetrahedronMesh* simulationMesh =
		gSceneCpuVolume->getSimulationMesh();
	const PxVec3* simulationVertices = simulationMesh->getVertices();
	const PxU32 simulationVertexCount = simulationMesh->getNbVertices();
	gSceneCpuElementAttachment =
		worldElementAttachmentCase || rigidElementAttachmentCase ||
		staticElementAttachmentCase ||
		kinematicElementAttachmentCase ||
		articulationElementAttachmentCase ||
		softPairAttachmentCase;
	if(gSceneCpuElementAttachment)
	{
		if(simulationMesh->getNbTetrahedrons() == 0)
			return false;
		const bool has16BitIndices =
			simulationMesh->getTetrahedronMeshFlags() &
				PxTetrahedronMeshFlag::e16_BIT_INDICES;
		if(has16BitIndices)
		{
			const PxU16* tetrahedra =
				static_cast<const PxU16*>(
					simulationMesh->getTetrahedrons());
			for(PxU32 endpoint = 0; endpoint < 4; endpoint++)
				gSceneCpuAttachmentVertices[endpoint] =
					tetrahedra[endpoint];
		}
		else
		{
			const PxU32* tetrahedra =
				static_cast<const PxU32*>(
					simulationMesh->getTetrahedrons());
			for(PxU32 endpoint = 0; endpoint < 4; endpoint++)
				gSceneCpuAttachmentVertices[endpoint] =
					tetrahedra[endpoint];
		}
		for(PxU32 endpoint = 0; endpoint < 4; endpoint++)
		{
			if(gSceneCpuAttachmentVertices[endpoint] >=
				simulationVertexCount)
				return false;
		}
	}
	PxReal* cookedInvMass =
		gSceneCpuVolume->getDeformableVolumeAuxData()->
			getGridModelInvMass();
	PxU32 deformingReverseMovingVertex = 0;
	if(deformingReverseSweptCcdCase &&
		simulationVertexCount > 0)
	{
		PxReal minimumRestRadiusSq =
			simulationVertices[0].magnitudeSquared();
		for(PxU32 i = 1; i < simulationVertexCount; ++i)
		{
			const PxReal restRadiusSq =
				simulationVertices[i].magnitudeSquared();
			if(restRadiusSq < minimumRestRadiusSq)
			{
				minimumRestRadiusSq = restRadiusSq;
				deformingReverseMovingVertex = i;
			}
		}
	}
	for(PxU32 i = 0; i < simulationVertexCount; i++)
	{
		const PxReal invMass = cookedInvMass
			? PxMax(cookedInvMass[i], 0.0f) : 1.0f;
		simPositionInvMass[i] =
			PxVec4(simulationVertices[i] + translation, invMass);
		simVelocity[i] = PxVec4(
			taskGraphWriteBackCase
				? PxVec3(0.35f, -0.25f, 0.0f)
				: rigidTriangleSteadyContactCase
				? PxVec3(0.0f)
				: motionControlsCase
				? PxVec3(10.0f, 0.0f, 0.0f)
				: smoothReverseFeatureCase
				? PxVec3(0.0f, -2.0f, 0.0f)
				: deformingReverseSweptCcdCase
				? (i == deformingReverseMovingVertex
					? PxVec3(0.0f, -720.0f, 0.0f)
					: PxVec3(0.0f))
				: speculativeCcdCase
				? PxVec3(
					0.0f,
					triangleSurfaceSweptCcdCase
						? 0.0f :
					staticSphereReverseSweptCcdCase
						? -132.0f :
					(movingKinematicFiniteSpeculativeCcdCase ||
					 dynamicFiniteRelativeSweptCcdCase ||
					 movingSphereReverseSweptCcdCase)
						? 0.0f :
					capsuleSpeculativeCcdCase
						? -180.0f :
					convexSpeculativeCcdCase
						? -220.0f :
					finiteSmoothSpeculativeCcdCase
						? -160.0f : -120.0f,
					0.0f)
				: PxVec3(0.0f),
			invMass);
	}
	PxDeformableVolumeExt::updateMass(
		*gSceneCpuVolume, 100.0f, 50.0f, simPositionInvMass);
	for(PxU32 i = 0; i < simulationVertexCount; i++)
		simVelocity[i].w = simPositionInvMass[i].w;
	PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
		*gSceneCpuVolume, simPositionInvMass,
		collisionPositionInvMass);
	const PxU32 collisionVertexCount =
		gSceneCpuVolume->getCollisionMesh()->getNbVertices();
	for(PxU32 i = 0; i < collisionVertexCount; i++)
		collisionRestPosition[i] = collisionPositionInvMass[i];
	if(sphereReverseSweptCcdCase ||
		triangleSurfaceReverseSweptCcdCase ||
		rotationalTriangleSurfaceSweptCcdCase)
	{
		gSceneCpuSphereReverseSweptInitialPositions.resize(
			collisionVertexCount);
		for(PxU32 i = 0; i < collisionVertexCount; ++i)
			gSceneCpuSphereReverseSweptInitialPositions[i] =
				collisionPositionInvMass[i].getXYZ();
	}
	if(deformingReverseSweptCcdCase)
	{
		PxArray<PxVec4> freeEndSimulationPositions;
		PxArray<PxVec4> freeEndCollisionPositions;
		freeEndSimulationPositions.resize(simulationVertexCount);
		freeEndCollisionPositions.resize(collisionVertexCount);
		for(PxU32 i = 0; i < simulationVertexCount; ++i)
		{
			freeEndSimulationPositions[i] = PxVec4(
				simPositionInvMass[i].getXYZ() +
					simVelocity[i].getXYZ() *
						gHeadlessOptions.dt,
				simPositionInvMass[i].w);
		}
		PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
			*gSceneCpuVolume,
			freeEndSimulationPositions.begin(),
			freeEndCollisionPositions.begin());
		gSceneCpuDeformingReverseSweptFreeEndPositions.resize(
			collisionVertexCount);
		for(PxU32 i = 0; i < collisionVertexCount; ++i)
			gSceneCpuDeformingReverseSweptFreeEndPositions[i] =
				freeEndCollisionPositions[i].getXYZ();
	}
	if(rotationalFiniteSpeculativeCcdCase ||
		rotationalFiniteReverseSweptCcdCase)
	{
		gSceneCpuCapsuleRotationalSweptInitialPositions.resize(
			collisionVertexCount);
		for(PxU32 i = 0; i < collisionVertexCount; ++i)
			gSceneCpuCapsuleRotationalSweptInitialPositions[i] =
				collisionPositionInvMass[i].getXYZ();
	}
	if(volumeKinematicTargetCase)
	{
		gSceneCpuVolumeKinematicTargets.resize(
			simulationVertexCount);
		gSceneCpuVolumeKinematicInitial.resize(
			simulationVertexCount);
		gSceneCpuVolumePartialProbe =
			partialKinematicTargetCase
				? simulationVertexCount - 1
				: PX_MAX_U32;
		for(PxU32 i = 0; i < simulationVertexCount; ++i)
		{
			const PxVec3 initial =
				simPositionInvMass[i].getXYZ();
			gSceneCpuVolumeKinematicInitial[i] = initial;
			const bool inactiveProbe =
				partialKinematicTargetCase &&
				i == gSceneCpuVolumePartialProbe;
			gSceneCpuVolumeKinematicTargets[i] =
				PxVec4(
					initial +
						(inactiveProbe
							? PxVec3(0.0f, 5.0f, 0.0f)
							: PxVec3(0.0f)),
					inactiveProbe ? 1.0f : 0.0f);
		}
		gSceneCpuVolume->setKinematicTargetBufferH(
			gSceneCpuVolumeKinematicTargets.begin());
		if(fullKinematicTargetCase || rigidTriangleSteadyContactCase)
			gSceneCpuVolume->setDeformableBodyFlag(
				PxDeformableBodyFlag::eKINEMATIC, true);
		else
			gSceneCpuVolume->setDeformableVolumeFlag(
				PxDeformableVolumeFlag::
					ePARTIALLY_KINEMATIC, true);
		gMetrics.sceneVolumeTargetBound = 1;
	}
	gSceneCpuVolume->markDirty(PxDeformableVolumeDataFlag::eALL);
	gSceneCpuVolume->setSolverIterationCounts(
		partialElementFilterCase ? 16u : 8u, 1);
	if(gHeadlessOptions.caseName == "scene-volume-skinning" &&
		!initializeVolumeSkinning())
		return false;
	if(taskGraphPureSoftCase || taskGraphWriteBackCase)
	{
		// Keep the fixed no-contact workload awake. P2's pure-soft fixture is
		// stationary; P3 adds two independent ballistic volumes so prediction
		// task ranges exercise actual predicted-position writes while remaining
		// free of OGC ownership and contact coupling.
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(taskGraphWorldPlaneCase)
	{
		// P5.3b validation fixture: one dense component, one Scene-owned plane
		// and no self/other OGC source. Gravity remains enabled so the task
		// transaction observes both empty and populated candidate streams.
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(taskGraphRigidBoxSdfCase)
	{
		// P5.12b validation fixture: one dense component and one Scene-static
		// box. Gravity produces empty, discrete and speculative swept epochs;
		// the rigid target remains world-static and within the strict admission.
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		gMetrics.speculativeCcdFlagApplied =
			gSceneCpuVolume->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) ? 1u : 0u;
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(taskGraphRigidSphereSdfCase)
	{
		// P5.13b validation fixture: one dense component and one world-static
		// sphere. Speculative CCD drives both current and swept sphere SDF
		// families while the two OGC feature suffixes remain parent-owned.
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		gMetrics.speculativeCcdFlagApplied =
			gSceneCpuVolume->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) ? 1u : 0u;
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(taskGraphRigidCapsuleSdfCase)
	{
		// P5.14b exercises both current and swept capsule SDF families. The
		// feature suffixes remain parent-owned after their canonical fan-in.
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		gMetrics.speculativeCcdFlagApplied =
			gSceneCpuVolume->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) ? 1u : 0u;
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(taskGraphRigidConvexSdfCase)
	{
		// P5.15b exercises both current and swept convex SDF families. The
		// feature suffixes remain parent-owned after their canonical fan-in.
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		gMetrics.speculativeCcdFlagApplied =
			gSceneCpuVolume->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) ? 1u : 0u;
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(taskGraphRigidTriangleSurfaceAnyCase)
	{
		// P5.16b validates the current/swept triangle-SDF fan-in under the same
		// speculative-CCD admission used by the analytic SDF families. P5.19
		// reuses that unchanged admission for its separate threshold fixture.
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		gMetrics.speculativeCcdFlagApplied =
			gSceneCpuVolume->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) ? 1u : 0u;
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(taskGraphRigidTriangleSurfaceFeatureOverlapCase)
	{
		// Keep the dynamic body at the constructed reverse-feature overlap.
		// It remains non-kinematic, so this probes the same P5 transaction
		// admission as production dynamic soft bodies.
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
	}
	if(motionControlsCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setLinearDamping(0.0f);
		gSceneCpuVolume->setMaxLinearVelocity(1.0f);
		gSceneCpuVolume->setSettlingThreshold(0.0f);
		gSceneCpuVolume->setSettlingDamping(0.0f);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(maxDepenetrationVelocityCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setLinearDamping(0.0f);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
		gSceneCpuVolume->setMaxDepenetrationVelocity(0.12f);
	}
	if(speculativeCcdCase || rigidTriangleSteadyContactCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		gSceneCpuVolume->setLinearDamping(0.0f);
		gSceneCpuVolume->setSettlingThreshold(0.0f);
		gSceneCpuVolume->setSettlingDamping(0.0f);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(smoothReverseFeatureCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setDeformableBodyFlag(
			PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
		gSceneCpuVolume->setLinearDamping(0.0f);
		gSceneCpuVolume->setSettlingThreshold(0.0f);
		gSceneCpuVolume->setSettlingDamping(0.0f);
		gSceneCpuVolume->setSleepThreshold(0.0f);
		gSceneCpuVolume->setWakeCounter(1.0f);
	}
	if(softSleepCase)
	{
		gSceneCpuVolume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
		gSceneCpuVolume->setSleepThreshold(1.0e-4f);
	}
	gMetrics.sceneHostBuffersInitialized = 1;

	gSceneCpuVolumeInitialCentroidY = getSceneCpuVolumeCentroidY();
	gScene->addActor(*gSceneCpuVolume);
	if(gSceneCpuVolume->getScene() != gScene)
		return false;
	gMetrics.sceneActorAdded = 1;
	if(gHeadlessOptions.caseName == "scene-volume-world-pin" ||
		worldElementAttachmentCase)
	{
		const PxU32 attachmentVertex = 0;
		gSceneCpuWorldPinTarget = getSceneCpuAttachmentPoint();
		PxDeformableAttachmentData attachmentData;
		PxVec4 worldCoordinate(gSceneCpuWorldPinTarget, 0.0f);
		attachmentData.actor[0] = gSceneCpuVolume;
		attachmentData.type[0] =
			worldElementAttachmentCase
				? PxDeformableAttachmentTargetType::eTETRAHEDRON
				: PxDeformableAttachmentTargetType::eVERTEX;
		attachmentData.indices[0].data = &attachmentVertex;
		attachmentData.indices[0].count = 1;
		if(worldElementAttachmentCase)
		{
			attachmentData.coords[0].data =
				&gSceneCpuAttachmentBarycentric;
			attachmentData.coords[0].count = 1;
		}
		attachmentData.actor[1] = NULL;
		attachmentData.type[1] =
			PxDeformableAttachmentTargetType::eWORLD;
		attachmentData.coords[1].data = &worldCoordinate;
		attachmentData.coords[1].count = 1;
		gSceneCpuWorldAttachment =
			gPhysics->createDeformableAttachment(attachmentData);
		if(!gSceneCpuWorldAttachment)
			return false;
		gMetrics.sceneWorldPinCreated = 1;
	}
	if(attachmentCase)
	{
		const PxU32 attachmentVertex = 0;
		gSceneCpuRigidAttachmentLocalOffset =
			(kinematicAttachmentCase || staticAttachmentCase)
				? PxVec3(0.0f, 0.5f, 0.0f)
				: PxVec3(0.0f);
		gSceneCpuRigidAttachmentInitialPosition =
			getSceneCpuAttachmentPoint() -
				gSceneCpuRigidAttachmentLocalOffset;
		PxRigidActor* attachmentActor = NULL;
		if(articulationAttachmentCase)
		{
			gSceneCpuAttachmentArticulation =
				gPhysics->createArticulationReducedCoordinate();
			if(!gSceneCpuAttachmentArticulation)
				return false;
			gMetrics.sceneArticulationCreated = 1;
			gSceneCpuAttachmentArticulation->setArticulationFlag(
				PxArticulationFlag::eFIX_BASE, true);
			gSceneCpuAttachmentArticulation->
				setSolverIterationCounts(16, 1);
			gSceneCpuAttachmentArticulation->
				setSleepThreshold(5.0e-4f);
			gSceneCpuArticulationRootInitialPose =
				PxTransform(gSceneCpuRigidAttachmentInitialPosition);
			gSceneCpuArticulationChildInitialPose =
				gSceneCpuArticulationRootInitialPose;
			gSceneCpuAttachmentRoot =
				gSceneCpuAttachmentArticulation->createLink(
					NULL, gSceneCpuArticulationRootInitialPose);
			gSceneCpuAttachmentLink =
				gSceneCpuAttachmentArticulation->createLink(
					gSceneCpuAttachmentRoot,
					gSceneCpuArticulationChildInitialPose);
			if(!gSceneCpuAttachmentRoot ||
				!gSceneCpuAttachmentLink)
				return false;
			PxArticulationLink* links[2] = {
				gSceneCpuAttachmentRoot,
				gSceneCpuAttachmentLink
			};
			for(PxU32 linkIndex = 0; linkIndex < 2; ++linkIndex)
			{
				links[linkIndex]->setMass(1.0f);
				links[linkIndex]->setMassSpaceInertiaTensor(
					PxVec3(1.0f));
				links[linkIndex]->setActorFlag(
					PxActorFlag::eDISABLE_GRAVITY, true);
				links[linkIndex]->setLinearDamping(0.5f);
				links[linkIndex]->setAngularDamping(0.5f);
			}
			PxArticulationJointReducedCoordinate* joint =
				gSceneCpuAttachmentLink->getInboundJoint();
			if(!joint)
				return false;
			joint->setJointType(
				PxArticulationJointType::ePRISMATIC);
			joint->setMotion(
				PxArticulationAxis::eX,
				PxArticulationMotion::eFREE);
			joint->setParentPose(PxTransform(PxIdentity));
			joint->setChildPose(PxTransform(PxIdentity));
			if(!gScene->addArticulation(
				*gSceneCpuAttachmentArticulation))
				return false;
			gSceneCpuAttachmentArticulation->putToSleep();
			gMetrics.sceneArticulationAdded = 1;
			gMetrics.sceneArticulationInitiallySleeping =
				gSceneCpuAttachmentArticulation->isSleeping()
					? 1u : 0u;
			gSceneCpuAttachmentBody = gSceneCpuAttachmentLink;
			attachmentActor = gSceneCpuAttachmentLink;
		}
		else if(staticAttachmentCase)
		{
			gSceneCpuStaticActor =
				gPhysics->createRigidStatic(
					PxTransform(
						gSceneCpuRigidAttachmentInitialPosition));
			if(!gSceneCpuStaticActor ||
				!gScene->addActor(*gSceneCpuStaticActor))
				return false;
			attachmentActor = gSceneCpuStaticActor;
			gMetrics.sceneKinematicActorAdded = 1;
		}
		else
		{
			gSceneCpuDynamicActor =
				gPhysics->createRigidDynamic(
					PxTransform(
						gSceneCpuRigidAttachmentInitialPosition));
			if(!gSceneCpuDynamicActor)
				return false;
			if(kinematicAttachmentCase)
				gSceneCpuDynamicActor->setRigidBodyFlag(
					PxRigidBodyFlag::eKINEMATIC, true);
			else
			{
				gSceneCpuDynamicActor->setMass(1.0f);
				gSceneCpuDynamicActor->setMassSpaceInertiaTensor(
					PxVec3(1.0f));
			}
			gSceneCpuDynamicActor->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuDynamicActor->setLinearDamping(0.5f);
			gSceneCpuDynamicActor->setAngularDamping(0.5f);
			if(!gScene->addActor(*gSceneCpuDynamicActor))
				return false;
			if(!kinematicAttachmentCase)
				gSceneCpuDynamicActor->putToSleep();
			else
				gMetrics.sceneKinematicActorAdded = 1;
			gMetrics.sceneDynamicActorAdded = 1;
			gMetrics.sceneDynamicInitiallySleeping =
				gSceneCpuDynamicActor->isSleeping() ? 1u : 0u;
			gSceneCpuAttachmentBody = gSceneCpuDynamicActor;
			attachmentActor = gSceneCpuDynamicActor;
		}
		gMetrics.sceneRigidAttachmentActorAdded = 1;
		gMetrics.sceneRigidAttachmentInitiallySleeping =
			staticAttachmentCase
				? 1u
				: articulationAttachmentCase
				? gMetrics.sceneArticulationInitiallySleeping
				: (gSceneCpuDynamicActor->isSleeping() ? 1u : 0u);

		PxDeformableAttachmentData attachmentData;
		PxVec4 rigidCoordinate(
			gSceneCpuRigidAttachmentLocalOffset, 0.0f);
		attachmentData.actor[0] = gSceneCpuVolume;
		attachmentData.type[0] =
			gSceneCpuElementAttachment
				? PxDeformableAttachmentTargetType::eTETRAHEDRON
				: PxDeformableAttachmentTargetType::eVERTEX;
		attachmentData.indices[0].data = &attachmentVertex;
		attachmentData.indices[0].count = 1;
		if(gSceneCpuElementAttachment)
		{
			attachmentData.coords[0].data =
				&gSceneCpuAttachmentBarycentric;
			attachmentData.coords[0].count = 1;
		}
		attachmentData.actor[1] = attachmentActor;
		attachmentData.type[1] =
			PxDeformableAttachmentTargetType::eRIGID;
		attachmentData.coords[1].data = &rigidCoordinate;
		attachmentData.coords[1].count = 1;
		gSceneCpuRigidAttachment =
			gPhysics->createDeformableAttachment(attachmentData);
		if(!gSceneCpuRigidAttachment)
			return false;
		gMetrics.sceneRigidAttachmentCreated = 1;
		if(!kinematicAttachmentCase &&
			!staticAttachmentCase)
		{
			for(PxU32 i = 0; i < simulationVertexCount; ++i)
				simVelocity[i] =
					PxVec4(
						0.5f, 0.0f, 0.0f,
						simVelocity[i].w);
			gSceneCpuVolume->markDirty(
				PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		}
	}
	if(gHeadlessOptions.caseName ==
			"scene-volume-element-filter" ||
		partialElementFilterCase)
	{
		if(!gSceneCpuStaticActor)
			return false;
		const PxTetrahedronMesh* collisionMesh =
			gSceneCpuVolume->getCollisionMesh();
		const PxU32 elementCount =
			collisionMesh->getNbTetrahedrons();
		if(partialElementFilterCase && elementCount < 2)
			return false;
		const PxU32 filteredElementCount =
			partialElementFilterCase ? 1u : elementCount;
		PxArray<PxU32> filteredElements;
		filteredElements.resize(filteredElementCount);
		for(PxU32 i = 0; i < filteredElementCount; ++i)
			filteredElements[i] = i;
		if(partialElementFilterCase)
		{
			const bool has16BitIndices =
				collisionMesh->getTetrahedronMeshFlags() &
					PxTetrahedronMeshFlag::e16_BIT_INDICES;
			const PxU16* tets16 = has16BitIndices
				? static_cast<const PxU16*>(
					collisionMesh->getTetrahedrons())
				: NULL;
			const PxU32* tets32 = has16BitIndices
				? NULL
				: static_cast<const PxU32*>(
					collisionMesh->getTetrahedrons());
			const PxVec3* vertices = collisionMesh->getVertices();
			PxVec3 selectedCentroid(0.0f);
			for(PxU32 i = 0; i < 4; ++i)
			{
				const PxU32 vertex =
					has16BitIndices ? tets16[i] : tets32[i];
				if(vertex >= collisionMesh->getNbVertices())
					return false;
				selectedCentroid += vertices[vertex];
			}
			gSceneCpuPartialFilterSelectedPositiveX =
				selectedCentroid.x > 0.0f;
		}
		PxDeformableElementFilterData filterData;
		filterData.actor[0] = gSceneCpuVolume;
		filterData.actor[1] = gSceneCpuStaticActor;
		filterData.groupElementCounts[0].data =
			&filteredElementCount;
		filterData.groupElementCounts[0].count = 1;
		filterData.groupElementIndices[0].data =
			filteredElements.begin();
		filterData.groupElementIndices[0].count =
			filteredElementCount;
		gSceneCpuElementFilter =
			gPhysics->createDeformableElementFilter(filterData);
		if(!gSceneCpuElementFilter)
			return false;
		gMetrics.sceneElementFilterCreated = 1;
	}
	if(softSleepCase)
		gMetrics.sceneSoftInitiallyAwake =
			gSceneCpuVolume->isSleeping() ? 0u : 1u;

	if(twoSoftVolumeCase)
	{
		if(!createAdditionalSceneCpuVolume(
			gSceneCpuSecondVolume,
			smoothReverseFeatureCase
				? PxVec3(1.0f, 0.34f, 0.0f)
				: triangleSurfaceSweptCcdCase
				? PxVec3(1.0f, 0.0f, 0.0f)
				: sphereReverseSweptCcdCase
				? PxVec3(
					1.0f,
					deformingReverseSweptCcdCase
						? 0.45f
					: staticSphereReverseSweptCcdCase
						? (convexReverseSweptCcdCase
							? 1.1f : 0.55f)
						: 0.0f,
					0.0f)
				: rotationalCapsuleSpeculativeCcdCase
				? PxVec3(1.0f, 0.0f, 0.0f)
				: speculativeCcdCase
				? PxVec3(1.0f, 1.2f, 0.0f)
			: maxDepenetrationVelocityCase
			? PxVec3(3.0f, -1.05f, 0.0f)
			: taskGraphPipelineCase
			? PxVec3(10.0f, taskGraphPipelineHeight, 0.0f)
			: PxVec3(
					softSoftWakeCase ? 3.0f :
						(softPairAttachmentCase ? 0.0f : 10.0f),
					multiSceneIsolationCase ? 6.0f :
						4.0f,
					0.0f),
			multiSceneIsolationCase ? gSceneCpuSecondScene : NULL))
			return false;
		gMetrics.sceneSecondVolumeActorCreated = 1;
		gMetrics.sceneSecondVolumeHostBuffersInitialized = 1;
		gMetrics.sceneSecondVolumeActorAdded = 1;
		gSceneCpuSecondVolumeInitialCentroidY =
			getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
		if(taskGraphWriteBackCase)
		{
			// Every entry is far from the others, has no collision work and owns a
			// distinct nonzero prediction/write-back range. The four-way variant
			// extends the same contract to a dispatcher-4 evidence fixture.
			if(!configureSceneCpuTaskGraphVolume(
					*gSceneCpuSecondVolume,
					PxVec3(-0.20f, 0.15f, 0.0f)))
				return false;
			if(taskGraphWriteBackFourWayCase)
			{
				if(taskGraphWriteBackHeterogeneousCase &&
					gSceneCpuTaskGraphExtraMeshes.size() != 1)
					return false;
				const PxVec3 translations[2] =
				{
					PxVec3(-10.0f,
						taskGraphPipelineCase ? taskGraphPipelineHeight : 4.0f,
						10.0f),
					PxVec3(10.0f,
						taskGraphPipelineCase ? taskGraphPipelineHeight : 4.0f,
						10.0f)
				};
				const PxVec3 velocities[2] =
				{
					PxVec3(0.15f, 0.20f, 0.0f),
					PxVec3(-0.30f, -0.10f, 0.0f)
				};
				for(PxU32 extraIndex = 0; extraIndex < 2; ++extraIndex)
				{
					PxDeformableVolume* extraVolume = NULL;
					PxDeformableVolumeMesh* const extraMesh =
						taskGraphWriteBackHeterogeneousCase
							? gSceneCpuTaskGraphExtraMeshes[0] : NULL;
					if(!createAdditionalSceneCpuVolume(
							extraVolume, translations[extraIndex],
							NULL, extraMesh))
						return false;
					gSceneCpuTaskGraphExtraVolumes.pushBack(extraVolume);
					if(!configureSceneCpuTaskGraphVolume(
							*extraVolume, velocities[extraIndex]))
						return false;
				}
			}
		}
		else if(mixedSleepIslandCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setSettlingThreshold(0.0f);
			gSceneCpuSecondVolume->setSettlingDamping(0.0f);
			gSceneCpuSecondVolume->setSleepThreshold(0.0f);
			PxVec4* secondVelocities =
				gSceneCpuSecondVolume->getSimVelocityBufferH();
			const PxU32 secondVertexCount =
				gSceneCpuSecondVolume->getSimulationMesh()->
					getNbVertices();
			if(!secondVelocities)
				return false;
			for(PxU32 i = 0; i < secondVertexCount; ++i)
				secondVelocities[i] = PxVec4(
					PxVec3(0.0f, 0.05f, 0.0f),
					secondVelocities[i].w);
			gSceneCpuSecondVolume->markDirty(
				PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		}
		else if(softChurnCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setSleepThreshold(1.0e-4f);
		}
		else if(multiSceneIsolationCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setSleepThreshold(1.0e-4f);
			PxVec4* secondVelocities =
				gSceneCpuSecondVolume->getSimVelocityBufferH();
			const PxU32 secondVertexCount =
				gSceneCpuSecondVolume->getSimulationMesh()->
					getNbVertices();
			if(!secondVelocities)
				return false;
			for(PxU32 i = 0; i < secondVertexCount; ++i)
				secondVelocities[i] = PxVec4(
					PxVec3(0.0f),
					secondVelocities[i].w);
			gSceneCpuSecondVolume->markDirty(
				PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		}
		else if(softSoftWakeCase || softPairAttachmentCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setSleepThreshold(1.0e-4f);
		}
		else if(motionControlsCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setMaxLinearVelocity(
				PX_MAX_F32);
			gSceneCpuSecondVolume->setSettlingThreshold(0.1f);
			gSceneCpuSecondVolume->setSettlingDamping(10.0f);
			gSceneCpuSecondVolume->setSleepThreshold(0.05f);
			gSceneCpuSecondVolume->setWakeCounter(0.2f);
			if(!setSceneCpuVolumeVelocity(
					*gSceneCpuSecondVolume,
					PxVec3(0.08f, 0.0f, 0.0f)))
				return false;
			gSceneCpuMotionInitialCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			gMetrics.motionControlStayedAwake = 1;
		}
		else if(maxDepenetrationVelocityCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setDeformableBodyFlag(
				PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setSleepThreshold(0.0f);
			gSceneCpuSecondVolume->setWakeCounter(1.0f);
			gSceneCpuSecondVolume->setMaxDepenetrationVelocity(
				PX_MAX_F32);
			gSceneCpuDepenetrationInitialMinY =
				getSceneCpuVolumeMinY(gSceneCpuVolume);
			gSceneCpuDepenetrationControlInitialMinY =
				getSceneCpuVolumeMinY(gSceneCpuSecondVolume);
			gMetrics.depenetrationLimitApplied =
				PxAbs(
					gSceneCpuVolume->
						getMaxDepenetrationVelocity() -
					0.12f) <= 1.0e-6f &&
				gSceneCpuSecondVolume->
					getMaxDepenetrationVelocity() >
						1.0e20f ? 1u : 0u;
		}
		else if(speculativeCcdCase)
		{
			gSceneCpuSecondVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			gSceneCpuSecondVolume->setDeformableBodyFlag(
				PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setSettlingThreshold(0.0f);
			gSceneCpuSecondVolume->setSettlingDamping(0.0f);
			gSceneCpuSecondVolume->setSleepThreshold(0.0f);
			gSceneCpuSecondVolume->setWakeCounter(1.0f);
			if(deformingReverseSweptCcdCase
				? !setSceneCpuVolumeDeformingReverseVelocity(
					*gSceneCpuSecondVolume, -720.0f)
				: !setSceneCpuVolumeVelocity(
					*gSceneCpuSecondVolume,
					PxVec3(
						0.0f,
						triangleSurfaceSweptCcdCase
							? 0.0f :
						staticSphereReverseSweptCcdCase
							? -132.0f :
						(movingKinematicFiniteSpeculativeCcdCase ||
						 dynamicFiniteRelativeSweptCcdCase ||
						 movingSphereReverseSweptCcdCase)
							? 0.0f :
						capsuleSpeculativeCcdCase
							? -180.0f :
						convexSpeculativeCcdCase
							? -220.0f :
						finiteSmoothSpeculativeCcdCase
							? -160.0f : -120.0f,
						0.0f)))
				return false;
			gMetrics.speculativeCcdFlagApplied =
				gSceneCpuVolume->getDeformableBodyFlags().isSet(
					PxDeformableBodyFlag::
						eENABLE_SPECULATIVE_CCD) &&
				!gSceneCpuSecondVolume->
					getDeformableBodyFlags().isSet(
						PxDeformableBodyFlag::
							eENABLE_SPECULATIVE_CCD)
					? 1u : 0u;
			if(movingKinematicFiniteSpeculativeCcdCase ||
				dynamicFiniteRelativeSweptCcdCase ||
				sphereReverseSweptCcdCase ||
				triangleSurfaceSweptCcdCase)
			{
				gSceneCpuMovingSpherePositiveInitial =
					getSceneCpuVolumeCentroid(gSceneCpuVolume);
				gSceneCpuMovingSphereNegativeInitial =
					getSceneCpuVolumeCentroid(
						gSceneCpuSecondVolume);
			}
		}
		else if(smoothReverseFeatureCase)
		{
			PxDeformableVolume* volumes[2] =
			{
				gSceneCpuVolume,
				gSceneCpuSecondVolume
			};
			for(PxU32 volumeIndex = 0;
				volumeIndex < 2; ++volumeIndex)
			{
				PxDeformableVolume* volume = volumes[volumeIndex];
				volume->setActorFlag(
					PxActorFlag::eDISABLE_GRAVITY, true);
				volume->setDeformableBodyFlag(
					PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,
					true);
				volume->setLinearDamping(0.0f);
				volume->setSettlingThreshold(0.0f);
				volume->setSettlingDamping(0.0f);
				volume->setSleepThreshold(0.0f);
				volume->setWakeCounter(1.0f);
				if(!setSceneCpuVolumeVelocity(
						*volume, PxVec3(0.0f, -2.0f, 0.0f)))
					return false;
			}
			gSceneCpuSphereReversePositiveInitial =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			gSceneCpuSphereReverseNegativeInitial =
				getSceneCpuVolumeCentroid(gSceneCpuSecondVolume);
		}
		if(softPairAttachmentCase)
		{
			const PxU32 collisionElementCount =
				gSceneCpuVolume->getCollisionMesh()->
					getNbTetrahedrons();
			if(collisionElementCount == 0 ||
				gSceneCpuSecondVolume->getCollisionMesh()->
					getNbTetrahedrons() != collisionElementCount)
				return false;
			PxArray<PxU32> filteredElements;
			filteredElements.resize(collisionElementCount);
			for(PxU32 i = 0; i < collisionElementCount; ++i)
				filteredElements[i] = i;
			PxDeformableElementFilterData filterData;
			for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
			{
				filterData.actor[endpoint] = endpoint == 0
					? static_cast<PxActor*>(gSceneCpuVolume)
					: static_cast<PxActor*>(
						gSceneCpuSecondVolume);
				filterData.groupElementCounts[endpoint].data =
					&collisionElementCount;
				filterData.groupElementCounts[endpoint].count = 1;
				filterData.groupElementIndices[endpoint].data =
					filteredElements.begin();
				filterData.groupElementIndices[endpoint].count =
					collisionElementCount;
			}
			gSceneCpuElementFilter =
				gPhysics->createDeformableElementFilter(
					filterData);
			if(!gSceneCpuElementFilter)
				return false;

			const PxU32 elementIndex = 0;
			PxDeformableAttachmentData attachmentData;
			for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
			{
				attachmentData.actor[endpoint] = endpoint == 0
					? static_cast<PxActor*>(gSceneCpuVolume)
					: static_cast<PxActor*>(
						gSceneCpuSecondVolume);
				attachmentData.type[endpoint] =
					PxDeformableAttachmentTargetType::
						eTETRAHEDRON;
				attachmentData.indices[endpoint].data =
					&elementIndex;
				attachmentData.indices[endpoint].count = 1;
				attachmentData.coords[endpoint].data =
					&gSceneCpuAttachmentBarycentric;
				attachmentData.coords[endpoint].count = 1;
			}
			gSceneCpuRigidAttachment =
				gPhysics->createDeformableAttachment(
					attachmentData);
			if(!gSceneCpuRigidAttachment)
				return false;
			gMetrics.sceneRigidAttachmentActorAdded = 1;
			gMetrics.sceneRigidAttachmentCreated = 1;
			gSceneCpuVolume->setLinearDamping(0.0f);
			gSceneCpuVolume->setSettlingThreshold(0.0f);
			gSceneCpuVolume->setSettlingDamping(0.0f);
			gSceneCpuSecondVolume->setLinearDamping(0.0f);
			gSceneCpuSecondVolume->setSettlingThreshold(0.0f);
			gSceneCpuSecondVolume->setSettlingDamping(0.0f);
			if(!setSceneCpuVolumeVelocity(
					*gSceneCpuVolume,
					PxVec3(0.0f, -0.1f, 0.0f)) ||
				!setSceneCpuVolumeVelocity(
					*gSceneCpuSecondVolume,
					PxVec3(0.0f, 0.1f, 0.0f)))
				return false;
		}
	}
	const PxU32 expectedVolumeCount =
		taskGraphWriteBackFourWayCase ? 4u :
		twoSoftVolumeCase ? 2u : 1u;
	if(multiSceneIsolationCase)
	{
		if(!gSceneCpuSecondScene ||
			gScene->getNbDeformableVolumes() != 1 ||
			gSceneCpuSecondScene->getNbDeformableVolumes() != 1)
			return false;
	}
	else if(gScene->getNbDeformableVolumes() != expectedVolumeCount)
		return false;
	if(taskGraphWriteBackCase &&
		!initializeSceneCpuVolumeRestStates(expectedVolumeCount))
		return false;

	gMetrics.initialized = 1;
	gMetrics.particles = 0;
	gMetrics.softBodies = expectedVolumeCount;
	gMetrics.tetElements = 0;
	auto accumulateVolumeTopology = [](const PxDeformableVolume* volume)
	{
		const PxTetrahedronMesh* mesh =
			volume ? volume->getSimulationMesh() : NULL;
		if(!mesh)
			return;
		gMetrics.particles += mesh->getNbVertices();
		gMetrics.tetElements += mesh->getNbTetrahedrons();
	};
	accumulateVolumeTopology(gSceneCpuVolume);
	accumulateVolumeTopology(gSceneCpuSecondVolume);
	for(PxU32 volumeId = 0;
		volumeId < gSceneCpuTaskGraphExtraVolumes.size(); ++volumeId)
		accumulateVolumeTopology(
			gSceneCpuTaskGraphExtraVolumes[volumeId]);
	gMetrics.surfaceTriangles =
		(surfaceTriangles.size() / 3) * expectedVolumeCount;
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes =
		multiSceneIsolationCase ? 2u :
			gScene->getNbDeformableVolumes();
	gMetrics.minDetF = 1.0f;
	gMetrics.maxDetF = 1.0f;
	gMetrics.minBodyVolumeRatio = 1.0f;
	gMetrics.maxBodyVolumeRatio = 1.0f;
	captureScenePerformanceTopology();
	return true;
}

static PxReal getSceneCpuVolumeMaxSpeed(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getSimulationMesh())
		return PX_MAX_F32;
	const PxVec4* velocities =
		volume->getSimVelocityBufferH();
	const PxU32 vertexCount =
		volume->getSimulationMesh()->getNbVertices();
	if(!velocities)
		return PX_MAX_F32;
	PxReal maxSpeed = 0.0f;
	for(PxU32 i = 0; i < vertexCount; ++i)
		maxSpeed = PxMax(
			maxSpeed, velocities[i].getXYZ().magnitude());
	return maxSpeed;
}

static PxReal getSceneCpuVolumeMaxSpeed()
{
	return getSceneCpuVolumeMaxSpeed(gSceneCpuVolume);
}

static PxReal getSceneCpuVolumeMinY(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getSimulationMesh())
		return PX_MAX_F32;
	const PxVec4* positions =
		volume->getSimPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getSimulationMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return PX_MAX_F32;
	PxReal minY = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
		minY = PxMin(minY, positions[i].y);
	return minY;
}

static PxReal getSceneCpuVolumeCollisionMinY(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getCollisionMesh())
		return PX_MAX_F32;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return PX_MAX_F32;
	PxReal minY = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
		minY = PxMin(minY, positions[i].y);
	return minY;
}

static PxReal getSceneCpuVolumeCollisionMaxY(
	PxDeformableVolume* volume)
{
	if(!volume || !volume->getCollisionMesh())
		return -PX_MAX_F32;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return -PX_MAX_F32;
	PxReal maxY = -PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
		maxY = PxMax(maxY, positions[i].y);
	return maxY;
}

static bool getSceneCpuVolumeSphereMinSeparation(
	PxDeformableVolume* volume, PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh())
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	const PxVec3 sphereCenters[] =
	{
		PxVec3(-2.0f, 0.25f, 0.0f),
		PxVec3(-1.0f, 0.25f, 0.0f),
		PxVec3(-2.0f, 0.25f, 1.0f),
		PxVec3(-1.0f, 0.25f, 1.0f),
		PxVec3(1.0f, 0.25f, 0.0f),
		PxVec3(2.0f, 0.25f, 0.0f),
		PxVec3(1.0f, 0.25f, 1.0f),
		PxVec3(2.0f, 0.25f, 1.0f)
	};
	minimumSeparation = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		for(PxU32 sphereIndex = 0;
			sphereIndex < sizeof(sphereCenters) /
				sizeof(sphereCenters[0]);
			++sphereIndex)
		{
			minimumSeparation = PxMin(
				minimumSeparation,
				(positions[i].getXYZ() -
					sphereCenters[sphereIndex]).magnitude() -
					0.3f);
		}
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeSingleSphereMinSeparation(
	PxDeformableVolume* volume,
	const PxVec3& sphereCenter,
	PxReal sphereRadius,
	PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh() ||
		!sphereCenter.isFinite() ||
		!PxIsFinite(sphereRadius) || sphereRadius <= 0.0f)
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	minimumSeparation = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		const PxVec3 position = positions[i].getXYZ();
		if(!position.isFinite())
			return false;
		minimumSeparation = PxMin(
			minimumSeparation,
			(position - sphereCenter).magnitude() -
				sphereRadius);
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}


static bool getSceneCpuVolumeCapsuleClusterMinSeparation(
	PxDeformableVolume* volume, PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh())
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	const PxVec3 capsuleCenters[] =
	{
		PxVec3(-2.0f, 0.25f, 0.0f),
		PxVec3(-1.0f, 0.25f, 0.0f),
		PxVec3(-2.0f, 0.25f, 1.0f),
		PxVec3(-1.0f, 0.25f, 1.0f),
		PxVec3(1.0f, 0.25f, 0.0f),
		PxVec3(2.0f, 0.25f, 0.0f),
		PxVec3(1.0f, 0.25f, 1.0f),
		PxVec3(2.0f, 0.25f, 1.0f)
	};
	minimumSeparation = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		const PxVec3 position = positions[i].getXYZ();
		if(!position.isFinite())
			return false;
		for(PxU32 capsuleIndex = 0;
			capsuleIndex < sizeof(capsuleCenters) /
				sizeof(capsuleCenters[0]);
			++capsuleIndex)
		{
			minimumSeparation = PxMin(
				minimumSeparation,
				getCapsuleSignedSeparation(
					position, capsuleCenters[capsuleIndex],
					0.3f, 0.2f));
		}
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeSingleCapsuleMinSeparation(
	PxDeformableVolume* volume,
	const PxTransform& capsulePose,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight,
	PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh() ||
		!capsulePose.isValid() ||
		!PxIsFinite(capsuleRadius) || capsuleRadius <= 0.0f ||
		!PxIsFinite(capsuleHalfHeight) || capsuleHalfHeight < 0.0f)
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	minimumSeparation = PX_MAX_F32;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		const PxVec3 position = positions[i].getXYZ();
		if(!position.isFinite())
			return false;
		minimumSeparation = PxMin(
			minimumSeparation,
			getCapsuleSignedSeparation(
				position, capsulePose,
				capsuleRadius, capsuleHalfHeight));
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeConvexMinSeparation(
	PxDeformableVolume* volume,
	const PxVec3* convexCenters,
	PxU32 convexCount,
	PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh() ||
		!convexCenters || convexCount == 0 ||
		!gSceneCpuRigidConvexMesh)
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	const PxConvexMeshGeometry geometry(gSceneCpuRigidConvexMesh);
	minimumSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex < vertexCount; ++vertexIndex)
	{
		const PxVec3 position =
			positions[vertexIndex].getXYZ();
		if(!position.isFinite())
			return false;
		for(PxU32 convexIndex = 0;
			convexIndex < convexCount; ++convexIndex)
		{
			if(!convexCenters[convexIndex].isFinite())
				return false;
			const PxReal squaredDistance =
				PxGeometryQuery::pointDistance(
					position, geometry,
					PxTransform(convexCenters[convexIndex]));
			if(!PxIsFinite(squaredDistance) ||
				squaredDistance < 0.0f)
				return false;
			minimumSeparation = PxMin(
				minimumSeparation,
				PxSqrt(squaredDistance));
		}
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}

static bool getSceneCpuVolumeSingleConvexMinSeparation(
	PxDeformableVolume* volume,
	const PxTransform& convexPose,
	PxReal& minimumSeparation)
{
	if(!volume || !volume->getCollisionMesh() ||
		!convexPose.isValid() || !gSceneCpuRigidConvexMesh)
		return false;
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount =
		volume->getCollisionMesh()->getNbVertices();
	if(!positions || vertexCount == 0)
		return false;
	const PxConvexMeshGeometry geometry(gSceneCpuRigidConvexMesh);
	minimumSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex < vertexCount; ++vertexIndex)
	{
		const PxVec3 point = positions[vertexIndex].getXYZ();
		if(!point.isFinite())
			return false;
		const PxReal squaredDistance =
			PxGeometryQuery::pointDistance(
				point, geometry, convexPose);
		if(!PxIsFinite(squaredDistance) ||
			squaredDistance < 0.0f)
			return false;
		minimumSeparation = PxMin(
			minimumSeparation, PxSqrt(squaredDistance));
	}
	return PxIsFinite(minimumSeparation) &&
		minimumSeparation < PX_MAX_F32;
}


static bool updateSceneSoftSleepWake()
{
	if(gHeadlessOptions.caseName != "scene-volume-sleep-wake")
		return true;
	if(!gSceneCpuVolume)
		return false;

	if(gMetrics.sceneSoftFirstSlept &&
		!gMetrics.sceneSoftCounterWakeIssued &&
		gMetrics.completedFrames >=
			gMetrics.sceneSoftFirstSleepFrame + 2)
	{
		if(PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneSoftStableWhileSleeping = 1;
		gSceneCpuVolume->setWakeCounter(0.1f);
		gMetrics.sceneSoftCounterWakeIssued = 1;
	}

	if(gMetrics.sceneSoftSecondSlept &&
		!gMetrics.sceneSoftVelocityWakeIssued &&
		gMetrics.completedFrames >=
			gMetrics.sceneSoftSecondSleepFrame + 2)
	{
		if(PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuSecondSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneSoftStableWhileSleeping = 1;
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxTetrahedronMesh* sceneSimulationMesh =
			gSceneCpuVolume->getSimulationMesh();
		const PxU32 vertexCount =
			sceneSimulationMesh->getNbVertices();
		if(!velocities)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] = PxVec4(
				PxVec3(0.0f, 0.5f, 0.0f), velocities[i].w);
		gSceneCpuVelocityWakeCentroidY =
			getSceneCpuVolumeCentroidY();
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		gMetrics.sceneSoftVelocityWakeIssued = 1;
	}
	if(gMetrics.sceneSoftMovedAfterVelocityWake &&
		!gMetrics.sceneSoftVelocityStopIssued)
	{
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxTetrahedronMesh* sceneSimulationMesh =
			gSceneCpuVolume->getSimulationMesh();
		const PxU32 vertexCount =
			sceneSimulationMesh->getNbVertices();
		if(!velocities)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] =
				PxVec4(PxVec3(0.0f), velocities[i].w);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		gSceneCpuVolume->setWakeCounter(0.0f);
		gMetrics.sceneSoftVelocityStopIssued = 1;
	}
	return true;
}

static bool updateSceneSoftRigidWake()
{
	if(gHeadlessOptions.caseName != "scene-volume-rigid-wake")
		return true;
	if(!gSceneCpuVolume)
		return false;

	if(gMetrics.sceneSoftFirstSlept &&
		!gMetrics.sceneSoftRigidWakeActorAdded &&
		gMetrics.completedFrames >=
			gMetrics.sceneSoftFirstSleepFrame + 2)
	{
		if(PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneSoftStableWhileSleeping = 1;
		gSceneCpuRigidWakeCentroidY =
			getSceneCpuVolumeCentroidY();
		if(!addSceneDynamicBox(
			PxVec3(0.0f, 3.70f, 0.0f),
			PxVec3(4.0f, 0.25f, 4.0f), false))
			return false;
		if(!PxRigidBodyExt::setMassAndUpdateInertia(
			*gSceneCpuDynamicActor, 1000.0f))
			return false;
		gSceneCpuDynamicActor->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
		gSceneCpuDynamicActor->setSleepThreshold(0.0f);
		gSceneCpuDynamicActor->wakeUp();
		gMetrics.sceneSoftRigidWakeActorAdded = 1;
	}
	if(gMetrics.sceneSoftWokeByRigid &&
		!gMetrics.sceneSoftVelocityStopIssued)
	{
		if(!gSceneCpuDynamicActor ||
			gSceneCpuDynamicActor->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuDynamicActor);
		gMetrics.sceneDynamicActorRemoved =
			gSceneCpuDynamicActor->getScene() == NULL ? 1u : 0u;
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxU32 vertexCount =
			gSceneCpuVolume->getSimulationMesh()->getNbVertices();
		if(!velocities)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] =
				PxVec4(PxVec3(0.0f), velocities[i].w);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		gSceneCpuVolume->setWakeCounter(0.0f);
		gMetrics.sceneSoftVelocityStopIssued = 1;
	}
	return true;
}

static bool setSceneCpuVolumeVelocity(
	PxDeformableVolume& volume, const PxVec3& velocity)
{
	PxVec4* velocities = volume.getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		volume.getSimulationMesh();
	if(!velocities || !simulationMesh)
		return false;
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	for(PxU32 i = 0; i < vertexCount; ++i)
		velocities[i] = PxVec4(velocity, velocities[i].w);
	volume.markDirty(PxDeformableVolumeDataFlag::eSIM_VELOCITY);
	return true;
}

static bool setSceneCpuVolumeDeformingReverseVelocity(
	PxDeformableVolume& volume, PxReal speed)
{
	PxVec4* velocities = volume.getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		volume.getSimulationMesh();
	if(!velocities || !simulationMesh || !PxIsFinite(speed))
		return false;
	const PxVec3* restVertices = simulationMesh->getVertices();
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	if(!restVertices || vertexCount == 0)
		return false;
	PxU32 movingVertex = 0;
	PxReal minimumRestRadiusSq =
		restVertices[0].magnitudeSquared();
	for(PxU32 i = 1; i < vertexCount; ++i)
	{
		const PxReal restRadiusSq =
			restVertices[i].magnitudeSquared();
		if(restRadiusSq < minimumRestRadiusSq)
		{
			minimumRestRadiusSq = restRadiusSq;
			movingVertex = i;
		}
	}
	for(PxU32 i = 0; i < vertexCount; ++i)
		velocities[i] = PxVec4(
			i == movingVertex
				? PxVec3(0.0f, speed, 0.0f)
				: PxVec3(0.0f),
			velocities[i].w);
	volume.markDirty(PxDeformableVolumeDataFlag::eSIM_VELOCITY);
	return true;
}

static bool updateSceneSoftActorChurn()
{
	if(gHeadlessOptions.caseName != "scene-volume-soft-churn")
		return true;
	if(!gSceneCpuVolume || !gSceneCpuSecondVolume || !gScene)
		return false;
	if(gMetrics.completedFrames < 30)
		return true;

	const PxU32 phase = (gMetrics.completedFrames - 30) % 6;
	if(gSceneCpuSoftChurnMovePending)
	{
		PxDeformableVolume* movingVolume =
			phase == 1 ? gSceneCpuSecondVolume :
			(phase == 3 ? gSceneCpuVolume : NULL);
		if(!movingVolume || movingVolume->getScene() != gScene)
			return false;
		const PxReal displacement =
			getSceneCpuVolumeCentroidY(movingVolume) -
			gSceneCpuSoftChurnCentroidY;
		if(!PxIsFinite(displacement) ||
			displacement <= 1.0e-5f ||
			displacement >= 0.01f)
			return false;
		gMetrics.sceneSoftChurnPostCompactMoveCount++;
		gSceneCpuSoftChurnMovePending = false;
	}
	if(phase == 0)
	{
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gMetrics.sceneSoftChurnRemoveCount++;
		gSceneCpuSoftChurnCentroidY =
			getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
		if(!setSceneCpuVolumeVelocity(
			*gSceneCpuSecondVolume, PxVec3(0.0f, 0.05f, 0.0f)))
			return false;
		gSceneCpuSoftChurnMovePending = true;
	}
	else if(phase == 1)
	{
		if(!setSceneCpuVolumeVelocity(
			*gSceneCpuSecondVolume, PxVec3(0.0f)))
			return false;
		gSceneCpuSecondVolume->setWakeCounter(0.0f);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		gMetrics.sceneSoftChurnReaddCount++;
	}
	else if(phase == 2)
	{
		if(gSceneCpuSecondVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuSecondVolume);
		if(gSceneCpuSecondVolume->getScene() != NULL)
			return false;
		gMetrics.sceneSoftChurnRemoveCount++;
		gSceneCpuSoftChurnCentroidY =
			getSceneCpuVolumeCentroidY(gSceneCpuVolume);
		if(!setSceneCpuVolumeVelocity(
			*gSceneCpuVolume, PxVec3(0.0f, 0.05f, 0.0f)))
			return false;
		gSceneCpuSoftChurnMovePending = true;
	}
	else if(phase == 3)
	{
		if(!setSceneCpuVolumeVelocity(
			*gSceneCpuVolume, PxVec3(0.0f)))
			return false;
		gSceneCpuVolume->setWakeCounter(0.0f);
		if(gSceneCpuSecondVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuSecondVolume);
		if(gSceneCpuSecondVolume->getScene() != gScene ||
			gScene->getNbDeformableVolumes() != 2)
			return false;
		gMetrics.sceneSoftChurnReaddCount++;
		gMetrics.sceneSoftChurnCycles++;
		const PxBounds3 firstBounds =
			gSceneCpuVolume->getWorldBounds();
		const PxBounds3 secondBounds =
			gSceneCpuSecondVolume->getWorldBounds();
		const bool stable =
			firstBounds.isValid() &&
			firstBounds.minimum.isFinite() &&
			firstBounds.maximum.isFinite() &&
			secondBounds.isValid() &&
			secondBounds.minimum.isFinite() &&
			secondBounds.maximum.isFinite();
		if(gMetrics.sceneSoftChurnCycles == 1)
			gMetrics.sceneSoftChurnStable = stable ? 1u : 0u;
		else if(!stable)
			gMetrics.sceneSoftChurnStable = 0;
	}
	return true;
}

static bool updateSceneSoftBufferMutation()
{
	if(gHeadlessOptions.caseName !=
		"scene-volume-buffer-mutation")
		return true;
	if(!gSceneCpuVolume ||
		!gSceneCpuVolume->getSimulationMesh())
		return false;

	PxVec4* positions =
		gSceneCpuVolume->getSimPositionInvMassBufferH();
	PxVec4* velocities =
		gSceneCpuVolume->getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		gSceneCpuVolume->getSimulationMesh();
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	if(!positions || !velocities || vertexCount < 2)
		return false;

	if(gMetrics.sceneSoftFirstSlept &&
		!gMetrics.sceneBufferMutationIssued &&
		gMetrics.completedFrames >=
			gMetrics.sceneSoftFirstSleepFrame + 2)
	{
		if(PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneSoftStableWhileSleeping = 1;
		gSceneCpuBufferOriginalInvMass = positions[0].w;
		if(!PxIsFinite(gSceneCpuBufferOriginalInvMass) ||
			gSceneCpuBufferOriginalInvMass <= 0.0f)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			const PxVec3 translated =
				positions[i].getXYZ() + PxVec3(0.0f, 0.25f, 0.0f);
			positions[i] = PxVec4(translated, positions[i].w);
			velocities[i] =
				PxVec4(PxVec3(0.0f), velocities[i].w);
		}
		positions[0].w = 0.0f;
		gSceneCpuBufferPinTarget = positions[0].getXYZ();
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlags(
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_POSITION_INVMASS) |
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_VELOCITY)));
		gMetrics.sceneBufferMutationIssued = 1;
	}
	else if(gMetrics.sceneBufferMutationApplied &&
		gMetrics.sceneBufferPinHeld &&
		!gMetrics.sceneBufferDriveIssued)
	{
		gSceneCpuBufferDynamicBaseline =
			positions[1].getXYZ();
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			const PxVec3 velocity =
				i == 0 ? PxVec3(0.0f) :
					PxVec3(0.05f, 0.0f, 0.0f);
			velocities[i] =
				PxVec4(velocity, velocities[i].w);
		}
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		gMetrics.sceneBufferDriveIssued = 1;
	}
	else if(gMetrics.sceneBufferDynamicMoved &&
		gMetrics.sceneBufferPinHeld &&
		!gMetrics.sceneBufferInvMassRestored)
	{
		positions[0].w = gSceneCpuBufferOriginalInvMass;
		gSceneCpuBufferRestoredBaseline =
			positions[0].getXYZ();
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] = PxVec4(
				PxVec3(0.05f, 0.0f, 0.0f),
				velocities[i].w);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlags(
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_POSITION_INVMASS) |
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_VELOCITY)));
		gMetrics.sceneBufferInvMassRestored = 1;
	}
	else if(gMetrics.sceneBufferRestoredMoved &&
		!gMetrics.sceneBufferResetIssued)
	{
		const PxVec3* simulationVertices =
			simulationMesh->getVertices();
		if(!simulationVertices)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			positions[i] = PxVec4(
				simulationVertices[i] +
					PxVec3(0.0f, 4.25f, 0.0f),
				i == 0 ? gSceneCpuBufferOriginalInvMass :
					positions[i].w);
			velocities[i] =
				PxVec4(PxVec3(0.0f), velocities[i].w);
		}
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlags(
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_POSITION_INVMASS) |
				PxU32(
					PxDeformableVolumeDataFlag::
						eSIM_VELOCITY)));
		gSceneCpuVolume->setWakeCounter(0.0f);
		gMetrics.sceneBufferResetIssued = 1;
	}
	return true;
}

// ---------------------------------------------------------------------------
static bool initPhysicsInternal(
	bool interactive, const std::string& caseName)
{
	gMetrics = DeformableVolumeMetrics();
	gPerformance = DeformableVolumePerformanceMetrics();
	gPerformance.warmupFrames = gProfileWarmupFrames;
	gSceneCpuSoftChurnCentroidY = 0.0f;
	gSceneCpuSoftChurnMovePending = false;
	gSceneCpuBufferPinTarget = PxVec3(0.0f);
	gSceneCpuBufferDynamicBaseline = PxVec3(0.0f);
	gSceneCpuBufferRestoredBaseline = PxVec3(0.0f);
	gSceneCpuBufferOriginalInvMass = 0.0f;
	gSceneCpuMultiPrimaryRemovedCentroidY = 0.0f;
	gSceneCpuMultiSecondaryAtReleaseCentroidY = 0.0f;
	gSceneCpuSoftSoftTargetBaseline = PxVec3(0.0f);
	gSceneCpuVolumeKinematicTargets.clear();
	gSceneCpuVolumeKinematicInitial.clear();
	gVolumeSkinningBindings.clear();
	gVolumeSkinningTriangles.clear();
	gVolumeSkinningPositions.clear();
	gVolumeSkinningNormals.clear();
	gVolumeSkinningInitialPositions.clear();
	gVolumeAvbdSkinningRenderData =
		VolumeAvbdSkinningRenderData();
	gVolumeSkinningMetrics = VolumeSkinningMetrics();
	gSphereReverseFeatureMetrics =
		ReverseFeatureMetrics();
	gSphereReverseSweptMetrics =
		ReverseSweptMetrics();
	gDeformingVolumeReverseSweptMetrics =
		DeformingReverseSweptMetrics();
	gCapsuleRotationalSweepMetrics =
		RotationalSweepMetrics();
	gSceneCpuDeformingReverseSweptFreeEndPositions.reset();
	gSceneCpuCapsuleRotationalSweptInitialPositions.reset();
	gSceneCpuVolumePartialProbe = PX_MAX_U32;
	gSceneCpuVolumePartialActivationStart = PxVec3(0.0f);
	gSceneCpuKinematicCommandY = 0.0f;
	gSceneCpuKinematicSoftBaselineY = 0.0f;
	gSceneCpuMovingSpherePositiveInitial = PxVec3(0.0f);
	gSceneCpuMovingSphereNegativeInitial = PxVec3(0.0f);
	gSceneCpuSphereReversePositiveInitial = PxVec3(0.0f);
	gSceneCpuSphereReverseNegativeInitial = PxVec3(0.0f);
	gSceneCpuElementAttachment = false;
	for(PxU32 i = 0; i < 4; ++i)
		gSceneCpuAttachmentVertices[i] = 0;
	gSceneCpuRigidAttachmentInitialPosition = PxVec3(0.0f);
	gSceneCpuRigidAttachmentLocalOffset = PxVec3(0.0f);
	gSceneCpuKinematicAttachmentProgress = 0.0f;
	gSceneCpuKinematicAttachmentSoftBaseline = PxVec3(0.0f);
	gSceneCpuKinematicAttachmentCommand = PxTransform(PxIdentity);
	gSceneCpuAttachmentArticulation = NULL;
	gSceneCpuAttachmentRoot = NULL;
	gSceneCpuAttachmentLink = NULL;
	gSceneCpuAttachmentBody = NULL;
	gSceneCpuArticulationRootInitialPose = PxTransform(PxIdentity);
	gSceneCpuArticulationChildInitialPose = PxTransform(PxIdentity);
	gErrorCallback.reset();
	gParticles.clear();
	gSoftBodies.clear();
	gContacts.clear();
	gRigidBoxes.clear();
	gSoftBodyRenderData.clear();
	gInitialCentroids.clear();
	initOGCParams();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;

	const char* pvdCapturePath =
		std::getenv("PHYSX_AVBD_PVD_CAPTURE_PATH");
	const bool capturePvd = pvdCapturePath && pvdCapturePath[0];
	if(interactive || capturePvd)
	{
		gPvd = PxCreatePvd(*gFoundation);
		if(gPvd)
		{
			PxPvdTransport* transport = capturePvd
				? PxDefaultPvdFileTransportCreate(pvdCapturePath)
				: PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
			if(transport)
				gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
		}
	}

	gPhysics    = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	if(!gExtensionsInitialized)
		return false;

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	// The isolated sandwich is a contact-law unit fixture: removing gravity
	// keeps the two approach velocities exactly symmetric so a passing result
	// measures local soft/rigid OGC response, not a later free-fall/ground
	// event.  The public visual showcase remains gravity-driven.
	sceneDesc.gravity =
		caseName == "scene-volume-ogc-sandwich" ||
		caseName == "scene-volume-element-filter" ||
		caseName == "scene-volume-partial-element-filter"
		? PxVec3(0.0f) : PxVec3(0.0f, -9.81f, 0.0f);
	sceneDesc.solverType = interactive ?
		PxSolverType::eAVBD : gHeadlessOptions.solverType;
	if(caseName == "scene-volume-element-filter" ||
		caseName == "scene-volume-partial-element-filter")
		sceneDesc.flags |= PxSceneFlag::eDISABLE_SLEEPING;
	const PxU32 workerCount =
		interactive ? 2u : gHeadlessOptions.dispatcherThreads;
	gDispatcher = PxDefaultCpuDispatcherCreate(workerCount);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader  = PxDefaultSimulationFilterShader;
	if((caseName == "scene-volume-visual-showcase" ||
		caseName == "scene-volume-ogc-sandwich" ||
		caseName == "scene-volume-sphere-soft-soft-glancing" ||
		caseName == "scene-volume-soft-soft-torque" ||
		caseName == "scene-volume-ground-embedded-tet-probe") &&
		!enableSceneCpuFixtureCollisionTelemetry())
		return false;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;
	// Vertex-block nonlinear GS is intentionally serial until a colored
	// schedule has a numerical-equivalence gate.
	gPerformance.softWorkers = 1;
	gMetrics.solverReadbackMatched =
		gScene->getSolverType() == sceneDesc.solverType;
	if(caseName == "scene-volume-multi-scene-isolation")
	{
		gSceneCpuSecondScene = gPhysics->createScene(sceneDesc);
		if(!gSceneCpuSecondScene)
			return false;
		gMetrics.sceneSecondSceneCreated = 1;
		gMetrics.sceneSecondSceneSolverMatched =
			gSceneCpuSecondScene->getSolverType() ==
				sceneDesc.solverType ? 1u : 0u;
	}

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if ((interactive || capturePvd) && pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
	if(!gMaterial)
		return false;
	if(caseName == "scene-volume-visual-showcase")
		return initSceneCpuVolumeVisualShowcase();
	if(caseName == "scene-volume-ogc-sandwich")
		return initSceneCpuVolumeOgcSandwich();
	if(caseName == "scene-volume-sphere-long-roll")
		return initSceneCpuVolumeSphereLongRoll();
	if(caseName == "scene-volume-sphere-soft-soft-glancing")
		return initSceneCpuVolumeSphereSoftSoftGlancing();
	if(caseName == "scene-volume-soft-soft-torque")
		return initSceneCpuVolumeSoftSoftTorque();
	if(caseName == "scene-volume-ground-embedded-tet-probe")
		return initSceneCpuVolumeGroundEmbeddedTetProbe();

	if(isSceneCpuVolumeCase(caseName))
	{
		if(caseName == "scene-volume-ground" ||
			caseName == "scene-volume-taskgraph-world-plane" ||
			caseName == "scene-volume-taskgraph-pipeline" ||
			caseName == "scene-volume-element-filter" ||
			caseName == "scene-volume-partial-element-filter" ||
			caseName ==
				"scene-volume-max-depenetration-velocity")
		{
			PxRigidStatic* ground = PxCreatePlane(
				*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
			if(!ground)
				return false;
			gScene->addActor(*ground);
			if(caseName == "scene-volume-element-filter" ||
				caseName ==
					"scene-volume-partial-element-filter")
				gSceneCpuStaticActor = ground;
		}
		else if(caseName == "scene-volume-static-box" ||
			caseName == "scene-volume-taskgraph-rigid-box-sdf" ||
			caseName == "scene-volume-static-churn")
		{
			if(!addSceneStaticBox(
				PxVec3(0.0f, 0.5f, 0.0f),
				PxVec3(20.0f, 0.5f, 20.0f)))
				return false;
		}
		else if(caseName == "scene-volume-taskgraph-rigid-sphere-sdf")
		{
			// One static sphere leaves this fixture inside the strict P5.5b
			// eligibility envelope: no plane/box/other primitive source exists.
			const PxVec3 sphereCenter(0.0f, -19.5f, 0.0f);
			if(!addSceneStaticSphereCluster(&sphereCenter, 1, 20.0f))
				return false;
		}
		else if(caseName == "scene-volume-taskgraph-rigid-capsule-sdf")
		{
			// One large static capsule gives the dense fixture a stable contact
			// region while excluding plane/box/sphere/self paths by construction.
			if(!addSceneStaticCapsule(
				PxVec3(0.0f, -19.5f, 0.0f), 20.0f, 3.0f))
				return false;
		}
		else if(caseName == "scene-volume-taskgraph-rigid-convex-sdf")
		{
			if(!addSceneStaticConvex(PxVec3(0.0f, -1.2f, 0.0f), false))
				return false;
		}
		else if(caseName == "scene-volume-taskgraph-rigid-triangle-surface")
		{
			if(!addSceneStaticTriangleMesh(PxVec3(0.0f, -1.2f, 0.0f), false))
				return false;
		}
		else if(isSceneCpuVolumeTaskGraphRigidTriangleSurfaceFeatureOverlapCase(
					caseName))
		{
			// Match the reverse-feature tetra used by the steady-contact oracle.
			// The dense dynamic cube begins intersecting it, instead of requiring
			// a kinematic target path which is outside P5 task admission.
			if(!addSceneStaticTriangleMesh(
					PxVec3(-1.63f, 0.0f, 0.33f), true))
				return false;
		}
		else if(isSceneCpuVolumeTaskGraphRigidTriangleSurfaceLargeCase(
					caseName))
		{
			// One static actor owns two distinct triangle-surface shapes.  They
			// share the fixed 8x8 mesh hierarchy but remain two immutable Scene
			// surfaces, which produces four canonical edge/face plan rows for the
			// one large body.  The upper plane receives the falling body first;
			// the lower plane keeps the input genuinely multi-surface.
			if(!addSceneStaticTriangleSurfacePair(
					PxVec3(0.0f, 1.2f, 0.0f),
					PxVec3(0.0f, -1.2f, 0.0f), false, false))
				return false;
		}
		else if(isSceneCpuVolumeTaskGraphRigidTriangleSurfaceThresholdCase(
					caseName))
		{
			// P5.20 keeps P5.19's two static surfaces but uses a 6^3 body. The
			// default 128-particle policy therefore yields three children, while
			// the explicit candidate can reach the same four-worker cap.
			if(!addSceneStaticTriangleSurfacePair(
					PxVec3(0.0f, 1.2f, 0.0f),
					PxVec3(0.0f, -1.2f, 0.0f), false, false))
				return false;
		}
		else if(isSceneCpuVolumeRigidTriangleSteadyContactCase(
					caseName))
		{
			// The moving AVBD volume is kinematic; the rigid tetra stays at the
			// same local reverse-feature pose as the static correctness case.
			if(!addSceneStaticTriangleMesh(
					PxVec3(-1.63f, 0.0f, 0.33f), true))
				return false;
		}
		else if(isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
					caseName))
		{
			const bool reverseFeature =
				isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
					caseName);
			const bool heightField =
				isSceneCpuVolumeHeightFieldSweptCcdCase(
					caseName);
			const bool rotationalTarget =
				isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
					caseName);
			const PxReal startY = -1.1f;
			const PxVec3 positiveCenter = rotationalTarget
				? (reverseFeature
					? (heightField
						? PxVec3(
							-3.22791624f, -1.25f,
							0.233333334f)
						: PxVec3(
							-2.32810450f, -0.75f,
							0.333333343f))
					: (heightField
						? PxVec3(
							-3.56124949f, -1.25f, -0.1f)
						: PxVec3(
							-2.66143775f, -0.75f, 0.0f)))
				: reverseFeature
					? (heightField
						? PxVec3(-1.93f, startY, 0.03f)
						: PxVec3(-1.63f, startY, 0.33f))
					: (heightField
						? PxVec3(-2.0f, startY, 0.0f)
						: PxVec3(-1.5f, startY, 0.5f));
			const PxVec3 negativeCenter = rotationalTarget
				? (reverseFeature
					? (heightField
						? PxVec3(
							-0.227916166f, -1.25f,
							0.233333334f)
						: PxVec3(
							0.671895504f, -0.75f,
							0.333333343f))
					: (heightField
						? PxVec3(
							-0.561249495f, -1.25f, -0.1f)
						: PxVec3(
							0.338562191f, -0.75f, 0.0f)))
				: reverseFeature
					? (heightField
						? PxVec3(1.07f, startY, 0.03f)
						: PxVec3(1.37f, startY, 0.33f))
					: (heightField
						? PxVec3(1.0f, startY, 0.0f)
						: PxVec3(1.5f, startY, 0.5f));
			if(rotationalTarget
				? !addSceneRotatingKinematicTriangleSurfacePair(
					positiveCenter, negativeCenter,
					heightField)
				: !addSceneMovingKinematicTriangleSurfacePair(
					positiveCenter, negativeCenter, 1.1f,
					heightField, reverseFeature))
				return false;
		}
		else if(caseName ==
			"scene-volume-deforming-sphere-reverse-swept-ccd")
		{
			const PxVec3 sphereCenters[] =
			{
				PxVec3(-1.68f, 0.0f, 0.32f),
				PxVec3(1.32f, 0.0f, 0.32f)
			};
			if(!addSceneStaticSphereCluster(
					sphereCenters, 2, 0.18f))
				return false;
		}
		else if(caseName ==
			"scene-volume-deforming-capsule-reverse-swept-ccd")
		{
			const PxVec3 capsuleCenters[] =
			{
				PxVec3(-1.68f, 0.0f, 0.32f),
				PxVec3(1.32f, 0.0f, 0.32f)
			};
			if(!addSceneStaticCapsuleCluster(
					capsuleCenters, 2, 0.18f, 0.02f))
				return false;
		}
		else if(caseName ==
			"scene-volume-deforming-convex-reverse-swept-ccd")
		{
			const PxVec3 convexCenters[] =
			{
				PxVec3(-1.68f, 0.0f, 0.32f),
				PxVec3(1.32f, 0.0f, 0.32f)
			};
			if(!addSceneStaticConvexCluster(
					convexCenters, 2,
					eSCENE_CPU_CONVEX_DEFORMING_REVERSE_SWEPT))
				return false;
		}
		else if(caseName ==
			"scene-volume-static-sphere-reverse-swept-ccd")
		{
			const PxVec3 sphereCenters[] =
			{
				PxVec3(-1.35f, 0.0f, 0.15f),
				PxVec3(1.65f, 0.0f, 0.15f)
			};
			if(!addSceneStaticSphereCluster(
					sphereCenters, 2, 0.25f))
				return false;
		}
		else if(caseName ==
			"scene-volume-static-capsule-reverse-swept-ccd")
		{
			const PxVec3 capsuleCenters[] =
			{
				PxVec3(-1.35f, 0.0f, 0.15f),
				PxVec3(1.65f, 0.0f, 0.15f)
			};
			if(!addSceneStaticCapsuleCluster(
					capsuleCenters, 2, 0.25f, 0.02f))
				return false;
		}
		else if(caseName ==
			"scene-volume-static-convex-reverse-swept-ccd")
		{
			const PxVec3 convexCenters[] =
			{
				PxVec3(-1.35f, 0.0f, 0.15f),
				PxVec3(1.65f, 0.0f, 0.15f)
			};
			if(!addSceneStaticConvexCluster(
					convexCenters, 2,
					eSCENE_CPU_CONVEX_REVERSE_SWEPT))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-sphere-reverse-swept-ccd")
		{
			if(!addSceneMovingKinematicFinitePair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					-1.1f, 0.25f, 0.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-capsule-reverse-swept-ccd")
		{
			if(!addSceneMovingKinematicFinitePair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					-1.1f, 0.25f, 0.02f))
				return false;
		}
		else if(caseName ==
			"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd")
		{
			if(!addSceneRotatingKinematicCapsulePair(
					PxVec3(-1.45f, -1.0f, 0.25f),
					PxVec3(1.55f, -1.0f, 0.25f),
					0.1f, 1.0f,
					PxQuat(
						PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					PxQuat(
						PxPi * 5.0f / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f))))
				return false;
		}
		else if(caseName ==
			"scene-volume-rotating-kinematic-convex-reverse-swept-ccd")
		{
			if(!addSceneRotatingKinematicConvexPair(
					PxVec3(-1.45f, -0.85f, 0.25f),
					PxVec3(1.55f, -0.85f, 0.25f),
					PxQuat(
						PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					PxQuat(
						PxPi * 5.0f / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f))))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-convex-reverse-swept-ccd")
		{
			if(!addSceneMovingKinematicConvexPair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					-1.1f,
					eSCENE_CPU_CONVEX_REVERSE_SWEPT))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-sphere-reverse-swept-ccd")
		{
			if(!addSceneDynamicFiniteSweepPair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					0.25f, 0.0f, -132.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-capsule-reverse-swept-ccd")
		{
			if(!addSceneDynamicFiniteSweepPair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					0.25f, 0.02f, -132.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd")
		{
			if(!addSceneDynamicRotatingCapsulePair(
					PxVec3(-1.45f, -1.0f, 0.25f),
					PxVec3(1.55f, -1.0f, 0.25f),
					0.1f, 1.0f,
					PxQuat(
						PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					10.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-rotating-convex-reverse-swept-ccd")
		{
			if(!addSceneDynamicRotatingConvexPair(
					PxVec3(-1.45f, -0.85f, 0.25f),
					PxVec3(1.55f, -0.85f, 0.25f),
					PxQuat(
						PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					10.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-convex-reverse-swept-ccd")
		{
			if(!addSceneDynamicConvexSweepPair(
					PxVec3(-1.35f, 1.1f, 0.15f),
					PxVec3(1.65f, 1.1f, 0.15f),
					-132.0f,
					eSCENE_CPU_CONVEX_REVERSE_SWEPT))
				return false;
		}
		else if(caseName == "scene-volume-speculative-ccd")
		{
			if(!addSceneStaticBox(
				PxVec3(0.0f, 0.5f, 0.5f),
				PxVec3(4.0f, 0.05f, 2.0f)))
				return false;
		}
		else if(caseName ==
			"scene-volume-plane-speculative-ccd")
		{
			PxRigidStatic* plane = PxCreatePlane(
				*gPhysics,
				PxPlane(0.0f, 1.0f, 0.0f, -0.5f),
				*gMaterial);
			if(!plane)
				return false;
			gScene->addActor(*plane);
		}
		else if(caseName ==
			"scene-volume-sphere-speculative-ccd")
		{
			const PxVec3 sphereCenters[] =
			{
				PxVec3(-2.0f, 0.25f, 0.0f),
				PxVec3(-1.0f, 0.25f, 0.0f),
				PxVec3(-2.0f, 0.25f, 1.0f),
				PxVec3(-1.0f, 0.25f, 1.0f),
				PxVec3(1.0f, 0.25f, 0.0f),
				PxVec3(2.0f, 0.25f, 0.0f),
				PxVec3(1.0f, 0.25f, 1.0f),
				PxVec3(2.0f, 0.25f, 1.0f)
			};
			if(!addSceneStaticSphereCluster(
				sphereCenters,
				sizeof(sphereCenters) /
					sizeof(sphereCenters[0]),
				0.3f))
				return false;
		}
		else if(caseName ==
			"scene-volume-capsule-speculative-ccd")
		{
			const PxVec3 capsuleCenters[] =
			{
				PxVec3(-2.0f, 0.25f, 0.0f),
				PxVec3(-1.0f, 0.25f, 0.0f),
				PxVec3(-2.0f, 0.25f, 1.0f),
				PxVec3(-1.0f, 0.25f, 1.0f),
				PxVec3(1.0f, 0.25f, 0.0f),
				PxVec3(2.0f, 0.25f, 0.0f),
				PxVec3(1.0f, 0.25f, 1.0f),
				PxVec3(2.0f, 0.25f, 1.0f)
			};
			if(!addSceneStaticCapsuleCluster(
				capsuleCenters,
				sizeof(capsuleCenters) /
					sizeof(capsuleCenters[0]),
				0.3f, 0.2f))
				return false;
		}
		else if(caseName ==
			"scene-volume-convex-speculative-ccd")
		{
			const PxVec3 convexCenters[] =
			{
				PxVec3(-1.5f, 0.25f, 0.5f),
				PxVec3(1.5f, 0.25f, 0.5f)
			};
			if(!addSceneStaticConvexCluster(
					convexCenters, 2,
					eSCENE_CPU_CONVEX_SWEPT_BOX))
				return false;
		}
		else if(caseName ==
			"scene-volume-sphere-reverse-feature")
		{
			const PxVec3 sphereCenter(
				-1.75f, 0.0f, 0.25f);
			if(!addSceneStaticSphereCluster(
					&sphereCenter, 1, 0.3f))
				return false;
		}
		else if(caseName ==
			"scene-volume-capsule-reverse-feature")
		{
			if(!addSceneStaticCapsule(
				PxVec3(-1.75f, 0.0f, 0.25f),
				0.3f, 0.15f))
				return false;
		}
		else if(caseName ==
			"scene-volume-convex-reverse-feature")
		{
			if(!addSceneStaticConvex(
				PxVec3(-1.75f, 0.0f, 0.25f), true))
				return false;
		}
		else if(caseName ==
			"scene-volume-triangle-mesh-reverse-feature")
		{
			if(!addSceneStaticTriangleMesh(
				PxVec3(-1.75f, 0.0f, 0.25f), true))
				return false;
		}
		else if(caseName ==
			"scene-volume-heightfield-reverse-feature")
		{
			if(!addSceneStaticHeightField(
				PxVec3(-2.05f, 0.0f, -0.05f), true))
				return false;
		}
		else if(caseName ==
			"scene-volume-moving-kinematic-sphere-speculative-ccd")
		{
			if(!addSceneMovingKinematicFinitePair(
					PxVec3(-1.5f, 2.8f, 0.5f),
					PxVec3(1.5f, 2.8f, 0.5f),
					0.0f, 0.8f, 0.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-moving-kinematic-capsule-speculative-ccd")
		{
			if(!addSceneMovingKinematicFinitePair(
					PxVec3(-1.5f, 2.8f, 0.5f),
					PxVec3(1.5f, 2.8f, 0.5f),
					0.0f, 0.8f, 0.3f))
				return false;
		}
		else if(caseName ==
			"scene-volume-rotating-kinematic-capsule-speculative-ccd")
		{
			if(!addSceneRotatingKinematicCapsulePair(
					PxVec3(-2.9f, 0.0f, 0.0f),
					PxVec3(0.1f, 0.0f, 0.0f),
					0.1f, 1.0f,
					PxQuat(
						-PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					PxQuat(
						PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f))))
				return false;
		}
		else if(caseName ==
			"scene-volume-rotating-kinematic-convex-speculative-ccd")
		{
			if(!addSceneRotatingKinematicConvexPair(
					PxVec3(-2.9f, 0.0f, 0.0f),
					PxVec3(0.1f, 0.0f, 0.0f),
					PxQuat(
						-PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					PxQuat(
						PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f))))
				return false;
		}
		else if(caseName ==
			"scene-volume-moving-kinematic-convex-speculative-ccd")
		{
			if(!addSceneMovingKinematicConvexPair(
					PxVec3(-1.5f, 2.8f, 0.5f),
					PxVec3(1.5f, 2.8f, 0.5f),
					0.0f,
					eSCENE_CPU_CONVEX_SWEPT_BOX))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-sphere-relative-swept-ccd")
		{
			// Keep both control endpoints strictly outside the soft volume.  The
			// flag-off actor must tunnel completely through the body; an endpoint
			// overlap would legitimately enter the current-pose OGC path and is not
			// a valid negative control for swept response.
			if(!addSceneDynamicFiniteSweepPair(
					PxVec3(-1.5f, 3.2f, 0.5f),
					PxVec3(1.5f, 3.2f, 0.5f),
					0.8f, 0.0f, -192.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-capsule-relative-swept-ccd")
		{
			if(!addSceneDynamicFiniteSweepPair(
					PxVec3(-1.5f, 3.2f, 0.5f),
					PxVec3(1.5f, 3.2f, 0.5f),
					0.8f, 0.3f, -192.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-rotating-capsule-relative-swept-ccd")
		{
			if(!addSceneDynamicRotatingCapsulePair(
					SCENE_CPU_DYNAMIC_ROTATING_CAPSULE_SWEEP_CENTER,
					SCENE_CPU_DYNAMIC_ROTATING_CAPSULE_CONTROL_CENTER,
					0.1f, 1.0f,
					PxQuat(
						-PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					1.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-rotating-convex-relative-swept-ccd")
		{
			if(!addSceneDynamicRotatingConvexPair(
					PxVec3(-2.9f, 0.0f, 0.0f),
					PxVec3(0.1f, 0.0f, 0.0f),
					PxQuat(
						-PxPi / 6.0f,
						PxVec3(0.0f, 0.0f, 1.0f)),
					1.0f))
				return false;
		}
		else if(caseName ==
			"scene-volume-dynamic-convex-relative-swept-ccd")
		{
			if(!addSceneDynamicConvexSweepPair(
					PxVec3(-1.5f, 2.8f, 0.5f),
					PxVec3(1.5f, 2.8f, 0.5f),
					-132.0f,
					eSCENE_CPU_CONVEX_SWEPT_BOX))
				return false;
		}
		else if(caseName == "scene-volume-multi-soft-islands")
		{
			if(!addSceneDynamicBox(
				PxVec3(-10.0f, 2.5f, 0.0f),
				PxVec3(4.0f, 0.25f, 4.0f)) ||
				!addSceneSecondDynamicBox(
					PxVec3(10.0f, 2.5f, 0.0f),
					PxVec3(4.0f, 0.25f, 4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-multi-dynamic-box")
		{
			if(!addSceneDynamicBox(
				PxVec3(0.0f, 2.5f, 0.0f),
				PxVec3(0.45f, 0.25f, 4.0f)) ||
				!addSceneSecondDynamicBox(
					PxVec3(1.0f, 2.5f, 0.0f),
					PxVec3(0.45f, 0.25f, 4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-kinematic-box")
		{
			if(!addSceneKinematicBox(
				PxVec3(0.0f, 3.5f, 0.0f),
				PxVec3(4.0f, 0.25f, 4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-kinematic-sphere")
		{
			if(!addSceneKinematicSphere(
				PxVec3(0.0f, 3.4f, 0.0f), 0.5f))
				return false;
		}
		else if(caseName == "scene-volume-kinematic-capsule")
		{
			if(!addSceneKinematicCapsule(
				PxVec3(0.0f, 3.4f, 0.0f), 0.5f, 0.3f))
				return false;
		}
		else if(caseName == "scene-volume-kinematic-convex")
		{
			if(!addSceneKinematicConvex(
				PxVec3(0.0f, 3.2f, 0.0f)))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-triangle-mesh")
		{
			if(!addSceneKinematicTriangleMesh(
				PxVec3(0.0f, 3.2f, 0.0f)))
				return false;
		}
		else if(caseName ==
			"scene-volume-kinematic-heightfield")
		{
			if(!addSceneKinematicHeightField(
				PxVec3(-4.0f, 3.2f, -4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-dynamic-box" ||
			caseName == "scene-volume-true-boundary-dynamic-box" ||
			caseName == "scene-volume-dynamic-churn")
		{
			if(!addSceneDynamicBox(
				PxVec3(0.0f, 2.5f, 0.0f),
				PxVec3(4.0f, 0.25f, 4.0f)))
				return false;
		}
		else if(caseName == "scene-volume-dynamic-sphere")
		{
			PxRigidStatic* ground = PxCreatePlane(
				*gPhysics,
				PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
				*gMaterial);
			if(!ground)
				return false;
			gScene->addActor(*ground);
			if(!addSceneDynamicSphere(
				PxVec3(0.0f, 1.0f, 0.0f), 0.8f))
				return false;
		}
		else if(caseName == "scene-volume-dynamic-capsule")
		{
			PxRigidStatic* ground = PxCreatePlane(
				*gPhysics,
				PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
				*gMaterial);
			if(!ground)
				return false;
			gScene->addActor(*ground);
			if(!addSceneDynamicCapsule(
				PxVec3(0.0f, 1.0f, 0.0f), 0.8f, 0.3f))
				return false;
		}
		else if(caseName == "scene-volume-dynamic-convex")
		{
			PxRigidStatic* ground = PxCreatePlane(
				*gPhysics,
				PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
				*gMaterial);
			if(!ground)
				return false;
			gScene->addActor(*ground);
			if(!addSceneDynamicConvex(
				PxVec3(0.0f, 1.0f, 0.0f)))
				return false;
		}
		return initSceneCpuVolumeLifecycle();
	}

	const bool denseNoContactCase =
		isComponentDenseNoContactCase(caseName);
	const bool manySmallNoContactCase =
		isComponentManySmallNoContactCase(caseName);
	const bool noContactCase = denseNoContactCase || manySmallNoContactCase;
	// These fixed pure-soft workloads intentionally have no rigid actor and no
	// gravity.  They are corpus members, not controls that may acquire contact
	// work as the implementation evolves.
	if(!noContactCase)
	{
		PxRigidStatic* ground = PxCreatePlane(
			*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
		if(!ground)
			return false;
		gScene->addActor(*ground);
	}

	if(caseName == "volume-ground")
	{
		addCubeSoftBody(PxVec3(0.0f, 3.0f, 0.0f), 0.5f, 3);
	}
	else if(denseNoContactCase)
	{
		// 12^3 cells create a stable M-sized workload (2,197 vertices and
		// 8,640 tetrahedra) without collision candidates.  Preserve this
		// topology in the P0 report; do not tune it down for later gates.
		addCubeSoftBody(PxVec3(0.0f, 3.0f, 0.0f), 2.0f, 12);
	}
	else if(manySmallNoContactCase)
	{
		// Fixed 4x4 array of independent small islands.  The spacing is
		// deliberately wider than the bodies, so a future scheduler is
		// measured on island work rather than accidental soft-soft contacts.
		for(PxU32 z = 0; z < 4; ++z)
		{
			for(PxU32 x = 0; x < 4; ++x)
			{
				addCubeSoftBody(
					PxVec3(3.0f * PxReal(x), 3.0f,
						3.0f * PxReal(z)),
					0.5f, 3);
			}
		}
	}
	else if(caseName == "volume-static-box")
	{
		addCubeSoftBody(PxVec3(0.0f, 4.0f, 0.0f), 0.5f, 3);
		if(!addRigidBox(
			PxVec3(0.0f, 0.5f, 0.0f),
			PxVec3(2.0f, 0.5f, 2.0f)))
		{
			return false;
		}
	}
	else if(caseName == "soft-soft")
	{
		addCubeSoftBody(PxVec3(0.0f, 1.0f, 0.0f), 0.5f, 3);
		addCubeSoftBody(PxVec3(0.0f, 4.0f, 0.0f), 0.5f, 3);
	}
	else if(caseName == "cone-ground")
	{
		addConeSoftBody(PxVec3(-0.8f, 11.0f, 1.2f));
	}
	else
	{

	// ------------------------------------------------------------------
	// Body 0: Tilted cuboid for visible soft-soft tumbling
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		const PxVec3 center(-1.8f, 8.0f, 0.0f);
		avbdGenerateSubdividedCubeTets(center, 1.0f, 4, verts, tets);
		scaleVerticesAboutCenter(verts, center, PxVec3(1.8f, 0.65f, 0.9f));
		rotateVerticesAroundZ(verts, center, -0.55f);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 160.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 1: Sphere restored as the soft-soft support body
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		avbdGenerateSubdividedSphereTets(PxVec3(-3.8f, 2.0f, 0.0f), 1.8f, 4, verts, tets);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 130.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 2: Cone glancing into the left stack
	//   Layered conical tets preserve the tapered collision surface.
	// ------------------------------------------------------------------
	addConeSoftBody(PxVec3(-0.8f, 11.0f, 1.2f));

	// ------------------------------------------------------------------
	// Body 3: Tilted cuboid (rigid-soft toppling rotation)
	//   Pre-rotated and offset on a narrow edge so rigid-soft torque is obvious.
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		PxVec3 center(7.0f, 4.2f, 0.0f);
		avbdGenerateSubdividedCubeTets(center, 1.0f, 3, verts, tets);
		scaleVerticesAboutCenter(verts, center, PxVec3(1.7f, 0.7f, 0.9f));
		rotateVerticesAroundZ(verts, center, 0.95f);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 160.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 4: Off-center follower that keeps Body 3 rotating after impact.
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		const PxVec3 center(5.4f, 8.8f, 0.3f);
		avbdGenerateSubdividedCubeTets(center, 0.85f, 3, verts, tets);
		rotateVerticesAroundZ(verts, center, -0.28f);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 120.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Rigid box obstacle (narrow support edge for Body 3)
	// ------------------------------------------------------------------
	if(!addRigidBox(
		PxVec3(7.6f, 0.55f, 0.0f),
		PxVec3(0.7f, 0.55f, 2.2f)))
	{
		return false;
	}
	}

	if(gSoftBodies.empty() || gParticles.empty())
		return false;

	// Contact prep can emit more than one surface feature per particle.  Keep
	// the capacity policy caller-owned and reserve outside the timed loop.
	const PxU32 contactCapacity = gParticles.size() * 4;
	gSoftWorkspace.reserve(gParticles.size(), contactCapacity);
	gContacts.reserve(contactCapacity);

	updateRenderData();
	gInitialCentroids.reserve(gSoftBodies.size());
	for(PxU32 bodyId = 0; bodyId < gSoftBodies.size(); ++bodyId)
		gInitialCentroids.pushBack(getSoftBodyCentroid(gSoftBodies[bodyId]));

	gMetrics.initialized = 1;
	gMetrics.particles = gParticles.size();
	gMetrics.softBodies = gSoftBodies.size();
	gMetrics.rigidBoxes = gRigidBoxes.size();
	for(PxU32 bodyId = 0; bodyId < gSoftBodies.size(); ++bodyId)
	{
		gMetrics.tetElements += gSoftBodies[bodyId].compiled.tetElements.size();
		gMetrics.surfaceTriangles +=
			gSoftBodies[bodyId].compiled.surfaceTriangles.size() / 3;
	}
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes = gScene->getNbDeformableVolumes();
	capturePerformanceTopology();

	printf("%s: %u particles, %u soft bodies, %u rigid boxes\n",
		AVBD_VOLUME_SNIPPET_NAME,
		gParticles.size(), gSoftBodies.size(), gRigidBoxes.size());
	printf(
		"[AVBD_COMPONENT_TOPOLOGY] particles=%u softBodies=%u "
		"tetElements=%u surfaceTriangles=%u rigidBoxes=%u "
		"sceneStatics=%u sceneDynamics=%u sceneDeformableVolumes=%u\n",
		gMetrics.particles, gMetrics.softBodies, gMetrics.tetElements,
		gMetrics.surfaceTriangles, gMetrics.rigidBoxes,
		gMetrics.sceneStatics, gMetrics.sceneDynamics,
		gMetrics.sceneDeformableVolumes);
	return true;
}

void initPhysics(bool interactive)
{
	gHeadlessOptions.solverType = PxSolverType::eAVBD;
	gHeadlessOptions.caseName = AVBD_VOLUME_VISUAL_CASE;
	gHeadlessOptions.frames = 600;
	gHeadlessOptions.dispatcherThreads = 2;
	if(!initPhysicsInternal(
		interactive, gHeadlessOptions.caseName))
		printf("%s initialization failed.\n",
			AVBD_VOLUME_SNIPPET_NAME);
}

// ---------------------------------------------------------------------------
// Contact re-detection callback for use inside avbdStepSoftBodies outer loop.
// Re-creates all ground + soft-soft contacts with fresh surface positions.
// ---------------------------------------------------------------------------
static void redetectContacts(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts, void* /*userData*/)
{
	AvbdSoftCollisionStats stats;
	avbdDetectAllOGCContacts(
		particles, numParticles,
		softBodies, numSoftBodies,
		gRigidBoxes.begin(), gRigidBoxes.size(),
		NULL, 0,
		contacts, gOGCParams, 0.0f, &stats,
		&gSoftWorkspace.contact);
	gFrameCollisionStats.accumulate(stats);
}

static void recordContactMetrics()
{
	PxU32 groundContacts = 0;
	PxU32 rigidContacts = 0;
	PxU32 softContacts = 0;
	for(PxU32 contactId = 0; contactId < gContacts.size(); ++contactId)
	{
		const AvbdSoftContact& contact = gContacts[contactId];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		if(!geometry.source.isValid())
			gMetrics.invalidContactSourceSamples++;
		if(geometry.source.type == AvbdSoftContactSource::eGROUND)
			groundContacts++;
		else if(
			geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE ||
			geometry.source.type == AvbdSoftContactSource::eSELF_SURFACE)
			softContacts++;
		else if(geometry.source.type == AvbdSoftContactSource::eRIGID_SDF)
			rigidContacts++;
		else
		{
			// Legacy fallback remains diagnostic-only while all production
			// detectors migrate to explicit source identities.
			if(geometry.targetKind ==
				AvbdSoftContactTargetKind::eWORLD_STATIC)
				groundContacts++;
			else if(geometry.targetKind ==
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE)
				softContacts++;
			else
				rigidContacts++;
		}
	}
	if(groundContacts)
		gMetrics.groundContactFrames++;
	if(rigidContacts)
		gMetrics.rigidContactFrames++;
	if(softContacts)
		gMetrics.softContactFrames++;
	gMetrics.maxGroundContacts =
		PxMax(gMetrics.maxGroundContacts, groundContacts);
	gMetrics.maxRigidContacts =
		PxMax(gMetrics.maxRigidContacts, rigidContacts);
	gMetrics.maxSoftContacts =
		PxMax(gMetrics.maxSoftContacts, softContacts);
}

static void recordStateMetrics()
{
	for(PxU32 particleId = 0; particleId < gParticles.size(); ++particleId)
	{
		const AvbdSoftParticle& particle = gParticles[particleId];
		if(!particle.position.isFinite() || !particle.velocity.isFinite())
		{
			gMetrics.nonFiniteParticleSamples++;
			continue;
		}
		gMetrics.minY = PxMin(gMetrics.minY, particle.position.y);
		gMetrics.maxY = PxMax(gMetrics.maxY, particle.position.y);
		gMetrics.maxParticleSpeed =
			PxMax(gMetrics.maxParticleSpeed, particle.velocity.magnitude());
	}

	for(PxU32 bodyId = 0; bodyId < gSoftBodies.size(); ++bodyId)
	{
		const AvbdSoftBody& body = gSoftBodies[bodyId];
		PxReal restVolume = 0.0f;
		PxReal currentVolume = 0.0f;
		for(PxU32 elementId = 0;
			elementId < body.compiled.tetElements.size(); ++elementId)
		{
			const AvbdTetElement& tet = body.compiled.tetElements[elementId];
			const PxVec3& x0 = gParticles[tet.p0].position;
			const PxMat33 ds(
				gParticles[tet.p1].position - x0,
				gParticles[tet.p2].position - x0,
				gParticles[tet.p3].position - x0);
			const PxReal detF = (ds * tet.DmInv).getDeterminant();
			restVolume += tet.restVolume;
			if(!PxIsFinite(detF))
			{
				gMetrics.invertedElementSamples++;
				if(gMetrics.firstInversionFrame == PX_MAX_U32)
				{
					gMetrics.firstInversionFrame = gMetrics.completedFrames;
					gMetrics.firstInversionBody = bodyId;
					gMetrics.firstInversionElement = elementId;
				}
				if(bodyId < 32)
					gMetrics.invertedBodiesMask |= 1u << bodyId;
				continue;
			}
			gMetrics.minDetF = PxMin(gMetrics.minDetF, detF);
			gMetrics.maxDetF = PxMax(gMetrics.maxDetF, detF);
			if(detF <= 0.0f)
			{
				gMetrics.invertedElementSamples++;
				if(gMetrics.firstInversionFrame == PX_MAX_U32)
				{
					gMetrics.firstInversionFrame = gMetrics.completedFrames;
					gMetrics.firstInversionBody = bodyId;
					gMetrics.firstInversionElement = elementId;
				}
				if(bodyId < 32)
					gMetrics.invertedBodiesMask |= 1u << bodyId;
			}
			currentVolume += detF * tet.restVolume;
		}
		if(restVolume > 0.0f)
		{
			const PxReal ratio = currentVolume / restVolume;
			if(PxIsFinite(ratio))
			{
				gMetrics.minBodyVolumeRatio =
					PxMin(gMetrics.minBodyVolumeRatio, ratio);
				gMetrics.maxBodyVolumeRatio =
					PxMax(gMetrics.maxBodyVolumeRatio, ratio);
			}
			else
				gMetrics.nonFiniteParticleSamples++;
		}
		const PxVec3 centroid = getSoftBodyCentroid(body);
		if(centroid.isFinite() && bodyId < gInitialCentroids.size())
		{
			gMetrics.maxCentroidDrop = PxMax(
				gMetrics.maxCentroidDrop,
				gInitialCentroids[bodyId].y - centroid.y);
		}
	}
}

static bool translateSceneCpuVolumeState(
	PxDeformableVolume& volume, const PxVec3& translation)
{
	PxVec4* positions = volume.getSimPositionInvMassBufferH();
	PxVec4* velocities = volume.getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		volume.getSimulationMesh();
	if(!positions || !velocities || !simulationMesh)
		return false;
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		positions[i] = PxVec4(
			positions[i].getXYZ() + translation,
			positions[i].w);
		velocities[i] =
			PxVec4(PxVec3(0.0f), velocities[i].w);
	}
	volume.markDirty(
		PxDeformableVolumeDataFlags(
			PxU32(
				PxDeformableVolumeDataFlag::
					eSIM_POSITION_INVMASS) |
			PxU32(
				PxDeformableVolumeDataFlag::
					eSIM_VELOCITY)));
	return true;
}

static bool resetSceneCpuVolumeState(
	PxDeformableVolume& volume, const PxVec3& translation)
{
	PxVec4* positions = volume.getSimPositionInvMassBufferH();
	PxVec4* velocities = volume.getSimVelocityBufferH();
	const PxTetrahedronMesh* simulationMesh =
		volume.getSimulationMesh();
	if(!positions || !velocities || !simulationMesh)
		return false;
	const PxVec3* restPositions = simulationMesh->getVertices();
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	if(!restPositions)
		return false;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		positions[i] = PxVec4(
			restPositions[i] + translation, positions[i].w);
		velocities[i] =
			PxVec4(PxVec3(0.0f), velocities[i].w);
	}
	volume.markDirty(
		PxDeformableVolumeDataFlags(
			PxU32(
				PxDeformableVolumeDataFlag::
					eSIM_POSITION_INVMASS) |
			PxU32(
				PxDeformableVolumeDataFlag::
					eSIM_VELOCITY)));
	return true;
}

static bool updateSceneSoftSoftWake()
{
	if(gHeadlessOptions.caseName !=
		"scene-volume-soft-soft-wake")
		return true;
	if(!gSceneCpuVolume || !gSceneCpuSecondVolume)
		return false;
	if(gMetrics.sceneSoftSoftTargetMoved &&
		!gMetrics.sceneSoftSoftResetIssued)
	{
		if(!resetSceneCpuVolumeState(
				*gSceneCpuVolume, PxVec3(0.0f, 4.0f, 0.0f)) ||
			!resetSceneCpuVolumeState(
				*gSceneCpuSecondVolume, PxVec3(3.0f, 4.0f, 0.0f)))
			return false;
		gSceneCpuVolume->setWakeCounter(0.0f);
		gSceneCpuSecondVolume->setWakeCounter(0.0f);
		gMetrics.sceneSoftSoftResetIssued = 1;
		return true;
	}
	if(!gMetrics.sceneSoftSoftBothSlept ||
		gMetrics.sceneSoftSoftDriveIssued)
		return true;

	gSceneCpuSoftSoftTargetBaseline =
		getSceneCpuVolumeCentroid(gSceneCpuVolume);
	if(!translateSceneCpuVolumeState(
			*gSceneCpuSecondVolume,
			PxVec3(-2.05f, 0.0f, 0.0f)))
		return false;
	gMetrics.sceneSoftSoftDriveIssued = 1;
	return true;
}

static bool stepSceneMultiSceneIsolation(PxReal dt)
{
	if(!gSceneCpuSecondScene || !gSceneCpuVolume ||
		!gSceneCpuSecondVolume)
		return false;

	const bool profileFrame =
		gMetrics.completedFrames >= gProfileWarmupFrames;
	PxTime frameTimer;
	if((gMetrics.completedFrames == 30 ||
			gMetrics.completedFrames == 92) &&
		!translateSceneCpuVolumeState(
			*gSceneCpuSecondVolume,
			PxVec3(0.0f, -0.08f, 0.0f)))
		return false;
	if(gMetrics.completedFrames == 60)
	{
		if(!gScene || gSceneCpuVolume->getScene() != gScene ||
			gScene->getNbDeformableVolumes() != 1)
			return false;
		gSceneCpuMultiPrimaryRemovedCentroidY =
			getSceneCpuVolumeCentroidY();
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL ||
			gScene->getNbDeformableVolumes() != 0)
			return false;
		gMetrics.sceneActorRemoved = 1;
	}
	if(gMetrics.sceneActorRemoved)
	{
		const PxReal detachedDeviation = PxAbs(
			getSceneCpuVolumeCentroidY() -
			gSceneCpuMultiPrimaryRemovedCentroidY);
		if(!PxIsFinite(detachedDeviation) ||
			detachedDeviation > 1.0e-6f)
			return false;
		if(gMetrics.completedFrames >= 62)
			gMetrics.sceneMultiPrimaryDetachedStable = 1;
	}
	if(gMetrics.completedFrames == 90)
	{
		if(!gScene || gScene->getNbDeformableVolumes() != 0 ||
			gSceneCpuVolume->getScene() != NULL)
			return false;
		gSceneCpuMultiSecondaryAtReleaseCentroidY =
			getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
		PX_RELEASE(gScene);
		gMetrics.scenePrimarySceneReleased = 1;
	}

	PxTime sceneTimer;
	if(gScene)
	{
		gScene->simulate(dt);
		if(!gScene->fetchResults(true))
		{
			gMetrics.fetchFailures++;
			return false;
		}
	}
	gSceneCpuSecondScene->simulate(dt);
	if(!gSceneCpuSecondScene->fetchResults(true))
	{
		gMetrics.fetchFailures++;
		return false;
	}
	const PxF64 sceneMs =
		sceneTimer.getElapsedSeconds() * 1000.0;
	gMetrics.completedFrames++;

	PxDeformableVolume* volumes[2] =
	{
		gSceneCpuVolume,
		gSceneCpuSecondVolume
	};
	for(PxU32 volumeId = 0; volumeId < 2; ++volumeId)
	{
		PxDeformableVolume* volume = volumes[volumeId];
		const PxVec4* positions =
			volume->getSimPositionInvMassBufferH();
		const PxVec4* velocities =
			volume->getSimVelocityBufferH();
		const PxU32 vertexCount =
			volume->getSimulationMesh()->getNbVertices();
		if(!positions || !velocities)
			return false;
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			const PxVec3 position = positions[i].getXYZ();
			const PxVec3 velocity = velocities[i].getXYZ();
			if(!position.isFinite() || !velocity.isFinite() ||
				!PxIsFinite(positions[i].w) ||
				!PxIsFinite(velocities[i].w))
			{
				gMetrics.nonFiniteParticleSamples++;
				continue;
			}
			gMetrics.minY = PxMin(gMetrics.minY, position.y);
			gMetrics.maxY = PxMax(gMetrics.maxY, position.y);
			gMetrics.maxParticleSpeed = PxMax(
				gMetrics.maxParticleSpeed, velocity.magnitude());
		}
	}

	if(gScene && gSceneCpuVolume->getScene() == gScene)
	{
		const PxBounds3 bounds = gSceneCpuVolume->getWorldBounds();
		if(bounds.isValid() && bounds.minimum.isFinite() &&
			bounds.maximum.isFinite())
			gMetrics.sceneBoundsFinite = 1;
		const bool sleeping = gSceneCpuVolume->isSleeping();
		if(!gMetrics.sceneSoftFirstSlept && sleeping)
		{
			gMetrics.sceneSoftFirstSlept = 1;
			gMetrics.sceneSoftFirstSleepFrame =
				gMetrics.completedFrames;
			gSceneCpuFirstSleepCentroidY =
				getSceneCpuVolumeCentroidY();
		}
		if(gMetrics.sceneSoftFirstSlept &&
			gMetrics.completedFrames >=
				gMetrics.sceneSoftFirstSleepFrame + 2 &&
			PxAbs(getSceneCpuVolumeCentroidY() -
				gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
			gMetrics.sceneMultiPrimaryStable = 1;
	}

	if(gSceneCpuSecondVolume->getScene() !=
			gSceneCpuSecondScene ||
		gSceneCpuSecondScene->getNbDeformableVolumes() != 1)
		return false;
	const PxReal secondCentroidY =
		getSceneCpuVolumeCentroidY(gSceneCpuSecondVolume);
	gMetrics.sceneSecondVolumeFinalCentroidY =
		secondCentroidY;
	gMetrics.sceneSecondVolumeMaxCentroidDrop = PxMax(
		gMetrics.sceneSecondVolumeMaxCentroidDrop,
		gSceneCpuSecondVolumeInitialCentroidY - secondCentroidY);
	gMetrics.maxCentroidDrop = PxMax(
		gMetrics.maxCentroidDrop,
		gSceneCpuSecondVolumeInitialCentroidY - secondCentroidY);
	const PxBounds3 secondBounds =
		gSceneCpuSecondVolume->getWorldBounds();
	if(secondBounds.isValid() &&
		secondBounds.minimum.isFinite() &&
		secondBounds.maximum.isFinite())
		gMetrics.sceneSecondVolumeBoundsFinite = 1;
	if(!gMetrics.scenePrimarySceneReleased &&
		gSceneCpuSecondVolumeInitialCentroidY -
			secondCentroidY > 0.02f)
		gMetrics.sceneMultiSecondaryUpdatedBeforeRelease = 1;
	if(gMetrics.scenePrimarySceneReleased &&
		gSceneCpuMultiSecondaryAtReleaseCentroidY -
			secondCentroidY > 0.02f)
		gMetrics.sceneMultiSecondaryUpdatedAfterRelease = 1;

	if(profileFrame)
	{
		gPerformance.profiledFrames++;
		gPerformance.sceneMs += sceneMs;
		gPerformance.stepSamplesMs.pushBack(
			PxReal(frameTimer.getElapsedSeconds() * 1000.0));
	}
	return true;
}

static bool updateSceneVolumeWorldPin()
{
	if(gHeadlessOptions.caseName != "scene-volume-world-pin" &&
		gHeadlessOptions.caseName !=
			"scene-volume-world-element-attachment")
		return true;
	if(!gScene || !gSceneCpuVolume ||
		!gSceneCpuWorldAttachment)
		return gMetrics.sceneWorldPinReleased != 0;

	const PxU32 churnFrame = PxMax<PxU32>(
		1, gHeadlessOptions.frames / 3);
	const PxU32 releaseFrame = PxMax<PxU32>(
		churnFrame + 1, 2 * gHeadlessOptions.frames / 3);
	if(gMetrics.completedFrames == churnFrame)
	{
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		gMetrics.sceneWorldPinActorReadded = 1;
	}
	if(gMetrics.completedFrames == releaseFrame)
	{
		gSceneCpuWorldAttachment->release();
		gSceneCpuWorldAttachment = NULL;
		gMetrics.sceneWorldPinReleased = 1;
	}
	return true;
}

static bool updateSceneVolumeRigidAttachment()
{
	const bool staticAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-static-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-static-element-attachment";
	const bool kinematicAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-element-attachment";
	const bool articulationAttachmentCase =
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-attachment" ||
		gHeadlessOptions.caseName ==
			"scene-volume-articulation-element-attachment";
	if(gHeadlessOptions.caseName !=
			"scene-volume-rigid-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-rigid-element-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-static-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-static-element-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-kinematic-element-attachment" &&
		gHeadlessOptions.caseName !=
			"scene-volume-articulation-element-attachment" &&
		!kinematicAttachmentCase &&
		!articulationAttachmentCase)
		return true;
	if(!gScene || !gSceneCpuVolume ||
		(!gSceneCpuAttachmentBody &&
		 !gSceneCpuStaticActor))
		return false;

	const PxU32 churnFrame = PxMax<PxU32>(
		1, gHeadlessOptions.frames / 3);
	const PxU32 releaseFrame = PxMax<PxU32>(
		churnFrame + 1, 2 * gHeadlessOptions.frames / 3);
	if(gMetrics.completedFrames == churnFrame)
	{
		if(!gSceneCpuRigidAttachment ||
			gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
	}
	if(gMetrics.completedFrames == releaseFrame)
	{
		if(!gSceneCpuRigidAttachment)
			return false;
		gSceneCpuRigidAttachment->release();
		gSceneCpuRigidAttachment = NULL;
		gMetrics.sceneRigidAttachmentReleased = 1;
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxU32 vertexCount =
			gSceneCpuVolume->getSimulationMesh()->
				getNbVertices();
		for(PxU32 i = 0; i < vertexCount; ++i)
			velocities[i] =
				PxVec4(1.0f, 0.0f, 0.0f, velocities[i].w);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eSIM_VELOCITY);
		if(!kinematicAttachmentCase &&
			!staticAttachmentCase &&
			!articulationAttachmentCase)
			gSceneCpuDynamicActor->setLinearVelocity(
				PxVec3(-1.0f, 0.0f, 0.0f));
	}
	else if((kinematicAttachmentCase || staticAttachmentCase) &&
		gMetrics.sceneSoftFirstSlept &&
		gSceneCpuRigidAttachment)
	{
		if(!gMetrics.sceneKinematicTargetIssued)
		{
			gSceneCpuKinematicAttachmentSoftBaseline =
				getSceneCpuAttachmentPoint();
			gSceneCpuKinematicAttachmentCommand =
				PxTransform(
					gSceneCpuRigidAttachmentInitialPosition);
			gMetrics.sceneKinematicTargetIssued = 1;
		}
		gSceneCpuKinematicAttachmentProgress = PxMin(
			// Complete the prescribed pose before the fixed two-thirds release
			// frame even when the logarithmic volume model needs a few additional
			// frames to reach the fixture's initial sleep threshold.
			gSceneCpuKinematicAttachmentProgress + 0.0125f,
			1.0f);
		gSceneCpuKinematicAttachmentCommand =
			PxTransform(
				gSceneCpuRigidAttachmentInitialPosition +
					PxVec3(
						gSceneCpuKinematicAttachmentProgress,
						0.0f, 0.0f),
				PxQuat(
					0.5f *
						gSceneCpuKinematicAttachmentProgress,
					PxVec3(0.0f, 0.0f, 1.0f)));
		if(staticAttachmentCase)
			gSceneCpuStaticActor->setGlobalPose(
				gSceneCpuKinematicAttachmentCommand);
		else
			gSceneCpuDynamicActor->setKinematicTarget(
				gSceneCpuKinematicAttachmentCommand);
	}
	return true;
}

static bool updateSceneVolumeSoftPairAttachment()
{
	if(gHeadlessOptions.caseName !=
		"scene-volume-volume-attachment")
		return true;
	if(!gScene || !gSceneCpuVolume ||
		!gSceneCpuSecondVolume)
		return false;

	const PxU32 churnFrame = PxMax<PxU32>(
		1, gHeadlessOptions.frames / 3);
	const PxU32 releaseFrame = PxMax<PxU32>(
		churnFrame + 1, 2 * gHeadlessOptions.frames / 3);
	if(gMetrics.completedFrames == churnFrame)
	{
		if(!gSceneCpuRigidAttachment ||
			gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
	}
	if(gMetrics.completedFrames == releaseFrame)
	{
		if(!gSceneCpuRigidAttachment)
			return false;
		gSceneCpuRigidAttachment->release();
		gSceneCpuRigidAttachment = NULL;
		gMetrics.sceneRigidAttachmentReleased = 1;
		if(!setSceneCpuVolumeVelocity(
				*gSceneCpuVolume,
				PxVec3(-0.5f, 0.0f, 0.0f)) ||
			!setSceneCpuVolumeVelocity(
				*gSceneCpuSecondVolume,
				PxVec3(0.5f, 0.0f, 0.0f)))
			return false;
	}
	return true;
}

static bool updateSceneVolumeElementFilter()
{
	const bool elementFilterCase =
		gHeadlessOptions.caseName ==
			"scene-volume-element-filter" ||
		gHeadlessOptions.caseName ==
			"scene-volume-partial-element-filter";
	if(!elementFilterCase)
		return true;
	if(!gScene || !gSceneCpuVolume)
		return false;

	const PxU32 churnFrame = PxMax<PxU32>(
		1, 2 * gHeadlessOptions.frames / 5);
	const PxU32 releaseFrame = PxMin<PxU32>(
		gHeadlessOptions.frames - 1, churnFrame + 12);
	if(gMetrics.completedFrames == churnFrame)
	{
		if(!gSceneCpuElementFilter ||
			gSceneCpuVolume->getScene() != gScene)
			return false;
		gScene->removeActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != NULL)
			return false;
		gScene->addActor(*gSceneCpuVolume);
		if(gSceneCpuVolume->getScene() != gScene)
			return false;
		PxVec4* positions =
			gSceneCpuVolume->getSimPositionInvMassBufferH();
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		PxVec4* collisionPositions =
			gSceneCpuVolume->getPositionInvMassBufferH();
		const PxU32 vertexCount =
			gSceneCpuVolume->getSimulationMesh()->
				getNbVertices();
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			positions[i].y -=
				SCENE_CPU_ELEMENT_FILTER_INITIAL_CLEARANCE +
				SCENE_CPU_ELEMENT_FILTER_TEST_PENETRATION;
			velocities[i] =
				PxVec4(0.0f, 0.0f, 0.0f, velocities[i].w);
		}
		PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
			*gSceneCpuVolume, positions, collisionPositions);
		gSceneCpuVolume->markDirty(
			PxDeformableVolumeDataFlag::eALL);
		gMetrics.sceneElementFilterActorReadded = 1;
	}
	if(gMetrics.completedFrames == releaseFrame)
	{
		if(!gSceneCpuElementFilter)
			return false;
		gSceneCpuElementFilter->release();
		gSceneCpuElementFilter = NULL;
		gMetrics.sceneElementFilterReleased = 1;
	}
	return true;
}

static bool updateSceneVolumeKinematicTargets()
{
	const bool fullTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-full-kinematic-target";
	const bool partialTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-partial-kinematic-target";
	const bool rigidTriangleSteadyContactCase =
		isSceneCpuVolumeRigidTriangleSteadyContactCase(
			gHeadlessOptions.caseName);
	if(!fullTargetCase && !partialTargetCase &&
		!rigidTriangleSteadyContactCase)
		return true;
	if(!gSceneCpuVolume ||
		gSceneCpuVolumeKinematicTargets.empty() ||
		gSceneCpuVolumeKinematicTargets.size() !=
			gSceneCpuVolumeKinematicInitial.size())
		return false;

	if(rigidTriangleSteadyContactCase)
	{
		// Eight frames give a stable, repeated downward/upward swept path.
		// The initial target is the start position; no direct buffer mutation
		// bypasses the public kinematic target API.
		const PxReal phase = PxReal(gMetrics.completedFrames % 8) *
			(PxTwoPi / 8.0f);
		const PxVec3 translation(
			0.0f, 1.1f * (PxCos(phase) - 1.0f), 0.0f);
		for(PxU32 i = 0;
			i < gSceneCpuVolumeKinematicTargets.size(); ++i)
		{
			gSceneCpuVolumeKinematicTargets[i] = PxVec4(
				gSceneCpuVolumeKinematicInitial[i] + translation,
				0.0f);
		}
		gMetrics.sceneVolumeTargetMutated = 1;
		return true;
	}

	const PxU32 driveStart = PxMax<PxU32>(
		1, gHeadlessOptions.frames / 5);
	const PxU32 driveEnd = PxMax<PxU32>(
		driveStart + 1, gHeadlessOptions.frames / 2);
	if(gMetrics.completedFrames >= driveStart)
	{
		const PxReal progress = PxClamp(
			PxReal(gMetrics.completedFrames - driveStart) /
				PxReal(driveEnd - driveStart),
			0.0f, 1.0f);
		const PxVec3 translation(
			0.5f * progress, 0.25f * progress, 0.0f);
		for(PxU32 i = 0;
			i < gSceneCpuVolumeKinematicTargets.size(); ++i)
		{
			if(partialTargetCase &&
				i == gSceneCpuVolumePartialProbe)
				continue;
			gSceneCpuVolumeKinematicTargets[i] =
				PxVec4(
					gSceneCpuVolumeKinematicInitial[i] +
						translation,
					0.0f);
		}
		gMetrics.sceneVolumeTargetMutated = 1;
	}

	const PxU32 activationFrame = PxMax<PxU32>(
		driveEnd + 1, 2 * gHeadlessOptions.frames / 3);
	if(partialTargetCase &&
		!gMetrics.sceneVolumePartialActivated &&
		gMetrics.completedFrames >= activationFrame)
	{
		if(gSceneCpuVolumePartialProbe >=
				gSceneCpuVolumeKinematicTargets.size())
			return false;
		const PxVec4* positions =
			gSceneCpuVolume->getSimPositionInvMassBufferH();
		if(!positions)
			return false;
		const PxU32 probe = gSceneCpuVolumePartialProbe;
		const PxVec3 oldDecoy =
			gSceneCpuVolumeKinematicTargets[probe].getXYZ();
		const PxVec3 current = positions[probe].getXYZ();
		gMetrics.sceneVolumePartialInactiveDecoyDistance =
			(oldDecoy - current).magnitude();
		if(gMetrics.
				sceneVolumePartialInactiveDecoyDistance > 2.0f)
			gMetrics.sceneVolumePartialInactiveIgnored = 1;
		gSceneCpuVolumePartialActivationStart = current;
		gSceneCpuVolumeKinematicTargets[probe] =
			PxVec4(current, 0.0f);
		gMetrics.sceneVolumePartialActivated = 1;
	}
	if(partialTargetCase &&
		gMetrics.sceneVolumePartialActivated)
	{
		const PxU32 probe = gSceneCpuVolumePartialProbe;
		if(probe >= gSceneCpuVolumeKinematicTargets.size())
			return false;
		const PxU32 activationDuration = PxMax<PxU32>(
			1, gHeadlessOptions.frames / 10);
		const PxReal activationProgress = PxClamp(
			PxReal(gMetrics.completedFrames - activationFrame) /
				PxReal(activationDuration),
			0.0f, 1.0f);
		gSceneCpuVolumeKinematicTargets[probe] =
			PxVec4(
				gSceneCpuVolumePartialActivationStart +
					PxVec3(
						// This is a partial-kinematic activation target, not
						// a particle-primal positive-J fixture: its active
						// vertex is static to the local solve.  Keep the
						// displacement inside the target-reach envelope; the
						// dedicated unit-tetra limiter fixture owns positive-J
						// kernel coverage.
						0.1f * activationProgress,
						0.0f, 0.0f),
				0.0f);
	}
	return true;
}

static bool sampleSceneVolumeKinematicTargets()
{
	const bool fullTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-full-kinematic-target";
	const bool partialTargetCase =
		gHeadlessOptions.caseName ==
			"scene-volume-partial-kinematic-target";
	const bool rigidTriangleSteadyContactCase =
		isSceneCpuVolumeRigidTriangleSteadyContactCase(
			gHeadlessOptions.caseName);
	if(!fullTargetCase && !partialTargetCase &&
		!rigidTriangleSteadyContactCase)
		return true;
	if(rigidTriangleSteadyContactCase)
		return true;
	if(!gSceneCpuVolume ||
		gSceneCpuVolumeKinematicTargets.empty() ||
		gSceneCpuVolumeKinematicTargets.size() !=
			gSceneCpuVolumeKinematicInitial.size())
		return false;
	const PxVec4* positions =
		gSceneCpuVolume->getSimPositionInvMassBufferH();
	if(!positions)
		return false;

	PxVec3 initialCentroid(0.0f);
	PxVec3 currentCentroid(0.0f);
	PxReal maxActiveError = 0.0f;
	for(PxU32 i = 0;
		i < gSceneCpuVolumeKinematicTargets.size(); ++i)
	{
		initialCentroid += gSceneCpuVolumeKinematicInitial[i];
		currentCentroid += positions[i].getXYZ();
		const bool active =
			fullTargetCase ||
			gSceneCpuVolumeKinematicTargets[i].w == 0.0f;
		if(active)
			maxActiveError = PxMax(
				maxActiveError,
				(positions[i].getXYZ() -
					gSceneCpuVolumeKinematicTargets[i].
						getXYZ()).magnitude());
	}
	const PxReal invCount = 1.0f /
		PxReal(gSceneCpuVolumeKinematicTargets.size());
	initialCentroid *= invCount;
	currentCentroid *= invCount;
	gMetrics.sceneVolumeTargetMaxDisplacement = PxMax(
		gMetrics.sceneVolumeTargetMaxDisplacement,
		(currentCentroid - initialCentroid).magnitude());
	gMetrics.sceneVolumeTargetFinalMaxError =
		maxActiveError;

	if(gMetrics.sceneVolumeTargetMutated &&
		!gSceneCpuVolume->isSleeping())
		gMetrics.sceneVolumeTargetWoke = 1;
	const PxU32 driveEnd = PxMax<PxU32>(
		PxMax<PxU32>(1, gHeadlessOptions.frames / 5) + 1,
		gHeadlessOptions.frames / 2);
	if(gMetrics.completedFrames > driveEnd &&
		maxActiveError <= 5.0e-3f)
		gMetrics.sceneVolumeTargetReached = 1;
	if(partialTargetCase &&
		gMetrics.sceneVolumePartialActivated &&
		gSceneCpuVolumePartialProbe <
			gSceneCpuVolumeKinematicTargets.size())
	{
		const PxU32 probe = gSceneCpuVolumePartialProbe;
		const PxReal probeError =
			(positions[probe].getXYZ() -
				gSceneCpuVolumeKinematicTargets[probe].
					getXYZ()).magnitude();
		if(probeError <= 5.0e-3f)
			gMetrics.sceneVolumePartialActivatedReached = 1;
	}
	return true;
}

static bool updateSceneKinematicBox()
{
	if(!isSceneCpuVolumeKinematicRigidCase(
		gHeadlessOptions.caseName))
		return true;
	if(!gSceneCpuVolume || !gSceneCpuDynamicActor ||
		gSceneCpuDynamicActor->getScene() != gScene)
		return false;
	if(!gMetrics.sceneSoftFirstSlept)
		return true;

	if(!gMetrics.sceneKinematicTargetIssued)
	{
		gSceneCpuKinematicSoftBaselineY =
			getSceneCpuVolumeCentroidY();
		gMetrics.sceneKinematicTargetIssued = 1;
	}

	const PxReal kinematicStep =
		gHeadlessOptions.caseName ==
			"scene-volume-kinematic-convex"
		? 0.0025f : 0.005f;
	const PxReal nextY = PxMin(
		gSceneCpuKinematicCommandY + kinematicStep, 4.10f);
	if(nextY > gSceneCpuKinematicCommandY)
	{
		gSceneCpuKinematicCommandY = nextY;
		gSceneCpuDynamicActor->setKinematicTarget(
			PxTransform(
				gHeadlessOptions.caseName ==
					"scene-volume-kinematic-heightfield"
					? PxVec3(-4.0f, nextY, -4.0f)
					: PxVec3(0.0f, nextY, 0.0f)));
	}
	return true;
}

static bool updateSceneCpuOgcSandwichPressureDrive()
{
	if(gHeadlessOptions.caseName != "scene-volume-ogc-sandwich" ||
		gMetrics.completedFrames >=
			SCENE_CPU_OGC_SANDWICH_PRESSURE_DRIVE_FRAMES)
		return true;
	if(!gSceneCpuVolume || !gSceneCpuSecondVolume ||
		gSceneCpuVolume->getScene() != gScene ||
		gSceneCpuSecondVolume->getScene() != gScene)
		return false;
	// This is deliberately a test-owned load, not a kinematic actor or a
	// speculative/swept CCD shortcut.  The public soft velocity buffers are
	// updated before the one authored scene step, while the rigid is left free
	// to receive the unified OGC response.
	return setSceneCpuVolumeVelocity(
			*gSceneCpuVolume,
			SCENE_CPU_OGC_SANDWICH_UPPER_JAW_VELOCITY) &&
		setSceneCpuVolumeVelocity(
			*gSceneCpuSecondVolume,
			SCENE_CPU_OGC_SANDWICH_LOWER_JAW_VELOCITY);
}

static bool stepPhysicsInternal(PxReal dt)
{
	if(gHeadlessOptions.caseName ==
		"scene-volume-multi-scene-isolation")
		return stepSceneMultiSceneIsolation(dt);

	const bool profileFrame =
		gMetrics.completedFrames >= gProfileWarmupFrames;
	PxTime frameTimer;
	if(isSceneCpuVolumeCase(gHeadlessOptions.caseName))
	{
		if(!updateSceneStaticChurn() ||
			!updateSceneDynamicChurn() ||
			!updateSceneMultiDynamicGate() ||
			!updateSceneSoftSleepWake() ||
			!updateSceneSoftRigidWake() ||
			!updateSceneSoftActorChurn() ||
			!updateSceneSoftBufferMutation() ||
			!updateSceneSoftSoftWake() ||
			!updateSceneVolumeWorldPin() ||
			!updateSceneVolumeRigidAttachment() ||
			!updateSceneVolumeSoftPairAttachment() ||
			!updateSceneVolumeElementFilter() ||
			!updateSceneVolumeKinematicTargets() ||
			!updateSceneKinematicBox() ||
			!updateSceneCpuOgcSandwichPressureDrive())
			return false;
		PxTime sceneTimer;
		// Every scene case, including the visual OGC showcase, advances exactly
		// once at the authored timestep.  Current-pose OGC must own its
		// penetration response; this path intentionally has no visual-only
		// microstepping, swept query, or CCD fallback.
		gScene->simulate(dt);
		if(!gScene->fetchResults(true))
		{
			gMetrics.fetchFailures++;
			return false;
		}
		const PxF64 sceneMs =
			sceneTimer.getElapsedSeconds() * 1000.0;
		gMetrics.completedFrames++;
		if(gHeadlessOptions.caseName == "scene-volume-visual-showcase")
		{
			if(!sampleSceneCpuVisualRotationMetrics(
					gSceneCpuVisualRotationMonitor) ||
				!sampleSceneCpuVisualRotationMetrics(
					gSceneCpuVisualPrimaryCubeRotationMonitor) ||
				!sampleSceneCpuVisualRotationMetrics(
					gSceneCpuVisualSphereRotationMonitor) ||
				!sampleSceneCpuVisualInteractionMetrics() ||
				!sampleSceneCpuVolumeHealth())
				return false;
		}
		else if(gHeadlessOptions.caseName == "scene-volume-ogc-sandwich")
		{
			if(!sampleSceneCpuVolumeHealth() ||
				!sampleSceneCpuOgcSandwichMetrics())
				return false;
		}
		else if(gHeadlessOptions.caseName ==
			"scene-volume-sphere-long-roll" ||
			gHeadlessOptions.caseName ==
			"scene-volume-sphere-soft-soft-glancing")
		{
			if(!sampleSceneCpuVisualRotationMetrics(
					gSceneCpuVisualSphereRotationMonitor) ||
				!sampleSceneCpuSphereLongRollContactPhaseMetrics() ||
				!sampleSceneCpuVolumeHealth())
				return false;
		}
		else if(gHeadlessOptions.caseName ==
			"scene-volume-soft-soft-torque")
		{
			if(!sampleSceneCpuSoftSoftTorqueMetrics() ||
				!sampleSceneCpuVolumeHealth())
				return false;
		}
		else if(gHeadlessOptions.caseName ==
			"scene-volume-ground-embedded-tet-probe")
		{
			if(!sampleSceneCpuGroundEmbeddedTetProbeMetrics() ||
				!sampleSceneCpuVolumeHealth())
				return false;
		}
		else if(isSceneCpuVolumeTaskGraphWriteBackCase(
			gHeadlessOptions.caseName) &&
			!sampleSceneCpuVolumeHealth())
			return false;
		if(!sampleSceneVolumeKinematicTargets())
			return false;
		if(!updateVolumeSkinning())
			return false;

		PxVec4* positions =
			gSceneCpuVolume->getSimPositionInvMassBufferH();
		PxVec4* velocities =
			gSceneCpuVolume->getSimVelocityBufferH();
		const PxTetrahedronMesh* sceneSimulationMesh =
			gSceneCpuVolume->getSimulationMesh();
		const PxU32 vertexCount =
			sceneSimulationMesh->getNbVertices();
		PxU32 nearGroundParticles = 0;
		PxU32 nearRigidParticles = 0;
		PxReal frameMinY = FLT_MAX;
		PxReal frameCollisionMinY = FLT_MAX;
		PxReal frameDynamicSurfaceSeparation = FLT_MAX;
		const bool multiDynamicBoxCase =
			gHeadlessOptions.caseName ==
				"scene-volume-multi-dynamic-box";
		const bool multiSoftIslandCase =
			gHeadlessOptions.caseName ==
				"scene-volume-multi-soft-islands";
		const bool mixedSleepIslandCase =
			gHeadlessOptions.caseName ==
				"scene-volume-mixed-sleep-islands";
		const bool softChurnCase =
			gHeadlessOptions.caseName ==
				"scene-volume-soft-churn";
		const bool bufferMutationCase =
			gHeadlessOptions.caseName ==
				"scene-volume-buffer-mutation";
		const bool partialElementFilterCase =
			gHeadlessOptions.caseName ==
				"scene-volume-partial-element-filter";
		const bool groundEmbeddedTetProbeCase =
			gHeadlessOptions.caseName ==
				"scene-volume-ground-embedded-tet-probe";
		const bool elementFilterCase =
			gHeadlessOptions.caseName ==
				"scene-volume-element-filter" ||
			partialElementFilterCase;
		const bool rigidAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-rigid-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-rigid-element-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-element-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-element-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-articulation-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-articulation-element-attachment";
		const bool staticAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-static-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-element-attachment";
		const bool kinematicAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-element-attachment";
		const bool articulationAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-articulation-attachment" ||
			gHeadlessOptions.caseName ==
				"scene-volume-articulation-element-attachment";
		const bool softSoftWakeCase =
			gHeadlessOptions.caseName ==
				"scene-volume-soft-soft-wake";
		const bool softPairAttachmentCase =
			gHeadlessOptions.caseName ==
				"scene-volume-volume-attachment";
		const bool motionControlsCase =
			gHeadlessOptions.caseName ==
				"scene-volume-motion-controls";
		const bool maxDepenetrationVelocityCase =
			gHeadlessOptions.caseName ==
				"scene-volume-max-depenetration-velocity";
		const bool triangleSurfaceSweptCcdCase =
			isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool triangleSurfaceReverseSweptCcdCase =
			isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool heightFieldSweptCcdCase =
			isSceneCpuVolumeHeightFieldSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalTriangleSurfaceSweptCcdCase =
			isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool sphereReverseSweptCcdCase =
			isSceneCpuVolumeSphereReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool deformingReverseSweptCcdCase =
			isSceneCpuVolumeDeformingReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool capsuleReverseSweptCcdCase =
			isSceneCpuVolumeCapsuleReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool convexReverseSweptCcdCase =
			isSceneCpuVolumeConvexReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalCapsuleReverseSweptCcdCase =
			isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalConvexReverseSweptCcdCase =
			isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
				gHeadlessOptions.caseName);
		const bool rotationalFiniteReverseSweptCcdCase =
			rotationalCapsuleReverseSweptCcdCase ||
			rotationalConvexReverseSweptCcdCase;
		const bool staticSphereReverseSweptCcdCase =
			deformingReverseSweptCcdCase ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-sphere-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-static-convex-reverse-swept-ccd";
		const bool kinematicSphereReverseSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-convex-reverse-swept-ccd";
		const bool dynamicSphereReverseSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-sphere-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-rotating-convex-reverse-swept-ccd" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-convex-reverse-swept-ccd";
		const bool speculativeCcdCase =
			isSceneCpuVolumeSpeculativeCcdCase(
				gHeadlessOptions.caseName) ||
			sphereReverseSweptCcdCase;
		const bool planeSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-plane-speculative-ccd";
		const bool sphereSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-sphere-speculative-ccd";
		const bool capsuleSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-capsule-speculative-ccd";
		const bool convexSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-convex-speculative-ccd";
		const bool finiteSmoothSpeculativeCcdCase =
			sphereSpeculativeCcdCase || capsuleSpeculativeCcdCase ||
			convexSpeculativeCcdCase;
		const bool rotatingKinematicCapsuleSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-rotating-kinematic-capsule-speculative-ccd";
		const bool rotatingKinematicConvexSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-rotating-kinematic-convex-speculative-ccd";
		const bool movingKinematicCapsuleSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-moving-kinematic-capsule-speculative-ccd" ||
			rotatingKinematicCapsuleSpeculativeCcdCase;
		const bool movingKinematicConvexSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-moving-kinematic-convex-speculative-ccd" ||
			rotatingKinematicConvexSpeculativeCcdCase;
		const bool movingKinematicFiniteSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-moving-kinematic-sphere-speculative-ccd" ||
			movingKinematicCapsuleSpeculativeCcdCase ||
			movingKinematicConvexSpeculativeCcdCase;
		const bool dynamicRotatingCapsuleSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-rotating-capsule-relative-swept-ccd";
		const bool dynamicRotatingConvexSpeculativeCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-rotating-convex-relative-swept-ccd";
		const bool dynamicCapsuleRelativeSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule-relative-swept-ccd" ||
			dynamicRotatingCapsuleSpeculativeCcdCase;
		const bool dynamicConvexRelativeSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-convex-relative-swept-ccd" ||
			dynamicRotatingConvexSpeculativeCcdCase;
		const bool dynamicFiniteRelativeSweptCcdCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-sphere-relative-swept-ccd" ||
			dynamicCapsuleRelativeSweptCcdCase ||
			dynamicConvexRelativeSweptCcdCase;
		const bool sphereReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-sphere-reverse-feature";
		const bool capsuleReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-capsule-reverse-feature";
		const bool convexReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-convex-reverse-feature";
		const bool triangleMeshReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-triangle-mesh-reverse-feature";
		const bool heightFieldReverseFeatureCase =
			gHeadlessOptions.caseName ==
				"scene-volume-heightfield-reverse-feature";
		const bool smoothReverseFeatureCase =
			sphereReverseFeatureCase ||
			capsuleReverseFeatureCase ||
			convexReverseFeatureCase ||
			triangleMeshReverseFeatureCase ||
			heightFieldReverseFeatureCase;
		const bool twoSoftVolumeCase =
			multiSoftIslandCase || mixedSleepIslandCase ||
			softChurnCase || softSoftWakeCase ||
			softPairAttachmentCase || motionControlsCase ||
			maxDepenetrationVelocityCase ||
			speculativeCcdCase || smoothReverseFeatureCase;
		const bool softSleepWakeCase =
			gHeadlessOptions.caseName ==
				"scene-volume-sleep-wake";
		const bool softRigidWakeCase =
			gHeadlessOptions.caseName ==
				"scene-volume-rigid-wake";
		const bool kinematicBoxCase =
			isSceneCpuVolumeKinematicRigidCase(
				gHeadlessOptions.caseName);
		const bool kinematicSphereCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-sphere";
		const bool kinematicCapsuleCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-capsule";
		const bool kinematicConvexCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-convex";
		const bool kinematicTriangleMeshCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-triangle-mesh";
		const bool kinematicHeightFieldCase =
			gHeadlessOptions.caseName ==
				"scene-volume-kinematic-heightfield";
		const bool twoDynamicActorsCase =
			multiDynamicBoxCase || multiSoftIslandCase ||
			movingKinematicFiniteSpeculativeCcdCase ||
			dynamicFiniteRelativeSweptCcdCase ||
			kinematicSphereReverseSweptCcdCase ||
			dynamicSphereReverseSweptCcdCase ||
			triangleSurfaceSweptCcdCase;
		const bool dynamicBoxCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-box" ||
			gHeadlessOptions.caseName ==
				"scene-volume-true-boundary-dynamic-box" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-sphere" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-convex" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-churn" ||
			twoDynamicActorsCase;
		const bool dynamicSphereCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-sphere" ||
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule";
		const bool dynamicCapsuleCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-capsule";
		const bool dynamicConvexCase =
			gHeadlessOptions.caseName ==
				"scene-volume-dynamic-convex";
		const bool dynamicActorInScene =
			(dynamicBoxCase || softRigidWakeCase) &&
			gSceneCpuDynamicActor &&
			gSceneCpuDynamicActor->getScene() == gScene;
		const bool secondDynamicActorInScene =
			twoDynamicActorsCase && gSceneCpuSecondDynamicActor &&
			gSceneCpuSecondDynamicActor->getScene() == gScene;
		if(gHeadlessOptions.caseName == "scene-volume-world-pin" ||
			gHeadlessOptions.caseName ==
				"scene-volume-world-element-attachment")
		{
			const PxReal displacement =
				(getSceneCpuAttachmentPoint() -
					gSceneCpuWorldPinTarget).magnitude();
			if(!gMetrics.sceneWorldPinReleased)
			{
				gMetrics.sceneWorldPinMaxDrift = PxMax(
					gMetrics.sceneWorldPinMaxDrift,
					displacement);
				gMetrics.sceneWorldPinHeld =
					gMetrics.sceneWorldPinMaxDrift <= 1.0e-4f
						? 1u : 0u;
			}
			else
			{
				gMetrics.sceneWorldPinReleasedMaxDisplacement =
					PxMax(
						gMetrics.
							sceneWorldPinReleasedMaxDisplacement,
						displacement);
				if(gMetrics.
					sceneWorldPinReleasedMaxDisplacement >
					1.0e-3f)
					gMetrics.sceneWorldPinMovedAfterRelease = 1;
			}
		}
		if(softPairAttachmentCase && gSceneCpuSecondVolume)
		{
			const PxReal separation =
				(getSceneCpuAttachmentPoint() -
					getSceneCpuAttachmentPoint(
						gSceneCpuSecondVolume)).magnitude();
			const PxReal targetDisplacement = PxAbs(
				getSceneCpuVolumeCentroidY(
					gSceneCpuSecondVolume) -
				gSceneCpuSecondVolumeInitialCentroidY);
			const PxVec4* targetVelocities =
				gSceneCpuSecondVolume->
					getSimVelocityBufferH();
			const PxU32 targetVertexCount =
				gSceneCpuSecondVolume->getSimulationMesh()->
					getNbVertices();
			PxReal targetMaxSpeed = 0.0f;
			if(!targetVelocities)
				return false;
			for(PxU32 i = 0; i < targetVertexCount; ++i)
				targetMaxSpeed = PxMax(
					targetMaxSpeed,
					targetVelocities[i].getXYZ().magnitude());
			if(!PxIsFinite(separation) ||
				!PxIsFinite(targetDisplacement) ||
				!PxIsFinite(targetMaxSpeed))
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.sceneRigidAttachmentMaxRigidSpeed = PxMax(
					gMetrics.sceneRigidAttachmentMaxRigidSpeed,
					targetMaxSpeed);
				gMetrics.
					sceneRigidAttachmentMaxRigidDisplacement = PxMax(
						gMetrics.
							sceneRigidAttachmentMaxRigidDisplacement,
						targetDisplacement);
				if(targetDisplacement > 0.02f)
					gMetrics.sceneRigidAttachmentRigidMoved = 1;
				if(!gSceneCpuSecondVolume->isSleeping())
					gMetrics.sceneRigidAttachmentRigidWoke = 1;
				const PxU32 churnFrame = PxMax<PxU32>(
					1, gHeadlessOptions.frames / 3);
				if(!gMetrics.sceneRigidAttachmentReleased &&
					gMetrics.completedFrames >= churnFrame)
				{
					gMetrics.sceneRigidAttachmentMaxDrift = PxMax(
						gMetrics.sceneRigidAttachmentMaxDrift,
						separation);
					if(gMetrics.completedFrames > churnFrame &&
						separation < 0.05f)
						gMetrics.
							sceneRigidAttachmentHeldAcrossReadd = 1;
				}
				else if(gMetrics.sceneRigidAttachmentReleased)
				{
					gMetrics.
						sceneRigidAttachmentReleasedSeparation = PxMax(
							gMetrics.
								sceneRigidAttachmentReleasedSeparation,
							separation);
					if(separation > 0.2f)
						gMetrics.
							sceneRigidAttachmentSeparatedAfterRelease =
								1;
				}
			}
		}
		if(motionControlsCase && gSceneCpuSecondVolume)
		{
			const PxVec3 controlCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxReal controlSpeed =
				getSceneCpuVolumeMaxSpeed(gSceneCpuVolume);
			const PxReal settlingSpeed =
				getSceneCpuVolumeMaxSpeed(gSceneCpuSecondVolume);
			if(!controlCentroid.isFinite() ||
				!PxIsFinite(controlSpeed) ||
				!PxIsFinite(settlingSpeed))
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.motionControlFinalSpeed = controlSpeed;
				gMetrics.motionSettlingFinalSpeed = settlingSpeed;
				if(gMetrics.completedFrames == 1)
				{
					gMetrics.
						motionMaxVelocityFirstStepDisplacement =
						(controlCentroid -
							gSceneCpuMotionInitialCentroid).
								magnitude();
					gMetrics.motionMaxVelocityFirstStepSpeed =
						controlSpeed;
					gMetrics.motionMaxVelocityBounded =
						gMetrics.
							motionMaxVelocityFirstStepDisplacement <=
								dt * 1.01f &&
						// maxLinearVelocity owns free preintegration;
						// Position-AL may add a small post-solve
						// constraint velocity.
						controlSpeed <= 1.02f ? 1u : 0u;
					if(settlingSpeed <= 0.07f)
						gMetrics.motionSettlingApplied = 1;
					if(!setSceneCpuVolumeVelocity(
							*gSceneCpuVolume,
							PxVec3(0.08f, 0.0f, 0.0f)))
						return false;
					gSceneCpuVolume->setMaxLinearVelocity(
						PX_MAX_F32);
					gSceneCpuVolume->setSettlingThreshold(0.1f);
					gSceneCpuVolume->setSettlingDamping(0.0f);
					gSceneCpuVolume->setSleepThreshold(0.05f);
					gSceneCpuVolume->setWakeCounter(0.2f);
				}
				else if(gMetrics.completedFrames > 1)
				{
					if(gSceneCpuVolume->isSleeping())
						gMetrics.motionControlStayedAwake = 0;
				}
				if(gSceneCpuSecondVolume->isSleeping())
					gMetrics.motionSettlingSlept = 1;
			}
		}
		if(maxDepenetrationVelocityCase &&
			gSceneCpuSecondVolume)
		{
			const PxReal limitedMinY =
				getSceneCpuVolumeMinY(gSceneCpuVolume);
			const PxReal controlMinY =
				getSceneCpuVolumeMinY(gSceneCpuSecondVolume);
			const PxReal limitedRise =
				limitedMinY -
				gSceneCpuDepenetrationInitialMinY;
			const PxReal controlRise =
				controlMinY -
				gSceneCpuDepenetrationControlInitialMinY;
			const PxReal limitedSpeed =
				getSceneCpuVolumeMaxSpeed(gSceneCpuVolume);
			if(!PxIsFinite(limitedRise) ||
				!PxIsFinite(controlRise) ||
				!PxIsFinite(limitedSpeed))
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.depenetrationLimitedMaxSpeed = PxMax(
					gMetrics.depenetrationLimitedMaxSpeed,
					limitedSpeed);
				if(gMetrics.completedFrames == 1)
				{
					gMetrics.
						depenetrationLimitedFirstStepRise =
							limitedRise;
					gMetrics.
						depenetrationControlFirstStepRise =
							controlRise;
					gMetrics.depenetrationFirstStepBounded =
						limitedRise >= -1.0e-6f &&
						limitedRise <= 0.12f * dt * 1.25f &&
						limitedSpeed <= 0.25f
							? 1u : 0u;
					gMetrics.depenetrationControlSeparated =
						controlRise >
							limitedRise + 5.0e-3f
								? 1u : 0u;
				}
				gMetrics.depenetrationLimitedFinalRise =
					limitedRise;
				if(limitedRise >
					gMetrics.
						depenetrationLimitedFirstStepRise +
							4.0e-3f)
					gMetrics.depenetrationGradualRecovery = 1;
			}
		}
		if(smoothReverseFeatureCase &&
			gSceneCpuSecondVolume &&
			gSceneCpuStaticActor &&
			gMetrics.completedFrames == 1)
		{
			const PxVec3 rigidCenter =
				heightFieldReverseFeatureCase
					? PxVec3(-2.05f, 0.0f, -0.05f)
					: PxVec3(-1.75f, 0.0f, 0.25f);
			const PxReal rigidRadius = 0.3f;
			const PxReal capsuleHalfHeight =
				capsuleReverseFeatureCase ? 0.15f : 0.0f;
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuSecondVolume);
			PxReal faceSeparation = PX_MAX_F32;
			PxReal minimumVertexSeparation = PX_MAX_F32;
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite() &&
				getSceneCpuVolumeSweptGeometrySampler().
					measureSmoothReverseSeparations(
					gSceneCpuVolume, rigidCenter, rigidRadius,
					capsuleHalfHeight,
					convexReverseFeatureCase,
					triangleMeshReverseFeatureCase,
					heightFieldReverseFeatureCase,
					faceSeparation,
					minimumVertexSeparation);
			if(!finite)
				gSphereReverseFeatureMetrics.nonFiniteSamples++;
			else
				updateReverseFeatureMetrics(
					gSphereReverseFeatureMetrics, positiveCentroid,
					negativeCentroid,
					gSceneCpuSphereReversePositiveInitial,
					gSceneCpuSphereReverseNegativeInitial,
					faceSeparation, minimumVertexSeparation);

			if(gSceneCpuVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuVolume);
				gMetrics.sceneActorRemoved =
					gSceneCpuVolume->getScene() == NULL ? 1u : 0u;
			}
			if(gSceneCpuSecondVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuSecondVolume);
				gMetrics.sceneSecondVolumeActorRemoved =
					gSceneCpuSecondVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuStaticActor->getScene() == gScene)
				gScene->removeActor(*gSceneCpuStaticActor);
		}
		else if(triangleSurfaceSweptCcdCase &&
			gSceneCpuSecondVolume &&
			gSceneCpuDynamicActor &&
			gSceneCpuSecondDynamicActor &&
			gMetrics.completedFrames == 1)
		{
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(
					gSceneCpuSecondVolume);
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite();
			if(!finite)
			{
				gSphereReverseSweptMetrics.nonFiniteSamples++;
				if(rotationalTriangleSurfaceSweptCcdCase)
					gCapsuleRotationalSweepMetrics.
						nonFiniteSamples++;
			}
			else
			{
				gSphereReverseSweptMetrics.
					positiveDisplacement =
						(positiveCentroid -
							gSceneCpuMovingSpherePositiveInitial).
								magnitude();
				gSphereReverseSweptMetrics.
					negativeDisplacement =
						(negativeCentroid -
							gSceneCpuMovingSphereNegativeInitial).
								magnitude();
				gSphereReverseSweptMetrics.positiveDrop =
					gSceneCpuMovingSpherePositiveInitial.y -
						positiveCentroid.y;
				gSphereReverseSweptMetrics.negativeDrop =
					gSceneCpuMovingSphereNegativeInitial.y -
						negativeCentroid.y;
				gSphereReverseSweptMetrics.positiveRigidDrop =
					0.0f;
				gSphereReverseSweptMetrics.negativeRigidDrop =
					0.0f;
				gSphereReverseSweptMetrics.faceSeparation =
					0.0f;
				gSphereReverseSweptMetrics.
					minimumVertexSweepSeparation =
						PX_MAX_F32;
				if(rotationalTriangleSurfaceSweptCcdCase)
				{
					const PxQuat startRotation(
						0.5235987756f,
						PxVec3(0.0f, 0.0f, 1.0f));
					const PxQuat endRotation(
						2.6179938780f,
						PxVec3(0.0f, 0.0f, 1.0f));
					const PxTransform positivePose =
						gSceneCpuDynamicActor->getGlobalPose();
					const PxTransform negativePose =
						gSceneCpuSecondDynamicActor->
							getGlobalPose();
					const PxTransform startPose(
						positivePose.p, startRotation);
					const PxTransform endPose(
						positivePose.p, endRotation);
					const bool geometryFinite =
						getSceneCpuVolumeSweptGeometrySampler().
							measureRotationalTriangleSurfaceSweepSeparations(
							startPose, endPose,
							heightFieldSweptCcdCase,
							triangleSurfaceReverseSweptCcdCase,
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation,
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation,
							gSphereReverseSweptMetrics.
								minimumVertexSweepSeparation);
					auto getAngularTravel = [&](
						const PxQuat& end)
					{
						const PxReal alignment = PxClamp(
							PxAbs(
								startRotation.dot(
									end.getNormalized())),
							0.0f, 1.0f);
						return 2.0f * PxAcos(alignment);
					};
					gCapsuleRotationalSweepMetrics.
						positiveAngularTravel =
							getAngularTravel(positivePose.q);
					gCapsuleRotationalSweepMetrics.
						negativeAngularTravel =
							getAngularTravel(negativePose.q);
					if(!geometryFinite)
						gCapsuleRotationalSweepMetrics.
							nonFiniteSamples++;
					gCapsuleRotationalSweepMetrics.
						sweepIsolated =
							geometryFinite &&
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation > 0.10f &&
							(triangleSurfaceReverseSweptCcdCase
								? gCapsuleRotationalSweepMetrics.
										midSweepMinSeparation <
									-0.05f
								: gCapsuleRotationalSweepMetrics.
										midSweepMinSeparation <
									0.01f)
								? 1u : 0u;
				}
				else if(triangleSurfaceReverseSweptCcdCase)
				{
					const PxReal startY = -1.1f;
					const PxVec3 rigidCenter =
						heightFieldSweptCcdCase
							? PxVec3(
								-1.93f, startY, 0.03f)
							: PxVec3(
								-1.63f, startY, 0.33f);
					const PxVec3 rigidTipStart =
						rigidCenter +
							(heightFieldSweptCcdCase
								? PxVec3(0.3f, 0.3f, 0.3f)
								: PxVec3(0.0f, 0.3f, 0.0f));
					const PxVec3 rigidTipSegment(
						0.0f, 2.2f, 0.0f);
					const PxReal denominator =
						rigidTipSegment.magnitudeSquared();
					for(PxU32 vertexIndex = 0;
						vertexIndex <
							gSceneCpuSphereReverseSweptInitialPositions.
								size();
						++vertexIndex)
					{
						const PxVec3 softVertex =
							gSceneCpuSphereReverseSweptInitialPositions[
								vertexIndex];
						const PxReal time = PxClamp(
							(softVertex - rigidTipStart).
								dot(rigidTipSegment) /
								denominator,
							0.0f, 1.0f);
						gSphereReverseSweptMetrics.
							minimumVertexSweepSeparation =
								PxMin(
									gSphereReverseSweptMetrics.
										minimumVertexSweepSeparation,
									(softVertex -
										(rigidTipStart +
											rigidTipSegment * time)).
										magnitude());
					}
				}
				gSphereReverseSweptMetrics.vertexSweepExcluded =
					!triangleSurfaceReverseSweptCcdCase ||
					gSphereReverseSweptMetrics.
							minimumVertexSweepSeparation > 0.05f
						? 1u : 0u;
				const PxReal kinematicResponseThreshold =
					rotationalTriangleSurfaceSweptCcdCase
						? 2.0e-3f
						: triangleSurfaceReverseSweptCcdCase
							? 5.0e-3f : 0.02f;
				gSphereReverseSweptMetrics.responseObserved =
					gSphereReverseSweptMetrics.positiveDisplacement >
							kinematicResponseThreshold &&
							gSphereReverseSweptMetrics.
								negativeDisplacement < 1.0e-2f
							? 1u : 0u;
				gSphereReverseSweptMetrics.negativeControlPassed =
					gSphereReverseSweptMetrics.
						negativeDisplacement < 1.0e-2f
							? 1u : 0u;
				gSphereReverseSweptMetrics.
					twoSidedResponseObserved = 1;
				gMetrics.speculativeCcdPreventedTunneling =
					gSphereReverseSweptMetrics.responseObserved;
				gMetrics.speculativeCcdNegativeControlTunneled =
					gSphereReverseSweptMetrics.
						negativeControlPassed;
			}

			if(gSceneCpuVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuVolume);
				gMetrics.sceneActorRemoved =
					gSceneCpuVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuSecondVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuSecondVolume);
				gMetrics.sceneSecondVolumeActorRemoved =
					gSceneCpuSecondVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuDynamicActor);
				gMetrics.sceneDynamicActorRemoved =
					gSceneCpuDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuSecondDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuSecondDynamicActor);
				gMetrics.sceneSecondDynamicActorRemoved =
					gSceneCpuSecondDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
		}
		else if(sphereReverseSweptCcdCase &&
			gSceneCpuSecondVolume &&
			(staticSphereReverseSweptCcdCase ||
				(gSceneCpuDynamicActor &&
				 gSceneCpuSecondDynamicActor)) &&
			gMetrics.completedFrames == 1)
		{
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuSecondVolume);
			const PxVec3 positiveSphereStart(
				deformingReverseSweptCcdCase
					? -1.68f
				: rotationalFiniteReverseSweptCcdCase
					? -1.45f : -1.35f,
				rotationalFiniteReverseSweptCcdCase
					? (rotationalConvexReverseSweptCcdCase
						? -0.85f : -1.0f) :
					(staticSphereReverseSweptCcdCase ? 0.0f : 1.1f),
				deformingReverseSweptCcdCase
					? 0.32f
				: rotationalFiniteReverseSweptCcdCase
					? 0.25f : 0.15f);
			const PxVec3 negativeSphereStart(
				deformingReverseSweptCcdCase
					? 1.32f
				: rotationalFiniteReverseSweptCcdCase
					? 1.55f : 1.65f,
				rotationalFiniteReverseSweptCcdCase
					? (rotationalConvexReverseSweptCcdCase
						? -0.85f : -1.0f) :
					(staticSphereReverseSweptCcdCase ? 0.0f : 1.1f),
				deformingReverseSweptCcdCase
					? 0.32f
				: rotationalFiniteReverseSweptCcdCase
					? 0.25f : 0.15f);
			const PxTransform positiveRigidPose =
				staticSphereReverseSweptCcdCase
					? PxTransform(positiveSphereStart)
					: gSceneCpuDynamicActor->getGlobalPose();
			const PxTransform negativeRigidPose =
				staticSphereReverseSweptCcdCase
					? PxTransform(negativeSphereStart)
					: gSceneCpuSecondDynamicActor->getGlobalPose();
			const PxVec3 positiveSphereCurrent =
				positiveRigidPose.p;
			const PxVec3 negativeSphereCurrent =
				negativeRigidPose.p;
			const PxVec3 sweepEnd =
				deformingReverseSweptCcdCase
					? positiveSphereStart
				: staticSphereReverseSweptCcdCase
					? positiveSphereStart +
						PxVec3(
							0.0f,
							convexReverseSweptCcdCase
								? 2.2f : 1.1f,
							0.0f)
					: positiveSphereCurrent;
			PxReal faceSeparation = PX_MAX_F32;
			PxReal minimumVertexSweepSeparation = PX_MAX_F32;
			if(rotationalFiniteReverseSweptCcdCase)
			{
				const PxQuat startRotation(
					PxPi / 6.0f,
					PxVec3(0.0f, 0.0f, 1.0f));
				auto getAngularTravel = [&](
					const PxQuat& endRotation)
				{
					const PxReal alignment = PxClamp(
						PxAbs(
							startRotation.dot(
								endRotation.getNormalized())),
						0.0f, 1.0f);
					return 2.0f * PxAcos(alignment);
				};
				gCapsuleRotationalSweepMetrics.
					positiveAngularTravel =
						getAngularTravel(positiveRigidPose.q);
				gCapsuleRotationalSweepMetrics.
					negativeAngularTravel =
						getAngularTravel(negativeRigidPose.q);
			}
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite() &&
				positiveSphereCurrent.isFinite() &&
				negativeSphereCurrent.isFinite() &&
				(deformingReverseSweptCcdCase
					? getSceneCpuVolumeSweptGeometrySampler().
						measureDeformingReverseSweptProof(
						gSceneCpuVolume,
						positiveRigidPose,
						convexReverseSweptCcdCase
							? 0.25f : 0.18f,
						capsuleReverseSweptCcdCase
							? 0.02f : 0.0f,
						convexReverseSweptCcdCase,
						gDeformingVolumeReverseSweptMetrics.
							endpointMinSeparation,
						gDeformingVolumeReverseSweptMetrics.
							midSweepMinSeparation,
						minimumVertexSweepSeparation)
				: rotationalFiniteReverseSweptCcdCase
					? (rotationalConvexReverseSweptCcdCase
						? getSceneCpuVolumeSweptGeometrySampler().
							measureRotationalConvexReverseSweptSeparations(
							gSceneCpuVolume,
							positiveRigidPose,
							PxTransform(
								positiveSphereStart,
								PxQuat(
									PxPi / 6.0f,
									PxVec3(0.0f, 0.0f, 1.0f))),
							PxTransform(
								negativeRigidPose.p -
									(negativeSphereStart -
									 positiveSphereStart),
								negativeRigidPose.q),
							faceSeparation,
							minimumVertexSweepSeparation,
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation,
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation)
						: getSceneCpuVolumeSweptGeometrySampler().
							measureRotationalCapsuleReverseSweptSeparations(
						gSceneCpuVolume,
						positiveRigidPose,
						PxTransform(
							positiveSphereStart,
							PxQuat(
								PxPi / 6.0f,
								PxVec3(0.0f, 0.0f, 1.0f))),
						PxTransform(
							negativeRigidPose.p -
								(negativeSphereStart -
								 positiveSphereStart),
							negativeRigidPose.q),
						0.1f, 1.0f,
						faceSeparation,
						minimumVertexSweepSeparation,
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation,
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation))
					: getSceneCpuVolumeSweptGeometrySampler().
						measureSphereReverseSweptSeparations(
						gSceneCpuVolume,
						positiveSphereCurrent,
						positiveSphereStart,
						sweepEnd,
						0.25f,
						capsuleReverseSweptCcdCase ? 0.02f : 0.0f,
						convexReverseSweptCcdCase,
						faceSeparation,
						minimumVertexSweepSeparation));
			if(!finite)
				gSphereReverseSweptMetrics.nonFiniteSamples++;
			else
			{
				gSphereReverseSweptMetrics.positiveDisplacement =
					(positiveCentroid -
						gSceneCpuMovingSpherePositiveInitial).
							magnitude();
				gSphereReverseSweptMetrics.negativeDisplacement =
					(negativeCentroid -
						gSceneCpuMovingSphereNegativeInitial).
							magnitude();
				gSphereReverseSweptMetrics.positiveDrop =
					gSceneCpuMovingSpherePositiveInitial.y -
						positiveCentroid.y;
				gSphereReverseSweptMetrics.negativeDrop =
					gSceneCpuMovingSphereNegativeInitial.y -
						negativeCentroid.y;
				gSphereReverseSweptMetrics.positiveRigidDrop =
					staticSphereReverseSweptCcdCase
						? 0.0f
					: rotationalFiniteReverseSweptCcdCase
						? gCapsuleRotationalSweepMetrics.
							positiveAngularTravel
						: gSceneCpuDynamicInitialY -
							positiveSphereCurrent.y;
				gSphereReverseSweptMetrics.negativeRigidDrop =
					staticSphereReverseSweptCcdCase
						? 0.0f
					: rotationalFiniteReverseSweptCcdCase
						? gCapsuleRotationalSweepMetrics.
							negativeAngularTravel
						: gSceneCpuSecondDynamicInitialY -
							negativeSphereCurrent.y;
				if(deformingReverseSweptCcdCase)
				{
					const PxVec4* positivePositions =
						gSceneCpuVolume->
							getPositionInvMassBufferH();
					const PxVec4* negativePositions =
						gSceneCpuSecondVolume->
							getPositionInvMassBufferH();
					const PxU32 positiveVertexCount =
						gSceneCpuVolume->getCollisionMesh()->
							getNbVertices();
					const PxU32 negativeVertexCount =
						gSceneCpuSecondVolume->getCollisionMesh()->
							getNbVertices();
					if(!positivePositions || !negativePositions ||
						positiveVertexCount != negativeVertexCount)
					{
						gSphereReverseSweptMetrics.
							nonFiniteSamples++;
					}
					else
					{
						for(PxU32 vertexIndex = 0;
							vertexIndex < positiveVertexCount;
							++vertexIndex)
						{
							const PxVec3 positiveLocal =
								positivePositions[vertexIndex].
									getXYZ() -
								PxVec3(-2.0f, 0.0f, 0.0f);
							const PxVec3 negativeLocal =
								negativePositions[vertexIndex].
									getXYZ() -
								PxVec3(1.0f, 0.0f, 0.0f);
							gDeformingVolumeReverseSweptMetrics.
								responseDelta = PxMax(
									gDeformingVolumeReverseSweptMetrics.
										responseDelta,
									(positiveLocal - negativeLocal).
										magnitude());
						}
					}
				}
				gSphereReverseSweptMetrics.faceSeparation =
					deformingReverseSweptCcdCase
						? gDeformingVolumeReverseSweptMetrics.
							endpointMinSeparation
						: faceSeparation;
				gSphereReverseSweptMetrics.
					minimumVertexSweepSeparation =
						minimumVertexSweepSeparation;
				gSphereReverseSweptMetrics.vertexSweepExcluded =
					minimumVertexSweepSeparation >
						(deformingReverseSweptCcdCase
							? 0.05f
						: rotationalFiniteReverseSweptCcdCase
							? 0.10f :
						((capsuleReverseSweptCcdCase ||
						  convexReverseSweptCcdCase)
							? 0.05f : 0.10f))
						? 1u : 0u;
				if(deformingReverseSweptCcdCase)
					gDeformingVolumeReverseSweptMetrics.
						geometricSweepIsolated =
							gDeformingVolumeReverseSweptMetrics.
								endpointMinSeparation > 0.02f &&
							gDeformingVolumeReverseSweptMetrics.
								midSweepMinSeparation <
									(convexReverseSweptCcdCase
										? 0.02f : 0.0f) &&
							minimumVertexSweepSeparation > 0.05f
								? 1u : 0u;
				if(rotationalFiniteReverseSweptCcdCase)
					gCapsuleRotationalSweepMetrics.sweepIsolated =
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation > 0.05f &&
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation < -0.05f &&
						minimumVertexSweepSeparation > 0.10f
							? 1u : 0u;
				gSphereReverseSweptMetrics.responseObserved =
					(deformingReverseSweptCcdCase
						? gDeformingVolumeReverseSweptMetrics.
								responseDelta > 0.01f &&
							gSphereReverseSweptMetrics.
								positiveDrop + 0.01f <
							gSphereReverseSweptMetrics.
								negativeDrop
					: staticSphereReverseSweptCcdCase
						? gSphereReverseSweptMetrics.positiveDrop +
								0.03f <
							gSphereReverseSweptMetrics.negativeDrop
						: gSphereReverseSweptMetrics.
								positiveDisplacement >
									(rotationalFiniteReverseSweptCcdCase
										? 0.02f :
									 dynamicSphereReverseSweptCcdCase
										? 0.01f : 0.02f) &&
							gSphereReverseSweptMetrics.
								negativeDisplacement < 5.0e-3f) &&
					(deformingReverseSweptCcdCase ||
						faceSeparation > -0.15f) ? 1u : 0u;
				gSphereReverseSweptMetrics.negativeControlPassed =
					(deformingReverseSweptCcdCase
						? gSphereReverseSweptMetrics.
								negativeDrop > 0.15f
					: staticSphereReverseSweptCcdCase
						? gSphereReverseSweptMetrics.negativeDrop >
							0.8f
						: gSphereReverseSweptMetrics.
							negativeDisplacement < 5.0e-3f) &&
					(!dynamicSphereReverseSweptCcdCase ||
						gSphereReverseSweptMetrics.
							negativeRigidDrop >
								(rotationalFiniteReverseSweptCcdCase
									? 0.8f : 1.5f))
						? 1u : 0u;
				gSphereReverseSweptMetrics.
					twoSidedResponseObserved =
						!dynamicSphereReverseSweptCcdCase ||
						gSphereReverseSweptMetrics.
								positiveRigidDrop + 0.05f <
							gSphereReverseSweptMetrics.
								negativeRigidDrop
							? 1u : 0u;
				gMetrics.speculativeCcdPreventedTunneling =
					gSphereReverseSweptMetrics.responseObserved;
				gMetrics.speculativeCcdNegativeControlTunneled =
					gSphereReverseSweptMetrics.
						negativeControlPassed;
			}

			if(gSceneCpuVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuVolume);
				gMetrics.sceneActorRemoved =
					gSceneCpuVolume->getScene() == NULL ? 1u : 0u;
			}
			if(gSceneCpuSecondVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuSecondVolume);
				gMetrics.sceneSecondVolumeActorRemoved =
					gSceneCpuSecondVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuStaticActor &&
				gSceneCpuStaticActor->getScene() == gScene)
				gScene->removeActor(*gSceneCpuStaticActor);
			if(gSceneCpuDynamicActor &&
				gSceneCpuDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuDynamicActor);
				gMetrics.sceneDynamicActorRemoved =
					gSceneCpuDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuSecondDynamicActor &&
				gSceneCpuSecondDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(
					*gSceneCpuSecondDynamicActor);
				gMetrics.sceneSecondDynamicActorRemoved =
					gSceneCpuSecondDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
		}
		else if(dynamicFiniteRelativeSweptCcdCase &&
			gSceneCpuSecondVolume &&
			gSceneCpuDynamicActor &&
			gSceneCpuSecondDynamicActor &&
			gMetrics.completedFrames == 1)
		{
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuSecondVolume);
			const PxTransform positiveRigidPose =
				gSceneCpuDynamicActor->getGlobalPose();
			const PxTransform negativeRigidPose =
				gSceneCpuSecondDynamicActor->getGlobalPose();
			const PxVec3 positiveRigidCenter =
				positiveRigidPose.p;
			const PxVec3 negativeRigidCenter =
				negativeRigidPose.p;
			if(dynamicRotatingCapsuleSpeculativeCcdCase ||
				dynamicRotatingConvexSpeculativeCcdCase)
			{
				const PxQuat startRotation(
					-PxPi / 6.0f,
					PxVec3(0.0f, 0.0f, 1.0f));
				auto getAngularTravel = [&](
					const PxQuat& endRotation)
				{
					const PxReal alignment = PxClamp(
						PxAbs(
							startRotation.dot(
								endRotation.getNormalized())),
						0.0f, 1.0f);
					return 2.0f * PxAcos(alignment);
				};
				gCapsuleRotationalSweepMetrics.
					positiveAngularTravel =
						getAngularTravel(positiveRigidPose.q);
				gCapsuleRotationalSweepMetrics.
					negativeAngularTravel =
						getAngularTravel(negativeRigidPose.q);
				const PxTransform startPose(
					dynamicRotatingCapsuleSpeculativeCcdCase
						? SCENE_CPU_DYNAMIC_ROTATING_CAPSULE_SWEEP_CENTER
						: PxVec3(-2.9f, 0.0f, 0.0f),
					startRotation);
				const PxTransform endPose(
					startPose.p,
					negativeRigidPose.q.getNormalized());
				bool rotationalMetricsFinite =
					dynamicRotatingConvexSpeculativeCcdCase
						? getSceneCpuVolumeSweptGeometrySampler().
							measureRotationalConvexPointSweepSeparations(
							startPose, endPose,
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation,
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation)
						: !gSceneCpuCapsuleRotationalSweptInitialPositions.
							empty();
				if(!dynamicRotatingConvexSpeculativeCcdCase)
				{
					for(PxU32 i = 0;
						i <
							gSceneCpuCapsuleRotationalSweptInitialPositions.
								size();
						++i)
					{
						const PxVec3 point =
							gSceneCpuCapsuleRotationalSweptInitialPositions[
								i];
						if(!point.isFinite())
						{
							rotationalMetricsFinite = false;
							break;
						}
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation = PxMin(
								gCapsuleRotationalSweepMetrics.
									endpointMinSeparation,
								PxMin(
									getCapsuleSignedSeparation(
										point, startPose, 0.1f, 1.0f),
									getCapsuleSignedSeparation(
										point, endPose, 0.1f, 1.0f)));
						for(PxU32 sample = 1; sample < 64; ++sample)
						{
							const PxReal time =
								PxReal(sample) / 64.0f;
							const PxTransform samplePose(
								startPose.p,
								PxSlerp(
									time, startRotation,
									negativeRigidPose.q.getNormalized()).
										getNormalized());
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation = PxMin(
									gCapsuleRotationalSweepMetrics.
										midSweepMinSeparation,
									getCapsuleSignedSeparation(
										point, samplePose,
										0.1f, 1.0f));
						}
					}
				}
				rotationalMetricsFinite =
					rotationalMetricsFinite &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation) &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation) &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							positiveAngularTravel) &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							negativeAngularTravel);
				if(!rotationalMetricsFinite)
				{
					gCapsuleRotationalSweepMetrics.nonFiniteSamples++;
					gMetrics.nonFiniteParticleSamples++;
				}
				else
				{
					gCapsuleRotationalSweepMetrics.sweepIsolated =
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation > 0.05f &&
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation <
								(dynamicRotatingConvexSpeculativeCcdCase
									? 1.0e-5f : -0.05f)
							? 1u : 0u;
				}
			}
			PxReal positiveMinSeparation = PX_MAX_F32;
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite() &&
				positiveRigidCenter.isFinite() &&
				negativeRigidCenter.isFinite() &&
				(dynamicConvexRelativeSweptCcdCase
					? getSceneCpuVolumeSingleConvexMinSeparation(
						gSceneCpuVolume,
						positiveRigidPose,
						positiveMinSeparation)
					: dynamicCapsuleRelativeSweptCcdCase
					? getSceneCpuVolumeSingleCapsuleMinSeparation(
						gSceneCpuVolume,
						positiveRigidPose,
						dynamicRotatingCapsuleSpeculativeCcdCase
							? 0.1f : 0.8f,
						dynamicRotatingCapsuleSpeculativeCcdCase
							? 1.0f : 0.3f,
						positiveMinSeparation)
					: getSceneCpuVolumeSingleSphereMinSeparation(
						gSceneCpuVolume,
						positiveRigidCenter,
						0.8f,
						positiveMinSeparation));
			if(!finite)
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.
					dynamicSphereSweepPositiveSoftDisplacement =
						(positiveCentroid -
							gSceneCpuMovingSpherePositiveInitial).
								magnitude();
				gMetrics.
					dynamicSphereSweepNegativeSoftDisplacement =
						(negativeCentroid -
							gSceneCpuMovingSphereNegativeInitial).
								magnitude();
				gMetrics.dynamicSphereSweepPositiveRigidDrop =
					(dynamicRotatingCapsuleSpeculativeCcdCase ||
					 dynamicRotatingConvexSpeculativeCcdCase)
						? gCapsuleRotationalSweepMetrics.
							positiveAngularTravel
						: gSceneCpuDynamicInitialY -
							positiveRigidCenter.y;
				gMetrics.dynamicSphereSweepNegativeRigidDrop =
					(dynamicRotatingCapsuleSpeculativeCcdCase ||
					 dynamicRotatingConvexSpeculativeCcdCase)
						? gCapsuleRotationalSweepMetrics.
							negativeAngularTravel
						: gSceneCpuSecondDynamicInitialY -
							negativeRigidCenter.y;
				gMetrics.
					dynamicSphereSweepPositiveMinSeparation =
						positiveMinSeparation;
				gMetrics.dynamicSphereSweepResponseObserved =
					gMetrics.
						dynamicSphereSweepPositiveSoftDisplacement >
							0.02f &&
					gMetrics.
						dynamicSphereSweepPositiveMinSeparation >
							-0.15f ? 1u : 0u;
				gMetrics.
					dynamicSphereSweepNegativeControlTunneled =
						gMetrics.
							dynamicSphereSweepNegativeSoftDisplacement <
								5.0e-3f &&
						gMetrics.
							dynamicSphereSweepNegativeRigidDrop >
								((dynamicRotatingCapsuleSpeculativeCcdCase ||
								  dynamicRotatingConvexSpeculativeCcdCase)
									? 0.8f : 1.5f)
							? 1u : 0u;
				gMetrics.
					dynamicSphereSweepTwoSidedResponseObserved =
						gMetrics.
							dynamicSphereSweepPositiveRigidDrop +
								0.05f <
						gMetrics.
							dynamicSphereSweepNegativeRigidDrop
							? 1u : 0u;
				gMetrics.speculativeCcdPreventedTunneling =
					gMetrics.dynamicSphereSweepResponseObserved;
				gMetrics.speculativeCcdNegativeControlTunneled =
					gMetrics.
						dynamicSphereSweepNegativeControlTunneled;
			}

			if(gSceneCpuVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuVolume);
				gMetrics.sceneActorRemoved =
					gSceneCpuVolume->getScene() == NULL ? 1u : 0u;
			}
			if(gSceneCpuSecondVolume->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuSecondVolume);
				gMetrics.sceneSecondVolumeActorRemoved =
					gSceneCpuSecondVolume->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(*gSceneCpuDynamicActor);
				gMetrics.sceneDynamicActorRemoved =
					gSceneCpuDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
			if(gSceneCpuSecondDynamicActor->getScene() == gScene)
			{
				gScene->removeActor(
					*gSceneCpuSecondDynamicActor);
				gMetrics.sceneSecondDynamicActorRemoved =
					gSceneCpuSecondDynamicActor->getScene() == NULL
						? 1u : 0u;
			}
		}
		else if(movingKinematicFiniteSpeculativeCcdCase &&
			gSceneCpuSecondVolume &&
			gSceneCpuDynamicActor)
		{
			if((rotatingKinematicCapsuleSpeculativeCcdCase ||
				rotatingKinematicConvexSpeculativeCcdCase) &&
				gMetrics.completedFrames == 1)
			{
				const PxVec3 rigidCenter(-2.9f, 0.0f, 0.0f);
				const PxTransform startPose(
					rigidCenter,
					PxQuat(
						-PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f)));
				const PxTransform endPose(
					rigidCenter,
					PxQuat(
						PxPi / 3.0f,
						PxVec3(0.0f, 0.0f, 1.0f)));
				const PxTransform midPose(rigidCenter);
				bool rotationalMetricsFinite =
					rotatingKinematicConvexSpeculativeCcdCase
						? getSceneCpuVolumeSweptGeometrySampler().
							measureRotationalConvexPointSweepSeparations(
							startPose, endPose,
							gCapsuleRotationalSweepMetrics.
								endpointMinSeparation,
							gCapsuleRotationalSweepMetrics.
								midSweepMinSeparation)
						: !gSceneCpuCapsuleRotationalSweptInitialPositions.
							empty();
				if(!rotatingKinematicConvexSpeculativeCcdCase)
				{
					for(PxU32 i = 0;
						i <
							gSceneCpuCapsuleRotationalSweptInitialPositions.
								size();
						++i)
					{
						const PxVec3 point =
							gSceneCpuCapsuleRotationalSweptInitialPositions[
								i];
						if(!point.isFinite())
						{
							rotationalMetricsFinite = false;
							break;
						}
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation = PxMin(
								gCapsuleRotationalSweepMetrics.
									endpointMinSeparation,
								PxMin(
									getCapsuleSignedSeparation(
										point, startPose, 0.1f, 1.0f),
									getCapsuleSignedSeparation(
										point, endPose, 0.1f, 1.0f)));
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation = PxMin(
								gCapsuleRotationalSweepMetrics.
									midSweepMinSeparation,
								getCapsuleSignedSeparation(
									point, midPose, 0.1f, 1.0f));
					}
				}
				rotationalMetricsFinite =
					rotationalMetricsFinite &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation) &&
					PxIsFinite(
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation);
				if(!rotationalMetricsFinite)
				{
					gCapsuleRotationalSweepMetrics.nonFiniteSamples++;
					gMetrics.nonFiniteParticleSamples++;
				}
				else
				{
					gCapsuleRotationalSweepMetrics.sweepIsolated =
						gCapsuleRotationalSweepMetrics.
							endpointMinSeparation > 0.05f &&
						gCapsuleRotationalSweepMetrics.
							midSweepMinSeparation <
								(rotatingKinematicConvexSpeculativeCcdCase
									? 1.0e-5f : -0.05f)
							? 1u : 0u;
				}
			}
			const PxVec3 positiveCentroid =
				getSceneCpuVolumeCentroid(gSceneCpuVolume);
			const PxVec3 negativeCentroid =
				getSceneCpuVolumeCentroid(
					gSceneCpuSecondVolume);
			PxReal positiveMinSeparation = PX_MAX_F32;
			const bool finite =
				positiveCentroid.isFinite() &&
				negativeCentroid.isFinite() &&
				(movingKinematicConvexSpeculativeCcdCase
					? getSceneCpuVolumeSingleConvexMinSeparation(
						gSceneCpuVolume,
						gSceneCpuDynamicActor->getGlobalPose(),
						positiveMinSeparation)
					: movingKinematicCapsuleSpeculativeCcdCase
					? getSceneCpuVolumeSingleCapsuleMinSeparation(
						gSceneCpuVolume,
						gSceneCpuDynamicActor->getGlobalPose(),
						rotatingKinematicCapsuleSpeculativeCcdCase
							? 0.1f : 0.8f,
						rotatingKinematicCapsuleSpeculativeCcdCase
							? 1.0f : 0.3f,
						positiveMinSeparation)
					: getSceneCpuVolumeSingleSphereMinSeparation(
						gSceneCpuVolume,
						gSceneCpuDynamicActor->getGlobalPose().p,
						0.8f, positiveMinSeparation));
			if(!finite)
				gMetrics.nonFiniteParticleSamples++;
			else if(gMetrics.completedFrames <= 3)
			{
				gMetrics.movingSpherePositiveDisplacement =
					PxMax(
						gMetrics.movingSpherePositiveDisplacement,
						(positiveCentroid -
							gSceneCpuMovingSpherePositiveInitial).
								magnitude());
				gMetrics.movingSphereNegativeDisplacement =
					PxMax(
						gMetrics.movingSphereNegativeDisplacement,
						(negativeCentroid -
							gSceneCpuMovingSphereNegativeInitial).
								magnitude());
				gMetrics.movingSpherePositiveMinSeparation =
					PxMin(
						gMetrics.movingSpherePositiveMinSeparation,
						positiveMinSeparation);
				if(gMetrics.completedFrames == 1)
				{
					gMetrics.movingSphereNegativeControlHeld =
						gMetrics.
							movingSphereNegativeDisplacement <
								5.0e-3f ? 1u : 0u;
					if(gSceneCpuSecondVolume->getScene() == gScene)
					{
						gScene->removeActor(
							*gSceneCpuSecondVolume);
						gMetrics.
							sceneSecondVolumeActorRemoved =
								gSceneCpuSecondVolume->
									getScene() == NULL ? 1u : 0u;
					}
					if(gSceneCpuSecondDynamicActor &&
						gSceneCpuSecondDynamicActor->getScene() ==
							gScene)
					{
						gScene->removeActor(
							*gSceneCpuSecondDynamicActor);
						gMetrics.
							sceneSecondDynamicActorRemoved =
								gSceneCpuSecondDynamicActor->
									getScene() == NULL ? 1u : 0u;
					}
				}
				if(gMetrics.completedFrames == 3)
				{
					gMetrics.movingSphereCcdResponseObserved =
						gMetrics.
							movingSpherePositiveDisplacement >
								0.02f &&
						gMetrics.
							movingSpherePositiveMinSeparation >
								-0.10f ? 1u : 0u;
					gMetrics.speculativeCcdPreventedTunneling =
						gMetrics.
							movingSphereCcdResponseObserved;
					gMetrics.
						speculativeCcdNegativeControlTunneled =
							gMetrics.
								movingSphereNegativeControlHeld;
					if(gSceneCpuVolume->getScene() == gScene)
					{
						gScene->removeActor(*gSceneCpuVolume);
						gMetrics.sceneActorRemoved =
							gSceneCpuVolume->getScene() == NULL
								? 1u : 0u;
					}
					if(gSceneCpuSecondVolume->getScene() == gScene)
					{
						gScene->removeActor(
							*gSceneCpuSecondVolume);
						gMetrics.
							sceneSecondVolumeActorRemoved =
								gSceneCpuSecondVolume->
									getScene() == NULL ? 1u : 0u;
					}
				}
			}
		}
		else if(speculativeCcdCase &&
			!sphereReverseSweptCcdCase &&
			!triangleSurfaceSweptCcdCase &&
			!dynamicFiniteRelativeSweptCcdCase &&
			gSceneCpuSecondVolume)
		{
			const PxReal positiveMinY =
				getSceneCpuVolumeCollisionMinY(gSceneCpuVolume);
			const PxReal negativeMaxY =
				getSceneCpuVolumeCollisionMaxY(
					gSceneCpuSecondVolume);
			PxReal positiveMinSeparation = PX_MAX_F32;
			const PxVec3 convexCenters[] =
			{
				PxVec3(-1.5f, 0.25f, 0.5f),
				PxVec3(1.5f, 0.25f, 0.5f)
			};
			const bool hasPositiveMinSeparation =
				!finiteSmoothSpeculativeCcdCase ||
				(convexSpeculativeCcdCase
					? getSceneCpuVolumeConvexMinSeparation(
						gSceneCpuVolume,
						convexCenters, 2,
						positiveMinSeparation)
					: capsuleSpeculativeCcdCase
					? getSceneCpuVolumeCapsuleClusterMinSeparation(
						gSceneCpuVolume, positiveMinSeparation)
					: getSceneCpuVolumeSphereMinSeparation(
						gSceneCpuVolume, positiveMinSeparation));
			if(!PxIsFinite(positiveMinY) ||
				!PxIsFinite(negativeMaxY) ||
				!hasPositiveMinSeparation)
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.speculativeCcdPositiveMinY =
					PxMin(
						gMetrics.speculativeCcdPositiveMinY,
						positiveMinY);
				if(finiteSmoothSpeculativeCcdCase)
					gMetrics.
						speculativeCcdPositiveMinSeparation =
							PxMin(
								gMetrics.
									speculativeCcdPositiveMinSeparation,
								positiveMinSeparation);
				gMetrics.speculativeCcdPreventedTunneling =
					(finiteSmoothSpeculativeCcdCase
						? gMetrics.
							speculativeCcdPositiveMinSeparation >=
								-0.05f
						: gMetrics.speculativeCcdPositiveMinY >=
							(planeSpeculativeCcdCase
								? 0.49f : 0.50f))
						? 1u : 0u;
				if(gMetrics.completedFrames <= 3)
				{
					if((!planeSpeculativeCcdCase &&
							!finiteSmoothSpeculativeCcdCase) ||
						convexSpeculativeCcdCase ||
						gMetrics.completedFrames == 1)
						gMetrics.speculativeCcdNegativeMaxY =
							negativeMaxY;
					const bool finalizeNegativeControl =
						convexSpeculativeCcdCase
							? gMetrics.completedFrames == 3
							: finiteSmoothSpeculativeCcdCase
							? gMetrics.completedFrames == 1
							: gMetrics.completedFrames == 3;
					if(finalizeNegativeControl)
					{
						gMetrics.
							speculativeCcdNegativeControlTunneled =
								gMetrics.
									speculativeCcdNegativeMaxY <=
									(planeSpeculativeCcdCase
										? 0.45f : 0.44f)
									? 1u : 0u;
						if(gSceneCpuSecondVolume->getScene() == gScene)
						{
							gScene->removeActor(
								*gSceneCpuSecondVolume);
							gMetrics.
								sceneSecondVolumeActorRemoved =
								gSceneCpuSecondVolume->
									getScene() == NULL ? 1u : 0u;
						}
					}
				}
			}
		}
		if(rigidAttachmentCase &&
			(gSceneCpuAttachmentBody || gSceneCpuStaticActor))
		{
			const PxTransform rigidPose =
				staticAttachmentCase
					? gSceneCpuStaticActor->getGlobalPose()
					: gSceneCpuAttachmentBody->getGlobalPose();
			const PxReal rigidSpeed =
				staticAttachmentCase
					? 0.0f
					: gSceneCpuAttachmentBody->getLinearVelocity().
						magnitude();
			const PxReal separation =
				(getSceneCpuAttachmentPoint() -
					rigidPose.transform(
						gSceneCpuRigidAttachmentLocalOffset)).
					magnitude();
			if(!rigidPose.isValid() ||
				!PxIsFinite(rigidSpeed) ||
				!PxIsFinite(separation))
			{
				gMetrics.nonFiniteParticleSamples++;
			}
			else
			{
				gMetrics.sceneRigidAttachmentMaxRigidSpeed =
					PxMax(
						gMetrics.sceneRigidAttachmentMaxRigidSpeed,
						rigidSpeed);
				if(!gMetrics.sceneRigidAttachmentReleased)
				{
					gMetrics.
						sceneRigidAttachmentMaxRigidDisplacement =
						PxMax(
							gMetrics.
								sceneRigidAttachmentMaxRigidDisplacement,
							(rigidPose.p -
								gSceneCpuRigidAttachmentInitialPosition).
								magnitude());
					if(gMetrics.
						sceneRigidAttachmentMaxRigidDisplacement >
							0.02f)
						gMetrics.
							sceneRigidAttachmentRigidMoved = 1;
					gMetrics.sceneRigidAttachmentMaxDrift =
						PxMax(
							gMetrics.
								sceneRigidAttachmentMaxDrift,
							separation);
					const PxU32 churnFrame = PxMax<PxU32>(
						1, gHeadlessOptions.frames / 3);
					if(gMetrics.completedFrames > churnFrame &&
						separation < 0.05f)
						gMetrics.
							sceneRigidAttachmentHeldAcrossReadd =
								1;
				}
				else
				{
					gMetrics.
						sceneRigidAttachmentReleasedSeparation =
						PxMax(
							gMetrics.
								sceneRigidAttachmentReleasedSeparation,
							separation);
					if(gMetrics.
						sceneRigidAttachmentReleasedSeparation >
							0.2f)
						gMetrics.
							sceneRigidAttachmentSeparatedAfterRelease =
								1;
				}
			}
			if(articulationAttachmentCase)
			{
				const PxTransform rootPose =
					gSceneCpuAttachmentRoot->getGlobalPose();
				const PxVec3 childDelta =
					rigidPose.p -
						gSceneCpuArticulationChildInitialPose.p;
				const PxReal rootDisplacement =
					(rootPose.p -
						gSceneCpuArticulationRootInitialPose.p).
							magnitude();
				const PxReal forbiddenDisplacement =
					PxSqrt(
						childDelta.y * childDelta.y +
						childDelta.z * childDelta.z);
				const PxQuat childOrientationError =
					rigidPose.q *
						gSceneCpuArticulationChildInitialPose.q.
							getConjugate();
				const PxReal childAngularDisplacement =
					2.0f * childOrientationError.
						getImaginaryPart().magnitude();
				if(!rootPose.isValid() ||
					!PxIsFinite(rootDisplacement) ||
					!PxIsFinite(forbiddenDisplacement) ||
					!PxIsFinite(childAngularDisplacement))
				{
					gMetrics.nonFiniteParticleSamples++;
				}
				else
				{
					gMetrics.sceneArticulationRootMaxDisplacement =
						PxMax(
							gMetrics.
								sceneArticulationRootMaxDisplacement,
							rootDisplacement);
					gMetrics.
						sceneArticulationChildMaxForbiddenDisplacement =
						PxMax(
							gMetrics.
								sceneArticulationChildMaxForbiddenDisplacement,
							forbiddenDisplacement);
					gMetrics.
						sceneArticulationChildMaxAngularDisplacement =
						PxMax(
							gMetrics.
								sceneArticulationChildMaxAngularDisplacement,
							childAngularDisplacement);
					if(!gSceneCpuAttachmentArticulation->isSleeping())
						gMetrics.sceneArticulationWoke = 1;
					if(rootDisplacement <= 1.0e-4f)
						gMetrics.sceneArticulationRootStable = 1;
					if(forbiddenDisplacement <= 1.0e-3f &&
						childAngularDisplacement <= 1.0e-3f)
						gMetrics.
							sceneArticulationJointSubspaceHeld = 1;
				}
			}
			else if(kinematicAttachmentCase ||
				staticAttachmentCase)
			{
				if(gMetrics.sceneKinematicTargetIssued)
				{
					if(!gSceneCpuVolume->isSleeping())
						gMetrics.sceneKinematicSoftWoke = 1;
					const PxReal softDisplacement =
						(getSceneCpuAttachmentPoint() -
							gSceneCpuKinematicAttachmentSoftBaseline).
								magnitude();
					gMetrics.sceneKinematicSoftDisplacement =
						PxMax(
							gMetrics.
								sceneKinematicSoftDisplacement,
							softDisplacement);
					if(softDisplacement > 0.02f)
						gMetrics.sceneKinematicSoftMoved = 1;
					const PxQuat orientationError =
						rigidPose.q *
							gSceneCpuKinematicAttachmentCommand.q.
								getConjugate();
					const PxReal poseError =
						(rigidPose.p -
							gSceneCpuKinematicAttachmentCommand.p).
								magnitude() +
						2.0f * orientationError.
							getImaginaryPart().magnitude();
					gMetrics.sceneKinematicMaxPoseError =
						PxMax(
							gMetrics.sceneKinematicMaxPoseError,
							poseError);
					if(gSceneCpuKinematicAttachmentProgress >= 1.0f &&
						poseError <= 1.0e-4f)
						gMetrics.sceneKinematicTargetReached = 1;
				}
			}
			else if(!staticAttachmentCase &&
				!gSceneCpuDynamicActor->isSleeping())
				gMetrics.sceneRigidAttachmentRigidWoke = 1;
		}
		if(softSleepWakeCase || softRigidWakeCase ||
			kinematicBoxCase ||
			bufferMutationCase || softSoftWakeCase ||
			kinematicAttachmentCase || staticAttachmentCase)
		{
			const bool sleeping = gSceneCpuVolume->isSleeping();
			const PxReal maxSpeed = getSceneCpuVolumeMaxSpeed();
			if(!gMetrics.sceneSoftFirstSlept && sleeping)
			{
				gMetrics.sceneSoftFirstSlept = 1;
				gMetrics.sceneSoftFirstSleepFrame =
					gMetrics.completedFrames;
				gSceneCpuFirstSleepCentroidY =
					getSceneCpuVolumeCentroidY();
				gMetrics.sceneSoftSleepWakeCounterZero =
					gSceneCpuVolume->getWakeCounter() == 0.0f
						? 1u : 0u;
				gMetrics.sceneSoftSleepVelocitiesZero =
					maxSpeed <= 1.0e-7f ? 1u : 0u;
			}
			else if(softSleepWakeCase &&
				gMetrics.sceneSoftCounterWakeIssued &&
				!gMetrics.sceneSoftWokeByCounter && !sleeping)
			{
				gMetrics.sceneSoftWokeByCounter = 1;
				gMetrics.sceneSoftCounterWakeFrame =
					gMetrics.completedFrames;
			}
			else if(softSleepWakeCase &&
				gMetrics.sceneSoftWokeByCounter &&
				!gMetrics.sceneSoftSecondSlept && sleeping)
			{
				gMetrics.sceneSoftSecondSlept = 1;
				gMetrics.sceneSoftSecondSleepFrame =
					gMetrics.completedFrames;
				gSceneCpuSecondSleepCentroidY =
					getSceneCpuVolumeCentroidY();
			}
			else if(softSleepWakeCase &&
				gMetrics.sceneSoftVelocityWakeIssued &&
				!gMetrics.sceneSoftWokeByVelocity && !sleeping)
			{
				gMetrics.sceneSoftWokeByVelocity = 1;
				gMetrics.sceneSoftVelocityWakeFrame =
					gMetrics.completedFrames;
			}
			if(softSleepWakeCase &&
				gMetrics.sceneSoftWokeByVelocity &&
				getSceneCpuVolumeCentroidY() -
					gSceneCpuVelocityWakeCentroidY > 0.02f)
				gMetrics.sceneSoftMovedAfterVelocityWake = 1;
			if(((softSleepWakeCase || softRigidWakeCase) &&
					gMetrics.sceneSoftVelocityStopIssued ||
				(bufferMutationCase &&
					gMetrics.sceneBufferResetIssued)) &&
				!gMetrics.sceneSoftFinalSlept && sleeping)
			{
				gMetrics.sceneSoftFinalSlept = 1;
				gMetrics.sceneSoftFinalSleepFrame =
					gMetrics.completedFrames;
			}
			if(softRigidWakeCase &&
				gMetrics.sceneSoftRigidWakeActorAdded &&
				!gMetrics.sceneSoftWokeByRigid && !sleeping)
			{
				gMetrics.sceneSoftWokeByRigid = 1;
				gMetrics.sceneSoftRigidWakeFrame =
					gMetrics.completedFrames;
			}
			if(softRigidWakeCase &&
				gMetrics.sceneSoftWokeByRigid &&
				PxAbs(
					getSceneCpuVolumeCentroidY() -
					gSceneCpuRigidWakeCentroidY) > 0.005f)
				gMetrics.sceneSoftMovedAfterRigidWake = 1;
			if(bufferMutationCase &&
				gMetrics.sceneBufferMutationIssued)
			{
				if(!gMetrics.sceneBufferMutationWoke &&
					!sleeping)
					gMetrics.sceneBufferMutationWoke = 1;
				const PxReal pinDeviation =
					(positions[0].getXYZ() -
						gSceneCpuBufferPinTarget).magnitude();
				const bool pinValid =
					PxIsFinite(pinDeviation) &&
					pinDeviation <= 1.0e-4f &&
					PxAbs(positions[0].w) <= 1.0e-7f;
				if(!gMetrics.sceneBufferMutationApplied &&
					pinValid)
				{
					gMetrics.sceneBufferMutationApplied = 1;
					gMetrics.sceneBufferPinHeld = 1;
				}
				else if(gMetrics.sceneBufferDriveIssued &&
					!gMetrics.sceneBufferInvMassRestored &&
					!pinValid)
					gMetrics.sceneBufferPinHeld = 0;
				if(gMetrics.sceneBufferDriveIssued &&
					!gMetrics.sceneBufferInvMassRestored &&
					(positions[1].getXYZ() -
						gSceneCpuBufferDynamicBaseline).
							magnitude() > 1.0e-5f)
					gMetrics.sceneBufferDynamicMoved = 1;
				if(gMetrics.sceneBufferInvMassRestored &&
					!gMetrics.sceneBufferResetIssued &&
					PxAbs(
						positions[0].w -
						gSceneCpuBufferOriginalInvMass) <=
							1.0e-7f &&
					(positions[0].getXYZ() -
						gSceneCpuBufferRestoredBaseline).
							magnitude() > 1.0e-5f)
					gMetrics.sceneBufferRestoredMoved = 1;
			}
		}
		if(softSoftWakeCase && gSceneCpuSecondVolume)
		{
			const bool targetSleeping =
				gSceneCpuVolume->isSleeping();
			const bool driverSleeping =
				gSceneCpuSecondVolume->isSleeping();
			if(!gMetrics.sceneSoftSoftDriveIssued &&
				targetSleeping && driverSleeping)
				gMetrics.sceneSoftSoftBothSlept = 1;
			if(gMetrics.sceneSoftSoftDriveIssued)
			{
				if(!driverSleeping)
					gMetrics.sceneSoftSoftDriverWoke = 1;
				if(!gMetrics.sceneSoftSoftTargetWoke &&
					!targetSleeping)
				{
					gMetrics.sceneSoftSoftTargetWoke = 1;
					gMetrics.sceneSoftSoftTargetWakeFrame =
						gMetrics.completedFrames;
				}
				if(gMetrics.sceneSoftSoftTargetWoke &&
					(getSceneCpuVolumeCentroid(
						gSceneCpuVolume) -
						gSceneCpuSoftSoftTargetBaseline).
							magnitude() > 0.002f)
					gMetrics.sceneSoftSoftTargetMoved = 1;
				if(gMetrics.sceneSoftSoftResetIssued &&
					targetSleeping && driverSleeping)
					gMetrics.sceneSoftSoftBothFinalSlept = 1;
			}
		}
		if(mixedSleepIslandCase && gSceneCpuSecondVolume)
		{
			const bool firstSleeping =
				gSceneCpuVolume->isSleeping();
			const bool secondSleeping =
				gSceneCpuSecondVolume->isSleeping();
			if(!gMetrics.sceneMixedFirstSlept && firstSleeping)
			{
				gMetrics.sceneMixedFirstSlept = 1;
				gMetrics.sceneMixedFirstSleepFrame =
					gMetrics.completedFrames;
				gSceneCpuFirstSleepCentroidY =
					getSceneCpuVolumeCentroidY();
			}
			if(gMetrics.sceneMixedFirstSlept && !secondSleeping)
				gMetrics.sceneMixedSecondStayedAwake = 1;
			if(gMetrics.sceneMixedFirstSlept &&
				gMetrics.completedFrames >=
					gMetrics.sceneMixedFirstSleepFrame + 2 &&
				PxAbs(
					getSceneCpuVolumeCentroidY() -
					gSceneCpuFirstSleepCentroidY) <= 1.0e-6f)
				gMetrics.sceneMixedFirstStable = 1;
			if(getSceneCpuVolumeCentroidY(
					gSceneCpuSecondVolume) -
				gSceneCpuSecondVolumeInitialCentroidY > 0.05f)
				gMetrics.sceneMixedSecondMoved = 1;
		}
		const PxReal dynamicHalfX =
			multiDynamicBoxCase ? 0.45f : 4.0f;
		PxTransform dynamicBoxPose(PxIdentity);
		PxTransform secondDynamicBoxPose(PxIdentity);
		PxTransform kinematicBoxPose(PxIdentity);
		if(kinematicBoxCase && gSceneCpuDynamicActor &&
			gSceneCpuDynamicActor->getScene() == gScene)
		{
			kinematicBoxPose =
				gSceneCpuDynamicActor->getGlobalPose();
			if(!kinematicBoxPose.isValid())
				gMetrics.nonFiniteParticleSamples++;
			else
			{
				gMetrics.sceneKinematicFinalY =
					kinematicBoxPose.p.y;
				if(gMetrics.sceneKinematicTargetIssued)
				{
					const PxReal poseError = PxAbs(
						kinematicBoxPose.p.y -
							gSceneCpuKinematicCommandY);
					gMetrics.sceneKinematicMaxPoseError =
						PxMax(
							gMetrics.
								sceneKinematicMaxPoseError,
							poseError);
					if(gSceneCpuKinematicCommandY >= 4.10f &&
						poseError <= 1.0e-4f)
						gMetrics.sceneKinematicTargetReached =
							1;
					if(!gSceneCpuVolume->isSleeping())
						gMetrics.sceneKinematicSoftWoke = 1;
					gMetrics.sceneKinematicSoftDisplacement =
						PxMax(
							gMetrics.
								sceneKinematicSoftDisplacement,
							PxAbs(
								getSceneCpuVolumeCentroidY() -
								gSceneCpuKinematicSoftBaselineY));
					if(gMetrics.
						sceneKinematicSoftDisplacement >
						0.02f)
						gMetrics.sceneKinematicSoftMoved = 1;
				}
			}
		}
		if(dynamicActorInScene)
		{
			if(gMetrics.sceneDynamicReaddedSleeping &&
				!gMetrics.sceneDynamicRewokeBySoft &&
				!gSceneCpuDynamicActor->isSleeping())
			{
				gMetrics.sceneDynamicRewokeBySoft = 1;
				gMetrics.sceneDynamicSecondWakeFrame =
					gMetrics.completedFrames;
			}
			else if(gMetrics.sceneDynamicInitiallySleeping &&
				!gMetrics.sceneDynamicWokeBySoft &&
				!gSceneCpuDynamicActor->isSleeping())
			{
				gMetrics.sceneDynamicWokeBySoft = 1;
				gMetrics.sceneDynamicFirstWakeFrame =
					gMetrics.completedFrames;
			}
			dynamicBoxPose =
				gSceneCpuDynamicActor->getGlobalPose();
			const PxVec3 dynamicVelocity =
				gSceneCpuDynamicActor->getLinearVelocity();
			if(!dynamicBoxPose.isValid() ||
				!dynamicVelocity.isFinite())
			{
				gMetrics.nonFiniteParticleSamples++;
			}
			else
			{
				gMetrics.sceneDynamicMinY = PxMin(
					gMetrics.sceneDynamicMinY,
					dynamicBoxPose.p.y);
				gMetrics.sceneDynamicFinalY =
					dynamicBoxPose.p.y;
				gMetrics.sceneDynamicMaxDrop = PxMax(
					gMetrics.sceneDynamicMaxDrop,
					gSceneCpuDynamicInitialY -
						dynamicBoxPose.p.y);
				gMetrics.sceneDynamicMaxDownSpeed = PxMax(
					gMetrics.sceneDynamicMaxDownSpeed,
					PxMax(-dynamicVelocity.y, 0.0f));
			}
		}
		if(secondDynamicActorInScene)
		{
			if(gMetrics.sceneSecondDynamicInitiallySleeping &&
				!gMetrics.sceneSecondDynamicWokeBySoft &&
				!gSceneCpuSecondDynamicActor->isSleeping())
			{
				gMetrics.sceneSecondDynamicWokeBySoft = 1;
				gMetrics.sceneSecondDynamicFirstWakeFrame =
					gMetrics.completedFrames;
			}
			secondDynamicBoxPose =
				gSceneCpuSecondDynamicActor->getGlobalPose();
			const PxVec3 dynamicVelocity =
				gSceneCpuSecondDynamicActor->getLinearVelocity();
			if(!secondDynamicBoxPose.isValid() ||
				!dynamicVelocity.isFinite())
			{
				gMetrics.nonFiniteParticleSamples++;
			}
			else
			{
				gMetrics.sceneSecondDynamicMinY = PxMin(
					gMetrics.sceneSecondDynamicMinY,
					secondDynamicBoxPose.p.y);
				gMetrics.sceneSecondDynamicFinalY =
					secondDynamicBoxPose.p.y;
				gMetrics.sceneSecondDynamicMaxDrop = PxMax(
					gMetrics.sceneSecondDynamicMaxDrop,
					gSceneCpuSecondDynamicInitialY -
						secondDynamicBoxPose.p.y);
				gMetrics.sceneSecondDynamicMaxDownSpeed = PxMax(
					gMetrics.sceneSecondDynamicMaxDownSpeed,
					PxMax(-dynamicVelocity.y, 0.0f));
			}
		}
		for(PxU32 i = 0; i < vertexCount; i++)
		{
			const PxVec3 position = positions[i].getXYZ();
			const PxVec3 velocity = velocities[i].getXYZ();
			if(!position.isFinite() || !velocity.isFinite() ||
				!PxIsFinite(positions[i].w))
			{
				gMetrics.nonFiniteParticleSamples++;
				continue;
			}
			gMetrics.minY = PxMin(gMetrics.minY, position.y);
			gMetrics.maxY = PxMax(gMetrics.maxY, position.y);
			frameMinY = PxMin(frameMinY, position.y);
			gMetrics.maxParticleSpeed = PxMax(
				gMetrics.maxParticleSpeed, velocity.magnitude());
			if(gHeadlessOptions.caseName == "scene-volume-ground" &&
				position.y <= 0.08f)
				nearGroundParticles++;
			if((gHeadlessOptions.caseName == "scene-volume-static-box" ||
				gHeadlessOptions.caseName == "scene-volume-static-churn") &&
				position.y <= 1.08f &&
				PxAbs(position.x) <= 20.08f &&
				PxAbs(position.z) <= 20.08f)
				nearRigidParticles++;
			if(kinematicBoxCase && kinematicBoxPose.isValid())
			{
				const PxVec3 localPosition =
					kinematicBoxPose.transformInv(position);
				PxVec3 capsuleRadial = localPosition;
				capsuleRadial.x -= PxClamp(
					capsuleRadial.x, -0.3f, 0.3f);
				const bool nearKinematicShape =
					kinematicCapsuleCase
						? capsuleRadial.magnitude() <= 0.58f
						: kinematicTriangleMeshCase
						? PxAbs(localPosition.x) <= 4.08f &&
							PxAbs(localPosition.z) <= 4.08f &&
							PxAbs(localPosition.y) <= 0.12f
						: kinematicHeightFieldCase
						? localPosition.x >= -0.08f &&
							localPosition.x <= 8.08f &&
							localPosition.z >= -0.08f &&
							localPosition.z <= 8.08f &&
							PxAbs(localPosition.y) <= 0.12f
						: kinematicConvexCase &&
							gSceneCpuRigidConvexMesh
						? PxGeometryQuery::pointDistance(
							position,
							PxConvexMeshGeometry(
								gSceneCpuRigidConvexMesh),
							kinematicBoxPose) <=
								0.08f * 0.08f
						: kinematicSphereCase
						? localPosition.magnitude() <= 0.58f
						: PxAbs(localPosition.x) <= 4.08f &&
							PxAbs(localPosition.z) <= 4.08f &&
							PxAbs(localPosition.y) <= 0.35f;
				if(nearKinematicShape)
					gMetrics.sceneKinematicContactObserved = 1;
			}
			if(dynamicActorInScene && dynamicBoxPose.isValid())
			{
				const PxVec3 localPosition =
					dynamicBoxPose.transformInv(position);
				if(dynamicConvexCase &&
					gSceneCpuRigidConvexMesh)
				{
					const PxReal squaredDistance =
						PxGeometryQuery::pointDistance(
							position,
							PxConvexMeshGeometry(
								gSceneCpuRigidConvexMesh),
							dynamicBoxPose);
					if(squaredDistance >= 0.0f &&
						PxIsFinite(squaredDistance))
					{
						const PxReal separation =
							PxSqrt(squaredDistance);
						frameDynamicSurfaceSeparation =
							PxMin(
								frameDynamicSurfaceSeparation,
								separation);
						gMetrics.
							minDynamicSurfaceSeparation =
							PxMin(
								gMetrics.
									minDynamicSurfaceSeparation,
								separation);
						if(separation <= 0.08f)
							nearRigidParticles++;
					}
				}
				else if(dynamicSphereCase)
				{
					PxVec3 radial = localPosition;
					if(dynamicCapsuleCase)
						radial.x -= PxClamp(
							radial.x, -0.3f, 0.3f);
					const PxReal radius = radial.magnitude();
					const PxReal separation = radius - 0.8f;
					frameDynamicSurfaceSeparation = PxMin(
						frameDynamicSurfaceSeparation,
						separation);
					gMetrics.minDynamicSurfaceSeparation = PxMin(
						gMetrics.minDynamicSurfaceSeparation,
						separation);
					if(radius >= 0.65f && radius <= 0.88f)
						nearRigidParticles++;
				}
				else if(
					PxAbs(localPosition.x) <= dynamicHalfX + 0.08f &&
					PxAbs(localPosition.z) <= 4.08f)
				{
					const PxReal separation =
						localPosition.y - 0.25f;
					frameDynamicSurfaceSeparation = PxMin(
						frameDynamicSurfaceSeparation,
						separation);
					gMetrics.minDynamicSurfaceSeparation = PxMin(
						gMetrics.minDynamicSurfaceSeparation,
						separation);
				}
				if(localPosition.y >= 0.10f &&
					localPosition.y <= 0.33f &&
					PxAbs(localPosition.x) <= dynamicHalfX + 0.08f &&
					PxAbs(localPosition.z) <= 4.08f)
					nearRigidParticles++;
			}
			if(secondDynamicActorInScene &&
				secondDynamicBoxPose.isValid())
			{
				const PxVec3 localPosition =
					secondDynamicBoxPose.transformInv(position);
				if(PxAbs(localPosition.x) <= dynamicHalfX + 0.08f &&
					PxAbs(localPosition.z) <= 4.08f)
				{
					const PxReal separation =
						localPosition.y - 0.25f;
					frameDynamicSurfaceSeparation = PxMin(
						frameDynamicSurfaceSeparation,
						separation);
					gMetrics.minDynamicSurfaceSeparation = PxMin(
						gMetrics.minDynamicSurfaceSeparation,
						separation);
				}
				if(localPosition.y >= 0.10f &&
					localPosition.y <= 0.33f &&
					PxAbs(localPosition.x) <= dynamicHalfX + 0.08f &&
					PxAbs(localPosition.z) <= 4.08f)
					nearRigidParticles++;
			}
		}
		if(groundEmbeddedTetProbeCase)
		{
			const PxTetrahedronMesh* const collisionMesh =
				gSceneCpuVolume->getCollisionMesh();
			const PxVec4* const collisionPositions =
				gSceneCpuVolume->getPositionInvMassBufferH();
			if(!collisionMesh || !collisionPositions ||
				collisionMesh->getNbVertices() != 4)
				return false;
			for(PxU32 vertexIndex = 0;
				vertexIndex < collisionMesh->getNbVertices(); ++vertexIndex)
			{
				const PxVec3 collisionPosition =
					collisionPositions[vertexIndex].getXYZ();
				if(!collisionPosition.isFinite())
				{
					gMetrics.nonFiniteParticleSamples++;
					continue;
				}
				if(collisionPosition.y <= 0.08f)
					nearGroundParticles++;
			}
		}
		if(partialElementFilterCase)
		{
			const PxVec4* collisionPositions =
				gSceneCpuVolume->getPositionInvMassBufferH();
			const PxTetrahedronMesh* collisionMesh =
				gSceneCpuVolume->getCollisionMesh();
			const PxVec3* collisionRestVertices =
				collisionMesh ? collisionMesh->getVertices() : NULL;
			const PxU32 collisionVertexCount =
				collisionMesh ? collisionMesh->getNbVertices() : 0;
			if(!collisionPositions || !collisionRestVertices ||
				collisionVertexCount == 0)
				return false;
			for(PxU32 collisionVertex = 0;
				collisionVertex < collisionVertexCount; ++collisionVertex)
			{
				const PxVec3 collisionPosition =
					collisionPositions[collisionVertex].getXYZ();
				if(!collisionPosition.isFinite())
				{
					gMetrics.nonFiniteParticleSamples++;
					continue;
				}
				frameCollisionMinY = PxMin(
					frameCollisionMinY, collisionPosition.y);
				if(gMetrics.sceneElementFilterReleased)
					continue;
				const bool positiveX =
					collisionRestVertices[collisionVertex].x > 0.0f;
				if(positiveX ==
					gSceneCpuPartialFilterSelectedPositiveX)
					gMetrics.sceneElementFilterMinY = PxMin(
						gMetrics.sceneElementFilterMinY,
						collisionPosition.y);
				else
				{
					gMetrics.scenePartialFilterUnfilteredMinY = PxMin(
						gMetrics.scenePartialFilterUnfilteredMinY,
						collisionPosition.y);
					if(collisionPosition.y >=
							-SCENE_CPU_ELEMENT_FILTER_SURFACE_TOLERANCE &&
						collisionPosition.y <=
							SCENE_CPU_ELEMENT_FILTER_CONTACT_OFFSET_LIMIT)
						gMetrics.scenePartialFilterUnfilteredContactHeld = 1;
				}
			}
		}
		const PxReal elementFilterFrameMinY =
			partialElementFilterCase ? frameCollisionMinY : frameMinY;
		if(elementFilterCase && elementFilterFrameMinY != FLT_MAX)
		{
			if(!gMetrics.sceneElementFilterReleased)
			{
				if(!partialElementFilterCase)
					gMetrics.sceneElementFilterMinY = PxMin(
						gMetrics.sceneElementFilterMinY,
						frameMinY);
				if(gMetrics.sceneElementFilterMinY <=
					-SCENE_CPU_ELEMENT_FILTER_MIN_SUPPRESSED_DEPTH)
					gMetrics.sceneElementFilterSuppressedContact =
						1;
			}
			else
				gMetrics.sceneElementFilterFinalMinY =
					elementFilterFrameMinY;
		}
		if(twoSoftVolumeCase && gSceneCpuSecondVolume)
		{
			PxVec4* secondPositions =
				gSceneCpuSecondVolume->
					getSimPositionInvMassBufferH();
			PxVec4* secondVelocities =
				gSceneCpuSecondVolume->
					getSimVelocityBufferH();
			const PxU32 secondVertexCount =
				gSceneCpuSecondVolume->getSimulationMesh()->
					getNbVertices();
			if(!secondPositions || !secondVelocities)
				return false;
			for(PxU32 i = 0; i < secondVertexCount; i++)
			{
				const PxVec3 position =
					secondPositions[i].getXYZ();
				const PxVec3 velocity =
					secondVelocities[i].getXYZ();
				if(!position.isFinite() ||
					!velocity.isFinite() ||
					!PxIsFinite(secondPositions[i].w))
				{
					gMetrics.nonFiniteParticleSamples++;
					continue;
				}
				gMetrics.minY =
					PxMin(gMetrics.minY, position.y);
				gMetrics.maxY =
					PxMax(gMetrics.maxY, position.y);
				gMetrics.maxParticleSpeed = PxMax(
					gMetrics.maxParticleSpeed,
					velocity.magnitude());
				if(secondDynamicActorInScene &&
					secondDynamicBoxPose.isValid())
				{
					const PxVec3 localPosition =
						secondDynamicBoxPose.transformInv(
							position);
					if(PxAbs(localPosition.x) <=
							dynamicHalfX + 0.08f &&
						PxAbs(localPosition.z) <= 4.08f)
					{
						const PxReal separation =
							localPosition.y - 0.25f;
						frameDynamicSurfaceSeparation =
							PxMin(
								frameDynamicSurfaceSeparation,
								separation);
						gMetrics.minDynamicSurfaceSeparation =
							PxMin(
								gMetrics.
									minDynamicSurfaceSeparation,
								separation);
					}
					if(localPosition.y >= 0.10f &&
						localPosition.y <= 0.33f &&
						PxAbs(localPosition.x) <=
							dynamicHalfX + 0.08f &&
						PxAbs(localPosition.z) <= 4.08f)
						nearRigidParticles++;
				}
			}
			const PxReal secondCentroidY =
				getSceneCpuVolumeCentroidY(
					gSceneCpuSecondVolume);
			gMetrics.sceneSecondVolumeFinalCentroidY =
				secondCentroidY;
			gMetrics.sceneSecondVolumeMaxCentroidDrop = PxMax(
				gMetrics.sceneSecondVolumeMaxCentroidDrop,
				gSceneCpuSecondVolumeInitialCentroidY -
					secondCentroidY);
			const PxBounds3 secondBounds =
				gSceneCpuSecondVolume->getWorldBounds();
			if(secondBounds.isValid() &&
				secondBounds.minimum.isFinite() &&
				secondBounds.maximum.isFinite())
				gMetrics.sceneSecondVolumeBoundsFinite = 1;
		}
		if(dynamicActorInScene &&
			PxIsFinite(frameDynamicSurfaceSeparation) &&
			frameDynamicSurfaceSeparation != FLT_MAX)
		{
			gMetrics.finalDynamicSurfaceSeparation =
				frameDynamicSurfaceSeparation;
		}
		// The predictive OGC edge is the first contact-admission event. Once it
		// wakes a sleeping target, offset-shell forces may move the rigid before
		// any sampled soft vertex reaches the narrower visual proximity band;
		// that is contact response, not gravity-free pre-contact drift.
		if(dynamicActorInScene && !gMetrics.sceneDynamicWokeBySoft &&
			gMetrics.rigidContactFrames == 0 &&
			nearRigidParticles == 0 && dynamicBoxPose.isValid())
		{
			gMetrics.sceneDynamicPreContactMaxDrop = PxMax(
				gMetrics.sceneDynamicPreContactMaxDrop,
				PxAbs(gSceneCpuDynamicInitialY -
					dynamicBoxPose.p.y));
			if(secondDynamicActorInScene &&
				secondDynamicBoxPose.isValid())
			{
				gMetrics.sceneSecondDynamicPreContactMaxDrop = PxMax(
					gMetrics.sceneSecondDynamicPreContactMaxDrop,
					PxAbs(gSceneCpuSecondDynamicInitialY -
						secondDynamicBoxPose.p.y));
			}
		}
		if(nearGroundParticles > 0)
		{
			gMetrics.groundContactFrames++;
			gMetrics.maxGroundContacts = PxMax(
				gMetrics.maxGroundContacts, nearGroundParticles);
		}
		if(nearRigidParticles > 0)
		{
			gMetrics.rigidContactFrames++;
			gMetrics.maxRigidContacts = PxMax(
				gMetrics.maxRigidContacts, nearRigidParticles);
		}
		const PxReal centroidY = getSceneCpuVolumeCentroidY();
		gMetrics.maxCentroidDrop = PxMax(
			gMetrics.maxCentroidDrop,
			gSceneCpuVolumeInitialCentroidY - centroidY);
		const PxBounds3 bounds = gSceneCpuVolume->getWorldBounds();
		if(bounds.isValid() &&
			bounds.minimum.isFinite() && bounds.maximum.isFinite())
			gMetrics.sceneBoundsFinite = 1;

		if(profileFrame)
		{
			PxSimulationStatistics sceneStatistics;
			gScene->getSimulationStatistics(sceneStatistics);
			accumulateScenePerformanceStatistics(sceneStatistics);
			gPerformance.profiledFrames++;
			gPerformance.sceneMs += sceneMs;
			gPerformance.stepSamplesMs.pushBack(
				PxReal(frameTimer.getElapsedSeconds() * 1000.0));
		}
		return true;
	}

	PxTime stageTimer;
	gFrameCollisionStats = AvbdSoftCollisionStats();
	gSoftWorkspace.contact.beginStep();

	// Initial contact detection: ground + soft-soft OGC + rigid-soft SDF.
	AvbdSoftCollisionStats initialCollisionStats;
	avbdDetectAllOGCContacts(
		gParticles.begin(), gParticles.size(),
		gSoftBodies.begin(), gSoftBodies.size(),
		gRigidBoxes.begin(), gRigidBoxes.size(),
		NULL, 0,
		gContacts, gOGCParams, 0.0f, &initialCollisionStats,
		&gSoftWorkspace.contact);
	gFrameCollisionStats.accumulate(initialCollisionStats);
	const PxF64 initialContactMs =
		stageTimer.getElapsedSeconds() * 1000.0;
	const PxU64 initialContactWorkspaceGrowthEvents =
		gSoftWorkspace.contact.growthEvents;
	const PxU64 initialContactWorkspaceGrowthBytes =
		gSoftWorkspace.contact.growthBytes;
	const PxU64 initialContactOutputGrowthEvents =
		gSoftWorkspace.contact.outputGrowthEvents;
	const PxU64 initialContactOutputGrowthBytes =
		gSoftWorkspace.contact.outputGrowthBytes;
	recordContactMetrics();

	// 8 outer iterations with contact re-detection between each.
	// Contacts are re-detected via callback so surface-point anchors
	// track the deforming geometry instead of going stale.
	AvbdSoftBodyStepStats stepStats;
	PxTime solverTimer;
	const PxVec3 gravity = isComponentNoContactCase(
		gHeadlessOptions.caseName)
		? PxVec3(0.0f) : PxVec3(0.0f, -9.81f, 0.0f);
	avbdStepSoftBodies(
		gParticles.begin(), gParticles.size(),
		gSoftBodies.begin(), gSoftBodies.size(),
		gContacts.begin(), gContacts.size(),
		dt, gravity, 8, 20, 1000.0f,
		redetectContacts, &gContacts, NULL, 0.92f, &stepStats,
		&gSoftWorkspace);
	// avbdStepSoftBodies starts its own solver-stage counters.  Preserve the
	// contact-prep growth observed before entering the solver and combine it
	// with the outer-loop redetection growth reported by the solver.
	stepStats.contactWorkspaceGrowthEvents +=
		initialContactWorkspaceGrowthEvents;
	stepStats.contactWorkspaceGrowthBytes +=
		initialContactWorkspaceGrowthBytes;
	stepStats.contactOutputGrowthEvents +=
		initialContactOutputGrowthEvents;
	stepStats.contactOutputGrowthBytes +=
		initialContactOutputGrowthBytes;
	const PxF64 solverMs = solverTimer.getElapsedSeconds() * 1000.0;

	PxTime sceneTimer;
	gScene->simulate(dt);
	if(!gScene->fetchResults(true))
	{
		gMetrics.fetchFailures++;
		return false;
	}
	const PxF64 sceneMs = sceneTimer.getElapsedSeconds() * 1000.0;

	PxTime metricsTimer;
	gMetrics.completedFrames++;
	recordStateMetrics();
	sendSoftBodiesToPvd();
	const PxF64 metricsMs = metricsTimer.getElapsedSeconds() * 1000.0;
	if(profileFrame)
	{
		gPerformance.profiledFrames++;
		gPerformance.initialContactMs += initialContactMs;
		gPerformance.solverMs += solverMs;
		gPerformance.sceneMs += sceneMs;
		gPerformance.metricsMs += metricsMs;
		accumulateStepStats(gPerformance.solverStages, stepStats);
		gPerformance.collision.accumulate(gFrameCollisionStats);
		gPerformance.collisionRigidParticleTests +=
			gFrameCollisionStats.rigidParticleBoxTests +
			gFrameCollisionStats.rigidParticleSphereTests +
			gFrameCollisionStats.rigidParticleCapsuleTests +
			gFrameCollisionStats.rigidParticleConvexTests +
			gFrameCollisionStats.rigidParticleTriangleSurfaceTests;
		gPerformance.stepSamplesMs.pushBack(
			PxReal(frameTimer.getElapsedSeconds() * 1000.0));
	}
	return true;
}

void stepPhysics(bool /*interactive*/)
{
	stepPhysicsInternal(1.0f / 60.0f);
}

static void finalizeMetrics()
{
	gMetrics.finalInsideParticles = 0;
	gMetrics.finalMinY = FLT_MAX;
	gMetrics.finalMaxY = -FLT_MAX;
	gMetrics.finalMaxParticleSpeed = 0.0f;
	if(isSceneCpuVolumeCase(gHeadlessOptions.caseName) &&
		gSceneCpuVolume)
	{
		auto collectVolume = [](PxDeformableVolume* volume)
		{
			if(!volume || !volume->getSimulationMesh())
				return;
			const PxVec4* positions =
				volume->getSimPositionInvMassBufferH();
			const PxVec4* velocities =
				volume->getSimVelocityBufferH();
			const PxU32 vertexCount =
				volume->getSimulationMesh()->getNbVertices();
			for(PxU32 i = 0; i < vertexCount; i++)
			{
				const PxVec3 position =
					positions[i].getXYZ();
				const PxVec3 velocity =
					velocities[i].getXYZ();
				if(!position.isFinite() ||
					!velocity.isFinite())
					continue;
				gMetrics.finalMinY =
					PxMin(gMetrics.finalMinY, position.y);
				gMetrics.finalMaxY =
					PxMax(gMetrics.finalMaxY, position.y);
			gMetrics.finalMaxParticleSpeed = PxMax(
					gMetrics.finalMaxParticleSpeed,
					velocity.magnitude());
			}
		};
		collectVolume(gSceneCpuVolume);
		collectVolume(gSceneCpuSecondVolume);
		for(PxU32 volumeId = 0;
			volumeId < gSceneCpuTaskGraphExtraVolumes.size(); ++volumeId)
			collectVolume(gSceneCpuTaskGraphExtraVolumes[volumeId]);
		if(gSceneCpuSecondVolume)
			gMetrics.sceneSecondVolumeFinalCentroidY =
				getSceneCpuVolumeCentroidY(
					gSceneCpuSecondVolume);
		return;
	}

	for(PxU32 particleId = 0; particleId < gParticles.size(); ++particleId)
	{
		const AvbdSoftParticle& particle = gParticles[particleId];
		if(!particle.position.isFinite() || !particle.velocity.isFinite())
			continue;
		gMetrics.finalMinY =
			PxMin(gMetrics.finalMinY, particle.position.y);
		gMetrics.finalMaxY =
			PxMax(gMetrics.finalMaxY, particle.position.y);
		gMetrics.finalMaxParticleSpeed = PxMax(
			gMetrics.finalMaxParticleSpeed, particle.velocity.magnitude());
	}

	for(PxU32 bodyA = 0; bodyA < gSoftBodies.size(); ++bodyA)
	{
		const AvbdSoftBody& source = gSoftBodies[bodyA];
		for(PxU32 bodyB = 0; bodyB < gSoftBodies.size(); ++bodyB)
		{
			if(bodyA == bodyB)
				continue;
			const AvbdSoftBody& target = gSoftBodies[bodyB];
			for(PxU32 localId = 0;
				localId < source.compiled.particleCount; ++localId)
			{
				const PxU32 particleId = source.compiled.particleStart + localId;
				if(avbdIsPointInsideTetMesh(
					gParticles[particleId].position,
					target.compiled.surfaceTriangles, gParticles.begin()))
				{
					gMetrics.finalInsideParticles++;
				}
			}
		}
	}
}

void cleanupPhysics(bool /*interactive*/)
{
	// PxArray uses the PhysX foundation allocator. Interactive shutdown does
	// not pass through finalizePerformanceMetrics(), so release the sample
	// storage here while the foundation broadcast allocator is still alive.
	// Keep the scalar visual metrics alive for the post-cleanup report, but
	// release their PxArray-backed surface caches before the foundation goes
	// away. The jaw pointers are no longer valid after the actor release below.
	gSceneCpuVisualInteractionMonitor.releaseSources();
	gSceneCpuOgcSandwichMonitor.releaseSources();
	gSceneCpuSoftSoftTorqueMonitor.releaseSources();
	if(gSceneCpuElementFilter)
	{
		gSceneCpuElementFilter->release();
		gSceneCpuElementFilter = NULL;
	}
	if(gSceneCpuWorldAttachment)
	{
		gSceneCpuWorldAttachment->release();
		gSceneCpuWorldAttachment = NULL;
	}
	if(gSceneCpuRigidAttachment)
	{
		gSceneCpuRigidAttachment->release();
		gSceneCpuRigidAttachment = NULL;
	}
	gSceneCpuVolumeHealthMonitor.reset();
	for(PxU32 volumeId = 0;
		volumeId < gSceneCpuTaskGraphExtraVolumes.size(); ++volumeId)
	{
		PxDeformableVolume* extraVolume =
			gSceneCpuTaskGraphExtraVolumes[volumeId];
		if(!extraVolume)
			continue;
		PxScene* volumeScene = extraVolume->getScene();
		if(volumeScene)
			volumeScene->removeActor(*extraVolume);
		extraVolume->release();
	}
	gSceneCpuTaskGraphExtraVolumes.reset();
	for(PxU32 meshIndex = 0;
		meshIndex < gSceneCpuTaskGraphExtraMeshes.size(); ++meshIndex)
		PX_RELEASE(gSceneCpuTaskGraphExtraMeshes[meshIndex]);
	gSceneCpuTaskGraphExtraMeshes.reset();
	if(gSceneCpuSecondVolume)
	{
		PxScene* volumeScene =
			gSceneCpuSecondVolume->getScene();
		if(volumeScene)
		{
			volumeScene->removeActor(*gSceneCpuSecondVolume);
			if(gSceneCpuSecondVolume->getScene() == NULL)
				gMetrics.sceneSecondVolumeActorRemoved = 1;
		}
		gSceneCpuSecondVolume->release();
		gSceneCpuSecondVolume = NULL;
		gMetrics.sceneSecondVolumeActorReleased = 1;
	}
	if(gSceneCpuVolume)
	{
		PxScene* volumeScene = gSceneCpuVolume->getScene();
		if(volumeScene)
		{
			volumeScene->removeActor(*gSceneCpuVolume);
			if(gSceneCpuVolume->getScene() == NULL &&
				volumeScene->getNbDeformableVolumes() == 0)
				gMetrics.sceneActorRemoved = 1;
		}
		gSceneCpuVolume->release();
		gSceneCpuVolume = NULL;
		gMetrics.sceneActorReleased = 1;
	}
	for(PxU32 meshIndex = 0;
		meshIndex < gSceneCpuVisualVolumeMeshes.size(); ++meshIndex)
		PX_RELEASE(gSceneCpuVisualVolumeMeshes[meshIndex]);
	gSceneCpuVisualVolumeMeshes.reset();
	gSceneCpuVisualRotationMonitor.releaseSource();
	gSceneCpuVisualPrimaryCubeRotationMonitor.releaseSource();
	gSceneCpuVisualSphereRotationMonitor.releaseSource();
	gSceneCpuSoftContactPhaseMonitor.releaseSource();
	PX_RELEASE(gSceneCpuVolumeMesh);
	PX_RELEASE(gSceneCpuVolumeMaterial);
	if(gSceneCpuAttachmentArticulation)
	{
		gSceneCpuAttachmentBody = NULL;
		gSceneCpuAttachmentLink = NULL;
		gSceneCpuAttachmentRoot = NULL;
		gSceneCpuAttachmentArticulation->release();
		gSceneCpuAttachmentArticulation = NULL;
	}
	if(gSceneCpuStaticActor)
	{
		if(gScene &&
			gSceneCpuStaticActor->getScene() == gScene)
			gScene->removeActor(*gSceneCpuStaticActor);
		gSceneCpuStaticActor->release();
		gSceneCpuStaticActor = NULL;
	}
	if(gSceneCpuDynamicActor)
	{
		if(gScene &&
			gSceneCpuDynamicActor->getScene() == gScene)
		{
			gScene->removeActor(*gSceneCpuDynamicActor);
			if(gSceneCpuDynamicActor->getScene() == NULL)
				gMetrics.sceneDynamicActorRemoved = 1;
		}
		gSceneCpuDynamicActor->release();
		gSceneCpuDynamicActor = NULL;
		gSceneCpuAttachmentBody = NULL;
		gMetrics.sceneDynamicActorReleased = 1;
	}
	if(gSceneCpuSecondDynamicActor)
	{
		if(gScene &&
			gSceneCpuSecondDynamicActor->getScene() == gScene)
		{
			gScene->removeActor(*gSceneCpuSecondDynamicActor);
			if(gSceneCpuSecondDynamicActor->getScene() == NULL)
				gMetrics.sceneSecondDynamicActorRemoved = 1;
		}
		gSceneCpuSecondDynamicActor->release();
		gSceneCpuSecondDynamicActor = NULL;
		gMetrics.sceneSecondDynamicActorReleased = 1;
	}
	PX_RELEASE(gSceneCpuRigidConvexMesh);
	PX_RELEASE(gSceneCpuRigidHeightField);
	PX_RELEASE(gSceneCpuRigidTriangleMesh);

	gPerformance.stepSamplesMs.reset();
	gSoftBodyRenderData.reset();
	gContacts.reset();
	gRigidBoxes.reset();
	gSoftBodies.reset();
	gParticles.reset();
	gInitialCentroids.reset();
	gSceneCpuVolumeKinematicTargets.reset();
	gSceneCpuVolumeKinematicInitial.reset();
	gSceneCpuSphereReverseSweptInitialPositions.reset();
	gSceneCpuDeformingReverseSweptFreeEndPositions.reset();
	gSceneCpuCapsuleRotationalSweptInitialPositions.reset();
	gVolumeSkinningBindings.reset();
	gVolumeSkinningTriangles.reset();
	gVolumeSkinningPositions.reset();
	gVolumeSkinningNormals.reset();
	gVolumeSkinningInitialPositions.reset();
	gVolumeAvbdSkinningRenderData =
		VolumeAvbdSkinningRenderData();
	gSoftWorkspace.reset();

	PX_RELEASE(gScene);
	if(gSceneCpuSecondScene)
	{
		PX_RELEASE(gSceneCpuSecondScene);
		gMetrics.sceneSecondSceneReleased = 1;
	}
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gMaterial);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	if (gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gScene && !gSceneCpuSecondScene &&
		!gDispatcher && !gMaterial && !gPhysics &&
		!gPvd && !gFoundation && !gSceneCpuVolume &&
		!gSceneCpuSecondVolume &&
		gSceneCpuTaskGraphExtraVolumes.empty() &&
		gSceneCpuTaskGraphExtraMeshes.empty() &&
		gSceneCpuVisualVolumeMeshes.empty() &&
		gSceneCpuVolumeHealthMonitor.empty() &&
		!gSceneCpuWorldAttachment &&
		!gSceneCpuRigidAttachment &&
		!gSceneCpuElementFilter &&
		!gSceneCpuVolumeMesh && !gSceneCpuVolumeMaterial &&
		!gSceneCpuStaticActor && !gSceneCpuDynamicActor &&
		!gSceneCpuSecondDynamicActor &&
		!gSceneCpuAttachmentArticulation &&
		!gSceneCpuAttachmentRoot &&
		!gSceneCpuAttachmentLink &&
		!gSceneCpuAttachmentBody ? 1u : 0u;

	printf("%s done.\n", AVBD_VOLUME_SNIPPET_NAME);
}

void keyPress(unsigned char /*key*/, const PxTransform& /*camera*/)
{
}

PxDeformableVolume* getPrimaryCpuAvbdVolume()
{
	return gSceneCpuVolume;
}

static bool isKnownCase(const std::string& caseName)
{
	return caseName == "volume-ground" ||
		caseName == "volume-performance-dense-no-contact" ||
		caseName == "volume-performance-many-small-no-contact" ||
		caseName == "volume-static-box" ||
	caseName == "soft-soft" ||
	caseName == "cone-ground" ||
	caseName == "scene-volume-visual-showcase" ||
	caseName == "scene-volume-ogc-sandwich" ||
	caseName == "scene-volume-sphere-long-roll" ||
	caseName == "scene-volume-sphere-soft-soft-glancing" ||
	caseName == "scene-volume-soft-soft-torque" ||
	caseName == "scene-volume-lifecycle" ||
		caseName == "scene-volume-corotational" ||
	isSceneCpuVolumeTaskGraphPureSoftCase(caseName) ||
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
		isSceneCpuVolumeTaskGraphWriteBackCase(caseName) ||
		caseName == "scene-volume-ground" ||
		caseName == "scene-volume-ground-embedded-tet-probe" ||
		caseName == "scene-volume-static-box" ||
		caseName == "scene-volume-static-churn" ||
		caseName == "scene-volume-dynamic-box" ||
		caseName == "scene-volume-true-boundary-dynamic-box" ||
		caseName == "scene-volume-dynamic-sphere" ||
		caseName == "scene-volume-dynamic-capsule" ||
		caseName == "scene-volume-dynamic-convex" ||
		caseName == "scene-volume-dynamic-churn" ||
		caseName == "scene-volume-multi-dynamic-box" ||
		caseName == "scene-volume-multi-soft-islands" ||
		caseName == "scene-volume-sleep-wake" ||
		caseName == "scene-volume-rigid-wake" ||
		caseName == "scene-volume-mixed-sleep-islands" ||
		caseName == "scene-volume-soft-churn" ||
		caseName == "scene-volume-buffer-mutation" ||
		caseName == "scene-volume-world-pin" ||
		caseName == "scene-volume-world-element-attachment" ||
		caseName == "scene-volume-rigid-attachment" ||
		caseName == "scene-volume-rigid-element-attachment" ||
		caseName == "scene-volume-static-attachment" ||
		caseName == "scene-volume-static-element-attachment" ||
		caseName == "scene-volume-kinematic-attachment" ||
		caseName == "scene-volume-kinematic-element-attachment" ||
		caseName == "scene-volume-articulation-attachment" ||
		caseName == "scene-volume-articulation-element-attachment" ||
		caseName == "scene-volume-element-filter" ||
		caseName == "scene-volume-partial-element-filter" ||
		isSceneCpuVolumeKinematicRigidCase(caseName) ||
		caseName == "scene-volume-full-kinematic-target" ||
		caseName == "scene-volume-partial-kinematic-target" ||
		caseName == "scene-volume-multi-scene-isolation" ||
		caseName == "scene-volume-soft-soft-wake" ||
		caseName == "scene-volume-volume-attachment" ||
		caseName == "scene-volume-skinning" ||
		caseName == "scene-volume-motion-controls" ||
		caseName ==
			"scene-volume-max-depenetration-velocity" ||
		caseName == "scene-volume-sphere-reverse-feature" ||
		caseName == "scene-volume-capsule-reverse-feature" ||
		caseName == "scene-volume-convex-reverse-feature" ||
		caseName ==
			"scene-volume-triangle-mesh-reverse-feature" ||
		caseName ==
			"scene-volume-heightfield-reverse-feature" ||
		isSceneCpuVolumeSpeculativeCcdCase(caseName) ||
		isSceneCpuVolumeSphereReverseSweptCcdCase(caseName) ||
		isSceneCpuVolumeRigidTriangleSteadyContactCase(caseName) ||
		caseName == "current-all";
}

static Headless::HeadlessResultContext buildHeadlessResultContext()
{
	Headless::HeadlessResultContext context;
	context.caseName = &gHeadlessOptions.caseName;
	context.solverName =
		Snippets::getSolverTypeName(gHeadlessOptions.solverType);
	context.frames = gHeadlessOptions.frames;
	context.dt = gHeadlessOptions.dt;
	context.dispatcherThreads = gHeadlessOptions.dispatcherThreads;
	context.parallelExecution =
		gHeadlessOptions.execution == Snippets::eHEADLESS_PARALLEL;
	context.sequentialExecution =
		gHeadlessOptions.execution == Snippets::eHEADLESS_SEQUENTIAL;
	context.metrics = &gMetrics;
	context.performance = &gPerformance;
	context.ogcSandwichMetrics =
		&gSceneCpuOgcSandwichMonitor.getMetrics();
	context.visualRotationMetrics = &gSceneCpuVisualRotationMetrics;
	context.visualPrimaryCubeRotationMetrics =
		&gSceneCpuVisualPrimaryCubeRotationMetrics;
	context.visualSphereRotationMetrics =
		&gSceneCpuVisualSphereRotationMetrics;
	context.visualInteractionMetrics =
		&gSceneCpuVisualInteractionMetrics;
	context.softContactPhaseMetrics = &gSceneCpuSoftContactPhaseMetrics;
	context.softSoftTorqueMetrics = &gSceneCpuSoftSoftTorqueMetrics;
	context.groundEmbeddedTetMetrics =
		&gSceneCpuGroundEmbeddedTetProbeMetrics;
	context.volumeHealthMonitor = &gSceneCpuVolumeHealthMonitor;
	context.volumeSkinningMetrics = &gVolumeSkinningMetrics;
	context.reverseFeatureMetrics = &gSphereReverseFeatureMetrics;
	context.reverseSweptMetrics = &gSphereReverseSweptMetrics;
	context.deformingReverseSweptMetrics =
		&gDeformingVolumeReverseSweptMetrics;
	context.rotationalSweepMetrics = &gCapsuleRotationalSweepMetrics;
	context.kinematicFiniteSweptMetrics = getKinematicFiniteSweptMetrics();
	context.dynamicFiniteSweptMetrics = getDynamicFiniteSweptMetrics();
	context.dynamicInitialY = gSceneCpuDynamicInitialY;
	context.secondDynamicInitialY = gSceneCpuSecondDynamicInitialY;
	context.visualSphereLongRunBounded =
		isSceneCpuVisualSphereLongRunBounded();
	context.sphereRollingKinematicsValid =
		isSceneCpuSphereRollingKinematicsValid();
	context.sphereLongRollRegressionPassed =
		isSceneCpuSphereLongRollRegressionPassed();
	context.sphereRollWindowBeginFrame =
		SCENE_CPU_SPHERE_ROLL_WINDOW_BEGIN_FRAME;
	context.sphereRollWindowEndFrame =
		SCENE_CPU_SPHERE_ROLL_WINDOW_END_FRAME;
	context.sphereRollCheckpointCount =
		SCENE_CPU_SPHERE_ROLL_CHECKPOINT_COUNT;
	context.sphereRollCheckpointInterval =
		SCENE_CPU_SPHERE_ROLL_CHECKPOINT_INTERVAL;
	context.ogcPressureDriveFrames =
		SCENE_CPU_OGC_SANDWICH_PRESSURE_DRIVE_FRAMES;
	context.softSoftTorqueMinAngularMomentum =
		SCENE_CPU_SOFT_SOFT_TORQUE_MIN_ANGULAR_MOMENTUM;
	context.softSoftTorqueMinAngularSpeed =
		SCENE_CPU_SOFT_SOFT_TORQUE_MIN_ANGULAR_SPEED;
	context.fatalErrors = gErrorCallback.getFatalCount();
	context.warningErrors = gErrorCallback.getWarningCount();
	return context;
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.caseName = AVBD_VOLUME_DEFAULT_CASE;
	defaults.frames = 600;
	defaults.dispatcherThreads = 2;
	std::string parseError;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, parseError))
	{
		printf("[AVBD_GATE_CONFIG_ERROR] %s\n", parseError.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	for(int argId = 1; argId < argc; ++argId)
	{
		if(!Snippets::isCommonHeadlessOption(argv[argId]))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] unknown option: %s\n",
				argv[argId]);
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
	}
	if(gHeadlessOptions.headless)
	{
		if(gHeadlessOptions.solverType != PxSolverType::eAVBD)
		{
			printf(
				"[AVBD_GATE_UNSUPPORTED] reason=component-is-avbd-only\n");
			return Snippets::eHEADLESS_UNSUPPORTED;
		}
		if(!isKnownCase(gHeadlessOptions.caseName))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] unknown case: %s\n",
				gHeadlessOptions.caseName.c_str());
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		const char* warmupEnvironment =
			std::getenv("PHYSX_AVBD_PROFILE_WARMUP");
		const char* rigidTriangleGridEnvironment =
			std::getenv("PHYSX_AVBD_RIGID_TRIANGLE_GRID_DIM");
		gProfileWarmupFrames = 0;
		gRigidTriangleMeshGridDimension = 1;
		if(warmupEnvironment && warmupEnvironment[0] &&
			(!Snippets::parseU32(
				warmupEnvironment, 0, 100000000u,
				gProfileWarmupFrames) ||
			 gProfileWarmupFrames >= gHeadlessOptions.frames))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] "
				"invalid PHYSX_AVBD_PROFILE_WARMUP\n");
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		if(rigidTriangleGridEnvironment &&
			rigidTriangleGridEnvironment[0] &&
			!Snippets::parseU32(
				rigidTriangleGridEnvironment, 1, 128,
				gRigidTriangleMeshGridDimension))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] "
				"invalid PHYSX_AVBD_RIGID_TRIANGLE_GRID_DIM\n");
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		if(isSceneCpuVolumeTaskGraphRigidTriangleSurfaceLargeCase(
			gHeadlessOptions.caseName) ||
			isSceneCpuVolumeTaskGraphRigidTriangleSurfaceThresholdCase(
			gHeadlessOptions.caseName))
		{
			// P5.19/P5.20 own a reproducible legacy two-triangle hierarchy and
			// must not inherit the optional P1 grid-size experiment from a caller.
			gRigidTriangleMeshGridDimension = 1;
		}
		if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] "
				"failed to apply execution environment\n");
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		Snippets::printHeadlessConfig(
			AVBD_VOLUME_SNIPPET_NAME, gHeadlessOptions);
		bool initialized = initPhysicsInternal(
			false, gHeadlessOptions.caseName);
		if(initialized)
		{
			for(PxU32 frame = 0;
				frame < gHeadlessOptions.frames; ++frame)
			{
				if(!stepPhysicsInternal(gHeadlessOptions.dt))
					break;
			}
			finalizeMetrics();
			finalizePerformanceMetrics(gPerformance);
		}
		cleanupPhysics(false);
		Headless::HeadlessResultContext resultContext =
			buildHeadlessResultContext();
		const bool passed =
			initialized &&
			Headless::validateHeadlessResult(resultContext);
		PerformanceReportConfig performanceConfig;
		performanceConfig.caseName = gHeadlessOptions.caseName.c_str();
		performanceConfig.executionName =
			Snippets::getExecutionName(gHeadlessOptions.execution);
		performanceConfig.dispatcherThreads =
			gHeadlessOptions.dispatcherThreads;
		printPerformanceResult(gPerformance, performanceConfig);
		Headless::printHeadlessResult(resultContext, passed);
		return passed ?
			Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
	}

#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	printf("%s: No render snippet, nothing to do.\n",
		AVBD_VOLUME_SNIPPET_NAME);
#endif

	return 0;
}
