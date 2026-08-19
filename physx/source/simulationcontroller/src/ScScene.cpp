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
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

#include "ScScene.h"
#include "BpBroadPhase.h"
#include "ScConstraintCore.h"
#include "ScArticulationJointCore.h"
#include "ScArticulationTendonCore.h"
#include "ScArticulationMimicJointCore.h"
#include "ScArticulationSim.h"
#include "ScArticulationTendonSim.h"
#include "ScArticulationMimicJointSim.h"
#include "ScDeformableSurfaceCore.h"
#include "ScDeformableVolumeCore.h"
#include "ScBodyCore.h"
#include "ScBodySim.h"
#include "ScStaticCore.h"
#include "ScShapeCore.h"
#include "ScTriggerInteraction.h"
#include "ScSimStats.h"
#include "PxsCCD.h"
#include "ScSimulationController.h"
#include "ScSqBoundsManager.h"
#include "ScArticulationCore.h"
#include "DyIslandManager.h"
#include "DyAvbdDynamics.h"
#include "DyAvbdGpuWaveBackend.h"
#include "DyAvbdSoftBodyComponent.h"
#include "DyDeformableSurface.h"
#include "DyDeformableVolume.h"
#include "foundation/PxHashMap.h"
#include "foundation/PxTime.h"
#include "geometry/PxHeightField.h"
#include "geometry/PxHeightFieldGeometry.h"
#include "geometry/PxMeshQuery.h"
#include "geometry/PxTriangle.h"
#include "geometry/PxTriangleMesh.h"
#include "GuTetrahedronMesh.h"
#include "PxsDeformableSurfaceMaterialCore.h"
#include "PxsMaterialCore.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <atomic>

#if defined(__APPLE__) && defined(__POWERPC__)
	#include <ppc_intrinsics.h>
#endif

#if PX_SUPPORT_GPU_PHYSX
	#include "PxvGlobals.h"
	#include "PxPhysXGpu.h"
	#include "PxsHeapMemoryAllocator.h"
	#include "cudamanager/PxCudaContextManager.h"
	#include "cudamanager/PxCudaContext.h"
#endif

#include "PxsMemoryManager.h"

#include "ScShapeInteraction.h"

#if PX_SUPPORT_GPU_PHYSX
	#include "PxDeformableSurface.h"
	#include "ScDeformableSurfaceSim.h"
	#include "PxDeformableVolume.h"
	#include "ScDeformableVolumeSim.h"
	#include "ScParticleSystemSim.h"
	#include "DyParticleSystem.h"
#endif

using namespace physx;
using namespace Cm;
using namespace Dy;
using namespace Sc;

PX_IMPLEMENT_OUTPUT_ERROR

static PxU32 getAvbdCpuIsaModeCode(const char* value)
{
	if(value && std::strcmp(value, "sse2") == 0)
		return 1u;
	if(value && std::strcmp(value, "avx2fma") == 0)
		return 2u;
	return value && std::strcmp(value, "auto") != 0 ? 3u : 0u;
}

static PxU32 getAvbdCpuIsaCapabilityMask(const Dy::AvbdCpuIsaCapabilities& capabilities)
{
	return (capabilities.sse2 ? 1u : 0u) |
		(capabilities.avx ? 2u : 0u) |
		(capabilities.osxsave ? 4u : 0u) |
		(capabilities.xmmYmmState ? 8u : 0u) |
		(capabilities.avx2 ? 16u : 0u) |
		(capabilities.fma ? 32u : 0u);
}

// This transaction joins the three independent static contact sources used by
// the visual cloth fixture: world planes, world-static boxes, and one
// self-colliding soft body.  It is the production fast path when its exact
// topology admission succeeds.  Child results are still merged in the
// canonical serial OGC order before contact-state transfer, so completion
// order never becomes a physics input.  Set the switch to exactly "0" only
// to force the serial authority path while diagnosing a scene.
static bool useAvbdStaticWorldSelfOgcTaskFanIn()
{
	static const bool enabled = []()
	{
		const char* const value = std::getenv(
			"PHYSX_AVBD_P5_STATIC_WORLD_SELF_TASK_FANIN");
		return !(value && value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

static const PxU32 eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK = 128u;

// Volume bodies do not expose the Surface pair-update/substep API.  Keep this
// narrowly scoped cadence experiment process-controlled so it can compare the
// same eight position sweeps as 8x1 versus a shorter outer schedule without
// changing the public default contact contract.
static bool useAvbdVolumeTest3x3Cadence()
{
	static const bool enabled = []()
	{
		const char* const value = std::getenv(
			"PHYSX_AVBD_VOLUME_TEST_3X3");
		return value && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

namespace physx { 
namespace Sc {

#if PX_SUPPORT_GPU_PHYSX

	class LLDeformableSurfacePool : public PxPool<DeformableSurface, PxAlignedAllocator<64> >
	{
	public:
		LLDeformableSurfacePool() {}
	};

	class LLDeformableVolumePool : public PxPool<DeformableVolume, PxAlignedAllocator<64> >
	{
	public:
		LLDeformableVolumePool() {}
	};

	class LLParticleSystemPool : public PxPool<ParticleSystem, PxAlignedAllocator<64> >
	{
	public:
		LLParticleSystemPool() {}
	};

#endif // PX_SUPPORT_GPU_PHYSX

	class AvbdCpuSoftScene :
		public PxUserAllocated,
		public Dy::AvbdSoftIslandProvider
	{
		enum EntryKind
		{
			eVOLUME,
			eSURFACE
		};
		enum
		{
			eELEMENT_FILTER_ALL = 0x000fffff
		};

		struct Entry
		{
			EntryKind					kind;
			DeformableVolumeCore*		volumeCore;
			DeformableSurfaceCore*		surfaceCore;
			PxTetrahedronMesh*			simulationMesh;
			PxTetrahedronMesh*			collisionMesh;
			PxDeformableVolumeAuxData*	auxData;
			PxTriangleMesh*				triangleMesh;
			PxU32						bodyIndex;
			void*						islandObject;
			PxNodeIndex					islandNode;
			bool						sleeping;

			Entry(
				DeformableVolumeCore& c,
				PxTetrahedronMesh& simMesh,
				PxTetrahedronMesh& collMesh,
				PxDeformableVolumeAuxData& aux,
				PxU32 body,
				Dy::DeformableVolume& object,
				PxNodeIndex node,
				bool startsSleeping)
				: kind(eVOLUME), volumeCore(&c), surfaceCore(NULL),
				  simulationMesh(&simMesh),
				  collisionMesh(&collMesh), auxData(&aux),
				  triangleMesh(NULL),
				  bodyIndex(body), islandObject(
					  static_cast<void*>(&object)),
				  islandNode(node), sleeping(startsSleeping)
			{
			}

			Entry(
				DeformableSurfaceCore& c,
				PxTriangleMesh& mesh,
				PxU32 body,
				Dy::DeformableSurface& object,
				PxNodeIndex node,
				bool startsSleeping)
				: kind(eSURFACE), volumeCore(NULL), surfaceCore(&c),
				  simulationMesh(NULL), collisionMesh(NULL),
				  auxData(NULL), triangleMesh(&mesh),
				  bodyIndex(body), islandObject(
					  static_cast<void*>(&object)),
				  islandNode(node), sleeping(startsSleeping)
			{
			}

			PX_FORCE_INLINE ActorCore* getActorCore() const
			{
				return kind == eVOLUME
					? static_cast<ActorCore*>(volumeCore)
					: static_cast<ActorCore*>(surfaceCore);
			}

			PX_FORCE_INLINE Dy::DeformableBodyCore& getBodyCore()
			{
				return kind == eVOLUME
					? static_cast<Dy::DeformableBodyCore&>(
						volumeCore->getCore())
					: static_cast<Dy::DeformableBodyCore&>(
						surfaceCore->getCore());
			}

			PX_FORCE_INLINE const Dy::DeformableBodyCore&
				getBodyCore() const
			{
				return kind == eVOLUME
					? static_cast<const Dy::DeformableBodyCore&>(
						volumeCore->getCore())
					: static_cast<const Dy::DeformableBodyCore&>(
						surfaceCore->getCore());
			}

			PX_FORCE_INLINE PxVec4* getPositionInvMass() const
			{
				return kind == eVOLUME
					? volumeCore->getCore().simPositionInvMass
					: surfaceCore->getCore().positionInvMass;
			}

			PX_FORCE_INLINE PxVec4* getVelocity() const
			{
				return kind == eVOLUME
					? volumeCore->getCore().simVelocity
					: surfaceCore->getCore().velocity;
			}

			PX_FORCE_INLINE PxActorFlags getActorFlags() const
			{
				return getBodyCore().actorFlags;
			}

			PX_FORCE_INLINE PxU16 getSolverIterationCounts() const
			{
				return getBodyCore().solverIterationCounts;
			}

			void destroyIslandObject()
			{
				if(!islandObject)
					return;
				if(kind == eVOLUME)
					static_cast<Dy::DeformableVolume*>(
						islandObject)->~DeformableVolume();
				else
					static_cast<Dy::DeformableSurface*>(
						islandObject)->~DeformableSurface();
				PX_FREE(islandObject);
				islandObject = NULL;
				islandNode = PxNodeIndex();
			}
		};

		struct StaticShapeEntry
		{
			StaticCore*	core;
			const ShapeCore*	shape;
			PxU64		primitiveKey;

			StaticShapeEntry(
				StaticCore& staticCore,
				ShapeCore& shapeCore,
				PxU64 key)
				: core(&staticCore), shape(&shapeCore),
				  primitiveKey(key)
			{
			}
		};

		struct DynamicShapeEntry
		{
			BodyCore*	core;
			const ShapeCore*	shape;
			PxU64		primitiveKey;

			DynamicShapeEntry(
				BodyCore& bodyCore,
				ShapeCore& shapeCore,
				PxU64 key)
				: core(&bodyCore), shape(&shapeCore),
				  primitiveKey(key)
			{
			}
		};

		struct WorldPinEntry
		{
			ActorCore*				softCore;
			Dy::AvbdSoftPoint		localPoint;
			PxVec3					worldTarget;
			PxU32					handle;

			WorldPinEntry(
				ActorCore& core,
				const Dy::AvbdSoftPoint& point,
				const PxVec3& target,
				PxU32 stableHandle)
				: softCore(&core), localPoint(point),
				  worldTarget(target), handle(stableHandle)
			{
			}
		};

		struct RigidAttachmentEntry
		{
			ActorCore*				softCore;
			BodyCore*				rigidCore;
			Dy::AvbdSoftPoint		localPoint;
			PxVec3					actorLocalTarget;
			PxVec3					alLambda;
			PxReal					k;
			PxReal					kMax;
			PxU32					handle;

			RigidAttachmentEntry(
				ActorCore& soft,
				BodyCore& rigid,
				const Dy::AvbdSoftPoint& point,
				const PxVec3& target,
				PxU32 stableHandle)
				: softCore(&soft), rigidCore(&rigid),
				  localPoint(point), actorLocalTarget(target),
				  alLambda(0.0f), k(1.0e8f), kMax(1.0e10f),
				  handle(stableHandle)
			{
			}
		};

		struct ArticulationAttachmentEntry
		{
			ActorCore*				softCore;
			BodyCore*				linkCore;
			Dy::AvbdSoftPoint		localPoint;
			PxVec3					actorLocalTarget;
			PxVec3					alLambda;
			PxReal					k;
			PxReal					kMax;
			PxU32					handle;

			ArticulationAttachmentEntry(
				ActorCore& soft,
				BodyCore& link,
				const Dy::AvbdSoftPoint& point,
				const PxVec3& target,
				PxU32 stableHandle)
				: softCore(&soft), linkCore(&link),
				  localPoint(point), actorLocalTarget(target),
				  alLambda(0.0f), k(1.0e8f), kMax(1.0e10f),
				  handle(stableHandle)
			{
			}
		};

		struct SoftPairAttachmentEntry
		{
			ActorCore*				softCore[2];
			Dy::AvbdSoftPoint		localPoint[2];
			PxVec3					alLambda;
			PxReal					k;
			PxReal					kMax;
			PxU32					handle;

			SoftPairAttachmentEntry(
				ActorCore& soft0,
				const Dy::AvbdSoftPoint& point0,
				ActorCore& soft1,
				const Dy::AvbdSoftPoint& point1,
				PxU32 stableHandle)
				: alLambda(0.0f), k(1.0e8f), kMax(1.0e10f),
				  handle(stableHandle)
			{
				softCore[0] = &soft0;
				softCore[1] = &soft1;
				localPoint[0] = point0;
				localPoint[1] = point1;
			}
		};

		struct PrescribedAttachmentEntry
		{
			ActorCore*				softCore;
			RigidCore*				prescribedCore;
			Dy::AvbdSoftPoint		localPoint;
			PxVec3					actorLocalTarget;
			PxVec3					worldTarget;
			PxVec3					previousWorldTarget;
			PxVec3					alLambda;
			PxReal					k;
			PxReal					kMax;
			PxU32					handle;
			bool					active;

			PrescribedAttachmentEntry(
				ActorCore& soft,
				RigidCore& prescribed,
				const Dy::AvbdSoftPoint& point,
				const PxVec3& actorTarget,
				const PxVec3& target,
				PxU32 stableHandle)
				: softCore(&soft), prescribedCore(&prescribed),
				  localPoint(point),
				  actorLocalTarget(actorTarget),
				  worldTarget(target),
				  previousWorldTarget(target),
				  alLambda(0.0f), k(1.0e8f), kMax(1.0e10f),
				  handle(stableHandle), active(true)
			{
			}
		};

		struct RigidActorFilterEntry
		{
			ActorCore*				softCore;
			ActorCore*				rigidCore;
			// Surface entries own public source triangles. Volume entries
			// own simulation tetrahedra compiled from public collision tets.
			PxArray<PxU32>			elementIndices;
			PxU32					handle;
			bool					filterAllElements;

			RigidActorFilterEntry(
				ActorCore& soft,
				ActorCore& rigid,
				const PxU32* elements,
				PxU32 elementCount,
				PxU32 stableHandle,
				bool filterAll)
				: softCore(&soft), rigidCore(&rigid),
				  handle(stableHandle),
				  filterAllElements(filterAll)
			{
				for(PxU32 i = 0; i < elementCount; ++i)
				{
					bool duplicate = false;
					for(PxU32 j = 0;
						j < elementIndices.size(); ++j)
						if(elementIndices[j] == elements[i])
						{
							duplicate = true;
							break;
						}
					if(!duplicate)
						elementIndices.pushBack(elements[i]);
				}
			}

			bool containsElement(PxU32 elementIndex) const
			{
				for(PxU32 i = 0; i < elementIndices.size(); ++i)
					if(elementIndices[i] == elementIndex)
						return true;
				return false;
			}
		};

		struct DeformablePairFilterEntry
		{
			struct ElementPair
			{
				PxU32 element0;
				PxU32 element1;

				ElementPair(PxU32 source0, PxU32 source1)
					: element0(source0), element1(source1)
				{
				}
			};

			ActorCore*				core0;
			ActorCore*				core1;
			PxArray<ElementPair>	elementPairs;
			PxU32					handle;

			DeformablePairFilterEntry(
				ActorCore& actor0,
				ActorCore& actor1,
				const PxU32* elements0,
				const PxU32* elements1,
				PxU32 pairCount,
				PxU32 stableHandle)
				: core0(&actor0), core1(&actor1),
				  handle(stableHandle)
			{
				for(PxU32 i = 0; i < pairCount; ++i)
				{
					bool duplicate = false;
					for(PxU32 j = 0;
						j < elementPairs.size(); ++j)
					{
						if(elementPairs[j].element0 ==
								elements0[i] &&
							elementPairs[j].element1 ==
								elements1[i])
						{
							duplicate = true;
							break;
						}
					}
					if(!duplicate)
						elementPairs.pushBack(
							ElementPair(
								elements0[i], elements1[i]));
				}
			}

			bool containsPair(
				ActorCore& queryCore,
				PxU32 queryElement,
				ActorCore& targetCore,
				PxU32 targetElement) const
			{
				const bool forward =
					core0 == &queryCore && core1 == &targetCore;
				const bool reverse =
					core1 == &queryCore && core0 == &targetCore;
				if(!forward && !reverse)
					return false;
				for(PxU32 i = 0; i < elementPairs.size(); ++i)
				{
					const PxU32 querySelection = forward
						? elementPairs[i].element0
						: elementPairs[i].element1;
					const PxU32 targetSelection = forward
						? elementPairs[i].element1
						: elementPairs[i].element0;
					if((querySelection == eELEMENT_FILTER_ALL ||
							querySelection == queryElement) &&
						(targetSelection == eELEMENT_FILTER_ALL ||
							targetSelection == targetElement))
						return true;
				}
				return false;
			}
		};

		struct NativeIslandEdgeEntry
		{
			ActorCore*				softCore;
			BodyCore*				rigidCore;
			IG::EdgeIndex			edgeIndex;
			bool					touched;

			NativeIslandEdgeEntry(
				ActorCore& soft,
				BodyCore& rigid,
				IG::EdgeIndex index)
				: softCore(&soft), rigidCore(&rigid),
				  edgeIndex(index), touched(true)
			{
			}
		};

		struct NativeSoftSoftIslandEdgeEntry
		{
			ActorCore*				softCore0;
			ActorCore*				softCore1;
			IG::EdgeIndex			edgeIndex;
			bool					touched;

			NativeSoftSoftIslandEdgeEntry(
				ActorCore& soft0,
				ActorCore& soft1,
				IG::EdgeIndex index)
				: softCore0(&soft0), softCore1(&soft1),
				  edgeIndex(index), touched(true)
			{
			}
		};

		struct IslandSelectionStorage
		{
			IG::IslandId					nativeIslandId;
			bool							touched;
			PxU32							selectedIsland;
			PxArray<PxU32>					entryIndices;
			PxArray<ActorCore*>				softCores;
			// A contiguous canonical particle range can be bound directly into
			// the native solve.  Bodies remain locally rebound for now, but this
			// removes the hot live-particle gather/scatter boundary without
			// changing the legacy local-index ABI.
			bool							usesCanonicalParticleRange;
			PxU32							canonicalParticleStart;
			PxU32							canonicalParticleCount;
			PxArray<PxU32>					globalParticleIndices;
			PxArray<Dy::AvbdSoftParticle>	particles;
			PxArray<Dy::AvbdSoftBody>		bodies;
			// Provider-owned compiled support program consumed directly by the
			// native mixed solver.  It is the common execution seam for a later
			// indexed/canonical island view.
			PxArray<PxU32>					particleBodyIndices;
			PxArray<PxU32>					contactStarts;
			PxArray<PxU32>					contactCounts;
			PxArray<Dy::AvbdSoftContactParticleRef>
											contactRefs;
			// Geometry-only companion CSR for complete triangle/OBB safety.
			// It is consumed by the OGC candidate admission filter, never by the
			// AL contact force/Hessian program.
			PxArray<PxU32>					triangleCoreSafetyStarts;
			PxArray<PxU32>					triangleCoreSafetyCounts;
			PxArray<Dy::AvbdSoftContactParticleRef>
											triangleCoreSafetyRefs;
			PxArray<PxU32>					rigidTargetContactStarts;
			PxArray<PxU32>					rigidTargetContactCounts;
			PxArray<PxU32>					rigidTargetContactRefs;
			PxArray<Dy::AvbdSelfCollisionAdjacency>
												selfCollisionAdjacencies;
			PxArray<PxU8>					selfCollisionEnabled;
			PxArray<Dy::AvbdRigidBox>		rigidBoxes;
			PxArray<Dy::AvbdRigidBox>		selectedDynamicBoxes;
			// Solver-task-owned copy of the cooked collision-domain topology and
			// its proxy-to-simulation embedding.  This is intentionally separate
			// from mSubsetCollision* scratch: the terminal current-pose OGC epoch
			// runs inside a native island task, where Scene-global subset scratch
			// would be neither immutable nor task-safe.
			PxArray<Dy::AvbdSoftBody>		terminalCollisionBodies;
			PxArray<Dy::AvbdWeightedContactPoint>
				terminalCollisionVertexMappings;
			PxArray<Dy::AvbdOgcPairState>		ogcPairStates;
			PxArray<PxU32>						ogcPairIndices;
			PxArray<PxU32>						ogcPairContactStarts;
			PxArray<PxU32>						ogcPairContactCounts;
			PxArray<PxU32>						ogcPairContactRefs;
			PxArray<Dy::AvbdRigidSphere>	rigidSpheres;
			PxArray<Dy::AvbdRigidSphere>	selectedDynamicSpheres;
			PxArray<Dy::AvbdRigidCapsule>	rigidCapsules;
			PxArray<Dy::AvbdRigidCapsule>	selectedDynamicCapsules;
			PxArray<Dy::AvbdRigidConvex>	rigidConvexes;
			PxArray<Dy::AvbdRigidConvex>	selectedDynamicConvexes;
			PxArray<Dy::AvbdSoftContact>	contacts;
			PxArray<Dy::AvbdSoftContact>	probeContacts;

			IslandSelectionStorage()
				: nativeIslandId(IG_INVALID_ISLAND), touched(false),
				  selectedIsland(PX_MAX_U32),
				  usesCanonicalParticleRange(false),
				  canonicalParticleStart(PX_MAX_U32),
				  canonicalParticleCount(0)
			{
			}
		};

		// Diagnostic-only identity record for comparing a future collision
		// refit/rebuild implementation with the current canonical OGC output.
		// It intentionally excludes floating geometry: this gate owns the
		// discrete contact-feature set, while existing E51 tests own the
		// numerical/contact-response checks.
		struct ContactSetTraceKey
		{
			PxU32 sourceType;
			PxU32 targetBodyIndex;
			PxU64 primitiveKey;
			PxU64 featureKey;
			PxU32 particleIndex;
			PxU32 targetKind;
			PxU32 targetIndex;
			PxU32 targetSourceElementIndex;
			PxU32 queryParticleIndices[3];
			PxU32 surfaceParticleIndices[3];
			// Optional current-pose triangle/box core provenance.  The contact
			// trace is diagnostic-only, but preserving the selected OBB exit face
			// makes it possible to distinguish a multi-face endpoint DCD case from
			// a plain vertex/SDF row without exposing any swept/CCD information.
			PxU32 triangleCoreExitFace;

			bool operator<(const ContactSetTraceKey& other) const
			{
				const PxU64 values[] =
				{
					sourceType, targetBodyIndex, primitiveKey, featureKey,
					particleIndex, targetKind, targetIndex,
					targetSourceElementIndex, queryParticleIndices[0],
					queryParticleIndices[1], queryParticleIndices[2],
					surfaceParticleIndices[0], surfaceParticleIndices[1],
					surfaceParticleIndices[2], triangleCoreExitFace
				};
				const PxU64 otherValues[] =
				{
					other.sourceType, other.targetBodyIndex,
					other.primitiveKey, other.featureKey,
					other.particleIndex, other.targetKind,
					other.targetIndex, other.targetSourceElementIndex,
					other.queryParticleIndices[0],
					other.queryParticleIndices[1],
					other.queryParticleIndices[2],
					other.surfaceParticleIndices[0],
					other.surfaceParticleIndices[1],
					other.surfaceParticleIndices[2],
					other.triangleCoreExitFace
				};
				for(PxU32 i = 0;
					i < sizeof(values) / sizeof(values[0]); i++)
				{
					if(values[i] != otherValues[i])
						return values[i] < otherValues[i];
				}
				return false;
			}
		};

		// P3 keeps the nonlinear component solve serial for now.  Its first
		// task-graph slice is deliberately limited to post-solve output ranges:
		// each range owns whole deformable entries and therefore disjoint host
		// particle/output buffers.  The parent Scene task joins these before it
		// mutates sleep/island state or continues the simulation pipeline.
		class WriteBackTask : public Cm::Task, public PxUserAllocated
		{
		public:
			WriteBackTask(PxU64 contextId, AvbdCpuSoftScene& scene)
				: Cm::Task(contextId), mScene(scene), mEntryBegin(0),
				  mEntryEnd(0), mTaskGraphContext(NULL)
			{
			}

			void configure(
				PxU32 entryBegin, PxU32 entryEnd,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(entryBegin < entryEnd);
				mEntryBegin = entryBegin;
				mEntryEnd = entryEnd;
				mTaskGraphContext = &taskGraphContext;
			}

			virtual void runInternal() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftWriteBack";
			}

		private:
			AvbdCpuSoftScene&					mScene;
			PxU32							mEntryBegin;
			PxU32							mEntryEnd;
			Dy::AvbdDynamicsContext*			mTaskGraphContext;
		};

		// The P3 pre-solve task owns only whole-entry particle ranges.  It runs
		// after the low-level ePREPARE prefix and before eRESUME's predicted
		// position OGC redetection, so no task observes or mutates contact state.
		class PredictionTask : public Cm::Task, public PxUserAllocated
		{
		public:
			PredictionTask(PxU64 contextId, AvbdCpuSoftScene& scene)
				: Cm::Task(contextId), mScene(scene), mEntryBegin(0),
				  mEntryEnd(0), mDt(0.0f), mGravity(0.0f),
				  mTaskGraphContext(NULL)
			{
			}

			void configure(
				PxU32 entryBegin, PxU32 entryEnd, PxReal dt,
				const PxVec3& gravity,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(entryBegin < entryEnd);
				mEntryBegin = entryBegin;
				mEntryEnd = entryEnd;
				mDt = dt;
				mGravity = gravity;
				mTaskGraphContext = &taskGraphContext;
			}

			virtual void runInternal() PX_OVERRIDE;

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftPrediction";
			}

		private:
			AvbdCpuSoftScene&					mScene;
			PxU32							mEntryBegin;
			PxU32							mEntryEnd;
			PxReal							mDt;
			PxVec3							mGravity;
			Dy::AvbdDynamicsContext*			mTaskGraphContext;
		};

		// A primal child owns either one stable packed causal-layer subrange or a
		// contiguous range of complete independent bodies. It receives no
		// Scene/contact/workspace mutable state beyond the frozen solve context,
		// and writes one private observation for the parent fan-in.
		class CausalLayerTask : public Cm::Task, public PxUserAllocated
		{
		public:
			CausalLayerTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mSolveContext(NULL), mBodies(NULL), mBodyCount(0),
				  mParticleBodyIndices(NULL), mNumParticles(0),
				  mPackedParticleIndices(NULL), mPackedBegin(0), mPackedEnd(0),
				  mIndependentBodyRange(false), mBodyBegin(0), mBodyEnd(0),
				  mObservation(NULL), mTaskGraphContext(NULL)
			{
			}

			void configure(
				const Dy::AvbdParticlePrimalSolveContext& solveContext,
				const Dy::AvbdSoftBody* bodies, PxU32 bodyCount,
				const PxU32* particleBodyIndices, PxU32 numParticles,
				const PxU32* packedParticleIndices,
				PxU32 packedBegin, PxU32 packedEnd,
				Dy::AvbdParticlePrimalRangeObservation& observation,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(bodies && bodyCount > 0);
				PX_ASSERT(particleBodyIndices && packedParticleIndices);
				PX_ASSERT(packedBegin < packedEnd);
				mSolveContext = &solveContext;
				mBodies = bodies;
				mBodyCount = bodyCount;
				mParticleBodyIndices = particleBodyIndices;
				mNumParticles = numParticles;
				mPackedParticleIndices = packedParticleIndices;
				mPackedBegin = packedBegin;
				mPackedEnd = packedEnd;
				mIndependentBodyRange = false;
				mBodyBegin = 0;
				mBodyEnd = 0;
				mObservation = &observation;
				mTaskGraphContext = &taskGraphContext;
			}

			void configureIndependentBodyRange(
				const Dy::AvbdParticlePrimalSolveContext& solveContext,
				const Dy::AvbdSoftBody* bodies, PxU32 bodyCount,
				PxU32 bodyBegin, PxU32 bodyEnd,
				Dy::AvbdParticlePrimalRangeObservation& observation,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(bodies && bodyCount > 1 &&
					bodyBegin < bodyEnd && bodyEnd <= bodyCount);
				mSolveContext = &solveContext;
				mBodies = bodies;
				mBodyCount = bodyCount;
				mParticleBodyIndices = NULL;
				mNumParticles = 0;
				mPackedParticleIndices = NULL;
				mPackedBegin = 0;
				mPackedEnd = 0;
				mIndependentBodyRange = true;
				mBodyBegin = bodyBegin;
				mBodyEnd = bodyEnd;
				mObservation = &observation;
				mTaskGraphContext = &taskGraphContext;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE;

			virtual void release() PX_OVERRIDE
			{
				// Keep the continuation local before returning this persistent task
				// to the pool: the final child can immediately wake its parent on
				// another dispatcher worker.
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleCausalLayerTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftCausalLayer";
			}

		private:
			AvbdCpuSoftScene&					mScene;
			PxU32						mPoolIndex;
			const Dy::AvbdParticlePrimalSolveContext*	mSolveContext;
			const Dy::AvbdSoftBody*			mBodies;
			PxU32						mBodyCount;
			const PxU32*					mParticleBodyIndices;
			PxU32						mNumParticles;
			const PxU32*					mPackedParticleIndices;
			PxU32						mPackedBegin;
			PxU32						mPackedEnd;
			bool						mIndependentBodyRange;
			PxU32						mBodyBegin;
			PxU32						mBodyEnd;
			Dy::AvbdParticlePrimalRangeObservation*	mObservation;
			Dy::AvbdDynamicsContext*			mTaskGraphContext;
		};

		// A parent fan-in task owns no particle range. It merges private
		// observations in fixed range order, advances the persistent step state,
		// and asks Scene to publish the next primal range or resume the existing
		// post-solve/write-back continuation.
		class CausalLayerFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			CausalLayerFinishTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mOwner.avbdCpuSoftComponentCausalLayerFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleCausalLayerFinishTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftCausalLayerFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.3b: the first collision worker is intentionally narrower than the
		// generic OGC rebuild.  It owns one particle interval of a world-plane
		// only transaction and may append solely to its pre-reserved private
		// stream.  The parent performs the fixed-order merge and every mutable
		// contact/workspace transition after the fan-in.
		class WorldPlaneContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			WorldPlaneContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mParticles(NULL), mNumParticles(0), mParticleBegin(0),
				  mParticleEnd(0), mPlanes(NULL), mNumPlanes(0),
				  mBodies(NULL), mNumBodies(0), mContacts(NULL), mMargin(0.0f),
				  mTaskGraphContext(NULL)
			{
			}

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdWorldPlane* planes, PxU32 numPlanes,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts, PxReal margin,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(particles && particleBegin < particleEnd);
				PX_ASSERT(particleEnd <= numParticles);
				PX_ASSERT(planes && numPlanes > 0 && bodies && numBodies > 0);
				mParticles = particles;
				mNumParticles = numParticles;
				mParticleBegin = particleBegin;
				mParticleEnd = particleEnd;
				mPlanes = planes;
				mNumPlanes = numPlanes;
				mBodies = bodies;
				mNumBodies = numBodies;
				mContacts = &contacts;
				mMargin = margin;
				mTaskGraphContext = &taskGraphContext;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE
			{
				PX_ASSERT(mParticles && mPlanes && mBodies && mContacts &&
					mTaskGraphContext);
				mTaskGraphContext->beginWorldPlaneContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginWorldPlaneContactTask();
				Dy::avbdDetectSoftWorldPlaneContactsRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mPlanes, mNumPlanes, *mContacts, mMargin,
					mBodies, mNumBodies);
				mScene.mStandaloneTaskGraphTelemetry.
					endWorldPlaneContactTask();
				mTaskGraphContext->endWorldPlaneContactTask();
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleWorldPlaneContactTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftWorldPlaneContact";
			}

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32						mPoolIndex;
			const Dy::AvbdSoftParticle*	mParticles;
			PxU32						mNumParticles;
			PxU32						mParticleBegin;
			PxU32						mParticleEnd;
			const Dy::AvbdWorldPlane*		mPlanes;
			PxU32						mNumPlanes;
			const Dy::AvbdSoftBody*		mBodies;
			PxU32						mNumBodies;
			PxArray<Dy::AvbdSoftContact>*	mContacts;
			PxReal						mMargin;
			Dy::AvbdDynamicsContext*		mTaskGraphContext;
		};

		class WorldPlaneContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			WorldPlaneContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mScene.mStandaloneTaskGraphTelemetry.
					recordWorldPlaneContactFanIn();
				mOwner.avbdCpuSoftComponentWorldPlaneContactFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleWorldPlaneContactFinishTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftWorldPlaneContactFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.12b range-owns both static-box SDF families, but retains independent
		// private streams. The parent stable-merges all current ranges before all
		// swept ranges, then appends the feature suffix. Both leaves read only
		// immutable inputs, so a child may compute them back-to-back.
		class RigidBoxSdfContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidBoxSdfContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mParticles(NULL), mNumParticles(0), mParticleBegin(0),
				  mParticleEnd(0), mBoxes(NULL), mNumBoxes(0),
				  mPreviousContacts(NULL), mNumPreviousContacts(0),
				  mBodies(NULL), mNumBodies(0), mContacts(NULL),
				  mSweptContacts(NULL), mMargin(0.0f),
				  mTaskGraphContext(NULL)
			{
			}

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidBox* boxes, PxU32 numBoxes,
				const Dy::AvbdSoftContact* previousContacts,
				PxU32 numPreviousContacts,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(particles && particleBegin < particleEnd);
				PX_ASSERT(particleEnd <= numParticles);
				PX_ASSERT(boxes && numBoxes > 0 && bodies && numBodies > 0);
				mParticles = particles;
				mNumParticles = numParticles;
				mParticleBegin = particleBegin;
				mParticleEnd = particleEnd;
				mBoxes = boxes;
				mNumBoxes = numBoxes;
				mPreviousContacts = previousContacts;
				mNumPreviousContacts = numPreviousContacts;
				mBodies = bodies;
				mNumBodies = numBodies;
				mContacts = &contacts;
				mSweptContacts = &sweptContacts;
				mMargin = margin;
				mTaskGraphContext = &taskGraphContext;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE
			{
				PX_ASSERT(mParticles && mBoxes && mBodies && mContacts &&
					mSweptContacts &&
					mTaskGraphContext);
				mTaskGraphContext->beginRigidBoxSdfContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginRigidBoxSdfContactTask();
				Dy::avbdDetectSoftRigidSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mBoxes, mNumBoxes, *mContacts, mMargin,
					mPreviousContacts, mNumPreviousContacts,
					mBodies, mNumBodies);
				Dy::avbdDetectSoftRigidSweptSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mBoxes, mNumBoxes, *mSweptContacts, mMargin,
					mBodies, mNumBodies);
				mScene.mStandaloneTaskGraphTelemetry.
					endRigidBoxSdfContactTask();
				mTaskGraphContext->endRigidBoxSdfContactTask();
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidBoxSdfContactTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidBoxSdfContact";
			}

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32						mPoolIndex;
			const Dy::AvbdSoftParticle*	mParticles;
			PxU32						mNumParticles;
			PxU32						mParticleBegin;
			PxU32						mParticleEnd;
			const Dy::AvbdRigidBox*		mBoxes;
			PxU32						mNumBoxes;
			const Dy::AvbdSoftContact*	mPreviousContacts;
			PxU32						mNumPreviousContacts;
			const Dy::AvbdSoftBody*		mBodies;
			PxU32						mNumBodies;
			PxArray<Dy::AvbdSoftContact>*	mContacts;
			PxArray<Dy::AvbdSoftContact>*	mSweptContacts;
			PxReal						mMargin;
			Dy::AvbdDynamicsContext*		mTaskGraphContext;
		};

		class RigidBoxSdfContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidBoxSdfContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mScene.mStandaloneTaskGraphTelemetry.
					recordRigidBoxSdfContactFanIn();
				mOwner.avbdCpuSoftComponentRigidBoxSdfContactFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidBoxSdfContactFinishTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidBoxSdfContactFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.13b range-owns the current and swept static-sphere SDF families,
		// with independent private streams. The parent merges the complete
		// current family before the swept family and retains both feature suffixes.
		class RigidSphereSdfContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidSphereSdfContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mParticles(NULL), mNumParticles(0), mParticleBegin(0),
				  mParticleEnd(0), mSpheres(NULL), mNumSpheres(0),
				  mBodies(NULL), mNumBodies(0), mContacts(NULL),
				  mSweptContacts(NULL), mMargin(0.0f),
				  mTaskGraphContext(NULL)
			{
			}

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidSphere* spheres, PxU32 numSpheres,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(particles && particleBegin < particleEnd);
				PX_ASSERT(particleEnd <= numParticles);
				PX_ASSERT(spheres && numSpheres > 0 && bodies && numBodies > 0);
				mParticles = particles;
				mNumParticles = numParticles;
				mParticleBegin = particleBegin;
				mParticleEnd = particleEnd;
				mSpheres = spheres;
				mNumSpheres = numSpheres;
				mBodies = bodies;
				mNumBodies = numBodies;
				mContacts = &contacts;
				mSweptContacts = &sweptContacts;
				mMargin = margin;
				mTaskGraphContext = &taskGraphContext;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE
			{
				PX_ASSERT(mParticles && mSpheres && mBodies && mContacts &&
					mSweptContacts && mTaskGraphContext);
				mTaskGraphContext->beginRigidSphereSdfContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginRigidSphereSdfContactTask();
				Dy::avbdDetectSoftRigidSphereSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mSpheres, mNumSpheres, *mContacts, mMargin,
					mBodies, mNumBodies);
				Dy::avbdDetectSoftRigidSphereSweptSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mSpheres, mNumSpheres, *mSweptContacts, mMargin,
					mBodies, mNumBodies);
				mScene.mStandaloneTaskGraphTelemetry.
					endRigidSphereSdfContactTask();
				mTaskGraphContext->endRigidSphereSdfContactTask();
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidSphereSdfContactTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidSphereSdfContact";
			}

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32						mPoolIndex;
			const Dy::AvbdSoftParticle*	mParticles;
			PxU32						mNumParticles;
			PxU32						mParticleBegin;
			PxU32						mParticleEnd;
			const Dy::AvbdRigidSphere*	mSpheres;
			PxU32						mNumSpheres;
			const Dy::AvbdSoftBody*		mBodies;
			PxU32						mNumBodies;
			PxArray<Dy::AvbdSoftContact>*	mContacts;
			PxArray<Dy::AvbdSoftContact>*	mSweptContacts;
			PxReal						mMargin;
			Dy::AvbdDynamicsContext*		mTaskGraphContext;
		};

		class RigidSphereSdfContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidSphereSdfContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mScene.mStandaloneTaskGraphTelemetry.
					recordRigidSphereSdfContactFanIn();
				mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidSphereSdfContactFinishTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidSphereSdfContactFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.14b range-owns current and swept static-capsule SDF in independent
		// streams. The Scene continuation remains mutually exclusive with spheres;
		// neither geometry, telemetry, nor output storage is shared.
		class RigidCapsuleSdfContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidCapsuleSdfContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mParticles(NULL), mNumParticles(0), mParticleBegin(0),
				  mParticleEnd(0), mCapsules(NULL), mNumCapsules(0),
				  mBodies(NULL), mNumBodies(0), mContacts(NULL),
				  mSweptContacts(NULL), mMargin(0.0f),
				  mTaskGraphContext(NULL)
			{
			}

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidCapsule* capsules, PxU32 numCapsules,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(particles && particleBegin < particleEnd);
				PX_ASSERT(particleEnd <= numParticles);
				PX_ASSERT(capsules && numCapsules > 0 && bodies && numBodies > 0);
				mParticles = particles;
				mNumParticles = numParticles;
				mParticleBegin = particleBegin;
				mParticleEnd = particleEnd;
				mCapsules = capsules;
				mNumCapsules = numCapsules;
				mBodies = bodies;
				mNumBodies = numBodies;
				mContacts = &contacts;
				mSweptContacts = &sweptContacts;
				mMargin = margin;
				mTaskGraphContext = &taskGraphContext;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE
			{
				PX_ASSERT(mParticles && mCapsules && mBodies && mContacts &&
					mSweptContacts && mTaskGraphContext);
				mTaskGraphContext->beginRigidCapsuleSdfContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginRigidCapsuleSdfContactTask();
				Dy::avbdDetectSoftRigidCapsuleSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mCapsules, mNumCapsules, *mContacts, mMargin,
					mBodies, mNumBodies);
				Dy::avbdDetectSoftRigidCapsuleSweptSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mCapsules, mNumCapsules, *mSweptContacts, mMargin,
					mBodies, mNumBodies);
				mScene.mStandaloneTaskGraphTelemetry.
					endRigidCapsuleSdfContactTask();
				mTaskGraphContext->endRigidCapsuleSdfContactTask();
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidCapsuleSdfContactTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidCapsuleSdfContact";
			}

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32						mPoolIndex;
			const Dy::AvbdSoftParticle*	mParticles;
			PxU32						mNumParticles;
			PxU32						mParticleBegin;
			PxU32						mParticleEnd;
			const Dy::AvbdRigidCapsule*	mCapsules;
			PxU32						mNumCapsules;
			const Dy::AvbdSoftBody*		mBodies;
			PxU32						mNumBodies;
			PxArray<Dy::AvbdSoftContact>*	mContacts;
			PxArray<Dy::AvbdSoftContact>*	mSweptContacts;
			PxReal						mMargin;
			Dy::AvbdDynamicsContext*		mTaskGraphContext;
		};

		class RigidCapsuleSdfContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidCapsuleSdfContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mScene.mStandaloneTaskGraphTelemetry.
					recordRigidCapsuleSdfContactFanIn();
				// Sphere and capsule use a single mutually-exclusive continuation
				// slot, so its owner dispatches by the pending transaction.
				mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidCapsuleSdfContactFinishTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidCapsuleSdfContactFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.15b range-owns current and swept static-convex SDF in independent
		// streams. Only the continuation readiness bit is shared with other
		// mutually exclusive smooth-rigid SDF transactions.
		class RigidConvexSdfContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidConvexSdfContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mParticles(NULL), mNumParticles(0), mParticleBegin(0),
				  mParticleEnd(0), mConvexes(NULL), mNumConvexes(0),
				  mBodies(NULL), mNumBodies(0), mContacts(NULL),
				  mSweptContacts(NULL), mMargin(0.0f),
				  mTaskGraphContext(NULL)
			{
			}

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidConvex* convexes, PxU32 numConvexes,
				const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
				PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(particles && particleBegin < particleEnd);
				PX_ASSERT(particleEnd <= numParticles);
				PX_ASSERT(convexes && numConvexes > 0 && bodies && numBodies > 0);
				mParticles = particles;
				mNumParticles = numParticles;
				mParticleBegin = particleBegin;
				mParticleEnd = particleEnd;
				mConvexes = convexes;
				mNumConvexes = numConvexes;
				mBodies = bodies;
				mNumBodies = numBodies;
				mContacts = &contacts;
				mSweptContacts = &sweptContacts;
				mMargin = margin;
				mTaskGraphContext = &taskGraphContext;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void runInternal() PX_OVERRIDE
			{
				PX_ASSERT(mParticles && mConvexes && mBodies && mContacts &&
					mSweptContacts && mTaskGraphContext);
				mTaskGraphContext->beginRigidConvexSdfContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginRigidConvexSdfContactTask();
				Dy::avbdDetectSoftRigidConvexSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mConvexes, mNumConvexes, *mContacts, mMargin,
					mBodies, mNumBodies);
				Dy::avbdDetectSoftRigidConvexSweptSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mConvexes, mNumConvexes, *mSweptContacts, mMargin,
					mBodies, mNumBodies);
				mScene.mStandaloneTaskGraphTelemetry.
					endRigidConvexSdfContactTask();
				mTaskGraphContext->endRigidConvexSdfContactTask();
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidConvexSdfContactTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidConvexSdfContact";
			}

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32						mPoolIndex;
			const Dy::AvbdSoftParticle*	mParticles;
			PxU32						mNumParticles;
			PxU32						mParticleBegin;
			PxU32						mParticleEnd;
			const Dy::AvbdRigidConvex*	mConvexes;
			PxU32						mNumConvexes;
			const Dy::AvbdSoftBody*		mBodies;
			PxU32						mNumBodies;
			PxArray<Dy::AvbdSoftContact>*	mContacts;
			PxArray<Dy::AvbdSoftContact>*	mSweptContacts;
			PxReal						mMargin;
			Dy::AvbdDynamicsContext*		mTaskGraphContext;
		};

		class RigidConvexSdfContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidConvexSdfContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mScene.mStandaloneTaskGraphTelemetry.
					recordRigidConvexSdfContactFanIn();
				mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidConvexSdfContactFinishTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidConvexSdfContactFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.17d range-owns every static triangle contact family. Current/swept
		// SDF stay particle-range streams; OGC features consume contiguous rows
		// of the immutable P5.17b parent plan. Every child carries complete BVH
		// query scratch, so the Scene descriptor remains read only while sibling
		// ranges run. Historical P5.16b: Both OGC feature suffixes remain parent-owned.
		class RigidTriangleSurfaceContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidTriangleSurfaceContactTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mParticles(NULL), mNumParticles(0), mParticleBegin(0),
				  mParticleEnd(0), mSurfaces(NULL), mNumSurfaces(0),
				  mBodies(NULL), mNumBodies(0), mContacts(NULL),
				  mSweptContacts(NULL), mFeaturePlan(NULL),
				  mFeaturePlanBegin(0), mFeaturePlanEnd(0),
				  mFeatureContacts(NULL), mFeaturePlanOutputs(NULL),
				  mFeaturePlanRowPrivateOutputs(false),
				  mFeaturePlanRoundRobin(false), mFeaturePlanTaskIndex(0),
				  mFeaturePlanTaskCount(0),
				  mForwardOwnerQueryStatsEnabled(false),
				  mDiscreteQueryStatsEnabled(false),
				  mDiscreteBodyLocalBoundsCullEnabled(false),
				  mForwardOwnerQueryStamp(0),
				  mForwardOwnerResultCacheEnabled(false),
				  mForwardOwnerResultCacheStamp(0), mMargin(0.0f),
				  mCollisionStats(NULL), mTaskGraphContext(NULL)
			{
			}

			PxU64 reserveBvhCandidateScratch(PxU32 triangleCapacity,
				PxU32 edgeCapacity, PxU32 vertexCapacity,
				PxU32 forwardOwnerQueryStampCapacity = 0,
				PxU32 forwardOwnerResultCacheCapacity = 0,
				PxU32 forwardOwnerResultCacheSurfaceSlotCapacity = 0)
			{
				const PxU32 oldTriangleCapacity =
					mQueryScratch.triangleBvhQueryCandidates.capacity();
				const PxU32 oldEdgeCapacity =
					mQueryScratch.edgeBvhQueryCandidates.capacity();
				const PxU32 oldVertexCapacity =
					mQueryScratch.vertexBvhQueryCandidates.capacity();
				const PxU32 oldEdgeStampCapacity =
					mQueryScratch.edgeBvhCandidateStamps.capacity();
				const PxU32 oldVertexStampCapacity =
					mQueryScratch.vertexBvhCandidateStamps.capacity();
				const PxU32 oldForwardOwnerStampCapacity =
					mForwardOwnerQueryStamps.capacity();
				const PxU32 oldForwardOwnerResultCacheEntryCapacity =
					mForwardOwnerResultCacheEntries.capacity();
				const PxU32 oldForwardOwnerResultCacheSurfaceSlotCapacity =
					mForwardOwnerResultCacheSurfaceSlots.capacity();
				mQueryScratch.reserve(
					triangleCapacity, edgeCapacity, vertexCapacity);
				if(forwardOwnerQueryStampCapacity)
					mForwardOwnerQueryStamps.reserve(
						forwardOwnerQueryStampCapacity);
				if(forwardOwnerResultCacheCapacity)
					mForwardOwnerResultCacheEntries.reserve(
						forwardOwnerResultCacheCapacity);
				if(forwardOwnerResultCacheSurfaceSlotCapacity)
					mForwardOwnerResultCacheSurfaceSlots.reserve(
						forwardOwnerResultCacheSurfaceSlotCapacity);
				const PxU64 queryScratchWordGrowth = PxU64(
					mQueryScratch.triangleBvhQueryCandidates.capacity() -
						oldTriangleCapacity) +
					PxU64(mQueryScratch.edgeBvhQueryCandidates.capacity() -
						oldEdgeCapacity) +
					PxU64(mQueryScratch.vertexBvhQueryCandidates.capacity() -
						oldVertexCapacity) +
					PxU64(mQueryScratch.edgeBvhCandidateStamps.capacity() -
						oldEdgeStampCapacity) +
					PxU64(mQueryScratch.vertexBvhCandidateStamps.capacity() -
					oldVertexStampCapacity) +
					PxU64(mForwardOwnerQueryStamps.capacity() -
					oldForwardOwnerStampCapacity) +
					PxU64(mForwardOwnerResultCacheEntries.capacity() -
					oldForwardOwnerResultCacheEntryCapacity) +
					PxU64(mForwardOwnerResultCacheSurfaceSlots.capacity() -
					oldForwardOwnerResultCacheSurfaceSlotCapacity);
				return queryScratchWordGrowth * sizeof(PxU32);
			}

			PxU64 getBvhCandidateScratchResidentPayloadBytes() const
			{
				const PxU64 queryScratchWords =
					PxU64(mQueryScratch.triangleBvhQueryCandidates.capacity()) +
					PxU64(mQueryScratch.edgeBvhQueryCandidates.capacity()) +
					PxU64(mQueryScratch.vertexBvhQueryCandidates.capacity()) +
					PxU64(mQueryScratch.edgeBvhCandidateStamps.capacity()) +
					PxU64(mQueryScratch.vertexBvhCandidateStamps.capacity()) +
					PxU64(mForwardOwnerQueryStamps.capacity()) +
					PxU64(mForwardOwnerResultCacheEntries.capacity()) +
					PxU64(mForwardOwnerResultCacheSurfaceSlots.capacity());
				return queryScratchWords * sizeof(PxU32);
			}

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdRigidTriangleSurface* surfaces,
				PxU32 numSurfaces, const Dy::AvbdSoftBody* bodies,
				PxU32 numBodies, PxArray<Dy::AvbdSoftContact>& contacts,
				PxArray<Dy::AvbdSoftContact>& sweptContacts,
				const Dy::AvbdRigidTriangleSurfaceFeaturePlan& featurePlan,
				PxU32 featurePlanBegin, PxU32 featurePlanEnd,
				PxArray<Dy::AvbdSoftContact>& featureContacts,
				PxReal margin, Dy::AvbdSoftCollisionStats* collisionStats,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(particles && particleBegin < particleEnd);
				PX_ASSERT(particleEnd <= numParticles);
				PX_ASSERT(surfaces && numSurfaces > 0 && bodies && numBodies > 0);
				mParticles = particles;
				mNumParticles = numParticles;
				mParticleBegin = particleBegin;
				mParticleEnd = particleEnd;
				mSurfaces = surfaces;
				mNumSurfaces = numSurfaces;
				mBodies = bodies;
				mNumBodies = numBodies;
				mContacts = &contacts;
				mSweptContacts = &sweptContacts;
				mFeaturePlan = &featurePlan;
				mFeaturePlanBegin = featurePlanBegin;
				mFeaturePlanEnd = featurePlanEnd;
				mFeatureContacts = &featureContacts;
				mFeaturePlanOutputs = NULL;
				mFeaturePlanRowPrivateOutputs = false;
				mFeaturePlanRoundRobin = false;
				mFeaturePlanTaskIndex = 0;
				mFeaturePlanTaskCount = 0;
				mForwardOwnerQueryStatsEnabled = false;
				mDiscreteQueryStatsEnabled = false;
				mDiscreteBodyLocalBoundsCullEnabled = false;
				mForwardOwnerResultCacheEnabled = false;
				mMargin = margin;
				mCollisionStats = collisionStats;
				mTaskGraphContext = &taskGraphContext;
			}

			void configureFeaturePlanRoundRobin(
				PxArray<PxArray<Dy::AvbdSoftContact> >& featurePlanOutputs,
				PxU32 taskIndex, PxU32 taskCount)
			{
				PX_ASSERT(taskCount > 0 && taskIndex < taskCount);
				mFeaturePlanOutputs = &featurePlanOutputs;
				mFeaturePlanRowPrivateOutputs = true;
				mFeaturePlanRoundRobin = true;
				mFeaturePlanTaskIndex = taskIndex;
				mFeaturePlanTaskCount = taskCount;
			}

			void configureFeaturePlanRowPrivateOutputs(
				PxArray<PxArray<Dy::AvbdSoftContact> >& featurePlanOutputs)
			{
				mFeaturePlanOutputs = &featurePlanOutputs;
				mFeaturePlanRowPrivateOutputs = true;
				mFeaturePlanRoundRobin = false;
				mFeaturePlanTaskIndex = 0;
				mFeaturePlanTaskCount = 0;
			}

			void configureForwardOwnerQueryStats()
			{
				const PxU64 requiredCapacity64 = PxU64(mNumParticles) *
					mNumSurfaces;
				PX_ASSERT(requiredCapacity64 <= PX_MAX_U32 && requiredCapacity64 > 0);
				const PxU32 requiredCapacity = PxU32(requiredCapacity64);
				if(mForwardOwnerQueryStamps.size() != requiredCapacity)
				{
					mForwardOwnerQueryStamps.resize(requiredCapacity);
					for(PxU32 index = 0; index < requiredCapacity; ++index)
						mForwardOwnerQueryStamps[index] = 0;
				}
				++mForwardOwnerQueryStamp;
				if(mForwardOwnerQueryStamp == 0)
				{
					mForwardOwnerQueryStamp = 1;
					for(PxU32 index = 0; index < requiredCapacity; ++index)
						mForwardOwnerQueryStamps[index] = 0;
				}
				mForwardOwnerQueryStats.configure(mForwardOwnerQueryStamps,
					mNumParticles, mNumSurfaces, mForwardOwnerQueryStamp);
				mForwardOwnerQueryStatsEnabled = true;
			}

			void configureDiscreteQueryStats()
			{
				mDiscreteQueryStatsEnabled = true;
			}

			void configureDiscreteBodyLocalBoundsCull()
			{
				mDiscreteBodyLocalBoundsCullEnabled = true;
			}

			void configureForwardOwnerResultCache()
			{
				PX_ASSERT(mFeaturePlan && mNumSurfaces > 0);
				if(mForwardOwnerResultCacheSurfaceSlots.size() != mNumSurfaces)
					mForwardOwnerResultCacheSurfaceSlots.resize(mNumSurfaces);
				for(PxU32 surfaceIndex = 0; surfaceIndex < mNumSurfaces;
					++surfaceIndex)
					mForwardOwnerResultCacheSurfaceSlots[surfaceIndex] = PX_MAX_U32;
				const PxU32 planBegin = mFeaturePlanRoundRobin ?
					mFeaturePlanTaskIndex : mFeaturePlanBegin;
				const PxU32 planEnd = mFeaturePlanRoundRobin ?
					mFeaturePlan->items.size() : mFeaturePlanEnd;
				const PxU32 planStride = mFeaturePlanRoundRobin ?
					mFeaturePlanTaskCount : 1u;
				PX_ASSERT(planStride > 0);
				PxU32 cachedSurfaceCount = 0;
				for(PxU32 planIndex = planBegin;
					planIndex < planEnd; planIndex += planStride)
				{
					const Dy::AvbdRigidTriangleSurfaceFeatureWorkItem& item =
						mFeaturePlan->items[planIndex];
					if(item.phase !=
						Dy::AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT ||
						item.surfaceIndex >= mNumSurfaces)
						continue;
					PxU32& slot =
						mForwardOwnerResultCacheSurfaceSlots[item.surfaceIndex];
					if(slot == PX_MAX_U32)
						slot = cachedSurfaceCount++;
				}
				if(cachedSurfaceCount == 0)
				{
					mForwardOwnerResultCacheEnabled = false;
					return;
				}
				const PxU64 requiredCapacity64 = PxU64(mNumParticles) *
					cachedSurfaceCount;
				PX_ASSERT(requiredCapacity64 <= PX_MAX_U32);
				const PxU32 requiredCapacity = PxU32(requiredCapacity64);
				if(mForwardOwnerResultCacheEntries.size() != requiredCapacity)
				{
					mForwardOwnerResultCacheEntries.resize(requiredCapacity);
					for(PxU32 index = 0; index < requiredCapacity; ++index)
						mForwardOwnerResultCacheEntries[index] = 0;
				}
				++mForwardOwnerResultCacheStamp;
				if(mForwardOwnerResultCacheStamp > (PX_MAX_U32 >> 1))
				{
					mForwardOwnerResultCacheStamp = 1;
					for(PxU32 index = 0; index < requiredCapacity; ++index)
						mForwardOwnerResultCacheEntries[index] = 0;
				}
				mForwardOwnerResultCache.configure(mForwardOwnerResultCacheEntries,
					mForwardOwnerResultCacheSurfaceSlots, mNumParticles,
					mNumSurfaces, cachedSurfaceCount,
					mForwardOwnerResultCacheStamp);
				mForwardOwnerResultCacheEnabled = true;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void runInternal() PX_OVERRIDE
			{
				PX_ASSERT(mParticles && mSurfaces && mBodies && mContacts &&
					mSweptContacts && mFeaturePlan && mFeatureContacts &&
					mTaskGraphContext);
				// Detailed leaf timing is diagnostic collision telemetry.  A normal
				// task candidate must not pay for clocks or timing atomics merely
				// because the task route is enabled.  mCollisionStats is supplied
				// only when the Scene-level cold telemetry policy is enabled.
				const bool detailedTelemetryEnabled = mCollisionStats != NULL;
				const PxU64 taskStartNanos = detailedTelemetryEnabled ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				mTaskGraphContext->beginRigidTriangleSurfaceContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginRigidTriangleSurfaceContactTask();
				const PxU64 currentSdfStartNanos = detailedTelemetryEnabled ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				Dy::avbdDetectSoftRigidTriangleSurfaceRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mSurfaces, mNumSurfaces, *mContacts,
					mQueryScratch.triangleBvhQueryCandidates,
					mMargin, mBodies, mNumBodies,
					mCollisionStats);
				const PxU64 sweptSdfStartNanos = detailedTelemetryEnabled ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				Dy::avbdDetectSoftRigidTriangleSurfaceSweptRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mSurfaces, mNumSurfaces, *mSweptContacts, mQueryScratch,
					mMargin, mBodies, mNumBodies, mCollisionStats);
				const PxU64 featureStartNanos = detailedTelemetryEnabled ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				Dy::AvbdRigidTriangleSurfaceFeaturePlanRangeTiming
					featurePlanTiming;
				const bool featureSweptSubstageTimingEnabled =
					detailedTelemetryEnabled &&
					Dy::avbdUseRigidTriangleSurfaceFeatureSweptSubstageTiming();
				const bool featureForwardOwnerQueryStatsEnabled =
					mForwardOwnerQueryStatsEnabled;
				const bool featureDiscreteQueryStatsEnabled =
					mDiscreteQueryStatsEnabled;
				const bool featureDiscreteBodyLocalBoundsCullEnabled =
					mDiscreteBodyLocalBoundsCullEnabled;
				const bool featureForwardOwnerResultCacheEnabled =
					mForwardOwnerResultCacheEnabled;
				Dy::AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming
					featureSweptSubstageTiming;
				Dy::AvbdRigidTriangleSurfaceDiscreteOGCQueryStats
					featureDiscreteQueryStats;
				if(mFeaturePlanRowPrivateOutputs)
				{
					PX_ASSERT(mFeaturePlanOutputs);
					const PxU32 featurePlanBegin = mFeaturePlanRoundRobin ?
						mFeaturePlanTaskIndex : mFeaturePlanBegin;
					const PxU32 featurePlanEnd = mFeaturePlanRoundRobin ?
						mFeaturePlan->items.size() : mFeaturePlanEnd;
					const PxU32 featurePlanStride = mFeaturePlanRoundRobin ?
						mFeaturePlanTaskCount : 1u;
					PX_ASSERT(featurePlanStride > 0);
					for(PxU32 planIndex = featurePlanBegin;
						planIndex < featurePlanEnd; planIndex += featurePlanStride)
						Dy::avbdDetectSoftRigidTriangleSurfaceOGCFeaturePlanRange(
							mParticles, mNumParticles, mSurfaces, mNumSurfaces,
							mBodies, mNumBodies, *mFeaturePlan,
							planIndex, planIndex + 1,
							(*mFeaturePlanOutputs)[planIndex], mQueryScratch,
							mMargin, mCollisionStats, detailedTelemetryEnabled ?
								&featurePlanTiming : NULL,
							featureSweptSubstageTimingEnabled ?
								&featureSweptSubstageTiming : NULL,
							featureForwardOwnerQueryStatsEnabled ?
								&mForwardOwnerQueryStats : NULL,
							featureForwardOwnerResultCacheEnabled ?
								&mForwardOwnerResultCache : NULL,
							featureDiscreteQueryStatsEnabled ?
								&featureDiscreteQueryStats : NULL,
							featureDiscreteBodyLocalBoundsCullEnabled);
				}
				else if(mFeaturePlanBegin < mFeaturePlanEnd)
					Dy::avbdDetectSoftRigidTriangleSurfaceOGCFeaturePlanRange(
						mParticles, mNumParticles, mSurfaces, mNumSurfaces,
						mBodies, mNumBodies, *mFeaturePlan,
						mFeaturePlanBegin, mFeaturePlanEnd, *mFeatureContacts,
						mQueryScratch, mMargin, mCollisionStats,
						detailedTelemetryEnabled ? &featurePlanTiming : NULL,
						featureSweptSubstageTimingEnabled ?
							&featureSweptSubstageTiming : NULL,
						featureForwardOwnerQueryStatsEnabled ?
							&mForwardOwnerQueryStats : NULL,
						featureForwardOwnerResultCacheEnabled ?
							&mForwardOwnerResultCache : NULL,
						featureDiscreteQueryStatsEnabled ?
							&featureDiscreteQueryStats : NULL,
						featureDiscreteBodyLocalBoundsCullEnabled);
				const PxU64 featureEndNanos = detailedTelemetryEnabled ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				mScene.mStandaloneTaskGraphTelemetry.
					endRigidTriangleSurfaceContactTask();
				mTaskGraphContext->endRigidTriangleSurfaceContactTask();
				if(detailedTelemetryEnabled)
				{
					const PxU64 taskEndNanos =
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u;
					mTaskGraphContext->recordRigidTriangleSurfaceContactTaskLeafWallTimes(
						sweptSdfStartNanos - currentSdfStartNanos,
						featureStartNanos - sweptSdfStartNanos,
						featureEndNanos - featureStartNanos);
					mTaskGraphContext->recordRigidTriangleSurfaceContactFeaturePlanTaskLeafWallTimes(
						featurePlanTiming.sweptEdgeNanos,
						featurePlanTiming.sweptTriangleNanos,
						featurePlanTiming.discreteEdgeNanos,
						featurePlanTiming.discreteTriangleNanos);
					if(featureSweptSubstageTimingEnabled)
						mTaskGraphContext->recordRigidTriangleSurfaceContactFeatureSweptSubstageWallTimes(
							featureSweptSubstageTiming.sweptEdgeForwardOwnerNanos,
							featureSweptSubstageTiming.sweptEdgeBvhRecoveryNanos,
							featureSweptSubstageTiming.sweptEdgeNarrowPhaseNanos,
							featureSweptSubstageTiming.sweptTriangleForwardOwnerNanos,
							featureSweptSubstageTiming.sweptTriangleBvhRecoveryNanos,
							featureSweptSubstageTiming.sweptTriangleNarrowPhaseNanos);
					mTaskGraphContext->recordRigidTriangleSurfaceContactTaskWallTime(
						taskEndNanos - taskStartNanos);
				}
				if(featureForwardOwnerQueryStatsEnabled)
					mTaskGraphContext->recordRigidTriangleSurfaceContactFeatureForwardOwnerQueries(
						mForwardOwnerQueryStats.queryCalls,
						mForwardOwnerQueryStats.uniqueQueries);
				if(featureForwardOwnerResultCacheEnabled)
					mTaskGraphContext->recordRigidTriangleSurfaceContactFeatureForwardOwnerCache(
						mForwardOwnerResultCache.hits,
						mForwardOwnerResultCache.misses);
				if(featureDiscreteQueryStatsEnabled)
					mTaskGraphContext->recordRigidTriangleSurfaceContactFeatureDiscreteQueryStats(
						featureDiscreteQueryStats.edgeBvhQueries,
						featureDiscreteQueryStats.edgeBvhTriangleCandidates,
						featureDiscreteQueryStats.edgeFeatureCandidates,
						featureDiscreteQueryStats.edgeFallbackQueries,
						featureDiscreteQueryStats.triangleBvhQueries,
						featureDiscreteQueryStats.triangleBvhTriangleCandidates,
						featureDiscreteQueryStats.triangleFeatureCandidates,
						featureDiscreteQueryStats.triangleFallbackQueries);
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidTriangleSurfaceContactTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidTriangleSurfaceContact";
			}

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32						mPoolIndex;
			const Dy::AvbdSoftParticle*	mParticles;
			PxU32						mNumParticles;
			PxU32						mParticleBegin;
			PxU32						mParticleEnd;
			const Dy::AvbdRigidTriangleSurface*	mSurfaces;
			PxU32						mNumSurfaces;
			const Dy::AvbdSoftBody*		mBodies;
			PxU32						mNumBodies;
			PxArray<Dy::AvbdSoftContact>*	mContacts;
			PxArray<Dy::AvbdSoftContact>*	mSweptContacts;
			const Dy::AvbdRigidTriangleSurfaceFeaturePlan*
										mFeaturePlan;
			PxU32						mFeaturePlanBegin;
			PxU32						mFeaturePlanEnd;
			PxArray<Dy::AvbdSoftContact>*	mFeatureContacts;
			PxArray<PxArray<Dy::AvbdSoftContact> >*
										mFeaturePlanOutputs;
			bool						mFeaturePlanRowPrivateOutputs;
			bool						mFeaturePlanRoundRobin;
			PxU32						mFeaturePlanTaskIndex;
			PxU32						mFeaturePlanTaskCount;
			Dy::AvbdRigidTriangleSurfaceQueryScratch mQueryScratch;
			PxArray<PxU32>						mForwardOwnerQueryStamps;
			Dy::AvbdRigidTriangleSurfaceForwardOwnerQueryStats
										mForwardOwnerQueryStats;
			bool						mForwardOwnerQueryStatsEnabled;
			bool						mDiscreteQueryStatsEnabled;
			bool						mDiscreteBodyLocalBoundsCullEnabled;
			PxU32						mForwardOwnerQueryStamp;
			PxArray<PxU32>						mForwardOwnerResultCacheEntries;
			PxArray<PxU32>						mForwardOwnerResultCacheSurfaceSlots;
			Dy::AvbdRigidTriangleSurfaceForwardOwnerResultCache
										mForwardOwnerResultCache;
			bool						mForwardOwnerResultCacheEnabled;
			PxU32						mForwardOwnerResultCacheStamp;
			PxReal						mMargin;
			Dy::AvbdSoftCollisionStats*	mCollisionStats;
			Dy::AvbdDynamicsContext*		mTaskGraphContext;
		};

		class RigidTriangleSurfaceContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			RigidTriangleSurfaceContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mScene.mStandaloneTaskGraphTelemetry.
					recordRigidTriangleSurfaceContactFanIn();
				mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleRigidTriangleSurfaceContactFinishTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftRigidTriangleSurfaceContactFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.9d consumes immutable soft-pair plan ranges after the parent has
		// completed the only mutable plan/refit epoch. Every child owns its
		// contact stream, statistics and VF/EE query scratch.
		class SoftPairContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			SoftPairContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mParticles(NULL), mNumParticles(0), mBodies(NULL),
				  mNumBodies(0), mRefitWorkspace(NULL),
				  mPlanBegin(0), mPlanEnd(0), mContacts(NULL),
				  mUseSurfaceTriangleBvh(false), mCollisionStats(NULL),
				  mTaskGraphContext(NULL)
			{
			}

			void reserveQueryScratch(PxU32 edgeCountA, PxU32 edgeCountB,
				PxU32 triangleCandidateCapacity)
			{
				mQueryScratch.reserve(
					edgeCountA, edgeCountB, triangleCandidateCapacity);
			}

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, const Dy::AvbdSoftBody* bodies,
				PxU32 numBodies,
				const Dy::AvbdSoftContactWorkspace& refitWorkspace,
				PxU32 planBegin, PxU32 planEnd,
				PxArray<Dy::AvbdSoftContact>& contacts,
				const Dy::AvbdOGCParams& params,
				bool useSurfaceTriangleBvh,
				Dy::AvbdSoftCollisionStats* collisionStats,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(particles && bodies && planBegin < planEnd);
				PX_ASSERT(planEnd <= refitWorkspace.softPairDetectionPlan.size());
				mParticles = particles;
				mNumParticles = numParticles;
				mBodies = bodies;
				mNumBodies = numBodies;
				mRefitWorkspace = &refitWorkspace;
				mPlanBegin = planBegin;
				mPlanEnd = planEnd;
				mContacts = &contacts;
				mParams = params;
				mUseSurfaceTriangleBvh = useSurfaceTriangleBvh;
				mCollisionStats = collisionStats;
				mTaskGraphContext = &taskGraphContext;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void runInternal() PX_OVERRIDE
			{
				PX_ASSERT(mParticles && mBodies && mRefitWorkspace && mContacts &&
					mTaskGraphContext);
				mTaskGraphContext->beginSoftPairContactTask();
				mScene.mStandaloneTaskGraphTelemetry.beginSoftPairContactTask();
				mQueryScratch.edgeBoundsA.clear();
				mQueryScratch.edgeBoundsB.clear();
				mQueryScratch.triangleCandidates.clear();
				Dy::avbdDetectSoftSoftOGCPlanRange(
					mParticles, mNumParticles, mBodies, mNumBodies,
					*mRefitWorkspace, NULL, mQueryScratch,
					mUseSurfaceTriangleBvh, mPlanBegin, mPlanEnd, *mContacts,
					mParams, mCollisionStats);
				mScene.mStandaloneTaskGraphTelemetry.endSoftPairContactTask();
				mTaskGraphContext->endSoftPairContactTask();
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleSoftPairContactTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftPairContact";
			}

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32						mPoolIndex;
			const Dy::AvbdSoftParticle*	mParticles;
			PxU32						mNumParticles;
			const Dy::AvbdSoftBody*		mBodies;
			PxU32						mNumBodies;
			const Dy::AvbdSoftContactWorkspace*	mRefitWorkspace;
			PxU32						mPlanBegin;
			PxU32						mPlanEnd;
			PxArray<Dy::AvbdSoftContact>*	mContacts;
			Dy::AvbdOGCParams				mParams;
			Dy::AvbdSoftSoftPairQueryScratch	mQueryScratch;
			bool						mUseSurfaceTriangleBvh;
			Dy::AvbdSoftCollisionStats*	mCollisionStats;
			Dy::AvbdDynamicsContext*		mTaskGraphContext;
		};

		class SoftPairContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			SoftPairContactFinishTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mScene.mStandaloneTaskGraphTelemetry.recordSoftPairContactFanIn();
				// The existing mutually-exclusive contact continuation dispatches
				// to the pending soft-pair transaction before any smooth-SDF leaf.
				mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleSoftPairContactFinishTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSoftPairContactFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&				mOwner;
			PxU32				mPoolIndex;
		};

		// P5.10b consumes a single body whose parent has already prepared its
		// self stress and triangle/edge BVH epoch. The child may own only a
		// contiguous VF or EE outer range; its query workspace is never shared.
		class SelfBvhContactTask : public Cm::Task, public PxUserAllocated
		{
		public:
			SelfBvhContactTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mParticles(NULL), mBody(NULL), mAdjacency(NULL),
				  mParentWorkspace(NULL), mSoftBodyIndex(0),
				  mVertexBegin(0), mVertexEnd(0), mEdgeBegin(0), mEdgeEnd(0),
				  mContacts(NULL), mCollisionStats(NULL), mTaskGraphContext(NULL)
			{
			}

			void reserveQueryScratch(const Dy::AvbdSoftBody& body)
			{
				mRangeWorkspace.reserveSelfCollisionSweep(
					body.compiled.tetElements.size(),
					body.compiled.surfaceTriangles.size() / 3,
					body.compiled.surfaceVertices.size(),
					body.compiled.surfaceEdges.size());
			}

			void configure(const Dy::AvbdSoftParticle* particles,
				const Dy::AvbdSoftBody& body, PxU32 softBodyIndex,
				const Dy::AvbdSelfCollisionAdjacency& adjacency,
				const Dy::AvbdSoftContactWorkspace& parentWorkspace,
				PxU32 vertexBegin, PxU32 vertexEnd,
				PxU32 edgeBegin, PxU32 edgeEnd,
				PxArray<Dy::AvbdSoftContact>& contacts,
				const Dy::AvbdOGCParams& params,
				Dy::AvbdSoftCollisionStats* collisionStats,
				Dy::AvbdDynamicsContext& taskGraphContext)
			{
				PX_ASSERT(particles && (vertexBegin < vertexEnd || edgeBegin < edgeEnd));
				mParticles = particles;
				mBody = &body;
				mAdjacency = &adjacency;
				mParentWorkspace = &parentWorkspace;
				mSoftBodyIndex = softBodyIndex;
				mVertexBegin = vertexBegin;
				mVertexEnd = vertexEnd;
				mEdgeBegin = edgeBegin;
				mEdgeEnd = edgeEnd;
				mContacts = &contacts;
				mParams = params;
				mCollisionStats = collisionStats;
				mTaskGraphContext = &taskGraphContext;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void runInternal() PX_OVERRIDE
			{
				PX_ASSERT(mParticles && mBody && mAdjacency && mParentWorkspace &&
					mContacts && mTaskGraphContext);
				mTaskGraphContext->beginSelfBvhContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginSelfBvhContactTask();
				Dy::avbdDetectSelfCollisionOGCBvhRange(
					mParticles, *mBody, mSoftBodyIndex, *mAdjacency,
					*mParentWorkspace, mRangeWorkspace,
					mVertexBegin, mVertexEnd, mEdgeBegin, mEdgeEnd,
					*mContacts, mParams, mCollisionStats);
				mScene.mStandaloneTaskGraphTelemetry.
					endSelfBvhContactTask();
				mTaskGraphContext->endSelfBvhContactTask();
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleSelfBvhContactTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSelfBvhContact";
			}

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32					mPoolIndex;
			const Dy::AvbdSoftParticle*	mParticles;
			const Dy::AvbdSoftBody*		mBody;
			const Dy::AvbdSelfCollisionAdjacency*	mAdjacency;
			const Dy::AvbdSoftContactWorkspace*	mParentWorkspace;
			PxU32					mSoftBodyIndex;
			PxU32					mVertexBegin;
			PxU32					mVertexEnd;
			PxU32					mEdgeBegin;
			PxU32					mEdgeEnd;
			PxArray<Dy::AvbdSoftContact>*	mContacts;
			Dy::AvbdOGCParams				mParams;
			Dy::AvbdSoftContactWorkspace	mRangeWorkspace;
			Dy::AvbdSoftCollisionStats*	mCollisionStats;
			Dy::AvbdDynamicsContext*		mTaskGraphContext;
		};

		class SelfBvhContactFinishTask : public Cm::Task, public PxUserAllocated
		{
		public:
			SelfBvhContactFinishTask(PxU64 contextId, AvbdCpuSoftScene& scene,
				Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mScene.mStandaloneTaskGraphTelemetry.
					recordSelfBvhContactFanIn();
				mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleSelfBvhContactFinishTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuSelfBvhContactFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&			mOwner;
			PxU32			mPoolIndex;
		};

		// One aggregate child owns matching disjoint ranges for all three source
		// families.  It writes one private stream per canonical source substage;
		// the parent never merges by child completion order.
		class StaticWorldSelfOgcContactTask :
			public Cm::Task, public PxUserAllocated
		{
		public:
			StaticWorldSelfOgcContactTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
				  mParticles(NULL), mNumParticles(0), mParticleBegin(0),
				  mParticleEnd(0), mPlanes(NULL), mNumPlanes(0), mBoxes(NULL),
				  mNumBoxes(0), mPreviousContacts(NULL),
				  mNumPreviousContacts(0), mBody(NULL), mAdjacency(NULL),
				  mPreparedWorkspace(NULL), mVertexBegin(0), mVertexEnd(0),
				  mEdgeBegin(0), mEdgeEnd(0), mWorldContacts(NULL),
				  mBoxContacts(NULL), mBoxSweptContacts(NULL),
				  mSelfVertexContacts(NULL), mSelfEdgeContacts(NULL),
				  mTaskStats(NULL), mTaskGraphContext(NULL), mMargin(0.0f)
			{
			}

			void reserveQueryScratch(const Dy::AvbdSoftBody& body)
			{
				mRangeWorkspace.reserveSelfCollisionSweep(
					body.compiled.tetElements.size(),
					body.compiled.surfaceTriangles.size() / 3,
					body.compiled.surfaceVertices.size(),
					body.compiled.surfaceEdges.size());
			}

			void configure(const Dy::AvbdSoftParticle* particles,
				PxU32 numParticles, PxU32 particleBegin, PxU32 particleEnd,
				const Dy::AvbdWorldPlane* planes, PxU32 numPlanes,
				const Dy::AvbdRigidBox* boxes, PxU32 numBoxes,
				const Dy::AvbdSoftContact* previousContacts,
				PxU32 numPreviousContacts, const Dy::AvbdSoftBody& body,
				const Dy::AvbdSelfCollisionAdjacency& adjacency,
				const Dy::AvbdSoftContactWorkspace& preparedWorkspace,
				PxU32 vertexBegin, PxU32 vertexEnd,
				PxU32 edgeBegin, PxU32 edgeEnd,
				PxArray<Dy::AvbdSoftContact>& worldContacts,
				PxArray<Dy::AvbdSoftContact>& boxContacts,
				PxArray<Dy::AvbdSoftContact>& boxSweptContacts,
				PxArray<Dy::AvbdSoftContact>& selfVertexContacts,
				PxArray<Dy::AvbdSoftContact>& selfEdgeContacts,
				const Dy::AvbdOGCParams& params,
				Dy::AvbdSoftCollisionStats* taskStats,
				Dy::AvbdDynamicsContext& taskGraphContext, PxReal margin)
			{
				PX_ASSERT(particles && particleBegin < particleEnd);
				PX_ASSERT(particleEnd <= numParticles && planes && numPlanes > 0);
				PX_ASSERT(boxes && numBoxes > 0 && vertexBegin < vertexEnd);
				PX_ASSERT(edgeBegin < edgeEnd);
				mParticles = particles;
				mNumParticles = numParticles;
				mParticleBegin = particleBegin;
				mParticleEnd = particleEnd;
				mPlanes = planes;
				mNumPlanes = numPlanes;
				mBoxes = boxes;
				mNumBoxes = numBoxes;
				mPreviousContacts = previousContacts;
				mNumPreviousContacts = numPreviousContacts;
				mBody = &body;
				mAdjacency = &adjacency;
				mPreparedWorkspace = &preparedWorkspace;
				mVertexBegin = vertexBegin;
				mVertexEnd = vertexEnd;
				mEdgeBegin = edgeBegin;
				mEdgeEnd = edgeEnd;
				mWorldContacts = &worldContacts;
				mBoxContacts = &boxContacts;
				mBoxSweptContacts = &boxSweptContacts;
				mSelfVertexContacts = &selfVertexContacts;
				mSelfEdgeContacts = &selfEdgeContacts;
				mParams = params;
				mTaskStats = taskStats;
				mTaskGraphContext = &taskGraphContext;
				mMargin = margin;
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const { return mPoolIndex; }

			virtual void runInternal() PX_OVERRIDE
			{
				PX_ASSERT(mParticles && mPlanes && mBoxes && mBody &&
					mAdjacency && mPreparedWorkspace && mWorldContacts &&
					mBoxContacts && mBoxSweptContacts && mSelfVertexContacts &&
					mSelfEdgeContacts && mTaskGraphContext);
				mTaskGraphContext->beginWorldPlaneContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginWorldPlaneContactTask();
				Dy::avbdDetectSoftWorldPlaneContactsRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mPlanes, mNumPlanes, *mWorldContacts, mMargin, mBody, 1);
				mScene.mStandaloneTaskGraphTelemetry.
					endWorldPlaneContactTask();
				mTaskGraphContext->endWorldPlaneContactTask();
				mTaskGraphContext->beginRigidBoxSdfContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginRigidBoxSdfContactTask();
				Dy::avbdDetectSoftRigidSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mBoxes, mNumBoxes, *mBoxContacts, mMargin,
					mPreviousContacts, mNumPreviousContacts, mBody, 1);
				Dy::avbdDetectSoftRigidSweptSDFRange(
					mParticles, mNumParticles, mParticleBegin, mParticleEnd,
					mBoxes, mNumBoxes, *mBoxSweptContacts, mMargin,
					mBody, 1);
				mScene.mStandaloneTaskGraphTelemetry.
					endRigidBoxSdfContactTask();
				mTaskGraphContext->endRigidBoxSdfContactTask();
				mTaskGraphContext->beginSelfBvhContactTask();
				mScene.mStandaloneTaskGraphTelemetry.
					beginSelfBvhContactTask();
				Dy::avbdDetectSelfCollisionOGCBvhRange(
					mParticles, *mBody, 0, *mAdjacency, *mPreparedWorkspace,
					mRangeWorkspace, mVertexBegin, mVertexEnd, 0, 0,
					*mSelfVertexContacts, mParams, mTaskStats);
				Dy::avbdDetectSelfCollisionOGCBvhRange(
					mParticles, *mBody, 0, *mAdjacency, *mPreparedWorkspace,
					mRangeWorkspace, 0, 0, mEdgeBegin, mEdgeEnd,
					*mSelfEdgeContacts, mParams, mTaskStats);
				mScene.mStandaloneTaskGraphTelemetry.
					endSelfBvhContactTask();
				mTaskGraphContext->endSelfBvhContactTask();
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleStaticWorldSelfOgcContactTask(mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuStaticWorldSelfOgcContact";
			}

		private:
			AvbdCpuSoftScene&				mScene;
			PxU32					mPoolIndex;
			const Dy::AvbdSoftParticle*	mParticles;
			PxU32					mNumParticles;
			PxU32					mParticleBegin;
			PxU32					mParticleEnd;
			const Dy::AvbdWorldPlane*	mPlanes;
			PxU32					mNumPlanes;
			const Dy::AvbdRigidBox*	mBoxes;
			PxU32					mNumBoxes;
			const Dy::AvbdSoftContact*	mPreviousContacts;
			PxU32					mNumPreviousContacts;
			const Dy::AvbdSoftBody*	mBody;
			const Dy::AvbdSelfCollisionAdjacency*	mAdjacency;
			const Dy::AvbdSoftContactWorkspace*	mPreparedWorkspace;
			PxU32					mVertexBegin;
			PxU32					mVertexEnd;
			PxU32					mEdgeBegin;
			PxU32					mEdgeEnd;
			PxArray<Dy::AvbdSoftContact>*	mWorldContacts;
			PxArray<Dy::AvbdSoftContact>*	mBoxContacts;
			PxArray<Dy::AvbdSoftContact>*	mBoxSweptContacts;
			PxArray<Dy::AvbdSoftContact>*	mSelfVertexContacts;
			PxArray<Dy::AvbdSoftContact>*	mSelfEdgeContacts;
			Dy::AvbdOGCParams				mParams;
			Dy::AvbdSoftContactWorkspace	mRangeWorkspace;
			Dy::AvbdSoftCollisionStats*	mTaskStats;
			Dy::AvbdDynamicsContext*	mTaskGraphContext;
			PxReal					mMargin;
		};

		// A single parent fan-in joins the private plane, box, and self-BVH
		// streams.  It deliberately uses the existing mutually-exclusive
		// smooth-rigid continuation slot: the pending aggregate transaction is
		// dispatched before any individual smooth-rigid leaf, while this distinct
		// pool keeps its lifetime separate from those source-specific pools.
		class StaticWorldSelfOgcContactFinishTask :
			public Cm::Task, public PxUserAllocated
		{
		public:
			StaticWorldSelfOgcContactFinishTask(PxU64 contextId,
				AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
				: Cm::Task(contextId), mScene(scene), mOwner(owner),
				  mPoolIndex(poolIndex)
			{
			}

			virtual void runInternal() PX_OVERRIDE
			{
				mScene.mStandaloneTaskGraphTelemetry.
					recordWorldPlaneContactFanIn();
				mScene.mStandaloneTaskGraphTelemetry.
					recordRigidBoxSdfContactFanIn();
				mScene.mStandaloneTaskGraphTelemetry.
					recordSelfBvhContactFanIn();
				mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
			}

			PX_FORCE_INLINE PxU32 getPoolIndex() const
			{
				return mPoolIndex;
			}

			virtual void release() PX_OVERRIDE
			{
				PxBaseTask* const continuation = mCont;
				mCont = NULL;
				mScene.recycleStaticWorldSelfOgcContactFinishTask(
					mPoolIndex);
				if(continuation)
					continuation->removeReference();
			}

			virtual const char* getName() const PX_OVERRIDE
			{
				return "ScScene.avbdCpuStaticWorldSelfOgcContactFinish";
			}

		private:
			AvbdCpuSoftScene&	mScene;
			Scene&			mOwner;
			PxU32			mPoolIndex;
		};

		// P3 pre-solve preparation is deliberately stateful even while it is
		// still resumed synchronously below.  The later prediction fan-in must
		// occur after this exact prefix (initial OGC detection and the iteration
		// plan), but before avbdStepSoftBodies() performs its predicted-position
		// redetection.  Keeping the plan explicit prevents a future continuation
		// from recomputing live Scene state after child tasks have started.
		struct ComponentFallbackPlan
		{
			ComponentFallbackPlan()
				: outerIterations(1), innerIterations(1),
				  totalPositionIterations(1),
				  initialContactWorkspaceGrowthEvents(0),
				  initialContactWorkspaceGrowthBytes(0),
				  initialContactSweepScratchGrowthEvents(0),
				  initialContactSweepScratchGrowthBytes(0),
				  initialContactOutputGrowthEvents(0),
				  initialContactOutputGrowthBytes(0)
			{
			}

			PxU32	outerIterations;
			PxU32	innerIterations;
			PxU32	totalPositionIterations;
			PxU64	initialContactWorkspaceGrowthEvents;
			PxU64	initialContactWorkspaceGrowthBytes;
			PxU64	initialContactSweepScratchGrowthEvents;
			PxU64	initialContactSweepScratchGrowthBytes;
			PxU64	initialContactOutputGrowthEvents;
			PxU64	initialContactOutputGrowthBytes;
		};

		// Keep the public Scene taskgraph counters at the task boundary rather
		// than in Dy::AvbdDynamicsContext.  The solver context intentionally has
		// no profiling atomics on its hot path, while these counters let callers
		// distinguish an actually dispatched relaxed-color solve from the serial
		// component fallback.  They are reset before a new Scene graph is
		// admitted and read only after its continuation has joined.
		struct StandaloneTaskGraphTelemetry
		{
			// Source-local OGC counters deliberately live at the Scene task
			// boundary. Dy::AvbdDynamicsContext exposes compatibility hooks for
			// these sources, but those hooks are intentionally no-ops so they do
			// not add atomics to the solver context's hot path.
			struct OGCSourceTaskTelemetry
			{
				std::atomic<PxU32>	submittedTasks;
				std::atomic<PxU32>	completedTasks;
				std::atomic<PxU32>	activeTasks;
				std::atomic<PxU32>	peakActiveTasks;
				std::atomic<PxU32>	fanIns;
				std::atomic<PxU32>	serialFallbacks;

				OGCSourceTaskTelemetry()
				{
					reset();
				}

				static void atomicMax(
					std::atomic<PxU32>& target, PxU32 value)
				{
					PxU32 observed = target.load(std::memory_order_relaxed);
					while(observed < value && !target.compare_exchange_weak(
						observed, value, std::memory_order_relaxed,
						std::memory_order_relaxed))
					{
					}
				}

				void reset()
				{
					submittedTasks.store(0, std::memory_order_relaxed);
					completedTasks.store(0, std::memory_order_relaxed);
					activeTasks.store(0, std::memory_order_relaxed);
					peakActiveTasks.store(0, std::memory_order_relaxed);
					fanIns.store(0, std::memory_order_relaxed);
					serialFallbacks.store(0, std::memory_order_relaxed);
				}

				void recordTasksSubmitted(PxU32 taskCount)
				{
					submittedTasks.fetch_add(
						taskCount, std::memory_order_relaxed);
				}

				void beginTask()
				{
					const PxU32 active = activeTasks.fetch_add(
						1, std::memory_order_relaxed) + 1;
					atomicMax(peakActiveTasks, active);
				}

				void endTask()
				{
					activeTasks.fetch_sub(1, std::memory_order_relaxed);
					completedTasks.fetch_add(1, std::memory_order_relaxed);
				}

				void recordFanIn()
				{
					fanIns.fetch_add(1, std::memory_order_relaxed);
				}

				void recordSerialFallback()
				{
					serialFallbacks.fetch_add(1, std::memory_order_relaxed);
				}
			};

			std::atomic<PxU32>	requestedDispatcherWorkers;
			std::atomic<PxU32>	submittedSolveTasks;
			std::atomic<PxU32>	completedSolveTasks;
			std::atomic<PxU32>	activeSolveTasks;
			std::atomic<PxU32>	peakActiveSolveTasks;
			std::atomic<PxU32>	serialSolveTasks;
			std::atomic<PxU32>	pureSoftEligibleIslands;
			std::atomic<PxU32>	pureSoftEligibleParticles;
			std::atomic<PxU32>	submittedCausalLayerTasks;
			std::atomic<PxU32>	completedCausalLayerTasks;
			std::atomic<PxU32>	activeCausalLayerTasks;
			std::atomic<PxU32>	peakActiveCausalLayerTasks;
			std::atomic<PxU32>	causalLayerFanIns;
			std::atomic<PxU32>	serialCausalLayerFallbacks;
			std::atomic<PxU32>	maxCausalLayerOccupancy;
			std::atomic<PxU64>	totalCausalLayerOccupancy;
			OGCSourceTaskTelemetry	worldPlaneContact;
			OGCSourceTaskTelemetry	rigidBoxSdfContact;
			OGCSourceTaskTelemetry	rigidSphereSdfContact;
			OGCSourceTaskTelemetry	rigidCapsuleSdfContact;
			OGCSourceTaskTelemetry	rigidConvexSdfContact;
			OGCSourceTaskTelemetry	rigidTriangleSurfaceContact;
			OGCSourceTaskTelemetry	softPairContact;
			OGCSourceTaskTelemetry	selfBvhContact;

			StandaloneTaskGraphTelemetry()
			{
				reset();
			}

			static void atomicMax(std::atomic<PxU32>& target, PxU32 value)
			{
				PxU32 observed = target.load(std::memory_order_relaxed);
				while(observed < value && !target.compare_exchange_weak(
					observed, value, std::memory_order_relaxed,
					std::memory_order_relaxed))
				{
				}
			}

			void reset(PxU32 dispatcherWorkers = 0)
			{
				requestedDispatcherWorkers.store(
					dispatcherWorkers, std::memory_order_relaxed);
				submittedSolveTasks.store(0, std::memory_order_relaxed);
				completedSolveTasks.store(0, std::memory_order_relaxed);
				activeSolveTasks.store(0, std::memory_order_relaxed);
				peakActiveSolveTasks.store(0, std::memory_order_relaxed);
				serialSolveTasks.store(0, std::memory_order_relaxed);
				pureSoftEligibleIslands.store(0, std::memory_order_relaxed);
				pureSoftEligibleParticles.store(0, std::memory_order_relaxed);
				submittedCausalLayerTasks.store(0, std::memory_order_relaxed);
				completedCausalLayerTasks.store(0, std::memory_order_relaxed);
				activeCausalLayerTasks.store(0, std::memory_order_relaxed);
				peakActiveCausalLayerTasks.store(0, std::memory_order_relaxed);
				causalLayerFanIns.store(0, std::memory_order_relaxed);
				serialCausalLayerFallbacks.store(0, std::memory_order_relaxed);
				maxCausalLayerOccupancy.store(0, std::memory_order_relaxed);
				totalCausalLayerOccupancy.store(0, std::memory_order_relaxed);
				worldPlaneContact.reset();
				rigidBoxSdfContact.reset();
				rigidSphereSdfContact.reset();
				rigidCapsuleSdfContact.reset();
				rigidConvexSdfContact.reset();
				rigidTriangleSurfaceContact.reset();
				softPairContact.reset();
				selfBvhContact.reset();
			}

			void recordSolveSubmission(
				PxU32 dispatcherWorkers, PxU32 particleCount)
			{
				requestedDispatcherWorkers.store(
					dispatcherWorkers, std::memory_order_relaxed);
				submittedSolveTasks.fetch_add(1, std::memory_order_relaxed);
				pureSoftEligibleIslands.fetch_add(1, std::memory_order_relaxed);
				pureSoftEligibleParticles.fetch_add(
					particleCount, std::memory_order_relaxed);
				// The Scene top-level task is the only solve owner.  Counting it at
				// submission also covers the interval while it is queued; child
				// primal tasks have their own active counter below.
				const PxU32 active = activeSolveTasks.fetch_add(
					1, std::memory_order_relaxed) + 1;
				atomicMax(peakActiveSolveTasks, active);
			}

			void recordSerialSolve(PxU32 dispatcherWorkers, PxU32 particleCount)
			{
				requestedDispatcherWorkers.store(
					dispatcherWorkers, std::memory_order_relaxed);
				serialSolveTasks.fetch_add(1, std::memory_order_relaxed);
				pureSoftEligibleIslands.fetch_add(1, std::memory_order_relaxed);
				pureSoftEligibleParticles.fetch_add(
					particleCount, std::memory_order_relaxed);
			}

			void beginSolveTask()
			{
				const PxU32 active = activeSolveTasks.fetch_add(
					1, std::memory_order_relaxed) + 1;
				atomicMax(peakActiveSolveTasks, active);
			}

			void endSolveTask()
			{
				PxU32 active = activeSolveTasks.load(
					std::memory_order_relaxed);
				while(active && !activeSolveTasks.compare_exchange_weak(
					active, active - 1, std::memory_order_relaxed,
					std::memory_order_relaxed))
				{
				}
				PxU32 completed = completedSolveTasks.load(
					std::memory_order_relaxed);
				const PxU32 submitted = submittedSolveTasks.load(
					std::memory_order_relaxed);
				while(completed < submitted && !completedSolveTasks.
					compare_exchange_weak(completed, completed + 1,
						std::memory_order_relaxed,
						std::memory_order_relaxed))
				{
				}
			}

			void recordCausalLayerTasksSubmitted(
				PxU32 taskCount, PxU32 occupancy)
			{
				submittedCausalLayerTasks.fetch_add(
					taskCount, std::memory_order_relaxed);
				atomicMax(maxCausalLayerOccupancy, occupancy);
				totalCausalLayerOccupancy.fetch_add(
					occupancy, std::memory_order_relaxed);
			}

			void beginCausalLayerTask()
			{
				const PxU32 active = activeCausalLayerTasks.fetch_add(
					1, std::memory_order_relaxed) + 1;
				atomicMax(peakActiveCausalLayerTasks, active);
			}

			void endCausalLayerTask()
			{
				activeCausalLayerTasks.fetch_sub(1, std::memory_order_relaxed);
				completedCausalLayerTasks.fetch_add(1, std::memory_order_relaxed);
			}

			void recordCausalLayerFanIn()
			{
				causalLayerFanIns.fetch_add(1, std::memory_order_relaxed);
			}

			void recordSerialCausalLayerFallback()
			{
				serialCausalLayerFallbacks.fetch_add(
					1, std::memory_order_relaxed);
			}

			void recordWorldPlaneContactTasksSubmitted(PxU32 taskCount)
			{
				worldPlaneContact.recordTasksSubmitted(taskCount);
			}

			void beginWorldPlaneContactTask()
			{
				worldPlaneContact.beginTask();
			}

			void endWorldPlaneContactTask()
			{
				worldPlaneContact.endTask();
			}

			void recordWorldPlaneContactFanIn()
			{
				worldPlaneContact.recordFanIn();
			}

			void recordSerialWorldPlaneContactFallback()
			{
				worldPlaneContact.recordSerialFallback();
			}

			void recordRigidBoxSdfContactTasksSubmitted(PxU32 taskCount)
			{
				rigidBoxSdfContact.recordTasksSubmitted(taskCount);
			}

			void beginRigidBoxSdfContactTask()
			{
				rigidBoxSdfContact.beginTask();
			}

			void endRigidBoxSdfContactTask()
			{
				rigidBoxSdfContact.endTask();
			}

			void recordRigidBoxSdfContactFanIn()
			{
				rigidBoxSdfContact.recordFanIn();
			}

			void recordSerialRigidBoxSdfContactFallback()
			{
				rigidBoxSdfContact.recordSerialFallback();
			}

			void recordRigidSphereSdfContactTasksSubmitted(PxU32 taskCount)
			{
				rigidSphereSdfContact.recordTasksSubmitted(taskCount);
			}

			void beginRigidSphereSdfContactTask()
			{
				rigidSphereSdfContact.beginTask();
			}

			void endRigidSphereSdfContactTask()
			{
				rigidSphereSdfContact.endTask();
			}

			void recordRigidSphereSdfContactFanIn()
			{
				rigidSphereSdfContact.recordFanIn();
			}

			void recordSerialRigidSphereSdfContactFallback()
			{
				rigidSphereSdfContact.recordSerialFallback();
			}

			void recordRigidCapsuleSdfContactTasksSubmitted(PxU32 taskCount)
			{
				rigidCapsuleSdfContact.recordTasksSubmitted(taskCount);
			}

			void beginRigidCapsuleSdfContactTask()
			{
				rigidCapsuleSdfContact.beginTask();
			}

			void endRigidCapsuleSdfContactTask()
			{
				rigidCapsuleSdfContact.endTask();
			}

			void recordRigidCapsuleSdfContactFanIn()
			{
				rigidCapsuleSdfContact.recordFanIn();
			}

			void recordSerialRigidCapsuleSdfContactFallback()
			{
				rigidCapsuleSdfContact.recordSerialFallback();
			}

			void recordRigidConvexSdfContactTasksSubmitted(PxU32 taskCount)
			{
				rigidConvexSdfContact.recordTasksSubmitted(taskCount);
			}

			void beginRigidConvexSdfContactTask()
			{
				rigidConvexSdfContact.beginTask();
			}

			void endRigidConvexSdfContactTask()
			{
				rigidConvexSdfContact.endTask();
			}

			void recordRigidConvexSdfContactFanIn()
			{
				rigidConvexSdfContact.recordFanIn();
			}

			void recordSerialRigidConvexSdfContactFallback()
			{
				rigidConvexSdfContact.recordSerialFallback();
			}

			void recordRigidTriangleSurfaceContactTasksSubmitted(PxU32 taskCount)
			{
				rigidTriangleSurfaceContact.recordTasksSubmitted(taskCount);
			}

			void beginRigidTriangleSurfaceContactTask()
			{
				rigidTriangleSurfaceContact.beginTask();
			}

			void endRigidTriangleSurfaceContactTask()
			{
				rigidTriangleSurfaceContact.endTask();
			}

			void recordRigidTriangleSurfaceContactFanIn()
			{
				rigidTriangleSurfaceContact.recordFanIn();
			}

			void recordSerialRigidTriangleSurfaceContactFallback()
			{
				rigidTriangleSurfaceContact.recordSerialFallback();
			}

			void recordSoftPairContactTasksSubmitted(PxU32 taskCount)
			{
				softPairContact.recordTasksSubmitted(taskCount);
			}

			void beginSoftPairContactTask()
			{
				softPairContact.beginTask();
			}

			void endSoftPairContactTask()
			{
				softPairContact.endTask();
			}

			void recordSoftPairContactFanIn()
			{
				softPairContact.recordFanIn();
			}

			void recordSerialSoftPairContactFallback()
			{
				softPairContact.recordSerialFallback();
			}

			void recordSelfBvhContactTasksSubmitted(PxU32 taskCount)
			{
				selfBvhContact.recordTasksSubmitted(taskCount);
			}

			void beginSelfBvhContactTask()
			{
				selfBvhContact.beginTask();
			}

			void endSelfBvhContactTask()
			{
				selfBvhContact.endTask();
			}

			void recordSelfBvhContactFanIn()
			{
				selfBvhContact.recordFanIn();
			}

			void recordSerialSelfBvhContactFallback()
			{
				selfBvhContact.recordSerialFallback();
			}
		};

	public:
		AvbdCpuSoftScene(
			const PxsDeformableVolumeMaterialManager&
				deformableMaterialManager,
			const PxsDeformableSurfaceMaterialManager&
				surfaceMaterialManager,
			const PxsMaterialManager& rigidMaterialManager,
			PxU64 contextId,
			IG::SimpleIslandManager& islandManager)
			: mContextId(contextId),
			  mDeformableMaterialManager(deformableMaterialManager),
			  mSurfaceMaterialManager(surfaceMaterialManager),
			  mRigidMaterialManager(rigidMaterialManager),
			  mIslandManager(islandManager),
			  mNextPrimitiveKey(1),
			  mRigidTriangleSurfaceCompileStamp(0),
			  mNextWorldPinHandle(1),
			  mNextRigidAttachmentHandle(1),
			  mNextArticulationAttachmentHandle(1),
			  mNextSoftPairAttachmentHandle(1),
			  mNextPrescribedAttachmentHandle(1),
			  mNextRigidActorFilterHandle(1),
			  mNextDeformablePairFilterHandle(1),
			  mDynamicsOwnsStep(false),
			  mDynamicsSelectedEntryCount(0),
			  mLastComponentFallbackSteps(0),
			  mLastNativeIslandSteps(0),
			  mComponentFallbackPlanPrepared(false),
			  mStandaloneComponentSolvePrepared(false),
			  mStandaloneComponentPostSolvePending(false),
			  mStandaloneTaskGraphDispatcherWorkers(0),
			  mStandaloneTaskGraphEnhancedDeterminism(false),
			  mStandaloneParticlePrimalSchedule(
				  Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR),
			  mP3ForceSplitPrediction(false),
			  mCollisionStatsEnabled(false),
			  mWorldPlaneContactTransactionPending(false),
			  mRigidBoxSdfContactTransactionPending(false),
			  mRigidSphereSdfContactTransactionPending(false),
			  mRigidCapsuleSdfContactTransactionPending(false),
			  mRigidConvexSdfContactTransactionPending(false),
			  mRigidTriangleSurfaceContactTransactionPending(false),
			  mRigidTriangleSurfaceContactTaskSubmitStartNanos(0),
			  mRigidTriangleSurfaceContactTaskSubmitEndNanos(0),
			  mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan(false),
			  mRigidTriangleSurfaceFeatureRoundRobinTaskPlan(false),
			  mSoftPairContactTransactionPending(false),
			  mSoftPairContactUseSurfaceTriangleBvh(false),
			  mSelfBvhContactTransactionPending(false),
			  mSelfBvhContactBodyIndex(PX_MAX_U32),
			  mStaticWorldSelfOgcContactTransactionPending(false),
			  mContactSetTraceFile(NULL),
			  mContactSetTraceDetectionIndex(0),
			  mWorkspacePreflightPending(true)
		{
			const char* const collisionTelemetry =
				std::getenv("PHYSX_AVBD_COLLISION_TELEMETRY");
			mCollisionStatsEnabled = collisionTelemetry &&
				collisionTelemetry[0] == '1' &&
				collisionTelemetry[1] == '\0';
			const char* const forceSplitPrediction =
				std::getenv("PHYSX_AVBD_P3_FORCE_SPLIT_PREDICTION");
			mP3ForceSplitPrediction = forceSplitPrediction &&
				forceSplitPrediction[0] == '1' &&
				forceSplitPrediction[1] == '\0';
			const char* const contactSetTracePath =
				std::getenv("PHYSX_AVBD_CONTACT_SET_TRACE");
			if(contactSetTracePath && contactSetTracePath[0] != '\0')
				mContactSetTraceFile = std::fopen(
					contactSetTracePath, "wb");
		}

		~AvbdCpuSoftScene()
		{
			for(PxU32 i = 0; i < mPredictionTasks.size(); i++)
				PX_DELETE(mPredictionTasks[i]);
			mPredictionTasks.clear();
			for(PxU32 i = 0; i < mWriteBackTasks.size(); i++)
				PX_DELETE(mWriteBackTasks[i]);
			mWriteBackTasks.clear();
			for(PxU32 i = 0; i < mCausalLayerTasks.size(); i++)
				PX_DELETE(mCausalLayerTasks[i]);
			mCausalLayerTasks.clear();
			for(PxU32 i = 0; i < mCausalLayerFinishTasks.size(); i++)
				PX_DELETE(mCausalLayerFinishTasks[i]);
			mCausalLayerFinishTasks.clear();
			for(PxU32 i = 0; i < mWorldPlaneContactTasks.size(); i++)
				PX_DELETE(mWorldPlaneContactTasks[i]);
			mWorldPlaneContactTasks.clear();
			for(PxU32 i = 0; i < mWorldPlaneContactFinishTasks.size(); i++)
				PX_DELETE(mWorldPlaneContactFinishTasks[i]);
			mWorldPlaneContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidBoxSdfContactTasks.size(); i++)
				PX_DELETE(mRigidBoxSdfContactTasks[i]);
			mRigidBoxSdfContactTasks.clear();
			for(PxU32 i = 0; i < mRigidBoxSdfContactFinishTasks.size(); i++)
				PX_DELETE(mRigidBoxSdfContactFinishTasks[i]);
			mRigidBoxSdfContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidSphereSdfContactTasks.size(); i++)
				PX_DELETE(mRigidSphereSdfContactTasks[i]);
			mRigidSphereSdfContactTasks.clear();
			for(PxU32 i = 0; i < mRigidSphereSdfContactFinishTasks.size(); i++)
				PX_DELETE(mRigidSphereSdfContactFinishTasks[i]);
			mRigidSphereSdfContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidCapsuleSdfContactTasks.size(); i++)
				PX_DELETE(mRigidCapsuleSdfContactTasks[i]);
			mRigidCapsuleSdfContactTasks.clear();
			for(PxU32 i = 0; i < mRigidCapsuleSdfContactFinishTasks.size(); i++)
				PX_DELETE(mRigidCapsuleSdfContactFinishTasks[i]);
			mRigidCapsuleSdfContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidConvexSdfContactTasks.size(); i++)
				PX_DELETE(mRigidConvexSdfContactTasks[i]);
			mRigidConvexSdfContactTasks.clear();
			for(PxU32 i = 0; i < mRigidConvexSdfContactFinishTasks.size(); i++)
				PX_DELETE(mRigidConvexSdfContactFinishTasks[i]);
			mRigidConvexSdfContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidTriangleSurfaceContactTasks.size(); i++)
				PX_DELETE(mRigidTriangleSurfaceContactTasks[i]);
			mRigidTriangleSurfaceContactTasks.clear();
			for(PxU32 i = 0; i < mRigidTriangleSurfaceContactFinishTasks.size(); i++)
				PX_DELETE(mRigidTriangleSurfaceContactFinishTasks[i]);
			mRigidTriangleSurfaceContactFinishTasks.clear();
			for(PxU32 i = 0; i < mSoftPairContactTasks.size(); i++)
				PX_DELETE(mSoftPairContactTasks[i]);
			mSoftPairContactTasks.clear();
			for(PxU32 i = 0; i < mSoftPairContactFinishTasks.size(); i++)
				PX_DELETE(mSoftPairContactFinishTasks[i]);
			mSoftPairContactFinishTasks.clear();
			for(PxU32 i = 0; i < mSelfBvhContactTasks.size(); i++)
				PX_DELETE(mSelfBvhContactTasks[i]);
			mSelfBvhContactTasks.clear();
			for(PxU32 i = 0; i < mSelfBvhContactFinishTasks.size(); i++)
				PX_DELETE(mSelfBvhContactFinishTasks[i]);
			mSelfBvhContactFinishTasks.clear();
			for(PxU32 i = 0;
				i < mStaticWorldSelfOgcContactTasks.size(); i++)
				PX_DELETE(mStaticWorldSelfOgcContactTasks[i]);
			mStaticWorldSelfOgcContactTasks.clear();
			for(PxU32 i = 0;
				i < mStaticWorldSelfOgcContactFinishTasks.size(); i++)
				PX_DELETE(mStaticWorldSelfOgcContactFinishTasks[i]);
			mStaticWorldSelfOgcContactFinishTasks.clear();
			clearNativeIslandEdges();
			clearIslandSelectionStorages();
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				Entry& entry = mEntries[i];
				mIslandManager.removeNode(entry.islandNode);
				entry.destroyIslandObject();
			}
			if(mContactSetTraceFile)
			{
				std::fclose(mContactSetTraceFile);
				mContactSetTraceFile = NULL;
			}
		}

		bool add(
			DeformableVolumeCore& core,
			PxTetrahedronMesh& simulationMesh,
			PxTetrahedronMesh& collisionMesh,
			PxDeformableVolumeAuxData& auxData,
			const PxsDeformableVolumeMaterialManager& materialManager)
		{
			const PxU32 numVertices = simulationMesh.getNbVertices();
			const PxU32 numTets = simulationMesh.getNbTetrahedrons();
			Dy::DeformableVolumeCore& dyCore = core.getCore();
			if(numVertices == 0 || numTets == 0 ||
				!dyCore.simPositionInvMass || !dyCore.simVelocity ||
				!dyCore.positionInvMass || !dyCore.restPosition)
				return false;
			if(!validateVolumeCollisionEmbedding(
				simulationMesh, collisionMesh, auxData))
				return false;

			for(PxU32 i = 0; i < mEntries.size(); i++)
				if(mEntries[i].volumeCore == &core)
					return false;

			for(PxU32 i = 0; i < numVertices; i++)
			{
				const PxVec3 position =
					dyCore.simPositionInvMass[i].getXYZ();
				if(!position.isFinite() ||
					!PxIsFinite(dyCore.simPositionInvMass[i].w) ||
					!dyCore.simVelocity[i].getXYZ().isFinite())
					return false;
			}
			core.initializeCpuAvbdSimulationRestPositions(
				dyCore.simPositionInvMass, numVertices);
			const PxArray<PxVec3>& restVertices =
				core.getCpuAvbdSimulationRestPositions();
			if(restVertices.size() != numVertices)
				return false;
			for(PxU32 i = 0; i < numVertices; i++)
				if(!restVertices[i].isFinite())
					return false;

			PxArray<PxU32> tetrahedra;
			tetrahedra.resize(4 * numTets);
			const bool has16BitIndices =
				simulationMesh.getTetrahedronMeshFlags() &
				PxTetrahedronMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source =
					static_cast<const PxU16*>(
						simulationMesh.getTetrahedrons());
				for(PxU32 i = 0; i < tetrahedra.size(); i++)
					tetrahedra[i] = source[i];
			}
			else
			{
				const PxU32* source =
					static_cast<const PxU32*>(
						simulationMesh.getTetrahedrons());
				for(PxU32 i = 0; i < tetrahedra.size(); i++)
					tetrahedra[i] = source[i];
			}
			for(PxU32 i = 0; i < tetrahedra.size(); i++)
				if(tetrahedra[i] >= numVertices)
					return false;

			const PxsDeformableVolumeMaterialCore* material =
				getMaterial(core, materialManager);
			const PxReal youngs = material ? material->youngs : 1.0e5f;
			const PxReal poissons =
				material ? material->poissons : 0.3f;
			const PxReal materialDamping =
				material ? material->elasticityDamping : 0.0f;
			const bool coRotationalVolumeModel =
				!material || material->materialModel ==
					PxDeformableVolumeMaterialModel::eCO_ROTATIONAL;
			const PxReal gravityScale =
				(core.getActorFlags() & PxActorFlag::eDISABLE_GRAVITY)
				? 0.0f : 1.0f;
			const PxU32 bodyIndex = mBodies.size();
			const PxU32 particleStart = Dy::avbdCreateSoftBody(
				restVertices.begin(), numVertices,
				tetrahedra.begin(), tetrahedra.size(),
				NULL, 0,
				youngs, poissons,
				1.0f, core.getLinearDamping() + materialDamping,
				0.0f, 0.01f,
				mParticles, mBodies, false,
				core.getSelfCollisionFilterDistance(),
				material ? material->dynamicFriction : 0.5f,
				coRotationalVolumeModel);

			for(PxU32 i = 0; i < numVertices; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				const PxVec4& positionInvMass =
					dyCore.simPositionInvMass[i];
				particle.position = positionInvMass.getXYZ();
				particle.initialPosition = particle.position;
				particle.predictedPosition = particle.position;
				particle.outerPosition = particle.position;
				particle.velocity = dyCore.simVelocity[i].getXYZ();
				particle.prevVelocity = particle.velocity;
				particle.invMass =
					PxMax(positionInvMass.w, 0.0f);
				particle.mass = particle.invMass > 0.0f
					? 1.0f / particle.invMass : 0.0f;
				particle.damping =
					core.getLinearDamping() + materialDamping;
				particle.gravityScale = gravityScale;
			}

			void* islandObjectMemory = PX_ALLOC(
				sizeof(Dy::DeformableVolume),
				"AVBD CPU deformable island object");
			Dy::DeformableVolume* islandObject =
				islandObjectMemory
				? PX_PLACEMENT_NEW(
					islandObjectMemory,
					Dy::DeformableVolume)(NULL, dyCore)
				: NULL;
			if(!islandObject)
			{
				mBodies.replaceWithLast(bodyIndex);
				mParticles.resize(particleStart);
				return false;
			}
			const PxNodeIndex islandNode =
				mIslandManager.addNode(
					false, false,
					IG::Node::eDEFORMABLE_VOLUME_TYPE,
					islandObject);
			PxReal maxSpeedSquared = 0.0f;
			for(PxU32 i = 0; i < numVertices; i++)
			{
				if(dyCore.simPositionInvMass[i].w > 0.0f)
					maxSpeedSquared = PxMax(
						maxSpeedSquared,
						dyCore.simVelocity[i].getXYZ().
							magnitudeSquared());
			}
			const PxReal sleepThreshold =
				PxMax(dyCore.sleepThreshold, 0.0f);
			const bool startsSleeping =
				dyCore.wakeCounter == 0.0f &&
				maxSpeedSquared <=
					sleepThreshold * sleepThreshold;
			if(startsSleeping)
				mIslandManager.deactivateNode(islandNode);
			else
				mIslandManager.activateNode(islandNode);
			mEntries.pushBack(Entry(
				core, simulationMesh, collisionMesh, auxData,
				bodyIndex, *islandObject, islandNode,
				startsSleeping));
			// Reserve topology-bounded solver and collision-query scratch while
			// the actor is added, before its first simulation step. Contact
			// output capacity remains subject to the separate source-aware
			// policy, because its density is collision-state dependent.
			mWorkspace.reserve(mParticles.size(), 0);
			reserveLifecycleContactCapacity();
			reserveLifecycleCollisionScratch();
			PX_ASSERT(mSelfCollisionAdjacencies.size() == bodyIndex);
			mSelfCollisionAdjacencies.resize(bodyIndex + 1);
			Dy::avbdBuildSelfCollisionAdjacency(
				mBodies[bodyIndex],
				mSelfCollisionAdjacencies[bodyIndex]);
			if(startsSleeping)
				sleepEntry(mEntries.back());
			else
			{
				dyCore.cpuAvbdSleeping = false;
				dyCore.cpuAvbdWakeRequested = false;
			}
			dyCore.dirty = false;
			dyCore.dirtyFlags = PxDeformableVolumeDataFlags(0);
			if(!rebuildCollisionDetectionScene())
			{
				reportInvalidCollisionEmbedding(
					"CPU AVBD failed to build the cooked collision-domain detection scene.");
				removeEntry(core);
				return false;
			}
			return true;
		}

		bool addSurface(
			DeformableSurfaceCore& core,
			PxTriangleMesh& triangleMesh)
		{
			const PxU32 numVertices = triangleMesh.getNbVertices();
			const PxU32 numTriangles = triangleMesh.getNbTriangles();
			Dy::DeformableSurfaceCore& dyCore = core.getCore();
			if(numVertices == 0 || numTriangles == 0 ||
				!dyCore.positionInvMass || !dyCore.velocity ||
				!dyCore.restPosition)
				return false;

			for(PxU32 i = 0; i < mEntries.size(); i++)
				if(mEntries[i].surfaceCore == &core)
					return false;

			PxArray<PxVec3> restVertices;
			restVertices.resize(numVertices);
			for(PxU32 i = 0; i < numVertices; i++)
			{
				const PxVec4& restPosition = dyCore.restPosition[i];
				const PxVec4& positionInvMass =
					dyCore.positionInvMass[i];
				if(!restPosition.getXYZ().isFinite() ||
					!positionInvMass.getXYZ().isFinite() ||
					!PxIsFinite(positionInvMass.w) ||
					!dyCore.velocity[i].getXYZ().isFinite())
					return false;
				restVertices[i] = restPosition.getXYZ();
			}

			PxArray<PxU32> triangles;
			triangles.resize(3 * numTriangles);
			const bool has16BitIndices =
				triangleMesh.getTriangleMeshFlags() &
				PxTriangleMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source =
					static_cast<const PxU16*>(
						triangleMesh.getTriangles());
				for(PxU32 i = 0; i < triangles.size(); i++)
					triangles[i] = source[i];
			}
			else
			{
				const PxU32* source =
					static_cast<const PxU32*>(
						triangleMesh.getTriangles());
				for(PxU32 i = 0; i < triangles.size(); i++)
					triangles[i] = source[i];
			}
			for(PxU32 i = 0; i < triangles.size(); i++)
				if(triangles[i] >= numVertices)
					return false;

			const PxsDeformableSurfaceMaterialCore* material =
				getSurfaceMaterial(core);
			const PxReal youngs =
				material ? material->youngs : 1.0e5f;
			const PxReal poissons =
				material ? material->poissons : 0.3f;
			const PxReal materialDamping =
				material ? material->elasticityDamping : 0.0f;
			const PxReal bendingStiffness =
				material ? material->bendingStiffness : 0.0f;
			const PxReal thickness =
				material ? PxMax(material->thickness, 1.0e-4f)
						 : 0.01f;
			const PxReal gravityScale =
				(core.getActorFlags() & PxActorFlag::eDISABLE_GRAVITY)
				? 0.0f : 1.0f;
			const PxU32 bodyIndex = mBodies.size();
			const PxU32 particleStart = Dy::avbdCreateSoftBody(
				restVertices.begin(), numVertices,
				NULL, 0,
				triangles.begin(), triangles.size(),
				youngs, poissons,
				1.0f, core.getLinearDamping() + materialDamping,
				bendingStiffness, thickness,
				mParticles, mBodies,
				(core.getSurfaceFlags() &
					PxDeformableSurfaceFlag::eENABLE_FLATTENING)
					? true : false,
				core.getSelfCollisionFilterDistance(),
				material ? material->dynamicFriction : 0.5f);

			for(PxU32 i = 0; i < numVertices; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				const PxVec4& positionInvMass =
					dyCore.positionInvMass[i];
				particle.position = positionInvMass.getXYZ();
				particle.initialPosition = restVertices[i];
				particle.predictedPosition = particle.position;
				particle.outerPosition = particle.position;
				particle.velocity = dyCore.velocity[i].getXYZ();
				particle.prevVelocity = particle.velocity;
				particle.invMass =
					PxMax(positionInvMass.w, 0.0f);
				particle.mass = particle.invMass > 0.0f
					? 1.0f / particle.invMass : 0.0f;
				particle.damping =
					core.getLinearDamping() + materialDamping;
				particle.gravityScale = gravityScale;
			}

			void* islandObjectMemory = PX_ALLOC(
				sizeof(Dy::DeformableSurface),
				"AVBD CPU deformable surface island object");
			Dy::DeformableSurface* islandObject =
				islandObjectMemory
					? PX_PLACEMENT_NEW(
						islandObjectMemory,
						Dy::DeformableSurface)(NULL, dyCore)
					: NULL;
			if(!islandObject)
			{
				mBodies.replaceWithLast(bodyIndex);
				mParticles.resize(particleStart);
				return false;
			}
			const PxNodeIndex islandNode =
				mIslandManager.addNode(
					false, false,
					IG::Node::eDEFORMABLE_SURFACE_TYPE,
					islandObject);
			PxReal maxSpeedSquared = 0.0f;
			for(PxU32 i = 0; i < numVertices; i++)
			{
				if(dyCore.positionInvMass[i].w > 0.0f)
					maxSpeedSquared = PxMax(
						maxSpeedSquared,
						dyCore.velocity[i].getXYZ().
							magnitudeSquared());
			}
			const PxReal sleepThreshold =
				PxMax(dyCore.sleepThreshold, 0.0f);
			const bool startsSleeping =
				dyCore.wakeCounter == 0.0f &&
				maxSpeedSquared <=
					sleepThreshold * sleepThreshold;
			if(startsSleeping)
				mIslandManager.deactivateNode(islandNode);
			else
				mIslandManager.activateNode(islandNode);
			mEntries.pushBack(Entry(
				core, triangleMesh, bodyIndex,
				*islandObject, islandNode, startsSleeping));
			// Keep particle/body-sized scratch out of the simulation hot path.
			// Known peer/rigid sources may reserve a budgeted contact capacity
			// here; unbounded contact density remains separately measured.
			mWorkspace.reserve(mParticles.size(), 0);
			reserveLifecycleContactCapacity();
			reserveLifecycleCollisionScratch();
			PX_ASSERT(mSelfCollisionAdjacencies.size() == bodyIndex);
			mSelfCollisionAdjacencies.resize(bodyIndex + 1);
			Dy::avbdBuildSelfCollisionAdjacency(
				mBodies[bodyIndex],
				mSelfCollisionAdjacencies[bodyIndex]);
			if(startsSleeping)
				sleepEntry(mEntries.back());
			else
			{
				dyCore.cpuAvbdSleeping = false;
				dyCore.cpuAvbdWakeRequested = false;
			}
			dyCore.dirty = false;
			dyCore.dirtyFlags = PxDeformableSurfaceDataFlags(0);
			if(!rebuildCollisionDetectionScene())
			{
				removeEntry(core);
				return false;
			}
			return true;
		}

		void addStaticShape(StaticCore& core, ShapeCore& shape)
		{
			for(PxU32 i = 0; i < mStaticShapes.size(); i++)
			{
				if(mStaticShapes[i].core == &core &&
					mStaticShapes[i].shape == &shape)
					return;
			}
			mStaticShapes.pushBack(StaticShapeEntry(
				core, shape, mNextPrimitiveKey++));
		}

		void removeStaticShape(
			StaticCore& core, const ShapeCore& shape)
		{
			for(PxU32 i = mStaticShapes.size(); i > 0; i--)
			{
				const StaticShapeEntry& entry = mStaticShapes[i - 1];
				if(entry.core == &core && entry.shape == &shape)
				{
					mStaticShapes.replaceWithLast(i - 1);
					return;
				}
			}
		}

		void removeStatic(StaticCore& core)
		{
			for(PxU32 i = mStaticShapes.size(); i > 0; i--)
			{
				if(mStaticShapes[i - 1].core == &core)
					mStaticShapes.replaceWithLast(i - 1);
			}
			removePrescribedAttachmentsForRigid(core);
		}

		void addDynamicShape(BodyCore& core, ShapeCore& shape)
		{
			for(PxU32 i = 0; i < mDynamicShapes.size(); i++)
			{
				if(mDynamicShapes[i].core == &core &&
					mDynamicShapes[i].shape == &shape)
					return;
			}
			mDynamicShapes.pushBack(DynamicShapeEntry(
				core, shape, mNextPrimitiveKey++));
		}

		void removeDynamicShape(
			BodyCore& core, const ShapeCore& shape)
		{
			for(PxU32 i = mDynamicShapes.size(); i > 0; i--)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[i - 1];
				if(entry.core == &core && entry.shape == &shape)
				{
					mDynamicShapes.replaceWithLast(i - 1);
					bool hasRemainingShape = false;
					for(PxU32 j = 0; j < mDynamicShapes.size(); j++)
					{
						if(mDynamicShapes[j].core == &core)
						{
							hasRemainingShape = true;
							break;
						}
					}
					if(!hasRemainingShape)
						removeNativeIslandEdgesForRigid(core);
					return;
				}
			}
		}

		void removeDynamic(BodyCore& core)
		{
			for(PxU32 i = mDynamicShapes.size(); i > 0; i--)
			{
				if(mDynamicShapes[i - 1].core == &core)
					mDynamicShapes.replaceWithLast(i - 1);
			}
			removePrescribedAttachmentsForRigid(core);
			removeRigidAttachmentsForRigid(core);
			removeArticulationAttachmentsForLink(core);
			removeNativeIslandEdgesForRigid(core);
		}

		void remove(DeformableVolumeCore& core)
		{
			removeEntry(core);
		}

		void removeSurface(DeformableSurfaceCore& core)
		{
			removeEntry(core);
		}

		bool buildLocalElementPoint(
			ActorCore& core,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			Dy::AvbdSoftPoint& point)
		{
			const Entry* entry = findEntry(core);
			if(!entry || entry->bodyIndex >= mBodies.size() ||
				!barycentric.isFinite())
				return false;
			const Dy::AvbdSoftBody& body =
				mBodies[entry->bodyIndex];
			const PxU32 endpointCount = surfaceElement ? 3u : 4u;
			const PxU32* topology = surfaceElement
				? body.compiled.triangles.begin()
				: body.compiled.tetrahedra.begin();
			const PxU32 topologyCount = surfaceElement
				? body.compiled.triangles.size()
				: body.compiled.tetrahedra.size();
			if(elementIndex >= topologyCount / endpointCount)
				return false;

			const PxReal weights[4] = {
				barycentric.x, barycentric.y,
				barycentric.z, barycentric.w};
			PxReal weightSum = 0.0f;
			for(PxU32 endpoint = 0;
				endpoint < endpointCount; endpoint++)
			{
				if(weights[endpoint] < 0.0f ||
					weights[endpoint] > 1.0f)
					return false;
				weightSum += weights[endpoint];
			}
			if(PxAbs(weightSum - 1.0f) > 1.0e-4f)
				return false;

			point.particleCount = endpointCount;
			for(PxU32 endpoint = 0; endpoint < endpointCount; endpoint++)
			{
				point.particleIndices[endpoint] =
					topology[elementIndex * endpointCount + endpoint];
				point.weights[endpoint] = weights[endpoint];
			}
			for(PxU32 endpoint = endpointCount; endpoint < 4; endpoint++)
			{
				point.particleIndices[endpoint] = PX_MAX_U32;
				point.weights[endpoint] = 0.0f;
			}
			return Dy::avbdIsSoftPointValid(
				point, 0, body.compiled.particleCount);
		}

		PxU32 addWorldPin(
			ActorCore& core,
			PxU32 localVertex,
			const PxVec3& worldTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addWorldPin(core, point, worldTarget);
		}

		PxU32 addWorldPin(
			ActorCore& core,
			const Dy::AvbdSoftPoint& localPoint,
			const PxVec3& worldTarget)
		{
			Entry* entry = findEntry(core);
			if(!entry || !Dy::avbdIsSoftPointValid(
					localPoint, 0, getParticleCount(*entry)) ||
				!worldTarget.isFinite())
				return PX_MAX_U32;

			PxU32 handle = mNextWorldPinHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextWorldPinHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mWorldPins.pushBack(WorldPinEntry(
				core, localPoint, worldTarget, handle));
			if(!rebuildEntryPins(*entry))
			{
				mWorldPins.popBack();
				return PX_MAX_U32;
			}
			wakeEntry(*entry, ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 addWorldElementPin(
			ActorCore& core,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& worldTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				core, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addWorldPin(core, point, worldTarget);
		}

		bool updateWorldPin(
			ActorCore& core,
			PxU32 handle,
			const PxVec3& worldTarget)
		{
			if(!worldTarget.isFinite())
				return false;
			for(PxU32 i = 0; i < mWorldPins.size(); i++)
			{
				WorldPinEntry& pin = mWorldPins[i];
				if(pin.softCore != &core || pin.handle != handle)
					continue;
				Entry* entry = findEntry(core);
				if(!entry)
					return false;
				const PxVec3 oldTarget = pin.worldTarget;
				pin.worldTarget = worldTarget;
				if(!rebuildEntryPins(*entry))
				{
					pin.worldTarget = oldTarget;
					const bool restored = rebuildEntryPins(*entry);
					PX_ASSERT(restored);
					PX_UNUSED(restored);
					return false;
				}
				wakeEntry(*entry, ScInternalWakeCounterResetValue);
				return true;
			}
			return false;
		}

		void removeWorldPin(
			ActorCore& core,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mWorldPins.size(); i++)
			{
				if(mWorldPins[i].softCore != &core ||
					mWorldPins[i].handle != handle)
					continue;
				mWorldPins.replaceWithLast(i);
				Entry* entry = findEntry(core);
				if(entry)
				{
					const bool rebuilt = rebuildEntryPins(*entry);
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
					wakeEntry(
						*entry, ScInternalWakeCounterResetValue);
				}
				return;
			}
		}

		bool computePrescribedAttachmentWorldTarget(
			RigidCore& prescribedCore,
			const PxVec3& actorLocalTarget,
			PxVec3& worldTarget) const
		{
			if(!actorLocalTarget.isFinite())
				return false;
			const PxActorType::Enum actorType =
				prescribedCore.getActorCoreType();
			if(actorType == PxActorType::eRIGID_STATIC)
			{
				const StaticCore& staticCore =
					static_cast<const StaticCore&>(
						prescribedCore);
				if(!staticCore.getSim())
					return false;
				const PxTransform& actorToWorld =
					staticCore.getActor2World();
				worldTarget =
					actorToWorld.transform(actorLocalTarget);
				return actorToWorld.isValid() &&
					worldTarget.isFinite();
			}
			if(actorType != PxActorType::eRIGID_DYNAMIC)
				return false;

			BodyCore& kinematicCore =
				static_cast<BodyCore&>(prescribedCore);
			BodySim* bodySim = kinematicCore.getSim();
			if(!bodySim || !bodySim->isKinematic() ||
				bodySim->isArticulationLink())
				return false;
			const PxsBodyCore& bodyCore = kinematicCore.getCore();
			PxTransform bodyToWorld = bodyCore.body2World;
			PxTransform commandedBodyToWorld;
			if(kinematicCore.getKinematicTarget(
				commandedBodyToWorld))
				bodyToWorld = commandedBodyToWorld;
			const PxVec3 bodyLocalTarget =
				bodyCore.getBody2Actor().getInverse().
					transform(actorLocalTarget);
			worldTarget = bodyToWorld.transform(bodyLocalTarget);
			return bodyToWorld.isValid() && worldTarget.isFinite();
		}

		PxU32 addKinematicAttachment(
			ActorCore& softCore,
			BodyCore& kinematicCore,
			PxU32 localVertex,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addPrescribedAttachment(
				softCore, kinematicCore, point,
				actorLocalTarget);
		}

		PxU32 addPrescribedAttachment(
			ActorCore& softCore,
			RigidCore& prescribedCore,
			const Dy::AvbdSoftPoint& localPoint,
			const PxVec3& actorLocalTarget)
		{
			Entry* entry = findEntry(softCore);
			PxVec3 worldTarget;
			if(!entry || !Dy::avbdIsSoftPointValid(
					localPoint, 0, getParticleCount(*entry)) ||
				!computePrescribedAttachmentWorldTarget(
					prescribedCore, actorLocalTarget,
					worldTarget))
				return PX_MAX_U32;

			PxU32 handle =
				mNextPrescribedAttachmentHandle++;
			if(handle == PX_MAX_U32)
				handle =
					mNextPrescribedAttachmentHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mPrescribedAttachments.pushBack(
				PrescribedAttachmentEntry(
					softCore, prescribedCore, localPoint,
					actorLocalTarget, worldTarget, handle));
			if(!rebuildEntryPins(*entry))
			{
				mPrescribedAttachments.popBack();
				const bool restored = rebuildEntryPins(*entry);
				PX_ASSERT(restored);
				PX_UNUSED(restored);
				return PX_MAX_U32;
			}
			wakeEntry(*entry, ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 addKinematicElementAttachment(
			ActorCore& softCore,
			BodyCore& kinematicCore,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				softCore, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addPrescribedAttachment(
				softCore, kinematicCore, point,
				actorLocalTarget);
		}

		PxU32 addStaticAttachment(
			ActorCore& softCore,
			StaticCore& staticCore,
			PxU32 localVertex,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addPrescribedAttachment(
				softCore, staticCore, point,
				actorLocalTarget);
		}

		PxU32 addStaticElementAttachment(
			ActorCore& softCore,
			StaticCore& staticCore,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				softCore, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addPrescribedAttachment(
				softCore, staticCore, point,
				actorLocalTarget);
		}

		bool updatePrescribedAttachment(
			ActorCore& softCore,
			PxU32 handle,
			const PxVec3& actorLocalTarget)
		{
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				PrescribedAttachmentEntry& attachment =
					mPrescribedAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				Entry* entry = findEntry(softCore);
				PxVec3 worldTarget;
				if(!entry ||
					!computePrescribedAttachmentWorldTarget(
						*attachment.prescribedCore,
						actorLocalTarget, worldTarget))
					return false;
				const PxVec3 oldActorLocalTarget =
					attachment.actorLocalTarget;
				const PxVec3 oldWorldTarget =
					attachment.worldTarget;
				const PxVec3 oldPreviousWorldTarget =
					attachment.previousWorldTarget;
				const PxVec3 oldLambda =
					attachment.alLambda;
				attachment.actorLocalTarget =
					actorLocalTarget;
				attachment.previousWorldTarget =
					worldTarget;
				attachment.worldTarget = worldTarget;
				attachment.alLambda = PxVec3(0.0f);
				if(!rebuildEntryPins(*entry))
				{
					attachment.actorLocalTarget =
						oldActorLocalTarget;
					attachment.worldTarget = oldWorldTarget;
					attachment.previousWorldTarget =
						oldPreviousWorldTarget;
					attachment.alLambda = oldLambda;
					const bool restored =
						rebuildEntryPins(*entry);
					PX_ASSERT(restored);
					PX_UNUSED(restored);
					return false;
				}
				wakeEntry(
					*entry, ScInternalWakeCounterResetValue);
				return true;
			}
			return false;
		}

		void removePrescribedAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				const PrescribedAttachmentEntry& attachment =
					mPrescribedAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				mPrescribedAttachments.replaceWithLast(i);
				Entry* entry = findEntry(softCore);
				if(entry)
				{
					const bool rebuilt =
						rebuildEntryPins(*entry);
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
					wakeEntry(
						*entry,
						ScInternalWakeCounterResetValue);
				}
				return;
			}
		}

		PxU32 addRigidAttachment(
			ActorCore& softCore,
			BodyCore& rigidCore,
			PxU32 localVertex,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addRigidAttachment(
				softCore, rigidCore, point, actorLocalTarget);
		}

		PxU32 addRigidAttachment(
			ActorCore& softCore,
			BodyCore& rigidCore,
			const Dy::AvbdSoftPoint& localPoint,
			const PxVec3& actorLocalTarget)
		{
			Entry* entry = findEntry(softCore);
			BodySim* bodySim = rigidCore.getSim();
			if(!entry || !Dy::avbdIsSoftPointValid(
					localPoint, 0, getParticleCount(*entry)) ||
				!actorLocalTarget.isFinite() || !bodySim ||
				bodySim->isKinematic() ||
				bodySim->isArticulationLink())
				return PX_MAX_U32;

			PxU32 handle = mNextRigidAttachmentHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextRigidAttachmentHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mRigidAttachments.pushBack(
				RigidAttachmentEntry(
					softCore, rigidCore, localPoint,
					actorLocalTarget, handle));
			clearIslandSelectionStorages();
			mDynamicsOwnsStep = false;
			ensureNativeIslandEdge(*entry, rigidCore);
			wakeEntry(*entry, ScInternalWakeCounterResetValue);
			rigidCore.wakeUp(ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 addRigidElementAttachment(
			ActorCore& softCore,
			BodyCore& rigidCore,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				softCore, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addRigidAttachment(
				softCore, rigidCore, point, actorLocalTarget);
		}

		bool updateRigidAttachment(
			ActorCore& softCore,
			PxU32 handle,
			const PxVec3& actorLocalTarget)
		{
			if(!actorLocalTarget.isFinite())
				return false;
			for(PxU32 i = 0; i < mRigidAttachments.size(); i++)
			{
				RigidAttachmentEntry& attachment =
					mRigidAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				Entry* entry = findEntry(softCore);
				BodySim* bodySim =
					attachment.rigidCore->getSim();
				if(!entry || !bodySim ||
					bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					return false;
				attachment.actorLocalTarget = actorLocalTarget;
				attachment.alLambda = PxVec3(0.0f);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				ensureNativeIslandEdge(
					*entry, *attachment.rigidCore);
				wakeEntry(
					*entry, ScInternalWakeCounterResetValue);
				attachment.rigidCore->wakeUp(
					ScInternalWakeCounterResetValue);
				return true;
			}
			return false;
		}

		void removeRigidAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mRigidAttachments.size(); i++)
			{
				const RigidAttachmentEntry& attachment =
					mRigidAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				BodyCore* rigidCore = attachment.rigidCore;
				mRigidAttachments.replaceWithLast(i);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				Entry* entry = findEntry(softCore);
				if(entry)
					wakeEntry(
						*entry, ScInternalWakeCounterResetValue);
				if(rigidCore && rigidCore->getSim())
					rigidCore->wakeUp(
						ScInternalWakeCounterResetValue);
				return;
			}
		}

		PxU32 addArticulationAttachment(
			ActorCore& softCore,
			BodyCore& linkCore,
			PxU32 localVertex,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addArticulationAttachment(
				softCore, linkCore, point, actorLocalTarget);
		}

		PxU32 addArticulationAttachment(
			ActorCore& softCore,
			BodyCore& linkCore,
			const Dy::AvbdSoftPoint& localPoint,
			const PxVec3& actorLocalTarget)
		{
			Entry* entry = findEntry(softCore);
			BodySim* bodySim = linkCore.getSim();
			if(!entry || !Dy::avbdIsSoftPointValid(
					localPoint, 0, getParticleCount(*entry)) ||
				!actorLocalTarget.isFinite() || !bodySim ||
				!bodySim->isArticulationLink() ||
				!bodySim->getArticulation())
				return PX_MAX_U32;

			PxU32 handle =
				mNextArticulationAttachmentHandle++;
			if(handle == PX_MAX_U32)
				handle =
					mNextArticulationAttachmentHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mArticulationAttachments.pushBack(
				ArticulationAttachmentEntry(
					softCore, linkCore, localPoint,
					actorLocalTarget, handle));
			clearIslandSelectionStorages();
			mDynamicsOwnsStep = false;
			ensureNativeIslandEdge(*entry, linkCore);
			wakeEntry(*entry, ScInternalWakeCounterResetValue);
			linkCore.wakeUp(ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 addArticulationElementAttachment(
			ActorCore& softCore,
			BodyCore& linkCore,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				softCore, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addArticulationAttachment(
				softCore, linkCore, point, actorLocalTarget);
		}

		bool updateArticulationAttachment(
			ActorCore& softCore,
			PxU32 handle,
			const PxVec3& actorLocalTarget)
		{
			if(!actorLocalTarget.isFinite())
				return false;
			for(PxU32 i = 0;
				i < mArticulationAttachments.size(); i++)
			{
				ArticulationAttachmentEntry& attachment =
					mArticulationAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				Entry* entry = findEntry(softCore);
				BodySim* bodySim =
					attachment.linkCore->getSim();
				if(!entry || !bodySim ||
					!bodySim->isArticulationLink() ||
					!bodySim->getArticulation())
					return false;
				attachment.actorLocalTarget = actorLocalTarget;
				attachment.alLambda = PxVec3(0.0f);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				ensureNativeIslandEdge(
					*entry, *attachment.linkCore);
				wakeEntry(
					*entry, ScInternalWakeCounterResetValue);
				attachment.linkCore->wakeUp(
					ScInternalWakeCounterResetValue);
				return true;
			}
			return false;
		}

		void removeArticulationAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mArticulationAttachments.size(); i++)
			{
				const ArticulationAttachmentEntry& attachment =
					mArticulationAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				BodyCore* linkCore = attachment.linkCore;
				mArticulationAttachments.replaceWithLast(i);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				Entry* entry = findEntry(softCore);
				if(entry)
					wakeEntry(
						*entry, ScInternalWakeCounterResetValue);
				if(linkCore && linkCore->getSim())
					linkCore->wakeUp(
						ScInternalWakeCounterResetValue);
				return;
			}
		}

		PxU32 addSoftPairAttachment(
			ActorCore& softCore0,
			const Dy::AvbdSoftPoint& localPoint0,
			ActorCore& softCore1,
			const Dy::AvbdSoftPoint& localPoint1)
		{
			Entry* entry0 = findEntry(softCore0);
			Entry* entry1 = findEntry(softCore1);
			if(&softCore0 == &softCore1 || !entry0 || !entry1 ||
				!Dy::avbdIsSoftPointValid(
					localPoint0, 0, getParticleCount(*entry0)) ||
				!Dy::avbdIsSoftPointValid(
					localPoint1, 0, getParticleCount(*entry1)))
				return PX_MAX_U32;

			PxU32 handle = mNextSoftPairAttachmentHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextSoftPairAttachmentHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mSoftPairAttachments.pushBack(
				SoftPairAttachmentEntry(
					softCore0, localPoint0,
					softCore1, localPoint1, handle));
			clearIslandSelectionStorages();
			mDynamicsOwnsStep = false;
			ensureNativeSoftSoftIslandEdge(*entry0, *entry1);
			wakeEntry(*entry0, ScInternalWakeCounterResetValue);
			wakeEntry(*entry1, ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 addSoftPairAttachment(
			ActorCore& softCore0,
			bool element0,
			PxU32 index0,
			const PxVec4& barycentric0,
			ActorCore& softCore1,
			bool element1,
			PxU32 index1,
			const PxVec4& barycentric1)
		{
			Entry* entry0 = findEntry(softCore0);
			Entry* entry1 = findEntry(softCore1);
			if(!entry0 || !entry1)
				return PX_MAX_U32;

			Dy::AvbdSoftPoint point0;
			Dy::AvbdSoftPoint point1;
			if(element0)
			{
				if(!buildLocalElementPoint(
					softCore0, entry0->kind == eSURFACE,
					index0, barycentric0, point0))
					return PX_MAX_U32;
			}
			else
				point0.setVertex(index0);
			if(element1)
			{
				if(!buildLocalElementPoint(
					softCore1, entry1->kind == eSURFACE,
					index1, barycentric1, point1))
					return PX_MAX_U32;
			}
			else
				point1.setVertex(index1);
			return addSoftPairAttachment(
				softCore0, point0, softCore1, point1);
		}

		void removeSoftPairAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mSoftPairAttachments.size(); i++)
			{
				const SoftPairAttachmentEntry& attachment =
					mSoftPairAttachments[i];
				if(attachment.softCore[0] != &softCore ||
					attachment.handle != handle)
					continue;
				ActorCore* softCore0 = attachment.softCore[0];
				ActorCore* softCore1 = attachment.softCore[1];
				mSoftPairAttachments.replaceWithLast(i);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				Entry* entry0 =
					softCore0 ? findEntry(*softCore0) : NULL;
				Entry* entry1 =
					softCore1 ? findEntry(*softCore1) : NULL;
				if(entry0)
					wakeEntry(
						*entry0, ScInternalWakeCounterResetValue);
				if(entry1)
					wakeEntry(
						*entry1, ScInternalWakeCounterResetValue);
				return;
			}
		}

		PxU32 addRigidActorFilter(
			ActorCore& softCore,
			ActorCore& rigidCore,
			const PxU32* elementIndices = NULL,
			PxU32 elementCount = 0,
			bool filterAllElements = true)
		{
			if(!findEntry(softCore) ||
				(!filterAllElements &&
					(!elementIndices || elementCount == 0)))
				return PX_MAX_U32;
			PxU32 handle = mNextRigidActorFilterHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextRigidActorFilterHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;
			mRigidActorFilters.pushBack(
				RigidActorFilterEntry(
					softCore, rigidCore,
					elementIndices, elementCount,
					handle, filterAllElements));
			return handle;
		}

		PxU32 addVolumeRigidActorFilter(
			DeformableVolumeCore& softCore,
			ActorCore& rigidCore,
			const PxU32* collisionElementIndices,
			PxU32 collisionElementCount,
			bool filterAllElements)
		{
			if(filterAllElements)
				return addRigidActorFilter(
					softCore, rigidCore);
			Entry* entry = findEntry(softCore);
			if(!entry || entry->kind != eVOLUME ||
				!collisionElementIndices ||
				collisionElementCount == 0 ||
				!entry->collisionMesh ||
				!entry->simulationMesh ||
				!entry->auxData)
				return PX_MAX_U32;
			const PxU32 publicElementCount =
				entry->collisionMesh->getNbTetrahedrons();
			for(PxU32 selectedIndex = 0;
				selectedIndex < collisionElementCount;
				++selectedIndex)
			{
				if(collisionElementIndices[selectedIndex] >= publicElementCount)
					return PX_MAX_U32;
			}
			return addRigidActorFilter(
				softCore, rigidCore,
				collisionElementIndices,
				collisionElementCount, false);
		}

		void removeRigidActorFilter(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mRigidActorFilters.size(); ++i)
			{
				const RigidActorFilterEntry& filter =
					mRigidActorFilters[i];
				if(filter.softCore == &softCore &&
					filter.handle == handle)
				{
					mRigidActorFilters.replaceWithLast(i);
					return;
				}
			}
		}

		PxU32 addCompiledDeformablePairFilter(
			ActorCore& core0,
			ActorCore& core1,
			const PxU32* elementIndices0,
			const PxU32* elementIndices1,
			PxU32 pairCount)
		{
			if(&core0 == &core1 || !elementIndices0 ||
				!elementIndices1 || pairCount == 0)
				return PX_MAX_U32;
			PxU32 handle = mNextDeformablePairFilterHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextDeformablePairFilterHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;
			mDeformablePairFilters.pushBack(
				DeformablePairFilterEntry(
					core0, core1, elementIndices0,
					elementIndices1, pairCount, handle));
			return handle;
		}

		bool expandVolumeCollisionElement(
			const Entry& entry,
			PxU32 collisionElement,
			PxArray<PxU32>& simulationElements) const
		{
			simulationElements.clear();
			if(collisionElement == eELEMENT_FILTER_ALL)
			{
				simulationElements.pushBack(eELEMENT_FILTER_ALL);
				return true;
			}
			if(entry.kind != eVOLUME ||
				!entry.collisionMesh ||
				!entry.simulationMesh ||
				!entry.auxData)
				return false;
			const PxU32 collisionElementCount =
				entry.collisionMesh->getNbTetrahedrons();
			if(collisionElement >= collisionElementCount)
				return false;
			simulationElements.pushBack(collisionElement);
			return true;
		}

		PxU32 addSurfaceSurfaceFilter(
			DeformableSurfaceCore& core0,
			DeformableSurfaceCore& core1,
			const PxU32* elementIndices0,
			const PxU32* elementIndices1,
			PxU32 pairCount)
		{
			if(&core0 == &core1 || !elementIndices0 ||
				!elementIndices1 || pairCount == 0)
				return PX_MAX_U32;
			Entry* entry0 = findEntry(core0);
			Entry* entry1 = findEntry(core1);
			if(!entry0 || !entry1 ||
				entry0->kind != eSURFACE ||
				entry1->kind != eSURFACE ||
				!entry0->triangleMesh ||
				!entry1->triangleMesh)
				return PX_MAX_U32;
			const PxU32 elementCount0 =
				entry0->triangleMesh->getNbTriangles();
			const PxU32 elementCount1 =
				entry1->triangleMesh->getNbTriangles();
			for(PxU32 i = 0; i < pairCount; ++i)
			{
				if((elementIndices0[i] != eELEMENT_FILTER_ALL &&
						elementIndices0[i] >= elementCount0) ||
					(elementIndices1[i] != eELEMENT_FILTER_ALL &&
						elementIndices1[i] >= elementCount1))
					return PX_MAX_U32;
			}
			return addCompiledDeformablePairFilter(
				core0, core1, elementIndices0,
				elementIndices1, pairCount);
		}

		PxU32 addVolumeSurfaceFilter(
			DeformableVolumeCore& volumeCore,
			DeformableSurfaceCore& surfaceCore,
			const PxU32* volumeCollisionElements,
			const PxU32* surfaceElements,
			PxU32 pairCount)
		{
			if(!volumeCollisionElements || !surfaceElements ||
				pairCount == 0)
				return PX_MAX_U32;
			Entry* volumeEntry = findEntry(volumeCore);
			Entry* surfaceEntry = findEntry(surfaceCore);
			if(!volumeEntry || !surfaceEntry ||
				volumeEntry->kind != eVOLUME ||
				surfaceEntry->kind != eSURFACE ||
				!surfaceEntry->triangleMesh)
				return PX_MAX_U32;
			const PxU32 surfaceElementCount =
				surfaceEntry->triangleMesh->getNbTriangles();
			PxArray<PxU32> compiledVolumeElements;
			PxArray<PxU32> compiledSurfaceElements;
			PxArray<PxU32> expandedVolumeElements;
			for(PxU32 pairIndex = 0;
				pairIndex < pairCount; ++pairIndex)
			{
				const PxU32 surfaceElement =
					surfaceElements[pairIndex];
				if(surfaceElement != eELEMENT_FILTER_ALL &&
					surfaceElement >= surfaceElementCount)
					return PX_MAX_U32;
				if(!expandVolumeCollisionElement(
					*volumeEntry,
					volumeCollisionElements[pairIndex],
					expandedVolumeElements))
					return PX_MAX_U32;
				for(PxU32 i = 0;
					i < expandedVolumeElements.size(); ++i)
				{
					compiledVolumeElements.pushBack(
						expandedVolumeElements[i]);
					compiledSurfaceElements.pushBack(
						surfaceElement);
				}
			}
			return addCompiledDeformablePairFilter(
				volumeCore, surfaceCore,
				compiledVolumeElements.begin(),
				compiledSurfaceElements.begin(),
				compiledVolumeElements.size());
		}

		PxU32 addVolumeVolumeFilter(
			DeformableVolumeCore& core0,
			DeformableVolumeCore& core1,
			const PxU32* collisionElements0,
			const PxU32* collisionElements1,
			PxU32 pairCount)
		{
			if(&core0 == &core1 || !collisionElements0 ||
				!collisionElements1 || pairCount == 0)
				return PX_MAX_U32;
			Entry* entry0 = findEntry(core0);
			Entry* entry1 = findEntry(core1);
			if(!entry0 || !entry1 ||
				entry0->kind != eVOLUME ||
				entry1->kind != eVOLUME)
				return PX_MAX_U32;
			PxArray<PxU32> compiledElements0;
			PxArray<PxU32> compiledElements1;
			PxArray<PxU32> expandedElements0;
			PxArray<PxU32> expandedElements1;
			for(PxU32 pairIndex = 0;
				pairIndex < pairCount; ++pairIndex)
			{
				if(!expandVolumeCollisionElement(
						*entry0,
						collisionElements0[pairIndex],
						expandedElements0) ||
					!expandVolumeCollisionElement(
						*entry1,
						collisionElements1[pairIndex],
						expandedElements1))
					return PX_MAX_U32;
				for(PxU32 i = 0;
					i < expandedElements0.size(); ++i)
				{
					for(PxU32 j = 0;
						j < expandedElements1.size(); ++j)
					{
						compiledElements0.pushBack(
							expandedElements0[i]);
						compiledElements1.pushBack(
							expandedElements1[j]);
					}
				}
			}
			return addCompiledDeformablePairFilter(
				core0, core1,
				compiledElements0.begin(),
				compiledElements1.begin(),
				compiledElements0.size());
		}

		void removeDeformablePairFilter(
			ActorCore& core,
			PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mDeformablePairFilters.size(); ++i)
			{
				const DeformablePairFilterEntry& filter =
					mDeformablePairFilters[i];
				if((filter.core0 == &core ||
						filter.core1 == &core) &&
					filter.handle == handle)
				{
					mDeformablePairFilters.replaceWithLast(i);
					return;
				}
			}
		}

		void removeEntry(ActorCore& core)
		{
			for(PxU32 entryIndex = 0;
				entryIndex < mEntries.size(); entryIndex++)
			{
				Entry& entry = mEntries[entryIndex];
				if(entry.getActorCore() != &core)
					continue;

				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				removeNativeIslandEdgesForSoft(core);
				removePrescribedAttachmentsForSoft(core);
				removeRigidAttachmentsForSoft(core);
				removeArticulationAttachmentsForSoft(core);
				removeSoftPairAttachmentsForSoft(core);
				removeWorldPinsForCore(core);
				mIslandManager.removeNode(entry.islandNode);
				entry.destroyIslandObject();

				const PxU32 removedParticleStart =
					getParticleStart(entry);
				const PxU32 removedParticleCount =
					getParticleCount(entry);
				for(PxU32 i = 0; i < mEntries.size(); i++)
				{
					if(i == entryIndex)
						continue;
					Entry& remainingEntry = mEntries[i];
					const PxU32 remainingParticleStart =
						getParticleStart(remainingEntry);
					const PxU32 remainingParticleCount =
						getParticleCount(remainingEntry);
					if(remainingParticleStart <
						removedParticleStart + removedParticleCount)
						continue;
					const PxU32 rebasedParticleStart =
						remainingParticleStart -
						removedParticleCount;
					const bool rebased =
						rebaseSoftBodyParticleRangeInPlace(
							mBodies[remainingEntry.bodyIndex],
							remainingParticleStart,
							remainingParticleCount,
							rebasedParticleStart);
					PX_ASSERT(rebased);
					PX_UNUSED(rebased);
				}
				mParticles.removeRange(
					removedParticleStart, removedParticleCount);
				mContacts.clear();
				mWorkspace.reset();

				const PxU32 removedBodyIndex = entry.bodyIndex;
				const PxU32 lastBodyIndex = mBodies.size() - 1;
				if(removedBodyIndex != lastBodyIndex)
				{
					for(PxU32 i = 0; i < mEntries.size(); i++)
					{
						if(mEntries[i].bodyIndex == lastBodyIndex)
						{
							mEntries[i].bodyIndex = removedBodyIndex;
							break;
						}
					}
				}
				PX_ASSERT(
					mSelfCollisionAdjacencies.size() == mBodies.size());
				mSelfCollisionAdjacencies.replaceWithLast(
					removedBodyIndex);
				mSelfCollisionEnabled.clear();
				mBodies.replaceWithLast(removedBodyIndex);
				mEntries.replaceWithLast(entryIndex);
				if(mEntries.empty())
				{
					clearNativeIslandEdges();
					mParticles.clear();
					mBodies.clear();
					mSelfCollisionAdjacencies.clear();
					mSelfCollisionEnabled.clear();
					mContacts.clear();
					mWorkspace.reset();
					mWorkspacePreflightPending = true;
					mCollisionParticles.clear();
					mCollisionBodies.clear();
					mCollisionVertexMappings.clear();
					mCollisionSelfCollisionAdjacencies.clear();
				}
				else
				{
					const bool rebuilt = rebuildCollisionDetectionScene();
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
				}
				return;
			}
		}

		void prepareIslandGeneration(
			PxReal dt, const PxVec3& gravity, bool sleepingEnabled)
		{
			if(mEntries.empty() || mParticles.empty())
				return;

			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				Dy::DeformableBodyCore& core =
					mEntries[i].getBodyCore();
				if(!sleepingEnabled ||
					core.cpuAvbdWakeRequested)
				{
					const PxReal wakeCounter =
						core.wakeCounter > 0.0f
							? core.wakeCounter
							: ScInternalWakeCounterResetValue;
					wakeEntry(
						mEntries[i], wakeCounter);
				}
				syncHostInputs(
					mEntries[i], mDeformableMaterialManager);
			}
			refreshVolumeKinematicTargets();
			refreshPrescribedAttachmentTargets();

			for(PxU32 i = 0; i < mNativeIslandEdges.size(); i++)
				mNativeIslandEdges[i].touched = false;
			for(PxU32 i = 0;
				i < mNativeSoftSoftIslandEdges.size(); i++)
				mNativeSoftSoftIslandEdges[i].touched = false;

			// Public rigid attachments are persistent island topology, unlike
			// proximity contacts. Keep their native edge alive even when the
			// attached actors have no overlapping simulation shapes.
			for(PxU32 i = 0; i < mRigidAttachments.size(); i++)
			{
				RigidAttachmentEntry& attachment =
					mRigidAttachments[i];
				Entry* softEntry =
					findEntry(*attachment.softCore);
				BodySim* bodySim =
					attachment.rigidCore->getSim();
				if(!softEntry || !bodySim ||
					bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;
				if(softEntry->sleeping && bodySim->isActive())
					wakeEntry(
						*softEntry,
						ScInternalWakeCounterResetValue);
				else if(!softEntry->sleeping &&
					!bodySim->isActive())
					attachment.rigidCore->wakeUp(
						ScInternalWakeCounterResetValue);
				ensureNativeIslandEdge(
					*softEntry, *attachment.rigidCore);
			}

			// Articulation-link attachments are persistent topology too, but
			// their solve owner is a generalized-coordinate position block.
			for(PxU32 i = 0;
				i < mArticulationAttachments.size(); i++)
			{
				ArticulationAttachmentEntry& attachment =
					mArticulationAttachments[i];
				Entry* softEntry =
					findEntry(*attachment.softCore);
				BodySim* bodySim =
					attachment.linkCore->getSim();
				if(!softEntry || !bodySim ||
					!bodySim->isArticulationLink() ||
					!bodySim->getArticulation())
					continue;
				if(softEntry->sleeping && bodySim->isActive())
					wakeEntry(
						*softEntry,
						ScInternalWakeCounterResetValue);
				else if(!softEntry->sleeping &&
					!bodySim->isActive())
					attachment.linkCore->wakeUp(
						ScInternalWakeCounterResetValue);
				ensureNativeIslandEdge(
					*softEntry, *attachment.linkCore);
			}

			// A public deformable-pair attachment is persistent island
			// topology. It must keep both soft actors in one selection even
			// after their collision bounds separate.
			for(PxU32 i = 0; i < mSoftPairAttachments.size(); i++)
			{
				SoftPairAttachmentEntry& attachment =
					mSoftPairAttachments[i];
				Entry* softEntry0 =
					findEntry(*attachment.softCore[0]);
				Entry* softEntry1 =
					findEntry(*attachment.softCore[1]);
				if(!softEntry0 || !softEntry1)
					continue;
				if(softEntry0->sleeping && !softEntry1->sleeping)
					wakeEntry(
						*softEntry0,
						ScInternalWakeCounterResetValue);
				else if(softEntry1->sleeping &&
					!softEntry0->sleeping)
					wakeEntry(
						*softEntry1,
						ScInternalWakeCounterResetValue);
				ensureNativeSoftSoftIslandEdge(
					*softEntry0, *softEntry1);
			}

			for(PxU32 softIndex = 0;
				softIndex < mEntries.size(); softIndex++)
			{
				Entry& softEntry = mEntries[softIndex];
				PxBounds3 softBounds;
				if(!computeCollisionDomainSoftBounds(softEntry, softBounds))
					continue;
				const bool speculativeCCDEnabled =
					softEntry.bodyIndex < mBodies.size() &&
					mBodies[softEntry.bodyIndex].compiled.
						speculativeCCDEnabled;
				if(speculativeCCDEnabled)
				{
					if(!expandSoftBoundsForPrediction(
						softEntry, dt, gravity, softBounds))
						continue;
				}
				else if(!computePredictedCollisionDomainSoftBounds(
					softEntry, dt, gravity, softBounds))
				{
					// Non-CCD AVBD uses an endpoint-only DCD admission below.
					// Do not retain the source-pose AABB here: that would turn
					// the topology decision into a swept candidate.
					continue;
				}

				for(PxU32 shapeIndex = 0;
					shapeIndex < mDynamicShapes.size(); shapeIndex++)
				{
					const DynamicShapeEntry& dynamicEntry =
						mDynamicShapes[shapeIndex];
					Dy::AvbdRigidBox box;
					PxBounds3 rigidBounds;
					if(compileDynamicBox(dynamicEntry, box))
					{
						rigidBounds = computeBoxBounds(box);
						// Keep non-CCD topology admission at one discrete endpoint.
						// A sphere bound is used only for the broad phase so arbitrary
						// rigid rotation cannot make us miss that endpoint; the narrow
						// phase below remains an OBB current-pose query, never a sweep.
						if(!speculativeCCDEnabled)
						{
							if(!computeDynamicEndpointEnvelopeBounds(
									dynamicEntry, box.center,
									box.halfExtent.magnitude(), dt, gravity,
									rigidBounds))
								continue;
						}
					}
					else
					{
						Dy::AvbdRigidSphere sphere;
						if(compileDynamicSphere(
								dynamicEntry, sphere))
						{
							rigidBounds = computeSphereBounds(sphere);
							if(!speculativeCCDEnabled)
							{
								if(!computeDynamicEndpointEnvelopeBounds(
										dynamicEntry, sphere.center,
										sphere.radius, dt, gravity,
										rigidBounds))
									continue;
							}
							// A dynamic sphere that crosses a soft actor within
							// one frame needs native island topology before the
							// solver-body prediction is available. Bound the
							// current/predicted body-center segment by the sphere
							// radius plus its shape offset; this is conservative
							// for arbitrary rotation and is public-flag gated.
							BodySim* sphereBodySim =
								dynamicEntry.core->getSim();
							if(speculativeCCDEnabled &&
								sphereBodySim &&
								!sphereBodySim->isKinematic())
							{
								const PxsBodyCore& bodyCore =
									dynamicEntry.core->getCore();
								const PxVec3 bodyCenter =
									bodyCore.body2World.p;
								const PxReal shapeOffset =
									(sphere.center - bodyCenter).
										magnitude();
								const PxReal envelopeRadius =
									sphere.radius + shapeOffset;
								const PxVec3 predictedBodyCenter =
									bodyCenter +
									bodyCore.linearVelocity * dt +
									(bodyCore.disableGravity
										? PxVec3(0.0f)
										: gravity * (dt * dt));
								if(!PxIsFinite(shapeOffset) ||
									!PxIsFinite(envelopeRadius) ||
									!predictedBodyCenter.isFinite())
									continue;
								const PxVec3 envelopeExtent(
									envelopeRadius);
								rigidBounds.include(
									bodyCenter - envelopeExtent);
								rigidBounds.include(
									bodyCenter + envelopeExtent);
								rigidBounds.include(
									predictedBodyCenter -
										envelopeExtent);
								rigidBounds.include(
									predictedBodyCenter +
										envelopeExtent);
							}
						}
						else
						{
							Dy::AvbdRigidCapsule capsule;
							if(compileDynamicCapsule(
									dynamicEntry, capsule))
							{
								rigidBounds =
									computeCapsuleBounds(capsule);
								if(!speculativeCCDEnabled)
								{
									if(!computeDynamicEndpointEnvelopeBounds(
											dynamicEntry, capsule.center,
											capsule.radius + capsule.halfHeight,
											dt, gravity, rigidBounds))
										continue;
								}
								BodySim* capsuleBodySim =
									dynamicEntry.core->getSim();
								if(speculativeCCDEnabled &&
									capsuleBodySim)
								{
									if(capsuleBodySim->isKinematic())
									{
										Dy::AvbdRigidCapsule
											previousCapsule = capsule;
										previousCapsule.center =
											capsule.previousCenter;
										previousCapsule.rotation =
											capsule.previousRotation;
										const PxBounds3 previousBounds =
											computeCapsuleBounds(
												previousCapsule);
										rigidBounds.include(
											previousBounds.minimum);
										rigidBounds.include(
											previousBounds.maximum);
										if(!Dy::
											avbdAreSweepRotationsEquivalent(
												capsule.
													previousRotation,
												capsule.rotation))
										{
											// Endpoint AABBs do not contain
											// the arc swept by a rotating
											// capsule. A center-segment
											// sphere envelope is conservative
											// for every intermediate
											// orientation and is only enabled
											// for a speculative source.
											const PxVec3 rotationExtent(
												capsule.radius +
													capsule.halfHeight);
											rigidBounds.include(
												capsule.previousCenter -
													rotationExtent);
											rigidBounds.include(
												capsule.previousCenter +
													rotationExtent);
											rigidBounds.include(
												capsule.center -
													rotationExtent);
											rigidBounds.include(
												capsule.center +
													rotationExtent);
										}
									}
									else
									{
										const PxsBodyCore& bodyCore =
											dynamicEntry.core->getCore();
										const PxVec3 bodyCenter =
											bodyCore.body2World.p;
										const PxReal shapeOffset =
											(capsule.center -
												bodyCenter).magnitude();
										const PxReal envelopeRadius =
											capsule.radius +
											capsule.halfHeight +
											shapeOffset;
										const PxVec3 predictedBodyCenter =
											bodyCenter +
											bodyCore.linearVelocity * dt +
											(bodyCore.disableGravity
												? PxVec3(0.0f)
												: gravity * (dt * dt));
										if(!PxIsFinite(shapeOffset) ||
											!PxIsFinite(
												envelopeRadius) ||
											!predictedBodyCenter.isFinite())
											continue;
										const PxVec3 envelopeExtent(
											envelopeRadius);
										rigidBounds.include(
											bodyCenter - envelopeExtent);
										rigidBounds.include(
											bodyCenter + envelopeExtent);
										rigidBounds.include(
											predictedBodyCenter -
												envelopeExtent);
										rigidBounds.include(
											predictedBodyCenter +
												envelopeExtent);
									}
								}
							}
							else
							{
								Dy::AvbdRigidConvex convex;
								if(compileDynamicConvex(
										dynamicEntry, convex))
								{
									rigidBounds =
										computeConvexBounds(convex);
									if(!speculativeCCDEnabled)
									{
										if(!computeDynamicEndpointEnvelopeBounds(
												dynamicEntry, convex.center,
												convex.localRadius, dt, gravity,
												rigidBounds))
											continue;
									}
									BodySim* convexBodySim =
										dynamicEntry.core->getSim();
									if(speculativeCCDEnabled &&
										convexBodySim)
									{
										if(convexBodySim->isKinematic())
										{
											Dy::AvbdRigidConvex
												previousConvex = convex;
											previousConvex.center =
												convex.previousCenter;
											previousConvex.rotation =
												convex.previousRotation;
											const PxBounds3 previousBounds =
												computeConvexBounds(
													previousConvex);
											rigidBounds.include(
												previousBounds.minimum);
											rigidBounds.include(
												previousBounds.maximum);
											if(!Dy::
												avbdAreSweepRotationsEquivalent(
													convex.
														previousRotation,
													convex.rotation))
											{
												// The convex is contained by
												// a shape-center sphere with
												// localRadius for every
												// intermediate orientation.
												const PxVec3 rotationExtent(
													convex.localRadius);
												rigidBounds.include(
													convex.previousCenter -
														rotationExtent);
												rigidBounds.include(
													convex.previousCenter +
														rotationExtent);
												rigidBounds.include(
													convex.center -
														rotationExtent);
												rigidBounds.include(
													convex.center +
														rotationExtent);
											}
										}
										else
										{
											const PxsBodyCore& bodyCore =
												dynamicEntry.core->getCore();
											const PxVec3 bodyCenter =
												bodyCore.body2World.p;
											const PxReal shapeOffset =
												(convex.center -
													bodyCenter).magnitude();
											const PxReal envelopeRadius =
												convex.localRadius +
												shapeOffset;
											const PxVec3
												predictedBodyCenter =
													bodyCenter +
													bodyCore.linearVelocity *
														dt +
													(bodyCore.disableGravity
														? PxVec3(0.0f)
														: gravity *
															(dt * dt));
											if(!PxIsFinite(shapeOffset) ||
												!PxIsFinite(
													envelopeRadius) ||
												!predictedBodyCenter.
													isFinite())
												continue;
											const PxVec3 envelopeExtent(
												envelopeRadius);
											rigidBounds.include(
												bodyCenter -
													envelopeExtent);
											rigidBounds.include(
												bodyCenter +
													envelopeExtent);
											rigidBounds.include(
												predictedBodyCenter -
													envelopeExtent);
											rigidBounds.include(
												predictedBodyCenter +
													envelopeExtent);
										}
									}
								}
								else
								{
									Dy::AvbdRigidTriangleSurface
										triangleSurface;
									if(!compileDynamicTriangleSurface(
											dynamicEntry,
											triangleSurface))
										continue;
									rigidBounds =
										computeTriangleSurfaceBounds(
											triangleSurface);
									BodySim* triangleSurfaceBodySim =
										dynamicEntry.core->getSim();
									if(speculativeCCDEnabled &&
										triangleSurfaceBodySim &&
										triangleSurfaceBodySim->
											isKinematic())
									{
										Dy::AvbdRigidTriangleSurface
											previousSurface =
												triangleSurface;
										previousSurface.center =
											triangleSurface.
												previousCenter;
										previousSurface.rotation =
											triangleSurface.
												previousRotation;
										const PxBounds3 previousBounds =
											computeTriangleSurfaceBounds(
												previousSurface);
										rigidBounds.include(
											previousBounds.minimum);
										rigidBounds.include(
											previousBounds.maximum);
										if(!Dy::
											avbdAreSweepRotationsEquivalent(
												triangleSurface.
													previousRotation,
												triangleSurface.rotation))
										{
											// Endpoint AABBs do not contain
											// the arc swept by a rotating
											// triangle surface. Every baked
											// vertex stays inside the
											// shape-center localRadius sphere.
											const PxVec3 rotationExtent(
												triangleSurface.
													localRadius);
											rigidBounds.include(
												triangleSurface.
													previousCenter -
													rotationExtent);
											rigidBounds.include(
												triangleSurface.
													previousCenter +
													rotationExtent);
											rigidBounds.include(
												triangleSurface.center -
													rotationExtent);
											rigidBounds.include(
												triangleSurface.center +
													rotationExtent);
										}
									}
								}
								if(rigidBounds.isEmpty())
									continue;
							}
						}
					}

					PxBounds3 candidateBounds = softBounds;
					const PxReal wakeMargin =
						2.0f * mContactParams.contactRadius +
						PxMax(
							dynamicEntry.shape->getContactOffset(),
							0.0f);
					candidateBounds.fattenSafe(wakeMargin);
					if(!candidateBounds.intersects(rigidBounds))
						continue;

					BodySim* bodySim = dynamicEntry.core->getSim();
					bool rigidNodeActive = false;
					if(bodySim)
					{
						const PxNodeIndex rigidNode =
							bodySim->getNodeIndex();
						const IG::IslandSim& accurateIslandSim =
							mIslandManager.getAccurateIslandSim();
						rigidNodeActive =
							rigidNode.isValid() &&
							rigidNode.index() <
								accurateIslandSim.getNbNodes() &&
							accurateIslandSim.getNode(
								rigidNode).isActive();
					}
					if(softEntry.sleeping && bodySim &&
						(bodySim->isActive() || rigidNodeActive))
					{
						wakeEntry(
							softEntry,
							ScInternalWakeCounterResetValue);
					}
					// Kinematics are prescribed one-way position targets.
					// They wake overlapping soft actors but must not enter
					// the two-sided rigid 6x6 AVBD island objective.
					if(bodySim && !bodySim->isKinematic())
						ensureNativeIslandEdge(
							softEntry, *dynamicEntry.core);
				}
			}

			for(PxU32 softIndex0 = 0;
				softIndex0 < mEntries.size(); softIndex0++)
			{
				Entry& softEntry0 = mEntries[softIndex0];
				PxBounds3 softBounds0;
				if(!computeCollisionDomainSoftBounds(softEntry0, softBounds0))
					continue;
				const PxReal wakeMargin =
					PxMax(mContactParams.contactRadius, 0.0f);
				softBounds0.fattenSafe(wakeMargin);

				for(PxU32 softIndex1 = softIndex0 + 1;
					softIndex1 < mEntries.size(); softIndex1++)
				{
					Entry& softEntry1 = mEntries[softIndex1];
					PxBounds3 softBounds1;
					if(!computeCollisionDomainSoftBounds(softEntry1, softBounds1) ||
						!softBounds0.intersects(softBounds1))
						continue;

					const bool soft0WasSleeping =
						softEntry0.sleeping;
					const bool soft1WasSleeping =
						softEntry1.sleeping;
					if(soft0WasSleeping && !soft1WasSleeping)
					{
						wakeEntry(
							softEntry0,
							ScInternalWakeCounterResetValue);
					}
					else if(soft1WasSleeping && !soft0WasSleeping)
					{
						wakeEntry(
							softEntry1,
							ScInternalWakeCounterResetValue);
					}
					ensureNativeSoftSoftIslandEdge(
						softEntry0, softEntry1);
				}
			}

			for(PxU32 i = mNativeIslandEdges.size();
				i > 0; i--)
			{
				NativeIslandEdgeEntry& edge =
					mNativeIslandEdges[i - 1];
				if(!edge.touched)
				{
					mIslandManager.removeConnection(
						edge.edgeIndex);
					mNativeIslandEdges.replaceWithLast(
						i - 1);
				}
			}
			for(PxU32 i = mNativeSoftSoftIslandEdges.size();
				i > 0; i--)
			{
				NativeSoftSoftIslandEdgeEntry& edge =
					mNativeSoftSoftIslandEdges[i - 1];
				if(!edge.touched)
				{
					mIslandManager.removeConnection(
						edge.edgeIndex);
					mNativeSoftSoftIslandEdges.replaceWithLast(
						i - 1);
				}
			}
		}

		virtual bool prepareSoftIslandSelections(
			Dy::AvbdSolverBody* solverBodies,
			PxsRigidBody* const* rigidBodies,
			Dy::FeatherstoneArticulation* const*
				articulationForBody,
			const PxU32* linkIndexForBody,
			const PxU32* islandBodyStarts,
			const PxU32* islandBodyCounts,
			const PxU32* activeIslandIds,
			PxU32 islandCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdSoftIslandSelection>& selections) PX_OVERRIDE
		{
			selections.clear();
			mDynamicsOwnsStep = false;
			mDynamicsSelectedEntryCount = 0;
			if(dt <= 0.0f || mBodies.empty() || !solverBodies ||
				!rigidBodies ||
				(!mArticulationAttachments.empty() &&
				 (!articulationForBody || !linkIndexForBody)) ||
				!islandBodyStarts ||
				!islandBodyCounts || !activeIslandIds ||
				islandCount == 0 || mEntries.empty())
				return false;

			const IG::IslandSim& islandSim =
				mIslandManager.getAccurateIslandSim();
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				Entry& entry = mEntries[i];
				if(!entry.sleeping)
					continue;
				const PxNodeIndex node = entry.islandNode;
				if(!node.isValid() ||
					node.index() >= islandSim.getNbNodes())
					return false;
				const IG::IslandId entryIslandId =
					islandSim.getIslandIds()[node.index()];
				for(PxU32 islandIndex = 0;
					islandIndex < islandCount; islandIndex++)
				{
					if(activeIslandIds[islandIndex] ==
							entryIslandId &&
						islandBodyCounts[islandIndex] > 0)
					{
						wakeEntry(
							entry,
							ScInternalWakeCounterResetValue);
						break;
					}
				}
			}

			PxU32 awakeEntryCount = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				if(mEntries[i].sleeping)
					continue;
				syncHostInputs(
					mEntries[i], mDeformableMaterialManager);
				awakeEntryCount++;
			}
			if(awakeEntryCount == 0)
				return false;

			// AVBD consumes the articulation response strictly as a
			// generalized inverse-mass operator for its position owner.
			// Refresh it once per attached articulation at prep time; do not
			// enter Featherstone's velocity-impulse solve.
			for(PxU32 attachmentIndex = 0;
				attachmentIndex <
					mArticulationAttachments.size();
				attachmentIndex++)
			{
				BodySim* linkSim =
					mArticulationAttachments[attachmentIndex].
						linkCore->getSim();
				Dy::FeatherstoneArticulation* articulation =
					linkSim ? linkSim->getArticulation() : NULL;
				if(!articulation)
					return false;
				bool alreadyPrepared = false;
				for(PxU32 priorIndex = 0;
					priorIndex < attachmentIndex; priorIndex++)
				{
					BodySim* priorLinkSim =
						mArticulationAttachments[priorIndex].
							linkCore->getSim();
					if(priorLinkSim &&
						priorLinkSim->getArticulation() ==
							articulation)
					{
						alreadyPrepared = true;
						break;
					}
				}
				if(!alreadyPrepared)
					articulation->
						prepareAvbdGeneralizedPositionResponse();
			}

			compileWorldStatics(mRigidMaterialManager);
			for(PxU32 i = 0; i < mIslandSelectionStorages.size(); i++)
			{
				mIslandSelectionStorages[i]->touched = false;
				mIslandSelectionStorages[i]->entryIndices.clear();
				mIslandSelectionStorages[i]->selectedIsland = PX_MAX_U32;
			}

			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				if(mEntries[i].sleeping)
					continue;
				const PxNodeIndex node = mEntries[i].islandNode;
				if(!node.isValid() ||
					node.index() >= islandSim.getNbNodes())
					return false;
				const IG::IslandId entryIslandId =
					islandSim.getIslandIds()[node.index()];
				if(entryIslandId == IG_INVALID_ISLAND)
					return false;

				IslandSelectionStorage* storage =
					acquireIslandSelectionStorage(entryIslandId);
				if(!storage)
					return false;
				storage->entryIndices.pushBack(i);
			}

			auto discardNativeSelections = [&]()
			{
				selections.clear();
				invalidateNativeIslandSelectionCaches();
			};

			PxU32 selectedEntryCount = 0;
			IslandSelectionStorage* singleNativeStorage = NULL;
			// A later all-awake promotion is safe only when every provisional
			// selection uses the same active rigid island.  Several disconnected
			// soft components may legitimately contact one dynamic rigid in the
			// same frame; they can be rebuilt as one native selection.  Distinct
			// rigid islands remain an ownership boundary and must fall back.
			PxU32 nativeSelectionIsland = PX_MAX_U32;
			bool nativeSelectionsShareIsland = true;
			bool duplicateNativeSelectionIsland = false;
			for(PxU32 storageIndex = 0;
				storageIndex < mIslandSelectionStorages.size();
				storageIndex++)
			{
				IslandSelectionStorage& storage =
					*mIslandSelectionStorages[storageIndex];
				if(!storage.touched || storage.entryIndices.empty())
					continue;

				for(PxU32 islandIndex = 0;
					islandIndex < islandCount; islandIndex++)
				{
					if(activeIslandIds[islandIndex] ==
						storage.nativeIslandId)
					{
						storage.selectedIsland = islandIndex;
						break;
					}
				}
				// A soft-rigid edge created during this frame's predictive
				// topology pass is visible to the speculative graph before
				// the accurate graph has necessarily merged its islands.
				// When the soft side still resolves to an empty native island,
				// bridge selection to the one unambiguous active rigid island
				// named by that same native edge. The edge remains authoritative
				// topology and the normal accurate-island merge owns later
				// frames; no out-of-island rigid index is ever fabricated.
				if(storage.selectedIsland == PX_MAX_U32 ||
					islandBodyCounts[storage.selectedIsland] == 0)
				{
					PxU32 bridgeIsland = PX_MAX_U32;
					bool bridgeAmbiguous = false;
					for(PxU32 entryOrder = 0;
						entryOrder < storage.entryIndices.size();
						entryOrder++)
					{
						const ActorCore* softCore =
							mEntries[
								storage.entryIndices[entryOrder]].
								getActorCore();
						for(PxU32 edgeIndex = 0;
							edgeIndex < mNativeIslandEdges.size();
							edgeIndex++)
						{
							const NativeIslandEdgeEntry& edge =
								mNativeIslandEdges[edgeIndex];
							if(!edge.touched ||
								edge.softCore != softCore)
								continue;
							BodySim* rigidSim =
								edge.rigidCore->getSim();
							if(!rigidSim)
								continue;
							const PxNodeIndex rigidNode =
								rigidSim->getNodeIndex();
							if(!rigidNode.isValid() ||
								rigidNode.index() >=
									islandSim.getNbNodes())
								continue;
							const IG::IslandId rigidIslandId =
								islandSim.getIslandIds()[
									rigidNode.index()];
							for(PxU32 islandIndex = 0;
								islandIndex < islandCount;
								islandIndex++)
							{
								if(activeIslandIds[islandIndex] !=
										rigidIslandId ||
									islandBodyCounts[islandIndex] == 0)
									continue;
								if(bridgeIsland == PX_MAX_U32)
									bridgeIsland = islandIndex;
								else if(bridgeIsland != islandIndex)
									bridgeAmbiguous = true;
								break;
							}
						}
					}
					if(!bridgeAmbiguous &&
						bridgeIsland != PX_MAX_U32)
						storage.selectedIsland = bridgeIsland;
				}
				if(storage.selectedIsland != PX_MAX_U32 &&
					islandBodyCounts[storage.selectedIsland] > 0)
				{
					for(PxU32 entryIndex = 0;
						entryIndex < storage.entryIndices.size();
						entryIndex++)
					{
						Entry& entry =
							mEntries[
								storage.entryIndices[entryIndex]];
						if(entry.sleeping)
							wakeEntry(
								entry,
								ScInternalWakeCounterResetValue);
					}
				}
				if(storage.selectedIsland == PX_MAX_U32 ||
					!buildIslandSelectionStorage(
						storage, solverBodies, rigidBodies,
						articulationForBody, linkIndexForBody,
						islandBodyStarts[storage.selectedIsland],
						islandBodyCounts[storage.selectedIsland],
						dt, gravity))
				{
					// This soft-only/native island has no unified rigid or
					// generalized target. Leave it for the component fallback
					// without discarding independent complete selections.
					storage.touched = false;
					storage.selectedIsland = PX_MAX_U32;
					continue;
				}

				PxU32 innerIterations = 1;
				for(PxU32 entryIndex = 0;
					entryIndex < storage.entryIndices.size();
					entryIndex++)
				{
					const Entry& entry =
						mEntries[storage.entryIndices[entryIndex]];
					innerIterations = PxMax<PxU32>(
						innerIterations,
						entry.getSolverIterationCounts() & 0xff);
				}

				Dy::AvbdSoftIslandSelection selection;
				selection.particles =
					getIslandSelectionParticles(storage);
				selection.numParticles =
					getIslandSelectionParticleCount(storage);
				selection.bodies = storage.bodies.begin();
				selection.numBodies = storage.bodies.size();
				selection.contacts = storage.contacts.begin();
				selection.numContacts = storage.contacts.size();
				selection.islandIndex = storage.selectedIsland;
				selection.iterationOverride = innerIterations;
				selection.executionPlan.particleBodyIndices =
					storage.particleBodyIndices.begin();
				selection.executionPlan.numParticleBodyIndices =
					storage.particleBodyIndices.size();
				selection.executionPlan.contactStarts =
					storage.contactStarts.begin();
				selection.executionPlan.numContactStarts =
					storage.contactStarts.size();
				selection.executionPlan.contactRefs =
					storage.contactRefs.begin();
				selection.executionPlan.numContactRefs =
					storage.contactRefs.size();
				selection.executionPlan.triangleCoreSafetyStarts =
					storage.triangleCoreSafetyStarts.begin();
				selection.executionPlan.numTriangleCoreSafetyStarts =
					storage.triangleCoreSafetyStarts.size();
				selection.executionPlan.triangleCoreSafetyRefs =
					storage.triangleCoreSafetyRefs.begin();
				selection.executionPlan.numTriangleCoreSafetyRefs =
					storage.triangleCoreSafetyRefs.size();
				selection.executionPlan.rigidTargetContactStarts =
					storage.rigidTargetContactStarts.begin();
				selection.executionPlan.numRigidTargetContactStarts =
					storage.rigidTargetContactStarts.size();
				selection.executionPlan.rigidTargetContactRefs =
					storage.rigidTargetContactRefs.begin();
				selection.executionPlan.numRigidTargetContactRefs =
					storage.rigidTargetContactRefs.size();
				selection.executionPlan.terminalRigidBoxes =
					storage.rigidBoxes.begin();
				selection.executionPlan.numTerminalRigidBoxes =
					storage.rigidBoxes.size();
				selection.executionPlan.terminalCollisionBodies =
					storage.terminalCollisionBodies.begin();
				selection.executionPlan.numTerminalCollisionBodies =
					storage.terminalCollisionBodies.size();
				selection.executionPlan.terminalCollisionVertexMappings =
					storage.terminalCollisionVertexMappings.begin();
				selection.executionPlan.numTerminalCollisionVertexMappings =
					storage.terminalCollisionVertexMappings.size();
				selection.executionPlan.terminalContactRadius =
					mContactParams.contactRadius;
				selection.executionPlan.ogcPairStates =
					storage.ogcPairStates.begin();
				selection.executionPlan.numOgcPairStates =
					storage.ogcPairStates.size();
				selection.executionPlan.ogcPairIndices =
					storage.ogcPairIndices.begin();
				selection.executionPlan.numOgcPairIndices =
					storage.ogcPairIndices.size();
				selection.executionPlan.ogcPairContactStarts =
					storage.ogcPairContactStarts.begin();
				selection.executionPlan.numOgcPairContactStarts =
					storage.ogcPairContactStarts.size();
				selection.executionPlan.ogcPairContactRefs =
					storage.ogcPairContactRefs.begin();
				selection.executionPlan.numOgcPairContactRefs =
					storage.ogcPairContactRefs.size();
				if(!selection.isComplete() ||
					!selection.executionPlan.isComplete(
						selection.numParticles))
				{
					discardNativeSelections();
					return false;
				}
				// `buildIslandSelectionStorage()` has already published the exact
				// same dt/gravity prediction into this particle buffer for swept
				// contact selection.  Only expose that lifecycle token after the
				// base support program has passed its provider-boundary check.
				selection.executionPlan.softPredictionPrepared = true;
				for(PxU32 priorIndex = 0;
					priorIndex < selections.size(); ++priorIndex)
				{
					if(selections[priorIndex].islandIndex ==
						storage.selectedIsland)
					{
						duplicateNativeSelectionIsland = true;
						break;
					}
				}
				selections.pushBack(selection);
				if(singleNativeStorage)
				{
					if(nativeSelectionIsland != storage.selectedIsland)
						nativeSelectionsShareIsland = false;
				}
				else
				{
					singleNativeStorage = &storage;
					nativeSelectionIsland = storage.selectedIsland;
				}
				selectedEntryCount += storage.entryIndices.size();
			}

			// A component fallback has no selected-entry mask today.  Letting it
			// run after a partial native selection advances selected particles and
			// contact state once through the component route and then overwrites
			// only part of that state with the native copyback.  That is not a
			// valid ownership split.  Likewise, the dynamics provider requires
			// every published selection to name a unique rigid island.  Until the
			// indexed fallback view owns a disjoint subset, both cases must become
			// one all-awake native selection or fall back completely.
			const bool needsSingleOwnerPromotion =
				selectedEntryCount != awakeEntryCount ||
				duplicateNativeSelectionIsland;
			if(needsSingleOwnerPromotion)
			{
				// A single dynamic-rigid island can safely own the entire awake
				// soft scene when every dynamic shape belongs to that same rigid
				// island. This includes distinct soft components that currently
				// touch the same free rigid through current-pose OGC. Attachments
				// are deliberately excluded: their generalized owners require the
				// original island membership to be preserved exactly.
				if(!singleNativeStorage || !nativeSelectionsShareIsland ||
					!mWorldPins.empty() ||
					!mRigidAttachments.empty() ||
					!mArticulationAttachments.empty() ||
					!mSoftPairAttachments.empty() ||
					!mPrescribedAttachments.empty() ||
					singleNativeStorage->selectedIsland >= islandCount)
				{
					discardNativeSelections();
					return false;
				}

				IslandSelectionStorage& storage = *singleNativeStorage;
				bool hasRigidBodyTarget = false;
				for(PxU32 contactIndex = 0;
					contactIndex < storage.contacts.size(); ++contactIndex)
				{
					const Dy::AvbdSoftContactGeometry& geometry =
						storage.contacts[contactIndex].geometry;
					if(geometry.hasRigidBodyTarget())
					{
						if(geometry.targetIndex >=
							islandBodyCounts[storage.selectedIsland])
						{
							discardNativeSelections();
							return false;
						}
						hasRigidBodyTarget = true;
					}
				}
				const PxU32 selectedBodyStart =
					islandBodyStarts[storage.selectedIsland];
				const PxU32 selectedBodyCount =
					islandBodyCounts[storage.selectedIsland];
				if(!hasRigidBodyTarget || selectedBodyCount == 0)
				{
					discardNativeSelections();
					return false;
				}

				// A promoted selection may only see dynamic shapes whose two-sided
				// rigid response lives in its selected island.  Static and
				// kinematic shapes remain ordinary world/prescribed targets.
				for(PxU32 shapeIndex = 0;
					shapeIndex < mDynamicShapes.size(); ++shapeIndex)
				{
					const DynamicShapeEntry& dynamicEntry =
						mDynamicShapes[shapeIndex];
					if(!dynamicEntry.core || !dynamicEntry.shape)
					{
						discardNativeSelections();
						return false;
					}
					const ShapeCore& shape = *dynamicEntry.shape;
					if(!(shape.getFlags() &
							PxShapeFlag::eSIMULATION_SHAPE))
						continue;
					BodySim* const bodySim =
						dynamicEntry.core->getSim();
					if(!bodySim)
					{
						discardNativeSelections();
						return false;
					}
					if(bodySim->isArticulationLink())
					{
						discardNativeSelections();
						return false;
					}
					if(bodySim->isKinematic())
						continue;
					const PxGeometryType::Enum geometryType =
						shape.getGeometryType();
					if(geometryType != PxGeometryType::eBOX &&
						geometryType != PxGeometryType::eSPHERE &&
						geometryType != PxGeometryType::eCAPSULE &&
						geometryType != PxGeometryType::eCONVEXMESH)
					{
						discardNativeSelections();
						return false;
					}
					PxU32 localRigidBodyIndex = PX_MAX_U32;
					if(!findRigidBodyIndexInIsland(
							*dynamicEntry.core, rigidBodies, solverBodies,
							selectedBodyStart, selectedBodyCount,
							localRigidBodyIndex))
					{
						discardNativeSelections();
						return false;
					}
				}

				for(PxU32 storageIndex = 0;
					storageIndex < mIslandSelectionStorages.size();
					++storageIndex)
				{
					IslandSelectionStorage& candidate =
						*mIslandSelectionStorages[storageIndex];
					if(&candidate != &storage)
						candidate.touched = false;
				}

				// Existing selections point into storage-owned arrays. The old
				// selection is no longer a valid view once the promoted storage is
				// rebuilt, so remove it before changing those arrays.
				selections.clear();
				storage.entryIndices.clear();
				for(PxU32 entryIndex = 0;
					entryIndex < mEntries.size(); ++entryIndex)
					if(!mEntries[entryIndex].sleeping)
						storage.entryIndices.pushBack(entryIndex);
				if(storage.entryIndices.size() != awakeEntryCount ||
					!buildIslandSelectionStorage(
						storage, solverBodies, rigidBodies,
						articulationForBody, linkIndexForBody,
						selectedBodyStart, selectedBodyCount, dt, gravity))
				{
					discardNativeSelections();
					return false;
				}

				hasRigidBodyTarget = false;
				for(PxU32 contactIndex = 0;
					contactIndex < storage.contacts.size(); ++contactIndex)
				{
					const Dy::AvbdSoftContactGeometry& geometry =
						storage.contacts[contactIndex].geometry;
					if(geometry.hasRigidBodyTarget())
					{
						if(geometry.targetIndex >= selectedBodyCount)
						{
							discardNativeSelections();
							return false;
						}
						hasRigidBodyTarget = true;
					}
				}
				if(!hasRigidBodyTarget)
				{
					discardNativeSelections();
					return false;
				}

				PxU32 innerIterations = 1;
				for(PxU32 entryIndex = 0;
					entryIndex < storage.entryIndices.size(); ++entryIndex)
				{
					const Entry& entry =
						mEntries[storage.entryIndices[entryIndex]];
					innerIterations = PxMax<PxU32>(
						innerIterations,
						entry.getSolverIterationCounts() & 0xff);
				}

				Dy::AvbdSoftIslandSelection promotedSelection;
				promotedSelection.particles =
					getIslandSelectionParticles(storage);
				promotedSelection.numParticles =
					getIslandSelectionParticleCount(storage);
				promotedSelection.bodies = storage.bodies.begin();
				promotedSelection.numBodies = storage.bodies.size();
				promotedSelection.contacts = storage.contacts.begin();
				promotedSelection.numContacts = storage.contacts.size();
				promotedSelection.islandIndex = storage.selectedIsland;
				promotedSelection.iterationOverride = innerIterations;
				promotedSelection.executionPlan.particleBodyIndices =
					storage.particleBodyIndices.begin();
				promotedSelection.executionPlan.numParticleBodyIndices =
					storage.particleBodyIndices.size();
				promotedSelection.executionPlan.contactStarts =
					storage.contactStarts.begin();
				promotedSelection.executionPlan.numContactStarts =
					storage.contactStarts.size();
				promotedSelection.executionPlan.contactRefs =
					storage.contactRefs.begin();
				promotedSelection.executionPlan.numContactRefs =
					storage.contactRefs.size();
				promotedSelection.executionPlan.triangleCoreSafetyStarts =
					storage.triangleCoreSafetyStarts.begin();
				promotedSelection.executionPlan.numTriangleCoreSafetyStarts =
					storage.triangleCoreSafetyStarts.size();
				promotedSelection.executionPlan.triangleCoreSafetyRefs =
					storage.triangleCoreSafetyRefs.begin();
				promotedSelection.executionPlan.numTriangleCoreSafetyRefs =
					storage.triangleCoreSafetyRefs.size();
				promotedSelection.executionPlan.rigidTargetContactStarts =
					storage.rigidTargetContactStarts.begin();
				promotedSelection.executionPlan.numRigidTargetContactStarts =
					storage.rigidTargetContactStarts.size();
				promotedSelection.executionPlan.rigidTargetContactRefs =
					storage.rigidTargetContactRefs.begin();
				promotedSelection.executionPlan.numRigidTargetContactRefs =
					storage.rigidTargetContactRefs.size();
				promotedSelection.executionPlan.terminalRigidBoxes =
					storage.rigidBoxes.begin();
				promotedSelection.executionPlan.numTerminalRigidBoxes =
					storage.rigidBoxes.size();
				promotedSelection.executionPlan.terminalCollisionBodies =
					storage.terminalCollisionBodies.begin();
				promotedSelection.executionPlan.numTerminalCollisionBodies =
					storage.terminalCollisionBodies.size();
				promotedSelection.executionPlan.terminalCollisionVertexMappings =
					storage.terminalCollisionVertexMappings.begin();
				promotedSelection.executionPlan.numTerminalCollisionVertexMappings =
					storage.terminalCollisionVertexMappings.size();
				promotedSelection.executionPlan.terminalContactRadius =
					mContactParams.contactRadius;
				promotedSelection.executionPlan.ogcPairStates =
					storage.ogcPairStates.begin();
				promotedSelection.executionPlan.numOgcPairStates =
					storage.ogcPairStates.size();
				promotedSelection.executionPlan.ogcPairIndices =
					storage.ogcPairIndices.begin();
				promotedSelection.executionPlan.numOgcPairIndices =
					storage.ogcPairIndices.size();
				promotedSelection.executionPlan.ogcPairContactStarts =
					storage.ogcPairContactStarts.begin();
				promotedSelection.executionPlan.numOgcPairContactStarts =
					storage.ogcPairContactStarts.size();
				promotedSelection.executionPlan.ogcPairContactRefs =
					storage.ogcPairContactRefs.begin();
				promotedSelection.executionPlan.numOgcPairContactRefs =
					storage.ogcPairContactRefs.size();
				if(!promotedSelection.isComplete() ||
					!promotedSelection.executionPlan.isComplete(
						promotedSelection.numParticles))
				{
					discardNativeSelections();
					return false;
				}
				promotedSelection.executionPlan.softPredictionPrepared = true;
				selections.clear();
				selections.pushBack(promotedSelection);
				selectedEntryCount = storage.entryIndices.size();
			}

			mDynamicsOwnsStep = !selections.empty();
			mDynamicsSelectedEntryCount =
				mDynamicsOwnsStep ? selectedEntryCount : 0;
			return mDynamicsOwnsStep;
		}

		// P5 collision-leaf tasks consume the direct simulation topology, while
		// the ordinary component lifecycle expands a cooked collision mesh back
		// to simulation-space Jacobians synchronously.  Keep that narrower P5
		// capability separate from P2/P3/P4/P6 scheduling: a distinct collision
		// mesh must not demote prediction, body-local primal work, or write-back
		// to the scalar reference path.
		bool hasDirectSimulationCollisionDomain() const
		{
			for(PxU32 entryIndex = 0; entryIndex < mEntries.size(); ++entryIndex)
			{
				const Entry& entry = mEntries[entryIndex];
				if(entry.kind == eVOLUME &&
					entry.collisionMesh != entry.simulationMesh)
					return false;
			}
			return true;
		}

		bool shouldScheduleStandaloneTaskGraph(
			PxU32 dispatcherWorkers) const
		{
			// A complete native rigid/soft selection already owns its step in
			// Dy::AvbdDynamicsContext.  Only a component-only pure-soft step may
			// be delegated here; never overlap the two ownership paths.
			const bool forceP45CausalLayerTaskFanIn =
				Dy::avbdUseCausalLayerTaskFanIn() &&
				Dy::avbdForceCausalLayerTaskFanIn();
			const bool forceP45CausalLayerTaskGraphReference =
				Dy::avbdForceCausalLayerTaskGraphReference();
			const bool forceP45CausalLayerTaskGraph =
				forceP45CausalLayerTaskFanIn ||
				forceP45CausalLayerTaskGraphReference;
			if(mDynamicsOwnsStep || mStandaloneTaskGraphEnhancedDeterminism ||
				dispatcherWorkers == 0 ||
				(!forceP45CausalLayerTaskGraph &&
					mParticles.size() < 128) || mBodies.empty())
				return false;
			if(forceP45CausalLayerTaskGraph)
				return true;

			// P2 deliberately creates one complete stage task, not a parallel
			// particle solve.  The estimate still accounts for the work that the
			// later P3/P4 graph will divide: material elements, current contact
			// set, collision sources and the fixed dispatch/barrier cost.  It is
			// independent of rigid body count, which is zero for this path.
			PxU64 materialElements = 0;
			for(PxU32 bodyIndex = 0; bodyIndex < mBodies.size(); bodyIndex++)
			{
				const Dy::AvbdSoftBodyCompiledData& compiled =
					mBodies[bodyIndex].compiled;
				materialElements += compiled.triElements.size();
				materialElements += compiled.tetElements.size();
				materialElements += compiled.bendElements.size();
			}
			const PxU64 collisionSources =
				PxU64(mWorldPlanes.size()) + PxU64(mRigidBoxes.size()) +
				PxU64(mRigidSpheres.size()) + PxU64(mRigidCapsules.size()) +
				PxU64(mRigidConvexes.size()) +
				PxU64(mRigidTriangleSurfaces.size()) +
				(mBodies.size() > 1 ? PxU64(mBodies.size() - 1) : 0);
			const PxU64 estimatedWork =
				PxU64(mParticles.size()) + materialElements * 2 +
				PxU64(mContacts.size()) * 16 + collisionSources * 32;
			const PxU64 dispatchAndBarrierCost = 512;
			return estimatedWork >= dispatchAndBarrierCost * 2;
		}

		// Resolve the component particle schedule at the Scene boundary, where
		// worker availability and the public determinism promise are known.  The
		// default production route is relaxed coloring on a useful parallel
		// workload; an explicit process policy remains available for diagnostics
		// and forced fallback.  Enhanced determinism is a contract, not a tuning
		// hint, so it always retains the ordered scalar authority.
		Dy::AvbdParticlePrimalSchedule getParticlePrimalSchedule() const
		{
			const Dy::AvbdParticlePrimalSchedule configured =
				Dy::avbdGetConfiguredParticlePrimalSchedule();
			if(mStandaloneTaskGraphEnhancedDeterminism)
				return Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR;
			if(configured != Dy::AvbdParticlePrimalSchedule::eDEFAULT)
				return configured;
			return mStandaloneTaskGraphDispatcherWorkers >= 2 &&
				mParticles.size() >= 128
				? Dy::AvbdParticlePrimalSchedule::eRELAXED_COLOR
				: Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR;
		}

		// A dense persistent soft/soft manifold has many mutually dependent
		// vertices and therefore produces many short color layers. Publishing
		// every one as a dispatcher task turns the layer barriers into the
		// dominant cost (especially with two workers). Keep the identical relaxed
		// color plan, but execute its ordered layers inline in the component
		// continuation once the previous epoch proves that this is a contact-dense
		// manifold. This follows the same batching principle used by CPU XPBD
		// solvers: parallelize useful independent batches, not synchronization
		// points between tiny constraint colors.
		bool shouldInlineDenseSoftPairColoredPrimal() const
		{
			if(mStandaloneParticlePrimalSchedule !=
				Dy::AvbdParticlePrimalSchedule::eRELAXED_COLOR ||
				mStandaloneTaskGraphDispatcherWorkers < 2)
				return false;

			static const PxU32 eMIN_SOFT_PAIR_ROWS_FOR_INLINE_BATCH = 16;
			PxU32 softPairRows = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < mContacts.size(); ++contactIndex)
			{
				const Dy::AvbdSoftContactGeometry& geometry =
					mContacts[contactIndex].geometry;
				if(geometry.source.type ==
						Dy::AvbdSoftContactSource::eSOFT_SURFACE &&
					geometry.targetKind ==
						Dy::AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
					++softPairRows >= eMIN_SOFT_PAIR_ROWS_FOR_INLINE_BATCH)
					return true;
			}
			return false;
		}

		PxU32 getStandaloneTaskGraphParticleCount() const
		{
			return mParticles.size();
		}

		void setStandaloneTaskGraphExecutionPolicy(
			PxU32 workerCount, bool enhancedDeterminism)
		{
			// A new Scene simulation cannot overlap the prior continuation.  Reset
			// the boundary-only counters here, before any child can be submitted.
			mStandaloneTaskGraphTelemetry.reset(workerCount);
			mStandaloneTaskGraphDispatcherWorkers = workerCount;
			mStandaloneTaskGraphEnhancedDeterminism = enhancedDeterminism;
			mStandaloneParticlePrimalSchedule =
				getParticlePrimalSchedule();
			// A scene may acquire its dispatcher after bodies were added. Reserve
			// the optional graph storage at that Scene boundary (never from a
			// worker) so changing from one worker to a relaxed production policy
			// cannot silently publish an incomplete plan for the whole frame.
			if(Dy::avbdUsesColoredParticlePrimalSchedule(
				mStandaloneParticlePrimalSchedule))
				mWorkspace.reserve(
					mParticles.size(), estimateInitialComponentContactCapacity(),
					mStandaloneParticlePrimalSchedule);
		}

		bool canUseIndependentBodySweepTaskFanIn() const
		{
			if(mStandaloneTaskGraphDispatcherWorkers < 2 ||
				Dy::avbdDisableIndependentBodySweepTaskFanIn() ||
				Dy::avbdUseSceneRedetectionBridge() || mBodies.size() < 2 ||
				!mContacts.empty() || !mWorldPlanes.empty() ||
				!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
				!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				!mRigidAttachments.empty() ||
				!mArticulationAttachments.empty() ||
				!mSoftPairAttachments.empty() ||
				!mPrescribedAttachments.empty() ||
				mSelfCollisionEnabled.size() != mBodies.size())
				return false;
			for(PxU32 bodyIndex = 0; bodyIndex < mBodies.size(); bodyIndex++)
			{
				// A P6 task owns a whole soft body.  With an empty contact epoch,
				// self-collision configuration and kinematic pins remain strictly
				// body-local: their primal writes are confined to this task's
				// particle range and their AL dual update stays at the parent
				// barrier.  Do not reject a production fast path merely because it
				// cannot reproduce the legacy global traversal.  Attachments still
				// couple another generalized owner and must remain serial here.
				if(!mBodies[bodyIndex].runtime.attachments.empty())
					return false;
			}
			return true;
		}

		// P3 prediction setup: all mutable Scene/OGC state up to the exact
		// low-level prediction boundary is complete when this returns. The
		// resulting particle write set is disjoint by whole Entry.
		bool prepareStandaloneComponentSolve(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager,
			bool sleepingEnabled)
		{
			PX_UNUSED(gravity);
			mStandaloneComponentPostSolvePending = false;
			mStandaloneComponentSolvePrepared = false;
			mLastStepStats.reset();
			if(mCollisionStatsEnabled)
				mLastCollisionStats = Dy::AvbdSoftCollisionStats();
			mLastComponentFallbackSteps = 0;
			mLastNativeIslandSteps = 0;
			if(dt <= 0.0f || mBodies.empty())
				return false;

			PxU32 awakeEntryCount = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
				if(!mEntries[i].sleeping)
					awakeEntryCount++;
			if(awakeEntryCount == 0 && sleepingEnabled)
			{
				mDynamicsOwnsStep = false;
				mDynamicsSelectedEntryCount = 0;
				return false;
			}

			// A native rigid/soft island owns both its solve results and its
			// selection restore.  P3 must never split that path.
			PX_ASSERT(!mDynamicsOwnsStep);
			if(mDynamicsOwnsStep)
				return false;

			// This entry reaches the asynchronous component task graph instead of
			// `step()`.  Preserve the same single-owner rule as the serial
			// fallback before it starts mutating canonical particle/contact state.
			invalidateNativeIslandSelectionCaches();
			prepareComponentFallback(materialManager, rigidMaterialManager);
			mStandaloneStepStats.reset();
			Dy::avbdStepSoftBodies(
				mParticles.begin(), mParticles.size(),
				mBodies.begin(), mBodies.size(),
				mContacts.begin(), mContacts.size(),
				dt, gravity,
				mComponentFallbackPlan.outerIterations,
				mComponentFallbackPlan.innerIterations,
				1000.0f,
				redetectContacts, &mContacts, this,
				0.92f, &mStandaloneStepStats, &mWorkspace,
				mComponentFallbackPlan.totalPositionIterations,
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(),
				mSelfCollisionEnabled.begin(),
				&mContactParams,
				Dy::AvbdSoftBodyStepExecutionMode::ePREPARE);
			mStandaloneComponentSolvePrepared = true;
			return true;
		}

		void predictStandaloneComponentRange(
			PxU32 entryBegin, PxU32 entryEnd,
			PxReal dt, const PxVec3& gravity)
		{
			PX_ASSERT(mStandaloneComponentSolvePrepared);
			PX_ASSERT(entryBegin <= entryEnd && entryEnd <= mEntries.size());
			// Soft-soft swept admission selects its path if either endpoint opts
			// into speculative CCD.  Keep the adaptive initial guess component
			// wide so one P3 entry cannot shorten only its side of that sweep.
			const bool useRigidInitialGuess =
				Dy::avbdCanUseSoftRigidPrimalInitialization(
					mParticles.begin(), mParticles.size(),
					mBodies.begin(), mBodies.size());
			const bool useAdaptiveInitialGuess = !useRigidInitialGuess &&
				Dy::avbdCanUseSoftAdaptivePrimalInitialization(
					mParticles.begin(), mParticles.size(),
					mBodies.begin(), mBodies.size());
			for(PxU32 entryIndex = entryBegin; entryIndex < entryEnd;
				entryIndex++)
			{
				const Entry& entry = mEntries[entryIndex];
				PX_ASSERT(entry.bodyIndex < mBodies.size());
				PX_ASSERT(entry.bodyIndex <
					mWorkspace.contact.softBodyBounds.size());
				PX_ASSERT(entry.bodyIndex <
					mWorkspace.contact.softBodyBoundsReady.size());
				Dy::avbdPredictSoftBodyParticles(
					mParticles.begin() + getParticleStart(entry),
					getParticleCount(entry), dt, gravity,
					useAdaptiveInitialGuess);
				if(useRigidInitialGuess)
					Dy::avbdApplySoftBodyRigidPrimalInitialGuess(
						mParticles.begin(), mParticles.size(),
						mBodies[entry.bodyIndex]);
				// P3 Slice 3: this entry exclusively owns its body and the
				// corresponding pre-sized bounds slot. The later continuation
				// alone marks the whole cache valid after every child completes.
				Dy::avbdComputeSoftBodyBounds(
					mParticles.begin(), mBodies[entry.bodyIndex],
					mWorkspace.contact.softBodyBounds[entry.bodyIndex]);
				mWorkspace.contact.softBodyBoundsReady[entry.bodyIndex] = 1;
			}
		}

		void predictStandaloneComponent(
			PxReal dt, const PxVec3& gravity)
		{
			predictStandaloneComponentRange(0, mEntries.size(), dt, gravity);
		}

		PxU32 getStandalonePredictionTaskCount(
			PxU32 dispatcherWorkers) const
		{
			if(!mStandaloneComponentSolvePrepared || mDynamicsOwnsStep ||
				dispatcherWorkers < 2)
				return 0;
			PxU32 awakeEntryCount = 0;
			PxU64 awakeParticleCount = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				const Entry& entry = mEntries[i];
				if(entry.sleeping)
					continue;
				awakeEntryCount++;
				awakeParticleCount += getParticleCount(entry);
			}
			// Prediction is streaming work, but its child/fan-in cost is paid
			// before any nonlinear work can start. Keep the initial threshold
			// conservative and identical to P3 write-back until timing data can
			// tune it separately.
			if(awakeEntryCount < 2 || awakeParticleCount < 1024)
				return 0;
			return PxMin(dispatcherWorkers, awakeEntryCount);
		}

		void submitStandalonePredictionTasks(
			PxU32 taskCount, PxReal dt, const PxVec3& gravity,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			PX_ASSERT(continuation);
			PX_ASSERT(taskCount == getStandalonePredictionTaskCount(
				PxMax(taskCount, 2u)));
			PX_ASSERT(taskCount > 0 && taskCount <= mEntries.size());
			while(mPredictionTasks.size() < taskCount)
				mPredictionTasks.pushBack(PX_NEW(PredictionTask)(
					mContextId, *this));
			const PxU32 entriesPerTask =
				(mEntries.size() + taskCount - 1) / taskCount;
			taskGraphContext.recordPredictionTasksSubmitted(taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; taskIndex++)
			{
				const PxU32 entryBegin = taskIndex * entriesPerTask;
				const PxU32 entryEnd = PxMin(
					entryBegin + entriesPerTask, mEntries.size());
				PX_ASSERT(entryBegin < entryEnd);
				PredictionTask& task = *mPredictionTasks[taskIndex];
				task.configure(entryBegin, entryEnd, dt, gravity,
					taskGraphContext);
				task.setContinuation(continuation);
				task.removeReference();
			}
		}

		// The state machine publishes one causal layer at a time. P4.5.3a
		// deliberately submits one whole-layer child as the taskgraph reference;
		// P4.5.3b may partition the same stable packed interval, but only under
		// its explicit validation switch.
		bool ensureCausalLayerTaskPool(
			PxU32 requiredChildTasks, Scene& owner,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			// One finishing task can be running while it publishes the next
			// layer's finish task, hence two parent-task slots in addition to the
			// currently submitted children.  Children are capped by dispatcher
			// worker count at the parent partition policy.
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			if(mCausalLayerTasks.size() >= requiredSlots &&
				mCausalLayerFinishTasks.size() >= requiredSlots &&
				mCausalLayerRangeObservations.capacity() >=
					requiredChildTasks)
				return true;
			const PxU32 oldTaskPointerCapacity =
				mCausalLayerTasks.capacity();
			const PxU32 oldFinishPointerCapacity =
				mCausalLayerFinishTasks.capacity();
			const PxU32 oldFreeTaskIndexCapacity =
				mFreeCausalLayerTaskIndices.capacity();
			const PxU32 oldFreeFinishIndexCapacity =
				mFreeCausalLayerFinishTaskIndices.capacity();
			const PxU32 oldObservationCapacity =
				mCausalLayerRangeObservations.capacity();
			const PxU32 oldTaskCount = mCausalLayerTasks.size();
			const PxU32 oldFinishTaskCount = mCausalLayerFinishTasks.size();
			// This function is called only by a parent continuation before it
			// submits a layer. Reserve free-list capacity first so dispatcher
			// release never allocates while recycling a task.
			mCausalLayerTasks.reserve(requiredSlots);
			mCausalLayerFinishTasks.reserve(requiredSlots);
			mFreeCausalLayerTaskIndices.reserve(requiredSlots);
			mFreeCausalLayerFinishTaskIndices.reserve(requiredSlots);
			// The parent clears/resizes one observation per child before making
			// any task runnable. Reserve here so neither a child release nor a
			// later same-size layer causes an observation-array allocation.
			mCausalLayerRangeObservations.reserve(requiredChildTasks);
			while(mCausalLayerTasks.size() < requiredSlots)
			{
				const PxU32 index = mCausalLayerTasks.size();
				mCausalLayerTasks.pushBack(PX_NEW(CausalLayerTask)(
					mContextId, *this, index));
				mFreeCausalLayerTaskIndices.pushBack(index);
			}
			while(mCausalLayerFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mCausalLayerFinishTasks.size();
				mCausalLayerFinishTasks.pushBack(PX_NEW(CausalLayerFinishTask)(
					mContextId, *this, owner, index));
				mFreeCausalLayerFinishTaskIndices.pushBack(index);
			}
			// This is intentionally an explicit payload accounting, not a claim
			// about allocator headers. It makes warmup-vs-steady task-pool growth
			// observable while preserving the child allocation prohibition.
			const PxU64 payloadGrowthBytes =
				PxU64(mCausalLayerTasks.capacity() - oldTaskPointerCapacity) *
					sizeof(CausalLayerTask*) +
				PxU64(mCausalLayerFinishTasks.capacity() -
					oldFinishPointerCapacity) * sizeof(CausalLayerFinishTask*) +
				PxU64(mFreeCausalLayerTaskIndices.capacity() -
					oldFreeTaskIndexCapacity) * sizeof(PxU32) +
				PxU64(mFreeCausalLayerFinishTaskIndices.capacity() -
					oldFreeFinishIndexCapacity) * sizeof(PxU32) +
				PxU64(mCausalLayerRangeObservations.capacity() -
					oldObservationCapacity) *
					sizeof(Dy::AvbdParticlePrimalRangeObservation) +
				PxU64(mCausalLayerTasks.size() - oldTaskCount) *
					sizeof(CausalLayerTask) +
				PxU64(mCausalLayerFinishTasks.size() - oldFinishTaskCount) *
					sizeof(CausalLayerFinishTask);
			if(payloadGrowthBytes)
				taskGraphContext.recordCausalLayerTaskPoolGrowth(
					payloadGrowthBytes);
			return true;
		}

		CausalLayerTask* acquireCausalLayerTask()
		{
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			if(mFreeCausalLayerTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeCausalLayerTaskIndices.back();
			mFreeCausalLayerTaskIndices.popBack();
			PX_ASSERT(index < mCausalLayerTasks.size());
			return mCausalLayerTasks[index];
		}

		CausalLayerFinishTask* acquireCausalLayerFinishTask()
		{
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			if(mFreeCausalLayerFinishTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeCausalLayerFinishTaskIndices.back();
			mFreeCausalLayerFinishTaskIndices.popBack();
			PX_ASSERT(index < mCausalLayerFinishTasks.size());
			return mCausalLayerFinishTasks[index];
		}

		void recycleCausalLayerTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			PX_ASSERT(index < mCausalLayerTasks.size());
			PX_ASSERT(mFreeCausalLayerTaskIndices.size() <
				mCausalLayerTasks.size());
			mFreeCausalLayerTaskIndices.pushBack(index);
		}

		void recycleCausalLayerFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			PX_ASSERT(index < mCausalLayerFinishTasks.size());
			PX_ASSERT(mFreeCausalLayerFinishTaskIndices.size() <
				mCausalLayerFinishTasks.size());
			mFreeCausalLayerFinishTaskIndices.pushBack(index);
		}

		bool hasCausalLayerTaskSlots(PxU32 taskCount)
		{
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			return mFreeCausalLayerTaskIndices.size() >= taskCount &&
				!mFreeCausalLayerFinishTaskIndices.empty();
		}

		PxU32 getCausalLayerTaskCount(
			PxU32 dispatcherWorkers, PxU32 layerOccupancy) const
		{
			// The ordered P4 path stays a one-range reference unless its explicit
			// validation switch is selected.  Relaxed colors are the production
			// throughput path: split a conflict-free color once it amortizes the
			// dispatch/fan-in boundary, while preserving the same owner proof.
			static const PxU32 eMIN_PARTICLES_PER_CAUSAL_LAYER_TASK = 16;
			const bool forceSmallLayerPartition =
				Dy::avbdForceCausalLayerTaskPartition();
			const bool relaxedColorFastPath =
				mStandaloneParticlePrimalSchedule ==
					Dy::AvbdParticlePrimalSchedule::eRELAXED_COLOR;
			if((!Dy::avbdUseCausalLayerTaskPartition() &&
				!relaxedColorFastPath) ||
				dispatcherWorkers < 2 ||
				(!forceSmallLayerPartition && layerOccupancy <
					eMIN_PARTICLES_PER_CAUSAL_LAYER_TASK))
				return 1;
			const PxU32 maxTasksByOccupancy =
				forceSmallLayerPartition ? layerOccupancy :
					(layerOccupancy +
						eMIN_PARTICLES_PER_CAUSAL_LAYER_TASK - 1) /
						eMIN_PARTICLES_PER_CAUSAL_LAYER_TASK;
			return PxMin(PxMin(dispatcherWorkers, maxTasksByOccupancy),
				layerOccupancy);
		}

		static PX_FORCE_INLINE PxU64 getIndependentBodySweepWorkEstimate(
			const Dy::AvbdSoftBody& body)
		{
			const Dy::AvbdSoftBodyCompiledData& compiled = body.compiled;
			const PxU64 work = PxU64(compiled.particleCount) +
				PxU64(compiled.triElements.size()) * 3u +
				PxU64(compiled.tetElements.size()) * 4u +
				PxU64(compiled.bendElements.size()) * 4u;
			return work ? work : 1u;
		}

		static PX_FORCE_INLINE PxU64 getIndependentBodySweepTarget(
			PxU64 totalWork, PxU32 boundaryIndex, PxU32 taskCount)
		{
			// Compute floor(totalWork * boundaryIndex / taskCount) without
			// overflowing the totalWork product.
			const PxU64 quotient = totalWork / taskCount;
			const PxU64 remainder = totalWork % taskCount;
			return quotient * boundaryIndex +
				(remainder * boundaryIndex) / taskCount;
		}

		static PX_FORCE_INLINE PxU64 getIndependentBodySweepDistance(
			PxU64 prefixWork, PxU64 targetWork)
		{
			return prefixWork > targetWork ? prefixWork - targetWork :
				targetWork - prefixWork;
		}

		bool submitStandaloneCausalLayerTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			if(!mStandaloneComponentSolvePrepared || !continuation)
				return false;
			const Dy::AvbdParticlePrimalSolveContext* bodySolveContext = NULL;
			const Dy::AvbdSoftBody* bodyRangeBodies = NULL;
			PxU32 bodyRangeCount = 0;
			const bool independentBodySweep =
				mStandaloneComponentStepState.
					getPublishedIndependentBodySweep(
						bodySolveContext, bodyRangeBodies, bodyRangeCount);
			PxU32 layerIndex = 0;
			PxU32 packedBegin = 0;
			PxU32 packedEnd = 0;
			const Dy::AvbdParticlePrimalSolveContext* solveContext = NULL;
			const Dy::AvbdSoftBody* bodies = NULL;
			PxU32 bodyCount = 0;
			const PxU32* particleBodyIndices = NULL;
			const PxU32* packedParticleIndices = NULL;
			if(independentBodySweep)
			{
				solveContext = bodySolveContext;
				bodies = bodyRangeBodies;
				bodyCount = bodyRangeCount;
				if(!solveContext || !bodies || bodyCount < 2)
					return false;
			}
			else if(!mStandaloneComponentStepState.getPublishedCausalLayer(
					layerIndex, packedBegin, packedEnd, solveContext, bodies,
					bodyCount, particleBodyIndices, packedParticleIndices) ||
					packedBegin >= packedEnd || !solveContext || !bodies ||
					!particleBodyIndices || !packedParticleIndices)
				return false;
			PX_UNUSED(layerIndex);
			PxU32 layerOccupancy = independentBodySweep ? 0u :
				packedEnd - packedBegin;
			if(independentBodySweep)
			{
				for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
					layerOccupancy +=
						bodies[bodyIndex].compiled.particleCount;
			}
			PxU64 independentBodyTotalWork = 0;
			PxU64 independentBodyMaximumWork = 0;
			if(independentBodySweep)
			{
				for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
				{
					const PxU64 bodyWork =
						getIndependentBodySweepWorkEstimate(bodies[bodyIndex]);
					independentBodyTotalWork += bodyWork;
					independentBodyMaximumWork =
						PxMax(independentBodyMaximumWork, bodyWork);
				}
			}
			PxU32 independentBodyTaskCount = 0;
			if(independentBodySweep)
			{
				PX_ASSERT(independentBodyMaximumWork > 0);
				const PxU64 wholeTasks =
					independentBodyTotalWork / independentBodyMaximumWork;
				const PxU64 remainder =
					independentBodyTotalWork % independentBodyMaximumWork;
				// A body is indivisible. Do not add a child for a tiny residual
				// range unless it represents at least half of the largest body's
				// primal work; that child cannot lower the dominant makespan enough
				// to repay another dispatch and fan-in transaction.
				const PxU32 usefulTaskCount = PxU32(
					wholeTasks +
					(remainder >= independentBodyMaximumWork -
						independentBodyMaximumWork / 2 ? 1u : 0u));
				independentBodyTaskCount = PxMin(
					dispatcherWorkers,
					PxMin(bodyCount, PxMax(2u, usefulTaskCount)));
			}
			const PxU32 taskCount = independentBodySweep ?
				independentBodyTaskCount :
				getCausalLayerTaskCount(dispatcherWorkers, layerOccupancy);
			if(taskCount == 0)
				return false;
			if(!ensureCausalLayerTaskPool(taskCount, owner, taskGraphContext) ||
				!hasCausalLayerTaskSlots(taskCount))
				return false;
			// Every range is a stable contiguous subinterval of the published
			// packed layer. Layer construction has already rejected every
			// structural/dynamic same-layer read/write conflict, so task ownership
			// is the range alone. The parent reduction remains ascending range
			// order regardless of dispatcher completion order.
			mCausalLayerRangeObservations.resize(taskCount);
			PX_ASSERT(mCausalLayerRangeObservations.capacity() >= taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; taskIndex++)
				mCausalLayerRangeObservations[taskIndex] =
					Dy::AvbdParticlePrimalRangeObservation();
			CausalLayerFinishTask* const finishTask =
				acquireCausalLayerFinishTask();
			if(!finishTask)
			{
				return false;
			}
			finishTask->setContinuation(continuation);
			taskGraphContext.recordCausalLayerTasksSubmitted(
				taskCount, layerOccupancy);
			mStandaloneTaskGraphTelemetry.recordCausalLayerTasksSubmitted(
				taskCount, layerOccupancy);
			const PxU32 particlesPerTask = independentBodySweep ? 0u :
				(layerOccupancy + taskCount - 1) / taskCount;
			PxU32 nextTaskBodyBegin = 0;
			PxU64 nextTaskBodyPrefixWork = 0;
			for(PxU32 taskIndex = 0; taskIndex < taskCount; taskIndex++)
			{
				PxU32 taskBodyBegin = 0;
				PxU32 taskBodyEnd = 0;
				if(independentBodySweep)
				{
					taskBodyBegin = nextTaskBodyBegin;
					if(taskIndex + 1 == taskCount)
					{
						taskBodyEnd = bodyCount;
						nextTaskBodyPrefixWork = independentBodyTotalWork;
					}
					else
					{
						// Keep every task non-empty and choose the closest cumulative
						// work boundary. Ranges remain contiguous whole-body intervals,
						// so dispatcher completion order cannot change solve ownership.
						const PxU32 remainingTasks = taskCount - taskIndex - 1;
						const PxU32 maximumBodyEnd = bodyCount - remainingTasks;
						const PxU64 targetWork = getIndependentBodySweepTarget(
							independentBodyTotalWork, taskIndex + 1, taskCount);
						taskBodyEnd = taskBodyBegin + 1;
						nextTaskBodyPrefixWork +=
							getIndependentBodySweepWorkEstimate(
								bodies[taskBodyBegin]);
						while(taskBodyEnd < maximumBodyEnd)
						{
							const PxU64 extendedPrefixWork =
								nextTaskBodyPrefixWork +
								getIndependentBodySweepWorkEstimate(
									bodies[taskBodyEnd]);
							if(getIndependentBodySweepDistance(
									extendedPrefixWork, targetWork) >
								getIndependentBodySweepDistance(
									nextTaskBodyPrefixWork, targetWork))
								break;
							nextTaskBodyPrefixWork = extendedPrefixWork;
							taskBodyEnd++;
						}
					}
					nextTaskBodyBegin = taskBodyEnd;
				}
				const PxU32 taskPackedBegin =
					independentBodySweep ? 0u :
						packedBegin + taskIndex * particlesPerTask;
				const PxU32 taskPackedEnd = independentBodySweep ? 0u :
					PxMin(taskPackedBegin + particlesPerTask, packedEnd);
				PX_ASSERT(independentBodySweep ?
					taskBodyBegin < taskBodyEnd :
					taskPackedBegin < taskPackedEnd);
				CausalLayerTask* const task = acquireCausalLayerTask();
				PX_ASSERT(task);
				if(!task)
				{
					// Slot availability was checked before the finish task was
					// acquired. This is defensive only; no child has been
					// submitted before this point in a valid Scene lifecycle.
					recycleCausalLayerFinishTask(finishTask->getPoolIndex());
					return false;
				}
				if(independentBodySweep)
					task->configureIndependentBodyRange(
						*solveContext, bodies, bodyCount,
						taskBodyBegin, taskBodyEnd,
						mCausalLayerRangeObservations[taskIndex],
						taskGraphContext);
				else
					task->configure(
						*solveContext, bodies, bodyCount, particleBodyIndices,
						mParticles.size(), packedParticleIndices,
						taskPackedBegin, taskPackedEnd,
						mCausalLayerRangeObservations[taskIndex],
						taskGraphContext);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			PX_ASSERT(!independentBodySweep ||
				(nextTaskBodyBegin == bodyCount &&
				 nextTaskBodyPrefixWork == independentBodyTotalWork));
			finishTask->removeReference();
			return true;
		}

		bool canUseWorldPlaneContactTaskTransaction() const
		{
			if(!Dy::avbdUseSceneRedetectionBridge() ||
				!Dy::avbdUseWorldPlaneContactTaskFanIn() ||
				mBodies.size() != 1 || mWorldPlanes.empty() ||
				!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
				!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				mSelfCollisionEnabled.size() != mBodies.size())
				return false;
			// The world-plane leaf has no private self-contact stream yet. Do not
			// bypass the serial all-OGC transaction until that source has its own
			// immutable plan and ordered merge contract.
			for(PxU32 bodyIndex = 0;
				bodyIndex < mSelfCollisionEnabled.size(); ++bodyIndex)
			{
				if(mSelfCollisionEnabled[bodyIndex])
					return false;
			}
			return true;
		}

		PxU32 getWorldPlaneContactTaskCount(
			PxU32 dispatcherWorkers) const
		{
			static const PxU32 eMIN_PARTICLES_PER_WORLD_PLANE_TASK = 128;
			const bool forceSmallTaskFanIn =
				Dy::avbdForceWorldPlaneContactTaskFanIn();
			if(!canUseWorldPlaneContactTaskTransaction() ||
				dispatcherWorkers < 2 || mParticles.size() <
					eMIN_PARTICLES_PER_WORLD_PLANE_TASK &&
				!forceSmallTaskFanIn)
				return 0;
			const PxU32 maxTasksByParticles = forceSmallTaskFanIn
				? mParticles.size()
				: (mParticles.size() +
					eMIN_PARTICLES_PER_WORLD_PLANE_TASK - 1) /
					eMIN_PARTICLES_PER_WORLD_PLANE_TASK;
			return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
				mParticles.size());
		}

		bool beginWorldPlaneContactTaskTransaction()
		{
			if(!canUseWorldPlaneContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending)
				return false;
			Dy::avbdBeginSoftContactRedetection(
				mContacts, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, mWorldPlanes.size(), false,
				0, 0, 0, 0, 0, mBodies.size(),
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(),
				mSelfCollisionEnabled.begin());
			mWorldPlaneContactTransactionPending = true;
			return true;
		}

		void completeWorldPlaneContactTaskTransaction()
		{
			PX_ASSERT(mWorldPlaneContactTransactionPending);
			for(PxU32 taskIndex = 0;
				taskIndex < mWorldPlaneContactTaskOutputs.size(); ++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mWorldPlaneContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0;
					contactIndex < source.size(); ++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedGroundContacts += mContacts.size();
			Dy::avbdCompleteSoftContactRedetection(
				mParticles.begin(), mContacts, mWorkspace.contact);
			removeRigidActorFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			removeDeformablePairFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			writeContactSetTrace(mContacts);
			mWorldPlaneContactTransactionPending = false;
		}

		bool ensureWorldPlaneContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner)
		{
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
			mWorldPlaneContactTasks.reserve(requiredSlots);
			mWorldPlaneContactFinishTasks.reserve(requiredSlots);
			mFreeWorldPlaneContactTaskIndices.reserve(requiredSlots);
			mFreeWorldPlaneContactFinishTaskIndices.reserve(requiredSlots);
			mWorldPlaneContactTaskOutputs.reserve(requiredChildTasks);
			while(mWorldPlaneContactTasks.size() < requiredSlots)
			{
				const PxU32 index = mWorldPlaneContactTasks.size();
				mWorldPlaneContactTasks.pushBack(PX_NEW(WorldPlaneContactTask)(
					mContextId, *this, index));
				mFreeWorldPlaneContactTaskIndices.pushBack(index);
			}
			while(mWorldPlaneContactFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mWorldPlaneContactFinishTasks.size();
				mWorldPlaneContactFinishTasks.pushBack(
					PX_NEW(WorldPlaneContactFinishTask)(
						mContextId, *this, owner, index));
				mFreeWorldPlaneContactFinishTaskIndices.pushBack(index);
			}
			return true;
		}

		WorldPlaneContactTask* acquireWorldPlaneContactTask()
		{
			PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
			if(mFreeWorldPlaneContactTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeWorldPlaneContactTaskIndices.back();
			mFreeWorldPlaneContactTaskIndices.popBack();
			return mWorldPlaneContactTasks[index];
		}

		WorldPlaneContactFinishTask* acquireWorldPlaneContactFinishTask()
		{
			PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
			if(mFreeWorldPlaneContactFinishTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeWorldPlaneContactFinishTaskIndices.back();
			mFreeWorldPlaneContactFinishTaskIndices.popBack();
			return mWorldPlaneContactFinishTasks[index];
		}

		void recycleWorldPlaneContactTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
			PX_ASSERT(index < mWorldPlaneContactTasks.size());
			mFreeWorldPlaneContactTaskIndices.pushBack(index);
		}

		void recycleWorldPlaneContactFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
			PX_ASSERT(index < mWorldPlaneContactFinishTasks.size());
			mFreeWorldPlaneContactFinishTaskIndices.pushBack(index);
		}

		bool submitStandaloneWorldPlaneContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			const PxU32 taskCount = getWorldPlaneContactTaskCount(
				dispatcherWorkers);
			if(!mWorldPlaneContactTransactionPending || !continuation ||
				taskCount == 0 ||
				!ensureWorldPlaneContactTaskPool(taskCount, owner))
				return false;
			{
				PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
				if(mFreeWorldPlaneContactTaskIndices.size() < taskCount ||
					mFreeWorldPlaneContactFinishTaskIndices.empty())
					return false;
			}
			const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
				mWorldPlanes.size();
			if(maxContactCount64 > PX_MAX_U32)
				return false;
			const PxU32 maxContactCount = PxU32(maxContactCount64);
			mContacts.reserve(maxContactCount);
			mWorldPlaneContactTaskOutputs.resize(taskCount);
			const PxU32 particlesPerTask =
				(mParticles.size() + taskCount - 1) / taskCount;
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				PxArray<Dy::AvbdSoftContact>& output =
					mWorldPlaneContactTaskOutputs[taskIndex];
				output.clear();
				output.reserve((particleEnd - particleBegin) *
					mWorldPlanes.size());
			}
			WorldPlaneContactFinishTask* const finishTask =
				acquireWorldPlaneContactFinishTask();
			if(!finishTask)
				return false;
			finishTask->setContinuation(continuation);
			taskGraphContext.recordWorldPlaneContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordWorldPlaneContactTasksSubmitted(taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				WorldPlaneContactTask* const task =
					acquireWorldPlaneContactTask();
				PX_ASSERT(task && particleBegin < particleEnd);
				if(!task)
				{
					recycleWorldPlaneContactFinishTask(
						finishTask->getPoolIndex());
					return false;
				}
				task->configure(
					mParticles.begin(), mParticles.size(),
					particleBegin, particleEnd,
					mWorldPlanes.begin(), mWorldPlanes.size(),
					mBodies.begin(), mBodies.size(),
					mWorldPlaneContactTaskOutputs[taskIndex],
					mContactParams.contactRadius, taskGraphContext);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			finishTask->removeReference();
			return true;
		}

		Dy::AvbdSoftBodyStepAdvanceResult
			advanceStandaloneComponentStateWithSceneRedetection(
				bool allowWorldPlaneTask = false,
				bool* worldPlaneContactTaskReady = NULL,
				bool allowRigidBoxSdfTask = false,
				bool* rigidBoxSdfContactTaskReady = NULL,
				bool allowRigidSphereSdfTask = false,
				bool* rigidSphereSdfContactTaskReady = NULL)
		{
			if(worldPlaneContactTaskReady)
				*worldPlaneContactTaskReady = false;
			if(rigidBoxSdfContactTaskReady)
				*rigidBoxSdfContactTaskReady = false;
			if(rigidSphereSdfContactTaskReady)
				*rigidSphereSdfContactTaskReady = false;
			for(;;)
			{
				const Dy::AvbdSoftBodyStepAdvanceResult result =
					mStandaloneComponentStepState.advance();
				if(result !=
					Dy::AvbdSoftBodyStepAdvanceResult::eREDETECTION_READY)
					return result;
				// The aggregate must claim this epoch before the source-specific
				// leaves. It owns one Begin/Complete pair and reconstructs their
				// serial source order after all private children have joined.
				if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
					beginStaticWorldSelfOgcContactTaskTransaction())
				{
					*rigidSphereSdfContactTaskReady = true;
					return result;
				}
				if(allowWorldPlaneTask && worldPlaneContactTaskReady &&
					beginWorldPlaneContactTaskTransaction())
				{
					*worldPlaneContactTaskReady = true;
					return result;
				}
				if(allowRigidBoxSdfTask && rigidBoxSdfContactTaskReady &&
					beginRigidBoxSdfContactTaskTransaction())
				{
					*rigidBoxSdfContactTaskReady = true;
					return result;
				}
				if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
					beginRigidSphereSdfContactTaskTransaction())
				{
					*rigidSphereSdfContactTaskReady = true;
					return result;
				}
				// Static sphere and capsule leaves are intentionally mutually
				// exclusive. Reuse the existing smooth-SDF continuation readiness
				// bit, while keeping the transaction/pool/telemetry below separate.
				if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
					beginRigidCapsuleSdfContactTaskTransaction())
				{
					*rigidSphereSdfContactTaskReady = true;
					return result;
				}
				if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
					beginRigidConvexSdfContactTaskTransaction())
				{
					*rigidSphereSdfContactTaskReady = true;
					return result;
				}
				if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
					beginRigidTriangleSurfaceContactTaskTransaction())
				{
					*rigidSphereSdfContactTaskReady = true;
					return result;
				}
				if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
					beginSoftPairContactTaskTransaction())
				{
					*rigidSphereSdfContactTaskReady = true;
					return result;
				}
				if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
					beginSelfBvhContactTaskTransaction())
				{
					*rigidSphereSdfContactTaskReady = true;
					return result;
				}
				// The state has published its only redetection boundary.  The
				// Scene parent owns callback execution, mutable contact storage,
				// filtering, state transfer, trace and the final post-detection
				// index rebuild; a future P5 child graph may replace only this
				// synchronous body with a candidate fan-in.
				redetectContacts(
					mParticles.begin(), mParticles.size(),
					mBodies.begin(), mBodies.size(), mContacts, this);
				if(!mStandaloneComponentStepState.
					completePendingRedetection())
					return Dy::AvbdSoftBodyStepAdvanceResult::eINVALID;
			}
		}

		bool runStandaloneComponentStateWithSceneRedetection()
		{
			for(;;)
			{
				const Dy::AvbdSoftBodyStepAdvanceResult result =
					advanceStandaloneComponentStateWithSceneRedetection();
				if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
					return true;
				if(result !=
					Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
					return false;
				PxU32 layerIndex = 0;
				PxU32 packedBegin = 0;
				PxU32 packedEnd = 0;
				const Dy::AvbdParticlePrimalSolveContext* solveContext = NULL;
				const Dy::AvbdSoftBody* bodies = NULL;
				PxU32 bodyCount = 0;
				const PxU32* particleBodyIndices = NULL;
				const PxU32* packedParticleIndices = NULL;
				if(!mStandaloneComponentStepState.getPublishedCausalLayer(
					layerIndex, packedBegin, packedEnd, solveContext, bodies,
					bodyCount, particleBodyIndices, packedParticleIndices))
					return false;
				PX_UNUSED(layerIndex);
				Dy::AvbdParticlePrimalRangeObservation observation;
				Dy::avbdSolveParticlePrimalPackedRange(
					*solveContext, bodies, bodyCount, particleBodyIndices,
					mParticles.size(), packedParticleIndices,
					packedBegin, packedEnd, observation);
				if(!mStandaloneComponentStepState.
					completePublishedCausalLayer(&observation, 1))
					return false;
			}
		}

		bool completeStandaloneWorldPlaneContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady,
			bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mWorldPlaneContactTransactionPending)
				return false;
			completeWorldPlaneContactTaskTransaction();
			taskGraphContext.recordWorldPlaneContactFanIn();
			if(!mStandaloneComponentStepState.completePendingRedetection())
				return false;
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				advanceStandaloneComponentStateWithSceneRedetection(
					true, &nextWorldPlaneContactTaskReady,
					true, &nextRigidBoxSdfContactTaskReady,
					true, &nextRigidSphereSdfContactTaskReady);
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
				return true;
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		bool finishStandaloneWorldPlaneContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady,
			bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(!mWorldPlaneContactTransactionPending)
				return false;
			taskGraphContext.recordSerialWorldPlaneContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialWorldPlaneContactFallback();
			mWorldPlaneContactTaskOutputs.clear();
			Dy::avbdDetectSoftWorldPlaneContacts(
				mParticles.begin(), mParticles.size(),
				mWorldPlanes.begin(), mWorldPlanes.size(), mContacts,
				mContactParams.contactRadius, mBodies.begin(), mBodies.size());
			return completeStandaloneWorldPlaneContactTask(
				 dt, taskGraphContext, nextLayerReady,
				 nextWorldPlaneContactTaskReady,
				 nextRigidBoxSdfContactTaskReady,
				 nextRigidSphereSdfContactTaskReady);
		}

		bool canUseRigidBoxSdfContactTaskTransaction() const
		{
			if(!Dy::avbdUseSceneRedetectionBridge() ||
				!Dy::avbdUseRigidBoxSdfContactTaskFanIn() ||
				mBodies.size() != 1 || mRigidBoxes.empty() ||
				!mWorldPlanes.empty() || !mRigidSpheres.empty() ||
				!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				mSelfCollisionEnabled.size() != mBodies.size())
				return false;
			// Dynamic/kinematic boxes enter native coupling or one-way target
			// paths. This first leaf is deliberately world-static only.
			for(PxU32 boxIndex = 0; boxIndex < mRigidBoxes.size(); ++boxIndex)
			{
				if(mRigidBoxes[boxIndex].targetKind !=
					Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
					return false;
			}
			for(PxU32 bodyIndex = 0;
				bodyIndex < mSelfCollisionEnabled.size(); ++bodyIndex)
			{
				if(mSelfCollisionEnabled[bodyIndex])
					return false;
			}
			return true;
		}

		PxU32 getRigidBoxSdfContactTaskCount(
			PxU32 dispatcherWorkers) const
		{
			static const PxU32 eMIN_PARTICLES_PER_RIGID_BOX_SDF_TASK = 128;
			const bool forceSmallTaskFanIn =
				Dy::avbdForceRigidBoxSdfContactTaskFanIn();
			if(!canUseRigidBoxSdfContactTaskTransaction() ||
				dispatcherWorkers < 2 || mParticles.size() <
					eMIN_PARTICLES_PER_RIGID_BOX_SDF_TASK &&
				!forceSmallTaskFanIn)
				return 0;
			const PxU32 maxTasksByParticles = forceSmallTaskFanIn
				? mParticles.size()
				: (mParticles.size() +
					eMIN_PARTICLES_PER_RIGID_BOX_SDF_TASK - 1) /
					eMIN_PARTICLES_PER_RIGID_BOX_SDF_TASK;
			return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
				mParticles.size());
		}

		bool beginRigidBoxSdfContactTaskTransaction()
		{
			if(!canUseRigidBoxSdfContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending ||
				mRigidBoxSdfContactTransactionPending)
				return false;
			Dy::avbdBeginSoftContactRedetection(
				mContacts, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, 0, false,
				mRigidBoxes.size(), 0, 0, 0, 0, mBodies.size(),
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(),
				mSelfCollisionEnabled.begin());
			mRigidBoxSdfContactTransactionPending = true;
			return true;
		}

		void completeRigidBoxSdfContactTaskTransaction()
		{
			PX_ASSERT(mRigidBoxSdfContactTransactionPending);
			const PxU32 rigidStart = mContacts.size();
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidBoxSdfContactTaskOutputs.size(); ++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidBoxSdfContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0;
					contactIndex < source.size(); ++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			// P5.12b's only contact-family merge point: all current-SDF ranges
			// precede all swept-SDF ranges. Never interleave ranges across these
			// two loops, even though each child computed both private outputs.
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidBoxSweptSdfContactTaskOutputs.size(); ++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidBoxSweptSdfContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0;
					contactIndex < source.size(); ++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			if(mCollisionStatsEnabled)
				mLastCollisionStats.rigidParticleBoxTests +=
					PxU64(mParticles.size()) * mRigidBoxes.size();
			// Feature OGC remains parent-owned after both private SDF families have
			// been merged in their legacy order.
			Dy::avbdDetectSoftRigidOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidBoxes.begin(), mRigidBoxes.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius);
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedRigidContacts +=
					mContacts.size() - rigidStart;
			Dy::avbdCompleteSoftContactRedetection(
				mParticles.begin(), mContacts, mWorkspace.contact);
			removeRigidActorFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			removeDeformablePairFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			writeContactSetTrace(mContacts);
			mRigidBoxSdfContactTransactionPending = false;
		}

		bool ensureRigidBoxSdfContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner)
		{
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
			mRigidBoxSdfContactTasks.reserve(requiredSlots);
			mRigidBoxSdfContactFinishTasks.reserve(requiredSlots);
			mFreeRigidBoxSdfContactTaskIndices.reserve(requiredSlots);
			mFreeRigidBoxSdfContactFinishTaskIndices.reserve(requiredSlots);
			mRigidBoxSdfContactTaskOutputs.reserve(requiredChildTasks);
			mRigidBoxSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
			while(mRigidBoxSdfContactTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidBoxSdfContactTasks.size();
				mRigidBoxSdfContactTasks.pushBack(PX_NEW(RigidBoxSdfContactTask)(
					mContextId, *this, index));
				mFreeRigidBoxSdfContactTaskIndices.pushBack(index);
			}
			while(mRigidBoxSdfContactFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidBoxSdfContactFinishTasks.size();
				mRigidBoxSdfContactFinishTasks.pushBack(
					PX_NEW(RigidBoxSdfContactFinishTask)(
						mContextId, *this, owner, index));
				mFreeRigidBoxSdfContactFinishTaskIndices.pushBack(index);
			}
			return true;
		}

		RigidBoxSdfContactTask* acquireRigidBoxSdfContactTask()
		{
			PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
			if(mFreeRigidBoxSdfContactTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeRigidBoxSdfContactTaskIndices.back();
			mFreeRigidBoxSdfContactTaskIndices.popBack();
			return mRigidBoxSdfContactTasks[index];
		}

		RigidBoxSdfContactFinishTask* acquireRigidBoxSdfContactFinishTask()
		{
			PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
			if(mFreeRigidBoxSdfContactFinishTaskIndices.empty())
				return NULL;
			const PxU32 index =
				mFreeRigidBoxSdfContactFinishTaskIndices.back();
			mFreeRigidBoxSdfContactFinishTaskIndices.popBack();
			return mRigidBoxSdfContactFinishTasks[index];
		}

		void recycleRigidBoxSdfContactTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
			PX_ASSERT(index < mRigidBoxSdfContactTasks.size());
			mFreeRigidBoxSdfContactTaskIndices.pushBack(index);
		}

		void recycleRigidBoxSdfContactFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
			PX_ASSERT(index < mRigidBoxSdfContactFinishTasks.size());
			mFreeRigidBoxSdfContactFinishTaskIndices.pushBack(index);
		}

		bool submitStandaloneRigidBoxSdfContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			const PxU32 taskCount = getRigidBoxSdfContactTaskCount(
				dispatcherWorkers);
			if(!mRigidBoxSdfContactTransactionPending || !continuation ||
				taskCount == 0 ||
				!ensureRigidBoxSdfContactTaskPool(taskCount, owner))
				return false;
			{
				PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
				if(mFreeRigidBoxSdfContactTaskIndices.size() < taskCount ||
					mFreeRigidBoxSdfContactFinishTaskIndices.empty())
					return false;
			}
			const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
				mRigidBoxes.size();
			if(maxContactCount64 > PX_MAX_U32)
				return false;
			mContacts.reserve(PxU32(maxContactCount64));
			mRigidBoxSdfContactTaskOutputs.resize(taskCount);
			mRigidBoxSweptSdfContactTaskOutputs.resize(taskCount);
			const PxU32 particlesPerTask =
				(mParticles.size() + taskCount - 1) / taskCount;
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				PxArray<Dy::AvbdSoftContact>& output =
					mRigidBoxSdfContactTaskOutputs[taskIndex];
				output.clear();
				output.reserve((particleEnd - particleBegin) *
					mRigidBoxes.size());
				PxArray<Dy::AvbdSoftContact>& sweptOutput =
					mRigidBoxSweptSdfContactTaskOutputs[taskIndex];
				sweptOutput.clear();
				sweptOutput.reserve((particleEnd - particleBegin) *
					mRigidBoxes.size());
			}
			RigidBoxSdfContactFinishTask* const finishTask =
				acquireRigidBoxSdfContactFinishTask();
			if(!finishTask)
				return false;
			finishTask->setContinuation(continuation);
			taskGraphContext.recordRigidBoxSdfContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordRigidBoxSdfContactTasksSubmitted(taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				RigidBoxSdfContactTask* const task =
					acquireRigidBoxSdfContactTask();
				PX_ASSERT(task && particleBegin < particleEnd);
				if(!task)
				{
					recycleRigidBoxSdfContactFinishTask(
						finishTask->getPoolIndex());
					return false;
				}
				task->configure(
					mParticles.begin(), mParticles.size(),
					particleBegin, particleEnd,
					mRigidBoxes.begin(), mRigidBoxes.size(),
					mWorkspace.contact.previousContacts.begin(),
					mWorkspace.contact.previousContacts.size(),
					mBodies.begin(), mBodies.size(),
					mRigidBoxSdfContactTaskOutputs[taskIndex],
					mRigidBoxSweptSdfContactTaskOutputs[taskIndex],
					mContactParams.contactRadius, taskGraphContext);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			finishTask->removeReference();
			return true;
		}

		bool completeStandaloneRigidBoxSdfContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mRigidBoxSdfContactTransactionPending)
				return false;
			completeRigidBoxSdfContactTaskTransaction();
			taskGraphContext.recordRigidBoxSdfContactFanIn();
			if(!mStandaloneComponentStepState.completePendingRedetection())
				return false;
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				advanceStandaloneComponentStateWithSceneRedetection(
					true, &nextWorldPlaneContactTaskReady,
					true, &nextRigidBoxSdfContactTaskReady,
					true, &nextRigidSphereSdfContactTaskReady);
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
				return true;
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		bool finishStandaloneRigidBoxSdfContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(!mRigidBoxSdfContactTransactionPending)
				return false;
			taskGraphContext.recordSerialRigidBoxSdfContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialRigidBoxSdfContactFallback();
			mRigidBoxSdfContactTaskOutputs.clear();
			mRigidBoxSweptSdfContactTaskOutputs.clear();
			Dy::avbdDetectSoftRigidSDF(
				mParticles.begin(), mParticles.size(),
				mRigidBoxes.begin(), mRigidBoxes.size(), mContacts,
				mContactParams.contactRadius,
				mWorkspace.contact.previousContacts.begin(),
				mWorkspace.contact.previousContacts.size(),
				mBodies.begin(), mBodies.size());
			// Keep the serial recovery path byte-for-byte equivalent in contact
			// family order to the P5.12b task transaction: current SDF first,
			// swept SDF second, then the parent-owned feature suffix.
			Dy::avbdDetectSoftRigidSweptSDF(
				mParticles.begin(), mParticles.size(),
				mRigidBoxes.begin(), mRigidBoxes.size(), mContacts,
				mContactParams.contactRadius,
				mBodies.begin(), mBodies.size());
			return completeStandaloneRigidBoxSdfContactTask(
				 dt, taskGraphContext, nextLayerReady,
				 nextWorldPlaneContactTaskReady,
				 nextRigidBoxSdfContactTaskReady,
				 nextRigidSphereSdfContactTaskReady);
		}

		bool canUseRigidSphereSdfContactTaskTransaction() const
		{
			if(!Dy::avbdUseSceneRedetectionBridge() ||
				!Dy::avbdUseRigidSphereSdfContactTaskFanIn() ||
				mBodies.size() != 1 || mRigidSpheres.empty() ||
				!mWorldPlanes.empty() || !mRigidBoxes.empty() ||
				!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				mSelfCollisionEnabled.size() != mBodies.size())
				return false;
			// Dynamic/kinematic spheres have relative-motion contact ownership.
			// This first sphere leaf is strictly world-static.
			for(PxU32 sphereIndex = 0;
				sphereIndex < mRigidSpheres.size(); ++sphereIndex)
			{
				if(mRigidSpheres[sphereIndex].targetKind !=
					Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
					return false;
			}
			for(PxU32 bodyIndex = 0;
				bodyIndex < mSelfCollisionEnabled.size(); ++bodyIndex)
			{
				if(mSelfCollisionEnabled[bodyIndex])
					return false;
			}
			return true;
		}

		PxU32 getRigidSphereSdfContactTaskCount(
			PxU32 dispatcherWorkers) const
		{
			static const PxU32 eMIN_PARTICLES_PER_RIGID_SPHERE_SDF_TASK = 128;
			const bool forceSmallTaskFanIn =
				Dy::avbdForceRigidSphereSdfContactTaskFanIn();
			if(!canUseRigidSphereSdfContactTaskTransaction() ||
				dispatcherWorkers < 2 || mParticles.size() <
					eMIN_PARTICLES_PER_RIGID_SPHERE_SDF_TASK &&
				!forceSmallTaskFanIn)
				return 0;
			const PxU32 maxTasksByParticles = forceSmallTaskFanIn
				? mParticles.size()
				: (mParticles.size() +
					eMIN_PARTICLES_PER_RIGID_SPHERE_SDF_TASK - 1) /
					eMIN_PARTICLES_PER_RIGID_SPHERE_SDF_TASK;
			return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
				mParticles.size());
		}

		bool beginRigidSphereSdfContactTaskTransaction()
		{
			if(!canUseRigidSphereSdfContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending ||
				mRigidBoxSdfContactTransactionPending ||
				mRigidSphereSdfContactTransactionPending ||
				mRigidCapsuleSdfContactTransactionPending ||
				mRigidConvexSdfContactTransactionPending)
				return false;
			Dy::avbdBeginSoftContactRedetection(
				mContacts, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, 0, false,
				0, mRigidSpheres.size(), 0, 0, 0, mBodies.size(),
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(),
				mSelfCollisionEnabled.begin());
			mRigidSphereSdfContactTransactionPending = true;
			return true;
		}

		void completeRigidSphereSdfContactTaskTransaction()
		{
			PX_ASSERT(mRigidSphereSdfContactTransactionPending);
			const PxU32 rigidStart = mContacts.size();
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidSphereSdfContactTaskOutputs.size(); ++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidSphereSdfContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0;
					contactIndex < source.size(); ++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			// P5.13b's only SDF-family merge point: every current-SDF range
			// precedes every swept-SDF range. Do not interleave both streams per
			// child even though a child evaluates its two leaves back-to-back.
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidSphereSweptSdfContactTaskOutputs.size();
				++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidSphereSweptSdfContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0;
					contactIndex < source.size(); ++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			if(mCollisionStatsEnabled)
				mLastCollisionStats.rigidParticleSphereTests +=
					PxU64(mParticles.size()) * mRigidSpheres.size();
			// Both feature suffixes remain parent-owned after current/swept fan-in.
			Dy::avbdDetectSoftRigidSphereSweptOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidSpheres.begin(), mRigidSpheres.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius,
				&mWorkspace.contact.rigidConvexForwardOwnerScratch);
			Dy::avbdDetectSoftRigidSphereOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidSpheres.begin(), mRigidSpheres.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius);
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedRigidContacts +=
					mContacts.size() - rigidStart;
			Dy::avbdCompleteSoftContactRedetection(
				mParticles.begin(), mContacts, mWorkspace.contact);
			removeRigidActorFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			removeDeformablePairFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			writeContactSetTrace(mContacts);
			mRigidSphereSdfContactTransactionPending = false;
		}

		bool ensureRigidSphereSdfContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner)
		{
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
			mRigidSphereSdfContactTasks.reserve(requiredSlots);
			mRigidSphereSdfContactFinishTasks.reserve(requiredSlots);
			mFreeRigidSphereSdfContactTaskIndices.reserve(requiredSlots);
			mFreeRigidSphereSdfContactFinishTaskIndices.reserve(requiredSlots);
			mRigidSphereSdfContactTaskOutputs.reserve(requiredChildTasks);
			mRigidSphereSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
			while(mRigidSphereSdfContactTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidSphereSdfContactTasks.size();
				mRigidSphereSdfContactTasks.pushBack(
					PX_NEW(RigidSphereSdfContactTask)(mContextId, *this, index));
				mFreeRigidSphereSdfContactTaskIndices.pushBack(index);
			}
			while(mRigidSphereSdfContactFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidSphereSdfContactFinishTasks.size();
				mRigidSphereSdfContactFinishTasks.pushBack(
					PX_NEW(RigidSphereSdfContactFinishTask)(
						mContextId, *this, owner, index));
				mFreeRigidSphereSdfContactFinishTaskIndices.pushBack(index);
			}
			return true;
		}

		RigidSphereSdfContactTask* acquireRigidSphereSdfContactTask()
		{
			PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
			if(mFreeRigidSphereSdfContactTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeRigidSphereSdfContactTaskIndices.back();
			mFreeRigidSphereSdfContactTaskIndices.popBack();
			return mRigidSphereSdfContactTasks[index];
		}

		RigidSphereSdfContactFinishTask*
			acquireRigidSphereSdfContactFinishTask()
		{
			PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
			if(mFreeRigidSphereSdfContactFinishTaskIndices.empty())
				return NULL;
			const PxU32 index =
				mFreeRigidSphereSdfContactFinishTaskIndices.back();
			mFreeRigidSphereSdfContactFinishTaskIndices.popBack();
			return mRigidSphereSdfContactFinishTasks[index];
		}

		void recycleRigidSphereSdfContactTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
			PX_ASSERT(index < mRigidSphereSdfContactTasks.size());
			mFreeRigidSphereSdfContactTaskIndices.pushBack(index);
		}

		void recycleRigidSphereSdfContactFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
			PX_ASSERT(index < mRigidSphereSdfContactFinishTasks.size());
			mFreeRigidSphereSdfContactFinishTaskIndices.pushBack(index);
		}

		bool submitStandaloneRigidSphereSdfContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			if(mStaticWorldSelfOgcContactTransactionPending)
				return submitStandaloneStaticWorldSelfOgcContactTask(
					dispatcherWorkers, owner, continuation, taskGraphContext);
			if(mSoftPairContactTransactionPending)
				return submitStandaloneSoftPairContactTask(
					dispatcherWorkers, owner, continuation, taskGraphContext);
			if(mSelfBvhContactTransactionPending)
				return submitStandaloneSelfBvhContactTask(
					dispatcherWorkers, owner, continuation, taskGraphContext);
			if(mRigidCapsuleSdfContactTransactionPending)
				return submitStandaloneRigidCapsuleSdfContactTask(
					dispatcherWorkers, owner, continuation, taskGraphContext);
			if(mRigidConvexSdfContactTransactionPending)
				return submitStandaloneRigidConvexSdfContactTask(
					dispatcherWorkers, owner, continuation, taskGraphContext);
			if(mRigidTriangleSurfaceContactTransactionPending)
				return submitStandaloneRigidTriangleSurfaceContactTask(
					dispatcherWorkers, owner, continuation, taskGraphContext);
			const PxU32 taskCount = getRigidSphereSdfContactTaskCount(
				dispatcherWorkers);
			if(!mRigidSphereSdfContactTransactionPending || !continuation ||
				taskCount == 0 ||
				!ensureRigidSphereSdfContactTaskPool(taskCount, owner))
				return false;
			{
				PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
				if(mFreeRigidSphereSdfContactTaskIndices.size() < taskCount ||
					mFreeRigidSphereSdfContactFinishTaskIndices.empty())
					return false;
			}
			const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
				mRigidSpheres.size();
			if(maxContactCount64 > PX_MAX_U32)
				return false;
			mContacts.reserve(PxU32(maxContactCount64));
			mRigidSphereSdfContactTaskOutputs.resize(taskCount);
			mRigidSphereSweptSdfContactTaskOutputs.resize(taskCount);
			const PxU32 particlesPerTask =
				(mParticles.size() + taskCount - 1) / taskCount;
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				PxArray<Dy::AvbdSoftContact>& output =
					mRigidSphereSdfContactTaskOutputs[taskIndex];
				output.clear();
				output.reserve((particleEnd - particleBegin) *
					mRigidSpheres.size());
				PxArray<Dy::AvbdSoftContact>& sweptOutput =
					mRigidSphereSweptSdfContactTaskOutputs[taskIndex];
				sweptOutput.clear();
				sweptOutput.reserve((particleEnd - particleBegin) *
					mRigidSpheres.size());
			}
			RigidSphereSdfContactFinishTask* const finishTask =
				acquireRigidSphereSdfContactFinishTask();
			if(!finishTask)
				return false;
			finishTask->setContinuation(continuation);
			taskGraphContext.recordRigidSphereSdfContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordRigidSphereSdfContactTasksSubmitted(taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				RigidSphereSdfContactTask* const task =
					acquireRigidSphereSdfContactTask();
				PX_ASSERT(task && particleBegin < particleEnd);
				if(!task)
				{
					recycleRigidSphereSdfContactFinishTask(
						finishTask->getPoolIndex());
					return false;
				}
				task->configure(
					mParticles.begin(), mParticles.size(),
					particleBegin, particleEnd,
				mRigidSpheres.begin(), mRigidSpheres.size(),
				mBodies.begin(), mBodies.size(),
				mRigidSphereSdfContactTaskOutputs[taskIndex],
				mRigidSphereSweptSdfContactTaskOutputs[taskIndex],
				mContactParams.contactRadius, taskGraphContext);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			finishTask->removeReference();
			return true;
		}

		bool completeStandaloneRigidSphereSdfContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(mStaticWorldSelfOgcContactTransactionPending)
				return completeStandaloneStaticWorldSelfOgcContactTask(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mSoftPairContactTransactionPending)
				return completeStandaloneSoftPairContactTask(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mSelfBvhContactTransactionPending)
				return completeStandaloneSelfBvhContactTask(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mRigidCapsuleSdfContactTransactionPending)
				return completeStandaloneRigidCapsuleSdfContactTask(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mRigidConvexSdfContactTransactionPending)
				return completeStandaloneRigidConvexSdfContactTask(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mRigidTriangleSurfaceContactTransactionPending)
				return completeStandaloneRigidTriangleSurfaceContactTask(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mRigidSphereSdfContactTransactionPending)
				return false;
			completeRigidSphereSdfContactTaskTransaction();
			taskGraphContext.recordRigidSphereSdfContactFanIn();
			if(!mStandaloneComponentStepState.completePendingRedetection())
				return false;
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				advanceStandaloneComponentStateWithSceneRedetection(
					true, &nextWorldPlaneContactTaskReady,
					true, &nextRigidBoxSdfContactTaskReady,
					true, &nextRigidSphereSdfContactTaskReady);
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
				return true;
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		bool finishStandaloneRigidSphereSdfContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(mStaticWorldSelfOgcContactTransactionPending)
				return finishStandaloneStaticWorldSelfOgcContactSerialFallback(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mSoftPairContactTransactionPending)
				return finishStandaloneSoftPairContactSerialFallback(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mSelfBvhContactTransactionPending)
				return finishStandaloneSelfBvhContactSerialFallback(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mRigidCapsuleSdfContactTransactionPending)
				return finishStandaloneRigidCapsuleSdfContactSerialFallback(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mRigidConvexSdfContactTransactionPending)
				return finishStandaloneRigidConvexSdfContactSerialFallback(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(mRigidTriangleSurfaceContactTransactionPending)
				return finishStandaloneRigidTriangleSurfaceContactSerialFallback(
					dt, taskGraphContext, nextLayerReady,
					nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
			if(!mRigidSphereSdfContactTransactionPending)
				return false;
			taskGraphContext.recordSerialRigidSphereSdfContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialRigidSphereSdfContactFallback();
			mRigidSphereSdfContactTaskOutputs.clear();
			mRigidSphereSweptSdfContactTaskOutputs.clear();
			Dy::avbdDetectSoftRigidSphereSDF(
				mParticles.begin(), mParticles.size(),
				mRigidSpheres.begin(), mRigidSpheres.size(), mContacts,
				mContactParams.contactRadius,
				mBodies.begin(), mBodies.size());
			// Match P5.13b's parent fan-in family order before the two feature
			// suffixes in completeRigidSphereSdfContactTaskTransaction().
			Dy::avbdDetectSoftRigidSphereSweptSDF(
				mParticles.begin(), mParticles.size(),
				mRigidSpheres.begin(), mRigidSpheres.size(), mContacts,
				mContactParams.contactRadius,
				mBodies.begin(), mBodies.size());
			return completeStandaloneRigidSphereSdfContactTask(
				dt, taskGraphContext, nextLayerReady,
				nextWorldPlaneContactTaskReady,
				nextRigidBoxSdfContactTaskReady,
				nextRigidSphereSdfContactTaskReady);
		}

		bool canUseRigidCapsuleSdfContactTaskTransaction() const
		{
			if(!Dy::avbdUseSceneRedetectionBridge() ||
				!Dy::avbdUseRigidCapsuleSdfContactTaskFanIn() ||
				mBodies.size() != 1 || mRigidCapsules.empty() ||
				!mWorldPlanes.empty() || !mRigidBoxes.empty() ||
				!mRigidSpheres.empty() || !mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				mSelfCollisionEnabled.size() != mBodies.size())
				return false;
			// Dynamic/kinematic capsules carry relative-motion ownership. The
			// P5.6b leaf deliberately accepts world-static capsules only.
			for(PxU32 capsuleIndex = 0;
				capsuleIndex < mRigidCapsules.size(); ++capsuleIndex)
			{
				if(mRigidCapsules[capsuleIndex].targetKind !=
					Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
					return false;
			}
			for(PxU32 bodyIndex = 0;
				bodyIndex < mSelfCollisionEnabled.size(); ++bodyIndex)
			{
				if(mSelfCollisionEnabled[bodyIndex])
					return false;
			}
			return true;
		}

		PxU32 getRigidCapsuleSdfContactTaskCount(
			PxU32 dispatcherWorkers) const
		{
			static const PxU32 eMIN_PARTICLES_PER_RIGID_CAPSULE_SDF_TASK = 128;
			const bool forceSmallTaskFanIn =
				Dy::avbdForceRigidCapsuleSdfContactTaskFanIn();
			if(!canUseRigidCapsuleSdfContactTaskTransaction() ||
				dispatcherWorkers < 2 || mParticles.size() <
					eMIN_PARTICLES_PER_RIGID_CAPSULE_SDF_TASK &&
				!forceSmallTaskFanIn)
				return 0;
			const PxU32 maxTasksByParticles = forceSmallTaskFanIn
				? mParticles.size()
				: (mParticles.size() +
					eMIN_PARTICLES_PER_RIGID_CAPSULE_SDF_TASK - 1) /
					eMIN_PARTICLES_PER_RIGID_CAPSULE_SDF_TASK;
			return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
				mParticles.size());
		}

		bool beginRigidCapsuleSdfContactTaskTransaction()
		{
			if(!canUseRigidCapsuleSdfContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending ||
				mRigidBoxSdfContactTransactionPending ||
				mRigidSphereSdfContactTransactionPending ||
				mRigidCapsuleSdfContactTransactionPending ||
				mRigidConvexSdfContactTransactionPending)
				return false;
			Dy::avbdBeginSoftContactRedetection(
				mContacts, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, 0, false,
				0, 0, mRigidCapsules.size(), 0, 0, mBodies.size(),
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(),
				mSelfCollisionEnabled.begin());
			mRigidCapsuleSdfContactTransactionPending = true;
			return true;
		}

		void completeRigidCapsuleSdfContactTaskTransaction()
		{
			PX_ASSERT(mRigidCapsuleSdfContactTransactionPending);
			const PxU32 rigidStart = mContacts.size();
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidCapsuleSdfContactTaskOutputs.size(); ++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidCapsuleSdfContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0;
					contactIndex < source.size(); ++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			// P5.14b's only SDF-family merge point: every current-SDF range
			// precedes every swept-SDF range. Do not interleave both streams per
			// child even though a child evaluates its two leaves back-to-back.
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidCapsuleSweptSdfContactTaskOutputs.size();
				++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidCapsuleSweptSdfContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0;
					contactIndex < source.size(); ++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			if(mCollisionStatsEnabled)
				mLastCollisionStats.rigidParticleCapsuleTests +=
					PxU64(mParticles.size()) * mRigidCapsules.size();
			// Both feature suffixes remain parent-owned after current/swept fan-in.
			Dy::avbdDetectSoftRigidCapsuleSweptOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidCapsules.begin(), mRigidCapsules.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius,
				&mWorkspace.contact.rigidConvexForwardOwnerScratch);
			Dy::avbdDetectSoftRigidCapsuleOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidCapsules.begin(), mRigidCapsules.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius);
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedRigidContacts +=
					mContacts.size() - rigidStart;
			Dy::avbdCompleteSoftContactRedetection(
				mParticles.begin(), mContacts, mWorkspace.contact);
			removeRigidActorFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			removeDeformablePairFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			writeContactSetTrace(mContacts);
			mRigidCapsuleSdfContactTransactionPending = false;
		}

		bool ensureRigidCapsuleSdfContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner)
		{
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
			mRigidCapsuleSdfContactTasks.reserve(requiredSlots);
			mRigidCapsuleSdfContactFinishTasks.reserve(requiredSlots);
			mFreeRigidCapsuleSdfContactTaskIndices.reserve(requiredSlots);
			mFreeRigidCapsuleSdfContactFinishTaskIndices.reserve(requiredSlots);
			mRigidCapsuleSdfContactTaskOutputs.reserve(requiredChildTasks);
			mRigidCapsuleSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
			while(mRigidCapsuleSdfContactTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidCapsuleSdfContactTasks.size();
				mRigidCapsuleSdfContactTasks.pushBack(
					PX_NEW(RigidCapsuleSdfContactTask)(
						mContextId, *this, index));
				mFreeRigidCapsuleSdfContactTaskIndices.pushBack(index);
			}
			while(mRigidCapsuleSdfContactFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidCapsuleSdfContactFinishTasks.size();
				mRigidCapsuleSdfContactFinishTasks.pushBack(
					PX_NEW(RigidCapsuleSdfContactFinishTask)(
						mContextId, *this, owner, index));
				mFreeRigidCapsuleSdfContactFinishTaskIndices.pushBack(index);
			}
			return true;
		}

		RigidCapsuleSdfContactTask* acquireRigidCapsuleSdfContactTask()
		{
			PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
			if(mFreeRigidCapsuleSdfContactTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeRigidCapsuleSdfContactTaskIndices.back();
			mFreeRigidCapsuleSdfContactTaskIndices.popBack();
			return mRigidCapsuleSdfContactTasks[index];
		}

		RigidCapsuleSdfContactFinishTask*
			acquireRigidCapsuleSdfContactFinishTask()
		{
			PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
			if(mFreeRigidCapsuleSdfContactFinishTaskIndices.empty())
				return NULL;
			const PxU32 index =
				mFreeRigidCapsuleSdfContactFinishTaskIndices.back();
			mFreeRigidCapsuleSdfContactFinishTaskIndices.popBack();
			return mRigidCapsuleSdfContactFinishTasks[index];
		}

		void recycleRigidCapsuleSdfContactTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
			PX_ASSERT(index < mRigidCapsuleSdfContactTasks.size());
			mFreeRigidCapsuleSdfContactTaskIndices.pushBack(index);
		}

		void recycleRigidCapsuleSdfContactFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
			PX_ASSERT(index < mRigidCapsuleSdfContactFinishTasks.size());
			mFreeRigidCapsuleSdfContactFinishTaskIndices.pushBack(index);
		}

		bool submitStandaloneRigidCapsuleSdfContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			const PxU32 taskCount = getRigidCapsuleSdfContactTaskCount(
				dispatcherWorkers);
			if(!mRigidCapsuleSdfContactTransactionPending || !continuation ||
				taskCount == 0 ||
				!ensureRigidCapsuleSdfContactTaskPool(taskCount, owner))
				return false;
			{
				PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
				if(mFreeRigidCapsuleSdfContactTaskIndices.size() < taskCount ||
					mFreeRigidCapsuleSdfContactFinishTaskIndices.empty())
					return false;
			}
			const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
				mRigidCapsules.size();
			if(maxContactCount64 > PX_MAX_U32)
				return false;
			mContacts.reserve(PxU32(maxContactCount64));
			mRigidCapsuleSdfContactTaskOutputs.resize(taskCount);
			mRigidCapsuleSweptSdfContactTaskOutputs.resize(taskCount);
			const PxU32 particlesPerTask =
				(mParticles.size() + taskCount - 1) / taskCount;
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				PxArray<Dy::AvbdSoftContact>& output =
					mRigidCapsuleSdfContactTaskOutputs[taskIndex];
				output.clear();
				output.reserve((particleEnd - particleBegin) *
					mRigidCapsules.size());
				PxArray<Dy::AvbdSoftContact>& sweptOutput =
					mRigidCapsuleSweptSdfContactTaskOutputs[taskIndex];
				sweptOutput.clear();
				sweptOutput.reserve((particleEnd - particleBegin) *
					mRigidCapsules.size());
			}
			RigidCapsuleSdfContactFinishTask* const finishTask =
				acquireRigidCapsuleSdfContactFinishTask();
			if(!finishTask)
				return false;
			finishTask->setContinuation(continuation);
			taskGraphContext.recordRigidCapsuleSdfContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordRigidCapsuleSdfContactTasksSubmitted(taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				RigidCapsuleSdfContactTask* const task =
					acquireRigidCapsuleSdfContactTask();
				PX_ASSERT(task && particleBegin < particleEnd);
				if(!task)
				{
					recycleRigidCapsuleSdfContactFinishTask(
						finishTask->getPoolIndex());
					return false;
				}
				task->configure(
					mParticles.begin(), mParticles.size(),
					particleBegin, particleEnd,
					mRigidCapsules.begin(), mRigidCapsules.size(),
				mBodies.begin(), mBodies.size(),
				mRigidCapsuleSdfContactTaskOutputs[taskIndex],
				mRigidCapsuleSweptSdfContactTaskOutputs[taskIndex],
				mContactParams.contactRadius, taskGraphContext);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			finishTask->removeReference();
			return true;
		}

		bool completeStandaloneRigidCapsuleSdfContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mRigidCapsuleSdfContactTransactionPending)
				return false;
			completeRigidCapsuleSdfContactTaskTransaction();
			taskGraphContext.recordRigidCapsuleSdfContactFanIn();
			if(!mStandaloneComponentStepState.completePendingRedetection())
				return false;
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				advanceStandaloneComponentStateWithSceneRedetection(
					true, &nextWorldPlaneContactTaskReady,
					true, &nextRigidBoxSdfContactTaskReady,
					true, &nextRigidSphereSdfContactTaskReady);
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
				return true;
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		bool finishStandaloneRigidCapsuleSdfContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(!mRigidCapsuleSdfContactTransactionPending)
				return false;
			taskGraphContext.recordSerialRigidCapsuleSdfContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialRigidCapsuleSdfContactFallback();
			mRigidCapsuleSdfContactTaskOutputs.clear();
			mRigidCapsuleSweptSdfContactTaskOutputs.clear();
			Dy::avbdDetectSoftRigidCapsuleSDF(
				mParticles.begin(), mParticles.size(),
				mRigidCapsules.begin(), mRigidCapsules.size(), mContacts,
				mContactParams.contactRadius,
				mBodies.begin(), mBodies.size());
			// Match P5.14b's parent fan-in family order before the two feature
			// suffixes in completeRigidCapsuleSdfContactTaskTransaction().
			Dy::avbdDetectSoftRigidCapsuleSweptSDF(
				mParticles.begin(), mParticles.size(),
				mRigidCapsules.begin(), mRigidCapsules.size(), mContacts,
				mContactParams.contactRadius,
				mBodies.begin(), mBodies.size());
			return completeStandaloneRigidCapsuleSdfContactTask(
				dt, taskGraphContext, nextLayerReady,
				nextWorldPlaneContactTaskReady,
				nextRigidBoxSdfContactTaskReady,
				nextRigidSphereSdfContactTaskReady);
		}

		bool canUseRigidConvexSdfContactTaskTransaction() const
		{
			if(!Dy::avbdUseSceneRedetectionBridge() ||
				!Dy::avbdUseRigidConvexSdfContactTaskFanIn() ||
				mBodies.size() != 1 || mRigidConvexes.empty() ||
				!mWorldPlanes.empty() || !mRigidBoxes.empty() ||
				!mRigidSpheres.empty() || !mRigidCapsules.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				mSelfCollisionEnabled.size() != mBodies.size())
				return false;
			for(PxU32 index = 0; index < mRigidConvexes.size(); ++index)
			{
				if(mRigidConvexes[index].targetKind !=
					Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
					return false;
			}
			for(PxU32 index = 0; index < mSelfCollisionEnabled.size(); ++index)
			{
				if(mSelfCollisionEnabled[index])
					return false;
			}
			return true;
		}

		PxU32 getRigidConvexSdfContactTaskCount(PxU32 dispatcherWorkers) const
		{
			static const PxU32 eMIN_PARTICLES_PER_RIGID_CONVEX_SDF_TASK = 128;
			const bool forceSmallTaskFanIn =
				Dy::avbdForceRigidConvexSdfContactTaskFanIn();
			if(!canUseRigidConvexSdfContactTaskTransaction() ||
				dispatcherWorkers < 2 || mParticles.size() <
					eMIN_PARTICLES_PER_RIGID_CONVEX_SDF_TASK &&
				!forceSmallTaskFanIn)
				return 0;
			const PxU32 maxTasksByParticles = forceSmallTaskFanIn
				? mParticles.size()
				: (mParticles.size() +
				eMIN_PARTICLES_PER_RIGID_CONVEX_SDF_TASK - 1) /
					eMIN_PARTICLES_PER_RIGID_CONVEX_SDF_TASK;
			return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
				mParticles.size());
		}

		bool beginRigidConvexSdfContactTaskTransaction()
		{
			if(!canUseRigidConvexSdfContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending ||
				mRigidBoxSdfContactTransactionPending ||
				mRigidSphereSdfContactTransactionPending ||
				mRigidCapsuleSdfContactTransactionPending ||
				mRigidConvexSdfContactTransactionPending)
				return false;
			Dy::avbdBeginSoftContactRedetection(
				mContacts, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, 0, false,
				0, 0, 0, mRigidConvexes.size(), 0, mBodies.size(),
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(),
				mSelfCollisionEnabled.begin());
			mRigidConvexSdfContactTransactionPending = true;
			return true;
		}

		void completeRigidConvexSdfContactTaskTransaction()
		{
			PX_ASSERT(mRigidConvexSdfContactTransactionPending);
			const PxU32 rigidStart = mContacts.size();
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidConvexSdfContactTaskOutputs.size(); ++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidConvexSdfContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0; contactIndex < source.size();
					++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			// P5.15b's only SDF-family merge point: every current-SDF range
			// precedes every swept-SDF range. Do not interleave both streams per
			// child even though a child evaluates its two leaves back-to-back.
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidConvexSweptSdfContactTaskOutputs.size();
				++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidConvexSweptSdfContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0; contactIndex < source.size();
					++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			if(mCollisionStatsEnabled)
				mLastCollisionStats.rigidParticleConvexTests +=
					PxU64(mParticles.size()) * mRigidConvexes.size();
			// Both feature suffixes remain parent-owned after current/swept fan-in.
			Dy::avbdDetectSoftRigidConvexSweptOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidConvexes.begin(), mRigidConvexes.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius,
				&mWorkspace.contact.rigidConvexForwardOwnerScratch);
			Dy::avbdDetectSoftRigidConvexOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidConvexes.begin(), mRigidConvexes.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius);
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedRigidContacts +=
					mContacts.size() - rigidStart;
			Dy::avbdCompleteSoftContactRedetection(
				mParticles.begin(), mContacts, mWorkspace.contact);
			removeRigidActorFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			removeDeformablePairFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			writeContactSetTrace(mContacts);
			mRigidConvexSdfContactTransactionPending = false;
		}

		bool ensureRigidConvexSdfContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner)
		{
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
			mRigidConvexSdfContactTasks.reserve(requiredSlots);
			mRigidConvexSdfContactFinishTasks.reserve(requiredSlots);
			mFreeRigidConvexSdfContactTaskIndices.reserve(requiredSlots);
			mFreeRigidConvexSdfContactFinishTaskIndices.reserve(requiredSlots);
			mRigidConvexSdfContactTaskOutputs.reserve(requiredChildTasks);
			mRigidConvexSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
			while(mRigidConvexSdfContactTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidConvexSdfContactTasks.size();
				mRigidConvexSdfContactTasks.pushBack(
					PX_NEW(RigidConvexSdfContactTask)(mContextId, *this, index));
				mFreeRigidConvexSdfContactTaskIndices.pushBack(index);
			}
			while(mRigidConvexSdfContactFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidConvexSdfContactFinishTasks.size();
				mRigidConvexSdfContactFinishTasks.pushBack(
					PX_NEW(RigidConvexSdfContactFinishTask)(
						mContextId, *this, owner, index));
				mFreeRigidConvexSdfContactFinishTaskIndices.pushBack(index);
			}
			return true;
		}

		RigidConvexSdfContactTask* acquireRigidConvexSdfContactTask()
		{
			PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
			if(mFreeRigidConvexSdfContactTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeRigidConvexSdfContactTaskIndices.back();
			mFreeRigidConvexSdfContactTaskIndices.popBack();
			return mRigidConvexSdfContactTasks[index];
		}

		RigidConvexSdfContactFinishTask*
			acquireRigidConvexSdfContactFinishTask()
		{
			PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
			if(mFreeRigidConvexSdfContactFinishTaskIndices.empty())
				return NULL;
			const PxU32 index =
				mFreeRigidConvexSdfContactFinishTaskIndices.back();
			mFreeRigidConvexSdfContactFinishTaskIndices.popBack();
			return mRigidConvexSdfContactFinishTasks[index];
		}

		void recycleRigidConvexSdfContactTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
			PX_ASSERT(index < mRigidConvexSdfContactTasks.size());
			mFreeRigidConvexSdfContactTaskIndices.pushBack(index);
		}

		void recycleRigidConvexSdfContactFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
			PX_ASSERT(index < mRigidConvexSdfContactFinishTasks.size());
			mFreeRigidConvexSdfContactFinishTaskIndices.pushBack(index);
		}

		bool submitStandaloneRigidConvexSdfContactTask(
			PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			const PxU32 taskCount = getRigidConvexSdfContactTaskCount(
				dispatcherWorkers);
			if(!mRigidConvexSdfContactTransactionPending || !continuation ||
				taskCount == 0 ||
				!ensureRigidConvexSdfContactTaskPool(taskCount, owner))
				return false;
			{
				PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
				if(mFreeRigidConvexSdfContactTaskIndices.size() < taskCount ||
					mFreeRigidConvexSdfContactFinishTaskIndices.empty())
					return false;
			}
			const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
				mRigidConvexes.size();
			if(maxContactCount64 > PX_MAX_U32)
				return false;
			mContacts.reserve(PxU32(maxContactCount64));
			mRigidConvexSdfContactTaskOutputs.resize(taskCount);
			mRigidConvexSweptSdfContactTaskOutputs.resize(taskCount);
			const PxU32 particlesPerTask =
				(mParticles.size() + taskCount - 1) / taskCount;
			for(PxU32 index = 0; index < taskCount; ++index)
			{
				const PxU32 begin = index * particlesPerTask;
				const PxU32 end = PxMin(begin + particlesPerTask,
					mParticles.size());
				PxArray<Dy::AvbdSoftContact>& output =
					mRigidConvexSdfContactTaskOutputs[index];
				output.clear();
				output.reserve((end - begin) * mRigidConvexes.size());
				PxArray<Dy::AvbdSoftContact>& sweptOutput =
					mRigidConvexSweptSdfContactTaskOutputs[index];
				sweptOutput.clear();
				sweptOutput.reserve((end - begin) * mRigidConvexes.size());
			}
			RigidConvexSdfContactFinishTask* const finishTask =
				acquireRigidConvexSdfContactFinishTask();
			if(!finishTask)
				return false;
			finishTask->setContinuation(continuation);
			taskGraphContext.recordRigidConvexSdfContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordRigidConvexSdfContactTasksSubmitted(taskCount);
			for(PxU32 index = 0; index < taskCount; ++index)
			{
				const PxU32 begin = index * particlesPerTask;
				const PxU32 end = PxMin(begin + particlesPerTask,
					mParticles.size());
				RigidConvexSdfContactTask* const task =
					acquireRigidConvexSdfContactTask();
				if(!task)
				{
					recycleRigidConvexSdfContactFinishTask(
						finishTask->getPoolIndex());
					return false;
				}
				task->configure(mParticles.begin(), mParticles.size(), begin, end,
				mRigidConvexes.begin(), mRigidConvexes.size(),
				mBodies.begin(), mBodies.size(),
				mRigidConvexSdfContactTaskOutputs[index],
				mRigidConvexSweptSdfContactTaskOutputs[index],
				mContactParams.contactRadius, taskGraphContext);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			finishTask->removeReference();
			return true;
		}

		bool completeStandaloneRigidConvexSdfContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mRigidConvexSdfContactTransactionPending)
				return false;
			completeRigidConvexSdfContactTaskTransaction();
			taskGraphContext.recordRigidConvexSdfContactFanIn();
			if(!mStandaloneComponentStepState.completePendingRedetection())
				return false;
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				advanceStandaloneComponentStateWithSceneRedetection(
					true, &nextWorldPlaneContactTaskReady,
					true, &nextRigidBoxSdfContactTaskReady,
					true, &nextRigidSphereSdfContactTaskReady);
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
				return true;
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		bool finishStandaloneRigidConvexSdfContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(!mRigidConvexSdfContactTransactionPending)
				return false;
			taskGraphContext.recordSerialRigidConvexSdfContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialRigidConvexSdfContactFallback();
			mRigidConvexSdfContactTaskOutputs.clear();
			mRigidConvexSweptSdfContactTaskOutputs.clear();
			Dy::avbdDetectSoftRigidConvexSDF(
				mParticles.begin(), mParticles.size(),
				mRigidConvexes.begin(), mRigidConvexes.size(), mContacts,
				mContactParams.contactRadius,
				mBodies.begin(), mBodies.size());
			// Match P5.15b's parent fan-in family order before the two feature
			// suffixes in completeRigidConvexSdfContactTaskTransaction().
			Dy::avbdDetectSoftRigidConvexSweptSDF(
				mParticles.begin(), mParticles.size(),
				mRigidConvexes.begin(), mRigidConvexes.size(), mContacts,
				mContactParams.contactRadius,
				mBodies.begin(), mBodies.size());
			return completeStandaloneRigidConvexSdfContactTask(
				dt, taskGraphContext, nextLayerReady,
				nextWorldPlaneContactTaskReady,
				nextRigidBoxSdfContactTaskReady,
				nextRigidSphereSdfContactTaskReady);
		}

		bool canUseRigidTriangleSurfaceContactTaskTransaction() const
		{
			if(!Dy::avbdUseSceneRedetectionBridge() ||
				!Dy::avbdUseRigidTriangleSurfaceContactTaskFanIn() ||
				mBodies.size() != 1 || mRigidTriangleSurfaces.empty() ||
				!mWorldPlanes.empty() || !mRigidBoxes.empty() ||
				!mRigidSpheres.empty() || !mRigidCapsules.empty() ||
				!mRigidConvexes.empty() ||
				mSelfCollisionEnabled.size() != mBodies.size())
				return false;
			for(PxU32 index = 0; index < mRigidTriangleSurfaces.size(); ++index)
				if(mRigidTriangleSurfaces[index].targetKind !=
					Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
					return false;
			for(PxU32 index = 0; index < mSelfCollisionEnabled.size(); ++index)
				if(mSelfCollisionEnabled[index])
					return false;
			return true;
		}

		PxU32 getRigidTriangleSurfaceContactTaskCount(PxU32 dispatcherWorkers) const
		{
			static const PxU32 eMIN_PARTICLES_PER_RIGID_TRIANGLE_TASK = 128;
			static const PxU32 eP5_THRESHOLD_CANDIDATE_MIN_PARTICLES = 96;
			const PxU32 minParticlesPerTask =
				Dy::avbdUseRigidTriangleSurfaceContactTaskThreshold96()
					? eP5_THRESHOLD_CANDIDATE_MIN_PARTICLES
					: eMIN_PARTICLES_PER_RIGID_TRIANGLE_TASK;
			const bool forceSmallTaskFanIn =
				Dy::avbdForceRigidTriangleSurfaceContactTaskFanIn();
			if(!canUseRigidTriangleSurfaceContactTaskTransaction() ||
				dispatcherWorkers < 2 || mParticles.size() <
				minParticlesPerTask && !forceSmallTaskFanIn)
				return 0;
			const PxU32 maxTasksByParticles = forceSmallTaskFanIn
				? mParticles.size() : (mParticles.size() +
				minParticlesPerTask - 1) / minParticlesPerTask;
			return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
				mParticles.size());
		}

		bool beginRigidTriangleSurfaceContactTaskTransaction()
		{
			if(!canUseRigidTriangleSurfaceContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending ||
				mRigidBoxSdfContactTransactionPending ||
				mRigidSphereSdfContactTransactionPending ||
				mRigidCapsuleSdfContactTransactionPending ||
				mRigidConvexSdfContactTransactionPending ||
				mRigidTriangleSurfaceContactTransactionPending)
				return false;
			Dy::avbdBeginSoftContactRedetection(
				mContacts, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, 0, false, 0, 0, 0, 0,
				mRigidTriangleSurfaces.size(), mBodies.size(),
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(), mSelfCollisionEnabled.begin());
			// P5.17d builds feature identity once in the parent. The child leaf
			// receives only immutable row intervals, never a mutable broadphase.
			Dy::avbdBuildRigidTriangleSurfaceOGCFeaturePlan(
				mBodies.begin(), mBodies.size(),
				mRigidTriangleSurfaces.size(),
				mRigidTriangleSurfaceFeaturePlan);
			mRigidTriangleSurfaceContactTransactionPending = true;
			return true;
		}

		void completeRigidTriangleSurfaceContactTaskTransaction()
		{
			PX_ASSERT(mRigidTriangleSurfaceContactTransactionPending);
			const PxU32 rigidStart = mContacts.size();
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidTriangleSurfaceContactTaskOutputs.size(); ++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidTriangleSurfaceContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0; contactIndex < source.size();
					++contactIndex)
					mContacts.pushBack(source[contactIndex]);
				if(mCollisionStatsEnabled && taskIndex <
					mRigidTriangleSurfaceContactTaskStats.size())
					mLastCollisionStats.accumulate(
						mRigidTriangleSurfaceContactTaskStats[taskIndex]);
			}
			// P5.16b's only triangle-SDF family merge point: every current-SDF
			// range precedes every swept-SDF range. Child tasks may evaluate both
			// leaves back-to-back, but contacts must never be interleaved by task.
			for(PxU32 taskIndex = 0;
				taskIndex < mRigidTriangleSurfaceSweptSdfContactTaskOutputs.size();
				++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mRigidTriangleSurfaceSweptSdfContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0; contactIndex < source.size();
					++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			if(mCollisionStatsEnabled)
				mLastCollisionStats.rigidParticleTriangleSurfaceTests +=
					PxU64(mParticles.size()) * mRigidTriangleSurfaces.size();
			// P5.17d's only feature-family merge point. P5.27's opt-in
			// round-robin candidate gives each immutable plan row a private output,
			// then restores the same canonical row order here. The accepted route
			// retains its contiguous task-range merge.
			const PxArray<PxArray<Dy::AvbdSoftContact> >& featureOutputs =
				mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan ?
					mRigidTriangleSurfaceFeatureContactPlanOutputs :
					mRigidTriangleSurfaceFeatureContactTaskOutputs;
			for(PxU32 taskIndex = 0; taskIndex < featureOutputs.size();
				++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					featureOutputs[taskIndex];
				for(PxU32 contactIndex = 0; contactIndex < source.size();
					++contactIndex)
					mContacts.pushBack(source[contactIndex]);
			}
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedRigidContacts +=
					mContacts.size() - rigidStart;
			Dy::avbdCompleteSoftContactRedetection(
				mParticles.begin(), mContacts, mWorkspace.contact);
			removeRigidActorFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			removeDeformablePairFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			writeContactSetTrace(mContacts);
			mRigidTriangleSurfaceContactTransactionPending = false;
			mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan = false;
			mRigidTriangleSurfaceFeatureRoundRobinTaskPlan = false;
		}

		bool ensureRigidTriangleSurfaceContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner, Dy::AvbdDynamicsContext& taskGraphContext)
		{
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
			const PxU32 oldTaskPointerCapacity =
				mRigidTriangleSurfaceContactTasks.capacity();
			const PxU32 oldFinishPointerCapacity =
				mRigidTriangleSurfaceContactFinishTasks.capacity();
			const PxU32 oldFreeTaskIndexCapacity =
				mFreeRigidTriangleSurfaceContactTaskIndices.capacity();
			const PxU32 oldFreeFinishIndexCapacity =
				mFreeRigidTriangleSurfaceContactFinishTaskIndices.capacity();
			const PxU32 oldCurrentOutputCapacity =
				mRigidTriangleSurfaceContactTaskOutputs.capacity();
			const PxU32 oldSweptOutputCapacity =
				mRigidTriangleSurfaceSweptSdfContactTaskOutputs.capacity();
			const PxU32 oldFeatureOutputCapacity =
				mRigidTriangleSurfaceFeatureContactTaskOutputs.capacity();
			const PxU32 oldStatisticsCapacity =
				mRigidTriangleSurfaceContactTaskStats.capacity();
			const PxU32 oldTaskCount = mRigidTriangleSurfaceContactTasks.size();
			const PxU32 oldFinishTaskCount =
				mRigidTriangleSurfaceContactFinishTasks.size();
			mRigidTriangleSurfaceContactTasks.reserve(requiredSlots);
			mRigidTriangleSurfaceContactFinishTasks.reserve(requiredSlots);
			mFreeRigidTriangleSurfaceContactTaskIndices.reserve(requiredSlots);
			mFreeRigidTriangleSurfaceContactFinishTaskIndices.reserve(requiredSlots);
			mRigidTriangleSurfaceContactTaskOutputs.reserve(requiredChildTasks);
			mRigidTriangleSurfaceSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
			mRigidTriangleSurfaceFeatureContactTaskOutputs.reserve(requiredChildTasks);
			mRigidTriangleSurfaceContactTaskStats.reserve(requiredChildTasks);
			while(mRigidTriangleSurfaceContactTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidTriangleSurfaceContactTasks.size();
				mRigidTriangleSurfaceContactTasks.pushBack(
					PX_NEW(RigidTriangleSurfaceContactTask)(mContextId, *this, index));
				mFreeRigidTriangleSurfaceContactTaskIndices.pushBack(index);
			}
			while(mRigidTriangleSurfaceContactFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mRigidTriangleSurfaceContactFinishTasks.size();
				mRigidTriangleSurfaceContactFinishTasks.pushBack(
					PX_NEW(RigidTriangleSurfaceContactFinishTask)(
						mContextId, *this, owner, index));
				mFreeRigidTriangleSurfaceContactFinishTaskIndices.pushBack(index);
			}
			const PxU64 payloadGrowthBytes =
				PxU64(mRigidTriangleSurfaceContactTasks.capacity() -
					oldTaskPointerCapacity) * sizeof(RigidTriangleSurfaceContactTask*) +
				PxU64(mRigidTriangleSurfaceContactFinishTasks.capacity() -
					oldFinishPointerCapacity) *
						sizeof(RigidTriangleSurfaceContactFinishTask*) +
				PxU64(mFreeRigidTriangleSurfaceContactTaskIndices.capacity() -
					oldFreeTaskIndexCapacity) * sizeof(PxU32) +
				PxU64(mFreeRigidTriangleSurfaceContactFinishTaskIndices.capacity() -
					oldFreeFinishIndexCapacity) * sizeof(PxU32) +
				PxU64(mRigidTriangleSurfaceContactTaskOutputs.capacity() -
					oldCurrentOutputCapacity) *
						sizeof(PxArray<Dy::AvbdSoftContact>) +
				PxU64(mRigidTriangleSurfaceSweptSdfContactTaskOutputs.capacity() -
					oldSweptOutputCapacity) *
						sizeof(PxArray<Dy::AvbdSoftContact>) +
				PxU64(mRigidTriangleSurfaceFeatureContactTaskOutputs.capacity() -
					oldFeatureOutputCapacity) *
						sizeof(PxArray<Dy::AvbdSoftContact>) +
				PxU64(mRigidTriangleSurfaceContactTaskStats.capacity() -
					oldStatisticsCapacity) * sizeof(Dy::AvbdSoftCollisionStats) +
				PxU64(mRigidTriangleSurfaceContactTasks.size() - oldTaskCount) *
						sizeof(RigidTriangleSurfaceContactTask) +
				PxU64(mRigidTriangleSurfaceContactFinishTasks.size() -
					oldFinishTaskCount) *
						sizeof(RigidTriangleSurfaceContactFinishTask);
			taskGraphContext.recordRigidTriangleSurfaceContactTaskPoolGrowth(
				payloadGrowthBytes);
			return true;
		}

		PxU64 getRigidTriangleSurfaceContactTaskPoolResidentPayloadBytes()
		{
			PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
			return PxU64(mRigidTriangleSurfaceContactTasks.capacity()) *
					sizeof(RigidTriangleSurfaceContactTask*) +
				PxU64(mRigidTriangleSurfaceContactFinishTasks.capacity()) *
					sizeof(RigidTriangleSurfaceContactFinishTask*) +
				PxU64(mFreeRigidTriangleSurfaceContactTaskIndices.capacity()) *
					sizeof(PxU32) +
				PxU64(mFreeRigidTriangleSurfaceContactFinishTaskIndices.capacity()) *
					sizeof(PxU32) +
				PxU64(mRigidTriangleSurfaceContactTaskOutputs.capacity()) *
					sizeof(PxArray<Dy::AvbdSoftContact>) +
				PxU64(mRigidTriangleSurfaceSweptSdfContactTaskOutputs.capacity()) *
					sizeof(PxArray<Dy::AvbdSoftContact>) +
				PxU64(mRigidTriangleSurfaceFeatureContactTaskOutputs.capacity()) *
					sizeof(PxArray<Dy::AvbdSoftContact>) +
				PxU64(mRigidTriangleSurfaceFeatureContactPlanOutputs.capacity()) *
					sizeof(PxArray<Dy::AvbdSoftContact>) +
				PxU64(mRigidTriangleSurfaceContactTaskStats.capacity()) *
					sizeof(Dy::AvbdSoftCollisionStats) +
				PxU64(mRigidTriangleSurfaceContactTasks.size()) *
					sizeof(RigidTriangleSurfaceContactTask) +
				PxU64(mRigidTriangleSurfaceContactFinishTasks.size()) *
					sizeof(RigidTriangleSurfaceContactFinishTask);
		}

		RigidTriangleSurfaceContactTask* acquireRigidTriangleSurfaceContactTask()
		{
			PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
			if(mFreeRigidTriangleSurfaceContactTaskIndices.empty()) return NULL;
			const PxU32 index = mFreeRigidTriangleSurfaceContactTaskIndices.back();
			mFreeRigidTriangleSurfaceContactTaskIndices.popBack();
			return mRigidTriangleSurfaceContactTasks[index];
		}

		RigidTriangleSurfaceContactFinishTask*
			acquireRigidTriangleSurfaceContactFinishTask()
		{
			PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
			if(mFreeRigidTriangleSurfaceContactFinishTaskIndices.empty()) return NULL;
			const PxU32 index =
				mFreeRigidTriangleSurfaceContactFinishTaskIndices.back();
			mFreeRigidTriangleSurfaceContactFinishTaskIndices.popBack();
			return mRigidTriangleSurfaceContactFinishTasks[index];
		}

		void recycleRigidTriangleSurfaceContactTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
			PX_ASSERT(index < mRigidTriangleSurfaceContactTasks.size());
			mFreeRigidTriangleSurfaceContactTaskIndices.pushBack(index);
		}

		void recycleRigidTriangleSurfaceContactFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
			PX_ASSERT(index < mRigidTriangleSurfaceContactFinishTasks.size());
			mFreeRigidTriangleSurfaceContactFinishTaskIndices.pushBack(index);
		}

		bool submitStandaloneRigidTriangleSurfaceContactTask(
			PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			const PxU32 taskCount = getRigidTriangleSurfaceContactTaskCount(
				dispatcherWorkers);
			if(!mRigidTriangleSurfaceContactTransactionPending || !continuation ||
				taskCount == 0 || !ensureRigidTriangleSurfaceContactTaskPool(
					taskCount, owner, taskGraphContext)) return false;
			{
				PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
				if(mFreeRigidTriangleSurfaceContactTaskIndices.size() < taskCount ||
					mFreeRigidTriangleSurfaceContactFinishTaskIndices.empty()) return false;
			}
			const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
				mRigidTriangleSurfaces.size();
			if(maxContactCount64 > PX_MAX_U32 / 2u) return false;
			PxU32 maxTriangleCandidateCount = 0;
			PxU32 maxEdgeCandidateCount = 0;
			PxU32 maxVertexCandidateCount = 0;
			for(PxU32 index = 0; index < mRigidTriangleSurfaces.size(); ++index)
			{
				maxTriangleCandidateCount = PxMax(maxTriangleCandidateCount,
					mRigidTriangleSurfaces[index].triangleBvhTriangleIndices.size());
				maxEdgeCandidateCount = PxMax(maxEdgeCandidateCount,
					mRigidTriangleSurfaces[index].edges.size());
				maxVertexCandidateCount = PxMax(maxVertexCandidateCount,
					mRigidTriangleSurfaces[index].vertices.size());
			}
			mContacts.reserve(PxU32(maxContactCount64 * 2u));
			const PxU32 featurePlanCount =
				mRigidTriangleSurfaceFeaturePlan.items.size();
			const bool useFeaturePlanRoundRobin = featurePlanCount > 0 &&
				Dy::avbdUseRigidTriangleSurfaceFeatureRoundRobinTaskPlan();
			const bool useFeaturePlanRowPrivateOutputs = featurePlanCount > 0 &&
				(useFeaturePlanRoundRobin ||
					Dy::avbdUseRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan());
			const bool useFeatureForwardOwnerQueryStats = featurePlanCount > 0 &&
				Dy::avbdUseRigidTriangleSurfaceFeatureForwardOwnerQueryStats();
			const bool useFeatureDiscreteQueryStats = featurePlanCount > 0 &&
				Dy::avbdUseRigidTriangleSurfaceFeatureDiscreteQueryStats();
			// P5.41 promotes P5.39's conservative cull only after this already
			// opt-in Scene task route has been admitted. The legacy disable is
			// for controlled A/B and must not broaden the serial/global policy.
			const bool useFeatureDiscreteBodyLocalBoundsCull =
				featurePlanCount > 0 &&
				!Dy::avbdDisableRigidTriangleSurfaceFeatureDiscreteBodyLocalBoundsCull();
			// P5.35's accepted triangle task caches the P5.31-proven exact
			// forward-owner result by default. The overall Scene task route is
			// still opt-in; the disable switch exists only for controlled legacy
			// comparisons and must not be treated as a default policy branch.
			const bool useFeatureForwardOwnerResultCache = featurePlanCount > 0 &&
				!Dy::avbdDisableRigidTriangleSurfaceFeatureForwardOwnerResultCache();
			const PxU64 forwardOwnerQueryStampCapacity64 =
				useFeatureForwardOwnerQueryStats ?
					PxU64(mParticles.size()) * mRigidTriangleSurfaces.size() : 0;
			if(forwardOwnerQueryStampCapacity64 > PX_MAX_U32) return false;
			const PxU32 oldFeaturePlanOutputCapacity =
				mRigidTriangleSurfaceFeatureContactPlanOutputs.capacity();
			mRigidTriangleSurfaceContactTaskOutputs.resize(taskCount);
			mRigidTriangleSurfaceSweptSdfContactTaskOutputs.resize(taskCount);
			mRigidTriangleSurfaceFeatureContactTaskOutputs.resize(taskCount);
			if(useFeaturePlanRowPrivateOutputs)
				mRigidTriangleSurfaceFeatureContactPlanOutputs.resize(featurePlanCount);
			else
				mRigidTriangleSurfaceFeatureContactPlanOutputs.clear();
			mRigidTriangleSurfaceContactTaskStats.resize(taskCount);
			mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan =
				useFeaturePlanRowPrivateOutputs;
			mRigidTriangleSurfaceFeatureRoundRobinTaskPlan = useFeaturePlanRoundRobin;
			taskGraphContext.recordRigidTriangleSurfaceContactTaskPoolGrowth(
				PxU64(mRigidTriangleSurfaceFeatureContactPlanOutputs.capacity() -
					oldFeaturePlanOutputCapacity) *
				sizeof(PxArray<Dy::AvbdSoftContact>));
			const PxU64 taskPoolResidentPayloadBytes =
				getRigidTriangleSurfaceContactTaskPoolResidentPayloadBytes();
			const PxU32 particlesPerTask =
				(mParticles.size() + taskCount - 1) / taskCount;
			const PxU32 featureRowsPerTask = featurePlanCount == 0 ? 0 :
				(featurePlanCount + taskCount - 1) / taskCount;
			PxU64 featureEdgePairWorkItems = 0;
			PxU64 featureFacePairWorkItems = 0;
			PxU64 outputGrowthBytes = 0;
			PxU64 outputResidentPayloadBytes = 0;
			PxU64 queryScratchResidentPayloadBytes = 0;
			PxU32 featureNonEmptyTaskRanges = 0;
			PxU32 featureMaxRowsPerTaskRange = 0;
			PxU64 maxCurrentSdfParticleWorkItemsPerTask = 0;
			PxU64 maxFeatureEdgePairWorkItemsPerTask = 0;
			PxU64 maxFeatureFacePairWorkItemsPerTask = 0;
			PxU64 maxFeaturePairWorkItemsPerTask = 0;
			for(PxU32 index = 0; index < taskCount; ++index)
			{
				const PxU32 begin = index * particlesPerTask;
				const PxU32 end = PxMin(begin + particlesPerTask, mParticles.size());
				maxCurrentSdfParticleWorkItemsPerTask = PxMax(
					maxCurrentSdfParticleWorkItemsPerTask,
					PxU64(end - begin) * mRigidTriangleSurfaces.size());
				const PxU32 oldCurrentOutputCapacity =
					mRigidTriangleSurfaceContactTaskOutputs[index].capacity();
				const PxU32 oldSweptOutputCapacity =
					mRigidTriangleSurfaceSweptSdfContactTaskOutputs[index].capacity();
				const PxU32 oldFeatureOutputCapacity =
					mRigidTriangleSurfaceFeatureContactTaskOutputs[index].capacity();
				mRigidTriangleSurfaceContactTaskOutputs[index].clear();
				mRigidTriangleSurfaceContactTaskOutputs[index].reserve(
					(end - begin) * mRigidTriangleSurfaces.size());
				mRigidTriangleSurfaceSweptSdfContactTaskOutputs[index].clear();
				mRigidTriangleSurfaceSweptSdfContactTaskOutputs[index].reserve(
					(end - begin) * mRigidTriangleSurfaces.size());
				const PxU32 featurePlanBegin = useFeaturePlanRoundRobin ? index :
					PxMin(index * featureRowsPerTask, featurePlanCount);
				const PxU32 featurePlanEnd = useFeaturePlanRoundRobin ?
					featurePlanCount : PxMin(
						featurePlanBegin + featureRowsPerTask, featurePlanCount);
				const PxU32 featurePlanStride = useFeaturePlanRoundRobin ?
					taskCount : 1u;
				PxU32 featureRowCount = 0;
				PxU64 featureContactCapacity = 0;
				PxU64 featureEdgePairWorkItemsThisTask = 0;
				PxU64 featureFacePairWorkItemsThisTask = 0;
				PxU64 featureOutputGrowthBytes = 0;
				PxU64 featureOutputResidentPayloadBytes = 0;
				for(PxU32 planIndex = featurePlanBegin;
					planIndex < featurePlanEnd; planIndex += featurePlanStride)
				{
					++featureRowCount;
					const Dy::AvbdRigidTriangleSurfaceFeatureWorkItem& workItem =
						mRigidTriangleSurfaceFeaturePlan.items[planIndex];
					PX_ASSERT(workItem.bodyIndex < mBodies.size() &&
						workItem.surfaceIndex < mRigidTriangleSurfaces.size());
					const Dy::AvbdRigidTriangleSurface& surface =
						mRigidTriangleSurfaces[workItem.surfaceIndex];
					const PxU64 primitivePairWorkItems = workItem.family ==
						Dy::AvbdRigidTriangleSurfaceFeatureWorkItem::eSOFT_EDGE
						? PxU64(workItem.primitiveEnd - workItem.primitiveBegin) *
							surface.edges.size()
						: PxU64(workItem.primitiveEnd - workItem.primitiveBegin) *
							surface.vertices.size();
					if(primitivePairWorkItems > PX_MAX_U32) return false;
					featureContactCapacity += primitivePairWorkItems;
					if(workItem.family ==
						Dy::AvbdRigidTriangleSurfaceFeatureWorkItem::eSOFT_EDGE)
					{
						featureEdgePairWorkItems += primitivePairWorkItems;
						featureEdgePairWorkItemsThisTask += primitivePairWorkItems;
					}
					else
					{
						featureFacePairWorkItems += primitivePairWorkItems;
						featureFacePairWorkItemsThisTask += primitivePairWorkItems;
					}
					if(featureContactCapacity > PX_MAX_U32) return false;
					if(useFeaturePlanRowPrivateOutputs)
					{
						PxArray<Dy::AvbdSoftContact>& featureOutput =
							mRigidTriangleSurfaceFeatureContactPlanOutputs[planIndex];
						const PxU32 oldFeaturePlanRowOutputCapacity =
							featureOutput.capacity();
						featureOutput.clear();
						featureOutput.reserve(PxU32(primitivePairWorkItems));
						featureOutputGrowthBytes += PxU64(
							featureOutput.capacity() - oldFeaturePlanRowOutputCapacity) *
							sizeof(Dy::AvbdSoftContact);
						featureOutputResidentPayloadBytes += PxU64(
							featureOutput.capacity()) * sizeof(Dy::AvbdSoftContact);
					}
				}
				if(featureRowCount)
				{
					++featureNonEmptyTaskRanges;
					featureMaxRowsPerTaskRange = PxMax(
						featureMaxRowsPerTaskRange, featureRowCount);
				}
				maxFeatureEdgePairWorkItemsPerTask = PxMax(
					maxFeatureEdgePairWorkItemsPerTask,
					featureEdgePairWorkItemsThisTask);
				maxFeatureFacePairWorkItemsPerTask = PxMax(
					maxFeatureFacePairWorkItemsPerTask,
					featureFacePairWorkItemsThisTask);
				maxFeaturePairWorkItemsPerTask = PxMax(
					maxFeaturePairWorkItemsPerTask,
					featureEdgePairWorkItemsThisTask + featureFacePairWorkItemsThisTask);
				if(!useFeaturePlanRowPrivateOutputs)
				{
					mRigidTriangleSurfaceFeatureContactTaskOutputs[index].clear();
					mRigidTriangleSurfaceFeatureContactTaskOutputs[index].reserve(
						PxU32(featureContactCapacity));
					featureOutputGrowthBytes += PxU64(
						mRigidTriangleSurfaceFeatureContactTaskOutputs[index].capacity() -
						oldFeatureOutputCapacity) * sizeof(Dy::AvbdSoftContact);
					featureOutputResidentPayloadBytes += PxU64(
						mRigidTriangleSurfaceFeatureContactTaskOutputs[index].capacity()) *
						sizeof(Dy::AvbdSoftContact);
				}
				outputGrowthBytes += PxU64(
					mRigidTriangleSurfaceContactTaskOutputs[index].capacity() -
						oldCurrentOutputCapacity) * sizeof(Dy::AvbdSoftContact) +
					PxU64(mRigidTriangleSurfaceSweptSdfContactTaskOutputs[index].capacity() -
						oldSweptOutputCapacity) * sizeof(Dy::AvbdSoftContact) +
					featureOutputGrowthBytes;
				outputResidentPayloadBytes += PxU64(
					mRigidTriangleSurfaceContactTaskOutputs[index].capacity()) *
					sizeof(Dy::AvbdSoftContact) + PxU64(
					mRigidTriangleSurfaceSweptSdfContactTaskOutputs[index].capacity()) *
					sizeof(Dy::AvbdSoftContact) +
					featureOutputResidentPayloadBytes;
				mRigidTriangleSurfaceContactTaskStats[index] =
					Dy::AvbdSoftCollisionStats();
			}
			RigidTriangleSurfaceContactFinishTask* const finishTask =
				acquireRigidTriangleSurfaceContactFinishTask();
			if(!finishTask) return false;
			finishTask->setContinuation(continuation);
				taskGraphContext.recordRigidTriangleSurfaceContactWork(
				PxU64(mParticles.size()) * mRigidTriangleSurfaces.size(),
				PxU64(mParticles.size()) * mRigidTriangleSurfaces.size(),
				featurePlanCount, featureEdgePairWorkItems,
				featureFacePairWorkItems, featureNonEmptyTaskRanges,
				featureMaxRowsPerTaskRange,
				maxCurrentSdfParticleWorkItemsPerTask,
				maxCurrentSdfParticleWorkItemsPerTask,
				maxFeatureEdgePairWorkItemsPerTask,
				maxFeatureFacePairWorkItemsPerTask,
				maxFeaturePairWorkItemsPerTask);
			if(useFeaturePlanRoundRobin)
				taskGraphContext.recordRigidTriangleSurfaceContactFeatureRoundRobinTaskFanIn();
			else if(useFeaturePlanRowPrivateOutputs)
				taskGraphContext.recordRigidTriangleSurfaceContactFeatureRowPrivateOutputTaskFanIn();
			taskGraphContext.recordRigidTriangleSurfaceContactOutputGrowth(
				outputGrowthBytes);
			taskGraphContext.recordRigidTriangleSurfaceContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordRigidTriangleSurfaceContactTasksSubmitted(taskCount);
			mRigidTriangleSurfaceContactTaskSubmitStartNanos =
				mCollisionStatsEnabled ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
			for(PxU32 index = 0; index < taskCount; ++index)
			{
				const PxU32 begin = index * particlesPerTask;
				const PxU32 end = PxMin(begin + particlesPerTask, mParticles.size());
				const PxU32 featurePlanBegin = useFeaturePlanRoundRobin ? index :
					PxMin(index * featureRowsPerTask, featurePlanCount);
				const PxU32 featurePlanEnd = useFeaturePlanRoundRobin ?
					featurePlanCount : PxMin(
						featurePlanBegin + featureRowsPerTask, featurePlanCount);
				const PxU32 featurePlanStride = useFeaturePlanRoundRobin ?
					taskCount : 1u;
				PxU32 forwardOwnerResultCacheSurfaceCount = 0;
				if(useFeatureForwardOwnerResultCache)
				{
					for(PxU32 planIndex = featurePlanBegin;
						planIndex < featurePlanEnd; planIndex += featurePlanStride)
					{
						const Dy::AvbdRigidTriangleSurfaceFeatureWorkItem& item =
							mRigidTriangleSurfaceFeaturePlan.items[planIndex];
						if(item.phase !=
							Dy::AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT)
							continue;
						bool alreadyCounted = false;
						for(PxU32 previousPlanIndex = featurePlanBegin;
							previousPlanIndex < planIndex;
							previousPlanIndex += featurePlanStride)
						{
							const Dy::AvbdRigidTriangleSurfaceFeatureWorkItem& previous =
								mRigidTriangleSurfaceFeaturePlan.items[
									previousPlanIndex];
							if(previous.phase ==
								Dy::AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT &&
								previous.surfaceIndex == item.surfaceIndex)
							{
								alreadyCounted = true;
								break;
							}
						}
						if(!alreadyCounted)
							++forwardOwnerResultCacheSurfaceCount;
					}
				}
				const PxU64 forwardOwnerResultCacheTaskCapacity64 =
					PxU64(mParticles.size()) *
					forwardOwnerResultCacheSurfaceCount;
				if(forwardOwnerResultCacheTaskCapacity64 > PX_MAX_U32)
					return false;
				RigidTriangleSurfaceContactTask* const task =
					acquireRigidTriangleSurfaceContactTask();
				if(!task)
				{
					mRigidTriangleSurfaceContactTaskSubmitStartNanos = 0;
					mRigidTriangleSurfaceContactTaskSubmitEndNanos = 0;
					recycleRigidTriangleSurfaceContactFinishTask(
						finishTask->getPoolIndex());
					return false;
				}
				taskGraphContext.recordRigidTriangleSurfaceContactQueryScratchGrowth(
					task->reserveBvhCandidateScratch(maxTriangleCandidateCount,
						maxEdgeCandidateCount, maxVertexCandidateCount,
						PxU32(forwardOwnerQueryStampCapacity64),
						PxU32(forwardOwnerResultCacheTaskCapacity64),
						forwardOwnerResultCacheSurfaceCount));
				queryScratchResidentPayloadBytes +=
					task->getBvhCandidateScratchResidentPayloadBytes();
				task->configure(mParticles.begin(), mParticles.size(), begin, end,
					mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
					mBodies.begin(), mBodies.size(),
					mRigidTriangleSurfaceContactTaskOutputs[index],
					mRigidTriangleSurfaceSweptSdfContactTaskOutputs[index],
					mRigidTriangleSurfaceFeaturePlan,
					featurePlanBegin, featurePlanEnd,
					mRigidTriangleSurfaceFeatureContactTaskOutputs[index],
					mContactParams.contactRadius,
					mCollisionStatsEnabled ?
						&mRigidTriangleSurfaceContactTaskStats[index] : NULL,
					taskGraphContext);
				if(useFeatureForwardOwnerQueryStats)
					task->configureForwardOwnerQueryStats();
				if(useFeatureDiscreteQueryStats)
					task->configureDiscreteQueryStats();
				if(useFeatureDiscreteBodyLocalBoundsCull)
					task->configureDiscreteBodyLocalBoundsCull();
				if(useFeatureForwardOwnerResultCache &&
					forwardOwnerResultCacheSurfaceCount > 0)
					task->configureForwardOwnerResultCache();
				if(useFeaturePlanRoundRobin)
					task->configureFeaturePlanRoundRobin(
						mRigidTriangleSurfaceFeatureContactPlanOutputs,
						index, taskCount);
				else if(useFeaturePlanRowPrivateOutputs)
					task->configureFeaturePlanRowPrivateOutputs(
						mRigidTriangleSurfaceFeatureContactPlanOutputs);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			taskGraphContext.recordRigidTriangleSurfaceContactResidentCapacity(
				taskPoolResidentPayloadBytes, outputResidentPayloadBytes,
				queryScratchResidentPayloadBytes);
			// Store the end before releasing the finish task's final parent
			// reference: that release can immediately execute the fan-in.
			mRigidTriangleSurfaceContactTaskSubmitEndNanos =
				mCollisionStatsEnabled ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
			if(mCollisionStatsEnabled)
				taskGraphContext.recordRigidTriangleSurfaceContactTaskSubmissionWallTime(
					mRigidTriangleSurfaceContactTaskSubmitEndNanos -
					mRigidTriangleSurfaceContactTaskSubmitStartNanos);
			finishTask->removeReference();
			return true;
		}

		bool completeStandaloneRigidTriangleSurfaceContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mRigidTriangleSurfaceContactTransactionPending) return false;
			if(mRigidTriangleSurfaceContactTaskSubmitStartNanos)
			{
				const PxU64 fanInEndNanos =
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u;
				taskGraphContext.recordRigidTriangleSurfaceContactFanInWallSpan(
					fanInEndNanos - mRigidTriangleSurfaceContactTaskSubmitStartNanos);
				if(mRigidTriangleSurfaceContactTaskSubmitEndNanos)
					taskGraphContext.recordRigidTriangleSurfaceContactPostSubmitWaitWallTime(
						fanInEndNanos -
						mRigidTriangleSurfaceContactTaskSubmitEndNanos);
				mRigidTriangleSurfaceContactTaskSubmitStartNanos = 0;
				mRigidTriangleSurfaceContactTaskSubmitEndNanos = 0;
			}
			// P5.23 keeps the parent-only completion distinct from the child
			// critical path above. This includes canonical stream merge and the
			// fixed contact completion boundary, never task submission or child
			// execution.
			const PxU64 parentCompletionStartNanos = mCollisionStatsEnabled ?
				PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
			completeRigidTriangleSurfaceContactTaskTransaction();
			taskGraphContext.recordRigidTriangleSurfaceContactFanIn();
			if(!mStandaloneComponentStepState.completePendingRedetection()) return false;
			if(mCollisionStatsEnabled)
				taskGraphContext.recordRigidTriangleSurfaceContactParentCompletionWallTime(
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
					parentCompletionStartNanos);
			const PxU64 postContinuationStartNanos = mCollisionStatsEnabled ?
				PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				advanceStandaloneComponentStateWithSceneRedetection(
					true, &nextWorldPlaneContactTaskReady,
					true, &nextRigidBoxSdfContactTaskReady,
					true, &nextRigidSphereSdfContactTaskReady);
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
			{
				if(mCollisionStatsEnabled)
					taskGraphContext.recordRigidTriangleSurfaceContactPostContinuationWallTime(
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						postContinuationStartNanos);
				return true;
			}
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				if(mCollisionStatsEnabled)
					taskGraphContext.recordRigidTriangleSurfaceContactPostContinuationWallTime(
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						postContinuationStartNanos);
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
			{
				if(mCollisionStatsEnabled)
					taskGraphContext.recordRigidTriangleSurfaceContactPostContinuationWallTime(
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						postContinuationStartNanos);
				return false;
			}
			finishStandaloneComponentSolve(dt);
			if(mCollisionStatsEnabled)
				taskGraphContext.recordRigidTriangleSurfaceContactPostContinuationWallTime(
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
					postContinuationStartNanos);
			return true;
		}

		bool finishStandaloneRigidTriangleSurfaceContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(!mRigidTriangleSurfaceContactTransactionPending) return false;
			taskGraphContext.recordSerialRigidTriangleSurfaceContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialRigidTriangleSurfaceContactFallback();
			mRigidTriangleSurfaceContactTaskOutputs.clear();
			mRigidTriangleSurfaceSweptSdfContactTaskOutputs.clear();
			mRigidTriangleSurfaceFeatureContactTaskOutputs.clear();
			mRigidTriangleSurfaceFeatureContactPlanOutputs.clear();
			mRigidTriangleSurfaceContactTaskStats.clear();
			mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan = false;
			mRigidTriangleSurfaceFeatureRoundRobinTaskPlan = false;
			mRigidTriangleSurfaceContactTaskSubmitEndNanos = 0;
			// P5.23's serial leaf covers exactly the current, swept and OGC
			// feature predicates that task children execute. Parent merge and
			// post-contact continuation are timed by the shared completion path.
			const PxU64 serialTransactionStartNanos = mCollisionStatsEnabled ?
				PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
			Dy::avbdDetectSoftRigidTriangleSurface(
				mParticles.begin(), mParticles.size(),
				mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
				mContacts, mContactParams.contactRadius,
				mBodies.begin(), mBodies.size(),
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			// Submission failure keeps the serial wrapper as the authority. The
			// transaction completion below now only merges child streams, so the
			// fallback explicitly retains the old current -> swept -> feature
			// suffix sequence before parent-only mutable completion.
			Dy::avbdDetectSoftRigidTriangleSurfaceSwept(
				mParticles.begin(), mParticles.size(),
				mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
				mContacts, mContactParams.contactRadius,
				mBodies.begin(), mBodies.size(),
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL, NULL,
				&mWorkspace.contact.rigidTriangleSurfaceForwardOwnerScratch);
			Dy::avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			if(mCollisionStatsEnabled)
				taskGraphContext.recordRigidTriangleSurfaceContactSerialTransactionWallTime(
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
					serialTransactionStartNanos);
			return completeStandaloneRigidTriangleSurfaceContactTask(
				dt, taskGraphContext, nextLayerReady,
				nextWorldPlaneContactTaskReady,
				nextRigidBoxSdfContactTaskReady,
				nextRigidSphereSdfContactTaskReady);
		}

		bool canUseSoftPairContactTaskTransaction() const
		{
			if(!Dy::avbdUseSceneRedetectionBridge() ||
				!Dy::avbdUseSoftPairContactTaskFanIn() ||
				mBodies.size() < 2 || !mWorldPlanes.empty() ||
				!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
				!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				mSelfCollisionEnabled.size() != mBodies.size())
				return false;
			// This first leaf is deliberately pure soft-pair OGC. Any self source
			// remains a serial owner until it can prove independent scratch and
			// ordering under the same Scene boundary.
			for(PxU32 bodyIndex = 0; bodyIndex < mSelfCollisionEnabled.size();
				++bodyIndex)
			{
				if(mSelfCollisionEnabled[bodyIndex])
					return false;
			}
			return true;
		}

		PxU32 getSoftPairContactTaskCount(PxU32 dispatcherWorkers) const
		{
			const PxU32 planCount =
				mWorkspace.contact.softPairDetectionPlan.size();
			if(!canUseSoftPairContactTaskTransaction() ||
				dispatcherWorkers < 2 || planCount == 0)
				return 0;
			// One task per immutable pair-plan range is the first safe unit. A
			// single pair is intentionally serial: splitting inside its VF/EE
			// stream would introduce a second ordering contract.
			if(planCount < 2 && !Dy::avbdForceSoftPairContactTaskFanIn())
				return 0;
			return PxMin(dispatcherWorkers, planCount);
		}

		bool beginSoftPairContactTaskTransaction()
		{
			if(!canUseSoftPairContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending ||
				mRigidBoxSdfContactTransactionPending ||
				mRigidSphereSdfContactTransactionPending ||
				mRigidCapsuleSdfContactTransactionPending ||
				mRigidConvexSdfContactTransactionPending ||
				mRigidTriangleSurfaceContactTransactionPending ||
				mSoftPairContactTransactionPending)
				return false;
			Dy::avbdBeginSoftContactRedetection(
				mContacts, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, 0, false, 0, 0, 0, 0, 0,
				mBodies.size(), mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(), mSelfCollisionEnabled.begin());
			Dy::avbdBuildSoftSoftOGCDetectionPlan(
				mParticles.begin(), mBodies.begin(), mBodies.size(),
				mContactParams,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
				mWorkspace.contact);
			mSoftPairContactUseSurfaceTriangleBvh =
				Dy::avbdRefitSoftSoftOGCDetectionPlan(
					mParticles.begin(), mBodies.begin(), mBodies.size(),
					mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
					mWorkspace.contact);
			mSoftPairContactTransactionPending = true;
			return true;
		}

		void completeSoftPairContactTaskTransaction()
		{
			PX_ASSERT(mSoftPairContactTransactionPending);
			const PxU32 softStart = mContacts.size();
			for(PxU32 taskIndex = 0;
				taskIndex < mSoftPairContactTaskOutputs.size(); ++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mSoftPairContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0; contactIndex < source.size();
					++contactIndex)
					mContacts.pushBack(source[contactIndex]);
				if(mCollisionStatsEnabled && taskIndex <
					mSoftPairContactTaskStats.size())
					mLastCollisionStats.accumulate(
						mSoftPairContactTaskStats[taskIndex]);
			}
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedSoftContacts +=
					mContacts.size() - softStart;
			Dy::avbdCompleteSoftContactRedetection(
				mParticles.begin(), mContacts, mWorkspace.contact);
			removeRigidActorFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			removeDeformablePairFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			writeContactSetTrace(mContacts);
			mSoftPairContactTransactionPending = false;
			mSoftPairContactUseSurfaceTriangleBvh = false;
		}

		bool ensureSoftPairContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner)
		{
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mSoftPairContactTaskPoolMutex);
			mSoftPairContactTasks.reserve(requiredSlots);
			mSoftPairContactFinishTasks.reserve(requiredSlots);
			mFreeSoftPairContactTaskIndices.reserve(requiredSlots);
			mFreeSoftPairContactFinishTaskIndices.reserve(requiredSlots);
			mSoftPairContactTaskOutputs.reserve(requiredChildTasks);
			mSoftPairContactTaskStats.reserve(requiredChildTasks);
			while(mSoftPairContactTasks.size() < requiredSlots)
			{
				const PxU32 index = mSoftPairContactTasks.size();
				mSoftPairContactTasks.pushBack(PX_NEW(SoftPairContactTask)(
					mContextId, *this, index));
				mFreeSoftPairContactTaskIndices.pushBack(index);
			}
			while(mSoftPairContactFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mSoftPairContactFinishTasks.size();
				mSoftPairContactFinishTasks.pushBack(
					PX_NEW(SoftPairContactFinishTask)(
						mContextId, *this, owner, index));
				mFreeSoftPairContactFinishTaskIndices.pushBack(index);
			}
			return true;
		}

		SoftPairContactTask* acquireSoftPairContactTask()
		{
			PxMutex::ScopedLock lock(mSoftPairContactTaskPoolMutex);
			if(mFreeSoftPairContactTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeSoftPairContactTaskIndices.back();
			mFreeSoftPairContactTaskIndices.popBack();
			return mSoftPairContactTasks[index];
		}

		SoftPairContactFinishTask* acquireSoftPairContactFinishTask()
		{
			PxMutex::ScopedLock lock(mSoftPairContactTaskPoolMutex);
			if(mFreeSoftPairContactFinishTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeSoftPairContactFinishTaskIndices.back();
			mFreeSoftPairContactFinishTaskIndices.popBack();
			return mSoftPairContactFinishTasks[index];
		}

		void recycleSoftPairContactTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mSoftPairContactTaskPoolMutex);
			PX_ASSERT(index < mSoftPairContactTasks.size());
			mFreeSoftPairContactTaskIndices.pushBack(index);
		}

		void recycleSoftPairContactFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mSoftPairContactTaskPoolMutex);
			PX_ASSERT(index < mSoftPairContactFinishTasks.size());
			mFreeSoftPairContactFinishTaskIndices.pushBack(index);
		}

		bool submitStandaloneSoftPairContactTask(
			PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			const PxU32 taskCount = getSoftPairContactTaskCount(dispatcherWorkers);
			const PxU32 planCount =
				mWorkspace.contact.softPairDetectionPlan.size();
			if(!mSoftPairContactTransactionPending || !continuation ||
				taskCount == 0 || !ensureSoftPairContactTaskPool(taskCount, owner))
				return false;
			{
				PxMutex::ScopedLock lock(mSoftPairContactTaskPoolMutex);
				if(mFreeSoftPairContactTaskIndices.size() < taskCount ||
					mFreeSoftPairContactFinishTaskIndices.empty())
					return false;
			}
			const PxU32 plansPerTask = (planCount + taskCount - 1) / taskCount;
			mSoftPairContactTaskOutputs.resize(taskCount);
			mSoftPairContactTaskStats.resize(taskCount);
			PxU64 totalContactCapacity = 0;
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 planBegin = taskIndex * plansPerTask;
				const PxU32 planEnd = PxMin(planBegin + plansPerTask, planCount);
				PxU64 outputCapacity = 0;
				for(PxU32 planIndex = planBegin; planIndex < planEnd; ++planIndex)
				{
					const Dy::AvbdSoftPairDetectionPlan& plan =
						mWorkspace.contact.softPairDetectionPlan[planIndex];
					const Dy::AvbdSoftBody& bodyA = mBodies[plan.bodyA];
					const Dy::AvbdSoftBody& bodyB = mBodies[plan.bodyB];
					outputCapacity += PxU64(bodyA.compiled.surfaceVertices.size()) +
						PxU64(bodyB.compiled.surfaceVertices.size()) +
						PxU64(bodyA.compiled.surfaceEdges.size()) *
							PxU64(bodyB.compiled.surfaceEdges.size());
				}
				if(outputCapacity > PX_MAX_U32 ||
					totalContactCapacity > PX_MAX_U32 - outputCapacity)
					return false;
				totalContactCapacity += outputCapacity;
				mSoftPairContactTaskOutputs[taskIndex].clear();
				mSoftPairContactTaskOutputs[taskIndex].reserve(
					PxU32(outputCapacity));
				mSoftPairContactTaskStats[taskIndex] =
					Dy::AvbdSoftCollisionStats();
			}
			mContacts.reserve(PxU32(totalContactCapacity));
			SoftPairContactFinishTask* const finishTask =
				acquireSoftPairContactFinishTask();
			if(!finishTask)
				return false;
			finishTask->setContinuation(continuation);
			taskGraphContext.recordSoftPairContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordSoftPairContactTasksSubmitted(taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 planBegin = taskIndex * plansPerTask;
				const PxU32 planEnd = PxMin(planBegin + plansPerTask, planCount);
				SoftPairContactTask* const task = acquireSoftPairContactTask();
				if(!task)
				{
					recycleSoftPairContactFinishTask(finishTask->getPoolIndex());
					return false;
				}
				PxU32 maxEdgesA = 0;
				PxU32 maxEdgesB = 0;
				PxU32 maxTriangleCandidates = 0;
				for(PxU32 planIndex = planBegin; planIndex < planEnd; ++planIndex)
				{
					const Dy::AvbdSoftPairDetectionPlan& plan =
						mWorkspace.contact.softPairDetectionPlan[planIndex];
					const Dy::AvbdSoftBody& bodyA = mBodies[plan.bodyA];
					const Dy::AvbdSoftBody& bodyB = mBodies[plan.bodyB];
					maxEdgesA = PxMax(maxEdgesA, bodyA.compiled.surfaceEdges.size());
					maxEdgesB = PxMax(maxEdgesB, bodyB.compiled.surfaceEdges.size());
					maxTriangleCandidates = PxMax(maxTriangleCandidates,
						PxMax(bodyA.compiled.surfaceTriangles.size() / 3,
							bodyB.compiled.surfaceTriangles.size() / 3));
				}
				task->reserveQueryScratch(
					maxEdgesA, maxEdgesB, maxTriangleCandidates);
				task->configure(mParticles.begin(), mParticles.size(),
					mBodies.begin(), mBodies.size(), mWorkspace.contact,
					planBegin, planEnd, mSoftPairContactTaskOutputs[taskIndex],
					mContactParams, mSoftPairContactUseSurfaceTriangleBvh,
					mCollisionStatsEnabled ? &mSoftPairContactTaskStats[taskIndex] : NULL,
					taskGraphContext);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			finishTask->removeReference();
			return true;
		}

		bool completeStandaloneSoftPairContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mSoftPairContactTransactionPending)
				return false;
			completeSoftPairContactTaskTransaction();
			taskGraphContext.recordSoftPairContactFanIn();
			if(!mStandaloneComponentStepState.completePendingRedetection())
				return false;
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				advanceStandaloneComponentStateWithSceneRedetection(
					true, &nextWorldPlaneContactTaskReady,
					true, &nextRigidBoxSdfContactTaskReady,
					true, &nextRigidSphereSdfContactTaskReady);
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
				return true;
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		bool finishStandaloneSoftPairContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(!mSoftPairContactTransactionPending)
				return false;
			taskGraphContext.recordSerialSoftPairContactFallback();
			mStandaloneTaskGraphTelemetry.recordSerialSoftPairContactFallback();
			mSoftPairContactTaskOutputs.clear();
			mSoftPairContactTaskStats.clear();
			Dy::avbdDetectSoftSoftOGCPlanRange(
				mParticles.begin(), mParticles.size(), mBodies.begin(),
				mBodies.size(), mWorkspace.contact, &mWorkspace.contact,
				mWorkspace.contact.softPairQueryScratch,
				mSoftPairContactUseSurfaceTriangleBvh, 0,
				mWorkspace.contact.softPairDetectionPlan.size(), mContacts,
				mContactParams,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			return completeStandaloneSoftPairContactTask(
				dt, taskGraphContext, nextLayerReady,
				nextWorldPlaneContactTaskReady,
				nextRigidBoxSdfContactTaskReady,
				nextRigidSphereSdfContactTaskReady);
		}

		bool canUseSelfBvhContactTaskTransaction() const
		{
			if(!Dy::avbdUseSceneRedetectionBridge() ||
				!Dy::avbdUseSelfBvhContactTaskFanIn() ||
				mBodies.size() != 1 || mSelfCollisionEnabled.size() != 1 ||
				mSelfCollisionAdjacencies.size() != 1 ||
				!mSelfCollisionEnabled[0] || !mWorldPlanes.empty() ||
				!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
				!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty())
				return false;
			return Dy::avbdCanUseSelfCollisionOGCBvhRanges(mBodies[0]);
		}

		PxU32 getSelfBvhContactTaskCount(PxU32 dispatcherWorkers) const
		{
			if(!canUseSelfBvhContactTaskTransaction() || dispatcherWorkers < 2)
				return 0;
			const Dy::AvbdSoftBody& body = mBodies[0];
			const PxU32 vertexTasks = body.compiled.surfaceVertices.empty() ? 0 :
				PxMin(dispatcherWorkers, body.compiled.surfaceVertices.size());
			const PxU32 edgeTasks = body.compiled.surfaceEdges.empty() ? 0 :
				PxMin(dispatcherWorkers, body.compiled.surfaceEdges.size());
			const PxU32 taskCount = vertexTasks + edgeTasks;
			if(taskCount < 2 && !Dy::avbdForceSelfBvhContactTaskFanIn())
				return 0;
			return taskCount;
		}

		bool beginSelfBvhContactTaskTransaction()
		{
			if(!canUseSelfBvhContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending ||
				mRigidBoxSdfContactTransactionPending ||
				mRigidSphereSdfContactTransactionPending ||
				mRigidCapsuleSdfContactTransactionPending ||
				mRigidConvexSdfContactTransactionPending ||
				mRigidTriangleSurfaceContactTransactionPending ||
				mSoftPairContactTransactionPending ||
				mSelfBvhContactTransactionPending)
				return false;
			Dy::avbdBeginSoftContactRedetection(
				mContacts, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, 0, false, 0, 0, 0, 0, 0,
				mBodies.size(), mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(), mSelfCollisionEnabled.begin());
			mSelfBvhContactBodyIndex = 0;
			const bool prepared = Dy::avbdPrepareSelfCollisionOGCBvhRanges(
				mParticles.begin(), mBodies[0], mSelfBvhContactBodyIndex,
				mSelfCollisionAdjacencies[0], mContactParams,
				mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			PX_ASSERT(prepared);
			if(!prepared)
				return false;
			mSelfBvhContactTransactionPending = true;
			return true;
		}

		void completeSelfBvhContactTaskTransaction()
		{
			PX_ASSERT(mSelfBvhContactTransactionPending);
			const PxU32 selfStart = mContacts.size();
			for(PxU32 taskIndex = 0;
				taskIndex < mSelfBvhContactTaskOutputs.size(); ++taskIndex)
			{
				const PxArray<Dy::AvbdSoftContact>& source =
					mSelfBvhContactTaskOutputs[taskIndex];
				for(PxU32 contactIndex = 0; contactIndex < source.size();
					++contactIndex)
					mContacts.pushBack(source[contactIndex]);
				if(mCollisionStatsEnabled && taskIndex <
					mSelfBvhContactTaskStats.size())
					mLastCollisionStats.accumulate(
						mSelfBvhContactTaskStats[taskIndex]);
			}
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedSelfContacts +=
					mContacts.size() - selfStart;
			Dy::avbdCompleteSoftContactRedetection(
				mParticles.begin(), mContacts, mWorkspace.contact);
			removeRigidActorFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			removeDeformablePairFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			writeContactSetTrace(mContacts);
			mSelfBvhContactTransactionPending = false;
			mSelfBvhContactBodyIndex = PX_MAX_U32;
		}

		bool ensureSelfBvhContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner)
		{
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
			mSelfBvhContactTasks.reserve(requiredSlots);
			mSelfBvhContactFinishTasks.reserve(requiredSlots);
			mFreeSelfBvhContactTaskIndices.reserve(requiredSlots);
			mFreeSelfBvhContactFinishTaskIndices.reserve(requiredSlots);
			mSelfBvhContactTaskOutputs.reserve(requiredChildTasks);
			mSelfBvhContactTaskStats.reserve(requiredChildTasks);
			while(mSelfBvhContactTasks.size() < requiredSlots)
			{
				const PxU32 index = mSelfBvhContactTasks.size();
				mSelfBvhContactTasks.pushBack(PX_NEW(SelfBvhContactTask)(
					mContextId, *this, index));
				mFreeSelfBvhContactTaskIndices.pushBack(index);
			}
			while(mSelfBvhContactFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mSelfBvhContactFinishTasks.size();
				mSelfBvhContactFinishTasks.pushBack(
					PX_NEW(SelfBvhContactFinishTask)(
						mContextId, *this, owner, index));
				mFreeSelfBvhContactFinishTaskIndices.pushBack(index);
			}
			return true;
		}

		SelfBvhContactTask* acquireSelfBvhContactTask()
		{
			PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
			if(mFreeSelfBvhContactTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeSelfBvhContactTaskIndices.back();
			mFreeSelfBvhContactTaskIndices.popBack();
			return mSelfBvhContactTasks[index];
		}

		SelfBvhContactFinishTask* acquireSelfBvhContactFinishTask()
		{
			PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
			if(mFreeSelfBvhContactFinishTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeSelfBvhContactFinishTaskIndices.back();
			mFreeSelfBvhContactFinishTaskIndices.popBack();
			return mSelfBvhContactFinishTasks[index];
		}

		void recycleSelfBvhContactTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
			PX_ASSERT(index < mSelfBvhContactTasks.size());
			mFreeSelfBvhContactTaskIndices.pushBack(index);
		}

		void recycleSelfBvhContactFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
			PX_ASSERT(index < mSelfBvhContactFinishTasks.size());
			mFreeSelfBvhContactFinishTaskIndices.pushBack(index);
		}

		bool submitStandaloneSelfBvhContactTask(
			PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			const PxU32 taskCount = getSelfBvhContactTaskCount(dispatcherWorkers);
			if(!mSelfBvhContactTransactionPending || !continuation ||
				taskCount == 0 || !ensureSelfBvhContactTaskPool(taskCount, owner))
				return false;
			{
				PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
				if(mFreeSelfBvhContactTaskIndices.size() < taskCount ||
					mFreeSelfBvhContactFinishTaskIndices.empty())
					return false;
			}
			const Dy::AvbdSoftBody& body = mBodies[mSelfBvhContactBodyIndex];
			const PxU32 vertexTaskCount = body.compiled.surfaceVertices.empty() ? 0 :
				PxMin(dispatcherWorkers, body.compiled.surfaceVertices.size());
			const PxU32 edgeTaskCount = body.compiled.surfaceEdges.empty() ? 0 :
				PxMin(dispatcherWorkers, body.compiled.surfaceEdges.size());
			mSelfBvhContactTaskOutputs.resize(taskCount);
			mSelfBvhContactTaskStats.resize(taskCount);
			PxU64 totalContactCapacity = 0;
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const bool isVertexPhase = taskIndex < vertexTaskCount;
				const PxU32 phaseTaskIndex = isVertexPhase ? taskIndex :
					taskIndex - vertexTaskCount;
				const PxU32 phaseTaskCount = isVertexPhase ? vertexTaskCount :
					edgeTaskCount;
				const PxU32 itemCount = isVertexPhase ?
					body.compiled.surfaceVertices.size() :
					body.compiled.surfaceEdges.size();
				const PxU32 itemsPerTask =
					(itemCount + phaseTaskCount - 1) / phaseTaskCount;
				const PxU32 itemBegin = phaseTaskIndex * itemsPerTask;
				const PxU32 itemEnd = PxMin(itemBegin + itemsPerTask, itemCount);
				const PxU64 outputCapacity = isVertexPhase
					? PxU64(itemEnd - itemBegin) *
						PxMax(body.compiled.surfaceTriangles.size() / 3, 1u)
					: PxU64(itemEnd - itemBegin) *
						PxMax(body.compiled.surfaceEdges.size(), 1u);
				if(outputCapacity > PX_MAX_U32 ||
					totalContactCapacity > PX_MAX_U32 - outputCapacity)
					return false;
				totalContactCapacity += outputCapacity;
				mSelfBvhContactTaskOutputs[taskIndex].clear();
				mSelfBvhContactTaskOutputs[taskIndex].reserve(PxU32(outputCapacity));
				mSelfBvhContactTaskStats[taskIndex] =
					Dy::AvbdSoftCollisionStats();
			}
			mContacts.reserve(PxU32(totalContactCapacity));
			SelfBvhContactFinishTask* const finishTask =
				acquireSelfBvhContactFinishTask();
			if(!finishTask)
				return false;
			finishTask->setContinuation(continuation);
			taskGraphContext.recordSelfBvhContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordSelfBvhContactTasksSubmitted(taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const bool isVertexPhase = taskIndex < vertexTaskCount;
				const PxU32 phaseTaskIndex = isVertexPhase ? taskIndex :
					taskIndex - vertexTaskCount;
				const PxU32 phaseTaskCount = isVertexPhase ? vertexTaskCount :
					edgeTaskCount;
				const PxU32 itemCount = isVertexPhase ?
					body.compiled.surfaceVertices.size() :
					body.compiled.surfaceEdges.size();
				const PxU32 itemsPerTask =
					(itemCount + phaseTaskCount - 1) / phaseTaskCount;
				const PxU32 itemBegin = phaseTaskIndex * itemsPerTask;
				const PxU32 itemEnd = PxMin(itemBegin + itemsPerTask, itemCount);
				SelfBvhContactTask* const task = acquireSelfBvhContactTask();
				if(!task)
				{
					recycleSelfBvhContactFinishTask(finishTask->getPoolIndex());
					return false;
				}
				task->reserveQueryScratch(body);
				task->configure(mParticles.begin(), body, mSelfBvhContactBodyIndex,
					mSelfCollisionAdjacencies[mSelfBvhContactBodyIndex],
					mWorkspace.contact,
					isVertexPhase ? itemBegin : 0,
					isVertexPhase ? itemEnd : 0,
					isVertexPhase ? 0 : itemBegin,
					isVertexPhase ? 0 : itemEnd,
					mSelfBvhContactTaskOutputs[taskIndex], mContactParams,
					mCollisionStatsEnabled ? &mSelfBvhContactTaskStats[taskIndex] : NULL,
					taskGraphContext);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			finishTask->removeReference();
			return true;
		}

		bool completeStandaloneSelfBvhContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mSelfBvhContactTransactionPending)
				return false;
			completeSelfBvhContactTaskTransaction();
			taskGraphContext.recordSelfBvhContactFanIn();
			if(!mStandaloneComponentStepState.completePendingRedetection())
				return false;
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				advanceStandaloneComponentStateWithSceneRedetection(
					true, &nextWorldPlaneContactTaskReady,
					true, &nextRigidBoxSdfContactTaskReady,
					true, &nextRigidSphereSdfContactTaskReady);
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
				return true;
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		bool finishStandaloneSelfBvhContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(!mSelfBvhContactTransactionPending)
				return false;
			taskGraphContext.recordSerialSelfBvhContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialSelfBvhContactFallback();
			mSelfBvhContactTaskOutputs.clear();
			mSelfBvhContactTaskStats.clear();
			const Dy::AvbdSoftBody& body = mBodies[mSelfBvhContactBodyIndex];
			Dy::avbdDetectSelfCollisionOGCBvhRange(
				mParticles.begin(), body, mSelfBvhContactBodyIndex,
				mSelfCollisionAdjacencies[mSelfBvhContactBodyIndex],
				mWorkspace.contact, mSelfBvhSerialRangeWorkspace,
				0, body.compiled.surfaceVertices.size(),
				0, body.compiled.surfaceEdges.size(), mContacts, mContactParams,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			return completeStandaloneSelfBvhContactTask(
				dt, taskGraphContext, nextLayerReady,
				nextWorldPlaneContactTaskReady,
				nextRigidBoxSdfContactTaskReady,
				nextRigidSphereSdfContactTaskReady);
		}

		bool canUseStaticWorldSelfOgcContactTaskTransaction() const
		{
			// This transaction changes the Scene redetection owner, so it must
			// only be admitted when it can actually submit at least two disjoint
			// workers.  Otherwise the ordinary serial callback remains both
			// cheaper and the authoritative fallback.
			if(!useAvbdStaticWorldSelfOgcTaskFanIn() ||
				mStandaloneTaskGraphEnhancedDeterminism ||
				mStandaloneTaskGraphDispatcherWorkers < 2 ||
				mParticles.size() <
					2u * eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK ||
				mBodies.size() != 1 ||
				mWorldPlanes.empty() || mRigidBoxes.empty() ||
				!mRigidSpheres.empty() || !mRigidCapsules.empty() ||
				!mRigidConvexes.empty() || !mRigidTriangleSurfaces.empty() ||
				mSelfCollisionEnabled.size() != 1 ||
				mSelfCollisionAdjacencies.size() != 1 ||
				!mSelfCollisionEnabled[0])
				return false;
			for(PxU32 boxIndex = 0; boxIndex < mRigidBoxes.size(); ++boxIndex)
			{
				if(mRigidBoxes[boxIndex].targetKind !=
					Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
					return false;
			}
			const Dy::AvbdSoftBody& body = mBodies[0];
			return body.compiled.surfaceVertices.size() >=
					2u * eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK &&
				body.compiled.surfaceEdges.size() >=
					2u * eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK &&
				Dy::avbdCanUseSelfCollisionOGCBvhRanges(body);
		}

		// Keep the ownership decision stable for every outer epoch of the
		// resumable solve.  The aggregate is a Scene-owned redetection bridge in
		// exactly the same sense as the legacy P5 leaves; testing only the raw
		// environment bridge here would run its first epoch in parallel and then
		// silently fall back to the synchronous callback.
		bool usesStandaloneSceneRedetectionBridge() const
		{
			return hasDirectSimulationCollisionDomain() &&
				(Dy::avbdUseSceneRedetectionBridge() ||
					canUseStaticWorldSelfOgcContactTaskTransaction());
		}

		PxU32 getStaticWorldSelfOgcContactTaskCount(
			PxU32 dispatcherWorkers) const
		{
			if(!canUseStaticWorldSelfOgcContactTaskTransaction() ||
				dispatcherWorkers < 2)
				return 0;
			const Dy::AvbdSoftBody& body = mBodies[0];
			// Every aggregate child owns five private contact streams and two
			// self-BVH ranges.  Cap fan-out by useful work rather than blindly
			// mirroring a high-core dispatcher; range order and the canonical
			// merge stay unchanged.
			const PxU32 maximumTasks = PxMin(
				mParticles.size() /
					eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK,
				PxMin(body.compiled.surfaceVertices.size() /
						eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK,
					body.compiled.surfaceEdges.size() /
						eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK));
			return maximumTasks < 2 ? 0 :
				PxMin(dispatcherWorkers, maximumTasks);
		}

		bool beginStaticWorldSelfOgcContactTaskTransaction()
		{
			if(!canUseStaticWorldSelfOgcContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending ||
				mRigidBoxSdfContactTransactionPending ||
				mRigidSphereSdfContactTransactionPending ||
				mRigidCapsuleSdfContactTransactionPending ||
				mRigidConvexSdfContactTransactionPending ||
				mRigidTriangleSurfaceContactTransactionPending ||
				mSoftPairContactTransactionPending ||
				mSelfBvhContactTransactionPending ||
				mStaticWorldSelfOgcContactTransactionPending)
				return false;

			Dy::avbdBeginSoftContactRedetection(
				mContacts, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, mWorldPlanes.size(), false,
				mRigidBoxes.size(), 0, 0, 0, 0, mBodies.size(),
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(), mSelfCollisionEnabled.begin());
			const bool prepared = Dy::avbdPrepareSelfCollisionOGCBvhRanges(
				mParticles.begin(), mBodies[0], 0, mSelfCollisionAdjacencies[0],
				mContactParams, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			PX_ASSERT(prepared);
			if(!prepared)
			{
				mContacts.assign(mWorkspace.contact.previousContacts.begin(),
					mWorkspace.contact.previousContacts.end());
				return false;
			}
			mStaticWorldSelfOgcContactTransactionPending = true;
			return true;
		}

		void completeStaticWorldSelfOgcContactTaskTransaction()
		{
			PX_ASSERT(mStaticWorldSelfOgcContactTransactionPending);
			auto appendOutputs = [this](
				const PxArray<PxArray<Dy::AvbdSoftContact> >& outputs)
			{
				for(PxU32 taskIndex = 0; taskIndex < outputs.size(); ++taskIndex)
				{
					const PxArray<Dy::AvbdSoftContact>& source =
						outputs[taskIndex];
					for(PxU32 contactIndex = 0;
						contactIndex < source.size(); ++contactIndex)
						mContacts.pushBack(source[contactIndex]);
				}
			};

			const PxU32 groundStart = mContacts.size();
			appendOutputs(mStaticWorldSelfOgcWorldTaskOutputs);
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedGroundContacts +=
					mContacts.size() - groundStart;

			const PxU32 rigidStart = mContacts.size();
			appendOutputs(mStaticWorldSelfOgcBoxTaskOutputs);
			appendOutputs(mStaticWorldSelfOgcBoxSweptTaskOutputs);
			// Feature OGC is intentionally parent-serial: the source stream now
			// has exactly the legacy all-current then all-swept prefix.
			Dy::avbdDetectSoftRigidOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidBoxes.begin(), mRigidBoxes.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius);
			if(mCollisionStatsEnabled)
			{
				mLastCollisionStats.rigidParticleBoxTests +=
					PxU64(mParticles.size()) * mRigidBoxes.size();
				mLastCollisionStats.generatedRigidContacts +=
					mContacts.size() - rigidStart;
			}

			const PxU32 selfStart = mContacts.size();
			appendOutputs(mStaticWorldSelfOgcSelfVertexTaskOutputs);
			appendOutputs(mStaticWorldSelfOgcSelfEdgeTaskOutputs);
			if(mCollisionStatsEnabled)
			{
				for(PxU32 taskIndex = 0;
					taskIndex < mStaticWorldSelfOgcTaskStats.size(); ++taskIndex)
					mLastCollisionStats.accumulate(
						mStaticWorldSelfOgcTaskStats[taskIndex]);
				mLastCollisionStats.generatedSelfContacts +=
					mContacts.size() - selfStart;
			}

			Dy::avbdCompleteSoftContactRedetection(
				mParticles.begin(), mContacts, mWorkspace.contact);
			removeRigidActorFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			removeDeformablePairFilteredContacts(
				mBodies.begin(), mBodies.size(), NULL, mContacts);
			writeContactSetTrace(mContacts);
			mStaticWorldSelfOgcContactTransactionPending = false;
		}

		bool ensureStaticWorldSelfOgcContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner)
		{
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(
				mStaticWorldSelfOgcContactFinishTaskPoolMutex);
			mStaticWorldSelfOgcContactTasks.reserve(requiredSlots);
			mStaticWorldSelfOgcContactFinishTasks.reserve(requiredSlots);
			mFreeStaticWorldSelfOgcContactTaskIndices.reserve(requiredSlots);
			mFreeStaticWorldSelfOgcContactFinishTaskIndices.reserve(requiredSlots);
			mStaticWorldSelfOgcWorldTaskOutputs.reserve(requiredChildTasks);
			mStaticWorldSelfOgcBoxTaskOutputs.reserve(requiredChildTasks);
			mStaticWorldSelfOgcBoxSweptTaskOutputs.reserve(requiredChildTasks);
			mStaticWorldSelfOgcSelfVertexTaskOutputs.reserve(requiredChildTasks);
			mStaticWorldSelfOgcSelfEdgeTaskOutputs.reserve(requiredChildTasks);
			mStaticWorldSelfOgcTaskStats.reserve(requiredChildTasks);
			while(mStaticWorldSelfOgcContactTasks.size() < requiredSlots)
			{
				const PxU32 index = mStaticWorldSelfOgcContactTasks.size();
				mStaticWorldSelfOgcContactTasks.pushBack(
					PX_NEW(StaticWorldSelfOgcContactTask)(
						mContextId, *this, index));
				mFreeStaticWorldSelfOgcContactTaskIndices.pushBack(index);
			}
			while(mStaticWorldSelfOgcContactFinishTasks.size() < requiredSlots)
			{
				const PxU32 index =
					mStaticWorldSelfOgcContactFinishTasks.size();
				mStaticWorldSelfOgcContactFinishTasks.pushBack(
					PX_NEW(StaticWorldSelfOgcContactFinishTask)(
						mContextId, *this, owner, index));
				mFreeStaticWorldSelfOgcContactFinishTaskIndices.pushBack(index);
			}
			return true;
		}

		StaticWorldSelfOgcContactTask* acquireStaticWorldSelfOgcContactTask()
		{
			PxMutex::ScopedLock lock(
				mStaticWorldSelfOgcContactFinishTaskPoolMutex);
			if(mFreeStaticWorldSelfOgcContactTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeStaticWorldSelfOgcContactTaskIndices.back();
			mFreeStaticWorldSelfOgcContactTaskIndices.popBack();
			return mStaticWorldSelfOgcContactTasks[index];
		}

		StaticWorldSelfOgcContactFinishTask*
			acquireStaticWorldSelfOgcContactFinishTask()
		{
			PxMutex::ScopedLock lock(
				mStaticWorldSelfOgcContactFinishTaskPoolMutex);
			if(mFreeStaticWorldSelfOgcContactFinishTaskIndices.empty())
				return NULL;
			const PxU32 index =
				mFreeStaticWorldSelfOgcContactFinishTaskIndices.back();
			mFreeStaticWorldSelfOgcContactFinishTaskIndices.popBack();
			return mStaticWorldSelfOgcContactFinishTasks[index];
		}

		void recycleStaticWorldSelfOgcContactTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(
				mStaticWorldSelfOgcContactFinishTaskPoolMutex);
			PX_ASSERT(index < mStaticWorldSelfOgcContactTasks.size());
			mFreeStaticWorldSelfOgcContactTaskIndices.pushBack(index);
		}

		void recycleStaticWorldSelfOgcContactFinishTask(PxU32 index)
		{
			PxMutex::ScopedLock lock(
				mStaticWorldSelfOgcContactFinishTaskPoolMutex);
			PX_ASSERT(index < mStaticWorldSelfOgcContactFinishTasks.size());
			mFreeStaticWorldSelfOgcContactFinishTaskIndices.pushBack(index);
		}

		bool submitStandaloneStaticWorldSelfOgcContactTask(
			PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			const PxU32 taskCount = getStaticWorldSelfOgcContactTaskCount(
				dispatcherWorkers);
			if(!mStaticWorldSelfOgcContactTransactionPending || !continuation ||
				taskCount == 0 ||
				!ensureStaticWorldSelfOgcContactTaskPool(taskCount, owner))
				return false;
			{
				PxMutex::ScopedLock lock(
					mStaticWorldSelfOgcContactFinishTaskPoolMutex);
				if(mFreeStaticWorldSelfOgcContactTaskIndices.size() < taskCount ||
					mFreeStaticWorldSelfOgcContactFinishTaskIndices.empty())
					return false;
			}

			const Dy::AvbdSoftBody& body = mBodies[0];
			mStaticWorldSelfOgcWorldTaskOutputs.resize(taskCount);
			mStaticWorldSelfOgcBoxTaskOutputs.resize(taskCount);
			mStaticWorldSelfOgcBoxSweptTaskOutputs.resize(taskCount);
			mStaticWorldSelfOgcSelfVertexTaskOutputs.resize(taskCount);
			mStaticWorldSelfOgcSelfEdgeTaskOutputs.resize(taskCount);
			mStaticWorldSelfOgcTaskStats.resize(taskCount);
			const PxU32 particlesPerTask =
				(mParticles.size() + taskCount - 1) / taskCount;
			const PxU32 verticesPerTask =
				(body.compiled.surfaceVertices.size() + taskCount - 1) /
				taskCount;
			const PxU32 edgesPerTask =
				(body.compiled.surfaceEdges.size() + taskCount - 1) / taskCount;
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				PxArray<Dy::AvbdSoftContact>& worldOutput =
					mStaticWorldSelfOgcWorldTaskOutputs[taskIndex];
				worldOutput.clear();
				worldOutput.reserve((particleEnd - particleBegin) *
					mWorldPlanes.size());
				PxArray<Dy::AvbdSoftContact>& boxOutput =
					mStaticWorldSelfOgcBoxTaskOutputs[taskIndex];
				boxOutput.clear();
				boxOutput.reserve((particleEnd - particleBegin) *
					mRigidBoxes.size());
				PxArray<Dy::AvbdSoftContact>& boxSweptOutput =
					mStaticWorldSelfOgcBoxSweptTaskOutputs[taskIndex];
				boxSweptOutput.clear();
				boxSweptOutput.reserve((particleEnd - particleBegin) *
					mRigidBoxes.size());
				mStaticWorldSelfOgcSelfVertexTaskOutputs[taskIndex].clear();
				mStaticWorldSelfOgcSelfEdgeTaskOutputs[taskIndex].clear();
				mStaticWorldSelfOgcTaskStats[taskIndex] =
					Dy::AvbdSoftCollisionStats();
			}

			StaticWorldSelfOgcContactFinishTask* const finishTask =
				acquireStaticWorldSelfOgcContactFinishTask();
			if(!finishTask)
				return false;
			finishTask->setContinuation(continuation);
			// Attribute the aggregate to each physical OGC source.  The child
			// ownership is shared, but source telemetry must not report a zero
			// collision stage merely because the canonical parent is unified.
			taskGraphContext.recordWorldPlaneContactTasksSubmitted(taskCount);
			taskGraphContext.recordRigidBoxSdfContactTasksSubmitted(taskCount);
			taskGraphContext.recordSelfBvhContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordWorldPlaneContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordRigidBoxSdfContactTasksSubmitted(taskCount);
			mStandaloneTaskGraphTelemetry.
				recordSelfBvhContactTasksSubmitted(taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
			{
				const PxU32 particleBegin = taskIndex * particlesPerTask;
				const PxU32 particleEnd = PxMin(
					particleBegin + particlesPerTask, mParticles.size());
				const PxU32 vertexBegin = taskIndex * verticesPerTask;
				const PxU32 vertexEnd = PxMin(
					vertexBegin + verticesPerTask,
					body.compiled.surfaceVertices.size());
				const PxU32 edgeBegin = taskIndex * edgesPerTask;
				const PxU32 edgeEnd = PxMin(edgeBegin + edgesPerTask,
					body.compiled.surfaceEdges.size());
				StaticWorldSelfOgcContactTask* const task =
					acquireStaticWorldSelfOgcContactTask();
				if(!task)
				{
					recycleStaticWorldSelfOgcContactFinishTask(
						finishTask->getPoolIndex());
					return false;
				}
				task->reserveQueryScratch(body);
				task->configure(mParticles.begin(), mParticles.size(),
					particleBegin, particleEnd, mWorldPlanes.begin(),
					mWorldPlanes.size(), mRigidBoxes.begin(), mRigidBoxes.size(),
					mWorkspace.contact.previousContacts.begin(),
					mWorkspace.contact.previousContacts.size(), body,
					mSelfCollisionAdjacencies[0], mWorkspace.contact,
					vertexBegin, vertexEnd, edgeBegin, edgeEnd,
					mStaticWorldSelfOgcWorldTaskOutputs[taskIndex],
					mStaticWorldSelfOgcBoxTaskOutputs[taskIndex],
					mStaticWorldSelfOgcBoxSweptTaskOutputs[taskIndex],
					mStaticWorldSelfOgcSelfVertexTaskOutputs[taskIndex],
					mStaticWorldSelfOgcSelfEdgeTaskOutputs[taskIndex],
					mContactParams, mCollisionStatsEnabled ?
						&mStaticWorldSelfOgcTaskStats[taskIndex] : NULL,
					taskGraphContext,
					mContactParams.contactRadius);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			finishTask->removeReference();
			return true;
		}

		bool completeStandaloneStaticWorldSelfOgcContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mStaticWorldSelfOgcContactTransactionPending)
				return false;
			completeStaticWorldSelfOgcContactTaskTransaction();
			taskGraphContext.recordWorldPlaneContactFanIn();
			taskGraphContext.recordRigidBoxSdfContactFanIn();
			taskGraphContext.recordSelfBvhContactFanIn();
			if(!mStandaloneComponentStepState.completePendingRedetection())
				return false;
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				advanceStandaloneComponentStateWithSceneRedetection(
					true, &nextWorldPlaneContactTaskReady,
					true, &nextRigidBoxSdfContactTaskReady,
					true, &nextRigidSphereSdfContactTaskReady);
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
				return true;
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		bool finishStandaloneStaticWorldSelfOgcContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			if(!mStaticWorldSelfOgcContactTransactionPending)
				return false;
			taskGraphContext.recordSerialWorldPlaneContactFallback();
			taskGraphContext.recordSerialRigidBoxSdfContactFallback();
			taskGraphContext.recordSerialSelfBvhContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialWorldPlaneContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialRigidBoxSdfContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialSelfBvhContactFallback();
			const Dy::AvbdSoftBody& body = mBodies[0];
			mStaticWorldSelfOgcWorldTaskOutputs.resize(1);
			mStaticWorldSelfOgcBoxTaskOutputs.resize(1);
			mStaticWorldSelfOgcBoxSweptTaskOutputs.resize(1);
			mStaticWorldSelfOgcSelfVertexTaskOutputs.resize(1);
			mStaticWorldSelfOgcSelfEdgeTaskOutputs.resize(1);
			mStaticWorldSelfOgcTaskStats.resize(1);
			mStaticWorldSelfOgcWorldTaskOutputs[0].clear();
			mStaticWorldSelfOgcBoxTaskOutputs[0].clear();
			mStaticWorldSelfOgcBoxSweptTaskOutputs[0].clear();
			mStaticWorldSelfOgcSelfVertexTaskOutputs[0].clear();
			mStaticWorldSelfOgcSelfEdgeTaskOutputs[0].clear();
			mStaticWorldSelfOgcTaskStats[0] = Dy::AvbdSoftCollisionStats();
			Dy::avbdDetectSoftWorldPlaneContactsRange(
				mParticles.begin(), mParticles.size(), 0, mParticles.size(),
				mWorldPlanes.begin(), mWorldPlanes.size(),
				mStaticWorldSelfOgcWorldTaskOutputs[0],
				mContactParams.contactRadius, &body, 1);
			Dy::avbdDetectSoftRigidSDFRange(
				mParticles.begin(), mParticles.size(), 0, mParticles.size(),
				mRigidBoxes.begin(), mRigidBoxes.size(),
				mStaticWorldSelfOgcBoxTaskOutputs[0],
				mContactParams.contactRadius,
				mWorkspace.contact.previousContacts.begin(),
				mWorkspace.contact.previousContacts.size(), &body, 1);
			Dy::avbdDetectSoftRigidSweptSDFRange(
				mParticles.begin(), mParticles.size(), 0, mParticles.size(),
				mRigidBoxes.begin(), mRigidBoxes.size(),
				mStaticWorldSelfOgcBoxSweptTaskOutputs[0],
				mContactParams.contactRadius, &body, 1);
			Dy::avbdDetectSelfCollisionOGCBvhRange(
				mParticles.begin(), body, 0, mSelfCollisionAdjacencies[0],
				mWorkspace.contact, mSelfBvhSerialRangeWorkspace, 0,
				body.compiled.surfaceVertices.size(), 0, 0,
				mStaticWorldSelfOgcSelfVertexTaskOutputs[0], mContactParams,
				mCollisionStatsEnabled ? &mStaticWorldSelfOgcTaskStats[0] : NULL);
			Dy::avbdDetectSelfCollisionOGCBvhRange(
				mParticles.begin(), body, 0, mSelfCollisionAdjacencies[0],
				mWorkspace.contact, mSelfBvhSerialRangeWorkspace, 0, 0, 0,
				body.compiled.surfaceEdges.size(),
				mStaticWorldSelfOgcSelfEdgeTaskOutputs[0], mContactParams,
				mCollisionStatsEnabled ? &mStaticWorldSelfOgcTaskStats[0] : NULL);
			return completeStandaloneStaticWorldSelfOgcContactTask(
				dt, taskGraphContext, nextLayerReady,
				nextWorldPlaneContactTaskReady,
				nextRigidBoxSdfContactTaskReady,
				nextRigidSphereSdfContactTaskReady);
		}

		bool completeStandaloneCausalLayerTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady,
			bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				mCausalLayerRangeObservations.empty())
				return false;
			const PxU64 reductionStartNanos = mCollisionStatsEnabled ?
				PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
			const bool completedIndependentBodySweep =
				mStandaloneComponentStepState.
					completePublishedIndependentBodySweep(
						mCausalLayerRangeObservations.begin(),
						mCausalLayerRangeObservations.size());
			if(!completedIndependentBodySweep &&
				!mStandaloneComponentStepState.completePublishedCausalLayer(
					mCausalLayerRangeObservations.begin(),
					mCausalLayerRangeObservations.size()))
				return false;
			taskGraphContext.recordCausalLayerFanIn(
				mCollisionStatsEnabled ? PxReal(
					(PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						reductionStartNanos) * 1.0e-9) : 0.0f);
			mStandaloneTaskGraphTelemetry.recordCausalLayerFanIn();
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				usesStandaloneSceneRedetectionBridge()
					? advanceStandaloneComponentStateWithSceneRedetection(
						true, &nextWorldPlaneContactTaskReady,
						true, &nextRigidBoxSdfContactTaskReady,
						true, &nextRigidSphereSdfContactTaskReady)
					: mStandaloneComponentStepState.advance();
			if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
				nextRigidSphereSdfContactTaskReady)
				return true;
			if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			{
				nextLayerReady = true;
				return true;
			}
			if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		bool finishStandaloneCausalLayerSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext)
		{
			PxU32 layerIndex = 0;
			PxU32 packedBegin = 0;
			PxU32 packedEnd = 0;
			const Dy::AvbdParticlePrimalSolveContext* solveContext = NULL;
			const Dy::AvbdSoftBody* bodies = NULL;
			PxU32 bodyCount = 0;
			const PxU32* particleBodyIndices = NULL;
			const PxU32* packedParticleIndices = NULL;
			if(mStandaloneComponentStepState.getPublishedCausalLayer(
				layerIndex, packedBegin, packedEnd, solveContext, bodies,
				bodyCount, particleBodyIndices, packedParticleIndices) &&
				packedEnd > packedBegin)
			{
				taskGraphContext.recordSerialCausalLayerFallback(
					packedEnd - packedBegin);
				mStandaloneTaskGraphTelemetry.
					recordSerialCausalLayerFallback();
			}
			else if(mStandaloneComponentStepState.
				getPublishedIndependentBodySweep(
					solveContext, bodies, bodyCount))
			{
				PxU32 bodyParticleCount = 0;
				for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
					bodyParticleCount +=
						bodies[bodyIndex].compiled.particleCount;
				taskGraphContext.recordSerialCausalLayerFallback(
					bodyParticleCount);
				mStandaloneTaskGraphTelemetry.
					recordSerialCausalLayerFallback();
			}
			if(usesStandaloneSceneRedetectionBridge())
			{
				if(!runStandaloneComponentStateWithSceneRedetection())
					return false;
			}
			else
				mStandaloneComponentStepState.runToCompletionSerial();
			if(!mStandaloneComponentStepState.isComplete())
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		void finishStandaloneComponentSolve(PxReal dt)
		{
			mStandaloneTaskGraphTelemetry.endSolveTask();
			mStandaloneStepStats.contactWorkspaceGrowthEvents +=
				mComponentFallbackPlan.initialContactWorkspaceGrowthEvents;
			mStandaloneStepStats.contactWorkspaceGrowthBytes +=
				mComponentFallbackPlan.initialContactWorkspaceGrowthBytes;
			mStandaloneStepStats.contactSweepScratchGrowthEvents +=
				mComponentFallbackPlan.initialContactSweepScratchGrowthEvents;
			mStandaloneStepStats.contactSweepScratchGrowthBytes +=
				mComponentFallbackPlan.initialContactSweepScratchGrowthBytes;
			mStandaloneStepStats.contactOutputGrowthEvents +=
				mComponentFallbackPlan.initialContactOutputGrowthEvents;
			mStandaloneStepStats.contactOutputGrowthBytes +=
				mComponentFallbackPlan.initialContactOutputGrowthBytes;
			mLastStepStats = mStandaloneStepStats;
			++mLastComponentFallbackSteps;
			mComponentFallbackPlanPrepared = false;
			mStandaloneComponentSolvePrepared = false;
			finalizeDeformableMotionControls(dt);
			mStandaloneComponentPostSolvePending = true;
		}

		bool resumeStandaloneComponentSolve(
			PxReal dt, const PxVec3& gravity,
			bool* causalLayerTaskReady = NULL,
			bool* worldPlaneContactTaskReady = NULL,
			bool* rigidBoxSdfContactTaskReady = NULL,
			bool* rigidSphereSdfContactTaskReady = NULL)
		{
			PX_ASSERT(mStandaloneComponentSolvePrepared);
			if(!mStandaloneComponentSolvePrepared)
				return false;
			if(causalLayerTaskReady)
				*causalLayerTaskReady = false;
			if(worldPlaneContactTaskReady)
				*worldPlaneContactTaskReady = false;
			if(rigidBoxSdfContactTaskReady)
				*rigidBoxSdfContactTaskReady = false;
			if(rigidSphereSdfContactTaskReady)
				*rigidSphereSdfContactTaskReady = false;
			// All prediction children have joined before this continuation. A
			// defensive completeness check preserves the serial broadphase if a
			// future ownership path omits a body.
			mWorkspace.contact.markSoftBodyBoundsReady();
			// A relaxed color schedule is itself an explicit production fast-path
			// request.  It must enter the resumable task state even when the old
			// P4 validation switch is off; the state still falls back to scalar
			// authority if a complete conflict plan cannot be published.
			mStandaloneParticlePrimalSchedule =
				getParticlePrimalSchedule();
			const bool useCausalLayerTaskFanIn = causalLayerTaskReady &&
				(Dy::avbdUseCausalLayerTaskFanIn() ||
				 Dy::avbdUsesColoredParticlePrimalSchedule(
					mStandaloneParticlePrimalSchedule));
			const bool useIndependentBodySweepTaskFanIn =
				causalLayerTaskReady &&
				canUseIndependentBodySweepTaskFanIn();
			// P5 task leaves have not yet learned the cooked-collision embedding.
			// The relaxed component taskgraph remains safe for that public setup:
			// it keeps redetection on the existing synchronous, authoritative
			// callback while prediction/primal/write-back may still fan out.
			const bool useStaticWorldSelfOgcTaskFanIn =
				canUseStaticWorldSelfOgcContactTaskTransaction();
			const bool useSceneRedetectionBridge =
				usesStandaloneSceneRedetectionBridge();
			const bool useWorldPlaneContactTaskFanIn =
				causalLayerTaskReady && worldPlaneContactTaskReady &&
				useSceneRedetectionBridge &&
				Dy::avbdUseWorldPlaneContactTaskFanIn();
			const bool useRigidBoxSdfContactTaskFanIn =
				causalLayerTaskReady && rigidBoxSdfContactTaskReady &&
				useSceneRedetectionBridge &&
				Dy::avbdUseRigidBoxSdfContactTaskFanIn();
			const bool useRigidSphereSdfContactTaskFanIn =
				causalLayerTaskReady && rigidSphereSdfContactTaskReady &&
				useSceneRedetectionBridge &&
				(Dy::avbdUseRigidSphereSdfContactTaskFanIn() ||
				 Dy::avbdUseRigidCapsuleSdfContactTaskFanIn() ||
					 Dy::avbdUseRigidConvexSdfContactTaskFanIn() ||
					 Dy::avbdUseRigidTriangleSurfaceContactTaskFanIn() ||
					 Dy::avbdUseSoftPairContactTaskFanIn() ||
					 Dy::avbdUseSelfBvhContactTaskFanIn() ||
					 useStaticWorldSelfOgcTaskFanIn);
			// Both ordered and relaxed colored paths use the persistent state so
			// Scene owns the inter-color barriers.  The ordered schedule remains a
			// reference oracle; relaxed colors intentionally need not reproduce
			// its per-particle traversal.
			const bool useColoredPrimalState =
				Dy::avbdUsesColoredParticlePrimalSchedule(
					mStandaloneParticlePrimalSchedule);
			if(Dy::avbdUsePersistentStepStateSerial() ||
				useCausalLayerTaskFanIn ||
				useIndependentBodySweepTaskFanIn ||
				useColoredPrimalState ||
				useSceneRedetectionBridge || useWorldPlaneContactTaskFanIn ||
				useRigidBoxSdfContactTaskFanIn ||
				useRigidSphereSdfContactTaskFanIn)
			{
				// The Scene-owned state spans all parent transitions. In the
				// P4.5.2c validation route it is consumed synchronously; P4.5.3
				// instead publishes precisely its first causal layer to Scene.
				const bool begun =
					mStandaloneComponentStepState.beginAfterPrediction(
						mParticles.begin(), mParticles.size(),
						mBodies.begin(), mBodies.size(),
						mContacts.begin(), mContacts.size(),
						dt, mComponentFallbackPlan.outerIterations,
						mComponentFallbackPlan.innerIterations,
						mComponentFallbackPlan.totalPositionIterations,
						1000.0f, redetectContacts, &mContacts, this,
						0.92f, &mStandaloneStepStats, mWorkspace,
					mSelfCollisionAdjacencies.begin(),
					mSelfCollisionAdjacencies.size(),
					mSelfCollisionEnabled.begin(), &mContactParams,
					mStandaloneParticlePrimalSchedule,
					useSceneRedetectionBridge,
					useIndependentBodySweepTaskFanIn);
				PX_ASSERT(begun);
				if(!begun)
					return false;
				if(useCausalLayerTaskFanIn ||
					useIndependentBodySweepTaskFanIn ||
					useWorldPlaneContactTaskFanIn ||
					useRigidBoxSdfContactTaskFanIn ||
					useRigidSphereSdfContactTaskFanIn)
				{
					const Dy::AvbdSoftBodyStepAdvanceResult result =
						useSceneRedetectionBridge
							? advanceStandaloneComponentStateWithSceneRedetection(
								useWorldPlaneContactTaskFanIn,
								worldPlaneContactTaskReady,
								useRigidBoxSdfContactTaskFanIn,
								rigidBoxSdfContactTaskReady,
								useRigidSphereSdfContactTaskFanIn,
								rigidSphereSdfContactTaskReady)
							: mStandaloneComponentStepState.advance();
					if(useWorldPlaneContactTaskFanIn &&
						*worldPlaneContactTaskReady)
						return false;
					if(useRigidBoxSdfContactTaskFanIn &&
						*rigidBoxSdfContactTaskReady)
						return false;
					if(useRigidSphereSdfContactTaskFanIn &&
						*rigidSphereSdfContactTaskReady)
						return false;
					if(result ==
						Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
					{
						// Initial redetection has now published the authoritative
						// contact epoch. Only at this point can a dense soft/soft
						// manifold be identified reliably; keep its relaxed color
						// plan but finish the short layers inline instead of creating
						// a task/fan-in pair for every color.
						if(shouldInlineDenseSoftPairColoredPrimal())
						{
							mStandaloneComponentStepState.
								runToCompletionSerial();
							if(!mStandaloneComponentStepState.isComplete())
								return false;
						}
						else
						{
							*causalLayerTaskReady = true;
							return false;
						}
					}
					else if(result !=
						Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
						return false;
				}
				else if(useSceneRedetectionBridge)
				{
					if(!runStandaloneComponentStateWithSceneRedetection())
						return false;
				}
				else
					mStandaloneComponentStepState.runToCompletionSerial();
				PX_ASSERT(mStandaloneComponentStepState.isComplete());
				if(!mStandaloneComponentStepState.isComplete())
					return false;
			}
			else
			{
				Dy::avbdStepSoftBodies(
					mParticles.begin(), mParticles.size(),
					mBodies.begin(), mBodies.size(),
					mContacts.begin(), mContacts.size(),
					dt, gravity,
					mComponentFallbackPlan.outerIterations,
					mComponentFallbackPlan.innerIterations,
					1000.0f,
					redetectContacts, &mContacts, this,
					0.92f, &mStandaloneStepStats, &mWorkspace,
					mComponentFallbackPlan.totalPositionIterations,
					mSelfCollisionAdjacencies.begin(),
					mSelfCollisionAdjacencies.size(),
					mSelfCollisionEnabled.begin(),
					&mContactParams,
					Dy::AvbdSoftBodyStepExecutionMode::eRESUME);
			}
			finishStandaloneComponentSolve(dt);
			return true;
		}

		// Serial authority for the split component route. This preserves the
		// exact same ePREPARE -> prediction -> eRESUME ordering as the task
		// graph, while running all three stages on the caller thread.
		bool stepStandaloneComponentSolveOnly(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager,
			bool sleepingEnabled)
		{
			if(!prepareStandaloneComponentSolve(
				dt, gravity, materialManager, rigidMaterialManager,
				sleepingEnabled))
				return false;
			predictStandaloneComponentRange(
				0, mEntries.size(), dt, gravity);
			return resumeStandaloneComponentSolve(dt, gravity);
		}

		// Returns the number of fixed entry ranges that can safely be submitted
		// after stepStandaloneComponentSolveOnly().  P3 intentionally avoids
		// per-particle partitioning: independent entries are the first proven
		// no-conflict boundary, and a small component remains serial.
		PxU32 getStandaloneWriteBackTaskCount(
			PxU32 dispatcherWorkers) const
		{
			if(!mStandaloneComponentPostSolvePending ||
				mDynamicsOwnsStep || dispatcherWorkers < 2)
				return 0;

			PxU32 awakeEntryCount = 0;
			PxU64 awakeParticleCount = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				const Entry& entry = mEntries[i];
				if(entry.sleeping)
					continue;
				awakeEntryCount++;
				awakeParticleCount += getParticleCount(entry);
			}
			// The task and fan-in overhead is not useful for small copies.  Keep
			// this threshold independent from the P2 solve eligibility: it only
			// describes output bandwidth and leaves the solve semantics untouched.
			if(awakeEntryCount < 2 || awakeParticleCount < 1024)
				return 0;
			return PxMin(dispatcherWorkers, awakeEntryCount);
		}

		void submitStandaloneWriteBackTasks(
			PxU32 taskCount, PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			PX_ASSERT(continuation);
			PX_ASSERT(taskCount == getStandaloneWriteBackTaskCount(
				PxMax(taskCount, 2u)));
			PX_ASSERT(taskCount > 0 && taskCount <= mEntries.size());
			while(mWriteBackTasks.size() < taskCount)
				mWriteBackTasks.pushBack(PX_NEW(WriteBackTask)(
					mContextId, *this));

			// Ranges are stable and aligned to whole entries.  Each task writes
			// only its entries' particle/output buffers; a later fan-in owns the
			// shared sleeping/island continuation.
			const PxU32 entriesPerTask =
				(mEntries.size() + taskCount - 1) / taskCount;
			taskGraphContext.recordWriteBackTasksSubmitted(taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; taskIndex++)
			{
				const PxU32 entryBegin = taskIndex * entriesPerTask;
				const PxU32 entryEnd = PxMin(
					entryBegin + entriesPerTask, mEntries.size());
				PX_ASSERT(entryBegin < entryEnd);
				WriteBackTask& task = *mWriteBackTasks[taskIndex];
				task.configure(entryBegin, entryEnd, taskGraphContext);
				task.setContinuation(continuation);
				task.removeReference();
			}
		}

		void writeBackStandaloneComponentRange(
			PxU32 entryBegin, PxU32 entryEnd)
		{
			PX_ASSERT(mStandaloneComponentPostSolvePending);
			PX_ASSERT(entryBegin <= entryEnd && entryEnd <= mEntries.size());
			for(PxU32 entryIndex = entryBegin; entryIndex < entryEnd;
				entryIndex++)
				writeBack(mEntries[entryIndex]);
		}

		void writeBackStandaloneComponent()
		{
			writeBackStandaloneComponentRange(0, mEntries.size());
		}

		void finishStandaloneComponentStep(
			PxReal dt, bool sleepingEnabled)
		{
			PX_ASSERT(mStandaloneComponentPostSolvePending);
			if(!mStandaloneComponentPostSolvePending)
				return;
			updateSleepStates(dt, sleepingEnabled);
			mWorkspacePreflightPending = false;
			mDynamicsSelectedEntryCount = 0;
			mStandaloneComponentPostSolvePending = false;
		}

		void step(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager,
			bool sleepingEnabled)
		{
			mStandaloneComponentPostSolvePending = false;
			// These values are deliberately step-local.  The public scene
			// statistics report a completed simulation step, while the snippet
			// accumulates only its profile window and can therefore assert that
			// warm-up reached a zero-growth steady state.
			mLastStepStats.reset();
			if(mCollisionStatsEnabled)
				mLastCollisionStats = Dy::AvbdSoftCollisionStats();
			mLastComponentFallbackSteps = 0;
			mLastNativeIslandSteps = 0;
			if(dt <= 0.0f)
				return;

			if(mBodies.empty())
				return;

			PxU32 awakeEntryCount = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
				if(!mEntries[i].sleeping)
					awakeEntryCount++;
			if(awakeEntryCount == 0 && sleepingEnabled)
			{
				mDynamicsOwnsStep = false;
				mDynamicsSelectedEntryCount = 0;
				return;
			}

			if(mDynamicsOwnsStep)
			{
				mLastNativeIslandSteps = 1;
				// Selection is all-or-nothing at the preparation boundary.  Do
				// not turn an unexpected wake/sleep transition between prep and
				// post-solve into a full component fallback: that would advance
				// the already-native particles a second time.  The changed
				// membership is safely reconsidered on the next prepare boundary.
				PX_ASSERT(mDynamicsSelectedEntryCount == awakeEntryCount);
				for(PxU32 storageIndex = 0;
					storageIndex < mIslandSelectionStorages.size();
					storageIndex++)
				{
					IslandSelectionStorage& storage =
						*mIslandSelectionStorages[storageIndex];
					if(storage.touched)
						copyIslandSelectionResults(storage);
				}
				finalizeDeformableMotionControls(dt);
				for(PxU32 i = 0; i < mEntries.size(); i++)
					writeBack(mEntries[i]);
				updateSleepStates(dt, sleepingEnabled);
				mWorkspacePreflightPending = false;
				mDynamicsOwnsStep = false;
				mDynamicsSelectedEntryCount = 0;
				return;
			}

			// The component route is now the sole owner for this frame.  A native
			// selection contact stream is a separate persistent AL/friction cache;
			// retaining it across this fallback would revive a state that did not
			// participate in the immediately preceding solve.
			invalidateNativeIslandSelectionCaches();
			stepComponentFallback(
				dt, gravity, materialManager,
				rigidMaterialManager);

			finalizeDeformableMotionControls(dt);
			for(PxU32 i = 0; i < mEntries.size(); i++)
				writeBack(mEntries[i]);
			updateSleepStates(dt, sleepingEnabled);
			mWorkspacePreflightPending = false;
			mDynamicsSelectedEntryCount = 0;
		}

	private:
		PxU32 estimateInitialComponentContactCapacity() const
		{
			// Rigid primitives and soft-body peers have a source-aware linear
			// initial reserve. Self collision is deliberately excluded: its
			// density has no useful linear topology bound and must remain
			// telemetry-visible until P1 has a separately accepted policy.
			const PxU64 sourceCount =
				PxU64(mWorldPlanes.size()) +
				PxU64(mRigidBoxes.size()) +
				PxU64(mRigidSpheres.size()) +
				PxU64(mRigidCapsules.size()) +
				PxU64(mRigidConvexes.size()) +
				PxU64(mRigidTriangleSurfaces.size()) +
				(mBodies.size() > 1 ? PxU64(mBodies.size() - 1) : 0);
			if(sourceCount == 0 || mParticles.empty())
				return 0;

			// The measured ground and two-surface corpus peaks at four contact
			// slots per particle. Keep a hard 2 MiB budget for all associated
			// output/state/incidence storage; an underestimated or dynamic case
			// falls back to normal, visible growth rather than over-reserving.
			const PxU64 bytesPerContact =
				PxU64(sizeof(Dy::AvbdSoftContact)) * 2 +
				PxU64(sizeof(Dy::AvbdSoftContactParticleRef)) *
					Dy::AVBD_CONTACT_MAX_PARTICLES +
				PxU64(sizeof(Dy::AvbdCompiledSoftVelocityObjective)) +
				sizeof(PxU8);
			const PxU64 capacityBudget =
				(2ull * 1024ull * 1024ull) / bytesPerContact;
			const PxU64 requestedCapacity =
				PxU64(mParticles.size()) * 4 * sourceCount;
			return PxU32(PxMin(requestedCapacity, capacityBudget));
		}

		void reserveLifecycleContactCapacity()
		{
			const PxU32 contactCapacity =
				estimateInitialComponentContactCapacity();
			if(contactCapacity)
				mContacts.reserve(contactCapacity);
			mWorkspace.reserve(
				mParticles.size(), contactCapacity,
				getParticlePrimalSchedule());
		}

		void reserveLifecycleCollisionScratch()
		{
			// One contact workspace is shared serially by all component bodies.
			// Reserve the largest self-query topology and the two largest soft-pair
			// edge sets at actor-add time. Refitting still writes current particle
			// bounds every detection; this only removes capacity allocation from
			// that hot path.
			PxU32 maxTetCount = 0;
			PxU32 maxTriangleCount = 0;
			PxU32 maxSurfaceVertexCount = 0;
			PxU32 maxEdgeCountA = 0;
			PxU32 maxEdgeCountB = 0;
			for(PxU32 bodyIndex = 0;
				bodyIndex < mBodies.size(); bodyIndex++)
			{
				const Dy::AvbdSoftBodyCompiledData& compiled =
					mBodies[bodyIndex].compiled;
				maxTetCount = PxMax(
					maxTetCount, compiled.tetElements.size());
				maxTriangleCount = PxMax(
					maxTriangleCount,
					compiled.surfaceTriangles.size() / 3);
				maxSurfaceVertexCount = PxMax(
					maxSurfaceVertexCount,
					compiled.surfaceVertices.size());
				const PxU32 edgeCount = compiled.surfaceEdges.size();
				if(edgeCount > maxEdgeCountA)
				{
					maxEdgeCountB = maxEdgeCountA;
					maxEdgeCountA = edgeCount;
				}
				else if(edgeCount > maxEdgeCountB)
					maxEdgeCountB = edgeCount;
			}
			mWorkspace.contact.reserveSelfCollisionSweep(
				maxTetCount, maxTriangleCount,
				maxSurfaceVertexCount, maxEdgeCountA);
			mWorkspace.contact.reserveSoftPairSweep(
				maxEdgeCountA, maxEdgeCountB,
				maxTriangleCount);
		}

		void prepareComponentFallback(
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager)
		{
			PX_ASSERT(!mComponentFallbackPlanPrepared);
			mComponentFallbackPlan = ComponentFallbackPlan();
			ComponentFallbackPlan& plan = mComponentFallbackPlan;
			PxU32 requestedPositionIterations = 1;
			PxU32 requestedCollisionPairUpdates = 0;
			PxU32 requestedCollisionSubsteps = 1;
			bool hasExplicitCollisionPairUpdates = false;
			bool allEntriesAreVolumes = !mEntries.empty();
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				syncHostInputs(mEntries[i], materialManager);
				allEntriesAreVolumes = allEntriesAreVolumes &&
					mEntries[i].kind == eVOLUME;
				requestedPositionIterations = PxMax<PxU32>(
					requestedPositionIterations,
					mEntries[i].getSolverIterationCounts() & 0xff);
				if(mEntries[i].kind == eSURFACE)
				{
					const PxU32 pairUpdates =
						mEntries[i].surfaceCore->
							getNbCollisionPairUpdatesPerTimestep();
					if(pairUpdates > 0)
					{
						hasExplicitCollisionPairUpdates = true;
						requestedCollisionPairUpdates =
							PxMax(
								requestedCollisionPairUpdates,
								pairUpdates);
					}
					requestedCollisionSubsteps = PxMax(
						requestedCollisionSubsteps,
						PxMax<PxU32>(
							mEntries[i].surfaceCore->
								getNbCollisionSubsteps(),
							1u));
				}
			}

			compileWorldStatics(rigidMaterialManager);
			if(mWorkspacePreflightPending)
			{
				// This is a one-time scene lifecycle preflight, before the
				// first component solve. Later soft actor adds use the same
				// helper at their actor-add boundary; other dynamic mutation
				// paths retain normal, telemetry-visible growth.
				reserveLifecycleContactCapacity();
			}
			// avbdStepSoftBodies() resets its own counters before solver work.
			// Reset and snapshot the initial OGC pass separately so capacity
			// growth in that pass is not lost from the scene-level telemetry.
			mWorkspace.contact.beginStep();
			detectContacts(
				mParticles.begin(), mParticles.size(),
				mBodies.begin(), mBodies.size(), mContacts);
			plan.initialContactWorkspaceGrowthEvents =
				mWorkspace.contact.growthEvents;
			plan.initialContactWorkspaceGrowthBytes =
				mWorkspace.contact.growthBytes;
			plan.initialContactSweepScratchGrowthEvents =
				mWorkspace.contact.sweepScratchGrowthEvents;
			plan.initialContactSweepScratchGrowthBytes =
				mWorkspace.contact.sweepScratchGrowthBytes;
			plan.initialContactOutputGrowthEvents =
				mWorkspace.contact.outputGrowthEvents;
			plan.initialContactOutputGrowthBytes =
				mWorkspace.contact.outputGrowthBytes;
			const bool needsContactRedetection =
				!mContacts.empty() || mBodies.size() > 1 ||
				!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
				!mRigidCapsules.empty() ||
				!mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				!mWorldPlanes.empty();
			// OGC may retain one same-time manifold across several material
			// sweeps only when every row has a conservative component-owned
			// safety bound. Dynamic rigid targets use the native shared
			// OgcPairState path instead, so do not pack this fallback schedule
			// merely because a scene happens to contain a contact.
			const bool canPackComponentOgcEpochs =
				allEntriesAreVolumes && !mContacts.empty() &&
				Dy::avbdCanReuseComponentOgcEpoch(
					mContacts.begin(), mContacts.size(),
					mBodies.begin(), mBodies.size(),
					mParticles.begin());
			// PxDeformableBody::setSolverIterationCounts() specifies the
			// minimum position-iteration budget for the complete timestep.
			// A non-zero Surface pair-update count explicitly selects the
			// number of OGC redetection stages; zero retains adaptive
			// redetection. Collision substeps set the minimum number of
			// contact-bearing sweeps in each stage without changing dt or
			// integrating elasticity more than the resulting solver budget.
			const PxU32 requestedRedetectionStages =
				needsContactRedetection
					? (hasExplicitCollisionPairUpdates
						? requestedCollisionPairUpdates
						: 8u)
					: 1u;
			const PxU32 minimumContactIterations =
				needsContactRedetection
					? PxMax<PxU32>(
						8u,
						requestedRedetectionStages *
							requestedCollisionSubsteps)
					: 1u;
			plan.totalPositionIterations = PxMax<PxU32>(
				requestedPositionIterations,
				minimumContactIterations);
			plan.outerIterations =
				needsContactRedetection
					? PxMin<PxU32>(
						requestedRedetectionStages,
						plan.totalPositionIterations)
					: 1u;
			plan.innerIterations =
				(plan.totalPositionIterations + plan.outerIterations - 1) /
				plan.outerIterations;
			// Retain the position-iteration budget, but batch the safe portion
			// into three OGC epochs.  If a candidate reaches its pair bound the
			// solver exits the epoch immediately and rebuilds the current-pose
			// manifold, so this is neither a CCD path nor a hidden substep.
			if(canPackComponentOgcEpochs &&
				needsContactRedetection &&
				!hasExplicitCollisionPairUpdates &&
				requestedCollisionSubsteps == 1u &&
				plan.totalPositionIterations >= 8u)
			{
				plan.outerIterations = PxMin<PxU32>(
					3u, plan.totalPositionIterations);
				plan.innerIterations =
					(plan.totalPositionIterations + plan.outerIterations - 1) /
					plan.outerIterations;
			}
			// This diagnostic route intentionally remains explicit: unlike the
			// safety-admitted scheduler above, it may exercise a cadence even when
			// the contact stream is not eligible for manifold reuse.
			else if(useAvbdVolumeTest3x3Cadence() && allEntriesAreVolumes &&
				needsContactRedetection &&
				!hasExplicitCollisionPairUpdates &&
				requestedRedetectionStages == 8u &&
				requestedCollisionSubsteps == 1u &&
				plan.totalPositionIterations == 8u &&
				plan.outerIterations == 8u)
			{
				plan.outerIterations = 3u;
				plan.innerIterations = 3u;
			}
			mComponentFallbackPlanPrepared = true;
		}

		void resumeComponentFallback(
			PxReal dt, const PxVec3& gravity)
		{
			PX_ASSERT(mComponentFallbackPlanPrepared);
			if(!mComponentFallbackPlanPrepared)
				return;
			const ComponentFallbackPlan& plan = mComponentFallbackPlan;
			// This is the synchronous fallback authority.  Do not inherit a stale
			// worker count from a prior asynchronous frame merely to choose a
			// different nonlinear trajectory; only an explicit process override
			// may request colored execution here.
			const Dy::AvbdParticlePrimalSchedule configuredSchedule =
				mStandaloneTaskGraphEnhancedDeterminism
					? Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR
					: Dy::avbdGetConfiguredParticlePrimalSchedule();
			const Dy::AvbdParticlePrimalSchedule particlePrimalSchedule =
				configuredSchedule ==
					Dy::AvbdParticlePrimalSchedule::eDEFAULT
					? Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR
					: configuredSchedule;

			const auto executeSoftBodyStep =
				[&](Dy::AvbdSoftBodyStepExecutionMode executionMode)
			{
					Dy::avbdStepSoftBodies(
						mParticles.begin(), mParticles.size(),
						mBodies.begin(), mBodies.size(),
						mContacts.begin(), mContacts.size(),
						dt, gravity,
						plan.outerIterations, plan.innerIterations,
						1000.0f,
						redetectContacts, &mContacts, this,
						0.92f, &mLastStepStats, &mWorkspace,
						plan.totalPositionIterations,
						mSelfCollisionAdjacencies.begin(),
						mSelfCollisionAdjacencies.size(),
						mSelfCollisionEnabled.begin(),
						&mContactParams, executionMode,
						particlePrimalSchedule);
				};
			if(mP3ForceSplitPrediction)
			{
				executeSoftBodyStep(
					Dy::AvbdSoftBodyStepExecutionMode::ePREPARE);
				const bool useRigidInitialGuess =
					Dy::avbdCanUseSoftRigidPrimalInitialization(
						mParticles.begin(), mParticles.size(),
						mBodies.begin(), mBodies.size());
				const bool useAdaptiveInitialGuess = !useRigidInitialGuess &&
					Dy::avbdCanUseSoftAdaptivePrimalInitialization(
						mParticles.begin(), mParticles.size(),
						mBodies.begin(), mBodies.size());
				Dy::avbdPredictSoftBodyParticles(
					mParticles.begin(), mParticles.size(), dt, gravity,
					useAdaptiveInitialGuess);
				if(useRigidInitialGuess)
				{
					for(PxU32 bodyIndex = 0;
						bodyIndex < mBodies.size(); bodyIndex++)
						Dy::avbdApplySoftBodyRigidPrimalInitialGuess(
							mParticles.begin(), mParticles.size(),
							mBodies[bodyIndex]);
				}
				executeSoftBodyStep(
					Dy::AvbdSoftBodyStepExecutionMode::eRESUME);
			}
			else
			{
				executeSoftBodyStep(
					Dy::AvbdSoftBodyStepExecutionMode::eFULL);
			}
			mLastStepStats.contactWorkspaceGrowthEvents +=
				plan.initialContactWorkspaceGrowthEvents;
			mLastStepStats.contactWorkspaceGrowthBytes +=
				plan.initialContactWorkspaceGrowthBytes;
			mLastStepStats.contactSweepScratchGrowthEvents +=
				plan.initialContactSweepScratchGrowthEvents;
			mLastStepStats.contactSweepScratchGrowthBytes +=
				plan.initialContactSweepScratchGrowthBytes;
			mLastStepStats.contactOutputGrowthEvents +=
				plan.initialContactOutputGrowthEvents;
			mLastStepStats.contactOutputGrowthBytes +=
				plan.initialContactOutputGrowthBytes;
			++mLastComponentFallbackSteps;
			mComponentFallbackPlanPrepared = false;
		}

		void stepComponentFallback(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager)
		{
			prepareComponentFallback(materialManager, rigidMaterialManager);
			resumeComponentFallback(dt, gravity);
		}

	public:
		void recordStandaloneTaskGraphSubmission(
			PxU32 dispatcherWorkers, PxU32 particleCount)
		{
			mStandaloneTaskGraphTelemetry.recordSolveSubmission(
				dispatcherWorkers, particleCount);
		}

		void recordStandaloneTaskGraphSerialSolve(
			PxU32 dispatcherWorkers, PxU32 particleCount)
		{
			mStandaloneTaskGraphTelemetry.recordSerialSolve(
				dispatcherWorkers, particleCount);
		}

		void finishStandaloneTaskGraphNoOp()
		{
			// A submitted root can discover that every component went to sleep
			// before its prepare stage.  It still completed as a Scene task, but
			// has no component post-solve to close the boundary telemetry.
			mStandaloneTaskGraphTelemetry.endSolveTask();
		}

		void writeAvbdCpuSoftBodyStatistics(
			PxSimulationStatistics& stats) const
		{
			stats.avbdCpuSoftBodyComponentFallbackSteps =
				mLastComponentFallbackSteps;
			stats.avbdCpuSoftBodyNativeIslandSteps =
				mLastNativeIslandSteps;
			stats.avbdCpuTaskGraphRequestedDispatcherWorkers =
				mStandaloneTaskGraphTelemetry.requestedDispatcherWorkers.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedSolveTasks =
				mStandaloneTaskGraphTelemetry.submittedSolveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedSolveTasks =
				mStandaloneTaskGraphTelemetry.completedSolveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveSolveTasks =
				mStandaloneTaskGraphTelemetry.peakActiveSolveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphBarrierTasks =
				mStandaloneTaskGraphTelemetry.causalLayerFanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialSolveTasks =
				mStandaloneTaskGraphTelemetry.serialSolveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPureSoftEligibleIslands =
				mStandaloneTaskGraphTelemetry.pureSoftEligibleIslands.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPureSoftEligibleParticles =
				mStandaloneTaskGraphTelemetry.pureSoftEligibleParticles.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedCausalLayerTasks =
				mStandaloneTaskGraphTelemetry.submittedCausalLayerTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedCausalLayerTasks =
				mStandaloneTaskGraphTelemetry.completedCausalLayerTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveCausalLayerTasks =
				mStandaloneTaskGraphTelemetry.peakActiveCausalLayerTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCausalLayerFanIns =
				mStandaloneTaskGraphTelemetry.causalLayerFanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialCausalLayerFallbacks =
				mStandaloneTaskGraphTelemetry.serialCausalLayerFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphMaxCausalLayerOccupancy =
				mStandaloneTaskGraphTelemetry.maxCausalLayerOccupancy.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphTotalCausalLayerOccupancy =
				mStandaloneTaskGraphTelemetry.totalCausalLayerOccupancy.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedWorldPlaneContactTasks =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedWorldPlaneContactTasks =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveWorldPlaneContactTasks =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphWorldPlaneContactFanIns =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialWorldPlaneContactFallbacks =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidBoxSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidBoxSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidBoxSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidBoxSdfContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidBoxSdfContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidSphereSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidSphereSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidSphereSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidSphereSdfContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidSphereSdfContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidCapsuleSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidCapsuleSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidCapsuleSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidCapsuleSdfContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidCapsuleSdfContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidConvexSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidConvexSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidConvexSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidConvexSdfContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidConvexSdfContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidTriangleSurfaceContactTasks =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidTriangleSurfaceContactTasks =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidTriangleSurfaceContactTasks =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidTriangleSurfaceContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidTriangleSurfaceContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedSoftPairContactTasks =
				mStandaloneTaskGraphTelemetry.softPairContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedSoftPairContactTasks =
				mStandaloneTaskGraphTelemetry.softPairContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveSoftPairContactTasks =
				mStandaloneTaskGraphTelemetry.softPairContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSoftPairContactFanIns =
				mStandaloneTaskGraphTelemetry.softPairContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialSoftPairContactFallbacks =
				mStandaloneTaskGraphTelemetry.softPairContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedSelfBvhContactTasks =
				mStandaloneTaskGraphTelemetry.selfBvhContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedSelfBvhContactTasks =
				mStandaloneTaskGraphTelemetry.selfBvhContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveSelfBvhContactTasks =
				mStandaloneTaskGraphTelemetry.selfBvhContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSelfBvhContactFanIns =
				mStandaloneTaskGraphTelemetry.selfBvhContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialSelfBvhContactFallbacks =
				mStandaloneTaskGraphTelemetry.selfBvhContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuSoftBodyWorkspaceGrowthEvents =
				mLastStepStats.workspaceGrowthEvents;
			stats.avbdCpuSoftBodyWorkspaceGrowthBytes =
				mLastStepStats.workspaceGrowthBytes;
			stats.avbdCpuSoftBodyContactWorkspaceGrowthEvents =
				mLastStepStats.contactWorkspaceGrowthEvents;
			stats.avbdCpuSoftBodyContactWorkspaceGrowthBytes =
				mLastStepStats.contactWorkspaceGrowthBytes;
			stats.avbdCpuSoftBodyContactSweepScratchGrowthEvents =
				mLastStepStats.contactSweepScratchGrowthEvents;
			stats.avbdCpuSoftBodyContactSweepScratchGrowthBytes =
				mLastStepStats.contactSweepScratchGrowthBytes;
			stats.avbdCpuSoftBodyContactOutputGrowthEvents =
				mLastStepStats.contactOutputGrowthEvents;
			stats.avbdCpuSoftBodyContactOutputGrowthBytes =
				mLastStepStats.contactOutputGrowthBytes;
			stats.avbdCpuSoftBodyPeakContactOutputCount =
				mLastStepStats.peakContactOutputCount;
			stats.avbdCpuSoftBodyPeakContactOutputCapacity =
				mLastStepStats.peakContactOutputCapacity;
			stats.avbdCpuSoftBodyPeakContactIncidenceCount =
				mLastStepStats.peakContactIncidenceCount;
			stats.avbdCpuSoftBodyPeakContactIncidenceCapacity =
				mLastStepStats.peakContactIncidenceCapacity;
			stats.avbdCpuSoftBodyPeakStateTransferContactCount =
				mLastStepStats.peakStateTransferContactCount;
			stats.avbdCpuSoftBodyPeakStateTransferContactCapacity =
				mLastStepStats.peakStateTransferContactCapacity;
			stats.avbdCpuSoftBodyPeakStateTransferUsedCapacity =
				mLastStepStats.peakStateTransferUsedCapacity;
			stats.avbdCpuSoftBodyParticlePrimalColorCount =
				mLastStepStats.particlePrimalColorCount;
			stats.avbdCpuSoftBodyParticlePrimalDynamicAccessGroupCount =
				mLastStepStats.particlePrimalDynamicAccessGroupCount;
			stats.avbdCpuSoftBodyParticlePrimalColoredSerialSweeps =
				mLastStepStats.particlePrimalColoredSerialSweeps;
			stats.avbdCpuSoftBodyParticlePrimalColoredSerialFallbackSweeps =
				mLastStepStats.particlePrimalColoredSerialFallbackSweeps;
			stats.avbdCpuSoftBodyGroundTetPatchGroundPositionAlRows =
				mLastStepStats.groundTetPatchGroundPositionAlRows;
			stats.avbdCpuSoftBodyGroundTetPatchFourSupportRows =
				mLastStepStats.groundTetPatchFourSupportRows;
			stats.avbdCpuSoftBodyGroundTetPatchSingleTetRows =
				mLastStepStats.groundTetPatchSingleTetRows;
			stats.avbdCpuSoftBodyGroundTetPatchActiveRows =
				mLastStepStats.groundTetPatchActiveRows;
			stats.avbdCpuSoftBodyWorldStaticVelocityTangentOwnerRows =
				mLastStepStats.worldStaticVelocityTangentOwnerRows;
			stats.avbdCpuSoftBodyWorldStaticVelocityTangentAppliedRows =
				mLastStepStats.worldStaticVelocityTangentAppliedRows;
			stats.avbdCpuSoftBodyParticlePrimalCensusDynamicParticleSolves =
				mLastStepStats.particlePrimalCensusDynamicParticleSolves;
			stats.avbdCpuSoftBodyParticlePrimalCensusTriangleEvaluations =
				mLastStepStats.particlePrimalCensusTriangleEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusCorotationalTetEvaluations =
				mLastStepStats.
					particlePrimalCensusCorotationalTetEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusNeoHookeanTetEvaluations =
				mLastStepStats.particlePrimalCensusNeoHookeanTetEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusBendingEvaluations =
				mLastStepStats.particlePrimalCensusBendingEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusContactEvaluations =
				mLastStepStats.particlePrimalCensusContactEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusTetPacket8FullPackets =
				mLastStepStats.particlePrimalCensusTetPacket8FullPackets;
			stats.avbdCpuSoftBodyParticlePrimalCensusTetPacket8TailLanes =
				mLastStepStats.particlePrimalCensusTetPacket8TailLanes;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrBodies =
				mLastStepStats.particlePrimalTetPacketIrBodies;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrPackets =
				mLastStepStats.particlePrimalTetPacketIrPackets;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrActiveLanes =
				mLastStepStats.particlePrimalTetPacketIrActiveLanes;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrTailLanes =
				mLastStepStats.particlePrimalTetPacketIrTailLanes;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrActiveTailLanes =
				mLastStepStats.particlePrimalTetPacketIrActiveTailLanes;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrInvalidBodies =
				mLastStepStats.particlePrimalTetPacketIrInvalidBodies;
			stats.avbdCpuSoftBodyCollisionDetectionCalls =
				mLastCollisionStats.detectionCalls;
			stats.avbdCpuSoftBodyCollisionBodyPairs =
				mLastCollisionStats.bodyPairs;
			stats.avbdCpuSoftBodyCollisionOverlappingBodyPairs =
				mLastCollisionStats.overlappingBodyPairs;
			stats.avbdCpuSoftBodyCollisionParticleSurfaceCandidates =
				mLastCollisionStats.particleSurfaceCandidates;
			stats.avbdCpuSoftBodyCollisionInsideTriangleTests =
				mLastCollisionStats.insideTriangleTests;
			stats.avbdCpuSoftBodyCollisionClosestTriangleTests =
				mLastCollisionStats.closestTriangleTests;
			stats.avbdCpuSoftBodyCollisionSelfTriangleTests =
				mLastCollisionStats.selfTriangleTests;
			stats.avbdCpuSoftBodyCollisionSelfTriangleBoundsBuilt =
				mLastCollisionStats.selfTriangleBoundsBuilt;
			stats.avbdCpuSoftBodyCollisionSelfVertexSweepEntriesBuilt =
				mLastCollisionStats.selfVertexSweepEntriesBuilt;
			stats.avbdCpuSoftBodyCollisionSelfEdgeBoundsBuilt =
				mLastCollisionStats.selfEdgeBoundsBuilt;
			stats.avbdCpuSoftBodyCollisionSurfaceBvhRefitNodes =
				mLastCollisionStats.surfaceTriangleBvhRefitNodes;
			stats.avbdCpuSoftBodyCollisionSurfaceBvhCandidates =
				mLastCollisionStats.surfaceTriangleBvhCandidateTriangles;
			stats.avbdCpuSoftBodyCollisionSurfaceEdgeBvhRefitNodes =
				mLastCollisionStats.surfaceEdgeBvhRefitNodes;
			stats.avbdCpuSoftBodyCollisionSurfaceEdgeBvhCandidates =
				mLastCollisionStats.surfaceEdgeBvhCandidateEdges;
			stats.avbdCpuSoftBodyCollisionRigidParticleTests =
				mLastCollisionStats.rigidParticleBoxTests +
				mLastCollisionStats.rigidParticleSphereTests +
				mLastCollisionStats.rigidParticleCapsuleTests +
				mLastCollisionStats.rigidParticleConvexTests +
				mLastCollisionStats.rigidParticleTriangleSurfaceTests;
			stats.avbdCpuSoftBodyCollisionRigidTriangleFaceCandidates =
				mLastCollisionStats.rigidTriangleSurfaceFaceCandidates;
			stats.avbdCpuSoftBodyCollisionRigidTriangleFaceTests =
				mLastCollisionStats.rigidTriangleSurfaceFaceTests;
			stats.avbdCpuSoftBodyCollisionRigidTriangleEdgeCandidates =
				mLastCollisionStats.rigidTriangleSurfaceEdgeCandidates;
			stats.avbdCpuSoftBodyCollisionRigidTriangleEdgeTests =
				mLastCollisionStats.rigidTriangleSurfaceEdgeTests;
			stats.avbdCpuSoftBodyCollisionRigidTriangleVertexCandidates =
				mLastCollisionStats.rigidTriangleSurfaceVertexCandidates;
			stats.avbdCpuSoftBodyCollisionRigidTriangleVertexTests =
				mLastCollisionStats.rigidTriangleSurfaceVertexTests;
			stats.avbdCpuSoftBodyCollisionGeneratedGroundContacts =
				mLastCollisionStats.generatedGroundContacts;
			stats.avbdCpuSoftBodyCollisionGeneratedRigidContacts =
				mLastCollisionStats.generatedRigidContacts;
			stats.avbdCpuSoftBodyCollisionGeneratedSoftContacts =
				mLastCollisionStats.generatedSoftContacts;
			stats.avbdCpuSoftBodyCollisionGeneratedSelfContacts =
				mLastCollisionStats.generatedSelfContacts;
		}

	private:
		const PxsDeformableSurfaceMaterialCore*
			getSurfaceMaterial(
				const DeformableSurfaceCore& core) const
		{
			const PxArray<PxU16>& handles =
				core.getCore().materialHandles;
			if(handles.empty() ||
				handles[0] == MATERIAL_INVALID_HANDLE ||
				handles[0] >= mSurfaceMaterialManager.getMaxSize())
				return NULL;
			const PxsDeformableSurfaceMaterialCore* material =
				mSurfaceMaterialManager.getMaterial(handles[0]);
			return material->mMaterialIndex == handles[0]
				? material : NULL;
		}

		bool rebuildSurfaceRestState(Entry& entry)
		{
			PX_ASSERT(entry.kind == eSURFACE && entry.surfaceCore);
			Dy::DeformableSurfaceCore& core =
				entry.surfaceCore->getCore();
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 numVertices = getParticleCount(entry);
			const PxU32 numTriangles =
				entry.triangleMesh->getNbTriangles();
			PxArray<PxVec3> restVertices;
			restVertices.resize(numVertices);
			for(PxU32 i = 0; i < numVertices; i++)
			{
				restVertices[i] = core.restPosition[i].getXYZ();
				if(!restVertices[i].isFinite())
					return false;
			}
			PxArray<PxU32> triangles;
			triangles.resize(3 * numTriangles);
			const bool has16BitIndices =
				entry.triangleMesh->getTriangleMeshFlags() &
				PxTriangleMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source =
					static_cast<const PxU16*>(
						entry.triangleMesh->getTriangles());
				for(PxU32 i = 0; i < triangles.size(); i++)
					triangles[i] = source[i];
			}
			else
			{
				const PxU32* source =
					static_cast<const PxU32*>(
						entry.triangleMesh->getTriangles());
				for(PxU32 i = 0; i < triangles.size(); i++)
					triangles[i] = source[i];
			}

			const PxsDeformableSurfaceMaterialCore* material =
				getSurfaceMaterial(*entry.surfaceCore);
			PxArray<Dy::AvbdSoftParticle> rebuiltParticles;
			PxArray<Dy::AvbdSoftBody> rebuiltBodies;
			Dy::avbdCreateSoftBody(
				restVertices.begin(), numVertices,
				NULL, 0,
				triangles.begin(), triangles.size(),
				material ? material->youngs : 1.0e5f,
				material ? material->poissons : 0.3f,
				1.0f,
				entry.surfaceCore->getLinearDamping() +
					(material ? material->elasticityDamping : 0.0f),
				material ? material->bendingStiffness : 0.0f,
				material
					? PxMax(material->thickness, 1.0e-4f)
					: 0.01f,
				rebuiltParticles, rebuiltBodies,
				(entry.surfaceCore->getSurfaceFlags() &
					PxDeformableSurfaceFlag::eENABLE_FLATTENING)
					? true : false,
				entry.surfaceCore->
					getSelfCollisionFilterDistance(),
				material ? material->dynamicFriction : 0.5f);
			if(rebuiltBodies.size() != 1 ||
				rebuiltParticles.size() != numVertices)
				return false;
			if(!rebaseSoftBodyParticleRangeInPlace(
				rebuiltBodies[0], 0, numVertices,
				particleStart))
				return false;
			mBodies[entry.bodyIndex] = rebuiltBodies[0];
			PX_ASSERT(
				entry.bodyIndex < mSelfCollisionAdjacencies.size());
			Dy::avbdBuildSelfCollisionAdjacency(
				mBodies[entry.bodyIndex],
				mSelfCollisionAdjacencies[entry.bodyIndex]);
			if(!rebuildEntryPins(entry))
				return false;
			for(PxU32 i = 0; i < numVertices; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				particle.initialPosition = restVertices[i];
				particle.elasticK = 0.0f;
			}
			return true;
		}

		Entry* findEntry(ActorCore& core)
		{
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				Entry& entry = mEntries[i];
				if(entry.getActorCore() == &core)
					return &entry;
			}
			return NULL;
		}

		PX_FORCE_INLINE PxU32 getParticleStart(
			const Entry& entry) const
		{
			PX_ASSERT(entry.bodyIndex < mBodies.size());
			return mBodies[entry.bodyIndex].compiled.particleStart;
		}

		PX_FORCE_INLINE PxU32 getParticleCount(
			const Entry& entry) const
		{
			PX_ASSERT(entry.bodyIndex < mBodies.size());
			return mBodies[entry.bodyIndex].compiled.particleCount;
		}

		bool isVolumeKinematicTargetActive(
			const Dy::DeformableVolumeCore& core,
			const PxVec4& target) const
		{
			if(!core.kinematicTarget ||
				!target.getXYZ().isFinite())
				return false;
			if(core.bodyFlags & PxDeformableBodyFlag::eKINEMATIC)
				return true;
			return
				(core.volumeFlags &
					PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC) &&
				target.w == 0.0f;
		}

		bool appendVolumeKinematicTargetPins(
			const Entry& entry,
			const PxArray<Dy::AvbdKinematicPin>& previousPins,
			PxArray<Dy::AvbdKinematicPin>& pins) const
		{
			if(entry.kind != eVOLUME || !entry.volumeCore)
				return true;
			const Dy::DeformableVolumeCore& core =
				entry.volumeCore->getCore();
			if(!core.kinematicTarget ||
				!(core.bodyFlags &
						PxDeformableBodyFlag::eKINEMATIC) &&
				!(core.volumeFlags &
						PxDeformableVolumeFlag::
							ePARTIALLY_KINEMATIC))
				return true;
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			for(PxU32 localIndex = 0;
				localIndex < particleCount; localIndex++)
			{
				const PxVec4& target =
					core.kinematicTarget[localIndex];
				if(!isVolumeKinematicTargetActive(core, target))
					continue;
				Dy::AvbdKinematicPin pin;
				pin.point.setVertex(particleStart + localIndex);
				pin.sourceHandle = localIndex;
				pin.targetKind =
					Dy::AvbdSoftPinTargetKind::
						eDEFORMABLE_KINEMATIC;
				pin.worldTarget = target.getXYZ();
				pin.previousWorldTarget = pin.worldTarget;
				pin.k = 1.0e8f;
				pin.kMax = 1.0e10f;
				for(PxU32 previousIndex = 0;
					previousIndex < previousPins.size();
					previousIndex++)
				{
					const Dy::AvbdKinematicPin& previous =
						previousPins[previousIndex];
					if(previous.targetKind != pin.targetKind ||
						previous.sourceHandle != localIndex)
						continue;
					pin.previousWorldTarget =
						previous.worldTarget;
					pin.alLambda = previous.alLambda;
					pin.k = previous.k;
					pin.kMax = previous.kMax;
					break;
				}
				pins.pushBack(pin);
			}
			return true;
		}

		bool rebuildEntryPins(Entry& entry)
		{
			if(entry.bodyIndex >= mBodies.size())
				return false;
			Dy::AvbdSoftBody& body = mBodies[entry.bodyIndex];
			const PxU32 particleStart =
				body.compiled.particleStart;
			const PxU32 particleCount =
				body.compiled.particleCount;
			const PxArray<Dy::AvbdKinematicPin> previousPins =
				body.runtime.pins;
			body.runtime.pins.clear();
			if(!appendVolumeKinematicTargetPins(
				entry, previousPins, body.runtime.pins))
				return false;
			for(PxU32 i = 0; i < mWorldPins.size(); i++)
			{
				const WorldPinEntry& source = mWorldPins[i];
				if(source.softCore != entry.getActorCore())
					continue;
				if(!Dy::avbdIsSoftPointValid(
					source.localPoint, 0, particleCount))
					return false;
				Dy::AvbdKinematicPin pin;
				pin.point = source.localPoint;
				for(PxU32 endpoint = 0;
					endpoint < pin.point.particleCount; endpoint++)
					pin.point.particleIndices[endpoint] +=
						particleStart;
				pin.sourceHandle = source.handle;
				pin.targetKind =
					Dy::AvbdSoftPinTargetKind::eWORLD_FIXED;
				pin.worldTarget = source.worldTarget;
				pin.previousWorldTarget =
					source.worldTarget;
				// Public vertex-to-world attachments are fixed positional
				// objectives. Keep their compliance below the public gate
				// tolerance while retaining the position-level AL owner.
				pin.k = 1.0e8f;
				pin.kMax = 1.0e10f;
				body.runtime.pins.pushBack(pin);
			}
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				PrescribedAttachmentEntry& source =
					mPrescribedAttachments[i];
				if(source.softCore != entry.getActorCore() ||
					!source.active)
					continue;
				if(!Dy::avbdIsSoftPointValid(
					source.localPoint, 0, particleCount))
					return false;
				PxVec3 worldTarget;
				if(!computePrescribedAttachmentWorldTarget(
					*source.prescribedCore,
					source.actorLocalTarget, worldTarget))
					return false;
				source.worldTarget = worldTarget;
				Dy::AvbdKinematicPin pin;
				pin.point = source.localPoint;
				for(PxU32 endpoint = 0;
					endpoint < pin.point.particleCount; endpoint++)
					pin.point.particleIndices[endpoint] +=
						particleStart;
				pin.sourceHandle = source.handle;
				pin.targetKind =
					Dy::AvbdSoftPinTargetKind::
						ePRESCRIBED_RIGID;
				pin.worldTarget = source.worldTarget;
				pin.previousWorldTarget =
					source.previousWorldTarget;
				pin.alLambda = source.alLambda;
				pin.k = source.k;
				pin.kMax = source.kMax;
				body.runtime.pins.pushBack(pin);
			}
			body.runtime.compileObjectiveProgram(
				particleStart, particleCount);
			return body.runtime.isObjectiveProgramCurrent(
				particleStart, particleCount);
		}

		void refreshVolumeKinematicTargets()
		{
			for(PxU32 entryIndex = 0;
				entryIndex < mEntries.size(); entryIndex++)
			{
				Entry& entry = mEntries[entryIndex];
				if(entry.kind != eVOLUME || !entry.volumeCore ||
					entry.bodyIndex >= mBodies.size())
					continue;
				const Dy::DeformableVolumeCore& core =
					entry.volumeCore->getCore();
				Dy::AvbdSoftBodyRuntimeState& runtime =
					mBodies[entry.bodyIndex].runtime;
				const PxU32 particleCount =
					getParticleCount(entry);
				PxU32 expectedCount = 0;
				if(core.kinematicTarget)
				{
					for(PxU32 localIndex = 0;
						localIndex < particleCount; localIndex++)
					{
						if(isVolumeKinematicTargetActive(
							core,
							core.kinematicTarget[localIndex]))
							expectedCount++;
					}
				}

				PxU32 existingCount = 0;
				bool needsRebuild = false;
				for(PxU32 pinIndex = 0;
					pinIndex < runtime.pins.size(); pinIndex++)
				{
					const Dy::AvbdKinematicPin& pin =
						runtime.pins[pinIndex];
					if(pin.targetKind !=
						Dy::AvbdSoftPinTargetKind::
							eDEFORMABLE_KINEMATIC)
						continue;
					existingCount++;
					if(pin.sourceHandle >= particleCount ||
						!core.kinematicTarget ||
						!isVolumeKinematicTargetActive(
							core,
							core.kinematicTarget[
								pin.sourceHandle]))
						needsRebuild = true;
				}
				needsRebuild =
					needsRebuild || existingCount != expectedCount;
				if(needsRebuild)
				{
					const bool rebuilt = rebuildEntryPins(entry);
					PX_ASSERT(rebuilt);
					if(rebuilt && expectedCount > 0)
						wakeEntry(
							entry,
							ScInternalWakeCounterResetValue);
					continue;
				}

				bool targetMoved = false;
				for(PxU32 pinIndex = 0;
					pinIndex < runtime.pins.size(); pinIndex++)
				{
					Dy::AvbdKinematicPin& pin =
						runtime.pins[pinIndex];
					if(pin.targetKind !=
						Dy::AvbdSoftPinTargetKind::
							eDEFORMABLE_KINEMATIC)
						continue;
					PX_ASSERT(core.kinematicTarget &&
						pin.sourceHandle < particleCount);
					if(!core.kinematicTarget ||
						pin.sourceHandle >= particleCount)
						continue;
					const PxVec3 previousTarget =
						pin.worldTarget;
					const PxVec3 worldTarget =
						core.kinematicTarget[
							pin.sourceHandle].getXYZ();
					pin.previousWorldTarget = previousTarget;
					pin.worldTarget = worldTarget;
					targetMoved = targetMoved ||
						(worldTarget - previousTarget).
							magnitudeSquared() > 1.0e-12f;
				}
				if(targetMoved)
					wakeEntry(
						entry,
						ScInternalWakeCounterResetValue);
			}
		}

		void refreshPrescribedAttachmentTargets()
		{
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				PrescribedAttachmentEntry& attachment =
					mPrescribedAttachments[i];
				Entry* entry =
					findEntry(*attachment.softCore);
				if(!entry)
					continue;
				PxVec3 worldTarget;
				const bool active =
					computePrescribedAttachmentWorldTarget(
						*attachment.prescribedCore,
						attachment.actorLocalTarget,
						worldTarget);
				if(active != attachment.active)
				{
					attachment.active = active;
					if(active)
					{
						attachment.worldTarget =
							worldTarget;
						attachment.previousWorldTarget =
							worldTarget;
					}
					const bool rebuilt =
						rebuildEntryPins(*entry);
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
					if(active)
						wakeEntry(
							*entry,
							ScInternalWakeCounterResetValue);
					continue;
				}
				if(!active)
					continue;

				Dy::AvbdSoftBodyRuntimeState& runtime =
					mBodies[entry->bodyIndex].runtime;
				Dy::AvbdKinematicPin* pin = NULL;
				for(PxU32 pinIndex = 0;
					pinIndex < runtime.pins.size(); pinIndex++)
				{
					Dy::AvbdKinematicPin& candidate =
						runtime.pins[pinIndex];
					if(candidate.targetKind ==
							Dy::AvbdSoftPinTargetKind::
								ePRESCRIBED_RIGID &&
						candidate.sourceHandle ==
							attachment.handle)
					{
						pin = &candidate;
						break;
					}
				}
				PX_ASSERT(pin);
				if(!pin)
					continue;
				const PxVec3 previousTarget =
					attachment.worldTarget;
				attachment.previousWorldTarget =
					previousTarget;
				attachment.worldTarget = worldTarget;
				pin->previousWorldTarget = previousTarget;
				pin->worldTarget = worldTarget;
				if((worldTarget - previousTarget).
					magnitudeSquared() > 1.0e-12f)
					wakeEntry(
						*entry,
						ScInternalWakeCounterResetValue);
			}
		}

		void removeWorldPinsForCore(ActorCore& core)
		{
			for(PxU32 i = mWorldPins.size(); i > 0; i--)
			{
				const WorldPinEntry& pin = mWorldPins[i - 1];
				if(pin.softCore == &core)
					mWorldPins.replaceWithLast(i - 1);
			}
		}

		void removeRigidAttachmentsForSoft(ActorCore& core)
		{
			bool removed = false;
			for(PxU32 i = mRigidAttachments.size(); i > 0; i--)
			{
				if(mRigidAttachments[i - 1].softCore == &core)
				{
					mRigidAttachments.replaceWithLast(i - 1);
					removed = true;
				}
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void removeArticulationAttachmentsForSoft(
			ActorCore& core)
		{
			bool removed = false;
			for(PxU32 i = mArticulationAttachments.size();
				i > 0; i--)
			{
				if(mArticulationAttachments[i - 1].softCore ==
					&core)
				{
					mArticulationAttachments.replaceWithLast(
						i - 1);
					removed = true;
				}
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void removeSoftPairAttachmentsForSoft(
			ActorCore& core)
		{
			bool removed = false;
			for(PxU32 i = mSoftPairAttachments.size(); i > 0; i--)
			{
				const SoftPairAttachmentEntry& attachment =
					mSoftPairAttachments[i - 1];
				if(attachment.softCore[0] != &core &&
					attachment.softCore[1] != &core)
					continue;
				ActorCore* otherCore =
					attachment.softCore[0] == &core
						? attachment.softCore[1]
						: attachment.softCore[0];
				mSoftPairAttachments.replaceWithLast(i - 1);
				Entry* otherEntry =
					otherCore ? findEntry(*otherCore) : NULL;
				if(otherEntry)
					wakeEntry(
						*otherEntry,
						ScInternalWakeCounterResetValue);
				removed = true;
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void removePrescribedAttachmentsForSoft(
			ActorCore& core)
		{
			for(PxU32 i = mPrescribedAttachments.size();
				i > 0; i--)
			{
				if(mPrescribedAttachments[i - 1].softCore ==
					&core)
					mPrescribedAttachments.replaceWithLast(
						i - 1);
			}
		}

		void removePrescribedAttachmentsForRigid(
			RigidCore& core)
		{
			for(PxU32 i = mPrescribedAttachments.size();
				i > 0; i--)
			{
				if(mPrescribedAttachments[i - 1].
					prescribedCore != &core)
					continue;
				ActorCore* softCore =
					mPrescribedAttachments[i - 1].softCore;
				mPrescribedAttachments.replaceWithLast(i - 1);
				Entry* entry =
					softCore ? findEntry(*softCore) : NULL;
				if(entry)
				{
					const bool rebuilt =
						rebuildEntryPins(*entry);
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
					wakeEntry(
						*entry,
						ScInternalWakeCounterResetValue);
				}
			}
		}

		void removeRigidAttachmentsForRigid(BodyCore& core)
		{
			bool removed = false;
			for(PxU32 i = mRigidAttachments.size(); i > 0; i--)
			{
				if(mRigidAttachments[i - 1].rigidCore == &core)
				{
					ActorCore* softCore =
						mRigidAttachments[i - 1].softCore;
					mRigidAttachments.replaceWithLast(i - 1);
					Entry* entry =
						softCore ? findEntry(*softCore) : NULL;
					if(entry)
						wakeEntry(
							*entry,
							ScInternalWakeCounterResetValue);
					removed = true;
				}
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void removeArticulationAttachmentsForLink(
			BodyCore& core)
		{
			bool removed = false;
			for(PxU32 i = mArticulationAttachments.size();
				i > 0; i--)
			{
				if(mArticulationAttachments[i - 1].linkCore !=
					&core)
					continue;
				ActorCore* softCore =
					mArticulationAttachments[i - 1].softCore;
				mArticulationAttachments.replaceWithLast(i - 1);
				Entry* entry =
					softCore ? findEntry(*softCore) : NULL;
				if(entry)
					wakeEntry(
						*entry,
						ScInternalWakeCounterResetValue);
				removed = true;
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void sleepEntry(Entry& entry)
		{
			Dy::DeformableBodyCore& core =
				entry.getBodyCore();
			PxVec4* velocities = entry.getVelocity();
			PX_ASSERT(velocities);
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			for(PxU32 i = 0; i < particleCount; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				particle.velocity = PxVec3(0.0f);
				particle.prevVelocity = PxVec3(0.0f);
				particle.predictedPosition = particle.position;
				particle.outerPosition = particle.position;
				particle.invMass = 0.0f;
				particle.mass = 0.0f;
				particle.gravityScale = 0.0f;
				const PxReal velocityW =
					velocities[i].w;
				velocities[i] =
					PxVec4(PxVec3(0.0f), velocityW);
			}
			core.wakeCounter = 0.0f;
			core.cpuAvbdSleeping = true;
			core.cpuAvbdWakeRequested = false;
			entry.sleeping = true;
			mIslandManager.deactivateNode(entry.islandNode);
		}

		void wakeEntry(Entry& entry, PxReal wakeCounter)
		{
			Dy::DeformableBodyCore& core =
				entry.getBodyCore();
			PxVec4* positions = entry.getPositionInvMass();
			PxVec4* velocities = entry.getVelocity();
			PX_ASSERT(positions && velocities);
			const PxReal gravityScale =
				(entry.getActorFlags() &
					PxActorFlag::eDISABLE_GRAVITY)
				? 0.0f : 1.0f;
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			for(PxU32 i = 0; i < particleCount; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				const PxReal invMass = PxMax(
					positions[i].w, 0.0f);
				particle.invMass = invMass;
				particle.mass =
					invMass > 0.0f ? 1.0f / invMass : 0.0f;
				particle.velocity =
					velocities[i].getXYZ();
				particle.prevVelocity = particle.velocity;
				particle.gravityScale = gravityScale;
			}
			core.wakeCounter = PxMax(wakeCounter, 0.0f);
			core.cpuAvbdSleeping = false;
			core.cpuAvbdWakeRequested = false;
			entry.sleeping = false;
			mIslandManager.activateNode(entry.islandNode);
		}

		void finalizeDeformableMotionControls(PxReal dt)
		{
			for(PxU32 entryIndex = 0;
				entryIndex < mEntries.size(); entryIndex++)
			{
				Entry& entry = mEntries[entryIndex];
				if(entry.sleeping)
					continue;

				const PxU32 particleStart =
					getParticleStart(entry);
				const PxU32 particleCount =
					getParticleCount(entry);
				PxReal maxSpeedSquared = 0.0f;
				for(PxU32 i = 0; i < particleCount; i++)
				{
					const Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					if(particle.invMass <= 0.0f)
						continue;
					maxSpeedSquared = PxMax(
						maxSpeedSquared,
						particle.velocity.magnitudeSquared());
				}

				const Dy::DeformableBodyCore& core =
					entry.getBodyCore();
				const PxReal settlingThreshold =
					PxMax(core.settlingThreshold, 0.0f);
				if(maxSpeedSquared >
					settlingThreshold * settlingThreshold)
					continue;
				const PxReal settlingScale = PxMax(
					1.0f -
						PxMax(core.settlingDamping, 0.0f) * dt,
					0.0f);
				if(settlingScale >= 1.0f)
					continue;
				for(PxU32 i = 0; i < particleCount; i++)
				{
					Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					if(particle.invMass <= 0.0f)
						continue;
					particle.velocity *= settlingScale;
					particle.prevVelocity = particle.velocity;
				}
			}
		}

		bool hasUnforcedRestBendingResidual(const Entry& entry) const
		{
			if(entry.bodyIndex >= mBodies.size() || !mContacts.empty() ||
				!(entry.getActorFlags() & PxActorFlag::eDISABLE_GRAVITY))
				return false;
			const Dy::AvbdSoftBody& body = mBodies[entry.bodyIndex];
			if(body.material.bendingStiffness <= 0.0f ||
				body.compiled.bendElements.empty() ||
				!body.runtime.compiledObjectives.empty() ||
				body.compiled.selfCollisionRestPositions.size() !=
					body.compiled.particleCount)
				return false;

			// A fixed vertex away from its authored rest position is an external
			// boundary condition; its equilibrium may legitimately retain bend.
			// Only the unforced/rest-boundary case has rest dihedrals as a valid
			// cold sleep certificate.
			for(PxU32 localIndex = 0;
				localIndex < body.compiled.particleCount; localIndex++)
			{
				const Dy::AvbdSoftParticle& particle = mParticles[
					body.compiled.particleStart + localIndex];
				if(particle.invMass <= 0.0f &&
					(particle.position -
					 body.compiled.selfCollisionRestPositions[localIndex]).
						magnitudeSquared() > 1.0e-8f)
					return false;
			}

			const PxReal bendingSleepAngleTolerance = 1.0e-2f;
			for(PxU32 bendingIndex = 0;
				bendingIndex < body.compiled.bendElements.size(); bendingIndex++)
			{
				const Dy::AvbdBendingElement& bending =
					body.compiled.bendElements[bendingIndex];
				const PxReal angle =
					Dy::AvbdSoftBodyCompiledData::computeDihedralAngle(
						mParticles[bending.opp0].position,
						mParticles[bending.opp1].position,
						mParticles[bending.edgeStart].position,
						mParticles[bending.edgeEnd].position);
				const PxReal angleDifference = angle - bending.restAngle;
				const PxReal wrappedError = PxAtan2(
					PxSin(angleDifference), PxCos(angleDifference));
				if(!PxIsFinite(wrappedError) ||
					PxAbs(wrappedError) > bendingSleepAngleTolerance)
					return true;
			}
			return false;
		}

		void updateSleepStates(
			PxReal dt, bool sleepingEnabled)
		{
			// Low velocity alone is not a stationarity certificate for AVBD.
			// Strong damping can remove velocity while the final particle block
			// still requests a visible elastic correction (most notably for a
			// slowly flattening cloth hinge).  The scalar component solve already
			// publishes its final pre-limiter H^-1 f displacement, so consume that
			// cold step-level certificate before freezing particle inverse masses.
			// It is intentionally conservative for multi-body components: one
			// unresolved body keeps the component awake rather than freezing a
			// coupled peer prematurely.
			const PxReal componentSleepResidualThreshold = 1.0e-4f;
			const bool componentResidualPending =
				mLastComponentFallbackSteps != 0 &&
				(!PxIsFinite(
					mLastStepStats.finalMaxLocalSolveDisplacement) ||
				 mLastStepStats.finalMaxLocalSolveDisplacement >
					componentSleepResidualThreshold);
			for(PxU32 entryIndex = 0;
				entryIndex < mEntries.size(); entryIndex++)
			{
				Entry& entry = mEntries[entryIndex];
				if(!sleepingEnabled)
				{
					if(entry.sleeping)
						wakeEntry(
							entry,
							ScInternalWakeCounterResetValue);
					continue;
				}
				if(entry.sleeping)
					continue;

				PxReal maxSpeedSquared = 0.0f;
				const PxU32 particleStart =
					getParticleStart(entry);
				const PxU32 particleCount =
					getParticleCount(entry);
				for(PxU32 i = 0; i < particleCount; i++)
				{
					const Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					if(particle.invMass <= 0.0f)
						continue;
					maxSpeedSquared = PxMax(
						maxSpeedSquared,
						particle.velocity.magnitudeSquared());
				}
				Dy::DeformableBodyCore& core =
					entry.getBodyCore();
				bool kinematicTargetResidualPending = false;
				if(entry.bodyIndex < mBodies.size())
				{
					const Dy::AvbdSoftBodyRuntimeState& runtime =
						mBodies[entry.bodyIndex].runtime;
					for(PxU32 pinIndex = 0;
						pinIndex < runtime.pins.size(); pinIndex++)
					{
						const Dy::AvbdKinematicPin& pin =
							runtime.pins[pinIndex];
						if(pin.targetKind !=
							Dy::AvbdSoftPinTargetKind::
								eDEFORMABLE_KINEMATIC)
							continue;
						const PxReal
							kinematicTargetResidualSquared =
								(Dy::avbdGetSoftPointPosition(
									pin.point,
									mParticles.begin()) -
								 pin.worldTarget).
									magnitudeSquared();
						if(kinematicTargetResidualSquared >
							1.0e-8f)
						{
							kinematicTargetResidualPending = true;
							break;
						}
					}
				}
				if(kinematicTargetResidualPending ||
					componentResidualPending)
				{
					core.wakeCounter = PxMax(
						core.wakeCounter,
						ScInternalWakeCounterResetValue);
					continue;
				}
				const PxReal sleepThreshold =
					PxMax(core.sleepThreshold, 0.0f);
				if(maxSpeedSquared >
					sleepThreshold * sleepThreshold)
				{
					core.wakeCounter = PxMax(
						core.wakeCounter,
						ScInternalWakeCounterResetValue);
					continue;
				}
				if(hasUnforcedRestBendingResidual(entry))
				{
					core.wakeCounter = PxMax(
						core.wakeCounter,
						ScInternalWakeCounterResetValue);
					continue;
				}

				core.wakeCounter = PxMax(
					core.wakeCounter - dt, 0.0f);
				if(core.wakeCounter == 0.0f)
					sleepEntry(entry);
			}
		}

		void clearIslandSelectionStorages()
		{
			for(PxU32 i = 0; i < mIslandSelectionStorages.size(); i++)
			{
				IslandSelectionStorage* storage =
					mIslandSelectionStorages[i];
				if(storage)
				{
					storage->~IslandSelectionStorage();
					PX_FREE(storage);
				}
			}
			mIslandSelectionStorages.clear();
		}

		void invalidateNativeIslandSelectionCaches()
		{
			for(PxU32 storageIndex = 0;
				storageIndex < mIslandSelectionStorages.size();
				storageIndex++)
			{
				IslandSelectionStorage* const storage =
					mIslandSelectionStorages[storageIndex];
				if(!storage)
					continue;
				// This is deliberately unconditional.  A build can fail after it
				// marked its storage untouched, so `touched` is not a lifetime
				// predicate for the local persistent contact stream.
				storage->contacts.clear();
				storage->particleBodyIndices.clear();
				storage->contactStarts.clear();
				storage->contactCounts.clear();
				storage->contactRefs.clear();
				storage->triangleCoreSafetyStarts.clear();
				storage->triangleCoreSafetyCounts.clear();
				storage->triangleCoreSafetyRefs.clear();
				storage->ogcPairContactStarts.clear();
				storage->ogcPairContactCounts.clear();
				storage->ogcPairContactRefs.clear();
				storage->rigidTargetContactStarts.clear();
				storage->rigidTargetContactCounts.clear();
				storage->rigidTargetContactRefs.clear();
				storage->softCores.clear();
				storage->touched = false;
				storage->selectedIsland = PX_MAX_U32;
				storage->entryIndices.clear();
			}
			mDynamicsOwnsStep = false;
			mDynamicsSelectedEntryCount = 0;
		}

		IslandSelectionStorage* acquireIslandSelectionStorage(
			IG::IslandId nativeIslandId)
		{
			for(PxU32 i = 0; i < mIslandSelectionStorages.size(); i++)
			{
				IslandSelectionStorage* storage =
					mIslandSelectionStorages[i];
				if(storage->touched &&
					storage->nativeIslandId == nativeIslandId)
					return storage;
			}
			for(PxU32 i = 0; i < mIslandSelectionStorages.size(); i++)
			{
				IslandSelectionStorage* storage =
					mIslandSelectionStorages[i];
				if(!storage->touched)
				{
					if(storage->nativeIslandId != nativeIslandId)
					{
						storage->contacts.clear();
						storage->softCores.clear();
					}
					storage->nativeIslandId = nativeIslandId;
					storage->touched = true;
					return storage;
				}
			}

			void* memory = PX_ALLOC(
				sizeof(IslandSelectionStorage),
				"AVBD CPU soft island selection storage");
			IslandSelectionStorage* storage = memory
				? PX_PLACEMENT_NEW(
					memory, IslandSelectionStorage)()
				: NULL;
			if(!storage)
				return NULL;
			storage->nativeIslandId = nativeIslandId;
			storage->touched = true;
			mIslandSelectionStorages.pushBack(storage);
			return storage;
		}

		static bool rebaseParticleIndex(
			PxU32& index, PxU32 globalStart,
			PxU32 particleCount, PxU32 localStart)
		{
			if(index == PX_MAX_U32)
				return true;
			if(index < globalStart ||
				index - globalStart >= particleCount)
				return false;
			index = localStart + (index - globalStart);
			return true;
		}

		static bool copyAndRebaseSoftBody(
			const Dy::AvbdSoftBody& source,
			PxU32 globalStart, PxU32 particleCount,
			PxU32 localStart,
			Dy::AvbdSoftBody& destination)
		{
			if(source.compiled.particleStart != globalStart ||
				source.compiled.particleCount != particleCount ||
				!source.runtime.attachments.empty())
				return false;
			destination = source;
			// Native selection appends island-local attachments immediately after
			// this rebase and compiles the objective program once from that final
			// state.  Do not compile an intermediate program that cannot be
			// consumed before it is invalidated again.
			return rebaseSoftBodyParticleRangeInPlace(
				destination, globalStart, particleCount, localStart, false);
		}

		static bool rebaseSoftBodyParticleRangeInPlace(
			Dy::AvbdSoftBody& body,
			PxU32 oldStart, PxU32 particleCount,
			PxU32 newStart, bool compileObjectives = true)
		{
			if(body.compiled.particleStart != oldStart ||
				body.compiled.particleCount != particleCount)
				return false;
			body.compiled.particleStart = newStart;
			for(PxU32 i = 0;
				i < body.compiled.triElements.size(); i++)
			{
				Dy::AvbdTriElement& element =
					body.compiled.triElements[i];
				if(!rebaseParticleIndex(
					element.p0, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p1, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p2, oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.tetElements.size(); i++)
			{
				Dy::AvbdTetElement& element =
					body.compiled.tetElements[i];
				if(!rebaseParticleIndex(
					element.p0, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p1, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p2, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p3, oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.bendElements.size(); i++)
			{
				Dy::AvbdBendingElement& element =
					body.compiled.bendElements[i];
				if(!rebaseParticleIndex(
					element.opp0, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.opp1, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.edgeStart, oldStart,
						particleCount, newStart) ||
					!rebaseParticleIndex(
						element.edgeEnd, oldStart,
						particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0; i < body.compiled.edges.size(); i++)
			{
				Dy::AvbdEdgeInfo& edge = body.compiled.edges[i];
				if(!rebaseParticleIndex(
					edge.p0, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						edge.p1, oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.surfaceTriangles.size(); i++)
			{
				if(!rebaseParticleIndex(
					body.compiled.surfaceTriangles[i],
					oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.surfaceVertices.size(); i++)
			{
				if(!rebaseParticleIndex(
					body.compiled.surfaceVertices[i],
					oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.surfaceEdges.size(); i++)
			{
				Dy::AvbdEdgeInfo& edge =
					body.compiled.surfaceEdges[i];
				if(!rebaseParticleIndex(
						edge.p0, oldStart,
						particleCount, newStart) ||
					!rebaseParticleIndex(
						edge.p1, oldStart,
						particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.runtime.attachments.size(); i++)
			{
				Dy::AvbdSoftPoint& point =
					body.runtime.attachments[i].point;
				for(PxU32 endpoint = 0;
					endpoint < point.particleCount; endpoint++)
				{
					if(!rebaseParticleIndex(
						point.particleIndices[endpoint],
						oldStart, particleCount, newStart))
						return false;
				}
			}
			for(PxU32 i = 0; i < body.runtime.pins.size(); i++)
			{
				Dy::AvbdSoftPoint& point =
					body.runtime.pins[i].point;
				for(PxU32 endpoint = 0;
					endpoint < point.particleCount; endpoint++)
				{
					if(!rebaseParticleIndex(
						point.particleIndices[endpoint],
						oldStart, particleCount, newStart))
						return false;
				}
			}
			if(!compileObjectives)
				return true;
			body.runtime.compileObjectiveProgram(
				newStart, particleCount);
			return body.runtime.isObjectiveProgramCurrent(
				newStart, particleCount);
		}

		bool getCanonicalIslandParticleRange(
			const IslandSelectionStorage& storage,
			PxU32& particleStart,
			PxU32& particleCount) const
		{
			particleStart = PX_MAX_U32;
			particleCount = 0;
			for(PxU32 entryOrder = 0;
				entryOrder < storage.entryIndices.size(); entryOrder++)
			{
				const PxU32 entryIndex = storage.entryIndices[entryOrder];
				if(entryIndex >= mEntries.size())
					return false;
				const Entry& entry = mEntries[entryIndex];
				const PxU32 entryParticleStart =
					getParticleStart(entry);
				const PxU32 entryParticleCount =
					getParticleCount(entry);
				if(entryParticleStart > mParticles.size() ||
					entryParticleCount >
						mParticles.size() - entryParticleStart)
					return false;
				if(particleStart == PX_MAX_U32)
					particleStart = entryParticleStart;
				if(entryParticleStart != particleStart + particleCount ||
					entryParticleCount > PX_MAX_U32 - particleCount)
					return false;
				particleCount += entryParticleCount;
			}
			return particleStart != PX_MAX_U32 && particleCount > 0;
		}

		Dy::AvbdSoftParticle* getIslandSelectionParticles(
			IslandSelectionStorage& storage)
		{
			if(storage.usesCanonicalParticleRange)
			{
				PX_ASSERT(
					storage.canonicalParticleStart <= mParticles.size() &&
					storage.canonicalParticleCount <=
						mParticles.size() -
						storage.canonicalParticleStart);
				return mParticles.begin() +
					storage.canonicalParticleStart;
			}
			return storage.particles.begin();
		}

		PxU32 getIslandSelectionParticleCount(
			const IslandSelectionStorage& storage) const
		{
			return storage.usesCanonicalParticleRange
				? storage.canonicalParticleCount
				: storage.particles.size();
		}

		bool compileIslandSelectionExecutionPlan(
			IslandSelectionStorage& storage,
			PxU32 numParticles, PxU32 numRigidBodies)
		{
			if(numParticles == 0 ||
				storage.bodies.size() != storage.entryIndices.size())
				return false;
			storage.ogcPairContactStarts.clear();
			storage.ogcPairContactCounts.clear();
			storage.ogcPairContactRefs.clear();

			storage.particleBodyIndices.resize(numParticles);
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				storage.particleBodyIndices[particleIndex] = PX_MAX_U32;
			for(PxU32 bodyIndex = 0;
				bodyIndex < storage.bodies.size(); bodyIndex++)
			{
				const Dy::AvbdSoftBody& body = storage.bodies[bodyIndex];
				const PxU32 particleStart =
					body.compiled.particleStart;
				const PxU32 particleCount =
					body.compiled.particleCount;
				if(particleStart > numParticles ||
					particleCount > numParticles - particleStart)
					return false;
				for(PxU32 localParticleIndex = 0;
					localParticleIndex < particleCount;
					localParticleIndex++)
				{
					const PxU32 particleIndex =
						particleStart + localParticleIndex;
					if(storage.particleBodyIndices[particleIndex] !=
						PX_MAX_U32)
						return false;
					storage.particleBodyIndices[particleIndex] = bodyIndex;
				}
			}
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				if(storage.particleBodyIndices[particleIndex] ==
					PX_MAX_U32)
					return false;

			// This is the same stable, source-ordered CSR used by the component
			// soft path.  Compiling it at the provider boundary lets the native
			// mixed solver consume one immutable execution program instead of
			// rebuilding ownership and incidence from raw contact rows.
			storage.contactStarts.resize(numParticles + 1);
			storage.contactCounts.resize(numParticles);
			// Every dynamic rigid endpoint needs the same immutable contact
			// ownership program as its soft counterpart.  Omitting the mirror for a
			// one-body island saved a tiny setup pass, but it also made the sole 6DOF
			// rigid block bypass the shared OGC pair state.  Keep the CSR for one
			// body as well: the O(contact) setup is bounded and makes mixed OGC
			// scheduling independent of island cardinality.
			const bool mayBuildRigidTargetContactPlan =
				numRigidBodies > 0 && !storage.contacts.empty();
			PxU32 rigidTargetContactCount = 0;
			storage.rigidTargetContactStarts.clear();
			storage.rigidTargetContactCounts.clear();
			storage.rigidTargetContactRefs.clear();
			if(mayBuildRigidTargetContactPlan)
			{
				storage.rigidTargetContactCounts.resize(numRigidBodies);
				for(PxU32 rigidBodyIndex = 0;
					rigidBodyIndex < numRigidBodies; rigidBodyIndex++)
					storage.rigidTargetContactCounts[rigidBodyIndex] = 0;
			}
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				storage.contactCounts[particleIndex] = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < storage.contacts.size(); contactIndex++)
			{
				const Dy::AvbdSoftContactGeometry& geometry =
					storage.contacts[contactIndex].geometry;
				if(mayBuildRigidTargetContactPlan &&
					geometry.hasRigidBodyTarget() &&
					geometry.targetIndex < numRigidBodies)
				{
					storage.rigidTargetContactCounts[
						geometry.targetIndex]++;
					rigidTargetContactCount++;
				}
				PxU32 particleIndices[
					Dy::AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleIndexCount =
					Dy::avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				for(PxU32 particleOffset = 0;
					particleOffset < particleIndexCount;
					particleOffset++)
				{
					const PxU32 particleIndex =
						particleIndices[particleOffset];
					if(particleIndex >= numParticles)
						continue;
					if(PxAbs(
						Dy::avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex)) > 1e-12f)
						storage.contactCounts[particleIndex]++;
				}
			}
			storage.contactStarts[0] = 0;
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				storage.contactStarts[particleIndex + 1] =
					storage.contactStarts[particleIndex] +
					storage.contactCounts[particleIndex];
			storage.contactRefs.resize(
				storage.contactStarts[numParticles]);

			// Reuse the two source-order contact passes required by the particle
			// CSR.  The rigid mirror is published only when it cuts at least half
			// the repeated body/contact visits; otherwise direct scans retain the
			// lower setup cost.  This keeps one-body dense dynamic contact on its
			// existing fast path.
			const PxU64 legacyRigidContactVisits =
				PxU64(numRigidBodies) * PxU64(storage.contacts.size());
			const bool publishRigidTargetContactPlan =
				mayBuildRigidTargetContactPlan &&
				PxU64(rigidTargetContactCount) * 2u <=
					legacyRigidContactVisits;
			if(publishRigidTargetContactPlan)
			{
				storage.rigidTargetContactStarts.resize(numRigidBodies + 1);
				storage.rigidTargetContactStarts[0] = 0;
				for(PxU32 rigidBodyIndex = 0;
					rigidBodyIndex < numRigidBodies; rigidBodyIndex++)
					storage.rigidTargetContactStarts[rigidBodyIndex + 1] =
						storage.rigidTargetContactStarts[rigidBodyIndex] +
						storage.rigidTargetContactCounts[rigidBodyIndex];
				storage.rigidTargetContactRefs.resize(
					storage.rigidTargetContactStarts[numRigidBodies]);
				for(PxU32 rigidBodyIndex = 0;
					rigidBodyIndex < numRigidBodies; rigidBodyIndex++)
					storage.rigidTargetContactCounts[rigidBodyIndex] = 0;
			}
			else
				storage.rigidTargetContactCounts.clear();
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				storage.contactCounts[particleIndex] = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < storage.contacts.size(); contactIndex++)
			{
				const Dy::AvbdSoftContactGeometry& geometry =
					storage.contacts[contactIndex].geometry;
				if(publishRigidTargetContactPlan &&
					geometry.hasRigidBodyTarget() &&
					geometry.targetIndex < numRigidBodies)
				{
					const PxU32 rigidBodyIndex = geometry.targetIndex;
					storage.rigidTargetContactRefs[
						storage.rigidTargetContactStarts[rigidBodyIndex] +
						storage.rigidTargetContactCounts[rigidBodyIndex]++] =
							contactIndex;
				}
				PxU32 particleIndices[
					Dy::AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleIndexCount =
					Dy::avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				for(PxU32 particleOffset = 0;
					particleOffset < particleIndexCount;
					particleOffset++)
				{
					const PxU32 particleIndex =
						particleIndices[particleOffset];
					if(particleIndex >= numParticles)
						continue;
					const PxReal jacobianScale =
						Dy::avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex);
					if(PxAbs(jacobianScale) <= 1e-12f)
						continue;
					storage.contactRefs[
						storage.contactStarts[particleIndex] +
						storage.contactCounts[particleIndex]++] =
							Dy::AvbdSoftContactParticleRef(
								contactIndex, jacobianScale);
				}
			}

			// Compile a second, geometry-only incidence program for triangle/OBB
			// core rows.  The normal contact CSR above deliberately contains only
			// the compact AL query support.  OGC admission, however, has to see the
			// independent embeddings of all three collision-triangle vertices or a
			// material update can move an unreferenced corner through a box.
			const PxU32 maxCoreSafetyParticles =
				3u * Dy::AVBD_CONTACT_POINT_MAX_SUPPORT;
			auto collectTriangleCoreSafetyParticles =
				[&](const Dy::AvbdSoftContactGeometry& geometry,
					PxU32* particleIndices) -> PxU32
			{
				if(!geometry.hasRigidBoxTriangleCoreExit ||
					(!geometry.hasRigidBodyTarget() &&
					 !geometry.hasWorldStaticTarget()))
					return 0;
				PxU32 count = 0;
				for(PxU32 vertex = 0; vertex < 3; ++vertex)
				{
					const Dy::AvbdWeightedContactPoint& point =
						geometry.rigidBoxTriangleCorePoints[vertex];
					if(point.count == 0 ||
						point.count > Dy::AVBD_CONTACT_POINT_MAX_SUPPORT)
						return PX_MAX_U32;
					for(PxU32 support = 0; support < point.count; ++support)
					{
						const PxU32 particleIndex =
							point.particleIndices[support];
						if(particleIndex >= numParticles)
							return PX_MAX_U32;
						bool duplicate = false;
						for(PxU32 prior = 0; prior < count; ++prior)
							duplicate |= particleIndices[prior] == particleIndex;
						if(!duplicate)
						{
							if(count >= maxCoreSafetyParticles)
								return PX_MAX_U32;
							particleIndices[count++] = particleIndex;
						}
					}
				}
				return count;
			};

			storage.triangleCoreSafetyStarts.resize(numParticles + 1);
			storage.triangleCoreSafetyCounts.resize(numParticles);
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; ++particleIndex)
				storage.triangleCoreSafetyCounts[particleIndex] = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < storage.contacts.size(); ++contactIndex)
			{
				PxU32 coreParticles[3 * Dy::AVBD_CONTACT_POINT_MAX_SUPPORT];
				const PxU32 coreParticleCount =
					collectTriangleCoreSafetyParticles(
						storage.contacts[contactIndex].geometry, coreParticles);
				if(coreParticleCount == PX_MAX_U32)
					continue;
				for(PxU32 coreParticle = 0;
					coreParticle < coreParticleCount; ++coreParticle)
					storage.triangleCoreSafetyCounts[
						coreParticles[coreParticle]]++;
			}
			storage.triangleCoreSafetyStarts[0] = 0;
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; ++particleIndex)
				storage.triangleCoreSafetyStarts[particleIndex + 1] =
					storage.triangleCoreSafetyStarts[particleIndex] +
					storage.triangleCoreSafetyCounts[particleIndex];
			storage.triangleCoreSafetyRefs.resize(
				storage.triangleCoreSafetyStarts[numParticles]);
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; ++particleIndex)
				storage.triangleCoreSafetyCounts[particleIndex] = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < storage.contacts.size(); ++contactIndex)
			{
				PxU32 coreParticles[3 * Dy::AVBD_CONTACT_POINT_MAX_SUPPORT];
				const PxU32 coreParticleCount =
					collectTriangleCoreSafetyParticles(
						storage.contacts[contactIndex].geometry, coreParticles);
				if(coreParticleCount == PX_MAX_U32)
					continue;
				for(PxU32 coreParticle = 0;
					coreParticle < coreParticleCount; ++coreParticle)
				{
					const PxU32 particleIndex = coreParticles[coreParticle];
					storage.triangleCoreSafetyRefs[
						storage.triangleCoreSafetyStarts[particleIndex] +
						storage.triangleCoreSafetyCounts[particleIndex]++] =
						Dy::AvbdSoftContactParticleRef(contactIndex, 1.0f);
				}
			}

			// Compile the inverse of `ogcPairIndices` once at the Scene/provider
			// boundary. Pair-owned OGC stages consume this stable source-order CSR
			// directly, rather than scanning every contact for every active pair.
			// If a provider ever publishes an incomplete pair map, drop this
			// optional acceleration structure and retain the solver's validated
			// direct-scan fallback.
			if(!storage.ogcPairStates.empty() &&
				storage.ogcPairIndices.size() == storage.contacts.size())
			{
				bool validPairMap = true;
				storage.ogcPairContactCounts.resize(
					storage.ogcPairStates.size());
				for(PxU32 pairIndex = 0;
					pairIndex < storage.ogcPairContactCounts.size(); ++pairIndex)
					storage.ogcPairContactCounts[pairIndex] = 0;
				for(PxU32 contactIndex = 0;
					contactIndex < storage.ogcPairIndices.size(); ++contactIndex)
				{
					const PxU32 pairIndex = storage.ogcPairIndices[contactIndex];
					if(pairIndex == PX_MAX_U32)
						continue;
					if(pairIndex >= storage.ogcPairStates.size())
					{
						validPairMap = false;
						break;
					}
					storage.ogcPairContactCounts[pairIndex]++;
				}
				if(validPairMap)
				{
					storage.ogcPairContactStarts.resize(
						storage.ogcPairStates.size() + 1);
					storage.ogcPairContactStarts[0] = 0;
					for(PxU32 pairIndex = 0;
						pairIndex < storage.ogcPairStates.size(); ++pairIndex)
						storage.ogcPairContactStarts[pairIndex + 1] =
							storage.ogcPairContactStarts[pairIndex] +
							storage.ogcPairContactCounts[pairIndex];
					storage.ogcPairContactRefs.resize(
						storage.ogcPairContactStarts[
							storage.ogcPairStates.size()]);
					for(PxU32 pairIndex = 0;
						pairIndex < storage.ogcPairContactCounts.size(); ++pairIndex)
						storage.ogcPairContactCounts[pairIndex] = 0;
					for(PxU32 contactIndex = 0;
						contactIndex < storage.ogcPairIndices.size(); ++contactIndex)
					{
						const PxU32 pairIndex = storage.ogcPairIndices[contactIndex];
						if(pairIndex == PX_MAX_U32)
							continue;
						storage.ogcPairContactRefs[
							storage.ogcPairContactStarts[pairIndex] +
							storage.ogcPairContactCounts[pairIndex]++] = contactIndex;
					}
				}
				else
					storage.ogcPairContactCounts.clear();
			}
			return true;
		}

		bool findRigidBodyIndexInIsland(
			BodyCore& rigidCore,
			PxsRigidBody* const* rigidBodies,
			const Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxU32& localBodyIndex) const
		{
			BodySim* bodySim = rigidCore.getSim();
			if(!bodySim || bodySim->isKinematic() ||
				bodySim->isArticulationLink())
				return false;
			const PxsRigidBody* lowLevelBody =
				&bodySim->getLowLevelBody();
			for(PxU32 i = 0; i < bodyCount; i++)
			{
				const PxU32 globalBodyIndex = bodyStart + i;
				if(rigidBodies[globalBodyIndex] != lowLevelBody)
					continue;
				if(solverBodies[globalBodyIndex].isStatic())
					return false;
				localBodyIndex = i;
				return true;
			}
			return false;
		}

		bool findArticulationBodyIndexInIsland(
			BodyCore& linkCore,
			Dy::FeatherstoneArticulation* const*
				articulationForBody,
			const PxU32* linkIndexForBody,
			const Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxU32& localBodyIndex) const
		{
			BodySim* bodySim = linkCore.getSim();
			if(!bodySim || !bodySim->isArticulationLink() ||
				!bodySim->getArticulation())
				return false;
			for(PxU32 i = 0; i < bodyCount; i++)
			{
				const PxU32 globalBodyIndex = bodyStart + i;
				Dy::FeatherstoneArticulation* articulation =
					articulationForBody[globalBodyIndex];
				if(!articulation ||
					articulation != bodySim->getArticulation())
					continue;
				const PxU32 linkIndex =
					linkIndexForBody[globalBodyIndex];
				const Dy::ArticulationData& data =
					articulation->getArticulationData();
				if(linkIndex >= data.getLinkCount() ||
					data.getLink(linkIndex).bodyCore !=
						&linkCore.getCore())
					continue;
				if(solverBodies[globalBodyIndex].isStatic())
					return false;
				localBodyIndex = i;
				return true;
			}
			return false;
		}

		RigidAttachmentEntry* findRigidAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mRigidAttachments.size(); i++)
			{
				RigidAttachmentEntry& attachment =
					mRigidAttachments[i];
				if(attachment.softCore == &softCore &&
					attachment.handle == handle)
					return &attachment;
			}
			return NULL;
		}

		ArticulationAttachmentEntry*
			findArticulationAttachment(
				ActorCore& softCore,
				PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mArticulationAttachments.size(); i++)
			{
				ArticulationAttachmentEntry& attachment =
					mArticulationAttachments[i];
				if(attachment.softCore == &softCore &&
					attachment.handle == handle)
					return &attachment;
			}
			return NULL;
		}

		SoftPairAttachmentEntry* findSoftPairAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mSoftPairAttachments.size(); i++)
			{
				SoftPairAttachmentEntry& attachment =
					mSoftPairAttachments[i];
				if(attachment.softCore[0] == &softCore &&
					attachment.handle == handle)
					return &attachment;
			}
			return NULL;
		}

		PrescribedAttachmentEntry* findPrescribedAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				PrescribedAttachmentEntry& attachment =
					mPrescribedAttachments[i];
				if(attachment.softCore == &softCore &&
					attachment.handle == handle)
					return &attachment;
			}
			return NULL;
		}

		bool buildIslandSelectionStorage(
			IslandSelectionStorage& storage,
			Dy::AvbdSolverBody* solverBodies,
			PxsRigidBody* const* rigidBodies,
			Dy::FeatherstoneArticulation* const*
				articulationForBody,
			const PxU32* linkIndexForBody,
			PxU32 bodyStart, PxU32 bodyCount,
			PxReal dt, const PxVec3& gravity)
		{
			bool membershipMatches =
				storage.softCores.size() == storage.entryIndices.size();
			for(PxU32 i = 0;
				membershipMatches && i < storage.entryIndices.size(); i++)
			{
				membershipMatches =
					storage.softCores[i] ==
					mEntries[storage.entryIndices[i]].
						getActorCore();
			}
			if(!membershipMatches)
				storage.contacts.clear();
			storage.softCores.clear();
			storage.usesCanonicalParticleRange = false;
			storage.canonicalParticleStart = PX_MAX_U32;
			storage.canonicalParticleCount = 0;
			storage.globalParticleIndices.clear();
			storage.particles.clear();
			storage.bodies.clear();
			storage.particleBodyIndices.clear();
			storage.contactStarts.clear();
			storage.contactCounts.clear();
			storage.contactRefs.clear();
			storage.triangleCoreSafetyStarts.clear();
			storage.triangleCoreSafetyCounts.clear();
			storage.triangleCoreSafetyRefs.clear();
			storage.rigidTargetContactStarts.clear();
			storage.rigidTargetContactCounts.clear();
			storage.rigidTargetContactRefs.clear();
			storage.selfCollisionAdjacencies.clear();
			storage.selfCollisionEnabled.clear();
			storage.rigidBoxes.clear();
			storage.selectedDynamicBoxes.clear();
			storage.terminalCollisionBodies.clear();
			storage.terminalCollisionVertexMappings.clear();
			storage.ogcPairStates.clear();
			storage.ogcPairIndices.clear();
			storage.rigidSpheres.clear();
			storage.selectedDynamicSpheres.clear();
			storage.rigidCapsules.clear();
			storage.selectedDynamicCapsules.clear();
			storage.rigidConvexes.clear();
			storage.selectedDynamicConvexes.clear();
			storage.probeContacts.clear();

			PxU32 canonicalParticleStart = PX_MAX_U32;
			PxU32 canonicalParticleCount = 0;
			if(getCanonicalIslandParticleRange(
				storage, canonicalParticleStart,
				canonicalParticleCount))
			{
				storage.usesCanonicalParticleRange = true;
				storage.canonicalParticleStart =
					canonicalParticleStart;
				storage.canonicalParticleCount =
					canonicalParticleCount;
			}

			for(PxU32 entryOrder = 0;
				entryOrder < storage.entryIndices.size(); entryOrder++)
			{
				const Entry& entry =
					mEntries[storage.entryIndices[entryOrder]];
				storage.softCores.pushBack(entry.getActorCore());
				const PxU32 particleStart =
					getParticleStart(entry);
				const PxU32 particleCount =
					getParticleCount(entry);
				const PxU32 localStart =
					storage.usesCanonicalParticleRange
						? particleStart -
							storage.canonicalParticleStart
						: storage.particles.size();
				if(!storage.usesCanonicalParticleRange)
				{
					for(PxU32 i = 0; i < particleCount; i++)
					{
						const PxU32 globalIndex = particleStart + i;
						if(globalIndex >= mParticles.size())
							return false;
						storage.globalParticleIndices.pushBack(globalIndex);
						storage.particles.pushBack(
							mParticles[globalIndex]);
					}
				}
				if(entry.bodyIndex >= mBodies.size())
					return false;
				if(entry.bodyIndex >=
					mSelfCollisionAdjacencies.size())
					return false;
				Dy::AvbdSoftBody localBody;
				if(!copyAndRebaseSoftBody(
					mBodies[entry.bodyIndex],
					particleStart, particleCount,
					localStart, localBody))
					return false;
				storage.bodies.pushBack(localBody);
				storage.selfCollisionAdjacencies.pushBack(
					mSelfCollisionAdjacencies[entry.bodyIndex]);
				storage.selfCollisionEnabled.pushBack(
					(entry.getBodyCore().bodyFlags &
						PxDeformableBodyFlag::
							eDISABLE_SELF_COLLISION)
					? 0u : 1u);
			}
			// Contact selection happens before the unified AVBD solve's
			// prediction stage. Publish the same current-frame soft prediction
			// now so the non-CCD endpoint-DCD path can select endpoint OGC
			// objectives; a validated native execution plan transfers ownership of
			// this pass to the solver instead of recomputing it before iteration.
			Dy::AvbdSoftParticle* const selectionParticles =
				getIslandSelectionParticles(storage);
			const PxU32 selectionParticleCount =
				getIslandSelectionParticleCount(storage);
			for(PxU32 particleIndex = 0;
				particleIndex < selectionParticleCount; particleIndex++)
				selectionParticles[particleIndex].computePrediction(
					dt, gravity);

			bool hasRigidAttachment = false;
			bool hasArticulationAttachment = false;
			bool hasSoftPairAttachment = false;
			for(PxU32 entryOrder = 0;
				entryOrder < storage.entryIndices.size(); entryOrder++)
			{
				const Entry& entry =
					mEntries[storage.entryIndices[entryOrder]];
				Dy::AvbdSoftBody& localBody =
					storage.bodies[entryOrder];
				for(PxU32 attachmentIndex = 0;
					attachmentIndex < mRigidAttachments.size();
					attachmentIndex++)
				{
					const RigidAttachmentEntry& source =
						mRigidAttachments[attachmentIndex];
					if(source.softCore != entry.getActorCore() ||
						!Dy::avbdIsSoftPointValid(
							source.localPoint, 0,
							localBody.compiled.particleCount))
						continue;
					PxU32 localRigidBodyIndex = PX_MAX_U32;
					if(!findRigidBodyIndexInIsland(
						*source.rigidCore, rigidBodies,
						solverBodies, bodyStart, bodyCount,
						localRigidBodyIndex))
						return false;

					Dy::AvbdSoftAttachment attachment;
					attachment.point = source.localPoint;
					for(PxU32 endpoint = 0;
						endpoint < attachment.point.particleCount;
						endpoint++)
					{
						attachment.point.
							particleIndices[endpoint] +=
								localBody.compiled.particleStart;
					}
					attachment.rigidBodyIdx =
						localRigidBodyIndex;
					attachment.sourceHandle = source.handle;
					attachment.targetKind =
						Dy::AvbdSoftAttachmentTargetKind::
							eDYNAMIC_RIGID;
					attachment.localOffset =
						source.rigidCore->getBody2Actor().
							getInverse().transform(
								source.actorLocalTarget);
					attachment.alLambda = source.alLambda;
					attachment.k = source.k;
					attachment.kMax = source.kMax;
					localBody.runtime.attachments.pushBack(
						attachment);
					hasRigidAttachment = true;
				}
				for(PxU32 attachmentIndex = 0;
					attachmentIndex <
						mArticulationAttachments.size();
					attachmentIndex++)
				{
					const ArticulationAttachmentEntry& source =
						mArticulationAttachments[
							attachmentIndex];
					if(source.softCore != entry.getActorCore() ||
						!Dy::avbdIsSoftPointValid(
							source.localPoint, 0,
							localBody.compiled.particleCount))
						continue;
					PxU32 localLinkBodyIndex = PX_MAX_U32;
					if(!findArticulationBodyIndexInIsland(
						*source.linkCore, articulationForBody,
						linkIndexForBody, solverBodies,
						bodyStart, bodyCount,
						localLinkBodyIndex))
						return false;

					Dy::AvbdSoftAttachment attachment;
					attachment.point = source.localPoint;
					for(PxU32 endpoint = 0;
						endpoint <
							attachment.point.particleCount;
						endpoint++)
						attachment.point.
							particleIndices[endpoint] +=
							localBody.compiled.particleStart;
					attachment.rigidBodyIdx =
						localLinkBodyIndex;
					attachment.sourceHandle = source.handle;
					attachment.targetKind =
						Dy::AvbdSoftAttachmentTargetKind::
							eARTICULATION_LINK;
					attachment.localOffset =
						source.linkCore->getBody2Actor().
							getInverse().transform(
								source.actorLocalTarget);
					attachment.alLambda = source.alLambda;
					attachment.k = source.k;
					attachment.kMax = source.kMax;
					localBody.runtime.attachments.pushBack(
						attachment);
					hasArticulationAttachment = true;
				}
				for(PxU32 attachmentIndex = 0;
					attachmentIndex < mSoftPairAttachments.size();
					attachmentIndex++)
				{
					const SoftPairAttachmentEntry& source =
						mSoftPairAttachments[attachmentIndex];
					if(source.softCore[0] != entry.getActorCore())
						continue;
					PxU32 targetEntryOrder = PX_MAX_U32;
					for(PxU32 candidate = 0;
						candidate < storage.softCores.size();
						candidate++)
					{
						if(storage.softCores[candidate] ==
							source.softCore[1])
						{
							targetEntryOrder = candidate;
							break;
						}
					}
					if(targetEntryOrder == PX_MAX_U32 ||
						targetEntryOrder >= storage.bodies.size())
						return false;
					const Dy::AvbdSoftBody& targetBody =
						storage.bodies[targetEntryOrder];
					if(!Dy::avbdIsSoftPointValid(
							source.localPoint[0], 0,
							localBody.compiled.particleCount) ||
						!Dy::avbdIsSoftPointValid(
							source.localPoint[1], 0,
							targetBody.compiled.particleCount))
						return false;

					Dy::AvbdSoftAttachment attachment;
					attachment.point = source.localPoint[0];
					for(PxU32 endpoint = 0;
						endpoint < attachment.point.particleCount;
						endpoint++)
						attachment.point.
							particleIndices[endpoint] +=
							localBody.compiled.particleStart;
					attachment.targetPoint = source.localPoint[1];
					for(PxU32 endpoint = 0;
						endpoint <
							attachment.targetPoint.particleCount;
						endpoint++)
						attachment.targetPoint.
							particleIndices[endpoint] +=
							targetBody.compiled.particleStart;
					attachment.rigidBodyIdx = PX_MAX_U32;
					attachment.sourceHandle = source.handle;
					attachment.targetKind =
						Dy::AvbdSoftAttachmentTargetKind::
							eDYNAMIC_SOFT;
					attachment.alLambda = source.alLambda;
					attachment.k = source.k;
					attachment.kMax = source.kMax;
					localBody.runtime.attachments.pushBack(
						attachment);
					hasSoftPairAttachment = true;
				}
				localBody.runtime.compileObjectiveProgram(
					localBody.compiled.particleStart,
					localBody.compiled.particleCount);
				if(!localBody.runtime.isObjectiveProgramCurrent(
					localBody.compiled.particleStart,
					localBody.compiled.particleCount))
					return false;
			}

			compileDynamicBoxesForIsland(
				rigidBodies, solverBodies, bodyStart, bodyCount,
				storage.selectedDynamicBoxes);
			compileDynamicSpheresForIsland(
				rigidBodies, solverBodies, bodyStart, bodyCount,
				dt, gravity, storage.selectedDynamicSpheres);
			compileDynamicCapsulesForIsland(
				rigidBodies, solverBodies, bodyStart, bodyCount,
				dt, gravity, storage.selectedDynamicCapsules);
			compileDynamicConvexesForIsland(
				rigidBodies, solverBodies, bodyStart, bodyCount,
				dt, gravity,
				storage.selectedDynamicConvexes);
			if(!storage.selectedDynamicBoxes.empty())
			{
				Dy::avbdDetectSoftRigidSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicBoxes.begin(),
					storage.selectedDynamicBoxes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					NULL, 0,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidSweptSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicBoxes.begin(),
					storage.selectedDynamicBoxes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicBoxes.begin(),
					storage.selectedDynamicBoxes.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
			}
			if(!storage.selectedDynamicSpheres.empty())
			{
				Dy::avbdDetectSoftRigidSphereSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidSphereSweptSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidSphereSweptOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
				Dy::avbdDetectSoftRigidSphereOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
			}
			if(!storage.selectedDynamicCapsules.empty())
			{
				Dy::avbdDetectSoftRigidCapsuleSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidCapsuleSweptSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidCapsuleSweptOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
				Dy::avbdDetectSoftRigidCapsuleOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
			}
			if(!storage.selectedDynamicConvexes.empty())
			{
				Dy::avbdDetectSoftRigidConvexSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidConvexSweptSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidConvexSweptOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
				Dy::avbdDetectSoftRigidConvexOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
			}
			// probeContacts is only a fast-path hint. A distinct public collision
			// mesh may overlap even when the simulation tetrahedra used by this
			// legacy probe do not; the authoritative proxy redetection below owns
			// the final island-selection decision.

			for(PxU32 i = 0; i < mRigidBoxes.size(); i++)
				storage.rigidBoxes.pushBack(mRigidBoxes[i]);
			for(PxU32 i = 0;
				i < storage.selectedDynamicBoxes.size(); i++)
				storage.rigidBoxes.pushBack(
					storage.selectedDynamicBoxes[i]);
			for(PxU32 i = 0; i < mRigidSpheres.size(); i++)
				storage.rigidSpheres.pushBack(mRigidSpheres[i]);
			for(PxU32 i = 0;
				i < storage.selectedDynamicSpheres.size(); i++)
				storage.rigidSpheres.pushBack(
					storage.selectedDynamicSpheres[i]);
			for(PxU32 i = 0; i < mRigidCapsules.size(); i++)
				storage.rigidCapsules.pushBack(mRigidCapsules[i]);
			for(PxU32 i = 0;
				i < storage.selectedDynamicCapsules.size(); i++)
				storage.rigidCapsules.pushBack(
					storage.selectedDynamicCapsules[i]);
			for(PxU32 i = 0; i < mRigidConvexes.size(); i++)
				storage.rigidConvexes.pushBack(
					mRigidConvexes[i]);
			for(PxU32 i = 0;
				i < storage.selectedDynamicConvexes.size(); i++)
				storage.rigidConvexes.pushBack(
					storage.selectedDynamicConvexes[i]);
			// A non-CCD dynamic-box selection must discover contacts at the same
			// discrete endpoint which the native solver will warmstart to.  Looking
			// only at the source pose leaves an endpoint-first overlap with no
			// prepared row, while appending a second source/endpoint contact stream
			// double-owns the same OGC feature.  Keep the scope deliberately narrow
			// until endpoint variants exist for every dynamic shape type.
			bool useEndpointOnlyBoxDcd =
				!storage.selectedDynamicBoxes.empty() &&
				storage.selectedDynamicSpheres.empty() &&
				storage.selectedDynamicCapsules.empty() &&
				storage.selectedDynamicConvexes.empty();
			for(PxU32 bodyIndex = 0;
				useEndpointOnlyBoxDcd &&
				bodyIndex < storage.bodies.size(); ++bodyIndex)
			{
				useEndpointOnlyBoxDcd =
					!storage.bodies[bodyIndex].compiled.
						speculativeCCDEnabled;
			}

			PxArray<Dy::AvbdSoftParticle> endpointParticles;
			PxArray<Dy::AvbdRigidBox> endpointRigidBoxes;
			Dy::AvbdSoftParticle* contactParticles = selectionParticles;
			const Dy::AvbdRigidBox* contactRigidBoxes =
				storage.rigidBoxes.begin();
			if(useEndpointOnlyBoxDcd)
			{
				endpointParticles.resize(selectionParticleCount);
				for(PxU32 particleIndex = 0;
					particleIndex < selectionParticleCount; ++particleIndex)
				{
					endpointParticles[particleIndex] =
						selectionParticles[particleIndex];
					Dy::AvbdSoftParticle& endpointParticle =
						endpointParticles[particleIndex];
					if(!endpointParticle.predictedPosition.isFinite())
						return false;
					// The temporary collision domain is one pose, not a segment.
					endpointParticle.position =
						endpointParticle.predictedPosition;
					endpointParticle.initialPosition =
						endpointParticle.predictedPosition;
					endpointParticle.outerPosition =
						endpointParticle.predictedPosition;
				}

				endpointRigidBoxes = storage.rigidBoxes;
				for(PxU32 boxIndex = 0;
					boxIndex < endpointRigidBoxes.size(); ++boxIndex)
				{
					Dy::AvbdRigidBox& endpointBox =
						endpointRigidBoxes[boxIndex];
					if(endpointBox.targetKind !=
						Dy::AvbdSoftContactTargetKind::eRIGID_BODY)
						continue;
					if(endpointBox.targetIndex >= bodyCount)
						return false;
					Dy::AvbdSolverBody& endpointBody =
						solverBodies[bodyStart + endpointBox.targetIndex];
					endpointBody.computePrediction(dt, gravity);
					const PxTransform endpointBodyToWorld(
						endpointBody.predictedPosition,
						endpointBody.predictedRotation);
					const PxTransform endpointShapeToWorld =
						endpointBodyToWorld *
						endpointBox.shapeToRigidBody;
					if(!endpointShapeToWorld.isValid())
						return false;
					endpointBox.center = endpointShapeToWorld.p;
					endpointBox.rotation = endpointShapeToWorld.q;
					// The selected source bodies are all non-CCD.  Pinning previous
					// to this same endpoint makes that invariant explicit even if a
					// future detector grows another swept branch.
					endpointBox.previousCenter = endpointBox.center;
					endpointBox.previousRotation = endpointBox.rotation;
				}
				contactParticles = endpointParticles.begin();
				contactRigidBoxes = endpointRigidBoxes.begin();
			}

			detectContacts(
				contactParticles, selectionParticleCount,
				storage.bodies.begin(), storage.bodies.size(),
				storage.contacts, contactRigidBoxes,
				storage.rigidBoxes.size(),
				storage.selfCollisionAdjacencies.begin(),
				storage.selfCollisionAdjacencies.size(),
				storage.selfCollisionEnabled.begin(),
				storage.softCores.begin(),
				storage.rigidSpheres.begin(),
				storage.rigidSpheres.size(),
				storage.rigidCapsules.begin(),
				storage.rigidCapsules.size(),
				storage.rigidConvexes.begin(),
				storage.rigidConvexes.size());
			// Preserve a task-local cooked collision proxy for the terminal OGC
			// epoch.  detectContacts may have used Scene-global subset scratch to
			// build the authoritative contact stream; copying just the immutable
			// topology/mapping here lets the native solver refit it from its final
			// pose without a Scene callback or a second owner of storage.contacts.
			if(!rebuildSubsetCollisionDetectionScene(
				selectionParticles, selectionParticleCount,
				storage.bodies.begin(), storage.bodies.size(),
				storage.softCores.begin()))
				return false;
			if(mSubsetCollisionBodies.size() != storage.bodies.size() ||
				mSubsetCollisionParticles.size() !=
					mSubsetCollisionVertexMappings.size())
				return false;
			storage.terminalCollisionBodies = mSubsetCollisionBodies;
			storage.terminalCollisionVertexMappings =
				mSubsetCollisionVertexMappings;
			// Build one mutable OGC epoch record for each *body/shape pair*, not
			// for every manifold row.  A soft collision face commonly produces
			// several SDF/feature rows with opposing normals; treating those rows
			// as independent rigid owners is precisely what causes a free rigid to
			// be alternately projected and ejected.  The contact stream remains AL
			// owned; this map only supplies the shared pair scheduler.
			storage.ogcPairIndices.resize(storage.contacts.size());
			for(PxU32 contactIndex = 0;
				contactIndex < storage.contacts.size(); ++contactIndex)
			{
				storage.ogcPairIndices[contactIndex] = PX_MAX_U32;
				const Dy::AvbdSoftContactGeometry& geometry =
					storage.contacts[contactIndex].geometry;
				const bool dynamicRigidTarget =
					geometry.source.type == Dy::AvbdSoftContactSource::eRIGID_SDF &&
					geometry.hasRigidBodyTarget() &&
					geometry.targetIndex < storage.bodies.size();
				const bool worldStaticTarget =
					(geometry.source.type == Dy::AvbdSoftContactSource::eRIGID_SDF ||
					 geometry.source.type == Dy::AvbdSoftContactSource::eGROUND) &&
					geometry.hasWorldStaticTarget();
				if((!dynamicRigidTarget && !worldStaticTarget) ||
					geometry.queryBodyIndex >= storage.bodies.size())
					continue;

				PxU32 pairIndex = PX_MAX_U32;
				for(PxU32 candidateIndex = 0;
					candidateIndex < storage.ogcPairStates.size(); ++candidateIndex)
				{
					const Dy::AvbdOgcPairState& candidate =
						storage.ogcPairStates[candidateIndex];
					if(candidate.sourceType == geometry.source.type &&
						candidate.targetKind == geometry.targetKind &&
						candidate.sourceBodyIndex == geometry.queryBodyIndex &&
						candidate.targetBodyIndex == geometry.targetIndex &&
						candidate.primitiveKey == geometry.source.primitiveKey)
					{
						pairIndex = candidateIndex;
						break;
					}
				}
				if(pairIndex == PX_MAX_U32)
				{
					pairIndex = storage.ogcPairStates.size();
					Dy::AvbdOgcPairState pair;
					pair.sourceType = geometry.source.type;
					pair.targetKind = geometry.targetKind;
					pair.sourceBodyIndex = geometry.queryBodyIndex;
					pair.targetBodyIndex = geometry.targetIndex;
					pair.primitiveKey = geometry.source.primitiveKey;
					storage.ogcPairStates.pushBack(pair);
				}
				++storage.ogcPairStates[pairIndex].contactCount;
				storage.ogcPairIndices[contactIndex] = pairIndex;
			}
			bool hasRigidTargetContact = false;
			for(PxU32 i = 0; i < storage.contacts.size(); i++)
				if(storage.contacts[i].geometry.hasRigidBodyTarget())
				{
					hasRigidTargetContact = true;
					break;
				}
			const bool isNativeEligible = hasRigidTargetContact ||
				hasRigidAttachment ||
				hasArticulationAttachment ||
				hasSoftPairAttachment;
			if(!isNativeEligible)
				return false;
			return compileIslandSelectionExecutionPlan(
				storage, selectionParticleCount, bodyCount);
		}

		void copyIslandSelectionResults(
			IslandSelectionStorage& storage)
		{
			if(!storage.usesCanonicalParticleRange)
			{
				PX_ASSERT(
					storage.particles.size() ==
					storage.globalParticleIndices.size());
				const PxU32 particleCount = PxMin(
					storage.particles.size(),
					storage.globalParticleIndices.size());
				for(PxU32 i = 0; i < particleCount; i++)
				{
					const PxU32 globalIndex =
						storage.globalParticleIndices[i];
					if(globalIndex < mParticles.size())
						mParticles[globalIndex] = storage.particles[i];
				}
			}
			else
			{
				// The native solver was bound directly to this canonical range;
				// only the separately rebound AL runtime state below requires a
				// transfer back to the Scene owner.
				PX_ASSERT(storage.particles.empty());
				PX_ASSERT(storage.globalParticleIndices.empty());
			}
			PX_ASSERT(
				storage.bodies.size() == storage.entryIndices.size());
			const PxU32 bodyCount = PxMin(
				storage.bodies.size(), storage.entryIndices.size());
			for(PxU32 i = 0; i < bodyCount; i++)
			{
				const Entry& entry =
					mEntries[storage.entryIndices[i]];
				if(entry.bodyIndex < mBodies.size())
				{
					Dy::AvbdSoftBodyRuntimeState& destination =
						mBodies[entry.bodyIndex].runtime;
					const Dy::AvbdSoftBodyRuntimeState& source =
						storage.bodies[i].runtime;
					PX_ASSERT(
						destination.pins.size() ==
							source.pins.size());
					const PxU32 pinCount = PxMin(
						destination.pins.size(),
						source.pins.size());
					for(PxU32 pinIndex = 0;
						pinIndex < pinCount; pinIndex++)
					{
						const Dy::AvbdKinematicPin& sourcePin =
							source.pins[pinIndex];
						destination.pins[pinIndex].alLambda =
							sourcePin.alLambda;
						destination.pins[pinIndex].k =
							sourcePin.k;
						destination.pins[pinIndex].kMax =
							sourcePin.kMax;
						if(sourcePin.targetKind ==
							Dy::AvbdSoftPinTargetKind::
								ePRESCRIBED_RIGID)
						{
							PrescribedAttachmentEntry*
								destinationAttachment =
									findPrescribedAttachment(
										*entry.getActorCore(),
										sourcePin.sourceHandle);
							if(destinationAttachment)
							{
								destinationAttachment->alLambda =
									sourcePin.alLambda;
								destinationAttachment->k =
									sourcePin.k;
								destinationAttachment->kMax =
									sourcePin.kMax;
							}
						}
					}
					for(PxU32 attachmentIndex = 0;
						attachmentIndex <
							source.attachments.size();
						attachmentIndex++)
					{
						const Dy::AvbdSoftAttachment&
							sourceAttachment =
								source.attachments[
									attachmentIndex];
						switch(sourceAttachment.targetKind)
						{
						case Dy::AvbdSoftAttachmentTargetKind::
							eDYNAMIC_RIGID:
						{
							const PxU32 handle =
								sourceAttachment.sourceHandle;
							RigidAttachmentEntry*
								destinationAttachment =
									findRigidAttachment(
										*entry.getActorCore(),
										handle);
							if(!destinationAttachment)
								continue;
							destinationAttachment->alLambda =
								sourceAttachment.alLambda;
							destinationAttachment->k =
								sourceAttachment.k;
							destinationAttachment->kMax =
								sourceAttachment.kMax;
							break;
						}
						case Dy::AvbdSoftAttachmentTargetKind::
							eARTICULATION_LINK:
						{
							const PxU32 handle =
								sourceAttachment.sourceHandle;
							ArticulationAttachmentEntry*
								destinationAttachment =
									findArticulationAttachment(
										*entry.getActorCore(),
										handle);
							if(!destinationAttachment)
								continue;
							destinationAttachment->alLambda =
								sourceAttachment.alLambda;
							destinationAttachment->k =
								sourceAttachment.k;
							destinationAttachment->kMax =
								sourceAttachment.kMax;
							break;
						}
						case Dy::AvbdSoftAttachmentTargetKind::
							eDYNAMIC_SOFT:
						{
							const PxU32 handle =
								sourceAttachment.sourceHandle;
							SoftPairAttachmentEntry*
								destinationAttachment =
									findSoftPairAttachment(
										*entry.getActorCore(),
										handle);
							if(!destinationAttachment)
								continue;
							destinationAttachment->alLambda =
								sourceAttachment.alLambda;
							destinationAttachment->k =
								sourceAttachment.k;
							destinationAttachment->kMax =
								sourceAttachment.kMax;
							break;
						}
						case Dy::AvbdSoftAttachmentTargetKind::
							eUNSUPPORTED:
						default:
							PX_ASSERT(false);
							break;
						}
					}
					PX_ASSERT(destination.attachments.empty());
					destination.compileObjectiveProgram(
						getParticleStart(entry),
						getParticleCount(entry));
				}
			}
		}

		PxU32 findNativeIslandEdge(
			const ActorCore* softCore,
			const BodyCore* rigidCore) const
		{
			for(PxU32 i = 0;
				i < mNativeIslandEdges.size(); i++)
			{
				const NativeIslandEdgeEntry& entry =
					mNativeIslandEdges[i];
				if(entry.softCore == softCore &&
					entry.rigidCore == rigidCore)
					return i;
			}
			return PX_MAX_U32;
		}

		void ensureNativeIslandEdge(
			Entry& softEntry, BodyCore& rigidCore)
		{
			const PxU32 existingIndex =
				findNativeIslandEdge(
					softEntry.getActorCore(), &rigidCore);
			if(existingIndex != PX_MAX_U32)
			{
				mNativeIslandEdges[existingIndex].touched = true;
				return;
			}

			BodySim* bodySim = rigidCore.getSim();
			if(!bodySim || !bodySim->getNodeIndex().isValid())
				return;
			const IG::EdgeIndex edgeIndex =
				mIslandManager.addContactManager(
					NULL, softEntry.islandNode,
					bodySim->getNodeIndex(), NULL,
					IG::Edge::eSOFT_BODY_CONTACT);
			mIslandManager.setEdgeConnected(
				edgeIndex, IG::Edge::eSOFT_BODY_CONTACT);
			mNativeIslandEdges.pushBack(
				NativeIslandEdgeEntry(
					*softEntry.getActorCore(),
					rigidCore, edgeIndex));
		}

		PxU32 findNativeSoftSoftIslandEdge(
			const ActorCore* softCore0,
			const ActorCore* softCore1) const
		{
			for(PxU32 i = 0;
				i < mNativeSoftSoftIslandEdges.size(); i++)
			{
				const NativeSoftSoftIslandEdgeEntry& entry =
					mNativeSoftSoftIslandEdges[i];
				if((entry.softCore0 == softCore0 &&
						entry.softCore1 == softCore1) ||
					(entry.softCore0 == softCore1 &&
						entry.softCore1 == softCore0))
					return i;
			}
			return PX_MAX_U32;
		}

		void ensureNativeSoftSoftIslandEdge(
			Entry& softEntry0, Entry& softEntry1)
		{
			const PxU32 existingIndex =
				findNativeSoftSoftIslandEdge(
					softEntry0.getActorCore(),
					softEntry1.getActorCore());
			if(existingIndex != PX_MAX_U32)
			{
				mNativeSoftSoftIslandEdges[existingIndex].touched = true;
				return;
			}
			if(!softEntry0.islandNode.isValid() ||
				!softEntry1.islandNode.isValid())
				return;

			const IG::EdgeIndex edgeIndex =
				mIslandManager.addContactManager(
					NULL, softEntry0.islandNode,
					softEntry1.islandNode, NULL,
					IG::Edge::eSOFT_BODY_CONTACT);
			mIslandManager.setEdgeConnected(
				edgeIndex, IG::Edge::eSOFT_BODY_CONTACT);
			mNativeSoftSoftIslandEdges.pushBack(
				NativeSoftSoftIslandEdgeEntry(
					*softEntry0.getActorCore(),
					*softEntry1.getActorCore(), edgeIndex));
		}

		void removeNativeIslandEdgesForRigid(BodyCore& core)
		{
			for(PxU32 i = mNativeIslandEdges.size();
				i > 0; i--)
			{
				if(mNativeIslandEdges[i - 1].rigidCore == &core)
				{
					mIslandManager.removeConnection(
						mNativeIslandEdges[i - 1].edgeIndex);
					mNativeIslandEdges.replaceWithLast(i - 1);
				}
			}
		}

		void removeNativeIslandEdgesForSoft(
			ActorCore& core)
		{
			for(PxU32 i = mNativeIslandEdges.size();
				i > 0; i--)
			{
				if(mNativeIslandEdges[i - 1].softCore == &core)
				{
					mIslandManager.removeConnection(
						mNativeIslandEdges[i - 1].edgeIndex);
					mNativeIslandEdges.replaceWithLast(i - 1);
				}
			}
			for(PxU32 i = mNativeSoftSoftIslandEdges.size();
				i > 0; i--)
			{
				const NativeSoftSoftIslandEdgeEntry& edge =
					mNativeSoftSoftIslandEdges[i - 1];
				if(edge.softCore0 == &core ||
					edge.softCore1 == &core)
				{
					mIslandManager.removeConnection(edge.edgeIndex);
					mNativeSoftSoftIslandEdges.replaceWithLast(i - 1);
				}
			}
		}

		void clearNativeIslandEdges()
		{
			for(PxU32 i = 0;
				i < mNativeIslandEdges.size(); i++)
				mIslandManager.removeConnection(
					mNativeIslandEdges[i].edgeIndex);
			mNativeIslandEdges.clear();
			for(PxU32 i = 0;
				i < mNativeSoftSoftIslandEdges.size(); i++)
				mIslandManager.removeConnection(
					mNativeSoftSoftIslandEdges[i].edgeIndex);
			mNativeSoftSoftIslandEdges.clear();
		}

		bool computeSoftBounds(
			const Entry& entry, PxBounds3& bounds) const
		{
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			if(particleCount == 0 ||
				particleStart > mParticles.size() ||
				particleCount >
					mParticles.size() - particleStart)
				return false;

			bounds = PxBounds3::empty();
			for(PxU32 i = 0; i < particleCount; i++)
			{
				const PxVec3& position =
					mParticles[particleStart + i].position;
				if(!position.isFinite())
					return false;
				bounds.include(position);
			}
			return !bounds.isEmpty();
		}

		// Native-island topology must use the same public collision boundary as
		// OGC.  A cooked volume's simulation tetrahedra can lie strictly inside
		// its collision mesh; culling only against their AABB can therefore see a
		// current-pose OGC row too late to connect the dynamic rigid island for
		// that step.
		bool computeCollisionDomainSoftBounds(
			const Entry& entry, PxBounds3& bounds) const
		{
			if(entry.bodyIndex >= mBodies.size() ||
				mCollisionBodies.size() != mBodies.size() ||
				entry.bodyIndex >= mCollisionBodies.size() ||
				entry.collisionMesh == entry.simulationMesh)
				return computeSoftBounds(entry, bounds);

			const Dy::AvbdSoftBodyCompiledData& collisionCompiled =
				mCollisionBodies[entry.bodyIndex].compiled;
			const PxU32 collisionParticleStart =
				collisionCompiled.particleStart;
			const PxU32 collisionParticleCount =
				collisionCompiled.particleCount;
			if(collisionParticleCount == 0 ||
				collisionParticleStart > mCollisionVertexMappings.size() ||
				collisionParticleCount >
					mCollisionVertexMappings.size() -
						collisionParticleStart)
				return false;

			bounds = PxBounds3::empty();
			for(PxU32 localParticleIndex = 0;
				localParticleIndex < collisionParticleCount;
				++localParticleIndex)
			{
				const Dy::AvbdWeightedContactPoint& mapping =
					mCollisionVertexMappings[
						collisionParticleStart + localParticleIndex];
				const PxVec3 position = evaluateWeightedParticlePosition(
					mapping, mParticles.begin(), mParticles.size(), 0);
				if(!position.isFinite())
					return false;
				bounds.include(position);
			}
			return !bounds.isEmpty();
		}

		bool expandSoftBoundsForPrediction(
			const Entry& entry, PxReal dt, const PxVec3& gravity,
			PxBounds3& bounds) const
		{
			if(dt <= 0.0f || !PxIsFinite(dt) ||
				!gravity.isFinite() || bounds.isEmpty())
				return false;
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			if(particleCount == 0 ||
				particleStart > mParticles.size() ||
				particleCount > mParticles.size() - particleStart)
				return false;
			const PxReal dtSq = dt * dt;
			for(PxU32 i = 0; i < particleCount; i++)
			{
				const Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				if(particle.invMass <= 0.0f)
					continue;
				const PxVec3 predictedPosition =
					particle.position + particle.velocity * dt +
					gravity * (particle.gravityScale * dtSq);
				if(!predictedPosition.isFinite())
					return false;
				bounds.include(predictedPosition);
			}
			return true;
		}

		// This is deliberately an endpoint-only DCD broad phase.  It is used
		// for sources that have opted out of speculative CCD: the resulting
		// contact pass evaluates the predicted endpoint only, never the segment
		// between the old and predicted poses.  Keep this separate from
		// expandSoftBoundsForPrediction(), which includes the old bounds for the
		// swept/CCD admission path.
		bool computePredictedCollisionDomainSoftBounds(
			const Entry& entry, PxReal dt, const PxVec3& gravity,
			PxBounds3& bounds) const
		{
			if(dt <= 0.0f || !PxIsFinite(dt) ||
				!gravity.isFinite() || entry.bodyIndex >= mBodies.size())
				return false;

			const PxReal dtSq = dt * dt;
			const auto predictPosition = [&] (
				const Dy::AvbdSoftParticle& particle,
				PxVec3& position)
			{
				if(!particle.position.isFinite() ||
					!particle.velocity.isFinite() ||
					!PxIsFinite(particle.gravityScale))
					return false;
				position = particle.position;
				if(particle.invMass > 0.0f)
					position += particle.velocity * dt +
						gravity * (particle.gravityScale * dtSq);
				return position.isFinite();
			};

			bounds = PxBounds3::empty();
			if(mCollisionBodies.size() != mBodies.size() ||
				entry.bodyIndex >= mCollisionBodies.size() ||
				entry.collisionMesh == entry.simulationMesh)
			{
				const PxU32 particleStart = getParticleStart(entry);
				const PxU32 particleCount = getParticleCount(entry);
				if(particleCount == 0 || particleStart > mParticles.size() ||
					particleCount > mParticles.size() - particleStart)
					return false;
				for(PxU32 i = 0; i < particleCount; ++i)
				{
					PxVec3 position;
					if(!predictPosition(mParticles[particleStart + i], position))
						return false;
					bounds.include(position);
				}
				return !bounds.isEmpty();
			}

			const Dy::AvbdSoftBodyCompiledData& collisionCompiled =
				mCollisionBodies[entry.bodyIndex].compiled;
			const PxU32 collisionParticleStart =
				collisionCompiled.particleStart;
			const PxU32 collisionParticleCount =
				collisionCompiled.particleCount;
			if(collisionParticleCount == 0 ||
				collisionParticleStart > mCollisionVertexMappings.size() ||
				collisionParticleCount >
					mCollisionVertexMappings.size() - collisionParticleStart)
				return false;
			for(PxU32 localParticleIndex = 0;
				localParticleIndex < collisionParticleCount;
				++localParticleIndex)
			{
				const Dy::AvbdWeightedContactPoint& mapping =
					mCollisionVertexMappings[
						collisionParticleStart + localParticleIndex];
				if(mapping.count == 0 ||
					mapping.count >
						Dy::AVBD_CONTACT_POINT_MAX_SUPPORT)
					return false;
				PxVec3 position(0.0f);
				for(PxU32 supportIndex = 0;
					supportIndex < mapping.count; ++supportIndex)
				{
					const PxU32 particleIndex =
						mapping.particleIndices[supportIndex];
					const PxReal weight = mapping.weights[supportIndex];
					if(particleIndex >= mParticles.size() ||
						!PxIsFinite(weight))
						return false;
					PxVec3 supportPosition;
					if(!predictPosition(
							mParticles[particleIndex], supportPosition))
						return false;
					position += supportPosition * weight;
				}
				if(!position.isFinite())
					return false;
				bounds.include(position);
			}
			return !bounds.isEmpty();
		}

		static PxBounds3 computeBoxBounds(
			const Dy::AvbdRigidBox& box)
		{
			const PxMat33 basis(box.rotation);
			const PxVec3& h = box.halfExtent;
			const PxVec3 extent(
				PxAbs(basis.column0.x) * h.x +
					PxAbs(basis.column1.x) * h.y +
					PxAbs(basis.column2.x) * h.z,
				PxAbs(basis.column0.y) * h.x +
					PxAbs(basis.column1.y) * h.y +
					PxAbs(basis.column2.y) * h.z,
				PxAbs(basis.column0.z) * h.x +
					PxAbs(basis.column1.z) * h.y +
					PxAbs(basis.column2.z) * h.z);
			return PxBounds3(
				box.center - extent, box.center + extent);
		}

		static PxBounds3 computeSphereBounds(
			const Dy::AvbdRigidSphere& sphere)
		{
			const PxVec3 extent(PxMax(sphere.radius, 0.0f));
			return PxBounds3(
				sphere.center - extent,
				sphere.center + extent);
		}

		static PxBounds3 computeCapsuleBounds(
			const Dy::AvbdRigidCapsule& capsule)
		{
			const PxVec3 axisOffset =
				capsule.rotation.getBasisVector0() *
					PxMax(capsule.halfHeight, 0.0f);
			const PxVec3 extent(PxMax(capsule.radius, 0.0f));
			const PxVec3 endpoint0 =
				capsule.center - axisOffset;
			const PxVec3 endpoint1 =
				capsule.center + axisOffset;
			return PxBounds3(
				endpoint0.minimum(endpoint1) - extent,
				endpoint0.maximum(endpoint1) + extent);
		}

		static PxBounds3 computeConvexBounds(
			const Dy::AvbdRigidConvex& convex)
		{
			PxBounds3 bounds = PxBounds3::empty();
			for(PxU32 vertexIndex = 0;
				vertexIndex < convex.vertices.size(); ++vertexIndex)
			{
				const PxVec3 worldVertex =
					convex.center +
					convex.rotation.rotate(
						convex.vertices[vertexIndex]);
				if(!worldVertex.isFinite())
					return PxBounds3::empty();
				bounds.include(worldVertex);
			}
			return bounds;
		}

		// Build a conservative AABB for one rigid shape at the same discrete
		// endpoint used by non-CCD soft admission. The sphere is an endpoint
		// orientation envelope, not a source-to-end sweep: only the predicted
		// body center is included. Keeping this helper common prevents smooth
		// primitives from silently falling back to source-pose topology while
		// boxes use endpoint ownership.
		static bool computeDynamicEndpointEnvelopeBounds(
			const DynamicShapeEntry& entry,
			const PxVec3& shapeCenter, PxReal shapeRadius,
			PxReal dt, const PxVec3& gravity, PxBounds3& bounds)
		{
			BodySim* const bodySim = entry.core ? entry.core->getSim() : NULL;
			if(!bodySim || bodySim->isKinematic() ||
				bodySim->isArticulationLink() ||
				!shapeCenter.isFinite() || !PxIsFinite(shapeRadius) ||
				shapeRadius < 0.0f)
				return false;
			const PxsBodyCore& bodyCore = entry.core->getCore();
			const PxVec3 bodyCenter = bodyCore.body2World.p;
			const PxVec3 predictedBodyCenter = bodyCenter +
				bodyCore.linearVelocity * dt +
				(bodyCore.disableGravity ? PxVec3(0.0f) :
					gravity * (dt * dt));
			const PxReal endpointRadius =
				shapeRadius + (shapeCenter - bodyCenter).magnitude();
			if(!bodyCenter.isFinite() || !predictedBodyCenter.isFinite() ||
				!PxIsFinite(endpointRadius))
				return false;
			const PxVec3 endpointExtent(endpointRadius);
			bounds = PxBounds3(
				predictedBodyCenter - endpointExtent,
				predictedBodyCenter + endpointExtent);
			return true;
		}

		static PxBounds3 computeTriangleSurfaceBounds(
			const Dy::AvbdRigidTriangleSurface& surface)
		{
			PxBounds3 bounds = PxBounds3::empty();
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertices.size();
				++vertexIndex)
			{
				const PxVec3 worldVertex =
					surface.center +
					surface.rotation.rotate(
						surface.vertices[vertexIndex].point);
				if(!worldVertex.isFinite())
					return PxBounds3::empty();
				bounds.include(worldVertex);
			}
			return bounds;
		}

		static void getRigidMaterialValues(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			PxMaterialTableIndex tableIndex,
			PxReal& friction, PxU8& combineMode)
		{
			const PxU16* materialIndices =
				shape.getMaterialIndices();
			const PxU32 materialCount =
				shape.getNbMaterialIndices();
			const PxU32 resolvedTableIndex =
				tableIndex == PxMaterialTableIndex(0xffff)
					? 0u : PxU32(tableIndex);
			friction = 0.5f;
			combineMode =
				PxU8(PxCombineMode::eAVERAGE);
			if(!materialIndices ||
				resolvedTableIndex >= materialCount)
				return;
			const PxU16 materialIndex =
				materialIndices[resolvedTableIndex];
			if(materialIndex == MATERIAL_INVALID_HANDLE ||
				materialIndex >= materialManager.getMaxSize())
				return;
			const PxsMaterialCore* material =
				materialManager.getMaterial(materialIndex);
			if(material->mMaterialIndex != materialIndex)
				return;
			friction =
				PxMax(material->dynamicFriction, 0.0f);
			combineMode =
				PxU8(material->getFrictionCombineMode());
		}

		static bool appendTriangleSurfaceTriangle(
			const PxTriangle& sourceTriangle,
			const PxU32 sourceVertexIndices[3],
			PxU32 sourceTriangleIndex,
			PxReal friction, PxU8 frictionCombineMode,
			PxHashMap<PxU32, PxU32>& vertexMap,
			PxHashMap<PxU64, PxU32>& edgeMap,
			Dy::AvbdRigidTriangleSurface& surface)
		{
			PxU32 vertices[3] =
				{PX_MAX_U32, PX_MAX_U32, PX_MAX_U32};
			for(PxU32 endpoint = 0;
				endpoint < 3; ++endpoint)
			{
				const PxHashMap<PxU32, PxU32>::Entry* entry =
					vertexMap.find(
						sourceVertexIndices[endpoint]);
				if(entry)
					vertices[endpoint] = entry->second;
				else
				{
					const PxVec3& point =
						sourceTriangle.verts[endpoint];
					if(!point.isFinite())
						return false;
					Dy::AvbdRigidTriangleSurfaceVertex vertex;
					vertex.point = point;
					vertex.friction = friction;
					vertex.frictionCombineMode =
						frictionCombineMode;
					vertex.sourceTriangleIndex =
						sourceTriangleIndex;
					vertex.outward = PxVec3(0.0f);
					vertices[endpoint] =
						surface.vertices.size();
					surface.vertices.pushBack(vertex);
					vertexMap.insert(
						sourceVertexIndices[endpoint],
						vertices[endpoint]);
				}
			}
			if(vertices[0] == vertices[1] ||
				vertices[0] == vertices[2] ||
				vertices[1] == vertices[2])
				return true;
			PxVec3 normal =
				(sourceTriangle.verts[1] -
					sourceTriangle.verts[0]).cross(
						sourceTriangle.verts[2] -
							sourceTriangle.verts[0]);
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(normalMagnitudeSq <= 1.0e-12f ||
				!PxIsFinite(normalMagnitudeSq))
				return true;
			normal *= PxRecipSqrt(normalMagnitudeSq);

			Dy::AvbdRigidTriangleSurfaceTriangle triangle;
			triangle.p0 = vertices[0];
			triangle.p1 = vertices[1];
			triangle.p2 = vertices[2];
			triangle.sourceTriangleIndex =
				sourceTriangleIndex;
			triangle.normal = normal;
			triangle.friction = friction;
			triangle.frictionCombineMode =
				frictionCombineMode;
			const PxU32 triangleIndex =
				surface.triangles.size();

			const PxU32 edgeEndpoints[3][2] =
			{
				{vertices[0], vertices[1]},
				{vertices[0], vertices[2]},
				{vertices[1], vertices[2]}
			};
			PxU32* triangleEdges[3] =
				{&triangle.edge0, &triangle.edge1,
				 &triangle.edge2};
			for(PxU32 localEdge = 0;
				localEdge < 3; ++localEdge)
			{
				const PxU32 edge0 = PxMin(
					edgeEndpoints[localEdge][0],
					edgeEndpoints[localEdge][1]);
				const PxU32 edge1 = PxMax(
					edgeEndpoints[localEdge][0],
					edgeEndpoints[localEdge][1]);
				const PxU64 edgeKey =
					(PxU64(edge0) << 32) | PxU64(edge1);
				const PxHashMap<PxU64, PxU32>::Entry* entry =
					edgeMap.find(edgeKey);
				PxU32 edgeIndex = PX_MAX_U32;
				if(entry)
					edgeIndex = entry->second;
				else
				{
					Dy::AvbdRigidTriangleSurfaceEdge edge;
					edge.p0 = edge0;
					edge.p1 = edge1;
					edge.outward = PxVec3(0.0f);
					edge.friction = friction;
					edge.frictionCombineMode =
						frictionCombineMode;
					edge.sourceTriangleIndex =
						sourceTriangleIndex;
					edgeIndex = surface.edges.size();
					surface.edges.pushBack(edge);
					edgeMap.insert(edgeKey, edgeIndex);
				}
				if(edgeIndex >= surface.edges.size())
					return false;
				Dy::AvbdRigidTriangleSurfaceEdge& edge =
					surface.edges[edgeIndex];
				if(edge.adjacentCount == 0)
					edge.triangle0 = triangleIndex;
				else if(edge.adjacentCount == 1)
					edge.triangle1 = triangleIndex;
				++edge.adjacentCount;
				edge.outward += normal;
				*triangleEdges[localEdge] = edgeIndex;
			}

			for(PxU32 endpoint = 0;
				endpoint < 3; ++endpoint)
				surface.vertices[vertices[endpoint]].
					outward += normal;
			surface.triangles.pushBack(triangle);
			return true;
		}

		static bool finalizeTriangleSurfaceTopology(
			Dy::AvbdRigidTriangleSurface& surface,
			bool suppressBoundaryEdges)
		{
			if(surface.vertices.size() < 3 ||
				surface.triangles.empty())
				return false;
			surface.localBounds = PxBounds3::empty();
			surface.localRadius = 0.0f;
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertices.size();
				++vertexIndex)
			{
				Dy::AvbdRigidTriangleSurfaceVertex& vertex =
					surface.vertices[vertexIndex];
				if(!vertex.point.isFinite())
					return false;
				surface.localBounds.include(vertex.point);
				surface.localRadius = PxMax(
					surface.localRadius,
					vertex.point.magnitude());
				const PxReal normalMagnitudeSq =
					vertex.outward.magnitudeSquared();
				if(normalMagnitudeSq > 1.0e-12f &&
					PxIsFinite(normalMagnitudeSq))
					vertex.outward *=
						PxRecipSqrt(normalMagnitudeSq);
				else
					vertex.outward =
						PxVec3(0.0f, 1.0f, 0.0f);
			}

			for(PxU32 edgeIndex = 0;
				edgeIndex < surface.edges.size();
				++edgeIndex)
			{
				Dy::AvbdRigidTriangleSurfaceEdge& edge =
					surface.edges[edgeIndex];
				edge.active = false;
				if(edge.adjacentCount == 1)
					edge.active = !suppressBoundaryEdges;
				else if(edge.adjacentCount == 2 &&
					edge.triangle0 < surface.triangles.size() &&
					edge.triangle1 < surface.triangles.size())
				{
					const Dy::AvbdRigidTriangleSurfaceTriangle&
						triangle0 =
							surface.triangles[edge.triangle0];
					const Dy::AvbdRigidTriangleSurfaceTriangle&
						triangle1 =
							surface.triangles[edge.triangle1];
					PxU32 opposite0 = triangle0.p0;
					if(opposite0 == edge.p0 ||
						opposite0 == edge.p1)
						opposite0 = triangle0.p1;
					if(opposite0 == edge.p0 ||
						opposite0 == edge.p1)
						opposite0 = triangle0.p2;
					if(opposite0 >= surface.vertices.size() ||
						triangle1.p0 >=
							surface.vertices.size())
						return false;
					const PxReal oppositePlaneDistance =
						triangle1.normal.dot(
							surface.vertices[opposite0].point -
							surface.vertices[
								triangle1.p0].point);
					const PxReal normalDot =
						triangle0.normal.dot(
							triangle1.normal);
					edge.active =
						(oppositePlaneDistance < 0.0f &&
						 normalDot < 0.999999f) ||
						normalDot < -0.999f;
				}
				const PxReal normalMagnitudeSq =
					edge.outward.magnitudeSquared();
				if(normalMagnitudeSq > 1.0e-12f &&
					PxIsFinite(normalMagnitudeSq))
					edge.outward *=
						PxRecipSqrt(normalMagnitudeSq);
				else if(edge.triangle0 <
					surface.triangles.size())
					edge.outward =
						surface.triangles[
							edge.triangle0].normal;
				if(edge.active)
				{
					if(edge.p0 < surface.vertices.size())
						surface.vertices[edge.p0].active = true;
					if(edge.p1 < surface.vertices.size())
						surface.vertices[edge.p1].active = true;
				}
			}
			return !surface.localBounds.isEmpty() &&
				PxIsFinite(surface.localRadius) &&
				surface.localRadius > 0.0f;
		}

		static PxBounds3 getRigidTriangleSurfaceTriangleBounds(
			const Dy::AvbdRigidTriangleSurface& surface,
			PxU32 triangleIndex)
		{
			PxBounds3 bounds(PxBounds3::empty());
			if(triangleIndex >= surface.triangles.size())
				return bounds;
			const Dy::AvbdRigidTriangleSurfaceTriangle& triangle =
				surface.triangles[triangleIndex];
			if(triangle.p0 >= surface.vertices.size() ||
				triangle.p1 >= surface.vertices.size() ||
				triangle.p2 >= surface.vertices.size())
				return bounds;
			bounds.include(surface.vertices[triangle.p0].point);
			bounds.include(surface.vertices[triangle.p1].point);
			bounds.include(surface.vertices[triangle.p2].point);
			return bounds;
		}

		static PxU32 buildRigidTriangleSurfaceBvhNode(
			Dy::AvbdRigidTriangleSurface& surface,
			PxU32 firstPrimitive, PxU32 primitiveCount)
		{
			const PxU32 nodeIndex = surface.triangleBvhNodes.size();
			Dy::AvbdRigidTriangleSurfaceBvhNode node;
			node.minimum = PxVec3(PX_MAX_F32);
			node.maximum = PxVec3(-PX_MAX_F32);
			node.leftChild = PX_MAX_U32;
			node.rightChild = PX_MAX_U32;
			node.firstPrimitive = firstPrimitive;
			node.primitiveCount = primitiveCount;
			for(PxU32 entry = firstPrimitive;
				entry < firstPrimitive + primitiveCount; ++entry)
			{
				const PxBounds3 bounds =
					getRigidTriangleSurfaceTriangleBounds(
						surface,
						surface.triangleBvhTriangleIndices[entry]);
				if(!bounds.isEmpty())
				{
					node.minimum = node.minimum.minimum(bounds.minimum);
					node.maximum = node.maximum.maximum(bounds.maximum);
				}
			}
			surface.triangleBvhNodes.pushBack(node);
			if(primitiveCount <= 4)
				return nodeIndex;

			const PxVec3 extent = node.maximum - node.minimum;
			const PxU32 axis = extent.y > extent.x && extent.y >= extent.z
				? 1u : extent.z > extent.x && extent.z > extent.y ? 2u : 0u;
			PxSort(
				surface.triangleBvhTriangleIndices.begin() +
					firstPrimitive,
				primitiveCount,
				[&surface, axis](PxU32 lhs, PxU32 rhs)
				{
					const PxBounds3 lhsBounds =
						getRigidTriangleSurfaceTriangleBounds(surface, lhs);
					const PxBounds3 rhsBounds =
						getRigidTriangleSurfaceTriangleBounds(surface, rhs);
					const PxVec3 lhsCenter =
						(lhsBounds.minimum + lhsBounds.maximum) * 0.5f;
					const PxVec3 rhsCenter =
						(rhsBounds.minimum + rhsBounds.maximum) * 0.5f;
					const PxReal lhsValue = axis == 0 ? lhsCenter.x :
						axis == 1 ? lhsCenter.y : lhsCenter.z;
					const PxReal rhsValue = axis == 0 ? rhsCenter.x :
						axis == 1 ? rhsCenter.y : rhsCenter.z;
					return lhsValue == rhsValue ? lhs < rhs :
						lhsValue < rhsValue;
				});
			const PxU32 leftCount = primitiveCount / 2;
			const PxU32 leftChild = buildRigidTriangleSurfaceBvhNode(
				surface, firstPrimitive, leftCount);
			const PxU32 rightChild = buildRigidTriangleSurfaceBvhNode(
				surface, firstPrimitive + leftCount,
				primitiveCount - leftCount);
			surface.triangleBvhNodes[nodeIndex].leftChild = leftChild;
			surface.triangleBvhNodes[nodeIndex].rightChild = rightChild;
			return nodeIndex;
		}

		static void buildRigidTriangleSurfaceBvh(
			Dy::AvbdRigidTriangleSurface& surface)
		{
			surface.triangleBvhTriangleIndices.clear();
			surface.triangleBvhNodes.clear();
			const PxU32 triangleCount = surface.triangles.size();
			if(triangleCount == 0)
				return;
			surface.triangleBvhTriangleIndices.reserve(triangleCount);
			surface.triangleBvhQueryCandidates.reserve(triangleCount);
			surface.edgeBvhQueryCandidates.reserve(surface.edges.size());
			surface.vertexBvhQueryCandidates.reserve(surface.vertices.size());
			surface.edgeBvhCandidateStamps.resize(surface.edges.size());
			surface.vertexBvhCandidateStamps.resize(surface.vertices.size());
			for(PxU32 edgeIndex = 0;
				edgeIndex < surface.edgeBvhCandidateStamps.size(); ++edgeIndex)
				surface.edgeBvhCandidateStamps[edgeIndex] = 0;
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertexBvhCandidateStamps.size(); ++vertexIndex)
				surface.vertexBvhCandidateStamps[vertexIndex] = 0;
			surface.featureBvhCandidateStamp = 0;
			for(PxU32 triangleIndex = 0;
				triangleIndex < triangleCount; ++triangleIndex)
				surface.triangleBvhTriangleIndices.pushBack(triangleIndex);
			buildRigidTriangleSurfaceBvhNode(
				surface, 0, triangleCount);
		}

		static bool compileTriangleMeshTopology(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxTriangleMeshGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface)
		{
			PxTriangleMesh* mesh = geometry.triangleMesh;
			if(!mesh || !geometry.isValid())
				return false;
			surface.vertices.clear();
			surface.edges.clear();
			surface.triangles.clear();
			PxHashMap<PxU32, PxU32> vertexMap;
			PxHashMap<PxU64, PxU32> edgeMap;
			for(PxU32 triangleIndex = 0;
				triangleIndex < mesh->getNbTriangles();
				++triangleIndex)
			{
				PxTriangle triangle;
				PxU32 vertexIndices[3] =
					{PX_MAX_U32, PX_MAX_U32, PX_MAX_U32};
				PxMeshQuery::getTriangle(
					geometry, PxTransform(PxIdentity),
					triangleIndex, triangle, vertexIndices);
				PxReal friction = 0.5f;
				PxU8 frictionCombineMode =
					PxU8(PxCombineMode::eAVERAGE);
				getRigidMaterialValues(
					shape, materialManager,
					mesh->getTriangleMaterialIndex(
						triangleIndex),
					friction, frictionCombineMode);
				if(!appendTriangleSurfaceTriangle(
						triangle, vertexIndices,
						triangleIndex, friction,
						frictionCombineMode,
						vertexMap, edgeMap, surface))
					return false;
			}
			if(!finalizeTriangleSurfaceTopology(surface, false))
				return false;
			buildRigidTriangleSurfaceBvh(surface);
			return true;
		}

		static bool compileHeightFieldTopology(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxHeightFieldGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface)
		{
			PxHeightField* heightField =
				geometry.heightField;
			if(!heightField || !geometry.isValid())
				return false;
			const PxU32 rows = heightField->getNbRows();
			const PxU32 columns =
				heightField->getNbColumns();
			if(rows < 2 || columns < 2)
				return false;
			surface.vertices.clear();
			surface.edges.clear();
			surface.triangles.clear();
			PxHashMap<PxU32, PxU32> vertexMap;
			PxHashMap<PxU64, PxU32> edgeMap;
			for(PxU32 row = 0; row + 1 < rows; ++row)
			{
				for(PxU32 column = 0;
					column + 1 < columns; ++column)
				{
					for(PxU32 localTriangle = 0;
						localTriangle < 2;
						++localTriangle)
					{
						const PxU32 triangleIndex =
							2 * (row * columns + column) +
							localTriangle;
						const PxMaterialTableIndex materialIndex =
							heightField->
								getTriangleMaterialIndex(
									triangleIndex);
						if(materialIndex ==
							PxHeightFieldMaterial::eHOLE)
							continue;
						PxTriangle triangle;
						PxU32 vertexIndices[3] =
							{PX_MAX_U32, PX_MAX_U32,
							 PX_MAX_U32};
						PxMeshQuery::getTriangle(
							geometry,
							PxTransform(PxIdentity),
							triangleIndex, triangle,
							vertexIndices);
						PxReal friction = 0.5f;
						PxU8 frictionCombineMode =
							PxU8(PxCombineMode::eAVERAGE);
						getRigidMaterialValues(
							shape, materialManager,
							materialIndex, friction,
							frictionCombineMode);
						if(!appendTriangleSurfaceTriangle(
								triangle, vertexIndices,
								triangleIndex, friction,
								frictionCombineMode,
								vertexMap, edgeMap, surface))
							return false;
					}
				}
			}
			const bool suppressBoundaryEdges =
				(heightField->getFlags() &
					PxHeightFieldFlag::eNO_BOUNDARY_EDGES)
				? true : false;
			if(!finalizeTriangleSurfaceTopology(
					surface, suppressBoundaryEdges))
				return false;
			buildRigidTriangleSurfaceBvh(surface);
			return true;
		}

		static bool sameTriangleSurfaceVec3(
			const PxVec3& lhs, const PxVec3& rhs)
		{
			return lhs.x == rhs.x && lhs.y == rhs.y &&
				lhs.z == rhs.z;
		}

		static bool sameTriangleSurfaceQuat(
			const PxQuat& lhs, const PxQuat& rhs)
		{
			return lhs.x == rhs.x && lhs.y == rhs.y &&
				lhs.z == rhs.z && lhs.w == rhs.w;
		}

		static bool getTriangleMeshMaterialValues(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxTriangleMesh& mesh,
			PxU32 sourceTriangleIndex, PxReal& friction,
			PxU8& frictionCombineMode)
		{
			if(sourceTriangleIndex >= mesh.getNbTriangles())
				return false;
			getRigidMaterialValues(
				shape, materialManager,
				mesh.getTriangleMaterialIndex(sourceTriangleIndex),
				friction, frictionCombineMode);
			return true;
		}

		static bool getHeightFieldMaterialValues(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxHeightField& heightField,
			PxU32 sourceTriangleIndex, PxReal& friction,
			PxU8& frictionCombineMode)
		{
			const PxMaterialTableIndex materialIndex =
				heightField.getTriangleMaterialIndex(
					sourceTriangleIndex);
			if(materialIndex == PxHeightFieldMaterial::eHOLE)
				return false;
			getRigidMaterialValues(
				shape, materialManager, materialIndex, friction,
				frictionCombineMode);
			return true;
		}

		static bool refreshTriangleMeshSurfaceMaterials(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxTriangleMeshGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface)
		{
			PxTriangleMesh* mesh = geometry.triangleMesh;
			if(!mesh || !geometry.isValid())
				return false;
			for(PxU32 triangleIndex = 0;
				triangleIndex < surface.triangles.size(); ++triangleIndex)
			{
				Dy::AvbdRigidTriangleSurfaceTriangle& triangle =
					surface.triangles[triangleIndex];
				if(!getTriangleMeshMaterialValues(
						shape, materialManager, *mesh,
						triangle.sourceTriangleIndex, triangle.friction,
						triangle.frictionCombineMode))
					return false;
			}
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertices.size(); ++vertexIndex)
			{
				Dy::AvbdRigidTriangleSurfaceVertex& vertex =
					surface.vertices[vertexIndex];
				if(!getTriangleMeshMaterialValues(
						shape, materialManager, *mesh,
						vertex.sourceTriangleIndex, vertex.friction,
						vertex.frictionCombineMode))
					return false;
			}
			for(PxU32 edgeIndex = 0;
				edgeIndex < surface.edges.size(); ++edgeIndex)
			{
				Dy::AvbdRigidTriangleSurfaceEdge& edge =
					surface.edges[edgeIndex];
				if(!getTriangleMeshMaterialValues(
						shape, materialManager, *mesh,
						edge.sourceTriangleIndex, edge.friction,
						edge.frictionCombineMode))
					return false;
			}
			return true;
		}

		static bool refreshHeightFieldSurfaceMaterials(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxHeightFieldGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface)
		{
			PxHeightField* heightField = geometry.heightField;
			if(!heightField || !geometry.isValid())
				return false;
			for(PxU32 triangleIndex = 0;
				triangleIndex < surface.triangles.size(); ++triangleIndex)
			{
				Dy::AvbdRigidTriangleSurfaceTriangle& triangle =
					surface.triangles[triangleIndex];
				if(!getHeightFieldMaterialValues(
						shape, materialManager, *heightField,
						triangle.sourceTriangleIndex, triangle.friction,
						triangle.frictionCombineMode))
					return false;
			}
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertices.size(); ++vertexIndex)
			{
				Dy::AvbdRigidTriangleSurfaceVertex& vertex =
					surface.vertices[vertexIndex];
				if(!getHeightFieldMaterialValues(
						shape, materialManager, *heightField,
						vertex.sourceTriangleIndex, vertex.friction,
						vertex.frictionCombineMode))
					return false;
			}
			for(PxU32 edgeIndex = 0;
				edgeIndex < surface.edges.size(); ++edgeIndex)
			{
				Dy::AvbdRigidTriangleSurfaceEdge& edge =
					surface.edges[edgeIndex];
				if(!getHeightFieldMaterialValues(
						shape, materialManager, *heightField,
						edge.sourceTriangleIndex, edge.friction,
						edge.frictionCombineMode))
					return false;
			}
			return true;
		}

		static bool triangleMeshTopologyMatches(
			const PxTriangleMeshGeometry& geometry,
			const Dy::AvbdRigidTriangleSurface& surface)
		{
			return surface.topologyGeometryType ==
					PxU8(PxGeometryType::eTRIANGLEMESH) &&
				surface.topologySource == geometry.triangleMesh &&
				sameTriangleSurfaceVec3(
					surface.topologyScale, geometry.scale.scale) &&
				sameTriangleSurfaceQuat(
					surface.topologyScaleRotation,
					geometry.scale.rotation);
		}

		static bool heightFieldTopologyMatches(
			const PxHeightFieldGeometry& geometry,
			const Dy::AvbdRigidTriangleSurface& surface)
		{
			return surface.topologyGeometryType ==
					PxU8(PxGeometryType::eHEIGHTFIELD) &&
				surface.topologySource == geometry.heightField &&
				geometry.heightField &&
				surface.topologyHeightScale == geometry.heightScale &&
				surface.topologyRowScale == geometry.rowScale &&
				surface.topologyColumnScale == geometry.columnScale &&
				surface.topologyContentTimestamp ==
					geometry.heightField->getTimestamp();
		}

		static void setTriangleMeshTopologyIdentity(
			const PxTriangleMeshGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface)
		{
			surface.topologySource = geometry.triangleMesh;
			surface.topologyGeometryType =
				PxU8(PxGeometryType::eTRIANGLEMESH);
			surface.topologyScale = geometry.scale.scale;
			surface.topologyScaleRotation = geometry.scale.rotation;
			surface.topologyHeightScale = 0.0f;
			surface.topologyRowScale = 0.0f;
			surface.topologyColumnScale = 0.0f;
			surface.topologyContentTimestamp = 0;
		}

		static void setHeightFieldTopologyIdentity(
			const PxHeightFieldGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface)
		{
			surface.topologySource = geometry.heightField;
			surface.topologyGeometryType =
				PxU8(PxGeometryType::eHEIGHTFIELD);
			surface.topologyScale = PxVec3(1.0f);
			surface.topologyScaleRotation = PxQuat(PxIdentity);
			surface.topologyHeightScale = geometry.heightScale;
			surface.topologyRowScale = geometry.rowScale;
			surface.topologyColumnScale = geometry.columnScale;
			surface.topologyContentTimestamp =
				geometry.heightField->getTimestamp();
		}

		static bool refreshRigidTriangleSurfaceTopology(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			Dy::AvbdRigidTriangleSurface& surface)
		{
			if(shape.getGeometryType() ==
				PxGeometryType::eTRIANGLEMESH)
			{
				const PxTriangleMeshGeometry& geometry =
					static_cast<const PxTriangleMeshGeometry&>(
						shape.getGeometry());
				if(triangleMeshTopologyMatches(geometry, surface) &&
					refreshTriangleMeshSurfaceMaterials(
						shape, materialManager, geometry, surface))
					return true;
				if(!compileTriangleMeshTopology(
						shape, materialManager, geometry, surface))
					return false;
				setTriangleMeshTopologyIdentity(geometry, surface);
				return true;
			}
			if(shape.getGeometryType() ==
				PxGeometryType::eHEIGHTFIELD)
			{
				const PxHeightFieldGeometry& geometry =
					static_cast<const PxHeightFieldGeometry&>(
						shape.getGeometry());
				if(heightFieldTopologyMatches(geometry, surface) &&
					refreshHeightFieldSurfaceMaterials(
						shape, materialManager, geometry, surface))
					return true;
				if(!compileHeightFieldTopology(
						shape, materialManager, geometry, surface))
					return false;
				setHeightFieldTopologyIdentity(geometry, surface);
				return true;
			}
			return false;
		}

		static bool compileConvexTopology(
			const PxConvexMeshGeometry& geometry,
			Dy::AvbdRigidConvex& convex)
		{
			PxConvexMesh* mesh = geometry.convexMesh;
			if(!mesh ||
				!geometry.scale.isValidForConvexMesh() ||
				!geometry.scale.rotation.isFinite())
				return false;
			const PxU32 vertexCount = mesh->getNbVertices();
			const PxU32 polygonCount = mesh->getNbPolygons();
			const PxVec3* sourceVertices = mesh->getVertices();
			const PxU8* polygonIndices = mesh->getIndexBuffer();
			if(vertexCount < 4 || polygonCount < 4 ||
				!sourceVertices || !polygonIndices)
				return false;

			convex.vertices.resize(vertexCount);
			convex.vertexNormals.resize(vertexCount);
			convex.faces.clear();
			convex.edges.clear();
			convex.triangles.clear();
			PxVec3 centroid(0.0f);
			for(PxU32 vertexIndex = 0;
				vertexIndex < vertexCount; ++vertexIndex)
			{
				const PxVec3 vertex =
					geometry.scale.transform(
						sourceVertices[vertexIndex]);
				if(!vertex.isFinite())
					return false;
				convex.vertices[vertexIndex] = vertex;
				convex.vertexNormals[vertexIndex] =
					PxVec3(0.0f);
				centroid += vertex;
			}
			centroid *= 1.0f / PxReal(vertexCount);

			for(PxU32 polygonIndex = 0;
				polygonIndex < polygonCount; ++polygonIndex)
			{
				PxHullPolygon polygon;
				if(!mesh->getPolygonData(
						polygonIndex, polygon) ||
					polygon.mNbVerts < 3)
					return false;
				const PxU32 firstVertex =
					polygonIndices[polygon.mIndexBase];
				if(firstVertex >= vertexCount)
					return false;
				PxVec3 faceNormal(0.0f);
				PxU32 normalVertex1 = PX_MAX_U32;
				PxU32 normalVertex2 = PX_MAX_U32;
				for(PxU32 localVertex = 1;
					localVertex + 1 < polygon.mNbVerts;
					++localVertex)
				{
					const PxU32 vertex1 =
						polygonIndices[
							polygon.mIndexBase +
							localVertex];
					const PxU32 vertex2 =
						polygonIndices[
							polygon.mIndexBase +
							localVertex + 1];
					if(vertex1 >= vertexCount ||
						vertex2 >= vertexCount)
						return false;
					const PxVec3 candidate =
						(convex.vertices[vertex1] -
							convex.vertices[firstVertex]).
							cross(
								convex.vertices[vertex2] -
								convex.vertices[
									firstVertex]);
					if(candidate.magnitudeSquared() >
						1.0e-12f)
					{
						faceNormal = candidate.getNormalized();
						normalVertex1 = vertex1;
						normalVertex2 = vertex2;
						break;
					}
				}
				if(normalVertex1 == PX_MAX_U32 ||
					normalVertex2 == PX_MAX_U32 ||
					!faceNormal.isFinite())
					return false;
				const bool reverseWinding =
					faceNormal.dot(
						convex.vertices[firstVertex] -
							centroid) < 0.0f;
				if(reverseWinding)
					faceNormal = -faceNormal;

				Dy::AvbdRigidConvexFace face;
				face.normal = faceNormal;
				face.offset =
					faceNormal.dot(
						convex.vertices[firstVertex]);
				const PxU32 faceIndex = convex.faces.size();
				convex.faces.pushBack(face);

				for(PxU32 localVertex = 0;
					localVertex < polygon.mNbVerts;
					++localVertex)
				{
					const PxU32 vertex =
						polygonIndices[
							polygon.mIndexBase +
							localVertex];
					const PxU32 nextVertex =
						polygonIndices[
							polygon.mIndexBase +
							((localVertex + 1) %
								polygon.mNbVerts)];
					if(vertex >= vertexCount ||
						nextVertex >= vertexCount)
						return false;
					convex.vertexNormals[vertex] +=
						faceNormal;
					const PxU32 edge0 =
						PxMin(vertex, nextVertex);
					const PxU32 edge1 =
						PxMax(vertex, nextVertex);
					PxU32 edgeIndex = PX_MAX_U32;
					for(PxU32 candidateIndex = 0;
						candidateIndex < convex.edges.size();
						++candidateIndex)
					{
						if(convex.edges[candidateIndex].p0 ==
								edge0 &&
							convex.edges[candidateIndex].p1 ==
								edge1)
						{
							edgeIndex = candidateIndex;
							break;
						}
					}
					if(edgeIndex == PX_MAX_U32)
					{
						Dy::AvbdRigidConvexEdge edge;
						edge.p0 = edge0;
						edge.p1 = edge1;
						edge.outward = faceNormal;
						convex.edges.pushBack(edge);
					}
					else
						convex.edges[edgeIndex].outward +=
							faceNormal;
				}

				for(PxU32 localTriangle = 0;
					localTriangle + 2 <
						polygon.mNbVerts;
					++localTriangle)
				{
					const PxU32 fan1 =
						polygonIndices[
							polygon.mIndexBase +
							localTriangle + 1];
					const PxU32 fan2 =
						polygonIndices[
							polygon.mIndexBase +
							localTriangle + 2];
					if(fan1 >= vertexCount ||
						fan2 >= vertexCount)
						return false;
					Dy::AvbdRigidConvexTriangle triangle;
					triangle.p0 = firstVertex;
					triangle.p1 = reverseWinding
						? fan2 : fan1;
					triangle.p2 = reverseWinding
						? fan1 : fan2;
					triangle.faceIndex = faceIndex;
					convex.triangles.pushBack(triangle);
				}
			}

			convex.localRadius = 0.0f;
			for(PxU32 vertexIndex = 0;
				vertexIndex < vertexCount; ++vertexIndex)
			{
				const PxReal normalMagnitudeSq =
					convex.vertexNormals[vertexIndex].
						magnitudeSquared();
				if(normalMagnitudeSq <= 1.0e-12f ||
					!PxIsFinite(normalMagnitudeSq))
					return false;
				convex.vertexNormals[vertexIndex] *=
					PxRecipSqrt(normalMagnitudeSq);
				convex.localRadius = PxMax(
					convex.localRadius,
					convex.vertices[vertexIndex].magnitude());
			}
			for(PxU32 edgeIndex = 0;
				edgeIndex < convex.edges.size(); ++edgeIndex)
			{
				const PxReal normalMagnitudeSq =
					convex.edges[edgeIndex].outward.
						magnitudeSquared();
				if(normalMagnitudeSq <= 1.0e-12f ||
					!PxIsFinite(normalMagnitudeSq))
					return false;
				convex.edges[edgeIndex].outward *=
					PxRecipSqrt(normalMagnitudeSq);
			}
			return PxIsFinite(convex.localRadius) &&
				convex.localRadius > 0.0f &&
				!convex.triangles.empty();
		}

		bool compileDynamicConvex(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidConvex& convex) const
		{
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE) ||
				shape.getGeometryType() !=
					PxGeometryType::eCONVEXMESH)
				return false;
			const PxConvexMeshGeometry& geometry =
				static_cast<const PxConvexMeshGeometry&>(
					shape.getGeometry());
			if(!compileConvexTopology(geometry, convex))
				return false;

			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid() ||
				!previousShapeToWorld.isValid())
				return false;
			convex.center = shapeToWorld.p;
			convex.rotation = shapeToWorld.q;
			convex.previousCenter = previousShapeToWorld.p;
			convex.previousRotation = previousShapeToWorld.q;
			convex.friction = getStaticFriction(
				shape, mRigidMaterialManager);
			convex.frictionCombineMode =
				getStaticFrictionCombineMode(
					shape, mRigidMaterialManager);
			convex.primitiveKey = entry.primitiveKey;
			return true;
		}

		bool compileDynamicTriangleSurface(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidTriangleSurface& surface) const
		{
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE))
				return false;
			if(!refreshRigidTriangleSurfaceTopology(
					shape, mRigidMaterialManager, surface))
				return false;

			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid() ||
				!previousShapeToWorld.isValid())
				return false;
			surface.center = shapeToWorld.p;
			surface.rotation = shapeToWorld.q;
			surface.previousCenter = previousShapeToWorld.p;
			surface.previousRotation =
				previousShapeToWorld.q;
			surface.primitiveKey = entry.primitiveKey;
			return true;
		}

		bool compileDynamicBox(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidBox& box) const
		{
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE) ||
				shape.getGeometryType() !=
					PxGeometryType::eBOX)
				return false;

			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
				bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
				bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid())
				return false;

			const PxBoxGeometry& geometry =
				static_cast<const PxBoxGeometry&>(
					shape.getGeometry());
			box.center = shapeToWorld.p;
			box.rotation = shapeToWorld.q;
			box.previousCenter = previousShapeToWorld.p;
			box.previousRotation = previousShapeToWorld.q;
			box.halfExtent = geometry.halfExtents;
			box.friction = getStaticFriction(
				shape, mRigidMaterialManager);
			box.frictionCombineMode =
				getStaticFrictionCombineMode(
					shape, mRigidMaterialManager);
			box.primitiveKey = entry.primitiveKey;
			return true;
		}

		bool compileDynamicSphere(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidSphere& sphere) const
		{
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE) ||
				shape.getGeometryType() !=
					PxGeometryType::eSPHERE)
				return false;

			const PxSphereGeometry& geometry =
				static_cast<const PxSphereGeometry&>(
					shape.getGeometry());
			if(geometry.radius <= 0.0f ||
				!PxIsFinite(geometry.radius))
				return false;
			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid() ||
				!previousShapeToWorld.isValid())
				return false;

			sphere.center = shapeToWorld.p;
			sphere.rotation = shapeToWorld.q;
			sphere.previousCenter = previousShapeToWorld.p;
			sphere.previousRotation = previousShapeToWorld.q;
			sphere.radius = geometry.radius;
			sphere.friction = getStaticFriction(
				shape, mRigidMaterialManager);
			sphere.frictionCombineMode =
				getStaticFrictionCombineMode(
					shape, mRigidMaterialManager);
			sphere.primitiveKey = entry.primitiveKey;
			return true;
		}

		bool compileDynamicCapsule(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidCapsule& capsule) const
		{
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE) ||
				shape.getGeometryType() !=
					PxGeometryType::eCAPSULE)
				return false;

			const PxCapsuleGeometry& geometry =
				static_cast<const PxCapsuleGeometry&>(
					shape.getGeometry());
			if(geometry.radius <= 0.0f ||
				geometry.halfHeight < 0.0f ||
				!PxIsFinite(geometry.radius) ||
				!PxIsFinite(geometry.halfHeight))
				return false;
			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid() ||
				!previousShapeToWorld.isValid())
				return false;

			capsule.center = shapeToWorld.p;
			capsule.rotation = shapeToWorld.q;
			capsule.previousCenter = previousShapeToWorld.p;
			capsule.previousRotation = previousShapeToWorld.q;
			capsule.radius = geometry.radius;
			capsule.halfHeight = geometry.halfHeight;
			capsule.friction = getStaticFriction(
				shape, mRigidMaterialManager);
			capsule.frictionCombineMode =
				getStaticFrictionCombineMode(
					shape, mRigidMaterialManager);
			capsule.primitiveKey = entry.primitiveKey;
			return true;
		}

		static const PxsDeformableVolumeMaterialCore* getMaterial(
			const DeformableVolumeCore& core,
			const PxsDeformableVolumeMaterialManager& materialManager)
		{
			const PxArray<PxU16>& handles =
				core.getCore().materialHandles;
			if(handles.empty() ||
				handles[0] == MATERIAL_INVALID_HANDLE ||
				handles[0] >= materialManager.getMaxSize())
				return NULL;
			const PxsDeformableVolumeMaterialCore* material =
				materialManager.getMaterial(handles[0]);
			return material->mMaterialIndex == handles[0]
				? material : NULL;
		}

		static PxReal getStaticFriction(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager)
		{
			const PxU16* materialIndices =
				shape.getMaterialIndices();
			if(!materialIndices ||
				shape.getNbMaterialIndices() == 0 ||
				materialIndices[0] == MATERIAL_INVALID_HANDLE ||
				materialIndices[0] >= materialManager.getMaxSize())
				return 0.5f;
			const PxsMaterialCore* material =
				materialManager.getMaterial(materialIndices[0]);
			return material->mMaterialIndex == materialIndices[0]
				? PxMax(material->dynamicFriction, 0.0f) : 0.5f;
		}

		static PxU8 getStaticFrictionCombineMode(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager)
		{
			const PxU16* materialIndices =
				shape.getMaterialIndices();
			if(!materialIndices ||
				shape.getNbMaterialIndices() == 0 ||
				materialIndices[0] == MATERIAL_INVALID_HANDLE ||
				materialIndices[0] >= materialManager.getMaxSize())
				return PxU8(PxCombineMode::eAVERAGE);
			const PxsMaterialCore* material =
				materialManager.getMaterial(materialIndices[0]);
			return material->mMaterialIndex == materialIndices[0]
				? PxU8(material->getFrictionCombineMode())
				: PxU8(PxCombineMode::eAVERAGE);
		}

		Dy::AvbdRigidTriangleSurface& getRigidTriangleSurface(
			PxU64 primitiveKey)
		{
			for(PxU32 surfaceIndex = 0;
				surfaceIndex < mRigidTriangleSurfaces.size();
				++surfaceIndex)
			{
				if(mRigidTriangleSurfaces[surfaceIndex].primitiveKey ==
					primitiveKey)
					return mRigidTriangleSurfaces[surfaceIndex];
			}
			const PxU32 newSurfaceIndex =
				mRigidTriangleSurfaces.size();
			mRigidTriangleSurfaces.pushBack(
				Dy::AvbdRigidTriangleSurface());
			mRigidTriangleSurfaces[newSurfaceIndex].primitiveKey =
				primitiveKey;
			return mRigidTriangleSurfaces[newSurfaceIndex];
		}

		void compileWorldStatics(
			const PxsMaterialManager& materialManager)
		{
			mWorldPlanes.clear();
			mRigidBoxes.clear();
			mRigidSpheres.clear();
			mRigidCapsules.clear();
			mRigidConvexes.clear();
			if(++mRigidTriangleSurfaceCompileStamp == 0)
			{
				mRigidTriangleSurfaceCompileStamp = 1;
				for(PxU32 surfaceIndex = 0;
					surfaceIndex < mRigidTriangleSurfaces.size();
					++surfaceIndex)
					mRigidTriangleSurfaces[surfaceIndex].
						sceneCompileStamp = 0;
			}
			const PxU32 triangleSurfaceCompileStamp =
				mRigidTriangleSurfaceCompileStamp;
			PxU32 triangleSurfaceCompileOrder = 0;
			for(PxU32 i = 0; i < mStaticShapes.size(); i++)
			{
				const StaticShapeEntry& entry = mStaticShapes[i];
				const ShapeCore& shape = *entry.shape;
				if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE))
					continue;
				const PxTransform shapeToWorld =
					entry.core->getActor2World() *
					shape.getShape2Actor();
				if(!shapeToWorld.isValid())
					continue;
				const PxReal friction =
					getStaticFriction(shape, materialManager);
				const PxU8 frictionCombineMode =
					getStaticFrictionCombineMode(
						shape, materialManager);
				if(shape.getGeometryType() ==
					PxGeometryType::ePLANE)
				{
					Dy::AvbdWorldPlane plane;
					plane.normal =
						shapeToWorld.q.rotate(
							PxVec3(1.0f, 0.0f, 0.0f)).
							getNormalized();
					plane.offset =
						plane.normal.dot(shapeToWorld.p);
					plane.friction = friction;
					plane.frictionCombineMode =
						frictionCombineMode;
					plane.primitiveKey = entry.primitiveKey;
					mWorldPlanes.pushBack(plane);
				}
				else if(shape.getGeometryType() ==
					PxGeometryType::eBOX)
				{
					const PxBoxGeometry& geometry =
						static_cast<const PxBoxGeometry&>(
							shape.getGeometry());
					Dy::AvbdRigidBox box;
					box.center = shapeToWorld.p;
					box.rotation = shapeToWorld.q;
					box.halfExtent = geometry.halfExtents;
					box.friction = friction;
					box.frictionCombineMode =
						frictionCombineMode;
					box.primitiveKey = entry.primitiveKey;
					mRigidBoxes.pushBack(box);
				}
				else if(shape.getGeometryType() ==
					PxGeometryType::eSPHERE)
				{
					const PxSphereGeometry& geometry =
						static_cast<const PxSphereGeometry&>(
							shape.getGeometry());
					if(geometry.radius <= 0.0f ||
						!PxIsFinite(geometry.radius))
						continue;
					Dy::AvbdRigidSphere sphere;
					sphere.center = shapeToWorld.p;
					sphere.rotation = shapeToWorld.q;
					sphere.radius = geometry.radius;
					sphere.friction = friction;
					sphere.frictionCombineMode =
						frictionCombineMode;
					sphere.primitiveKey = entry.primitiveKey;
					mRigidSpheres.pushBack(sphere);
				}
				else if(shape.getGeometryType() ==
					PxGeometryType::eCAPSULE)
				{
					const PxCapsuleGeometry& geometry =
						static_cast<const PxCapsuleGeometry&>(
							shape.getGeometry());
					if(geometry.radius <= 0.0f ||
						geometry.halfHeight < 0.0f ||
						!PxIsFinite(geometry.radius) ||
						!PxIsFinite(geometry.halfHeight))
						continue;
					Dy::AvbdRigidCapsule capsule;
					capsule.center = shapeToWorld.p;
					capsule.rotation = shapeToWorld.q;
					capsule.radius = geometry.radius;
					capsule.halfHeight = geometry.halfHeight;
					capsule.friction = friction;
					capsule.frictionCombineMode =
						frictionCombineMode;
					capsule.primitiveKey = entry.primitiveKey;
					mRigidCapsules.pushBack(capsule);
				}
				else if(shape.getGeometryType() ==
					PxGeometryType::eCONVEXMESH)
				{
					const PxConvexMeshGeometry& geometry =
						static_cast<
							const PxConvexMeshGeometry&>(
								shape.getGeometry());
					Dy::AvbdRigidConvex convex;
					if(!compileConvexTopology(
							geometry, convex))
						continue;
					convex.center = shapeToWorld.p;
					convex.rotation = shapeToWorld.q;
					convex.previousCenter = shapeToWorld.p;
					convex.previousRotation = shapeToWorld.q;
					convex.friction = friction;
					convex.frictionCombineMode =
						frictionCombineMode;
					convex.primitiveKey =
						entry.primitiveKey;
					mRigidConvexes.pushBack(convex);
				}
				else if(shape.getGeometryType() ==
						PxGeometryType::eTRIANGLEMESH ||
					shape.getGeometryType() ==
						PxGeometryType::eHEIGHTFIELD)
				{
					Dy::AvbdRigidTriangleSurface& surface =
						getRigidTriangleSurface(entry.primitiveKey);
					if(!refreshRigidTriangleSurfaceTopology(
							shape, materialManager, surface))
						continue;
					surface.center = shapeToWorld.p;
					surface.rotation = shapeToWorld.q;
					surface.previousCenter = shapeToWorld.p;
					surface.previousRotation =
						shapeToWorld.q;
					surface.primitiveKey =
						entry.primitiveKey;
					surface.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eWORLD_STATIC;
					surface.targetIndex = PX_MAX_U32;
					surface.shapeToRigidBody = PxTransform(PxIdentity);
					surface.sceneCompileStamp =
						triangleSurfaceCompileStamp;
					surface.sceneCompileOrder =
						triangleSurfaceCompileOrder++;
				}
			}
			for(PxU32 i = 0; i < mDynamicShapes.size(); i++)
			{
				const DynamicShapeEntry& entry = mDynamicShapes[i];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || !bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;
				Dy::AvbdRigidBox box;
				if(compileDynamicBox(entry, box))
				{
					// A prescribed kinematic is a one-way moving position
					// objective.  Its explicit prep owner keeps it out of
					// both world-static warmstart and the rigid 6x6 block.
					box.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					mRigidBoxes.pushBack(box);
					continue;
				}
				Dy::AvbdRigidSphere sphere;
				if(compileDynamicSphere(entry, sphere))
				{
					sphere.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					mRigidSpheres.pushBack(sphere);
					continue;
				}
				Dy::AvbdRigidCapsule capsule;
				if(compileDynamicCapsule(entry, capsule))
				{
					capsule.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					mRigidCapsules.pushBack(capsule);
					continue;
				}
				Dy::AvbdRigidConvex convex;
				if(compileDynamicConvex(entry, convex))
				{
					convex.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					mRigidConvexes.pushBack(convex);
					continue;
				}
				Dy::AvbdRigidTriangleSurface& surface =
					getRigidTriangleSurface(entry.primitiveKey);
				if(compileDynamicTriangleSurface(entry, surface))
				{
					surface.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					surface.targetIndex = PX_MAX_U32;
					surface.shapeToRigidBody = PxTransform(PxIdentity);
					surface.sceneCompileStamp =
						triangleSurfaceCompileStamp;
					surface.sceneCompileOrder =
						triangleSurfaceCompileOrder++;
				}
			}
			// Retire shapes removed from the scene (or whose geometry became
			// invalid) after the complete static-then-kinematic traversal.
			// Reorder the retained cache to that traversal order so contact
			// generation keeps the exact legacy source ordering.
			for(PxU32 surfaceIndex = mRigidTriangleSurfaces.size();
				surfaceIndex > 0; --surfaceIndex)
			{
				if(mRigidTriangleSurfaces[surfaceIndex - 1].
					sceneCompileStamp != triangleSurfaceCompileStamp)
					mRigidTriangleSurfaces.replaceWithLast(
						surfaceIndex - 1);
			}
			for(PxU32 expectedOrder = 0;
				expectedOrder < mRigidTriangleSurfaces.size();
				++expectedOrder)
			{
				for(PxU32 surfaceIndex = expectedOrder + 1;
					surfaceIndex < mRigidTriangleSurfaces.size();
					++surfaceIndex)
				{
					if(mRigidTriangleSurfaces[surfaceIndex].
						sceneCompileOrder == expectedOrder)
					{
						PxSwap(mRigidTriangleSurfaces[expectedOrder],
							mRigidTriangleSurfaces[surfaceIndex]);
						break;
					}
				}
			}
		}

		void compileDynamicBoxesForIsland(
			PxsRigidBody* const* rigidBodies,
			const Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxArray<Dy::AvbdRigidBox>& boxes)
		{
			boxes.clear();
			for(PxU32 shapeIndex = 0;
				shapeIndex < mDynamicShapes.size(); shapeIndex++)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[shapeIndex];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;
				const ShapeCore& shape = *entry.shape;
				if(!(shape.getFlags() &
						PxShapeFlag::eSIMULATION_SHAPE) ||
					shape.getGeometryType() !=
						PxGeometryType::eBOX)
					continue;

				const PxsRigidBody* lowLevelBody =
					&bodySim->getLowLevelBody();
				PxU32 globalBodyIndex = PX_MAX_U32;
				for(PxU32 localBodyIndex = 0;
					localBodyIndex < bodyCount; localBodyIndex++)
				{
					const PxU32 candidateIndex =
						bodyStart + localBodyIndex;
					if(rigidBodies[candidateIndex] ==
						lowLevelBody)
					{
						globalBodyIndex = candidateIndex;
						break;
					}
				}
				if(globalBodyIndex == PX_MAX_U32 ||
					solverBodies[globalBodyIndex].isStatic())
					continue;

				const PxsBodyCore& bodyCore =
					entry.core->getCore();
				const PxTransform actorToWorld =
					bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
				const PxTransform shapeToWorld =
					actorToWorld * shape.getShape2Actor();
				if(!shapeToWorld.isValid())
					continue;

				const PxBoxGeometry& geometry =
					static_cast<const PxBoxGeometry&>(
						shape.getGeometry());
				Dy::AvbdRigidBox box;
				box.center = shapeToWorld.p;
				box.rotation = shapeToWorld.q;
				box.halfExtent = geometry.halfExtents;
				box.friction = getStaticFriction(
					shape, mRigidMaterialManager);
				box.frictionCombineMode =
					getStaticFrictionCombineMode(
						shape, mRigidMaterialManager);
				box.primitiveKey = entry.primitiveKey;
				box.targetKind =
					Dy::AvbdSoftContactTargetKind::eRIGID_BODY;
				box.targetIndex =
					globalBodyIndex - bodyStart;
				box.shapeToRigidBody =
					bodyCore.body2World.getInverse() *
					shapeToWorld;
				boxes.pushBack(box);
			}
		}

		void compileDynamicSpheresForIsland(
			PxsRigidBody* const* rigidBodies,
			Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdRigidSphere>& spheres)
		{
			spheres.clear();
			for(PxU32 shapeIndex = 0;
				shapeIndex < mDynamicShapes.size(); shapeIndex++)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[shapeIndex];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;

				Dy::AvbdRigidSphere sphere;
				if(!compileDynamicSphere(entry, sphere))
					continue;

				const PxsRigidBody* lowLevelBody =
					&bodySim->getLowLevelBody();
				PxU32 globalBodyIndex = PX_MAX_U32;
				for(PxU32 localBodyIndex = 0;
					localBodyIndex < bodyCount; localBodyIndex++)
				{
					const PxU32 candidateIndex =
						bodyStart + localBodyIndex;
					if(rigidBodies[candidateIndex] ==
						lowLevelBody)
					{
						globalBodyIndex = candidateIndex;
						break;
					}
				}
				if(globalBodyIndex == PX_MAX_U32 ||
					solverBodies[globalBodyIndex].isStatic())
					continue;

				const PxsBodyCore& bodyCore =
					entry.core->getCore();
				const PxTransform shapeToWorld(
					sphere.center, sphere.rotation);
				sphere.targetKind =
					Dy::AvbdSoftContactTargetKind::eRIGID_BODY;
				sphere.targetIndex =
					globalBodyIndex - bodyStart;
				sphere.shapeToRigidBody =
					bodyCore.body2World.getInverse() *
						shapeToWorld;
				Dy::AvbdSolverBody& solverBody =
					solverBodies[globalBodyIndex];
				solverBody.computePrediction(dt, gravity);
				const PxTransform predictedBodyToWorld(
					solverBody.predictedPosition,
					solverBody.predictedRotation);
				const PxTransform predictedShapeToWorld =
					predictedBodyToWorld * sphere.shapeToRigidBody;
				if(predictedShapeToWorld.isValid())
				{
					sphere.predictedCenter =
						predictedShapeToWorld.p;
					sphere.predictedRotation =
						predictedShapeToWorld.q;
					sphere.predictedPoseValid = true;
				}
				spheres.pushBack(sphere);
			}
		}

		void compileDynamicCapsulesForIsland(
			PxsRigidBody* const* rigidBodies,
			Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdRigidCapsule>& capsules)
		{
			capsules.clear();
			for(PxU32 shapeIndex = 0;
				shapeIndex < mDynamicShapes.size(); shapeIndex++)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[shapeIndex];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;

				Dy::AvbdRigidCapsule capsule;
				if(!compileDynamicCapsule(entry, capsule))
					continue;

				const PxsRigidBody* lowLevelBody =
					&bodySim->getLowLevelBody();
				PxU32 globalBodyIndex = PX_MAX_U32;
				for(PxU32 localBodyIndex = 0;
					localBodyIndex < bodyCount; localBodyIndex++)
				{
					const PxU32 candidateIndex =
						bodyStart + localBodyIndex;
					if(rigidBodies[candidateIndex] ==
						lowLevelBody)
					{
						globalBodyIndex = candidateIndex;
						break;
					}
				}
				if(globalBodyIndex == PX_MAX_U32 ||
					solverBodies[globalBodyIndex].isStatic())
					continue;

				const PxsBodyCore& bodyCore =
					entry.core->getCore();
				const PxTransform shapeToWorld(
					capsule.center, capsule.rotation);
				capsule.targetKind =
					Dy::AvbdSoftContactTargetKind::eRIGID_BODY;
				capsule.targetIndex =
					globalBodyIndex - bodyStart;
				capsule.shapeToRigidBody =
					bodyCore.body2World.getInverse() *
						shapeToWorld;
				Dy::AvbdSolverBody& solverBody =
					solverBodies[globalBodyIndex];
				solverBody.computePrediction(dt, gravity);
				const PxTransform predictedBodyToWorld(
					solverBody.predictedPosition,
					solverBody.predictedRotation);
				const PxTransform predictedShapeToWorld =
					predictedBodyToWorld *
						capsule.shapeToRigidBody;
				if(predictedShapeToWorld.isValid())
				{
					capsule.predictedCenter =
						predictedShapeToWorld.p;
					capsule.predictedRotation =
						predictedShapeToWorld.q;
					capsule.predictedPoseValid = true;
				}
				capsules.pushBack(capsule);
			}
		}

		void compileDynamicConvexesForIsland(
			PxsRigidBody* const* rigidBodies,
			Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdRigidConvex>& convexes)
		{
			convexes.clear();
			for(PxU32 shapeIndex = 0;
				shapeIndex < mDynamicShapes.size(); ++shapeIndex)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[shapeIndex];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;
				Dy::AvbdRigidConvex convex;
				if(!compileDynamicConvex(entry, convex))
					continue;
				const PxsRigidBody* lowLevelBody =
					&bodySim->getLowLevelBody();
				PxU32 globalBodyIndex = PX_MAX_U32;
				for(PxU32 localBodyIndex = 0;
					localBodyIndex < bodyCount;
					++localBodyIndex)
				{
					const PxU32 candidateIndex =
						bodyStart + localBodyIndex;
					if(rigidBodies[candidateIndex] ==
						lowLevelBody)
					{
						globalBodyIndex = candidateIndex;
						break;
					}
				}
				if(globalBodyIndex == PX_MAX_U32 ||
					solverBodies[globalBodyIndex].isStatic())
					continue;
				const PxsBodyCore& bodyCore =
					entry.core->getCore();
				const PxTransform shapeToWorld(
					convex.center, convex.rotation);
				convex.targetKind =
					Dy::AvbdSoftContactTargetKind::
						eRIGID_BODY;
				convex.targetIndex =
					globalBodyIndex - bodyStart;
				convex.shapeToRigidBody =
					bodyCore.body2World.getInverse() *
						shapeToWorld;
				Dy::AvbdSolverBody& solverBody =
					solverBodies[globalBodyIndex];
				solverBody.computePrediction(dt, gravity);
				const PxTransform predictedBodyToWorld(
					solverBody.predictedPosition,
					solverBody.predictedRotation);
				const PxTransform predictedShapeToWorld =
					predictedBodyToWorld *
						convex.shapeToRigidBody;
				if(predictedShapeToWorld.isValid())
				{
					convex.predictedCenter =
						predictedShapeToWorld.p;
					convex.predictedRotation =
						predictedShapeToWorld.q;
					convex.predictedPoseValid = true;
				}
				convexes.pushBack(convex);
			}
		}

		static bool readTetrahedronIndices(
			const PxTetrahedronMesh& mesh, PxArray<PxU32>& indices)
		{
			const PxU32 indexCount = mesh.getNbTetrahedrons() * 4;
			indices.resize(indexCount);
			const bool has16BitIndices =
				mesh.getTetrahedronMeshFlags() &
					PxTetrahedronMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source = static_cast<const PxU16*>(
					mesh.getTetrahedrons());
				if(!source && indexCount)
					return false;
				for(PxU32 i = 0; i < indexCount; ++i)
					indices[i] = source[i];
			}
			else
			{
				const PxU32* source = static_cast<const PxU32*>(
					mesh.getTetrahedrons());
				if(!source && indexCount)
					return false;
				for(PxU32 i = 0; i < indexCount; ++i)
					indices[i] = source[i];
			}
			for(PxU32 i = 0; i < indexCount; ++i)
				if(indices[i] >= mesh.getNbVertices())
					return false;
			return true;
		}

		static bool readTriangleIndices(
			const PxTriangleMesh& mesh, PxArray<PxU32>& indices)
		{
			const PxU32 indexCount = mesh.getNbTriangles() * 3;
			indices.resize(indexCount);
			const bool has16BitIndices =
				mesh.getTriangleMeshFlags() &
					PxTriangleMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source = static_cast<const PxU16*>(
					mesh.getTriangles());
				if(!source && indexCount)
					return false;
				for(PxU32 i = 0; i < indexCount; ++i)
					indices[i] = source[i];
			}
			else
			{
				const PxU32* source = static_cast<const PxU32*>(
					mesh.getTriangles());
				if(!source && indexCount)
					return false;
				for(PxU32 i = 0; i < indexCount; ++i)
					indices[i] = source[i];
			}
			for(PxU32 i = 0; i < indexCount; ++i)
				if(indices[i] >= mesh.getNbVertices())
					return false;
			return true;
		}

		static void reportInvalidCollisionEmbedding(const char* reason)
		{
			PxGetFoundation().error(
				PxErrorCode::eINVALID_PARAMETER, PX_FL, reason);
		}

		static bool validateVolumeCollisionEmbedding(
			PxTetrahedronMesh& simulationMesh,
			PxTetrahedronMesh& collisionMesh,
			PxDeformableVolumeAuxData& publicAuxData)
		{
			PxArray<PxU32> simulationTets;
			PxArray<PxU32> collisionTets;
			if(!readTetrahedronIndices(simulationMesh, simulationTets) ||
				!readTetrahedronIndices(collisionMesh, collisionTets))
			{
				reportInvalidCollisionEmbedding(
					"CPU AVBD deformable volume has invalid tetrahedron indices.");
				return false;
			}
			const PxU32 collisionVertexCount = collisionMesh.getNbVertices();
			if(&collisionMesh == &simulationMesh)
				return collisionVertexCount == simulationMesh.getNbVertices();

			Gu::DeformableVolumeAuxData& auxData =
				static_cast<Gu::DeformableVolumeAuxData&>(publicAuxData);
			const PxU32* remap = auxData.mVertsRemapInGridModel;
			const PxReal* barycentrics =
				auxData.mVertsBarycentricInGridModel;
			if(!remap || !barycentrics)
			{
				reportInvalidCollisionEmbedding(
					"CPU AVBD requires cooked collision-to-simulation vertex embedding for distinct meshes.");
				return false;
			}

			const PxVec3* simulationVertices = simulationMesh.getVertices();
			const PxVec3* collisionVertices = collisionMesh.getVertices();
			if(!simulationVertices || !collisionVertices)
				return false;
			PxBounds3 collisionBounds = PxBounds3::empty();
			for(PxU32 vertexIndex = 0; vertexIndex < collisionVertexCount;
				++vertexIndex)
				collisionBounds.include(collisionVertices[vertexIndex]);
			const PxReal objectScale = PxMax(
				collisionBounds.getDimensions().magnitude(), 1.0f);
			const PxReal restTolerance = 1.0e-4f * objectScale;
			for(PxU32 vertexIndex = 0; vertexIndex < collisionVertexCount;
				++vertexIndex)
			{
				const PxU32 tetIndex = remap[vertexIndex];
				if(tetIndex >= simulationMesh.getNbTetrahedrons())
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD collision embedding references an invalid simulation tetrahedron.");
					return false;
				}
				PxVec3 embeddedRest(0.0f);
				PxReal weightSum = 0.0f;
				for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
				{
					const PxReal weight = barycentrics[4 * vertexIndex + endpoint];
					if(!PxIsFinite(weight))
					{
						reportInvalidCollisionEmbedding(
							"CPU AVBD collision embedding contains a non-finite barycentric weight.");
						return false;
					}
					const PxU32 localVertex = simulationTets[4 * tetIndex + endpoint];
					embeddedRest += simulationVertices[localVertex] * weight;
					weightSum += weight;
				}
				if(!embeddedRest.isFinite() || !PxIsFinite(weightSum) ||
					PxAbs(weightSum - 1.0f) > 1.0e-3f ||
					(embeddedRest - collisionVertices[vertexIndex]).magnitude() >
						restTolerance)
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD collision embedding fails the cooked rest-position invariant.");
					return false;
				}
			}
			return true;
		}

		static PxVec3 evaluateWeightedParticlePosition(
			const Dy::AvbdWeightedContactPoint& point,
			const Dy::AvbdSoftParticle* particles, PxU32 particleCount,
			PxU32 source)
		{
			PxVec3 value(0.0f);
			for(PxU32 i = 0; i < point.count; ++i)
			{
				const PxU32 particleIndex = point.particleIndices[i];
				if(particleIndex >= particleCount)
					return PxVec3(PX_MAX_F32);
				const Dy::AvbdSoftParticle& particle = particles[particleIndex];
				const PxVec3 sample = source == 0 ? particle.position :
					source == 1 ? particle.predictedPosition :
					source == 2 ? particle.initialPosition :
					particle.outerPosition;
				value += sample * point.weights[i];
			}
			return value;
		}

		bool rebuildCollisionDetectionScene()
		{
			mCollisionParticles.clear();
			mCollisionBodies.clear();
			mCollisionVertexMappings.clear();
			mCollisionSelfCollisionAdjacencies.clear();
			for(PxU32 bodyIndex = 0; bodyIndex < mBodies.size(); ++bodyIndex)
			{
				Entry* entry = NULL;
				for(PxU32 entryIndex = 0; entryIndex < mEntries.size();
					++entryIndex)
					if(mEntries[entryIndex].bodyIndex == bodyIndex)
					{
						entry = &mEntries[entryIndex];
						break;
					}
				if(!entry)
					return false;

				PxArray<PxVec3> vertices;
				PxArray<PxU32> elements;
				const PxU32 sourceParticleStart =
					mBodies[bodyIndex].compiled.particleStart;
				if(entry->kind == eVOLUME)
				{
					if(!validateVolumeCollisionEmbedding(
						*entry->simulationMesh, *entry->collisionMesh,
						*entry->auxData) ||
						!readTetrahedronIndices(*entry->collisionMesh, elements))
						return false;
					const PxU32 vertexCount = entry->collisionMesh->getNbVertices();
					vertices.resize(vertexCount);
					PxArray<PxU32> simulationTets;
					if(!readTetrahedronIndices(*entry->simulationMesh,
						simulationTets))
						return false;
					Gu::DeformableVolumeAuxData& auxData =
						static_cast<Gu::DeformableVolumeAuxData&>(
							*entry->auxData);
					for(PxU32 vertexIndex = 0; vertexIndex < vertexCount;
						++vertexIndex)
					{
						Dy::AvbdWeightedContactPoint mapping;
						if(entry->collisionMesh == entry->simulationMesh)
							mapping.setVertex(sourceParticleStart + vertexIndex);
						else
						{
							const PxU32 tetIndex =
								auxData.mVertsRemapInGridModel[vertexIndex];
							for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
							{
								const PxU32 localParticle =
									simulationTets[4 * tetIndex + endpoint];
								if(!mapping.appendMerged(
									sourceParticleStart + localParticle,
									auxData.mVertsBarycentricInGridModel[
										4 * vertexIndex + endpoint]))
									return false;
							}
							mapping.removeNearZero();
						}
						if(mapping.count == 0)
							return false;
						vertices[vertexIndex] = evaluateWeightedParticlePosition(
							mapping, mParticles.begin(), mParticles.size(), 0);
						mCollisionVertexMappings.pushBack(mapping);
					}
					Dy::avbdCreateSoftBody(
						vertices.begin(), vertices.size(),
						elements.begin(), elements.size(), NULL, 0,
						1.0f, 0.3f, 1.0f, 0.0f, 0.0f, 0.01f,
						mCollisionParticles, mCollisionBodies, false,
						mBodies[bodyIndex].compiled.selfCollisionFilterDistance,
						mBodies[bodyIndex].material.dynamicFriction, false);
				}
				else
				{
					if(!readTriangleIndices(*entry->triangleMesh, elements))
						return false;
					const PxU32 vertexCount = entry->triangleMesh->getNbVertices();
					vertices.resize(vertexCount);
					for(PxU32 vertexIndex = 0; vertexIndex < vertexCount;
						++vertexIndex)
					{
						Dy::AvbdWeightedContactPoint mapping;
						mapping.setVertex(sourceParticleStart + vertexIndex);
						vertices[vertexIndex] =
							mParticles[sourceParticleStart + vertexIndex].position;
						mCollisionVertexMappings.pushBack(mapping);
					}
					Dy::avbdCreateSoftBody(
						vertices.begin(), vertices.size(), NULL, 0,
						elements.begin(), elements.size(),
						1.0f, 0.3f, 1.0f, 0.0f, 0.0f, 0.01f,
						mCollisionParticles, mCollisionBodies, false,
						mBodies[bodyIndex].compiled.selfCollisionFilterDistance,
						mBodies[bodyIndex].material.dynamicFriction);
				}
				if(mCollisionBodies.size() != bodyIndex + 1)
					return false;
				Dy::AvbdSoftBodyCompiledData& collisionCompiled =
					mCollisionBodies[bodyIndex].compiled;
				const Dy::AvbdSoftBodyCompiledData& sourceCompiled =
					mBodies[bodyIndex].compiled;
				collisionCompiled.maxDepenetrationVelocity =
					sourceCompiled.maxDepenetrationVelocity;
				collisionCompiled.selfCollisionStressTolerance =
					sourceCompiled.selfCollisionStressTolerance;
				collisionCompiled.speculativeCCDEnabled =
					sourceCompiled.speculativeCCDEnabled;
			}
			if(mCollisionParticles.size() != mCollisionVertexMappings.size())
				return false;
			Dy::avbdBuildAllSelfCollisionAdjacencies(
				mCollisionBodies.begin(), mCollisionBodies.size(),
				mCollisionSelfCollisionAdjacencies);
			return mCollisionBodies.size() == mBodies.size();
		}

		bool refreshCollisionDetectionScene(
			const Dy::AvbdSoftParticle* sourceParticles,
			PxU32 sourceParticleCount)
		{
			if(mCollisionParticles.size() != mCollisionVertexMappings.size() ||
				mCollisionBodies.size() != mBodies.size())
				return false;
			for(PxU32 vertexIndex = 0;
				vertexIndex < mCollisionParticles.size(); ++vertexIndex)
			{
				const Dy::AvbdWeightedContactPoint& mapping =
					mCollisionVertexMappings[vertexIndex];
				Dy::AvbdSoftParticle& destination =
					mCollisionParticles[vertexIndex];
				destination.position = evaluateWeightedParticlePosition(
					mapping, sourceParticles, sourceParticleCount, 0);
				destination.predictedPosition = evaluateWeightedParticlePosition(
					mapping, sourceParticles, sourceParticleCount, 1);
				destination.initialPosition = evaluateWeightedParticlePosition(
					mapping, sourceParticles, sourceParticleCount, 2);
				destination.outerPosition = evaluateWeightedParticlePosition(
					mapping, sourceParticles, sourceParticleCount, 3);
				destination.velocity = PxVec3(0.0f);
				destination.prevVelocity = PxVec3(0.0f);
				bool dynamic = false;
				for(PxU32 endpoint = 0; endpoint < mapping.count; ++endpoint)
				{
					const PxU32 sourceIndex = mapping.particleIndices[endpoint];
					if(sourceIndex >= sourceParticleCount)
						return false;
					destination.velocity += sourceParticles[sourceIndex].velocity *
						mapping.weights[endpoint];
					destination.prevVelocity +=
						sourceParticles[sourceIndex].prevVelocity *
						mapping.weights[endpoint];
					dynamic = dynamic || sourceParticles[sourceIndex].invMass > 0.0f;
				}
				destination.invMass = dynamic ? 1.0f : 0.0f;
				destination.mass = dynamic ? 1.0f : 0.0f;
				if(!destination.position.isFinite() ||
					!destination.predictedPosition.isFinite() ||
					!destination.initialPosition.isFinite() ||
					!destination.velocity.isFinite())
					return false;
			}
			for(PxU32 bodyIndex = 0; bodyIndex < mBodies.size(); ++bodyIndex)
			{
				mCollisionBodies[bodyIndex].compiled.speculativeCCDEnabled =
					mBodies[bodyIndex].compiled.speculativeCCDEnabled;
				mCollisionBodies[bodyIndex].compiled.maxDepenetrationVelocity =
					mBodies[bodyIndex].compiled.maxDepenetrationVelocity;
				mCollisionBodies[bodyIndex].compiled.selfCollisionStressTolerance =
					mBodies[bodyIndex].compiled.selfCollisionStressTolerance;
			}
			// Keep the public collision buffer synchronized with the exact same
			// evaluated positions consumed by this detection epoch.
			if(sourceParticles == mParticles.begin() &&
				sourceParticleCount == mParticles.size())
			{
				for(PxU32 entryIndex = 0; entryIndex < mEntries.size();
					++entryIndex)
				{
					Entry& entry = mEntries[entryIndex];
					if(entry.kind != eVOLUME ||
						entry.bodyIndex >= mCollisionBodies.size())
						continue;
					Dy::DeformableVolumeCore& core =
						entry.volumeCore->getCore();
					const Dy::AvbdSoftBodyCompiledData& collisionCompiled =
						mCollisionBodies[entry.bodyIndex].compiled;
					for(PxU32 localVertex = 0;
						localVertex < collisionCompiled.particleCount;
						++localVertex)
					{
						const PxReal invMass = core.positionInvMass[localVertex].w;
						core.positionInvMass[localVertex] = PxVec4(
							mCollisionParticles[
								collisionCompiled.particleStart + localVertex].position,
							invMass);
					}
				}
			}
			return true;
		}

		bool expandCollisionDetectionPoint(
			const PxU32* proxyIndices, const PxReal* proxyWeights,
			PxU32 proxyCount,
			const PxArray<Dy::AvbdWeightedContactPoint>& vertexMappings,
			Dy::AvbdWeightedContactPoint& output) const
		{
			output.clear();
			for(PxU32 proxyOrder = 0; proxyOrder < proxyCount; ++proxyOrder)
			{
				const PxU32 proxyIndex = proxyIndices[proxyOrder];
				if(proxyIndex >= vertexMappings.size() ||
					!PxIsFinite(proxyWeights[proxyOrder]))
					return false;
				const Dy::AvbdWeightedContactPoint& vertexMapping =
					vertexMappings[proxyIndex];
				for(PxU32 endpoint = 0; endpoint < vertexMapping.count; ++endpoint)
					if(!output.appendMerged(
						vertexMapping.particleIndices[endpoint],
						proxyWeights[proxyOrder] * vertexMapping.weights[endpoint]))
						return false;
			}
			output.removeNearZero();
			return output.count != 0;
		}

		bool rebuildSubsetCollisionDetectionScene(
			const Dy::AvbdSoftParticle* sourceParticles,
			PxU32 sourceParticleCount,
			const Dy::AvbdSoftBody* sourceBodies, PxU32 sourceBodyCount,
			ActorCore* const* softCores)
		{
			if(!sourceParticles || !sourceBodies || !softCores ||
				sourceBodyCount == 0 ||
				mCollisionBodies.size() != mBodies.size())
				return false;
			mSubsetCollisionParticles.clear();
			mSubsetCollisionBodies.clear();
			mSubsetCollisionVertexMappings.clear();
			mSubsetCollisionSelfCollisionAdjacencies.clear();

			for(PxU32 localBodyIndex = 0; localBodyIndex < sourceBodyCount;
				++localBodyIndex)
			{
				Entry* entry = softCores[localBodyIndex]
					? findEntry(*softCores[localBodyIndex]) : NULL;
				if(!entry || entry->bodyIndex >= mCollisionBodies.size())
					return false;
				const Dy::AvbdSoftBody& globalCollisionBody =
					mCollisionBodies[entry->bodyIndex];
				const PxU32 oldCollisionStart =
					globalCollisionBody.compiled.particleStart;
				const PxU32 collisionVertexCount =
					globalCollisionBody.compiled.particleCount;
				if(oldCollisionStart > mCollisionParticles.size() ||
					collisionVertexCount >
						mCollisionParticles.size() - oldCollisionStart)
					return false;
				const PxU32 newCollisionStart =
					mSubsetCollisionParticles.size();
				for(PxU32 localVertex = 0;
					localVertex < collisionVertexCount; ++localVertex)
					mSubsetCollisionParticles.pushBack(
						mCollisionParticles[oldCollisionStart + localVertex]);
				mSubsetCollisionBodies.pushBack(globalCollisionBody);
				if(!rebaseSoftBodyParticleRangeInPlace(
					mSubsetCollisionBodies.back(), oldCollisionStart,
					collisionVertexCount, newCollisionStart))
					return false;
				Dy::AvbdSoftBodyCompiledData& subsetCompiled =
					mSubsetCollisionBodies.back().compiled;
				const Dy::AvbdSoftBodyCompiledData& sourceCompiled =
					sourceBodies[localBodyIndex].compiled;
				// The global collision proxy owns immutable cooked topology, but its
				// frame-varying body controls may predate the latest public host sync.
				// A selected island must consume the same controls as its authoritative
				// simulation body, just like refreshCollisionDetectionScene() does for
				// the full-scene path.
				subsetCompiled.speculativeCCDEnabled =
					sourceCompiled.speculativeCCDEnabled;
				subsetCompiled.maxDepenetrationVelocity =
					sourceCompiled.maxDepenetrationVelocity;
				subsetCompiled.selfCollisionStressTolerance =
					sourceCompiled.selfCollisionStressTolerance;

				const PxU32 sourceParticleStart =
					sourceBodies[localBodyIndex].compiled.particleStart;
				if(entry->kind == eVOLUME)
				{
					PxArray<PxU32> simulationTets;
					if(!readTetrahedronIndices(
						*entry->simulationMesh, simulationTets))
						return false;
					Gu::DeformableVolumeAuxData& auxData =
						static_cast<Gu::DeformableVolumeAuxData&>(
							*entry->auxData);
					for(PxU32 localVertex = 0;
						localVertex < collisionVertexCount; ++localVertex)
					{
						Dy::AvbdWeightedContactPoint mapping;
						if(entry->collisionMesh == entry->simulationMesh)
							mapping.setVertex(sourceParticleStart + localVertex);
						else
						{
							const PxU32 tetIndex =
								auxData.mVertsRemapInGridModel[localVertex];
							if(tetIndex >= entry->simulationMesh->getNbTetrahedrons())
								return false;
							for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
							{
								const PxU32 localParticle =
									simulationTets[4 * tetIndex + endpoint];
								if(!mapping.appendMerged(
									sourceParticleStart + localParticle,
									auxData.mVertsBarycentricInGridModel[
										4 * localVertex + endpoint]))
									return false;
							}
							mapping.removeNearZero();
						}
						mSubsetCollisionVertexMappings.pushBack(mapping);
					}
				}
				else
				{
					for(PxU32 localVertex = 0;
						localVertex < collisionVertexCount; ++localVertex)
					{
						Dy::AvbdWeightedContactPoint mapping;
						mapping.setVertex(sourceParticleStart + localVertex);
						mSubsetCollisionVertexMappings.pushBack(mapping);
					}
				}
			}
			if(mSubsetCollisionParticles.size() !=
				mSubsetCollisionVertexMappings.size())
				return false;

			for(PxU32 vertexIndex = 0;
				vertexIndex < mSubsetCollisionParticles.size(); ++vertexIndex)
			{
				const Dy::AvbdWeightedContactPoint& mapping =
					mSubsetCollisionVertexMappings[vertexIndex];
				Dy::AvbdSoftParticle& destination =
					mSubsetCollisionParticles[vertexIndex];
				destination.position = evaluateWeightedParticlePosition(
					mapping, sourceParticles, sourceParticleCount, 0);
				destination.predictedPosition = evaluateWeightedParticlePosition(
					mapping, sourceParticles, sourceParticleCount, 1);
				destination.initialPosition = evaluateWeightedParticlePosition(
					mapping, sourceParticles, sourceParticleCount, 2);
				destination.outerPosition = evaluateWeightedParticlePosition(
					mapping, sourceParticles, sourceParticleCount, 3);
				destination.velocity = PxVec3(0.0f);
				destination.prevVelocity = PxVec3(0.0f);
				bool dynamic = false;
				for(PxU32 endpoint = 0; endpoint < mapping.count; ++endpoint)
				{
					const PxU32 sourceIndex = mapping.particleIndices[endpoint];
					if(sourceIndex >= sourceParticleCount)
						return false;
					destination.velocity += sourceParticles[sourceIndex].velocity *
						mapping.weights[endpoint];
					destination.prevVelocity +=
						sourceParticles[sourceIndex].prevVelocity *
						mapping.weights[endpoint];
					dynamic = dynamic || sourceParticles[sourceIndex].invMass > 0.0f;
				}
				destination.invMass = dynamic ? 1.0f : 0.0f;
				destination.mass = dynamic ? 1.0f : 0.0f;
			}
			Dy::avbdBuildAllSelfCollisionAdjacencies(
				mSubsetCollisionBodies.begin(),
				mSubsetCollisionBodies.size(),
				mSubsetCollisionSelfCollisionAdjacencies);
			return mSubsetCollisionBodies.size() == sourceBodyCount;
		}

		PxU32 findCollisionBodyForParticle(
			PxU32 particleIndex,
			const PxArray<Dy::AvbdSoftBody>& collisionBodies) const
		{
			for(PxU32 bodyIndex = 0; bodyIndex < collisionBodies.size();
				++bodyIndex)
			{
				const Dy::AvbdSoftBodyCompiledData& compiled =
					collisionBodies[bodyIndex].compiled;
				if(particleIndex >= compiled.particleStart &&
					particleIndex - compiled.particleStart < compiled.particleCount)
					return bodyIndex;
			}
			return PX_MAX_U32;
		}

		PxU32 resolveCollisionElementForFeature(
			const Dy::AvbdSoftContactGeometry& geometry,
			const Dy::AvbdSoftBodyCompiledData& compiled,
			PxU32 collisionFeatureParticle) const
		{
			PxU32 featureVertices[3] =
			{
				collisionFeatureParticle, PX_MAX_U32, PX_MAX_U32
			};
			PxU32 featureVertexCount = 1;
			if(geometry.hasBarycentricQueryPoint())
			{
				featureVertexCount = 0;
				while(featureVertexCount < 3 &&
					geometry.queryParticleIndices[featureVertexCount] !=
						PX_MAX_U32)
				{
					featureVertices[featureVertexCount] =
						geometry.queryParticleIndices[featureVertexCount];
					++featureVertexCount;
				}
			}
			if(featureVertexCount == 0 ||
				compiled.surfaceTriangleElementIndices.size() !=
					compiled.surfaceTriangles.size() / 3)
				return PX_MAX_U32;

			PxU32 owner = PX_MAX_U32;
			for(PxU32 triangleIndex = 0;
				triangleIndex < compiled.surfaceTriangles.size() / 3;
				++triangleIndex)
			{
				const PxU32* triangle =
					compiled.surfaceTriangles.begin() + 3 * triangleIndex;
				bool containsFeature = true;
				for(PxU32 featureIndex = 0;
					featureIndex < featureVertexCount; ++featureIndex)
				{
					const PxU32 featureVertex =
						featureVertices[featureIndex];
					if(triangle[0] != featureVertex &&
						triangle[1] != featureVertex &&
						triangle[2] != featureVertex)
					{
						containsFeature = false;
						break;
					}
				}
				if(containsFeature)
					owner = PxMin(
						owner,
						compiled.surfaceTriangleElementIndices[triangleIndex]);
			}
			return owner;
		}

		bool expandCollisionDetectionContacts(
			PxArray<Dy::AvbdSoftContact>& contacts,
			PxU32 simulationParticleCount,
			const PxArray<Dy::AvbdWeightedContactPoint>& vertexMappings,
			const PxArray<Dy::AvbdSoftBody>& collisionBodies) const
		{
			for(PxU32 contactIndex = 0; contactIndex < contacts.size();
				++contactIndex)
			{
				Dy::AvbdSoftContactGeometry& geometry =
					contacts[contactIndex].geometry;
				const PxU32 collisionFeatureParticle = geometry.particleIdx;
				geometry.collisionFeatureParticleIdx = collisionFeatureParticle;
				geometry.queryBodyIndex =
					findCollisionBodyForParticle(
						collisionFeatureParticle, collisionBodies);
				if(geometry.queryBodyIndex == PX_MAX_U32)
					return false;
				if(geometry.hasBarycentricQueryPoint())
				{
					PxU32 count = 0;
					while(count < 3 && geometry.queryParticleIndices[count] !=
						PX_MAX_U32)
						++count;
					if(!expandCollisionDetectionPoint(
						geometry.queryParticleIndices, geometry.queryWeights,
						count, vertexMappings, geometry.queryPoint))
						return false;
					// A triangle-core OGC row needs more than its centroid once the
					// nonlinear island solver starts moving both endpoints. Preserve
					// the independent embedded supports of all three proxy vertices so
					// that the pair trust region can test the whole triangle at the
					// candidate soft/rigid poses.
					if(geometry.hasRigidBoxTriangleCoreExit)
					{
						if(count != 3)
							return false;
						for(PxU32 vertex = 0; vertex < 3; ++vertex)
						{
							const PxU32 proxyIndex[1] =
							{
								geometry.queryParticleIndices[vertex]
							};
							const PxReal proxyWeight[1] = {1.0f};
							if(!expandCollisionDetectionPoint(
								proxyIndex, proxyWeight, 1, vertexMappings,
								geometry.rigidBoxTriangleCorePoints[vertex]))
								return false;
							for(PxU32 endpoint = 0;
								endpoint < geometry.rigidBoxTriangleCorePoints[vertex].count;
								++endpoint)
							{
								if(geometry.rigidBoxTriangleCorePoints[vertex]
									.particleIndices[endpoint] >= simulationParticleCount)
									return false;
							}
						}
					}
				}
				else
				{
					const PxU32 proxyIndices[1] = {collisionFeatureParticle};
					const PxReal proxyWeights[1] = {1.0f};
					if(!expandCollisionDetectionPoint(
						proxyIndices, proxyWeights, 1, vertexMappings,
						geometry.queryPoint))
						return false;
				}
				for(PxU32 endpoint = 0; endpoint < geometry.queryPoint.count;
					++endpoint)
					if(geometry.queryPoint.particleIndices[endpoint] >=
						simulationParticleCount)
						return false;
				geometry.particleIdx = geometry.queryPoint.particleIndices[0];

				const Dy::AvbdSoftBodyCompiledData& queryCompiled =
					collisionBodies[geometry.queryBodyIndex].compiled;
				geometry.queryCollisionElementIndex =
					resolveCollisionElementForFeature(
						geometry, queryCompiled, collisionFeatureParticle);
				if(geometry.queryCollisionElementIndex == PX_MAX_U32 &&
					collisionFeatureParticle >= queryCompiled.particleStart)
				{
					const PxU32 localParticle =
						collisionFeatureParticle - queryCompiled.particleStart;
					if(localParticle < queryCompiled.elementAdjacency.size())
					{
						const PxArray<Dy::AvbdParticleElementRef>& refs =
							queryCompiled.triElements.empty()
								? queryCompiled.elementAdjacency[localParticle].tetRefs
								: queryCompiled.elementAdjacency[localParticle].triRefs;
						if(!refs.empty())
						{
							const PxU32 compiledElement = refs[0].index;
							if(queryCompiled.triElements.empty() &&
								compiledElement < queryCompiled.tetElements.size())
								geometry.queryCollisionElementIndex =
									queryCompiled.tetElements[compiledElement].sourceElementIndex;
							else if(compiledElement < queryCompiled.triElements.size())
								geometry.queryCollisionElementIndex =
									queryCompiled.triElements[compiledElement].sourceElementIndex;
						}
					}
				}

				if(geometry.hasDeformableSurfaceTarget())
				{
					PxU32 count = 0;
					while(count < 3 && geometry.surfaceParticleIndices[count] !=
						PX_MAX_U32)
						++count;
					if(!expandCollisionDetectionPoint(
						geometry.surfaceParticleIndices, geometry.surfaceWeights,
						count, vertexMappings, geometry.targetPoint))
						return false;
					for(PxU32 endpoint = 0; endpoint < geometry.targetPoint.count;
						++endpoint)
						if(geometry.targetPoint.particleIndices[endpoint] >=
							simulationParticleCount)
							return false;
					geometry.targetCollisionElementIndex =
						geometry.targetSourceElementIndex;
				}
			}
			return true;
		}

		void refreshSelfCollisionEnabled()
		{
			mSelfCollisionEnabled.resize(mBodies.size());
			for(PxU32 i = 0; i < mSelfCollisionEnabled.size(); i++)
				mSelfCollisionEnabled[i] = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				const Entry& entry = mEntries[i];
				if(entry.bodyIndex < mSelfCollisionEnabled.size())
				{
					mSelfCollisionEnabled[entry.bodyIndex] =
						(entry.getBodyCore().bodyFlags &
							PxDeformableBodyFlag::
								eDISABLE_SELF_COLLISION)
						? 0u : 1u;
				}
			}
		}

		ActorCore* findRigidCoreForPrimitive(
			PxU64 primitiveKey) const
		{
			for(PxU32 i = 0; i < mStaticShapes.size(); ++i)
				if(mStaticShapes[i].primitiveKey == primitiveKey)
					return static_cast<ActorCore*>(
						mStaticShapes[i].core);
			for(PxU32 i = 0; i < mDynamicShapes.size(); ++i)
				if(mDynamicShapes[i].primitiveKey == primitiveKey)
					return static_cast<ActorCore*>(
						mDynamicShapes[i].core);
			return NULL;
		}

		ActorCore* findSoftCoreForContactBody(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			ActorCore* const* softCores,
			PxU32 particleIndex) const
		{
			for(PxU32 bodyIndex = 0;
				bodyIndex < numBodies; ++bodyIndex)
			{
				const Dy::AvbdSoftBodyCompiledData& compiled =
					bodies[bodyIndex].compiled;
				if(particleIndex < compiled.particleStart ||
					particleIndex >=
						compiled.particleStart +
						compiled.particleCount)
					continue;
				if(softCores)
					return softCores[bodyIndex];
				if(bodies == mBodies.begin() ||
					bodies == mCollisionBodies.begin())
				{
					for(PxU32 entryIndex = 0;
						entryIndex < mEntries.size(); ++entryIndex)
					{
						const Entry& entry = mEntries[entryIndex];
						if(entry.bodyIndex == bodyIndex)
							return entry.getActorCore();
					}
				}
				return NULL;
			}
			return NULL;
		}

		ActorCore* findSoftCoreForContactBodyIndex(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			ActorCore* const* softCores,
			PxU32 bodyIndex) const
		{
			if(bodyIndex >= numBodies)
				return NULL;
			if(softCores)
				return softCores[bodyIndex];
			if(bodies == mBodies.begin() ||
				bodies == mCollisionBodies.begin())
			{
				for(PxU32 entryIndex = 0;
					entryIndex < mEntries.size(); ++entryIndex)
				{
					const Entry& entry = mEntries[entryIndex];
					if(entry.bodyIndex == bodyIndex)
						return entry.getActorCore();
				}
			}
			return NULL;
		}

		const Dy::AvbdSoftBody* findSoftBodyForContactParticle(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			PxU32 particleIndex) const
		{
			for(PxU32 bodyIndex = 0;
				bodyIndex < numBodies; ++bodyIndex)
			{
				const Dy::AvbdSoftBodyCompiledData& compiled =
					bodies[bodyIndex].compiled;
				if(particleIndex >= compiled.particleStart &&
					particleIndex <
						compiled.particleStart +
							compiled.particleCount)
					return &bodies[bodyIndex];
			}
			return NULL;
		}

		bool isRigidActorContactFiltered(
			const Dy::AvbdSoftBody& body,
			ActorCore& softCore,
			ActorCore& rigidCore,
			PxU32 particleIndex) const
		{
			bool hasMatchingFilter = false;
			for(PxU32 filterIndex = 0;
				filterIndex < mRigidActorFilters.size();
				++filterIndex)
			{
				const RigidActorFilterEntry& filter =
					mRigidActorFilters[filterIndex];
				if(filter.softCore != &softCore ||
					filter.rigidCore != &rigidCore)
					continue;
				hasMatchingFilter = true;
				if(filter.filterAllElements)
					return true;
			}
			if(!hasMatchingFilter ||
				particleIndex < body.compiled.particleStart)
				return false;
			const PxU32 localParticle =
				particleIndex - body.compiled.particleStart;
			if(localParticle >=
				body.compiled.elementAdjacency.size())
				return false;

			// Rigid contact generation is particle-sampled on the collision
			// proxy. Surface filters own public triangles and Volume filters own
			// public collision tetrahedra directly. In both domains, remove the
			// objective only when every incident
			// element is covered by the union of active filter objects.
			const PxArray<Dy::AvbdParticleElementRef>& incident =
				body.compiled.triElements.empty()
					? body.compiled.elementAdjacency[
						localParticle].tetRefs
					: body.compiled.elementAdjacency[
						localParticle].triRefs;
			if(incident.empty())
				return false;
			const bool volumeOwnership =
				body.compiled.triElements.empty();
			for(PxU32 refIndex = 0;
				refIndex < incident.size(); ++refIndex)
			{
				const PxU32 compiledElementIndex =
					incident[refIndex].index;
				PxU32 sourceElementIndex = PX_MAX_U32;
				if(volumeOwnership)
				{
					if(compiledElementIndex >=
						body.compiled.tetElements.size())
						return false;
					sourceElementIndex =
						body.compiled.tetElements[
							compiledElementIndex].
								sourceElementIndex;
				}
				else
				{
					if(compiledElementIndex >=
						body.compiled.triElements.size())
						return false;
					sourceElementIndex =
						body.compiled.triElements[
							compiledElementIndex].
								sourceElementIndex;
				}
				bool elementFiltered = false;
				for(PxU32 filterIndex = 0;
					filterIndex < mRigidActorFilters.size();
					++filterIndex)
				{
					const RigidActorFilterEntry& filter =
						mRigidActorFilters[filterIndex];
					if(filter.softCore == &softCore &&
						filter.rigidCore == &rigidCore &&
						filter.containsElement(
							sourceElementIndex))
					{
						elementFiltered = true;
						break;
					}
				}
				if(!elementFiltered)
					return false;
			}
			return true;
		}

		bool isDeformablePairContactFiltered(
			const Dy::AvbdSoftBody& queryBody,
			ActorCore& queryCore,
			ActorCore& targetCore,
			PxU32 queryParticleIndex,
			PxU32 targetSourceElementIndex) const
		{
			if(targetSourceElementIndex == PX_MAX_U32 ||
				queryParticleIndex <
					queryBody.compiled.particleStart)
				return false;
			const PxU32 localParticle =
				queryParticleIndex -
					queryBody.compiled.particleStart;
			if(localParticle >=
				queryBody.compiled.elementAdjacency.size())
				return false;
			const bool volumeOwnership =
				queryBody.compiled.triElements.empty();
			const PxArray<Dy::AvbdParticleElementRef>& incident =
				volumeOwnership
					? queryBody.compiled.elementAdjacency[
						localParticle].tetRefs
					: queryBody.compiled.elementAdjacency[
						localParticle].triRefs;
			if(incident.empty())
				return false;

			// Contact detection samples a query particle against one
			// explicit target boundary face. A shared query particle
			// belongs to every incident source element, so the prepared
			// objective is removed only when the union of active filter
			// objects covers every query/target source-element pair.
			for(PxU32 refIndex = 0;
				refIndex < incident.size(); ++refIndex)
			{
				const PxU32 compiledElementIndex =
					incident[refIndex].index;
				PxU32 querySourceElementIndex = PX_MAX_U32;
				if(volumeOwnership)
				{
					if(compiledElementIndex >=
						queryBody.compiled.tetElements.size())
						return false;
					querySourceElementIndex =
						queryBody.compiled.tetElements[
							compiledElementIndex].
								sourceElementIndex;
				}
				else
				{
					if(compiledElementIndex >=
						queryBody.compiled.triElements.size())
						return false;
					querySourceElementIndex =
						queryBody.compiled.triElements[
							compiledElementIndex].
								sourceElementIndex;
				}
				bool pairFiltered = false;
				for(PxU32 filterIndex = 0;
					filterIndex < mDeformablePairFilters.size();
					++filterIndex)
				{
					if(mDeformablePairFilters[filterIndex].
						containsPair(
							queryCore,
							querySourceElementIndex,
							targetCore,
							targetSourceElementIndex))
					{
						pairFiltered = true;
						break;
					}
				}
				if(!pairFiltered)
					return false;
			}
			return true;
		}

		void removeRigidActorFilteredContacts(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			ActorCore* const* softCores,
			PxArray<Dy::AvbdSoftContact>& contacts) const
		{
			if(mRigidActorFilters.empty())
				return;
			PxU32 writeIndex = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < contacts.size(); ++contactIndex)
			{
				const Dy::AvbdSoftContact& contact =
					contacts[contactIndex];
				const Dy::AvbdSoftContactGeometry& geometry =
					contact.geometry;
				const bool rigidSource =
					geometry.source.type ==
						Dy::AvbdSoftContactSource::eGROUND ||
					geometry.source.type ==
						Dy::AvbdSoftContactSource::eRIGID_SDF;
				bool filtered = false;
				if(rigidSource)
				{
					ActorCore* softCore =
						findSoftCoreForContactBody(
							bodies, numBodies, softCores,
							geometry.particleIdx);
					ActorCore* rigidCore =
						findRigidCoreForPrimitive(
							geometry.source.primitiveKey);
					const Dy::AvbdSoftBody* softBody =
						findSoftBodyForContactParticle(
							bodies, numBodies,
							geometry.particleIdx);
					if(softCore && rigidCore && softBody)
						filtered =
							isRigidActorContactFiltered(
								*softBody, *softCore,
								*rigidCore,
								geometry.particleIdx);
				}
				if(!filtered)
				{
					if(writeIndex != contactIndex)
						contacts[writeIndex] =
							contacts[contactIndex];
					++writeIndex;
				}
			}
			contacts.resize(writeIndex);
		}

		void removeDeformablePairFilteredContacts(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			ActorCore* const* softCores,
			PxArray<Dy::AvbdSoftContact>& contacts) const
		{
			if(mDeformablePairFilters.empty())
				return;
			PxU32 writeIndex = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < contacts.size(); ++contactIndex)
			{
				const Dy::AvbdSoftContact& contact =
					contacts[contactIndex];
				const Dy::AvbdSoftContactGeometry& geometry =
					contact.geometry;
				bool filtered = false;
				if(geometry.source.type ==
					Dy::AvbdSoftContactSource::eSOFT_SURFACE)
				{
					ActorCore* queryCore =
						findSoftCoreForContactBody(
							bodies, numBodies, softCores,
							geometry.particleIdx);
					ActorCore* targetCore =
						findSoftCoreForContactBodyIndex(
							bodies, numBodies, softCores,
							geometry.source.targetBodyIndex);
					const Dy::AvbdSoftBody* queryBody =
						findSoftBodyForContactParticle(
							bodies, numBodies,
							geometry.particleIdx);
					if(queryCore && targetCore && queryBody)
						filtered =
							isDeformablePairContactFiltered(
								*queryBody, *queryCore,
								*targetCore,
								geometry.particleIdx,
								geometry.
									targetSourceElementIndex);
				}
				if(!filtered)
				{
					if(writeIndex != contactIndex)
						contacts[writeIndex] =
							contacts[contactIndex];
					++writeIndex;
				}
			}
			contacts.resize(writeIndex);
		}

		void detectContacts(
			Dy::AvbdSoftParticle* particles,
			PxU32 numParticles,
			Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			PxArray<Dy::AvbdSoftContact>& contacts,
			const Dy::AvbdRigidBox* rigidBoxes = NULL,
			PxU32 numRigidBoxes = 0,
			const Dy::AvbdSelfCollisionAdjacency*
				selfCollisionAdjacencies = NULL,
			PxU32 numSelfCollisionAdjacencies = 0,
			const PxU8* selfCollisionEnabled = NULL,
			ActorCore* const* softCores = NULL,
			const Dy::AvbdRigidSphere* rigidSpheres = NULL,
			PxU32 numRigidSpheres = 0,
			const Dy::AvbdRigidCapsule* rigidCapsules = NULL,
			PxU32 numRigidCapsules = 0,
			const Dy::AvbdRigidConvex* rigidConvexes = NULL,
			PxU32 numRigidConvexes = 0,
			const Dy::AvbdRigidTriangleSurface*
				rigidTriangleSurfaces = NULL,
			PxU32 numRigidTriangleSurfaces = 0)
		{
			if(!rigidBoxes)
			{
				rigidBoxes = mRigidBoxes.begin();
				numRigidBoxes = mRigidBoxes.size();
			}
			if(!rigidSpheres)
			{
				rigidSpheres = mRigidSpheres.begin();
				numRigidSpheres = mRigidSpheres.size();
			}
			if(!rigidCapsules)
			{
				rigidCapsules = mRigidCapsules.begin();
				numRigidCapsules = mRigidCapsules.size();
			}
			if(!rigidConvexes)
			{
				rigidConvexes = mRigidConvexes.begin();
				numRigidConvexes = mRigidConvexes.size();
			}
			if(!rigidTriangleSurfaces)
			{
				rigidTriangleSurfaces =
					mRigidTriangleSurfaces.begin();
				numRigidTriangleSurfaces =
					mRigidTriangleSurfaces.size();
			}
			if(!selfCollisionAdjacencies &&
				bodies == mBodies.begin() &&
				numBodies == mBodies.size())
			{
				PX_ASSERT(
					mSelfCollisionAdjacencies.size() == mBodies.size());
				refreshSelfCollisionEnabled();
				selfCollisionAdjacencies =
					mSelfCollisionAdjacencies.begin();
				numSelfCollisionAdjacencies =
					mSelfCollisionAdjacencies.size();
				selfCollisionEnabled = mSelfCollisionEnabled.begin();
			}
			const bool fullSceneCollisionRequest =
				particles == mParticles.begin() &&
				numParticles == mParticles.size() &&
				bodies == mBodies.begin() &&
				numBodies == mBodies.size();
			if(fullSceneCollisionRequest)
			{
				bool exactSimulationCollisionDomain = true;
				for(PxU32 entryIndex = 0;
					entryIndex < mEntries.size(); ++entryIndex)
				{
					const Entry& entry = mEntries[entryIndex];
					if(entry.kind == eVOLUME &&
						entry.collisionMesh != entry.simulationMesh)
					{
						exactSimulationCollisionDomain = false;
						break;
					}
				}
				if(exactSimulationCollisionDomain)
				{
					// A shared mesh object is already the authoritative public
					// collision domain. Keeping this identity case direct makes
					// serial redetection and P5's range leaves consume the same
					// particles, topology and contact-state metadata. Distinct
					// cooked meshes remain on the proxy/embedding path below.
					Dy::avbdDetectAllOGCContacts(
						particles, numParticles, bodies, numBodies,
						rigidBoxes, numRigidBoxes,
						selfCollisionAdjacencies,
						numSelfCollisionAdjacencies,
						contacts, mContactParams, 0.0f,
						mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
						&mWorkspace.contact,
						mWorldPlanes.begin(), mWorldPlanes.size(), false,
						selfCollisionEnabled,
						rigidSpheres, numRigidSpheres,
						rigidCapsules, numRigidCapsules,
						rigidConvexes, numRigidConvexes,
						rigidTriangleSurfaces,
						numRigidTriangleSurfaces);
					removeRigidActorFilteredContacts(
						bodies, numBodies, softCores, contacts);
					removeDeformablePairFilteredContacts(
						bodies, numBodies, softCores, contacts);
					if(mContactSetTraceFile)
						writeContactSetTrace(contacts);
					return;
				}
				// Public CPU AVBD collision is defined on the cooked collision
				// domain.  Never reinterpret the simulation bodies as collision
				// geometry when the proxy scene is incomplete: doing so changes the
				// contact surface, feature identities and friction anchors mid-step.
				if(mCollisionBodies.size() != mBodies.size())
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD collision-domain proxy scene is incomplete before contact detection.");
					contacts.clear();
					return;
				}
				if(!refreshCollisionDetectionScene(particles, numParticles))
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD collision embedding became invalid before contact detection.");
					contacts.clear();
					return;
				}
				refreshSelfCollisionEnabled();
				Dy::avbdDetectAllOGCContacts(
					mCollisionParticles.begin(), mCollisionParticles.size(),
					mCollisionBodies.begin(), mCollisionBodies.size(),
					rigidBoxes, numRigidBoxes,
					mCollisionSelfCollisionAdjacencies.begin(),
					mCollisionSelfCollisionAdjacencies.size(),
					contacts, mContactParams, 0.0f,
					mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
					&mWorkspace.contact,
					mWorldPlanes.begin(), mWorldPlanes.size(), false,
					mSelfCollisionEnabled.begin(),
					rigidSpheres, numRigidSpheres,
					rigidCapsules, numRigidCapsules,
					rigidConvexes, numRigidConvexes,
					rigidTriangleSurfaces, numRigidTriangleSurfaces);
				removeRigidActorFilteredContacts(
					mCollisionBodies.begin(), mCollisionBodies.size(),
					softCores, contacts);
				removeDeformablePairFilteredContacts(
					mCollisionBodies.begin(), mCollisionBodies.size(),
					softCores, contacts);
				if(!expandCollisionDetectionContacts(
					contacts, numParticles, mCollisionVertexMappings,
					mCollisionBodies))
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD failed to expand a collision-domain contact into simulation particles.");
					contacts.clear();
					return;
				}
				if(mContactSetTraceFile)
					writeContactSetTrace(contacts);
				return;
			}
			const bool useSubsetCookedCollisionDomain =
				softCores && particles && bodies && numBodies > 0;
			if(useSubsetCookedCollisionDomain)
			{
				if(!rebuildSubsetCollisionDetectionScene(
						particles, numParticles, bodies, numBodies, softCores))
				{
					// An island selected from public actors must retain the same
					// collision domain as the full Scene.  Falling through to the
					// simulation bodies would make OGC collide against the voxel FEM
					// boundary and can leave persistent friction state attached to a
					// geometrically unrelated feature.
					reportInvalidCollisionEmbedding(
						"CPU AVBD failed to build the collision-domain proxy for a public actor subset.");
					contacts.clear();
					return;
				}
				Dy::avbdDetectAllOGCContacts(
					mSubsetCollisionParticles.begin(),
					mSubsetCollisionParticles.size(),
					mSubsetCollisionBodies.begin(),
					mSubsetCollisionBodies.size(),
					rigidBoxes, numRigidBoxes,
					mSubsetCollisionSelfCollisionAdjacencies.begin(),
					mSubsetCollisionSelfCollisionAdjacencies.size(),
					contacts, mContactParams, 0.0f,
					mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
					&mWorkspace.contact,
					mWorldPlanes.begin(), mWorldPlanes.size(), false,
					selfCollisionEnabled,
					rigidSpheres, numRigidSpheres,
					rigidCapsules, numRigidCapsules,
					rigidConvexes, numRigidConvexes,
					rigidTriangleSurfaces, numRigidTriangleSurfaces);
				removeRigidActorFilteredContacts(
					mSubsetCollisionBodies.begin(),
					mSubsetCollisionBodies.size(), softCores, contacts);
				removeDeformablePairFilteredContacts(
					mSubsetCollisionBodies.begin(),
					mSubsetCollisionBodies.size(), softCores, contacts);
				if(!expandCollisionDetectionContacts(
					contacts, numParticles,
					mSubsetCollisionVertexMappings,
					mSubsetCollisionBodies))
				{
					contacts.clear();
					return;
				}
				if(mContactSetTraceFile)
					writeContactSetTrace(contacts);
				return;
			}
			// This direct-domain path is reserved for legacy low-level callers
			// that do not represent public Scene actors.  Public full-Scene and
			// subset requests have both returned above and therefore cannot
			// silently fall back from their collision proxy to simulation tets.
			Dy::avbdDetectAllOGCContacts(
				particles, numParticles,
				bodies, numBodies,
				rigidBoxes, numRigidBoxes,
				selfCollisionAdjacencies,
				numSelfCollisionAdjacencies,
				contacts, mContactParams, 0.0f,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
				&mWorkspace.contact,
				mWorldPlanes.begin(), mWorldPlanes.size(),
				false, selfCollisionEnabled,
				rigidSpheres, numRigidSpheres,
				rigidCapsules, numRigidCapsules,
				rigidConvexes, numRigidConvexes,
				rigidTriangleSurfaces,
				numRigidTriangleSurfaces);
			removeRigidActorFilteredContacts(
				bodies, numBodies, softCores, contacts);
			removeDeformablePairFilteredContacts(
				bodies, numBodies, softCores, contacts);
			if(mContactSetTraceFile)
				writeContactSetTrace(contacts);
		}

		void writeContactSetTrace(
			const PxArray<Dy::AvbdSoftContact>& contacts)
		{
			if(!mContactSetTraceFile)
				return;
			mContactSetTraceKeys.clear();
			mContactSetTraceKeys.reserve(contacts.size());
			for(PxU32 contactIndex = 0;
				contactIndex < contacts.size(); contactIndex++)
			{
				const Dy::AvbdSoftContactGeometry& geometry =
					contacts[contactIndex].geometry;
				ContactSetTraceKey key;
				key.sourceType = PxU32(geometry.source.type);
				key.targetBodyIndex = geometry.source.targetBodyIndex;
				key.primitiveKey = geometry.source.primitiveKey;
				key.featureKey = geometry.source.featureKey;
				key.particleIndex = geometry.particleIdx;
				key.targetKind = PxU32(geometry.targetKind);
				key.targetIndex = geometry.targetIndex;
				key.targetSourceElementIndex =
					geometry.targetSourceElementIndex;
				key.triangleCoreExitFace = PX_MAX_U32;
				if(geometry.hasRigidBoxTriangleCoreExit)
				{
					const PxVec3& normal =
						geometry.rigidBoxTriangleCoreExitNormalLocal;
					const PxReal ax = PxAbs(normal.x);
					const PxReal ay = PxAbs(normal.y);
					const PxReal az = PxAbs(normal.z);
					if(normal.isFinite())
					{
						if(ax >= ay && ax >= az)
							key.triangleCoreExitFace =
								normal.x >= 0.0f ? 0u : 1u;
						else if(ay >= az)
							key.triangleCoreExitFace =
								normal.y >= 0.0f ? 2u : 3u;
						else
							key.triangleCoreExitFace =
								normal.z >= 0.0f ? 4u : 5u;
					}
				}
				for(PxU32 i = 0; i < 3; i++)
				{
					key.queryParticleIndices[i] =
						geometry.queryParticleIndices[i];
					key.surfaceParticleIndices[i] =
						geometry.surfaceParticleIndices[i];
				}
				mContactSetTraceKeys.pushBack(key);
			}
			PxSort(mContactSetTraceKeys.begin(),
				mContactSetTraceKeys.size());
			std::fprintf(mContactSetTraceFile, "detection=%u contacts=%u\n",
				mContactSetTraceDetectionIndex++, contacts.size());
			for(PxU32 keyIndex = 0;
				keyIndex < mContactSetTraceKeys.size(); keyIndex++)
			{
				const ContactSetTraceKey& key =
					mContactSetTraceKeys[keyIndex];
				std::fprintf(mContactSetTraceFile,
					"%u %u %016llx %016llx %u %u %u %u %u %u %u %u %u %u %u\n",
					key.sourceType, key.targetBodyIndex,
					static_cast<unsigned long long>(key.primitiveKey),
					static_cast<unsigned long long>(key.featureKey),
					key.particleIndex, key.targetKind, key.targetIndex,
					key.targetSourceElementIndex,
					key.queryParticleIndices[0], key.queryParticleIndices[1],
					key.queryParticleIndices[2],
					key.surfaceParticleIndices[0],
					key.surfaceParticleIndices[1],
					key.surfaceParticleIndices[2],
					key.triangleCoreExitFace);
			}
		}

		static void redetectContacts(
			Dy::AvbdSoftParticle* particles,
			PxU32 numParticles,
			Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			PxArray<Dy::AvbdSoftContact>& contacts,
			void* userData)
		{
			static_cast<AvbdCpuSoftScene*>(userData)->
				detectContacts(
					particles, numParticles,
					bodies, numBodies, contacts);
		}

		void refreshSurfaceFlattening(Entry& entry)
		{
			if(entry.kind != eSURFACE || !entry.surfaceCore ||
				entry.bodyIndex >= mBodies.size())
				return;
			const bool flatteningEnabled =
				(entry.surfaceCore->getSurfaceFlags() &
					PxDeformableSurfaceFlag::eENABLE_FLATTENING)
				? true : false;
			mBodies[entry.bodyIndex].compiled.
				compileBendingRestAngles(flatteningEnabled);
		}

		void applyDeformablePreintegrationControls(Entry& entry)
		{
			const PxReal maxLinearVelocity =
				PxMax(entry.getBodyCore().maxLinearVelocity, 0.0f);
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			for(PxU32 i = 0; i < particleCount; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				if(particle.invMass <= 0.0f ||
					!particle.velocity.isFinite())
					continue;
				const PxReal maxComponent = PxMax(
					PxAbs(particle.velocity.x),
					PxMax(
						PxAbs(particle.velocity.y),
						PxAbs(particle.velocity.z)));
				if(maxComponent == 0.0f)
					continue;
				const PxVec3 scaledVelocity =
					particle.velocity / maxComponent;
				const PxReal scaledMagnitude =
					scaledVelocity.magnitude();
				const PxReal limitedMaxComponent =
					maxLinearVelocity / scaledMagnitude;
				if(maxComponent <= limitedMaxComponent)
					continue;
				particle.velocity =
					scaledVelocity * limitedMaxComponent;
				particle.prevVelocity = particle.velocity;
			}
		}

		void syncHostInputs(
			Entry& entry,
			const PxsDeformableVolumeMaterialManager& materialManager)
		{
			Dy::DeformableBodyCore& bodyCore =
				entry.getBodyCore();
			bool syncPositions = false;
			bool syncVelocities = false;
			bool syncRestPositions = false;
			if(entry.kind == eVOLUME)
			{
				const Dy::DeformableVolumeCore& core =
					entry.volumeCore->getCore();
				syncPositions = core.dirtyFlags &
					PxDeformableVolumeDataFlag::
						eSIM_POSITION_INVMASS;
				syncVelocities = core.dirtyFlags &
					PxDeformableVolumeDataFlag::eSIM_VELOCITY;
			}
			else
			{
				const Dy::DeformableSurfaceCore& core =
					entry.surfaceCore->getCore();
				syncPositions = core.dirtyFlags &
					PxDeformableSurfaceDataFlag::ePOSITION_INVMASS;
				syncVelocities = core.dirtyFlags &
					PxDeformableSurfaceDataFlag::eVELOCITY;
				syncRestPositions = core.dirtyFlags &
					PxDeformableSurfaceDataFlag::eREST_POSITION;
				if(syncRestPositions)
				{
					const bool rebuilt =
						rebuildSurfaceRestState(entry);
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
				}
			}

			Dy::AvbdSoftBody& body = mBodies[entry.bodyIndex];
			body.compiled.selfCollisionFilterDistance =
				PxMax(bodyCore.selfCollisionFilterDistance, 0.0f);
			// Rest position changes rebuild the compiled body above. A filter
			// distance change keeps topology intact, but must refresh the bounded
			// immutable rest-space exclusion cache before the next OGC query.
			body.compiled.ensureSelfCollisionRestVertexTriangleFilter();
			body.compiled.maxDepenetrationVelocity =
				PxMax(-bodyCore.maxPenetrationBias, 0.0f);
			body.compiled.selfCollisionStressTolerance =
				bodyCore.selfCollisionStressTolerance;
			body.compiled.speculativeCCDEnabled =
				bodyCore.bodyFlags.isSet(
					PxDeformableBodyFlag::
						eENABLE_SPECULATIVE_CCD);
			refreshSurfaceFlattening(entry);
			if(entry.kind == eVOLUME)
			{
				const PxsDeformableVolumeMaterialCore* material =
					getMaterial(*entry.volumeCore, materialManager);
				body.material.youngsModulus =
					material ? material->youngs : 1.0e5f;
				body.material.poissonsRatio =
					material ? material->poissons : 0.3f;
				body.material.damping =
					bodyCore.linearDamping +
					(material ? material->elasticityDamping : 0.0f);
				body.material.bendingStiffness = 0.0f;
				body.material.bendingDamping = 0.0f;
				body.material.thickness = 0.01f;
				body.material.dynamicFriction = material
					? PxMax(material->dynamicFriction, 0.0f)
					: 0.5f;
				const bool coRotationalVolumeModel =
					!material ||
					material->materialModel ==
						PxDeformableVolumeMaterialModel::
							eCO_ROTATIONAL;
				if(body.material.coRotationalVolumeModel !=
					coRotationalVolumeModel)
				{
					body.material.coRotationalVolumeModel =
						coRotationalVolumeModel;
					if(coRotationalVolumeModel)
						body.compiled.buildTetIncidencePacketProgram();
					else
						body.compiled.invalidateTetIncidencePacketProgram();
				}
			}
			else
			{
				const PxsDeformableSurfaceMaterialCore* material =
					getSurfaceMaterial(*entry.surfaceCore);
				body.material.youngsModulus =
					material ? material->youngs : 1.0e5f;
				body.material.poissonsRatio =
					material ? material->poissons : 0.3f;
				body.material.damping =
					bodyCore.linearDamping +
					(material ? material->elasticityDamping : 0.0f);
				body.material.bendingStiffness =
					material ? material->bendingStiffness : 0.0f;
				body.material.bendingDamping =
					material ? material->bendingDamping : 0.0f;
				body.material.thickness = material
					? PxMax(material->thickness, 1.0e-4f)
					: 0.01f;
				body.material.dynamicFriction = material
					? PxMax(material->dynamicFriction, 0.0f)
					: 0.5f;
				body.material.coRotationalVolumeModel = true;
			}
			body.material.computeLameParameters();
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			const PxReal gravityScale =
				(entry.getActorFlags() &
					PxActorFlag::eDISABLE_GRAVITY)
				? 0.0f : 1.0f;
			for(PxU32 i = 0; i < particleCount; i++)
				mParticles[particleStart + i].gravityScale =
					gravityScale;

			PxVec4* positions = entry.getPositionInvMass();
			PxVec4* velocities = entry.getVelocity();
			PX_ASSERT(positions && velocities);
			if(syncPositions || syncVelocities || bodyCore.dirty)
			{
				for(PxU32 i = 0; i < particleCount; i++)
				{
					Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					if(syncPositions)
					{
						const PxVec4& positionInvMass =
							positions[i];
						particle.position =
							positionInvMass.getXYZ();
						if(entry.kind == eVOLUME)
							particle.initialPosition =
								particle.position;
						particle.predictedPosition = particle.position;
						particle.outerPosition = particle.position;
						particle.invMass =
							PxMax(positionInvMass.w, 0.0f);
						particle.mass = particle.invMass > 0.0f
							? 1.0f / particle.invMass : 0.0f;
						particle.elasticK = 0.0f;
					}
					if(syncVelocities)
					{
						particle.velocity =
							velocities[i].getXYZ();
						particle.prevVelocity = particle.velocity;
					}
					particle.damping = body.material.damping;
				}
			}
			applyDeformablePreintegrationControls(entry);
			bodyCore.dirty = false;
			if(entry.kind == eVOLUME)
				entry.volumeCore->getCore().dirtyFlags =
					PxDeformableVolumeDataFlags(0);
			else
				entry.surfaceCore->getCore().dirtyFlags =
					PxDeformableSurfaceDataFlags(0);
		}

		void writeBack(Entry& entry)
		{
			if(entry.sleeping)
				return;
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			if(entry.kind == eSURFACE)
			{
				Dy::DeformableSurfaceCore& core =
					entry.surfaceCore->getCore();
				for(PxU32 i = 0; i < particleCount; i++)
				{
					const Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					core.positionInvMass[i] =
						PxVec4(particle.position, particle.invMass);
					const PxReal velocityW = core.velocity[i].w;
					core.velocity[i] =
						PxVec4(particle.velocity, velocityW);
				}
				return;
			}

			Dy::DeformableVolumeCore& core =
				entry.volumeCore->getCore();
			for(PxU32 i = 0; i < particleCount; i++)
			{
				const Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				core.simPositionInvMass[i] =
					PxVec4(particle.position, particle.invMass);
				core.simVelocity[i] =
					PxVec4(particle.velocity, particle.invMass);
			}

			const PxU32 collisionVertexCount =
				entry.collisionMesh->getNbVertices();
			if(entry.collisionMesh == entry.simulationMesh &&
				collisionVertexCount == particleCount)
			{
				for(PxU32 i = 0; i < collisionVertexCount; i++)
				{
					const PxReal invMass =
						core.positionInvMass[i].w;
					core.positionInvMass[i] = PxVec4(
						mParticles[particleStart + i].position,
						invMass);
				}
				return;
			}

			Gu::DeformableVolumeAuxData& auxData =
				static_cast<Gu::DeformableVolumeAuxData&>(
					*entry.auxData);
			const PxU32* remap =
				auxData.mVertsRemapInGridModel;
			const PxReal* barycentrics =
				auxData.mVertsBarycentricInGridModel;
			if(!remap || !barycentrics)
			{
				// Distinct meshes are rejected at registration when cooked mapping
				// is absent. Never reinterpret the first N simulation vertices as
				// collision vertices here.
				PX_ASSERT(false);
				return;
			}

			const bool has16BitIndices =
				entry.simulationMesh->getTetrahedronMeshFlags() &
				PxTetrahedronMeshFlag::e16_BIT_INDICES;
			const PxU16* tets16 = has16BitIndices
				? static_cast<const PxU16*>(
					entry.simulationMesh->getTetrahedrons()) : NULL;
			const PxU32* tets32 = has16BitIndices ? NULL
				: static_cast<const PxU32*>(
					entry.simulationMesh->getTetrahedrons());
			for(PxU32 i = 0; i < collisionVertexCount; i++)
			{
				const PxU32 tetIndex = remap[i];
				if(tetIndex >= entry.simulationMesh->getNbTetrahedrons())
				{
					PX_ASSERT(false);
					return;
				}
				PxVec3 position(0.0f);
				for(PxU32 j = 0; j < 4; j++)
				{
					const PxU32 localParticle = has16BitIndices
						? tets16[4 * tetIndex + j]
						: tets32[4 * tetIndex + j];
					position +=
						mParticles[
							particleStart + localParticle].position *
						barycentrics[4 * i + j];
				}
				const PxReal invMass = core.positionInvMass[i].w;
				core.positionInvMass[i] =
					PxVec4(position, invMass);
			}
		}

		PxArray<Entry>					mEntries;
		PxArray<PredictionTask*>			mPredictionTasks;
		PxArray<WriteBackTask*>			mWriteBackTasks;
		// P4.5.3 keeps a bounded Scene-owned task pool. It is grown only by a
		// parent before a layer is submitted and recycled only after dispatcher
		// release; children never allocate, resize or inspect Scene state.
		PxMutex							mCausalLayerTaskPoolMutex;
		PxArray<CausalLayerTask*>			mCausalLayerTasks;
		PxArray<CausalLayerFinishTask*>	mCausalLayerFinishTasks;
		PxArray<PxU32>					mFreeCausalLayerTaskIndices;
		PxArray<PxU32>					mFreeCausalLayerFinishTaskIndices;
		PxArray<Dy::AvbdParticlePrimalRangeObservation>
									mCausalLayerRangeObservations;
		// P5.3b uses a separate, bounded pool because collision leaves own
		// private contact streams rather than primal range observations.
		PxMutex							mWorldPlaneContactTaskPoolMutex;
		PxArray<WorldPlaneContactTask*>	mWorldPlaneContactTasks;
		PxArray<WorldPlaneContactFinishTask*>
									mWorldPlaneContactFinishTasks;
		PxArray<PxU32>					mFreeWorldPlaneContactTaskIndices;
		PxArray<PxU32>					mFreeWorldPlaneContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mWorldPlaneContactTaskOutputs;
		// P5.4b owns a distinct bounded pool so a rigid-box SDF epoch cannot
		// borrow/relabel world-plane task telemetry or private output storage.
		PxMutex							mRigidBoxSdfContactTaskPoolMutex;
		PxArray<RigidBoxSdfContactTask*>	mRigidBoxSdfContactTasks;
		PxArray<RigidBoxSdfContactFinishTask*>
									mRigidBoxSdfContactFinishTasks;
		PxArray<PxU32>					mFreeRigidBoxSdfContactTaskIndices;
		PxArray<PxU32>					mFreeRigidBoxSdfContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidBoxSdfContactTaskOutputs;
		// P5.12b keeps the swept family separate until parent fan-in so the
		// canonical current-then-swept contact order is mechanically visible.
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidBoxSweptSdfContactTaskOutputs;
		// P5.5b does not borrow the box pool: its eligibility and the parent
		// sphere swept/feature suffix are independently observable.
		PxMutex							mRigidSphereSdfContactTaskPoolMutex;
		PxArray<RigidSphereSdfContactTask*>	mRigidSphereSdfContactTasks;
		PxArray<RigidSphereSdfContactFinishTask*>
									mRigidSphereSdfContactFinishTasks;
		PxArray<PxU32>					mFreeRigidSphereSdfContactTaskIndices;
		PxArray<PxU32>					mFreeRigidSphereSdfContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidSphereSdfContactTaskOutputs;
		// P5.13b retains swept ranges independently until the parent completes
		// the canonical all-current then all-swept family merge.
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidSphereSweptSdfContactTaskOutputs;
		// P5.14b retains swept capsule ranges independently until the parent
		// completes the canonical all-current then all-swept family merge.
		// The only shared object with spheres is the continuation slot, which is
		// mutually exclusive by primitive eligibility.
		PxMutex							mRigidCapsuleSdfContactTaskPoolMutex;
		PxArray<RigidCapsuleSdfContactTask*>	mRigidCapsuleSdfContactTasks;
		PxArray<RigidCapsuleSdfContactFinishTask*>
									mRigidCapsuleSdfContactFinishTasks;
		PxArray<PxU32>					mFreeRigidCapsuleSdfContactTaskIndices;
		PxArray<PxU32>					mFreeRigidCapsuleSdfContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidCapsuleSdfContactTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidCapsuleSweptSdfContactTaskOutputs;
		PxMutex							mRigidConvexSdfContactTaskPoolMutex;
		PxArray<RigidConvexSdfContactTask*>	mRigidConvexSdfContactTasks;
		PxArray<RigidConvexSdfContactFinishTask*>
									mRigidConvexSdfContactFinishTasks;
		PxArray<PxU32>					mFreeRigidConvexSdfContactTaskIndices;
		PxArray<PxU32>					mFreeRigidConvexSdfContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidConvexSdfContactTaskOutputs;
		// P5.15b retains swept convex ranges independently until the parent
		// completes the canonical all-current then all-swept family merge.
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidConvexSweptSdfContactTaskOutputs;
		// P5.16b retains current and swept triangle-SDF range outputs until the
		// parent completes the canonical all-current then all-swept merge.
		PxMutex							mRigidTriangleSurfaceContactTaskPoolMutex;
		PxArray<RigidTriangleSurfaceContactTask*>
									mRigidTriangleSurfaceContactTasks;
		PxArray<RigidTriangleSurfaceContactFinishTask*>
									mRigidTriangleSurfaceContactFinishTasks;
		PxArray<PxU32>					mFreeRigidTriangleSurfaceContactTaskIndices;
		PxArray<PxU32>					mFreeRigidTriangleSurfaceContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidTriangleSurfaceContactTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidTriangleSurfaceSweptSdfContactTaskOutputs;
		// P5.17d feature rows are partitioned by canonical plan index and
		// stable-merged only by the parent after both SDF output families.
		Dy::AvbdRigidTriangleSurfaceFeaturePlan
									mRigidTriangleSurfaceFeaturePlan;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidTriangleSurfaceFeatureContactTaskOutputs;
		// P5.27 default-off candidate: one complete private output per canonical
		// feature-plan row, independently of child scheduling order.
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidTriangleSurfaceFeatureContactPlanOutputs;
		bool						mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan;
		bool						mRigidTriangleSurfaceFeatureRoundRobinTaskPlan;
		PxArray<Dy::AvbdSoftCollisionStats>
									mRigidTriangleSurfaceContactTaskStats;
		// P5.9d's soft-pair leaves never borrow the serial soft-pair query
		// scratch. Their pooled tasks carry that scratch privately.
		PxMutex							mSoftPairContactTaskPoolMutex;
		PxArray<SoftPairContactTask*>	mSoftPairContactTasks;
		PxArray<SoftPairContactFinishTask*>
									mSoftPairContactFinishTasks;
		PxArray<PxU32>					mFreeSoftPairContactTaskIndices;
		PxArray<PxU32>					mFreeSoftPairContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mSoftPairContactTaskOutputs;
		PxArray<Dy::AvbdSoftCollisionStats>
									mSoftPairContactTaskStats;
		// P5.10b keeps self-BVH leaves separate from pair leaves. Task output
		// order encodes the required VF phase followed by the EE phase.
		PxMutex							mSelfBvhContactTaskPoolMutex;
		PxArray<SelfBvhContactTask*>	mSelfBvhContactTasks;
		PxArray<SelfBvhContactFinishTask*>
									mSelfBvhContactFinishTasks;
		PxArray<PxU32>					mFreeSelfBvhContactTaskIndices;
		PxArray<PxU32>					mFreeSelfBvhContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mSelfBvhContactTaskOutputs;
		PxArray<Dy::AvbdSoftCollisionStats>
									mSelfBvhContactTaskStats;
		Dy::AvbdSoftContactWorkspace		mSelfBvhSerialRangeWorkspace;
		// P5 static-world+self aggregate: each child owns a disjoint range, but
		// its source streams remain physically separate until the parent merges
		// world, box-current, box-swept, box-features, self-VF, then self-EE.
		PxMutex							mStaticWorldSelfOgcContactFinishTaskPoolMutex;
		PxArray<StaticWorldSelfOgcContactTask*>
									mStaticWorldSelfOgcContactTasks;
		PxArray<StaticWorldSelfOgcContactFinishTask*>
									mStaticWorldSelfOgcContactFinishTasks;
		PxArray<PxU32>					mFreeStaticWorldSelfOgcContactTaskIndices;
		PxArray<PxU32>					mFreeStaticWorldSelfOgcContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcWorldTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcBoxTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcBoxSweptTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcSelfVertexTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcSelfEdgeTaskOutputs;
		PxArray<Dy::AvbdSoftCollisionStats>
									mStaticWorldSelfOgcTaskStats;
		PxArray<StaticShapeEntry>		mStaticShapes;
		PxArray<DynamicShapeEntry>		mDynamicShapes;
		PxArray<WorldPinEntry>			mWorldPins;
		PxArray<RigidAttachmentEntry>	mRigidAttachments;
		PxArray<ArticulationAttachmentEntry>
										mArticulationAttachments;
		PxArray<SoftPairAttachmentEntry>
										mSoftPairAttachments;
		PxArray<PrescribedAttachmentEntry>
										mPrescribedAttachments;
		PxArray<RigidActorFilterEntry>	mRigidActorFilters;
		PxArray<DeformablePairFilterEntry>
										mDeformablePairFilters;
		PxArray<NativeIslandEdgeEntry>	mNativeIslandEdges;
		PxArray<NativeSoftSoftIslandEdgeEntry>
										mNativeSoftSoftIslandEdges;
		PxArray<IslandSelectionStorage*>	mIslandSelectionStorages;
		PxArray<Dy::AvbdSoftParticle>	mParticles;
		PxArray<Dy::AvbdSoftBody>		mBodies;
		// Public CPU AVBD detects against the cooked collision-domain mesh.
		// These proxy particles are geometry only; prepared contacts are expanded
		// through mCollisionVertexMappings before either solver sees them.
		PxArray<Dy::AvbdSoftParticle>	mCollisionParticles;
		PxArray<Dy::AvbdSoftBody>		mCollisionBodies;
		PxArray<Dy::AvbdWeightedContactPoint>
										mCollisionVertexMappings;
		PxArray<Dy::AvbdSelfCollisionAdjacency>
										mCollisionSelfCollisionAdjacencies;
		PxArray<Dy::AvbdSoftParticle>	mSubsetCollisionParticles;
		PxArray<Dy::AvbdSoftBody>		mSubsetCollisionBodies;
		PxArray<Dy::AvbdWeightedContactPoint>
										mSubsetCollisionVertexMappings;
		PxArray<Dy::AvbdSelfCollisionAdjacency>
										mSubsetCollisionSelfCollisionAdjacencies;
		PxArray<Dy::AvbdSelfCollisionAdjacency>
										mSelfCollisionAdjacencies;
		PxArray<PxU8>					mSelfCollisionEnabled;
		PxArray<Dy::AvbdWorldPlane>		mWorldPlanes;
		PxArray<Dy::AvbdRigidBox>		mRigidBoxes;
		PxArray<Dy::AvbdRigidSphere>		mRigidSpheres;
		PxArray<Dy::AvbdRigidCapsule>	mRigidCapsules;
		PxArray<Dy::AvbdRigidConvex>		mRigidConvexes;
		PxArray<Dy::AvbdRigidTriangleSurface>
										mRigidTriangleSurfaces;
		PxArray<Dy::AvbdSoftContact>		mContacts;
		Dy::AvbdOGCParams				mContactParams;
		Dy::AvbdSoftBodyWorkspace		mWorkspace;
		Dy::AvbdSoftBodyStepState		mStandaloneComponentStepState;
		Dy::AvbdSoftBodyStepStats		mLastStepStats;
		Dy::AvbdSoftBodyStepStats		mStandaloneStepStats;
		Dy::AvbdSoftCollisionStats		mLastCollisionStats;
		ComponentFallbackPlan			mComponentFallbackPlan;
		PxArray<ContactSetTraceKey>		mContactSetTraceKeys;
		std::FILE*						mContactSetTraceFile;
		PxU32							mContactSetTraceDetectionIndex;
		PxU64							mContextId;
		const PxsDeformableVolumeMaterialManager&
										mDeformableMaterialManager;
		const PxsDeformableSurfaceMaterialManager&
										mSurfaceMaterialManager;
		const PxsMaterialManager&		mRigidMaterialManager;
		IG::SimpleIslandManager&		mIslandManager;
		PxU64							mNextPrimitiveKey;
		PxU32							mRigidTriangleSurfaceCompileStamp;
		PxU32							mNextWorldPinHandle;
		PxU32							mNextRigidAttachmentHandle;
		PxU32							mNextArticulationAttachmentHandle;
		PxU32							mNextSoftPairAttachmentHandle;
		PxU32							mNextPrescribedAttachmentHandle;
		PxU32							mNextRigidActorFilterHandle;
		PxU32							mNextDeformablePairFilterHandle;
		bool							mDynamicsOwnsStep;
		PxU32							mDynamicsSelectedEntryCount;
		PxU32							mLastComponentFallbackSteps;
		PxU32							mLastNativeIslandSteps;
		bool							mComponentFallbackPlanPrepared;
		bool							mStandaloneComponentSolvePrepared;
		bool							mStandaloneComponentPostSolvePending;
		PxU32						mStandaloneTaskGraphDispatcherWorkers;
		bool						mStandaloneTaskGraphEnhancedDeterminism;
		Dy::AvbdParticlePrimalSchedule	mStandaloneParticlePrimalSchedule;
		StandaloneTaskGraphTelemetry	mStandaloneTaskGraphTelemetry;
		bool							mP3ForceSplitPrediction;
		bool							mCollisionStatsEnabled;
		bool							mWorldPlaneContactTransactionPending;
		bool							mRigidBoxSdfContactTransactionPending;
		bool							mRigidSphereSdfContactTransactionPending;
		bool							mRigidCapsuleSdfContactTransactionPending;
		bool							mRigidConvexSdfContactTransactionPending;
			bool							mRigidTriangleSurfaceContactTransactionPending;
			PxU64						mRigidTriangleSurfaceContactTaskSubmitStartNanos;
			PxU64						mRigidTriangleSurfaceContactTaskSubmitEndNanos;
		bool							mSoftPairContactTransactionPending;
		bool							mSoftPairContactUseSurfaceTriangleBvh;
		bool							mSelfBvhContactTransactionPending;
		PxU32						mSelfBvhContactBodyIndex;
		bool							mStaticWorldSelfOgcContactTransactionPending;
		bool							mWorkspacePreflightPending;
	};

	void AvbdCpuSoftScene::WriteBackTask::runInternal()
	{
		PX_ASSERT(mTaskGraphContext);
		mTaskGraphContext->beginWriteBackTask();
		mScene.writeBackStandaloneComponentRange(mEntryBegin, mEntryEnd);
		mTaskGraphContext->endWriteBackTask();
	}

	void AvbdCpuSoftScene::PredictionTask::runInternal()
	{
		PX_ASSERT(mTaskGraphContext);
		mTaskGraphContext->beginPredictionTask();
		mScene.predictStandaloneComponentRange(
			mEntryBegin, mEntryEnd, mDt, mGravity);
		mTaskGraphContext->endPredictionTask();
	}

		void AvbdCpuSoftScene::CausalLayerTask::runInternal()
		{
		PX_ASSERT(mTaskGraphContext && mSolveContext && mBodies);
		PX_ASSERT(mObservation);
		PX_ASSERT(mIndependentBodyRange ?
			(mBodyBegin < mBodyEnd && mBodyEnd <= mBodyCount) :
			(mParticleBodyIndices && mPackedParticleIndices &&
				mPackedBegin < mPackedEnd));
			mTaskGraphContext->beginCausalLayerTask();
			mScene.mStandaloneTaskGraphTelemetry.beginCausalLayerTask();
		if(mIndependentBodyRange)
			Dy::avbdSolveParticlePrimalIndependentBodyRange(
				*mSolveContext, mBodies, mBodyCount,
				mBodyBegin, mBodyEnd, *mObservation);
		else
			Dy::avbdSolveParticlePrimalPackedRange(
				*mSolveContext, mBodies, mBodyCount, mParticleBodyIndices,
				mNumParticles, mPackedParticleIndices,
				mPackedBegin, mPackedEnd, *mObservation);
			mScene.mStandaloneTaskGraphTelemetry.endCausalLayerTask();
			mTaskGraphContext->endCausalLayerTask();
		}

static const char* sFilterShaderDataMemAllocId = "SceneDesc filterShaderData";

}}

void PxcDisplayContactCacheStats();

static const bool gUseNewTaskAllocationScheme = false;

namespace
{
	class ScAfterIntegrationTask : public Cm::Task
	{
	public:
		static const PxU32 MaxTasks = 256;
	private:
		const PxNodeIndex* const	mIndices;
		const PxU32					mNumBodies;
		PxsContext*					mContext;
		Context*					mDynamicsContext;
		const UpdateCachedParams	mParams;
		Sc::Scene&					mScene;
	
	public:

		ScAfterIntegrationTask(const PxNodeIndex* const indices, PxU32 numBodies, PxsContext* context, Context* dynamicsContext, PxsTransformCache& cache, Sc::Scene& scene) :
			Cm::Task		(scene.getContextId()),
			mIndices		(indices),
			mNumBodies		(numBodies),
			mContext		(context),
			mDynamicsContext(dynamicsContext),
			mParams			(cache, scene.getBoundsArray()),
			mScene			(scene)
		{
		}

		// PT: warning, this runs in parallel with updateArticulationAfterIntegration and updateKinematicCached, and all of these touching the getChangedAABBMgActorHandleMap() bitmap
		virtual void runInternal() PX_OVERRIDE
		{
			const PxU32 rigidBodyOffset = Sc::BodySim::getRigidBodyOffset();

			Sc::BodySim* ccdBodies[MaxTasks];
			Sc::BodySim* activateBodies[MaxTasks];
			Sc::BodySim* deactivateBodies[MaxTasks];
			PxU32 nbBpUpdates = 0, nbCcdBodies = 0;

			IG::SimpleIslandManager& manager = *mScene.getSimpleIslandManager();
			const IG::IslandSim& islandSim = manager.getAccurateIslandSim();

			Sc::BodySim* frozen[MaxTasks], * unfrozen[MaxTasks];
			PxU32 nbFrozen = 0, nbUnfrozen = 0;
			PxU32 nbActivated = 0, nbDeactivated = 0;

			PinnableBitMap& changedAABBMgrHandles = mScene.getAABBManager()->getChangedAABBMgActorHandleMap();

			for(PxU32 i = 0; i < mNumBodies; i++)
			{
				PxsRigidBody* rigid = getRigidBodyFromIG(islandSim, mIndices[i]);
				Sc::BodySim* bodySim = reinterpret_cast<Sc::BodySim*>(reinterpret_cast<PxU8*>(rigid) - rigidBodyOffset);
				
				PxsBodyCore& bodyCore = bodySim->getBodyCore().getCore();
				//If we got in this code, then this is an active object this frame. The solver computed the new wakeCounter and we 
				//commit it at this stage. We need to do it this way to avoid a race condition between the solver and the island gen, where
				//the island gen may have deactivated a body while the solver decided to change its wake counter.
				bodyCore.wakeCounter = bodyCore.solverWakeCounter;
				PxsRigidBody& llBody = bodySim->getLowLevelBody();

				const PxIntBool isFrozen = bodySim->isFrozen();
				if(!isFrozen)
				{
					nbBpUpdates++;

					// PT: TODO: this one does not reach the GPU code. This is only an issue when Direct GPU is enabled.
					bodySim->updateCached(mParams, &changedAABBMgrHandles, true, true);
				}

				if(llBody.isFreezeThisFrame() && isFrozen)
					frozen[nbFrozen++] = bodySim;	// PT: we cannot call freezeTransforms directly from here, as the "destroySqBounds" call inside it is not thread-safe yet.
				else if(llBody.isUnfreezeThisFrame())
					unfrozen[nbUnfrozen++] = bodySim;

				if(bodyCore.mFlags & PxRigidBodyFlag::eENABLE_CCD)
					ccdBodies[nbCcdBodies++] = bodySim;

				if(llBody.isActivateThisFrame())
				{
					PX_ASSERT(!llBody.isDeactivateThisFrame());
					activateBodies[nbActivated++] = bodySim;
				}
				else if(llBody.isDeactivateThisFrame())
				{
					deactivateBodies[nbDeactivated++] = bodySim;
				}
				llBody.clearAllFrameFlags();
			}
			if(nbBpUpdates)
			{
				mParams.mTransformCache.setChangedState();
				mParams.mBoundsArray.setChangedState();
			}

			if(nbUnfrozen >0 || nbFrozen > 0 || nbCcdBodies>0 || nbActivated>0 || nbDeactivated>0)
			{
				mContext->getLock().lock();
			
				PxArray<Sc::BodySim*>& sceneCcdBodies = mScene.getCcdBodies();
				for (PxU32 i = 0; i < nbCcdBodies; i++)
					sceneCcdBodies.pushBack(ccdBodies[i]);

				for(PxU32 i=0;i<nbFrozen;i++)
				{
					PX_ASSERT(frozen[i]->isFrozen());
					//frozen[i]->freezeTransforms(mParams, &changedAABBMgrHandles);
					// PT: this new version only updates the transform flags and does not touch changedAABBMgrHandles anymore.
					// We still need to run it inside the context lock, as the function still calls "destroySqBounds", which is not thread-safe.
					frozen[i]->freezeTransforms(mParams.mTransformCache);
				}

				for(PxU32 i=0;i<nbUnfrozen;i++)
				{
					PX_ASSERT(!unfrozen[i]->isFrozen());
					unfrozen[i]->createSqBounds();
				}
			
				for(PxU32 i = 0; i < nbActivated; ++i)
					activateBodies[i]->notifyNotReadyForSleeping();

				for(PxU32 i = 0; i < nbDeactivated; ++i)
					deactivateBodies[i]->notifyReadyForSleeping();

				mContext->getLock().unlock();
			}
		}

		virtual const char* getName() const PX_OVERRIDE
		{
			return "ScScene.afterIntegrationTask";
		}

	private:
		PX_NOCOPY(ScAfterIntegrationTask)
	};

	class ScSimulationControllerCallback : public PxsSimulationControllerCallback
	{
		Sc::Scene* mScene; 
	public:

		ScSimulationControllerCallback(Sc::Scene* scene) : mScene(scene)
		{
		}
	
		virtual void updateScBodyAndShapeSim(PxBaseTask* continuation)	PX_OVERRIDE
		{
			PxsContext* contextLL = mScene->getLowLevelContext();
			IG::SimpleIslandManager* islandManager = mScene->getSimpleIslandManager();
			Dy::Context* dynamicContext = mScene->getDynamicsContext();

			Cm::FlushPool& flushPool = contextLL->getTaskPool();

			const PxU32 MaxBodiesPerTask = ScAfterIntegrationTask::MaxTasks;

			PxsTransformCache& cache = contextLL->getTransformCache();

			const IG::IslandSim& islandSim = islandManager->getAccurateIslandSim();

			/*const*/ PxU32 numBodies = islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);

			const PxNodeIndex*const nodeIndices = islandSim.getActiveNodes(IG::Node::eRIGID_BODY_TYPE);

			const PxU32 rigidBodyOffset = Sc::BodySim::getRigidBodyOffset();

			// PT: TASK-CREATION TAG
			if(!gUseNewTaskAllocationScheme)
			{
				PxU32 nbShapes = 0;
				PxU32 startIdx = 0;
				for (PxU32 i = 0; i < numBodies; i++)
				{
					if (nbShapes >= MaxBodiesPerTask)
					{
						ScAfterIntegrationTask* task = PX_PLACEMENT_NEW(flushPool.allocate(sizeof(ScAfterIntegrationTask)), ScAfterIntegrationTask(nodeIndices + startIdx, i - startIdx,
							contextLL, dynamicContext, cache, *mScene));

						startTask(task, continuation);

						startIdx = i;
						nbShapes = 0;
					}
					PxsRigidBody* rigid = getRigidBodyFromIG(islandSim, nodeIndices[i]);
					Sc::BodySim* bodySim = reinterpret_cast<Sc::BodySim*>(reinterpret_cast<PxU8*>(rigid) - rigidBodyOffset);
					nbShapes += PxMax(1u, bodySim->getNbShapes()); //Always add at least 1 shape in, even if the body has zero shapes because there is still some per-body overhead
				}

				if (nbShapes)
				{
					ScAfterIntegrationTask* task = PX_PLACEMENT_NEW(flushPool.allocate(sizeof(ScAfterIntegrationTask)), ScAfterIntegrationTask(nodeIndices + startIdx, numBodies - startIdx,
						contextLL, dynamicContext, cache, *mScene));

					startTask(task, continuation);
				}
			}
			else
			{
				// PT:
				const PxU32 numCpuTasks = continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount();

				PxU32 nbPerTask;
				if(numCpuTasks)
					nbPerTask = numBodies > numCpuTasks ? numBodies / numCpuTasks : numBodies;
				else
					nbPerTask = numBodies;

				// PT: we need to respect that limit even with a single thread, because of hardcoded buffer limits in ScAfterIntegrationTask.
				if(nbPerTask>MaxBodiesPerTask)
					nbPerTask = MaxBodiesPerTask;

				PxU32 start = 0;
				while(numBodies)
				{
					const PxU32 nb = numBodies < nbPerTask ? numBodies : nbPerTask;

					ScAfterIntegrationTask* task = PX_PLACEMENT_NEW(flushPool.allocate(sizeof(ScAfterIntegrationTask)), ScAfterIntegrationTask(nodeIndices+start, nb, 
						contextLL, dynamicContext, cache, *mScene));

					start += nb;
					numBodies -= nb;

					startTask(task, continuation);
				}
			}
		}

		virtual PxU32 getNbCcdBodies()	PX_OVERRIDE
		{ 
			return mScene->getCcdBodies().size(); 
		}
	};

	// PT: TODO: what is this Pxg class doing here?
	class PxgUpdateBodyAndShapeStatusTask : public Cm::Task
	{
	public:
		static const PxU32 MaxTasks = 2048;
	private:
		const PxNodeIndex* const mNodeIndices;
		const PxU32 mNumBodies;
		Sc::Scene& mScene;
		void**	mRigidBodyLL;
		PxU32*	mActivatedBodies;
		PxU32*	mDeactivatedBodies;
		PxI32&	mCCDBodyWriteIndex;
	
	public:

		PxgUpdateBodyAndShapeStatusTask(const PxNodeIndex* const indices, PxU32 numBodies, void** rigidBodyLL, PxU32* activatedBodies, PxU32* deactivatedBodies, Sc::Scene& scene, PxI32& ccdBodyWriteIndex) : 
			Cm::Task			(scene.getContextId()),
			mNodeIndices		(indices),
			mNumBodies			(numBodies),
			mScene				(scene),
			mRigidBodyLL		(rigidBodyLL),
			mActivatedBodies	(activatedBodies),
			mDeactivatedBodies	(deactivatedBodies),
			mCCDBodyWriteIndex	(ccdBodyWriteIndex)
		{
		}

		virtual void runInternal() PX_OVERRIDE
		{
			IG::SimpleIslandManager& islandManager = *mScene.getSimpleIslandManager();
			const IG::IslandSim& islandSim = islandManager.getAccurateIslandSim();

			PxU32 nbCcdBodies = 0;

			PxArray<Sc::BodySim*>& sceneCcdBodies = mScene.getCcdBodies();
			Sc::BodySim* ccdBodies[MaxTasks];

			const size_t bodyOffset =  PX_OFFSET_OF_RT(Sc::BodySim, getLowLevelBody());

			for(PxU32 i=0; i<mNumBodies; ++i)
			{
				const PxU32 nodeIndex = mNodeIndices[i].index();
				PxsRigidBody* rigidLL = reinterpret_cast<PxsRigidBody*>(mRigidBodyLL[nodeIndex]);

				PxsBodyCore* bodyCore = &rigidLL->getCore();
				bodyCore->wakeCounter = bodyCore->solverWakeCounter;
				//we can set the frozen/unfrozen flag in GPU, but we have copied the internalflags
				//from the solverbodysleepdata to pxsbodycore, so we just need to clear the frozen flag in here
				rigidLL->clearAllFrameFlags();

				PX_ASSERT(mActivatedBodies[nodeIndex] <= 1);
				PX_ASSERT(mDeactivatedBodies[nodeIndex] <= 1);
				if(mActivatedBodies[nodeIndex])
				{
					PX_ASSERT(bodyCore->wakeCounter > 0.0f);
					islandManager.activateNode(mNodeIndices[i]);
				}
				else if(mDeactivatedBodies[nodeIndex])
				{
					//KS - the CPU code can reset the wake counter due to lost touches in parallel with the solver, so we need to verify
					//that the wakeCounter is still 0 before deactivating the node
					if (bodyCore->wakeCounter == 0.0f)
					{
						islandManager.deactivateNode(mNodeIndices[i]);
					}
				}

				if (bodyCore->mFlags & PxRigidBodyFlag::eENABLE_CCD)
				{
					PxsRigidBody* rigidBody = getRigidBodyFromIG(islandSim, mNodeIndices[i]);
					Sc::BodySim* bodySim = reinterpret_cast<Sc::BodySim*>(reinterpret_cast<PxU8*>(rigidBody) - bodyOffset);
					ccdBodies[nbCcdBodies++] = bodySim;
				}
			}
			if(nbCcdBodies > 0)
			{
				PxI32 startIndex = PxAtomicAdd(&mCCDBodyWriteIndex, PxI32(nbCcdBodies)) - PxI32(nbCcdBodies);
				for(PxU32 a = 0; a < nbCcdBodies; ++a)
				{
					sceneCcdBodies[startIndex + a] = ccdBodies[a];
				}
			}
		}

		virtual const char* getName() const PX_OVERRIDE
		{
			return "ScScene.PxgUpdateBodyAndShapeStatusTask";
		}

	private:
		PX_NOCOPY(PxgUpdateBodyAndShapeStatusTask)
	};

#if PX_SUPPORT_GPU_PHYSX
	// PT: TODO: what is this Pxg class doing here?
	class PxgSimulationControllerCallback : public PxsSimulationControllerCallback
	{
		Sc::Scene* mScene; 
		PxI32 mCcdBodyWriteIndex;

	public:
		PxgSimulationControllerCallback(Sc::Scene* scene) : mScene(scene), mCcdBodyWriteIndex(0)
		{
		}

		virtual void updateScBodyAndShapeSim(PxBaseTask* continuation)	PX_OVERRIDE
		{
			IG::SimpleIslandManager* islandManager = mScene->getSimpleIslandManager();
			PxsSimulationController* simulationController = mScene->getSimulationController();
			PxsContext*	contextLL = mScene->getLowLevelContext();
			IG::IslandSim& islandSim = islandManager->getAccurateIslandSim();
			const PxU32 numBodies = islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);
			const PxNodeIndex*const nodeIndices = islandSim.getActiveNodes(IG::Node::eRIGID_BODY_TYPE);

			PxU32* activatedBodies = simulationController->getActiveBodies();
			PxU32* deactivatedBodies = simulationController->getDeactiveBodies();

			//PxsRigidBody** rigidBodyLL = simulationController->getRigidBodies();
			void** rigidBodyLL = simulationController->getRigidBodies();

			Cm::FlushPool& flushPool = contextLL->getTaskPool();

			PxArray<Sc::BodySim*>& ccdBodies = mScene->getCcdBodies();
			ccdBodies.forceSize_Unsafe(0);
			ccdBodies.reserve(numBodies);
			ccdBodies.forceSize_Unsafe(numBodies);

			mCcdBodyWriteIndex = 0;

			// PT: TASK-CREATION TAG
			for(PxU32 i = 0; i < numBodies; i+=PxgUpdateBodyAndShapeStatusTask::MaxTasks)
			{
				PxgUpdateBodyAndShapeStatusTask* task = PX_PLACEMENT_NEW(flushPool.allocate(sizeof(PxgUpdateBodyAndShapeStatusTask)), PxgUpdateBodyAndShapeStatusTask(nodeIndices + i, 
					PxMin(PxgUpdateBodyAndShapeStatusTask::MaxTasks, numBodies - i), rigidBodyLL, activatedBodies, deactivatedBodies, *mScene, mCcdBodyWriteIndex));
				task->setContinuation(continuation);
				task->removeReference();
			}
		
			const PxU32 nbFrozenShapes = simulationController->getNbFrozenShapes();
			const PxU32 nbUnfrozenShapes = simulationController->getNbUnfrozenShapes();

			if(nbFrozenShapes || nbUnfrozenShapes)
			{
				PxU32* unfrozenShapeIndices = simulationController->getUnfrozenShapes();
				PxU32* frozenShapeIndices = simulationController->getFrozenShapes();

				Sc::ShapeSimBase** shapeSimsLL = simulationController->getShapeSims();
	
				for(PxU32 i=0; i<nbFrozenShapes; ++i)
				{
					Sc::ShapeSimBase* shape = shapeSimsLL[frozenShapeIndices[i]];
					PX_ASSERT(shape);
					shape->destroySqBounds();
				}

				for(PxU32 i=0; i<nbUnfrozenShapes; ++i)
				{
					Sc::ShapeSimBase* shape = shapeSimsLL[unfrozenShapeIndices[i]];
					PX_ASSERT(shape);
					shape->createSqBounds();
				}
			}

			if (simulationController->hasDeformableSurfaces())
			{
				//KS - technically, there's a race condition calling activateNode/deactivateNode, but we know that it is 
				//safe because these deactivate/activate calls came from the solver. This means that we know that the 
				//actors are active currently, so at most we are just clearing/setting the ready for sleeping flag.
				//None of the more complex logic that touching shared state will be executed.
				const PxU32 nbActivatedSurfaces = simulationController->getNbActivatedDeformableSurfaces();
				Dy::DeformableSurface** activatedSurfaces = simulationController->getActivatedDeformableSurfaces();
				for (PxU32 i = 0; i < nbActivatedSurfaces; ++i)
				{
					PxNodeIndex nodeIndex = activatedSurfaces[i]->getSim()->getNodeIndex();
					islandManager->activateNode(nodeIndex);
				}

				const PxU32 nbDeactivatedSurfaces = simulationController->getNbDeactivatedDeformableSurfaces();
				Dy::DeformableSurface** deactivatedSurfaces = simulationController->getDeactivatedDeformableSurfaces();
				for (PxU32 i = 0; i < nbDeactivatedSurfaces; ++i)
				{
					PxNodeIndex nodeIndex = deactivatedSurfaces[i]->getSim()->getNodeIndex();
					islandManager->deactivateNode(nodeIndex);
				}
			}

			if (simulationController->hasDeformableVolumes())
			{
				//KS - technically, there's a race condition calling activateNode/deactivateNode, but we know that it is 
				//safe because these deactivate/activate calls came from the solver. This means that we know that the 
				//actors are active currently, so at most we are just clearing/setting the ready for sleeping flag.
				//None of the more complex logic that touching shared state will be executed.

				const PxU32 nbDeactivatedVolumes = simulationController->getNbDeactivatedDeformableVolumes();
				Dy::DeformableVolume** deactivatedVolumes = simulationController->getDeactivatedDeformableVolumes();
				for (PxU32 i = 0; i < nbDeactivatedVolumes; ++i)
				{
					PxNodeIndex nodeIndex = deactivatedVolumes[i]->getSim()->getNodeIndex();
					islandManager->deactivateNode(nodeIndex);
				}

				const PxU32 nbActivatedVolumes = simulationController->getNbActivatedDeformableVolumes();
				Dy::DeformableVolume** activatedVolumes = simulationController->getActivatedDeformableVolumes();
				for (PxU32 i = 0; i < nbActivatedVolumes; ++i)
				{
					PxNodeIndex nodeIndex = activatedVolumes[i]->getSim()->getNodeIndex();
					islandManager->activateNode(nodeIndex);
				}
			}
		}

		virtual PxU32	getNbCcdBodies()	PX_OVERRIDE
		{
			return PxU32(mCcdBodyWriteIndex);
		}
	};
#endif
}

static Bp::AABBManagerBase* createAABBManagerCPU(const PxSceneDesc& desc, Bp::BroadPhase* broadPhase, Bp::BoundsArray* boundsArray, PinnableArray<PxReal>* contactDistances, VirtualAllocatorCallback& allocator, PxU64 contextID)
{
	return PX_NEW(Bp::AABBManager)(*broadPhase, *boundsArray, *contactDistances,
		desc.limits.maxNbAggregates, desc.limits.maxNbStaticShapes + desc.limits.maxNbDynamicShapes, allocator, contextID,
		desc.kineKineFilteringMode, desc.staticKineFilteringMode);
}

#if PX_SUPPORT_GPU_PHYSX
static Bp::AABBManagerBase* createAABBManagerGPU(PxsKernelWranglerManager* kernelWrangler, PxCudaContextManager* cudaContextManager, PxsHeapMemoryAllocatorManager& heapMemoryAllocationManager,
												const PxSceneDesc& desc, Bp::BroadPhase* broadPhase, Bp::BoundsArray* boundsArray, PinnableArray<PxReal>* contactDistances,
												PxU64 contextID)
{
	return PxvGetPhysXGpu(true)->createGpuAABBManager(
		kernelWrangler,
		cudaContextManager,
		desc.gpuComputeVersion,
		desc.gpuDynamicsConfig,
		heapMemoryAllocationManager,
		*broadPhase, *boundsArray, *contactDistances,
		desc.limits.maxNbAggregates, desc.limits.maxNbStaticShapes + desc.limits.maxNbDynamicShapes, contextID,
		desc.kineKineFilteringMode, desc.staticKineFilteringMode);
}
#endif

Sc::Scene::Scene(const PxSceneDesc& desc, PxU64 contextID) :
	mContextId						(contextID),
	mActiveBodies					("sceneActiveBodies"),
	mActiveKinematicBodyCount		(0),
	mActiveDynamicBodyCount			(0),
	mActiveKinematicsCopy			(NULL),
	mActiveKinematicsCopyCapacity	(0),
	mPointerBlock8Pool				("scenePointerBlock8Pool"),
	mPointerBlock16Pool				("scenePointerBlock16Pool"),
	mPointerBlock32Pool				("scenePointerBlock32Pool"),
	mLLContext						(NULL),
	mAABBManager					(NULL),
	mCCDContext						(NULL),
	mNumFastMovingShapes			(0),
	mCCDPass						(0),
	mSimpleIslandManager			(NULL),
	mDynamicsContext				(NULL),
	mMemoryManager					(NULL),
#if PX_SUPPORT_GPU_PHYSX
	mAvbdGpuDynamicsContext			(NULL),
	mAvbdGpuCallbackSink			(NULL),
	mGpuWranglerManagers			(NULL),
	mHeapMemoryAllocationManager	(NULL),
#endif
	mSimulationController			(NULL),
	mSimulationControllerCallback	(NULL),
	mAvbdCpuSoftScene				(NULL),
	mGravity						(PxVec3(0.0f)),
	mDt								(0),
	mOneOverDt						(0),
	mTimeStamp						(1),		// PT: has to start to 1 to fix determinism bug. I don't know why yet but it works.
	mReportShapePairTimeStamp		(0),
	mTriggerBufferAPI				("sceneTriggerBufferAPI"),
	mArticulations					("sceneArticulations"),
	mBrokenConstraints				("sceneBrokenConstraints"),
	mActiveBreakableConstraints		("sceneActiveBreakableConstraints"),
	mMemBlock128Pool				("PxsContext ConstraintBlock128Pool"),
	mMemBlock256Pool				("PxsContext ConstraintBlock256Pool"),
	mMemBlock384Pool				("PxsContext ConstraintBlock384Pool"),
	mMemBlock512Pool				("PxsContext ConstraintBlock512Pool"),
	mNPhaseCore						(NULL),
	mKineKineFilteringMode			(desc.kineKineFilteringMode),
	mStaticKineFilteringMode		(desc.staticKineFilteringMode),
	mSleepBodies					("sceneSleepBodies"),
	mWokeBodies						("sceneWokeBodies"),
	mEnableStabilization			(desc.flags & PxSceneFlag::eENABLE_STABILIZATION),
	mActiveActors					("clientActiveActors"),
	mFrozenActors					("clientFrozenActors"),
	mClientPosePreviewBodies		("clientPosePreviewBodies"),
	mClientPosePreviewBuffer		("clientPosePreviewBuffer"),
	mSimulationEventCallback		(NULL),
	mInternalFlags					(SceneInternalFlag::eSCENE_DEFAULT),
	mPublicFlags					(desc.flags),
	mAnchorCore						(PxTransform(PxIdentity)),
	mStaticAnchor					(NULL),
	mConstraintSimPool				("ScScene::ConstraintSim"),
	mConstraintInteractionPool		("ScScene::ConstraintInteraction"),
	mBatchRemoveState				(NULL),
	mLostTouchPairs					("sceneLostTouchPairs"),
	mVisualizationParameterChanged	(false),
	mMaxNbArticulationLinks			(0),
	mNbRigidStatics					(0),
	mNbRigidDynamics				(0),
	mNbRigidKinematic				(0),
	mSecondPassNarrowPhase			(contextID, this, "ScScene.secondPassNarrowPhase"),
	mPostNarrowPhase				(contextID, this, "ScScene.postNarrowPhase"),
	mFinalizationPhase				(contextID, this, "ScScene.finalizationPhase"),
	mUpdateCCDMultiPass				(contextID, this, "ScScene.updateCCDMultiPass"),
	mAfterIntegration				(contextID, this, "ScScene.afterIntegration"),
	mPostSolver						(contextID, this, "ScScene.postSolver"),
	mAvbdCpuSoftComponentStep	(contextID, this, "ScScene.avbdCpuSoftComponentStep"),
	mAvbdCpuSoftComponentPredictionFinish(contextID, this, "ScScene.avbdCpuSoftComponentPredictionFinish"),
	mAvbdCpuSoftComponentWriteBackFinish(contextID, this, "ScScene.avbdCpuSoftComponentWriteBackFinish"),
	mSolver							(contextID, this, "ScScene.rigidBodySolver"),
	mUpdateBodies					(contextID, this, "ScScene.updateBodies"),
	mUpdateShapes					(contextID, this, "ScScene.updateShapes"),
	mUpdateSimulationController		(contextID, this, "ScScene.updateSimulationController"),
	mUpdateDynamics					(contextID, this, "ScScene.updateDynamics"),
	mUpdateDynamicsPostPartitioning	(contextID, this, "ScScene.updateDynamicsPostPartitioning"),
	mProcessLostContactsTask		(contextID, this, "ScScene.processLostContact"),
	mProcessLostContactsTask2		(contextID, this, "ScScene.processLostContact2"),
	mProcessLostContactsTask3		(contextID, this, "ScScene.processLostContact3"),
	mDestroyManagersTask			(contextID, this, "ScScene.destroyManagers"),
	mLostTouchReportsTask			(contextID, this, "ScScene.lostTouchReports"),
	mUnregisterInteractionsTask		(contextID, this, "ScScene.unregisterInteractions"),
	mProcessNarrowPhaseLostTouchTasks(contextID, this, "ScScene.processNpLostTouchTask"),
	mProcessNPLostTouchEvents		(contextID, this, "ScScene.processNPLostTouchEvents"),
	mPostThirdPassIslandGenTask		(contextID, this, "ScScene.postThirdPassIslandGenTask"),
#if !USE_SPLIT_SECOND_PASS_ISLAND_GEN
	mPostIslandGen					(contextID, this, "ScScene.postIslandGen"),
#endif
	mIslandGen						(contextID, this, "ScScene.islandGen"),
	mPreRigidBodyNarrowPhase		(contextID, this, "ScScene.preRigidBodyNarrowPhase"),
	mSetEdgesConnectedTask			(contextID, this, "ScScene.setEdgesConnectedTask"),
	mUpdateBoundAndShapeTask		(contextID, this, "ScScene.updateBoundsAndShapesTask"),
	mRigidBodyNarrowPhase			(contextID, this, "ScScene.rigidBodyNarrowPhase"),
	mRigidBodyNPhaseUnlock			(contextID, this, "ScScene.unblockNarrowPhase"),
	mPostBroadPhase					(contextID, this, "ScScene.postBroadPhase"),
	mPostBroadPhaseCont				(contextID, this, "ScScene.postBroadPhaseCont"),
	mPostBroadPhase2				(contextID, this, "ScScene.postBroadPhase2"),
	mPostBroadPhase3				(contextID, this, "ScScene.postBroadPhase3"),
	mPreallocateContactManagers		(contextID, this, "ScScene.preallocateContactManagers"),
	mIslandInsertion				(contextID, this, "ScScene.islandInsertion"),
	mRegisterContactManagers		(contextID, this, "ScScene.registerContactManagers"),
	mRegisterInteractions			(contextID, this, "ScScene.registerInteractions"),
	mRegisterSceneInteractions		(contextID, this, "ScScene.registerSceneInteractions"),
	mBroadPhase						(contextID, this, "ScScene.broadPhase"),
	mAdvanceStep					(contextID, this, "ScScene.advanceStep"),
	mCollideStep					(contextID, this, "ScScene.collideStep"),	
	mBpFirstPass					(contextID, this, "ScScene.broadPhaseFirstPass"),
	mBpSecondPass					(contextID, this, "ScScene.broadPhaseSecondPass"),
	mBpUpdate						(contextID, this, "ScScene.updateBroadPhase"),
	mPreIntegrate                   (contextID, this, "ScScene.preIntegrate"),
	mTaskPool						(16384),
	mTaskManager					(NULL),
	mCudaContextManager				(desc.cudaContextManager),
	mBodyAccelerationTask			(NULL),
	mContactReportsNeedPostSolverVelocity(false),
	mUseGpuDynamics					(false),
	mUseGpuBp						(false),
	mCCDBp							(false),
	mSimulationStage				(SimulationStage::eCOMPLETE),
	mPosePreviewBodies				("scenePosePreviewBodies"),
	mOverlapFilterTaskHead			(NULL),
	mOverlapCreatedTaskHead			(NULL),
	mIslandInsertionTaskHead		(NULL),
	mIsCollisionPhaseActive			(false),
	mIsDirectGPUAPIInitialized		(false),
	mOnSleepingStateChanged			(NULL)
#if PX_SUPPORT_GPU_PHYSX
	,mDeformableSurfaces			("sceneDeformableSurfaces"), 
	mDeformableVolumes				("sceneDeformableVolumes"),
	mParticleSystems				("sceneParticleSystems")
#endif
{
#if PX_SUPPORT_GPU_PHYSX
	mLLDeformableSurfacePool	= PX_NEW(LLDeformableSurfacePool);
	mLLDeformableVolumePool		= PX_NEW(LLDeformableVolumePool);
	mLLParticleSystemPool		= PX_NEW(LLParticleSystemPool);

	mWokeDeformableVolumeListValid = true;
	mSleepDeformableVolumeListValid = true;
#endif

	for(PxU32 type = 0; type < InteractionType::eTRACKED_IN_SCENE_COUNT; ++type)
		mInteractions[type].reserve(64);

	for (int i=0; i < InteractionType::eTRACKED_IN_SCENE_COUNT; ++i)
		mActiveInteractionCount[i] = 0;

	mStats						= PX_NEW(SimStats);
	mConstraintIDTracker		= PX_NEW(ObjectIDTracker);
	mActorIDTracker				= PX_NEW(ObjectIDTracker);
	mElementIDPool				= PX_NEW(ObjectIDTracker);

	mTriggerBufferExtraData		= reinterpret_cast<TriggerBufferExtraData*>(PX_ALLOC(sizeof(TriggerBufferExtraData), "ScScene::TriggerBufferExtraData"));
	PX_PLACEMENT_NEW(mTriggerBufferExtraData, TriggerBufferExtraData("ScScene::TriggerPairExtraData"));

	mStaticSimPool				= PX_NEW(PreallocatingPool<StaticSim>)(64, "StaticSim");
	mBodySimPool				= PX_NEW(PreallocatingPool<BodySim>)(64, "BodySim");
	mShapeSimPool				= PX_NEW(PreallocatingPool<ShapeSim>)(128, "ShapeSim");
	mSimStateDataPool			= PX_NEW(PxPool<SimStateData>)("ScScene::SimStateData");

	mSqBoundsManager			= PX_NEW(SqBoundsManager);

	mTaskManager				= PxTaskManager::createTaskManager(*PxGetErrorCallback(), desc.cpuDispatcher);

	for(PxU32 i=0; i<PxGeometryType::eGEOMETRY_COUNT; i++)
		mNbGeometries[i] = 0;

	bool useGpuDynamics = false;
	bool useGpuBroadphase = false;
	bool useGpuAvbdBackend = false;
#if !PX_SUPPORT_GPU_PHYSX
	PX_UNUSED(useGpuAvbdBackend);
#endif

#if PX_SUPPORT_GPU_PHYSX
	if(desc.flags & PxSceneFlag::eENABLE_GPU_DYNAMICS)
	{
		// AVBD keeps the CPU scene/taskgraph as the authority and may attach a
		// dedicated GPU owner-wave backend.  It must never enter the generic GPU
		// PGS/TGS dynamics context.
		if(desc.solverType == PxSolverType::eAVBD)
		{
#if defined(PX_ENABLE_EXPERIMENTAL_AVBD_GPU_OWNER_WAVE)
			if(!mCudaContextManager)
				outputError<PxErrorCode::eDEBUG_WARNING>(__LINE__, "GPU AVBD backend unavailable, using CPU AVBD");
			else if(mCudaContextManager->supportsArchSM30())
				useGpuAvbdBackend = true;
#else
			outputError<PxErrorCode::eDEBUG_WARNING>(__LINE__,
				"GPU AVBD owner-wave support is experimental and disabled in this build; using CPU AVBD");
#endif
		}
		else if(!mCudaContextManager)
			outputError<PxErrorCode::eDEBUG_WARNING>(__LINE__, "GPU solver pipeline failed, switching to software");
		else if(mCudaContextManager->supportsArchSM30())
			useGpuDynamics = true;
	}

	if(desc.broadPhaseType == PxBroadPhaseType::eGPU)
	{
		if(!mCudaContextManager)
			outputError<PxErrorCode::eDEBUG_WARNING>(__LINE__, "GPU Bp pipeline failed, switching to software");
		else if(mCudaContextManager->supportsArchSM30())
			useGpuBroadphase = true;
	}
#endif

	mUseGpuDynamics = useGpuDynamics;
	mUseGpuBp = useGpuBroadphase;

	mLLContext = PX_NEW(PxsContext)(desc, mTaskManager, mTaskPool, mCudaContextManager, desc.contactPairSlabSize, contextID);
	
	if (mLLContext == 0)
	{
		outputError<PxErrorCode::eINVALID_PARAMETER>(__LINE__, "Failed to create context!");
		return;
	}
	mLLContext->setMaterialManager(&getMaterialManager());

	// Allocator used for shared data that may be accessed as cuda host memory depending on the pipeline
	// configuration (useGpuBroadphase, useGpuDynamics, directAPI). Cases where device mapped memory is
	// required, are handled separately. It gets used for Bp::BoundsArray, Bp::AABBManagerBase, contact distances,
	// Dy::Context (base class for dynamics context) and PxsTransformCache
	VirtualAllocatorCallback* allocator = NULL;

#if PX_SUPPORT_GPU_PHYSX
	if (useGpuBroadphase || useGpuDynamics || useGpuAvbdBackend)
	{
		PxPhysXGpu* physxGpu = PxvGetPhysXGpu(true);

		// PT: this creates a PxgMemoryManager, whose host memory allocator is a PxgCudaHostMemoryAllocatorCallback
		mMemoryManager = physxGpu->createGpuMemoryManager(mLLContext->getCudaContextManager());
		mGpuWranglerManagers = physxGpu->getGpuKernelWranglerManager(mLLContext->getCudaContextManager());
		// PT: this creates a PxgHeapMemoryAllocatorManager
		mHeapMemoryAllocationManager = physxGpu->createGpuHeapMemoryAllocatorManager(desc.gpuDynamicsConfig.heapCapacity, mMemoryManager, desc.gpuComputeVersion);
		allocator = mHeapMemoryAllocationManager->mPinnedHostMemoryAllocator;
	}
	else
#endif
	{
		// PT: this creates a PxsDefaultMemoryManager
		mMemoryManager = createDefaultMemoryManager();
		allocator = mMemoryManager->getPinnedHostMemoryAllocator();
	}
	PX_ASSERT(allocator);

	Bp::BroadPhase* broadPhase = NULL;

	//Note: broadphase should be independent of AABBManager.  MBP uses it to call getBPBounds but it has 
	//already been passed all bounds in BroadPhase::update() so should use that instead.
	// PT: above comment is obsolete: MBP now doesn't call getBPBounds anymore (except in commented out code)
	// and it is instead the GPU broadphase which is not independent from the GPU AABB manager.......
	if(!useGpuBroadphase)
	{
		PxBroadPhaseType::Enum broadPhaseType = desc.broadPhaseType;

		if (broadPhaseType == PxBroadPhaseType::eGPU)
			broadPhaseType = PxBroadPhaseType::eABP;

		broadPhase = Bp::BroadPhase::create(
			broadPhaseType, 
			desc.limits.maxNbRegions, 
			desc.limits.maxNbBroadPhaseOverlaps, 
			desc.limits.maxNbStaticShapes, 
			desc.limits.maxNbDynamicShapes,
			contextID);
	}
#if PX_SUPPORT_GPU_PHYSX
	else
	{
		PxGpuBroadPhaseDesc defaultGpuBPDesc;
		broadPhase = PxvGetPhysXGpu(true)->createGpuBroadPhase(	desc.gpuBroadPhaseDesc ? *desc.gpuBroadPhaseDesc : defaultGpuBPDesc,
																mGpuWranglerManagers, mLLContext->getCudaContextManager(),
																desc.gpuComputeVersion, desc.gpuDynamicsConfig,
																*mHeapMemoryAllocationManager, contextID);
	}
#endif

#if PX_SUPPORT_GPU_PHYSX
	const bool directAPI = mPublicFlags & PxSceneFlag::eENABLE_DIRECT_GPU_API;
	if(directAPI)
	{
		PX_ASSERT(mHeapMemoryAllocationManager);
		// Direct pipeline needs mapped bounds: mergeBoundsAndTransformsChanges
		mBoundsArray = PxvGetPhysXGpu(true)->createGpuBounds(*mHeapMemoryAllocationManager->mPinnedHostMappedMemoryAllocator);
	}
	else
#endif
	{
		// Non-direct pipeline is fine with regular pinned memory: copyBoundsAndTransforms
		mBoundsArray = PX_NEW(Bp::BoundsArray)(*allocator);
	}
	
	mContactDistance = PX_PLACEMENT_NEW(PX_ALLOC(sizeof(PinnableArray<PxReal>), "ContactDistance"), PinnableArray<PxReal>)(*allocator);
	mHasContactDistanceChanged = false;

	const bool useEnhancedDeterminism = mPublicFlags & PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;

	mSimpleIslandManager = PX_NEW(IG::SimpleIslandManager)(useEnhancedDeterminism, useGpuBroadphase || useGpuDynamics, contextID);
	PX_ASSERT(mSimpleIslandManager);

	PxvNphaseImplementationFallback* cpuNphaseImplementation = createNphaseImplementationContext(*mLLContext, &mSimpleIslandManager->getAccurateIslandSim(), *allocator, useGpuDynamics);

	if (!useGpuDynamics)
	{
		// PT: we must pass mPublicFlags to the contexts in case it has been tweaked by the above code

		if (desc.solverType == PxSolverType::ePGS)
		{
			mDynamicsContext = createDynamicsContext(&mLLContext->getNpMemBlockPool(), mLLContext->getTaskPool(), mLLContext->getSimStats(),
													*allocator, &getMaterialManager(), *mSimpleIslandManager, contextID,
													desc.maxBiasCoefficient, desc.getTolerancesScale().length, mPublicFlags);
		}
		else if (desc.solverType == PxSolverType::eAVBD)
		{
			mDynamicsContext = createAVBDDynamicsContext
			(&mLLContext->getNpMemBlockPool(), mLLContext->getScratchAllocator(),
				mLLContext->getTaskPool(), mLLContext->getSimStats(), &mLLContext->getTaskManager(), *allocator, &getMaterialManager(),
				*mSimpleIslandManager, contextID, desc.maxBiasCoefficient, desc.getTolerancesScale().length,
				desc.avbdIterations, desc.avbdJointIterationOverride,
				desc.avbdEnableEarlyStop, mPublicFlags);
		}
		else
		{
			mDynamicsContext = createTGSDynamicsContext(&mLLContext->getNpMemBlockPool(), mLLContext->getTaskPool(), mLLContext->getSimStats(),
														*allocator, &getMaterialManager(), *mSimpleIslandManager, contextID,
																desc.getTolerancesScale().length, mPublicFlags);
		}

#if PX_SUPPORT_GPU_PHYSX && defined(PX_ENABLE_EXPERIMENTAL_AVBD_GPU_OWNER_WAVE)
		if(useGpuAvbdBackend && desc.solverType == PxSolverType::eAVBD)
		{
			PxPhysXGpu* physxGpu = PxvGetPhysXGpu(true);
			PxAvbdGpuDynamicsContext gpuAvbdContext;
			const bool created = physxGpu->createGpuAvbdDynamicsContext(
				mGpuWranglerManagers, mCudaContextManager, *mSimpleIslandManager,
				*allocator,
				*mHeapMemoryAllocationManager->mPinnedHostMappedMemoryAllocator,
				desc.maxBiasCoefficient, mLLContext->getSimStats(),
				desc.getTolerancesScale().length, contextID, mPublicFlags,
				gpuAvbdContext);
			Dy::AvbdDynamicsContext* cpuAvbdContext =
			static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
			Dy::AvbdRigidGpuWaveCallbackTable callbacks;
			const bool configured = created && gpuAvbdContext.context &&
				gpuAvbdContext.callbackSink && gpuAvbdContext.waveBackend &&
				cpuAvbdContext->getRigidGpuWaveCallbackTable(callbacks) &&
				gpuAvbdContext.callbackSink->setCpuWaveCallbacks(callbacks) &&
				gpuAvbdContext.callbackSink->attachToCpuAvbdContext(*cpuAvbdContext) &&
				gpuAvbdContext.callbackSink->enableOwnerWaveBackend();
			if(configured)
			{
				mAvbdGpuDynamicsContext = gpuAvbdContext.context;
				mAvbdGpuCallbackSink = gpuAvbdContext.callbackSink;
			}
			else if(gpuAvbdContext.context)
			{
				if(gpuAvbdContext.callbackSink &&
					cpuAvbdContext->getRigidGpuWaveBackend() ==
						gpuAvbdContext.waveBackend)
					gpuAvbdContext.callbackSink->detachFromCpuAvbdContext(*cpuAvbdContext);
				gpuAvbdContext.context->destroy();
				outputError<PxErrorCode::eINTERNAL_ERROR>(__LINE__, "GPU AVBD owner-wave hybrid setup failed, using CPU AVBD");
			}
			else
			{
				outputError<PxErrorCode::eDEBUG_WARNING>(__LINE__, "GPU AVBD owner-wave context creation failed, using CPU AVBD");
			}
		}
#endif

		mLLContext->setNphaseImplementationContext(cpuNphaseImplementation);

		mSimulationControllerCallback = PX_NEW(ScSimulationControllerCallback)(this);
		mSimulationController = PX_NEW(SimulationController)(mSimulationControllerCallback);

		if (!useGpuBroadphase)
			mAABBManager = createAABBManagerCPU(desc, broadPhase, mBoundsArray, mContactDistance, *allocator, contextID);
#if PX_SUPPORT_GPU_PHYSX
		else
			mAABBManager = createAABBManagerGPU(mGpuWranglerManagers, mLLContext->getCudaContextManager(), *mHeapMemoryAllocationManager, desc, broadPhase, mBoundsArray, mContactDistance, contextID);
#endif
	}
	else
	{
#if PX_SUPPORT_GPU_PHYSX
		const bool enableBodyAccelerations = mPublicFlags & PxSceneFlag::eENABLE_BODY_ACCELERATIONS;

		PxPhysXGpu* physxGpu = PxvGetPhysXGpu(true);

		// PT: why are we using mPublicFlags in one case and desc in other cases?

		mDynamicsContext = physxGpu->createGpuDynamicsContext(mLLContext->getTaskPool(), mGpuWranglerManagers, mLLContext->getCudaContextManager(),
			desc.gpuDynamicsConfig, *mSimpleIslandManager, desc.gpuMaxNumPartitions, desc.gpuMaxNumStaticPartitions,
			desc.maxBiasCoefficient, desc.gpuComputeVersion, mLLContext->getSimStats(), *mHeapMemoryAllocationManager, 
			desc.solverType, desc.getTolerancesScale().length, contextID, mPublicFlags);

		void* contactStreamBase = NULL;
		void* patchStreamBase = NULL;
		void* forceAndIndiceStreamBase = NULL;

		mDynamicsContext->getDataStreamBase(contactStreamBase, patchStreamBase, forceAndIndiceStreamBase);

		mLLContext->setNphaseFallbackImplementationContext(cpuNphaseImplementation);

		PxvNphaseImplementationContext* gpuNphaseImplementation = physxGpu->createGpuNphaseImplementationContext(*mLLContext, mGpuWranglerManagers, cpuNphaseImplementation, desc.gpuDynamicsConfig, contactStreamBase, patchStreamBase,
			forceAndIndiceStreamBase, *mBoundsArray, &mSimpleIslandManager->getAccurateIslandSim(), mDynamicsContext, desc.gpuComputeVersion, *mHeapMemoryAllocationManager, useGpuBroadphase);

		mSimulationControllerCallback = PX_NEW(PxgSimulationControllerCallback)(this);

		mSimulationController = physxGpu->createGpuSimulationController(mGpuWranglerManagers, mLLContext->getCudaContextManager(),
			mDynamicsContext, gpuNphaseImplementation, broadPhase, useGpuBroadphase, mSimulationControllerCallback, desc.gpuComputeVersion, *mHeapMemoryAllocationManager,
			desc.gpuDynamicsConfig.maxDeformableVolumeContacts,
			desc.gpuDynamicsConfig.maxDeformableSurfaceContacts,
			desc.gpuDynamicsConfig.maxParticleContacts,
			desc.gpuDynamicsConfig.collisionStackSize, enableBodyAccelerations);

		mSimulationController->setBounds(mBoundsArray);
		mDynamicsContext->setSimulationController(mSimulationController);

		mLLContext->setNphaseImplementationContext(gpuNphaseImplementation);

		mLLContext->mContactStreamPool = &mDynamicsContext->getContactStreamPool();
		mLLContext->mPatchStreamPool = &mDynamicsContext->getPatchStreamPool();
		mLLContext->mForceAndIndiceStreamPool = &mDynamicsContext->getForceStreamPool();
		mLLContext->mFrictionPatchStreamPool = &mDynamicsContext->getFrictionPatchStreamPool();

		if (!useGpuBroadphase)
			mAABBManager = createAABBManagerCPU(desc, broadPhase, mBoundsArray, mContactDistance, *allocator, contextID);
		else
			mAABBManager = createAABBManagerGPU(mGpuWranglerManagers, mLLContext->getCudaContextManager(), 
				*mHeapMemoryAllocationManager, desc, broadPhase, mBoundsArray, mContactDistance, contextID);
#endif
	}

	if(!useGpuDynamics &&
		desc.solverType == PxSolverType::eAVBD)
	{
		mAvbdCpuSoftScene = PX_NEW(AvbdCpuSoftScene)(
			mDeformableVolumeMaterialManager,
			mDeformableSurfaceMaterialManager,
			mMaterialManager,
			mContextId,
			*mSimpleIslandManager);
		static_cast<Dy::AvbdDynamicsContext*>(
			mDynamicsContext)->setSoftIslandProvider(
				mAvbdCpuSoftScene);
	}

	//Construct the bitmap of updated actors required as input to the broadphase update
	if(desc.limits.maxNbBodies)
	{
		// PT: TODO: revisit this. Why do we handle the added/removed and updated bitmaps entirely differently, in different places? And what is this weird formula here?
		mAABBManager->getChangedAABBMgActorHandleMap().resize((2*desc.limits.maxNbBodies + 256) & ~255);
	}

#if PX_SUPPORT_GPU_PHYSX
	if(directAPI)
	{
		// Direct pipeline needs mapped memory transform cache: mergeBoundsAndTransformsChanges
		mLLContext->createTransformCache(*mHeapMemoryAllocationManager->mPinnedHostMappedMemoryAllocator, Cm::PinnableAllocatorFallback::eDISABLED);
	}
	else
#endif
	{
		// Non-direct pipeline is fine with regular pinned memory that can fallback to pageable memory: copyBoundsAndTransforms
		mLLContext->createTransformCache(*allocator, Cm::PinnableAllocatorFallback::eENABLED);
	}

	mLLContext->setContactDistance(mContactDistance);

	mCCDContext = PX_NEW(PxsCCDContext)(mLLContext, mDynamicsContext->getThresholdStream(), *mLLContext->getNphaseImplementationContext(), desc.ccdThreshold, useGpuBroadphase || useGpuDynamics);
	
	setSolverBatchSize(desc.solverBatchSize);
	setSolverArticBatchSize(desc.solverArticulationBatchSize);
	mDynamicsContext->setFrictionOffsetThreshold(desc.frictionOffsetThreshold);
	mDynamicsContext->setCCDSeparationThreshold(desc.ccdMaxSeparation);
	mDynamicsContext->setCorrelationDistance(desc.frictionCorrelationDistance);

	const PxTolerancesScale& scale = Physics::getInstance().getTolerancesScale();
	mLLContext->setMeshContactMargin(0.01f * scale.length);
	mLLContext->setToleranceLength(scale.length);

	// the original descriptor uses 
	//    bounce iff impact velocity  > threshold
	// but LL use 
	//    bounce iff separation velocity < -threshold 
	// hence we negate here.

	mDynamicsContext->setBounceThreshold(-desc.bounceThresholdVelocity);

	mStaticAnchor = mStaticSimPool->construct(*this, mAnchorCore);

	mNPhaseCore = PX_NEW(NPhaseCore)(*this, desc);

	// Init dominance matrix
	{
		//init all dominance pairs such that:
		//if g1 == g2, then (1.0f, 1.0f) is returned
		//if g1 <  g2, then (0.0f, 1.0f) is returned
		//if g1 >  g2, then (1.0f, 0.0f) is returned

		PxU32 mask = ~PxU32(1);
		for (unsigned i = 0; i < PX_MAX_DOMINANCE_GROUP; ++i, mask <<= 1)
			mDominanceBitMatrix[i] = ~mask;
	}
		
//	DeterminismDebugger::begin();

	mWokeBodyListValid = true;
	mSleepBodyListValid = true;

	//load from desc:
	setLimits(desc.limits);

	// Create broad phase
	mBroadphaseManager.setBroadPhaseCallback(desc.broadPhaseCallback);

	setGravity(desc.gravity);

	setPCM(desc.flags & PxSceneFlag::eENABLE_PCM);

	setContactCache(!(desc.flags & PxSceneFlag::eDISABLE_CONTACT_CACHE));
	setSimulationEventCallback(desc.simulationEventCallback);
	setContactModifyCallback(desc.contactModifyCallback);
	setCCDContactModifyCallback(desc.ccdContactModifyCallback);
	if (desc.deformableVolumePostSolveCallback)
		setDeformableVolumeGpuPostSolveCallback(desc.deformableVolumePostSolveCallback);
	if (desc.deformableSurfacePostSolveCallback)
		setDeformableSurfaceGpuPostSolveCallback(desc.deformableSurfacePostSolveCallback);
	setCCDMaxPasses(desc.ccdMaxPasses);
	PX_ASSERT(mNPhaseCore); // refactor paranoia
	
	PX_ASSERT(	((desc.filterShaderData) && (desc.filterShaderDataSize > 0)) ||
				(!(desc.filterShaderData) && (desc.filterShaderDataSize == 0))	);
	if (desc.filterShaderData)
	{
		mFilterShaderData = PX_ALLOC(desc.filterShaderDataSize, sFilterShaderDataMemAllocId);
		PxMemCopy(mFilterShaderData, desc.filterShaderData, desc.filterShaderDataSize);
		mFilterShaderDataSize = desc.filterShaderDataSize;
		mFilterShaderDataCapacity = desc.filterShaderDataSize;
	}
	else
	{
		mFilterShaderData = NULL;
		mFilterShaderDataSize = 0;
		mFilterShaderDataCapacity = 0;
	}
	mFilterShader = desc.filterShader;
	mFilterCallback = desc.filterCallback;
}

bool Sc::Scene::addAvbdCpuDeformableVolume(
	DeformableVolumeCore& core,
	PxTetrahedronMesh& simulationMesh,
	PxTetrahedronMesh& collisionMesh,
	PxDeformableVolumeAuxData& auxData)
{
	if(getSolverType() != PxSolverType::eAVBD || isUsingGpuDynamics())
		return false;
	if(!mAvbdCpuSoftScene)
	{
		mAvbdCpuSoftScene = PX_NEW(AvbdCpuSoftScene)(
			mDeformableVolumeMaterialManager,
			mDeformableSurfaceMaterialManager,
			mMaterialManager,
			mContextId,
			*mSimpleIslandManager);
		static_cast<Dy::AvbdDynamicsContext*>(
			mDynamicsContext)->setSoftIslandProvider(
				mAvbdCpuSoftScene);
	}
	return mAvbdCpuSoftScene->add(
		core, simulationMesh, collisionMesh, auxData,
		mDeformableVolumeMaterialManager);
}

void Sc::Scene::removeAvbdCpuDeformableVolume(
	DeformableVolumeCore& core)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->remove(core);
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeWorldPin(
	DeformableVolumeCore& core,
	PxU32 localVertex,
	const PxVec3& worldTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addWorldPin(
			core, localVertex, worldTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeWorldElementAttachment(
	DeformableVolumeCore& core,
	PxU32 tetrahedronIndex,
	const PxVec4& barycentric,
	const PxVec3& worldTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addWorldElementPin(
			core, false, tetrahedronIndex, barycentric, worldTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableVolumeWorldPin(
	DeformableVolumeCore& core,
	PxU32 handle,
	const PxVec3& worldTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updateWorldPin(
			core, handle, worldTarget);
}

void Sc::Scene::removeAvbdCpuDeformableVolumeWorldPin(
	DeformableVolumeCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeWorldPin(core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeRigidAttachment(
	DeformableVolumeCore& core,
	BodyCore& rigidCore,
	PxU32 localVertex,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addRigidAttachment(
			core, rigidCore, localVertex, actorLocalTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeRigidElementAttachment(
	DeformableVolumeCore& core,
	BodyCore& rigidCore,
	PxU32 tetrahedronIndex,
	const PxVec4& barycentric,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addRigidElementAttachment(
			core, rigidCore, false, tetrahedronIndex,
			barycentric, actorLocalTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableVolumeRigidAttachment(
	DeformableVolumeCore& core,
	PxU32 handle,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updateRigidAttachment(
			core, handle, actorLocalTarget);
}

void Sc::Scene::removeAvbdCpuDeformableVolumeRigidAttachment(
	DeformableVolumeCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeRigidAttachment(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeArticulationAttachment(
	DeformableVolumeCore& core,
	BodyCore& linkCore,
	PxU32 localVertex,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addArticulationAttachment(
			core, linkCore, localVertex, actorLocalTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeArticulationElementAttachment(
	DeformableVolumeCore& core,
	BodyCore& linkCore,
	PxU32 tetrahedronIndex,
	const PxVec4& barycentric,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addArticulationElementAttachment(
			core, linkCore, false, tetrahedronIndex,
			barycentric, actorLocalTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableVolumeArticulationAttachment(
	DeformableVolumeCore& core,
	PxU32 handle,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updateArticulationAttachment(
			core, handle, actorLocalTarget);
}

void Sc::Scene::removeAvbdCpuDeformableVolumeArticulationAttachment(
	DeformableVolumeCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeArticulationAttachment(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeKinematicAttachment(
	DeformableVolumeCore& core,
	BodyCore& kinematicCore,
	PxU32 localVertex,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addKinematicAttachment(
			core, kinematicCore, localVertex, actorLocalTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeKinematicElementAttachment(
	DeformableVolumeCore& core,
	BodyCore& kinematicCore,
	PxU32 tetrahedronIndex,
	const PxVec4& barycentric,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addKinematicElementAttachment(
			core, kinematicCore, false, tetrahedronIndex,
			barycentric, actorLocalTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableVolumeKinematicAttachment(
	DeformableVolumeCore& core,
	PxU32 handle,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updatePrescribedAttachment(
			core, handle, actorLocalTarget);
}

void Sc::Scene::removeAvbdCpuDeformableVolumeKinematicAttachment(
	DeformableVolumeCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removePrescribedAttachment(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeStaticAttachment(
	DeformableVolumeCore& core,
	StaticCore& staticCore,
	PxU32 localVertex,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addStaticAttachment(
			core, staticCore, localVertex, actorLocalTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeStaticElementAttachment(
	DeformableVolumeCore& core,
	StaticCore& staticCore,
	PxU32 tetrahedronIndex,
	const PxVec4& barycentric,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addStaticElementAttachment(
			core, staticCore, false, tetrahedronIndex,
			barycentric, actorLocalTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableVolumeStaticAttachment(
	DeformableVolumeCore& core,
	PxU32 handle,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updatePrescribedAttachment(
			core, handle, actorLocalTarget);
}

void Sc::Scene::removeAvbdCpuDeformableVolumeStaticAttachment(
	DeformableVolumeCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removePrescribedAttachment(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeRigidActorFilter(
	DeformableVolumeCore& core,
	ActorCore& rigidCore,
	const PxU32* elementIndices,
	PxU32 elementCount,
	bool filterAllElements)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addVolumeRigidActorFilter(
			core, rigidCore, elementIndices, elementCount,
			filterAllElements)
		: PX_MAX_U32;
}

void Sc::Scene::removeAvbdCpuDeformableVolumeRigidActorFilter(
	DeformableVolumeCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeRigidActorFilter(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformablePairAttachment(
	ActorCore& core0,
	bool element0,
	PxU32 index0,
	const PxVec4& barycentric0,
	ActorCore& core1,
	bool element1,
	PxU32 index1,
	const PxVec4& barycentric1)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addSoftPairAttachment(
			core0, element0, index0, barycentric0,
			core1, element1, index1, barycentric1)
		: PX_MAX_U32;
}

void Sc::Scene::removeAvbdCpuDeformablePairAttachment(
	ActorCore& core0,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeSoftPairAttachment(
			core0, handle);
}

bool Sc::Scene::addAvbdCpuDeformableSurface(
	DeformableSurfaceCore& core,
	PxTriangleMesh& triangleMesh)
{
	if(getSolverType() != PxSolverType::eAVBD ||
		isUsingGpuDynamics())
		return false;
	if(!mAvbdCpuSoftScene)
	{
		mAvbdCpuSoftScene = PX_NEW(AvbdCpuSoftScene)(
			mDeformableVolumeMaterialManager,
			mDeformableSurfaceMaterialManager,
			mMaterialManager,
			mContextId,
			*mSimpleIslandManager);
		static_cast<Dy::AvbdDynamicsContext*>(
			mDynamicsContext)->setSoftIslandProvider(
				mAvbdCpuSoftScene);
	}
	return mAvbdCpuSoftScene->addSurface(
		core, triangleMesh);
}

void Sc::Scene::removeAvbdCpuDeformableSurface(
	DeformableSurfaceCore& core)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeSurface(core);
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceWorldPin(
	DeformableSurfaceCore& core,
	PxU32 localVertex,
	const PxVec3& worldTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addWorldPin(
			core, localVertex, worldTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceWorldElementAttachment(
	DeformableSurfaceCore& core,
	PxU32 triangleIndex,
	const PxVec4& barycentric,
	const PxVec3& worldTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addWorldElementPin(
			core, true, triangleIndex, barycentric, worldTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableSurfaceWorldPin(
	DeformableSurfaceCore& core,
	PxU32 handle,
	const PxVec3& worldTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updateWorldPin(
			core, handle, worldTarget);
}

void Sc::Scene::removeAvbdCpuDeformableSurfaceWorldPin(
	DeformableSurfaceCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeWorldPin(core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceRigidAttachment(
	DeformableSurfaceCore& core,
	BodyCore& rigidCore,
	PxU32 localVertex,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addRigidAttachment(
			core, rigidCore, localVertex, actorLocalTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceRigidElementAttachment(
	DeformableSurfaceCore& core,
	BodyCore& rigidCore,
	PxU32 triangleIndex,
	const PxVec4& barycentric,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addRigidElementAttachment(
			core, rigidCore, true, triangleIndex,
			barycentric, actorLocalTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableSurfaceRigidAttachment(
	DeformableSurfaceCore& core,
	PxU32 handle,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updateRigidAttachment(
			core, handle, actorLocalTarget);
}

void Sc::Scene::removeAvbdCpuDeformableSurfaceRigidAttachment(
	DeformableSurfaceCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeRigidAttachment(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceArticulationAttachment(
	DeformableSurfaceCore& core,
	BodyCore& linkCore,
	PxU32 localVertex,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addArticulationAttachment(
			core, linkCore, localVertex, actorLocalTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceArticulationElementAttachment(
	DeformableSurfaceCore& core,
	BodyCore& linkCore,
	PxU32 triangleIndex,
	const PxVec4& barycentric,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addArticulationElementAttachment(
			core, linkCore, true, triangleIndex,
			barycentric, actorLocalTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableSurfaceArticulationAttachment(
	DeformableSurfaceCore& core,
	PxU32 handle,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updateArticulationAttachment(
			core, handle, actorLocalTarget);
}

void Sc::Scene::removeAvbdCpuDeformableSurfaceArticulationAttachment(
	DeformableSurfaceCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeArticulationAttachment(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceKinematicAttachment(
	DeformableSurfaceCore& core,
	BodyCore& kinematicCore,
	PxU32 localVertex,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addKinematicAttachment(
			core, kinematicCore, localVertex, actorLocalTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceKinematicElementAttachment(
	DeformableSurfaceCore& core,
	BodyCore& kinematicCore,
	PxU32 triangleIndex,
	const PxVec4& barycentric,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addKinematicElementAttachment(
			core, kinematicCore, true, triangleIndex,
			barycentric, actorLocalTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableSurfaceKinematicAttachment(
	DeformableSurfaceCore& core,
	PxU32 handle,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updatePrescribedAttachment(
			core, handle, actorLocalTarget);
}

void Sc::Scene::removeAvbdCpuDeformableSurfaceKinematicAttachment(
	DeformableSurfaceCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removePrescribedAttachment(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceStaticAttachment(
	DeformableSurfaceCore& core,
	StaticCore& staticCore,
	PxU32 localVertex,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addStaticAttachment(
			core, staticCore, localVertex, actorLocalTarget)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceStaticElementAttachment(
	DeformableSurfaceCore& core,
	StaticCore& staticCore,
	PxU32 triangleIndex,
	const PxVec4& barycentric,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addStaticElementAttachment(
			core, staticCore, true, triangleIndex,
			barycentric, actorLocalTarget)
		: PX_MAX_U32;
}

bool Sc::Scene::updateAvbdCpuDeformableSurfaceStaticAttachment(
	DeformableSurfaceCore& core,
	PxU32 handle,
	const PxVec3& actorLocalTarget)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->updatePrescribedAttachment(
			core, handle, actorLocalTarget);
}

void Sc::Scene::removeAvbdCpuDeformableSurfaceStaticAttachment(
	DeformableSurfaceCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removePrescribedAttachment(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceRigidActorFilter(
	DeformableSurfaceCore& core,
	ActorCore& rigidCore,
	const PxU32* elementIndices,
	PxU32 elementCount,
	bool filterAllElements)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addRigidActorFilter(
			core, rigidCore, elementIndices, elementCount,
			filterAllElements)
		: PX_MAX_U32;
}

void Sc::Scene::removeAvbdCpuDeformableSurfaceRigidActorFilter(
	DeformableSurfaceCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeRigidActorFilter(
			core, handle);
}

PxU32 Sc::Scene::addAvbdCpuDeformableSurfaceSurfaceFilter(
	DeformableSurfaceCore& core0,
	DeformableSurfaceCore& core1,
	const PxU32* elementIndices0,
	const PxU32* elementIndices1,
	PxU32 pairCount)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addSurfaceSurfaceFilter(
			core0, core1, elementIndices0, elementIndices1,
			pairCount)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeSurfaceFilter(
	DeformableVolumeCore& volumeCore,
	DeformableSurfaceCore& surfaceCore,
	const PxU32* volumeElementIndices,
	const PxU32* surfaceElementIndices,
	PxU32 pairCount)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addVolumeSurfaceFilter(
			volumeCore, surfaceCore, volumeElementIndices,
			surfaceElementIndices, pairCount)
		: PX_MAX_U32;
}

PxU32 Sc::Scene::addAvbdCpuDeformableVolumeVolumeFilter(
	DeformableVolumeCore& core0,
	DeformableVolumeCore& core1,
	const PxU32* elementIndices0,
	const PxU32* elementIndices1,
	PxU32 pairCount)
{
	return mAvbdCpuSoftScene
		? mAvbdCpuSoftScene->addVolumeVolumeFilter(
			core0, core1, elementIndices0,
			elementIndices1, pairCount)
		: PX_MAX_U32;
}

void Sc::Scene::removeAvbdCpuDeformablePairFilter(
	DeformableSurfaceCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeDeformablePairFilter(
			core, handle);
}

void Sc::Scene::removeAvbdCpuDeformablePairFilter(
	DeformableVolumeCore& core,
	PxU32 handle)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeDeformablePairFilter(
			core, handle);
}

void Sc::Scene::addAvbdCpuStaticShape(
	StaticCore& core, ShapeCore& shape)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->addStaticShape(core, shape);
}

void Sc::Scene::removeAvbdCpuStaticShape(
	StaticCore& core, const ShapeCore& shape)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeStaticShape(core, shape);
}

void Sc::Scene::removeAvbdCpuStatic(StaticCore& core)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeStatic(core);
}

void Sc::Scene::addAvbdCpuDynamicShape(
	BodyCore& core, ShapeCore& shape)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->addDynamicShape(core, shape);
}

void Sc::Scene::removeAvbdCpuDynamicShape(
	BodyCore& core, const ShapeCore& shape)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeDynamicShape(core, shape);
}

void Sc::Scene::removeAvbdCpuDynamic(BodyCore& core)
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->removeDynamic(core);
}

void Sc::Scene::stepAvbdCpuDeformableVolumes()
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->step(
			mDt, mGravity, mDeformableVolumeMaterialManager,
			mMaterialManager,
			!(mPublicFlags & PxSceneFlag::eDISABLE_SLEEPING));
}

bool Sc::Scene::stepAvbdCpuDeformableVolumesSolveOnly()
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->stepStandaloneComponentSolveOnly(
			mDt, mGravity, mDeformableVolumeMaterialManager,
			mMaterialManager,
			!(mPublicFlags & PxSceneFlag::eDISABLE_SLEEPING));
}

bool Sc::Scene::prepareAvbdCpuDeformableVolumesSolve()
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->prepareStandaloneComponentSolve(
			mDt, mGravity, mDeformableVolumeMaterialManager,
			mMaterialManager,
			!(mPublicFlags & PxSceneFlag::eDISABLE_SLEEPING));
}

void Sc::Scene::predictAvbdCpuDeformableVolumes()
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->predictStandaloneComponent(mDt, mGravity);
}

bool Sc::Scene::resumeAvbdCpuDeformableVolumesSolve(
	bool* causalLayerTaskReady, bool* worldPlaneContactTaskReady,
	bool* rigidBoxSdfContactTaskReady,
	bool* rigidSphereSdfContactTaskReady)
{
	return mAvbdCpuSoftScene &&
		mAvbdCpuSoftScene->resumeStandaloneComponentSolve(
			mDt, mGravity, causalLayerTaskReady,
			worldPlaneContactTaskReady, rigidBoxSdfContactTaskReady,
			rigidSphereSdfContactTaskReady);
}

void Sc::Scene::writeBackAvbdCpuDeformableVolumes()
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->writeBackStandaloneComponent();
}

void Sc::Scene::finishAvbdCpuDeformableVolumesStandaloneStep()
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->finishStandaloneComponentStep(
			mDt, !(mPublicFlags & PxSceneFlag::eDISABLE_SLEEPING));
}

void Sc::Scene::finishAvbdCpuSoftComponentNoOpTask()
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->finishStandaloneTaskGraphNoOp();
}

bool Sc::Scene::scheduleAvbdCpuSoftComponentStep(
	PxBaseTask* continuation)
{
	if(!mAvbdCpuSoftScene ||
		getSolverType() != PxSolverType::eAVBD ||
		isUsingGpuDynamics() || !mDynamicsContext ||
		!continuation || !continuation->getTaskManager())
		return false;

	const PxU32 dispatcherWorkers =
		continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount();
	const bool enhancedDeterminism =
		mPublicFlags & PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;
	mAvbdCpuSoftScene->setStandaloneTaskGraphExecutionPolicy(
		dispatcherWorkers, enhancedDeterminism);
	if(!mAvbdCpuSoftScene->shouldScheduleStandaloneTaskGraph(
		dispatcherWorkers))
		return false;

	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	const PxU32 particleCount =
		mAvbdCpuSoftScene->getStandaloneTaskGraphParticleCount();
	const bool forcedP45CausalLayerValidation =
		particleCount < 128 &&
		((Dy::avbdUseCausalLayerTaskFanIn() &&
			Dy::avbdForceCausalLayerTaskFanIn()) ||
			Dy::avbdForceCausalLayerTaskGraphReference());
	if(context.isTaskGraphSerialMode())
	{
		context.recordStandaloneSoftSerialSolve(particleCount);
		mAvbdCpuSoftScene->recordStandaloneTaskGraphSerialSolve(
			dispatcherWorkers, particleCount);
		return false;
	}

	context.recordStandaloneSoftSolveTaskSubmitted(
		particleCount, forcedP45CausalLayerValidation);
	mAvbdCpuSoftScene->recordStandaloneTaskGraphSubmission(
		dispatcherWorkers, particleCount);
	mAvbdCpuSoftComponentStep.setContinuation(continuation);
	mAvbdCpuSoftComponentStep.removeReference();
	return true;
}

bool Sc::Scene::scheduleAvbdCpuSoftComponentWriteBack(
	PxBaseTask* continuation)
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext || !continuation ||
		!continuation->getTaskManager())
		return false;

	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	if(context.isTaskGraphSerialMode())
		return false;
	const PxU32 dispatcherWorkers =
		continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount();
	const PxU32 taskCount =
		mAvbdCpuSoftScene->getStandaloneWriteBackTaskCount(
			dispatcherWorkers);
	if(taskCount == 0)
		return false;

	// This persistent delegate is the fan-in.  It takes one continuation
	// reference before the parent component task releases its own, then runs
	// only after every entry-local write-back task completed.
	mAvbdCpuSoftComponentWriteBackFinish.setContinuation(continuation);
	mAvbdCpuSoftScene->submitStandaloneWriteBackTasks(
		taskCount, &mAvbdCpuSoftComponentWriteBackFinish, context);
	mAvbdCpuSoftComponentWriteBackFinish.removeReference();
	return true;
}

bool Sc::Scene::scheduleAvbdCpuSoftComponentCausalLayer(
	PxBaseTask* continuation)
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext || !continuation ||
		!continuation->getTaskManager())
		return false;
	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	if(context.isTaskGraphSerialMode())
		return false;
	const PxU32 dispatcherWorkers =
		continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount();
	return mAvbdCpuSoftScene->submitStandaloneCausalLayerTask(
		dispatcherWorkers, *this, continuation, context);
}

bool Sc::Scene::finishAvbdCpuSoftComponentCausalLayerSerialFallback()
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext)
		return false;
	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	return mAvbdCpuSoftScene->finishStandaloneCausalLayerSerialFallback(
		mDt, context);
}

bool Sc::Scene::scheduleAvbdCpuSoftComponentWorldPlaneContact(
	PxBaseTask* continuation)
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext || !continuation ||
		!continuation->getTaskManager())
		return false;
	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	if(context.isTaskGraphSerialMode())
		return false;
	const PxU32 dispatcherWorkers =
		continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount();
	return mAvbdCpuSoftScene->submitStandaloneWorldPlaneContactTask(
		dispatcherWorkers, *this, continuation, context);
}

bool Sc::Scene::finishAvbdCpuSoftComponentWorldPlaneContactSerialFallback(
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext)
		return false;
	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	return mAvbdCpuSoftScene->finishStandaloneWorldPlaneContactSerialFallback(
		mDt, context, nextLayerReady, nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

bool Sc::Scene::scheduleAvbdCpuSoftComponentRigidBoxSdfContact(
	PxBaseTask* continuation)
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext || !continuation ||
		!continuation->getTaskManager())
		return false;
	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	if(context.isTaskGraphSerialMode())
		return false;
	const PxU32 dispatcherWorkers =
		continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount();
	return mAvbdCpuSoftScene->submitStandaloneRigidBoxSdfContactTask(
		dispatcherWorkers, *this, continuation, context);
}

bool Sc::Scene::finishAvbdCpuSoftComponentRigidBoxSdfContactSerialFallback(
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext)
		return false;
	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	return mAvbdCpuSoftScene->finishStandaloneRigidBoxSdfContactSerialFallback(
		mDt, context, nextLayerReady, nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

bool Sc::Scene::scheduleAvbdCpuSoftComponentRigidSphereSdfContact(
	PxBaseTask* continuation)
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext || !continuation ||
		!continuation->getTaskManager())
		return false;
	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	if(context.isTaskGraphSerialMode())
		return false;
	const PxU32 dispatcherWorkers =
		continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount();
	return mAvbdCpuSoftScene->submitStandaloneRigidSphereSdfContactTask(
		dispatcherWorkers, *this, continuation, context);
}

bool Sc::Scene::finishAvbdCpuSoftComponentRigidSphereSdfContactSerialFallback(
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext)
		return false;
	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	return mAvbdCpuSoftScene->finishStandaloneRigidSphereSdfContactSerialFallback(
		mDt, context, nextLayerReady, nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

void Sc::Scene::avbdCpuSoftComponentCausalLayerFinish(
	PxBaseTask* continuation)
{
	PX_PROFILE_ZONE("Sc::Scene::avbdCpuSoftComponentCausalLayerFinish",
		mContextId);
	Dy::AvbdDynamicsContext* const avbdContext =
		static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	PX_ASSERT(avbdContext && mAvbdCpuSoftScene);
	if(!avbdContext || !mAvbdCpuSoftScene)
		return;
	bool nextLayerReady = false;
	bool nextWorldPlaneContactTaskReady = false;
	bool nextRigidBoxSdfContactTaskReady = false;
	bool nextRigidSphereSdfContactTaskReady = false;
	bool componentStepCompleted =
		mAvbdCpuSoftScene->completeStandaloneCausalLayerTask(
			mDt, *avbdContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	while(componentStepCompleted &&
		(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
		 nextRigidSphereSdfContactTaskReady))
	{
		if(nextWorldPlaneContactTaskReady)
		{
			if(scheduleAvbdCpuSoftComponentWorldPlaneContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentWorldPlaneContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
		else if(nextRigidBoxSdfContactTaskReady)
		{
			if(scheduleAvbdCpuSoftComponentRigidBoxSdfContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentRigidBoxSdfContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
		else
		{
			if(scheduleAvbdCpuSoftComponentRigidSphereSdfContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentRigidSphereSdfContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
	}
	if(nextLayerReady &&
		scheduleAvbdCpuSoftComponentCausalLayer(continuation))
		return;
	if(nextLayerReady)
		componentStepCompleted =
			finishAvbdCpuSoftComponentCausalLayerSerialFallback();
	finishAvbdCpuSoftComponentNoOpTask();
	avbdContext->endSolveTask();
	if(componentStepCompleted &&
		scheduleAvbdCpuSoftComponentWriteBack(continuation))
		return;
	if(componentStepCompleted)
	{
		avbdContext->recordSerialWriteBackStage();
		writeBackAvbdCpuDeformableVolumes();
		finishAvbdCpuDeformableVolumesStandaloneStep();
	}
	finishPostSolverAfterAvbdCpuSoftStep();
}

void Sc::Scene::avbdCpuSoftComponentWorldPlaneContactFinish(
	PxBaseTask* continuation)
{
	PX_PROFILE_ZONE("Sc::Scene::avbdCpuSoftComponentWorldPlaneContactFinish",
		mContextId);
	Dy::AvbdDynamicsContext* const avbdContext =
		static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	PX_ASSERT(avbdContext && mAvbdCpuSoftScene);
	if(!avbdContext || !mAvbdCpuSoftScene)
		return;
	bool nextLayerReady = false;
	bool nextWorldPlaneContactTaskReady = false;
	bool nextRigidBoxSdfContactTaskReady = false;
	bool nextRigidSphereSdfContactTaskReady = false;
	bool componentStepCompleted =
		mAvbdCpuSoftScene->completeStandaloneWorldPlaneContactTask(
			mDt, *avbdContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	while(componentStepCompleted &&
		(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
		 nextRigidSphereSdfContactTaskReady))
	{
		if(nextWorldPlaneContactTaskReady)
		{
			if(scheduleAvbdCpuSoftComponentWorldPlaneContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentWorldPlaneContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
		else if(nextRigidBoxSdfContactTaskReady)
		{
			if(scheduleAvbdCpuSoftComponentRigidBoxSdfContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentRigidBoxSdfContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
		else
		{
			if(scheduleAvbdCpuSoftComponentRigidSphereSdfContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentRigidSphereSdfContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
	}
	if(nextLayerReady &&
		scheduleAvbdCpuSoftComponentCausalLayer(continuation))
		return;
	if(nextLayerReady)
		componentStepCompleted =
			finishAvbdCpuSoftComponentCausalLayerSerialFallback();
	finishAvbdCpuSoftComponentNoOpTask();
	avbdContext->endSolveTask();
	if(componentStepCompleted &&
		scheduleAvbdCpuSoftComponentWriteBack(continuation))
		return;
	if(componentStepCompleted)
	{
		avbdContext->recordSerialWriteBackStage();
		writeBackAvbdCpuDeformableVolumes();
		finishAvbdCpuDeformableVolumesStandaloneStep();
	}
	finishPostSolverAfterAvbdCpuSoftStep();
}

void Sc::Scene::avbdCpuSoftComponentRigidBoxSdfContactFinish(
	PxBaseTask* continuation)
{
	PX_PROFILE_ZONE("Sc::Scene::avbdCpuSoftComponentRigidBoxSdfContactFinish",
		mContextId);
	Dy::AvbdDynamicsContext* const avbdContext =
		static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	PX_ASSERT(avbdContext && mAvbdCpuSoftScene);
	if(!avbdContext || !mAvbdCpuSoftScene)
		return;
	bool nextLayerReady = false;
	bool nextWorldPlaneContactTaskReady = false;
	bool nextRigidBoxSdfContactTaskReady = false;
	bool nextRigidSphereSdfContactTaskReady = false;
	bool componentStepCompleted =
		mAvbdCpuSoftScene->completeStandaloneRigidBoxSdfContactTask(
			mDt, *avbdContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	while(componentStepCompleted &&
		(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
		 nextRigidSphereSdfContactTaskReady))
	{
		if(nextWorldPlaneContactTaskReady)
		{
			if(scheduleAvbdCpuSoftComponentWorldPlaneContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentWorldPlaneContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
		else if(nextRigidBoxSdfContactTaskReady)
		{
			if(scheduleAvbdCpuSoftComponentRigidBoxSdfContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentRigidBoxSdfContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
		else
		{
			if(scheduleAvbdCpuSoftComponentRigidSphereSdfContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentRigidSphereSdfContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
	}
	if(nextLayerReady &&
		scheduleAvbdCpuSoftComponentCausalLayer(continuation))
		return;
	if(nextLayerReady)
		componentStepCompleted =
			finishAvbdCpuSoftComponentCausalLayerSerialFallback();
	finishAvbdCpuSoftComponentNoOpTask();
	avbdContext->endSolveTask();
	if(componentStepCompleted &&
		scheduleAvbdCpuSoftComponentWriteBack(continuation))
		return;
	if(componentStepCompleted)
	{
		avbdContext->recordSerialWriteBackStage();
		writeBackAvbdCpuDeformableVolumes();
		finishAvbdCpuDeformableVolumesStandaloneStep();
	}
	finishPostSolverAfterAvbdCpuSoftStep();
}

void Sc::Scene::avbdCpuSoftComponentRigidSphereSdfContactFinish(
	PxBaseTask* continuation)
{
	PX_PROFILE_ZONE("Sc::Scene::avbdCpuSoftComponentRigidSphereSdfContactFinish",
		mContextId);
	Dy::AvbdDynamicsContext* const avbdContext =
		static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	PX_ASSERT(avbdContext && mAvbdCpuSoftScene);
	if(!avbdContext || !mAvbdCpuSoftScene)
		return;
	bool nextLayerReady = false;
	bool nextWorldPlaneContactTaskReady = false;
	bool nextRigidBoxSdfContactTaskReady = false;
	bool nextRigidSphereSdfContactTaskReady = false;
	bool componentStepCompleted =
		mAvbdCpuSoftScene->completeStandaloneRigidSphereSdfContactTask(
			mDt, *avbdContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	while(componentStepCompleted &&
		(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
		 nextRigidSphereSdfContactTaskReady))
	{
		if(nextWorldPlaneContactTaskReady)
		{
			if(scheduleAvbdCpuSoftComponentWorldPlaneContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentWorldPlaneContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
		else if(nextRigidBoxSdfContactTaskReady)
		{
			if(scheduleAvbdCpuSoftComponentRigidBoxSdfContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentRigidBoxSdfContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
		else
		{
			if(scheduleAvbdCpuSoftComponentRigidSphereSdfContact(continuation))
				return;
			componentStepCompleted =
				finishAvbdCpuSoftComponentRigidSphereSdfContactSerialFallback(
					nextLayerReady, nextWorldPlaneContactTaskReady,
					nextRigidBoxSdfContactTaskReady,
					nextRigidSphereSdfContactTaskReady);
		}
	}
	if(nextLayerReady &&
		scheduleAvbdCpuSoftComponentCausalLayer(continuation))
		return;
	if(nextLayerReady)
		componentStepCompleted =
			finishAvbdCpuSoftComponentCausalLayerSerialFallback();
	finishAvbdCpuSoftComponentNoOpTask();
	avbdContext->endSolveTask();
	if(componentStepCompleted &&
		scheduleAvbdCpuSoftComponentWriteBack(continuation))
		return;
	if(componentStepCompleted)
	{
		avbdContext->recordSerialWriteBackStage();
		writeBackAvbdCpuDeformableVolumes();
		finishAvbdCpuDeformableVolumesStandaloneStep();
	}
	finishPostSolverAfterAvbdCpuSoftStep();
}

bool Sc::Scene::scheduleAvbdCpuSoftComponentPrediction(
	PxBaseTask* continuation)
{
	if(!mAvbdCpuSoftScene || !mDynamicsContext || !continuation ||
		!continuation->getTaskManager())
		return false;

	Dy::AvbdDynamicsContext& context =
		*static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
	if(context.isTaskGraphSerialMode())
		return false;
	const PxU32 dispatcherWorkers =
		continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount();
	const PxU32 taskCount =
		mAvbdCpuSoftScene->getStandalonePredictionTaskCount(
			dispatcherWorkers);
	if(taskCount == 0)
		return false;

	// The prediction finish retains the original Scene continuation until all
	// disjoint particle ranges complete, then performs the serial OGC/contact
	// resume and forwards into the existing write-back fan-in.
	mAvbdCpuSoftComponentPredictionFinish.setContinuation(continuation);
	mAvbdCpuSoftScene->submitStandalonePredictionTasks(
		taskCount, mDt, mGravity,
		&mAvbdCpuSoftComponentPredictionFinish, context);
	mAvbdCpuSoftComponentPredictionFinish.removeReference();
	return true;
}

void Sc::Scene::prepareAvbdCpuSoftIslandGeneration()
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->prepareIslandGeneration(
			mDt, mGravity,
			!(mPublicFlags & PxSceneFlag::eDISABLE_SLEEPING));
}

void Sc::Scene::release()
{
	// TODO: PT: check virtual stuff

	mTimeStamp++;

	if(mAvbdCpuSoftScene &&
		getSolverType() == PxSolverType::eAVBD &&
		!isUsingGpuDynamics())
	{
		static_cast<Dy::AvbdDynamicsContext*>(
			mDynamicsContext)->setSoftIslandProvider(NULL);
	}
	PX_DELETE(mAvbdCpuSoftScene);
	mAvbdCpuSoftScene = NULL;

	//collisionSpace.purgeAllPairs();

	//purgePairs();
	//releaseTagData();

	// We know release all the shapes before the collision space
	//collisionSpace.deleteAllShapes();

	//collisionSpace.release();

	//DeterminismDebugger::end();

	PX_FREE(mActiveKinematicsCopy);

	PX_DELETE(mNPhaseCore);

	PX_FREE(mFilterShaderData);

	if(mStaticAnchor)
		mStaticSimPool->destroy(mStaticAnchor);

	// Free object IDs and the deleted object id map
	postReportsCleanup();

	//before the task manager
	if (mLLContext)
	{
		if(mLLContext->getNphaseFallbackImplementationContext())
		{
			mLLContext->getNphaseFallbackImplementationContext()->destroy();
			mLLContext->setNphaseFallbackImplementationContext(NULL);
		}

		if(mLLContext->getNphaseImplementationContext())
		{
			mLLContext->getNphaseImplementationContext()->destroy();
			mLLContext->setNphaseImplementationContext(NULL);
		}
	}

	PX_DELETE(mSqBoundsManager);
	PX_DELETE(mBoundsArray);

	PX_DELETE(mSimStateDataPool);
	PX_DELETE(mStaticSimPool);
	PX_DELETE(mShapeSimPool);
	PX_DELETE(mBodySimPool);
#if PX_SUPPORT_GPU_PHYSX
	gpu_releasePools();
#endif
	mTriggerBufferExtraData->~TriggerBufferExtraData();
	PX_FREE(mTriggerBufferExtraData);

	PX_DELETE(mElementIDPool);
	PX_DELETE(mActorIDTracker);
	PX_DELETE(mConstraintIDTracker);
	PX_DELETE(mStats);

	Bp::BroadPhase* broadPhase = mAABBManager->getBroadPhase();
	mAABBManager->destroy();
	PX_RELEASE(broadPhase);

	PX_DELETE(mSimulationControllerCallback);
	PX_DELETE(mSimulationController);

#if PX_SUPPORT_GPU_PHYSX
	if(mAvbdGpuDynamicsContext)
	{
		Dy::AvbdDynamicsContext* cpuAvbdContext =
			static_cast<Dy::AvbdDynamicsContext*>(mDynamicsContext);
		if(mAvbdGpuCallbackSink && cpuAvbdContext)
			mAvbdGpuCallbackSink->detachFromCpuAvbdContext(*cpuAvbdContext);
		mAvbdGpuDynamicsContext->destroy();
		mAvbdGpuDynamicsContext = NULL;
		mAvbdGpuCallbackSink = NULL;
	}
#endif

	mDynamicsContext->destroy();

	PX_DELETE(mCCDContext);

	PX_DELETE(mSimpleIslandManager);

	PX_RELEASE(mTaskManager);
	PX_DELETE(mLLContext);

	// PT: TODO: revisit this
	mContactDistance->~PinnableArray<PxReal>();
	PX_FREE(mContactDistance);

#if PX_SUPPORT_GPU_PHYSX
	gpu_release();
#endif

	PX_DELETE(mMemoryManager);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::preAllocate(PxU32 nbStatics, PxU32 nbBodies, PxU32 nbStaticShapes, PxU32 nbDynamicShapes)
{
	// PT: TODO: this is only used for my addActors benchmark for now. Pre-allocate more arrays here.

	mActiveBodies.reserve(PxMax<PxU32>(64,nbBodies));

	mStaticSimPool->preAllocate(nbStatics);

	mBodySimPool->preAllocate(nbBodies);

	mShapeSimPool->preAllocate(nbStaticShapes + nbDynamicShapes);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::addDirtyArticulationSim(Sc::ArticulationSim* artiSim)
{
	artiSim->setDirtyFlag(ArticulationSimDirtyFlag::eUPDATE);
	mDirtyArticulationSims.insert(artiSim);
}

void Sc::Scene::removeDirtyArticulationSim(Sc::ArticulationSim* artiSim)
{
	artiSim->setDirtyFlag(ArticulationSimDirtyFlag::eNONE);
	mDirtyArticulationSims.erase(artiSim);
}

void Sc::Scene::addToActiveList(ActorSim& actorSim)
{
	PX_ASSERT(actorSim.getActiveListIndex() >= SC_NOT_IN_ACTIVE_LIST_INDEX);

	ActorCore* appendedActorCore = &actorSim.getActorCore();

	if (actorSim.isDynamicRigid())
	{
		// Sort: kinematic before dynamic
		const PxU32 size = mActiveBodies.size();
		PxU32 incomingBodyActiveListIndex = size;							// PT: by default we append the incoming body at the end of the current array.

		BodySim& bodySim = static_cast<BodySim&>(actorSim);
		if (bodySim.isKinematic())											// PT: Except if incoming body is kinematic, in which case:
		{
			const PxU32 nbKinematics = mActiveKinematicBodyCount++;			// PT: - we increase their number
			if (nbKinematics != size)										// PT: - if there's at least one dynamic in the array...
			{
				PX_ASSERT(appendedActorCore != mActiveBodies[nbKinematics]);
				appendedActorCore = mActiveBodies[nbKinematics];			// PT: ...then we grab the first dynamic after the kinematics...
				appendedActorCore->getSim()->setActiveListIndex(size);		// PT: ...and we move that one back to the end of the array...

				mActiveBodies[nbKinematics] = static_cast<BodyCore*>(&actorSim.getActorCore());			// PT: ...while the incoming kine replaces the dynamic we moved out.
				incomingBodyActiveListIndex = nbKinematics;					// PT: ...thus the incoming kine's index is the prev #kines.
			}
		}
		
		// for active compound rigids add to separate array, so we dont have to traverse all active actors
		if (bodySim.readInternalFlag(BodySim::BF_IS_COMPOUND_RIGID))
		{
			PX_ASSERT(actorSim.getActiveCompoundListIndex() >= SC_NOT_IN_ACTIVE_LIST_INDEX);
			const PxU32 compoundIndex = mActiveCompoundBodies.size();
			mActiveCompoundBodies.pushBack(static_cast<BodyCore*>(appendedActorCore));
			actorSim.setActiveCompoundListIndex(compoundIndex);
		}

		actorSim.setActiveListIndex(incomingBodyActiveListIndex);			// PT: will be 'size' or 'nbKinematics', 'dynamicIndex'
		mActiveBodies.pushBack(static_cast<BodyCore*>(appendedActorCore));	// PT: will be the incoming object or the first dynamic we moved out.
	}
#if PX_SUPPORT_GPU_PHYSX
	else
		gpu_addToActiveList(actorSim, appendedActorCore);
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void removeFromActiveCompoundBodyList(Sc::ActorSim& actorSim, PxArray<Sc::BodyCore*>& activeCompoundBodies)
{
	const PxU32 removedCompoundIndex = actorSim.getActiveCompoundListIndex();
	PX_ASSERT(removedCompoundIndex < SC_NOT_IN_ACTIVE_LIST_INDEX);
	actorSim.setActiveCompoundListIndex(SC_NOT_IN_ACTIVE_LIST_INDEX);

	const PxU32 newCompoundSize = activeCompoundBodies.size() - 1;

	if(removedCompoundIndex != newCompoundSize)
	{
		Sc::BodyCore* lastBody = activeCompoundBodies[newCompoundSize];
		activeCompoundBodies[removedCompoundIndex] = lastBody;
		lastBody->getSim()->setActiveCompoundListIndex(removedCompoundIndex);
	}
	activeCompoundBodies.forceSize_Unsafe(newCompoundSize);
}

void Sc::Scene::removeFromActiveCompoundBodyList(BodySim& body)
{
	::removeFromActiveCompoundBodyList(body, mActiveCompoundBodies);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::removeFromActiveList(ActorSim& actorSim)
{
	PxU32 removedActiveIndex = actorSim.getActiveListIndex();
	PX_ASSERT(removedActiveIndex < SC_NOT_IN_ACTIVE_LIST_INDEX);
	actorSim.setActiveListIndex(SC_NOT_IN_ACTIVE_LIST_INDEX);

	if (actorSim.isDynamicRigid())
	{
		PX_ASSERT(mActiveBodies[removedActiveIndex] == &actorSim.getActorCore());

		const PxU32 newSize = mActiveBodies.size() - 1;

		BodySim& bodySim = static_cast<BodySim&>(actorSim);

		// Sort: kinematic before dynamic,
		if (removedActiveIndex < mActiveKinematicBodyCount)	// PT: same as 'body.isKinematic()' but without accessing the Core data
		{
			PX_ASSERT(mActiveKinematicBodyCount);
			PX_ASSERT(bodySim.isKinematic());
			const PxU32 swapIndex = --mActiveKinematicBodyCount;
			if (newSize != swapIndex				// PT: equal if the array only contains kinematics
				&& removedActiveIndex < swapIndex)	// PT: i.e. "if we don't remove the last kinematic"
			{
				BodyCore* swapBody = mActiveBodies[swapIndex];
				swapBody->getSim()->setActiveListIndex(removedActiveIndex);
				mActiveBodies[removedActiveIndex] = swapBody;
				removedActiveIndex = swapIndex;
			}
		}

		// for active compound rigids add to separate array, so we dont have to traverse all active actors
		// A.B. TODO we should handle kinematic switch, no need to hold kinematic rigids in compound list
		if(bodySim.readInternalFlag(BodySim::BF_IS_COMPOUND_RIGID))
			::removeFromActiveCompoundBodyList(actorSim, mActiveCompoundBodies);

		if (removedActiveIndex != newSize)
		{
			Sc::BodyCore* lastBody = mActiveBodies[newSize];
			mActiveBodies[removedActiveIndex] = lastBody;
			lastBody->getSim()->setActiveListIndex(removedActiveIndex);
		}
		mActiveBodies.forceSize_Unsafe(newSize);
	}
#if PX_SUPPORT_GPU_PHYSX
	else
		gpu_removeFromActiveList(actorSim, removedActiveIndex);
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::swapInActiveBodyList(BodySim& body)
{
	PX_ASSERT(!body.isStaticRigid() && !body.isDeformableSurface() && !body.isDeformableVolume() && !body.isParticleSystem());
	const PxU32 activeListIndex = body.getActiveListIndex();
	PX_ASSERT(activeListIndex < SC_NOT_IN_ACTIVE_LIST_INDEX);

	PxU32 swapIndex;
	PxU32 newActiveKinematicBodyCount;
	if(activeListIndex < mActiveKinematicBodyCount)
	{
		// kinematic -> dynamic
		PX_ASSERT(!body.isKinematic());  // the corresponding flag gets switched before this call
		PX_ASSERT(mActiveKinematicBodyCount > 0);  // there has to be at least one kinematic body

		swapIndex = mActiveKinematicBodyCount - 1;
		newActiveKinematicBodyCount = swapIndex;
	}
	else
	{
		// dynamic -> kinematic
		PX_ASSERT(body.isKinematic());  // the corresponding flag gets switched before this call
		PX_ASSERT(mActiveKinematicBodyCount < mActiveBodies.size());  // there has to be at least one dynamic body

		swapIndex = mActiveKinematicBodyCount;
		newActiveKinematicBodyCount = swapIndex + 1;
	}

	BodyCore*& swapBodyRef = mActiveBodies[swapIndex];
	body.setActiveListIndex(swapIndex);
	BodyCore* swapBody = swapBodyRef;
	swapBodyRef = &body.getBodyCore();

	swapBody->getSim()->setActiveListIndex(activeListIndex);
	mActiveBodies[activeListIndex] = swapBody;

	mActiveKinematicBodyCount = newActiveKinematicBodyCount;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::registerInteraction(ElementSimInteraction* interaction, bool active)
{
	const InteractionType::Enum type = interaction->getType();
	const PxU32 sceneArrayIndex = mInteractions[type].size();
	interaction->setInteractionId(sceneArrayIndex);

	mInteractions[type].pushBack(interaction);
	if (active)
	{
		if (sceneArrayIndex > mActiveInteractionCount[type])
			swapInteractionArrayIndices(sceneArrayIndex, mActiveInteractionCount[type], type);
		mActiveInteractionCount[type]++;
	}

	mNPhaseCore->registerInteraction(interaction);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::unregisterInteraction(ElementSimInteraction* interaction)
{
	const InteractionType::Enum type = interaction->getType();
	const PxU32 sceneArrayIndex = interaction->getInteractionId();
	PX_ASSERT(sceneArrayIndex != PX_INVALID_INTERACTION_SCENE_ID);
	if (sceneArrayIndex == PX_INVALID_INTERACTION_SCENE_ID)
	{
		// vreutskyy: Defensive coding added for OM-117470. In theory this should not be needed, as we
		// shouldn't be able to unregister twice. We could/should remove this eventually.
		// ### DEFENSIVE
		outputError<PxErrorCode::eINTERNAL_ERROR>(__LINE__, "Unexpectedly unregistered an interaction that does not have a valid interaction ID.");
		return;
	}
	mInteractions[type].replaceWithLast(sceneArrayIndex);
	interaction->setInteractionId(PX_INVALID_INTERACTION_SCENE_ID);
	if (sceneArrayIndex<mInteractions[type].size()) // The removed interaction was the last one, do not reset its sceneArrayIndex
		mInteractions[type][sceneArrayIndex]->setInteractionId(sceneArrayIndex);
	if (sceneArrayIndex<mActiveInteractionCount[type])
	{
		mActiveInteractionCount[type]--;
		if (mActiveInteractionCount[type]<mInteractions[type].size())
			swapInteractionArrayIndices(sceneArrayIndex, mActiveInteractionCount[type], type);
	}

	mNPhaseCore->unregisterInteraction(interaction);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::swapInteractionArrayIndices(PxU32 id1, PxU32 id2, InteractionType::Enum type)
{
	PxArray<ElementSimInteraction*>& interArray = mInteractions[type];
	ElementSimInteraction* interaction1 = interArray[id1];
	ElementSimInteraction* interaction2 = interArray[id2];
	interArray[id1] = interaction2;
	interArray[id2] = interaction1;
	interaction1->setInteractionId(id2);
	interaction2->setInteractionId(id1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::notifyInteractionActivated(Interaction* interaction)
{
	PX_ASSERT((interaction->getType() == InteractionType::eOVERLAP) || (interaction->getType() == InteractionType::eTRIGGER));
	PX_ASSERT(interaction->readInteractionFlag(InteractionFlag::eIS_ACTIVE));
	PX_ASSERT(interaction->getInteractionId() != PX_INVALID_INTERACTION_SCENE_ID);

	const InteractionType::Enum type = interaction->getType();

	PX_ASSERT(interaction->getInteractionId() >= mActiveInteractionCount[type]);
	
	if (mActiveInteractionCount[type] < mInteractions[type].size())
		swapInteractionArrayIndices(mActiveInteractionCount[type], interaction->getInteractionId(), type);
	mActiveInteractionCount[type]++;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::notifyInteractionDeactivated(Interaction* interaction)
{
	PX_ASSERT((interaction->getType() == InteractionType::eOVERLAP) || (interaction->getType() == InteractionType::eTRIGGER));
	PX_ASSERT(!interaction->readInteractionFlag(InteractionFlag::eIS_ACTIVE));
	PX_ASSERT(interaction->getInteractionId() != PX_INVALID_INTERACTION_SCENE_ID);

	const InteractionType::Enum type = interaction->getType();
	PX_ASSERT(interaction->getInteractionId() < mActiveInteractionCount[type]);

	if (mActiveInteractionCount[type] > 1)
		swapInteractionArrayIndices(mActiveInteractionCount[type]-1, interaction->getInteractionId(), type);
	mActiveInteractionCount[type]--;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void** Sc::Scene::allocatePointerBlock(PxU32 size)
{
	PX_ASSERT(size>32 || size == 32 || size == 16 || size == 8);
	void* ptr;
	if(size==8)
		ptr = mPointerBlock8Pool.construct();
	else if(size == 16)
		ptr = mPointerBlock16Pool.construct();
	else if(size == 32)
		ptr = mPointerBlock32Pool.construct();
	else
		ptr = PX_ALLOC(size * sizeof(void*), "void*");

	return reinterpret_cast<void**>(ptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::deallocatePointerBlock(void** block, PxU32 size)
{
	PX_ASSERT(size>32 || size == 32 || size == 16 || size == 8);
	if(size==8)
		mPointerBlock8Pool.destroy(reinterpret_cast<PointerBlock8*>(block));
	else if(size == 16)
		mPointerBlock16Pool.destroy(reinterpret_cast<PointerBlock16*>(block));
	else if(size == 32)
		mPointerBlock32Pool.destroy(reinterpret_cast<PointerBlock32*>(block));
	else
		PX_FREE(block);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::setFilterShaderData(const void* data, PxU32 dataSize)
{
	PX_UNUSED(sFilterShaderDataMemAllocId);

	if (data)
	{
		PX_ASSERT(dataSize > 0);

		void* buffer;

		if (dataSize <= mFilterShaderDataCapacity)
			buffer = mFilterShaderData;
		else
		{
			buffer = PX_ALLOC(dataSize, sFilterShaderDataMemAllocId);
			if (buffer)
			{
				mFilterShaderDataCapacity = dataSize;
				PX_FREE(mFilterShaderData);
			}
			else
			{
				outputError<PxErrorCode::eOUT_OF_MEMORY>(__LINE__, "Failed to allocate memory for filter shader data!");
				return;
			}
		}

		PxMemCopy(buffer, data, dataSize);
		mFilterShaderData = buffer;
		mFilterShaderDataSize = dataSize;
	}
	else
	{
		PX_ASSERT(dataSize == 0);

		PX_FREE(mFilterShaderData);
		mFilterShaderDataSize = 0;
		mFilterShaderDataCapacity = 0;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//stepSetup is called in solve, but not collide
void Sc::Scene::stepSetupSolve(PxBaseTask* continuation)
{
	PX_PROFILE_ZONE("Sim.stepSetupSolve", mContextId);

	kinematicsSetup(continuation);
}

void Sc::Scene::advance(PxReal timeStep, PxBaseTask* continuation)
{
	if(timeStep != 0.0f)
	{
		setElapsedTime(timeStep);

		mAdvanceStep.setContinuation(continuation);

		stepSetupSolve(&mAdvanceStep);		
		
		mAdvanceStep.removeReference();
	}
}

void Sc::Scene::collide(PxReal timeStep, PxBaseTask* continuation)
{
	mDt = timeStep;

	stepSetupCollide(continuation);

	mLLContext->beginUpdate();

	mCollideStep.setContinuation(continuation);
	mCollideStep.removeReference();
}

void Sc::Scene::endSimulation()
{
	// Handle user contact filtering
	// Note: Do this before the contact callbacks get fired since the filter callback might
	//       trigger contact reports (touch lost due to re-filtering)

	mBroadphaseManager.prepareOutOfBoundsCallbacks(mAABBManager);

	PxsContactManagerOutputIterator outputs = mLLContext->getNphaseImplementationContext()->getContactManagerOutputs();

	mNPhaseCore->fireCustomFilteringCallbacks(outputs);

	mNPhaseCore->preparePersistentContactEventListForNextFrame();

	mSimulationController->releaseDeferredArticulationIds();

#if PX_SUPPORT_GPU_PHYSX
	mSimulationController->releaseDeferredSoftBodyIds();

	mSimulationController->releaseDeferredFEMClothIds();

	mSimulationController->releaseDeferredParticleSystemIds();

	mAABBManager->releaseDeferredAggregateIds();
#endif

	// End step / update time stamps
	{
		mTimeStamp++;
	//  INVALID_SLEEP_COUNTER is 0xffffffff. Therefore the last bit is masked. Look at Body::isForcedToSleep() for example.
	//	if(timeStamp==PX_INVALID_U32)	timeStamp = 0;	// Reserve INVALID_ID for something else
		mTimeStamp &= 0x7fffffff;

		mReportShapePairTimeStamp++;  // to make sure that deleted shapes/actors after fetchResults() create new report pairs
	}

	PxcDisplayContactCacheStats();
}

void Sc::Scene::flush(bool sendPendingReports)
{
	if (sendPendingReports)
	{
		fireQueuedContactCallbacks();
		mNPhaseCore->clearContactReportStream();
		mNPhaseCore->clearContactReportActorPairs(true);

		fireTriggerCallbacks();
	}
	else
	{
		mNPhaseCore->clearContactReportActorPairs(true);  // To clear the actor pair set
	}
	postReportsCleanup();
	mNPhaseCore->freeContactReportStreamMemory();

	mTriggerBufferAPI.reset();
	mTriggerBufferExtraData->reset();

	mBrokenConstraints.reset();

	mBroadphaseManager.flush(mAABBManager);

	clearSleepWakeBodies();  //!!! If we send out these reports on flush then this would not be necessary

	mActorIDTracker->reset();
	mElementIDPool->reset();

	processLostTouchPairs();  // Processes the lost touch bodies
	PX_ASSERT(mLostTouchPairs.size() == 0);
	mLostTouchPairs.reset();
	// Does not seem worth deleting the bitmap for the lost touch pair list

	mActiveBodies.shrink();

	for(PxU32 i=0; i < InteractionType::eTRACKED_IN_SCENE_COUNT; i++)
	{
		mInteractions[i].shrink();
	}

	//!!! TODO: look into retrieving memory from the NPhaseCore & Broadphase class (all the pools in there etc.)

	mLLContext->getNpMemBlockPool().releaseUnusedBlocks();
}

// User callbacks

void Sc::Scene::setSimulationEventCallback(PxSimulationEventCallback* callback)
{
	if(!mSimulationEventCallback && callback)
	{
		// if there was no callback before, the sleeping bodies have to be prepared for potential notification events (no shortcut possible anymore)
		BodyCore* const* sleepingBodies = mSleepBodies.getEntries();
		for (PxU32 i = 0; i < mSleepBodies.size(); i++)
		{
			sleepingBodies[i]->getSim()->raiseInternalFlag(BodySim::BF_SLEEP_NOTIFY);
		}

#if PX_SUPPORT_GPU_PHYSX
		gpu_setSimulationEventCallback(callback);
#endif
	}

	mSimulationEventCallback = callback;
}

PxSimulationEventCallback* Sc::Scene::getSimulationEventCallback() const
{
	return mSimulationEventCallback;
}

void Sc::Scene::removeBody(BodySim& body)	//this also notifies any connected joints!
{
	BodyCore& core = body.getBodyCore();

	// Remove from sleepBodies array
	mSleepBodies.erase(&core);
	PX_ASSERT(!mSleepBodies.contains(&core));

	// Remove from wokeBodies array
	mWokeBodies.erase(&core);
	PX_ASSERT(!mWokeBodies.contains(&core));

	if (body.isActive() && (core.getFlags() & PxRigidBodyFlag::eENABLE_POSE_INTEGRATION_PREVIEW))
		removeFromPosePreviewList(body);
	else
		PX_ASSERT(!isInPosePreviewList(body));

	markReleasedBodyIDForLostTouch(body.getActorID());
}

void Sc::Scene::addConstraint(ConstraintCore& constraint, RigidCore* body0, RigidCore* body1)
{
	ConstraintSim* sim = mConstraintSimPool.construct(constraint, body0, body1, *this);

	addConstraintToMap(constraint, body0, body1);

	mConstraints.insert(&constraint);

	getSimulationController()->addJoint(sim->getLowLevelConstraint());
}

void Sc::Scene::removeConstraint(ConstraintCore& constraint)
{
	ConstraintSim* cSim = constraint.getSim();

	if (cSim)
	{
		removeConstraintFromMap(*cSim->getInteraction());

		mConstraintSimPool.destroy(cSim);
	}

	mConstraints.erase(&constraint);
}

void Sc::Scene::addConstraintToMap(ConstraintCore& constraint, RigidCore* body0, RigidCore* body1)
{
	PxNodeIndex nodeIndex0, nodeIndex1;

	ActorSim* sim0 = NULL;
	ActorSim* sim1 = NULL;

	if (body0)
	{
		sim0 = body0->getSim();
		nodeIndex0 = sim0->getNodeIndex();
	}
	if (body1)
	{
		sim1 = body1->getSim();
		nodeIndex1 = sim1->getNodeIndex();
	}

	if (nodeIndex1 < nodeIndex0)
		PxSwap(sim0, sim1);

	mConstraintMap.insert(PxPair<const Sc::ActorSim*, const Sc::ActorSim*>(sim0, sim1), &constraint);
}

void Sc::Scene::removeConstraintFromMap(const ConstraintInteraction& interaction)
{
	PxNodeIndex nodeIndex0, nodeIndex1;

	Sc::ActorSim* bSim = &interaction.getActorSim0();
	Sc::ActorSim* bSim1 = &interaction.getActorSim1();

	if (bSim)
		nodeIndex0 = bSim->getNodeIndex();
	if (bSim1)
		nodeIndex1 = bSim1->getNodeIndex();

	if (nodeIndex1 < nodeIndex0)
		PxSwap(bSim, bSim1);

	mConstraintMap.erase(PxPair<const Sc::ActorSim*, const Sc::ActorSim*>(bSim, bSim1));
}

void Sc::Scene::addArticulation(ArticulationCore& articulation, BodyCore& root)
{
	ArticulationSim* sim = PX_NEW(ArticulationSim)(articulation, *this, root);

	if(!sim)
		return;

	mArticulations.insert(&articulation);

	addDirtyArticulationSim(sim);
}

void Sc::Scene::removeArticulation(ArticulationCore& articulation)
{
	ArticulationSim* a = articulation.getSim();

	Sc::ArticulationSimDirtyFlags dirtyFlags = a->getDirtyFlag();
	const bool isDirty = (dirtyFlags & Sc::ArticulationSimDirtyFlag::eUPDATE);
	if(isDirty)
		removeDirtyArticulationSim(a);

	PX_DELETE(a);
	mArticulations.erase(&articulation);
}

void Sc::Scene::addArticulationJoint(ArticulationJointCore& joint, BodyCore& parent, BodyCore& child)
{
	ArticulationJointSim* sim = mArticulationJointSimPool.construct(joint, *parent.getSim(), *child.getSim());
	PX_UNUSED(sim);
}

void Sc::Scene::removeArticulationJoint(ArticulationJointCore& joint)
{
	ArticulationJointSim* sim = joint.getSim();
	mArticulationJointSimPool.destroy(sim);
}

void Sc::Scene::addArticulationTendon(ArticulationSpatialTendonCore& tendon)
{
	ArticulationSpatialTendonSim* sim = PX_NEW(ArticulationSpatialTendonSim)(tendon, *this);
	PX_UNUSED(sim);
}

void Sc::Scene::removeArticulationTendon(ArticulationSpatialTendonCore& tendon)
{
	ArticulationSpatialTendonSim* sim = tendon.getSim();
	PX_DELETE(sim);
}

void Sc::Scene::addArticulationTendon(ArticulationFixedTendonCore& tendon)
{
	ArticulationFixedTendonSim* sim = PX_NEW(ArticulationFixedTendonSim)(tendon, *this);

	PX_UNUSED(sim);
}

void Sc::Scene::removeArticulationTendon(ArticulationFixedTendonCore& tendon)
{
	ArticulationFixedTendonSim* sim = tendon.getSim();
	PX_DELETE(sim);
}

void Sc::Scene::addArticulationMimicJoint(ArticulationMimicJointCore& mimicJoint)
{
	//This might look like a forgotten allocation but it really isn't.
	//ArticulationMimicJointSim constructor does all the work here to make sure that
	//mimicJoint ends up maintaining a reference to sim.
	ArticulationMimicJointSim* sim = PX_NEW(ArticulationMimicJointSim)(mimicJoint, *this);
	PX_UNUSED(sim);
}

void Sc::Scene::removeArticulationMimicJoint(ArticulationMimicJointCore& mimicJoint)
{
	ArticulationMimicJointSim* sim = mimicJoint.getSim();
	PX_DELETE(sim);
}

void Sc::Scene::addArticulationSimControl(Sc::ArticulationCore& core)
{
	Sc::ArticulationSim* sim = core.getSim();
	if (sim)
		mSimulationController->addArticulation(sim, sim->getIslandNodeIndex());
}

void Sc::Scene::removeArticulationSimControl(Sc::ArticulationCore& core)
{
	Sc::ArticulationSim* sim = core.getSim();
	if (sim)
		mSimulationController->releaseArticulation(sim, sim->getIslandNodeIndex());
}

void* Sc::Scene::allocateConstraintBlock(PxU32 size)
{
	if(size<=128)
		return mMemBlock128Pool.construct();
	else if(size<=256)
		return mMemBlock256Pool.construct();
	else  if(size<=384)
		return mMemBlock384Pool.construct();
	else  if(size<=512)
		return mMemBlock512Pool.construct();
	else
		return PX_ALLOC(size, "ConstraintBlock");
}

void Sc::Scene::deallocateConstraintBlock(void* ptr, PxU32 size)
{
	if(size<=128)
		mMemBlock128Pool.destroy(reinterpret_cast<MemBlock128*>(ptr));
	else if(size<=256)
		mMemBlock256Pool.destroy(reinterpret_cast<MemBlock256*>(ptr));
	else  if(size<=384)
		mMemBlock384Pool.destroy(reinterpret_cast<MemBlock384*>(ptr));
	else  if(size<=512)
		mMemBlock512Pool.destroy(reinterpret_cast<MemBlock512*>(ptr));
	else
		PX_FREE(ptr);
}

void Sc::Scene::postReportsCleanup()
{
	mElementIDPool->processPendingReleases();
	mElementIDPool->clearDeletedIDMap();

	mActorIDTracker->processPendingReleases();
	mActorIDTracker->clearDeletedIDMap();

	mConstraintIDTracker->processPendingReleases();
	mConstraintIDTracker->clearDeletedIDMap();

#if PX_SUPPORT_GPU_PHYSX
	// AD: if we use either GPU BP or GPU dynamics.
	if (mHeapMemoryAllocationManager)
	{
		mHeapMemoryAllocationManager->flushDeferredDeallocs();		
	}
#endif
}

PX_COMPILE_TIME_ASSERT(sizeof(PxTransform32)==sizeof(PxsCachedTransform));

// PT: TODO: move this out of Sc? this is only called by Np
void Sc::Scene::syncSceneQueryBounds(SqBoundsSync& sync, SqRefFinder& finder)
{
	const PxsTransformCache& cache = mLLContext->getTransformCache();

	mSqBoundsManager->syncBounds(sync, finder, mBoundsArray->begin(), reinterpret_cast<const PxTransform32*>(cache.getTransforms()), mContextId, mDirtyShapeSimMap);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Sc::Scene::resizeReleasedBodyIDMaps(PxU32 maxActors, PxU32 numActors)
{ 
	mLostTouchPairsDeletedBodyIDs.resize(maxActors);
	mActorIDTracker->resizeDeletedIDMap(maxActors,numActors); 
	mElementIDPool->resizeDeletedIDMap(maxActors,numActors);
}

void Sc::Scene::finalizeContactStreamAndCreateHeader(PxContactPairHeader& header, const ActorPairReport& aPair, ContactStreamManager& cs, PxU32 removedShapeTestMask)
{
	PxU8* stream = mNPhaseCore->getContactReportPairData(cs.bufferIndex);
	PxU32 streamManagerFlag = cs.getFlags();
	ContactShapePair* contactPairs = cs.getShapePairs(stream);
	const PxU32 nbShapePairs = cs.currentPairCount;
	PX_ASSERT(nbShapePairs > 0);

	if (streamManagerFlag & removedShapeTestMask)
	{
		// At least one shape of this actor pair has been deleted. Need to traverse the contact buffer,
		// find the pairs which contain deleted shapes and set the flags accordingly.

		ContactStreamManager::convertDeletedShapesInContactStream(contactPairs, nbShapePairs, getElementIDPool());
	}
	PX_ASSERT(contactPairs);

	ObjectIDTracker& ActorIDTracker = getActorIDTracker();
	header.actors[0] = aPair.getPxActorA();
	header.actors[1] = aPair.getPxActorB();
	PxU16 headerFlags = 0;
	if (ActorIDTracker.isDeletedID(aPair.getActorAID()))
		headerFlags |= PxContactPairHeaderFlag::eREMOVED_ACTOR_0;
	if (ActorIDTracker.isDeletedID(aPair.getActorBID()))
		headerFlags |= PxContactPairHeaderFlag::eREMOVED_ACTOR_1;
	header.flags = PxContactPairHeaderFlags(headerFlags);
	header.pairs = reinterpret_cast<PxContactPair*>(contactPairs);
	header.nbPairs = nbShapePairs;

	PxU16 extraDataSize = cs.extraDataSize;
	if (!extraDataSize)
		header.extraDataStream = NULL;
	else
	{
		PX_ASSERT(extraDataSize >= sizeof(ContactStreamHeader));
		extraDataSize -= sizeof(ContactStreamHeader);
		header.extraDataStream = stream + sizeof(ContactStreamHeader);

		if (streamManagerFlag & ContactStreamManagerFlag::eNEEDS_POST_SOLVER_VELOCITY)
		{
			PX_ASSERT(!(headerFlags & PxTo16(PxContactPairHeaderFlag::eREMOVED_ACTOR_0 | PxContactPairHeaderFlag::eREMOVED_ACTOR_1)));
			cs.setContactReportPostSolverVelocity(stream, aPair.getActorA(), aPair.getActorB());
		}
	}
	header.extraDataStreamSize = extraDataSize;
}

const PxArray<PxContactPairHeader>& Sc::Scene::getQueuedContactPairHeaders()
{
	const PxU32 removedShapeTestMask = PxU32(ContactStreamManagerFlag::eTEST_FOR_REMOVED_SHAPES);

	ActorPairReport*const* actorPairs = mNPhaseCore->getContactReportActorPairs();
	PxU32 nbActorPairs = mNPhaseCore->getNbContactReportActorPairs();
	mQueuedContactPairHeaders.reserve(nbActorPairs);
	mQueuedContactPairHeaders.clear();

	for (PxU32 i = 0; i < nbActorPairs; i++)
	{
		if (i < (nbActorPairs - 1))
			PxPrefetchLine(actorPairs[i + 1]);

		ActorPairReport* aPair = actorPairs[i];
		ContactStreamManager& cs = aPair->getContactStreamManager();
		if (cs.getFlags() & ContactStreamManagerFlag::eINVALID_STREAM)
			continue;

		if (i + 1 < nbActorPairs)
			PxPrefetch(&(actorPairs[i + 1]->getContactStreamManager()));

		PxContactPairHeader& pairHeader = *mQueuedContactPairHeaders.insert();
		finalizeContactStreamAndCreateHeader(pairHeader, *aPair, cs, removedShapeTestMask);

		cs.maxPairCount = cs.currentPairCount;
		cs.setMaxExtraDataSize(cs.extraDataSize);
	}

	return mQueuedContactPairHeaders;
}

/*
Threading: called in the context of the user thread, but only after the physics thread has finished its run
*/
void Sc::Scene::fireQueuedContactCallbacks()
{
	if(mSimulationEventCallback)
	{
		const PxU32 removedShapeTestMask = PxU32(ContactStreamManagerFlag::eTEST_FOR_REMOVED_SHAPES);

		ActorPairReport*const* actorPairs = mNPhaseCore->getContactReportActorPairs();
		PxU32 nbActorPairs = mNPhaseCore->getNbContactReportActorPairs();
		for(PxU32 i=0; i < nbActorPairs; i++)
		{
			if (i < (nbActorPairs - 1))
				PxPrefetchLine(actorPairs[i+1]);

			ActorPairReport* aPair = actorPairs[i];
			ContactStreamManager* cs = &aPair->getContactStreamManager();
			if (cs == NULL || cs->getFlags() & ContactStreamManagerFlag::eINVALID_STREAM)
				continue;
			
			if (i + 1 < nbActorPairs)
				PxPrefetch(&(actorPairs[i+1]->getContactStreamManager()));

			PxContactPairHeader pairHeader;
			finalizeContactStreamAndCreateHeader(pairHeader, *aPair, *cs, removedShapeTestMask);

			{
				PX_PROFILE_ZONE("USERCODE - PxSimulationEventCallback::onContact", mContextId);
				mSimulationEventCallback->onContact(pairHeader, pairHeader.pairs, pairHeader.nbPairs);
			}

			// estimates for next frame
			cs->maxPairCount = cs->currentPairCount;
			cs->setMaxExtraDataSize(cs->extraDataSize);
		}
	}
}

PX_FORCE_INLINE void markDeletedShapes(Sc::ObjectIDTracker& idTracker, Sc::TriggerPairExtraData& tped, PxTriggerPair& pair)
{
	PxTriggerPairFlags::InternalType flags = 0;
	if (idTracker.isDeletedID(tped.shape0ID))
		flags |= PxTriggerPairFlag::eREMOVED_SHAPE_TRIGGER;
	if (idTracker.isDeletedID(tped.shape1ID))
		flags |= PxTriggerPairFlag::eREMOVED_SHAPE_OTHER;

	pair.flags = PxTriggerPairFlags(flags);
}

void Sc::Scene::fireTriggerCallbacks()
{
	// triggers
	const PxU32 nbTriggerPairs = mTriggerBufferAPI.size();
	PX_ASSERT(nbTriggerPairs == mTriggerBufferExtraData->size());
	if(nbTriggerPairs) 
	{
		// cases to take into account:
		// - no simulation/trigger shape has been removed -> no need to test shape references for removed shapes
		// - simulation/trigger shapes have been removed  -> test the events that have 
		//   a marker for removed shapes set
		//
		const bool hasRemovedShapes = mElementIDPool->getDeletedIDCount() > 0;

		if(mSimulationEventCallback)
		{
			if (hasRemovedShapes)
			{
				for(PxU32 i = 0; i < nbTriggerPairs; i++)
				{
					PxTriggerPair& triggerPair = mTriggerBufferAPI[i];

					if ((PxTriggerPairFlags::InternalType(triggerPair.flags) & TriggerPairFlag::eTEST_FOR_REMOVED_SHAPES))
						markDeletedShapes(*mElementIDPool, (*mTriggerBufferExtraData)[i], triggerPair);
				}
			}

			{
				PX_PROFILE_ZONE("USERCODE - PxSimulationEventCallback::onTrigger", mContextId);
				mSimulationEventCallback->onTrigger(mTriggerBufferAPI.begin(), nbTriggerPairs);
			}
		}
	}

	// PT: clear the buffer **even when there's no simulationEventCallback**.
	mTriggerBufferAPI.clear();
	mTriggerBufferExtraData->clear();
}

/*
Threading: called in the context of the user thread, but only after the physics thread has finished its run
*/
void Sc::Scene::fireCallbacksPostSync()
{
	//
	// Fire sleep & woken callbacks
	//

	// A body should be either in the sleep or the woken list. If it is in both, remove it from the list it was
	// least recently added to.

	if(!mSleepBodyListValid)
		cleanUpSleepBodies();

	if(!mWokeBodyListValid)
		cleanUpWokenBodies();

#if PX_SUPPORT_GPU_PHYSX
	const PxU32 maxGpuSizeNeeded = gpu_cleanUpSleepAndWokenBodies();
#endif

	if(mSimulationEventCallback || mOnSleepingStateChanged)
	{
		// allocate temporary data
		const PxU32 nbSleep = mSleepBodies.size();
		const PxU32 nbWoken = mWokeBodies.size();
#if PX_SUPPORT_GPU_PHYSX
		const PxU32 arrSize = PxMax(PxMax(nbSleep, nbWoken), maxGpuSizeNeeded);
#else
		const PxU32 arrSize = PxMax(nbSleep, nbWoken);
#endif
		PxActor** actors = arrSize ? reinterpret_cast<PxActor**>(PX_ALLOC(arrSize*sizeof(PxActor*), "PxActor*")) : NULL;
		if(actors)
		{
			if(nbSleep)
			{
				PxU32 destSlot = 0;
				BodyCore* const* sleepingBodies = mSleepBodies.getEntries();
				for(PxU32 i=0; i<nbSleep; i++)
				{
					BodyCore* body = sleepingBodies[i];
					if (body->getActorFlags() & PxActorFlag::eSEND_SLEEP_NOTIFIES)
						actors[destSlot++] = body->getPxActor();
					if (mOnSleepingStateChanged)
						mOnSleepingStateChanged(*static_cast<PxRigidDynamic*>(body->getPxActor()), true);
				}

				if(destSlot && mSimulationEventCallback)
				{
					PX_PROFILE_ZONE("USERCODE - PxSimulationEventCallback::onSleep", mContextId);
					mSimulationEventCallback->onSleep(actors, destSlot);
				}

				//if (PX_DBG_IS_CONNECTED())
				//{
				//	for (PxU32 i = 0; i < nbSleep; ++i)
				//	{
				//		BodyCore* body = mSleepBodies[i];
				//		PX_ASSERT(body->getActorType() == PxActorType::eRIGID_DYNAMIC);
				//	}
				//}
			}

#if PX_SUPPORT_GPU_PHYSX
			gpu_fireOnSleepCallback(actors);
#endif

			// do the same thing for bodies that have just woken up

			if(nbWoken)
			{
				PxU32 destSlot = 0;
				BodyCore* const* wokenBodies = mWokeBodies.getEntries();
				for(PxU32 i=0; i<nbWoken; i++)
				{
					BodyCore* body = wokenBodies[i];
					if(body->getActorFlags() & PxActorFlag::eSEND_SLEEP_NOTIFIES)
						actors[destSlot++] = body->getPxActor();
					if (mOnSleepingStateChanged)
						mOnSleepingStateChanged(*static_cast<PxRigidDynamic*>(body->getPxActor()), false);
				}

				if(destSlot && mSimulationEventCallback)
				{
					PX_PROFILE_ZONE("USERCODE - PxSimulationEventCallback::onWake", mContextId);
					mSimulationEventCallback->onWake(actors, destSlot);
				}
			}

#if PX_SUPPORT_GPU_PHYSX
			gpu_fireOnWakeCallback(actors);
#endif
			PX_FREE(actors);
		}
	}

	clearSleepWakeBodies();
}

void Sc::Scene::postCallbacksPreSync()
{
	PX_PROFILE_ZONE("Sim.postCallbackPreSync", mContextId);
	// clear contact stream data
	mNPhaseCore->clearContactReportStream();
	mNPhaseCore->clearContactReportActorPairs(false);

	postCallbacksPreSyncKinematics();

	releaseConstraints(true); //release constraint blocks at the end of the frame, so user can retrieve the blocks
}

void Sc::Scene::getStats(PxSimulationStatistics& s) const
{
	mStats->readOut(s, mLLContext->getSimStats());
	getAvbdCpuSoftBodyStatistics(s);
	s.nbStaticBodies = mNbRigidStatics;
	s.nbDynamicBodies = mNbRigidDynamics;
	s.nbKinematicBodies = mNbRigidKinematic;
	s.nbArticulations = mArticulations.size(); 

	s.nbAggregates = mAABBManager->getNbActiveAggregates();
	for(PxU32 i=0; i<PxGeometryType::eGEOMETRY_COUNT; i++)
		s.nbShapes[i] = mNbGeometries[i];

#if PX_SUPPORT_GPU_PHYSX
	if (mHeapMemoryAllocationManager)
	{
		s.gpuMemHeap = mHeapMemoryAllocationManager->getDeviceMemorySize();

		const PxsHeapStats& deviceHeapStats = mHeapMemoryAllocationManager->getDeviceHeapStats();
		s.gpuMemHeapBroadPhase = deviceHeapStats.stats[PxsHeapStats::eBROADPHASE];
		s.gpuMemHeapNarrowPhase = deviceHeapStats.stats[PxsHeapStats::eNARROWPHASE];
		s.gpuMemHeapSolver = deviceHeapStats.stats[PxsHeapStats::eSOLVER];
		s.gpuMemHeapArticulation = deviceHeapStats.stats[PxsHeapStats::eARTICULATION];
		s.gpuMemHeapSimulation = deviceHeapStats.stats[PxsHeapStats::eSIMULATION];
		s.gpuMemHeapSimulationArticulation = deviceHeapStats.stats[PxsHeapStats::eSIMULATION_ARTICULATION];
		s.gpuMemHeapSimulationParticles = deviceHeapStats.stats[PxsHeapStats::eSIMULATION_PARTICLES];
		s.gpuMemHeapSimulationDeformableSurface = deviceHeapStats.stats[PxsHeapStats::eSIMULATION_FEMCLOTH];
		s.gpuMemHeapSimulationDeformableVolume = deviceHeapStats.stats[PxsHeapStats::eSIMULATION_SOFTBODY];
		s.gpuMemHeapParticles = deviceHeapStats.stats[PxsHeapStats::eSHARED_PARTICLES];
		s.gpuMemHeapDeformableSurfaces = deviceHeapStats.stats[PxsHeapStats::eSHARED_FEMCLOTH];
		s.gpuMemHeapDeformableVolumes = deviceHeapStats.stats[PxsHeapStats::eSHARED_SOFTBODY];
		s.gpuMemHeapOther = deviceHeapStats.stats[PxsHeapStats::eOTHER];
	}
	else
#endif
	{
		s.gpuMemHeap = 0;
		s.gpuMemParticles = 0;
		s.gpuMemDeformableSurfaces = 0;
		s.gpuMemDeformableVolumes = 0;
		s.gpuMemHeapBroadPhase = 0;
		s.gpuMemHeapNarrowPhase = 0;
		s.gpuMemHeapSolver = 0;
		s.gpuMemHeapArticulation = 0;
		s.gpuMemHeapSimulation = 0;
		s.gpuMemHeapSimulationArticulation = 0;
		s.gpuMemHeapSimulationParticles = 0;
		s.gpuMemHeapSimulationDeformableSurface = 0;
		s.gpuMemHeapSimulationDeformableVolume = 0;
		s.gpuMemHeapParticles = 0;
		s.gpuMemHeapDeformableSurfaces = 0;
		s.gpuMemHeapOther = 0;
	}
}

void Sc::Scene::getAvbdCpuSoftBodyStatistics(
	PxSimulationStatistics& stats) const
{
	if(mAvbdCpuSoftScene)
		mAvbdCpuSoftScene->writeAvbdCpuSoftBodyStatistics(stats);
	if(mDynamicsContext &&
		getSolverType() == PxSolverType::eAVBD &&
		!isUsingGpuDynamics())
	{
		const Dy::AvbdCpuIsaDispatch& cpuIsa = Dy::getAvbdCpuIsaDispatch();
		stats.avbdCpuIsaRequested = getAvbdCpuIsaModeCode(cpuIsa.requestedIsa);
		stats.avbdCpuIsaSelected = getAvbdCpuIsaModeCode(cpuIsa.selectedIsa);
		stats.avbdCpuIsaCompiledBackendMask = 1u |
			(cpuIsa.avx2FmaBackendCompiled ? 2u : 0u);
		stats.avbdCpuIsaCapabilityMask =
			getAvbdCpuIsaCapabilityMask(cpuIsa.capabilities);
		stats.avbdCpuIsaForceModeRejected = cpuIsa.forceModeRejected ? 1u : 0u;
		stats.avbdCpuIsaKernelSelfTestPassed =
			cpuIsa.kernelSelfTestPassed ? 1u : 0u;
		stats.avbdCpuIsaFmaUsed = cpuIsa.fmaUsed ? 1u : 0u;
		stats.avbdCpuIsaKernelSelfTestValue = cpuIsa.kernelSelfTestValue;
		// Taskgraph telemetry is maintained at the Scene task boundary so the
		// solver hot path remains free of shared profiling state.
	}
}

void Sc::Scene::addShapes(NpShape *const* shapes, PxU32 nbShapes, size_t ptrOffset, RigidSim& bodySim, PxBounds3* outBounds)
{
	const PxNodeIndex nodeIndex = bodySim.getNodeIndex();
	
	PxvNphaseImplementationContext* context = mLLContext->getNphaseImplementationContext();

	for(PxU32 i=0;i<nbShapes;i++)
	{
		// PT: TODO: drop the offsets and let me include NpShape.h from here! This is just NpShape::getCore()....
		ShapeCore& sc = *reinterpret_cast<ShapeCore*>(size_t(shapes[i])+ptrOffset);

		//PxBounds3* target = uninflatedBounds ? uninflatedBounds + i : uninflatedBounds;
		//mShapeSimPool->construct(sim, sc, llBody, target);

		ShapeSim* shapeSim = mShapeSimPool->construct(bodySim, sc);
		mNbGeometries[sc.getGeometryType()]++;

		mSimulationController->addPxgShape(shapeSim, shapeSim->getPxsShapeCore(), shapeSim->getActorNodeIndex(), shapeSim->getElementID());

		if(outBounds)
		{
			PxU32 elementID = shapeSim->getElementID();
#if PX_SUPPORT_GPU_PHYSX
			outBounds[i] = mBoundsArray->hadAllocationFailure() ? PxBounds3::empty() : mBoundsArray->getBounds(elementID);
#else
			outBounds[i] = mBoundsArray->getBounds(elementID);
#endif
		}
		
		//I register the shape if its either not an articulation link or if the nodeIndex has already been
		//assigned. On insertion, the articulation will not have this nodeIndex correctly assigned at this stage
		if (bodySim.getActorType() != PxActorType::eARTICULATION_LINK || !nodeIndex.isStaticBody())
			context->registerShape(nodeIndex, sc, shapeSim->getElementID(), bodySim.getPxActor());
		if(bodySim.getActorType() == PxActorType::eRIGID_STATIC)
			addAvbdCpuStaticShape(
				static_cast<StaticCore&>(bodySim.getRigidCore()), sc);
		else if(bodySim.getActorType() ==
			PxActorType::eRIGID_DYNAMIC)
			addAvbdCpuDynamicShape(
				static_cast<BodyCore&>(bodySim.getRigidCore()), sc);
	}
}

void Sc::Scene::removeShapes(Sc::RigidSim& sim, PxInlineArray<Sc::ShapeSim*, 64>& shapesBuffer , PxInlineArray<const Sc::ShapeCore*,64>& removedShapes, bool wakeOnLostTouch)
{
	PxU32 nbElems = sim.getNbElements();
	Sc::ElementSim** elems = sim.getElements();
	while (nbElems--)
	{
		Sc::ShapeSim* s = static_cast<Sc::ShapeSim*>(*elems++);
		// can do two 2x the allocs in the worst case, but actors with >64 shapes are not common
		shapesBuffer.pushBack(s);
		removedShapes.pushBack(&s->getCore());
	}

	for(PxU32 i=0;i<shapesBuffer.size();i++)
		removeShape_(*shapesBuffer[i], wakeOnLostTouch);
}

void Sc::Scene::addStatic(StaticCore& ro, NpShape*const *shapes, PxU32 nbShapes, size_t shapePtrOffset, PxBounds3* uninflatedBounds)
{
	PX_ASSERT(ro.getActorCoreType() == PxActorType::eRIGID_STATIC);

	// sim objects do all the necessary work of adding themselves to broad phase,
	// activation, registering with the interaction system, etc

	StaticSim* sim = mStaticSimPool->construct(*this, ro);
	
	mNbRigidStatics++;
	addShapes(shapes, nbShapes, shapePtrOffset, *sim, uninflatedBounds);
}

void Sc::Scene::removeStatic(StaticCore& ro, PxInlineArray<const Sc::ShapeCore*,64>& removedShapes, bool wakeOnLostTouch)
{
	PX_ASSERT(ro.getActorCoreType() == PxActorType::eRIGID_STATIC);

	StaticSim* sim = ro.getSim();
	if(sim)
	{
		if(mBatchRemoveState)
		{
			removeShapes(*sim, mBatchRemoveState->bufferedShapes ,removedShapes, wakeOnLostTouch);
		}
		else
		{
			PxInlineArray<Sc::ShapeSim*, 64>  shapesBuffer;
			removeShapes(*sim, shapesBuffer ,removedShapes, wakeOnLostTouch);
		}
		mStaticSimPool->destroy(static_cast<Sc::StaticSim*>(ro.getSim()));
		mNbRigidStatics--;
	}
}

void Sc::Scene::addBody(BodyCore& body, NpShape*const *shapes, PxU32 nbShapes, size_t shapePtrOffset, PxBounds3* outBounds, bool compound)
{
	// sim objects do all the necessary work of adding themselves to broad phase,
	// activation, registering with the interaction system, etc

	BodySim* sim = mBodySimPool->construct(*this, body, compound);

	const bool isArticulationLink = sim->isArticulationLink();

	if (sim->getLowLevelBody().mCore->mFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD && sim->isActive())
	{
		if (isArticulationLink)
		{
			if (sim->getNodeIndex().isValid())
				mSpeculativeCDDArticulationBitMap.growAndSet(sim->getNodeIndex().index());
		}
		else
			mSpeculativeCCDRigidBodyBitMap.growAndSet(sim->getNodeIndex().index());
	}
	//if rigid body is articulation link, the node index will be invalid. We should add the link to the scene after we add the
	//articulation for gpu
	if(sim->getNodeIndex().isValid())
		mSimulationController->addDynamic(&sim->getLowLevelBody(), sim->getNodeIndex());
	if(sim->getSimStateData(true) && sim->getSimStateData(true)->isKine())
		mNbRigidKinematic++;
	else
		mNbRigidDynamics++;
	addShapes(shapes, nbShapes, shapePtrOffset, *sim, outBounds);

	mDynamicsContext->setStateDirty(true);
}

void Sc::Scene::removeBody(BodyCore& body, PxInlineArray<const Sc::ShapeCore*,64>& removedShapes, bool wakeOnLostTouch)
{
	BodySim *sim = body.getSim();	
	if(sim)
	{
		removeAvbdCpuDynamic(body);
		if(mBatchRemoveState)
		{
			removeShapes(*sim, mBatchRemoveState->bufferedShapes, removedShapes, wakeOnLostTouch);
		}
		else
		{
			PxInlineArray<Sc::ShapeSim*, 64>  shapesBuffer;
			removeShapes(*sim, shapesBuffer, removedShapes, wakeOnLostTouch);
		}

		if(!sim->isArticulationLink())
		{
			//clear bit map
			if(sim->getLowLevelBody().mCore->mFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD)
				sim->getScene().resetSpeculativeCCDRigidBody(sim->getNodeIndex().index());				
		}
		else
		{
			// PT: TODO: missing call to resetSpeculativeCCDArticulationLink ? 

			sim->getArticulation()->removeBody(*sim);
		}
		if(sim->getSimStateData(true) && sim->getSimStateData(true)->isKine())
		{
			body.onRemoveKinematicFromScene();

			mNbRigidKinematic--;
		}
		else
			mNbRigidDynamics--;

		mBodySimPool->destroy(sim);

		mDynamicsContext->setStateDirty(true);
	}
}

// PT: TODO: refactor with addShapes
void Sc::Scene::addShape_(RigidSim& owner, ShapeCore& shapeCore)
{
	ShapeSim* sim = mShapeSimPool->construct(owner, shapeCore);
	mNbGeometries[shapeCore.getGeometryType()]++;

	//register shape
	mSimulationController->addPxgShape(sim, sim->getPxsShapeCore(), sim->getActorNodeIndex(), sim->getElementID());

	registerShapeInNphase(&owner.getRigidCore(), shapeCore, sim->getElementID());
	if(owner.getActorType() == PxActorType::eRIGID_STATIC)
		addAvbdCpuStaticShape(
			static_cast<StaticCore&>(owner.getRigidCore()), shapeCore);
	else if(owner.getActorType() == PxActorType::eRIGID_DYNAMIC)
		addAvbdCpuDynamicShape(
			static_cast<BodyCore&>(owner.getRigidCore()), shapeCore);
}

// PT: TODO: refactor with removeShapes
void Sc::Scene::removeShape_(ShapeSim& shape, bool wakeOnLostTouch)
{
	//BodySim* body = shape.getBodySim();
	//if(body)
	//	body->postShapeDetach();

	RigidSim& owner = shape.getRbSim();
	if(owner.getActorType() == PxActorType::eRIGID_STATIC)
		removeAvbdCpuStaticShape(
			static_cast<StaticCore&>(owner.getRigidCore()),
			shape.getCore());
	else if(owner.getActorType() == PxActorType::eRIGID_DYNAMIC)
		removeAvbdCpuDynamicShape(
			static_cast<BodyCore&>(owner.getRigidCore()),
			shape.getCore());
	
	unregisterShapeFromNphase(shape.getCore(), shape.getElementID());

	mSimulationController->removePxgShape(shape.getElementID());

	mNbGeometries[shape.getCore().getGeometryType()]--;
	shape.removeFromBroadPhase(wakeOnLostTouch);
	mShapeSimPool->destroy(&shape);
}

void Sc::Scene::registerShapeInNphase(Sc::RigidCore* rigidCore, const ShapeCore& shape, const PxU32 transformCacheID)
{
	RigidSim* sim = rigidCore->getSim();
	if(sim)
		mLLContext->getNphaseImplementationContext()->registerShape(sim->getNodeIndex(), shape, transformCacheID, sim->getPxActor());
}

void Sc::Scene::unregisterShapeFromNphase(const ShapeCore& shape, const PxU32 transformCacheID)
{
	mLLContext->getNphaseImplementationContext()->unregisterShape(shape, transformCacheID);
}

void Sc::Scene::notifyNphaseOnUpdateShapeMaterial(const ShapeCore& shapeCore)
{
	mLLContext->getNphaseImplementationContext()->updateShapeMaterial(shapeCore);
}

void Sc::Scene::startBatchInsertion(BatchInsertionState&state)
{
	state.shapeSim = mShapeSimPool->allocateAndPrefetch();
	state.staticSim = mStaticSimPool->allocateAndPrefetch();
	state.bodySim = mBodySimPool->allocateAndPrefetch();														   
}

void Sc::Scene::addShapes(NpShape*const* shapes, PxU32 nbShapes, size_t ptrOffset, RigidSim& rigidSim, ShapeSim*& prefetchedShapeSim, PxBounds3* outBounds)
{
	for(PxU32 i=0;i<nbShapes;i++)
	{
		if(i+1<nbShapes)
			PxPrefetch(shapes[i+1], PxU32(ptrOffset+sizeof(Sc::ShapeCore)));
		ShapeSim* nextShapeSim = mShapeSimPool->allocateAndPrefetch();
		ShapeCore& sc = *PxPointerOffset<ShapeCore*>(shapes[i], ptrdiff_t(ptrOffset));
		PX_PLACEMENT_NEW(prefetchedShapeSim, ShapeSim(rigidSim, sc));
		const PxU32 elementID = prefetchedShapeSim->getElementID();

#if PX_SUPPORT_GPU_PHYSX
		outBounds[i] = mBoundsArray->hadAllocationFailure() ? PxBounds3::empty() : mBoundsArray->getBounds(elementID);
#else
		outBounds[i] = mBoundsArray->getBounds(elementID);
#endif

		// PT: TODO: revisit getActorNodeIndex() vs rigidSim.getNodeIndex()
		mSimulationController->addPxgShape(prefetchedShapeSim, prefetchedShapeSim->getPxsShapeCore(), prefetchedShapeSim->getActorNodeIndex(), elementID);
		mLLContext->getNphaseImplementationContext()->registerShape(rigidSim.getNodeIndex(), sc, elementID, rigidSim.getPxActor());
		if(rigidSim.getActorType() == PxActorType::eRIGID_STATIC)
			addAvbdCpuStaticShape(
				static_cast<StaticCore&>(rigidSim.getRigidCore()), sc);
		else if(rigidSim.getActorType() ==
			PxActorType::eRIGID_DYNAMIC)
			addAvbdCpuDynamicShape(
				static_cast<BodyCore&>(rigidSim.getRigidCore()), sc);

		prefetchedShapeSim = nextShapeSim;
		mNbGeometries[sc.getGeometryType()]++;
	}
}

void Sc::Scene::addStatic(PxActor* actor, BatchInsertionState& s, PxBounds3* outBounds)
{
	// static core has been prefetched by caller
	Sc::StaticSim* sim = s.staticSim;		// static core has been prefetched by the caller

	const Cm::PtrTable* shapeTable = PxPointerOffset<const Cm::PtrTable*>(actor, s.staticShapeTableOffset);
	void*const* shapes = shapeTable->getPtrs();

	mStaticSimPool->construct(sim, *this, *PxPointerOffset<Sc::StaticCore*>(actor, s.staticActorOffset));
	s.staticSim = mStaticSimPool->allocateAndPrefetch();

	addShapes(reinterpret_cast<NpShape*const*>(shapes), shapeTable->getCount(), size_t(s.shapeOffset), *sim, s.shapeSim, outBounds);
	mNbRigidStatics++;
}

void Sc::Scene::addBody(PxActor* actor, BatchInsertionState& s, PxBounds3* outBounds, bool compound)
{
	Sc::BodySim* sim = s.bodySim;		// body core has been prefetched by the caller

	const Cm::PtrTable* shapeTable = PxPointerOffset<const Cm::PtrTable*>(actor, s.dynamicShapeTableOffset);
	void*const* shapes = shapeTable->getPtrs();

	Sc::BodyCore* bodyCore = PxPointerOffset<Sc::BodyCore*>(actor, s.dynamicActorOffset);
	mBodySimPool->construct(sim, *this, *bodyCore, compound);	
	s.bodySim = mBodySimPool->allocateAndPrefetch();

	if(sim->getLowLevelBody().mCore->mFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD)
	{
		if(sim->isArticulationLink())
			mSpeculativeCDDArticulationBitMap.growAndSet(sim->getNodeIndex().index());
		else
			mSpeculativeCCDRigidBodyBitMap.growAndSet(sim->getNodeIndex().index());
	}

	if(sim->getNodeIndex().isValid())
		mSimulationController->addDynamic(&sim->getLowLevelBody(), sim->getNodeIndex());

	addShapes(reinterpret_cast<NpShape*const*>(shapes), shapeTable->getCount(), size_t(s.shapeOffset), *sim, s.shapeSim, outBounds);

	const SimStateData* simStateData = bodyCore->getSim()->getSimStateData(true);
	if(simStateData && simStateData->isKine())
		mNbRigidKinematic++;
	else
		mNbRigidDynamics++;

	mDynamicsContext->setStateDirty(true);
}

void Sc::Scene::finishBatchInsertion(BatchInsertionState& state)
{
	// a little bit lazy - we could deal with the last one in the batch specially to avoid overallocating by one.
	
	mStaticSimPool->releasePreallocated(static_cast<Sc::StaticSim*>(state.staticSim));	
	mBodySimPool->releasePreallocated(static_cast<Sc::BodySim*>(state.bodySim));
	mShapeSimPool->releasePreallocated(state.shapeSim);
}

// PT: TODO: why is this in Sc?
void Sc::Scene::initContactsIterator(ContactIterator& contactIterator, PxsContactManagerOutputIterator& outputs)
{
	outputs = mLLContext->getNphaseImplementationContext()->getContactManagerOutputs();
	ElementSimInteraction** first = mInteractions[Sc::InteractionType::eOVERLAP].begin();
	contactIterator = ContactIterator(first, first + mActiveInteractionCount[Sc::InteractionType::eOVERLAP], outputs);
}

void Sc::Scene::setDominanceGroupPair(PxDominanceGroup group1, PxDominanceGroup group2, const PxDominanceGroupPair& dominance)
{
	struct {
		void operator()(PxU32& bits, PxDominanceGroup shift, PxReal weight)
		{
			if(weight != PxReal(0))
				bits |=  (PxU32(1) << shift);
			else 
				bits &= ~(PxU32(1) << shift);
		}
	} bitsetter;

	bitsetter(mDominanceBitMatrix[group1], group2, dominance.dominance0);
	bitsetter(mDominanceBitMatrix[group2], group1, dominance.dominance1);

	mInternalFlags |= SceneInternalFlag::eSCENE_SIP_STATES_DIRTY_DOMINANCE;		//force an update on all interactions on matrix change -- very expensive but we have no choice!!
}

PxDominanceGroupPair Sc::Scene::getDominanceGroupPair(PxDominanceGroup group1, PxDominanceGroup group2) const
{
	const PxU8 dom0 = PxU8((mDominanceBitMatrix[group1]>>group2) & 0x1 ? 1u : 0u);
	const PxU8 dom1 = PxU8((mDominanceBitMatrix[group2]>>group1) & 0x1 ? 1u : 0u);
	return PxDominanceGroupPair(dom0, dom1);
}

PxU32 Sc::Scene::getDefaultContactReportStreamBufferSize() const
{
	return mNPhaseCore->getDefaultContactReportStreamBufferSize();
}

void Sc::Scene::buildActiveActors()
{
	{
		PxU32 numActiveBodies = 0;
		BodyCore*const* PX_RESTRICT activeBodies;
		if (!(getFlags() & PxSceneFlag::eEXCLUDE_KINEMATICS_FROM_ACTIVE_ACTORS))
		{
			numActiveBodies = getNumActiveBodies();
			activeBodies = getActiveBodiesArray();
		}
		else
		{
			numActiveBodies = getActiveDynamicBodiesCount();
			activeBodies = getActiveDynamicBodies();
		}

		mActiveActors.clear();

		for (PxU32 i = 0; i < numActiveBodies; i++)
		{
			if (!activeBodies[i]->isFrozen())
			{
				PxRigidActor* ra = static_cast<PxRigidActor*>(activeBodies[i]->getPxActor());
				PX_ASSERT(ra);
				mActiveActors.pushBack(ra);
			}
		}
	}

#if PX_SUPPORT_GPU_PHYSX
	gpu_buildActiveActors();
#endif
}

// PT: TODO: unify buildActiveActors & buildActiveAndFrozenActors
void Sc::Scene::buildActiveAndFrozenActors()
{
	{
		PxU32 numActiveBodies = 0;
		BodyCore*const* PX_RESTRICT activeBodies;
		if (!(getFlags() & PxSceneFlag::eEXCLUDE_KINEMATICS_FROM_ACTIVE_ACTORS))
		{
			numActiveBodies = getNumActiveBodies();
			activeBodies = getActiveBodiesArray();
		}
		else
		{
			numActiveBodies = getActiveDynamicBodiesCount();
			activeBodies = getActiveDynamicBodies();
		}

		mActiveActors.clear();
		mFrozenActors.clear();

		for (PxU32 i = 0; i < numActiveBodies; i++)
		{
			PxRigidActor* ra = static_cast<PxRigidActor*>(activeBodies[i]->getPxActor());
			PX_ASSERT(ra);

			if (!activeBodies[i]->isFrozen())
				mActiveActors.pushBack(ra);
			else
				mFrozenActors.pushBack(ra);
		}
	}

#if PX_SUPPORT_GPU_PHYSX
	gpu_buildActiveAndFrozenActors();
#endif
}

PxActor** Sc::Scene::getActiveActors(PxU32& nbActorsOut)
{
	nbActorsOut = mActiveActors.size();
	
	if(!nbActorsOut)
		return NULL;

	return mActiveActors.begin();
}

void Sc::Scene::setActiveActors(PxActor** actors, PxU32 nbActors)
{
	mActiveActors.forceSize_Unsafe(0);
	mActiveActors.resize(nbActors);
	PxMemCopy(mActiveActors.begin(), actors, sizeof(PxActor*) * nbActors);
}

PxActor** Sc::Scene::getFrozenActors(PxU32& nbActorsOut)
{
	nbActorsOut = mFrozenActors.size();

	if(!nbActorsOut)
		return NULL;

	return mFrozenActors.begin();
}

void Sc::Scene::reserveTriggerReportBufferSpace(const PxU32 pairCount, PxTriggerPair*& triggerPairBuffer, TriggerPairExtraData*& triggerPairExtraBuffer)
{
	const PxU32 oldSize = mTriggerBufferAPI.size();
	const PxU32 newSize = oldSize + pairCount;
	const PxU32 newCapacity = PxU32(newSize * 1.5f);
	mTriggerBufferAPI.reserve(newCapacity);
	mTriggerBufferAPI.forceSize_Unsafe(newSize);
	triggerPairBuffer = mTriggerBufferAPI.begin() + oldSize;

	PX_ASSERT(oldSize == mTriggerBufferExtraData->size());
	mTriggerBufferExtraData->reserve(newCapacity);
	mTriggerBufferExtraData->forceSize_Unsafe(newSize);
	triggerPairExtraBuffer = mTriggerBufferExtraData->begin() + oldSize;
}

template<const bool sleepOrWoke, class T>
static void clearBodies(PxCoalescedHashSet<T*>& bodies)
{
	T* const* sleepingOrWokenBodies = bodies.getEntries();
	const PxU32 nb = bodies.size();
	for(PxU32 i=0; i<nb; i++)
	{
		ActorSim* body = sleepingOrWokenBodies[i]->getSim();

		if(sleepOrWoke)
		{
			PX_ASSERT(!body->readInternalFlag(ActorSim::BF_WAKEUP_NOTIFY));
			body->clearInternalFlag(ActorSim::BF_SLEEP_NOTIFY);
		}
		else
		{
			PX_ASSERT(!body->readInternalFlag(ActorSim::BF_SLEEP_NOTIFY));
			body->clearInternalFlag(ActorSim::BF_WAKEUP_NOTIFY);
		}

		// A body can be in both lists depending on the sequence of events
		body->clearInternalFlag(ActorSim::BF_IS_IN_SLEEP_LIST);
		body->clearInternalFlag(ActorSim::BF_IS_IN_WAKEUP_LIST);
	}
	bodies.clear();
}

void Sc::Scene::clearSleepWakeBodies()
{
	// Clear sleep/woken marker flags
	clearBodies<true>(mSleepBodies);
	clearBodies<false>(mWokeBodies);

	mSleepBodies.clear();
	mWokeBodies.clear();
	mWokeBodyListValid = true;
	mSleepBodyListValid = true;

#if PX_SUPPORT_GPU_PHYSX
	gpu_clearSleepWakeBodies();
#endif
}

void Sc::Scene::onBodySleep(BodySim* body)
{
	if (!mSimulationEventCallback && !mOnSleepingStateChanged)
		return;

	if (body->readInternalFlag(ActorSim::BF_WAKEUP_NOTIFY))
	{
		PX_ASSERT(!body->readInternalFlag(ActorSim::BF_SLEEP_NOTIFY));

		// Body is in the list of woken bodies, hence, mark this list as dirty such that it gets cleaned up before
		// being sent to the user
		body->clearInternalFlag(ActorSim::BF_WAKEUP_NOTIFY);
		mWokeBodyListValid = false;
	}

	body->raiseInternalFlag(ActorSim::BF_SLEEP_NOTIFY);

	// Avoid multiple insertion (the user can do multiple transitions between asleep and awake)
	if (!body->readInternalFlag(ActorSim::BF_IS_IN_SLEEP_LIST))
	{
		PX_ASSERT(!mSleepBodies.contains(&body->getBodyCore()));
		mSleepBodies.insert(&body->getBodyCore());
		body->raiseInternalFlag(ActorSim::BF_IS_IN_SLEEP_LIST);
	}
}

void Sc::Scene::onBodyWakeUp(BodySim* body)
{
	if(!mSimulationEventCallback && !mOnSleepingStateChanged)
		return;

	if (body->readInternalFlag(BodySim::BF_SLEEP_NOTIFY))
	{
		PX_ASSERT(!body->readInternalFlag(BodySim::BF_WAKEUP_NOTIFY));

		// Body is in the list of sleeping bodies, hence, mark this list as dirty such it gets cleaned up before
		// being sent to the user
		body->clearInternalFlag(BodySim::BF_SLEEP_NOTIFY);
		mSleepBodyListValid = false;
	}

	body->raiseInternalFlag(BodySim::BF_WAKEUP_NOTIFY);

	// Avoid multiple insertion (the user can do multiple transitions between asleep and awake)
	if (!body->readInternalFlag(BodySim::BF_IS_IN_WAKEUP_LIST))
	{
		PX_ASSERT(!mWokeBodies.contains(&body->getBodyCore()));
		mWokeBodies.insert(&body->getBodyCore());
		body->raiseInternalFlag(BodySim::BF_IS_IN_WAKEUP_LIST);
	}
}

PX_INLINE void Sc::Scene::cleanUpSleepBodies()
{
	BodyCore* const* bodyArray = mSleepBodies.getEntries();
	PxU32 bodyCount = mSleepBodies.size();

	IG::IslandSim& islandSim = mSimpleIslandManager->getAccurateIslandSim();

	while (bodyCount--)
	{
		ActorSim* actor = bodyArray[bodyCount]->getSim();
		BodySim* body = static_cast<BodySim*>(actor);

		if (body->readInternalFlag(static_cast<BodySim::InternalFlags>(ActorSim::BF_WAKEUP_NOTIFY)))
		{
			body->clearInternalFlag(static_cast<BodySim::InternalFlags>(ActorSim::BF_IS_IN_WAKEUP_LIST));
			mSleepBodies.erase(bodyArray[bodyCount]);
		}
		else if (islandSim.getNode(body->getNodeIndex()).isActive())
		{
			//This body is still active in the island simulation, so the request to deactivate the actor by the application must have failed. Recover by undoing this
			mSleepBodies.erase(bodyArray[bodyCount]);
			actor->internalWakeUp();

		}
	}

	mSleepBodyListValid = true;
}

PX_INLINE void Sc::Scene::cleanUpWokenBodies()
{
	cleanUpSleepOrWokenBodies(mWokeBodies, BodySim::BF_SLEEP_NOTIFY, mWokeBodyListValid);
}

PX_INLINE void Sc::Scene::cleanUpSleepOrWokenBodies(PxCoalescedHashSet<BodyCore*>& bodyList, PxU32 removeFlag, bool& validMarker)
{
	// With our current logic it can happen that a body is added to the sleep as well as the woken body list in the
	// same frame.
	//
	// Examples:
	// - Kinematic is created (added to woken list) but has not target (-> deactivation -> added to sleep list)
	// - Dynamic is created (added to woken list) but is forced to sleep by user (-> deactivation -> added to sleep list)
	//
	// This code traverses the sleep/woken body list and removes bodies which have been initially added to the given
	// list but do not belong to it anymore.

	BodyCore* const* bodyArray = bodyList.getEntries();
	PxU32 bodyCount = bodyList.size();
	while (bodyCount--)
	{
		BodySim* body = bodyArray[bodyCount]->getSim();

		if (body->readInternalFlag(static_cast<BodySim::InternalFlags>(removeFlag)))
			bodyList.erase(bodyArray[bodyCount]);
	}

	validMarker = true;
}

PxU32 Sc::Scene::createAggregate(void* userData, PxU32 maxNumShapes, PxAggregateFilterHint filterHint, PxU32 envID)
{
	const Bp::BoundsIndex index = getElementIDPool().createID();
#if PX_SUPPORT_GPU_PHYSX
	if(!mBoundsArray->initEntry(index))
	{
		PxGetFoundation().error(PxErrorCode::eOUT_OF_MEMORY, PX_FL, "Sc::Scene::createAggregate: failed to allocate pinned memory bounds");
		getCudaContextManager()->getCudaContext()->setAbortMode(true);
		// let aggregate creation continue to be able to remove it cleanly later.
	}
#else
	mBoundsArray->initEntry(index);
#endif

	mLLContext->getNphaseImplementationContext()->registerAggregate(index);
#if BP_USE_AGGREGATE_GROUP_TAIL
	return mAABBManager->createAggregate(index, Bp::FilterGroup::eINVALID, userData, maxNumShapes, filterHint, envID);
#else
	// PT: TODO: ideally a static compound would have a static group
	const PxU32 rigidId	= getRigidIDTracker().createID();
	const Bp::FilterGroup::Enum bpGroup = Bp::FilterGroup::Enum(rigidId + Bp::FilterGroup::eDYNAMICS_BASE);
	return mAABBManager->createAggregate(index, bpGroup, userData, maxNumShapes, filterHint, envID);
#endif
}

void Sc::Scene::deleteAggregate(PxU32 id)
{
	Bp::BoundsIndex index;
	Bp::FilterGroup::Enum bpGroup;
#if BP_USE_AGGREGATE_GROUP_TAIL
	if(mAABBManager->destroyAggregate(index, bpGroup, id))
	{
		getElementIDPool().releaseID(index);
	}
#else
	if(mAABBManager->destroyAggregate(index, bpGroup, id))
	{
		getElementIDPool().releaseID(index);

		// PT: this is clumsy....
		const PxU32 rigidId	= PxU32(bpGroup) - Bp::FilterGroup::eDYNAMICS_BASE;
		getRigidIDTracker().releaseID(rigidId);
	}
#endif
}

void Sc::Scene::shiftOrigin(const PxVec3& shift)
{
	// adjust low level context
	mLLContext->shiftOrigin(shift);

	// adjust bounds array
	mBoundsArray->shiftOrigin(shift);

	// adjust broadphase
	mAABBManager->shiftOrigin(shift);

	// adjust constraints
	ConstraintCore*const * constraints = mConstraints.getEntries();
	for(PxU32 i=0, size = mConstraints.size(); i < size; i++)
		constraints[i]->getPxConnector()->onOriginShift(shift);
}

///////////////////////////////////////////////////////////////////////////////

// PT: onActivate() functions should be called when an interaction is activated or created, and return true if activation
// should proceed else return false (for example: joint interaction between two kinematics should not get activated)
bool Sc::activateInteraction(Sc::Interaction* interaction)
{
	switch(interaction->getType())
	{
		case InteractionType::eOVERLAP:
			return static_cast<Sc::ShapeInteraction*>(interaction)->onActivate(NULL);

		case InteractionType::eTRIGGER:
			return static_cast<Sc::TriggerInteraction*>(interaction)->onActivate();

		case InteractionType::eMARKER:
			// PT: ElementInteractionMarker::onActivate() always returns false (always inactive).
			return false;

		case InteractionType::eCONSTRAINTSHADER:
			return static_cast<Sc::ConstraintInteraction*>(interaction)->onActivate();

		case InteractionType::eARTICULATION:
			return static_cast<Sc::ArticulationJointSim*>(interaction)->onActivate();

		case InteractionType::eTRACKED_IN_SCENE_COUNT:
		case InteractionType::eINVALID:
		PX_ASSERT(0);
		break;
	}
	return false;
}

// PT: onDeactivate() functions should be called when an interaction is deactivated, and return true if deactivation should proceed
// else return false (for example: joint interaction between two kinematics can ignore deactivation because it always is deactivated)
/*static*/ bool deactivateInteraction(Sc::Interaction* interaction, const Sc::InteractionType::Enum type)
{
	switch(type)
	{
		case InteractionType::eOVERLAP:
			return static_cast<Sc::ShapeInteraction*>(interaction)->onDeactivate();

		case InteractionType::eTRIGGER:
			return static_cast<Sc::TriggerInteraction*>(interaction)->onDeactivate();

		case InteractionType::eMARKER:
			// PT: ElementInteractionMarker::onDeactivate() always returns true.
			return true;

		case InteractionType::eCONSTRAINTSHADER:
			return static_cast<Sc::ConstraintInteraction*>(interaction)->onDeactivate();

		case InteractionType::eARTICULATION:
			return static_cast<Sc::ArticulationJointSim*>(interaction)->onDeactivate();

		case InteractionType::eTRACKED_IN_SCENE_COUNT:
		case InteractionType::eINVALID:
		PX_ASSERT(0);
		break;
	}
	return false;
}

void Sc::activateInteractions(Sc::ActorSim& actorSim)
{
	const PxU32 nbInteractions = actorSim.getActorInteractionCount();
	if(!nbInteractions)
		return;

	Interaction** interactions = actorSim.getActorInteractions();
	Scene& scene = actorSim.getScene();

	for(PxU32 i=0; i<nbInteractions; ++i)
	{
		PxPrefetchLine(interactions[PxMin(i+1,nbInteractions-1)]);
		Interaction* interaction = interactions[i];

		if(!interaction->readInteractionFlag(InteractionFlag::eIS_ACTIVE))
		{
			const InteractionType::Enum type = interaction->getType();
			const bool isNotIGControlled = type != InteractionType::eOVERLAP && type != InteractionType::eMARKER;

			if(isNotIGControlled)
			{
				const bool proceed = activateInteraction(interaction);
				if(proceed && (type < InteractionType::eTRACKED_IN_SCENE_COUNT))
					scene.notifyInteractionActivated(interaction);	// PT: we can reach this line for trigger interactions
			}
		}
	}
}

void Sc::deactivateInteractions(Sc::ActorSim& actorSim)
{
	const PxU32 nbInteractions = actorSim.getActorInteractionCount();
	if(!nbInteractions)
		return;

	Interaction** interactions = actorSim.getActorInteractions();
	Scene& scene = actorSim.getScene();

	for(PxU32 i=0; i<nbInteractions; ++i)
	{
		PxPrefetchLine(interactions[PxMin(i+1,nbInteractions-1)]);
		Interaction* interaction = interactions[i];

		if(interaction->readInteractionFlag(InteractionFlag::eIS_ACTIVE))
		{
			const InteractionType::Enum type = interaction->getType();
			const bool isNotIGControlled = type != InteractionType::eOVERLAP && type != InteractionType::eMARKER;
			if(isNotIGControlled)
			{
				const bool proceed = deactivateInteraction(interaction, type);
				if(proceed && (type < InteractionType::eTRACKED_IN_SCENE_COUNT))
					scene.notifyInteractionDeactivated(interaction);	// PT: we can reach this line for trigger interactions
			}
		}
	}
}

Sc::ConstraintCore*	Sc::Scene::findConstraintCore(const Sc::ActorSim* sim0, const Sc::ActorSim* sim1)
{
	const PxNodeIndex ind0 = sim0->getNodeIndex();
	const PxNodeIndex ind1 = sim1->getNodeIndex();

	if(ind1 < ind0)
		PxSwap(sim0, sim1);

	const PxHashMap<PxPair<const Sc::ActorSim*, const Sc::ActorSim*>, Sc::ConstraintCore*>::Entry* entry = mConstraintMap.find(PxPair<const Sc::ActorSim*, const Sc::ActorSim*>(sim0, sim1));
	return entry ? entry->second : NULL;
}

// PT: start moving PX_SUPPORT_GPU_PHYSX bits to the end of the file. Ideally/eventually they would move to a separate class or file,
// to clearly decouple the CPU and GPU parts of the scene/pipeline.
#if PX_SUPPORT_GPU_PHYSX
void Sc::Scene::gpu_updateBodySim(Sc::BodySim& bodySim)
{
	ArticulationSim* artiSim = bodySim.getArticulation();
	mSimulationController->updateDynamic(artiSim, bodySim.getNodeIndex());
}

void Sc::Scene::gpu_releasePools()
{
	PX_DELETE(mLLDeformableSurfacePool);
	PX_DELETE(mLLDeformableVolumePool);
	PX_DELETE(mLLParticleSystemPool);
}

void Sc::Scene::gpu_release()
{
	PX_DELETE(mHeapMemoryAllocationManager);
}

template<class T>
static void addToActiveArray(PxArray<T*>& activeArray, ActorSim& actorSim, ActorCore* core)
{
	const PxU32 activeListIndex = activeArray.size();
	actorSim.setActiveListIndex(activeListIndex);
	activeArray.pushBack(static_cast<T*>(core));
}

void Sc::Scene::gpu_addToActiveList(ActorSim& actorSim, ActorCore* appendedActorCore)
{
	if (actorSim.isDeformableSurface())
		addToActiveArray(mActiveDeformableSurfaces, actorSim, appendedActorCore);
	else if (actorSim.isDeformableVolume())
		addToActiveArray(mActiveDeformableVolumes, actorSim, appendedActorCore);
	else if (actorSim.isParticleSystem())
		addToActiveArray(mActiveParticleSystems, actorSim, appendedActorCore);
}

template<class T>
static void removeFromActiveArray(PxArray<T*>& activeArray, PxU32 removedActiveIndex)
{
	const PxU32 newSize = activeArray.size() - 1;

	if(removedActiveIndex != newSize)
	{
		T* lastBody = activeArray[newSize];
		activeArray[removedActiveIndex] = lastBody;
		lastBody->getSim()->setActiveListIndex(removedActiveIndex);
	}
	activeArray.forceSize_Unsafe(newSize);
}

void Sc::Scene::gpu_removeFromActiveList(ActorSim& actorSim, PxU32 removedActiveIndex)
{
	if(actorSim.isDeformableSurface())
		removeFromActiveArray(mActiveDeformableSurfaces, removedActiveIndex);
	else if(actorSim.isDeformableVolume())
		removeFromActiveArray(mActiveDeformableVolumes, removedActiveIndex);
	else if(actorSim.isParticleSystem())
		removeFromActiveArray(mActiveParticleSystems, removedActiveIndex);
}

void Sc::Scene::gpu_clearSleepWakeBodies()
{
	clearBodies<true>(mSleepDeformableVolumes);
	clearBodies<false>(mWokeDeformableVolumes);

	mWokeDeformableVolumeListValid = true;
	mSleepDeformableVolumeListValid = true;
}

void Sc::Scene::gpu_buildActiveActors()
{
	{
		PxU32 numActiveDeformableVolumes = getNumActiveDeformableVolumes();
		DeformableVolumeCore*const* PX_RESTRICT activeDeformableVolumes = getActiveDeformableVolumesArray();

		mActiveDeformableVolumeActors.clear();

		for (PxU32 i = 0; i < numActiveDeformableVolumes; i++)
		{
			PxActor* ra = activeDeformableVolumes[i]->getPxActor();
			mActiveDeformableVolumeActors.pushBack(ra);
		}
	}
}

void Sc::Scene::gpu_buildActiveAndFrozenActors()
{
	{
		PxU32 numActiveDeformableVolumes = getNumActiveDeformableVolumes();
		DeformableVolumeCore*const* PX_RESTRICT activeDeformableVolumes = getActiveDeformableVolumesArray();
		
		mActiveDeformableVolumeActors.clear();

		for (PxU32 i = 0; i < numActiveDeformableVolumes; i++)
		{
			PxActor* ra = activeDeformableVolumes[i]->getPxActor();
			mActiveDeformableVolumeActors.pushBack(ra);
		}
	}
}

void Sc::Scene::gpu_setSimulationEventCallback(PxSimulationEventCallback* /*callback*/)
{
	DeformableVolumeCore* const* sleepingDeformableVolumes = mSleepDeformableVolumes.getEntries();
	for (PxU32 i = 0; i < mSleepDeformableVolumes.size(); i++)
	{
		sleepingDeformableVolumes[i]->getSim()->raiseInternalFlag(BodySim::BF_SLEEP_NOTIFY);
	}

	//DeformableSurfaceCore* const* sleepingDeformableSurfaces = mSleepDeformableSurfaces.getEntries();
	//for (PxU32 i = 0; i < mSleepDeformableSurfaces.size(); i++)
	//{
	//	sleepingDeformableSurfaces[i]->getSim()->raiseInternalFlag(BodySim::BF_SLEEP_NOTIFY);
	//}
}

PxU32 Sc::Scene::gpu_cleanUpSleepAndWokenBodies()
{
	if (!mSleepDeformableVolumeListValid)
		cleanUpSleepDeformableVolumes();

	if (!mWokeBodyListValid)
		cleanUpWokenDeformableVolumes();

	const PxU32 nbVolumeSleep = mSleepDeformableVolumes.size();
	const PxU32 nbVolumeWoken = mWokeDeformableVolumes.size();
	return PxMax(nbVolumeWoken, nbVolumeSleep);
}

static void gpu_fireSleepingCallback(	PxActor** actors, const PxCoalescedHashSet<Sc::DeformableVolumeCore*>& bodies,
										PxSimulationEventCallback* simulationEventCallback, PxU64 contextID,
										Scene::SleepingStateChangedCallback onSleepingStateChanged, bool sleeping)
{
	PX_UNUSED(contextID);
	const PxU32 nbBodies = bodies.size();
	if (nbBodies)
	{
		PxU32 destSlot = 0;
		Sc::DeformableVolumeCore* const* sleepingDeformableVolumes = bodies.getEntries();
		for (PxU32 i = 0; i<nbBodies; i++)
		{
			Sc::DeformableVolumeCore* body = sleepingDeformableVolumes[i];
			if (body->getActorFlags() & PxActorFlag::eSEND_SLEEP_NOTIFIES)
				actors[destSlot++] = body->getPxActor();
			if (onSleepingStateChanged)
				onSleepingStateChanged(*static_cast<PxRigidDynamic*>(body->getPxActor()), sleeping);
		}

		if (destSlot && simulationEventCallback)
		{
			if(sleeping)
			{
				PX_PROFILE_ZONE("USERCODE - PxSimulationEventCallback::onSleep", contextID);
				simulationEventCallback->onSleep(actors, destSlot);
			}
			else
			{
				PX_PROFILE_ZONE("USERCODE - PxSimulationEventCallback::onWake", contextID);
				simulationEventCallback->onWake(actors, destSlot);
			}
		}
	}
}

void Sc::Scene::gpu_fireOnSleepCallback(PxActor** actors)
{
	//ML: need to create and API for the onSleep for deformable volume
	gpu_fireSleepingCallback(actors, mSleepDeformableVolumes, mSimulationEventCallback, mContextId, mOnSleepingStateChanged, true);
}

void Sc::Scene::gpu_fireOnWakeCallback(PxActor** actors)
{
	//ML: need to create an API for woken deformable volume
	gpu_fireSleepingCallback(actors, mWokeDeformableVolumes, mSimulationEventCallback, mContextId, mOnSleepingStateChanged, false);
}

void Sc::Scene::gpu_updateBounds()
{
	bool gpuStateChanged = false;

	PinnableBitMap& changedMap = mAABBManager->getChangedAABBMgActorHandleMap();

	//update deformable volumes world bound
	Sc::DeformableVolumeCore* const* deformableVolumes = mDeformableVolumes.getEntries();
	PxU32 size = mDeformableVolumes.size();
	if (mUseGpuBp)
	{
		for (PxU32 i = 0; i < size; ++i)
			changedMap.growAndSet(deformableVolumes[i]->getSim()->getShapeSim().getElementID());

		if(size)
			gpuStateChanged = true;
	}
	else
	{
		for (PxU32 i = 0; i < size; ++i)
		{
			DeformableVolumeSim* volumeSim = deformableVolumes[i]->getSim();
			ShapeSimBase& shapeSim = volumeSim->getShapeSim();

			PxBounds3 worldBounds = volumeSim->getWorldBounds();
			worldBounds.fattenSafe(shapeSim.getContactOffset()); // fatten for fast moving colliders
			mBoundsArray->setBounds(worldBounds, shapeSim.getElementID());
			changedMap.growAndSet(shapeSim.getElementID());
		}
	}

	// update FEM-cloth world bound
	Sc::DeformableSurfaceCore* const* deformableSurfaces = mDeformableSurfaces.getEntries();
	size = mDeformableSurfaces.size();
	if (mUseGpuBp)
	{
		for (PxU32 i = 0; i < size; ++i)
			changedMap.growAndSet(deformableSurfaces[i]->getSim()->getShapeSim().getElementID());

		if(size)
			gpuStateChanged = true;
	}
	else
	{
		for (PxU32 i = 0; i < size; ++i)
		{
			Sc::DeformableSurfaceSim* surfaceSim = deformableSurfaces[i]->getSim();
			ShapeSimBase& shapeSim = surfaceSim->getShapeSim();

			PxBounds3 worldBounds = surfaceSim->getWorldBounds();
			worldBounds.fattenSafe(shapeSim.getContactOffset()); // fatten for fast moving colliders
			mBoundsArray->setBounds(worldBounds, shapeSim.getElementID());
			changedMap.growAndSet(shapeSim.getElementID());
		}
	}

	//upate the actor handle of particle system in AABB manager 
	Sc::ParticleSystemCore* const* particleSystems = mParticleSystems.getEntries();

	size = mParticleSystems.size();
	if (mUseGpuBp)
	{
		for (PxU32 i = 0; i < size; ++i)
		{
			Sc::ShapeSimBase& ps = particleSystems[i]->getSim()->getShapeSim();

			//we are updating the bound in GPU so we just need to set the actor handle in CPU to make sure
			//the GPU BP will process the particles
			if (!(static_cast<Sc::ParticleSystemSim&>(ps.getActor()).getCore().getFlags() & PxParticleFlag::eDISABLE_RIGID_COLLISION))
			{
				changedMap.growAndSet(ps.getElementID());
				gpuStateChanged = true;
			}
		}

		if(gpuStateChanged)
			mAABBManager->setGPUStateChanged();
	}
	else
	{
		for (PxU32 i = 0; i < size; ++i)
		{
			ShapeSimBase& shapeSim = particleSystems[i]->getSim()->getShapeSim();

			const PxVec3 offset(shapeSim.getContactOffset());	// fatten for fast moving colliders
			mBoundsArray->setBounds(PxBounds3(-offset, offset), shapeSim.getElementID());
			changedMap.growAndSet(shapeSim.getElementID());
		}
	}
}

void Sc::Scene::addDeformableSurface(DeformableSurfaceCore& deformableSurface)
{
	DeformableSurfaceSim* sim = PX_NEW(DeformableSurfaceSim)(deformableSurface, *this);

	if (sim && (sim->getLowLevelDeformableSurface() == NULL))
	{
		PX_DELETE(sim);
		return;
	}

	mDeformableSurfaces.insert(&deformableSurface);
	mStats->gpuMemSizeDeformableSurfaces += deformableSurface.getGpuMemStat();
}

void Sc::Scene::removeDeformableSurface(DeformableSurfaceCore& deformableSurface)
{
	DeformableSurfaceSim* a = deformableSurface.getSim();
	if(a)
		markReleasedBodyIDForLostTouch(a->getActorID());
	PX_DELETE(a);
	mDeformableSurfaces.erase(&deformableSurface);
	mStats->gpuMemSizeDeformableSurfaces -= deformableSurface.getGpuMemStat();
}

void Sc::Scene::addDeformableVolume(DeformableVolumeCore& deformableVolume)
{
	DeformableVolumeSim* sim = PX_NEW(DeformableVolumeSim)(deformableVolume, *this);

	if (sim && (sim->getLowLevelDeformableVolume() == NULL))
	{
		PX_DELETE(sim);
		return;
	}

	mDeformableVolumes.insert(&deformableVolume);
	mStats->gpuMemSizeDeformableVolumes += deformableVolume.getGpuMemStat();
}

void Sc::Scene::removeDeformableVolume(DeformableVolumeCore& deformableVolume)
{
	DeformableVolumeSim* a = deformableVolume.getSim();
	if(a)
		markReleasedBodyIDForLostTouch(a->getActorID());
	PX_DELETE(a);
	mDeformableVolumes.erase(&deformableVolume);
	mStats->gpuMemSizeDeformableVolumes -= deformableVolume.getGpuMemStat();
}

void Sc::Scene::addParticleSystem(ParticleSystemCore& particleSystem)
{
	ParticleSystemSim* sim = PX_NEW(ParticleSystemSim)(particleSystem, *this);

	Dy::ParticleSystem* dyParticleSystem = sim->getLowLevelParticleSystem();

	if (sim && (dyParticleSystem == NULL))
	{
		PX_DELETE(sim);
		return;
	}

	mParticleSystems.insert(&particleSystem);
	mStats->gpuMemSizeParticles += particleSystem.getShapeCore().getGpuMemStat();
}

void Sc::Scene::removeParticleSystem(ParticleSystemCore& particleSystem)
{
	ParticleSystemSim* a = particleSystem.getSim();
	if(a)
		markReleasedBodyIDForLostTouch(a->getActorID());
	PX_DELETE(a);
	mParticleSystems.erase(&particleSystem);
	mStats->gpuMemSizeParticles -= particleSystem.getShapeCore().getGpuMemStat();
}

Dy::DeformableSurface* Sc::Scene::createLLDeformableSurface(Sc::DeformableSurfaceSim* sim)
{
	return mLLDeformableSurfacePool->construct(sim, sim->getCore().getCore());
}

void Sc::Scene::destroyLLDeformableSurface(Dy::DeformableSurface& deformableSurface)
{
	mLLDeformableSurfacePool->destroy(&deformableSurface);
}

Dy::DeformableVolume* Sc::Scene::createLLDeformableVolume(Sc::DeformableVolumeSim* sim)
{
	return mLLDeformableVolumePool->construct(sim, sim->getCore().getCore());
}

void Sc::Scene::destroyLLDeformableVolume(Dy::DeformableVolume& deformableVolume)
{
	mLLDeformableVolumePool->destroy(&deformableVolume);
}

Dy::ParticleSystem*	Sc::Scene::createLLParticleSystem(Sc::ParticleSystemSim* sim)
{
	return mLLParticleSystemPool->construct(sim->getCore().getShapeCore().getLLCore());
}

void Sc::Scene::destroyLLParticleSystem(Dy::ParticleSystem& particleSystem)
{
	return mLLParticleSystemPool->destroy(&particleSystem);
}

PX_INLINE void Sc::Scene::cleanUpSleepDeformableVolumes()
{
	DeformableVolumeCore* const* bodyArray = mSleepDeformableVolumes.getEntries();
	PxU32 bodyCount = mSleepBodies.size();

	IG::IslandSim& islandSim = mSimpleIslandManager->getAccurateIslandSim();

	while (bodyCount--)
	{
		DeformableVolumeSim* body = bodyArray[bodyCount]->getSim();

		if (body->readInternalFlag(static_cast<BodySim::InternalFlags>(ActorSim::BF_WAKEUP_NOTIFY)))
		{
			body->clearInternalFlag(static_cast<BodySim::InternalFlags>(ActorSim::BF_IS_IN_WAKEUP_LIST));
			mSleepDeformableVolumes.erase(bodyArray[bodyCount]);
		}
		else if (islandSim.getNode(body->getNodeIndex()).isActive())
		{
			//This body is still active in the island simulation, so the request to deactivate the actor by the application must have failed. Recover by undoing this
			mSleepDeformableVolumes.erase(bodyArray[bodyCount]);
			body->internalWakeUp();
		}
	}

	mSleepBodyListValid = true;
}

PX_INLINE void Sc::Scene::cleanUpWokenDeformableVolumes()
{
	cleanUpSleepOrWokenDeformableVolumes(mWokeDeformableVolumes, BodySim::BF_SLEEP_NOTIFY, mWokeDeformableVolumeListValid);
}

PX_INLINE void Sc::Scene::cleanUpSleepOrWokenDeformableVolumes(PxCoalescedHashSet<DeformableVolumeCore*>& bodyList, PxU32 removeFlag, bool& validMarker)
{
	// With our current logic it can happen that a body is added to the sleep as well as the woken body list in the
	// same frame.
	//
	// Examples:
	// - Kinematic is created (added to woken list) but has not target (-> deactivation -> added to sleep list)
	// - Dynamic is created (added to woken list) but is forced to sleep by user (-> deactivation -> added to sleep list)
	//
	// This code traverses the sleep/woken body list and removes bodies which have been initially added to the given
	// list but do not belong to it anymore.

	DeformableVolumeCore* const* bodyArray = bodyList.getEntries();
	PxU32 bodyCount = bodyList.size();
	while (bodyCount--)
	{
		DeformableVolumeSim* body = bodyArray[bodyCount]->getSim();

		if (body->readInternalFlag(static_cast<BodySim::InternalFlags>(removeFlag)))
			bodyList.erase(bodyArray[bodyCount]);
	}

	validMarker = true;
}

void Sc::Scene::addDeformableSurfaceSimControl(Sc::DeformableSurfaceCore& core)
{
	Sc::DeformableSurfaceSim* sim = core.getSim();

	if (sim)
	{
		mSimulationController->addFEMCloth(sim->getLowLevelDeformableSurface(), sim->getNodeIndex());

		mLLContext->getNphaseImplementationContext()->registerShape(sim->getNodeIndex(), sim->getShapeSim().getCore(), sim->getShapeSim().getElementID(), sim->getPxActor(), true);
	}
}

void Sc::Scene::removeDeformableSurfaceSimControl(Sc::DeformableSurfaceCore& core)
{
	Sc::DeformableSurfaceSim* sim = core.getSim();

	if (sim)
	{
		mLLContext->getNphaseImplementationContext()->unregisterShape(sim->getShapeSim().getCore(), sim->getShapeSim().getElementID(), true);
		mSimulationController->releaseFEMCloth(sim->getLowLevelDeformableSurface());
	}
}

void Sc::Scene::addDeformableVolumeSimControl(Sc::DeformableVolumeCore& core)
{
	Sc::DeformableVolumeSim* sim = core.getSim();

	if (sim)
	{
		mSimulationController->addSoftBody(sim->getLowLevelDeformableVolume(), sim->getNodeIndex());

		mLLContext->getNphaseImplementationContext()->registerShape(sim->getNodeIndex(), sim->getShapeSim().getCore(), sim->getShapeSim().getElementID(), sim->getPxActor());
	}
}

void Sc::Scene::removeDeformableVolumeSimControl(Sc::DeformableVolumeCore& core)
{
	Sc::DeformableVolumeSim* sim = core.getSim();

	if (sim)
	{
		mLLContext->getNphaseImplementationContext()->unregisterShape(sim->getShapeSim().getCore(), sim->getShapeSim().getElementID());
		mSimulationController->releaseSoftBody(sim->getLowLevelDeformableVolume());
	}
}

static PX_FORCE_INLINE void addToIslandManager(
	PxHashMap<PxPair<PxU32, PxU32>, DeformableRigidInteraction>& interactionMap,
	IG::SimpleIslandManager* islandManager, const ActorSim& sim, PxNodeIndex nodeIndex, IG::Edge::EdgeType edgeType)
{
	PxPair<PxU32, PxU32> pair(sim.getNodeIndex().index(), nodeIndex.index());
	DeformableRigidInteraction& interaction = interactionMap[pair];

	if (interaction.mCount == 0)
	{
		// PT: TODO: clarify why they do these IM calls exactly
		const IG::EdgeIndex edgeIdx = islandManager->addContactManager(NULL, sim.getNodeIndex(), nodeIndex, NULL, edgeType);
		islandManager->setEdgeConnected(edgeIdx, edgeType);
		interaction.mIndex = edgeIdx;
	}
	interaction.mCount++;
}

static PX_FORCE_INLINE void removeFromIslandManager(
	PxHashMap<PxPair<PxU32, PxU32>, DeformableRigidInteraction>& interactionMap,
	IG::SimpleIslandManager* islandManager, const ActorSim& sim, PxNodeIndex nodeIndex)
{
	PxPair<PxU32, PxU32> pair(sim.getNodeIndex().index(), nodeIndex.index());
	DeformableRigidInteraction& interaction = interactionMap[pair];
	interaction.mCount--;
	if (interaction.mCount == 0)
	{
		islandManager->removeConnection(interaction.mIndex);
		interactionMap.erase(pair);
	}
}


PxU32 Sc::Scene::addRigidAttachment(Sc::BodyCore* core, Sc::DeformableVolumeSim& sim, PxU32 vertId, const PxVec3& actorSpacePose,
	bool doConversion)
{
	PxNodeIndex nodeIndex;
	PxsRigidBody* body = NULL;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
		body = &core->getSim()->getLowLevelBody();
	}

	PxU32 handle = mSimulationController->addRigidAttachment(sim.getLowLevelDeformableVolume(), sim.getNodeIndex(), body,
		nodeIndex, vertId, actorSpacePose, sim.isActive(), doConversion);

	addToIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, nodeIndex, IG::Edge::eSOFT_BODY_CONTACT);

	return handle;
}

void Sc::Scene::removeRigidAttachment(Sc::BodyCore* core, Sc::DeformableVolumeSim& sim, PxU32 handle)
{
	PxNodeIndex nodeIndex;
	
	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
	}

	mSimulationController->removeRigidAttachment(sim.getLowLevelDeformableVolume(), handle);

	removeFromIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, nodeIndex);
}

void Sc::Scene::addTetRigidFilter(Sc::BodyCore* core, Sc::DeformableVolumeSim& sim, PxU32 tetIdx)
{
	PxNodeIndex nodeIndex;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
	}

	mSimulationController->addTetRigidFilter(sim.getLowLevelDeformableVolume(), nodeIndex, tetIdx);
}

void Sc::Scene::removeTetRigidFilter(Sc::BodyCore* core, Sc::DeformableVolumeSim& sim, PxU32 tetIdx)
{
	PxNodeIndex nodeIndex;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
	}
	mSimulationController->removeTetRigidFilter(sim.getLowLevelDeformableVolume(), nodeIndex, tetIdx);
}

PxU32 Sc::Scene::addTetRigidAttachment(Sc::BodyCore* core, Sc::DeformableVolumeSim& sim, PxU32 tetIdx, const PxVec4& barycentric, const PxVec3& actorSpacePose,
	bool doConversion)
{
	PxNodeIndex nodeIndex;
	PxsRigidBody* body = NULL;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
		body = &core->getSim()->getLowLevelBody();
	}

	PxU32 handle = mSimulationController->addTetRigidAttachment(sim.getLowLevelDeformableVolume(), body, nodeIndex,
		tetIdx, barycentric, actorSpacePose, sim.isActive(), doConversion);

	addToIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, nodeIndex, IG::Edge::eSOFT_BODY_CONTACT);

	return handle;
}

void Sc::Scene::addSoftBodyFilter(DeformableVolumeCore& core, PxU32 tetIdx0, DeformableVolumeSim& sim, PxU32 tetIdx1)
{
	Sc::DeformableVolumeSim& bSim = *core.getSim();

	mSimulationController->addSoftBodyFilter(bSim.getLowLevelDeformableVolume(), sim.getLowLevelDeformableVolume(), tetIdx0, tetIdx1);
}

void Sc::Scene::removeSoftBodyFilter(DeformableVolumeCore& core, PxU32 tetIdx0, DeformableVolumeSim& sim, PxU32 tetIdx1)
{
	Sc::DeformableVolumeSim& bSim = *core.getSim();
	mSimulationController->removeSoftBodyFilter(bSim.getLowLevelDeformableVolume(), sim.getLowLevelDeformableVolume(), tetIdx0, tetIdx1);
}

void Sc::Scene::addSoftBodyFilters(DeformableVolumeCore& core, DeformableVolumeSim& sim, PxU32* tetIndices0, PxU32* tetIndices1, PxU32 tetIndicesSize)
{
	Sc::DeformableVolumeSim& bSim = *core.getSim();

	mSimulationController->addSoftBodyFilters(bSim.getLowLevelDeformableVolume(), sim.getLowLevelDeformableVolume(), tetIndices0, tetIndices1, tetIndicesSize);
}

void Sc::Scene::removeSoftBodyFilters(DeformableVolumeCore& core, DeformableVolumeSim& sim, PxU32* tetIndices0, PxU32* tetIndices1, PxU32 tetIndicesSize)
{
	Sc::DeformableVolumeSim& bSim = *core.getSim();
	mSimulationController->removeSoftBodyFilters(bSim.getLowLevelDeformableVolume(), sim.getLowLevelDeformableVolume(), tetIndices0, tetIndices1, tetIndicesSize);
}

PxU32 Sc::Scene::addSoftBodyAttachment(DeformableVolumeCore& core, PxU32 tetIdx0, const PxVec4& tetBarycentric0, Sc::DeformableVolumeSim& sim, PxU32 tetIdx1, const PxVec4& tetBarycentric1,
	bool doConversion)
{
	Sc::DeformableVolumeSim& bSim = *core.getSim();

	PxU32 handle = mSimulationController->addSoftBodyAttachment(bSim.getLowLevelDeformableVolume(), sim.getLowLevelDeformableVolume(), tetIdx0, tetIdx1,
		tetBarycentric0, tetBarycentric1, sim.isActive() || bSim.isActive(), doConversion);

	addToIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, bSim.getNodeIndex(), IG::Edge::eSOFT_BODY_CONTACT);

	return handle;
}

void Sc::Scene::removeSoftBodyAttachment(DeformableVolumeCore& core, Sc::DeformableVolumeSim& sim, PxU32 handle)
{
	Sc::DeformableVolumeSim& bSim = *core.getSim();
	mSimulationController->removeSoftBodyAttachment(bSim.getLowLevelDeformableVolume(), handle);

	removeFromIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, bSim.getNodeIndex());
}

void Sc::Scene::addClothFilter(Sc::DeformableSurfaceCore& core, PxU32 triIdx, Sc::DeformableVolumeSim& sim, PxU32 tetIdx)
{
	Sc::DeformableSurfaceSim& bSim = *core.getSim();

	mSimulationController->addClothFilter(sim.getLowLevelDeformableVolume(), bSim.getLowLevelDeformableSurface(), triIdx,tetIdx);
}

void Sc::Scene::removeClothFilter(Sc::DeformableSurfaceCore& core, PxU32 triIdx, Sc::DeformableVolumeSim& sim, PxU32 tetIdx)
{
	Sc::DeformableSurfaceSim& bSim = *core.getSim();
	mSimulationController->removeClothFilter(sim.getLowLevelDeformableVolume(), bSim.getLowLevelDeformableSurface(), triIdx, tetIdx);
}

PxU32 Sc::Scene::addClothAttachment(Sc::DeformableSurfaceCore& core, PxU32 triIdx, const PxVec4& triBarycentric, Sc::DeformableVolumeSim& sim, PxU32 tetIdx,
	const PxVec4& tetBarycentric, bool doConversion)
{
	Sc::DeformableSurfaceSim& bSim = *core.getSim();

	PxU32 handle = mSimulationController->addClothAttachment(sim.getLowLevelDeformableVolume(), bSim.getLowLevelDeformableSurface(), triIdx, triBarycentric,
		tetIdx, tetBarycentric, sim.isActive(), doConversion);

	addToIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, bSim.getNodeIndex(), IG::Edge::eFEM_CLOTH_CONTACT);

	return handle;
}

void Sc::Scene::removeClothAttachment(Sc::DeformableSurfaceCore& core, Sc::DeformableVolumeSim& sim, PxU32 handle)
{
	PX_UNUSED(core);
	Sc::DeformableSurfaceSim& bSim = *core.getSim();
	mSimulationController->removeClothAttachment(sim.getLowLevelDeformableVolume(), handle);

	removeFromIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, bSim.getNodeIndex());
}

PxU32 Sc::Scene::addRigidAttachment(Sc::BodyCore* core, Sc::DeformableSurfaceSim& sim, PxU32 vertId, const PxVec3& actorSpacePose)
{
	PxNodeIndex nodeIndex;
	PxsRigidBody* body = NULL;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
		body = &core->getSim()->getLowLevelBody();
	}

	PxU32 handle = mSimulationController->addRigidAttachment(sim.getLowLevelDeformableSurface(), sim.getNodeIndex(), body, nodeIndex,
		vertId, actorSpacePose, sim.isActive());

	addToIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, nodeIndex, IG::Edge::eFEM_CLOTH_CONTACT);

	return handle;
}

void Sc::Scene::removeRigidAttachment(Sc::BodyCore* core, Sc::DeformableSurfaceSim& sim, PxU32 handle)
{
	PxNodeIndex nodeIndex;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
	}

	mSimulationController->removeRigidAttachment(sim.getLowLevelDeformableSurface(), handle);

	removeFromIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, nodeIndex);
}

void Sc::Scene::addTriRigidFilter(Sc::BodyCore* core, Sc::DeformableSurfaceSim& sim, PxU32 triIdx)
{
	PxNodeIndex nodeIndex;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
	}

	mSimulationController->addTriRigidFilter(sim.getLowLevelDeformableSurface(), nodeIndex, triIdx);
}

void Sc::Scene::removeTriRigidFilter(Sc::BodyCore* core, Sc::DeformableSurfaceSim& sim, PxU32 triIdx)
{
	PxNodeIndex nodeIndex;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
	}

	mSimulationController->removeTriRigidFilter(sim.getLowLevelDeformableSurface(), nodeIndex, triIdx);
}

PxU32 Sc::Scene::addTriRigidAttachment(Sc::BodyCore* core, Sc::DeformableSurfaceSim& sim, PxU32 triIdx, const PxVec4& barycentric, const PxVec3& actorSpacePose)
{
	PxNodeIndex nodeIndex;
	PxsRigidBody* body = NULL;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
		body = &core->getSim()->getLowLevelBody();
	}

	PxU32 handle = mSimulationController->addTriRigidAttachment(sim.getLowLevelDeformableSurface(), body, nodeIndex, triIdx, barycentric, actorSpacePose, sim.isActive());

	addToIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, nodeIndex, IG::Edge::eFEM_CLOTH_CONTACT);

	return handle;
}

void Sc::Scene::removeTriRigidAttachment(Sc::BodyCore* core, Sc::DeformableSurfaceSim& sim, PxU32 handle)
{
	PxNodeIndex nodeIndex;

	if (core)
	{
		nodeIndex = core->getSim()->getNodeIndex();
	}

	mSimulationController->removeTriRigidAttachment(sim.getLowLevelDeformableSurface(), handle);

	removeFromIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim, nodeIndex);
}

void Sc::Scene::addClothFilter(DeformableSurfaceCore& core0, PxU32 triIdx0, Sc::DeformableSurfaceSim& sim1, PxU32 triIdx1)
{
	Sc::DeformableSurfaceSim& sim0 = *core0.getSim();

	mSimulationController->addClothFilter(sim0.getLowLevelDeformableSurface(), sim1.getLowLevelDeformableSurface(), triIdx0, triIdx1);
}

void Sc::Scene::removeClothFilter(DeformableSurfaceCore& core, PxU32 triIdx0, DeformableSurfaceSim& sim1, PxU32 triIdx1)
{
	Sc::DeformableSurfaceSim& sim0 = *core.getSim();
	mSimulationController->removeClothFilter(sim0.getLowLevelDeformableSurface(), sim1.getLowLevelDeformableSurface(), triIdx0, triIdx1);
}

PxU32 Sc::Scene::addTriClothAttachment(DeformableSurfaceCore& core, PxU32 triIdx0, const PxVec4& barycentric0, Sc::DeformableSurfaceSim& sim1, PxU32 triIdx1, const PxVec4& barycentric1)
{
	Sc::DeformableSurfaceSim& sim0 = *core.getSim();

	PxU32 handle = mSimulationController->addTriClothAttachment(sim0.getLowLevelDeformableSurface(), sim1.getLowLevelDeformableSurface(), triIdx0, triIdx1, barycentric0, barycentric1, sim1.isActive() || sim0.isActive());

	addToIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim0, sim1.getNodeIndex(), IG::Edge::eFEM_CLOTH_CONTACT);

	return handle;
}

void Sc::Scene::removeTriClothAttachment(DeformableSurfaceCore& core, DeformableSurfaceSim& sim1, PxU32 handle)
{
	Sc::DeformableSurfaceSim& sim0 = *core.getSim();
	mSimulationController->removeTriClothAttachment(sim0.getLowLevelDeformableSurface(), handle);

	removeFromIslandManager(mDeformableRigidInteractionMap, mSimpleIslandManager, sim0, sim1.getNodeIndex());
}

void Sc::Scene::addParticleSystemSimControl(Sc::ParticleSystemCore& core)
{
	Sc::ParticleSystemSim* sim = core.getSim();

	if (sim)
	{
		mSimulationController->addParticleSystem(sim->getLowLevelParticleSystem(), sim->getNodeIndex());
		
		mLLContext->getNphaseImplementationContext()->registerShape(sim->getNodeIndex(), sim->getCore().getShapeCore(), sim->getLowLevelParticleSystem()->getElementId(), sim->getPxActor());
	}
}

void Sc::Scene::removeParticleSystemSimControl(Sc::ParticleSystemCore& core)
{
	Sc::ParticleSystemSim* sim = core.getSim();

	if (sim)
	{
		mLLContext->getNphaseImplementationContext()->unregisterShape(sim->getCore().getShapeCore(), sim->getShapeSim().getElementID());
		mSimulationController->releaseParticleSystem(sim->getLowLevelParticleSystem());
	}
}

PxActor** Sc::Scene::getActiveDeformableVolumeActors(PxU32& nbActorsOut)
{
	nbActorsOut = mActiveDeformableVolumeActors.size();

	if (!nbActorsOut)
		return NULL;

	return mActiveDeformableVolumeActors.begin();
}

void Sc::Scene::setActiveDeformableVolumeActors(PxActor** actors, PxU32 nbActors)
{
	mActiveDeformableVolumeActors.forceSize_Unsafe(0);
	mActiveDeformableVolumeActors.resize(nbActors);
	PxMemCopy(mActiveDeformableVolumeActors.begin(), actors, sizeof(PxActor*) * nbActors);
}

//PxActor** Sc::Scene::getActiveDeformableSurfaceActors(PxU32& nbActorsOut)
//{
//	nbActorsOut = mActiveDeformableSurfaceActors.size();
//
//	if (!nbActorsOut)
//		return NULL;
//
//	return mActiveDeformableSurfaceActors.begin();
//}
//
//void Sc::Scene::setActiveDeformableSurfaceActors(PxActor** actors, PxU32 nbActors)
//{
//	mActiveDeformableSurfaceActors.forceSize_Unsafe(0);
//	mActiveDeformableSurfaceActors.resize(nbActors);
//	PxMemCopy(mActiveDeformableSurfaceActors.begin(), actors, sizeof(PxActor*) * nbActors);
//}

#endif //PX_SUPPORT_GPU_PHYSX
