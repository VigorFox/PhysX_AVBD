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
#include "DyAvbdSoftBodyComponent.h"
#include "DyDeformableSurface.h"
#include "DyDeformableVolume.h"
#include "foundation/PxHashMap.h"
#include "geometry/PxHeightField.h"
#include "geometry/PxHeightFieldGeometry.h"
#include "geometry/PxMeshQuery.h"
#include "geometry/PxTriangle.h"
#include "geometry/PxTriangleMesh.h"
#include "GuTetrahedronMesh.h"
#include "PxsDeformableSurfaceMaterialCore.h"
#include "PxsMaterialCore.h"

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
			PxArray<PxU32>					globalParticleIndices;
			PxArray<Dy::AvbdSoftParticle>	particles;
			PxArray<Dy::AvbdSoftBody>		bodies;
			PxArray<Dy::AvbdSelfCollisionAdjacency>
												selfCollisionAdjacencies;
			PxArray<PxU8>					selfCollisionEnabled;
			PxArray<Dy::AvbdRigidBox>		rigidBoxes;
			PxArray<Dy::AvbdRigidBox>		selectedDynamicBoxes;
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
				  selectedIsland(PX_MAX_U32)
			{
			}
		};

	public:
		AvbdCpuSoftScene(
			const PxsDeformableVolumeMaterialManager&
				deformableMaterialManager,
			const PxsDeformableSurfaceMaterialManager&
				surfaceMaterialManager,
			const PxsMaterialManager& rigidMaterialManager,
			IG::SimpleIslandManager& islandManager)
			: mDeformableMaterialManager(deformableMaterialManager),
			  mSurfaceMaterialManager(surfaceMaterialManager),
			  mRigidMaterialManager(rigidMaterialManager),
			  mIslandManager(islandManager),
			  mNextPrimitiveKey(1),
			  mNextWorldPinHandle(1),
			  mNextRigidAttachmentHandle(1),
			  mNextArticulationAttachmentHandle(1),
			  mNextSoftPairAttachmentHandle(1),
			  mNextPrescribedAttachmentHandle(1),
			  mNextRigidActorFilterHandle(1),
			  mNextDeformablePairFilterHandle(1),
			  mDynamicsOwnsStep(false),
			  mDynamicsSelectedEntryCount(0)
		{
		}

		~AvbdCpuSoftScene()
		{
			clearNativeIslandEdges();
			clearIslandSelectionStorages();
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				Entry& entry = mEntries[i];
				mIslandManager.removeNode(entry.islandNode);
				entry.destroyIslandObject();
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
				material ? material->dynamicFriction : 0.5f);

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
			const Gu::DeformableVolumeAuxData& auxData =
				static_cast<
					const Gu::DeformableVolumeAuxData&>(
						*entry->auxData);
			const PxU32 publicElementCount =
				entry->collisionMesh->getNbTetrahedrons();
			const PxU32 simulationElementCount =
				entry->simulationMesh->getNbTetrahedrons();
			if(!auxData.mTetsAccumulatedRemapColToSim ||
				!auxData.mTetsRemapColToSim ||
				auxData.mTetsRemapSize == 0)
				return PX_MAX_U32;

			PxArray<PxU32> simulationElements;
			for(PxU32 selectedIndex = 0;
				selectedIndex < collisionElementCount;
				++selectedIndex)
			{
				const PxU32 collisionElement =
					collisionElementIndices[selectedIndex];
				if(collisionElement >= publicElementCount)
					return PX_MAX_U32;
				const PxU32 begin = collisionElement == 0
					? 0
					: auxData.
						mTetsAccumulatedRemapColToSim[
							collisionElement - 1];
				const PxU32 end =
					auxData.mTetsAccumulatedRemapColToSim[
						collisionElement];
				if(end <= begin || end > auxData.mTetsRemapSize)
					return PX_MAX_U32;
				for(PxU32 remapIndex = begin;
					remapIndex < end; ++remapIndex)
				{
					const PxU32 simulationElement =
						auxData.mTetsRemapColToSim[
							remapIndex];
					if(simulationElement >=
						simulationElementCount)
						return PX_MAX_U32;
					simulationElements.pushBack(
						simulationElement);
				}
			}
			if(simulationElements.empty())
				return PX_MAX_U32;
			return addRigidActorFilter(
				softCore, rigidCore,
				simulationElements.begin(),
				simulationElements.size(), false);
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
			const PxU32 simulationElementCount =
				entry.simulationMesh->getNbTetrahedrons();
			if(collisionElement >= collisionElementCount)
				return false;
			const Gu::DeformableVolumeAuxData& auxData =
				static_cast<
					const Gu::DeformableVolumeAuxData&>(
						*entry.auxData);
			if(!auxData.mTetsAccumulatedRemapColToSim ||
				!auxData.mTetsRemapColToSim ||
				auxData.mTetsRemapSize == 0)
				return false;
			const PxU32 begin = collisionElement == 0
				? 0
				: auxData.mTetsAccumulatedRemapColToSim[
					collisionElement - 1];
			const PxU32 end =
				auxData.mTetsAccumulatedRemapColToSim[
					collisionElement];
			if(end <= begin || end > auxData.mTetsRemapSize)
				return false;
			for(PxU32 remapIndex = begin;
				remapIndex < end; ++remapIndex)
			{
				const PxU32 simulationElement =
					auxData.mTetsRemapColToSim[remapIndex];
				if(simulationElement >= simulationElementCount)
					return false;
				simulationElements.pushBack(simulationElement);
			}
			return !simulationElements.empty();
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
				if(!computeSoftBounds(softEntry, softBounds))
					continue;
				const bool speculativeCCDEnabled =
					softEntry.bodyIndex < mBodies.size() &&
					mBodies[softEntry.bodyIndex].compiled.
						speculativeCCDEnabled;
				if(speculativeCCDEnabled &&
					!expandSoftBoundsForPrediction(
						softEntry, dt, gravity, softBounds))
					continue;

				for(PxU32 shapeIndex = 0;
					shapeIndex < mDynamicShapes.size(); shapeIndex++)
				{
					const DynamicShapeEntry& dynamicEntry =
						mDynamicShapes[shapeIndex];
					Dy::AvbdRigidBox box;
					PxBounds3 rigidBounds;
					if(compileDynamicBox(dynamicEntry, box))
						rigidBounds = computeBoxBounds(box);
					else
					{
						Dy::AvbdRigidSphere sphere;
						if(compileDynamicSphere(
								dynamicEntry, sphere))
						{
							rigidBounds = computeSphereBounds(sphere);
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
				if(!computeSoftBounds(softEntry0, softBounds0))
					continue;
				const PxReal wakeMargin =
					PxMax(mContactParams.contactRadius, 0.0f);
				softBounds0.fattenSafe(wakeMargin);

				for(PxU32 softIndex1 = softIndex0 + 1;
					softIndex1 < mEntries.size(); softIndex1++)
				{
					Entry& softEntry1 = mEntries[softIndex1];
					PxBounds3 softBounds1;
					if(!computeSoftBounds(softEntry1, softBounds1) ||
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

			PxU32 selectedEntryCount = 0;
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
				selection.particles = storage.particles.begin();
				selection.numParticles = storage.particles.size();
				selection.bodies = storage.bodies.begin();
				selection.numBodies = storage.bodies.size();
				selection.contacts = storage.contacts.begin();
				selection.numContacts = storage.contacts.size();
				selection.islandIndex = storage.selectedIsland;
				selection.iterationOverride = innerIterations;
				if(!selection.isComplete())
				{
					selections.clear();
					return false;
				}
				selections.pushBack(selection);
				selectedEntryCount += storage.entryIndices.size();
			}

			mDynamicsOwnsStep = !selections.empty();
			mDynamicsSelectedEntryCount =
				mDynamicsOwnsStep ? selectedEntryCount : 0;
			return mDynamicsOwnsStep;
		}

		void step(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager,
			bool sleepingEnabled)
		{
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
				// Complete native selections have already been solved by the
				// unified rigid/soft AVBD path. Advance any unrelated awake
				// soft-only actors through the established component path,
				// then restore the selected particles and AL runtime state so
				// no selected body is double-owned.
				if(mDynamicsSelectedEntryCount < awakeEntryCount)
					stepComponentFallback(
						dt, gravity, materialManager,
						rigidMaterialManager);
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
				mDynamicsOwnsStep = false;
				mDynamicsSelectedEntryCount = 0;
				return;
			}

			stepComponentFallback(
				dt, gravity, materialManager,
				rigidMaterialManager);

			finalizeDeformableMotionControls(dt);
			for(PxU32 i = 0; i < mEntries.size(); i++)
				writeBack(mEntries[i]);
			updateSleepStates(dt, sleepingEnabled);
			mDynamicsSelectedEntryCount = 0;
		}

	private:
		void stepComponentFallback(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager)
		{
			PxU32 requestedPositionIterations = 1;
			PxU32 requestedCollisionPairUpdates = 0;
			PxU32 requestedCollisionSubsteps = 1;
			bool hasExplicitCollisionPairUpdates = false;
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				syncHostInputs(mEntries[i], materialManager);
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
			detectContacts(
				mParticles.begin(), mParticles.size(),
				mBodies.begin(), mBodies.size(), mContacts);
			const bool needsContactRedetection =
				!mContacts.empty() || mBodies.size() > 1 ||
				!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
				!mRigidCapsules.empty() ||
				!mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				!mWorldPlanes.empty();
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
			const PxU32 totalPositionIterations = PxMax<PxU32>(
				requestedPositionIterations,
				minimumContactIterations);
			const PxU32 outerIterations =
				needsContactRedetection
					? PxMin<PxU32>(
						requestedRedetectionStages,
						totalPositionIterations)
					: 1u;
			const PxU32 innerIterations =
				(totalPositionIterations + outerIterations - 1) /
				outerIterations;

			Dy::avbdStepSoftBodies(
				mParticles.begin(), mParticles.size(),
				mBodies.begin(), mBodies.size(),
				mContacts.begin(), mContacts.size(),
				dt, gravity,
				outerIterations, innerIterations,
				1000.0f,
				redetectContacts, &mContacts, this,
				0.92f, NULL, &mWorkspace,
				totalPositionIterations,
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(),
				mSelfCollisionEnabled.begin(),
				&mContactParams);
		}

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

		void updateSleepStates(
			PxReal dt, bool sleepingEnabled)
		{
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
				if(kinematicTargetResidualPending)
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
			return rebaseSoftBodyParticleRangeInPlace(
				destination, globalStart, particleCount, localStart);
		}

		static bool rebaseSoftBodyParticleRangeInPlace(
			Dy::AvbdSoftBody& body,
			PxU32 oldStart, PxU32 particleCount,
			PxU32 newStart)
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
			body.runtime.compileObjectiveProgram(
				newStart, particleCount);
			return body.runtime.isObjectiveProgramCurrent(
				newStart, particleCount);
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
			storage.globalParticleIndices.clear();
			storage.particles.clear();
			storage.bodies.clear();
			storage.selfCollisionAdjacencies.clear();
			storage.selfCollisionEnabled.clear();
			storage.rigidBoxes.clear();
			storage.selectedDynamicBoxes.clear();
			storage.rigidSpheres.clear();
			storage.selectedDynamicSpheres.clear();
			storage.rigidCapsules.clear();
			storage.selectedDynamicCapsules.clear();
			storage.rigidConvexes.clear();
			storage.selectedDynamicConvexes.clear();
			storage.probeContacts.clear();

			for(PxU32 entryOrder = 0;
				entryOrder < storage.entryIndices.size(); entryOrder++)
			{
				const Entry& entry =
					mEntries[storage.entryIndices[entryOrder]];
				storage.softCores.pushBack(entry.getActorCore());
				const PxU32 localStart = storage.particles.size();
				const PxU32 particleStart =
					getParticleStart(entry);
				const PxU32 particleCount =
					getParticleCount(entry);
				for(PxU32 i = 0; i < particleCount; i++)
				{
					const PxU32 globalIndex = particleStart + i;
					if(globalIndex >= mParticles.size())
						return false;
					storage.globalParticleIndices.pushBack(globalIndex);
					storage.particles.pushBack(mParticles[globalIndex]);
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
			// now so swept contact prep can select a first-impact objective;
			// the solver recomputes this idempotently before iteration.
			for(PxU32 particleIndex = 0;
				particleIndex < storage.particles.size(); particleIndex++)
				storage.particles[particleIndex].
					computePrediction(dt, gravity);

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
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicBoxes.begin(),
					storage.selectedDynamicBoxes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					NULL, 0,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidSweptSDF(
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicBoxes.begin(),
					storage.selectedDynamicBoxes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidOGCFeatures(
					storage.particles.begin(),
					storage.particles.size(),
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
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidSphereSweptSDF(
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidSphereSweptOGCFeatures(
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
				Dy::avbdDetectSoftRigidSphereOGCFeatures(
					storage.particles.begin(),
					storage.particles.size(),
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
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidCapsuleSweptSDF(
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidCapsuleSweptOGCFeatures(
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
				Dy::avbdDetectSoftRigidCapsuleOGCFeatures(
					storage.particles.begin(),
					storage.particles.size(),
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
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidConvexSweptSDF(
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidConvexSweptOGCFeatures(
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
				Dy::avbdDetectSoftRigidConvexOGCFeatures(
					storage.particles.begin(),
					storage.particles.size(),
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
			}
			if(storage.probeContacts.empty() &&
				!hasRigidAttachment &&
				!hasArticulationAttachment &&
				!hasSoftPairAttachment)
				return false;

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
			detectContacts(
				storage.particles.begin(), storage.particles.size(),
				storage.bodies.begin(), storage.bodies.size(),
				storage.contacts, storage.rigidBoxes.begin(),
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

			for(PxU32 i = 0; i < storage.contacts.size(); i++)
				if(storage.contacts[i].geometry.hasRigidBodyTarget())
					return true;
			return hasRigidAttachment ||
				hasArticulationAttachment ||
				hasSoftPairAttachment;
		}

		void copyIslandSelectionResults(
			IslandSelectionStorage& storage)
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
			return finalizeTriangleSurfaceTopology(
				surface, false);
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
			return finalizeTriangleSurfaceTopology(
				surface, suppressBoundaryEdges);
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
			bool topologyCompiled = false;
			if(shape.getGeometryType() ==
				PxGeometryType::eTRIANGLEMESH)
			{
				const PxTriangleMeshGeometry& geometry =
					static_cast<
						const PxTriangleMeshGeometry&>(
							shape.getGeometry());
				topologyCompiled = compileTriangleMeshTopology(
					shape, mRigidMaterialManager,
					geometry, surface);
			}
			else if(shape.getGeometryType() ==
				PxGeometryType::eHEIGHTFIELD)
			{
				const PxHeightFieldGeometry& geometry =
					static_cast<
						const PxHeightFieldGeometry&>(
							shape.getGeometry());
				topologyCompiled = compileHeightFieldTopology(
					shape, mRigidMaterialManager,
					geometry, surface);
			}
			if(!topologyCompiled)
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

		void compileWorldStatics(
			const PxsMaterialManager& materialManager)
		{
			mWorldPlanes.clear();
			mRigidBoxes.clear();
			mRigidSpheres.clear();
			mRigidCapsules.clear();
			mRigidConvexes.clear();
			mRigidTriangleSurfaces.clear();
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
					Dy::AvbdRigidTriangleSurface surface;
					const bool compiled =
						shape.getGeometryType() ==
							PxGeometryType::eTRIANGLEMESH
						? compileTriangleMeshTopology(
							shape, materialManager,
							static_cast<
								const PxTriangleMeshGeometry&>(
									shape.getGeometry()),
							surface)
						: compileHeightFieldTopology(
							shape, materialManager,
							static_cast<
								const PxHeightFieldGeometry&>(
									shape.getGeometry()),
							surface);
					if(!compiled)
						continue;
					surface.center = shapeToWorld.p;
					surface.rotation = shapeToWorld.q;
					surface.previousCenter = shapeToWorld.p;
					surface.previousRotation =
						shapeToWorld.q;
					surface.primitiveKey =
						entry.primitiveKey;
					mRigidTriangleSurfaces.pushBack(
						surface);
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
				Dy::AvbdRigidTriangleSurface surface;
				if(compileDynamicTriangleSurface(
						entry, surface))
				{
					surface.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					mRigidTriangleSurfaces.pushBack(
						surface);
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
				if(bodies == mBodies.begin())
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
			if(bodies == mBodies.begin())
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

			// Rigid contact generation is particle-sampled. Surface filters
			// own source triangles directly. Volume filters are compiled
			// from public collision tetrahedra through the cooked overlap
			// mapping and therefore own source simulation tetrahedra here.
			// In both domains, remove the objective only when every incident
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
			Dy::avbdDetectAllOGCContacts(
				particles, numParticles,
				bodies, numBodies,
				rigidBoxes, numRigidBoxes,
				selfCollisionAdjacencies,
				numSelfCollisionAdjacencies,
				contacts, mContactParams, 0.0f,
				NULL, &mWorkspace.contact,
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
				body.material.coRotationalVolumeModel =
					!material ||
					material->materialModel ==
						PxDeformableVolumeMaterialModel::
							eCO_ROTATIONAL;
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
				const PxU32 count =
					PxMin(collisionVertexCount, particleCount);
				for(PxU32 i = 0; i < count; i++)
				{
					const PxReal invMass =
						core.positionInvMass[i].w;
					core.positionInvMass[i] = PxVec4(
						mParticles[particleStart + i].position,
						invMass);
				}
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
		const PxsDeformableVolumeMaterialManager&
										mDeformableMaterialManager;
		const PxsDeformableSurfaceMaterialManager&
										mSurfaceMaterialManager;
		const PxsMaterialManager&		mRigidMaterialManager;
		IG::SimpleIslandManager&		mIslandManager;
		PxU64							mNextPrimitiveKey;
		PxU32							mNextWorldPinHandle;
		PxU32							mNextRigidAttachmentHandle;
		PxU32							mNextArticulationAttachmentHandle;
		PxU32							mNextSoftPairAttachmentHandle;
		PxU32							mNextPrescribedAttachmentHandle;
		PxU32							mNextRigidActorFilterHandle;
		PxU32							mNextDeformablePairFilterHandle;
		bool							mDynamicsOwnsStep;
		PxU32							mDynamicsSelectedEntryCount;
	};

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

#if PX_SUPPORT_GPU_PHYSX
	if(desc.flags & PxSceneFlag::eENABLE_GPU_DYNAMICS)
	{
		if(!mCudaContextManager)
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
	if (useGpuBroadphase || useGpuDynamics)
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
				*mSimpleIslandManager, contextID, desc.maxBiasCoefficient, desc.getTolerancesScale().length, mPublicFlags);
		}
		else
		{
			mDynamicsContext = createTGSDynamicsContext(&mLLContext->getNpMemBlockPool(), mLLContext->getTaskPool(), mLLContext->getSimStats(),
														*allocator, &getMaterialManager(), *mSimpleIslandManager, contextID,
														desc.getTolerancesScale().length, mPublicFlags);
		}

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
