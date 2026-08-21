// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_SCENE_ENTRIES_H
#define SC_AVBD_SCENE_ENTRIES_H

#include "ScActorCore.h"
#include "ScBodyCore.h"
#include "ScDeformableSurfaceCore.h"
#include "ScDeformableVolumeCore.h"
#include "ScRigidCore.h"
#include "ScShapeCore.h"
#include "ScStaticCore.h"
#include "DyDeformableSurface.h"
#include "DyDeformableVolume.h"
#include "DyIslandManager.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"

namespace physx
{
class PxDeformableVolumeAuxData;
class PxTetrahedronMesh;
class PxTriangleMesh;

namespace Sc
{

enum AvbdSceneEntryKind
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
			AvbdSceneEntryKind			kind;
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

} // namespace Sc
} // namespace physx

#endif // SC_AVBD_SCENE_ENTRIES_H
