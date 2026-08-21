// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#ifndef DY_AVBD_SOFT_BODY_TOPOLOGY_H
#define DY_AVBD_SOFT_BODY_TOPOLOGY_H

#include "foundation/PxArray.h"
#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Dy
{

struct AvbdParticleElementRef
{
	PxU32 index;
	PxU8 vOrder;
	PxU8 padding[3];
};

struct AvbdParticleElementAdjacency
{
	PxArray<AvbdParticleElementRef> triRefs;
	PxArray<AvbdParticleElementRef> tetRefs;
	PxArray<AvbdParticleElementRef> bendRefs;
};

// A packet lane is the matching scalar tetRefs ordinal. Packet evaluation is
// not allowed to redefine the canonical scalar reduction order.
static const PxU32 eAVBD_TET_INCIDENCE_PACKET_WIDTH = 8;

struct AvbdTetIncidencePacket8
{
	PxU32 tetIndices[eAVBD_TET_INCIDENCE_PACKET_WIDTH];
	PxU8 vertexOrders[eAVBD_TET_INCIDENCE_PACKET_WIDTH];
	PxU8 validMask;
	PxU8 padding[3];

	AvbdTetIncidencePacket8()
		: validMask(0), padding{0, 0, 0}
	{
		for(PxU32 lane = 0; lane < eAVBD_TET_INCIDENCE_PACKET_WIDTH;
			lane++)
		{
			tetIndices[lane] = PX_MAX_U32;
			vertexOrders[lane] = PX_MAX_U8;
		}
	}
};

struct AvbdTetIncidencePacketRange
{
	PxU32 packetStart;
	PxU32 packetCount;

	AvbdTetIncidencePacketRange() : packetStart(0), packetCount(0) {}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_TOPOLOGY_H
