// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#ifndef DY_AVBD_SOFT_BODY_SCHEDULING_H
#define DY_AVBD_SOFT_BODY_SCHEDULING_H

#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_SCHEDULING_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_SCHEDULING_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_SCHEDULING_API
#endif

enum class AvbdParticlePrimalSchedule : PxU8
{
	eDEFAULT,
	eSERIAL_LINEAR,
	eCOLORED_SERIAL,
	eRELAXED_COLOR
};

PX_FORCE_INLINE bool avbdUsesColoredParticlePrimalSchedule(
	AvbdParticlePrimalSchedule schedule)
{
	return schedule == AvbdParticlePrimalSchedule::eCOLORED_SERIAL ||
		schedule == AvbdParticlePrimalSchedule::eRELAXED_COLOR;
}

DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdUsePersistentStepStateSerial();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdUseCausalLayerTaskFanIn();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdDisableIndependentBodySweepTaskFanIn();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdForceCausalLayerTaskFanIn();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdForceCausalLayerTaskGraphReference();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdUseCausalLayerTaskPartition();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdForceCausalLayerTaskPartition();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdUseSceneRedetectionBridge();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdUseWorldPlaneContactTaskFanIn();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdUseRigidBoxSdfContactTaskFanIn();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdUseRigidSphereSdfContactTaskFanIn();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdUseRigidCapsuleSdfContactTaskFanIn();
DY_AVBD_SOFT_BODY_SCHEDULING_API bool avbdUseRigidConvexSdfContactTaskFanIn();

#undef DY_AVBD_SOFT_BODY_SCHEDULING_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_SCHEDULING_H
