// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_MATERIAL_H
#define DY_AVBD_CONTACT_MATERIAL_H

#include "PxMaterial.h"
#include "foundation/PxMath.h"

namespace physx
{
namespace Dy
{

PX_FORCE_INLINE PxReal avbdCombineDeformableRigidFriction(
	PxReal deformableFriction, PxReal rigidFriction,
	PxU8 frictionCombineMode)
{
	const PxReal soft = PxMax(deformableFriction, 0.0f);
	const PxReal rigid = PxMax(rigidFriction, 0.0f);
	switch(static_cast<PxCombineMode::Enum>(frictionCombineMode))
	{
	case PxCombineMode::eMIN:
		return PxMin(soft, rigid);
	case PxCombineMode::eMULTIPLY:
		return soft * rigid;
	case PxCombineMode::eMAX:
		return PxMax(soft, rigid);
	case PxCombineMode::eAVERAGE:
	default:
		return 0.5f * (soft + rigid);
	}
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_MATERIAL_H
