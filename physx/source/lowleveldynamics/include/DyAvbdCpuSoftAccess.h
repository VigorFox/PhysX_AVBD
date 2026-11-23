// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CPU_SOFT_ACCESS_H
#define DY_AVBD_CPU_SOFT_ACCESS_H

// Thin Dy-facing diagnostics for the AVBD backend (no Scene *Avbd* methods).
// Deformable actors enter through the regular Host/IG path; there is deliberately
// no attach/fetch sidecar API here.

#include "foundation/PxVec3.h"
#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Dy
{

class Context;

// P5 Kernel table (Dy API). deviceKind 0 = SCALAR. CUDA slots stay unbound.
bool avbdKernelTableInfo(Context* ctx, PxU32& deviceKind, bool& ldlt6Bound, bool& ldlt3Bound);
bool avbdKernelLdltSelfCheck();

} // namespace Dy
} // namespace physx

#endif
