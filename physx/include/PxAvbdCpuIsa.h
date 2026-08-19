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
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#ifndef PX_AVBD_CPU_ISA_H
#define PX_AVBD_CPU_ISA_H

#include "PxPhysXConfig.h"
#include "foundation/PxSimpleTypes.h"

namespace physx {

/** AVBD CPU ISA request and selected-backend codes. */
struct PxAvbdCpuIsaMode
{
	enum Enum
	{
		eAUTO = 0,
		eSSE2 = 1,
		eAVX2_FMA = 2,
		eINVALID = 3
	};
};

/** Bit values used by PxAvbdCpuIsaTelemetry::compiledBackendMask. */
struct PxAvbdCpuIsaBackendFlag
{
	enum Enum
	{
		eSSE2 = 1 << 0,
		eAVX2_FMA = 1 << 1
	};
};

/** Bit values used by PxAvbdCpuIsaTelemetry::capabilityMask. */
struct PxAvbdCpuIsaCapabilityFlag
{
	enum Enum
	{
		eSSE2 = 1 << 0,
		eAVX = 1 << 1,
		eOSXSAVE = 1 << 2,
		eXMM_YMM_STATE = 1 << 3,
		eAVX2 = 1 << 4,
		eFMA = 1 << 5
	};
};

/**
\brief Process-wide AVBD CPU ISA dispatch state.

The dispatch decision is made once on first use. This diagnostic snapshot is
read-only and is also available to standalone AVBD component paths that do not
own a PxScene or PxSimulationStatistics instance.
*/
struct PxAvbdCpuIsaTelemetry
{
	PxU32	requestedIsa;		//!< PxAvbdCpuIsaMode::Enum.
	PxU32	selectedIsa;		//!< PxAvbdCpuIsaMode::Enum.
	PxU32	compiledBackendMask;	//!< PxAvbdCpuIsaBackendFlag::Enum bits.
	PxU32	capabilityMask;	//!< PxAvbdCpuIsaCapabilityFlag::Enum bits.
	PxU32	forceModeRejected;	//!< Nonzero if a forced request failed closed.
	PxU32	kernelSelfTestPassed;	//!< Nonzero if the selected probe self-test passed.
	PxU32	fmaUsed;		//!< Nonzero only when AVX2+FMA is selected and tested.
	PxReal	kernelSelfTestValue;	//!< Selected isolated probe result (36.0f on pass).

	PX_FORCE_INLINE PxAvbdCpuIsaTelemetry()
	:	requestedIsa(PxAvbdCpuIsaMode::eAUTO),
		selectedIsa(PxAvbdCpuIsaMode::eAUTO),
		compiledBackendMask(0),
		capabilityMask(0),
		forceModeRejected(0),
		kernelSelfTestPassed(0),
		fmaUsed(0),
		kernelSelfTestValue(0.0f)
	{
	}
};

/** Retrieve the once-selected process-wide AVBD CPU ISA telemetry snapshot. */
PX_C_EXPORT PX_PHYSX_CORE_API void PX_CALL_CONV
PxGetAvbdCpuIsaTelemetry(PxAvbdCpuIsaTelemetry& telemetry);

} // namespace physx

#endif // PX_AVBD_CPU_ISA_H
