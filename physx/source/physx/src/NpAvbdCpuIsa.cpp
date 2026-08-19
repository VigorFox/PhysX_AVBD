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

#include "PxAvbdCpuIsa.h"
#include "../../lowleveldynamics/src/DyAvbdCpuIsa.h"

#include <cstring>

namespace physx {
namespace {

static PxU32 getAvbdCpuIsaModeCode(const char* mode)
{
	if(std::strcmp(mode, "sse2") == 0)
		return PxAvbdCpuIsaMode::eSSE2;
	if(std::strcmp(mode, "avx2fma") == 0)
		return PxAvbdCpuIsaMode::eAVX2_FMA;
	if(std::strcmp(mode, "invalid") == 0)
		return PxAvbdCpuIsaMode::eINVALID;
	return PxAvbdCpuIsaMode::eAUTO;
}

static PxU32 getAvbdCpuIsaCompiledBackendMask(const Dy::AvbdCpuIsaDispatch& dispatch)
{
	PxU32 mask = PxAvbdCpuIsaBackendFlag::eSSE2;
	if(dispatch.avx2FmaBackendCompiled)
		mask |= PxAvbdCpuIsaBackendFlag::eAVX2_FMA;
	return mask;
}

static PxU32 getAvbdCpuIsaCapabilityMask(const Dy::AvbdCpuIsaCapabilities& capabilities)
{
	return (capabilities.sse2 ? PxAvbdCpuIsaCapabilityFlag::eSSE2 : 0u) |
		(capabilities.avx ? PxAvbdCpuIsaCapabilityFlag::eAVX : 0u) |
		(capabilities.osxsave ? PxAvbdCpuIsaCapabilityFlag::eOSXSAVE : 0u) |
		(capabilities.xmmYmmState ? PxAvbdCpuIsaCapabilityFlag::eXMM_YMM_STATE : 0u) |
		(capabilities.avx2 ? PxAvbdCpuIsaCapabilityFlag::eAVX2 : 0u) |
		(capabilities.fma ? PxAvbdCpuIsaCapabilityFlag::eFMA : 0u);
}

} // namespace

PX_C_EXPORT PX_PHYSX_CORE_API void PX_CALL_CONV
PxGetAvbdCpuIsaTelemetry(PxAvbdCpuIsaTelemetry& telemetry)
{
	const Dy::AvbdCpuIsaDispatch& dispatch = Dy::getAvbdCpuIsaDispatch();
	telemetry.requestedIsa = getAvbdCpuIsaModeCode(dispatch.requestedIsa);
	telemetry.selectedIsa = getAvbdCpuIsaModeCode(dispatch.selectedIsa);
	telemetry.compiledBackendMask = getAvbdCpuIsaCompiledBackendMask(dispatch);
	telemetry.capabilityMask = getAvbdCpuIsaCapabilityMask(dispatch.capabilities);
	telemetry.forceModeRejected = dispatch.forceModeRejected ? 1u : 0u;
	telemetry.kernelSelfTestPassed = dispatch.kernelSelfTestPassed ? 1u : 0u;
	telemetry.fmaUsed = dispatch.fmaUsed ? 1u : 0u;
	telemetry.kernelSelfTestValue = dispatch.kernelSelfTestValue;
}

} // namespace physx

namespace physx {
namespace Dy {

extern "C" PX_PHYSX_CORE_API AvbdCpuIsaCorotationalTetPacket8Fn
PX_CALL_CONV PxAvbdCpuIsaCorotationalTetPacket8FunctionInternal()
{
	return getAvbdCpuIsaFunctionTable().corotationalTetPacket8;
}

} // namespace Dy
} // namespace physx
