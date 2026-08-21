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

#include "avbd/backend/cpu/DyAvbdCpuIsa.h"

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace physx {
namespace Dy {

PxF32 avbdCpuIsaSse2ProbeDot8(const PxF32* lhs, const PxF32* rhs);

#if defined(PX_AVBD_CPU_AVX2_FMA_COMPILED)
PxF32 avbdCpuIsaAvx2FmaProbeDot8(const PxF32* lhs, const PxF32* rhs);
void avbdCpuIsaAvx2FmaCorotationalTetPacket8(
	const AvbdTetMaterialPacket8Input& input,
	PxF32 mu, PxF32 lambda,
	AvbdTetMaterialPacket8Output& output);
void avbdCpuIsaAvx2FmaNeoHookeanTetPacket8(
	const AvbdTetMaterialPacket8Input& input,
	PxF32 mu, PxF32 lambda, PxF32 alpha,
	AvbdTetMaterialPacket8Output& output);
#endif

namespace {

struct AvbdCpuIsaState {
	AvbdCpuIsaDispatch dispatch;
	AvbdCpuIsaFunctionTable functionTable;
};

static AvbdCpuIsaCapabilities detectAvbdCpuIsaCapabilities()
{
	AvbdCpuIsaCapabilities capabilities = { false, false, false, false, false, false };

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
	int registers[4] = { 0, 0, 0, 0 };
	__cpuidex(registers, 0, 0);
	const int maxBasicLeaf = registers[0];
	if(maxBasicLeaf < 1)
		return capabilities;

	__cpuidex(registers, 1, 0);
	const PxU32 ecx = PxU32(registers[2]);
	const PxU32 edx = PxU32(registers[3]);
	capabilities.sse2 = (edx & (1u << 26)) != 0;
	capabilities.avx = (ecx & (1u << 28)) != 0;
	capabilities.osxsave = (ecx & (1u << 27)) != 0;
	capabilities.fma = (ecx & (1u << 12)) != 0;
	if(capabilities.avx && capabilities.osxsave)
	{
		const unsigned __int64 xcr0 = _xgetbv(0);
		capabilities.xmmYmmState = (xcr0 & 0x6u) == 0x6u;
	}
	if(maxBasicLeaf >= 7)
	{
		__cpuidex(registers, 7, 0);
		capabilities.avx2 = (PxU32(registers[1]) & (1u << 5)) != 0;
	}
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
	const unsigned int maxBasicLeaf = __get_cpuid_max(0, 0);
	if(maxBasicLeaf < 1)
		return capabilities;

	unsigned int eax = 0;
	unsigned int ebx = 0;
	unsigned int ecx = 0;
	unsigned int edx = 0;
	__cpuid_count(1, 0, eax, ebx, ecx, edx);
	capabilities.sse2 = (edx & (1u << 26)) != 0;
	capabilities.avx = (ecx & (1u << 28)) != 0;
	capabilities.osxsave = (ecx & (1u << 27)) != 0;
	capabilities.fma = (ecx & (1u << 12)) != 0;
	if(capabilities.avx && capabilities.osxsave)
	{
		unsigned int xcr0Low = 0;
		unsigned int xcr0High = 0;
		__asm__ volatile("xgetbv" : "=a"(xcr0Low), "=d"(xcr0High) : "c"(0));
		const PxU64 xcr0 = PxU64(xcr0Low) | (PxU64(xcr0High) << 32);
		capabilities.xmmYmmState = (xcr0 & 0x6u) == 0x6u;
	}
	if(maxBasicLeaf >= 7)
	{
		__cpuid_count(7, 0, eax, ebx, ecx, edx);
		capabilities.avx2 = (ebx & (1u << 5)) != 0;
	}
#endif

	return capabilities;
}

static bool isRequestedIsa(const char* value, const char* expected)
{
	return value && std::strcmp(value, expected) == 0;
}

static bool isEnvFlagEnabled(const char* name)
{
	const char* value = std::getenv(name);
	return value && value[0] && value[0] != '0';
}

static const char* getRequestedIsaName(const char* value)
{
	if(!value || !value[0] || isRequestedIsa(value, "auto"))
		return "auto";
	if(isRequestedIsa(value, "sse2"))
		return "sse2";
	if(isRequestedIsa(value, "avx2fma"))
		return "avx2fma";
	return "invalid";
}

static bool hasCompiledAvx2FmaBackend()
{
#if defined(PX_AVBD_CPU_AVX2_FMA_COMPILED)
	return true;
#else
	return false;
#endif
}

static const char* getCompiledIsaBackendNames()
{
	return hasCompiledAvx2FmaBackend() ? "sse2,avx2fma" : "sse2";
}

static void selectSse2Backend(AvbdCpuIsaState& state)
{
	state.dispatch.selectedIsa = "sse2";
	state.functionTable.probeDot8 = avbdCpuIsaSse2ProbeDot8;
	state.functionTable.corotationalTetPacket8 = NULL;
	state.functionTable.neoHookeanTetPacket8 = NULL;
}

static void selectAvx2FmaBackend(AvbdCpuIsaState& state)
{
#if defined(PX_AVBD_CPU_AVX2_FMA_COMPILED)
	state.dispatch.selectedIsa = "avx2fma";
	state.functionTable.probeDot8 = avbdCpuIsaAvx2FmaProbeDot8;
	state.functionTable.corotationalTetPacket8 =
		avbdCpuIsaAvx2FmaCorotationalTetPacket8;
	state.functionTable.neoHookeanTetPacket8 =
		avbdCpuIsaAvx2FmaNeoHookeanTetPacket8;
#else
	selectSse2Backend(state);
#endif
}

static AvbdCpuIsaState initializeAvbdCpuIsaState()
{
	AvbdCpuIsaState state;
	state.dispatch.requestedIsa = getRequestedIsaName(std::getenv("PX_AVBD_CPU_ISA"));
	state.dispatch.selectedIsa = "sse2";
	state.dispatch.compiledIsaBackends = getCompiledIsaBackendNames();
	state.dispatch.capabilities = detectAvbdCpuIsaCapabilities();
	if(isEnvFlagEnabled("PX_AVBD_CPU_ISA_TEST_DISABLE_AVX2_FMA"))
	{
		// Test-only fault injection. This is deliberately applied before the
		// function table is selected; it cannot execute the wide backend.
		state.dispatch.capabilities.avx2 = false;
		state.dispatch.capabilities.fma = false;
	}
	state.dispatch.avx2FmaBackendCompiled = hasCompiledAvx2FmaBackend();
	state.dispatch.forceModeRejected = false;
	state.dispatch.kernelSelfTestPassed = false;
	state.dispatch.fmaUsed = false;
	state.dispatch.kernelSelfTestValue = 0.0f;
	selectSse2Backend(state);

	const bool avx2FmaExecutable = hasCompiledAvx2FmaBackend() &&
		state.dispatch.capabilities.hasAvx2FmaBackendSupport();
	if(isRequestedIsa(state.dispatch.requestedIsa, "avx2fma"))
	{
		if(avx2FmaExecutable)
			selectAvx2FmaBackend(state);
		else
			state.dispatch.forceModeRejected = true;
	}
	else if(isRequestedIsa(state.dispatch.requestedIsa, "auto"))
	{
		if(avx2FmaExecutable)
			selectAvx2FmaBackend(state);
	}
	else if(!isRequestedIsa(state.dispatch.requestedIsa, "sse2"))
	{
		state.dispatch.forceModeRejected = true;
	}

	const PxF32 lhs[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
	const PxF32 rhs[8] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
	state.dispatch.kernelSelfTestValue = state.functionTable.probeDot8(lhs, rhs);
	state.dispatch.kernelSelfTestPassed = state.dispatch.kernelSelfTestValue == 36.0f;
	state.dispatch.fmaUsed =
		isRequestedIsa(state.dispatch.selectedIsa, "avx2fma") &&
		state.dispatch.kernelSelfTestPassed;

	// A failing probe must never leave an AVX/FMA function table installed.
	if(!state.dispatch.kernelSelfTestPassed &&
		isRequestedIsa(state.dispatch.selectedIsa, "avx2fma"))
	{
		selectSse2Backend(state);
		state.dispatch.forceModeRejected = true;
		state.dispatch.kernelSelfTestValue = state.functionTable.probeDot8(lhs, rhs);
		state.dispatch.kernelSelfTestPassed = state.dispatch.kernelSelfTestValue == 36.0f;
		state.dispatch.fmaUsed = false;
	}

	return state;
}

static const AvbdCpuIsaState& getAvbdCpuIsaState()
{
	static const AvbdCpuIsaState state = initializeAvbdCpuIsaState();
	return state;
}

} // namespace

const AvbdCpuIsaDispatch& getAvbdCpuIsaDispatch()
{
	return getAvbdCpuIsaState().dispatch;
}

const AvbdCpuIsaFunctionTable& getAvbdCpuIsaFunctionTable()
{
	return getAvbdCpuIsaState().functionTable;
}

} // namespace Dy
} // namespace physx
