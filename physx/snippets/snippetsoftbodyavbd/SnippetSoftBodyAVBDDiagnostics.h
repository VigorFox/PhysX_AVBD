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
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef PHYSX_SNIPPET_SOFT_BODY_AVBD_DIAGNOSTICS_H
#define PHYSX_SNIPPET_SOFT_BODY_AVBD_DIAGNOSTICS_H

#include "avbd/solver/soft/DyAvbdSoftBody.h"

namespace SnippetSoftBodyAVBDDiagnostics
{

bool isRotationTraceEnabled();
physx::PxU32 getRotationTraceInterval();

void captureBodyReferenceLocals(
	const physx::PxArray<physx::Dy::AvbdSoftParticle>& particles,
	const physx::PxArray<physx::Dy::AvbdSoftBody>& bodies,
	physx::PxArray<physx::PxArray<physx::PxVec3> >& refs);

void printBodyRotationTrace(
	const char* label,
	physx::PxU32 frame,
	const physx::PxArray<physx::Dy::AvbdSoftParticle>& particles,
	const physx::PxArray<physx::Dy::AvbdSoftBody>& bodies,
	const physx::PxArray<physx::PxArray<physx::PxVec3> >& refs);

} // namespace SnippetSoftBodyAVBDDiagnostics

#endif // PHYSX_SNIPPET_SOFT_BODY_AVBD_DIAGNOSTICS_H
