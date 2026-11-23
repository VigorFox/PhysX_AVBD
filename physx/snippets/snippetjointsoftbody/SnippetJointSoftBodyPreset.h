// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace JointSoftBodySnippet
{
inline bool setProcessOption(const char* name,const char* value)
{
#ifdef _WIN32
	return _putenv_s(name,value)==0;
#else
	return setenv(name,value,1)==0;
#endif
}
// Shared application preset. Bounded is an explicit numerical reference only.
// This does not change the SDK-wide solver default or provide a fallback.
inline bool configureNativeJointStep()
{
	const char* mode=std::getenv("PHYSX_AVBD_COMMON_STEP");
	if(mode && std::strcmp(mode,"bounded") && std::strcmp(mode,"budget"))
	{
		std::fprintf(stderr,"Native joint preset requires COMMON_STEP=bounded or budget.\n");return false;
	}
	return (std::getenv("PHYSX_AVBD_COMMON_STEP") || setProcessOption("PHYSX_AVBD_COMMON_STEP","budget")) &&
		(std::getenv("PHYSX_AVBD_LOCAL_ITERATIONS") || setProcessOption("PHYSX_AVBD_LOCAL_ITERATIONS","4")) &&
		setProcessOption("PHYSX_NP_GEOMETRY_CERTIFICATES","2");
}
}
