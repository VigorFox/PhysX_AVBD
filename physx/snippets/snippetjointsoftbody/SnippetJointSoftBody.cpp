// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "SnippetJointSoftBodyApp.h"
int snippetMain(int argc,const char*const* argv)
{
	return JointSoftBodySnippet::runApplication(argc,argv,false);
}
