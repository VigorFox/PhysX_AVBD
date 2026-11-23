// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../snippetjointsoftbody/SnippetJointSoftBodyPreset.h"
#include <vector>

namespace Snippets
{
struct DeformableAvbdLaunch
{
	enum Model { Particle3, Rigid6 };
	Model model = Particle3;
	std::vector<const char*> args;

	bool parse(int argc, const char* const* argv)
	{
		bool explicitModel = false;
		args.push_back(argv[0]);
		for(int i = 1; i < argc; ++i)
		{
			Model requested;
			if(!std::strcmp(argv[i], "--particle3")) requested = Particle3;
			else if(!std::strcmp(argv[i], "--rigid6")) requested = Rigid6;
			else if(!std::strcmp(argv[i], "--fem"))
            {
                std::fprintf(stderr, "The FEM+AVBD model has been retired. Select --particle3 or --rigid6.\n");
                return false;
            }
			else { args.push_back(argv[i]); continue; }
			if(explicitModel && model != requested)
			{
				std::fprintf(stderr, "Select one deformable model (--particle3 or --rigid6).\n");
				return false;
			}
			model = requested;
			explicitModel = true;
		}
		return true;
	}

	static bool configureParticle3()
	{
		using JointSoftBodySnippet::setProcessOption;
		return setProcessOption("PHYSX_AVBD_COMMON_STEP", "budget") &&
			setProcessOption("PHYSX_AVBD_LOCAL_ITERATIONS", "4") &&
			setProcessOption("PHYSX_NP_GEOMETRY_CERTIFICATES", "2");
	}
};
}
