#pragma once
#include "foundation/PxFoundation.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"
#include "extensions/PxDefaultAllocator.h"
#include <cstdio>
#include <cstdlib>

// Declare before any SDK containers, so their destructors run first.
struct AvbdProbeFoundation : physx::PxErrorCallback
{
    physx::PxDefaultAllocator allocator;
    physx::PxFoundation* foundation;
    AvbdProbeFoundation() : foundation(PxCreateFoundation(PX_PHYSICS_VERSION, allocator, *this))
    { if(!foundation) std::exit(2); }
    ~AvbdProbeFoundation() { foundation->release(); }
    void reportError(physx::PxErrorCode::Enum, const char* message, const char*, int) override
    { std::fprintf(stderr, "%s\n", message); std::exit(2); }
};
