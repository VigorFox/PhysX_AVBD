// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef SNIPPET_ALLOCATION_STATS_H
#define SNIPPET_ALLOCATION_STATS_H
#include "extensions/PxDefaultAllocator.h"
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
namespace Snippets
{
// Opt-in measurement runs only. Disabled runs retain the default allocator.
// The 16-byte prefix preserves the SDK allocation alignment. These counters
// cover callback allocations, not OS working set or standard-library storage.
class AllocationStats : public physx::PxAllocatorCallback
{
	physx::PxDefaultAllocator backing;
	bool enabled;
	std::atomic<physx::PxU64> live,peak,allocations;
public:
	AllocationStats() : enabled(std::getenv("PHYSX_SNIPPET_ALLOCATION_STATS") && std::strcmp(std::getenv("PHYSX_SNIPPET_ALLOCATION_STATS"),"1")==0),live(0),peak(0),allocations(0) {}
	void* allocate(size_t size,const char* name,const char* file,int line) PX_OVERRIDE
	{
		if(!enabled) return backing.allocate(size,name,file,line);
		void* raw=backing.allocate(size+16,name,file,line);if(!raw) return NULL;
		*static_cast<physx::PxU64*>(raw)=physx::PxU64(size);
		const physx::PxU64 current=live.fetch_add(size)+size;++allocations;
		physx::PxU64 previous=peak.load();while(previous<current && !peak.compare_exchange_weak(previous,current)) {}
		return static_cast<char*>(raw)+16;
	}
	void deallocate(void* ptr) PX_OVERRIDE
	{
		if(!enabled) { backing.deallocate(ptr);return; }
		if(!ptr) return;void* raw=static_cast<char*>(ptr)-16;live.fetch_sub(*static_cast<physx::PxU64*>(raw));backing.deallocate(raw);
	}
	void report(const char* phase) const
	{
		if(enabled) { std::printf("[SNIPPET_ALLOCATIONS] phase=%s liveBytes=%llu peakBytes=%llu allocations=%llu\n",phase,static_cast<unsigned long long>(live.load()),static_cast<unsigned long long>(peak.load()),static_cast<unsigned long long>(allocations.load()));std::fflush(stdout); }
	}
};
}
#endif
