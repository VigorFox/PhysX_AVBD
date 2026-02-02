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

#include "DyAvbdTasks.h"
#include "DyAvbdBodyConversion.h"
#include "DyAvbdDynamics.h"
#include "PxsRigidBody.h"
#include "DyFeatherstoneArticulation.h"
#include "DyVArticulation.h"
#include <cstdio>

// Debug logging macro
#if defined(AVBD_ENABLE_LOG)
  #define AVBD_LOG(fmt, ...)                                                   \
    printf("[AVBD] " fmt "\n", ##__VA_ARGS__);                               \
    fflush(stdout)
#else
  #define AVBD_LOG(...) do { } while (0)
#endif

namespace physx {
namespace Dy {

void AvbdTask::release() {
  AVBD_LOG("Task release: %s", getName());
  // CRITICAL: Must call base class release() which calls
  // mCont->removeReference() to notify the continuation that this task is done.
  PxLightCpuTask::release();
  // After base class release, destroy ourselves
  mContext.destroyTask(this);
}

void AvbdSolveIslandTask::release() {
  // Release constraint maps to prevent memory leak
  // Each frame builds new maps, so we must free them when task completes
  PxAllocatorCallback& allocator = mContext.getAllocator();
  mBatch.contactMap.release(allocator);
  mBatch.sphericalMap.release(allocator);
  mBatch.fixedMap.release(allocator);
  mBatch.revoluteMap.release(allocator);
  mBatch.prismaticMap.release(allocator);
  mBatch.d6Map.release(allocator);
  
  // Call base class release
  AvbdTask::release();
}

void AvbdWriteBackTask::run() {
  AVBD_LOG("AvbdWriteBackTask::run() START - numBodies=%u", mNumBodies);
  
  for (PxU32 i = 0; i < mNumBodies; ++i) {
    if (mAvbdBodies[i].isStatic())
      continue;
      
    if (mRigidBodies[i]) {
      // Regular rigid body - writeback to body core
      writeBackAvbdSolverBody(mAvbdBodies[i], mRigidBodies[i]->getCore());
    }
    else if (mArticulationForBody && mLinkIndexForBody) {
      // Articulation link - writeback to articulation link body core
      FeatherstoneArticulation* articulation = mArticulationForBody[i];
      PxU32 linkIndex = mLinkIndexForBody[i];
      
      if (articulation && linkIndex != PX_MAX_U32) {
        ArticulationData& artData = articulation->getArticulationData();
        if (linkIndex < artData.getLinkCount()) {
          ArticulationLink& link = artData.getLink(linkIndex);
          PxsBodyCore* bodyCore = link.bodyCore;
          
          if (bodyCore) {
            // Write back position and rotation
            bodyCore->body2World.p = mAvbdBodies[i].position;
            bodyCore->body2World.q = mAvbdBodies[i].rotation;
            
            // Write back velocities
            bodyCore->linearVelocity = mAvbdBodies[i].linearVelocity;
            bodyCore->angularVelocity = mAvbdBodies[i].angularVelocity;
          }
        }
      }
    }
  }
  
  AVBD_LOG("AvbdWriteBackTask::run() END");
}

} // namespace Dy
} // namespace physx
