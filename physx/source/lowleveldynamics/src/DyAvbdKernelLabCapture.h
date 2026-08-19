// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Opt-in diagnostic capture for avbd_kernel_lab.  This object never
// participates in solving: update() reserves its storage before the island
// task is submitted, the color barrier only copies fields into that storage,
// and mergeResults() performs any file I/O after the task graph is complete.

#ifndef DY_AVBD_KERNEL_LAB_CAPTURE_H
#define DY_AVBD_KERNEL_LAB_CAPTURE_H

#include "DyAvbdSolver.h"
#include "DyAvbdKernelLabTraceContract.h"
#include "foundation/PxArray.h"

namespace physx {
namespace Dy {

class AvbdKernelLabCapture {
public:
  static const PxU32 eINVALID_TICKET = PX_MAX_U32;

  AvbdKernelLabCapture();
  ~AvbdKernelLabCapture();

  // Called by the serial update thread before it submits an island task.  A
  // single exact island target owns the reservation, avoiding hot-path locks
  // and making capture capacity failure fail closed before workers run.
  PxU32 reserve(PxU32 islandIndex, PxU32 bodyCount, PxU32 contactCount);

  // Called at submitRigidColor() before the range is run.  This performs only
  // bounds checks and field copies into arrays reserved by reserve().
  void capturePreRange(PxU32 ticket, const AvbdRigidSolveContext& context,
                       const AvbdSolverConfig& config, PxU32 islandIndex,
                       PxU32 colorIndex, const PxU32* ownerOrder,
                       PxU32 begin, PxU32 end, PxU32 workerCount,
                       PxU32 taskGrainBodies, PxU32 taskCount,
                       PxU32 taskChunkBodies);

  // Called after the same color's synchronous run or child fan-in.  It only
  // copies the selected owners' poses; it never reads or writes a live row.
  void capturePostRange(PxU32 ticket, const AvbdRigidSolveContext& context,
                        PxU32 colorIndex);

  // Called from mergeResults() after all AVBD tasks have finished.  The only
  // file I/O lives here, outside the worker and solver range hot paths.
  void flush();

private:
  enum State {
    eARMED,
    eRESERVED,
    ePRE_RANGE_READY,
    eREADY_TO_WRITE,
    eREJECTED,
    eWRITTEN,
    eWRITE_FAILED
  };

  bool reserveArrays(PxU32 bodyCount, PxU32 contactCount, PxU32 maxRefs);
  bool finalizePayloadHashes();
  bool writeFile();
  void reject();

  State mState;
  PxU32 mTargetIsland;
  PxU32 mTargetIteration;
  PxU32 mTargetColor;
  PxU32 mMinimumBodies;
  PxU32 mTicket;
  PxU32 mReservedBodyCount;
  PxU32 mReservedContactCount;
  PxU32 mReservedRefCount;
  bool mOutputPathValid;
  char mOutputPath[512];
  AvbdKernelLabTraceHeader mHeader;
  PxArray<AvbdKernelLabTraceBody> mBodies;
  PxArray<AvbdKernelLabTraceContact> mContacts;
  PxArray<PxU32> mMapOffsets;
  PxArray<PxU32> mMapCounts;
  PxArray<PxU32> mMapIndices;
  PxArray<PxU32> mColorOffsets;
  PxArray<PxU32> mColorBodies;
  PxArray<PxU32> mOwnerOrder;
  PxArray<AvbdKernelLabTraceOwnerPose> mPostOwnerPoses;
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_KERNEL_LAB_CAPTURE_H
