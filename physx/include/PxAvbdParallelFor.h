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

#ifndef PX_AVBD_PARALLEL_FOR_H
#define PX_AVBD_PARALLEL_FOR_H

//=============================================================================
// Lightweight parallel_for for AVBD solver
//
// Uses a persistent thread pool that stays alive across calls. The pool
// spawns on first use and is destroyed at program exit.
//
// Falls back to sequential execution when count < threshold or when
// only 1 hardware thread is available.
//
// Threshold: per-particle VBD/BCD work is ~1-3us. Thread dispatch overhead
// is ~20-50us (condition_variable notify + wake). So we need at least
// ~128 particles before parallel dispatch outweighs the overhead.
//=============================================================================

#include "foundation/PxAllocator.h"
#include "foundation/PxArray.h"
#include "foundation/PxAtomic.h"
#include "foundation/PxMutex.h"
#include "foundation/PxSync.h"
#include "foundation/PxThread.h"

namespace physx {
namespace Dy {

// Minimum work items to justify parallelism overhead.
// Per-item VBD work ~1-3us, dispatch overhead ~20-50us -> need >=128 items
// for parallel dispatch to pay off.
static constexpr unsigned AVBD_PARALLEL_MIN_ITEMS = 128;

//-----------------------------------------------------------------------------
// Simple thread pool with chunked dispatch
//-----------------------------------------------------------------------------
class AvbdThreadPool {
  struct SharedState {
    volatile PxI32 generation;
    volatile PxI32 workersFinished;
    volatile PxI32 nextChunk;
    volatile PxI32 shutdown;
    volatile PxI32 activeWorkers;
    unsigned workBegin;
    unsigned workEnd;
    unsigned chunkSize;
    void (*func)(void *, unsigned);
    void *funcData;

    SharedState()
        : generation(0), workersFinished(0), nextChunk(0), shutdown(0),
          activeWorkers(0), workBegin(0), workEnd(0), chunkSize(1),
          func(NULL), funcData(NULL) {}
  };

  class WorkerThread : public PxThread {
  public:
    WorkerThread(SharedState &state, PxSync &startEvent)
        : PxThread(), mState(state), mStartEvent(startEvent) {}

    virtual void execute() {
      PxI32 localGeneration = 0;
      for (;;) {
        for (;;) {
          if (PxAtomicCompareExchange(&mState.shutdown, 0, 0) != 0) {
            return;
          }

          const PxI32 generation = PxAtomicCompareExchange(&mState.generation, 0, 0);
          if (generation != localGeneration) {
            localGeneration = generation;
            break;
          }

          mStartEvent.wait(1);
        }

        processChunks(mState);
        PxAtomicIncrement(&mState.workersFinished);
      }
    }

  private:
    SharedState &mState;
    PxSync &mStartEvent;
  };

  struct ParallelForData {
    const void *func;
    void (*invoke)(const void *, unsigned);
  };

public:
  static AvbdThreadPool &instance() {
    static AvbdThreadPool pool;
    return pool;
  }

  unsigned numWorkers() const { return mNumWorkers; }

  template <typename Func>
  void parallelFor(unsigned begin, unsigned end, const Func &func) {
    if (begin >= end)
      return;

    const unsigned count = end - begin;
    if (count == 1 || mNumWorkers == 0) {
      for (unsigned i = begin; i < end; ++i)
        func(i);
      return;
    }

    const unsigned minChunkSize = 32;
    unsigned maxWorkers = (count + minChunkSize - 1) / minChunkSize;
    if (maxWorkers > mNumWorkers)
      maxWorkers = mNumWorkers;

    const unsigned activeWorkers = maxWorkers;
    const unsigned totalThreads = activeWorkers + 1;
    const unsigned chunkSize = (count + totalThreads - 1) / totalThreads;

    ParallelForData data;
    data.func = &func;
    data.invoke = &invokeFunc<Func>;

    mState.func = &invokeThunk;
    mState.funcData = &data;
    mState.workBegin = begin;
    mState.workEnd = end;
    mState.chunkSize = chunkSize;
    PxAtomicExchange(&mState.nextChunk, 0);
    PxAtomicExchange(&mState.workersFinished, 0);
    PxAtomicExchange(&mState.activeWorkers, static_cast<PxI32>(activeWorkers));

    {
      PxMutex::ScopedLock lock(mMutex);
      PxAtomicIncrement(&mState.generation);
      mStartEvent.set();
    }

    processChunks(mState);

    while (static_cast<unsigned>(PxAtomicCompareExchange(&mState.workersFinished, 0, 0)) < activeWorkers) {
      PxThread::yield();
    }

    {
      PxMutex::ScopedLock lock(mMutex);
      mStartEvent.reset();
    }

    mState.func = NULL;
    mState.funcData = NULL;
  }

private:
  AvbdThreadPool() : mWorkers(), mNumWorkers(0), mMutex(), mStartEvent() {
    unsigned hwThreads = PxThread::getNbPhysicalCores();
    if (hwThreads == 0)
      hwThreads = 1;

    mNumWorkers = (hwThreads > 1) ? (hwThreads - 1) : 0;
    if (mNumWorkers > 15)
      mNumWorkers = 15;

    mWorkers.reserve(mNumWorkers);
    for (unsigned i = 0; i < mNumWorkers; ++i) {
      WorkerThread *worker = PX_NEW(WorkerThread)(mState, mStartEvent);
      mWorkers.pushBack(worker);
      worker->start();
    }
  }

  ~AvbdThreadPool() {
    PxAtomicExchange(&mState.shutdown, 1);
    {
      PxMutex::ScopedLock lock(mMutex);
      PxAtomicIncrement(&mState.generation);
      mStartEvent.set();
    }

    for (PxU32 i = 0; i < mWorkers.size(); ++i) {
      mWorkers[i]->waitForQuit();
      PX_DELETE(mWorkers[i]);
    }
  }

  AvbdThreadPool(const AvbdThreadPool &) = delete;
  AvbdThreadPool &operator=(const AvbdThreadPool &) = delete;

  static PX_FORCE_INLINE void processChunks(SharedState &state) {
    for (;;) {
      const PxI32 chunkIdx = PxAtomicIncrement(&state.nextChunk) - 1;
      const unsigned chunkBegin = state.workBegin + static_cast<unsigned>(chunkIdx) * state.chunkSize;
      if (chunkBegin >= state.workEnd)
        break;

      unsigned chunkEnd = chunkBegin + state.chunkSize;
      if (chunkEnd > state.workEnd)
        chunkEnd = state.workEnd;

      for (unsigned i = chunkBegin; i < chunkEnd; ++i) {
        state.func(state.funcData, i);
      }
    }
  }

  static void invokeThunk(void *funcData, unsigned i) {
    ParallelForData *data = reinterpret_cast<ParallelForData *>(funcData);
    data->invoke(data->func, i);
  }

  template <typename Func>
  static void invokeFunc(const void *func, unsigned i) {
    (*reinterpret_cast<const Func *>(func))(i);
  }

  SharedState mState;
  PxArray<WorkerThread *> mWorkers;
  unsigned mNumWorkers;
  PxMutex mMutex;
  PxSync mStartEvent;
};

//-----------------------------------------------------------------------------
// avbdParallelFor(begin, end, func)
//
// Executes func(i) for i in [begin, end), distributing work to a thread pool.
// Falls back to sequential when work is trivial.
//-----------------------------------------------------------------------------
template <typename Func>
inline void avbdParallelFor(unsigned begin, unsigned end, const Func &func) {
  const unsigned count = end - begin;
  if (count < AVBD_PARALLEL_MIN_ITEMS ||
      AvbdThreadPool::instance().numWorkers() == 0) {
    for (unsigned i = begin; i < end; ++i)
      func(i);
    return;
  }
  AvbdThreadPool::instance().parallelFor(begin, end, func);
}

//-----------------------------------------------------------------------------
// Atomic float max helper for parallel reductions
//-----------------------------------------------------------------------------
inline void avbdAtomicMaxFloat(volatile PxI32 &targetBits, float value) {
  union FloatBits {
    float f;
    PxI32 i;
  } valueBits, expectedBits, desiredBits;

  valueBits.f = value;
  expectedBits.i = PxAtomicCompareExchange(&targetBits, 0, 0);

  while (value > expectedBits.f) {
    desiredBits.i = valueBits.i;
    const PxI32 original = PxAtomicCompareExchange(&targetBits, desiredBits.i, expectedBits.i);
    if (original == expectedBits.i) {
      break;
    }
    expectedBits.i = original;
  }
}

} // namespace Dy
} // namespace physx

#endif // PX_AVBD_PARALLEL_FOR_H
