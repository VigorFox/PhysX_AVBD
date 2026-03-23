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

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

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
public:
  static AvbdThreadPool &instance() {
    static AvbdThreadPool pool;
    return pool;
  }

  unsigned numWorkers() const { return mNumWorkers; }

  // Execute func(i) for i in [begin, end) across worker threads.
  // Blocks until all work is done.
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

    // Determine how many workers to actually use (don't wake more than needed)
    const unsigned minChunkSize = 32;
    unsigned maxWorkers = (count + minChunkSize - 1) / minChunkSize;
    if (maxWorkers > mNumWorkers)
      maxWorkers = mNumWorkers;
    const unsigned activeWorkers = maxWorkers;
    const unsigned totalThreads = activeWorkers + 1; // workers + caller
    const unsigned chunkSize =
        (count + totalThreads - 1) / totalThreads;

    mFunc = [&func](unsigned i) { func(i); };
    mWorkBegin = begin;
    mWorkEnd = end;
    mChunkSize = chunkSize;
    mNextChunk.store(0, std::memory_order_relaxed);
    mActiveWorkers = activeWorkers;
    mWorkersFinished.store(0, std::memory_order_relaxed);

    {
      std::lock_guard<std::mutex> lock(mMutex);
      mGeneration++;
    }
    // Wake only the needed workers
    if (activeWorkers >= mNumWorkers)
      mCondVar.notify_all();
    else
      for (unsigned w = 0; w < activeWorkers; ++w)
        mCondVar.notify_one();

    processChunks();

    // Spin-wait for active workers only
    while (mWorkersFinished.load(std::memory_order_acquire) < activeWorkers) {
      std::this_thread::yield();
    }

    mFunc = nullptr;
  }

private:
  AvbdThreadPool() {
    unsigned hwThreads = std::thread::hardware_concurrency();
    mNumWorkers = (hwThreads > 1) ? (hwThreads - 1) : 0;
    if (mNumWorkers > 15)
      mNumWorkers = 15;

    mShutdown = false;
    mGeneration = 0;
    mActiveWorkers = 0;

    for (unsigned i = 0; i < mNumWorkers; ++i) {
      mWorkers.emplace_back([this] { workerLoop(); });
    }
  }

  ~AvbdThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mMutex);
      mShutdown = true;
      mGeneration++;
    }
    mCondVar.notify_all();
    for (auto &t : mWorkers)
      t.join();
  }

  AvbdThreadPool(const AvbdThreadPool &) = delete;
  AvbdThreadPool &operator=(const AvbdThreadPool &) = delete;

  void processChunks() {
    for (;;) {
      unsigned chunkIdx =
          mNextChunk.fetch_add(1, std::memory_order_relaxed);
      unsigned chunkBegin = mWorkBegin + chunkIdx * mChunkSize;
      if (chunkBegin >= mWorkEnd)
        break;
      unsigned chunkEnd = chunkBegin + mChunkSize;
      if (chunkEnd > mWorkEnd)
        chunkEnd = mWorkEnd;
      for (unsigned i = chunkBegin; i < chunkEnd; ++i)
        mFunc(i);
    }
  }

  void workerLoop() {
    unsigned localGen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mMutex);
        mCondVar.wait(lock,
                      [&] { return mGeneration != localGen || mShutdown; });
        if (mShutdown)
          return;
        localGen = mGeneration;
      }
      processChunks();
      mWorkersFinished.fetch_add(1, std::memory_order_release);
    }
  }

  unsigned mNumWorkers;
  std::vector<std::thread> mWorkers;
  std::mutex mMutex;
  std::condition_variable mCondVar;
  bool mShutdown;
  unsigned mGeneration;

  std::function<void(unsigned)> mFunc;
  unsigned mWorkBegin;
  unsigned mWorkEnd;
  unsigned mChunkSize;
  unsigned mActiveWorkers;
  std::atomic<unsigned> mNextChunk;
  std::atomic<unsigned> mWorkersFinished;
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
inline void avbdAtomicMaxFloat(std::atomic<float> &target, float value) {
  float expected = target.load(std::memory_order_relaxed);
  while (value > expected &&
         !target.compare_exchange_weak(expected, value,
                                       std::memory_order_relaxed)) {
  }
}

} // namespace Dy
} // namespace physx

#endif // PX_AVBD_PARALLEL_FOR_H
