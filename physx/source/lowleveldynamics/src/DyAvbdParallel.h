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

#ifndef DY_AVBD_PARALLEL_H
#define DY_AVBD_PARALLEL_H

#include "DyAvbdConstraint.h"
#include "DyAvbdSolverBody.h"
#include "DyAvbdTypes.h"
#include "foundation/PxAllocatorCallback.h"
#include "foundation/PxArray.h"

namespace physx {
namespace Dy {

/**
 * @brief Color batch for parallel processing
 *
 * Groups constraints that can be solved in parallel (no shared bodies).
 * Each batch contains constraints with the same color.
 */
struct AvbdColorBatch {
  physx::PxU32 *constraintIndices; //!< Indices into constraint array
  physx::PxU32 numConstraints;     //!< Number of constraints in batch
  physx::PxU32 capacity;           //!< Allocated capacity
  physx::PxU32 colorId;            //!< Color ID for this batch

  inline void reset() { numConstraints = 0; }

  inline void initialize(physx::PxU32 cap, physx::PxU32 color,
                         physx::PxAllocatorCallback &allocator) {
    capacity = cap;
    colorId = color;
    numConstraints = 0;
    if (cap > 0) {
      constraintIndices = static_cast<physx::PxU32 *>(allocator.allocate(
          sizeof(physx::PxU32) * cap, "AvbdColorBatch", __FILE__, __LINE__));
    } else {
      constraintIndices = nullptr;
    }
  }

  inline void release(physx::PxAllocatorCallback &allocator) {
    if (constraintIndices) {
      allocator.deallocate(constraintIndices);
      constraintIndices = nullptr;
    }
    capacity = 0;
    numConstraints = 0;
  }

  inline void addConstraint(physx::PxU32 idx) {
    if (numConstraints < capacity) {
      constraintIndices[numConstraints++] = idx;
    }
  }
};

/**
 * @brief Parallel coloring manager for AVBD solver
 *
 * Implements constraint-based graph coloring where constraints are colored
 * such that no two constraints sharing a body have the same color.
 * This allows constraints with the same color to be solved in parallel.
 */
class AvbdParallelColoring {
public:
  static const physx::PxU32 MAX_COLORS = 32; //!< Maximum color count
  static const physx::PxU32 STATIC_COLOR =
      0; //!< Color for constraints with only statics

  AvbdParallelColoring()
      : mNumColors(0), mNumBatches(0), mAllocator(nullptr),
        mInitialized(false) {
    for (physx::PxU32 i = 0; i < MAX_COLORS; ++i) {
      mConstraintBatches[i].constraintIndices = nullptr;
      mConstraintBatches[i].capacity = 0;
      mConstraintBatches[i].numConstraints = 0;
    }
  }

  ~AvbdParallelColoring() { release(); }

  /**
   * @brief Initialize with expected capacity
   */
  void initialize(physx::PxU32 maxConstraints,
                  physx::PxAllocatorCallback &allocator) {
    mAllocator = &allocator;
    mInitialized = true;

    // Allocate full capacity for each batch to prevent constraint loss
    // when color distribution is uneven. In the worst case, all constraints
    // could be assigned to a single color.
    // TODO(future): Use dynamic resizing or smarter capacity estimation
    for (physx::PxU32 i = 0; i < MAX_COLORS; ++i) {
      mConstraintBatches[i].initialize(maxConstraints, i, allocator);
    }
  }

  /**
   * @brief Release resources
   */
  void release() {
    if (mInitialized && mAllocator) {
      for (physx::PxU32 i = 0; i < MAX_COLORS; ++i) {
        mConstraintBatches[i].release(*mAllocator);
      }
    }
    mNumColors = 0;
    mNumBatches = 0;
    mInitialized = false;
  }

  /**
   * @brief Perform graph coloring on constraints
   *
   * Colors constraints based on body sharing - constraints sharing
   * a dynamic body get different colors.
   *
   * @return Number of colors used
   */
  physx::PxU32 colorConstraints(AvbdContactConstraint *constraints,
                                 physx::PxU32 numConstraints,
                                 AvbdSolverBody *bodies,
                                 physx::PxU32 numBodies) {
    if (!mInitialized || numConstraints == 0 || !mAllocator) {
      return 0;
    }

    // Reset batches
    for (physx::PxU32 i = 0; i < MAX_COLORS; ++i) {
      mConstraintBatches[i].reset();
    }

    // Handle edge case of zero bodies
    if (numBodies == 0) {
      return 0;
    }

    // Track which color each body was last used in
    // Using a simple array for fast lookup
    physx::PxU32 *bodyLastColor = static_cast<physx::PxU32 *>(
        mAllocator->allocate(sizeof(physx::PxU32) * numBodies, "bodyLastColor",
                             __FILE__, __LINE__));

    // Check if allocation failed
    if (!bodyLastColor) {
      // Cannot perform coloring - return 0 colors
      return 0;
    }

    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      bodyLastColor[i] = 0xFFFFFFFF; // No color assigned yet
    }

    mNumColors = 0;

    // Greedy coloring of constraints
    for (physx::PxU32 c = 0; c < numConstraints; ++c) {
      physx::PxU32 bodyA = constraints[c].header.bodyIndexA;
      physx::PxU32 bodyB = constraints[c].header.bodyIndexB;

      // Skip invalid constraints
      if (bodyA >= numBodies && bodyB >= numBodies) {
        continue;
      }

      // Handle static-static case (shouldn't happen but be safe)
      bool isAStatic = (bodyA >= numBodies) || bodies[bodyA].isStatic();
      bool isBStatic = (bodyB >= numBodies) || bodies[bodyB].isStatic();

      if (isAStatic && isBStatic) {
        // Static-static constraint, add to color 0
        mConstraintBatches[STATIC_COLOR].addConstraint(c);
        constraints[c].header.colorGroup = STATIC_COLOR;
        continue;
      }

      // Find colors blocked by involved bodies
      physx::PxU32 blockedColors = 0;

      if (bodyA < numBodies && !bodies[bodyA].isStatic()) {
        if (bodyLastColor[bodyA] < MAX_COLORS) {
          blockedColors |= (1u << bodyLastColor[bodyA]);
        }
      }
      if (bodyB < numBodies && !bodies[bodyB].isStatic()) {
        if (bodyLastColor[bodyB] < MAX_COLORS) {
          blockedColors |= (1u << bodyLastColor[bodyB]);
        }
      }

      // Find first available color (starting from 1, 0 is for statics)
      physx::PxU32 color = 1;
      while ((blockedColors & (1u << color)) != 0 && color < MAX_COLORS) {
        color++;
      }

      // Assign constraint to this color
      if (color < MAX_COLORS) {
        mConstraintBatches[color].addConstraint(c);
        constraints[c].header.colorGroup = static_cast<physx::PxU16>(color);

        // Update body color tracking
        if (bodyA < numBodies && !bodies[bodyA].isStatic()) {
          bodyLastColor[bodyA] = color;
        }
        if (bodyB < numBodies && !bodies[bodyB].isStatic()) {
          bodyLastColor[bodyB] = color;
        }

        if (color + 1 > mNumColors) {
          mNumColors = color + 1;
        }
      }
    }

    mAllocator->deallocate(bodyLastColor);

    // Count active batches
    mNumBatches = 0;
    for (physx::PxU32 i = 0; i < mNumColors; ++i) {
      if (mConstraintBatches[i].numConstraints > 0) {
        mNumBatches++;
      }
    }

    return mNumColors;
  }

  /**
   * @brief Get constraint batch for a color
   */
  const AvbdColorBatch &getBatch(physx::PxU32 colorIndex) const {
    return mConstraintBatches[colorIndex];
  }

  /**
   * @brief Get pointer to all constraint batches (for pre-computed coloring)
   */
  AvbdColorBatch *getBatches() { return mConstraintBatches; }

  /**
   * @brief Get number of colors used
   */
  physx::PxU32 getNumColors() const { return mNumColors; }

  /**
   * @brief Get number of non-empty batches
   */
  physx::PxU32 getNumBatches() const { return mNumBatches; }

  /**
   * @brief Check if coloring is initialized
   */
  bool isInitialized() const { return mInitialized; }

private:
  AvbdColorBatch mConstraintBatches[MAX_COLORS];
  physx::PxU32 mNumColors;
  physx::PxU32 mNumBatches;
  physx::PxAllocatorCallback *mAllocator;
  bool mInitialized;
};

/**
 * @brief Body color batch for parallel processing
 *
 * Groups bodies that can be solved in parallel (no shared constraints).
 * Each batch contains bodies with the same color.
 */
struct AvbdBodyColorBatch {
  physx::PxU32 *bodyIndices; //!< Indices into body array
  physx::PxU32 numBodies;     //!< Number of bodies in batch
  physx::PxU32 capacity;      //!< Allocated capacity
  physx::PxU32 colorId;       //!< Color ID for this batch

  inline void reset() { numBodies = 0; }

  inline void initialize(physx::PxU32 cap, physx::PxU32 color,
                         physx::PxAllocatorCallback &allocator) {
    capacity = cap;
    colorId = color;
    numBodies = 0;
    if (cap > 0) {
      bodyIndices = static_cast<physx::PxU32 *>(allocator.allocate(
          sizeof(physx::PxU32) * cap, "AvbdBodyColorBatch", __FILE__, __LINE__));
    } else {
      bodyIndices = nullptr;
    }
  }

  inline void release(physx::PxAllocatorCallback &allocator) {
    if (bodyIndices) {
      allocator.deallocate(bodyIndices);
      bodyIndices = nullptr;
    }
    capacity = 0;
    numBodies = 0;
  }

  inline void addBody(physx::PxU32 idx) {
    if (numBodies < capacity) {
      bodyIndices[numBodies++] = idx;
    }
  }
};

/**
 * @brief Body-based parallel coloring manager for AVBD solver
 *
 * Implements body-based graph coloring where bodies are colored
 * such that no two bodies sharing a constraint have the same color.
 * This allows bodies with the same color to be solved in parallel,
 * which is more efficient for block coordinate descent.
 */
class AvbdBodyParallelColoring {
public:
  static const physx::PxU32 MAX_COLORS = 32; //!< Maximum color count
  static const physx::PxU32 STATIC_COLOR =
      0; //!< Color for static bodies

  AvbdBodyParallelColoring()
      : mNumColors(0), mNumBatches(0), mAllocator(nullptr),
        mInitialized(false) {
    for (physx::PxU32 i = 0; i < MAX_COLORS; ++i) {
      mBodyBatches[i].bodyIndices = nullptr;
      mBodyBatches[i].capacity = 0;
      mBodyBatches[i].numBodies = 0;
    }
  }

  ~AvbdBodyParallelColoring() { release(); }

  /**
   * @brief Initialize with expected capacity
   */
  void initialize(physx::PxU32 maxBodies,
                  physx::PxAllocatorCallback &allocator) {
    mAllocator = &allocator;
    mInitialized = true;

    // Allocate full capacity for each batch
    for (physx::PxU32 i = 0; i < MAX_COLORS; ++i) {
      mBodyBatches[i].initialize(maxBodies, i, allocator);
    }
  }

  /**
   * @brief Release resources
   */
  void release() {
    if (mInitialized && mAllocator) {
      for (physx::PxU32 i = 0; i < MAX_COLORS; ++i) {
        mBodyBatches[i].release(*mAllocator);
      }
    }
    mNumColors = 0;
    mNumBatches = 0;
    mInitialized = false;
  }

  /**
   * @brief Perform graph coloring on bodies
   *
   * Colors bodies based on constraint sharing - bodies sharing
   * a constraint get different colors.
   *
   * @return Number of colors used
   */
  physx::PxU32 colorBodies(AvbdContactConstraint *constraints,
                            physx::PxU32 numConstraints,
                            AvbdSolverBody *bodies,
                            physx::PxU32 numBodies) {
    if (!mInitialized || numBodies == 0 || !mAllocator) {
      return 0;
    }

    // Reset batches
    for (physx::PxU32 i = 0; i < MAX_COLORS; ++i) {
      mBodyBatches[i].reset();
    }

    // Build adjacency list for bodies
    // Two bodies are adjacent if they share a constraint
    physx::PxU32 *adjacencyMask = static_cast<physx::PxU32 *>(
        mAllocator->allocate(sizeof(physx::PxU32) * numBodies, "adjacencyMask",
                             __FILE__, __LINE__));

    // Check if allocation failed
    if (!adjacencyMask) {
      // Cannot perform coloring - return 0 colors
      return 0;
    }

    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      adjacencyMask[i] = 0;
      bodies[i].colorGroup = 0xFFFFFFFF; // Uncolored
    }

    // Build adjacency from constraints
    for (physx::PxU32 c = 0; c < numConstraints; ++c) {
      physx::PxU32 bodyA = constraints[c].header.bodyIndexA;
      physx::PxU32 bodyB = constraints[c].header.bodyIndexB;

      if (bodyA < numBodies && bodyB < numBodies) {
        adjacencyMask[bodyA] |= (1u << bodyB);
        adjacencyMask[bodyB] |= (1u << bodyA);
      }
    }

    mNumColors = 0;

    // Greedy coloring of bodies
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].isStatic()) {
        bodies[i].colorGroup = STATIC_COLOR;
        mBodyBatches[STATIC_COLOR].addBody(i);
        continue;
      }

      // Find colors used by neighbors
      physx::PxU32 usedColors = 0;
      physx::PxU32 neighborMask = adjacencyMask[i];
      
      // Check each neighbor's color
      for (physx::PxU32 j = 0; j < numBodies && j < 32; ++j) {
        if ((neighborMask & (1u << j)) && bodies[j].colorGroup < MAX_COLORS) {
          usedColors |= (1u << bodies[j].colorGroup);
        }
      }

      // Find first available color (starting from 1, 0 is for statics)
      physx::PxU32 color = 1;
      while ((usedColors & (1u << color)) != 0 && color < MAX_COLORS) {
        color++;
      }

      if (color < MAX_COLORS) {
        bodies[i].colorGroup = color;
        mBodyBatches[color].addBody(i);
        if (color + 1 > mNumColors) {
          mNumColors = color + 1;
        }
      }
    }

    mAllocator->deallocate(adjacencyMask);

    // Count active batches
    mNumBatches = 0;
    for (physx::PxU32 i = 0; i < mNumColors; ++i) {
      if (mBodyBatches[i].numBodies > 0) {
        mNumBatches++;
      }
    }

    return mNumColors;
  }

  /**
   * @brief Get body batch for a color
   */
  const AvbdBodyColorBatch &getBatch(physx::PxU32 colorIndex) const {
    return mBodyBatches[colorIndex];
  }

  /**
   * @brief Get pointer to all body batches (for pre-computed coloring)
   */
  AvbdBodyColorBatch *getBatches() { return mBodyBatches; }

  /**
   * @brief Get number of colors used
   */
  physx::PxU32 getNumColors() const { return mNumColors; }

  /**
   * @brief Get number of non-empty batches
   */
  physx::PxU32 getNumBatches() const { return mNumBatches; }

  /**
   * @brief Check if coloring is initialized
   */
  bool isInitialized() const { return mInitialized; }

private:
  AvbdBodyColorBatch mBodyBatches[MAX_COLORS];
  physx::PxU32 mNumColors;
  physx::PxU32 mNumBatches;
  physx::PxAllocatorCallback *mAllocator;
  bool mInitialized;
};

/**
 * @brief Parallel iteration helper
 *
 * Provides methods for parallel processing of constraint batches.
 */
class AvbdParallelIterator {
public:
  /**
   * @brief Process all constraints in a color batch (sequential)
   */
  template <typename ConstraintSolver>
  static void processColorBatchSequential(const AvbdColorBatch &batch,
                                          AvbdContactConstraint *constraints,
                                          AvbdSolverBody *bodies,
                                          physx::PxU32 numBodies,
                                          ConstraintSolver &solver) {
    for (physx::PxU32 i = 0; i < batch.numConstraints; ++i) {
      physx::PxU32 constraintIdx = batch.constraintIndices[i];
      solver.solveConstraint(constraints[constraintIdx], bodies, numBodies);
    }
  }

  /**
   * @brief Process constraints by color in order
   *
   * This provides data-race-free iteration where constraints
   * within the same color can be processed in parallel in the future.
   */
  template <typename ConstraintSolver>
  static void processAllColorsSequential(const AvbdParallelColoring &coloring,
                                         AvbdContactConstraint *constraints,
                                         AvbdSolverBody *bodies,
                                         physx::PxU32 numBodies,
                                         ConstraintSolver &solver) {
    for (physx::PxU32 color = 0; color < coloring.getNumColors(); ++color) {
      const AvbdColorBatch &batch = coloring.getBatch(color);
      processColorBatchSequential(batch, constraints, bodies, numBodies,
                                  solver);
    }
  }

  /**
   * @brief Process all bodies in a color batch (sequential)
   */
  template <typename BodySolver>
  static void processBodyColorBatchSequential(const AvbdBodyColorBatch &batch,
                                             AvbdSolverBody *bodies,
                                             physx::PxU32 numBodies,
                                             BodySolver &solver) {
    for (physx::PxU32 i = 0; i < batch.numBodies; ++i) {
      physx::PxU32 bodyIdx = batch.bodyIndices[i];
      solver.solveBody(bodies[bodyIdx]);
    }
  }

  /**
   * @brief Process bodies by color in order
   *
   * This provides data-race-free iteration where bodies
   * within the same color can be processed in parallel in the future.
   * This is the true block coordinate descent approach.
   */
  template <typename BodySolver>
  static void processAllBodyColorsSequential(const AvbdBodyParallelColoring &coloring,
                                             AvbdSolverBody *bodies,
                                             physx::PxU32 numBodies,
                                             BodySolver &solver) {
    for (physx::PxU32 color = 0; color < coloring.getNumColors(); ++color) {
      const AvbdBodyColorBatch &batch = coloring.getBatch(color);
      processBodyColorBatchSequential(batch, bodies, numBodies, solver);
    }
  }
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_PARALLEL_H
