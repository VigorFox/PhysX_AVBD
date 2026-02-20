# AVBD Solver - Development Notes

> **Update (2026-02-14)**: Lambda warm-starting has been implemented. See §1.6 for implementation details.

## Why AVBD

AVBD (Augmented Variable Block Descent) was introduced into PhysX for the following strategic goals:

1. **High mass-ratio joint stability**: Enable "ultimate hand" gameplay where a kinematic hand (effectively infinite mass) manipulates light objects via joints. TGS/PGS velocity-level solvers suffer from condition number explosion at extreme mass ratios. AVBD's Augmented Lagrangian framework provides mass-ratio-insensitive convergence.

2. **Whole-scene solver**: When AVBD is enabled for ultimate hand joints, it serves as the **sole solver for the entire scene**. All other objects (props, environment, stacked objects) must also solve stably and efficiently under AVBD. Contact stability is therefore a baseline requirement, not a secondary concern.

3. **Multiplayer determinism**: Position-level solvers directly manipulate positions, avoiding the error-accumulation problem of velocity-level integration. This provides better cross-platform floating-point consistency, which is critical for lockstep or state-sync multiplayer architectures.

4. **Cloth & soft body unification**: AVBD's position-level block descent framework naturally extends to deformable bodies. The per-body/per-vertex Jacobi structure maps efficiently to CPU SIMD and GPU compute shaders, enabling a single solver architecture for rigid bodies, cloth, and soft bodies with unified contact handling.

### Roadmap

```
Contact AL stability (DONE)         Joint AL fix (NEXT)
  All non-joint objects stable    →    Ultimate hand works
  AVBD usable as sole solver           High mass-ratio joints stable
            ↓                                    ↓
  Lambda warm-starting (DONE)          Cloth / soft body / GPU
  Reduce iterations → perf             Unified solver architecture
            ↓                                    ↓
              Multiplayer determinism across all the above
```

## Current Configuration (Defaults)

| Parameter            | Value   | Notes                          |
|----------------------|---------|--------------------------------|
| outerIterations      | 1       | AL multiplier updates          |
| innerIterations      | 4       | Block descent iterations per AL step |
| baumgarte            | 0.3     | Position correction factor     |
| angularContactScale  | 0.2     | Angular correction scale from contact normals |
| velocityDamping      | 0.99    | Linear velocity damping        |
| angularDamping       | 0.95    | Angular velocity damping       |
| initialRho           | 1e4     | Initial AL penalty parameter   |
| maxRho               | 1e8     | Maximum AL penalty parameter   |

**Default path**: 3x3 decoupled local solve (`enableLocal6x6Solve = false`).

**Recommended for stability** (stacking/joints): `outerIterations=4`, `innerIterations=8`.

## Known Issues

### 1. Low iteration count (1x4) does not achieve stable stacking

**Status**: Open

**Description**: The AVBD paper recommends outerIterations=1, innerIterations=4 as sufficient for stable simulation. However, the current implementation requires outerIterations=4, innerIterations=8 (32 total inner iterations) to achieve stable box stacking. With 1x4 (4 total), stacked boxes slowly sink or collapse.

**Root cause analysis**: The Augmented Lagrangian mechanism is partially functional but not yet converging fast enough with low iterations. Contributing factors:

- **angularContactScale = 0.2**: Angular correction from contacts is intentionally scaled down to 20% to prevent lateral drift from asymmetric contact patches. This may slow convergence compared to full angular correction.

- **No compliance in effective mass**: The effective mass `w` does not include a compliance term `alpha/h^2`, which the AVBD/XPBD literature uses to regularize the system and improve convergence characteristics.

- **Lambda warm-starting is implemented but may need tuning**: The implementation uses decay factors (`wsAlpha=0.95`, `wsGamma=0.99`) that may need adjustment for optimal convergence.

**Workaround**: Use outerIterations=4, innerIterations=8.

### 2. Lambda warm-starting implementation details ✅

**Status**: Implemented

**Description**: Lambda warm-starting has been fully implemented in [`DyAvbdDynamics.cpp`](physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp:338-406).

**Implementation details**:

1. **Cache structure** ([`DyAvbdDynamics.h:116-125`](physx/source/lowleveldynamics/src/DyAvbdDynamics.h:116-125)):
   ```cpp
   struct CachedLambda {
     PxReal lambda;          // Normal constraint lambda
     PxReal tangentLambda0;  // Friction lambda 1
     PxReal tangentLambda1;  // Friction lambda 2
    PxReal penalty;         // Adaptive penalty for normal (persists across frames)
     PxReal tangentPenalty0;  // Adaptive penalty for tangent 0
     PxReal tangentPenalty1;  // Adaptive penalty for tangent 1
    PxU8 frameAge;          // Frames since last update (0 = current frame)
     PxU8 padding[3];
   };
   ```

2. **Initialization** ([`DyAvbdDynamics.cpp:338-342`](physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp:338-342)):
   ```cpp
   mEnableLambdaWarmStart = true;  // Default enabled
   mLambdaCache.resize(4096);      // Pre-allocate for ~1000 contact managers x 4 contacts
   ```

3. **Frame-start cache aging** ([`DyAvbdDynamics.cpp:422-430`](physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp:422-430)):
   ```cpp
   if (mEnableLambdaWarmStart) {
     for (PxU32 i = 0; i < mLambdaCache.size(); ++i) {
       if (mLambdaCache[i].frameAge < 255) {
         mLambdaCache[i].frameAge++;
       }
     }
   }
   ```

4. **Warm-start reading** ([`DyAvbdDynamics.cpp:1285-1316`](physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp:1285-1316)):
   ```cpp
   if (mEnableLambdaWarmStart && cacheIdx < mLambdaCache.size()) {
     CachedLambda &cached = mLambdaCache[cacheIdx];
     if (cached.frameAge <= LAMBDA_MAX_AGE) {  // LAMBDA_MAX_AGE = 3
       constraint.header.lambda = cached.lambda * wsAlpha * wsGamma;
       constraint.tangentLambda0 = cached.tangentLambda0 * wsAlpha * wsGamma;
       constraint.tangentLambda1 = cached.tangentLambda1 * wsAlpha * wsGamma;
       constraint.header.penalty = PxClamp(cached.penalty * wsGamma, wsPenaltyMin, wsPenaltyMax);
       // ... (tangent penalties similarly decayed)
     } else {
       // Cache expired, reset to zero
       constraint.header.lambda = 0.0f;
     }
   }
   ```

5. **Write-back after solve** ([`DyAvbdDynamics.cpp:377-406`](physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp:377-406)):
   ```cpp
   void writeLambdaToCache(AvbdDynamicsContext &ctx, AvbdContactConstraint *constraints, PxU32 numConstraints) {
     for (PxU32 i = 0; i < numConstraints; ++i) {
       const AvbdContactConstraint &constraint = constraints[i];
       const PxU32 cacheIdx = constraint.cacheIndex;
       AvbdDynamicsContext::CachedLambda &cached = cache[cacheIdx];
       cached.lambda = constraint.header.lambda;
       cached.tangentLambda0 = constraint.tangentLambda0;
       cached.tangentLambda1 = constraint.tangentLambda1;
       cached.penalty = constraint.header.penalty;
       cached.tangentPenalty0 = constraint.tangentPenalty0;
       cached.tangentPenalty1 = constraint.tangentPenalty1;
       cached.frameAge = 0;  // Reset age on update
     }
   }
   ```

**Parameters**:
- `LAMBDA_MAX_AGE = 3` - Maximum frames to keep cached lambda
- `wsAlpha = 0.95` - Lambda decay factor
- `wsGamma = 0.99` - Penalty decay factor
- `wsPenaltyMin = 1000.0f` - Minimum penalty value
- `wsPenaltyMax = 1e9f` - Maximum penalty value

**Benefits**:
- ✅ Lambda values persist across frames for persistent contacts
- ✅ Adaptive penalty parameters are preserved
- ✅ Decaying warm-start avoids numerical instability
- ✅ Frame-age mechanism automatically cleans expired cache entries
- ✅ Supports both normal and tangential constraints

**Impact**: Significantly improves convergence speed, especially for persistent contact scenarios.

**Known limitation**: The decay factors (`wsAlpha=0.95`, `wsGamma=0.99`) may need tuning to achieve paper-recommended 1×4 iteration stability.
