# AVBD Solver - Development Notes

> **Update (2026-04-02)**: Articulation iteration-efficiency pass completed. Full `SnippetAvbdArticulation` regression now passes at 10 solver iterations; 8 still fails in the loaded Scissor Lift case.
>
> **Update (2026-03-07)**: D6 Unification complete ("万物皆D6"). All joint types (Spherical, Fixed, Revolute, Prismatic) unified into single D6 constraint path. Standalone algorithm fully synced with PhysX. 53/53 tests pass.

Status Legend: `Integrated` = merged into main code path; `Accepted` = integrated and validated by acceptance checks; `Pending` = not complete or acceptance gate not closed.

## Why AVBD

AVBD (Augmented Variable Block Descent) was introduced into PhysX for the following strategic goals:

1. **High mass-ratio joint stability**: Enable "ultimate hand" gameplay where a kinematic hand (effectively infinite mass) manipulates light objects via joints. TGS/PGS velocity-level solvers suffer from condition number explosion at extreme mass ratios. AVBD's Augmented Lagrangian framework provides mass-ratio-insensitive convergence.

2. **Whole-scene solver**: When AVBD is enabled for ultimate hand joints, it serves as the **sole solver for the entire scene**. All other objects (props, environment, stacked objects) must also solve stably and efficiently under AVBD. Contact stability is therefore a baseline requirement, not a secondary concern.

3. **Multiplayer determinism**: Position-level solvers directly manipulate positions, avoiding the error-accumulation problem of velocity-level integration. This provides better cross-platform floating-point consistency, which is critical for lockstep or state-sync multiplayer architectures.

4. **Cloth & soft body unification**: AVBD's position-level block descent framework naturally extends to deformable bodies. The per-body/per-vertex Jacobi structure maps efficiently to CPU SIMD and GPU compute shaders, enabling a single solver architecture for rigid bodies, cloth, and soft bodies with unified contact handling.

### Roadmap

```
Contact AL stability (DONE)           D6 Unified Joint System (DONE)
  All non-joint objects stable      →   All joints unified into D6 path
  AVBD usable as sole solver             Spherical/Fixed/Revolute/Prismatic/D6/Gear: accepted
            ↓                                      ↓
  Lambda warm-starting (DONE)            Cloth / soft body / articulation
  Reduce iterations → perf               SOA refactoring, GPU path
            ↓                                      ↓
                Multiplayer determinism across all the above
```

      ## Articulation Iteration-Efficiency Update (2026-04-02)

      ### Goal

      Lower the articulation iteration budget globally, not just for a single scene, while preserving the accepted 29/29 PhysX articulation baseline.

      ### What changed

      1. **D6/articulation warm-starting**
        - Added cached AL multiplier reuse for D6 joints and articulation-internal joints, mirroring the existing contact warm-start path.
        - Cache entries now preserve linear, angular, drive, and cone lambdas across frames.

      2. **Iteration diagnostics**
        - Added env-driven diagnostics in the AVBD dynamics path:
          - `PHYSX_AVBD_ITER_DIAG`
          - `PHYSX_AVBD_ITER_DIAG_EVERY`
          - `PHYSX_AVBD_ITER_DIAG_SEQUENTIAL`
        - These report requested vs executed iterations, joint-island counts, row composition, lambda peaks, and dominant joint sources.
        - Sequential mode is the trustworthy mode for per-island articulation analysis.

      3. **Conservative early-stop**
        - Added pose-delta-based early-stop in both the contact path and the joint path.
        - The maximum requested budget is still preserved, but already-stable islands can stop early instead of always burning the full iteration count.

      4. **Acceleration-drive semantics fixed in the solver**
        - The global lowering blocker was the articulation `eACCELERATION` drive case rather than only the Scissor Lift scene.
        - Internal articulation D6 translation now preserves acceleration-drive flags and per-axis stiffness data.
        - The solver applies response-scaled implicit coefficients in both primal and dual drive updates for acceleration drives.
        - A prep-only approximation was tested first and rejected; the final fix had to live in the solver path.

      ### Validation result

      - Full `SnippetAvbdArticulation` regression passes at **12** iterations.
      - Full `SnippetAvbdArticulation` regression passes at **10** iterations.
      - Full `SnippetAvbdArticulation` regression fails at **8** iterations.
      - Remaining 8-iteration failure is the **loaded Scissor Lift** stability path, not the earlier acceleration-drive test.

      ### Current conclusion

      This optimization is a successful stage result, not the final endpoint:

      - **Repository-wide verified articulation floor**: 10 iterations
      - **Still pending**: lowering the loaded Scissor Lift path from 10 to 8 without regressing the accepted baseline

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

**Important runtime policy**: bodies touching **Prismatic** are forced to local 6x6 solve, even when global 3x3 mode is enabled.

**Recommended for stability** (stacking/joints): `outerIterations=4`, `innerIterations=8`.

## Known Issues

### 0. Low iteration count (1x4) does not achieve stable stacking

**Status**: Pending

**Description**: The AVBD paper recommends outerIterations=1, innerIterations=4 as sufficient for stable simulation. However, the current implementation requires outerIterations=4, innerIterations=8 (32 total inner iterations) to achieve stable box stacking. With 1x4 (4 total), stacked boxes slowly sink or collapse.

**Root cause analysis**: The Augmented Lagrangian mechanism is partially functional but not yet converging fast enough with low iterations. Contributing factors:

- **angularContactScale = 0.2**: Angular correction from contacts is intentionally scaled down to 20% to prevent lateral drift from asymmetric contact patches. This may slow convergence compared to full angular correction.

- **No compliance in effective mass**: The effective mass `w` does not include a compliance term `alpha/h^2`, which the AVBD/XPBD literature uses to regularize the system and improve convergence characteristics.

- **Lambda warm-starting is implemented but may need tuning**: The implementation uses decay factors (`wsAlpha=0.95`, `wsGamma=0.99`) that may need adjustment for optimal convergence.

**Workaround**: Use outerIterations=4, innerIterations=8.

### 1. Lambda warm-starting implementation details ✅

**Status**: Accepted

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

### 2. Documentation and implementation can drift without parity gates

**Status**: Pending

**Description**: The project now contains both integrated and acceptance-pending items. Without explicit parity/acceptance gates, documentation can overstate completion.

**Mitigation**:
- Keep three explicit statuses in docs: `integrated`, `accepted`, `pending`.
- Require PhysX↔standalone parity checks for all newly integrated joint semantics.
