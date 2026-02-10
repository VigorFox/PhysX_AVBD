# AVBD Solver - Development Notes

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
  Lambda warm-starting                 Cloth / soft body / GPU
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

- **Lambda not warm-started across frames**: `prepareAvbdContacts()` in DyAvbdDynamics.cpp resets `constraint.header.lambda = 0.0f` every frame. This forces the AL multiplier to reconverge from zero each frame, discarding the "contact force memory" that should carry over for persistent contacts. Fixing this requires contact persistence tracking (matching contacts across frames).

- **angularContactScale = 0.2**: Angular correction from contacts is intentionally scaled down to 20% to prevent lateral drift from asymmetric contact patches. This may slow convergence compared to full angular correction.

- **No compliance in effective mass**: The effective mass `w` does not include a compliance term `alpha/h^2`, which the AVBD/XPBD literature uses to regularize the system and improve convergence characteristics.

**Workaround**: Use outerIterations=4, innerIterations=8.

### 2. Lambda warm-starting not implemented

**Status**: Open

**Description**: `constraint.header.lambda = 0.0f` is set unconditionally in `prepareAvbdContacts()` (DyAvbdDynamics.cpp ~L1093). The AL multiplier never carries over between simulation frames, even for persistent contacts.

**Impact**: Slower convergence - the solver must rebuild contact force estimates from scratch every frame instead of starting from a good initial guess. This is the primary reason high iteration counts are needed.

**Fix approach**: Implement contact persistence tracking to identify the same contact across frames, then warm-start lambda from the previous frame's converged value.
