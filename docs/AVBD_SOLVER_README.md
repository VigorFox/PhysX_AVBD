# AVBD Solver - Development Notes

## Current Configuration

| Parameter            | Value   | Notes                          |
|----------------------|---------|--------------------------------|
| outerIterations      | 4       | AL multiplier updates          |
| innerIterations      | 8       | Block descent iterations per AL step |
| baumgarte            | 0.3     | Position correction factor     |
| angularContactScale  | 0.2     | Angular correction scale from contact normals |
| velocityDamping      | 0.99    | Linear velocity damping        |
| angularDamping       | 0.95    | Angular velocity damping       |
| initialRho           | 1e6     | Initial AL penalty parameter   |
| maxRho               | 1e8     | Maximum AL penalty parameter   |

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
