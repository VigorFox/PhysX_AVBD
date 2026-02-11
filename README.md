# NVIDIA PhysX + AVBD Solver

> 🔬 **Research Fork**: Experimental AVBD (Augmented Variable Block Descent) constraint solver integrated into NVIDIA PhysX SDK.

Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved. BSD-3-Clause License.

## Project Status

| Feature | Status | Notes |
|---------|--------|-------|
| Contact Solver | ✅ Working | AL outer loop + 6x6 Hessian local solve |
| Spherical Joint | ✅ AVBD Hessian | Pure penalty, accumulates into body 6x6 system |
| Fixed Joint | ✅ AVBD Hessian | Position + rotation (world-frame error) |
| D6 Joint | ✅ AVBD Hessian | Locked linear DOFs + angular velocity damping (SLERP) |
| Revolute Joint | ⚠️ GS Fallback | Gauss-Seidel correction, not yet in Hessian |
| Prismatic Joint | ⚠️ GS Fallback | Gauss-Seidel correction, not yet in Hessian |
| Gear Joint | ⚠️ GS Fallback | Gauss-Seidel correction, not yet in Hessian |
| Motor Drive | ✅ Working | Torque-based RevoluteJoint motor |
| Joint Limits | ✅ Working | Revolute, Prismatic, Spherical cone, D6 |
| O(K) Constraint Lookup | ✅ Optimized | Per-body constraint map eliminates O(N) scan |
| Multi-threaded Islands | ✅ Thread-safe | Per-island constraint mappings |
| Custom Joint | ❌ Not Available | Custom constraint callbacks unsupported |
| Rack & Pinion | ❌ Not Available | RackAndPinionJoint unsupported |
| Mimic Joint | ❌ Not Available | MimicJoint unsupported |
| Articulation | ❌ Not Available | Currently unsupported |
| Sleep / Wake | ❌ Not Available | Not implemented |

**For research and evaluation only. Not production-ready.**

## Why AVBD?

PhysX's built-in TGS/PGS are **velocity-level** iterative solvers that hit fundamental limits in several scenarios:

| Problem | TGS/PGS Limitation | AVBD Solution |
|---------|---------------------|---------------|
| **High mass-ratio joints** | Condition number explosion, rubber-banding | Augmented Lagrangian, mass-ratio-insensitive convergence |
| **Multiplayer sync** | Velocity integration accumulates FP error | Position-level solve, better cross-platform consistency |
| **Cloth & soft body** | Requires separate solver | Position-level block descent maps to CPU SIMD / GPU compute |

AVBD introduces a **unified position-level constraint solving framework** to support:

1. **"Ultimate Hand" gameplay** -- A kinematic hand (infinite mass) grips light objects via joints. Requires unconditionally stable high mass-ratio joint solving.
2. **Whole-scene solver** -- AVBD as the sole solver for the entire scene. All rigid bodies must solve stably.
3. **Multiplayer determinism** -- Position-level solving avoids velocity integration error accumulation.
4. **Cloth & soft body unification** -- Per-body Jacobi structure extends to deformable bodies.

### Roadmap

```
Contact AL stability (DONE)         Joint Hessian integration (IN PROGRESS)
  Rigid body contacts stable      ->    Spherical, Fixed, D6: DONE
  AVBD usable as whole-scene solver     Revolute, Prismatic, Gear: TODO
            |                                    |
  Lambda warm-starting                 Cloth / soft body / GPU
  Reduce iterations -> performance     Unified solver architecture
            |                                    |
              Multiplayer determinism across all the above
```

> See [docs/AVBD_SOLVER_README.md](docs/AVBD_SOLVER_README.md) and [docs/SOLVER_ALGORITHM_ANALYSIS.md](docs/SOLVER_ALGORITHM_ANALYSIS.md) for details.

## Solver Architecture

### Unified AVBD Hessian Approach

The solver accumulates **contacts and joints** into a single per-body 6x6 Hessian system, then solves via LDLT:

```
For each body i:
  H = M/h^2 * I_6x6                       (inertia)
  g = M/h^2 * (x_i - x_tilde)             (inertial RHS)

  For each contact on body i:
    H += rho * J^T J                       (contact penalty)
    g += J * (rho * C + lambda)            (contact gradient)

  For each spherical/fixed/D6 joint on body i:
    H += rho * J^T J                       (joint penalty, locked DOFs)
    g += J * (rho * C)                     (pure penalty, lambda = 0)
    H_ang += (damping/h^2) * I_3x3        (angular velocity damping)
    g_ang += (damping/h^2) * deltaW_init   (damping gradient)

  delta = LDLT_solve(H, g)
  x_i -= delta
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pure penalty for joints** (no lambda dual update) | Body-centric BCD cannot propagate constraint satisfaction along chains fast enough for AL dual to converge. Lambda grows as N * rho * C per frame, causing explosion. Pure penalty with rho >> M/h^2 is sufficient. |
| **World-frame rotation error** | Fixed joint `computeRotationViolation` computes error as `(rotA * relRot) * rotB^-1` in world frame, matching the world-frame angular Jacobian `J = [0, e_k]`. |
| **No joint sign on angular damping** | Damping is a LOCAL property of each body. Using the joint sign (bodyA=+1, bodyB=-1) would flip the gradient for bodyB, injecting energy instead of dissipating it. |
| **GS immediate-apply for remaining joints** | Revolute, prismatic, gear joints use GS with immediate position update (not Jacobi averaging), improving chain convergence. |

### Comparison with TGS/PGS

| Property | PGS | TGS | AVBD |
|----------|-----|-----|------|
| Solve Level | Velocity | Velocity | **Position** |
| Convergence | Linear | Sublinear | AL-augmented |
| Stack Stability | Fair | Good | **Good** |
| Mass-ratio Robustness | Poor | Fair | **Good** |
| Joint Chain Support | N/A | Implicit | **Hessian (Sph/Fix/D6) + GS (Rev/Pris/Gear)** |

## Quick Start

### Build

```bash
cd physx
./generate_projects.bat  # Windows
./generate_projects.sh   # Linux
```

### Enable AVBD

```cpp
PxSceneDesc sceneDesc(physics->getTolerancesScale());
sceneDesc.solverType = PxSolverType::eAVBD;
```

## Source Structure

```
physx/source/lowleveldynamics/src/
  DyAvbdSolver.h/cpp       # Core solver (solveLocalSystemWithJoints, solveWithJoints)
  DyAvbdDynamics.h/cpp      # PhysX integration
  DyAvbdTasks.h/cpp          # Multi-threading
  DyAvbdTypes.h              # Config & data structures (AvbdSolverConfig)
  DyAvbdConstraint.h         # Constraint definitions (all joint types)
  DyAvbdJointSolver.h/cpp    # Joint multiplier updates, GS corrections
  DyAvbdSolverBody.h         # Body state
```

## Profiling

PVD Profile Zones:
- `AVBD.update` -- Total update time
- `AVBD.solveWithJoints` -- Main solver loop
- `AVBD.blockDescentWithJoints` -- Constraint iterations
- `AVBD.updateLambda` -- Multiplier updates

## Known Limitations

1. **No Articulation support** -- Articulated bodies not implemented
2. **No Sleep/Wake** -- Bodies remain active
3. **CPU only** -- No GPU acceleration
4. **D6 angular damping tuning** -- SLERP damping with `isAcceleration=true` and high coefficients may over-damp; parameter tuning needed
5. **Revolute/Prismatic/Gear not in Hessian** -- Still use GS fallback, planned for future migration
6. **Joint solver deviates from AVBD AL -- high mass-ratio instability expected**
   Joint constraints (Spherical, Fixed, D6) use the AVBD BCD Hessian structure
   but with **pure penalty** (lambda = 0, no AL dual update). This deviates from
   the paper's Augmented Lagrangian formulation. The reason: BCD propagates
   constraint satisfaction only 1 body per iteration; for a chain of N bodies,
   the violation C is still large after inner iterations, and the dual update
   lambda += rho * C causes lambda to explode.

   Pure penalty requires rho >> M/h^2 to enforce constraints. For heavy bodies
   (e.g. 1000 kg, M/h^2 = 3,600,000), the current rho = 1e6 is insufficient
   (ratio = 0.28x). This means **high mass-ratio joint scenarios (the primary
   motivation for AVBD) will not work correctly** until full AL is restored.

   Potential fix: **chain-ordered Gauss-Seidel** -- solve bodies along joint
   chain topology (anchor to leaf) so constraint satisfaction propagates fully
   before AL dual update. This makes lambda see converged C values, preventing
   explosion while retaining mass-ratio-insensitive AL convergence.

## Original PhysX Documentation

- [PhysX User Guide](https://nvidia-omniverse.github.io/PhysX/physx/index.html)
- [API Documentation](https://nvidia-omniverse.github.io/PhysX)

## License

NVIDIA PhysX BSD-3-Clause. See [LICENSE.md](LICENSE.md).
