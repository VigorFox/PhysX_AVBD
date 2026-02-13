# NVIDIA PhysX + AVBD Solver

> 🔬 **Research Fork**: Experimental AVBD (Augmented Variable Block Descent) constraint solver integrated into NVIDIA PhysX SDK.

Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved. BSD-3-Clause License.

## Project Status

| Feature | Status | Notes |
|---------|--------|-------|
| Contact Solver | ✅ Working | AL outer loop + 6x6 Hessian local solve |
| Spherical Joint | ✅ AVBD Hessian | In local system (6x6 + 3x3), with stabilized dual update |
| Fixed Joint | ✅ AVBD Hessian | Position + rotation rows, with stabilized dual update |
| D6 Joint | ✅ AVBD Hessian | Locked linear DOFs + angular damping in local system |
| Revolute Joint | ⚠️ GS Fallback | Gauss-Seidel correction, not yet in Hessian |
| Prismatic Joint | ⚠️ GS Fallback | Gauss-Seidel correction, not yet in Hessian |
| Gear Joint | ⚠️ GS Fallback | Gauss-Seidel correction, not yet in Hessian |
| Motor Drive | ❌ Regression | `SnippetJointDrive` currently failing after joint algorithm update |
| Joint Limits | ⚠️ Partial | Limits exist, but drive/limit stability still being retuned |
| SnippetChainmail | ✅ Added | Extreme stress scene for dense joint mesh + high-speed impact |
| 3x3 Decoupled Path | ⚠️ Functional with limits | Joints included in accumulation, but dense mesh impact is weaker than 6x6 |
| O(K) Constraint Lookup | ✅ Optimized | Per-body constraint map eliminates O(N) scan |
| Multi-threaded Islands | ✅ Thread-safe | Per-island constraint mappings |
| Custom Joint | ❌ Not Available | Custom constraint callbacks unsupported |
| Rack & Pinion | ❌ Not Available | RackAndPinionJoint unsupported |
| Mimic Joint | ❌ Not Available | MimicJoint unsupported |
| Articulation | ❌ Not Available | Currently unsupported |
| Sleep / Wake | ❌ Not Available | Not implemented |

**For research and evaluation only. Not production-ready.**

### Current Validation Snapshot

- ✅ Joint chain stability under 3x3 and 6x6 local solves (with spherical/fixed/D6 accumulation)
- ✅ `SnippetChainmail` integrated for extreme impact regression testing
- ⚠️ `SnippetJointDrive` is currently a known failing scenario after recent AVBD joint algorithm changes
- ⚠️ 3x3 path is kept for cost/perf fallback, but dense joint-mesh impact scenes should prefer 6x6

## SnippetChainmail Demo



https://github.com/user-attachments/assets/2ab299c7-8f7f-4bf2-b8b5-7de8033b17f8



## Why AVBD?

PhysX's built-in TGS/PGS are **velocity-level** iterative solvers that hit fundamental limits in several scenarios:

| Problem | TGS/PGS Limitation | AVBD Solution |
|---------|---------------------|---------------|
| **High mass-ratio joints** | Condition number explosion, rubber-banding | Augmented Lagrangian, mass-ratio-insensitive convergence |
| **Multiplayer sync** | Velocity integration accumulates FP error | Position-level solve, better cross-platform consistency |
| **Cloth & soft body** | Requires separate solver | Position-level block descent maps to CPU SIMD / GPU compute |

AVBD introduces a **unified position-level constraint solving framework** to support:

1. **"Ultra Hand" gameplay** -- A kinematic hand (infinite mass) grips light objects via joints. Requires unconditionally stable high mass-ratio joint solving.
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
    H += rho_eff * J^T J                   (rho_eff = max(rho, M/h^2))
    g += J * (rho_eff * C + lambda)        (primal term)
    H_ang += (damping/h^2) * I_3x3         (angular damping)
    g_ang += (damping/h^2) * deltaW_init   (damping gradient)

  Joint dual update (stabilized):
    rhoDual = min(Mh^2, rho^2/(rho + Mh^2))
    lambda  = decay * lambda + rhoDual * C

  delta = LDLT_solve(H, g)
  x_i -= delta
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Stabilized AL dual for joints** | Uses bounded dual step and decay (`rhoDual`, `lambdaDecay`) to reduce overshoot while keeping AL memory in chain/impact scenarios. |
| **World-frame rotation error** | Fixed joint `computeRotationViolation` computes error as `(rotA * relRot) * rotB^-1` in world frame, matching the world-frame angular Jacobian `J = [0, e_k]`. |
| **No joint sign on angular damping** | Damping is a LOCAL property of each body. Using the joint sign (bodyA=+1, bodyB=-1) would flip the gradient for bodyB, injecting energy instead of dissipating it. |
| **3x3/6x6 accumulation parity** | Both paths now accumulate contacts + spherical/fixed/D6 identically; only local solve differs (6x6 LDLT vs decoupled 3x3 blocks). |
| **GS immediate-apply for remaining joints** | Revolute, prismatic, gear joints still use GS immediate update while waiting for full Hessian migration. |

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
4. **3x3 dense-mesh coupling loss** -- 3x3 decoupled local solve drops linear-angular off-diagonal coupling, so dense joint meshes under impact (e.g. small fast ball on chainmail) are significantly weaker than 6x6
5. **`SnippetJointDrive` regression** -- currently failing after recent AVBD joint algorithm update; drive/limit path is under retuning
6. **Revolute/Prismatic/Gear not in Hessian** -- still GS fallback, planned for migration into unified local system
7. **Extreme-scene tuning still in progress** -- chainmail-style stress scenarios required algorithm updates and remain the primary tuning target

## Original PhysX Documentation

- [PhysX User Guide](https://nvidia-omniverse.github.io/PhysX/physx/index.html)
- [API Documentation](https://nvidia-omniverse.github.io/PhysX)

## License

NVIDIA PhysX BSD-3-Clause. See [LICENSE.md](LICENSE.md).

