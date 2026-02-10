# NVIDIA PhysX + AVBD Solver

> 🔬 **Research Fork**: Experimental AVBD (Augmented Variable Block Descent) constraint solver integrated into NVIDIA PhysX SDK.

Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved. BSD-3-Clause License.

## ⚠️ Project Status

| Feature | Status | Notes |
|---------|--------|-------|
| Rigid Body Solver | ✅ Working | Contacts + 6 joint types |
| Joint System | ⚠️ In Progress | Revolute, Prismatic, Spherical, Fixed, D6, Gear |
| Motor Drive | ✅ Working | Torque-based RevoluteJoint motor |
| Joint Limits | ✅ Working | Revolute, Prismatic, Spherical cone, D6 |
| Custom Joint | ❌ Not Available | Custom constraint callbacks unsupported |
| Rack & Pinion | ❌ Not Available | RackAndPinionJoint unsupported |
| Mimic Joint | ❌ Not Available | MimicJoint unsupported |
| O(M) Constraint Lookup | ✅ Optimized | Eliminates O(N²) complexity |
| Multi-threaded Islands | ✅ Thread-safe | Per-island constraint mappings |
| Articulation | ❌ Not Available | Currently unsupported |
| Sleep / Wake | ❌ Not Available | Not implemented |
| Friction Model | ⚠️ Basic | Coulomb approximation |

**For research and evaluation only. Not production-ready.**

## Why AVBD?

PhysX's built-in TGS/PGS are **velocity-level** iterative solvers that hit fundamental limits in several scenarios:

| Problem | TGS/PGS Limitation | AVBD Solution |
|---------|---------------------|---------------|
| **High mass-ratio joints** | Condition number explosion → rubber-banding / explosion | Augmented Lagrangian provides mass-ratio-insensitive convergence |
| **Multiplayer sync** | Velocity integration accumulates floating-point error → state drift | Position-level solve manipulates positions directly, better cross-platform consistency |
| **Cloth & soft body** | Requires separate solver, cannot unify with rigid bodies | Position-level block descent + Jacobi maps naturally to CPU SIMD / GPU compute |

AVBD introduces a **unified position-level constraint solving framework** to support the following goals:

1. 🎮 **"Ultimate Hand" gameplay** — A kinematic hand (effectively infinite mass) grips light objects via joints. Requires unconditionally stable high mass-ratio joint solving.
2. 🌍 **Whole-scene solver** — AVBD serves as the sole solver for the entire scene. All rigid bodies (props, environment, stacked objects) must solve stably and efficiently.
3. 🔄 **Multiplayer determinism** — Position-level solving avoids velocity integration error accumulation, suitable for lockstep or state-sync multiplayer architectures.
4. 🧶 **Cloth & soft body unification** — The per-body/per-vertex Jacobi structure of position-level block descent extends directly to deformable bodies, enabling a single solver for rigid bodies, cloth, and soft bodies with unified contact handling.

### Roadmap

```
Contact AL stability (DONE)         Joint AL fix (NEXT)
  Rigid body contacts stable      →    High mass-ratio joints stable
  AVBD usable as whole-scene solver     "Ultimate hand" gameplay works
            ↓                                    ↓
  Lambda warm-starting                 Cloth / soft body / GPU
  Reduce iterations → performance      Unified solver architecture
            ↓                                    ↓
              Multiplayer determinism across all the above
```

> See [docs/AVBD_SOLVER_README.md](docs/AVBD_SOLVER_README.md) and [docs/SOLVER_ALGORITHM_ANALYSIS.md](docs/SOLVER_ALGORITHM_ANALYSIS.md) for details.

## AVBD Solver Overview

This implementation is a **position-based constraint solver** that combines:
- **Augmented Lagrangian (AL) outer loop** — Multiplier updates with correct sign convention (`λ = max(0, λ - ρC)`)
- **Local system solve (default 3x3 decoupled)** — Per-body position/rotation solve using AL terms
- **Optional full 6x6 LDLT solve** — Coupled position+rotation local solve via `enableLocal6x6Solve`
- **Jacobi within body, Gauss-Seidel between bodies** — Reduces intra-body bias
- **Island-level Parallelism** — Independent islands solve concurrently

> ⚠️ **Note**: The default path is the decoupled 3x3 local solve (`enableLocal6x6Solve = false`). It preserves the AL formulation while reducing cost compared to the full 6x6 path. The earlier hybrid approach is treated as a long-term research target. See [SOLVER_ALGORITHM_ANALYSIS.md](docs/SOLVER_ALGORITHM_ANALYSIS.md) for details.

### Comparison with TGS/PGS

| Property | PGS | TGS | AVBD (3x3) | AVBD (6x6) |
|----------|-----|-----|----------------|------------|
| Solve Level | Velocity | Velocity | **Position** | **Position** |
| Convergence | Linear | Sublinear | AL-augmented linear | Quadratic |
| Stack Stability | Fair | Good | **Good (4×8 iter)** | **Excellent** |
| Mass-ratio Robustness | Poor | Fair | **Good (with AL)** | **Excellent** |
| Cost per Iteration | Low | Medium | **Low-Medium** | High |

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
├── DyAvbdSolver.h/cpp       # Core solver
├── DyAvbdDynamics.h/cpp     # PhysX integration
├── DyAvbdTasks.h/cpp        # Multi-threading
├── DyAvbdTypes.h            # Config & data structures
├── DyAvbdConstraint.h       # Constraint definitions
├── DyAvbdJointSolver.h/cpp  # Joint solving
└── DyAvbdSolverBody.h       # Body state
```

## Profiling

PVD Profile Zones available:
- `AVBD.update` - Total update time
- `AVBD.solveWithJoints` - Main solver loop
- `AVBD.blockDescentWithJoints` - Constraint iterations
- `AVBD.updateLambda` - Multiplier updates

## Known Limitations

1. **No Articulation support** - Articulated bodies not implemented
2. **No Sleep/Wake** - Bodies remain active
3. **CPU only** - No GPU acceleration

## Original PhysX Documentation

- [PhysX User Guide](https://nvidia-omniverse.github.io/PhysX/physx/index.html)
- [API Documentation](https://nvidia-omniverse.github.io/PhysX)

## License

NVIDIA PhysX BSD-3-Clause. See [LICENSE.md](LICENSE.md).
