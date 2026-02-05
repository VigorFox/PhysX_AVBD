# NVIDIA PhysX + AVBD Solver

> 🔬 **Research Fork**: Experimental AVBD (Augmented Variable Block Descent) constraint solver integrated into NVIDIA PhysX SDK.

Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved. BSD-3-Clause License.

## ⚠️ Project Status

| Feature | Status | Notes |
|---------|--------|-------|
| Rigid Body Solver | ✅ Working | Contacts + 6 joint types |
| Joint System | ✅ Working | Revolute, Prismatic, Spherical, Fixed, D6, Gear |
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

## AVBD Solver Overview

AVBD is a position-based constraint solver using:
- **Block Coordinate Descent** - Per-body 6x6 local system solve
- **Augmented Lagrangian** - Multiplier updates for constraint satisfaction
- **Island-level Parallelism** - Independent islands solve concurrently

### Comparison with TGS/PGS

| Property | PGS | TGS | AVBD |
|----------|-----|-----|------|
| Solve Level | Velocity | Velocity | **Position** |
| Convergence | Linear | Sublinear | Quadratic |
| Stack Stability | Fair | Good | **Excellent** |
| Cost per Iteration | Low | Medium | Medium-High |

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
