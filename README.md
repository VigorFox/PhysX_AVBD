# NVIDIA PhysX + AVBD Solver

> 🔬 **Research Fork**: Experimental AVBD (Augmented Variable Block Descent) constraint solver integrated into NVIDIA PhysX SDK.

Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved. BSD-3-Clause License.

Upstream baseline: **NVIDIA PhysX 5.9.0** (`110.1-omni-and-physx-5.9.0`, `517a0073715120e114ee055b63b26c95e00d9039`).

## ⚠️ Project Status

Status Legend: `Integrated` = merged into main code path; `Accepted` = integrated and fully validated by current acceptance gates; `Early` = prototype path exists but is not acceptance-validated and still has major gaps; `Pending` = not complete or acceptance not closed.

| Feature | Status | Notes |
|---------|--------|-------|
| Rigid Body Solver | ✅ Accepted | Contacts + unified AVBD local solve |
| **D6 Unified Joint** | ✅ Accepted | All joint types (Spherical, Fixed, Revolute, Prismatic) unified into single D6 constraint path |
| Joint Limits | ✅ Accepted | Revolute angle, Prismatic linear, Spherical cone, D6 per-axis |
| Motor Drive | ✅ Accepted | Post-solve torque motor for revolute; SLERP drive for D6 |
| Gear Joint | ✅ Accepted | Velocity-ratio constraint with post-solve motor |
| Standalone Alignment | ✅ Accepted | Rigid/joint D6 path is aligned with avbd_standalone; standalone soft body has progressed further than the current PhysX port |
| Regression Baseline | ⏳ Pending | Standalone: 118/118; the checked articulation gate is stable again, while full PhysX acceptance remains open on SnippetJoint impact coverage and the moving-mesh stress criterion |
| O(M) Constraint Lookup | ✅ Accepted | Eliminates O(N²) complexity |
| Multi-threaded Islands | ✅ Accepted | Per-island constraint mappings |
| Friction Model | ✅ Accepted | Coulomb cone, per-material coefficients from PxContactPatch |
| Soft Body | ⚠️ Early | The PhysX AVBD soft-body path remains an early research implementation with performance and architecture gaps |
| Moving Triangle-Mesh Contact | 🔧 Integrated | Rigid bodies on a vertex-updated triangle mesh pass the default 7200-frame gate; the separate stress policy remains open |
| Custom Joint | ⏳ Pending | Custom constraint callbacks unsupported |
| Rack & Pinion | ⏳ Pending | RackAndPinionJoint unsupported |
| Mimic Joint | ⏳ Pending | MimicJoint unsupported |
| Articulation | ✅ Accepted | Pure AVBD penalty path passes the strengthened 31/31 suite; asymmetric angular limits and the multi-cycle scissor-lift gate are covered headlessly |
| Sleep / Wake | ⏳ Pending | Not implemented |

**For research and evaluation only. Not production-ready.**

## Recent Progress

### PhysX 5.9 Alignment (2026-07)

- Merged the official PhysX 5.9.0 baseline and regenerated the Windows CPU-only checked solution with the 5.9 `/MD` runtime layout.
- Aligned AVBD with the 5.9 allocator, pinnable bitmap, threshold stream, island-edge traversal, D6 angular-drive slots, and constraint metadata APIs.
- Restored the upstream 16-bit low-level constraint flags. AVBD joint concrete types now use a solver-side table instead of consuming flag bits or changing the shared `Dy::Constraint` layout.
- Synchronized AVBD articulation write-back with both 5.9 motion-velocity buffers, preventing driven and falling articulations from being put to sleep using stale velocity state.
- The post-migration checked gates pass: standalone `118/118`, articulation `31/31`, the 10000-frame scissor lift, SnippetJoint, and the default/sphere-shot moving-mesh tests.

### Moving Triangle-Mesh Contact Stability (2026-07)

- Despite its name, `SnippetDeformableMesh` is not a soft-body solver test. Its boxes and shot sphere are rigid dynamics, while the ground is a rigid-static triangle mesh whose vertices are rewritten each frame and mirrored into AVBD as a kinematic shell. The gate exercises rigid-body contact against a moving/deforming surface, not soft-body elasticity, FEM, or soft-particle dynamics.
- The long-run box launch and fall-through defect in `SnippetDeformableMesh` is fixed for the default stack. The AVBD headless gate now runs 7200 frames and completes repeatably with `maxSpeed=27.5623`, `settledSunkBoxes=0`, and `ok=1`.
- The recovery includes exact narrow-phase contact-row allocation, identity-checked and task-safe contact caching, reference-aligned body-vs-static semantics, and task-owned solver statistics.
- `SnippetJoint` headless output has been reduced from thousands of periodic node dumps to configuration plus final revolute, prismatic, and fixed summaries.
- This result does not claim TGS footprint parity or closure of the separate, looser `--headless-stress` policy; both remain known limitations of the current moving-mesh validation envelope.

### Articulation Correctness Recovery (2026-07)

- Fixed-base articulation roots now enter AVBD as fully initialized static solver bodies with zero inverse mass and inverse inertia, matching PhysX filtering and write-back semantics.
- Articulation angular coordinates measure frame B relative to frame A, while the shared D6 error row measures A relative to B. Limited twist/swing intervals are now converted as `[low, high] -> [-high, -low]`; this prevents the asymmetric scissor joints from crossing the authored stop and collapsing onto a lower branch after several cycles.
- AVBD now preserves the immediately preceding body velocity sample across solver-body gather, so adaptive position warm-starting sees the intended cross-frame acceleration instead of `prevLinearVelocity == linearVelocity` after every reinitialization.
- Test 11 is now a real floating-base ground-contact gate. It checks shape bounds against the plane and parent/child anchor coincidence instead of relying on a fixed-base chain and a permissive link-center threshold; repeated default and forced-sequential samples both pass 30/30.
- Fixed-base behavior has its own pose assertion in Test 1. Drive coverage now checks velocity tracking, relative-frame position tracking, anchor closure, and mass-invariant acceleration drives on both twist and swing2 axes. Test 3 independently drives both sides of an asymmetric twist interval.
- `SnippetArticulationRC` now defaults to 3600 headless frames and compares base-local platform height at the same drive phase across cycles. The final AVBD 10000-frame run retained all 18 sampled cycles with 1.8% maximum relative drift, no stall, and less than 0.4 degrees of twist-limit violation; the TGS reference also passes.

### Articulation Solver (2026-03)

The March integration implemented articulation support using a **pure AVBD penalty-based architecture** — no Featherstone dependency. All articulation internal joints are solved as AL constraint rows in the same block descent loop as contacts and external D6 joints.

Note: AVBD articulation/joint solving is maximal-coordinate oriented on the solver side, but the public API still uses `PxArticulationReducedCoordinate` naming because upstream PhysX 5 removed the older solver-neutral `PxArticulation` abstraction layer.

Key achievements:
- **31/31 current regression gate**: the strengthened articulation suite is stable after fixed-base, drive, angular-limit, and validation-semantics alignment; Test 11 also passes its dedicated repeated default and sequential gates.
- **Iteration-efficiency milestone**: the April validation passed the full PhysX articulation regression at **10 solver iterations**. This reduction came from D6/articulation lambda warm-starting, conservative early-stop, targeted articulation iteration diagnostics, and a solver-side fix for `eACCELERATION` drive semantics.
- **12 bugs fixed** during integration: motion encoding (2-bit-per-axis), position drive error computation, eFIX penalty boost, iteration count byte order, and more.
- **Per-island adaptive iterations**: Articulations use `setSolverIterationCounts(N)` for higher iteration budgets; contact-only islands default to 8 iterations.
- **Exceeds Featherstone hybrid ceiling**: The alternating-solve lag in Featherstone coupling was the dominant error source for strongly coupled systems. Unified penalty solving eliminates this boundary.
- **Standalone**: full suite now passes at 118/118 (101 rigid/artic + 17 soft body). The rigid/artic lineage still includes convergence acceleration (Anderson Acceleration 47%, Chebyshev 29%), ID extraction via λ*, solver-is-IK, and mimic joints.

### Articulation Iteration Efficiency (2026-04)

Recent work focused on lowering the articulation iteration budget globally instead of only tuning a single snippet scene.

- **Warm-start extension**: D6/articulation joints now reuse cached AL multipliers across frames, not just contacts.
- **Measurement-first diagnostics**: `PHYSX_AVBD_ITER_DIAG`, `PHYSX_AVBD_ITER_DIAG_EVERY`, and `PHYSX_AVBD_ITER_DIAG_SEQUENTIAL` expose requested vs executed iterations, joint-row composition, and dominant lambda sources so bottlenecks can be localized before retuning.
- **Drive semantic fix**: articulation-internal `PxArticulationDriveType::eACCELERATION` is now handled in the solver using response-scaled implicit coefficients instead of being approximated only in constraint prep.
- **Historical verified floor**: the April full-suite validation passed at **10** iterations. **8** iterations failed in the loaded Scissor Lift case. Test 11 itself is now stable across dedicated 8/16/32/64-iteration sweeps, so that older failure should not be attributed to its ground-contact path.

### D6 Unification

All joint types have been unified into a single D6 constraint path. Per-type independent solvers (Spherical, Fixed, Revolute, Prismatic) have been replaced by one shared `addD6Contribution()` / `updateD6Dual()` pipeline, with joint behavior determined entirely by motion masks (LOCKED/FREE/LIMITED per DOF).

Key changes:
- **Architecture**: ~400 lines of redundant per-type constraint code removed; all joints route through unified D6 primal + dual path.
- **Angular constraints**: Cross-product axis alignment for revolute-pattern D6 joints, replacing quaternion tangent-space error. Immune to twist-angle amplification at large rotations.
- **Angular error**: Axis-angle decomposition (`2·acos(w)·axis`) replaces tangent-space `2·vec(errQ)`, accurate at large angles.
- **Motor**: Post-solve torque motor decoupled from ADMM constraint Hessian, replacing in-iteration AL velocity drive.
- **Gear joint**: Dual update moved inside ADMM iteration loop; NaN from driveForceLimit overflow fixed.
- **Cone limit**: Per-body joint frame axes derived from `localFrameA`/`localFrameB`, replacing shared axis.
- **Joint frames**: `localFrameB` derived from initial relative rotation at joint creation. All factory methods updated.
- **Standalone sync**: rigid/joint D6 behavior remains aligned with `avbd_standalone`, while standalone soft body has already moved to a VBD+AVBD path that is not yet mirrored by the current PhysX port.

### Friction Integration

Friction was already fully implemented in the AVBD solver (3-DOF contact model: 1 normal + 2 tangent), but PhysX contact preparation hardcoded `friction = 0.5f` and `restitution = 0.0f` instead of reading from materials.

Key changes:
- **Material read-through**: `constraint.friction` and `constraint.restitution` now read from `PxContactPatch::dynamicFriction` / `restitution` (combined by narrowphase).
- **Tangent basis**: Aligned with standalone — `PxAbs(normal.y) > 0.9f` branch for robustness.
- **Standalone tests**: 18 friction-specific tests (slope sliding, anisotropy, Coulomb cone, geometric mean combining, warmstart, penalty growth, etc.).

### Soft Body Status (EARLY)

The PhysX AVBD soft-body path is still in an early prototype stage.

- Native AVBD soft-particle/VBD pieces and the current OGC-based collision experiments exist. `SnippetDeformableMesh` is not evidence for this path because all simulated bodies in that scene are rigid.
- `avbd_standalone` soft body is already accepted with a full 118/118 standalone pass set (101 rigid/artic + 17 soft body), but that maturity has not yet carried over to the current PhysX port.
- It is **not** part of the accepted regression baseline summarized above.
- Current implementation has **major performance problems** and should be treated as a research path, not a production-ready or even feature-complete baseline.
- Near-term work is expected to focus on architecture cleanup, data layout, and performance before soft body results should be interpreted as representative.

### Current Validation Snapshot

- ✅ Upstream baseline and checked binaries are PhysX 5.9.0; Windows CPU-only outputs use `win.x86_64.vc143.md`.
- ✅ Standalone full suite passes: `118 PASSED / 0 FAILED`.
- ✅ Checked builds pass for `SnippetJoint`, `SnippetAvbdArticulation`, `SnippetArticulationRC`, and `SnippetDeformableMesh`.
- ✅ The default AVBD moving-triangle-mesh rigid-stack gate passes 7200 headless frames with finite state and zero settled sunk boxes; the TGS reference also passes.
- ✅ Focused articulation Test 3 passes `2/2`, Test 16 passes `9/9`, and Test 17 passes `1/1`; `SnippetArticulationRC` passes its 3600-frame cycle gate and a 10000-frame AVBD extension with 18 samples and 2.4% maximum relative drift, without a stall, non-finite state, or `Illegal BroadPhaseUpdateData`.
- ✅ Friction reads per-material coefficients; Coulomb cone and augmented-Lagrangian behavior remain validated.
- ✅ The strengthened full articulation suite passes `31/31`; focused Test 11 passes `30/30` under both default and forced-sequential execution with finite geometry and bounded joint-anchor error.
- ⚠️ `SnippetJoint` produces a bounded, clean headless smoke result, but the required one-sphere-per-chain impact launcher is not currently present, so the formal impact gate remains pending.
- ⚠️ The moving-mesh `--headless-stress` policy remains looser than the default stack gate and is not evidence of zero transient sinking.
- ⚠️ PhysX AVBD soft body remains an early research path with major performance work still pending.

## SnippetChainmail Demo

https://github.com/user-attachments/assets/2ab299c7-8f7f-4bf2-b8b5-7de8033b17f8

## Why AVBD?

PhysX's built-in TGS/PGS are **velocity-level** iterative solvers that hit fundamental limits in several scenarios:

| Problem | TGS/PGS Limitation | AVBD Direction |
|---------|---------------------|----------------|
| **High mass-ratio joints** | Condition number explosion, rubber-banding | Augmented Lagrangian + local Hessian solve |
| **Multiplayer sync** | Velocity integration accumulates FP error | Position-level solve with stronger state consistency |
| **Cloth & soft body** | Requires separate solver pipelines | Position-level framework is more naturally extensible |

AVBD introduces a **unified position-level constraint solving framework** targeting:

1. Stable high mass-ratio interaction chains.
2. Whole-scene robustness under mixed contact/joint constraints.
3. Better deterministic behavior for server-authoritative simulation.
4. Future rigid/soft-body unification on a common optimization-style solver structure.

### Roadmap Snapshot

```
Contact AL stability (DONE)         D6 Unified Joint System (DONE)
  Rigid body contacts stable      ->  All joints unified into D6 path
  AVBD usable as whole-scene solver   Spherical/Fixed/Revolute/Prismatic/D6/Gear: accepted
            |                                    |
  Lambda warm-starting (DONE)        Articulation Solver (ACCEPTED)
  Iteration-efficiency tuning        31/31 plus multi-cycle RC gate
            |                        Per-island adaptive iterations
            |                                    |
Soft body / performance / GPU path (EARLY)
	SOA refactoring, multiplayer determinism
```

## Solver Architecture

### Unified AVBD Hessian Approach

The solver accumulates **contacts and joints** into a per-body local system (typically 6x6), then solves via LDLT:

```
For each body i:
	H = M/h^2 * I_6x6
	g = M/h^2 * (x_i - x_tilde)

	For each contact/joint row:
		H += rho_eff * J^T J
		g += J * (rho_eff * C + lambda)

	Dual update (stabilized AL):
		rhoDual = min(Mh^2, rho^2/(rho + Mh^2))
		lambda  = decay * lambda + rhoDual * C

	delta = LDLT_solve(H, g)
	x_i -= delta
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Unified D6 joint path** | All joint types (Spherical, Fixed, Revolute, Prismatic) map to a single D6 constraint with motion masks. |
| **Cross-product axis alignment** | Revolute-pattern angular constraints use `twistA x twistB` instead of quaternion error, avoiding twist amplification. |
| **Post-solve motor** | Motor torque applied after ADMM iterations, decoupled from constraint Hessian for stability. |
| **Stabilized AL dual for joints** | Bounded dual step + decay (`rhoDual`, `lambdaDecay`) reduces overshoot while retaining AL memory. |
| **Prismatic force-6x6 on touch** | Prevents instability from 3x3 decoupling under strong position-rotation coupling. |
| **Standalone/PhysX algorithm parity** | Rigid/joint paths share the same core constraint formulation and dual update logic; standalone soft body has advanced to a VBD+AVBD path that is not yet fully mirrored in PhysX. |

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
./generate_projects.bat vc17win64-cpu-only  # Windows
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
├── DyAvbdSolver.h/cpp            # Shared contact and post-AL solver stages
├── DyAvbdSolverJointPath.cpp     # Solver joint path (solveWithJoints entry)
├── DyAvbdKinematicShell.h/cpp    # Moving triangle-mesh contact bridge
├── DyAvbdJointProjection.h/cpp   # Per-joint-type constraint projection & multiplier update
├── DyAvbdDynamics.h/cpp          # PhysX integration & frame orchestration
├── DyAvbdDynamicsPrep.cpp        # Contact & joint constraint preparation
├── DyAvbdTasks.h/cpp             # Multi-threading
├── DyAvbdTypes.h                 # Config & data structures
├── DyAvbdConstraint.h            # Constraint definitions
└── DyAvbdSolverBody.h/cpp        # Body state
```

## Profiling

PVD Profile Zones available:
- `AVBD.update` - Total update time
- `AVBD.solveWithJoints` - Main solver loop
- `AVBD.blockDescentWithJoints` - Constraint iterations
- `AVBD.updateLambda` - Multiplier updates

## Known Limitations

1. **No Sleep/Wake** - Bodies remain active
2. **CPU only** - No GPU acceleration
3. **Articulation low-budget edge cases** - The April validation floor was 10 iterations; the loaded Scissor Lift case still fails at 8
4. **Soft body performance** - The current PhysX AVBD soft-body path remains early-stage and has major performance problems
5. **SnippetJoint impact coverage** - The checked harness currently provides a bounded no-impact smoke, not the required one-sphere-per-chain gate
6. **Moving-mesh stress policy** - The default `SnippetDeformableMesh` stack is recovered, but the separate stress acceptance criterion still needs tightening

## Original PhysX Documentation

- [PhysX User Guide](https://nvidia-omniverse.github.io/PhysX/physx/index.html)
- [API Documentation](https://nvidia-omniverse.github.io/PhysX)

## License

NVIDIA PhysX BSD-3-Clause. See [LICENSE.md](LICENSE.md).
