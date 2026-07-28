# NVIDIA PhysX + AVBD Solver

> 🔬 **Research Fork**: Experimental AVBD (Augmented Variable Block Descent) constraint solver integrated into NVIDIA PhysX SDK.

Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved. BSD-3-Clause License.

Upstream baseline: **NVIDIA PhysX 5.9.0** (`110.1-omni-and-physx-5.9.0`, `517a0073715120e114ee055b63b26c95e00d9039`).

## ⚠️ Project Status

Status Legend: `Integrated` = merged into main code path; `Accepted` = integrated and fully validated by current acceptance gates; `Early` = prototype path exists but is not acceptance-validated and still has major gaps; `Pending` = not complete or acceptance not closed; `Unsupported` = currently functionally unavailable with AVBD.

| Feature | Status | Notes |
|---------|--------|-------|
| Rigid Body Solver | ✅ Accepted | Contacts + unified AVBD local solve |
| **D6 Unified Joint** | ✅ Accepted | All joint types (Spherical, Fixed, Revolute, Prismatic) unified into single D6 constraint path |
| Joint Limits | ✅ Accepted | Revolute angle, Prismatic linear, Spherical cone, D6 per-axis |
| Motor Drive | ✅ Accepted | Post-solve torque motor for revolute; SLERP drive for D6 |
| Gear Joint | ✅ Accepted | Velocity-ratio/phase/external-impulse physics plus binary round-trip dependency identity and post-load consumption (`30/30`) |
| Standalone Alignment | ✅ Accepted | Rigid/joint D6 path is aligned with avbd_standalone; standalone soft body has progressed further than the current PhysX port |
| Regression Baseline | ✅ Accepted | Breadth checkpoint: standalone `149/149`; all 46 AVBD in-scope CPU Snippets have reproducible headless gates |
| O(M) Constraint Lookup | ✅ Accepted | Eliminates O(N²) complexity |
| Multi-threaded Islands | ✅ Accepted | Per-island constraint mappings |
| Friction Model | ✅ Accepted | Coulomb cone, per-material coefficients from PxContactPatch |
| Soft Body | ⚠️ Early | Existing CPU component correctness is gated by `SnippetSoftBodyAVBD` and `SnippetDeformableVolumeAVBD`; public CPU `PxScene` deformable actors and performance remain open |
| Moving Triangle-Mesh Contact | 🔧 Integrated | Rigid bodies on a vertex-updated triangle mesh pass the default 7200-frame gate; the separate stress policy remains open |
| Custom Joint | ✅ Accepted | `SnippetCustomJoint` retains the pulley public-wrench/break matrix (`12/12`) plus multi-row aggregation, force spring, restitution and force drive-limit modes (`12/12`) |
| Rack & Pinion | ✅ Accepted | `SnippetRackJoint` gates positive/negative ratio, bidirectional passive response and binary round-trip consumption for the centered principal-axis topology (`18/18`); wider topology remains pending |
| Mimic Joint | ✅ Accepted | `SnippetMimicJoint` retains its hard ratio/offset/bidirectional matrix (`12/12`) and adds compliant response plus two simultaneous mimic equations (`12/12`); shared-endpoint and wider topology remain pending |
| Fixed Tendon | ✅ Accepted | `SnippetFixedTendon` gates fixed-root serial angular plus sibling branch angular/linear tendons, rest spring/damping, runtime offset actuation and branch length limits; asymmetric reciprocal coefficients and wider trees remain pending |
| Spatial Tendon | ✅ Accepted | `SnippetSpatialTendon` gates fixed/moving intermediate paths, shared-root multi-leaf rows, angular/linear endpoints, rest spring/damping and leaf-local limits (`12/12` new modes plus `6/6` legacy); moving-root, contact and wider topology remain pending |
| Contact Modification | ✅ Accepted | `SnippetContactModification` gates modified normal, normal/tangent target velocity, zero and finite max-impulse response, and zero plus finite asymmetric local mass/inertia scales against unit-scale controls for TGS plus AVBD parallel/sequential execution (`27/27`) |
| Contact Report | ✅ Accepted | `SnippetContactReport` gates normal point/impulse payload, CPU PGS-only force-threshold `FOUND/LOST` events, and public friction-anchor positions/impulses for body-static and ordinary dynamic-dynamic contacts (`10/10`) |
| Continuous Collision Detection | ✅ Accepted | `SnippetCCD` gates built-in linear, speculative/angular and combined full CCD, linear dynamic-dynamic CCD, and the raycast extension against static/dynamic targets, with three no-CCD tunneling controls (`24/24`) |
| CCD Contact Report | ✅ Accepted | `SnippetContactReportCCD` gates the original multi-hit report, angular speculative `TOUCH_FOUND`, combined full-CCD found/sweep events, dynamic-dynamic CCD payload, and the RaycastCCD post-fetch boundary (`16/16`) |
| Custom Geometry Collision | ✅ Accepted | `SnippetCustomGeometryCollision` gates callback-generated falling, sliding, and impact contacts through TGS plus AVBD parallel/sequential execution, with nonzero solver response and no fall-through |
| Voxel Custom Geometry | ✅ Accepted | `SnippetCustomGeometry` gates voxel callback-generated drop and impact contacts through TGS plus AVBD parallel/sequential execution, with public nonzero impulses and complete teardown |
| Custom Convex | ✅ Accepted | `SnippetCustomConvex` gates instrumented cylinder/cone callbacks through TGS plus AVBD parallel/sequential execution, with finite generated contacts, public nonzero impulses, no fall-through, and callback-safe teardown |
| Articulation | ✅ Accepted | Pure AVBD penalty path passes the strengthened 31/31 suite; asymmetric angular limits and the multi-cycle scissor-lift gate are covered headlessly |
| Sleep / Wake | ✅ Accepted | Free-body idle sleep, static-contact settling, contact-island wake propagation, re-sleep, and disable-sleep behavior are headlessly gated |
| Rigid-Body Lock Flags | ✅ Accepted | All six public linear/angular lock flags are gated against TGS for initial velocity and runtime impulse excitation in AVBD parallel/sequential execution |

**For research and evaluation only. Not production-ready.**

## Recent Progress

### PhysX 5.9 Alignment (2026-07)

- Merged the official PhysX 5.9.0 baseline and regenerated the Windows CPU-only checked solution with the 5.9 `/MD` runtime layout.
- Aligned AVBD with the 5.9 allocator, pinnable bitmap, threshold stream, island-edge traversal, D6 angular-drive slots, and constraint metadata APIs.
- Restored the upstream 16-bit low-level constraint flags. AVBD joint concrete types now use a solver-side table instead of consuming flag bits or changing the shared `Dy::Constraint` layout.
- Synchronized AVBD articulation write-back with both 5.9 motion-velocity buffers, preventing driven and falling articulations from being put to sleep using stale velocity state.
- Connected ordinary AVBD rigid-body write-back to the shared PhysX sleep lifecycle. Idle sleep, static-contact settling, contact-island wake propagation, re-sleep, and `eDISABLE_SLEEPING` now have headless gates.
- Copied and enforced all six `PxRigidDynamicLockFlag` axes through AVBD prediction, constraint stages, final pose, and velocity write-back. Initial-velocity and runtime-impulse witnesses pass in parallel and sequential execution.
- Expanded `SnippetCCD` from a static-wall smoke into a `24/24` matrix covering angular/full CCD, built-in and raycast dynamic-dynamic paths, public flag readback, real negative controls, finite response, and cleanup. The review found no additional AVBD solver defect.
- Expanded `SnippetContactModification` to a `27/27` matrix and fixed AVBD finite `maxImpulse` semantics across the primal solve, split depenetration, and final normal-velocity response; a `0.25` impulse cap now matches the TGS pass-through witness.
- Expanded `SnippetContactReport` to a `10/10` matrix. Force-threshold events follow their documented CPU PGS-only boundary, while AVBD now exposes public friction anchors and applied tangential impulses for both body-static and ordinary dynamic-dynamic contacts through `PxContactPair::extractFrictionAnchors()`.
- Expanded `SnippetContactReportCCD` to a `16/16` matrix covering the original multi-hit sweep, angular speculative and full CCD event semantics, true dynamic-dynamic sweep payload, and the RaycastCCD post-fetch correction boundary. This extension found no additional AVBD solver defect.
- The current checked gates pass: standalone `149/149`, cross-Snippet `14/14`, Contact Report `10/10`, CCD Contact Report `16/16`, Contact Modification `27/27`, CCD `24/24`, articulation `31/31`, the 10000-frame scissor lift, SnippetJoint, and the default/sphere-shot moving-mesh tests.

### Rigid Mixed-Solve Ownership Tightening (2026-07)

- Ordinary contact-only rigid material components with zero restitution and at most 16 contact points now use one complete owner for simultaneous normal and two-axis Coulomb response. Candidate state is committed atomically only after the complete component passes its physical residual checks.
- Nonzero-restitution components and components above 16 contact points fail closed as whole components. They are not split into partially owned row subsets. A nonzero-restitution component owner was implemented and then reverted because it deterministically regressed the unchanged full `SnippetToleranceScale` acceptance gate; no gate threshold or scene-specific exception was introduced.
- The final focused gates pass: P4AI passive material component `12/12`, P4AF passive rigid-static manifold `12/12`, Contact Modification `27/27`, full ToleranceScale `6/6`, restitution spatial `6/6`, and restitution-threshold `12/12`.
- The fixed owner inventory passes `8/8`, but fallback is not yet rare: four cases still report 4141 sampled fallback rows and 24531 corrections. HelloWorld stack/ball account for 3780 of those rows; articulation joint-mixed ownership is a separate remaining gap.
- The wider six/12/24-body standalone fixtures establish scalable-solver authority and failure-first behavior, but do not enable the rejected production path. P3R remains opt-in/default-off. Restitution-capable and scalable material-component ownership are the next dedicated correctness task.

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

### Articulation Mimic Coupling (2026-07)

- AVBD ingests articulation-internal mimic equations as shared 1D rows. Hard rows close both position and velocity equations; compliant rows use mass-scaled natural-frequency/damping-ratio coefficients and retain the hard mimic velocity derivative.
- The original hard matrix remains `12/12 PASS`. A separate compliant/multi-mimic failure-first pack moved from TGS `4/4 PASS` plus AVBD `8/8 real FAIL` (stationary followers) to a complete `12/12 PASS`.
- The accepted topology is fixed-parent, sibling, centered and single-DOF. The multi fixture proves two disjoint mimic equations in one articulation; shared-endpoint graphs and wider topology remain future hardening.

### Articulation Fixed Tendon Coupling (2026-07)

- AVBD emits compliant articulation-internal fixed-tendon rows for fixed-root, centered, single-DOF serial angular and two-sibling branch topologies. Branch rows support angular or prismatic-X coordinates.
- The row uses the public fixed-tendon length equation, physical rest stiffness/damping and additive low/high length-limit stiffness. It remains separate from hard generic/mimic rows: no AL dual multiplier or exact hard projection is applied.
- The branch/linear/limit failure-first pack moved from TGS `6/6 PASS` plus AVBD `12/12 real FAIL` with stationary followers to a complete `18/18 PASS`; the original serial drive/offset matrix remains `6/6 PASS`.
- Coefficient and reciprocal coefficient must currently be equal. Asymmetric reciprocal response, wider trees, serial linear/limit and mixed-axis paths remain outside the accepted boundary.

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

The PhysX AVBD soft-body path is still in an early prototype stage, but its existing CPU component now has an accepted correctness baseline.

- Native AVBD soft-particle/VBD pieces and the current OGC-based collision experiments exist. `SnippetSoftBodyAVBD` passes its self-contained component suite `74/74`; `SnippetDeformableVolumeAVBD` passes five isolated component/lifecycle cases repeated twice (`10/10`) with finite state and zero inverted tetrahedra.
- `SnippetDeformableVolumeAVBD` directly owns and steps soft-particle arrays. It reports `sceneSoftIntegration=0`: its `PxScene` contains no public deformable actor, so this is component correctness plus AVBD scene coexistence—not a public CPU `PxScene` deformable backend gate.
- `SnippetDeformableMesh` is not evidence for this path because all simulated bodies in that scene are rigid.
- `avbd_standalone` passes its full `149/149` suite. A direct port of the PhysX positive-J displacement limiter regressed standalone material semantics and was reverted rather than forced into alignment.
- The two existing PhysX CPU soft-body component Snippets are now part of the regression baseline; public scene integration remains outside that accepted slice.
- Current implementation has **major performance problems** and should be treated as a research path, not a production-ready or even feature-complete baseline.
- Soft-body optimization remains deferred until the post-inventory capability review closes any remaining functional correctness blockers.

### Current Validation Snapshot

- Checkpoint date: **2026-07-28**.
- The breadth-first CPU inventory is complete: **61/61** executable Snippets are classified, **46/46** AVBD `PxScene` Snippets have reproducible headless gates, and 15 CPU tools/query/cooking Snippets are outside standard solver dynamics.
- All render-built validation executables are launched through dedicated hidden Python runners with explicit `--headless` arguments, timeout handling, visible-window rejection, process-tree cleanup, and fail-closed authority parsing.
- The final standalone suite passes **149/149**. The post-relink shared-DLL cross matrix passes **14/14**.
- `SnippetJointDrive` passes its cumulative **1176/1176** matrix; dynamic angular-position is **288/288**, contact angular-position is **24/24**, moving-kinematic SLERP is **18/18**, and the public legacy elliptical cone is **6/6**.
- `SnippetJoint` retains force-pair **24/24**, external-disabled constraint **6/6**, native asymmetric spherical cone **12/12**, passive native prismatic/revolute reaction and break **18/18**, and legacy fixed no-break/break **6/6**.
- `SnippetCustomJoint` retains its public wrench/break and generic-row mode matrices (**12/12** each). Rack/Gear runtime plus binary round-trip matrices pass **18/18** and **30/30**.
- Mimic, Fixed Tendon, and Spatial Tendon public mode packs pass **12/12**, **18/18**, and **12/12**, while their legacy matrices remain green.
- Articulation coverage retains the strengthened **31/31** suite, the 3600-frame RC cycle gate, and the 10000-frame RC extension without stall or non-finite state.
- Rigid contact coverage includes stack/impact, CCD, contact report/modification, custom geometry/convex, moving mesh, tolerance scaling, gyroscopic response, split simulation/fetch, serialization, multithreading, triggers, MBP, and CPU Vehicle Snippets.
- Existing CPU soft-body components retain `SnippetSoftBodyAVBD` **74/74** and `SnippetDeformableVolumeAVBD` **10/10**. They do not represent a public CPU `PxScene` deformable backend.
- Remaining correctness boundaries are explicit: wider native-joint topology and selected extreme/contact-combined variants, including mixed lock-flag constraint topologies. Performance work remains deferred until the remaining capability checks are resolved.
- Local handoff, audit, probe history, and per-iteration reports are intentionally not part of this checkpoint.

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

1. **Lock/sleep coverage boundary** - The six free-rigid-body lock axes, static-contact sleep, and two-body contact-island wake propagation are accepted; mixed lock flags with broader contact/joint topologies remain outside the current gate
2. **Joint coverage boundaries** - generic custom modes are gated only for the accepted row packs; mimic remains limited to fixed-parent centered single-DOF siblings (including two disjoint equations), and rack-and-pinion remains limited to centered principal-axis topology
3. **CPU only** - No GPU acceleration
4. **Articulation low-budget edge cases** - The April validation floor was 10 iterations; the loaded Scissor Lift case still fails at 8
5. **Soft body architecture and performance** - Existing CPU component correctness is gated, but public CPU `PxScene` deformable actors are not integrated and the component remains slow
6. **Joint reaction and break coverage** - Native unconstrained dynamic-dynamic fixed-pair reaction, shared external `eDISABLE_CONSTRAINT` ingestion, asymmetric spherical-cone response, and centered passive prismatic/revolute reaction/break are accepted; limited, driven, motorized, off-center, contact-combined and wider native topology remain incomplete
7. **Moving-mesh stress policy** - The default `SnippetDeformableMesh` stack is recovered, but the separate stress acceptance criterion still needs tightening

## Original PhysX Documentation

- [PhysX User Guide](https://nvidia-omniverse.github.io/PhysX/physx/index.html)
- [API Documentation](https://nvidia-omniverse.github.io/PhysX)

## License

NVIDIA PhysX BSD-3-Clause. See [LICENSE.md](LICENSE.md).
