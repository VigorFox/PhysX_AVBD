# NVIDIA PhysX + AVBD Solver

Experimental integration of an **Augmented Vertex Block Descent (AVBD)**
position-level solver into NVIDIA PhysX.

This research fork targets a common solver framework for rigid bodies, joints,
articulations, cloth and soft bodies. It is based on **NVIDIA PhysX 5.9.0**
(`110.1-omni-and-physx-5.9.0`,
`517a0073715120e114ee055b63b26c95e00d9039`).

> This project is intended for research and evaluation. It is not
> production-ready.

## Status

`Accepted` means the implementation is integrated and covered by the current
headless acceptance gates. It does not imply production readiness.

| Area | Status | Current boundary |
|---|---|---|
| Rigid contacts | Accepted | Position-level contact, friction, restitution, sleep/wake, lock flags and CCD |
| Joints | Accepted | Unified D6 path for fixed, spherical, revolute, prismatic, D6 and gear joints |
| Articulations | Accepted | Pure AVBD constraint path, drives, limits, mimic and selected tendon topologies |
| Contact extensions | Accepted | Contact modification/reporting, custom geometry and custom convex paths |
| Moving triangle meshes | Integrated | Default and sphere-shot gates pass; wider stress policy remains open |
| CPU cloth and soft bodies | Accepted for correctness | Public `PxScene` Surface/Volume lifecycle, interaction, collision and material semantics |
| Performance and GPU | In progress | CPU ISA dispatch foundation is integrated; AoSoA/SIMD solver kernels and the AVBD GPU backend remain incomplete |

## What AVBD Adds

- A position-level block-coordinate solver using local body/vertex Hessians and
  augmented-Lagrangian multiplier updates.
- A unified D6 formulation for ordinary joints and articulation-internal
  constraints.
- Typed compiled objective ownership, so each physical source has one explicit
  solve or finalize owner.
- Island-level parallel execution with per-island solver budgets.
- A public CPU `PxScene` backend for `PxDeformableSurface` and
  `PxDeformableVolume`.
- Dedicated headless Snippet runners and source-level anti-regression gates.

## CPU Deformables

The CPU AVBD deformable path is correctness-complete for the public API surface
currently implemented in this branch.

- Scene-owned actor lifecycle, host-buffer synchronization, bounds, sleep/wake,
  writeback and teardown.
- Surface bending, flattening and bending damping.
- Volume co-rotational and Neo-Hookean material models.
- Static, kinematic and dynamic rigid-soft interaction.
- Soft-soft and self collision with discrete and swept OGC vertex-face and
  edge-edge ownership.
- Plane, box, sphere, capsule, convex, triangle-mesh and heightfield collision
  within their documented public support boundaries.
- Deformable attachments, element filters, kinematic targets and CPU
  Surface/Volume skinning.
- CPU tetrahedral cooking without a required GPU payload.

The implementation is still correctness-first. Collision redetection, swept
OGC queries, mapping scans and temporary allocations are known optimization
targets. CPU AVBD results should not be used as a direct efficiency comparison
with native GPU FEM until an AVBD GPU backend exists.

## Validation Snapshot

Checkpoint: **2026-07-31**

| Gate | Result |
|---|---:|
| Standalone AVBD suite | `149/149` |
| Articulation suite | `31/31` plus 10,000-frame scissor-lift run |
| Joint-drive matrix | `1176/1176` |
| Private soft-body acceptance | `202/202`, repeated twice |
| Public Surface matrix | 94 cases × 2 repeats, parallel and sequential |
| Public Volume matrix | 96 cases × 2 repeats, parallel and sequential |
| AVBD source gates | `37/37` |
| Shared-DLL cross matrix | `14/14` |
| Mixed-owner matrix | `8/8` |
| FEM feature Snippets | `4/4` |
| Strict `/W4 /WX` rebuilds | `6/6` |

Render-built validation executables are launched through the dedicated hidden
Python runners in `tools/`; the runners reject visible windows, enforce
timeouts and clean up process trees.

## Quick Start

Generate projects:

```powershell
Set-Location physx
.\generate_projects.bat vc17win64-cpu-only
```

Linux project generation uses `generate_projects.sh` with the desired preset.

Select AVBD when creating a scene:

```cpp
PxSceneDesc sceneDesc(physics->getTolerancesScale());
sceneDesc.solverType = PxSolverType::eAVBD;
```

The main validation entry points are:

```text
tools/run_snippet_soft_body_avbd_headless.py
tools/run_snippet_deformable_surface_avbd_headless.py
tools/run_snippet_deformable_volume_avbd_headless.py
tools/run_snippet_deformable_avbd_feature_demos_headless.py
```

## Architecture

For each rigid body or soft vertex block, AVBD accumulates inertial, elastic and
constraint contributions into a local system:

```text
H = M / h² + Σ ρ JᵀJ
g = M / h² (x - x̃) + Σ Jᵀ(ρC + λ)
Δx = -H⁻¹g
```

The solver alternates local block updates with stabilized augmented-Lagrangian
dual updates. Contacts, joints, attachments and deformable constraints enter
through compiled objective programs rather than runtime ownership bit
combinations.

Important implementation areas:

```text
physx/source/lowleveldynamics/src/
  DyAvbdSolver.cpp
  DyAvbdSolverJointPath.cpp
  DyAvbdSoftBodyComponent.h
  DyAvbdDynamics.cpp

physx/source/simulationcontroller/src/ScScene.cpp
physx/snippets/snippetsoftbodyavbd/
physx/snippets/snippetdeformablesurfaceavbd/
physx/snippets/snippetdeformablevolumeavbd/
```

## Profiling

The existing PVD profile zones include:

- `AVBD.update`
- `AVBD.solveWithJoints`
- `AVBD.blockDescentWithJoints`
- `AVBD.updateLambda`

Soft-body performance work should separately measure collision preparation,
elastic/contact solving, host-buffer writeback and CPU skinning.

## Known Boundaries

- No AVBD GPU backend is currently available.
- Public CPU deformable correctness is accepted, but whole-solver performance
  has not been optimized.
- Native-island mid-step Scene topology recreation and public CPU stress-tensor
  readback remain open architecture/API work.
- Surface anisotropy requires a future public material contract.
- Wider joint/tendon topologies, very low articulation iteration budgets and
  selected contact-combined variants remain outside the accepted gates.
- Serialization/PVD, concurrent Scene, extreme scale and long-duration soak
  coverage require further hardening.
- The separate moving-mesh stress policy remains less strict than the default
  acceptance path.

## Demo

[SnippetChainmail video](https://github.com/user-attachments/assets/2ab299c7-8f7f-4bf2-b8b5-7de8033b17f8)

## Upstream Documentation

- [PhysX User Guide](https://nvidia-omniverse.github.io/PhysX/physx/index.html)
- [PhysX API Documentation](https://nvidia-omniverse.github.io/PhysX)

## License

Copyright (c) 2008-2026 NVIDIA Corporation.

NVIDIA PhysX is distributed under the BSD-3-Clause license. See
[LICENSE.md](LICENSE.md).
