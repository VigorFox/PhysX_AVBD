# NVIDIA PhysX with AVBD

This repository is an experimental integration of an Augmented Vertex Block
Descent (AVBD) position-level solver into NVIDIA PhysX. The branch develops one
solver framework for rigid bodies, joints, articulations, deformable surfaces,
and deformable volumes.

The fork is based on NVIDIA PhysX 5.9.0 (`110.1-omni-and-physx-5.9.0`, upstream
commit `517a0073715120e114ee055b63b26c95e00d9039`). It is intended for research,
validation, and continued solver development; it is not production-ready.

## Current Status

The core AVBD pipeline is integrated and its principal rigid, articulation,
joint, and CPU-deformable paths have deterministic headless coverage. The
current implementation is correctness-first: a number of specific acceptance
gaps remain and the CPU soft-body path still has substantial optimization work.

| Area | Current state |
|---|---|
| Rigid bodies | Integrated: contacts, friction, restitution, CCD, sleep/wake, lock flags, and deterministic trace witnesses |
| Joints | Integrated through a unified D6 path; selected non-contact and gear matrices pass |
| Articulations | Integrated: parallel and sequential suites, drives, limits, mimic joints, and selected tendons pass |
| CPU deformables | Integrated: public Surface/Volume lifecycle, materials, attachments, soft-soft, self, and rigid-soft collision |
| Mixed contact | Unified OGC/contact ownership is in place; several dissipation and swept-response cases remain open |
| Performance | CPU ISA dispatch exists; broader AoSoA/SIMD kernel work and allocation/query reductions remain open |
| GPU | The AVBD GPU backend is not complete |

The maintained list of reproducible failures, closure conditions, and rerun
commands is [docs/AVBD_CORRECTNESS_DEBT.md](docs/AVBD_CORRECTNESS_DEBT.md).
Do not interpret a green build as closure of those physical acceptance debts.

## What This Fork Adds

- Position-level block-coordinate solves using per-body and per-vertex local
  systems with augmented-Lagrangian multiplier updates.
- Unified rigid, joint, articulation, and deformable scheduling inside PhysX
  islands.
- Shared OGC pair state and explicit post-AL ownership for rigid-soft and
  soft-soft contact response.
- A public CPU `PxScene` backend for `PxDeformableSurface` and
  `PxDeformableVolume`.
- Deterministic headless Snippet gates for solver correctness, sleep, energy,
  collision response, and regressions.
- Inspectable `.pxtrace` capture for the AVBD `SnippetHelloWorld` scenario.

## Build

The actively verified Windows configuration uses Visual Studio 2022 and the
CPU-only x64 preset:

```powershell
Set-Location physx
.\generate_projects.bat vc17win64-cpu-only
cmake --build .\compiler\vc17win64-cpu-only --config checked --parallel
```

To rebuild one validation target while iterating, pass its CMake target name:

```powershell
cmake --build .\compiler\vc17win64-cpu-only --config checked --target SnippetHelloWorld --parallel
```

Linux and AArch64 CPU-only presets are also present under
`physx/buildtools/presets/public/`; use `generate_projects.sh` with the matching
preset. These platforms have not received the same verification run described
below.

## Select AVBD

Set the scene solver type when constructing a `PxScene`:

```cpp
PxSceneDesc sceneDesc(physics->getTolerancesScale());
sceneDesc.solverType = PxSolverType::eAVBD;
```

Existing TGS and PGS scene selection remains available. AVBD correctness should
be evaluated through `eAVBD` directly, rather than by routing unsupported cases
through another solver.

## Examples and Diagnostics

| Snippet | Purpose |
|---|---|
| `SnippetHelloWorld` | Rigid stacks, projectiles, sleep, ground sliding, energy checks, and trace capture |
| `SnippetJoint` / `SnippetJointDrive` | D6-backed joints, limits, drives, break forces, and contact coupling |
| `SnippetAVBDArticulation` | Articulation links, drives, limits, tendons, and parallel/sequential execution |
| `SnippetDeformableSurfaceAVBD` | Cloth/surface material, attachment, self, soft-soft, and rigid-soft cases |
| `SnippetDeformableVolumeAVBD` | Volumetric soft-body materials, contacts, attachments, and wake behavior |
| `SnippetDeformableMesh` | Rigid interaction with moving and uneven triangle meshes |
| `SnippetSoftBodyAVBD` | Component-level AVBD soft-body regression corpus |

Run the HelloWorld scenario without rendering:

```powershell
<path-to-checked-bin>\SnippetHelloWorld_64.exe --headless
```

Record an inspectable trace while reproducing a visual issue:

```powershell
<path-to-checked-bin>\SnippetHelloWorld_64.exe --capture
<path-to-checked-bin>\SnippetHelloWorld_64.exe --capture=D:\traces\hello_world.pxtrace
```

When no path is supplied, the executable writes a timestamped `.pxtrace` in
the process working directory. Local `.pxtrace` recordings are ignored by
Git.

## Verification Snapshot

Last consolidated run: **2026-08-25**, Windows x64 checked build.

| Gate | Result |
|---|---:|
| Cross-Snippet acceptance (`cross14`) | `14/14` |
| AVBD soft-body component executable | `65/65` tests, `366/366` assertions |
| Deformable-volume correctness | `10/10` |
| Deformable-surface non-performance | `87/94` |
| CCD acceptance | `24/24` |
| Selected non-contact D6/drive matrices | `262/262` |
| Gear-joint acceptance | `30/30` |
| Contact modification | `22/23` |
| Contact-plus-drive matrix (`contact72`) | `24/72` |
| Articulation, parallel and sequential | `31/31` in each lane |
| Fixed and spatial tendon suites | `6/6` in each suite |
| Mimic and custom-constraint suites | `12/12` in each suite |

The failed rows are known, reproducible debt rather than ignored results. In
particular, the open surface cases cover mixed-contact dissipation and swept
CCD response; the contact-drive and contact-modification failures cover public
mass scaling, force limits, and horizontal conservation semantics.

Primary reruns:

```powershell
python -B tools/run_avbd_cross_snippets_headless.py --suite cross14 --timeout 180
python -B tools/run_snippet_deformable_volume_avbd_headless.py --mode correctness --execution parallel --timeout 180
python -B tools/run_snippet_joint_drive_headless.py --suite contact72 --timeout 60
python -B tools/run_snippet_contact_modification_headless.py --mode acceptance --timeout 120
```

Surface cases are selected individually with `--case <case>`; the debt ledger
lists the exact failing case names and the required 600-frame rerun command.

The soft-body executable currently covers test IDs 1 through 65, while
`tools/run_snippet_soft_body_avbd_headless.py` still expects 1 through 63. Run
the checked executable directly until that wrapper drift is closed; see debt
item C5 for the exact limitation.

## Solver Model

For a rigid body or soft vertex block, AVBD accumulates inertial, elastic, and
constraint contributions into a local system. In compact form:

```text
H = M / h^2 + sum(rho * J^T * J)
g = M / h^2 * (x - x_hat) + sum(J^T * (rho * C + lambda))
delta_x = -inverse(H) * g
```

The solver alternates local block updates with stabilized
augmented-Lagrangian dual updates. Compiled objectives and shared pair state
assign one explicit owner to contact preparation, primal response, velocity
recovery, and terminal writeback.

## Source Layout

AVBD-specific low-level dynamics code is grouped by responsibility:

```text
physx/source/lowleveldynamics/src/avbd/
  backend/       CPU ISA dispatch and GPU bridge scaffolding
  contact/       contact detection, geometry, preparation, and workspaces
  core/          shared constraint and projection types
  diagnostics/   retained correctness and profiling diagnostics
  ogc/           pair state, admission, response, trust regions, and terminal state
  pipeline/      dynamics integration and task entry points
  scheduling/    island parallel policy
  solver/
    joint/       D6, articulation, drive, tendon, and joint phases
    post_al/     pose, velocity, friction, recovery, and final response phases
    rigid/       rigid-body conversion and phases
    soft/        deformable mechanics, primal/dual solve, topology, and workspace
```

Scene integration is similarly isolated:

```text
physx/source/simulationcontroller/src/avbd/
  contact/       scene-side collision views and proxies
  lifecycle/     actor and attachment lifecycle
  scene/         CPU soft-scene state, synchronization, and statistics
  scheduling/    contact task graphs and policies
  selection/     island and OGC pair planning
```

## Performance Work

Useful PVD profile zones include `AVBD.update`, `AVBD.solveWithJoints`,
`AVBD.blockDescentWithJoints`, and `AVBD.updateLambda`. For deformables,
measure collision preparation, OGC redetection and geometry queries, elastic
and contact solves, temporary/workspace allocation, host-buffer writeback, and
CPU skinning separately. The current CPU deformable results should not be used
as an efficiency comparison with the native GPU FEM path.

## Known Boundaries

- No complete AVBD GPU backend is available.
- The CPU deformable implementation remains correctness-first and is not yet
  optimized for large production workloads.
- Contact mass scaling and force-limit semantics, several rigid spatial
  recovery cases, mixed rigid-soft dissipation, and selected swept surface
  cases remain open.
- Native-island mid-step topology recreation, CPU stress-tensor readback,
  surface anisotropy, serialization/PVD completeness, concurrent-scene stress,
  extreme-scale coverage, and long-duration soak coverage require more work.
- Validation claims in this README apply only to the stated checked Windows
  build and should be refreshed whenever the matrix changes.

## Upstream Documentation

- [PhysX User Guide](https://nvidia-omniverse.github.io/PhysX/physx/index.html)
- [PhysX API Documentation](https://nvidia-omniverse.github.io/PhysX)

## License

Copyright (c) 2008-2026 NVIDIA Corporation.

NVIDIA PhysX is distributed under the BSD-3-Clause license. See
[LICENSE.md](LICENSE.md).
