# PhysX AVBD

Select the CPU backend with `sceneDesc.solverType = PxSolverType::eAVBD`. PhysX owns broad phase, narrow phase, island membership, sleeping and scene publication. The backend compiles Host inputs into a single constraint row and block program. Ordinary joints use `solverPrep`, contacts use the PhysX narrow-phase output, and native deformable materials can supply cooked batches.

## Shared step

The simulation uses step-start contacts, inertial targets, colored in-place primal updates, shared dual updates and one pose/velocity publication. Rigid bodies and articulation links have six degrees of freedom per block. Particle vertices have three. Each rigid/deformable contact has one shared reaction.

For rigid velocity contact, the finalized contact point defines both endpoint arms. The normal impulse is unilateral and respects its authored maximum. The two tangential impulses share the Coulomb disk of the same normal load. Independent components and disjoint writable endpoints can run in parallel. Kinematic velocity still enters the relative velocity at contact.

<a id="soft-optimization-no-fork"></a>

## Deformable operators

Cloth and volume may use specialized assembly, cooked topology, compact material data and kernels in the shared step. Both endpoints of a mixed constraint consume the same physical state and reaction. The solver publishes each body once. A separate integration loop, global post-correction or silent fallback violates this contract.

<a id="parallel-performance-first"></a>

## Parallel acceptance

The native Particle3 path uses a shared budget of four primal iterations. Evaluate physical quality, whole-step cost and actual worker scaling together. Validate action and reaction, penetration, strain, release and bounded energy independently on each run. Different schedules may produce different floating-point trajectories. CPU and future GPU implementations must preserve parallel work and bounded local kernels.

## Headless checks

With the Checked SDK and snippets built, run from the repository root.

```text
python tools/run_snippet_hello_world_headless.py --cases gated
python tools/run_snippet_joint_headless.py --cases gated
python tools/run_snippet_articulation_rc_headless.py --cases gated
python tools/run_avbd_row_program_probe.py
```

The three snippet runners also accept `--solver pgs` and `--solver tgs`. Compare performance on cases where both solvers meet the physical contract. Measure active contact frames separately from sleeping frames. The default headless cloth and five-body volume cases are `static-box` and `visual-showcase`.
