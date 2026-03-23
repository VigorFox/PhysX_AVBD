# PhysX_AVBD Project Roadmap

- **Repository**: `VigorFox/PhysX_AVBD`
- **Author**: Vulpes (@VigorFox)
- **Version**: 0.6 (Soft Body / Cloth)
- **Last Updated**: March 15, 2026

**Overview**: This roadmap serves as a formal guide for AI Agents (e.g., Claude, Gemini, ChatGPT) to assist in the iterative development of PhysX_AVBD. The project forks NVIDIA's PhysX SDK and integrates an experimental position-level Augmented Vertex Block Descent (AVBD) solver, with 99.9% AI-generated code. 
**Status Legend**: `Integrated` = merged into main code path; `Accepted` = integrated and validated by acceptance checks; `Pending` = not complete or acceptance gate not closed.
**Current status**: **Soft body standalone is `Accepted`** — VBD energy-based elasticity (StVK + Neo-Hookean + analytic bending) + AVBD adaptive penalty contacts. Standalone: 118/118 tests (101 rigid/artic + 17 soft body). PhysX port in progress: enabling GPU-only deformable snippets on CPU AVBD path.
**Prior milestones**: Articulation solver `Accepted` (29/29 PhysX tests). D6 unification ("万物皆D6"). Standalone regression: 99→118 tests.

The roadmap is phased, with checklists for each stage. Prioritize CPU paths, unified rigid/soft-body solving, and alignment with PhysX architecture. Use AI prompts for code generation, testing, and optimization. Track progress via GitHub issues/PRs and update X thread (ID: `2021997979444687179`) after each milestone for visibility.

## Guiding Principles
- **AI-Driven Development**: Leverage AI Agents for 99%+ code tasks. Prompts should include: `"Implement [feature] in C++ for PhysX_AVBD, ensuring compatibility with 6x6 Hessian and multi-threaded islands."`
- **Testing**: Every change must include unit tests (e.g., via PhysX snippets) and stress tests (e.g., extended SnippetChainmail demo with high-density joints/meshes).
- **Metrics**: Measure stability (convergence iterations), performance (FPS/CPU usage), and compatibility (fallback to Gauss-Seidel minimized).
- **Milestones**: Tag releases (e.g., v0.3) and share demos/GIFs on X/GitHub.
- **Timeline**: 3-6 months total, assuming weekly 10-20 hours post-AI quota recovery.
- **Risks**: Monitor numerical stability; revert to warm-starting if jitter occurs.

---

## Phase 1: Core Solver Unification (✅ Joint System Complete)
**Goal**: Fully unify all joint types under AVBD Hessian, eliminating fallbacks. Extend to basic soft-body constraints for CPU unified rigid/soft solving.

- [x] Gear Joint unification into 6x6 Hessian (stabilized dual updates, static anchors).
- [x] Prismatic integration into AVBD Hessian path (position/rotation/limit rows + dual updates).
- [x] **D6 Unification ("万物皆D6")**: All joint types (Spherical, Fixed, Revolute, Prismatic) unified into single D6 constraint path with motion masks.
- [x] Revolute acceptance gate: Cross-product axis alignment, post-solve motor, SnippetJoint validation complete.
- [ ] Distance joint into Hessian (full limits and drives).
- [x] Standalone alignment with PhysX: identical algorithm for all D6 joints, 53/53 tests pass.
- [x] Regression baseline: 53 aligned cases covering all joint types, limits, drives, gear, and stability.
- [x] CPU Soft-Body Support (✅ Standalone `Accepted`, PhysX port `Pending`):
  - [x] **Standalone VBD+AVBD soft body**: StVK membrane (triangles), Neo-Hookean volume (tets), analytic dihedral bending — all as VBD energy evaluators with per-vertex 3×3 block solve. AVBD adaptive k for contacts/attachments/pins (no λ). 17 tests (test104–test120), 118/118 total.
  - [ ] **PhysX port**: Integrate VBD soft body into `AvbdDynamicsContext` CPU path.
    - Add `AvbdSoftBody` / `AvbdSoftParticle` to AVBD solver (mirror standalone `avbd_softbody.h`).
    - Create CPU-side `PxDeformableVolume` / `PxDeformableSurface` creation path (no `CudaContextManager` required).
    - Enable existing GPU-only snippets (DeformableVolume, DeformableSurface, DeformableVolumeAttachment, DeformableVolumeKinematic) to run on CPU when `PxSolverType::eAVBD`.
    - Mesh cooking: reuse `PxDeformableVolumeExt::createDeformableVolumeMesh` (already CPU).
  - [x] Rigid-soft bidirectional coupling: `addAttachmentContribution_rigid()` (particle→rigid 6×6 LHS).
  - **Architecture note**: Standalone uses VBD (energy gradient + Hessian) for elasticity, NOT AL constraints. See `docs/AVBD_SOFTBODY_ROADMAP.md` for full details.
- [x] Contact lambda warm-starting (cache + aging + decay) integrated.
- [ ] Adaptive substepping for faster convergence.
  - **AI Prompt Example**: `"Generate C++ code to integrate Prismatic Joint into AVBD Hessian, with O(1) lookups and multi-thread compatibility."`
  - **Milestone**: Distance joint + soft-body baseline integrated via pure AVBD. Update demo GIF.

---

## Phase 2: Advanced Multibody Systems (✅ Articulation Solver Complete)
**Goal**: Support complex articulated bodies.

- [x] **Pure AVBD Articulation** (no Featherstone dependency):
  - Articulation internal joints (eFIX, eREVOLUTE, eSPHERICAL, ePRISMATIC) as penalty constraints in unified ADMM loop.
  - Joint limits, position/velocity drives, friction — all as AL constraint rows.
  - Closed kinematic loops (D6 cross-links) handled natively without special treatment.
  - Per-island adaptive iteration override via `setSolverIterationCounts()`.
  - 12 bugs fixed during integration (motion encoding, drive errors, penalty balance, byte order, etc.).
  - **Result**: 29/29 PhysX tests pass (15 consecutive deterministic runs). Exceeds Featherstone hybrid ceiling (28/1).
- [x] Scissor lift with loaded boxes stable through full 10s simulation.
- [x] Standalone: 99/99 tests including convergence acceleration (AA 47%, Chebyshev 29%).
- [ ] Coexistence with Native GPU FEM Soft-Bodies (Flag-based switching: GPU path uses XPBD FEM, CPU AVBD path uses VBD energy).
- [ ] PhysX↔Standalone behavior parity harness.
  - **Milestone**: ✅ Articulation solver accepted. Scissor lift exceeds TGS+Featherstone stability.

---

## Phase 3: Collision System Upgrade — OGC (2 Weeks)
**Goal**: Replace conservative CCD with OGC (Offset Geometric Contact), a penetration-free collision algorithm proposed by NVIDIA & University of Utah at SIGGRAPH 2025.

### OGC Algorithm Overview

**Background**: Prior art IPC (Incremental Potential Contact) guarantees penetration-free simulation but relies on global line search for collision time computation, forcing frequent global synchronization that cripples GPU parallel efficiency — especially for complex self-collision (e.g., multi-layer cloth stacking).

**Core Innovations**:

1. **Volumetric Surface Offsetting** — Instead of treating each primitive (vertex/edge/face) as an independent collider, OGC offsets the entire mesh surface along its normals, constructing a continuous volumetric shell with finite thickness around thin shells and line segments.

2. **Orthogonal Contact Forces** — Because contacts are computed on the smoothly offset surface, the algorithm mathematically guarantees that contact forces are always strictly orthogonal to the contact surface. This eliminates the non-orthogonal parasitic stretch forces that plague traditional penalty-based solvers when collision nodes slide across polygon surfaces, removing unnatural "rubber-band" artifacts at their root.

3. **Local Displacement Bounds (Safety Bubble)** — To guarantee absolute penetration-free behavior without global line search, OGC dynamically computes a local conservative displacement bound ("safety bubble") per vertex each iteration. As long as every vertex stays within its safe radius, the entire system is guaranteed penetration-free.

4. **Massively Parallel Local Operations** — By reducing penetration prevention to local-bound constraints, collision handling no longer requires global synchronization. The large coupled global solve is decomposed into purely local computations, perfectly suited for modern GPU architectures — thousands of collision constraints solved independently and simultaneously.

**Performance**: 300× speedup over IPC while maintaining penetration-free guarantees. Enables real-time simulation of 50-layer cloth stacking and complex yarn knotting.

**Known Limitation**: Under certain parameters, conservative displacement bounds may cause simulated objects to exhibit slightly excessive "bounciness."

### Integration Plan

- [ ] Integrate OGC within AVBD framework:
  - Replace current penalty + projection hybrid with OGC offset-surface contact.
  - Implement per-vertex safety bubble for penetration-free guarantee.
  - Ensure orthogonal contact forces for soft-soft collision (eliminates normal-flip problem).
  - Pre-reserve GPU-friendly paths.
  - **AI Prompt Example**: `"Implement OGC in C++ for PhysX_AVBD, integrating volumetric surface offsetting, orthogonal contact forces, and local displacement bounds."`
  - **Milestone**: High-velocity impacts without tunneling. Soft-soft collision without penetration. Update SnippetDeformableVolumeAVBD to demonstrate zero overlap.
  - **Reference**: See [OGC_ALGORITHM_FORMULAS.md](OGC_ALGORITHM_FORMULAS.md) for complete formula extraction from the paper.
  - **Paper PDF**: [Offset_Geometric_Contact-SIGGRAPH2025.pdf](Offset_Geometric_Contact-SIGGRAPH2025.pdf)

---

## Phase 4: Performance Engineering (2-3 Weeks)
**Goal**: Optimize for efficiency in game server environments.

- [ ] Full SOA (Structure of Arrays) Refactoring:
  - Restructure constraints, bodies, and contacts for cache efficiency.
  - Enhance multi-threaded islands.
- [ ] Initial GPU Path Stub (Optional: Basic CUDA integration for AVBD core).
  - **AI Prompt Example**: `"Refactor PhysX_AVBD data structures to SOA in C++, optimizing for CPU cache and threading."`
  - **Milestone**: 2-4x performance gain in benchmarks. Profiled results in README.

---

## Phase 5: Production Readiness (2-4 Weeks)
**Goal**: Make AVBD viable for real-world use, focusing on server-side reliability.

- [ ] Sleep/Wake Implementation and Tuning:
  - Energy threshold detection and island sleep management.
  - Predictive wake-up for server synchronization.
- [ ] Unity/Unreal Integration Demo:
  - Compile as DLL + simple plugin/package.
  - Front/back-end consistency test (client prediction + server authority).
- [ ] Comprehensive Documentation:
  - API examples, performance benchmarks (tables in README), and AVBD-paper-to-code mapping table.
  - Contribution guidelines for community forks.
  - **AI Prompt Example**: `"Add Sleep/Wake logic to PhysX_AVBD in C++, with tunable thresholds and server-friendly wakes."`
  - **Milestone**: Full feature parity with core PhysX (research-grade). Release v1.0 with Unity demo video.

---

## Post-Roadmap Considerations
- **Community Engagement**: After Phase 3, submit to NVIDIA forums or SIGGRAPH Asia for feedback.
- **Extensions**: If time allows, add GPU full acceleration or ML surrogates for AVBD.
- **Monitoring**: Track GitHub stars (target: 30+), X views (target: 5K+), and potential NVIDIA collabs.
- **Review Cycle**: Revisit roadmap after each phase; adjust based on AI feedback or issues.

> This roadmap positions `PhysX_AVBD` as a trailblazing AI-driven fork, emphasizing CPU soft-body unification and server physics. 
> **For AI Agents:** Use this as your primary guide for code suggestions and validations.