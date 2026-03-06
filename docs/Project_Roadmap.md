# PhysX_AVBD Project Roadmap

- **Repository**: `VigorFox/PhysX_AVBD`
- **Author**: Vulpes (@VigorFox)
- **Version**: 0.3 (Prismatic Hessian + Standalone Alignment)
- **Last Updated**: March 2, 2026

**Overview**: This roadmap serves as a formal guide for AI Agents (e.g., Claude, Gemini, ChatGPT) to assist in the iterative development of PhysX_AVBD. The project forks NVIDIA's PhysX SDK and integrates an experimental position-level Augmented Vertex Block Descent (AVBD) solver, with 99.9% AI-generated code. 
**Status Legend**: `Integrated` = merged into main code path; `Accepted` = integrated and validated by acceptance checks; `Pending` = not complete or acceptance gate not closed.
**Current status**: Prismatic/Gear are `Integrated` in PhysX Hessian path with stabilized dual updates retained. Revolute is `Integrated` in solver code path but remains `Pending` for acceptance (SnippetJoint visual validation + full checklist). Prismatic limit sign semantics and solve-path policy are `Accepted` between `physx` and `avbd_standalone` (Prismatic-touching bodies force 6x6 solve). Standalone default regression baseline is `Accepted` at 48 aligned cases.

The roadmap is phased, with checklists for each stage. Prioritize CPU paths, unified rigid/soft-body solving, and alignment with PhysX architecture. Use AI prompts for code generation, testing, and optimization. Track progress via GitHub issues/PRs and update X thread (ID: `2021997979444687179`) after each milestone for visibility.

## Guiding Principles
- **AI-Driven Development**: Leverage AI Agents for 99%+ code tasks. Prompts should include: `"Implement [feature] in C++ for PhysX_AVBD, ensuring compatibility with 6x6 Hessian and multi-threaded islands."`
- **Testing**: Every change must include unit tests (e.g., via PhysX snippets) and stress tests (e.g., extended SnippetChainmail demo with high-density joints/meshes).
- **Metrics**: Measure stability (convergence iterations), performance (FPS/CPU usage), and compatibility (fallback to Gauss-Seidel minimized).
- **Milestones**: Tag releases (e.g., v0.3) and share demos/GIFs on X/GitHub.
- **Timeline**: 3-6 months total, assuming weekly 10-20 hours post-AI quota recovery.
- **Risks**: Monitor numerical stability; revert to warm-starting if jitter occurs.

---

## Phase 1: Core Solver Unification (In Progress - Gear Integrated)
**Goal**: Fully unify all joint types under AVBD Hessian, eliminating fallbacks. Extend to basic soft-body constraints for CPU unified rigid/soft solving.

- [x] Gear Joint unification into 6x6 Hessian (stabilized dual updates, static anchors).
- [x] Prismatic integration into AVBD Hessian path (position/rotation/limit rows + dual updates).
- [ ] Revolute acceptance gate: `Integrated` in Hessian path, but `Pending` SnippetJoint visual validation + full acceptance checks.
- [ ] Distance joint into Hessian (full limits and drives).
- [x] Standalone alignment with PhysX for Prismatic semantics (signed limit violation + dual clamp policy + force-6x6-on-touch).
- [x] Regression baseline cleanup (default standalone suite: 48 aligned cases).
- [ ] CPU Soft-Body Support:
  - Implement tetrahedral/triangular constraints (distance, volume, bending).
  - Create `PxAVBDSoftBody` actor (CPU host memory path).
  - Enable rigid-soft bidirectional coupling (attachments, collisions).
  - Minimal demo: 1000-tetrahedron sphere drop + ground collision.
- [x] Contact lambda warm-starting (cache + aging + decay) integrated.
- [ ] Adaptive substepping for faster convergence.
  - **AI Prompt Example**: `"Generate C++ code to integrate Prismatic Joint into AVBD Hessian, with O(1) lookups and multi-thread compatibility."`
  - **Milestone**: Distance joint + soft-body baseline integrated via pure AVBD. Update demo GIF.

---

## Phase 2: Advanced Multibody Systems (2-3 Weeks)
**Goal**: Support complex articulated bodies, aligning with PhysX's reduced-coordinate features.

- [ ] Featherstone Articulation Compatibility:
  - Implement reduced-coordinate multibody dynamics.
  - Support joint limits, drives, and contacts.
  - Test scenarios: Ragdoll, vehicle suspension, robotic arm.
- [ ] Coexistence with Native GPU FEM Soft-Bodies (Flag-based switching).
 - [ ] PhysX↔Standalone behavior parity harness:
  - Build scene-level parity checks (residual trends, convergence iterations, drift envelopes).
  - Include Prismatic limit crossing and mixed-joint chains as mandatory parity scenes.
  - **AI Prompt Example**: `"Extend AVBD to Featherstone Articulations in PhysX, ensuring Hessian consistency and island threading."`
  - **Milestone**: Mixed articulated rigid/soft scenes stable. New demo: Articulated chainmail with soft attachments.

---

## Phase 3: Collision System Upgrade (2 Weeks)
**Goal**: Replace conservative CCD with modern, penetration-free handling.

- [ ] Integrate OGC (Offset Geometric Contact) Algorithm:
  - Embed within AVBD framework as CCD replacement.
  - Support high-speed objects, soft-body contacts, and zero penetration.
  - Pre-reserve GPU-friendly paths.
  - **AI Prompt Example**: `"Implement OGC in C++ for PhysX_AVBD, integrating with position-level projections and contacts."`
  - **Milestone**: High-velocity impacts without tunneling. Update SnippetChainmail to include fast-moving projectiles.

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