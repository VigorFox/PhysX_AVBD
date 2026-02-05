# AVBD Articulation Support Analysis

> **Analysis Date**: February 5, 2026  
> **Status**: Articulation support NOT implemented in AVBD solver

## Executive Summary

After careful analysis of the PhysX Articulation API and comparison with TGS solver architecture, we have determined that **AVBD should NOT attempt to replace Featherstone for articulation internal constraints**. Instead, the correct approach is a **hybrid architecture** similar to TGS.

**Decision**: Removed the incorrect `AvbdArticulationAdapter` implementation (~740 lines deleted).

---

## 1. Why the Previous Approach Was Wrong

The deleted `AvbdArticulationAdapter` attempted to:
- Treat each Articulation Link as an independent AVBD rigid body
- Connect links with AVBD SphericalJoint constraints
- Implement PD drives and joint limits inside AVBD

### Problems with This Approach

| Issue | Impact |
|-------|--------|
| **Coordinate Space Mismatch** | Articulation uses reduced coordinates (joint angles), AVBD uses Cartesian coordinates |
| **Poor Convergence** | Chain structures need O(chain_length) iterations to propagate constraints; AVBD assumes independent constraints |
| **DOF Mismatch** | Articulation joints have 1-6 DOF; SphericalJoint is always 3 DOF |
| **Joint Space Control Lost** | PD drives should operate on `jointPosition`/`jointVelocity`, not Cartesian velocity |
| **Algorithm Incompatibility** | Inverse dynamics requires analytical solutions; AVBD is iterative |

---

## 2. Correct Architecture (Following TGS Pattern)

TGS solver handles Articulation correctly using this pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Solver Loop                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for each iteration:                                            │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ 1. Articulation Internal Constraints (Featherstone)     │  │
│    │    articulation->solveInternalConstraints(...)          │  │
│    │    - Joint drives, joint limits, tendons, mimic joints  │  │
│    │    - 100% handled by FeatherstoneArticulation           │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ 2. External Constraints (TGS/AVBD Solver)               │  │
│    │    - Link ↔ RigidBody collisions                        │  │
│    │    - Link ↔ RigidBody joints                            │  │
│    │    - Uses SolverExtBody abstraction for Link access     │  │
│    │    - Calls articulation->getImpulseResponse()           │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ 3. RigidBody Constraints (TGS/AVBD Solver)              │  │
│    │    - Body ↔ Body collisions                             │  │
│    │    - Body ↔ Body joints                                 │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key TGS Components for Articulation

| Component | Purpose |
|-----------|---------|
| `SolverExtBodyStep` | Unified abstraction for RigidBody OR Articulation Link |
| `getImpulseResponse()` | Computes velocity change from impulse; delegates to `articulation->getImpulseResponse()` for links |
| `solveInternalConstraints()` | Featherstone handles all internal articulation constraints |
| Alternating solve | Internal constraints first, then external constraints |

---

## 3. What AVBD Needs to Support Articulation

### 3.1 Required New Components (~400 lines)

```cpp
// Unified body abstraction (similar to TGS SolverExtBodyStep)
class AvbdSolverExtBody {
    union {
        const FeatherstoneArticulation* mArticulation;
        const AvbdSolverBody* mBody;
    };
    PxU32 mLinkIndex;
    
    // Impulse response (delegates to articulation for links)
    PxReal getImpulseResponse(const Cm::SpatialVector& impulse, 
                               Cm::SpatialVector& deltaV);
    
    // Velocity accessors
    PxVec3 getLinVel() const;
    PxVec3 getAngVel() const;
    void applyImpulse(const Cm::SpatialVector& impulse);
};
```

### 3.2 Solver Loop Modification

```cpp
void AvbdSolver::solveWithArticulations(...) {
    for (PxU32 iter = 0; iter < iterations; ++iter) {
        // Step 1: Featherstone solves internal constraints
        for (PxU32 i = 0; i < numArticulations; ++i) {
            articulations[i]->solveInternalConstraints(dt, invDt, ...);
        }
        
        // Step 2: AVBD solves external constraints (Link-Body, Body-Body)
        solveExternalConstraints(bodies, extBodies, constraints);
    }
}
```

### 3.3 Constraint Preparation Changes

- Support `Link ↔ RigidBody` collision constraints
- Support `Link ↔ RigidBody` joint constraints  
- Use `AvbdSolverExtBody` for unified body access

---

## 4. Implementation Effort Estimate

| Task | Lines | Complexity | Time |
|------|-------|------------|------|
| ~~Delete incorrect adapter~~ | ~~-740~~ | ~~Low~~ | ~~Done~~ |
| `AvbdSolverExtBody` class | +100 | Medium | 2-3 days |
| Constraint prep modification | +200 | High | 1 week |
| Solver loop integration | +100 | Medium | 2-3 days |
| Testing & debugging | - | High | 1 week |
| **Total** | **~400 net** | | **2-4 weeks** |

---

## 5. What Will Work After Implementation

| Feature | Status | Notes |
|---------|--------|-------|
| Articulation internal dynamics | ✅ Via Featherstone | Joint drives, limits, tendons, mimic joints |
| Articulation ↔ RigidBody collision | 🔧 Needs implementation | Uses `getImpulseResponse()` |
| Articulation ↔ RigidBody joints | 🔧 Needs implementation | External joint constraints |
| Link self-collision | 🔧 Needs implementation | Uses `getImpulseSelfResponse()` |
| Inverse dynamics API | ✅ Via Featherstone | `computeMassMatrix()`, `computeJointForce()`, etc. |

---

## 6. What Will NOT Work

| Feature | Reason |
|---------|--------|
| AVBD solving articulation internal constraints | Algorithm mismatch (position-based vs reduced coordinates) |
| Custom AVBD joint drives for articulation | Should use Featherstone's native implementation |
| Pure AVBD articulation (no Featherstone) | Fundamentally incompatible architectures |

---

## 7. Conclusion

The correct path forward is:

1. ✅ **Keep Featherstone** for all articulation internal constraints
2. ✅ **Deleted incorrect adapter** that tried to replace Featherstone
3. 🔧 **Implement hybrid architecture** following TGS pattern
4. 🔧 **Add `AvbdSolverExtBody`** for unified Link/Body access
5. 🔧 **Modify constraint prep** to support Link-Body interactions

This approach:
- Leverages Featherstone's O(n) optimal algorithm for articulations
- Uses AVBD's strengths for rigid body constraints
- Follows proven architecture from TGS solver
- Minimizes code changes while maximizing compatibility

---

## 8. Deleted Files

The following files were removed as they implemented an incorrect approach:

- `physx/source/lowleveldynamics/src/DyAvbdArticulationAdapter.cpp` (~450 lines)
- `physx/source/lowleveldynamics/src/DyAvbdArticulationAdapter.h` (~290 lines)

Updated:
- `physx/source/compiler/cmake/LowLevelDynamics.cmake` — Removed file references

---

## References

- `physx/source/lowleveldynamics/src/DyTGSDynamics.cpp` — TGS articulation handling
- `physx/source/lowleveldynamics/src/DyTGSContactPrep.cpp` — `SolverExtBodyStep` implementation
- `physx/source/lowleveldynamics/src/DyFeatherstoneArticulation.cpp` — `solveInternalConstraints()`
- `physx/include/PxArticulationReducedCoordinate.h` — Full PhysX Articulation API
