# AVBD Articulation Implementation Gap Analysis

> **Analysis Date**: February 5, 2026  
> **Scope**: Comparison of PhysX Articulation API vs current AVBD AvbdArticulationAdapter implementation

## Executive Summary

The current `AvbdArticulationAdapter` is a **preliminary bridging framework** between Featherstone articulation and AVBD solver. It provides basic initialization and state synchronization but lacks most advanced features required for robotics simulation.

**Overall Articulation Completeness: ~15-20%**

---

## 1. Currently Implemented Features

| Category | Function | Status | Notes |
|----------|----------|--------|-------|
| **Initialization** | `initialize()` | ✅ Implemented | Extracts joint parameters from Featherstone |
| **State Sync** | `syncDriveTargetsToAvbd()` | ✅ Implemented | Syncs position/velocity to AVBD |
| **State Sync** | `syncStateToArticulation()` | ✅ Implemented | Syncs results back to Featherstone |
| **Forward Dynamics** | `solveForwardDynamics()` | ✅ Implemented | Calls AVBD solver for constraints |
| **Joint Drives** | `applyJointDrives()` | ⚠️ Partial | PD control, but only affects linear velocity |
| **Joint Limits** | `applyJointLimits()` | ⚠️ Partial | Position clamping + velocity zeroing |
| **Joint Type Mapping** | `getJointType()` | ✅ Implemented | Maps PxArticulationJointType |
| **Parameter Extraction** | `extractJointParameters()` | ✅ Implemented | Extracts drive/limit parameters |

---

## 2. Missing Core Features

### 2.1 Inverse Dynamics — Completely Unimplemented

| API | Status | Priority |
|-----|--------|----------|
| `computeMassMatrix()` | ❌ Returns false | **Critical** - Required for robot control |
| `computeJointForce()` | ❌ Returns false | **Critical** - Required for torque control |
| `computeJointAcceleration()` | ❌ Not implemented | **Critical** - Required for motion planning |
| `computeGravityCompensation()` | ❌ Not implemented | **High** - Gravity compensation control |
| `computeCoriolisCompensation()` | ❌ Not implemented | **High** - Coriolis compensation |
| `computeGeneralizedExternalForce()` | ❌ Not implemented | **Medium** - External force reaction |
| `computeDenseJacobian()` | ❌ Not implemented | **High** - Jacobian matrix for IK/control |
| `computeCentroidalMomentumMatrix()` | ❌ Not implemented | **Medium** - Centroidal momentum |
| `commonInit()` | ❌ Not implemented | **High** - Inverse dynamics preparation |

### 2.2 Advanced Joint Features — Unimplemented

| Feature | Status | Notes |
|---------|--------|-------|
| **Multi-DOF Joints** | ❌ | Only processes primary axis |
| **Spherical Joint 3-DOF** | ❌ | Three DOF (Twist, Swing1, Swing2) not fully handled |
| **Joint Armature** | ❌ | `setArmature()` inertia augmentation not implemented |
| **Joint Friction** | ❌ | `setFrictionParams()` not implemented |
| **Max Joint Velocity** | ❌ | `setMaxJointVelocity()` limit not implemented |

### 2.3 Tendons — Completely Unimplemented

| API | Status | Priority |
|-----|--------|----------|
| `createSpatialTendon()` | ❌ Not implemented | **Medium** - Soft robotics/biomechanics |
| `createFixedTendon()` | ❌ Not implemented | **Medium** - Tendon-driven mechanisms |
| `getNbSpatialTendons()` | ❌ Not implemented | |
| `getNbFixedTendons()` | ❌ Not implemented | |

### 2.4 Mimic Joints — Completely Unimplemented

| API | Status | Priority |
|-----|--------|----------|
| `createMimicJoint()` | ❌ Not implemented | **High** - Gear linkage/synchronized motion |
| `getMimicJoints()` | ❌ Not implemented | |
| `getNbMimicJoints()` | ❌ Not implemented | |

> **Note**: MimicJoint is similar to standard GearJoint but specifically for joint coupling within an articulation.

### 2.5 Root Link State Management — Unimplemented

| API | Status | Notes |
|-----|--------|-------|
| `setRootGlobalPose()` | ❌ | Set root link world pose |
| `getRootGlobalPose()` | ❌ | Get root link world pose |
| `setRootLinearVelocity()` | ❌ | Set root link CoM linear velocity |
| `getRootLinearVelocity()` | ❌ | Get root link CoM linear velocity |
| `setRootAngularVelocity()` | ❌ | Set root link angular velocity |
| `getRootAngularVelocity()` | ❌ | Get root link angular velocity |
| `updateKinematic()` | ❌ | Update link states |

### 2.6 Cache System — Unimplemented

| API | Status | Notes |
|-----|--------|-------|
| `createCache()` | ❌ | Create state cache |
| `applyCache()` | ❌ | Apply cached state |
| `copyInternalStateToCache()` | ❌ | Export internal state |
| `zeroCache()` | ❌ | Zero cache data |
| `packJointData()` | ❌ | Maximal → reduced coordinates |
| `unpackJointData()` | ❌ | Reduced → maximal coordinates |

### 2.7 Sleep/Wake Management — Unimplemented

| API | Status |
|-----|--------|
| `isSleeping()` | ❌ |
| `wakeUp()` | ❌ |
| `putToSleep()` | ❌ |
| `setSleepThreshold()` | ❌ |
| `setWakeCounter()` | ❌ |
| `setStabilizationThreshold()` | ❌ |

---

## 3. Current Implementation Issues

### 3.1 applyJointDrives() Problem

**Current implementation** (DyAvbdArticulationAdapter.cpp:398-412):
```cpp
PxVec3 force = mLinkData[i].localAxis * driveForce;
bodies[i].linearVelocity += force * bodies[i].invMass * dt;
```

**Issues**:
- Treats torque as linear force
- Does not apply angular momentum for revolute joints
- Does not handle bidirectional parent-child force application

**Should be**:
```cpp
// Revolute joints should apply torque
bodies[i].angularVelocity += mLinkData[i].localAxis * (driveForce * bodies[i].invInertia.x) * dt;
// Also apply reaction to parent body
```

### 3.2 applyJointLimits() Problem

**Current implementation** — simple position clamp:
```cpp
if (mLinkData[i].jointPosition < mLinkData[i].limitLower) {
    mLinkData[i].jointPosition = mLinkData[i].limitLower;
    mLinkData[i].jointVelocity = 0.0f;
}
```

**Issues**:
- Directly modifies joint position instead of applying constraint forces
- Not coordinated with AVBD solver's constraint system
- No elastic bounce/reaction force

### 3.3 Multi-DOF Support

**Current implementation** — only takes first unlocked axis:
```cpp
const PxArticulationAxis::Enum primaryAxis = getFirstUnlockedAxis(*jointCore);
```

**Issues**:
- Spherical joints can have 3 DOF (Twist, Swing1, Swing2)
- Currently only processes the first axis, losing other degrees of freedom

---

## 4. Module Completeness Assessment

| Module | Completeness | Notes |
|--------|--------------|-------|
| **Infrastructure** | ~70% | Initialization and sync framework established |
| **Forward Dynamics** | ~40% | Drive/limit implementation needs rewrite |
| **Inverse Dynamics** | **0%** | Completely unimplemented |
| **Tendons** | **0%** | Completely unimplemented |
| **Mimic Joints** | **0%** | Completely unimplemented |
| **Cache System** | **0%** | Completely unimplemented |
| **Root State** | ~20% | Framework exists but not integrated |
| **Sleep Management** | **0%** | Completely unimplemented |

---

## 5. Implementation Priority Recommendations

### P0 — Must Implement (Robot/Character Physics Foundation)

1. **Fix applyJointDrives()** — Distinguish revolute (torque) vs prismatic (force)
2. **Fix applyJointLimits()** — Use constraint forces instead of position clamping
3. **Multi-DOF Support** — Handle all 3 DOF for Spherical joints

### P1 — High Priority (Controller Requirements)

4. **Inverse Dynamics**: `computeMassMatrix()`, `computeJointForce()`, `computeJointAcceleration()`
5. **Root State Management**: `setRootGlobalPose()`, `get/setRootLinearVelocity()`, etc.
6. **`computeDenseJacobian()`** — Required for IK/motion planning

### P2 — Medium Priority

7. **Cache System** — Batch state update performance optimization
8. **Gravity/Coriolis Compensation** — Advanced control
9. **Joint Armature & Friction** — Physical realism

### P3 — Low Priority

10. **Tendons** — Special application scenarios
11. **Mimic Joints** — Can temporarily use external GearJoint as workaround
12. **Sleep Management** — Performance optimization

---

## 6. Conclusion

The current **AvbdArticulationAdapter** is a **preliminary bridging framework** that has accomplished:
- Featherstone ↔ AVBD data structure mapping
- Basic state synchronization flow
- Simplified PD drive and position limits

**Critical Missing Features**:
- **Inverse Dynamics** (core functionality for robot control) is completely unimplemented
- **Multi-DOF joints** only process primary axis
- **Drive/Limit** implementation is inconsistent with AVBD constraint system

If the goal is to support robotics simulation or physics-based character control, **inverse dynamics and correct drive implementation are the most urgent tasks**.

---

## References

- `physx/include/PxArticulationReducedCoordinate.h` — Full PhysX Articulation API
- `physx/include/PxArticulationJointReducedCoordinate.h` — Joint API
- `physx/source/lowleveldynamics/src/DyAvbdArticulationAdapter.cpp` — Current AVBD implementation
- `physx/source/lowleveldynamics/src/DyAvbdArticulationAdapter.h` — Interface definition
