# Unify Contact Constraint Mathematical Definition (AVBD Hybrid Solver)

> **Status**: ✅ Implementation complete. Document updated after implementation revealed `penetrationDepth` is a required part of C(x), not redundant metadata (see §1.6 for root cause analysis).  
> **Purpose**: Provide a clear, self-consistent mathematical definition of the contact constraint used by the AVBD hybrid solver, and a concrete refactoring plan so the *fast path*, *6×6 path*, and *Augmented Lagrangian (AL) multiplier update* operate on the **same** constraint function \(C(x)\).  
> **Scope**: Contacts (normal constraint). Friction is out-of-scope here except for notes on consistency.

---

## 1. Background / Problem Statement

The solver currently has multiple “contact violation” notions in different places:

- `AvbdContactConstraint::computeViolation(...)` defines:
  \[
  C_{\text{geom}}(x) = (p_A + R_A r_A - (p_B + R_B r_B)) \cdot n
  \]
  i.e. **signed normal gap** between the two contact points.

- `AvbdSolver::solveBodyLocalConstraintsFast(...)` computes a different scalar:
  \[
  s(x) = (worldPointA - worldPointB)\cdot n + \texttt{penetrationDepth}
  \]
  and only corrects when \(s(x) < 0\).

- `penetrationDepth` is assigned directly from contact generation input (`convertContact(..., penetration, ...)`), so it is not guaranteed to be a “rest distance” or consistent sign with the solver’s constraint definition.

This creates a **self-inconsistency risk**:

- The *inner solve* (fast path) may be solving a different constraint than the *outer AL update* (multiplier update), and different again from the *6×6 energy/Hessian path*.
- As a result, tuning \(\rho\), \(\lambda\), and Baumgarte can become non-intuitive, and convergence/stability may depend on accidental cancellations.

**Goal**: Choose one canonical mathematical definition for contact constraint \(C(x)\), and enforce it everywhere.

---

## 2. Canonical Contact Constraint Definition

### 2.1 Constraint function

The canonical signed gap including the narrow-phase offset:

\[
C(x) = (x_A - x_B)\cdot n + d_{\text{pen}}
\]
where:
- \(x_A = p_A + R_A r_A\) is body A contact point in world space
- \(x_B = p_B + R_B r_B\) is body B contact point in world space
- \(n\) is the contact normal (from B toward A)
- \(d_{\text{pen}}\) = `penetrationDepth`, the initial signed separation from narrow phase (negative when penetrating)

**Why include \(d_{\text{pen}}\)**: Since `contactPointA` and `contactPointB` are both derived from the same world point at contact creation, the geometric dot product alone starts at ~0. The narrow-phase separation provides the actual initial gap.

### 2.2 Inequality interpretation

Define the non-penetration constraint as:

\[
C(x) \ge 0
\]

Then:
- **Penetration**: \(C(x) < 0\) (violated)
- **Separated**: \(C(x) > 0\) (satisfied)
- **Touching**: \(C(x) \approx 0\)

### 2.3 Activation condition

A contact is *active for correction* iff:

- \(C(x) < 0\) (penetrating)

This should be used consistently in:
- fast path correction
- slow path correction
- 6×6 local system energy gradient/hessian assembly
- multiplier update & convergence statistics

---

## 3. What to Do With `penetrationDepth`

### 3.1 Current state — two assignment sources with conflicting semantics

The `penetrationDepth` field is declared in `AvbdContactConstraint.h` (~L122):
```cpp
physx::PxReal penetrationDepth; //!< Current penetration depth (negative = separated)
```

It has **two distinct assignment sources**:

#### Source 1: `convertContact()` (`AvbdContactConstraint.cpp`, ~L138)
```cpp
outContact.penetrationDepth = penetration;
```
The `penetration` parameter comes from custom contact generation. Its sign convention depends on the caller.

#### Source 2: `createContactConstraintsFromPhysX()` (`AvbdSolver.cpp`, ~L1108)
```cpp
constraint.penetrationDepth = contact->separation;
```
This copies PhysX's `PxContactPatch::separation` field. In PhysX, `separation` is **negative when penetrating** — which matches the `computeViolation()` convention ("negative = penetration").

### 3.2 Known sign convention conflict ⚠️

The field's own doxygen comment says `"negative = separated"`, but:
- `computeViolation()` returns `"negative = penetration"` (documented at ~L153)
- PhysX `separation` uses `"negative = penetration"`
- `isActive()` checks `penetrationDepth < 0.0f` to detect **penetration** (contradicts the field comment)

This means the field comment is **wrong** (or at least inconsistent with actual usage). The code treats `penetrationDepth < 0` as penetrating, not separated.

**Action for Agent**: fix the field comment to match actual semantics:
```cpp
physx::PxReal penetrationDepth; //!< Contact separation (negative = penetrating)
```

### 3.3 Role in C(x) — Essential, Not Redundant

**`penetrationDepth` is an essential part of the canonical C(x)**, not optional metadata.

**Root cause**: At contact creation, both `contactPointA` and `contactPointB` are projections of the **same world contact point** into each body's local frame. Therefore, at creation time:
\[
(x_A - x_B) \cdot n \approx 0
\]
The only source of initial penetration information is `penetrationDepth` (from `contact->separation`). During solver iterations, as bodies move, the geometric dot product tracks the *change from initial configuration*, while `penetrationDepth` provides the *initial offset*. Both terms are needed:
\[
C(x) = \underbrace{(x_A - x_B) \cdot n}_{\text{displacement from creation}} + \underbrace{d_{\text{pen}}}_{\text{initial offset}}
\]

**Lesson learned**: An earlier version of this document (and the first implementation attempt) recommended removing `penetrationDepth` from inline solve paths. This caused all stacked boxes in SnippetHelloWorld to self-compress and sink into the ground, because the solver could not detect initial penetration.

### 3.4 Optional future extension: Rest distance / skin

If you later want a skin/rest separation \(d_{\text{rest}} \ge 0\), define:

\[
C(x) = (x_A - x_B)\cdot n + d_{\text{pen}} - d_{\text{rest}}
\]

Add a separate `restDistance` field rather than overloading `penetrationDepth`.

---

## 4. Completed Code Unification (Implementation Record)

> This section documents the changes that were actually made. The original plan (Version 4.0) contained an error: it recommended removing `penetrationDepth` from inline solve paths. Implementation revealed this was incorrect — see §3.3 for the root cause analysis.

### 4.1 Single helper: `computeViolation()` confirmed as source of truth

`AvbdContactConstraint::computeViolation(posA, rotA, posB, rotB)` now returns the canonical C(x):
```cpp
return (worldPointA - worldPointB).dot(contactNormal) + penetrationDepth;
```

Higher-level wrapper `computeContactViolation(contact, bodyA, bodyB)` delegates to it.

### 4.2 Fast path, slow fallback, shared helper: unified to `dot + penetrationDepth`

All three inline solve paths now use the same formula:

| Location | Variable | Formula |
|----------|----------|---------|
| `solveBodyLocalConstraintsFast()` (~L2425) | `violation` | `dot(worldPosA-worldPosB, n) + contact.penetrationDepth` |
| `solveBodyLocalConstraints()` (~L646) | `violation` | `dot(worldPosA-worldPosB, n) + contacts[c].penetrationDepth` |
| `computeContactCorrection()` (~L958) | `violation` | `dot(worldPosA-worldPosB, n) + contact.penetrationDepth` |

**Changes made**:
- Renamed `separation` → `violation` throughout for clarity
- Ensured `+ penetrationDepth` is present in all three (it was already present before, and correctly so)
- Inner solve correction now targets `C(x) = lambda/rho` (not zero): `correctionMag = max(0, -(violation - lambda/rho) / w * baumgarte)`. This lets the AL multiplier accumulate contact force across outer iterations.
- Jacobi accumulation within each body: all contact corrections computed against same body state, sum applied once. This eliminates asymmetric rotational artifacts from sequential processing.
- Angular correction scaled by `angularContactScale` (default 0.2) to prevent drift from asymmetric contact patches

### 4.3 AL multiplier update: sign corrected and violation unified ✅

`updateLagrangianMultipliers()` calls `computeContactViolation()` → `computeFullViolation()`, which includes `penetrationDepth`.

Two critical changes were made:

1. **Sign correction**: AL update changed from `lambda + rho * C` to `lambda - rho * C`. For the inequality constraint `C(x) >= 0`, when `C < 0` (penetrating), `-rho * C` is positive, so `lambda` increases to build contact force. When `C > 0` (separated), `lambda` decreases toward 0.
2. **Violation unification**: `computeContactViolation()` now delegates to `computeFullViolation()` (not `computeViolation()`), ensuring the AL update sees the same constraint function as the inner solve. Without this fix, the inner solve drives `fullViolation → 0`, which leaves `geometricViolation ≈ -penetrationDepth > 0`, causing lambda to be clamped to 0 every iteration.

### 4.4 6×6 path: upgraded from static `penetrationDepth` to dynamic `computeFullViolation()` ✅

This was the **actual critical bug**. The 6×6 path previously used `contacts[c].penetrationDepth` directly as the violation value — a static value set once during contact creation that never updates as bodies move during solve iterations.

Three functions were updated to call `computeFullViolation()`:

| Function | Old code | New code |
|----------|----------|----------|
| `buildHessianMatrix()` | `violation = contacts[c].penetrationDepth` | `violation = contacts[c].computeFullViolation(bA.position, bA.rotation, bB.position, bB.rotation)` |
| `buildGradientVector()` | `violation = contacts[c].penetrationDepth` | `violation = contacts[c].computeFullViolation(bA.position, bA.rotation, bB.position, bB.rotation)` |
| `computeEnergyGradient()` | `violation = contacts[c].penetrationDepth` | `violation = contacts[c].computeFullViolation(...)` |

Function signatures for `buildHessianMatrix`, `buildGradientVector`, and `solveLocalSystem` were extended with `AvbdSolverBody *bodies, PxU32 numBodies` to provide body state access.

### 4.5 `isActive()` updated to body-state-aware

Changed from:
```cpp
return penetrationDepth < 0.0f || header.lambda > 0.0f;
```
To:
```cpp
PX_FORCE_INLINE bool isActive(const PxVec3& posA, const PxQuat& rotA,
                               const PxVec3& posB, const PxQuat& rotB) const {
    return computeFullViolation(posA, rotA, posB, rotB) < 0.0f || header.lambda > 0.0f;
}
```

**Note**: `isActive()` is currently dead code (no callers found in codebase). Updated for correctness in case future code uses it.

### 4.6 Summary of verification

```bash
# Inline paths include penetrationDepth:
grep -n 'penetrationDepth' DyAvbdSolver.cpp
# → 3 hits in fast/slow/shared violation calculations ✅

# 6×6 path calls computeFullViolation():
grep -n 'computeFullViolation' DyAvbdSolver.cpp
# → 3 hits in buildHessianMatrix, buildGradientVector, computeEnergyGradient ✅

# AL path calls computeContactViolation():
grep -n 'computeContactViolation' DyAvbdSolver.cpp
# → hits in updateLagrangianMultipliers ✅

# No path uses penetrationDepth ALONE as violation (old 6×6 bug):
grep -n 'violation = .*penetrationDepth' DyAvbdSolver.cpp
# → 0 hits ✅
```

---

## 5. Sign Convention Checklist (Must Be Explicit)

To avoid silent sign bugs, document and verify:

1. **Normal direction**: does `contactNormal` point from B→A or A→B?
2. **Constraint**: is the desired inequality \(C(x)\ge 0\) or \(C(x)\le 0\)?
3. **Active condition**: is penetration detected by \(C(x) < 0\) or \(C(x) > 0\)?
4. **AL projection update**:
   - For inequality constraint \(C(x)\ge 0\), the corrected projected AL update is:
     \[
     \lambda \leftarrow \max(0,\ \lambda - \rho\,C(x))
     \]
   - When \(C < 0\) (penetrating), \(-\rho C > 0\), so \(\lambda\) increases (builds contact force).
   - When \(C > 0\) (separated), \(-\rho C < 0\), so \(\lambda\) decreases toward 0.

### 5.1 Known sign convention conflicts (RESOLVED ✅)

All conflicts have been resolved:

| Location | Before | After |
|----------|--------|-------|
| `penetrationDepth` field comment | `"negative = separated"` ❌ | `"negative = penetrating"` ✅ |
| `computeContactViolation()` wrapper | called `computeViolation()` (geometric only) ❌ | calls `computeFullViolation()` (dot + penetrationDepth) ✅ |
| `isActive()` | used `penetrationDepth` directly | uses `computeViolation()` ✅ |

### 5.2 Current sign convention (documented in code)

```cpp
// Sign convention: C(x) = (worldPointA - worldPointB).dot(normal) + penetrationDepth
//   C(x) < 0   =>  penetration (constraint violated)
//   C(x) >= 0  =>  separated or touching (constraint satisfied)
//   contactNormal points from B toward A
//   Inequality enforced: C(x) >= 0
//   AL update: lambda = max(0, lambda - rho * C)
```

---

## 6. Expected Impact / Benefits

After unification:

- **Interpretability** improves: all parts of solver talk about the same \(C(x)\).
- **Parameter tuning** becomes meaningful:
  - changing \(\rho\) affects AL consistently
  - Baumgarte affects correction magnitude consistently
- **Regression testing** becomes easier:
  - penetration RMS should correlate directly with \(C(x)\) statistics
- Reduces risk of “fast path looks stable only because of accidental bias terms”.

---

## 7. Acceptance Tests

### 7.1 Code-level consistency verification

All violation computations must use `dot(xA-xB, n) + penetrationDepth`:

```bash
# Inline paths must include penetrationDepth
grep -n 'penetrationDepth' DyAvbdSolver.cpp
# Expected: 3 hits in fast/slow/shared violation calculations

# 6×6 path and AL update must call computeViolation/computeContactViolation
grep -n 'computeViolation' DyAvbdSolver.cpp
# Expected: hits in buildHessianMatrix, buildGradientVector, computeEnergyGradient
grep -n 'computeContactViolation' DyAvbdSolver.cpp
# Expected: hits in updateLagrangianMultipliers

# No path should use penetrationDepth ALONE as violation
grep -n 'violation = .*penetrationDepth' DyAvbdSolver.cpp
# Expected: zero hits (the old 6×6 bug pattern)
```

### 7.2 Simulation tests

1. **Static plane contact** (SnippetHelloWorld):
   - Stacked boxes should rest stably.
   - No self-compression, no sinking into ground.
   - After settling, \(C(x) \approx 0\) for resting contacts.

2. **Consistency check**:
   - With temporary logging, verify fast path, AL update, and 6×6 path all report the same \(C(x)\) for the same contact at the same iteration.

3. **6×6 vs fast path**:
   - Both paths should reduce penetration error in the same direction.
   - 6×6 path violation should now **change across iterations** (it was constant before the fix).

4. **Regression check** per `STABILITY_REGRESSION_PLAN_CONTACT_C_UNIFICATION_Version4.md`.

---

## 8. Notes on Future Improvements

Status of previously proposed improvements:

- ✅ **DONE**: Incorporating \(\lambda\) into fast-path correction — inner solve now targets `C(x) = lambda/rho`
- ✅ **DONE**: Better aggregation — replaced \(\rho\)-weighted averaging with Jacobi accumulation
- ⬜ **Open**: Adding compliance \(+\alpha/h^2\) to effective mass for tunable softness
- ⬜ **Open**: Lambda warm-starting across frames (see `AVBD_SOLVER_README.md` Known Issue #2)
- ⬜ **Open**: Reduce iteration requirements from 4×8 to paper-recommended 1×4

---