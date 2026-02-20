# AVBD Solver Algorithm Analysis

> **Analysis Date**: February 7, 2026  
> **Author**: Code review and reverse engineering  
> **Status**: Updated to reflect AL-augmented Jacobi correction implementation

## Executive Summary

The AVBD solver currently uses a **decoupled 3x3 local solve** and reserves
the prior hybrid approach as a long-term research target. The current path
combines:

1. **Augmented Lagrangian (AL) outer loop** with correct sign: `lambda = max(0, lambda - rho * C)`
2. **AL local system solve (default 3x3 decoupled)** per body for position and rotation
3. **Gauss-Seidel between bodies** with Jacobi-style accumulation inside each body
4. **Angular contact scaling** (0.2) to prevent drift from asymmetric contact patches
5. **Velocity damping + pseudo-sleep** for post-settle micro-jitter elimination

The inner solve incorporates the AL multiplier `lambda` as a correction target, enabling the penalty term and Lagrange multiplier to cooperate across outer iterations. The full 6×6 path exists but is disabled by default; the 3x3 path achieves stable stacking at outerIterations=4, innerIterations=8.

**Strategic motivation**: AVBD was introduced for (1) high mass-ratio joint stability ("ultimate hand" — kinematic hand gripping light objects), and (2) future cloth/soft body unification — the position-level block descent structure maps naturally to deformable bodies and GPU compute.

---

## 1. Standard AVBD Algorithm (Paper)

The original AVBD (Augmented Variable Block Descent) algorithm solves a local 6×6 optimization problem for each rigid body:

$$\mathbf{x}_i^{k+1} = \arg\min_{\mathbf{x}_i} \left[ \frac{1}{2h^2}\|\mathbf{x}_i - \tilde{\mathbf{x}}_i\|_{\mathbf{M}_i}^2 + \sum_j \left( \frac{\rho_j}{2} C_j^2 + \lambda_j C_j \right) \right]$$

**Standard implementation requires**:
```cpp
for each outer iteration:
    for each body i:
        // Build 6x6 Hessian matrix
        H = M/h² + Σ(ρ * Jᵢᵀ * Jᵢ)
        
        // Build 6-vector gradient
        g = M/h² * (x - x̃) + Σ(Jᵢᵀ * (ρ*C + λ))
        
        // Solve 6x6 system via LDLT decomposition
        Δx = solve(H, -g)  // ~200-300 FLOPs
        
        body.state += Δx
    
    // Update Lagrangian multipliers
    for each constraint:
        λ += ρ * C
```

**Characteristics**:
- Quadratic convergence
- High per-iteration cost (~300 FLOPs per body)
- Handles constraint coupling within a body

---

## 2. Current Implementation (Default Path)

The current implementation uses a **3x3 decoupled local solve** with AL terms:

```cpp
for each outer iteration:
    for each body i:
        // Build 3x3 linear system and 3x3 angular system
        A_lin = M/h^2 + sum(penalty * Jlin * Jlin^T)
        A_ang = I/h^2 + sum(penalty * Jang * Jang^T)

        // rhs uses AL force: f = penalty * C + lambda (with unilateral clamp)
        rhs_lin = M/h^2 * (x - x_tilde) + sum(Jlin * f)
        rhs_ang = I/h^2 * (theta - theta_tilde) + sum(Jang * f)

        // Solve 3x3 systems and apply update
        delta_pos = solve(A_lin, rhs_lin)
        delta_ang = solve(A_ang, rhs_ang)
        body.pos -= delta_pos
        body.rot -= delta_ang
    
    // AL multiplier update (note sign: λ - ρ * C)
    for each constraint:
        λ = max(0, λ - ρ * C)
```

**Characteristics**:
- AL-augmented convergence (lambda accumulates across outer iterations)
- Low per-iteration cost (~35 FLOPs per constraint)
- Jacobi within body (eliminates intra-body processing bias), GS between bodies
- Angular correction scaled by `angularContactScale` (0.2 default) to prevent drift

---

## 3. Key Differences

| Aspect | Standard AVBD | Current Implementation |
|--------|---------------|------------------------|
| Per-body computation | 6×6 Hessian + LDLT solve | 3×3 linear + 3×3 angular solves |
| Constraint coupling | ✅ Handled via Hessian | ⚠️ Approximated by block-diagonal split |
| Cost per body | ~300 FLOPs | Low (3×3 solves + per-constraint accumulation) |
| Convergence rate | Quadratic | AL-augmented linear |
| Parallelism | Jacobi-style (bodies) | Jacobi within body, GS between bodies |
| Outer loop | Augmented Lagrangian ✅ | Augmented Lagrangian ✅ (sign corrected) |
| Inner solve target | C(x) = 0 via Newton | C(x) = lambda/rho via AL force |
| Angular correction | Full | Scaled by angularContactScale (0.2) |

---

## 4. Evidence in Code

### 4.1 Two Paths Exist

```cpp
// DyAvbdSolver.cpp, line ~525
if (mConfig.enableLocal6x6Solve) {
    // Full 6x6 path (disabled by default)
    solveLocalSystem(bodies[i], bodies, numBodies, contacts, numContacts, dt, invDt2);
} else {
    // Default 3x3 decoupled path
    solveLocalSystem3x3(bodies[i], bodies, numBodies, contacts, numContacts, dt, invDt2);
}
```

### 4.2 Config Default

```cpp
// DyAvbdTypes.h, line ~227
enableLocal6x6Solve(false),  // Disabled by default!
```

### 4.3 Comment Hints

```cpp
bool enableLocal6x6Solve; //!< Use 6x6 local system solve in block descent
                          //!< (fallback to Gauss-Seidel when false)
                          //           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          //           The "fallback" became the default
```

---

## 5. Hybrid Path as Long-Term Research Target

Based on code structure and naming conventions, the likely sequence of events:

The earlier hybrid idea (AL-augmented Jacobi per-constraint correction) is
now treated as a **long-term research target** rather than the default. The
current production path is the 3x3 decoupled local solve, which preserves the
AL formulation while being cheaper than the full 6x6 system.

---

## 6. Is This a Bug or a Feature?

### Arguments for "Bug" 🐛

- Does not match paper algorithm
- No theoretical convergence guarantee
- Constraint coupling is lost
- README previously claimed "6x6 block system solve"

### Arguments for "Feature" ✨

- **5-10x faster** than full 6×6 path
- **Still stable** for high stacks (Augmented Lagrangian helps)
- **Parallelizable** (Jacobi-style body iteration)
- **Practical** for real-time simulation

### Verdict

**Default is 3x3**: The current solver uses a decoupled 3x3 local solve with
AL terms as the default path. The hybrid per-constraint Jacobi approach is
documented as a long-term research direction, not the active default.

---

## 7. Comparison with Other Algorithms

| Algorithm | Iteration Cost | Convergence | Parallelism | Stack Stability |
|-----------|---------------|-------------|-------------|-----------------|
| PBD | ~50/constraint | Linear | ❌ Sequential | Poor |
| XPBD | ~60/constraint | Linear | ❌ Sequential | Fair |
| Standard AVBD | ~300/body | Quadratic | ✅ Jacobi | Excellent |
| **This Implementation** | **~35/constraint** | **AL-augmented linear** | **✅ Jacobi+GS** | **Good (4×8 iter)** |

---

## 8. Recommendations

### Completed ✅

- ✅ AL sign corrected (`lambda - rho * C` instead of `+ rho * C`)
- ✅ Inner solve targets `C(x) = lambda/rho` (lambda incorporated into fast-path)
- ✅ Replaced ρ-weighted averaging with Jacobi accumulation
- ✅ Angular contact scaling (`angularContactScale = 0.2`)
- ✅ Velocity damping + pseudo-sleep for post-settle stability
- ✅ Document the actual algorithm (this document)

### Remaining Options

#### Option A: Implement Lambda Warm-Starting (HIGH PRIORITY)

- Current: `lambda = 0` every frame (see `AVBD_SOLVER_README.md` Known Issue #2)
- Fix: Track contacts across frames and carry over lambda values
- Expected: Reduce required iterations from 4×8 toward 1×4

#### Option B: Add Compliance Term

- Add `α/h²` to effective mass: `correction = -C / (w + α/h²)`
- May improve convergence and allow tunable softness

#### Option C: Fix to Standard AVBD

- Enable `enableLocal6x6Solve` by default for higher accuracy
- Optimize 6×6 path (SIMD, sparse matrix tricks)

---

## 9. Configuration Options

```cpp
AvbdSolverConfig config;

// Use full 6x6 AVBD (slower, more accurate)
config.enableLocal6x6Solve = true;

// Use simplified path (faster, default)
config.enableLocal6x6Solve = false;
```

---

## 10. Conclusion

The current AVBD solver is a **hybrid of Augmented Lagrangian and per-constraint Jacobi projection**, with AL-augmented correction targets and proper sign convention. The inner solve drives `C(x) → lambda/rho` while the outer AL loop updates `lambda = max(0, lambda - rho * C)`, enabling lambda to accumulate as a "contact force memory" across outer iterations.

Lambda warm-starting has been implemented and is functional as of 2026-02-14. The cache-based approach preserves lambda values across frames with decay factors (`wsAlpha=0.95`, `wsGamma=0.99`) to avoid numerical instability. For details, see [`AVBD_SOLVER_README.md`](AVBD_SOLVER_README.md).

The implementation is stable for stacked rigid body scenarios at `outerIterations=4, innerIterations=8`. Achieving stability at lower iteration counts (paper-recommended 1×4) remains an open improvement, with lambda warm-starting implementation complete but decay factors potentially requiring tuning.

The full 6×6 path exists but is disabled by default. The fast path is recommended for real-time simulation.

---

## References

- AVBD Paper: "Augmented Variable Block Descent for Rigid Body Simulation"
- XPBD Paper: Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
- Source: `physx/source/lowleveldynamics/src/DyAvbdSolver.cpp`
