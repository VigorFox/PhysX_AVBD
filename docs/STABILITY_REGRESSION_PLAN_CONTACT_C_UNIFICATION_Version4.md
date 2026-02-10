# Stability Regression Plan (After Unifying Contact Constraint C(x))

> **Goal**: Verify that after unifying the mathematical contact constraint \(C(x)\) and implementing AL-augmented Jacobi correction, the solver becomes **more stable, more predictable, and easier to tune**.  
> **Changes under test**: (1) All paths use `computeFullViolation()` = dot + penetrationDepth. (2) AL sign corrected: `lambda - rho * C`. (3) Inner solve targets `C(x) = lambda/rho`. (4) 3x3 decoupled local solve (default). (5) `angularContactScale = 0.2`. (6) Velocity damping + pseudo-sleep.  
> **Current status**: ✅ Stable at outerIterations=4, innerIterations=8. ⬜ Paper-default 1×4 still insufficient (known issue).

---

## 1. What “stability improvement” means (measurable)

After the change, compared to baseline:

1. **Fewer oscillations** across inner/outer iterations:
   - RMS penetration error should decrease more smoothly.
   - Active contact count should not “flip-flop” excessively.

2. **More predictable parameter effects**:
   - Increasing `outerIterations` or increasing \(\rho\) should not make things worse unexpectedly.
   - Changing `baumgarte` should scale correction magnitude consistently.

3. **Reduced reliance on clamps**:
   - Fewer frames hit `maxPositionCorrection`, `maxAngularCorrection`, or the hardcoded rotation clamp.
   - Less frequent “LDLT fallback” (if 6×6 path is tested).

4. **No catastrophic cases**:
   - No NaNs, no exploding positions/rotations, no persistent deep penetrations.

---

## 2. Required instrumentation (minimal and high-signal)

### 2.1 Per-frame summary (always enabled in debug/profiling builds)

Log once per simulation step:

- `frameIndex`
- `dt`
- Solver config snapshot:
  - `outerIterations`, `innerIterations`
  - `initialRho`, `rhoScale`, `maxRho`
  - `baumgarte`, `angularContactScale`
  - `velocityDamping`, `angularDamping`
  - `enableLocal6x6Solve` (3x3 default vs 6x6)
- Global solver stats:
  - `mStats.constraintError` (RMS of `computeFullViolation()` over active contacts)
  - `mStats.activeConstraints`
  - Mean/max lambda values across contacts
  - (optional) time cost per stage if already available

### 2.2 Per-iteration (recommended for the regression runs)

Log once per **outer** iteration and once per **inner** iteration:

- `outerIter`, `innerIter`
- `activeConstraints`
- `rmsC` (RMS of negative violations only, i.e. penetrating contacts)
- `minC` (most negative penetration)
- `maxC` (largest positive separation among contacts considered active/near-contact)
- `meanAbsDeltaPos` / `maxAbsDeltaPos` applied to bodies this iteration
- `meanAbsDeltaTheta` / `maxAbsDeltaTheta`
- Clamp counters:
  - `numPosClamps` (position correction clamped)
  - `numAngClamps` (angular correction clamped)
  - `numRotClamp01rad` (the hard clamp `min(angle, 0.1f)` triggered)
- If 6×6 path enabled:
  - `numLDLTFails` (LDLT decomposition failures)
  - `numGradientFallbacks`

**Acceptance criterion**: After unification, the reported `rmsC/minC` must correspond to the **same scalar used by fast-path correction** and AL multiplier update.

---

## 3. Minimal test suite (scenarios)

Each scenario should be run for a fixed number of frames (e.g. 600 frames = 10 seconds at 60 Hz), and repeated with a small matrix of parameters (section 4).

### Scenario A: Single box on plane (baseline sanity)
- 1 dynamic box falling onto static plane.
- Expect:
  - Penetration resolves quickly.
  - No jitter when resting.
- Metrics to watch:
  - resting `rmsC` near 0 (small negative tolerated, but stable)
  - stable activeConstraints count (should settle)

### Scenario B: Tall stack (stress convergence)
- e.g. 20–50 boxes stacked vertically on plane.
- Expect:
  - No “melting” stack.
  - Penetration bounded and does not grow with time.
- Metrics:
  - `minC` should not drift more negative over time
  - clamp counters should not explode
  - strong sensitivity to dt/iterations should reduce after unification

### Scenario C: Pyramid stack (multi-contact coupling)
- Classic pyramid of boxes (more lateral contacts).
- Expect:
  - Less jitter / less lateral drift.
- Metrics:
  - oscillation in `activeConstraints` and `rmsC` should reduce

### Scenario D: Edge contact / corner case
- A box resting on the edge of another box (or on a thin ledge).
- Expect:
  - No excessive spin.
  - Angular clamp not triggered constantly.
- Metrics:
  - `numRotClamp01rad` should reduce or remain bounded

### Scenario E: Fast-moving impact (robustness)
- A box fired at a wall/stack at high speed.
- Expect:
  - No NaNs.
  - Penetration corrected over subsequent frames.
- Metrics:
  - transient spike OK, but should decay; not persist or diverge

### Scenario F (optional): Two dynamic bodies contact
- Two dynamic bodies colliding (not world-static).
- Expect:
  - Symmetric behavior (no consistent bias pushing one body more than the other).
- Metrics:
  - compare corrections applied to A vs B qualitatively (or via summary stats)

---

## 4. Parameter matrix (what to vary)

For each scenario, run a small grid. Keep it minimal to avoid combinatorial explosion.

### Core grid
- `dt`: { 1/30, 1/60, 1/120 }
- `outerIterations`: { 1, 2, 4 }
- `innerIterations`: { 2, 4, 8 }
- `baumgarte`: { 0.1, 0.3, 0.6 }
- `angularContactScale`: { 0.1, 0.2, 0.5, 1.0 }

### Penalty schedule
- `initialRho`: { 1e4, 1e5, 1e6 }
- `rhoScale`: { 1.0, 2.0 }
- `maxRho`: { 1e6, 1e8 }

### Damping parameters
- `velocityDamping`: { 0.95, 0.99, 1.0 }
- `angularDamping`: { 0.9, 0.95, 1.0 }

### Solver path toggles
- `enableLocal6x6Solve`: { false, true } (run fewer combinations for true due to cost)

**Recommended execution**:
- 3x3 default path: run full grid for Scenarios A–E
- 6x6 path: run reduced grid (dt 1/60, outer 2, inner 4, baumgarte 0.3) + one stress case (Scenario B)

---

## 5. Pass/Fail criteria (pragmatic)

### Must-pass (hard failures)
- Any NaN/Inf in body transforms
- `rmsC` diverges monotonically for > N frames in a resting scenario
- `activeConstraints` becomes zero while visibly intersecting geometry
- Explosive clamp counts (e.g. `numRotClamp01rad` triggers on most bodies for most iterations after settling)

### Should-improve (soft expectations)
After unification, compared to baseline:

- Lower variance in `rmsC` during rest (Scenario A/B)
- Fewer abrupt jumps in `minC`
- More monotonic decay of penetration after impact (Scenario E)
- Reduced sensitivity to dt changes (dt halved should not dramatically change resting penetration)

---

## 6. Diagnostic plots (high value, optional)

For each run, generate time series:

- `rmsC` vs frame
- `minC` vs frame
- `activeConstraints` vs frame
- `maxAbsDeltaPos` and `maxAbsDeltaTheta` vs frame
- clamp counters vs frame

**Interpretation**:
- Oscillations: sawtooth or alternating patterns per iteration
- Instability: increasing envelope or frequent clamp saturation
- Improvement: smoother decay and tighter bounds

---

## 7. Implementation Notes

### Completed ✅

1. All paths use `computeFullViolation()` = `dot(xA-xB, n) + penetrationDepth` as canonical `C(x)`.
2. Fast-path activation: `if (violation >= lambdaOverRho && lambda <= 0) continue` (AL-augmented).
3. Fast-path correction: `max(0, -(C - lambda/rho) / w * baumgarte)` (targets `lambda/rho`, not 0).
4. AL update: `lambda = max(0, lambda - rho * C)` (sign corrected).
5. Sign convention documented in `DyAvbdConstraint.h` comment block.
6. Jacobi accumulation for contacts within each body.

### Testing Notes

- When comparing before/after, keep:
  - identical initial conditions
  - identical random seeds (if any)
  - deterministic mode enabled for reproducibility if available
- Current stable config: outerIterations=4, innerIterations=8 (DyAvbdDynamics.cpp)
- Paper-default config: outerIterations=1, innerIterations=4 (AvbdSolverConfig constructor) — known unstable
- See `AVBD_SOLVER_README.md` for known issues

---

## 8. Deliverables

- A log file per test run (CSV/JSON)
- A short summary table per scenario:
  - final `rmsC`, worst `minC`, average clamp triggers, any failures
- (Optional) plots for representative runs

---