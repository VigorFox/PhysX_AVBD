# AVBD Soft Body / Cloth System — Standalone Implementation

> **Created**: 2026-03-15  
> **Last Updated**: 2026-03-15  
> **Status**: ✅ **ALL PHASES COMPLETE** — 118/118 tests PASS  
> **Soft Body Tests**: test104–test120 (17 tests)  
> **Architecture**: VBD energy-based elasticity + AVBD adaptive penalty contacts

---

## Executive Summary

软体（Deformable Volume）和布料（Deformable Surface）已完整实现并通过全部测试。
架构遵循 **VBD (Vertex Block Descent)** 论文框架，弹性力使用能量梯度 +
Hessian 直接求解，接触/附着约束使用 AVBD 自适应罚函数。

**核心设计原则**：

1. 每个软体顶点 = 一个 `SoftParticle`（3-DOF，无旋转）
2. **弹性力**由能量函数直接推导 force + Hessian（纯 VBD，无 λ 对偶变量）
3. **接触/附着/销钉约束**使用 AVBD 自适应罚 k（无 λ，仅 k 增长）
4. 与 rigid + articulation 在同一 iteration loop 中求解

**关键架构决策**：弹性力不走 AL (Augmented Lagrangian) 路径。VBD 论文证明
对粒子系统直接能量最小化的 block descent 比 AL 约束更自然、更稳定。
接触类约束保留 AVBD 框架（自适应 k），因为不等式约束需要激活/去激活机制。

### 与初始计划的主要偏差

| 初始计划 | 实际实现 | 原因 |
|---------|---------|------|
| AL 框架（距离/体积/弯曲约束 + λ/ρ） | VBD 能量最小化（StVK/NH/解析弯曲） | Newton VBD (SIGGRAPH 2024) 证明能量方法更优 |
| 简单弹簧距离约束 → E 映射 | StVK 应变能（三角面膜） | 连续介质级别精度，非 game-quality 近似 |
| 标量体积约束 + Jacobian | Neo-Hookean 四面体能量 | 直接使用变形梯度 F，有解析 Hessian |
| 有限差分弯曲 Jacobian | 解析二面角导数（skew matrix） | 无数值噪声，精确 Gauss-Newton Hessian |
| `x -= H⁻¹ rhs`（AL 约束减残差） | `x += H⁻¹ f`（VBD 加位移） | VBD 符号约定：力指向最优解 |
| 全部约束有 λ + ρ 对偶变量 | 弹性无对偶，接触仅 k | 弹性力是保守力，无需 AL 对偶 |

---

## 1. Architecture: VBD + AVBD Hybrid

### 1.1 Framework Split

```
┌──────────────────────────────────────────────────────────────────┐
│                   AVBD Unified Solver Loop                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Rigid Bodies (6-DOF): Contact / D6 / Artic (AVBD AL)     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Soft Particles (3-DOF) — per-vertex VBD:                  │  │
│  │                                                            │  │
│  │    VBD (energy → force + Hessian, NO λ):                   │  │
│  │      • StVK membrane (TriElement)                          │  │
│  │      • Neo-Hookean volume (TetElement)                     │  │
│  │      • Analytic dihedral bending (BendingElement)          │  │
│  │                                                            │  │
│  │    AVBD (adaptive k only, NO λ):                           │  │
│  │      • Ground/rigid contact (SoftContact)                  │  │
│  │      • Attachment to rigid body (AttachmentConstraint)     │  │
│  │      • Kinematic pin (KinematicPin)                        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Soft ↔ Rigid coupling: attachment contributes to rigid    │  │
│  │  body's 6×6 LHS/RHS via addAttachmentContribution_rigid() │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 VBD vs AL: Why Energy-Based?

| Dimension | AL 约束 (初始计划) | VBD 能量 (实际实现) |
|-----------|--------------------|---------------------|
| 弹性模型 | 距离/体积/弯曲标量约束 | StVK / Neo-Hookean / 解析弯曲 |
| 对偶变量 | 每个约束 λ + ρ | **无**（弹性力无需 AL） |
| Primal 更新 | `x -= H⁻¹(ρC + λ)` | `x += H⁻¹ f`（VBD 约定） |
| 物理精度 | 弹簧映射（近似） | 连续介质应变能（精确） |
| 参考文献 | XPBD (Macklin 2016) | VBD (Chen 2024) / Newton (SIGGRAPH 2024) |
| 代码复杂度 | 低（标量约束 Jacobian） | 中（需要 StVK/NH 解析 Hessian） |
| 稳定性 | 需要仔细调 ρ | 能量 Hessian 天然半正定 |

Newton VBD (SIGGRAPH 2024) 证明：对粒子系统，直接能量最小化的 block descent
在收敛速度和稳定性上均优于 AL/XPBD 约束方案。

### 1.3 Energy / Constraint Taxonomy

| 类型 | 算法 | 对偶变量 | Primal 贡献 | Dual 更新 |
|------|------|---------|-------------|-----------|
| StVK 三角面膜 | VBD | 无 | f + H 累加 | 无 |
| Neo-Hookean 四面体 | VBD | 无 | f + H 累加 | 无 |
| 解析弯曲 | VBD | 无 | f + H 累加 | 无 |
| 地面/刚体接触 | AVBD | k（自适应罚） | f + H 累加 | k += β|C| |
| 附着约束 | AVBD | k（自适应罚） | f + H 累加 | k += β|C| |
| 销钉约束 | AVBD | k（自适应罚） | f + H 累加 | k += β|C| |

---

## 2. Data Structures

### 2.1 SoftParticle — 3-DOF Mass Point

```cpp
struct SoftParticle {
    Vec3 position;
    Vec3 velocity;
    Vec3 prevVelocity;
    Vec3 initialPosition;      // position at start of step
    Vec3 predictedPosition;    // inertial prediction (pos + v*dt + g*dt²)
    float mass;                // 0 = pinned/kinematic
    float invMass;
    float damping;
};
```

### 2.2 VBD Element Types — Precomputed Rest-State Data

```cpp
// StVK membrane energy (triangles)
struct TriElement {
    uint32_t p0, p1, p2;           // global particle indices
    float DmInv00, DmInv01;        // 2×2 inverse of reference edge matrix
    float DmInv10, DmInv11;
    float restArea;
};

// Neo-Hookean volume energy (tetrahedra)
struct TetElement {
    uint32_t p0, p1, p2, p3;      // global particle indices
    Mat33 DmInv;                    // 3×3 inverse of reference edge matrix
    float restVolume;
};

// Analytic dihedral bending (Newton convention)
struct BendingElement {
    uint32_t opp0, opp1;           // wing vertices (opposite to shared edge)
    uint32_t edgeStart, edgeEnd;   // shared edge vertices
    float restAngle;
    float restLength;               // rest edge length for stiffness scaling
};

// Diagnostic only
struct EdgeInfo {
    uint32_t p0, p1;
    float restLength;
};
```

### 2.3 AVBD Constraint Types — Adaptive Penalty k Only

```cpp
struct AttachmentConstraint {
    uint32_t particleIdx;       // global soft particle index
    uint32_t rigidBodyIdx;      // index into solver.bodies[]
    Vec3 localOffset;           // attachment point in rigid body local frame
    float k;                    // adaptive penalty (NO lambda)
    float kMax;
};

struct KinematicPin {
    uint32_t particleIdx;
    Vec3 worldTarget;
    float k;
    float kMax;
};

struct SoftContact {
    uint32_t particleIdx;
    uint32_t rigidBodyIdx;      // UINT32_MAX = ground
    Vec3 normal;
    float depth;
    float k;                    // adaptive penalty (NO lambda)
    float ke;                   // material stiffness cap
    float friction;
    Vec3 tangent1, tangent2;
};
```

### 2.4 SoftBody — Mesh + VBD Elements + AVBD Constraints

```cpp
struct SoftBody {
    uint32_t particleStart, particleCount;

    // Topology (local indices)
    std::vector<uint32_t> triangles;    // 3 per tri
    std::vector<uint32_t> tetrahedra;   // 4 per tet

    // Material
    float youngsModulus, poissonsRatio, density, damping;
    float bendingStiffness, thickness;

    // Lamé parameters (computed from E, ν)
    float mu, lambda;     // mu = E/(2(1+ν)), lambda = Eν/((1+ν)(1-2ν))

    // VBD elements (built at setup, no dual variables)
    std::vector<TriElement> triElements;
    std::vector<TetElement> tetElements;
    std::vector<BendingElement> bendElements;
    std::vector<EdgeInfo> edges;

    // AVBD constraints (adaptive k only)
    std::vector<AttachmentConstraint> attachments;
    std::vector<KinematicPin> pins;

    void buildElements(particles);   // builds all element types from topology
};
```

---

## 3. Mathematical Formulation

### 3.1 VBD Primal Update (Per Vertex)

每个顶点 $i$ 求解 3×3 系统：

$$x_i \leftarrow x_i + H_i^{-1} f_i$$

其中：

$$f_i = \frac{m}{h^2}(\tilde{x}_i - x_i) + \sum_e \left(-\frac{\partial E_e}{\partial x_i}\right) + \sum_c f_c$$

$$H_i = \frac{m}{h^2}I + \sum_e \frac{\partial^2 E_e}{\partial x_i^2} + \sum_c H_c$$

$\tilde{x}_i$ = 惯性预测位置, $E_e$ = 弹性能量, $f_c / H_c$ = AVBD 约束贡献。

### 3.2 StVK Membrane Energy (TriElement)

变形梯度 $F \in \mathbb{R}^{3 \times 2}$：

$$F = D_s D_m^{-1}, \quad D_s = [x_1 - x_0 \;|\; x_2 - x_0]$$

Green 应变张量 $G = \frac{1}{2}(F^T F - I)$。

PK1 应力 $P = F(2\mu G + \lambda \text{tr}(G) I)$。

Per-vertex force: $f = -A_0 \cdot P \cdot m_k$（$m_k$ 是 $D_m^{-1}$ 的列或其负和）。

Hessian: 通过 Cauchy-Green 不变量的二阶导数链式法则（解析）。

### 3.3 Neo-Hookean Volume Energy (TetElement)

变形梯度 $F \in \mathbb{R}^{3 \times 3}$：

$$F = D_s D_m^{-1}, \quad D_s = [e_1 \;|\; e_2 \;|\; e_3]$$

Stable Neo-Hookean with $\alpha = 1 + \mu / \lambda$。

Per-vertex selector $m$ (from $D_m^{-1}$ rows)。

Force: $f = -V_0 (\mu F m + \lambda (J - \alpha) \text{cof}(F) \cdot m)$

**简化 Hessian**（cofactor 导数项在 per-vertex 收缩后消失）：

$$H = V_0 (\mu |m|^2 I + \lambda \cdot (\text{cof} \cdot m) \otimes (\text{cof} \cdot m))$$

此简化基于 Newton 参考实现中验证的数学性质：cofactor 导数的反对称块结构在
$G^T H G$ 收缩后恰好归零。

### 3.4 Analytic Dihedral Bending (BendingElement)

顶点顺序 $[x_0, x_1, x_2, x_3]$ = $[\text{opp0}, \text{opp1}, \text{edgeStart}, \text{edgeEnd}]$（Newton 约定）。

弯曲能量 $E = \frac{k \cdot L_{\text{rest}}}{2} (\theta - \theta_0)^2$。

角度导数 $d\theta / dx$ 通过 skew 矩阵 + 归一化向量导数的完整链式求导获得：

1. 面法向 $n_1 = (x_2 - x_0) \times (x_3 - x_0)$，$n_2 = (x_3 - x_1) \times (x_2 - x_1)$
2. 对每个顶点 $x_k$ 计算 $\partial n_1 / \partial x_k$，$\partial n_2 / \partial x_k$（通过叉积规则）
3. 归一化导数 $\partial \hat{n} / \partial x = \frac{1}{|n|}(I - \hat{n}\hat{n}^T) \partial n / \partial x$
4. 角度导数 $d\theta = d\sin\theta \cdot \cos\theta - d\cos\theta \cdot \sin\theta$

Gauss-Newton Hessian: $H \approx k \cdot (d\theta/dx)(d\theta/dx)^T$

### 3.5 AVBD Contact (adaptive k)

法向穿透 penalty:
- Force: $f = k \cdot d \cdot n$（推粒子出去）
- Hessian: $H = k \cdot n \otimes n$
- IPC 正则化 Coulomb 摩擦: 切向分量 + 投影器 $P = I - n \otimes n$

### 3.6 AVBD Attachment / Pin (adaptive k)

二次罚函数 $E = \frac{k}{2}|C|^2$：
- Force = $-kC$
- Hessian = $kI$

### 3.7 Dual Update (AVBD Constraints Only)

弹性力（StVK / NH / Bending）**无对偶更新**。

AVBD 约束仅增长 k：

$$k \leftarrow \min(k + \beta |C|, \; k_{\max})$$

### 3.8 Warmstart

| 约束类型 | Warmstart 策略 |
|---------|---------------|
| 弹性力 | 无（无对偶变量） |
| Attachment / Pin | $k = \text{clamp}(\gamma \cdot k, \; k_{\min}, \; k_{\max})$ |
| Contact | $k = \min(k_{\text{init}}, \; k_e)$（帧间重置） |

---

## 4. Solver Integration

### 4.1 Primal: Soft Particle 3×3 Block Solve

```cpp
// Per-vertex VBD: accumulate force + Hessian, then solve
for (uint32_t spi = 0; spi < nSoftParticles; spi++) {
    float mOverDt2 = sp.mass / dt2;

    // Inertial term
    Vec3 f3 = (sp.predictedPosition - sp.position) * mOverDt2;
    Mat33 H3 = Mat33::diag(mOverDt2);

    // VBD elastic contributions (no lambda)
    for (tri ∈ triElements touching spi)
        evaluateStVKForceHessian(tri, vOrder, mu, lam, particles, f, H);
    for (tet ∈ tetElements touching spi)
        evaluateNeoHookeanForceHessian(tet, vOrder, mu, lam, particles, f, H);
    for (bend ∈ bendElements touching spi)
        evaluateBendingForceHessian(bend, vOrder, stiffness, particles, f, H);

    // AVBD constraint contributions (adaptive k)
    for (attach touching spi) evaluateAttachmentForceHessian_particle(...);
    for (pin touching spi) evaluatePinForceHessian(...);
    for (contact touching spi) evaluateContactForceHessian(...);

    // Solve: displacement = H⁻¹ * f, then x += displacement (VBD sign)
    sp.position += H3.inverse() * f3;
}
```

### 4.2 Rigid Body Side: Attachment Contribution

当 attachment 约束涉及刚体时，`addAttachmentContribution_rigid()` 向刚体的
6×6 LHS/RHS 添加贡献（与 D6 joint 位置约束结构相同）。不需要额外 solver 路径。

### 4.3 Dual Update (Contacts Only)

```cpp
// Only AVBD constraints have dual update (k growth)
for (attach) updateAttachmentDual(ac, particles, bodies, beta);
for (pin) updatePinDual(kp, particles, beta);
for (contact) updateSoftContactDual(sc, particles, beta);
// Elastic forces: NO dual update
```

---

## 5. Implementation Status — ALL PHASES COMPLETE

### Phase 1: Core Soft Body (Tet Volume + Ground Contact) ✅

| Task | Status |
|------|--------|
| `SoftParticle` struct + solver integration | ✅ |
| `SoftBody` struct + `buildElements()` | ✅ |
| Neo-Hookean TetElement (force + Hessian) | ✅ |
| StVK TriElement (force + Hessian) | ✅ |
| Soft-ground contact detection | ✅ |
| 3×3 block solve for particles | ✅ |

**Tests**:

| Test | Name | Validates | Status |
|------|------|-----------|--------|
| test104 | `softBodySingleTet` | Single tet drop: volume preserved, ground contact | ✅ PASS |
| test105 | `softBodyCubeDrop` | 5-tet cube: edges preserved, no penetration | ✅ PASS |
| test106 | `softBodyDistanceOnly` | Triangle mesh edge preservation | ✅ PASS |
| test107 | `softBodyVolumePreserve` | Volume preservation within tolerance | ✅ PASS |
| test108 | `softBodyStackOnGround` | Soft body resting on ground: stable | ✅ PASS |
| test109 | `softBodyMultiple` | Multiple soft bodies: stable, no explosion | ✅ PASS |

### Phase 2: Bending + Cloth Surface ✅

| Task | Status |
|------|--------|
| Analytic dihedral bending (skew matrix derivatives) | ✅ |
| Triangle mesh → bending pair extractor (edge adjacency) | ✅ |
| Cloth grid mesh generator (`generateClothGrid`) | ✅ |

**Tests**:

| Test | Name | Validates | Status |
|------|------|-----------|--------|
| test110 | `clothDrape` | Cloth drapes over obstacle | ✅ PASS |
| test111 | `clothBendingStiffness` | Bending stiffness comparison | ✅ PASS |
| test112 | `clothPinnedCorners` | Pinned edges: center sag, boundary stable | ✅ PASS |

### Phase 3: Soft-Rigid Coupling ✅

| Task | Status |
|------|--------|
| AttachmentConstraint (particle ↔ rigid body) | ✅ |
| KinematicPin (particle → world target) | ✅ |
| Soft-rigid contact detection (particle ↔ box) | ✅ |
| `addAttachmentContribution_rigid()` for rigid 6×6 LHS | ✅ |

**Tests**:

| Test | Name | Validates | Status |
|------|------|-----------|--------|
| test113 | `softRigidAttach` | Soft-rigid attachment: coupled motion | ✅ PASS |
| test114 | `softOnRigidBox` | Soft on rigid box: resting correctly | ✅ PASS |
| test115 | `kinematicPinOscillate` | Pin oscillation: pinned follow, free hang | ✅ PASS |
| test116 | `rigidFallOnSoft` | Rigid on soft: stable cushioning | ✅ PASS |

### Phase 4: Material Parameters + Convergence ✅

| Task | Status |
|------|--------|
| Lamé parameters from (E, ν): mu = E/(2(1+ν)), lambda = Eν/((1+ν)(1-2ν)) | ✅ |
| Material stiffness monotonicity | ✅ |
| Poisson ratio → volume preservation | ✅ |
| Convergence benchmark | ✅ |

**Tests**:

| Test | Name | Validates | Status |
|------|------|-----------|--------|
| test117 | `materialStiffness` | Stiffer material deforms less | ✅ PASS |
| test118 | `materialPoisson` | Higher ν preserves volume better | ✅ PASS |
| test119 | `convergenceSoftBench` | Monotonic improvement with iterations | ✅ PASS |

### Phase 5: Unified Scene ✅

| Test | Name | Validates | Status |
|------|------|-----------|--------|
| test120 | `unifiedScene` | Rigid + articulation + soft body: stable | ✅ PASS |

---

## 6. Test Summary

**Total standalone tests**: 118/118 PASS (101 rigid/artic + 17 soft body)

| Range | Domain | Count | Status |
|-------|--------|-------|--------|
| test1–test73 | Rigid bodies: stacking, collision, joints, friction | 73 | ✅ |
| test74–test103 | Articulation: AL joints, drives, IK, convergence | 28 | ✅ |
| test104–test120 | Soft body / cloth: VBD + AVBD | 17 | ✅ |
| **Total** | | **118** | **118/118 PASS** |

> Note: test48 and test50 are skipped (not counted).
> Tests test102–test103 are articulation (D6 loop closure + scissor lift).

---

## 7. File Organization

```
avbd_standalone/
├── avbd_types.h               # Body, SoftParticle (referenced from softbody.h)
├── avbd_math.h                # Vec3, Mat33, Mat22, Quat, outer(), skew()
├── avbd_softbody.h            # VBD elements, AVBD constraints, evaluators, mesh gen
├── avbd_solver.h              # Solver struct (bodies, softBodies, softParticles, ...)
├── avbd_solver.cpp            # Unified solver loop: rigid primal → soft primal → dual
├── avbd_collision.h           # Rigid-rigid collision detection
├── avbd_articulation.h        # Articulation AL constraints
├── avbd_tests_softbody.cpp    # test104–test120
└── avbd_main.cpp              # All test registration
```

### Key Functions in avbd_softbody.h

| Function | Purpose |
|----------|---------|
| `evaluateStVKForceHessian()` | StVK membrane: per-vertex (f, H) |
| `evaluateNeoHookeanForceHessian()` | Neo-Hookean tet: per-vertex (f, H) |
| `evaluateBendingForceHessian()` | Analytic dihedral: per-vertex (f, H) |
| `evaluateContactForceHessian()` | Ground/rigid contact: (f, H) with IPC friction |
| `evaluatePinForceHessian()` | Kinematic pin: (f, H) |
| `evaluateAttachmentForceHessian_particle()` | Attachment on particle side: (f, H) |
| `addAttachmentContribution_rigid()` | Attachment on rigid body side: 6×6 LHS/RHS |
| `updateAttachmentDual()` | k += β\|C\| |
| `updatePinDual()` | k += β\|C\| |
| `updateSoftContactDual()` | k += β\|C\| |
| `generateCubeTets()` | 8-vertex cube → 5 tetrahedra |
| `generateSubdividedCubeTets()` | N³ subdivided cube → 5N³ tets |
| `generateClothGrid()` | M×N grid → 2(M-1)(N-1) triangles |

---

## 8. Compliance with VBD / AVBD Papers

### 8.1 VBD (Chen et al. 2024) Compliance

| Paper Requirement | Implementation | Status |
|-------------------|----------------|--------|
| Per-vertex 3×3 block solve | `H3.inverse() * f3` | ✅ |
| $x += H^{-1} f$ (add displacement) | `sp.position = sp.position + displacement` | ✅ |
| Energy gradient as force | `evaluateStVK/NH/BendingForceHessian()` | ✅ |
| Energy Hessian as block matrix | Analytic per-vertex Hessian | ✅ |
| No λ for elastic forces | No dual variables in TriElement/TetElement/BendingElement | ✅ |

### 8.2 AVBD / Newton (SIGGRAPH 2024) Compliance

| Paper Requirement | Implementation | Status |
|-------------------|----------------|--------|
| Contacts: adaptive k only (no λ) | `SoftContact.k`, no lambda field | ✅ |
| Dual: k += β\|C\|, capped | `updateSoftContactDual()` | ✅ |
| IPC regularized Coulomb friction | `eps_u` smoothing + tangent projector | ✅ |
| Warmstart: contacts reset k | `sc.k = min(k_init, ke)` | ✅ |
| Warmstart: pins/attachments decay k | `k = clamp(γ·k, kMin, kMax)` | ✅ |

---

## 9. Future Work

| Feature | Priority | Notes |
|---------|----------|-------|
| Soft–soft collision detection | Medium | Spatial hash for particle–triangle proximity |
| Articulation on soft ground | Low | test121: artic standing on deformable surface |
| Performance benchmark | Low | test122: 100 rigid + 10 soft + 2 artic timing |
| Anderson Accel for soft particles | Low | AA framework exists, needs particle state in pack/unpack |
| Chebyshev for soft particles | Low | Position over-relaxation on particle positions |
| Port to PhysX AVBD path | Future | After standalone validation complete |

---

## References

- **Chen et al.**, "Vertex Block Descent" (2024) — VBD algorithm and per-vertex energy minimization
- **Newton** (Disney Research + Google DeepMind + NVIDIA), SIGGRAPH 2024 — Reference implementation
  (`particle_vbd_kernels.py`, `solver_vbd.py`, `rigid_vbd_kernels.py`)
- **Stable Neo-Hookean**: Smith et al., "Stable Neo-Hookean Flesh Simulation" (2018)
- **IPC Friction**: Li et al., "Incremental Potential Contact" (2020)
- AVBD Articulation Gap Analysis: `docs/ARTICULATION_GAP_ANALYSIS.md`
- PhysX `PxDeformableVolume` API: `physx/include/PxDeformableVolume.h`
- PhysX `PxDeformableSurface` API: `physx/include/PxDeformableSurface.h`

---

> **For AI Agents**: Soft body test numbers are test104–test120. Tests test1–test103 are
> occupied by rigid body + articulation tests. When adding new tests beyond this
> document, start from test121 or later.
> The `avbd_softbody.h` file contains ALL soft body code: VBD element types, energy
> evaluators, AVBD constraint types, dual updates, collision detection, and mesh
> generators. Integration into the solver is in `avbd_solver.cpp`.
> **Key convention**: VBD elastic forces use `x += H⁻¹f` (add displacement, not subtract).
> AVBD constraints use adaptive k only — no lambda, no ρ.
