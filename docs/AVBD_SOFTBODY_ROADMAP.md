# AVBD Soft Body / Cloth System — Standalone Roadmap

> **Created**: 2026-03-15  
> **Status**: Phase 1 — In Progress  
> **Standalone Tests**: Starting from test102  
> **Architecture**: Pure AVBD AL (Augmented Lagrangian) — unified with rigid + joint + articulation

---

## Executive Summary

将软体（Deformable Volume）和布料（Deformable Surface）约束作为标准 AL 约束行集成进
现有 AVBD block descent 求解器。不引入单独的 FEM 求解器或 PBD 子系统——所有约束
（distance、volume、bending、contact、attachment）在同一个能量函数中统一求解。

**核心设计原则**：

1. 每个软体顶点 = 一个质点 body（3-DOF，无旋转）
2. 约束类型全部复用 AL 框架：`LHS += ρ J^T J`，`RHS += J^T (ρC + λ)`
3. 软体-刚体耦合 = 附着约束（bilateral AL row），零额外架构
4. 与 rigid + articulation 在同一 iteration loop 中交替求解

---

## 1. Architecture: Soft Body in AVBD Framework

### 1.1 Particle-Based Representation

PhysX GPU 软体使用 FEM 四面体 + Neo-Hookean / Co-Rotational 应变能。
AVBD standalone 采用更轻量的 **particle + constraint** 方案：

```
┌─────────────────────────────────────────────────────────────────┐
│                    AVBD Unified Solver                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Rigid Bodies (6-DOF)  ←─── existing ───→  Contact / D6 / Artic│
│                                                                 │
│  Soft Particles (3-DOF) ←── new ──→  Distance / Volume / Bend  │
│                                                                 │
│  Attachment constraints  ←── new ──→  Soft↔Rigid coupling       │
│                                                                 │
│  All constraints in ONE energy function, ONE iteration loop     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Not FEM?

| Dimension | FEM (PhysX GPU) | Particle + Constraint (AVBD) |
|-----------|-----------------|------------------------------|
| 求解器 | 专用隐式 FEM solver | 现有 AL block descent（零新代码路径） |
| 网格 | 四面体体素化 + collision mesh | 三角形表面 / 四面体（约束而非刚度矩阵） |
| 材料模型 | Neo-Hookean / Co-Rotational | Young's modulus → distance stiffness 映射 |
| 耦合 | 单独 attachment solver | 同一 AL 循环内的 bilateral row |
| 代码量 | ~10k LOC | ~500-800 LOC |
| 精度 | 连续介质级别 | Game-quality（足够支撑实时应用） |

AVBD 的价值命题是 **统一性**，不是 FEM 精度。证明所有约束类型在一个 solver 循环中
共存才是目标。

### 1.3 Constraint Taxonomy

| 约束类型 | Jacobian 维度 | 作用 | AL 对偶变量 |
|----------|--------------|------|-------------|
| **Distance** | 1×3 per particle pair | 边长保持 (stretch resistance) | `λ_dist` per edge |
| **Volume** | 1×3 per tet vertex (×4) | 四面体体积保持 | `λ_vol` per tet |
| **Bending** | 1×3 per dihedral vert (×4) | 二面角抗弯 (wrinkle formation) | `λ_bend` per dihedral |
| **Contact** | 1×3 per particle | 粒子-地面 / 粒子-刚体碰撞 | `λ_contact` per pair |
| **Attachment** | 1×3 (bilateral) | 粒子绑定到刚体表面点 | `λ_attach` per pair |
| **Kinematic Pin** | 1×3 (bilateral) | 粒子固定到世界位置 | `λ_pin` per vertex |

---

## 2. Data Structures

### 2.1 SoftParticle — 3-DOF Mass Point

```cpp
struct SoftParticle {
    Vec3 position;           // current position
    Vec3 velocity;           // current velocity
    Vec3 predictedPosition;  // inertial prediction (pos + v*dt + g*dt²)
    float invMass;           // 0 = kinematic/pinned

    // Material (mapped from Young's modulus)
    float damping;           // velocity damping coefficient

    // Index into solver body array
    // SoftParticles occupy a separate range after rigid bodies
};
```

### 2.2 SoftBody — Mesh + Constraints

```cpp
struct SoftBody {
    // Mesh topology
    std::vector<uint32_t> particleIndices;   // into solver's softParticles[]
    std::vector<uint32_t> triangles;         // 3 indices per tri (surface mesh)
    std::vector<uint32_t> tetrahedra;        // 4 indices per tet (volume mesh, optional)

    // Material
    float youngsModulus;     // E: maps to distance constraint stiffness
    float poissonsRatio;     // ν: maps to volume constraint stiffness
    float density;           // kg/m³: determines particle mass
    float damping;           // elasticity damping

    // Bending (surface only)
    float bendingStiffness;  // resistance to dihedral angle change
    float bendingDamping;

    // Derived constraints (built at setup)
    struct DistanceConstraint {
        uint32_t p0, p1;     // particle indices (local to this soft body)
        float restLength;    // ||p0 - p1|| at rest
        float lambda;        // AL dual variable
        float rho;           // penalty parameter
    };
    std::vector<DistanceConstraint> distConstraints;

    struct VolumeConstraint {
        uint32_t p0, p1, p2, p3;  // tet vertex indices
        float restVolume;         // signed volume at rest (1/6 * det)
        float lambda;
        float rho;
    };
    std::vector<VolumeConstraint> volConstraints;

    struct BendingConstraint {
        uint32_t p0, p1, p2, p3;  // shared edge (p0,p1), wing vertices (p2,p3)
        float restAngle;          // dihedral angle at rest
        float lambda;
        float rho;
    };
    std::vector<BendingConstraint> bendConstraints;

    // Soft–Rigid attachment
    struct AttachmentConstraint {
        uint32_t particleIdx;     // soft particle (local)
        uint32_t rigidBodyIdx;    // rigid body in solver.bodies[]
        Vec3 localOffset;         // attachment point in rigid body local frame
        Vec3 lambda;              // bilateral AL dual (3-DOF)
        float rho;
    };
    std::vector<AttachmentConstraint> attachments;

    // Kinematic pins
    struct KinematicPin {
        uint32_t particleIdx;
        Vec3 worldTarget;         // fixed world position
        Vec3 lambda;
        float rho;
    };
    std::vector<KinematicPin> pins;
};
```

### 2.3 Collision: Soft Particle ↔ Ground / Rigid Body

```cpp
struct SoftContact {
    uint32_t particleIdx;       // soft particle (global index)
    uint32_t rigidBodyIdx;      // UINT32_MAX = ground
    Vec3 normal;                // contact normal (world)
    float depth;                // penetration depth (negative = separated)

    float lambda;               // normal AL dual
    float rho;                  // penalty
    float friction;             // Coulomb friction coefficient

    // Friction
    float lambdaT1, lambdaT2;  // tangent AL duals
    Vec3 tangent1, tangent2;    // friction basis
};
```

---

## 3. Mathematical Formulation

### 3.1 Distance Constraint

保持边长不变：

$$C_{\text{dist}} = \|\mathbf{p}_1 - \mathbf{p}_0\| - L_0$$

Jacobians:

$$J_0 = -\hat{\mathbf{n}} \in \mathbb{R}^{1 \times 3}, \quad J_1 = +\hat{\mathbf{n}} \in \mathbb{R}^{1 \times 3}$$

其中 $\hat{\mathbf{n}} = \frac{\mathbf{p}_1 - \mathbf{p}_0}{\|\mathbf{p}_1 - \mathbf{p}_0\|}$。

Stiffness mapping: $\rho_{\text{dist}} = k_E \cdot E \cdot A / L_0$，其中 $A$ 是横截面面积估计，$k_E$ 是无量纲缩放系数。

### 3.2 Volume Constraint (四面体)

保持四面体符号体积不变：

$$C_{\text{vol}} = V - V_0, \quad V = \frac{1}{6} (\mathbf{p}_1 - \mathbf{p}_0) \cdot \left[(\mathbf{p}_2 - \mathbf{p}_0) \times (\mathbf{p}_3 - \mathbf{p}_0)\right]$$

Jacobians（对 $\mathbf{p}_i$ 的偏导）：

$$J_1 = \frac{1}{6}(\mathbf{p}_2 - \mathbf{p}_0) \times (\mathbf{p}_3 - \mathbf{p}_0)$$
$$J_2 = \frac{1}{6}(\mathbf{p}_3 - \mathbf{p}_0) \times (\mathbf{p}_1 - \mathbf{p}_0)$$
$$J_3 = \frac{1}{6}(\mathbf{p}_1 - \mathbf{p}_0) \times (\mathbf{p}_2 - \mathbf{p}_0)$$
$$J_0 = -(J_1 + J_2 + J_3)$$

Stiffness mapping: $\rho_{\text{vol}} = k_\nu \cdot E / (3(1-2\nu))$ (体积模量)。当 $\nu \to 0.5$ 时 $\rho_{\text{vol}} \to \infty$，即不可压缩。

### 3.3 Bending Constraint (二面角)

共享边 $(p_0, p_1)$，翼顶点 $(p_2, p_3)$：

$$C_{\text{bend}} = \theta - \theta_0$$

其中 $\theta = \arctan2(\|\mathbf{n}_1 \times \mathbf{n}_2\|, \, \mathbf{n}_1 \cdot \mathbf{n}_2)$，
$\mathbf{n}_1 = (\mathbf{p}_1 - \mathbf{p}_0) \times (\mathbf{p}_2 - \mathbf{p}_0)$，
$\mathbf{n}_2 = (\mathbf{p}_1 - \mathbf{p}_0) \times (\mathbf{p}_3 - \mathbf{p}_0)$。

Jacobians 通过离散外微分或有限差分计算。

Stiffness: $\rho_{\text{bend}} = k_b \cdot E \cdot t^3 / (12(1-\nu^2))$ (Kirchhoff 板弯曲刚度)，
其中 $t$ 是厚度。

### 3.4 Soft Contact

粒子对地面/刚体的法向穿透：

$$C_{\text{contact}} = \mathbf{n} \cdot (\mathbf{p}_i - \mathbf{p}_{\text{surface}})$$

与现有 rigid-body contact 结构相同，但 particle 侧只有 3×3 block（无角动量）。

### 3.5 Attachment Constraint (Soft-Rigid Coupling)

将 soft particle 绑定到 rigid body 表面上的一点：

$$C_{\text{attach}} = \mathbf{p}_{\text{soft}} - (\mathbf{x}_{\text{rigid}} + R_{\text{rigid}} \cdot \mathbf{r}_{\text{local}})$$

Jacobians:
- 对 soft particle: $J_{\text{soft}} = I_{3 \times 3}$
- 对 rigid body: $J_{\text{rigid,lin}} = -I_{3 \times 3}$, $J_{\text{rigid,ang}} = [\mathbf{r}_{\text{world}}]_\times$

这与 D6 joint 的位置约束结构完全相同。

---

## 4. Solver Integration

### 4.1 Block Descent 中的 3-DOF Particle

现有 solver 对 rigid body 使用 6×6 LHS + 6×1 delta。
Soft particle 只有 3-DOF（纯平移），使用 3×3 LHS + 3×1 delta：

```cpp
// In primal update loop:
if (isSoftParticle(bi)) {
    Mat33 LHS = Mat33::diagonal(invMass / dt²);  // mass term
    Vec3  RHS = LHS * (pos - predictedPos);       // inertia

    // Distance constraints touching this particle
    for (auto& dc : distConstraintsOf(bi)) { ... addDistContribution(LHS3, RHS3) }
    // Volume constraints touching this particle
    for (auto& vc : volConstraintsOf(bi)) { ... addVolContribution(LHS3, RHS3) }
    // Bending constraints touching this particle
    for (auto& bc : bendConstraintsOf(bi)) { ... addBendContribution(LHS3, RHS3) }
    // Ground/rigid contacts touching this particle
    for (auto& sc : softContactsOf(bi)) { ... addContactContribution(LHS3, RHS3) }
    // Attachment to rigid body
    for (auto& ac : attachmentsOf(bi)) { ... addAttachContribution(LHS3, RHS3) }
    // Kinematic pin
    for (auto& kp : pinsOf(bi)) { ... addPinContribution(LHS3, RHS3) }

    Vec3 delta = solveLDLT_3x3(LHS, RHS);  // or just inverse since 3×3
    pos_bi -= delta;
}
```

### 4.2 Dual Update

遵循现有 AL 框架：

```cpp
// After primal sweep:
for (auto& dc : distConstraints) {
    float C = computeDistC(dc);
    dc.lambda = dc.rho * C + dc.lambda;  // no clamping (bilateral)
    // Adaptive penalty growth
    if (fabsf(C) > tol) dc.rho = min(dc.rho + beta * fabsf(C), RHO_MAX);
}

for (auto& vc : volConstraints) {
    float C = computeVolC(vc);
    vc.lambda = vc.rho * C + vc.lambda;
    if (fabsf(C) > tol) vc.rho = min(vc.rho + beta * fabsf(C), RHO_MAX);
}

for (auto& bc : bendConstraints) {
    float C = computeBendC(bc);
    bc.lambda = bc.rho * C + bc.lambda;
    if (fabsf(C) > tol) bc.rho = min(bc.rho + beta * fabsf(C), RHO_MAX);
}

for (auto& sc : softContacts) {
    float C = computeContactC(sc);
    sc.lambda = max(0.0f, sc.rho * C + sc.lambda);  // unilateral
    // friction dual update similar to rigid contacts
}
```

### 4.3 Rigid Body 侧的 Attachment / Contact 贡献

当 attachment 或 soft contact 涉及一个 rigid body 时，约束对 rigid body 的 6×6 LHS
也有贡献（与 D6Joint 的位置部分相同）。这意味着 soft-rigid coupling 不需要任何新的
solver 路径——只是在 rigid body 的 primal update 中多了几个约束行。

---

## 5. Implementation Phases

### Phase 1: Core Soft Body Constraints (Distance + Volume + Ground Contact)

**目标**：单个软体在重力下掉落到地面，保持形状。

| Task | Est. LOC | Priority |
|------|----------|----------|
| `SoftParticle` struct + solver integration | +80 | P0 |
| `SoftBody` struct + mesh builder | +100 | P0 |
| Distance constraint (primal + dual) | +80 | P0 |
| Volume constraint (primal + dual) | +100 | P0 |
| Soft-ground contact detection + constraint | +60 | P0 |
| 3×3 block solve for particles | +30 | P0 |
| **Subtotal** | **~450** | |

**测试 (test102–test107)**：

| Test | Name | Scenario | Pass Criteria |
|------|------|----------|---------------|
| 102 | `softBodySingleTet` | 1 个四面体自由落体到地面 | 体积保持 > 95%, 不穿透 |
| 103 | `softBodyCubeDropPhysX` | 四面体化立方体(~24 tets)落到地面 | 形变后回弹, 边长偏差 < 5% |
| 104 | `softBodyDistanceOnly` | 弹簧网格(无 volume): 三角网格下落 | 边长恢复, 不坍缩 |
| 105 | `softBodyVolumePreserve` | 压缩立方体, 验证体积守恒 | `|V-V0|/V0 < 2%` |
| 106 | `softBodyStackOnRigid` | 软体立方体放在刚体平面上静止 | 不穿透, COM 稳定 |
| 107 | `softBodyMultiple` | 2个软体相互碰撞 | 不互穿, 各自体积保持 |

### Phase 2: Bending + Cloth Surface

**目标**：三角形表面网格支持弯曲约束，实现布料效果。

| Task | Est. LOC | Priority |
|------|----------|----------|
| Bending constraint (dihedral angle) | +100 | P1 |
| Triangle mesh → bending pair extractor | +60 | P1 |
| Cloth-specific SoftBody factory (grid mesh) | +50 | P1 |
| **Subtotal** | **~210** | |

**测试 (test108–test111)**：

| Test | Name | Scenario | Pass Criteria |
|------|------|----------|---------------|
| 108 | `clothDrapePhysX` | 布料(20×20)下落到球形障碍物 | 包裹球体, 不穿透 |
| 109 | `clothBendingStiffness` | 布料悬挂一端, 对比有/无 bending | 有 bending: 平板状; 无: 柔软下垂 |
| 110 | `clothSelfCollision` | 布料折叠, 验证面-面不穿透 | 折叠后厚度 > 0 |
| 111 | `clothPinnedCorners` | 四角固定的布料, 中心受重力下垂 | 对称下垂, 稳定振荡衰减 |

### Phase 3: Soft-Rigid Coupling (Attachment + Kinematic)

**目标**：软体与刚体之间的约束耦合。

| Task | Est. LOC | Priority |
|------|----------|----------|
| Attachment constraint (soft↔rigid) | +60 | P1 |
| Kinematic pin constraint | +30 | P1 |
| Contact: soft particle ↔ rigid box | +80 | P1 |
| **Subtotal** | **~170** | |

**测试 (test112–test116)**：

| Test | Name | Scenario | PhysX 参照 |
|------|------|----------|------------|
| 112 | `softRigidAttachPhysX` | 软体绑定到下落刚体, 一起运动 | SnippetDeformableVolumeAttachment |
| 113 | `softOnRigidBox` | 软体立方体落到刚体箱子上方 | SnippetDeformableVolume |
| 114 | `kinematicPinOscillate` | 部分顶点被 kinematic 振荡驱动 | SnippetDeformableVolumeKinematic |
| 115 | `clothOnRotatingSphere` | 布料落到旋转球体上, 包裹 | SnippetDeformableSurface |
| 116 | `rigidFallOnSoft` | 刚体盒子从高处落到软体上, 软体缓冲变形 | — |

### Phase 4: Material Mapping + Convergence

**目标**：将 PhysX 材料参数 (E, ν) 映射到 AVBD penalty stiffness，验证收敛性。

| Task | Est. LOC | Priority |
|------|----------|----------|
| Material → penalty mapping函数 | +40 | P2 |
| Adaptive penalty for soft constraints | +30 | P2 |
| Convergence benchmark (iteration vs error) | +60 | P2 |
| AA + Chebyshev for soft particles | +20 (已有框架) | P2 |
| **Subtotal** | **~150** | |

**测试 (test117–test119)**：

| Test | Name | Scenario | Pass Criteria |
|------|------|----------|---------------|
| 117 | `materialStiffness` | 同形状不同 E 值, 验证形变量成反比 | E_high 形变 < E_low 形变 |
| 118 | `materialPoisson` | 同 E 不同 ν, ν→0.5 不可压缩 | ν=0.48: `|ΔV/V0| < 0.5%` |
| 119 | `convergenceSoftBench` | 20-tet cube, 测量迭代收敛曲线 | 10 iter 内 dist violation < 1e-3 |

### Phase 5: Mixed Scene Benchmark

**目标**：rigid + articulation + soft body 完全统一场景。

| Test | Name | Scenario | 验证 |
|------|------|----------|------|
| 120 | `unifiedScene` | 刚体堆叠 + 铰接体 + 软体 + 布料 | 全部约束同时求解, 无爆炸 |
| 121 | `articulationOnSoftGround` | 铰接体站在软体地面上 | 铰接关节 + 软体接触正确耦合 |
| 122 | `benchmark1000` | 100 rigid + 10 soft(24 tets each) + 2 artic | 性能基准(帧时间记录) |

---

## 6. PhysX Snippet → AVBD Test 映射

| PhysX Snippet | AVBD Test | 核心差异 |
|---------------|-----------|---------|
| SnippetDeformableVolume | test103, test106, test107 | PhysX: GPU FEM; AVBD: CPU AL particle |
| SnippetDeformableVolumeAttachment | test112, test116 | PhysX: barycentric grid; AVBD: vertex-level attachment |
| SnippetDeformableVolumeKinematic | test114 | PhysX: per-vertex invMass=0; AVBD: KinematicPin constraint |
| SnippetDeformableSurface | test108, test111, test115 | PhysX: triangle FEM; AVBD: distance + bending constraints |
| SnippetPBDCloth (deprecated) | test109, test110 | PhysX: PBD springs; AVBD: AL distance + bending |

---

## 7. Stiffness Mapping: PhysX Material → AVBD Penalty

PhysX 使用连续介质力学参数 $(E, \nu)$，AVBD 需要映射到 per-constraint penalty $\rho$：

### 7.1 Distance Constraint Penalty

$$\rho_{\text{dist}} = \frac{E \cdot A_{\text{eff}}}{L_0}$$

其中 $A_{\text{eff}}$ 是约束对应的等效横截面积（从 Voronoi 区域估算）。

### 7.2 Volume Constraint Penalty

$$\rho_{\text{vol}} = \frac{E}{3(1 - 2\nu)} \cdot \frac{1}{V_0^{1/3}}$$

$\nu \to 0.5$ 时趋近不可压缩（$\rho_{\text{vol}} \to \infty$）。实际中 clamp 到 `RHO_MAX`。

### 7.3 Bending Constraint Penalty

$$\rho_{\text{bend}} = \frac{E \cdot t^3}{12(1 - \nu^2)} \cdot \frac{1}{A_{\text{tri}}}$$

其中 $t$ 是厚度，$A_{\text{tri}}$ 是共享三角形面积。

---

## 8. Mesh Generation Utilities (Standalone)

Standalone 不使用 PhysX 的 cooking / voxelization。需要内置简单的网格生成：

| 形状 | 生成方式 | 用途 |
|------|---------|------|
| Cube (tet mesh) | 8 顶点 → 5 或 6 四面体 | 基础 volume 测试 |
| Subdivided cube | 细分为 N³ voxels → 5N³ tets | 精度/收敛测试 |
| Quad grid (cloth) | M×N grid → 2(M-1)(N-1) triangles | 布料测试 |
| Sphere (surface) | UV 参数化三角化 | 碰撞障碍物 |

---

## 9. File Organization

```
avbd_standalone/
├── avbd_types.h               # + SoftParticle, SoftBody structs
├── avbd_softbody.h            # NEW: soft body constraint math + builders
├── avbd_solver.h              # + std::vector<SoftBody>
├── avbd_solver.cpp            # + soft primal/dual integration
├── avbd_collision.h           # + soft-ground, soft-rigid contact detection
├── avbd_tests_softbody.cpp    # NEW: test102–test122
└── avbd_main.cpp              # + soft body test registration
```

---

## 10. Risk Assessment

| 风险 | 可能性 | 影响 | 缓解 |
|------|--------|------|------|
| Distance constraint 对长链收敛慢 | 高 | 中 | 与 articulation 经验一致; AA + Chebyshev 已验证 |
| Volume constraint Jacobian 退化(扁平 tet) | 中 | 高 | 正则化: `rho = max(rho, eps)` + degenerate tet 检测 |
| Bending constraint 数值碰撞(dihedral ≈ π) | 中 | 中 | 小角度近似 + sinc 安全除法 |
| 大顶点数(>1000)下 primal sweep 慢 | 高 | 低 | 这是性能问题，留给 Phase 4 + PhysX 移植阶段 |
| Soft-soft collision detection | 中 | 中 | Phase 1 不实现; Phase 3 仅 particle-rigid; 后续可加 spatial hash |

---

## 11. Success Criteria

| Phase | Gate | Metric |
|-------|------|--------|
| Phase 1 | 软体基础 | test102–107 全通, volume deviation < 2% |
| Phase 2 | 布料 | test108–111 全通, bending 视觉正确 |
| Phase 3 | 耦合 | test112–116 全通, attachment 稳定 |
| Phase 4 | 材料 | test117–119 全通, material mapping 单调 |
| Phase 5 | 统一 | test120–122 全通, rigid+artic+soft 同场景无爆炸 |

**最终 standalone 目标**: `122/122 tests PASS`（99 existing + 21 new soft body + 2 benchmark）

---

## References

- PhysX `PxDeformableVolume` API: `physx/include/PxDeformableVolume.h`
- PhysX `PxDeformableSurface` API: `physx/include/PxDeformableSurface.h`
- SnippetDeformableVolume: `physx/snippets/snippetdeformablevolume/`
- SnippetDeformableSurface: `physx/snippets/snippetdeformablesurface/`
- SnippetDeformableVolumeAttachment: `physx/snippets/snippetdeformablevolumeattachment/`
- SnippetDeformableVolumeKinematic: `physx/snippets/snippetdeformablevolumekinematic/`
- Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics" (2016)
- Müller et al., "Position Based Dynamics" (2007)
- AVBD Articulation Gap Analysis: `docs/ARTICULATION_GAP_ANALYSIS.md`

---

> **For AI Agents**: Soft body test numbers start at test102. Tests test1–test101 are
> occupied by rigid body + articulation tests. When adding new tests beyond this
> roadmap, start from test123 or later.  
> The `avbd_softbody.h` file contains all constraint math (distance, volume, bending).  
> Integration into the solver is in `avbd_solver.cpp` inside the existing primal/dual
> block descent loop — soft particles use 3×3 blocks, not 6×6.
