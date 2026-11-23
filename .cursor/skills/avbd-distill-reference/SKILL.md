---
name: avbd-distill-reference
description: >-
  Distills AVBD equations from PhysX_AVBD_2, avbd-demo3d, and webphysics
  into the PhysX_AVBD_3 backend without copying trees or welding references
  into one if. Use when a solver layer is lying, a gate is red, or when
  mining the green path for inertia, AL, GS commit, warmstart, or material NCP.
---

# 对照参考：禁止照搬，必须提炼

`PhysX_AVBD_2`、`reference/avbd-demo3d`、`reference/webphysics` 回答的是**不同层**。禁止 `cp -r avbd/`，禁止复制 `DyAvbdSolver.cpp` / Scene `*Avbd*`，禁止把三份参考焊进同一条 `if`。

当前工作是**求解器后端统一**。步进编排（谁在何时做 DCD / 惯性 / primal / dual / 写回）以 WebPhysics 的一条管线为准；AVBD_2 只贡献场景合同、尺度与绿测数字。禁止把 AVBD_2 里 artic / 软体 / 关节各自独立的 pass、公共平移 sidecar、几何专用 task graph 搬进本树当「绿测路径」。

整步诊断用 skill `avbd-whole-step`。当前层合同：`.cursor/skills/avbd-whole-step/layers.md`。加能力不得另开管线：skill `avbd-unified-row`。

从 `_AVBD` 只迁物理合同、epoch 与 fail-closed；不迁 owner 枚举、Scene glue、standalone 双轨、几何 task graph。必要时用紧凑 payload，仍挂在 `AvbdRow` 上。

## 默认动作

卡住时去绿测真正执行的方程里挖**那一层**，压成本树最小形式，落地后用 smoke / HelloWorld 门禁验证。不要闭门调参，不要再加门禁特例。

1. 先问：这段参考规定的是 A/B/C/D/E 哪一层？
2. 定位绿测真正执行的方程 / 策略 / 常数（不是过期注释或死代码）。
3. 压成与本树不变量兼容：`AvbdRow`、原地 GS / 着色提交、无 Scene `*Avbd*`、无 owner 枚举、无 source gate。
4. 写进 `physx/source/lowleveldynamics/src/avbd/`，跑门禁。

## 哪份参考管哪一层

| 参考 | 层 | 不要当成 |
|---|---|---|
| `reference/avbd-demo3d`（`solver.cpp` / `manifold.cpp`） | B/C 位置 AL：原地 GS、`C=(1-α)C0+JΔx`、`λ←Π(λ+ρC)`、独立 `rhoT` | 材料反弹、PhysX 尺度、写回 |
| `reference/webphysics`（`integration.ts` / `avbdState.ts`） | **统一步进管线**：`DCD(x^t)` → 全员惯性目标 / adaptive guess → 着色 primal（接触+关节+弹簧同一 6×6/3×3）→ dual → BDF。GPU：同色并行、色间提交 | 叶→根扫序；全岛 Jacobi；按物体类型再开一条积分 |
| `PhysX_AVBD_2` 绿测路径 | PhysX 合同与尺度：几何热启动闸门、`ρ≥2/(w dt²)`、材料互补块、headless 数字 | 整文件、Scene glue、owner 断言、artic 公共平移、软体独立 DCD/材料 pass、关节独立积分 |

论文循环以 demo3d 为准。**Backend 编排以 webphysics 为准。** Host 场景合同以 AVBD_2 headless 协议为准。

## 压成最小形式时保留

- 刚体 6×6、顶点 3×3；中间表示只有 `AvbdRow`
- primal 默认原地 GS。GPU：接触图着色，同色并行、色间提交。禁止为躲开 GS 改全岛 Jacobi
- 迭代次数放 backend 常量，不往 `PxSceneDesc` 加 `avbd*` 字段
- ASCII 注释（MSVC C4819 / 代码页 936）

## 提炼过的形状（对照，不是粘贴清单）

- 接触 λ：几何持久性闸门（局部锚点 + 法向），不是流索引
- 法向 ρ：per-row `2/(J M⁻¹ Jᵀ dt²)`，不是全局常数地板
- 切向：独立 `rhoT`，锥内才增长
- 速度：contact-only 连通组件独占 solve-start；不要 `hasContact` 泛洪
- 材料：累加冲量 NCP，不要每轮 `-v/w`
- drive ρ：`min(0.25·ρ_lock, 100/h)`，不要把 `stiffness=1e5` 当 AL ρ
