# AGENTS.md — PhysX_AVBD_3

## 主目标（本树）

在官方 PhysX SDK **5.9.0**（tag `110.1-omni-and-physx-5.9.0`）上做 **AVBD 蒸馏移植**：干净 Host 上接一种 SolverBackend，不从 `D:\projects\PhysX_AVBD_2` 搬 11 万行。

- **架构 / 禁止清单 / 完成标准**：`docs/REFACTOR_OUTLINE.md`（本地开发工作规格）。发布版见[架构合同与 headless 用法](docs/AVBD_ARCHITECTURE.md)
- **阶段状态 / 怎么跑**：`docs/ITERATION.md`
- **公式源**：论文 AL 以 `reference/avbd-demo3d` 为准；**统一步进管线**以 `reference/webphysics` 为准（`DCD(x^t) →` 全员惯性目标 → 着色 primal → dual → BDF）；PhysX 合同与尺度从 `PhysX_AVBD_2` 绿测路径蒸馏，**不要**照搬 AVBD_2 里 artic / 软体 / 关节各自独立的 pass
- **规格源**：AVBD_2 的场景合同 / headless 协议；不要 source-gate / owner 断言

AVBD 是 PhysX 里的一个 SolverBackend，不是第二套引擎。BP / NP / island / sleep / CCD / `Sc::Scene` 与求解器类型无关。

## 必读 SKILLS

**软体与布料优先并行，不得为逐位／轨迹相等牺牲并行性**：遵守唯一规格 [§0.2](docs/AVBD_ARCHITECTURE.md#parallel-performance-first)。包含“数学等价优化”在内均按物理质量与成本验收；旧线程一致性门禁和历史浮点差异不能覆盖此用户要求。当前设计见 并行优化计划（本地 `docs/NATIVE_SOFT_CLOTH_PARALLEL_PLAN_20261109.md`）。

**新算法若为准确性牺牲 AVBD 并行性，直接判定失败。** 2026-11-17 已授权将 Particle3 实验完整归档、从 `f9abbe0` 重启；后续必须同时验收正确性、整步成本和并行性，固定统一 4 次迭代。唯一规则仍在大纲 §0.2；归档与重启计划见 成本约束重启（本地 `docs/NATIVE_COST_RESTART_20261117.md`）。不得把归档实验整条管线回灌，或以“先正确、以后再优化”恢复它。

动手改求解器或排查门禁 / 观感前，读并遵守：

| Skill | 何时 |
|---|---|
| [`.cursor/skills/avbd-whole-step/SKILL.md`](.cursor/skills/avbd-whole-step/SKILL.md) | 能量、抽搐、嵌入、自旋、假绿、想加 `if` / 帽 / ρ 旋钮 |
| [`.cursor/skills/avbd-distill-reference/SKILL.md`](.cursor/skills/avbd-distill-reference/SKILL.md) | 去 AVBD_2 / demo3d / webphysics 挖方案 |
| [`.cursor/skills/avbd-unified-row/SKILL.md`](.cursor/skills/avbd-unified-row/SKILL.md) | 加关节 / CCD / 软体；任何软体性能、归并、预烘焙、并行 / GPU 优化；新 flag / sidecar 或第二条管线 |

层合同：[layers.md](.cursor/skills/avbd-whole-step/layers.md)。勿重试：[dead-paths.md](.cursor/skills/avbd-whole-step/dead-paths.md)。

拿不准时先改大纲再改代码。代码与 layers 冲突时改其中一方，禁止在本文件里另写一套过期分叉。

**软体优化不得分叉是用户设定的长期硬边界**：动手前必读并遵守 [大纲 §0.1](docs/AVBD_ARCHITECTURE.md#soft-optimization-no-fork)。允许统一框架内专用 kernel、预烘焙和约束归并；不得恢复刚软两套物理循环、事后补偿或隐藏回退。Agent 不得借“先改大纲”自行弱化该边界；普通优化无需另行确认。

## 当前阶段

见 `docs/ITERATION.md`。一次只做一个能力阶段；未通过不得声明下一阶段／生产资格。当前阶段内的性能、局部机制与并行改造可继续，不要求旧完整五体先绿或复现旧轨迹，按 [大纲 §0.2](docs/AVBD_ARCHITECTURE.md#parallel-performance-first) 独立验收。PGS/TGS 每个涉及物理实现的阶段仍须通过公共回归。

未完成项优先做其余 snippet 名单（OGC 高速非穿透 / ContactModification）。HelloWorld 刚体合同已绿，含 `sub-bounce-stack`。Joint / ArticulationRC 剪刀 HARD 已绿。可视化 snippet 的门禁放 `SnippetXxxHeadless.{h,cpp}`，原 cpp 只留入口。

C1–C4 **不是**停船。非穿透、能量有界、能睡、contact modification、force limit、关闭 CCD 后的 OGC 高速极性、软刚作用反作用 **是**停船。详见大纲第 0 节。

## 架构不变量

- Host 管物理事实，Backend 管打包数学，Table 管在哪执行
- 约束中间表示只有 `AvbdRow`；刚体永远 6×6、顶点永远 3×3。加能力必须仍进统一 row/block；否则先改大纲。禁止 sidecar/flag 网络重造 owner，禁止 Scene glue / standalone 双轨 / 几何专用 task graph
- 接触只读 `PxsContactManagerOutput`（NP）；普通关节走 `solverPrep` → `Px1DConstraint`。原生材料可用 Host 批次直接编译到统一 row/block；数据组装与算子不必沿用完整 joint，详见大纲 §0.1
- 关节树：link 当 6-DOF body + inbound 1D row，不接 Featherstone 内循环、不克隆 Scene GPU 上下文
- 迭代次数放 backend 常量，不往 `PxSceneDesc` 加 `avbd*` 字段
- Scene 步进路径除构造外无 `if (eAVBD)` / `*Avbd*` API
- primal 默认原地 GS；GPU 着色后同色并行、色间提交。禁止叶→根 / 贴地多解一次
- 刚体一步合同见 [layers.md](.cursor/skills/avbd-whole-step/layers.md)（统一 `pHat` epoch、世界角增量、contact-only 材料独占）
- 软刚同一对一份 dual；Host overlap 不得无条件唤醒动态刚体；未接的公开 API 必须文档化为不支持

## 禁止

见大纲第 8 节。本树尤其禁止：复制 `DyAvbdSolver.cpp`、Scene `*Avbd*` 方法、source gate、owner 枚举、standalone 与 PhysX 双写同一公式。

**禁止用补丁修物理求解问题**（skill `avbd-whole-step`）。矩阵绿但观感错，先怀疑假绿和互斥分叉。

**默认原地 GS**。不要为躲开 GS 改全岛 Jacobi。

## 怎么验证

```
sceneDesc.solverType = PxSolverType::eAVBD;
tools/avbd_smoke.exe
python tools/run_snippet_hello_world_headless.py
python tools/run_snippet_hello_world_headless.py --cases sub-bounce-stack
python tools/run_snippet_joint_headless.py --cases gated
python tools/run_snippet_articulation_rc_headless.py --cases gated
python tools/run_snippet_articulation_rc_headless.py --solver pgs --cases gated
```

PGS/TGS 默认 snippet 必须仍能编过、行为不变。HelloWorld 已走 AVBD；`--headless` 走 `SnippetHelloWorldHeadless.cpp`。SnippetJoint 可视化仍默认 PGS；headless 门禁走 `SnippetJointHeadless.cpp`（AVBD gated + PGS `--solver pgs`）。SnippetArticulationRC 可视化仍默认 AVBD；headless 走 `SnippetArticulationHeadless.cpp`（AVBD / PGS `--solver pgs` 剪刀 HARD 已绿）。

## 仓库其它目录

| Directory | What it is |
|---|---|
| [`physx/`](physx/) | PhysX SDK — 本树改动集中在 `lowleveldynamics/src/avbd/` |
| [`ovphysx/`](ovphysx/) | USD-based Python/C 绑定；本阶段不是主线 |
| [`blast/`](blast/) | Blast SDK |
| [`flow/`](flow/) | Flow SDK |
| [`omni/`](omni/) | Omniverse PhysX extensions |
