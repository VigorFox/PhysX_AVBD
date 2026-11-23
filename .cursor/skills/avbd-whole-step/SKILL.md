---
name: avbd-whole-step
description: >-
  Diagnoses and fixes PhysX AVBD rigid/soft solver bugs by whole-step layer
  contracts (inertia, warmstart, primal, dual, writeback, material NCP).
  Use when energy explodes, stacks twitch or interpenetrate, bodies spin up
  after impact, sleep fails, gates are green but visuals are wrong, or when
  tempted to add hasContact, speed caps, rho knobs, Jacobi caps, or extra ifs.
---

# AVBD 整步分解（禁止打补丁）

物理求解问题**禁止用补丁修**。本 skill 是工作方式，不是风格建议。

规格：`docs/REFACTOR_OUTLINE.md`。阶段状态：`docs/ITERATION.md`。
当前 A–E 合同：[layers.md](layers.md)。勿重试：[dead-paths.md](dead-paths.md)。
对照参考：skill `avbd-distill-reference`。加能力或做软体性能优化：skill `avbd-unified-row`；必须遵守唯一规格 [§0.1 软体优化不得分叉](../../../docs/AVBD_ARCHITECTURE.md#soft-optimization-no-fork)。

## 开工清单

软体／布料的并行和优化验收遵守 [唯一规格 §0.2](../../../docs/AVBD_ARCHITECTURE.md#parallel-performance-first)：逐位／轨迹相等不属于物理合同，不得为复现旧浮点结果串行化。历史反例应区分物理失败与数值差异。

卡住时先写层，再改代码。对照参考时先问「它规定的是哪一层」。

```
- [ ] 现象是哪一层在说谎（A–E，见 layers.md）
- [ ] 写出该层合同（谁拥有 pose / 速度 / C0 / λ）
- [ ] 删掉互斥分叉，改那一层的方程
- [ ] 用 pose / |ω| / 重叠验证观感，不只看矩阵绿
```

## 本能 vs 要求

错误：门禁红了 / 观感错了 → 再加 `if`、门槛、位移帽、ρ 地板、场景特例，让**这一道**变绿。

正确：整步分解——惯性目标、热启动、primal、对偶、速度写回、材料互补各管哪一层。改那一层的算法，删掉互斥分叉。

落地时拒：

- 为过某一门禁加 `hasContact` / `|v|` 写回 / Jacobi 1.8cm 帽 / 全局常数 ρ 地板
- 把 demo3d、webphysics、AVBD_2 焊进同一条 `if`（它们回答不同层）
- 从 AVBD_2 搬 artic / 软体 / 关节各自独立的 pass 来「修」统一 backend；步进编排对照 WebPhysics
- 红了就调 `ρ`、`α`、迭代、`C0/dt`，不先写出方程
- 用矩阵绿证明观感对（假绿：`minY` 贴地而塔中段悬空）

## 层怎么判

| 层 | 谎言长这样 |
|---|---|
| A 惯性 | 接触拓扑选了另一个起点；`pHat` 与材料用的步前速度不是同一 epoch |
| B primal | 重力压出弹簧；角 `log` 与 `applyDelta` 不在同一空间；按深度换 `C0` 定律 |
| C 对偶 | 冲击 λ 热启动到重排流形；切向共用长大后的法向 ρ |
| D 写回 | 世界左乘积分，体局部 `q⁻¹q+` 当 ω；AL 去穿透整段写成速度 |
| E 材料 | 每轮把 `-v/w` 当新冲量；用 `|Δv|` 门槛猜谁拥有速度 |

「谁拥有速度」是约束程序的属性，不是运行时门槛。contact-only 动态连通块独占 solve-start；带 joint / soft / artic 的组件拒绝独占。不要用 `|Δv|>1` 在 BDF 与 `v-` 之间切换。

李代数是一层合同：积分、惯性残差、BDF 必须同一只世界 `log`（`q+ q-⁻¹`，与 `q ← exp(Δθ) q` 一致）。

材料未知量是**累加冲量**。法向 `j ← max(0, j+(target−vn)/w)`；切向 `j ← Π_cone(j − ω vt)`。不要每轮重打 `-vt/w`。

## 验证

PGS/TGS 每阶段都要绿。AVBD 门禁：

```
tools/avbd_smoke.exe
python tools/run_snippet_hello_world_headless.py
python tools/run_snippet_hello_world_headless.py --cases sub-bounce-stack
```

矩阵绿但抽搐 / 嵌入 / 自旋：pose-log 或长时间 `|ω|` / 单帧 `Δθ`，不要再叠 `if`。代码与 [layers.md](layers.md) 冲突时，改 layers 或改代码，禁止让 `AGENTS.md` 留过期分叉。
