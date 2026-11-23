---
name: avbd-unified-row
description: >-
  Keeps PhysX_AVBD_3 on one Host/Backend/Table path and one AvbdRow/block
  dataflow when adding joints, CCD, articulations, soft, or contact
  modification, and when optimizing soft-body performance, constraint bundles,
  prebaking, or parallel/GPU execution. Use when growing AvbdRow, adding flags/sidecars, tempting
  an owner enum, a second pipeline, Scene glue, or a geometry task graph.
---

# 统一 row / block，不发明 owner

最大的未来风险：随着能力补齐，`AvbdRow` 变成装下所有语义的巨型联合体，或通过大量 sidecar / flag 重新发明 `_AVBD` 的 owner 系统。

诊断求解器用 `avbd-whole-step`。挖参考用 `avbd-distill-reference`。本 skill 管**加能力及做优化时数据流还在不在一条管线**。

## 软体优化的强制入口

同时必读唯一规格 [§0.2 并行优先、不要求逐位一致](../../../docs/AVBD_ARCHITECTURE.md#parallel-performance-first)。不得以数学等价、线程对照或 GPU 移植为由恢复结果相等门禁；按独立物理质量、总工作及真实扩展性验收。

任何软体提速、局部块／约束归并、预烘焙、活动集或 GPU 优化，开工前必读唯一规格 [§0.1 软体优化不得分叉](../../../docs/AVBD_ARCHITECTURE.md#soft-optimization-no-fork)。专用材料 kernel 和紧凑表示可以保留在统一步进中；独立软体物理循环、事后刚软补偿及隐藏回退被禁止，不能以“临时”或性能更好豁免。

方案和验收记录按该节回答：优化的方程／表示；状态与缓存的来源、更新及失效；进入统一 primal／dual／准入／BDF 的位置；共享反力、跨块项、工作上限及实际耦合回归。允许改变数值迭代轨迹，不要求逐位一致。

Agent 不得通过自行修改规格解除这一用户边界；框架内优化照常自主实施。独立实验只是同源机制驱动，不授权第二条生产线。规则细节只在大纲维护，不在本 skill 另建一份。

用户于 2026-11-13 明确：统一循环不要求沿用完整 joint 数据组装／算子。按大纲 §0.1 使用 Host 原生材料批次、专用 packer 和核是允许的；不要把普通关节的 solverPrep 合同误扩张成所有材料必须逐 joint 创建／准备。当前设计见 紧凑材料方案（本地 `docs/NATIVE_MATERIAL_COMPACT_DESIGN_20261113.md`）。

## 理想路线

- 保留 `_3` 的 Host / Backend / Table 边界
- 步进编排对照 `reference/webphysics`：一条 `DCD →` 惯性 → 着色 primal → dual → BDF，刚体 / artic link / 粒子只是同一调度上的块尺寸
- 从 `_AVBD` 迁移物理合同、epoch 定义和 fail-closed 原则；**不**迁移 AVBD_2 里 artic / 软体 / 关节各自独立的 pass
- 公共 `AvbdRow` header + 按 `KernelId` 索引的紧凑 payload 数组（接触/关节/drive/VBD 各一份 SoA）。当前把 joint/artic/drive/contact 字段堆进 `AvbdRow` 只是可行性原型，不是目标 IR。不迁移 owner 枚举网络
- 迁移测试矩阵，不迁移 Scene glue、standalone 双轨和几何专用 task graph
- 每增加一种能力，都要求它仍能进入统一 row / block 数据流，否则先修改架构规格，不在旁边新建第二条管线

## 开工前问

```
- [ ] 输入来自 Host NP、普通关节 solverPrep 或合法的原生材料批次，且支持仍进入 Host island node 吗？
- [ ] 中间表示仍是统一 row/block 数据流（公共 header + KernelId payload），而不是新 IR 或往 `AvbdRow` 再塞联合体字段？
- [ ] 执行仍走 Backend + KernelTable，没有 Scene *Avbd* / 第二上下文？
- [ ] 软体优化满足大纲 §0.1，仍共享物理步、完整耦合和一次发布／BDF，而非仅共用函数名？
- [ ] 缺合同是 fail-closed（文档化为不支持 + 门禁），不是静默丢掉或另开求解器？
- [ ] 测的是 AVBD_2 / snippet 矩阵合同，不是把 Scene glue 搬回来？
```

任一题是否：先定位缺失的数据流合同，再修改方案或规格；不得以修改规格绕过 §0.1 的用户边界。

## 迁 / 不迁

| 迁 | 不迁 |
|---|---|
| 物理合同（穿透、能量、睡眠、反弹、force limit、CCD 极性） | owner / kind / span 枚举网络 |
| WebPhysics 统一步进：全员同一 predictor、同一 primal、同一 dual、一次 BDF | AVBD_2 artic 公共平移 / pin-hat、软体每 sweep 全量 DCD、关节独立积分 |
| 统一 solve-start epoch；contact-only 组件由程序编译，不由 `|v|` 猜 | source gate、owner 断言、`hasContact`/`artic==NULL` 另选起点 |
| fail-closed：未接 API 具名不支持 | Scene `*Avbd*`、`if (eAVBD)` 热路径 |
| 测试矩阵 / headless 协议 | standalone 与 PhysX 双写公式 |
| 紧凑 payload 数组（按 KernelId：J/C/ρ/λ/μ/材料/驱动系数） | 几何专用 task graph、按接触类别分求解器、往 `AvbdRow` 堆 artic/drive/OGC 字段 |

`AvbdRow` 允许 `kind` + 少量 flags + `payloadIndex`。不允许：每能力一个 sidecar 图、每几何一条 pack 路径、同一 span 双 owner、把 `DyAvbdDynamics.cpp` 的单体 TU 当成最终 PhysX 风格落点。

顶点仍 3×3（VBD），刚体仍 6×6。软刚同一对一份 dual。那是同一数据流上的块尺寸，不是第二条管线。
