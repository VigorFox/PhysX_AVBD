# 布料装配与行关联复用的 Checked 对照

[实验报告](../../../docs/CLOTH_CPU_ASSEMBLY_20261123.md)记录默认 50×50 落布、单场景四布和完整五体的整步、长尾、进程 CPU 与内存。

| [checked.zip](checked.zip) 内的目录或文件 | 内容 |
|---|---|
| `paired/` | 第一版并行装配及拓扑复用，三场景 × 两档 worker × 两种实现 × 四组正式计时，另有独立 profile |
| `retained/` | 最终版同时复用材料载荷和支撑，同一协议，Windows 默认调度 |
| `pinned/` | 最终版与同期原生产，同一组八个 E 核，默认布和四布，4 / 8 workers，三组交错正式计时和独立 profile |
| `qa/` | 最终构建的公共和物理门禁汇总、行关联探针、静止资产换版、四布接触、长时落布与五体卸载日志 |
| `diag.log`、`host-screen.log`、`retained-screen.log` | 原型中的 Host NP 读取时机失败与修正后的真实四布轨迹 |
| `parallel-pack.patch`、`retained-pack.patch` | 从本轮开始时的工作树分别重建两版核心，均已在当时的源文件上验证应用及内容 |

各计时目录的 `summary.json` 保留每对原数、加载的库和 snippet 的 SHA-256、进程掩码和扩展目标判断。每次运行有逐帧 wall time CSV 与独立的物理门禁日志。整步包括 `simulate + fetchResults`，诊断运行单独采集。五体专用并行候选的数据存放在[独立工程](../../../experiments/avbd_contact_lab/fixtures/five-parallel-20261123/README.md)。
