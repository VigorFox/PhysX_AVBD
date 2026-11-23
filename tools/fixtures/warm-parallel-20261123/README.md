# CPU warmstart 与同色派工对照

[实验说明](../../../docs/CPU_WARM_ADMISSION_20261123.md)列出了整步、进程 CPU、固定帧长尾及阶段跨度。`measurements.zip` 中均为 Checked headless 实测。

| 目录 | 内容 |
|---|---|
| `paired/` | 单独并行化初猜的四组交错对照 |
| `combined/` | 默认 Checked 同时启用初猜并行与同色普通窗口归并的四组交错对照 |
| `eight-cores/` | 四布，两档 worker 共用八个 E 核的进程掩码，两组交错对照 |
| `fans/` | 单场景 32 个高价度根，每根 384 面，共四组交错对照及一次独立 profile |
| `fixed-cpu/` | 五体固定 18 帧，两个 worker 档各六对，逐帧进程 CPU 时间、周期和整步 |
| `qa/` | 最终 Checked 的回归汇总及初猜反例、落布、五体关键日志 |

布与五体的正式计时保存所有帧的 CSV 和运行门禁，profile 单独采集。四布对照的 `summary.json` 和高价度探针的 `identity.json` 记录二进制 SHA-256。固定帧实验两边使用同一个 `CpuWindow.exe`（SHA-256 `d71c8f82fa9021f82378e5adf10ddc0f7d36270f8933d8d7073269b3bd5c7e0d`），其采样位置见[进程探针构建器](../mixed-batch-20261123/build_cpu.py)。`combined.patch`、`warm-only.patch` 均相对本轮开始时的工作树生成，已在临时目录验证可还原对应的四个核心源文件。
