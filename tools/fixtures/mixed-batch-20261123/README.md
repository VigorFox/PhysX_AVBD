# 混合颜色派工的三方 Checked 对照

固定五体帧号、运行顺序、重复数和采用条件见 [registration.json](registration.json)。完整结果见[实验记录](../../../docs/MIXED_BATCH_TAIL_20261123.md)。

| 文件 | 用途 |
|---|---|
| `candidate.patch` | 在本次生产基线的 SIMD 工作树上应用的完整预分配候选，可用 `git apply --check` 检查 |
| `atomic-to-batch.patch` | 从上轮原子领取候选到本轮预分配的增量，源文件使用上轮实验副本的换行格式 |
| `batch-to-reserved.patch`、`reserve-summary.json` | 独占第一个 256 行就绪叶片的机制复测，相对连续预分配的增量与 12 次追加运行统计 |
| `tail-probes.json.gz` | 原子、预分配两种汇合探针各自相对生产基线的补丁 |
| `study.json.gz`、`raw.json.gz` | 运行顺序、二进制指纹、逐步 CSV、逐波汇合数据、逐帧进程 CPU 及物理回归原始输出 |
| `analysis.json` | 全部重复、两半样本、成对比值、扩展性与末任务统计 |
| `manifest.json` | 三种运行时、两种汇合探针和独立 CPU 探针的 SHA-256 |

在仓库根目录重算统计即可核对归档。脚本将所需输入解压到临时目录，再逐项比对结果。

```powershell
python tools/fixtures/mixed-batch-20261123/replay.py
```

同机重新采样使用独立的 Checked 运行目录。生产与原子版本分别沿用 `tools/benchmark-results/color-overlap-20261123/baseline` 和 `tools/benchmark-results/mixed-claim-20261123/candidate`。在 `tools/benchmark-results/mixed-batch-20261123` 中，`batch` 是相同 SDK 组件配上预分配候选的目录。目录中的 `build.py` 按本地已经生成的 MSVC Checked tlog 重编、独立链接 DLL。`build_tail.py` 生成两种独立诊断 DLL，`build_cpu.py` 为三种 DLL 配同一个进程 CPU 探针可执行文件。当前工作树与编译产物需要与清单中的基线相符。

公共回归由 `tools/verify_particle3_shared_np.py` 在 `AVBD_PROBE_BINARY_DIR` 指向 `batch` 时执行。结果放到实验目录的 `qa/summary.json`。随后在该实验目录运行 `python study.py` 和 `python analyze.py`。驱动将回归、计时、CPU 探针和汇合探针分开执行。汇合探针增加的时钟读取只用于归因。
