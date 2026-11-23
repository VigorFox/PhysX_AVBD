# 同色局部块就绪推进的 Checked 对照

对照对象为 2026-11-21 SIMD 工作树上的生产 Checked DLL。本目录保存首版按块就绪的完整对照，[root-merge](root-merge/) 保存将最后一次二元归约并入块释放信号后的独立完整对照。每组的 `manifest.json` 记录二进制及源码指纹。两份 `candidate.patch` 均可相对原生产工作树独立应用。技术说明见[工作树实验记录](../../../docs/BLOCK_READY_TREE_20261123.md)。

`raw.json.gz` 是按相对文件名索引的日志和 CSV，`runs.json.gz` 保存每次运行的指标。`registration.json` 给出冻结帧、重复数和判定口径。`physics.json` 为完整公共回归结果。`multi-star.json` 和 `wide-rigid.json` 给出两个额外的真实 SDK 压力用例。

在仓库根目录执行以下命令，可以在临时目录重新计算成对统计并与归档比对。

```powershell
python tools/fixtures/ready-tree-20261123/replay.py
python tools/fixtures/ready-tree-20261123/root-merge/replay.py
```

同机重新采样时，使用 `tools/benchmark-results` 下的独立实验目录。基线沿用同级的 `color-overlap-20261123/baseline`，固定帧 CPU 探针来自 `mixed-batch-20261123/cpu-base`。复制基线运行文件至实验的 `candidate`，在 `src/avbd` 放入打过补丁的求解器源文件。将本目录的 Python 驱动与注册文件复制到实验目录后，先运行 `build.py` 编译候选，再在保留生产源码的工作树上运行 `build_diagnostics.py` 编译两边的同口径跨度探针。这些脚本复用本机 MSVC 已生成的 Checked 构建配置。

将 `AVBD_PROBE_BINARY_DIR` 指向候选目录，用仓库的 `tools/verify_particle3_shared_np.py` 生成实验目录下的 `qa/summary.json`，运行 `stress.py` 和 `rigid.py` 验证多根、同步执行和刚体高价度路径。最后运行 `study.py` 与 `analyze.py`。正式整步时间、逐帧 CPU 和工作树探针在不同进程中采样，物理回归与正式计时顺序执行。
