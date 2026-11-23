# 混合颜色派工复现

[实验记录](../../../docs/MIXED_COLOR_CLAIMS_20261123.md)说明物理合同、样本选择和计时口径。基线是本树 11 月 21 日的 Checked SIMD 工作树，三个被改动文件的原始 SHA-256 记在 `manifest.json`。在该版本上可以用 `git apply --check` 校验 `candidate.patch`，在独立源码副本应用后按原 Checked 配置构建。

同机复测可将两个 Checked 运行目录分别传给仓库的驱动，输出目录使用新的空闲路径。驱动关闭求解诊断，交替启动两种二进制，检查所有 360 个完整步和物理窗口。

```powershell
python tools/benchmark_avbd_simd.py --baseline $referenceBin --candidate $candidateBin --workers 4 8 --frames 360 --runs 8 --output $firstRunDir
python tools/benchmark_avbd_simd.py --baseline $candidateBin --candidate $referenceBin --workers 4 8 --frames 360 --runs 6 --output $confirmationDir
```

`entry.json.gz` 保存逐帧回调入口排队时间，`trace.json.gz` 保存逻辑工作项与数值入口诊断，`raw-runs.json.gz` 按相对路径保存各次输出和逐帧 CSV。解压 JSON 映射、按键名恢复文件，再将两份逐帧诊断解压为同名 JSON，并复制 `registration.json` 到结果根目录，就可以运行 `python analyze.py <结果根目录>` 重算统计。正式回归记录在 `quality-native/summary.json`。计时及诊断 DLL 指纹均可在清单核对。
