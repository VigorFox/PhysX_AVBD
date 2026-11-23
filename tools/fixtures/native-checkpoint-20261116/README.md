# 原生软体／布料当前成果快照

2026-11-16，保留去刚性单元改造之前的默认 Rigid6 原生材料路径。当前能力与历史失败说明以 `docs/ITERATION.md`、`docs/SOFT_BODY_ACCEPTANCE.md` 和 `docs/REFACTOR_OUTLINE.md` 为准。

- 默认四轮统一 primal／dual／准入／BDF；原生体积与布料均走当前 budget 路径。
- 完整回归 40／40，通过 0／4／8 workers、五体及两项反作用 1200 帧、原公共 AVBD／PGS／TGS 回归；额外 bounded、分组覆盖和任务界检查通过。
- 三组相邻交替性能对照、两组线程扩展性结果保留；没有逐位或轨迹相等验收。
- 其后仅修正锥体显示网格绕序，离线全部朝外、默认五体 360 帧 headless 通过。源／产物指纹与上述完整验证之间的差异已核对。
- `final-suite.json` 的 `firstExit` 以及单项的 `previousAttempt` 保留新增测试辅助函数首次编译失败的事实；修正后仅重跑该测试。生产场景没有因此改算法或放宽门禁。

仍未完成：稳定实时（五体 P95 约 57.5 ms）、充分 CPU 扩展性、真实 GPU 执行核、Host 原生材料批次、共享顶点 Particle3 在默认四轮路径中的完整接管、任意碰撞几何／组合的全面资格。当前布料仍为刚性薄片离散，格子内部形变能力不足。

这些 JSON 是本机历史验证记录，性能数值不能作为其它机器的阈值。命令和原始日志路径保留原运行环境；大型原始日志／二进制仍位于被忽略的 `tools/benchmark-results/` 和 `physx/bin/`。本目录不存放生成的可执行文件。

重新构建：

```powershell
cmake --build physx/compiler/vc17win64-cpu-only --config checked --target SnippetDeformableSurfaceAVBD SnippetDeformableVolumeAVBD SnippetJointSoftBody SnippetHelloWorld SnippetJoint SnippetArticulationRC -- /m:4 /nologo /v:minimal
```

场景和算子验证命令见 `final-validation.json`；将其中绝对仓库路径替换为当前 checkout。性能复测使用 `tools/benchmark_avbd_native_parallel.py`。本次提交是已验证阶段快照，不授予上述未完成能力的生产资格。
