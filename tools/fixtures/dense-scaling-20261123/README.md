# 高密度单场景强扩展实验

[实验说明与对照表](../../../docs/DENSE_STRONG_SCALING_20261123.md)给出场景、时间窗口、统计方法和采用决定。

- `ordinary-window.patch` 是实际测量的普通颜色批次归并实现。它相对于本轮生产源码，只涉及 `DyAvbdDynamics.cpp` 和 `DyAvbdColorTasks.h`。在包含此前工作区改动的隔离源码副本内，可以先执行 `git apply --check tools/fixtures/dense-scaling-20261123/ordinary-window.patch`，应用补丁后编译 Checked SDK。两组 SDK 使用同一份更新后的 snippet。
- `measurements.zip` 保存布料与五体每次重复的完整步 CSV、场景门禁、逐帧诊断及各 SDK 指纹。目录 `paired-window` 为归并实验的交错 A/B，`production` 为独立生产对照，`eight-E`、`eight-E-paired` 分别为固定八个 E 核的生产对照和交错 A/B，`fans-paired` 为上轮就绪树与生产的 32 中心对照，`refined` 记录 99×99 反例。公共回归结果在 `qa/summary.json`。
- 多中心实验中的就绪树源码在[上一轮测过的补丁](../ready-tree-20261123/root-merge/candidate.patch)。普通颜色归并与就绪树分别对照生产 DLL，实验间没有叠加补丁。

生产 DLL SHA-256 为 `6b3badbf3c6de581705b22fa9a00ca9d35d01e33d255f2d990327ece30005e27`，普通颜色候选为 `eb7cf4791aa68b51cad43138a679fa615cf1e224a0300ee0a4f73c7aced57946`，就绪树为 `ce4f62437941054dfad95a76595cd2979562476c839fc2cc99625c8ab9a9f26a`。
