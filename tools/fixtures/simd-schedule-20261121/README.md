# Unified SIMD scheduling evidence

See [implementation and acceptance](../../../docs/AVBD_SIMD_SCHEDULING_20261121.md) and [manifest](manifest.json).

Each `.json.gz` decompresses to a UTF-8 JSON object. `sources.json.gz` maps repository paths to exact candidate source contents. `rejected-sources.json.gz` contains the S2 source snapshots and rejected scheduling variants; these are historical inputs, not production files. `results.json.gz` contains full run summaries and preregistrations. `raw-runs.json.gz` preserves per-frame CSV, event logs, profiles and the original failed benchmark report.

The adopted candidate removes no-op dual dispatch and uses nodal energy scratch in the shared four-iteration AVBD pipeline. Cloth8's fourteen-pair median mean/P95 ratios are 0.94793/0.94379. The original five-body 5% target and relative Rigid6 model-scaling gates remain failed. GPU execution has not been qualified. No samples or physics thresholds were changed to grant those qualifications.
