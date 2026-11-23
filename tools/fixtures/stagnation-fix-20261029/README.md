# Stagnation repair evidence, 2026-10-29

Base: `ead9fb3`. Scope and limitations: [implementation report](../../../docs/AVBD_STAGNATION_FIX_20261029.md).

- `validation.json`: separate numerical, public, old NP-trial and new common-path observations. Timing runs were not isolated.
- `geometry.txt`, `publication.txt`, `independent.txt`: actual SDK scene tests, not frozen solves.
- `five-new-first16.txt`: full five-body asset, Checked, common path enabled. **Only 15 accepted frames; frame 16 rejects.** This is not a 360-frame scene qualification.
- `failed16.json.gz`, `failed16-next.json.gz`: original 16-/24-contact sticking failures, now passing the strict shared Table replay. Corresponding `*-table.json` records precise/fast, cold/captured seeds.
- `failed16-third.json.gz`: unresolved 26-contact optional SOC input. No PASS claimed.
- `remaining-main27.json.gz`: still-unresolved 27-contact primary NCP. `remaining-main27-table.json` preserves independent replay failures; numerical replay uses its own reported reduced-input budget.
- `final-mechanisms.json`: shared-kernel mechanism counterexamples, not scene/GPU qualification.

Reproduce the positive frozen tests:

```powershell
python tools/replay_avbd_coulomb_capture.py tools/fixtures/stagnation-fix-20261029/failed16.json.gz
python tools/replay_avbd_coulomb_capture.py tools/fixtures/stagnation-fix-20261029/failed16-next.json.gz
python tools/analyze_avbd_stagnation.py
python tools/run_avbd_common_step_probe.py
$env:PHYSX_AVBD_COMMON_STEP = '1'
python tools/run_avbd_np_geometry_scene_probe.py
python tools/run_avbd_np_geometry_scene_probe.py 0 independent
```

Reproduce the **expected failing** short full-asset diagnostic (a short request also reports `reason=incomplete`; inspect fatal errors and accepted/rejected frame counters separately):

```powershell
$env:PHYSX_AVBD_COMMON_STEP = '1'
$env:PHYSX_AVBD_COMMON_STATS = '1'
$env:PHYSX_NP_GEOMETRY_CERTIFICATES = '2'
& physx/bin/win.x86_64.vc143.md/checked/SnippetJointSoftBody_64.exe --headless --case=visual-showcase --solver=avbd --frames=16 --grid=5 --dispatcher-threads=4 --volume-material
```

For old-path regression, remove the common-step/statistics variables and run `tools/run_snippet_joint_soft_body_headless.py --volume-material --np-domain trial --cases reaction,compression-recovery,visual-showcase --frames 360 --timeout 600`. NP trial is still experimental; default public PGS/TGS/AVBD regressions are separate.
