# AVBD correctness debt

This document is the maintained list of known AVBD correctness gaps. It is not
an iteration journal. Add an item only when a deterministic public or headless
gate exists, and remove it only after that gate and the surrounding regression
matrix pass.

## Verification baseline

- Last full verification: 2026-08-25, Windows x64 checked build.
- Pre-squash tested head: `77c2223f`.
- Comparison points: parent `8e228ce6` and remote baseline `4b1b9d9a`.
- The failures below reproduce before `77c2223f`; the deformable-mesh friction
  repair did not introduce them.
- The accepted cross-snippet matrix remains green: `cross14` passed 14/14.

The following surrounding coverage was green during the same run:

- AVBD soft-body component tests: 65/65 tests, 366/366 assertions.
- Deformable-volume correctness suite: 10/10 cases.
- Deformable-surface non-performance suite: 87/94 cases.
- CCD acceptance: 24/24 cases.
- Non-contact D6/drive matrices: 262/262 cases.
- Gear, fixed tendon, spatial tendon, mimic joint, custom constraint, native
  break/reaction, spherical-cone and revolute-motor acceptance suites.
- Parallel and sequential AVBD articulation suites, ArticulationRC, chainmail,
  HelloWorld sleep, deformable-mesh stack and sphere-shot gates.

## Open debt

### C1 - Contact mass scaling and force-limit semantics

Priority: highest. This affects public contact modification behavior and the
contact-plus-joint path. The two failures are likely related, but a shared root
cause has not yet been proved.

- `SnippetJointDrive` `contact72`: 24/72 pass and 48/72 fail.
  - 12 `contact_horizontal_conservation` failures at unit mass scaling.
  - 12 `force_mass_scaling` failures at mass scale 10.
  - 12 `contact_horizontal_conservation` failures with a high force limit.
  - 12 `force_limit_ignored` failures with a low force limit.
- `SnippetContactModification` acceptance: 22/23 physical cases pass.
  `finite-scales-avbd-parallel` fails with `finite_scales_not_consumed`; the
  finite inverse-mass and inverse-inertia scales do not produce the required
  change in angular response.

Closure requires all 72 contact-drive configurations and all contact
modification acceptance cases to pass for AVBD parallel and sequential lanes,
without weakening the TGS-calibrated conservation, scaling or force-limit
oracles.

### C2 - Rigid spatial recovery and restitution parity

Priority: high. The individual simulations remain finite and repeatable, but
the acceptance-level physical comparison fails.

- Deep tilted overlap: AVBD launches with peak angular speed `0.098110199`,
  while TGS is approximately `3.9e-6`. The AVBD body settles to
  `0.00309117627`, but the pose-derived initial angular launch is invalid.
- Finite off-center impulse: AVBD reports
  `offcenter_finite_impulse_not_exercised` instead of completing the finite
  spatial response contract.
- Tilted spatial restitution: AVBD peak upward velocity is `1.55201185` versus
  TGS `2.05320573`, narrowly exceeding the current `0.5` parity bound.

Closure requires the three rigid acceptance runners to pass with their current
TGS-calibrated limits and with identical AVBD parallel/sequential results.

### C3 - Deformable-surface mixed-contact dissipation

Priority: high. Four 600-frame cases retain excessive soft velocity after the
contact event:

- `surface-dynamic-sphere`: peak surface speed `17.9770145`, final speed
  `1.97252917`.
- `surface-dynamic-capsule`: excessive mixed-rigid tail speed.
- `surface-dynamic-convex`: excessive peak and tail speed, followed by escape
  from the fixture.
- `surface-volume-wake`: driving-surface final speed `1.1338532`; the woken
  volume final speed is `0.240314946`.

Closure requires all four cases to pass at 600 frames in parallel and
sequential execution. The fix must preserve the already-green surface ground,
material-friction, soft-soft wake, attachment, bending, flattening and motion
control cases.

### C4 - Deformable-surface swept CCD response gaps

Priority: high for collision completeness. Three 600-frame cases fail:

- `surface-dynamic-rotating-capsule-relative-swept-ccd`: no distinguishable
  two-sided angular response.
- `surface-dynamic-rotating-convex-reverse-swept-ccd`: the convex crosses the
  swept soft feature without the required response.
- `surface-static-heightfield-reverse-swept-ccd`: no triangle-surface swept
  response is observed.

Closure requires the positive cases to respond without tunneling while their
paired negative controls remain inactive. Do not close these by relaxing the
feature or response witnesses.

### C5 - Soft-body component runner coverage drift

Priority: low, but fix before relying on the wrapper in CI. The executable now
runs tests 1 through 65 and reports 366/366 assertions passing, while
`tools/run_snippet_soft_body_avbd_headless.py` still expects tests 1 through 63.
The wrapper therefore reports a false failure after two physically successful
runs.

Closure requires the wrapper's accepted ID range and command-line choices to
include tests 64 and 65, followed by a green acceptance run.

## Authoritative reruns

```powershell
python -B tools/run_avbd_cross_snippets_headless.py --suite cross14 --timeout 180
python -B tools/run_snippet_deformable_volume_avbd_headless.py --mode correctness --execution parallel --timeout 180
python -B tools/run_snippet_joint_drive_headless.py --suite contact72 --timeout 60
python -B tools/run_snippet_contact_modification_headless.py --mode acceptance --timeout 120
python -B tools/run_avbd_rigid_deep_overlap_headless.py --mode acceptance --timeout 120
python -B tools/run_avbd_rigid_finite_impulse_offcenter_headless.py --mode acceptance --timeout 120
python -B tools/run_avbd_rigid_restitution_spatial_headless.py --mode acceptance --timeout 120
```

Run each failing deformable-surface case with:

```powershell
python -B tools/run_snippet_deformable_surface_avbd_headless.py --case <case> --frames 600 --execution parallel --timeout 180
```

The checked binaries must be rebuilt from the tested source before using these
results as merge or release evidence.
