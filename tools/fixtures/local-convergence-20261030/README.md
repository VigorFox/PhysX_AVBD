# Local convergence evidence, 2026-10-30

Final retained implementation: original-force mixed Newton, composed Anderson history, optional full SOC trial with original-direction restoration, and affine-domain backtracking bounds. The default common-step switch remains off. See [the report](../../../docs/LOCAL_CONVERGENCE_20261030.md).

`retained-five360.json` records a **failed** 360-frame target: 28 accepted, nonstationary frames and rejection of frame 29. `retained-reaction360.json` records 360 accepted stationary frames. A diagnostic process stopped on rejection is not a completed gate. Runs overlapped other verification and do not establish performance speedups.

`remaining-135.json.gz` is the final failed frozen program. `retained135*` fails the local engineering solver; `retained135-strict*` independently verifies a strict root. `retained240*` and `retained280*` pass engineering closure and an independent full-H displacement check; their strict validation remains false. The earlier 240-row source is `../local-patches-20261029/certificate-failure.json.gz`.

`trial-*`, `trials/`, and the projected-stagnation capture are withdrawn variants, retained only as counterexamples. They are not qualifications of the final code. Geometry captures contain the frozen program and prior force, not an exact replay of all candidate poses and geometry.

`manifest.json` records source and DLL fingerprints, invocation environment, and evidence hashes. `retained-checks.json`, `retained-replays.json`, and `public-checks.json` record commands and outcomes. Expected local failure of the remaining input is intentionally preserved.
