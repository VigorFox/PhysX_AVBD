# Cost-constrained native restart

Baseline: `f9abbe0cd869136d20cba6462551bbb141a75cd0`.
Complete experiment archive: branch `codex/archive-particle3-20261117`, commit
`b51642f3897b82632e9ff0a023d47f4600757153`.

`extraction.json` identifies the archived working tree and extracted file hashes.
The two ZIPs contain frozen research candidates, physical tests and counterexamples;
they are not build inputs and are not an alternative production solver. Retrieve
missing dependencies from the archive only after reviewing their necessity and
whole-step cost. See `docs/NATIVE_COST_RESTART_20261117.md` for transfer boundaries.

`baseline-summary.json` records 43/43 checkpoint checks, four 360-frame budget
audits, and 18 unprofiled timing runs (three repeats at 0/4/8 workers per scene).
`baseline-evidence.zip` preserves raw logs, initial dependency failures, source/
product fingerprints and benchmark reports. `build-provenance.json` records the
fresh CPU Checked configuration. `publication.json` records the verified overlay
onto default Checked products and the local backup of overwritten artifacts.
Raw build/run logs remain under `tools/benchmark-results/restart-baseline-20261117`.
Initial missing-FreeGLUT launches are retained as infrastructure failures; selective
retry requires unchanged hashes of every previously verified PhysX source/product.
No physical threshold or solver implementation was modified for the restart.
