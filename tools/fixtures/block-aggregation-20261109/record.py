from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import sys

root = Path.cwd()
work = Path(__file__).parent
out = root / 'tools/fixtures/block-aggregation-20261109'
out.mkdir(parents=True, exist_ok=True)
result = subprocess.run([sys.executable, str(root / 'tools/run_avbd_contact_bundle_probe.py')],
                        cwd=root, capture_output=True)
(work / 'mechanism.log').write_bytes(result.stdout + result.stderr)
print(result.stdout.decode('utf8', errors='replace'))
assert result.returncode == 0, result.returncode
entries = []

def save(path, relative, compress=False):
    data = path.read_bytes()
    target = out / relative
    if compress:
        target = target.with_name(target.name + '.gz')
    target.parent.mkdir(parents=True, exist_ok=True)
    stored = gzip.compress(data, mtime=0) if compress else data
    target.write_bytes(stored)
    entries.append(dict(file=target.relative_to(out).as_posix(), source=path.relative_to(root).as_posix(),
                        sha256=hashlib.sha256(data).hexdigest(), storedSha256=hashlib.sha256(stored).hexdigest()))

for name in ('inventory.json', 'mechanism.log', 'record.py'):
    save(work / name, name)
inventory = json.loads((work / 'inventory.json').read_text())
for item in inventory:
    path = root / item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    save(path, 'inputs/' + path.name, path.suffix != '.gz')
for name in ('tools/analyze_avbd_block_aggregation.py', 'tools/avbd_contact_bundle_probe.cpp',
             'tools/run_avbd_contact_bundle_probe.py', 'tools/validate_avbd_common_capture.py',
             'docs/LOCAL_BLOCK_AGGREGATION_20261109.md', 'docs/REFACTOR_OUTLINE.md', 'docs/ITERATION.md',
             '.cursor/skills/avbd-whole-step/dead-paths.md',
             'physx/source/lowleveldynamics/src/avbd/DyAvbdNpDomain.h',
             'physx/source/lowleveldynamics/src/avbd/DyAvbdCommonProgram.h'):
    save(root / name, 'sources/' + name.replace('/', '__'), True)
manifest = dict(scope='Read-only grouping audit and mechanism qualification; no production solver change',
                productionChanged=False, fullFiveQualified=False, speedupMeasured=False,
                baseline='tools/fixtures/bounded-curvature-model-20261106/manifest.json',
                tests=dict(mechanismPrecise=True, mechanismFast=True, solvedCaptureRoundtrip=True), entries=entries)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
for item in entries:
    assert hashlib.sha256((out / item['file']).read_bytes()).hexdigest() == item['storedSha256']
print(json.dumps(dict(files=len(entries), verified=True)))
