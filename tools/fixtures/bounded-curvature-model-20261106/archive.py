from pathlib import Path
import gzip, hashlib, json, difflib

root = Path.cwd()
source = Path(__file__).parent
out = root / 'tools/fixtures/bounded-curvature-model-20261106'
out.mkdir(parents=True, exist_ok=True)
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
                        bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                        storedSha256=hashlib.sha256(stored).hexdigest()))

for p in sorted(source.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts:
        continue
    if p.suffix not in ('.txt', '.log', '.json', '.py', '.npz', '.before', '.split', '.capture'):
        continue
    save(p, p.relative_to(source), p.suffix != '.npz' and (p.stat().st_size > 100000 or p.suffix in ('.txt', '.log', '.before', '.split', '.capture')))

core = [
    'DyAvbdElasticFrame.h', 'DyAvbdCommonProgram.h', 'DyAvbdCommonResidual.h',
    'DyAvbdBoundedResponse.h', 'DyAvbdCurvatureMetric.h', 'DyAvbdDynamics.h',
    'DyAvbdSolve.cpp', 'DyAvbdContactPrep.cpp',
]
paths = ['physx/source/lowleveldynamics/src/avbd/' + name for name in core]
paths += ['tools/' + name for name in (
    'run_avbd_local_response_probe.py', 'avbd_local_response_probe.cpp',
    'avbd_frame_curvature_probe.cpp', 'run_avbd_frame_curvature_probe.py',
    'avbd_bounded_response_probe.cpp', 'avbd_np_factorization_probe.cpp',
    'run_avbd_np_factorization_probe.py',
)]
paths += ['docs/ASSEMBLED_CURVATURE_20261106.md', 'docs/REFACTOR_OUTLINE.md', 'docs/ITERATION.md',
          '.cursor/skills/avbd-whole-step/layers.md', '.cursor/skills/avbd-whole-step/dead-paths.md']
for path in paths:
    save(root / path, 'sources/' + path.replace('/', '__'), True)

delta = []
for name in core:
    before = source / (name + '.before')
    current = root / 'physx/source/lowleveldynamics/src/avbd' / name
    old = before.read_text(encoding='utf8').splitlines(True) if before.exists() else []
    new = current.read_text(encoding='utf8').splitlines(True)
    diff = ''.join(difflib.unified_diff(old, new, fromfile='turn-start/' + name, tofile='retained/' + name))
    path = source / (name + '.turn.diff')
    path.write_text(diff, encoding='utf8')
    save(path, 'diffs/' + path.name, True)
    lines = diff.splitlines()
    delta.append(dict(file=name, added=sum(x.startswith('+') and not x.startswith('+++') for x in lines),
                      removed=sum(x.startswith('-') and not x.startswith('---') for x in lines)))
for name in ('DyAvbdElasticFrameCurvature.h', 'DyAvbdMaterialAssembly.h'):
    assert (root / 'physx/source/lowleveldynamics/src/avbd' / name).read_bytes() == (source / (name + '.before')).read_bytes()
dll = root / 'physx/bin/win.x86_64.vc143.md/checked/PhysX_64.dll'
manifest = dict(
    scope='Assembled native-frame curvature K, original PSD metric M, local nonpositive-mode protection; complete implicit normal-domain query is mechanism-only',
    baseline='tools/fixtures/bounded-work-20261106/manifest.json',
    finalDllSha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
    finalDllRuns=['final18-t4', 'reaction-final'],
    longRunProvenance='local: build-local.log, before lazy metric allocation and diagnostic cleanup; not assigned the final DLL hash',
    defaultChanged=False, numericalBudgetsChanged=False, tolerancesChanged=False,
    fullFiveQualified=False, gpuQualified=False, productionImplicitDomain=False,
    result=dict(finalFive18Accepted=18, finalFive18Nonstationary=0, longFiveAccepted=37,
                longFiveNonstationary=5, longFiveTimedOut=True, reaction360Passed=True),
    coreDelta=delta, entries=entries)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
for item in entries:
    p = out / item['file']
    data = p.read_bytes()
    assert hashlib.sha256(data).hexdigest() == item['storedSha256'], p
    if p.suffix == '.gz':
        data = gzip.decompress(data)
    assert hashlib.sha256(data).hexdigest() == item['sha256'], p
print(json.dumps(dict(files=len(entries), bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                      verified=True, coreDelta=delta)))
