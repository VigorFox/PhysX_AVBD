from pathlib import Path
import gzip,hashlib,json,difflib

root=Path.cwd();source=Path(__file__).parent;out=root/'tools/fixtures/bounded-work-20261106'
out.mkdir(parents=True,exist_ok=True);entries=[]
def save(path,relative,compress=False):
    assert path.is_file(),path
    data=path.read_bytes();target=out/relative
    if compress:target=target.with_name(target.name+'.gz')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(gzip.compress(data,mtime=0) if compress else data)
    entries.append(dict(file=str(target.relative_to(out)),source=str(path.relative_to(root)),bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),storedSha256=hashlib.sha256(target.read_bytes()).hexdigest()))
for p in source.iterdir():
    if p.is_file() and (p.suffix in ('.log','.txt','.json','.py') or p.name.startswith('DyAvbd') or p.name=='stagnation-probe.spectral.cpp'):
        save(p,p.name,p.suffix in ('.log','.txt') or 'candidate' in p.name or 'failure' in p.name or p.name.startswith('DyAvbd'))
for name in ('publication-t0','publication-t1','publication-t4'):
    for p in (source/name).iterdir():
        if p.is_file():save(p,name+'/'+p.name,p.suffix=='.log')
paths=['physx/source/lowleveldynamics/src/avbd/'+n for n in (
    'DyAvbdCommonProgram.h','DyAvbdCommonResidual.h','DyAvbdElasticFrame.h','DyAvbdRigidMetric.h',
    'DyAvbdDynamics.cpp','DyAvbdDynamics.h','DyAvbdSolve.cpp')]
paths+=['tools/avbd_stagnation_mechanism_probe.cpp','tools/analyze_avbd_stagnation.py',
        'docs/BOUNDED_WORK_CURVATURE_20261106.md','docs/REFACTOR_OUTLINE.md','docs/ITERATION.md',
        '.cursor/skills/avbd-whole-step/dead-paths.md']
for path in paths:save(root/path,'sources/'+path.replace('/','__'),True)
for name in ('DyAvbdDynamics.h','DyAvbdDynamics.cpp'):
    assert (root/'physx/source/lowleveldynamics/src/avbd'/name).read_bytes()==(source/(name+'.before')).read_bytes()
for name in ('DyAvbdSolve.cpp','DyAvbdCommonResidual.h'):
    current=(root/'physx/source/lowleveldynamics/src/avbd'/name).read_text()
    assert not any(token in current for token in ('secantStart','unitSearch','commonLineScale','COMMON_LINE_EXTENSION'))
    before=(source/(name+'.before')).read_text()
    (out/(name+'.turn.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),current.splitlines(True),fromfile='turn-start',tofile='retained')),encoding='utf8')
dll=root/'physx/bin/win.x86_64.vc143.md/checked/PhysX_64.dll'
manifest=dict(scope='Stable quadratic material work; all scalar extension trials withdrawn; full five-body still fails at frame 18',
              dllSha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
              baseline='tools/fixtures/bounded-root-20261106/manifest.json',
              defaultChanged=False,numericalBudgetsChanged=False,fullFiveQualified=False,entries=entries)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
for item in entries:
    p=out/item['file'];data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==item['storedSha256']
    if p.suffix=='.gz':data=gzip.decompress(data)
    assert hashlib.sha256(data).hexdigest()==item['sha256']
print(json.dumps(dict(files=len(entries),bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),verified=True)))
