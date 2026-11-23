"""720-frame 8-worker five-body checks, after closing the timing series."""
from pathlib import Path
import json,subprocess
import study
p=Path(__file__).resolve().parent
assert json.loads((p/'runs.json').read_bytes())['complete']
target=p/'long';target.mkdir(exist_ok=True)
result=[]
for rep in range(2):
    for arm in (['base','ready'] if rep==0 else ['ready','base']):
        log=target/f'{arm}-{rep}.log'
        cmd=[study.bins[arm]/'SnippetDeformableVolumeAVBD_64.exe','--headless','--case=visual-showcase','--frames=720','--dispatcher-threads=8']
        with log.open('wb') as stream:
            subprocess.run(cmd,cwd=study.repo,env=study.env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=150)
        lines=log.read_text().splitlines()
        read=lambda tag:study.fields(lines,tag)
        gate,=read('AVBD_GATE');assert gate['status']=='PASS' and gate['completedFrames']=='720'
        tip,=read('FIVE_TIP');ped,=read('FIVE_PEDESTAL')
        assert ped['status']=='PASS' and 0<int(tip['floorBandFrame'])-int(tip['impactFrame'])<=120
        skin=read('FIVE_SKIN_STRETCH');assert len(skin)==5 and all(float(s['longestDoubledS'])<.5 for s in skin)
        impacts={r['pair']:r for r in read('FIVE_REBOUND')}
        r=impacts['prism-sphere'];assert 0<int(r['departureFrame'])-int(r['contactFrame'])<=60
        assert all(0<int(x['apexFrame'])-int(x['contactFrame'])<=60 and float(x['up'])>.1*float(x['incoming']) and float(x['recoveredVolume'])>float(x['minVolume']) for x in impacts.values())
        result.append(dict(arm=arm,rep=rep,rebound=float(r['up']),rise=float(r['rise']),depart=int(r['departureFrame']),tip=float(tip['halfSecondRad']),strain=float(gate['finalEdgeStrain']),minDet=float(gate['minDetF'])))
(target/'summary.json').write_text(json.dumps(result,indent=2))
print(result)

