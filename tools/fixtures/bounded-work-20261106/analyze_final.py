from pathlib import Path
p=Path(__file__).with_name('analyze.py')
s=p.read_text()
s=s.replace("json.loads(gzip.decompress((fixture/'terminal-candidate.json.gz').read_bytes()))", "json.loads((out/'five-final-candidate.json').read_text())")
s=s.replace("json.loads(gzip.decompress((fixture/'terminal-candidate.json.trial.json.gz').read_bytes()))", "json.loads((out/'five-final-candidate.json.trial.json').read_text())")
s=s.replace('decomposition.json','decomposition-final.json')
exec(compile(s,str(p),'exec'))
