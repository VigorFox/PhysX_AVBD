from pathlib import Path
source=Path('tools/benchmark-results/bounded-curvature-20261106/validate.py')
s=source.read_text().replace('bounded-curvature-20261106','bounded-curvature-model-20261106')
s=s.replace('jobs.extend([', '''jobs.extend([
 ('curvature',[sys.executable,'tools/run_avbd_frame_curvature_probe.py'],{}),
 ('factorization',[sys.executable,'tools/run_avbd_np_factorization_probe.py'],{}),
 ('bounded',[sys.executable,'tools/run_avbd_bounded_response_probe.py'],{}),
 ('stagnation',[sys.executable,'tools/analyze_avbd_stagnation.py','--report',str(out/'stagnation.json')],{}),
 ('frozen',[sys.executable,'tools/run_avbd_bounded_convergence_probe.py','--reports',str(out/'frozen')],{}),''')
(Path(__file__).parent/'validate.py').write_text(s)
