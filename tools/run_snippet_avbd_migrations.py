#!/usr/bin/env python3
"""Run the distilled Chainmail and deformable AVBD snippet matrices."""
from __future__ import print_function
import argparse, os, subprocess, sys

import avbd_gate

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BIN=os.path.join(ROOT,"physx","bin","win.x86_64.vc143.md","checked")
MATRICES={
 "chainmail":("SnippetChainmail_64.exe",[("topology",1,0),("sag",180,0),("impact",600,0),("projectile",360,0)]),
 "volume":("SnippetDeformableVolumeAVBD_64.exe",[("ground",180,0),("rolling-sphere",120,0),("static-box",180,0),("dynamic-box",180,0),("rigid-on-soft",180,0),("soft-soft",180,0),("soft-soft-recovery",360,0),("free-fall-triplet",60,0),("rigid-squeeze",180,0),("visual-showcase",360,0),("embedded-sync",180,0),("embedded-same-count",180,0),("buffer-mutation",1,0)]),
 "surface":("SnippetDeformableSurfaceAVBD_64.exe",[("ground",180,0),("stretch",180,0),("free-fall-flat",30,0),("bending",180,0),("world-pin",1,3),("static-box",360,0),("dynamic-box",1,3),("sleep-wake",1,3),("buffer-mutation",1,0),("self-collision",1,3),("soft-soft",1,3),("surface-volume",1,3),("skinning",1,3)])
}

def known_cases(name):
 return [case for case,_,_ in MATRICES[name][1]]

def fields(line):
 out, errors = avbd_gate.parse_fields(line)
 return out if not errors else out

def run(bin_dir,name,solver,case,frames,expected,timeout):
 exe,specs=MATRICES[name];argv=[os.path.join(bin_dir,exe),"--headless","--solver="+solver,"--case="+case,"--frames="+str(frames),"--dispatcher-threads=2"]
 if name in ("volume","surface"): argv.append("--particle3") # Retain the public deformable actor contracts.
 if not os.path.isfile(argv[0]):
  print("[AVBD_MIGRATION_ERROR] snippet=%s case=%s missing executable: %s"%(name,case,argv[0]))
  return False
 try:p=subprocess.run(argv,cwd=bin_dir,text=True,capture_output=True,timeout=timeout)
 except subprocess.TimeoutExpired:print("[AVBD_MIGRATION_ERROR] snippet=%s case=%s timeout=1"%(name,case));return False
 text=p.stdout+(("\n"+p.stderr) if p.stderr else "");lines=avbd_gate.gate_lines(text)
 expected_status = "unsupported" if expected==3 else "pass"
 ok, record = avbd_gate.evaluate_gate_text(text, {
  "case": case,
  "expected_status": expected_status,
  "required_fields": ["status", "case"],
 })
 # Windows launchers can normalize non-zero application exits; the structured
 # capability record is authoritative for unsupported cells.
 ok=ok and avbd_gate.exit_matches_status(p.returncode, expected_status,
  normalized_nonzero=(os.name=="nt"))
 if expected==0 and record.get("status") in ("FAIL", "ERROR") and p.returncode==0:
  ok=False
 print("[AVBD_MIGRATION] snippet=%s case=%s exit=%s result=%s"%(name,case,p.returncode,"PASS" if ok else "FAIL"));sys.stdout.write(text)
 for error in record.get("errors") or []:
  print("[AVBD_MIGRATION_ERROR] snippet=%s case=%s error=%s"%(name,case,error))
 return ok

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--bin-dir",default=DEFAULT_BIN);ap.add_argument("--snippet",choices=("all",)+tuple(MATRICES),default="all");ap.add_argument("--solver",default="avbd");ap.add_argument("--cases",default="all");ap.add_argument("--timeout",type=float,default=300);a=ap.parse_args();names=list(MATRICES.keys()) if a.snippet=="all" else [a.snippet]
 known=[]
 for name in names:
  known.extend(known_cases(name))
 error, selected = avbd_gate.resolve_selection(a.cases, set(known), aliases={"all": None})
 if error:
  print("[AVBD_MIGRATION_ERROR] %s"%error)
  print("[AVBD_MIGRATION_COUNTS] requested=%s selected=0 executed=0 completed=0"%(len(avbd_gate.parse_case_list(a.cases))))
  print("[AVBD_MIGRATION_MATRIX] result=FAIL")
  return avbd_gate.CONFIG_ERROR
 requested = len(known) if selected is None else len(selected)
 executed=0; completed=0; ok=True
 for name in names:
  for case,frames,expected in MATRICES[name][1]:
   if selected is not None and case not in selected: continue
   executed += 1
   case_ok = run(os.path.abspath(a.bin_dir),name,a.solver,case,frames,expected,a.timeout)
   if case_ok: completed += 1
   ok = case_ok and ok
 if executed==0:
  print("[AVBD_MIGRATION_ERROR] empty execution set")
  ok=False
  code=avbd_gate.CONFIG_ERROR
 else:
  code = 0 if ok else 1
 print("[AVBD_MIGRATION_COUNTS] requested=%s selected=%s executed=%s completed=%s"%(requested, executed, executed, completed))
 print("[AVBD_MIGRATION_MATRIX] result=%s"%("PASS" if ok else "FAIL"));return code
if __name__=="__main__":sys.exit(main())
