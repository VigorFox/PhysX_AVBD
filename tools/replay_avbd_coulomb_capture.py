"""Replay a frozen NCP through the production Table and independently check it.

This has no scene, pose or integration authority. The fixture is a captured
response operator, not a second implementation of the solver.
"""
import argparse
import gzip
import json
from pathlib import Path
import tempfile

import numpy as np
from run_avbd_ogc_certificate_probe import ROOT, run_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    payload = args.capture.read_bytes()
    if args.capture.suffix == ".gz":
        payload = gzip.decompress(payload)
    data = json.loads(payload)
    W = np.asarray(data["response"], dtype=float)
    if data.get("columns"):
        W = W.T
    d = np.asarray(data["demand"], dtype=float)
    c = np.asarray(data["compliance"], dtype=float)
    mu = np.asarray(data["mu"], dtype=float)
    lo = np.asarray(data["lower"], dtype=float)
    hi = np.asarray(data["upper"], dtype=float)
    assert np.all(data["coulomb"]) and np.all(lo[::3] == 0)
    assert np.all(np.isposinf(hi)) and np.all(np.isneginf(lo.reshape(-1, 3)[:, 1:])), "Unbounded Coulomb captures only"
    records = []
    with tempfile.TemporaryDirectory(prefix="avbd_ncp_replay_") as directory:
        work = Path(directory)
        def literal(values):
            return ",".join("-std::numeric_limits<double>::infinity()" if v == -np.inf else "std::numeric_limits<double>::infinity()" if v == np.inf
                            else format(float(v), ".17g") for v in values)
        for mode in ("precise", "fast"):
            for seed in ("cold", "captured"):
                arrays = dict(W=W.ravel(), demand=d, mu=mu, lower=lo, upper=hi, compliance=c,
                              force=np.zeros(len(d)) if seed == "cold" else np.asarray(data["force"]))
                source = work / "probe.cpp"
                source.write_text('''#include "avbd/DyKernelTable.h"
#include "avbd/DyAvbdCoulombNcp.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
using namespace physx;using namespace physx::Dy;
''' + "\n".join("double " + name + "[]={" + literal(values) + "};" for name, values in arrays.items()) + f'''
const int n={len(d)};
void apply(void*,const double* x,double* y)
{{for(int i=0;i<n;++i){{y[i]=0;for(int j=0;j<n;++j)y[i]+=W[n*i+j]*x[j];}}}}
int main(int argc,char** argv)
{{
    AvbdProbeFoundation foundation;const auto table=KernelTable::makeAvbdCpuScalar();
    PxArray<double> scratch(avbdCoulombNcpWorkSize(n/3));AvbdCoulombNcpResult result={{}};
    AvbdCoulombNcpArgs args={{nullptr,apply,n/3,8*(n+1),demand,mu,lower,upper,compliance,nullptr,force,scratch.begin(),&result}};
    table.launch(AVBD_COULOMB_NCP,&args);FILE* file=argc>1?std::fopen(argv[1],"wb"):nullptr;if(!file)return 2;
    std::fprintf(file,"{{\\\"converged\\\":%s,\\\"iterations\\\":%u,\\\"force\\\":[",result.converged?"true":"false",result.iterations);
    for(int i=0;i<n;++i)std::fprintf(file,"%s%.17g",i?",":"",force[i]);std::fprintf(file,"]}}");std::fclose(file);
    return result.converged?0:1;
}}
''', encoding="ascii")
                output = work / "result.json"
                code = run_probe(source, args=(str(output),), include_dirs=(ROOT / "tools",),
                                 compiler_options=("/fp:" + mode,),
                                 extra_sources=(ROOT / "physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp",))
                if code == 2 or not output.is_file():
                    raise RuntimeError(f"Table replay failed: {mode}/{seed}, exit {code}")
                result = json.loads(output.read_text())
                f = np.asarray(result.pop("force"))
                full = W + np.diag(c)
                motion = d + full @ f
                rho = np.array([1 / np.linalg.eigvalsh(full[i:i+3, i:i+3])[-1]
                                for i in range(0, len(d), 3)])
                trial = (f - np.repeat(rho, 3) * motion).reshape(-1, 3)
                projected = np.clip(trial, lo.reshape(-1, 3), hi.reshape(-1, 3))
                radius = mu * np.maximum(0, f[::3])
                length = np.linalg.norm(trial[:, 1:], axis=1)
                outside = length > radius
                projected[outside, 1:] *= (radius[outside] / length[outside])[:, None]
                scale = max(1, max(abs(f)), max(abs(d) * np.repeat(rho, 3)))
                error = float(max(abs(f - projected.ravel())) / scale)
                physical = bool(min(f[::3]) >= -1e-10 and np.all(f <= hi + 1e-10) and
                                np.all(f >= lo - 1e-10) and min(motion[::3]) >= -1e-9 and
                                max(abs(f[::3] * motion[::3])) <= 1e-7 and
                                max(np.linalg.norm(f.reshape(-1, 3)[:, 1:], axis=1) - radius) <= 1e-7)
                record = dict(mode=mode, seed=seed, **result, originalRoot=error, physical=physical,
                              passed=bool(result["converged"] and error < 1e-10 and physical))
                records.append(record)
                print(json.dumps(record), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return int(not all(r["passed"] for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
