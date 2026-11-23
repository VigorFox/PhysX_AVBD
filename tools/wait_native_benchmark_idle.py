"""Wait for a quiet machine, then run the complete paired native benchmark.

Never changes another process's priority, affinity or lifetime. A compiler
observed during measurement quarantines that batch, including otherwise fast
samples; it cannot grant performance qualification.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watch-pid', type=int, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpus', type=int, nargs='+', help='Pass an identical controlled CPU set to both variants')
    p.add_argument('--idle-cpu', type=float, default=20,
        help='Start condition only; independent of the paired cost/quality gates')
    a = p.parse_args()
    if not 0 < a.idle_cpu <= 100:
        p.error('--idle-cpu must be in (0,100]')
    if a.output.exists():
        p.error('Use a new output directory')
    watch = psutil.Process(a.watch_pid) if psutil.pid_exists(a.watch_pid) else None
    history = []
    record = a.output.parent / (a.output.name + '-environment.json')
    compiler_names = {'cl.exe', 'clang.exe', 'clang-cl.exe', 'link.exe', 'lld-link.exe'}

    def sample(phase):
        compilers = [x.info['pid'] for x in psutil.process_iter(['pid', 'name'])
                     if (x.info['name'] or '').lower() in compiler_names]
        value = dict(time=datetime.now(timezone.utc).isoformat(), phase=phase,
            cpuPercent=psutil.cpu_percent(interval=1), compilers=len(compilers),
            watchedBuildRunning=bool(watch and watch.is_running()))
        history.append(value)
        record.write_text(json.dumps(history, indent=2) + '\n')
        return value

    quiet = []
    while True:
        v = sample('waiting')
        if not v['watchedBuildRunning'] and not v['compilers']:
            quiet.append(v['cpuPercent'])
            quiet = quiet[-5:]
        else:
            quiet = []
        if len(history) % 6 == 1 or quiet:
            print('IDLE_GATE', v, 'quietSamples', len(quiet), flush=True)
        if (len(quiet) == 5 and sum(quiet) / 5 <= a.idle_cpu
                and max(quiet) - min(quiet) <= 15):
            break
        time.sleep(5)
    print('Starting six complete interleaved pairs per scene/worker', flush=True)
    command = [sys.executable, str(Path(__file__).with_name('benchmark_native_material_pair.py')),
        '--baseline', str(a.baseline), '--candidate', str(a.candidate),
        '--output', str(a.output), '--repeats', '6']
    if a.cpus:
        command += ['--cpus'] + [str(x) for x in a.cpus]
    child = subprocess.Popen(command)
    clean = True
    while child.poll() is None:
        v = sample('measuring')
        clean &= not v['compilers'] and not v['watchedBuildRunning']
        time.sleep(5)
    report_path = a.output / 'report.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        report['environmentClean'] = clean
        report['environmentRecord'] = str(record.resolve())
        report['environmentStartCondition'] = dict(cpuMeanMaximum=a.idle_cpu,
            samples=5, spreadMaximum=15, cpuWindow=quiet)
        report['numericalCostQualification'] = report.get('qualified', False)
        report['qualified'] = report.get('qualified', False) and clean
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    return child.returncode if clean else 2


if __name__ == '__main__':
    raise SystemExit(main())
