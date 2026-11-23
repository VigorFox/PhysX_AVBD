from pathlib import Path
import shutil,subprocess,sys
p=Path(__file__).resolve().parent
repo=p.parents[2];base=p.parent/'color-overlap-20261123/baseline'
for arm in ['base','ready']:
    folder=p/('diag-'+arm);folder.mkdir(exist_ok=True)
    src=folder/'src/avbd';dst=folder/'candidate'
    shutil.copytree(p/'src/avbd' if arm=='ready' else repo/'physx/source/lowleveldynamics/src/avbd',src)
    shutil.copytree(base,dst)
    h=src/'DyAvbdColorTasks.h';s=h.read_text();s=s.replace('    enum { LeafRows=16 };','    PxU64 measuredWideStart=0, measuredWideSpan=0;\n    PxU32 measuredWideColors=0;\n    enum { LeafRows=16 };');h.write_bytes(s.encode())
    f=src/'DyAvbdDynamics.cpp';s=f.read_text()
    old='                if(!dispatchLocalWork())return;'
    assert s.count(old)==1
    s=s.replace(old,'''                const bool measureWide=mColorTasks && !mColorTasks->offsets.empty() &&
                    mColorTasks->wideOffsets[mStep.nextColor]!=mColorTasks->wideOffsets[mStep.nextColor+1];
                if(measureWide && !mColorTasks->measuredWideStart)
                    mColorTasks->measuredWideStart=PxTime::getCurrentCounterValue();
                if(!dispatchLocalWork())return;
                if(measureWide)
                {
                    mColorTasks->measuredWideSpan+=PxTime::getCurrentCounterValue()-mColorTasks->measuredWideStart;
                    ++mColorTasks->measuredWideColors;
                    mColorTasks->measuredWideStart=0;
                }''')
    old='        PxU32 particles=0;for(const auto& block:mBlocks)'
    assert s.count(old)==1
    s=s.replace(old,'''        const double wideMs=PxTime::getCounterFrequency().toTensOfNanos(mColorTasks?mColorTasks->measuredWideSpan:0)*1.e-5;
        std::fprintf(stderr,"[AVBD_WIDE_INTERVAL] colors=%u ms=%.7f\\n",mColorTasks?mColorTasks->measuredWideColors:0,wideMs);
        if(mColorTasks){mColorTasks->measuredWideStart=mColorTasks->measuredWideSpan=0;mColorTasks->measuredWideColors=0;}
''' .replace('%.7f\n','%.7f\\n')+old)
    f.write_bytes(s.encode())
    build=(p/'build.py').read_text().replace('repo=p.parents[2]','repo=p.parents[3]')
    (folder/'build.py').write_bytes(build.encode())
    subprocess.run([sys.executable,folder/'build.py'],check=True)

