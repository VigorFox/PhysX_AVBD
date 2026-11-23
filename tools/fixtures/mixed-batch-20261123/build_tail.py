from pathlib import Path
import shutil, subprocess, sys
p=Path(__file__).resolve().parent
prev=p.parent/'mixed-claim-20261123'

def replace(text,old,new):
    assert text.count(old)==1, old
    return text.replace(old,new)

for mode,src,runtime in [('atomic',prev/'src',prev/'candidate'),('batch',p/'batch-src',p/'batch')]:
    out=mode+'-tail'
    dest=p/(out+'-src')
    shutil.copytree(src,dest)
    shutil.copytree(runtime,p/out)
    header=dest/'avbd/DyAvbdColorTasks.h'
    header.write_text(replace(header.read_text(),'    struct ClaimTiming','''    struct JoinObservation
    {
        PxU64 offered, queued, service, gap, computeAlone, closed, leafTicks, tileTicks, hottest;
        PxU32 iteration, color, lanes, winner, longest, claims, hotLeaf, hotKind, hotRows, hotIndex;
    };
    struct LaneObservation
    {
        PxU64 leaf, tile, hottest;
        PxU32 claims, hotLeaf, hotKind, hotRows, hotIndex;
    };
    struct JoinTrace
    {
        LaneObservation lane[32];
        PxU32 winner=0;
        PxU64 closed=0;
        bool pending=false;
        PxArray<JoinObservation> waves;
    } joinTrace;
    struct ClaimTiming'''))
    f=dest/'avbd/DyAvbdDynamics.cpp'
    s=f.read_text()
    s=replace(s,'    using AvbdColorLane::AvbdColorLane;', '''    using AvbdColorLane::AvbdColorLane;
    void release() PX_OVERRIDE
    {
        if(!mContext.mStep.localTimeline){AvbdColorLane::release();return;}
        auto* join=static_cast<AvbdAdvanceTask*>(mCont);
        if(join->completeInline())
        {
            auto& trace=mContext.mColorTasks->joinTrace;
            trace.winner=mSlot;trace.closed=PxTime::getCurrentCounterValue();
            PX_SIMD_GUARD;
            mContext.advancePrimal();
        }
    }''')
    s=replace(s,'            ++plan.claims.waves;', '            plan.joinTrace.pending=true;\n            ++plan.claims.waves;')
    s=replace(s,'    const bool complete=join->completeInline();\n    if(complete)finishLocalProfileWave();','''    const bool complete=join->completeInline();
    if(complete)
    {
        if(mStep.localTimeline){plan.joinTrace.winner=0;plan.joinTrace.closed=PxTime::getCurrentCounterValue();}
        finishLocalProfileWave();
    }''')
    s=replace(s,'    auto hottest=StepState::HotItem{};', '''    auto hottest=StepState::HotItem{};
    AvbdColorTasks::LaneObservation observed={};''')
    s=replace(s,'            if(mStep.hotItems[slot].ticks>hottest.ticks)', '''            if(mStep.localTimeline)
            {
                const auto ticks=mStep.localWaveTaskTicks[slot];
                ++observed.claims;
                if(leaf)observed.leaf+=ticks;else observed.tile+=ticks;
                if(ticks>=observed.hottest)
                {
                    observed.hottest=ticks;observed.hotLeaf=leaf;
                    observed.hotKind=mStep.hotItems[slot].kind;
                    observed.hotRows=leaf?mStep.hotItems[slot].rows:plan.tiles[index].rows;
                    observed.hotIndex=index;
                }
            }
            if(mStep.hotItems[slot].ticks>hottest.ticks)''')
    s=replace(s,'            PxAtomicAdd(&plan.claims.itemWaitTicks', '            plan.joinTrace.lane[slot]=observed;\n            PxAtomicAdd(&plan.claims.itemWaitTicks')
    s=replace(s,'    // Read-only offline schedule estimate.', '''    if(mStep.localTimeline && mColorTasks && mColorTasks->joinTrace.pending)
    {
        auto& trace=mColorTasks->joinTrace;
        const PxU32 slot=trace.winner;
        const auto& observed=trace.lane[slot];
        const PxU64 start=mStep.localWaveTaskStarts[slot],finish=start+durations[slot];
        PxU64 otherEnd=0;
        for(PxU32 i=0;i<mStep.localWaveTaskCount;++i)
            if(i!=slot)otherEnd=PxMax(otherEnd,mStep.localWaveTaskStarts[i]+durations[i]);
        PX_ASSERT(mStep.localWaveTaskCount>=2 && trace.closed>=PxMax(finish,otherEnd));
        trace.waves.pushBack({mStep.localWaveTaskReady[slot]-mStep.localWaveStart,
            start-mStep.localWaveTaskReady[slot],durations[slot],
            finish>otherEnd?finish-otherEnd:0,finish>otherEnd?finish-PxMax(start,otherEnd):0,
            trace.closed-PxMax(finish,otherEnd),observed.leaf,observed.tile,observed.hottest,
            mStep.iteration+1,mStep.nextColor,mStep.localWaveTaskCount,slot,durations[slot]==longest,
            observed.claims,observed.hotLeaf,observed.hotKind,observed.hotRows,observed.hotIndex});
        trace.pending=false;
    }
    // Read-only offline schedule estimate.''')
    s=replace(s,'mColorTasks->claims={};','mColorTasks->claims={};mColorTasks->joinTrace.waves.clear();')
    s=replace(s,'                    const auto& claims=mColorTasks->claims;',r'''                    const auto& trace=mColorTasks->joinTrace;
                    std::fprintf(stderr,"[AVBD_JOIN_FRAME] waves=%u\n",trace.waves.size());
                    for(const auto& v:trace.waves)
                        std::fprintf(stderr,"[AVBD_JOIN] iteration=%u color=%u lanes=%u winner=%u longest=%u claims=%u hotLeaf=%u hotKind=%u hotRows=%u hotIndex=%u offer=%.4f queue=%.4f service=%.4f gap=%.4f alone=%.4f closed=%.4f leaf=%.4f tile=%.4f hottest=%.4f\n",
                            v.iteration,v.color,v.lanes,v.winner,v.longest,v.claims,v.hotLeaf,v.hotKind,v.hotRows,v.hotIndex,
                            v.offered*ms*1000,v.queued*ms*1000,v.service*ms*1000,v.gap*ms*1000,v.computeAlone*ms*1000,
                            v.closed*ms*1000,v.leafTicks*ms*1000,v.tileTicks*ms*1000,v.hottest*ms*1000);
                    const auto& claims=mColorTasks->claims;''')
    f.write_bytes(s.encode())
    subprocess.run([sys.executable,p/'build.py',out],check=True)

