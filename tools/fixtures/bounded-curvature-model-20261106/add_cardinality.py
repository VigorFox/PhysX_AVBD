from pathlib import Path
p=Path('physx/source/lowleveldynamics/src/avbd/DyAvbdContactPrep.cpp');b=p.read_bytes();out=Path(__file__).parent
(out/'DyAvbdContactPrep.cpp.before').write_bytes(b)
old=b'\tmContactRowCount = mRows.size();'
text='''
    if(mStep.commonProgram && std::getenv("PHYSX_AVBD_COMMON_STATS"))
    {
        PxU64 pairs=0,largest=0;PxU32 side0=0,side1=0;
        for(const auto& domain:mNpDomains)if(domain.rowStart!=PX_MAX_U32)
        {const PxU64 count=PxU64(domain.count[0])*domain.count[1];pairs+=count;
            if(count>largest){largest=count;side0=domain.count[0];side1=domain.count[1];}}
        std::fprintf(stderr,"[AVBD_NP_CARDINALITY] domains=%u pairs=%llu largest=%llu side0=%u side1=%u rows=%u\\n",
            mNpDomains.size(),static_cast<unsigned long long>(pairs),static_cast<unsigned long long>(largest),side0,side1,mContactRowCount);
    }
'''.replace('\n','\r\n').encode()
assert b.count(old)>=1;p.write_bytes(b.replace(old,old+text,1))
