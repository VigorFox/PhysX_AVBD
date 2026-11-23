from pathlib import Path
root=Path.cwd();base=root/'physx/source/lowleveldynamics/src/avbd'
def edit(path,old,new):
 b=path.read_bytes();old=old.encode();new=new.encode()
 if b.count(old)==0:old=old.replace(b'\n',b'\r\n');new=new.replace(b'\n',b'\r\n')
 assert b.count(old)==1,(path,old[:80]);path.write_bytes(b.replace(old,new))
p=root/'docs/REFACTOR_OUTLINE.md';s=p.read_text(encoding='utf8');start=s.index('2026-11-06 ');end=s.index('\n',start)
s=s[:start]+'2026-11-06 work evaluation: energy-slope and displacement-spectral extension trials regressed the actual five-body run and are withdrawn. Verify stable differences of quadratic material coordinates before summing their energies. Store coordinate/conjugate pairs only in transient CommonProgram evaluation; use the identical material source and the symmetric product-difference identity, with no new constitutive law, physical state, H/J, tolerance, search or iteration change. Retain only after analytic and actual-scene checks.'+s[end:];p.write_text(s,encoding='utf8')
p=root/'tools/avbd_stagnation_mechanism_probe.cpp';b=p.read_bytes();a=b.index(b'    // Over-majorized material:');z=b.index(b'    auto shifted=from;',a);p.write_bytes(b[:a]+b[z:])
edit(base/'DyAvbdCommonProgram.h','    PxArray<double> materialEnergies;','''    PxArray<double> materialEnergies;
    // Quadratic local potentials retain their coordinates until subtraction.
    // Each pair represents .5 * coordinate * conjugate, not physical history.
    struct EnergyPair {double coordinate,conjugate;};
    PxArray<EnergyPair> energyPairs;''')
edit(base/'DyAvbdCommonResidual.h','from.materialEnergies.size()!=to.materialEnergies.size())return false;','from.materialEnergies.size()!=to.materialEnergies.size() || from.energyPairs.size()!=to.energyPairs.size())return false;')
edit(base/'DyAvbdCommonResidual.h','''        for(PxU32 i=0;i<from.materialEnergies.size();++i)work+=to.materialEnergies[i]-from.materialEnergies[i];''','''        for(PxU32 i=0;i<from.materialEnergies.size();++i)work+=to.materialEnergies[i]-from.materialEnergies[i];
        for(PxU32 i=0;i<from.energyPairs.size();++i)
        {
            const auto& a=from.energyPairs[i];const auto& b=to.energyPairs[i];
            // Subtract before multiplication; a large unchanged strain must
            // not erase the work of a weak mode inside the same material.
            volatile double dx=b.coordinate-a.coordinate,dy=b.conjugate-a.conjugate;
            work+=.25*(dx*(b.conjugate+a.conjugate)+dy*(b.coordinate+a.coordinate));
        }''')
edit(base/'DyAvbdElasticFrame.h','    double energy,gradient[12],root[9][12];','    double energy,gradient[12],root[9][12],energyPairs[2*FINITE_JOINT_ROWS][2];')
edit(base/'DyAvbdElasticFrame.h','''        result.energy+=.5*(k*C*C+dh*delta*delta);''','''        result.energyPairs[2*r][0]=C;result.energyPairs[2*r][1]=k*C;
        result.energyPairs[2*r+1][0]=delta;result.energyPairs[2*r+1][1]=dh*delta;
        result.energy+=.5*(k*C*C+dh*delta*delta);''')
edit(base/'DyAvbdRigidMetric.h','''	PxVec3T<T> gradient;
	PxMat33T<T> gaussNewton;''','''	PxVec3T<T> gradient;
	PxVec3T<T> coordinate,conjugate;
	PxMat33T<T> gaussNewton;''')
edit(base/'DyAvbdRigidMetric.h','''	result.energy = T(.5)*residual.dot(weightedResidual);''','''	result.coordinate=residual;result.conjugate=weightedResidual;
	result.energy = T(.5)*residual.dot(weightedResidual);''')
edit(base/'DyAvbdSolve.cpp','''        p.materialEnergies.pushBack(.5*double(body.massInvH2)*dp.magnitudeSquared());p.materialEnergies.pushBack(rotation.energy);''','''        for(PxU32 d=0;d<3;++d)
        {
            p.energyPairs.pushBack({dp[d],double(body.massInvH2)*dp[d]});
            p.energyPairs.pushBack({rotation.coordinate[d],rotation.conjugate[d]});
        }''')
edit(base/'DyAvbdSolve.cpp','''                p.materialEnergies.pushBack(e.energy);
                const PxU32 ends[2]''','''                for(PxU32 k=0;k<2*FINITE_JOINT_ROWS;++k)p.energyPairs.pushBack({e.energyPairs[k][0],e.energyPairs[k][1]});
                const PxU32 ends[2]''')
