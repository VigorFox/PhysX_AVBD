from pathlib import Path

root = Path.cwd()
spec = root / 'docs/REFACTOR_OUTLINE.md'
s = spec.read_text(encoding='utf-8')
start = s.index('2026-11-06 外层曲率续修试验：')
end = s.index('\n', start)
s = s[:start] + '2026-11-06 外层曲率续修试验：总功导数割线初值令五体提前一帧拒绝，已撤回。改验同一单位候选的存储位移 s 与既有完整 H 修正 c，在原 motionTolerance 单位下估计正割线曲率，alpha=(s·s)/(s·(s-c))。仅正曲率且 alpha>1 时试一次扩展初值；失败恢复原单位步，仍消耗原 16 次搜索预算。扩展不外推反力，使用本轮共同求得的原可行反力；原几何、力界、真实功与实际驻点检查全部保留，使用扩展的帧必须最终闭合。预测是步内同分量 scratch，拓扑刷新清除；不改 H/J、物理历史、容差或轮次。实际五体改善并通过回归前不得保留。' + s[end:]
spec.write_text(s, encoding='utf-8')

base = root/'physx/source/lowleveldynamics/src/avbd'
p = base/'DyAvbdCommonResidual.h'
b = (root/'tools/benchmark-results/bounded-curvature-20261106/DyAvbdCommonResidual.h.before').read_bytes()
nl = b'\r\n' if b'\r\n' in b else b'\n'
addition = '''    // Spectral initial proposal in physical displacement units. The already
    // computed full-H correction is a residual, never applied to the pose.
    // Nonpositive curvature supplies no extension; original admission remains
    // responsible for every proposed candidate.
    static double secantStart(const double* delta,const double* correction,
        const double* tolerance,PxU32 size)
    {
        double length=0,curvature=0;
        for(PxU32 i=0;i<size;++i)
        {
            const double s=delta[i]/tolerance[i],y=(delta[i]-correction[i])/tolerance[i];
            length+=s*s;curvature+=s*y;
        }
        if(!(curvature>0 && length>curvature))return 1;
        const double alpha=length/curvature;return std::isfinite(alpha)?alpha:1;
    }
'''.replace('\n',nl.decode()).encode()
needle=b'struct AvbdCommonResidual'+nl+b'{'+nl
assert b.count(needle)==1
p.write_bytes(b.replace(needle,needle+addition))
p=base/'DyAvbdSolve.cpp';b=p.read_bytes()
old=b'''double work=0,slope=0,endSlope=0;
        const bool descent=AvbdCommonResidual::workDescent(p,actual,force.begin(),delta.begin(),work,slope,
            publication.bounded && fraction==1?&endSlope:NULL);'''
new=b'''double work=0,slope=0;
        const bool descent=AvbdCommonResidual::workDescent(p,actual,force.begin(),delta.begin(),work,slope);'''
if b.count(old)==0: old=old.replace(b'\n',b'\r\n');new=new.replace(b'\n',b'\r\n')
assert b.count(old)==1;b=b.replace(old,new)
old=b'''mStep.commonLineScale[mStep.commonComponent]=fraction==1 && descent && !closed?
                AvbdCommonResidual::secantStart(slope,endSlope):1;'''
new=b'''mStep.commonLineScale[mStep.commonComponent]=fraction==1 && descent && !closed && rootSmall && publication.solved?
                AvbdCommonResidual::secantStart(delta.begin(),publication.correction.begin(),motionTolerance.begin(),delta.size()):1;'''
if b.count(old)==0: old=old.replace(b'\n',b'\r\n');new=new.replace(b'\n',b'\r\n')
assert b.count(old)==1;p.write_bytes(b.replace(old,new))
