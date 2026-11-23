from pathlib import Path
root=Path.cwd();out=Path(__file__).parent;base=root/'physx/source/lowleveldynamics/src/avbd'
for name in ('DyAvbdSolve.cpp','DyAvbdDynamics.cpp','DyAvbdDynamics.h','DyAvbdCommonResidual.h'):
 (out/(name+'.stable-work')).write_bytes((base/name).read_bytes())
for name in ('DyAvbdDynamics.cpp','DyAvbdDynamics.h'):
 (base/name).write_bytes((out/(name+'.spectral')).read_bytes())
p=base/'DyAvbdCommonResidual.h';b=p.read_bytes();s=(out/'DyAvbdCommonResidual.h.spectral').read_bytes()
start=s.index(b'    // Spectral initial');end=s.index(b'    // Hd=',start);needle=b'    // Hd='
assert b.count(needle)==1;p.write_bytes(b.replace(needle,s[start:end]+needle))
p=base/'DyAvbdSolve.cpp';b=p.read_bytes();s=(out/'DyAvbdSolve.cpp.spectral').read_bytes()
start=b.index(b'struct AvbdCommonPublication');end=b.index(b'bool AvbdDynamicsContext::',start+30)
# All publication code precedes the following component entry point.
begin=b.index(b'struct AvbdCommonPublication');stop=b.index(b'void AvbdDynamicsContext::',b.index(b'bool AvbdDynamicsContext::resumeCommonPublication'))
bs=s.index(b'struct AvbdCommonPublication');be=s.index(b'void AvbdDynamicsContext::',s.index(b'bool AvbdDynamicsContext::resumeCommonPublication'))
p.write_bytes(b[:begin]+s[bs:be]+b[stop:])
p=root/'docs/REFACTOR_OUTLINE.md';s=p.read_text(encoding='utf8');i=s.index('\n',s.index('2026-11-06 '))
s=s[:i]+' With stable work in place, one combined spectral-initial-step experiment is permitted under the same original 16-search bound and final closure requirement; retain only if the actual five-body and regressions improve/pass.'+s[i:];p.write_text(s,encoding='utf8')
