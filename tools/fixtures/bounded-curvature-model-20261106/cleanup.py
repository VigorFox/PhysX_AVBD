from pathlib import Path
p=Path('physx/source/lowleveldynamics/src/avbd/DyAvbdSolve.cpp');b=p.read_bytes()
def replace(old,new):
 global b
 a=old.encode();z=new.encode()
 if a not in b:a=a.replace(b'\n',b'\r\n');z=z.replace(b'\n',b'\r\n')
 assert b.count(a)==1,old[:80];b=b.replace(a,z)
replace('if(buildH){p.H.pushBack(entry);if(preserveCurvature)p.metricH.pushBack(entry);}','if(buildH)p.H.pushBack(entry);')
replace('''                if(!e.valid){reason="invalid_frame_material";return false;}
                for(PxU32 k=0;k<FINITE_JOINT_ROWS;''','''                if(!e.valid){reason="invalid_frame_material";return false;}
                if(preserveCurvature && buildH && p.metricH.empty())p.metricH=p.H;
                for(PxU32 k=0;k<FINITE_JOINT_ROWS;''')
replace('''                    if(preserveCurvature)
                    {''','''                    if(!p.metricH.empty())
                    {''')
replace('''    if(preserveCurvature)for(PxU32 e=0;e<p.H.size();++e)if(p.H[e].row==p.H[e].column)''','''    if(!p.metricH.empty())for(PxU32 e=0;e<p.H.size();++e)if(p.H[e].row==p.H[e].column)''')
replace('''                    // Full PSD material operator, including cross-group terms.
                    // No subtraction from an AL matrix and no float cancellation
                    // of diagonal material terms against their cross terms.''','''                    // The PSD operator is the numerical metric. In the bounded
                    // native-frame path retain the original signed second
                    // variation in the model, including every cross-body term.''')
replace('''        // Same complete H as the direction, with a new RHS and no force rows.
        // This certificate has the identical bounded 3/6 block executor.''','''        // The original complete PSD metric defines the unchanged engineering
        // certificate; the model operator can retain signed material curvature.
        // Both requests use the identical bounded 3/6 block executor.''')
# Diagnostics use the same retained raw material evaluation as compilation.
start=b.index(b'                PxTransformT<double> bodies[2],frames[2];avbdElasticFrameSections(m.authored,endpoints,bodies,frames);')
end=b.index(b'                const auto e=avbdEvaluateElasticFrame(m,endpoints,true);',start)
b=b[:start]+b[end:]
replace('''j?",":"",H[j]);''','''j?",":"",e.hessian[j]);''')
p.write_bytes(b)
