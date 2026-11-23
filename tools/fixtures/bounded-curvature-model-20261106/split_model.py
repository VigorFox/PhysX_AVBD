from pathlib import Path
base=Path('physx/source/lowleveldynamics/src/avbd')
def edit(name,old,new):
 p=base/name;b=p.read_bytes();a=old.encode();z=new.encode()
 if a not in b:a=a.replace(b'\n',b'\r\n');z=z.replace(b'\n',b'\r\n')
 assert b.count(a)==1,(name,old[:80]);p.write_bytes(b.replace(a,z))
edit('DyAvbdElasticFrame.h','root[9][12],energyPairs','root[9][12],hessian[81],energyPairs')
edit('DyAvbdElasticFrame.h','''            result.valid &= avbdElasticFramePositiveRoot(H,result.root);''','''            for(PxU32 i=0;i<81;++i)result.hessian[i]=H[i];
            result.valid &= avbdElasticFramePositiveRoot(H,result.root);''')
edit('DyAvbdCommonProgram.h','''    PxArray<double> materialEnergies;''','''    // Optional PSD metric, aligned with H after finish(). H remains the
    // actual linearized operator; the metric never changes its stationary root.
    PxArray<AvbdCommonHBlock> metricH;
    const AvbdCommonHBlock& metricBlock(PxU32 e) const {return metricH.empty()?H[e]:metricH[e];}
    PxArray<double> materialEnergies;''')
edit('DyAvbdCommonProgram.h','''        merge(H,36);merge(J,18);''','''        merge(H,36);merge(J,18);merge(metricH,36);
        if(!metricH.empty())
        {
            if(metricH.size()!=H.size())return false;
            for(PxU32 e=0;e<H.size();++e)
            {
                if(metricH[e].row!=H[e].row || metricH[e].column!=H[e].column)return false;
                for(PxU32 k=0;k<36;++k)if(!std::isfinite(metricH[e].values[k]))return false;
            }
        }''')
edit('DyAvbdCommonProgram.h','''        matrix("H",H,36);std::fprintf(file,",");matrix("J",J,18);''','''        matrix("H",H,36);std::fprintf(file,",");
        if(!metricH.empty()){matrix("metricH",metricH,36);std::fprintf(file,",");}
        matrix("J",J,18);''')
edit('DyAvbdCommonResidual.h','''        for(const auto& h:p.H)if(h.row==h.column)''','''        for(const auto& h:p.metricH.empty()?p.H:p.metricH)if(h.row==h.column)''')
edit('DyAvbdBoundedResponse.h','''if(!check && h.column==b)for(PxU32 d=0;d<36;++d)v.a[d]+=h.values[d];''','''if(!check && h.column==b)for(PxU32 d=0;d<36;++d)v.a[d]+=p.metricBlock(p.hOffsets[b]+entry).values[d];''')
edit('DyAvbdBoundedResponse.h','''avbdSolveSmallBlock(p.H[diagonal[id]].values,rhs,x,n)''','''avbdSolveSmallBlock(p.metricBlock(diagonal[id]).values,rhs,x,n)''')
edit('DyAvbdDynamics.h','''const AvbdProgramSubset* subset=NULL,bool buildH=true);''','''const AvbdProgramSubset* subset=NULL,bool buildH=true,bool preserveCurvature=false);''')
edit('DyAvbdSolve.cpp','''    const AvbdProgramSubset* subset,bool buildH)
{''','''    const AvbdProgramSubset* subset,bool buildH,bool preserveCurvature)
{''')
edit('DyAvbdSolve.cpp','''        if(buildH)p.H.pushBack(entry);''','''        if(buildH){p.H.pushBack(entry);if(preserveCurvature)p.metricH.pushBack(entry);}''')
edit('DyAvbdSolve.cpp','''            AvbdRigidAssemblyT<double> assembled[4]={};AvbdGroupFactor factors[9]={};''','''            AvbdRigidAssemblyT<double> assembled[4]={};AvbdGroupFactor factors[9]={};
            double frameH[81]={};''')
edit('DyAvbdSolve.cpp','''                const PxU32 ends[2]={row.bodyA,row.bodyB};avbdAssembleElasticFrame(r,e,ends,ids,count,assembled);''','''                const PxU32 ends[2]={row.bodyA,row.bodyB};avbdAssembleElasticFrame(r,e,ends,ids,count,assembled);
                if(preserveCurvature && buildH)for(PxU32 k=0;k<81;++k)frameH[k]=e.hessian[k];''')
edit('DyAvbdSolve.cpp','''                    p.H.pushBack(entry);
                }''','''                    if(preserveCurvature)
                    {
                        p.metricH.pushBack(entry);
                        if(rank==9)
                        {
                            for(double& v:entry.values)v=0;
                            const PxU32 ends[2]={row.bodyA,row.bodyB};
                            for(PxU32 a=0;a<2;++a)if(ends[a]==ids[s])for(PxU32 b=0;b<2;++b)if(ends[b]==ids[t])
                                for(PxU32 d=0;d<6;++d)for(PxU32 k=0;k<6;++k)
                                    if(!(assembled[s].lockedMask&(1u<<d)) && !(assembled[t].lockedMask&(1u<<k)))
                                        entry.values[6*d+k]+=(a && d<3?-1:1)*(b && k<3?-1:1)*
                                            frameH[9*(d<3?d:d+3*a)+(k<3?k:k+3*b)];
                        }
                    }
                    p.H.pushBack(entry);
                }''')
edit('DyAvbdSolve.cpp','''    if(!compileCommonProgram(p,reason,NULL,&subset))''','''    if(!compileCommonProgram(p,reason,NULL,&subset,true,bounded))''')
edit('DyAvbdSolve.cpp','''         s.linear.H=s.p.H;s.linear.base=s.rhs;s.linear.finish();}''','''         s.linear.H=s.p.metricH.empty()?s.p.H:s.p.metricH;s.linear.base=s.rhs;s.linear.finish();}''')
