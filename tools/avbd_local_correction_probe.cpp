#include "avbd/DyAvbdLocalResponse.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
using namespace physx;using namespace physx::Dy;
int main()
{
    AvbdProbeFoundation foundation;double derivativeError=0,adjointError=0,operatorError=0;unsigned failed=0;
    for(unsigned c=0;c<120;++c)
    {
        const bool friction=c%3!=0;const double inf=std::numeric_limits<double>::infinity();
        const double f[3]={c%4==0?-.7:1.1,.31,-.23},u[3]={.13*(int(c%7)-3),.19*(int(c%5)-2),.17*(int(c%4)-1)};
        const double lo[3]={0,friction?-inf:-.4,friction?-inf:-.2},hi[3]={c%2?inf:.9,friction?inf:.5,friction?inf:.3};
        const double rho=.7+.01*c,mu=.2+.03*(c%9),eps=1.e-7;double A[9],B[9];
        AvbdCoulombNcp::rootDerivative(f,u,rho,mu,lo,hi,friction,A,B);
        for(unsigned side=0;side<2;++side)for(unsigned d=0;d<3;++d)
        {
            double plus[3],minus[3],fp[3],fm[3],up[3],um[3];
            for(unsigned q=0;q<3;++q){fp[q]=fm[q]=f[q];up[q]=um[q]=u[q];}
            if(side){fp[d]+=eps;fm[d]-=eps;}else{up[d]+=eps;um[d]-=eps;}
            avbdCoulombRoot(fp,up,rho,mu,lo,hi,friction,plus);avbdCoulombRoot(fm,um,rho,mu,lo,hi,friction,minus);
            for(unsigned q=0;q<3;++q)derivativeError=std::max(derivativeError,std::abs((plus[q]-minus[q])/(2*eps)-(side?B:A)[3*q+d]));
        }
    }
    AvbdCommonProgram p;p.dimensions.pushBack(3);p.dimensions.pushBack(6);p.sourceBlocks.pushBack(0);p.sourceBlocks.pushBack(1);p.base.resize(9,.13);
    for(unsigned b=0;b<2;++b){AvbdCommonHBlock h={};h.row=h.column=b;for(unsigned d=0;d<p.dimensions[b];++d)h.values[6*d+d]=2+.2*d;p.H.pushBack(h);}
    for(unsigned b=0;b<2;++b){AvbdCommonHBlock h={};h.row=b;h.column=1-b;for(unsigned d=0;d<3;++d)h.values[6*d+d]=-.1;p.H.pushBack(h);}
    for(unsigned r=0;r<3;++r)
    {
        p.sourceRows.pushBack(r);p.mu.pushBack(.4);p.coulomb.pushBack(1);
        for(unsigned q=0;q<3;++q){p.offset.pushBack(.07*(q+1));p.compliance.pushBack(.002);p.lower.pushBack(q?-std::numeric_limits<double>::infinity():0);p.upper.pushBack(std::numeric_limits<double>::infinity());}
        for(unsigned b=0;b<2;++b){AvbdCommonJBlock j={};j.row=r;j.column=b;
            for(unsigned q=0;q<3;++q)for(unsigned d=0;d<p.dimensions[b];++d)j.values[6*q+d]=(b?-1:1)*(.1*(q+1)+.02*(d+1)+.03*r);p.J.pushBack(j);}
    }
    if(!p.finish())return 2;auto table=KernelTable::makeAvbdCpuScalar();AvbdLocalResponse local(p,table);
    PxArray<PxU8> enabled(3,1);enabled[1]=0;AvbdLocalCorrection op(p,local.penalty,enabled,&local,AvbdLocalResponse::precondition);
    PxArray<double> x(9,.03),force(9,0),res(op.size),v(op.size),w(op.size),av(op.size),atw(op.size),rp(op.size),rm(op.size);
    for(unsigned r:{0u,2u}){force[3*r]=1;force[3*r+1]=.15;force[3*r+2]=-.1;}
    op.evaluate(x.begin(),force.begin(),res.begin(),true);
    for(unsigned d=0;d<op.size;++d){v[d]=.03*(int(d%7)-3);w[d]=.02*(int(d%5)-2);}
    op.apply(v.begin(),av.begin());op.transpose(w.begin(),atw.begin());
    adjointError=std::abs(AvbdCoulombNcp::dot(av.begin(),w.begin(),int(op.size))-AvbdCoulombNcp::dot(v.begin(),atw.begin(),int(op.size)));
    PxArray<double> xp=x,xm=x,fp=force,fm=force;const double eps=1.e-7;
    for(unsigned d=0;d<9;++d){xp[d]+=eps*v[d];xm[d]-=eps*v[d];}
    for(unsigned s=0;s<op.rows.size();++s)for(unsigned q=0;q<3;++q)
    {const unsigned k=3*op.rows[s]+q;fp[k]+=eps*local.penalty[k]*v[9+3*s+q];fm[k]-=eps*local.penalty[k]*v[9+3*s+q];}
    op.evaluate(xp.begin(),fp.begin(),rp.begin(),false);op.evaluate(xm.begin(),fm.begin(),rm.begin(),false);
    for(unsigned d=0;d<op.size;++d)operatorError=std::max(operatorError,std::abs((rp[d]-rm[d])/(2*eps)-av[d]));
    // Two distinct normal certificates share the exact same tangent motion.
    // Preserve their generalized force and original cones under redistribution.
    for(unsigned e=p.jOffsets[2];e<p.jOffsets[3];++e)
        for(unsigned q=1;q<3;++q)for(unsigned d=0;d<p.dimensions[p.J[e].column];++d)
            p.J[e].values[6*q+d]=p.J[p.jOffsets[0]+e-p.jOffsets[2]].values[6*q+d];
    for(unsigned r:{0u,2u})for(unsigned q=0;q<3;++q){p.compliance[3*r+q]=0;p.offset[3*r+q]=0;}
    AvbdLocalResponse mergedLocal(p,table);AvbdLocalCorrection merged(p,mergedLocal.penalty,enabled,&mergedLocal,AvbdLocalResponse::precondition);
    failed+=merged.tangentGroup[1]!=0;
    PxArray<double> original(9,0),packed,expanded,reactionBefore(9),reactionAfter(9),zero(9,0);
    original[0]=1.1;original[1]=.39;original[2]=.1;original[6]=2.2;original[7]=-.7;original[8]=-.2;
    packed=original;merged.pack(packed.begin());expanded=packed;merged.unpack(expanded.begin());
    p.transpose(original.begin(),reactionBefore.begin());p.transpose(expanded.begin(),reactionAfter.begin());double reactionError=0;
    for(unsigned d=0;d<9;++d)reactionError=std::max(reactionError,std::abs(reactionBefore[d]-reactionAfter[d]));
    failed+=reactionError>1.e-13;
    for(unsigned r:{0u,2u})
    {double root[3],u[3]={};avbdCoulombRoot(expanded.begin()+3*r,u,1,p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,true,root);
        failed+=expanded[3*r]!=original[3*r] || std::hypot(expanded[3*r+1],expanded[3*r+2])>p.mu[r]*expanded[3*r]+1.e-14;
        for(double value:root)failed+=std::abs(value)>1.e-13;}
    // Sliding aggregate capacity depends on BOTH normal coordinates.
    for(unsigned r:{0u,2u}){p.offset[3*r+1]=.17;p.offset[3*r+2]=.11;}
    merged.evaluate(zero.begin(),packed.begin(),res.begin(),true);merged.apply(v.begin(),av.begin());merged.transpose(w.begin(),atw.begin());
    adjointError=std::max(adjointError,std::abs(AvbdCoulombNcp::dot(av.begin(),w.begin(),int(merged.size))-AvbdCoulombNcp::dot(v.begin(),atw.begin(),int(merged.size))));
    xp=zero;xm=zero;fp=packed;fm=packed;
    for(unsigned d=0;d<9;++d){xp[d]+=eps*v[d];xm[d]-=eps*v[d];}
    for(unsigned s=0;s<merged.rows.size();++s)for(unsigned q=0;q<3;++q)
    {const unsigned k=3*merged.rows[s]+q;fp[k]+=eps*mergedLocal.penalty[k]*v[9+3*s+q];fm[k]-=eps*mergedLocal.penalty[k]*v[9+3*s+q];}
    merged.evaluate(xp.begin(),fp.begin(),rp.begin(),false);merged.evaluate(xm.begin(),fm.begin(),rm.begin(),false);
    for(unsigned d=0;d<merged.size;++d)operatorError=std::max(operatorError,std::abs((rp[d]-rm[d])/(2*eps)-av[d]));
    p.compliance[7]=.01;AvbdLocalCorrection compliant(p,mergedLocal.penalty,enabled,&mergedLocal,AvbdLocalResponse::precondition);failed+=compliant.tangentGroup[1]!=1;
    p.compliance[7]=0;p.lower[6]=-1;
    AvbdLocalCorrection signedNormal(p,mergedLocal.penalty,enabled,&mergedLocal,AvbdLocalResponse::precondition);failed+=signedNormal.tangentGroup[1]!=1;
    p.lower[6]=0;p.J[p.jOffsets[2]].values[6]+=1.e-12;
    AvbdLocalCorrection distinct(p,mergedLocal.penalty,enabled,&mergedLocal,AvbdLocalResponse::precondition);failed+=distinct.tangentGroup[1]!=1;
    failed+=derivativeError>1.e-7 || adjointError>1.e-11 || operatorError>1.e-7;
    // Independent half-space ray oracle: x+1>=0, y+1>=0, ray (-2,-8,0).
    AvbdCommonProgram g;g.dimensions.pushBack(3);g.sourceBlocks.pushBack(0);g.base.resize(3,0);
    for(unsigned r=0;r<2;++r)
    {
        g.sourceRows.pushBack(r);g.mu.pushBack(0);g.coulomb.pushBack(0);
        for(unsigned q=0;q<3;++q){g.offset.pushBack(q?0:1);g.lower.pushBack(0);g.upper.pushBack(q?0:std::numeric_limits<double>::infinity());g.compliance.pushBack(0);}
        AvbdCommonJBlock j={};j.row=r;j.column=0;j.values[r]=1;g.J.pushBack(j);
    }
    if(!g.finish())return 2;const double ray[3]={-2,-8,0};
    const double bound=g.linearFeasibleFraction(ray);
    failed+=bound!=.125 || 1-8*bound<0 || 1-2*bound<0 || 1-8*(bound+1.e-6)>=0;
    g.compliance[3]=.1;failed+=g.linearFeasibleFraction(ray)!=.5;
    g.upper[0]=1;failed+=g.linearFeasibleFraction(ray)!=1;
    g.upper[0]=std::numeric_limits<double>::infinity();g.offset[0]=0;failed+=g.linearFeasibleFraction(ray)!=0;
    std::printf("[LOCAL_CORRECTION] rootCases=120 rayCases=4 aggregate=1 reactionError=%.9g derivativeError=%.9g adjointError=%.9g operatorError=%.9g status=%s\n",reactionError,derivativeError,adjointError,operatorError,failed?"FAIL":"PASS");
    return failed?1:0;
}
