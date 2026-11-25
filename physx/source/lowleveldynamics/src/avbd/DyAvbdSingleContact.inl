// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Included after PatchVelocity. Both CPU widths use the same local contact law.
namespace contactMaterial
{
struct SinglePointImpulse
{
    PxF64 n, u, v;
};

template<class V>
static bool solveSinglePoint(AvbdContactMaterial& program, const AvbdMaterialPatch& patch, AvbdMaterialRow* states)
{
    const AvbdMaterialPoint& point=program.points[patch.begin];
    AvbdMaterialRow& row=states[point.state];
    if(!row.solveNormal || !row.solveTangent) return false;

    PatchVelocity<V> velocity(program,patch);
    const auto mobility=[&](PxU32 i,PxU32 j)
    {
        PxF64 value=0;
        for(PxU32 k=0; k<3; ++k)
            for(PxU32 lane=0; lane<4; ++lane)
                value+=PxF64(point.axes[i].jacobian[k][lane])*point.axes[j].response[k][lane];
        return value;
    };
    const PxF64 nn=mobility(0,0), nu=mobility(0,1), nv=mobility(0,2);
    const PxF64 uu=mobility(1,1), vv=mobility(2,2), uv=mobility(1,2);
    if(!(nn>0 && uu>0 && vv>0)) return false;

    // Remove this point's current impulse to obtain its free closing/slip speed.
    const PxF64 n=row.normalImpulse, u=row.tangentImpulse0, v=row.tangentImpulse1;
    const PxF64 bn=(velocity.speed(point.axes[0])-row.normalTarget)-nn*n-nu*u-nv*v;
    const PxF64 bu=velocity.speed(point.axes[1])-nu*n-uu*u-uv*v;
    const PxF64 bv=velocity.speed(point.axes[2])-nv*n-uv*u-vv*v;
    const PxF64 upper=row.maxNormalImpulse, load=row.positionNormalImpulse, mu=point.friction;
    const PxF64 freeNormal=PxClamp(-bn/nn,0.,upper);
    SinglePointImpulse solution={freeNormal,0.,0.};
    if(mu*(load+freeNormal)>0)
    {
        // In slip v_t = -eta*j_t. Use x = scale/(scale+eta) on [0,1].
        // At x=1 this is the full sticking solve. For a bounded normal,
        // eliminate that coordinate at its active bound and solve the same 2x2.
        const PxF64 scale=(uu+vv)*.5, un=nu/nn, vn=nv/nn;
        const auto evaluate=[&](PxF64 x, PxF64& gap, PxF64& slope)
        {
            bool interior=true;
            PxF64 baseN=-bn/nn;
            for(PxU32 pass=0; pass<2; ++pass)
            {
                const PxF64 a=(1.-x)*scale+x*(interior ? uu-un*nu : uu);
                const PxF64 d=(1.-x)*scale+x*(interior ? vv-vn*nv : vv);
                const PxF64 c=x*(interior ? uv-un*nv : uv);
                const PxF64 det=a*d-c*c;
                if(!(det>0)) return false;
                const PxF64 rhsU=interior ? bu-un*bn : bu+nu*baseN;
                const PxF64 rhsV=interior ? bv-vn*bn : bv+nv*baseN;
                const PxF64 zu=(-d*rhsU+c*rhsV)/det, zv=(c*rhsU-a*rhsV)/det;
                solution.u=x*zu; solution.v=x*zv;
                solution.n=interior ? baseN-un*solution.u-vn*solution.v : baseN;
                if(interior && (solution.n<0 || solution.n>upper))
                {
                    baseN=PxClamp(solution.n,0.,upper);
                    interior=false;
                    continue;
                }
                const PxF64 length=PxSqrt(solution.u*solution.u+solution.v*solution.v);
                const PxF64 du=scale*(d*zu-c*zv)/det, dv=scale*(-c*zu+a*zv)/det;
                const PxF64 dn=interior ? -un*du-vn*dv : 0.;
                gap=length-mu*(load+solution.n);
                slope=(length>0 ? (solution.u*du+solution.v*dv)/length : PxSqrt(du*du+dv*dv))-mu*dn;
                return true;
            }
            return false;
        };

        PxF64 gap,slope;
        if(!evaluate(1.,gap,slope)) return false;
        if(gap>0)
        {
            const PxF64 initialRadius=mu*(load+freeNormal);
            PxF64 x=initialRadius/(initialRadius+gap), lo=0, hi=1;
            for(PxU32 step=0; step<5; ++step)
            {
                if(!evaluate(x,gap,slope)) return false;
                if(PxAbs(gap)<=1.e-7*mu*(load+solution.n)) break;
                if(gap>0) hi=x; else lo=x;
                const PxF64 next=slope!=0 ? x-gap/slope : lo;
                x=next>lo && next<hi ? next : (lo+hi)*.5;
            }
            const PxF64 length=PxSqrt(solution.u*solution.u+solution.v*solution.v);
            if(length>0)
            {
                const PxF64 ratio=mu*(load+solution.n)/length;
                solution.u*=ratio; solution.v*=ratio;
            }
        }
    }
    const PxReal newN=PxReal(solution.n), newU=PxReal(solution.u), newV=PxReal(solution.v);
    if(!PxIsFinite(newN) || !PxIsFinite(newU) || !PxIsFinite(newV)) return false;
    velocity.scatter(point.axes[0],newN-row.normalImpulse);
    velocity.scatter(point.axes[1],newU-row.tangentImpulse0);
    velocity.scatter(point.axes[2],newV-row.tangentImpulse1);
    velocity.store();
    row.normalImpulse=newN;
    row.normalCapacity=row.positionNormalImpulse+newN;
    row.tangentImpulse0=newU;
    row.tangentImpulse1=newV;
    return true;
}
}
