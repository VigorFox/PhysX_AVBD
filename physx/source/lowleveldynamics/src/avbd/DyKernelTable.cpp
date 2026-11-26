// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/DyKernelTable.h"
#include "avbd/DyAvbdPointResponse.h"
#include "avbd/DyAvbdContactMaterial.h"
#include "avbd/DyAvbdMaterialAssembly.h"
#include "avbd/DyAvbdSparseSymmetric.h"
#include "avbd/DyAvbdLocalResponse.h"
#include "avbd/DyAvbdBoundedResponse.h"
#include "avbd/DyAvbdNonlinearBlock.h"
#include "avbd/DyAvbdNonlinearGroup.h"
#include "avbd/DyAvbdLocalWork.h"
#include "avbd/DyAvbdKernelsScalar.h"
#include "avbd/DyAvbdVbd.h"
#include "avbd/DyAvbdBlockGroup.h"
#include "avbd/DyAvbdElasticFrame.h"
#include "avbd/DyAvbdFrameAssembly.h"
#include "avbd/DyAvbdNodalShell.h"
#include "avbd/DyAvbdNodalVolume.h"
#include "avbd/DyAvbdHingeBatch.h"
#include "avbd/DyAvbdMembraneBatch.h"
#include "foundation/PxMemory.h"
#include "foundation/PxMath.h"
#include <cstring>

#if defined(PX_AVBD_AVX256)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
namespace physx { namespace Dy { void avbdContactMaterialAvx256(void*); void avbdRankOneAvx256(void*); } }
#endif

using namespace physx;
using namespace Dy;

PX_COMPILE_TIME_ASSERT(DEVICE_SCALAR == 0);

namespace
{
#if defined(PX_AVBD_AVX256)
bool avbdUseAvx256()
{
    // The cap lets Checked headless exercise the baseline ISA on a newer CPU.
    const char* cap=std::getenv("PHYSX_AVBD_CPU_ISA");
    if(cap && std::strcmp(cap,"sse2")==0) return false;
#if defined(_MSC_VER)
    int cpu[4];
    __cpuidex(cpu,1,0);
    const int avxState=(1<<27)|(1<<28);
    return (cpu[2]&avxState)==avxState && (_xgetbv(0)&6)==6;
#else
    return __builtin_cpu_supports("avx");
#endif
}
#endif
void launchScalarRankOne(void* data)
{
    const auto& a=*static_cast<AvbdRankOneArgs*>(data);
    avbdAddOuter(a.matrix,a.linear,a.angular,a.scale);
}

#if PX_X64
void launchRigidRankOne(void* data)
{
    const AvbdRankOneArgs& a=*static_cast<AvbdRankOneArgs*>(data);
    const PxReal j[6]={a.linear.x,a.linear.y,a.linear.z,a.angular.x,a.angular.y,a.angular.z};
    const __m128 low=_mm_loadu_ps(j), high=_mm_loadu_ps(j+2);
    const __m128 middle=_mm_setr_ps(j[4],j[5],j[0],j[1]);
    for(PxU32 r=0;r<6;r+=2)
    {
        const __m128 even=_mm_set1_ps(a.scale*j[r]), odd=_mm_set1_ps(a.scale*j[r+1]);
        PxReal* x=a.matrix[r]; PxReal* y=a.matrix[r+1];
        _mm_storeu_ps(x,_mm_add_ps(_mm_loadu_ps(x),_mm_mul_ps(even,low)));
        const __m128 boundary=_mm_add_ps(_mm_loadu_ps(x+4),_mm_mul_ps(_mm_movelh_ps(even,odd),middle));
        _mm_storeu_ps(x+4,boundary);
        _mm_storeu_ps(y+2,_mm_add_ps(_mm_loadu_ps(y+2),_mm_mul_ps(odd,high)));
    }
}
#endif

void launchNodalShell(void* args)
{auto& a=*static_cast<AvbdNodalShellArgs*>(args);*a.output=avbdSampleNodalShell(*a.rest,a.positions,a.mu,a.lambda,a.bending,a.corner,a.weights,a.work);}
void launchNodalHingeBatch(void* args)
{AvbdHingeBatchKernel<AvbdNodalReal>::sample(*static_cast<AvbdHingeBatchArgs*>(args));}
void launchMembraneBatch(void* args)
{AvbdMembraneBatchKernel<AvbdNodalReal>::sample(*static_cast<AvbdMembraneBatchArgs*>(args));}
void launchNodalSolve3(void* args)
{
    auto& a=*static_cast<AvbdNodalSolveArgs*>(args);
    a.solved=a.cross?avbdSolveNodalPair(*a.metric,*a.cross,*a.otherMetric,*a.gradient,*a.otherGradient,*a.step,*a.otherStep,a.range):
        avbdSolveNodal3(*a.metric,*a.gradient,*a.step,a.range);
}
void launchMembranePrimal(void* args)
{
    auto& a=*static_cast<AvbdMembranePrimalArgs*>(args);
    a.solved=avbdMembraneTrialForce(a.rest,a.position,a.mobility,a.previous,a.mu,a.lambda,a.force);
}
void launchNodalVolume(void* args)
{auto& a=*static_cast<AvbdNodalVolumeArgs*>(args);*a.output=avbdSampleNodalVolume(*a.rest,a.positions,a.mu,a.lambda,a.corner,a.weights,a.startMetric,a.viscosity,a.work);}
void launchNonlinearBlock(void* args)
{static_cast<AvbdNonlinearBlockArgs*>(args)->block->solve();}
void launchLocalNonlinearGroup(void* args)
{
    auto& a=*static_cast<AvbdLocalNonlinearGroupArgs*>(args);
    if(a.loop)
        a.solved=!a.loop->failed.load(std::memory_order_relaxed) && (a.reverse?a.loop->backColumn(a.pivot):a.loop->factorColumn(a.pivot));
    else a.group->solve();
}
void launchNonlinearGroup(void* args)
{static_cast<AvbdNonlinearGroupArgs*>(args)->group->solve();}
void launchLocalReduction(void* args)
{auto& a=*static_cast<AvbdLocalReductionArgs*>(args);a.result->merge(*a.a,*a.b,a.trial);}
void launchSparseFactor(void* args)
{auto& a=*static_cast<AvbdSparseFactorArgs*>(args);*a.solved=a.matrix->factor();}
void launchSparseSolve(void* args)
{auto& a=*static_cast<AvbdSparseSolveArgs*>(args);*a.solved=a.matrix->solve(a.rhs,a.solution);}
void launchLocalPrimal(void* args)
{auto& a=*static_cast<AvbdLocalSweepArgs*>(args);a.response->primal(a.force,a.step,a.begin,a.end);}
void launchLocalDual(void* args)
{auto& a=*static_cast<AvbdLocalSweepArgs*>(args);a.response->dual(a.step,a.force,a.begin,a.end);}
void launchLocalCorrection(void* args)
{auto& a=*static_cast<AvbdLocalCorrectionArgs*>(args);a.accepted=a.correction->correct(*a.step,*a.force,*a.iterations);}
void launchBoundedBatch(void* args)
{auto& a=*static_cast<AvbdBoundedBatchArgs*>(args);a.executed=a.response->run(a.begin,a.end);}
void launchCoulombNcp(void* args)
{
    avbdSolveCoulombNcp(*static_cast<AvbdCoulombNcpArgs*>(args));
}
void launchMixedKkt(void* args)
{
    avbdSolveMixedKkt(*static_cast<AvbdMixedKktArgs*>(args));
}
struct FrozenOperator
{
    void* context;
    AvbdOperatorFn product,inverse;
    void apply(const double* x,double* y) const {product(context,x,y);}
    void precondition(const double* x,double* y) const {inverse(context,x,y);}
};
void launchLinearResponse(void* args)
{
    AvbdLinearResponseArgs& a=*static_cast<AvbdLinearResponseArgs*>(args);
    const FrozenOperator op={a.context,a.apply,a.precondition};
    *a.result=avbdLinearResponse(op,a.size,a.rhs,a.solution,a.work,a.budget,
        a.relativeTolerance,a.absoluteTolerance);
}
void launchNormalCone(void* args)
{
    AvbdNormalConeArgs& a=*static_cast<AvbdNormalConeArgs*>(args);
    const FrozenOperator op={a.context,a.apply,NULL};
    *a.result=avbdSolveNormalCone(op,a.size,a.bilateral,a.demand,a.diagonal,
        a.force,a.work,a.active,a.activeBudget,a.krylovBudget,a.tolerance);
}
void launchNpDomain(void* args)
{
    AvbdNpDomainArgs& a=*static_cast<AvbdNpDomainArgs*>(args);
    if(a.query){*a.query=avbdNpCompleteQuery(*a.domain,a.start,a.end);*a.accepted=a.query->accepted;}
    else *a.accepted=avbdNpDomainPath(*a.domain,a.start,a.end);
}

void launchRigidGroup(void* args)
{
    AvbdGroupSolveArgs* a=static_cast<AvbdGroupSolveArgs*>(args);
    *a->solved=avbdSolveBlockGroup(*a);
}
void launchTetrahedronMaterial(void* args)
{
    AvbdTetrahedronArgs* a=static_cast<AvbdTetrahedronArgs*>(args);
    *a->evaluation=avbdEvaluateTetrahedron(*a->material,a->points);
}
void launchElasticFrameMaterial(void* args)
{
    AvbdElasticFrameArgs* a=static_cast<AvbdElasticFrameArgs*>(args);
    switch(a->request)
    {
    case AvbdElasticFrameRequest::Energy:
        *static_cast<AvbdElasticFrameEnergy*>(a->output)=avbdEvaluateElasticFrameEnergy(*a->material,a->poses);break;
    case AvbdElasticFrameRequest::Contribution:
        *static_cast<AvbdElasticFrameContribution*>(a->output)=avbdEvaluateElasticFrameContribution(*a->material,a->poses,a->derivativeBody);break;
    case AvbdElasticFrameRequest::Reference:
    case AvbdElasticFrameRequest::ReferenceGradient:
        *static_cast<AvbdElasticFrameEvaluation*>(a->output)=avbdEvaluateElasticFrame(*a->material,a->poses,true,
            a->request==AvbdElasticFrameRequest::Reference);break;
    }
}
void launchFrameAssembly(void* args)
{avbdAssembleFramePacket(*static_cast<AvbdFrameAssemblyArgs*>(args));}

void launchPointResponse(void* args)
{
    auto& job = *static_cast<AvbdPointResponseArgs*>(args);
    job.solved = job.patch->evaluate(job.jacobian, job.response);
}

void launchLdlt6(void* args)
{
	AvbdLdlt6Args* a = static_cast<AvbdLdlt6Args*>(args);
	*a->ok = avbdKernelLdlt6(a->A, a->b, a->x);
}

void launchLdlt3(void* args)
{
	AvbdLdlt3Args* a = static_cast<AvbdLdlt3Args*>(args);
	*a->ok = avbdKernelLdlt3(*a->A, *a->b, *a->x);
}
void launchTriangleDomain(void* args)
{
	AvbdTriangleDomainArgs* a=static_cast<AvbdTriangleDomainArgs*>(args);
	avbdTriangleDomainPlanes(*a->geometry,a->points,a->scales,a->weights,a->planes);
}
void launchContactFrame(void* args)
{
 AvbdContactFrameArgs* a=static_cast<AvbdContactFrameArgs*>(args);
 *a->evaluation=avbdEvaluateContactFrame(*a->geometry,a->points,a->offset);
}
void launchTriangleDomainScreen(void* args)
{
 AvbdTriangleDomainScreenArgs* a=static_cast<AvbdTriangleDomainScreenArgs*>(args);
 *a->separated=avbdTriangleDomainSeparated(*a->geometry,a->projections,a->weights,*a->delta,
  a->refs,a->countA,a->count,a->absoluteWeights,a->inputIntervals);
}
} // namespace

KernelTable KernelTable::makeAvbdCpuScalar()
{
	KernelTable t;
	PxMemZero(&t, sizeof(t));
	t.device = DEVICE_SCALAR;
    t.bound[AVBD_NODAL_SHELL]=&launchNodalShell;
    t.bound[AVBD_NODAL_HINGE_BATCH]=&launchNodalHingeBatch;
    t.bound[AVBD_MEMBRANE_BATCH]=&launchMembraneBatch;
    t.bound[AVBD_CONTACT_MATERIAL_BATCH]=&avbdContactMaterialBatchImpl<contactMaterial::ScalarSpatial>;
    t.bound[AVBD_RIGID_RANK_ONE]=&launchScalarRankOne;
    t.bound[AVBD_MEMBRANE_PRIMAL]=&launchMembranePrimal;
    t.bound[AVBD_NODAL_VOLUME]=&launchNodalVolume;
    t.bound[AVBD_NODAL_SOLVE3]=&launchNodalSolve3;
    t.bound[AVBD_LINEAR_RESPONSE]=&launchLinearResponse;
    t.bound[AVBD_NORMAL_CONE]=&launchNormalCone;
    t.bound[AVBD_MIXED_KKT]=&launchMixedKkt;
    t.bound[AVBD_COULOMB_NCP]=&launchCoulombNcp;
    t.bound[AVBD_SPARSE_FACTOR]=&launchSparseFactor;
    t.bound[AVBD_SPARSE_SOLVE]=&launchSparseSolve;
    t.bound[AVBD_LOCAL_PRIMAL]=&launchLocalPrimal;
    t.bound[AVBD_LOCAL_DUAL]=&launchLocalDual;
    t.bound[AVBD_LOCAL_CORRECTION]=&launchLocalCorrection;
    t.bound[AVBD_BOUNDED_BATCH]=&launchBoundedBatch;
    t.bound[AVBD_NONLINEAR_BLOCK]=&launchNonlinearBlock;
    t.bound[AVBD_NONLINEAR_GROUP]=&launchNonlinearGroup;
    t.bound[AVBD_LOCAL_NONLINEAR_GROUP]=&launchLocalNonlinearGroup;
    t.bound[AVBD_LOCAL_REDUCTION]=&launchLocalReduction;
	t.bound[AVBD_NP_DOMAIN]=&launchNpDomain;
	t.bound[AVBD_RIGID_GROUP]=&launchRigidGroup;
	t.bound[AVBD_TETRAHEDRON_MATERIAL] = &launchTetrahedronMaterial;
    t.bound[AVBD_ELASTIC_FRAME_MATERIAL] = &launchElasticFrameMaterial;
    t.bound[AVBD_ELASTIC_FRAME_BATCH] = &launchFrameAssembly;
	t.bound[AVBD_LDLT6] = &launchLdlt6;
    t.bound[AVBD_POINT_RESPONSE] = &launchPointResponse;
	t.bound[AVBD_LDLT3] = &launchLdlt3;
	t.bound[AVBD_TRIANGLE_DOMAIN] = &launchTriangleDomain;
	t.bound[AVBD_CONTACT_FRAME] = &launchContactFrame;
	t.bound[AVBD_TRIANGLE_DOMAIN_SCREEN] = &launchTriangleDomainScreen;
	return t;
}

KernelTable KernelTable::makeAvbdCpu()
{
    KernelTable t=makeAvbdCpuScalar();
#if PX_X64
    t.device=DEVICE_SSE2;
    t.bound[AVBD_CONTACT_MATERIAL_BATCH]=&avbdContactMaterialBatch;
    t.bound[AVBD_RIGID_RANK_ONE]=&launchRigidRankOne;
#if defined(PX_AVBD_AVX256)
    const bool wide=avbdUseAvx256();
    if(std::getenv("PHYSX_AVBD_MATERIAL_STATS"))
        std::fprintf(stderr,"[AVBD_CPU_KERNELS] rigidRankOne=%s contactMaterial=%s\n",wide?"avx256":"sse2",wide?"avx256":"sse2");
    if(wide)
    {
        t.device=DEVICE_AVX256;
        t.bound[AVBD_CONTACT_MATERIAL_BATCH]=&avbdContactMaterialAvx256;
        t.bound[AVBD_RIGID_RANK_ONE]=&avbdRankOneAvx256;
    }
#endif
#endif
    return t;
}

bool Dy::avbdKernelLdltSelfCheck()
{
	PxReal A[6][6];
	PxReal b[6];
	PxReal x[6];
	for(PxU32 i = 0; i < 6; ++i)
	{
		b[i] = PxReal(i + 1);
		x[i] = 0.0f;
		for(PxU32 j = 0; j < 6; ++j)
			A[i][j] = (i == j) ? 1.0f : 0.0f;
	}
	if(!avbdKernelLdlt6(A, b, x))
		return false;
	for(PxU32 i = 0; i < 6; ++i)
	{
		if(PxAbs(x[i] - b[i]) > 1.0e-4f)
			return false;
	}

	const PxMat33 A3 = PxMat33::createDiagonal(PxVec3(2.0f, 3.0f, 4.0f));
	const PxVec3 b3(2.0f, 6.0f, 12.0f);
	PxVec3 x3(0.0f);
	if(!avbdKernelLdlt3(A3, b3, x3))
		return false;
	return PxAbs(x3.x - 1.0f) < 1.0e-4f && PxAbs(x3.y - 2.0f) < 1.0e-4f &&
		PxAbs(x3.z - 3.0f) < 1.0e-4f;
}
