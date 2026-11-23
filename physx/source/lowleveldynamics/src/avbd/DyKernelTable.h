// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_KERNEL_TABLE_H
#define DY_KERNEL_TABLE_H

// KernelId x DeviceKind table (outline section 4). Hot path is launch(id, args)
// with the device pointer bound at context construct - no if(gpu)/if(avx2).
// CUDA AVBD_LDLT6 fills later; v1 CPU binds SCALAR only.

#include "foundation/PxAssert.h"
#include "foundation/PxMat33.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"
#include "avbd/DyAvbdContactGeometry.h"
#include "avbd/DyAvbdTetrahedron.h"
#include "avbd/DyAvbdNpDomain.h"
#include "avbd/DyAvbdNormalCone.h"
#include "avbd/DyAvbdLinearResponse.h"
#include "avbd/DyAvbdMixedKkt.h"
#include "avbd/DyAvbdCoulombNcp.h"

namespace physx
{
namespace Dy
{

enum KernelId
{
	// 3xx AVBD (outline section 4.2). Dense indices; comments keep the stable names.
	AVBD_ASSEMBLE_INERTIA = 0,	// 3xx
	AVBD_ASSEMBLE_CONTACT,		// 3xx
	AVBD_ASSEMBLE_JOINT,		// 3xx
	AVBD_LDLT6,					// 3xx - bound this slice
	AVBD_LDLT3,					// 3xx - bound this slice
	AVBD_DUAL_CONTACT,			// 3xx
	AVBD_DUAL_EQUALITY,			// 3xx


	AVBD_POSE_INTEGRATE,		// 3xx
	// 0xx Host
	PREDICT_INERTIAL,
	WRITEBACK_BODY,
	SLEEP_MARK,
	AVBD_TRIANGLE_DOMAIN,
	AVBD_CONTACT_FRAME,
	AVBD_TRIANGLE_DOMAIN_SCREEN,
	AVBD_TETRAHEDRON_MATERIAL,
    AVBD_RIGID_GROUP,
    AVBD_ELASTIC_FRAME_MATERIAL,
    AVBD_NP_DOMAIN,
    AVBD_LINEAR_RESPONSE,
    AVBD_NORMAL_CONE,
    AVBD_MIXED_KKT,
    AVBD_COULOMB_NCP,
    AVBD_SPARSE_FACTOR,
    AVBD_SPARSE_SOLVE,
    AVBD_LOCAL_PRIMAL,
    AVBD_LOCAL_DUAL,
    AVBD_LOCAL_CORRECTION,
    AVBD_BOUNDED_BATCH,
	AVBD_NONLINEAR_BLOCK,
	AVBD_NONLINEAR_GROUP,
	AVBD_LOCAL_NONLINEAR_GROUP,
	AVBD_LOCAL_REDUCTION,
	AVBD_ELASTIC_FRAME_BATCH,
	AVBD_NODAL_SHELL,
	AVBD_NODAL_VOLUME,
	AVBD_NODAL_SOLVE3,
    AVBD_MEMBRANE_PRIMAL,
    AVBD_NODAL_HINGE_BATCH,
    AVBD_MEMBRANE_BATCH,
    AVBD_CONTACT_MATERIAL_BATCH,
	KERNEL_ID_COUNT
};

enum DeviceKind
{
	DEVICE_SCALAR = 0,
	DEVICE_SSE2,
	DEVICE_AVX2_FMA,
	DEVICE_CUDA,
	DEVICE_KIND_COUNT
};

typedef void (*KernelFn)(void*);

// These operators consume the same frozen row/block program as local primal.
// Callbacks own no physics state; Table selects execution, never a new model.
typedef void (*AvbdOperatorFn)(void*,const double*,double*);
struct AvbdSparseSymmetric;
struct AvbdLocalResponse;
struct AvbdBoundedResponse;
struct AvbdNonlinearBlock;
struct AvbdNonlinearBlockArgs {AvbdNonlinearBlock* block;};
struct AvbdLocalPartial;
struct AvbdLocalReductionArgs {const AvbdLocalPartial* a;const AvbdLocalPartial* b;AvbdLocalPartial* result;bool trial;};
struct AvbdBoundedBatchArgs {AvbdBoundedResponse* response;PxU32 begin,end;bool executed;};
struct AvbdLocalCorrection;
struct AvbdLocalCorrectionArgs {AvbdLocalCorrection* correction;PxArray<double>* step;PxArray<double>* force;PxU32* iterations;bool accepted;};
struct AvbdLocalSweepArgs {AvbdLocalResponse* response;double* step;double* force;PxU32 begin,end;};
struct AvbdSparseFactorArgs {AvbdSparseSymmetric* matrix;bool* solved;};
struct AvbdSparseSolveArgs {AvbdSparseSymmetric* matrix;const double* rhs;double* solution;bool* solved;};
struct AvbdLinearResponseArgs
{
    void* context;
    AvbdOperatorFn apply,precondition;
    PxU32 size,budget;
    const double* rhs;
    double *solution,*work;
    double relativeTolerance,absoluteTolerance;
    AvbdLinearResponseResult* result;
};
struct AvbdNormalConeArgs
{
    void* context;
    AvbdOperatorFn apply;
    PxU32 size,activeBudget,krylovBudget;
    const PxU8* bilateral;
    const double *demand,*diagonal;
    double *force,*work;
    PxU8* active;
    double tolerance;
    AvbdNormalConeResult* result;
};

struct AvbdLdlt6Args
{
	PxReal (*A)[6];
	const PxReal* b;
	PxReal* x;
	bool* ok;
};

struct AvbdLdlt3Args
{
	const PxMat33* A;
	const PxVec3* b;
	PxVec3* x;
	bool* ok;
};

struct AvbdTriangleDomainArgs
{
	const AvbdTriangleDomainGeometry* geometry;
	const PxVec3d* points;
	const PxVec3d* scales;
	const double* weights;
	AvbdOgcProjectionPlane* planes;
};

struct AvbdContactFrameArgs
{
 const AvbdContactFrameGeometry* geometry;
 const PxVec3d* points;
 double offset;
 AvbdContactFrameEvaluation* evaluation;
};

struct AvbdTriangleDomainScreenArgs
{
 const AvbdTriangleDomainGeometry* geometry;
 const AvbdTrianglePointProjection* projections;
 const double* weights;
 const PxVec3d* delta;
 bool* separated;
 const AvbdTriangleProjectionRef* refs=NULL;
 PxU32 countA=3,count=6;
 const double* absoluteWeights=NULL;
 bool inputIntervals=false;
};

struct AvbdTetrahedronArgs
{
    const AvbdTetrahedronMaterial* material;
    const PxVec3d* points;
    AvbdTetrahedronEvaluation* evaluation;
};

struct KernelTable
{
	KernelFn bound[KERNEL_ID_COUNT];
	DeviceKind device;

	PX_FORCE_INLINE void launch(KernelId id, void* args) const
	{
		PX_ASSERT(PxU32(id) < KERNEL_ID_COUNT);
		PX_ASSERT(bound[id]);
		bound[id](args);
	}

	PX_FORCE_INLINE bool isBound(KernelId id) const
	{
		return bound[id] != 0;
	}

	static KernelTable makeAvbdCpuScalar();
};

bool avbdKernelLdltSelfCheck();

} // namespace Dy
} // namespace physx

#endif
