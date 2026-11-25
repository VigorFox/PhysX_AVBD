// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_ROWS_H
#define DY_AVBD_ROWS_H
#include "foundation/PxArray.h"
#include "foundation/PxVec3.h"
#include "foundation/PxAssert.h"
namespace physx { namespace Dy {
struct ArticulationJointCore;
enum AvbdRowFlag
{
	eAVBD_ROW_UNILATERAL = 1 << 0,
	eAVBD_ROW_CONTACT = 1 << 1,
	eAVBD_ROW_INACTIVE = 1 << 2,
	eAVBD_ROW_DRIVE = 1 << 3,
	// The row owns heterogeneous supports in mBlocks/mRowSupports.  This is not
	// a soft-pair policy bit: world/soft, soft/rigid and soft/soft rows all use
	// the same block-row contract.
	eAVBD_ROW_HETEROGENEOUS = 1 << 4,
	// Compatibility spelling for code that only distinguishes the rigid-only
	// evaluator from the heterogeneous evaluator.  New producers use the name
	// above so the IR does not encode a scene/contact-family owner.
	eAVBD_ROW_SOFT_PAIR = eAVBD_ROW_HETEROGENEOUS,
	eAVBD_ROW_FRESH = 1 << 5,
	eAVBD_ROW_OUTPUT_FORCE = 1 << 6,
	// Three bilateral translation rows compiled as one point-coincidence block.
	eAVBD_ROW_POINT_LOCK = 1 << 7
};

enum AvbdRowKernel
{
	eAVBD_ROW_KERNEL_CONSTRAINT = 0,
	eAVBD_ROW_KERNEL_TRIANGLE_DOMAIN,
	eAVBD_ROW_KERNEL_CONTACT_FRAME,
	eAVBD_ROW_KERNEL_FINITE_SPRING,
	eAVBD_ROW_KERNEL_TETRAHEDRON,
    eAVBD_ROW_KERNEL_ELASTIC_FRAME,
    eAVBD_ROW_KERNEL_NP_CONTACT,
    eAVBD_ROW_KERNEL_NODAL_SHELL,
    eAVBD_ROW_KERNEL_NODAL_VOLUME,
    eAVBD_ROW_KERNEL_ATTACHMENT,
    eAVBD_ROW_KERNEL_NODAL_CONTACT
};

PX_FORCE_INLINE bool avbdIsNodalMaterial(PxU32 kernel)
{return kernel==eAVBD_ROW_KERNEL_NODAL_SHELL || kernel==eAVBD_ROW_KERNEL_NODAL_VOLUME;}

// Persistent common IR. The two rigid endpoints occupy the same eight bytes
// as a two-entry support slice; heterogeneous supports use the common CSR.
struct AvbdRow
{
    PxU32 geometryIndex; // 1-based typed geometry/material payload; zero: none.
    PxU32 supportStart;
    PxU32 scalarIndex; // 1-based; finite frame/tet materials have no scalar state.
    PxU32 bodyA,bodyB;
    PxU16 supportCount;
    PxU8 kernelId,flags;
    AvbdRow& header(){return *this;}
    const AvbdRow& header() const {return *this;}
};
// Compatibility payload for ordinary scalar constraints and contacts. It is
// never allocated for finite frame/tet rows. Further contact/joint SoA splitting
// can replace this payload without changing the common program or row identity.
struct AvbdScalarRowPayload
{
	PxVec3 n;
	PxVec3 rA;
	PxVec3 rB;
	// Current DCD feature anchors are immutable geometry data for the OGC
	// region. Contact AL may retain older rA/rB anchors while sticking; sharing
	// those two meanings can collapse a valid conservative region to zero.
	PxVec3 ogcRA;
	PxVec3 ogcRB;
	PxVec3 jLinA;
	PxVec3 jAngA;
	PxVec3 jLinB;
	PxVec3 jAngB;
	PxReal C0;
	// solverPrep geometricError at the current linearization pose (pLin/qLin).
	// C0 is the frozen step-start continuation; CLin + J*dx is C_geom.
	PxReal CLin;
	// Contact geometry offset (processed separation/rest distance).  C0 is the
	// independent AL continuation anchor and may be depth-conditioned.
	PxReal contactOffset;
	// Heterogeneous (backend-DCD) rows only: OGC offset radius, so geometric
	// clearance = C + contactRadius and the particle conservative region is
	// built on the clearance.  Rigid NP rows are plain Position-AL rows with
	// C = separation - restDistance and contactRadius = 0.
	PxReal contactRadius;
	// Curvature of the geometric contact energy.  Keep this separate from rho:
	// rho belongs to augmented coordinates (joints and passive friction), while
	// OGC normal contact is an ordinary potential with no persistent normal dual.
	PxReal contactStiffness;
	PxReal rho;
	// demo3d keeps separate tangential penalties that start at PENALTY_MIN
	// and only grow while sticking inside the cone; sharing the (warm/grown)
	// normal rho rails the friction force at the cone bound during impacts.
	PxReal rhoT0;
	PxReal rhoT1;
	PxReal tangentC0;
	PxReal tangentC1;
	PxReal lambda;
	PxReal mu;
	PxReal lambdaT0;
	PxReal lambdaT1;
	PxVec3 t0;
	PxVec3 t1;
	PxReal minLambda;
	PxReal maxLambda;
	PxReal restitution;
	PxU8 articAxis;
	PxU8 articFlip;
	// solverPrep angular rows whose objective is an integrated step angle.
	// This covers velocity targets and angular couplings such as gear rows
	// without inspecting an extension concrete type.
	PxU8 integratedAngular;
	// A Px1D spring/drive is a physical force objective, not a hard-row AL
	// multiplier. Its position and this-step velocity residuals share J but
	// retain their authored coefficients.
	PxU8 accelerationDrive;
	PxReal driveStiffness;
	PxReal driveDampingOverDt;
	PxReal driveVelocityTarget;
	// Compiled material ownership: geometry normal is always Position-AL;
	// passive tangents participate only when this row's velocity program gives
	// Position-AL (or a later pose-residual owner) that source.
	PxU8 positionTangent;
	// Inbound artic 1D: geometric C/J from current parent/child poses.
	// NULL for contacts and solverPrep rows.
	const ArticulationJointCore* articJoint;
	PxReal articTarget;
	// Rigid NP prefix publishes a point impulse (N s). An attachment
    // suffix binds its own persistent force coordinate (N).
    PxReal* reactionOutput;
};

// Short-lived views never escape a synchronous solve/preparation operation.
// Const views preserve read-only access to both header and numerical payload.
template<class Header,class Scalar> class AvbdRowView
{
    Header* mHeader;
    Scalar* mPayload;
public:
    AvbdRowView(Header& h,Scalar* s) : mHeader(&h),mPayload(s) {}
    template<class H,class S> AvbdRowView(const AvbdRowView<H,S>& v) : AvbdRowView(v.header(),v.payload()) {}
    Header& header() const {return *mHeader;}
    Scalar* payload() const {return mPayload;}
    bool hasScalar() const {return mPayload!=NULL;}
    Scalar& scalar() const {PX_ASSERT(mPayload);return *mPayload;}
};
typedef AvbdRowView<AvbdRow,AvbdScalarRowPayload> AvbdRowRef;
typedef AvbdRowView<const AvbdRow,const AvbdScalarRowPayload> AvbdRowConstRef;

// Owned transient preparation packet. It is consumed by append, not retained
// beside the compact program. This also supports explicit diagnostic copying.
struct AvbdRowBuilder : AvbdRow
{
    AvbdScalarRowPayload value;
    AvbdRowBuilder() = default;
    template<class H,class S> AvbdRowBuilder(const AvbdRowView<H,S>& row)
        : AvbdRow(row.header()),value(row.hasScalar()?row.scalar():AvbdScalarRowPayload{}) {}
    AvbdScalarRowPayload& scalar() {return value;}
    const AvbdScalarRowPayload& scalar() const {return value;}
    operator AvbdRowRef() {return AvbdRowRef(*this,&value);}
    operator AvbdRowConstRef() const {return AvbdRowConstRef(*this,&value);}
};

class AvbdRows
{
    PxArray<AvbdRow> mHeaders;
    PxArray<AvbdScalarRowPayload> mScalars;
public:
    PxU32 size() const {return mHeaders.size();}
    bool empty() const {return mHeaders.empty();}
    void reserve(PxU32 count) {mHeaders.reserve(count);}
    bool appendMaterialSlots(PxU32 count) {return mHeaders.resize(size()+count);}
    void clear() {mHeaders.clear();mScalars.clear();}
    void truncate(PxU32 count)
    {
        PX_ASSERT(count<=size());
        if(!count){clear();return;}
        // Payloads are appended in row order. Truncation reclaims their suffix;
        // row identities and history matching are unchanged by the split.
        for(PxU32 i=size();i>count;--i)if(mHeaders[i-1].scalarIndex)
        {PX_ASSERT(mHeaders[i-1].scalarIndex==mScalars.size());mScalars.popBack();}
        mHeaders.forceSize_Unsafe(count);
    }
    void pushBack(const AvbdRowBuilder& source)
    {
        AvbdRow h=source;h.scalarIndex=0;
        if(h.kernelId!=eAVBD_ROW_KERNEL_ELASTIC_FRAME && h.kernelId!=eAVBD_ROW_KERNEL_TETRAHEDRON && !avbdIsNodalMaterial(h.kernelId))
        {mScalars.pushBack(source.scalar());h.scalarIndex=mScalars.size();}
        mHeaders.pushBack(h);
    }
    void pushBack(const AvbdRow& source)
    {
        PX_ASSERT(source.kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME || source.kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON || avbdIsNodalMaterial(source.kernelId));
        AvbdRow h=source;h.scalarIndex=0;mHeaders.pushBack(h);
    }
    AvbdRowRef operator[](PxU32 i)
    {auto& h=mHeaders[i];return AvbdRowRef(h,h.scalarIndex?&mScalars[h.scalarIndex-1]:NULL);}
    AvbdRowConstRef operator[](PxU32 i) const
    {const auto& h=mHeaders[i];return AvbdRowConstRef(h,h.scalarIndex?&mScalars[h.scalarIndex-1]:NULL);}
    AvbdRowRef back() {return (*this)[size()-1];}
    template<class Rows> struct Iterator
    {
        Rows* rows;PxU32 index;
        auto operator*() const -> decltype((*rows)[index]) {return (*rows)[index];}
        Iterator& operator++(){++index;return *this;}
        bool operator!=(const Iterator& other) const {return index!=other.index;}
    };
    Iterator<AvbdRows> begin(){return {this,0};}
    Iterator<AvbdRows> end(){return {this,size()};}
    Iterator<const AvbdRows> begin() const {return {this,0};}
    Iterator<const AvbdRows> end() const {return {this,size()};}
    PxU32 scalarCount() const {return mScalars.size();}
    PxU64 bytes() const {return PxU64(size())*sizeof(AvbdRow)+PxU64(scalarCount())*sizeof(AvbdScalarRowPayload);}
    PxU64 capacityBytes() const
    {return PxU64(mHeaders.capacity())*sizeof(AvbdRow)+PxU64(mScalars.capacity())*sizeof(AvbdScalarRowPayload);}
};
} }
#endif
