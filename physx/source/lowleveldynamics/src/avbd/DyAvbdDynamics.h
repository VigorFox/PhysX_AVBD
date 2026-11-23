// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_DYNAMICS_H
#define DY_AVBD_DYNAMICS_H

#include "DyDynamicsBase.h"
#include "avbd/DyAvbdRows.h"
#include "avbd/DyAvbdContactMaterial.h"
#include "avbd/DyAvbdNodalShell.h"
#include "avbd/DyAvbdNodalVolume.h"
#include "avbd/DyAvbdHingeBatch.h"
#include "avbd/DyAvbdMembraneBatch.h"
#include "avbd/DyAvbdNodalContact.h"
#include "avbd/DyAvbdOgc.h"
#include "avbd/DyAvbdRowProgram.h"
#include "avbd/DyAvbdElasticFrame.h"
#include "avbd/DyAvbdBodyIndex.h"
#include "avbd/DyAvbdMaterialAssembly.h"
#include "avbd/DyAvbdContactGeometry.h"
#include "avbd/DyAvbdVbd.h"
#include "avbd/DyKernelTable.h"
#include "DyAvbdCpuSoftAccess.h"
#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxMat33.h"
#include "foundation/PxAtomic.h"
#include "PxsIslandSim.h"

namespace physx
{
class PxsContactManager;
class PxDeformableVolumeAuxData;
class PxTetrahedronMesh;
namespace Dy
{

struct Constraint;
class AvbdAdvanceTask;
class AvbdBlockSolveTask;
class AvbdColorLane;
struct AvbdCommonProgram;
struct AvbdCommonPublication;
struct AvbdBoundedResponse;
struct AvbdLocalWork;
struct AvbdLocalBlockState;
struct AvbdLocalPartial;
struct AvbdColorTasks;
struct AvbdNodalAssembly;
struct AvbdNodalTrial;
struct AvbdNodalEnergy;
struct AvbdWideNomination;
struct AvbdProgramSubset;
class FeatherstoneArticulation;
struct ArticulationJointCore;

struct AvbdSolverBody
{
	PxVec3 p;
	PxQuat q;
	PxVec3 pHat;
	PxQuat qHat;
	PxVec3 pStart;
	PxQuat qStart;
	PxVec3 pLin;
	PxQuat qLin;
	PxVec3 linVel;
	PxVec3 angVel;
	// Final velocity is staged inside the backend until every rigid and
	// heterogeneous material row has closed. Host state is published once.
	PxVec3 outputLinVel;
	PxVec3 outputAngVel;
	PxReal invMass;
	PxVec3 invInertia;
	// Frozen BDF1 metric, compiled once from qStart and dt.  The nonlinear
	// primal reuses it on every GS sweep.
	PxReal massInvH2;
	PxMat33 inertiaWorldInvH2;
	PxU8 disableGravity;
	PxU8 isKinematic;
	PxU8 hasStaticTouch;
	PxU8 hasContact;
	PxsBodyCore* core;
	PxsRigidBody* rigid;
	FeatherstoneArticulation* artic;
	PxU32 linkIndex;
};

// A candidate owns only a pose. All reference epochs and certificates are
// borrowed read-only, never copied into each line-search trial.
struct AvbdBodyPose
{
    const PxVec3 &p, &pHat, &pStart, &pLin;
    const PxQuat &q, &qHat, &qStart, &qLin;
    AvbdBodyPose(const AvbdSolverBody& body,const PxTransform* pose=NULL)
        : p(pose?pose->p:body.p),pHat(body.pHat),pStart(body.pStart),pLin(body.pLin),
          q(pose?pose->q:body.q),qHat(body.qHat),qStart(body.qStart),qLin(body.qLin) {}
};

// Substitute an indexed group, or the complete pose array when indices is NULL.
struct AvbdRigidPoseView
{
    const PxArray<AvbdSolverBody>& bodies;
    const PxU32* indices;
    const PxTransform* candidates;
    PxU32 count;
    PxU32 size() const { return bodies.size(); }
    AvbdBodyPose operator[](PxU32 i) const
    {
        if(!indices) return AvbdBodyPose(bodies[i],i<count?candidates+i:NULL);
        for(PxU32 s=0;s<count;++s) if(indices[s]==i) return AvbdBodyPose(bodies[i],candidates+s);
        return AvbdBodyPose(bodies[i]);
    }
};
struct AvbdRigidGroupRow
{
    PxU32 row,representative,supportCount;
};
struct AvbdRigidGroup
{
    PxU32 blocks[16],count,rowStart,rowCount,sharedStart,sharedCount;
    // Written only by this colored group's task; read after the task barrier.
    PxU32 attempts,unsupported,kernelSolved,accepted,products;
    PxU32 frictionAttempts,frictionAccepted,sharedContactAttempts,sharedContactAccepted;
    PxU32 materialEvaluations,materialEndpointUses,rightHandSides;
    PxU64 assemblyTicks=0,factorTicks=0,searchTicks=0,geometryTicks=0,modelTicks=0;
    PxU64 assemblyPartTicks[4]={},assemblyPartRows[4]={}; // inertia, frame, tet, scalar
};

enum AvbdBlockType
{
	eAVBD_BLOCK_PARTICLE3 = 0,
	eAVBD_BLOCK_RIGID6 = 1
};

// A Particle3 block is selected by its local elastic energy, never by the
// Host object that owns it.  Contact, OGC, dual and material rows consume the
// direct particle state below and are therefore independent of this kernel id.
enum AvbdParticleKernel
{
	eAVBD_PARTICLE_KERNEL_NONE = 0,
	eAVBD_PARTICLE_KERNEL_TET,
	eAVBD_PARTICLE_KERNEL_TRIANGLE
};

struct AvbdBlock
{
	PxU32 objectIndex; // local-energy payload index or rigid-body index
	PxU32 stateIndex;  // particle index; unused by rigid blocks
	AvbdSoftParticle* particle; // direct state for Particle3; NULL for Rigid6
	PxU8 type;
	PxU8 particleKernel;
};

struct AvbdRowSupport
{
	PxU32 block;
	PxReal weight;
	// Only rigid supports use a body-local point. Particle supports use their
	// block position directly.
	PxVec3 localPoint;
};

// Borrowed typed payloads for the common contact evaluators.
struct AvbdContactGeometryView
{
 const PxArray<AvbdContactPlaneGeometry>& planes;
 const PxArray<AvbdContactFrameGeometry>* frames;
 const KernelTable* kernels;
 const AvbdNodalContact* nodal=NULL;
 AvbdContactGeometryView(const PxArray<AvbdContactPlaneGeometry>& p)
  :planes(p),frames(NULL),kernels(NULL) {}
 AvbdContactGeometryView(const PxArray<AvbdContactPlaneGeometry>& p,
  const PxArray<AvbdContactFrameGeometry>& f,const KernelTable& k)
  :planes(p),frames(&f),kernels(&k) {}
};

static const PxU32 AVBD_MAX_JOINT_ROWS = 20;
static const PxU32 AVBD_MAX_ARTIC_ROWS = 12;

struct AvbdJointSrc
{
	Constraint* constraint;
    // Index in the typed preparation payload selected by constraint->multiBody.
    PxU32 payloadIndex;
	PxU32 bodyA;
	PxU32 bodyB;
	PxU32 nbRows;
};

// Ordinary solverPrep history only. A finite four-point material has no AL
// slots, point-lock anchors or per-axis history and allocates none of this.
struct AvbdJointState
{
	PxVec3 body0WorldOffset;
	PxVec3 pointLocalA;
	PxVec3 pointLocalB;
	PxU8 pointLock;
	PxU8 pointInitialized;
	PxU8 passiveLock;
	PxU8 integratedInitialized[AVBD_MAX_JOINT_ROWS];
	PxReal integratedC0[AVBD_MAX_JOINT_ROWS];
	PxReal lambda[AVBD_MAX_JOINT_ROWS];
	PxReal rho[AVBD_MAX_JOINT_ROWS];
};

struct AvbdArticJointSrc
{
	ArticulationJointCore* joint;
	PxU32 parentBody;
	PxU32 childBody;
	PxU32 nbRows;
	PxReal lambda[AVBD_MAX_ARTIC_ROWS];
	PxReal rho[AVBD_MAX_ARTIC_ROWS];
};

struct AvbdContactWarm
{
	const PxsContactManager* cm;
	// A numerical proof direction, never a force normal. Revalidate against
	// complete current NP supports before reuse, even for a persistent pair.
	PxVec3d proofAxis=PxVec3d(0);
	PxU32 materialKey;
	PxVec3 n;
	// Fresh narrow-phase anchors identify the manifold point.  rA/rB and the
	// paired offset are the material constraint tuple and may remain fixed
	// while a sticking point migrates inside the same NP feature.
	PxVec3 detectionRA;
	PxVec3 detectionRB;
	PxVec3 rA;
	PxVec3 rB;
	PxVec3 t0;
	PxVec3 t1;
	PxReal lambda;
	PxReal lambdaT0;
	PxReal lambdaT1;
	PxReal rho;
	PxReal rhoT0;
	PxReal rhoT1;
	PxReal contactOffset;
	PxU8 stick;
};

// Per-step velocity complementarity program.  Quantities derived from the
// frozen solve-start pose/velocity are compiled once; only accumulated
// impulses and current constrained velocities change during the GS sweeps.
struct AvbdFixedReaction
{
    PxU32 row, writeback;
    PxReal sign;
    PxVec3 arm;
};

static const PxU32 AVBD_DEFORMABLE_ITERS = 20;
static const PxU32 AVBD_DEFORMABLE_ITERS_MAX = 128;

class DeformableVolume;
class DeformableSurface;

struct AvbdSoftBoundaryEdgeTopology
{
	PxU32 v0, v1;
	PxU32 face0, face1;
};

// Rest element program and instance state for an island deformable volume.
struct AvbdIslandSoftCache
{
	AvbdNodalVolumeElement* nodalRest;
    // Certified initial-iterate chart for a volume on stationary parallel plane rows.
    PxVec3 supportNormal,previousSupportNormal;
    PxReal supportFraction,supportFriction,maximumSpeedSquared;
    bool planarSupport;
    bool hasContact;
    PxU32 publishedStamp;
    PxReal publishedDt;
	const PxTetrahedronMesh* nodalRestMesh;
	const PxDeformableVolumeAuxData* nodalRestAux;
	DeformableVolume* volume;
	AvbdSoftParticle* particles;
	PxU32 nbTets;
	PxU32 nbParticles;
	PxU32 blockStart;
	PxU32 liveStamp;
	PxU8 coupled; // 1: defer writeback; same-snapshot 3x3 + 6x6
};

// One GPU workgroup owns the complete vertex set. Each mask is a set of
// independent triangle lanes, consumed in the same order by the CPU Table.
struct AvbdMembranePacket
{
    PxU32 first=0,masks[32]={};
    PxU8 rows=0,nodes=0,passes=0,parallelFaces=0;
};
struct AvbdMembraneCache : PxUserAllocated
{
    PxArray<PxU32> offsets,faces;
    PxArray<AvbdMembranePacket> packets;
    PxArray<PxVec3> response;
    PxU32 firstRow=0,published=0;
    PxReal dt=0;
    AvbdNodalReal mu=0,lambda=0;
};

struct AvbdIslandSurfaceCache
{
    PxU64 topologyRevision;
    PxU32 packedStamp,packedSupport,packedBlock,packedMaterial,packedRows;
	AvbdNodalShellRest* nodalRest;
    bool hasContact;
    PxU32 publishedStamp;
    PxReal publishedDt;
    bool bakedFlattening;
    PxReal meanRestArea, longitudinalAreaModulus;
    AvbdMembraneCache* membrane;
	const PxTriangleMesh* restMesh;
	DeformableSurface* surface;
	AvbdSoftTri* triangles;
	AvbdSoftBending* bends;
	AvbdSoftParticle* particles;
	AvbdSoftBoundaryEdgeTopology* edges;
	PxU32 nbTriangles;
	PxU32 nbBends;
	PxU32 nbParticles;
	PxU32 nbEdges;
	PxU32 blockStart;
	PxU32 liveStamp;
};

class AvbdDynamicsContext : public DynamicsContextBase
{
	PX_NOCOPY(AvbdDynamicsContext)
public:
	AvbdDynamicsContext(PxcNpMemBlockPool* memBlockPool, Cm::FlushPool& taskPool,
		PxvSimStats& simStats, Cm::VirtualAllocatorCallback& allocator,
		PxsMaterialManager* materialManager, IG::SimpleIslandManager& islandManager,
		PxU64 contextID, PxReal maxBiasCoefficient, PxReal lengthScale,
		PxSceneFlags sceneFlags);

	virtual ~AvbdDynamicsContext();

	virtual void destroy() PX_OVERRIDE;
	virtual void update(Cm::FlushPool& flushPool, PxBaseTask* continuation,
		PxBaseTask* postPartitioningTask, PxBaseTask* lostTouchTask,
		PxvNphaseImplementationContext* nPhase, PxU32 maxPatchesPerCM,
		PxU32 maxArticulationLinks, PxReal dt, const PxVec3& gravity,
		Cm::PinnableBitMap& changedHandleMap) PX_OVERRIDE;
	virtual void mergeResults() PX_OVERRIDE;
	virtual void setSimulationController(PxsSimulationController* sc) PX_OVERRIDE
	{
		mSimulationController = sc;
	}
	virtual PxSolverType::Enum getSolverType() const PX_OVERRIDE
	{
		return PxSolverType::eAVBD;
	}
    virtual bool supportsTetrahedralConstraints() const PX_OVERRIDE { return true; }
    virtual void invalidateConstraint(const Constraint* constraint) PX_OVERRIDE;
	virtual bool usesInertialDcdContactQuery() const PX_OVERRIDE { return true; }

	const KernelTable& getKernelTable() const
	{
		return mKernels;
	}

private:
	friend class AvbdBlockSolveTask;
    friend class AvbdColorLane;
	friend class AvbdAdvanceTask;
	void compileBlockColors();
	void advancePrimal();
	void finishStep();
	void preparePrimalSweep();
	void solvePrimalBatch();
    void solveLocalBatch(PxU32 begin,PxU32 end,PxU32 stage,PxU32 slot);
    bool dispatchLocalBatch(PxU32 stage);
    bool submitLocalWave(PxU32 stage,PxU32 count,const PxU32* starts,const PxU32* stops,PxU32 firstTile=0);
    PxU32 localItemBlocks(PxU32 block) const;
    PxU32 localItemRows(PxU32 block,PxU32 count) const;
    void bakeLocalColorTasks();
    PxU32 localColorStart(PxU32 color) const;
    PxU32 localDualRows() const;
    void runColorLane(PxU32 slot);
    void executeColorLeaf(PxU32 tile);
    void assembleNodalRows(PxU32 block,PxU32 begin,PxU32 end,AvbdNodalAssembly& out);
    void finishNodalAssembly(PxU32 block,const AvbdNodalAssembly& rows);
    PxReal finishRigidAssembly(AvbdLocalBlockState& state,AvbdLocalPartial& partial);
    bool dispatchLocalWork();
    void beginNodalTrial(PxU32 block,const AvbdNodalAssembly& rows,AvbdNodalTrial& trial);
    void makeNodalCandidate(AvbdNodalTrial& trial);
    bool sampleNodalCandidate(PxU32 row,PxU32 block,const PxVec3& candidate,AvbdNodalReal& energy,
        PxU32 peer=PX_MAX_U32,const PxVec3* peerCandidate=NULL,AvbdNodalTrialVertices* vertices=NULL);
    void sampleNodalTrial(PxU32 block,PxU32 begin,PxU32 end,const AvbdNodalTrial& trial,AvbdNodalEnergy& out);
    void acceptNodalTrial(PxU32 block,const AvbdNodalAssembly& rows,AvbdNodalTrial& trial,const AvbdNodalEnergy& sample);
    void solveLocalWorkJob(PxU32 job);
    void sampleLocalBlock(const AvbdLocalBlockState& state,PxU32 begin,PxU32 end,AvbdLocalPartial& partial,bool trial);
    void directLocalBlock(AvbdLocalBlockState& state,const AvbdLocalPartial& partial);
    void finishLocalBlockTrial(AvbdLocalBlockState& state,const AvbdLocalPartial& partial);
    PxReal solveLocalRigidBlock(PxU32 body);
    bool localRowDomainAdmissible(const AvbdRowConstRef& row,const AvbdRigidPoseView& poses);
    double localBlockInertialEnergy(PxU32 body,const PxTransform& pose) const;
    void validateLocalRow(PxU32 row);
    bool localNpCandidateAdmissible(const AvbdRowConstRef& row,const AvbdRigidPoseView& poses);
    void finishLocalProfileWave();
    bool localCapability();
    void solveBoundedBatch(PxU32 begin,PxU32 end,PxU32 slot);
    bool dispatchBoundedWave();
	void solvePrimalBlock(PxU32 blockIndex);
	PxReal finishPrimalSweep(bool* coupledDomainStationary);
	AvbdContactGeometryView contactGeometry() const
	{ AvbdContactGeometryView view(mContactPlaneGeometry,mContactFrameGeometry,mKernels);view.nodal=mNodalContacts.begin();return view; }
	void initSolverBody(AvbdSolverBody& b, PxsBodyCore& core, PxsRigidBody* rigid,
		const PxVec3& gravity, PxReal dt);
	void collectArticulationLinks(PxReal dt, const PxVec3& gravity);
	void buildContactRows();
	void compileHeterogeneousBlocks();
	void compileRowProgram();
    void compileNodalSurfaceRows();
    void packSurfaceRows(PxU32 first,PxU32 end);
    void compileNodalVolumeRows();
    void compileNodalContactRows();
    PxArray<AvbdNodalContact> mNodalContacts;
    PxArray<PxU32> mNodalImpactRows; // Loaded NP rows proposing a volume-particle seed.
    PxArray<PxU32> mAttachmentRows; // First of three Cartesian rows per material point.
    void seedNodalImpact(PxU32 contactIndex);
    bool finishPrimalPreparation();
    PxVec3 warmParticleTarget(const AvbdBlock& block,PxU32 index) const;
    template<class Point> bool warmMaterialAdmissible(const AvbdRowConstRef& row,const Point& point) const;
    void prepareWarmChecks();
    void checkWarmRows(PxU32 maskIndex);
    bool dispatchRowRanges(PxU32 stage,PxU32& cursor,PxU32 end,PxU32 perTask);
    // One immutable initial guess until the row checks join. Later candidate
    // edits invalidate those results and use the ordinary incident-row recheck.
    PxArray<PxVec3> mWarmPoints;
    PxArray<PxTransform> mWarmPoses;
    PxArray<PxU32> mWarmFailureMasks;
    PxArray<PxI64> mNodalImpactVotes; // Transient normal initial-iterate proposals.
    PxArray<AvbdNodalFriction> mNodalFriction, mPreviousNodalFriction;
    PxU32 mNodalFrictionCapacity=0;
    AvbdFrictionOverflow mNodalFrictionOverflow;
    void solveLocalParticleBlock(PxU32 block);
    void solveLocalParticlePair(PxU32 first,PxU32 second);
    PxVec4 sampleNodalLoad(PxU32 block,PxU32 begin,PxU32 end,bool canPair);
    void nominateNodalPartner(PxU32 job);
    PxU32 nominationJobs() const;
    PxU32 nominationRows(PxU32 job) const;
    AvbdWideNomination* mWideNomination=NULL;
    void conditionNodalLoad(PxU32 row);
    PxU32 nodalPartner(PxU32 block) const;
    PxArray<PxU32> mNodalPartners;
    PxArray<PxVec4> mNodalIncidentLoad;
    PxArray<PxVec3> mIncidenceDisplacement;
    PxArray<PxVec3> mIncidenceRotation;
    PxArray<PxU32> mIncidenceTaskShape;
    AvbdColorTasks* mColorTasks=NULL;
    PxU32 incidenceHistoryEnd(bool includeScatter) const;
    PxU32 incidenceHistoryRows(PxU32 job) const;
    PxVec3 gatherIncidenceHistory(PxU32 block,PxU32 first,PxU32 end);
    void scatterIncidenceHistory(PxU32 block,PxU32 first,PxU32 end,const PxVec3& sum);
    void prepareIncidenceHistory(PxU32 block);
    bool compileIncidencePrimal();

    void bakeMembranePackets(AvbdIslandSurfaceCache& surface);
    void solveIncidenceFace(PxU32 row);
    void solveIncidencePrimal(PxU32 row);
    struct ContactPacket {PxU8 rows,entries;};
    PxArray<ContactPacket> mIncidencePackets;
    PxArray<const AvbdMembranePacket*> mIncidenceFacePackets;
    void solveIncidenceContact(PxU32 row);
    enum { NodalMatchStage=4, NodalLoadStage=5, IncidenceHistoryStage=6, NodalImpactStage=7, WarmCheckStage=8, SurfacePackStage=9, MaterialSolveStage=10, MaterialResidualStage=11, MaterialIslandStage=12, LocalStageCount=13 };
    AvbdNodalShellSampleT<AvbdNodalReal> sampleNodalRow(PxU32 row,PxU32 block,const PxVec3* candidate=NULL,
        PxU32 peer=PX_MAX_U32,const PxVec3* peerCandidate=NULL,AvbdNodalWorkT<AvbdNodalReal>* work=NULL,
        AvbdNodalPlaneRange* plane=NULL,AvbdNodalTrialVertices* vertices=NULL);
    PxU32 gatherNodalHinges(PxU32 block,PxU32 begin,PxU32 end,const PxVec3* candidate,AvbdHingeBatchArgs& args);
    bool isNodalHinge(PxU32 row) const;
    PxU32 gatherNodalMembranes(PxU32 block,PxU32 begin,PxU32 end,AvbdMembraneBatchArgs& args);
    bool sampleNodalCandidates(PxU32 block,PxU32 begin,PxU32 end,const PxVec3& candidate,AvbdNodalReal& energy);
	void compileVelocityComponents(PxArray<PxU32>& parents) const;
	void buildJointSources(PxU32 nbIslands, const IG::IslandId* islandIds);
	void buildArticJointSources();
	bool refreshJointRows(bool beginStep = false);
	void refreshArticJointRows();
    AvbdTetrahedronEvaluation evaluateTetrahedron(const AvbdRowConstRef& row,
        PxU32 candidateIndex=PX_MAX_U32, const PxTransform* candidate=NULL) const;
	void warmstartBlocks();
	void traceLocalMomentum(const char* stage,bool output=false) const;
	bool admitWarmstartCandidate(const AvbdProgramSubset* subset=NULL,PxReal fraction=1.f);
    bool accumulateRigidMaterialRow(PxU32 ri,const PxU32* blocks,PxU32 count,AvbdRigidAssembly* assemblies);
    bool compileCommonProgram(AvbdCommonProgram& program,const char*& reason,const AvbdRigidPoseView* candidate=NULL,
        const AvbdProgramSubset* subset=NULL,bool buildH=true,bool preserveCurvature=false);
    bool solveCommonCandidate(bool& stationary);
    bool commonCapability();
    bool beginCommonPublication(const AvbdProgramSubset& subset,bool bounded);
    bool resumeCommonPublication(bool& stationary);
    void clearCommonPublication();
    bool beginBoundedCommon();
    bool nextBoundedCommonComponent();
    void startBoundedResponse();
    bool completeBoundedCommon(bool& done,bool& stationary);
    bool solveCommonComponent(const AvbdProgramSubset& subset,bool& stationary);
    void captureCommonProgram();
    template<class Assembly> void accumulateRigidConstraintRow(PxU32 i,PxU32 ri,PxReal invH2,Assembly& assembly);
    bool linearizeRigidRow(const AvbdRowConstRef& row,PxU32 i,PxReal& C,PxVec3& jLin,PxVec3& jAng,const AvbdRigidPoseView* candidate=NULL) const;
	void solvePrimalBody(PxU32 i, PxReal invH2,
		PxVec3& outDp, PxVec3& outDth, PxVec3& gradientLinear,
		PxVec3& gradientAngular, PxReal (*metric)[6]=NULL);
	double rigidBlockEnergy(PxU32 i, const PxTransform& candidate,
		PxReal invH2) const;
    template<class PoseView>
    double rigidRowEnergy(PxU32 row,PxU32 block,const PoseView& poses,PxReal invH2) const;
    bool solvePrimalRigidGroup(PxU32 root,PxReal invH2);
    void solveLocalRigidGroup(PxU32 root,PxReal invH2);
    template<PxU32 Capacity> void solveLocalRigidGroupKernel(PxU32 root,PxReal invH2);
	PxReal solvePrimalRigidBlock(PxU32 i, PxReal invH2);
	void applyDual();
    void applyDualRange(PxU32 begin,PxU32 end);
	void prepareImpactMaterial(PxReal dt,const PxArray<PxU8>& materialBody);
	void applyHeterogeneousContactMaterial();
	void seedAnchoredJointReactions();
	void writeJointDualState();
	void writeJointConstraintWriteback();
	void prepareRigidVelocities(PxReal dt);
	void finalizeRigidBodies(PxReal dt);
	void prepareIslandDeformableVolumes(PxReal dt, const PxVec3& gravity);
	void prepareIslandDeformableSurfaces(PxReal dt, const PxVec3& gravity);
	bool anyCoupledDeformableVolume() const;
	bool anyLiveDeformableSurface() const;
	void clipRigidContactStep(PxU32 bodyIndex, PxVec3& dp, PxVec3& dth) const;
	bool rigidContactCandidateAdmissible(PxU32 bodyIndex, const PxTransform& candidate);
	bool projectNpCandidate(PxU32 bodyIndex, PxTransform& candidate) const;
	void prepareDeformableVelocities(PxReal dt);
	void finalizeDeformableBlocks(PxReal dt);
	void pruneIslandSoftCache();
	void publishIslandSoftCache(AvbdIslandSoftCache& cache);
	void clearIslandSoftCache();
	AvbdIslandSoftCache* getIslandSoftCache(DeformableVolume* volume);
	void clearIslandSurfaceCache();
	void pruneIslandSurfaceCache();
	AvbdIslandSurfaceCache* getIslandSurfaceCache(DeformableSurface* surface);
	PxU32 findBody(const PxsBodyCore* core) const;
	PxTransform bodyPose(PxU32 index, const PxsBodyCore* fallback) const;
	PxU32 solverBodyOf(PxU8 indexType, PxU32 solverBody) const;
	PxU32 contactBody(const PxsIndexedContactManager& cm, PxU32 side,const AvbdBodyIndex& indices) const;

	PxArray<AvbdSolverBody> mBodies;
	PxArray<AvbdBlock> mBlocks;
	AvbdRows mRows;
	PxArray<AvbdRowSupport> mRowSupports;
    PxArray<AvbdTetrahedronMaterial> mTetrahedronMaterials;
    AvbdElasticFrameMaterials mElasticFrameMaterials;
    PxArray<AvbdRigidGroup> mRigidGroups;
    PxArray<PxU8> mLocalTranslationClosed; // Step-local numerical chart, derived from complete row supports.
    PxArray<PxU32> mRigidGroupRoots,mRigidGroupSharedRows;
    PxArray<PxU32> mRigidGroupFactorOffsets;
    PxArray<AvbdRigidGroupRow> mRigidGroupRows;
	PxArray<AvbdContactPlaneGeometry> mContactPlaneGeometry;
	PxArray<AvbdNpDomain> mNpDomains;
	PxU32 mNpDomainUnsupported, mNpDomainPolicyExcluded, mNpDomainUnseparated;
	PxU32 mNpDomainMode;
	PxU32 mNpDomainAxisRecovered;
	PxArray<AvbdContactFrameGeometry> mContactFrameGeometry;
	PxArray<AvbdJointSrc> mJoints;
	PxArray<AvbdJointState> mJointStates;
	PxArray<AvbdArticJointSrc> mArticJoints;
	PxArray<AvbdContactWarm> mContactWarm;
	PxArray<PxU8> mMaterialBodyScratch;
	PxArray<PxU8> mMaterialIncidentScratch;
	PxArray<PxU8> mMaterialTangentBodyScratch;
	PxArray<PxU32> mMaterialComponentParentScratch;
	PxArray<PxU8> mMaterialJointComponentScratch;
	PxArray<AvbdMaterialRow> mMaterialRowsScratch;
    AvbdContactMaterial mContactMaterial;
    bool advanceContactMaterial();
    void publishContactMaterial();
    PxArray<AvbdFixedReaction> mFixedReactions;
    PxArray<PxU32> mReleasedRigidBodies;
	AvbdRowProgram mRowProgram;
	// Per heterogeneous block/contact velocity program: x is the accumulated
	// normal impulse and y/z are the two accumulated Coulomb coordinates.
	PxArray<PxVec3> mHeterogeneousMaterialImpulseScratch;
	PxArray<PxVec3> mChebyPrevPosScratch;
	PxArray<PxVec3> mChebyPrevPrevPosScratch;
	PxArray<PxQuat> mChebyPrevRotScratch;
	PxArray<PxQuat> mChebyPrevPrevRotScratch;
	PxArray<AvbdIslandSoftCache> mIslandSoftCache;
	PxArray<AvbdIslandSurfaceCache> mIslandSurfaceCache;
	// Rebuilt for each Particle3 solve; capacity survives, GS state does not.
	PxU32 mIslandSoftLiveStamp;
	PxU32 mIslandSurfaceLiveStamp;
	PxU32 mContactRowCount;
    PxU32 mNativeMaterialEnd=0;
    struct SurfaceBatch
    {
        PxU32 cacheIndex,firstRow,firstSupport,firstMaterial,count;
        AvbdNodalReal mu,lambda,bending;
        bool warm,reuseSupports,reuseMaterials;
    };
    PxU64 mSurfaceRevision=0;
    PxArray<SurfaceBatch> mSurfaceBatches;
    PxArray<AvbdRowProgram::PrefixPart> mSurfacePrefix;
    PxArray<AvbdNodalShellMaterial> mNodalShellMaterials;
    PxArray<AvbdNodalVolumeMaterial> mNodalVolumeMaterials;
    PxArray<AvbdNodalMat33> mNodalVolumeStartMetrics;
	PxU32 mNpContactRowCount;
	PxU32 mIterations;
	PxReal mLengthScale;
	PxVec3 mGravity;
	KernelTable mKernels;
	AvbdOGCParams mOgc;
	volatile PxI32 mLocalDomainFailed;
	struct StepState
	{
        bool hasGroundedJoint = false;
		PxBaseTask *continuation = NULL;
		PxU32 iteration = 0, minimumIterations = 0, minimumJointIterations = 0, iterationLimit = 0;
		PxU32 convergedContactJointSweeps = 0, convergedHeterogeneousSweeps = 0, nextColor = 0;
        bool reverseColors=false,matchNodal=false,preparingInertia=false;
        PxU32 impactCursor=0, warmMaskCursor=0;
        bool packingRows=false;
        PxU32 surfaceCursor=0;
        bool warmStarted=false;
        PxU64 warmStartTicks=0;
        PxU32 warmCheckedRows=0,warmFailedRows=0;
        PxU64 prepareCheckpoint=0;
        PxU32 nominationCursor=0,loadCursor=0,historyCursor=0;
        bool incidencePrimal=false,hasWideNodes=false;
        PxU32 membranePacketFaces=0,membranePacketColors=0,membraneParallelFaces=0;
        volatile PxI32 membranePrimalSolves=0,contactPrimalSolves=0;
		PxU32 submittedTasks = 0, workers = 0;
        volatile PxI32 nodalSolves=0,nodalPairUpdates=0;
        PxU32 nominationMaxRows=0;
        bool countNodalSolves=false;
        PxU64 boundedWaves = 0, boundedBatches = 0, boundedSweeps = 0;
        PxU64 boundedJobs=0,boundedGroups=0;
        // Opt-in, per-job scratch; the coordinator reads it only after fan-in.
        bool profile=false,localTimeline=false;
        PxU32 profileJobs=0;
        PxU64 jobTicks[32]={},groupStart=0,callerDone=0,responseStart=0;
        PxU64 jobTotal=0,jobMaximum=0,groupTicks=0,tailTicks=0;
        PxU64 compileTicks=0,setupTicks=0,responseTicks=0,admissionTicks=0;
        PxU64 backendStart=0,packTicks=0,contactPackTicks=0;
        PxU64 prepareTicks[8]={},localShape[17][3]={};
        PxU64 warmTicks=0,warmDomainTests=0,warmDomainReuses=0;
        PxU64 responseBuilds=0,responseResets=0;
        PxArray<PxU32> commonOffsets,commonBlocks;
        PxArray<PxU8> commonConstant;
        PxU32 commonComponent=0;
        bool commonClosed=true;
		PxReal chebyOmega = 1.0f;
		bool hasParticleBlocks = false, rowProgramValid = false, contactJointClosure = false;
        bool commonProgram = false, boundedCommon = false;
        PxU32 localBudget=0,localCursor=0,dualCursor=0,queryCursor=0;
        PxU64 localBlocks=0,localRows=0,localColorWork=0;
        AvbdRigidGroup retiredGroupStats={};
        PxU32 localColorBuilds=0;
        PxU64 colorTicks=0;
        PxI64 localTaskTicks[LocalStageCount]={};
        PxU64 localWaveTicks[LocalStageCount]={},localCapacityTicks[LocalStageCount]={},localWaves[LocalStageCount]={},localWaveStart=0;
        PxU64 localWorkBoundTicks[LocalStageCount]={},localLongestTicks[LocalStageCount]={},localWaveTaskTicks[128]={};
        PxU32 bakedColorPlans=0,reusedColorPlans=0,bakedColorWaves=0,splitPrimalBlocks=0,splitRowLeaves=0;
        struct HotItem {PxU64 ticks;PxU32 kind,rows;};
        HotItem hotItems[128]={};
        PxU64 criticalItemTicks[4]={},criticalItemRows[4]={},criticalItemCount[4]={};
        PxU64 localWaveTaskStarts[128]={},localStartupTicks[LocalStageCount]={},localDrainTicks[LocalStageCount]={},localLptTicks[LocalStageCount]={};
        PxU64 localWaveTaskReady[128]={},localWaveTaskThreads[128]={};
        PxU64 localReadyIdleTicks[LocalStageCount]={},localWidthIdleTicks[LocalStageCount]={},localTrace[128][4]={};
        PxU64 localTraceSpan=0,localTraceLpt=0,localTraceLoss=0;
        PxU32 localTraceStage=0,localTraceJobs=0,localTraceLanes=0;
        PxU32 localWaveTaskCount=0;
        PxU32 localWaveStage=0,localWaveJobs=0;
        PxU32 localQueuePeak=0,localPrimalMaxBlocks=0,localPrimalMaxRows=0,localDualMaxRows=0,localQueryMaxRows=0;
        PxU64 tiledBlocks=0,tiledLeaves=0,tiledJobs=0,tiledWaves=0,tiledScratchBytes=0;
        PxU32 tiledMaxRows=0;
        PxI32 localQueries=0;
        PxI64 localPointVisits=0;
        PxI32 localPrimalQueries=0,localSearchTrials=0,localSearchExhausted=0;
        PxI64 localPrimalPointVisits=0;
        PxI32 localTransportedProofs=0;
        PxI32 localReselectedProofs=0;
		bool materialPending = false;
        bool useJointChebyshev = false, requiresDomainClosure = false, heterogeneousConverged = false;
		bool contactJointConverged = false, lastCoupledDomainStationary = false, sweepPrepared = false;
	};
	StepState mStep;
	AvbdAdvanceTask* mColorContinuation=NULL;
	PxArray<AvbdBlockSolveTask*> mBlockTasks;
    PxArray<AvbdColorLane*> mColorLanes;
    AvbdCommonPublication* mCommonPublication=NULL;
    AvbdBoundedResponse* mBoundedResponse=NULL;
    AvbdLocalWork* mLocalWork=NULL;
    PxU32 mBoundedCursor=0;
	PxArray<PxU32> mColorOffsets,mColorBlocks;
	volatile PxI32 mColorCursor=0;
	PxU32 mColorEnd=0;
	PxArray<PxReal> mPrimalSteps;
	PxArray<PxU32> mPrimalIdleIteration,mPrimalChangedIteration,mPrimalIdleSkips;
	volatile PxI32 mActivePrimalTasks=0,mPeakPrimalTasks=0;
};

} // namespace Dy
} // namespace physx

#endif
