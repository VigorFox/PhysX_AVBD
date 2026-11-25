#include "SnippetChainmailHeadless.h"
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>

using namespace physx;
namespace {
static const PxU32 W=30, H=30, BODY_COUNT=W*H, JOINT_COUNT=(W-1)*H+W*(H-1);
static const PxReal SPACING=0.65f, NET_Y=35.0f;
static int gNetTag, gBallTag;

struct Metrics {
	double stepMs=0,maxStepMs=0,observerMs=0,contactMs=0;PxU32 contactFrames=0;
	PxU32 completed, fetchFailures, nonFinite, contacts, contactPoints;
	PxU32 firstContactFrame;
	PxReal maxJointError, maxCornerDrift, centerMinY, preContactCenterMinY, ballMinY, maxSpeed, maxNetSpeed, maxBallSpeed, maxPostContactBallVy, reboundRise, finalCenterY, finalBallY, maxBallPlanarSeparation;
	Metrics():completed(0),fetchFailures(0),nonFinite(0),contacts(0),contactPoints(0),
		firstContactFrame(PX_MAX_U32),maxJointError(0),maxCornerDrift(0),centerMinY(PX_MAX_F32),preContactCenterMinY(PX_MAX_F32),ballMinY(PX_MAX_F32),maxSpeed(0),maxNetSpeed(0),maxBallSpeed(0),maxPostContactBallVy(-PX_MAX_F32),reboundRise(0),finalCenterY(0),finalBallY(0),maxBallPlanarSeparation(0){}
};

struct Callback : PxSimulationEventCallback {
	struct Witness { const PxRigidBody* a; const PxRigidBody* b; PxVec3 point, normal; };
	std::vector<Witness> witnesses;
	bool observeSlip = false;
	Metrics* metrics; Callback():metrics(NULL){}
	void onConstraintBreak(PxConstraintInfo*,PxU32){} void onWake(PxActor**,PxU32){}
	void onSleep(PxActor**,PxU32){} void onTrigger(PxTriggerPair*,PxU32){}
	void onAdvance(const PxRigidBody*const*,const PxTransform*,const PxU32){}
	void onContact(const PxContactPairHeader& h,const PxContactPair* p,PxU32 n){
		if(observeSlip && !(h.flags & (PxContactPairHeaderFlag::eREMOVED_ACTOR_0 | PxContactPairHeaderFlag::eREMOVED_ACTOR_1)) &&
			(h.actors[0]->userData==&gBallTag || h.actors[1]->userData==&gBallTag)){
			const auto* a=h.actors[0]->is<PxRigidBody>(); const auto* b=h.actors[1]->is<PxRigidBody>();
			for(PxU32 i=0;i<n;++i){
				PxContactPairPoint points[64];
				const PxU32 count=p[i].extractContacts(points,64);
				for(PxU32 j=0;j<count;++j)
					if(points[j].separation<=0)
						witnesses.push_back({a,b,points[j].position,points[j].normal});
			}
		}
		if(!metrics) return; const bool hit=(h.actors[0]->userData==&gBallTag&&h.actors[1]->userData==&gNetTag)||(h.actors[1]->userData==&gBallTag&&h.actors[0]->userData==&gNetTag);
		if(hit) for(PxU32 i=0;i<n;++i){ if(metrics->firstContactFrame==PX_MAX_U32)metrics->firstContactFrame=metrics->completed; ++metrics->contacts; metrics->contactPoints+=p[i].contactCount; }
	}
};

static PxFilterFlags filter(PxFilterObjectAttributes,PxFilterData a,PxFilterObjectAttributes,PxFilterData b,PxPairFlags& f,const void*,PxU32){
	if(a.word0==1&&b.word0==1) return PxFilterFlag::eSUPPRESS;
	f=PxPairFlag::eCONTACT_DEFAULT|PxPairFlag::eNOTIFY_TOUCH_FOUND|PxPairFlag::eNOTIFY_TOUCH_PERSISTS|PxPairFlag::eNOTIFY_CONTACT_POINTS; return PxFilterFlags();
}

struct Fixture {
	Snippets::TrackingErrorCallback errors; PxDefaultAllocator allocator;
	PxFoundation* foundation; PxPhysics* physics; PxDefaultCpuDispatcher* dispatcher;
	PxScene* scene; PxMaterial* material; PxRigidDynamic* nodes[BODY_COUNT]; PxRigidDynamic* ball;
	std::vector<PxJoint*> joints; Callback callback; Metrics metrics; bool extensions;
	Fixture():foundation(NULL),physics(NULL),dispatcher(NULL),scene(NULL),material(NULL),ball(NULL),extensions(false){ std::memset(nodes,0,sizeof(nodes)); callback.metrics=&metrics; }
	~Fixture(){ cleanup(); }
	void cleanup(){ for(size_t i=0;i<joints.size();++i) if(joints[i]) joints[i]->release(); joints.clear(); PX_RELEASE(scene); PX_RELEASE(material); PX_RELEASE(dispatcher); if(extensions){PxCloseExtensions();extensions=false;} PX_RELEASE(physics); PX_RELEASE(foundation); }
	void tagShapes(PxRigidDynamic& body){ PxShape* s[3]; PxU32 n=body.getShapes(s,3); PxFilterData d; d.word0=1; for(PxU32 i=0;i<n;++i)s[i]->setSimulationFilterData(d); }
	void strut(PxRigidDynamic& body,const PxVec3& dir){ const PxReal r=0.06f; PxShape* s=physics->createShape(PxCapsuleGeometry(r,SPACING*0.5f-r),*material,true); const PxQuat q=dir.x>0.5f?PxQuat(PxIdentity):PxQuat(-PxHalfPi,PxVec3(0,1,0)); s->setLocalPose(PxTransform(dir*(SPACING*0.5f),q)); body.attachShape(*s); s->release(); }
	bool init(const Snippets::HeadlessOptions& o,bool withBall,bool projectile){
		foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors); if(!foundation)return false;
		physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),true,NULL); if(!physics)return false;
		extensions=PxInitExtensions(*physics,NULL); dispatcher=PxDefaultCpuDispatcherCreate(o.dispatcherThreads);
		PxSceneDesc d(physics->getTolerancesScale()); d.gravity=PxVec3(0,-9.81f,0); d.solverType=o.solverType; d.cpuDispatcher=dispatcher; d.filterShader=filter; d.simulationEventCallback=&callback;
		scene=physics->createScene(d); material=physics->createMaterial(0.5f,0.5f,0.2f); if(!scene||!material)return false;
		PxRigidStatic* ground=PxCreatePlane(*physics,PxPlane(0,1,0,0),*material); scene->addActor(*ground);
		const PxReal x0=-SPACING*PxReal(W-1)*0.5f,z0=-SPACING*PxReal(H-1)*0.5f;
		for(PxU32 r=0;r<H;++r)for(PxU32 c=0;c<W;++c){const PxU32 i=r*W+c; const PxVec3 p(x0+c*SPACING,NET_Y,z0+r*SPACING); nodes[i]=PxCreateDynamic(*physics,PxTransform(p),PxSphereGeometry(0.12f),*material,3.0f); nodes[i]->setAngularDamping(0.5f); if(c+1<W)strut(*nodes[i],PxVec3(1,0,0)); if(r+1<H)strut(*nodes[i],PxVec3(0,0,1)); tagShapes(*nodes[i]); nodes[i]->userData=&gNetTag; const bool corner=(r==0||r+1==H)&&(c==0||c+1==W);if(corner)nodes[i]->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true);scene->addActor(*nodes[i]);if(corner)nodes[i]->setKinematicTarget(PxTransform(p));}
		const PxVec3 hx(SPACING*0.5f,0,0),hz(0,0,SPACING*0.5f);
		for(PxU32 r=0;r<H;++r)for(PxU32 c=0;c<W;++c){const PxU32 i=r*W+c; if(c+1<W)joints.push_back(PxSphericalJointCreate(*physics,nodes[i],PxTransform(hx),nodes[i+1],PxTransform(-hx))); if(r+1<H)joints.push_back(PxSphericalJointCreate(*physics,nodes[i],PxTransform(hz),nodes[i+W],PxTransform(-hz)));}
		if(withBall){ball=PxCreateDynamic(*physics,PxTransform(PxVec3(0,projectile?NET_Y+15.0f:70.0f,0)),PxSphereGeometry(projectile?1.0f:2.0f),*material,projectile?100.0f:300.0f); ball->setAngularDamping(0.5f); if(projectile)ball->setLinearVelocity(PxVec3(0,-50,0)); ball->userData=&gBallTag; scene->addActor(*ball);}
		return extensions&&joints.size()==JOINT_COUNT;
	}
	void sample(){
		PxReal frameSpeed=0.0f,frameAngularSpeed=0.0f,frameJointError=0.0f;
		PxBounds3 netBounds=PxBounds3::empty();
		const PxReal x0=-SPACING*PxReal(W-1)*0.5f,z0=-SPACING*PxReal(H-1)*0.5f;
		for(PxU32 r=0;r<H;++r)for(PxU32 c=0;c<W;++c){PxRigidDynamic* b=nodes[r*W+c]; const PxTransform p=b->getGlobalPose(); if(ball)netBounds.include(b->getWorldBounds(1.0f)); const PxVec3 v=b->getLinearVelocity(); if(!p.isFinite()||!v.isFinite()||!b->getAngularVelocity().isFinite())++metrics.nonFinite; const PxReal speed=v.magnitude(); metrics.maxSpeed=PxMax(metrics.maxSpeed,speed);metrics.maxNetSpeed=PxMax(metrics.maxNetSpeed,speed); if((r==0||r+1==H)&&(c==0||c+1==W)){const PxVec3 q(x0+c*SPACING,NET_Y,z0+r*SPACING);metrics.maxCornerDrift=PxMax(metrics.maxCornerDrift,(p.p-q).magnitude());}}
		metrics.finalCenterY=nodes[(H/2)*W+W/2]->getGlobalPose().p.y;metrics.centerMinY=PxMin(metrics.centerMinY,metrics.finalCenterY);if(metrics.firstContactFrame==PX_MAX_U32)metrics.preContactCenterMinY=PxMin(metrics.preContactCenterMinY,metrics.finalCenterY);
		for(PxU32 i=0;i<BODY_COUNT;++i){frameSpeed=PxMax(frameSpeed,nodes[i]->getLinearVelocity().magnitude());frameAngularSpeed=PxMax(frameAngularSpeed,nodes[i]->getAngularVelocity().magnitude());}
		for(size_t i=0;i<joints.size();++i){PxRigidActor*a=NULL,*b=NULL;joints[i]->getActors(a,b);const PxVec3 pa=a->getGlobalPose().transform(joints[i]->getLocalPose(PxJointActorIndex::eACTOR0)).p;const PxVec3 pb=b->getGlobalPose().transform(joints[i]->getLocalPose(PxJointActorIndex::eACTOR1)).p;frameJointError=PxMax(frameJointError,(pa-pb).magnitude());}
		metrics.maxJointError=PxMax(metrics.maxJointError,frameJointError);
		// Height alone accepts a ball flying past the edge of the net.  Disjoint
		// projected shape bounds prove loss of retention, even while airborne.
		// Overlap is only a necessary condition; the other physical gates remain.
		if(ball){
			const PxBounds3 ballBounds=ball->getWorldBounds(1.0f);
			const PxReal gapX=PxMax(netBounds.minimum.x-ballBounds.maximum.x,ballBounds.minimum.x-netBounds.maximum.x);
			const PxReal gapZ=PxMax(netBounds.minimum.z-ballBounds.maximum.z,ballBounds.minimum.z-netBounds.maximum.z);
			metrics.maxBallPlanarSeparation=PxMax(metrics.maxBallPlanarSeparation,PxMax(gapX,gapZ));
		}
		if(ball){metrics.finalBallY=ball->getGlobalPose().p.y;metrics.ballMinY=PxMin(metrics.ballMinY,metrics.finalBallY);metrics.maxBallSpeed=PxMax(metrics.maxBallSpeed,ball->getLinearVelocity().magnitude());metrics.maxSpeed=PxMax(metrics.maxSpeed,metrics.maxBallSpeed);if(metrics.firstContactFrame!=PX_MAX_U32){metrics.maxPostContactBallVy=PxMax(metrics.maxPostContactBallVy,ball->getLinearVelocity().y);metrics.reboundRise=PxMax(metrics.reboundRise,metrics.finalBallY-metrics.ballMinY);}}
		if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_AVBD_HEADLESS_PROGRESS"))){std::printf("[CHAINMAIL_PROGRESS] frame=%u centerY=%.9g ballY=%.9g maxSpeed=%.9g maxAngularSpeed=%.9g jointError=%.9g contacts=%u ballX=%.9g ballZ=%.9g ballVx=%.9g ballVy=%.9g ballVz=%.9g\n",metrics.completed,double(metrics.finalCenterY),double(metrics.finalBallY),double(frameSpeed),double(frameAngularSpeed),double(frameJointError),metrics.contacts,double(ball?ball->getGlobalPose().p.x:0),double(ball?ball->getGlobalPose().p.z:0),double(ball?ball->getLinearVelocity().x:0),double(ball?ball->getLinearVelocity().y:0),double(ball?ball->getLinearVelocity().z:0));std::fflush(stdout);}
	}
	bool step(PxReal dt){
		const auto begin=std::chrono::steady_clock::now();const PxU32 contacts=metrics.contacts;
		if(!scene->simulate(dt)){++metrics.fetchFailures;return false;}if(!scene->fetchResults(true)){++metrics.fetchFailures;return false;}
		const auto fetched=std::chrono::steady_clock::now();const double elapsed=std::chrono::duration<double,std::milli>(fetched-begin).count();
		metrics.stepMs+=elapsed;metrics.maxStepMs=PxMax(metrics.maxStepMs,elapsed);
		if(metrics.contacts>contacts){metrics.contactMs+=elapsed;++metrics.contactFrames;}
		++metrics.completed;sample();metrics.observerMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fetched).count();return true;
	}
};

#include "SnippetChainmailVolley.inl"
static bool supportedCase(const char* c){return Snippets::equalsIgnoreCase(c,"topology")||Snippets::equalsIgnoreCase(c,"sag")||Snippets::equalsIgnoreCase(c,"impact")||Snippets::equalsIgnoreCase(c,"projectile")||Snippets::equalsIgnoreCase(c,"volley");}
}

bool snippetChainmailHeadlessShouldRun(int argc,const char*const*argv){if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS")))return true;for(int i=1;i<argc;++i)if(argv[i]&&std::strcmp(argv[i],"--headless")==0)return true;return false;}

int snippetChainmailHeadlessMain(int argc,const char*const*argv){
	Snippets::HeadlessOptions d;d.headless=true;d.caseName="impact";d.frames=600; Snippets::HeadlessOptions o;std::string error;
	if(!Snippets::parseCommonHeadlessOptions(argc,argv,d,o,error)||!supportedCase(o.caseName.c_str())){std::fprintf(stderr,"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetChainmail reason=%s\n",error.empty()?"unsupported_case":error.c_str());return Snippets::eHEADLESS_CONFIG_ERROR;}
	for(int i=1;i<argc;++i)if(!Snippets::isCommonHeadlessOption(argv[i])){std::fprintf(stderr,"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetChainmail reason=unknown_option\n");return Snippets::eHEADLESS_CONFIG_ERROR;}
	const bool volley=Snippets::equalsIgnoreCase(o.caseName.c_str(),"volley");
	if(volley && !o.framesExplicit)o.frames=4200;
	if(!Snippets::applyExecutionEnvironment(o))return Snippets::eHEADLESS_CONFIG_ERROR; Snippets::printHeadlessConfig("SnippetChainmail",o);
	if(volley)return runVolley(o);
	const bool topology=Snippets::equalsIgnoreCase(o.caseName.c_str(),"topology"),sag=Snippets::equalsIgnoreCase(o.caseName.c_str(),"sag"),projectile=Snippets::equalsIgnoreCase(o.caseName.c_str(),"projectile");
	Fixture f; bool ok=f.init(o,!topology&&!sag,projectile); const PxU32 frames=topology?1:o.frames; for(PxU32 i=0;ok&&i<frames;++i)ok=f.step(o.dt);
	const Metrics&m=f.metrics;const bool activity=topology||(m.centerMinY<NET_Y-1e-4f); const bool contact=(topology||sag)||(m.contacts>0&&m.contactPoints>0); const bool retained=(topology||sag)||(m.finalBallY>20.0f&&m.ballMinY>20.0f&&m.finalCenterY>20.0f&&m.maxBallPlanarSeparation==0.0f); const bool impactEnergy=topology||sag||(projectile?(m.maxBallSpeed<70.0f&&m.maxNetSpeed<60.0f):(m.maxBallSpeed<35.0f&&m.maxNetSpeed<60.0f));
	std::printf("[CHAINMAIL_TIMING] workers=%u commonStep=%s npGeometry=%s completed=%u averageStepMs=%.9g maxStepMs=%.9g contactFrames=%u averageContactMs=%.9g averageObserverMs=%.9g\n",
		o.dispatcherThreads,std::getenv("PHYSX_AVBD_COMMON_STEP")?std::getenv("PHYSX_AVBD_COMMON_STEP"):"default",
		std::getenv("PHYSX_NP_GEOMETRY_CERTIFICATES")?std::getenv("PHYSX_NP_GEOMETRY_CERTIFICATES"):"off",m.completed,
		m.completed?m.stepMs/m.completed:0,m.maxStepMs,m.contactFrames,m.contactFrames?m.contactMs/m.contactFrames:0,m.completed?m.observerMs/m.completed:0);
	// Minimum deflection observes flexibility. A center height at one later
	// frame also depends on rebound phase and is only a reference envelope.
	const bool legacyEnvelope=topology||sag||(projectile?(m.centerMinY<32.0f&&m.finalCenterY<34.0f):(m.centerMinY<30.25f&&m.finalCenterY<31.8f));
	const bool flexible=topology||sag||(projectile?m.centerMinY<32.0f:m.centerMinY<30.25f);
	const bool bounded=impactEnergy&&m.maxJointError<1.0f&&m.maxCornerDrift<1e-4f;
	std::printf("[CHAINMAIL_QUALITY] schema=1 acceptance=engineering-motion-20261023 minDeflectionPass=%u legacyMotionEnvelopePass=%u legacyFinalCenterMax=%.9g finalCenterY=%.9g\n",
		flexible?1u:0u,legacyEnvelope?1u:0u,double(projectile?34.0f:31.8f),double(m.finalCenterY));
	// Preserve the reference AVBD motion, not merely containment.  The old
	// implementation passed a loose min-Y gate while the no-contact joint graph
	// rejected gravity and the material phase returned almost no stored energy.
	const bool referenceMotion=o.solverType!=PxSolverType::eAVBD||topology||
		(sag?m.centerMinY<32.0f:
			(projectile?(m.maxPostContactBallVy>7.0f&&m.reboundRise>5.0f):
				(m.preContactCenterMinY<31.5f&&m.maxPostContactBallVy>7.0f&&m.reboundRise>5.0f)));
	const char* reason="none";if(!ok)reason="initialization_or_fetch";else if(m.nonFinite||f.errors.getFatalCount())reason="runtime_error";else if(!activity)reason="missing_net_activity";else if(!contact)reason="missing_ball_net_contact";else if(!retained)reason="ball_not_retained";else if(!flexible)reason="net_too_stiff";else if(!referenceMotion)reason="reference_motion_missing";else if(!bounded)reason="unbounded_or_constraint_error";
	const bool pass=ok&&!m.nonFinite&&!f.errors.getFatalCount()&&activity&&contact&&retained&&flexible&&referenceMotion&&bounded;
	const char* gateReason=!pass&&ok&&!m.nonFinite&&!f.errors.getFatalCount()?"physical_contract":reason;
	std::printf("[AVBD_GATE] schema=1 snippet=SnippetChainmail solver=%s case=%s execution=%s frames=%u completedFrames=%u status=%s reason=%s detail=%s capability=SUPPORTED validation=GATED gridBodies=%u jointCount=%u contacts=%u contactPoints=%u firstContactFrame=%u finalCenterY=%.9g centerMinY=%.9g preContactCenterMinY=%.9g finalBallY=%.9g ballMinY=%.9g maxPostContactBallVy=%.9g reboundRise=%.9g maxSpeed=%.9g maxNetSpeed=%.9g maxBallSpeed=%.9g maxJointError=%.9g maxCornerDrift=%.9g maxBallPlanarSeparation=%.9g nonFinite=%u fetchFailures=%u fatalErrors=%u\n",Snippets::getSolverTypeName(o.solverType),o.caseName.c_str(),Snippets::getExecutionName(o.execution),frames,m.completed,pass?"PASS":"FAIL",gateReason,reason,BODY_COUNT,JOINT_COUNT,m.contacts,m.contactPoints,m.firstContactFrame,double(m.finalCenterY),double(m.centerMinY),double(m.preContactCenterMinY),double(m.finalBallY),double(m.ballMinY),double(m.maxPostContactBallVy),double(m.reboundRise),double(m.maxSpeed),double(m.maxNetSpeed),double(m.maxBallSpeed),double(m.maxJointError),double(m.maxCornerDrift),double(m.maxBallPlanarSeparation),m.nonFinite,m.fetchFailures,f.errors.getFatalCount());
	return pass?Snippets::eHEADLESS_PASS:Snippets::eHEADLESS_GATE_FAILED;
}
