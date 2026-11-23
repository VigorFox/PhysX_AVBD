#ifdef RENDER_SNIPPET
#include "PxPhysicsAPI.h"
#include "../snippetrender/SnippetCamera.h"
#include "../snippetrender/SnippetRender.h"
using namespace physx;
extern void initPhysics(bool); extern void stepPhysics(bool); extern void cleanupPhysics(bool);
extern void keyPress(unsigned char, const PxTransform&);
namespace { Snippets::Camera* sCamera;
void renderCallback() { stepPhysics(true); Snippets::startRender(sCamera); PxScene* scene; PxGetPhysics().getScenes(&scene, 1); const PxU32 n=scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC|PxActorTypeFlag::eRIGID_STATIC); if(n){ PxArray<PxRigidActor*> a(n); scene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC|PxActorTypeFlag::eRIGID_STATIC, reinterpret_cast<PxActor**>(a.begin()), n); Snippets::renderActors(a.begin(), n, false, PxVec3(0,0.75f,0), NULL, true, false); } Snippets::finishRender(); }
void exitCallback(){ delete sCamera; cleanupPhysics(true); } }
void renderLoop(){ sCamera=new Snippets::Camera(PxVec3(30,50,30),PxVec3(-0.5f,-0.4f,-0.5f)); Snippets::setupDefault("PhysX Snippet Chainmail",sCamera,keyPress,renderCallback,exitCallback); initPhysics(true); glutMainLoop(); }
#endif
