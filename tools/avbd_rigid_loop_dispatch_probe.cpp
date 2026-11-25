#include "PxPhysicsAPI.h"
#include <cstdio>

using namespace physx;

namespace
{
const PxU32 RingCount = 48, RingSize = 4, FrameCount = 120;
const PxVec3 corners[RingSize] = {PxVec3(-.5f, 0, -.5f), PxVec3(.5f, 0, -.5f),
                                  PxVec3(.5f, 0, .5f), PxVec3(-.5f, 0, .5f)};

PxVec3 origin(PxU32 ring)
{
    return PxVec3(4.f * (ring % 8), 0, 4.f * (ring / 8));
}

bool run(PxPhysics& physics, PxMaterial& material, PxU32 workers)
{
    PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(workers);
    if (!dispatcher)
        return false;
    PxSceneDesc desc(physics.getTolerancesScale());
    desc.cpuDispatcher = dispatcher;
    desc.filterShader = PxDefaultSimulationFilterShader;
    desc.gravity = PxVec3(0);
    desc.solverType = PxSolverType::eAVBD;
    PxScene* scene = physics.createScene(desc);
    if (!scene)
    {
        dispatcher->release();
        return false;
    }

    // More independent roots than fit in one wave. Alternating velocities
    // load each loop while its center of mass has an analytic free flight.
    const PxVec3 translation(.25f, .1f, -.12f);
    PxRigidDynamic* bodies[RingCount][RingSize] = {};
    PxJoint* joints[RingCount][RingSize] = {};
    bool valid = true;
    for (PxU32 ring = 0; valid && ring < RingCount; ++ring)
    {
        for (PxU32 i = 0; valid && i < RingSize; ++i)
        {
            auto* body = PxCreateDynamic(physics, PxTransform(origin(ring) + corners[i]),
                                         PxSphereGeometry(.15f), material, 1.f);
            bodies[ring][i] = body;
            valid = body != NULL;
            if (body)
            {
                body->setLinearDamping(0);
                body->setAngularDamping(0);
                body->setSleepThreshold(0);
                const PxReal speed = (.08f + .01f * ring) * (i % 2 ? 1.f : -1.f);
                body->setLinearVelocity(translation + PxVec3(0, speed, 0));
                scene->addActor(*body);
            }
        }
        for (PxU32 i = 0; valid && i < RingSize; ++i)
        {
            const PxU32 next = (i + 1) % RingSize;
            const PxVec3 arm = (corners[next] - corners[i]) * .5f;
            joints[ring][i] = PxSphericalJointCreate(physics, bodies[ring][i], PxTransform(arm),
                                                     bodies[ring][next], PxTransform(-arm));
            valid = joints[ring][i] != NULL;
        }
    }

    const PxReal dt = 1.f / 60.f;
    PxReal anchorGap = 0, centerError = 0;
    PxU32 completed = 0;
    while (valid && completed < FrameCount)
    {
        PxU32 error = 0;
        valid = scene->simulate(dt) && scene->fetchResults(true, &error) && !error;
        if (!valid)
            break;
        ++completed;
        for (PxU32 ring = 0; ring < RingCount; ++ring)
        {
            PxVec3 center(0);
            for (PxU32 i = 0; i < RingSize; ++i)
            {
                const PxU32 next = (i + 1) % RingSize;
                const PxTransform a = bodies[ring][i]->getGlobalPose(), b = bodies[ring][next]->getGlobalPose();
                valid &= a.isFinite() && b.isFinite();
                const PxVec3 arm = (corners[next] - corners[i]) * .5f;
                anchorGap = PxMax(anchorGap, (a.transform(arm) - b.transform(-arm)).magnitude());
                center += a.p * (1.f / RingSize);
            }
            centerError = PxMax(centerError, (center - origin(ring) - translation * (completed * dt)).magnitude());
        }
    }
    valid &= anchorGap < .02f && centerError < .01f;
    std::printf("[RIGID_LOOP_DISPATCH] workers=%u frames=%u anchorGap=%g centerError=%g status=%s\n",
                workers, completed, double(anchorGap), double(centerError), valid ? "PASS" : "FAIL");
    for (auto& ring : joints)
        for (auto* joint : ring)
            if (joint)
                joint->release();
    scene->release();
    for (auto& ring : bodies)
        for (auto* body : ring)
            if (body)
                body->release();
    dispatcher->release();
    return valid;
}
} // namespace

int main()
{
    PxDefaultAllocator allocator;
    PxDefaultErrorCallback errors;
    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
    if (!foundation)
        return 2;
    PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
    if (!physics || !PxInitExtensions(*physics, NULL))
        return 2;
    PxMaterial* material = physics->createMaterial(.4f, .4f, 0);
    if (!material)
        return 2;
    const bool four = run(*physics, *material, 4);
    const bool eight = run(*physics, *material, 8);
    material->release();
    PxCloseExtensions();
    physics->release();
    foundation->release();
    return four && eight ? 0 : 1;
}
