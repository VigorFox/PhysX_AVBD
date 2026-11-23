// Independent benchmark client of the unmodified, pinned Jolt library.
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/PlaneShape.h>
#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace JPH;

class BroadLayers final : public BroadPhaseLayerInterface
{
public:
    uint GetNumBroadPhaseLayers() const override { return 2; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer layer) const override { return BroadPhaseLayer(uint8(layer)); }
};
class BroadFilter final : public ObjectVsBroadPhaseLayerFilter
{
public:
    bool ShouldCollide(ObjectLayer a, BroadPhaseLayer b) const override { return a != 0 || b.GetValue() != 0; }
};
class PairFilter final : public ObjectLayerPairFilter
{
public:
    bool ShouldCollide(ObjectLayer a, ObjectLayer b) const override { return a != 0 || b != 0; }
};

static void token(std::istream& in, const char* expected)
{
    std::string value;
    if(!(in >> value) || value != expected) throw std::runtime_error(std::string("Expected ") + expected);
}
static Vec3 readVec(std::istream& in)
{
    float x, y, z;
    if(!(in >> x >> y >> z) || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        throw std::runtime_error("Invalid vector");
    return Vec3(x, y, z);
}
static Quat readQuat(std::istream& in)
{
    float x, y, z, w;
    if(!(in >> x >> y >> z >> w)) throw std::runtime_error("Invalid quaternion");
    return Quat(x, y, z, w).Normalized();
}
static void append(std::vector<float>& out, Vec3Arg p)
{
    out.push_back(p.GetX()); out.push_back(p.GetY()); out.push_back(p.GetZ());
}

int main(int argc, char** argv)
{
    try
    {
        if(argc != 7) throw std::runtime_error("Usage: JoltSoftBenchmark scene.txt trace.bin frames iterations edgeCompliance volumeCompliance");
        const uint frames = uint(std::stoul(argv[3])), iterations = uint(std::stoul(argv[4]));
        const float edgeCompliance = std::stof(argv[5]), volumeCompliance = std::stof(argv[6]);
        if(!frames || !iterations || iterations > 1024 || !std::isfinite(edgeCompliance) ||
            !std::isfinite(volumeCompliance) || edgeCompliance < 0 || volumeCompliance < 0)
            throw std::runtime_error("Invalid benchmark parameters");
        RegisterDefaultAllocator();
        Factory::sInstance = new Factory();
        RegisterTypes();
        {
            std::ifstream input(argv[1]);
            token(input, "AVBD_JOLT_SCENE"); token(input, "1"); token(input, "case");
            std::string caseName; input >> caseName;
            token(input, "dt"); float dt; input >> dt;
            token(input, "gravity"); const Vec3 gravity = readVec(input);
            token(input, "soft");
            uint nv, nt, collisionVertices, collisionTets;
            float young, poisson, friction, elasticityDamping, damping, maxVelocity;
            input >> nv >> nt >> collisionVertices >> collisionTets >> young >> poisson >> friction
                >> elasticityDamping >> damping >> maxVelocity;
            if(!input || nv == 0 || nt == 0 || nv > 10000000 || nt > 10000000 || !(dt > 0))
                throw std::runtime_error("Invalid scene metadata");
            token(input, "vertices");
            Ref<SoftBodySharedSettings> shared = new SoftBodySharedSettings;
            for(uint i = 0; i < nv; ++i)
            {
                const Vec3 p = readVec(input);
                float invMass; input >> invMass;
                const Vec3 v = readVec(input);
                Float3 pf, vf; p.StoreFloat3(&pf); v.StoreFloat3(&vf);
                if(!(invMass > 0)) throw std::runtime_error("Expected positive inverse mass");
                shared->mVertices.emplace_back(pf, vf, invMass);
            }
            token(input, "tets");
            std::set<std::array<uint, 2>> edges;
            std::set<std::array<uint, 4>> uniqueTets;
            struct FaceEntry { std::array<uint, 3> oriented; uint count = 0; };
            std::map<std::array<uint, 3>, FaceEntry> faces;
            for(uint i = 0; i < nt; ++i)
            {
                std::array<uint, 4> ids;
                for(uint& id : ids) { input >> id; if(id >= nv) throw std::runtime_error("Invalid tet index"); }
                auto tetKey = ids; std::sort(tetKey.begin(), tetKey.end()); uniqueTets.insert(tetKey);
                auto position = [&](uint id) { return Vec3(shared->mVertices[id].mPosition); };
                const Vec3 p = position(ids[0]);
                const float det = (position(ids[1]) - p).Cross(position(ids[2]) - p).Dot(position(ids[3]) - p);
                if(std::abs(det) < 1.0e-12f) throw std::runtime_error("Degenerate tet");
                // Jolt's volume gradient assumes positive orientation.
                if(det < 0) std::swap(ids[1], ids[2]);
                shared->mVolumeConstraints.emplace_back(ids[0], ids[1], ids[2], ids[3], volumeCompliance);
                for(uint a = 0; a < 4; ++a)
                    for(uint b = a + 1; b < 4; ++b)
                        edges.insert({ std::min(ids[a], ids[b]), std::max(ids[a], ids[b]) });
                for(uint opposite = 0; opposite < 4; ++opposite)
                {
                    std::array<uint, 3> face; uint k = 0;
                    for(uint j = 0; j < 4; ++j) if(j != opposite) face[k++] = ids[j];
                    const Vec3 a = position(face[0]);
                    if((position(face[1]) - a).Cross(position(face[2]) - a).Dot(position(ids[opposite]) - a) > 0)
                        std::swap(face[1], face[2]);
                    auto key = face; std::sort(key.begin(), key.end());
                    auto& entry = faces[key]; entry.oriented = face; ++entry.count;
                }
            }
            for(const auto& edge : edges) shared->mEdgeConstraints.emplace_back(edge[0], edge[1], edgeCompliance);
            uint nonManifoldFaces = 0;
            for(const auto& item : faces)
            {
                // PhysX voxel cooking can produce overlapping background tets.
                // Preserve every volume constraint and report the topology;
                // do not silently deduplicate the workload for one engine.
                if(item.second.count > 2) ++nonManifoldFaces;
                if(item.second.count == 1)
                {
                    const auto& f = item.second.oriented;
                    shared->AddFace(SoftBodySharedSettings::Face(f[0], f[1], f[2]));
                }
            }
            shared->CalculateEdgeLengths();
            shared->CalculateVolumeConstraintVolumes();
            shared->Optimize();

            BroadLayers layers; BroadFilter broadFilter; PairFilter pairFilter;
            PhysicsSystem system;
            system.Init(1024, 0, 4096, 4096, layers, broadFilter, pairFilter);
            system.SetGravity(gravity);
            // The AVBD Volume backend uses the soft material's friction directly.
            system.SetCombineFriction([](const Body& a, const SubShapeID&, const Body& b, const SubShapeID&) {
                return a.IsSoftBody() ? a.GetFriction() : b.IsSoftBody() ? b.GetFriction() :
                    0.5f * (a.GetFriction() + b.GetFriction());
            });
            BodyInterface& bodies = system.GetBodyInterface();
            token(input, "rigids"); uint nr; input >> nr;
            std::vector<BodyID> rigids;
            for(uint i = 0; i < nr; ++i)
            {
                std::string type; uint dynamic;
                input >> type >> dynamic;
                const Vec3 p = readVec(input); const Quat q = readQuat(input); const Vec3 half = readVec(input);
                float mass, mu, restitution, linearDamping, angularDamping;
                input >> mass >> mu >> restitution >> linearDamping >> angularDamping;
                const Vec3 inertia = readVec(input);
                RefConst<Shape> shape;
                if(type == "box") shape = new BoxShape(half, 0.0f);
                else if(type == "plane" && !dynamic) shape = new PlaneShape(Plane(Vec3::sAxisX(), 0));
                else throw std::runtime_error("Unsupported rigid shape");
                BodyCreationSettings settings(shape, RVec3(p), q, dynamic ? EMotionType::Dynamic : EMotionType::Static,
                    dynamic ? 1 : 0);
                settings.mFriction = mu; settings.mRestitution = restitution;
                settings.mLinearDamping = linearDamping; settings.mAngularDamping = angularDamping;
                if(dynamic)
                {
                    settings.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
                    settings.mMassPropertiesOverride.mMass = mass;
                    settings.mMassPropertiesOverride.mInertia = Mat44::sScale(inertia);
                }
                BodyID id = bodies.CreateAndAddBody(settings, dynamic ? EActivation::Activate : EActivation::DontActivate);
                if(id.IsInvalid()) throw std::runtime_error("Rigid creation failed");
                rigids.push_back(id);
            }
            std::string trailing;
            if(!input || input >> trailing) throw std::runtime_error("Invalid scene ending");
            SoftBodyCreationSettings soft(shared, RVec3::sZero(), Quat::sIdentity(), 1);
            soft.mNumIterations = iterations;
            soft.mLinearDamping = damping;
            soft.mMaxLinearVelocity = maxVelocity;
            soft.mFriction = friction;
            const BodyID softID = bodies.CreateAndAddSoftBody(soft, EActivation::Activate);
            if(softID.IsInvalid()) throw std::runtime_error("Soft body creation failed");
            system.OptimizeBroadPhase();
            TempAllocatorImpl allocator(32 * 1024 * 1024);
            JobSystemSingleThreaded jobs(cMaxPhysicsJobs);
            std::ofstream trace(argv[2], std::ios::binary);
            if(!trace) throw std::runtime_error("Cannot open trace");
            std::vector<float> row; row.reserve(2 + nv * 3 + nr * 7);
            double totalMs = 0;
            for(uint frame = 0; frame < frames; ++frame)
            {
                const bool active = bodies.IsActive(softID);
                const auto start = std::chrono::steady_clock::now();
                const EPhysicsUpdateError error = system.Update(dt, 1, &allocator, &jobs);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                if(error != EPhysicsUpdateError::None) throw std::runtime_error("Physics Update failed");
                totalMs += ms;
                row.clear(); row.push_back(float(ms)); row.push_back(active ? 1.0f : 0.0f);
                const Body& body = *system.GetBodyLockInterfaceNoLock().TryGetBody(softID);
                const auto& vertices = static_cast<const SoftBodyMotionProperties*>(body.GetMotionProperties())->GetVertices();
                const RMat44 transform = body.GetCenterOfMassTransform();
                for(const auto& vertex : vertices) append(row, Vec3(transform * vertex.mPosition));
                for(BodyID id : rigids)
                {
                    append(row, Vec3(bodies.GetPosition(id)));
                    const Quat rotation = bodies.GetRotation(id);
                    row.push_back(rotation.GetX()); row.push_back(rotation.GetY());
                    row.push_back(rotation.GetZ()); row.push_back(rotation.GetW());
                }
                trace.write(reinterpret_cast<const char*>(row.data()), std::streamsize(row.size() * sizeof(float)));
                if(!trace) throw std::runtime_error("Trace write failed");
            }
            trace.close();
            if(!trace) throw std::runtime_error("Trace close failed");
            std::printf("[JOLT_BENCH] case=%s frames=%u simulationVertices=%u simulationTets=%u edges=%u boundaryFaces=%u iterations=%u threads=1 edgeCompliance=%.9g volumeCompliance=%.9g averageStepMs=%.9g duplicateTets=%u nonManifoldFaces=%u\n",
                caseName.c_str(), frames, nv, nt, unsigned(edges.size()), unsigned(shared->mFaces.size()), iterations,
                double(edgeCompliance), double(volumeCompliance), totalMs / frames,
                nt - unsigned(uniqueTets.size()), nonManifoldFaces);
            bodies.RemoveBody(softID); bodies.DestroyBody(softID);
            for(BodyID id : rigids) { bodies.RemoveBody(id); bodies.DestroyBody(id); }
        }
        UnregisterTypes(); delete Factory::sInstance; Factory::sInstance = nullptr;
        return 0;
    }
    catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
