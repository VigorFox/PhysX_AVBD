// Single-scene throughput probe for simultaneously ready high-valence material blocks.
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include <algorithm>
#include <chrono>
#include <vector>

using namespace physx;

int main(int argc, char** argv)
{
	if(argc != 2) return 2;
	const PxU32 workers = PxU32(std::atoi(argv[1]));
	constexpr PxU32 roots = 32, faces = 384, frames = 72;
	const auto origin = [](PxU32 r) { return PxVec3(4.f * (r % 8), 2.f, 4.f * (r / 8)); };
	AvbdProbeFoundation foundation;
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation.foundation, PxTolerancesScale());
	if(!physics) return 2;
	auto* dispatcher = PxDefaultCpuDispatcherCreate(workers);
	PxSceneDesc desc(physics->getTolerancesScale());
	desc.solverType = PxSolverType::eAVBD;
	desc.cpuDispatcher = dispatcher;
	desc.filterShader = PxDefaultSimulationFilterShader;
	desc.gravity = PxVec3(0);
	auto* scene = physics->createScene(desc);
	std::vector<PxVec3> vertices;
	std::vector<PxU32> indices;
	for(PxU32 r = 0; r < roots; ++r)
	{
		const PxU32 start = PxU32(vertices.size());
		vertices.push_back(origin(r));
		for(PxU32 j = 0; j < faces; ++j)
		{
			const PxReal a = PxTwoPi * j / faces;
			vertices.push_back(origin(r) + 1.5f * PxVec3(PxCos(a), 0, PxSin(a)));
			indices.insert(indices.end(), {start, start + 1 + (j+1)%faces, start + 1 + j});
		}
	}
	PxTriangleMeshDesc input;
	input.points.count = PxU32(vertices.size()); input.points.stride = sizeof(PxVec3); input.points.data = vertices.data();
	input.triangles.count = roots * faces; input.triangles.stride = 3 * sizeof(PxU32); input.triangles.data = indices.data();
	PxCookingParams cook(physics->getTolerancesScale());
	cook.buildTriangleAdjacencies = true;
	auto* mesh = PxCreateTriangleMesh(cook, input, physics->getPhysicsInsertionCallback());
	auto* material = physics->createDeformableSurfaceMaterial(1.e5f, .3f, 0, .02f, .01f);
	auto* sheet = physics->createDeformableSurface(PxDeformableSurfaceBackend::eCPU_AVBD);
	if(!scene || !mesh || !material || !sheet || mesh->getNbTriangles() != roots*faces) return 2;
	auto* shape = physics->createShape(PxTriangleMeshGeometry(mesh), &material, 1, true, PxShapeFlag::eSIMULATION_SHAPE);
	if(!shape || !sheet->attachShape(*shape)) return 2;
	shape->release();
	auto* p = sheet->getPositionInvMassBufferH();
	auto* v = sheet->getVelocityBufferH();
	auto* rest = sheet->getRestPositionBufferH();
	std::vector<PxU32> centers;
	for(PxU32 i = 0; i < mesh->getNbVertices(); ++i)
	{
		const auto x = mesh->getVertices()[i];
		const PxU32 root = PxU32(x.x / 4.f + .5f) + 8 * PxU32(x.z / 4.f + .5f);
		const bool free = root < roots && (x - origin(root)).magnitudeSquared() < 1.e-8f;
		p[i] = PxVec4(x, free ? 1.f : 0.f); rest[i] = PxVec4(x, 0); v[i] = PxVec4(0.f);
		if(free) { centers.push_back(i); p[i].x += .02f; }
	}
	if(centers.size() != roots) return 2;
	sheet->setSolverIterationCounts(4, 1);
	sheet->setSleepThreshold(0);
	sheet->setLinearDamping(0);
	sheet->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
	sheet->markDirty(PxDeformableSurfaceDataFlag::eALL);
	if(!scene->addActor(*sheet)) return 2;
	std::vector<PxReal> best(roots, .02f);
	std::vector<double> times;
	bool valid = true;
	PxReal peak = 0, pinError = 0;
	for(PxU32 f = 1; f <= frames; ++f)
	{
		const auto start = std::chrono::steady_clock::now();
		if(!scene->simulate(1.f/60) || !scene->fetchResults(true)) return 1;
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
		if(f > 8) times.push_back(ms);
		for(PxU32 i = 0; i < mesh->getNbVertices(); ++i)
		{
			valid &= p[i].isFinite() && v[i].isFinite();
			const PxReal delta = (p[i].getXYZ()-rest[i].getXYZ()).magnitude();
			if(!p[i].w) pinError = PxMax(pinError, delta);
			else peak = PxMax(peak, delta);
		}
		for(PxU32 r = 0; r < roots; ++r)
			best[r] = PxMin(best[r], (p[centers[r]].getXYZ()-rest[centers[r]].getXYZ()).magnitude());
	}
	const auto worstBest = *std::max_element(best.begin(), best.end());
	valid &= worstBest < .019f && peak < .021f && pinError < 1.e-6f;
	double total = 0;
	for(double t : times) total += t;
	std::sort(times.begin(), times.end());
	std::printf("[FAN_SCALING] roots=%u faces=%u frames=%u samples=%u workers=%u worstBest=%.8g peak=%.8g pinError=%.8g meanMs=%.8g p95Ms=%.8g status=%s\n",
		roots, roots*faces, frames, PxU32(times.size()), workers, double(worstBest), double(peak), double(pinError),
		total/times.size(), times[(times.size()-1)*95/100], valid ? "PASS" : "FAIL");
	sheet->release(); mesh->release(); material->release(); scene->release(); dispatcher->release(); physics->release();
	return valid ? 0 : 1;
}
