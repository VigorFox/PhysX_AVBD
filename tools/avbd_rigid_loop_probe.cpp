#include "PxPhysicsAPI.h"
#include "avbd/DyAvbdRigidLoop.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>
using namespace physx;
using physx::Dy::AvbdRigidLoop;

struct Edge
{
    unsigned n, ids[2];
};
static bool check(unsigned n, unsigned workers, bool forest)
{
    std::vector<Edge> edges;
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < i; ++j)
        {
            if (forest && i / 8 != j / 8)
                continue;
            if (j + 1 == i || (i * 11 + j * 7) % 13 == 0)
                edges.push_back({2, {i, j}});
        }
    AvbdRigidLoop matrix;
    auto *m = &matrix;
    const auto initialEdges = edges;
    const unsigned width = 6 * n;
    double worst = 0, workError = 0;
    for (unsigned sweep = 0; sweep < 4; ++sweep)
    {
        if (sweep == 1)
            std::reverse(edges.begin(), edges.end());
        else if (sweep == 2 && !edges.empty())
            edges.pop_back();
        else if (sweep == 3)
            edges = initialEdges;
        m->bake(n, edges);
        std::vector<double> dense(width * width, 0), g(width);
        for (unsigned i = 0; i < n; ++i)
        {
            m->clearColumn(i);
            m->locks[i] = 0;
            for (unsigned d = 0; d < 6; ++d)
            {
                const unsigned u = 6 * m->order[i] + d;
                m->at(i, i)[d][d] = dense[u * width + u] = 2 + .03 * u;
                m->g[i][d] = g[u] = std::cos(.19 * u + sweep);
            }
        }
        for (const auto &edge : edges)
        {
            const auto *id = edge.ids;
            const unsigned a = m->perm[id[0]], b = m->perm[id[1]];
            const unsigned src = std::min(a, b), dst = std::max(a, b);
            unsigned slot = 0;
            while (slot < m->nb[src] && m->after[src][slot] != dst)
                ++slot;
            if (slot == m->nb[src])
                return false;
            auto &peer = m->peerData[m->rawSlot[m->edgeOffset[src] + slot]];
            const double k = 35. + id[0];
            double j[2][6];
            for (unsigned s = 0; s < 2; ++s)
                for (unsigned d = 0; d < 6; ++d)
                    j[s][d] = (s ? -1. : 1.) * std::sin(.1 + id[0] * .052 + d + .11 * s);
            for (unsigned s = 0; s < 2; ++s)
                for (unsigned d = 0; d < 6; ++d)
                {
                    const unsigned u = 6 * id[s] + d;
                    const double c = (s ? -1 : 1) * .01 * (1 + d);
                    g[u] += c;
                    const auto own = m->perm[id[s]] == src;
                    if (own)
                        m->g[src][d] += c;
                    else
                        peer.g[d] += c;
                    for (unsigned v = 0; v < 6; ++v)
                    {
                        const auto h = k * j[s][d] * j[s][v];
                        if (own)
                            m->at(src, src)[d][v] += h;
                        else
                            peer.h[d][v] += h;
                        for (unsigned t = 0; t < 2; ++t)
                            dense[u * width + 6 * id[t] + v] += k * j[s][d] * j[t][v];
                    }
                }
            m->cross(id, j, k);
        }
        m->resetTasks();
        std::atomic<unsigned> factors{0}, backs{0};
        const auto reverse = [&](auto &&self, unsigned i) -> void
        {
            if (!m->backColumn(i))
                m->failed.store(true);
            ++backs;
            for (unsigned j = 0; j < m->childCount[i]; ++j)
                self(self, m->children[i][j]);
        };
        std::vector<std::thread> threads;
        for (unsigned w = 0; w < workers; ++w)
            threads.emplace_back(
                [&, w]()
                {
                    for (unsigned j = w; j < n; j += workers)
                    {
                        unsigned i = (j + sweep) % n;
                        while (m->arrive(i))
                        {
                            if (!m->factorColumn(i))
                                m->failed.store(true);
                            ++factors;
                            if (m->parent[i] == AvbdRigidLoop::Bodies)
                            {
                                reverse(reverse, i);
                                break;
                            }
                            i = m->parent[i];
                        }
                    }
                });
        for (auto &t : threads)
            t.join();
        if (m->failed.load() || factors.load() != n || backs.load() != n)
            return false;
        double work = 0, slope = 0;
        for (unsigned u = 0; u < width; ++u)
        {
            double ax = 0;
            for (unsigned v = 0; v < width; ++v)
                ax += dense[u * width + v] * m->step[v / 6][v % 6];
            worst = std::max(worst, std::abs(ax + g[u]));
            work += m->step[u / 6][u % 6] * (g[u] + .5 * ax);
        }
        workError = std::max(workError, std::abs(m->model(m->step, slope) - work));
    }
    const bool ok = worst < 1.e-10 && workError < 1.e-10;
    std::printf("[RIGID_LOOP] bodies=%u workers=%u forest=%d residual=%.8g trialError=%.8g status=%s\n", n,
                workers, int(forest), worst, workError, ok ? "PASS" : "FAIL");
    return ok;
}
int main()
{
    PxDefaultAllocator alloc;
    PxDefaultErrorCallback errors;
    auto *foundation = PxCreateFoundation(PX_PHYSICS_VERSION, alloc, errors);
    if (!foundation)
        return 1;
    bool ok = true;
    for (unsigned n : {1u, 4u, 26u, 32u})
        for (unsigned w : {1u, 4u, 8u})
            for (bool forest : {false, true})
                ok = check(n, w, forest) && ok;
    foundation->release();
    return ok ? 0 : 1;
}
