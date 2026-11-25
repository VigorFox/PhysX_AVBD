// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cmath>
#include "foundation/PxUserAllocated.h"
#include "foundation/PxTransform.h"
#include <algorithm>
#include <atomic>
#include "foundation/PxArray.h"
#include <cstdint>

// Eliminate six spatial coordinates at a time. The symbolic fill pattern is
// baked from the same row endpoints, including inactive slots that may turn on.
namespace physx
{
namespace Dy
{
struct AvbdRigidLoop : PxUserAllocated
{
    enum
    {
        Bodies = 32,
        Width = 6
    };
    using Tile = double[Width][Width];
    struct Cell
    {
        Tile h;
    };
    physx::PxArray<Cell> numeric;
    unsigned short cell[Bodies][Bodies], rawSlot[Bodies * (Bodies - 1) / 2];
    unsigned base[Bodies], rawBegin[Bodies + 1];
    Tile &at(unsigned row, unsigned col)
    {
        return numeric[cell[col][row]].h;
    }
    const Tile &at(unsigned row, unsigned col) const
    {
        return numeric[cell[col][row]].h;
    }
    double g[Bodies][Width], v[Bodies][Width], step[Bodies][Width];
    unsigned savedCount = Bodies + 1;
    uint32_t savedEdges[Bodies];
    template <unsigned N> static bool factor(double (&a)[N][N], unsigned n)
    {
        for (unsigned i = 0; i < n; ++i)
            for (unsigned j = 0; j <= i; ++j)
            {
                double v = a[i][j];
                for (unsigned k = 0; k < j; ++k)
                    v -= a[i][k] * a[j][k];
                if (i == j)
                {
                    if (!(v > 0) || !std::isfinite(v))
                        return false;
                    a[i][i] = 1 / std::sqrt(v);
                }
                else
                    a[i][j] = v * a[j][j];
            }
        return true;
    }
    template <unsigned N> static void forward(const double (&l)[N][N], unsigned n, double *x)
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned k = 0; k < i; ++k)
                x[i] -= l[i][k] * x[k];
            x[i] *= l[i][i];
        }
    }
    template <unsigned N> static void backward(const double (&l)[N][N], unsigned n, double *x)
    {
        for (unsigned i = n; i-- > 0;)
        {
            for (unsigned k = i + 1; k < n; ++k)
                x[i] -= l[k][i] * x[k];
            x[i] *= l[i][i];
        }
    }

    unsigned n = 0, order[Bodies], perm[Bodies], nb[Bodies], after[Bodies][Bodies];
    using Mask = uint32_t;
    static unsigned population(Mask word)
    {
        unsigned count = 0;
        for (; word; word &= word - 1)
            ++count;
        return count;
    }
    static Mask levels(const Mask *graph, Mask domain, Mask start, Mask *ring, unsigned &count)
    {
        Mask seen = start;
        count = 0;
        while (start)
        {
            ring[count++] = start;
            Mask next = 0;
            for (unsigned i = 0; i < Bodies; ++i)
                if (start & (Mask(1) << i))
                    next |= graph[i];
            start = next & domain & ~seen;
            seen |= start;
        }
        return seen;
    }
    static void separatorDepth(const Mask *graph, Mask domain, unsigned depth, unsigned *priority)
    {
        while (domain)
        {
            Mask ring[Bodies];
            unsigned count;
            const auto seed = domain & (~domain + 1);
            const auto component = levels(graph, domain, seed, ring, count);
            domain &= ~component;
            const auto size = population(component);
            for (unsigned i = 0; i < Bodies; ++i)
                if (component & (Mask(1) << i))
                    priority[i] = depth;
            if (size <= 4)
                continue;
            const auto far = ring[count - 1];
            levels(graph, component, far & (~far + 1), ring, count);
            unsigned best = ~0u, left = population(ring[0]);
            Mask cut = 0;
            for (unsigned j = 1; j + 1 < count; ++j)
            {
                const auto width = population(ring[j]), right = size - left - width;
                if (4 * std::min(left, right) >= size)
                {
                    const auto cost = width * (Bodies + 1) + std::max(left, right);
                    if (cost < best)
                    {
                        best = cost;
                        cut = ring[j];
                    }
                }
                left += width;
            }
            if (cut)
                separatorDepth(graph, component & ~cut, depth + 1, priority);
        }
    }

    template <class Samples> void bake(unsigned bodies, const Samples &samples)
    {
        n = bodies;
        uint32_t adjacent[Bodies] = {}, alive = n == 32 ? ~uint32_t(0) : (uint32_t(1) << n) - 1;
        for (const auto &s : samples)
            if (s.n == 2)
            {
                adjacent[s.ids[0]] |= uint32_t(1) << s.ids[1];
                adjacent[s.ids[1]] |= uint32_t(1) << s.ids[0];
            }
        if (savedCount == n && std::equal(adjacent, adjacent + n, savedEdges))
            return;
        savedCount = n;
        std::copy(adjacent, adjacent + n, savedEdges);
        unsigned priority[Bodies] = {};
        separatorDepth(adjacent, alive, 0, priority);
        for (unsigned p = 0; p < n; ++p)
        {
            unsigned best = 0, minCost = ~0u;
            for (unsigned i = 0; i < n; ++i)
                if ((alive >> i) & 1)
                {
                    const auto cost = (Bodies - priority[i]) * (Bodies + 1) + population(adjacent[i] & alive);
                    if (cost < minCost)
                    {
                        minCost = cost;
                        best = i;
                    }
                }
            perm[best] = p;
            order[p] = best;
            alive &= ~(uint32_t(1) << best);
            auto next = adjacent[best] & alive;
            nb[p] = 0;
            for (unsigned i = 0; i < n; ++i)
                if ((next >> i) & 1)
                {
                    after[p][nb[p]++] = i;
                    adjacent[i] |= next & ~(uint32_t(1) << i);
                }
        }
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned s = 0; s < nb[i]; ++s)
                after[i][s] = perm[after[i][s]];
            std::sort(after[i], after[i] + nb[i]);
        }
        bakeTree();
        unsigned size = 0;
        for (unsigned i = 0; i < n; ++i)
        {
            base[i] = size;
            cell[i][i] = static_cast<unsigned short>(size++);
            for (unsigned a = 0; a < nb[i]; ++a)
            {
                cell[i][after[i][a]] = static_cast<unsigned short>(size++);
                rawSlot[edgeOffset[i] + a] = 65535;
            }
            ++size; // Columns written by different workers share no cache line.
        }
        numeric.resize(size);
        for (const auto &row : samples)
            if (row.n == 2)
            {
                const auto a = perm[row.ids[0]], b = perm[row.ids[1]], i = std::min(a, b), j = std::max(a, b);
                rawSlot[edgeOffset[i] + cell[i][j] - base[i] - 1] = 0;
            }
        unsigned raw = 0;
        for (unsigned i = 0; i < n; ++i)
        {
            rawBegin[i] = raw;
            for (unsigned j = 0; j < nb[i]; ++j)
                if (rawSlot[edgeOffset[i] + j] != 65535)
                    rawSlot[edgeOffset[i] + j] = static_cast<unsigned short>(raw++);
        }
        rawBegin[n] = raw;
        peerData.resize(raw);
    }
    struct Source
    {
        unsigned char pivot, slot;
    };
    struct Contribution
    {
        Tile h;
        double g[6];
    };
    Source predecessors[Bodies][Bodies];
    unsigned predCount[Bodies], childCount[Bodies], children[Bodies][Bodies];
    unsigned parent[Bodies], edgeOffset[Bodies], forkCount;
    physx::PxArray<Contribution> peerData;
    struct Counter
    {
        std::atomic<unsigned> value;
        unsigned char padding[60];
    };
    Counter pending[Bodies];
    unsigned post[Bodies], backWork[Bodies];
    uint32_t forkMask = 0;
    unsigned char forkTicket[Bodies];
    std::atomic<bool> failed{false};
    unsigned locks[Bodies];
    // Each raw row is produced by its first pivot. All other supports are
    // ancestors of that pivot in the filled elimination tree.
    void bakeTree()
    {
        unsigned edges = 0;
        forkCount = 0;
        for (unsigned i = 0; i < n; ++i)
        {
            predCount[i] = childCount[i] = 0;
        }
        for (unsigned i = 0; i < n; ++i)
        {
            edgeOffset[i] = edges;
            edges += nb[i];
            parent[i] = nb[i] ? after[i][0] : Bodies;
            if (nb[i])
            {
                const auto p = parent[i];
                children[p][childCount[p]++] = i;
            }
            for (unsigned a = 0; a < nb[i]; ++a)
            {
                const auto b = after[i][a];
                predecessors[b][predCount[b]++] = {static_cast<unsigned char>(i), static_cast<unsigned char>(a)};
            }
        }

        for (unsigned i = 0; i < n; ++i)
        {
            backWork[i] = 2 + nb[i];
            for (unsigned j = 0; j < childCount[i]; ++j)
                backWork[i] += backWork[children[i][j]];
            std::sort(children[i], children[i] + childCount[i],
                      [&](unsigned a, unsigned b) { return backWork[a] > backWork[b]; });
        }
        forkMask = 0;
        for (unsigned slot = 0; slot < 2; ++slot)
        {
            unsigned best = Bodies, most = 0;
            for (unsigned i = 0; i < n; ++i)
                if (childCount[i] > 1 && !(forkMask & (uint32_t(1) << i)))
                {
                    const unsigned siblingWork = backWork[i] - backWork[children[i][0]] - (2 + nb[i]);
                    if (siblingWork > most)
                    {
                        most = siblingWork;
                        best = i;
                    }
                }
            if (best == Bodies || most < 4)
                break;
            forkMask |= uint32_t(1) << best;
            forkTicket[best] = static_cast<unsigned char>(forkCount++);
        }
        unsigned cursor = 0;
        for (unsigned i = 0; i < n; ++i)
            if (parent[i] == Bodies)
                postorder(i, cursor);
    }
    void postorder(unsigned i, unsigned &cursor)
    {
        for (unsigned j = 0; j < childCount[i]; ++j)
            postorder(children[i][j], cursor);
        post[cursor++] = i;
    }
    bool splitBack(unsigned i) const
    {
        return (forkMask & (uint32_t(1) << i)) != 0;
    }
    void resetTasks()
    {
        failed.store(false, std::memory_order_relaxed);
        for (unsigned i = 0; i < n; ++i)
            pending[i].value.store(childCount[i] + 1, std::memory_order_relaxed);
    }
    bool arrive(unsigned i)
    {
        return pending[i].value.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }
    void clearColumn(unsigned i)
    {
        for (auto &row : numeric[base[i]].h)
            for (double &x : row)
                x = 0;
        for (unsigned a = 0; a < nb[i]; ++a)
        {
            for (auto &row : numeric[base[i] + a + 1].h)
                for (double &x : row)
                    x = 0;
        }
        for (unsigned j = rawBegin[i]; j < rawBegin[i + 1]; ++j)
            peerData[j] = Contribution{};
    }
    bool factorColumn(unsigned i)
    {
        auto &diag = numeric[base[i]].h;
        for (unsigned d = 0; d < 6; ++d)
            v[i][d] = -g[i][d];
        for (unsigned p = 0; p < predCount[i]; ++p)
        {
            const auto src = predecessors[i][p];
            const unsigned k = src.pivot;
            const auto raw = rawSlot[edgeOffset[k] + src.slot];
            const auto &t = numeric[base[k] + src.slot + 1].h;
            for (unsigned d = 0; d < 6; ++d)
            {

                for (unsigned a = 0; a < 6; ++a)
                    v[i][d] -= t[d][a] * v[k][a];
            }
            if (raw != 65535)
            {
                const auto &peer = peerData[raw];
                for (unsigned d = 0; d < 6; ++d)
                {
                    v[i][d] -= peer.g[d];
                    g[i][d] += peer.g[d];
                    for (unsigned e = 0; e < 6; ++e)
                        diag[d][e] += peer.h[d][e];
                }
            }
            Tile transposed;
            for (unsigned d = 0; d < 6; ++d)
                for (unsigned e = 0; e < 6; ++e)
                    transposed[d][e] = t[e][d];
            for (unsigned j = src.slot; j < nb[k]; ++j)
            {
                const auto b = after[k][j];
                auto &dest = at(b, i);
                const auto &u = numeric[base[k] + j + 1].h;
                for (unsigned d = 0; d < 6; ++d)
                    for (unsigned e = 0; e < 6; ++e)
                    {
                        double dot = u[d][0] * transposed[0][e];
                        for (unsigned a = 1; a < 6; ++a)
                            dot += u[d][a] * transposed[a][e];
                        dest[d][e] -= dot;
                    }
            }
        }
        for (unsigned d = 0; d < 6; ++d)
            if (locks[i] & (1u << d))
            {
                v[i][d] = g[i][d] = 0;
                for (unsigned j = 0; j < 6; ++j)
                    diag[d][j] = diag[j][d] = d == j ? 1. : 0.;
            }
        if (!factor(diag, 6))
            return false;
        forward(diag, 6, v[i]);
        for (unsigned j = 0; j < nb[i]; ++j)
            for (auto &row : numeric[base[i] + j + 1].h)
                forward(diag, 6, row);
        return true;
    }
    bool backColumn(unsigned i)
    {
        auto &x = step[order[i]];
        for (unsigned d = 0; d < 6; ++d)
            x[d] = v[i][d];
        for (unsigned j = 0; j < nb[i]; ++j)
        {
            const auto b = after[i][j];
            const auto &t = at(b, i);
            const auto &y = step[order[b]];
            for (unsigned e = 0; e < 6; ++e)
                for (unsigned d = 0; d < 6; ++d)
                    x[d] -= t[e][d] * y[e];
        }
        backward(numeric[base[i]].h, 6, x);
        for (double a : x)
            if (!std::isfinite(a))
                return false;
        return true;
    }

    template <class T> void cross(const unsigned id[2], const T (&j)[2][6], double k)
    {
        const unsigned r = perm[id[0]] > perm[id[1]] ? 0 : 1;
        Tile &h = at(perm[id[r]], perm[id[1 - r]]);
        for (unsigned d = 0; d < 6; ++d)
            for (unsigned e = 0; e < 6; ++e)
                h[d][e] += k * j[r][d] * j[1 - r][e];
    }
    double model(const double (&delta)[Bodies][6], double &slope) const
    {
        double q = 0;
        slope = 0;
        for (unsigned i = 0; i < n; ++i)
        {
            const auto &x = delta[order[i]];
            const auto &diag = numeric[base[i]].h;
            for (unsigned d = 0; d < 6; ++d)
            {
                slope += g[i][d] * x[d];
                double y = x[d] / diag[d][d];
                for (unsigned e = d + 1; e < 6; ++e)
                    y += diag[e][d] * x[e];
                for (unsigned j = 0; j < nb[i]; ++j)
                {
                    const auto b = after[i][j];
                    for (unsigned e = 0; e < 6; ++e)
                        y += at(b, i)[e][d] * delta[order[b]][e];
                }
                q += y * y;
            }
        }
        return slope + .5 * q;
    }
    enum
    {
        Leaves = 8
    };
    struct RowSample
    {
        unsigned n, ids[2], edge;
    };
    physx::PxArray<RowSample> samples;
    physx::PxArray<unsigned> incidence[Bodies];
    physx::PxTransform fullPose[Bodies];
    double fullDelta[Bodies][6], fullSlope[Bodies], fullQuadratic[Bodies];
    float fullStep[Bodies];
    bool fullSane[Bodies];
    physx::PxArray<unsigned> geometryRows;
    bool wholeLoopJob = false;
    unsigned queueBase = 0;
    double energy[Bodies];
    bool scalarColumn[Bodies];
};
} // namespace Dy
} // namespace physx
