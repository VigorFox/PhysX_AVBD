// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
static bool isLoopEquality(const AvbdRowConstRef &row)
{
    return row.header().kernelId == eAVBD_ROW_KERNEL_CONSTRAINT && !row.header().supportCount &&
           !(row.header().flags &
             (eAVBD_ROW_DRIVE | eAVBD_ROW_CONTACT | eAVBD_ROW_UNILATERAL | eAVBD_ROW_INACTIVE)) &&
           row.scalar().minLambda <= -PX_MAX_F32 && row.scalar().maxLambda >= PX_MAX_F32;
}

bool AvbdDynamicsContext::compileRigidLoops(const PxArray<PxU8> &writable, const PxArray<PxU32> &seedRows,
                                            PxArray<PxU32> &scores)
{
    PxArray<PxU32> refs(mBodies.size(), 0);
    for (PxU32 i = 0; i < mRigidGroups.size(); ++i)
    {
        auto &a = mRigidGroups[i];
        if (!writable[i] || !a.count || a.count > 4)
            continue;
        for (unsigned j = 0; j < a.count; ++j)
        {
            const auto b = a.blocks[j];
            refs[i] += mRowProgram.blockOffsets[b + 1] - mRowProgram.blockOffsets[b];
        }
        if (refs[i] > 256)
            continue;
        a.leafCount = 1;
        a.cuts[0] = 0;
        a.cuts[1] = PxU8(a.count);
    }
    for (PxU32 r : seedRows)
    {
        auto row = mRows[r];
        if (!isLoopEquality(row))
            continue;
        auto u = row.header().bodyA, v = row.header().bodyB;
        if (u >= writable.size() || v >= writable.size() || !writable[u] || !writable[v])
            continue;
        u = mRigidGroupRoots[u];
        v = mRigidGroupRoots[v];
        if (u == v)
            continue;
        if (mRigidGroups[v].count > mRigidGroups[u].count)
            std::swap(u, v);
        auto &a = mRigidGroups[u];
        auto &b = mRigidGroups[v];
        if (!a.leafCount || !b.leafCount || a.leafCount + b.leafCount > AvbdRigidLoop::Leaves ||
            refs[u] + refs[v] > 1024)
            continue;
        for (unsigned j = 1; j <= b.leafCount; ++j)
            a.cuts[a.leafCount + j] = PxU8(a.count + b.cuts[j]);
        for (unsigned j = 0; j < b.count; ++j)
        {
            const auto id = b.blocks[j];
            a.blocks[a.count++] = id;
            mRigidGroupRoots[id] = u;
        }
        a.leafCount += b.leafCount;
        refs[u] += refs[v];
        b.count = 0;
        b.leafCount = 0;
    }
    // Count distinct internal body pairs. Several axes of one hinge
    // represent a single graph edge. Single boundaries retain the
    // existing hard-coordinate leaf operator.
    for (PxU32 i = 0; i < mRigidGroups.size(); ++i)
    {
        auto &a = mRigidGroups[i];
        if (!a.leafCount)
            continue;

        PxU32 adjacent[AvbdRigidLoop::Bodies] = {}, edges = 0;
        bool closed = true;
        for (unsigned x = 0; x < a.count; ++x)
        {
            auto id = a.blocks[x];
            for (auto j = mRowProgram.blockOffsets[id]; j < mRowProgram.blockOffsets[id + 1]; ++j)
            {
                const auto row = mRows[mRowProgram.blockRows[j]];
                const auto &h = row.header();
                const auto other = id == h.bodyA ? h.bodyB : h.bodyA;
                if (h.kernelId == eAVBD_ROW_KERNEL_CONSTRAINT && !(h.flags & eAVBD_ROW_CONTACT) &&
                    other < writable.size() && writable[other] && mRigidGroupRoots[other] != i)
                    closed = false;
                if (!isLoopEquality(row))
                    continue;
                for (unsigned y = 0; y < x; ++y)
                    if (a.blocks[y] == other && !(adjacent[x] & (1u << y)))
                    {
                        adjacent[x] |= 1u << y;
                        ++edges;
                    }
            }
        }
        if (closed && edges >= a.count)
            continue;
        const auto merged = a;
        for (unsigned j = 0; j < merged.count; ++j)
        {
            const auto id = merged.blocks[j];
            auto &singleton = mRigidGroups[id];
            singleton.leafCount = 0;
            singleton.count = 1;
            singleton.blocks[0] = id;
            mRigidGroupRoots[id] = id;
        }
    }
    for (PxU32 root = 0; root < mRigidGroups.size(); ++root)
    {
        auto &a = mRigidGroups[root];
        if (a.leafCount < 2)
            continue;
        const auto initialLeaves = a.leafCount;
        while (a.count < AvbdRigidLoop::Bodies)
        {
            const bool append = a.leafCount > initialLeaves && a.count - a.cuts[a.leafCount - 1] < 4;
            if (!append && a.leafCount == AvbdRigidLoop::Leaves)
                break;
            PxU32 best = PX_MAX_U32, bestScore = 0;
            PxInlineArray<PxU32, 128> touched;
            for (unsigned x = 0; x < a.count; ++x)
            {
                const auto id = a.blocks[x];
                for (auto j = mRowProgram.blockOffsets[id]; j < mRowProgram.blockOffsets[id + 1]; ++j)
                {
                    const auto row = mRows[mRowProgram.blockRows[j]];
                    if (mRowProgram.blockRows[j] >= mNpContactRowCount ||
                        !(row.header().flags & eAVBD_ROW_CONTACT) || (row.header().flags & eAVBD_ROW_INACTIVE))
                        continue;
                    auto b = row.header().bodyA == id ? row.header().bodyB : row.header().bodyA;
                    if (b >= writable.size() || !writable[b] || mRigidGroupRoots[b] != b ||
                        mRigidGroups[b].count != 1 || refs[b] + refs[root] > 1024)
                        continue;
                    if (!scores[b])
                        touched.pushBack(b);
                    ++scores[b];
                }
            }
            for (auto b : touched)
            {
                unsigned score = scores[b];
                scores[b] = 0;
                if (score > bestScore)
                {
                    best = b;
                    bestScore = score;
                }
            }
            if (best == PX_MAX_U32)
                break;
            if (!append)
                ++a.leafCount;
            a.blocks[a.count++] = best;
            a.cuts[a.leafCount] = PxU8(a.count);
            refs[root] += refs[best];
            mRigidGroups[best].count = 0;
            mRigidGroupRoots[best] = root;
        }
    }

    PxU32 slot = 0;
    for (auto &a : mRigidGroups)
        if (a.leafCount && a.count > 1)
        {
            if (slot == mLoopScratch.size())
                mLoopScratch.pushBack(PX_NEW(AvbdRigidLoop)());
            a.loopSlot = slot++;
            for (PxU32 member = 0; member < a.count; ++member)
                scores[a.blocks[member]] = member;
        }
    mStep.iterationLimit = mStep.minimumIterations;
    if (!slot)
        return false;
    for (const auto &row : mRows)
    {
        const auto &h = row.header();
        if ((h.flags & (eAVBD_ROW_INACTIVE | eAVBD_ROW_CONTACT)) == (eAVBD_ROW_INACTIVE | eAVBD_ROW_CONTACT))
            continue;
        PxU32 root = PX_MAX_U32;
        bool outside = false;
        for (PxU32 id : {h.bodyA, h.bodyB})
        {
            if (id >= writable.size() || !writable[id])
                continue;
            const auto owner = mRigidGroupRoots[id];
            if (!mRigidGroups[owner].leafCount)
            {
                outside = true;
                continue;
            }
            if (root != PX_MAX_U32 && owner != root)
                return true;
            root = owner;
        }
        // A separate free contact component retains its own material solve.
        // A joint row or a contact across the block boundary needs the original joint budget.
        if (outside && (!(h.flags & eAVBD_ROW_CONTACT) || root != PX_MAX_U32))
            return true;
    }
    mStep.iterationLimit = 4;
    for (PxU32 id = 0; id < mBodies.size(); ++id)
        if (writable[id] && !mRigidGroups[mRigidGroupRoots[id]].leafCount && mBodies[id].core)
            mStep.iterationLimit =
                PxMax(mStep.iterationLimit, PxU32(mBodies[id].core->solverIterationCounts & 0xff));
    mStep.useJointChebyshev = false;
    return true;
}
void AvbdDynamicsContext::bindRigidLoopRows(PxU32 root, const PxArray<PxU32> &memberIndex)
{
    auto &group = mRigidGroups[root];
    auto &block = *mLoopScratch[group.loopSlot];
    auto &samples = block.samples;
    samples.resize(group.rowCount);
    for (unsigned j = 0; j < group.count; ++j)
        block.incidence[j].clear();
    for (unsigned k = 0; k < group.rowCount; ++k)
    {
        const auto &row = mRows[mRigidGroupRows[group.rowStart + k].row].header();
        auto &sample = samples[k];
        sample.n = 0;
        if (row.bodyA < mRigidGroupRoots.size() && mRigidGroupRoots[row.bodyA] == root)
            sample.ids[sample.n++] = memberIndex[row.bodyA];
        if (row.bodyB != row.bodyA && row.bodyB < mRigidGroupRoots.size() && mRigidGroupRoots[row.bodyB] == root)
            sample.ids[sample.n++] = memberIndex[row.bodyB];
        if (sample.n == 2 && sample.ids[1] < sample.ids[0])
            std::swap(sample.ids[0], sample.ids[1]);
    }
    block.bake(group.count, samples);
    block.geometryRows.clear();
    for (unsigned slot = 0; slot < samples.size(); ++slot)
    {
        auto &row = samples[slot];
        if (!row.n)
        {
            mLocalDomainFailed = 1;
            return;
        }
        auto pivot = block.perm[row.ids[0]];
        if (row.n == 2)
        {
            auto j = block.perm[row.ids[1]];
            if (j < pivot)
                std::swap(pivot, j);
            unsigned edge = 0;
            while (edge < block.nb[pivot] && block.after[pivot][edge] != j)
                ++edge;
            if (edge == block.nb[pivot])
            {
                mLocalDomainFailed = 1;
                return;
            }
            row.edge = block.rawSlot[block.edgeOffset[pivot] + edge];
        }
        block.incidence[pivot].pushBack(slot);
        const auto ri = mRigidGroupRows[group.rowStart + slot].row;
        const auto kernel = mRows[ri].header().kernelId;
        if (kernel == eAVBD_ROW_KERNEL_NP_CONTACT || kernel == eAVBD_ROW_KERNEL_TETRAHEDRON)
            block.geometryRows.pushBack(ri);
    }
}
void AvbdDynamicsContext::bakeLoopWaves()
{
    mLoopOwners.clear();
    mLoopJobs.clear();
    mLoopWaves.clear();
    mLoopOwnerOffsets.clear();
    mLoopColors.clear();
    if (mStep.localBudget)
        return;
    const unsigned workers = mStep.continuation ? PxClamp(mStep.workers, 1u, 32u) : 1u;
    struct Item
    {
        LoopJob job;
        unsigned cost;
    };
    PxArray<Item> items;
    PxArray<unsigned> ordinary;
    mLoopColors.pushBack(0);
    mLoopOwnerOffsets.pushBack(0);
    for (unsigned color = 0; color + 1 < mColorOffsets.size(); ++color)
    {
        ordinary.clear();
        const unsigned first = mLoopOwners.size();
        for (unsigned slot = mColorOffsets[color]; slot < mColorOffsets[color + 1]; ++slot)
        {
            const auto root = mColorBlocks[slot];
            if (root < mRigidGroups.size() && mRigidGroups[root].leafCount && mRigidGroups[root].count > 1)
                mLoopOwners.pushBack(root);
            else
                ordinary.pushBack(root);
        }
        unsigned cursor = first, normal = 0;
        while (first < mLoopOwners.size() && (cursor < mLoopOwners.size() || normal < ordinary.size()))
        {
            LoopWave w = {};
            w.beginOwner = cursor;
            items.clear();
            // Independent same-color roots already fill the workers. Bake complete
            // trees as jobs so their pivots and acceptance need no cross-task join.
            const bool wholeTrees = workers > 1 && mLoopOwners.size() - cursor >= workers;
            const unsigned jobLimit = wholeTrees ? PxMin(PxU32(PX_ARRAY_SIZE(w.start)), 2 * workers) : workers;
            const unsigned stop = PxMin(cursor + (wholeTrees ? jobLimit : 4u), mLoopOwners.size());
            for (; cursor < stop; ++cursor)
            {
                const auto root = mLoopOwners[cursor];
                auto &b = *mLoopScratch[mRigidGroups[root].loopSlot];
                b.wholeLoopJob = wholeTrees;
                if (wholeTrees)
                {
                    items.pushBack({{root, AvbdRigidLoop::Bodies}, 1 + mRigidGroups[root].rowCount});
                    continue;
                }
                for (unsigned j = 0; j < b.n; ++j)
                {
                    const auto i = b.post[j];
                    unsigned updates = 0;
                    for (unsigned k = 0; k < b.predCount[i]; ++k)
                    {
                        auto link = b.predecessors[i][k];
                        updates += b.nb[link.pivot] - link.slot;
                    }
                    items.pushBack({{root, i}, 1 + 3 * b.incidence[i].size() + b.nb[i] + updates});
                }
                w.forks += b.forkCount;
            }
            w.endOwner = stop;
            while (normal < ordinary.size() && items.size() < 128)
            {
                const auto id = ordinary[normal++];
                items.pushBack({{id, PX_MAX_U32}, 1 + localItemRows(id, localItemBlocks(id))});
            }
            w.count = PxMin(jobLimit, items.size());
            unsigned lo = 0, hi = 0;
            for (const auto &item : items)
            {
                lo = PxMax(lo, item.cost);
                hi += item.cost;
            }
            while (lo < hi)
            {
                const auto mid = (lo + hi) / 2;
                unsigned segments = 1, cost = 0;
                for (const auto &item : items)
                {
                    if (cost + item.cost > mid)
                    {
                        ++segments;
                        cost = 0;
                    }
                    cost += item.cost;
                }
                if (segments > w.count)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            unsigned pos = 0;
            for (unsigned k = 0; k < w.count; ++k)
            {
                w.start[k] = mLoopJobs.size();
                unsigned cost = 0;
                do
                {
                    cost += items[pos].cost;
                    mLoopJobs.pushBack(items[pos++].job);
                } while (pos < items.size() - (w.count - 1 - k) && cost + items[pos].cost <= lo);
                w.end[k] = mLoopJobs.size();
            }
            mLoopWaves.pushBack(w);
        }
        mLoopColors.pushBack(mLoopWaves.size());
        mLoopOwnerOffsets.pushBack(mLoopOwners.size());
    }
}

bool AvbdDynamicsContext::dispatchReadyWave(const LoopWave &w)
{
    unsigned limit = w.count - 1; // The advancing worker executes one seed range inline.
    for (unsigned k = w.beginOwner; k < w.endOwner; ++k)
    {
        auto &b = *mLoopScratch[mRigidGroups[mLoopOwners[k]].loopSlot];
        if (b.wholeLoopJob)
            b.failed.store(false, std::memory_order_relaxed);
        else
        {
            b.resetTasks();
            b.queueBase = limit;
            limit += b.forkCount;
        }
    }
    const bool parallel = mStep.continuation && mStep.workers > 1;
    const unsigned refs = w.count + (parallel ? w.forks : 0);
    if (refs == 1)
    {
        runReadySeed(w.start[0], w.end[0], false);
        return true;
    }
    if (!mColorContinuation)
        mColorContinuation = PX_NEW(AvbdAdvanceTask)(*this, mContextID);
    while (mReadyTasks.size() < limit)
        mReadyTasks.pushBack(PX_NEW(AvbdReadyTask)(*this, mContextID));
    mColorContinuation->reserveWave(*mStep.continuation->getTaskManager(), refs);
    mStep.submittedTasks += refs - 1;
    for (unsigned j = 0; j + 1 < w.count; ++j)
        mReadyTasks[j]->submit(*mColorContinuation, PX_MAX_U32, w.start[j], w.end[j]);
    runReadySeed(w.start[w.count - 1], w.end[w.count - 1], true);
    return mColorContinuation->completeInline();
}

void AvbdDynamicsContext::runReadySeed(unsigned first, unsigned end, bool parallel)
{
    for (unsigned j = first; j < end; ++j)
    {
        const auto job = mLoopJobs[j];
        if (job.pivot == PX_MAX_U32)
        {
            solvePrimalBlock(job.root);
            continue;
        }
        auto &b = *mLoopScratch[mRigidGroups[job.root].loopSlot];
        if (job.pivot == AvbdRigidLoop::Bodies)
        {
            for (unsigned walk = 0; walk < b.n; ++walk)
            {
                const unsigned pivot = b.post[walk];
                sampleLoopRows(job.root, pivot);
                if (b.failed.load(std::memory_order_relaxed))
                    break;
                AvbdLocalNonlinearGroupArgs args = {NULL, &b};
                args.pivot = pivot;
                mKernels.launch(AVBD_LOCAL_NONLINEAR_GROUP, &args);
                if (!args.solved)
                {
                    b.failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            if (!b.failed.load(std::memory_order_relaxed))
            {
                for (unsigned pivot = 0; pivot < b.n; ++pivot)
                    if (b.parent[pivot] == AvbdRigidLoop::Bodies)
                        runReadyBack(job.root, pivot, false);
            }
            solvePrimalBlock(job.root);
            continue;
        }
        unsigned i = job.pivot;
        sampleLoopRows(job.root, i);
        // The last arriving child or local assembly performs the pivot inline.
        while (b.arrive(i))
        {
            AvbdLocalNonlinearGroupArgs args = {NULL, &b};
            args.pivot = i;
            if (!b.failed.load(std::memory_order_relaxed))
            {
                mKernels.launch(AVBD_LOCAL_NONLINEAR_GROUP, &args);
                if (!args.solved)
                    b.failed.store(true, std::memory_order_relaxed);
            }
            if (b.parent[i] == AvbdRigidLoop::Bodies)
            {
                runReadyBack(job.root, i, parallel);
                break;
            }
            i = b.parent[i];
        }
    }
}

void AvbdDynamicsContext::runReadySiblings(unsigned root, unsigned i)
{
    const auto &b = *mLoopScratch[mRigidGroups[root].loopSlot];
    for (unsigned k = 1; k < b.childCount[i]; ++k)
        runReadyBack(root, b.children[i][k], true);
}

void AvbdDynamicsContext::runReadyBack(unsigned root, unsigned i, bool parallel)
{
    auto &b = *mLoopScratch[mRigidGroups[root].loopSlot];
    AvbdLocalNonlinearGroupArgs args = {NULL, &b};
    args.pivot = i;
    args.reverse = true;
    // Drain every reserved continuation even when an ancestor could not factor.
    if (!b.failed.load(std::memory_order_relaxed))
    {
        mKernels.launch(AVBD_LOCAL_NONLINEAR_GROUP, &args);
        if (args.solved)
            prepareBackTrial(root, i);
        else
            b.failed.store(true, std::memory_order_relaxed);
    }
    const auto count = b.childCount[i];
    if (parallel && b.splitBack(i))
    {
        mReadyTasks[b.queueBase + b.forkTicket[i]]->submit(*mColorContinuation, root, i, 0);
        runReadyBack(root, b.children[i][0], parallel);
    }
    else
        for (unsigned k = 0; k < count; ++k)
            runReadyBack(root, b.children[i][k], parallel);
}
