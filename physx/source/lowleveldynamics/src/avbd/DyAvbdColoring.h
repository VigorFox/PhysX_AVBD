// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_COLORING_H
#define DY_AVBD_COLORING_H
#include "foundation/PxArray.h"
#include "foundation/PxMath.h"
#include "avbd/DyAvbdRowProgram.h"

namespace physx
{
namespace Dy
{

// Execution metadata only: each scope lists blocks with conflicting accesses.
// A reorderable prefix uses deterministic greedy colors; the remaining blocks
// retain their reference conflict order, after every conflicting prefix block.
// Both parts produce one conflict-free, in-place multi-color GS program.
// Incidence storage is linear in authored supports, not quadratic in bodies.
class AvbdBlockColoring
{
	PxArray<PxU32> mStarts, mBlocks;
	struct Adjacency
	{
		const PxU32* offsets;
		const PxU32* neighbors;
		PxU32 start, count;
	};
	PxArray<Adjacency> mAdjacency;
	const AvbdRowProgram* mProgram;

  public:
	AvbdBlockColoring() : mProgram(NULL) { mStarts.pushBack(0); }
	// Borrow the canonical incidence for the duration of this compilation.
	// Material/OGC scopes below supplement the same graph; row supports are not copied.
	void addProgram(const AvbdRowProgram& program) { mProgram = &program; }
	void addAdjacency(const PxU32* offsets, const PxU32* neighbors,
		PxU32 start, PxU32 count)
	{
		Adjacency view={offsets,neighbors,start,count};
		mAdjacency.pushBack(view);
	}
	void addScope(const PxU32 *blocks, PxU32 count, const PxArray<PxU8> &writable)
	{
		for (PxU32 i = 0; i < count; ++i)
			if (blocks[i] < writable.size() && writable[blocks[i]])
				mBlocks.pushBack(blocks[i]);
		mStarts.pushBack(mBlocks.size());
	}
	void build(const PxArray<PxU32> &order, PxU32 blockCount, PxArray<PxU32> &offsets,
	           PxArray<PxU32> &blocks, PxU32 reorderableCount = 0) const
	{
		PxArray<PxU32> incidenceOffsets(blockCount + 1, 0);
		for (PxU32 i = 0; i < mBlocks.size(); ++i)
			++incidenceOffsets[mBlocks[i] + 1];
		for (PxU32 i = 1; i < incidenceOffsets.size(); ++i)
			incidenceOffsets[i] += incidenceOffsets[i - 1];
		PxArray<PxU32> cursors(incidenceOffsets), refs(mBlocks.size());
		for (PxU32 scope = 0; scope + 1 < mStarts.size(); ++scope)
			for (PxU32 i = mStarts[scope]; i < mStarts[scope + 1]; ++i)
				refs[cursors[mBlocks[i]]++] = scope;
		PxArray<PxU32> frontier(mStarts.size(), 0), colors(blockCount, PX_MAX_U32);
		PxArray<PxU32> rowFrontier(mProgram ? mProgram->rowOffsets.size() : 0, 0);
		PxArray<PxU32> used(blockCount, PX_MAX_U32);
		PxU32 colorCount = 0;
		for (PxU32 i = 0; i < order.size(); ++i)
		{
			const PxU32 block = order[i];
			PxU32 color = 0;
			if(mProgram)
				for(PxU32 ref = mProgram->blockOffsets[block]; ref < mProgram->blockOffsets[block + 1]; ++ref)
				{
					const PxU32 row = mProgram->blockRows[ref];
					if(i < reorderableCount)
						mProgram->forEachBlock(row, [&](PxU32 neighbor)
						{ if(colors[neighbor] != PX_MAX_U32) used[colors[neighbor]] = i; });
					else color = PxMax(color, rowFrontier[row]);
				}
			// Rest-material edges are compiled with the mesh and borrowed here.
			// Dynamic contact scopes still participate in the same color decision.
			for(PxU32 a = 0; a < mAdjacency.size(); ++a)
			{
				const Adjacency& view=mAdjacency[a];
				if(block < view.start || block-view.start >= view.count) continue;
				const PxU32 local=block-view.start;
				for(PxU32 n=view.offsets[local]; n<view.offsets[local+1]; ++n)
				{
					const PxU32 neighbor=view.start+view.neighbors[n];
					if(neighbor >= blockCount || colors[neighbor] == PX_MAX_U32) continue;
					if(i < reorderableCount) used[colors[neighbor]]=i;
					else color=PxMax(color,colors[neighbor]+1);
				}
			}
			if (i < reorderableCount)
			{
				for (PxU32 r = incidenceOffsets[block]; r < incidenceOffsets[block + 1]; ++r)
					for (PxU32 s = mStarts[refs[r]]; s < mStarts[refs[r] + 1]; ++s)
					{
						const PxU32 neighborColor = colors[mBlocks[s]];
						if (neighborColor != PX_MAX_U32) used[neighborColor] = i;
					}
				while (color < blockCount && used[color] == i) ++color;
			}
			else
				for (PxU32 r = incidenceOffsets[block]; r < incidenceOffsets[block + 1]; ++r)
					color = PxMax(color, frontier[refs[r]]);
			colors[block] = color;
			colorCount = PxMax(colorCount, color + 1);
			if(mProgram)
				for(PxU32 ref = mProgram->blockOffsets[block]; ref < mProgram->blockOffsets[block + 1]; ++ref)
					rowFrontier[mProgram->blockRows[ref]] = PxMax(rowFrontier[mProgram->blockRows[ref]], color + 1);
			for (PxU32 r = incidenceOffsets[block]; r < incidenceOffsets[block + 1]; ++r)
				frontier[refs[r]] = PxMax(frontier[refs[r]], color + 1);
		}
		offsets.clear();
		offsets.resize(colorCount + 1, 0);
		for (PxU32 i = 0; i < order.size(); ++i)
			++offsets[colors[order[i]] + 1];
		for (PxU32 i = 1; i < offsets.size(); ++i)
			offsets[i] += offsets[i - 1];
		cursors = offsets;
		blocks.resize(order.size());
		for (PxU32 i = 0; i < order.size(); ++i)
			blocks[cursors[colors[order[i]]]++] = order[i];
	}
};
} // namespace Dy
} // namespace physx
#endif
