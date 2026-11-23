// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_ROW_PROGRAM_H
#define DY_AVBD_ROW_PROGRAM_H

#include "foundation/PxArray.h"
#include "foundation/PxMath.h"

namespace physx
{
namespace Dy
{

// One topology for every row kernel and block dimension. No physical state is
// cached here. Read dependencies include zero-gradient and read-only supports.
// Both orientations are CSR: storage is linear in distinct declared supports.
class AvbdRowProgram
{
    PxArray<PxU32> mVisited, mCursors;
    static PxU32 root(PxArray<PxU32>& parents, PxU32 block)
    {
        PxU32 result = block;
        while(parents[result] != result) result = parents[result];
        while(parents[block] != block)
        {
            const PxU32 next = parents[block];
            parents[block] = result;
            block = next;
        }
        return result;
    }

public:
    // Ordered immutable row ranges. An asset revision identifies its complete
    // local support topology. Block bases can change on island reassembly.
    struct PrefixPart
    {
        PxU64 revision;
        PxU32 firstBlock, rows;
        bool operator==(const PrefixPart& p) const
        {return revision==p.revision && firstBlock==p.firstBlock && rows==p.rows;}
    };
    // Compact operation selections, not another constraint representation.
    // Consecutive source rows share one descriptor; no state is cached.
    struct Ranges
    {
        struct Span {PxU32 first,count,offset;};
        PxArray<Span> spans;
        PxU32 rows=0;
        void clear(){spans.clear();rows=0;}
        void copyPrefix(const Ranges& source,PxU32 end)
        {
            clear();
            for(auto span:source.spans)
            {
                if(span.first>=end)break;
                span.count=PxMin(span.count,end-span.first);
                spans.pushBack(span);rows+=span.count;
            }
        }
        void append(PxU32 row)
        {
            if(!spans.empty() && spans.back().first+spans.back().count==row)++spans.back().count;
            else spans.pushBack({row,1,rows});
            ++rows;
        }
        PxU32 locate(PxU32 slot,PxU32& first) const
        {
            PX_ASSERT(slot<rows);
            PxU32 lo=0,hi=spans.size();
            while(hi-lo>1)
            {
                const PxU32 mid=lo+(hi-lo)/2;
                if(spans[mid].offset<=slot)lo=mid;else hi=mid;
            }
            const auto& span=spans[lo];const PxU32 used=slot-span.offset;
            first=span.first+used;return span.count-used;
        }
    };
    Ranges dualRows,historyRows;
    PxArray<PxU32> rowOffsets, rowBlocks, blockOffsets, blockRows;

private:
    PxArray<PrefixPart> mParts;
    PxArray<PxU32> mPrefixOffsets,mPrefixRows;
    Ranges mPrefixDual,mPrefixHistory;
    PxU32 mReusedRows=0;
    bool reject()
    {
        mParts.clear();mReusedRows=0;
        rowOffsets.clear();rowBlocks.clear();blockOffsets.clear();blockRows.clear();
        dualRows.clear();historyRows.clear();
        return false;
    }
public:
    PxU32 reusedRows() const {return mReusedRows;}
    PxU64 prefixBytes() const
    {
        return PxU64(sizeof(PxU32))*(mPrefixOffsets.size()+mPrefixRows.size())+PxU64(sizeof(PrefixPart))*mParts.size()+
            PxU64(sizeof(Ranges::Span))*(mPrefixDual.spans.size()+mPrefixHistory.spans.size());
    }
    // enumerate(row, emit) returns false for an invalid source range. Only
    // PX_MAX_U32 denotes the world; an out-of-range live block is an error.
    template<class Enumerate>
    bool compile(PxU32 rows, PxU32 blocks, const Enumerate& enumerate,
                 const PxArray<PrefixPart>* fixed=NULL)
    {
        const PxU32 partCount=fixed?fixed->size():0;
        mReusedRows=0;
        bool reuse=partCount && partCount==mParts.size() && mPrefixOffsets.size()==blocks+1;
        PxU32 prefix=0;
        for(PxU32 i=0;i<partCount;++i)
        {
            const auto& part=(*fixed)[i];
            if(part.rows>rows-prefix)return reject();
            prefix+=part.rows;
            reuse=reuse && mParts[i]==part;
        }
        if(reuse)mReusedRows=prefix;
        if(reuse && prefix==rows && rowOffsets.size()==rows+1)return true;
        blockOffsets.resize(blocks + 1);
        if(reuse)
        {
            // Forward indices of the prefix already occupy the final storage.
            rowOffsets.forceSize_Unsafe(prefix+1);
            rowBlocks.forceSize_Unsafe(rowOffsets.back());
            blockOffsets[0]=0;
            for(PxU32 b=0;b<blocks;++b)blockOffsets[b+1]=mPrefixOffsets[b+1]-mPrefixOffsets[b];
            dualRows=mPrefixDual;historyRows=mPrefixHistory;
        }
        else
        {
            rowOffsets.clear();rowBlocks.clear();dualRows.clear();historyRows.clear();
            mParts.clear();
            PxMemZero(blockOffsets.begin(),size_t(blocks+1)*sizeof(PxU32));
            rowOffsets.pushBack(0);
        }
        mVisited.resize(blocks);
        if(blocks)PxMemSet(mVisited.begin(),0xff,size_t(blocks)*sizeof(PxU32));
        const PxU32 first=reuse?prefix:0;
        bool valid = true;
        for(PxU32 r = first; r < rows && valid; ++r)
        {
            const bool sourceValid = enumerate(r, [&](PxU32 b)
            {
                if(b == PX_MAX_U32) return;
                if(b >= blocks) { valid = false; return; }
                if(mVisited[b] == r) return;
                mVisited[b] = r;
                rowBlocks.pushBack(b);
                ++blockOffsets[b + 1];
            });
            valid = valid && sourceValid;
            rowOffsets.pushBack(rowBlocks.size());
        }
        if(!valid)return reject();
        for(PxU32 b = 1; b <= blocks; ++b) blockOffsets[b] += blockOffsets[b - 1];
        mCursors.resize(blocks);
        blockRows.resizeUninitialized(rowBlocks.size());
        if(reuse)
        {
            for(PxU32 b=0;b<blocks;++b)
            {
                const PxU32 count=mPrefixOffsets[b+1]-mPrefixOffsets[b];
                if(count)PxMemCopy(blockRows.begin()+blockOffsets[b],mPrefixRows.begin()+mPrefixOffsets[b],size_t(count)*sizeof(PxU32));
                mCursors[b]=blockOffsets[b]+count;
            }
        }
        else if(blocks)PxMemCopy(mCursors.begin(),blockOffsets.begin(),size_t(blocks)*sizeof(PxU32));
        for(PxU32 r = first; r < rows; ++r)
            for(PxU32 s = rowOffsets[r]; s < rowOffsets[r + 1]; ++s)
                blockRows[mCursors[rowBlocks[s]]++] = r;
        if(!reuse && prefix)
        {
            mPrefixOffsets.resize(blocks+1);mPrefixRows.resizeUninitialized(rowOffsets[prefix]);
            PxU32 cursor=0;
            for(PxU32 b=0;b<blocks;++b)
            {
                mPrefixOffsets[b]=cursor;
                for(PxU32 s=blockOffsets[b];s<blockOffsets[b+1] && blockRows[s]<prefix;++s)
                    mPrefixRows[cursor++]=blockRows[s];
            }
            mPrefixOffsets[blocks]=cursor;
            mPrefixDual.copyPrefix(dualRows,prefix);mPrefixHistory.copyPrefix(historyRows,prefix);
            mParts=*fixed;
        }
        return true;
    }

    template<class Visit>
    void forEachBlock(PxU32 row, const Visit& visit) const
    {
        for(PxU32 s = rowOffsets[row]; s < rowOffsets[row + 1]; ++s)
            visit(rowBlocks[s]);
    }

    // Static/kinematic supports are inputs, not bridges between responding
    // components. Inactive rows retain topology but contribute no coupling.
    template<class Responds, class Active>
    void components(PxArray<PxU32>& parents, const Responds& responds,
        const Active& active) const
    {
        const PxU32 count = blockOffsets.empty() ? 0 : blockOffsets.size() - 1;
        parents.resize(count);
        for(PxU32 b = 0; b < count; ++b) parents[b] = b;
        for(PxU32 r = 0; r + 1 < rowOffsets.size(); ++r)
        {
            if(!active(r)) continue;
            PxU32 first = PX_MAX_U32;
            forEachBlock(r, [&](PxU32 b)
            {
                if(!responds(b)) return;
                const PxU32 rb = root(parents, b);
                if(first == PX_MAX_U32) first = rb;
                else parents[rb] = first;
            });
        }
        for(PxU32 b = 0; b < count; ++b) parents[b] = root(parents, b);
    }

    PxU64 bytes() const
    {
        return PxU64(sizeof(PxU32)) * (rowOffsets.size() + rowBlocks.size() +
            blockOffsets.size() + blockRows.size() + mVisited.size() + mCursors.size())+
            PxU64(sizeof(Ranges::Span))*(dualRows.spans.size()+historyRows.spans.size())+prefixBytes();
    }
};
}
}
#endif
