// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_SPARSE_SYMMETRIC_H
#define DY_AVBD_SPARSE_SYMMETRIC_H
#include <algorithm>
#include <cmath>
#include <limits>
#include "foundation/PxArray.h"

namespace physx { namespace Dy {
// Derived sparse coefficients only; physical blocks remain Particle3/Rigid6.
template<template<class> class Vector> struct AvbdSparseSymmetricBase
{
    struct Entry { int a,b; double value; };
    int n;
    Vector<Entry> entries;
    Vector<int> permutation,position,offsets,rows,entrySlots;
    Vector<double> values,diagonal,scale,rhs,temporary;
    unsigned factorizations=0,solves=0;
    double smallestPivot=0;
    bool compiled=false,factored=false;
    explicit AvbdSparseSymmetricBase(int size):n(size) {}
    void add(int a,int b,double value){compiled=factored=false;entries.push_back({a,b,value});}
    int slot(int column,int row) const
    {
        const auto begin=rows.begin()+offsets[column],end=rows.begin()+offsets[column+1];
        const auto it=std::lower_bound(begin,end,row);
        return it!=end && *it==row?int(it-rows.begin()):-1;
    }
    bool compile()
    {
        compiled=factored=false;
        if(n<0)return false;
        for(const auto& e:entries)
            if(e.a<0 || e.b<0 || e.a>=n || e.b>=n || !std::isfinite(e.value))return false;
        offsets.clear();rows.clear();entrySlots.clear();
        Vector<Vector<int>> graph(n);
        for(const auto& e:entries)if(e.a!=e.b){graph[e.a].push_back(e.b);graph[e.b].push_back(e.a);}
        for(auto& row:graph){std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());}
        permutation.resize(n);position.assign(n,-1);
        Vector<Vector<int>> columns(n);Vector<int> merged;
        for(int k=0;k<n;++k)
        {
            int best=-1;
            for(int i=0;i<n;++i)if(position[i]<0 && (best<0 || graph[i].size()<graph[best].size()))best=i;
            permutation[k]=best;position[best]=k;
            columns[k]=graph[best];
            // The elimination clique is one sorted union per neighbor, not
            // repeated tree lookups/allocations for every edge. The graph and
            // minimum-degree tie order are unchanged; scratch remains sparse.
            for(int i:columns[k])
            {
                const auto& old=graph[i];const auto& neighbors=columns[k];
                merged.clear();merged.reserve(old.size()+neighbors.size());decltype(old.size()) a=0,b=0;
                while(a<old.size() || b<neighbors.size())
                {
                    const int value=b==neighbors.size() || (a<old.size() && old[a]<neighbors[b])?old[a]:neighbors[b];
                    if(a<old.size() && old[a]==value)++a;if(b<neighbors.size() && neighbors[b]==value)++b;
                    if(value!=i && value!=best)merged.push_back(value);
                }
                graph[i].swap(merged);
            }
            graph[best].clear();
        }
        offsets.push_back(0);
        for(auto& column:columns)
        {
            for(int& row:column)row=position[row];std::sort(column.begin(),column.end());
            rows.insert(rows.end(),column.begin(),column.end());offsets.push_back(int(rows.size()));
        }
        for(const auto& e:entries)
        {
            const int a=position[e.a],b=position[e.b];
            entrySlots.push_back(a==b?-1-a:slot(std::min(a,b),std::max(a,b)));
        }
        values.resize(rows.size());diagonal.resize(n);scale.resize(n);rhs.resize(n);temporary.resize(n);
        compiled=true;return true;
    }
    struct PositivePivot
    {
        void prepare(const Vector<double>&) {}
        bool operator()(int,double& value) const {return value>0;}
    };
    bool factor() {PositivePivot policy;return factor(policy);}
    template<class PivotPolicy> bool factor(PivotPolicy& policy)
    {
        factored=false;if(!compiled)return false;
        ++factorizations;
        std::fill(scale.begin(),scale.end(),1.);
        // Symmetric Ruiz equilibration changes coordinates, not equations.
        for(int pass=0;pass<8;++pass)
        {
            std::fill(temporary.begin(),temporary.end(),0.);
            for(const auto& e:entries)
            {
                const double v=std::abs(e.value*scale[e.a]*scale[e.b]);
                if(!std::isfinite(v))return false;
                temporary[e.a]=std::max(temporary[e.a],v);temporary[e.b]=std::max(temporary[e.b],v);
            }
            for(int i=0;i<n;++i){if(!(temporary[i]>0))return false;scale[i]/=std::sqrt(temporary[i]);}
        }
        std::fill(values.begin(),values.end(),0.);std::fill(diagonal.begin(),diagonal.end(),0.);
        for(decltype(entries.size()) e=0;e<entries.size();++e)
        {
            const Entry& entry=entries[e];const double value=entry.value*scale[entry.a]*scale[entry.b];
            if(entrySlots[e]<0)diagonal[-1-entrySlots[e]]+=value;else values[entrySlots[e]]+=value;
        }
        smallestPivot=std::numeric_limits<double>::infinity();
        policy.prepare(diagonal);
        for(int k=0;k<n;++k)
        {
            double d=diagonal[k];if(!std::isfinite(d))return false;
            smallestPivot=std::min(smallestPivot,std::abs(d));
            if(!policy(k,d) || !std::isfinite(d) || d==0)return false;
            diagonal[k]=d;
            for(int p=offsets[k];p<offsets[k+1];++p)
            {
                const int i=rows[p];const double v=values[p],l=v/d;
                diagonal[i]-=v*l;
                int destination=offsets[i];
                for(int q=p+1;q<offsets[k+1];++q)
                {
                    // Both lists are sorted and symbolic fill guarantees the
                    // destinations. Walk once instead of binary-searching each update.
                    while(destination<offsets[i+1] && rows[destination]<rows[q])++destination;
                    if(destination==offsets[i+1] || rows[destination]!=rows[q])return false;
                    values[destination]-=l*values[q];
                }
                values[p]=l;
            }
        }
        factored=true;return true;
    }
    bool solve(const double* b,double* x)
    {
        if(!factored)return false;
        ++solves;
        for(int k=0;k<n;++k)rhs[k]=b[permutation[k]]*scale[permutation[k]];
        for(int k=0;k<n;++k)for(int p=offsets[k];p<offsets[k+1];++p)rhs[rows[p]]-=values[p]*rhs[k];
        for(int k=0;k<n;++k)rhs[k]/=diagonal[k];
        for(int k=n;k-->0;)for(int p=offsets[k];p<offsets[k+1];++p)rhs[k]-=values[p]*rhs[rows[p]];
        for(int k=0;k<n;++k)
        {x[permutation[k]]=rhs[k]*scale[permutation[k]];if(!std::isfinite(x[permutation[k]]))return false;}
        return true;
    }
    size_t bytes() const
    {
        return entries.size()*sizeof(Entry)+(permutation.size()+position.size()+offsets.size()+rows.size()+entrySlots.size())*sizeof(int)+
            (values.size()+diagonal.size()+scale.size()+rhs.size()+temporary.size())*sizeof(double);
    }
};
// SDK allocation/exception policy; the lab instantiates the SAME elimination
// with std::vector, without linking another engine or allocator runtime.
template<class T> struct AvbdSparseArray : PxArray<T>
{
    using PxArray<T>::PxArray;
    void push_back(const T& v){this->pushBack(v);}
    void assign(PxU32 n,const T& v){this->resize(n);std::fill(this->begin(),this->end(),v);}
    void erase(T* first,T* last){this->removeRange(PxU32(first-this->begin()),PxU32(last-first));}
    void insert(T*,const T* first,const T* last){for(;first!=last;++first)this->pushBack(*first);}
};
struct AvbdSparseSymmetric : AvbdSparseSymmetricBase<AvbdSparseArray>
{
    explicit AvbdSparseSymmetric(int n):AvbdSparseSymmetricBase(n){}
};
} }
#endif
