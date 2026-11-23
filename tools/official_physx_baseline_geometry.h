// Read-only geometry observer for the unmodified official PhysX baseline.
// Measures individual collision-tetrahedron intersections across bodies.
// maxTetMtd is NOT the minimum translation needed to separate soft-body unions.
#ifndef OFFICIAL_PHYSX_BASELINE_GEOMETRY_H
#define OFFICIAL_PHYSX_BASELINE_GEOMETRY_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace OfficialPhysxBaselineGeometry
{
struct Vec3
{
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
    Vec3 operator-(const Vec3& b) const { return Vec3(x-b.x, y-b.y, z-b.z); }
    double dot(const Vec3& b) const { return x*b.x+y*b.y+z*b.z; }
    Vec3 cross(const Vec3& b) const
    { return Vec3(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); }
};

// Supports PxVec4/float3 positions and either SDK index format without copying
// or including any SDK headers. The first three floats are the actual published
// collision position. The fourth float (inverse mass) is never interpreted.
struct BodyView
{
    const void* positions;
    std::size_t vertexCount;
    std::size_t positionStrideBytes;
    const void* tetrahedra;
    std::size_t tetrahedronCount;
    std::size_t tetrahedronStrideBytes;
    bool indices16Bit;
    BodyView() : positions(NULL), vertexCount(0), positionStrideBytes(16),
        tetrahedra(NULL), tetrahedronCount(0), tetrahedronStrideBytes(0), indices16Bit(false) {}
};

struct Report
{
    std::uint64_t candidateBodyPairs, candidateTetPairs, overlappingTetPairs;
    std::uint64_t validTetrahedra, degenerateTetrahedra, invalidTetrahedra;
    double maxTetMtd;
    std::size_t maxBodyA, maxBodyB, maxTetA, maxTetB;
    bool valid;
    Report() : candidateBodyPairs(0), candidateTetPairs(0), overlappingTetPairs(0),
        validTetrahedra(0), degenerateTetrahedra(0), invalidTetrahedra(0), maxTetMtd(0),
        maxBodyA(0), maxBodyB(0), maxTetA(0), maxTetB(0), valid(true) {}
};

namespace Detail
{
struct Bounds
{
    Vec3 lo, hi;
    Bounds() : lo(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()), hi(-lo.x,-lo.y,-lo.z) {}
    void include(const Vec3& p)
    {
        lo=Vec3(std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z));
        hi=Vec3(std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z));
    }
    bool overlaps(const Bounds& b) const
    { return lo.x<b.hi.x && hi.x>b.lo.x && lo.y<b.hi.y && hi.y>b.lo.y && lo.z<b.hi.z && hi.z>b.lo.z; }
};
struct Tet { Vec3 p[4]; Bounds bounds; std::size_t index; };
struct Body { std::vector<Tet> tets; Bounds bounds; };

inline bool axisDepth(const Tet& a, const Tet& b, const Vec3& n, double& depth)
{
    const double length2=n.dot(n);
    if(length2==0) return true;
    double amin=0,amax=0,bmin=std::numeric_limits<double>::infinity(),bmax=-bmin;
    // Relative coordinates avoid cancellation under large common translations.
    for(unsigned i=0;i<4;++i)
    {
        const double av=(a.p[i]-a.p[0]).dot(n),bv=(b.p[i]-a.p[0]).dot(n);
        amin=std::min(amin,av);amax=std::max(amax,av);bmin=std::min(bmin,bv);bmax=std::max(bmax,bv);
    }
    const double overlap=std::min(amax-bmin,bmax-amin);
    if(overlap<=0) return false;
    depth=std::min(depth,overlap/std::sqrt(length2));
    return true;
}

inline double tetDepth(const Tet& a, const Tet& b)
{
    static const unsigned edge[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    static const unsigned face[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
    double depth=std::numeric_limits<double>::infinity();
    for(unsigned i=0;i<4;++i)
    {
        const unsigned* f=face[i];
        if(!axisDepth(a,b,(a.p[f[1]]-a.p[f[0]]).cross(a.p[f[2]]-a.p[f[0]]),depth)) return 0;
        if(!axisDepth(a,b,(b.p[f[1]]-b.p[f[0]]).cross(b.p[f[2]]-b.p[f[0]]),depth)) return 0;
    }
    for(unsigned i=0;i<6;++i) for(unsigned j=0;j<6;++j)
        if(!axisDepth(a,b,(a.p[edge[i][1]]-a.p[edge[i][0]]).cross(b.p[edge[j][1]]-b.p[edge[j][0]]),depth)) return 0;
    return depth;
}

inline Body prepare(const BodyView& view, Report& report)
{
    Body out;
    const std::size_t indexBytes=view.indices16Bit?sizeof(std::uint16_t):sizeof(std::uint32_t);
    const std::size_t indexStride=view.tetrahedronStrideBytes?view.tetrahedronStrideBytes:4*indexBytes;
    if(view.tetrahedronCount && (!view.positions || !view.tetrahedra || view.positionStrideBytes<3*sizeof(float) || indexStride<4*indexBytes))
    { report.valid=false;report.invalidTetrahedra+=view.tetrahedronCount;return out; }
    out.tets.reserve(view.tetrahedronCount);
    for(std::size_t i=0;i<view.tetrahedronCount;++i)
    {
        Tet tet;tet.index=i;bool valid=true;
        const unsigned char* ids=static_cast<const unsigned char*>(view.tetrahedra)+i*indexStride;
        for(unsigned k=0;k<4;++k)
        {
            std::uint32_t index=0;
            if(view.indices16Bit) { std::uint16_t small;std::memcpy(&small,ids+k*indexBytes,indexBytes);index=small; }
            else std::memcpy(&index,ids+k*indexBytes,indexBytes);
            if(index>=view.vertexCount) { valid=false;break; }
            float p[3];std::memcpy(p,static_cast<const unsigned char*>(view.positions)+index*view.positionStrideBytes,sizeof(p));
            if(!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) { valid=false;break; }
            tet.p[k]=Vec3(p[0],p[1],p[2]);tet.bounds.include(tet.p[k]);
        }
        if(!valid) { ++report.invalidTetrahedra;report.valid=false;continue; }
        const double det=(tet.p[1]-tet.p[0]).dot((tet.p[2]-tet.p[0]).cross(tet.p[3]-tet.p[0]));
        if(det==0) { ++report.degenerateTetrahedra;report.valid=false;continue; }
        ++report.validTetrahedra;out.bounds.include(tet.bounds.lo);out.bounds.include(tet.bounds.hi);out.tets.push_back(tet);
    }
    std::sort(out.tets.begin(),out.tets.end(),[](const Tet& a,const Tet& b)
    { return a.bounds.lo.x<b.bounds.lo.x || (a.bounds.lo.x==b.bounds.lo.x && a.index<b.index); });
    return out;
}
}

// AABB filtering is only a broadphase. All reported intersections pass the
// complete tetrahedron SAT (eight face normals and 36 edge cross products).
// Same-body intersections and ground-plane depth are intentionally separate
// measurements: no adjacency or material filter is guessed in this observer.
inline Report observe(const BodyView* views, std::size_t count)
{
    Report report;
    if(count && !views) { report.valid=false;return report; }
    std::vector<Detail::Body> bodies; bodies.reserve(count);
    for(std::size_t i=0;i<count;++i) bodies.push_back(Detail::prepare(views[i],report));
    for(std::size_t a=0;a<count;++a) for(std::size_t b=a+1;b<count;++b)
    {
        const Detail::Body& first=bodies[a];const Detail::Body& second=bodies[b];
        if(first.tets.empty() || second.tets.empty() || !first.bounds.overlaps(second.bounds)) continue;
        ++report.candidateBodyPairs;
        for(std::size_t i=0;i<first.tets.size();++i)
        {
            const Detail::Tet& ta=first.tets[i];
            for(std::size_t j=0;j<second.tets.size() && second.tets[j].bounds.lo.x<ta.bounds.hi.x;++j)
            {
                const Detail::Tet& tb=second.tets[j];
                if(!ta.bounds.overlaps(tb.bounds)) continue;
                ++report.candidateTetPairs;
                const double depth=Detail::tetDepth(ta,tb);
                if(depth<=0) continue;
                ++report.overlappingTetPairs;
                if(depth>report.maxTetMtd)
                { report.maxTetMtd=depth;report.maxBodyA=a;report.maxBodyB=b;report.maxTetA=ta.index;report.maxTetB=tb.index; }
            }
        }
    }
    return report;
}
}
#endif
