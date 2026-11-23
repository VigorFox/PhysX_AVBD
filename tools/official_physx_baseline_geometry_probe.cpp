// Standalone analytic checks for the read-only official-baseline observer.
#include "official_physx_baseline_geometry.h"
#include <cstdio>
using namespace OfficialPhysxBaselineGeometry;
struct Point { float x,y,z,w; };
static BodyView view(const Point* p,std::size_t n,const std::uint32_t* t,std::size_t count)
{ BodyView b;b.positions=p;b.vertexCount=n;b.tetrahedra=t;b.tetrahedronCount=count;return b; }
static unsigned failures=0,checks=0;
static void check(bool condition,const char* label)
{ ++checks;if(!condition) { ++failures;std::printf("FAIL %s\n",label); } }
int main()
{
    const std::uint32_t tet[]={0,1,2,3};
    Point a[4]={{0,0,0,1},{1,0,0,1},{0,1,0,1},{0,0,1,1}};
    Point b[4]={{1,1,1,1},{0,1,1,1},{1,0,1,1},{1,1,0,1}};
    BodyView pair[]={view(a,4,tet,1),view(b,4,tet,1)};
    Report r=observe(pair,2);
    check(r.valid && r.candidateTetPairs==1 && r.overlappingTetPairs==0,"overlapping AABBs are not penetration");
    for(unsigned i=0;i<4;++i) { b[i]=a[i];b[i].x+=0.25f; }
    r=observe(pair,2);
    check(r.valid && r.overlappingTetPairs==1 && std::abs(r.maxTetMtd-0.75/std::sqrt(3.0))<1.e-12,"analytic tetrahedron MTD");
    const double depth=r.maxTetMtd;
    for(unsigned i=0;i<4;++i)
    {
        const Point pa=a[i],pb=b[i];
        a[i]={-pa.y+4,pa.z-8,-pa.x+2,1};b[i]={-pb.y+4,pb.z-8,-pb.x+2,1};
    }
    r=observe(pair,2);check(r.valid && std::abs(r.maxTetMtd-depth)<1.e-12,"common rigid transform");
    const std::uint16_t small[]={0,1,2,3};pair[1].tetrahedra=small;pair[1].indices16Bit=true;
    r=observe(pair,2);check(r.valid && std::abs(r.maxTetMtd-depth)<1.e-12,"16 bit indices");
    pair[1].vertexCount=3;r=observe(pair,2);check(!r.valid && r.invalidTetrahedra==1,"invalid index is reported");
    Point cubeA[8],cubeB[8];for(unsigned i=0;i<8;++i) cubeA[i]={float(i&1),float((i>>1)&1),float((i>>2)&1),1};
    const unsigned permutations[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    std::uint32_t cubeTets[24];for(unsigned i=0;i<6;++i)
    { cubeTets[4*i]=0;cubeTets[4*i+1]=1u<<permutations[i][0];cubeTets[4*i+2]=cubeTets[4*i+1]|(1u<<permutations[i][1]);cubeTets[4*i+3]=7; }
    pair[0]=view(cubeA,8,cubeTets,6);pair[1]=view(cubeB,8,cubeTets,6);
    for(unsigned i=0;i<8;++i) { cubeB[i]=cubeA[i];cubeB[i].x+=1.25f; }
    r=observe(pair,2);check(r.valid && r.overlappingTetPairs==0,"tet-mesh boxes separated");
    for(unsigned i=0;i<8;++i) { cubeB[i]=cubeA[i];cubeB[i].x+=1.f; }
    r=observe(pair,2);check(r.valid && r.overlappingTetPairs==0,"tet-mesh boxes touching");
    for(unsigned i=0;i<8;++i) { cubeB[i]=cubeA[i];cubeB[i].x+=0.875f; }
    r=observe(pair,2);const Report box=r;
    check(r.valid && r.overlappingTetPairs>0 && r.maxTetMtd>0 && r.maxTetMtd<=0.125,"tet-level MTD bounded by analytic box MTD");
    for(unsigned i=0;i<8;++i)
    { const Point pa=cubeA[i],pb=cubeB[i];cubeA[i]={pa.z+4,-pa.x+8,-pa.y+2,1};cubeB[i]={pb.z+4,-pb.x+8,-pb.y+2,1}; }
    r=observe(pair,2);check(r.valid && r.overlappingTetPairs==box.overlappingTetPairs && std::abs(r.maxTetMtd-box.maxTetMtd)<1.e-12,"tet-mesh common rigid transform");
    check(observe(pair,1).overlappingTetPairs==0,"same body excluded");
    Point flat[4]={{0,0,0,1},{1,0,0,1},{0,1,0,1},{1,1,0,1}};pair[0]=view(flat,4,tet,1);
    r=observe(pair,1);check(!r.valid && r.degenerateTetrahedra==1,"degenerate tetrahedron reported");
    std::printf("OFFICIAL_GEOMETRY_PROBE checks=%u failures=%u boxTetPairs=%llu boxMaxTetMtd=%.17g\n",checks,failures,
        static_cast<unsigned long long>(box.overlappingTetPairs),box.maxTetMtd);
    return failures?1:0;
}
