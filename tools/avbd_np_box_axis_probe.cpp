// Validate the Host's candidate axis against independently captured world vertices.
#include "PxsContactBoxAxis.h"
#include <cstdio>
#include <algorithm>
using namespace physx;
int main(int argc,char** argv)
{
    if(argc!=2)return 2;FILE* file=std::fopen(argv[1],"rb");if(!file)return 2;
    PxsContactGeometry g;PxVec3d vertices[2][8];
    for(unsigned s=0;s<2;++s)
    {
        float v[10];if(std::fread(v,sizeof(float),10,file)!=10)return 2;
        g.shapes[s].kind=PxsContactShapeGeometry::eBOX;
        g.shapes[s].shapeToWorld=PxTransform(PxVec3(v[0],v[1],v[2]),PxQuat(v[3],v[4],v[5],v[6]));
        g.shapes[s].halfExtents=PxVec3(v[7],v[8],v[9]);
        for(auto& point:vertices[s]){double xyz[3];if(std::fread(xyz,sizeof(double),3,file)!=3)return 2;point=PxVec3d(xyz[0],xyz[1],xyz[2]);}
    }
    double axis[3];if(std::fread(axis,sizeof(double),3,file)!=3)return 2;std::fclose(file);
    g.separatingAxis=PxVec3d(axis[0],axis[1],axis[2]);g.axisValid=true;
    auto gap=[&](PxVec3d n){double value=DBL_MAX;for(const auto& a:vertices[0])for(const auto& b:vertices[1])value=PxMin(value,n.dot(a-b));return value;};
    const double oldGap=gap(g.separatingAxis);pxsContactBoxSeparatingAxis(g);const double newGap=gap(g.separatingAxis);
    bool pass=oldGap<0 && newGap>0 && g.axisValid;
    const PxVec3d selected=g.separatingAxis;std::swap(g.shapes[0],g.shapes[1]);g.separatingAxis=-selected;
    pxsContactBoxSeparatingAxis(g);pass=pass && (selected+g.separatingAxis).magnitude()<1.e-14;
    // Identical boxes truly intersect: no selected direction may claim a gap.
    g.shapes[1]=g.shapes[0];pxsContactBoxSeparatingAxis(g);
    const PxQuat& stored=g.shapes[0].shapeToWorld.q;const PxQuatd q=PxQuatd(stored.x,stored.y,stored.z,stored.w).getNormalized();
    const PxVec3d local=q.rotateInv(g.separatingAxis);const PxVec3 h=g.shapes[0].halfExtents;
    const double overlapGap=-2*(PxAbs(local.x)*h.x+PxAbs(local.y)*h.y+PxAbs(local.z)*h.z);pass=pass && overlapGap<0;
    g.shapes[0].kind=PxsContactShapeGeometry::eSPHERE;g.separatingAxis=selected;pxsContactBoxSeparatingAxis(g);pass=pass && g.separatingAxis==selected;
    // Skew thin boxes: the separating direction is an edge cross product;
    // none of the six face normals separates. Enumerate corners as oracle.
    g.shapes[0].kind=PxsContactShapeGeometry::eBOX;
    g.shapes[0].shapeToWorld=PxTransform(PxVec3(0),PxQuat(.13f,.21f,.07f,.96f).getNormalized());
    g.shapes[1].shapeToWorld=PxTransform(PxVec3(0),PxQuat(-.29f,.09f,.33f,.89f).getNormalized());
    g.shapes[0].halfExtents=g.shapes[1].halfExtents=PxVec3(.8f,.05f,.05f);
    PxVec3d face[2][3];
    for(unsigned s=0;s<2;++s)
    {const PxQuat& v=g.shapes[s].shapeToWorld.q;const PxQuatd rotation=PxQuatd(v.x,v.y,v.z,v.w).getNormalized();
        for(unsigned d=0;d<3;++d){PxVec3d axis(0);axis[d]=1;face[s][d]=rotation.rotate(axis);}}
    auto cornerGap=[&](PxVec3d normal)
    {double ranges[2]={DBL_MAX,-DBL_MAX};for(unsigned s=0;s<2;++s)for(unsigned i=0;i<8;++i)
        {const auto& shape=g.shapes[s];PxVec3d v(shape.shapeToWorld.p.x,shape.shapeToWorld.p.y,shape.shapeToWorld.p.z);
            for(unsigned d=0;d<3;++d)v+=face[s][d]*double((i&(1u<<d))?shape.halfExtents[d]:-shape.halfExtents[d]);
            const double dot=normal.dot(v);ranges[s]=s?PxMax(ranges[s],dot):PxMin(ranges[s],dot);}return ranges[0]-ranges[1];};
    const PxVec3d edge=face[0][0].cross(face[1][0]).getNormalized(),shift=edge*(-cornerGap(edge)+1.e-4);
    g.shapes[0].shapeToWorld.p=PxVec3(float(shift.x),float(shift.y),float(shift.z));
    double faceGap=-DBL_MAX;for(unsigned s=0;s<2;++s)for(unsigned d=0;d<3;++d)
        faceGap=PxMax(faceGap,PxMax(cornerGap(face[s][d]),cornerGap(-face[s][d])));
    g.axisValid=false;pxsContactBoxSeparatingAxis(g);const double edgeGap=cornerGap(g.separatingAxis);
    pass=pass && faceGap<0 && edgeGap>0;
    std::printf("[NP_BOX_AXIS] cases=5 oldGap=%.17g selectedGap=%.17g overlapGap=%.17g faceGap=%.17g edgeGap=%.17g status=%s\n",oldGap,newGap,overlapGap,faceGap,edgeGap,pass?"PASS":"FAIL");
    return pass?0:1;
}
