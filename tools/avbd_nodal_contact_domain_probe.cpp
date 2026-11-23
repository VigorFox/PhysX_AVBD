// Complete-feature counterexamples, independent of the solver trajectory.
#include "PxPhysicsAPI.h"
#include "GuIntersectionTriangleBox.h"
#include "avbd_probe_foundation.h"
#include "PxsDeformableTriangleFilter.h"
#include "avbd/DyAvbdNodalContact.h"
#include "avbd/DyAvbdRows.h"
#include <cstdio>
using namespace physx;
using namespace physx::Dy;
struct Pose {PxVec3 p;PxQuat q;};

static bool checkComposedEmbeddedPath()
{
    PxsContactGeometry facts;
    PxsDeformableGeometry volume;
    PxsDeformableNpState state;
    volume.state=&state; facts.deformable[0]=&volume;
    auto& box=facts.shapes[1];
    box.kind=PxsContactShapeGeometry::eBOX; box.count=8; box.halfExtents=PxVec3(.02f,.2f,.2f);
    PxsDeformablePointSupport embedding={};
    embedding.count=2; embedding.nodes[0]=1; embedding.nodes[1]=4;
    embedding.weights[0]=embedding.weights[1]=.5f;
    state.mesh.supports.pushBack(embedding);
    state.mesh.points.pushBack(PxsDeformableVec3(.5,0,0));
    PxsDeformableContact feature={};
    feature.point[0]=PxVec3(.5f,0,0); feature.vertexMask=1; feature.normal=PxVec3(1,0,0); feature.feature[1]=1;
    AvbdNodalContact contact={&facts,&feature,{3,PX_MAX_U32},0};
    Pose unused[1]={};
    const auto certify=[&](unsigned moved) {
        return avbdNodalContactPath(contact,[&](PxU32 b){
            if(b!=4 && b!=7) return PxVec3(10,3,2); // collision vertex zero is not a simulation node
            return PxVec3((moved&(b==4?1u:2u))?-.1f:.5f,0,0);
        },unused);
    };
    avbdPrepareNodalProjection(contact);
    if(contact.projection!=AvbdNodalContact::StaticSide0)return false;
    const bool crossed=certify(0) && certify(1) && certify(2) && !certify(3);
    feature.point[0]=PxVec3(-.1f,0,0); feature.normal=-feature.normal; feature.feature[1]=0;
    state.mesh.points[0]=PxsDeformableVec3(-.1,0,0);
    avbdPrepareNodalProjection(contact);
    return crossed && certify(3);
}

static bool checkFilteredTrianglePath()
{
    // Rest filtering picked an off-axis witness. The two coplanar triangles
    // have overlapping y projections, but disjoint x projections.
    PxVec4 x[]={{0,0,0,1},{1,0,0,1},{0,1,0,1},{1.1f,0,0,1},{2.1f,0,0,1},{1.1f,1,0,1}};
    const PxU32 ids[]={0,1,2,3,4,5};
    PxVec4* p=x;
    PxsDeformableNpState state;PxsDeformableGeometry sheet;PxsContactGeometry facts;
    sheet.positions=&p;sheet.collision={ids,2,6,3,4};sheet.simulation=sheet.collision;sheet.state=&state;
    if(!state.mesh.bake(sheet) || !state.mesh.sampleVertices(sheet,0,6))return false;
    facts.deformable[0]=facts.deformable[1]=&sheet;
    PxsDeformableContact f={};f.feature[1]=1;f.normal=PxVec3(0,1,0);f.point[0]=x[0].getXYZ();
    AvbdNodalContact c={&facts,&f,{0,0},.01f};
    Pose unused[1]={};
    const auto certify=[&](PxReal dx,PxReal common)
    {
        return avbdNodalContactPath(c,[&](PxU32 i){return x[i].getXYZ()+PxVec3((i<3?0.f:dx)+common,0,0);},unused);
    };
    const auto outcomes=[&](){return certify(0,0) && certify(0,3) && !certify(-.095f,0) &&
        !certify(-.5f,0) && !certify(-4.f,0);};
    if(!outcomes())return false;
    avbdPrepareNodalProjection(c);
    if(c.projection!=AvbdNodalContact::TrianglePair || !outcomes())return false;
    c.restDistance=0;
    avbdPrepareNodalProjection(c);
    if(certify(-.5f,0) || certify(-4.f,0))return false;
    // The first prepared direction can close while a second direction still
    // separates the complete path. Search must remain available in that case.
    for(PxU32 i=3;i<6;++i)x[i].y+=2.5f;
    if(!state.mesh.sampleVertices(sheet,0,6))return false;
    f.normal=PxVec3(-1,0,0);
    avbdPrepareNodalProjection(c);
    if(c.projection!=AvbdNodalContact::TrianglePair || !certify(-.5f,0))return false;
    // A different feature in the next DCD epoch cannot inherit this axis.
    f.feature[1]=0;
    avbdPrepareNodalProjection(c);
    return c.projection==AvbdNodalContact::General;
}

static bool checkRestDistanceBound()
{
    PxVec3 rest[2][3]={{{0,0,0},{1,0,0},{0,0,1}},{{1.05f,0,0},{2,0,0},{1,0,1.1f}}};
    PxVec3 live[2][3];
    for(PxU32 s=0;s<2;++s)for(PxU32 i=0;i<3;++i)
    {
        const auto p=rest[s][i];live[s][i]=PxVec3(3-p.z,4+p.y,p.x-2);
    }
    if(!pxsTriangleRestFiltered(live,rest,.08f,.12f) || pxsTriangleRestFiltered(live,rest,.08f,.08f))return false;
    // Collapse just one triangle onto the other: genuinely distant material
    // can now touch. Exercise each SIMD lane and both endpoint orderings.
    for(PxU32 s=0;s<2;++s)for(PxU32 i=0;i<3;++i)
    {
        const auto saved=live[s][i];
        PxU32 far=0;
        for(PxU32 j=1;j<3;++j)
            if((rest[s][i]-rest[1-s][j]).magnitudeSquared()>(rest[s][i]-rest[1-s][far]).magnitudeSquared())far=j;
        live[s][i]=live[1-s][far];
        if(pxsTriangleRestFiltered(live,rest,.08f,.12f))return false;
        live[s][i]=saved;
    }
    // Independent interior-witness oracle at different strain/reach ratios.
    const PxVec3 weights[]={{1,0,0},{0,1,0},{0,0,1},{.5f,.5f,0},{0,.5f,.5f},{.5f,0,.5f},{.2f,.3f,.5f}};
    const auto interpolate=[](const PxVec3 (&tri)[3],const PxVec3& w)
    {return tri[0]*w.x+tri[1]*w.y+tri[2]*w.z;};
    PxU32 culled=0,kept=0;
    for(PxU32 k=0;k<80;++k)
    {
        const PxReal reach=.1f+.02f*(k%17),distance=reach+.08f*(k%29);
        for(PxU32 i=0;i<3;++i)
            live[1][i]=rest[1][i]*PxReal(1-.025f*(k%13))+PxVec3(-.04f*(k%19),.01f*i,0);
        for(PxU32 i=0;i<3;++i)live[0][i]=rest[0][i];
        if(!pxsTriangleRestFiltered(live,rest,reach,distance)){++kept;continue;}
        ++culled;
        for(const auto& a:weights)for(const auto& b:weights)
            if((interpolate(live[0],a)-interpolate(live[1],b)).magnitudeSquared()<=reach*reach &&
               (interpolate(rest[0],a)-interpolate(rest[1],b)).magnitudeSquared()>distance*distance)return false;
    }
    return culled && kept;
}

static bool checkSweptSheetBox()
{
    PxVec4 x[]={{-.2f,.4f,0,1},{.2f,.4f,0,1},{0,.4f,.2f,1}};
    PxVec4* positions=x;const PxU32 ids[]={0,1,2};
    PxsDeformableNpState state;PxsDeformableGeometry sheet;PxsContactGeometry facts;
    sheet.state=&state;sheet.positions=&positions;sheet.collision={ids,1,3,3,4};sheet.simulation=sheet.collision;
    if(!state.mesh.bake(sheet) || !state.mesh.sampleVertices(sheet,0,3))return false;
    facts.deformable[0]=&sheet;
    auto& box=facts.shapes[1];box.kind=PxsContactShapeGeometry::eBOX;box.count=8;box.halfExtents=PxVec3(.1f);
    PxsDeformableContact feature={};feature.normal=PxVec3(1,0,0);feature.point[0]=x[1].getXYZ();
    AvbdNodalContact domain={&facts,&feature,{0,PX_MAX_U32},.02f};
    avbdPrepareNodalProjection(domain);
    Pose unused[1]={};
    const auto admits=[&](PxReal height)
    {
        return avbdNodalContactPath(domain,[&](PxU32 id){return PxVec3(x[id].x,height,x[id].z);},unused);
    };
    // The force axis overlaps at both ends. An alternate box face clears the
    // first path. The second spends the skin; the third crosses and exits.
    if(!admits(.15f) || admits(.11f) || admits(-.4f))return false;
    feature.normal=PxVec3(0,1,0);
    avbdPrepareNodalProjection(domain);
    const auto crossing=[&](PxU32 id){return PxVec3(x[id].x,-.4f,x[id].z);};
    // 0.8 m of travel, 0.28 m of usable clearance. The rounded seed
    // must remain admissible just inside the time of first skin contact.
    const auto fraction=avbdSheetBoxPredictorFraction(domain,crossing);
    if(!(fraction>.349f && fraction<=.35f) || !admits(.4f-.8f*fraction))return false;

    // A face rolls from 15 to 75 degrees around the same box edge. Its
    // clearance at the endpoints is 0.03; midway it is 0.03*cos(30 degrees).
    box.halfExtents=PxVec3(1);
    const PxVec3 normals[]={PxVec3(.96592583f,.25881905f,0),PxVec3(.25881905f,.96592583f,0)};
    PxVec3 end[3];
    for(PxU32 k=0;k<2;++k)
    {
        const auto n=normals[k],t=PxVec3(-n.y,n.x,0);
        const auto origin=PxVec3(1,1,0)+n*.03f;
        const PxVec3 corners[]={origin-.4f*t-PxVec3(0,0,.2f),origin+.4f*t-PxVec3(0,0,.2f),origin+PxVec3(0,0,.2f)};
        for(PxU32 v=0;v<3;++v)
            if(k)end[v]=corners[v];else x[v]=PxVec4(corners[v],1);
    }
    if(!state.mesh.sampleVertices(sheet,0,3))return false;
    feature.normal=normals[0];feature.point[0]=x[0].getXYZ();
    const auto roll=[&](PxReal skin)
    {
        domain.restDistance=skin;
        avbdPrepareNodalProjection(domain);
        return avbdNodalContactPath(domain,[&](PxU32 id){return end[id];},unused);
    };
    return roll(.02f) && !roll(.028f);
}

static bool checkFrictionEpoch()
{
    PxsContactGeometry pairs[2];
    AvbdNodalFriction old={};
    old.pair=pairs; old.birthEpoch=6; old.epoch=17; old.vertexMask=1;
    old.feature[0]=4; old.feature[1]=7; old.normal=PxVec3(0,0,1);
    old.traction=PxVec3(2,-3,0); old.stickSlip=PxVec3(.25f,.5f,0);
    auto next=old; ++next.epoch;
    for(unsigned observations : {0u,1u,16u,48u})
    {
        AvbdScalarRowPayload law={};
        law.t0=PxVec3(1,0,0); law.t1=PxVec3(0,1,0);
        for(unsigned i=0;i<observations;++i)
            if(!avbdReadNodalFriction(old,next,true,law)) return false;
        if(!avbdReadNodalFriction(old,next,true,law) || law.lambdaT0!=2 || law.lambdaT1!=-3 ||
           law.tangentC0!=.25f || law.tangentC1!=.5f) return false;
    }
    for(unsigned scenario=0;scenario<7;++scenario)
    {
        auto changed=next;
        if(scenario==0) changed.pair=pairs+1;
        if(scenario==1) changed.birthEpoch=next.epoch;  // same pool address, new lifetime
        if(scenario==2) ++changed.epoch;                // missing/rejected previous publication
        if(scenario==3) changed.epoch=old.epoch;
        if(scenario==4) ++changed.feature[0];
        if(scenario==5) changed.vertexMask=2;
        if(scenario==6) changed.normal=-old.normal;
        AvbdScalarRowPayload law={};
        if(avbdReadNodalFriction(old,changed,true,law) || law.lambdaT0 || law.lambdaT1) return false;
    }
    return true;
}
// Force the crowded-table case independently of allocator addresses and worker order.
static bool checkFrictionOverflow()
{
    PxsContactGeometry pairs[3];
    AvbdFrictionOverflow index;
    for(PxU32 count : {0u,1u,32u,33u,257u})
    {
        PxArray<AvbdNodalFriction> history(8);
        for(PxU32 i=0;i<count;++i)
        {
            AvbdNodalFriction f={};
            f.pair=&pairs[i%3];
            f.feature[0]=(i%7)<<24;
            f.feature[1]=(count-i)/3;
            f.vertexMask=PxU8(i%4);
            history.pushBack(f);
        }
        index.build(history,8);
        for(PxU32 i=8;i<history.size();++i)
        {
            if(index.find(history,history[i])!=i)return false;
            AvbdNodalFriction absent=history[i];absent.vertexMask=7;
            if(index.find(history,absent)!=PX_MAX_U32)return false;
        }
    }
    return true;
}

static bool checkMixedWrench()
{
    const PxQuat start(.39f,PxVec3(0,1,0)),end(.58f,PxVec3(0,0,1));
    const PxVec3 arm(.4f,.7f,-.1f),center(.25f,-.4f,.1f),axis=PxVec3(.7f,.2f,-.4f).getNormalized();
    const auto anchor=center+end.rotate(arm);
    const PxVec3 point=anchor+PxVec3(.025f,-.013f,.009f);
    const auto coordinate=[&](const PxQuat& q,const PxVec3& x)
    {return axis.dot(avbdNodalMaterialVector(start,q,x-(center+q.rotate(arm))));};
    const auto linear=avbdNodalWorldAxis(start,end,axis);
    const auto torque=avbdNodalWitnessArm(end,arm,point-anchor,-1.f).cross(-linear);
    const auto direction=PxVec3(.2f,-.5f,.3f).getNormalized();
    const PxReal h=1.e-3f;
    const auto derivative=(coordinate(PxQuat(h,direction)*end,point)-coordinate(PxQuat(-h,direction)*end,point))/(2*h);
    // Equal/opposite forces must have zero net torque about any origin.
    return PxAbs(coordinate(end,anchor))<1.e-6f && PxAbs(derivative-torque.dot(direction))<1.e-4f &&
        ((point-center).cross(linear)+torque).magnitude()<1.e-6f;
}
static bool checkEmbeddedImpact()
{
    struct Particle { PxVec3 pStart,pHat;PxReal invMass; };
    Particle p[3]={{PxVec3(0),PxVec3(-2,0,0),1},{PxVec3(0),PxVec3(-2,0,0),.5f},
        {PxVec3(0),PxVec3(0),.25f}};
    PxsContactGeometry facts;PxsDeformableGeometry hosts[2];PxsDeformableNpState state[2];
    for(PxU32 s=0;s<2;++s)
    {
        facts.deformable[s]=&hosts[s];hosts[s].state=&state[s];
        state[s].mesh.points.pushBack(PxsDeformableVec3(s?0:1,0,0));
        PxsDeformablePointSupport e={};e.count=s?1:2;e.nodes[0]=0;e.weights[0]=s?1:1.25;
        e.nodes[1]=1;e.weights[1]=-.25;state[s].mesh.supports.pushBack(e);
    }
    PxsDeformableContact feature={};
    feature.point[0]=PxVec3(1,0,0);
    feature.point[1]=PxVec3(0);
    feature.normal=PxVec3(1,0,0);
    feature.vertexMask=3;
    AvbdNodalContact c={&facts,&feature,{0,2},0};
    const auto project=[&](PxReal (&delta)[3])
    {
        avbdNominatePairImpact(c,.1f,[&](PxU32 b)->const Particle& {return p[b];},
            [&](PxU32 b,PxReal step){delta[b]=step;});
    };
    PxReal delta[3]={},moving[3]={},separating[3]={};
    project(delta);
    for(auto& node:p)node.pHat+=PxVec3(3,4,-1);
    project(moving);
    for(auto& node:p)node.pHat=node.pStart;
    project(separating);
    const PxReal gap=-1+1.25f*delta[0]-.25f*delta[1]-delta[2];
    PxReal momentum=0;
    for(PxU32 b=0;b<3;++b)
    {
        momentum+=delta[b]/p[b].invMass;
        if(PxAbs(moving[b]-delta[b])>1.e-5f || separating[b]!=0)return false;
    }
    return PxAbs(gap-.1f)<1.e-5f && PxAbs(momentum)<1.e-5f && delta[1]<0;
}

static bool compareAffineTets()
{
    PxVec4 vertices[2][4];PxVec4* buffers[]={vertices[0],vertices[1]};
    const PxU32 ids[]={0,1,2,3};
    PxsDeformableGeometry geometry[2];PxsDeformableNpState state[2];PxsContactGeometry facts;
    for(PxU32 s=0;s<2;++s)
    {
        const PxReal x=s?-1.f:1.f;
        vertices[s][0]=PxVec4(x,0,0,1);vertices[s][1]=PxVec4(x+1,0,0,1);
        vertices[s][2]=PxVec4(x,1,0,1);vertices[s][3]=PxVec4(x,0,1,1);
        auto& g=geometry[s];g.positions=&buffers[s];g.collision={ids,1,4,4,4};
        g.simulation=g.collision;g.state=&state[s];facts.deformable[s]=&g;
        if(!state[s].mesh.bake(g) || !state[s].mesh.sampleVertices(g,0,4))return false;
    }
    PxsDeformableContact feature={};feature.normal=PxVec3(1,0,0);feature.point[0]=vertices[0][0].getXYZ();
    AvbdNodalContact c={&facts,&feature,{0,4},.125f};Pose unused[1]={};
    const bool certified=avbdTetsSeparated(c,[&](PxU32 s,PxU32 i){return state[s].mesh.points[i];});
    PxU32 accepted=0,rejected=0;
    for(PxU32 k=0;k<64;++k)
    {
        const auto node=[&](PxU32 b)
        {
            const PxU32 side=b/4,vertex=b%4;
            PxVec3 p=vertices[side][vertex].getXYZ();
            p.y+=PxReal(k);
            if(vertex==k%4)p.x+=(side?.5f:-1.f)*PxReal(k/4)*.25f;
            return p;
        };
        c.separatedTets=false;const bool reference=avbdNodalContactPath(c,node,unused);
        c.separatedTets=certified;
        if(avbdNodalContactPath(c,node,unused)!=reference)return false;
        reference?++accepted:++rejected;
        if(!reference)continue;
        for(PxU32 moving=0;moving<2;++moving)
        {
            AvbdNodalReal frozen=0;
            AvbdNodalTrialVertices assembly;
            if(!avbdTetsSeparated(c,[&](PxU32 s,PxU32 id){return assembly.read(c,s,id,node);},
                                   1-moving,&frozen))return false;
            // A different corner wins for each move. Re-read the other side
            // on the next assembly, including when both assets use vertex 0.
            for(PxU32 corner=0;corner<4;++corner)
            {
                const auto trial=[&](PxU32 b){
                    auto p=node(b);
                    if(b==4*moving+corner)p.x+=moving?4.f:-4.f;
                    return p;
                };
                AvbdNodalTrialVertices verticesAtTrial;
                const bool reduced=avbdTetsSeparated(c,[&](PxU32 s,PxU32 id){return verticesAtTrial.read(c,s,id,trial);},
                                                     1-moving,&frozen,true);
                if(reduced!=avbdNodalContactPath(c,trial,unused))return false;
            }
        }
    }
    // Same collision ID and side can occur in two actors, or in different
    // collision embeddings of the same simulation blocks.
    AvbdNodalTrialVertices cache;
    PxsDeformablePointSupport support={};
    support.count=2;support.nodes[0]=0;support.nodes[1]=1;
    support.weights[0]=1.25f;support.weights[1]=-.25f;
    for(auto& s:state){s.mesh.leafBase=0;s.mesh.supports[0]=support;}
    state[1].mesh.supports[0].nodes[0]=2;
    const auto node=[](PxU32 b){return PxVec3(PxReal(b),0,0);};
    for(unsigned i=0;i<8;++i)
    {
        facts.deformable[0]=&geometry[i%2];c.first[0]=(i/2)%2?0:4;
        const PxReal expected=PxReal(c.first[0])+(i%2?2.25f:-.25f);
        for(unsigned hit=0;hit<2;++hit)if(cache.read(c,0,0,node).x!=expected)return false;
    }
    c.first[0]=0;facts.deformable[0]=&geometry[0];
    return certified && accepted && rejected && avbdNodalPairSide(c,3)==0 && avbdNodalPairSide(c,4)==1;
}
static bool checkPedestalEdge()
{
    const PxVec3 pivot(.5f,.5f,0);
    const auto vertex=[&](PxU32 i,PxReal angle,PxReal gap)
    {
        const PxVec3 n(PxSin(angle),PxCos(angle),0),along(n.y,-n.x,0);
        const PxVec3 face[]={{-.5f,-.4f,0},{.5f,-.4f,0},{0,.4f,0},{0,0,.3f}};
        return pivot+n*(gap+face[i].z)+along*face[i].x+PxVec3(0,0,face[i].y);
    };
    PxVec4 initial[4];PxVec4* data=initial;const PxU32 ids[]={0,1,2,3};
    for(PxU32 i=0;i<4;++i)initial[i]=PxVec4(vertex(i,0,.04f),1);
    PxsDeformableGeometry g;PxsDeformableNpState state;PxsContactGeometry facts;
    g.positions=&data;g.collision={ids,1,4,4,4};g.simulation=g.collision;g.state=&state;
    if(!state.mesh.bake(g) || !state.mesh.sampleVertices(g,0,4))return false;
    facts.deformable[0]=&g;
    auto& box=facts.shapes[1];box.kind=PxsContactShapeGeometry::eBOX;box.halfExtents=PxVec3(.5f);box.count=8;
    PxsDeformableContact f={};f.normal=PxVec3(0,1,0);f.point[0]=initial[0].getXYZ();
    AvbdNodalContact c={&facts,&f,{0,PX_MAX_U32},0};Pose unused[1]={};
    const auto separated=[&](PxU32 i){return vertex(i,.5f,.04f);};
    const auto crossing=[&](PxU32 i){return vertex(i,.5f,-.15f);};
    return avbdNodalContactPath(c,separated,unused) && !avbdNodalContactPath(c,crossing,unused);
}
int main()
{
    AvbdProbeFoundation foundation;
    const bool triangle=checkFilteredTrianglePath(),filter=checkRestDistanceBound();
    std::printf("[TRIANGLE_CONTACT_DOMAIN] sweptPath=%u materialDistanceBound=%u\n",unsigned(triangle),unsigned(filter));
    if(!triangle || !filter || !checkSweptSheetBox())return 1;
    const bool composed=checkComposedEmbeddedPath(), epochs=checkFrictionEpoch();
    std::printf("[NODAL_EMBEDDING_HISTORY] composedFinitePath=%u lifetimeAndObservations=%u\n",unsigned(composed),unsigned(epochs));
    if(!composed || !epochs) return 1;
    PxVec4 storage[3]={PxVec4(.7f,-2,-2,1),PxVec4(.7f,2,-2,1),PxVec4(.7f,0,2,1)};
    PxVec4* positions=storage;const PxU32 ids[3]={0,1,2};
    PxsDeformableGeometry geometry;PxsDeformableNpState state;
    geometry.positions=&positions;geometry.collision={ids,1,3,3,4};geometry.simulation=geometry.collision;geometry.state=&state;
    if(!state.mesh.bake(geometry) || !state.mesh.sampleVertices(geometry,0,3))return 2;
    PxsContactGeometry facts;facts.deformable[0]=&geometry;
    auto& box=facts.shapes[1];box.kind=PxsContactShapeGeometry::eBOX;box.count=8;box.halfExtents=PxVec3(2,.01f,.01f);
    const float angle=80.f*PxPi/180;
    box.shapeToWorld=PxTransform(PxVec3(0),PxQuat(-angle,PxVec3(0,0,1)));
    PxsDeformableContact feature={};feature.normal=PxVec3(1,0,0);feature.point[0]=storage[0].getXYZ();
    feature.weights[0]=PxVec4(1,0,0,0);feature.feature[0]=0;
    AvbdNodalContact contact={&facts,&feature,{0,0},0};
    Pose body[1]={{PxVec3(0),box.shapeToWorld.q}};
    PxVec3 trial[3];for(PxU32 i=0;i<3;++i)trial[i]=storage[i].getXYZ();
    const auto node=[&](PxU32 i){return trial[i];};
    const bool pedestalEdge=checkPedestalEdge();
    std::printf("[PEDESTAL_EDGE] sweptFaceAndCrossing=%u\n",unsigned(pedestalEdge));
    const bool impact=checkEmbeddedImpact();
    std::printf("[EMBEDDED_IMPACT] signedMassProjection=%u\n",unsigned(impact));
    bool pass=checkFrictionOverflow() && impact && pedestalEdge && checkMixedWrench() && compareAffineTets() && avbdNodalContactPath(contact,node,body);
    body[0].q=PxQuat(angle,PxVec3(0,0,1));
    // Both endpoint poses separate, but the rotating box crosses the sheet.
    const bool rotationRejected=!avbdNodalContactPath(contact,node,body);pass &= rotationRejected;
    const auto original=box.shapeToWorld;box.shapeToWorld.q=body[0].q;
    const bool endValid=avbdNodalContactPath(contact,node,body);pass &= endValid;box.shapeToWorld=original;
    // The selected force witness stays still; a different triangle corner crosses.
    body[0].q=box.shapeToWorld.q;trial[2].x=.1f;
    const bool omittedVertexRejected=!avbdNodalContactPath(contact,node,body);pass &= omittedVertexRejected;
    trial[2]=storage[2].getXYZ();
    // Compare analytic oriented-box extrema with all eight explicit vertices
    // for off-center shapes, translations, rotations and both contact sides.
    const auto savedBox=box;PxVec3 corners[8];
    for(PxU32 i=0;i<8;++i)corners[i]=avbdNodalRigidVertex(box,i);
    PxU32 mismatches=0,accepted=0,rejected=0;
    for(PxU32 k=0;k<360;++k)
    {
        box=savedBox;
        const PxVec3 axis=PxVec3(1,.3f,.1f).getNormalized(),second=PxVec3(-.1f,.8f,.5f).getNormalized();
        box.shapeToBody=PxTransform(PxVec3(.17f,-.29f,.41f),PxQuat(.37f,axis));
        const PxTransform start(PxVec3(PxReal(k%15)*.25f-3,.13f,.07f),PxQuat(PxReal(k)*.037f,second));
        box.shapeToWorld=start*box.shapeToBody;
        body[0]={start.p+PxVec3(.04f,-.022f,.17f),PxQuat(PxReal(k)*.048f+.3f,axis)*start.q};
        feature.normal=PxVec3(1,PxReal(int(k%5)-2)*.04f,PxReal(int(k%7)-3)*.03f).getNormalized();
        contact.first[1]=(k&1)?0:PX_MAX_U32;
        PxsContactGeometry swapped;swapped.deformable[1]=&geometry;
        AvbdNodalContact reverse=contact;reverse.geometry=&swapped;reverse.first[0]=contact.first[1];reverse.first[1]=0;
        const auto test=[&](){
            if(k%3){avbdPrepareNodalProjection(contact);return avbdNodalContactPath(contact,node,body);}
            swapped.shapes[0]=box;feature.normal=-feature.normal;
            avbdPrepareNodalProjection(reverse);
            const bool value=avbdNodalContactPath(reverse,node,body);feature.normal=-feature.normal;return value;};
        const bool analytic=test();box.kind=PxsContactShapeGeometry::eVERTICES;box.vertices=corners;
        bool reference=test();
        // Static samples here have identical start/end vertices and zero skin.
        // PhysX's scalar triangle/box oracle covers all separating directions.
        // The moving-box comparison still uses all eight explicit vertices.
        if(contact.first[1]==PX_MAX_U32)
        {
            const auto& pose=box.shapeToWorld;
            reference=!Gu::intersectTriangleBox_ReferenceCode(PxVec3(0),box.halfExtents,
                pose.transformInv(trial[0]),pose.transformInv(trial[1]),pose.transformInv(trial[2]));
        }
        mismatches+=analytic!=reference;
        if(reference)++accepted;else ++rejected;
    }
    pass &= !mismatches && accepted && rejected;
    // An off-centre point follows a rotating box face with positive skin.
    // It passes behind the initial world plane while separated throughout.
    box=savedBox;box.shapeToBody=PxTransform(PxIdentity);box.shapeToWorld=PxTransform(PxIdentity);box.halfExtents=PxVec3(.5f,1,.4f);
    feature.normal=PxVec3(1,0,0);feature.vertexMask=1;contact.first[1]=0;
    storage[0]=PxVec4(.55f,.8f,0,1);
    pass &= state.mesh.sampleVertices(geometry,0,3);
    body[0]={PxVec3(0),PxQuat(.3f,PxVec3(0,0,1))};
    trial[0]=body[0].q.rotate(storage[0].getXYZ());
    const bool carriedClear=avbdNodalContactPath(contact,node,body);
    trial[0]=body[0].q.rotate(PxVec3(.42f,.8f,0));
    const bool carriedCrossing=!avbdNodalContactPath(contact,node,body);
    pass &= carriedClear && carriedCrossing;
    std::printf("[MIXED_WRENCH] derivativeAndTorque=%u carriedClear=%u crossingRejected=%u\n",
        unsigned(checkMixedWrench()),unsigned(carriedClear),unsigned(carriedCrossing));
    storage[0]=PxVec4(.7f,-2,-2,1);trial[0]=storage[0].getXYZ();feature.vertexMask=0;
    pass &= state.mesh.sampleVertices(geometry,0,3);
    box=savedBox;feature.normal=PxVec3(1,0,0);
    // A world plane and the exact same affine feature path.
    facts.shapes[1]=PxsContactShapeGeometry();facts.shapes[1].kind=PxsContactShapeGeometry::ePLANE;
    contact.first[1]=PX_MAX_U32;avbdPrepareNodalProjection(contact);pass &= avbdNodalContactPath(contact,node,body);
    trial[1].x=-.01f;const bool planeRejected=!avbdNodalContactPath(contact,node,body);pass &= planeRejected;
    feature.vertexMask=1;feature.feature[0]=0;trial[0].x=-5.e-9f;
    avbdPrepareNodalProjection(contact);
    const bool shallowPlaneCrossingRejected=!avbdNodalContactPath(contact,node,body);
    trial[0].x=0;const bool touchingAllowed=avbdNodalContactPath(contact,node,body);
    pass &= shallowPlaneCrossingRejected && touchingAllowed;
    // Project (6,8) onto the radius-five disk. Compare its displacement
    // derivative in both tangent directions, including the radial null mode.
    const PxVec3d j0(1,0,0),j1(0,1,0),traction(6,8,0);
    const auto metric=avbdNodalSlidingMetric(j0,j1,3.,4.,5.,.5,2.);
    const auto projected=[&](const PxVec3d& dx){return (traction+2.*dx).getNormalized()*5.;};
    for(const auto dir:{j0,j1})
    {
        const auto derivative=(projected(dir*1.e-4)-projected(-dir*1.e-4))/(2.e-4);
        pass &= (metric*dir-derivative).magnitude()<1.e-7;
    }
    pass &= (metric*traction).magnitude()<1.e-12;
    std::printf("[PLANE_SUPPORT_CHECK] shallowCrossingRejected=%u touchingAllowed=%u\n",
        unsigned(shallowPlaneCrossingRejected),unsigned(touchingAllowed));
    std::printf("[BOX_SUPPORT_CHECK] accepted=%u rejected=%u mismatches=%u\n",accepted,rejected,mismatches);
    std::printf("[NODAL_CONTACT_DOMAIN] scalarBytes=%u rotationCrossingRejected=%u endpointValid=%u zeroForceVertexRejected=%u planeCrossingRejected=%u status=%s\n",
        unsigned(sizeof(AvbdNodalReal)),unsigned(rotationRejected),unsigned(endValid),unsigned(omittedVertexRejected),unsigned(planeRejected),pass?"PASS":"FAIL");
    return pass?0:1;
}
