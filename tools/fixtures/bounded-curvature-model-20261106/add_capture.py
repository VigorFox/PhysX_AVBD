from pathlib import Path
p=Path('physx/source/lowleveldynamics/src/avbd/DyAvbdSolve.cpp');b=p.read_bytes()
old=b'        const std::string detail=std::string(path)+".trial.json";file=std::fopen(detail.c_str(),"wb");'
addition='''        // Diagnostic only: distinguish material PSD projection from the
        // original second variation. No captured operator enters execution.
        const std::string curvaturePath=std::string(path)+".curvature.json";
        file=std::fopen(curvaturePath.c_str(),"wb");
        if(file)
        {
            std::fprintf(file,"{\\"frames\\":[");bool first=true;
            for(PxU32 r:subset.rows)
            {
                const auto& row=mRows[r];
                if((row.flags&eAVBD_ROW_INACTIVE) || row.kernelId!=eAVBD_ROW_KERNEL_ELASTIC_FRAME)continue;
                const auto& m=mElasticFrameMaterials[row.geometryIndex-1];
                const PxTransform endpoints[2]={PxTransform(mBodies[row.bodyA].p,mBodies[row.bodyA].q),PxTransform(mBodies[row.bodyB].p,mBodies[row.bodyB].q)};
                PxTransformT<double> bodies[2],frames[2];avbdElasticFrameSections(m.authored,endpoints,bodies,frames);
                double weight[FINITE_JOINT_ROWS],bias[FINITE_JOINT_ROWS],H[81];
                for(PxU32 j=0;j<FINITE_JOINT_ROWS;++j)
                {const PxU32 axis=finiteJointDriveIndex(j);weight[j]=m.authored.stiffness[axis]+m.dampingOverDt[axis];bias[j]=m.dampingOverDt[axis]*m.initial[j];}
                finiteJointMaterialHessian(frames[0],frames[1],bodies[0].p,bodies[1].p,weight,bias,H);
                const auto e=avbdEvaluateElasticFrame(m,endpoints,true);
                std::fprintf(file,"%s{\\"row\\":%u,\\"a\\":%u,\\"b\\":%u,\\"valid\\":%s,\\"raw\\":[",first?"":",",r,row.bodyA,row.bodyB,e.valid?"true":"false");first=false;
                for(PxU32 j=0;j<81;++j)std::fprintf(file,"%s%.17g",j?",":"",H[j]);
                std::fprintf(file,"],\\"projected\\":[");
                for(PxU32 j=0;j<9;++j)for(PxU32 k=0;k<9;++k)
                {double value=0;for(PxU32 z=0;z<9;++z)value+=double(PxReal(e.root[z][j<6?j:j+3]))*double(PxReal(e.root[z][k<6?k:k+3]));
                    std::fprintf(file,"%s%.17g",j||k?",":"",value);}
                std::fprintf(file,"]}");
            }
            std::fprintf(file,"]}\\n");std::fclose(file);
        }
'''.replace('\n','\r\n').encode()
assert b.count(old)==1;p.write_bytes(b.replace(old,addition+old))
