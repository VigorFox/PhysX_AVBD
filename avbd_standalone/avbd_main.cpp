#include "avbd_solver.h"
#include <cstdio>
#include <string>
#include <vector>

// External test functions
// Reliability
bool test1_singleBoxOnGround();
bool test2_twoBoxStack();
bool test3_fiveBoxTower();
bool test4_pyramid();
bool test5_dropFromHeight();
bool test6_perFrameRegenWithCache();
bool test7_physxScale();
bool test8_asymmetricMass();
bool test9_tenBoxTower();
bool test10_longTermStability();

// Collision
bool test11_collisionSingleBox();
bool test12_collisionThreeStack();
bool test13_collisionDrop();
bool test14_collisionPhysxTower();
bool test15_pyramidStack();
bool test16_pyramidNoFriction();

// Joints
bool test17_sphericalJointChain();
bool test18_fixedJointChain();
bool test19_d6JointChain();
bool test20_d6JointChain_snippetJoint();
bool test21_highMassRatioChain();
bool test22_meshChainmail();
bool test23_heavyBallOnMesh();
bool test24_fastBallOnChainmail();
bool test25_smallBallOnChainmail();
bool test26_snippetChainmailReplica();
bool test27_joints3x3Solve();

// Drives: 4 modes × 4 variants = 16 tests
bool test28_linearX_default();
bool test29_linearX_rotFrameA();
bool test30_linearX_rotBodyB();
bool test31_linearX_rotBoth();
bool test32_twist_default();
bool test33_twist_rotFrameA();
bool test34_twist_rotBodyB();
bool test35_twist_rotBoth();
bool test36_swing1_default();
bool test37_swing1_rotFrameA();
bool test38_swing1_rotBodyB();
bool test39_swing1_rotBoth();
bool test40_slerp_default();
bool test41_slerp_rotFrameA();
bool test42_slerp_rotBodyB();
bool test43_slerp_rotBoth();
bool test44_sphericalConeLimit();
bool test45_gearJoint_basicRatio();
bool test46_gearJoint_oppositeDir();
bool test47_prismaticJoint_basic();
bool test48_prismaticJoint_drive();
bool test49_prismaticChain_6x6();
bool test50_prismaticChain_3x3();

// Revolute joints
bool test51_revoluteJoint_basic();
bool test52_revoluteJoint_limit();
bool test53_revoluteJoint_drive();
bool test54_revoluteJoint_axisAlign();
bool test55_revoluteJoint_jitterRepro();

// Friction
bool test56_tiltedPlane_zeroFriction();
bool test57_tiltedPlane_highFriction();
bool test58_frictionComparison_lowVsHigh();
bool test59_zeroFriction_noDeceleration();
bool test60_highFriction_stopsQuickly();
bool test61_pyramidFrictionStability();
bool test62_stackedBoxOffset_frictionHolds();
bool test63_lateralPush_frictionResists();
bool test64_frictionIsotropy();
bool test65_dynamicDynamicFriction();
bool test66_massRatioFriction();
bool test67_frictionSweep_monotonic();
bool test68_rotationalFriction();
bool test69_restingContactNoDrift();
bool test70_tangentDirection_negativeX();
bool test71_coulombCone_noExplosion();
bool test72_geometricMeanFriction();
bool test73_longTermFrictionStability();

// Articulation (pure AVBD AL constraints)
bool test74_articulationPendulum();
bool test75_articulationChain5();
bool test76_articulationOnGround();
bool test77_articulationWithLimits();
bool test78_articulationSpherical();
bool test79_articulationFixed();
bool test80_articulationPrismatic();
bool test81_articulationPrismaticLimits();
bool test82_articulationPDDrive();
bool test83_articulationJointFriction();
bool test84_articulationConstraintAccuracy();
bool test85_articulationMixedJoints();
bool test86_articulationFloatingBase();
bool test87_articulationBranching();
bool test88_articulationVelocityDrive();
bool test89_articulationMassRatio();
bool test90_articulationDriveGravComp();
bool test91_articulationIDExtraction();
bool test92_articulationEndEffectorIK();
bool test93_articulationLongChain();
bool test94_articulationPrismaticDriveTracking();
bool test95_articulationMultiArticulation();
bool test96_articulationFloatingBaseMomentum();
bool test97_articulationMimicJoint();
bool test98_convergenceBenchmark();
bool test99_treeSweepConvergence();
bool test100_andersonAcceleration();
bool test101_chebyshevSemiIterative();
bool test102_articulationD6LoopClosure();
bool test103_scissorLiftValidation();

// Soft body tests
bool test104_softBodyFreeFall();
bool test105_softBodyGroundSettle();
bool test106_softBodyVolumePreservation();
bool test107_softBodyMaterialStiffness();
bool test108_softBodyLongTermStability();
bool test109_softBodyStacked();
bool test110_softBodyConvergence();
bool test111_softBodyToppling();
bool test112_softBodyAngularMomentum();

bool test114_deformableSphereShot_sequential_gate();
bool test115_deformableAggregated_noFriction6x6();
bool test116_deformableFriction_dominantSequential();
bool test117_deformableStaticAnchor_motion();
bool test118_boxOnGround_aggregatedUnchangedBySequentialMode();
bool test119_kinematicShell_sphereShot();
bool test120_kinematicShell_stressHarness();
bool test121_kinematicShell_vs_staticAnchor_sphereShot();
bool test122_prismaticReverseEndpointFrameA();
bool test123_d6VelocityDriveReverseEndpointFrameA();
bool test124_d6LockedLinearReactionWriteback();
bool test125_d6OffsetCoupledReaction();
bool test126_matrixFreeIslandOperator();
bool test127_linearPositionDriveDiscreteEquation();
bool test128_linearPositionDriveOutputForceSemantics();
bool test129_angularTwistVelocityDriveOutputForceSemantics();
bool test130_angularSwing1VelocityDriveOutputForceSemantics();
bool test131_angularSwing2VelocityDriveOutputForceSemantics();
bool test132_angularSlerpVelocityDriveOutputForceSemantics();
bool test133_linearPositionDriveOffsetMomentSemantics();
bool test134_angularTwistPositionDriveDiscreteEquation();
bool test135_linearAccelerationDriveEffectiveMassSemantics();
bool test136_angularSwing1PositionDriveDiscreteEquation();
bool test137_angularSwing2PositionDriveDiscreteEquation();
bool test138_angularSlerpPositionDriveDiscreteEquation();
bool test139_gearJointImpulseProjectionDiscreteEquation();
bool test140_dynamicDynamicAngularPositionDriveDiscreteEquation();
bool test141_bodyUnilateralProjectionAuthority();
bool test142_componentUnilateralProjectionAuthority();
bool test143_boundedComponentPositionImpulseAuthority();
bool test144_supportAxisRigidNullModeAuthority();
bool test145_finiteDriveStaticFrictionPositionAuthority();
bool test146_unequalMassFrictionWeightShareAuthority();
bool test147_slopedSupportFrictionPositionAuthority();
bool test148_contactPositionOutputForceOwnerAuthority();
bool test149_passiveRigidStaticFrictionManifoldAuthority();
bool test150_passiveRigidContactMaterialComponentAuthority();
bool test151_restitutionRigidMaterialComponentAuthority();
bool test152_transientRestitutionBroadMaterialComponentAuthority();
bool probe153_scalableBroadMaterialComponentAccelerated();
bool probe154_broadMaterialComponentScalingAuthority();
bool probe155_materialInterfaceWrenchAuthority();
bool probe156_materialMultilevelGraphAuthority();
bool probe157_materialSpatialTransferAuthority();
bool probe127_canonicalPyramidFrictionOrder();
bool probe128_canonicalBridgeFrictionOrder();
bool probe129_contactPcgFixedPairDrop();
bool probe130_contactPcgRevolutePair();
bool probe131_contactPcgPrismaticPair();
bool probe132_contactPcgLinearDrivePair();
bool probe133_contactPcgLinearAccelerationDrivePair();
bool probe134_contactPcgTwistDrivePair();
bool probe135_contactPcgSwingDrivePairs();
bool probe136_contactPcgSlerpDrivePair();

int gTestsPassed = 0;
int gTestsFailed = 0;

int main(int argc, char **argv) {
  if (argc == 2 &&
      std::string(argv[1]) ==
          "--probe=transient-broad-material") {
    return test152_transientRestitutionBroadMaterialComponentAuthority()
               ? 0
               : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) ==
          "--probe=scalable-broad-material") {
    return probe153_scalableBroadMaterialComponentAccelerated()
               ? 0
               : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) ==
          "--probe=broad-material-scaling-authority") {
    return probe154_broadMaterialComponentScalingAuthority()
               ? 0
               : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) ==
          "--probe=material-interface-wrench-authority") {
    return probe155_materialInterfaceWrenchAuthority() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) ==
          "--probe=material-multilevel-graph-authority") {
    return probe156_materialMultilevelGraphAuthority() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) ==
          "--probe=material-spatial-transfer-authority") {
    return probe157_materialSpatialTransferAuthority() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=canonical-pyramid-friction") {
    return probe127_canonicalPyramidFrictionOrder() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=canonical-bridge-friction") {
    return probe128_canonicalBridgeFrictionOrder() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=contact-pcg-fixed-pair") {
    return probe129_contactPcgFixedPairDrop() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=contact-pcg-revolute-pair") {
    return probe130_contactPcgRevolutePair() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=contact-pcg-prismatic-pair") {
    return probe131_contactPcgPrismaticPair() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=contact-pcg-linear-drive") {
    return probe132_contactPcgLinearDrivePair() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) ==
          "--probe=contact-pcg-linear-acceleration-drive") {
    return probe133_contactPcgLinearAccelerationDrivePair() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=contact-pcg-twist-drive") {
    return probe134_contactPcgTwistDrivePair() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=contact-pcg-swing-drives") {
    return probe135_contactPcgSwingDrivePairs() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=contact-pcg-slerp-drive") {
    return probe136_contactPcgSlerpDrivePair() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--probe=body-unilateral-projection") {
    return test141_bodyUnilateralProjectionAuthority() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) ==
          "--probe=component-unilateral-projection") {
    return test142_componentUnilateralProjectionAuthority() ? 0 : 1;
  }
  if (argc == 2 &&
      std::string(argv[1]) ==
          "--probe=bounded-component-position-impulse") {
    return test143_boundedComponentPositionImpulseAuthority() ? 0 : 1;
  }
  const bool contactCandidateSuite =
      argc == 2 &&
      std::string(argv[1]) == "--suite=canonical-contact-pcg";
  const bool contactLegacyAuthoringSuite =
      argc == 2 &&
      std::string(argv[1]) == "--suite=contact-pcg-legacy-authoring";
  if (argc != 1 && !contactCandidateSuite &&
      !contactLegacyAuthoringSuite) {
    fprintf(stderr,
            "Usage: avbd_test.exe "
            "[--probe=canonical-pyramid-friction|"
            "--probe=canonical-bridge-friction|"
            "--probe=contact-pcg-fixed-pair|"
            "--probe=contact-pcg-revolute-pair|"
            "--probe=contact-pcg-prismatic-pair|"
            "--probe=contact-pcg-linear-drive|"
            "--probe=contact-pcg-linear-acceleration-drive|"
            "--probe=contact-pcg-twist-drive|"
            "--probe=contact-pcg-swing-drives|"
            "--probe=contact-pcg-slerp-drive|"
            "--probe=body-unilateral-projection|"
            "--probe=component-unilateral-projection|"
            "--probe=bounded-component-position-impulse|"
            "--probe=transient-broad-material|"
            "--probe=scalable-broad-material|"
            "--probe=broad-material-scaling-authority|"
            "--probe=material-interface-wrench-authority|"
            "--probe=material-multilevel-graph-authority|"
            "--probe=material-spatial-transfer-authority|"
            "--suite=canonical-contact-pcg|"
            "--suite=contact-pcg-legacy-authoring]\n");
    return 2;
  }
  AvbdRef::setContactIslandPcgSuiteProbeEnabled(
      contactCandidateSuite || contactLegacyAuthoringSuite);
  AvbdRef::setCanonicalRigidContactAuthoringSuiteProbeEnabled(
      contactCandidateSuite);
  printf("=========================================\n");
  printf("Running AVBD Refactored Tests (149 Cases, contact=%s, authoring=%s)\n",
         contactCandidateSuite || contactLegacyAuthoringSuite
             ? "island-pcg"
             : "baseline",
         contactCandidateSuite ? "canonical" : "legacy");
  printf("=========================================\n");

  test1_singleBoxOnGround();
  test2_twoBoxStack();
  test3_fiveBoxTower();
  test4_pyramid();
  test5_dropFromHeight();
  test6_perFrameRegenWithCache();
  test7_physxScale();
  test8_asymmetricMass();
  test9_tenBoxTower();
  test10_longTermStability();

  test11_collisionSingleBox();
  test12_collisionThreeStack();
  test13_collisionDrop();
  test14_collisionPhysxTower();
  test15_pyramidStack();
  test16_pyramidNoFriction();

  test17_sphericalJointChain();
  test18_fixedJointChain();
  test19_d6JointChain();
  test20_d6JointChain_snippetJoint();
  test21_highMassRatioChain();
  test22_meshChainmail();
  test23_heavyBallOnMesh();
  test24_fastBallOnChainmail();
  test25_smallBallOnChainmail();
  test26_snippetChainmailReplica();
  test27_joints3x3Solve();

  // Drive tests: linearX
  test28_linearX_default();
  test29_linearX_rotFrameA();
  test30_linearX_rotBodyB();
  test31_linearX_rotBoth();
  // Drive tests: twist
  test32_twist_default();
  test33_twist_rotFrameA();
  test34_twist_rotBodyB();
  test35_twist_rotBoth();
  // Drive tests: swing1
  test36_swing1_default();
  test37_swing1_rotFrameA();
  test38_swing1_rotBodyB();
  test39_swing1_rotBoth();
  // Drive tests: SLERP
  test40_slerp_default();
  test41_slerp_rotFrameA();
  test42_slerp_rotBodyB();
  test43_slerp_rotBoth();
  test44_sphericalConeLimit();
  test45_gearJoint_basicRatio();
  test46_gearJoint_oppositeDir();
  test47_prismaticJoint_basic();
  // Prismatic chain tests
  test49_prismaticChain_6x6();

  // Revolute joint tests
  test51_revoluteJoint_basic();
  test52_revoluteJoint_limit();
  test53_revoluteJoint_drive();
  test54_revoluteJoint_axisAlign();
  test55_revoluteJoint_jitterRepro();
  // Friction tests
  test56_tiltedPlane_zeroFriction();
  test57_tiltedPlane_highFriction();
  test58_frictionComparison_lowVsHigh();
  test59_zeroFriction_noDeceleration();
  test60_highFriction_stopsQuickly();
  test61_pyramidFrictionStability();
  test62_stackedBoxOffset_frictionHolds();
  test63_lateralPush_frictionResists();
  test64_frictionIsotropy();
  test65_dynamicDynamicFriction();
  test66_massRatioFriction();
  test67_frictionSweep_monotonic();
  test68_rotationalFriction();
  test69_restingContactNoDrift();
  test70_tangentDirection_negativeX();
  test71_coulombCone_noExplosion();
  test72_geometricMeanFriction();
  test73_longTermFrictionStability();

  // Articulation tests (pure AVBD AL constraints)
  test74_articulationPendulum();
  test75_articulationChain5();
  test76_articulationOnGround();
  test77_articulationWithLimits();
  test78_articulationSpherical();
  test79_articulationFixed();
  test80_articulationPrismatic();
  test81_articulationPrismaticLimits();
  test82_articulationPDDrive();
  test83_articulationJointFriction();
  test84_articulationConstraintAccuracy();
  test85_articulationMixedJoints();
  test86_articulationFloatingBase();
  test87_articulationBranching();
  test88_articulationVelocityDrive();
  test89_articulationMassRatio();
  test90_articulationDriveGravComp();
  test91_articulationIDExtraction();
  test92_articulationEndEffectorIK();
  test93_articulationLongChain();
  test94_articulationPrismaticDriveTracking();
  test95_articulationMultiArticulation();
  test96_articulationFloatingBaseMomentum();
  test97_articulationMimicJoint();

  // Phase 3: Convergence & Performance
  test98_convergenceBenchmark();
  test99_treeSweepConvergence();
  test100_andersonAcceleration();
  test101_chebyshevSemiIterative();
  // Phase 4: Scissor Lift Validation
  test102_articulationD6LoopClosure();
  test103_scissorLiftValidation();

  // Soft body tests
  test104_softBodyFreeFall();
  test105_softBodyGroundSettle();
  test106_softBodyVolumePreservation();
  test107_softBodyMaterialStiffness();
  test108_softBodyLongTermStability();
  test109_softBodyStacked();
  test110_softBodyConvergence();
  test111_softBodyToppling();
  test112_softBodyAngularMomentum();

  test114_deformableSphereShot_sequential_gate();
  test115_deformableAggregated_noFriction6x6();
  test116_deformableFriction_dominantSequential();
  test117_deformableStaticAnchor_motion();
  test118_boxOnGround_aggregatedUnchangedBySequentialMode();
  test119_kinematicShell_sphereShot();
  test120_kinematicShell_stressHarness();
  test121_kinematicShell_vs_staticAnchor_sphereShot();
  test122_prismaticReverseEndpointFrameA();
  test123_d6VelocityDriveReverseEndpointFrameA();
  test124_d6LockedLinearReactionWriteback();
  test125_d6OffsetCoupledReaction();
  test126_matrixFreeIslandOperator();
  test127_linearPositionDriveDiscreteEquation();
  test128_linearPositionDriveOutputForceSemantics();
  test129_angularTwistVelocityDriveOutputForceSemantics();
  test130_angularSwing1VelocityDriveOutputForceSemantics();
  test131_angularSwing2VelocityDriveOutputForceSemantics();
  test132_angularSlerpVelocityDriveOutputForceSemantics();
  test133_linearPositionDriveOffsetMomentSemantics();
  test134_angularTwistPositionDriveDiscreteEquation();
  test135_linearAccelerationDriveEffectiveMassSemantics();
  test136_angularSwing1PositionDriveDiscreteEquation();
  test137_angularSwing2PositionDriveDiscreteEquation();
  test138_angularSlerpPositionDriveDiscreteEquation();
  test139_gearJointImpulseProjectionDiscreteEquation();
  test140_dynamicDynamicAngularPositionDriveDiscreteEquation();
  test141_bodyUnilateralProjectionAuthority();
  test142_componentUnilateralProjectionAuthority();
  test143_boundedComponentPositionImpulseAuthority();
  test144_supportAxisRigidNullModeAuthority();
  test145_finiteDriveStaticFrictionPositionAuthority();
  test146_unequalMassFrictionWeightShareAuthority();
  test147_slopedSupportFrictionPositionAuthority();
  test148_contactPositionOutputForceOwnerAuthority();
  test149_passiveRigidStaticFrictionManifoldAuthority();
  test150_passiveRigidContactMaterialComponentAuthority();
  test151_restitutionRigidMaterialComponentAuthority();
  test152_transientRestitutionBroadMaterialComponentAuthority();

  printf("\n=========================================\n");
  printf("Tests Passed: %d\n", gTestsPassed);
  printf("Tests Failed: %d\n", gTestsFailed);
  printf("=========================================\n");

  return gTestsFailed > 0 ? 1 : 0;
}
