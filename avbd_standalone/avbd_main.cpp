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

// Soft body / Cloth
bool test104_softBodySingleTet();
bool test105_softBodyCubeDrop();
bool test106_softBodyDistanceOnly();
bool test107_softBodyVolumePreserve();
bool test108_softBodyStackOnGround();
bool test109_softBodyMultiple();
bool test110_clothDrape();
bool test111_clothBendingStiffness();
bool test112_clothPinnedCorners();
bool test113_softRigidAttach();
bool test114_softOnRigidBox();
bool test115_kinematicPinOscillate();
bool test116_rigidFallOnSoft();
bool test117_materialStiffness();
bool test118_materialPoisson();
bool test119_convergenceSoftBench();
bool test120_unifiedScene();

int gTestsPassed = 0;
int gTestsFailed = 0;

int main() {
  printf("=========================================\n");
  printf("Running AVBD Refactored Tests (48 Cases)\n");
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

  // Soft body / Cloth tests
  test104_softBodySingleTet();
  test105_softBodyCubeDrop();
  test106_softBodyDistanceOnly();
  test107_softBodyVolumePreserve();
  test108_softBodyStackOnGround();
  test109_softBodyMultiple();
  test110_clothDrape();
  test111_clothBendingStiffness();
  test112_clothPinnedCorners();
  test113_softRigidAttach();
  test114_softOnRigidBox();
  test115_kinematicPinOscillate();
  test116_rigidFallOnSoft();
  test117_materialStiffness();
  test118_materialPoisson();
  test119_convergenceSoftBench();
  test120_unifiedScene();

  printf("\n=========================================\n");
  printf("Tests Passed: %d\n", gTestsPassed);
  printf("Tests Failed: %d\n", gTestsFailed);
  printf("=========================================\n");

  return gTestsFailed > 0 ? 1 : 0;
}
