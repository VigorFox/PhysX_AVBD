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

int gTestsPassed = 0;
int gTestsFailed = 0;

int main() {
  printf("=========================================\n");
  printf("Running AVBD Refactored Tests (43 Cases)\n");
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

  printf("\n=========================================\n");
  printf("Tests Passed: %d\n", gTestsPassed);
  printf("Tests Failed: %d\n", gTestsFailed);
  printf("=========================================\n");

  return gTestsFailed > 0 ? 1 : 0;
}
