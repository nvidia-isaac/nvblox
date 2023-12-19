#include <gtest/gtest.h>

#include <string>

#include "nvblox/integrators/weighting_function.h"
#include "nvblox/tests/weighting_utils.h"

using namespace nvblox;

constexpr float kEpsilon = 1e-4;

class WeightingFunctionTest : public ::testing::Test {
 protected:
  WeightingFunctionTest() {}

  void SetUp() override {}

  unified_ptr<WeightingFunction> weighting_function_;
};

TEST_F(WeightingFunctionTest, TestConstantWeight) {
  weighting_function_ = test_utils::createWeightingFunction(
      WeightingFunctionType::kConstantWeight);

  float surface_distance_from_camera = 10.0f;
  float voxel_distance_from_camera = 10.0f;
  float truncation_distance = 1.0f;

  float weight1 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 8.0f;
  float weight2 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 11.0f;
  float weight3 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  EXPECT_NEAR(weight1, 1.0f, kEpsilon);
  EXPECT_NEAR(weight2, 1.0f, kEpsilon);
  EXPECT_NEAR(weight3, 1.0f, kEpsilon);
}

TEST_F(WeightingFunctionTest, TestConstantDropoffWeight) {
  weighting_function_ = test_utils::createWeightingFunction(
      WeightingFunctionType::kConstantDropoffWeight);

  float surface_distance_from_camera = 10.0f;
  float voxel_distance_from_camera = 10.0f;
  float truncation_distance = 1.0f;

  float weight1 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 8.0f;
  float weight2 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 11.0f;
  float weight3 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 10.5f;
  float weight4 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 0.0f;
  float weight5 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  EXPECT_NEAR(weight1, 1.0f, kEpsilon);
  EXPECT_NEAR(weight2, 1.0f, kEpsilon);
  EXPECT_NEAR(weight3, 0.0f, kEpsilon);
  EXPECT_NEAR(weight4, 0.5f, kEpsilon);
  EXPECT_NEAR(weight5, 1.0f, kEpsilon);
}

TEST_F(WeightingFunctionTest, TestInverseSquare) {
  weighting_function_ = test_utils::createWeightingFunction(
      WeightingFunctionType::kInverseSquareWeight);

  float surface_distance_from_camera = 10.0f;
  float voxel_distance_from_camera = 10.0f;
  float truncation_distance = 1.0f;

  float weight1 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 5.0f;
  float weight2 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 11.0f;
  float weight3 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 0.0f;
  float weight4 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  EXPECT_NEAR(weight1, 0.01f, kEpsilon);
  EXPECT_NEAR(weight2, 0.04f, kEpsilon);
  EXPECT_NEAR(weight3, 0.0f, kEpsilon);
  EXPECT_NEAR(weight4, 1.0f, kEpsilon);
}

TEST_F(WeightingFunctionTest, TestLinearWithMax) {
  weighting_function_ = test_utils::createWeightingFunction(
      WeightingFunctionType::kLinearWithMax);

  float surface_distance_from_camera = 10.0f;
  float truncation_distance = 1.0f;

  float voxel_distance_from_camera = 0.0f;
  float weight1 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 0.5f;
  float weight2 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 1.0f;
  float weight3 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  voxel_distance_from_camera = 5.0f;
  float weight4 = test_utils::computeWeight(
      weighting_function_, surface_distance_from_camera,
      voxel_distance_from_camera, truncation_distance);

  EXPECT_NEAR(weight1, 1.0f, kEpsilon);
  EXPECT_NEAR(weight2, 1.0f, kEpsilon);
  EXPECT_NEAR(weight3, 1.0f, kEpsilon);
  EXPECT_NEAR(weight4, 0.2f, kEpsilon);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
