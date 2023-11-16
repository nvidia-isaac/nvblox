/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <gtest/gtest.h>
#include <cmath>

#include "nvblox/integrators/projective_occupancy_integrator.h"
#include "nvblox/interpolation/interpolation_3d.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/primitives/scene.h"

#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

constexpr float kAcceptablePercentageOverThreshold = 2.0;  // 2.0 %

class OccupancyIntegratorTest : public ::testing::Test {
 protected:
  OccupancyIntegratorTest()
      : layer_(voxel_size_m_, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  // Test layer
  constexpr static float voxel_size_m_ = 0.1;
  OccupancyLayer layer_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

TEST_F(OccupancyIntegratorTest, ReconstructPlane) {
  // Make sure this is deterministic.
  std::srand(0);

  // Plane centered at (0,0,depth) with random (slight) slant
  const test_utils::Plane plane = test_utils::Plane(
      Vector3f(0.0f, 0.0f, 5.0f),
      Vector3f(test_utils::randomFloatInRange(-0.25, 0.25),
               test_utils::randomFloatInRange(-0.25, 0.25), -1.0f));

  // // Get a depth map of our view of the plane.
  const DepthImage depth_frame = test_utils::getDepthImage(plane, camera_);

  if (FLAGS_nvblox_test_file_output) {
    std::string filepath = "./depth_frame_occupancy_test.png";
    io::writeToPng(filepath, depth_frame);
  }

  // Integrate into a layer
  std::unique_ptr<ProjectiveOccupancyIntegrator> integrator_ptr;
  integrator_ptr = std::make_unique<ProjectiveOccupancyIntegrator>();

  const Transform T_L_C = Transform::Identity();

  integrator_ptr->truncation_distance_vox(10.0f);
  integrator_ptr->integrateFrame(depth_frame, T_L_C, camera_, &layer_);

  // // Sample some points on the plane, within the camera view.
  constexpr int kNumberOfPointsToCheck = 1000;
  const Eigen::MatrixX2f u_random_C =
      test_utils::getRandomPixelLocations(kNumberOfPointsToCheck, camera_);
  const Eigen::MatrixX3f p_check_L =
      test_utils::backProjectToPlaneVectorized(u_random_C, plane, camera_);

  // // Get the distance of these surface points
  std::vector<Vector3f> points_L;
  points_L.reserve(p_check_L.rows());
  for (int i = 0; i < p_check_L.rows(); i++) {
    points_L.push_back(p_check_L.row(i));
  }
  std::vector<float> probabilities;
  std::vector<bool> success_flags;
  interpolation::interpolateOnCPU(points_L, layer_, &probabilities,
                                  &success_flags);
  EXPECT_EQ(success_flags.size(), kNumberOfPointsToCheck);
  EXPECT_EQ(probabilities.size(), success_flags.size());

  // Check that all interpolations worked and that the probability increased
  int num_failures = 0;
  int num_bad_flags = 0;
  for (size_t i = 0; i < probabilities.size(); i++) {
    EXPECT_TRUE(success_flags[i]);
    if (!success_flags[i]) {
      num_bad_flags++;
    }
    EXPECT_GT(probabilities[i], 0.5f);
    if (probabilities[i] <= 0.5f) {
      num_failures++;
    }
  }
  LOG(INFO) << "Num of invalid points: " << num_failures;
  LOG(INFO) << "num_bad_flags: " << num_bad_flags << " / "
            << probabilities.size();

  if (FLAGS_nvblox_test_file_output) {
    io::outputVoxelLayerToPly(layer_, "occupancy_layer.ply");
  }
}

TEST_F(OccupancyIntegratorTest, SphereSceneTest) {
  constexpr float kTrajectoryRadius = 4.0f;
  constexpr float kTrajectoryHeight = 2.0f;
  constexpr int kNumTrajectoryPoints = 80;
  constexpr float kTruncationDistanceVox = 2;
  constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * voxel_size_m_;

  // Get the ground truth SDF of a sphere in a box.
  primitives::Scene scene = test_utils::getSphereInBox();
  OccupancyLayer gt_layer(voxel_size_m_, MemoryType::kUnified);
  scene.generateLayerFromScene(kTruncationDistanceMeters, &gt_layer);

  // Create an integrator.
  ProjectiveOccupancyIntegrator integrator;
  integrator.truncation_distance_vox(kTruncationDistanceVox);

  // Simulate a trajectory of the requisite amount of points, on the circle
  // around the sphere.
  const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);

  // Create a depth frame. We share this memory buffer for the entire
  // trajectory.
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);

  OccupancyLayer layer_gpu(layer_.voxel_size(), MemoryType::kUnified);

  for (size_t i = 0; i < kNumTrajectoryPoints; i++) {
    const float theta = radians_increment * i;
    // Convert polar to cartesian coordinates.
    Vector3f cartesian_coordinates(kTrajectoryRadius * std::cos(theta),
                                   kTrajectoryRadius * std::sin(theta),
                                   kTrajectoryHeight);
    // The camera has its z axis pointing towards the origin.
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    // Construct a transform from camera to scene with this.
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(cartesian_coordinates);

    // Generate a depth image of the scene.
    constexpr float kMaxDist = 10.0;
    scene.generateDepthImageFromScene(camera_, T_S_C, kMaxDist, &depth_frame);

    // Integrate this depth image.
    integrator.integrateFrame(depth_frame, T_S_C, camera_, &layer_gpu);
  }

  // Now do some checks...
  // Check every voxel in the map.
  int total_num_voxels = 0;
  int num_voxel_invalid = 0;
  auto lambda = [&](const Index3D& block_index, const Index3D& voxel_index,
                    const OccupancyVoxel* voxel) {
    const float probability = probabilityFromLogOdds(voxel->log_odds);
    // Get the corresponding point from the GT layer.
    const OccupancyVoxel* gt_voxel =
        getVoxelAtBlockAndVoxelIndex<OccupancyVoxel>(gt_layer, block_index,
                                                     voxel_index);
    if (gt_voxel != nullptr) {
      const float gt_probability = probabilityFromLogOdds(gt_voxel->log_odds);
      const bool false_negative = gt_probability >= 0.9f && probability < 0.5f;
      const bool false_positive = gt_probability <= 0.1f && probability > 0.5f;
      if (false_positive || false_negative) {
        num_voxel_invalid++;
      }
      total_num_voxels++;
    }
  };
  callFunctionOnAllVoxels<OccupancyVoxel>(layer_gpu, lambda);
  float percentage_invalid = static_cast<float>(num_voxel_invalid) /
                             static_cast<float>(total_num_voxels) * 100.0f;
  EXPECT_LT(percentage_invalid, kAcceptablePercentageOverThreshold);
  std::cout << "num_voxel_invalid: " << num_voxel_invalid << std::endl;
  std::cout << "total_num_voxels: " << total_num_voxels << std::endl;
  std::cout << "percentage_invalid: " << percentage_invalid << std::endl;

  if (FLAGS_nvblox_test_file_output) {
    io::outputVoxelLayerToPly(gt_layer, "occupancy_sphere_gt.ply");
    io::outputVoxelLayerToPly(layer_gpu, "occupancy_sphere.ply");
  }
}

TEST_F(OccupancyIntegratorTest, MarkUnobservedFree) {
  constexpr float voxel_size_m = 0.1;
  OccupancyLayer occupancy_layer(voxel_size_m, MemoryType::kUnified);

  EXPECT_EQ(occupancy_layer.numAllocatedBlocks(), 0);

  // Do the observation.
  const Vector3f center(0.0, 0.0, 0.0);
  const float radius = 1.0;

  ProjectiveOccupancyIntegrator integrator;
  integrator.markUnobservedFreeInsideRadius(center, radius, &occupancy_layer);

  // Check some blocks got allocated
  CHECK_GT(occupancy_layer.numAllocatedBlocks(), 0);

  // Check the blocks
  // If the log_odds is zero, then it means the voxel is unobserved. If it is
  // less than zero then it has been observed and is unoccupied
  callFunctionOnAllVoxels<OccupancyVoxel>(
      occupancy_layer,
      [](const Index3D&, const Index3D&, const OccupancyVoxel* voxel) -> void {
        constexpr float kLogOddsUnobserved = 0;
        EXPECT_LT(voxel->log_odds, kLogOddsUnobserved);
      });
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
