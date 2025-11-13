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
#include "nvblox/tests/sensor_fixture.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

constexpr float kAcceptablePercentageOverThreshold = 2.5;

template <typename SensorType>
class OccupancyIntegratorTestFixture
    : public test_utils::SensorFixture<SensorType> {
 protected:
  OccupancyIntegratorTestFixture()
      : layer_(voxel_size_m_, MemoryType::kUnified) {}

  // Test layer
  constexpr static float voxel_size_m_ = 0.1;
  OccupancyLayer layer_;
};

using SensorTypes =
    ::testing::Types<Camera, Lidar, test_utils::CustomCameraSensor>;
TYPED_TEST_SUITE(OccupancyIntegratorTestFixture, SensorTypes);

TYPED_TEST(OccupancyIntegratorTestFixture, ReconstructPlane) {
  // Make sure this is deterministic.
  std::srand(0);

  // Plane centered at (0,0,depth) with random (slight) slant
  const Vector3f direction =
      Vector3f(test_utils::randomFloatInRange(-0.25, 0.25),
               test_utils::randomFloatInRange(-0.25, 0.25), -1.0f);
  const primitives::Plane plane =
      primitives::Plane(Vector3f(0.0f, 0.0f, 5.f), direction.normalized());

  // Get a depth map of our view of the plane.
  const DepthImage depth_frame =
      test_utils::getPlaneDepthImage(plane, this->sensor());

  if (FLAGS_nvblox_test_file_output) {
    std::string filepath = "./depth_frame_occupancy_test.png";
    io::writeToPng(filepath, depth_frame);
  }

  // Integrate into a layer
  std::unique_ptr<ProjectiveOccupancyIntegrator> integrator_ptr;
  integrator_ptr = std::make_unique<ProjectiveOccupancyIntegrator>();

  const Transform T_L_C = Transform::Identity();
  integrator_ptr->max_integration_distance_m(10.);
  integrator_ptr->truncation_distance_vox(10.f);
  integrator_ptr->integrateFrame(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_L_C,
      this->sensor(), &this->layer_);

  // Sample some points on the plane, within the camera view.
  constexpr int kNumberOfPointsToCheck = 1000;
  const Eigen::MatrixX2f u_random_C =
      test_utils::getRandomPixelLocationsWhereDepthIsValid(
          kNumberOfPointsToCheck, this->sensor(), depth_frame);
  const Eigen::MatrixX3f p_check_L = test_utils::backProjectVectorized(
      u_random_C, depth_frame, this->sensor());

  // Get the distance of these surface points
  std::vector<Vector3f> points_L;
  points_L.reserve(p_check_L.rows());
  for (int i = 0; i < p_check_L.rows(); i++) {
    points_L.push_back(p_check_L.row(i));
  }
  std::vector<float> probabilities;
  std::vector<bool> success_flags;
  interpolation::interpolateOnCPU(points_L, this->layer_, &probabilities,
                                  &success_flags);
  EXPECT_EQ(success_flags.size(), kNumberOfPointsToCheck);
  EXPECT_EQ(probabilities.size(), success_flags.size());

  // Check that all interpolations worked and that the probability increased.
  // Lidar sensor have low spatial resolution which results in sparsely
  // populated voxels that doesn't lend themselves well to interpolation.
  // Therefore the interpolation test is only performed for Camera sensors.
  if constexpr (std::is_same<TypeParam, Camera>::value) {
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
  }

  if (FLAGS_nvblox_test_file_output) {
    io::outputVoxelLayerToPly(this->layer_, "occupancy_layer.ply");
  }
}

TYPED_TEST(OccupancyIntegratorTestFixture, SphereSceneTest) {
  constexpr float kTrajectoryRadius = 4.0f;
  constexpr float kTrajectoryHeight = 2.0f;
  constexpr int kNumTrajectoryPoints = 80;
  constexpr float kTruncationDistanceVox = 2;
  constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * this->voxel_size_m_;

  // Get the ground truth SDF of a sphere in a box.
  primitives::Scene scene = test_utils::getSphereInBox();
  OccupancyLayer gt_layer(this->voxel_size_m_, MemoryType::kUnified);
  scene.generateLayerFromScene(kTruncationDistanceMeters, &gt_layer);

  // Create an integrator.
  ProjectiveOccupancyIntegrator integrator;
  integrator.truncation_distance_vox(kTruncationDistanceVox);

  // Simulate a trajectory of the requisite amount of points, on the circle
  // around the sphere.
  const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);

  // Create a depth frame. We share this memory buffer for the entire
  // trajectory.
  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);

  OccupancyLayer layer_gpu(this->layer_.voxel_size(), MemoryType::kUnified);

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
    scene.generateDepthImageFromScene(this->sensor(), T_S_C, kMaxDist,
                                      &depth_frame);

    // Integrate this depth image.
    integrator.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), &layer_gpu);
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
    io::writeToPng("depth_frame_occupancy_sphere.png", depth_frame);
    io::outputVoxelLayerToPly(gt_layer, "occupancy_sphere_gt.ply");
    io::outputVoxelLayerToPly(layer_gpu, "occupancy_sphere.ply");
  }
}

TEST(OccupancyIntegratorTest, MarkUnobservedFree) {
  constexpr float voxel_size_m = 0.1;
  OccupancyLayer occupancy_layer(voxel_size_m, MemoryType::kUnified);

  EXPECT_EQ(occupancy_layer.numBlocks(), 0);

  // Do the observation.
  const Vector3f center(0.0, 0.0, 0.0);
  const float radius = 1.0;

  ProjectiveOccupancyIntegrator integrator;
  integrator.markUnobservedFreeInsideRadius(center, radius, &occupancy_layer);

  // Check some blocks got allocated
  CHECK_GT(occupancy_layer.numBlocks(), 0);

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

TYPED_TEST(OccupancyIntegratorTestFixture, MaskedDepthPixels) {
  // Generate a depth image of a sphere
  constexpr float kSphereRadius = 2.f;
  constexpr float kSphereZPos = 5.f;
  constexpr float kMaxDist = kSphereZPos;
  primitives::Scene scene;
  scene.addPrimitive(std::make_unique<primitives::Sphere>(
      Vector3f(0.0f, 0.0f, kSphereZPos), kSphereRadius));
  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);
  scene.generateDepthImageFromScene(this->sensor(), Transform::Identity(),
                                    kMaxDist, &depth_frame, kMaxDist);

  // Mask all depth pixels below a threshold.
  constexpr float kMaskDepthThreshold = kSphereZPos - 0.8 * kSphereRadius;
  MonoImage mask(depth_frame.rows(), depth_frame.cols(), MemoryType::kUnified);
  for (int y = 0; y < depth_frame.rows(); ++y)
    for (int x = 0; x < depth_frame.cols(); ++x) {
      const float depth = depth_frame(y, x);
      if (depth < kMaskDepthThreshold) {
        mask(y, x) = 255;
      }
    }

  // Integrate masked depth image
  ProjectiveOccupancyIntegrator integrator;
  integrator.integrateFrame({depth_frame, mask}, Transform::Identity(),
                            this->sensor(), &this->layer_);

  // Check that all voxels projected into the unmasked area are integrated
  // as "unobserved", i.e. log_odds = 0
  int num_checked = 0;
  std::vector<Index3D> block_indices = this->layer_.getAllBlockIndices();
  for (auto& block_index : block_indices) {
    auto block_ptr = this->layer_.getBlockAtIndex(block_index);
    for (auto voxel_itr = block_ptr->begin(); voxel_itr != block_ptr->end();
         ++voxel_itr) {
      Vector3f voxel_pos = getCenterPositionFromBlockIndexAndVoxelIndex(
          this->layer_.block_size(), block_index, voxel_itr.index());
      Vector2f p2d;
      if (this->sensor().project(voxel_pos, &p2d)) {
        const Index2D pixel_pos = p2d.array().floor().cast<int>();
        if (((p2d - pixel_pos.cast<float>()).array() > 1.f - 1e-4f).any()) {
          // Ignore voxels that project very close to pixel boundaries.
          // This is needed because of rounding errors on jetson platforms.
          continue;
        }
        if (!mask(pixel_pos.y(), pixel_pos.x())) {
          EXPECT_EQ(voxel_itr->log_odds, 0.f);
          ++num_checked;
        }
      }
    }
  }
  // Sanity check that we actually had some unmasked voxels
  EXPECT_GT(num_checked, 0);
}

TYPED_TEST(OccupancyIntegratorTestFixture, InvalidDepthHandling) {
  // Test that invalid depth values are not integrated
  // Unlike TSDF, occupancy has no invalid_depth_decay_factor - invalid pixels
  // simply skip integration

  // Use sensor's actual dimensions
  const int kImageWidth = this->sensor().width();
  const int kImageHeight = this->sensor().height();
  DepthImage depth_image(kImageHeight, kImageWidth, MemoryType::kUnified);

  // Helper to set all depth values
  auto set_all_depth = [&](float value) {
    for (int i = 0; i < depth_image.numel(); i++) {
      depth_image(i) = value;
    }
  };

  auto count_voxels_integrated = [](const OccupancyLayer& layer) {
    int count = 0;
    callFunctionOnAllVoxels<OccupancyVoxel>(
        layer,
        [&](const Index3D&, const Index3D&, const OccupancyVoxel* voxel) {
          // Voxel is integrated if its log_odds changed from default (0)
          if (std::abs(voxel->log_odds) > 1e-6f) ++count;
        });
    return count;
  };

  ProjectiveOccupancyIntegrator integrator;

  // Test 1: All pixels invalid - no voxels should be integrated
  // Test different types of invalid depth values
  const std::vector<float> invalid_values = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -1.0f,
      0.0f,
      -10.0f};

  for (const float invalid_value : invalid_values) {
    set_all_depth(invalid_value);

    integrator.integrateFrame(
        MaskedDepthImageConstView(depth_image, kMaskActiveEverywhere),
        Transform::Identity(), this->sensor(), &this->layer_, nullptr);

    const int voxels_integrated = count_voxels_integrated(this->layer_);
    EXPECT_EQ(voxels_integrated, 0)
        << "No voxels should be integrated for invalid depth value: "
        << invalid_value;
  }

  // Test 2: Valid depth - some voxels should be integrated
  set_all_depth(2.0f);

  integrator.integrateFrame(
      MaskedDepthImageConstView(depth_image, kMaskActiveEverywhere),
      Transform::Identity(), this->sensor(), &this->layer_, nullptr);

  const int voxels_integrated_after_valid =
      count_voxels_integrated(this->layer_);
  EXPECT_GT(voxels_integrated_after_valid, 0)
      << "Some voxels should be integrated after valid depth";

  // Test 3: Invalid depth after valid - no change (no decay for invalid depth
  // in occupancy) Store the current state
  std::vector<float> log_odds_before;
  callFunctionOnAllVoxels<OccupancyVoxel>(
      this->layer_,
      [&](const Index3D&, const Index3D&, const OccupancyVoxel* voxel) {
        log_odds_before.push_back(voxel->log_odds);
      });

  set_all_depth(std::numeric_limits<float>::infinity());

  integrator.integrateFrame(
      MaskedDepthImageConstView(depth_image, kMaskActiveEverywhere),
      Transform::Identity(), this->sensor(), &this->layer_, nullptr);

  // Verify log_odds values haven't changed
  int idx = 0;
  callFunctionOnAllVoxels<OccupancyVoxel>(
      this->layer_,
      [&](const Index3D&, const Index3D&, const OccupancyVoxel* voxel) {
        EXPECT_FLOAT_EQ(voxel->log_odds, log_odds_before[idx])
            << "Log odds should not change when integrating invalid depth "
            << "(occupancy has no decay factor)";
        idx++;
      });
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
