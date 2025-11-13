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

#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/weighting_function.h"
#include "nvblox/interpolation/interpolation_3d.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/tests/custom_camera_sensor.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/projective_tsdf_integrator_cpu.h"
#include "nvblox/tests/sensor_fixture.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

constexpr float kInterpolatedSurfaceDistanceEpsilon = 1e-6;

// Parameters which define a passing test which measures the difference between
// CPU and GPU integration. We accept a test if
// <(kAcceptablePercentageOverThreshold)% voxels have less than
// (kGPUvsCPUDifferenceThresholdM)meters difference.
// NOTE(alexmillane): Obviously there are some slight differences in the voxel
// distance values in a small number of voxels.
constexpr float kGPUvsCPUDifferenceThresholdM = 1e-3;      // 1mm
constexpr float kAcceptablePercentageOverThreshold = 0.4;  // 0.4%

// The tsdf test is templated on both sensor type and device type. Gtest typed
// tests need a single template. Therefore we combine the two into a trait
// struct
struct CameraAndGPUTrait {
  using SensorType = Camera;
  static constexpr DeviceType device_type = DeviceType::kGPU;
};
struct CameraAndCPUTrait {
  using SensorType = Camera;
  static constexpr DeviceType device_type = DeviceType::kCPU;
};

struct CustomCameraAndGPUTrait {
  using SensorType = test_utils::CustomCameraSensor;
  static constexpr DeviceType device_type = DeviceType::kGPU;
};
struct CustomCameraAndCPUTrait {
  using SensorType = test_utils::CustomCameraSensor;
  static constexpr DeviceType device_type = DeviceType::kCPU;
};

struct LidarAndGPUTrait {
  using SensorType = Lidar;
  static constexpr DeviceType device_type = DeviceType::kGPU;
};
struct LidarAndCPUTrait {
  using SensorType = Lidar;
  static constexpr DeviceType device_type = DeviceType::kCPU;
};

template <typename TestTraits>
class TsdfIntegratorTestFixture
    : public test_utils::SensorFixture<typename TestTraits::SensorType> {
 protected:
  TsdfIntegratorTestFixture() : layer_(voxel_size_m_, MemoryType::kUnified) {}

  // Test layer
  constexpr static float voxel_size_m_ = 0.2;
  TsdfLayer layer_;

  // How much error we expect on the surface
  constexpr static float surface_reconstruction_allowable_distance_error_vox_ =
      2.0f;
  constexpr static float surface_reconstruction_allowable_distance_error_m_ =
      surface_reconstruction_allowable_distance_error_vox_ * voxel_size_m_;

  std::unique_ptr<ProjectiveTsdfIntegrator> create_integrator() {
    if (TestTraits::device_type == DeviceType::kCPU) {
      return std::make_unique<ProjectiveTsdfIntegratorCPU>();
    } else {
      return std::make_unique<ProjectiveTsdfIntegrator>();
    }
  }
};

using TsdfIntegratorTestTypes =
    ::testing::Types<CameraAndGPUTrait, CameraAndCPUTrait, LidarAndGPUTrait,
                     CustomCameraAndCPUTrait, CustomCameraAndGPUTrait>;
TYPED_TEST_SUITE(TsdfIntegratorTestFixture, TsdfIntegratorTestTypes);

TYPED_TEST(TsdfIntegratorTestFixture, ReconstructPlane) {
  // Make sure this is deterministic.
  std::srand(0);

  // Plane centered at (0,0,depth) with random (slight) slant
  Vector3f direction(test_utils::randomFloatInRange(-0.25, 0.25),
                     test_utils::randomFloatInRange(-0.25, 0.25), -1.0f);

  primitives::Plane plane(Vector3f(0.f, 0.0f, 5.0f), direction.normalized());
  Transform T_L_C = Transform::Identity();
  DepthImage depth_frame =
      test_utils::getPlaneDepthImage(plane, this->sensor(), T_L_C);

  if (FLAGS_nvblox_test_file_output) {
    std::string filepath = "./depth_frame_tsdf_test.png";
    io::writeToPng(filepath, depth_frame);
  }

  // Integrate into a layer
  std::unique_ptr<ProjectiveTsdfIntegrator> integrator_ptr =
      this->create_integrator();

  integrator_ptr->max_integration_distance_m(10.F);
  integrator_ptr->truncation_distance_vox(10.0f);

  integrator_ptr->integrateFrame(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_L_C,
      this->sensor(), &(this->layer_));

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
  std::vector<float> distances;
  std::vector<bool> success_flags;
  interpolation::interpolateOnCPU(points_L, this->layer_, &distances,
                                  &success_flags);
  EXPECT_EQ(success_flags.size(), kNumberOfPointsToCheck);
  EXPECT_EQ(distances.size(), success_flags.size());

  // Check that something actually got integrated
  float max_distance = std::numeric_limits<float>::min();
  auto lambda = [&max_distance](const Index3D&, const Index3D&,
                                const TsdfVoxel* voxel) {
    if (voxel->distance > max_distance) max_distance = voxel->distance;
  };
  callFunctionOnAllVoxels<TsdfVoxel>(this->layer_, lambda);
  const float nothing_integrator_indicator = this->voxel_size_m_;
  EXPECT_GT(max_distance, nothing_integrator_indicator);

  // Check that all interpolations worked and that the distance is close to zero
  int num_failures = 0;
  int num_bad_flags = 0;
  for (size_t i = 0; i < distances.size(); i++) {
    // We only interpolate with the camera. The produced lidar grid is too
    // sparse.
    if constexpr (std::is_same<typename TypeParam::SensorType, Camera>::value) {
      EXPECT_TRUE(success_flags[i]);
      if (!success_flags[i]) {
        num_bad_flags++;
      }
    }
    EXPECT_NEAR(distances[i], 0.0f,
                this->surface_reconstruction_allowable_distance_error_m_);
    if (std::abs(distances[i]) >
        this->surface_reconstruction_allowable_distance_error_m_) {
      num_failures++;
    }
  }
  LOG(INFO) << "Num points greater than allowable distance: " << num_failures;
  LOG(INFO) << "num_bad_flags: " << num_bad_flags << " / " << distances.size();
}

TYPED_TEST(TsdfIntegratorTestFixture, SphereSceneTest) {
  constexpr float kTrajectoryRadius = 4.0f;
  constexpr float kTrajectoryHeight = 2.0f;
  constexpr int kNumTrajectoryPoints = 80;
  constexpr float kTruncationDistanceVox = 2;
  constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * this->voxel_size_m_;
  // Maximum distance to consider for scene generation.
  constexpr float kMaxDist = 10.0;
  constexpr float kMinWeight = 1.0;

  // Tolerance for error.
  constexpr float kDistanceErrorTolerance = kTruncationDistanceMeters;

  // Get the ground truth SDF of a sphere in a box.
  primitives::Scene scene = test_utils::getSphereInBox();
  TsdfLayer gt_layer(this->voxel_size_m_, MemoryType::kUnified);
  scene.generateLayerFromScene(kTruncationDistanceMeters, &gt_layer);

  // Create an integrator.
  ProjectiveTsdfIntegratorCPU integrator_cpu;
  ProjectiveTsdfIntegrator integrator_gpu;
  integrator_cpu.truncation_distance_vox(kTruncationDistanceVox);
  integrator_gpu.truncation_distance_vox(kTruncationDistanceVox);

  // Simulate a trajectory of the requisite amount of points, on the circle
  // around the sphere.
  const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);

  // Create a depth frame. We share this memory buffer for the entire
  // trajectory.
  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);

  // Two layers, one for CPU integration and one for GPU integration
  TsdfLayer layer_cpu(this->layer_.voxel_size(), MemoryType::kUnified);
  TsdfLayer layer_gpu(this->layer_.voxel_size(), MemoryType::kUnified);

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
    scene.generateDepthImageFromScene(this->sensor(), T_S_C, kMaxDist,
                                      &depth_frame);

    // Integrate this depth image.
    integrator_cpu.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), &layer_cpu);
    integrator_gpu.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), &layer_gpu);
  }

  // Now do some checks...
  // Check every voxel in the map.
  int total_num_voxels = 0;
  int num_voxel_big_error = 0;
  auto lambda = [&](const Index3D& block_index, const Index3D& voxel_index,
                    const TsdfVoxel* voxel) {
    if (voxel->weight >= kMinWeight) {
      // Get the corresponding point from the GT layer.
      const TsdfVoxel* gt_voxel = getVoxelAtBlockAndVoxelIndex<TsdfVoxel>(
          gt_layer, block_index, voxel_index);
      if (gt_voxel != nullptr) {
        if (std::fabs(voxel->distance - gt_voxel->distance) >
            kDistanceErrorTolerance) {
          num_voxel_big_error++;
        }
        total_num_voxels++;
      }
    }
  };
  callFunctionOnAllVoxels<TsdfVoxel>(layer_cpu, lambda);
  float percent_large_error = static_cast<float>(num_voxel_big_error) /
                              static_cast<float>(total_num_voxels) * 100.0f;
  std::cout << "CPU: num_voxel_big_error: " << num_voxel_big_error << std::endl;
  std::cout << "CPU: total_num_voxels: " << total_num_voxels << std::endl;
  std::cout << "CPU: percent_large_error: " << percent_large_error << std::endl;
  EXPECT_LT(percent_large_error, kAcceptablePercentageOverThreshold);
  num_voxel_big_error = 0;
  total_num_voxels = 0;
  callFunctionOnAllVoxels<TsdfVoxel>(layer_gpu, lambda);
  percent_large_error = static_cast<float>(num_voxel_big_error) /
                        static_cast<float>(total_num_voxels) * 100.0f;
  EXPECT_LT(percent_large_error, kAcceptablePercentageOverThreshold);
  std::cout << "GPU: num_voxel_big_error: " << num_voxel_big_error << std::endl;
  std::cout << "GPU: total_num_voxels: " << total_num_voxels << std::endl;
  std::cout << "GPU: percent_large_error: " << percent_large_error << std::endl;

  if (FLAGS_nvblox_test_file_output) {
    io::outputVoxelLayerToPly(layer_gpu, "test_tsdf_projective_gpu.ply");
    io::outputVoxelLayerToPly(layer_cpu, "test_tsdf_projective_cpu.ply");
    io::outputVoxelLayerToPly(gt_layer, "test_tsdf_projective_gt.ply");
  }

  // Compare the layers
  ASSERT_GE(layer_cpu.numBlocks(), layer_gpu.numBlocks());
  size_t total_num_voxels_observed = 0;
  size_t num_voxels_over_threshold = 0;
  for (const Index3D& block_index : layer_gpu.getAllBlockIndices()) {
    const auto block_cpu = layer_cpu.getBlockAtIndex(block_index);
    const auto block_gpu = layer_gpu.getBlockAtIndex(block_index);
    for (int x = 0; x < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; x++) {
      for (int y = 0; y < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; y++) {
        for (int z = 0; z < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; z++) {
          const float diff = block_gpu->voxels[x][y][z].distance -
                             block_cpu->voxels[x][y][z].distance;
          // EXPECT_NEAR(diff, 0.0f, kGPUvsCPUDifferenceThresholdM);
          if (block_gpu->voxels[x][y][z].weight > 0.0f) {
            ++total_num_voxels_observed;
            if (std::abs(diff) > kGPUvsCPUDifferenceThresholdM) {
              ++num_voxels_over_threshold;
            }
          }
        }
      }
    }
  }

  const float percentage_over_threshold =
      100.0f * (static_cast<float>(num_voxels_over_threshold) /
                static_cast<float>(total_num_voxels_observed));
  EXPECT_LT(percentage_over_threshold, kAcceptablePercentageOverThreshold);
  LOG(INFO) << "Percentage of voxels with a difference greater than "
            << kGPUvsCPUDifferenceThresholdM << ": "
            << percentage_over_threshold << "%";
  LOG(INFO) << "total_num_voxels_observed: " << total_num_voxels_observed
            << std::endl;
  LOG(INFO) << "num_voxels_over_threshold: " << num_voxels_over_threshold
            << std::endl;
}

TEST(TsdfIntegratorTest, MarkUnobservedFree) {
  constexpr float voxel_size_m = 0.1;
  TsdfLayer tsdf_layer(voxel_size_m, MemoryType::kUnified);

  EXPECT_EQ(tsdf_layer.numBlocks(), 0);

  // Do the observation.
  const Vector3f center(0.0, 0.0, 0.0);
  const float radius = 1.0;

  ProjectiveTsdfIntegrator integrator;
  integrator.markUnobservedFreeInsideRadius(center, radius, &tsdf_layer);

  // Check some blocks got allocated
  CHECK_GT(tsdf_layer.numBlocks(), 0);

  // Check the blocks
  const float truncation_distance_m =
      integrator.truncation_distance_vox() * voxel_size_m;
  callFunctionOnAllVoxels<TsdfVoxel>(
      tsdf_layer,
      [truncation_distance_m](const Index3D&, const Index3D&,
                              const TsdfVoxel* voxel) -> void {
        constexpr float kEps = 0.001;
        EXPECT_NEAR(voxel->distance, truncation_distance_m, kEps);
        EXPECT_GT(voxel->weight, 0.0f);
      });
}

TEST(TsdfIntegratorTest, GettersAndSetters) {
  ProjectiveTsdfIntegrator integrator;
  integrator.max_weight(1.0);
  EXPECT_EQ(integrator.max_weight(), 1.0);
  integrator.marked_unobserved_voxels_distance_m(2.0);
  EXPECT_EQ(integrator.marked_unobserved_voxels_distance_m(), 2.0);
  integrator.marked_unobserved_voxels_weight(3.0);
  EXPECT_EQ(integrator.marked_unobserved_voxels_weight(), 3.0);
  integrator.weighting_function_type(
      WeightingFunctionType::kInverseSquareWeight);
  EXPECT_EQ(integrator.weighting_function_type(),
            WeightingFunctionType::kInverseSquareWeight);
}

TYPED_TEST(TsdfIntegratorTestFixture, WeightingFunction) {
  // Integrator
  std::unique_ptr<ProjectiveTsdfIntegrator> integrator_ptr =
      this->create_integrator();
  integrator_ptr->max_weight(100.0);

  // Check that weighting function gets initialized to the default
  EXPECT_EQ(integrator_ptr->weighting_function_type(),
            kProjectiveIntegratorWeightingModeParamDesc.default_value);

  // Change to constant weight
  integrator_ptr->weighting_function_type(
      WeightingFunctionType::kConstantWeight);

  // Plane centered at (0,0,5)
  Transform T_L_C = Transform::Identity();
  const float kPlaneDistance = 5.0f;
  primitives::Plane plane(Vector3f(0.0f, 0.0f, kPlaneDistance),
                          Vector3f(0.0f, 0.0f, -1.0f));
  DepthImage depth_frame =
      test_utils::getPlaneDepthImage(plane, this->sensor(), T_L_C);

  // Integrate a frame

  std::vector<Index3D> updated_blocks;
  integrator_ptr->integrateFrame(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_L_C,
      this->sensor(), &this->layer_, &updated_blocks);

  // Check that something actually happened
  EXPECT_GT(updated_blocks.size(), 0);

  // Go over the voxels and check that they have the constant weight that we
  // expect.
  int num_voxels_observed = 0;
  for (const Index3D& block_idx : this->layer_.getAllBlockIndices()) {
    // Get each voxel and it's position
    auto block_ptr = this->layer_.getBlockAtIndex(block_idx);
    constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
    for (int x = 0; x < kVoxelsPerSide; x++) {
      for (int y = 0; y < kVoxelsPerSide; y++) {
        for (int z = 0; z < kVoxelsPerSide; z++) {
          // Get the voxel and check it has weight 1.0
          const TsdfVoxel& voxel = block_ptr->voxels[x][y][z];
          constexpr float kFloatEps = 1e-4;
          if (voxel.weight > kFloatEps) {
            ++num_voxels_observed;
            EXPECT_NEAR(voxel.weight, 1.0f, kFloatEps);
          }
        }
      }
    }
  }
  LOG(INFO) << "Number of voxels observed: " << num_voxels_observed;
  EXPECT_GT(num_voxels_observed, 0);

  // Integrate using a different weighting function
  constexpr WeightingFunctionType kTestedWeightFunctionType =
      WeightingFunctionType::kInverseSquareWeight;
  integrator_ptr->weighting_function_type(kTestedWeightFunctionType);
  this->layer_.clear();
  updated_blocks.clear();
  EXPECT_EQ(this->layer_.numBlocks(), 0);
  EXPECT_EQ(updated_blocks.size(), 0);
  integrator_ptr->integrateFrame(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere),
      Transform::Identity(), this->sensor(), &this->layer_, &updated_blocks);
  // Check that something actually happened
  EXPECT_GT(updated_blocks.size(), 0);
  EXPECT_EQ(integrator_ptr->weighting_function_type(),
            kTestedWeightFunctionType);

  // Weighting function to test against
  auto weighting_function = WeightingFunction(kTestedWeightFunctionType);
  CHECK_EQ(static_cast<int>(weighting_function.type()),
           static_cast<int>(kTestedWeightFunctionType));

  num_voxels_observed = 0;
  for (const Index3D& block_idx : this->layer_.getAllBlockIndices()) {
    // Get each voxel and it's position
    auto block_ptr = this->layer_.getBlockAtIndex(block_idx);
    constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
    for (int x = 0; x < kVoxelsPerSide; x++) {
      for (int y = 0; y < kVoxelsPerSide; y++) {
        for (int z = 0; z < kVoxelsPerSide; z++) {
          // Get the voxel
          const TsdfVoxel& voxel = block_ptr->voxels[x][y][z];
          constexpr float kFloatEps = 1e-4;
          if (voxel.weight > kFloatEps) {
            ++num_voxels_observed;

            // Get the depth of the voxel
            const Index3D voxel_idx(x, y, z);
            const Vector3f voxel_center =
                getCenterPositionFromBlockIndexAndVoxelIndex(
                    this->layer_.block_size(), block_idx, voxel_idx);
            const float voxel_depth = this->sensor().getDepth(voxel_center);

            // Look up the depth
            Vector2f uv;
            ASSERT_TRUE(this->sensor().project(voxel_center, &uv));
            const float surface_depth = depth_frame(uv.y(), uv.x());

            // In the camera case, surface depth == plane distance
            if constexpr (std::is_same<typename TypeParam::SensorType,
                                       Camera>::value) {
              ASSERT_NEAR(surface_depth, kPlaneDistance, 1E-5);
            }

            const float weight =
                weighting_function(surface_depth, voxel_depth,
                                   integrator_ptr->get_truncation_distance_m(
                                       this->layer_.voxel_size()));

            // Weight
            EXPECT_NEAR(voxel.weight, weight, kFloatEps);
            CHECK_EQ(
                static_cast<int>(kTestedWeightFunctionType),
                static_cast<int>(WeightingFunctionType::kInverseSquareWeight));

            // Hand computing the inverse square weight and checking it matches
            const float weight_hand_computed =
                1.0f / (voxel_depth * voxel_depth);
            EXPECT_NEAR(voxel.weight, weight_hand_computed, kFloatEps);
          }
        }
      }
    }
  }
  LOG(INFO) << "Number of voxels observed: " << num_voxels_observed;
  EXPECT_GT(num_voxels_observed, 0);
}

TYPED_TEST(TsdfIntegratorTestFixture, mask) {
  // Depth image with a constant depth
  constexpr float kDepth = 5.F;
  primitives::Plane plane(Vector3f(0.f, 0.0f, kDepth),
                          Vector3f(0.f, 0.f, -1.f));
  DepthImage depth_frame =
      test_utils::getPlaneDepthImage(plane, this->sensor());

  // First integrate without mask to create blocks
  std::unique_ptr<ProjectiveTsdfIntegrator> integrator_ptr =
      this->create_integrator();

  std::vector<Index3D> updated_blocks;
  integrator_ptr->integrateFrame(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere),
      Transform::Identity(), this->sensor(), &this->layer_, &updated_blocks);
  EXPECT_GT(updated_blocks.size(), 0);

  // Set distance of all blocks to some reference value
  constexpr float kReference = 99.F;
  auto block_pointers = this->layer_.getAllBlockPointers();
  for (auto block_ptr : block_pointers) {
    for (auto& voxel : (*block_ptr)) {
      voxel.distance = kReference;
    }
  }

  // Now, integrate with empty mask. This should only affect voxels in front of
  // the surface
  MonoImage mask_frame(this->sensor().height(), this->sensor().width(),
                       MemoryType::kHost);
  ASSERT_EQ(mask_frame.cols(), depth_frame.cols());
  ASSERT_EQ(mask_frame.rows(), depth_frame.rows());

  mask_frame.setZeroAsync(CudaStreamOwning());
  integrator_ptr->integrateFrame({depth_frame, mask_frame},
                                 Transform::Identity(), this->sensor(),
                                 &this->layer_, &updated_blocks);
  EXPECT_GT(updated_blocks.size(), 0);

  // Distance of voxels beyond positive truncation distance should equal to
  // reference value
  const float truncation_distance_m =
      integrator_ptr->truncation_distance_vox() * this->voxel_size_m_;
  int num_voxels = 0;
  int num_projected = 0;
  callFunctionOnAllVoxels<TsdfVoxel>(
      this->layer_,
      [&](const Index3D& block_index, const Index3D& voxel_index,
          const TsdfVoxel* voxel) -> void {
        const Vector3f pos = getCenterPositionFromBlockIndexAndVoxelIndex(
            this->layer_.block_size(), block_index, voxel_index);
        constexpr float kEps = 1E-3;
        Vector2f uv;
        ++num_voxels;
        if (this->sensor().project(pos, &uv)) {
          ++num_projected;
          const float depth = depth_frame(uv.y(), uv.x());
          if constexpr (std::is_same<typename TypeParam::SensorType,
                                     Camera>::value) {
            EXPECT_NEAR(depth, kDepth, 1E-5);
          }

          if (this->sensor().getDepth(pos) > depth - truncation_distance_m) {
            EXPECT_NEAR(voxel->distance, kReference, kEps);
          } else {
            EXPECT_LT(voxel->distance, kReference + kEps);
          }
        }
        return;
      });

  // Sanity check that we actually checked some voxels
  EXPECT_GT(static_cast<float>(num_projected) / num_voxels, 0.5F);
}

TYPED_TEST(TsdfIntegratorTestFixture, InvalidDepthHandling) {
  // Test that invalid depth values are not integrated and
  // can be decayed by invalid_depth_decay_factor.

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

  auto count_voxels_with_weight = [](const TsdfLayer& layer) {
    int count = 0;
    callFunctionOnAllVoxels<TsdfVoxel>(
        layer, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
          if (voxel->weight > 0.0f) ++count;
        });
    return count;
  };

  auto get_total_weight = [](const TsdfLayer& layer) {
    float total_weight = 0.0f;
    callFunctionOnAllVoxels<TsdfVoxel>(
        layer, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
          total_weight += voxel->weight;
        });
    return total_weight;
  };

  std::unique_ptr<ProjectiveTsdfIntegrator> integrator_ptr =
      this->create_integrator();

  // Use constant weighting for predictable weights
  integrator_ptr->weighting_function_type(
      WeightingFunctionType::kConstantWeight);
  constexpr float kConstantWeight = 1.0f;

  // Enable invalid depth decay (default is -1.0 which disables it)
  constexpr float kDecayFactor = 0.8f;
  integrator_ptr->invalid_depth_decay_factor(kDecayFactor);

  // Test 1: All pixels invalid - no voxels should have weight
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

    integrator_ptr->integrateFrame(
        MaskedDepthImageConstView(depth_image, kMaskActiveEverywhere),
        Transform::Identity(), this->sensor(), &this->layer_, nullptr);

    const int voxels_with_weight = count_voxels_with_weight(this->layer_);
    const float total_weight = get_total_weight(this->layer_);
    EXPECT_EQ(voxels_with_weight, 0)
        << "No voxels should have weight for invalid depth value: "
        << invalid_value;
    EXPECT_FLOAT_EQ(total_weight, 0.0f)
        << "Total weight should be exactly 0 for invalid depth value: "
        << invalid_value;
  }

  // Test 2: Valid depth - voxels should gain exact constant weight
  set_all_depth(2.0f);

  integrator_ptr->integrateFrame(
      MaskedDepthImageConstView(depth_image, kMaskActiveEverywhere),
      Transform::Identity(), this->sensor(), &this->layer_, nullptr);

  // Each integrated voxel should have exactly the constant weight
  const int voxels_with_weight_valid = count_voxels_with_weight(this->layer_);
  const float total_weight_after_valid = get_total_weight(this->layer_);
  const float expected_weight_per_voxel = kConstantWeight;
  const float expected_total_weight =
      voxels_with_weight_valid * expected_weight_per_voxel;
  EXPECT_GT(voxels_with_weight_valid, 0)
      << "Some voxels should have weight after integrating valid depth";
  EXPECT_NEAR(total_weight_after_valid, expected_total_weight, 1e-3f)
      << "With constant weighting, total weight should equal "
      << "num_voxels * constant_weight";

  // Test 3: Invalid depth again - weight should decay by exact factor
  // Test that decay works consistently across different invalid values
  set_all_depth(std::numeric_limits<float>::infinity());

  integrator_ptr->integrateFrame(
      MaskedDepthImageConstView(depth_image, kMaskActiveEverywhere),
      Transform::Identity(), this->sensor(), &this->layer_, nullptr);

  const float total_weight_after_decay = get_total_weight(this->layer_);
  const float expected_weight_after_decay =
      total_weight_after_valid * kDecayFactor;
  EXPECT_NEAR(total_weight_after_decay, expected_weight_after_decay, 1e-3f)
      << "After integrating invalid depth (infinity), total weight should be "
      << "multiplied by decay factor (" << kDecayFactor << ")";

  // Test 4: Apply decay multiple times with different invalid values
  set_all_depth(-1.0f);  // negative value
  integrator_ptr->integrateFrame(
      MaskedDepthImageConstView(depth_image, kMaskActiveEverywhere),
      Transform::Identity(), this->sensor(), &this->layer_, nullptr);

  const float total_weight_after_second_decay = get_total_weight(this->layer_);
  const float expected_weight_after_second_decay =
      expected_weight_after_decay * kDecayFactor;
  EXPECT_NEAR(total_weight_after_second_decay,
              expected_weight_after_second_decay, 1e-3f)
      << "Decay should compound with different invalid values: "
      << "weight after 2nd decay = initial * decay^2";

  // Test 5: Apply decay with yet another invalid value (0)
  set_all_depth(0.0f);
  integrator_ptr->integrateFrame(
      MaskedDepthImageConstView(depth_image, kMaskActiveEverywhere),
      Transform::Identity(), this->sensor(), &this->layer_, nullptr);

  const float total_weight_after_third_decay = get_total_weight(this->layer_);
  const float expected_weight_after_third_decay =
      expected_weight_after_second_decay * kDecayFactor;
  EXPECT_NEAR(total_weight_after_third_decay, expected_weight_after_third_decay,
              1e-3f)
      << "Decay should compound consistently: weight after 3rd decay = initial "
         "* decay^3";
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
