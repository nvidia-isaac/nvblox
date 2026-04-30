/*
Copyright 2022-2026 NVIDIA CORPORATION

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

#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/integrators/projective_appearance_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/interpolation/interpolation_3d.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/tests/gpu_image_routines.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/sensor_fixture.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

// -----------------------------------------------------------------------------
// Test Fixture
// -----------------------------------------------------------------------------

template <typename SensorType>
class AppearanceWithDepthTestFixture
    : public test_utils::SensorFixture<SensorType> {
 protected:
  AppearanceWithDepthTestFixture()
      : tsdf_layer_(voxel_size_m_, MemoryType::kUnified),
        color_layer_(voxel_size_m_, MemoryType::kUnified) {
    // Setup sphere-in-box scene
    scene_ = test_utils::getSphereInBox();
  }

  // Helper to generate solid color image
  ColorImage generateSolidColorImage(const Color& color) {
    ColorImage image(this->sensor().height(), this->sensor().width());
    test_utils::setImageConstantOnGpu(color, &image);
    return image;
  }

  // Helper to count voxels with weight
  template <typename VoxelType>
  int countVoxelsWithWeight(const VoxelBlockLayer<VoxelType>& layer) {
    int count = 0;
    callFunctionOnAllVoxels<VoxelType>(
        layer, [&](const Index3D&, const Index3D&, const VoxelType* voxel) {
          if (getVoxelWeight(*voxel) > 0.0f) {
            ++count;
          }
        });
    return count;
  }

  // Helper to get voxel weight (works for both TsdfVoxel and ColorVoxel)
  float getVoxelWeight(const TsdfVoxel& voxel) { return voxel.weight; }
  float getVoxelWeight(const ColorVoxel& voxel) {
    return __half2float(voxel.weight);
  }

  // Test parameters
  constexpr static float voxel_size_m_ = 0.1f;
  constexpr static float block_size_m_ =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * voxel_size_m_;
  constexpr static float truncation_distance_vox_ = 4.0f;
  constexpr static float truncation_distance_m_ =
      truncation_distance_vox_ * voxel_size_m_;

  // Trajectory parameters
  constexpr static float kTrajectoryRadius = 4.0f;
  constexpr static float kTrajectoryHeight = 2.0f;
  constexpr static int kNumTrajectoryPoints = 40;
  constexpr static float kMaxDist = 10.0f;

  // Layers
  TsdfLayer tsdf_layer_;
  ColorLayer color_layer_;

  // Scene
  primitives::Scene scene_;
};

// Only test with Camera since appearance with depth requires pinhole projection
using AppearanceWithDepthTestTypes = ::testing::Types<Camera>;
TYPED_TEST_SUITE(AppearanceWithDepthTestFixture, AppearanceWithDepthTestTypes);

// -----------------------------------------------------------------------------
// Test: GettersAndSetters
// -----------------------------------------------------------------------------

TEST(AppearanceWithDepthParametersTest, ColorIntegratorMeasurementWeight) {
  ProjectiveColorIntegrator color_integrator;

  // Test measurement_weight getter/setter (used when integrating color with
  // depth for appearance integration with depth)
  const float default_weight = color_integrator.measurement_weight();
  EXPECT_GT(default_weight, 0.0f);
  EXPECT_LE(default_weight, 1.0f);

  color_integrator.measurement_weight(0.5f);
  EXPECT_FLOAT_EQ(color_integrator.measurement_weight(), 0.5f);

  color_integrator.measurement_weight(0.1f);
  EXPECT_FLOAT_EQ(color_integrator.measurement_weight(), 0.1f);

  color_integrator.measurement_weight(1.0f);
  EXPECT_FLOAT_EQ(color_integrator.measurement_weight(), 1.0f);

  color_integrator.measurement_weight(0.01f);
  EXPECT_FLOAT_EQ(color_integrator.measurement_weight(), 0.01f);
}

// -----------------------------------------------------------------------------
// Test: BasicIntegration
// -----------------------------------------------------------------------------

TYPED_TEST(AppearanceWithDepthTestFixture, BasicIntegration) {
  ProjectiveTsdfIntegrator tsdf_integrator;
  tsdf_integrator.truncation_distance_vox(this->truncation_distance_vox_);
  tsdf_integrator.max_integration_distance_m(this->kMaxDist);

  ProjectiveColorIntegrator color_integrator;
  color_integrator.truncation_distance_vox(this->truncation_distance_vox_);

  const Color test_color = Color::Red();
  const ColorImage color_image = this->generateSolidColorImage(test_color);

  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);

  const float radians_increment = 2 * M_PI / this->kNumTrajectoryPoints;

  std::vector<Index3D> all_updated_blocks;

  for (int i = 0; i < this->kNumTrajectoryPoints; i++) {
    const float theta = radians_increment * i;

    Vector3f position(this->kTrajectoryRadius * std::cos(theta),
                      this->kTrajectoryRadius * std::sin(theta),
                      this->kTrajectoryHeight);

    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(position);

    this->scene_.generateDepthImageFromScene(this->sensor(), T_S_C,
                                             this->kMaxDist, &depth_frame);

    std::vector<Index3D> updated_blocks;
    tsdf_integrator.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), &this->tsdf_layer_, &updated_blocks);

    color_integrator.integrateFrame(
        MaskedColorImageConstView(color_image, kMaskActiveEverywhere),
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), this->tsdf_layer_, &this->color_layer_);

    std::copy(updated_blocks.begin(), updated_blocks.end(),
              std::back_inserter(all_updated_blocks));
  }

  // Validation: Both layers should have blocks
  EXPECT_GT(this->tsdf_layer_.numBlocks(), 0);
  EXPECT_GT(this->color_layer_.numBlocks(), 0);
  EXPECT_GT(all_updated_blocks.size(), 0);

  LOG(INFO) << "TSDF layer blocks: " << this->tsdf_layer_.numBlocks();
  LOG(INFO) << "Color layer blocks: " << this->color_layer_.numBlocks();

  // Validation: TSDF voxels should have weight
  const int num_tsdf_voxels = this->countVoxelsWithWeight(this->tsdf_layer_);
  EXPECT_GT(num_tsdf_voxels, 0);
  LOG(INFO) << "TSDF voxels with weight: " << num_tsdf_voxels;

  // Validation: Color voxels should have weight and correct color
  int num_color_voxels = 0;
  int num_correct_color = 0;
  callFunctionOnAllVoxels<ColorVoxel>(
      this->color_layer_,
      [&](const Index3D&, const Index3D&, const ColorVoxel* voxel) {
        if (__half2float(voxel->weight) > 0.0f) {
          ++num_color_voxels;
          if (voxel->color == test_color) {
            ++num_correct_color;
          }
        }
      });
  EXPECT_GT(num_color_voxels, 0);
  EXPECT_EQ(num_color_voxels, num_correct_color);
  LOG(INFO) << "Color voxels with weight: " << num_color_voxels;

  // Validation: All color blocks should have corresponding TSDF blocks
  for (const Index3D& block_idx : this->color_layer_.getAllBlockIndices()) {
    EXPECT_NE(this->tsdf_layer_.getBlockAtIndex(block_idx), nullptr);
  }

  // Optional: Output mesh for visualization
  if (FLAGS_nvblox_test_file_output) {
    ColorMeshIntegrator mesh_integrator;
    ColorMeshLayer mesh_layer(this->block_size_m_, MemoryType::kDevice);
    EXPECT_TRUE(mesh_integrator.integrateMeshFromDistanceField(
        this->tsdf_layer_, &mesh_layer));
    mesh_integrator.updateAppearance(this->color_layer_, &mesh_layer);
    io::outputColorMeshLayerToPly(
        mesh_layer, "appearance_with_depth_basic_integration.ply");
  }
}

// -----------------------------------------------------------------------------
// Test: TsdfQualityMatchesSeparateIntegration
// -----------------------------------------------------------------------------

TYPED_TEST(AppearanceWithDepthTestFixture,
           TsdfQualityMatchesSeparateIntegration) {
  // Create two separate layer pairs
  TsdfLayer tsdf_with_color(this->voxel_size_m_, MemoryType::kUnified);
  ColorLayer color_with_depth(this->voxel_size_m_, MemoryType::kUnified);
  TsdfLayer tsdf_depth_only(this->voxel_size_m_, MemoryType::kUnified);

  ProjectiveTsdfIntegrator tsdf_integrator;
  tsdf_integrator.truncation_distance_vox(this->truncation_distance_vox_);
  tsdf_integrator.max_integration_distance_m(this->kMaxDist);

  ProjectiveColorIntegrator color_integrator;
  color_integrator.truncation_distance_vox(this->truncation_distance_vox_);

  const ColorImage color_image = this->generateSolidColorImage(Color::Red());
  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);

  constexpr int kNumFrames = 20;
  const float radians_increment = 2 * M_PI / kNumFrames;

  for (int i = 0; i < kNumFrames; i++) {
    const float theta = radians_increment * i;

    Vector3f position(this->kTrajectoryRadius * std::cos(theta),
                      this->kTrajectoryRadius * std::sin(theta),
                      this->kTrajectoryHeight);

    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(position);

    this->scene_.generateDepthImageFromScene(this->sensor(), T_S_C,
                                             this->kMaxDist, &depth_frame);

    // Two-pass: TSDF then color with depth
    tsdf_integrator.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), &tsdf_with_color, nullptr);
    color_integrator.integrateFrame(
        MaskedColorImageConstView(color_image, kMaskActiveEverywhere),
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), tsdf_with_color, &color_with_depth);

    // Depth-only (TSDF only) for comparison
    tsdf_integrator.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), &tsdf_depth_only, nullptr);
  }

  // Both TSDF layers should have the same number of blocks
  EXPECT_EQ(tsdf_with_color.numBlocks(), tsdf_depth_only.numBlocks());

  // Compare voxel values
  int num_compared = 0;
  int num_matching = 0;
  constexpr float kDistanceThreshold = 1e-4f;
  constexpr float kWeightThreshold = 1e-4f;

  for (const Index3D& block_idx : tsdf_with_color.getAllBlockIndices()) {
    const auto block_with_color = tsdf_with_color.getBlockAtIndex(block_idx);
    const auto block_depth = tsdf_depth_only.getBlockAtIndex(block_idx);

    ASSERT_NE(block_depth, nullptr)
        << "Block missing in depth-only layer at " << block_idx.transpose();

    constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
    for (int x = 0; x < kVoxelsPerSide; x++) {
      for (int y = 0; y < kVoxelsPerSide; y++) {
        for (int z = 0; z < kVoxelsPerSide; z++) {
          const TsdfVoxel& voxel_with_color = block_with_color->voxels[x][y][z];
          const TsdfVoxel& voxel_depth = block_depth->voxels[x][y][z];

          if (voxel_with_color.weight > 0.0f || voxel_depth.weight > 0.0f) {
            ++num_compared;
            const float dist_diff =
                std::abs(voxel_with_color.distance - voxel_depth.distance);
            const float weight_diff =
                std::abs(voxel_with_color.weight - voxel_depth.weight);

            if (dist_diff < kDistanceThreshold &&
                weight_diff < kWeightThreshold) {
              ++num_matching;
            }
          }
        }
      }
    }
  }

  const float match_ratio =
      static_cast<float>(num_matching) / static_cast<float>(num_compared);
  LOG(INFO) << "TSDF match ratio: " << match_ratio << " (" << num_matching
            << "/" << num_compared << ")";

  // Expect >99% match
  // Note: Small differences may occur due to floating-point order of operations
  // between the two code paths, but TSDF values should be nearly identical.
  EXPECT_GT(match_ratio, 0.99f);
}

// -----------------------------------------------------------------------------
// Test: OcclusionHandling
// -----------------------------------------------------------------------------

TYPED_TEST(AppearanceWithDepthTestFixture, OcclusionHandling) {
  // Create scene with two spheres at different depths
  primitives::Scene occlusion_scene;

  constexpr float kSphereRadius = 1.5f;
  const Vector3f front_center(4.0f, 0.0f, 0.0f);  // Closer to camera
  const Vector3f back_center(8.0f, 0.0f, 0.0f);   // Behind front sphere

  occlusion_scene.aabb() = AxisAlignedBoundingBox(Vector3f(-2.0f, -5.0f, -5.0f),
                                                  Vector3f(12.0f, 5.0f, 5.0f));
  occlusion_scene.addPrimitive(
      std::make_unique<primitives::Sphere>(front_center, kSphereRadius));
  occlusion_scene.addPrimitive(
      std::make_unique<primitives::Sphere>(back_center, kSphereRadius));

  // Create layers
  TsdfLayer tsdf_layer(this->voxel_size_m_, MemoryType::kUnified);
  ColorLayer color_layer(this->voxel_size_m_, MemoryType::kUnified);

  ProjectiveTsdfIntegrator tsdf_integrator;
  tsdf_integrator.truncation_distance_vox(this->truncation_distance_vox_);

  ProjectiveColorIntegrator color_integrator;
  color_integrator.truncation_distance_vox(this->truncation_distance_vox_);

  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(
      Eigen::Quaternionf(Eigen::AngleAxisf(M_PI / 2, Vector3f::UnitY())));

  constexpr float kOcclusionMaxDist = 15.0f;
  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);
  occlusion_scene.generateDepthImageFromScene(this->sensor(), T_S_C,
                                              kOcclusionMaxDist, &depth_frame);

  const Color test_color = Color::Red();
  const ColorImage color_image = this->generateSolidColorImage(test_color);

  std::vector<Index3D> updated_blocks;
  tsdf_integrator.integrateFrame(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
      this->sensor(), &tsdf_layer, &updated_blocks);
  color_integrator.integrateFrame(
      MaskedColorImageConstView(color_image, kMaskActiveEverywhere),
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
      this->sensor(), tsdf_layer, &color_layer);

  // Helper to get points on sphere surface
  auto getPointsOnSphere = [](const Vector3f& center, float radius,
                              int points_per_axis = 10) {
    std::vector<Vector3f> points;
    for (int az = 0; az < 2 * points_per_axis; az++) {
      for (int el = 0; el < points_per_axis; el++) {
        const float azimuth = az * M_PI / points_per_axis - M_PI;
        const float elevation = el * M_PI / points_per_axis - M_PI / 2.0f;
        Vector3f p =
            radius * Vector3f(cos(azimuth) * sin(elevation),
                              sin(azimuth) * sin(elevation), cos(elevation));
        points.push_back(center + p);
      }
    }
    return points;
  };

  // Check front sphere - should have colored voxels
  int front_colored = 0;
  int front_total = 0;
  const auto front_points = getPointsOnSphere(front_center, kSphereRadius);
  for (const Vector3f& p : front_points) {
    const ColorVoxel* voxel;
    if (getVoxelAtPosition<ColorVoxel>(color_layer, p, &voxel)) {
      ++front_total;
      if (__half2float(voxel->weight) > 0.0f) {
        ++front_colored;
      }
    }
  }
  LOG(INFO) << "Front sphere: " << front_colored << "/" << front_total
            << " voxels colored";
  EXPECT_GT(front_colored, 0) << "Front sphere should have colored voxels";

  // Check back sphere - should NOT have colored voxels (occluded)
  int back_colored = 0;
  int back_total = 0;
  const auto back_points = getPointsOnSphere(back_center, kSphereRadius);
  for (const Vector3f& p : back_points) {
    const ColorVoxel* voxel;
    if (getVoxelAtPosition<ColorVoxel>(color_layer, p, &voxel)) {
      ++back_total;
      if (__half2float(voxel->weight) > 0.0f) {
        ++back_colored;
      }
    }
  }
  LOG(INFO) << "Back sphere: " << back_colored << "/" << back_total
            << " voxels colored (should be ~0)";

  // Back sphere should have significantly fewer colored voxels
  // (ideally 0, but allow small margin for voxels at edges)
  const float back_ratio =
      (front_colored > 0)
          ? static_cast<float>(back_colored) / static_cast<float>(front_colored)
          : 0.0f;
  EXPECT_LT(back_ratio, 0.1f)
      << "Back sphere should have <10% colored voxels compared to front";

  // Optional: Output mesh for visualization
  if (FLAGS_nvblox_test_file_output) {
    ColorMeshIntegrator mesh_integrator;
    ColorMeshLayer mesh_layer(this->block_size_m_, MemoryType::kDevice);
    EXPECT_TRUE(mesh_integrator.integrateMeshFromDistanceField(tsdf_layer,
                                                               &mesh_layer));
    mesh_integrator.updateAppearance(color_layer, &mesh_layer);
    io::outputColorMeshLayerToPly(mesh_layer,
                                  "appearance_with_depth_occlusion_test.ply");
  }
}

// -----------------------------------------------------------------------------
// Test: ColorExponentialFilter
// -----------------------------------------------------------------------------

TYPED_TEST(AppearanceWithDepthTestFixture, ColorExponentialFilter) {
  // Simple plane scene
  primitives::Scene plane_scene;
  plane_scene.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, 0.0f),
                                              Vector3f(5.0f, 5.0f, 5.0f));
  plane_scene.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(0.0f, 0.0f, 2.0f), Vector3f(0.0f, 0.0f, -1.0f)));

  // Create layers
  TsdfLayer tsdf_layer(this->voxel_size_m_, MemoryType::kUnified);
  ColorLayer color_layer(this->voxel_size_m_, MemoryType::kUnified);

  ProjectiveTsdfIntegrator tsdf_integrator;
  tsdf_integrator.truncation_distance_vox(this->truncation_distance_vox_);

  ProjectiveColorIntegrator color_integrator;
  color_integrator.truncation_distance_vox(this->truncation_distance_vox_);
  color_integrator.measurement_weight(0.5f);

  Transform T_S_C = Transform::Identity();

  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);
  plane_scene.generateDepthImageFromScene(this->sensor(), T_S_C, 5.0f,
                                          &depth_frame);

  const ColorImage red_image = this->generateSolidColorImage(Color::Red());
  tsdf_integrator.integrateFrame(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
      this->sensor(), &tsdf_layer, nullptr);
  color_integrator.integrateFrame(
      MaskedColorImageConstView(red_image, kMaskActiveEverywhere),
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
      this->sensor(), tsdf_layer, &color_layer);

  // Verify first integration produces red voxels
  int num_red_voxels = 0;
  callFunctionOnAllVoxels<ColorVoxel>(
      color_layer,
      [&](const Index3D&, const Index3D&, const ColorVoxel* voxel) {
        if (__half2float(voxel->weight) > 0.0f &&
            voxel->color == Color::Red()) {
          ++num_red_voxels;
        }
      });
  EXPECT_GT(num_red_voxels, 0) << "First integration should produce red voxels";
  LOG(INFO) << "After first integration: " << num_red_voxels << " red voxels";

  const ColorImage blue_image = this->generateSolidColorImage(Color::Blue());
  tsdf_integrator.integrateFrame(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
      this->sensor(), &tsdf_layer, nullptr);
  color_integrator.integrateFrame(
      MaskedColorImageConstView(blue_image, kMaskActiveEverywhere),
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
      this->sensor(), tsdf_layer, &color_layer);

  // Verify second integration produces blended colors
  int num_blended = 0;
  int num_with_weight = 0;
  callFunctionOnAllVoxels<ColorVoxel>(
      color_layer,
      [&](const Index3D&, const Index3D&, const ColorVoxel* voxel) {
        if (__half2float(voxel->weight) > 0.0f) {
          ++num_with_weight;
          // Blended color should have both red and blue components
          // Red (255,0,0) + Blue (0,0,255) with 50% weight = ~(127,0,127)
          const bool has_red = voxel->color.r() > 0;
          const bool has_blue = voxel->color.b() > 0;
          const bool not_pure_red = voxel->color.r() < 255;
          const bool not_pure_blue = voxel->color.b() < 255;

          if (has_red && has_blue && not_pure_red && not_pure_blue) {
            ++num_blended;
          }
        }
      });

  LOG(INFO) << "After second integration: " << num_blended << "/"
            << num_with_weight << " blended voxels";

  // Most voxels should show blending
  const float blend_ratio = (num_with_weight > 0)
                                ? static_cast<float>(num_blended) /
                                      static_cast<float>(num_with_weight)
                                : 0.0f;
  EXPECT_GT(blend_ratio, 0.9f)
      << "Most voxels should show color blending after two frames";
}

// -----------------------------------------------------------------------------
// Test: InvalidDepthHandling
// -----------------------------------------------------------------------------

TYPED_TEST(AppearanceWithDepthTestFixture, InvalidDepthHandling) {
  TsdfLayer tsdf_layer(this->voxel_size_m_, MemoryType::kUnified);
  ColorLayer color_layer(this->voxel_size_m_, MemoryType::kUnified);

  ProjectiveTsdfIntegrator tsdf_integrator;
  tsdf_integrator.truncation_distance_vox(this->truncation_distance_vox_);

  ProjectiveColorIntegrator color_integrator;
  color_integrator.truncation_distance_vox(this->truncation_distance_vox_);

  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);

  const ColorImage color_image = this->generateSolidColorImage(Color::Red());

  Transform T_S_C = Transform::Identity();

  const std::vector<float> invalid_values = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -1.0f,
      0.0f,
      -10.0f};

  for (const float invalid_value : invalid_values) {
    tsdf_layer.clear();
    color_layer.clear();

    for (int i = 0; i < depth_frame.numel(); i++) {
      depth_frame(i) = invalid_value;
    }

    tsdf_integrator.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), &tsdf_layer, nullptr);
    color_integrator.integrateFrame(
        MaskedColorImageConstView(color_image, kMaskActiveEverywhere),
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
        this->sensor(), tsdf_layer, &color_layer);

    // Verify no voxels were integrated
    const int tsdf_voxels = this->countVoxelsWithWeight(tsdf_layer);
    const int color_voxels = this->countVoxelsWithWeight(color_layer);

    EXPECT_EQ(tsdf_voxels, 0) << "No TSDF voxels should be integrated for "
                                 "invalid depth value: "
                              << invalid_value;
    EXPECT_EQ(color_voxels, 0) << "No color voxels should be integrated for "
                                  "invalid depth value: "
                               << invalid_value;
  }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
