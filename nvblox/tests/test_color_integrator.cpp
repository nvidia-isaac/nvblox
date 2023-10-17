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

#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/integrators/projective_color_integrator.h"
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
#include "nvblox/tests/utils.h"

using namespace nvblox;

class ColorIntegrationTest : public ::testing::Test {
 protected:
  ColorIntegrationTest()
      : kSphereCenter(Vector3f(0.0f, 0.0f, 2.0f)),
        gt_layer_(voxel_size_m_, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {
    // Scene is bounded to -5, -5, 0 to 5, 5, 5.
    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, 0.0f),
                                           Vector3f(5.0f, 5.0f, 5.0f));
    // Create a scene with a ground plane and a sphere.
    scene_.addGroundLevel(0.0f);
    scene_.addCeiling(5.0f);
    scene_.addPrimitive(
        std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 2.0f), 2.0f));
    // Add bounding planes at 5 meters. Basically makes it sphere in a box.
    scene_.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);

    // Get the ground truth SDF for it.
    scene_.generateLayerFromScene(truncation_distance_m_, &gt_layer_);
  }

  // Scenes
  constexpr static float kSphereRadius = 2.0f;
  const Vector3f kSphereCenter;

  // Test layer
  constexpr static float voxel_size_m_ = 0.1;
  constexpr static float block_size_m_ =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * voxel_size_m_;
  TsdfLayer gt_layer_;

  // Truncation distance
  constexpr static float truncation_distance_vox_ = 4;
  constexpr static float truncation_distance_m_ =
      truncation_distance_vox_ * voxel_size_m_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;

  // Test Scene
  primitives::Scene scene_;
};

// ProjectiveTsdfIntegrator child that gives the tests access to the internal
// functions.
class TestProjectiveColorIntegratorGPU : public ProjectiveColorIntegrator {
 public:
  TestProjectiveColorIntegratorGPU() : ProjectiveColorIntegrator() {}
  FRIEND_TEST(ColorIntegrationTest, TruncationBandTest);
};

bool colorsEqualIgnoreAlpha(const Color& color_1, const Color& color_2) {
  return (color_1.r == color_2.r) && (color_1.g == color_2.g) &&
         (color_1.b == color_2.b);
}

ColorImage generateSolidColorImage(const Color& color, const int height,
                                   const int width) {
  // Generate a random color for this scene
  ColorImage image(height, width);
  nvblox::test_utils::setImageConstantOnGpu(color, &image);
  return image;
}

std::vector<Eigen::Vector3f> getPointsOnASphere(const float radius,
                                                const Eigen::Vector3f& center,
                                                const int points_per_rad = 10) {
  std::vector<Eigen::Vector3f> sphere_points;
  for (int azimuth_idx = 0; azimuth_idx < 2 * points_per_rad; azimuth_idx++) {
    for (int elevation_idx = 0; elevation_idx < points_per_rad;
         elevation_idx++) {
      const float azimuth = azimuth_idx * M_PI / points_per_rad - M_PI;
      const float elevation =
          elevation_idx * M_PI / points_per_rad - M_PI / 2.0f;
      Eigen::Vector3f p =
          radius * Eigen::Vector3f(cos(azimuth) * sin(elevation),
                                   sin(azimuth) * sin(elevation),
                                   cos(elevation));
      p += center;
      sphere_points.push_back(p);
    }
  }
  return sphere_points;
}

float checkSphereColor(const ColorLayer& color_layer, const Vector3f& center,
                       const float radius, const Color& color) {
  // Check that each sphere is colored appropriately (if observed)
  int num_observed = 0;
  int num_tested = 0;
  auto check_color = [&num_tested, &num_observed](
                         const ColorVoxel& voxel,
                         const Color& color_2) -> void {
    ++num_tested;
    constexpr float kMinVoxelWeight = 1e-3;
    if (voxel.weight >= kMinVoxelWeight) {
      EXPECT_TRUE(colorsEqualIgnoreAlpha(voxel.color, color_2));
      ++num_observed;
    }
  };

  const std::vector<Eigen::Vector3f> sphere_points =
      getPointsOnASphere(radius, center);
  for (const Vector3f p : sphere_points) {
    const ColorVoxel* color_voxel;
    EXPECT_TRUE(getVoxelAtPosition<ColorVoxel>(color_layer, p, &color_voxel));
    check_color(*color_voxel, color);
  }

  const float ratio_observed_points =
      static_cast<float>(num_observed) / static_cast<float>(num_tested);
  return ratio_observed_points;
}

TEST_F(ColorIntegrationTest, GettersAndSetters) {
  ProjectiveColorIntegrator color_integrator;
  color_integrator.truncation_distance_vox(1);
  EXPECT_EQ(color_integrator.truncation_distance_vox(), 1);
  color_integrator.sphere_tracing_ray_subsampling_factor(2);
  EXPECT_EQ(color_integrator.sphere_tracing_ray_subsampling_factor(), 2);
  color_integrator.max_weight(3.0f);
  EXPECT_EQ(color_integrator.max_weight(), 3.0f);
  color_integrator.max_integration_distance_m(4.0f);
  EXPECT_EQ(color_integrator.max_integration_distance_m(), 4.0f);
  color_integrator.weighting_function_type(
      WeightingFunctionType::kInverseSquareWeight);
  EXPECT_EQ(color_integrator.weighting_function_type(),
            WeightingFunctionType::kInverseSquareWeight);
}

TEST_F(ColorIntegrationTest, TruncationBandTest) {
  // Check the GPU version against a hand-rolled CPU implementation.
  TestProjectiveColorIntegratorGPU integrator;

  // The distance from the surface that we "pass" blocks within.
  constexpr float kTestDistance = voxel_size_m_;

  std::vector<Index3D> all_indices = gt_layer_.getAllBlockIndices();
  std::vector<Index3D> valid_indices =
      integrator.reduceBlocksToThoseInTruncationBand(all_indices, gt_layer_,
                                                     kTestDistance);

  // Horrible N^2 complexity set_difference implementation. But easy to write :)
  std::vector<Index3D> not_valid_indices;
  for (const Index3D& idx : all_indices) {
    if (std::find(valid_indices.begin(), valid_indices.end(), idx) ==
        valid_indices.end()) {
      not_valid_indices.push_back(idx);
    }
  }

  // Check indices touching band
  for (const Index3D& idx : valid_indices) {
    const auto block_ptr = gt_layer_.getBlockAtIndex(idx);
    bool touches_band = false;
    auto touches_band_lambda = [&touches_band, kTestDistance](
                                   const Index3D&,
                                   const TsdfVoxel* voxel) -> void {
      if (std::abs(voxel->distance) <= kTestDistance) {
        touches_band = true;
      }
    };
    callFunctionOnAllVoxels<TsdfVoxel>(*block_ptr, touches_band_lambda);
    EXPECT_TRUE(touches_band);
  }

  // Check indices NOT touching band
  for (const Index3D& idx : not_valid_indices) {
    const auto block_ptr = gt_layer_.getBlockAtIndex(idx);
    bool touches_band = false;
    auto touches_band_lambda = [&touches_band, kTestDistance](
                                   const Index3D&,
                                   const TsdfVoxel* voxel) -> void {
      if (std::abs(voxel->distance) <= kTestDistance) {
        touches_band = true;
      }
    };
    callFunctionOnAllVoxels<TsdfVoxel>(*block_ptr, touches_band_lambda);
    EXPECT_FALSE(touches_band);
  }
}

TEST_F(ColorIntegrationTest, IntegrateColorToGroundTruthDistanceField) {
  // Create an integrator.
  ProjectiveColorIntegrator color_integrator;

  // Simulate a trajectory of the requisite amount of points, on the circle
  // around the sphere.
  constexpr float kTrajectoryRadius = 4.0f;
  constexpr float kTrajectoryHeight = 2.0f;
  constexpr int kNumTrajectoryPoints = 80;
  const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);

  // Color layer
  ColorLayer color_layer(voxel_size_m_, MemoryType::kDevice);

  // Generate a random color for this scene
  const Color color = Color::Red();
  const ColorImage image = generateSolidColorImage(color, height_, width_);

  // Set keeping track of which blocks were touched during the test
  Index3DSet touched_blocks;

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

    // Generate an image with a single color
    std::vector<Index3D> updated_blocks;
    color_integrator.integrateFrame(image, T_S_C, camera_, gt_layer_,
                                    &color_layer, &updated_blocks);
    // Accumulate touched block indices
    std::copy(updated_blocks.begin(), updated_blocks.end(),
              std::inserter(touched_blocks, touched_blocks.end()));
  }

  // Create a host copy of the layer.
  ColorLayer color_layer_host(voxel_size_m_, MemoryType::kHost);
  color_layer_host.copyFrom(color_layer);

  // Lambda that checks if voxels have the passed color (if they have weight >
  // 0)
  auto color_check_lambda = [&color](const Index3D&,
                                     const ColorVoxel* voxel) -> void {
    if (voxel->weight > 0.0f) {
      EXPECT_TRUE(colorsEqualIgnoreAlpha(voxel->color, color));
    }
  };

  // Check that all touched blocks are the color we chose
  for (const Index3D& block_idx : touched_blocks) {
    callFunctionOnAllVoxels<ColorVoxel>(
        *color_layer_host.getBlockAtIndex(block_idx), color_check_lambda);
  }

  // Check that most points on the surface of the sphere have been observed
  int num_points_on_sphere_surface_observed = 0;
  const std::vector<Eigen::Vector3f> sphere_points =
      getPointsOnASphere(kSphereRadius, kSphereCenter);
  const int num_surface_points_tested = sphere_points.size();
  for (const Vector3f p : sphere_points) {
    const ColorVoxel* color_voxel;
    EXPECT_TRUE(
        getVoxelAtPosition<ColorVoxel>(color_layer_host, p, &color_voxel));
    if (color_voxel->weight >= 1.0f) {
      ++num_points_on_sphere_surface_observed;
    }
  }
  const float ratio_observed_surface_points =
      static_cast<float>(num_points_on_sphere_surface_observed) /
      static_cast<float>(num_surface_points_tested);
  std::cout << "num_points_on_sphere_surface_observed: "
            << num_points_on_sphere_surface_observed << std::endl;
  std::cout << "num_surface_points_tested: " << num_surface_points_tested
            << std::endl;
  std::cout << "ratio_observed_surface_points: "
            << ratio_observed_surface_points << std::endl;
  EXPECT_GT(ratio_observed_surface_points, 0.5);

  // Check that all color blocks have a corresponding block in the tsdf layer
  for (const Index3D block_idx : color_layer_host.getAllBlockIndices()) {
    EXPECT_NE(gt_layer_.getBlockAtIndex(block_idx), nullptr);
  }

  // Generate a mesh from the "reconstruction"
  MeshIntegrator mesh_integrator;
  BlockLayer<MeshBlock> mesh_layer(block_size_m_, MemoryType::kDevice);
  EXPECT_TRUE(
      mesh_integrator.integrateMeshFromDistanceField(gt_layer_, &mesh_layer));
  mesh_integrator.colorMesh(color_layer, &mesh_layer);

  // Write to file
  if (FLAGS_nvblox_test_file_output) {
    io::outputMeshLayerToPly(mesh_layer, "color_sphere_mesh.ply");
  }
}

TEST_F(ColorIntegrationTest, ColoredSpheres) {
  constexpr float kTruncationDistanceVox = 2;
  constexpr float truncation_distance_m =
      kTruncationDistanceVox * voxel_size_m_;

  primitives::Scene scene_2;

  constexpr float kSphereRadiusTest = 2.0f;
  const Eigen::Vector3f center_1 = Vector3f(5.0f, 0.0f, 0.0f);
  const Eigen::Vector3f center_2 = Vector3f(5.0f, 5.0f, 0.0f);
  const Eigen::Vector3f center_3 = Vector3f(5.0f, 10.0f, 0.0f);

  // Scene is bounded to -5, -5, 0 to 5, 5, 5.
  scene_2.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, -5.0f),
                                          Vector3f(10.0f, 15.0f, 5.0f));
  // Create a scene with a ground plane and a sphere.
  scene_2.addPrimitive(
      std::make_unique<primitives::Sphere>(center_1, kSphereRadiusTest));
  scene_2.addPrimitive(
      std::make_unique<primitives::Sphere>(center_2, kSphereRadiusTest));
  scene_2.addPrimitive(
      std::make_unique<primitives::Sphere>(center_3, kSphereRadiusTest));

  // Camera
  // Slightly narrow field of view so each image below sees only a single
  // sphere.
  constexpr float fu = 450;
  constexpr float fv = 450;
  constexpr int width = 640;
  constexpr int height = 480;
  constexpr float cu = static_cast<float>(width_) / 2.0f;
  constexpr float cv = static_cast<float>(height_) / 2.0f;
  Camera camera(fu, fv, cu, cv, width, height);

  // Simulate camera views
  // Rotate 90 degress around the y axis to point camera-Z along world-X
  Transform T_S_C1 = Transform::Identity();
  T_S_C1.prerotate(
      Eigen::Quaternionf(Eigen::AngleAxisf(M_PI / 2, Vector3f::UnitY())));
  T_S_C1.pretranslate(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
  Transform T_S_C2 = Transform::Identity();
  T_S_C2.prerotate(
      Eigen::Quaternionf(Eigen::AngleAxisf(M_PI / 2, Vector3f::UnitY())));
  T_S_C2.pretranslate(Eigen::Vector3f(0.0f, 5.0f, 0.0f));
  Transform T_S_C3 = Transform::Identity();
  T_S_C3.prerotate(
      Eigen::Quaternionf(Eigen::AngleAxisf(M_PI / 2, Vector3f::UnitY())));
  T_S_C3.pretranslate(Eigen::Vector3f(0.0f, 10.0f, 0.0f));

  // TSDF
  scene_2.generateLayerFromScene(truncation_distance_m, &gt_layer_);

  // Color layer
  ProjectiveColorIntegrator color_integrator;
  ColorLayer color_layer(voxel_size_m_, MemoryType::kDevice);

  const auto color_1 = Color::Red();
  const auto color_2 = Color::Green();
  const auto color_3 = Color::Blue();

  const auto image_1 = generateSolidColorImage(color_1, height_, width_);
  const auto image_2 = generateSolidColorImage(color_2, height_, width_);
  const auto image_3 = generateSolidColorImage(color_3, height_, width_);

  std::vector<Index3D> updated_blocks;
  color_integrator.integrateFrame(image_1, T_S_C1, camera, gt_layer_,
                                  &color_layer, &updated_blocks);
  color_integrator.integrateFrame(image_2, T_S_C2, camera, gt_layer_,
                                  &color_layer, &updated_blocks);
  color_integrator.integrateFrame(image_3, T_S_C3, camera, gt_layer_,
                                  &color_layer, &updated_blocks);

  // Generate a mesh from the "reconstruction"
  MeshIntegrator mesh_integrator;
  BlockLayer<MeshBlock> mesh_layer(block_size_m_, MemoryType::kDevice);
  EXPECT_TRUE(
      mesh_integrator.integrateMeshFromDistanceField(gt_layer_, &mesh_layer));
  mesh_integrator.colorMesh(color_layer, &mesh_layer);

  ColorLayer color_layer_host(block_size_m_, MemoryType::kHost);
  color_layer_host.copyFrom(color_layer);

  const float sphere_1_observed_ratio =
      checkSphereColor(color_layer_host, center_1, kSphereRadius, color_1);
  const float sphere_2_observed_ratio =
      checkSphereColor(color_layer_host, center_2, kSphereRadius, color_2);
  const float sphere_3_observed_ratio =
      checkSphereColor(color_layer_host, center_3, kSphereRadius, color_3);

  EXPECT_GT(sphere_1_observed_ratio, 0.2);
  EXPECT_GT(sphere_2_observed_ratio, 0.2);
  EXPECT_GT(sphere_3_observed_ratio, 0.2);

  std::cout << "sphere_1_observed_ratio: " << sphere_1_observed_ratio
            << std::endl;
  std::cout << "sphere_2_observed_ratio: " << sphere_2_observed_ratio
            << std::endl;
  std::cout << "sphere_3_observed_ratio: " << sphere_3_observed_ratio
            << std::endl;

  // Write to file
  if (FLAGS_nvblox_test_file_output) {
    io::outputMeshLayerToPly(mesh_layer, "colored_spheres.ply");
  }
}

TEST_F(ColorIntegrationTest, OcclusionTesting) {
  primitives::Scene scene_3;

  // Scene: Two spheres, one occluded by the other.
  scene_3.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, -5.0f),
                                          Vector3f(15.0f, 15.0f, 5.0f));
  constexpr float kSphereRadiusTest = 2.0f;

  const Vector3f center_1(5.0f, 0.0f, 0.0f);
  const Vector3f center_2(10.0f, 0.0f, 0.0f);

  scene_3.addPrimitive(
      std::make_unique<primitives::Sphere>(center_1, kSphereRadiusTest));
  scene_3.addPrimitive(
      std::make_unique<primitives::Sphere>(center_2, kSphereRadiusTest));
  scene_3.generateLayerFromScene(truncation_distance_m_, &gt_layer_);

  // Viewpoint
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(
      Eigen::Quaternionf(Eigen::AngleAxisf(M_PI / 2, Vector3f::UnitY())));

  // Integrate a color image
  ProjectiveColorIntegrator color_integrator;
  ColorLayer color_layer(voxel_size_m_, MemoryType::kDevice);

  const auto color_1 = Color::Red();
  const auto image_1 = generateSolidColorImage(color_1, height_, width_);

  std::vector<Index3D> updated_blocks;
  color_integrator.integrateFrame(image_1, T_S_C, camera_, gt_layer_,
                                  &color_layer, &updated_blocks);

  ColorLayer color_layer_host(voxel_size_m_, MemoryType::kHost);
  color_layer_host.copyFrom(color_layer);

  // Check front sphere (observed voxels red)
  const float sphere_1_observed_ratio =
      checkSphereColor(color_layer_host, center_1, kSphereRadius, color_1);
  EXPECT_GT(sphere_1_observed_ratio, 0.2);
  std::cout << "sphere_1_observed_ratio: " << sphere_1_observed_ratio
            << std::endl;

  // Check back sphere (no observed voxels)
  const std::vector<Eigen::Vector3f> sphere_points =
      getPointsOnASphere(kSphereRadius, center_2);
  for (const Vector3f p : sphere_points) {
    const ColorVoxel* color_voxel;
    const bool block_allocated =
        getVoxelAtPosition<ColorVoxel>(color_layer_host, p, &color_voxel);
    if (block_allocated) {
      EXPECT_EQ(color_voxel->weight, 0.0f);
    }
  }

  // Generate a mesh from the "reconstruction"
  MeshIntegrator mesh_integrator;
  MeshLayer mesh_layer(block_size_m_, MemoryType::kDevice);
  EXPECT_TRUE(
      mesh_integrator.integrateMeshFromDistanceField(gt_layer_, &mesh_layer));
  mesh_integrator.colorMesh(color_layer, &mesh_layer);

  if (FLAGS_nvblox_test_file_output) {
    io::outputMeshLayerToPly(mesh_layer, "colored_spheres_occluded.ply");
  }
}

TEST_F(ColorIntegrationTest, WeightingFunction) {
  // Integrator
  ProjectiveColorIntegrator integrator;

  // Change to constant weight
  EXPECT_EQ(integrator.weighting_function_type(),
            ProjectiveColorIntegrator::kDefaultWeightingFunctionType);
  integrator.weighting_function_type(
      WeightingFunctionType::kInverseSquareWeight);
  EXPECT_EQ(integrator.weighting_function_type(),
            WeightingFunctionType::kInverseSquareWeight);

  // Scene
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-10.0f, -10.0f, -10.0f),
                                        Vector3f(10.0f, 10.0f, 10.0f));
  scene.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(0.0f, 0.0f, 5.0f), Vector3f(0.0f, 0.0f, -1.0f)));

  // Layers
  constexpr float kMaxDist = 5.0f;
  constexpr float kVoxelSizeM = 0.1f;
  TsdfLayer tsdf_layer(kVoxelSizeM, MemoryType::kUnified);
  ColorLayer color_layer(kVoxelSizeM, MemoryType::kUnified);
  scene.generateLayerFromScene(kMaxDist, &tsdf_layer);

  // Make a flat red image.
  ColorImage color_image(camera_.rows(), camera_.cols(), MemoryType::kUnified);
  for (int i = 0; i < color_image.numel(); i++) {
    color_image(i) = Color::Red();
  }

  // Integrate a frame
  std::vector<Index3D> updated_blocks;
  integrator.integrateFrame(color_image, Transform::Identity(), camera_,
                            tsdf_layer, &color_layer, &updated_blocks);

  // Check that something actually happened
  EXPECT_GT(updated_blocks.size(), 0);
  EXPECT_GT(color_layer.numAllocatedBlocks(), 0);

  // Go over the voxels and check that they have the constant weight that we
  // expect.
  int num_voxels_observed = 0;
  for (const Index3D& block_idx : color_layer.getAllBlockIndices()) {
    // Get each voxel and it's position
    auto block_ptr = color_layer.getBlockAtIndex(block_idx);
    constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
    for (int x = 0; x < kVoxelsPerSide; x++) {
      for (int y = 0; y < kVoxelsPerSide; y++) {
        for (int z = 0; z < kVoxelsPerSide; z++) {
          // Get the voxel and check it has weight 1.0
          const ColorVoxel& voxel = block_ptr->voxels[x][y][z];
          constexpr float kFloatEps = 1e-4;
          if (voxel.weight > kFloatEps) {
            // Get the depth of the voxel
            const Index3D voxel_idx(x, y, z);
            const Vector3f voxel_center =
                getCenterPositionFromBlockIndexAndVoxelIndex(
                    color_layer.block_size(), block_idx, voxel_idx);
            const float voxel_depth = voxel_center.z();

            // Hand computing the inverse square weight
            const float weight_hand_computed =
                1.0f / (voxel_depth * voxel_depth);

            // Check
            EXPECT_NEAR(voxel.weight, weight_hand_computed, kFloatEps);
            ++num_voxels_observed;
          }
        }
      }
    }
  }
  LOG(INFO) << "Number of voxels observed: " << num_voxels_observed;
  EXPECT_GT(num_voxels_observed, 0);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
