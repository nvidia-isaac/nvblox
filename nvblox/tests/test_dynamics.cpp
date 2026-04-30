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

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/dynamics/dynamics_detection.h"
#include "nvblox/integrators/freespace_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/tests/sensor_fixture.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;
template <typename SensorType>
class DynamicsTester : public ::test_utils::SensorFixture<SensorType> {
 protected:
  // Test layer
  constexpr static float voxel_size_m_ = 0.05;
  constexpr static float truncation_distance_vox_ = 4;
  constexpr static float truncation_distance_meters_ =
      truncation_distance_vox_ * voxel_size_m_;
  constexpr static float max_integration_distance_ = 20.0f;
};

using SensorTypes = ::testing::Types<Camera, Lidar>;
TYPED_TEST_SUITE(DynamicsTester, SensorTypes);

TYPED_TEST(DynamicsTester, PrimitiveScene) {
  // Overview of test steps:
  // (1) create a scene with a sphere in a box
  // (2) generate a freespace layer from the scene
  // (3) add a cube to the scene and create a depth image from it
  // (4) check dynamics on the depth image:
  //     - the cube should be detected dynamic because it was not integrated
  //       into the freespace layer
  constexpr float kMaxProjectionDist = 10.0;

  // Get a sample scene
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-3.5f, -3.0f, 0.0f),
                                        Vector3f(5.0f, 3.0f, 3.0f));
  scene.addGroundLevel(0.0f);
  scene.addCeiling(3.0f);
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.2f, 1.0f), 1.0f));
  scene.addPlaneBoundaries(-3.5f, 5.0f, -3.0f, 2.0f);

  // Calculate the freespace layer of the scene
  FreespaceLayer freespace_layer_L(this->voxel_size_m_, MemoryType::kUnified);
  scene.generateLayerFromScene(this->truncation_distance_meters_,
                               &freespace_layer_L);

  // Define the camera to layer transform
  Transform T_L_C = test_utils::getSensorTLC<TypeParam>();
  T_L_C.pretranslate(Vector3f(-3.f, 0.F, 1.0F));

  // Create a depth frame of the scene
  DepthImage depth_frame_C(this->sensor().height(), this->sensor().width(),
                           MemoryType::kUnified);
  scene.generateDepthImageFromScene(this->sensor(), T_L_C, kMaxProjectionDist,
                                    &depth_frame_C);

  // Check that there are no dynamics detected
  DynamicsDetection detector(std::make_shared<CudaStreamOwning>());
  detector.computeDynamics(depth_frame_C, freespace_layer_L, this->sensor(),
                           T_L_C);
  auto dynamic_points = detector.getDynamicPointsHost();
  EXPECT_EQ(dynamic_points.cols(), 0);

  // Add a cube to the scene
  Vector3f cube_center(-1.0f, -1.5f, 2.0f);
  Vector3f cube_size(1.5f, 0.5f, 0.5f);
  scene.addPrimitive(
      std::make_unique<primitives::Cube>(cube_center, cube_size));

  // Create a depth frame of the scene including the cube
  DepthImage depth_frame_cube_C(this->sensor().height(), this->sensor().width(),
                                MemoryType::kUnified);
  scene.generateDepthImageFromScene(this->sensor(), T_L_C, kMaxProjectionDist,
                                    &depth_frame_cube_C);

  // Check that the cube is detected as dynamic
  // as it was added after creating the layer
  detector.computeDynamics(depth_frame_cube_C, freespace_layer_L,
                           this->sensor(), T_L_C);
  auto dynamic_points_cube = detector.getDynamicPointsHost();
  auto& dynamic_overlay_cube = detector.getDynamicOverlayImage();
  EXPECT_GT(dynamic_points_cube.cols(), 0);

  constexpr float kFloatEpsilon = 1e-4;
  for (int i = 0; i < dynamic_points_cube.cols(); i++) {
    Vector3f point = dynamic_points_cube.col(i);

    // The point should be contained by the cube
    EXPECT_LE(std::abs(cube_center.x() - point.x()),
              cube_size.x() / 2.0f + kFloatEpsilon);
    EXPECT_LE(std::abs(cube_center.y() - point.y()),
              cube_size.y() / 2.0f + kFloatEpsilon);
    EXPECT_LE(std::abs(cube_center.z() - point.z()),
              cube_size.z() / 2.0f + kFloatEpsilon);

    // The point should be part of at least one side plane of the cube
    bool on_cube_boundary = false;
    on_cube_boundary |=
        std::abs(cube_center.x() - point.x()) - cube_size.x() <= kFloatEpsilon;
    on_cube_boundary |=
        std::abs(cube_center.y() - point.y()) - cube_size.y() <= kFloatEpsilon;
    on_cube_boundary |=
        std::abs(cube_center.z() - point.z()) - cube_size.z() <= kFloatEpsilon;
    EXPECT_TRUE(on_cube_boundary);
  }

  if (FLAGS_nvblox_test_file_output) {
    io::writeToPng("depth_image_C.png", depth_frame_C);
    io::writeToPng("depth_frame_cube_C.png", depth_frame_cube_C);
    io::writeToPng("dynamic_overlay_cube.png", dynamic_overlay_cube);
    io::outputPointMatrixToPly(dynamic_points_cube, "dynamic_points.ply");
    // Update the layer to include the cube (for visualization)
    scene.generateLayerFromScene(this->truncation_distance_meters_,
                                 &freespace_layer_L);
    io::outputVoxelLayerToPly(freespace_layer_L, "freespace_layer_L.ply");
  }
}

TYPED_TEST(DynamicsTester, HumanDataset) {
  // Overview of test steps:
  // (1) load two consecutive depth images that come from a static camera and
  //     show a human walking.
  // (2) generate a tsdf and freespace layer from the first image
  // (3) compute the dynamics on the second image compared
  //     to the freespace layer
  // (4) "check" that the human is detected dynamic

  // Load frames
  DepthImage depth_frame_C(MemoryType::kUnified),
      depth_frame_L(MemoryType::kUnified);
  ColorImage color_frame_S(MemoryType::kUnified),
      color_frame_L(MemoryType::kUnified);
  const std::string base_path =
      test_utils::getTestDataPath("data/human_dataset/");
  EXPECT_TRUE(io::readFromPng(base_path + "depth_image_1.png", &depth_frame_L));
  EXPECT_TRUE(io::readFromPng(base_path + "depth_image_2.png", &depth_frame_C));

  // Load camera
  Camera camera;
  Eigen::Matrix3f camera_intrinsic_matrix;
  EXPECT_TRUE(datasets::threedmatch::internal::parseCameraFromFile(
      base_path + "camera-intrinsics.txt", &camera_intrinsic_matrix));
  camera = Camera::fromIntrinsicsMatrix(
      camera_intrinsic_matrix, depth_frame_L.width(), depth_frame_L.height());

  // Create the tsdf layer with the first depth frame
  ProjectiveTsdfIntegrator integrator;
  integrator.truncation_distance_vox(this->truncation_distance_vox_);
  integrator.max_integration_distance_m(this->max_integration_distance_);
  TsdfLayer tsdf_layer_L(this->voxel_size_m_, MemoryType::kUnified);
  std::vector<Index3D> updated_blocks;
  integrator.integrateFrame(
      MaskedDepthImageConstView(depth_frame_L, kMaskActiveEverywhere),
      Transform::Identity(), camera, &tsdf_layer_L, &updated_blocks);

  // Prepare the freespace layer and integrator
  FreespaceLayer freespace_layer_L(this->voxel_size_m_, MemoryType::kUnified);
  FreespaceIntegrator freespace_integrator;
  const Time duration_to_change_to_freespace_ms{1000};
  // This parameter must be smaller than truncation distance
  freespace_integrator.max_tsdf_distance_for_occupancy_m(
      this->truncation_distance_meters_ * 0.75);
  freespace_integrator.min_duration_since_occupied_for_freespace_ms(
      duration_to_change_to_freespace_ms);
  freespace_integrator.check_neighborhood(false);  // make testing easier

  // Calculate the freespace from the tsdf layer
  const Time start_time_ms{100};
  // Initialize layer
  freespace_integrator.updateFreespaceLayer<Camera>(
      tsdf_layer_L.getAllBlockIndices(), start_time_ms, tsdf_layer_L, {},
      &freespace_layer_L);
  // Update to generate high confidence freespace in free areas
  freespace_integrator.updateFreespaceLayer<Camera>(
      tsdf_layer_L.getAllBlockIndices(),
      start_time_ms + duration_to_change_to_freespace_ms, tsdf_layer_L, {},
      &freespace_layer_L);

  // Compute the dynamics on the second depth frame (relative to the tsdf)
  DynamicsDetection detector(std::make_shared<CudaStreamOwning>());
  detector.computeDynamics(depth_frame_C, freespace_layer_L, camera,
                           Transform::Identity());
  auto dynamic_points = detector.getDynamicPointsHost();
  auto& dynamic_overlay = detector.getDynamicOverlayImage();

  // Note: This only tests that there are a reasonable amount of dynamic points
  // If you wish to get a better idea of the performance, check the output data.
  EXPECT_GT(dynamic_points.cols(), depth_frame_C.numel() / 20.0f);
  EXPECT_LT(dynamic_points.cols(), depth_frame_C.numel() / 5.0f);

  if (FLAGS_nvblox_test_file_output) {
    io::writeToPng("depth_frame_L.png", depth_frame_L);
    io::writeToPng("depth_frame_C.png", depth_frame_C);
    io::writeToPng("dynamic_overlay.png", dynamic_overlay);
    io::outputPointMatrixToPly(dynamic_points, "dynamic_points.ply");
    io::outputVoxelLayerToPly(freespace_layer_L, "freespace_layer_L.ply");
    io::outputVoxelLayerToPly(tsdf_layer_L, "tsdf_layer_L.ply");
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
