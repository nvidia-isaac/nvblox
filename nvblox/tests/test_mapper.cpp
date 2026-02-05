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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <filesystem>

#include "nvblox/core/types.h"
#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

primitives::Scene getSphereInABoxScene(const Vector3f& sphere_center,
                                       const float sphere_radius) {
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-11.0f, -11.0f, -1.0f),
                                        Vector3f(11.0f, 11.0f, 11.0f));
  scene.addGroundLevel(0.0f);
  scene.addCeiling(10.0f);
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(sphere_center, sphere_radius));
  scene.addPlaneBoundaries(-10.0f, 10.0f, -10.0f, 10.0f);
  return scene;
}

bool allMeshPointsOnSphere(const ColorMeshLayer& mesh_layer,
                           const Vector3f& center, const float sphere_radius) {
  const std::shared_ptr<const ColorMesh> mesh =
      mesh_layer.getMesh(CudaStreamOwning());
  std::vector<Vector3f> vertices =
      mesh->vertices.toVectorAsync(CudaStreamOwning());
  for (const Vector3f& p : vertices) {
    constexpr float kVertexEps = 0.01;
    if (std::abs((p - center).norm() - sphere_radius) > kVertexEps) {
      return false;
    }
  }
  return true;
}

TEST(MapperTest, SettersAndGetters) {
  Mapper mapper(0.05f, MemoryType::kDevice);

  mapper.do_depth_preprocessing(true);
  EXPECT_TRUE(mapper.do_depth_preprocessing());
  mapper.do_depth_preprocessing(false);
  EXPECT_FALSE(mapper.do_depth_preprocessing());

  mapper.depth_preprocessing_num_dilations(123);
  EXPECT_EQ(mapper.depth_preprocessing_num_dilations(), 123);
}

TEST(MapperTest, ClearOutsideSphere) {
  // Create a scene with a sphere
  const Vector3f sphere_center(0.0f, 0.0f, 5.0f);
  const float sphere_radius = 2.0f;
  primitives::Scene scene = getSphereInABoxScene(sphere_center, sphere_radius);

  constexpr float voxel_size_m = 0.1;
  Mapper mapper(voxel_size_m, MemoryType::kDevice);

  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);

  scene.generateLayerFromScene(1.0, &tsdf_layer_host);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);

  EXPECT_GT(mapper.tsdf_layer().numBlocks(), 0);

  mapper.updateColorMesh(UpdateFullLayer::kYes);
  mapper.updateEsdf(UpdateFullLayer::kYes);

  // allocate color, just so we can clear later
  for (const Index3D& idx : mapper.tsdf_layer().getAllBlockIndices()) {
    mapper.color_layer().allocateBlockAtIndex(idx);
  }
  const float num_blocks_before_clear = mapper.tsdf_layer().numBlocks();

  // Create a copy of the mesh layer on host.
  ColorMeshLayer mesh_layer_host(voxel_size_m, MemoryType::kHost);
  mesh_layer_host.copyFrom(mapper.color_mesh_layer());

  // Not all mesh points are on the sphere (walls are there).
  EXPECT_FALSE(
      allMeshPointsOnSphere(mesh_layer_host, sphere_center, sphere_radius));

  // Clearing outside of sphere
  mapper.clearOutsideRadius(sphere_center, sphere_radius);

  EXPECT_GT(mapper.tsdf_layer().numBlocks(), 0);
  EXPECT_LT(mapper.tsdf_layer().numBlocks(), num_blocks_before_clear);
  EXPECT_EQ(mapper.tsdf_layer().numBlocks(), mapper.esdf_layer().numBlocks());
  EXPECT_EQ(mapper.tsdf_layer().numBlocks(), mapper.color_layer().numBlocks());

  ColorMeshLayer mesh_layer_host2(voxel_size_m, MemoryType::kHost);
  mesh_layer_host2.copyFrom(mapper.color_mesh_layer());

  // Test resulting mesh
  EXPECT_TRUE(
      allMeshPointsOnSphere(mesh_layer_host2, sphere_center, sphere_radius));

  if (FLAGS_nvblox_test_file_output) {
    io::outputColorMeshLayerToPly(mapper.color_mesh_layer(), "mapper_test.ply");
  }
}

TEST(MapperTest, IntegrateDepthWithMask) {
  // Scene
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, -5.0f),
                                        Vector3f(5.0f, 5.0f, 5.0f));
  scene.addGroundLevel(-4.0f);
  scene.addCeiling(4.0f);
  scene.addPlaneBoundaries(-4.0f, 4.0f, -4.0f, 4.0f);

  // Camera
  constexpr static float fu = 300;
  constexpr static float fv = 300;
  constexpr static int width = 640;
  constexpr static int height = 480;
  Camera camera(fu, fv, 0.5 * width, 0.5 * height, width, height);

  // Looking down the x-axis
  Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
  Eigen::Vector3f translation(0.0, 0.0, 0.0);
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation_base);
  T_S_C.pretranslate(translation);

  // Synthetic view of dreams
  DepthImage depth_image(height, width, MemoryType::kHost);
  constexpr float kSyntheticViewMaxDist = 20.0f;
  scene.generateDepthImageFromScene(camera, T_S_C, kSyntheticViewMaxDist,
                                    &depth_image);

  // Integrate a single frame with mask.
  const float voxel_size_m = 0.1;
  const float truncation_distance_vox = 2.F;

  Mapper mapper(voxel_size_m, MemoryType::kHost);
  MapperParams params;
  params.projective_integrator_params
      .projective_integrator_truncation_distance_vox = truncation_distance_vox;
  mapper.setMapperParams(params);

  MonoImage mask(depth_image.rows(), depth_image.cols(), MemoryType::kHost);
  mask.setZeroAsync(CudaStreamOwning());
  mapper.integrateDepth(MaskedDepthImageConstView(depth_image, mask), T_S_C,
                        camera);

  // We expect TSDF distances up to (but not including) the surface - with
  // truncation distance as margin.
  int num_voxels = 0;
  for (auto& block : mapper.tsdf_layer().getAllBlockPointers()) {
    for (auto& voxel : (*block)) {
      if (voxel.weight > 0) {
        EXPECT_GE(voxel.distance, truncation_distance_vox * voxel_size_m);
        ++num_voxels;
      }
    }
  }
  EXPECT_GT(num_voxels, 0);
}

TEST(MapperTest, GenerateEsdfInFakeObservedAreas) {
  // Scene
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, -5.0f),
                                        Vector3f(5.0f, 5.0f, 5.0f));
  scene.addGroundLevel(-4.0f);
  scene.addCeiling(4.0f);
  scene.addPlaneBoundaries(-4.0f, 4.0f, -4.0f, 4.0f);

  // Camera
  constexpr static float fu = 300;
  constexpr static float fv = 300;
  constexpr static int width = 640;
  constexpr static int height = 480;
  Camera camera(fu, fv, 0.5 * width, 0.5 * height, width, height);

  // Looking down the x-axis
  Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
  Eigen::Vector3f translation(0.0, 0.0, 0.0);
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation_base);
  T_S_C.pretranslate(translation);

  // Synthetic view of dreams
  DepthImage depth_image(height, width, MemoryType::kUnified);
  constexpr float kSyntheticViewMaxDist = 20.0f;
  scene.generateDepthImageFromScene(camera, T_S_C, kSyntheticViewMaxDist,
                                    &depth_image);

  // Mapper
  const float voxel_size_m = 0.1;
  Mapper mapper(voxel_size_m, MemoryType::kUnified);

  // Integrate a single frame
  mapper.integrateDepth(depth_image, T_S_C, camera);

  // Produce the ESDF
  mapper.updateEsdf();

  // Check that TSDF/ESDF is allocated in view, but not allocated behind the
  // robot
  EXPECT_TRUE(mapper.tsdf_layer().getBlockAtPosition(Vector3f(1.0, 0.0, 0.0)));
  EXPECT_FALSE(
      mapper.tsdf_layer().getBlockAtPosition(Vector3f(-1.0, 0.0, 0.0)));
  EXPECT_TRUE(mapper.esdf_layer().getBlockAtPosition(Vector3f(1.0, 0.0, 0.0)));
  EXPECT_FALSE(
      mapper.esdf_layer().getBlockAtPosition(Vector3f(-1.0, 0.0, 0.0)));

  // Get a voxel in the truncation band BEFORE marking below. Used in a test
  // later.
  auto vox_and_flag_before = mapper.tsdf_layer().getVoxel({4.0, 0.0, 0.0});
  EXPECT_TRUE(vox_and_flag_before.second);
  const TsdfVoxel voxel_in_band_before = vox_and_flag_before.first;

  // Fake observation
  const Eigen::Vector3f center(0.0, 0.0, 0.0);
  const float radius = 5.0;
  mapper.markUnobservedTsdfFreeInsideRadius(center, radius);

  // Check that:
  // - TSDF allocated behind robot.
  // - ESDF not-allocated behind robot.
  EXPECT_TRUE(mapper.tsdf_layer().getBlockAtPosition(Vector3f(-1.0, 0.0, 0.0)));
  EXPECT_FALSE(
      mapper.esdf_layer().getBlockAtPosition(Vector3f(-1.0, 0.0, 0.0)));

  // Update the ESDF
  mapper.updateEsdf();

  // Check that both TSDF and ESDF allocated behind robot
  EXPECT_TRUE(mapper.tsdf_layer().getBlockAtPosition(Vector3f(-1.0, 0.0, 0.0)));
  EXPECT_TRUE(mapper.esdf_layer().getBlockAtPosition(Vector3f(-1.0, 0.0, 0.0)));

  // Check that ESDF voxels in block behind that camera are observed and have
  // some positive value.
  auto esdf_block_ptr =
      mapper.esdf_layer().getBlockAtIndex(Index3D(-1.0, 0.0, 0.0));
  for (int x = 0; x < TsdfBlock::kVoxelsPerSide; x++) {
    for (int y = 0; y < TsdfBlock::kVoxelsPerSide; y++) {
      for (int z = 0; z < TsdfBlock::kVoxelsPerSide; z++) {
        auto esdf_voxel = esdf_block_ptr->voxels[x][y][y];
        EXPECT_TRUE(esdf_voxel.observed);
        EXPECT_GT(esdf_voxel.squared_distance_vox, 0);
      }
    }
  }

  // Check that previously observed voxels in the truncation band are
  // unaffected
  auto vox_and_flag_after = mapper.tsdf_layer().getVoxel({4.0, 0.0, 0.0});
  EXPECT_TRUE(vox_and_flag_after.second);
  auto voxel_in_band_after = vox_and_flag_after.first;
  constexpr float kEps = 1e-4;
  // One actual observation
  // NOTE(alexmillane): This weight of 1.0 is currently hardcoded in the
  // TsdfIntegrator. This may change at some point which would cause this test
  // to fail.
  EXPECT_NEAR(voxel_in_band_before.weight, voxel_in_band_after.weight, kEps);
  EXPECT_NEAR(voxel_in_band_before.distance, voxel_in_band_after.distance,
              kEps);
  // Distance is less than one voxel from the plane at 4m.
  EXPECT_LT(voxel_in_band_before.distance, voxel_size_m);

  // Save debug files
  if (FLAGS_nvblox_test_file_output) {
    io::writeToPng("./mapper_test_depth_image.png", depth_image);
    mapper.saveColorMeshAsPly("./mapper_test_plane_mesh.ply");
    mapper.saveEsdfAsPly("./mapper_test_plane_esdf.ply");
  }
}

class MapperLayerStreamerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Make a scene 6x6x3 meters big.
    primitives::Scene scene;

    Camera camera(300, 300, 320, 240, 640, 480);
    scene.addPrimitive(std::make_unique<primitives::Plane>(
        Vector3f(0.0f, 0.0, 1.0), Vector3f(0, 0, -1)));
    Transform T_S_C = Transform::Identity();
    DepthImage depth_frame(camera.height(), camera.width(),
                           MemoryType::kUnified);

    constexpr float max_dist = 10.0;
    scene.generateDepthImageFromScene(camera, T_S_C, max_dist, &depth_frame);

    TsdfLayer tsdf_layer_host(kVoxelSizeM, MemoryType::kHost);
    scene.generateLayerFromScene(1.0, &tsdf_layer_host);

    mapper_.tsdf_layer().copyFrom(tsdf_layer_host);
    mapper_.integrateDepth(depth_frame, T_S_C, camera);
    mapper_.updateColorMesh();
    EXPECT_GT(mapper_.tsdf_layer().numBlocks(), 0);
  }
  static constexpr float kVoxelSizeM = 0.1F;
  Mapper mapper_{kVoxelSizeM, MemoryType::kHost};
};

TEST_F(MapperLayerStreamerTest, ZeroMbps) {
  mapper_.serializeSelectedLayers(LayerType::kColorMesh, 0);
  ASSERT_EQ(mapper_.serializedColorMeshLayer()->block_indices.size(), 0);
}

TEST_F(MapperLayerStreamerTest, UnLimited) {
  mapper_.serializeSelectedLayers(LayerType::kColorMesh,
                                  kLayerStreamerUnlimitedBandwidth);
  ASSERT_GT(mapper_.serializedColorMeshLayer()->block_indices.size(), 0);
}

TEST_F(MapperLayerStreamerTest, Limited) {
  constexpr float kBandwidthLimit = 10.0F;
  mapper_.serializeSelectedLayers(LayerType::kColorMesh, kBandwidthLimit);
  // Since the mesh size is based on bandwidth estimation it's difficult to
  // predict how large it will be. Therefore we check for a nozero size here.
  ASSERT_GT(mapper_.serializedColorMeshLayer()->block_indices.size(), 0);
}

TEST_F(MapperLayerStreamerTest, MultipleLayers) {
  mapper_.serializeSelectedLayers(
      LayerTypeBitMask(LayerType::kColorMesh) | LayerType::kTsdf,
      kLayerStreamerUnlimitedBandwidth);
  ASSERT_GT(mapper_.serializedColorMeshLayer()->block_indices.size(), 0);
  ASSERT_GT(mapper_.serializedTsdfLayer()->block_indices.size(), 0);

  ASSERT_EQ(mapper_.serializedColorLayer()->block_indices.size(), 0);
  ASSERT_EQ(mapper_.serializedEsdfLayer()->block_indices.size(), 0);
}

TEST_F(MapperLayerStreamerTest, ColorAndTsdfHasSameNumberOfBlocks) {
  mapper_.serializeSelectedLayers(
      LayerTypeBitMask(LayerTypeBitMask(LayerType::kColor) | LayerType::kTsdf),
      kLayerStreamerUnlimitedBandwidth);

  ASSERT_GT(mapper_.serializedColorLayer()->block_indices.size(), 0);
  ASSERT_EQ(mapper_.serializedColorLayer()->block_indices.size(),
            mapper_.serializedTsdfLayer()->block_indices.size());
}

TEST(MapperTest, SaveAndLoad) {
  // Create a scene with a sphere
  const Vector3f sphere_center(0.0f, 0.0f, 5.0f);
  const float sphere_radius = 2.0f;
  primitives::Scene scene = getSphereInABoxScene(sphere_center, sphere_radius);

  constexpr float voxel_size_m = 0.1;
  Mapper mapper(voxel_size_m, MemoryType::kDevice);

  TsdfLayer tsdf_layer_host(voxel_size_m, MemoryType::kHost);

  scene.generateLayerFromScene(1.0, &tsdf_layer_host);
  mapper.tsdf_layer().copyFrom(tsdf_layer_host);

  EXPECT_GT(mapper.tsdf_layer().numBlocks(), 0);

  mapper.updateColorMesh(UpdateFullLayer::kYes);
  mapper.updateFeatureMesh(UpdateFullLayer::kYes);
  mapper.updateEsdf(UpdateFullLayer::kYes);

  // Save map to a temp file
  std::string fname =
      (std::filesystem::temp_directory_path() / "test_mapper.map").string();
  mapper.saveLayerCake(fname);

  // Load and validate
  Mapper mapper2(voxel_size_m, MemoryType::kDevice);
  mapper2.loadMap(fname);

  EXPECT_EQ(mapper.tsdf_layer().numBlocks(), mapper2.tsdf_layer().numBlocks());
  EXPECT_EQ(mapper.esdf_layer().numBlocks(), mapper2.esdf_layer().numBlocks());
  EXPECT_EQ(mapper.color_layer().numBlocks(),
            mapper2.color_layer().numBlocks());
  EXPECT_EQ(mapper.color_mesh_layer().numBlocks(),
            mapper2.color_mesh_layer().numBlocks());
  EXPECT_EQ(mapper.feature_mesh_layer().numBlocks(),
            mapper2.feature_mesh_layer().numBlocks());
  EXPECT_EQ(mapper.occupancy_layer().numBlocks(),
            mapper2.occupancy_layer().numBlocks());
  EXPECT_EQ(mapper.freespace_layer().numBlocks(),
            mapper2.freespace_layer().numBlocks());
}

TEST(MapperTest, IntegratePointcloud) {
  // Create a simple pointcloud
  Pointcloud pointcloud(MemoryType::kUnified);
  std::vector<Vector3f> points_host = {
      Vector3f(1.0f, 0.0f, 0.0f),  Vector3f(2.0f, 0.0f, 0.0f),
      Vector3f(3.0f, 0.0f, 0.0f),  Vector3f(1.5f, 0.5f, 0.0f),
      Vector3f(2.5f, -0.5f, 0.0f), Vector3f(2.0f, 1.0f, 0.0f),
      Vector3f(2.0f, -1.0f, 0.0f), Vector3f(3.0f, 0.5f, 0.5f)};
  pointcloud.copyPointsFromAsync(points_host, CudaStreamOwning());

  // Create a Lidar sensor with valid dimensions
  const int num_azimuth_divisions = 1024;
  const int num_elevation_divisions = 16;
  const float min_valid_range_m = 0.1f;
  const float vertical_fov_rad = 30.0f * M_PI / 180.0f;
  Lidar lidar(num_azimuth_divisions, num_elevation_divisions, min_valid_range_m,
              vertical_fov_rad);

  // Create Mapper
  const float voxel_size_m = 0.1f;
  Mapper mapper(voxel_size_m, MemoryType::kDevice, ProjectiveLayerType::kTsdf);

  // Integrate pointcloud without motion compensation
  Transform T_L_S = Transform::Identity();
  bool use_lidar_motion_compensation = false;
  mapper.integrateDepth(pointcloud, T_L_S, lidar,
                        use_lidar_motion_compensation);

  // Verify integration worked - should have allocated some blocks
  const int num_blocks_after_first = mapper.tsdf_layer().numBlocks();
  EXPECT_GT(num_blocks_after_first, 0);

  // Integrate pointcloud again with motion compensation from a different
  // location Motion compensation requires timestamps for each point
  const int64_t scan_duration_ms_value = 100;
  Time scan_duration_ms(scan_duration_ms_value);
  std::vector<Time> timestamps_ms(points_host.size());
  for (size_t i = 0; i < points_host.size(); i++) {
    // Spread timestamps evenly across scan duration
    timestamps_ms[i] = Time(i * scan_duration_ms_value / points_host.size());
  }
  pointcloud.copyTimestampsFromAsync(timestamps_ms, CudaStreamOwning());

  // Start from a different position to cover new space
  Transform T_L_S_scanStart2 = Transform::Identity();
  T_L_S_scanStart2.translate(Vector3f(0.0f, 3.0f, 0.0f));  // Start 3m away in y
  Transform T_L_S_scanEnd2 = Transform::Identity();
  T_L_S_scanEnd2.translate(Vector3f(0.0f, 4.0f, 0.0f));  // End 4m away in y
  use_lidar_motion_compensation = true;
  mapper.integrateDepth(pointcloud, T_L_S_scanStart2, lidar,
                        use_lidar_motion_compensation, T_L_S_scanEnd2,
                        scan_duration_ms);

  // Verify more blocks were allocated after second integration
  const int num_blocks_after_second = mapper.tsdf_layer().numBlocks();
  EXPECT_GT(num_blocks_after_second, num_blocks_after_first);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
