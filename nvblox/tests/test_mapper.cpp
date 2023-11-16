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
#include "nvblox/utils/logging.h"

#include "nvblox/core/types.h"
#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/primitives/scene.h"
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

bool allMeshPointsOnSphere(const MeshLayer& mesh_layer, const Vector3f& center,
                           const float sphere_radius) {
  Mesh mesh = Mesh::fromLayer(mesh_layer);
  for (const Vector3f p : mesh.vertices) {
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

  EXPECT_GT(mapper.tsdf_layer().numAllocatedBlocks(), 0);

  mapper.updateFullMesh();
  mapper.updateFullEsdf();

  // allocate color, just so we can clear later
  for (const Index3D& idx : mapper.tsdf_layer().getAllBlockIndices()) {
    mapper.color_layer().allocateBlockAtIndex(idx);
  }

  // Create a copy of the mesh layer on host.
  MeshLayer mesh_layer_host(voxel_size_m, MemoryType::kHost);
  mesh_layer_host.copyFrom(mapper.mesh_layer());

  // Not all mesh points are on the sphere (walls are there).
  EXPECT_FALSE(
      allMeshPointsOnSphere(mesh_layer_host, sphere_center, sphere_radius));

  // Clearing outside of sphere
  mapper.clearOutsideRadius(sphere_center, sphere_radius);

  EXPECT_EQ(mapper.tsdf_layer().numAllocatedBlocks(),
            mapper.esdf_layer().numAllocatedBlocks());
  EXPECT_EQ(mapper.tsdf_layer().numAllocatedBlocks(),
            mapper.color_layer().numAllocatedBlocks());

  MeshLayer mesh_layer_host2(voxel_size_m, MemoryType::kHost);
  mesh_layer_host2.copyFrom(mapper.mesh_layer());

  // Test resulting mesh
  EXPECT_TRUE(
      allMeshPointsOnSphere(mesh_layer_host2, sphere_center, sphere_radius));

  if (FLAGS_nvblox_test_file_output) {
    io::outputMeshLayerToPly(mapper.mesh_layer(), "mapper_test.ply");
  }
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
  Camera camera(fu, fv, width, height);

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

  // Check that previously observed voxels in the truncation band are unaffected
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
    mapper.saveMeshAsPly("./mapper_test_plane_mesh.ply");
    mapper.saveEsdfAsPly("./mapper_test_plane_esdf.ply");
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
