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

#include "nvblox/core/types.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/view_calculator.h"
#include "nvblox/io/csv.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/lidar.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-4;
class LidarIntegrationTest : public ::testing::Test {
 protected:
  LidarIntegrationTest()
      : lidar(num_azimuth_divisions, num_elevation_divisions,
              vertical_fov_rad) {
    //
  }

  const int num_azimuth_divisions = 1024;
  const int num_elevation_divisions = 16;
  const float vertical_fov_rad = 30.0f * M_PI / 180.0;

  const Lidar lidar;
  ;
};

Eigen::MatrixX3f generateSpherePointcloud(int num_azimuth_divisions,
                                          int num_elevation_divisions,
                                          float vertical_fov_rad,
                                          float radius) {
  // Pointcloud
  Eigen::MatrixX3f pointcloud(num_azimuth_divisions * num_elevation_divisions,
                              3);

  // Construct a pointcloud of a points at random distances.
  const float azimuth_increments_rad = 2 * M_PI / num_azimuth_divisions;
  const float polar_increments_rad =
      vertical_fov_rad / (num_elevation_divisions - 1);
  const float half_vertical_fov_rad = vertical_fov_rad / 2.0f;
  int point_idx = 0;
  for (int az_idx = 0; az_idx < num_azimuth_divisions; az_idx++) {
    for (int el_idx = 0; el_idx < num_elevation_divisions; el_idx++) {
      // Spherical coords
      const float azimuth_rad = az_idx * azimuth_increments_rad - M_PI;
      const float polar_rad =
          el_idx * polar_increments_rad + (M_PI / 2.0 - half_vertical_fov_rad);

      // Spherical to Cartesian
      const float x = radius * sin(polar_rad) * cos(azimuth_rad);
      const float y = radius * sin(polar_rad) * sin(azimuth_rad);
      const float z = radius * cos(polar_rad);

      pointcloud(point_idx, 0) = x;
      pointcloud(point_idx, 1) = y;
      pointcloud(point_idx, 2) = z;

      point_idx++;
    }
  }
  return pointcloud;
}

DepthImage depthImageFromPointcloud(const Eigen::MatrixX3f& pointcloud,
                                    const Lidar& lidar) {
  DepthImage depth_image(lidar.num_elevation_divisions(),
                         lidar.num_azimuth_divisions(), MemoryType::kUnified);
  for (int idx = 0; idx < pointcloud.rows(); idx++) {
    const Vector3f p_C = pointcloud.row(idx);
    Index2D u_C;
    lidar.project(p_C, &u_C);
    depth_image(u_C.y(), u_C.x()) = p_C.norm();
  }
  return depth_image;
}

TEST_F(LidarIntegrationTest, LidarAabb) {
  Transform T_L_C = Transform::Identity();
  AxisAlignedBoundingBox aabb = lidar.getViewAABB(T_L_C, 0.0f, 1.0);

  const float top_beam_z = 1.0 * sin(vertical_fov_rad / 2.0f);

  EXPECT_NEAR(aabb.min().x(), -1.0, kFloatEpsilon);
  EXPECT_NEAR(aabb.min().y(), -1.0, kFloatEpsilon);
  EXPECT_NEAR(aabb.min().z(), -top_beam_z, kFloatEpsilon);
  EXPECT_NEAR(aabb.max().x(), 1.0, kFloatEpsilon);
  EXPECT_NEAR(aabb.max().y(), 1.0, kFloatEpsilon);
  EXPECT_NEAR(aabb.max().z(), top_beam_z, kFloatEpsilon);

  T_L_C.translate(Vector3f(1.0f, 1.0f, 1.0f));

  aabb = lidar.getViewAABB(T_L_C, 0.0f, 1.0);

  EXPECT_NEAR(aabb.min().x(), 0.0, kFloatEpsilon);
  EXPECT_NEAR(aabb.min().y(), 0.0, kFloatEpsilon);
  EXPECT_NEAR(aabb.min().z(), -top_beam_z + 1.0f, kFloatEpsilon);
  EXPECT_NEAR(aabb.max().x(), 2.0, kFloatEpsilon);
  EXPECT_NEAR(aabb.max().y(), 2.0, kFloatEpsilon);
  EXPECT_NEAR(aabb.max().z(), top_beam_z + 1.0f, kFloatEpsilon);
}

TEST_F(LidarIntegrationTest, LidarBlocksInView) {
  // Params
  const float voxel_size = 0.1f;
  const float max_integration_distance_m = 20.0f;
  const float truncation_distance_vox = 4.0f;
  const float truncation_distance_m = truncation_distance_vox * voxel_size;

  // Generate data
  const float sphere_radius = 10.0;
  const Eigen::MatrixX3f pointcloud =
      generateSpherePointcloud(num_azimuth_divisions, num_elevation_divisions,
                               vertical_fov_rad, sphere_radius);
  const DepthImage depth_image = depthImageFromPointcloud(pointcloud, lidar);

  ViewCalculator view_calculator;

  const Transform T_L_C = Transform::Identity();

  std::vector<Index3D> blocks_in_view =
      view_calculator.getBlocksInImageViewRaycast(
          depth_image, T_L_C, lidar, voxelSizeToBlockSize(voxel_size),
          truncation_distance_m, max_integration_distance_m);

  Eigen::MatrixX3i blocks_in_view_mat(blocks_in_view.size(), 3);
  for (size_t idx = 0; idx < blocks_in_view.size(); idx++) {
    blocks_in_view_mat.row(idx) = blocks_in_view[idx];
  }

  // Offsets matrix
  Eigen::Matrix<float, 8, 3> offsets;
  offsets << 0, 0, 0,  // NOLINT
      0, 0, 1,         // NOLINT
      0, 1, 0,         // NOLINT
      0, 1, 1,         // NOLINT
      1, 0, 0,         // NOLINT
      1, 0, 1,         // NOLINT
      1, 1, 0,         // NOLINT
      1, 1, 1;         // NOLINT

  // Check that all these blocks have at least one corner in view
  for (const Index3D block_index : blocks_in_view) {
    // Lower corner position
    const Vector3f p_low = getPositionFromBlockIndex(
        voxelSizeToBlockSize(voxel_size), block_index);
    // Project each block corner
    bool did_at_least_one_corner_project = false;
    float min_corner_distance = 1000.0;
    for (int i = 0; i < 8; i++) {
      Vector2f u_C;
      const Vector3f p_C = p_low + offsets.row(i).transpose();
      bool did_project = lidar.project(p_C, &u_C);
      did_at_least_one_corner_project |= did_project;
      if (did_project) {
        min_corner_distance = std::min(min_corner_distance, p_C.norm());
      }
    }
    EXPECT_LT(min_corner_distance,
              sphere_radius + voxelSizeToBlockSize(voxel_size));
    EXPECT_TRUE(did_at_least_one_corner_project);
  }

  // Debug. Write block indices to file so I can display later.
  if (FLAGS_nvblox_test_file_output) {
    const std::string blocks_in_view_filepath = "./blocks_in_view_lidar.csv";
    io::writeToCsv(blocks_in_view_filepath, blocks_in_view_mat);
  }
}

TEST_F(LidarIntegrationTest, SurroundingSphere) {
  // Generate data
  const float sphere_radius = 10.0;
  const Eigen::MatrixX3f pointcloud =
      generateSpherePointcloud(num_azimuth_divisions, num_elevation_divisions,
                               vertical_fov_rad, sphere_radius);
  const DepthImage depth_image = depthImageFromPointcloud(pointcloud, lidar);

  // Pose: At the center of the sphere.
  const Transform T_L_C = Transform::Identity();

  // Integrate into a layer
  const float voxel_size = 0.1f;
  TsdfLayer layer(voxel_size, MemoryType::kDevice);

  ProjectiveTsdfIntegrator tsdf_integrator;
  tsdf_integrator.max_integration_distance_m(sphere_radius + 5.0f);
  tsdf_integrator.integrateFrame(depth_image, T_L_C, lidar, &layer);

  // Mesh
  MeshLayer mesh_layer(voxelSizeToBlockSize(voxel_size), MemoryType::kDevice);
  MeshIntegrator mesh_integrator;
  mesh_integrator.integrateMeshFromDistanceField(layer, &mesh_layer);

  // Check all TSDF voxels have close to the distance they should.
  const float block_size = voxelSizeToBlockSize(voxel_size);
  const float truncation_distance_m =
      tsdf_integrator.truncation_distance_vox() * voxel_size;
  auto lambda = [&](const Index3D& block_index, const Index3D& voxel_index,
                    const TsdfVoxel* voxel) {
    if (voxel->weight > 0.0f) {
      const Vector3f p_L = getCenterPositionFromBlockIndexAndVoxelIndex(
          block_size, block_index, voxel_index);
      float gt_distance = sphere_radius - p_L.norm();
      gt_distance = std::min(gt_distance, truncation_distance_m);
      gt_distance = std::max(gt_distance, -truncation_distance_m);
      EXPECT_NEAR(gt_distance, voxel->distance, kFloatEpsilon);
    }
  };
  callFunctionOnAllVoxels<TsdfVoxel>(layer, lambda);

  if (FLAGS_nvblox_test_file_output) {
    // Write out the mesh
    const std::string mesh_filepath = "lidar_sphere_mesh.ply";
    io::outputMeshLayerToPly(mesh_layer, mesh_filepath);
    // Output
    const std::string filepath = "sphere_lidar_image.png";
    io::writeToPng(filepath, depth_image);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
