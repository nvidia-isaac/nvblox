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
#include "nvblox/datasets/image_loader.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/scene.h"

using namespace nvblox;

TEST(MeshColoringTests, UniformColorSphere) {
  // The uniform color
  const Color kTestColor = Color::Purple();

  // Layer params
  constexpr float voxel_size_m = 0.1;
  constexpr float block_size_m =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * voxel_size_m;

  // TSDF layer
  TsdfLayer tsdf_layer(voxel_size_m, MemoryType::kUnified);

  // Build the test scene
  constexpr float kTruncationDistanceVox = 2;
  constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * voxel_size_m;

  // Scene is bounded to -5, -5, 0 to 5, 5, 5.
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, 0.0f),
                                        Vector3f(5.0f, 5.0f, 5.0f));
  // Create a scene with a ground plane and a sphere.
  scene.addGroundLevel(0.0f);
  scene.addCeiling(5.0f);
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 2.0f), 2.0f));
  // Add bounding planes at 5 meters. Basically makes it sphere in a box.
  scene.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);

  // Get the ground truth SDF for it.
  scene.generateLayerFromScene(kTruncationDistanceMeters, &tsdf_layer);

  // Make a ColorLayer with a solid color
  ColorLayer color_layer(voxel_size_m, MemoryType::kUnified);
  for (const Index3D& block_idx : tsdf_layer.getAllBlockIndices()) {
    ColorBlock::Ptr color_block = color_layer.allocateBlockAtIndex(block_idx);
    callFunctionOnAllVoxels<ColorVoxel>(
        color_block.get(), [&kTestColor](const Index3D&, ColorVoxel* voxel) {
          voxel->color = kTestColor;
        });
  }

  // Generate a mesh from the "reconstruction"
  MeshIntegrator mesh_integrator;
  MeshLayer mesh_layer(block_size_m, MemoryType::kUnified);
  EXPECT_TRUE(
      mesh_integrator.integrateMeshFromDistanceField(tsdf_layer, &mesh_layer));
  mesh_integrator.colorMesh(color_layer, &mesh_layer);

  // Check that all the mesh points are correctly colored
  callFunctionOnAllBlocks<MeshBlock>(
      mesh_layer, [&kTestColor](const Index3D&, const MeshBlock* mesh_block) {
        EXPECT_EQ(mesh_block->vertices.size(), mesh_block->colors.size());
        for (const Color& color : mesh_block->colors) {
          EXPECT_EQ(color, kTestColor);
        }
      });
}

TEST(MeshColoringTests, CPUvsGPUon3DMatch) {
  // Load 3dmatch image
  const std::string base_path = "../tests/data/3dmatch";
  constexpr int seq_id = 1;
  DepthImage depth_image_1(MemoryType::kDevice);
  ColorImage color_image_1(MemoryType::kDevice);
  EXPECT_TRUE(datasets::load16BitDepthImage(
      datasets::threedmatch::internal::getPathForDepthImage(base_path, seq_id,
                                                            0),
      &depth_image_1));
  EXPECT_TRUE(datasets::load8BitColorImage(
      datasets::threedmatch::internal::getPathForColorImage(base_path, seq_id,
                                                            0),
      &color_image_1));
  EXPECT_EQ(depth_image_1.width(), color_image_1.width());
  EXPECT_EQ(depth_image_1.height(), color_image_1.height());

  // Parse 3x3 camera intrinsics matrix from 3D Match format: space-separated.
  Eigen::Matrix3f camera_intrinsic_matrix;
  EXPECT_TRUE(datasets::threedmatch::internal::parseCameraFromFile(
      datasets::threedmatch::internal::getPathForCameraIntrinsics(base_path),
      &camera_intrinsic_matrix));
  const auto camera = Camera::fromIntrinsicsMatrix(
      camera_intrinsic_matrix, depth_image_1.width(), depth_image_1.height());

  // Integrate depth
  constexpr float kVoxelSizeM = 0.05f;
  const float kBlockSizeM = VoxelBlock<TsdfVoxel>::kVoxelsPerSide * kVoxelSizeM;
  ProjectiveTsdfIntegrator tsdf_integrator;
  TsdfLayer tsdf_layer(kVoxelSizeM, MemoryType::kDevice);
  tsdf_integrator.integrateFrame(depth_image_1, Transform::Identity(), camera,
                                 &tsdf_layer);

  // Integrate Color (GPU)
  ProjectiveColorIntegrator color_integrator;
  ColorLayer color_layer(kVoxelSizeM, MemoryType::kDevice);
  color_integrator.integrateFrame(color_image_1, Transform::Identity(), camera,
                                  tsdf_layer, &color_layer);
  ColorLayer color_layer_host(kVoxelSizeM, MemoryType::kHost);
  color_layer_host.copyFrom(color_layer);

  // Generate a mesh from the "reconstruction"
  MeshIntegrator mesh_integrator;
  MeshLayer mesh_layer_colored_on_gpu(kBlockSizeM, MemoryType::kDevice);
  EXPECT_TRUE(mesh_integrator.integrateMeshFromDistanceField(
      tsdf_layer, &mesh_layer_colored_on_gpu));

  // Copy the mesh
  MeshLayer mesh_layer_colored_on_cpu(kBlockSizeM, MemoryType::kHost);
  mesh_layer_colored_on_cpu.copyFrom(mesh_layer_colored_on_gpu);

  // Color on GPU and CPU
  mesh_integrator.colorMeshGPU(color_layer, &mesh_layer_colored_on_gpu);
  mesh_integrator.colorMeshCPU(color_layer_host, &mesh_layer_colored_on_cpu);

  // Compare colors between the two implementations
  int num_same = 0;
  int num_diff = 0;
  int num_diff_outside = 0;
  int total_vertices = 0;

  MeshLayer mesh_layer_colored_on_gpu_host(kBlockSizeM, MemoryType::kHost);
  mesh_layer_colored_on_gpu_host.copyFrom(mesh_layer_colored_on_gpu);

  auto block_indices_gpu = mesh_layer_colored_on_gpu_host.getAllBlockIndices();
  auto block_indices_cpu = mesh_layer_colored_on_cpu.getAllBlockIndices();
  EXPECT_EQ(block_indices_gpu.size(), block_indices_cpu.size());
  for (size_t idx = 0; idx < block_indices_gpu.size(); idx++) {
    const Index3D& block_idx = block_indices_gpu[idx];

    MeshBlock::ConstPtr block_gpu =
        mesh_layer_colored_on_gpu_host.getBlockAtIndex(block_idx);
    MeshBlock::ConstPtr block_cpu =
        mesh_layer_colored_on_cpu.getBlockAtIndex(block_idx);
    CHECK(block_gpu);
    CHECK(block_cpu);

    EXPECT_EQ(block_gpu->vertices.size(), block_cpu->vertices.size());
    EXPECT_EQ(block_gpu->colors.size(), block_cpu->colors.size());
    EXPECT_EQ(block_gpu->vertices.size(), block_gpu->colors.size());
    for (size_t i = 0; i < block_gpu->colors.size(); i++) {
      EXPECT_TRUE(
          (block_gpu->vertices[i].array() == block_cpu->vertices[i].array())
              .all());
      if (block_gpu->colors[i] == block_cpu->colors[i]) {
        num_same++;
      } else {
        num_diff++;
        // OK so we have different colors at this vertex.
        // This CAN occur because of vertices that leave block boundaries. (For
        // speed we only take the closest color voxel in the corresponding
        // block).
        // Let's check that this is indeed the case here.

        // Calculate the position of this vertex in the block
        const Vector3f p_V_B_m =
            block_cpu->vertices[i] -
            getPositionFromBlockIndex(kBlockSizeM, block_idx);
        const Vector3f p_V_B_vox = p_V_B_m / kVoxelSizeM;
        if ((p_V_B_vox.array() > VoxelBlock<TsdfVoxel>::kVoxelsPerSide).any()) {
          num_diff_outside++;
        }
      }
    }
    total_vertices += block_cpu->vertices.size();
  }

  // OK so there are a few verices which get different colors and are WITHIN
  // block boundaries. From looking at them this is because they sit exactly on
  // voxel boundaries and presumably are rounded different ways in the CPU and
  // GPU implementations. Let's just check that the number of such occurances
  // are exceedingly small.
  constexpr float kAllowablePercentageDifferingColorWithinBlock = 0.1f;  // 0.1%
  const float percentage_different =
      100.0f * (num_diff - num_diff_outside) / total_vertices;
  EXPECT_LT(percentage_different,
            kAllowablePercentageDifferingColorWithinBlock);

  std::cout << "Number of vertices assigned the SAME color between CPU and GPU "
               "implementations: "
            << num_same << std::endl;
  std::cout << "Number of vertices assigned DIFFERING color between CPU and "
               "GPU implementations: "
            << num_diff << std::endl;
  std::cout << "of these, number with vertexes outside block boundaries: "
            << num_diff_outside << std::endl;
  std::cout
      << "The percentage of vertexes with different colors within blocks: "
      << percentage_different << "%" << std::endl;
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
