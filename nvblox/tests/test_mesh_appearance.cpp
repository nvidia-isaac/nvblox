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
#include "nvblox/integrators/projective_appearance_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

template <typename AppearanceVoxelType>
typename AppearanceVoxelType::ArrayType getTestAppearance() {
  if constexpr (std::is_same_v<AppearanceVoxelType, ColorVoxel>) {
    return Color::Purple();
  } else if constexpr (std::is_same_v<AppearanceVoxelType, FeatureVoxel>) {
    // Return a feature array with all values set to 1.0
    auto features = FeatureArray();
    for (size_t i = 0; i < features.size(); i++) {
      features[i] = 1.0f;
    }
    return features;
  } else {
    assert(false);
  }
}

template <typename AppearanceVoxelType>
class MeshAppearanceTest : public ::testing::Test {
 public:
  using AppearanceType = typename AppearanceVoxelType::ArrayType;
  using MeshBlockType = MeshBlock<AppearanceType>;
  using MeshLayerType = MeshBlockLayer<AppearanceType>;
  using AppearanceBlockType = VoxelBlock<AppearanceVoxelType>;
  using AppearanceLayerType = VoxelBlockLayer<AppearanceVoxelType>;

 protected:
  static constexpr float kVoxelSizeM = 0.1;
  static constexpr float kBlockSizeM =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * kVoxelSizeM;
  static constexpr float kTruncationDistanceVox = 2;
  static constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * kVoxelSizeM;
  const AppearanceType kTestAppearance =
      getTestAppearance<AppearanceVoxelType>();
  TsdfLayer tsdf_layer{kVoxelSizeM, MemoryType::kUnified};
  AppearanceLayerType appearance_layer{kVoxelSizeM, MemoryType::kUnified};
  MeshIntegrator<AppearanceVoxelType> mesh_integrator;
  MeshLayerType mesh_layer{kBlockSizeM, MemoryType::kUnified};

  void SetUp() override {
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
    for (const Index3D& block_idx : tsdf_layer.getAllBlockIndices()) {
      typename AppearanceBlockType::Ptr color_block =
          appearance_layer.allocateBlockAtIndex(block_idx);
      callFunctionOnAllVoxels<AppearanceVoxelType>(
          color_block.get(),
          [this](const Index3D&, AppearanceVoxelType* voxel) {
            if constexpr (std::is_same_v<AppearanceVoxelType, ColorVoxel>) {
              voxel->color = this->kTestAppearance;
            } else if constexpr (std::is_same_v<AppearanceVoxelType,
                                                FeatureVoxel>) {
              voxel->feature = this->kTestAppearance;
            } else {
              assert(false);
            }
          });
    }
  }
};

using AppearanceVoxelTypes = ::testing::Types<ColorVoxel, FeatureVoxel>;
TYPED_TEST_SUITE(MeshAppearanceTest, AppearanceVoxelTypes);

TYPED_TEST(MeshAppearanceTest, UniformColorSphere) {
  // Generate a mesh from the "reconstruction"
  EXPECT_TRUE(this->mesh_integrator.integrateMeshFromDistanceField(
      this->tsdf_layer, &this->mesh_layer));
  this->mesh_integrator.updateAppearance(this->appearance_layer,
                                         &this->mesh_layer);

  // Check that all the mesh points are correctly colored
  callFunctionOnAllBlocks<typename TestFixture::MeshBlockType>(
      this->mesh_layer,
      [this](const Index3D&,
             const typename TestFixture::MeshBlockType* mesh_block) {
        EXPECT_EQ(mesh_block->vertices.size(),
                  mesh_block->vertex_appearances.size());
        for (const typename TestFixture::AppearanceType& appearance :
             mesh_block->vertex_appearances) {
          // Compare arrays element-wise since operator== is not defined
          for (size_t i = 0; i < appearance.size(); i++) {
            EXPECT_EQ(appearance[i], this->kTestAppearance[i]);
          }
        }
      });
}

TEST(MeshColoringTest, CPUvsGPUon3DMatch) {
  // Load 3dmatch image
  const std::string base_path = test_utils::getTestDataPath("data/3dmatch");
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
  tsdf_integrator.integrateFrame(
      MaskedDepthImageConstView(depth_image_1, kMaskActiveEverywhere),
      Transform::Identity(), camera, &tsdf_layer);

  // Integrate Color (GPU)
  ProjectiveColorIntegrator color_integrator;
  ColorLayer color_layer(kVoxelSizeM, MemoryType::kDevice);
  color_integrator.integrateFrame(
      MaskedColorImageConstView(color_image_1, kMaskActiveEverywhere),
      Transform::Identity(), camera, tsdf_layer, &color_layer);
  ColorLayer color_layer_host(kVoxelSizeM, MemoryType::kHost);
  color_layer_host.copyFrom(color_layer);

  // Generate a mesh from the "reconstruction"
  ColorMeshIntegrator mesh_integrator;
  ColorMeshLayer mesh_layer_colored_on_gpu(kBlockSizeM, MemoryType::kDevice);
  EXPECT_TRUE(mesh_integrator.integrateMeshFromDistanceField(
      tsdf_layer, &mesh_layer_colored_on_gpu));

  // Copy the mesh
  ColorMeshLayer mesh_layer_colored_on_cpu(kBlockSizeM, MemoryType::kHost);
  mesh_layer_colored_on_cpu.copyFrom(mesh_layer_colored_on_gpu);

  // Color on GPU and CPU
  mesh_integrator.updateAppearanceGPU(color_layer, &mesh_layer_colored_on_gpu);
  mesh_integrator.updateAppearanceCPU(
      color_layer_host, &mesh_layer_colored_on_cpu, CudaStreamOwning());

  // Compare colors between the two implementations
  int num_same = 0;
  int num_diff = 0;
  int num_diff_outside = 0;
  int total_vertices = 0;

  ColorMeshLayer mesh_layer_colored_on_gpu_host(kBlockSizeM, MemoryType::kHost);
  mesh_layer_colored_on_gpu_host.copyFrom(mesh_layer_colored_on_gpu);

  auto block_indices_gpu = mesh_layer_colored_on_gpu_host.getAllBlockIndices();
  auto block_indices_cpu = mesh_layer_colored_on_cpu.getAllBlockIndices();
  EXPECT_EQ(block_indices_gpu.size(), block_indices_cpu.size());
  for (size_t idx = 0; idx < block_indices_gpu.size(); idx++) {
    const Index3D& block_idx = block_indices_gpu[idx];

    ColorMeshBlock::ConstPtr block_gpu =
        mesh_layer_colored_on_gpu_host.getBlockAtIndex(block_idx);
    ColorMeshBlock::ConstPtr block_cpu =
        mesh_layer_colored_on_cpu.getBlockAtIndex(block_idx);
    CHECK(block_gpu);
    CHECK(block_cpu);

    EXPECT_EQ(block_gpu->vertices.size(), block_cpu->vertices.size());
    EXPECT_EQ(block_gpu->vertex_appearances.size(),
              block_cpu->vertex_appearances.size());
    EXPECT_EQ(block_gpu->vertices.size(), block_gpu->vertex_appearances.size());
    for (size_t i = 0; i < block_gpu->vertex_appearances.size(); i++) {
      EXPECT_TRUE(
          (block_gpu->vertices[i].array() == block_cpu->vertices[i].array())
              .all());
      if (block_gpu->vertex_appearances[i] ==
          block_cpu->vertex_appearances[i]) {
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
