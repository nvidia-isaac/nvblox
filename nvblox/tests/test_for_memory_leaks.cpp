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
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/map/common_names.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

#include "nvblox/tests/test_utils_cuda.h"

using namespace nvblox;

std::pair<int, float> getFreeGPUMemory() {
  size_t free_bytes;
  size_t total_bytes;
  checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
  const int free_mb = free_bytes / 1e6;
  const int total_mb = total_bytes / 1e6;
  const float free_percent =
      static_cast<float>(free_mb) * 100.0f / static_cast<float>(total_mb);
  return {free_mb, free_percent};
}

TEST(MemoryLeakTest, UnifiedVectorInt) {
  // Allocate a bunch of data
  unified_vector<int> vec;
  constexpr int kNumTestingRounds = 10;
  constexpr int kMegaBytesPerRound = 100;
  constexpr int kNumVariablesToAddPerRound =
      kMegaBytesPerRound * 1e6 / sizeof(int);
  int start_free_gpu_memory_mb;
  std::tie(start_free_gpu_memory_mb, std::ignore) = getFreeGPUMemory();
  for (int test_idx = 0; test_idx < kNumTestingRounds; test_idx++) {
    for (int i = 0; i < kNumVariablesToAddPerRound; i++) {
      vec.push_back(i);
    }
    test_utils::addOneToAllGPU(&vec);
    // Check (only on the final round)
    if (test_idx == (kNumTestingRounds - 1)) {
      for (int i = 0; i < kNumVariablesToAddPerRound; i++) {
        EXPECT_EQ(vec[i], i + 1);
      }
    }
    vec.clear();
    // Debug
    int free_gpu_memory_percent;
    std::tie(std::ignore, free_gpu_memory_percent) = getFreeGPUMemory();
    std::cout << "Percentage free: " << free_gpu_memory_percent << "\%"
              << std::endl;
  }
  int end_free_gpu_memory_mb;
  std::tie(end_free_gpu_memory_mb, std::ignore) = getFreeGPUMemory();

  // Check that there's approximately the same left over at the end
  int reduction_in_free_gpu_memory_mb =
      start_free_gpu_memory_mb - end_free_gpu_memory_mb;
  std::cout << "(Int) Memory difference: " << reduction_in_free_gpu_memory_mb
            << std::endl;

  // Check that memory isn't depleting.
  // NOTE(alexmillane): We dont know what else is going on on the GPU but
  // hopefully nothing that allocates 100Mb during this test run...
  // constexpr int kMaxAllowableMemoryReductionMb = 100;
  // EXPECT_LT(reduction_in_free_gpu_memory_mb, kMaxAllowableMemoryReductionMb);
}

TEST(MemoryLeakTest, 3DMatchMeshing) {
  // Load 3dmatch image
  const std::string base_path = "data/3dmatch";
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

  // Params
  constexpr float kVoxelSizeM = 0.05f;
  const float kBlockSizeM = VoxelBlock<TsdfVoxel>::kVoxelsPerSide * kVoxelSizeM;
  ProjectiveTsdfIntegrator tsdf_integrator;

  // Memory start
  int start_free_gpu_memory_mb;
  std::tie(start_free_gpu_memory_mb, std::ignore) = getFreeGPUMemory();

  constexpr int kNumRounds = 100;
  // Generate a mesh
  MeshIntegrator mesh_integrator;
  MeshLayer mesh_layer_gpu(kBlockSizeM, MemoryType::kDevice);
  for (int i = 0; i < kNumRounds; i++) {
    std::cout << "i: " << i << std::endl;
    // Integrate depth
    TsdfLayer tsdf_layer(kVoxelSizeM, MemoryType::kDevice);
    tsdf_integrator.integrateFrame(depth_image_1, Transform::Identity(), camera,
                                   &tsdf_layer);

    EXPECT_TRUE(mesh_integrator.integrateMeshFromDistanceField(
        tsdf_layer, &mesh_layer_gpu));
    tsdf_layer.clear();
    mesh_layer_gpu.clear();
    // Debug
    int free_gpu_memory_percent;
    std::tie(std::ignore, free_gpu_memory_percent) = getFreeGPUMemory();
    std::cout << "Percentage free: " << free_gpu_memory_percent << "\%"
              << std::endl;
  }
  int end_free_gpu_memory_mb;
  std::tie(end_free_gpu_memory_mb, std::ignore) = getFreeGPUMemory();

  // Check that there's approximately the same left over at the end
  int reduction_in_free_gpu_memory_mb =
      start_free_gpu_memory_mb - end_free_gpu_memory_mb;
  std::cout << "(3D Matching) Memory difference: "
            << reduction_in_free_gpu_memory_mb << std::endl;

  // Check that memory isn't depleting.
  // NOTE(alexmillane): We dont know what else is going on on the GPU but
  // hopefully nothing that allocates 100Mb during this test run...
  // constexpr int kMaxAllowableMemoryReductionMb = 100;
  // EXPECT_LT(reduction_in_free_gpu_memory_mb, kMaxAllowableMemoryReductionMb);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
