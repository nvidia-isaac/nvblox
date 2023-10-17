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
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"

#include "nvblox/tests/projective_tsdf_integrator_cuda_components.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

// TODO: Decide where to put test epsilons
constexpr float kFloatEpsilon = 1e-4;
constexpr float kImagePlaneEpsilon = 1e-2;

class CudaTsdfIntegratorTest : public ::testing::Test {
 protected:
  CudaTsdfIntegratorTest()
      : layer_(block_size_m_, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  // Test layer
  constexpr static float voxel_size_m_ = 0.2;
  constexpr static float block_size_m_ =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * voxel_size_m_;
  TsdfLayer layer_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

Transform getRandomTransform() {
  return Transform(
      Eigen::Translation3f(Vector3f::Random()) *
      Eigen::AngleAxisf(
          test_utils::randomFloatInRange(-M_PI, M_PI),
          test_utils::getRandomVector3fInRange(-1.0f, 1.0f).normalized()));
}

TEST_F(CudaTsdfIntegratorTest, GPUVectorizedPointsTransform) {
  // Make sure this is deterministic.
  std::srand(0);

  const Transform T_B_A = getRandomTransform();

  // Transform vectors on the CPU
  constexpr int kNumVecs = 1000;
  const Eigen::Matrix<float, 3, kNumVecs> vecs_A =
      Eigen::Matrix<float, 3, kNumVecs>::Random();
  const Eigen::Matrix<float, 3, kNumVecs> vecs_B =
      (T_B_A.rotation() * vecs_A).colwise() + T_B_A.translation();

  // Transform vectors on the GPU
  const Eigen::Matrix3Xf vecs_B_from_GPU =
      test_utils::transformPointsOnGPU(T_B_A, vecs_A);

  // Check we get the same result.
  EXPECT_NEAR((vecs_B - vecs_B_from_GPU).maxCoeff(), 0.0f, kFloatEpsilon);
}

TEST_F(CudaTsdfIntegratorTest, GpuBlockCenterProjection) {
  std::vector<Index3D> block_indices;
  block_indices.push_back(Index3D(0.0f, 0.0f, 0.0f));

  // Camera Pose
  Transform T_C_L = Transform::Identity();

  // Project on CPU
  std::vector<test_utils::BlockProjectionResult> cpu_block_projections;
  cpu_block_projections.reserve(block_indices.size());
  for (const Index3D& block_index : block_indices) {
    test_utils::BlockProjectionResult results_block;
    int lin_voxel_idx = 0;
    for (int x = 0; x < VoxelBlock<bool>::kVoxelsPerSide; x++) {
      for (int y = 0; y < VoxelBlock<bool>::kVoxelsPerSide; y++) {
        for (int z = 0; z < VoxelBlock<bool>::kVoxelsPerSide; z++) {
          const Vector3f p_L = getCenterPositionFromBlockIndexAndVoxelIndex(
              layer_.block_size(), block_index, Index3D(x, y, z));
          const Vector3f p_C = T_C_L * p_L;
          Eigen::Vector2f u_px;
          if (!camera_.project(p_C, &u_px)) {
            u_px = Eigen::Vector2f(0.0f, 0.0f);
          }
          results_block.row(lin_voxel_idx) = u_px;
          ++lin_voxel_idx;
        }
      }
    }
    cpu_block_projections.push_back(results_block);
  }

  // Project on GPU
  std::vector<test_utils::BlockProjectionResult> gpu_block_projections =
      test_utils::projectBlocksOnGPU(block_indices, camera_, T_C_L, &layer_);

  // Check
  for (size_t i = 0; i < gpu_block_projections.size(); i++) {
    EXPECT_NEAR(
        (cpu_block_projections[i] - gpu_block_projections[i]).maxCoeff(), 0.0f,
        kFloatEpsilon);
  }
}

TEST_F(CudaTsdfIntegratorTest, GpuDepthImageInterpolation) {
  // Make sure this is deterministic.
  std::srand(0);

  // The frames {x_indices_, y_indices_} are set up such that if you
  // interpolate, you should get the interpolated position back. Note that here,
  // (in contrast to the 3D voxel grid) I consider that pixel indices correspond
  // to centers (rather than low-side-corners).

  DepthImage depth_frame_col_indices(height_, width_, MemoryType::kUnified);
  DepthImage depth_frame_row_indices(height_, width_, MemoryType::kUnified);
  for (int col_idx = 0; col_idx < width_; col_idx++) {
    for (int row_idx = 0; row_idx < height_; row_idx++) {
      depth_frame_col_indices(row_idx, col_idx) = col_idx;
      depth_frame_row_indices(row_idx, col_idx) = row_idx;
    }
  }

  // Generate random locations on the image plane
  constexpr int kNumTests = 1000;
  Eigen::MatrixX2f u_px_vec(kNumTests, 2);
  for (int i = 0; i < kNumTests; i++) {
    u_px_vec.row(i) = Vector2f(
        test_utils::randomFloatInRange(0.5f, static_cast<float>(width_ - 0.5f)),
        test_utils::randomFloatInRange(0.5f,
                                       static_cast<float>(height_ - 0.5f)));
  }

  // CPU interpolation
  Eigen::VectorXf results_col_cpu(kNumTests);
  Eigen::VectorXf results_row_cpu(kNumTests);
  for (int i = 0; i < kNumTests; i++) {
    // Interpolate x and y grids
    EXPECT_TRUE(interpolation::interpolate2DLinear(
        depth_frame_col_indices, u_px_vec.row(i), &results_col_cpu(i)));
    EXPECT_TRUE(interpolation::interpolate2DLinear(
        depth_frame_row_indices, u_px_vec.row(i), &results_row_cpu(i)));
  }

  // GPU interpolation
  const Eigen::VectorXf results_col_gpu =
      test_utils::interpolatePointsOnGPU(depth_frame_col_indices, u_px_vec);
  const Eigen::VectorXf results_row_gpu =
      test_utils::interpolatePointsOnGPU(depth_frame_row_indices, u_px_vec);

  EXPECT_NEAR((results_col_cpu - results_col_gpu).maxCoeff(), 0.0f,
              kImagePlaneEpsilon);
  EXPECT_NEAR((results_row_cpu - results_row_gpu).maxCoeff(), 0.0f,
              kImagePlaneEpsilon);
}

TEST_F(CudaTsdfIntegratorTest, SetBlocksOnGPU) {
  // Make sure this is deterministic.
  std::srand(0);

  // Allocating blocks for setting
  constexpr int kNumBlocks = 10;
  for (int i = 0; i < kNumBlocks; i++) {
    layer_.allocateBlockAtIndex(test_utils::getRandomIndex3dInRange(-10, 10));
  }

  // The voxel to their linear indices
  test_utils::setVoxelBlockOnGPU(&layer_);

  // Check
  for (const Index3D& block_idx : layer_.getAllBlockIndices()) {
    const auto block_ptr = layer_.getBlockAtIndex(block_idx);
    float lin_idx = 0.0f;
    for (int x = 0; x < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; x++) {
      for (int y = 0; y < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; y++) {
        for (int z = 0; z < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; z++) {
          EXPECT_NEAR(block_ptr->voxels[x][y][z].distance, lin_idx,
                      kFloatEpsilon);
          ++lin_idx;
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
