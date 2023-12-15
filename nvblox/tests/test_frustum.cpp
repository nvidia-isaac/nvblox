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
#include <string>

#include "nvblox/core/indexing.h"
#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/core/types.h"
#include "nvblox/datasets/3dmatch.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/integrators/view_calculator.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/rays/ray_caster.h"
#include "nvblox/utils/timing.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

class FrustumTest : public ::testing::Test {
 protected:
  void SetUp() override {
    timing::Timing::Reset();
    std::srand(0);
    block_size_ = VoxelBlock<bool>::kVoxelsPerSide * voxel_size_;

    // Make the scene 6x6x3 meters big.
    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-3.0f, -3.0f, 0.0f),
                                           Vector3f(3.0f, 3.0f, 3.0f));

    // Arbitrary camera
    constexpr float fu = 300;
    constexpr float fv = 300;
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr float cu = static_cast<float>(width) / 2.0f;
    constexpr float cv = static_cast<float>(height) / 2.0f;
    camera_.reset(new Camera(fu, fv, cu, cv, width, height));

    base_path_ = "./data/3dmatch/";
  }

  static constexpr float kFloatEpsilon = 1e-4;

  float block_size_;
  float voxel_size_ = 0.05;

  // A simulation scene.
  primitives::Scene scene_;

  // A simulation camera.
  std::unique_ptr<Camera> camera_;

  // Base path for 3D Match dataset.
  std::string base_path_;

  ViewCalculator frustum_;
};

TEST_F(FrustumTest, FarPlaneImageTest) {
  // We create a scene that is a flat plane 10 meters from the origin.
  constexpr float kPlaneDistance = 10.0f;
  float max_distance = kPlaneDistance;
  scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(kPlaneDistance, 0.0, 0.0), Vector3f(-1, 0, 0)));

  // Create a pose at the origin looking forward.
  Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation_base);

  // Generate a depth frame with max distance == plane distance.
  DepthImage depth_frame(camera_->height(), camera_->width(),
                         MemoryType::kUnified);
  scene_.generateDepthImageFromScene(*camera_, T_S_C, 2 * kPlaneDistance,
                                     &depth_frame);

  // Figure out what the GT should be.
  std::vector<Index3D> blocks_in_view = ViewCalculator::getBlocksInViewPlanes(
      T_S_C, *camera_, block_size_, max_distance);
  std::vector<Index3D> blocks_in_image_view =
      ViewCalculator::getBlocksInImageViewPlanes(
          depth_frame, T_S_C, *camera_, block_size_, 0.0f, max_distance);
  EXPECT_EQ(blocks_in_view.size(), blocks_in_image_view.size());

  // Now get the actual thing to test.
  std::vector<Index3D> blocks_in_cuda_view =
      frustum_.getBlocksInImageViewRaycast(depth_frame, T_S_C, *camera_,
                                           block_size_, 0.0f,
                                           max_distance + kFloatEpsilon);

  // Sort all of the entries.
  std::sort(blocks_in_view.begin(), blocks_in_view.end(),
            VectorCompare<Index3D>());
  std::sort(blocks_in_image_view.begin(), blocks_in_image_view.end(),
            VectorCompare<Index3D>());
  std::sort(blocks_in_cuda_view.begin(), blocks_in_cuda_view.end(),
            VectorCompare<Index3D>());

  size_t min_size =
      std::min(blocks_in_cuda_view.size(), blocks_in_image_view.size());
  for (size_t i = 0; i < min_size; i++) {
    EXPECT_EQ(blocks_in_view[i], blocks_in_image_view[i]);
  }

  // We will now raycast through every single pixel in the original image.
  for (int u = 0; u < camera_->rows(); u++) {
    for (int v = 0; v < camera_->cols(); v++) {
      // Get the depth at this image point.
      float depth = depth_frame(u, v);
      Vector3f p_C = depth * camera_->vectorFromPixelIndices(Index2D(v, u));
      Vector3f p_L = T_S_C * p_C;

      Index3D block_index = getBlockIndexFromPositionInLayer(block_size_, p_L);

      EXPECT_TRUE(std::binary_search(blocks_in_cuda_view.begin(),
                                     blocks_in_cuda_view.end(), block_index,
                                     VectorCompare<Index3D>()))
          << block_index;
      // Now raycast back to the center.
      // Ok raycast to the correct point in the block.
      RayCaster raycaster(T_S_C.translation() / block_size_, p_L / block_size_);
      Index3D ray_index = Index3D::Zero();
      while (raycaster.nextRayIndex(&ray_index)) {
        EXPECT_TRUE(std::binary_search(blocks_in_cuda_view.begin(),
                                       blocks_in_cuda_view.end(), ray_index,
                                       VectorCompare<Index3D>()))
            << ray_index;
      }
    }
  }

  std::cout << timing::Timing::Print();
}

TEST_F(FrustumTest, PlaneWithGround) {
  // We create a scene that is a flat plane 10 meters from the origin.
  constexpr float kPlaneDistance = 10.0f;
  float max_distance = kPlaneDistance;
  scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(kPlaneDistance, 0.0, 0.0), Vector3f(-1, 0, 0)));
  scene_.addGroundLevel(-1.0f);

  // Create a pose at the origin looking forward.
  Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation_base);

  // Generate a depth frame with max distance == plane distance.
  DepthImage depth_frame(camera_->height(), camera_->width(),
                         MemoryType::kUnified);
  scene_.generateDepthImageFromScene(*camera_, T_S_C, 2 * kPlaneDistance,
                                     &depth_frame);

  // Figure out what the GT should be.
  timing::Timer blocks_in_view_timer("blocks_in_view");
  std::vector<Index3D> blocks_in_view = ViewCalculator::getBlocksInViewPlanes(
      T_S_C, *camera_, block_size_, max_distance);
  blocks_in_view_timer.Stop();
  timing::Timer blocks_in_image_view_timer("blocks_in_image_view");
  std::vector<Index3D> blocks_in_image_view =
      ViewCalculator::getBlocksInImageViewPlanes(
          depth_frame, T_S_C, *camera_, block_size_, 0.0f, max_distance);
  blocks_in_image_view_timer.Stop();

  EXPECT_EQ(blocks_in_view.size(), blocks_in_image_view.size());

  // Now get the actual thing to test.
  timing::Timer blocks_in_cuda_view_timer("blocks_in_cuda_view");
  std::vector<Index3D> blocks_in_cuda_view =
      frustum_.getBlocksInImageViewRaycast(depth_frame, T_S_C, *camera_,
                                           block_size_, 0.0f,
                                           max_distance + kFloatEpsilon);
  EXPECT_LT(blocks_in_cuda_view.size(), blocks_in_image_view.size());
  blocks_in_cuda_view_timer.Stop();

  // Sort all of the entries.
  std::sort(blocks_in_view.begin(), blocks_in_view.end(),
            VectorCompare<Index3D>());
  std::sort(blocks_in_image_view.begin(), blocks_in_image_view.end(),
            VectorCompare<Index3D>());
  std::sort(blocks_in_cuda_view.begin(), blocks_in_cuda_view.end(),
            VectorCompare<Index3D>());

  for (size_t i = 0; i < blocks_in_image_view.size(); i++) {
    EXPECT_EQ(blocks_in_view[i], blocks_in_image_view[i]);
  }

  // Ok now the hard part. We expect the raycast to EVERY PIXEL to succeed
  // in only going through allocated blocks.
  // We make this easy by just making a TSDF layer.
  TsdfLayer tsdf_layer(voxel_size_, nvblox::MemoryType::kUnified);
  TsdfLayer tsdf_layer_cuda(voxel_size_, nvblox::MemoryType::kUnified);

  for (const Index3D& block_index : blocks_in_cuda_view) {
    TsdfBlock::Ptr block = tsdf_layer_cuda.allocateBlockAtIndex(block_index);
    for (int x = 0; x < TsdfBlock::kVoxelsPerSide; x++) {
      for (int y = 0; y < TsdfBlock::kVoxelsPerSide; y++) {
        for (int z = 0; z < TsdfBlock::kVoxelsPerSide; z++) {
          block->voxels[x][y][z].weight = 1;
        }
      }
    }
  }

  // We will now raycast through every single pixel in the original image.
  for (int u = 0; u < camera_->rows(); u++) {
    for (int v = 0; v < camera_->cols(); v++) {
      // Get the depth at this image point.
      float depth = depth_frame(u, v);
      Vector3f p_C = depth * camera_->vectorFromPixelIndices(Index2D(v, u));
      Vector3f p_L = T_S_C * p_C;

      Index3D block_index = getBlockIndexFromPositionInLayer(block_size_, p_L);

      EXPECT_TRUE(tsdf_layer_cuda.isBlockAllocated(block_index)) << block_index;

      // Now raycast back to the center.
      // Ok raycast to the correct point in the block.
      RayCaster raycaster(T_S_C.translation() / block_size_, p_L / block_size_);
      Index3D ray_index = Index3D::Zero();
      while (raycaster.nextRayIndex(&ray_index)) {
        EXPECT_TRUE(tsdf_layer_cuda.isBlockAllocated(ray_index)) << ray_index;

        if (!tsdf_layer_cuda.isBlockAllocated(ray_index)) {
          TsdfBlock::Ptr block = tsdf_layer.allocateBlockAtIndex(block_index);
          for (int x = 0; x < TsdfBlock::kVoxelsPerSide; x++) {
            for (int y = 0; y < TsdfBlock::kVoxelsPerSide; y++) {
              for (int z = 0; z < TsdfBlock::kVoxelsPerSide; z++) {
                block->voxels[x][y][z].weight = 1;
              }
            }
          }
        }
      }
    }
  }

  if (FLAGS_nvblox_test_file_output) {
    for (const Index3D& block_index : blocks_in_image_view) {
      TsdfBlock::Ptr block = tsdf_layer.allocateBlockAtIndex(block_index);
      for (int x = 0; x < TsdfBlock::kVoxelsPerSide; x++) {
        for (int y = 0; y < TsdfBlock::kVoxelsPerSide; y++) {
          for (int z = 0; z < TsdfBlock::kVoxelsPerSide; z++) {
            block->voxels[x][y][z].weight = 1;
          }
        }
      }
    }

    io::writeToPng("test_frustum_image.png", depth_frame);

    io::outputVoxelLayerToPly(tsdf_layer, "test_frustum_blocks_image.ply");
    io::outputVoxelLayerToPly(tsdf_layer_cuda, "test_frustum_blocks_cuda.ply");
  }
  std::cout << timing::Timing::Print();
}

// TODO(helen): fix this test with frustum checking.
TEST_F(FrustumTest, BlocksInView) {
  // Arranging a situation where we have a predictable number of blocks in
  // view We choose a situation with with a single z-colomn of blocks in
  // view by making a camera with 4 pixels viewing the furthest blocks'
  // corners.

  constexpr float kBlockSize = 1.0f;

  // Let's touch a collumn of 10 blocks in the z direction.
  constexpr float kDistanceToBlockCenters = 9.5f;

  // Design a camera that just views the far blocks outer corners
  constexpr float cu = 1.0;
  constexpr float cv = 1.0;
  constexpr int width = 2;
  constexpr int height = 2;
  // Pixel locations touched by the blocks upper-right corner.
  // NOTE(alexmillane): we just slightly lengthen the focal length to not
  // have boundary effects.
  constexpr float u = static_cast<float>(width);
  constexpr float v = static_cast<float>(height);
  constexpr float fu =
      static_cast<float>(u - cu) * 2.0 * kDistanceToBlockCenters / kBlockSize +
      0.01;
  constexpr float fv =
      static_cast<float>(v - cv) * 2.0 * kDistanceToBlockCenters / kBlockSize +
      0.01;
  Camera camera(fu, fv, cu, cv, width, height);

  // Depth frame with 4 pixels, some pixels at the far block depth, some a
  // the first.
  DepthImage depth_frame(2, 2, MemoryType::kUnified);
  depth_frame(0, 0) = kBlockSize / 2.0f;
  depth_frame(1, 0) = kBlockSize / 2.0f;
  depth_frame(0, 1) = kDistanceToBlockCenters;
  depth_frame(1, 1) = kDistanceToBlockCenters;

  // Camera looking down z axis, sitting centered on a block center in x and
  // y
  Transform T_L_C;
  T_L_C = Eigen::Translation3f(0.5f, 0.5f, 0.0f);

  // Get the block indices
  constexpr float kMaxDist = 10.0f;
  constexpr float kTruncationDistance = 0.0f;
  const std::vector<Index3D> blocks_in_view =
      ViewCalculator::getBlocksInImageViewPlanes(depth_frame, T_L_C, camera,
                                                 kBlockSize,
                                                 kTruncationDistance, kMaxDist);

  EXPECT_EQ(blocks_in_view.size(), 10);
  for (size_t i = 0; i < blocks_in_view.size(); i++) {
    EXPECT_TRUE((blocks_in_view[i].array() == Index3D(0, 0, i).array()).all());
  }
}

TEST_F(FrustumTest, ThreeDMatch) {
  // Get the first frame, a camera, and a pose.
  constexpr int kSequenceNum = 1;
  constexpr int kFrameNumber = 0;
  float max_distance = 10.0f;

  std::unique_ptr<datasets::ImageLoader<DepthImage>> depth_image_loader =
      datasets::threedmatch::internal::createDepthImageLoader(base_path_,
                                                              kSequenceNum);

  // Get the first image.
  DepthImage depth_frame(MemoryType::kDevice);
  ASSERT_TRUE(depth_image_loader->getNextImage(&depth_frame));

  // Get the tranform.
  Transform T_L_C;
  ASSERT_TRUE(datasets::threedmatch::internal::parsePoseFromFile(
      datasets::threedmatch::internal::getPathForFramePose(
          base_path_, kSequenceNum, kFrameNumber),
      &T_L_C));

  // Create a camera object.
  int image_width = depth_frame.cols();
  int image_height = depth_frame.rows();
  const std::string intrinsics_filename =
      datasets::threedmatch::internal::getPathForCameraIntrinsics(base_path_);
  Eigen::Matrix3f camera_intrinsics;
  ASSERT_TRUE(datasets::threedmatch::internal::parseCameraFromFile(
      intrinsics_filename, &camera_intrinsics));
  Camera camera = Camera::fromIntrinsicsMatrix(camera_intrinsics, image_width,
                                               image_height);

  for (int i = 0; i < 100; i++) {
    // Now get the actual thing to test.
    timing::Timer blocks_in_cuda_view_timer("blocks_in_cuda_view");
    std::vector<Index3D> blocks_in_cuda_view =
        frustum_.getBlocksInImageViewRaycast(depth_frame, T_L_C, camera,
                                             block_size_, 0.0f, max_distance);
    blocks_in_cuda_view_timer.Stop();

    // Figure out what the GT should be.
    timing::Timer blocks_in_view_timer("blocks_in_view");
    std::vector<Index3D> blocks_in_view = ViewCalculator::getBlocksInViewPlanes(
        T_L_C, camera, block_size_, max_distance);
    blocks_in_view_timer.Stop();
    timing::Timer blocks_in_image_view_timer("blocks_in_image_view");
    std::vector<Index3D> blocks_in_image_view =
        ViewCalculator::getBlocksInImageViewPlanes(
            depth_frame, T_L_C, camera, block_size_, 0.0f, max_distance);
    blocks_in_image_view_timer.Stop();
  }

  std::cout << timing::Timing::Print();
}

class FrustumRayTracingSubsamplingTest
    : public FrustumTest,
      public ::testing::WithParamInterface<int> {
 protected:
  // Yo dawg I heard you like params
};

TEST_P(FrustumRayTracingSubsamplingTest, RayTracePixels) {
  // Arranging a situation where we have a predictable number of blocks in
  // view
  // |--|--|
  // |--|--|
  // |--|--|
  // |--|--|
  //   \ /
  //    *   --camera
  constexpr float kBlockSize = 1.0f;

  // Let's touch a 2x2x3 collumn of blocks in the z direction.
  constexpr float kDistanceToBlockCenters = 2.5f;

  // Design a camera that just views the far blocks outer corners
  constexpr int width = 3;
  constexpr int height = 3;
  constexpr float cu = static_cast<float>(width - 1) / 2.0f;
  constexpr float cv = static_cast<float>(height - 1) / 2.0f;
  // Calculate focal lengths such that the extreme pixels shoot rays though back
  // blocks' centers
  constexpr float u = static_cast<float>(width - 1);
  constexpr float v = static_cast<float>(height - 1);
  constexpr float fu = static_cast<float>(u - cu) * kDistanceToBlockCenters /
                       (0.5f * kBlockSize);
  constexpr float fv = static_cast<float>(v - cv) * kDistanceToBlockCenters /
                       (0.5f * kBlockSize);
  Camera camera(fu, fv, cu, cv, width, height);

  // Depth frame with 4 pixels, some pixels at the far block depth, some a
  // the first.
  DepthImage depth_frame(height, width, MemoryType::kUnified);
  for (int lin_idx = 0; lin_idx < depth_frame.numel(); lin_idx++) {
    depth_frame(lin_idx) = kDistanceToBlockCenters;
  }

  // Camera looking down z axis, sitting between blocks the in x and
  // y dimensions
  Transform T_L_C;
  T_L_C = Eigen::Translation3f(1.0f, 1.0f, 0.0f);

  ViewCalculator view_calculator;

  unsigned int raycast_subsampling_factor = GetParam();
  view_calculator.raycast_subsampling_factor(raycast_subsampling_factor);

  const std::vector<Index3D> blocks_in_view =
      view_calculator.getBlocksInImageViewRaycast(
          depth_frame, T_L_C, camera, kBlockSize, 0.0,
          kDistanceToBlockCenters + 1.0f);

  std::for_each(blocks_in_view.begin(), blocks_in_view.end(),
                [](const auto& block_idx) {
                  EXPECT_TRUE(block_idx.x() == 0 || block_idx.x() == 1);
                  EXPECT_TRUE(block_idx.y() == 0 || block_idx.y() == 1);
                  EXPECT_TRUE(block_idx.z() == 0 || block_idx.z() == 1 ||
                              block_idx.z() == 2);
                });

  // 2 x 2 x 3 block volume
  EXPECT_EQ(blocks_in_view.size(), 12);
}

INSTANTIATE_TEST_CASE_P(FrustumTest, FrustumRayTracingSubsamplingTest,
                        ::testing::Values(1, 2));

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
