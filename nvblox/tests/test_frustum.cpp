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
#include "nvblox/tests/sensor_fixture.h"
#include "nvblox/utils/timing.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

template <typename SensorType>
class FrustumTest : public test_utils::SensorFixture<SensorType> {
 protected:
  FrustumTest()
      : test_utils::SensorFixture<SensorType>(
            Lidar(20, 20, 1E-2, 90.F * M_PI / 180.F),
            Camera(300, 300, 320, 240, 640, 480)) {}
  void SetUp() override {
    timing::Timing::Reset();
    std::srand(0);
    block_size_ = voxelSizeToBlockSize(voxel_size_);

    // Make the scene 6x6x3 meters big.
    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-3.0f, -3.0f, 0.0f),
                                           Vector3f(3.0f, 3.0f, 3.0f));

    base_path_ = test_utils::getTestDataPath("data/3dmatch/");

    // NOTE(alexmillane): In the test we have situations where we expect
    // different results from exactly the same viewpoint so we turn off caching
    // here.
    view_calculator_.cache_last_viewpoint(false);

    // Subsamling factor is different depending on sensor type.
    view_calculator_.raycast_subsampling_factor(subsampling_factor());
  }

  constexpr int subsampling_factor();

  static constexpr float kFloatEpsilon = 1e-4;

  float block_size_;
  float voxel_size_ = 0.05;

  primitives::Scene scene_;
  TypeIndexedStore sensor_store_;

  // Base path for 3D Match dataset.
  std::string base_path_;

  ViewCalculator view_calculator_;
};

// For Lidar we disable subsampling since the sensor is too sparse to handle it
// well.
template <>
int FrustumTest<Lidar>::subsampling_factor() {
  return 1;
}

// Camera uses default subsampling factor.
template <>
int FrustumTest<Camera>::subsampling_factor() {
  return kRaycastSubsamplingFactorDesc.default_value;
}

using SensorTypes = ::testing::Types<Camera, Lidar>;
TYPED_TEST_SUITE(FrustumTest, SensorTypes);

TYPED_TEST(FrustumTest, FarPlaneImageTest) {
  // We create a scene that is a flat plane 10 meters from the origin.
  constexpr float kPlaneDistance = 10.0f;

  this->scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(kPlaneDistance, 0.0, 0.0), Vector3f(-1, 0, 0)));

  // Create a pose at the origin looking forward.
  Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation_base);

  // Generate a depth frame with max distance == plane distance.
  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);
  constexpr float kInvalidDepth = -1.F;
  this->scene_.generateDepthImageFromScene(
      this->sensor(), T_S_C, 2 * kPlaneDistance, &depth_frame, kInvalidDepth);

  // We need to go integrate beyond the plane distance since lidar store ranges,
  // not z-depths.
  float max_distance = 10 * kPlaneDistance;
  std::vector<Index3D> blocks_in_cuda_view =
      this->view_calculator_.getBlocksInImageViewRaycast(
          MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
          this->sensor(), this->block_size_, 0.0f, max_distance);

  // Sort all of the entries.
  std::sort(blocks_in_cuda_view.begin(), blocks_in_cuda_view.end(),
            VectorCompare<Index3D>());

  int num_valid_pixels = 0;
  // We will now raycast through every single pixel in the original image.
  for (int u = 0; u < this->sensor().rows(); u++) {
    for (int v = 0; v < this->sensor().cols(); v++) {
      // Get the depth at this image point.
      float depth = depth_frame(u, v);
      Vector3f p_C =
          depth * this->sensor().vectorFromPixelIndices(Index2D(v, u));
      Vector3f p_L = T_S_C * p_C;

      // the 360 lidar scan doesn't see the whole plane, hence some depth values
      // will be invalid.
      if (depth != kInvalidDepth) {
        ++num_valid_pixels;
        Index3D block_index =
            getBlockIndexFromPositionInLayer(this->block_size_, p_L);

        EXPECT_TRUE(std::binary_search(blocks_in_cuda_view.begin(),
                                       blocks_in_cuda_view.end(), block_index,
                                       VectorCompare<Index3D>()))
            << block_index;

        // Now raycast back to the center.
        // Ok raycast to the correct point in the block.
        RayCaster raycaster(T_S_C.translation() / this->block_size_,
                            p_L / this->block_size_);
        Index3D ray_index = Index3D::Zero();
        while (raycaster.nextRayIndex(&ray_index)) {
          EXPECT_TRUE(std::binary_search(blocks_in_cuda_view.begin(),
                                         blocks_in_cuda_view.end(), ray_index,
                                         VectorCompare<Index3D>()))
              << ray_index;
        }
      }
    }
  }

  EXPECT_GE(num_valid_pixels, std::max(10, depth_frame.numel() / 10));

  std::cout << timing::Timing::Print();
}

TYPED_TEST(FrustumTest, PlaneWithGround) {
  // We create a scene that is a flat plane 10 meters from the origin.
  constexpr float kPlaneDistance = 10.0f;

  // We need to go integrate beyond the plane distance since lidar store ranges,
  // not z-depths.
  float max_distance = 10.F * kPlaneDistance;
  this->scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(kPlaneDistance, 0.0, 0.0), Vector3f(-1, 0, 0)));
  this->scene_.addGroundLevel(-1.0f);

  // Create a pose at the origin looking forward.
  Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation_base);

  // Generate a depth frame with max distance == plane distance.
  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);
  this->scene_.generateDepthImageFromScene(this->sensor(), T_S_C,
                                           2 * kPlaneDistance, &depth_frame);

  // Figure out what the GT should be.
  timing::Timer blocks_in_view_timer("blocks_in_view");
  ViewCalculator view_calculator;
  std::vector<Index3D> blocks_in_view =
      view_calculator.getBlocksInImageViewProjection(
          T_S_C, this->sensor(), this->block_size_, max_distance);
  blocks_in_view_timer.Stop();

  // Now get the actual thing to test.
  timing::Timer blocks_in_cuda_view_timer("blocks_in_cuda_view");
  std::vector<Index3D> blocks_in_cuda_view =
      this->view_calculator_.getBlocksInImageViewRaycast(
          MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_S_C,
          this->sensor(), this->block_size_, 0.0f,
          max_distance + this->kFloatEpsilon);
  EXPECT_LT(blocks_in_cuda_view.size(), blocks_in_view.size());
  blocks_in_cuda_view_timer.Stop();

  // Sort all of the entries.
  std::sort(blocks_in_view.begin(), blocks_in_view.end(),
            VectorCompare<Index3D>());
  std::sort(blocks_in_cuda_view.begin(), blocks_in_cuda_view.end(),
            VectorCompare<Index3D>());

  // Ok now the hard part. We expect the raycast to EVERY PIXEL to succeed
  // in only going through allocated blocks.
  // We make this easy by just making a TSDF layer.
  TsdfLayer tsdf_layer(this->voxel_size_, nvblox::MemoryType::kUnified);
  TsdfLayer tsdf_layer_cuda(this->voxel_size_, nvblox::MemoryType::kUnified);

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
  for (int u = 0; u < this->sensor().rows(); u++) {
    for (int v = 0; v < this->sensor().cols(); v++) {
      // Get the depth at this image point.
      float depth = depth_frame(u, v);
      Vector3f p_C =
          depth * this->sensor().vectorFromPixelIndices(Index2D(v, u));
      Vector3f p_L = T_S_C * p_C;

      Index3D block_index =
          getBlockIndexFromPositionInLayer(this->block_size_, p_L);

      EXPECT_TRUE(tsdf_layer_cuda.isBlockAllocated(block_index)) << block_index;

      // Now raycast back to the center.
      // Ok raycast to the correct point in the block.
      RayCaster raycaster(T_S_C.translation() / this->block_size_,
                          p_L / this->block_size_);
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
    for (const Index3D& block_index : blocks_in_view) {
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

// 3DMatch is camera only
using FrustumTestCamera = FrustumTest<Camera>;
TEST_F(FrustumTestCamera, ThreeDMatch) {
  // Get the first frame, a camera, and a pose.
  constexpr int kSequenceNum = 1;
  constexpr int kFrameNumber = 0;
  float max_distance = 10.0f;

  constexpr bool kUseMultithreaded = false;
  std::unique_ptr<datasets::ImageLoader<DepthImage>> depth_image_loader =
      datasets::threedmatch::internal::createDepthImageLoader(
          base_path_, kSequenceNum, kUseMultithreaded);

  // Get the first image.
  DepthImage depth_frame(MemoryType::kDevice);
  ASSERT_TRUE(depth_image_loader->getNextImage(&depth_frame));

  // Get the transform.
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
        this->view_calculator_.getBlocksInImageViewRaycast(
            MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere),
            T_L_C, camera, this->block_size_, 0.0f, max_distance);
    blocks_in_cuda_view_timer.Stop();

    // Figure out what the GT should be.
    timing::Timer blocks_in_view_timer("blocks_in_view");
    ViewCalculator view_calculator;
    std::vector<Index3D> blocks_in_view =
        view_calculator.getBlocksInImageViewProjection(
            T_L_C, camera, this->block_size_, max_distance);
    blocks_in_view_timer.Stop();
  }

  std::cout << timing::Timing::Print();
}

class FrustumRayTracingSubsamplingTest
    : public FrustumTest<Camera>,
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
          MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_L_C,
          camera, kBlockSize, 0.0, kDistanceToBlockCenters + 1.0f);

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

// 3DMatch is camera only
TEST_F(FrustumTestCamera, ViewpointCache) {
  // Load some 3DMatch data
  constexpr int kSeqID = 1;
  constexpr bool kMultithreadedLoading = false;
  auto data_loader = datasets::threedmatch::DataLoader::create(
      base_path_, kSeqID, kMultithreadedLoading);
  EXPECT_TRUE(data_loader) << "Cant find the test input data.";

  DepthImage depth_frame(MemoryType::kDevice);
  ColorImage color_frame(MemoryType::kDevice);
  Transform T_L_C;
  Camera camera;
  data_loader->loadNext(&depth_frame, &T_L_C, &camera, &color_frame);

  const float max_integration_distance_behind_surface_m =
      4.0f * this->voxel_size_;
  const float max_integration_distance_m = 10.0f;

  // Two view calculators not sharing caches.
  ViewCalculator view_calculator_1;
  ViewCalculator view_calculator_2;

  std::vector<Index3D> blocks_in_view_1 =
      view_calculator_1.getBlocksInImageViewRaycast(
          MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_L_C,
          camera, this->block_size_, max_integration_distance_behind_surface_m,
          max_integration_distance_m);

  DepthImage zero_depth_image(MemoryType::kDevice);
  zero_depth_image.copyFrom(depth_frame);
  zero_depth_image.setZeroAsync(CudaStreamOwning());

  std::vector<Index3D> blocks_in_view_2 =
      view_calculator_2.getBlocksInImageViewRaycast(
          MaskedDepthImageConstView(zero_depth_image, kMaskActiveEverywhere),
          T_L_C, camera, this->block_size_,
          max_integration_distance_behind_surface_m,
          max_integration_distance_m);

  EXPECT_GT(blocks_in_view_1.size(), 0);
  EXPECT_NE(blocks_in_view_1.size(), blocks_in_view_2.size());

  // Now start to share caches.
  auto viewpoint_cache = std::make_shared<ViewpointCache>();
  view_calculator_1.set_viewpoint_cache(
      viewpoint_cache, ViewCalculator::CalculationType::kRaycasting);
  view_calculator_2.set_viewpoint_cache(
      viewpoint_cache, ViewCalculator::CalculationType::kRaycasting);

  // Repeat the test and see that we now get the same results for the two
  // calculations
  blocks_in_view_1 = view_calculator_1.getBlocksInImageViewRaycast(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_L_C,
      camera, this->block_size_, max_integration_distance_behind_surface_m,
      max_integration_distance_m);

  blocks_in_view_2 = view_calculator_2.getBlocksInImageViewRaycast(
      MaskedDepthImageConstView(zero_depth_image, kMaskActiveEverywhere), T_L_C,
      camera, this->block_size_, max_integration_distance_behind_surface_m,
      max_integration_distance_m);

  EXPECT_GT(blocks_in_view_1.size(), 0);
  EXPECT_EQ(blocks_in_view_1.size(), blocks_in_view_2.size());
}

// We have specialized (camera-optimized) version of
// getBlocksInImageViewProjection(). We expect its output to be identical to the
// general version.
TEST_F(FrustumTestCamera, getBlocksInImageViewProjection_specialization) {
  Transform T_S_C = Transform::Identity();

  const float max_distance = 10.F;

  // Run with the camera-specialized function.
  std::vector<Index3D> blocks_in_cuda_view_specialization =
      this->view_calculator_.getBlocksInImageViewProjection(
          T_S_C, this->sensor(), this->block_size_, max_distance);

  // Create a derived camera class that is identical to the other camera. Using
  // this one will trigger the generic function since the type is different.
  class GenericCamera : public Camera {};
  GenericCamera generic_camera;
  static_cast<Camera&>(generic_camera) = this->sensor();

  std::vector<Index3D> blocks_in_cuda_view_generic =
      this->view_calculator_.getBlocksInImageViewProjection(
          T_S_C, generic_camera, this->block_size_, max_distance);

  constexpr int kExpectedNumBlocks = 23042;
  ASSERT_EQ(blocks_in_cuda_view_specialization.size(), kExpectedNumBlocks);
  ASSERT_EQ(blocks_in_cuda_view_generic.size(), kExpectedNumBlocks);

  // Sort all of the entries and compare
  std::sort(blocks_in_cuda_view_specialization.begin(),
            blocks_in_cuda_view_specialization.end(), VectorCompare<Index3D>());
  std::sort(blocks_in_cuda_view_generic.begin(),
            blocks_in_cuda_view_generic.end(), VectorCompare<Index3D>());
  for (size_t i = 0; i < blocks_in_cuda_view_generic.size(); ++i) {
    EXPECT_EQ(blocks_in_cuda_view_generic[i],
              blocks_in_cuda_view_specialization[i]);
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
