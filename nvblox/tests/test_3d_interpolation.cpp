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

#include "nvblox/interpolation/interpolation_3d.h"
#include "nvblox/map/layer.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-6;

constexpr int kXAxisIndex = 0;
constexpr int kYAxisIndex = 1;
constexpr int kZAxisIndex = 2;

template <typename VoxelType>
using SetVoxelFunctionType = std::function<void(VoxelType& voxel, float value)>;

template <typename VoxelType>
void fillVoxelsWithIndices(VoxelBlock<VoxelType>* block_ptr,
                           const int xyz_index,
                           SetVoxelFunctionType<VoxelType> set_voxel,
                           const int offset = 0) {
  CHECK_GE(xyz_index, 0);
  CHECK_LT(xyz_index, 3);
  Index3D index;
  for (index.x() = 0; index.x() < TsdfBlock::kVoxelsPerSide; index.x()++) {
    for (index.y() = 0; index.y() < TsdfBlock::kVoxelsPerSide; index.y()++) {
      for (index.z() = 0; index.z() < TsdfBlock::kVoxelsPerSide; index.z()++) {
        set_voxel(block_ptr->voxels[index.x()][index.y()][index.z()],
                  static_cast<float>(index[xyz_index] + offset));
      }
    }
  }
}

void fillVoxelsWithIndices(TsdfBlock* block_ptr, const int xyz_index,
                           const int offset = 0) {
  auto setVoxelLambda = [](TsdfVoxel& voxel, float value) {
    voxel.distance = value;
    voxel.weight = 1.0f;
  };
  fillVoxelsWithIndices<TsdfVoxel>(block_ptr, xyz_index, setVoxelLambda,
                                   offset);
}

void fillVoxelsWithIndices(EsdfBlock* block_ptr, const int xyz_index,
                           const int offset = 0) {
  auto setVoxelLambda = [](EsdfVoxel& voxel, float value) {
    voxel.squared_distance_vox = std::pow(value, 2);
    voxel.observed = true;
  };
  fillVoxelsWithIndices<EsdfVoxel>(block_ptr, xyz_index, setVoxelLambda,
                                   offset);
}

TEST(InterpolatorTest, NeighboursTest) {
  // Empty layer
  constexpr float kVoxelSize = 1.0f;
  TsdfLayer layer(kVoxelSize, MemoryType::kUnified);

  // Allocate a block with its origin at the origin
  TsdfBlock::Ptr block_ptr = layer.allocateBlockAtIndex(Index3D(0, 0, 0));

  // Fill with such that the distances are equal to the x-index;
  fillVoxelsWithIndices(block_ptr.get(), kXAxisIndex);

  // Dummy function which always says a voxel is valid
  auto valid_lambda = [](const TsdfVoxel&) -> bool { return true; };

  // Check the surrounding voxels
  std::array<TsdfVoxel, 8> voxels;
  Vector3f p_L = 0.0 * Vector3f::Ones();
  EXPECT_FALSE(interpolation::internal::getSurroundingVoxels3D<TsdfVoxel>(
      p_L, layer, valid_lambda, &voxels));
  p_L = 0.1 * Vector3f::Ones();
  EXPECT_FALSE(interpolation::internal::getSurroundingVoxels3D<TsdfVoxel>(
      p_L, layer, valid_lambda, &voxels));
  p_L = 0.6 * Vector3f::Ones();
  EXPECT_TRUE(interpolation::internal::getSurroundingVoxels3D<TsdfVoxel>(
      p_L, layer, valid_lambda, &voxels));
  // Check that the surrounding voxels have x values {0,0,0,0,1,1,1,1}
  for (int i = 0; i < 8; i++) {
    if (i < 4) {
      EXPECT_EQ(voxels[i].distance, 0);
    } else {
      EXPECT_EQ(voxels[i].distance, 1);
    }
  }
  // Ask for neighbours outside the block bounds on the x-dimension
  p_L = Vector3f(static_cast<float>(TsdfBlock::kVoxelsPerSide), 0.6f, 0.6f);
  EXPECT_FALSE(interpolation::internal::getSurroundingVoxels3D<TsdfVoxel>(
      p_L, layer, valid_lambda, &voxels));
  // Add a new block
  block_ptr = layer.allocateBlockAtIndex(Index3D(1, 0, 0));
  fillVoxelsWithIndices(block_ptr.get(), kXAxisIndex);
  // Now the test just outside the first block on the x-dimension should pass.
  EXPECT_TRUE(interpolation::internal::getSurroundingVoxels3D<TsdfVoxel>(
      p_L, layer, valid_lambda, &voxels));
  // Check that the surrounding voxels have x values {7,7,7,7,0,0,0,0}
  for (int i = 0; i < 8; i++) {
    if (i < 4) {
      EXPECT_EQ(voxels[i].distance, 7);
    } else {
      EXPECT_EQ(voxels[i].distance, 0);
    }
  }
}

TEST(InterpolatorTest, OffsetTest) {
  // Empty layer
  constexpr float kVoxelSize = 1.0f;
  TsdfLayer layer(kVoxelSize, MemoryType::kUnified);

  // Allocate a block with its origin at the origin
  TsdfBlock::Ptr block_ptr = layer.allocateBlockAtIndex(Index3D(0, 0, 0));

  // Dummy function which always says a voxel is valid
  auto valid_lambda = [](const TsdfVoxel&) -> bool { return true; };

  // Get the surrounding voxels (in this test we just look at the offset vector)
  std::array<TsdfVoxel, 8> voxels;
  Vector3f p_offset_L;
  Vector3f p_L = 0.5 * Vector3f::Ones();
  EXPECT_TRUE(interpolation::internal::getSurroundingVoxels3D<TsdfVoxel>(
      p_L, layer, valid_lambda, &voxels, &p_offset_L));

  // TODO(alexmillane): Let's make some eigen checks....
  EXPECT_NEAR(p_offset_L.x(), 0.0f, kFloatEpsilon);
  EXPECT_NEAR(p_offset_L.y(), 0.0f, kFloatEpsilon);
  EXPECT_NEAR(p_offset_L.z(), 0.0f, kFloatEpsilon);

  p_L = 1.0 * Vector3f::Ones();
  EXPECT_TRUE(interpolation::internal::getSurroundingVoxels3D<TsdfVoxel>(
      p_L, layer, valid_lambda, &voxels, &p_offset_L));
  EXPECT_NEAR(p_offset_L.x(), 0.5f, kFloatEpsilon);
  EXPECT_NEAR(p_offset_L.y(), 0.5f, kFloatEpsilon);
  EXPECT_NEAR(p_offset_L.z(), 0.5f, kFloatEpsilon);
}

TEST(InterpolatorTest, InterpolationTest) {
  // Make sure this is deterministic.
  std::srand(0);

  // Create 3 layers where distances equal to x,y,z indices respectively. The
  // idea is that interpolating in these layers at point p_L gives you p_L back
  // for all points within the block (minus half a voxel width
  // because indexing starts at the low-corner but interpolation is wrt to voxel
  // center :)).
  constexpr float kVoxelSize = 1.0f;
  constexpr float KTestBlockSize = TsdfBlock::kVoxelsPerSide * kVoxelSize;
  // TSDF
  TsdfLayer layer_x_tsdf(kVoxelSize, MemoryType::kUnified);
  TsdfLayer layer_y_tsdf(kVoxelSize, MemoryType::kUnified);
  TsdfLayer layer_z_tsdf(kVoxelSize, MemoryType::kUnified);
  TsdfBlock::Ptr tsdf_block_x_ptr =
      layer_x_tsdf.allocateBlockAtIndex(Index3D(0, 0, 0));
  TsdfBlock::Ptr tsdf_block_y_ptr =
      layer_y_tsdf.allocateBlockAtIndex(Index3D(0, 0, 0));
  TsdfBlock::Ptr tsdf_block_z_ptr =
      layer_z_tsdf.allocateBlockAtIndex(Index3D(0, 0, 0));
  fillVoxelsWithIndices(tsdf_block_x_ptr.get(), kXAxisIndex);
  fillVoxelsWithIndices(tsdf_block_y_ptr.get(), kYAxisIndex);
  fillVoxelsWithIndices(tsdf_block_z_ptr.get(), kZAxisIndex);
  // ESDF
  EsdfLayer layer_x_esdf(kVoxelSize, MemoryType::kUnified);
  EsdfLayer layer_y_esdf(kVoxelSize, MemoryType::kUnified);
  EsdfLayer layer_z_esdf(kVoxelSize, MemoryType::kUnified);
  EsdfBlock::Ptr esdf_block_x_ptr =
      layer_x_esdf.allocateBlockAtIndex(Index3D(0, 0, 0));
  EsdfBlock::Ptr esdf_block_y_ptr =
      layer_y_esdf.allocateBlockAtIndex(Index3D(0, 0, 0));
  EsdfBlock::Ptr esdf_block_z_ptr =
      layer_z_esdf.allocateBlockAtIndex(Index3D(0, 0, 0));
  fillVoxelsWithIndices(esdf_block_x_ptr.get(), kXAxisIndex);
  fillVoxelsWithIndices(esdf_block_y_ptr.get(), kYAxisIndex);
  fillVoxelsWithIndices(esdf_block_z_ptr.get(), kZAxisIndex);

  // Testing random points inside the block
  constexpr int kNumTests = 1000;
  for (int i = 0; i < kNumTests; i++) {
    // Random point
    const float half_voxel_size = kVoxelSize / 2.0f;
    const Vector3f p_L = test_utils::getRandomVector3fInRange(
        half_voxel_size, KTestBlockSize - half_voxel_size);

    // Interpolate
    float interpolated_distance_x;
    float interpolated_distance_y;
    float interpolated_distance_z;
    // TSDF
    EXPECT_TRUE(interpolation::interpolateOnCPU(p_L, layer_x_tsdf,
                                                &interpolated_distance_x));
    EXPECT_TRUE(interpolation::interpolateOnCPU(p_L, layer_y_tsdf,
                                                &interpolated_distance_y));
    EXPECT_TRUE(interpolation::interpolateOnCPU(p_L, layer_z_tsdf,
                                                &interpolated_distance_z));
    // Check
    EXPECT_NEAR(interpolated_distance_x, p_L.x() - 0.5f, kFloatEpsilon);
    EXPECT_NEAR(interpolated_distance_y, p_L.y() - 0.5f, kFloatEpsilon);
    EXPECT_NEAR(interpolated_distance_z, p_L.z() - 0.5f, kFloatEpsilon);
    // ESDF
    EXPECT_TRUE(interpolation::interpolateOnCPU(p_L, layer_x_esdf,
                                                &interpolated_distance_x));
    EXPECT_TRUE(interpolation::interpolateOnCPU(p_L, layer_y_esdf,
                                                &interpolated_distance_y));
    EXPECT_TRUE(interpolation::interpolateOnCPU(p_L, layer_z_esdf,
                                                &interpolated_distance_z));
    // Check
    EXPECT_NEAR(interpolated_distance_x, p_L.x() - 0.5f, kFloatEpsilon);
    EXPECT_NEAR(interpolated_distance_y, p_L.y() - 0.5f, kFloatEpsilon);
    EXPECT_NEAR(interpolated_distance_z, p_L.z() - 0.5f, kFloatEpsilon);
  }
}

TEST(InterpolatorTest, PrimitivesInterpolationTest) {
  // Maximum distance to consider for scene generation.
  constexpr float kMaxDist = 10.0;

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
  constexpr float kVoxelSize_m = 0.2;
  TsdfLayer gt_layer(kVoxelSize_m, MemoryType::kUnified);
  constexpr float kTruncationDistanceMeters = 10;
  scene.generateLayerFromScene(kTruncationDistanceMeters, &gt_layer);

  // Generate some random points in the arena
  constexpr int kNumPointsToTest = 1000;
  std::vector<Vector3f> p_L_vec(kNumPointsToTest);
  std::generate(p_L_vec.begin(), p_L_vec.end(), []() {
    return Vector3f(test_utils::randomFloatInRange(-5.0f + kVoxelSize_m,
                                                   5.0f - kVoxelSize_m),
                    test_utils::randomFloatInRange(-5.0f + kVoxelSize_m,
                                                   5.0f - kVoxelSize_m),
                    test_utils::randomFloatInRange(0.0f + kVoxelSize_m,
                                                   5.0f - kVoxelSize_m));
  });

  // Interpolate the distance field at these locations
  std::vector<float> interpolated_distances;
  std::vector<bool> success_flags;
  interpolation::interpolateOnCPU(p_L_vec, gt_layer, &interpolated_distances,
                                  &success_flags);

  // Get the true distances
  std::vector<float> gt_distances;
  for (const Vector3f& p_L : p_L_vec) {
    gt_distances.push_back(scene.getSignedDistanceToPoint(p_L, kMaxDist));
  }

  // Check
  for (size_t i = 0; i < p_L_vec.size(); i++) {
    EXPECT_NEAR(interpolated_distances[i], gt_distances[i], 0.5 * kVoxelSize_m);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
