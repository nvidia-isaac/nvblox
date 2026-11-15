/*
Copyright 2024 NVIDIA CORPORATION

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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/indexing.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/internal/cuda/layer_to_3d_grid.cuh"
#include "nvblox/map/layer.h"
#include "nvblox/map/unified_3d_grid.h"

using namespace nvblox;

class LayerToUnifiedGridTest : public ::testing::Test {
 protected:
  LayerToUnifiedGridTest()
      : esdf_layer_(voxel_size_m_, MemoryType::kUnified),
        cuda_stream_(std::make_shared<CudaStreamOwning>()) {
    default_voxel_.squared_distance_vox = 1.0;
  }
  // Layer to test with
  const float voxel_size_m_ = 0.05;
  EsdfLayer esdf_layer_;

  // Default value of voxels in the dense
  EsdfVoxel default_voxel_;

  std::shared_ptr<CudaStreamOwning> cuda_stream_;
};

Index3D toGlobalVoxelIndex(const Index3D& block_idx, const Index3D& voxel_idx) {
  return block_idx * VoxelBlock<bool>::kVoxelsPerSide + voxel_idx;
}

float globalVoxelIndexToTestValue(const Index3D& global_voxel_idx) {
  Index3DHash hash;
  constexpr size_t kMaxValue = 1000;
  return static_cast<float>(hash(global_voxel_idx) % kMaxValue);
}

void setLayerToTestValues(EsdfLayer* layer_ptr) {
  callFunctionOnAllVoxels<EsdfVoxel>(
      layer_ptr, [](const Index3D& block_index, const Index3D& voxel_index,
                    EsdfVoxel* voxel) {
        voxel->squared_distance_vox = globalVoxelIndexToTestValue(
            toGlobalVoxelIndex(block_index, voxel_index));
      });
}

template <typename CellType, typename VoxelType>
void checkGridHasTestValue(
    const Unified3DGrid<CellType>& grid,
    const VoxelBlockLayer<VoxelType>& layer, const float default_value,
    std::function<float(const CellType& cell)> extract_test_value_op) {
  // Lambda to check if a global voxel index is in the layer.
  auto is_in_layer = [](const Index3D& global_voxel_idx,
                        const EsdfLayer& esdf_layer) -> bool {
    const Vector3f p_L =
        global_voxel_idx.cast<float>() * esdf_layer.voxel_size();
    const Index3D block_idx =
        getBlockIndexFromPositionInLayer(esdf_layer.block_size(), p_L);
    return esdf_layer.isBlockAllocated(block_idx);
  };
  // Iterate and check
  // - Values in the layer have been copied over.
  // - Blocks not in the layer are set to their default value.
  int num_checked = 0;
  int num_default = 0;
  const Index3D max_index = grid.min_index() + grid.aabb_size();
  for (int x = grid.min_index().x(); x < max_index.x(); x++) {
    for (int y = grid.min_index().y(); y < max_index.y(); y++) {
      for (int z = grid.min_index().z(); z < max_index.z(); z++) {
        const Index3D global_voxel_idx(x, y, z);
        EXPECT_TRUE(grid.isInsideGrid(global_voxel_idx));
        const float value = extract_test_value_op(grid(global_voxel_idx));
        constexpr float kEps = 1e-6;
        if (is_in_layer(global_voxel_idx, layer)) {
          EXPECT_NEAR(value, globalVoxelIndexToTestValue(global_voxel_idx),
                      kEps);
          num_checked++;
        } else {
          EXPECT_NEAR(value, default_value, kEps);
          num_default++;
        }
      }
    }
  }
  EXPECT_GT(num_checked, 0);
  LOG(INFO) << "num_checked: " << num_checked;
  LOG(INFO) << "num_default: " << num_default;
}

void checkGridHasTestValue(const Unified3DGrid<EsdfVoxel>& grid,
                           const EsdfLayer& layer,
                           const EsdfVoxel& default_voxel) {
  std::function<float(const EsdfVoxel& cell)> extract_test_value_op =
      [](const EsdfVoxel& voxel) -> float {
    return voxel.squared_distance_vox;
  };
  checkGridHasTestValue(grid, layer, default_voxel.squared_distance_vox,
                        extract_test_value_op);
}

TEST_F(LayerToUnifiedGridTest, WholeLayer) {
  // Allocate blocks
  const std::vector<Index3D> blocks = {
      Index3D(-1, 0, 0),  // NOLINT
      Index3D(0, 0, 0),   // NOLINT
      Index3D(1, 0, 0),   // NOLINT
      Index3D(1, 0, 1)    // NOLINT
  };
  esdf_layer_.allocateBlocksAtIndices(blocks, *cuda_stream_);

  // Set the test values
  setLayerToTestValues(&esdf_layer_);

  // AABB
  const auto aabb = getAABBOfAllocatedBlocks(esdf_layer_);

  // Convert to the grid.
  Unified3DGrid<EsdfVoxel> grid(MemoryType::kUnified);
  voxelLayerToDenseVoxelGridInAABBAsync(esdf_layer_, aabb, default_voxel_,
                                        &grid, *cuda_stream_);
  cuda_stream_->synchronize();
  cudaPeekAtLastError();
  EXPECT_TRUE((aabb.sizes().array() > 0).all());
  EXPECT_TRUE((grid.aabb_size().array() > 0).all());

  // Check the output grid
  // Iterate over the output dense grid and check that everything has been
  // faithfully copied.
  checkGridHasTestValue(grid, esdf_layer_, default_voxel_);
}

TEST_F(LayerToUnifiedGridTest, PartialLayer) {
  // Allocate blocks
  const Index3D block_idx_1 = Index3D(0, 0, 0);
  const Index3D block_idx_2 = Index3D(1, 1, 1);
  const std::vector<Index3D> blocks = {
      block_idx_1,  // NOLINT
      block_idx_2,  // NOLINT
  };
  esdf_layer_.allocateBlocksAtIndices(blocks, *cuda_stream_);

  // Set the test values
  setLayerToTestValues(&esdf_layer_);

  // Construct an AABB which is in the middle of two blocks.
  const Index3D voxel_idx(3, 3, 3);  // voxle in the middle(ish)
  const Vector3f aabb_min = getCenterPositionFromBlockIndexAndVoxelIndex(
      esdf_layer_.block_size(), block_idx_1, voxel_idx);
  const Vector3f aabb_max = getCenterPositionFromBlockIndexAndVoxelIndex(
      esdf_layer_.block_size(), block_idx_2, voxel_idx);
  const auto aabb = AxisAlignedBoundingBox(aabb_min, aabb_max);

  // Convert to the grid.
  Unified3DGrid<EsdfVoxel> grid(MemoryType::kUnified);
  voxelLayerToDenseVoxelGridInAABBAsync(esdf_layer_, aabb, default_voxel_,
                                        &grid, *cuda_stream_);
  cuda_stream_->synchronize();
  cudaPeekAtLastError();
  EXPECT_TRUE((aabb.sizes().array() > 0).all());
  EXPECT_TRUE((grid.aabb_size().array() > 0).all());

  // Check the output grid
  // Iterate over the output dense grid and check that everything has been
  // faithfully copied.
  checkGridHasTestValue(grid, esdf_layer_, default_voxel_);
}

TEST_F(LayerToUnifiedGridTest, CopyLayerToHost) {
  // Allocate blocks
  const Index3D block_idx = Index3D(0, 0, 0);
  esdf_layer_.allocateBlockAtIndex(block_idx);

  // Set the test values
  setLayerToTestValues(&esdf_layer_);

  // AABB
  const auto aabb = getAABBOfAllocatedBlocks(esdf_layer_);

  // Convert to the grid.
  Unified3DGrid<EsdfVoxel> grid_device(MemoryType::kDevice);
  voxelLayerToDenseVoxelGridInAABBAsync(esdf_layer_, aabb, default_voxel_,
                                        &grid_device, *cuda_stream_);
  EXPECT_TRUE((aabb.sizes().array() > 0).all());
  EXPECT_TRUE((grid_device.aabb_size().array() > 0).all());

  // Copy Grid to CPU
  Unified3DGrid<EsdfVoxel> grid_host(MemoryType::kHost);
  grid_host.copyFromAsync(grid_device, *cuda_stream_);
  cuda_stream_->synchronize();
  cudaPeekAtLastError();

  // Check the output grid
  // Iterate over the output dense grid and check that everything has been
  // faithfully copied.
  checkGridHasTestValue(grid_host, esdf_layer_, default_voxel_);
}

struct ExtractDistanceFunctor {
  __device__ __inline__ float operator()(const EsdfVoxel& voxel) const {
    return voxel.squared_distance_vox;
  }
};

TEST_F(LayerToUnifiedGridTest, CopyLayerWithConversion) {
  // Allocate blocks
  const Index3D block_idx = Index3D(0, 0, 0);
  esdf_layer_.allocateBlockAtIndex(block_idx);

  // Set the test values
  setLayerToTestValues(&esdf_layer_);

  // AABB
  const auto aabb = getAABBOfAllocatedBlocks(esdf_layer_);

  // The conversion functor
  ExtractDistanceFunctor conversion_op;

  // The default output value
  const float default_value = -1.0f;

  // Convert to the grid.
  Unified3DGrid<float> grid(MemoryType::kUnified);
  voxelLayerToDenseVoxelGridInAABBAsync(esdf_layer_, aabb, default_value,
                                        conversion_op, &grid, *cuda_stream_);
  cuda_stream_->synchronize();
  cudaPeekAtLastError();
  EXPECT_TRUE((aabb.sizes().array() > 0).all());
  EXPECT_TRUE((grid.aabb_size().array() > 0).all());

  // Check the output grid
  // Iterate over the output dense grid and check that everything has been
  // faithfully copied.
  checkGridHasTestValue<float, EsdfVoxel>(
      grid, esdf_layer_, default_value,
      [](const float& value) -> float { return value; });
}

TEST_F(LayerToUnifiedGridTest, ReusedBufferWritesBoundaryCells) {
  EsdfLayer empty_layer(0.2f, MemoryType::kUnified);
  Unified3DGrid<float> grid(MemoryType::kUnified);
  constexpr float kSentinelValue = 123.0f;
  constexpr float kDefaultValue = -1.0f;

  // Reuse an existing allocation so unwritten cells retain a known value.
  grid.setAABBAsync(Index3D::Zero(), Index3D(2, 2, 20), *cuda_stream_);
  cuda_stream_->synchronize();
  for (float& value : grid.data()) {
    value = kSentinelValue;
  }

  // The upper z bound is slightly below 3.2 m in floating point, but the
  // output's robust rounding includes voxel 16, which belongs to block 2.
  const Vector3f minimum(0.0f, 0.0f, -0.6f);
  const Vector3f size(0.2f, 0.2f, 3.8f);
  const AxisAlignedBoundingBox aabb(minimum, minimum + size);
  voxelLayerToDenseVoxelGridInAABBAsync(empty_layer, aabb, kDefaultValue,
                                        ExtractDistanceFunctor{}, &grid,
                                        *cuda_stream_);
  checkCudaErrors(cudaPeekAtLastError());
  cuda_stream_->synchronize();

  ASSERT_TRUE(grid.isInsideGrid(Index3D(0, 0, 16)));
  // An empty layer must overwrite every output cell with the default.
  for (size_t i = 0; i < grid.data().size(); ++i) {
    EXPECT_FLOAT_EQ(grid.data()[i], kDefaultValue) << "cell " << i;
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
