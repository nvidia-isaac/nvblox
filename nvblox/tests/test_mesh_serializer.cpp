/*
Copyright 2023 NVIDIA CORPORATION

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
#include <algorithm>
#include <random>
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/serialization/mesh_serializer_gpu.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

class MeshSerializerGpuTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    std::srand(0);
    constexpr float kVoxelSize = 0.1;

    // Create a SDF layer scene
    TsdfLayer::Ptr sdf_layer;
    sdf_layer.reset(new TsdfLayer(kVoxelSize, MemoryType::kUnified));
    primitives::Scene scene;
    scene.aabb() = AxisAlignedBoundingBox(Vector3f(-3.0f, -3.0f, 0.0f),
                                          Vector3f(3.0f, 3.0f, 3.0f));
    scene.addPrimitive(std::make_unique<primitives::Plane>(
        Vector3f(0.0, 0.0, 0.0), Vector3f(-1, 0, 0)));
    scene.generateLayerFromScene(4 * kVoxelSize, sdf_layer.get());

    // Integrate mesh
    ColorMeshIntegrator mesh_integrator;
    mesh_layer_.reset(
        new ColorMeshLayer(sdf_layer->block_size(), MemoryType::kUnified));
    mesh_integrator.weld_vertices(false);
    EXPECT_TRUE(mesh_integrator.integrateMeshFromDistanceField(
        *sdf_layer, mesh_layer_.get(), DeviceType::kCPU));

    //  Need some blocks to work with
    EXPECT_GE(mesh_layer_->size(), 10);

    // Generate some colors
    std::vector<Index3D> all_indices = mesh_layer_->getAllBlockIndices();
    for (auto index : all_indices) {
      ColorMeshBlock* mesh_block = mesh_layer_->getBlockAtIndex(index).get();

      mesh_block->vertex_appearances.resizeAsync(mesh_block->vertices.size(),
                                                 CudaStreamOwning());
      for (size_t i = 0; i < mesh_block->vertex_appearances.size(); ++i) {
        mesh_block->vertex_appearances[i] = Color(i % 256, i % 256, i % 256);
      }
    }
  }

  /// Compare serialized flat buffers + offsets against the source mesh blocks.
  template <typename VertVec, typename ColorVec, typename TriVec>
  void validateSerializedMeshContents(
      const std::vector<nvblox::Index3D>& serialized_block_indices,
      const VertVec& vertices, const ColorVec& vertex_appearances,
      const TriVec& triangle_indices,
      const host_vector<int32_t>& vertex_block_offsets,
      const host_vector<int32_t>& triangle_index_block_offsets) {
    ASSERT_EQ(vertex_block_offsets.size(), serialized_block_indices.size() + 1);
    ASSERT_EQ(triangle_index_block_offsets.size(),
              serialized_block_indices.size() + 1);

    EXPECT_EQ(vertex_block_offsets[0], 0);
    EXPECT_EQ(triangle_index_block_offsets[0], 0);

    int serialized_vertex_index = 0;
    int serialized_triangle_index = 0;
    for (size_t i = 0; i < serialized_block_indices.size(); ++i) {
      EXPECT_EQ(vertex_block_offsets[i], serialized_vertex_index);
      EXPECT_EQ(triangle_index_block_offsets[i], serialized_triangle_index);

      const nvblox::ColorMeshBlock* mesh_block =
          mesh_layer_->getBlockAtIndex(serialized_block_indices[i]).get();

      ASSERT_NE(mesh_block, nullptr);

      ASSERT_GE(vertices.size(), static_cast<size_t>(serialized_vertex_index) +
                                     mesh_block->vertices.size());
      ASSERT_GE(vertex_appearances.size(),
                static_cast<size_t>(serialized_vertex_index) +
                    mesh_block->vertex_appearances.size());
      ASSERT_GE(triangle_indices.size(),
                static_cast<size_t>(serialized_triangle_index) +
                    mesh_block->triangles.size());

      const size_t verts_in_block =
          static_cast<size_t>(vertex_block_offsets[i + 1]) -
          static_cast<size_t>(vertex_block_offsets[i]);
      const size_t tris_in_block =
          static_cast<size_t>(triangle_index_block_offsets[i + 1]) -
          static_cast<size_t>(triangle_index_block_offsets[i]);
      ASSERT_EQ(mesh_block->vertices.size(), verts_in_block);
      ASSERT_EQ(mesh_block->triangles.size(), tris_in_block);

      for (size_t j = 0; j < mesh_block->vertices.size(); ++j) {
        for (int k = 0; k < 3; ++k) {
          EXPECT_EQ(vertices[serialized_vertex_index][k],
                    mesh_block->vertices[j][k]);
        }
        EXPECT_EQ(vertex_appearances[serialized_vertex_index].r(),
                  mesh_block->vertex_appearances[j].r());
        EXPECT_EQ(vertex_appearances[serialized_vertex_index].g(),
                  mesh_block->vertex_appearances[j].g());
        EXPECT_EQ(vertex_appearances[serialized_vertex_index].b(),
                  mesh_block->vertex_appearances[j].b());
        ++serialized_vertex_index;
      }

      for (size_t j = 0; j < mesh_block->triangles.size(); ++j) {
        EXPECT_EQ(triangle_indices[serialized_triangle_index],
                  mesh_block->triangles[j]);
        ++serialized_triangle_index;
      }
    }
  }

  void validateSerializedMesh(
      const std::vector<nvblox::Index3D>& serialized_block_indices) {
    const std::shared_ptr<SerializedColorMeshLayer> result =
        serializer_.getSerializedLayer();
    validateSerializedMeshContents(
        serialized_block_indices, result->vertices, result->vertex_appearances,
        result->triangle_indices, result->vertex_block_offsets,
        result->triangle_index_block_offsets);
  }

  // Data generators

  // Test subjects
  ColorMeshLayer::Ptr mesh_layer_;
  ColorMeshSerializerGpu serializer_;
};

TEST_F(MeshSerializerGpuTestFixture, serializeAllBlocks) {
  const std::vector<Index3D> block_indices_to_serialize =
      mesh_layer_->getAllBlockIndices();
  EXPECT_FALSE(block_indices_to_serialize.empty());

  serializer_.serialize(*(mesh_layer_.get()), block_indices_to_serialize,
                        CudaStreamOwning());

  validateSerializedMesh(block_indices_to_serialize);
}

TEST_F(MeshSerializerGpuTestFixture, serializeToDeviceAllBlocks) {
  const std::vector<Index3D> block_indices_to_serialize =
      mesh_layer_->getAllBlockIndices();
  EXPECT_FALSE(block_indices_to_serialize.empty());

  CudaStreamOwning stream;
  const std::shared_ptr<ColorMeshSerializerGpu::SerializedLayerTypeDevice>
      device_result = serializer_.serializeToDevice(
          *(mesh_layer_.get()), block_indices_to_serialize, stream, true);

  EXPECT_EQ(device_result->block_indices, block_indices_to_serialize);

  std::vector<Vector3f> vertices_host =
      device_result->vertices.toVectorAsync(stream);
  std::vector<Color> appearances_host =
      device_result->vertex_appearances.toVectorAsync(stream);
  std::vector<int> triangle_indices_host =
      device_result->triangle_indices.toVectorAsync(stream);
  stream.synchronize();

  validateSerializedMeshContents(block_indices_to_serialize, vertices_host,
                                 appearances_host, triangle_indices_host,
                                 device_result->vertex_block_offsets,
                                 device_result->triangle_index_block_offsets);
}

TEST_F(MeshSerializerGpuTestFixture, serializeSomeblocks) {
  // Shuffle the list of indices
  std::vector<Index3D> all_indices = mesh_layer_->getAllBlockIndices();
  std::shuffle(all_indices.begin(), all_indices.end(),
               std::default_random_engine());

  // Truncate the list
  const size_t num_blocks_to_serialize = all_indices.size() / 2;
  EXPECT_NE(num_blocks_to_serialize, 0);

  const std::vector<Index3D> block_indices_to_serialize(
      all_indices.begin(),
      std::next(all_indices.begin(), num_blocks_to_serialize));

  serializer_.serialize(*(mesh_layer_.get()), block_indices_to_serialize,
                        CudaStreamOwning());

  validateSerializedMesh(block_indices_to_serialize);
}

TEST_F(MeshSerializerGpuTestFixture, serializeFirstBlock) {
  const std::vector<Index3D> block_indices_to_serialize = {
      mesh_layer_->getAllBlockIndices().front()};

  serializer_.serialize(*(mesh_layer_.get()), block_indices_to_serialize,
                        CudaStreamOwning());

  validateSerializedMesh(block_indices_to_serialize);
}

TEST_F(MeshSerializerGpuTestFixture, serializeLastBlock) {
  const std::vector<Index3D> block_indices_to_serialize = {
      mesh_layer_->getAllBlockIndices().back()};

  serializer_.serialize(*(mesh_layer_.get()), block_indices_to_serialize,
                        CudaStreamOwning());

  validateSerializedMesh(block_indices_to_serialize);
}

TEST_F(MeshSerializerGpuTestFixture, serializeNoBlocks) {
  const std::vector<Index3D> block_indices_to_serialize;

  const std::shared_ptr<SerializedColorMeshLayer> result =
      serializer_.serialize(*(mesh_layer_.get()), block_indices_to_serialize,
                            CudaStreamOwning());

  ASSERT_TRUE(result->vertices.empty());
  ASSERT_TRUE(result->vertex_appearances.empty());
  ASSERT_TRUE(result->triangle_indices.empty());
}

TEST(MeshSerializerGpuTest, serializeOneEmptyBlock) {
  ColorMeshLayer mesh_layer(1.f, MemoryType::kDevice);

  Index3D index(0.F, 0.F, 0.F);
  mesh_layer.allocateBlockAtIndex(index);

  ColorMeshSerializerGpu serializer;
  serializer.serialize(mesh_layer, {index}, CudaStreamOwning());
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
