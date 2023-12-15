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
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/serialization/mesh_serializer.hpp"
#include "nvblox/tests/utils.h"

using namespace nvblox;

class MeshSerializerTest : public ::testing::Test {
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
    MeshIntegrator mesh_integrator;
    mesh_layer_.reset(
        new MeshLayer(sdf_layer->block_size(), MemoryType::kUnified));
    mesh_integrator.weld_vertices(false);
    EXPECT_TRUE(mesh_integrator.integrateMeshFromDistanceField(
        *sdf_layer, mesh_layer_.get(), DeviceType::kCPU));

    //  Need some blocks to work with
    EXPECT_GE(mesh_layer_->size(), 10);

    // Generate some colors
    std::vector<Index3D> all_indices = mesh_layer_->getAllBlockIndices();
    for (auto index : all_indices) {
      MeshBlock* mesh_block = mesh_layer_->getBlockAtIndex(index).get();

      mesh_block->colors.resize(mesh_block->vertices.size());
      for (size_t i = 0; i < mesh_block->colors.size(); ++i) {
        mesh_block->colors[i] = Color(i % 256, i % 256, i % 256, i % 256);
      }
    }
  }

  void validateSerializedMesh(
      const std::vector<nvblox::Index3D>& serialized_block_indices) {
    const std::shared_ptr<const SerializedMesh> result =
        serializer_.getSerializedMesh();
    ASSERT_EQ(result->vertex_block_offsets.size(),
              serialized_block_indices.size() + 1);
    ASSERT_EQ(result->triangle_index_block_offsets.size(),
              serialized_block_indices.size() + 1);

    EXPECT_EQ(result->vertex_block_offsets[0], 0);
    EXPECT_EQ(result->triangle_index_block_offsets[0], 0);

    int serialized_vertex_index = 0;
    int serialized_triangle_index = 0;
    for (size_t i = 0; i < serialized_block_indices.size(); ++i) {
      EXPECT_EQ(result->vertex_block_offsets[i], serialized_vertex_index);
      EXPECT_EQ(result->triangle_index_block_offsets[i],
                serialized_triangle_index);

      const nvblox::MeshBlock* mesh_block =
          mesh_layer_->getBlockAtIndex(serialized_block_indices[i]).get();

      ASSERT_NE(mesh_block, nullptr);

      ASSERT_GE(result->vertices.size(),
                serialized_vertex_index + mesh_block->vertices.size());
      ASSERT_GE(result->colors.size(),
                serialized_vertex_index + mesh_block->colors.size());
      ASSERT_GE(result->triangle_indices.size(),
                serialized_triangle_index + mesh_block->triangles.size());

      ASSERT_EQ(mesh_block->vertices.size(), result->getNumVerticesInBlock(i));
      ASSERT_EQ(mesh_block->triangles.size(),
                result->getNumTriangleIndicesInBlock(i));

      for (size_t j = 0; j < mesh_block->vertices.size(); ++j) {
        for (int k = 0; k < 3; ++k) {
          EXPECT_EQ(result->vertices[serialized_vertex_index][k],
                    mesh_block->vertices[j][k]);
        }
        EXPECT_EQ(result->colors[serialized_vertex_index].r,
                  mesh_block->colors[j].r);
        EXPECT_EQ(result->colors[serialized_vertex_index].g,
                  mesh_block->colors[j].g);
        EXPECT_EQ(result->colors[serialized_vertex_index].b,
                  mesh_block->colors[j].b);
        EXPECT_EQ(result->colors[serialized_vertex_index].a,
                  mesh_block->colors[j].a);
        ++serialized_vertex_index;
      }

      for (size_t j = 0; j < mesh_block->triangles.size(); ++j) {
        EXPECT_EQ(result->triangle_indices[serialized_triangle_index],
                  mesh_block->triangles[j]);
        ++serialized_triangle_index;
      }
    }
  }

  // Data generators

  // Test subjects
  MeshLayer::Ptr mesh_layer_;
  MeshSerializer serializer_;
};

TEST_F(MeshSerializerTest, serializeAllBlocks) {
  const std::vector<Index3D> block_indices_to_serialize =
      mesh_layer_->getAllBlockIndices();
  EXPECT_FALSE(block_indices_to_serialize.empty());

  serializer_.serializeMesh(*(mesh_layer_.get()), block_indices_to_serialize,
                            CudaStreamOwning());

  validateSerializedMesh(block_indices_to_serialize);
}

TEST_F(MeshSerializerTest, serializeSomeblocks) {
  // Shuffle the list of indices
  std::vector<Index3D> all_indices = mesh_layer_->getAllBlockIndices();
  std::random_shuffle(all_indices.begin(), all_indices.end());

  // Truncate the list
  const size_t num_blocks_to_serialize = all_indices.size() / 2;
  EXPECT_NE(num_blocks_to_serialize, 0);

  const std::vector<Index3D> block_indices_to_serialize(
      all_indices.begin(),
      std::next(all_indices.begin(), num_blocks_to_serialize));

  serializer_.serializeMesh(*(mesh_layer_.get()), block_indices_to_serialize,
                            CudaStreamOwning());

  validateSerializedMesh(block_indices_to_serialize);
}

TEST_F(MeshSerializerTest, serializeFirstBlock) {
  const std::vector<Index3D> block_indices_to_serialize = {
      mesh_layer_->getAllBlockIndices().front()};

  serializer_.serializeMesh(*(mesh_layer_.get()), block_indices_to_serialize,
                            CudaStreamOwning());

  validateSerializedMesh(block_indices_to_serialize);
}

TEST_F(MeshSerializerTest, serializeLastBlock) {
  const std::vector<Index3D> block_indices_to_serialize = {
      mesh_layer_->getAllBlockIndices().back()};

  serializer_.serializeMesh(*(mesh_layer_.get()), block_indices_to_serialize,
                            CudaStreamOwning());

  validateSerializedMesh(block_indices_to_serialize);
}

TEST_F(MeshSerializerTest, serializeNoBlocks) {
  const std::vector<Index3D> block_indices_to_serialize;

  const std::shared_ptr<const SerializedMesh> result =
      serializer_.serializeMesh(*(mesh_layer_.get()),
                                block_indices_to_serialize, CudaStreamOwning());

  ASSERT_TRUE(result->vertices.empty());
  ASSERT_TRUE(result->colors.empty());
  ASSERT_TRUE(result->triangle_indices.empty());
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
